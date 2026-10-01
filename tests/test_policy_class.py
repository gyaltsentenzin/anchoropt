"""POLICY as a first-class object: signal -> policy -> execution, with theta optimized not guessed.

Each test pins a property whose violation would either re-hardcode "policy = threshold" or let a
one-shot LLM number become the deployed value again (Self-Evolve R1's failure).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

import bfcl_runtime as R                                              # noqa: E402
from anchoropt.learning.policy_class import (                         # noqa: E402
    IMPLEMENTED_CLASSES, ParameterDomain, PolicyClass, PolicySpec, PolicySpecError, ThetaResult,
    optimize_theta, quantile_grid, summarize_sweep, train_objective,
)

DOM = ParameterDomain(name="below", kind=float, low=0.0, high=1.0,
                      observed_from="best_similarity", monotone="lower_fires_less")


def _spec(**over):
    base = dict(boundary="post_execution", signal="retrieval_similarity_below_threshold",
                action="reprompt", policy_class=PolicyClass.PARAMETERIZED, domains=(DOM,))
    base.update(over)
    return PolicySpec(**base)


def _res(theta, net, fire, n=89):
    gains = tuple(f"g{i}" for i in range(max(net, 0)))
    losses = tuple(f"l{i}" for i in range(max(-net, 0)))
    return ThetaResult(theta=theta, gains=gains, losses=losses, firings=fire, cases_fired=fire,
                       n=n, interventions_executed=fire, accuracy_delta_pp=100.0 * net / n)


# ---- policy is not hardcoded to threshold ------------------------------------------------------

def test_unimplemented_classes_are_declared_but_refused():
    """Requirement 7: the richer classes must be nameable so adding them is not a rewrite, and
    refused so nobody thinks they work."""
    assert PolicyClass.CLASSIFIER not in IMPLEMENTED_CLASSES
    assert PolicyClass.LLM_POLICY not in IMPLEMENTED_CLASSES
    with pytest.raises(PolicySpecError, match="NOT IMPLEMENTED"):
        _spec(policy_class=PolicyClass.CLASSIFIER, domains=())


def test_deterministic_policy_needs_no_parameters():
    spec = _spec(policy_class=PolicyClass.DETERMINISTIC, domains=())
    assert not spec.is_parameterized
    assert spec.candidate_thetas() == ({},)


def test_a_parameterized_policy_must_declare_a_domain():
    with pytest.raises(PolicySpecError, match="at least one parameter domain"):
        _spec(domains=())


def test_a_deterministic_policy_with_domains_is_a_contradiction():
    with pytest.raises(PolicySpecError, match="that is a PARAMETERIZED policy"):
        _spec(policy_class=PolicyClass.DETERMINISTIC)


# ---- the runtime declares domains generically --------------------------------------------------

def test_runtime_exposes_domains_per_parameter_not_per_special_case():
    """Requirement 1: generic. A categorical parameter proves it is not threshold-only."""
    sim = R.parameter_domains("retrieval_similarity_below_threshold")
    assert len(sim) == 1 and sim[0].name == "below" and sim[0].observed_from == "best_similarity"
    cat = R.parameter_domains("no_informative_result")
    assert cat and cat[0].values == ("empty_collection", "all_floor_scores")
    assert R.parameter_domains("container_at_capacity") == ()


def test_policy_class_is_derived_not_declared_twice():
    assert R.policy_class_for("retrieval_similarity_below_threshold") is PolicyClass.PARAMETERIZED
    assert R.policy_class_for("container_at_capacity") is PolicyClass.DETERMINISTIC


# ---- the grid comes from observed data --------------------------------------------------------

def test_grid_is_drawn_from_observed_values():
    """Requirement 5: a linear sweep spends its budget where no data lives, so most arms would be
    identical to control or fire on everything."""
    obs = [0.03, 0.14, 0.32, 0.39, 0.43, 0.49, 0.55, 0.62, 0.69, 0.88]
    grid = quantile_grid(obs, n=9, domain=DOM)
    assert grid
    assert all(min(obs) <= v <= max(obs) for v in grid)
    assert len(set(grid)) == len(grid), "duplicate candidates waste an arm each"


def test_grid_is_deterministic():
    obs = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert quantile_grid(obs, domain=DOM) == quantile_grid(obs, domain=DOM)


def test_no_observations_refuses_rather_than_inventing_a_sweep():
    with pytest.raises(PolicySpecError, match="Refusing to invent"):
        _spec().candidate_thetas({"best_similarity": []})


def test_historical_thresholds_are_not_privileged():
    """A5's 0.30 may appear ONLY because the observed distribution put it there."""
    obs = [0.80, 0.85, 0.90, 0.95]
    grid = quantile_grid(obs, domain=DOM)
    assert 0.30 not in grid


def test_multidimensional_theta_is_refused_not_silently_producted():
    d2 = ParameterDomain(name="other", kind=float, low=0.0, high=1.0, observed_from="x")
    with pytest.raises(PolicySpecError, match="ONE parameter"):
        _spec(domains=(DOM, d2)).candidate_thetas({"best_similarity": [0.1, 0.5]})


# ---- the LLM number is a hint, never the deployed value ---------------------------------------

def test_llm_hint_is_provenance_only_and_not_in_the_deployed_theta():
    """Requirement 2, and R1's whole lesson: below=0.75 was chosen one-shot and cost -3.37 pp."""
    spec = _spec(llm_hint={"below": 0.75})
    results = [_res({"below": 0.32}, 2, 9), _res({"below": 0.43}, 5, 21),
               _res({"below": 0.75}, -3, 46)]
    best, all_r = optimize_theta(spec, evaluate=lambda th: next(
        r for r in results if r.theta == th), observations={"best_similarity": [0.32, 0.43, 0.75]})
    assert best != dict(spec.llm_hint), "the hint must not survive as the deployed value"
    assert spec.llm_hint == {"below": 0.75}, "but it is preserved for provenance"
    assert "provenance only" in summarize_sweep(spec, all_r)


def test_hint_outside_the_domain_is_clamped_not_rejected():
    assert DOM.clamp(1.7) == 1.0
    assert DOM.clamp(-0.2) == 0.0


# ---- the training objective --------------------------------------------------------------------

def test_objective_maximizes_net():
    a, b = _res({"below": 0.3}, 5, 20), _res({"below": 0.6}, 1, 20)
    assert train_objective(a) > train_objective(b)


def test_an_inert_theta_never_ties_an_engaged_one():
    """R1's candidate 3 scored +0.00 pp with 0 firings. That is proof of NON-attribution, and it
    must not read as a tie with a genuinely neutral engaged arm."""
    inert = _res({"below": 0.01}, 0, 0)
    engaged = _res({"below": 0.30}, 0, 12)
    assert train_objective(engaged) > train_objective(inert)


def test_at_equal_net_the_more_selective_theta_wins():
    """R1's -3.37 pp came from firing on 52% of cases, 11 already passing. Less interference for
    the same result is strictly better."""
    broad = _res({"below": 0.70}, 3, 46)
    narrow = _res({"below": 0.40}, 3, 12)
    assert train_objective(narrow) > train_objective(broad)


def test_objective_is_total_and_stable_for_deterministic_policies():
    r = ThetaResult(theta={}, gains=(), losses=(), firings=0, cases_fired=0, n=10)
    assert train_objective(r) == train_objective(r)


def test_sweep_is_reported_whole_not_just_argmax():
    """The SHAPE of J over theta is the finding: a peak, a plateau, and no-engaged-theta are three
    different conclusions."""
    spec = _spec()
    results = [_res({"below": v}, net, fire) for v, net, fire in
               ((0.2, 0, 0), (0.4, 4, 15), (0.6, 1, 40))]
    best, all_r = optimize_theta(spec, evaluate=lambda th: next(
        r for r in results if r.theta == th), observations={"best_similarity": [0.2, 0.4, 0.6]})
    assert len(all_r) == 3
    assert best == {"below": 0.4}
