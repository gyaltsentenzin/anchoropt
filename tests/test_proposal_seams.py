"""The attributor/proposer separation, enforced by schema rather than requested in a prompt.

Each test pins a way the separation could quietly collapse:

  * an attributor that names the decision point has already made AnchorOpt's choice
  * a model-asserted support number is not a measurement
  * a POLICY-block proposal may not author a new signal
  * AnchorOpt, not the proposer, decides legality
  * a reroute must name a real tool with real argument names
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

import bfcl_capabilities as C                                          # noqa: E402
import bfcl_runtime as R                                               # noqa: E402
from anchoropt.anchor import Action, IncisionPoint                      # noqa: E402
from anchoropt.learning.proposal_seams import (                        # noqa: E402
    AttributionSchemaError, ProposalSchemaError, compute_support, ground_theta,
    ingest_attribution, ingest_proposal, rank_by_support, validate_proposal,
)

FIELDS = C.all_fields()

GOOD_ATTRIBUTION = {
    "case_id": "vector_42-finance-7",
    "failure_mechanism": "the model concluded the store held no answer after a core retrieve "
                         "returned nothing well-matched",
    "evidence": [{"step": 4, "call": "core_memory_retrieve(query='budget')",
                  "result": '{"ranked_results": []}'}],
    "consequential_decision": "whether to search the other container before answering",
    "causal_region": {"step": 4, "phase": "after the retrieve returned"},
    "proposed_behavior_change": "consult archival before concluding the fact is absent",
}


# ---- the attributor may not do AnchorOpt's job -------------------------------------------------

@pytest.mark.parametrize("key,value", [
    ("boundary", "post_execution"),
    ("incision_point", "post_execution"),
    ("signal", "retrieval_weak"),
    ("action", "reroute"),
    ("theta", {"destination": "archival_memory_retrieve"}),
])
def test_attributor_naming_anchoropt_decisions_is_rejected(key, value):
    rec = dict(GOOD_ATTRIBUTION, **{key: value})
    with pytest.raises(AttributionSchemaError, match="must not decide"):
        ingest_attribution(rec, provider="claude-attributor")


@pytest.mark.parametrize("key", ["support", "linked_downstream_loss", "n_cases", "confidence"])
def test_model_asserted_measurements_are_rejected(key):
    """A number produced by a language model is not a measurement. An invented-but-plausible
    support figure is indistinguishable from a measured one."""
    rec = dict(GOOD_ATTRIBUTION, **{key: 81})
    with pytest.raises(AttributionSchemaError, match="must not decide"):
        ingest_attribution(rec, provider="claude-attributor")


def test_incomplete_attribution_is_rejected():
    rec = dict(GOOD_ATTRIBUTION)
    del rec["consequential_decision"]
    with pytest.raises(AttributionSchemaError, match="missing"):
        ingest_attribution(rec, provider="claude-attributor")


def test_good_attribution_becomes_a_diagnosis_with_the_region_kept_as_prose():
    d = ingest_attribution(GOOD_ATTRIBUTION, provider="claude-attributor")
    assert d.case_id == "vector_42-finance-7"
    assert "ranked_results" in d.evidence
    region = d.metadata["causal_region"]
    assert region["step"] == 4
    assert region["phase"] == "after the retrieve returned"
    # crucially NOT an IncisionPoint
    assert not isinstance(region["phase"], IncisionPoint)


def test_support_is_computed_by_grouping_not_taken_from_a_record():
    ds = [ingest_attribution(dict(GOOD_ATTRIBUTION, case_id=f"c{i}"), provider="p")
          for i in range(3)]
    other = ingest_attribution(dict(GOOD_ATTRIBUTION, case_id="other",
                                    consequential_decision="whether to clear the container"),
                               provider="p")
    support = compute_support([*ds, other])
    assert sorted(support.values(), reverse=True) == [3, 1]
    assert rank_by_support([other, *ds])[0].case_id in {"c0", "c1", "c2"}


# ---- POLICY vs SIGNAL block ------------------------------------------------------------------

def _proposal(**over):
    base = {"boundary": "post_execution", "action": "reprompt",
            "signal_name": "identifier_not_found", "theta": {"text": "check archival"},
            "diagnosis_case_ids": ["vector_42-finance-7"]}
    base.update(over)
    return base


def test_policy_block_refuses_a_newly_authored_signal():
    """Phi is FROZEN in the policy block. Letting every proposal invent a signal dissolves the
    block-coordinate structure into one unconstrained call."""
    rec = _proposal(signal_expr={"field": "proposes_read", "op": "is_true"})
    with pytest.raises(ProposalSchemaError, match="POLICY block"):
        ingest_proposal(rec, allow_new_signal=False, declared_signals=R.declared_signals())


def test_policy_block_refuses_an_undeclared_signal_name():
    rec = _proposal(signal_name="something_nobody_declared")
    with pytest.raises(ProposalSchemaError, match="not in the declared vocabulary"):
        ingest_proposal(rec, allow_new_signal=False, declared_signals=R.declared_signals())


def test_signal_block_accepts_a_declarative_expression():
    rec = _proposal(signal_name="read_returned_nothing_useful",
                    signal_expr={"all": [{"field": "proposes_read", "op": "is_true"},
                                         {"field": "result", "op": "is_vacuous"}]})
    c = ingest_proposal(rec, allow_new_signal=True, declared_signals=R.declared_signals())
    assert c.signal_expr is not None
    compiled, why = validate_proposal(c, runtime=R, host=R.HOST, fields=FIELDS)
    assert why == "legal", why
    assert compiled.boundaries == frozenset({"post_execution"})


def test_action_outside_the_closed_vocabulary_is_rejected():
    with pytest.raises(ProposalSchemaError, match="closed semantic vocabulary"):
        ingest_proposal(_proposal(action="patch_the_prompt"), allow_new_signal=True,
                        declared_signals=R.declared_signals())


def test_unknown_decision_point_is_rejected():
    with pytest.raises(ProposalSchemaError, match="three legal points"):
        ingest_proposal(_proposal(boundary="during_execution"), allow_new_signal=True,
                        declared_signals=R.declared_signals())


# ---- AnchorOpt decides legality ---------------------------------------------------------------

def test_unexecutable_cell_is_declined_with_its_own_reason():
    c = ingest_proposal(_proposal(boundary="post_execution", action="suppress",
                                  theta={"reason": "x"}),
                        allow_new_signal=True, declared_signals=R.declared_signals())
    _compiled, why = validate_proposal(c, runtime=R, host=R.HOST, fields=FIELDS)
    assert why.startswith("action_not_executable"), why


def test_signal_not_observable_at_the_proposed_boundary_is_declined():
    rec = _proposal(boundary="post_generation_pre_exec", action="reprompt",
                    signal_name="weak_retrieval",
                    signal_expr={"field": "best_similarity", "op": "lt", "value": 0.3})
    c = ingest_proposal(rec, allow_new_signal=True, declared_signals=R.declared_signals())
    _compiled, why = validate_proposal(c, runtime=R, host=R.HOST, fields=FIELDS)
    assert why.startswith("signal_not_observable"), why


def test_a_proposal_explaining_no_episode_is_declined():
    """Without a linked diagnosis, engagement cannot be checked -- and a positive delta with zero
    engagement is proof of NON-attribution."""
    c = ingest_proposal(_proposal(diagnosis_case_ids=[]), allow_new_signal=False,
                        declared_signals=R.declared_signals())
    _compiled, why = validate_proposal(c, runtime=R, host=R.HOST, fields=FIELDS)
    assert why.startswith("no_diagnosis_linked"), why


# ---- grounded theta ---------------------------------------------------------------------------

def test_reroute_to_a_nonexistent_tool_is_declined():
    ok, why = ground_theta(Action.REROUTE, {"destination": "magic_memory_fix"}, runtime=R)
    assert not ok and "not a tool" in why


def test_reroute_with_wrong_argument_names_is_declined():
    """A2: emitting `key=` universally would have built an invalid call and been misscored as
    'the substitute did not help'."""
    ok, why = ground_theta(Action.REROUTE,
                           {"destination": "archival_memory_remove", "args_patch": {"key": "k"}},
                           runtime=R)
    assert not ok and "does not accept" in why


def test_reroute_to_a_real_tool_with_real_args_is_grounded():
    ok, _why = ground_theta(Action.REROUTE,
                            {"destination": "archival_memory_retrieve",
                             "args_patch": {"query": "budget"}}, runtime=R)
    assert ok
