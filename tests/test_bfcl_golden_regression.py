"""L1+L2 for the accepted controllers through BFCL's REAL dispatch logic.

This is the file that would have caught our repeated yo-yo. Every check here runs the real installer
(`scripts/install_controller.py`), the real predicates (`bfcl_signals.SIGNALS`), the real phase rule and
the real dispatch guard conditions transcribed from the patches that run on the cluster.

It does NOT claim to prove an end-to-end store mutation -- that needs the isolated runtime, and L3
reports an explicit SKIP rather than a bland pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

from anchoropt.learning.golden_registry import GoldenRegistry            # noqa: E402
from anchoropt.learning.regression_protocol import (                     # noqa: E402
    DID_NOT_ACT, DID_NOT_FIRE, FIRED_ON_NEGATIVE, L1_CONTRACT, L2_SMOKE, L3_PAIRED,
    OPERATION_INFEASIBLE, PROBE_UNAVAILABLE, RegressionSuite)
from bfcl_regression_host import BfclRegressionHost                      # noqa: E402

REGISTRY = REPO / "rounds" / "GOLDEN" / "registry.json"


@pytest.fixture(scope="module")
def suite():
    return RegressionSuite(GoldenRegistry.load(REGISTRY), BfclRegressionHost())


def test_the_declared_signal_seam_resolves(suite):
    """The defect L2 found on its first run: `anchoropt.memory_gates` had no evaluate_signal here.

    `install_controller` resolves a declared signal through that module. It ships from
    patches/bv on the cluster; in this repo `anchoropt.memory_gates` resolved to the evaluator's
    unrelated memory_gates.py, the import raised, and the real `fires_on` turned that into False --
    a controller that can never fire while reporting a clean non-firing.
    """
    from anchoropt.memory_gates import evaluate_signal
    assert evaluate_signal("no_tool_call_at_all",
                           {"has_generation": True, "proposes_tool_call": False,
                            "tool_calls_so_far": 0}, {}) is True


def test_an_unknown_signal_raises_rather_than_reading_as_not_fired(suite):
    from anchoropt.memory_gates import evaluate_signal
    with pytest.raises(Exception, match="not a declared signal"):
        evaluate_signal("no_such_signal_anywhere", {}, {})


def test_every_accepted_controller_passes_L1_contract(suite):
    """A SUPERSET assertion: the registry grows, and every entry in it must pass."""
    res = suite.run_contract()
    assert res.ok, res.certificates
    assert {"a4_zero_call_reprompt_vector", "rec_sum_capacity_recovery"} <= set(res.passed)
    assert set(res.passed) == {e.name for e in suite.registry.all()}


def test_every_accepted_controller_passes_L2_through_the_real_path(suite):
    """Positive states fire AND act; negative states do not fire. Superset, so new entries are covered."""
    res = suite.run_smoke()
    assert res.ok, [str(c) for c in res.certificates]
    assert {"a4_zero_call_reprompt_vector", "rec_sum_capacity_recovery"} <= set(res.passed)
    assert set(res.passed) == {e.name for e in suite.registry.all()}


def test_L3_is_skipped_explicitly_not_passed_silently(suite):
    res = suite.run_paired()
    assert res.skipped and all(c.reason == PROBE_UNAVAILABLE for c in res.skipped)
    assert not res.passed, "a paired measurement needs the GPU runtime; it must not report a pass"


# -- the guard conditions are the point -------------------------------------------------------

def test_a_step_that_made_a_tool_call_never_reaches_the_A4_branch(suite):
    """The host requires _query_tool_calls == 0. A test calling fires_on directly skips this."""
    e = suite.registry.get("a4_zero_call_reprompt_vector")
    r = suite.host.smoke_execute(e, {"has_generation": True, "proposes_tool_call": False,
                                     "tool_calls_so_far": 2, "phase": "query",
                                     "step_count": 0, "max_steps_per_turn": 20})
    assert not r["fired"] and "_query_tool_calls == 0" in r["detail"]


def test_the_once_per_turn_latch_is_respected(suite):
    """A4's retry budget IS the host's existing latch, not a second copy."""
    e = suite.registry.get("a4_zero_call_reprompt_vector")
    r = suite.host.smoke_execute(e, {"has_generation": True, "proposes_tool_call": False,
                                     "tool_calls_so_far": 0, "phase": "query",
                                     "step_count": 0, "max_steps_per_turn": 20,
                                     "_zero_call_reprompted": True})
    assert not r["fired"] and "latch" in r["detail"]


def test_disable_gates_closes_every_boundary(suite):
    for name in ("a4_zero_call_reprompt_vector", "rec_sum_capacity_recovery"):
        e = suite.registry.get(name)
        st = dict(e.positive_states[0]); st["disable_gates"] = True
        assert not suite.host.smoke_execute(e, st)["fired"]


def test_phase_eligibility_is_symmetric(suite):
    """A prereq controller must not fire during a query, and vice versa."""
    e = suite.registry.get("rec_sum_capacity_recovery")      # phase=prereq
    st = dict(e.positive_states[0]); st["phase"] = "query"
    r = suite.host.smoke_execute(e, st)
    assert not r["fired"] and "not eligible" in r["detail"]


def test_the_kv_state_is_a_negative_for_the_rec_sum_controller(suite):
    """The refutation, pinned: an entries-count refusal must not trigger a char reduction."""
    e = suite.registry.get("rec_sum_capacity_recovery")
    r = suite.host.smoke_execute(e, {"error_kind": "no_capacity", "proposes_write": True,
                                     "phase": "prereq"})
    assert not r["fired"]


# -- the protocol must FAIL when something really regresses ------------------------------------

def test_a_wrong_capability_identity_is_refused_at_dispatch(suite):
    """Right signal, wrong executor: not this intervention."""
    import copy
    e = copy.deepcopy(suite.registry.get("rec_sum_capacity_recovery"))
    object.__setattr__(e, "spec", dict(e.spec, capability_id="someone_elses_executor"))
    r = suite.host.smoke_execute(e, dict(e.positive_states[0]))
    assert r["fired"] and not r["acted"] and "identity gate" in r["detail"]


def test_an_infeasible_operator_fails_L1(suite):
    """If someone repoints this controller at the slot-count constraint, L1 must stop it."""
    import copy
    e = copy.deepcopy(suite.registry.get("rec_sum_capacity_recovery"))
    object.__setattr__(e, "identity",
                       type(e.identity)(**{**e.identity.__dict__, "signal": "container_at_capacity"}))
    certs = suite.host.contract_check(e)
    assert any(c.reason == OPERATION_INFEASIBLE for c in certs), [str(c) for c in certs]


def test_a_reprompt_with_no_instruction_cannot_act(suite):
    import copy
    e = copy.deepcopy(suite.registry.get("a4_zero_call_reprompt_vector"))
    object.__setattr__(e, "spec", dict(e.spec, eta={"instruction": "  ", "retry_budget": 1}))
    r = suite.host.smoke_execute(e, dict(e.positive_states[0]))
    assert r["fired"] and not r["acted"] and "nothing would be injected" in r["detail"]
