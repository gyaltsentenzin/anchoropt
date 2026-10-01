"""The outer-loop driver: sequencing and the audit trail, with every decision injected.

The property under test is that the driver owns NO rules. It must sequence, prune with existing
checks, and record why -- and in particular it must never collapse "nothing survives under this
vocabulary" into "done", which is the distinction `phase_switch.py` exists to preserve.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "tb2_deepagents"))

import tb2_adapter as tb2  # noqa: E402
from anchoropt.anchor import Action, IncisionPoint  # noqa: E402
from anchoropt.learning.self_evolve import (  # noqa: E402
    CONTINUE_POLICY, EXPAND_ATTRIBUTION, STOP, Candidate, run, validate_candidates,
)
from anchoropt.runtime import ResidualDiagnosis  # noqa: E402

DIAG = (ResidualDiagnosis(case_id="c1", mechanism="concluded without verifying",
                          evidence="assertion failure", consequential_decision="emit terminal",
                          proposed_behavior_change="verify first", provider="test"),)

GOOD = Candidate(boundary=IncisionPoint.POST_GENERATION_PRE_EXEC,
                 signal="terminal_response_proposed", action=Action.REPROMPT,
                 theta={"text": "verify"})


def test_validation_prunes_an_unexecutable_cell_with_an_attributable_reason():
    bad = Candidate(boundary=IncisionPoint.POST_EXECUTION, signal="terminal_response_proposed",
                    action=Action.REROUTE, theta={"tool_name": "x", "reason": "r"})
    ok, pruned = validate_candidates([GOOD, bad], host=tb2.HOST, adapter=tb2)
    assert ok == [GOOD]
    assert len(pruned) == 1 and pruned[0][1].startswith("unexecutable_in_host")


def test_validation_prunes_an_undeclared_signal():
    bad = Candidate(boundary=IncisionPoint.PRE_GENERATION, signal="not_a_signal",
                    action=Action.REPROMPT, theta={"text": "x"})
    ok, pruned = validate_candidates([bad], host=tb2.HOST, adapter=tb2)
    assert not ok and pruned[0][1].startswith("signal_not_declared")


def test_validation_prunes_missing_required_params():
    bad = Candidate(boundary=IncisionPoint.POST_GENERATION_PRE_EXEC, signal="explore_streak",
                    action=Action.SUPPRESS, theta={"reason": "r"})
    ok, pruned = validate_candidates([bad], host=tb2.HOST, adapter=tb2)
    assert not ok and pruned[0][1].startswith("bad_signal_params")


def test_continue_phase_records_survivors_and_prunes():
    trail = run(diagnose=lambda: DIAG, decide_phase=lambda d: CONTINUE_POLICY,
                propose=lambda d: [GOOD], expand=lambda d: [],
                host=tb2.HOST, adapter=tb2, max_iterations=1)
    assert len(trail) == 1
    assert trail[0].phase == CONTINUE_POLICY
    assert trail[0].validated == (GOOD,)


def test_stop_is_reported_only_when_the_phase_rule_says_stop():
    trail = run(diagnose=lambda: DIAG, decide_phase=lambda d: STOP,
                propose=lambda d: [], expand=lambda d: [],
                host=tb2.HOST, adapter=tb2)
    assert [r.phase for r in trail] == [STOP]


def test_policy_exhaustion_is_not_reported_as_stop():
    """The distinction phase_switch exists to preserve: 'nothing survives in this representation'
    is not 'the residual is addressed'."""
    unexecutable = Candidate(boundary=IncisionPoint.POST_EXECUTION,
                             signal="terminal_response_proposed", action=Action.REPROMPT,
                             theta={"text": "x"})
    trail = run(diagnose=lambda: DIAG, decide_phase=lambda d: CONTINUE_POLICY,
                propose=lambda d: [unexecutable], expand=lambda d: [],
                host=tb2.HOST, adapter=tb2, max_iterations=3)
    assert len(trail) == 1
    assert trail[0].phase == CONTINUE_POLICY
    assert trail[0].validated == ()
    assert all(r.phase != STOP for r in trail)


def test_expansion_that_proposes_nothing_terminates_without_claiming_success():
    trail = run(diagnose=lambda: DIAG, decide_phase=lambda d: EXPAND_ATTRIBUTION,
                propose=lambda d: [], expand=lambda d: [],
                host=tb2.HOST, adapter=tb2, max_iterations=3)
    assert [r.phase for r in trail] == [EXPAND_ATTRIBUTION]
    assert "not addressable" in trail[0].note
    assert all(r.phase != STOP for r in trail)


def test_expansion_then_policy_resumes():
    phases = iter([EXPAND_ATTRIBUTION, CONTINUE_POLICY])
    trail = run(diagnose=lambda: DIAG, decide_phase=lambda d: next(phases),
                propose=lambda d: [GOOD], expand=lambda d: ["a_new_signal"],
                host=tb2.HOST, adapter=tb2, max_iterations=2)
    assert [r.phase for r in trail] == [EXPAND_ATTRIBUTION, CONTINUE_POLICY]
    assert trail[1].validated == (GOOD,)


def test_the_provider_is_swappable_without_touching_the_driver():
    """Generality check: Self-Harness must be replaceable by another miner."""
    other = (ResidualDiagnosis(case_id="x", mechanism="m", evidence="e",
                               consequential_decision="d", proposed_behavior_change="b",
                               provider="some_other_miner"),)
    trail = run(diagnose=lambda: other, decide_phase=lambda d: CONTINUE_POLICY,
                propose=lambda d: [GOOD], expand=lambda d: [],
                host=tb2.HOST, adapter=tb2, max_iterations=1)
    assert trail[0].n_diagnoses == 1


def test_the_driver_holds_no_thresholds_of_its_own():
    """Any rule it appears to apply must be imported, not duplicated."""
    src = (REPO / "anchoropt" / "learning" / "self_evolve.py").read_text()
    for smell in ("0.10", "0.50", "S1_", "S2_", "SIGN_TEST", "p_value", "0.05"):
        assert smell not in src, f"threshold {smell!r} duplicated into the driver"
