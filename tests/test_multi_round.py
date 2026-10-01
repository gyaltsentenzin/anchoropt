"""The multi-round driver. Benchmark-independent: the step function is a stub returning outcomes.

Two properties carry the weight:

  1. RE-MINING HAPPENS AFTER AN ACCEPTANCE AND BEFORE THE NEXT DIAGNOSIS, against the NEW incumbent.
     Skipping it is how a loop measures every anchor against the original baseline and reports a
     stack that was never assembled.
  2. THE RUN STOPS FOR A STATED REASON, and a loop that keeps selecting arms whose evidence never
     arrives stops rather than spinning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from anchoropt.learning.acceptance_criteria import evaluate_criteria
from anchoropt.learning.multi_round import (
    CONSECUTIVE_NULLS, MAX_ROUNDS, NO_CANDIDATE, REMINE_FAILED, STOP_REQUESTED,
    InterventionRequired, RoundOutcome, run_rounds)
from anchoropt.learning.round_ledger import (
    MEASURED_LOSER, OUTCOME_ACCEPTED, OUTCOME_BLOCKED, OUTCOME_NO_CANDIDATE, OUTCOME_REJECTED,
    OUTCOME_SELECTED_ONLY, RETRYABLE, SETTLED, RoundLedger)
from anchoropt.learning.self_evolve import EvaluationResult


# ------------------------------------------------------------------ stubs (no benchmark anywhere)

@dataclass
class Inc:
    incumbent_id: str


@dataclass
class Cand:
    label: str


@dataclass
class Step:
    """Shaped like self_evolve.StepResult in the attributes the driver reads."""

    outcome: str
    selected_candidate: Any = None
    evaluation: Any = None
    new_incumbent: Any = None


def _ev(net=1, n=10):
    control = {f"c{i}": False for i in range(n)}
    candidate = dict(control)
    gains = tuple(f"c{i}" for i in range(max(net, 0)))
    for g in gains:
        candidate[g] = True
    return EvaluationResult(control=control, candidate=candidate, gains=gains, losses=(),
                            telemetry={}, denominator_ok=True)


def _scripted(*outcomes):
    """A step_fn replaying a script. Records the ledger state it was handed, per round."""
    seen: list[dict] = []
    it = iter(outcomes)

    def step_fn(incumbent, ledger):
        seen.append({"incumbent_id": getattr(incumbent, "incumbent_id", None),
                     **ledger.selection_kwargs()})
        return next(it)

    step_fn.seen = seen  # type: ignore[attr-defined]
    return step_fn


def _report(*, accepted: bool, pending: bool = False):
    """An AcceptanceReport that is accepted, FAIL-rejected, or PENDING-rejected."""
    return evaluate_criteria(
        _ev(net=1 if accepted or pending else 0),
        dev={"arm_score": 9, "control_score": 9, "n": 20},
        safety={"clears_added": 0, "information_losing_removes": 0, "verified_relocations": 1},
        telemetry=({"interventions_executed": 5} if pending
                   else {"interventions_executed": 5, "mechanism_verified": 5}))


# ------------------------------------------------------------------ the moving incumbent

def test_an_acceptance_advances_the_incumbent_for_the_NEXT_round():
    """The next round's control arm is this round's candidate."""
    step_fn = _scripted(
        Step(OUTCOME_ACCEPTED, Cand("a"), _ev(1), new_incumbent=Inc("P1")),
        Step(OUTCOME_REJECTED, Cand("b"), _ev(0)))
    run = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=2)
    assert [s["incumbent_id"] for s in step_fn.seen] == ["P0", "P1"]
    assert run.final_incumbent.incumbent_id == "P1"


def test_remine_runs_after_an_acceptance_against_the_NEW_incumbent():
    seen: list[str] = []
    step_fn = _scripted(
        Step(OUTCOME_ACCEPTED, Cand("a"), _ev(1), new_incumbent=Inc("P1")),
        Step(OUTCOME_NO_CANDIDATE))
    run = run_rounds(Inc("P0"), step_fn=step_fn,
                     remine=lambda inc: seen.append(inc.incumbent_id), max_rounds=2)
    assert seen == ["P1"], "re-mining a superseded incumbent produces a stale residual"
    assert run.rounds[0].remined is True


def test_remine_does_not_run_when_nothing_was_accepted():
    """The residual has not changed, so re-mining would only cost a round."""
    seen: list[str] = []
    step_fn = _scripted(Step(OUTCOME_REJECTED, Cand("a"), _ev(0)))
    run = run_rounds(Inc("P0"), step_fn=step_fn, remine=seen.append, max_rounds=1)
    assert seen == [] and run.rounds[0].remined is False


def test_a_failed_remine_stops_the_run_rather_than_measuring_a_stale_residual():
    def boom(_inc):
        raise RuntimeError("trajectory dir missing")

    step_fn = _scripted(
        Step(OUTCOME_ACCEPTED, Cand("a"), _ev(1), new_incumbent=Inc("P1")),
        Step(OUTCOME_ACCEPTED, Cand("b"), _ev(1), new_incumbent=Inc("P2")))
    run = run_rounds(Inc("P0"), step_fn=step_fn, remine=boom, max_rounds=2)
    assert run.terminal == REMINE_FAILED
    assert len(run.rounds) == 1, "the second round must not run on a stale residual"
    assert not run.unattended
    assert "trajectory dir missing" in run.interventions[0].detail


def test_acceptance_without_a_new_incumbent_is_flagged_as_an_intervention():
    """A promote seam that was never wired would silently re-measure against the OLD incumbent."""
    step_fn = _scripted(Step(OUTCOME_ACCEPTED, Cand("a"), _ev(1), new_incumbent=None))
    run = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=1)
    assert not run.unattended
    assert "no new incumbent" in run.interventions[0].what
    assert run.final_incumbent.incumbent_id == "P0"


# ------------------------------------------------------------------ the ledger is carried

def test_the_ledger_handed_to_round_N_reflects_what_round_N_minus_1_concluded():
    """THE POINT OF THE DRIVER. Round 2 must see round 1's conclusion in selection_kwargs."""
    step_fn = _scripted(
        Step(OUTCOME_ACCEPTED, Cand("won"), _ev(1), new_incumbent=Inc("P1")),
        Step(OUTCOME_REJECTED, Cand("lost"), _ev(0)),
        Step(OUTCOME_NO_CANDIDATE))
    run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=3)
    assert step_fn.seen[0]["settled"] == () and step_fn.seen[0]["measured_losers"] == ()
    assert step_fn.seen[1]["settled"] == ("won",)
    assert step_fn.seen[2]["settled"] == ("won",)
    assert step_fn.seen[2]["measured_losers"] == ("lost",)


def test_a_PENDING_rejection_leaves_the_label_eligible_across_rounds():
    """The driver must not turn missing evidence into a permanent exclusion."""
    step_fn = _scripted(
        Step(OUTCOME_REJECTED, Cand("a"), _ev(1)),
        Step(OUTCOME_NO_CANDIDATE))
    run = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=2,
                     report_of=lambda r: _report(accepted=False, pending=True))
    assert run.ledger.measured_losers == ()
    assert run.ledger.retryable == ("a",)
    assert step_fn.seen[1]["measured_losers"] == ()


def test_without_report_of_a_rejection_is_recorded_as_a_measured_loser():
    """The conservative default: eligibility is pruned rather than a round re-spent. Documented as
    losing the FAIL/PENDING distinction, so this pins that it is a CHOICE, not an accident."""
    step_fn = _scripted(Step(OUTCOME_REJECTED, Cand("a"), _ev(0)))
    run = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=1)
    assert run.ledger.measured_losers == ("a",)


def test_an_externally_supplied_ledger_is_used_and_mutated_in_place():
    """So a run can resume from a saved ledger."""
    led = RoundLedger()
    led.record(label="old", outcome=OUTCOME_ACCEPTED)
    led.advance()
    step_fn = _scripted(Step(OUTCOME_NO_CANDIDATE))
    run = run_rounds(Inc("P0"), step_fn=step_fn, ledger=led, max_rounds=1)
    assert run.ledger is led
    assert step_fn.seen[0]["settled"] == ("old",)


# ------------------------------------------------------------------ termination

def test_no_candidate_terminates_as_space_exhausted():
    run = run_rounds(Inc("P0"), step_fn=_scripted(Step(OUTCOME_NO_CANDIDATE)), max_rounds=5)
    assert run.terminal == NO_CANDIDATE and len(run.rounds) == 1


def test_max_rounds_is_reported_as_a_budget_not_an_exhaustion():
    step_fn = _scripted(*[Step(OUTCOME_ACCEPTED, Cand(f"a{i}"), _ev(1),
                               new_incumbent=Inc(f"P{i+1}")) for i in range(3)])
    run = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=3)
    assert run.terminal == MAX_ROUNDS and len(run.accepted_labels) == 3


def test_consecutive_unresolvable_rounds_stop_the_run():
    """A loop that keeps selecting arms whose evidence never arrives is stuck, not working."""
    step_fn = _scripted(*[Step(OUTCOME_BLOCKED, Cand(f"a{i}"), _ev(1)) for i in range(5)])
    run = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=5, patience=2,
                     report_of=lambda r: _report(accepted=False, pending=True))
    assert run.terminal == CONSECUTIVE_NULLS and len(run.rounds) == 2
    assert run.ledger.retryable == ("a0", "a1")


def test_a_measured_loser_counts_as_progress_so_patience_resets():
    """Pruning a hypothesis changes what future rounds can measure: that is progress."""
    step_fn = _scripted(
        Step(OUTCOME_BLOCKED, Cand("a"), _ev(1)),
        Step(OUTCOME_REJECTED, Cand("b"), _ev(0)),
        Step(OUTCOME_BLOCKED, Cand("c"), _ev(1)),
        Step(OUTCOME_NO_CANDIDATE))
    run = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=4, patience=2,
                     report_of=lambda r: (_report(accepted=False, pending=True)
                                          if r.outcome == OUTCOME_BLOCKED
                                          else _report(accepted=False)))
    assert run.terminal == NO_CANDIDATE, "the loser in round 2 must reset the null streak"
    assert len(run.rounds) == 4


def test_an_acceptance_resets_patience():
    step_fn = _scripted(
        Step(OUTCOME_BLOCKED, Cand("a"), _ev(1)),
        Step(OUTCOME_ACCEPTED, Cand("b"), _ev(1), new_incumbent=Inc("P1")),
        Step(OUTCOME_BLOCKED, Cand("c"), _ev(1)),
        Step(OUTCOME_BLOCKED, Cand("d"), _ev(1)))
    run = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=4, patience=2,
                     report_of=lambda r: _report(accepted=r.outcome == OUTCOME_ACCEPTED,
                                                 pending=r.outcome != OUTCOME_ACCEPTED))
    assert run.terminal == CONSECUTIVE_NULLS and len(run.rounds) == 4


def test_should_continue_can_stop_the_run_before_a_round():
    step_fn = _scripted(Step(OUTCOME_NO_CANDIDATE))
    run = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=3,
                     should_continue=lambda r: False)
    assert run.terminal == STOP_REQUESTED and run.rounds == []


def test_zero_rounds_still_reports_a_terminal():
    run = run_rounds(Inc("P0"), step_fn=_scripted(), max_rounds=0)
    assert run.terminal == MAX_ROUNDS


# ------------------------------------------------------------------ honesty of the run record

def test_a_clean_run_reports_no_intervention_during_the_run():
    step_fn = _scripted(Step(OUTCOME_ACCEPTED, Cand("a"), _ev(1), new_incumbent=Inc("P1")),
                        Step(OUTCOME_NO_CANDIDATE))
    run = run_rounds(Inc("P0"), step_fn=step_fn, remine=lambda i: None, max_rounds=2)
    assert run.unattended and run.to_json()["no_intervention_during_run"] is True


def test_the_unattended_flag_is_falsified_by_any_intervention_not_by_a_judgement():
    """`unattended` is a mechanical fact about the run: did any round stop for a human?

    It is NOT a claim that the experiment needed no human setup -- it plainly did. This pins the
    only thing the flag can honestly mean, by showing a single recorded intervention falsifies it
    while nothing else does.
    """
    clean = run_rounds(Inc("P0"), step_fn=_scripted(Step(OUTCOME_NO_CANDIDATE)), max_rounds=1)
    assert clean.unattended

    # An acceptance with no promote seam wired is an intervention, and it alone flips the flag.
    dirty = run_rounds(Inc("P0"),
                       step_fn=_scripted(Step(OUTCOME_ACCEPTED, Cand("a"), _ev(1))),
                       max_rounds=1)
    assert dirty.interventions and not dirty.unattended
    assert dirty.to_json()["no_intervention_during_run"] is False

    # The serialised key is named for what it measures, so it cannot be quoted as a broader claim.
    assert "no_intervention_during_run" in dirty.to_json()
    assert "unattended" not in dirty.to_json()


def test_the_whole_run_serialises():
    step_fn = _scripted(Step(OUTCOME_ACCEPTED, Cand("a"), _ev(1), new_incumbent=Inc("P1")),
                        Step(OUTCOME_REJECTED, Cand("b"), _ev(0)),
                        Step(OUTCOME_NO_CANDIDATE))
    run = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=3)
    d = json.loads(json.dumps(run.to_json()))
    assert d["terminal"] == NO_CANDIDATE
    assert d["accepted_labels"] == ["a"]
    assert d["ledger"]["settled"] == ["a"] and d["ledger"]["measured_losers"] == ["b"]
    assert len(d["rounds"]) == 3


def test_summary_names_the_terminal_and_every_round():
    step_fn = _scripted(Step(OUTCOME_ACCEPTED, Cand("a"), _ev(2), new_incumbent=Inc("P1")),
                        Step(OUTCOME_NO_CANDIDATE))
    s = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=2).summary()
    assert NO_CANDIDATE in s and "R0 accepted" in s and "net=+2" in s


def test_selected_only_concludes_nothing_and_counts_as_a_null():
    """The arm was chosen and never measured, so nothing may be concluded about it."""
    step_fn = _scripted(Step(OUTCOME_SELECTED_ONLY, Cand("a")),
                        Step(OUTCOME_SELECTED_ONLY, Cand("b")))
    run = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=4, patience=2)
    assert run.ledger.entries == []
    assert run.terminal == CONSECUTIVE_NULLS


def test_label_is_read_from_a_mapping_candidate_too():
    step_fn = _scripted(Step(OUTCOME_REJECTED, {"label": "from_mapping"}, _ev(0)))
    run = run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=1)
    assert run.ledger.measured_losers == ("from_mapping",)


def test_a_missing_label_on_a_measured_round_raises_rather_than_recording_an_anonymous_verdict():
    step_fn = _scripted(Step(OUTCOME_REJECTED, None, _ev(0)))
    with pytest.raises(ValueError):
        run_rounds(Inc("P0"), step_fn=step_fn, max_rounds=1)
