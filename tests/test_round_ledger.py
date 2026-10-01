"""Cross-round state. The load-bearing tests are about the ASYMMETRY between rejection kinds.

A rejection that MEASURED a negative must permanently exclude the label; a rejection caused by
ABSENT evidence must not. Getting that wrong is costly in opposite directions -- premature false
exhaustion, or re-spending GPU rounds on a known failure forever -- so both directions are pinned.

Benchmark-independent by construction: labels are opaque strings and no test names a corpus, a
store, or a signal.
"""

from __future__ import annotations

import json

import pytest

from anchoropt.learning.acceptance_criteria import (
    check_dev, check_mechanism, check_safety, check_train, evaluate_criteria)
from anchoropt.learning.candidate_selection import (
    MEASURED_LOSER as SEL_MEASURED_LOSER, SETTLED as SEL_SETTLED, Candidate, select_candidate)
from anchoropt.learning.round_ledger import (
    MEASURED_LOSER, OUTCOME_ACCEPTED, OUTCOME_BLOCKED, OUTCOME_NO_CANDIDATE, OUTCOME_REJECTED,
    OUTCOME_SELECTED_ONLY, RETRYABLE, SETTLED, LedgerEntry, RoundLedger, classify_rejection,
    seed_settled_from_registry)
from anchoropt.learning.self_evolve import EvaluationResult


def _ev(gains=(), losses=(), n=10, telemetry=None):
    control = {f"c{i}": False for i in range(n)}
    candidate = dict(control)
    for g in gains:
        candidate[g] = True
    for l in losses:
        control[l] = True
    return EvaluationResult(control=control, candidate=candidate, gains=tuple(gains),
                            losses=tuple(losses), telemetry=telemetry or {}, denominator_ok=True)


def _report(*, net_positive=True, dev=None, safety=None, telemetry=None):
    return evaluate_criteria(
        _ev(gains=("c1",) if net_positive else (), telemetry=telemetry),
        dev=dev, safety=safety)


# ------------------------------------------------------------------ the contract with self_evolve

def test_outcome_strings_match_self_evolve_exactly():
    """The ledger restates them so it need not import the stepper. Drift must fail, not pass."""
    from anchoropt.learning import self_evolve as se
    assert (OUTCOME_ACCEPTED, OUTCOME_REJECTED, OUTCOME_NO_CANDIDATE, OUTCOME_BLOCKED,
            OUTCOME_SELECTED_ONLY) == (se.OUTCOME_ACCEPTED, se.OUTCOME_REJECTED,
                                       se.OUTCOME_NO_CANDIDATE, se.OUTCOME_BLOCKED,
                                       se.OUTCOME_SELECTED_ONLY)


def test_selection_kwargs_splat_into_select_candidate():
    """THE JOINT THIS MODULE EXISTS TO CLOSE. `select_candidate` documented these two sets as
    caller-carried and had no caller outside its own tests."""
    led = RoundLedger()
    led.record(label="won", outcome=OUTCOME_ACCEPTED)
    led.advance()
    led.record(label="lost", outcome=OUTCOME_REJECTED, report=None)

    sel = select_candidate([Candidate("won", "transform", "s", "b", support=9, fires_on_states=5),
                            Candidate("lost", "transform", "s", "b", support=8, fires_on_states=5),
                            Candidate("fresh", "transform", "s", "b", support=1, fires_on_states=5)],
                           **led.selection_kwargs())
    assert sel.chosen.label == "fresh"
    reasons = {e.label: e.reason for e in sel.excluded}
    assert reasons["won"] == SEL_SETTLED
    assert reasons["lost"] == SEL_MEASURED_LOSER


# ------------------------------------------------------------------ the asymmetry

def test_accepted_becomes_settled():
    led = RoundLedger()
    e = led.record(label="a", outcome=OUTCOME_ACCEPTED)
    assert e.disposition == SETTLED and led.settled == ("a",)
    assert e.excludes_from_future_rounds


def test_rejection_on_a_measured_FAIL_is_a_loser():
    """Criterion 1 measured net <= 0. That is a verdict from evidence."""
    rep = _report(net_positive=False, dev={"arm_score": 9, "control_score": 9, "n": 20},
                  safety={"clears_added": 0, "information_losing_removes": 0},
                  telemetry={"interventions_executed": 5, "mechanism_verified": 5})
    assert rep.by_number(1).verdict == "FAIL"
    led = RoundLedger()
    e = led.record(label="a", outcome=OUTCOME_REJECTED, report=rep, net=-2)
    assert e.disposition == MEASURED_LOSER
    assert led.measured_losers == ("a",) and led.retryable == ()
    assert "C1" in e.reason


def test_rejection_on_PENDING_evidence_is_RETRYABLE_not_a_loser():
    """THE CENTRAL CASE. Absent mechanism telemetry measured nothing about the mechanism.
    Excluding the label would convert missing evidence into a permanent negative verdict -- exactly
    what the PENDING verdict exists to prevent."""
    rep = _report(dev={"arm_score": 9, "control_score": 9, "n": 20},
                  safety={"clears_added": 0, "information_losing_removes": 0},
                  telemetry={"interventions_executed": 5})      # no mechanism_verified key
    assert rep.by_number(3).verdict == "PENDING_VALIDATION"
    led = RoundLedger()
    e = led.record(label="a", outcome=OUTCOME_REJECTED, report=rep)
    assert e.disposition == RETRYABLE
    assert not e.excludes_from_future_rounds
    assert led.measured_losers == () and led.retryable == ("a",)
    assert e.remedy, "a retryable entry must carry what would resolve it"


def test_a_retryable_label_stays_eligible_in_the_next_round():
    """The consequence of the above, at the seam: missing evidence must not silently prune."""
    rep = _report(dev={"arm_score": 9, "control_score": 9, "n": 20},
                  safety={"clears_added": 0, "information_losing_removes": 0},
                  telemetry={"interventions_executed": 5})
    led = RoundLedger()
    led.record(label="a", outcome=OUTCOME_REJECTED, report=rep)
    sel = select_candidate([Candidate("a", "transform", "s", "b", support=5, fires_on_states=3)],
                           **led.selection_kwargs())
    assert sel.chosen is not None and sel.chosen.label == "a"


def test_a_FAIL_dominates_a_coexisting_PENDING():
    """net <= 0 is not made better by more telemetry."""
    rep = _report(net_positive=False, dev=None, safety=None, telemetry={"interventions_executed": 5})
    assert rep.by_number(1).verdict == "FAIL"
    assert any(c.verdict == "PENDING_VALIDATION" for c in rep.blocking)
    disp, reason, _ = classify_rejection(rep)
    assert disp == MEASURED_LOSER and "C1" in reason


def test_blocked_is_always_retryable_because_it_is_infrastructure():
    """`step()` refuses to let a bad denominator reach the acceptance rule. The ledger must not
    undo that refusal one layer up by recording it as a negative result."""
    led = RoundLedger()
    e = led.record(label="a", outcome=OUTCOME_BLOCKED)
    assert e.disposition == RETRYABLE
    assert led.measured_losers == ()
    assert "denominator" in e.reason and e.remedy


@pytest.mark.parametrize("outcome", [OUTCOME_NO_CANDIDATE, OUTCOME_SELECTED_ONLY])
def test_outcomes_that_conclude_nothing_record_nothing(outcome):
    """SELECTED_ONLY never measured the arm it chose; NO_CANDIDATE had no arm."""
    led = RoundLedger()
    assert led.record(label="a", outcome=outcome) is None
    assert led.entries == [] and led.measured_losers == () and led.settled == ()


def test_default_accept_rejection_is_a_measurement():
    """With no report the round used the v0.1 default accept, which rejects on net <= 0 or zero
    engagement. Both are measurements, so the label is a loser rather than retryable."""
    disp, reason, _ = classify_rejection(None)
    assert disp == MEASURED_LOSER and reason


def test_rejection_with_no_blocking_criterion_does_not_invent_a_verdict():
    rep = _report(dev={"arm_score": 9, "control_score": 9, "n": 20},
                  safety={"clears_added": 0, "information_losing_removes": 0,
                          "verified_relocations": 1},
                  telemetry={"interventions_executed": 5, "mechanism_verified": 5})
    assert rep.accepted and not rep.blocking
    disp, _, remedy = classify_rejection(rep)
    assert disp == RETRYABLE and remedy


def test_an_uninformative_dev_pass_does_not_make_a_label_retryable():
    """Criterion 2 on a degenerate split PASSES as preregistered, so it is not a blocker and cannot
    push the round into a rejection. Pins the corrected criterion 2 semantics at this seam."""
    rep = _report(dev={"arm_score": 0, "control_score": 0, "n": 24},
                  safety={"clears_added": 0, "information_losing_removes": 0},
                  telemetry={"interventions_executed": 5, "mechanism_verified": 5})
    assert rep.by_number(2).verdict == "PASS_UNINFORMATIVE"
    assert rep.accepted
    led = RoundLedger()
    assert led.record(label="a", outcome=OUTCOME_ACCEPTED, report=rep).disposition == SETTLED


# ------------------------------------------------------------------ bookkeeping

def test_a_label_later_accepted_is_no_longer_a_loser():
    """A hardened re-measurement supersedes an earlier negative; the audit trail keeps both."""
    led = RoundLedger()
    led.record(label="a", outcome=OUTCOME_REJECTED, report=None)
    led.advance()
    led.record(label="a", outcome=OUTCOME_ACCEPTED)
    assert led.settled == ("a",) and led.measured_losers == ()
    assert len(led.for_label("a")) == 2, "both conclusions stay on the record"


def test_settled_wins_over_retryable_too():
    led = RoundLedger()
    led.record(label="a", outcome=OUTCOME_BLOCKED)
    led.advance()
    led.record(label="a", outcome=OUTCOME_ACCEPTED)
    assert led.settled == ("a",) and led.retryable == ()


def test_round_index_advances_and_is_recorded():
    led = RoundLedger()
    led.record(label="a", outcome=OUTCOME_REJECTED, report=None)
    assert led.advance() == 1
    led.record(label="b", outcome=OUTCOME_REJECTED, report=None)
    assert [e.round_index for e in led.entries] == [0, 1]


def test_a_candidate_outcome_without_a_label_is_an_error():
    with pytest.raises(ValueError):
        RoundLedger().record(label="", outcome=OUTCOME_ACCEPTED)


def test_unknown_outcome_raises_rather_than_being_ignored():
    with pytest.raises(ValueError):
        RoundLedger().record(label="a", outcome="something_new")


def test_seed_from_registry_is_idempotent_and_marks_prior_rounds():
    class E:
        def __init__(self, name):
            self.name = name

    led = RoundLedger()
    assert seed_settled_from_registry(led, [E("x"), E("y")]) == ("x", "y")
    assert seed_settled_from_registry(led, [E("x"), E("y")]) == ()
    assert led.settled == ("x", "y")
    assert all(e.round_index == -1 for e in led.entries), "prior acceptances are not this run's"


def test_seed_accepts_bare_strings_so_the_core_needs_no_registry_import():
    led = RoundLedger()
    seed_settled_from_registry(led, ["a"])
    assert led.settled == ("a",)


# ------------------------------------------------------------------ persistence and reporting

def test_round_trips_through_json_preserving_every_disposition():
    led = RoundLedger()
    led.record(label="won", outcome=OUTCOME_ACCEPTED)
    led.advance()
    led.record(label="lost", outcome=OUTCOME_REJECTED, report=None, net=-3)
    led.advance()
    led.record(label="unknown", outcome=OUTCOME_BLOCKED)
    led.advance()

    back = RoundLedger.from_json(json.loads(json.dumps(led.to_json())))
    assert back.rounds_run == led.rounds_run
    assert back.settled == led.settled
    assert back.measured_losers == led.measured_losers
    assert back.retryable == led.retryable
    assert [e.reason for e in back.entries] == [e.reason for e in led.entries]


def test_save_and_load(tmp_path):
    led = RoundLedger()
    led.record(label="a", outcome=OUTCOME_ACCEPTED)
    p = led.save(tmp_path / "sub" / "ledger.json")
    assert RoundLedger.load(p).settled == ("a",)


def test_the_persisted_form_carries_the_evidence_not_just_the_sets():
    """A future round declining to re-spend an evaluation must be able to say which round measured
    it and what the verdict was, reconstructed from disk without re-running anything."""
    rep = _report(net_positive=False, dev={"arm_score": 9, "control_score": 9, "n": 20},
                  safety={"clears_added": 0, "information_losing_removes": 0},
                  telemetry={"interventions_executed": 5, "mechanism_verified": 5})
    led = RoundLedger()
    led.record(label="a", outcome=OUTCOME_REJECTED, report=rep, net=-2)
    d = led.to_json()
    e = d["entries"][0]
    assert e["round_index"] == 0 and e["net"] == -2 and "C1" in e["reason"]


def test_summary_surfaces_remedies_as_run_blockers():
    led = RoundLedger()
    led.record(label="a", outcome=OUTCOME_BLOCKED)
    s = led.summary()
    assert "RETRYABLE" in s and "REMEDY for a" in s


# ------------------------------------------------------------------ the cell/label bridge
# The ledger remembers LABELS; candidate_search excludes CELLS. If those two memories disagree, a
# settled controller gets re-measured through whichever one the round happens to consult -- which is
# exactly the defect the composition test caught. So the cell is canonical and a label renders it.

def test_a_cell_label_round_trips():
    from anchoropt.learning.round_ledger import cell_from_label, label_for_cell
    cell = ("post_generation_pre_exec", "terminal_response_proposed", "reprompt")
    assert cell_from_label(label_for_cell(cell)) == cell


def test_a_label_that_does_not_encode_a_cell_yields_no_exclusion():
    """Guessing a cell from an arbitrary label would exclude a controller nobody decided about,
    which is worse than excluding none."""
    from anchoropt.learning.round_ledger import cell_from_label, excluded_cells
    assert cell_from_label("some_run_specific_name") is None
    led = RoundLedger()
    led.record(label="some_run_specific_name", outcome=OUTCOME_ACCEPTED)
    assert excluded_cells(led) == ()
    assert led.settled == ("some_run_specific_name",), "the label memory is unaffected"


def test_settled_and_measured_losers_both_exclude_their_cells():
    from anchoropt.learning.round_ledger import excluded_cells, label_for_cell
    led = RoundLedger()
    led.record(label=label_for_cell(("b1", "s", "a")), outcome=OUTCOME_ACCEPTED)
    led.advance()
    led.record(label=label_for_cell(("b2", "s", "a")), outcome=OUTCOME_REJECTED, report=None)
    assert set(excluded_cells(led)) == {("b1", "s", "a"), ("b2", "s", "a")}


def test_a_RETRYABLE_cell_is_NOT_excluded():
    """The asymmetry must hold at the identity level too, or the two memories disagree: absent
    evidence would permanently remove the controller through the cell path while the label path
    still called it eligible."""
    from anchoropt.learning.round_ledger import excluded_cells, label_for_cell
    led = RoundLedger()
    led.record(label=label_for_cell(("b", "s", "a")), outcome=OUTCOME_BLOCKED)
    assert led.retryable == (label_for_cell(("b", "s", "a")),)
    assert excluded_cells(led) == ()


def test_excluded_cells_agrees_with_the_label_sets_by_construction():
    """Property: a cell is excluded iff its label is in settled|measured_losers."""
    from anchoropt.learning.round_ledger import cell_from_label, excluded_cells
    from anchoropt.learning.round_ledger import label_for_cell as L
    led = RoundLedger()
    for i, outcome in enumerate((OUTCOME_ACCEPTED, OUTCOME_REJECTED, OUTCOME_BLOCKED)):
        led.record(label=L((f"b{i}", "s", "a")), outcome=outcome,
                   report=None if outcome == OUTCOME_REJECTED else None)
        led.advance()
    blocked_labels = set(led.settled) | set(led.measured_losers)
    assert {cell_from_label(l) for l in blocked_labels} == set(excluded_cells(led))
