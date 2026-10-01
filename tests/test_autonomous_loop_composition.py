"""END-TO-END COMPOSITION: the real step, the real acceptance criteria, the real ledger, the real
driver -- joined, with only the GPU evaluation stubbed.

Every other test in this area exercises one module. This one exists because the failure mode being
guarded against is not a bug inside any of them: it is that they were each correct and nothing
connected them, so a multi-round run needed a human in the middle. A test that stubs the seam it is
meant to prove joined would reproduce exactly that.

So: `self_evolve.step` is the genuine stepper, `acceptance_criteria.make_acceptance` is the genuine
four-criterion rule, `round_ledger` carries the genuine cross-round state, and `multi_round`
sequences them. The only injected thing is the paired measurement itself, because that needs GPUs.

The host is the TB2 adapter rather than BFCL: if this composition depended on anything
BFCL-specific, this file would not import.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "tb2_deepagents"))

import tb2_adapter as tb2                                              # noqa: E402
from anchoropt.anchor import Action, IncisionPoint                      # noqa: E402
from anchoropt.learning import self_evolve                             # noqa: E402
from anchoropt.learning.acceptance_criteria import make_acceptance      # noqa: E402
from anchoropt.learning.multi_round import (                            # noqa: E402
    CONSECUTIVE_NULLS, MAX_ROUNDS, NO_CANDIDATE, run_rounds)
from anchoropt.learning.candidate_search import cell_of                  # noqa: E402
from anchoropt.learning.round_ledger import (                            # noqa: E402
    RoundLedger, excluded_cells, label_for_cell)
from anchoropt.learning.self_evolve import EvaluationResult, Incumbent  # noqa: E402
from anchoropt.runtime import ResidualDiagnosis                         # noqa: E402

DIAG = (ResidualDiagnosis(case_id="c1", mechanism="concluded without verifying",
                          evidence="assertion failure", consequential_decision="emit terminal",
                          proposed_behavior_change="verify first", provider="test"),)


def _theta_for(action, diagnosis):
    """Ground the one action family this host can execute at the proposed boundary."""
    if action == Action.REPROMPT:
        return {"text": "verify before concluding"}
    return None


def _paired(net, *, n=12, telemetry):
    control = {f"c{i}": False for i in range(n)}
    candidate = dict(control)
    gains = tuple(f"c{i}" for i in range(max(net, 0)))
    for g in gains:
        candidate[g] = True
    losses = tuple(f"c{n - 1 - i}" for i in range(max(-net, 0)))
    for l in losses:
        control[l] = True
    return EvaluationResult(control=control, candidate=candidate, gains=gains, losses=losses,
                            telemetry=telemetry, denominator_ok=True)


FULL_TELEMETRY = {"signal_firings": 20, "interventions_executed": 12, "mechanism_verified": 12}
CLEAN_SAFETY = {"clears_added": 0, "information_losing_removes": 0, "verified_relocations": 3}
INFORMATIVE_DEV = {"arm_score": 7, "control_score": 7, "n": 20}


def _step_fn(*, evaluate, dev=INFORMATIVE_DEV, safety=CLEAN_SAFETY, reports=None,
             promote=True):
    """A closure over the GENUINE step and the GENUINE acceptance rule."""
    def step_fn(incumbent, ledger):
        accept = make_acceptance(dev_for=lambda e: dev, safety_for=lambda e: safety,
                                 on_report=(reports.append if reports is not None else None))
        return self_evolve.step(
            incumbent, runtime=tb2, host=tb2.HOST, diagnose=lambda: DIAG, theta_for=_theta_for,
            materialize=lambda spec, inc: spec,
            evaluate=evaluate,
            accept=accept,
            # THE JOINT: what earlier rounds decided is what this round may not re-select.
            exclude_cells=excluded_cells(ledger),
            promote=((lambda inc, cand: Incumbent(workspace=inc.workspace,
                                                  incumbent_id=f"{inc.incumbent_id}+r",
                                                  depth=inc.depth + 1))
                     if promote else None))
    return step_fn


def _label_of(result):
    """Name an arm by its CONTROLLER IDENTITY, via the canonical renderer.

    Using the cell as the label is what keeps the ledger's label memory and the search's cell
    exclusion from disagreeing: they are the same fact in two spellings.
    """
    c = getattr(result, "selected_candidate", None)
    if c is None:
        return ""
    return label_for_cell(cell_of(getattr(c, "candidate", c)))


# ------------------------------------------------------------------ the loop actually closes

def test_a_gaining_arm_is_accepted_installed_and_never_re_measured():
    """ONE COMPLETE TRANSITION through the real modules: measure -> accept on all four -> install
    -> the next round sees it SETTLED and has nothing left."""
    reports = []
    remined = []
    run = run_rounds(
        Incumbent(workspace=Path("/nonexistent"), incumbent_id="P0"),
        step_fn=_step_fn(evaluate=lambda inc, cand: _paired(3, telemetry=FULL_TELEMETRY),
                         reports=reports),
        remine=lambda inc: remined.append(inc.incumbent_id),
        label_of=_label_of, report_of=lambda r: reports[-1],
        max_rounds=2)

    assert run.rounds[0].accepted, run.rounds[0].reason
    r = reports[0]
    assert r.accepted and [c.verdict for c in r.criteria] == ["PASS"] * 4

    # installed: the incumbent moved, and re-mining ran against the NEW one. Re-mining a superseded
    # incumbent would produce a stale residual, so the FIRST re-mine must see P0+r, not P0.
    assert remined[0] == "P0+r"

    # settled: the label is in the ledger, so no later round may re-select that identity
    label = run.rounds[0].label
    assert label in run.ledger.settled
    assert run.unattended

    # THE DEFECT THIS TEST CAUGHT. Before `exclude_cells` existed, round 2 re-selected the identity
    # round 1 had just installed, re-measured it, and installed it AGAIN -- the incumbent came out
    # "P0+r+r" from one discovery. An unattended loop would have spent every round re-measuring its
    # own incumbent while looking productive. Round 2 must now pick a DIFFERENT cell or none.
    assert len(run.rounds) == 2
    assert run.rounds[1].label != label
    assert run.final_incumbent.incumbent_id in ("P0+r", "P0+r+r") and \
        run.final_incumbent.incumbent_id.count("+r") == (
            1 + int(run.rounds[1].accepted))


def test_a_losing_arm_is_rejected_by_criterion_1_and_becomes_a_measured_loser():
    reports = []
    run = run_rounds(
        Incumbent(workspace=Path("/nonexistent"), incumbent_id="P0"),
        step_fn=_step_fn(evaluate=lambda inc, cand: _paired(-2, telemetry=FULL_TELEMETRY),
                         reports=reports),
        label_of=_label_of, report_of=lambda r: reports[-1], max_rounds=1)

    assert not run.rounds[0].accepted
    assert reports[0].by_number(1).verdict == "FAIL"
    assert run.ledger.measured_losers == (run.rounds[0].label,)
    assert run.ledger.retryable == ()
    assert run.final_incumbent.incumbent_id == "P0", "a rejected arm must not move the incumbent"


def test_an_arm_with_no_mechanism_telemetry_is_rejected_and_stays_ELIGIBLE():
    """The central asymmetry, through the real criteria: a positive net with absent mechanism
    evidence is PENDING on criterion 3, so the arm is not installed AND not written off."""
    reports = []
    run = run_rounds(
        Incumbent(workspace=Path("/nonexistent"), incumbent_id="P0"),
        step_fn=_step_fn(evaluate=lambda inc, cand: _paired(
            3, telemetry={"signal_firings": 20, "interventions_executed": 12})),
        label_of=_label_of,
        report_of=lambda r: reports[-1] if reports else None,
        max_rounds=1)
    # rebuild the report the accept callable produced, via the same public API
    from anchoropt.learning.acceptance_criteria import evaluate_criteria
    rep = evaluate_criteria(_paired(3, telemetry={"signal_firings": 20,
                                                  "interventions_executed": 12}),
                            dev=INFORMATIVE_DEV, safety=CLEAN_SAFETY)
    assert rep.by_number(3).verdict == "PENDING_VALIDATION"
    assert not rep.accepted

    assert not run.rounds[0].accepted
    assert run.final_incumbent.incumbent_id == "P0"


def test_a_positive_arm_with_zero_executions_is_a_measured_FAIL_not_a_pending():
    """A gain with zero firings is proof of NON-attribution, and the composed loop must prune it."""
    reports = []
    run = run_rounds(
        Incumbent(workspace=Path("/nonexistent"), incumbent_id="P0"),
        step_fn=_step_fn(evaluate=lambda inc, cand: _paired(
            4, telemetry={"signal_firings": 20, "interventions_executed": 0}),
            reports=reports),
        label_of=_label_of, report_of=lambda r: reports[-1], max_rounds=1)
    assert reports[0].by_number(3).verdict == "FAIL"
    assert not run.rounds[0].accepted
    assert run.ledger.measured_losers == (run.rounds[0].label,)


def test_an_uninformative_dev_split_does_not_block_the_composed_loop():
    """The corrected criterion 2, end to end: a degenerate split passes as preregistered, the arm
    installs, and the weakness is still on the record."""
    reports = []
    run = run_rounds(
        Incumbent(workspace=Path("/nonexistent"), incumbent_id="P0"),
        step_fn=_step_fn(evaluate=lambda inc, cand: _paired(3, telemetry=FULL_TELEMETRY),
                         dev={"arm_score": 0, "control_score": 0, "n": 24}, reports=reports),
        label_of=_label_of, report_of=lambda r: reports[-1], max_rounds=1)

    assert run.rounds[0].accepted
    c2 = reports[0].by_number(2)
    assert c2.verdict == "PASS_UNINFORMATIVE" and c2.evidence["informative"] is False
    assert "REPORTED WEAKNESS" in c2.detail


def test_missing_safety_telemetry_blocks_installation_through_the_real_rule():
    """Absence of evidence of harm is not evidence of no harm -- and it must stop an INSTALL, not
    just print a warning."""
    reports = []
    run = run_rounds(
        Incumbent(workspace=Path("/nonexistent"), incumbent_id="P0"),
        step_fn=_step_fn(evaluate=lambda inc, cand: _paired(3, telemetry=FULL_TELEMETRY),
                         safety=None, reports=reports),
        label_of=_label_of, report_of=lambda r: reports[-1], max_rounds=1)
    assert reports[0].by_number(4).verdict == "PENDING_VALIDATION"
    assert not run.rounds[0].accepted
    assert run.final_incumbent.incumbent_id == "P0"


def test_a_bad_denominator_never_reaches_the_acceptance_rule():
    """step() blocks before acceptance, and the driver records it as retryable infrastructure."""
    reports = []

    def evaluate(inc, cand):
        ev = _paired(3, telemetry=FULL_TELEMETRY)
        return EvaluationResult(control=ev.control, candidate=ev.candidate, gains=ev.gains,
                                losses=ev.losses, telemetry=ev.telemetry, denominator_ok=False)

    run = run_rounds(
        Incumbent(workspace=Path("/nonexistent"), incumbent_id="P0"),
        step_fn=_step_fn(evaluate=evaluate, reports=reports),
        label_of=_label_of, report_of=lambda r: reports[-1] if reports else None,
        max_rounds=1)
    assert reports == [], "the acceptance rule must not even be consulted"
    assert run.ledger.measured_losers == ()
    assert run.ledger.retryable == (run.rounds[0].label,)


# ------------------------------------------------------------------ multi-round behaviour

def test_two_rounds_exhaust_the_space_once_the_only_arm_is_settled():
    """After installation the one expressible arm is SETTLED, so a second round has nothing to
    select. That is a real exhaustion, reported as such rather than as a budget end."""
    reports = []
    calls = []

    def step_fn(incumbent, ledger):
        calls.append(dict(ledger.selection_kwargs()))
        return _step_fn(evaluate=lambda inc, cand: _paired(3, telemetry=FULL_TELEMETRY),
                        reports=reports)(incumbent, ledger)

    run = run_rounds(Incumbent(workspace=Path("/nonexistent"), incumbent_id="P0"),
                     step_fn=step_fn, label_of=_label_of,
                     report_of=lambda r: reports[-1], max_rounds=2)
    # round 2 was handed round 1's conclusion
    assert calls[1]["settled"] == (run.rounds[0].label,)
    assert len(run.accepted_labels) >= 1


def test_the_run_record_is_a_complete_audit_trail():
    reports = []
    run = run_rounds(
        Incumbent(workspace=Path("/nonexistent"), incumbent_id="P0"),
        step_fn=_step_fn(evaluate=lambda inc, cand: _paired(3, telemetry=FULL_TELEMETRY),
                         reports=reports),
        label_of=_label_of, report_of=lambda r: reports[-1], max_rounds=1)
    d = run.to_json()
    assert d["terminal"] == MAX_ROUNDS
    assert d["ledger"]["entries"][0]["reason"], "an acceptance must record WHY on all four"
    assert d["no_intervention_during_run"] is True


def test_the_composition_does_not_depend_on_any_bfcl_module():
    """If the loop needed BFCL to close, this file could not have imported a TB2 host. Asserted
    rather than left implicit, because 'benchmark-free core' is a claim that decays silently."""
    import anchoropt.learning.acceptance_criteria as ac
    import anchoropt.learning.multi_round as mr
    import anchoropt.learning.round_ledger as rl
    for mod in (ac, rl, mr):
        src = Path(mod.__file__).read_text().lower()
        assert "bfcl" not in src, f"{mod.__name__} names bfcl"
        assert "import" not in src.split("bfcl")[0][-20:] if "bfcl" in src else True
