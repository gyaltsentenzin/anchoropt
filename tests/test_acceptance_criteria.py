"""The four criteria must be checkable by machine, and must REFUSE on absent evidence.

The load-bearing tests here are the negative ones. Every way this project has previously fooled
itself reduces to reading silence as success, so a criterion with no telemetry must not pass.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from anchoropt.learning.acceptance_criteria import (
    FAIL, PASS, PASS_UNINFORMATIVE, PENDING, AcceptanceReport, check_dev, check_mechanism,
    check_safety, check_train, evaluate_criteria, make_acceptance)
from anchoropt.learning.self_evolve import EvaluationResult

REPO = pathlib.Path(__file__).resolve().parents[1]


def _ev(gains=(), losses=(), n=10, telemetry=None):
    control = {f"c{i}": False for i in range(n)}
    candidate = dict(control)
    for g in gains:
        candidate[g] = True
    for l in losses:
        control[l] = True
    return EvaluationResult(control=control, candidate=candidate, gains=tuple(gains),
                            losses=tuple(losses), telemetry=telemetry or {},
                            denominator_ok=True)


# ---------------------------------------------------------------- criterion 1

def test_c1_passes_on_positive_net():
    r = check_train(_ev(gains=("c1", "c2"), losses=("c3",)))
    assert r.verdict == PASS and r.evidence["net"] == 1


@pytest.mark.parametrize("gains,losses", [((), ()), (("c1",), ("c2",)), ((), ("c1",))])
def test_c1_fails_on_non_positive_net(gains, losses):
    assert check_train(_ev(gains=gains, losses=losses)).verdict == FAIL


def test_c1_is_pending_not_pass_when_nothing_was_measured():
    """Zero paired cases is not a clean null -- it is an absent measurement."""
    r = check_train(EvaluationResult(control={}, candidate={}, denominator_ok=True))
    assert r.verdict == PENDING


# ---------------------------------------------------------------- criterion 2

def test_c2_net_zero_is_a_PASS_because_a_gain_is_not_required():
    """The predefined rule is NON-REGRESSION. Requiring a dev gain would be a stricter rule."""
    r = check_dev({"arm_score": 9, "control_score": 9, "n": 20})
    assert r.verdict == PASS
    assert "not required" in r.detail


def test_c2_positive_net_passes():
    assert check_dev({"arm_score": 11, "control_score": 9, "n": 20}).verdict == PASS


def test_c2_fails_only_on_an_actual_aggregate_regression():
    r = check_dev({"arm_score": 8, "control_score": 9, "n": 20})
    assert r.verdict == FAIL and r.evidence["net"] == -1


def test_c2_degenerate_split_PASSES_but_is_flagged_uninformative():
    """kv dev: arm 0/24 AND control 0/24.

    The preregistered rule is NO AGGREGATE REGRESSION, and there was none, so this PASSES -- an
    uninformative split must be reported, NOT used to retrospectively raise the threshold. Blocking
    here would substitute a stricter rule chosen after seeing the data, which is the same class of
    error as weakening one. What the checker owes is that the weakness travels with the verdict.
    """
    r = check_dev({"arm_score": 0, "control_score": 0, "n": 24, "firings": 14})
    assert r.verdict == PASS_UNINFORMATIVE
    assert r.allows_accept
    assert r.evidence["informative"] is False
    assert "could not have detected a regression" in r.detail
    assert "does NOT gate" in r.remedy


def test_c2_an_informative_pass_is_distinguishable_from_an_uninformative_one():
    """Both pass; only one carries transfer evidence. A consumer must be able to tell them apart."""
    strong = check_dev({"arm_score": 9, "control_score": 9, "n": 20})
    weak = check_dev({"arm_score": 0, "control_score": 0, "n": 24})
    assert strong.allows_accept and weak.allows_accept
    assert strong.evidence["informative"] is True
    assert weak.evidence["informative"] is False
    assert strong.verdict != weak.verdict


def test_c2_missing_dev_is_pending():
    assert check_dev(None).verdict == PENDING
    assert not check_dev(None).allows_accept


# ---------------------------------------------------------------- criterion 3

def test_c3_requires_positive_verified_evidence():
    r = check_mechanism({"signal_firings": 84, "interventions_executed": 67,
                         "mechanism_verified": 67})
    assert r.verdict == PASS


def test_c3_zero_executions_is_a_FAIL_not_a_pending():
    """A positive delta with zero executions is proof of NON-attribution."""
    r = check_mechanism({"signal_firings": 84, "interventions_executed": 0})
    assert r.verdict == FAIL


def test_c3_firing_without_verification_is_pending():
    """Installation is not execution, and firing is not validation."""
    r = check_mechanism({"signal_firings": 84, "interventions_executed": 67})
    assert r.verdict == PENDING
    assert not r.allows_accept


def test_c3_executions_but_zero_verified_fails():
    r = check_mechanism({"signal_firings": 84, "interventions_executed": 67,
                         "mechanism_verified": 0})
    assert r.verdict == FAIL


def test_c3_no_telemetry_at_all_is_pending():
    assert check_mechanism({}).verdict == PENDING
    assert check_mechanism(None).verdict == PENDING


def test_c3_unattributed_requests_are_MISSING_EVIDENCE_not_a_pass():
    """The 84-request population must SUM. A request with no recorded reason is missing evidence,
    which is exactly the mistake the relocation reconciliation corrected."""
    r = check_mechanism({"signal_firings": 84, "interventions_executed": 67,
                         "mechanism_verified": 67, "mechanism_requested": 84})
    assert r.verdict == PENDING
    assert r.evidence["unattributed"] == 17
    assert not r.allows_accept


def test_c3_fully_attributed_population_passes():
    """The reconciled kv partition: 67 verified + 1 guard-decline + 16 bound-refusals = 84."""
    r = check_mechanism({"signal_firings": 84, "interventions_executed": 67,
                         "mechanism_verified": 67, "mechanism_requested": 84,
                         "mechanism_declined_with_reason": 1,
                         "mechanism_refused_by_guard": 16})
    assert r.verdict == PASS
    assert r.evidence["unattributed"] == 0


# ---------------------------------------------------------------- criterion 4

def test_c4_passes_with_zero_destructive_and_reports_relocations():
    r = check_safety({"clears_added": 0, "information_losing_removes": 0,
                      "verified_relocations": 67})
    assert r.verdict == PASS
    assert r.evidence["verified_relocations"] == 67


def test_c4_any_clear_added_fails_categorically():
    r = check_safety({"clears_added": 1, "information_losing_removes": 0})
    assert r.verdict == FAIL and "categorically forbidden" in r.detail


def test_c4_information_losing_remove_fails():
    r = check_safety({"clears_added": 0, "information_losing_removes": 1})
    assert r.verdict == FAIL


def test_c4_verified_relocations_are_never_a_failure():
    """Counting a verified relocation as destructive would forbid the only safe capacity
    recovery this project has."""
    r = check_safety({"clears_added": 0, "information_losing_removes": 0,
                      "verified_relocations": 10_000})
    assert r.verdict == PASS


def test_c4_absent_counters_are_pending_not_pass():
    assert check_safety(None).verdict == PENDING
    assert check_safety({"verified_relocations": 67}).verdict == PENDING
    assert check_safety({"clears_added": 0}).verdict == PENDING


# ---------------------------------------------------------------- the report

def test_report_accepts_only_when_every_criterion_allows():
    rep = evaluate_criteria(
        _ev(gains=("c1",), telemetry={"signal_firings": 10, "interventions_executed": 5,
                                      "mechanism_verified": 5}),
        dev={"arm_score": 9, "control_score": 9, "n": 20},
        safety={"clears_added": 0, "information_losing_removes": 0, "verified_relocations": 5})
    assert rep.accepted and not rep.blocking


def test_report_blocks_on_a_single_pending():
    rep = evaluate_criteria(
        _ev(gains=("c1",), telemetry={"signal_firings": 10, "interventions_executed": 5,
                                      "mechanism_verified": 5}),
        dev={"arm_score": 9, "control_score": 9, "n": 20},
        safety=None)
    assert not rep.accepted
    assert [c.number for c in rep.blocking] == [4]


def test_report_with_no_evidence_at_all_rejects():
    rep = evaluate_criteria(_ev(gains=("c1",)))
    assert not rep.accepted
    assert {c.number for c in rep.blocking} == {2, 3, 4}


def test_report_is_json_serialisable():
    rep = evaluate_criteria(_ev(gains=("c1",)))
    assert json.loads(json.dumps(rep.to_dict()))["accepted"] is False


def test_all_four_criteria_are_always_reported():
    rep = evaluate_criteria(_ev())
    assert [c.number for c in rep.criteria] == [1, 2, 3, 4]


# ---------------------------------------------------------------- the seam

def test_make_acceptance_matches_the_self_evolve_seam_signature():
    accept = make_acceptance(
        dev_for=lambda e: {"arm_score": 9, "control_score": 9, "n": 20},
        safety_for=lambda e: {"clears_added": 0, "information_losing_removes": 0,
                              "verified_relocations": 3})
    ok, reason = accept(_ev(gains=("c1",), telemetry={"signal_firings": 10,
                                                      "interventions_executed": 5,
                                                      "mechanism_verified": 5}))
    assert ok is True and isinstance(reason, str)


def test_make_acceptance_refuses_when_evidence_was_never_gathered():
    """Supplying no dev/safety lookups is legitimate, and must REJECT rather than install."""
    accept = make_acceptance()
    ok, reason = accept(_ev(gains=("c1",), telemetry={"signal_firings": 10,
                                                      "interventions_executed": 5,
                                                      "mechanism_verified": 5}))
    assert ok is False
    assert "PENDING_VALIDATION" in reason


def test_the_seam_is_accepted_by_self_evolve_step():
    """Guards against drift between this module and the seam it plugs into."""
    import inspect

    from anchoropt.learning import self_evolve
    assert "accept" in inspect.signature(self_evolve.step).parameters


def test_on_report_hook_receives_the_full_report():
    seen: list[AcceptanceReport] = []
    accept = make_acceptance(on_report=seen.append)
    accept(_ev(gains=("c1",)))
    assert len(seen) == 1 and len(seen[0].criteria) == 4


# ---------------------------------------------------------------- replay of the measured arm

def test_replays_the_accepted_kv_relocation_to_its_RECORDED_verdict():
    """The measured kv relocation arm, fed through the machine checker, must reproduce what was
    adjudicated BY HAND: C1 PASS, C2 pass-with-weakness-reported, C3 PASS, C4 PASS -- ACCEPTED.

    This is the regression test for the whole module: an unattended round must reach the same
    verdict a human reached on the same evidence, on the rule as preregistered. C2 is the load
    bearing one -- the hand adjudication recorded it as satisfied with the degenerate split noted
    alongside, and the checker must do exactly that: PASS_UNINFORMATIVE, accepted, informative
    False. An earlier version of this module blocked here, which quietly raised the threshold above
    the preregistered rule; that behaviour is retracted and this test pins the correction.
    """
    rep = evaluate_criteria(
        _ev(gains=tuple(f"g{i}" for i in range(5)), losses=("l0",), n=105,
            telemetry={"signal_firings": 84, "interventions_executed": 67,
                       "mechanism_verified": 67, "mechanism_requested": 84,
                       "mechanism_declined_with_reason": 1, "mechanism_refused_by_guard": 16}),
        dev={"arm_score": 0, "control_score": 0, "n": 24, "firings": 14},
        safety={"clears_added": 0, "information_losing_removes": 0, "verified_relocations": 67})

    assert rep.by_number(1).verdict == PASS
    assert rep.by_number(1).evidence["net"] == 4
    assert rep.by_number(2).verdict == PASS_UNINFORMATIVE
    assert rep.by_number(2).evidence["informative"] is False
    assert rep.by_number(3).verdict == PASS
    assert rep.by_number(4).verdict == PASS
    assert rep.accepted
    assert not rep.blocking
    # the weakness must survive serialisation -- it is what gets reported with the acceptance
    c2 = next(c for c in rep.to_dict()["criteria"] if c["number"] == 2)
    assert c2["evidence"]["informative"] is False


def test_the_withdrawn_61_denominator_would_now_be_caught():
    """Reporting 61 as the verified count against 84 requests leaves 6 unattributed, so the
    checker refuses instead of passing. The reconciliation's correction is enforced, not just
    documented."""
    r = check_mechanism({"signal_firings": 84, "interventions_executed": 61,
                         "mechanism_verified": 61, "mechanism_requested": 84,
                         "mechanism_declined_with_reason": 1,
                         "mechanism_refused_by_guard": 16})
    assert r.verdict == PENDING
    assert r.evidence["unattributed"] == 6


def test_reconciliation_record_agrees_with_the_numbers_pinned_here():
    """If the record and these tests ever disagree, one of them is stale."""
    p = REPO / "rounds" / "AUTONOMY" / "RELOCATION_CHAIN_RECONCILED.json"
    d = json.loads(p.read_text())
    part = d["reconciliation"]["exhaustive_partition_of_the_84"]
    assert part["A_verified_and_retry_landed_and_gate"] == 61
    assert part["B_verified_but_retry_hit_a_DIFFERENT_constraint"] == 6
    assert part["C_declined_by_the_lying_executor_guard"] == 1
    assert part["E_bound_guard_refused_before_any_action"] == 16
    assert part["total"] == 84
    assert d["measured_counts"]["write_verified"] == 67
    assert d["criterion_4_impact"]["verified_relocations"] == 67
