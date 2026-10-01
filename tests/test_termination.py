"""Stopping rules, at three levels -- and the line between a negative result and a non-result.

The defect that motivated this file was live in the cycle-2 driver:

    state = IMPROVED if (net > 0 and ok_denom) else NO_BENEFIT

A denominator MISMATCH -- two runs that scored different cases, i.e. no paired comparison at all --
was therefore recorded as NO_BENEFIT: a broken measurement banked as evidence against the candidate.
"""

from __future__ import annotations

import pytest

from anchoropt.learning import search_state as S
from anchoropt.learning.termination import (
    BUDGET_EXHAUSTED, EVALUATION_INVALID, GLOBAL_CONTINUE, GLOBAL_STOP_BUDGET,
    GLOBAL_STOP_EXHAUSTED, INSUFFICIENT_SUPPORT, LEDGER_STATUS, MIN_SUPPORT, NEGATIVE_RESULTS,
    NO_MEASURED_IMPROVEMENT, OUTCOMES, PROMOTED, SEARCH_EXHAUSTED, ResidualOutcome, Validity,
    channel_integrity, classify_residual, global_verdict,
)


class FakeSearch:
    def __init__(self, state=S.REALIZABLE_UNMEASURED, n=1):
        self.state = state
        self.candidates = ["c"] * n


def _cls(**kw):
    kw.setdefault("family", "f1")
    kw.setdefault("search_outcome", FakeSearch())
    kw.setdefault("support", 24)
    return classify_residual(**kw)


# ---- the five outcomes -------------------------------------------------------------------------

def test_search_exhausted_when_nothing_measurable_could_be_built():
    o = _cls(search_outcome=FakeSearch(S.SIGNAL_EXPANSION_EXHAUSTED, n=0))
    assert o.outcome == SEARCH_EXHAUSTED
    assert "expansion" in o.detail


def test_no_measured_improvement_when_candidates_were_evaluated_and_failed():
    o = _cls(evaluation={"candidates_evaluated": 3, "accepted": False}, validity=Validity())
    assert o.outcome == NO_MEASURED_IMPROVEMENT
    assert o.is_negative_result


def test_insufficient_support_below_the_declared_floor():
    o = _cls(support=MIN_SUPPORT - 1)
    assert o.outcome == INSUFFICIENT_SUPPORT
    assert not o.is_negative_result


def test_budget_exhausted_with_work_left_is_not_a_null():
    o = _cls(budget_remaining=False, novel_candidates_remaining=4)
    assert o.outcome == BUDGET_EXHAUSTED
    assert not o.is_negative_result


def test_promoted_requires_a_valid_evaluation():
    o = _cls(evaluation={"candidates_evaluated": 2, "accepted": True}, validity=Validity())
    assert o.outcome == PROMOTED


# ---- EVALUATION_INVALID is never a negative result ----------------------------------------------

@pytest.mark.parametrize("v,why", [
    (Validity(denominator_ok=False), "denominator"),
    (Validity(contamination_free=False), "contamination"),
    (Validity(channel_ok=False), "channel"),
])
def test_an_invalid_evaluation_is_never_a_negative_result(v, why):
    o = _cls(evaluation={"candidates_evaluated": 3, "accepted": False}, validity=v)
    assert o.outcome == EVALUATION_INVALID
    assert not o.is_negative_result, "an unbelievable measurement was banked as evidence"
    assert any(why in r for r in o.validity.reasons())


def test_the_exact_defect_a_broken_denominator_is_not_no_benefit():
    """THE regression. Previously: `IMPROVED if (net>0 and ok_denom) else NO_BENEFIT`."""
    o = _cls(evaluation={"candidates_evaluated": 1, "accepted": False},
             validity=Validity(denominator_ok=False))
    assert o.outcome != NO_MEASURED_IMPROVEMENT
    assert o.outcome == EVALUATION_INVALID


def test_validity_is_checked_before_acceptance():
    """A 'positive' result on an invalid round must not promote."""
    o = _cls(evaluation={"candidates_evaluated": 1, "accepted": True},
             validity=Validity(contamination_free=False))
    assert o.outcome == EVALUATION_INVALID


def test_an_invalid_evaluation_requeues_rather_than_settles():
    o = _cls(evaluation={"candidates_evaluated": 1, "accepted": False},
             validity=Validity(channel_ok=False))
    assert not o.settles_family
    assert o.ledger_status == "unresolved"


# ---- channel integrity -------------------------------------------------------------------------

def test_gains_with_zero_firings_fail_channel_integrity():
    ok, msg = channel_integrity(gains=5, gains_on_fired=0, firings=0)
    assert not ok and "ZERO firings" in msg


def test_no_threshold_is_enforced_by_default_but_the_fraction_is_reported():
    """11 of 22 gains off fired cases is a REAL measured round; a default threshold would silently
    reclassify it. So the fraction is reported and the call is left to whoever owns the claim."""
    ok, msg = channel_integrity(gains=22, gains_on_fired=11, firings=49)
    assert ok
    assert "11/22" in msg and "50%" in msg
    assert "not enforced" in msg


def test_a_declared_threshold_is_enforced_when_supplied():
    ok, _ = channel_integrity(gains=22, gains_on_fired=11, firings=49,
                              min_fraction_on_fired=0.75)
    assert not ok
    ok2, _ = channel_integrity(gains=24, gains_on_fired=20, firings=49,
                               min_fraction_on_fired=0.75)
    assert ok2


# ---- level 3: global ---------------------------------------------------------------------------

def test_a_single_rejected_arm_never_terminates_evolution():
    """The failure mode this level exists to prevent -- a rejection is the most visible outcome."""
    one = _cls(evaluation={"candidates_evaluated": 1, "accepted": False}, validity=Validity())
    g = global_verdict([one], families_eligible=4)
    assert g.verdict == GLOBAL_CONTINUE
    assert not g.pass_complete


def test_a_promotion_always_continues():
    o = _cls(evaluation={"candidates_evaluated": 1, "accepted": True}, validity=Validity())
    g = global_verdict([o], families_eligible=1)
    assert g.verdict == GLOBAL_CONTINUE
    assert "regenerate" in g.detail


def test_stopping_requires_a_complete_pass_with_no_promotion_and_nothing_novel():
    settled = [_cls(family=f, evaluation={"candidates_evaluated": 1, "accepted": False},
                    validity=Validity()) for f in ("a", "b", "c")]
    g = global_verdict(settled, families_eligible=3)
    assert g.verdict == GLOBAL_STOP_EXHAUSTED
    assert g.pass_complete


def test_remaining_novel_candidates_block_stopping_even_on_a_complete_pass():
    settled = [_cls(family="a", evaluation={"candidates_evaluated": 1, "accepted": False},
                    validity=Validity(), novel_candidates_remaining=2)]
    g = global_verdict(settled, families_eligible=1)
    assert g.verdict == GLOBAL_CONTINUE
    assert g.novel_candidates_remaining == 2


def test_an_invalid_family_keeps_the_loop_alive():
    outs = [_cls(family="a", evaluation={"candidates_evaluated": 1, "accepted": False},
                 validity=Validity()),
            _cls(family="b", evaluation={"candidates_evaluated": 1, "accepted": False},
                 validity=Validity(denominator_ok=False))]
    g = global_verdict(outs, families_eligible=2)
    assert g.verdict == GLOBAL_CONTINUE
    assert g.invalid == ("b",)


def test_budget_stop_is_reported_as_a_budget_stop_not_an_exhausted_search():
    o = _cls(budget_remaining=False, novel_candidates_remaining=3)
    g = global_verdict([o], budget_remaining=False, families_eligible=1)
    assert g.verdict == GLOBAL_STOP_BUDGET
    assert "NOT a null result" in g.detail


# ---- bookkeeping -------------------------------------------------------------------------------

def test_every_outcome_maps_to_an_existing_ledger_status():
    from anchoropt.learning import evidence_ledger as L
    known = set(L._STATUSES)
    for outcome, status in LEDGER_STATUS.items():
        assert status in known, f"{outcome} -> {status!r} is not a ledger status"


def test_exactly_two_outcomes_are_negative_results():
    assert set(NEGATIVE_RESULTS) == {NO_MEASURED_IMPROVEMENT, SEARCH_EXHAUSTED}
    assert set(NEGATIVE_RESULTS) <= set(OUTCOMES)


def test_the_frozen_search_states_are_not_redefined_here():
    """NO_BENEFIT stays search-local; NO_MEASURED_IMPROVEMENT is the family verdict."""
    assert S.NO_BENEFIT not in OUTCOMES
    assert NO_MEASURED_IMPROVEMENT not in S.STATES


def test_outcomes_serialize_for_a_round_record():
    o = _cls(evaluation={"candidates_evaluated": 1, "accepted": False}, validity=Validity())
    d = o.as_dict()
    for k in ("outcome", "is_negative_result", "settles_family", "ledger_status", "validity"):
        assert k in d
