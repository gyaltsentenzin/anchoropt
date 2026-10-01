"""Unresolved evidence must never be recorded as a measured negative result.

Benchmark-independent, per the core-fix rule: every generic core change carries a regression test that
does not mention a benchmark. The distinction pinned here is the one the whole verdict lattice exists
for -- a criterion that is PENDING was never measured, so the candidate is UNRESOLVED, not refuted.
Banking it as NO_MEASURED_IMPROVEMENT would let `excluded_cells` permanently retire a cell whose
question was never answered.
"""

from __future__ import annotations

# ---------------------------------------------------------------- missing evidence is not a result

def test_unresolved_acceptance_evidence_is_EVALUATION_INVALID_not_a_negative_result():
    """A criterion that is PENDING was never measured, so the candidate is UNRESOLVED.

    Benchmark-independent. Recording it as NO_MEASURED_IMPROVEMENT would bank a negative result
    against a candidate nobody evaluated, and `excluded_cells` would then permanently exclude the
    cell -- the search would stop asking a question it never answered.
    """
    from anchoropt.learning.termination import Validity, classify_residual

    v = Validity(evidence_complete=False,
                 missing_evidence=("C4 safety_three_counters: no safety telemetry",))
    term = classify_residual(family="f", search_outcome=None, support=20,
                             evaluation={"candidates_evaluated": 3, "accepted": False},
                             validity=v, budget_remaining=True)
    assert term.outcome == "EVALUATION_INVALID"
    assert not term.is_negative_result


def test_evidence_complete_defaults_true_so_existing_rounds_are_unaffected():
    from anchoropt.learning.termination import Validity

    assert Validity().ok
    assert Validity(evidence_complete=False).ok is False


def test_missing_evidence_travels_with_the_verdict():
    from anchoropt.learning.termination import Validity

    v = Validity(evidence_complete=False, missing_evidence=("C3 mechanism: no telemetry",))
    assert any("UNRESOLVED rather than refuted" in r for r in v.reasons())
    assert "C3 mechanism: no telemetry" in v.as_dict()["missing_evidence"]
