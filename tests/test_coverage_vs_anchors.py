"""The coverage matcher must not be confidently wrong in either direction.

Pins the defect a first version actually had: a shared-token matcher reported A3 NOT RECOVERED while
the mechanism sat in the library (`duplicate` vs `redundant_proposed_write` share no token), AND
credited that same candidate to A8 (both share `capacity`/`proposed`). One record, two wrong answers.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchoropt.learning.candidate_library import (            # noqa: E402
    ACCEPTED, PENDING_VALIDATION, CandidateLibrary, CandidateRecord, EvaluationContext, Measurement)
from anchoropt.learning.golden_registry import ControllerIdentity   # noqa: E402

_spec = importlib.util.spec_from_file_location("cva", ROOT / "scripts" / "coverage_vs_anchors.py")
cva = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cva)


def row(anchor):
    return next(r for r in cva.anchor_rows() if r["anchor"] == anchor)


def cand(name, signal, boundary="post_generation_pre_exec", action="suppress",
         operator="suppress", state=PENDING_VALIDATION, cap="cancel_proposed"):
    return CandidateRecord(
        name=name,
        identity=ControllerIdentity(boundary=boundary, signal=signal, action=action,
                                    operator=operator, capability_id=cap, phase="prereq"),
        spec={"signal": signal}, state=state)


def test_all_eight_anchors_are_enumerated_with_a_family_label():
    rows = cva.anchor_rows()
    assert [r["anchor"] for r in rows] == ["A1", "A2", "A3", "A4", "A5", "A7", "A8", "A9"]
    assert all(r["family"] for r in rows), "every anchor needs a family label for the report"


def test_A3_matches_redundant_proposed_write_despite_sharing_no_word_with_duplicate():
    """The false NEGATIVE half of the original defect."""
    lib = CandidateLibrary([cand("r1_redundant", "redundant_proposed_write")])
    assert [h.name for h in cva.match(row("A3"), lib)] == ["r1_redundant"]


def test_A8_does_NOT_claim_the_duplicate_write_candidate():
    """The false POSITIVE half: A8 is clear-prevention, not duplicate suppression."""
    lib = CandidateLibrary([cand("r1_redundant", "redundant_proposed_write")])
    assert cva.match(row("A8"), lib) == []


def test_A8_trigger_matches_only_its_own_condition():
    lib = CandidateLibrary([cand("r1_clear", "clear_proposed_at_capacity")])
    assert [h.name for h in cva.match(row("A8"), lib)] == ["r1_clear"]


def test_the_two_suppression_candidates_map_ONE_TO_ONE_onto_A3_and_A8():
    """Both live at the same boundary with the same action, so only the condition separates them."""
    lib = CandidateLibrary([cand("r1_redundant", "redundant_proposed_write"),
                            cand("r1_clear", "clear_proposed_at_capacity")])
    assert [h.name for h in cva.match(row("A3"), lib)] == ["r1_redundant"]
    assert [h.name for h in cva.match(row("A8"), lib)] == ["r1_clear"]


# ---------------------------------------------------------------------------------------------------
# MECHANISM COMPLETENESS -- the trigger is not the mechanism
# ---------------------------------------------------------------------------------------------------

def test_a_cancel_only_candidate_does_NOT_fully_recover_A8():
    """docs/ANCHORS.md: A8 suppresses the clear, EVICTS a redundant copy, VERIFIES one remains and
    RETRIES the blocked write -- "reuses A5's validated action with a genuinely new trigger". A
    cancel-only candidate prevents the destruction and performs none of the repair."""
    gap = cva.mechanism_gap("A8", cand("r1_clear", "clear_proposed_at_capacity"))
    assert set(gap) == {"evict_redundant_copy", "verify_copy_remains", "retry_blocked_write"}


def test_a_cancel_only_candidate_DOES_fully_recover_A3():
    """A3 is cancel-only BY DESIGN -- the asymmetry that makes the A8 gap meaningful rather than a
    blanket penalty on suppression."""
    assert cva.mechanism_gap("A3", cand("r1_redundant", "redundant_proposed_write")) == ()


def test_a_candidate_naming_the_repair_steps_closes_the_A8_gap():
    """The gap must be closable by a better candidate, or it is not a measurement."""
    full = cand("r1_clear_full", "clear_proposed_at_capacity",
                cap="suppress_clear_then_evict_redundant_copy_verify_and_retry_blocked_write")
    assert cva.mechanism_gap("A8", full) == ()


def test_the_cancel_only_A8_candidate_keeps_its_mechanism_GAP():
    """The cancel-only arm must never read as a full A8 match, however many A8 rounds have run.

    This test previously asserted that NO A8 candidate was a full match. That premise expired the
    moment R8 measured `dedup_clear_conditional_suppression`, which names evict/verify/retry and so
    legitimately closes the gap -- the capability was BUILT in the interim. Asserting the old premise
    would now be asserting that real progress had not happened. What must stay true is the asymmetry
    that mattered: the cancel-only candidate is a weaker mechanism and must keep its gap.
    """
    lib = CandidateLibrary.load(ROOT / "rounds" / "GOLDEN" / "candidates.json")
    if len(lib) == 0:
        import pytest
        pytest.skip("candidate library not seeded in this checkout")
    hits = cva.match(row("A8"), lib)
    assert hits, "A8's trigger should be present in the library"
    cancel_only = [h for h in hits if "cancel_proposed" in h.identity.capability_id]
    assert cancel_only, f"the cancel-only candidate should still be present: {[h.name for h in hits]}"
    for h in cancel_only:
        gap = cva.mechanism_gap("A8", h)
        assert set(gap) == {"evict_redundant_copy", "verify_copy_remains", "retry_blocked_write"}, gap


def test_a_FULL_A8_mechanism_in_the_library_closes_the_gap():
    """The complement: a candidate naming the repair steps must be recognised as a full match, or the
    coverage report would under-credit a capability that was actually built."""
    lib = CandidateLibrary.load(ROOT / "rounds" / "GOLDEN" / "candidates.json")
    if len(lib) == 0:
        import pytest
        pytest.skip("candidate library not seeded in this checkout")
    full = [h for h in cva.match(row("A8"), lib) if not cva.mechanism_gap("A8", h)]
    if not full:
        import pytest
        pytest.skip("no full-mechanism A8 candidate recorded yet in this checkout")
    # It is a full MECHANISM match; that is independent of whether it was ACCEPTED. R8 measured it
    # MEASURED_NEGATIVE, and coverage tracks expressibility, not acceptance.
    for h in full:
        assert "evict" in h.identity.capability_id or "evict" in str(h.spec)


def test_the_recorded_backend_scope_is_preserved_for_A8():
    """A8's record states every observed event is kv; A3's are collisions against keys A1 injected."""
    assert cva.ANCHOR_BACKEND.get("A8") == "kv"


def test_a_different_ACTION_at_the_same_boundary_is_not_a_match():
    lib = CandidateLibrary([cand("r", "redundant_proposed_write", action="reprompt",
                                 operator="reprompt")])
    assert cva.match(row("A3"), lib) == []


def test_a_different_BOUNDARY_with_the_same_condition_is_not_a_match():
    lib = CandidateLibrary([cand("r", "clear_proposed_at_capacity", boundary="post_execution")])
    assert cva.match(row("A8"), lib) == []


def test_A1_matches_the_kv_capacity_spelling():
    lib = CandidateLibrary([cand("kv_reloc", "container_at_capacity", boundary="post_execution",
                                 action="reroute", operator="transform", cap="relocate_then_retry")])
    assert [h.name for h in cva.match(row("A1"), lib)] == ["kv_reloc"]


def test_A9_is_not_recovered_by_an_empty_library():
    assert cva.match(row("A9"), CandidateLibrary()) == []


def test_an_empty_signal_never_matches_anything():
    """A record with no signal must not be silently credited to an anchor."""
    lib = CandidateLibrary([cand("nosig", "")])
    assert all(cva.match(row(a), lib) == [] for a in ("A1", "A3", "A8"))


def test_every_anchor_has_a_declared_condition_signal_set():
    """A missing entry would silently fall back to substring matching only."""
    for r in cva.anchor_rows():
        assert cva.CONDITION_SIGNALS.get(r["anchor"]), f"{r['anchor']} has no declared signals"


def test_the_real_library_reports_three_accepted_in_the_incumbent(tmp_path):
    """End-to-end against the shipped records, so the report cannot drift from the library."""
    lib = CandidateLibrary.load(ROOT / "rounds" / "GOLDEN" / "candidates.json")
    if len(lib) == 0:
        import pytest
        pytest.skip("candidate library not seeded in this checkout")
    in_incumbent = [r["anchor"] for r in cva.anchor_rows()
                    if any(h.state == ACCEPTED for h in cva.match(r, lib))]
    assert in_incumbent == ["A1", "A4", "A7"]
