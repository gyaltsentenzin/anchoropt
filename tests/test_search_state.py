"""The block-coordinate transition order IS the algorithmic claim, so it is tested directly.

    at a fixed WHERE:  optimize existing WHAT x HOW first
                       -> realizable but no improvement: exhaust HOW/eta
                       -> then expand WHAT
                       -> then retry HOW at the SAME boundary
    move WHERE earlier ONLY when the boundary is unrepairable or its WHAT/HOW search is exhausted

Also pins the two things that make a generic vocabulary honest: the table is total (no state can
leave the optimizer without a next move), and an unrecognised upstream code stays VISIBLE rather than
being silently bucketed.
"""

from __future__ import annotations

import pytest

from anchoropt.learning.search_state import (
    ACTION_UNAVAILABLE, ALL_CELLS_ALREADY_DECIDED, BOUNDARY_EXHAUSTED, BOUNDARY_NOT_REPAIRABLE,
    COORDINATES, DONE,
    ETA_UNSUPPORTED, HOW, IMPROVED, NEXT_COORDINATE, NO_BENEFIT, NO_CANDIDATE, PRIMITIVE_MISSING,
    PROMOTE, REALIZABLE_UNMEASURED, SIGNAL_BLOCKED, SIGNAL_EXPANDED, SIGNAL_EXPANSION_EXHAUSTED,
    STATES, WHAT, WHERE, Attempt, certificate_from, classify, is_terminal, next_coordinate,
)


def test_the_transition_table_is_total_over_states():
    """No state may leave the optimizer with no next move."""
    assert set(NEXT_COORDINATE) == set(STATES)
    for state in STATES:
        assert next_coordinate(state) in COORDINATES


# ---- the claimed ordering, one assertion per clause ---------------------------------------------

def test_signal_blocked_expands_what_and_does_NOT_move_the_boundary():
    """Phi cannot express the condition HERE -> widen the representation at this boundary."""
    assert next_coordinate(SIGNAL_BLOCKED) == WHAT


def test_after_expanding_what_the_search_retries_how_at_the_SAME_boundary():
    """The point of expanding is to make an intervention buildable HERE, so HOW is retried here."""
    assert next_coordinate(SIGNAL_EXPANDED) == HOW


def test_no_benefit_exhausts_how_before_anything_else():
    """Measured and it did not help -> other eta/action choices at this boundary come first."""
    assert next_coordinate(NO_BENEFIT) == HOW


@pytest.mark.parametrize("state", [ACTION_UNAVAILABLE, ETA_UNSUPPORTED, PRIMITIVE_MISSING])
def test_a_how_failure_tries_another_how(state):
    assert next_coordinate(state) == HOW


def test_nothing_buildable_under_current_phi_widens_phi_before_leaving_the_boundary():
    """NO_CANDIDATE is a WHAT problem: the action space was never the binding constraint."""
    assert next_coordinate(NO_CANDIDATE) == WHAT


@pytest.mark.parametrize("state", [BOUNDARY_NOT_REPAIRABLE, BOUNDARY_EXHAUSTED,
                                   SIGNAL_EXPANSION_EXHAUSTED])
def test_where_moves_only_when_the_boundary_is_unrepairable_or_exhausted(state):
    assert next_coordinate(state) == WHERE


def test_exactly_these_states_move_the_boundary():
    """The ordering claim is as much about what does NOT move WHERE as what does.

    Three of the four are about THIS boundary being spent: unrepairable, exhausted, or out of
    representation. `ALL_CELLS_ALREADY_DECIDED` is the fourth and it is about the SEARCH HISTORY --
    every arm here names a controller identity an earlier round already decided. It moves WHERE for
    the same reason the others do (there is nothing left to learn here) but it must NOT be confused
    with them: nothing was MEASURED at this boundary, so it may not route to WHAT. Routing it to WHAT
    would expand the representation because a previous round SUCCEEDED, which is expansion off a
    verdict nobody produced -- and in the restart loop that rebuilt the same arms forever.
    """
    movers = {s for s in STATES if next_coordinate(s) == WHERE}
    assert movers == {BOUNDARY_NOT_REPAIRABLE, BOUNDARY_EXHAUSTED, SIGNAL_EXPANSION_EXHAUSTED,
                      ALL_CELLS_ALREADY_DECIDED}


def test_an_already_decided_boundary_does_NOT_widen_phi():
    """The distinction the state exists to preserve: history is not a representation limit."""
    assert next_coordinate(ALL_CELLS_ALREADY_DECIDED) == WHERE
    assert next_coordinate(ALL_CELLS_ALREADY_DECIDED) != WHAT


def test_improvement_leaves_the_search_to_promote():
    assert next_coordinate(IMPROVED) == PROMOTE
    assert is_terminal(IMPROVED)


def test_unmeasured_is_terminal_and_is_NOT_no_benefit():
    """A controller built but never evaluated is a STRUCTURAL result, not a failure to improve.

    Collapsing these would report a measurement that never happened -- the same error as believing a
    null from a channel that was never live.
    """
    assert next_coordinate(REALIZABLE_UNMEASURED) == DONE
    assert is_terminal(REALIZABLE_UNMEASURED)
    assert REALIZABLE_UNMEASURED != NO_BENEFIT
    assert not is_terminal(NO_BENEFIT)


def test_an_unknown_state_is_conservative():
    assert next_coordinate("SOMETHING_NEW") == DONE


# ---- feasibility certificates -------------------------------------------------------------------

def test_an_upstream_code_is_classified_but_RETAINED():
    """The generic state is for dispatch; the specific code is what names the fact."""
    c = classify("SUPPRESS_INFEASIBLE", boundary="post_generation_pre_exec", signal="phi",
                 action="suppress", missing=("suppressed_operation",), detail="no grounding")
    assert c.state == ETA_UNSUPPORTED
    assert c.reason_code == "SUPPRESS_INFEASIBLE"      # not thrown away
    assert c.missing == ("suppressed_operation",)
    assert c.classified


def test_an_unrecognised_code_is_visible_not_silently_bucketed():
    c = classify("a_code_nobody_mapped")
    assert not c.classified, "a new upstream code must be visible as unclassified"
    assert c.reason_code == "a_code_nobody_mapped"


def test_signal_unobservable_is_a_WHAT_problem_not_a_HOW_problem():
    """Misclassifying this would send the optimizer hunting for actions when Phi is the gap."""
    assert classify("signal_not_observable_at_boundary").state == SIGNAL_BLOCKED
    assert next_coordinate(classify("signal_not_observable_at_boundary").state) == WHAT


def test_certificate_from_a_rejected_arm_shaped_record():
    class Rej:
        action = "suppress"
        reason_code = "not_materializable_by_host_executor"
        missing = ("eta",)
        detail = "no executor"
    c = certificate_from(Rej(), boundary="post_execution", signal="phi")
    assert c.state == PRIMITIVE_MISSING
    assert c.action == "suppress" and c.signal == "phi"


# ---- the attempt record -------------------------------------------------------------------------

def test_an_attempt_derives_the_coordinate_it_changed():
    a = Attempt(boundary="post_execution", state=SIGNAL_BLOCKED)
    assert a.coordinate_changed == WHAT
    assert a.as_dict()["coordinate_changed"] == WHAT
