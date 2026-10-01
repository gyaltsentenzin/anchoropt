"""The BFCL adapter's own constraint declarations, checked against the backends' real units."""

from __future__ import annotations

from anchoropt.learning.constraint_feasibility import FEASIBLE, INFEASIBLE
from benchmarks.bfcl_v4.bfcl_constraints import (
    CHARS_IN_BLOB, OCCUPIED_SLOTS, bfcl_constraint_contract, constraint_for_signal)


def test_capacity_is_three_constraints_not_one():
    c = bfcl_constraint_contract()
    units = {x.name: x.quantity for x in c.constraints}
    assert units["blob_would_overflow"] == CHARS_IN_BLOB
    assert units["no_capacity"] == OCCUPIED_SLOTS
    assert units["entry_too_long"] == "chars_in_entry"
    # Three distinct quantities: collapsing any two is what made the kv arm look feasible.
    assert len(set(units.values())) == 3


def test_the_kv_arm_that_was_emitted_is_now_refused_mechanically():
    """post_execution/container_at_capacity/transform:reduce_* -- infeasible, with a named remedy."""
    c = bfcl_constraint_contract()
    cons = constraint_for_signal("container_at_capacity")
    assert cons == "no_capacity"
    v = c.check("reduce_preserving_facts", cons)
    assert v.status == INFEASIBLE, v
    assert "evict_entry" in c.feasible_operators(cons)


def test_the_accepted_rec_sum_recovery_stays_feasible():
    """Regression guard: the declarations must not refute the accepted controller."""
    c = bfcl_constraint_contract()
    cons = constraint_for_signal("append_would_exceed_cap")
    assert cons == "blob_would_overflow"
    assert c.check("reduce_preserving_facts", cons).status == FEASIBLE


def test_substituting_a_destination_reduces_nothing_by_itself():
    """Rerouting into the container that is already full relieves no constraint."""
    c = bfcl_constraint_contract()
    assert c.check("substitute_destination", "no_capacity").status == INFEASIBLE


def test_an_unmapped_signal_is_explicit_not_assumed():
    assert constraint_for_signal("no_tool_call_at_all") == ""
