"""An operator must be feasible for the CONSTRAINT, not merely runnable at the boundary.

The KV case is pinned here as a permanent regression: a correct signal (84 real refusals) carrying an
operator that cannot change the constrained quantity. It reached arm emission once; it must never
again.
"""

from __future__ import annotations

import pytest

from anchoropt.learning.constraint_feasibility import (
    FEASIBLE, INFEASIBLE, UNKNOWN, Constraint, ConstraintContract, OperatorEffect)


def test_reducing_chars_cannot_relieve_a_slot_count_constraint():
    """THE KV REFUTATION, mechanized."""
    c = ConstraintContract(
        constraints=[Constraint("no_capacity", quantity="occupied_slots", unit="entries")],
        effects=[OperatorEffect("reduce_preserving_facts", reduces=("chars_in_blob",))])
    v = c.check("reduce_preserving_facts", "no_capacity")
    assert v.status == INFEASIBLE, v
    assert not v.ok
    # The verdict must name both quantities and give a remedy -- a bare False is what let this ship.
    assert "occupied_slots" in v.detail and "chars_in_blob" in v.detail
    assert "entries" in v.detail
    assert v.remedy


def test_reducing_chars_does_relieve_a_char_constraint():
    """The accepted rec_sum recovery: same operator, different constraint, feasible."""
    c = ConstraintContract(
        constraints=[Constraint("blob_would_overflow", quantity="chars_in_blob", unit="characters")],
        effects=[OperatorEffect("reduce_preserving_facts", reduces=("chars_in_blob",))])
    v = c.check("reduce_preserving_facts", "blob_would_overflow")
    assert v.status == FEASIBLE and v.ok, v


def test_eviction_is_the_operator_that_relieves_a_slot_constraint():
    c = ConstraintContract(
        constraints=[Constraint("no_capacity", quantity="occupied_slots")],
        effects=[OperatorEffect("reduce_preserving_facts", reduces=("chars_in_blob",)),
                 OperatorEffect("evict_entry", reduces=("occupied_slots",))])
    assert c.check("evict_entry", "no_capacity").status == FEASIBLE
    assert c.feasible_operators("no_capacity") == ("evict_entry",)


@pytest.mark.parametrize("cons,eff,why", [
    (False, True, "constraint is not declared"),
    (True, False, "declares no effect"),
    (False, False, "neither"),
])
def test_an_undeclared_side_is_UNKNOWN_and_never_a_silent_pass(cons, eff, why):
    """UNKNOWN must be distinguishable from FEASIBLE: treating it as a pass reintroduces the defect."""
    c = ConstraintContract(
        constraints=[Constraint("k", quantity="q")] if cons else [],
        effects=[OperatorEffect("op", reduces=("q",))] if eff else [])
    v = c.check("op", "k")
    assert v.status == UNKNOWN and not v.ok, v
    assert why in v.detail
    assert v.remedy, "an UNKNOWN must say which declaration is missing"


def test_core_module_names_no_benchmark_concept():
    """This contract must stay portable: no memory, slot, archival or character vocabulary."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "anchoropt" / "learning" / "constraint_feasibility.py").read_text()
    # Identifiers/literals, not prose: the module's own docstring legitimately explains the case.
    for bad in ("archival_memory", "core_memory", "rec_sum", "memory_kv", "bfcl"):
        assert bad not in src, f"core names the benchmark identifier {bad!r}"
