"""Two controllers at ONE cell must each reach their OWN executor.

`post_execution/reroute` now has two named executors -- a character reduction and a slot relocation.
"Each works alone" is not evidence they work together: the host resolves executors per cell, so the
second controller can be routed to the first's mechanism, or can consume the firing and leave the first
inert. The registry emits a composition certificate for exactly this shape; this file measures it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "benchmarks" / "bfcl_v4", REPO / "scripts"):
    sys.path.insert(0, str(p))

from anchoropt.learning.golden_registry import ControllerIdentity, GoldenRegistry   # noqa: E402
from bfcl_regression_host import (_ensure_declared_signal_seam,                     # noqa: E402
                                  _load_installer)

REDUCE = "reduce_payload_preserving_facts_and_replace"
RELOC = "relocate_entry_preserving_information_then_retry"

SPECS = {
    "rec_sum_reduction": {
        "name": "append_would_exceed_cap", "locus": "post_execution", "action": "reroute",
        "operator": "transform", "capability_id": REDUCE, "phase": "prereq",
        "predicate": {"declared_signal": "append_would_exceed_cap", "params": {}}},
    "kv_relocation": {
        "name": "container_at_capacity", "locus": "post_execution", "action": "reroute",
        "operator": "transform", "capability_id": RELOC, "phase": "prereq",
        "predicate": {"declared_signal": "container_at_capacity", "params": {}}},
}

SLOT_STATE = {"error_kind": "no_capacity", "proposes_write": True}
CHAR_STATE = {"error_kind": "blob_would_overflow", "proposes_write": True}


@pytest.fixture(scope="module")
def controllers():
    _ensure_declared_signal_seam()
    inst = _load_installer()
    return {k: inst.SpecPredicate(v) for k, v in SPECS.items()}


def _admits(want, spec):
    """The identity gate both patches apply, in logic."""
    cap = str(spec.get("capability_id") or "")
    op = str(spec.get("operator") or spec.get("action") or "")
    return cap == want and op in ("transform", "substitute", "reroute")


def test_exactly_one_controller_fires_per_state(controllers):
    """Neither predicate consumes the other's condition."""
    assert [k for k, c in controllers.items() if c.fires_on(SLOT_STATE)] == ["kv_relocation"]
    assert [k for k, c in controllers.items() if c.fires_on(CHAR_STATE)] == ["rec_sum_reduction"]


def test_each_firing_is_admitted_by_exactly_one_executor(controllers):
    for state, expect_cap, expect_name in ((SLOT_STATE, RELOC, "kv_relocation"),
                                           (CHAR_STATE, REDUCE, "rec_sum_reduction")):
        fired = [k for k, c in controllers.items() if c.fires_on(state)]
        assert fired == [expect_name]
        admitted = [k for k in fired if _admits(expect_cap, SPECS[k])]
        other = REDUCE if expect_cap is RELOC else RELOC
        assert admitted == [expect_name]
        assert [k for k in fired if _admits(other, SPECS[k])] == [], "routed to the wrong executor"


def test_installing_the_relocation_does_not_make_the_accepted_one_inert(controllers):
    """The shadowing failure mode: the new controller present, the old one silent."""
    assert controllers["rec_sum_reduction"].fires_on(CHAR_STATE) is True


def test_the_registry_FLAGS_this_shared_cell_rather_than_assuming_it_is_fine():
    """A same-cell/different-capability install must not pass quietly."""
    reg = GoldenRegistry.load(REPO / "rounds" / "GOLDEN" / "registry.json")
    conflicts = reg.composition_conflicts(ControllerIdentity(
        "post_execution", "container_at_capacity", "reroute", "transform", RELOC, "prereq"))
    assert conflicts, "the shared cell must raise a certificate"
    assert any("different capability_id" in c and "smoke-test BOTH" in c for c in conflicts)


def test_a_spec_with_no_capability_id_is_admitted_by_NEITHER_executor():
    """An unnamed controller at a two-executor cell must not pick one by luck."""
    bare = {**SPECS["kv_relocation"], "capability_id": ""}
    assert not _admits(RELOC, bare) and not _admits(REDUCE, bare)
