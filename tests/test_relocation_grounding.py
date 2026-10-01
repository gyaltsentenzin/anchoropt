"""A slot-count constraint must ground a RELOCATION, and a character constraint must not.

The measured failure this prevents: the kv round emitted
`post_execution/container_at_capacity/transform:reduce_value_for_kv`. Shortening a value cannot free a
slot, so that arm would have measured a null indistinguishable from "the mechanism does not transfer".
Meanwhile the teacher independently asked for relocation -- "capacity pressure is relieved by moving
content rather than by deleting it" -- and it was recorded UNREALIZABLE because no action could express it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

import bfcl_runtime as R                                              # noqa: E402
from anchoropt.anchor import IncisionPoint                            # noqa: E402
from anchoropt.learning.action_contract import CONTRACTS, Operator     # noqa: E402

POST = IncisionPoint.POST_EXECUTION


@pytest.fixture(autouse=True)
def _kv_cell(monkeypatch):
    """Ground only the mined backend: a grounding naming another is an alias, not a distinct arm."""
    monkeypatch.setenv("ANCHOROPT_CELL", "kv")


def _ok(groundings):
    return [g for g in groundings if not g.get("infeasible")]


# -- the refutation, now mechanical ------------------------------------------------------------

def test_a_slot_constraint_grounds_NO_reduce_variant():
    g = _ok(R.ground_transforms("container_at_capacity", POST))
    assert not any("reduce_" in str(x.get("variant")) for x in g), g


def test_a_slot_constraint_grounds_a_RELOCATION_instead():
    """The feasible sibling is offered rather than a bare refusal."""
    g = _ok(R.ground_transforms("container_at_capacity", POST))
    assert len(g) == 1
    assert g[0]["variant"] == "relocate_entry_for_kv"
    assert g[0]["eta"]["operator"] == "relocate_entry_preserving_information"


def test_a_character_constraint_still_grounds_the_reduction():
    """The accepted rec_sum mechanism's sibling on kv must not be broken by this change."""
    g = _ok(R.ground_transforms("append_would_exceed_cap", POST))
    assert len(g) == 1 and g[0]["variant"] == "reduce_value_for_kv"


def test_a_character_constraint_grounds_NO_relocation():
    """The mirror image: relocation frees slots and cannot shorten anything."""
    g = R.ground_capacity_relocations("append_would_exceed_cap", POST)
    assert not _ok(g)
    assert g and g[0]["infeasible"] and "chars_in_blob" in g[0]["detail"]


# -- the grounding must satisfy the contract it claims -----------------------------------------

def test_the_relocation_grounding_satisfies_the_TRANSFORM_contract():
    """Missing a required key means the arm cannot instantiate -- a ghost grounding."""
    g = _ok(R.ground_transforms("container_at_capacity", POST))[0]
    for key in CONTRACTS[Operator.TRANSFORM].required:
        assert key in g["eta"], f"TRANSFORM requires {key!r}"


def test_the_relocation_carries_its_capability_identity():
    """This cell has two executors; without the identity the wrong one can run."""
    g = _ok(R.ground_transforms("container_at_capacity", POST))[0]
    assert (g["grounding"]["capability_id"]
            == "relocate_entry_preserving_information_then_retry")


def test_the_relocation_declares_verify_before_remove():
    g = _ok(R.ground_transforms("container_at_capacity", POST))[0]
    pres = g["eta"]["preservation"].lower()
    assert "before it is removed" in pres
    assert "nothing is removed" in pres


def test_the_relocation_retries_the_ORIGINAL_call_not_a_substitute():
    """Substitution writes the new fact elsewhere; relocation makes room and retries verbatim."""
    g = _ok(R.ground_transforms("container_at_capacity", POST))[0]
    assert g["eta"]["retry_semantics"] == "retry_original_verbatim_after_capacity_freed"


# -- scope ---------------------------------------------------------------------------------------

def test_relocation_grounds_only_at_post_execution():
    """The refusal is observable only after the write was attempted."""
    assert R.ground_capacity_relocations("container_at_capacity",
                                        IncisionPoint.POST_GENERATION_PRE_EXEC) == []


def test_a_signal_with_no_declared_constraint_grounds_nothing():
    assert R.ground_capacity_relocations("no_tool_call_at_all", POST) == []


def test_only_the_mined_backend_is_grounded(monkeypatch):
    """One arm per mechanism: an alias for another backend is a 3x GPU bill for no information."""
    monkeypatch.setenv("ANCHOROPT_CELL", "vector")
    g = _ok(R.ground_capacity_relocations("container_at_capacity", POST))
    assert [x["variant"] for x in g] == ["relocate_entry_for_vector"]


def test_an_infeasible_grounding_carries_an_actionable_remedy():
    g = R.ground_capacity_relocations("append_would_exceed_cap", POST)[0]
    assert g["remedy"] and "occupied_slots" in g["detail"]
