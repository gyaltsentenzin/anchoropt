"""A capability must claim only signals whose CONSTRAINT its operator can relieve.

The defect this locks out: `reduce_payload_preserving_facts_and_replace` declared
`container_at_capacity` and `container_slots_exhausted`. Those are ENTRIES-count refusals, and shortening
a payload frees zero slots -- so an autonomously grounded arm reached emission and had to be refuted at
feasibility. The overclaim in the capability table is what let it ground.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

import bfcl_runtime as R                                                   # noqa: E402
from anchoropt.learning.constraint_feasibility import FEASIBLE             # noqa: E402
from bfcl_constraints import (bfcl_constraint_contract,                    # noqa: E402
                              constraint_for_signal)

# Which operator each NAMED capacity executor actually applies. Only these two are checked: the
# cell-default entry is a signal-agnostic reroute whose destination is synthesized from live state.
CAPABILITY_OPERATOR = {
    "reduce_payload_preserving_facts_and_replace": "reduce_preserving_facts",
    "relocate_entry_preserving_information_then_retry": "relocate_entry",
}


def _named_capacity_caps():
    return [c for c in R._CAPABILITIES if (c.capability_id or "") in CAPABILITY_OPERATOR]


def test_both_named_capacity_executors_are_declared():
    assert {c.capability_id for c in _named_capacity_caps()} == set(CAPABILITY_OPERATOR)


def test_every_claimed_signal_is_feasible_for_that_capability():
    """The general rule, applied to the whole table rather than to one remembered case."""
    contract = bfcl_constraint_contract()
    bad = []
    for cap in _named_capacity_caps():
        op = CAPABILITY_OPERATOR[cap.capability_id]
        for sig in cap.signals:
            cons = constraint_for_signal(sig)
            if not cons:
                continue                      # not a capacity signal; nothing to check here
            v = contract.check(op, cons)
            if v.status != FEASIBLE:
                bad.append(f"{cap.capability_id} claims {sig!r} ({cons}) but {op!r} is {v.status}: "
                           f"{v.detail}")
    assert not bad, "\n".join(bad)


def test_the_reduction_executor_no_longer_claims_a_slot_count_signal():
    """The specific regression, pinned by name."""
    cap = next(c for c in _named_capacity_caps()
               if c.capability_id == "reduce_payload_preserving_facts_and_replace")
    assert "container_at_capacity" not in cap.signals
    assert "container_slots_exhausted" not in cap.signals
    assert "append_would_exceed_cap" in cap.signals


def test_the_relocation_executor_does_not_claim_a_character_signal():
    """The same overclaim in the other direction would be equally wrong."""
    cap = next(c for c in _named_capacity_caps()
               if c.capability_id == "relocate_entry_preserving_information_then_retry")
    assert "append_would_exceed_cap" not in cap.signals
    assert "container_at_capacity" in cap.signals


def test_the_two_named_executors_do_not_overlap():
    """Two executors claiming one signal at one cell is how the wrong mechanism runs."""
    caps = {c.capability_id: set(c.signals) for c in _named_capacity_caps()}
    a, b = caps.values()
    assert not (a & b), f"overlapping claims: {a & b}"


def test_the_relocation_capability_declares_its_binding_and_is_not_a_ghost():
    cap = next(c for c in _named_capacity_caps()
               if c.capability_id == "relocate_entry_preserving_information_then_retry")
    assert "capacity_relocate" in cap.binding, "a capability with no binding is a ghost"
    assert not cap.disabled_reason, cap.disabled_reason
