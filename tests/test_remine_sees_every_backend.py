"""The remine step's failure signatures must cover EVERY shipped backend's store messages.

THE DEFECT THIS PINS (rounds/AUTONOMY/R12). `FAILURE_SIGNATURES` is a hand-maintained lexical list and
every entry in it was a kv or rec_sum message, so NONE of vector's three capacity refusals matched:

    "Memory size exceeds maximum size of 7 entries."          41 occurrences, core slots
    "Entry length exceeds maximum length of 300 characters."  121 occurrences, per-entry chars
    "Memory size exceeds maximum size of 50 entries."         archival slots

So the remine step could not see a single vector failure as a residual. That is a SECOND, independent
blindness on the same cell as the `constraint_state` one, at a different layer -- even with live state
supplied, mining could not classify the failure. Both have the same shape: a hand-maintained list of
one backend's vocabulary standing in for "every backend".

The adapter already declared this correctly in `benchmarks/bfcl_v4/bfcl_constraints.py`
("is full" / "exceeds maximum size" -> no_capacity; "entry length" -> entry_too_long), so the defect
was this list being maintained separately from that declaration. This test ties them together.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _remine():
    spec = importlib.util.spec_from_file_location(
        "_remine_under_test", REPO / "scripts/remine_from_incumbent.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


#: The REAL store messages, one per refusal shape each shipped backend can raise. Sourced from the
#: backend modules' own strings, not paraphrased.
REAL_MESSAGES = {
    "kv_core_full": "Core memory is full. Please clear some entries.",
    "kv_duplicate_key": "Key name must be unique.",
    "rec_sum_blob_overflow": "Memory is too long after appending.",
    "vector_core_slots": "Memory size exceeds maximum size of 7 entries.",
    "vector_entry_length": "Entry length exceeds maximum length of 300 characters.",
    "vector_archival_slots": "Memory size exceeds maximum size of 50 entries.",
}


@pytest.mark.parametrize("name", sorted(REAL_MESSAGES))
def test_every_backends_refusal_is_visible_to_the_miner(name):
    sigs = _remine().FAILURE_SIGNATURES
    msg = REAL_MESSAGES[name].lower()
    assert any(s in msg for s in sigs), (
        f"{name} is INVISIBLE to remine: {REAL_MESSAGES[name]!r} matches none of {sigs}. "
        "A whole cell's residual would be unmineable."
    )


def test_the_slot_and_character_units_stay_distinguishable():
    """Not one collapsed "exceeds maximum" signature.

    "exceeds maximum size" bounds ENTRIES and "entry length exceeds" bounds CHARACTERS, and the remedy
    that can work differs: shortening a value frees ZERO slots. bfcl_constraints.py makes exactly this
    split (no_capacity / entry_too_long), and collapsing the signatures would erase it at the miner.
    """
    sigs = _remine().FAILURE_SIGNATURES
    slots = "memory size exceeds maximum size of 7 entries."
    chars = "entry length exceeds maximum length of 300 characters."
    slot_sigs = {s for s in sigs if s in slots}
    char_sigs = {s for s in sigs if s in chars}
    assert slot_sigs, "slot refusals unmatched"
    assert char_sigs, "entry-length refusals unmatched"
    assert slot_sigs != char_sigs, (
        "the two UNITS share an identical signature set, so the miner cannot tell a slot refusal "
        "from a character refusal -- and only one of them can be repaired by shortening"
    )


def test_the_miner_agrees_with_the_adapters_own_constraint_declaration():
    """The two lists must not drift: every constraint the adapter declares needs a visible message."""
    sys.path.insert(0, str(REPO / "benchmarks/bfcl_v4"))
    from bfcl_constraints import CONSTRAINTS            # noqa: E402

    sigs = _remine().FAILURE_SIGNATURES
    declared = {c.name for c in CONSTRAINTS}
    assert {"no_capacity", "entry_too_long", "blob_would_overflow"} <= declared

    # One real message per declared constraint; each must be visible to the miner.
    witness = {
        "no_capacity": "Memory size exceeds maximum size of 7 entries.",
        "entry_too_long": "Entry length exceeds maximum length of 300 characters.",
        "blob_would_overflow": "Memory is too long after appending.",
    }
    for cname, msg in witness.items():
        assert any(s in msg.lower() for s in sigs), (
            f"the adapter declares constraint {cname!r} but the miner cannot see its message {msg!r}"
        )
