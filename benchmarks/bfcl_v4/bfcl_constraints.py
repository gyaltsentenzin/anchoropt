"""BFCL v4 memory: which quantity each capacity refusal bounds, and what each operator changes.

THE BENCHMARK FACT CORE MUST NOT KNOW. "Capacity" is not one constraint here. `adapter.error_kind`
splits refusals three ways and the UNIT differs, which decides which remedy can work at all:

    blob_would_overflow   "too long after appending"                CHARACTERS in a single blob
    entry_too_long        "entry length" / "entry is too long"      CHARACTERS in one entry
    no_capacity           "is full" / "exceeds maximum size"        ENTRIES (occupied slots)

Measured counts in the raw traces: 218 `no_capacity` in kv, 121 `entry_too_long` in vector, 278
`blob_would_overflow` in rec_sum.

The kv/archival refusal is literally `{"error": "Memory size exceeds maximum size of 7 entries."}`.
Shortening a value therefore frees ZERO slots, which is why an autonomously proposed payload-reduction
transform on `container_at_capacity` was refuted at grounding rather than measured. The accepted
rec_sum recovery is feasible for the opposite reason: its constraint bounds characters in one blob, and
reducing the payload is exactly a reduction of that quantity.

Relocation is declared with care. Moving an item out of a container frees a slot IN THAT CONTAINER --
but rerouting INTO the container that is already full relieves nothing. So `relocate` reduces
`occupied_slots` only, and the destination's own capacity remains a separate constraint the host still
enforces at execution.

Eviction is declared here because it is the operator the kv refutation NAMES as missing: it is the only
declared operator that reduces `occupied_slots`. Declaring it does not implement it and does not hand
any policy to the proposer -- it records which quantity the operation would change, so core can tell a
feasible arm from an infeasible one.
"""

from __future__ import annotations

from anchoropt.learning.constraint_feasibility import (
    Constraint, ConstraintContract, OperatorEffect)

# Quantity tokens. Opaque to core; meaningful only here.
CHARS_IN_BLOB = "chars_in_blob"
CHARS_IN_ENTRY = "chars_in_entry"
OCCUPIED_SLOTS = "occupied_slots"

# The three capacity constraints, keyed by the error_kind the adapter classifies.
CONSTRAINTS = (
    Constraint(name="blob_would_overflow", quantity=CHARS_IN_BLOB, unit="characters",
               raised_by="single-blob store (rec_sum): one string, no second container"),
    Constraint(name="entry_too_long", quantity=CHARS_IN_ENTRY, unit="characters",
               raised_by="per-entry length limit (vector)"),
    Constraint(name="no_capacity", quantity=OCCUPIED_SLOTS, unit="entries",
               raised_by="entry-count limit (kv / archival): 'exceeds maximum size of N entries'"),
)

# What each operator can actually reduce.
OPERATOR_EFFECTS = (
    # The accepted rec_sum recovery. Reduces characters; cannot free a slot.
    OperatorEffect(operator="reduce_preserving_facts",
                   reduces=(CHARS_IN_BLOB, CHARS_IN_ENTRY),
                   unaffected=(OCCUPIED_SLOTS,)),
    # Rewriting one entry shorter. Same unit family, still no slot freed.
    OperatorEffect(operator="rewrite_entry_shorter",
                   reduces=(CHARS_IN_ENTRY,),
                   unaffected=(OCCUPIED_SLOTS,)),
    # Removing an item frees a slot in the container it was removed FROM.
    OperatorEffect(operator="evict_entry",
                   reduces=(OCCUPIED_SLOTS,),
                   unaffected=(CHARS_IN_ENTRY,)),
    # Moving an item out frees a slot in the SOURCE container. The destination's own capacity stays
    # a separate constraint -- rerouting into an already-full container relieves nothing.
    OperatorEffect(operator="relocate_entry",
                   reduces=(OCCUPIED_SLOTS,),
                   unaffected=(CHARS_IN_ENTRY,)),
    # Substituting the destination changes WHICH call runs; it reduces no quantity by itself.
    OperatorEffect(operator="substitute_destination",
                   reduces=(),
                   unaffected=(CHARS_IN_BLOB, CHARS_IN_ENTRY, OCCUPIED_SLOTS)),
)

# Which signal observes which constraint. Separate from CONSTRAINTS because a signal name is the
# learner's vocabulary while error_kind is the host's.
SIGNAL_CONSTRAINT = {
    "append_would_exceed_cap": "blob_would_overflow",
    "container_at_capacity": "no_capacity",
    "container_slots_exhausted": "no_capacity",
    "clear_proposed_at_capacity": "no_capacity",
    "entry_exceeds_len_limit": "entry_too_long",
}


def bfcl_constraint_contract() -> ConstraintContract:
    """The host's declaration, ready for core to enforce."""
    return ConstraintContract(constraints=CONSTRAINTS, effects=OPERATOR_EFFECTS)


def constraint_for_signal(signal: str) -> str:
    """The constraint a signal observes, or "" when it observes none.

    Empty is explicit: a signal with no declared constraint gets UNKNOWN from core rather than a
    silent pass.
    """
    return SIGNAL_CONSTRAINT.get(signal, "")
