#!/usr/bin/env python3
"""A5: a write was refused because the container is FULL. Can a slot be freed WITHOUT losing a fact?

A1 answers "the destination cannot hold this" by RELOCATING the payload. That works until the
relocation target itself saturates. A5 is the anchor for the state A1's own success produces: the
archival container is at its slot cap, and the pending write has nowhere left to go.

The naive repair is to evict something. That is a destructive act on a store whose contents are the
task's only memory, and the project has three measured rejections proving how badly it goes when the
choice of victim is delegated or heuristic (see `docs/KEEP_OR_DEFER.md`). A5 is the narrow case where
eviction is provably lossless:

    the container holds the SAME value twice  ->  removing one copy destroys no information

That is the whole mechanism. It is not "evict the least useful entry" -- no utility model exists, and
inventing one is the selection problem that `docs/SATURATION_AND_SELECTION.md` shows cannot be solved
from inside a single decision point. A5 only ever removes a copy of something the container still
holds, then retries the write that was refused.

THE FOUR STEPS, AND WHY EACH IS LOAD-BEARING

    1. READ THE LIVE CONTAINER at the decision point.
    2. PICK A DETERMINISTIC VICTIM: the last-written copy of a value with >= 2 copies.
    3. VERIFY A COPY REMAINS -- against the store, after the removal, before the retry.
    4. RETRY THE ORIGINAL CALL VERBATIM.

Step 1 is not an implementation detail. The first version of this anchor read its own episode-local
write log instead of the store, saw a median of 4 entries where the live container held ~50, never
found a victim, and the arm was VOID. Prerequisite episodes write the store that later episodes read,
so a controller restricted to its own log is not looking at the state available at its own decision
point.

Step 2 preserves the EARLIEST copy. Later queries are likelier to reference the first thing written,
and "last-written" is well defined here because ids are assigned monotonically. The rule is
deterministic, so no model is consulted about what to destroy.

Step 3 is the invariant that makes the anchor defensible, so it is CHECKED rather than argued. The
check runs against the live container after the removal: reasoning that a copy must remain is not the
same as observing that one does.

Step 4 replays the ORIGINAL call byte for byte. Never synthesize a replacement -- a synthesized
`add(text=...)` retry failed on every kv firing (`unexpected keyword argument`) *after* the eviction
had already succeeded, which destroyed a duplicate and lost the write: strictly worse than doing
nothing. Backend call signatures differ; the recorded call is already correct for its backend.

BOUNDED, BECAUSE UNBOUNDED EVICTION IS A CLEAR IN SLOW MOTION

At most `MAX_EVICTIONS_PER_EPISODE` firings. A loop that frees a slot per write, unbounded, empties
the container one entry at a time while every individual step passes the losslessness test.

WHAT THIS ANCHOR COST, RECORDED RATHER THAN SMOOTHED OVER

A5 was installed on an EXPLICIT OVERRIDE of a frozen acceptance clause requiring zero losses: it lost
3 cases on one cell against a zero variance floor. The losses were traced and are NOT the eviction --
`core_memory_remove` calls rose 42 -> 96 after A5 began firing, i.e. the model, seeing the store
change under it, became more destructive on its own. That is INDUCED MODEL DRIFT, a downstream
consequence of intervening at all, and it is the reason a provably information-preserving mechanism
can still lose cases. The override is recorded in `rounds/T6_A5_archival_full/OVERRIDE.md`; the 3
losses are baked into every later arm's baseline.

BACKEND SHAPE IS NOT PORTABLE, AND THE TRIGGER STRING IS NOT EITHER

The trigger phrasing differs per backend ("memory size exceeds maximum size" on vector, "Long term
memory is full" on kv), and the container shapes differ (`dict[int -> str]` under `._store` on
vector, a plain `dict[key -> value]` on the instance for kv). `live_container` therefore returns
None for "cannot read" and {} for "readable but empty", which are different facts: the upstream
version conflated them behind `if not live`, so its guard failed open exactly when absence was most
certain.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# A loop that frees one slot per pending write is a wholesale clear in slow motion.
MAX_EVICTIONS_PER_EPISODE = 3

# Per-backend "the container is full" phrasings. Vector and kv say it DIFFERENTLY, and an anchor
# keyed on one string simply never fires on the other backend -- measured, not hypothetical.
SLOT_FULL_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "vector": re.compile(r"memory size exceeds maximum size", re.IGNORECASE),
    "kv": re.compile(r"long term memory is full", re.IGNORECASE),
}

_ADD_RE = re.compile(r"archival_memory_add\s*\(")


def is_container_full_error(result: Any, backend: str) -> bool:
    """Does `result` say the archival container is out of SLOTS, on this backend?

    Slot exhaustion only. An entry-length rejection is a different locus with a different repair
    (A6), and conflating them sends the wrong mechanism at the failure.
    """
    text = str(result or "")
    if "error" not in text.lower():
        return False
    pattern = SLOT_FULL_PATTERNS.get(backend)
    return bool(pattern and pattern.search(text))


def is_archival_add(call: Any) -> bool:
    return bool(_ADD_RE.search(str(call or "")))


def normalize(value: Any) -> str:
    """Identity for duplicate detection: whitespace-collapsed, lowercased. NOTHING FUZZY.

    A near-duplicate is not a duplicate. Two similar entries are not interchangeable, so a fuzzy
    comparator would break the information-preservation invariant that is this anchor's entire
    justification -- it would let A5 destroy a fact on the grounds that something like it remains.
    """
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def live_container(instances: Any, container: str = "archival") -> dict[int, str] | None:
    """The live container as `{entry_id: normalized_text}`.

    Returns **None** when no instance exposes a readable container, and **{}** when one is readable
    and empty. Those are different facts and the caller must be able to tell them apart: a guard
    that treats "I cannot see the store" the same as "the store is empty" fails OPEN precisely when
    absence is most certain.

    Handles both shipped shapes -- vector keeps `dict[int -> str]` under `.archival_memory._store`,
    kv keeps a plain `dict` directly on `.archival_memory`.
    """
    found_readable = False
    values = (
        instances.values() if isinstance(instances, dict)
        else instances if isinstance(instances, Iterable) else []
    )
    for inst in values:
        holder = getattr(inst, f"{container}_memory", None)
        if holder is None:
            continue
        store = getattr(holder, "_store", None)          # vector
        if not isinstance(store, dict):
            store = holder if isinstance(holder, dict) else None   # kv
        if not isinstance(store, dict):
            continue
        found_readable = True
        if store:
            # Insertion order is preserved and ids are monotonic, so sorting by id gives
            # write order -- which is what "last-written" in `pick_victim` means.
            return {int(k): normalize(v) for k, v in sorted(store.items(), key=lambda kv: int(kv[0]))}
    return {} if found_readable else None


def pick_victim(container: Mapping[int, str]) -> tuple[int, str, int] | None:
    """The LAST-WRITTEN copy of a value that appears at least twice, or None.

    Returns `(entry_id, normalized_value, total_copies)`. Deterministic: no model, no scoring, no
    tie-break that depends on anything but write order. Preserves the EARLIEST copy, which later
    queries are likelier to have referenced.

    None means "no lossless eviction exists here" -- which is a legitimate and common answer, and
    the anchor must then decline rather than fall back to a destructive choice.
    """
    counts: dict[str, int] = {}
    for value in container.values():
        counts[value] = counts.get(value, 0) + 1
    duplicated = {v for v, n in counts.items() if n >= 2}
    if not duplicated:
        return None
    victim = None
    for entry_id, value in container.items():
        if value in duplicated:
            victim = (entry_id, value, counts[value])
    return victim


def copies_remaining(container: Mapping[int, str], value: str) -> int:
    """How many copies of `value` the container still holds. Step 3's observation."""
    return sum(1 for v in container.values() if v == value)


def plan_eviction(
    call: Any,
    result: Any,
    instances: Any,
    backend: str,
    evictions_so_far: int = 0,
) -> dict[str, Any]:
    """Decide whether a lossless eviction is admissible, and return the plan or the refusal.

    Pure decision logic: it reads state and returns a verdict. It never executes anything, so the
    order "look up strictly BEFORE execution, then act" is structurally enforced instead of being a
    convention someone has to remember.

    The returned dict always carries `fire` and `reason`. When `fire` is True it also carries
    `remove_call` (the deterministic removal to dispatch) and `retry_call` (the ORIGINAL call, to be
    replayed verbatim afterwards).
    """
    if not is_archival_add(call):
        return {"fire": False, "reason": "not an archival add"}
    if not is_container_full_error(result, backend):
        return {"fire": False, "reason": "not a slot-exhaustion error on this backend"}
    if evictions_so_far >= MAX_EVICTIONS_PER_EPISODE:
        return {"fire": False, "reason": f"episode budget spent ({MAX_EVICTIONS_PER_EPISODE})"}

    live = live_container(instances, "archival")
    if live is None:
        # Cannot see the store => cannot prove losslessness => decline. Never fail open here.
        return {"fire": False, "reason": "archival container unreadable"}
    if not live:
        return {"fire": False, "reason": "archival container readable but empty", "live_entries": 0}

    victim = pick_victim(live)
    if victim is None:
        return {
            "fire": False,
            "reason": "no duplicate exists, so no eviction is lossless",
            "live_entries": len(live),
            "live_distinct": len(set(live.values())),
        }

    entry_id, value, total = victim
    if total < 2:
        # Unreachable via pick_victim, asserted anyway: this is THE invariant, and a mechanism
        # whose justification rests on one condition should check it rather than trust its caller.
        return {"fire": False, "reason": "invariant refused: fewer than two copies"}

    return {
        "fire": True,
        "reason": f"{total} copies of the victim value exist; removing one is lossless",
        "remove_call": f"archival_memory_remove(vec_id={entry_id})",
        "retry_call": str(call),            # VERBATIM. Never synthesize a replacement.
        "victim_id": entry_id,
        "victim_value": value,
        "copies_before": total,
        "live_entries": len(live),
    }


def verify_invariant(instances: Any, value: str, container: str = "archival") -> dict[str, Any]:
    """Step 3, run AFTER the removal and BEFORE the retry.

    Observation, not inference. `ok` False means the anchor just destroyed the last copy of
    something, which is the one outcome its acceptance argument forbids -- so it must be visible in
    telemetry rather than deduced later from an accuracy delta.
    """
    live = live_container(instances, container)
    if live is None:
        return {"ok": False, "remaining": None, "reason": "container unreadable after removal"}
    remaining = copies_remaining(live, value)
    return {
        "ok": remaining >= 1,
        "remaining": remaining,
        "reason": "a copy remains" if remaining >= 1 else "INVARIANT VIOLATED: no copy remains",
    }


def evicted_and_retried(plan: Mapping[str, Any]) -> Sequence[str]:
    """The dispatch order this anchor requires: remove, then retry the original.

    Exposed as data so a port can assert the sequence rather than reimplement it.
    """
    if not plan.get("fire"):
        return ()
    return (str(plan["remove_call"]), str(plan["retry_call"]))
