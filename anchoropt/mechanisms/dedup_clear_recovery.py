#!/usr/bin/env python3
"""A8: a DESTRUCTIVE CLEAR is proposed on a full container. Can capacity be supplied instead?

Every other capacity anchor waits for a refusal and then repairs. This one fires on a *proposal* --
the model is about to call `clear` -- because after the clear executes there is nothing left to save.
It is the only anchor in the stack whose entire value depends on the decision point being
pre-execution.

    clear proposed on a container at capacity
      -> read the LIVE container
      -> does an exact duplicate exist (>= 2 copies under the SAME normalisation A5 uses)?
           YES -> suppress the clear, evict exactly ONE redundant copy (last-written, preserving
                  the earliest), verify >= 1 equivalent copy remains, retry the blocked write
                  VERBATIM
           NO  -> FALL THROUGH UNCHANGED. The clear executes exactly as it would have.

DELIBERATELY LAYER 1 OF A LADDER, AND NOTHING MORE
--------------------------------------------------
No concat, no consolidation, no merge, no summarisation, no generic "refuse destruction", no capacity
reasoning beyond "is there a redundant copy". Bundling consolidation in here would have made the two
effects indistinguishable in the measurement; keeping it to exact dedup is what makes its +1.65 pp
attributable to dedup specifically. See `CAPACITY_LADDER` in rounds/anchors.py.

THE FALL-THROUGH IS THE ENTIRE SAFETY ARGUMENT
----------------------------------------------
A predecessor refused clears UNCONDITIONALLY: 85 refusals, and **0 of 14 episodes closed**. Median kv
prereq steps went 19 -> 42 and max 43 -> 189. Refusing destruction *without supplying capacity* leaves
the model no route out, so it re-proposes rather than revises.

So the rule here is narrow on purpose: **suppress only when this gate can itself supply the capacity in
the same action.** Where it cannot, letting the destructive call proceed is the CORRECT branch, not a
missed opportunity. Every uncertain branch below returns "do not interfere" for that reason.

WHY THIS IS NOT A5 WITH A WIDER TRIGGER
---------------------------------------
A5's action is exactly right and 30/30 causally validated, but its TRIGGER cannot reach these states,
for two independent reasons:

  1. A5's trigger regex matches the VECTOR phrasing of "archival is full". Every event here is kv,
     which words it differently. A5 is a STRUCTURAL no-op on kv -- its recorded "kv 0.00" is
     unreachability, not a weak effect.
  2. A5 fires POST-EXEC on a blocked add. This fires PRE-EXEC on a proposed clear, one step later.

So A8 is a NEW TRIGGER reusing A5's VALIDATED ACTION -- a good outcome rather than a coincidence: the
action was already known to be lossless, and only the decision point was missing.

EQUIVALENCE IS A5'S, NOT A NEW ONE
----------------------------------
`normalize` is A5's rule verbatim: whitespace-collapsed, lowercased, NO fuzzy comparator. A
near-duplicate is not a duplicate -- two similar entries are not interchangeable, and fuzzy matching
would break the information-preservation invariant that is this anchor's whole justification.

It holds in practice for a reason worth knowing: the duplicates are *self-inflicted and byte-identical*.
Every observed victim is a `*_unique`-suffixed key the model invented to work around a
"key must be unique" refusal, so its value matches its unsuffixed partner exactly. The model
manufactured the redundancy while retrying.

MEASURED: 15/15 firings with `copies_before = 2` and `copies_remaining = 1` on every one, live 50 -> 49,
retry landed 15/15, zero invariant violations, byte-reproducible across two independent runs. Its
SUPPORT, though, is the narrowest in the stack -- one (backend, scenario) cell -- and dev cannot confirm
it. Read `A8_EVIDENCE` before quoting the delta.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

CLEAR_RE = re.compile(r"\b(core|archival)_memory_clear\s*\(")

# A loop that frees one slot per proposed clear would be a wholesale clear in slow motion. In the
# measured runs the duplicate SUPPLY bound first (max 6 firings per episode), so this is a backstop
# rather than a throttle -- and note A5's cap of 3 would have refused firings 4 and 5, declining
# capacity this gate can supply losslessly. Do not copy a sibling anchor's bound without checking.
MAX_PER_EPISODE = 8


def container_of(call: Any) -> str | None:
    """Which container a proposed clear targets, or None if it is not a clear at all."""
    match = CLEAR_RE.search(str(call or ""))
    return match.group(1) if match else None


def normalize(value: Any) -> str:
    """A5's identity rule, verbatim: whitespace-collapsed and lowercased. NOTHING FUZZY.

    Shared with `lossless_eviction` on purpose. Two anchors that claim the same
    information-preservation invariant must mean the same thing by "duplicate", or one of them is
    quietly weaker than its evidence.
    """
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def pick_victim(container: Mapping[Any, str]) -> tuple[Any, str, int, list[Any]] | None:
    """`(victim_key, normalized_value, total_copies, preserved_keys)`, or None if nothing is safe.

    Takes the LAST-WRITTEN copy of a value appearing >= 2 times and preserves the earliest, which
    later queries are likelier to have referenced. Deterministic: it depends only on insertion order
    and the normalised values, so no model decides what to destroy.

    Returning the preserved keys explicitly -- rather than just a count -- is what lets `verify` check
    the invariant against live state afterwards instead of inferring it.
    """
    counts: dict[str, int] = {}
    for value in container.values():
        key = normalize(value)
        counts[key] = counts.get(key, 0) + 1
    duplicated = {value for value, n in counts.items() if n >= 2}
    if not duplicated:
        return None

    victim: tuple[Any, str] | None = None
    for entry_key, value in container.items():      # insertion order -> last match is last-written
        if normalize(value) in duplicated:
            victim = (entry_key, normalize(value))
    if victim is None:
        return None

    victim_key, victim_value = victim
    preserved = [
        k for k, v in container.items() if normalize(v) == victim_value and k != victim_key
    ]
    return victim_key, victim_value, counts[victim_value], preserved


def removal_call(backend_id_kwarg: str, container: str, victim_key: Any) -> str:
    """The removal call in the backend's OWN address space.

    kv removes by key, vector by numeric id. Getting this wrong dispatches a call the tool rejects,
    which the gate would then report as a *failed eviction* rather than as a bug in itself -- so the
    address space comes from the adapter, never from an assumption about which backend we are on.
    """
    if backend_id_kwarg == "vec_id":
        return f"{container}_memory_remove(vec_id={victim_key})"
    return f"{container}_memory_remove({backend_id_kwarg}={victim_key!r})"


def plan_recovery(
    call: Any,
    live: Mapping[Any, str] | None,
    pending_write: Mapping[str, Any] | None,
    capacity: int | None,
    id_kwarg: str = "key",
    firings_so_far: int = 0,
) -> dict[str, Any]:
    """Decide whether capacity can be supplied instead of clearing. Reads state; executes nothing.

    Returns `fire` plus a reason. Every declining branch is explicit and intentional: the default is
    NOT to interfere, because interfering without a remedy is the measured failure mode described in
    the module docstring.
    """
    container = container_of(call)
    if container is None:
        return {"fire": False, "reason": "not a proposed clear"}
    if firings_so_far >= MAX_PER_EPISODE:
        return {"fire": False, "reason": f"episode budget spent ({MAX_PER_EPISODE})"}
    if live is None:
        return {"fire": False, "reason": "live container unreadable -- not interfering"}
    if not live:
        return {"fire": False, "reason": "live container empty -- nothing to evict"}
    if capacity is not None and len(live) < capacity:
        # Not a capacity-driven clear. The user may genuinely have asked to forget something, and
        # that is not this anchor's business.
        return {
            "fire": False,
            "reason": f"container not at capacity ({len(live)}/{capacity}) -- not interfering",
        }
    if not (pending_write or {}).get("call"):
        return {
            "fire": False,
            "reason": "no pending blocked write recorded -- nothing to retry, not interfering",
        }
    if (pending_write or {}).get("container") != container:
        return {
            "fire": False,
            "reason": (
                f"pending write targets {(pending_write or {}).get('container')!r}, clear targets "
                f"{container!r} -- freeing this container would not admit that write"
            ),
        }

    victim = pick_victim(live)
    if victim is None:
        # The container is full and holds nothing redundant. Dedup CANNOT create room here, and this
        # is the state a later ladder layer owns -- 5 of 5 remaining executed clears look like this.
        return {
            "fire": False,
            "reason": (
                f"no exact duplicate in {container} ({len(live)} entries, all distinct) -- "
                "not interfering; this state needs consolidation, not dedup"
            ),
            "ladder_layer_2_candidate": True,
        }

    victim_key, _victim_value, copies, preserved = victim
    if copies < 2 or not preserved:
        # Unreachable via pick_victim, asserted anyway: this is THE invariant, and a mechanism whose
        # justification rests on one condition should check it rather than trust its caller.
        return {
            "fire": False,
            "reason": f"invariant refused: copies={copies}, preserved={len(preserved)}",
        }

    return {
        "fire": True,
        "reason": (
            f"{container} is at capacity ({len(live)}/{capacity}) and holds {copies} identical "
            f"copies of one entry; removing one destroys no information"
        ),
        "container": container,
        "victim_key": victim_key,
        "copies_before": copies,
        "preserved_keys": preserved[:6],
        "remove_call": removal_call(id_kwarg, container, victim_key),
        "retry_call": (pending_write or {})["call"],     # VERBATIM. Never synthesize.
        "live_size": len(live),
        "capacity": capacity,
        "invariant": "the victim is gone AND at least one equivalent copy remains",
        "observation": observation_for(container, len(live), capacity, copies, victim_key),
    }


def observation_for(
    container: str, live_size: int, capacity: int | None, copies: int, victim_key: Any
) -> str:
    """What the model is told in place of the clear it asked for.

    It has to explain that capacity WAS supplied, or the model has no reason to stop re-proposing --
    which is the difference between this gate and the predecessor that refused clears and stalled.
    """
    return (
        f"{container.capitalize()} memory is at capacity ({live_size}/{capacity}) and held {copies} "
        f"identical copies of one entry. Removed the redundant copy {victim_key!r} (an equivalent "
        "copy remains) and re-applied the blocked write, so no other entry was destroyed."
    )


def verify(live_after: Mapping[Any, str] | None, plan: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen invariant, checked against LIVE state AFTER the eviction and BEFORE the retry.

    Two properties, both required:
      * the victim is GONE            -- the eviction actually happened
      * >= 1 equivalent copy REMAINS  -- no information was destroyed

    Observation, not inference. `ok=False` means this anchor just destroyed the last copy of
    something, which is the one outcome its acceptance argument forbids, so it must be visible in
    telemetry rather than deduced later from an accuracy delta.
    """
    if live_after is None:
        return {"ok": False, "reason": "container unreadable after the eviction", "checked": True}

    victim_key = plan["victim_key"]
    gone = victim_key not in live_after

    target: str | None = None
    for key in plan.get("preserved_keys") or ():
        if key in live_after:
            target = normalize(live_after[key])
            break
    remaining = (
        sum(1 for value in live_after.values() if normalize(value) == target) if target else 0
    )

    ok = gone and remaining >= 1
    return {
        "ok": ok,
        "checked": True,
        "victim_removed": gone,
        "copies_remaining": remaining,
        "size_after": len(live_after),
        "reason": (
            "victim removed and an equivalent copy remains" if ok
            else f"INVARIANT VIOLATED: victim_removed={gone}, copies_remaining={remaining}"
        ),
    }


def dispatch_sequence(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Evict, then retry the original write. Stated as data so a port can assert the order.

    The retry must come last and must be the recorded call, byte for byte.
    """
    if not plan.get("fire"):
        return ()
    return (str(plan["remove_call"]), str(plan["retry_call"]))
