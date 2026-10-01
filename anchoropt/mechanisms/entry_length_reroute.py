#!/usr/bin/env python3
"""A6: a write was refused because the ENTRY is too long, not because the container is full.

*** A6 IS DEFERRED, NOT ACCEPTED -- the remedy regresses the validation split. ***

This module ships because the SIGNAL is real and unaddressed and the measurement is the most
transferable result in the project -- not because the remedy is in the stack. Read
`rounds/A6_deferred/DEFERRED.md` before using any of it. Under the governing acceptance rule
(`docs/ACCEPTANCE_RULE.md`) it fails two criteria INDEPENDENTLY:

    criterion 2  dev net -1.19 pp (8 gains / 9 losses, 17 discordant -- it churns 20% of dev to a
                 net effect of -1)
    criterion 3  its OWN target backend is negative: vector -2.50 pp. An anchor that engages
                 exactly where designed and makes things worse there is not a tolerance question.

Every variant tried is negative on dev, so this is not a tuning problem and tuning is CLOSED. The
standing instruction is NOT to re-enable the flags; four reopening conditions are recorded in
DEFERRED.md.

Two parts, built together because neither half is viable alone:

    RELOCATE   catch the per-entry length rejection and re-address the payload to a container
               with a larger per-entry cap.
    DEDUP      before any write to that backend reaches the executor, drop it if the target
               container already holds a BYTE-IDENTICAL payload.

The dedup half exists because the relocate half, unguarded, spent 85% of the archival slots it was
granted on redundant content: the model retries the same over-long write, A6 reroutes it again, and
the reroute manufactures the duplicate saturation that later forces a destructive clear. A repair
that generates the pressure it was built to relieve is not a repair.

WHY THIS IS A SEPARATE ANCHOR FROM A1

A1's signal is "the container has no room" ('is full', 'exceeds maximum size'). Verified by direct
call: those strings match NEITHER entry-length phrasing. Different signal, different repair -- A1
cannot fire here, and the cap asymmetry A6 exploits is a per-ENTRY one (300 chars in core vs 2000 in
archival), not a slot count.

The action is a pure relocation: every attributable rejected payload measured on this line was under
the destination's per-entry cap, so the destination accepts it VERBATIM. No condensation, no split,
no model call. When that stops being true the anchor must decline rather than start rewriting -- a
rewrite is A7's problem and it is a much harder one.

WHY IT WAS DEFERRED: A6 IS A SELECTION INTERVENTION IN A CAPACITY COSTUME
------------------------------------------------------------------------
A6 helps on train (+1.65 pp) and REGRESSES on the validation split (-1.19 pp), and the mechanism is
measured, not speculated. On dev A6 fired 44 times while A1's own dispatches collapsed 22 -> 6: A6
catches the payload on the entry-length error and reroutes it BEFORE the write ever reaches the
core-full state where A1 would have acted. A6 does not add a capability. It PRE-EMPTS A1 on the same
payloads.

On a container already at its slot cap that means A6 cannot add information. It can only reorder
which entries win the fixed number of slots. Final stores: 57 distinct facts vs 56, with 54 facts
destroyed and 53 added -- a near-total content swap at constant saturated capacity, keeping the same
TOPICS in different PHRASINGS. Whether that helps is a coin flip on how the surviving phrasing aligns
with the queries, which is exactly the observed sign flip.

So the honest description is: it relieves entry-length pressure where the store has headroom, and it
merely reshuffles slots where it does not -- and the corpus it was measured on is mostly saturated.
`displacement_check` in `anchoropt/learning/exposure.py` is the diagnostic that distinguishes those two
regimes, and `docs/SATURATION_AND_SELECTION.md` is the full argument. RUN IT before porting any
capacity-shaped anchor anywhere; that diagnostic exists because of this deferral.

DEDUP IS RAW-EXACT, AND THAT BEAT NORMALIZED ON PURPOSE
-------------------------------------------------------
(Worth keeping even though the anchor is deferred: if duplicate suppression is ever revisited, start
from raw byte equality. This is the half of A6 with a result that survives it.)
Byte equality only. A normalized comparator (casefold, strip punctuation, collapse whitespace) was
measured against it and lost: normalized over-suppressed, retaining 17 held-out facts where raw-exact
retained 56. Informative variants matter, so suppression is NOT monotonically good -- two phrasings of
a topic can answer different questions. Raw-exact declines every paraphrase by construction, which is
the conservative direction when the cost of a false positive is silent data loss.

The remaining tier is paraphrase: 15.8% of allowed writes are >= 0.90-similar one-edit paraphrases and
73.7% are < 0.50 genuinely distinct. Exact matching cannot reach the former, and merging them would
change which phrasing survives -- the same coin flip described above. Do not "improve" this predicate
into a semantic one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# Per-entry length rejections. Distinct from A1's container-full strings, verified by direct call.
ENTRY_TOO_LONG_RE = re.compile(r"entry length exceeds maximum length|entry is too long", re.IGNORECASE)

_CORE_ADD_RE = re.compile(r"\bcore_memory_add\s*\(")
_ANY_ADD_RE = re.compile(r"\b(core|archival)_memory_add\s*\(")
_KWARG_VALUE_RE = re.compile(r"(?:text|value|content)\s*=\s*(['\"])(.*?)\1", re.DOTALL)
_POSITIONAL_VALUE_RE = re.compile(r"\(\s*(['\"])(.*)\1\s*\)\s*$", re.DOTALL)

# Per-entry caps, from the shipped backends. The asymmetry is what makes relocation work.
ENTRY_CAPS: Mapping[str, int] = {"core": 300, "archival": 2000}


def is_entry_too_long(result: Any) -> bool:
    """Is this result a per-ENTRY length rejection (as opposed to slot exhaustion)?"""
    return bool(ENTRY_TOO_LONG_RE.search(str(result or "")))


def payload_of(call: Any) -> str | None:
    """The VERBATIM payload of a write call, or None if it cannot be parsed.

    Both call shapes are handled: `add(text='...')` as the model emits it, and `add('...')` as the
    reroute primitive emits it. Returning None on an unparseable call matters -- the caller must then
    let the call through unchanged rather than silently drop a write it failed to understand.
    """
    text = str(call or "")
    match = _KWARG_VALUE_RE.search(text)
    if match:
        return match.group(2)
    match = _POSITIONAL_VALUE_RE.search(text)
    return match.group(2) if match else None


def container_of(call: Any) -> str | None:
    """Which container a write addresses ('core' or 'archival'), or None."""
    match = _ANY_ADD_RE.search(str(call or ""))
    return match.group(1) if match else None


def find_failing_call(calls: Sequence[Any], results: Sequence[Any]) -> str | None:
    """The first core write whose paired result is an entry-length rejection, else None."""
    for call, result in zip(calls or (), results or ()):
        text = str(call)
        if _CORE_ADD_RE.search(text) and is_entry_too_long(result):
            return text
    return None


def plan_reroute(call: Any, result: Any, destination: str = "archival") -> dict[str, Any]:
    """Relocate an over-long entry to a container whose per-entry cap can hold it.

    Declines when the payload would not fit the destination either. That case needs a rewrite, and a
    rewrite is a different anchor with a much weaker guarantee -- silently starting to condense here
    would smuggle a lossy operation in behind a lossless one's evidence.
    """
    if not is_entry_too_long(result):
        return {"fire": False, "reason": "not an entry-length rejection"}
    payload = payload_of(call)
    if payload is None:
        return {"fire": False, "reason": "payload unparseable; passing the call through untouched"}
    cap = ENTRY_CAPS.get(destination)
    if cap is not None and len(payload) > cap:
        return {
            "fire": False,
            "reason": f"payload {len(payload)} exceeds the {destination} per-entry cap {cap}; "
                      "relocation cannot help and rewriting is a different anchor",
            "payload_len": len(payload),
        }
    return {
        "fire": True,
        "reason": f"{destination} per-entry cap {cap} accepts {len(payload)} chars verbatim",
        "reroute_call": f"{destination}_memory_add('{payload}')",
        "payload_len": len(payload),
        "destination": destination,
    }


def _live(instances: Any, container: str) -> dict[Any, str] | None:
    """Live contents of `container` as `{id: RAW text}`, or None if unreadable.

    RAW: no normalization anywhere in this module. The comparison is byte equality, so normalizing
    on read would silently make the predicate fuzzy.
    """
    found = False
    values = (
        instances.values() if isinstance(instances, dict)
        else instances if isinstance(instances, Iterable) else []
    )
    for inst in values:
        holder = getattr(inst, f"{container}_memory", None)
        if holder is None:
            continue
        store = getattr(holder, "_store", None)
        if not isinstance(store, dict):
            store = holder if isinstance(holder, dict) else None
        if not isinstance(store, dict):
            continue
        found = True
        if store:
            return {k: str(v) for k, v in store.items()}
    return {} if found else None


def raw_duplicate(call: Any, instances: Any) -> tuple[bool, Any, int]:
    """`(is_duplicate, existing_id, live_count)` on RAW byte equality in the TARGET container.

    The target container is part of the predicate. core and archival are independent namespaces, so
    comparing across them would suppress a write that would in fact have landed.
    """
    container = container_of(call)
    if container is None:
        return False, None, 0
    live = _live(instances, container)
    if live is None:
        return False, None, 0
    payload = payload_of(call)
    if not isinstance(payload, str):
        return False, None, len(live)
    for entry_id, value in live.items():
        if value == payload:            # RAW byte equality. No normalization. Deliberate.
            return True, entry_id, len(live)
    return False, None, len(live)


def dedup_filter(
    calls: Sequence[Any],
    instances: Any,
    backend: str,
    exposed_backend: str = "vector",
) -> tuple[list[Any], dict[int, str], list[dict[str, Any]]]:
    """Drop writes whose payload the target container already holds byte-identically.

    Returns `(kept, withheld, decisions)`:

        kept        the calls to execute, in order, with duplicates removed
        withheld    {ORIGINAL INDEX -> success-equivalent observation}
        decisions   per-call telemetry, so engagement is reported from the gate's own view

    `withheld` is keyed by ORIGINAL INDEX because the caller must splice the observations back at the
    positions the model's proposals occupied -- results are paired with proposals POSITIONALLY, and
    one off-by-one silently pairs a result with the wrong call. Use `splice_results` for that, in ONE
    pass: two separate splice passes over the same iterator double-consume it and drop a real result.

    Gated to `exposed_backend`. The other backends are then untouched by construction, which is what
    makes them a free noise estimate on every paired run -- see `exposure_weighted_delta`.
    """
    if backend != exposed_backend:
        return list(calls or ()), {}, []

    kept: list[Any] = []
    withheld: dict[int, str] = {}
    decisions: list[dict[str, Any]] = []

    for index, call in enumerate(calls or ()):
        text = str(call)
        if not _ANY_ADD_RE.search(text):
            kept.append(call)
            continue
        try:
            duplicate, existing_id, live_count = raw_duplicate(text, instances)
        except Exception as exc:  # noqa: BLE001 - a guard must never break the run it guards
            decisions.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            kept.append(call)
            continue
        decisions.append({
            "index": index,
            "container": container_of(text),
            "live_count": live_count,
            "duplicate": bool(duplicate),
            "existing_id": None if existing_id is None else str(existing_id),
            "payload_len": len(payload_of(text) or ""),
        })
        if duplicate:
            withheld[index] = already_stored_observation(existing_id)
        else:
            kept.append(call)

    return kept, withheld, decisions


def already_stored_observation(existing_id: Any) -> str:
    """What the model is told in place of a suppressed duplicate write.

    A success-equivalent observation: the write it asked for is, as far as the store is concerned,
    already done. Telling it the write FAILED would invite a retry loop -- the exact pressure this
    half of the anchor exists to remove.
    """
    return f"{{'id': {existing_id!r}, 'status': 'already stored'}}"


def splice_results(
    results: Sequence[Any],
    withheld: Mapping[int, str],
    total: int,
) -> list[Any]:
    """Reassemble the full result list, in ONE pass, with withheld observations at their own indices.

    `results` are the outcomes of the KEPT calls, in order. `total` is the original call count. The
    invariant is `len(return) == total`, so a proposal at index i keeps the result at index i.

    One pass over all withhold maps, always. Two guards each splicing separately produced
    `[rA, CACHED_B, None, WITHHELD_C]` where the truth was `[rA, CACHED_B, WITHHELD_C, rD]` -- the
    last real result silently dropped, with every unit test still green.
    """
    out: list[Any] = []
    stream = iter(results or ())
    for index in range(total):
        if index in withheld:
            out.append(withheld[index])
        else:
            out.append(next(stream, None))
    return out
