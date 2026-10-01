"""Free ONE slot in a constrained store by RELOCATING an entry that is preserved elsewhere first.

THE RESIDUAL THIS SERVES, measured on the frozen raw-H0 kv runs (132 trajectories):

    218 core writes refused with "Core memory is full. Please clear some entries."
    208 of those 218 (95.4%) are followed by `core_memory_clear` -- a wholesale destruction of
        everything accumulated, including entries written by earlier episodes on the same chain

So the failure is not "the store is full". It is that the agent's own recovery move destroys the
information. The repair is to free exactly one slot without losing anything.

WHY RELOCATION IS SAFE HERE, and it is a measured fact rather than an assumption:

    core     MAX_CORE_MEMORY_SIZE = 7        MAX_CORE_MEMORY_ENTRY_LENGTH = 300
    archival MAX_ARCHIVAL_MEMORY_SIZE = 50   MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH = 2000

Archival is strictly more permissive on both axes, so a value that fits in core always fits in
archival. And at all 218 refusals the reconstructed archival occupancy was <= 1 of 50 -- headroom in
every single case. This is why relocation is offered here and eviction-by-deletion is NOT: deleting an
entry frees the same slot while destroying the fact, which is the very harm the residual is about. A
single-entry removal without preservation is not acceptable merely because it avoids a wholesale clear.

THE ORDER IS THE SAFETY PROPERTY. Destination write first, VERIFIED against live state, and only then
the source removal:

    1. read the LIVE store (not our own log -- A5 v1 read the log, saw 4 entries against a real ~50,
       found no victim, and the whole arm was void)
    2. pick a victim deterministically
    3. WRITE it to the destination and confirm the write landed by reading the destination back
    4. only then REMOVE it from the source, and confirm the removal landed
    5. retry the originally refused call VERBATIM, and only because a slot is now actually free
    6. if step 3 fails, STOP: the source entry is untouched and nothing was lost

Every step is checked against live state rather than inferred from a return string, because a return
string once made 32/32 repairs look successful while only 4 writes landed.

BOUNDED. `MAX_RELOCATIONS_PER_EPISODE` exists so a repair cannot become a wholesale clear in slow
motion -- A5's own words, and its own bound of 3.

WHAT THIS MODULE IS NOT. It is not a policy. It does not decide WHEN to run: no trigger, no threshold,
no priority. A controller supplies the condition; this supplies the operation. The historical A5
implementation was read for its safety discipline (live-state reads, a copy must remain, bounded
firings, verify then act) and that discipline is reproduced here for a different container pair.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

# A repair must not become a slow-motion clear. A5 uses 3 per episode; the same bound applies.
MAX_RELOCATIONS_PER_EPISODE = 3

_KEY_RE = re.compile(r"\bkey\s*=\s*(['\"])(.*?)\1", re.S)
_VALUE_RE = re.compile(r"\bvalue\s*=\s*(['\"])(.*?)\1", re.S)

# Store pairs: (source attribute, destination attribute, destination add verb, source remove verb).
# Declared as data so a backend with a different pair needs no new code path.
# `addressing` is the part that differs between the shipped backends, and it is DATA because the
# module's contract says a backend with a different pair needs no new code path:
#
#   "keyed"    kv      stores are plain dicts on the instance; entries are addressed by the model's
#                      own key, so add/remove carry key= and value=.
#   "ordinal"  vector  stores are VectorStore OBJECTS whose dict lives at `._store` as
#                      dict[int -> str]; ids are AUTO-ASSIGNED and monotonic, so add carries only the
#                      payload and remove carries the id. Insertion order is preserved, which is what
#                      makes `pick_victim`'s "first-inserted" rule deterministic here too.
#
# Both pairs share the SAME safety property and the SAME step order. Only the accessor and the two
# call templates change. `anchoropt/mechanisms/lossless_eviction._live_container` already unwraps both
# shapes for exactly this reason; this brings the relocation primitive in line with it.
RELOCATION_PAIRS = (
    {"source": "core_memory", "destination": "archival_memory",
     "addressing": "keyed", "store_attr": None,
     "add_verb": "archival_memory_add", "remove_verb": "core_memory_remove",
     "add_template": "{add_verb}(key='{key}', value={value!r})",
     "remove_template": "{remove_verb}(key='{key}')",
     "source_cap": 7, "destination_cap": 50,
     "why_safe": ("destination caps are strictly larger on both axes (50 vs 7 entries, 2000 vs 300 "
                  "chars), so a value that fits the source always fits the destination")},
    {"source": "core_memory", "destination": "archival_memory",
     "addressing": "ordinal", "store_attr": "_store",
     "add_verb": "archival_memory_add", "remove_verb": "core_memory_remove",
     "add_template": "{add_verb}(text={value!r})",
     "remove_template": "{remove_verb}(vec_id={key})",
     "source_cap": 7, "destination_cap": 50,
     "why_safe": ("same caps as the keyed pair, read from memory_vector's own MAX_* constants "
                  "(7/300 core, 50/2000 archival), so the same strict-dominance argument holds")},
)


class RelocationRefused(Exception):
    """An explicit refusal. Never a silent no-op: a silent one reads as success."""


def _live_stores(involved_instances: Any, pair: Mapping[str, Any]) -> tuple[dict, dict] | None:
    """The LIVE source and destination dicts from the tool system's own instance.

    Reads the instance attributes directly, the same handle the constraint-state capture uses, then
    unwraps the dict the PAIR declares in `store_attr` (None = the attribute IS the dict).

    Returning None on an unreachable store is deliberate and stays: a primitive that cannot see the
    store must decline rather than guess. What changed is that "not a dict" is no longer the same
    question as "not reachable" -- vector's stores are VectorStore objects holding their dict at
    `._store`, and reading through to it is addressing the backend, not guessing about it.

    The dicts are returned BY REFERENCE, which is load-bearing: every verification step below reads
    them back after a dispatch to check live state rather than trusting a return string.
    """
    insts = ((involved_instances or {}).values()
             if isinstance(involved_instances, dict) else (involved_instances or []))
    attr = pair.get("store_attr")

    def _unwrap(holder: Any) -> dict | None:
        if holder is None:
            return None
        if attr:
            inner = getattr(holder, str(attr), None)
            return inner if isinstance(inner, dict) else None
        return holder if isinstance(holder, dict) else None

    for inst in insts:
        src = _unwrap(getattr(inst, str(pair["source"]), None))
        dst = _unwrap(getattr(inst, str(pair["destination"]), None))
        if src is not None and dst is not None:
            return src, dst
    return None


#: Addressings this process is allowed to use. Default: all of them.
#:
#: `ANCHOROPT_RELOCATE_ADDRESSING` exists so a PAIRED CONTROL can be run with the pre-R12 behaviour
#: (`keyed` only, i.e. declining on vector) against an arm with both, WITHOUT shipping two builds of
#: this file. Two builds is how an import race voided earlier rounds: the control and the arm must
#: differ in exactly one declared way, and an env var is auditable from the job's own log whereas a
#: second copy on the path is not.
#:
#: It can only ever NARROW the declared set -- it cannot invent a pair -- so a typo yields a decline
#: with a reason rather than a silently different mechanism.
_ADDRESSING_ENV = "ANCHOROPT_RELOCATE_ADDRESSING"


def enabled_addressings() -> tuple[str, ...]:
    """The addressings this process may use, from the env, defaulting to every declared one."""
    import os
    raw = (os.environ.get(_ADDRESSING_ENV) or "").strip()
    declared = tuple(str(p.get("addressing") or "keyed") for p in RELOCATION_PAIRS)
    if not raw:
        return declared
    asked = tuple(t.strip() for t in raw.split(",") if t.strip())
    return tuple(a for a in declared if a in asked)


def _select_pair(involved_instances: Any) -> tuple[Mapping[str, Any], tuple[dict, dict]] | None:
    """The FIRST declared, ENABLED pair whose stores this instance actually exposes.

    Selection is by REACHABILITY, never by a backend name or a stack position: the instance answers
    which addressing applies, so no caller has to pass a backend label and no pair can be chosen for
    a store it cannot read.
    """
    allowed = enabled_addressings()
    for pair in RELOCATION_PAIRS:
        if str(pair.get("addressing") or "keyed") not in allowed:
            continue
        stores = _live_stores(involved_instances, pair)
        if stores is not None:
            return pair, stores
    return None


def already_preserved(key: Any, value: str, destination: Mapping[Any, str],
                      addressing: str = "keyed") -> bool:
    """Is `value` ALREADY in the destination, under this pair's notion of identity?

    The two shipped backends do not share one, and conflating them is a DATA-LOSS bug rather than a
    tidiness issue:

      keyed   (kv)     the model owns the key, so the same key in both stores with the same value is
                       genuinely the same entry -- identity is (key, value).
      ordinal (vector) ids are AUTO-ASSIGNED per store, so core id 3 and archival id 3 are unrelated.
                       Comparing ids would let a coincidental collision claim "already preserved",
                       and the caller would then SKIP the destination write and still remove the
                       source entry -- destroying the fact. Identity is therefore the VALUE alone.
    """
    if str(addressing) == "ordinal":
        return any(v == value for v in destination.values())
    return key in destination and destination[key] == value


def pick_victim(source: Mapping[str, str], destination: Mapping[str, str],
                protect: Sequence[str] = (), addressing: str = "keyed") -> tuple[str, str] | None:
    """Choose ONE entry to relocate. Deterministic, and never one already in the destination.

    Order of preference, and each clause has a reason:
      1. an entry ALREADY preserved in the destination (per `already_preserved`) -- relocating it is
         a no-op for information, so it is the cheapest possible slot. (A5's duplicate-first logic,
         reused: prefer the victim whose loss cannot cost anything.)
      2. otherwise the FIRST-INSERTED entry not in `protect`. Dict order is insertion order, and the
         oldest entry is the one the current turn is least likely to be about. `protect` carries keys
         the caller knows are live. For `ordinal` stores ids are monotonic, so iteration order is
         still write order and "first-inserted" means the same thing.

    Returns (key, value) or None. None means "no eligible victim", which must be reported, not
    worked around by widening the rule.
    """
    prot = {str(p) for p in (protect or ())}
    for k, v in source.items():
        if str(k) in prot:
            continue
        if already_preserved(k, v, destination, addressing):
            return k, v                      # already preserved: relocating costs nothing
    for k, v in source.items():
        if str(k) not in prot:
            return k, v
    return None


def relocate_and_retry(*, failing_call: str, involved_instances: Any, execute,
                       relocations_so_far: int = 0, protect: Sequence[str] = ()) -> dict:
    """Free one source slot by a VERIFIED relocation, then retry `failing_call` verbatim.

    `execute` is the host's own call executor: `execute([call_str]) -> (results, _)`. Passing it in
    keeps this module free of any evaluator import and means the repair runs through exactly the same
    dispatch the model's own calls do.

    Returns a telemetry dict. Every exit is explicit and names what happened; nothing returns a bare
    False. Keys are prefixed `relocate_` so the trajectory sidecar can allowlist them as a group --
    that allowlist has silently dropped intervention telemetry three times.
    """
    out: dict[str, Any] = {"relocate_attempted": True, "relocate_ok": False}

    if relocations_so_far >= MAX_RELOCATIONS_PER_EPISODE:
        out["relocate_declined"] = (
            f"bounded: {relocations_so_far} relocations already this episode "
            f"(max {MAX_RELOCATIONS_PER_EPISODE}) -- an unbounded cascade is a clear in slow motion")
        return out

    # Pair chosen by REACHABILITY against the live instance, not by a backend label.
    selected = _select_pair(involved_instances)
    if selected is None:
        names = sorted({str(p["source"]) for p in RELOCATION_PAIRS} |
                       {str(p["destination"]) for p in RELOCATION_PAIRS})
        shapes = sorted(enabled_addressings())
        out["relocate_enabled_addressings"] = ",".join(shapes)
        out["relocate_declined"] = (
            f"live stores unavailable: no instance exposes a readable store pair among {names} "
            f"in any ENABLED addressing {shapes}")
        return out
    pair, (source, destination) = selected
    out["relocate_addressing"] = str(pair.get("addressing"))
    out["relocate_source_entries_before"] = len(source)
    out["relocate_destination_entries_before"] = len(destination)

    # The destination must genuinely have room; otherwise this frees nothing and risks a loss.
    if len(destination) >= int(pair["destination_cap"]):
        out["relocate_declined"] = (
            f"destination {pair['destination']} is itself at capacity "
            f"({len(destination)}/{pair['destination_cap']}): relocation would not free a slot safely")
        return out

    addressing = str(pair.get("addressing") or "keyed")
    victim = pick_victim(source, destination, protect=protect, addressing=addressing)
    if victim is None:
        out["relocate_declined"] = "no eligible victim in the source store"
        return out
    vkey, vval = victim
    out["relocate_victim_key"] = vkey
    out["relocate_victim_chars"] = len(str(vval))
    out["relocate_victim_already_in_destination"] = bool(
        already_preserved(vkey, vval, destination, addressing))

    # ---- STEP 1: WRITE TO THE DESTINATION FIRST, then VERIFY against live state ----------------
    if out["relocate_victim_already_in_destination"]:
        out["relocate_write_skipped"] = "an identical copy is already in the destination"
    else:
        add_call = str(pair["add_template"]).format(
            add_verb=pair["add_verb"], key=vkey, value=str(vval))
        try:
            res, _ = execute([add_call])
        except Exception as exc:
            out["relocate_declined"] = f"destination write raised {type(exc).__name__}: {exc}"
            return out
        out["relocate_write_result"] = " ".join(str(r) for r in (res or []))[:160]
        # VERIFY AGAINST LIVE STATE, not the return string. A return string once made 32/32 repairs
        # look successful while only 4 writes landed.
        if not already_preserved(vkey, vval, destination, addressing):
            out["relocate_declined"] = (
                "destination write did NOT land (live state disagrees with the result string); "
                "the source entry is untouched and nothing was lost")
            out["relocate_preserved_source_intact"] = source.get(vkey) == vval
            return out
    out["relocate_write_verified"] = True

    # ---- STEP 2: ONLY NOW remove from the source, and verify that too -------------------------
    rm_call = str(pair["remove_template"]).format(
        remove_verb=pair["remove_verb"], key=vkey, value=str(vval))
    try:
        rres, _ = execute([rm_call])
    except Exception as exc:
        out["relocate_declined"] = f"source removal raised {type(exc).__name__}: {exc}"
        out["relocate_copy_survives_in_destination"] = already_preserved(
        vkey, vval, destination, addressing)
        return out
    out["relocate_remove_result"] = " ".join(str(r) for r in (rres or []))[:160]
    if vkey in source:
        out["relocate_declined"] = ("source removal did NOT land; no slot was freed. The copy in the "
                                    "destination remains, so the fact is still preserved twice")
        out["relocate_copy_survives_in_destination"] = already_preserved(
        vkey, vval, destination, addressing)
        return out
    out["relocate_removed_verified"] = True
    # THE INVARIANT, asserted at the point of action rather than audited afterwards.
    out["relocate_copy_survives_in_destination"] = already_preserved(
        vkey, vval, destination, addressing)
    if not out["relocate_copy_survives_in_destination"]:
        out["relocate_invariant_violated"] = True     # must never happen: write was verified first

    # ---- STEP 3: a slot is genuinely free -> retry the original call VERBATIM ------------------
    out["relocate_source_entries_after"] = len(source)
    out["relocate_slot_freed"] = len(source) < out["relocate_source_entries_before"]
    if not out["relocate_slot_freed"]:
        out["relocate_declined"] = "no slot was actually freed; not retrying"
        return out

    try:
        tres, _ = execute([failing_call])
    except Exception as exc:
        out["relocate_declined"] = f"retry raised {type(exc).__name__}: {exc}"
        return out
    blob = " ".join(str(r) for r in (tres or []))
    out["relocate_retry_result"] = blob[:160]
    out["relocate_retry_landed"] = "error" not in blob.lower()
    out["relocate_ok"] = bool(out["relocate_retry_landed"])
    # What the ORIGINAL call was trying to store, for the preservation audit.
    m = _KEY_RE.search(str(failing_call))
    if m:
        out["relocate_retried_key"] = m.group(2)
    out["relocate_extra_results"] = list(tres or [])
    return out
