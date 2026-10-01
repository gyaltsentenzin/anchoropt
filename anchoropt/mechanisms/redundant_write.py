#!/usr/bin/env python3
"""A3: is a proposed write REDUNDANT with what the store already holds?

The decision is made PRE-EXECUTION -- post_generation_pre_execution, the only incision point
where cancelling a call is free. After execution the duplicate has already been rejected and
the turn's step is spent; before generation there is no call to inspect.

WHY THIS EXISTS

A1 rescues a core-full write by rerouting it to archival under the same key. The model, unaware
that A1 acted, later issues its own `archival_memory_add(key=K, ...)` and the store answers
"Key name must be unique". Measured on the clean A1+A2 corpus: **all 59** such collisions are
against keys A1 itself injected, and **54/59 (92 %)** carry a value token-identical to the one
A1 rescued. So the model's write is overwhelmingly a redundant repeat of work already done.

    A1 rescues the failed write.  A3 suppresses the model's redundant re-write.

DELIBERATELY EXACT, NOT SEMANTIC

The predicate is: same store, same key, NORMALISED-IDENTICAL value. Nothing fuzzy. 55 of 59
collisions (93 %) are exact duplicates after whitespace normalisation, so a deterministic rule
covers almost all of the locus without ever risking the suppression of genuinely new
information. The 4 that remain are three paraphrases and one materially different fact
(`lifestyle_changes`: "cutting out sugary snacks…" vs "ergonomic chairs, good shoes…") -- those
must stay residual and be re-mined, not swept up by a loosened comparator.

Normalisation is minimal on purpose: strip and collapse whitespace, nothing else. Every extra
transform (casefolding, punctuation stripping, stemming) is another chance to call two
different facts the same, and the cost of a false positive here is silent data loss.

NEVER USE TELEMETRY VALUES

`*_substituted_call` is truncated to ~300 characters. Comparing against it produced a **wrong
verdict twice**: 8 spurious "refused" replays, and a "64 % materially new" reading that
inverted to "92 % redundant" once untruncated values were used. `assert_untruncated` exists so
that class of error fails loudly instead of silently changing a conclusion.
"""

from __future__ import annotations

import re

# Telemetry fields are capped at this length by the sidecar; a value at or beyond the cap is
# assumed truncated and unusable for comparison.
TELEMETRY_CAP = 300


def normalize(value: str | None) -> str:
    """Whitespace-only normalisation. Intentionally weak -- see module docstring."""
    return re.sub(r"\s+", " ", value or "").strip()


def assert_untruncated(value: str | None, source: str) -> str:
    """Refuse to compare a value that may have been truncated by telemetry.

    A truncated value silently compares as "different", which turns a redundant write into an
    apparently-novel one. That has already produced two wrong answers in this project, so it
    raises rather than warns.
    """
    v = value or ""
    if len(v) >= TELEMETRY_CAP:
        raise ValueError(
            "value from %r is %d chars, at or beyond the %d-char telemetry cap -- it may be "
            "truncated. Compare against the untruncated call text (`decoded`) or live store "
            "state, never against *_substituted_call." % (source, len(v), TELEMETRY_CAP))
    return v


def is_redundant(proposed_value: str | None,
                 stored_value: str | None) -> tuple[bool, str]:
    """(suppress?, reason). True only on a normalised-exact match.

    Returns False -- explicitly, with a reason -- whenever the answer is not certain: absent
    key, missing value, or any difference at all. The default is to let the write through.
    """
    if stored_value is None:
        return False, "key not present in the store: the write is not a duplicate"
    if proposed_value is None:
        return False, "proposed call carries no value to compare"
    a, b = normalize(proposed_value), normalize(stored_value)
    if not a or not b:
        return False, "one side is empty after normalisation -- cannot establish redundancy"
    if a == b:
        return True, "normalised-identical value already stored under this key"
    return False, "values differ (%d vs %d chars) -- may be new information" % (len(b), len(a))


# The pre-refactor literal, kept as the standalone fallback so this file imports on its own.
_HISTORICAL_STATE_ATTRS: tuple[str, ...] = ("core_memory", "archival_memory", "memory")


def _state_attrs() -> tuple[str, ...]:
    """Attribute names a store may expose its entries under.

    Read from `StoreAdapter.STATE_ATTRS` so a port declares them once beside its other store facts,
    instead of editing this mechanism. Falls back to the historical tuple if the adapter module is
    unavailable, which keeps this file importable on its own.
    """
    try:
        from anchoropt.mechanisms.constraint_repair import STATE_ATTRS
        return STATE_ATTRS
    except ImportError:                                 # pragma: no cover - import-order safety
        return _HISTORICAL_STATE_ATTRS


def store_lookup(mem_inst, key: str) -> str | None:
    """Read the live value at `key`, from whichever container the backend exposes.

    Duck-typed and container-agnostic: the caller does not say which store to look in, so a
    backend with differently-named containers needs no change here. Returns None when the key
    is absent everywhere, which is_redundant treats as "not a duplicate".
    """
    # >>> THE ONLY BENCHMARK SEAM IN THIS FILE, and it now comes from the store adapter rather than
    # a literal tuple. Everything else here is duck-typed. See docs/GENERALIZABILITY.md.
    for attr in _state_attrs():
        container = getattr(mem_inst, attr, None)
        if isinstance(container, dict) and key in container:
            v = container[key]
            return v if isinstance(v, str) else str(v)
    return None


def should_suppress(call_args: dict[str, str], mem_inst,
                    key_field: str = "key",
                    value_field: str = "value") -> tuple[bool, str]:
    """Top-level predicate for the pre-execution gate.

    Field names are parameters, not constants, so a backend keying on different argument names
    supplies them rather than requiring a code change here.
    """
    key = (call_args or {}).get(key_field)
    if not key:
        return False, "call carries no key -- nothing to compare against"
    proposed = (call_args or {}).get(value_field)
    stored = store_lookup(mem_inst, key)
    return is_redundant(proposed, stored)

# ---------------------------------------------------------------------------------------------------
# C2: KV REDUNDANT-WRITE SUPPRESSION -- same key + CANONICALLY EQUIVALENT information.
#
# A3 already suppresses redundant ARCHIVAL writes whose value is whitespace-identical. Measuring the 24
# T1 events on ground-truth kv state shows A3 catches EXACTLY ZERO of them, for two separable reasons:
#
#     6 events   the call is `core_memory_add`, which A3's match_substrings never matches
#    18 events   the value differs ONLY in punctuation or clause order, which whitespace-only
#                normalisation reads as different:
#                    stored  1. $8.5B semiconductor merger(QuantumChip Technologies & ...
#                    wanted  1. $8.5B semiconductor merger (QuantumChip Technologies & ...
#
# So this is a genuine gap, not a re-implementation. The fix is deliberately two narrow changes and
# nothing else: extend to core, and compare canonically.
#
# CANONICAL EQUIVALENCE, and why it is still LOSSLESS. `canonical` reduces a value to the sorted
# multiset of its alphanumeric tokens. Two values with identical token content in any order and any
# punctuation carry the same information, so suppressing the second stores nothing new -- the store
# already answers that question. This is strictly stronger than whitespace normalisation and strictly
# weaker than semantic matching, which is the point: no paraphrase, no stemming, no synonym is ever
# treated as equivalent.
#
# WHAT IS DELIBERATELY EXCLUDED. The single genuine MERGE case stays out of scope:
#     lifestyle_changes  stored "cutting out sugary snacks, watching carb intake"
#                        wanted "ergonomic chairs, good shoes, avoiding joint overuse"
# Different token content, so `canonical` does not match it and the write proceeds untouched. Merging
# would mean composing a new value -- a broader action space than suppression, and B1 lost 6.60pp with a
# broad one. Suppression alone cannot invent, corrupt, or lose anything: the only outcome is that a call
# the tool would have REJECTED is never issued.
TOKEN_RE = None


def canonical(value):
    """Sorted multiset of alphanumeric tokens -- punctuation- and order-insensitive.

    Chosen over difflib similarity because it is exact and has no threshold to tune. A ratio-based
    comparator would need a cutoff, and every cutoff is a place where two genuinely different facts get
    called the same. Here equivalence is decidable: identical token content, or not.
    """
    import re as _re
    return " ".join(sorted(_re.findall(r"[a-z0-9]+", str(value or "").lower())))


def is_redundant_canonical(proposed_value, stored_value):
    """(suppress?, reason). True on canonical-token equality, or when the proposal adds no new tokens.

    The containment branch matters because it is the same guarantee in a weaker form: if every token of
    the proposal already appears in the stored value, the proposal contributes nothing addressable.
    """
    if stored_value is None:
        return False, "key not present in the store: the write is not a duplicate"
    if proposed_value is None:
        return False, "proposed call carries no value to compare"
    exact, why = is_redundant(proposed_value, stored_value)
    if exact:
        return True, why
    a, b = canonical(proposed_value), canonical(stored_value)
    if not a or not b:
        return False, "one side is empty after canonicalisation -- cannot establish redundancy"
    if a == b:
        return True, "CANONICALLY IDENTICAL -- same token content, differs only in punctuation or order"
    at, bt = set(a.split()), set(b.split())
    if at <= bt:
        return True, "every token of the proposed value is already present under this key"
    return False, ("token content differs (%d proposed tokens, %d new) -- may be new information"
                   % (len(at), len(at - bt)))

def store_lookup_container(mem_inst, key, container):
    """Read `key` from a SPECIFIC container only. Returns None if absent from that container.

    WHY THIS EXISTS -- a defect the C2 dry run caught before launch.

    `store_lookup` is container-AGNOSTIC: it scans core_memory then archival_memory and returns the
    first hit. That is safe for A3, which only ever matches `archival_memory_add`. It is WRONG once
    `core_memory_add` is matched, because the two containers have SEPARATE KEY NAMESPACES:

        archival_memory = {"role": "Managing Director"}      core_memory = {}
        core_memory_add(key="role", value="Managing Director")   -> WOULD SUCCEED

    The cross-container lookup called that redundant and would have suppressed a write that lands.
    The dry run measured 264 such cases and 184 fires on keys absent from the target container, then
    refused to launch. Suppressing a write the tool would have ACCEPTED is real data loss, which is
    exactly the failure mode the whole lossless-reclamation framing exists to avoid.
    """
    d = getattr(mem_inst, container, None)
    if isinstance(d, dict):
        inner = d.get("store")
        if isinstance(inner, dict):          # vector-shaped {"next_id": N, "store": {...}}
            d = inner
        if key in d:
            v = d[key]
            return v if isinstance(v, str) else str(v)
    return None


def should_suppress_kv(call_args, mem_inst, key_field="key", value_field="value",
                       container=None):
    """C2 predicate: canonical equivalence, core-inclusive, and CONTAINER-CORRECT.

    `container` names the container the call targets ("core_memory" / "archival_memory"). It is
    required in practice: without it the comparison can cross namespaces, which is data loss rather
    than suppression. When omitted the function REFUSES rather than guessing.
    """
    key = (call_args or {}).get(key_field)
    if not key:
        return False, "call carries no key -- nothing to compare against"
    if container is None:
        return False, ("no target container supplied -- refusing to compare across namespaces "
                       "(core and archival keys are independent)")
    proposed = (call_args or {}).get(value_field)
    stored = store_lookup_container(mem_inst, key, container)
    return is_redundant_canonical(proposed, stored)

# ---------------------------------------------------------------------------------------------------
# C3: FIX A3's CONTAINER-AGNOSTIC LOOKUP.
#
# Building C2 exposed a defect in A3 rather than only a gap. A3 is scoped to `archival_memory_add`, yet
# it MISSED 18 archival duplicates in the C2 run. Cause: `store_lookup` scans core_memory THEN
# archival_memory and returns the first hit, so when the same key exists in core with a DIFFERENT value,
# A3 compares the proposed archival value against the CORE value, sees a difference, and lets a genuine
# duplicate through:
#
#     core     = {"diabetes_medication": "short core summary"}
#     archival = {"diabetes_medication": "Metformin 1000 mg twice daily"}
#     proposed   archival_memory_add(key="diabetes_medication", value="Metformin 1000 mg twice daily")
#     A3 -> False ("values differ (18 vs 29 chars)")        <-- WRONG, it is an exact duplicate
#
# The defect is symmetric with the one caught pre-launch on C2: a wrong-container comparison can either
# suppress a write that would LAND (data loss) or fail to suppress one that is genuinely redundant
# (this). A3's accepted +2.64pp was measured with it present.
#
# This is kept SEPARATE from C2 so the two remedies stay attributable: C2 adds canonical equivalence and
# core coverage; C3 only makes A3 look in the container the call actually targets. Same rule, same
# whitespace-only normalisation, same exactness -- nothing about A3's semantics changes.
def should_suppress_exact_container(call_args, mem_inst, key_field="key", value_field="value",
                                   container=None):
    """A3's ORIGINAL rule -- normalised-exact only -- but reading the CORRECT container.

    Deliberately does NOT inherit C2's canonical equivalence. Mixing the two would make it impossible
    to tell whether a gain came from looking in the right place or from a looser comparator.
    """
    key = (call_args or {}).get(key_field)
    if not key:
        return False, "call carries no key -- nothing to compare against"
    if container is None:
        return False, ("no target container supplied -- refusing to compare across namespaces "
                       "(core and archival keys are independent)")
    proposed = (call_args or {}).get(value_field)
    stored = store_lookup_container(mem_inst, key, container)
    return is_redundant(proposed, stored)

