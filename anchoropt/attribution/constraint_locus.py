#!/usr/bin/env python3
"""Semantic locus identity for candidate mining: canonicalize by VIOLATED CONSTRAINT.

The miner keys loci on `_canon_error` -- a digit-masked, lowercased error STRING. That makes
locus identity lexical, so aggregation happens wherever backends happen to share phrasing and
nowhere else. Measured on the A0+A1 corpus, one decision constraint fragmented across four
"candidates" purely by wording:

    core memory is full ...                 241  kv       core_memory_add
    memory size exceeds maximum size of N   168  vector   archival_memory_add / core_memory_add
    long term memory is full ...             62  kv       archival_memory_add
    entry will be too long after appending  341  rec_sum  memory_append

...while two genuinely different constraints shared a family resemblance ("capacity") and would
have been merged by any similarity heuristic:

    entry length exceeds maximum length      53  vector   core_memory_add   arg `text` oversized
    entry is too long ...                    29  kv       core_memory_add   arg `value` oversized

The rule is INTERVENTION EQUIVALENCE, not similarity: two failures share a locus iff they
violate the same decision constraint. Equivalence is decided on three DIAGNOSTIC facts:

    (constraint_type, constrained_resource, violation_mode)

DIAGNOSIS IS KEPT INDEPENDENT OF REMEDY, deliberately. An earlier draft keyed on the repair
kind (tool-change vs argument-change), which leaks policy into diagnosis: the locus would then
be defined by the intervention we happen to have chosen, so installing a different remedy would
silently redefine what counts as the same failure -- and ranking would no longer be comparable
across iterations. The remedy is selected LATER, per backend, by the adapter.

    (capacity, container, no_remaining_capacity)   <- the store cannot admit another item
    (size,     item,      exceeds_per_item_limit)  <- the item itself violates a per-item cap

These must NOT merge. In the first the item is valid and the container is exhausted; in the
second the container is fine and the payload is oversized. `test_negative_*` pins that.

BENCHMARK-AGNOSTIC BY CONSTRUCTION. This module contains no tool names, no backend names and
no numeric limits. It classifies from (a) evidence that a named resource is exhausted vs (b)
evidence that a measured quantity exceeded a stated bound, plus the observed call's arguments.
The lexical cues live in one table, `_CUES`, which is data an adapter can extend; everything
else is structural. `describe()` exists so a caller can log WHY two payloads merged.

Usage:
    from constraint_locus import constraint_locus
    key = constraint_locus(error_text, call_args=parsed_args)   # -> tuple | None
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# ── Vocabulary ───────────────────────────────────────────────────────────────
CAPACITY = "capacity"
SIZE = "size"
CONTAINER = "container"
ITEM = "item"
NO_REMAINING_CAPACITY = "no_remaining_capacity"
EXCEEDS_PER_ITEM_LIMIT = "exceeds_per_item_limit"

Locus = Tuple[str, str, str]

# Lexical CUES only -- the classification below is structural. Two independent families:
#
#   exhausted: a named resource has no room left. The item is not implicated.
#   over_bound: a measured quantity exceeded a stated bound. WHICH object is over the bound
#               is decided structurally (see _oversized_arg), because the wording alone is
#               ambiguous -- rec_sum says "entry will be too long" when the ENTRY is 169
#               chars against a 10000 cap and the BLOB is what is exhausted.
_CUES = {
    "exhausted": (
        "is full",
        "no space",
        "no room",
        "cannot admit",
        "at capacity",
        "exceeds maximum size",      # a SIZE-of-collection statement: how many items fit
    ),
    "over_bound": (
        "too long",
        "exceeds maximum length",
        "exceeds the maximum length",
        "longer than",
        "shorten",
    ),
}

# A per-item bound is only credible if some argument is actually near or over THE STATED
# BOUND. This must be RELATIVE, not absolute: an earlier draft used a fixed 200-char floor,
# which misfiled 341 events. Those messages state a 10000-char aggregate bound while carrying
# a 236-char item -- comfortably inside the bound, so the item is not the violator, yet 236
# cleared a fixed 200 floor and the locus was read as an item violation. Judging "large"
# without reference to the bound it is supposedly exceeding is meaningless.
#
# An argument counts as the violator only at >= this fraction of the stated bound.
_ITEM_BOUND_FRACTION = 0.5
# Used only when the message states no bound at all.
_MIN_PLAUSIBLE_ITEM_LEN = 200


def _norm(text: str) -> str:
    """Digit-masked, punctuation-stripped, lowercased -- for CUE MATCHING ONLY.

    Never used as an identity: that is the defect this module exists to fix.
    """
    s = re.sub(r"\d+", "N", text or "")
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _error_message(text: str) -> Optional[str]:
    """Pull the error payload, mirroring policy_tree._canon_error's admission rule.

    ALSO accepts an already-extracted message. The strict form -- requiring the `"error":`
    wrapper -- is right when scanning raw tool results, but it silently rejected messages a
    caller had already extracted, so every such observation classified as None and vanished
    from mining. That is precisely the failure this module was written to prevent, reproduced
    one layer up. The relaxed branch is last, so raw-payload behaviour is unchanged: a
    non-error tool result still has no constraint cue and still returns None from classify().
    """
    # Tool results arrive as dicts from the live evaluator and as strings from trajectory
    # sidecars. Accept both: requiring a string made the live path raise TypeError, and
    # `str(dict)` renders single quotes so the double-quoted regex below misses it anyway --
    # the capture recorded locus=None on every real capacity failure.
    if isinstance(text, dict):
        val = text.get("error")
        return str(val) if val else None
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    m = re.search(r'"error"\s*:\s*"([^"]+)"', text)
    if not m:
        # Single-quoted rendering of a dict, e.g. str({'error': 'Entry is too long.'}).
        m = re.search(r"'error'\s*:\s*'([^']+)'", text)
    if m:
        return m.group(1)
    s = (text or "").strip()
    if not s:
        return None
    if re.match(r"^\s*\[?\s*['\"]?(?:Error|Exception)\b", s, re.I):
        return s
    # Bare message: admit it only if it is not itself a structured payload. A JSON-looking
    # blob without an "error" key is a success result, not a message.
    if not s.startswith(("{", "[")):
        return s
    return None


def _stated_bound(norm_msg: str) -> Optional[int]:
    """The largest number the message states, as its declared bound.

    Read from the NORMALIZED message, where digits are masked to N, so the raw text has to be
    re-scanned by the caller. Largest rather than first: "shorten the entry to less than 300
    characters" states one number, while "exceeds maximum length of 300" may be preceded by an
    index or id, and the bound is the larger quantity in every observed phrasing.
    """
    nums = [int(x) for x in re.findall(r"\d+", norm_msg)]
    return max(nums) if nums else None


def _oversized_arg(call_args: Optional[Dict[str, str]],
                   bound: Optional[int] = None) -> Optional[str]:
    """Name of the argument plausibly violating a per-item bound, else None.

    STRUCTURAL, not lexical: it asks whether the call actually carried a payload big enough to
    be the violator OF THE STATED BOUND. This is what separates 'the item is too big' from
    'the container is exhausted and the error text happens to blame the item'.
    """
    if not call_args:
        return None
    floor = (int(bound * _ITEM_BOUND_FRACTION) if bound else _MIN_PLAUSIBLE_ITEM_LEN)
    big = [(k, v) for k, v in call_args.items()
           if isinstance(v, str) and len(v) >= floor]
    if not big:
        return None
    return max(big, key=lambda kv: len(kv[1]))[0]


def classify(error_text: str, call_args: Optional[Dict[str, str]] = None) -> Optional[Dict]:
    """Full diagnosis, with the reasoning attached. None if not a constraint violation."""
    msg = _error_message(error_text)
    if msg is None:
        return None
    norm = _norm(msg)

    exhausted = any(c in norm for c in _CUES["exhausted"])
    over_bound = any(c in norm for c in _CUES["over_bound"])
    if not (exhausted or over_bound):
        return None

    # The bound is read from the RAW message (digits intact), while cue matching uses the
    # masked form -- so a limit of 300 vs 10000 changes what counts as an oversized item.
    arg = _oversized_arg(call_args, _stated_bound(msg))

    # Order matters. An explicit exhaustion statement is unambiguous, so it wins outright.
    # An over-bound statement is only an ITEM violation when a large argument is actually
    # present; otherwise the bound being crossed belongs to the container (an aggregate),
    # and the message merely names the item that triggered the check.
    if exhausted and not (over_bound and arg):
        return {"locus": (CAPACITY, CONTAINER, NO_REMAINING_CAPACITY),
                "why": "resource reported exhausted", "oversized_arg": None}
    if over_bound and arg:
        return {"locus": (SIZE, ITEM, EXCEEDS_PER_ITEM_LIMIT),
                "why": "argument %r is %d chars" % (arg, len(call_args[arg])),
                "oversized_arg": arg}
    if over_bound:
        return {"locus": (CAPACITY, CONTAINER, NO_REMAINING_CAPACITY),
                "why": "bound crossed with no oversized argument -> aggregate constraint",
                "oversized_arg": None}
    return None


def constraint_locus(error_text: str,
                     call_args: Optional[Dict[str, str]] = None) -> Optional[Locus]:
    """The locus key, or None when the payload is not a constraint violation."""
    d = classify(error_text, call_args)
    return d["locus"] if d else None


def describe(error_text: str, call_args: Optional[Dict[str, str]] = None) -> str:
    """Human-readable diagnosis, for logging why two payloads merged."""
    d = classify(error_text, call_args)
    if not d:
        return "not a constraint violation"
    return "%s/%s/%s -- %s" % (d["locus"] + (d["why"],))
