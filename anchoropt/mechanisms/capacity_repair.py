#!/usr/bin/env python3
"""capacity_repair -- ONE anchor covering all three backends, deterministic at inference.

The concept is shared: a write was REFUSED because the destination cannot hold it. Each
backend states that differently and admits a different repair, so CONDITION, CONSTRAINT and
VALIDATOR are all backend-specific data; the control flow is not.

    condition   the per-backend error strings that mean "cannot hold it"
    constraint  the schema-derived budget, and whether it is per-entry or aggregate
    repair      the operation that satisfies it, and how its argument is produced
    validator   the mechanical check that the repair is admissible

NO LLM AT INFERENCE. Every repair here is computed: copy an argument to another store, or
select a fact-preserving subset of the text until it fits. Generation is a BUILD-time tool for
writing this table, never a runtime dependency.

Two repair kinds, both deterministic:
    RELOCATE  the payload is fine, the destination is not -> re-address it to another store
    REDUCE    the destination is fine, the payload is not  -> shrink it, keeping the facts
"""
from __future__ import annotations

import re
from typing import Any

RELOCATE, REDUCE = "relocate", "reduce"

# Sentence split that keeps decimals and abbreviations intact: split on .!? only when
# followed by whitespace and a capital or digit.
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_NUM = re.compile(r"\d[\d,.]*")


def _nums(t: str) -> set:
    return {re.sub(r"[,.]$", "", x).replace(",", "") for x in _NUM.findall(t or "")}


def reduce_preserving_facts(text: str, budget: int) -> str | None:
    """Shrink `text` to <= budget while keeping every sentence that carries a number.

    Deterministic and auditable: sentences are ranked fact-carrying first, then by original
    order, and appended while they fit. Returns None when even the fact-carrying sentences
    cannot fit -- refusing is correct there, because dropping a number is information loss and
    the caller must fall through rather than write a lossy value.
    """
    if text is None:
        return None
    if len(text) <= budget:
        return text
    sents = [s for s in _SENT.split(text) if s.strip()]
    if not sents:
        return None
    keep = [s for s in sents if _NUM.search(s)]
    drop = [s for s in sents if not _NUM.search(s)]
    out, used = [], 0
    for s in keep:                      # facts first, in original order
        if used + len(s) + 1 > budget:
            return None                 # cannot keep all facts -> refuse
        out.append(s); used += len(s) + 1
    for s in drop:                      # then context, while it fits
        if used + len(s) + 1 > budget:
            break
        out.append(s); used += len(s) + 1
    joined = " ".join(sorted(out, key=lambda x: sents.index(x)))
    return joined if len(joined) <= budget else None


# ═══════════════════════════════════════════════════════════════════════════════
#  >>> THE BENCHMARK SEAM. Everything below this line is BFCL-v4-specific. <<<
#
#  To port this anchor, replace this TABLE and nothing else:
#    - `match`        your error strings
#    - `operation`    your destination tool name
#    - `carry` /
#      `reduce_arg`   your argument names
#    - `budget_from`  which of your capacity limits applies
#
#  The control flow beneath it (match_condition / budget_for / validate / repair) and
#  `reduce_preserving_facts` above are DATA-DRIVEN and domain-agnostic -- they read these values
#  and never interpret them. That split is the whole reason this file ports cheaply.
#
#  See docs/GENERALIZABILITY.md for the seams that are NOT this cheap.
# ═══════════════════════════════════════════════════════════════════════════════
# Values only. The control flow below reads them; it does not know what they mean.
BACKENDS: dict[str, dict[str, Any]] = {
    "kv": {
        "conditions": [
            {"match": ("is full", "exceeds maximum size"), "kind": RELOCATE,
             "operation": "archival_memory_add", "carry": ("key", "value"),
             "budget_from": "archival_entry"},
            {"match": ("Entry is too long", "exceeds maximum length"), "kind": REDUCE,
             "operation": "core_memory_add", "reduce_arg": "value",
             "budget_from": "core_entry"},
        ],
    },
    "vector": {
        "conditions": [
            {"match": ("is full", "exceeds maximum size"), "kind": RELOCATE,
             "operation": "archival_memory_add", "carry": ("text",),
             "budget_from": "archival_entry"},
            {"match": ("Entry length exceeds maximum length", "Entry is too long"),
             "kind": REDUCE, "operation": "core_memory_add", "reduce_arg": "text",
             "budget_from": "core_entry"},
        ],
    },
    "rec_sum": {
        # One blob, so there is nowhere to relocate TO: the only admissible repair is to
        # reduce the aggregate. The operation REPLACES rather than inserts, which is why a
        # "destination full" check must not veto it.
        "conditions": [
            {"match": ("will be too long after appending", "exceeds maximum"), "kind": REDUCE,
             "operation": "memory_update", "reduce_arg": "text",
             "budget_from": "aggregate", "replaces": True, "object": "store"},
        ],
    },
}


def match_condition(backend: str, error_text: str) -> dict[str, Any] | None:
    """The per-backend CONDITION test. Order matters: the first match wins, and the tables
    put RELOCATE before REDUCE so a capacity failure is not misread as a size failure."""
    for cond in (BACKENDS.get(backend or "", {}).get("conditions") or ()):
        if any(m in str(error_text) for m in cond["match"]):
            return cond
    return None


def budget_for(cond: dict[str, Any], caps: dict[str, int], state: dict[str, Any]) -> int | None:
    """The schema-derived CONSTRAINT, per backend.

    `aggregate` is the only one needing live state: a replacement write gets the whole cap,
    because it overwrites what is already there. Reading it from the error text instead would
    be wrong -- rec_sum's message says "shorten the entry to less than 10000" when the entry
    is 14 characters and the BLOB is what is full.
    """
    src = cond.get("budget_from")
    if src == "core_entry":
        return caps.get("ENTRY")
    if src == "archival_entry":
        return caps.get("ARCH_ENTRY")
    if src == "aggregate":
        return caps.get("BLOB")
    return None


def validate(cond: dict[str, Any], args: dict[str, str], budget: int | None,
             original: dict[str, str], state: dict[str, Any]) -> tuple[bool, str]:
    """The backend-specific VALIDATOR. Mechanical; no model output is trusted.

    Two rules learned the hard way, both from validators that rejected correct repairs:
      * a full destination blocks an INSERT, never a REPLACEMENT (`replaces`);
      * fact preservation checks NUMBERS only -- "Computer Science" -> "CS" is abbreviation,
        not information loss.
    """
    if budget is not None:
        payload = max((len(v) for v in args.values() if isinstance(v, str)), default=0)
        if payload > budget:
            return False, "payload %d exceeds budget %d" % (payload, budget)
    dest = cond.get("operation", "")
    if not cond.get("replaces") and dest in (state.get("full_operations") or []):
        return False, "destination for %s is full" % dest
    lost = sorted(_nums(" ".join(original.values())) - _nums(" ".join(args.values())))
    if lost:
        return False, "dropped numeric fact(s): %s" % lost[:5]
    return True, "ok"


def repair(backend: str, failing_args: dict[str, str], error_text: str,
           caps: dict[str, int], state: dict[str, Any],
           store_text: str | None = None) -> dict[str, Any] | None:
    """The shared control flow. Returns a repair record, or None to fall through.

        condition -> constraint -> is the argument available? -> repair -> validate
    """
    cond = match_condition(backend, error_text)
    if cond is None:
        return None
    budget = budget_for(cond, caps, state)
    if budget is None:
        return None

    if cond["kind"] == RELOCATE:
        # every carried argument must already be present: no invention
        if any(a not in failing_args for a in cond["carry"]):
            return None
        args = {a: failing_args[a] for a in cond["carry"]}
    else:
        field = cond["reduce_arg"]
        src = store_text if cond.get("object") == "store" else failing_args.get(field)
        if src is None:
            return None
        if cond.get("object") == "store" and failing_args.get(field):
            # the aggregate must hold the existing store AND the incoming fact
            src = (src + " " + failing_args[field]).strip()
        reduced = reduce_preserving_facts(src, budget)
        # An EMPTY result is a failure, not a success. reduce_preserving_facts returns "" when
        # the source carries no fact-bearing sentence at all, and the caller then built
        # `value=''` and reported ok=True -- writing an empty entry instead of refusing. That
        # is worse than the crash it replaced: a crash is visible, an empty write is not.
        if not reduced:
            return None
        # PRESERVE the other arguments. Replacing the whole argument set with just the
        # reduced field drops every sibling argument -- harmless for a single-argument
        # operation like memory_update(text=...), FATAL for core_memory_add(key=, value=):
        # the rebuilt call lost `key` and the executor raised
        #   "core_memory_add() missing 1 required positional argument: 'key'"
        # on every kv reduce. The harness guard caught it as a crash payload (8 steps in
        # 2 episodes) rather than it being scored as a model failure, which is exactly what
        # that guard exists for.
        #
        # When the constrained object is the STORE, the operation REPLACES the aggregate, so
        # only the reduced field is meaningful and siblings are deliberately not carried.
        if cond.get("object") == "store":
            args = {field: reduced}
        else:
            args = {**failing_args, field: reduced}

    original = dict(failing_args)
    if cond.get("object") == "store" and store_text:
        original = {**failing_args, "_store": store_text}
    ok, why = validate(cond, args, budget, original, state)
    if not ok:
        return {"ok": False, "reason": why, "kind": cond["kind"],
                "operation": cond["operation"]}
    body = ", ".join("%s=%r" % (k, v) for k, v in args.items())
    return {"ok": True, "kind": cond["kind"], "operation": cond["operation"],
            "call": "%s(%s)" % (cond["operation"], body), "budget": budget,
            "payload_len": max(len(v) for v in args.values())}
