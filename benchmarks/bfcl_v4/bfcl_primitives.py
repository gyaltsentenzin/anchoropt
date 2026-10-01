"""Generic executable primitives this benchmark offers. HOW only -- never WHEN.

    Controller = (locus, phi, action, eta)
    at locus:  state = observable_state();  if phi(state): execute(action, eta, state)

A primitive here implements the EXECUTE half and nothing else. It does not know what condition invoked
it, what threshold was involved, or why the caller thought it was a good idea. The learned/synthesized
phi owns WHEN, exclusively.

WHY THIS FILE EXISTS. The evaluator's existing read-merge path hard-codes its own arming condition
(`decline unless the observed score is below 0.30`), so a controller could be built on a synthesized
predicate and the executor would still apply its own trigger -- phi could not own WHEN, and two
predicates with different thresholds fired on identical states. That is not a tuning problem; it makes
the abstraction untestable, because the thing under test (does phi decide?) is decided elsewhere.

So these primitives take state and parameters, act, and return what they did. Every decision about
whether to act has already been made by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def additional_read_and_merge(destination: str, state: Mapping[str, Any], *,
                              top_k: int | None = None,
                              read_fn=None) -> dict[str, Any]:
    """Read `destination` with the episode's own query, merge into the returned payload by score.

    HOW, exhaustively; WHEN, not at all. There is no threshold, no notion of "weak", and no check on
    whether the merge seems worthwhile -- if the caller invoked this, the caller had already decided.

    `read_fn(destination, query, top_k) -> sequence of (score, text)` is injected by the host, so this
    function contains no tool names or transport of its own. Returns a record of what it did, including
    `fired=False` with a reason when it could not act, so an inert invocation is distinguishable from a
    refusal to invoke.
    """
    query = state.get("query") or state.get("last_query")
    if not query:
        return {"fired": False, "reason": "no query in state to reissue"}
    if read_fn is None:
        return {"fired": False, "reason": "host supplied no read function"}
    k = int(top_k or state.get("top_k") or 5)
    try:
        incoming = list(read_fn(destination, query, k) or [])
    except Exception as exc:
        return {"fired": False, "reason": f"read failed: {type(exc).__name__}: {exc}"}
    if not incoming:
        return {"fired": False, "reason": "destination returned nothing", "destination": destination}

    existing = list(state.get("scored_entries") or [])
    merged = sorted(list(existing) + list(incoming), key=lambda e: -float(e[0]))[:k]
    return {
        "fired": True,
        "destination": destination,
        "n_existing": len(existing),
        "n_incoming": len(incoming),
        "n_merged": len(merged),
        "best_before": (max((float(e[0]) for e in existing), default=None)),
        "best_after": (float(merged[0][0]) if merged else None),
        "merged": merged,
        # the original entries are RETAINED in the union; this is a merge, not a replacement
        "preserved_existing": True,
    }


def declared_primitives() -> dict[str, dict[str, Any]]:
    """What this adapter offers core, described by SIGNATURE and EFFECT -- not by when to use it."""
    return {
        "additional_read_and_merge": {
            "params": {"destination": "str: a readable surface name from the host's own schema",
                       "top_k": "int, optional: how many merged entries to return"},
            "effect": ("issues one read against `destination` using the episode's query, merges the "
                       "result with the entries already in state by score, and returns the union"),
            "preserves": "the entries already present are retained in the union",
            "decides_when": False,
        },
    }
