"""Backward search over the decision boundaries the trajectory actually contains.

    trajectory -> ordered consequential decision boundaries -> search backward over them

    for b in reverse(boundaries):
        repairable here? -> expressible under Phi? -> (expand Phi if blocked)
          -> Controller(locus=b, phi, action, eta) -> realizable by a generic primitive? -> evaluate
        promoted  -> regenerate, remine, RESTART from the latest boundary
        exhausted -> move to the previous boundary

WHY BOUNDARIES AND NOT SEMANTIC STAGES. An earlier version took an adapter-declared ordering of named
phases as an algorithm input. That is hand-engineering: the order was chosen by a human who already knew
which failures mattered, and every new benchmark would have had to invent its own taxonomy before the
optimizer could run. The ordering is already IN the trajectory -- decisions happen in
sequence -- so core derives it and needs no vocabulary for what the decisions mean.

Semantic labels remain useful for reading results, and an adapter may still supply them; `label_for`
is consulted only when present and only for reporting. Nothing in the search depends on it.

WHY BACKWARD. A failure observed late may be repairable late, and a repair there is better attributed
than one applied earlier: the earlier intervention must be right about everything that happens after it,
while the later one sees the actual failure. So the search moves earlier only when a boundary is
exhausted.

ORDER COMES FROM THE TRAJECTORY, WITH ONE ESCAPE HATCH. `boundaries_of` reads the realized events in
order. If temporal order is insufficient for some host, the adapter may expose a generic `depends_on`
relation and core will topologically sort by it -- a dependency relation, not a semantic taxonomy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

BOUNDARY_PROMOTED = "BOUNDARY_PROMOTED"
BOUNDARY_EXHAUSTED = "BOUNDARY_EXHAUSTED"
BOUNDARY_NOT_REPAIRABLE = "BOUNDARY_NOT_REPAIRABLE"
BOUNDARY_PRIMITIVE_MISSING = "BOUNDARY_PRIMITIVE_MISSING"
BOUNDARY_NO_CANDIDATE = "BOUNDARY_NO_CANDIDATE"


@dataclass(frozen=True)
class Boundary:
    """One consequential decision point in a realized trajectory.

    `key` identifies the decision kind at this point (whatever the adapter calls it); `index` is its
    position in the realized order. `label` is descriptive only -- the search never consults it.
    """

    key: str
    index: int
    label: str = ""

    def __str__(self) -> str:
        return f"{self.index}:{self.key}" + (f" ({self.label})" if self.label else "")


def boundaries_of(events: Sequence[Mapping[str, Any]], *, adapter: Any = None) -> list[Boundary]:
    """Ordered decision boundaries derived FROM the trajectory.

    A boundary is a point where the agent made a consequential choice. Core asks the adapter which
    events those are (`is_decision`) and what to call the point (`boundary_key`); absent either, it
    falls back to every event that carries a boundary field, in realized order.

    Duplicates collapse to their FIRST occurrence: a boundary kind that recurs is one decision point
    visited repeatedly, not several distinct places to intervene -- searching it once per occurrence
    would multiply the same candidate set by the trajectory length.
    """
    is_decision = getattr(adapter, "is_decision", None)
    key_of = getattr(adapter, "boundary_key", None)
    label_of = getattr(adapter, "label_for", None)
    out: list[Boundary] = []
    seen: set[str] = set()
    for i, ev in enumerate(events or []):
        if is_decision is not None and not is_decision(ev):
            continue
        key = str(key_of(ev)) if key_of is not None else str(ev.get("boundary") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(Boundary(key=key, index=i,
                            label=(str(label_of(key)) if label_of is not None else "")))
    return out


def order_boundaries(boundaries: Sequence[Boundary], *, adapter: Any = None) -> list[Boundary]:
    """Earliest-to-latest. Temporal order by default; a generic dependency relation if one is offered.

    `adapter.depends_on(key) -> iterable of keys that must precede it` is the escape hatch for a host
    whose realized order does not reflect dependence. It is a DEPENDENCY relation, deliberately not a
    named taxonomy: it says what comes before what, never what a stage means.
    """
    depends_on = getattr(adapter, "depends_on", None)
    if depends_on is None:
        return sorted(boundaries, key=lambda b: b.index)

    by_key = {b.key: b for b in boundaries}
    resolved: list[Boundary] = []
    placed: set[str] = set()
    remaining = sorted(boundaries, key=lambda b: b.index)
    # Kahn-style, falling back to realized order whenever a dependency is unsatisfiable (a cycle, or a
    # prerequisite outside this trajectory). A scheduler that deadlocks on a malformed relation is worse
    # than one that degrades to the order the events actually happened in.
    progress = True
    while remaining and progress:
        progress = False
        for b in list(remaining):
            preds = {str(k) for k in (depends_on(b.key) or ()) if str(k) in by_key}
            if preds <= placed:
                resolved.append(b)
                placed.add(b.key)
                remaining.remove(b)
                progress = True
    return resolved + sorted(remaining, key=lambda b: b.index)


@dataclass
class BoundaryAttempt:
    """What happened at one boundary, in enough detail to explain the schedule afterwards."""

    boundary: str
    state: str
    label: str = ""
    signal_blocked: int = 0
    signals_expanded: tuple[str, ...] = ()
    controllers_built: int = 0
    controllers_evaluated: int = 0
    primitive_missing: tuple[str, ...] = ()
    promoted: Any | None = None
    objective: Any | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"boundary": self.boundary, "label": self.label, "state": self.state,
                "signal_blocked": self.signal_blocked,
                "signals_expanded": list(self.signals_expanded),
                "controllers_built": self.controllers_built,
                "controllers_evaluated": self.controllers_evaluated,
                "primitive_missing": list(self.primitive_missing),
                "promoted": (self.promoted.as_dict()
                             if hasattr(self.promoted, "as_dict") else self.promoted),
                "objective": self.objective, "detail": self.detail}


@dataclass
class SearchResult:
    attempts: list[BoundaryAttempt] = field(default_factory=list)
    promoted: Any | None = None
    restarts: int = 0
    boundaries: tuple[str, ...] = ()

    @property
    def visited(self) -> tuple[str, ...]:
        return tuple(a.boundary for a in self.attempts)

    @property
    def moves_earlier(self) -> int:
        return sum(1 for a in self.attempts
                   if a.state in (BOUNDARY_EXHAUSTED, BOUNDARY_NOT_REPAIRABLE,
                                  BOUNDARY_NO_CANDIDATE, BOUNDARY_PRIMITIVE_MISSING))

    def as_dict(self) -> dict[str, Any]:
        return {"boundaries": list(self.boundaries), "visited": list(self.visited),
                "attempts": [a.as_dict() for a in self.attempts],
                "promoted": (self.promoted.as_dict()
                             if hasattr(self.promoted, "as_dict") else self.promoted),
                "restarts": self.restarts, "moves_earlier": self.moves_earlier}


def backward_boundary_search(residual: Any, *, boundaries: Sequence[Boundary],
                             hooks: Mapping[str, Callable],
                             max_restarts: int = 3) -> SearchResult:
    """Search the boundaries latest-first. Signal expansion is nested inside each boundary's turn.

    Hooks, each already implemented elsewhere -- this orders them and adds nothing:
        repairable_at(residual, boundary)      -> bool
        expressible(residual, boundary)        -> (bool, reason)
        expand(residual, boundary)             -> newly available signal names   [optional]
        build(residual, boundary, signals)     -> sequence of Controllers
        realizable(controller)                 -> (bool, primitive_or_reason)
        evaluate(controller)                   -> objective, higher better, or None if unmeasurable
        improves(objective)                    -> bool
        promote(controller)                    -> None
        remine()                               -> the new residual, or None        [optional]
    """
    ordered = list(boundaries)
    result = SearchResult(boundaries=tuple(b.key for b in ordered))
    current = residual
    restarts = 0
    i = len(ordered) - 1                       # the LATEST boundary first

    while i >= 0:
        b = ordered[i]
        att = BoundaryAttempt(boundary=b.key, label=b.label, state=BOUNDARY_EXHAUSTED)

        if not hooks["repairable_at"](current, b):
            att.state = BOUNDARY_NOT_REPAIRABLE
            att.detail = "the residual does not manifest at this decision point"
            result.attempts.append(att)
            i -= 1
            continue

        ok, why = hooks["expressible"](current, b)
        signals: tuple[str, ...] = ()
        if not ok:
            att.signal_blocked += 1
            att.detail = f"SIGNAL_BLOCKED: {why}"
            expand = hooks.get("expand")
            if expand is None:
                result.attempts.append(att)
                i -= 1
                continue
            signals = tuple(expand(current, b) or ())
            att.signals_expanded = signals
            if not signals:
                att.detail += " | expansion produced nothing"
                result.attempts.append(att)
                i -= 1
                continue

        controllers = list(hooks["build"](current, b, signals) or ())
        att.controllers_built = len(controllers)
        if not controllers:
            att.state = BOUNDARY_NO_CANDIDATE
            att.detail = (att.detail + " | no controller could be built").strip(" |")
            result.attempts.append(att)
            i -= 1
            continue

        promoted, missing = None, []
        better = hooks.get("better", lambda a, c: a > c)
        for c in controllers:
            can, info = hooks["realizable"](c)
            if not can:
                missing.append(str(info))
                continue
            obj = hooks["evaluate"](c)
            att.controllers_evaluated += 1
            if obj is None:
                continue
            if att.objective is None or better(obj, att.objective):
                att.objective = obj
            if hooks["improves"](obj):
                promoted = c
                break
        att.primitive_missing = tuple(dict.fromkeys(missing))

        if promoted is not None:
            hooks["promote"](promoted)
            att.state, att.promoted = BOUNDARY_PROMOTED, promoted
            result.attempts.append(att)
            result.promoted = promoted
            nxt = hooks["remine"]() if "remine" in hooks else None
            if nxt is None or restarts >= max_restarts:
                break
            # RESTART AT THE LATEST BOUNDARY: the promotion changed the trajectories, so every decision
            # after it faces a different residual now. Continuing earlier would search a stale problem.
            current = nxt
            restarts += 1
            result.restarts = restarts
            i = len(ordered) - 1
            continue

        if missing and att.controllers_evaluated == 0:
            att.state = BOUNDARY_PRIMITIVE_MISSING
            att.detail = (att.detail + f" | no generic primitive realizes: {missing[:3]}").strip(" |")
        result.attempts.append(att)
        i -= 1

    return result
