"""A signal the optimizer LEARNED: its predicate, where it was validated, and where it came from.

    LearnedSignal = (predicate, boundaries validated at, provenance, firing evidence)

WHY A WRAPPER AND NOT `CompiledSignal`. `signal_lang.CompiledSignal` is the right idea for the
expression language it belongs to -- it already pairs a predicate with the boundaries it is
observable at -- but it is built only by `compile_signal` from a JSON expression tree and its
predicate takes `(state, params)`. The grammar's `Atom`/`Conjunction` carry no expression tree and
evaluate on `(state)` alone. Making them fit would mean synthesizing a fake expr and a fake params
argument to satisfy a constructor, which distorts a type to avoid a dataclass. So this wraps, and
`from_compiled` adapts a `CompiledSignal` in when one is what you have.

THE GAP THIS CLOSES. The triple (predicate, boundary, provenance) used to exist only as three
keyword arguments in flight to `runtime.install_signal(...)`, which left it to each adapter to decide
what to keep. One adapter kept the predicate and dropped the boundary, so asking "where is this
signal observable?" had no answer for exactly the signals expansion creates -- and the caller that
asked caught the resulting error and substituted "nowhere", which reads like a considered answer.
A learned signal is one object, and the boundary it was VALIDATED at travels with it.

`boundaries` means validated-here, not declared-here. That distinction is the whole point: a
synthesized predicate has no declaration to appeal to, so the only honest basis for claiming it is
observable somewhere is that it was checked there. An empty set is therefore meaningful and is not a
bug -- it says nobody has established anywhere this signal can be read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def _boundary_value(b: Any) -> str:
    return str(getattr(b, "value", b))


@dataclass(frozen=True)
class LearnedSignal:
    """A predicate plus the metadata that makes it usable: where it holds, and why it exists."""

    name: str
    predicate: Any                                  # anything with .evaluate(state) -> bool
    boundaries: frozenset[str] = frozenset()        # boundaries it was VALIDATED at
    provenance: str = ""
    fired_on: int = 0                               # firing evidence from validation
    total_states: int = 0
    source: str = "synthesized"                     # synthesized | compiled | declared

    def evaluate(self, state: Mapping[str, Any]) -> bool:
        """Duck-typed like every other predicate in core, so existing consumers accept this."""
        return bool(self.predicate.evaluate(state))

    def observable_at(self, boundary: Any) -> bool:
        """Was this signal validated at `boundary`? Same question `CompiledSignal` answers."""
        return _boundary_value(boundary) in self.boundaries

    def describe(self) -> str:
        inner = getattr(self.predicate, "describe", None)
        return inner() if callable(inner) else str(self.predicate)

    @property
    def firing_rate(self) -> float | None:
        return (self.fired_on / self.total_states) if self.total_states else None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "predicate": self.describe(),
                "boundaries": sorted(self.boundaries), "provenance": self.provenance,
                "fired_on": self.fired_on, "total_states": self.total_states,
                "firing_rate": self.firing_rate, "source": self.source}


def learned_from(predicate: Any, *, name: str = "", boundaries: Sequence[Any] = (),
                 provenance: str = "", fired_on: int = 0, total_states: int = 0,
                 source: str = "synthesized") -> LearnedSignal:
    """Wrap a bare predicate, taking its own name when one was not supplied."""
    if not name:
        got = getattr(predicate, "name", None)
        name = got() if callable(got) else str(got or "phi")
    return LearnedSignal(name=str(name), predicate=predicate,
                         boundaries=frozenset(_boundary_value(b) for b in boundaries),
                         provenance=str(provenance or ""), fired_on=int(fired_on),
                         total_states=int(total_states), source=str(source))


def from_compiled(compiled: Any, *, params: Mapping[str, Any] | None = None,
                  provenance: str = "") -> LearnedSignal:
    """Adapt a `signal_lang.CompiledSignal` in, binding its `params` argument.

    `CompiledSignal.predicate` takes `(state, params)`; everything downstream here expects
    `.evaluate(state)`. Binding the params is the adaptation, and it is done once, here, rather than
    by teaching every consumer about a second predicate arity.
    """
    bound = dict(params or {})

    class _Bound:
        def evaluate(self, state: Mapping[str, Any]) -> bool:
            return bool(compiled.predicate(state, bound))

        def describe(self) -> str:
            return f"{compiled.name} over {sorted(compiled.fields_used)}"

    return LearnedSignal(name=str(compiled.name), predicate=_Bound(),
                         boundaries=frozenset(compiled.boundaries),
                         provenance=provenance or f"compiled from {dict(compiled.expr)!r}",
                         source="compiled")


# ---- behavioural identity ------------------------------------------------------------------------
#
# A signal IS its firing behaviour on the states that matter. Two syntactically different predicates
# that fire on exactly the same states are the same signal for every purpose the optimizer has, and
# an "expansion" that adds one of them has added no information.
#
# `controller.firing_set` already computes this index set for a Controller, and `realization.
# verify_projection` computes it inline for two predicates in order to compare them. This reuses the
# same notion for a bare predicate so all three agree, rather than adding a fourth way to ask.


def firing_vector(predicate: Any, states: Sequence[Mapping[str, Any]]) -> tuple[bool, ...]:
    """F_phi = (phi(s1), ..., phi(sn)). The comparable object, positionally aligned to `states`."""
    out = []
    for s in states:
        try:
            out.append(bool(predicate.evaluate(s)))
        except Exception:
            # A predicate that raises on a state has not fired there. Recorded as False rather than
            # propagated: a raising phi is not a trigger, which is the rule the runtime hook follows.
            out.append(False)
    return tuple(out)


def fingerprint(predicate: Any, states: Sequence[Mapping[str, Any]]) -> frozenset[int]:
    """The trigger set as indices -- the same object `controller.firing_set` returns."""
    return frozenset(i for i, fired in enumerate(firing_vector(predicate, states)) if fired)


def behaviourally_identical(a: Any, b: Any, states: Sequence[Mapping[str, Any]]) -> bool:
    """Do these two predicates fire on exactly the same observed states?"""
    return firing_vector(a, states) == firing_vector(b, states)


def dedupe_by_behaviour(predicates: Sequence[Any],
                        states: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Keep one representative per distinct firing vector, in the order given.

    Syntactic dedup cannot do this: a threshold grid over one field yields many names and, on a small
    state sample, often only a handful of distinct behaviours. Searching the duplicates costs
    evaluations that cannot differ in outcome.
    """
    seen: set[tuple[bool, ...]] = set()
    out: list[Any] = []
    for p in predicates:
        fv = firing_vector(p, states)
        if fv in seen:
            continue
        seen.add(fv)
        out.append(p)
    return out


def adds_information(candidate: Any, existing: Sequence[Any],
                     states: Sequence[Mapping[str, Any]]) -> bool:
    """Would adding `candidate` to Phi distinguish any state the current Phi cannot?

    A constant predicate adds nothing (it separates no states), and neither does one whose firing
    vector some existing signal already realizes. This is how "did expansion actually expand Phi"
    gets a deterministic answer instead of an argument about naming.
    """
    fv = firing_vector(candidate, states)
    if len(set(fv)) < 2:                      # constant on the observed states: separates nothing
        return False
    return all(fv != firing_vector(e, states) for e in existing)
