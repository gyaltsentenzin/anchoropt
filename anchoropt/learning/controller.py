"""The controller abstraction, and the only runtime semantics Self-Evolve needs.

    Controller = (locus, phi, action, eta)

    when locus is reached:
        state = observable_state()
        if phi(state):
            execute(action, eta, state)

That is the whole contract. Everything benchmark-specific lives behind two adapter-supplied callables:
`observable_state` and a primitive registry. Core names no observable, no tool, no tuning constant
and no storage surface; a test greps this module for domain vocabulary, including the words this
paragraph would otherwise use to disclaim them.

THE DIVISION, stated because it is what went wrong before this module existed:

    CORE     attribution, phi, locus, action + eta, evaluation, promote/reject
    ADAPTER  typed observable state, generic executable primitives, scoring/environment mechanics

The failure this fixes: the host's read-merge path hard-coded its own arming condition, so the executor
decided WHEN and phi could not. Two synthesized predicates parameterised differently then fired on
IDENTICAL states -- which makes the abstraction untestable, because the property under test (does phi
decide?) was being decided elsewhere. A primitive implements HOW; phi owns WHEN, exclusively.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Predicate(Protocol):
    """phi. Anything that can answer a yes/no question about observable state."""

    def evaluate(self, state: Mapping[str, Any]) -> bool:
        ...


@dataclass(frozen=True)
class Controller:
    """(locus, phi, action, eta) -- the optimizer's output, executable as written."""

    locus: Any
    phi: Predicate
    action: str
    eta: Mapping[str, Any] = field(default_factory=dict)
    provenance: str = ""

    @property
    def name(self) -> str:
        sig = getattr(self.phi, "name", None)
        sig = sig() if callable(sig) else (sig or "phi")
        return f"{self.action}@{getattr(self.locus, 'value', self.locus)}[{sig}]"

    def fires_on(self, state: Mapping[str, Any]) -> bool:
        """WHEN. The only place that question is answered."""
        return bool(self.phi.evaluate(state))

    def as_dict(self) -> dict[str, Any]:
        desc = getattr(self.phi, "describe", None)
        return {"locus": getattr(self.locus, "value", str(self.locus)),
                "phi": desc() if callable(desc) else str(self.phi),
                "action": self.action,
                "eta": {k: str(v) for k, v in dict(self.eta).items()},
                "provenance": self.provenance}


@dataclass(frozen=True)
class Firing:
    """One evaluation of a controller against one state, and what the primitive did."""

    fired: bool
    state_index: int
    effect: Mapping[str, Any] | None = None
    reason: str = ""


def run_controller(controller: Controller,
                   states: Sequence[Mapping[str, Any]],
                   primitives: Mapping[str, Callable[..., Mapping[str, Any]]],
                   *, dry_run: bool = False) -> list[Firing]:
    """Apply the runtime semantics above to a sequence of observed states.

    `primitives` is the adapter's registry: name -> callable implementing HOW. Core looks up
    `eta["primitive"]`, passes the declared parameters and the state, and records the result. It does
    not inspect what the primitive did or second-guess whether it should have run.

    A primitive that is missing is reported, never simulated -- an unimplemented HOW must not look like
    a controller that chose not to act.
    """
    out: list[Firing] = []
    pname = str(dict(controller.eta).get("primitive") or "")
    params = {k: v for k, v in dict(controller.eta).items() if k != "primitive"}
    for i, state in enumerate(states):
        if not controller.fires_on(state):
            out.append(Firing(False, i, None, "phi did not hold"))
            continue
        if dry_run:
            out.append(Firing(True, i, None, "dry run: phi held, primitive not invoked"))
            continue
        fn = primitives.get(pname)
        if fn is None:
            out.append(Firing(True, i, None,
                              f"phi held but no primitive named {pname!r} is available"))
            continue
        try:
            effect = fn(state=state, **params)
        except TypeError as exc:
            out.append(Firing(True, i, None, f"primitive signature mismatch: {exc}"))
            continue
        out.append(Firing(True, i, dict(effect or {}), ""))
    return out


def firing_set(controller: Controller, states: Sequence[Mapping[str, Any]]) -> set[int]:
    """Indices where phi holds. The comparable object when asking whether two controllers differ."""
    return {i for i, s in enumerate(states) if controller.fires_on(s)}
