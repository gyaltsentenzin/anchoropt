"""The one generic runtime hook: let an externally supplied phi decide WHEN, at any locus.

    Controller = (locus, phi, action, eta)
    at locus:  state = observable_state();  if phi(state): execute(action, eta, state)

A host wires this ONCE per locus it wants to expose. Thereafter every Self-Evolve controller -- present
and future, any signal, any benchmark -- reaches that locus with no further host change. That is why
this is infrastructure rather than tuning for one candidate: the alternative is a bespoke gate per
controller, which is how a host ends up owning the trigger and silently overriding phi.

WHAT THE HOOK REPLACES. A host that hard-codes its own arming condition inside an executor cannot accept
an externally supplied trigger: two predicates parameterised differently then fire on identical states,
and the optimizer is varying something the host ignores. Measured on one host: its merge path armed on a
module constant occurring once, with no policy path, so three synthesized thresholds collapsed to one.

THE CONTRACT, deliberately narrow:
  * the hook decides ONLY whether to proceed. It performs no action and mutates no state.
  * with no controller installed it returns the host's own decision UNCHANGED, so wiring it is a no-op
    until something is installed -- a host can adopt it without changing any existing measurement.
  * a controller whose phi raises is treated as NOT FIRING, and the error is recorded. A predicate that
    throws must never be read as a trigger.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Installed controllers, keyed by the locus they act at. Empty by default: a host that wires the hook
# and installs nothing behaves exactly as before.
_INSTALLED: dict[str, list[Any]] = {}
_ERRORS: list[str] = []


def install(locus: str, controller: Any) -> None:
    """Register a controller at `locus`. Core-side only; no host state is touched."""
    _INSTALLED.setdefault(str(locus), []).append(controller)


def installed(locus: str | None = None) -> tuple[Any, ...]:
    if locus is None:
        return tuple(c for cs in _INSTALLED.values() for c in cs)
    return tuple(_INSTALLED.get(str(locus), ()))


def reset() -> None:
    """Drop every installed controller. Used between arms so one cannot inherit another's policy."""
    _INSTALLED.clear()
    _ERRORS.clear()


def errors() -> tuple[str, ...]:
    return tuple(_ERRORS)


@dataclass(frozen=True)
class HookDecision:
    """Whether to proceed at a locus, and which controller said so."""

    proceed: bool
    controller: Any | None = None
    eta: Mapping[str, Any] = field(default_factory=dict)
    source: str = "host"          # "host" when no controller is installed

    @property
    def by_controller(self) -> bool:
        return self.source == "controller"


def decide(locus: str, state: Mapping[str, Any], *, host_default: bool) -> HookDecision:
    """THE HOOK. One call per locus in the host.

    Returns `host_default` unchanged when no controller is installed at this locus, so adopting the
    hook changes nothing until a controller exists. When one is installed, its phi -- and only its phi --
    decides.
    """
    controllers = _INSTALLED.get(str(locus), ())
    if not controllers:
        return HookDecision(bool(host_default), None, {}, "host")
    for c in controllers:
        try:
            fired = bool(c.fires_on(state))
        except Exception as exc:
            _ERRORS.append(f"{getattr(c, 'name', 'controller')}: "
                           f"{type(exc).__name__}: {exc}")
            continue                  # a raising predicate is NOT a trigger
        if fired:
            return HookDecision(True, c, dict(getattr(c, "eta", {}) or {}), "controller")
    return HookDecision(False, None, {}, "controller")
