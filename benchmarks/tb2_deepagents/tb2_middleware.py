"""The adapter-owned AnchorOpt middleware, and the minimal persisted controller format.

WHY THIS FILE EXISTS -- two guarantees a CANDIDATE must never be responsible for.

1. GRAPH ROUTING. LangChain only creates the conditional edge back to the model node when the
   hook method carries `__can_jump_to__`, which `@hook_config(can_jump_to=[...])` sets. Without
   it, returning `{"jump_to": "model"}` is **silently discarded**: the live smoke test executed
   the intervention, reported the jump, and no second generation happened. That is graph
   metadata, not semantics -- it cannot be expressed in `(l, phi, mu, theta)` -- so the
   middleware declares it once, here, for every controller.

2. EPISODE-SCOPED ONE-SHOT STATE. A `self.consumed` boolean is only safe if a fresh middleware
   instance is guaranteed per episode, and nothing in the TB2/Harbor path guarantees that. A
   latched flag would silently suppress the intervention in every later episode of a shared
   process -- an arm that stops intervening while still reporting as the candidate arm. So the
   guard is keyed on EPISODE IDENTITY derived from the conversation itself, and `reset()` is
   available for a lifecycle that can call it.

Nothing here is controller-specific: the middleware is constructed from a persisted
`ControllerSpec` and works for any (l, phi, mu, theta) the host can execute.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anchoropt.anchor import Action, IncisionPoint

import tb2_adapter as tb2

SPEC_FORMAT = "anchoropt.controller.v1"


# ------------------------------------------------------------------------------------------------
# The minimal persisted controller format
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ControllerSpec:
    """A saved AnchorOpt controller: exactly (l, phi, mu, theta) plus what is needed to load it.

    Deliberately carries NO task id, no benchmark concept, and nothing `verify_exact`-specific --
    `theta.text` is opaque payload. `signal_params` is present because a declared signal may
    require typed parameters; `controller_id` exists only so artifacts can be named and
    telemetry attributed.

    This is the smallest thing that round 1 needs. It is not a package system.
    """

    controller_id: str
    boundary: str                                   # l, IncisionPoint value
    signal: str                                     # phi, from the host's declared vocabulary
    action: str                                     # mu, an Action value
    theta: Mapping[str, Any] = field(default_factory=dict)
    signal_params: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"format": SPEC_FORMAT, "controller_id": self.controller_id,
                           "boundary": self.boundary, "signal": self.signal,
                           "action": self.action, "theta": dict(self.theta),
                           "signal_params": dict(self.signal_params)}, indent=2, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ControllerSpec:
        d = json.loads(Path(path).read_text())
        if d.get("format") != SPEC_FORMAT:
            raise ValueError(f"controller spec format must be {SPEC_FORMAT!r}, got {d.get('format')!r}")
        return cls(controller_id=str(d["controller_id"]), boundary=str(d["boundary"]),
                   signal=str(d["signal"]), action=str(d["action"]),
                   theta=dict(d.get("theta") or {}),
                   signal_params=dict(d.get("signal_params") or {}))

    # -- resolution into typed objects, validated against the host ------------------------------
    @property
    def l(self) -> IncisionPoint:  # noqa: E743
        return IncisionPoint(self.boundary)

    @property
    def mu(self) -> Action:
        return Action(self.action)

    def validate(self) -> None:
        """Fail loudly at LOAD time, not mid-episode.

        Three checks: the signal is declared by this host, its params satisfy the schema, and the
        action is executable at this boundary (`U_H(l)`). A spec that cannot run must not produce
        a middleware that silently no-ops.
        """
        if self.signal not in tb2.declared_signals():
            raise ValueError(f"signal {self.signal!r} not declared by {tb2.NAME}; "
                             f"declared: {list(tb2.declared_signals())}")
        tb2.evaluate_signal(self.signal, {}, self.signal_params)   # param-schema check
        tb2.HOST.require(self.l, self.mu)                          # U_H(l)


# ------------------------------------------------------------------------------------------------
# The middleware
# ------------------------------------------------------------------------------------------------
def build_middleware(spec: ControllerSpec, *, one_shot: bool = True):
    """Construct the LangChain middleware for `spec`. Imports LangChain lazily.

    `one_shot` bounds interventions to one per episode. Off means every firing intervenes, which
    for a REPROMPT that re-decides is an infinite loop -- so the default is on.
    """
    from langchain.agents.middleware import AgentMiddleware, hook_config
    from langchain_core.messages import HumanMessage

    spec.validate()

    class AnchorOptMiddleware(AgentMiddleware):
        """Routes the declared boundary through the adapter. One instance may serve many episodes."""

        def __init__(self) -> None:
            super().__init__()
            self.spec = spec
            self.one_shot = one_shot
            # Episode-scoped, NOT a bare boolean. Keyed on episode identity so a previous
            # episode's intervention cannot suppress the next one.
            self._intervened: set[str] = set()
            self.telemetry: dict[str, Any] = {
                "boundary_exposures": 0, "signal_firings": 0, "interventions_executed": 0,
                "loop_prevention_events": 0, "episodes_seen": 0, "errors": [],
            }

        def reset(self) -> None:
            """Explicit lifecycle reset, for a runner that prefers it to identity-keying."""
            self._intervened.clear()

        @staticmethod
        def _episode_key(messages) -> str:
            """Identity of the episode from its own first message.

            The first HumanMessage is the task instruction and is stable for the whole episode
            while later messages accumulate -- so it identifies the episode without the runner
            having to pass an id in.
            """
            for m in messages:
                if type(m).__name__ == "HumanMessage" or getattr(m, "type", "") == "human":
                    return str(getattr(m, "content", ""))[:200]
            return "<no-human-message>"

        # THE GUARANTEE: declared once, for every controller. A candidate never sees this.
        @hook_config(can_jump_to=["model"])
        def after_model(self, state, runtime=None):
            msgs = list((state or {}).get("messages") or ())
            self.telemetry["boundary_exposures"] += 1
            key = self._episode_key(msgs)
            try:
                rec = tb2.normalize_event({"boundary": IncisionPoint.POST_GENERATION_PRE_EXEC,
                                           "messages": msgs})
                st = tb2.observable_state(rec)
                fired = tb2.evaluate_signal(self.spec.signal, st, self.spec.signal_params)
            except Exception as exc:                        # never kill an episode on telemetry
                self.telemetry["errors"].append(f"{type(exc).__name__}: {exc}")
                return None
            if not fired:
                return None
            self.telemetry["signal_firings"] += 1

            if self.one_shot and key in self._intervened:
                self.telemetry["loop_prevention_events"] += 1
                return None

            try:
                directive = tb2.apply_action(self.spec.mu, self.spec.l, self.spec.theta, st)
            except Exception as exc:
                self.telemetry["errors"].append(f"{type(exc).__name__}: {exc}")
                return None
            if not directive.get("executed"):
                return None

            self.telemetry["interventions_executed"] += 1
            self._intervened.add(key)
            self.telemetry["episodes_seen"] = len(self._intervened)

            # The adapter chooses the CHANNEL from the semantic directive.
            update: dict[str, Any] = {"messages": [HumanMessage(content=directive["text"])]}
            if directive.get("request_redecision"):
                update["jump_to"] = "model"
            return update

    return AnchorOptMiddleware()
