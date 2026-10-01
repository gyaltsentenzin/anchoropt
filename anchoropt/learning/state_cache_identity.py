"""What must go into a cache key for STATE that an intervention can change.

THE DEFECT THIS ANSWERS
-----------------------
A host cached the state its scored work is graded against, keying the cache on the corpus, the model,
the policy flags and a code fingerprint -- but NOT on the intervention installed while that state was
built. Two arms running different controllers therefore computed the SAME key. Measured: arm and
control both produced base_key c1f8885511077611 with different controllers installed. Nothing went
wrong only because the harness happened to give each arm its own cache directory; the key itself was
blind, and a single shared directory would have served one arm's state to the other silently.

That is the failure mode: not a crash, not a wrong number that looks wrong, but one arm scored
against the other arm's state, with a cache "hit" logged as success.

THE RULE
--------
A cache key for mutable state must cover everything that can CHANGE that state. An intervention that
acts while state is being built changes it by definition, so its identity belongs in the key. State
that no intervention touched, and state that an intervention repaired, are not the same state, and a
key that cannot tell them apart is not an identity.

The converse matters just as much, or the fix would destroy the thing it protects. An intervention
that acts only AFTER state is built cannot have changed it, so folding its identity into the key
would force a needless rebuild and defeat caching for every downstream controller. So the key covers
state-affecting interventions ONLY, and the phase is what decides which those are.

DELIBERATE SHARING IS DIFFERENT FROM A BLIND COLLISION
-----------------------------------------------------
Two arms sometimes SHOULD read one store -- holding construction constant is how a delayed-effect
controller's query-time effect is isolated at all. That is legitimate, and it is not what this module
prevents. The difference is that deliberate sharing is *declared*: the caller asks for it, by name,
and the record says so. A blind collision is the same reuse with nothing recording that it happened.
`shared_state_key` exists so the honest case has a way to say what it is doing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Phases whose interventions can change state a later phase reads. A controller declared for one of
#: these is state-affecting and MUST enter the key. Hosts name their phases differently, so the
#: caller passes its own set; these are the defaults for a build-then-score shape.
STATE_BUILDING_PHASES = frozenset({"prereq", "setup", "build", "warm", "construct", "any"})


def _canon(value: Any) -> str:
    """Stable text for a value, order-insensitive for mappings and sets, so a key is reproducible."""
    if isinstance(value, Mapping):
        return "{" + ",".join(f"{k}:{_canon(value[k])}" for k in sorted(map(str, value))) + "}"
    if isinstance(value, (set, frozenset)):
        return "[" + ",".join(sorted(_canon(v) for v in value)) + "]"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canon(v) for v in value) + "]"
    return json.dumps(value, sort_keys=True, default=str)


@dataclass(frozen=True)
class StateKeyInputs:
    """Everything that can change the cached state, and nothing that cannot.

    `controller` is the installed intervention's identity -- whatever the host uses to distinguish one
    from another (a spec dict, a cell tuple, a name). `controller_phase` is the phase it was declared
    for, and it is what decides whether the controller enters the key at all.
    """

    base: Mapping[str, Any]
    controller: Any = None
    controller_phase: str | None = None
    state_building_phases: frozenset[str] = STATE_BUILDING_PHASES

    @property
    def controller_affects_state(self) -> bool:
        """Can the installed intervention change the state being cached?

        An installed controller with NO declared phase is treated as state-affecting. That is the
        conservative direction on purpose: the cost of being wrong is a needless rebuild, whereas the
        cost in the other direction is one arm scored against another arm's state.
        """
        if self.controller is None:
            return False
        if self.controller_phase is None:
            return True
        return str(self.controller_phase).lower() in self.state_building_phases


@dataclass(frozen=True)
class StateKey:
    """The key, plus why it came out the way it did. The explanation is not decoration: a cache that
    silently reuses state is only auditable if the key can say what it covered."""

    key: str
    covers_controller: bool
    controller_identity: str | None
    reason: str
    shared_declared: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "covers_controller": self.covers_controller,
                "controller_identity": self.controller_identity, "reason": self.reason,
                "shared_declared": self.shared_declared}


def state_cache_key(inputs: StateKeyInputs, *, digest_size: int = 16) -> StateKey:
    """Key the cached state on the base inputs AND on any intervention that can change it."""
    parts = [_canon(inputs.base)]
    ident: str | None = None
    covers = inputs.controller_affects_state
    if covers:
        ident = _canon(inputs.controller)
        parts.append("controller=" + ident)
        why = (f"controller declared phase {inputs.controller_phase!r} can change the cached state, "
               f"so its identity is part of that state's identity"
               if inputs.controller_phase is not None else
               "an installed controller with NO declared phase is treated as state-affecting; a "
               "needless rebuild is the cheap error, cross-arm reuse the expensive one")
    elif inputs.controller is None:
        why = "no controller installed; the base inputs fully determine the state"
    else:
        why = (f"controller declared phase {inputs.controller_phase!r} acts only after the state is "
               f"built, so it cannot have changed it and must NOT enter the key -- folding it in "
               f"would force a rebuild per downstream controller and defeat caching")
    return StateKey(hashlib.sha256("|".join(parts).encode()).hexdigest()[:digest_size],
                    covers, ident, why)


def shared_state_key(inputs: StateKeyInputs, *, declared_by: str,
                     digest_size: int = 16) -> StateKey:
    """A key that DELIBERATELY ignores the controller, so two arms read one store.

    This is the legitimate case the strict key would otherwise block: to measure a delayed-effect
    controller's query-time effect, construction must be held constant, which means both arms reading
    identical state. `declared_by` is required and is recorded in the reason -- the point of this
    function is that deliberate sharing leaves a trace, which is exactly what a blind collision does
    not do. Never reach for it to silence an unexpected rebuild.
    """
    if not declared_by:
        raise ValueError("shared_state_key requires declared_by: sharing state across arms must be "
                         "attributable to a protocol that asked for it")
    base = state_cache_key(StateKeyInputs(inputs.base), digest_size=digest_size)
    return StateKey(base.key, False, _canon(inputs.controller) if inputs.controller else None,
                    f"controller DELIBERATELY excluded so arms share one store, declared by "
                    f"{declared_by!r}: construction is held constant so the paired delta isolates "
                    f"the post-construction effect", shared_declared=True)


def collision_report(keys_by_arm: Mapping[str, StateKey]) -> dict[str, Any]:
    """Do two arms with DIFFERENT controllers share a key, and was that declared?

    Returns `blind_collisions` -- the failure -- separately from `declared_sharing`, which is a
    protocol choice. A consumer that treats every shared key as a fault would forbid the matched
    construction design; one that treats none as a fault is blind in the original way.
    """
    by_key: dict[str, list[str]] = {}
    for arm, sk in keys_by_arm.items():
        by_key.setdefault(sk.key, []).append(arm)
    blind, declared = [], []
    for key, arms in sorted(by_key.items()):
        if len(arms) < 2:
            continue
        idents = {keys_by_arm[a].controller_identity for a in arms}
        if len(idents) < 2:
            continue  # same controller, same key: correct reuse
        (declared if all(keys_by_arm[a].shared_declared for a in arms) else blind).append(
            {"key": key, "arms": sorted(arms), "distinct_controllers": len(idents)})
    return {"blind_collisions": blind, "declared_sharing": declared, "ok": not blind,
            "detail": ("no arm reads another arm's state under a blind key" if not blind else
                       f"{len(blind)} key(s) shared by arms with different controllers and NOT "
                       f"declared -- one arm would be scored against another's state")}
