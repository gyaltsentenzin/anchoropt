"""Accepted controllers as permanent regression fixtures -- the anti-yo-yo ledger.

WHY THIS FILE EXISTS. Across manual anchor development and then autonomous discovery, the same
failure recurred: a fix for one mechanism silently made an already-accepted one unreachable. Real
instances, all measured: a boundary-advance change expanded the signal set with no evaluator; four
runner copies of one probe drifted apart so a repaired check still ran stale logic; a trajectory
sidecar allowlist dropped an intervention's telemetry three separate times, each time making a live
controller look inert.

None of those were caught by "does the new thing work?" -- each needed "does the OLD thing still
work?", asked mechanically. That is the whole purpose here.

THE RULE THIS ENCODES. An accepted controller is preserved until a NEWLY MEASURED version explicitly
supersedes it. Superseding is an act with evidence attached (`supersedes`, plus its own measurement),
never a silent overwrite. `GoldenRegistry.register` REFUSES to replace an entry that does not
name the one it supersedes -- so losing an acceptance requires saying so out loud.

WHAT CORE KNOWS AND DOES NOT. Core owns the record shape, the identity/immutability rules and the
three regression LEVELS. It knows nothing about what a signal observes or what an operator does to a
store: every field below is an opaque string or mapping supplied by the adapter. A benchmark's
fixtures, host operations and evaluator path live on the adapter side of `RegressionSuite`.

PROVENANCE IS PART OF THE FIXTURE. A measurement that cannot be re-run is not a regression baseline,
so the record carries the runtime version, the patches required to execute it, the split, the model
and the config fingerprint. `anchoropt_arm_identity_telemetry`: flags, fingerprints and filenames all
agreed once and were all wrong -- only firing telemetry identified the arm. Hence `telemetry` and
`per_case` are REQUIRED for an accepted entry, and `AcceptedController.validate` rejects a record
that claims an effect with no execution evidence behind it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence


class RegistryRefusal(Exception):
    """An explicit, actionable refusal.

    Never returned as an empty result: the recurring defect this module exists to prevent is a
    silent no-op that reads as success. Every raise carries what was wrong and what to do.
    """


# ---------------------------------------------------------------------------------------------------
# WHAT AN ACCEPTANCE IS
# ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ControllerIdentity:
    """The (boundary, signal, action, operator, capability, phase) tuple that names a controller.

    capability_id is part of the IDENTITY, not decoration. Measured: two arms carried the reduce
    executor's capability_id on a reroute operation, and a host with two executors at one cell would
    have run whichever it resolved first -- a different mechanism under the accepted controller's
    name. Identity must therefore pin the executor.
    """

    boundary: str
    signal: str
    action: str
    operator: str = ""
    capability_id: str = ""
    phase: str = ""

    def key(self) -> str:
        return "/".join((self.boundary, self.signal, self.action,
                         self.operator or "-", self.capability_id or "-", self.phase or "-"))

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.key()


@dataclass(frozen=True)
class RuntimeProvenance:
    """Everything needed to EXECUTE the fixture again.

    `patches` is the field whose absence bit us: an isolated runtime needed several evaluator patches
    applied in order, and without them the controller installs cleanly and runs as the control.
    """

    runtime: str = ""
    evaluator_version: str = ""
    patches: tuple[str, ...] = ()
    split: str = ""
    model: str = ""
    config_fingerprint: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class ExecutionTelemetry:
    """The counters that prove the intervention RAN, not merely that it installed.

    `declined` is here because 0 declines is the informative reading: it distinguishes "the identity
    gate admitted the accepted controller" from "the gate refused it and the arm silently became the
    control".
    """

    requested: int = 0
    fired: int = 0
    episodes: int = 0
    acted: int = 0
    declined: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    def proves_execution(self) -> bool:
        return self.fired > 0 and self.acted > 0


@dataclass(frozen=True)
class PairedOutcome:
    """The per-case reference. Totals are not enough.

    Measured reason: an equal total is also produced by one gain cancelling one loss. Revalidation
    compares the id->correct MAPS and the gain/loss SETS, so `case_ids` carries them.
    """

    arm_correct: int = 0
    control_correct: int = 0
    n_scored: int = 0
    gains: tuple[str, ...] = ()
    losses: tuple[str, ...] = ()
    case_ids: Mapping[str, bool] = field(default_factory=dict)

    @property
    def net(self) -> int:
        return self.arm_correct - self.control_correct


@dataclass(frozen=True)
class AcceptedController:
    """One permanent regression fixture.

    `origin` separates an AUTONOMOUSLY accepted controller from a manual anchor kept as a positive
    control. Both protect the adapter's capabilities; only the first is a result, and conflating them
    would let a hand-built mechanism be reported as a discovery.
    """

    name: str
    identity: ControllerIdentity
    spec: Mapping[str, Any]
    provenance: RuntimeProvenance
    telemetry: ExecutionTelemetry
    outcome: PairedOutcome
    origin: str = "autonomous"            # "autonomous" | "manual_positive_control"
    positive_states: tuple[Mapping[str, Any], ...] = ()
    negative_states: tuple[Mapping[str, Any], ...] = ()
    caveats: tuple[str, ...] = ()
    supersedes: str = ""
    version: int = 1

    ORIGINS = ("autonomous", "manual_positive_control")

    def validate(self) -> None:
        """Refuse a fixture that cannot serve as a baseline. Explicit, itemised."""
        bad: list[str] = []
        if not self.name:
            bad.append("name is empty")
        if self.origin not in self.ORIGINS:
            bad.append(f"origin {self.origin!r} not in {self.ORIGINS}")
        if not self.identity.boundary or not self.identity.signal or not self.identity.action:
            bad.append("identity needs boundary, signal and action")
        if not self.identity.capability_id:
            bad.append("identity.capability_id is empty: without it a two-executor cell can run a "
                       "different mechanism under this controller's name")
        if self.origin == "autonomous":
            # A result must be re-runnable and must have actually executed.
            if not self.provenance.split:
                bad.append("provenance.split missing: a measurement with no split cannot be re-run")
            if not self.provenance.model:
                bad.append("provenance.model missing")
            if not self.telemetry.proves_execution():
                bad.append(f"telemetry does not prove execution (fired={self.telemetry.fired}, "
                           f"acted={self.telemetry.acted}): installation is not execution")
            if self.outcome.n_scored <= 0:
                bad.append("outcome.n_scored is 0: no paired measurement recorded")
            if not self.outcome.case_ids:
                bad.append("outcome.case_ids is empty: per-case reference is required because an "
                           "equal total can hide offsetting flips")
            # The gain/loss sets must agree with the recorded net, or the reference is internally
            # inconsistent and every later comparison inherits the error.
            if len(self.outcome.gains) - len(self.outcome.losses) != self.outcome.net:
                bad.append(f"gains({len(self.outcome.gains)}) - losses({len(self.outcome.losses)}) "
                           f"!= net({self.outcome.net})")
        if bad:
            raise RegistryRefusal(f"{self.name or '<unnamed>'}: " + "; ".join(bad))

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["identity_key"] = self.identity.key()
        return d


# ---------------------------------------------------------------------------------------------------
# THE REGISTRY
# ---------------------------------------------------------------------------------------------------

class GoldenRegistry:
    """Accepted controllers, preserved until explicitly superseded."""

    def __init__(self, entries: Sequence[AcceptedController] = ()) -> None:
        self._by_name: dict[str, AcceptedController] = {}
        for e in entries:
            self.register(e)

    # -- population ---------------------------------------------------------------------------
    def register(self, entry: AcceptedController) -> AcceptedController:
        entry.validate()
        prior = self._by_name.get(entry.name)
        if prior is not None:
            # THE ANTI-YO-YO RULE. Replacing an acceptance is allowed only as an explicit,
            # measured supersession.
            if entry.supersedes != prior.name:
                raise RegistryRefusal(
                    f"{entry.name!r} already accepted at version {prior.version}. To replace it, "
                    f"set supersedes={prior.name!r} and attach the NEW measurement. An accepted "
                    f"controller is never silently overwritten.")
            if entry.version <= prior.version:
                raise RegistryRefusal(
                    f"{entry.name!r} supersedes version {prior.version} but declares version "
                    f"{entry.version}: a superseding version must increase.")
        self._by_name[entry.name] = entry
        return entry

    # -- access -------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def get(self, name: str) -> AcceptedController:
        try:
            return self._by_name[name]
        except KeyError:
            raise RegistryRefusal(
                f"no accepted controller named {name!r}; known: {sorted(self._by_name)}") from None

    def all(self) -> tuple[AcceptedController, ...]:
        return tuple(self._by_name[k] for k in sorted(self._by_name))

    def autonomous(self) -> tuple[AcceptedController, ...]:
        return tuple(e for e in self.all() if e.origin == "autonomous")

    def positive_controls(self) -> tuple[AcceptedController, ...]:
        return tuple(e for e in self.all() if e.origin == "manual_positive_control")

    # -- composition --------------------------------------------------------------------------
    def composition_conflicts(self, candidate: ControllerIdentity) -> tuple[str, ...]:
        """Ways a NEW controller could break an accepted one. Never assume they compose.

        Two controllers each working alone is not evidence they work together: the second can
        occupy the same (boundary, action) cell with a DIFFERENT capability_id and be resolved
        first, or it can consume the same signal at the same boundary and shadow the firing.

        Returns human-readable conflict descriptions -- actionable, not a bare bool. An empty
        tuple means no STATIC conflict; it is not a substitute for the runtime smoke test, which
        is the only thing that can show a controller stopped firing.
        """
        out: list[str] = []
        for e in self.all():
            a = e.identity
            if a.boundary != candidate.boundary:
                continue
            same_cell = a.action == candidate.action
            if same_cell and a.capability_id != candidate.capability_id:
                out.append(
                    f"{e.name}: same cell ({a.boundary}/{a.action}) but a different capability_id "
                    f"({a.capability_id!r} vs {candidate.capability_id!r}). A host that resolves one "
                    f"executor per cell may route {e.name} to the wrong one -- assert capability "
                    f"identity at dispatch and smoke-test BOTH.")
            if a.signal == candidate.signal and same_cell:
                out.append(
                    f"{e.name}: identical (boundary, signal, action) -- the new controller can "
                    f"consume the firing and leave {e.name} inert. Verify {e.name} still fires.")
        return tuple(out)

    # -- persistence --------------------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {"version": 1, "entries": [e.to_json() for e in self.all()]}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2, sort_keys=False) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "GoldenRegistry":
        raw = json.loads(Path(path).read_text())
        reg = cls()
        for d in raw.get("entries", ()):
            d = dict(d)
            d.pop("identity_key", None)
            reg._by_name[d["name"]] = AcceptedController(
                name=d["name"],
                identity=ControllerIdentity(**d["identity"]),
                spec=d.get("spec", {}),
                provenance=RuntimeProvenance(**d.get("provenance", {})),
                telemetry=ExecutionTelemetry(**d.get("telemetry", {})),
                outcome=PairedOutcome(**d.get("outcome", {})),
                origin=d.get("origin", "autonomous"),
                positive_states=tuple(d.get("positive_states", ())),
                negative_states=tuple(d.get("negative_states", ())),
                caveats=tuple(d.get("caveats", ())),
                supersedes=d.get("supersedes", ""),
                version=int(d.get("version", 1)))
        return reg
