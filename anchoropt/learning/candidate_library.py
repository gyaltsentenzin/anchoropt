"""Every DISCOVERED mechanism, not just the accepted ones -- and the stack a result was measured against.

THE DEFECT THIS ANSWERS
-----------------------
Measured, on this project's own BFCL run: `rounds/AUTORUN/R1_RESULT.json` records
`controllers_installed: 0` and `incumbent/correct: 18` of 105 on the kv cell -- and 18 is *exactly* the
control score of `kv_core_capacity_relocation`, an autonomously accepted controller worth +4 on that
same cell, already sitting in the golden registry with full telemetry and per-case outcomes. The round
re-searched from bare H0 a mechanism the project had already found and measured.

Nothing was lost. Nothing was read back. Those are different failures with different fixes, and this
module is the second one.

WHY A SIBLING RECORD AND NOT A LOOSENED `AcceptedController`
------------------------------------------------------------
`AcceptedController.validate()` refuses a record whose telemetry does not prove execution and whose
outcome has no per-case reference. That refusal is the anti-yo-yo rule doing its job, and a DISCOVERED
candidate legitimately has neither. Widening it to admit unmeasured candidates would delete the one
guarantee the registry exists to provide, so accepted entries keep exactly one gatekeeper
(`GoldenRegistry.register`) and candidates live here.

WHAT `PENDING_VALIDATION` MEANS, AND WHY IT IS NOT A LOSER
----------------------------------------------------------
This is the state the project kept losing. Four round-1 candidates declared `phase=prereq` and were
evaluated by a query-time protocol that eliminated the phase they act in: the result was
`NO_OPPORTUNITY`, which is *not* a refutation. `round_ledger` already draws this line -- RETRYABLE
"does NOT exclude: its evidence was absent, not negative" -- and this module carries the same rule into
storage. A candidate whose evaluation was invalid stays available for validation forever. Only a
candidate that was *really measured* and did not help becomes MEASURED_NEGATIVE.

THE STACK FINGERPRINT
---------------------
A measurement that cannot name its incumbent cannot be compared to another one. `rec["incumbent"]` was
a path string, so "accepted" carried no statement about *what else was installed at the time*. A
controller accepted against H0 is not thereby accepted against H0+A+B: it may be redundant with B, or
shadowed by it at a shared (boundary, action) cell. `stack_fingerprint` makes that visible in the
record instead of leaving it to be remembered.

Core stays host-agnostic: every field below is an opaque string or mapping supplied by the adapter.
This module never learns what a signal observes or what an operator does to a store.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .golden_registry import ControllerIdentity, RegistryRefusal

# ---------------------------------------------------------------------------------------------------
# THE LIFECYCLE STATES
# ---------------------------------------------------------------------------------------------------

#: Emitted by a round as an executable spec; never evaluated.
DISCOVERED = "DISCOVERED"
#: Evaluated, but the evaluation could not bear on it (invalid instrument, missing evidence,
#: NO_OPPORTUNITY, budget exhausted). AVAILABLE FOR VALIDATION -- never an exclusion.
PENDING_VALIDATION = "PENDING_VALIDATION"
#: Really measured against a named incumbent, and did not help. An exclusion for THAT incumbent only.
MEASURED_NEGATIVE = "MEASURED_NEGATIVE"
#: Passed the acceptance protocol and was installed. Mirrored in the golden registry.
ACCEPTED = "ACCEPTED"
#: Replaced by a newer measured version, which names it.
SUPERSEDED = "SUPERSEDED"

STATES = (DISCOVERED, PENDING_VALIDATION, MEASURED_NEGATIVE, ACCEPTED, SUPERSEDED)

#: States that must NOT be used to skip re-evaluation. PENDING_VALIDATION is deliberately absent from
#: the exclusion set: that is the whole point of separating it from MEASURED_NEGATIVE.
_EXCLUDING = (ACCEPTED, MEASURED_NEGATIVE, SUPERSEDED)


def stack_fingerprint(identity_keys: Iterable[str]) -> str:
    """Name a controller STACK by what is in it, order-insensitively.

    Order-insensitive on purpose: `decide()` is first-fire-wins, so ordering can matter to behaviour,
    but two runs installing the same SET are comparing the same *composition hypothesis*. When order
    is load-bearing the stack file records it explicitly; the fingerprint answers "which controllers
    were installed", which is the question a result needs to state.

    The empty stack -- bare H0 -- gets the literal 'H0' rather than the hash of nothing, because
    'measured against H0' is the single most-quoted baseline in this project and it should be legible
    in a filename and a log line.
    """
    keys = sorted({str(k) for k in identity_keys if str(k).strip()})
    if not keys:
        return "H0"
    return "stack_" + hashlib.sha256("\n".join(keys).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class EvaluationContext:
    """WHAT a measurement was taken against. Without this a net is a number with no referent."""

    incumbent_stack: tuple[str, ...] = ()      # identity_keys installed while measuring
    incumbent_token: str = ""                  # the host's own run identifier, for re-running it
    cell: str = ""                             # (domain x backend) -- the split unit
    split: str = ""
    model: str = ""
    scored_phase: str = ""                     # where the delta was measured
    acting_phase: str = ""                     # where the controller ACTS (may differ -- that is fine)

    @property
    def fingerprint(self) -> str:
        return stack_fingerprint(self.incumbent_stack)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stack_fingerprint"] = self.fingerprint
        return d


@dataclass(frozen=True)
class Measurement:
    """A paired result, or the reason there is not one.

    `valid` is separate from the counts and is the field that keeps a broken instrument from being
    banked as evidence against a candidate. `net` is stored rather than derived so a record round-trips
    without recomputation, and `validate()` checks the two agree.
    """

    arm_correct: int = 0
    control_correct: int = 0
    n_scored: int = 0
    gains: tuple[str, ...] = ()
    losses: tuple[str, ...] = ()
    valid: bool = True
    invalid_reason: str = ""
    firings: int = 0
    detail: str = ""

    @property
    def net(self) -> int:
        return len(self.gains) - len(self.losses)


@dataclass
class CandidateRecord:
    """One discovered mechanism, at whatever stage of the lifecycle it has reached."""

    name: str
    identity: ControllerIdentity
    spec: Mapping[str, Any]
    state: str = DISCOVERED
    context: EvaluationContext = field(default_factory=EvaluationContext)
    measurement: Measurement | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    round_id: str = ""
    supersedes: str = ""
    notes: tuple[str, ...] = ()
    version: int = 1

    def validate(self) -> None:
        bad: list[str] = []
        if not self.name:
            bad.append("name is empty")
        if self.state not in STATES:
            bad.append(f"state {self.state!r} not in {STATES}")
        if not self.identity.capability_id:
            bad.append("identity.capability_id is empty: a two-executor cell could run a different "
                       "mechanism under this candidate's name")
        if not self.spec:
            bad.append("spec is empty: a record that cannot be re-executed is not a record of a "
                       "mechanism")
        m = self.measurement
        if self.state in (MEASURED_NEGATIVE, ACCEPTED):
            # These two states ASSERT that a real measurement happened. The others do not.
            if m is None:
                bad.append(f"state {self.state} requires a measurement")
            else:
                if not m.valid:
                    bad.append(f"state {self.state} claims a measurement that is marked invalid "
                               f"({m.invalid_reason or 'no reason given'}); an unbelievable "
                               f"measurement is PENDING_VALIDATION, not a verdict")
                if m.n_scored <= 0:
                    bad.append(f"state {self.state} requires n_scored > 0")
                if not self.context.incumbent_stack and not self.context.incumbent_token:
                    bad.append(f"state {self.state} requires an evaluation context naming the "
                               f"incumbent it was measured against")
        if m is not None and m.arm_correct - m.control_correct != m.net:
            bad.append(f"arm({m.arm_correct}) - control({m.control_correct}) != "
                       f"gains-losses({m.net}): the record is internally inconsistent")
        if self.state == SUPERSEDED and not self.supersedes and not self.notes:
            bad.append("SUPERSEDED must name what replaced it, in `supersedes` or `notes`")
        if bad:
            raise RegistryRefusal(f"{self.name or '<unnamed>'}: " + "; ".join(bad))

    @property
    def excludes_re_evaluation(self) -> bool:
        """Whether this record is a reason NOT to measure the mechanism again.

        PENDING_VALIDATION returns False -- deliberately. That is the state a candidate lands in when
        the instrument was wrong, and re-measuring it is exactly what should happen.
        """
        return self.state in _EXCLUDING

    def excludes_against(self, stack: Sequence[str]) -> bool:
        """A MEASURED_NEGATIVE verdict is true only against the incumbent that produced it.

        Once other controllers are installed the composition has changed, so the old negative says
        nothing about the new one and the candidate becomes eligible again. ACCEPTED and SUPERSEDED
        are permanent: they are facts about the mechanism's history, not about one comparison.
        """
        if self.state in (ACCEPTED, SUPERSEDED):
            return True
        if self.state != MEASURED_NEGATIVE:
            return False
        return stack_fingerprint(stack) == self.context.fingerprint

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["identity_key"] = self.identity.key()
        d["context"] = self.context.to_dict()
        d["excludes_re_evaluation"] = self.excludes_re_evaluation
        if self.measurement is not None:
            d["measurement"] = dict(asdict(self.measurement), net=self.measurement.net)
        return d


class CandidateLibrary:
    """The durable record of every mechanism this project has discovered.

    Used for persistence, deduplication, regression protection and retrospective coverage assessment.
    It is deliberately NOT a source of proposals: handing a proposer a library of known-good
    controllers and their gains would make any subsequent "rediscovery" unfalsifiable. Nothing here
    is fed forward as a target; `coverage_against` exists to let a HUMAN compare after the fact.
    """

    def __init__(self, records: Sequence[CandidateRecord] = ()) -> None:
        self._by_name: dict[str, CandidateRecord] = {}
        for r in records:
            self.upsert(r)

    # -- population ---------------------------------------------------------------------------
    def upsert(self, record: CandidateRecord) -> CandidateRecord:
        """Record a candidate, or ADVANCE one already present.

        Advancing is allowed; regressing is not. A record that has been ACCEPTED does not quietly
        become DISCOVERED again because a later round re-emitted the same spec -- that would be the
        yo-yo this module exists to stop, arriving through the back door.
        """
        record.validate()
        prior = self._by_name.get(record.name)
        if prior is not None and prior.state == ACCEPTED and record.state == DISCOVERED:
            raise RegistryRefusal(
                f"{record.name!r} is ACCEPTED; re-emitting it as DISCOVERED would silently un-accept "
                f"it. To replace it, record the new measurement with state ACCEPTED and "
                f"supersedes={prior.name!r}, or record the re-emission under a new name.")
        self._by_name[record.name] = record
        return record

    # -- access -------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def get(self, name: str) -> CandidateRecord:
        try:
            return self._by_name[name]
        except KeyError:
            raise RegistryRefusal(
                f"no candidate named {name!r}; known: {sorted(self._by_name)}") from None

    def all(self) -> tuple[CandidateRecord, ...]:
        return tuple(self._by_name[k] for k in sorted(self._by_name))

    def in_state(self, state: str) -> tuple[CandidateRecord, ...]:
        if state not in STATES:
            raise RegistryRefusal(f"unknown state {state!r}; known: {STATES}")
        return tuple(r for r in self.all() if r.state == state)

    def pending(self) -> tuple[CandidateRecord, ...]:
        """Candidates awaiting a VALID evaluation. The queue a next round should start from."""
        return self.in_state(PENDING_VALIDATION)

    def accepted(self) -> tuple[CandidateRecord, ...]:
        return self.in_state(ACCEPTED)

    # -- the questions a round actually asks ---------------------------------------------------
    def installed_stack(self, cell: str = "") -> tuple[str, ...]:
        """The identity_keys of ACCEPTED controllers -- i.e. the CURRENT incumbent, as a stack.

        This is the call whose absence caused the defect: with no way to ask "what is installed?",
        every round started from H0 by default.
        """
        out = {r.identity.key() for r in self.accepted()
               if not cell or not r.context.cell or r.context.cell == cell}
        return tuple(sorted(out))

    def eligible(self, candidates: Iterable[CandidateRecord] | Iterable[str],
                 stack: Sequence[str] = ()) -> tuple[str, ...]:
        """Which of these mechanisms may still be measured against `stack`.

        Unknown names are ELIGIBLE. A mechanism this library has never seen is not thereby
        disqualified -- treating absence as exclusion is the same error as reading silence as a result.
        """
        out: list[str] = []
        for c in candidates:
            name = c if isinstance(c, str) else c.name
            rec = self._by_name.get(name)
            if rec is None or not rec.excludes_against(stack):
                out.append(str(name))
        return tuple(out)

    def coverage_against(self, library: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Retrospective comparison with a reference set of mechanisms, by IDENTITY.

        For reporting only, and only after a discovery is already recorded. `library` maps a reference
        name to a mapping carrying at least the identity fields; matching is on the identity key, so a
        rediscovery counts only when it names the same (boundary, signal, action, operator, capability,
        phase) tuple -- not when it merely sounds similar in a summary.
        """
        mine = {r.identity.key(): r for r in self.all()}
        recovered: dict[str, str] = {}
        missing: list[str] = []
        for ref_name, ident in library.items():
            try:
                key = ControllerIdentity(**{k: str(ident.get(k, "") or "")
                                            for k in ("boundary", "signal", "action", "operator",
                                                      "capability_id", "phase")}).key()
            except TypeError:                               # pragma: no cover - defensive
                missing.append(ref_name)
                continue
            hit = mine.get(key)
            if hit is None:
                missing.append(ref_name)
            else:
                recovered[ref_name] = f"{hit.name} [{hit.state}]"
        return {"n_reference": len(library), "recovered": recovered,
                "n_recovered": len(recovered), "not_recovered": sorted(missing)}

    # -- persistence --------------------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {"version": 1, "records": [r.to_json() for r in self.all()]}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2, sort_keys=False) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "CandidateLibrary":
        """Rebuild from disk. A missing file is an EMPTY library, not an error -- the first round of a
        new workstream has nothing to load, and that must not need a special case at the call site."""
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        lib = cls()
        for d in raw.get("records", ()):
            d = dict(d)
            d.pop("identity_key", None)
            d.pop("excludes_re_evaluation", None)
            ctx = dict(d.get("context") or {})
            ctx.pop("stack_fingerprint", None)
            ctx["incumbent_stack"] = tuple(ctx.get("incumbent_stack") or ())
            meas = d.get("measurement")
            if meas is not None:
                meas = dict(meas)
                meas.pop("net", None)
                meas["gains"] = tuple(meas.get("gains") or ())
                meas["losses"] = tuple(meas.get("losses") or ())
            rec = CandidateRecord(
                name=d["name"],
                identity=ControllerIdentity(**d["identity"]),
                spec=d.get("spec", {}),
                state=d.get("state", DISCOVERED),
                context=EvaluationContext(**ctx),
                measurement=Measurement(**meas) if meas is not None else None,
                provenance=d.get("provenance", {}),
                round_id=d.get("round_id", ""),
                supersedes=d.get("supersedes", ""),
                notes=tuple(d.get("notes", ())),
                version=int(d.get("version", 1)))
            # Validate on LOAD as well as on write. A file edited by hand, or written by an older
            # version, is exactly where an inconsistent record comes from, and the load path is the
            # last place it can be caught before a round reasons from it.
            rec.validate()
            lib._by_name[rec.name] = rec
        return lib
