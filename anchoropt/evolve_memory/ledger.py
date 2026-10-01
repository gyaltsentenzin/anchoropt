"""Append-only experiment memory: what was tried, what happened, and what must not be retried.

    results/self_evolve/memory/experiments.jsonl     concrete attempts
    results/self_evolve/memory/lessons.jsonl         structural lessons

TWO KINDS OF MEMORY, KEPT APART ON PURPOSE
------------------------------------------
* an EXPERIMENT is a concrete (residual, locus, signal, action, eta) that was actually tried, with its
  measured outcome. Retrieved only when a new candidate is relevant to it.
* a LESSON is a methodological constraint that outlived its experiment ("a signal must be observable at
  the boundary"). It applies to residuals it was never measured on.

Merging them would let a lesson inherit an experiment's narrow scope, or an experiment's number get
quoted as a general rule. The R3 round produced one of each and they are not interchangeable: "se3a2
beat se3a1 by 2 cases on train" is an experiment that did NOT replicate, while "changes on non-fired
episodes are not the controller's effect" is a lesson that holds regardless.

APPEND-ONLY, and status is never rewritten in place. A candidate that was `rejected` and later
`accepted` under different conditions gets a SECOND row; collapsing them would erase the fact that the
first evaluation happened.
"""

from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from anchoropt.evolve_memory.fingerprint import (
    candidate_fingerprint, context_key, fingerprint_parts, residual_family,
)

STATUSES = ("accepted", "rejected", "infeasible", "invalid")
DEFAULT_ROOT = pathlib.Path("results/self_evolve/memory")

# `invalid` is its own status and not a flavour of `rejected`. A REJECTED candidate was measured and
# lost; an INVALID one produced a number that cannot be believed (the seed-2 "replication" was a
# bit-exact re-execution). Merging them would let a later round treat an unmeasured candidate as
# refuted -- and never retry something that has never actually been tested.


@dataclass(frozen=True)
class Experiment:
    """One attempted policy and its outcome."""

    round_id: str
    residual: str
    locus: str
    signal: str
    action: str
    eta: Mapping[str, Any] = field(default_factory=dict)
    status: str = "rejected"
    diagnosis: str = ""
    reason: str = ""
    gains: int | None = None
    losses: int | None = None
    firings: int | None = None
    steps_delta: int | None = None
    calls_delta: int | None = None
    context: str = ""
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} not in {STATUSES}")

    @property
    def fingerprint(self) -> str:
        return candidate_fingerprint(residual=self.residual, locus=self.locus,
                                     signal=self.signal, action=self.action, eta=self.eta)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["eta"] = {k: str(v) for k, v in dict(self.eta).items()}
        d.update(fingerprint_parts(residual=self.residual, locus=self.locus, signal=self.signal,
                                   action=self.action, eta=self.eta))
        return d

    @property
    def net(self) -> int | None:
        if self.gains is None or self.losses is None:
            return None
        return self.gains - self.losses

    def one_line(self) -> str:
        """The compact form shown to a role. Outcome first, because that is what informs a choice."""
        bits = []
        if self.net is not None:
            bits.append(f"net {self.net:+d} (+{self.gains}/-{self.losses})")
        if self.firings is not None:
            bits.append(f"{self.firings} firings")
        outcome = ", ".join(bits) if bits else "not measured"
        eta_bit = "; ".join(f"{k}={str(v)[:60]}" for k, v in dict(self.eta).items()) or "(no eta)"
        return f"{self.action}@{self.locus} on {self.signal} [{eta_bit}] -> {self.status.upper()}: {outcome}. {self.reason}".strip()


@dataclass(frozen=True)
class Lesson:
    """A structural constraint that outlives the experiment that produced it."""

    key: str
    text: str
    origin: str = ""
    applies_to: tuple[str, ...] = ()      # empty = universal
    kind: str = "structural"              # structural | invalid_approach

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "applies_to": list(self.applies_to)}


class ExperimentLedger:
    """Append-only JSONL over experiments + lessons. Deterministic reads, no rewriting."""

    def __init__(self, root: pathlib.Path | str = DEFAULT_ROOT):
        self.root = pathlib.Path(root)
        self.experiments_path = self.root / "experiments.jsonl"
        self.lessons_path = self.root / "lessons.jsonl"

    # ------------------------------------------------------------------ write
    def record(self, exp: Experiment) -> Experiment:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.experiments_path, "a") as fh:
            fh.write(json.dumps(exp.as_dict(), default=str) + "\n")
        return exp

    def record_lesson(self, lesson: Lesson) -> Lesson:
        self.root.mkdir(parents=True, exist_ok=True)
        if any(l.key == lesson.key for l in self.lessons()):
            return lesson                       # idempotent by key; lessons are not duplicated
        with open(self.lessons_path, "a") as fh:
            fh.write(json.dumps(lesson.as_dict(), default=str) + "\n")
        return lesson

    # ------------------------------------------------------------------ read
    def _rows(self, path: pathlib.Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        out = []
        for line in open(path):
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def experiments(self) -> list[Experiment]:
        out = []
        for r in self._rows(self.experiments_path):
            out.append(Experiment(
                round_id=r.get("round_id", ""), residual=r.get("residual", ""),
                locus=r.get("locus", ""), signal=r.get("signal", ""), action=r.get("action", ""),
                eta=r.get("eta") or {}, status=r.get("status", "rejected"),
                diagnosis=r.get("diagnosis", ""), reason=r.get("reason", ""),
                gains=r.get("gains"), losses=r.get("losses"), firings=r.get("firings"),
                steps_delta=r.get("steps_delta"), calls_delta=r.get("calls_delta"),
                context=r.get("context", ""), ts=r.get("ts", 0.0)))
        return out

    def lessons(self) -> list[Lesson]:
        return [Lesson(key=r.get("key", ""), text=r.get("text", ""), origin=r.get("origin", ""),
                       applies_to=tuple(r.get("applies_to") or ()), kind=r.get("kind", "structural"))
                for r in self._rows(self.lessons_path)]

    # ------------------------------------------------------------------ query
    def by_fingerprint(self, fp: str) -> list[Experiment]:
        return [e for e in self.experiments() if e.fingerprint == fp]

    def prior_result(self, *, residual: str, locus: str, signal: str, action: str,
                     eta: Mapping[str, Any] | None = None,
                     context: str | None = None) -> Experiment | None:
        """The most recent COMPARABLE prior evaluation of this exact candidate, or None.

        `context` must match when given: a result measured on another cell is not evidence about this
        one. An `invalid` prior does NOT count as a prior result -- it was never validly measured, so
        the candidate is still untried.
        """
        fp = candidate_fingerprint(residual=residual, locus=locus, signal=signal, action=action,
                                   eta=eta)
        hits = [e for e in self.by_fingerprint(fp) if e.status != "invalid"]
        if context is not None:
            hits = [e for e in hits if not e.context or e.context == context]
        return max(hits, key=lambda e: e.ts) if hits else None

    def for_residual(self, residual: str, *, limit: int = 6) -> list[Experiment]:
        """Attempts against the same residual FAMILY, newest first."""
        fam = residual_family(residual)
        hits = [e for e in self.experiments() if residual_family(e.residual) == fam]
        return sorted(hits, key=lambda e: -e.ts)[:limit]

    def for_locus(self, locus: str, signal: str | None = None, *, limit: int = 6) -> list[Experiment]:
        """Attempts at the same locus (optionally the same signal), newest first.

        Relevant even across residuals: whether an executor can realize something at a boundary is a
        property of the host, not of the failure being fixed.
        """
        from anchoropt.evolve_memory.fingerprint import normalize_text
        lo = normalize_text(locus)
        hits = [e for e in self.experiments() if normalize_text(e.locus) == lo]
        if signal is not None:
            sig = normalize_text(signal)
            hits = [e for e in hits if normalize_text(e.signal) == sig]
        return sorted(hits, key=lambda e: -e.ts)[:limit]

    def invalid_approaches(self) -> list[Lesson]:
        return [l for l in self.lessons() if l.kind == "invalid_approach"]

    def structural_lessons(self) -> list[Lesson]:
        return [l for l in self.lessons() if l.kind == "structural"]
