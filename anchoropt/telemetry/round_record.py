"""The residual-round summary object: schema + logger. NOTHING RUNS A ROUND HERE.

`RoundRecord` is a frozen dataclass with exactly the fields the brief specifies, plus the three
telemetry blocks. `RoundLogger` appends records to JSONL and can reload them. Neither builds a
proposal, chooses a residual, nor evaluates anything.

WHY `residuals_before` / `residuals_after` ARE A LIST OF (key, support), NOT A COUNT
----------------------------------------------------------------------------------
The comparison the loop exists to make is distributional -- R3's original ranking was 4/3/1 by
support, and "did re-mining change the shape" cannot be answered by a total. `residual_shape()`
renders the support multiset (e.g. "4/3/1") so before/after are comparable at a glance even when the
residual KEYS change, which they will once an anchor is promoted.

WHY `objective_delta` AND `cost_delta` ARE SEPARATE AND BOTH OPTIONAL
--------------------------------------------------------------------
A round can legitimately end with an accuracy delta and no cost measurement (tokens are not
instrumented on this substrate), and averaging an unmeasured cost as 0 would manufacture a
favourable cost claim. Absent means absent.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResidualEntry:
    """One ranked residual, as the loop saw it."""

    rank: int
    key: str
    support: int
    expressible: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateEta:
    """One concrete eta_mu realization offered for evaluation, and what became of it."""

    variant: str
    eta: Mapping[str, Any]
    materializable: bool
    rejection_reason: str | None = None
    evaluated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"variant": self.variant, "eta": {k: str(v) for k, v in dict(self.eta).items()},
                "materializable": self.materializable,
                "rejection_reason": self.rejection_reason, "evaluated": self.evaluated}


def residual_shape(residuals: Sequence[ResidualEntry]) -> str:
    """Support multiset as a compact string, e.g. '4/3/1'. Comparable across rounds."""
    return "/".join(str(r.support) for r in sorted(residuals, key=lambda r: (-r.support, r.key)))


@dataclass(frozen=True)
class RoundRecord:
    """One residual round, end to end. Schema only -- constructing it does not run anything."""

    round_id: str
    incumbent_id: str
    residuals_before: tuple[ResidualEntry, ...] = ()
    selected_residual: str | None = None
    selected_locus: str | None = None
    selected_signal: str | None = None
    feasible_actions: tuple[str, ...] = ()
    candidate_etas: tuple[CandidateEta, ...] = ()
    winner: str | None = None
    objective_delta: float | None = None
    cost_delta: Mapping[str, Any] | None = None
    residuals_after: tuple[ResidualEntry, ...] = ()
    # observational blocks, each optional
    funnel: Mapping[str, Any] | None = None
    cost: Mapping[str, Any] | None = None
    locality: Mapping[str, Any] | None = None
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "round_id": self.round_id,
            "incumbent_id": self.incumbent_id,
            "residuals_before": [r.as_dict() for r in self.residuals_before],
            "residuals_before_shape": residual_shape(self.residuals_before),
            "selected_residual": self.selected_residual,
            "selected_locus": self.selected_locus,
            "selected_signal": self.selected_signal,
            "feasible_actions": list(self.feasible_actions),
            "candidate_etas": [c.as_dict() for c in self.candidate_etas],
            "winner": self.winner,
            "objective_delta": self.objective_delta,
            "cost_delta": dict(self.cost_delta) if self.cost_delta is not None else None,
            "residuals_after": [r.as_dict() for r in self.residuals_after],
            "residuals_after_shape": residual_shape(self.residuals_after),
            "funnel": dict(self.funnel) if self.funnel is not None else None,
            "cost": dict(self.cost) if self.cost is not None else None,
            "locality": dict(self.locality) if self.locality is not None else None,
            "notes": self.notes,
        }

    def residual_shift(self) -> dict[str, Any]:
        """Before -> after, as shapes plus which residual KEYS appeared and disappeared."""
        before = {r.key for r in self.residuals_before}
        after = {r.key for r in self.residuals_after}
        return {"before_shape": residual_shape(self.residuals_before),
                "after_shape": residual_shape(self.residuals_after),
                "resolved": sorted(before - after), "introduced": sorted(after - before),
                "persisting": sorted(before & after)}


@dataclass
class RoundLogger:
    """Append-only JSONL of RoundRecords. A logger, not a runner."""

    path: pathlib.Path

    def append(self, record: RoundRecord) -> RoundRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record.as_dict(), default=str) + "\n")
        return record

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in open(self.path):
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def latest(self) -> dict[str, Any] | None:
        rows = self.load()
        return rows[-1] if rows else None
