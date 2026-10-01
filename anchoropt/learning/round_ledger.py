"""CROSS-ROUND STATE. The one thing an unattended multi-round run needs and nothing implemented.

WHAT WAS MISSING, EXACTLY
------------------------
`candidate_selection.select_candidate` takes `settled` and `measured_losers` and documents them as
"label sets the caller carries across rounds". `self_evolve.step()` runs exactly one round and holds
no memory. `round_runner.promote` advances the moving incumbent. So every stage of the loop existed
and the thing joining round N's OUTCOME to round N+1's ELIGIBILITY did not: there was no caller, and
`select_candidate` was reachable only from its own tests.

That gap is why the orchestration pilot needed a human between rounds. It is not a missing
abstraction or a defect in the optimizer -- it is missing experiment wiring, and this module is the
wiring. No new rule, no new threshold, no new ranking term.

WHY A LEDGER RATHER THAN TWO SETS
---------------------------------
Two `set[str]` would be enough to make the loop run and would destroy the audit trail, which is the
part that matters. A label lands in `measured_losers` because a measurement said so, and a round that
declines to re-spend a paired evaluation on it must be able to say WHICH round measured it and WHAT
the verdict was. So each entry carries its round, outcome, and the acceptance reason, and the ledger
is JSON round-trippable -- an unattended run's decisions are reconstructible afterwards from disk
without re-running anything.

THE ASYMMETRY BETWEEN THE FOUR OUTCOMES
---------------------------------------
This is the whole substance of the module and it is deliberately not uniform:

  ACCEPTED       -> SETTLED. Never measured again. The incumbent moves; re-mining is against the
                    NEW incumbent, so its own exposures are no longer residual.
  REJECTED       -> MEASURED_LOSER, but ONLY when the rejection was a measurement. A candidate
                    rejected because criterion 1 measured net <= 0 is a loser. A candidate rejected
                    because criterion 3 telemetry was PENDING_VALIDATION measured NOTHING about the
                    mechanism -- excluding it would convert missing evidence into a permanent
                    negative verdict, which is the exact error the PENDING verdict exists to prevent.
                    Those become RETRYABLE instead, with the remedy that would resolve them.
  BLOCKED        -> RETRYABLE, always. A denominator failure is infrastructure, not efficacy;
                    `step()` already refuses to let it reach the acceptance rule, and the ledger
                    must not undo that by recording it as a negative result.
  NO_CANDIDATE   -> nothing is recorded against any label; the round is terminal for its residual.

Getting this wrong in either direction is costly and in opposite ways: treat every rejection as a
loser and the loop permanently discards hypotheses it never actually tested, converging early on a
false exhaustion; treat none as losers and it re-spends GPU rounds on the same measured failure
forever. The discriminator is whether a criterion reached a verdict FROM EVIDENCE.

BENCHMARK-FREE
--------------
Everything here reads labels, outcome strings, and criterion verdicts. Nothing about what a signal
observes, what a store does, or which corpus produced it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# How a label came to be excluded. Distinct from candidate_selection's EXCLUSION reasons, which say
# why an arm was skipped in ONE round; these say what a round CONCLUDED about it.
SETTLED = "settled_accepted"
MEASURED_LOSER = "measured_no_benefit_or_negative"
RETRYABLE = "retryable_evidence_missing"

#: Outcome strings from `self_evolve`, restated here so the ledger does not import the stepper it is
#: meant to be usable without. Mismatch is caught by a test, not assumed.
OUTCOME_ACCEPTED = "accepted"
OUTCOME_REJECTED = "rejected"
OUTCOME_NO_CANDIDATE = "no_candidate"
OUTCOME_BLOCKED = "blocked"
OUTCOME_SELECTED_ONLY = "selected_only"


@dataclass(frozen=True)
class LedgerEntry:
    """One conclusion about one label, with the round and evidence that produced it."""

    label: str
    disposition: str
    round_index: int
    outcome: str
    reason: str = ""
    remedy: str = ""
    net: int | None = None
    recorded_at: int = field(default_factory=lambda: int(time.time()))

    @property
    def excludes_from_future_rounds(self) -> bool:
        """RETRYABLE does NOT exclude: its evidence was absent, not negative."""
        return self.disposition in (SETTLED, MEASURED_LOSER)

    def to_json(self) -> dict[str, Any]:
        return {"label": self.label, "disposition": self.disposition,
                "round_index": self.round_index, "outcome": self.outcome, "reason": self.reason,
                "remedy": self.remedy, "net": self.net, "recorded_at": self.recorded_at}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> LedgerEntry:
        return cls(label=str(d["label"]), disposition=str(d["disposition"]),
                   round_index=int(d["round_index"]), outcome=str(d.get("outcome", "")),
                   reason=str(d.get("reason", "")), remedy=str(d.get("remedy", "")),
                   net=(None if d.get("net") is None else int(d["net"])),
                   recorded_at=int(d.get("recorded_at", 0)))


def classify_rejection(report) -> tuple[str, str, str]:
    """Was a rejection a MEASUREMENT or an absence of evidence? Returns (disposition, reason, remedy).

    `report` is an `acceptance_criteria.AcceptanceReport`, or None when the round used the v0.1
    default accept (which implements criterion 1 plus engagement only).

    A criterion that FAILED reached a verdict from evidence -- that is a measured negative, and the
    label is a loser. A criterion left PENDING_VALIDATION did not: the mechanism telemetry was
    absent, the dev split was never run, the safety counters were never gathered. Recording that as a
    negative result would launder missing evidence into a verdict, so it is retryable and carries the
    remedy that would resolve it.

    When both kinds are present the FAIL dominates: something was genuinely measured to be bad, and
    no amount of additional telemetry changes a net <= 0.
    """
    if report is None:
        # The v0.1 default accept rejects on net <= 0 or zero engagement. Both are measurements.
        return MEASURED_LOSER, "rejected by the default acceptance rule", ""

    failed = [c for c in report.blocking if c.verdict == "FAIL"]
    if failed:
        return (MEASURED_LOSER,
                "; ".join(f"C{c.number} {c.name} FAIL: {c.detail}" for c in failed),
                "")
    pending = [c for c in report.blocking if c.verdict != "FAIL"]
    if pending:
        return (RETRYABLE,
                "; ".join(f"C{c.number} {c.name} {c.verdict}: {c.detail}" for c in pending),
                "; ".join(c.remedy for c in pending if c.remedy))
    # `blocking` empty on a rejection means the caller disagreed with the report. Do not invent a
    # verdict for it.
    return RETRYABLE, "rejected with no blocking criterion recorded", \
        "reconcile the accept callable with the acceptance report it produced"


@dataclass
class RoundLedger:
    """The durable cross-round state. Append-only within a run; JSON round-trippable."""

    entries: list[LedgerEntry] = field(default_factory=list)
    rounds_run: int = 0

    # -- the two sets `select_candidate` asks for -------------------------------------------------
    @property
    def settled(self) -> tuple[str, ...]:
        return tuple(sorted({e.label for e in self.entries if e.disposition == SETTLED}))

    @property
    def measured_losers(self) -> tuple[str, ...]:
        """Excludes anything later SETTLED: a label that eventually got accepted is not a loser."""
        settled = set(self.settled)
        return tuple(sorted({e.label for e in self.entries
                             if e.disposition == MEASURED_LOSER and e.label not in settled}))

    @property
    def retryable(self) -> tuple[str, ...]:
        """Labels whose evidence was MISSING. Reported so the run's blockers are visible, and
        deliberately NOT excluded from selection."""
        blocked = set(self.settled) | set(self.measured_losers)
        return tuple(sorted({e.label for e in self.entries
                             if e.disposition == RETRYABLE and e.label not in blocked}))

    def selection_kwargs(self) -> dict[str, tuple[str, ...]]:
        """Splat straight into `select_candidate(...)`. The joint this module exists to close."""
        return {"settled": self.settled, "measured_losers": self.measured_losers}

    # -- recording -------------------------------------------------------------------------------
    def record(self, *, label: str, outcome: str, report=None, net: int | None = None,
               round_index: int | None = None) -> LedgerEntry | None:
        """Record one round's conclusion. Returns the entry, or None when nothing is concluded.

        `NO_CANDIDATE` and `SELECTED_ONLY` conclude nothing about any label: the first had no arm to
        conclude about, and the second never measured the arm it chose.
        """
        idx = self.rounds_run if round_index is None else int(round_index)
        if outcome in (OUTCOME_NO_CANDIDATE, OUTCOME_SELECTED_ONLY):
            return None
        if not label:
            raise ValueError(f"outcome {outcome!r} concerns a candidate but no label was given")

        if outcome == OUTCOME_ACCEPTED:
            entry = LedgerEntry(label, SETTLED, idx, outcome,
                                reason=(report.reason() if report is not None else "accepted"),
                                net=net)
        elif outcome == OUTCOME_BLOCKED:
            # Infrastructure, not efficacy. `step()` refuses to let this reach the acceptance rule;
            # recording it as a negative would undo that refusal one layer up.
            entry = LedgerEntry(label, RETRYABLE, idx, outcome,
                                reason="denominator integrity failed -- the case coverage cannot be "
                                       "trusted, so nothing was measured about the mechanism",
                                remedy="repair the evaluation's case coverage and re-run this arm",
                                net=net)
        elif outcome == OUTCOME_REJECTED:
            disp, reason, remedy = classify_rejection(report)
            entry = LedgerEntry(label, disp, idx, outcome, reason=reason, remedy=remedy, net=net)
        else:
            raise ValueError(f"unknown outcome {outcome!r}")

        self.entries.append(entry)
        return entry

    def advance(self) -> int:
        """Close the current round. Returns the new round index."""
        self.rounds_run += 1
        return self.rounds_run

    def for_label(self, label: str) -> tuple[LedgerEntry, ...]:
        return tuple(e for e in self.entries if e.label == label)

    # -- reporting -------------------------------------------------------------------------------
    def summary(self) -> str:
        lines = [f"rounds_run={self.rounds_run}  entries={len(self.entries)}"]
        for name, labels in (("SETTLED", self.settled), ("MEASURED_LOSER", self.measured_losers),
                             ("RETRYABLE", self.retryable)):
            lines.append(f"  {name} ({len(labels)}): {', '.join(labels) or '-'}")
        for e in self.entries:
            if e.disposition == RETRYABLE and e.remedy:
                lines.append(f"  REMEDY for {e.label} (round {e.round_index}): {e.remedy}")
        return "\n".join(lines)

    # -- persistence -----------------------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {"version": 1, "rounds_run": self.rounds_run,
                "entries": [e.to_json() for e in self.entries],
                "settled": list(self.settled), "measured_losers": list(self.measured_losers),
                "retryable": list(self.retryable)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> RoundLedger:
        return cls(entries=[LedgerEntry.from_json(e) for e in d.get("entries", [])],
                   rounds_run=int(d.get("rounds_run", 0)))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> RoundLedger:
        return cls.from_json(json.loads(Path(path).read_text()))


#: Separator for a label that encodes a controller identity. The ledger keys on opaque labels and
#: `candidate_search` keys on `(boundary, signal, action)` tuples, so ONE of them must be canonical
#: or the two memories disagree silently. The cell is canonical; a label is its rendering.
CELL_SEP = "/"


def label_for_cell(cell: Sequence[str]) -> str:
    """Render a controller identity as a ledger label. The inverse of `cell_from_label`."""
    return CELL_SEP.join(str(p) for p in cell)


def cell_from_label(label: str) -> tuple[str, ...] | None:
    """Parse a ledger label back to a controller identity, or None if it does not encode one.

    Returning None rather than raising matters: a run may use labels that are not cells at all, and
    those simply contribute no cell exclusion. Guessing a cell from an arbitrary label would exclude
    a controller nobody decided about, which is worse than excluding none.
    """
    parts = label.split(CELL_SEP)
    return tuple(parts) if len(parts) == 3 and all(parts) else None


def excluded_cells(ledger: RoundLedger) -> tuple[tuple[str, ...], ...]:
    """Controller identities an earlier round DECIDED, for `candidate_search(exclude_cells=...)`.

    SETTLED and MEASURED_LOSER both exclude; RETRYABLE does not -- its evidence was absent, not
    negative, so the identity must stay selectable. That is the same asymmetry the label sets
    enforce, applied at the identity level, and both must agree or a settled controller gets
    re-measured through whichever memory the round happens to consult.
    """
    out: list[tuple[str, ...]] = []
    for label in list(ledger.settled) + list(ledger.measured_losers):
        cell = cell_from_label(label)
        if cell is not None and cell not in out:
            out.append(cell)
    return tuple(out)


def seed_settled_from_registry(ledger: RoundLedger, accepted: Iterable[Any],
                               *, round_index: int = -1) -> tuple[str, ...]:
    """Seed SETTLED from already-accepted controllers so a fresh run does not re-measure them.

    `accepted` is any iterable of objects with a `.name` -- `GoldenRegistry.all()` satisfies it.
    Round index -1 marks "settled before this run began", distinguishing prior acceptances from ones
    this run made. Idempotent.
    """
    added: list[str] = []
    known = set(ledger.settled)
    for entry in accepted:
        name = str(getattr(entry, "name", entry))
        if name in known:
            continue
        ledger.entries.append(LedgerEntry(
            name, SETTLED, round_index, OUTCOME_ACCEPTED,
            reason="accepted before this run; preserved as a permanent fixture"))
        known.add(name)
        added.append(name)
    return tuple(added)
