"""The MULTI-ROUND DRIVER: loop `self_evolve.step`, carry the ledger, re-mine the moving incumbent.

WHAT THIS ADDS, AND WHAT IT REFUSES TO ADD
------------------------------------------
`self_evolve.step()` is one round. `round_ledger.RoundLedger` is the cross-round memory. This is the
loop around them, and it is deliberately thin: it owns NO ranking, NO threshold, NO acceptance logic
and NO benchmark knowledge. Every decision is delegated:

    step_fn(incumbent, ledger)  -> StepResult        usually a closure over self_evolve.step
    remine(incumbent)           -> None              adapter-side; re-diagnose the NEW incumbent
    label_of(StepResult)        -> str               how this run names the arm it measured

`remine` is injected rather than called because re-mining reads a host's own trajectories and result
files -- that is an ADAPTER capability, and wiring it into the core would put a benchmark's directory
layout inside the optimizer. The driver's contract is only that re-mining happens after an acceptance
and BEFORE the next diagnosis, because the point of the moving incumbent is that the next round's
residual is the residual of what is now installed. Skipping it is how a loop ends up measuring every
anchor against the original baseline and reporting a stack that was never assembled.

WHY IT STOPS
------------
An unattended run must terminate for a stated reason, and "ran out of rounds" is the least
informative one. Four terminal conditions, each recorded:

    MAX_ROUNDS          the budget was spent; says nothing about the search space
    NO_CANDIDATE        the round produced no arm -- the space is exhausted under this policy
    CONSECUTIVE_NULLS   `patience` rounds in a row concluded nothing new; a loop that keeps
                        selecting arms it cannot resolve is stuck, not working
    REMINE_FAILED       re-mining raised; the next round's residual would be STALE, and measuring
                        against a stale residual is worse than stopping

A round that BLOCKS or produces a RETRYABLE rejection is not progress, and it is not a loser either
(see round_ledger). Counting those toward patience is what keeps a run from spinning forever on an
arm whose evidence never arrives -- while the ledger keeps the arm eligible, so a later round with
the telemetry fixed can still measure it.

WHAT IT DOES NOT DO
-------------------
It does not decide that a run was unattended. That is a claim about what a human did during it, and
no code can assert it. The driver records the interventions it needed (`InterventionRequired`), and a
run with a non-empty list of those is attended by definition.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from anchoropt.learning.round_ledger import (
    OUTCOME_ACCEPTED, OUTCOME_BLOCKED, OUTCOME_NO_CANDIDATE, OUTCOME_REJECTED,
    OUTCOME_SELECTED_ONLY, RETRYABLE, RoundLedger,
)

MAX_ROUNDS = "max_rounds_reached"
NO_CANDIDATE = "no_candidate_space_exhausted"
CONSECUTIVE_NULLS = "consecutive_rounds_concluded_nothing"
REMINE_FAILED = "remine_failed_next_residual_would_be_stale"
STOP_REQUESTED = "stop_requested_by_caller"


@dataclass(frozen=True)
class InterventionRequired:
    """Something the loop could not do for itself. A run with any of these was NOT unattended."""

    round_index: int
    what: str
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"round {self.round_index}: {self.what}" + (f" -- {self.detail}" if self.detail else "")


@dataclass(frozen=True)
class RoundOutcome:
    """One round, as the driver saw it."""

    round_index: int
    outcome: str
    label: str = ""
    net: int | None = None
    accepted: bool = False
    reason: str = ""
    incumbent_id: str = ""
    remined: bool = False
    elapsed_s: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {"round_index": self.round_index, "outcome": self.outcome, "label": self.label,
                "net": self.net, "accepted": self.accepted, "reason": self.reason,
                "incumbent_id": self.incumbent_id, "remined": self.remined,
                "elapsed_s": round(self.elapsed_s, 3)}


@dataclass
class RunResult:
    """The whole run. `terminal` always states WHY it stopped."""

    rounds: list[RoundOutcome] = field(default_factory=list)
    ledger: RoundLedger = field(default_factory=RoundLedger)
    terminal: str = ""
    interventions: list[InterventionRequired] = field(default_factory=list)
    final_incumbent: Any = None

    @property
    def accepted_labels(self) -> tuple[str, ...]:
        return tuple(r.label for r in self.rounds if r.accepted)

    @property
    def unattended(self) -> bool:
        """True only if the loop needed nothing from a human DURING the run.

        This is NOT a claim that the run required no human setup -- it plainly did. It says only
        that no round stopped for an intervention.
        """
        return not self.interventions

    def to_json(self) -> dict[str, Any]:
        return {"terminal": self.terminal,
                "rounds": [r.to_json() for r in self.rounds],
                "accepted_labels": list(self.accepted_labels),
                "interventions": [{"round_index": i.round_index, "what": i.what,
                                   "detail": i.detail} for i in self.interventions],
                "no_intervention_during_run": self.unattended,
                "ledger": self.ledger.to_json()}

    def summary(self) -> str:
        lines = [f"terminal={self.terminal}  rounds={len(self.rounds)}  "
                 f"accepted={len(self.accepted_labels)}"]
        for r in self.rounds:
            net = "" if r.net is None else f" net={r.net:+d}"
            lines.append(f"  R{r.round_index} {r.outcome}{net} {r.label}"
                         + ("  [REMINED]" if r.remined else ""))
        lines.append(self.ledger.summary())
        if self.interventions:
            lines.append(f"  INTERVENTIONS REQUIRED ({len(self.interventions)}) -- this run was "
                         f"NOT unattended:")
            lines.extend(f"    {i}" for i in self.interventions)
        return "\n".join(lines)


def _default_label_of(result) -> str:
    """The arm's label, from whatever the step result carries. Never invents one."""
    cand = getattr(result, "selected_candidate", None)
    for attr in ("label", "name", "candidate_id"):
        v = getattr(cand, attr, None)
        if v:
            return str(v)
    if isinstance(cand, Mapping):
        for k in ("label", "name", "candidate_id"):
            if cand.get(k):
                return str(cand[k])
    return ""


def run_rounds(incumbent: Any, *,
               step_fn: Callable[[Any, RoundLedger], Any],
               remine: Callable[[Any], None] | None = None,
               label_of: Callable[[Any], str] = _default_label_of,
               report_of: Callable[[Any], Any] | None = None,
               max_rounds: int = 3,
               patience: int = 2,
               ledger: RoundLedger | None = None,
               on_round: Callable[[RoundOutcome], None] | None = None,
               should_continue: Callable[[RunResult], bool] | None = None) -> RunResult:
    """Drive rounds until a stated terminal condition. Returns the full run record.

    `step_fn` receives the CURRENT incumbent and the ledger, so the caller's closure can pass
    `ledger.selection_kwargs()` into its own selection and thereby honour what earlier rounds
    concluded. The driver does not select; it only guarantees the ledger is current when asked.

    `report_of(step_result)` supplies the AcceptanceReport for ledger classification. Without it a
    rejection is recorded as a measured loser (the v0.1 default-accept reading), which is the
    CONSERVATIVE direction for eligibility but loses the FAIL/PENDING distinction -- so a run that
    wants retryable arms preserved must supply it.
    """
    led = ledger if ledger is not None else RoundLedger()
    run = RunResult(ledger=led, final_incumbent=incumbent)
    nulls = 0

    for i in range(max_rounds):
        if should_continue is not None and not should_continue(run):
            run.terminal = STOP_REQUESTED
            return run

        t0 = time.time()
        result = step_fn(incumbent, led)
        outcome = str(getattr(result, "outcome", ""))
        label = label_of(result)
        ev = getattr(result, "evaluation", None)
        net = None if ev is None else int(getattr(ev, "net", 0))
        report = report_of(result) if report_of is not None else None

        entry = led.record(label=label, outcome=outcome, report=report, net=net,
                           round_index=i)
        accepted = outcome == OUTCOME_ACCEPTED

        # An acceptance moves the incumbent. Everything after this round is measured against the
        # NEW one, which is the whole point of re-mining before the next diagnosis.
        remined = False
        if accepted:
            new_inc = getattr(result, "new_incumbent", None)
            if new_inc is None:
                run.interventions.append(InterventionRequired(
                    i, "accepted but no new incumbent was returned",
                    "the step's promote seam was not wired, so the next round would re-measure "
                    "against the OLD incumbent and report a stack that was never assembled"))
            else:
                incumbent = new_inc
                run.final_incumbent = new_inc

        rec = RoundOutcome(
            round_index=i, outcome=outcome, label=label, net=net, accepted=accepted,
            reason=(entry.reason if entry is not None else ""),
            incumbent_id=str(getattr(incumbent, "incumbent_id", "") or ""),
            elapsed_s=time.time() - t0)

        if accepted and remine is not None:
            try:
                remine(incumbent)
                remined = True
            except Exception as exc:  # a stale residual is worse than stopping
                run.rounds.append(rec)
                led.advance()
                run.interventions.append(InterventionRequired(
                    i, "re-mining the new incumbent failed", f"{type(exc).__name__}: {exc}"))
                run.terminal = REMINE_FAILED
                return run
            rec = replace(rec, remined=remined)

        run.rounds.append(rec)
        if on_round is not None:
            on_round(rec)
        led.advance()

        # PROGRESS means a conclusion that changes what future rounds are eligible to measure.
        # An acceptance does; a measured loser does; a RETRYABLE entry does not, because the arm
        # stays eligible and the identical round could repeat forever.
        progressed = accepted or (entry is not None and entry.disposition != RETRYABLE)
        nulls = 0 if progressed else nulls + 1

        if outcome == OUTCOME_NO_CANDIDATE:
            run.terminal = NO_CANDIDATE
            return run
        if nulls >= patience:
            run.terminal = CONSECUTIVE_NULLS
            return run

    run.terminal = MAX_ROUNDS
    return run
