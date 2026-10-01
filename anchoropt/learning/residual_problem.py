"""RESIDUAL PROBLEMS as first-class objects, so the ranking that already exists is ENFORCED.

THE DEFECT THIS FIXES
---------------------
The residual ranking was computed and then discarded. `rank_by_support` returned an ordered tuple,
and that order reached the proposer only as list order in a prompt; `candidate_search.search()` then
enumerated candidates across EVERY diagnosis and ranked the CONTROLLERS globally. Its docstring says
so plainly -- "Select across EVERY diagnosis's candidates, not from whichever diagnosis happened to
come first" -- which fixed a real bug (brief order deciding the round) by flattening the hierarchy.

Measured consequence, from the R1 audit: the top residual (support 4, 50% coverage) had exactly one
candidate covering all its cases, that candidate was infeasible, and nothing then searched an
alternative action for it. The evaluation budget went to a candidate covering 1 of 4 cases of that
residual, which survived because it was INSTANTIABLE. The intended hierarchy is:

    rank residuals -> freeze R1 -> search policies for R1 only -> measure -> promote or exhaust -> R2

and controller ranking belongs strictly INSIDE a chosen residual.

WHAT THIS MODULE DOES NOT CHANGE
--------------------------------
The priority METRIC. Ranking is still `compute_support` -- the count of diagnoses sharing a
normalized `consequential_decision` -- exactly as before, deliberately. One architectural change at a
time: this enforces the existing ranking rather than improving it. `compute_support`'s own docstring
already records that linked downstream loss is the stronger measure it cannot compute, and that stays
true and stays stated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anchoropt.runtime import ResidualDiagnosis

# Terminal states for one residual's policy search. Kept apart because they lead somewhere different.
POLICY_ACCEPTED = "POLICY_ACCEPTED"
POLICY_EXHAUSTED_CURRENT_SPACE = "POLICY_EXHAUSTED_CURRENT_SPACE"
SIGNAL_BLOCKED_RESIDUAL = "SIGNAL_BLOCKED_RESIDUAL"


@dataclass(frozen=True)
class ResidualProblem:
    """One prioritized residual failure mode: the unit the outer loop iterates over.

    The unit is the PROBLEM, not the controller. `key` is the normalized consequential decision the
    member diagnoses share -- the same grouping `compute_support` already uses, so no second
    definition of "same failure" can drift from the first.
    """

    key: str
    diagnoses: tuple[ResidualDiagnosis, ...]
    rank: int = 0
    coverage: float = 0.0                 # share of the residual this problem accounts for (S1)
    saturation: float = 0.0               # share already settled by the incumbent (S2)
    expressible: bool = True              # can ANY declared signal observe it (S3)
    phase_note: str = ""

    @property
    def support(self) -> int:
        return len(self.diagnoses)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(d.case_id for d in self.diagnoses)

    def scoped_diagnoses(self) -> tuple[ResidualDiagnosis, ...]:
        """The ONLY diagnoses a proposer may see while this problem is frozen.

        Load-bearing: passing the whole residual lets the proposer answer about a different, easier
        problem, and that is how the R1 round ended up optimizing a 1-case locus.
        """
        return self.diagnoses

    def __str__(self) -> str:
        return (f"R{self.rank}[support={self.support} cov={100*self.coverage:.1f}% "
                f"{'expressible' if self.expressible else 'SIGNAL_BLOCKED'}] {self.key[:60]}")


def build_residual_problems(diagnoses: Sequence[ResidualDiagnosis], *,
                            expressible: Callable[[ResidualDiagnosis], bool] | None = None,
                            engaged: Mapping[str, int] | None = None,
                            ) -> tuple[ResidualProblem, ...]:
    """Group diagnoses into ranked residual problems. UNCHANGED metric: support, then key.

    Ordering is `(-support, key)` -- the same as `rank_by_support`'s `(-support, case_id)`, lifted
    from diagnoses to groups. Deterministic, so a rerun searches the same problem first.

    A problem no declared signal can observe is RETAINED and marked inexpressible rather than dropped:
    it belongs to the SIGNAL block's backlog, and dropping it here would lose the mechanism text.
    """
    from anchoropt.learning.proposal_seams import compute_support

    grouped: dict[str, list[ResidualDiagnosis]] = {}
    for d in diagnoses:
        key = " ".join(str(d.consequential_decision).lower().split())
        grouped.setdefault(key, []).append(d)

    total = max(len(diagnoses), 1)
    engaged = engaged or {}
    ordered = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    out: list[ResidualProblem] = []
    for i, (key, members) in enumerate(ordered, start=1):
        expr = True if expressible is None else any(expressible(d) for d in members)
        out.append(ResidualProblem(
            key=key, diagnoses=tuple(members), rank=i,
            coverage=len(members) / total,
            saturation=engaged.get(key, 0) / max(len(members), 1),
            expressible=expr,
            phase_note=("expressible under the current vocabulary" if expr
                        else "no declared signal observes this -- SIGNAL-block backlog")))
    return tuple(out)


@dataclass
class ResidualSearchLedger:
    """What each prioritized residual has had TRIED, so exhaustion is proven rather than assumed.

    A residual leaves the queue only when every (l, phi) the proposer offers for it has had its full
    action family instantiated and either measured-and-rejected or proven infeasible. One infeasible
    action is not exhaustion, and the R1 audit is what this exists to prevent recurring.
    """

    problem: ResidualProblem
    attempted_loci: list[tuple[str, str]] = field(default_factory=list)   # (boundary, signal)
    infeasible: list[tuple[str, str, str]] = field(default_factory=list)  # (locus, action, reason)
    measured: list[tuple[str, int]] = field(default_factory=list)         # (arm label, net)
    accepted: Any = None
    state: str = ""

    def note_locus(self, boundary: str, signal: str) -> None:
        cell = (boundary, signal)
        if cell not in self.attempted_loci:
            self.attempted_loci.append(cell)

    def note_infeasible(self, locus: str, action: str, reason: str) -> None:
        self.infeasible.append((locus, action, reason))

    def note_measured(self, label: str, net: int) -> None:
        self.measured.append((label, net))

    def exhausted_reason(self) -> str:
        """Why this residual is leaving the queue -- stated, never inferred from a single failure.

        `POLICY_EXHAUSTED_CURRENT_SPACE` is deliberately scoped: under the CURRENT Phi, host
        capabilities, proposer-offered loci and action contracts. It is not a claim of global
        impossibility, and a residual that no signal can express is recorded as SIGNAL-block backlog
        instead, because those two lead to different next steps.
        """
        if not self.problem.expressible:
            return (f"{SIGNAL_BLOCKED_RESIDUAL}: no declared signal observes this residual, so no "
                    f"policy can be proposed for it. PRESERVED for the SIGNAL block -- this is a "
                    f"representation limit, not a solved or impossible problem.")
        return (f"{POLICY_EXHAUSTED_CURRENT_SPACE}: {len(self.attempted_loci)} locus/loci tried, "
                f"{len(self.measured)} arm(s) measured and rejected, {len(self.infeasible)} "
                f"infeasible cell(s). Scoped to the CURRENT Phi, host capabilities, "
                f"proposer-offered intervention set and action contracts -- NOT a global claim.")

    def summary(self) -> str:
        lines = [f"  {self.problem}", f"    state: {self.state or '(in progress)'}"]
        for b, s in self.attempted_loci:
            lines.append(f"    tried locus: {b} / {s}")
        for locus, action, reason in self.infeasible:
            lines.append(f"    INFEASIBLE {locus} {action}: {reason[:78]}")
        for label, net in self.measured:
            lines.append(f"    measured {label}: net {net:+d}")
        return "\n".join(lines)
