"""P3: the JOINER that closes the evaluate -> promote seam. No new algorithm.

THE GAP THIS CLOSES
-------------------
`self_evolve.step()` already implements the whole round -- diagnose, search, materialize, evaluate,
denominator gate, acceptance rule, promote. It accepts `materialize`/`evaluate`/`promote` as INJECTED
seams and returns `OUTCOME_SELECTED_ONLY` when any is absent. Nothing ever supplied them: the only
caller (`examples/self_evolve_tb2.py`) passed `diagnose` and `theta_for` only, so every round in the
repo stopped at selection. Both halves of the missing wiring already existed --

    paired evaluation + denominator integrity   an earlier internal evaluation harness
    moving incumbent (supersede, depth+1)       an earlier internal workflow module

-- and were simply not joined. This module joins them and adds no decision of its own.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
No acceptance rule (that is `self_evolve._default_accept`), no scoring, no threshold, no candidate
search. A second copy of a threshold is how two halves of a loop start disagreeing about what was
decided. Everything here is bookkeeping and plumbing.

DENOMINATOR INTEGRITY IS A PRECONDITION, NOT A RESULT
-----------------------------------------------------
`paired_evaluation` sets `denominator_ok` only when BOTH arms cover exactly the intended cases and
the two independent scorer paths AGREE. `step()` routes a False to OUTCOME_BLOCKED before the
acceptance rule sees it. That ordering is not defensive style: a corrupted denominator previously
produced wreckage that read as a clean null result, and an arm whose case coverage cannot be trusted
must never reach an acceptance decision.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anchoropt.learning.self_evolve import EvaluationResult, Incumbent


# ================================================================================================
# evaluate
# ================================================================================================

@dataclass(frozen=True)
class ArmOutcome:
    """One arm's per-case strict-pass map plus the coverage facts needed to trust it.

    `by_primary` and `by_secondary` are the TWO independent scorer paths. Keeping them separate up
    to the comparison is the whole point -- a single path cannot detect its own missing cases.
    """

    arm: str
    by_primary: Mapping[str, bool]
    by_secondary: Mapping[str, bool]
    telemetry: Mapping[str, Any] = field(default_factory=dict)

    def disagreements(self) -> tuple[str, ...]:
        common = set(self.by_primary) & set(self.by_secondary)
        return tuple(sorted(c for c in common if self.by_primary[c] != self.by_secondary[c]))


def denominator_report(arms: Sequence[ArmOutcome], intended: Sequence[str]) -> tuple[bool, list[str]]:
    """BOTH scorer paths of BOTH arms must cover exactly `intended` AND agree. Guard 3's rule.

    Reproduced here rather than imported because the original reads job directories from disk
    for one specific harness; this takes the already-parsed maps so it works for any benchmark.
    The rule is identical and deliberately so.
    """
    problems: list[str] = []
    want = set(intended)
    for arm in arms:
        for name, d in (("primary", arm.by_primary), ("secondary", arm.by_secondary)):
            missing = want - set(d)
            extra = set(d) - want
            if missing:
                problems.append(f"{arm.arm}/{name}: missing {sorted(missing)}")
            if extra:
                problems.append(f"{arm.arm}/{name}: unexpected {sorted(extra)}")
        for case in arm.disagreements():
            problems.append(f"{arm.arm}/{case}: primary={arm.by_primary[case]} != "
                            f"secondary={arm.by_secondary[case]}")
    return (not problems), problems


def paired_evaluation(control: ArmOutcome, candidate: ArmOutcome,
                      intended: Sequence[str]) -> EvaluationResult:
    """Build the `EvaluationResult` the acceptance rule consumes, from two scored arms.

    Gains and losses are computed on the PAIRED cases only -- cases present in both arms. A case
    missing from one arm cannot be a gain or a loss; it is a denominator problem, and it is reported
    as one rather than silently scored as a regression.
    """
    ok, problems = denominator_report((control, candidate), intended)
    paired = sorted(set(control.by_primary) & set(candidate.by_primary))
    gains = tuple(c for c in paired if candidate.by_primary[c] and not control.by_primary[c])
    losses = tuple(c for c in paired if control.by_primary[c] and not candidate.by_primary[c])

    telemetry = dict(candidate.telemetry)
    telemetry.setdefault("n_paired", len(paired))
    telemetry["denominator_problems"] = problems

    detail = (f"{len(gains)} gains / {len(losses)} losses on {len(paired)} paired cases"
              if ok else f"DENOMINATOR INTEGRITY FAILED: {len(problems)} problem(s)")
    return EvaluationResult(
        control={c: control.by_primary[c] for c in paired},
        candidate={c: candidate.by_primary[c] for c in paired},
        gains=gains, losses=losses, telemetry=telemetry,
        denominator_ok=ok, detail=detail)


def make_evaluator(run_arm: Callable[[str, Any], ArmOutcome], intended: Sequence[str]
                   ) -> Callable[[Incumbent, Any], EvaluationResult]:
    """Adapt a per-arm runner into the `evaluate(incumbent, candidate_ws)` seam `step()` expects.

    `run_arm(arm_name, workspace) -> ArmOutcome` is the benchmark's business: for TB2 it is the
    hardened runner's per-case loop with its infra-retry guard, for BFCL it is the memory evaluator.
    The CONTROL arm is re-run every round rather than reused from the previous round's candidate,
    because the incumbent's world is what the next round must be measured against.
    """
    def evaluate(incumbent: Incumbent, candidate_ws: Any) -> EvaluationResult:
        control = run_arm("control", incumbent.workspace)
        candidate = run_arm("candidate", candidate_ws)
        return paired_evaluation(control, candidate, intended)
    return evaluate


# ================================================================================================
# promote -- the moving incumbent
# ================================================================================================
#
# `create_branch_from_surfaces` semantics, reimplemented over a plain dict so the core does not
# depend on any external harness import. Verified against that function: supersede the active
# branch, depth+1, advance `active_branch_id`, record the accepted candidate. Those four are the
# whole of the moving-incumbent contract.
#
# WHY REIMPLEMENTED RATHER THAN IMPORTED. The original lives in an external workflow script that
# pulls in that project's own paths and queue machinery. Importing it would make an AnchorOpt
# round depend on that external checkout for ~15 lines of dict bookkeeping, so it is reimplemented
# here instead.


def new_branch_state(root_id: str = "P0") -> dict[str, Any]:
    """A fresh branch state whose only branch is the root incumbent, active at depth 0."""
    return {"branches": [{"branch_id": root_id, "parent_branch_id": None, "status": "active",
                          "depth": 0, "created_at": int(time.time())}],
            "active_branch_id": root_id}


def _unique_branch_id(state: Mapping[str, Any], base: str) -> str:
    existing = {str(b.get("branch_id")) for b in state.get("branches", [])}
    if base not in existing:
        return base
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


def promote(state: dict[str, Any], *, parent_id: str, candidate_id: str,
            surfaces: Mapping[str, str], evaluation: EvaluationResult | None = None,
            mechanism_family: str = "") -> dict[str, Any]:
    """Supersede the active branch and advance the incumbent. Returns the new child branch.

    The candidate's OWN evaluation becomes the child's baseline: next round's control arm is this
    round's candidate. That is the moving incumbent, and skipping it is how a loop ends up measuring
    every anchor against the original baseline and reporting a stack that was never assembled.
    """
    parent = next((b for b in state["branches"] if b["branch_id"] == parent_id), None)
    if parent is None:
        raise ValueError(f"unknown parent branch {parent_id!r}")
    for branch in state["branches"]:
        if branch["branch_id"] == state["active_branch_id"]:
            branch["status"] = "superseded"
    child = {
        "branch_id": _unique_branch_id(state, f"{parent_id}+{candidate_id}"),
        "parent_branch_id": parent_id,
        "status": "active",
        "depth": int(parent.get("depth", 0)) + 1,
        "eval_surfaces": dict(surfaces),
        "proposer_surfaces": dict(surfaces),
        "accepted_candidate_id": candidate_id,
        "accepted_mechanism_family": mechanism_family,
        "created_at": int(time.time()),
    }
    if evaluation is not None:
        child["baseline_result"] = {
            "gains": list(evaluation.gains), "losses": list(evaluation.losses),
            "net": evaluation.net, "n_paired": len(evaluation.control),
            "denominator_ok": evaluation.denominator_ok,
        }
    state["branches"].append(child)
    state["active_branch_id"] = child["branch_id"]
    return child


def validate_branch_state(state: Mapping[str, Any]) -> None:
    """Exactly one active branch, and it is `active_branch_id`. Cheap, and it catches a real bug:
    a promotion that forgets to supersede leaves two active branches and the next round re-mines
    against the OLD incumbent."""
    active = [b["branch_id"] for b in state["branches"] if b.get("status") == "active"]
    if len(active) != 1:
        raise ValueError(f"expected exactly 1 active branch, found {active}")
    if active[0] != state.get("active_branch_id"):
        raise ValueError(f"active_branch_id {state.get('active_branch_id')!r} != {active[0]!r}")


def make_promoter(state: dict[str, Any], state_path: Path | None = None,
                  mechanism_family: str = "") -> Callable[..., Incumbent]:
    """Adapt `promote` into the `promote(incumbent, candidate_ws)` seam `step()` expects."""
    def do_promote(incumbent: Incumbent, candidate_ws: Any,
                   evaluation: EvaluationResult | None = None,
                   candidate_id: str = "candidate") -> Incumbent:
        child = promote(state, parent_id=incumbent.incumbent_id, candidate_id=candidate_id,
                        surfaces={"baseline": str(candidate_ws)}, evaluation=evaluation,
                        mechanism_family=mechanism_family)
        validate_branch_state(state)
        if state_path is not None:
            Path(state_path).write_text(json.dumps(state, indent=2) + "\n")
        return Incumbent(workspace=candidate_ws, incumbent_id=child["branch_id"],
                         depth=child["depth"])
    return do_promote
