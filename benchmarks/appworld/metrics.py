"""Recomputes the AppWorld x AnchorOpt metric table from what is on disk. One command, no LLM calls.

WHY THIS IS A GENERATOR AND NOT A CHECKED-IN TABLE OF NUMBERS
--------------------------------------------------------------------------------------------------
Every number below already exists somewhere on disk: the baselines in the round-1 `baseline_eval`
fragments, the residual in each split's `environment_io.md` logs, the search state in whatever
`optimize_residual` returns when re-run against them. Transcribing them into a committed table
would create exactly the second source of truth this repo's own `.gitignore` refuses for
`verification_report.json` ("recomputed on demand ... committing it would create a second source of
truth that can go stale"). So this file recomputes, and `--out` is where the caller puts the result.

WHAT "HELD-IN" AND "HELD-OUT" MEAN HERE, AND WHY THEY ARE NOT NEW SPLITS
--------------------------------------------------------------------------------------------------
They are an earlier internal harness's existing pair, reused verbatim rather than re-carved:
held-in is AppWorld's own official `train` split (90 tasks / 30 scenarios) and held-out is
`sh_heldout` (90 tasks / 30 scenarios, drawn from `test_normal` + `test_challenge` BY SCENARIO,
SEED=42 -- because SGC is a mean over scenarios of min(per-task success), so a task-level draw
would fragment scenarios and inflate it). Both are already baselined twice per model with full
per-task logs, which is why no new baseline run is needed.

WHERE THE BASELINE ARTIFACTS ACTUALLY LIVE -- NOT THE SHARED APPWORLD TREE
--------------------------------------------------------------------------------------------------
`$APPWORLD_DEV_ROOT/experiments/outputs/...` holds only the `dev` runs. The `train`/`sh_heldout`
runs live inside each round cell under `$APPWORLD_ROUNDS_DIR`, and each cell directory is itself a
valid `APPWORLD_ROOT` (it has its own `data/` and `experiments/outputs/...`). So this file switches
`path_store` to the relevant cell per model/split via `path_store.update_root`, which is the whole
reason the adapter can mine these splits with no code change and no new episode. Set both env vars
to point at your own round-cell tree -- see `benchmarks/appworld/env.sh`.

Round 1 cells only: later rounds' baselines for some models reflect that harness's own accepted
GEPA edits, so their "baseline" is no longer the stock harness and is not a valid incumbent for an
AnchorOpt comparison.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from typing import Any

from appworld.common.path_store import path_store

from anchoropt.learning.search_state import next_coordinate
from anchoropt.learning.structured_search import optimize_residual

ROUNDS_DIR = os.environ.get("APPWORLD_ROUNDS_DIR", os.path.expanduser("~/appworld_rounds"))
DEV_ROOT = os.environ.get("APPWORLD_DEV_ROOT", os.path.expanduser("~/appworld"))

# round_key -> the AppWorld (creator, model) that round's cells actually evaluated
MODELS: dict[str, dict[str, str]] = {
    "minimax-m2-5": {"round_key": "minimax", "creator": "minimax"},
    "qwen3.6-35b-a3b": {"round_key": "qwen", "creator": "alibaba"},
    "granite-4.1-30b": {"round_key": "granite30b", "creator": "ibm"},
}
SPLITS: dict[str, str] = {"held_in": "train", "held_out": "sh_heldout"}


def _fragments(round_key: str, split: str) -> list[dict[str, Any]]:
    """Every repeat's measured summary for one model/split, from round 1's baseline eval."""
    pattern = os.path.join(ROUNDS_DIR, round_key, "round1", "baseline_eval", "fragments", "*.json")
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            frag = json.load(f)
        if frag.get("split") == split:
            frag["_cell"] = os.path.splitext(os.path.basename(path))[0]
            out.append(frag)
    return sorted(out, key=lambda f: f.get("repeat", 0))


def _baseline(round_key: str, split: str) -> dict[str, Any]:
    frags = _fragments(round_key, split)
    if not frags:
        return {"status": "MISSING", "note": f"no round-1 baseline fragment for split {split!r}"}
    tgc = [f["task_goal_completion"] for f in frags]
    sgc = [f["scenario_goal_completion"] for f in frags]
    return {
        "status": "MEASURED",
        "tasks": frags[0]["total"],
        "repeats": [
            {
                "repeat": f.get("repeat"),
                "cell": f["_cell"],
                "passed": f["passed"],
                "total": f["total"],
                "task_goal_completion": f["task_goal_completion"],
                "scenario_goal_completion": f["scenario_goal_completion"],
            }
            for f in frags
        ],
        "task_goal_completion_mean": round(statistics.fmean(tgc), 2),
        "scenario_goal_completion_mean": round(statistics.fmean(sgc), 2),
    }


def _tgc_sgc(vec: dict[str, bool]) -> tuple[float, float]:
    """AppWorld's two headline metrics from a per-task correctness vector.

    TGC is the per-task pass rate. SGC is the mean over SCENARIOS of min(per-task success) -- i.e.
    the fraction of scenarios where every task passes -- which is why it is the stricter of the two
    and why a controller that fixes one task in a scenario while breaking another moves TGC by zero
    but can still move SGC. Both formulas are validated against an earlier internal harness's own
    recorded fragment numbers for minimax/qwen on train and sh_heldout (exact match on TGC and SGC).
    """
    if not vec:
        return 0.0, 0.0
    tgc = 100.0 * sum(vec.values()) / len(vec)
    scenarios: dict[str, list[bool]] = {}
    for task_id, ok in vec.items():
        scenarios.setdefault(task_id.split("_")[0], []).append(ok)
    sgc = 100.0 * sum(1 for tasks in scenarios.values() if all(tasks)) / len(scenarios)
    return round(tgc, 2), round(sgc, 2)


def _outcome(evaluation_path: str, results_path: str | None) -> dict[str, Any]:
    """The before/after picture on ONE split: the incumbent's own TGC/SGC, and what each measured
    arm turns them into.

    The after-vector is the incumbent's correctness vector with that arm's `gains` flipped to True
    and its `losses` flipped to False. That reconstruction is exact rather than approximate ONLY when
    the arm was scored over the whole split -- which is what `--check-regressions` does (residual +
    every already-passing task, n = all 90). Without it an arm's `losses` is unconditionally empty
    because nothing outside the residual was run, so `denominator_ok` and `n` are reported here too:
    an arm whose `n` is short of the split has an after-number that is a floor, not a measurement.
    """
    with open(evaluation_path, encoding="utf-8") as f:
        individual = (json.load(f).get("individual") or {})
    before = {tid: bool(rec.get("success")) for tid, rec in individual.items()}
    b_tgc, b_sgc = _tgc_sgc(before)
    out: dict[str, Any] = {
        "baseline": {
            "passed": sum(before.values()), "total": len(before),
            "task_goal_completion": b_tgc, "scenario_goal_completion": b_sgc,
        },
        "arms": [],
    }
    if not (results_path and os.path.exists(results_path)):
        out["status"] = "NO_ARM_MEASURED"
        return out

    with open(results_path, encoding="utf-8") as f:
        payload = json.load(f)
    for rec in sorted(payload.get("results") or [], key=lambda r: r.get("arm_label", "")):
        gains = tuple(rec.get("gains") or ())
        losses = tuple(rec.get("losses") or ())
        after = dict(before)
        for tid in gains:
            after[tid] = True
        for tid in losses:
            after[tid] = False
        a_tgc, a_sgc = _tgc_sgc(after)
        out["arms"].append({
            "arm_label": rec.get("arm_label"),
            "scored_n": rec.get("n"),
            "covers_whole_split": rec.get("n") == len(before),
            "denominator_ok": rec.get("denominator_ok"),
            "cases_fired": rec.get("cases_fired"),
            "interventions_executed": rec.get("interventions_executed"),
            "gains": len(gains), "losses": len(losses), "net": len(gains) - len(losses),
            "after_passed": sum(after.values()),
            "after_task_goal_completion": a_tgc,
            "after_scenario_goal_completion": a_sgc,
            "delta_tgc_pp": round(a_tgc - b_tgc, 2),
            "delta_sgc_pp": round(a_sgc - b_sgc, 2),
        })
    out["status"] = "MEASURED" if out["arms"] else "NO_ARM_MEASURED"
    return out


def _cell_root(round_key: str, split: str, repeat: int = 1) -> str:
    suffix = "r" + str(repeat)
    return os.path.join(ROUNDS_DIR, round_key, "round1", "baseline_eval", "cells", f"{split}_{suffix}")


def _evaluation_path(root: str, creator: str, model: str, split: str) -> str:
    return os.path.join(
        root, "experiments", "outputs", "simplified_react_code_agent",
        creator, model, split, "evaluations", f"{split}.json",
    )


def _search(evaluation_path: str, results_path: str | None) -> dict[str, Any]:
    """One `optimize_residual` step against this split's own residual, and whatever has been measured
    for it so far. Imported lazily so `--help` and the baseline-only path need no AppWorld agent deps.
    """
    from evaluate import incumbent_identity, load_evaluation  # noqa: PLC0415
    from residual import _infer_experiment_name, mine_residual  # noqa: PLC0415

    from adapter import ADAPTER  # noqa: PLC0415

    experiment_name = _infer_experiment_name(evaluation_path)
    residual, records = mine_residual(evaluation_path, experiment_name)
    _incumbent_id, incumbent_token = incumbent_identity(evaluation_path, experiment_name)
    evaluation = load_evaluation(results_path, incumbent_token=incumbent_token)

    outcome = optimize_residual(
        residual, runtime=ADAPTER, host=ADAPTER.HOST, events=records, states=records,
        evaluate=evaluation, improves=lambda res: res is not None and res.net > 0,
    )
    return {
        "residual_tasks": len(residual.case_ids),
        "mined_records": len(records),
        "state": outcome.state,
        "next_coordinate": next_coordinate(outcome.state),
        "boundaries_visited": list(outcome.visited),
        "arms_measured": len(evaluation.served),
        "arms_unmeasured": len(evaluation.missing),
        "candidates": [
            {"arm_label": a.label, "boundary": a.boundary.value, "action": a.action.value,
             "signal": a.signal}
            for a in outcome.candidates
        ],
        "promoted": (
            {"arm_label": outcome.promoted.label,
             "boundary": outcome.promoted.boundary.value,
             "action": outcome.promoted.action.value,
             "signal": outcome.promoted.signal,
             "eta": dict(outcome.promoted.eta)}
            if outcome.state == "IMPROVED" and outcome.promoted is not None else None
        ),
        "attempts": [
            {"boundary": a.boundary, "state": a.state, "built": a.candidates_built,
             "evaluated": a.candidates_evaluated}
            for a in outcome.attempts
        ],
    }


def build(results_dir: str | None) -> dict[str, Any]:
    original_root = path_store.root
    report: dict[str, Any] = {
        "splits": {
            "held_in": {"dataset": "train", "tasks": 90, "scenarios": 30,
                        "provenance": "AppWorld's own official train split"},
            "held_out": {"dataset": "sh_heldout", "tasks": 90, "scenarios": 30,
                         "provenance": "scenario-stratified draw from test_normal + test_challenge, "
                                       "SEED=42"},
        },
        "incumbent": "round-1 baseline_eval = the stock AppWorld "
                     "simplified_react_code_agent, before any harness edit",
        "models": {},
    }

    try:
        for model, meta in MODELS.items():
            entry: dict[str, Any] = {"creator": meta["creator"], "baseline": {}, "anchoropt": {}}
            for label, split in SPLITS.items():
                entry["baseline"][label] = _baseline(meta["round_key"], split)

                root = _cell_root(meta["round_key"], split)
                evaluation_path = _evaluation_path(root, meta["creator"], model, split)
                if not os.path.exists(evaluation_path):
                    entry["anchoropt"][label] = {
                        "status": "MISSING", "note": f"no evaluation at {evaluation_path}"
                    }
                    continue

                # Round dirs are named by round_key + split role, matching what the score jobs
                # actually write (e.g. runs/minimax_heldin/arm_results.json).
                results_path = None
                if results_dir:
                    suffix = "heldin" if label == "held_in" else "heldout"
                    cand = os.path.join(
                        results_dir, f"{meta['round_key']}_{suffix}", "arm_results.json"
                    )
                    results_path = cand if os.path.exists(cand) else None

                entry.setdefault("outcome", {})[label] = _outcome(evaluation_path, results_path)

                path_store.update_root(root)
                try:
                    entry["anchoropt"][label] = {
                        "status": "SEARCHED", "appworld_root": root,
                        "evaluation": evaluation_path,
                        "measured_results": results_path,
                        **_search(evaluation_path, results_path),
                    }
                except Exception as exc:  # reported, never silently dropped
                    entry["anchoropt"][label] = {
                        "status": "ERROR", "error": f"{type(exc).__name__}: {exc}",
                        "appworld_root": root,
                    }
            report["models"][model] = entry
    finally:
        path_store.update_root(original_root)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="write the metric JSON here (default: stdout)")
    parser.add_argument(
        "--results-dir", default=None,
        help="directory holding <model>_<held_in|held_out>/arm_results.json from `run_round.py "
             "score`; without it every arm reads as unmeasured, which is a real state, not a blank",
    )
    args = parser.parse_args(argv)

    report = build(args.results_dir)
    text = json.dumps(report, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
