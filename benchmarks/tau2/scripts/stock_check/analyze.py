#!/usr/bin/env python3
"""Stock-pipeline check, qwen3.6 / retail / test: does AnchorOpt's runner score stock the way tau-bench's own
runner (as GEPA calls it) does?

    python analyze.py        # STOCKCHECK_DIR: where run_check.sh wrote (default below)

NEW (run side by side, same endpoint, same time, concurrency 3 each):
  ours        AnchorOpt `validate --p0-only`   ours_{a,b}/heldout_p0_test_rep{1,2}.json
  tau2runner  GEPA's `eval_prompt.py --stock`  tau2runner/stock_seed{301..304}.json   (tau2 run_tasks)
EARLIER (for context): AnchorOpt's 4 committed stock runs, GEPA's 2 stock runs, Self-Harness's 2.
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
from pathlib import Path

# Every path here is site-local and has no sensible default: the originals were specific users' home
# and project directories on our filesystem. Set the ones you have; the comparisons that need a path
# you have not set are skipped rather than reading someone else's tree.
HERE = Path(os.environ.get("STOCKCHECK_DIR", ""))
REPO = Path(os.environ.get("STOCKCHECK_REPO", ""))
AO_OLD = REPO / "benchmarks/tau2/rounds/SELFTEACH_R1/qwen3_6_35b_a3b__self__retail"
GEPA = Path(os.environ.get("STOCKCHECK_GEPA_DIR", ""))
SH = Path(os.environ.get("STOCKCHECK_SH_DIR", ""))
CLEAN = {"user_stop", "agent_stop"}

if not str(HERE):
    raise SystemExit(
        "STOCKCHECK_DIR is not set: point it at the directory run_check.sh wrote into. "
        "STOCKCHECK_REPO / STOCKCHECK_GEPA_DIR / STOCKCHECK_SH_DIR are optional comparison trees.")


def _j(p):
    return json.loads(Path(p).read_text())


def ours(paths):
    runs = []
    for p in paths:
        r = _j(p)
        void = {c for c, t in r["termination"].items() if str(t).startswith("harness_error")}
        runs.append(({str(c): bool(s) for c, s in r["solved"].items() if c not in void}, len(void)))
    return runs


def gepa_style(paths):
    runs = []
    for p in paths:
        e = _j(p)
        runs.append(({str(t["task_id"]): bool(t["strict_pass"]) for t in e["tasks"]}, e["n_infra_errors"]))
    return runs


def sh():
    runs = []
    for r in (1, 2):
        sims = _j(SH / f"baseline_eval/cells/test_r{r}/results.json")["simulations"]
        runs.append(({str(s["task_id"]): (s.get("reward_info") or {}).get("reward") == 1.0
                      and s.get("termination_reason") in CLEAN for s in sims}, 0))
    return runs


def summary(name, runs, tasks):
    tot = [sum(r[t] for t in tasks) for r, _ in runs]
    sd = st.stdev(tot) if len(tot) > 1 else float("nan")
    print(f"  {name:44} runs {tot}  mean {st.fmean(tot):5.2f}/{len(tasks)}  "
          f"= {100 * st.fmean(tot) / len(tasks):5.1f}%  sd {sd:4.2f}  void {[v for _, v in runs]}")
    return tot


def compare(label, a_runs, b_runs, tasks):
    """b - a: by run totals (between-run noise only) and paired over tasks (tasks as the unit)."""
    ta = [sum(r[t] for t in tasks) for r, _ in a_runs]
    tb = [sum(r[t] for t in tasks) for r, _ in b_runs]
    d = st.fmean(tb) - st.fmean(ta)
    se_runs = math.sqrt((st.variance(ta) if len(ta) > 1 else 0) / len(ta)
                        + (st.variance(tb) if len(tb) > 1 else 0) / len(tb))
    ra = {t: st.fmean(r[t] for r, _ in a_runs) for t in tasks}
    rb = {t: st.fmean(r[t] for r, _ in b_runs) for t in tasks}
    diffs = [rb[t] - ra[t] for t in tasks]
    se_tasks = st.stdev(diffs) / math.sqrt(len(diffs)) * len(tasks)
    z = lambda se: d / se if se else float("nan")          # noqa: E731
    print(f"  {label:44} {d:+5.2f} tasks = {100 * d / len(tasks):+5.1f} pts   "
          f"se(runs) {se_runs:4.2f} z {z(se_runs):+5.2f}   se(tasks, paired) {se_tasks:4.2f} z {z(se_tasks):+5.2f}")
    return ra, rb


def main():
    new_ours = ours(sorted(HERE.glob("ours_*/heldout_p0_test_rep*.json")))
    new_t2 = gepa_style(sorted((HERE / "tau2runner").glob("stock_seed*.json")))
    old_ours = ours(sorted(AO_OLD.glob("heldout_p0_test_rep*.json")))
    old_gepa = gepa_style([GEPA / "retail__qwen36__self/eval_stock_test.json",
                           GEPA / "retail__qwen36__sonnet/eval_stock_test.json"])
    old_sh = sh()
    groups = {"new: AnchorOpt runner": new_ours, "new: tau-bench runner (GEPA's eval_prompt)": new_t2,
              "earlier: AnchorOpt (4 runs)": old_ours, "earlier: GEPA (2 runs)": old_gepa,
              "earlier: Self-Harness (2 runs)": old_sh}
    have = {k: v for k, v in groups.items() if v}
    tasks = sorted(set.intersection(*[set(r) for v in have.values() for r, _ in v]))
    print(f"\nqwen3.6 / retail / test -- {len(tasks)} tasks scored in every run\n")
    for k, v in have.items():
        summary(k, v, tasks)
    if new_ours and new_t2:
        print("\nTHE CHECK (run side by side):")
        ra, rb = compare("tau-bench runner - AnchorOpt runner", new_ours, new_t2, tasks)
        print("\nPOOLED with the earlier runs of each pipeline:")
        compare("GEPA-style (new 4 + GEPA's 2) - AnchorOpt (new + old 4)", new_ours + old_ours,
                new_t2 + old_gepa, tasks)
        print("\nDRIFT within each pipeline (new - earlier):")
        compare("AnchorOpt runner, new - earlier", old_ours, new_ours, tasks)
        compare("GEPA-style, new tau-bench runner - GEPA's earlier", old_gepa, new_t2, tasks)
        worst = sorted(tasks, key=lambda t: rb[t] - ra[t])
        print("\ntasks where the two new runners disagree most (success rate, AnchorOpt vs tau-bench runner):")
        for t in worst[:4] + worst[-6:]:
            if abs(rb[t] - ra[t]) >= 0.5:
                print(f"  task {t:>4}: {ra[t]:.2f} vs {rb[t]:.2f}")


if __name__ == "__main__":
    main()
