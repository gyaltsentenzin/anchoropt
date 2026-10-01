"""WHOLE-CORPUS recovery table: accuracy of an ACTUALLY EXECUTED stack, against the historical ladder.

NAMED `corpus_recovery_table` and NOT `recovery_table`, because `scripts/recovery_table.py` already
exists and does something different -- it prints the ANCHOR-RECOVERY table from rounds/REC_* and owns
the NOT_TESTABLE-vs-failure distinction. I overwrote it, which broke two of its tests, and the tests
caught it. Two scripts, two questions: that one asks "did the search rediscover the anchor's
structure", this one asks "what does an executed stack score on the whole corpus".

WHY THIS IS A SCRIPT AND NOT ARITHMETIC IN A REPORT
---------------------------------------------------
The historical ladder is WHOLE-CORPUS (n = 303 = kv 105 + vector 89 + rec_sum 109). Every measurement in
this workstream has been per-cell. Combining per-cell results by hand is how "104/303 = 34.32%" came to be
quoted as a recovery figure when the composed stack had never been run -- three separately measured
deltas, added.

So this script will only compute a whole-corpus number from **run directories it can see**, one per cell,
for the SAME stack. If a cell is missing it says so and refuses to report a total. There is no code path
that adds per-cell deltas measured at different times.

WHAT IT REFUSES TO DO
---------------------
* No total from fewer than all three cells -- a two-cell "whole corpus" figure is not one.
* No arm/control pairing across different case sets: the per-cell denominators are checked against the
  canonical split (105/89/109) and a mismatch is reported, not silently accepted.
* No recovery percentage unless the arm's controllers actually FIRED in the cells they are credited with.
  `--require-firings` reads the sidecar and blocks the row otherwise, because a cell where nothing fired
  contributes an unattributed delta.

THE HISTORICAL LADDER is recorded here as data, from docs/ANCHORS.md and docs/THE_LOOP.md, so a recovery
percentage always has its denominator attached.

Usage:
  python scripts/corpus_recovery_table.py --results /tmp/r9 \
      --arm-tags kv=r9_all3_kv,vector=r9_all3_vector,rec_sum=r9_all3_rec_sum \
      --control-tags kv=auto_ctl,vector=r9_h0_vector,rec_sum=r9_h0_rec_sum \
      --label "A1+A4+A7 composed"
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: The canonical cell split. A denominator that disagrees is a different population, not a rounding issue.
CANONICAL = {"kv": 105, "vector": 89, "rec_sum": 109}
CORPUS_N = sum(CANONICAL.values())          # 303

#: H0, measured. rounds/H0_SESSION/FINDINGS.md, verified on BV in BV_VERIFICATION.md.
H0 = {"kv": 18, "vector": 19, "rec_sum": 54}
H0_TOTAL = sum(H0.values())                 # 91 -> 30.03%

#: The historical ladder, whole corpus. docs/ANCHORS.md + docs/THE_LOOP.md.
LADDER = [("H0", None, 30.03), ("T1 A1", "+3.63", 33.66), ("T2 A2", "+0.99", 34.65),
          ("T3 A3", "+3.96", 38.61), ("T5 A4", "+3.63", 42.24), ("T6 A5", "+1.32", 43.56),
          ("T7 A7", "+2.31", 45.87), ("T8 A8", "+1.65", 47.52), ("T9 A9", "+4.95", 52.48)]
HIST_TARGET = 52.48
HIST_GAIN = HIST_TARGET - 30.03             # +22.45

#: Keys meaning an intervention ACTED. Hook/decline bookkeeping is deliberately excluded: a
#: `relocate_requested_gate` in vector means the hook was consulted, not that a relocation happened, and
#: counting it would report cross-cell activity that did not occur.
EFFECT_KEYS = ("relocate_gate", "upstream_withheld_gate", "upstream_zero_call_gate",
               "capacity_repair_gate", "dedup_clear_gate")


def load_cell(run: pathlib.Path) -> tuple[int, int] | None:
    """(correct, n) over SCORED query cases, or None when the run has no results."""
    for pat in ("run/eval_*results*.json", "eval_*results*.json"):
        hits = sorted(glob.glob(str(run / pat)))
        if hits:
            payload = json.loads(pathlib.Path(hits[0]).read_text())
            rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
            q = [r for r in rows if not r.get("is_prereq")]
            return sum(bool(r.get("valid")) for r in q), len(q)
    return None


def effects(run: pathlib.Path) -> dict[str, int]:
    c: collections.Counter = collections.Counter()
    for ph in ("prereq", "query"):
        for f in glob.glob(str(run / "run" / "traj" / ph / "*.json")):
            try:
                d = json.loads(pathlib.Path(f).read_text())
            except Exception:
                continue
            for s in d.get("steps") or []:
                for k, v in (s or {}).items():
                    if v and k in EFFECT_KEYS:
                        c[k] += 1
    return dict(c)


def _parse(spec: str) -> dict[str, str]:
    out = {}
    for part in spec.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, required=True)
    ap.add_argument("--arm-tags", required=True, help="cell=tag,cell=tag,... (arm run tags)")
    ap.add_argument("--control-tags", default="", help="cell=tag,... (defaults to measured H0)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--label", default="stack")
    ap.add_argument("--require-firings", action="store_true",
                    help="block the row if a cell's arm shows no controller effects")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    arms, ctls = _parse(a.arm_tags), _parse(a.control_tags)
    rows, problems = {}, []

    for cell in ("kv", "vector", "rec_sum"):
        tag = arms.get(cell)
        if not tag:
            problems.append(f"{cell}: no arm tag given")
            continue
        run = a.results / f"{tag}_{a.split}"
        got = load_cell(run)
        if got is None:
            problems.append(f"{cell}: no results under {run}")
            continue
        correct, n = got
        if n != CANONICAL[cell]:
            problems.append(f"{cell}: n={n}, canonical is {CANONICAL[cell]} -- DIFFERENT POPULATION")
        eff = effects(run)
        if a.require_firings and not eff:
            problems.append(f"{cell}: arm shows NO controller effects -- its delta is unattributed")
        # control
        ctag = ctls.get(cell)
        if ctag:
            cgot = load_cell(a.results / f"{ctag}_{a.split}")
            if cgot is None:
                problems.append(f"{cell}: no CONTROL results for tag {ctag}")
                ccorrect = None
            else:
                ccorrect, cn = cgot
                if cn != n:
                    problems.append(f"{cell}: control n={cn} != arm n={n} -- not a paired comparison")
        else:
            ccorrect = H0[cell]
        rows[cell] = {"arm_tag": tag, "correct": correct, "n": n, "control": ccorrect, "effects": eff}

    print("=" * 92)
    print(f"RECOVERY TABLE -- {a.label}   (split={a.split})")
    print("=" * 92)
    print(f"{'cell':9s} {'arm':>9s} {'control':>9s} {'delta':>7s}   effects")
    for cell in ("kv", "vector", "rec_sum"):
        r = rows.get(cell)
        if r is None:
            print(f"{cell:9s} {'MISSING':>9s}")
            continue
        ctl = r["control"]
        dl = f"{r['correct'] - ctl:+d}" if ctl is not None else "?"
        eff = ", ".join(f"{k.replace('_gate','')}={v}" for k, v in sorted(r["effects"].items())) or "NONE"
        print(f"{cell:9s} {r['correct']:>4}/{r['n']:<4} {str(ctl):>9s} {dl:>7s}   {eff}")

    complete = len(rows) == 3 and not any("DIFFERENT POPULATION" in p or "unattributed" in p
                                          or "not a paired" in p for p in problems)
    out = {"label": a.label, "split": a.split, "rows": rows, "problems": problems,
           "complete": complete, "historical_ladder": LADDER,
           "historical_target_pct": HIST_TARGET, "historical_gain_pp": HIST_GAIN}

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  ! {p}")

    if not complete:
        print("\nNO WHOLE-CORPUS TOTAL REPORTED.")
        print("  A total needs all three cells, canonical denominators, paired controls and real")
        print("  firings. Adding per-cell deltas measured at different times is exactly how the")
        print("  34.32% projection came to be quoted as recovery -- this script will not do it.")
    else:
        arm_tot = sum(r["correct"] for r in rows.values())
        ctl_tot = sum(r["control"] for r in rows.values())
        arm_pct, ctl_pct = 100 * arm_tot / CORPUS_N, 100 * ctl_tot / CORPUS_N
        gain = arm_pct - 30.03
        print(f"\nWHOLE CORPUS (n={CORPUS_N}, EXECUTED as a stack):")
        print(f"  control  {ctl_tot}/{CORPUS_N} = {ctl_pct:.2f}%")
        print(f"  arm      {arm_tot}/{CORPUS_N} = {arm_pct:.2f}%")
        print(f"  vs measured H0 {H0_TOTAL}/{CORPUS_N} = 30.03%  ->  {gain:+.2f} pp")
        print(f"\n  historical target 52.48% ({HIST_GAIN:+.2f} pp)")
        print(f"  RECOVERED: {gain:.2f} of {HIST_GAIN:.2f} pp = {100 * gain / HIST_GAIN:.1f}%")
        out.update({"arm_total": arm_tot, "control_total": ctl_tot, "arm_pct": round(arm_pct, 2),
                    "gain_pp": round(gain, 2),
                    "pct_of_historical_gain": round(100 * gain / HIST_GAIN, 1)})

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {a.json}")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
