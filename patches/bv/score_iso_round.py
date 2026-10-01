#!/usr/bin/env python3
"""Score the isolated rec_sum round, keeping FOUR levels of claim strictly apart.

The distinction this script exists to enforce, because collapsing any two of them is how this project
has produced confidently wrong results before:

  1 SIGNAL FIRING        the predicate returned True on some states. Says NOTHING about accuracy.
  2 ACTUAL INTERVENTION  the mechanism ran -- calls were withheld. An arm with firings but zero
                         interventions IS the control, and its paired zero is not a null about the
                         idea, it is a null about nothing.
  3 MEASURED IMPROVEMENT net > 0 on the paired cases, with the denominator verified. Criterion 1 of
                         four. This is `TRAIN_IMPROVED_PENDING_VALIDATION`, never acceptance.
  4 FULL ACCEPTANCE      all four criteria of docs/ACCEPTANCE_RULE.md: train net > 0, independent dev
                         non-regression, causal attribution from the controller's own telemetry, and
                         safety. This script NEVER awards it -- dev is a separate split and is not
                         read here, by design.
"""

from __future__ import annotations

import glob
import json
import math
import pathlib
import sys

W = "/path/to/isolated-workspace/results"


def load(d: str):
    p = pathlib.Path(d) / "run" / "eval_train_results.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    rows = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
    return {str(r["id"]): bool(r.get("valid")) for r in rows if not r.get("is_prereq")}


def gates(d: str):
    """Firings and interventions from the TRAJECTORY SIDECAR -- never a registry-derived dict."""
    fired = set()
    withheld = 0
    for sub in ("query", "prereq"):
        for f in glob.glob(f"{d}/run/traj/{sub}/*.json"):
            try:
                ep = json.load(open(f))
            except Exception:
                continue
            cid = ep.get("case_id")
            for s in ep.get("steps") or []:
                for k, v in s.items():
                    if k.endswith("_gate") and v:
                        fired.add(cid)
                # INTERVENTIONS: read the EXACT key the commitment-gate path writes.
                #
                # `upstream_withheld_gate` is an INT COUNT of calls withheld at this step
                # (memory_evaluator.py ~4828), and `upstream_withheld_calls` is the list of call
                # strings. Guessing the name is how a previous summary reported 0 firings against a
                # real count of 46 -- a zero-intervention arm is indistinguishable from an unparsed
                # telemetry channel, so the name is read off the writer, not invented.
                n = s.get("upstream_withheld_gate")
                if isinstance(n, bool):
                    n = int(n)
                if isinstance(n, int) and n > 0:
                    withheld += n
                elif isinstance(s.get("upstream_withheld_calls"), list):
                    withheld += len(s["upstream_withheld_calls"])
    return fired, withheld


def sign_p(g: int, l: int) -> float:
    n = g + l
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, k) for k in range(0, min(g, l) + 1)) / 2 ** n * 2)


def main() -> int:
    ctl = load(f"{W}/iso_ctl_train")
    if ctl is None:
        print("control not finished")
        return 1
    cf, cw = gates(f"{W}/iso_ctl_train")
    print(f"CONTROL  {sum(ctl.values())}/{len(ctl)}   firings={len(cf)} withheld_calls={cw}")
    if cw:
        print("  ! the CONTROL withheld calls -- it is not a clean baseline")
    print()
    hdr = ("%-28s %7s %7s %8s %6s %6s %5s %9s %11s %s"
           % ("arm", "score", "delta", "net(g/l)", "fired", "withh", "p", "level", "denominator", "verdict"))
    print(hdr)
    print("-" * len(hdr))
    out = []
    for d in sorted(glob.glob(f"{W}/iso_*_train")):
        tag = pathlib.Path(d).name.replace("_train", "")
        if tag == "iso_ctl":
            continue
        arm = load(d)
        if arm is None:
            print("%-28s (unfinished)" % tag[:28])
            continue
        common = sorted(set(ctl) & set(arm))
        gains = [c for c in common if not ctl[c] and arm[c]]
        losses = [c for c in common if ctl[c] and not arm[c]]
        net = len(gains) - len(losses)
        af, aw = gates(d)
        denom_ok = set(ctl) == set(arm)
        delta = 100.0 * (sum(arm[c] for c in common) - sum(ctl[c] for c in common)) / max(len(common), 1)

        # THE FOUR LEVELS, in order. Each one is a precondition for the next.
        if aw == 0:
            level, verdict = "1-FIRING" if af else "0-NOTHING", \
                "NOT MEASURABLE: zero interventions -- this arm ran as the CONTROL"
        elif not denom_ok:
            level, verdict = "2-INTERVENED", "INVALID: denominator mismatch, not a paired comparison"
        elif net > 0:
            level, verdict = "3-TRAIN_IMPROVED_PENDING_VALIDATION", \
                "criterion 1 of 4 only -- dev/attribution/safety NOT applied here"
        else:
            level, verdict = "2-INTERVENED", "NO_BENEFIT: measured, intervened, did not improve"

        print("%-28s %3d/%-3d %+7.2f %4d(%d/%d) %6d %6d %5.3f %-9s %11s %s"
              % (tag[:28], sum(arm[c] for c in common), len(common), delta, net, len(gains),
                 len(losses), len(af), aw, sign_p(len(gains), len(losses)),
                 level.split("-")[0], "OK" if denom_ok else "MISMATCH", verdict[:46]))
        out.append({"arm": tag, "n": len(common), "score": sum(arm[c] for c in common),
                    "control_score": sum(ctl[c] for c in common), "delta_pp": round(delta, 4),
                    "gains": gains, "losses": losses, "net": net,
                    "cases_fired": len(af), "interventions_executed": aw,
                    "denominator_ok": denom_ok, "level": level, "verdict": verdict,
                    "sign_test_p": round(sign_p(len(gains), len(losses)), 4),
                    "incumbent_token": "/path/to/isolated-workspace/results/iso_ctl_train/run"})
    pathlib.Path(f"{W}/round_results.json").write_text(
        json.dumps({"incumbent_token": "/path/to/isolated-workspace/results/iso_ctl_train/run",
                    "results": out}, indent=2) + "\n")
    print()
    print("NOTE  Level 3 is TRAIN ONLY. No arm here can reach level 4: dev is a separate split and is")
    print("      deliberately not read by this script, so acceptance cannot be awarded from it.")
    print(f"wrote {W}/round_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
