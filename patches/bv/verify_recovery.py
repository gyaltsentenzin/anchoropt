#!/usr/bin/env python3
"""Verify a RECOVERY intervention on what it PRESERVED, not on whether it fired.

The distinction the suppression results forced. For a suppressor the question is "was the call
withheld". For a recovery it is:

  1 did the repair DISPATCH a replacement?
  2 did that replacement SUCCEED -- i.e. did the store accept it?
  3 was the INFORMATION preserved -- do the original payload's facts survive in the substituted one?
  4 did the store end up with MORE content than the control, not less?

A recovery that fires, dispatches, and is refused is indistinguishable from a no-op on a results table,
and a recovery that "succeeds" while dropping the facts is worse than doing nothing.
"""

from __future__ import annotations

import glob
import json
import re
import sys

ERR = re.compile(r"error|cannot|unable|must be|full|exceed|invalid|unique", re.I)
NUM = re.compile(r"\d[\d,.]*")


def facts(text: str) -> set:
    """The comparable factual tokens: numbers and capitalised terms. Deliberately coarse."""
    t = str(text or "")
    return set(NUM.findall(t)) | set(re.findall(r"\b[A-Z][a-z]{2,}\b", t))


def main() -> int:
    tag = sys.argv[1]
    ctl = sys.argv[2] if len(sys.argv) > 2 else None
    base = f"/path/to/isolated-workspace/results/{tag}_train/run/traj"

    fired = dispatched = succeeded = refused = 0
    preserved = lost = 0
    shed_total = 0
    examples = []
    for sub in ("prereq", "query"):
        for f in sorted(glob.glob(f"{base}/{sub}/*.json")):
            try:
                ep = json.load(open(f))
            except Exception:
                continue
            for s in ep.get("steps") or []:
                if not s.get("capacity_repair_gate"):
                    continue
                fired += 1
                rep = str(s.get("capacity_repair_replaced_call") or "")
                sub_call = str(s.get("capacity_repair_substituted_call") or "")
                shed = int(s.get("capacity_repair_shed_chars") or 0)
                shed_total += shed
                if sub_call:
                    dispatched += 1
                # the repair appends its result to tool_results
                res = [str(r) for r in (s.get("tool_results") or [])]
                last = res[-1] if res else ""
                if last and not ERR.search(last):
                    succeeded += 1
                elif last:
                    refused += 1
                if rep and sub_call:
                    a, b = facts(rep), facts(sub_call)
                    keep = len(a & b) / max(len(a), 1)
                    (preserved if keep >= 0.9 else lost).__iadd__ if False else None
                    if keep >= 0.9:
                        preserved += 1
                    else:
                        lost += 1
                    if len(examples) < 3:
                        examples.append((ep.get("case_id"), shed, round(keep, 3)))

    print("RECOVERY VERIFICATION -- %s" % tag)
    print("  1 fired                       : %d" % fired)
    print("  2 dispatched a replacement    : %d" % dispatched)
    print("  3 replacement ACCEPTED by store: %d   (refused %d)" % (succeeded, refused))
    print("  4 facts preserved (>=90%%)      : %d   (lost %d)" % (preserved, lost))
    print("     characters shed total      : %d" % shed_total)
    for cid, shed, keep in examples:
        print("     e.g. %-40s shed=%-6d fact_retention=%.3f" % (str(cid)[:40], shed, keep))
    if fired and not dispatched:
        print("  ! FIRED BUT NEVER DISPATCHED -- the mechanism did not run")
    if dispatched and not succeeded:
        print("  ! DISPATCHED BUT NEVER ACCEPTED -- the store refused every replacement")

    if ctl:
        def store_size(t):
            tot = 0
            for sub in ("prereq",):
                for f in glob.glob(f"/path/to/isolated-workspace/results/{t}_train/run/traj/{sub}/*.json"):
                    try:
                        ep = json.load(open(f))
                    except Exception:
                        continue
                    for s in ep.get("steps") or []:
                        rs = [str(r) for r in (s.get("tool_results") or [])]
                        for i, c in enumerate(s.get("decoded") or []):
                            r = rs[i] if i < len(rs) else ""
                            if r and not ERR.search(r):
                                tot += len(str(c))
            return tot
        a, c = store_size(tag), store_size(ctl)
        print("  5 committed write volume      : arm %d chars vs control %d (%+.1f%%)"
              % (a, c, 100 * (a - c) / max(c, 1)))
        print("     -> a recovery should store MORE than the control, not less")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
