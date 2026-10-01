#!/usr/bin/env python3
"""ATTRIBUTION EXPANSION: propose structural signals for the residual no locus explains.

Runs only when `phase_switch.py` returns EXPAND_ATTRIBUTION. Operates on the UNEXPLAINED cases --
failing queries that no locus in the current vocabulary accounts for, because nothing errored.

A backward-trace CLASS is not a detector. `never_stored` is a post-hoc label on an episode; a
miner needs a condition that is **checkable at a decision point**. So each candidate here is a
step-local predicate, and each is scored on the only two numbers that decide whether it is a
signal at all:

    coverage   share of unexplained failures it fires on
    precision  fires-on-failure / (fires-on-failure + fires-on-success)

A condition that fires on successes as often as failures is not a signal however plausible it
sounds. That is not hypothetical: `single_read_then_answer` covered 81 failures and 78 successes
(precision 0.51) -- it reads exactly like premature answering and carries no information.

Accepted candidates are proposed only. Attaching a policy is the NEXT phase, after the expanded
vocabulary is frozen, so that ranking and acceptance still run against a fixed world.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import remine_incumbent as rm      # noqa: E402
import op_canonicalize_locus as oc  # noqa: E402

MIN_PRECISION = 0.75
MIN_COVERAGE = 0.10

_CALL = re.compile(r"([a-z_][a-z_0-9]*)\s*\(")


def _step_features(ep: Dict) -> Dict[str, bool]:
    """Step-local, decision-point-checkable conditions.

    Every one of these must be answerable from what is visible AT a step -- no outcome labels, no
    hindsight. That is the difference between a detector and a diagnosis.
    """
    steps = ep.get("steps") or []
    calls: List[str] = []
    reads = writes = empty_reads = 0
    seen = collections.Counter()
    repeated = 0
    read_before_answer = False
    for s in steps:
        dec = s.get("decoded") or []
        rs = s.get("tool_results") or []
        for j, c in enumerate(dec):
            m = _CALL.search(str(c))
            nm = m.group(1) if m else "?"
            calls.append(nm)
            seen[str(c)[:80]] += 1
            if seen[str(c)[:80]] > 1:
                repeated += 1
            r = str(rs[j]) if j < len(rs) else ""
            if any(v in nm for v in ("retrieve", "search", "list")):
                reads += 1
                read_before_answer = True
                if re.search(r'\[\s*\]|\{\s*\}|"keys"\s*:\s*\[\s*\]|"ranked_results"\s*:\s*\[\s*\]', r):
                    empty_reads += 1
            if any(v in nm for v in ("add", "append", "update")):
                writes += 1
    return {
        # THE VALIDATED CANDIDATE: the model answers without consulting memory at all.
        "no_tool_call_at_all": len(calls) == 0,
        "called_but_never_read": len(calls) > 0 and reads == 0,
        # A read happened and came back empty -- successful-looking, information-free.
        "every_read_empty": reads > 0 and empty_reads == reads,
        "some_read_empty": empty_reads > 0,
        "wrote_without_reading": writes > 0 and reads == 0,
        "repeated_identical_call": repeated > 0,
        "single_read_only": reads == 1,
        "answered_after_read": read_before_answer,
        "short_episode": len(steps) <= 2,
    }


def expand(incumbent: Path, artifact: Path) -> Dict:
    art = oc.LocusArtifact.load(artifact)
    valid = {r["id"]: bool(r.get("valid"))
             for r in json.load(open(incumbent / "eval_train_results.json"))["results"]
             if not r.get("is_prereq")}

    fail_feats, succ_feats = collections.Counter(), collections.Counter()
    n_unexplained = n_succ = 0
    per_case = {}

    for p in sorted((incumbent / "traj" / "query").glob("*.json")):
        try:
            ep = json.load(open(p))
        except Exception:
            continue
        qid = ep.get("case_id")
        ok = valid.get(qid)
        if ok is None:
            continue
        # EXPLAINED? any step of this query carries a locus in the current vocabulary.
        explained = False
        for s in ep.get("steps") or []:
            dec = s.get("decoded") or []
            res = s.get("tool_results") or []
            for j, c in enumerate(dec):
                r = res[j] if j < len(res) else ""
                a = {k: v for k, _q, v in rm._ARG.findall(str(c))}
                if art and art.locus_of(r, a):
                    explained = True
        feats = _step_features(ep)
        if ok:
            n_succ += 1
            for k, v in feats.items():
                if v:
                    succ_feats[k] += 1
        elif not explained:
            n_unexplained += 1
            per_case[qid] = [k for k, v in feats.items() if v]
            for k, v in feats.items():
                if v:
                    fail_feats[k] += 1

    cands = []
    for k in sorted(set(fail_feats) | set(succ_feats)):
        a, b = fail_feats[k], succ_feats[k]
        prec = a / max(a + b, 1)
        cov = a / max(n_unexplained, 1)
        cands.append({"signal": k, "in_unexplained_fail": a, "in_success": b,
                      "precision": round(prec, 3), "coverage": round(cov, 3),
                      "accepted": bool(prec >= MIN_PRECISION and cov >= MIN_COVERAGE)})
    cands.sort(key=lambda c: (-c["accepted"], -c["coverage"]))
    return {"n_unexplained": n_unexplained, "n_success": n_succ,
            "candidates": cands, "per_case": per_case}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", required=True, type=Path)
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    out = expand(a.incumbent, a.artifact)
    print("ATTRIBUTION EXPANSION over the UNEXPLAINED residual")
    print("  unexplained failing queries: %d" % out["n_unexplained"])
    print("  passing queries (control)  : %d\n" % out["n_success"])
    print("%-26s %10s %9s %10s %9s  %s"
          % ("candidate signal", "unexp_fail", "success", "precision", "coverage", "verdict"))
    for c in out["candidates"]:
        print("%-26s %10d %9d %10.2f %8.0f%%  %s"
              % (c["signal"], c["in_unexplained_fail"], c["in_success"],
                 c["precision"], 100 * c["coverage"],
                 "ACCEPT" if c["accepted"] else
                 ("fires on successes" if c["precision"] < MIN_PRECISION else "too rare")))
    acc = [c for c in out["candidates"] if c["accepted"]]
    print()
    if acc:
        print("PROPOSED for the expanded vocabulary (no policy attached yet):")
        for c in acc:
            print("   %-26s coverage %2.0f%%  precision %.2f"
                  % (c["signal"], 100 * c["coverage"], c["precision"]))
        print("\nNext: freeze the expanded vocabulary, then resume the policy loop against it.")
    else:
        print("No candidate clears precision >= %.2f and coverage >= %.0f%%."
              % (MIN_PRECISION, 100 * MIN_COVERAGE))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
