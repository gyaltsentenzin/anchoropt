#!/usr/bin/env python3
"""The PHASE-SWITCH rule: continue policy search, stop, or expand attribution?

Coordinated descent has two phases and the loop must decide between them QUANTITATIVELY, from the
current policy world, rather than by a human noticing that the ranking looks thin:

    POLICY PHASE       attribution vocabulary FIXED; mine -> rank -> intervene -> freeze -> re-mine
    ATTRIBUTION PHASE  policy exhausted; expand the vocabulary, freeze it, resume the policy phase

After every incumbent update the residual is partitioned:

    explained    a failing query linked to a locus in the current detector vocabulary
    unexplained  a failing query no locus in that vocabulary accounts for

and then, using the ALREADY-FROZEN S1/S2/S5 thresholds:

    CONTINUE            >=1 explained locus survives S1 and S2  -> keep optimising policy
    EXPAND ATTRIBUTION   0 survive AND unexplained dominates    -> the vocabulary is the limit
    STOP                 0 survive AND unexplained is small     -> genuinely done

THE TWO TERMINAL CASES ARE NOT THE SAME, and conflating them is the failure this module prevents.
"Nothing survives the gates" alone does not mean the work is done -- it means nothing survives *in
the current representation*. Only when the unexplained mass is ALSO small is the residual actually
addressed. Otherwise the bottleneck has moved from policy to signal, and forcing another anchor
would be optimising a vocabulary that cannot see 87% of what is left.

Thresholds are inherited, not re-invented: S1 = 10% of residual (derived from corpus power -- 10%
of ~200 cases is ~20, and the sign test needs >=9 net), S2 = 50% settled, S5 = unexplained mass
exceeds the largest explained candidate. See docs/STOPPING_CRITERIA_FROZEN.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import remine_incumbent as rm      # noqa: E402
import op_canonicalize_locus as oc  # noqa: E402

# Inherited from STOPPING_CRITERIA_FROZEN.md -- do not tune here.
S1_COVERAGE_FLOOR = 0.10
S2_SATURATION_CEIL = 0.50

CONTINUE, EXPAND, STOP = "CONTINUE_POLICY", "EXPAND_ATTRIBUTION", "STOP"


def partition(incumbent: Path, artifact: Path) -> Tuple[Dict, List[Dict]]:
    """Split the residual into explained / unexplained, and rank the explained loci."""
    rows, meta = rm.mine(incumbent, artifact, "PHASE-CHECK", use_llm=False, overwrite=True)
    art = oc.LocusArtifact.load(artifact)
    n_fail = meta["n_failing_queries"]

    valid = {r["id"]: bool(r.get("valid"))
             for r in json.load(open(incumbent / "eval_train_results.json"))["results"]
             if not r.get("is_prereq")}

    # A failing query is EXPLAINED if any locus in the vocabulary links to it. Reuse the ranker's
    # own linkage so "explained" means exactly what "linked_loss" means -- no second definition.
    fails = {q for q, ok in valid.items() if not ok}
    explained_qids = set()
    per_locus_q = collections.defaultdict(set)
    for p in sorted((incumbent / "traj" / "query").glob("*.json")):
        try:
            ep = json.load(open(p))
        except Exception:
            continue
        qid = ep.get("case_id")
        if qid not in fails:
            continue
        for s in ep.get("steps") or []:
            dec = s.get("decoded") or []
            res = s.get("tool_results") or []
            for j, c in enumerate(dec):
                r = res[j] if j < len(res) else ""
                a = {k: v for k, _q, v in rm._ARG.findall(str(c))}
                k = art.locus_of(r, a) if art else None
                if k:
                    explained_qids.add(qid)
                    per_locus_q["/".join(k)].add(qid)

    # Chain-level linkage credits loci for queries in the same chain; count that too, because the
    # ranker uses it and the phase rule must be consistent with the ranker.
    linked_total = set()
    for r in rows:
        linked_total |= per_locus_q.get("/".join(r["locus"]), set())

    summary = {
        "residual": n_fail,
        "explained_direct": len(explained_qids),
        "explained_by_ranker_linkage": sum(r["linked_loss"] for r in rows),
        "unexplained": n_fail - len(explained_qids),
        "corrections": meta.get("total_corrections", 0),
        "episodes_with_corrections": meta.get("episodes_with_corrections", 0),
    }
    return summary, rows


def decide(summary: Dict, rows: List[Dict]) -> Tuple[str, List[str], List[str]]:
    """(phase, surviving loci, the reasons -- every threshold shown, not just the verdict)."""
    n = max(summary["residual"], 1)
    reasons, surviving = [], []
    for r in rows:
        lk = "/".join(r["locus"])
        cov = r["linked_loss"] / n
        sat = r["settled_by_incumbent"] / max(r["raw_events"], 1)
        s1 = cov >= S1_COVERAGE_FLOOR
        s2 = sat < S2_SATURATION_CEIL
        if s1 and s2:
            surviving.append(lk)
        reasons.append("%-44s cov %5.1f%% %-8s sat %3.0f%% %-8s" % (
            lk[:44], 100 * cov, "S1-ok" if s1 else "S1-STOP",
            100 * sat, "S2-ok" if s2 else "S2-STOP"))

    if surviving:
        return CONTINUE, surviving, reasons

    # Nothing survives. S5 decides WHICH terminal case this is.
    explained_mass = max((r["linked_loss"] for r in rows), default=0)
    unexplained = summary["unexplained"]
    dominates = unexplained > explained_mass
    reasons.append("")
    reasons.append("no explained locus survives the policy gates")
    reasons.append("  largest explained candidate : %d linked (%.0f%% of residual)"
                   % (explained_mass, 100 * explained_mass / n))
    reasons.append("  unexplained residual        : %d (%.0f%% of residual)"
                   % (unexplained, 100 * unexplained / n))
    reasons.append("  S5 -- unexplained dominates : %s" % dominates)
    return (EXPAND if dominates else STOP), [], reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", required=True, type=Path)
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    summary, rows = partition(a.incumbent, a.artifact)
    phase, surviving, reasons = decide(summary, rows)

    print("PHASE CHECK -- incumbent %s\n" % a.incumbent)
    print("residual failing queries      : %d" % summary["residual"])
    print("  explained by the vocabulary : %d (%.0f%%)"
          % (summary["explained_direct"],
             100 * summary["explained_direct"] / max(summary["residual"], 1)))
    print("  UNEXPLAINED                 : %d (%.0f%%)"
          % (summary["unexplained"],
             100 * summary["unexplained"] / max(summary["residual"], 1)))
    print("corrections by incumbent      : %d across %d episodes\n"
          % (summary["corrections"], summary["episodes_with_corrections"]))
    for line in reasons:
        print("   " + line if line else "")
    print()
    print("PHASE = %s" % phase)
    if phase == CONTINUE:
        print("  surviving loci: %s" % ", ".join(surviving))
        print("  -> continue the policy loop on the highest-ranked survivor")
    elif phase == EXPAND:
        print("  -> the SIGNAL REPRESENTATION is exhausted, not the headroom.")
        print("     Run attribution expansion over the unexplained cases; do NOT force an anchor.")
    else:
        print("  -> genuine stop: little actionable explained mass AND little unexplained mass.")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"phase": phase, "summary": summary, "surviving": surviving,
                   "loci": [{"locus": r["locus"], "linked": r["linked_loss"],
                             "settled": r["settled_by_incumbent"],
                             "raw_events": r["raw_events"]} for r in rows]},
                  open(a.json, "w"), indent=2)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
