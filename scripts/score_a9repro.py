#!/usr/bin/env python3
"""Score the A9 reproduction round: does the UNIFIED core reproduce a validated+promoted controller?

    python scripts/score_a9repro.py --root results/a9repro

Reports, per arm, against the shared control:
  * the CLEAN comparison (65 queries) -- the causal claim
  * the ALL-89 figure, labelled approximate, for comparability with the historical +17.98pp
  * firings from the *_gate trajectory sidecar, never from a registry-derived dict
  * whether every gain sits on a fired case (the arithmetic that has exposed three wiring bugs)
  * prereq contamination per arm, so the exclusion is re-derived rather than trusted

THE EXCLUSION IS PRE-REGISTERED in rounds/A9REPRO/PREREGISTERED_EXCLUSION.md, committed before any
arm had an accuracy number. This script RE-DERIVES it from the dependency graph rather than reading a
stored list, so a drifted list cannot silently change the comparison.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
W = 100


def load(run: pathlib.Path):
    res = next(iter(sorted(run.glob("eval_*results*.json"))), None)
    if res is None:
        return None, {}, {}
    payload = json.load(open(res))
    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    correct = {str(r["id"]): bool(r.get("valid")) for r in rows if not r.get("is_prereq")}
    steps = {}
    for kind in ("query", "prereq"):
        d = run / "traj" / kind
        if not d.exists():
            continue
        steps.setdefault(kind, {})
        for fp in sorted(d.glob("*.json")):
            try:
                ep = json.load(open(fp))
            except Exception:
                continue
            steps[kind][str(ep.get("case_id"))] = ep.get("steps") or []
    return correct, steps.get("query", {}), steps.get("prereq", {})


def fired_cases(steps_by_case) -> set[str]:
    """Firings from the TRAJECTORY SIDECAR. `gates_fired` is registry-derived and blind to these."""
    out = set()
    for cid, steps in steps_by_case.items():
        for s in steps:
            if any(k == "controller_fired_gate" and v for k, v in s.items()):
                out.add(cid)
                break
    return out


def contaminated(shard: pathlib.Path, fired_prereqs: set[str]) -> set[str]:
    """Scored queries transitively depending on a prereq where the controller fired."""
    cases = json.load(open(shard))
    byid = {str(c["id"]): c for c in cases}

    def deps(cid, seen=None):
        seen = seen if seen is not None else set()
        for d in (byid.get(cid, {}).get("depends_on") or []):
            d = str(d)
            if d in seen:
                continue
            seen.add(d)
            deps(d, seen)
        return seen

    return {cid for cid in byid if "prereq" not in cid and (deps(cid) & fired_prereqs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=REPO / "results/a9repro")
    ap.add_argument("--control", default="a9rctl")
    ap.add_argument("--shard", type=pathlib.Path,
                    default=REPO / "data/shard_vector_train_cases.json")
    a = ap.parse_args()

    cc, cq, cp = load(a.root / a.control)
    if cc is None:
        raise SystemExit(f"control {a.control} has no results yet")

    print("=" * W)
    print("A9 REPRODUCTION THROUGH THE UNIFIED CORE  (core tag structured-search-core-v1)")
    print("  controller SHAPE rediscovered by optimize_residual; theta from observed quantiles")
    print("=" * W)

    arms = sorted(d.name for d in a.root.iterdir()
                  if d.is_dir() and d.name != a.control and (d / "traj").exists())
    rows = []
    for name in arms:
        ac, aq, ap_ = load(a.root / name)
        if ac is None:
            print(f"\n{name}: no results yet -- skipped")
            continue
        # Re-derive the exclusion from THIS arm's own prereq firings.
        bad_prereq = fired_cases(ap_)
        excl = contaminated(a.shard, bad_prereq) if bad_prereq else set()
        common = sorted(set(cc) & set(ac))
        clean = [c for c in common if c not in excl]
        fired = fired_cases(aq)

        def stats(cases):
            g = [c for c in cases if not cc[c] and ac[c]]
            l = [c for c in cases if cc[c] and not ac[c]]
            b, m = sum(cc[c] for c in cases), sum(ac[c] for c in cases)
            d = 100.0 * (m - b) / max(1, len(cases))
            return b, m, d, g, l

        b89, m89, d89, g89, l89 = stats(common)
        b65, m65, d65, g65, l65 = stats(clean)
        on_fired = [c for c in g65 if c in fired]
        rows.append((name, len(clean), b65, m65, d65, len(g65), len(l65),
                     len(fired & set(clean)), len(on_fired), d89, b89, m89, len(bad_prereq),
                     len(excl)))

        print(f"\n---- {name} " + "-" * (W - 6 - len(name)))
        print(f"  prereq firings: {len(bad_prereq)}  -> excludes {len(excl)} scored queries")
        print(f"  DENOMINATOR INTEGRITY: {'OK' if set(cc) == set(ac) else 'MISMATCH'}")
        print(f"  CLEAN (n={len(clean)}, the causal comparison):")
        print(f"      control {b65}/{len(clean)}   arm {m65}/{len(clean)}   "
              f"{d65:+.2f}pp   +{len(g65)}/-{len(l65)}")
        print(f"      firings {len(fired & set(clean))}; gains on a fired case "
              f"{len(on_fired)}/{len(g65)}")
        print(f"  ALL 89 (approximate -- prereq-contaminated, for comparability only):")
        print(f"      control {b89}/{len(common)}   arm {m89}/{len(common)}   "
              f"{d89:+.2f}pp   +{len(g89)}/-{len(l89)}")

    print("\n" + "=" * W)
    print(f"{'arm':10s} {'n':4s} {'ctl':5s} {'arm':5s} {'CLEAN dpp':10s} {'+/-':8s} "
          f"{'fired':6s} {'on-fired':9s} {'all89 dpp':10s}")
    print("-" * W)
    for (n, cl, b, m, d, g, l, f, of, d89, b89, m89, npq, nex) in rows:
        print(f"{n:10s} {cl:4d} {b:5d} {m:5d} {d:+10.2f} {f'+{g}/-{l}':8s} "
              f"{f:6d} {f'{of}/{g}':9s} {d89:+10.2f}")
    print("=" * W)
    print("Historical reference: +17.98pp on all 89 (pre-refactor core, prereq firings never audited).")
    print("A delta on 65 is NOT numerically comparable to that; the reproduction claim is about the")
    print("KIND of controller and the direction/rough magnitude of its effect.")
    out = a.root / "score.json"
    json.dump([{"arm": r[0], "n_clean": r[1], "control": r[2], "arm_correct": r[3],
                "delta_pp_clean": r[4], "gains": r[5], "losses": r[6], "firings": r[7],
                "gains_on_fired": r[8], "delta_pp_all89": r[9], "prereq_firings": r[12],
                "excluded_queries": r[13]} for r in rows], open(out, "w"), indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
