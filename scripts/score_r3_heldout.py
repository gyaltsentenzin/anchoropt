#!/usr/bin/env python3
"""Held-out validation for the recovered A4 controller. FROZEN criterion, test split, per cell + pooled.

    python scripts/score_r3_heldout.py --results <dir> --cells vector,kv,rec_sum

Applies the same criterion as scripts/score_r3_replication.py (anchoropt/learning/
prestate_eligibility.py, frozen in 91b3662). Nothing here re-derives eligibility or promotion.

THE CORRECTED INTERPRETATION IS BUILT IN
----------------------------------------
* NO "sampling drift" language anywhere. Decoding is greedy (temperature=0 + fixed seed), so the
  substrate is deterministic and there is no pre-intervention sampling noise to subtract.
* GATE BOOKKEEPING ROWS ARE NOT MODEL DECISIONS. When the executor fires it appends its own row
  (status=None, decoded=[], <fire_key>=True); step/first-decision comparisons must exclude those, or
  the arm's bookkeeping gets compared against the control's first real decision. `real_steps()` does.
* `on_target_shared` remains the causal comparison population: the CONTROL was a genuine zero-call
  state AND the arm fired there. `opportunity_created` is the injection's DOWNSTREAM effect (a later
  turn reaching a commitment the control never reached), reported separately -- never as noise.

POOLING. Cells are indivisible units and are reported per cell first. The pooled line sums them,
which is legitimate for a paired count but is NOT a significance claim: at these n the round cannot
separate the arms statistically, and the promotion criterion is qualitative.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anchoropt.learning.prestate_eligibility import (  # noqa: E402
    FIRE_KEY, engagement_split, first_fire_index, qualifying_in_control,
)

ARMS = ("ctl", "a1", "a2")
LBL = {"ctl": "control", "a1": "se3a1 verify_before_answering",
       "a2": "se3a2 search_other_container"}
READ = ("retrieve", "search")


def real_steps(steps):
    """Steps that are MODEL DECISIONS. Excludes pure gate-bookkeeping rows.

    A row with no status and no decoded calls is the executor recording that it fired; treating it as
    a decision is the error that produced the retracted step-0 'drift' finding.
    """
    return [s for s in steps
            if not (s.get("status") is None and not (s.get("decoded") or []))]


def load(root: pathlib.Path, arm: str, cell: str):
    d = root / f"se3{arm}ho_{cell}_test"
    res = d / "run" / "eval_test_results.json"
    if not res.exists():
        cand = sorted(d.glob("run/eval_*results*.json"))
        if not cand:
            raise SystemExit(f"se3{arm}ho_{cell}: missing {res}")
        res = cand[0]
    payload = json.load(open(res))
    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    correct = {str(r["id"]): bool(r.get("valid")) for r in rows if not r.get("is_prereq")}
    secondary = {str(r["id"]): bool(r.get("valid")) and not str(r.get("error_type") or "").strip()
                 for r in rows if not r.get("is_prereq")}
    steps = {}
    for fp in sorted((d / "run" / "traj" / "query").glob("*.json")):
        try:
            ep = json.load(open(fp))
        except Exception:
            continue
        steps[str(ep.get("case_id"))] = ep.get("steps") or []
    seed = (payload.get("provenance") or {}).get("seed") if isinstance(payload, dict) else None
    return correct, secondary, steps, seed


def cell_metrics(C, CS, AC, AS, cases):
    """All reported quantities for one arm in one cell."""
    split = engagement_split(CS, AS)
    target = {c for c in cases if qualifying_in_control(CS.get(c, []))}
    fired = {c for c in cases if first_fire_index(AS.get(c, [])) is not None}
    shared = split["on_target_shared"] & set(cases)
    g = sorted(c for c in cases if AC[c] and not C[c])
    l = sorted(c for c in cases if C[c] and not AC[c])
    conv = sorted(c for c in shared if AC[c] and not C[c])
    broke = sorted(c for c in shared if C[c] and not AC[c])
    # cost, over real decisions only
    steps_a = sum(len(real_steps(AS.get(c, []))) for c in cases)
    steps_c = sum(len(real_steps(CS.get(c, []))) for c in cases)
    calls_a = sum(len(s.get("decoded") or []) for c in cases for s in AS.get(c, []))
    calls_c = sum(len(s.get("decoded") or []) for c in cases for s in CS.get(c, []))
    reads_a = sum(1 for c in cases for s in AS.get(c, [])
                  for x in (s.get("decoded") or []) if any(m in str(x) for m in READ))
    reads_c = sum(1 for c in cases for s in CS.get(c, [])
                  for x in (s.get("decoded") or []) if any(m in str(x) for m in READ))
    interventions = sum(sum(1 for s in AS.get(c, []) if s.get(FIRE_KEY)) for c in cases)
    with_read = sum(1 for c in fired
                    if any(any(m in str(x) for m in READ)
                           for s in AS.get(c, []) for x in (s.get("decoded") or [])))
    return {
        "n": len(cases), "correct": sum(AC[c] for c in cases),
        "gains": g, "losses": l,
        "n_target": len(target), "n_fired": len(fired), "n_shared": len(shared),
        "conversions": conv, "broke": broke,
        "opportunity_created": len(split["opportunity_created"] & set(cases)),
        "opportunity_missed": len(split["opportunity_missed"] & set(cases)),
        "interventions": interventions,
        "steps": steps_a, "steps_ctl": steps_c,
        "llm_calls": steps_a, "llm_calls_ctl": steps_c,
        "tool_calls": calls_a, "tool_calls_ctl": calls_c,
        "reads": reads_a, "reads_ctl": reads_c,
        "firings_with_read": with_read,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, required=True)
    ap.add_argument("--cells", default="vector,kv,rec_sum")
    ap.add_argument("--json-out", type=pathlib.Path)
    a = ap.parse_args()
    cells = [c.strip() for c in a.cells.split(",") if c.strip()]

    W = 104
    print("=" * W)
    print("R3 HELD-OUT VALIDATION  ·  test split (no cell shared with train)  ·  frozen criterion")
    print("  l=post_generation_pre_exec  phi=no_tool_call_at_all  mu=REPROMPT  retry=1  eta frozen")
    print("=" * W)

    data, seeds = {}, set()
    for cell in cells:
        for arm in ARMS:
            data[(cell, arm)] = load(a.results, arm, cell)
            seeds.add(data[(cell, arm)][3])
    out = {"cells": {}, "pooled": {}}
    print(f"\n[0] decode seed (identical across arms required): {sorted(x for x in seeds if x)}")
    if len({x for x in seeds if x is not None}) > 1:
        print("  FAILED: arms decoded with different seeds")
        return 3

    # ------------------------------------------------------------------ prereq contamination
    # PREREQ EPISODES WRITE THE STORE THE QUERIES READ. If the gate fires during a prereq, the arm
    # altered the store its own queries are then scored against, and the cell is not a paired
    # comparison at all -- the arms answered different questions. Measured on this round: kv fired
    # 22 (se3a1) and 12 (se3a2) times in prereqs while the control fired 0, and the kv prereq call
    # sequences differ across all three arms; vector and rec_sum are byte-identical. So kv is
    # CONFOUNDED and must be excluded from the causal comparison, not merely noted.
    print(f"\n[0b] PREREQ CONTAMINATION  (a firing here rewrites the store the queries read)")
    contaminated = []
    for cell in cells:
        sigs, fires = {}, {}
        for arm in ARMS:
            d = a.results / f"se3{arm}ho_{cell}_test" / "run" / "traj" / "prereq"
            calls, f = [], 0
            for fp in sorted(d.glob("*.json")):
                try:
                    ep = json.load(open(fp))
                except Exception:
                    continue
                calls += [str(x) for st in (ep.get("steps") or []) for x in (st.get("decoded") or [])]
                f += sum(1 for st in (ep.get("steps") or []) if st.get(FIRE_KEY))
            sigs[arm] = "|".join(calls)
            fires[arm] = f
        same = len(set(sigs.values())) == 1
        bad = any(fires[x] for x in ("a1", "a2")) or not same
        print(f"  {cell:8s} prereq gate firings ctl/a1/a2 = "
              f"{fires['ctl']}/{fires['a1']}/{fires['a2']}   prereq trajectories "
              f"{'identical' if same else 'DIFFER'}   -> {'CONFOUNDED' if bad else 'clean'}")
        if bad:
            contaminated.append(cell)
    if contaminated:
        print(f"\n  CONFOUNDED CELLS: {contaminated}")
        print("  The arm modified the store its own queries are scored against, so gains there are")
        print("  not attributable to the controller. Re-run with --cells excluding them for the")
        print("  causal comparison; the numbers below still include every cell you passed.")
    out["contaminated_cells"] = contaminated

    print(f"\n[1] DENOMINATOR INTEGRITY  (precondition)")
    ok_all, per_cell_cases = True, {}
    for cell in cells:
        C, S, CS, _ = data[(cell, "ctl")]
        intended = sorted(C)
        per_cell_cases[cell] = intended
        for arm in ARMS:
            AC, AS_, AST, _ = data[(cell, arm)]
            missing = set(intended) - set(AC)
            extra = set(AC) - set(intended)
            dis = {c for c in set(AC) & set(AS_) if AC[c] != AS_[c]}
            tmiss = set(intended) - set(AST)
            ok = not (missing or extra or dis or tmiss)
            ok_all &= ok
            print(f"  {cell:8s} se3{arm:<4s} n={len(AC):3d} missing={len(missing)} extra={len(extra)} "
                  f"disagree={len(dis)} traj-missing={len(tmiss)} -> {'OK' if ok else 'FAILED'}")
    print(f"\n  DENOMINATOR INTEGRITY: {'OK' if ok_all else 'FAILED'}")
    if not ok_all:
        print("  Refusing to compute any delta.")
        return 2

    agg = {arm: {} for arm in ("a1", "a2")}
    for cell in cells:
        cases = per_cell_cases[cell]
        C, _, CS, _ = data[(cell, "ctl")]
        cc = sum(C[c] for c in cases)
        ctl_target = sum(1 for c in cases if qualifying_in_control(CS.get(c, [])))
        ctl_fired = sum(1 for c in cases if first_fire_index(CS.get(c, [])) is not None)
        print(f"\n{'-' * W}\nCELL {cell}  (n={len(cases)})")
        print(f"  control                accuracy {cc}/{len(cases)} = {100*cc/len(cases):.2f}%"
              f"   target states {ctl_target}   firings {ctl_fired} (must be 0)")
        out["cells"][cell] = {"n": len(cases), "control_correct": cc, "target_states": ctl_target}
        for arm in ("a1", "a2"):
            AC, _, AS, _ = data[(cell, arm)]
            m = cell_metrics(C, CS, AC, AS, cases)
            d = 100 * (m["correct"] - cc) / len(cases)
            print(f"\n  se3{arm} ({LBL[arm]})")
            print(f"    accuracy            {m['correct']}/{m['n']} = {100*m['correct']/m['n']:.2f}%"
                  f"   delta {d:+.2f} pp")
            print(f"    gains / losses      +{len(m['gains'])} / -{len(m['losses'])}   "
                  f"net {len(m['gains'])-len(m['losses']):+d}")
            print(f"    firing rate         {m['n_fired']}/{m['n']} = "
                  f"{100*m['n_fired']/m['n']:.1f}%   interventions {m['interventions']}")
            print(f"    target coverage     {m['n_shared']}/{m['n_target']} target states acted on"
                  + (f" = {100*m['n_shared']/m['n_target']:.1f}%" if m["n_target"] else " (no targets)"))
            print(f"    on-target conv.     {len(m['conversions'])} converted / {m['n_shared']} "
                  f"on-target firings"
                  + (f" = {100*len(m['conversions'])/m['n_shared']:.1f}%" if m["n_shared"] else "")
                  + f"   broke {len(m['broke'])}")
            print(f"    off-target firings  {m['opportunity_created']}"
                  f"   (missed opportunities {m['opportunity_missed']})")
            print(f"    steps/LLM calls     {m['steps']} vs ctl {m['steps_ctl']} "
                  f"({m['steps']-m['steps_ctl']:+d})   [real decisions; gate rows excluded]")
            print(f"    tool calls          {m['tool_calls']} vs ctl {m['tool_calls_ctl']} "
                  f"({m['tool_calls']-m['tool_calls_ctl']:+d})")
            print(f"    retrievals          {m['reads']} vs ctl {m['reads_ctl']} "
                  f"({m['reads']-m['reads_ctl']:+d})")
            ncv = len(m["conversions"])
            print(f"    cost per corrected  "
                  + (f"{(m['steps']-m['steps_ctl'])/ncv:+.2f} extra steps per fix" if ncv
                     else "no conversions -- undefined"))
            print(f"    MECHANISM           {m['firings_with_read']}/{m['n_fired']} firings produced "
                  f">=1 real retrieval" + ("  OK" if m["n_fired"] and
                                           m["firings_with_read"] == m["n_fired"] else
                                           ("  <-- INCOMPLETE" if m["n_fired"] else "  (no firings)")))
            out["cells"][cell][arm] = {k: (v if not isinstance(v, list) else len(v))
                                       for k, v in m.items()}
            for k in ("n", "correct", "n_target", "n_fired", "n_shared", "interventions",
                      "steps", "steps_ctl", "tool_calls", "tool_calls_ctl", "reads", "reads_ctl",
                      "opportunity_created", "opportunity_missed", "firings_with_read"):
                agg[arm][k] = agg[arm].get(k, 0) + m[k]
            for k in ("gains", "losses", "conversions", "broke"):
                agg[arm][k] = agg[arm].get(k, 0) + len(m[k])
        agg.setdefault("ctl", {})
        agg["ctl"]["correct"] = agg["ctl"].get("correct", 0) + cc
        agg["ctl"]["n"] = agg["ctl"].get("n", 0) + len(cases)
        agg["ctl"]["n_target"] = agg["ctl"].get("n_target", 0) + ctl_target

    print(f"\n{'=' * W}\nPOOLED ACROSS CELLS  (a paired count, NOT a significance claim)")
    cn, cc = agg["ctl"]["n"], agg["ctl"]["correct"]
    print(f"  control  {cc}/{cn} = {100*cc/cn:.2f}%   target states {agg['ctl']['n_target']}")
    for arm in ("a1", "a2"):
        A = agg[arm]
        d = 100 * (A["correct"] - cc) / cn
        print(f"  se3{arm}    {A['correct']}/{A['n']} = {100*A['correct']/A['n']:.2f}%  "
              f"delta {d:+.2f} pp   +{A['gains']}/-{A['losses']}  net {A['gains']-A['losses']:+d}   "
              f"fired {A['n_fired']}  on-target {A['n_shared']}  conv {A['conversions']}  "
              f"broke {A['broke']}")
        out["pooled"][arm] = dict(A)
    out["pooled"]["ctl"] = dict(agg["ctl"])

    print(f"\n{'=' * W}\nFROZEN PROMOTION CRITERION (evaluated on the pooled held-out cells)")
    A1, A2 = agg["a1"], agg["a2"]
    c1 = (A2["correct"] - cc) > 0 or (A2["gains"] - A2["losses"]) > 0
    c2 = A2["broke"] == 0
    c3 = A2["n_fired"] > 0 and A2["firings_with_read"] == A2["n_fired"]
    c4 = (A2["conversions"] - A2["broke"]) >= (A1["conversions"] - A1["broke"])
    for lbl, ok in (("se3a2 positive on the independent cells", c1),
                    ("no material regressions (0 on-target breakage)", c2),
                    ("mechanism intact (every firing -> >=1 real retrieval)", c3),
                    ("se3a2 >= se3a1 on the frozen criterion", c4)):
        print(f"  [{'PASS' if ok else 'FAIL'}] {lbl}")
    promote = bool(c1 and c2 and c3 and c4)
    out["promote"] = promote
    print(f"\n  PROMOTE se3a2: {'YES' if promote else 'NO'}")
    if A2["n_fired"] == 0:
        print("  NOTE: zero firings on held-out means the signal does not occur in these cells --")
        print("        that is a statement about trigger availability, NOT about the controller.")
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=2, default=str)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
