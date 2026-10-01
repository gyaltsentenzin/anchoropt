#!/usr/bin/env python3
"""Score an R3 round against the FROZEN criterion. Reports the eight required quantities in order.

    python scripts/score_r3_replication.py --results <dir> --tag-suffix s2
    python scripts/score_r3_replication.py --results <dir> --compare-to <seed42 summary.json>

The criterion lives in `anchoropt/learning/prestate_eligibility.py` and was committed BEFORE the
seed-2 run (91b3662, docs/SELFEVOLVE_R3_REPLICATION_PROTOCOL.md). This script only APPLIES it.

WHY THE PRIMARY SUBSET IS THE TARGET POPULATION AND NOT THE PRESTATE PREFIX
--------------------------------------------------------------------------
Common-prestate eligibility is the stated rule, and on THIS signal it is vacuous: the reprompt fires
on the model's first decision, so the compared prefix is empty wherever an arm fires (measured on
seed 42: common_fire_index == 0 in all 35 firing cases). It is still computed and reported, because
its emptiness is itself a fact about the round. But the causal comparison runs on
`on_target_shared` -- the control was a genuine zero-call state AND the arm fired there.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anchoropt.learning.prestate_eligibility import (  # noqa: E402
    FIRE_KEY, common_fire_index, eligible_cases, engagement_split, first_fire_index,
    qualifying_in_control,
)

ARMS = ("ctl", "a1", "a2")
LABEL = {"ctl": "control (native, gates off)",
         "a1": "eta_mu = verify_before_answering (146 ch)",
         "a2": "eta_mu = search_other_container (102 ch)"}


def load(root: pathlib.Path, arm: str, suffix: str):
    """(outcome_by_case, steps_by_case) from the run directory."""
    d = root / f"se3{arm}{suffix}_vector_train"
    res = d / "run" / "eval_train_results.json"
    if not res.exists():
        raise SystemExit(f"se3{arm}{suffix}: missing {res}")
    payload = json.load(open(res))
    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    # THE EFFECTIVE DECODE SEED, from the run's own provenance -- not from the runner's log echo.
    # `scripts/env.sh` exports SEED=42 and the runner echoes it, but that variable NEVER feeds
    # `--seed`; the flag is a literal in the runner. So a seed-2 run prints "SEED=42" in its log
    # while decoding with seed 2. Reading the echo would have declared the replication invalid;
    # reading provenance (args.seed, run_memory_eval.py) is what actually ran.
    prov = payload.get("provenance") or {} if isinstance(payload, dict) else {}
    seed = prov.get("seed")
    outcome, secondary = {}, {}
    for r in rows:
        if r.get("is_prereq"):
            continue
        cid = str(r["id"])
        outcome[cid] = bool(r.get("valid"))
        secondary[cid] = bool(r.get("valid")) and not str(r.get("error_type") or "").strip()
    steps = {}
    for fp in sorted((d / "run" / "traj" / "query").glob("*.json")):
        try:
            ep = json.load(open(fp))
        except Exception:
            continue
        steps[str(ep.get("case_id"))] = ep.get("steps") or []
    return outcome, secondary, steps, seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, required=True)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--json-out", type=pathlib.Path)
    ap.add_argument("--compare-to", type=pathlib.Path,
                    help="a prior run's json-out, to check whether the ORDERING replicates")
    a = ap.parse_args()

    O, S, T, seeds = {}, {}, {}, {}
    for arm in ARMS:
        O[arm], S[arm], T[arm], seeds[arm] = load(a.results, arm, a.tag_suffix)

    W = 100
    print("=" * W)
    print(f"R3 {'REPLICATION' if a.tag_suffix else 'ROUND'}  ·  vector/train  ·  frozen criterion "
          f"(docs/SELFEVOLVE_R3_REPLICATION_PROTOCOL.md)")
    print("=" * W)

    # ---------------------------------------------------------------- 0. seed identity
    print(f"\n[0] EFFECTIVE DECODE SEED (from each run's provenance, not the log echo)")
    for arm in ARMS:
        print(f"  se3{arm+a.tag_suffix:<8s} seed={seeds[arm]}")
    if len(set(seeds.values())) != 1:
        print("  FAILED: arms decoded with DIFFERENT seeds -- they are not a paired comparison.")
        return 3
    out_seed = next(iter(seeds.values()))

    # ---------------------------------------------------------------- 1. denominator integrity
    intended = sorted(O["ctl"])
    n = len(intended)
    print(f"\n[1] DENOMINATOR INTEGRITY  (precondition -- no deltas printed if it fails)")
    ok_all = True
    for arm in ARMS:
        missing = set(intended) - set(O[arm])
        extra = set(O[arm]) - set(intended)
        dis = {c for c in set(O[arm]) & set(S[arm]) if O[arm][c] != S[arm][c]}
        traj_missing = set(intended) - set(T[arm])
        ok = not (missing or extra or dis or traj_missing)
        ok_all &= ok
        print(f"  se3{arm+a.tag_suffix:<8s} n={len(O[arm]):3d}  missing={len(missing)}  "
              f"extra={len(extra)}  scorer-disagree={len(dis)}  traj-missing={len(traj_missing)}  "
              f"-> {'OK' if ok else 'FAILED'}")
    print(f"\n  DENOMINATOR INTEGRITY: {'OK' if ok_all else 'FAILED'}")
    if not ok_all:
        print("  Refusing to compute any delta.")
        return 2

    ctl_correct = sum(O["ctl"].values())
    out = {"n": n, "control_correct": ctl_correct, "seed_suffix": a.tag_suffix,
           "decode_seed": out_seed, "arms": {}}

    # ---------------------------------------------------------------- 2. raw
    print(f"\n[2] RAW ACCURACY / NET")
    print(f"  control: {ctl_correct}/{n} = {100*ctl_correct/n:.2f}%")
    for arm in ("a1", "a2"):
        c = sum(O[arm].values())
        g = [k for k in intended if O[arm][k] and not O["ctl"][k]]
        l = [k for k in intended if O["ctl"][k] and not O[arm][k]]
        f = sum(1 for k in intended if first_fire_index(T[arm].get(k, [])) is not None)
        print(f"  se3{arm}: {c}/{n} = {100*c/n:.2f}%  delta {100*(c-ctl_correct)/n:+.2f} pp   "
              f"+{len(g)}/-{len(l)}  net {len(g)-len(l):+d}   firings {f}")
        out["arms"][arm] = {"raw_correct": c, "raw_delta_pp": 100*(c-ctl_correct)/n,
                           "raw_gains": len(g), "raw_losses": len(l), "raw_net": len(g)-len(l),
                           "firings": f}

    # ---------------------------------------------------------------- 3. drift
    print(f"\n[3] PRE-INTERVENTION DRIFT")
    elig, drifted = eligible_cases({f"se3{x}": T[x] for x in ARMS}, control="se3ctl")
    cuts = {}
    for cse in intended:
        cuts.setdefault(common_fire_index({x: T[x].get(cse, []) for x in ARMS}), 0)
        cuts[common_fire_index({x: T[x].get(cse, []) for x in ARMS})] += 1
    print(f"  prestate-prefix drift : " +
          "  ".join(f"se3{x}={len(drifted['se3'+x])}" for x in ARMS))
    print(f"  common_fire_index dist: "
          f"{{{', '.join(f'{k}: {v}' for k, v in sorted(cuts.items(), key=lambda kv: (kv[0] is None, kv[0])))}}}")
    if cuts.get(0):
        print(f"  NOTE: the cut is step 0 on {cuts[0]} cases, so the compared prefix is EMPTY there --")
        print(f"        prestate eligibility passes VACUOUSLY on the treated cases (known, pre-recorded).")
    splits = {arm: engagement_split(T["ctl"], T[arm]) for arm in ("a1", "a2")}
    for arm in ("a1", "a2"):
        sp = splits[arm]
        print(f"  se3{arm}: opportunity_created={len(sp['opportunity_created']):3d}  "
              f"(arm fired where the control was NOT a zero-call state -> sampling, NOT attributable)")
        out["arms"][arm]["opportunity_created"] = len(sp["opportunity_created"])
        out["arms"][arm]["prestate_drift"] = len(drifted["se3" + arm])

    # ---------------------------------------------------------------- 4/5. subset + parity
    target = {c for c in intended if qualifying_in_control(T["ctl"].get(c, []))}
    shared = splits["a1"]["on_target_shared"] & splits["a2"]["on_target_shared"]
    print(f"\n[4] SUBSET SIZE")
    print(f"  control target population (zero-call states) : {len(target)}")
    print(f"  on_target_shared per arm                     : "
          f"a1={len(splits['a1']['on_target_shared'])}  a2={len(splits['a2']['on_target_shared'])}")
    print(f"  BOTH arms fired (the causal comparison set)   : {len(shared)}")
    print(f"  opportunity_missed (ctl qualified, no firing) : "
          f"a1={len(splits['a1']['opportunity_missed'])}  a2={len(splits['a2']['opportunity_missed'])}")
    print(f"  prestate-eligible (reported, vacuous here)   : {len(elig)}/{n}")
    out["target_n"] = len(target)
    out["shared_n"] = len(shared)

    print(f"\n[5] FIRING PARITY ON THE COMPARISON SET")
    fp = {arm: sum(1 for c in shared if first_fire_index(T[arm][c]) is not None)
          for arm in ("a1", "a2")}
    print(f"  firings within the shared set: a1={fp['a1']}  a2={fp['a2']}  "
          f"{'MATCHED' if fp['a1'] == fp['a2'] else 'MISMATCHED'}")
    print(f"  (by construction both fired on every shared case, so this pins the set, not the arms)")

    # ---------------------------------------------------------------- 6/7/8
    print(f"\n[6] ATTRIBUTABLE GAINS / LOSSES  (on_target_shared only)")
    ctl_shared = sum(1 for c in shared if O["ctl"][c])
    for arm in ("a1", "a2"):
        g = sorted(c for c in shared if O[arm][c] and not O["ctl"][c])
        l = sorted(c for c in shared if O["ctl"][c] and not O[arm][c])
        cor = sum(1 for c in shared if O[arm][c])
        print(f"  se3{arm}: {cor}/{len(shared)} vs ctl {ctl_shared}/{len(shared)}   "
              f"+{len(g)}/-{len(l)}  net {len(g)-len(l):+d}   "
              f"delta {100*(cor-ctl_shared)/max(len(shared),1):+.1f} pp")
        if l:
            print(f"      LOSSES: {l}")
        out["arms"][arm].update({"attr_gains": len(g), "attr_losses": len(l),
                                 "attr_net": len(g) - len(l), "shared_correct": cor,
                                 "shared_delta_pp": 100*(cor-ctl_shared)/max(len(shared), 1)})

    print(f"\n[7] ON-TARGET CONVERSION RATE")
    for arm in ("a1", "a2"):
        fired_t = [c for c in shared if first_fire_index(T[arm][c]) is not None]
        conv = [c for c in fired_t if O[arm][c] and not O["ctl"][c]]
        broke = [c for c in fired_t if O["ctl"][c] and not O[arm][c]]
        rate = 100 * len(conv) / max(len(fired_t), 1)
        print(f"  se3{arm}: fired {len(fired_t)}  converted {len(conv)}  broke {len(broke)}  "
              f"precision {rate:.1f}%")
        out["arms"][arm]["conversion_pct"] = rate
        out["arms"][arm]["converted"] = len(conv)

    print(f"\n[8] OVERHEAD  (mechanism check: a firing must produce a REAL retrieval)")
    for arm in ("a1", "a2"):
        fired_t = [c for c in shared if first_fire_index(T[arm][c]) is not None]
        reads = steps_extra = 0
        with_read = 0
        for c in fired_t:
            calls = [str(x) for s in T[arm][c] for x in (s.get("decoded") or [])]
            r = [x for x in calls if "retrieve" in x or "search" in x]
            reads += len(r)
            with_read += 1 if r else 0
            steps_extra += len(T[arm][c]) - len(T["ctl"][c])
        print(f"  se3{arm}: reads/case {reads/max(len(fired_t),1):.2f}   extra steps {steps_extra:+d}   "
              f"firings that produced >=1 retrieval: {with_read}/{len(fired_t)}")
        out["arms"][arm].update({"reads_per_case": reads/max(len(fired_t), 1),
                                 "extra_steps": steps_extra,
                                 "firings_with_retrieval": with_read,
                                 "fired_on_target": len(fired_t)})

    # ---------------------------------------------------------------- verdict
    print(f"\n{'=' * W}\nVERDICT against the frozen promotion rule")
    a1, a2 = out["arms"]["a1"], out["arms"]["a2"]
    c1 = a2["attr_net"] > 0
    c2 = a2["attr_losses"] == 0
    c3 = a2["firings_with_retrieval"] == a2["fired_on_target"] and a2["fired_on_target"] > 0
    c4 = (a2["attr_net"], a2["conversion_pct"]) >= (a1["attr_net"], a1["conversion_pct"])
    for label, ok in (("se3a2 positive on the on-target subset", c1),
                      ("no working cases broken", c2),
                      ("mechanism intact (every firing -> >=1 real retrieval)", c3),
                      ("se3a2 >= se3a1 on the frozen criterion", c4)):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    ordering = "se3a2 > se3a1 > control" if c4 and a1["attr_net"] >= 0 else "NOT REPLICATED"
    print(f"\n  ordering on this seed: {ordering}")
    out["promote"] = bool(c1 and c2 and c3 and c4)
    out["ordering_replicated"] = bool(c4 and a1["attr_net"] >= 0)
    print(f"  PROMOTE se3a2: {'YES' if out['promote'] else 'NO -- STOP and report instability'}")

    # ---------------------------------------------------------------- independence guard
    # A REPLICATION MUST BE AN INDEPENDENT EXECUTION. On this substrate decoding is greedy
    # (temperature=0 + fixed seed, memory_evaluator.py "W1 determinism"), so changing --seed changes
    # NOTHING: the seed-2 round reproduced seed 42 bit for bit, 0 differing step fields across all
    # 89 episodes in all three arms. The scorer printed PROMOTE: YES on what was one observation
    # counted twice. It now refuses that on its own rather than relying on someone noticing.
    if a.compare_to and a.compare_to.exists():
        prior_path = a.compare_to.parent
        identical = []
        for arm in ARMS:
            here = a.results / f"se3{arm}{a.tag_suffix}_vector_train" / "run" / "traj" / "query"
            for other_suffix in ("", "s2"):
                there = prior_path / f"se3{arm}{other_suffix}_vector_train" / "run" / "traj" / "query"
                if there == here or not there.exists() or not here.exists():
                    continue
                same = total = 0
                for fp in sorted(here.glob("*.json")):
                    q = there / fp.name
                    if not q.exists():
                        continue
                    total += 1
                    try:
                        if json.load(open(fp)).get("steps") == json.load(open(q)).get("steps"):
                            same += 1
                    except Exception:
                        pass
                if total and same == total:
                    identical.append(f"se3{arm}: all {total} episodes byte-identical to {there.parts[-4]}")
        if identical:
            print(f"\n{'!' * W}")
            print("INDEPENDENCE FAILED -- this is a RE-EXECUTION, not a replication:")
            for line in identical:
                print(f"  {line}")
            print("Every step field matches, so this round carries NO new evidence and the promotion")
            print("criteria cannot be evaluated on it. Decoding here is greedy (temperature=0), so the")
            print("decode seed is INERT; independent variation must come from a different cell, a")
            print("different eta paraphrase, or resampling -- not from --seed.")
            print(f"{'!' * W}")
            out["independent"] = False
            out["promote"] = False

    if a.compare_to and a.compare_to.exists():
        prior = json.load(open(a.compare_to))
        print(f"\n  --- vs {a.compare_to.name} ---")
        for arm in ("a1", "a2"):
            p = prior.get("arms", {}).get(arm, {})
            print(f"    se3{arm}: attr_net {p.get('attr_net', p.get('stable_net', 'n/a'))} -> "
                  f"{out['arms'][arm]['attr_net']}   conversion "
                  f"{p.get('conversion_pct', 'n/a')} -> {out['arms'][arm]['conversion_pct']:.1f}%")

    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=2)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
