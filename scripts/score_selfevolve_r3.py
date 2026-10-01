#!/usr/bin/env python3
"""Score Self-Evolve R3: the first eta_mu-ONLY counterfactual. Denominator integrity FIRST.

    python scripts/score_selfevolve_r3.py --results <dir with se3*_vector_train/>

WHAT MAKES R3 DIFFERENT FROM R1/R2
----------------------------------
R1 varied the controller; R2 swept theta_phi with eta frozen at the proposer's guess. R3 holds
(l, phi, mu) and theta_phi FIXED and varies ONLY eta_mu -- the instruction string. Both arms run the
same executor at the same boundary on the same signal, so a difference between them is attributable
to the action PARAMETER and to nothing else. That is the comparison the architecture exists to make.

    l   = post_generation_pre_exec        (recovered by the live proposer, anchors scrubbed)
    phi = no_tool_call_at_all             deterministic -> NO theta grid, 1 theta per arm
    mu  = REPROMPT                        the only materializable action here (executor-backed)
    eta = se3a1 verify_before_answering  |  se3a2 search_other_container

se3a3 (state_absence_if_unfound, retry_budget=0) is NOT an arm: the executor fixes retry_budget=1
and coercing it would run different semantics under the proposal's name. Rejected, never run.

ORDER IS THE POINT
------------------
Nothing about accuracy is printed until every arm covers exactly the intended cases and two
independent scorer paths agree. A corrupted denominator has already once produced wreckage that read
as a clean null result.

ENGAGEMENT IS NOT OPTIONAL
--------------------------
A delta with zero firings proves NON-attribution, never a subtle effect. Firings come from the
per-case `gates_fired` dict and the `zero_call_reprompt_gate` step flag -- the channels this
executor actually writes. `exec.jsonl` is a TOOL-CALL log carrying no gate keys; scanning it for
`*_gate` once reported 0 firings beside a real count of 46.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anchoropt.learning.round_runner import ArmOutcome, paired_evaluation  # noqa: E402

ARMS = {
    "se3ctl": ("control", "native incumbent, every gate off"),
    "se3a1": ("arm 1", "eta_mu = verify_before_answering (146 chars)"),
    "se3a2": ("arm 2", "eta_mu = search_other_container (102 chars)"),
}
CONTROL = "se3ctl"

# THE TELEMETRY SOURCE FOR THIS ARM IS THE TRAJECTORY SIDECAR, NOT `gates_fired`.
#
# Verified empirically before the run, because getting this wrong is the recurring bug:
#   * memory_evaluator sets step_record["zero_call_reprompt_gate"] = True at the firing site.
#   * `gates_fired` in eval_*_results.json is built by run_memory_train.py from
#     `telemetry_flags()`, which is REGISTRY-derived (30 keys). A4 is a REMEDY FLAG, not a
#     registry gate, so `zero_call_reprompt_gate` IS NOT IN IT -- the nearest key is the
#     unrelated `turn_start_action_gate`.
#   * the executor does NOT set `injection_key`, so `signals_fired` misses it too.
#   * traj_sidecar keeps any truthy key ending in `_gate`, so the per-step record survives there.
#
# Measured on results/a1barm_kv_test (A4 armed): sidecar = 8 firings on 8 cases, while
# gates_fired does not carry the key at all. Reading `gates_fired` would have reported ZERO
# firings for a working arm and called a real intervention non-attributable.
GATE_FLAGS = ("zero_call_reprompt_gate", "low_similarity_reprompt_gate")
SIDECAR_KEY = "zero_call_reprompt_gate"


def load_arm(root: pathlib.Path, arm: str) -> tuple[dict, dict, dict]:
    """(primary, secondary, telemetry) for one arm. Two independent scorer paths."""
    d = root / f"{arm}_vector_train"
    res = d / "run" / "eval_train_results.json"
    if not res.exists():
        alt = sorted(d.glob("run/**/eval*results*.json")) or sorted(d.glob("run/**/eval*.json"))
        if not alt:
            raise SystemExit(f"{arm}: no eval json under {d}")
        res = alt[0]
    payload = json.load(open(res))
    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload

    primary, secondary = {}, {}
    for r in rows:
        if r.get("is_prereq"):
            continue
        cid = str(r["id"])
        primary[cid] = bool(r.get("valid"))
        secondary[cid] = bool(r.get("valid")) and not str(r.get("error_type") or "").strip()

    telemetry = {"signal_firings": 0, "interventions_executed": 0, "gate_detail": {},
                 "cases_fired": 0, "exec_log_lines": 0, "steps": 0, "tool_calls": 0}
    exec_log = d / "exec.jsonl"
    if exec_log.exists():
        telemetry["exec_log_lines"] = sum(1 for _ in open(exec_log))

    # PRIMARY firing source: the per-episode trajectory sidecar (see SIDECAR_KEY note above).
    sc_fires = sc_cases = 0
    traj = d / "run" / "traj" / "query"
    sidecar_seen = traj.exists()
    for fp in sorted(traj.glob("*.json")) if sidecar_seen else []:
        try:
            ep = json.load(open(fp))
        except Exception:
            continue
        n = sum(1 for st in (ep.get("steps") or [])
                if isinstance(st, dict) and st.get(SIDECAR_KEY))
        if n:
            sc_cases += 1
            sc_fires += n
    telemetry["sidecar_present"] = sidecar_seen
    telemetry["sidecar_firings"] = sc_fires
    telemetry["sidecar_cases"] = sc_cases

    for r in rows:
        fired_here = False
        for k, v in (r.get("gates_fired") or {}).items():
            if v:
                telemetry["gate_detail"][k] = telemetry["gate_detail"].get(k, 0) + 1
                telemetry["signal_firings"] += 1
                fired_here = True
        for flag in GATE_FLAGS:
            if r.get(flag):
                telemetry["gate_detail"][flag] = telemetry["gate_detail"].get(flag, 0) + 1
                telemetry["signal_firings"] += 1
                fired_here = True
        # OVERHEAD, where the records carry it: a reprompt costs an extra generation per firing.
        for key in ("n_steps", "steps", "step_count"):
            if isinstance(r.get(key), int):
                telemetry["steps"] += r[key]
                break
        for key in ("n_tool_calls", "tool_calls_total"):
            if isinstance(r.get(key), int):
                telemetry["tool_calls"] += r[key]
                break
        if fired_here:
            telemetry["cases_fired"] += 1
    # The sidecar is AUTHORITATIVE for this executor; gates_fired structurally cannot see it.
    telemetry["gates_fired_firings"] = telemetry["signal_firings"]
    if sidecar_seen:
        telemetry["signal_firings"] = sc_fires
        telemetry["cases_fired"] = sc_cases
    # For a reprompt the firing IS the execution: the text is injected at the firing site.
    telemetry["interventions_executed"] = telemetry["signal_firings"]
    return primary, secondary, telemetry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, required=True)
    ap.add_argument("--json-out", type=pathlib.Path)
    a = ap.parse_args()

    loaded = {}
    for arm in ARMS:
        try:
            loaded[arm] = load_arm(a.results, arm)
        except SystemExit as exc:
            print(f"MISSING ARM {arm}: {exc}")
    if CONTROL not in loaded:
        print("no control arm -- nothing is interpretable")
        return 1

    intended = sorted(loaded[CONTROL][0])
    n = len(intended)
    print("=" * 100)
    print(f"SELF-EVOLVE R3  ·  eta_mu-only counterfactual  ·  vector/train  ·  intended cases: {n}")
    print("  l=post_generation_pre_exec  phi=no_tool_call_at_all  mu=REPROMPT  (theta: deterministic)")
    print("=" * 100)

    print("\n[1] DENOMINATOR INTEGRITY -- a precondition, not a diagnostic")
    outcomes = {arm: ArmOutcome(arm=arm, by_primary=p, by_secondary=s, telemetry=t)
                for arm, (p, s, t) in loaded.items()}
    all_ok = True
    for arm, o in outcomes.items():
        missing = set(intended) - set(o.by_primary)
        extra = set(o.by_primary) - set(intended)
        dis = o.disagreements()
        ok = not (missing or extra or dis)
        all_ok &= ok
        print(f"  {arm:8s} n={len(o.by_primary):4d}  missing={len(missing):3d}  extra={len(extra):3d}  "
              f"scorer-disagreements={len(dis):3d}  -> {'OK' if ok else 'FAILED'}")
        for c in list(dis)[:5]:
            print(f"      disagree {c}: primary={o.by_primary[c]} secondary={o.by_secondary[c]}")
    print(f"\n  DENOMINATOR INTEGRITY: {'OK' if all_ok else 'FAILED'}")
    if not all_ok:
        print("\n  Refusing to compute or print any accuracy delta.")
        return 2

    print("\n[2] PAIRED SCORING vs control")
    ctl = outcomes[CONTROL]
    ctl_correct = sum(ctl.by_primary.values())
    ctl_tel = ctl.telemetry
    print(f"  control: {ctl_correct}/{n} = {100*ctl_correct/n:.2f}%   "
          f"firings={ctl_tel['signal_firings']} (MUST be 0)")
    if ctl_tel["signal_firings"]:
        print("  WARNING: the control fired. It is not a native baseline.")

    summary = {"n": n, "control_correct": ctl_correct, "arms": {}}
    for arm in ("se3a1", "se3a2"):
        if arm not in outcomes:
            continue
        label, note = ARMS[arm]
        o = outcomes[arm]
        ev = paired_evaluation(ctl, o, intended)
        correct = sum(o.by_primary.values())
        delta = 100 * (correct - ctl_correct) / n
        tel = o.telemetry
        conv = (ev.net / tel["signal_firings"]) if tel["signal_firings"] else 0.0
        print(f"\n  {arm}  ({label} -- {note})")
        print(f"    accuracy      : {correct}/{n} = {100*correct/n:.2f}%   delta {delta:+.2f} pp")
        print(f"    gains/losses  : +{len(ev.gains)} / -{len(ev.losses)}   net {ev.net:+d}")
        print(f"    firings       : {tel['signal_firings']}  (on {tel['cases_fired']} of {n} cases)")
        print(f"    executed      : {tel['interventions_executed']}")
        print(f"    net/firing    : {conv:+.3f}")
        print(f"    source        : sidecar={tel['sidecar_firings']} on {tel['sidecar_cases']} cases"
              f"   (gates_fired channel: {tel['gates_fired_firings']}"
              f"{'  <- structurally blind to this remedy flag, expected 0' if not tel['gates_fired_firings'] else ''})")
        if not tel["sidecar_present"]:
            print("    WARNING: no trajectory sidecar -- firing count is NOT trustworthy for this arm")
        if tel["gate_detail"]:
            print(f"    gate detail   : {dict(sorted(tel['gate_detail'].items()))}")
        if tel["steps"] or ctl_tel["steps"]:
            print(f"    overhead      : steps {ctl_tel['steps']} -> {tel['steps']} "
                  f"({tel['steps']-ctl_tel['steps']:+d}); "
                  f"tool_calls {ctl_tel['tool_calls']} -> {tel['tool_calls']} "
                  f"({tel['tool_calls']-ctl_tel['tool_calls']:+d})")
        if not tel["signal_firings"]:
            print("    NON-ATTRIBUTABLE: zero firings. Any delta here is noise, not this eta_mu.")
        print(f"    gains         : {sorted(ev.gains)[:8]}")
        print(f"    losses        : {sorted(ev.losses)[:8]}")
        summary["arms"][arm] = {
            "correct": correct, "delta_pp": delta, "gains": len(ev.gains),
            "losses": len(ev.losses), "net": ev.net, "firings": tel["signal_firings"],
            "cases_fired": tel["cases_fired"], "gate_detail": tel["gate_detail"],
            "net_per_firing": conv}

    print("\n[3] ETA_MU COMPARISON -- the point of the round")
    armed = [x for x in ("se3a1", "se3a2") if x in outcomes]
    if len(armed) == 2:
        f1, f2 = (outcomes[x].telemetry["signal_firings"] for x in armed)
        print(f"  Both arms share (l, phi, mu) and the executor, so they are comparable ONLY if they")
        print(f"  engaged the same states: firings se3a1={f1} se3a2={f2} "
              f"{'(MATCHED)' if f1 == f2 else '(MISMATCHED -- eta changed WHEN it fires, not just what it says)'}")
        best = max(armed, key=lambda x: summary["arms"][x]["net"])
        nets = {x: summary["arms"][x]["net"] for x in armed}
        print(f"  net: {nets}")
        if all(v <= 0 for v in nets.values()):
            print("  BOTH FAIL (no positive net). Per the round's rule: STOP and diagnose.")
            print("  Do NOT fall through to another R1 locus.")
            summary["verdict"] = "both_fail_stop_and_diagnose"
        elif list(nets.values()).count(max(nets.values())) > 1:
            print("  TIE on net. No promotion without a tiebreak on a stated criterion.")
            summary["verdict"] = "tie"
        elif f1 != f2:
            # Unequal firings mean the arms did not engage the same states, so raw net is not a
            # like-for-like comparison and MUST NOT decide. Section [4] ranks on the drift-free
            # subset; this is deliberately not a promotion verdict.
            print(f"  raw-net leader is {best}, but the firing counts differ -- raw net is NOT")
            print("  like-for-like. DEFER to the drift-free subset in section [4] for the ranking.")
            summary["verdict"] = "deferred_to_stable_subset"
        else:
            print(f"  WINNER on net: {best}  -> eligible for promotion + re-mine")
            summary["verdict"] = f"winner:{best}"
    # ------------------------------------------------------------------ drift-free subset
    # THE FIRING-PARITY MISMATCH IS REAL AND IT INVERTS THE RANKING. Step-0 generation is not
    # reproducible across arms on this substrate: the gate runs AFTER step-0 generation, so a
    # step-0 decode that differs from the control cannot have been caused by the arm. Such cases
    # move into and out of the qualifying population by sampling alone, and in R3 that drift
    # (11 of 89) was LARGER than the effect being measured. So restrict to cases where every arm
    # agrees on the step-0 state, where a delta IS the intervention.
    print("\n[4] DRIFT-FREE SUBSET -- where a delta is the intervention, not sampling")
    s0 = {}
    for arm in loaded:
        d = a.results / f"{arm}_vector_train" / "run" / "traj" / "query"
        per = {}
        for fp in sorted(d.glob("*.json")):
            try:
                ep = json.load(open(fp))
            except Exception:
                continue
            st = ep.get("steps") or []
            first = next((x for x in st if isinstance(x, dict) and x.get("step") == 0), None)
            per[str(ep.get("case_id"))] = len(first.get("decoded") or []) if first else None
        s0[arm] = per
    if all(s0.get(arm) for arm in loaded):
        common = set.intersection(*(set(s0[arm]) for arm in loaded))
        stable = sorted(k for k in common
                        if len({s0[arm][k] for arm in loaded}) == 1)
        drift = sorted(common - set(stable))
        print(f"  step-0 identical across all arms: {len(stable)}/{len(common)}  "
              f"(drifted: {len(drift)})")
        for arm in loaded:
            n_d = sum(1 for k in common if s0[arm][k] != s0[CONTROL][k])
            print(f"    {arm:8s} step-0 differs from control on {n_d} cases")
        if stable:
            cz = sum(1 for k in stable if ctl.by_primary.get(k))
            target = [k for k in stable if s0[CONTROL][k] == 0]
            ct = sum(1 for k in target if ctl.by_primary.get(k))
            print(f"  control on stable: {cz}/{len(stable)}   "
                  f"control on TARGET (control-qualifying): {ct}/{len(target)}")
            for arm in ("se3a1", "se3a2"):
                if arm not in outcomes:
                    continue
                o = outcomes[arm]
                g = [k for k in stable if o.by_primary.get(k) and not ctl.by_primary.get(k)]
                l = [k for k in stable if ctl.by_primary.get(k) and not o.by_primary.get(k)]
                at = sum(1 for k in target if o.by_primary.get(k))
                td = 100 * (at - ct) / len(target) if target else 0.0
                print(f"    {arm}: net {len(g)-len(l):+d} (+{len(g)}/-{len(l)}) on {len(stable)} stable   "
                      f"on-target {at}/{len(target)} vs {ct}/{len(target)} = {td:+.1f} pp")
                summary["arms"][arm].update({
                    "stable_n": len(stable), "stable_net": len(g) - len(l),
                    "stable_gains": len(g), "stable_losses": len(l),
                    "target_n": len(target), "target_correct": at,
                    "target_control": ct, "target_delta_pp": td})
            summary["stable_n"] = len(stable)
            summary["drift_n"] = len(drift)
            ranked = {arm: summary["arms"][arm].get("stable_net", 0)
                      for arm in ("se3a1", "se3a2") if arm in outcomes}
            if ranked:
                top = max(ranked, key=lambda k: (ranked[k],
                                                 summary["arms"][k].get("target_delta_pp", 0)))
                if all(v <= 0 for v in ranked.values()):
                    print("\n  BOTH FAIL on the drift-free subset: STOP and diagnose.")
                    summary["stable_verdict"] = "both_fail_stop_and_diagnose"
                else:
                    print(f"\n  DRIFT-FREE WINNER: {top}  (stable net {ranked[top]:+d}, "
                          f"on-target {summary['arms'][top].get('target_delta_pp', 0):+.1f} pp)")
                    summary["stable_verdict"] = f"winner:{top}"
            print("\n  NOTE: rank on this subset, not on raw net. Raw net can credit an arm for a"
                  "\n  case where it never fired -- in R3 that happened twice and swapped the winner.")
    else:
        print("  no trajectories found -- cannot separate drift from effect")

    if a.json_out:
        json.dump(summary, open(a.json_out, "w"), indent=2)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
