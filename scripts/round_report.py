#!/usr/bin/env python3
"""Print the full picture for a completed round: outcome, cost, locality, optimization, residuals.

    python scripts/round_report.py --results /tmp/r3_real --arms se3ctl,se3a1,se3a2
    python scripts/round_report.py --results <dir> --tag-suffix s2 --funnel <build.json> \
        --residuals-before 4,3,1 --json-out round.json

READ-ONLY. It loads run artifacts and prints; it changes no policy, promotes nothing, and imports no
decision code beyond the frozen eligibility module the locality metrics are defined on.

WHY THIS EXISTS: the eventual Self-Harness comparison cannot be a single accuracy number. Two systems
that reach the same accuracy differ in how much they perturb, what inference costs, and how much
search they needed to get there -- and none of that is visible in an accuracy table.

HONESTY RULES BUILT IN
----------------------
* an unmeasured quantity prints "not instrumented", never 0;
* changes on episodes the controller never fired on are printed SEPARATELY from its gains, because
  they cannot be its effect;
* the optimization funnel prints the stages it was actually given -- a stage with no data prints "-",
  so a shrink factor is never computed from an assumed denominator.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anchoropt.telemetry import (  # noqa: E402
    FUNNEL_STAGES, FunnelReport, arm_cost, cost_deltas, locality_report, residual_shape,
)
from anchoropt.telemetry.round_record import ResidualEntry  # noqa: E402

FIRE = "zero_call_reprompt_gate"


def load_arm(root: pathlib.Path, tag: str):
    d = root / f"{tag}_vector_train"
    res = d / "run" / "eval_train_results.json"
    if not res.exists():
        raise SystemExit(f"{tag}: missing {res}")
    payload = json.load(open(res))
    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    correct = {str(r["id"]): bool(r.get("valid")) for r in rows if not r.get("is_prereq")}
    steps = {}
    for fp in sorted((d / "run" / "traj" / "query").glob("*.json")):
        try:
            ep = json.load(open(fp))
        except Exception:
            continue
        steps[str(ep.get("case_id"))] = ep.get("steps") or []
    seed = (payload.get("provenance") or {}).get("seed") if isinstance(payload, dict) else None
    return correct, steps, seed


def fmt(v, unit="", nd=2):
    if v is None:
        return "not instrumented"
    if isinstance(v, float):
        return f"{v:+.{nd}f}{unit}" if unit == " pp" else f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, required=True)
    ap.add_argument("--arms", default="", help="comma list; default se3ctl,se3a1,se3a2 (+suffix)")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--funnel", type=pathlib.Path,
                    help="json with stage counts and/or rejection counts")
    ap.add_argument("--residuals-before", default="", help="support list, e.g. 4,3,1")
    ap.add_argument("--residuals-after", default="")
    ap.add_argument("--eval-episodes", type=int, help="episodes spent evaluating candidates")
    ap.add_argument("--gpu-seconds", type=float)
    ap.add_argument("--json-out", type=pathlib.Path)
    a = ap.parse_args()

    tags = [t.strip() for t in a.arms.split(",") if t.strip()] or \
        [f"se3{x}{a.tag_suffix}" for x in ("ctl", "a1", "a2")]
    ctl_tag, arm_tags = tags[0], tags[1:]
    C, CS, seed = load_arm(a.results, ctl_tag)
    arms = {t: load_arm(a.results, t) for t in arm_tags}

    W = 96
    print("=" * W)
    print(f"ROUND REPORT  ·  control={ctl_tag}  ·  arms={', '.join(arm_tags)}  ·  decode seed={seed}")
    print("=" * W)
    out = {"control": ctl_tag, "seed": seed, "arms": {}}

    n = len(C)
    ctl_correct = sum(C.values())
    print(f"\nOUTCOME")
    print(f"  control              {ctl_correct}/{n} = {100*ctl_correct/n:.2f}%")
    for t in arm_tags:
        AC, _, s2 = arms[t]
        common = sorted(set(C) & set(AC))
        cor = sum(AC[c] for c in common)
        g = [c for c in common if AC[c] and not C[c]]
        l = [c for c in common if C[c] and not AC[c]]
        print(f"  {t:<20s} {cor}/{len(common)} = {100*cor/len(common):.2f}%   "
              f"delta {100*(cor-ctl_correct)/len(common):+.2f} pp   "
              f"+{len(g)}/-{len(l)}  net {len(g)-len(l):+d}"
              + ("" if s2 == seed else f"   [SEED MISMATCH: {s2}]"))
        out["arms"][t] = {"correct": cor, "n": len(common), "gains": len(g), "losses": len(l),
                          "net": len(g) - len(l)}

    print(f"\nINFERENCE COST   (per-episode means; tokens/latency are not instrumented here)")
    ctl_cost = arm_cost(ctl_tag, CS, correct_by_case=C, fire_key=FIRE)
    print(f"  {'arm':<20s} {'steps':>7s} {'llm':>6s} {'tools':>7s} {'reads':>7s} "
          f"{'interv':>7s} {'tokens':>18s}")
    rows = [(ctl_tag, ctl_cost)] + [(t, arm_cost(t, arms[t][1], correct_by_case=arms[t][0],
                                                 fire_key=FIRE)) for t in arm_tags]
    for t, ac in rows:
        m = ac.as_dict()["mean_per_episode"]
        print(f"  {t:<20s} {m['steps']:>7.2f} {m['llm_calls']:>6.2f} {m['tool_calls']:>7.2f} "
              f"{m['retrieval_calls']:>7.2f} "
              f"{ac.as_dict()['totals']['interventions']:>7d} "
              f"{'not instrumented':>18s}")
    for t, ac in rows[1:]:
        cd = cost_deltas(ac, ctl_cost)
        of, ou = cd["overhead_on_fired"], cd["overhead_on_untouched"]
        cc = cd["cost_per_corrected_failure"]
        print(f"\n  {t} vs control:")
        print(f"    overhead on FIRED episodes    n={of['n']:<4d} steps {fmt(of['steps'])}  "
              f"reads {fmt(of['retrieval_calls'])}")
        print(f"    overhead on UNTOUCHED         n={ou['n']:<4d} steps {fmt(ou['steps'])}"
              + ("   <- MUST be ~0; non-zero means the arm perturbed episodes it never fired on"
                 if ou["steps"] not in (None, 0.0) else ""))
        print(f"    cost per corrected failure    converted={cc['n_converted']}  "
              f"extra steps/fix {fmt(cc['steps'])}  tokens {fmt(cc['total_tokens'])}")
        out["arms"][t]["cost"] = cd

    print(f"\nLOCALITY")
    for t in arm_tags:
        AC, AS, _ = arms[t]
        rep = locality_report(t, CS, AS, C, AC, fire_key=FIRE)
        r = rep.as_dict()
        print(f"  {t}:")
        print(f"    touched {rep.n_fired}/{rep.n_episodes} episodes "
              f"(rate {fmt(rep.episode_touch_rate)})   "
              f"target states {rep.n_target}, covered {rep.n_fired_on_target} "
              f"(coverage {fmt(rep.target_coverage)})")
        print(f"    on FIRED   : +{rep.gains_fired} / -{rep.losses_fired}   "
              f"on-target precision {fmt(rep.on_target_precision)}")
        print(f"    on NON-FIRED: {rep.changed_not_fired} changed "
              f"(+{rep.gains_not_fired}/-{rep.losses_not_fired})  <- cannot be the intervention")
        print(f"    opportunity created {rep.opportunity_created} (off-target firing "
              f"{fmt(rep.off_target_firing_rate)}) · missed {rep.opportunity_missed}")
        out["arms"][t]["locality"] = r

    print(f"\nOPTIMIZATION COST")
    if a.funnel and a.funnel.exists():
        raw = json.load(open(a.funnel))
        stages = raw.get("stages", raw)
        rej = raw.get("rejections", {})
        line = "  " + " -> ".join(
            f"{s}={stages.get(s, '-')}" for s in FUNNEL_STAGES)
        print(line)
        prev_k = prev_v = None
        for s in FUNNEL_STAGES:
            v = stages.get(s)
            if isinstance(v, int) and isinstance(prev_v, int) and prev_v:
                print(f"    {prev_k} -> {s}: x{v/prev_v:.2f}")
            if isinstance(v, int):
                prev_k, prev_v = s, v
        if rej:
            print(f"  rejections: {dict(sorted(rej.items()))}")
        out["funnel"] = raw
    else:
        print("  no funnel supplied (--funnel) -- candidate counts not recorded for this round")
    print(f"  evaluation episodes : {a.eval_episodes if a.eval_episodes is not None else 'not recorded'}")
    print(f"  evaluation tokens   : not instrumented")
    print(f"  GPU seconds         : {a.gpu_seconds if a.gpu_seconds is not None else 'not recorded'}")

    print(f"\nRESIDUAL PROGRESS")
    def parse(spec):
        vals = [int(x) for x in spec.split(",") if x.strip()]
        return tuple(ResidualEntry(i + 1, f"r{i+1}", v) for i, v in enumerate(vals))
    before = parse(a.residuals_before) if a.residuals_before else ()
    after = parse(a.residuals_after) if a.residuals_after else ()
    print(f"  before: {residual_shape(before) if before else 'not supplied'}")
    print(f"  after : {residual_shape(after) if after else 'not supplied (round not promoted)'}")
    out["residuals"] = {"before": residual_shape(before) if before else None,
                        "after": residual_shape(after) if after else None}

    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=2, default=str)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
