#!/usr/bin/env python3
"""Score Self-Evolve R1: denominator integrity FIRST, then paired scoring, then telemetry.

    python scripts/score_selfevolve_r1.py --results <dir with se1*_vector_train/>

ORDER IS THE POINT
------------------
Denominator integrity is a PRECONDITION, not a diagnostic. Nothing about accuracy is printed until
every arm covers exactly the intended cases and the two independent scorer paths agree, because a
corrupted denominator has already once produced wreckage that read as a clean null result. If the
check fails this script prints the problems and exits NON-ZERO without computing a single delta.

WHAT EACH ARM MEANS -- carried in the output so a reader cannot lose it
----------------------------------------------------------------------
    control     native incumbent, every gate off
    candidate 2 FAITHFUL install of the autonomously proposed (l, phi, mu, theta)
    candidate 3 APPROXIMATE install: the proposal's phi was per-call vacuity
                (no_informative_result); the evaluator's nearest trigger is a failed-search STREAK.
                Not the same condition, so a result here is about the approximation.
    candidate 1 NOT INSTALLABLE -- outside the realizable host space U_H(l). Reported, never run.

ENGAGEMENT IS NOT OPTIONAL
--------------------------
A positive delta with zero firings is proof of NON-attribution, never of a subtle effect. So firings
and interventions-executed are read from the arm's OWN decision-point telemetry and printed beside
every delta.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anchoropt.learning.round_runner import ArmOutcome, paired_evaluation  # noqa: E402

ARMS = {
    "se1ctl": ("control", "native incumbent, every gate off"),
    "se1c2": ("candidate 2", "FAITHFUL install of the proposed (l, phi, mu, theta)"),
    "se1c3": ("candidate 3", "APPROXIMATE install (streak, not per-call vacuity)"),
}


def load_arm(root: pathlib.Path, arm: str) -> tuple[dict, dict, dict]:
    """(primary, secondary, telemetry) for one arm.

    TWO INDEPENDENT SCORER PATHS, kept separate up to the comparison:
      primary    the evaluator's own per-case `valid` in eval_train.json
      secondary  recomputed from the per-case records' error_type/final answer presence
    A single path cannot detect its own missing cases, which is the whole reason for two.
    """
    d = root / f"{arm}_vector_train"
    res = d / "run" / "eval_train_results.json"
    if not res.exists():
        alt = sorted(d.glob("run/**/eval*.json"))
        if not alt:
            raise SystemExit(f"{arm}: no eval json under {d}")
        res = alt[0]
    payload = json.load(open(res))
    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload

    primary, secondary = {}, {}
    for r in rows:
        if r.get("is_prereq"):
            continue                      # prereqs are setup, never scored
        cid = str(r["id"])
        primary[cid] = bool(r.get("valid"))
        # independent recompute: a case passes iff it is valid AND carries no error_type
        secondary[cid] = bool(r.get("valid")) and not str(r.get("error_type") or "").strip()

    # ENGAGEMENT. Read from the arm's own per-case decision-point flags, which is where this
    # evaluator records them. `exec.jsonl` is a TOOL-CALL log (call/result/state_before/...) and
    # carries no gate keys at all -- an earlier version of this function scanned it for `*_gate`
    # and reported 0 firings beside a per-case count of 46, i.e. it would have called a real
    # intervention non-attributable. A zero-firing arm and an unparsed telemetry channel look
    # identical in a summary, so this counts the channel that actually exists.
    telemetry = {"signal_firings": 0, "interventions_executed": 0, "gate_detail": {},
                 "cases_fired": 0, "exec_log_lines": 0}
    exec_log = d / "exec.jsonl"
    if exec_log.exists():
        telemetry["exec_log_lines"] = sum(1 for _ in open(exec_log))

    for r in rows:
        fired_here = False
        for k, v in (r.get("gates_fired") or {}).items():
            if v:
                telemetry["gate_detail"][k] = telemetry["gate_detail"].get(k, 0) + 1
                telemetry["signal_firings"] += 1
                fired_here = True
        for flag in ("low_similarity_reprompt_gate", "g1_gate_fired", "d3_gate_fired"):
            if r.get(flag):
                telemetry["gate_detail"][flag] = telemetry["gate_detail"].get(flag, 0) + 1
                telemetry["signal_firings"] += 1
                fired_here = True
        for sig in (r.get("signals_fired") or []):
            telemetry["gate_detail"][f"signal:{sig}"] = \
                telemetry["gate_detail"].get(f"signal:{sig}", 0) + 1
        if fired_here:
            telemetry["cases_fired"] += 1
    # Every firing of a reprompt gate IS its execution: the text is injected at the firing site.
    telemetry["interventions_executed"] = telemetry["signal_firings"]
    return primary, secondary, telemetry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, required=True)
    a = ap.parse_args()

    loaded = {}
    for arm in ARMS:
        try:
            loaded[arm] = load_arm(a.results, arm)
        except SystemExit as exc:
            print(f"MISSING ARM {arm}: {exc}")
    if "se1ctl" not in loaded:
        print("no control arm -- nothing is interpretable")
        return 1

    intended = sorted(loaded["se1ctl"][0])
    print("=" * 96)
    print(f"SELF-EVOLVE R1  ·  vector/train  ·  intended cases: {len(intended)}")
    print("=" * 96)

    # ---------------------------------------------------------------- denominator FIRST
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
        print("\n  Refusing to compute or print any accuracy delta. A corrupted denominator has "
              "already\n  once produced wreckage that read as a clean null result.")
        return 2

    # ---------------------------------------------------------------- paired scoring
    print("\n[2] PAIRED SCORING vs control")
    ctl = outcomes["se1ctl"]
    n = len(intended)
    ctl_correct = sum(ctl.by_primary.values())
    print(f"  control: {ctl_correct}/{n} = {100*ctl_correct/n:.2f}%")

    verdicts = {}
    for arm in ("se1c2", "se1c3"):
        if arm not in outcomes:
            continue
        label, note = ARMS[arm]
        ev = paired_evaluation(ctl, outcomes[arm], intended)
        correct = sum(outcomes[arm].by_primary.values())
        delta = 100 * (correct - ctl_correct) / n
        tel = outcomes[arm].telemetry
        print(f"\n  {arm}  ({label} -- {note})")
        print(f"    accuracy      : {correct}/{n} = {100*correct/n:.2f}%   delta {delta:+.2f} pp")
        print(f"    gains/losses  : +{len(ev.gains)} / -{len(ev.losses)}   net {ev.net:+d}")
        print(f"    firings       : {tel['signal_firings']}  (on {tel['cases_fired']} of {n} cases)")
        print(f"    executed      : {tel['interventions_executed']}")
        if tel["gate_detail"]:
            print(f"    gate detail   : {dict(sorted(tel['gate_detail'].items()))}")
        print(f"    denominator_ok: {ev.denominator_ok}")
        verdicts[arm] = {"label": label, "delta_pp": round(delta, 4), "net": ev.net,
                         "gains": list(ev.gains), "losses": list(ev.losses),
                         "firings": tel["signal_firings"],
                         "executed": tel["interventions_executed"],
                         "cases_fired": tel["cases_fired"],
                         "gate_detail": tel["gate_detail"], "n": n,
                         "correct": correct, "control_correct": ctl_correct}

    # ---------------------------------------------------------------- acceptance
    print("\n[3] ACCEPTANCE RULE (AnchorOpt's, on measurement -- not the proposer's ranking)")
    from anchoropt.learning.self_evolve import _default_accept
    accepted = []
    for arm, v in verdicts.items():
        ev = paired_evaluation(ctl, outcomes[arm], intended)
        ok, reason = _default_accept(ev)
        print(f"  {arm}: {'ACCEPT' if ok else 'REJECT'} -- {reason}")
        if ok:
            accepted.append((arm, v))
    if accepted:
        win = max(accepted, key=lambda av: av[1]["net"])
        print(f"\n  WINNER ON MEASUREMENT: {win[0]} (net {win[1]['net']:+d})")
    else:
        print("\n  no candidate cleared the acceptance rule -- nothing is promoted")

    out = a.results / "selfevolve_r1_score.json"
    out.write_text(json.dumps({"denominator_ok": all_ok, "n": n,
                               "control_correct": ctl_correct, "arms": verdicts,
                               "accepted": [x[0] for x in accepted],
                               "candidate_1": "NOT INSTALLABLE -- outside U_H(l)"},
                              indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
