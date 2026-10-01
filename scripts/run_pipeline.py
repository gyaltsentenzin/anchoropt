#!/usr/bin/env python3
"""The A1-A4 pipeline: compose the anchors, state what to expect, and check what you got.

This is the entry point for "turn all four anchors on and reproduce the result". It has three modes,
because the honest answer to "will I get 42.24%?" depends on which one you can run.

    --show      print the composed policy: which anchors, which gates, which incision points.
                No GPU, no harness. Always works.

    --expect    print the exact numbers a full run must produce, per arm, with the acceptance
                bars that were frozen BEFORE those arms ran. No GPU. Always works.

    --check DIR compare a completed run's eval_*.json against those numbers and say PASS/FAIL
                per arm. This is how you confirm your own re-run matches ours.

Running the agent itself is NOT done here -- it needs a GPU and the BFCL v4 evaluator, which lives in
the working repo (see REPRODUCE.md Tier 3). This script covers everything around that: what to run,
what to expect, and whether you got it.

    python scripts/run_pipeline.py --show
    python scripts/run_pipeline.py --expect
    python scripts/run_pipeline.py --check /path/to/my/run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rounds.anchors import ANCHORS, CUMULATIVE, DEFERRED, PROGRESSION

# The composed A1-A4 policy. Every anchor accumulates into the LAST round's artifact -- that single
# file is the full incumbent, which is why it is the one to hand a runner.
FINAL_POLICY = REPO / "rounds" / "T5_A4_no_tool_call" / "policy.json"

# The FULL frozen stack: A1-A5 + A7. A7 itself has no policy flag (dispatch-gated), but this is the
# artifact that turns the five flagged anchors on and is the one to hand a runner.
EXTENDED_POLICY = REPO / "rounds" / "T8_A8_dedup_clear" / "policy.json"
# T9 adds A9 on top of the T8 stack -- one gate key. Kept separate because T8 is the fully MEASURED
# endpoint this repo can RECOMPUTE OFFLINE -- A9's shard artifacts are still on the cluster -- so a
# reproduction should start from T8. T9's figures are measured, just not recomputable here.
T9_POLICY = REPO / "rounds" / "T9_A9_xcontainer_merge" / "policy.json"

# Which switch in the policy belongs to which anchor. Stated here so `--show` can attribute every
# enabled flag to the round that earned it, rather than presenting an opaque config.
#
# A1-A4's switches live in FINAL_POLICY; A5's also appears in EXTENDED_POLICY. A7 is gated at
# DISPATCH on the backend rather than by a policy flag, so it has no switch to show -- recorded as
# an empty list rather than omitted, because "no flag" is a fact about the anchor.
ANCHOR_SWITCHES = {
    "A1": ["enable_capacity_repair", "enable_reroute",
           "gate:on_domain_error_core_full", "gate:on_core_full_rerouted"],
    "A2": ["gate:on_domain_error_key_not_found"],
    "A3": ["enable_redundant_write_suppress", "gate:on_redundant_write_suppressed"],
    "A4": ["enable_zero_call_reprompt"],
    "A5": ["enable_archival_evict_duplicate", "gate:on_archival_full_evict_duplicate"],
    "A7": [],
    "A8": ["gate:on_dedup_clear_recovery"],
    # A9 is doubly gated: this policy flag AND ANCHOROPT_XCM=1. Both are required, which is why an arm
    # shard that forgets the export runs as a control -- exactly how the delivery defect was caught.
    "A9": ["gate:on_low_similarity_cross_container"],
}

# Which anchors each shipped policy artifact turns on.
POLICY_SCOPE = {
    "A1": FINAL_POLICY, "A2": FINAL_POLICY, "A3": FINAL_POLICY, "A4": FINAL_POLICY,
    "A5": EXTENDED_POLICY, "A7": None, "A8": EXTENDED_POLICY, "A9": T9_POLICY,
}

# Arms a full reproduction must run, and what each must produce. Sourced from rounds/anchors.py so
# there is one source of truth.
EXPECTED = [
    ("native control (no anchors)", "train", 91, 303, "policy with every gate off"),
    ("A1-A4", "train", 128, 303, "rounds/T5_A4_no_tool_call/policy.json"),
    ("full stack (A1-A5+A7+A8)", "train", 144, 303, "rounds/T8_A8_dedup_clear/policy.json + A7 env"),
    ("native control (no anchors)", "dev", 14, 84, "policy with every gate off"),
    ("A1-A4", "dev", 22, 84, "rounds/T5_A4_no_tool_call/policy.json"),
    ("full stack (A1-A5+A7+A8)", "dev", 34, 84, "rounds/T8_A8_dedup_clear/policy.json + A7 env"),
    # A9 is reproduced on the VECTOR SHARD, which is where it was measured. There is deliberately no
    # whole-corpus row: those cells are measured but NOT reproducible from artifacts shipped here, and
    # listing a target this repo cannot reach offline would misrepresent what a reproduction verifies.
    ("+A9 (vector shard only)", "train/vector", 54, 89,
     "rounds/T9_A9_xcontainer_merge/policy.json + A7 env + ANCHOROPT_XCM=1"),
    ("+A9 (vector shard only)", "dev/vector", 21, 40,
     "rounds/T9_A9_xcontainer_merge/policy.json + A7 env + ANCHOROPT_XCM=1"),
]


def _switch_state(pol: dict, name: str) -> bool:
    if name.startswith("gate:"):
        return bool((pol.get("gate_enabled") or {}).get(name[5:], False))
    return bool(pol.get(name, False))


def show() -> int:
    pol = json.loads(FINAL_POLICY.read_text())
    extended = json.loads(EXTENDED_POLICY.read_text())
    # Each anchor's switches are read from the policy that actually CONTAINS it, keyed by
    # POLICY_SCOPE, rather than from one file that may predate it. A9 lives only in T9.
    _by_path = {FINAL_POLICY: pol, EXTENDED_POLICY: extended}
    print("=" * 78)
    print("THE COMPOSED A1-A4 POLICY")
    print("=" * 78)
    print(f"  artifact: {FINAL_POLICY.relative_to(REPO)}")
    print("  This ONE file is the full incumbent -- each anchor accumulated into it, so handing a")
    print("  runner this policy turns all four on at once.")
    print()
    print("  A1-A4's results ship as JSON in each round and this repo recomputes them. A5 and A7")
    print("  are accepted under the same rule, but their numbers are TRANSCRIBED from the working")
    print("  repo. A6 was REJECTED -- see rounds/A6_deferred/.")
    print()

    for a in ANCHORS:
        source = POLICY_SCOPE[a.name]
        if source is None:                       # gated at dispatch, no policy flag at all (A7)
            active = {}
        else:
            if source not in _by_path:
                _by_path[source] = json.loads(source.read_text())
            active = _by_path[source]
        switches = ANCHOR_SWITCHES[a.name]
        on = [s for s in switches if _switch_state(active, s)]
        off = [s for s in switches if not _switch_state(active, s)]
        print(f"  {a.name}  {a.incision_point.value:26s} {a.action.value:9s} {a.kind}")
        print(f"       locus     {a.locus}")
        if not switches:
            print("       enabled   (no policy flag -- gated at DISPATCH on the backend)")
        else:
            print(f"       enabled   {', '.join(on) if on else '(none)'}")
        if off:
            print(f"       NOT set   {', '.join(off)}")
        if source is not None and source is not FINAL_POLICY:
            print(f"       artifact  {source.relative_to(REPO)}")

    text = (pol.get("templates") or {}).get("on_turn_start_action", "").strip()
    if text:
        print()
        print("  A4's injected text (the anchor IS this text, fired at post-gen/pre-exec):")
        for line in (text[:300] + ("..." if len(text) > 300 else "")).splitlines():
            print(f"       {line}")

    preamble = (pol.get("templates") or {}).get("on_memory_preamble", "")
    print()
    print(f"  global preamble: {'EMPTY -- as required by this line' if not preamble.strip() else 'PRESENT (!)'}")
    print()
    print("  Note the pair that looks contradictory and is not:")
    print("      gate on_turn_start_action = false     <- A4 v1's pre-generation trigger, REJECTED")
    print("      enable_zero_call_reprompt = true      <- A4 v2's post-gen trigger, ACCEPTED")
    print("  Same signal, same text, one incision point later: -4.95 pp becomes +3.63 pp.")
    return 0


def expect() -> int:
    print("=" * 78)
    print("WHAT A FULL RUN MUST PRODUCE")
    print("=" * 78)
    print("  Run control and all-anchors-on, on each of train and DEV.")
    print("  Both arms of a split must run in the SAME job -- only within-job paired comparison is")
    print("  licensed. Never difference accuracies across jobs.")
    print()
    print("  DEV, not test: it is the accept/reject criterion, so it is a VALIDATION set and the")
    print("  reported figure is partly selected-on. There is no third reserved split.")
    print()
    print(f"  {'arm':26s} {'split':10s} {'expected':>12s}   policy")
    print("  " + "-" * 74)
    for label, split, ok, n, pol in EXPECTED:
        print(f"  {label:26s} {split:10s} {ok:3d}/{n:3d} = {100.0*ok/n:5.2f} %   {pol}")

    print()
    print("  Deltas that must follow, for the FULL stack:")
    print("    train  {:.2f} % -> {:.2f} %   = +{:.2f} pp".format(
        CUMULATIVE["train_from"], CUMULATIVE["train_to"], CUMULATIVE["train_delta_pp"]))
    print("    dev    {:.2f} % -> {:.2f} %   = +{:.2f} pp".format(
        CUMULATIVE["dev_from"], CUMULATIVE["dev_to"], CUMULATIVE["dev_delta_pp"]))
    print()
    print("  No p-value: the benchmark is DETERMINISTIC (byte-identical exec logs across")
    print("  independent runs), so there is no sampling distribution for a null. Report the paired")
    print("  (gains, losses) and the mechanism instead.")
    print()
    print("  REJECTED, and not part of the above -- run it only to reproduce the rejection:")
    for name, rec in DEFERRED.items():
        print(f"    {name}  train {rec['train_pp']:+.2f} pp / dev {rec['dev_pp']:+.2f} pp, "
              f"on-target {rec['on_target_dev_pp']:+.2f} pp -> {rec['status']}")
    print()
    print("  Per-round intermediate accuracies, if you run the rounds separately:")
    for r in PROGRESSION:
        tag = r.anchor.name if r.anchor else "none"
        print(f"    {r.tag:3s} {tag:5s} {r.train_correct:3d}/{r.train_n} = {r.train_acc:5.2f} %")

    print()
    print("  RUN PROTOCOL -- shard by backend, and submit the shards CONCURRENTLY.")
    print("    Every paired arm runs as one isolated job per backend, each serial internally.")
    print("    A 3-backend x 2-arm contrast is SIX SIMULTANEOUS JOBS, one GPU each, not six")
    print("    sequential ones. Every sub-job is independent: own process, own store, own")
    print("    snapshot-cache key, and chains are contiguous per backend so no chain can see")
    print("    another backend's store. There is nothing to serialise.")
    print()
    print("    It buys DETERMINISM, not just wall-clock: whole-corpus runs are NOT")
    print("    byte-reproducible -- two runs of the same effective policy diverge even when the")
    print("    treatment fires zero times. Sharded pairs are bit-identical except in the shard")
    print("    where the treatment fires. So a paired train number from a whole-corpus run must")
    print("    not be quoted. See docs/SATURATION_AND_SELECTION.md section 5.")
    print()
    print("    Ports must be distinct per (arm x backend), not per backend.")
    print("    This does NOT relax --store-workers 1: parallelism WITHIN a store build is")
    print("    forbidden; parallelism ACROSS isolated runs is measured at 0 flips.")
    print()
    print("  CONDITIONS. Change any of these and the numbers are not comparable:")
    print("    model granite-4.1-8b   temperature 0.001   store-workers 1")
    print("    balanced fold split, leakage 0, 303 scored train / 84 scored held-out")
    print("    BFCL v4 with the empty-memory kv ZeroDivisionError fix applied")
    print()
    print("  temperature=0.001 is load-bearing: it is why the replay variance floor is ZERO")
    print("  (0/12 stores, 0/303 calls, 0/303 flips), and therefore why a single-case loss counts.")
    print("  At a higher temperature these exact counts will NOT reproduce.")
    return 0


def check(run_dir: Path) -> int:
    """Compare a completed run against the expected numbers."""
    print("=" * 78)
    print(f"CHECKING {run_dir}")
    print("=" * 78)

    found = sorted(run_dir.rglob("eval_*results*.json"))
    if not found:
        print(f"  no eval_*results*.json under {run_dir}")
        print("  expected the output of a paired run (control + anchors-on).")
        return 1

    targets = {(ok, n) for _, _, ok, n, _ in EXPECTED}
    failures = []
    for f in found:
        try:
            rows = json.loads(f.read_text())["results"]
        except Exception as e:  # noqa: BLE001 - a malformed file is a checkable outcome
            print(f"  {f.name:44s} UNREADABLE ({type(e).__name__})")
            failures.append(str(f))
            continue
        q = [r for r in rows if not r.get("is_prereq")]
        ok, n = sum(1 for r in q if r.get("valid")), len(q)
        match = next((lbl for lbl, _s, e_ok, e_n, _p in EXPECTED if (e_ok, e_n) == (ok, n)), None)
        verdict = f"matches '{match}'" if match else "NO expected arm has this score"
        flag = "  PASS" if match else "  <-- unexpected"
        print(f"  {f.relative_to(run_dir)!s:44s} {ok:3d}/{n:3d} = {100.0*ok/n:5.2f} %  {verdict}{flag}")
        if not match:
            failures.append(f"{f.name}: {ok}/{n} matches no expected arm")

    print()
    if failures:
        print("FAIL -- at least one arm does not match the published numbers:")
        for x in failures:
            print(f"  - {x}")
        print()
        print("  Before concluding the anchors do not work, check the CONDITIONS in --expect.")
        print("  A different model, temperature, or split changes these counts legitimately.")
        return 1
    print(f"PASS -- all {len(found)} arm(s) match published numbers "
          f"(of {len(targets)} distinct expected scores).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compose the A1-A4 anchors, state expected results, check a completed run.",
        epilog="Running the agent needs a GPU and the BFCL v4 evaluator -- see REPRODUCE.md Tier 3.",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--show", action="store_true", help="print the composed A1-A4 policy")
    g.add_argument("--expect", action="store_true", help="print the numbers a full run must produce")
    g.add_argument("--check", metavar="DIR", help="check a completed run's results against them")
    args = ap.parse_args()

    if args.show:
        return show()
    if args.expect:
        return expect()
    return check(Path(args.check).expanduser().resolve())


if __name__ == "__main__":
    sys.exit(main())
