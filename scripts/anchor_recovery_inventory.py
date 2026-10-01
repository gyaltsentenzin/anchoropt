#!/usr/bin/env python3
"""Inventory the 8 accepted anchors and reconstruct each one's DISCOVERY CONTEXT. No GPU, no LLM.

    python scripts/anchor_recovery_inventory.py [--json-out docs/anchor_inventory.json]

WHY THIS COMES FIRST
    The paper claim is "Self-Evolve autonomously reconstructs interventions previously found through
    manual/iterative AnchorOpt development". Replaying that requires knowing, per anchor, the state of
    the world WHEN IT WAS FOUND -- which incumbent was in place, which failures were on the table, and
    crucially whether the signal it needs was even declared at that point. An anchor whose signal did
    not yet exist cannot be "recovered" by a search restricted to declared signals; that is a
    different experiment (signal invention), and conflating the two would report a search failure for
    a vocabulary reason.

    So: inventory first, pick the feasible subset, replay those. This costs nothing and prevents
    spending 16 GPU runs to discover which 4 were ill-posed.

THE HISTORICAL INCUMBENT IS THE LADDER, NOT LEAVE-ONE-OUT
    `rounds/T1..T9` each froze the stack as of that step, so anchor A_n's discovery incumbent is
    T(n-1)'s policy -- the anchors that already existed then, and only those. This is strictly more
    faithful than "full stack minus X": A3's own attribution says the model rewrites a value *A1
    already archived*, so A3 was discovered from a world containing A1 and NOT containing A5..A9.

RECOVERABILITY IS DECIDED BY THREE FACTS, all read from the code
    signal_declared   is a signal observing this condition in Phi_BFCL today?
    executor_exists   is (locus, action) in the runtime EXECUTORS registry?
    trigger_source    does the condition arise natively, or only because an earlier anchor created it?
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

import bfcl_runtime as runtime                                    # noqa: E402
from rounds.anchors import ANCHORS                                # noqa: E402

# Ladder position -> the round directory that FROZE the stack including that anchor.
LADDER = {"A1": "T1_A1_capacity", "A2": "T2_A2_not_found", "A3": "T3_A3_duplicate",
          "A4": "T5_A4_no_tool_call", "A5": "T6_A5_archival_full", "A7": "T7_A7_blob_overflow",
          "A8": "T8_A8_dedup_clear", "A9": "T9_A9_xcontainer_merge"}
ORDER = ["A1", "A2", "A3", "A4", "A5", "A7", "A8", "A9"]

# The policy switch(es) each anchor turns on -- from scripts/run_pipeline.ANCHOR_SWITCHES, so there is
# one source of truth for "installing anchor X".
SWITCHES = {
    "A1": ["enable_capacity_repair", "enable_reroute",
           "gate:on_domain_error_core_full", "gate:on_core_full_rerouted"],
    "A2": ["gate:on_domain_error_key_not_found"],
    "A3": ["enable_redundant_write_suppress", "gate:on_redundant_write_suppressed"],
    "A4": ["enable_zero_call_reprompt"],
    "A5": ["enable_archival_evict_duplicate", "gate:on_archival_full_evict_duplicate"],
    "A7": [],                                     # env-gated: ANCHOROPT_A7C, no policy flag
    "A8": ["gate:on_dedup_clear_recovery"],
    "A9": ["gate:on_low_similarity_cross_container"],
}

# Which backend cell each anchor's mechanism actually engages, from the acceptance records.
BACKEND = {"A1": "kv+vector", "A2": "kv", "A3": "kv", "A4": "vector+kv+rec_sum",
           "A5": "vector", "A7": "rec_sum", "A8": "kv", "A9": "vector"}

# Whether the anchor's trigger arises NATIVELY or is manufactured by an earlier anchor. Measured, not
# assumed: on the native incumbent (vector/train, n=89) A4 fired 26/89, A1 saw 1 event and A3 zero,
# and A3's own attribution says the collisions are against keys A1 injected.
TRIGGER_SOURCE = {
    "A1": "native (write blocked by a full container -- occurs without help)",
    "A2": "native (a read misses a key that exists elsewhere)",
    "A3": "DEPENDENT on A1: all 59 observed collisions are against keys A1 archived; 0 events natively",
    "A4": "native (26/89 = 29% on the native incumbent -- the highest native rate of any anchor)",
    "A5": "DEPENDENT on A1/A3 having filled archival: triggers on an archival-full duplicate write",
    "A7": "native within rec_sum (a blob append exceeds the cap)",
    "A8": "DEPENDENT on A1: a clear proposed against a container A1 filled",
    "A9": "native (retrieval similarity below threshold), but its CONTROL is the A1-A8 stack",
}

# What "equivalent" would mean for this anchor, stated BEFORE any replay so the bar cannot move.
EQUIVALENCE = {
    "A1": "reroute a capacity-blocked write to the other container, preserving the value",
    "A2": "after a failed lookup, retry the read against the container that holds it",
    "A3": "cancel a duplicate write before dispatch, preserving one copy",
    "A4": "intercept a zero-tool-call commitment and induce >=1 retrieval before answering",
    "A5": "on an archival-full duplicate, evict one redundant copy and retry the write verbatim",
    "A7": "compact/route a blob append that would exceed the cap, without losing content",
    "A8": "suppress a destructive clear, evict one duplicate instead, retry the blocked write",
    "A9": "on a weak core retrieval, consult the other container and merge the evidence",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=pathlib.Path)
    a = ap.parse_args()
    by_name = {x.name: x for x in ANCHORS}
    rows = []
    W = 104

    print("=" * W)
    print("ANCHOR RECOVERY INVENTORY  ·  discovery contexts reconstructed from the ladder")
    print("=" * W)

    for i, name in enumerate(ORDER):
        anc = by_name[name]
        prior = LADDER[ORDER[i - 1]] if i > 0 else "native (all gates off)"
        locus = anc.incision_point
        # is a signal observing this condition declared today?
        declared = runtime.declared_signals()
        sig_guess = anc.locus.split("/")[-1]
        sig_declared = [s for s in declared if sig_guess in s or s in anc.locus]
        ex = runtime.executor_for(locus, anc.action)
        rows.append({
            "anchor": name,
            "prior_incumbent": prior,
            "known_locus": locus.value,
            "known_signal_locus_string": anc.locus,
            "known_action": anc.action.value,
            "attribution": anc.attribution,
            "known_effect": (anc.notes or "").strip()[:200],
            "backend": BACKEND[name],
            "switches": SWITCHES[name],
            "signal_declared_today": sig_declared,
            "executor": ex["remedy_flag"] if ex else None,
            "materializable": bool(ex),
            "trigger_source": TRIGGER_SOURCE[name],
            "equivalence_criterion": EQUIVALENCE[name],
        })

    for r in rows:
        print(f"\n{'-' * W}\n{r['anchor']}   locus={r['known_locus']}   action={r['known_action']}"
              f"   backend={r['backend']}")
        print(f"  prior incumbent (discovery context): {r['prior_incumbent']}")
        print(f"  attribution (the attributor's prose, NOT the locus string):")
        print(f"      \"{r['attribution']}\"")
        print(f"  known effect        : {r['known_effect'][:96]}")
        print(f"  signal declared now : {r['signal_declared_today'] or 'NONE MATCHING -- signal gap'}")
        print(f"  executor for (l,mu) : {r['executor'] or 'NONE -- not materializable today'}")
        print(f"  trigger source      : {r['trigger_source']}")
        print(f"  equivalence means   : {r['equivalence_criterion']}")

    print(f"\n{'=' * W}\nFEASIBILITY TRIAGE for replaying Self-Evolve")
    print("=" * W)
    ready, blocked_exec, blocked_sig = [], [], []
    for r in rows:
        if not r["materializable"]:
            blocked_exec.append(r["anchor"])
        elif not r["signal_declared_today"]:
            blocked_sig.append(r["anchor"])
        else:
            ready.append(r["anchor"])
    print(f"  READY (signal declared AND executor exists) : {ready or 'none'}")
    print(f"  BLOCKED, no executor in the registry        : {blocked_exec or 'none'}")
    print(f"  BLOCKED, no declared signal                 : {blocked_sig or 'none'}")
    print()
    print("  NOTE the no-executor blockers are a REGISTRY-COVERAGE gap, not an absent capability:")
    print("  the evaluator implements enable_reroute, enable_capacity_repair,")
    print("  enable_redundant_write_suppress, enable_archival_evict_duplicate and")
    print("  on_low_similarity_cross_container. They are simply not declared in runtime.EXECUTORS,")
    print("  so executor_supports() prunes them before evaluation. Declaring them is a small,")
    print("  auditable change -- but it must be done from the evaluator's real firing sites, not guessed.")
    if a.json_out:
        json.dump(rows, open(a.json_out, "w"), indent=2)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
