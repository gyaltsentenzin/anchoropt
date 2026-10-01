#!/usr/bin/env python3
"""Seed the v0.2 experiment memory with what we already measured. Idempotent.

    python scripts/seed_evolve_memory.py [--root results/self_evolve/memory]

Every row here is a MEASURED result or a defect we paid for, with its round named. Nothing is
invented, and nothing is entered as `accepted` that was not actually promoted.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anchoropt.evolve_memory import Experiment, ExperimentLedger, Lesson, context_key  # noqa: E402

VECTOR_TRAIN = context_key(host="bfcl_v4_memory", split="train", backend="vector")
HELDOUT = context_key(host="bfcl_v4_memory", split="test", backend="vector+rec_sum")

R_A4 = "whether to commit to a final answer on the basis of the first non-empty retrieval, without checking it"
R_LOWSIM = "whether to accept the first non-empty retrieval as sufficient evidence for the final answer"

EXPERIMENTS = [
    # ---- A4 locus, the two evaluated eta realizations -------------------------------------------
    Experiment(
        round_id="SE3", residual=R_A4, locus="post_generation_pre_exec",
        signal="no_tool_call_at_all", action="reprompt",
        eta={"instruction": "Before answering, check whether the retrieved content actually states "
                            "the fact being asked for. If it does not, search again with different "
                            "terms.", "retry_budget": 1},
        status="rejected", context=VECTOR_TRAIN,
        gains=2, losses=0, firings=24, steps_delta=71,
        diagnosis="model commits to a final answer having made no tool call at all",
        reason="se3a1: engages correctly (24/24 real retrievals) but converts only 2/24 = 8.3% and "
               "invites a SECOND search in 20 of 24 cases (1.96 reads/case). Lost the eta_mu "
               "comparison to se3a2 on train and TIED it on held-out; costs +7.00 steps per fix."),
    Experiment(
        round_id="SE3", residual=R_A4, locus="post_generation_pre_exec",
        signal="no_tool_call_at_all", action="reprompt",
        eta={"instruction": "The container you searched may not hold this fact. Search the other "
                            "memory container before answering.", "retry_budget": 1},
        status="accepted", context=VECTOR_TRAIN,
        gains=4, losses=0, firings=24, steps_delta=57,
        diagnosis="model commits to a final answer having made no tool call at all",
        reason="se3a2 = A4', PROMOTED 2026-09-18. Train 4/24 = 16.7% conversion, 1.25 reads/case. "
               "Held-out (2 clean cells, n=60): +3.33pp, +2/-0, 12/12 target coverage, 0 off-target, "
               "mechanism 12/12. TIE-BREAK WAS COST, NOT ACCURACY: se3a1 ties it on held-out accuracy "
               "and regressions; se3a2 costs +3.50 vs +7.00 extra steps per fix. The train 4-vs-2 "
               "conversion gap was 2 cases and did NOT replicate."),
    Experiment(
        round_id="SE3", residual=R_A4, locus="post_generation_pre_exec",
        signal="no_tool_call_at_all", action="reprompt",
        eta={"instruction": "If no retrieved record states the requested value, say so explicitly "
                            "instead of guessing.", "retry_budget": 0},
        status="infeasible", context=VECTOR_TRAIN,
        diagnosis="model commits to a final answer having made no tool call at all",
        reason="se3a3: executor_parameter_conflict -- enable_zero_call_reprompt FIXES retry_budget=1 "
               "and this eta declares 0. Coercing it would run different semantics under the "
               "proposal's name, so it was rejected before evaluation, never run."),
    # ---- the low-similarity family, measured net-negative at every theta ------------------------
    Experiment(
        round_id="R1", residual=R_LOWSIM, locus="post_execution",
        signal="retrieval_similarity_below_threshold", action="reprompt",
        eta={"instruction": "Before answering, check whether the retrieved content actually states "
                            "the fact being asked for. If it does not, search again with different "
                            "terms.", "retry_budget": 1, "below": 0.75},
        status="rejected", context=VECTOR_TRAIN, gains=1, losses=4, firings=46,
        diagnosis="model accepts a weak retrieval as sufficient evidence",
        reason="c2 faithful install: -3.37pp, engaged and HURT (4 of 5 changed cases fired). theta=0.75 "
               "EXCEEDS the observed maximum similarity (max 0.6984), so it was never a threshold -- it "
               "was an unconditional reprompt."),
    Experiment(
        round_id="R2", residual=R_LOWSIM, locus="post_execution",
        signal="retrieval_similarity_below_threshold", action="reprompt",
        eta={"instruction": "Before answering, check whether the retrieved content actually states "
                            "the fact being asked for. If it does not, search again with different "
                            "terms.", "retry_budget": 1},
        status="rejected", context=VECTOR_TRAIN, gains=0, losses=5, firings=46,
        diagnosis="model accepts a weak retrieval as sufficient evidence",
        reason="theta sweep over 9 data-derived quantiles: NET-NEGATIVE AT EVERY THETA, monotonically "
               "worse as the threshold rises (-1 at 0.1135 to -5 at 0.5045). Fixing theta does not "
               "rescue this action: the reprompt tells the model to distrust a retrieval that was "
               "adequate. This rejects the FRAMING, not the whole family."),
    Experiment(
        round_id="R1", residual=R_LOWSIM, locus="post_execution",
        signal="no_informative_result", action="reprompt",
        eta={"instruction": "(installed as a failed-search streak trigger)", "retry_budget": 1},
        status="invalid", context=VECTOR_TRAIN, gains=0, losses=0, firings=0,
        diagnosis="model accepts a vacuous retrieval as sufficient evidence",
        reason="c3 approximate install: fired ZERO times. The installed trigger was a failed-search "
               "STREAK while the proposal was per-call vacuity -- not the same condition, so the run "
               "says nothing about the proposal. A zero-firing arm is not a null result."),
    # ---- reroute has no executor here -----------------------------------------------------------
    Experiment(
        round_id="SE3", residual=R_LOWSIM, locus="post_execution", signal="no_informative_result",
        action="reroute", eta={},
        status="infeasible", context=VECTOR_TRAIN,
        diagnosis="model accepts a vacuous retrieval as sufficient evidence",
        reason="post_execution/reroute has NO EXECUTOR on this host. 6 substitute arms + 1 transform "
               "arm were admissible in U_H(l) and never runnable; a reported '113 counterfactuals' was "
               "inflated by 91 non-materializable arms."),
]

LESSONS = [
    Lesson(key="upstream_cause", kind="structural", origin="A1/A3 recovery",
           text="The causal error may be UPSTREAM of the observed failure. Later anchors' triggers can "
                "be consequences of earlier anchors, so a residual's visible symptom is not "
                "necessarily where the intervention belongs."),
    Lesson(key="localize_first", kind="structural", origin="policy-opt architecture",
           text="Localize BEFORE selecting an action. Fixing (locus, signal) first sharply reduces the "
                "action space -- on the A4 locus it collapsed 10 admissible candidates to 2 "
                "materializable ones; choosing an action first lets the proposer invent families the "
                "runtime already knows are impossible."),
    Lesson(key="signal_observable", kind="structural", origin="A4v1 (-4.95pp)",
           text="The signal must be OBSERVABLE at the intervention boundary. A4v1 fired at step 0 "
                "because 'about to answer without looking' is not observable before generation; it "
                "injected in 303/303 episodes instead of ~48 and cost -4.95pp. That was a "
                "specification error, not a failed hypothesis."),
    Lesson(key="admissible_ne_materializable", kind="structural", origin="76ed8a2",
           text="ADMISSIBLE is not MATERIALIZABLE. U_H(locus) declares what the host permits; a real "
                "executor must exist for (locus, action), cover the signal, and consume eta UNCHANGED. "
                "A parameter the executor would coerce must be rejected, because coercion runs "
                "different semantics under the proposal's name."),
    Lesson(key="non_fired_not_attributable", kind="structural", origin="SE3 held-out",
           text="Do NOT attribute changes on non-fired episodes to the controller. If the arm never "
                "fired on a case, its outcome change is not the intervention -- check that every "
                "reported gain sits on a fired episode before quoting a delta."),
    Lesson(key="prereq_contamination", kind="structural", origin="SE3 held-out kv + SE1 rerun",
           text="PREREQ state can contaminate downstream evaluation. Prereq episodes write the store "
                "queries read, and the executor is armed for them too: held-out kv had 22/12/0 prereq "
                "firings across arms with differing prereq trajectories, so that cell was not a paired "
                "comparison. Check prereq firings per arm before trusting a delta."),
    Lesson(key="gate_rows_not_decisions", kind="structural", origin="SE3 step-0 retraction",
           text="Gate BOOKKEEPING rows are not model decisions. When the executor fires it appends a "
                "row with no status and no decoded calls; counting it as a decision compares the arm's "
                "firing record against the control's first real decision. This produced a retracted "
                "'11/89 sampling drift' finding."),
    Lesson(key="measurement_decides", kind="structural", origin="R1/R2/SE3",
           text="Measurement decides, not the proposer's preference and not raw net. Raw net once "
                "credited an arm for two cases where it never fired and picked the wrong winner; rank "
                "on the attributable population."),
    # ---- invalid approaches ---------------------------------------------------------------------
    Lesson(key="seed_variation_invalid", kind="invalid_approach", origin="seed-2 round",
           text="Varying the DECODE SEED is not a replication on this substrate. Decoding is greedy "
                "(temperature=0 + fixed seed), so the seed is inert: the seed-2 round reproduced "
                "seed 42 BIT-FOR-BIT (0 differing step fields across 89 episodes in 3 arms). "
                "Independent variation must come from a different cell, an eta paraphrase, or "
                "resampling."),
    Lesson(key="gates_fired_blind", kind="invalid_approach", origin="R3 telemetry",
           text="Reading firings from `gates_fired` is invalid for REMEDY-FLAG executors. That dict is "
                "built from telemetry_flags(), which is registry-derived, so zero_call_reprompt_gate "
                "is absent from it; the per-step trajectory sidecar is the only true source. Measured: "
                "sidecar 8 firings where gates_fired carried the key not at all."),
    Lesson(key="prestate_vacuous_step0", kind="invalid_approach", origin="SE3 replication protocol",
           text="Common-prestate eligibility is VACUOUS for a signal that fires on the model's first "
                "decision: the compared prefix is empty wherever the arm fires (measured: cut==step 0 "
                "in all 35 firing cases), so it passes without checking anything. Use the target "
                "population instead."),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path,
                    default=REPO / "results/self_evolve/memory")
    a = ap.parse_args()
    led = ExperimentLedger(a.root)
    have = {e.fingerprint for e in led.experiments()}
    added = 0
    for e in EXPERIMENTS:
        if e.fingerprint in have:
            continue
        led.record(e)
        have.add(e.fingerprint)
        added += 1
    n_lessons = len(led.lessons())
    for l in LESSONS:
        led.record_lesson(l)
    print(f"experiments: +{added} (total {len(led.experiments())})")
    print(f"lessons    : +{len(led.lessons()) - n_lessons} (total {len(led.lessons())})")
    print(f"root       : {a.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
