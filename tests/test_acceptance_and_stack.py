"""The acceptance rule, the accepted stack, and the anchor that failed the rule.

These tests exist because the *rejection* is the most losable part of the record. Gains keep
themselves; the sentence explaining that an anchor was dropped, which criteria it failed, and that its
signal is still open is what an edit pass deletes. So each is pinned to the number it belongs to.

They also pin two facts a reader would otherwise take on trust:

    * the re-mine ledgers are frozen, so "the miner chose this locus" is checkable, and
    * T6-T7's accuracies are TRANSCRIBED, not recomputed here, unlike T0-T5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchoropt.anchor import Action, IncisionPoint
from rounds.anchors import (
    A5,
    A6_EVIDENCE,
    A7,
    A7_EVIDENCE,
    ACCEPTANCE_CLASSES,
    ACCEPTANCE_RULE,
    ANCHORS,
    AUDIT,
    CALL_COST,
    CUMULATIVE,
    DEFERRED,
    E1,
    E1_EVIDENCE,
    EFFICIENCY_REQUIREMENTS,
    MEASUREMENT_GENERATIONS,
    NAMING_TRAPS,
    PROGRESSION,
    REMINE_LEDGERS,
    SAFETY,
    SPLITS,
    audit_table,
    delta_table,
)

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------------------------

def test_the_rule_has_four_criteria_in_order():
    names = [name for name, _ in ACCEPTANCE_RULE["criteria"]]
    assert names == ["train", "dev", "attribution", "safety"]


def test_the_rule_is_explicitly_not_zero_loss():
    """A zero-loss rule would have rejected A5, which is the concrete reason for the wording."""
    text = ACCEPTANCE_RULE["is_not"].lower()
    assert "zero losses" in text
    assert "aggregate" in text
    assert "a5" in text, "the rule states which anchor a zero-loss reading would have cost"


def test_the_rule_mandates_paired_shape_reporting():
    text = ACCEPTANCE_RULE["reporting"].lower()
    assert "gains" in text and "losses" in text
    assert "shape" in text


def test_the_rejected_epsilon_is_recorded_with_its_reason():
    """A tolerance that admits exactly the one anchor it decides is fitted, not principled."""
    text = ACCEPTANCE_RULE["epsilon_note"]
    assert "1.19" in text
    assert "exactly" in text.lower()
    assert abs(DEFERRED["A6"]["dev_pp"]) == pytest.approx(1.19, abs=0.01)


def test_p_values_are_diagnostic_because_the_benchmark_is_deterministic():
    text = ACCEPTANCE_RULE["p_values"].lower()
    assert "determinis" in text
    assert "sampling" in text
    assert CUMULATIVE["p_value"] is None


def test_two_acceptance_classes_exist_and_efficiency_is_one_of_them():
    """Applying the accuracy rule to an efficiency anchor is a category error."""
    assert set(ACCEPTANCE_CLASSES) == {"accuracy", "efficiency"}
    assert "0 pp delta is a PASS" in ACCEPTANCE_CLASSES["efficiency"]
    assert ACCEPTANCE_RULE["class"] == "accuracy"


def test_the_efficiency_class_has_its_own_three_requirements():
    """Each came from a measured failure, and none is a restatement of the accuracy rule."""
    assert len(EFFICIENCY_REQUIREMENTS) == 3
    joined = " ".join(f"{a} {b}" for a, b in EFFICIENCY_REQUIREMENTS).lower()
    assert "executed" in joined and "proposals" in joined
    assert "verbatim" in joined
    assert "16 -> 0" in joined


# ---------------------------------------------------------------------------------------------
# The stack
# ---------------------------------------------------------------------------------------------

def test_the_stack_is_eight_accuracy_anchors():
    assert [a.name for a in ANCHORS] == ["A1", "A2", "A3", "A4", "A5", "A7", "A8", "A9"]
    assert CUMULATIVE["n_anchors"] == 8


def test_a6_is_not_in_the_stack():
    assert "A6" not in {a.name for a in ANCHORS}


def test_the_progression_chain_is_arithmetic_not_asserted():
    assert len({r.train_n for r in PROGRESSION}) == 1, "one denominator, always"
    first, last = PROGRESSION[0], PROGRESSION[-1]
    assert first.train_acc == pytest.approx(CUMULATIVE["train_from"], abs=0.01)
    assert last.train_acc == pytest.approx(CUMULATIVE["train_to"], abs=0.01)
    assert last.train_acc - first.train_acc == pytest.approx(
        CUMULATIVE["train_delta_pp"], abs=0.01
    )


def test_the_dev_chain_closes_too():
    """An earlier version of this record had a dev chain that did NOT close. It must now."""
    dev_rounds = [r for r in PROGRESSION if r.dev_acc is not None]
    assert dev_rounds[0].dev_acc == pytest.approx(CUMULATIVE["dev_from"], abs=0.01)
    assert dev_rounds[-1].dev_acc == pytest.approx(CUMULATIVE["dev_to"], abs=0.01)
    import itertools

    walked = dev_rounds[0].dev_acc + sum(
        b.dev_acc - a.dev_acc for a, b in itertools.pairwise(dev_rounds)
    )
    assert walked == pytest.approx(CUMULATIVE["dev_to"], abs=0.01)


def test_every_anchor_in_the_audit_has_a_verdict_and_the_rejected_one_says_so():
    assert set(AUDIT) == {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"}, (
        "the audit covers the REJECTED anchor too -- an audit of only what passed is not an audit"
    )
    in_stack = {a.name for a in ANCHORS}
    for name, row in AUDIT.items():
        verdict = row[-1]
        assert verdict, f"{name} has no verdict"
        if name not in in_stack and name != "A1":
            assert "REJECT" in verdict
        else:
            assert "REJECT" not in verdict


def test_safety_criterion_is_recorded_for_the_stack():
    assert SAFETY == {"clears_added": 0, "removes_added": 0}


# ---------------------------------------------------------------------------------------------
# The rejected anchor
# ---------------------------------------------------------------------------------------------

def test_a6_fails_two_criteria_and_both_are_named():
    rec = DEFERRED["A6"]
    assert rec["fails_criteria"] == (2, 3)
    assert rec["dev_pp"] < 0, "criterion 2"
    assert rec["on_target_dev_pp"] < 0, "criterion 3 -- on target and NEGATIVE"


def test_a6s_on_target_failure_is_independent_of_the_aggregate():
    """The cleanest statement of the failure: it engages as designed and makes things worse there.

    That is why no choice of tolerance rescues it -- criterion 3 does not involve the aggregate.
    """
    assert A6_EVIDENCE["exposed_dev_delta_pp"] < 0
    assert A6_EVIDENCE["exposed_backend"] == "vector"


def test_a6s_signal_is_deferred_not_refuted():
    """DEFERRED, not REJECTED: the signal is real and open; only the remedy fell short.

    The word matters. "Rejected" would imply the locus was a mistake to pursue -- but the miner ranked
    it first, and the errors it targets are still firing.
    """
    rec = DEFERRED["A6"]
    status = rec["status"].lower()          # the claim is pinned, not the capitalisation
    assert "deferred" in status and "not refuted" in status
    assert "not ready" in status, "the remedy is what fell short, not the signal"
    signal = rec["signal_still_present"]
    assert signal["entry_too_long_events"]["train"] > 0
    assert signal["entry_too_long_events"]["dev"] > 0
    assert "lost again" in signal["consequence"]
    assert "1.65" in rec["cost_of_dropping"]


def test_a6_records_reopening_conditions_and_forbids_just_re_enabling():
    rec = DEFERRED["A6"]
    assert len(rec["reopen_conditions"]) == 4
    assert "do NOT simply re-enable" in rec["standing_instruction"]


def test_a6_displacement_reproduces_the_selection_verdict():
    """The shipped diagnostic must classify A6's own measured figures as SELECTION."""
    from anchoropt.learning.exposure import displacement_check

    sites = A6_EVIDENCE["dev_sites"]
    report = displacement_check(
        incumbent_sites={name: before for name, (before, _) in sites.items() if before},
        candidate_sites={name: after for name, (_, after) in sites.items()},
    )
    assert report.verdict == "SELECTION"
    assert report.displaces_incumbent


def test_a6_content_swap_shows_no_information_added():
    before, after = A6_EVIDENCE["final_store_distinct"]
    out, added = A6_EVIDENCE["facts_swapped"]
    assert abs(before - after) <= 1, "capacity is effectively constant"
    assert out > 40 and added > 40, "a near-total swap, not an incremental addition"


def test_every_a6_variant_is_negative_on_dev():
    """Not a tuning problem: all three tried variants regress. Tuning is closed."""
    assert A6_EVIDENCE["all_variants_negative_on_dev"] is True
    for name, v in A6_EVIDENCE["variants"].items():
        assert v["dev_pp"] < 0, f"{name} is not negative on dev; update the claim"
    assert A6_EVIDENCE["best_variant_is_train_maximum"] is False
    best = A6_EVIDENCE["variants"][A6_EVIDENCE["best_variant"]]
    assert best["dev_pp"] == max(v["dev_pp"] for v in A6_EVIDENCE["variants"].values())


def test_a6_did_not_block_a7():
    """Checked, not assumed: A6 never fires on the backend A7 acts on."""
    text = DEFERRED["A6"]["a7_independence"].lower()
    assert "bit-identical" in text
    assert "not a stepping stone" in text


def test_a6_is_not_a_numbered_round_directory():
    """A rejected anchor must not look like a round of the progression."""
    assert (REPO / "rounds" / "A6_deferred").is_dir()
    assert (REPO / "rounds" / "A6_deferred" / "DEFERRED.md").exists()
    assert not list((REPO / "rounds").glob("T*A6*")), "A6 must not be numbered as a round"


# ---------------------------------------------------------------------------------------------
# A5 -- process criticism and evidence, both kept
# ---------------------------------------------------------------------------------------------

def test_a5_is_an_eviction_at_post_execution():
    assert A5.incision_point is IncisionPoint.POST_EXECUTION
    assert A5.action is Action.REROUTE
    assert not A5.incision_point.can_prevent_execution


def test_a5_mechanism_is_causally_validated_not_argued():
    """Criterion 3 asks for telemetry, and the 30/30 figures are what satisfies it."""
    assert "30" in A5.notes
    assert "CAUSALLY VALIDATED" in A5.notes


def test_a5_keeps_the_process_criticism_alongside_the_evidence():
    """The dev result does not erase the process failure, and the record says both."""
    notes = A5.notes.lower()
    assert "override" in notes
    assert "procedurally wrong and substantively right" in notes


def test_a5_ships_an_override_document():
    d = REPO / "rounds" / "T6_A5_archival_full"
    assert (d / "OVERRIDE.md").exists()
    assert "override" in (d / "OVERRIDE.md").read_text().lower()


# ---------------------------------------------------------------------------------------------
# A7 -- exposure, and the end-to-end confirmation
# ---------------------------------------------------------------------------------------------

def test_a7_exposure_dilution_is_arithmetic():
    ev = A7_EVIDENCE
    diluted = ev["exposed_train_delta_pp"] * ev["exposed_train_n"] / 303
    assert diluted == pytest.approx(ev["corpus_train_delta_pp"], abs=0.05)


def test_a7_corpus_figures_are_measured_not_spliced():
    """The three-backend confirmation replaced the spliced figures. The label must follow."""
    assert A7_EVIDENCE["measured_end_to_end"] is True
    assert A7_EVIDENCE["corpus_figures_are_spliced"] is False


def test_a7_off_target_is_a_bit_exact_no_op():
    """Stronger than the criterion required: not accuracy-neutral, the SAME computation."""
    ev = A7_EVIDENCE
    assert ev["off_target_cases"] == 258
    assert ev["off_target_disagreements"] == 0
    assert "byte-identical" in ev["off_target_exec_logs"]


def test_a7_simplest_variant_won():
    variants = A7_EVIDENCE["variants"]
    assert variants["v3_fit_and_shorter"] == max(variants.values())
    assert variants["v4_ratio_floor"] < 0
    for name, value in variants.items():
        if name != "v3_fit_and_shorter":
            assert value < variants["v3_fit_and_shorter"]


def test_a7_records_its_llm_call_and_its_decode_channel():
    assert A7_EVIDENCE["uses_llm_at_inference"] is True
    assert A7.params["uses_llm_at_inference"] is True
    assert "1 of 26" in A7_EVIDENCE["decode_channel"]


def test_a7_unexplained_gains_are_called_unexplained():
    caveats = " ".join(A7_EVIDENCE["caveats"]).lower()
    assert "unexplained" in caveats and "19 of" in caveats


# ---------------------------------------------------------------------------------------------
# E1 -- accepted on a second objective, and not currently active
# ---------------------------------------------------------------------------------------------

def test_e1_is_accepted_on_the_efficiency_objective():
    assert E1_EVIDENCE["accepted"] is True
    assert E1_EVIDENCE["acceptance_class"] == "efficiency"
    assert E1_EVIDENCE["criteria_passed"] == "8/8"
    assert E1_EVIDENCE["task_delta_pp"] == 0.00
    assert E1_EVIDENCE["flips"] == 0


def test_e1_is_suppress_at_the_only_admissible_point():
    assert E1.action is Action.SUPPRESS
    assert E1.incision_point is IncisionPoint.POST_GENERATION_PRE_EXEC
    assert E1.incision_point.sees_proposed_call
    assert E1.incision_point.can_prevent_execution


def test_accepted_is_not_the_same_as_active():
    """The distinction a previous pass collapsed. Both halves must be stated."""
    assert E1_EVIDENCE["accepted"] is True
    assert E1_EVIDENCE["active_in_frozen_policy"] is False
    why = E1_EVIDENCE["why_inactive"].lower()
    assert "0 times" in why
    assert "oversight" in why and "open item" in why, (
        "an assembly oversight that cannot be proven from artifacts is an OPEN ITEM, not a claim"
    )
    assert "none of them are affected" in why, (
        "the accuracy numbers must be stated as unaffected either way"
    )


def test_e1_reactivation_requires_reconfirming_against_the_current_stack():
    text = E1_EVIDENCE["to_reactivate"].lower()
    assert "current stack" in text
    assert "a1-a4" in text, "the last confirmation was against an older incumbent"


def test_e1_is_not_in_the_accuracy_progression():
    """Different objective, different table. A 0 pp row would read as a failed accuracy anchor."""
    assert E1 not in ANCHORS
    assert all(r.anchor is not E1 for r in PROGRESSION)
    assert "E1" not in delta_table()


def test_e1_evidence_states_the_scope_that_prevents_overstatement():
    ev = E1_EVIDENCE
    assert ev["calls_withheld"] == 16
    assert ev["calls_sent_to_execute"] == 0
    assert ev["executed_redundancy_before"] == 16
    assert ev["executed_redundancy_after"] == 0
    assert ev["firings_in_scored_episodes"] == 0, "THE limiting fact"
    assert ev["dev_run"] is None
    assert ev["backends_fired"] == ("vector",)
    assert "kv" in ev["backends_claimed"], "claimed but unobserved -- the caveat must survive"
    assert len(ev["caveats"]) >= 5


def test_e1_criteria_file_is_not_called_frozen():
    d = REPO / "rounds" / "E1_efficiency"
    assert (d / "CRITERIA.md").exists()
    assert not (d / "FROZEN.md").exists()


# ---------------------------------------------------------------------------------------------
# Call cost -- the efficiency class's own criterion
# ---------------------------------------------------------------------------------------------

def test_call_cost_shows_the_stack_is_not_uniformly_cheaper():
    """Reporting only the saving would be selective: A7 saves calls, A5 spends them."""
    m = CALL_COST["measurements"]
    assert m["A7_rec_sum_train"]["steps"] < m["A7_rec_sum_train"]["control_steps"]
    assert m["A5_vector_dev"]["steps"] > m["A5_vector_dev"]["control_steps"]
    assert "accuracy_bought" in m["A5_vector_dev"]
    assert "accuracy_bought" in m["A7_rec_sum_train"]


def test_call_cost_records_that_the_budget_never_binds():
    assert CALL_COST["budget"] == 20
    assert CALL_COST["budget_ever_binding"] is False
    caveats = " ".join(CALL_COST["caveats"]).lower()
    assert "query phase only" in caveats, "the prereq phase needs separate counting"
    assert "never state a percentage token saving" in caveats


# ---------------------------------------------------------------------------------------------
# The re-mine, and reproducibility
# ---------------------------------------------------------------------------------------------

def test_remine_ledgers_ship_and_are_frozen():
    for key, meta in REMINE_LEDGERS.items():
        path = REPO / meta["path"]
        assert path.exists(), f"{key} ledger is missing: {meta['path']}"
        data = json.loads(path.read_text())
        assert data.get("frozen") is True
        assert data.get("fingerprint") == meta["fingerprint"], (
            f"{key} fingerprint drifted -- a frozen ledger must never be edited"
        )


def test_the_miner_chose_the_locus_a6_targeted():
    """A6's rejection does not retract the RANKING that selected its locus."""
    meta = REMINE_LEDGERS["n0t6"]
    assert meta["targeted_by"] == "A6"
    assert meta["rank1_locus"] == DEFERRED["A6"]["locus"]
    data = json.loads((REPO / meta["path"]).read_text())
    groups = {
        "/".join([g["constraint_type"], g["resource"], g["violation_mode"]]): g["events"]
        for g in data["groups"]
    }
    assert groups[meta["rank1_locus"]] == meta["rank1_events"]


def test_sharding_is_the_authoritative_measurement_generation():
    mg = MEASUREMENT_GENERATIONS
    assert "sharded" in mg["authoritative"]
    assert "concurrent" in mg["authoritative"]
    assert "not byte-reproducible" in mg["why"]
    assert "SIX SIMULTANEOUS" in mg["concurrency"], "concurrency is the point, not a nicety"


def test_the_one_stochastic_channel_is_named():
    text = MEASUREMENT_GENERATIONS["one_stochastic_channel"].lower()
    assert "a7" in text and "1 of 26" in text
    assert "rare" in text


def test_naming_traps_are_recorded_with_the_general_lesson():
    joined = " ".join(f"{a} {b}" for a, b in NAMING_TRAPS).lower()
    assert "e1" in joined and "renamed" in joined
    assert "a label is a hypothesis" in joined


def test_early_rounds_are_recomputable_from_shipped_results():
    """T0-T5 must agree with the shipped JSONs -- these are not transcribed."""
    def scored(rel: str) -> tuple[int, int]:
        rows = json.loads((REPO / "rounds" / rel).read_text())["results"]
        queries = [r for r in rows if not r.get("is_prereq")]
        return sum(1 for r in queries if r.get("valid")), len(queries)

    # T1/T3/T5 must agree case-for-case with what they shipped.
    checks = {
        "T1": "T1_A1_capacity/result/eval_train.json",
        "T3": "T3_A3_duplicate/result/eval_train.json",
        "T5": "T5_A4_no_tool_call/result/eval_train.json",
    }
    for tag, rel in checks.items():
        rnd = next(r for r in PROGRESSION if r.tag == tag)
        assert scored(rel) == (rnd.train_correct, rnd.train_n), (
            f"{tag} disagrees with its shipped result file"
        )

    # T0 and T2 come from the LATER native measurement, and the shipped artifacts predate it. The
    # divergence is expected and recorded, so pin the size of it rather than asserting equality --
    # otherwise the next reader "fixes" an artifact that is a faithful record of what that arm did.
    from rounds.anchors import NATIVE_BASELINE, RECOMPUTABLE

    assert RECOMPUTABLE["shipped_t0_predates_the_native_measurement"] is True
    shipped_t0, _ = scored("T1_A1_capacity/result/baseline_T0_train.json")
    t0 = next(r for r in PROGRESSION if r.tag == "T0")
    assert shipped_t0 == NATIVE_BASELINE["previously_quoted_train"] == 88
    assert t0.train_correct == NATIVE_BASELINE["train_correct"] == 91
    assert t0.train_correct - shipped_t0 == 3, "the recorded divergence is exactly 3 cases"


def test_early_dev_rounds_are_recomputable_too():
    """The dev column for T1-T5 comes from shipped eval_test.json artifacts, not from prose.

    These were in the repo all along and the progression simply did not read them. Pinning them here
    means the dev chain is as artifact-backed as the train chain for those rounds.
    """
    def scored(rel: str) -> tuple[int, int]:
        rows = json.loads((REPO / "rounds" / rel).read_text())["results"]
        queries = [r for r in rows if not r.get("is_prereq")]
        return sum(1 for r in queries if r.get("valid")), len(queries)

    checks = {
        "T1": "T1_A1_capacity/result/eval_test.json",
        "T2": "T2_A2_not_found/result/eval_test.json",
        "T3": "T3_A3_duplicate/result/eval_test.json",
        "T5": "T5_A4_no_tool_call/result/eval_test.json",
    }
    for tag, rel in checks.items():
        rnd = next(r for r in PROGRESSION if r.tag == tag)
        assert scored(rel) == (rnd.dev_correct, rnd.dev_n), (
            f"{tag}'s dev figure disagrees with its shipped result file"
        )

    # T4 installed nothing, so its dev figure must equal T3's by construction.
    t3, t4 = (next(r for r in PROGRESSION if r.tag == tag) for tag in ("T3", "T4"))
    assert t4.dev_correct == t3.dev_correct


def test_every_installed_round_reports_a_dev_figure():
    """Dev is the accept/reject criterion, so a round without one has not been judged under the rule."""
    for rnd in PROGRESSION:
        assert rnd.dev_correct is not None, (
            f"{rnd.tag} has no dev figure -- dev IS criterion 2, so it cannot be blank"
        )


def test_t8_ships_its_measured_per_backend_results():
    """T8's numbers are now recomputable here, not transcribed: six shards, all of them shipped.

    This is the only round whose per-backend split ships, which is why the README can call those
    cells measured.
    """
    from rounds.anchors import CUMULATIVE, PER_BACKEND

    result = REPO / "rounds" / "T8_A8_dedup_clear" / "result"
    assert result.is_dir(), "T8 must ship the artifacts its cells are read from"

    totals = {"train": [0, 0], "test": [0, 0]}
    for backend in PER_BACKEND["train"]:
        for split, key in (("train", "train"), ("test", "dev")):
            path = result / f"stack_{backend}_{split}.json"
            assert path.exists(), f"missing {path.name}"
            rows = json.loads(path.read_text())
            rows = rows["results"] if isinstance(rows, dict) else rows
            queries = [r for r in rows if not r.get("is_prereq")]
            correct = sum(1 for r in queries if r.get("valid"))
            assert (correct, len(queries)) == PER_BACKEND[key][backend], (
                f"{backend}/{split} disagrees with PER_BACKEND"
            )
            totals[split][0] += correct
            totals[split][1] += len(queries)

    _m = CUMULATIVE["measured_endpoint"]        # T8: these artifacts are the six-shard T8 run
    assert tuple(totals["train"]) == tuple(_m["train"])
    assert tuple(totals["test"]) == tuple(_m["dev"])


def test_the_remaining_transcribed_rounds_say_so():
    for name in ("T6_A5_archival_full", "T7_A7_blob_overflow"):
        assert not (REPO / "rounds" / name / "result").exists(), (
            f"{name} now ships results -- move it into the recomputed set"
        )
    assert "transcribed" in (REPO / "README.md").read_text().lower()


def test_the_frozen_policy_matches_the_cited_hash_and_excludes_a6():
    import hashlib

    # T8 is the fully MEASURED endpoint and the file a reproduction should start from, so its hash is
    # pinned separately from the current stack's. Both are checked; neither substitutes for the other.
    t8 = REPO / "rounds" / "T8_A8_dedup_clear" / "policy.json"
    assert hashlib.sha256(t8.read_bytes()).hexdigest() == \
        CUMULATIVE["policy_sha256_measured_endpoint_t8"], "the T8 policy no longer matches its hash"

    policy = REPO / "rounds" / "T9_A9_xcontainer_merge" / "policy.json"
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    assert digest == CUMULATIVE["policy_sha256"], "the frozen policy no longer matches its hash"

    # T9 must differ from T8 in EXACTLY one gate key. A stack policy that drifted anywhere else would
    # make A9's paired delta un-attributable.
    import json
    a, b = json.loads(t8.read_text()), json.loads(policy.read_text())
    changed = {k for k in set(a["gate_enabled"]) | set(b["gate_enabled"])
               if a["gate_enabled"].get(k) != b["gate_enabled"].get(k)}
    assert changed == {"on_low_similarity_cross_container"}, f"T9 differs by more than A9: {changed}"

    flags = json.loads(policy.read_text())
    assert "enable_entry_length_reroute" not in flags, "A6's flag must be gone, not just false"
    assert "enable_vector_raw_exact_dedup" not in flags
    assert flags["enable_archival_evict_duplicate"] is True, "A5 is in the stack"
    assert flags["gate_enabled"]["on_dedup_clear_recovery"] is True, "A8 is in the stack"


def test_no_round_introduces_a_global_preamble():
    for path in sorted((REPO / "rounds").glob("T*/policy.json")):
        policy = json.loads(path.read_text())
        preamble = (policy.get("templates") or {}).get("on_memory_preamble", "")
        assert not preamble.strip(), f"{path.parent.name} has a global preamble; this line forbids it"


# ---------------------------------------------------------------------------------------------
# Rendering: the tables must show the rejection too
# ---------------------------------------------------------------------------------------------

def test_the_delta_table_shows_the_rejected_anchor_beneath_the_gains():
    """A reader must not be able to see the progression without seeing what was rejected."""
    table = delta_table()
    assert "A6" in table
    assert "Rejected under the same rule" in table
    for anchor in ANCHORS:
        assert anchor.name in table


def test_the_audit_table_includes_the_rejected_anchor():
    table = audit_table()
    assert "REJECT" in table
    assert "A6" in table


def test_splits_are_named_train_and_dev():
    assert set(SPLITS) == {"train", "dev"}
    assert SPLITS["dev"]["is_validation_not_test"] is True
    assert SPLITS["dev"]["one_case_pp"] == pytest.approx(1.19, abs=0.01)


def test_the_deferred_mechanism_module_declares_its_own_status():
    """Someone reading the code, not the docs, must learn A6 is not in the stack.

    The module ships because the signal is real. A reader who copies it without seeing the verdict
    would be porting a remedy that was measured to regress the validation split.
    """
    src = (REPO / "anchoropt" / "mechanisms" / "entry_length_reroute.py").read_text()
    assert "DEFERRED, NOT ACCEPTED" in src
    assert "-1.19" in src and "-2.50" in src, "both failing criteria must be quantified in-module"
    assert "DEFERRED.md" in src, "the module must point at the full record"


# ---------------------------------------------------------------------------------------------
# The substrate framing: the decision generalizes, the action does not
# ---------------------------------------------------------------------------------------------
#
# This is the interpretive claim that makes the later anchors legible rather than looking like a
# retreat into per-backend hacks, so both places that state it are pinned. It is also the claim a
# reader is most likely to challenge, which is exactly why it should not be quietly editable.

def test_the_readme_states_the_decision_action_split():
    body = (REPO / "README.md").read_text()
    low = body.lower()
    assert "substrate" in low, "the framing must be named, not merely implied"
    assert "decision principle stays general" in low
    assert "preserve information before resorting to destructive recovery" in low, (
        "the general principle A5/A7 both instantiate must be stated explicitly"
    )
    # And the per-substrate instantiations, which are what make the principle checkable.
    for backend in ("vector", "rec_sum", "kv"):
        assert backend in body, f"{backend}'s instantiation of the principle must be shown"


def test_the_generalizability_doc_warns_against_porting_the_ACTION():
    """The expensive bug this framing prevents: porting an action across substrates."""
    body = (REPO / "docs" / "GENERALIZABILITY.md").read_text()
    low = body.lower()
    assert "do not port the action" in low
    assert "kv" in low and "vector" in low
    assert "worse than doing nothing" in low, (
        "the measured consequence is what makes the warning credible rather than stylistic"
    )
    # The specialization must be framed as EXPECTED, or it reads as a limitation being hidden.
    assert "expected rather than worrying" in low or "expected" in low
    assert "substrate-specific bottleneck" in low


def test_the_specialization_is_tied_back_to_exposure():
    """The framing has a cost, and the cost is stated: a substrate anchor buys one backend."""
    body = (REPO / "docs" / "GENERALIZABILITY.md").read_text()
    assert "7.34" in body and "2.31" in body, (
        "the exposure dilution is the concrete consequence of substrate-specific anchors"
    )


def test_the_principle_is_falsifiable_and_a6_is_the_counterexample():
    """A principle every anchor satisfies by construction is decoration, not a claim.

    A6 is what gives it a discriminating edge: same capacity pressure, but on a saturated store where
    no information-preserving action exists. Both places that state the principle must say so, or a
    reader is entitled to ask what it rules out.
    """
    for rel in ("README.md", "docs/GENERALIZABILITY.md"):
        low = (REPO / rel).read_text().lower()
        assert "falsifiable" in low, f"{rel} must say the principle rules something out"
        assert "saturated" in low, f"{rel} must name the condition under which it fails"
        assert "fall back" in low, (
            f"{rel} must state that declining is the correct branch when no preserving action exists"
        )


# ---------------------------------------------------------------------------------------------
# A8 -- the best-validated mechanism and the narrowest support in the stack
# ---------------------------------------------------------------------------------------------

def test_a8_is_a_suppression_at_the_commitment_gate():
    """It PREVENTS a destructive action, so post-execution would be useless by construction."""
    from rounds.anchors import A8

    assert A8.action is Action.SUPPRESS
    assert A8.incision_point is IncisionPoint.POST_GENERATION_PRE_EXEC
    assert A8.incision_point.can_prevent_execution, "the whole value is stopping the clear"
    assert A8.kind == "commitment_gate"


def test_a8_mechanism_is_validated_on_every_single_firing():
    from rounds.anchors import A8_EVIDENCE as ev

    assert ev["firings"] == 15
    assert "2 on every firing" in ev["copies_before"]
    assert "1 on every firing" in ev["copies_remaining"]
    assert "LOSSLESS" in ev["copies_remaining"]
    assert ev["invariant_violations"] == 0
    assert ev["retry_landed"] == "15/15"
    assert ev["byte_reproducible_across_runs"] is True
    assert ev["exposure_matched_prediction"] is True, (
        "the firings landed in exactly the diagnosed episodes -- that is what makes it attribution"
    )


def test_a8_safety_shows_clears_fell_and_steps_did_not_explode():
    """A suppressed action the model re-proposes forever is a step explosion. Check the median."""
    from rounds.anchors import A8_EVIDENCE as ev

    ctl_clears, arm_clears = ev["clears_executed"]
    assert arm_clears < ctl_clears, "fewer destructive clears must actually execute"
    assert ev["clears_suppressed"] == 15
    assert ev["force_quit"] == (0, 0)
    assert ev["median_prereq_steps"][0] == ev["median_prereq_steps"][1], "median must be unchanged"


def test_a8_records_that_its_support_is_one_cell_and_dev_is_vacuous():
    """The two limits that bound what may be claimed. Both must survive an edit pass."""
    from rounds.anchors import A8_EVIDENCE as ev

    caveats = " ".join(ev["caveats"]).lower()
    assert "one cell" in caveats and "n=1 cell" in caveats.replace(" = ", "=")
    assert "vacuously" in caveats
    assert "zero independent evidence" in caveats
    assert "exhaustible" in caveats, "dedup running out is what motivates the next layer"
    # Dev is a no-op, so the dev column must not move.
    assert ev["dev_delta_pp"] == 0.00
    assert ev["dev_flips"] == "0 gains / 0 losses"


def test_a8_is_not_a5_retuned_and_the_record_says_why():
    """'You already have an eviction anchor' is the obvious objection; it was checked."""
    from rounds.anchors import A8

    notes = A8.notes
    assert "NOT A5 TUNING" in notes.upper() or "not A5 tuning" in notes
    assert "post-exec" in notes and "pre-exec" in notes, "the decision points differ"
    assert "VECTOR" in notes.upper(), "A5's trigger matches the vector string; these events are kv"


def test_a8_declining_is_a_first_class_branch():
    """Where no duplicate exists, letting the destructive call proceed is CORRECT."""
    from rounds.anchors import A8

    assert "FALL THROUGH UNCHANGED" in str(A8.params["otherwise"])


# ---------------------------------------------------------------------------------------------
# The capacity ladder
# ---------------------------------------------------------------------------------------------

def test_the_ladder_is_ordered_and_layer_1_is_the_accepted_one():
    from rounds.anchors import CAPACITY_LADDER as ladder

    numbers = [n for n, _, _, _ in ladder["layers"]]
    assert numbers == [1, 2, 3], "the ladder is an ORDERING, so the order is the content"
    assert "ACCEPTED as A8" in ladder["layers"][0][3]
    assert "NEXT" in ladder["layers"][1][3]
    assert "destructive" in ladder["layers"][2][1]


def test_the_ladder_was_confirmed_from_live_state_not_replay():
    """Measured from the destructive-call audit; a replay produced impossible readings."""
    from rounds.anchors import CAPACITY_LADDER as ladder

    assert "live" in ladder["confirmed_by"]
    assert "5 of 5" in ladder["remaining_residual"]
    assert "NO duplicate" in ladder["remaining_residual"]


def test_the_ladder_principle_matches_the_substrate_framing():
    """The ladder is the capacity principle made operational, so the wording must agree."""
    from rounds.anchors import CAPACITY_LADDER as ladder

    principle = ladder["principle"].lower()
    assert "preserve information" in principle
    assert "destructive" in principle
    # And the same sentence must appear in the docs that state the framing.
    for rel in ("README.md", "docs/GENERALIZABILITY.md"):
        body = (REPO / rel).read_text().lower()
        assert "preserve information before resorting to destructive recovery" in body, (
            f"{rel} must state the principle the ladder operationalises"
        )


def test_layer_1_alone_was_deliberate():
    """Bundling consolidation in would have made the two indistinguishable."""
    from rounds.anchors import CAPACITY_LADDER as ladder

    assert "attributable in isolation" in ladder["why_layer_1_alone"]


# ---------------------------------------------------------------------------------------------
# The native baseline, measured
# ---------------------------------------------------------------------------------------------

def test_the_native_baseline_is_measured_and_the_correction_is_recorded():
    from rounds.anchors import CUMULATIVE
    from rounds.anchors import NATIVE_BASELINE as nb

    assert nb["measured"] is True
    assert nb["train_correct"] == 91 and nb["previously_quoted_train"] == 88
    assert nb["dev_reproduced_exactly"] is True, "dev matched exactly; only train moved"
    assert nb["train_acc"] == CUMULATIVE["train_from"], (
        "the progression must start from the MEASURED baseline"
    )
    assert "not byte-reproducible" in nb["why_it_moved"]


def test_native_is_not_literally_zero_intervention_and_says_so():
    """One gate has a code-level default. It confounds nothing, but the word 'native' overstates."""
    from rounds.anchors import NATIVE_BASELINE as nb

    caveat = nb["caveat"].lower()          # the claim is pinned, not the capitalisation
    assert "code-level default" in caveat
    assert "confounds no comparison" in caveat
    assert "not literally zero-intervention" in caveat


def test_the_progression_is_monotone_on_both_splits():
    """The property the acceptance rule is designed to produce, achieved without a zero-loss bar."""
    from rounds.anchors import CUMULATIVE

    assert CUMULATIVE["monotone_on_both_splits"] is True
    train = [r.train_acc for r in PROGRESSION]
    assert train == sorted(train), "train must never go down"
    dev = [r.dev_acc for r in PROGRESSION if r.dev_acc is not None]
    assert dev == sorted(dev), "dev must never go down either"


def test_the_a8_numeral_collision_is_recorded():
    """The working repo's 'N0-A8' is a DIFFERENT anchor from this repo's A8.

    Same numeral, different gate keys, and the other one's control is this repo's accepted stack. This
    is the third instance of the same class of error the traps list exists for, so it is written down
    before it can be quoted wrongly rather than after.
    """
    joined = " ".join(f"{a} {b}" for a, b in NAMING_TRAPS)
    assert "on_dedup_clear_recovery" in joined
    assert "on_low_similarity_cross_container" in joined
    assert "Read the gate key, not the numeral" in joined


# ---------------------------------------------------------------------------------------------
# The per-backend split, measured
# ---------------------------------------------------------------------------------------------

def test_the_per_backend_cells_sum_to_the_published_totals_on_both_splits():
    """Three cells that each look right but do not add up would mean one is stale."""
    from rounds.anchors import CUMULATIVE
    from rounds.anchors import PER_BACKEND as pb

    assert pb["measured"] is True and pb["all_rc_zero"] is True

    train_c = sum(c for c, _ in pb["train"].values())
    train_n = sum(n for _, n in pb["train"].values())
    assert (train_c, train_n) == tuple(CUMULATIVE["measured_endpoint"]["train"])

    dev_c = sum(c for c, _ in pb["dev"].values())
    dev_n = sum(n for _, n in pb["dev"].values())
    assert (dev_c, dev_n) == tuple(CUMULATIVE["measured_endpoint"]["dev"])


def test_the_measurement_matched_the_derivation_it_replaced():
    """Replacing a derivation with a measurement that AGREES is the strongest of the outcomes."""
    from rounds.anchors import PER_BACKEND as pb

    assert pb["matched_the_derivation"] is True


def test_the_cross_check_that_disagreed_is_kept_not_deleted():
    """A cross-check only ever agreeing teaches nothing about whether it would catch anything.

    This one was wrong by one case on two backends while summing correctly, which is precisely the
    failure a total-only check cannot see. Keeping the record is the point.
    """
    from rounds.anchors import PER_BACKEND as pb

    wrong = pb["derivation_walk_was_wrong_by"]
    assert set(wrong) == {"kv", "rec_sum"}
    assert sum(wrong.values()) == 0, (
        "two compensating one-case errors -- which is why the totals agreed and the cells did not"
    )
    assert "harness repair" in pb["why_the_walk_drifts"]


def test_measuring_the_cells_did_not_upgrade_what_they_can_support():
    """A measured cell is still one cell. The exposure caveat must survive the upgrade."""
    from rounds.anchors import PER_BACKEND as pb

    caveat = pb["caveat"].lower()
    assert "descriptive" in caveat
    assert "cannot reach significance" in caveat
    assert "unevenly" in caveat


def test_the_native_control_is_measured_per_backend_and_sums_to_the_baseline():
    """Both columns of the per-backend table are now measured, from shipped artifacts."""
    from rounds.anchors import NATIVE_BASELINE as nb

    result = REPO / "rounds" / "T1_A1_capacity" / "result"
    total = 0
    for backend, expected in nb["per_backend"].items():
        path = result / f"native_{backend}_train.json"
        assert path.exists(), f"missing {path.name}"
        rows = json.loads(path.read_text())
        rows = rows["results"] if isinstance(rows, dict) else rows
        queries = [r for r in rows if not r.get("is_prereq")]
        correct = sum(1 for r in queries if r.get("valid"))
        assert (correct, len(queries)) == expected, f"{backend} disagrees with NATIVE_BASELINE"
        total += correct
    assert total == nb["train_correct"] == 91


def test_the_baseline_drift_is_localized_and_the_shape_is_recorded():
    """The +3 was known; WHERE it lives was not, and the net was hiding structure.

    4 gained and 1 lost, all five in one scenario cell. "Three cases drifted" and "one cell was
    re-scored" are different findings, and only the case-level view separates them -- the same reason
    the acceptance rule demands paired counts beside every delta.
    """
    from rounds.anchors import NATIVE_BASELINE as nb

    where = nb["where_the_drift_lives"].lower()
    assert "entirely in kv" in where
    assert "reproduce the shipped artifact exactly" in where

    cases = nb["drift_cases"]
    assert len(cases["gained"]) == 4 and len(cases["lost"]) == 1
    assert len(cases["gained"]) - len(cases["lost"]) == 3, "must net to the recorded +3"
    assert all("healthcare" in c for c in cases["gained"] + cases["lost"]), (
        "all five changes are in the same scenario cell -- that is the finding"
    )


def test_both_per_backend_columns_give_the_published_cumulative():
    """control -> stack, per backend, must reproduce the headline delta."""
    from rounds.anchors import CUMULATIVE, NATIVE_BASELINE, PER_BACKEND

    ctl = sum(c for c, _ in NATIVE_BASELINE["per_backend"].values())
    stack = sum(c for c, _ in PER_BACKEND["train"].values())
    n = sum(n for _, n in PER_BACKEND["train"].values())
    assert ctl == 91 and stack == 144 and n == 303
    # Control -> T8, the measured chain. A9's contribution is a vector-shard measurement and is
    # asserted separately in test_a9_is_recorded_as_a_shard_measurement_not_a_corpus_one.
    _m = CUMULATIVE["measured_endpoint"]
    assert round(100 * (stack - ctl) / n, 2) == pytest.approx(
        _m["train_acc"] - CUMULATIVE["train_from"], abs=0.01)
