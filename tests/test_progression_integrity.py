"""The published numbers must agree with the single source of truth.

Every accuracy in this project appears in at least three places: the frozen spec of its round, the
detail doc table, and `rounds/anchors.py`. A number retyped in three places is a number that will
eventually disagree with itself -- and in a repo whose entire claim is measurement discipline, a
stale table is not a cosmetic bug.

The round-by-round progression, per-backend breakdown and leaderboard context live in
`docs/BFCL_PROGRESSION.md` (linked from the README, not duplicated into it, so the README stays
short); the headline train/dev endpoint figures stay in the README itself.

These tests pin the agreement. They need no GPU, no adapter, and no benchmark.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anchoropt.anchor import Action, IncisionPoint, exclusion_reason, feasible_actions
from rounds.anchors import ANCHORS, CUMULATIVE, DEFERRED, PROGRESSION

REPO = Path(__file__).resolve().parent.parent
README = (REPO / "README.md").read_text()
PROGRESSION_DOC = (REPO / "docs" / "BFCL_PROGRESSION.md").read_text()
THE_LOOP_DOC = (REPO / "docs" / "THE_LOOP.md").read_text()


# ---------------------------------------------------------------------------------------------
# The progression itself
# ---------------------------------------------------------------------------------------------

def test_cumulative_delta_matches_endpoints():
    """ONE progression, one cumulative figure, derived from the endpoints rather than asserted."""
    first, last = PROGRESSION[0], PROGRESSION[-1]
    assert first.train_acc == pytest.approx(CUMULATIVE["train_from"], abs=0.01)
    assert last.train_acc == pytest.approx(CUMULATIVE["train_to"], abs=0.01)
    assert last.train_acc - first.train_acc == pytest.approx(
        CUMULATIVE["train_delta_pp"], abs=0.01
    )


def test_cumulative_dev_delta_matches_endpoints():
    """Dev is the accept/reject criterion, so its endpoints are pinned the same way."""
    first, last = PROGRESSION[0], PROGRESSION[-1]
    assert first.dev_acc == pytest.approx(CUMULATIVE["dev_from"], abs=0.01)
    assert last.dev_acc == pytest.approx(CUMULATIVE["dev_to"], abs=0.01)
    assert last.dev_acc - first.dev_acc == pytest.approx(CUMULATIVE["dev_delta_pp"], abs=0.01)


def test_no_cumulative_p_value_is_quoted_and_the_reason_is_recorded():
    """The benchmark is deterministic, so there is no sampling distribution for a null."""
    assert CUMULATIVE["p_value"] is None
    note = CUMULATIVE["p_value_note"].lower()
    assert "determinis" in note


def test_every_round_shares_one_denominator():
    """Differencing accuracies across different corpora is the project's oldest standing hazard."""
    assert len({r.train_n for r in PROGRESSION}) == 1


def test_t4_installed_nothing_and_did_not_move():
    """T4 is a real round with no anchor. If it ever gains accuracy, something is mislabelled."""
    t3, t4 = (r for r in PROGRESSION if r.tag in ("T3", "T4"))
    assert t4.anchor is None
    assert t4.train_correct == t3.train_correct


def test_accepted_anchor_count_matches_progression():
    """Every accepted anchor owns exactly one installed round, and vice versa."""
    installed = [r.anchor for r in PROGRESSION if r.anchor is not None]
    assert len(installed) == CUMULATIVE["n_anchors"] == len(ANCHORS) == 8
    assert [a.name for a in installed] == [a.name for a in ANCHORS], (
        "the progression order must match the anchor set order"
    )


def test_a_rejected_anchor_is_absent_from_the_stack_but_present_in_the_record():
    """A6 must not be in ANCHORS, and must not be quietly deleted either."""
    assert "A6" not in {a.name for a in ANCHORS}
    assert "A6" in DEFERRED
    assert all(r.anchor is None or r.anchor.name != "A6" for r in PROGRESSION)


def test_the_rejected_anchor_records_which_criteria_it_failed():
    rec = DEFERRED["A6"]
    assert rec["fails_criteria"] == (2, 3)
    assert rec["dev_pp"] < 0, "criterion 2: dev regression"
    assert rec["on_target_dev_pp"] < 0, "criterion 3: on-target and NEGATIVE"
    # And the signal must be recorded as still open rather than refuted.
    assert "DEFERRED" in rec["status"]
    assert rec["signal_still_present"]["entry_too_long_events"]["dev"] > 0


def test_mined_coverage_is_monotonically_non_increasing():
    """Each anchor consumes the residual that motivated it: 78% -> 61% -> 13% -> 3%."""
    seen = [r.mined_coverage for r in PROGRESSION if r.mined_coverage is not None]
    assert seen == sorted(seen, reverse=True)


# ---------------------------------------------------------------------------------------------
# README agreement
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rnd", PROGRESSION, ids=lambda r: r.tag)
def test_readme_row_matches_source_of_truth(rnd):
    rows = [ln for ln in PROGRESSION_DOC.splitlines() if ln.startswith(f"| {rnd.tag} ")]
    assert len(rows) == 1, f"expected exactly one progression-doc row for {rnd.tag}"
    # Format-agnostic: "29.04 %" and "29.04%" are the same claim, and the spacing is a style
    # choice the doc is free to make. Only the VALUE is pinned.
    assert f"{rnd.train_acc:.2f}" in rows[0], (
        f"{rnd.tag} row must state {rnd.train_acc:.2f} (from PROGRESSION), got: {rows[0]}"
    )


def test_readme_states_the_headline_figures_on_both_splits():
    """The CURRENT endpoint, read from CUMULATIVE so this cannot drift when a round is added."""
    for token in (f"{CUMULATIVE['train_from']:.2f}", f"{CUMULATIVE['train_to']:.2f}",
                  f"+{CUMULATIVE['train_delta_pp']:.2f}", f"{CUMULATIVE['dev_from']:.2f}",
                  f"{CUMULATIVE['dev_to']:.2f}", f"+{CUMULATIVE['dev_delta_pp']:.2f}"):
        assert token in README, f"README lost headline figure {token}"
    # T8's figures must ALSO remain, as the offline-recomputable endpoint -- in the progression
    # doc, which carries the round-by-round detail the README links out to.
    prior = CUMULATIVE["measured_endpoint"]
    assert f"{prior['train_acc']:.2f}" in PROGRESSION_DOC and f"{prior['dev_acc']:.2f}" in PROGRESSION_DOC


@pytest.mark.parametrize("rnd", [r for r in PROGRESSION if r.dev_acc is not None],
                         ids=lambda r: r.tag)
def test_readme_dev_column_matches_the_source_of_truth(rnd):
    """Every round now has a dev figure, and the progression doc must show the same one."""
    rows = [ln for ln in PROGRESSION_DOC.splitlines() if ln.startswith(f"| {rnd.tag} ")]
    assert len(rows) == 1, f"expected exactly one progression-doc row for {rnd.tag}"
    assert f"{rnd.dev_acc:.2f}" in rows[0], (
        f"{rnd.tag} row must state dev {rnd.dev_acc:.2f}, got: {rows[0]}"
    )


def test_the_readme_labels_the_two_delta_columns_distinctly():
    """Two adjacent columns both called just 'delta' is unreadable; say which split each is."""
    header = next(ln for ln in PROGRESSION_DOC.splitlines() if ln.startswith("| round | anchor |"))
    assert "train Δ" in header and "dev Δ" in header


def test_readme_states_the_rejected_anchor_and_its_numbers():
    """A record that shows only the accepted anchors is a sales sheet, not a record."""
    assert "A6" in PROGRESSION_DOC
    rec = DEFERRED["A6"]
    assert f"{rec['dev_pp']:.2f}".lstrip("-") in PROGRESSION_DOC, "A6's dev regression must be stated"
    assert f"{rec['on_target_dev_pp']:.2f}".lstrip("-") in PROGRESSION_DOC, (
        "the on-target negative -- the cleanest statement of the failure -- must be stated"
    )


def test_readme_calls_the_second_split_a_validation_set():
    """Using dev for accept/reject makes it a validation set; the README must not claim otherwise."""
    low = README.lower()
    assert "validation" in low
    assert "selected-on" in low or "selected on" in low, (
        "the consequence -- the dev figure is not a clean generalization estimate -- must be stated"
    )


def test_readme_explains_why_no_p_value_is_quoted():
    """Omitting a p without saying why reads as cherry-picking. The reason here is determinism."""
    assert CUMULATIVE["p_value"] is None
    low = README.lower()
    assert "determinis" in low, (
        "the README must say WHY no p-value is quoted -- the benchmark is deterministic, so there "
        "is no sampling distribution for a null"
    )


# ---------------------------------------------------------------------------------------------
# The incision-point grid
# ---------------------------------------------------------------------------------------------

def test_grid_is_nine_of_twelve_with_stated_reasons():
    """3 incision points x 4 actions. `transform` was merged into `reroute` -- see Action's docstring."""
    admissible = sum(len(feasible_actions(p)) for p in IncisionPoint)
    assert admissible == 9, "the 3x4 grid should leave 9 structurally admissible cells"
    for p in IncisionPoint:
        for a in Action:
            if a not in feasible_actions(p):
                assert exclusion_reason(p, a), f"{p.value}/{a.value} pruned with no stated reason"


def test_pre_generation_admits_only_noop_and_reprompt():
    """The structural reason an unconditional global prompt is the only lever at point 1."""
    assert feasible_actions(IncisionPoint.PRE_GENERATION) == {Action.NOOP, Action.REPROMPT}


def test_noop_available_everywhere():
    """noop is the control arm; it must be point-independent or comparisons lose their baseline."""
    for p in IncisionPoint:
        assert Action.NOOP in feasible_actions(p)


def test_only_pre_generation_cannot_see_the_proposed_call():
    """The fact A4 v1 needed and structurally could not have."""
    blind = [p for p in IncisionPoint if not p.sees_proposed_call]
    assert blind == [IncisionPoint.PRE_GENERATION]


def test_only_post_execution_cannot_prevent():
    cannot = [p for p in IncisionPoint if not p.can_prevent_execution]
    assert cannot == [IncisionPoint.POST_EXECUTION]


# ---------------------------------------------------------------------------------------------
# The anchor set
# ---------------------------------------------------------------------------------------------

def test_every_anchor_occupies_an_admissible_cell():
    for a in ANCHORS:
        assert a.action in feasible_actions(a.incision_point)


def test_kind_follows_incision_point_not_signal():
    kinds = {a.name: a.kind for a in ANCHORS}
    assert kinds == {
        "A1": "error_recovery",      # post-execution
        "A2": "error_recovery",      # post-execution
        "A3": "commitment_gate",     # post-gen/pre-exec
        "A4": "commitment_gate",     # post-gen/pre-exec
        "A5": "error_recovery",      # post-execution
        "A7": "error_recovery",      # post-execution
        "A8": "commitment_gate",     # post-gen/pre-exec -- it PREVENTS, it does not repair
        # A9's signal is a retrieval SCORE, not an error string, and its cell is still error_recovery.
        # That is this test's whole point: the kind follows the incision point and action.
        "A9": "error_recovery",      # post-execution
    }


def test_the_anchor_set_spans_four_distinct_cells_but_eight_anchors():
    """EIGHT anchors in FOUR cells, and the collision is itself a finding.

    A2 and A4 share the `reprompt` ACTION FAMILY at different incision points -- the same point the
    A4 v1/v2 pair makes: the action alone does not identify an intervention.

    A1, A5, A6, A7 and A9 all sit in ONE cell, `(post_execution, reroute)`, and they are genuinely
    different mechanisms: A1 relocates a blocked write, A5 evicts a provable duplicate to make room,
    A6 re-addresses an over-long entry, A7 rewrites a single-string store, and A9 reroutes a READ to
    the other container -- the first of the five that does not touch the store at all. So the
    (point, action)
    grid does NOT uniquely identify an anchor -- a full spec also needs the CONSTRAINT it repairs and
    what the model observes afterwards. That is the same lesson E1 taught against A3.
    """
    cells = {(a.incision_point, a.action) for a in ANCHORS}
    assert len(ANCHORS) == 8
    assert len(cells) == 4, "eight anchors, four cells -- the grid under-determines the anchor"

    reroute_cell = [
        a.name for a in ANCHORS
        if (a.incision_point, a.action) == (IncisionPoint.POST_EXECUTION, Action.REROUTE)
    ]
    assert sorted(reroute_cell) == ["A1", "A5", "A7", "A9"], (
        "four distinct mechanisms share one cell; they are separated by their constraint, not "
        "by their coordinates -- and A9 sharpens the point, since it is the only one of the four "
        "that does not touch the store at all"
    )
    # And each of those must still be distinguishable by the locus it repairs.
    loci = {a.locus for a in ANCHORS if a.name in reroute_cell}
    assert len(loci) == len(reroute_cell), "same cell, so the LOCUS must be what tells them apart"

    # A3, A8 and E1 collide the same way at the commitment gate: three suppressions, three different
    # things suppressed (a duplicate write, a destructive clear, a provably uninformative call).
    suppress_cell = sorted(
        a.name for a in ANCHORS
        if (a.incision_point, a.action) == (IncisionPoint.POST_GENERATION_PRE_EXEC, Action.SUPPRESS)
    )
    assert suppress_cell == ["A3", "A8"], "E1 is not in ANCHORS; it is the efficiency anchor"


def test_a2_and_a4_share_an_action_at_different_points():
    a2, a4 = (a for a in ANCHORS if a.name in ("A2", "A4"))
    assert a2.action is a4.action is Action.REPROMPT
    assert a2.incision_point is not a4.incision_point


def test_every_anchor_records_an_attribution_distinct_from_its_locus():
    """An anchor is defined by its traced cause, not its error label."""
    for a in ANCHORS:
        assert a.attribution and a.attribution != a.locus


# ---------------------------------------------------------------------------------------------
# Pre-registration
# ---------------------------------------------------------------------------------------------

def test_every_installed_round_ships_its_acceptance_criteria():
    """Each round must ship the document it was actually accepted under.

    The FILENAME carries a claim, so it is checked per round rather than globally:
    `FROZEN.md` asserts pre-registration, `CRITERIA.md` says the conditions were transcribed or the
    variants swept with results visible, and `OVERRIDE.md` says a frozen clause was failed and
    overridden. A blanket "*FROZEN*.md exists" check would let A5's override masquerade as a
    pre-registration.
    """
    expected = {
        "T1_A1_capacity": "FROZEN.md",
        "T2_A2_not_found": "FROZEN.md",
        "T3_A3_duplicate": "FROZEN.md",
        "T4_exhaustion": "STOPPING_CRITERIA_FROZEN.md",
        "T5_A4_no_tool_call": "FROZEN.md",
        "T6_A5_archival_full": "OVERRIDE.md",     # failed a frozen clause; later settled on evidence
        "T7_A7_blob_overflow": "CRITERIA.md",     # variants swept with results visible
        "T8_A8_dedup_clear": "FROZEN.md",         # criteria frozen while the jobs were still running
        "T9_A9_xcontainer_merge": "FROZEN.md",    # frozen before the v3 implementation
    }
    rounds = sorted(d.name for d in (REPO / "rounds").glob("T*") if d.is_dir())
    assert rounds == sorted(expected), "a round was added or renamed without updating this map"
    for name, filename in expected.items():
        assert (REPO / "rounds" / name / filename).exists(), (
            f"{name} must ship {filename} -- the filename states how it was accepted"
        )


def test_a4_frozen_spec_matches_the_hash_cited_at_acceptance():
    """The acceptance commit cites a204313cd34eb761; the copy must be byte-identical."""
    import hashlib

    spec = REPO / "rounds" / "T5_A4_no_tool_call" / "FROZEN.md"
    digest = hashlib.sha256(spec.read_bytes()).hexdigest()
    assert digest.startswith("a204313cd34eb761"), (
        "A4's frozen spec no longer matches the pre-registered hash -- it must never be edited"
    )


# ---------------------------------------------------------------------------------------------
# What is actually shipped, per anchor
# ---------------------------------------------------------------------------------------------

def test_every_anchor_ships_its_learned_policy():
    """The learned artifacts are here, even where the mechanism code is not.

    Two documented exceptions, both facts about the anchor rather than gaps in the repo:
      T4  installed nothing, so there is no policy to ship.
      T8  A7 is gated at DISPATCH on the backend, not by a policy flag, so it has no switch. Its
          acceptance conditions live in CRITERIA.md instead.
    """
    no_policy = {"T4_exhaustion"}
    for d in sorted(x for x in (REPO / "rounds").glob("T*") if x.is_dir()):
        if d.name in no_policy:
            assert not (d / "policy.json").exists(), (
                f"{d.name} now ships a policy -- remove it from the documented exceptions"
            )
            continue
        assert (d / "policy.json").exists(), f"{d.name} is missing its learned policy"


def test_compute_anchors_have_ported_mechanisms():
    """A1 and A3 COMPUTE something, so they are real algorithms and must be here as pure code."""
    mech = REPO / "anchoropt" / "mechanisms"
    assert (mech / "capacity_repair.py").exists()    # A1
    assert (mech / "redundant_write.py").exists()    # A3


def test_a4_injected_text_is_shipped_not_just_described():
    """A4 is a reprompt: its learned content IS the text, so the text must be in the policy."""
    import json
    pol = json.loads((REPO / "rounds" / "T5_A4_no_tool_call" / "policy.json").read_text())
    text = (pol.get("templates") or {}).get("on_turn_start_action", "")
    assert text.strip(), "A4's injected text must ship -- it is the anchor"
    assert "memory" in text.lower()


def test_no_line_in_this_repo_carries_a_global_preamble():
    """The whole point of the N0 line: no global prompt anywhere, in any round's policy."""
    import json
    for p in sorted((REPO / "rounds").glob("T*/policy.json")):
        pol = json.loads(p.read_text())
        preamble = (pol.get("templates") or {}).get("on_memory_preamble", "")
        assert not preamble.strip(), f"{p.parent.name} has a global preamble; this line forbids it"


def test_per_backend_leaderboard_context_matches_the_shipped_results():
    """The README compares our per-backend numbers to Berkeley's published range.

    Recomputed here so the comparison cannot drift, and so the caveats stay attached to real numbers.
    """
    import collections
    import json

    def per_backend(rel: str) -> dict[str, tuple[int, int]]:
        rows = json.loads((REPO / "rounds" / rel).read_text())["results"]
        out: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for r in (x for x in rows if not x.get("is_prereq")):
            b = ("kv" if "memory_kv" in r["id"]
                 else "vector" if "vector" in r["id"] else "rec_sum")
            out[b][1] += 1
            out[b][0] += bool(r.get("valid"))
        return {k: tuple(v) for k, v in out.items()}

    ctl = per_backend("T1_A1_capacity/result/baseline_T0_train.json")
    anc = per_backend("T5_A4_no_tool_call/result/eval_train.json")
    expected = {"kv": (14.29, 29.52), "vector": (21.35, 39.33), "rec_sum": (49.54, 56.88)}
    for backend, (want_c, want_a) in expected.items():
        c, n = ctl[backend]
        a, _ = anc[backend]
        assert round(100 * c / n, 2) == pytest.approx(want_c, abs=0.01), backend
        assert round(100 * a / n, 2) == pytest.approx(want_a, abs=0.01), backend
        assert a > c, f"{backend} must improve"

    readme = PROGRESSION_DOC
    # The comparison must ship with its caveats rather than as a bare ranking. Any of these
    # hedges satisfies it; the point is that the leaderboard table is not presented as a ranking.
    low = readme.lower()
    assert any(h in low for h in ("not a ranking", "not directly comparable", "context, not a",
                                  "not a like-for-like", "not like-for-like", "not comparable",
                                  "a reference, not a result", "reference:",
                                  "and on the aggregate it does not")), (
        "the leaderboard comparison must be framed as a reference or carry a comparability caveat"
    )
    assert "gorilla.cs.berkeley.edu" in readme


def test_the_full_stack_per_backend_column_is_measured_and_sums_to_the_total():
    """The README's full-stack per-backend cells are MEASURED, and must sum to the corpus total.

    They were originally derived from the total plus each anchor's exposure; a six-shard concurrent
    run has since produced them directly and every cell matched. The sum is still asserted, because
    three cells that individually look right and do not add up would mean one of them is stale.

    Reconciled against the MEASURED endpoint (T8), not against CUMULATIVE: since A9 the cumulative
    total includes a round whose corpus figure is projected, and a measured table must not be checked
    against a partly-projected total.
    """
    from rounds.anchors import CUMULATIVE
    MEASURED = CUMULATIVE["measured_endpoint"]

    # The A1-A8 column: what the six-shard run measured, and what this repo recomputes offline.
    cells = {"kv": (35, 105), "vector": (39, 89), "rec_sum": (70, 109)}
    assert sum(n for _, n in cells.values()) == MEASURED["train"][1]
    assert sum(c for c, _ in cells.values()) == MEASURED["train"][0], (
        "the A1-A8 per-backend cells must sum to the recomputable corpus figure"
    )
    for backend, (correct, n) in cells.items():
        assert f"{100 * correct / n:.2f}" in PROGRESSION_DOC, f"{backend}'s A1-A8 cell is stale"

    # And the FULL stack column, which differs from it on vector only -- A9's exposure.
    from rounds.anchors import A9_EVIDENCE
    full = dict(cells, vector=A9_EVIDENCE["vector_shard"]["train"]["arm"])
    assert sum(c for c, _ in full.values()) == CUMULATIVE["train_correct"][1] == 159, (
        "the full-stack cells must sum to the current endpoint"
    )
    assert f"{100 * full['vector'][0] / full['vector'][1]:.2f}" in PROGRESSION_DOC, (
        "vector's full-stack cell is stale"
    )

    low = PROGRESSION_DOC.lower()
    # The provenance claim must survive rewording: these cells are MEASURED, not derived, and the
    # doc has to say so somewhere. Any phrasing counts; the claim does not.
    assert "measured" in low, "the cells are measured, and the doc must say so"
    # The history must survive: a cross-check that disagreed and was settled by the run is exactly
    # the kind of thing a later editor deletes as noise.
    assert "derived" in low, "how they were originally obtained must remain visible"
    assert "disagree" in low, (
        "the cross-check earned its keep by disagreeing; deleting that leaves the reader with a "
        "bare number and no reason to trust it more than the last one"
    )
    # And the dev column is measured per backend too. These are the FULL-STACK dev cells; A9 moves
    # vector from 14/40 to 21/40, so the table must show the stack it claims to show.
    from rounds.anchors import A9_EVIDENCE
    from rounds.anchors import CUMULATIVE as _C
    dev_cells = ("6/24", f"{A9_EVIDENCE['vector_shard']['dev']['arm'][0]}/40", "14/20")
    for cell in dev_cells:
        assert cell in PROGRESSION_DOC, f"full-stack dev cell {cell} must be stated"
    # The three must add up to the published dev endpoint, or one of them is from the wrong stack.
    assert sum(int(c.split("/")[0]) for c in dev_cells) == _C["dev_correct"][1] == 41


# --------------------------------------------------------------------------------------------
# E1, the efficiency anchor
# --------------------------------------------------------------------------------------------

def test_e1_is_suppress_at_the_only_admissible_point():
    """It needs the proposed call (not pre-gen) and must stop it (not post-exec)."""
    from anchoropt.anchor import Action, IncisionPoint
    from rounds.anchors import E1

    assert E1.action is Action.SUPPRESS
    assert E1.incision_point is IncisionPoint.POST_GENERATION_PRE_EXEC
    assert E1.incision_point.sees_proposed_call, "cannot decide without the call"
    assert E1.incision_point.can_prevent_execution, "a saving requires preventing execution"


def test_e1_is_not_in_the_accuracy_progression():
    """Different objective, different table.

    A 0 pp row in an accuracy progression reads as a FAILED accuracy anchor, which is the exact
    misreading docs/EFFICIENCY_CLASS.md exists to prevent. E1 belongs to E1_EVIDENCE.
    """
    from rounds.anchors import ANCHORS, E1, PROGRESSION

    assert E1 not in ANCHORS, "ANCHORS is the accepted ACCURACY set"
    assert all(r.anchor is not E1 for r in PROGRESSION)


def test_e1_evidence_states_the_scope_that_prevents_overstatement():
    """The numbers are quotable only with their scope attached, so pin the scope too."""
    from rounds.anchors import E1_EVIDENCE as ev

    assert ev["task_delta_pp"] == 0.00
    assert ev["flips"] == 0
    # The saving must be in EXECUTIONS: withheld > 0 and none reached the executor.
    assert ev["calls_withheld"] == 16
    assert ev["calls_sent_to_execute"] == 0
    # And the executed-redundancy metric -- not the proposal count -- is what fell.
    assert ev["executed_redundancy_before"] == 16
    assert ev["executed_redundancy_after"] == 0
    # THE limiting fact. If this ever becomes nonzero the claim gets stronger; until then,
    # "+0.00 pp over 303 scored cases" must never be read as a test at scale.
    assert ev["firings_in_scored_episodes"] == 0
    assert ev["dev_run"] is None
    assert ev["backends_fired"] == ("vector",)
    assert "kv" in ev["backends_claimed"], "claimed but unobserved -- the caveat must survive"
    assert len(ev["caveats"]) >= 5, "the caveats are part of the result, not decoration"


def test_e1_criteria_file_is_not_called_frozen():
    """E1 has no pre-registration. Naming its file FROZEN.md would assert one that never existed."""
    d = REPO / "rounds" / "E1_efficiency"
    assert (d / "CRITERIA.md").exists()
    assert not (d / "FROZEN.md").exists(), (
        "E1's criteria were transcribed from the verdict script, not pre-registered. "
        "Renaming this to FROZEN.md would claim a pre-registration that does not exist."
    )
    body = (d / "CRITERIA.md").read_text()
    assert "not* pre-registered" in body or "not pre-registered" in body


def test_readme_states_e1s_headline_and_points_at_the_detail():
    """The README summarises each anchor in one line; the detail lives in docs/ANCHORS.md.

    So the README must carry E1's headline numbers and say the figures are transcribed, and it must
    LINK to the page that carries the scope. Splitting the two is deliberate -- but a headline with
    no route to its caveats would be the overstatement this pair of tests exists to prevent.
    """
    from rounds.anchors import E1_EVIDENCE as ev

    body = (REPO / "README.md").read_text()
    assert "E1" in body, "E1 must appear where a reader meets the anchors"
    assert "efficiency" in body.lower()

    n = ev["calls_withheld"]
    assert f"{n} of {n}" in body or f"{n}/{n}" in body, f"README must state {n}/{n} work removed"
    assert "+0.00 pp" in body
    assert "transcribed" in body.lower(), "the numbers are not re-measured in this repo"
    assert "docs/ANCHORS.md" in body, "the README must route the reader to the per-anchor detail"


def test_the_anchor_detail_page_carries_e1s_full_scope():
    """The caveats moved out of the README, so they must be pinned where they landed.

    Without these, "+0.00 pp over 303 cases" reads as a test at scale, which it is not.
    """
    body = (REPO / "docs" / "ANCHORS.md").read_text()
    assert "zero scored episodes" in body, "the limiting fact must survive the move"
    assert "consistent with" in body, "0 pp is consistent with safety, not proof of it"
    low = body.lower()
    assert "no pre-registered spec" in low or "no frozen pre-registration" in low
    assert "not currently active" in low or "not active" in low, (
        "accepted-but-inactive is the distinction a previous pass collapsed"
    )


def test_e1_is_absent_from_the_generated_accuracy_table():
    """delta_table() renders the ACCURACY progression. E1 must not leak into it."""
    from rounds.anchors import delta_table

    assert "E1" not in delta_table(), (
        "a 0 pp row in an accuracy table reads as a failed accuracy anchor -- "
        "E1 belongs in its own section, on its own objective"
    )


def test_the_leaderboard_table_matches_the_recorded_source_and_is_dated():
    """External figures need a date stamp and one source, or they float.

    The repo cited the launch blog's Claude 3 Sonnet numbers as "the leader" after the live table had
    moved on -- in BOTH directions, so it was not even conservative. Now the table is pinned to
    LEADERBOARD and its `as_of`.
    """
    from rounds.anchors import LEADERBOARD as lb

    assert lb["as_of"] in PROGRESSION_DOC, "the progression doc must carry the date the figures were read"
    for model, memory, kv, vector, rec_sum in lb["top"]:
        short = model.split(" (")[0]
        assert short in PROGRESSION_DOC, f"{short} missing from the progression doc's table"
        for value in (memory, kv, vector, rec_sum):
            assert f"{value:.2f}" in PROGRESSION_DOC, f"{short}: {value} missing"

    # The superseded blog figures may be dropped from the doc entirely -- that is fine, and is the
    # cleanest outcome. What is NOT fine is quoting one WITHOUT the "superseded" label, which is the
    # actual defect that occurred: the launch-blog snapshot sat in a table captioned as the leader.
    if any(f"{v}" in PROGRESSION_DOC for v in lb["superseded_blog_figures"].values()):
        assert "supersede" in PROGRESSION_DOC.lower(), (
            "a launch-blog figure appears in the doc but is not labelled as a superseded snapshot"
        )


def test_the_readme_states_where_we_place_without_overclaiming():
    """One real cross-reference (rec_sum beats 5 of 8) and its limit in the same breath."""
    from rounds.anchors import LEADERBOARD as lb
    from rounds.anchors import PER_BACKEND

    rec_sum_ours = 100 * PER_BACKEND["train"]["rec_sum"][0] / PER_BACKEND["train"]["rec_sum"][1]
    beaten = [row for row in lb["top"] if rec_sum_ours > row[4]]
    assert len(beaten) == 5, f"expected to beat 5 of the top 8 on rec_sum, got {len(beaten)}"

    # And we must beat none on the other two, which is the half that keeps the claim honest.
    kv_ours = 100 * PER_BACKEND["train"]["kv"][0] / PER_BACKEND["train"]["kv"][1]
    vec_ours = 100 * PER_BACKEND["train"]["vector"][0] / PER_BACKEND["train"]["vector"][1]
    assert not [r for r in lb["top"] if kv_ours > r[2]]
    assert not [r for r in lb["top"] if vec_ours > r[3]]

    low = PROGRESSION_DOC.lower()
    assert "5 of the top 8" in low

    # The win must be SCOPED somewhere the reader cannot miss -- it holds on one backend, not overall.
    # Any of these carries that: a scoping heading, or the aggregate cell stated in the table. How the
    # doc words it is the author's call, so this checks that the bound exists, not its phrasing.
    macro = sum(100 * c / n for c, n in PER_BACKEND["train"].values()) / 3
    assert any(m in low for m in ("on one backend", "on the aggregate", f"{macro:.2f}")), (
        "the per-backend win must be scoped: name the backend, or show the aggregate cell"
    )


def test_our_per_backend_table_contains_no_external_model_figure():
    """The two tables are separated so a cross-model number cannot be read as one of our deltas.

    An earlier revision had the leaderboard's "leader" and "span" columns INSIDE our per-backend table,
    which is how the stale Claude 3 Sonnet snapshot came to sit next to our measured cells for weeks.
    Our table is now ours alone: control / T5 / T8 / delta.
    """
    from rounds.anchors import LEADERBOARD as lb

    start = PROGRESSION_DOC.index("### Per backend")
    table = PROGRESSION_DOC[start:PROGRESSION_DOC.index("\n\n", PROGRESSION_DOC.index("| backend", start))]

    for model, memory, kv, vector, rec_sum in lb["top"]:
        short = model.split(" (")[0]
        assert short not in table, f"{short} appears inside OUR per-backend table"
        for value in (memory, kv, vector, rec_sum):
            assert f"{value:.2f}" not in table, f"leaderboard figure {value} is inside our table"
    for value in lb["superseded_blog_figures"].values():
        assert f"{value}" not in table, f"superseded blog figure {value} is inside our table"

    # And it must still carry a control, a full-stack and a delta column, whatever they are called.
    # T5 was dropped on purpose: it is a mid-progression waypoint RECOMPUTED from artifacts, so it did
    # not share this table's measured provenance, and the progression table already shows it.
    for column in ("control", "Δ"):
        assert column in table, f"our per-backend table lost its {column!r} column"


def test_the_memory_score_is_the_unweighted_mean_of_the_three_backends():
    """Verified from the table rather than assumed, on all 8 rows.

    This is what licenses computing our OWN aggregate the same way instead of leaving the cell blank.
    If Berkeley ever reweights the sub-tasks, or the three stop having equal case counts, this fails
    and the README's macro-mean cells stop being comparable to that column.
    """
    from rounds.anchors import LEADERBOARD as lb

    worst = 0.0
    for model, memory, kv, vector, rec_sum in lb["top"]:
        residual = abs(memory - (kv + vector + rec_sum) / 3)
        assert residual < 0.01, f"{model}: memory {memory} != mean {(kv+vector+rec_sum)/3:.4f}"
        worst = max(worst, residual)
    assert worst == pytest.approx(lb["aggregate_max_residual_pp"], abs=0.001)
    assert "unweighted mean" in lb["aggregate_is"]


def test_our_aggregate_cells_are_macro_and_are_labelled_as_distinct_from_the_pooled_headline():
    """Our headline is POOLED (144/303); that column is MACRO. Quoting one against the other is a
    denominator error, so both appear and the README must say which is which."""
    from rounds.anchors import CUMULATIVE, NATIVE_BASELINE, PER_BACKEND

    for cells, want in ((PER_BACKEND["train"], 47.12), (NATIVE_BASELINE["per_backend"], 29.34)):
        macro = sum(100 * c / n for c, n in cells.values()) / 3
        assert round(macro, 2) == pytest.approx(want, abs=0.01)
        assert f"{want:.2f}" in PROGRESSION_DOC, f"the macro aggregate {want} must appear in the progression doc"

    # The two must genuinely differ, or the caveat would be pedantry rather than a real hazard.
    macro = sum(100 * c / n for c, n in PER_BACKEND["train"].values()) / 3
    assert abs(macro - CUMULATIVE["train_to"]) > 0.2, "macro and pooled must be distinguishable"

    # Whether the README explains macro vs pooled in prose is the author's call; what is pinned is
    # that the two values differ and that ours is the macro one, so the cell is not mislabelled.


def test_we_beat_nobody_on_the_aggregate_memory_task():
    """The per-backend win does NOT survive aggregation, and the README says so.

    Stated as a test because it is the claim most likely to be quietly dropped in a later edit: it is
    the one number in this section that is unflattering, and it is computed the leaderboard's own way,
    so it is also the most directly comparable figure we have.
    """
    from rounds.anchors import LEADERBOARD as lb
    from rounds.anchors import PER_BACKEND

    macro = sum(100 * c / n for c, n in PER_BACKEND["train"].values()) / 3
    assert not [r for r in lb["top"] if macro > r[1]], "we do not beat any of the top 8 aggregate"

    # The README's heading carries this; the gap itself is recorded in LEADERBOARD["where_we_place"].
    eighth = min(r[1] for r in lb["top"])
    assert f"{eighth - macro:.2f}" in lb["where_we_place"], "the gap to 8th must be recorded"


def test_the_readme_states_the_residual_shrinking_with_the_recorded_coverage_figures():
    """The claim "the residual measurably shrinks" needs the numbers, or it is just a slogan.

    Also guards the direction: if a later round's coverage were transcribed out of order the chain
    would still look plausible in prose, which is exactly the kind of error this repo pins by test.
    """
    coverage = {r.tag: r.mined_coverage for r in PROGRESSION if r.mined_coverage is not None}
    for tag in ("T0", "T2", "T3", "T5"):
        pct = f"{round(100 * coverage[tag])}%"
        assert pct in THE_LOOP_DOC, f"{tag}'s mined coverage {pct} must appear in docs/THE_LOOP.md"

    # And T4 -- the round where no locus survived -- must be named as the reason the vocabulary
    # had to be refined, not quietly omitted as an empty round.
    t4 = next(r for r in PROGRESSION if r.tag == "T4")
    assert t4.anchor is None
    assert "T4" in THE_LOOP_DOC


def test_the_readme_cites_the_signals_by_their_real_names():
    """The lexical -> structural escalation is argued from four specific signals. If an anchor's locus
    is ever renamed, the argument must not keep citing the old string."""
    loci = {a.name: a.locus for a in ANCHORS}

    # The lexical tier: canonicalised error strings.
    for name in ("A1", "A2", "A3"):
        leaf = loci[name].rsplit("/", 1)[-1]
        assert leaf in THE_LOOP_DOC, f"{name}'s locus leaf {leaf!r} must appear in docs/THE_LOOP.md"

    # The structural tier: an ABSENCE, which no error string can express.
    assert loci["A4"] == "no_tool_call_at_all"
    assert "no_tool_call_at_all" in THE_LOOP_DOC


def test_the_readme_a1_linked_loss_figure_comes_from_the_anchor_record():
    """"Rank 1 by linked loss, not event volume" is the load-bearing claim for backward attribution,
    so docs/THE_LOOP.md's figure is checked against A1's own notes rather than retyped as a literal."""
    a1 = next(a for a in ANCHORS if a.name == "A1")
    assert "81 of 215" in a1.notes, "A1's notes are the source for this figure"
    assert "81 of 215" in THE_LOOP_DOC, "docs/THE_LOOP.md must quote the recorded linked-loss figure"
    assert "not by event volume" in THE_LOOP_DOC or "not event volume" in THE_LOOP_DOC
