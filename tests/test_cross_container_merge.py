"""A9's mechanism, and the boundary between what was measured and what was projected.

Two things are pinned here, and the second is the unusual one.

MECHANISM. The merge's safety argument is that it is ADDITIVE: a core entry that still ranks in the
top k survives, and ties break toward core. That is what makes a 66%-precision trigger acceptable, so
it is asserted directly rather than trusted from the design.

PROVENANCE. A9's vector shard is measured; its whole-corpus contribution is arithmetic. A projection
that quietly becomes a measurement is the single most likely way this record degrades, so the
distinction is enforced: the projected figure must equal the measured net added to the measured
endpoint, and it must be labelled everywhere it appears.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anchoropt.mechanisms import cross_container_merge as xcm
from rounds.anchors import A9, A9_EVIDENCE, CUMULATIVE, PROGRESSION

README = (REPO / "README.md").read_text()
PROGRESSION_DOC = (REPO / "docs" / "BFCL_PROGRESSION.md").read_text()


def _payload(*triples) -> str:
    """A retrieve result in the harness's own wire shape."""
    body = ", ".join(
        f'{{"id": {i}, "similarity_score": {s}, "text": "{t}"}}' for s, i, t in triples
    )
    return f'{{"result": [{body}]}}'


# ------------------------------------------------------------------------------------------------
# The trigger
# ------------------------------------------------------------------------------------------------

def test_a_weak_core_retrieve_fires():
    plan = xcm.plan_merge(
        ['core_memory_retrieve(query="where did we go", top_k=5)'],
        [_payload((0.14, 3, "friendship stuff"), (0.13, 4, "more friendship"))],
    )
    assert plan["fire"] is True
    assert plan["core_max"] == pytest.approx(0.14)
    assert plan["core_index"] == 0
    assert "archival_memory_retrieve" in plan["archival_call"]
    assert "where did we go" in plan["archival_call"], "the query must be reused VERBATIM"


def test_a_confident_core_retrieve_declines():
    """At or above threshold the store already answered. Firing here would add a call for nothing."""
    plan = xcm.plan_merge(
        ['core_memory_retrieve(query="q", top_k=5)'],
        [_payload((0.91, 1, "the answer"))],
    )
    assert plan["fire"] is False
    assert plan["declined"] == "core_confident"


def test_the_threshold_boundary_is_exclusive():
    """core_max == THRESHOLD is CONFIDENT. An off-by-one here would silently change the arm."""
    at = xcm.plan_merge(['core_memory_retrieve(query="q")'], [_payload((xcm.THRESHOLD, 1, "x"))])
    below = xcm.plan_merge(['core_memory_retrieve(query="q")'], [_payload((0.2999, 1, "x"))])
    assert at["fire"] is False
    assert below["fire"] is True


def test_a_non_vector_backend_is_a_structural_no_op():
    """Not a tuning choice: kv and rec_sum expose no per-entry similarity, so the trigger is
    undefined there. This is why A9 cannot move the other two shards, which is in turn what licenses
    the whole-corpus projection."""
    plan = xcm.plan_merge(
        ['core_memory_retrieve(query="q")'], [_payload((0.1, 1, "x"))], is_vector=False
    )
    assert plan["fire"] is False
    assert "similarity" in plan["declined"]


def test_retrieve_all_is_a_different_locus():
    """No query means nothing to re-dispatch."""
    plan = xcm.plan_merge(["core_memory_retrieve_all()"], [_payload((0.1, 1, "x"))])
    assert plan["fire"] is False


def test_an_errored_retrieve_belongs_to_the_capacity_anchors():
    plan = xcm.plan_merge(
        ['core_memory_retrieve(query="q")'],
        ['{"error": "core memory is full"}'],
        looks_like_error=lambda r: "error" in str(r),
    )
    assert plan["fire"] is False


# ------------------------------------------------------------------------------------------------
# The merge, and its safety property
# ------------------------------------------------------------------------------------------------

def test_entries_are_ordered_globally_by_score_regardless_of_store():
    """What the merge actually guarantees: one global ranking. NOT that a primary entry survives --
    see test_the_whole_primary_set_can_be_evicted, which is the retracted claim's counterexample."""
    core = xcm.parse_entries(_payload((0.55, 1, "good core"), (0.05, 2, "weak core")))
    arch = xcm.parse_entries(_payload((0.70, 9, "great archival"), (0.60, 8, "good archival")))
    merged = xcm.merge(core, arch, top_k=2)

    assert [e.entry_id for e in merged] == [9, 8]
    merged3 = xcm.merge(core, arch, top_k=3)
    assert 1 in [e.entry_id for e in merged3], "a core entry above the cut is kept -- because it RANKS"
    # Global ranking, i.e. sorted by score regardless of container.
    assert [e.score for e in merged3] == sorted((e.score for e in merged3), reverse=True)


def test_the_whole_primary_set_can_be_evicted_and_that_falsified_the_safety_claim():
    """The counterexample that retracted "never a displaced correct answer".

    A9 was accepted partly on an argument that the merge is "ADDITIVE by construction", so a false
    firing "costs one extra read, never a displaced correct answer". Both held-out losses had
    core_topk_before=5 and core_entries_retained=0: archival out-scored core ~2x and global ranking
    evicted EVERY core entry. On 89-student-9 the control answered correctly from a core entry the
    merge dropped.

    Displacement is not the exceptional case -- it is the EXPECTED one, because this anchor only fires
    when the primary store scored weakly, which is exactly when the sibling is likely to dominate.
    """
    core = xcm.parse_entries(_payload((0.256, 2, "sailing as problem solving"), (0.197, 3, "friends")))
    arch = xcm.parse_entries(_payload((0.516, 8, "scheduling"), (0.405, 9, "gaming")))
    merged = xcm.merge(core, arch, top_k=2)

    assert all(e.source == "archival" for e in merged), "total primary eviction is reachable"
    assert not [e for e in merged if e.source == "core"]


def test_displacement_is_reported_per_firing_rather_than_called_an_invariant():
    """It must be VISIBLE in telemetry, not inferred later from an accuracy delta -- and it must not
    be labelled an invariant, since it is not one."""
    plan = xcm.plan_merge(
        ['core_memory_retrieve(query="hobby", top_k=2)'],
        [_payload((0.256, 2, "sailing"), (0.197, 3, "friends"))],
    )
    out = xcm.complete_merge(plan, _payload((0.516, 8, "scheduling"), (0.405, 9, "gaming")))

    assert out["core_entries_retained"] == 0
    assert sorted(out["core_dropped"]) == [2, 3]
    assert "invariant" not in out, "the retracted claim must not be re-asserted as an invariant"
    assert "2 of 2" in out["displacement"]


def test_ties_break_toward_core():
    """An archival entry scoring EXACTLY equal must not evict a core entry."""
    core = xcm.parse_entries(_payload((0.42, 1, "core")))
    arch = xcm.parse_entries(_payload((0.42, 9, "archival")))
    merged = xcm.merge(core, arch, top_k=1)
    assert merged[0].source == "core"
    assert merged[0].entry_id == 1


def test_completing_a_merge_reports_non_destructiveness_per_firing():
    """The invariant is MEASURED and recorded, not asserted from the design."""
    plan = xcm.plan_merge(
        ['core_memory_retrieve(query="trip", top_k=2)'],
        [_payload((0.14, 3, "friends"), (0.13, 4, "more friends"))],
    )
    out = xcm.complete_merge(plan, _payload((0.62, 59, "London: Tower, British Museum")))

    assert out["fire"] is True
    assert out["archival_promoted"] == 1
    assert out["new_max"] == pytest.approx(0.62)
    assert out["core_index"] == 0, "the merged result replaces the weak one AT ITS INDEX"
    assert out["core_topk_before"] == 2
    assert out["core_entries_retained"] == 1
    assert out["core_dropped"] == [4], "the displaced id must be named, not just counted"


def test_an_empty_or_failed_archival_read_declines_rather_than_degrading():
    plan = xcm.plan_merge(['core_memory_retrieve(query="q")'], [_payload((0.1, 1, "x"))])
    assert xcm.complete_merge(plan, "")["declined"] == "archival_dispatch_failed"
    assert xcm.complete_merge(plan, '{"result": []}')["declined"] == "archival_empty"


def test_only_one_extra_call_is_ever_dispatched():
    """The cost of a false firing, stated as data so a port can assert it."""
    plan = xcm.plan_merge(['core_memory_retrieve(query="q")'], [_payload((0.1, 1, "x"))])
    assert len(xcm.dispatch_sequence(plan)) == 1
    assert xcm.dispatch_sequence({"fire": False}) == ()


# ------------------------------------------------------------------------------------------------
# Delivery -- the defect this anchor is named for
# ------------------------------------------------------------------------------------------------

def test_delivery_is_verified_against_the_text_the_model_actually_RECEIVED():
    """The check that would have caught the original defect on day one.

    v1/v2 wrote the merged payload to the telemetry sidecar and reported success. The only check that
    catches that is reading back from the object the MODEL consumes.
    """
    plan = xcm.plan_merge(['core_memory_retrieve(query="trip", top_k=1)'],
                          [_payload((0.14, 3, "friends"))])
    out = xcm.complete_merge(plan, _payload((0.62, 59, "London: Tower, British Museum")))

    delivered = xcm.verify_delivery("...tool result: London: Tower, British Museum ...", out)
    assert delivered["attested"] and delivered["delivered"]

    # The real failure: computed correctly, absent from the prompt. `London` appeared zero times.
    missed = xcm.verify_delivery("...friends at 0.140, 0.136, 0.133...", out)
    assert missed["attested"] and missed["delivered"] is False
    assert "VIOLATION" in missed["reason"]


def test_an_uncaptured_prompt_abstains_and_is_not_counted_as_a_failure():
    """`attested=False` means NOT MEASURED. Conflating that with "not delivered" produced a
    self-inconsistent first Gate 0 pass ("55/66 delivered, 0 violations") -- the original error in
    mirror image."""
    plan = xcm.plan_merge(['core_memory_retrieve(query="q")'], [_payload((0.1, 1, "x"))])
    out = xcm.complete_merge(plan, _payload((0.5, 9, "y")))
    result = xcm.verify_delivery(None, out)
    assert result["attested"] is False
    assert "NOT a failure" in result["reason"]
    assert "delivered" not in result, "abstaining must not report a delivery verdict either way"


# ------------------------------------------------------------------------------------------------
# Measured vs projected -- the provenance boundary
# ------------------------------------------------------------------------------------------------

def test_no_row_is_projected_any_more_and_the_flag_is_kept_for_future_rounds():
    """T9 was the only projected row; the three-shard run confirmed it, so nothing is projected now.

    The `projected` field stays on Round rather than being deleted: "measured or projected?" has to be
    answerable for any round, and the next anchor may well land shard-first the way this one did.
    """
    assert [r.tag for r in PROGRESSION if r.projected] == []
    assert CUMULATIVE["endpoint_projected"] is False
    assert "projected" in PROGRESSION[0].__dataclass_fields__, (
        "keep the flag available -- the next anchor may also arrive shard-first"
    )


def test_the_confirmed_corpus_figures_still_equal_the_shard_net_plus_the_prior_endpoint():
    """The arithmetic that WAS the projection must still hold now that it is measured.

    This is the more interesting assertion after confirmation, not a weaker one: the three-shard run
    reproduced the projected cells exactly, so the identity below is evidence the projection METHOD was
    sound -- and if a future re-measurement breaks it, that method is what should be doubted.
    """
    shard = A9_EVIDENCE["vector_shard"]
    corpus = A9_EVIDENCE["whole_corpus"]
    prior = CUMULATIVE["measured_endpoint"]          # T8

    train_net = shard["train"]["arm"][0] - shard["train"]["control"][0]
    dev_net = shard["dev"]["arm"][0] - shard["dev"]["control"][0]
    assert (train_net, dev_net) == (15, 7)

    assert corpus["train"]["to"][0] == prior["train"][0] + train_net == 159
    assert corpus["dev"]["to"][0] == prior["dev"][0] + dev_net == 41
    assert "MEASURED" in corpus["status"]
    assert PROGRESSION[-1].train_correct == 159
    assert PROGRESSION[-1].dev_correct == 41


def test_non_interference_is_measured_not_inferred_from_the_backend_guard():
    """"The code cannot touch those backends" is an ARGUMENT. A9's own history is why that is not
    enough: the delivery defect was also a sound-looking argument about what the gate did."""
    ni = A9_EVIDENCE["non_interference"]
    assert ni["gate_firings_off_target"] == 0
    assert ni["gate_declines_off_target"] == 0
    for backend in ("kv", "rec_sum"):
        for split in ("train", "dev"):
            assert "byte-identical" in ni[backend][split]
    assert "MEASURED" in CUMULATIVE["endpoint_confirmed"]


def test_the_additivity_retraction_is_recorded_with_the_verdict_left_standing():
    """A retraction inside an ACCEPTED anchor. Both halves must stay legible: the rationale is false,
    and the acceptance holds because the rule requires net >= 0, not zero losses."""
    r = A9_EVIDENCE["additivity_retracted"]
    assert "NEVER a displaced correct answer" in r["claimed"]
    assert "core_entries_retained=0" in r["falsified_by"]
    assert "MECHANISM" in r["reasoning_error"] and "OUTCOME" in r["reasoning_error"]
    assert "NOT NON-DESTRUCTIVE" in r["corrected"]
    assert "net dev >= 0" in r["why_acceptance_stands"].lower()
    assert "retracted" in r["do_not_reuse"].lower()
    # The mechanism module must carry it too, not only the ledger.
    module = (REPO / "anchoropt" / "mechanisms" / "cross_container_merge.py").read_text()
    assert "RETRACTED" in module
    assert "NET-POSITIVE, NOT NON-DESTRUCTIVE" in module


def test_the_readme_records_that_the_projection_was_confirmed_and_keeps_t8_visible():
    """After confirmation the README must say the cells were once projected and are now measured.

    Deleting the projection history would make the record look tidier than it was: for one commit the
    headline rested on arithmetic, and the fact that the run reproduced it exactly is evidence about
    the METHOD that is worth keeping.
    """
    low = PROGRESSION_DOC.lower()
    assert "projection" in low or "projected" in low, "the history must not be erased"
    assert "reproduced them" in low or "reproduced it" in low
    assert "three-shard" in low
    # T8 stays visible as the offline-recomputable endpoint a reproduction starts from.
    assert "47.52" in PROGRESSION_DOC
    assert "recompute offline" in low or "recompute" in low
    # The vector-shard figures are the anchor's own measured claim.
    for token in ("60.7", "52.5", "+16.9", "+17.5"):
        assert token in PROGRESSION_DOC, f"the measured vector figure {token} is missing"
    # And the retraction must be visible to a reader of the headline, not buried in a ledger.
    assert "non-destructiveness" in low or "not on non-destructiveness" in low


def test_the_shard_figures_are_never_presented_as_corpus_figures():
    """Reading 60.7% as a corpus number overstates A9 by ~3x. The units must be attached."""
    idx = PROGRESSION_DOC.index("60.7")
    window = PROGRESSION_DOC[max(0, idx - 400): idx + 400].lower()
    assert "vector" in window, "a shard figure must carry its shard label within sight of it"


def test_a9_declares_its_scope_and_both_open_items_closed_with_history_kept():
    """Both residuals are closed. What they SAID is kept, because one of them closed by discovering
    its own premise was wrong -- the 'unattested' firings were stale artifacts from a previous run,
    not a gap in the code. Deleting the item would delete that lesson too."""
    assert "vector" in A9.params["scope"]

    closed = A9_EVIDENCE["closed"]
    assert set(closed) == {"prereq_delivery", "three_shard_confirmation"}
    assert "STALE ARTIFACTS" in closed["prereq_delivery"]
    assert "55/55" in closed["prereq_delivery"] and "ZERO unattested" in closed["prereq_delivery"]

    # The original wording survives, so the record shows what was believed and when.
    was = " ".join(A9_EVIDENCE["was_open"])
    assert "PREREQ-PHASE DELIVERY IS UNVERIFIED" in was
    assert "THREE-SHARD CONFIRMATION NOT RUN" in was


def test_the_noise_floor_is_recorded_because_it_bounds_every_single_shard_claim():
    """A 1.1 pp difference on the SAME arm, from store-build nondeterminism. Without this number a
    reader cannot tell which single-shard differences elsewhere in the repo are meaningful."""
    floor = A9_EVIDENCE["noise_floor"]
    assert "61.8" in floor and "60.7" in floor
    assert "1.1 pp" in floor
    # And it must explain the cache-sharing rule it implies, which is protocol, not trivia.
    assert "SHARE a snapshot cache" in floor


def test_the_voided_experiments_are_recorded_rather_than_deleted():
    """Seven retractions. A record that keeps only surviving conclusions is a sales sheet."""
    voided = A9_EVIDENCE["delivery_defect"]["voided"]
    assert len(voided) == 7
    assert any("two-model probe" in v for v in voided)
    assert any("0.00pp" in v for v in voided)


def test_a9_is_not_claimed_as_the_first_read_side_anchor():
    """A2 is read-side too, and an earlier draft of this port called A9 "the first read-side anchor".

    That was wrong, and it mattered: A2 and A9 share a diagnosis -- the value is in archival and the
    read did not consult archival -- so the interesting claim is not primacy but the TRIGGER TIER.
    A2 needs an error string; A9 fires where the retrieve succeeded and nothing raised.
    """
    from rounds.anchors import A2

    assert "archive" in A2.attribution or "archival" in A2.attribution, (
        "A2 is the read-side anchor A9 must be compared against"
    )

    for path in ("README.md", "docs/ANCHORS.md", "rounds/T9_A9_xcontainer_merge/README.md"):
        body = (REPO / path).read_text().lower()
        assert "first anchor on the read side" not in body, f"{path} revives the overclaim"
        assert "first read-side anchor" not in body or "not the first read-side anchor" in body, (
            f"{path} claims A9 is the first read-side anchor; A2 is"
        )

    assert "not the first read-side anchor" in A9.notes.lower() or "NOT the first" in A9.notes


def test_the_a2_a9_distinction_is_recorded_where_a9_is_described():
    """The distinction that replaced the overclaim must actually be stated, not just removed."""
    for path in ("docs/ANCHORS.md", "rounds/T9_A9_xcontainer_merge/README.md"):
        body = (REPO / path).read_text()
        low = body.lower()
        assert "A2" in body, f"{path} must name A2 when describing A9"
        assert "error" in low and "score" in low, (
            f"{path} must contrast A2's error-string trigger with A9's score trigger"
        )
        assert "reprompt" in low, f"{path} must state that A2 reprompts while A9 acts itself"


# ------------------------------------------------------------------------------------------------
# Generality: the decision must survive a substrate with different names and a different wire format
# ------------------------------------------------------------------------------------------------

def test_the_decision_works_on_a_substrate_with_different_store_names_and_format():
    """The point of RetrievalAdapter, demonstrated rather than asserted in a docstring.

    A hypothetical substrate: stores called `working` and `longterm`, a different call syntax, and a
    CSV-ish payload instead of JSON. No function in the mechanism is edited -- only the adapter.
    """
    def parse_csv(result):
        out = []
        for row in str(result or "").strip().splitlines():
            if not row.strip():
                continue
            rid, score, text = row.split(",", 2)
            out.append(xcm.Entry(float(score), "", int(rid), text))
        out.sort(key=lambda e: -e.score)
        return out

    adapter = xcm.RetrievalAdapter(
        name="working_longterm_pair",
        primary="working",
        sibling="longterm",
        threshold=0.5,                                   # a different threshold, too
        search_re=re.compile(r"\bsearch_working\s*\("),
        sibling_call_template='search_{store}(query="{query}", k={top_k})',
        parse=parse_csv,
    )

    plan = xcm.plan_merge(
        ['search_working(query="budget", top_k=2)'],
        ["3,0.20,some note\n4,0.10,another note"],
        adapter=adapter,
    )
    assert plan["fire"] is True, "0.20 < 0.5 is weak on this substrate"
    assert plan["archival_call"] == 'search_longterm(query="budget", k=2)'
    assert plan["adapter"] == "working_longterm_pair"
    assert "working" in plan["reason"] and "longterm" in plan["reason"]

    out = xcm.complete_merge(plan, "9,0.80,the budget is in longterm", adapter=adapter)
    assert out["fire"] is True
    assert out["archival_promoted"] == 1
    assert out["new_max"] == pytest.approx(0.80)
    assert out["core_entries_retained"] == 1, "additivity holds on any substrate"


def test_the_thresholds_are_per_adapter_not_global():
    """A score that is weak on one substrate may be confident on another. 0.40 is weak at t=0.5 and
    confident at t=0.30, and the same code must give both answers."""
    strict = xcm.RetrievalAdapter(name="strict", primary="a", sibling="b", threshold=0.5,
                                  search_re=re.compile(r"\bread_a\s*\("))
    lenient = xcm.RetrievalAdapter(name="lenient", primary="a", sibling="b", threshold=0.30,
                                   search_re=re.compile(r"\bread_a\s*\("))
    calls, results = ['read_a(query="q")'], [_payload((0.40, 1, "x"))]
    assert xcm.plan_merge(calls, results, adapter=strict)["fire"] is True
    assert xcm.plan_merge(calls, results, adapter=lenient)["fire"] is False


def test_the_shipped_adapter_does_not_name_a_benchmark():
    """The core must not name a benchmark in executable code -- an earlier draft called this constant
    BFCL_VECTOR and tests/test_adapter_parity.py caught it. Named for the substrate SHAPE instead."""
    assert "bfcl" not in xcm.CORE_ARCHIVAL.name.lower()
    assert xcm.CORE_ARCHIVAL.primary == "core" and xcm.CORE_ARCHIVAL.sibling == "archival"


def test_the_action_family_mismatch_is_recorded_not_hidden():
    """A9 is filed REROUTE, and it substitutes no call -- the retrieve already succeeded.

    Recorded as an open classification question rather than fixed by adding a fifth action family on
    the strength of one anchor. If a second observation-augmenting anchor arrives, revisit.
    """
    from anchoropt.anchor import Action

    assert A9.action is Action.REROUTE
    caveat = A9_EVIDENCE["action_family_caveat"].lower()
    assert "substitutes no call" in caveat
    assert "augment" in caveat, "the honest description of the operation must be named"
