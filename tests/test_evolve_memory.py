"""Self-Evolve v0.2 memory: the guards as CODE, not as prompt text a role may ignore."""

import json
import pathlib

import pytest

from anchoropt.evolve_memory import (
    EVALUATE, INSTALLED, REUSE, Experiment, ExperimentLedger, Lesson, candidate_fingerprint,
    context_key, experience_block, normalize_eta, rank_key, reorder_residuals, residual_penalty,
    screen_all, screen_candidate,
)

CTX = context_key(host="h", split="train", backend="vector")
OTHER = context_key(host="h", split="train", backend="kv")
R = "whether to commit to a final answer without checking the retrieval"


def led(tmp_path) -> ExperimentLedger:
    return ExperimentLedger(tmp_path / "mem")


def exp(**kw):
    base = dict(round_id="T", residual=R, locus="post_generation_pre_exec",
                signal="no_tool_call_at_all", action="reprompt",
                eta={"instruction": "Do the thing.", "retry_budget": 1},
                status="rejected", context=CTX)
    base.update(kw)
    return Experiment(**base)


# ---------------------------------------------------------------- fingerprints
def test_formatting_differences_collapse():
    a = candidate_fingerprint(residual=R, locus="POST_GENERATION_PRE_EXEC ", signal="no_tool_call_at_all",
                              action="REPROMPT", eta={"instruction": "Do  the thing.", "retry_budget": 1})
    b = candidate_fingerprint(residual=R, locus="post_generation_pre_exec", signal="no_tool_call_at_all",
                              action="reprompt", eta={"retry_budget": 1.0, "instruction": "do the thing"})
    assert a == b, "case, whitespace, key order and 1-vs-1.0 must not create a new candidate"


def test_DIFFERENT_WORDING_IS_A_DIFFERENT_CANDIDATE():
    """se3a1 vs se3a2 differ only in instruction TEXT and measured differently, so text is
    load-bearing and must never be normalized away."""
    k = dict(residual=R, locus="post_generation_pre_exec", signal="no_tool_call_at_all",
             action="reprompt")
    a = candidate_fingerprint(**k, eta={"instruction": "Check the retrieved content.", "retry_budget": 1})
    b = candidate_fingerprint(**k, eta={"instruction": "Search the other container.", "retry_budget": 1})
    assert a != b


def test_retry_budget_change_is_a_different_candidate():
    """se3a3 differed from se3a2 only in retry_budget and was infeasible; it is not the same arm."""
    k = dict(residual=R, locus="post_generation_pre_exec", signal="no_tool_call_at_all",
             action="reprompt")
    assert (candidate_fingerprint(**k, eta={"instruction": "x", "retry_budget": 1})
            != candidate_fingerprint(**k, eta={"instruction": "x", "retry_budget": 0}))


def test_non_identifying_eta_keys_are_excluded():
    k = dict(residual=R, locus="l", signal="s", action="reprompt")
    assert (candidate_fingerprint(**k, eta={"instruction": "x", "detail": "a", "variant": "v1"})
            == candidate_fingerprint(**k, eta={"instruction": "x", "detail": "b", "variant": "v2"}))


# ---------------------------------------------------------------- screening
def test_untried_candidate_is_evaluated(tmp_path):
    L = led(tmp_path)
    v = screen_candidate(L, residual=R, locus="l", signal="s", action="reprompt", eta={"i": 1})
    assert v.decision == EVALUATE and v.spend_gpu


def test_previously_rejected_candidate_is_REUSED_not_re_evaluated(tmp_path):
    L = led(tmp_path)
    L.record(exp(gains=1, losses=4, firings=46, reason="engaged and hurt"))
    v = screen_candidate(L, residual=R, locus="post_generation_pre_exec",
                         signal="no_tool_call_at_all", action="reprompt",
                         eta={"instruction": "Do the thing.", "retry_budget": 1}, context=CTX)
    assert v.decision == REUSE and not v.spend_gpu
    assert "net -3" in v.reason


def test_ACCEPTED_candidate_is_INSTALLED_not_merely_reused(tmp_path):
    """Evaluating the promoted controller measures the incumbent against itself.

    SE1's third proposal was exactly the promoted A4' controller, so this is not hypothetical -- and
    reporting its original delta as a new finding is the error being prevented.
    """
    L = led(tmp_path)
    L.record(exp(status="accepted", gains=4, losses=0, firings=24, reason="promoted"))
    v = screen_candidate(L, residual=R, locus="post_generation_pre_exec",
                         signal="no_tool_call_at_all", action="reprompt",
                         eta={"instruction": "Do the thing.", "retry_budget": 1}, context=CTX)
    assert v.decision == INSTALLED and not v.spend_gpu
    assert "incumbent against itself" in v.reason


def test_INVALID_prior_does_NOT_block_a_retry(tmp_path):
    """An invalid run produced a number that cannot be believed, so the candidate is still untried.

    The c3 arm fired zero times and the seed-2 round was a re-execution; treating either as refutation
    would permanently retire a candidate that was never actually tested.
    """
    L = led(tmp_path)
    L.record(exp(status="invalid", firings=0, reason="fired zero times"))
    v = screen_candidate(L, residual=R, locus="post_generation_pre_exec",
                         signal="no_tool_call_at_all", action="reprompt",
                         eta={"instruction": "Do the thing.", "retry_budget": 1}, context=CTX)
    assert v.decision == EVALUATE and v.spend_gpu


def test_a_result_from_ANOTHER_CELL_does_not_block(tmp_path):
    """Cells are indivisible; a rec_sum/kv result is not evidence about vector."""
    L = led(tmp_path)
    L.record(exp(context=OTHER, gains=0, losses=3))
    v = screen_candidate(L, residual=R, locus="post_generation_pre_exec",
                         signal="no_tool_call_at_all", action="reprompt",
                         eta={"instruction": "Do the thing.", "retry_budget": 1}, context=CTX)
    assert v.decision == EVALUATE


def test_an_INSTALLED_controller_stays_installed_under_a_DIFFERENT_residual(tmp_path):
    """Whether a controller is already in the incumbent is a property of the INTERVENTION.

    This test previously asserted the opposite -- that a new residual family makes an accepted
    controller evaluable again -- and that is precisely the SE1 bug: re-mining produced a different
    residual family, the proposer re-proposed the already-promoted A4' controller under it, and a
    residual-scoped screen called it "new". Evaluating it would measure the incumbent against itself.
    """
    L = led(tmp_path)
    L.record(exp(status="accepted"))
    v = screen_candidate(L, residual="a completely different consequential decision",
                         locus="post_generation_pre_exec", signal="no_tool_call_at_all",
                         action="reprompt",
                         eta={"instruction": "Do the thing.", "retry_budget": 1}, context=CTX)
    assert v.decision == INSTALLED


def test_a_DIFFERENT_INTERVENTION_is_still_evaluated(tmp_path):
    """The screen must not become a blanket veto: change the eta and it is a new candidate."""
    L = led(tmp_path)
    L.record(exp(status="accepted"))
    v = screen_candidate(L, residual=R, locus="post_generation_pre_exec",
                         signal="no_tool_call_at_all", action="reprompt",
                         eta={"instruction": "Something else entirely.", "retry_budget": 1},
                         context=CTX)
    assert v.decision == EVALUATE


def test_screen_all_preserves_order(tmp_path):
    L = led(tmp_path)
    cands = [{"locus": "l", "signal": "s", "action": "reprompt", "eta": {"i": i}} for i in range(3)]
    out = screen_all(L, cands, residual=R)
    assert [c["eta"]["i"] for c, _ in out] == [0, 1, 2]


# ---------------------------------------------------------------- downweighting
def test_failed_attempts_downweight_a_residual_but_never_exclude_it(tmp_path):
    L = led(tmp_path)
    w0, why0 = residual_penalty(L, R)
    assert w0 == 1.0 and why0 == "untried"
    for i in range(6):
        L.record(exp(eta={"instruction": f"try {i}", "retry_budget": 1}))
    w1, why1 = residual_penalty(L, R)
    assert w1 < w0, "a tried-and-failed residual must rank lower"
    assert w1 >= 0.25, "but it must never be fully suppressed -- the space was refuted, not the residual"
    assert "failed attempt" in why1


def test_downweighting_reorders_equal_support_residuals(tmp_path):
    L = led(tmp_path)
    for i in range(3):
        L.record(exp(residual="tried residual", eta={"instruction": f"t{i}", "retry_budget": 1}))

    class P:
        def __init__(self, key, support):
            self.key, self.support = key, support
    tried, fresh = P("tried residual", 4), P("fresh residual", 4)
    order = [p.key for p, _, _ in reorder_residuals(L, [tried, fresh])]
    assert order == ["fresh residual", "tried residual"], "equal support -> untried first"


def test_rank_key_is_deterministic(tmp_path):
    L = led(tmp_path)
    assert rank_key(L, R, 4) == rank_key(L, R, 4)


# ---------------------------------------------------------------- context block
def test_experience_block_is_empty_for_a_fresh_ledger(tmp_path):
    assert experience_block(led(tmp_path)) == ""


def test_experience_block_has_the_three_sections_and_is_BOUNDED(tmp_path):
    L = led(tmp_path)
    for i in range(30):
        L.record(exp(eta={"instruction": f"attempt {i}", "retry_budget": 1}))
    L.record_lesson(Lesson(key="k1", text="localize before selecting an action"))
    L.record_lesson(Lesson(key="k2", text="seed variation is not replication",
                           kind="invalid_approach"))
    block = experience_block(L, residual=R, locus="post_generation_pre_exec")
    assert "RELEVANT PRIOR EXPERIENCE" in block
    assert "Structural lessons:" in block
    assert "Previously tried for similar residuals:" in block
    assert "Known invalid approaches:" in block
    # 30 attempts recorded, at most 6 shown -- the whole history must NOT be dumped
    assert block.count("-> REJECTED") <= 6


def test_lessons_are_idempotent_by_key(tmp_path):
    L = led(tmp_path)
    L.record_lesson(Lesson(key="dup", text="first"))
    L.record_lesson(Lesson(key="dup", text="second"))
    assert len([l for l in L.lessons() if l.key == "dup"]) == 1


def test_ledger_is_append_only_and_keeps_both_rows(tmp_path):
    L = led(tmp_path)
    L.record(exp(status="rejected", reason="first"))
    L.record(exp(status="accepted", reason="later, under other conditions"))
    rows = L.experiments()
    assert len(rows) == 2, "a status change must not overwrite the earlier evaluation"
    assert {r.status for r in rows} == {"rejected", "accepted"}


def test_bad_status_is_refused():
    with pytest.raises(ValueError):
        exp(status="probably_fine")


# ---------------------------------------------------------------- primitives
def test_primitive_registry_reads_the_RUNTIME_not_a_hand_list():
    import sys, pathlib as _p
    sys.path.insert(0, str(_p.Path(__file__).resolve().parent.parent / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    from anchoropt.evolve_memory.primitives import available, prompt_block, unavailable
    names = {p.name for p in available(rt)}
    assert "reprompt@post_generation_pre_exec" in names
    # reroute@post_execution WAS in the unavailable set; the registry has since been completed from
    # the evaluator's real firing sites, so it is available now. The registry reads the runtime, so
    # this test tracks the runtime rather than pinning a stale list -- which is the property under test.
    assert "reroute@post_execution" in names
    assert "suppress@post_generation_pre_exec" in names
    block = prompt_block(rt)
    assert "enable_zero_call_reprompt" in block
    # the executor's fixed parameter must be stated, since violating it is what killed se3a3
    assert "retry_budget is FIXED at 1" in block


def test_v02_memory_is_not_imported_by_the_v01_cycle_or_by_learning():
    """Additive: v0.2 must not have changed what SE1 runs."""
    import re
    repo = pathlib.Path(__file__).resolve().parent.parent
    targets = [repo / "scripts" / "self_evolve_cycle.py"] + \
        [p for p in (repo / "anchoropt" / "learning").rglob("*.py")]
    for f in targets:
        assert not re.search(r"anchoropt\.evolve_memory", f.read_text()), \
            f"{f.name} imports the v0.2 memory; wiring is a separate, deliberate step"


# ---------------------------------------------------------------- canonical residual families
def test_SE1_seventeen_prose_keys_collapse_to_one_family():
    """The measured defect: 65 diagnoses under 17 prose keys, all one failure mode.

    Support ranking on raw prose read 4/4/4/.../1 -- sixteen ties at support 4, none of which is
    really support 4 -- and coverage fell from 50% to 6.2% for what is structurally ONE problem.
    """
    from anchoropt.evolve_memory.families import family_of
    variants = [
        "Whether an empty or unhelpful retrieval result may be accepted as sufficient grounds to answer",
        "Whether an empty retrieval result may be accepted as adequate grounding for a final answer",
        "Whether to commit to a final answer after retrieval returned zero results",
        "Whether to produce a final answer when retrieval has returned zero supporting records",
        "Whether to commit to a final answer while no retrieved memory content is in hand",
        "Whether to answer from an empty retrieval result instead of re-querying memory",
    ]
    assert len({family_of(v) for v in variants}) == 1
    assert family_of(variants[0]) == "answer_despite_unhelpful_retrieval"


def test_the_three_retrieval_families_stay_DISTINCT():
    """Nothing-called, returned-nothing and returned-something-weak are three different conditions
    needing three different interventions; collapsing them would merge A4 with the net-negative
    low-similarity family."""
    from anchoropt.evolve_memory.families import family_of
    assert family_of("whether to commit to a final answer having made no tool call at all") \
        == "answer_without_any_tool_call"
    # NB the decision must actually mention answering -- "accepted as adequate grounding" alone
    # matches no family and correctly falls back, which is what the first version of this test hit.
    assert family_of("whether an empty retrieval result may be accepted as adequate grounding "
                     "for a final answer") == "answer_despite_unhelpful_retrieval"
    assert family_of("whether to accept the first non-empty retrieval as sufficient evidence") \
        == "answer_on_weak_first_retrieval"


def test_NON_EMPTY_is_not_read_as_EMPTY():
    """'non-empty' contains 'empty'. A bare substring test sent the weak-retrieval family into the
    empty-result family -- caught only because the discrimination test failed."""
    from anchoropt.evolve_memory.families import family_of
    for phrase in ("the first non-empty retrieval", "a single nonempty result"):
        text = f"whether to accept {phrase} as sufficient evidence for the final answer"
        assert family_of(text) == "answer_on_weak_first_retrieval", text


def test_unmatched_decisions_keep_their_own_identity():
    """An unrecognized failure mode must NOT be merged into a catch-all bucket."""
    from anchoropt.evolve_memory.families import family_of, is_canonical
    fid = family_of("whether to delete the production database on a tuesday")
    assert not is_canonical(fid)
    assert "production database" in fid


def test_unmatched_report_is_the_audit_surface():
    from anchoropt.evolve_memory.families import unmatched_report

    class D:
        def __init__(self, t):
            self.consequential_decision = t
    ds = [D("whether an empty retrieval may be accepted to answer")] * 3 + \
         [D("some entirely novel decision about widgets")] * 2
    rep = unmatched_report(ds)
    assert len(rep) == 1 and rep[0][1] == 2, "only the unmatched ones are reported, with counts"


def test_fingerprints_and_downweighting_key_on_the_FAMILY_not_the_prose(tmp_path):
    """All three of ranking, memory and downweighting must share one key, per the v0.2 brief."""
    L = led(tmp_path)
    prose_a = "whether an empty retrieval result may be accepted as adequate grounding for an answer"
    prose_b = "whether to commit to a final answer after retrieval returned zero supporting records"
    L.record(exp(residual=prose_a, gains=0, losses=2, reason="tried under prose A"))
    # a candidate proposed under a DIFFERENT paraphrase of the same family must be recognized
    v = screen_candidate(L, residual=prose_b, locus="post_generation_pre_exec",
                         signal="no_tool_call_at_all", action="reprompt",
                         eta={"instruction": "Do the thing.", "retry_budget": 1}, context=CTX)
    assert v.decision == REUSE, "a paraphrase must hit the ledger, not buy a second GPU run"
    wa, _ = residual_penalty(L, prose_a)
    wb, _ = residual_penalty(L, prose_b)
    assert wa == wb < 1.0, "paraphrases must share the same downweight"
