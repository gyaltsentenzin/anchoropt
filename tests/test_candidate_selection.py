"""The pre-registered selection policy: deterministic, auditable, and it catches the pilot's defect.

The policy is specified in rounds/AUTONOMY/PREREGISTERED_POLICY.md and was committed BEFORE this
implementation. These tests assert the implementation matches it, including the replay of the pilot's own
six emitted arms.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchoropt.learning.candidate_selection import (
    CAPABILITY_MISMATCH, HISTORY_INFORMED, MEASURED_LOSER, NEUTRAL, NEVER_FIRES, POLICIES,
    POLICY_EXHAUSTED, SETTLED, Candidate, select_candidate)

REPO = Path(__file__).resolve().parent.parent
PILOT = REPO / "rounds" / "AUTORUN" / "r1"


def C(label, operator="transform", support=1, fires=10, **kw):
    kw.setdefault("signal", "sig")
    kw.setdefault("boundary", "post_execution")
    return Candidate(label=label, operator=operator, support=support, fires_on_states=fires,
                     total_states=100, **kw)


# -- rule order ---------------------------------------------------------------------------------

def test_higher_residual_support_wins_first():
    """The pipeline's own ranking is the primary key, not the operator family."""
    s = select_candidate([C("a", operator="suppress", support=8),
                          C("b", operator="transform", support=4)])
    assert s.chosen.label == "a"


def test_recovery_beats_suppression_within_one_support_level():
    """The one substantive prior: suppression measured -3/-4 here, recovery +5/+4."""
    s = select_candidate([C("sup", operator="suppress", support=8),
                          C("rec", operator="transform", support=8)],
                         policy=HISTORY_INFORMED)
    assert s.chosen.label == "rec"


def test_reprompt_sits_between_recovery_and_suppression():
    s = select_candidate([C("sup", operator="suppress", support=8),
                          C("rep", operator="reprompt", support=8),
                          C("rec", operator="substitute", support=8)],
                         policy=HISTORY_INFORMED)
    assert [c.label for c in s.ranked] == ["rec", "rep", "sup"]


def test_an_undeclared_operator_family_sorts_LAST_never_first():
    s = select_candidate([C("weird", operator="mystery", support=8),
                          C("sup", operator="suppress", support=8)],
                         policy=HISTORY_INFORMED)
    assert s.chosen.label == "sup"


def test_fewer_projected_firings_wins_within_a_family():
    """The narrower hypothesis: a 97%-firing predicate is a recorded diagnostic warning."""
    s = select_candidate([C("wide", fires=500, support=8), C("narrow", fires=40, support=8)])
    assert s.chosen.label == "narrow"


def test_the_tie_break_is_lexicographic_and_fully_deterministic():
    arms = [C("zzz", support=8, fires=10), C("aaa", support=8, fires=10)]
    assert select_candidate(arms).chosen.label == "aaa"
    assert select_candidate(list(reversed(arms))).chosen.label == "aaa"


def test_the_same_round_always_selects_the_same_arm():
    import itertools
    arms = [C("a", operator="suppress", support=8, fires=83),
            C("b", operator="transform", support=8, fires=84),
            C("c", operator="substitute", support=4, fires=84)]
    for pol in POLICIES:
        picks = {select_candidate(list(p), policy=pol).chosen.label
                 for p in itertools.permutations(arms)}
        assert len(picks) == 1, f"{pol}: selection must not depend on input order"


# -- eligibility filters ------------------------------------------------------------------------

def test_a_settled_controller_is_excluded_with_a_reason():
    s = select_candidate([C("installed", support=9), C("fresh", support=1)],
                         settled=["installed"])
    assert s.chosen.label == "fresh"
    assert any(e.reason == SETTLED for e in s.excluded)


def test_a_measured_loser_is_not_re_measured():
    """A round is not re-spent on an arm already measured no-benefit or negative."""
    s = select_candidate([C("lost", support=9), C("new", support=1)], measured_losers=["lost"])
    assert s.chosen.label == "new"
    assert any(e.reason == MEASURED_LOSER for e in s.excluded)


def test_an_arm_that_fires_nowhere_is_excluded():
    s = select_candidate([C("dead", support=9, fires=0), C("live", support=1)])
    assert s.chosen.label == "live"
    assert any(e.reason == NEVER_FIRES for e in s.excluded)


def test_an_infeasible_arm_is_excluded():
    s = select_candidate([C("bad", support=9, feasible=False), C("ok", support=1)])
    assert s.chosen.label == "ok"


def test_THE_PILOT_DEFECT_a_capability_mismatch_is_excluded():
    """A substitute arm carrying the relocation executor's id would measure the CONTROL."""
    s = select_candidate([
        C("substitute-arm", operator="substitute", support=9,
          capability_id="relocate_entry_preserving_information_then_retry",
          resolved_capability_id="substitute_destination_and_replace"),
        C("clean", support=1)])
    assert s.chosen.label == "clean"
    e = [x for x in s.excluded if x.reason == CAPABILITY_MISMATCH][0]
    assert "relocate_entry_preserving_information_then_retry" in e.detail


def test_no_eligible_candidate_is_a_RESULT_not_an_error():
    s = select_candidate([C("x", support=1)], settled=["x"])
    assert not s.ok and s.terminal == POLICY_EXHAUSTED
    assert "NO ELIGIBLE CANDIDATE" in s.rationale()


def test_the_rationale_names_the_choice_and_every_exclusion():
    s = select_candidate([C("chosen", support=8), C("dead", support=9, fires=0)])
    r = s.rationale()
    assert "SELECTED chosen" in r and "EXCLUDED dead" in r and "support=8" in r


# -- replay of the actual pilot round -----------------------------------------------------------

@pytest.mark.skipif(not (PILOT / "arm_manifest.json").exists(), reason="pilot round not present")
def test_replaying_the_PILOT_ROUND_reproduces_a_defensible_choice():
    """Against the six arms the pilot actually emitted, with their real firing counts.

    The two substitute arms carry the relocation capability_id (the recorded defect), so the policy
    excludes them and selects the relocation transform -- the same arm the pilot measured, but by RULE
    rather than by a human reading the ranking.
    """
    arms = json.loads((PILOT / "arm_manifest.json").read_text())["arms"]
    RELOC = "relocate_entry_preserving_information_then_retry"
    cands = []
    for a in arms:
        op = a.get("operator") or ("suppress" if "suppress" in a["arm_label"] else "transform")
        # capacity-family arms share the top residual's support; the suppression-only signals sit lower
        support = 8 if "container_at_capacity" in a["arm_label"] else 4
        cap = RELOC if "container_at_capacity" in a["arm_label"] else ""
        resolved = RELOC if op == "transform" else (
            "substitute_destination_and_replace" if op == "substitute" else "")
        cands.append(Candidate(label=a["arm_label"], operator=op, signal=a["signal"],
                               boundary=a["boundary"], support=support,
                               fires_on_states=a["fires_on_states"], total_states=a["total_states"],
                               capability_id=cap, resolved_capability_id=resolved))
    s = select_candidate(cands)
    assert s.ok
    assert "relocate_entry_for_kv" in s.chosen.label
    assert len([e for e in s.excluded if e.reason == CAPABILITY_MISMATCH]) == 2


# -- the two policies must stay distinguishable --------------------------------------------------

def test_NEUTRAL_is_the_default_so_no_caller_gets_our_history_by_accident():
    s = select_candidate([C("x", support=1)])
    assert s.policy == NEUTRAL


def test_the_neutral_policy_does_NOT_encode_an_operator_preference():
    """Same support, suppression narrower: neutral takes the narrower arm, history takes recovery."""
    arms = [C("sup", operator="suppress", support=8, fires=83),
            C("rec", operator="transform", support=8, fires=84)]
    assert select_candidate(arms, policy=NEUTRAL).chosen.label == "sup"
    assert select_candidate(arms, policy=HISTORY_INFORMED).chosen.label == "rec"


def test_both_policies_share_the_residual_ranking_as_the_primary_key():
    arms = [C("low", operator="transform", support=2), C("high", operator="suppress", support=9)]
    for pol in POLICIES:
        assert select_candidate(arms, policy=pol).chosen.label == "high", pol


def test_every_selection_records_which_policy_chose():
    """A result cannot be quoted without the prior that produced it."""
    for pol in POLICIES:
        s = select_candidate([C("a", support=1)], policy=pol)
        assert s.policy == pol and f"policy={pol}" in s.rationale()


def test_an_unknown_policy_is_refused_rather_than_silently_defaulted():
    import pytest as _pt
    with _pt.raises(ValueError, match="unknown selection policy"):
        select_candidate([C("a")], policy="whatever_i_feel_like")
