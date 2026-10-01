"""The v0.1 structured search: ResidualDiagnosis -> (l, phi, mu, theta).

The properties under test are the ones whose violation produced a wrong candidate during
development, each recorded here so it cannot silently return:

  * a step localizes a REGION -- all three boundaries stay candidates
  * attribution is a GATE, not a weight (an unimplicated signal is pruned, not ranked lower)
  * selection ranges over ALL diagnoses, not whichever came first
  * a generic token in common is not expressibility
  * selection is deterministic
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "tb2_deepagents"))

import tb2_adapter as tb2  # noqa: E402
from anchoropt.anchor import Action, IncisionPoint  # noqa: E402
from anchoropt.learning.candidate_search import (  # noqa: E402
    SIGNAL_BLOCKED, candidate_loci, enumerate_candidates, expressible_under, search, select,
)
from anchoropt.runtime import ResidualDiagnosis  # noqa: E402

TERMINAL = ResidualDiagnosis(
    case_id="mailman",
    mechanism="premature conclusion: the agent concluded with a chat response instead of "
              "executing the final postfix start command",
    evidence="AssertionError: Mailing list does not exist",
    consequential_decision="the decision to emit a terminal response",
    proposed_behavior_change="verify before concluding")

DOMAIN = ResidualDiagnosis(
    case_id="dna-insert",
    mechanism="automated primer design: the reverse primer was incomplete, missing 23nt",
    evidence="AssertionError: Annealed part of forward primer must be between 15 and 45",
    consequential_decision="the content of a generated biological sequence",
    proposed_behavior_change="fix the primer arithmetic")


def theta_for(action, diagnosis):
    if action is Action.REPROMPT:
        return {"text": diagnosis.proposed_behavior_change or "verify"}
    if action is Action.SUPPRESS:
        return {"reason": "test"}
    return None


# ---- a step is a region, not a locus ----------------------------------------------------------

def test_all_three_boundaries_stay_candidates():
    """A diagnosis localizes a region; narrowing to one boundary here is the conflation the
    architecture exists to prevent."""
    assert set(candidate_loci(step_hint=7)) == set(IncisionPoint)


# ---- expressibility ---------------------------------------------------------------------------

def test_a_terminal_response_failure_is_expressible():
    ok, why = expressible_under(TERMINAL, runtime=tb2)
    assert ok, why
    assert "terminal_response_proposed" in why


def test_a_domain_reasoning_failure_is_signal_blocked():
    """Primer arithmetic has no runtime observable. This is the Phi-expansion backlog, not a bug."""
    ok, why = expressible_under(DOMAIN, runtime=tb2)
    assert not ok
    assert why.startswith(SIGNAL_BLOCKED)


def test_a_single_generic_token_is_not_expressibility():
    """'response' alone matched `terminal_response_proposed` and selected a SUPPRESS controller
    for a primer-design failure. One weak token in common is a coincidence."""
    d = ResidualDiagnosis(case_id="x", mechanism="the response was malformed",
                          evidence="e", consequential_decision="d",
                          proposed_behavior_change="b")
    ok, _ = expressible_under(d, runtime=tb2)
    assert not ok


def test_declared_aliases_make_the_canonical_case_expressible():
    """The vocabulary and the diagnosis use different words for the same observable; the adapter
    declares the bridge, and the core never hard-codes domain words."""
    assert "premature conclusion" in tb2.signal_aliases("terminal_response_proposed")
    ok, _ = expressible_under(TERMINAL, runtime=tb2)
    assert ok


# ---- attribution is a gate --------------------------------------------------------------------

def test_an_unimplicated_signal_is_pruned_not_downranked():
    """Scoring instead of pruning let a 3.0 unimplicated candidate beat 13.0 attributed ones."""
    cands = enumerate_candidates(TERMINAL, runtime=tb2, host=tb2.HOST, theta_for=theta_for)
    for c in cands:
        if c.survived:
            assert c.signal in {"terminal_response_proposed"}, c.signal
    assert any(c.pruned_reason.startswith("signal_not_implicated") for c in cands)


def test_every_prune_carries_a_named_reason():
    cands = enumerate_candidates(TERMINAL, runtime=tb2, host=tb2.HOST, theta_for=theta_for)
    for c in cands:
        if not c.survived:
            assert ":" in c.pruned_reason, c
            assert c.pruned_reason.split(":")[0] in {
                "unexecutable_in_host", "signal_not_observable_at_boundary",
                "signal_params_unconfigured", "no_theta_available", "signal_not_implicated"}


def test_noop_is_never_proposed():
    """NOOP is the control arm, not a candidate."""
    cands = enumerate_candidates(TERMINAL, runtime=tb2, host=tb2.HOST, theta_for=theta_for)
    assert all(c.action is not Action.NOOP for c in cands)


def test_an_action_without_theta_is_pruned_not_given_empty_params():
    cands = enumerate_candidates(TERMINAL, runtime=tb2, host=tb2.HOST,
                                 theta_for=lambda a, d: None)
    assert cands and all(not c.survived for c in cands)
    assert any(c.pruned_reason.startswith("no_theta_available") for c in cands)


# ---- selection --------------------------------------------------------------------------------

def test_selection_ranges_over_all_diagnoses_not_the_first():
    """Residual ordering is arbitrary; letting it decide made the outcome depend on brief order."""
    sel_a, _, _ = search([DOMAIN, TERMINAL], runtime=tb2, host=tb2.HOST, theta_for=theta_for)
    sel_b, _, _ = search([TERMINAL, DOMAIN], runtime=tb2, host=tb2.HOST, theta_for=theta_for)
    assert sel_a is not None and sel_b is not None
    assert (sel_a.boundary, sel_a.signal, sel_a.action) == (sel_b.boundary, sel_b.signal, sel_b.action)


def test_selection_is_deterministic():
    picks = {tuple(str(x) for x in (s.boundary, s.signal, s.action))
             for s in (search([TERMINAL], runtime=tb2, host=tb2.HOST,
                              theta_for=theta_for)[0] for _ in range(5))}
    assert len(picks) == 1


def test_a_blocked_diagnosis_is_recorded_not_dropped():
    sel, considered, blocked = search([DOMAIN], runtime=tb2, host=tb2.HOST, theta_for=theta_for)
    assert sel is None and not considered
    assert len(blocked) == 1 and blocked[0][0].case_id == "dna-insert"


def test_select_returns_none_when_nothing_survives():
    assert select([]) is None


# ------------------------------------------------------------------ cross-round cell exclusion
# A round is stateless by design, and the ONLY de-duplication was within a round (`select_top_k` on
# the cell). So across rounds the search re-selected the identity it had already installed: measured
# live, `step()` chose the same (boundary, signal, action) cell twice running against the same
# residual, and an unattended loop would spend every round re-measuring its own incumbent.

def test_cell_of_is_the_controller_identity_and_ignores_theta():
    """A different parameterization of the same cell is a REVISION of one controller, not a second
    one -- the registry's supersession rule governs that, not the identity."""
    from anchoropt.learning.candidate_search import ScoredCandidate, cell_of
    a = ScoredCandidate(boundary=IncisionPoint.PRE_GENERATION, signal="s", action=Action.REPROMPT,
                        theta={"text": "one"})
    b = ScoredCandidate(boundary=IncisionPoint.PRE_GENERATION, signal="s", action=Action.REPROMPT,
                        theta={"text": "two"})
    assert cell_of(a) == cell_of(b) == ("pre_generation", "s", "reprompt")


def test_select_top_k_dedups_on_the_named_cell():
    """Pins that the within-round de-duplication and the cross-round exclusion use the SAME
    identity function; two spellings of it would drift."""
    from anchoropt.learning.candidate_search import ScoredCandidate, cell_of, select_top_k
    same = [ScoredCandidate(boundary=IncisionPoint.PRE_GENERATION, signal="s",
                            action=Action.REPROMPT, theta={"text": t}, evidence=3)
            for t in ("one", "two")]
    out = select_top_k(same, k=3)
    assert len(out) == 1 and cell_of(out[0]) == ("pre_generation", "s", "reprompt")


def _diag():
    return ResidualDiagnosis(
        case_id="c1", mechanism="concluded without verifying", evidence="assertion failure",
        consequential_decision="emit terminal", proposed_behavior_change="verify first",
        provider="test")


def _theta(action, diagnosis):
    return {"text": "verify"} if action == Action.REPROMPT else None


def test_search_excludes_a_cell_an_earlier_round_decided():
    from anchoropt.learning.candidate_search import ALREADY_DECIDED_IN_AN_EARLIER_ROUND, cell_of
    first, _, _ = search([_diag()], runtime=tb2, host=tb2.HOST, theta_for=_theta)
    assert first is not None
    again, considered, _ = search([_diag()], runtime=tb2, host=tb2.HOST, theta_for=_theta,
                                  exclude_cells=[cell_of(first)])
    assert again is None or cell_of(again) != cell_of(first)
    pruned = [c for c in considered if c.pruned_reason == ALREADY_DECIDED_IN_AN_EARLIER_ROUND]
    assert [cell_of(c) for c in pruned] == [cell_of(first)]


def test_an_excluded_candidate_is_RECORDED_not_dropped():
    """"Already decided" is an auditable outcome. Silently omitting it makes a round unable to say
    what it declined to re-run."""
    from anchoropt.learning.candidate_search import ALREADY_DECIDED_IN_AN_EARLIER_ROUND, cell_of
    first, base, _ = search([_diag()], runtime=tb2, host=tb2.HOST, theta_for=_theta)
    _, considered, _ = search([_diag()], runtime=tb2, host=tb2.HOST, theta_for=_theta,
                              exclude_cells=[cell_of(first)])
    assert len(considered) == len(base), "the candidate must still appear, with a reason"
    assert any(c.pruned_reason == ALREADY_DECIDED_IN_AN_EARLIER_ROUND for c in considered)


def test_excluding_nothing_is_the_previous_behaviour_exactly():
    """The fix must be inert when no exclusions are supplied, so every existing result stands."""
    from anchoropt.learning.candidate_search import cell_of
    a, ca, ba = search([_diag()], runtime=tb2, host=tb2.HOST, theta_for=_theta)
    b, cb, bb = search([_diag()], runtime=tb2, host=tb2.HOST, theta_for=_theta, exclude_cells=())
    assert cell_of(a) == cell_of(b)
    assert [c.pruned_reason for c in ca] == [c.pruned_reason for c in cb]
    assert len(ba) == len(bb)


def test_excluding_an_ALREADY_pruned_candidate_does_not_overwrite_its_reason():
    """An infeasible candidate's own pruned_reason is the more informative one, and losing it would
    hide a host limitation behind a bookkeeping message."""
    from anchoropt.learning.candidate_search import cell_of
    _, considered, _ = search([_diag()], runtime=tb2, host=tb2.HOST, theta_for=_theta)
    already = [c for c in considered if c.pruned_reason]
    assert already, "this host prunes at least one candidate"
    cells = [cell_of(c) for c in already]
    _, after, _ = search([_diag()], runtime=tb2, host=tb2.HOST, theta_for=_theta,
                         exclude_cells=cells)
    by_cell = {cell_of(c): c.pruned_reason for c in after}
    for c in already:
        assert by_cell[cell_of(c)] == c.pruned_reason


def test_step_honours_exclude_cells_so_a_loop_cannot_reinstall_its_own_incumbent():
    """The live defect, pinned: two consecutive rounds against the SAME residual chose the identical
    cell. With the memory supplied, the second round must move on."""
    from anchoropt.learning.candidate_search import cell_of
    from anchoropt.learning.self_evolve import Incumbent, step

    seen: set[tuple[str, str, str]] = set()
    picks = []
    for i in range(2):
        r = step(Incumbent(workspace=Path("/nonexistent"), incumbent_id=f"P{i}"),
                 runtime=tb2, host=tb2.HOST, diagnose=_diag_seq, theta_for=_theta,
                 exclude_cells=seen)
        if r.selected_candidate is None:
            break
        cell = cell_of(getattr(r.selected_candidate, "candidate", r.selected_candidate))
        picks.append(cell)
        seen.add(cell)
    assert len(set(picks)) == len(picks), f"a cell was re-selected across rounds: {picks}"


def _diag_seq():
    return (_diag(),)
