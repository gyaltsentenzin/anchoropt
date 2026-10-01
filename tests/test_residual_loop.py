"""The residual hierarchy: prioritized problem first, controller ranking strictly inside it.

The regression these pin is measured, not hypothetical. In Self-Evolve R1 the top residual
(support 4, 50% coverage) had one candidate covering all its cases, that candidate was infeasible,
nothing searched an alternative action for it, and the budget went to a candidate covering 1 of its
4 cases which survived because it was INSTANTIABLE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

import bfcl_runtime as R                                                # noqa: E402
from anchoropt.anchor import Action, IncisionPoint                       # noqa: E402
from anchoropt.learning.anchor_policy_opt import (                       # noqa: E402
    FrozenIncumbent, SearchSpaceProposal,
)
from anchoropt.learning.policy_class import ThetaResult                  # noqa: E402
from anchoropt.learning.residual_loop import run_residual_round, summarize_round  # noqa: E402
from anchoropt.learning.residual_problem import (                        # noqa: E402
    POLICY_ACCEPTED, POLICY_EXHAUSTED_CURRENT_SPACE, SIGNAL_BLOCKED_RESIDUAL,
    build_residual_problems,
)
from anchoropt.runtime import ResidualDiagnosis                          # noqa: E402

SIM = "retrieval_similarity_below_threshold"
POST = IncisionPoint.POST_EXECUTION
GATE = IncisionPoint.POST_GENERATION_PRE_EXEC
OBS = {"best_similarity": [0.05, 0.12, 0.2, 0.31, 0.44, 0.69]}


def _d(cid, decision):
    return ResidualDiagnosis(case_id=cid, mechanism=f"mechanism of {cid}", evidence="e",
                             consequential_decision=decision, proposed_behavior_change="")


# The R1 shape: 4 / 3 / 1 over three consequential decisions.
BIG = "commit to a final answer on the first non-empty retrieval"
MID = "accept the first non-empty retrieval as sufficient evidence"
SMALL = "answer from context alone without querying memory"
R1_RESIDUAL = ([_d(f"big{i}", BIG) for i in range(4)]
               + [_d(f"mid{i}", MID) for i in range(3)]
               + [_d("small0", SMALL)])


def _res(net, fire=5, n=89):
    return ThetaResult(theta={"below": 0.31},
                       gains=tuple(f"g{i}" for i in range(max(net, 0))),
                       losses=tuple(f"l{i}" for i in range(max(-net, 0))),
                       firings=fire, cases_fired=fire, n=n, interventions_executed=fire)


def _evaluate_net(net):
    return lambda arm, theta: _res(net)


def _accept_positive(r):
    return (r.net > 0 and r.interventions_executed > 0), "net>0 and engaged"


# ---- ranking -----------------------------------------------------------------------------------

def test_residual_problems_rank_by_support_metric_unchanged():
    problems = build_residual_problems(R1_RESIDUAL)
    assert [p.support for p in problems] == [4, 3, 1]
    assert [p.rank for p in problems] == [1, 2, 3]
    assert problems[0].key.startswith("commit to a final answer")
    assert abs(problems[0].coverage - 0.5) < 1e-9


def test_ranking_is_deterministic():
    a = [p.key for p in build_residual_problems(R1_RESIDUAL)]
    b = [p.key for p in build_residual_problems(list(reversed(R1_RESIDUAL)))]
    assert a == b


def test_higher_support_residual_is_searched_first():
    seen = []

    def propose_for(problem):
        seen.append(problem.key)
        return ()                                    # no proposals -> exhausted, move on

    run_residual_round(R1_RESIDUAL, runtime=R, host=R.HOST, propose_for=propose_for,
                       evaluate=_evaluate_net(0), accept=_accept_positive,
                       incumbent=FrozenIncumbent("P0", "tok"))
    assert seen[0].startswith("commit to a final answer"), seen
    assert len(seen) == 3 and seen == sorted(seen, key=lambda k: -len(
        [d for d in R1_RESIDUAL if " ".join(d.consequential_decision.lower().split()) == k]))


# ---- one residual at a time --------------------------------------------------------------------

def test_proposer_sees_only_the_frozen_residuals_diagnoses():
    """Passing the whole residual lets the proposer answer about a different, easier problem."""
    handed = {}

    def propose_for(problem):
        handed[problem.rank] = set(d.case_id for d in problem.scoped_diagnoses())
        return ()

    run_residual_round(R1_RESIDUAL, runtime=R, host=R.HOST, propose_for=propose_for,
                       evaluate=_evaluate_net(0), accept=_accept_positive,
                       incumbent=FrozenIncumbent("P0", "tok"))
    assert handed[1] == {"big0", "big1", "big2", "big3"}
    assert handed[2] == {"mid0", "mid1", "mid2"}
    assert not (handed[1] & handed[2])


def test_a_proposal_citing_another_residual_is_refused():
    """R2 candidates must not enter the same AnchorPolicyOpt call as R1's."""
    def propose_for(problem):
        return (SearchSpaceProposal(
            boundary=POST, signal=SIM, action_set=(Action.REPROMPT,),
            diagnosis_case_ids=("big0", "mid0")),)          # mid0 belongs to R2

    with pytest.raises(ValueError, match="another residual"):
        run_residual_round(R1_RESIDUAL, runtime=R, host=R.HOST, propose_for=propose_for,
                           evaluate=_evaluate_net(0), accept=_accept_positive,
                           incumbent=FrozenIncumbent("P0", "tok"))


# ---- an infeasible action must not demote the residual -----------------------------------------

def test_infeasible_first_action_keeps_the_residual_frozen():
    """The measured R1 defect: one infeasible action moved the search to a lower residual."""
    calls = []

    def propose_for(problem):
        calls.append(problem.rank)
        if problem.rank == 1:
            # SUPPRESS at post_execution is structurally inadmissible -> infeasible. REPROMPT at the
            # same locus HAS a host executor, so it is the materializable alternative. (REROUTE is
            # admissible in U_H(l) but this host has no executor for it, so it would be rejected
            # too -- which is why the alternative here must be the one that can actually run.)
            return (SearchSpaceProposal(boundary=POST, signal=SIM,
                                        action_set=(Action.SUPPRESS, Action.REPROMPT),
                                        diagnosis_case_ids=tuple(problem.case_ids)),)
        return ()

    out = run_residual_round(R1_RESIDUAL, runtime=R, host=R.HOST, propose_for=propose_for,
                             evaluate=_evaluate_net(-1), accept=_accept_positive,
                             incumbent=FrozenIncumbent("P0", "tok"), observations=OBS)
    led = out.ledger_for(out.problems[0].key)
    assert any("suppress" in a for _l, a, _r in led.infeasible), led.infeasible
    assert led.measured, "the feasible alternative action for R1 must still be measured"
    assert calls[0] == 1


def test_alternative_loci_for_the_same_residual_are_tried_before_moving_on():
    tried = []

    def propose_for(problem):
        if problem.rank != 1:
            return ()
        return (
            SearchSpaceProposal(boundary=GATE, signal="no_tool_call_at_all",
                                action_set=(Action.REROUTE,),          # ungroundable at the gate
                                diagnosis_case_ids=tuple(problem.case_ids)),
            SearchSpaceProposal(boundary=POST, signal=SIM, action_set=(Action.REPROMPT,),
                                diagnosis_case_ids=tuple(problem.case_ids)),
        )

    out = run_residual_round(R1_RESIDUAL, runtime=R, host=R.HOST, propose_for=propose_for,
                             evaluate=_evaluate_net(-1), accept=_accept_positive,
                             incumbent=FrozenIncumbent("P0", "tok"), observations=OBS)
    led = out.ledger_for(out.problems[0].key)
    tried = led.attempted_loci
    assert len(tried) == 2, f"both offered loci for R1 must be tried: {tried}"
    assert ("post_generation_pre_exec", "no_tool_call_at_all") in tried
    assert (POST.value, SIM) in tried


# ---- exhaustion ---------------------------------------------------------------------------------

def test_moving_to_r2_requires_exhaustion_of_r1():
    order = []

    def propose_for(problem):
        order.append(problem.rank)
        return (SearchSpaceProposal(boundary=POST, signal=SIM, action_set=(Action.REPROMPT,),
                                    diagnosis_case_ids=tuple(problem.case_ids)),)

    out = run_residual_round(R1_RESIDUAL, runtime=R, host=R.HOST, propose_for=propose_for,
                             evaluate=_evaluate_net(-2), accept=_accept_positive,
                             incumbent=FrozenIncumbent("P0", "tok"), observations=OBS)
    assert order == [1, 2, 3], "residuals must be visited in rank order"
    led = out.ledger_for(out.problems[0].key)
    assert led.state == POLICY_EXHAUSTED_CURRENT_SPACE
    assert led.measured, "exhaustion must follow MEASUREMENT, not a guess"


def test_exhaustion_is_scoped_not_a_global_claim():
    def propose_for(problem):
        return () if problem.rank != 1 else (
            SearchSpaceProposal(boundary=POST, signal=SIM, action_set=(Action.REPROMPT,),
                                diagnosis_case_ids=tuple(problem.case_ids)),)

    out = run_residual_round(R1_RESIDUAL, runtime=R, host=R.HOST, propose_for=propose_for,
                             evaluate=_evaluate_net(-2), accept=_accept_positive,
                             incumbent=FrozenIncumbent("P0", "tok"), observations=OBS)
    reason = out.ledger_for(out.problems[0].key).exhausted_reason()
    assert POLICY_EXHAUSTED_CURRENT_SPACE in reason
    assert "NOT a global claim" in reason
    assert "CURRENT Phi" in reason


def test_inexpressible_residual_is_preserved_for_the_signal_block():
    def expressible(d):
        return not d.case_id.startswith("big")

    out = run_residual_round(R1_RESIDUAL, runtime=R, host=R.HOST,
                             propose_for=lambda p: (), evaluate=_evaluate_net(0),
                             accept=_accept_positive, incumbent=FrozenIncumbent("P0", "tok"),
                             expressible=expressible)
    led = out.ledger_for(out.problems[0].key)
    assert led.state == SIGNAL_BLOCKED_RESIDUAL
    assert "representation limit" in led.exhausted_reason()
    assert "SIGNAL block" in led.exhausted_reason()


# ---- acceptance ends the round ------------------------------------------------------------------

def test_an_accepted_r1_policy_ends_the_round_for_promotion_and_remine():
    visited = []

    def propose_for(problem):
        visited.append(problem.rank)
        return (SearchSpaceProposal(boundary=POST, signal=SIM, action_set=(Action.REPROMPT,),
                                    diagnosis_case_ids=tuple(problem.case_ids)),)

    out = run_residual_round(R1_RESIDUAL, runtime=R, host=R.HOST, propose_for=propose_for,
                             evaluate=_evaluate_net(+4), accept=_accept_positive,
                             incumbent=FrozenIncumbent("P0", "tok"), observations=OBS)
    assert out.outcome == POLICY_ACCEPTED
    assert out.accepted_problem.rank == 1
    assert visited == [1], "no lower residual may be searched after an acceptance"
    assert "re-mine" in summarize_round(out)


# ---- THE regression test for the measured R1 failure -------------------------------------------

def test_easy_low_priority_controller_cannot_preempt_the_top_residual():
    """THE R1 REGRESSION, as it actually happened.

    R1 (support 4) offers a broad controller whose first action is INFEASIBLE. R3 (support 1) offers
    an easy, immediately feasible, POSITIVE controller. The old loop enumerated both and took the
    instantiable one. The fixed loop must keep working on R1 -- and here R1 has a feasible
    alternative action that also measures positive, so R1 must win.
    """
    def propose_for(problem):
        if problem.rank == 1:
            return (SearchSpaceProposal(
                boundary=POST, signal=SIM,
                action_set=(Action.SUPPRESS, Action.REPROMPT),   # suppress infeasible, reprompt not
                diagnosis_case_ids=tuple(problem.case_ids)),)
        # the easy low-priority controller that hijacked the R1 round
        return (SearchSpaceProposal(boundary=POST, signal=SIM, action_set=(Action.REPROMPT,),
                                    diagnosis_case_ids=tuple(problem.case_ids)),)

    out = run_residual_round(R1_RESIDUAL, runtime=R, host=R.HOST, propose_for=propose_for,
                             evaluate=_evaluate_net(+3), accept=_accept_positive,
                             incumbent=FrozenIncumbent("P0", "tok"), observations=OBS)
    assert out.accepted_problem is not None
    assert out.accepted_problem.rank == 1, "the top residual must win, not the easy one"
    assert out.accepted_problem.support == 4
    assert out.searched_keys == (out.problems[0].key,), \
        "no lower-priority residual may even be searched"
    led = out.ledger_for(out.problems[0].key)
    assert any("suppress" in a for _l, a, _r in led.infeasible), \
        "the infeasible action must be recorded, not hidden"
