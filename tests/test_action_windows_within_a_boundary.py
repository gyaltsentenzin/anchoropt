"""One incision point can contain several WINDOWS with different executable actions.

WHY THIS EXISTS (R13). `post_generation_pre_exec` is three places in the BFCL host:

    zero_call        the step proposed NO tool call      -- nothing to cancel
    call_filtering   calls queued for dispatch           -- a suppress can drop one
    answer_boundary  decision finished, about to answer  -- NO pending calls, nothing to cancel

MEASURED, not reasoned: a `suppress` controller was installed at the answer boundary, FIRED, and
executed nothing -- the window's only action is injecting a message, so it declined with "controller
has no eta.instruction". That arm was voided.

THE COST OF NOT HAVING THIS. The oracle-localization test offered `reprompt` and `suppress` as equally
feasible, their theta tables tied byte-for-byte, and the winner fell out of `Action` enum sort order. A
tie between a real candidate and an inexecutable one is not a tie -- it is
anchoropt-degenerate-acceptance with the degeneracy moved one layer up, into the candidate set.

WHERE IT LIVES. `HostProfile` is keyed by `IncisionPoint` and validates that its keys ARE IncisionPoints,
so a window cannot be expressed there without changing core. It is declared in the ADAPTER
(`ACTION_WINDOWS`), and core ASKS the runtime rather than knowing what a window is.
"""

from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

from anchoropt.learning.policy_family import (                     # noqa: E402
    SemanticProposal, expand_policy_family)
from anchoropt.learning.policy_class import PolicyClass            # noqa: E402
from anchoropt.runtime import Action, IncisionPoint                # noqa: E402

GATE = IncisionPoint.POST_GENERATION_PRE_EXEC
SIGNAL = "retrieval_similarity_below_threshold"


def _rt():
    import bfcl_runtime
    return bfcl_runtime


def test_a_window_narrows_and_can_never_widen():
    rt = _rt()
    at_boundary = rt.feasible_actions(GATE)
    for site in rt.ACTION_WINDOWS:
        narrowed = rt.feasible_actions(GATE, site)
        assert narrowed <= at_boundary, (
            f"window {site!r} widens the host's declaration -- a window may only narrow")


def test_suppress_is_executable_where_calls_are_pending_and_not_where_they_are_not():
    rt = _rt()
    assert Action.SUPPRESS in rt.feasible_actions(GATE, "call_filtering"), (
        "suppress must remain executable where calls are queued -- that is where accepted "
        "suppressors act")
    for site in ("answer_boundary", "zero_call"):
        assert Action.SUPPRESS not in rt.feasible_actions(GATE, site), (
            f"suppress is offered at {site!r}, where there is no pending call to cancel")


def test_reprompt_survives_in_every_window():
    """The action that CAN act at all three, so narrowing must not remove it."""
    rt = _rt()
    for site in rt.ACTION_WINDOWS:
        assert Action.REPROMPT in rt.feasible_actions(GATE, site), site


def test_an_unknown_window_falls_back_to_the_boundary_not_to_nothing():
    """A typo must not prune every action -- that would read as 'this host can do nothing here'."""
    rt = _rt()
    assert rt.feasible_actions(GATE, "not-a-window") == rt.feasible_actions(GATE)
    assert rt.feasible_actions(GATE, "") == rt.feasible_actions(GATE)


def test_the_search_prunes_an_inexecutable_action_WITH_A_REASON():
    """The behaviour the voided arm paid for: suppress pruned, not offered as an equal candidate."""
    rt = _rt()
    proposal = SemanticProposal(
        boundary=GATE, signal=SIGNAL, policy_class=PolicyClass.PARAMETERIZED,
        site="answer_boundary", action_preference="", theta_hint={})
    feasible, infeasible = expand_policy_family(proposal, runtime=rt, host=rt.HOST)

    actions = {c.action for c in feasible}
    assert actions == {Action.REPROMPT}, f"expected reprompt alone, got {actions}"
    reasons = {i.action: i.reason for i in infeasible}
    assert Action.SUPPRESS in reasons, "suppress was neither offered nor pruned -- it vanished"
    assert "not_executable_in_window" in reasons[Action.SUPPRESS]
    assert "answer_boundary" in reasons[Action.SUPPRESS], (
        "the prune reason must name the window, or it is unattributable")


def test_omitting_the_window_preserves_the_old_behaviour_exactly():
    """Every existing caller passes no site. Their search must be unchanged."""
    rt = _rt()
    with_site = SemanticProposal(boundary=GATE, signal=SIGNAL, site="call_filtering")
    no_site = SemanticProposal(boundary=GATE, signal=SIGNAL)
    a, _ = expand_policy_family(with_site, runtime=rt, host=rt.HOST)
    b, _ = expand_policy_family(no_site, runtime=rt, host=rt.HOST)
    assert {c.action for c in a} == {c.action for c in b}, (
        "naming the window where suppress IS executable changed the candidate set; the two must agree")
