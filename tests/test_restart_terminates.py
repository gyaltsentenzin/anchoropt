"""THE RESTART AFTER A PROMOTION MUST TERMINATE, AND MUST NOT RE-MEASURE WHAT IT PROMOTED.

`optimize_residual` is the search the BFCL discoveries actually ran, and it owns the multi-round
loop itself: on a promotion it calls `promote(arm)`, then `remine()`, then restarts the WHERE sweep
against the new residual. Two things were missing from that restart and this file pins both.

MEASURED, on the toy host, before the fix:

    121 evaluations, 1 DISTINCT CELL, 1 promotion
    the promoted cell was re-evaluated 120 more times and the loop never returned

Why it could not stop:
  * `out.state` is set to IMPROVED on promotion and was never reset, so the guard
    `if out.state == IMPROVED: continue` fired on EVERY later iteration -- continuing without
    decrementing `i`, so the same boundary was swept forever;
  * `max_restarts` could not bound it, because `restarts` increments only on a PROMOTION, and the
    re-evaluated arm scores net 0 against the now-moved incumbent, so `improves` is false, so there
    is no promotion, so the counter never advances. The bound was unreachable from inside the cycle.
  * nothing recorded that the cell had already been decided, so even a terminating loop would spend
    its next round's entire budget re-measuring its own incumbent.

Nothing here is BFCL-specific: the host is `examples/toy_host`, which has no model and no network.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "examples" / "toy_host")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from anchoropt.learning.search_state import IMPROVED          # noqa: E402
from anchoropt.learning.structured_search import (  # noqa: E402
    _cell_of_arm, optimize_residual)

#: Far above anything a correct run needs on this host: a correct sweep promotes once and stops,
#: and the pre-fix defect blew past 120. A cap is the only way to assert TERMINATION in a test.
_HARD_CAP = 200

#: Wall-clock ceiling for one search. An EVALUATION cap cannot see every non-termination: two of the
#: defects here spin in arm CONSTRUCTION, evaluating nothing at all, so a counting evaluator never
#: fires and the test hangs instead of failing. A hang is not a result, so termination gets its own
#: watchdog.
_DEADLINE_S = 20.0


def _deadline():
    """A callable that raises once the wall clock is spent. Passed to the evaluator AND consulted in
    a wrapper around the host, so a loop that never evaluates is still caught."""
    import time
    t0 = time.monotonic()

    def check():
        if time.monotonic() - t0 > _DEADLINE_S:
            raise _Stop(f"did not terminate: still running after {_DEADLINE_S}s with no result")

    return check


class _Stop(Exception):
    """Raised by the counting evaluator so a non-terminating search fails as a FAILURE."""


@pytest.fixture()
def toy():
    import toy_adapter
    importlib.reload(toy_adapter)
    ad = toy_adapter.ADAPTER
    ad.reset_expanded_signals()
    return ad


def _run(toy, *, always_remine: bool):
    """Drive optimize_residual with a real evaluator, a real promote and a real remine.

    `always_remine=True` is the load-bearing configuration: it says "there is still residual to
    fix", which is the ordinary case for any benchmark whose first controller does not solve
    everything, and it is the only way to reach the restart at all.
    """
    import test_toy_host_e2e as T
    from toy_host.runtime import run_corpus

    residual, events, states, _base, _failing = T._residual_and_events()
    inner = T._Ev(toy, incumbent=None)
    seen: list[tuple[str, str, str]] = []
    promoted: list[tuple[str, str, str]] = []
    inc = {"ctl": None}

    due = _deadline()

    def evaluate(arm, theta=None):
        due()
        cell = (arm.boundary.value, arm.signal, arm.action.value)
        seen.append(cell)
        if len(seen) > _HARD_CAP:
            raise _Stop(f"did not terminate: {len(seen)} evaluations, "
                        f"{len(set(seen))} distinct cells")
        return inner(arm, theta)

    # THE WATCHDOG ON THE CONSTRUCTION PATH. Several defects this file pins spin while BUILDING arms
    # and never reach an evaluator, so the evaluation cap above cannot see them and the test would
    # HANG rather than fail -- and a hang is not a result. Measured spin sites were
    # `action_contract.instantiate` (via `AnchorPolicyOpt.build_arms`) and `toy_adapter.states_at`,
    # i.e. not one chokepoint, so the clock is checked on `build_arms` itself: every arm the schedule
    # constructs comes through it, and it is an EXISTING seam -- no production hook is added for the
    # benefit of a test.
    from anchoropt.learning.anchor_policy_opt import AnchorPolicyOpt

    _real_build = AnchorPolicyOpt.build_arms

    def _watched(self, *a, **k):
        due()
        return _real_build(self, *a, **k)

    AnchorPolicyOpt.build_arms = _watched

    def promote(arm):
        promoted.append((arm.boundary.value, arm.signal, arm.action.value))
        inc["ctl"] = inner.controller_for(arm)

    def remine():
        got = run_corpus(inc["ctl"])
        inner.baseline = got          # rebase: the moved incumbent is the new control
        still = sorted(c for c, ok in got["solved"].items() if not ok)
        if not (still or always_remine):
            return None

        class _R:
            key = "residual after promotion"
            case_ids = tuple(still) or residual.case_ids
            rank = 1

        return _R()

    try:
        out = optimize_residual(residual, runtime=toy, host=toy.HOST, events=events, states=states,
                                evaluate=evaluate, promote=promote, remine=remine, max_restarts=3)
    finally:
        AnchorPolicyOpt.build_arms = _real_build
    return out, seen, promoted


def test_the_restart_terminates_when_there_is_still_residual(toy):
    """The defect: this call never returned. Termination is the assertion."""
    try:
        out, seen, promoted = _run(toy, always_remine=True)
    except _Stop as e:
        pytest.fail(str(e))
    assert out.state == IMPROVED
    assert len(seen) <= _HARD_CAP


def test_a_promoted_cell_is_never_re_evaluated_after_its_promotion(toy):
    """Budget spent re-measuring the incumbent is budget not spent discovering anything.

    This is the same contract `candidate_search.select_top_k` states for a single round -- the same
    cell reached twice is ONE controller -- enforced across the restart boundary.
    """
    out, seen, promoted = _run(toy, always_remine=True)
    assert promoted, "the host must promote at least once or this test proves nothing"
    for cell in promoted:
        after = seen[seen.index(cell) + 1:]
        assert cell not in after, (
            f"the promoted cell {cell} was re-evaluated {after.count(cell)} times after it was "
            f"promoted and installed")


def test_the_loop_does_not_spend_its_whole_budget_on_one_cell(toy):
    """The pre-fix signature was 121 evaluations over exactly ONE distinct cell."""
    out, seen, promoted = _run(toy, always_remine=True)
    assert len(set(seen)) >= 1
    if len(seen) > 3:
        assert len(set(seen)) > 1, (
            f"{len(seen)} evaluations over a single cell {set(seen)} -- the search is re-measuring "
            f"one controller instead of searching")


def test_the_pre_existing_single_promotion_path_is_unchanged(toy):
    """When remine reports an EMPTY residual the search returns after one promotion, exactly as
    before. The fix must not disturb the behaviour every existing result was measured under."""
    out, seen, promoted = _run(toy, always_remine=False)
    assert out.state == IMPROVED
    assert len(promoted) == 1
    assert out.restarts == 0


# ================================================================================================
# WHAT THE CELL MEMORY MUST NOT COST: HOW-EXHAUSTION WITHIN A ROUND
# ================================================================================================

def test_two_eta_groundings_of_one_cell_are_both_measured_in_the_same_round(toy):
    """A cell-level memory must NOT collapse the HOW search.

    `(post_generation_pre_exec, nothing_proposed, reprompt)` is realized by two different grounded
    etas on this host -- advise_archive and advise_shorten. They are the SAME cell and DIFFERENT
    arms, and step 2 of the schedule exists precisely to exhaust them. If `decided` were consulted
    per-arm inside the evaluation loop rather than partitioned once before it, the second eta would
    be skipped as "already decided" and the HOW-exhaustion claim would be silently false.

    This is the counterweight to the exclusion tests above: they demand the cell NOT be re-measured
    across rounds, this one demands its etas ARE all measured within one.
    """
    out, seen, _promoted = _run(toy, always_remine=True)
    labels = [c for c in seen]
    reprompts = [c for c in labels if c[2] == "reprompt"]
    assert len(reprompts) >= 2, (
        f"only {len(reprompts)} reprompt arm(s) measured -- the eta search collapsed: {seen}")


def test_an_arm_the_evaluator_DECLINED_to_score_stays_eligible(toy):
    """No verdict means no decision. A budget refusal must not become a permanent rejection.

    `_eval` returns None both when there is no evaluator and when the budget is spent, and the
    outcome is BUDGET_EXHAUSTED precisely so a run of unmeasurable arms is not read as a null. If
    such an arm were added to `decided`, the requeue it is entitled to would silently never happen.
    """
    import test_toy_host_e2e as T
    from toy_host.runtime import run_corpus

    residual, events, states, _b, _f = T._residual_and_events()
    inner = T._Ev(toy, incumbent=None)
    inc = {"ctl": None}
    offered: list[tuple[str, str, str]] = []
    declined: list[tuple[str, str, str]] = []

    def evaluate(arm, theta=None):
        cell = (arm.boundary.value, arm.signal, arm.action.value)
        offered.append(cell)
        if len(offered) > _HARD_CAP:
            raise _Stop("did not terminate")
        # Decline the FIRST arm outright, score everything after it.
        if len(offered) == 1:
            declined.append(cell)
            return None
        return inner(arm, theta)

    def promote(arm):
        inc["ctl"] = inner.controller_for(arm)

    def remine():
        got = run_corpus(inc["ctl"])
        inner.baseline = got
        still = sorted(c for c, ok in got["solved"].items() if not ok)

        class _R:
            key = "r"
            case_ids = tuple(still) or residual.case_ids
            rank = 1

        return _R()

    out = optimize_residual(residual, runtime=toy, host=toy.HOST, events=events, states=states,
                            evaluate=evaluate, promote=promote, remine=remine, max_restarts=3)
    assert declined, "the test must actually decline an arm"
    # The declined cell carries no verdict, so nothing in the outcome may report one about it.
    for att in out.attempts:
        if att.state == "ALL_CELLS_ALREADY_DECIDED":
            assert att.detail, "a decided-out boundary must say so"
    assert out.state  # terminated at all


def test_the_two_selection_paths_agree_on_what_a_controller_identity_IS(toy):
    """`structured_search._cell_of_arm` and `candidate_search.cell_of` must name the same triple.

    They are deliberately separate functions -- two independent selection paths that must not be
    coupled -- so the thing that keeps them honest is this test. If they ever disagree, a cell
    settled on one path stays eligible on the other and the loop re-measures its own incumbent
    through whichever door it happens to use.
    """
    import inspect

    from anchoropt.learning import candidate_search
    from anchoropt.learning.structured_search import _cell_of_arm

    class _B:
        value = "post_execution"

    class _A:
        value = "reroute"

    class _Arm:
        boundary = _B()
        signal = "filing_failed"
        action = _A()

    class _Cand:
        boundary = _B()
        signal = "filing_failed"
        action = _A()
        theta = {"anything": 1}

    assert _cell_of_arm(_Arm()) == candidate_search.cell_of(_Cand())
    # and neither may read theta/eta: a re-parameterization is a revision, not a new controller
    assert "theta" not in inspect.getsource(_cell_of_arm)


# ================================================================================================
# THE LEDGER'S MEMORY MUST REACH **THIS** SEARCH, NOT ONLY `candidate_search`
# ================================================================================================

def test_the_ledgers_excluded_cells_feed_optimize_residual_unchanged():
    """`round_ledger.excluded_cells()` was written for `candidate_search(exclude_cells=...)`.

    `optimize_residual` is a DIFFERENT selection path -- the one the BFCL discoveries actually ran --
    and it takes `decided_cells`. Both consume the same (boundary, signal, action) identity, so the
    ledger's output must feed this search with no translation step. A translation step is exactly
    where the two memories would drift apart, and then a settled controller gets re-measured through
    whichever memory the round happens to consult.

    The asymmetry is the load-bearing part: SETTLED and MEASURED_LOSER exclude, RETRYABLE does NOT.
    A RETRYABLE identity's evidence was ABSENT, not negative, so it must stay selectable.
    """
    import inspect

    from anchoropt.learning.round_ledger import (OUTCOME_ACCEPTED, OUTCOME_BLOCKED,
                                                 OUTCOME_REJECTED, RoundLedger, excluded_cells,
                                                 label_for_cell)

    assert "decided_cells" in inspect.signature(optimize_residual).parameters

    settled = ("post_execution", "filing_failed", "reroute")
    loser = ("post_generation_pre_exec", "nothing_proposed", "suppress")
    retryable = ("pre_execution", "budget_low", "advise")

    led = RoundLedger()
    led.record(label=label_for_cell(settled), outcome=OUTCOME_ACCEPTED, net=3)
    led.advance()
    led.record(label=label_for_cell(loser), outcome=OUTCOME_REJECTED, net=-1)
    led.advance()
    led.record(label=label_for_cell(retryable), outcome=OUTCOME_BLOCKED)

    # exactly the coercion `optimize_residual` performs on the argument
    decided = {tuple(c) for c in excluded_cells(led)}

    assert settled in decided, "an accepted controller must not be re-measured"
    assert loser in decided, "a measured rejection is a decided cell"
    assert retryable not in decided, (
        "a BLOCKED round produced no verdict, so its identity must stay eligible -- excluding it "
        "would turn absent evidence into a permanent rejection")
    assert all(len(c) == 3 and all(isinstance(x, str) for x in c) for c in decided)


# =================================================================================================
# Every promotion must be able to restart -- there are THREE promotion sites, not one.
# =================================================================================================
#
# `optimize_residual` promotes from the primary HOW sweep, from the grounded FRONTIER sweep, and from
# the Phi-EXPANSION sweep. Only the first ever called `remine`, so a winner found on either of the
# other two returned immediately: the incumbent moved, the trajectories changed, and the search
# never looked at the new residual. The multi-round schedule silently collapsed to a single
# promotion exactly when the winner came from expansion -- which is the path that discovers a
# core-synthesized signal, i.e. the most interesting discoveries.
#
# Measured before the fix, on the toy host with an evaluator that promotes only at a boundary
# reached by expansion: `remine` called 0 times, `out.restarts == 0`, `out.state == IMPROVED`.

def _residual_fixture():
    import test_toy_host_e2e as T
    return T._residual_and_events()


def test_a_promotion_from_any_sweep_consults_remine():
    """A promotion is a promotion: whichever sweep found it, the residual it was measured against
    is now stale, so the schedule owes the caller a re-mine."""
    import importlib

    import toy_adapter
    importlib.reload(toy_adapter)
    toy = toy_adapter.ADAPTER
    toy.reset_expanded_signals()
    residual, events, states, _b, _f = _residual_fixture()

    # Nothing on the primary sweep wins. The only winner sits at a boundary/signal pair that the
    # core must SYNTHESIZE, so the promotion can only come from the expansion path.
    gen, offered, remines = {"n": 0}, [], []

    def ev(arm, theta=None):
        from anchoropt.learning.policy_class import ThetaResult
        cell = _cell_of_arm(arm)
        offered.append((gen["n"], cell))
        if len(offered) > _HARD_CAP:
            raise _Stop(f"non-termination: {len(offered)} evaluations")
        # win only on an expanded (synthesized) signal -- the toy host's shipped signals are
        # 'filing_failed' / 'nothing_proposed' / 'error_kind'; expansion produces 'payload_chars_*'
        win = cell[1].startswith("payload_chars")
        return ThetaResult(theta={}, gains=("g1", "g2") if win else (), losses=(),
                           firings=3, cases_fired=3, n=12, interventions_executed=3)

    def rm():
        remines.append(gen["n"])
        gen["n"] += 1
        return None          # one re-mine is all this test needs to observe

    out = optimize_residual(residual, runtime=toy, host=toy.HOST, events=events, states=states,
                               evaluate=ev, promote=lambda a: None, remine=rm, max_restarts=3)

    promoted_cells = [c for _g, c in offered if c[1].startswith("payload_chars")]
    assert out.state == IMPROVED, f"fixture did not promote from expansion: {out.state}"
    assert promoted_cells, "the expansion path was never reached, so this test proves nothing"
    assert remines, ("promoted from the expansion sweep without ever calling remine: the incumbent "
                     "moved and the search never saw the new residual")


def test_a_promotion_from_expansion_actually_SWEEPS_the_new_residual():
    """Calling `remine` is not the same as using what it returned.

    The three restart-resumption branches were each unpinned by the call-site test above: disabling
    them left it green, because it only asserted that `remine` ran. What the schedule owes is that
    the arms measured AFTER a promotion are measured against the residual the re-mine produced, so
    this asserts an evaluation is attributed to generation >= 1.
    """
    import importlib

    import toy_adapter
    importlib.reload(toy_adapter)
    toy = toy_adapter.ADAPTER
    toy.reset_expanded_signals()
    residual, events, states, _b, _f = _residual_fixture()

    from anchoropt.learning.policy_class import ThetaResult
    gen, offered, remines = {"n": 0}, [], []

    def ev(arm, theta=None):
        cell = _cell_of_arm(arm)
        offered.append((gen["n"], cell))
        if len(offered) > _HARD_CAP:
            raise _Stop(f"non-termination: {len(offered)} evaluations")
        # Promote once, off a SYNTHESIZED signal, and only in generation 0.
        win = gen["n"] == 0 and cell[1].startswith("payload_chars")
        return ThetaResult(theta={}, gains=("g1", "g2") if win else (), losses=(),
                           firings=3, cases_fired=3, n=12, interventions_executed=3)

    def rm():
        remines.append(gen["n"])
        gen["n"] += 1
        if gen["n"] > 2:
            return None
        class _R:
            key = f"gen{gen['n']}"
            case_ids = residual.case_ids
            rank = 1
        return _R()

    out = optimize_residual(residual, runtime=toy, host=toy.HOST, events=events, states=states,
                            evaluate=ev, promote=lambda a: None, remine=rm, max_restarts=3)

    assert out.state == IMPROVED
    assert remines, "never re-mined after promoting from expansion"
    after = [c for g, c in offered if g >= 1]
    assert after, ("re-mined and then threw the result away: nothing was measured against the new "
                   f"residual. restarts={out.restarts}, evaluations={len(offered)}")
    assert out.restarts >= 1, f"restart not counted: {out.restarts}"


def test_a_promotion_from_the_grounded_FRONTIER_also_restarts():
    """The third promotion site, pinned separately.

    The frontier sweep is a WHERE move under a fixed Phi: after HOW is exhausted at the current
    boundary it measures already-grounded arms at EARLIER ones. It promoted and returned without
    re-mining, and its two resumption branches were the last unpinned ones in this file.

    The fixture wins only on the toy host's shipped `nothing_proposed` signal at the earlier
    boundary, reached while the later boundary yields NO_BENEFIT -- which is the frontier's entry
    condition -- and only in generation 0, so a restart must show work in generation >= 1.
    """
    import importlib

    import toy_adapter
    importlib.reload(toy_adapter)
    toy = toy_adapter.ADAPTER
    toy.reset_expanded_signals()
    residual, events, states, _b, _f = _residual_fixture()

    from anchoropt.learning.policy_class import ThetaResult
    gen, offered, remines, frontier = {"n": 0}, [], [], []

    def ev(arm, theta=None):
        cell = _cell_of_arm(arm)
        offered.append((gen["n"], cell))
        if len(offered) > _HARD_CAP:
            raise _Stop(f"non-termination: {len(offered)} evaluations")
        win = gen["n"] == 0 and cell[0] == "post_generation_pre_exec" \
            and cell[1] == "nothing_proposed" and cell[2] == "suppress"
        return ThetaResult(theta={}, gains=("g1", "g2") if win else (), losses=(),
                           firings=3, cases_fired=3, n=12, interventions_executed=3)

    def rm():
        remines.append(gen["n"])
        gen["n"] += 1
        if gen["n"] > 2:
            return None
        class _R:
            key = f"gen{gen['n']}"
            case_ids = residual.case_ids
            rank = 1
        return _R()

    out = optimize_residual(residual, runtime=toy, host=toy.HOST, events=events, states=states,
                            evaluate=ev, promote=lambda a: None, remine=rm, max_restarts=3)
    frontier.extend(out.frontier_boundaries)

    assert out.state == IMPROVED
    assert frontier, "the frontier sweep was never entered, so this test proves nothing"
    assert remines, "promoted on the frontier without re-mining"
    assert [c for g, c in offered if g >= 1], (
        "re-mined after a frontier promotion but measured nothing against the new residual")
    # The promoting Attempt must be recorded EXACTLY ONCE, and must not then have a NO_BENEFIT
    # verdict banked over it. Falling through the frontier epilogue appends the SAME object a second
    # time and overwrites its state, so the round both double-counts its own work and reports the
    # boundary that just promoted as having produced no benefit.
    assert sum(1 for a in out.attempts if a is out.attempts[1]) == 1, (
        "the promoting Attempt was appended twice")
    promoting = [a for a in out.attempts if a.promoted is not None]
    assert len(promoting) == 1, f"expected one promoting attempt, got {len(promoting)}"
    assert promoting[0].state == IMPROVED, (
        f"a NO_BENEFIT verdict was banked over the attempt that promoted: {promoting[0].state}")
