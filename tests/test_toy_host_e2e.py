"""THE END-TO-END ACCEPTANCE TEST for the core algorithm, on a deterministic host.

WHY THIS FILE IS THE RELEASE GATE
---------------------------------
Every other test exercises one component. This one runs the whole loop the paper claims:

    residual -> backward WHERE localization -> WHAT expansion -> HOW grounding
             -> REAL paired evaluation -> acceptance -> moving incumbent -> RE-MINE

against `examples/toy_host`, which has no model, no sampling and no network, so the assertions are on
exact transitions rather than on "something happened". Nothing about the benchmark is load-bearing:
if core needed a BFCL concept, this file could not exist.

THE THREE OUTCOME CLASSES, kept apart. Collapsing any pair of them is how a null gets believed:
    UNEVALUATED  built and grounded, NOBODY RAN IT           -> REALIZABLE_UNMEASURED
    NO_BENEFIT   built, MEASURED, and it did not help        -> a real negative result
    BUDGET       arms left unmeasured when the budget ran out -> requeue, do not classify
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EX = REPO / "examples" / "toy_host"
for p in (str(REPO), str(EX)):
    if p not in sys.path:
        sys.path.insert(0, p)

from anchoropt.learning.policy_class import ThetaResult                        # noqa: E402
from anchoropt.learning.search_state import (  # noqa: E402
    SIGNAL_EXPANDED,                                  # noqa: E402
    IMPROVED, NO_BENEFIT, REALIZABLE_UNMEASURED,
)
from anchoropt.learning.structured_search import localize, optimize_residual   # noqa: E402


@pytest.fixture()
def toy():
    """A FRESH adapter per test: expanded signals and capability edits must not leak between tests."""
    import importlib

    import toy_adapter
    importlib.reload(toy_adapter)
    ad = toy_adapter.ADAPTER
    ad.reset_expanded_signals()
    return ad


def _baseline():
    from toy_host.runtime import run_corpus
    return run_corpus(None)


def _residual_and_events():
    base = _baseline()
    failing = sorted(c for c, ok in base["solved"].items() if not ok)
    events = [e for cid in failing for e in base["events"][cid]]
    states = [dict(e, boundary=("post_generation_pre_exec" if e["kind"] == "propose"
                                else "post_execution")) for e in events]

    class _R:
        key = "the filing is lost"
        case_ids = tuple(failing)
        rank = 1

    return _R(), events, states, base, failing


class _Ev:
    """The real paired evaluator: runs the host and compares per-case outcomes to the incumbent."""

    def __init__(self, adapter, incumbent=None, budget=None):
        from toy_host.runtime import run_corpus
        self.ad, self.budget = adapter, budget
        self.baseline = run_corpus(incumbent)
        self.n = 0
        self.exhausted = False
        self.log = []

    def controller_for(self, arm):
        from toy_host.runtime import Controller
        pred = self.ad.expanded.get(arm.signal)
        if pred is None:
            pred = lambda st, _s=arm.signal: bool(self.ad.evaluate_signal(_s, st))  # noqa: E731
        return Controller(boundary=arm.boundary.value, action=arm.action.value,
                          predicate=pred, eta=dict(arm.eta), label=arm.label)

    def __call__(self, arm, theta=None):
        from toy_host.runtime import run_corpus
        if self.budget is not None and self.n >= self.budget:
            self.exhausted = True
            return None
        self.n += 1
        got = run_corpus(self.controller_for(arm))
        gains = tuple(c for c, ok in got["solved"].items() if ok and not self.baseline["solved"][c])
        losses = tuple(c for c, ok in got["solved"].items() if not ok and self.baseline["solved"][c])
        r = ThetaResult(theta=dict(theta or {}), gains=gains, losses=losses,
                        firings=got["interventions_executed"],
                        cases_fired=len(got["cases_fired"]), n=got["n"],
                        interventions_executed=got["interventions_executed"],
                        accuracy_delta_pp=100.0 * (got["n_solved"] - self.baseline["n_solved"])
                        / max(1, got["n"]))
        self.log.append(r)
        return r


# ================================================================================================
# THE HOST ITSELF IS DETERMINISTIC AND DISCRIMINATING
# ================================================================================================

def test_the_toy_host_is_deterministic():
    """Byte-identical runs. Without this, nothing below is an acceptance test."""
    from toy_host.runtime import run_corpus
    a, b = run_corpus(None), run_corpus(None)
    assert a["solved"] == b["solved"] and a["n_solved"] == b["n_solved"]


def test_the_residual_is_real_and_the_host_can_discriminate_among_repairs():
    """A targeted controller must beat a fire-everywhere one, or MEASUREMENT does no work here.

    If every candidate scored the same, the demo would rubber-stamp whichever arm was built first and
    the "measurement chooses" claim would be untested.
    """
    from toy_host.runtime import Controller, run_corpus
    base = run_corpus(None)
    assert base["n_solved"] == 4 and base["n"] == 12

    targeted = run_corpus(Controller("post_generation_pre_exec", "suppress",
                                     lambda s: int(s.get("payload_chars") or 0) > 80,
                                     {"retry_budget": 1}))
    everywhere = run_corpus(Controller("post_generation_pre_exec", "suppress",
                                       lambda s: True, {"retry_budget": 1}))
    assert targeted["n_solved"] == 12
    assert everywhere["n_solved"] < targeted["n_solved"], \
        "an indiscriminate controller must measurably underperform a targeted one"


# ================================================================================================
# THE ALGORITHM
# ================================================================================================

def test_WHERE_localization_is_backward_and_derived_from_the_trajectory(toy):
    """Boundaries come from the events, ordered, and the search starts at the LATEST."""
    _r, events, _s, _b, _f = _residual_and_events()
    ordered = localize(events, runtime=toy)
    assert [b.key for b in ordered] == ["before_filing", "after_filing"], \
        "localization must be derived from the trajectory, earliest to latest"


def test_the_full_schedule_expands_PHI_and_moves_WHERE_only_after_exhausting_a_boundary(toy):
    """THE ORDERING CLAIM, measured. With no repair available at the later boundary the search must:

        exhaust HOW at the later boundary
        -> SWEEP THE GROUNDED FRONTIER: the same seeded Phi at the earlier boundary, measured
        -> only THEN expand WHAT -> retry HOW -> synthesize a signal the shipped Phi lacks -> improve

    THE FRONTIER SWEEP COMES FIRST, and that ordering is the rule rather than an implementation
    detail: a grounded arm is one attribution already supports, while a synthesized predicate is a
    guess that happens to discriminate. Spending the speculative move before the grounded one is
    backwards -- measured on a real round, a search whose one seeded arm was rejected expanded Phi
    into 40 invented atoms and never reached four already-grounded arms at an earlier boundary.

    Every step is driven by real paired evaluation, and the movement is observed rather than declared.
    """
    toy._CAPS = {k: v for k, v in toy._CAPS.items() if k[0] != "post_execution"}
    residual, events, states, base, _f = _residual_and_events()
    ev = _Ev(toy)

    out = optimize_residual(residual, runtime=toy, host=toy.HOST, events=events, states=states,
                            evaluate=ev, improves=lambda r: r is not None and r.net > 0)

    assert out.state == IMPROVED, [(a.boundary, a.state) for a in out.attempts]
    assert out.moves_earlier >= 1, "the search must have moved to the EARLIER boundary"
    # THE GROUNDED FRONTIER WAS SWEPT, and before any expansion: the earlier boundary was searched
    # under the SEEDED Phi first. `moves_earlier` counts every transition, so it is >= 1 rather than
    # exactly 1 once the frontier visit is included -- the ordering claim is what matters, and the
    # attempt sequence below pins it.
    assert out.frontier_boundaries, "the grounded frontier was never swept"
    seq = [(a.boundary, a.state) for a in out.attempts]
    frontier_at = next(i for i, a in enumerate(out.attempts)
                       if (a.detail or "").startswith("grounded frontier"))
    expanded_at = next((i for i, a in enumerate(out.attempts)
                        if a.state == SIGNAL_EXPANDED or a.signals_expanded), len(out.attempts))
    assert frontier_at < expanded_at, (
        f"Phi was expanded before the grounded frontier was swept: {seq}")
    assert out.signals_installed, "the gate's condition is not in shipped Phi; it must be SYNTHESIZED"
    assert out.promoted.boundary.value == "post_generation_pre_exec"
    assert out.promoted.signal in set(out.signals_installed), \
        "the accepted controller must run on the signal core installed, not a shipped one"

    # A boundary was genuinely exhausted before the move, and the gate's shipped-Phi arms were
    # MEASURED and rejected -- not skipped.
    assert any(a.state == NO_BENEFIT and a.candidates_evaluated > 0 for a in out.attempts), \
        "the shipped-Phi arms at the gate must be measured, not assumed"
    assert ev.n >= 4, f"every arm must be really evaluated, only {ev.n} were"


def test_acceptance_moves_the_incumbent_and_the_RESIDUAL_IS_REMINED(toy):
    """P0 -> P1 and then re-mine: the next anchor is fitted to the residual P1 leaves, not P0's."""
    residual, events, states, base, failing = _residual_and_events()
    ev = _Ev(toy)
    out = optimize_residual(residual, runtime=toy, host=toy.HOST, events=events, states=states,
                            evaluate=ev, improves=lambda r: r is not None and r.net > 0)
    assert out.state == IMPROVED and out.promoted is not None

    p1 = _Ev(toy, incumbent=ev.controller_for(out.promoted))
    assert p1.baseline["n_solved"] > base["n_solved"], "the incumbent must actually move"

    remined = sorted(c for c, ok in p1.baseline["solved"].items() if not ok)
    assert set(remined) < set(failing), "the re-mined residual must be strictly smaller than P0's"


# ================================================================================================
# THE THREE OUTCOME CLASSES
# ================================================================================================

def test_UNEVALUATED_is_not_NO_BENEFIT(toy):
    """No evaluator => REALIZABLE_UNMEASURED. Reporting NO_BENEFIT would assert a measurement that
    never happened -- the same error class as believing a null from a channel that was never live."""
    residual, events, states, _b, _f = _residual_and_events()
    out = optimize_residual(residual, runtime=toy, host=toy.HOST, events=events, states=states,
                            evaluate=None)
    assert out.state == REALIZABLE_UNMEASURED
    assert out.state != NO_BENEFIT
    assert out.candidates, "arms must still be BUILT -- the structural result is the point"
    assert all(a.candidates_evaluated == 0 for a in out.attempts)


def test_NO_BENEFIT_requires_an_actual_MEASUREMENT(toy):
    """An evaluator that measures every arm honestly and finds nothing must yield NO_BENEFIT.

    The distinction from the test above is the whole point: same arms, same host, and the only
    difference is whether anything was RUN.
    """
    residual, events, states, _b, _f = _residual_and_events()
    seen = {"n": 0}

    def flat(arm, theta=None):
        seen["n"] += 1
        return ThetaResult(theta={}, gains=(), losses=(), firings=1, cases_fired=1, n=12,
                           interventions_executed=1, accuracy_delta_pp=0.0)

    out = optimize_residual(residual, runtime=toy, host=toy.HOST, events=events, states=states,
                            evaluate=flat, improves=lambda r: r is not None and r.net > 0)
    assert seen["n"] > 0, "arms must have been measured"
    assert out.state in (NO_BENEFIT, "BOUNDARY_EXHAUSTED", "SIGNAL_EXPANSION_EXHAUSTED"), out.state
    assert out.state != REALIZABLE_UNMEASURED, \
        "arms WERE measured, so this is a negative result and not a structural one"
    assert any(a.candidates_evaluated > 0 for a in out.attempts)


def test_BUDGET_EXHAUSTION_leaves_arms_UNMEASURED_and_is_not_a_negative_result(toy):
    """A budget-capped evaluator returns None for arms it did not run.

    `None` must not be scored as 0: that would rank an unmeasured arm against measured ones. The
    search treats it as no result, and the caller reports exhaustion -- requeue, not classify.
    """
    residual, events, states, _b, _f = _residual_and_events()
    ev = _Ev(toy, budget=0)
    out = optimize_residual(residual, runtime=toy, host=toy.HOST, events=events, states=states,
                            evaluate=ev, improves=lambda r: r is not None and r.net > 0)
    assert ev.exhausted
    assert ev.log == [], "nothing may be measured under a zero budget"
    assert out.state != NO_BENEFIT, \
        "a budget-exhausted search has measured nothing; calling it NO_BENEFIT banks a null it "\
        "never observed"
    assert out.state == REALIZABLE_UNMEASURED


def test_an_unmeasured_arm_is_never_ranked_against_a_measured_one(toy):
    """`None` from the evaluator must be skipped, not coerced to a score."""
    residual, events, states, _b, _f = _residual_and_events()
    calls = {"n": 0}

    def half(arm, theta=None):
        calls["n"] += 1
        if calls["n"] % 2:
            return None                       # unmeasured
        return ThetaResult(theta={}, gains=(), losses=("c07",), firings=1, cases_fired=1, n=12,
                           interventions_executed=1, accuracy_delta_pp=-8.3)

    out = optimize_residual(residual, runtime=toy, host=toy.HOST, events=events, states=states,
                            evaluate=half, improves=lambda r: r is not None and r.net > 0)
    assert out.state != IMPROVED, "a net-negative measured arm must not be accepted"


# ================================================================================================
# THE OBJECTIVE-COMPARISON DEFECT THIS EXAMPLE EXPOSED
# ================================================================================================

def test_the_schedule_ranks_ThetaResult_by_J_train_and_not_by_raw_comparison(toy):
    """REGRESSION. The schedule compared evaluator outputs with `>`, which raises TypeError for a
    `ThetaResult` -- the exact type the real optimizer produces. So the only evaluator that could run
    end to end was a scalar stub, and the first REAL paired evaluation crashed.

    Ranking now goes through `train_objective`, which is the project's single definition of J_train
    (net, then ENGAGEMENT, then selectivity, then a stable tiebreak).
    """
    from anchoropt.learning.structured_search import _objective_key

    worse = ThetaResult(theta={}, gains=("a",), losses=("b", "c"), firings=1, cases_fired=1, n=10,
                        interventions_executed=1)
    better = ThetaResult(theta={}, gains=("a", "b"), losses=(), firings=1, cases_fired=1, n=10,
                         interventions_executed=1)
    assert _objective_key(better) > _objective_key(worse)

    # A scalar evaluator must keep working unchanged.
    assert _objective_key(3) > _objective_key(1)

    # ENGAGEMENT breaks a net tie: an arm that never executed cannot outrank one that did.
    inert = ThetaResult(theta={}, gains=(), losses=(), firings=0, cases_fired=0, n=10,
                        interventions_executed=0)
    engaged = ThetaResult(theta={}, gains=(), losses=(), firings=2, cases_fired=2, n=10,
                          interventions_executed=2)
    assert _objective_key(engaged) > _objective_key(inert)


# ================================================================================================
# THE DEMO RUNS, AND ITS OUTPUT IS STABLE
# ================================================================================================

@pytest.mark.parametrize("flags,expect", [
    ([], "ACCEPTED"),
    # The grounded-frontier sweep adds a boundary visit before any expansion, so the transition
    # count is no longer 1. What the demo must show is that the sweep HAPPENED and the search still
    # moved earlier -- the count itself is not the claim.
    (["--no-late-repair"], "GROUNDED FRONTIER swept before any expansion"),
    (["--no-evaluate"], "UNEVALUATED"),
    (["--budget", "0"], "BUDGET EXHAUSTED"),
])
def test_the_demo_script_runs_and_reports_the_right_class(flags, expect):
    """A collaborator's first command must work from a clean checkout."""
    out = subprocess.run([sys.executable, str(EX / "demo.py"), *flags],
                         capture_output=True, text=True, cwd=str(REPO), timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert expect in out.stdout, out.stdout[-2000:]


def test_the_demo_is_byte_identical_across_runs():
    """Determinism, at the level a collaborator sees it."""
    runs = [subprocess.run([sys.executable, str(EX / "demo.py"), "--no-late-repair"],
                           capture_output=True, text=True, cwd=str(REPO), timeout=300).stdout
            for _ in range(2)]
    assert runs[0] == runs[1], "the toy example must be reproducible byte for byte"


# ================================================================================================
# SUPPRESS AND REPROMPT MUST BE GENUINELY DIFFERENT INTERVENTIONS
# ================================================================================================
#
# If the host manufactures a corrective instruction for a controller that supplied none, the two
# families measure identically for a reason the host invented -- and any comparison between them is
# meaningless. These tests pin the separation from both sides.

def test_SUPPRESS_repairs_by_OMISSION_and_is_given_no_instruction():
    """REGRESSION. This file once read `hint or "Archive it instead"` at the retry site, so a
    suppression supplying NO instruction was handed the reprompt's text anyway -- making SUPPRESS
    secretly a REPROMPT.

    It is the same error the project already retracted once (R10): attributing a repair to a corrective
    instruction when the instruction was inert and the real cause was omission. The agent must re-plan
    from the ABSENCE of its own call, with nothing said to it.
    """
    from toy_host.runtime import Controller, ToyRuntime, run_corpus

    sup = Controller("post_generation_pre_exec", "suppress",
                     lambda s: int(s.get("payload_chars") or 0) > 80, {"retry_budget": 1})
    assert dict(sup.eta) == {"retry_budget": 1}, \
        "a suppression carries NO instruction -- that is what distinguishes it"
    assert run_corpus(sup)["n_solved"] == 12, "omission alone must be sufficient to repair"

    # And the agent's policy must react to `blocked` WITHOUT any hint text.
    call = ToyRuntime._propose("x" * 200, hint="", blocked=("file_report",))
    assert call["tool"] == "archive_report", \
        "the agent must re-plan from the absence of its filing, with no instruction"


def test_REPROMPT_depends_on_the_instruction_it_actually_declares():
    """The other side: an EMPTY instruction must repair NOTHING.

    This is the property the real upstream cell FAILED (D3) -- there, two arms differing only in
    `instruction` were byte-identical, proving the text never reached the model. Here it does reach the
    agent, so content changes the outcome, and a contentless reprompt is measurably inert.
    """
    from toy_host.runtime import Controller, run_corpus

    def rp(text):
        return run_corpus(Controller(
            "post_generation_pre_exec", "reprompt",
            lambda s: int(s.get("payload_chars") or 0) > 80,
            {"instruction": text, "retry_budget": 1}))["n_solved"]

    assert rp("") == 4, "an empty instruction must repair nothing -- otherwise the text is inert"
    assert rp("The drawer rejects long reports. Archive it instead.") == 12
    assert rp("Shorten the report before filing it.") == 12


def test_the_two_reprompt_variants_produce_DIFFERENT_trajectories():
    """Two semantically distinct instructions must not be byte-identical.

    Identical trajectories across instruction variants is precisely the evidence that condemned the
    upstream cell. Here the variants reach the same SCORE by different routes, and the stored artifacts
    differ -- which is what makes measuring more than one variant worthwhile.
    """
    from toy_host.runtime import Controller, run_corpus

    def store(text):
        c = Controller("post_generation_pre_exec", "reprompt",
                       lambda s: int(s.get("payload_chars") or 0) > 80,
                       {"instruction": text, "retry_budget": 1})
        got = run_corpus(c)
        return [e for cid in sorted(got["events"]) for e in got["events"][cid]]

    archive = store("The drawer rejects long reports. Archive it instead.")
    shorten = store("Shorten the report before filing it.")
    assert archive != shorten, \
        "instruction variants that produce identical trajectories would be an inert parameter"


def test_suppression_and_reprompt_take_DIFFERENT_routes_to_their_repair():
    """Same score, different mechanism -- and the artifacts prove which one ran."""
    from toy_host.runtime import Controller, run_corpus

    fires = lambda s: int(s.get("payload_chars") or 0) > 80  # noqa: E731
    sup = run_corpus(Controller("post_generation_pre_exec", "suppress", fires,
                                {"retry_budget": 1}))
    rp = run_corpus(Controller("post_generation_pre_exec", "reprompt", fires,
                               {"instruction": "Shorten the report before filing it.",
                                "retry_budget": 1}))
    assert sup["n_solved"] == rp["n_solved"] == 12
    # The SHORTEN reprompt files a truncated report into the drawer; suppression archives it whole.
    sup_ev = [e for cid in sorted(sup["events"]) for e in sup["events"][cid]]
    rp_ev = [e for cid in sorted(rp["events"]) for e in rp["events"][cid]]
    assert sup_ev != rp_ev, "two different mechanisms must leave different traces"


def test_the_budget_is_PER_EPISODE_and_exhaustion_degrades_to_control():
    """The semantics the suppress capability declares via `consumes=("retry_budget",)`.

    Both halves were once wrong elsewhere: a budget read from per-step state never binds (an unbounded
    loop that looks bounded), and on exhaustion the withheld call must DISPATCH rather than be silently
    dropped -- the arm degrades to control instead of losing the operation.
    """
    from toy_host.runtime import Controller, Episode, ToyRuntime

    long_report = "A detailed incident report. " * 6
    ctl = Controller("post_generation_pre_exec", "suppress",
                     lambda s: int(s.get("payload_chars") or 0) > 80, {"retry_budget": 1})
    rt = ToyRuntime()
    # Two long reports in ONE episode: the first is withheld, the second exhausts the budget.
    out = rt.run_episode(Episode("e1", (long_report, long_report)), ctl)
    assert rt.budget_spent["e1"] == 1, "the budget is spent once per EPISODE, not once per step"
    # The second filing was NOT withheld, so it hit the drawer and was rejected -- control behaviour.
    assert any(e["kind"] == "execute" and e.get("error_kind") == "drawer_rejected_oversize"
               for e in out["events"]), \
        "on exhaustion the withheld call must dispatch: the arm degrades to control"


def test_evaluation_counts_reflect_COMPLETED_evaluations_only(toy):
    """Under a partial budget, the count must equal what actually ran -- not what was attempted."""
    # The later boundary has no repair, so the search must build and measure SEVERAL arms at the gate
    # -- more than the budget allows. With the full capability set it improves on the first arm and
    # never reaches the cap, which would make the assertion vacuous.
    toy._CAPS = {k: v for k, v in toy._CAPS.items() if k[0] != "post_execution"}
    residual, events, states, _b, _f = _residual_and_events()
    ev = _Ev(toy, budget=2)
    optimize_residual(residual, runtime=toy, host=toy.HOST, events=events, states=states,
                      evaluate=ev, improves=lambda r: r is not None and r.net > 0)
    assert ev.n == 2, f"exactly the budget must be consumed, got {ev.n}"
    assert len(ev.log) == 2, "the log must hold one entry per COMPLETED evaluation"
    assert ev.exhausted, "and exhaustion must be reported"
    # THE COUNT IS OF COMPLETED EVALUATIONS. Attempts beyond the budget returned None and must not be
    # counted, logged, or scored -- an unmeasured arm has no result.
    assert all(r is not None for r in ev.log)


def test_the_accepted_arm_is_the_OPTIMIZER_argmax_not_the_first_positive_one(toy):
    """SELECTION CONSISTENCY. `AnchorPolicyOpt.optimize` must return the J_train argmax.

    `optimize_residual` accepts the first arm that satisfies `improves` -- correct for the SCHEDULE,
    whose job is to find a repair and move on. But the POLICY round's job is to pick the best arm among
    counterfactuals, and this asserts the optimizer does that rather than short-circuiting: a
    deliberately better arm presented LAST must still win.
    """
    from anchoropt.learning.anchor_policy_opt import (
        AnchorPolicyOpt, FrozenIncumbent, SearchSpaceProposal,
    )
    from anchoropt.anchor import Action, IncisionPoint

    ev = _Ev(toy)
    prop = SearchSpaceProposal(
        boundary=IncisionPoint.POST_GENERATION_PRE_EXEC, signal="nothing_proposed",
        action_set=(Action.REPROMPT, Action.SUPPRESS), diagnosis_case_ids=("c01",),
        preferred_action="suppress")          # PROVENANCE ONLY -- must not decide the winner

    scores = {}

    def scored(arm, theta=None):
        # Rank arms by an explicit, deliberately non-first-wins ordering: the LAST reprompt variant is
        # best. If selection short-circuited on the first positive arm, it would pick the other one.
        net = {"reprompt:advise_archive": 1, "reprompt:advise_shorten": 5,
               "suppress:cancel_proposed": 2}.get(arm.instantiated.label, 0)
        r = ThetaResult(theta=dict(theta or {}), gains=tuple(f"g{i}" for i in range(net)),
                        losses=(), firings=1, cases_fired=1, n=12, interventions_executed=1)
        scores[arm.instantiated.label] = r.net
        return r

    res = AnchorPolicyOpt(runtime=toy, host=toy.HOST).optimize(
        prop, incumbent=FrozenIncumbent("P0", "tok"), evaluate=scored)

    assert res.winner is not None, res.rejected
    assert res.winner.instantiated.label == "reprompt:advise_shorten", \
        f"the argmax of J_train must win, not the first positive arm: {scores}"
    assert res.preference_was_right is False, \
        "the proposer preferred suppress and measurement overruled it -- recorded, not obeyed"
