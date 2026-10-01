"""WITHOUT MEASUREMENT THE SEARCH MAY NOT INFER ANYTHING. Benchmark-independent.

The defect this closes, measured on a real H0 round: the propose phase passed a stub evaluator
(`evaluate=lambda _a: 0, improves=lambda _o: False`). Every candidate scored identically, `improves`
was never true, and the round therefore declared the seeded Phi NO_BENEFIT *without running
anything*. Phi expansion then fired on that manufactured verdict and re-enumerated the grammar: a
search seeded with 4 grounded observables emitted 117 arms, 113 of them from expansion.

Expanding the representation is a decision with a trigger, and the trigger is MEASURED
insufficiency. Backward movement has the same requirement: a boundary is exhausted when its
candidates were measured and none helped -- not when nobody looked.

So with `evaluate=None` the search must keep its boundary, its seeded Phi and its candidate set, and
report REALIZABLE_UNMEASURED. Three things it must NOT do: expand, move earlier, or promote.

No benchmark, no adapter, no model: a hand-built runtime with two boundaries and five signals.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.learning.search_state import (
    BOUNDARY_EXHAUSTED, IMPROVED, NO_BENEFIT, REALIZABLE_UNMEASURED, SEARCH_BUDGET_EXHAUSTED,
    SIGNAL_EXPANDED,
)
from anchoropt.learning.structured_search import optimize_residual


class _Residual:
    def __init__(self, observables=("alpha",)):
        self.observables = tuple(observables)
        self.case_ids = ("c1", "c2")
        self.key = "a grouped family"
        self.support = 2
        self.rank = 1


# TWO boundaries, so backward movement is observable if it happens.
EVENTS = [{"kind": "decision", "boundary": "early", "step": 0},
          {"kind": "decision", "boundary": "late", "step": 1}]


class _Runtime:
    def __init__(self):
        self.expansion_calls = 0

    def declared_signals(self):
        return ("alpha", "beta", "gamma")

    def signal_boundaries(self, name):
        return frozenset({IncisionPoint.POST_EXECUTION, IncisionPoint.POST_GENERATION_PRE_EXEC})

    def is_decision(self, event):
        return True

    def boundary_key(self, event):
        return event["boundary"]

    def boundary_from_key(self, key):
        return (IncisionPoint.POST_EXECUTION if key == "late"
                else IncisionPoint.POST_GENERATION_PRE_EXEC)

    def evaluate_signal(self, name, state, params):
        return True

    def states_at(self, boundary, states):
        return list(states or ())

    # --- grounding: one reprompt variant, so a candidate can actually be built
    def ground_reprompt(self, signal, boundary):
        return [{"variant": "advise", "eta": {"instruction": "do the other thing",
                                              "retry_budget": 1}}]

    def executor_capabilities(self, boundary, action):
        from anchoropt.learning.executor_capability import ExecutorCapability
        return (ExecutorCapability(
            boundary=str(getattr(boundary, "value", boundary)),
            action=str(getattr(action, "value", action)),
            capability_id="fake_injector", binding="a test double that injects",
            consumes=("instruction", "retry_budget"), signals=self.declared_signals(),
            fixed={"retry_budget": 1}),)

    # If the search tries to expand, we will see it here.
    def expand_signals(self, *a, **k):
        self.expansion_calls += 1
        return []

    def install_signal(self, *a, **k):
        self.expansion_calls += 1
        return None


class _Host:
    def require(self, boundary, action):
        return True

    def executable_actions(self, boundary):
        return (Action.REPROMPT,)


def _run(**kw):
    rt = _Runtime()
    out = optimize_residual(_Residual(), runtime=rt, host=_Host(), events=EVENTS,
                            states=[{"alpha": True}, {"alpha": False}], **kw)
    return out, rt


# ------------------------------------------------------- no evaluator: no inference of any kind

def test_no_evaluator_yields_REALIZABLE_UNMEASURED():
    out, _ = _run()
    assert out.state == REALIZABLE_UNMEASURED, out.state


def test_no_evaluator_NEVER_reports_NO_BENEFIT():
    """NO_BENEFIT asserts a measurement. Without one it is a fabricated negative result."""
    out, _ = _run()
    assert out.state != NO_BENEFIT
    assert all(a.state != NO_BENEFIT for a in out.attempts), [a.state for a in out.attempts]


def test_no_evaluator_does_NOT_expand_phi():
    out, rt = _run()
    assert out.signals_installed == [], out.signals_installed
    assert not out.expanded
    assert rt.expansion_calls == 0, "expansion was attempted without a measured verdict"
    assert all(a.state != SIGNAL_EXPANDED for a in out.attempts)


def test_no_evaluator_does_NOT_move_earlier():
    """A boundary is exhausted by measurement, not by absence of it.

    THE INVARIANT IS ABOUT EXHAUSTION, NOT ABOUT VISITING. This used to assert that exactly ONE
    boundary was visited, which is a stronger claim than the invariant needs and was measured wrong:
    the proposer localized a signal at an EARLIER boundary, the search stopped at the later one, and the
    mechanism behind 63 failing queries never had a single arm built. Collecting candidates at a
    boundary the seeded Phi is observable at is not a verdict about either boundary.

    So: no boundary may be marked exhausted, nothing may be promoted, and `moves_earlier` -- which
    counts the schedule GIVING UP on a boundary -- must stay 0. Visiting the localized boundaries is
    recorded separately, as `localization_sweep`.
    """
    out, _ = _run()
    assert out.moves_earlier == 0, out.moves_earlier
    assert out.promoted is None
    assert not any(a.state == BOUNDARY_EXHAUSTED for a in out.attempts), \
        "a boundary was declared exhausted without any measurement"
    # Every boundary beyond the first must be part of the localization sweep, never a give-up move.
    swept = set(out.localization_sweep)
    extra = [a.boundary for a in out.attempts[1:] if a.boundary not in swept]
    assert not extra, f"visited {extra} outside the localization sweep and without measuring"


def test_no_evaluator_does_NOT_promote():
    out, _ = _run()
    assert out.promoted is None
    assert out.state != IMPROVED


def test_no_evaluator_PRESERVES_the_seeded_phi_and_candidates():
    """The candidate set is the deliverable: the caller measures it and comes back."""
    out, _ = _run()
    assert out.phi_source == "residual_family"
    assert tuple(out.phi_seeded) == ("alpha",)
    assert out.candidates, "the built candidates were discarded"


def test_no_evaluator_reports_zero_completed_evaluations():
    out, _ = _run()
    assert out.evaluations_completed == 0


def test_the_attempt_says_WHY_it_stopped():
    """A reader must not have to infer that the halt was about measurement."""
    out, _ = _run()
    detail = " ".join(a.detail or "" for a in out.attempts).lower()
    assert "evaluator" in detail and "expansion" in detail, detail


# ------------------------------------------------------- budget: a spend limit, not a verdict

class _Ev:
    """Scores every arm, so only the budget can stop the search."""

    def __init__(self):
        self.calls = 0

    def __call__(self, arm):
        self.calls += 1
        class R:
            net = 0
        return R()


def test_a_spent_budget_is_BUDGET_EXHAUSTED_not_NO_BENEFIT():
    ev = _Ev()
    out, _ = _run(evaluate=ev, improves=lambda o: getattr(o, "net", 0) > 0, eval_budget=1)
    assert out.state == SEARCH_BUDGET_EXHAUSTED, out.state
    assert out.state != NO_BENEFIT


def test_the_budget_actually_binds():
    ev = _Ev()
    out, _ = _run(evaluate=ev, improves=lambda o: getattr(o, "net", 0) > 0, eval_budget=1)
    assert out.evaluations_completed == 1, out.evaluations_completed
    assert out.budget == 1


def test_an_unbudgeted_run_is_unaffected():
    """Budget is opt-in: absent one, behaviour is exactly as before."""
    ev = _Ev()
    out, _ = _run(evaluate=ev, improves=lambda o: getattr(o, "net", 0) > 0)
    assert out.state != SEARCH_BUDGET_EXHAUSTED
    assert out.budget is None


def test_declined_evaluations_do_not_consume_budget():
    """An evaluator returning None has not measured, so it has not spent."""
    out, _ = _run(evaluate=lambda a: None, improves=lambda o: False, eval_budget=2)
    assert out.evaluations_completed == 0
    assert out.state != SEARCH_BUDGET_EXHAUSTED


def test_the_budget_state_is_DONE_not_an_advance():
    """BUDGET_EXHAUSTED requeues the round; it must not advance a coordinate on no verdict."""
    from anchoropt.learning.search_state import DONE, next_coordinate
    assert next_coordinate(SEARCH_BUDGET_EXHAUSTED) == DONE
