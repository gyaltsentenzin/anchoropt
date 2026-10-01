"""A MEASURED no-benefit is what licenses Phi expansion -- and optimization then resumes.

The pair of invariants this file holds together:

  * without measurement the search halts (tests/test_no_evaluator_no_inference.py);
  * WITH a measured verdict that the seeded representation does not help, the search must widen Phi
    AT THE SAME BOUNDARY and keep going -- otherwise the halt above turns a seed into a cage, and a
    round that measured honestly would be indistinguishable from one that gave up.

The defect this guards against is the mirror of the stub-evaluator bug: fixing "expansion fires on a
manufactured verdict" by never expanding at all. Both are failures of the same rule -- expansion is a
decision with a trigger, and the trigger is measurement.

Benchmark-independent: a hand-built runtime with two boundaries, three declared signals, and an
expansion hook that offers one synthesized signal.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.learning.search_state import (
    IMPROVED, NO_BENEFIT, REALIZABLE_UNMEASURED, SIGNAL_EXPANDED,
)
from anchoropt.learning.structured_search import optimize_residual


class _Residual:
    def __init__(self, observables=("alpha",)):
        self.observables = tuple(observables)
        self.case_ids = ("c1", "c2")
        self.key = "a grouped family"
        self.support = 2
        self.rank = 1


EVENTS = [{"kind": "decision", "boundary": "late", "step": 0}]


class _Runtime:
    """One boundary. `expand_attribution` offers a synthesized signal the seeded Phi did not contain."""

    EXPANDED_SIGNALS: dict = {}

    def __init__(self):
        self.installed: list[str] = []
        self.EXPANDED_SIGNALS = {}

    def declared_signals(self):
        return ("alpha",)

    def signal_boundaries(self, name):
        return frozenset({IncisionPoint.POST_EXECUTION})

    def is_decision(self, event):
        return True

    def boundary_key(self, event):
        return "late"

    def boundary_from_key(self, key):
        return IncisionPoint.POST_EXECUTION

    def evaluate_signal(self, name, state, params):
        return True

    def states_at(self, boundary, states):
        return list(states or ())

    def ground_reprompt(self, signal, boundary):
        return [{"variant": "advise", "eta": {"instruction": "try the other thing",
                                              "retry_budget": 1}}]

    def executor_capabilities(self, boundary, action):
        from anchoropt.learning.executor_capability import ExecutorCapability
        return (ExecutorCapability(
            boundary=str(getattr(boundary, "value", boundary)),
            action=str(getattr(action, "value", action)),
            capability_id="fake_injector", binding="a test double that injects",
            consumes=("instruction", "retry_budget"),
            signals=("alpha",), signal_agnostic=True, fixed={"retry_budget": 1}),)

    # --- EXPANSION. Core synthesizes predicates over the fields a boundary DECLARES, using its own
    # grammar; the runtime's part is to declare the fields and to install what core hands back.
    def synthesis_fields(self, boundary=None):
        class _D:
            def __init__(self, name, typ):
                self.name, self.type, self.boundaries, self.doc = name, typ, (), ""
        return {"beta": _D("beta", bool)}

    def install_signal(self, name, pred, **k):
        self.installed.append(name)
        self.EXPANDED_SIGNALS[name] = pred
        return name

    def expanded_signal_names(self):
        return tuple(self.installed)


class _Pred:
    field = "beta"

    def evaluate(self, state):
        return bool(state.get("beta"))

    fires_on = evaluate


class _Host:
    def require(self, boundary, action):
        return True

    def executable_actions(self, boundary):
        return (Action.REPROMPT,)


class _R:
    """A measured result. ORDERED, because core ranks whatever the evaluator returns and an
    unorderable object raises -- the same TypeError that once meant only a scalar stub could run."""

    def __init__(self, net):
        self.net = net

    def __lt__(self, other):
        return self.net < getattr(other, "net", other)

    def __gt__(self, other):
        return self.net > getattr(other, "net", other)

    def __eq__(self, other):
        return self.net == getattr(other, "net", other)


class _Ev:
    """Scores every arm. `good` names the signal whose arm is beneficial; others measure 0."""

    def __init__(self, good=None):
        self.good = good
        self.seen: list[str] = []

    def __call__(self, arm):
        self.seen.append(arm.signal)
        return _R(1 if (self.good and arm.signal == self.good) else 0)


def _run(ev, **kw):
    rt = _Runtime()
    out = optimize_residual(_Residual(), runtime=rt, host=_Host(), events=EVENTS,
                            states=[{"alpha": True, "beta": True},
                                    {"alpha": False, "beta": False}],
                            evaluate=ev, improves=lambda o: getattr(o, "net", 0) > 0, **kw)
    return out, rt


def test_a_measured_no_benefit_EXPANDS_phi_at_the_same_boundary():
    """The seeded signal measures 0, so the representation is shown insufficient BY MEASUREMENT."""
    ev = _Ev(good=None)
    out, rt = _run(ev)
    assert "alpha" in ev.seen, "the seeded signal was never measured"
    assert out.signals_installed, f"a measured no-benefit did not widen Phi: {out.state}"
    # Core names the synthesized signals itself, from the declared FIELD -- the test must not
    # prescribe the name, only that a new one was installed and it reads the widened field.
    assert any("beta" in n for n in out.signals_installed), out.signals_installed
    # The EXPANSION attempt records the verdict it reached (NO_BENEFIT here -- the widened signal was
    # measured and did not help), not SIGNAL_EXPANDED. `signals_installed` is what evidences the
    # widening; asserting on the attempt state would pin a label rather than the behaviour.
    assert len(out.attempts) >= 2, (
        f"only one attempt: Phi was widened but not re-searched -- {[a.state for a in out.attempts]}")
    # SAME boundary: widening precedes moving earlier.
    assert set(a.boundary for a in out.attempts) == {"late"}


def test_optimization_RESUMES_on_the_widened_phi():
    """Expansion is not the end state: the new signal's arms must actually be measured."""
    ev = _Ev(good=None)
    out, _ = _run(ev)
    assert any("beta" in n for n in ev.seen), (
        f"Phi widened but the new signal's arms were never evaluated: measured {ev.seen}")
    assert out.evaluations_completed >= 2, out.evaluations_completed


def test_a_beneficial_EXPANDED_arm_is_promoted():
    """The whole point of widening: a controller found there can win."""
    ev = _Ev(good="beta")                     # the name core synthesizes from the declared field
    out, _ = _run(ev)
    assert out.state == IMPROVED, out.state
    assert out.promoted is not None
    assert "beta" in out.promoted.signal, out.promoted.signal


def test_a_beneficial_SEEDED_arm_short_circuits_expansion():
    """If attribution's own representation works, there is nothing to widen."""
    ev = _Ev(good="alpha")
    out, _ = _run(ev)
    assert out.state == IMPROVED, out.state
    assert out.promoted.signal == "alpha"
    assert out.signals_installed == [], "Phi was widened despite a beneficial seeded arm"


def test_the_expansion_is_NOT_reached_when_the_budget_stops_the_round_first():
    """A spend limit must not be laundered into a licence to widen."""
    ev = _Ev(good=None)
    out, _ = _run(ev, eval_budget=1)
    assert out.evaluations_completed == 1, out.evaluations_completed
    # The seeded arm was measured and scored 0, but the round STOPPED on the budget before the
    # no-benefit verdict could be reached over the whole seeded set -- so nothing licenses widening.
    assert out.state != NO_BENEFIT, out.state
    assert out.state != IMPROVED


def test_without_an_evaluator_the_same_setup_does_NOT_expand():
    """The control for this whole file: identical runtime, no evaluator, no widening."""
    rt = _Runtime()
    out = optimize_residual(_Residual(), runtime=rt, host=_Host(), events=EVENTS,
                            states=[{"alpha": True, "beta": True}])
    assert out.state == REALIZABLE_UNMEASURED
    assert out.signals_installed == []
    assert rt.installed == []
