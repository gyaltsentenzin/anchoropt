"""EXHAUST THE GROUNDED FRONTIER BEFORE SYNTHESIZING A NEW REPRESENTATION.

The rule, general: for one residual family, search every feasible WHAT x HOW its ATTRIBUTED Phi
already supports, across all consequential boundaries, before widening Phi through the grammar. An
attributed signal is evidence about this failure; a synthesized predicate is a guess that happens to
discriminate. Spending the speculative move first is backwards.

Measured on a real round, which is why this exists: the one seeded arm at the latest boundary was
measured at net -3, and the schedule immediately expanded Phi into 40 synthesized atoms AT THE SAME
boundary -- never reaching four already-grounded arms at an earlier one, reachable from the same
seeded Phi with no synthesis at all. HOW was exhausted there; the REPRESENTATION was not the thing
that had run out.

Benchmark-independent: two boundaries, a hand-built runtime, no adapter and no model.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.learning.search_state import (
    IMPROVED, NO_BENEFIT, REALIZABLE_UNMEASURED, SEARCH_BUDGET_EXHAUSTED,
)
from anchoropt.learning.structured_search import optimize_residual

LATE, EARLY = "late", "early"


class _Residual:
    """Two attributed observables: one visible at each boundary."""

    observables = ("late_sig", "early_sig")
    case_ids = ("c1", "c2")
    key = "one grouped family"
    support = 2
    rank = 1


EVENTS = [{"kind": "decision", "boundary": EARLY, "step": 0},
          {"kind": "decision", "boundary": LATE, "step": 1}]


class _Runtime:
    """`late_sig` is observable only at the late boundary, `early_sig` only at the early one."""

    EXPANDED_SIGNALS: dict = {}

    def __init__(self):
        self.installed: list[str] = []
        self.EXPANDED_SIGNALS = {}

    def declared_signals(self):
        return ("late_sig", "early_sig")

    def signal_boundaries(self, name):
        return frozenset({IncisionPoint.POST_EXECUTION} if name == "late_sig"
                         else {IncisionPoint.POST_GENERATION_PRE_EXEC})

    def is_decision(self, event):
        return True

    def boundary_key(self, event):
        return event["boundary"]

    def boundary_from_key(self, key):
        return (IncisionPoint.POST_EXECUTION if key == LATE
                else IncisionPoint.POST_GENERATION_PRE_EXEC)

    def evaluate_signal(self, name, state, params):
        return True

    def states_at(self, boundary, states):
        return list(states or ())

    def ground_reprompt(self, signal, boundary):
        return [{"variant": "advise", "eta": {"instruction": "do it differently",
                                              "retry_budget": 1}}]

    def executor_capabilities(self, boundary, action):
        from anchoropt.learning.executor_capability import ExecutorCapability
        return (ExecutorCapability(
            boundary=str(getattr(boundary, "value", boundary)),
            action=str(getattr(action, "value", action)),
            capability_id="fake_injector", binding="a test double that injects",
            consumes=("instruction", "retry_budget"),
            signals=("late_sig", "early_sig"), signal_agnostic=True,
            fixed={"retry_budget": 1}),)

    # Expansion is available -- the point is that it is not REACHED until the frontier is done.
    def synthesis_fields(self, boundary=None):
        class _D:
            def __init__(self, name, typ):
                self.name, self.type, self.boundaries, self.doc = name, typ, (), ""
        return {"invented": _D("invented", bool)}

    def install_signal(self, name, pred, **k):
        self.installed.append(name)
        self.EXPANDED_SIGNALS[name] = pred
        return name

    def expanded_signal_names(self):
        return tuple(self.installed)


class _Host:
    def require(self, boundary, action):
        return True

    def executable_actions(self, boundary):
        return (Action.REPROMPT,)


class _R:
    def __init__(self, net):
        self.net = net

    def __lt__(self, o): return self.net < getattr(o, "net", o)
    def __gt__(self, o): return self.net > getattr(o, "net", o)
    def __eq__(self, o): return self.net == getattr(o, "net", o)


class _Ev:
    """`scores` maps a signal name to a net. A signal absent from it is DECLINED (None)."""

    def __init__(self, scores):
        self.scores = dict(scores)
        self.seen: list[str] = []

    def __call__(self, arm):
        self.seen.append(arm.signal)
        if arm.signal in self.scores:
            return _R(self.scores[arm.signal])
        return None


def _run(ev, **kw):
    rt = _Runtime()
    out = optimize_residual(_Residual(), runtime=rt, host=_Host(), events=EVENTS,
                            states=[{"late_sig": True, "early_sig": True, "invented": True},
                                    {"late_sig": False, "early_sig": False, "invented": False}],
                            evaluate=ev, improves=lambda o: getattr(o, "net", 0) > 0, **kw)
    return out, rt


# ============================================================ THE SEQUENCE THIS FILE EXISTS FOR

def test_a_negative_at_the_LATE_boundary_generates_EARLY_arms_from_the_SAME_phi():
    """late measured negative -> earlier-boundary arms from the existing Phi -> NO expansion yet."""
    ev = _Ev({"late_sig": -3})                       # early_sig is DECLINED, not scored
    out, rt = _run(ev)

    assert "late_sig" in ev.seen, "the seeded late arm was never measured"
    assert "early_sig" in ev.seen, (
        f"the earlier boundary was never searched under the seeded Phi: measured {ev.seen}")
    # AND NOT A SINGLE SYNTHESIZED SIGNAL.
    assert out.signals_installed == [], (
        f"Phi was expanded before the grounded frontier was exhausted: {out.signals_installed}")
    assert rt.installed == []
    assert out.frontier_boundaries, "the frontier sweep left no provenance"


def test_the_unmeasured_frontier_is_UNMEASURED_not_a_null():
    """An arm nobody scored is what the caller must go and measure."""
    ev = _Ev({"late_sig": -3})
    out, _ = _run(ev)
    assert out.state == REALIZABLE_UNMEASURED, out.state
    assert out.state != NO_BENEFIT
    # The measured negative is PRESERVED as such, and not re-measured.
    assert ev.seen.count("late_sig") == 1, f"the rejected arm was re-evaluated: {ev.seen}"


def test_the_frontier_arms_are_GROUNDED_and_returned():
    ev = _Ev({"late_sig": -3})
    out, _ = _run(ev)
    sigs = {c.signal for c in out.candidates}
    assert "early_sig" in sigs, f"the frontier's arms were not returned: {sigs}"


def test_provenance_records_WHY_the_earlier_boundary_was_reached():
    ev = _Ev({"late_sig": -3})
    out, _ = _run(ev)
    details = " ".join(a.detail or "" for a in out.attempts)
    assert "grounded frontier" in details, details
    d = out.as_dict()
    assert d["frontier_boundaries"], d


def test_a_beneficial_FRONTIER_arm_is_promoted_without_any_expansion():
    """The frontier is not a formality: a controller found there can win outright."""
    ev = _Ev({"late_sig": -3, "early_sig": 5})
    out, _ = _run(ev)
    assert out.state == IMPROVED, out.state
    assert out.promoted.signal == "early_sig"
    assert out.signals_installed == [], "Phi was expanded despite a winning grounded arm"


# ============================================================ EXPANSION STAYS REACHABLE

def test_expansion_IS_reached_once_the_whole_grounded_frontier_is_measured():
    """The second half of the rule: exhausting the frontier is what licenses synthesis."""
    ev = _Ev({"late_sig": -3, "early_sig": -1})       # every grounded arm MEASURED, none helped
    out, rt = _run(ev)
    assert "late_sig" in ev.seen and "early_sig" in ev.seen
    assert out.signals_installed, (
        f"the grounded frontier was exhausted by measurement and Phi was never widened: {out.state}")
    assert rt.installed, "no signal was installed"


def test_a_beneficial_EXPANDED_arm_is_promoted_after_the_frontier_fails():
    ev = _Ev({"late_sig": -3, "early_sig": -1, "invented": 7})
    out, _ = _run(ev)
    assert out.state == IMPROVED, out.state
    assert "invented" in out.promoted.signal, out.promoted.signal


# ============================================================ NO CYCLING

def test_the_frontier_is_swept_ONCE_not_per_boundary():
    """Without a latch, a negative at each boundary would re-sweep the others forever."""
    ev = _Ev({"late_sig": -3, "early_sig": -1})
    out, _ = _run(ev)
    assert ev.seen.count("late_sig") == 1, f"the late arm was re-measured: {ev.seen}"
    assert ev.seen.count("early_sig") == 1, f"the early arm was re-measured: {ev.seen}"


def test_a_budget_stop_during_the_sweep_does_not_license_expansion():
    ev = _Ev({"late_sig": -3, "early_sig": -1})
    out, _ = _run(ev, eval_budget=1)
    assert out.evaluations_completed == 1
    assert out.signals_installed == [], "Phi widened after a budget stop, on no verdict"
    assert out.state != NO_BENEFIT


def test_without_an_evaluator_neither_the_sweep_nor_expansion_happens():
    """The control: no measurement, no inference of any kind -- including no frontier sweep."""
    rt = _Runtime()
    out = optimize_residual(_Residual(), runtime=rt, host=_Host(), events=EVENTS,
                            states=[{"late_sig": True, "early_sig": True}])
    assert out.state == REALIZABLE_UNMEASURED
    assert out.signals_installed == []
    assert out.frontier_boundaries == (), (
        "the frontier was swept without a measured verdict to license it")
