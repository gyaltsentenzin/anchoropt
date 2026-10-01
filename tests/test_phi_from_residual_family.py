"""THE INVARIANT: residual attribution constrains the subsequent decision space.

If a grouped failure family identifies grounded observables, structured search must start from THOSE
-- not discard them and enumerate the host's whole signal grammar.

Measured on a real H0 round before this held: attribution pooled 24 vector diagnoses into one family
whose observables were [append_would_exceed_cap, clear_proposed_at_capacity, container_at_capacity,
container_slots_exhausted]. The search then enumerated 51 grammar atoms over every declared field and
emitted 117 arms, of which exactly ONE named an observable the family had identified -- and the
family's top observable produced NO arm at all. The grouping was computed and then thrown away, so the
search optimized a space the residual had never implicated.

This is a core invariant, not a BFCL detail: any host whose attribution yields grounded observables
gets the same guarantee, and a host whose attribution yields none falls back to the declared set.

WHAT THIS MUST NOT BREAK. Expansion stays reachable: a seeded Phi that yields no beneficial controller
must still widen, or a seed becomes a cage. That is asserted here too.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.learning.structured_search import optimize_residual


class _Residual:
    """A residual family, exactly as `pool_by_observable` presents one to the search."""

    def __init__(self, observables=(), case_ids=("c1",)):
        self.observables = tuple(observables)
        self.case_ids = tuple(case_ids)
        self.key = "the grouped failure family"
        self.support = len(self.case_ids)
        self.rank = 1


DECLARED = ("alpha", "beta", "gamma", "delta", "epsilon")

EVENTS = [{"kind": "decision", "boundary": "the_only_boundary", "step": 0}]


class _Runtime:
    """A minimal host: one boundary, five declared signals, one executable action."""

    HOST = None

    def __init__(self):
        self.asked_for = []

    # --- declarations
    def declared_signals(self):
        return DECLARED

    def signal_boundaries(self, name):
        return frozenset({IncisionPoint.POST_EXECUTION})

    def is_decision(self, event):
        return True

    def boundary_key(self, event):
        return "the_only_boundary"

    def boundary_from_key(self, key):
        return IncisionPoint.POST_EXECUTION

    def evaluate_signal(self, name, state, params):
        self.asked_for.append(name)
        return True

    def states_at(self, boundary, states):
        return list(states or ())


def _run(observables, **kw):
    rt = _Runtime()
    res = _Residual(observables)
    out = optimize_residual(res, runtime=rt, host=_host(), events=EVENTS,
                            states=[{"alpha": True}], **kw)
    return out, rt


def _host():
    class H:
        def require(self, boundary, action):
            return True

        def executable_actions(self, boundary):
            return (Action.REPROMPT,)
    return H()


# ------------------------------------------------------------------ the invariant

def test_phi_is_seeded_from_the_family_observables():
    out, _ = _run(("beta", "delta"))
    assert out.phi_source == "residual_family", out.phi_source
    assert set(out.phi_seeded) == {"beta", "delta"}, out.phi_seeded


def test_the_whole_grammar_is_NOT_enumerated_when_the_family_named_observables():
    """The defect, stated as a test: 2 named observables must not become 5 searched signals."""
    out, _ = _run(("beta", "delta"))
    assert len(out.phi_seeded) == 2, (
        f"searched {len(out.phi_seeded)} signals for a family that named 2: {out.phi_seeded}")
    assert "alpha" not in out.phi_seeded, "an unimplicated declared signal entered the space"


def test_a_family_with_no_observables_falls_back_to_the_declared_set():
    """Attribution that grounds nothing must not silently produce an empty search."""
    out, _ = _run(())
    assert out.phi_source == "declared"
    assert set(out.phi_seeded) == set(DECLARED)


def test_an_explicit_signals_argument_still_wins():
    """An ablation must be able to hand the search a representation of its choosing."""
    out, _ = _run(("beta",), signals=("gamma",))
    assert out.phi_source == "caller"
    assert tuple(out.phi_seeded) == ("gamma",)


def test_an_observable_the_host_cannot_evaluate_is_DROPPED_and_REPORTED():
    """An attributor naming a condition the adapter does not declare is a finding, not a crash."""
    out, _ = _run(("beta", "not_a_declared_signal"))
    assert out.phi_source == "residual_family"
    assert tuple(out.phi_seeded) == ("beta",)
    assert "not_a_declared_signal" in out.phi_dropped


def test_a_family_whose_observables_are_ALL_undeclared_falls_back_loudly():
    """Falling back is right; doing it silently is not -- the dropped set must say what happened."""
    out, _ = _run(("nope_1", "nope_2"))
    assert out.phi_source == "declared_fallback"
    assert set(out.phi_seeded) == set(DECLARED)
    assert set(out.phi_dropped) == {"nope_1", "nope_2"}


def test_the_seeded_phi_is_what_the_runtime_is_ASKED_about():
    """Behavioural, not declarative: the search must actually query only the seeded signals first."""
    out, rt = _run(("beta", "delta"))
    first = rt.asked_for[:2]
    assert set(first) <= {"beta", "delta"}, (
        f"the search asked about {first} before the family's own observables")


# ------------------------------------------------------------------ the seed is not a cage

def test_expansion_is_still_reachable_from_a_seeded_phi():
    """A seed that yields nothing must widen. Otherwise the invariant traps the search."""
    src = (REPO / "anchoropt" / "learning" / "structured_search.py").read_text()
    i = src.index("out.phi_seeded = tuple(base_phi)")
    after = src[i:]
    assert "expand" in after.lower(), "no expansion path remains after seeding"


def test_the_provenance_is_recorded_in_the_artifact():
    """A round seeded from the family and one that enumerated the grammar are different experiments."""
    out, _ = _run(("beta",))
    d = out.as_dict()
    for k in ("phi_source", "phi_seeded", "phi_dropped"):
        assert k in d, f"{k} is not in the recorded outcome"
    assert d["phi_source"] == "residual_family"
