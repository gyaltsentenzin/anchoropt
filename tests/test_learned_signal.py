"""A learned signal carries its own boundaries and provenance, and IS its firing behaviour.

Two defects motivate this file:

  * (predicate, boundary, provenance) used to exist only as keyword arguments in flight to
    `runtime.install_signal`, so an adapter could keep the predicate and drop the boundary -- and
    then "where is this observable?" answered "nowhere", which reads like a considered answer;
  * a threshold grid produces many NAMES and few distinct BEHAVIOURS, so searching syntactic
    variants spends evaluations that cannot differ in outcome.
"""

from __future__ import annotations

from anchoropt.learning.controller import Controller, firing_set
from anchoropt.learning.learned_signal import (
    LearnedSignal, adds_information, behaviourally_identical, dedupe_by_behaviour, fingerprint,
    firing_vector, from_compiled, learned_from,
)
from anchoropt.learning.realization import verify_projection
from anchoropt.learning.signal_grammar import Atom, Conjunction

STATES = [{"x": i / 10.0, "flag": i % 2 == 0} for i in range(10)]


# ---- metadata travels with the predicate --------------------------------------------------------

def test_a_learned_signal_keeps_the_boundary_it_was_validated_at():
    s = learned_from(Atom(field="x", op="lt", value=0.5), boundaries=["post_execution"],
                     provenance="synthesized from observed quantiles")
    assert s.observable_at("post_execution")
    assert not s.observable_at("pre_generation")
    assert s.provenance


def test_no_recorded_boundary_means_observable_NOWHERE_not_everywhere():
    """An empty set is meaningful: nobody established anywhere this signal can be read."""
    s = learned_from(Atom(field="x", op="truthy"))
    assert s.boundaries == frozenset()
    assert not s.observable_at("post_execution")


def test_a_learned_signal_is_duck_type_compatible_with_existing_consumers():
    """`.evaluate(state)` is the protocol every core predicate consumer already uses."""
    s = learned_from(Atom(field="flag", op="truthy"), boundaries=["post_execution"])
    assert s.evaluate({"flag": True}) and not s.evaluate({"flag": False})
    ctl = Controller(locus="post_execution", phi=s, action="reroute")
    assert ctl.fires_on({"flag": True})


def test_verify_projection_accepts_a_learned_signal_directly():
    """Reuse, not reimplementation: the existing trigger-set comparison takes these unchanged."""
    a = learned_from(Atom(field="x", op="lt", value=0.5))
    b = learned_from(Atom(field="x", op="lt", value=0.45))
    v = verify_projection(a, b, STATES)
    assert v.accepted and v.disagreement == 0.0


def test_a_compiled_signal_adapts_in_with_its_params_bound():
    class Fake:
        name = "compiled_phi"
        expr = {"field": "x", "op": "lt", "param": "t"}
        boundaries = frozenset({"post_execution"})
        fields_used = frozenset({"x"})
        params_used = frozenset({"t"})

        @staticmethod
        def predicate(state, params):
            return state.get("x", 1.0) < params.get("t", 0.0)

    s = from_compiled(Fake(), params={"t": 0.5})
    assert s.observable_at("post_execution")
    assert s.evaluate({"x": 0.1}) and not s.evaluate({"x": 0.9})
    assert s.source == "compiled"


# ---- behavioural identity -----------------------------------------------------------------------

def test_the_firing_vector_is_positionally_aligned_to_the_states():
    fv = firing_vector(Atom(field="x", op="lt", value=0.3), STATES)
    assert fv == (True, True, True, False, False, False, False, False, False, False)


def test_fingerprint_agrees_with_the_existing_controller_firing_set():
    """Three notions of a trigger set now agree instead of coexisting."""
    phi = learned_from(Atom(field="x", op="lt", value=0.5))
    ctl = Controller(locus="post_execution", phi=phi, action="reroute")
    assert fingerprint(phi, STATES) == frozenset(firing_set(ctl, STATES))


def test_two_differently_named_thresholds_can_be_the_SAME_signal():
    a, b = Atom(field="x", op="lt", value=0.5), Atom(field="x", op="lt", value=0.45)
    assert a.name() != b.name()
    assert behaviourally_identical(a, b, STATES)


def test_dedupe_keeps_one_representative_per_behaviour_in_order():
    a = Atom(field="x", op="lt", value=0.5)
    b = Atom(field="x", op="lt", value=0.45)          # same behaviour as a
    c = Atom(field="flag", op="truthy")
    kept = dedupe_by_behaviour([a, b, c], STATES)
    assert [p.name() for p in kept] == [a.name(), c.name()]


def test_a_constant_predicate_adds_no_information():
    """It separates no states, so installing it widens Phi in name only."""
    assert not adds_information(Atom(field="absent_field", op="truthy"), [], STATES)


def test_a_predicate_that_duplicates_an_existing_behaviour_adds_no_information():
    a = Atom(field="x", op="lt", value=0.5)
    b = Atom(field="x", op="lt", value=0.45)
    assert not adds_information(b, [a], STATES)
    assert adds_information(Atom(field="flag", op="truthy"), [a], STATES)


def test_a_raising_predicate_has_not_fired_rather_than_propagating():
    """A raising phi is not a trigger -- the same rule the runtime hook follows."""
    class Boom:
        def evaluate(self, state):
            raise RuntimeError("no such field")

    assert firing_vector(Boom(), STATES) == tuple([False] * len(STATES))


def test_a_conjunction_is_handled_like_any_other_predicate():
    conj = Conjunction(terms=(Atom(field="x", op="lt", value=0.5),
                              Atom(field="flag", op="truthy")))
    fv = firing_vector(conj, STATES)
    assert fv == tuple((s["x"] < 0.5 and s["flag"]) for s in STATES)
