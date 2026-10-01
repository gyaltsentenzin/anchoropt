"""The spec-driven controller installer: core emits data, this evaluates it.

Its predecessor hard-coded two predicate shapes and one env threshold, which cannot carry a
synthesized phi whose shape is unknown when the runner is written. Hand-transcribing one would put a
human back inside the loop at the point the loop is supposed to own.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from install_controller import SpecPredicate      # noqa: E402

A9_SPEC = {
    "name": "not_searched_other_container__and__best_similarity_lt_0.294",
    "locus": "post_execution",
    "eta": {"primitive": "additional_read_and_merge"},
    "predicate": {"all": [{"field": "best_similarity", "op": "lt", "value": 0.294},
                          {"field": "searched_other_container", "op": "falsy"}]},
}


def _hand(state, theta=0.294):
    """install_phi.py's hand-written conjunction, reproduced as the equivalence reference."""
    v = state.get("best_similarity")
    if v is None:
        return False
    if state.get("searched_other_container") is not False:
        return False
    return float(v) < theta


PROBES = [
    {"best_similarity": 0.20, "searched_other_container": False},
    {"best_similarity": 0.45, "searched_other_container": False},
    {"best_similarity": 0.20, "searched_other_container": True},
    {"searched_other_container": False},
    {"best_similarity": 0.20},
    {"best_similarity": 0.294, "searched_other_container": False},
    {},
]


@pytest.mark.parametrize("state", PROBES)
def test_the_spec_predicate_matches_the_hand_written_one(state):
    """Equivalence on every probe, including the edge cases, or the arm measures something else."""
    assert SpecPredicate(A9_SPEC).fires_on(state) is _hand(state), state


def test_a_missing_field_is_not_a_firing():
    """"The condition did not hold" must stay distinguishable from "the state lacked the field"."""
    p = SpecPredicate({"predicate": {"field": "nope", "op": "lt", "value": 1.0}})
    assert not p.fires_on({})


def test_falsy_means_observed_false_not_merely_absent():
    """A conjunct like "the other store was NOT consulted" must not hold where nothing was recorded.

    Otherwise the arm fires on episodes its residual does not contain.
    """
    p = SpecPredicate({"predicate": {"field": "flag", "op": "falsy"}})
    assert p.fires_on({"flag": False})
    assert not p.fires_on({}), "absent was treated as observed-False"
    assert not p.fires_on({"flag": None})


def test_an_unknown_operator_is_refused_loudly():
    with pytest.raises(ValueError, match="unknown op"):
        SpecPredicate({"predicate": {"field": "x", "op": "approximately"}})


def test_a_malformed_conjunction_is_refused():
    with pytest.raises(ValueError):
        SpecPredicate({"predicate": {"all": []}})
    with pytest.raises(ValueError):
        SpecPredicate({"predicate": {"all": [{"field": "x"}]}})


def test_a_raising_predicate_is_not_a_trigger():
    p = SpecPredicate({"predicate": {"field": "x", "op": "lt", "value": 1.0}})
    assert not p.fires_on({"x": "not-a-number"})


def test_eta_and_name_travel_with_the_controller():
    p = SpecPredicate(A9_SPEC)
    assert p.name.startswith("not_searched_other_container")
    assert p.eta["primitive"] == "additional_read_and_merge"


def test_installing_from_a_file_registers_at_the_declared_locus():
    from anchoropt.runtime_hook import decide, installed, reset
    reset()
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(A9_SPEC, fh)
    fh.close()
    try:
        os.environ["ANCHOROPT_CONTROLLER_SPEC"] = fh.name
        import importlib

        import install_controller
        importlib.reload(install_controller)
        assert len(installed("post_execution")) == 1
        d = decide("post_execution",
                   {"best_similarity": 0.20, "searched_other_container": False},
                   host_default=False)
        assert d.proceed and d.by_controller
    finally:
        os.environ.pop("ANCHOROPT_CONTROLLER_SPEC", None)
        os.unlink(fh.name)
        reset()


# ---- the spec must satisfy BOTH predicate interfaces --------------------------------------------

def test_the_spec_predicate_answers_to_evaluate_as_well_as_fires_on():
    """The runtime hook calls .fires_on(); every core comparison utility calls .evaluate().

    Exposing only one made a round-trip check report ZERO firings against a predicate that fires 14
    times on real states: `firing_vector` called the missing method, the AttributeError was swallowed
    by the "a raising phi is not a trigger" rule, and the result was a plausible all-False vector.
    Right at runtime, dangerous in a verification path.
    """
    p = SpecPredicate(A9_SPEC)
    st = {"best_similarity": 0.2, "searched_other_container": False}
    assert p.fires_on(st) is True
    assert p.evaluate(st) is True
    assert p.evaluate({}) is False


def test_the_core_firing_utilities_accept_a_spec_predicate():
    from anchoropt.learning.learned_signal import fingerprint, firing_vector
    states = [{"best_similarity": i / 10.0, "searched_other_container": i % 2 == 0}
              for i in range(10)]
    p = SpecPredicate(A9_SPEC)
    fv = firing_vector(p, states)
    assert any(fv), "a spec predicate that fires reported an all-False vector"
    assert fingerprint(p, states) == frozenset(i for i, f in enumerate(fv) if f)


def test_an_emitted_spec_reproduces_the_grammar_predicate_exactly():
    """The property that makes machine-emitted specs safe: no semantic drift in translation."""
    from anchoropt.learning.learned_signal import firing_vector
    from anchoropt.learning.signal_grammar import Atom, Conjunction
    ref = Conjunction(terms=(Atom(field="best_similarity", op="lt", value=0.294),
                             Atom(field="searched_other_container", op="falsy")))
    spec = SpecPredicate(A9_SPEC)
    states = [{"best_similarity": i / 20.0, "searched_other_container": i % 3 == 0}
              for i in range(20)] + [{}, {"best_similarity": 0.1}]
    assert firing_vector(spec, states) == firing_vector(ref, states)
