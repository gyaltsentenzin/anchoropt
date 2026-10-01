"""The declarative signal language: proposed phi as DATA, validated by us.

Each test pins a property whose violation would either re-privilege the historical anchors or open
the closed space:

  * a proposer can only name DECLARED fields, with declared types
  * operators are a closed set, type-checked
  * the boundary set is DERIVED, so the A4-v1 defect class is unrepresentable
  * no code path executes proposal text
  * the eight accepted triggers are EXPRESSIBLE from the alphabet, not pre-registered
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

import bfcl_capabilities as C                                          # noqa: E402
from anchoropt.learning.signal_lang import (                           # noqa: E402
    MAX_LEAVES, SignalSpecError, compile_signal, describe,
)

FIELDS = C.all_fields()


# ---- the space stays closed --------------------------------------------------------------------

def test_undeclared_field_is_rejected():
    with pytest.raises(SignalSpecError, match="not declared"):
        compile_signal("x", {"field": "arbitrary_thing", "op": "is_true"}, fields=FIELDS)


def test_unknown_operator_is_rejected():
    with pytest.raises(SignalSpecError, match="closed set"):
        compile_signal("x", {"field": "proposes_write", "op": "exec"}, fields=FIELDS)


def test_operator_type_is_checked():
    """`lt` on a string field is not a signal, it is a bug waiting to fire silently."""
    with pytest.raises(SignalSpecError, match="cannot be applied"):
        compile_signal("x", {"field": "error_kind", "op": "lt", "value": 3}, fields=FIELDS)


def test_enum_value_is_checked():
    with pytest.raises(SignalSpecError, match="takes one of"):
        compile_signal("x", {"field": "error_kind", "op": "eq", "value": "made_up_kind"},
                       fields=FIELDS)


def test_expression_size_is_bounded():
    leaves = [{"field": "proposes_write", "op": "is_true"} for _ in range(MAX_LEAVES + 1)]
    with pytest.raises(SignalSpecError, match="leaves"):
        compile_signal("x", {"all": leaves}, fields=FIELDS)


def test_missing_required_value_is_rejected():
    with pytest.raises(SignalSpecError, match="requires"):
        compile_signal("x", {"field": "best_similarity", "op": "lt"}, fields=FIELDS)


# ---- boundaries are derived, not asserted -----------------------------------------------------

def test_boundary_set_is_the_intersection_of_its_fields():
    c = compile_signal("x", {"all": [{"field": "proposes_clear", "op": "is_true"},
                                     {"field": "result", "op": "is_vacuous"}]}, fields=FIELDS)
    assert c.boundaries == frozenset({"post_execution"})


def test_a_condition_observable_nowhere_is_rejected():
    """The A4-v1 defect class: a signal combining facts that never coexist at one decision point
    is observable at neither, and an anchor on it claims a locus it does not have."""
    fields = dict(FIELDS)
    fields["pre_only"] = C.Field("pre_only", bool, ("pre_generation",), "synthetic")
    with pytest.raises(SignalSpecError, match="never coexist"):
        compile_signal("x", {"all": [{"field": "pre_only", "op": "is_true"},
                                     {"field": "result", "op": "is_vacuous"}]}, fields=fields)


def test_post_generation_only_fields_exclude_pre_generation():
    c = compile_signal("x", {"field": "proposes_tool_call", "op": "is_false"}, fields=FIELDS)
    assert "pre_generation" not in c.boundaries


# ---- vacuity is the existing detector ---------------------------------------------------------

def test_is_vacuous_delegates_to_policy_tree():
    c = compile_signal("x", {"field": "result", "op": "is_vacuous"}, fields=FIELDS)
    assert c.predicate({"result": '{"ranked_results": []}'}, {}) is True
    assert c.predicate({"result": '{"ranked_results": [[0.9, "hit"]]}'}, {}) is False
    assert c.predicate({"result": None}, {}) is False


def test_vacuous_kind_discriminates():
    c = compile_signal("x", {"field": "result", "op": "vacuous_kind",
                             "value": "all_floor_scores"}, fields=FIELDS)
    assert c.predicate({"result": '{"r": [[0.0, "a"], [0.0, "b"]]}'}, {}) is True
    assert c.predicate({"result": '{"r": []}'}, {}) is False


# ---- parameters ------------------------------------------------------------------------------

def test_missing_param_raises_rather_than_defaulting():
    """A signal run with a guessed threshold is not the signal that was proposed."""
    c = compile_signal("x", {"field": "best_similarity", "op": "lt", "param": "below"},
                       fields=FIELDS)
    assert c.params_used == frozenset({"below"})
    with pytest.raises(KeyError):
        c.predicate({"best_similarity": 0.1}, {})
    assert c.predicate({"best_similarity": 0.1}, {"below": 0.3}) is True


def test_absent_field_value_is_false_not_an_exception():
    """A backend exposing no similarity scores must not crash the signal."""
    c = compile_signal("x", {"field": "best_similarity", "op": "lt", "value": 0.3}, fields=FIELDS)
    assert c.predicate({"best_similarity": None}, {}) is False


# ---- the historical anchors are expressible, not privileged -----------------------------------

def test_all_eight_accepted_triggers_compile_from_the_alphabet():
    """The point of the whole refactor: a proposer restricted to the declared alphabet could have
    reached every accepted controller without any of them being pre-registered."""
    from fixtures.replay_exprs import EXPECTED_BOUNDARY, REPLAY_EXPRS

    assert set(REPLAY_EXPRS) == set(EXPECTED_BOUNDARY)
    for anchor, spec in REPLAY_EXPRS.items():
        c = compile_signal(spec["signal"], spec["expr"], fields=FIELDS)
        assert EXPECTED_BOUNDARY[anchor] in c.boundaries, (
            f"{anchor}: compiled to {sorted(c.boundaries)}, accepted anchor fires at "
            f"{EXPECTED_BOUNDARY[anchor]}")


def test_history_enters_via_carried_fields_not_cross_boundary_refs():
    """A3/A5/A8 span time. They must stay SINGLE-LOCUS, reading carried summaries the wrapper
    maintains -- a signal straddling two decision points is observable at neither."""
    from fixtures.replay_exprs import REPLAY_EXPRS

    carried = set(C.carried_fields())
    for anchor in ("A3", "A5", "A8"):
        c = compile_signal(REPLAY_EXPRS[anchor]["signal"], REPLAY_EXPRS[anchor]["expr"],
                           fields=FIELDS)
        assert c.fields_used & carried, f"{anchor} should read a carried summary"
        assert len(c.boundaries) >= 1


def test_describe_is_readable():
    expr = {"all": [{"field": "proposes_read", "op": "is_true"},
                    {"field": "best_similarity", "op": "lt", "param": "below"}]}
    assert describe(expr) == "(proposes_read is_true AND best_similarity lt $below)"
