"""Backward search over trajectory decision boundaries. No semantic stage taxonomy anywhere."""

import pathlib
import re

from anchoropt.learning.boundary_search import (
    BOUNDARY_EXHAUSTED, BOUNDARY_NOT_REPAIRABLE, BOUNDARY_NO_CANDIDATE,
    BOUNDARY_PRIMITIVE_MISSING, BOUNDARY_PROMOTED, Boundary, backward_boundary_search,
    boundaries_of, order_boundaries,
)

B = [Boundary("early", 0), Boundary("late", 1)]


def hooks(**over):
    base = dict(
        repairable_at=lambda r, b: True,
        expressible=lambda r, b: (True, "ok"),
        build=lambda r, b, sigs: ["c1"],
        realizable=lambda c: (True, "prim"),
        evaluate=lambda c: 0,
        improves=lambda o: False,
        promote=lambda c: None,
        remine=lambda: None,
    )
    base.update(over)
    return base


# ---------------------------------------------------------------- deriving boundaries
def test_boundaries_come_FROM_the_trajectory_in_realized_order():
    events = [{"boundary": "a"}, {"boundary": "b"}, {"boundary": "c"}]
    bs = boundaries_of(events)
    assert [b.key for b in bs] == ["a", "b", "c"]
    assert [b.index for b in bs] == [0, 1, 2]


def test_a_recurring_boundary_collapses_to_its_FIRST_occurrence():
    """A boundary kind that recurs is one decision point visited repeatedly, not several places to
    intervene; searching it per occurrence would multiply the candidate set by trajectory length."""
    bs = boundaries_of([{"boundary": "a"}, {"boundary": "b"}, {"boundary": "a"}])
    assert [b.key for b in bs] == ["a", "b"]


def test_non_decision_events_are_skipped_when_the_adapter_says_so():
    class A:
        def is_decision(self, ev):
            return ev.get("kind") == "decision"
        def boundary_key(self, ev):
            return ev["at"]
    events = [{"kind": "noise", "at": "x"}, {"kind": "decision", "at": "y"}]
    assert [b.key for b in boundaries_of(events, adapter=A())] == ["y"]


def test_temporal_order_is_the_default():
    bs = [Boundary("late", 5), Boundary("early", 1)]
    assert [b.key for b in order_boundaries(bs)] == ["early", "late"]


def test_a_generic_DEPENDENCY_relation_can_override_temporal_order():
    """The escape hatch is a dependency relation -- what precedes what -- never a named taxonomy."""
    class A:
        def depends_on(self, key):
            return ("b",) if key == "a" else ()
    bs = [Boundary("a", 0), Boundary("b", 1)]
    assert [x.key for x in order_boundaries(bs, adapter=A())] == ["b", "a"]


def test_an_unsatisfiable_dependency_degrades_to_realized_order_rather_than_deadlocking():
    class Cyclic:
        def depends_on(self, key):
            return {"a": ("b",), "b": ("a",)}.get(key, ())
    bs = [Boundary("a", 0), Boundary("b", 1)]
    got = [x.key for x in order_boundaries(bs, adapter=Cyclic())]
    assert sorted(got) == ["a", "b"], "a cycle must not drop or duplicate a boundary"


# ---------------------------------------------------------------- the search
def test_the_LATEST_boundary_is_searched_first():
    r = backward_boundary_search("R", boundaries=B, hooks=hooks())
    assert r.visited == ("late", "early")


def test_it_moves_earlier_only_when_a_boundary_is_exhausted():
    seen = []
    r = backward_boundary_search("R", boundaries=B,
                                 hooks=hooks(build=lambda r_, b, s: seen.append(b.key) or ["c"]))
    assert seen == ["late", "early"]
    assert r.moves_earlier == 2


def test_a_boundary_where_the_residual_does_not_manifest_is_skipped_unsearched():
    built = []
    h = hooks(repairable_at=lambda r_, b: b.key != "late",
              build=lambda r_, b, s: built.append(b.key) or ["c"])
    r = backward_boundary_search("R", boundaries=B, hooks=h)
    assert built == ["early"]
    assert r.attempts[0].state == BOUNDARY_NOT_REPAIRABLE


def test_SIGNAL_BLOCKED_expansion_is_NESTED_inside_each_boundary_turn():
    calls = []
    h = hooks(expressible=lambda r_, b: (False, "nothing observes it"),
              expand=lambda r_, b: calls.append(b.key) or ("phi_new",),
              build=lambda r_, b, s: ["c"] if s == ("phi_new",) else [])
    r = backward_boundary_search("R", boundaries=B, hooks=h)
    assert calls == ["late", "early"], "expansion runs per boundary, not once globally"
    assert r.attempts[0].signal_blocked == 1
    assert r.attempts[0].signals_expanded == ("phi_new",)


def test_a_blocked_boundary_with_no_expansion_hook_is_exhausted_not_crashed():
    h = hooks(expressible=lambda r_, b: (False, "blocked"))
    h.pop("expand", None)
    r = backward_boundary_search("R", boundaries=B, hooks=h)
    assert all(a.state == BOUNDARY_EXHAUSTED for a in r.attempts)


def test_a_missing_primitive_is_DISTINCT_from_a_policy_that_lost():
    """The first is a substrate gap; the second is evidence about the policy. Collapsing them would
    report an unimplemented HOW as a refuted hypothesis."""
    r = backward_boundary_search("R", boundaries=B,
                                 hooks=hooks(realizable=lambda c: (False, "nothing does that")))
    assert r.attempts[0].state == BOUNDARY_PRIMITIVE_MISSING
    assert r.attempts[0].controllers_evaluated == 0
    assert "nothing does that" in r.attempts[0].primitive_missing


def test_no_candidate_is_its_own_outcome():
    r = backward_boundary_search("R", boundaries=B, hooks=hooks(build=lambda r_, b, s: []))
    assert all(a.state == BOUNDARY_NO_CANDIDATE for a in r.attempts)


def test_an_improvement_promotes_and_restarts_at_the_LATEST_boundary():
    """A promotion changes the trajectories, so every later decision faces a different residual."""
    nxt = ["R2", None]
    h = hooks(improves=lambda o: True, evaluate=lambda c: 1, remine=lambda: nxt.pop(0))
    r = backward_boundary_search("R", boundaries=B, hooks=h, max_restarts=3)
    assert r.promoted == "c1" and r.restarts == 1
    assert r.attempts[0].state == BOUNDARY_PROMOTED
    assert r.visited[0] == "late" and r.visited[1] == "late", "restart returns to the latest boundary"


def test_restarts_are_bounded_so_an_always_improving_loop_terminates():
    h = hooks(improves=lambda o: True, evaluate=lambda c: 1, remine=lambda: "R_next")
    assert backward_boundary_search("R", boundaries=B, hooks=h, max_restarts=2).restarts == 2


# ---------------------------------------------------------------- genericity
def test_core_declares_no_semantic_stage_vocabulary():
    src = (pathlib.Path(__file__).resolve().parent.parent /
           "anchoropt" / "learning" / "boundary_search.py").read_text().lower()
    for word in ("read", "write", "archival", "kv", "rec_sum", "bfcl", "container",
                 "similarity", "retrieve", "memory", "stage_order"):
        assert not re.search(rf"\b{word}\b", src), f"core names a benchmark concept: {word!r}"


def test_the_adapter_no_longer_declares_a_stage_taxonomy():
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                           "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    assert not hasattr(rt, "STAGE_ORDER")
    assert not hasattr(rt, "stage_order")
    assert not hasattr(rt, "loci_for_stage")
    # what remains is structural, plus labels that are descriptive only
    assert hasattr(rt, "is_decision") and hasattr(rt, "boundary_key")
    assert rt.label_for("post_execution")            # a label exists
    assert rt.label_for("nonexistent_boundary") == ""  # and is optional


def test_the_same_search_drives_an_UNRELATED_trajectory_shape():
    """Another benchmark works without inventing a stage taxonomy: boundaries come from its events."""
    events = [{"boundary": "plan"}, {"boundary": "act"}, {"boundary": "verify"}]
    bs = order_boundaries(boundaries_of(events))
    order = []
    backward_boundary_search("R", boundaries=bs,
                             hooks=hooks(build=lambda r_, b, s: order.append(b.key) or []))
    assert order == ["verify", "act", "plan"]
