"""The destructive-clear signal must be SATISFIABLE at the boundary it is declared on.

`clear_proposed_at_capacity` reads `container_full`. The commitment-gate state is built per proposed
call and carried neither that field nor `error_kind`, so the signal could only ever return False at
runtime -- while passing every synthetic test that supplied the field by hand. That is the
declared-vs-supplied gap, and this project has paid for it twice before: a payload-size predicate
fired 0 times on an episode holding two over-cap writes, and 16 fields were declared at this boundary
where the hook supplied 5.

Why the fix is episode-scoped rather than per-call: whether a store is full is established by an
EARLIER step's tool result, and this boundary is pre-dispatch by construction. Measured: a capacity
error appears in a prior step's tool_results in 23 of 27 real storage episodes, so the evidence
exists -- it was never carried to the boundary that needed it.

It is a FIELD, not a condition. It reports that a limit was observed; core decides whether that
matters, at which boundary, and in conjunction with what. Nothing here says "do not clear".
"""

from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

GATE = "post_generation_pre_exec"
SIGNAL = "clear_proposed_at_capacity"


def _rt():
    import bfcl_runtime
    return bfcl_runtime


def _live_state(**over):
    """The state shape the LIVE hook builds -- nothing supplied by hand beyond the overrides."""
    st = {"proposed_call": "core_memory_clear()", "call_index": 0, "phase": "prereq",
          "step_index": 3, "n_proposed_calls": 1, "container_full": False,
          "proposes_write": False, "proposes_read": False, "proposes_clear": True,
          "proposes_remove": False, "container": "core",
          "proposed_payload_chars": 0, "proposed_arg_count": 0, "boundary": GATE}
    st.update(over)
    return st


def test_the_signal_is_satisfiable_on_the_live_state_shape():
    """The regression. Before the fix this was False for every possible live state."""
    rt = _rt()
    assert rt.evaluate_signal(SIGNAL, _live_state(container_full=True), {}) is True


def test_it_does_not_fire_before_a_limit_is_observed():
    rt = _rt()
    assert rt.evaluate_signal(SIGNAL, _live_state(container_full=False), {}) is False


def test_it_does_not_fire_on_a_non_destructive_call_at_capacity():
    """Capacity alone is not the condition -- the DESTRUCTIVE proposal is."""
    rt = _rt()
    st = _live_state(container_full=True, proposes_clear=False, proposes_write=True,
                     proposed_call="core_memory_add(text='x')")
    assert rt.evaluate_signal(SIGNAL, st, {}) is False


def test_the_field_is_declared_AND_supplied_at_the_gate():
    """Either half alone is silent: declared-but-unsupplied answers False forever, and
    supplied-but-undeclared is invisible to synthesis."""
    rt = _rt()
    assert "container_full" in rt.synthesis_fields(GATE), "not offered to synthesis"
    assert "container_full" in rt.hook_state_fields(GATE), "not declared as hook-supplied"


def test_the_projection_always_carries_the_field():
    """A key that is sometimes absent lets a predicate over it silently answer False."""
    rt = _rt()
    states = rt.states_at(GATE, [{"decoded": ["core_memory_clear()"], "step": 1, "phase": "prereq"}])
    assert states and all("container_full" in s for s in states)
    # Absent evidence means OBSERVED-NO-LIMIT, which is a fact, not a missing measurement.
    assert states[0]["container_full"] is False


def test_the_projection_carries_an_observed_limit_forward():
    rt = _rt()
    states = rt.states_at(GATE, [{"decoded": ["core_memory_clear()"], "step": 4,
                                  "phase": "prereq", "container_full": True}])
    assert states[0]["container_full"] is True


def test_destruction_and_retrieval_failure_stay_SEPARABLE():
    """The distinction the whole discovery question rests on: a destroyed fact is not a missed read.

    THE INVARIANT WAS STRENGTHENED, NOT RELAXED (R13). This used to assert the two signals lived at
    DISJOINT BOUNDARIES, which was a sound proxy only while nothing supplied the retrieval signal at
    the commitment window. Once an executor did (the answer-boundary seam), that proxy would have
    forced a choice between keeping a stale declaration and losing the check.

    So the check now tests what actually matters and what the proxy was standing in for: the two
    signals read DISJOINT STATE FIELDS, so neither is reachable from the other's state even when both
    are declared at the same boundary. That holds regardless of future boundary declarations, whereas
    the disjointness proxy would silently stop protecting anything the moment a third signal was added.
    """
    rt = _rt()
    destroyed = _live_state(container_full=True)
    assert rt.evaluate_signal(SIGNAL, destroyed, {}) is True

    # 1. The retrieval-side signal CANNOT fire on the destruction state: that state carries no
    #    `best_similarity` and `proposes_read` is False. This is the real separability.
    assert rt.evaluate_signal("retrieval_similarity_below_threshold",
                              {**destroyed, "boundary": GATE}, {"below": 0.25}) is False

    # 2. And the fields each predicate READS are disjoint -- derived from source, so this cannot
    #    drift from what the code does.
    destruction_fields = rt.signal_fields(SIGNAL)
    retrieval_fields = rt.signal_fields("retrieval_similarity_below_threshold")
    assert not (destruction_fields & retrieval_fields), (
        f"a destroyed fact and a missed read now share evidence: "
        f"{sorted(destruction_fields & retrieval_fields)}")

    # 3. The converse: a weak-retrieval state must not fire the destruction signal.
    weak_read = {"phase": "query", "step_index": 3, "boundary": GATE,
                 "best_similarity": 0.05, "proposes_read": True,
                 "proposes_clear": False, "container_full": False}
    assert rt.evaluate_signal("retrieval_similarity_below_threshold",
                              weak_read, {"below": 0.25}) is True
    assert rt.evaluate_signal(SIGNAL, weak_read, {}) is False


def test_no_post_execution_fact_sneaks_in_with_it():
    """`container_full` is an accumulated OBSERVATION, not this step's result. The boundary holds."""
    rt = _rt()
    states = rt.states_at(GATE, [{"decoded": ["core_memory_clear()"], "step": 4, "phase": "prereq",
                                  "container_full": True, "best_similarity": 0.9,
                                  "tool_results": ["{}"], "error_kind": "no_capacity"}])
    for leaked in ("best_similarity", "tool_results", "error_kind", "scored_entries"):
        assert leaked not in states[0], f"{leaked} leaked into the commitment gate"
