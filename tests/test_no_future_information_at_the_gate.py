"""A pre-execution predicate must not read the outcome of the call it is deciding about.

`container_full` is read at the commitment gate, which is pre-dispatch: this step's own tool results
do not exist there yet. The first version of the offline accumulator folded st["tool_results"] in
BEFORE emitting the state, so a step whose own write returned "core memory is full" already carried
container_full=True -- information the live hook could not have had when it decided.

Two harms, and the second is worse: it inflates firing, and it makes an offline-validated predicate
unreproducible at runtime, which is the declared-vs-supplied defect class arriving from the future
instead of from absence.

The live hook is the reference: at the gate, `execution_results` holds the PREVIOUS step's results
(`self._execute` assigns the current step's results later -- memory_evaluator.py:4653, well after the
gate at ~4575). The offline projection must match that, step for step.
"""

from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "integrations" / "claude_roles"))


def _m():
    import self_evolve_cycle2
    return self_evolve_cycle2


# step 0 writes and is told the store is full; step 1 then proposes a clear.
EPISODE = {
    "ep": [
        {"status": "executed", "decoded": ["core_memory_add(text='a')"],
         "tool_results": ["{'error': 'core memory is full'}"]},
        {"status": "executed", "decoded": ["core_memory_clear()"], "tool_results": ["{}"]},
        {"status": "executed", "decoded": ["core_memory_add(text='b')"], "tool_results": ["{'id': 1}"]},
    ]
}


def test_the_step_that_REVEALS_the_limit_does_not_yet_carry_it():
    """THE LEAK, stated directly: step 0's own result must not be visible to step 0's decision."""
    states = _m().observable_states(EPISODE)
    assert states[0]["container_full"] is False, (
        "the step whose own tool result reported the limit carried it at its own pre-exec decision "
        "-- that is future information")


def test_the_NEXT_step_does_carry_it():
    """Backward-looking is not the same as blind: a limit observed earlier must persist."""
    states = _m().observable_states(EPISODE)
    assert states[1]["container_full"] is True
    assert states[2]["container_full"] is True, "the fact must persist for the rest of the episode"


def test_it_never_un_sets_once_observed():
    states = _m().observable_states(EPISODE)
    seen = [s["container_full"] for s in states]
    assert seen == sorted(seen), f"the fact went back to False: {seen}"


def test_an_episode_with_no_capacity_error_never_sets_it():
    states = _m().observable_states({"ep": [
        {"status": "executed", "decoded": ["core_memory_add(text='a')"],
         "tool_results": ["{'id': 0}"]},
        {"status": "executed", "decoded": ["core_memory_retrieve(query='a')"],
         "tool_results": ["{'result': []}"]},
    ]})
    assert all(s["container_full"] is False for s in states)


def test_the_fact_does_not_cross_EPISODES():
    """Episode-scoped: one episode's full store says nothing about the next episode's."""
    states = _m().observable_states({
        "a_first": EPISODE["ep"],
        "b_second": [{"status": "executed", "decoded": ["core_memory_add(text='x')"],
                      "tool_results": ["{'id': 0}"]}],
    })
    # sorted by case id: a_first (3 steps) then b_second (1 step)
    assert states[-1]["container_full"] is False, "the fact leaked into a later episode"


def test_the_field_is_always_present():
    """A sometimes-absent key lets a predicate over it silently answer False."""
    for s in _m().observable_states(EPISODE):
        assert "container_full" in s


def test_the_live_hook_reads_the_PREVIOUS_steps_results():
    """Pin the reference the offline projection mirrors, so a divergence is caught here.

    Source-level and adapter-side by nature: core must never read host source, but this test is the
    adapter's own contract check, and the ordering is the whole correctness argument.
    """
    ev = REPO / "benchmarks" / "bfcl_v4" / "evaluator" / "memory_evaluator.py"
    if not ev.exists():                                  # pragma: no cover
        import pytest
        pytest.skip("vendored evaluator not present")
    src = ev.read_text()
    # The gate hook must come BEFORE the assignment of this step's execution results.
    gate = src.find("step_record: Dict = {")
    exec_assign = src.find("execution_results, involved_instances = self._execute(")
    if gate < 0 or exec_assign < 0:                       # pragma: no cover
        import pytest
        pytest.skip("the vendored copy does not carry both sites")
    assert gate < exec_assign, (
        "this step's execution results are assigned BEFORE the pre-exec state is built -- a "
        "pre-execution predicate could then read the outcome of the call it is deciding about")
