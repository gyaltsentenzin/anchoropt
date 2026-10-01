"""A signal declared at a boundary must be SATISFIABLE there. Third instance of one defect class.

`retrieval_similarity_below_threshold` requires `proposes_read`. That field was neither declared nor
supplied at post_execution, so on real query states it was present 0/53 while `best_similarity` was
present in 22 -- the signal could not fire at the only boundary it is declared on, and every
read-side residual family (support 9-11, with a WORKING executor) produced arms that never fired.

Same class as `container_full` at the commitment gate, one field over. The rule this file pins: a
boundary must SUPPLY every field its declared signals read, and the fact must be derivable from
information available AT that boundary.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "integrations" / "claude_roles"))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
sys.path.insert(0, str(REPO))

GATE = "post_execution"
SIG = "retrieval_similarity_below_threshold"


def _rt():
    import bfcl_runtime
    return bfcl_runtime


def _drv():
    import self_evolve_cycle2
    return self_evolve_cycle2


EPISODE = {
    "q": [
        {"status": "executed", "decoded": ["core_memory_retrieve(query='x',top_k=5)"],
         "tool_results": ['{"result": [{"id": 0, "similarity_score": 0.11, "text": "t"}]}']},
        {"status": "executed", "decoded": ["core_memory_add(text='not a read')"],
         "tool_results": ['{"id": 1}']},
    ]
}


def test_the_field_is_DECLARED_and_SUPPLIED_at_the_boundary():
    """Either half alone is silent: unsupplied answers False forever, undeclared is invisible."""
    rt = _rt()
    assert "proposes_read" in rt.hook_state_fields(GATE), "not declared as hook-supplied"
    assert "proposes_read" in rt.synthesis_fields(GATE), "not offered to synthesis"


def test_a_read_step_carries_proposes_read_and_a_write_step_does_not():
    states = _drv().observable_states(EPISODE)
    assert states[0]["proposes_read"] is True, "a retrieve step did not register as a read"
    assert states[1]["proposes_read"] is False, "a write step registered as a read"


def test_the_field_is_ALWAYS_present():
    """A sometimes-absent key lets a predicate over it silently answer False."""
    for s in _drv().observable_states(EPISODE):
        assert "proposes_read" in s


def test_the_signal_is_SATISFIABLE_on_a_real_shaped_state():
    """The regression: before the fix this was False for every possible query state."""
    rt = _rt()
    st = dict(_drv().observable_states(EPISODE)[0], boundary=GATE)
    assert rt.evaluate_signal(SIG, st, {"below": 0.3}) is True, st
    # And it discriminates on theta, which is what makes it a PARAMETERIZED signal rather than a
    # constant -- an arm whose signal fires identically at every theta has no grid to search.
    assert rt.evaluate_signal(SIG, st, {"below": 0.05}) is False


def test_it_does_not_fire_on_a_step_that_proposed_no_read():
    rt = _rt()
    st = dict(_drv().observable_states(EPISODE)[1], boundary=GATE)
    st["best_similarity"] = 0.1          # a score inherited from elsewhere must not be enough
    assert rt.evaluate_signal(SIG, st, {"below": 0.3}) is False


def test_the_classification_comes_from_the_ADAPTER_not_a_local_guess():
    """A second read-vocabulary in the driver is a second definition that can drift."""
    src = (REPO / "scripts" / "self_evolve_cycle2.py").read_text()
    i = src.index("def _is_read(")
    body = src[i:i + 700]
    assert "import adapter" in body and "is_read" in body, (
        "the driver classifies reads itself instead of asking the adapter")


def test_an_unclassifiable_call_yields_FALSE_not_an_invented_fact():
    assert _drv()._is_read("!!! not a call !!!") is False


@pytest.mark.parametrize("phase", ["query"])
def test_no_pre_dispatch_fact_leaks_into_the_post_execution_state(phase):
    """`proposes_read` is backward-looking here: the call has already run. Nothing from the NEXT
    step may appear, which is the future-information rule one boundary over."""
    states = _drv().observable_states(EPISODE)
    # step 0 is a read; step 1 is a write. Step 0 must not know about step 1.
    assert states[0]["proposes_read"] is True
    assert states[1]["proposes_read"] is False
