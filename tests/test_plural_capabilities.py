"""A (boundary, action) cell may hold MORE THAN ONE executor, and core must resolve WHICH.

The defect this closes, measured on this project's own host: `post_generation_pre_exec/reprompt`
names two mechanisms -- one that appends a user message and regenerates, one that writes the
instruction to the step record and does nothing else. The capability index was a dict keyed on the
cell, so the inert one (declared last) silently replaced the working one, and a live mechanism that
an earlier round had measured a real paired effect through became unreachable.

That is a BOOKKEEPING loss, not a validity finding, and the distinction is the whole point: the
inert executor is still refused, and refusing it must not cost us the injector beside it.

The second property is what keeps the fix safe. Answering "some executor at this cell works" and
then letting the host choose which to run would evaluate a mechanism core never validated -- the
arm-identity defect class from a new direction. So `resolve` returns the capability it accepted and
that id travels on the arm.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

from anchoropt.learning import executor_capability as ec
from anchoropt.learning.executor_capability import ExecutorCapability

CELL = ("post_generation_pre_exec", "reprompt")
WORKING = "inject_trailing_message_and_regenerate"
INERT = "write_instruction_to_step_record"


def _adapter():
    import bfcl_runtime
    return bfcl_runtime


# ------------------------------------------------------------------ core: the resolver

def _pair():
    inert = ExecutorCapability(
        boundary="b", action="reprompt", capability_id="inert", binding="site_a",
        consumes=(), signals=("s",), disabled_reason="telemetry only")
    live = ExecutorCapability(
        boundary="b", action="reprompt", capability_id="live", binding="site_b",
        consumes=("instruction", "retry_budget"), signals=("s",))
    return inert, live


def _resolve(caps, **kw):
    base = dict(boundary="b", action="reprompt", signal="s",
                eta={"instruction": "go", "retry_budget": 1},
                required_eta=("instruction", "retry_budget"))
    base.update(kw)
    return ec.resolve(caps, **base)


def test_a_disabled_sibling_does_not_mask_a_working_executor():
    """The regression, stated directly."""
    inert, live = _pair()
    cap, why = _resolve([inert, live])
    assert cap is not None and cap.cid == "live", why


def test_resolution_is_order_independent_for_feasibility():
    inert, live = _pair()
    assert _resolve([live, inert])[0].cid == "live"
    assert _resolve([inert, live])[0].cid == "live"


def test_a_cell_with_only_an_inert_executor_is_still_refused():
    """The correction must survive: refusing the cell was wrong, refusing the executor is right."""
    inert, _ = _pair()
    cap, why = _resolve([inert])
    assert cap is None
    assert "executor_disabled_by_host" in why


def test_the_rejection_reports_every_executor_it_tried():
    """Two executors fail for DIFFERENT reasons; a porter needs both."""
    inert, live = _pair()
    wrong_signal = ExecutorCapability(
        boundary="b", action="reprompt", capability_id="other", binding="site_c",
        consumes=("instruction", "retry_budget"), signals=("different_signal",))
    cap, why = _resolve([inert, wrong_signal])
    assert cap is None
    assert "inert" in why and "other" in why
    assert "executor_disabled_by_host" in why and "signal_unsupported" in why


def test_no_capability_at_all_is_reported_as_no_executor():
    cap, why = _resolve([])
    assert cap is None and "no_executor_capability" in why
    cap, why = _resolve(None)
    assert cap is None and "no_executor_capability" in why


def test_a_single_capability_behaves_exactly_as_before():
    """Backward compatibility: a host declaring one executor per cell is unaffected."""
    _, live = _pair()
    assert _resolve(live)[0].cid == "live"      # bare, not in a list
    assert _resolve([live])[0].cid == "live"


def test_cid_defaults_to_the_cell_when_unset():
    """An adapter that never heard of capability ids keeps working and still names something."""
    c = ExecutorCapability(boundary="post_execution", action="reroute", binding="x")
    assert c.cid == "post_execution/reroute"


def test_ghost_rejection_survives_the_plural_path():
    """An unbound capability is still a ghost, plural or not."""
    ghost = ExecutorCapability(boundary="b", action="reprompt", capability_id="ghost",
                              binding="", consumes=("instruction",), signals=("s",))
    cap, why = _resolve([ghost])
    assert cap is None and "executor_capability_unbound" in why


# ------------------------------------------- core: the selected capability binds to the arm

def test_the_arm_records_the_capability_core_validated():
    """THE BINDING GUARANTEE. Feasibility must not be a cell-level 'yes' with a free choice after."""
    from anchoropt.anchor import Action, IncisionPoint
    from anchoropt.learning.anchor_policy_opt import AnchorPolicyOpt, SearchSpaceProposal
    R = _adapter()
    prop = SearchSpaceProposal(
        boundary=IncisionPoint.POST_GENERATION_PRE_EXEC,
        signal="no_tool_call_at_all", action_set=(Action.REPROMPT,),
        diagnosis_case_ids=("c1",))
    arms, _rej = AnchorPolicyOpt(runtime=R, host=R.HOST).build_arms(prop)
    for arm in arms:
        assert arm.capability_id, "an arm at a multi-executor cell must name its executor"
        assert arm.capability_id != INERT, "no arm may be attributed to the inert executor"


# ------------------------------------------------------------------ adapter: the BFCL cell

def test_the_bfcl_reprompt_cell_declares_both_executors():
    R = _adapter()
    cids = [c.cid for c in R.executor_capabilities(*CELL)]
    assert WORKING in cids, "the injector must be reachable"
    assert INERT in cids, "the inert executor must stay declared, not deleted"


def test_the_working_injector_is_enabled_and_the_inert_one_is_not():
    R = _adapter()
    caps = {c.cid: c for c in R.executor_capabilities(*CELL)}
    assert caps[WORKING].is_enabled and caps[WORKING].is_bound
    assert not caps[INERT].is_enabled
    assert "never injected" in caps[INERT].disabled_reason


def test_the_injector_binding_names_the_code_that_regenerates():
    """A binding must name executing code. This one's claim is that it INJECTS, so say where."""
    R = _adapter()
    caps = {c.cid: c for c in R.executor_capabilities(*CELL)}
    b = caps[WORKING].binding
    assert "_add_next_turn_user_message_prompting" in b
    assert "regenerat" in b.lower()


def test_the_historical_R3_arm_shape_is_materializable_again():
    """R3 executed a zero-tool-call recovery through this executor and measured a paired effect.

    Its arm carried an instruction string and a retry budget the executor fixes at 1. That exact
    shape must resolve -- to the injector, never to the inert sibling.
    """
    R = _adapter()
    caps = R.executor_capabilities(*CELL)
    cap, why = ec.resolve(
        caps, boundary=CELL[0], action="reprompt", signal="no_tool_call_at_all",
        eta={"instruction": "Search the other memory container before answering.",
             "retry_budget": 1},
        required_eta=("instruction", "retry_budget"))
    assert cap is not None, f"R3's arm shape is not materializable: {why}"
    assert cap.cid == WORKING, cap.cid


def test_the_evaluator_really_injects_at_the_declared_site():
    """BEHAVIOURAL, not declarative: the binding is only true if the code does it.

    Core must never read host source, but a test may -- this is the adapter's own contract check,
    and it is the evidence that re-enabling this executor is a correction and not a ghost.
    """
    src = (REPO / "benchmarks" / "bfcl_v4" / "evaluator" / "memory_evaluator.py").read_text()
    i = src.index("enable_zero_call_reprompt")
    window = src[i:i + 1600]
    assert "_add_next_turn_user_message_prompting" in window, \
        "the zero-call branch no longer injects -- this executor must be DISABLED again"
    assert "on_turn_start_action" in window, "it no longer reads the eta slot it declares"


def test_the_legacy_singular_hook_prefers_an_enabled_executor():
    """A caller on the old hook must not be handed the disabled sibling."""
    R = _adapter()
    assert R.executor_capability(*CELL).cid == WORKING


def test_no_capability_id_names_an_accepted_anchor():
    """Ids describe MECHANISMS. An id like `a4_zero_call` would ship the answer key to the learner."""
    import re
    R = _adapter()
    for (b, a), caps in R._CAPABILITY_INDEX.items():
        for c in caps:
            assert not re.search(r"\bA[0-9]\b", c.cid), c.cid
            assert not re.search(r"(?i)\b(a[0-9]|anchor|e1)_", c.cid), c.cid
