"""A declared capability must not promise what its live executor does not do.

THE CONTAMINATION THIS PREVENTS. An arm whose executor silently ignores its action or its required
parameters runs as the CONTROL, and its paired result is then a measured zero against an intervention
that never happened. That is indistinguishable from a real NO_BENEFIT on a results table, so it does
not merely waste GPU time -- it corrupts the finding.

Two live defects this file pins, both found by auditing the running evaluator:

  * `post_execution/reroute` declared `eta_is_computed=True, consumes=()` -- a positive claim that the
    executor derives everything from live state. The live hook opens with
        if _prim != "additional_read_and_merge": return {"fired": False, ...}
    so four grounded arms carried no `primitive`, passed every check, installed cleanly, and would
    each have reported not-fired. `retry_semantics` was read NOWHERE, so `replace_original` and
    `retry_after` were byte-identical arms.
  * the reprompt executor reads its instruction from `templates.on_turn_start_action`, not from the
    candidate dict. That channel is legitimate -- an earlier round measured a real effect through it
    -- but undeclared it means two arms differing only in `instruction` execute identically unless
    the runner happens to write that text into the policy.

Benchmark-independent: the contract is exercised directly.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

from anchoropt.learning.executor_capability import ETA_NOT_CONSUMED, ExecutorCapability, supports


def _cap(**kw):
    base = dict(boundary="b", action="reroute", binding="a real site",
                signals=("sig",), eta_is_computed=True)
    base.update(kw)
    return ExecutorCapability(**base)


# ------------------------------------------------------- requires_eta: the reroute defect

def test_an_arm_missing_eta_the_executor_DEMANDS_is_refused():
    cap = _cap(requires_eta={"primitive": ("additional_read_and_merge",)})
    ok, why = supports(cap, boundary="b", action="reroute", signal="sig",
                       eta={"destination": "elsewhere"})
    assert not ok
    assert "primitive" in why and "CONTROL" in why, why


def test_eta_is_computed_does_NOT_exempt_a_required_key():
    """The flag says a candidate cannot STEER the executor -- not that it needs nothing."""
    cap = _cap(eta_is_computed=True, consumes=(),
               requires_eta={"primitive": ("additional_read_and_merge",)})
    ok, _ = supports(cap, boundary="b", action="reroute", signal="sig", eta={})
    assert not ok


def test_a_required_key_with_the_WRONG_value_is_refused():
    cap = _cap(requires_eta={"primitive": ("additional_read_and_merge",)})
    ok, why = supports(cap, boundary="b", action="reroute", signal="sig",
                       eta={"primitive": "something_else"})
    assert not ok and "something_else" in why


def test_a_required_key_with_an_ACCEPTED_value_passes():
    cap = _cap(requires_eta={"primitive": ("additional_read_and_merge",)})
    ok, why = supports(cap, boundary="b", action="reroute", signal="sig",
                       eta={"primitive": "additional_read_and_merge"})
    assert ok, why


def test_an_empty_allowed_set_means_any_non_empty_value():
    cap = _cap(requires_eta={"primitive": ()})
    assert supports(cap, boundary="b", action="reroute", signal="sig",
                    eta={"primitive": "anything"})[0]
    assert not supports(cap, boundary="b", action="reroute", signal="sig",
                        eta={"primitive": ""})[0]


# ------------------------------------------------------- the retry_semantics defect, pinned

def test_two_arms_differing_only_in_an_UNREAD_key_cannot_both_be_feasible():
    """THE DEFECT BY NAME. `retry_semantics` was read nowhere, so replace_original and retry_after
    were the same arm. Declaring it required is what stops a 2x sweep that cannot differ."""
    cap = _cap(requires_eta={"retry_semantics": ("replace_original", "retry_after")})
    for sem in ("replace_original", "retry_after"):
        ok, _ = supports(cap, boundary="b", action="reroute", signal="sig",
                         eta={"retry_semantics": sem})
        assert ok, sem
    # And an arm that omits it -- which is what the two byte-identical arms effectively were --
    # is refused rather than measured twice.
    ok, why = supports(cap, boundary="b", action="reroute", signal="sig", eta={})
    assert not ok and "retry_semantics" in why


# ------------------------------------------------------- unsupported_eta

def test_a_parameter_the_executor_CANNOT_honour_is_refused_not_ignored():
    cap = _cap(action="reprompt", consumes=("instruction",), eta_is_computed=False,
               unsupported_eta={"retry_budget": "bounded by a latch, not a parameter"})
    ok, why = supports(cap, boundary="b", action="reprompt", signal="sig",
                       eta={"instruction": "x", "retry_budget": 3},
                       required_eta=("instruction",))
    assert not ok
    assert "latch" in why, why


def test_an_absent_unsupported_key_is_fine():
    cap = _cap(action="reprompt", consumes=("instruction",), eta_is_computed=False,
               unsupported_eta={"retry_budget": "bounded by a latch"})
    assert supports(cap, boundary="b", action="reprompt", signal="sig",
                    eta={"instruction": "x"}, required_eta=("instruction",))[0]


# ------------------------------------------------------- the live BFCL declarations

def _rt():
    import bfcl_runtime
    return bfcl_runtime


def _reroute_cap(kind: str):
    """The reroute cell hosts MORE THAN ONE executor, so index [0] is not an identity.

    Two distinct mechanisms legitimately share (post_execution, reroute): an
    additional-read-and-merge that gates on `eta['primitive']`, and a fact-preserving payload
    reduction whose eta is computed from live state. Selecting by position made this guard fail the
    moment a second one was declared -- and a guard that depends on declaration ORDER is testing the
    wrong thing. Select by the property under test instead.
    """
    caps = list(_rt().executor_capabilities("post_execution", "reroute"))
    if kind == "merge":
        hits = [c for c in caps if "merge" in (c.capability_id or "") or c.requires_eta]
    else:
        hits = [c for c in caps if "reduce" in (c.capability_id or "")]
    assert hits, f"no {kind} executor declared at post_execution/reroute; declared: " \
                 f"{[c.capability_id for c in caps]}"
    return hits[0]


def test_the_live_reroute_cell_declares_what_its_hook_DEMANDS():
    cap = _reroute_cap("merge")
    assert "primitive" in cap.requires_eta, (
        "the reroute capability does not declare the primitive its hook gates on -- arms would run "
        "as the control")
    assert "additional_read_and_merge" in cap.requires_eta["primitive"]
    assert "retry_semantics" in cap.requires_eta, (
        "retry_semantics is undeclared, so replace_original and retry_after are the same arm")


def test_the_live_reprompt_cell_declares_its_DELIVERY_CHANNEL():
    caps = {c.cid: c for c in _rt().executor_capabilities("post_generation_pre_exec", "reprompt")}
    cap = caps["inject_trailing_message_and_regenerate"]
    assert "instruction" in cap.eta_delivered_via, (
        "the policy-template channel is undeclared, so two arms differing only in `instruction` "
        "would execute identically unless the runner happened to write it there")
    assert "on_turn_start_action" in cap.eta_delivered_via["instruction"]
    # And the latch-fixed budget is declared as FIXED, so a mismatched request conflicts.
    assert cap.fixed.get("retry_budget") == 1


def test_a_SUBSTITUTE_reroute_arm_still_refuses_without_the_primitive_it_gates_on():
    """The merge executor's contract is unchanged: no `primitive`, no execution.

    THIS TEST USED TO ASSERT SOMETHING ELSE. It read "the capacity family is not exhausted -- its
    executor is missing", and it was right at the time: the only executor at this cell demanded
    `eta['primitive']='additional_read_and_merge'`, which no grounding emitted, so every capacity arm
    was correctly refused EXECUTOR_UNAVAILABLE.
    
    That fact has CHANGED, and deliberately: the host's own fact-preserving payload reduction
    (`_try_capacity_repair`, generic and already validated on 352 live captures) is now declared at
    this cell, so a capacity arm CAN execute -- via reduction, not via merge. Leaving the old
    assertion would pin a limitation that no longer exists.
    
    What must still hold is the per-executor contract: an arm resolved against the MERGE executor,
    carrying no `primitive`, is still refused. That is the invariant the original test protected.
    """
    from anchoropt.anchor import Action, IncisionPoint
    from anchoropt.learning import executor_capability as ec
    from anchoropt.learning.action_contract import CONTRACTS, instantiate, operators_of
    rt = _rt()
    B = IncisionPoint.POST_EXECUTION
    refusals = 0
    for sig in ("container_at_capacity", "container_slots_exhausted"):
        for op in operators_of(Action.REROUTE):
            built, fail = instantiate(op, signal=sig, boundary=B, runtime=rt)
            for inst in (built or []):
                # Resolve against the MERGE executor ONLY -- the one whose hook gates on `primitive`.
                merge_only = [c for c in rt.executor_capabilities(B, Action.REROUTE)
                              if c.requires_eta]
                if not merge_only:
                    continue
                cap, why = ec.resolve(merge_only, boundary=B,
                                      action=Action.REROUTE, signal=sig, eta=inst.eta,
                                      required_eta=tuple(CONTRACTS[op].required))
                if op.value == "substitute":
                    assert cap is None, f"{sig}/{inst.variant} resolved against the merge executor " \
                                        f"without the primitive its hook gates on"
                    assert "primitive" in why or "retry_semantics" in why, why
                    refusals += 1
    assert refusals >= 2, f"expected substitute arms to be refused by the merge executor, saw {refusals}"


def test_the_capacity_family_now_HAS_an_executor_via_payload_reduction():
    """The complement of the test above: the family is no longer executor-unavailable.

    A fact-preserving reduction is declared at this cell, its eta is COMPUTED from live state, and it
    accepts the capacity signals -- so the family that was EXECUTOR_UNAVAILABLE is now measurable. It
    must still be a DIFFERENT executor from the merge one, or capability identity is meaningless.
    """
    rt = _rt()
    caps = list(rt.executor_capabilities("post_execution", "reroute"))
    reduce_caps = [c for c in caps if "reduce" in (c.capability_id or "")]
    assert reduce_caps, f"declared: {[c.capability_id for c in caps]}"
    cap = reduce_caps[0]
    assert cap.is_bound and cap.is_enabled
    assert not cap.requires_eta, "its eta is computed from live state, so it demands none"
    assert "append_would_exceed_cap" in cap.signals
    assert len(caps) >= 2, "the cell must host BOTH executors, or identity is not being tested"
