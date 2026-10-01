"""Each arm must carry the identity of the executor core ACTUALLY resolved for it.

THE DEFECT, root-caused rather than filtered. `Action.REROUTE` has two operator readings -- `substitute`
(replace the proposed call) and `transform` (reshape the state a later decision reads) -- and this host
declares a separate executor for each at post_execution. `executor_capability.resolve()` was never told
WHICH reading the arm used, so it returned the first capability that supported the signal and eta. Both
readings therefore collapsed onto `relocate_entry_preserving_information_then_retry`, and 2 of the 6 specs
the pilot emitted carried an identity their operator does not implement. Under the host's identity gate
those are refused at dispatch: the arm silently becomes the control while being reported as an
intervention.

The fix is at source -- `ExecutorCapability.operators` plus an operator filter in `resolve` -- and the
selector's independent check stays as a fail-closed safeguard.

NO HAND-BUILT EXPECTED IDENTITIES HERE. Every capability id in this file comes from the adapter's own
declaration table via real resolution, and the arms come from the pilot's actual manifest on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

import bfcl_runtime as R                                                    # noqa: E402
from anchoropt.anchor import Action, IncisionPoint                          # noqa: E402
from anchoropt.learning import executor_capability as EC                    # noqa: E402
from anchoropt.learning.action_contract import CONTRACTS, Operator          # noqa: E402

PILOT_MANIFEST = REPO / "rounds" / "AUTORUN" / "r1" / "arm_manifest.json"
PILOT_SPECS = REPO / "rounds" / "AUTORUN" / "r1" / "controllers.json"


@pytest.fixture(autouse=True)
def _kv(monkeypatch):
    monkeypatch.setenv("ANCHOROPT_CELL", "kv")


def _resolve(operator: str, eta: dict, signal="container_at_capacity"):
    """Real resolution against the adapter's real capability table."""
    caps = R.executor_capabilities("post_execution", Action.REROUTE)
    op = Operator(operator)
    required = tuple(CONTRACTS[op].required)
    cap, why = EC.resolve(caps, boundary=IncisionPoint.POST_EXECUTION, action=Action.REROUTE,
                          signal=signal, eta=eta, required_eta=required, operator=operator)
    return cap, why


# -- the root cause ------------------------------------------------------------------------------

def test_the_two_named_capacity_executors_declare_their_operator_reading():
    """Without this declaration the cell cannot distinguish its own mechanisms."""
    caps = [c for c in R.executor_capabilities("post_execution", Action.REROUTE) if c.capability_id]
    assert caps, "the host must declare named capacity executors"
    for c in caps:
        assert c.operators, f"{c.capability_id} declares no operator reading"


def test_a_substitute_arm_does_not_resolve_to_a_transform_executor():
    """The pilot defect, from the real table: this used to return the relocation executor."""
    cap, why = _resolve("substitute", {"destination": "archival_memory_add",
                                       "argument_mapping": "{'key': 'key'}",
                                       "retry_semantics": "replace_original"})
    if cap is not None:
        assert "transform" not in (cap.operators or ()), why
        assert cap.capability_id != "relocate_entry_preserving_information_then_retry"
    else:
        assert "implements operator(s)" in why or "not 'substitute'" in why


def test_a_transform_arm_still_resolves_to_the_transform_executor():
    """The fix must not break the accepted mechanism's own resolution."""
    cap, _ = _resolve("transform", {
        "target_surface": "constrained_container_occupancy",
        "operator": "relocate_entry_preserving_information", "preservation": "x",
        "destination_operation": "archival_memory_add", "budget_from": "archival_entry",
        "retry_semantics": "retry_original_verbatim_after_capacity_freed"})
    assert cap is not None and cap.capability_id == "relocate_entry_preserving_information_then_retry"


def test_an_executor_declaring_no_operator_still_resolves_for_any_reading():
    """Backwards compatibility: a one-mechanism cell must behave exactly as before."""
    from anchoropt.learning.executor_capability import ExecutorCapability
    cap = ExecutorCapability(boundary="post_execution", action="reroute", binding="x",
                             signals=("container_at_capacity",), signal_agnostic=True,
                             eta_is_computed=True)
    got, why = EC.resolve([cap], boundary=IncisionPoint.POST_EXECUTION, action=Action.REROUTE,
                          signal="container_at_capacity", eta={"destination": "d"},
                          operator="substitute")
    assert got is cap, why


# -- replay of the real pilot manifest -----------------------------------------------------------

@pytest.mark.skipif(not PILOT_MANIFEST.exists(), reason="pilot manifest not present")
def test_every_pilot_arm_now_resolves_to_an_identity_ITS_OPERATOR_IMPLEMENTS():
    """The six arms the pilot actually emitted, re-resolved against the real table."""
    arms = json.loads(PILOT_MANIFEST.read_text())["arms"]
    reroute = [a for a in arms if a.get("action") == "reroute"]
    assert len(reroute) == 3, [a["arm_label"] for a in reroute]
    for a in reroute:
        cap, why = _resolve(a["operator"], dict(a.get("eta") or {}), signal=a["signal"])
        if cap is None:
            continue                       # refused is acceptable; a WRONG identity is not
        declared = tuple(cap.operators or ())
        assert (not declared) or a["operator"] in declared, (
            f"{a['arm_label']} -> {cap.capability_id} which implements {declared}")


@pytest.mark.skipif(not PILOT_SPECS.exists(), reason="pilot specs not present")
def test_the_pilot_specs_ON_DISK_still_carry_the_defect_and_are_preserved():
    """The frozen pilot is NOT rewritten: its record must keep showing what actually happened."""
    specs = json.loads(PILOT_SPECS.read_text())
    bad = [s for s in specs if s.get("operator") == "substitute"
           and s.get("capability_id") == "relocate_entry_preserving_information_then_retry"]
    assert len(bad) == 2, "the pilot record must preserve the 2 mismatched specs as evidence"


# -- the DUAL of the original defect: an undeclared executor masking a reading ---------------------

def _cell(*specs):
    """Build a cell from (cid, operators) pairs. Signal-agnostic and eta-computed so the ONLY thing
    under test is the operator rule."""
    from anchoropt.learning.executor_capability import ExecutorCapability
    return [ExecutorCapability(boundary="post_execution", action="reroute", capability_id=cid,
                              binding="b", signals=("sig",), operators=ops,
                              signal_agnostic=True, eta_is_computed=True)
            for cid, ops in specs]


def _res(caps, operator):
    return EC.resolve(caps, boundary=IncisionPoint.POST_EXECUTION, action=Action.REROUTE,
                      signal="sig", eta={"anything": 1}, operator=operator)


def test_an_undeclared_executor_CANNOT_mask_a_reading_in_an_ambiguous_cell():
    """Constructed and confirmed before the fix: substitute resolved to the undeclared executor.

    That is the same arm-identity failure as the original defect wearing the opposite hat -- an
    executor silently accepting a reading it may not implement.
    """
    caps = _cell(("the_transform_one", ("transform",)), ("unknown_reading", ()))
    cap, _ = _res(caps, "transform")
    assert cap is not None and cap.capability_id == "the_transform_one"
    cap, why = _res(caps, "substitute")
    assert cap is None, f"masked by {cap.capability_id if cap else None}"
    assert "declares no operator reading" in why and "ambiguous" in why
    assert "declare `operators`" in why, "the refusal must say how to fix it"


def test_a_cell_where_NOTHING_declares_a_reading_is_unchanged():
    """Backward compatibility, exactly where it is sound: one mechanism, or a host not yet migrated."""
    caps = _cell(("legacy_a", ()), ("legacy_b", ()))
    for op in ("transform", "substitute"):
        cap, why = _res(caps, op)
        assert cap is not None and cap.capability_id == "legacy_a", why


def test_a_cell_whose_executors_AGREE_on_one_reading_refuses_another_reading():
    caps = _cell(("t1", ("transform",)), ("t2", ("transform",)))
    assert _res(caps, "transform")[0].capability_id == "t1"
    assert _res(caps, "substitute")[0] is None


def test_a_properly_declared_mixed_cell_resolves_each_reading_to_its_own_executor():
    caps = _cell(("t", ("transform",)), ("s", ("substitute",)))
    assert _res(caps, "transform")[0].capability_id == "t"
    assert _res(caps, "substitute")[0].capability_id == "s"


def test_resolution_without_an_operator_argument_is_unchanged():
    """Callers that do not pass an operator keep the previous behaviour."""
    caps = _cell(("first", ("transform",)), ("second", ()))
    cap, _ = EC.resolve(caps, boundary=IncisionPoint.POST_EXECUTION, action=Action.REROUTE,
                        signal="sig", eta={"anything": 1})
    assert cap is not None and cap.capability_id == "first"


def test_the_real_bfcl_reroute_cell_is_now_unambiguous_for_its_named_executors():
    """Both named capacity executors declare transform, so the cell states its own readings."""
    caps = [c for c in R.executor_capabilities("post_execution", Action.REROUTE) if c.capability_id]
    assert caps and all("transform" in (c.operators or ()) for c in caps)
