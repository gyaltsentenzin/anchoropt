"""The whole chain, offline: grounded arm -> installed controller -> real predicate -> real store.

This is the pre-GPU gate. Each link here has broken independently in a way that produced a measured
number for an intervention that never ran:

  grounding      an infeasible operator ground and was emitted (the reduce-on-slots arm)
  installation   a spec installed without its predicate module; the arm silently became the control
  predicate      a declared signal raised on import and `fires_on` reported False
  identity       a spec named one executor while another would have run
  execution      a repair reported success from a return string while nothing landed

A source marker, a successful install or a simulated predicate test is NOT evidence of live execution --
so this drives the REAL store class and asserts on LIVE state.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "benchmarks" / "bfcl_v4", REPO / "scripts",
          REPO / "benchmarks" / "bfcl_v4" / "harness"):
    sys.path.insert(0, str(p))

import bfcl_runtime as R                                                    # noqa: E402
import capacity_relocate as CR                                              # noqa: E402
from anchoropt.anchor import IncisionPoint                                  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_kv import (  # noqa: E402
    MemoryAPI_kv)
from bfcl_regression_host import (_ensure_declared_signal_seam,             # noqa: E402
                                  _load_installer)

WANT_CAP = "relocate_entry_preserving_information_then_retry"


@pytest.fixture
def arm(monkeypatch):
    monkeypatch.setenv("ANCHOROPT_CELL", "kv")
    g = [x for x in R.ground_transforms("container_at_capacity", IncisionPoint.POST_EXECUTION)
         if not x.get("infeasible")]
    assert len(g) == 1, g
    return g[0]


@pytest.fixture
def controller(arm):
    _ensure_declared_signal_seam()
    spec = {"name": "container_at_capacity", "locus": "post_execution", "action": "reroute",
            "operator": "transform", "variant": arm["variant"], "eta": arm["eta"],
            "capability_id": arm["grounding"]["capability_id"], "phase": "prereq",
            "predicate": {"declared_signal": "container_at_capacity", "params": {}}}
    return _load_installer().SpecPredicate(spec), spec


@pytest.fixture
def full_core():
    api = MemoryAPI_kv()
    api._load_scenario({"model_result_dir": pathlib.Path(tempfile.mkdtemp()),
                        "test_id": "memory_kv_0-x-0", "scenario": "x", "long_context": False})
    for i in range(7):
        assert "error" not in api.core_memory_add(key="fact_%s" % "abcdefg"[i], value="value %d" % i)
    return api


def _executor(api):
    def run(calls):
        return [eval("api." + c.strip(), {"api": api}) for c in calls], None   # noqa: S307
    return run


def test_link1_the_arm_grounds_with_the_right_capability(arm):
    assert arm["variant"] == "relocate_entry_for_kv"
    assert arm["grounding"]["capability_id"] == WANT_CAP


def test_link2_the_real_installer_builds_it(controller):
    ctl, _ = controller
    assert ctl.phase == "prereq" and ctl.name


def test_link3_the_predicate_discriminates(controller):
    """Fires on the slot refusal; silent on the character cap it cannot help."""
    ctl, _ = controller
    assert ctl.fires_on({"error_kind": "no_capacity", "proposes_write": True}) is True
    assert ctl.fires_on({"error_kind": "blob_would_overflow", "proposes_write": True}) is False
    assert ctl.fires_on({"error_kind": None, "proposes_write": True}) is False


def test_link4_the_identity_gate_admits_this_spec_and_refuses_others(controller):
    """Transcribed from patches/bv/bv_generic_capacity_relocate.py."""
    _, spec = controller

    def admits(s):
        return (str(s.get("capability_id") or "") == WANT_CAP
                and str(s.get("operator") or s.get("action") or "")
                in ("transform", "substitute", "reroute"))

    assert admits(spec)
    assert not admits({**spec, "capability_id": "reduce_payload_preserving_facts_and_replace"})
    assert not admits({**spec, "capability_id": ""})


def test_link5_the_refusal_is_real_before_the_repair(full_core):
    r = full_core.core_memory_add(key="new_fact", value="x")
    assert r.get("error") == "Core memory is full. Please clear some entries."


def test_the_whole_chain_lands_the_refused_write_and_loses_nothing(full_core):
    before = dict(full_core.core_memory)
    out = CR.relocate_and_retry(
        failing_call="core_memory_add(key='new_fact', value='the refused fact')",
        involved_instances=[full_core], execute=_executor(full_core), relocations_so_far=0)

    # every verification step, on LIVE state
    assert out["relocate_write_verified"] and out["relocate_removed_verified"]
    assert out["relocate_copy_survives_in_destination"]
    assert out["relocate_slot_freed"] and out["relocate_retry_landed"] and out["relocate_ok"]

    # the outcome the residual is about
    assert full_core.core_memory["new_fact"] == "the refused fact"
    after = {**full_core.archival_memory, **full_core.core_memory}
    lost = {k: v for k, v in before.items() if after.get(k) != v}
    assert not lost, f"information lost: {lost}"
    assert len(full_core.core_memory) == 7 and len(full_core.archival_memory) == 1


def test_the_repair_issues_no_clear(full_core):
    """208 of 218 refusals were followed by a wholesale clear. This must add none."""
    calls = []

    def logging_exec(cs):
        calls.extend(cs)
        return _executor(full_core)(cs)

    CR.relocate_and_retry(failing_call="core_memory_add(key='new_fact', value='v')",
                          involved_instances=[full_core], execute=logging_exec)
    assert not [c for c in calls if "clear" in c.lower()]
