"""The core-capacity relocation primitive, driven against the REAL KV store class.

Not a mock: `MemoryAPI_kv` from the benchmark's own func_source_code, so the caps, the error strings and
the key-format rules are the ones the model actually hits.

Every test here corresponds to a safety requirement: destination write before source removal, live-state
verification, the source intact on failure, precise removal, retry only after capacity is real, and a
bounded budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4" / "harness"))

import capacity_relocate as CR                                          # noqa: E402
# THE REAL STORE CLASS, imported as a package member so its own metaclass import resolves.
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_kv import (  # noqa: E402
    MAX_ARCHIVAL_MEMORY_SIZE, MAX_CORE_MEMORY_SIZE, MemoryAPI_kv)


@pytest.fixture(autouse=True)
def _snapshot_dir(tmp_path):
    """The real store's metaclass wants the harness's snapshot config. Give it a real temp dir."""
    global _CFG
    _CFG = {"model_result_dir": tmp_path, "test_id": "memory_kv_0-test-0", "scenario": "test",
            "long_context": False}
    return _CFG


def _full_core(n=MAX_CORE_MEMORY_SIZE):
    api = MemoryAPI_kv()
    api._load_scenario(_CFG)
    for i in range(n):
        r = api.core_memory_add(key="fact_%s" % "abcdefghij"[i], value="value number %d" % i)
        assert "error" not in r, r
    return api


def _executor(api, log=None):
    """Runs a call string against the real store, the way the host's _execute does."""
    def run(calls):
        out = []
        for c in calls:
            if log is not None:
                log.append(c)
            out.append(eval("api." + c.strip(), {"api": api}))       # noqa: S307 - test harness
        return out, None
    return run


def test_the_backend_really_is_slot_limited_and_archival_is_more_permissive():
    """The measured premise: relocation can work only because the destination is larger."""
    assert MAX_CORE_MEMORY_SIZE == 7 and MAX_ARCHIVAL_MEMORY_SIZE == 50
    api = _full_core()
    r = api.core_memory_add(key="one_more", value="x")
    assert r.get("error") == "Core memory is full. Please clear some entries."
    # Shortening does NOT help: the constraint is slots, not characters.
    assert api.core_memory_add(key="one_more", value="")["error"].startswith("Core memory is full")


def test_a_verified_relocation_frees_a_slot_and_the_retry_lands():
    api = _full_core()
    log = []
    res = CR.relocate_and_retry(
        failing_call="core_memory_add(key='new_fact', value='the fact that was refused')",
        involved_instances=[api], execute=_executor(api, log))
    assert res["relocate_ok"], res
    assert res["relocate_write_verified"] and res["relocate_removed_verified"]
    assert res["relocate_slot_freed"] and res["relocate_retry_landed"]
    # ORDER: destination write strictly before source removal.
    assert log[0].startswith("archival_memory_add"), log
    assert log[1].startswith("core_memory_remove"), log
    assert log[2].startswith("core_memory_add"), log
    # The new fact landed, the victim survives in archival, and nothing was destroyed.
    assert api.core_memory["new_fact"] == "the fact that was refused"
    vk = res["relocate_victim_key"]
    assert vk not in api.core_memory and api.archival_memory[vk] == "value number 0"
    assert len(api.core_memory) == MAX_CORE_MEMORY_SIZE


def test_no_information_is_lost_across_the_whole_operation():
    """Every fact present before must still be reachable somewhere after."""
    api = _full_core()
    before = dict(api.core_memory)
    res = CR.relocate_and_retry(
        failing_call="core_memory_add(key='new_fact', value='v')",
        involved_instances=[api], execute=_executor(api))
    assert res["relocate_ok"]
    after = {**api.archival_memory, **api.core_memory}
    missing = {k: v for k, v in before.items() if after.get(k) != v}
    assert not missing, f"information lost: {missing}"


def test_a_failed_destination_write_leaves_the_SOURCE_UNCHANGED():
    """The critical ordering guarantee: if preservation fails, nothing is removed."""
    api = _full_core()
    before = dict(api.core_memory)

    def refuse_add(calls):
        # The destination write fails; everything else would have worked.
        return [{"error": "simulated destination failure"} for _ in calls], None

    res = CR.relocate_and_retry(
        failing_call="core_memory_add(key='new_fact', value='v')",
        involved_instances=[api], execute=refuse_add)
    assert not res["relocate_ok"]
    assert "did NOT land" in res["relocate_declined"]
    assert res["relocate_preserved_source_intact"] is True
    assert api.core_memory == before, "a failed preservation must not remove anything"
    assert "new_fact" not in api.core_memory


def test_verification_reads_LIVE_state_not_the_result_string():
    """A return string once made 32/32 repairs look successful while only 4 writes landed."""
    api = _full_core()
    before = dict(api.core_memory)

    def lying_executor(calls):
        # Claims success, writes nothing.
        return [{"status": "Key added."} for _ in calls], None

    res = CR.relocate_and_retry(
        failing_call="core_memory_add(key='new_fact', value='v')",
        involved_instances=[api], execute=lying_executor)
    assert not res["relocate_ok"] and "live state disagrees" in res["relocate_declined"]
    assert api.core_memory == before


def test_removal_targets_precisely_the_verified_entry():
    api = _full_core()
    log = []
    res = CR.relocate_and_retry(
        failing_call="core_memory_add(key='new_fact', value='v')",
        involved_instances=[api], execute=_executor(api, log))
    vk = res["relocate_victim_key"]
    removes = [c for c in log if c.startswith("core_memory_remove")]
    assert len(removes) == 1 and ("key='%s'" % vk) in removes[0]


def test_the_budget_is_bounded():
    api = _full_core()
    res = CR.relocate_and_retry(
        failing_call="core_memory_add(key='new_fact', value='v')",
        involved_instances=[api], execute=_executor(api),
        relocations_so_far=CR.MAX_RELOCATIONS_PER_EPISODE)
    assert not res["relocate_ok"] and "bounded" in res["relocate_declined"]
    assert len(api.core_memory) == MAX_CORE_MEMORY_SIZE, "a declined repair must change nothing"


def test_a_full_destination_declines_rather_than_risking_a_loss():
    api = _full_core()
    # Fill the destination to its cap directly: the point is a full destination, not how it got there.
    for i in range(MAX_ARCHIVAL_MEMORY_SIZE):
        api.archival_memory["arch_key_%s" % chr(ord("a") + i % 26) + "_%d" % i] = "v"
    res = CR.relocate_and_retry(
        failing_call="core_memory_add(key='new_fact', value='v')",
        involved_instances=[api], execute=_executor(api))
    assert not res["relocate_ok"] and "itself at capacity" in res["relocate_declined"]


def test_protected_keys_are_never_chosen():
    api = _full_core()
    protect = list(api.core_memory)[:-1]
    res = CR.relocate_and_retry(
        failing_call="core_memory_add(key='new_fact', value='v')",
        involved_instances=[api], execute=_executor(api), protect=protect)
    assert res["relocate_ok"]
    assert res["relocate_victim_key"] == list(protect and api.archival_memory)[0]
    assert res["relocate_victim_key"] not in protect


def test_an_entry_already_copied_in_the_destination_is_preferred():
    """Cheapest possible slot: relocating it cannot cost information."""
    api = _full_core()
    dup = list(api.core_memory)[3]
    api.archival_memory[dup] = api.core_memory[dup]
    res = CR.relocate_and_retry(
        failing_call="core_memory_add(key='new_fact', value='v')",
        involved_instances=[api], execute=_executor(api))
    assert res["relocate_victim_key"] == dup
    assert res["relocate_victim_already_in_destination"] is True


def test_unavailable_live_stores_decline_explicitly():
    class Opaque:
        pass
    res = CR.relocate_and_retry(failing_call="core_memory_add(key='k', value='v')",
                                involved_instances=[Opaque()], execute=lambda c: ([], None))
    assert not res["relocate_ok"] and "live stores unavailable" in res["relocate_declined"]


def test_no_retry_happens_unless_a_slot_was_actually_freed():
    """'Retry only after capacity has actually been freed' -- asserted, not assumed."""
    api = _full_core()
    log = []
    real = _executor(api, log)

    def remove_is_a_noop(calls):
        if calls and calls[0].startswith("core_memory_remove"):
            return [{"status": "Key removed."}], None      # lies; the key stays
        return real(calls)

    res = CR.relocate_and_retry(
        failing_call="core_memory_add(key='new_fact', value='v')",
        involved_instances=[api], execute=remove_is_a_noop)
    assert not res["relocate_ok"] and "did NOT land" in res["relocate_declined"]
    assert not any(c.startswith("core_memory_add(key='new_fact'") for c in log)
    assert res["relocate_copy_survives_in_destination"] is True


def test_the_primitive_encodes_no_trigger_or_threshold():
    """It is an OPERATION, not a policy: no signal name, no condition, no priority."""
    src = (REPO / "scripts" / "capacity_relocate.py").read_text()
    for forbidden in ("error_kind", "container_at_capacity", "no_capacity", "fires_on",
                      "declared_signal"):
        assert forbidden not in src, f"the primitive encodes a trigger: {forbidden!r}"
