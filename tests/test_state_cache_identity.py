"""A cache key for mutable state must cover whatever can change that state.

The measured defect, reproduced here on hosts that are not memory benchmarks: two arms with different
controllers computed the SAME key for the state their scored work is graded against. Nothing broke
only because each arm happened to get its own directory. The key itself was blind.

Both directions are load-bearing:
  * a state-building controller MUST change the key, or one arm is scored against another's state;
  * a post-construction controller must NOT change it, or caching is defeated for every downstream
    controller and the fix costs more than the defect.
"""

from __future__ import annotations

import json

import pytest

from anchoropt.learning.state_cache_identity import (
    STATE_BUILDING_PHASES, StateKeyInputs, collision_report, shared_state_key, state_cache_key)

BASE = {"corpus": "shard-A", "model": "m-1", "code_fp": "abc123", "flags": {"gates": "on"}}
CTL_A = {"boundary": "on_write", "signal": "at_capacity", "action": "relocate"}
CTL_B = {"boundary": "on_write", "signal": "at_capacity", "action": "evict"}


def _k(controller=None, phase=None, base=None):
    return state_cache_key(StateKeyInputs(base or BASE, controller, phase))


# ============================================================ THE DEFECT

def test_two_arms_with_DIFFERENT_state_building_controllers_get_DIFFERENT_keys():
    """THE REGRESSION TEST. This is the measured defect: both arms produced one key."""
    a, b = _k(CTL_A, "setup"), _k(CTL_B, "setup")
    assert a.key != b.key
    assert a.covers_controller and b.covers_controller


def test_the_control_and_an_arm_do_not_collide():
    """The control installs nothing; the arm installs a state-building controller."""
    assert _k(None).key != _k(CTL_A, "setup").key


def test_the_same_controller_reuses_the_same_state_which_is_CORRECT_caching():
    """The fix must not turn every run into a rebuild."""
    assert _k(CTL_A, "setup").key == _k(CTL_A, "setup").key


@pytest.mark.parametrize("phase", sorted(STATE_BUILDING_PHASES))
def test_every_declared_state_building_phase_enters_the_key(phase):
    assert _k(CTL_A, phase).covers_controller


# ============================================================ THE CONVERSE, WHICH MATTERS AS MUCH

def test_a_POST_CONSTRUCTION_controller_must_NOT_change_the_key():
    """It cannot have changed state that was already built. Folding it in would force a rebuild per
    downstream controller and defeat caching -- a fix that costs more than the defect it prevents."""
    assert _k(CTL_A, "query").key == _k(None).key
    assert not _k(CTL_A, "query").covers_controller


def test_two_different_post_construction_controllers_SHARE_state_legitimately():
    assert _k(CTL_A, "query").key == _k(CTL_B, "scored").key


def test_an_UNDECLARED_phase_is_treated_as_state_affecting():
    """Conservative on purpose: a needless rebuild is the cheap error; cross-arm reuse is not."""
    assert _k(CTL_A, None).covers_controller
    assert _k(CTL_A, None).key != _k(None).key


# ============================================================ the base inputs still count

def test_a_change_to_any_base_input_still_changes_the_key():
    assert _k(base={**BASE, "code_fp": "zzz"}).key != _k().key


def test_the_key_is_order_insensitive_and_reproducible():
    """A key that depends on dict ordering is not an identity."""
    b1 = {"a": 1, "b": {"x": 1, "y": 2}}
    b2 = {"b": {"y": 2, "x": 1}, "a": 1}
    assert _k(base=b1).key == _k(base=b2).key


def test_the_key_explains_itself():
    assert "can change the cached state" in _k(CTL_A, "setup").reason
    assert "cannot have changed it" in _k(CTL_A, "query").reason


# ============================================================ deliberate sharing

def test_declared_sharing_gives_two_arms_ONE_key_on_purpose():
    """The matched-construction design NEEDS this: holding construction constant is how a
    delayed-effect controller's post-construction effect is isolated at all."""
    a = shared_state_key(StateKeyInputs(BASE, CTL_A, "setup"), declared_by="R2-PHASE-MATCHED")
    b = shared_state_key(StateKeyInputs(BASE, None), declared_by="R2-PHASE-MATCHED")
    assert a.key == b.key and a.shared_declared
    assert "DELIBERATELY excluded" in a.reason and "R2-PHASE-MATCHED" in a.reason


def test_sharing_must_be_ATTRIBUTABLE():
    """Unattributed sharing is indistinguishable from the blind collision."""
    with pytest.raises(ValueError):
        shared_state_key(StateKeyInputs(BASE, CTL_A, "setup"), declared_by="")


# ============================================================ the auditor

def test_a_blind_collision_is_REPORTED():
    """Simulates the measured round: two different controllers, one key, nothing declared."""
    from anchoropt.learning.state_cache_identity import StateKey
    rep = collision_report({"arm": StateKey("same", False, "ctl_a", "blind"),
                            "control": StateKey("same", False, "ctl_b", "blind")})
    assert not rep["ok"] and len(rep["blind_collisions"]) == 1
    assert "scored against another" in rep["detail"]


def test_DECLARED_sharing_is_not_reported_as_a_fault():
    """Treating every shared key as a fault would forbid the matched-construction protocol."""
    a = shared_state_key(StateKeyInputs(BASE, CTL_A, "setup"), declared_by="proto")
    b = shared_state_key(StateKeyInputs(BASE, None), declared_by="proto")
    rep = collision_report({"arm": a, "control": b})
    assert rep["ok"] and len(rep["declared_sharing"]) == 1


def test_the_strict_keys_produce_no_collision_at_all():
    rep = collision_report({"arm": _k(CTL_A, "setup"), "control": _k(None)})
    assert rep["ok"] and not rep["declared_sharing"]


def test_same_controller_sharing_a_key_is_not_a_collision():
    rep = collision_report({"run1": _k(CTL_A, "setup"), "run2": _k(CTL_A, "setup")})
    assert rep["ok"]


def test_report_is_json_serialisable():
    assert json.loads(json.dumps(collision_report({"a": _k(CTL_A, "setup")})))["ok"] is True


def test_the_module_names_no_benchmark():
    import pathlib

    import anchoropt.learning.state_cache_identity as mod
    src = pathlib.Path(mod.__file__).read_text().lower()
    for w in ("bfcl", "gorilla", "granite", "snapshot_store", "memory_evaluator"):
        assert w not in src, f"core module references {w!r}"
