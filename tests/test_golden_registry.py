"""The anti-yo-yo registry: accepted controllers survive until explicitly superseded.

Every test here corresponds to a way we actually lost a working mechanism.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchoropt.learning.golden_registry import (
    AcceptedController, ControllerIdentity, ExecutionTelemetry, GoldenRegistry, PairedOutcome,
    RegistryRefusal, RuntimeProvenance)

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "rounds" / "GOLDEN" / "registry.json"


def _entry(name="c", version=1, supersedes="", origin="autonomous", **kw):
    base = dict(
        name=name,
        identity=ControllerIdentity("post_execution", "sig", "reroute", "transform", "cap", "prereq"),
        spec={"signal": "sig"},
        provenance=RuntimeProvenance(split="train", model="m"),
        telemetry=ExecutionTelemetry(requested=10, fired=3, episodes=2, acted=3),
        outcome=PairedOutcome(arm_correct=5, control_correct=3, n_scored=10,
                              gains=("a", "b"), losses=(), case_ids={"a": True, "b": True}),
        origin=origin, version=version, supersedes=supersedes)
    base.update(kw)
    return AcceptedController(**base)


# -- the core rule ----------------------------------------------------------------------------

def test_an_accepted_controller_is_never_silently_overwritten():
    reg = GoldenRegistry([_entry("c", version=1)])
    with pytest.raises(RegistryRefusal, match="already accepted"):
        reg.register(_entry("c", version=2))          # no supersedes -> refused
    assert reg.get("c").version == 1


def test_superseding_requires_naming_the_prior_and_a_higher_version():
    reg = GoldenRegistry([_entry("c", version=1)])
    reg.register(_entry("c", version=2, supersedes="c"))
    assert reg.get("c").version == 2
    with pytest.raises(RegistryRefusal, match="must increase"):
        reg.register(_entry("c", version=2, supersedes="c"))


def test_missing_entry_raises_an_actionable_refusal_not_none():
    reg = GoldenRegistry([_entry("c")])
    with pytest.raises(RegistryRefusal, match="known:"):
        reg.get("nope")


# -- fixture quality --------------------------------------------------------------------------

def test_installation_is_not_execution():
    """fired=0 must be refused: an installed spec that never ran is not a baseline."""
    with pytest.raises(RegistryRefusal, match="installation is not execution"):
        GoldenRegistry([_entry(telemetry=ExecutionTelemetry(requested=9, fired=0, acted=0))])


def test_capability_identity_is_mandatory():
    with pytest.raises(RegistryRefusal, match="capability_id"):
        GoldenRegistry([_entry(identity=ControllerIdentity("b", "s", "reroute", "transform", "", ""))])


def test_per_case_reference_is_mandatory_for_an_autonomous_result():
    with pytest.raises(RegistryRefusal, match="equal total can hide offsetting flips"):
        GoldenRegistry([_entry(outcome=PairedOutcome(arm_correct=5, control_correct=3, n_scored=10,
                                                     gains=("a", "b"), losses=(), case_ids={}))])


def test_internally_inconsistent_gain_loss_sets_are_refused():
    with pytest.raises(RegistryRefusal, match="!= net"):
        GoldenRegistry([_entry(outcome=PairedOutcome(arm_correct=5, control_correct=3, n_scored=10,
                                                     gains=("a",), losses=(), case_ids={"a": True}))])


def test_a_manual_positive_control_is_held_to_a_lighter_bar_but_kept_distinct():
    """Manual anchors probe the adapter's capabilities; they are NOT autonomous results."""
    reg = GoldenRegistry([_entry("manual", origin="manual_positive_control",
                                 provenance=RuntimeProvenance(),
                                 telemetry=ExecutionTelemetry(),
                                 outcome=PairedOutcome())])
    assert reg.positive_controls() and not reg.autonomous()


# -- composition ------------------------------------------------------------------------------

def test_same_cell_different_capability_is_flagged():
    """Two controllers at one (boundary, action) with different executors: the wrong one can win."""
    reg = GoldenRegistry([_entry("acc")])
    conflicts = reg.composition_conflicts(
        ControllerIdentity("post_execution", "other", "reroute", "transform", "OTHER_CAP", "prereq"))
    assert conflicts and "different capability_id" in conflicts[0]
    assert "smoke-test BOTH" in conflicts[0]


def test_identical_signal_at_the_same_cell_can_shadow_the_accepted_one():
    reg = GoldenRegistry([_entry("acc")])
    conflicts = reg.composition_conflicts(
        ControllerIdentity("post_execution", "sig", "reroute", "transform", "cap", "prereq"))
    assert any("leave acc inert" in c for c in conflicts)


def test_a_different_boundary_is_not_a_conflict():
    reg = GoldenRegistry([_entry("acc")])
    assert reg.composition_conflicts(
        ControllerIdentity("pre_generation", "sig", "reroute", "transform", "cap", "prereq")) == ()


# -- persistence + the real fixtures ----------------------------------------------------------

def test_round_trip_preserves_per_case_outcomes(tmp_path):
    reg = GoldenRegistry([_entry("c")])
    back = GoldenRegistry.load(reg.save(tmp_path / "r.json"))
    assert back.get("c").outcome.case_ids == {"a": True, "b": True}
    assert back.get("c").identity.capability_id == "cap"


def test_the_two_autonomous_acceptances_are_registered_and_valid():
    reg = GoldenRegistry.load(REGISTRY_PATH)
    names = {e.name for e in reg.autonomous()}
    assert {"a4_zero_call_reprompt_vector", "rec_sum_capacity_recovery"} <= names
    for e in reg.all():
        e.validate()
        assert e.telemetry.proves_execution(), f"{e.name} has no execution evidence"
        assert e.positive_states, f"{e.name} has no discriminating positive state"


def test_a4_and_rec_sum_reference_values_are_pinned():
    """The numbers themselves are the regression baseline."""
    reg = GoldenRegistry.load(REGISTRY_PATH)
    a4 = reg.get("a4_zero_call_reprompt_vector")
    assert (a4.outcome.arm_correct, a4.outcome.control_correct, a4.outcome.n_scored) == (23, 19, 89)
    assert a4.outcome.net == 4 and len(a4.outcome.gains) == 4 and len(a4.outcome.losses) == 0
    assert a4.telemetry.fired == 26
    rs = reg.get("rec_sum_capacity_recovery")
    assert (rs.outcome.arm_correct, rs.outcome.control_correct, rs.outcome.n_scored) == (59, 54, 109)
    assert rs.outcome.net == 5 and len(rs.outcome.gains) == 7 and len(rs.outcome.losses) == 2
    assert rs.telemetry.fired == 21 and rs.telemetry.acted == 21 and rs.telemetry.declined == 0
    assert rs.version == 2


def test_the_preservation_caveat_is_preserved_and_not_described_as_loss_free():
    """Standing instruction: never restore the 100% claim, never call the aggregate a guarantee."""
    reg = GoldenRegistry.load(REGISTRY_PATH)
    blob = " ".join(reg.get("rec_sum_capacity_recovery").caveats)
    assert "0.534" in blob and "0.286" in blob
    assert "NOT a per-episode guarantee" in blob
    assert "RETRACTED" in blob
    assert "loss-free" in blob.lower()      # present only as the explicit negation
    assert "NOT LOSS-FREE" in blob.upper()


def test_criterion_2_is_recorded_as_non_regression_not_a_required_gain():
    """A train gain with no held-out loss is a full PASS; the registry must say so."""
    reg = GoldenRegistry.load(REGISTRY_PATH)
    for name in ("a4_zero_call_reprompt_vector", "rec_sum_capacity_recovery"):
        e = reg.get(name)
        txt = (" ".join(e.caveats) + " " + json.dumps(e.spec)).lower()
        assert "no aggregate regression" in txt or "non-regression" in txt, name
