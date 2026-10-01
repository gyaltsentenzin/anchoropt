"""The sidecar -> canonical-vocabulary translation, and above all its REFUSALS.

The whole reason criteria 3 and 4 had no production consumer is that no BFCL mechanism ever emitted
the canonical counter names. This module supplies them. The tests that matter most here are the ones
asserting that an unmeasured counter comes back ABSENT rather than 0, because a 0 would turn missing
evidence into a measured negative result and install on silence.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "bfcl_v4"
                       / "evaluator"))

from mechanism_evidence import (  # noqa: E402
    FAMILY_READINGS, _count, families_present, mechanism_evidence)

from anchoropt.learning.acceptance_criteria import (  # noqa: E402
    PASS, PENDING, check_mechanism, check_safety)


# ------------------------------------------------------------------ absent is not zero

def test_count_returns_None_when_no_step_carried_the_key():
    assert _count([{"a": 1}, {"a": 0}], "b") is None


def test_count_returns_zero_when_the_key_was_present_but_never_truthy():
    """Present-and-false is a MEASUREMENT. It must be distinguishable from absent."""
    assert _count([{"b": False}, {"b": 0}], "b") == 0


def test_a_family_with_no_declared_reading_yields_PENDING_not_a_clean_pass():
    tel, saf, prov = mechanism_evidence({"c0": [{"mystery_gate": True}]})
    assert saf is None
    assert check_mechanism(tel).verdict == PENDING
    assert check_safety(saf).verdict == PENDING


def test_suppression_only_arm_leaves_criterion_4_PENDING():
    """A suppression removes CALLS, not memory entries, so it has no clears reading. Asserting
    criterion 4 clean from that absence would be reading silence as safety."""
    steps = {"c0": [{"suppress_gate": True, "suppress_removed_n": 2}]}
    tel, saf, _ = mechanism_evidence(steps)
    assert check_mechanism(tel).verdict == PASS
    assert saf is None
    assert check_safety(saf).verdict == PENDING


# ------------------------------------------------------------------ the measured relocation arm

def _relocation_steps(n_verified=67, n_requested=84, n_declined=1, n_refused=16):
    """Shape of the measured kv relocation arm, at the reconciled denominators."""
    steps = []
    for i in range(n_requested):
        s = {"relocate_requested_gate": True, "relocate_controller": "kvrl_a1"}
        if i < n_verified:
            s["relocate_write_verified"] = True
        if i < n_declined:
            s["relocate_declined"] = "lying executor guard"
        if n_requested - i <= n_refused:
            s["relocate_declined_identity_gate"] = "bound guard"
        if i < n_verified:
            # The primitive records the invariant on the verified-removal path ONLY, which is where
            # a removal could lose information. Setting it on a request that never reached a removal
            # would model a producer the live code does not have.
            s["relocate_copy_survives_in_destination"] = True
        steps.append(s)
    return {f"c{i}": [s] for i, s in enumerate(steps)}


def test_relocation_arm_reproduces_the_reconciled_denominators():
    tel, saf, prov = mechanism_evidence(_relocation_steps(), family="relocate")
    assert tel["mechanism_requested"] == 84
    assert tel["mechanism_verified"] == 67
    assert tel["interventions_executed"] == 67
    assert saf["verified_relocations"] == 67
    assert saf["information_losing_removes"] == 0
    # provenance must name the reading, so an acceptance is traceable to the key that justified it
    assert prov["readings"]["mechanism_verified"][0]["step_key"] == "relocate_write_verified"


def test_relocation_uses_the_PRESERVATION_denominator_not_the_end_to_end_one():
    """`relocate_gate` counts 61 end-to-end successes; preservation is 67. Reading the smaller number
    would understate the safety evidence -- the conflation that was withdrawn."""
    assert FAMILY_READINGS["relocate"]["verified_relocations"] == "relocate_write_verified"
    assert "relocate_gate" not in FAMILY_READINGS["relocate"].values()


def test_an_invariant_violation_is_reported_as_information_losing():
    steps = _relocation_steps(n_verified=2, n_requested=2, n_declined=0, n_refused=0)
    for st in steps.values():
        # the removal happened and the copy was NOT found live in the destination
        st[0]["relocate_copy_survives_in_destination"] = False
    _, saf, _ = mechanism_evidence(steps, family="relocate")
    assert saf["information_losing_removes"] == 2
    assert check_safety(saf).verdict != PASS


def test_unattributed_requests_survive_translation_and_block():
    """84 requested with only 40 verified and nothing declined leaves 44 unattributed => PENDING."""
    tel, _, _ = mechanism_evidence(
        _relocation_steps(n_verified=40, n_requested=84, n_declined=0, n_refused=0),
        family="relocate")
    r = check_mechanism(tel)
    assert r.verdict == PENDING
    assert r.evidence["unattributed"] == 44


# ------------------------------------------------------------------ firing count agreement

def test_signal_firings_uses_the_same_gate_suffix_rule_as_the_validity_guards():
    """Criterion 3's firing number and channel integrity's must not be able to disagree."""
    steps = {"c0": [{"relocate_requested_gate": True}],
             "c1": [{"other_gate": True}],
             "c2": [{"nothing": 1}]}
    tel, _, _ = mechanism_evidence(steps)
    assert tel["signal_firings"] == 2


def test_families_present_detects_only_families_that_left_a_trace():
    assert families_present([{"relocate_write_verified": True}]) == ["relocate"]
    assert families_present([{"unrelated": 1}]) == []


# ------------------------------------------------------------------ criterion 4's clears counter

def test_clears_added_is_PAIRED_so_a_clear_the_control_also_made_is_not_added():
    """An arm that merely fails to prevent an existing clear has ADDED nothing.

    The invariant observable is supplied so the OTHER destructive counter is measured too --
    without it criterion 4 is PENDING for a different and also correct reason, which would make this
    test pass or fail for the wrong cause.
    """
    arm = {"c0": [{"decoded": "core_memory_clear()", "relocate_write_verified": True,
                   "relocate_copy_survives_in_destination": True}]}
    ctl = {"c0": [{"decoded": "core_memory_clear()"}]}
    _, saf, _ = mechanism_evidence(arm, control_steps=ctl, family="relocate")
    assert saf["clears_added"] == 0
    assert check_safety(saf).verdict == PASS


def test_a_clear_the_arm_introduced_is_counted_and_FAILS_criterion_4():
    arm = {"c0": [{"decoded": "core_memory_clear()", "relocate_write_verified": True,
                   "relocate_copy_survives_in_destination": True}]}
    ctl = {"c0": [{"decoded": "archival_memory_insert()"}]}
    _, saf, _ = mechanism_evidence(arm, control_steps=ctl, family="relocate")
    assert saf["clears_added"] == 1
    assert check_safety(saf).verdict == "FAIL"


def test_no_control_arm_leaves_clears_UNMEASURED_and_criterion_4_PENDING():
    """Added clears are a paired quantity. Without a control there is no measurement, and an
    unmeasured safety property must not be asserted clean."""
    arm = {"c0": [{"decoded": "x", "relocate_write_verified": True}]}
    _, saf, prov = mechanism_evidence(arm, family="relocate")
    assert "clears_added" not in (saf or {})
    assert check_safety(saf).verdict == PENDING
    assert any("PAIRED" in m for m in prov["missing"])


def test_absent_decoded_leaves_clears_UNMEASURED_rather_than_zero():
    """A dropped observation is not evidence of safety."""
    arm = {"c0": [{"relocate_write_verified": True}]}
    ctl = {"c0": [{}]}
    _, saf, prov = mechanism_evidence(arm, control_steps=ctl, family="relocate")
    assert "clears_added" not in (saf or {})
    assert any("Absent is NOT zero" in m for m in prov["missing"])


def test_the_full_paired_relocation_arm_reaches_a_COMPLETE_criterion_4():
    """With both arms supplying `decoded` and no added clears, criterion 4 finally PASSES on
    measurement rather than on assumption."""
    arm = _relocation_steps(n_verified=10, n_requested=14, n_declined=2, n_refused=2)
    for st in arm.values():
        st[0]["decoded"] = "archival_memory_insert()"
    ctl = {c: [{"decoded": "archival_memory_insert()"}] for c in arm}
    tel, saf, _ = mechanism_evidence(arm, control_steps=ctl, family="relocate")
    assert check_mechanism(tel).verdict == PASS
    assert check_safety(saf).verdict == PASS
    assert saf == {"verified_relocations": 10, "information_losing_removes": 0, "clears_added": 0}


def test_interventions_executed_is_SYNTHESIZED_not_read_from_a_raw_step_key():
    """The translator must not depend on a step literally carrying `interventions_executed`.

    Measured on the live isolated BV sidecar after patching its allowlist: the four canonical
    `mechanism_*` keys and the three safety counters all survive, but a raw `interventions_executed`
    key is DROPPED -- it has no `mechanism_` prefix, no `_gate` suffix, and is not in `_STEP_FIELDS`.

    That drop is harmless only because this translator SYNTHESIZES the count from the family's
    declared `executed` reading (for relocation, `relocate_write_verified`), which does survive. The
    property is load-bearing: `check_mechanism` reads `interventions_executed` first and treats zero
    as a categorical FAIL, so a translator that read the raw key would turn a working controller into
    a measured refutation on nothing but an allowlist omission. This test pins the synthesis so a
    later "simplification" to a direct read cannot reintroduce it.
    """
    steps = {"c0": [{"relocate_controller": "x", "relocate_requested_gate": True,
                     "relocate_write_verified": True, "relocate_removed_verified": True,
                     "relocate_copy_survives_in_destination": True}]}
    assert not any("interventions_executed" in s for st in steps.values() for s in st)
    tel, _saf, prov = mechanism_evidence(steps)
    assert tel["interventions_executed"] == 1
    src = [r for r in prov["readings"]["interventions_executed"]]
    assert src[0]["step_key"] == "relocate_write_verified", (
        "the count must be traceable to the family reading that produced it")


# ------------------------------------------------- the write-only-on-failure reading defect
#
# `capacity_relocate.py` sets `relocate_invariant_violated` ONLY inside `if not copy_survives`. A
# violation flag written only when it trips cannot distinguish a clean arm from an unmeasured one --
# the key is absent in both -- so declaring it as criterion 4's reading made C4 PENDING_VALIDATION
# for every clean relocation arm, with no input that could ever PASS. The four tests below pin the
# correction: read the positive invariant and negate it.

def _clean_relocate_step(**kw):
    s = {"relocate_controller": "x", "relocate_requested_gate": True,
         "relocate_write_verified": True, "relocate_removed_verified": True,
         "relocate_copy_survives_in_destination": True,
         "decoded": "core_memory_append(...)"}
    s.update(kw)
    return s


_CTL = {"c0": [{"decoded": "core_memory_append(...)"}]}


def test_a_CLEAN_relocation_arm_can_actually_PASS_criterion_4():
    """The load-bearing one. Before the fix this returned PENDING for a perfectly safe arm."""
    from anchoropt.learning.acceptance_criteria import PASS, check_safety
    _tel, safety, prov = mechanism_evidence(
        {"c0": [_clean_relocate_step() for _ in range(3)]}, control_steps=_CTL)
    assert safety["information_losing_removes"] == 0
    assert check_safety(safety).verdict == PASS
    # and the number must be traceable to a negation, not look like a violation count
    assert prov["readings"]["information_losing_removes"][0]["negated"] is True


def test_a_removal_whose_copy_did_not_survive_is_counted_and_FAILS():
    from anchoropt.learning.acceptance_criteria import FAIL, check_safety
    _tel, safety, _p = mechanism_evidence(
        {"c0": [_clean_relocate_step(),
                _clean_relocate_step(relocate_copy_survives_in_destination=False),
                _clean_relocate_step()]}, control_steps=_CTL)
    assert safety["information_losing_removes"] == 1
    assert check_safety(safety).verdict == FAIL


def test_an_ABSENT_invariant_observable_stays_PENDING_and_is_never_a_clean_zero():
    """Negating a reading must not convert a dropped measurement into evidence of safety."""
    from anchoropt.learning.acceptance_criteria import PENDING, check_safety
    bare = {"relocate_controller": "x", "relocate_requested_gate": True,
            "relocate_write_verified": True, "decoded": "core_memory_append(...)"}
    _tel, safety, prov = mechanism_evidence({"c0": [bare]}, control_steps=_CTL)
    assert "information_losing_removes" not in safety
    assert check_safety(safety).verdict == PENDING
    assert any("information_losing_removes" in m for m in prov["missing"])


def test_the_violation_only_flag_would_NOT_have_been_measurable():
    """Documents WHY the declaration changed: the old key is absent on a clean arm, so `_count`
    returns None and the counter is omitted -- indistinguishable from never having measured it."""
    from mechanism_evidence import _count
    clean = [_clean_relocate_step() for _ in range(3)]
    assert _count(clean, "relocate_invariant_violated") is None
    assert _count(clean, "relocate_copy_survives_in_destination", negated=True) == 0

# ------------------------------------------------- a slot with SEVERAL declared readings

def test_two_independent_guards_are_SUMMED_into_one_refused_counter():
    """A mechanism refused by two different guards only partitions if BOTH are counted.

    The relocate family is refused by an identity gate and, separately, by an episode bound. Counting
    one and dropping the other leaves the requested population short, which criterion 3 correctly
    reports as unattributed -- so the reading has to sum. The guards are mutually exclusive per step
    (a request refused on identity never reaches the bound check), so a sum cannot double-count.
    """
    steps = [{"relocate_requested_gate": True, "relocate_declined_identity_gate": "wrong cap"},
             {"relocate_requested_gate": True, "relocate_declined_bound_gate": "episode_bound_reached(3/3)"},
             {"relocate_requested_gate": True, "relocate_declined_bound_gate": "step_budget_exhausted(9/9)"}]
    tel, _saf, prov = mechanism_evidence({"c": steps}, family="relocate")
    assert tel["mechanism_refused_by_guard"] == 3
    keys = {r["step_key"] for r in prov["readings"]["mechanism_refused_by_guard"]}
    assert keys == {"relocate_declined_identity_gate", "relocate_declined_bound_gate"}


def test_a_slot_whose_readings_are_ALL_absent_is_still_omitted_not_zeroed():
    """The whole point of the None contract. Adding a second reading must not turn a slot that
    nothing measured into a clean zero -- that is the defect this module exists to prevent."""
    steps = [{"relocate_requested_gate": True, "relocate_write_verified": True}]
    tel, _saf, prov = mechanism_evidence({"c": steps}, family="relocate")
    assert "mechanism_refused_by_guard" not in tel
    assert any("mechanism_refused_by_guard" in m for m in prov["missing"])


def test_one_present_reading_resolves_the_slot_even_if_the_other_is_absent():
    """Partial availability is a real state: the identity gate can be present while the bound guard
    never trips. The slot must resolve on what WAS measured rather than refusing wholesale."""
    steps = [{"relocate_requested_gate": True, "relocate_declined_identity_gate": "wrong cap"}]
    tel, _saf, _prov = mechanism_evidence({"c": steps}, family="relocate")
    assert tel["mechanism_refused_by_guard"] == 1


def test_the_measured_84_request_population_SUMS_once_the_bound_reason_exists():
    """Job 1834414 reproduced 84 = 67 verified + 1 declined + 16 silent, and criterion 3 refused on
    the 16. With the bound reason recorded the population partitions exactly and criterion 3 passes,
    which is what makes this a MEASUREMENT fix rather than a threshold change."""
    steps = []
    for _ in range(67):
        steps.append({"relocate_requested_gate": True, "relocate_write_verified": True,
                      "relocate_copy_survives_in_destination": True})
    steps.append({"relocate_requested_gate": True, "relocate_declined": "lying executor"})
    for _ in range(16):
        steps.append({"relocate_requested_gate": True,
                      "relocate_declined_bound_gate": "episode_bound_reached(3/3)"})
    tel, _saf, _prov = mechanism_evidence({"c": steps}, family="relocate")
    assert tel["mechanism_requested"] == 84
    assert tel["mechanism_verified"] == 67
    assert tel["mechanism_declined_with_reason"] == 1
    assert tel["mechanism_refused_by_guard"] == 16
    r = check_mechanism(tel)
    assert r.evidence["unattributed"] == 0
    assert r.verdict == PASS


def test_the_bound_reason_key_is_ALWAYS_truthy_so_the_sidecar_cannot_drop_it():
    """`relocate_declined_bound_gate` survives the sidecar on the *_gate TRUTHY-suffix rule, so a
    falsy value would be silently dropped -- reproducing the very defect the key was added to fix.

    The patch builds the reason as `",".join(_rl_why) or "condition_false"`, and the fallback is
    what makes the guarantee unconditional: an empty reason list still yields a non-empty string.
    Pinned here because the property lives in a patch script's generated text, where no other test
    would notice it regressing.
    """
    for gates_disabled in (False, True):
        for bound_ok in (True, False):
            for step_ok in (True, False):
                why = []
                if gates_disabled:
                    why.append("gates_disabled")
                if not bound_ok:
                    why.append("episode_bound_reached(3/3)")
                if not step_ok:
                    why.append("step_budget_exhausted(9/9)")
                value = ",".join(why) or "condition_false"
                assert value, (gates_disabled, bound_ok, step_ok)


def test_the_bound_reason_patch_emits_the_or_fallback():
    """Guards the source of that guarantee: if the `or` fallback is ever dropped from the patch, the
    test above still passes on its own copy of the logic, so the patch text itself must be checked."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    text = (repo / "patches" / "bv" / "bv_reloc_bound_reason.py").read_text()
    assert 'or "condition_false"' in text
    assert 'step_record["relocate_declined_bound_gate"]' in text
