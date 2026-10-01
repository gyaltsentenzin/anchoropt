"""A8's full mechanism must be READABLE by criteria 3 and 4, not merely executable.

Caught before the measuring jobs finished, which is the only reason it cost nothing: the A8 executor
emits `dedup_clear_*` keys, `families_present` derives families ONLY from declared reading keys, and
nothing in FAMILY_READINGS matched them. So C3 and C4 would have reported PENDING_VALIDATION on a
mechanism that ran correctly -- the same allowlist/reading class that has swallowed intervention
telemetry repeatedly on this project.

The table's own docstring states the rule this pins: "A family absent from this table contributes NO
canonical counters... the remedy is to declare the reading alongside the mechanism, in the same commit."
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchoropt.learning.acceptance_criteria import check_mechanism, check_safety   # noqa: E402

_p = ROOT / "benchmarks" / "bfcl_v4" / "evaluator" / "mechanism_evidence.py"
_spec = importlib.util.spec_from_file_location("me", _p)
me = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(me)


def _arm(*steps):
    return {"case": list(steps)}


FIRED_VERIFIED = {"dedup_clear_requested_gate": True, "dedup_clear_gate": True,
                  "dedup_clear_verified_gate": True, "dedup_clear_copies_remaining": 1,
                  "dedup_clear_victim_removed": True}
FIRED_VIOLATED = {"dedup_clear_requested_gate": True, "dedup_clear_gate": True,
                  "dedup_clear_verified_gate": False,
                  "dedup_clear_invariant_violated_gate": "copies_remaining=0"}
DECLINED = {"dedup_clear_requested_gate": True,
            "dedup_clear_reason_gate": "container not at capacity -- not interfering"}


# ---------------------------------------------------------------------------------------------------
# THE FAMILY MUST RESOLVE AT ALL
# ---------------------------------------------------------------------------------------------------

def test_the_dedup_clear_family_has_a_declared_reading():
    """Without this, every A8 measurement reports PENDING however well the mechanism ran."""
    assert "dedup_clear" in me.FAMILY_READINGS


def test_the_arms_own_keys_RESOLVE_to_that_family():
    """`families_present` matches on declared reading VALUES, so the declaration must name the keys
    the executor actually writes -- not keys that merely look related."""
    _tel, _safety, prov = me.mechanism_evidence(_arm(FIRED_VERIFIED))
    assert prov["families_read"] == ["dedup_clear"], prov
    # `missing` legitimately names readings NO STEP CARRIED -- a one-firing arm has no declined and no
    # refused branch, and a single arm has no control for the paired clears counter. That is the
    # module reporting absent OBSERVATIONS, which is exactly what keeps a gap from becoming a clean
    # zero. What must not appear is "no declared reading", which is the undeclared-family case.
    assert not any("no declared reading" in m for m in prov["missing"]), prov["missing"]


# ---------------------------------------------------------------------------------------------------
# CRITERION 3 MUST DISCRIMINATE
# ---------------------------------------------------------------------------------------------------

def test_a_verified_eviction_PASSES_criterion_3():
    tel, _s, _p = me.mechanism_evidence(_arm(FIRED_VERIFIED))
    assert tel["interventions_executed"] == 1
    assert tel["mechanism_verified"] == 1
    assert check_mechanism(tel).verdict == "PASS"


def test_an_UNVERIFIED_eviction_FAILS_criterion_3():
    """The mechanism acted and could not confirm a copy survived. That is the one outcome A8's
    acceptance argument forbids, and it must fail rather than go unnoticed."""
    tel, _s, _p = me.mechanism_evidence(_arm(FIRED_VIOLATED))
    assert tel["interventions_executed"] == 1
    assert tel["mechanism_verified"] == 0
    assert check_mechanism(tel).verdict == "FAIL"


def test_verified_reads_the_VERIFICATION_not_the_firing():
    """If `verified` read `dedup_clear_gate`, a violating arm would count as verified. It must read
    `dedup_clear_verified_gate`, which is only true after the live-state check passed."""
    assert me.FAMILY_READINGS["dedup_clear"]["verified"] == "dedup_clear_verified_gate"
    assert me.FAMILY_READINGS["dedup_clear"]["verified"] != \
        me.FAMILY_READINGS["dedup_clear"]["executed"]


def test_a_declining_request_is_ATTRIBUTED_not_silently_missing():
    """Every non-firing branch records a reason, so the requested population sums. This is the defect
    the relocate bound guard had until its reason was added: key absence is not a recorded reason."""
    tel, _s, _p = me.mechanism_evidence(_arm(FIRED_VERIFIED, DECLINED))
    assert tel["mechanism_requested"] == 2
    assert tel["mechanism_declined_with_reason"] == 1
    assert tel["interventions_executed"] == 1


def test_an_arm_that_only_DECLINED_cannot_claim_attribution():
    """Zero executions means any delta is unattributable, whatever the accuracy did."""
    tel, _s, _p = me.mechanism_evidence(_arm(DECLINED))
    assert check_mechanism(tel).verdict == "FAIL"


# ---------------------------------------------------------------------------------------------------
# CRITERION 4 -- and why the counter is a NEGATION
# ---------------------------------------------------------------------------------------------------

def test_criterion_4_PASSES_on_a_clean_paired_arm():
    """`clears_added` is PAIRED (arm minus control) from the model's actual calls, so it needs both
    sides -- which is what real scoring supplies."""
    arm = _arm(dict(FIRED_VERIFIED, decoded=["core_memory_add(key='a',value='b')"]))
    ctl = {"case": [{"decoded": ["core_memory_clear(user_id='u')"]}]}
    _tel, safety, _p = me.mechanism_evidence(arm, control_steps=ctl)
    assert safety["information_losing_removes"] == 0
    assert safety["clears_added"] == 0
    assert check_safety(safety).verdict == "PASS"


def test_the_counter_is_a_CONJUNCTION_not_a_negation_of_the_confirmation():
    """It was declared as _Negated('dedup_clear_verified_gate') -- count every firing that did not
    verify. R6f showed that is wrong: an eviction whose store ended UNCHANGED did not verify and lost
    nothing, so the negation charged a violation the mechanism had not committed. The honest counter is
    victim_removed AND copies_remaining == 0, which needs two observables and so is derived."""
    reading = me.FAMILY_READINGS["dedup_clear"]["information_losing_removes"]
    assert reading == me._DERIVED, "must be the derived conjunction, not a single-key negation"


def test_an_arm_PREVENTING_clears_is_never_charged_for_them():
    """A8 exists to stop destructive clears. An arm with FEWER clears than its control must not be
    penalised by the paired counter."""
    arm = _arm(dict(FIRED_VERIFIED, decoded=["core_memory_add(key='a',value='b')"]))
    ctl = {"case": [{"decoded": ["core_memory_clear(user_id='u')",
                                 "core_memory_clear(user_id='v')"]}]}
    _tel, safety, _p = me.mechanism_evidence(arm, control_steps=ctl)
    assert safety["clears_added"] == 0, "preventing clears must never read as adding them"


# ---------------------------------------------------------------------------------------------------
# THE THIRD OUTCOME -- acted, calls landed, net effect nil. Measured on R6f.
# ---------------------------------------------------------------------------------------------------

ACTED_NO_EFFECT = {"dedup_clear_requested_gate": True, "dedup_clear_gate": True,
                   "dedup_clear_verified_gate": False, "dedup_clear_victim_removed": False,
                   "dedup_clear_copies_remaining": 2, "dedup_clear_copies_before": 2}
VERIFIED = {"dedup_clear_requested_gate": True, "dedup_clear_gate": True,
            "dedup_clear_verified_gate": True, "dedup_clear_victim_removed": True,
            "dedup_clear_copies_remaining": 1, "dedup_clear_copies_before": 2}
REALLY_LOSSY = {"dedup_clear_requested_gate": True, "dedup_clear_gate": True,
                "dedup_clear_verified_gate": False, "dedup_clear_victim_removed": True,
                "dedup_clear_copies_remaining": 0, "dedup_clear_copies_before": 1}


def test_a_NO_OP_firing_is_not_charged_as_information_loss():
    """R6f's eighth eviction: both calls succeeded, the retry re-added the key the eviction removed, so
    the container ended as it began. Nothing was lost. Charging it would FAIL criterion 4 on a violation
    the mechanism did not commit -- the same error as hiding one, in the opposite direction."""
    _tel, safety, _p = me.mechanism_evidence(
        {"c": [dict(VERIFIED, decoded=["core_memory_add(key='a',value='b')"])] * 7
              + [ACTED_NO_EFFECT]}, control_steps={"c": [{"decoded": []}]})
    assert safety["information_losing_removes"] == 0
    assert check_safety(safety).verdict == "PASS"


def test_a_GENUINE_last_copy_destruction_still_fails_criterion_4():
    """The counter must not have been loosened into uselessness by the no-op fix."""
    _tel, safety, _p = me.mechanism_evidence(
        {"c": [dict(REALLY_LOSSY, decoded=[])]}, control_steps={"c": [{"decoded": []}]})
    assert safety["information_losing_removes"] == 1
    assert check_safety(safety).verdict == "FAIL"


def test_the_requested_population_SUMS_with_the_third_outcome():
    """Criterion 3 demands requested == verified + declined + refused. R6f's actual distribution left
    ONE unattributed until the no-effect outcome had its own counter, and check_mechanism correctly
    returned PENDING for it: 'a request with no recorded reason is missing evidence, not safe
    behaviour'."""
    steps = {"c": [VERIFIED] * 7 + [ACTED_NO_EFFECT] + [DECLINED] * 18}
    tel, _s, _p = me.mechanism_evidence(steps)
    assert tel["mechanism_requested"] == 26
    assert tel["mechanism_verified"] == 7
    assert tel["mechanism_declined_with_reason"] == 18
    assert tel["mechanism_refused_by_guard"] == 1, "the no-effect firing must be counted"
    total = (tel["mechanism_verified"] + tel["mechanism_declined_with_reason"]
             + tel["mechanism_refused_by_guard"])
    assert total == tel["mechanism_requested"], f"population must sum: {tel}"
    assert check_mechanism(tel).verdict == "PASS"


def test_a_no_effect_firing_is_provenance_distinguishable_from_a_guard_refusal():
    """They share the sum bucket; they must not be confusable in a report."""
    _tel, _s, prov = me.mechanism_evidence({"c": [VERIFIED, ACTED_NO_EFFECT]})
    reads = prov["readings"].get("mechanism_refused_by_guard", [])
    assert any(r.get("derived") == "acted_no_effect" for r in reads), reads


def test_an_unrecorded_observable_stays_PENDING_rather_than_becoming_zero():
    """A dropped measurement must not read as a clean arm."""
    bare = {"dedup_clear_requested_gate": True, "dedup_clear_gate": True,
            "dedup_clear_verified_gate": True}
    _tel, safety, prov = me.mechanism_evidence({"c": [bare]},
                                               control_steps={"c": [{"decoded": []}]})
    # `safety` is None (not {}) when nothing resolved, which is stronger still: criterion 4 then reports
    # "no safety telemetry -- absence of evidence of harm is not evidence of no harm".
    assert safety is None or "information_losing_removes" not in safety, \
        f"without victim_removed the counter must be ABSENT, not 0: {safety}"
    assert check_safety(safety).verdict == "PENDING_VALIDATION"
    assert any("derived" in m for m in prov["missing"]), prov["missing"]
