"""The persistence lifecycle: DISCOVERED -> PENDING -> ACCEPTED, and what must NOT be forgotten.

Every test here corresponds to a way this project actually lost accumulated progress, so each one
names the failure it pins rather than just the method it calls.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest                                                            # noqa: E402

from anchoropt.learning.candidate_library import (                        # noqa: E402
    ACCEPTED, DISCOVERED, MEASURED_NEGATIVE, PENDING_VALIDATION, STATES, SUPERSEDED,
    CandidateLibrary, CandidateRecord, EvaluationContext, Measurement, stack_fingerprint)
from anchoropt.learning.golden_registry import ControllerIdentity, RegistryRefusal  # noqa: E402


def ident(signal="container_at_capacity", action="reroute", operator="transform",
          boundary="post_execution", capability_id="relocate_then_retry", phase="prereq"):
    return ControllerIdentity(boundary=boundary, signal=signal, action=action, operator=operator,
                              capability_id=capability_id, phase=phase)


def rec(name="c1", state=DISCOVERED, measurement=None, context=None, identity=None, **kw):
    return CandidateRecord(name=name, identity=identity or ident(), spec={"signal": "s", "eta": {}},
                           state=state, measurement=measurement,
                           context=context or EvaluationContext(), **kw)


def meas(arm=22, ctl=18, n=105, gains=("g1", "g2", "g3", "g4", "g5"), losses=("l1",), **kw):
    return Measurement(arm_correct=arm, control_correct=ctl, n_scored=n, gains=gains,
                       losses=losses, **kw)


def ctx(stack=("k1",), cell="kv", token="results/h0/run"):
    return EvaluationContext(incumbent_stack=tuple(stack), cell=cell, incumbent_token=token,
                             split="kv train", model="granite-4.1-8b")


# ---------------------------------------------------------------------------------------------------
# THE STATES
# ---------------------------------------------------------------------------------------------------

def test_the_five_states_the_user_asked_for_all_exist():
    assert STATES == (DISCOVERED, PENDING_VALIDATION, MEASURED_NEGATIVE, ACCEPTED, SUPERSEDED)


def test_pending_validation_is_NOT_an_exclusion():
    """THE central rule. Four round-1 candidates were evaluated by an instrument that eliminated the
    phase they act in; that is not a refutation, and they must stay measurable."""
    r = rec(state=PENDING_VALIDATION)
    assert r.excludes_re_evaluation is False
    assert r.excludes_against(("anything",)) is False


def test_measured_negative_and_accepted_DO_exclude():
    assert rec(state=MEASURED_NEGATIVE, measurement=meas(arm=18, ctl=18, gains=(), losses=()),
               context=ctx()).excludes_re_evaluation is True
    assert rec(state=ACCEPTED, measurement=meas(), context=ctx()).excludes_re_evaluation is True


def test_discovered_does_not_exclude_because_it_was_never_measured():
    assert rec(state=DISCOVERED).excludes_re_evaluation is False


# ---------------------------------------------------------------------------------------------------
# THE MOVING INCUMBENT
# ---------------------------------------------------------------------------------------------------

def test_a_negative_verdict_expires_when_the_incumbent_MOVES():
    """A controller measured negative against H0 is not thereby negative against H0+A: the
    composition changed, so the old comparison says nothing about the new one."""
    r = rec(state=MEASURED_NEGATIVE, measurement=meas(arm=18, ctl=18, gains=(), losses=()),
            context=ctx(stack=()))
    assert r.excludes_against(()) is True            # same incumbent -> still excluded
    assert r.excludes_against(("newly_installed",)) is False   # moved -> eligible again


def test_accepted_exclusion_is_permanent_across_stacks():
    """ACCEPTED is a fact about history, not about one comparison."""
    r = rec(state=ACCEPTED, measurement=meas(), context=ctx(stack=("k1",)))
    assert r.excludes_against(("k1",)) is True
    assert r.excludes_against(("k1", "k2", "k3")) is True


def test_installed_stack_reports_the_accepted_controllers():
    """The call whose ABSENCE caused every round to restart from H0."""
    lib = CandidateLibrary()
    lib.upsert(rec("a", state=ACCEPTED, measurement=meas(), context=ctx(),
                   identity=ident(signal="s_a", capability_id="cap_a")))
    lib.upsert(rec("b", state=PENDING_VALIDATION, identity=ident(signal="s_b",
                                                                 capability_id="cap_b")))
    stack = lib.installed_stack()
    assert len(stack) == 1 and "s_a" in stack[0]


def test_installed_stack_is_empty_for_a_fresh_library_and_that_is_H0():
    assert CandidateLibrary().installed_stack() == ()
    assert stack_fingerprint(()) == "H0"


def test_stack_fingerprint_is_order_insensitive_but_content_sensitive():
    assert stack_fingerprint(("a", "b")) == stack_fingerprint(("b", "a"))
    assert stack_fingerprint(("a", "b")) != stack_fingerprint(("a", "c"))
    assert stack_fingerprint(("a",)) != stack_fingerprint(())


# ---------------------------------------------------------------------------------------------------
# REFUSALS -- each one a silent-corruption path
# ---------------------------------------------------------------------------------------------------

def test_a_verdict_state_REQUIRES_a_measurement():
    with pytest.raises(RegistryRefusal, match="requires a measurement"):
        rec(state=ACCEPTED).validate()


def test_an_INVALID_measurement_cannot_produce_a_verdict():
    """R2's byte-identical runs: a net of 0 between two identical runs is the absence of a
    measurement, and must not be bankable as a negative result."""
    m = meas(arm=18, ctl=18, gains=(), losses=(), valid=False,
             invalid_reason="NO_OPPORTUNITY: acting phase did not execute")
    with pytest.raises(RegistryRefusal, match="marked invalid"):
        rec(state=MEASURED_NEGATIVE, measurement=m, context=ctx()).validate()


def test_a_verdict_REQUIRES_naming_the_incumbent_it_was_measured_against():
    with pytest.raises(RegistryRefusal, match="naming the incumbent"):
        rec(state=ACCEPTED, measurement=meas(),
            context=EvaluationContext(cell="kv")).validate()


def test_an_internally_inconsistent_measurement_is_refused():
    with pytest.raises(RegistryRefusal, match="internally inconsistent"):
        rec(state=ACCEPTED, measurement=meas(arm=25, ctl=18), context=ctx()).validate()


def test_a_spec_is_required_because_a_record_must_be_re_executable():
    r = CandidateRecord(name="x", identity=ident(), spec={})
    with pytest.raises(RegistryRefusal, match="spec is empty"):
        r.validate()


def test_capability_id_is_required_in_the_identity():
    r = CandidateRecord(name="x", identity=ident(capability_id=""), spec={"a": 1})
    with pytest.raises(RegistryRefusal, match="capability_id is empty"):
        r.validate()


def test_an_ACCEPTED_record_cannot_be_silently_un_accepted():
    """The yo-yo arriving through the back door: a later round re-emits the same spec as a fresh
    discovery and the acceptance evaporates."""
    lib = CandidateLibrary()
    lib.upsert(rec("a", state=ACCEPTED, measurement=meas(), context=ctx()))
    with pytest.raises(RegistryRefusal, match="would silently un-accept"):
        lib.upsert(rec("a", state=DISCOVERED))


def test_advancing_a_state_IS_allowed():
    lib = CandidateLibrary()
    lib.upsert(rec("a", state=DISCOVERED))
    lib.upsert(rec("a", state=PENDING_VALIDATION))
    lib.upsert(rec("a", state=ACCEPTED, measurement=meas(), context=ctx()))
    assert lib.get("a").state == ACCEPTED


def test_unknown_state_is_refused_rather_than_coerced():
    with pytest.raises(RegistryRefusal, match="not in"):
        rec(state="probably_fine").validate()


def test_superseded_must_name_what_replaced_it():
    with pytest.raises(RegistryRefusal, match="must name what replaced it"):
        rec(state=SUPERSEDED).validate()


# ---------------------------------------------------------------------------------------------------
# ELIGIBILITY
# ---------------------------------------------------------------------------------------------------

def test_an_UNKNOWN_mechanism_is_eligible_not_excluded():
    """Absence is not exclusion -- the same error class as reading silence as a result."""
    assert CandidateLibrary().eligible(["never_seen"]) == ("never_seen",)


def test_eligible_filters_only_what_the_stack_makes_settled():
    lib = CandidateLibrary()
    lib.upsert(rec("acc", state=ACCEPTED, measurement=meas(), context=ctx(stack=("k1",))))
    lib.upsert(rec("neg", state=MEASURED_NEGATIVE, measurement=meas(arm=18, ctl=18, gains=(),
                                                                    losses=()),
                   context=ctx(stack=("k1",)), identity=ident(signal="s2")))
    lib.upsert(rec("pend", state=PENDING_VALIDATION, identity=ident(signal="s3")))
    assert lib.eligible(["acc", "neg", "pend", "new"], stack=("k1",)) == ("pend", "new")
    # the incumbent moved: the negative expires, the acceptance does not
    assert lib.eligible(["acc", "neg", "pend", "new"], stack=("k1", "k2")) == ("neg", "pend", "new")


# ---------------------------------------------------------------------------------------------------
# ROUND-TRIP -- the "restart the session" milestone
# ---------------------------------------------------------------------------------------------------

def test_a_fresh_process_recovers_the_incumbent_and_the_pending_queue(tmp_path):
    lib = CandidateLibrary()
    lib.upsert(rec("acc", state=ACCEPTED, measurement=meas(), context=ctx(stack=("k0",)),
                   round_id="R4", provenance={"job": "1840001"}))
    lib.upsert(rec("pend", state=PENDING_VALIDATION, identity=ident(signal="s_pend"),
                   notes=("NO_OPPORTUNITY: query-time protocol eliminated the prereq phase",)))
    lib.upsert(rec("neg", state=MEASURED_NEGATIVE, identity=ident(signal="s_neg"),
                   measurement=meas(arm=17, ctl=18, gains=(), losses=("l1",)), context=ctx()))
    p = lib.save(tmp_path / "candidates.json")

    back = CandidateLibrary.load(p)
    assert len(back) == 3
    assert back.installed_stack() == lib.installed_stack()
    assert [r.name for r in back.pending()] == ["pend"]
    assert back.get("acc").measurement.net == 4
    assert back.get("neg").measurement.net == -1
    assert back.get("pend").notes[0].startswith("NO_OPPORTUNITY")
    assert back.get("acc").provenance["job"] == "1840001"
    assert back.get("acc").context.cell == "kv"


def test_loading_a_missing_file_is_an_empty_library_not_an_error(tmp_path):
    assert len(CandidateLibrary.load(tmp_path / "nope.json")) == 0


def test_load_REVALIDATES_so_a_hand_edited_file_cannot_poison_a_round(tmp_path):
    lib = CandidateLibrary()
    lib.upsert(rec("a", state=ACCEPTED, measurement=meas(), context=ctx()))
    p = lib.save(tmp_path / "c.json")
    raw = p.read_text().replace('"arm_correct": 22', '"arm_correct": 99')
    p.write_text(raw)
    with pytest.raises(RegistryRefusal, match="internally inconsistent"):
        CandidateLibrary.load(p)


# ---------------------------------------------------------------------------------------------------
# COVERAGE -- reporting only, and matched on IDENTITY not on a name that sounds similar
# ---------------------------------------------------------------------------------------------------

def test_coverage_matches_on_identity_and_reports_what_is_missing():
    lib = CandidateLibrary()
    lib.upsert(rec("mine", state=ACCEPTED, measurement=meas(), context=ctx(),
                   identity=ident(signal="container_at_capacity")))
    reference = {
        "A1_capacity": {"boundary": "post_execution", "signal": "container_at_capacity",
                        "action": "reroute", "operator": "transform",
                        "capability_id": "relocate_then_retry", "phase": "prereq"},
        "A4_zero_call": {"boundary": "post_generation_pre_exec", "signal": "no_tool_call_at_all",
                         "action": "reprompt", "operator": "reprompt",
                         "capability_id": "inject_and_regenerate", "phase": "query"},
    }
    out = lib.coverage_against(reference)
    assert out["n_recovered"] == 1
    assert out["recovered"]["A1_capacity"] == "mine [ACCEPTED]"
    assert out["not_recovered"] == ["A4_zero_call"]


def test_a_near_miss_identity_does_NOT_count_as_recovered():
    """Same signal, different capability: a different mechanism under a similar name."""
    lib = CandidateLibrary()
    lib.upsert(rec("mine", identity=ident(capability_id="something_else")))
    out = lib.coverage_against({"A1": {"boundary": "post_execution",
                                       "signal": "container_at_capacity", "action": "reroute",
                                       "operator": "transform",
                                       "capability_id": "relocate_then_retry",
                                       "phase": "prereq"}})
    assert out["n_recovered"] == 0 and out["not_recovered"] == ["A1"]
