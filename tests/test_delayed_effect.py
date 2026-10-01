"""The delayed-effect contract, tested on hosts that have nothing to do with memory stores.

Two families of test carry the weight.

THE WIDENING MUST WORK. A controller that acts during state construction and is scored later must be
creditable through the host's declared dependency graph, because the alternative -- demanding a
firing inside the scored unit's own trace -- makes a whole class of consequential intervention
unmeasurable by construction.

THE WIDENING MUST NOT LAUNDER. Every way the original defect could slip through must STILL fail:
an unmatched construction protocol, a scored unit with an empty cone, a graph inferred from id
prefixes, telemetry borrowed from the wrong phase. If those passed, this module would be a way to
accept the exact round it was written in response to -- so `test_the_original_confounded_round_is_
STILL_rejected` is the single most important test in the file.

Hosts used here are a CI fixture pipeline and a build cache. Neither is a memory benchmark; if these
tests only made sense for BFCL the module would not be generic.
"""

from __future__ import annotations

import json

import pytest

from anchoropt.learning.acceptance_criteria import PENDING, check_mechanism
from anchoropt.learning.delayed_effect import (  # noqa: F401
    NO_OPPORTUNITY,
    DELAYED, DISCONNECTED, IMMEDIATE, UNSUPPORTED, acting_site,
    attribute_through_dependencies, causal_cone, evaluate_delayed_effect,
    matched_initial_conditions, phase_scoped_verification)

# ---------------------------------------------------------------------------------------- host A
# A CI agent that edits shared fixtures during a setup phase; later tests read them.
CI_PHASES = {"fixture:db": "setup", "fixture:auth": "setup",
             "test:login": "scored", "test:billing": "scored", "test:health": "scored"}
CI_GRAPH = {"test:login": ["fixture:auth"], "test:billing": ["fixture:auth", "fixture:db"],
            "test:health": []}

# ---------------------------------------------------------------------------------------- host B
# A build agent that populates a cache during a warm phase; later compilations read it.
BUILD_PHASES = {"warm:deps": "warm", "compile:app": "build", "compile:cli": "build"}
BUILD_GRAPH = {"compile:app": ["warm:deps"], "compile:cli": ["warm:deps"]}


# ================================================================== recording the acting site

def test_acting_site_reports_the_phase_the_controller_ACTUALLY_ran_in():
    site = acting_site({"fixture:auth": True, "fixture:db": True}, CI_PHASES,
                       declared_phase="setup")
    assert site.phase == "setup" and site.fired and site.phase_as_declared
    assert site.units == frozenset({"fixture:auth", "fixture:db"})


def test_a_controller_that_ran_nowhere_is_reported_as_such_not_as_missing_data():
    site = acting_site({"fixture:auth": False}, CI_PHASES, declared_phase="setup")
    assert not site.fired and site.phase == "setup"


def test_running_outside_the_declared_phase_is_visible():
    """The spec said setup; it fired on scored units. That disagreement is itself a finding."""
    site = acting_site({"test:login": True}, CI_PHASES, declared_phase="setup")
    assert site.phase == "scored"
    assert not site.phase_as_declared


def test_a_phase_declared_any_is_never_out_of_phase():
    assert acting_site({"test:login": True}, CI_PHASES, declared_phase="any").phase_as_declared


def test_unlabelled_firings_are_kept_as_unknown_not_silently_dropped():
    """Discarding an unlabelled firing is how a controller that ran comes to look inert."""
    site = acting_site({"mystery:unit": True}, CI_PHASES)
    assert site.fired and site.phase == "unknown" and "mystery:unit" in site.units


def test_the_acting_phase_is_deterministic_under_a_tie():
    a = acting_site({"fixture:db": True, "test:login": True}, CI_PHASES)
    b = acting_site({"test:login": True, "fixture:db": True}, CI_PHASES)
    assert a.phase == b.phase  # not dict-order dependent


# ================================================================== the graph

def test_causal_cone_is_transitive_and_cycle_safe():
    g = {"c": ["b"], "b": ["a"], "a": ["c"]}  # a cycle
    assert causal_cone("c", g) == ["b", "a"]


def test_a_unit_with_no_declared_dependencies_has_an_empty_cone():
    assert causal_cone("test:health", CI_GRAPH) == []


# ================================================================== the widening WORKS

def test_a_setup_phase_firing_is_credited_to_the_scored_test_that_depends_on_it():
    """THE POINT OF THE MODULE. The controller never ran during test:login, and must still be
    creditable for it, because test:login declares that it reads what fixture:auth built."""
    site = acting_site({"fixture:auth": True}, CI_PHASES, declared_phase="setup")
    att = attribute_through_dependencies(["test:login"], site, CI_GRAPH)
    assert att.mode == DELAYED
    assert att.attributed == frozenset({"test:login"})
    assert att.cones["test:login"] == ["fixture:auth"]


def test_the_same_structure_holds_for_a_build_cache_host():
    """If this only worked for one host's vocabulary it would not be generic."""
    site = acting_site({"warm:deps": True}, BUILD_PHASES, declared_phase="warm")
    att = attribute_through_dependencies(["compile:app", "compile:cli"], site, BUILD_GRAPH)
    assert att.mode == DELAYED and att.n_attributed == 2


def test_transitive_dependencies_are_credited_not_just_direct_ones():
    g = {"scored": ["mid"], "mid": ["deep"]}
    site = acting_site({"deep": True}, {"deep": "setup", "scored": "scored"})
    assert attribute_through_dependencies(["scored"], site, g).n_attributed == 1


def test_a_same_unit_firing_is_IMMEDIATE_not_delayed():
    site = acting_site({"test:login": True}, CI_PHASES)
    assert attribute_through_dependencies(["test:login"], site, CI_GRAPH).mode == IMMEDIATE


# ================================================================== the widening must NOT LAUNDER

def test_a_gained_unit_with_an_EMPTY_cone_stays_unattributed():
    """test:health declares no dependencies, so a setup firing cannot explain its gain. This is the
    check that keeps the contract honest: widening where evidence may be found must not widen what
    counts as evidence."""
    site = acting_site({"fixture:auth": True}, CI_PHASES, declared_phase="setup")
    att = attribute_through_dependencies(["test:health"], site, CI_GRAPH)
    assert att.unattributed == frozenset({"test:health"}) and att.n_attributed == 0
    assert att.mode == DISCONNECTED


def test_a_firing_on_an_unrelated_setup_unit_does_not_travel():
    """fixture:db is real, fired, and in the setup phase -- and test:login does not read it."""
    site = acting_site({"fixture:db": True}, CI_PHASES, declared_phase="setup")
    assert attribute_through_dependencies(["test:login"], site, CI_GRAPH).n_attributed == 0


def test_zero_firings_anywhere_is_still_DISCONNECTED():
    site = acting_site({}, CI_PHASES, declared_phase="setup")
    att = attribute_through_dependencies(["test:login"], site, CI_GRAPH)
    assert att.mode == DISCONNECTED and att.n_attributed == 0


def test_an_absent_graph_credits_only_same_unit_firings():
    """A host that declares no dependencies gets no delayed credit -- it does not get free credit.
    Inferring a graph from id prefixes here would invent provenance the host never asserted."""
    site = acting_site({"fixture:auth": True}, CI_PHASES, declared_phase="setup")
    att = attribute_through_dependencies(["test:login"], site, {})
    assert att.mode == DISCONNECTED and att.n_attributed == 0
    assert "no dependency graph" in att.detail


# ================================================================== matched initial conditions

def test_matched_when_both_arms_start_and_construct_identically():
    m = matched_initial_conditions({"initial_state_fingerprint": "s0", "construction_protocol": "p1"},
                                   {"initial_state_fingerprint": "s0", "construction_protocol": "p1"})
    assert m.ok and m.initial_equivalent and m.construction_matched


def test_downstream_divergence_is_PERMITTED_because_that_is_the_whole_point():
    """The intervention is supposed to change future state. Only the START and the CONSTRUCTION
    PROTOCOL must match; forbidding divergence would forbid this class of controller entirely."""
    m = matched_initial_conditions(
        {"initial_state_fingerprint": "s0", "construction_protocol": "p1", "final_state": "X"},
        {"initial_state_fingerprint": "s0", "construction_protocol": "p1", "final_state": "Y"})
    assert m.ok


def test_different_construction_protocol_is_CONFOUNDED_even_from_the_same_start():
    """THE ORIGINAL DEFECT. Same starting state, but only the arm repaired the state the scored
    units are graded against -- so the delta measures the state, not the mechanism."""
    m = matched_initial_conditions(
        {"initial_state_fingerprint": "s0", "construction_protocol": "controller_active"},
        {"initial_state_fingerprint": "s0", "construction_protocol": "plain"})
    assert m.initial_equivalent and not m.construction_matched and not m.ok
    assert "confounded with construction" in m.detail


def test_different_initial_state_fails_outright():
    m = matched_initial_conditions({"initial_state_fingerprint": "s0"},
                                   {"initial_state_fingerprint": "s9"})
    assert not m.initial_equivalent and not m.ok


@pytest.mark.parametrize("arm,ctl", [
    ({}, {}),
    ({"initial_state_fingerprint": "s0"}, {}),
    ({}, {"initial_state_fingerprint": "s0"}),
])
def test_an_UNREPORTED_initial_state_is_not_a_matched_one(arm, ctl):
    """'Neither arm reported' and 'both reported the same' are different claims."""
    assert not matched_initial_conditions(arm, ctl).ok


def test_a_matched_start_with_unreported_construction_is_not_matched_construction():
    m = matched_initial_conditions({"initial_state_fingerprint": "s0"},
                                   {"initial_state_fingerprint": "s0"})
    assert m.initial_equivalent and not m.construction_matched
    assert "unreported" in m.detail


# ================================================================== verification in the right phase

def test_verification_uses_the_ACTING_phase_telemetry():
    site = acting_site({"fixture:auth": True}, CI_PHASES, declared_phase="setup")
    tel, note = phase_scoped_verification(
        {"setup": {"mechanism_verified": 12}, "scored": {"mechanism_verified": 0}}, site)
    assert tel == {"mechanism_verified": 12}
    assert "setup" in note


def test_missing_acting_phase_telemetry_yields_PENDING_never_a_substitution():
    """Falling back to another phase's telemetry is the substitution that caused the original
    defect. An empty slice makes check_mechanism PENDING, which is the honest verdict."""
    site = acting_site({"fixture:auth": True}, CI_PHASES, declared_phase="setup")
    tel, note = phase_scoped_verification({"scored": {"mechanism_verified": 99}}, site)
    assert tel == {}
    assert check_mechanism(tel).verdict == PENDING
    assert "must NOT be substituted" in note


def test_no_firing_means_no_acting_phase_to_verify_in():
    tel, note = phase_scoped_verification({"setup": {"mechanism_verified": 5}},
                                          acting_site({}, CI_PHASES))
    assert tel == {} and "absent rather than negative" in note


# ================================================================== the whole contract

def _report(**kw):
    base = dict(
        gained_units=["test:login"], firings_by_unit={"fixture:auth": True}, phase_of=CI_PHASES,
        graph=CI_GRAPH, telemetry_by_phase={"setup": {"interventions_executed": 2,
                                                      "mechanism_verified": 2}},
        arm_conditions={"initial_state_fingerprint": "s0", "construction_protocol": "p1"},
        control_conditions={"initial_state_fingerprint": "s0", "construction_protocol": "p1"},
        declared_phase="setup")
    base.update(kw)
    return evaluate_delayed_effect(**base)


def test_a_clean_delayed_intervention_is_INTERPRETABLE():
    r = _report()
    assert r.interpretable and r.channel_ok and r.contamination_free
    assert r.attribution.mode == DELAYED
    assert r.verification_telemetry["mechanism_verified"] == 2


def test_the_original_confounded_round_is_STILL_rejected():
    """THE REGRESSION TEST FOR THE DEFECT THIS MODULE ANSWERS.

    Shape of the measured round, in a CI host's vocabulary: the controller fired during construction,
    the gained unit declares NO dependency on anything it touched, and only the arm ran the
    controller during construction. Both failure modes are reported, separately, because the remedies
    differ -- one needs a better mechanism, the other a better protocol. If this test ever passes,
    the module has become a way to launder the round it was written in response to.
    """
    r = _report(gained_units=["test:health"],  # declares no dependencies
                control_conditions={"initial_state_fingerprint": "s0",
                                    "construction_protocol": "plain"})
    assert not r.interpretable
    assert not r.channel_ok, "an empty causal cone must not be creditable"
    assert not r.contamination_free, "unmatched construction must remain a validity failure"
    assert r.attribution.mode == DISCONNECTED


def test_confounded_construction_alone_blocks_even_with_a_perfect_channel():
    r = _report(control_conditions={"initial_state_fingerprint": "s0",
                                    "construction_protocol": "plain"})
    assert r.channel_ok and not r.contamination_free and not r.interpretable


def test_a_perfect_protocol_with_no_channel_also_blocks():
    r = _report(gained_units=["test:health"])
    assert r.contamination_free and not r.channel_ok and not r.interpretable


def test_a_host_without_phases_or_a_graph_is_UNSUPPORTED_not_refuted():
    """'This host cannot express delayed effects' is a statement about the host. It must be
    distinguishable from 'this controller has no causal channel'."""
    r = _report(phase_of={}, graph={})
    assert not r.supported and r.attribution.mode == UNSUPPORTED
    assert "not about the controller" in r.attribution.detail


def test_the_report_is_json_serialisable():
    assert json.loads(json.dumps(_report().to_dict()))["interpretable"] is True


def test_nothing_in_the_module_mentions_the_benchmark_it_came_from():
    """Genericity enforced, not just asserted in a docstring. Benchmark-specific readings belong in
    the adapter; a core module naming one is how BFCL conditions leak into the core."""
    import pathlib

    import anchoropt.learning.delayed_effect as mod
    src = pathlib.Path(mod.__file__).read_text().lower()
    for word in ("bfcl", "gorilla", "kv_", "granite", "archival", "snapshot_store"):
        assert word not in src, f"core module references {word!r}"


# =================================================================================================
# the construction-identity contract the BFCL adapter must satisfy
#
# These are host-agnostic: they pin what `matched_initial_conditions` must conclude from the
# construction strings an adapter hands it. The strings below are the shape the BFCL adapter now
# produces (`store:<cache_dir>/<base_key>/v_<variant_key>`), and the reason the cache-dir component
# is load-bearing is measured, not assumed: the host's variant_key hashes storage-trigger TEMPLATE
# TEXT, which an installed controller does not change, so two arms that each built their own store
# get the SAME base_key AND the SAME variant_key. The path is the only component that separates
# them. A future simplification that drops it would silently launder the round-1 confound.
# =================================================================================================

def _cond(fp, proto):
    return {"initial_state_fingerprint": fp, "construction_protocol": proto}


def test_one_SHARED_store_read_by_both_arms_is_matched_construction():
    """The point of the corrected protocol: whoever ran the builder, one store is one construction."""
    shared = "store:/results/shared_cache/0b94ed89/v_f74e49da8dfe"
    m = matched_initial_conditions(_cond("0b94ed89/v_f74e49da8dfe", shared),
                                   _cond("0b94ed89/v_f74e49da8dfe", shared))
    assert m.initial_equivalent
    assert m.construction_matched
    assert m.ok, m.detail
    assert "downstream divergence is expected and permitted" in m.detail


def test_TWO_SELF_BUILT_stores_are_NOT_matched_even_with_identical_variant_keys():
    """Measured on the real host: base_key AND variant_key both agree across two arms that built
    separate stores, because neither hashes the controller. Only the store PATH differs, so the
    identity must keep it."""
    m = matched_initial_conditions(
        _cond("0b94ed89/built-here", "store:/results/arm_cache/0b94ed89/v_f74e49da8dfe"),
        _cond("0b94ed89/built-here", "store:/results/ctl_cache/0b94ed89/v_f74e49da8dfe"))
    assert not m.construction_matched
    assert not m.ok
    assert "construction protocol DIFFERS" in m.detail


def test_an_UNRESOLVED_construction_never_passes_for_a_matched_one():
    """When the store cannot be resolved the adapter must fail closed: two arms whose construction
    is merely unverifiable must not compare equal just because both are unverifiable."""
    m = matched_initial_conditions(
        _cond("0b94ed89/built-here", "UNRESOLVED:/results/shared/0b94ed89:fired_on_24"),
        _cond("0b94ed89/built-here", "UNRESOLVED:/results/shared/0b94ed89:fired_on_0"))
    assert not m.construction_matched


# =================================================================================================
# NO_OPPORTUNITY: the evaluation eliminated the acting phase while controlling for it
#
# An intervention that acts at a state transition can be silenced by the very comparison designed to
# isolate it: hold construction constant by SHARING the constructed state, and the construction phase
# no longer runs. The delta is then 0 because nothing happened, which is indistinguishable from a
# measured null unless the contract says so. These use the CI host: a controller that repairs fixtures
# during setup, evaluated in a run that reused the fixtures and only executed the scored tests.
# =================================================================================================

def test_a_declared_acting_phase_the_run_never_executed_is_NO_OPPORTUNITY():
    r = evaluate_delayed_effect(
        gained_units=[], firings_by_unit={}, phase_of=CI_PHASES, graph=CI_GRAPH,
        telemetry_by_phase={"setup": {}, "scored": {}},
        arm_conditions={"initial_state_fingerprint": "s0", "construction_protocol": "shared"},
        control_conditions={"initial_state_fingerprint": "s0", "construction_protocol": "shared"},
        declared_phase="setup", phases_executed=["scored"])
    assert r.attribution.mode == NO_OPPORTUNITY
    assert not r.had_opportunity
    assert not r.interpretable, "no opportunity to act cannot be an interpretable evaluation"
    # the correction under test DID work -- that must still be visible
    assert r.contamination_free, "construction was held constant; that is not what failed here"
    assert "never had the opportunity" in r.attribution.detail


def test_NO_OPPORTUNITY_is_not_claimed_when_the_host_reports_nothing():
    """Absence of a report is not evidence of absence of opportunity."""
    r = evaluate_delayed_effect(
        gained_units=[], firings_by_unit={}, phase_of=CI_PHASES, graph=CI_GRAPH,
        telemetry_by_phase={"setup": {}}, arm_conditions={}, control_conditions={},
        declared_phase="setup", phases_executed=None)
    assert r.attribution.mode != NO_OPPORTUNITY
    assert r.had_opportunity


def test_a_controller_that_DID_fire_is_never_NO_OPPORTUNITY():
    """It plainly had the opportunity, whatever the host reports about phases."""
    r = evaluate_delayed_effect(
        gained_units=["test:login"], firings_by_unit={"fixture:auth": True}, phase_of=CI_PHASES,
        graph=CI_GRAPH, telemetry_by_phase={"setup": {"interventions_executed": 1}},
        arm_conditions={"initial_state_fingerprint": "s0", "construction_protocol": "p"},
        control_conditions={"initial_state_fingerprint": "s0", "construction_protocol": "p"},
        declared_phase="setup", phases_executed=["scored"])
    assert r.attribution.mode == DELAYED
    assert r.had_opportunity


def test_a_phase_that_DID_execute_but_produced_no_firing_stays_DISCONNECTED_not_NO_OPPORTUNITY():
    """The distinction is load-bearing: this controller HAD its chance in setup and did nothing, which
    is a fact about the controller. NO_OPPORTUNITY would excuse it."""
    r = evaluate_delayed_effect(
        gained_units=["test:health"], firings_by_unit={}, phase_of=CI_PHASES, graph=CI_GRAPH,
        telemetry_by_phase={"setup": {}}, arm_conditions={}, control_conditions={},
        declared_phase="setup", phases_executed=["setup", "scored"])
    assert r.attribution.mode != NO_OPPORTUNITY
    assert r.had_opportunity
    assert not r.interpretable


def test_the_R2_shape_reproduces_on_a_NON_MEMORY_host():
    """R2 as measured, on the build host: shared warm cache means the warm phase never ran, so a
    compile-phase controller declaring phase=warm had no opportunity. Zero delta, no measurement."""
    r = evaluate_delayed_effect(
        gained_units=[], firings_by_unit={}, phase_of=BUILD_PHASES, graph=BUILD_GRAPH,
        telemetry_by_phase={"warm": {}, "build": {}},
        arm_conditions={"initial_state_fingerprint": "cache_v7",
                        "construction_protocol": "store:/shared/cache_v7"},
        control_conditions={"initial_state_fingerprint": "cache_v7",
                            "construction_protocol": "store:/shared/cache_v7"},
        declared_phase="warm", phases_executed=["build"])
    assert r.attribution.mode == NO_OPPORTUNITY
    assert r.contamination_free and r.conditions.ok
    assert not r.interpretable


# ---------------------------------------------------------------- the two flags are independent

def test_a_fingerprint_MISMATCH_does_not_silently_condemn_a_MATCHED_construction():
    """Measured on the real host (R2): the two arms named the SAME store as their construction and
    DIFFERENT initial fingerprints, because the loader named the variant it read and the builder
    named itself. An earlier version returned early on the fingerprint mismatch and reported
    `construction_matched=False` without ever comparing the protocols.

    That is not a stricter check, it is a WRONG one: `contamination_free` reads only
    `construction_matched`, so the round would report a construction confound that was never
    measured. Manufacturing an invalidity and hiding one are the same class of error -- in both the
    reported reason is not the measured reason.
    """
    shared = "store:/results/shared_cache/0b94ed89/v_f74e49da8dfe"
    m = matched_initial_conditions(_cond("0b94ed89/v_f74e49da8dfe", shared),
                                   _cond("0b94ed89/built-here", shared))
    assert not m.initial_equivalent          # the fingerprints genuinely differ, and that is reported
    assert m.construction_matched            # but construction was MEASURED, and it matched
    assert not m.ok                          # the pair is still not usable, for the real reason
    assert "DIFFERENT initial state" in m.detail
    assert "construction protocol DIFFERS" not in m.detail


def test_both_failures_are_reported_together_when_both_are_real():
    m = matched_initial_conditions(_cond("aaa/x", "store:/a/aaa/v_1"),
                                   _cond("bbb/y", "store:/b/bbb/v_2"))
    assert not m.initial_equivalent and not m.construction_matched
    assert "DIFFERENT initial state" in m.detail
    assert "construction protocol DIFFERS" in m.detail
