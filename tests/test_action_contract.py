"""Action contracts: an action LABEL is not an executable policy until eta_mu is grounded.

Each test pins something whose violation would let an incomplete action become an evaluation arm, or
let a historical answer key in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

import bfcl_runtime as R                                                # noqa: E402
from anchoropt.anchor import Action, IncisionPoint                       # noqa: E402
from anchoropt.learning.action_contract import (                         # noqa: E402
    CONTRACTS, InstantiatedAction, Operator, REPROMPT_INFEASIBLE, REROUTE_INFEASIBLE,
    SUPPRESS_INFEASIBLE, TRANSFORM_INFEASIBLE, instantiate, operators_of, validate,
)
from anchoropt.learning.anchor_policy_opt import (                       # noqa: E402
    AnchorPolicyOpt, FrozenIncumbent, SearchSpaceProposal,
)
from anchoropt.learning.policy_class import ThetaResult                  # noqa: E402

SIM = "retrieval_similarity_below_threshold"
POST = IncisionPoint.POST_EXECUTION
GATE = IncisionPoint.POST_GENERATION_PRE_EXEC


# ---- the paper's vocabulary maps onto the stored enum ------------------------------------------

def test_reroute_has_two_operator_readings():
    """The stored enum has four values, the paper five operators: REROUTE covers substitute AND
    transform, which have DIFFERENT contracts. The enum is not renamed -- frozen artifacts use it."""
    assert operators_of(Action.REROUTE) == (Operator.SUBSTITUTE, Operator.TRANSFORM)
    assert operators_of(Action.REPROMPT) == (Operator.REPROMPT,)
    assert operators_of(Action.SUPPRESS) == (Operator.SUPPRESS,)
    assert Operator.SUBSTITUTE.action is Action.REROUTE
    assert Operator.TRANSFORM.action is Action.REROUTE


def test_every_family_declares_its_required_action_parameters():
    for op, c in CONTRACTS.items():
        assert c.required, f"{op.value} declares no eta_mu"
        assert c.grounded_by.startswith("ground_")
        assert c.infeasible_code.endswith("_INFEASIBLE")


# ---- hard validation, independent of any prompt -----------------------------------------------

@pytest.mark.parametrize("op,eta,code", [
    (Operator.SUBSTITUTE, {"destination": "", "argument_mapping": {}, "retry_semantics": "x"},
     REROUTE_INFEASIBLE),
    (Operator.REPROMPT, {"instruction": "   ", "retry_budget": 1}, REPROMPT_INFEASIBLE),
    (Operator.SUPPRESS, {"suppressed_operation": "", "preservation": "p"}, SUPPRESS_INFEASIBLE),
    (Operator.TRANSFORM, {"target_surface": "", "operator": "o", "preservation": "p"},
     TRANSFORM_INFEASIBLE),
])
def test_an_incomplete_family_cannot_become_an_arm(op, eta, code):
    ok, why = validate(InstantiatedAction(operator=op, eta=eta))
    assert not ok and code in why


def test_missing_required_parameter_is_named_not_merely_counted():
    ok, why = validate(InstantiatedAction(operator=Operator.SUBSTITUTE,
                                          eta={"destination": "archival_memory_retrieve"}))
    assert not ok
    assert "argument_mapping" in why and "retry_semantics" in why


# ---- grounding enumerates, never chooses ------------------------------------------------------

def test_substitute_enumerates_every_credible_destination():
    """R1 evaluated one destination because the proposer named one action. Choosing one arbitrarily
    is what this contract exists to prevent."""
    built, failure = instantiate(Operator.SUBSTITUTE, signal=SIM, boundary=POST, runtime=R)
    assert failure is None
    dests = {a.eta["destination"] for a in built}
    assert len(dests) >= 2, f"only enumerated {dests}"
    for a in built:
        assert R.is_known_tool(a.eta["destination"])
        assert not R.unknown_args(a.eta["destination"], list(a.eta["argument_mapping"]))


def test_both_retry_semantics_are_separate_arms():
    built, _f = instantiate(Operator.SUBSTITUTE, signal=SIM, boundary=POST, runtime=R)
    sems = {a.eta["retry_semantics"] for a in built}
    assert sems == {"replace_original", "retry_after"}


def test_reprompt_grounds_several_distinct_instructions():
    """R2 measured ONE framing net-negative at every threshold. That rejects the framing, not the
    family -- and only measuring more than one can tell the difference."""
    built, failure = instantiate(Operator.REPROMPT, signal=SIM, boundary=POST, runtime=R)
    assert failure is None and len(built) >= 2
    texts = {a.eta["instruction"] for a in built}
    assert len(texts) == len(built), "instructions must be distinct to be worth separate arms"


def test_transform_is_distinct_from_substitute():
    """substitute changes WHICH OPERATION RUNS; transform changes the STATE a later decision reads."""
    sub, _ = instantiate(Operator.SUBSTITUTE, signal=SIM, boundary=POST, runtime=R)
    tra, _ = instantiate(Operator.TRANSFORM, signal=SIM, boundary=POST, runtime=R)
    assert tra, "a scored alternative read exists, so an exact merge is groundable"
    assert all("destination" in a.eta for a in sub)
    assert all("target_surface" in a.eta and "operator" in a.eta for a in tra)


def test_transform_needs_a_result_so_it_is_post_execution_only():
    built, failure = instantiate(Operator.TRANSFORM, signal=SIM, boundary=GATE, runtime=R)
    assert not built and failure is not None and failure.code == TRANSFORM_INFEASIBLE


def test_suppress_needs_a_proposed_operation_at_the_commitment_gate():
    none_here, failure = instantiate(Operator.SUPPRESS, signal=SIM, boundary=POST, runtime=R)
    assert not none_here and failure.code == SUPPRESS_INFEASIBLE
    built, f2 = instantiate(Operator.SUPPRESS, signal="clear_proposed_at_capacity",
                            boundary=GATE, runtime=R)
    assert f2 is None and built and built[0].eta["suppressed_operation"]


def test_ungroundable_family_is_infeasible_never_downgraded():
    """A silent conversion of an ungroundable REROUTE into a REPROMPT would measure a different
    controller under the first one's name."""
    built, failure = instantiate(Operator.SUBSTITUTE, signal="no_tool_call_at_all",
                                 boundary=GATE, runtime=R)
    assert not built and failure is not None
    assert failure.code == REROUTE_INFEASIBLE
    assert failure.missing, "the missing requirement must be named"


# ---- eta_mu and theta_phi stay separate -------------------------------------------------------

def _proposal(**over):
    base = dict(boundary=POST, signal=SIM, action_set=(Action.REPROMPT, Action.REROUTE),
                rationale="weak retrieval then answered", diagnosis_case_ids=("c1",),
                preferred_action="reprompt", theta_hint={"below": 0.75})
    base.update(over)
    return SearchSpaceProposal(**base)


def test_arms_carry_eta_and_the_signal_grid_carries_theta():
    """One generic `theta` dict obscured which parameters belong to the signal versus the action --
    and that is how R2 swept theta while eta stayed frozen at the proposer's guess."""
    opt = AnchorPolicyOpt(runtime=R, host=R.HOST)
    arms, _rej = opt.build_arms(_proposal())
    assert arms
    for a in arms:
        assert "below" not in a.eta, "theta_phi must not be inside eta_mu"
        assert a.signal_domains and a.signal_domains[0].name == "below"


def test_the_full_family_is_INSTANTIATED_even_where_it_is_not_materializable():
    """Two different questions, and conflating them inflated an arm count by 91.

    `instantiate` completes every family the contract can ground -- that is the search space. The
    EXECUTOR rule then decides which of those the host can actually run. Both numbers matter: the
    first says what the proposer's suggestion covers, the second says what can be measured."""
    from anchoropt.learning.action_contract import instantiate

    for op in (Operator.REPROMPT, Operator.SUBSTITUTE, Operator.TRANSFORM):
        built, failure = instantiate(op, signal=SIM, boundary=POST, runtime=R)
        assert built and failure is None, f"{op.value} should be groundable by contract"

    opt = AnchorPolicyOpt(runtime=R, host=R.HOST)
    arms, rejected = opt.build_arms(_proposal())
    ops = {a.instantiated.operator for a in arms}
    # The executor registry was completed from the evaluator's real firing sites, so this host
    # DOES run reroute at post_execution under both operator readings (substitute via the
    # capacity-repair remedy, transform via the archival-evict remedy). Before that, only
    # reprompt was declared and the undeclared cells were pruned before evaluation -- which
    # would have made an anchor-recovery study report a low rate for a REGISTRY reason rather
    # than a search one. The point of the test is unchanged: instantiation and
    # materializability are different questions.
    assert Operator.REPROMPT in ops
    assert ops <= {Operator.REPROMPT, Operator.SUBSTITUTE, Operator.TRANSFORM}, \
        f"unexpected operator materialized at post_execution: {ops}"
    # 2 arms: the honourable reprompt etas. It was 9 while the reroute cell claimed
    # `eta_is_computed=True, consumes=()` -- a positive claim that its executor derives everything
    # from live state. The LIVE hook opens with
    #     if _prim != "additional_read_and_merge": return {"fired": False, ...}
    # so those 7 arms carried no `primitive`, would each have reported not-fired, and would have run
    # as the CONTROL: a measured zero against an intervention that never ran. `requires_eta` now
    # refuses them at construction with that reason, which is the honest state -- EXECUTOR_UNAVAILABLE,
    # not a negative result about the capacity family.
    #
    # One reprompt eta is ALSO still rejected, on the executor's fixed retry_budget, so the rule is
    # doing work in both directions rather than waving everything through.
    assert any(r.reason_code == "not_materializable_by_host_executor" for r in rejected)
    # THE REFUSAL MUST NAME WHAT IS MISSING, so the record says EXECUTOR_UNAVAILABLE rather than
    # leaving the arm to be measured as a null. Two admissible reasons, both specific:
    #
    #   `primitive`                 the executor demands an eta key this arm does not carry
    #   `implements operator(s)`    the executor implements a DIFFERENT operator reading of this action
    #
    # The second was added when the capacity executors began declaring `operators`. It fires EARLIER
    # than the eta check -- an executor that does not implement this reading is not a candidate at all,
    # so asking whether its eta matches would be answering the wrong question. Either reason is a
    # named refusal; what would be wrong is a bare "no executor".
    reroute_refusals = [r for r in rejected
                       if r.reason_code == "not_materializable_by_host_executor"
                       and ("primitive" in (r.detail or "")
                            or "implements operator(s)" in (r.detail or ""))]
    assert reroute_refusals, (
        "the reroute arms must be refused NAMING what is missing -- the demanded eta key, or the "
        "operator reading the executor does not implement -- so the record says EXECUTOR_UNAVAILABLE "
        "rather than leaving them to be measured as a null")
    assert len(arms) == 2, f"2 honourable reprompt etas, got {len(arms)}: {[a.signal for a in arms]}"


def test_preference_and_hint_do_not_change_the_arms():
    opt = AnchorPolicyOpt(runtime=R, host=R.HOST)
    a, _ = opt.build_arms(_proposal(preferred_action="reprompt", theta_hint={"below": 0.75}))
    b, _ = opt.build_arms(_proposal(preferred_action="reroute", theta_hint={"below": 0.1}))
    assert {x.label for x in a} == {x.label for x in b}


def test_measurement_can_overrule_the_preferred_eta():
    """Measurement, not the proposer, picks among the materializable arms. The preferred ACTION here
    is the only one with an executor, so the choice is between its eta realizations -- which is the
    honest form of this test on this host."""
    obs = {"best_similarity": [0.1, 0.2, 0.3, 0.4]}

    def evaluate(arm, theta):
        net = 7 if arm.instantiated.variant == "search_other_container" else -2
        return ThetaResult(theta=theta, gains=tuple(f"g{i}" for i in range(max(net, 0))),
                           losses=tuple(f"l{i}" for i in range(max(-net, 0))),
                           firings=5, cases_fired=5, n=89, interventions_executed=5)

    opt = AnchorPolicyOpt(runtime=R, host=R.HOST)
    out = opt.optimize(_proposal(), incumbent=FrozenIncumbent("P0", "tok"),
                       evaluate=evaluate, observations=obs)
    assert out.winner.instantiated.variant == "search_other_container"


def test_incumbent_is_frozen_for_the_whole_round():
    """If the incumbent moved between arms the arms would not be comparable and argmax J is
    meaningless."""
    inc = FrozenIncumbent("P0", "tok-A")
    with pytest.raises(RuntimeError, match="incumbent drifted"):
        inc.assert_same("tok-B")


# ---- A9 isolation ------------------------------------------------------------------------------

def test_no_accepted_anchor_identifier_or_measured_delta_in_the_runtime_source():
    """Widened from an A9-only check after the recovery-target audit found more than expected.

    The learner-visible runtime must name NO accepted anchor and quote NO measured delta. Comments
    do not reach the proposer -- it reads all_fields()/tool_schema() output -- but a comment that
    records the answer is how a fixture becomes privileged when someone later restores a default
    from a docstring. `bfcl_runtime.py` had stated one anchor's -4.95/+3.63 boundary result outright,
    which is the answer to the very question a boundary-recovery experiment tests."""
    import re

    for mod in ("bfcl_runtime.py", "bfcl_signals.py", "bfcl_capabilities.py"):
        src = (REPO / "benchmarks" / "bfcl_v4" / mod).read_text()
        hits = re.findall(r"\bA[0-9]\b", src)
        assert not hits, f"{mod} names accepted anchor(s): {sorted(set(hits))}"
        for token in ("on_low_similarity_cross_container", "ANCHOROPT_XCM", "xcm",
                      "core_max_below_threshold", "on_premature_idk", "4.95", "3.63"):
            assert token not in src, f"{mod} leaks an anchor identifier or delta: {token!r}"


def test_no_measured_threshold_or_alias_is_shipped():
    import bfcl_signals as sig
    assert dict(sig.SIGNAL_PROBE_PARAMS) == {}
    assert dict(sig.SIGNAL_ALIASES) == {}
    assert dict(R.REROUTE_DESTINATIONS) == {}
    dom = R.parameter_domains(SIM)[0]
    assert dom.values == () and (dom.low, dom.high) == (0.0, 1.0)


def test_grounded_arms_contain_no_a9_configuration():
    opt = AnchorPolicyOpt(runtime=R, host=R.HOST)
    arms, _rej = opt.build_arms(_proposal())
    blob = repr([dict(a.eta) for a in arms]).lower()
    for token in ("0.30", "xcm", "cross_container", "tie_break"):
        assert token not in blob, f"A9 configuration leaked: {token!r}"


def test_a_categorical_signal_parameter_yields_arms_not_an_empty_grid():
    """The dry-run bug: `no_informative_result`'s parameter is categorical (`kind`), and computing
    its grid with `quantile_grid` returns () -- silently zeroing 10 otherwise-valid arms. Callers
    must go through PolicySpec.candidate_thetas(), which dispatches on the domain kind."""
    from anchoropt.learning.policy_class import PolicySpec, PolicyClass, quantile_grid

    dom = R.parameter_domains("no_informative_result")[0]
    assert dom.values and not dom.observed_from, "this domain is categorical"
    assert quantile_grid([0.1, 0.2, 0.3], domain=dom) == (), "the trap being pinned"

    spec = PolicySpec(boundary=POST, signal="no_informative_result", action=Action.REPROMPT,
                      policy_class=PolicyClass.PARAMETERIZED, domains=(dom,))
    grid = spec.candidate_thetas({"best_similarity": [0.1, 0.2]})
    assert len(grid) == 2
    assert {g["kind"] for g in grid} == {"empty_collection", "all_floor_scores"}


# ---- executor-backed feasibility ---------------------------------------------------------------

def test_abstractly_legal_but_non_materializable_policy_is_rejected_before_evaluation():
    """The gap the targeted recovery experiment exposed. Three REPROMPT arms passed every abstract
    check -- U_H(l) admissible, contract satisfied, hard validation clean -- and one declared a retry
    budget the host executor cannot honour. Feasibility must come from an EXECUTOR, not from the
    action declaration alone, and the rejection must happen before an arm is measured.

    MEASURED AT `post_execution`, NOT AT THE COMMITMENT GATE. This test used to use the gate, where
    the upstream reprompt cell is now DISABLED -- its instruction is written to the step record and
    never injected, so every arm there is the control arm under a reprompt label (D3). The property
    under test is the retry-budget conflict, which is unchanged and lives at the post-execution cell
    that genuinely injects and regenerates.
    """
    from anchoropt.learning.anchor_policy_opt import REJECT_NO_EXECUTOR

    prop = SearchSpaceProposal(
        boundary=POST, signal="retrieval_similarity_below_threshold",
        action_set=(Action.REPROMPT,), diagnosis_case_ids=("c1",))
    opt = AnchorPolicyOpt(runtime=R, host=R.HOST)
    arms, rejected = opt.build_arms(prop)

    labels = {a.instantiated.variant for a in arms}
    assert "verify_before_answering" in labels
    assert "state_absence_if_unfound" not in labels, \
        "an eta the executor cannot honour must not become an arm"
    codes = {r.reason_code for r in rejected}
    assert REJECT_NO_EXECUTOR in codes, rejected
    conflict = [r for r in rejected if r.reason_code == REJECT_NO_EXECUTOR]
    assert any("retry_budget" in r.detail for r in conflict), conflict


def test_the_upstream_reprompt_cell_is_DISABLED_because_its_instruction_never_reaches_the_agent():
    """D3. An action whose parameter does not reach the model must not be measurable.

    The upstream hook assigns the instruction to the step record and nothing else: it appends no
    message and does not regenerate. Two arms differing ONLY in `instruction` produced byte-identical
    trajectories across 13/13 storage episodes -- the strongest available evidence that the field is
    inert, since a decode that saw different text would differ.

    Measuring such a cell would report the CONTROL arm under a reprompt label, which is the
    silent-null class. So the capability is declared present-but-disabled, and core refuses it with a
    reason rather than pruning it silently (a silently absent cell reads as "nobody thought of it").
    """
    from anchoropt.learning.anchor_policy_opt import REJECT_NO_EXECUTOR

    prop = SearchSpaceProposal(
        boundary=GATE, signal="no_tool_call_at_all", action_set=(Action.REPROMPT,),
        diagnosis_case_ids=("c1",))
    arms, rejected = AnchorPolicyOpt(runtime=R, host=R.HOST).build_arms(prop)

    # THE CELL IS NO LONGER UNIFORMLY REFUSED, and that is the correction. It holds two executors:
    # an inject-and-regenerate one that DOES reach the model, and this inert one. What must hold is
    # that no arm is ever attributed to the INERT executor -- refusing the whole cell also pruned the
    # working mechanism, which is a validity claim the evidence does not support.
    for arm in arms:
        assert arm.capability_id != "write_instruction_to_step_record", (
            "an arm was built against the executor proven not to reach the model")
    # Whatever is built must name the executor core actually validated, so a runner cannot install a
    # different sibling than the one feasibility accepted.
    assert all(arm.capability_id for arm in arms), (
        f"every arm at a multi-executor cell must name its capability: {arms}")
    # And the inert executor must still be REFUSED on its own, with its reason recorded rather than
    # silently pruned -- a silently absent cell reads as "nobody thought of it".
    from anchoropt.learning import executor_capability as _ec
    inert = [c for c in R.executor_capabilities(GATE, Action.REPROMPT)
             if c.cid == "write_instruction_to_step_record"]
    assert inert, "the inert executor must stay declared"
    cap, why = _ec.resolve(inert, boundary=GATE, action=Action.REPROMPT,
                           signal="no_tool_call_at_all",
                           eta={"instruction": "x", "retry_budget": 1},
                           required_eta=("instruction", "retry_budget"))
    assert cap is None, "the inert executor must never be resolved as feasible"
    assert "executor_disabled_by_host" in why, why
    assert any("never injected" in r.detail for r in rejected), \
        "the rejection must say WHY, so a porter can tell this from an absent executor"


def test_an_admissible_action_is_rejected_when_its_executor_does_not_COVER_the_signal():
    """SUPPRESS is admissible at the commitment gate and this host now HAS a suppress executor --
    but it observes duplicate/clear conditions, not a zero-tool-call one.

    Before the registry was completed this rejection read `no_executor`. It now reads
    `executor_signal_unsupported`, which is strictly more informative: "the host cannot do this at
    all" and "the host does this, for other conditions" are different facts, and only the second
    tells a proposer the cell is worth trying with a different signal.
    """
    ok, why = R.executor_supports(GATE, Action.SUPPRESS, "no_tool_call_at_all", {})
    assert not ok
    assert "executor_signal_unsupported" in why
    assert Action.SUPPRESS in R.HOST.executable_actions(GATE), "it IS admissible in U_H(l)"
    ok2, _ = R.executor_supports(GATE, Action.SUPPRESS, "duplicate_identifier", {})
    assert ok2, "a signal the executor DOES cover must be accepted at the same cell"


def test_a_cell_with_NO_executor_at_all_still_reports_no_executor():
    """The `no_executor` branch must stay reachable -- PRE_GENERATION declares NOOP only."""
    from anchoropt.anchor import IncisionPoint
    ok, why = R.executor_supports(IncisionPoint.PRE_GENERATION, Action.REPROMPT,
                                  "no_tool_call_at_all", {})
    assert not ok and "no_executor" in why


def test_the_executor_is_generic_over_signals_not_anchor_specific():
    """The post-generation reprompt window is a boundary/action executor: the same window and
    injection mechanism serve more than one signal in this evaluator."""
    gate_exec = R.executor_for(GATE, Action.REPROMPT)
    post_exec = R.executor_for(POST, Action.REPROMPT)
    assert gate_exec and post_exec
    assert gate_exec["remedy_flag"] != post_exec["remedy_flag"]
    for spec in (gate_exec, post_exec):
        assert "eta_slot" in spec and "trigger" in spec and "fixed" in spec
    # no accepted-anchor identifier anywhere in the registry
    import re
    assert not re.findall(r"\bA[0-9]\b", repr(R.EXECUTORS))


def test_no_answer_key_terminology_in_the_executor_registry():
    """The EXECUTORS registry must describe mechanisms, never name accepted controllers.

    Executors are registered by reading the evaluator's real firing sites, and the natural way to
    document one is "this is anchor X's gate" -- which puts the answer key in the learner-visible
    runtime and turns a recovery experiment into a lookup. The module-wide leak test catches anchor
    identifiers and specific gate keys; this one pins the registry itself, including operator variants
    and every nested field, because a new cell is exactly where such a term gets introduced.
    """
    import json
    import re

    blob = json.dumps({f"{b}/{a}": v for (b, a), v in R.EXECUTORS.items()}, default=str).lower()
    # accepted-controller identifiers and the private gate/env names that identify one
    for token in ("xcm", "cross_container", "core_max_below_threshold", "on_low_similarity_cross",
                  "on_premature_idk", "anchoropt_xcm", "a9", "a1b", "dcr"):
        assert token not in blob, f"the executor registry leaks an answer-key term: {token!r}"
    assert not re.findall(r"\ba[0-9]\b", blob), \
        f"the executor registry names an accepted anchor: {sorted(set(re.findall(r'.a[0-9].', blob)))}"


def test_the_read_merge_primitive_is_declared_and_signal_agnostic():
    """The design principle the A9 round exposed, pinned as a test.

    Signal expansion is useless unless the executor substrate holds a COMPATIBLE PRIMITIVE. Synthesis
    produced a predicate about a store never being consulted while every executor at that boundary was
    a write-side repair, so a controller on that predicate could be BUILT and could never ENGAGE -- the
    gap closed in Phi and reopened one step later. A read-side primitive must therefore exist, be
    reachable, and accept signals it was not shipped knowing about.
    """
    # The primitive is the TRANSFORM operator's merge grounding, not a separate operator: a
    # runtime-only operator name can never become an arm, because action_contract.instantiate
    # enumerates operators from the Operator enum.
    variants = R.ground_transforms("retrieval_similarity_below_threshold", "post_execution")
    assert variants, "no transform grounding at post_execution"
    merge = [v for v in variants if "merge_additional_read" in str(v["eta"].get("operator", ""))]
    assert merge, f"no read-side MERGE primitive; got {[v['variant'] for v in variants]}"
    eta = merge[0]["eta"]
    assert eta.get("target_surface") == "returned_observation", \
        "a read-side merge must rewrite the RETURNED OBSERVATION, not a later surface"
    assert "retained" in str(eta.get("preservation", "")).lower(), \
        "the original result must be preserved -- a merge that replaces it is a different action"
    # and the cell must accept a signal it was not shipped knowing about
    cell = R.EXECUTORS[("post_execution", "reroute")]
    assert cell.get("signal_agnostic") is True, \
        "an EXPANDED signal must be able to drive this cell; its mechanism reads live state"
    # and the high-level action space is UNCHANGED -- this is an operator, not a new Action
    from anchoropt.anchor import Action
    assert {a.value for a in Action} == {"noop", "reprompt", "suppress", "reroute"}


def test_an_EXPANDED_signal_is_NOT_exempt_from_parameter_validation():
    """The population that most needs the fixed-parameter check was exempt from it.

    `executor_supports` returned True early for an expanded signal on a signal-agnostic cell, before
    reaching the fixed-parameter loop. But a SYNTHESIZED predicate is exactly the case that carries a
    data-derived threshold, and if the executor imposes its own the arm measures a trigger nobody
    proposed -- a silent coercion, which is what this contract exists to refuse. Measured: three
    synthesized thresholds (0.1710 / 0.1816 / 0.2940) all sat below an executor-fixed 0.30, so every one
    would have fired on a strict superset of its own condition.
    """
    from anchoropt.anchor import Action, IncisionPoint

    R.reset_expanded_signals()
    try:
        sig = "probe_expanded__and__probe_threshold_lt_0p2940"
        R.install_signal(sig, lambda s: True, boundary=IncisionPoint.POST_EXECUTION,
                         provenance="test")
        fixed = R.EXECUTORS[("post_execution", "reroute")].get("fixed") or {}
        assert fixed, "the cell must DECLARE the parameters it imposes, or nothing can be compared"
        key, value = next(iter(fixed.items()))

        ok, why = R.executor_supports(IncisionPoint.POST_EXECUTION, Action.REROUTE, sig,
                                      {key: value - 0.01})
        assert not ok, "a conflicting parameter on an expanded signal must be REJECTED"
        assert "executor_parameter_conflict" in why

        ok2, _ = R.executor_supports(IncisionPoint.POST_EXECUTION, Action.REROUTE, sig, {key: value})
        assert ok2, "the executor's own value must still be accepted"
        ok3, _ = R.executor_supports(IncisionPoint.POST_EXECUTION, Action.REROUTE, sig, {})
        assert ok3, "an eta that does not mention the parameter is not a conflict"
    finally:
        R.reset_expanded_signals()
