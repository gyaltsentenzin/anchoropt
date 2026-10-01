"""Functional tests for the observational telemetry, including the paths that mislead."""

import json
import pathlib

from anchoropt.telemetry import (
    FUNNEL_STAGES, ArmCost, CandidateEta, ResidualEntry, RoundLogger, RoundRecord, SearchFunnel,
    arm_cost, classify_rejection, cost_deltas, locality_report, residual_shape,
)
from anchoropt.telemetry.runtime_cost import episode_cost

FIRE = "zero_call_reprompt_gate"


def step(i, *, decoded=(), fired=False, status="executed"):
    s = {"turn": 0, "step": i, "status": status, "decoded": list(decoded)}
    if fired:
        s[FIRE] = True
    return s


# ---------------------------------------------------------------- funnel
def test_classify_rejection_maps_the_codes_actually_emitted():
    assert classify_rejection("signal_not_observable_at_boundary") == "signal_not_observable"
    assert classify_rejection("not_executable_in_host") == "action_not_admissible"
    assert classify_rejection("REPROMPT_INFEASIBLE: no content") == "eta_incompatible"
    assert classify_rejection("no_parameter_grid") == "evaluation_failure"


def test_no_executor_and_eta_conflict_are_DIFFERENT_findings():
    """Both arrive as `not_materializable_by_host_executor`; the detail separates them.

    Merging them would hide whether the host lacks the capability or merely rejects the parameters --
    the distinction that produced the retracted 'host-capability gap'.
    """
    code = "not_materializable_by_host_executor"
    assert classify_rejection(code, "no_executor: this host has no executor for x/y") == "no_executor"
    assert classify_rejection(
        code, "executor_parameter_conflict: fixes retry_budget=1") == "eta_incompatible"
    assert classify_rejection(
        code, "executor_signal_unsupported: evaluates [...]") == "signal_not_observable"


def test_unknown_code_is_flagged_not_silently_bucketed():
    f = SearchFunnel()
    assert f.record_rejection("some_new_code_upstream") == "unclassified"
    assert f.report().unclassified == ("some_new_code_upstream",)
    assert sum(f.report().rejections.values()) == 0


def test_observe_build_counts_admissible_as_executable_plus_downstream_rejections():
    class RJ:
        def __init__(self, code, detail=""):
            self.reason_code, self.detail = code, detail
    arms = [object(), object()]
    rejected = [
        RJ("not_materializable_by_host_executor", "no_executor: none here"),   # admissible
        RJ("not_materializable_by_host_executor",
           "executor_parameter_conflict: retry_budget"),                        # admissible
        RJ("not_executable_in_host", "host cannot"),                            # NOT admissible
    ]
    f = SearchFunnel()
    f.observe_build(arms, rejected, proposed=10, localized=6)
    r = f.report()
    assert r.stages["executable"] == 2
    assert r.stages["admissible"] == 4, "2 executable + 2 rejected downstream of U_H(l)"
    assert r.stages["proposed"] == 10 and r.stages["localized"] == 6
    assert r.rejections["no_executor"] == 1
    assert r.rejections["eta_incompatible"] == 1
    assert r.rejections["action_not_admissible"] == 1


def test_shrink_may_exceed_one_because_the_UNITS_change():
    """proposed/localized count FAMILIES; admissible onward count grounded arms.

    One family becomes several arms, so localized->admissible is expansion. Reporting it as
    "shrink" without saying so invites the reading that localization made the space bigger.
    """
    f = SearchFunnel()
    f.record_stage("localized", 3)
    f.record_stage("admissible", 21)
    f.record_stage("executable", 4)
    r = f.report()
    assert r.shrink("localized", "admissible") > 1.0
    assert r.pruning_ratio == 4 / 21
    d = r.as_dict()
    assert d["pruning_ratio_admissible_to_executable"] == 4 / 21
    assert "expansion" in d["units_note"]


def test_shrink_reports_none_rather_than_dividing_by_zero():
    f = SearchFunnel()
    assert f.report().shrink("proposed", "localized") is None
    assert set(f.report().as_dict()["stages"]) == set(FUNNEL_STAGES)


# ---------------------------------------------------------------- cost
def test_episode_cost_counts_reads_interventions_and_steps():
    e = episode_cost("c1", [step(0, fired=True),
                            step(1, decoded=["archival_memory_retrieve(q)"]),
                            step(2, decoded=["core_memory_add(x)"], status="answer_end_turn")])
    assert (e.steps, e.llm_calls, e.tool_calls, e.retrieval_calls) == (3, 3, 2, 1)
    assert e.interventions == 1 and e.fired


def test_missing_tokens_are_NONE_not_zero():
    """A missing measurement and a measured zero are different facts."""
    e = episode_cost("c1", [step(0)])
    assert e.input_tokens is None and e.generated_tokens is None and e.total_tokens is None
    a = ArmCost(arm="a", episodes=(e,))
    assert a.tokens_available is False
    assert a.cost_per_successful_episode()["total_tokens"] is None


def test_overhead_split_by_fired_vs_untouched():
    ctl = arm_cost("ctl", {"c1": [step(0, status="answer_end_turn")],
                           "c2": [step(0, decoded=["r()"]), step(1)]})
    arm = arm_cost("a", {"c1": [step(0, fired=True), step(1, decoded=["r()"]), step(2)],
                         "c2": [step(0, decoded=["r()"]), step(1)]})
    fired = arm.overhead_on_fired(ctl)
    untouched = arm.overhead_on_untouched(ctl)
    assert fired["n"] == 1 and fired["steps"] == 2.0
    assert untouched["n"] == 1 and untouched["steps"] == 0.0, "an untouched episode must cost the same"
    assert fired["total_tokens"] is None, "tokens not instrumented -> None, not 0"


def test_cost_per_corrected_failure_uses_CONVERSIONS_as_the_denominator():
    ctl = arm_cost("ctl", {"c1": [step(0)], "c2": [step(0)]},
                   correct_by_case={"c1": False, "c2": True})
    arm = arm_cost("a", {"c1": [step(0, fired=True), step(1)], "c2": [step(0)]},
                   correct_by_case={"c1": True, "c2": True})
    r = arm.cost_per_corrected_failure(ctl)
    assert r["n_converted"] == 1, "c2 was already correct; the arm inherits it for free"
    assert r["steps"] == 1.0


def test_cost_deltas_reports_whether_tokens_were_measured():
    ctl = arm_cost("ctl", {"c1": [step(0)]})
    arm = arm_cost("a", {"c1": [step(0, fired=True)]})
    assert cost_deltas(arm, ctl)["tokens_measured"] is False


# ---------------------------------------------------------------- locality
def test_locality_separates_fired_from_non_fired_changes():
    """A change on an episode the controller never touched cannot be the intervention."""
    ctl_steps = {"t1": [step(0, status="answer_end_turn")],          # target: zero calls
                 "t2": [step(0, status="answer_end_turn")],          # target
                 "o1": [step(0, decoded=["r()"], status="answer_end_turn")]}  # not a target
    arm_steps = {"t1": [step(0, fired=True), step(1, decoded=["r()"])],
                 "t2": [step(0, status="answer_end_turn")],          # missed opportunity
                 "o1": [step(0, decoded=["r()"], status="answer_end_turn")]}
    rep = locality_report("a",
                          ctl_steps, arm_steps,
                          {"t1": False, "t2": False, "o1": False},
                          {"t1": True, "t2": False, "o1": True})
    assert rep.n_target == 2 and rep.n_fired == 1
    assert rep.gains_fired == 1
    assert rep.changed_not_fired == 1 and rep.gains_not_fired == 1, "o1 flipped with no firing"
    assert rep.opportunity_missed == 1
    assert rep.target_coverage == 0.5
    d = rep.as_dict()
    assert "cannot be the intervention" in d["non_fired_episodes"]["note"]


def test_off_target_firing_is_reported():
    ctl_steps = {"o1": [step(0, decoded=["r()"], status="answer_end_turn")]}
    arm_steps = {"o1": [step(0, fired=True), step(1, decoded=["r()"])]}
    rep = locality_report("a", ctl_steps, arm_steps, {"o1": False}, {"o1": True})
    assert rep.opportunity_created == 1
    assert rep.off_target_firing_rate == 1.0
    assert rep.n_fired_on_target == 0


# ---------------------------------------------------------------- round record
def test_residual_shape_is_the_comparable_distribution():
    rs = (ResidualEntry(1, "a", 4), ResidualEntry(2, "b", 3), ResidualEntry(3, "c", 1))
    assert residual_shape(rs) == "4/3/1"


def test_round_record_roundtrips_and_reports_the_residual_shift(tmp_path):
    rec = RoundRecord(
        round_id="R3", incumbent_id="native",
        residuals_before=(ResidualEntry(1, "k1", 4), ResidualEntry(2, "k2", 3),
                          ResidualEntry(3, "k3", 1)),
        selected_residual="k1", selected_locus="post_generation_pre_exec",
        selected_signal="no_tool_call_at_all", feasible_actions=("noop", "reprompt", "suppress"),
        candidate_etas=(CandidateEta("v1", {"retry_budget": 1}, True, evaluated=True),
                        CandidateEta("v3", {"retry_budget": 0}, False,
                                     rejection_reason="eta_incompatible")),
        winner="v1", objective_delta=16.7,
        residuals_after=(ResidualEntry(1, "k2", 3), ResidualEntry(2, "k4", 2)))
    log = RoundLogger(tmp_path / "rounds.jsonl")
    log.append(rec)
    rows = log.load()
    assert len(rows) == 1
    assert rows[0]["residuals_before_shape"] == "4/3/1"
    assert rows[0]["residuals_after_shape"] == "3/2"
    assert rows[0]["cost_delta"] is None, "unmeasured cost stays absent, never 0"
    shift = rec.residual_shift()
    assert shift["resolved"] == ["k1", "k3"] and shift["introduced"] == ["k4"]
    assert shift["persisting"] == ["k2"]


def test_logger_is_append_only(tmp_path):
    log = RoundLogger(tmp_path / "r.jsonl")
    log.append(RoundRecord(round_id="R1", incumbent_id="i"))
    log.append(RoundRecord(round_id="R2", incumbent_id="i"))
    assert [r["round_id"] for r in log.load()] == ["R1", "R2"]
    assert log.latest()["round_id"] == "R2"
