"""The CCTU adapter: the classifiers, U_H(l), the boundary expansion, and the label rule.

The claims under test are the ones this integration rests on, and each is here because getting it
wrong is SILENT rather than loud:

  * the violation vocabulary is CLOSED and matches upstream's own handlers. A class the classifier
    misses simply vanishes from the residual.
  * a trajectory yields TWO events per turn. One event per turn derives a single boundary, and a
    residual with one boundary can never be observed to move earlier -- the template calls this the
    most valuable lesson from the first port.
  * the SCORING LABELS are unreachable from observable state. A discovery made with the answer key in
    the room is a lookup, however unused it looks.
  * `inf` never reaches a declared numeric field, because `signal_grammar` would turn it into a
    threshold that fires on everything.
  * asking "would this turn violate anything" does not charge the budget it is asking about.

Also asserted: no CCTU vocabulary in `anchoropt/`, and importing this adapter does not seize the
single global adapter slot from BFCL.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CCTU = REPO / "benchmarks" / "cctu"
for _p in (str(REPO), str(CCTU)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cctu_adapter as cctu  # noqa: E402
import cctu_capabilities as caps  # noqa: E402
import cctu_signals as sig  # noqa: E402
import cctu_state as state_mod  # noqa: E402
from anchoropt.anchor import Action, IncisionPoint  # noqa: E402
from anchoropt.runtime import ActionNotExecutable  # noqa: E402

DATA = CCTU / "data"
_HAS_CORPUS = (DATA / "input_data.jsonl").exists()
needs_corpus = pytest.mark.skipif(not _HAS_CORPUS, reason="corpus not present in this checkout")


def _turn(content="x", calls=(), feedback=(), boundary="post_generation_pre_exec"):
    return cctu.normalize_event({
        "boundary": boundary, "has_generation": True,
        "message": {"content": content, "tool_calls": list(calls)},
        "feedback": list(feedback)})


def _call(cid="0", name="t", args="{}"):
    return {"id": cid, "function": {"name": name, "arguments": args}}


@pytest.fixture(autouse=True)
def _isolate_module_globals():
    """`HANDLER_ERRORS` and the expanded-signal tables are module-global and accumulate.

    Without this, one test's recorded handler failure is another test's assertion, and an expanded
    signal outlives the test that installed it. Both are deliberately global in the runtime -- a
    session's Phi and a session's predictive failures are session facts -- so isolation belongs here.
    """
    state_mod.HANDLER_ERRORS.clear()
    sig.UNDECLARED_CLASSES_SEEN.clear()
    cctu.reset_expanded_signals()
    yield
    state_mod.HANDLER_ERRORS.clear()
    sig.UNDECLARED_CLASSES_SEEN.clear()
    cctu.reset_expanded_signals()


# ================================================================================================
# 1. the violation vocabulary is closed, and it is UPSTREAM'S
# ================================================================================================
def test_every_violation_message_upstream_emits_is_classified_and_declared():
    """The residual's whole vocabulary, checked against the handlers that produce it.

    A class the classifier cannot name does not become an unknown -- it vanishes, and the residual
    silently shrinks. So this walks upstream's own source for every
    `INSTRUCTION FOLLOWING ERROR: <CLASS> NOT FOLLOWED!` literal and requires that each one
    canonicalizes to a DECLARED class.
    """
    literal = re.compile(r"INSTRUCTION FOLLOWING ERROR:\s*([A-Z0-9 _]+?)\s*NOT FOLLOWED!")
    found = set()
    for path in (CCTU / "utils" / "constraint_checker").rglob("*.py"):
        found |= set(literal.findall(path.read_text()))
    assert found, "no violation templates found -- this test is looking in the wrong place"

    classified = {sig.violation_class_of(f"INSTRUCTION FOLLOWING ERROR: {raw} NOT FOLLOWED!")
                  for raw in found}
    assert None not in classified
    undeclared = classified - set(caps.VIOLATION_CLASSES)
    assert not undeclared, (
        f"upstream emits violation class(es) {sorted(undeclared)} that cctu_capabilities does not "
        f"declare. A class the residual cannot name is a class the residual loses.")
    assert not sig.UNDECLARED_CLASSES_SEEN, sorted(sig.UNDECLARED_CLASSES_SEEN)


def test_no_declared_class_is_unreachable():
    """A declared class upstream never emits is dead vocabulary, and dead vocabulary hides drift."""
    literal = re.compile(r"INSTRUCTION FOLLOWING ERROR:\s*([A-Z0-9 _]+?)\s*NOT FOLLOWED!")
    emitted = set()
    for path in (CCTU / "utils" / "constraint_checker").rglob("*.py"):
        for raw in literal.findall(path.read_text()):
            emitted.add(sig.violation_class_of(f"INSTRUCTION FOLLOWING ERROR: {raw} NOT FOLLOWED!"))
    assert set(caps.VIOLATION_CLASSES) == emitted


def test_the_classifier_agrees_with_the_regex_that_decides_SR_and_PSR():
    """One owner plus a parity test, the precedent `benchmarks/bfcl_v4/adapter.py` set.

    `evaluation.py`'s regex is what SR and PSR are computed from. If this classifier disagreed, a
    signal would fire on a different population than the metric scores.
    """
    src = (CCTU / "evaluation.py").read_text()
    pattern = re.search(r'_ERR_RE = re\.compile\(\s*\n\s*(r"[^"]+")', src)
    assert pattern, "evaluation.py's _ERR_RE is not in the expected shape"
    assert pattern.group(1).strip('r"') in sig._ERR_RE.pattern

    probes = ["INSTRUCTION FOLLOWING ERROR: MAX CALL TIMES NOT FOLLOWED! total 5, max 4.",
              "instruction following error: format not followed! missing bold",
              "an ordinary result mentioning FORMAT and limits", "",
              "{'events': []}"]
    for text in probes:
        assert bool(re.search(sig._ERR_RE, text)) is (sig.violation_class_of(text) is not None)


def test_an_undeclared_class_is_still_a_violation_and_is_recorded():
    """Deliberately better than None-and-vanish: upstream drift must fail a test, not shrink a residual."""
    sig.UNDECLARED_CLASSES_SEEN.clear()
    try:
        got = sig.violation_class_of("INSTRUCTION FOLLOWING ERROR: BRAND NEW RULE NOT FOLLOWED!")
        assert got == "brand_new_rule"
        assert "brand_new_rule" in sig.UNDECLARED_CLASSES_SEEN
    finally:
        sig.UNDECLARED_CLASSES_SEEN.clear()


def test_both_feedback_channels_are_read():
    """Upstream splits violations by turn kind: tool-role per call, user-role merged on a final turn.

    A classifier reading one channel would miss every MIN-class and response-class violation --
    exactly the half that is not preventable.
    """
    tool_side = _turn(calls=[_call()], feedback=[
        {"role": "tool", "tool_call_id": "0",
         "content": "INSTRUCTION FOLLOWING ERROR: TOOL ORDER NOT FOLLOWED! ..."}],
        boundary="post_execution")
    user_side = _turn(feedback=[
        {"role": "user",
         "content": "INSTRUCTION FOLLOWING ERROR: MIN ROUND NOT FOLLOWED! ..."}],
        boundary="post_execution")
    assert tool_side["violation_class"] == "tool_order"
    assert user_side["violation_class"] == "min_round"
    assert tool_side["violation_is_behavior"] and user_side["violation_is_resource"]


def test_dimension_of_refuses_to_guess():
    assert caps.dimension_of("tool_order") == "behavior"
    assert caps.dimension_of("not_a_class") is None
    assert caps.dimension_of(None) is None


# ================================================================================================
# 2. the harness-fault classifier, and upstream's spelling
# ================================================================================================
def test_a_tool_fault_is_detected_on_upstreams_own_spelling():
    """`occured`, with one `r`. Matching the corrected spelling would make this return False for
    every fault -- and a detector that never fires is indistinguishable from a benchmark with no
    faults, so the whole class would be invisible to mining rather than merely unhandled."""
    assert sig.is_tool_fault("an error occured when call solar_event_finder: boom") is True
    assert sig.is_tool_fault("an error occurred when call x: boom") is False
    assert sig.is_tool_fault('{"events": [1]}') is False


def test_upstream_still_spells_it_that_way():
    """If upstream fixes the typo, this test fails instead of the detector silently going quiet."""
    src = (CCTU / "utils" / "utils.py").read_text()
    assert "an error occured when call" in src, (
        "upstream's tool-fault message changed spelling; cctu_signals._TOOL_FAULT_RE must follow, or "
        "every harness fault becomes invisible to mining")


def test_a_fault_is_never_also_vacuous():
    """Double-counting one event as two conditions inflates support."""
    fault = "an error occured when call t: boom"
    assert sig.is_tool_fault(fault) and sig.result_vacuity_kind(fault) is None
    e = _turn(calls=[_call()], boundary="post_execution",
              feedback=[{"role": "tool", "tool_call_id": "0", "content": fault}])
    assert e["result_is_error"] is True and e["result_is_empty"] is False


# ================================================================================================
# 3. vacuity delegates to the shared detector and extends only what it cannot see
# ================================================================================================
def test_vacuity_delegates_for_the_shape_the_core_detector_owns():
    from anchoropt.learning.policy_tree import vacuous_result_kind
    payload = '{"ranked_results": []}'
    assert vacuous_result_kind(payload) == "empty_collection"
    assert sig.result_vacuity_kind(payload) == "empty_collection"


@pytest.mark.parametrize("payload", ["", "   ", "None", "[]", "{}", "null"])
def test_vacuity_extends_to_the_shapes_the_core_detector_cannot_reach(payload):
    """`_iter_result_collections` returns immediately unless the payload is a JSON OBJECT with
    list-valued fields, so a bare empty collection or the literal "None" is invisible to it. CCTU
    produces all three, because `call_function` json.dumps a dict/list and str()s everything else."""
    from anchoropt.learning.policy_tree import vacuous_result_kind
    assert vacuous_result_kind(payload) is None
    assert sig.result_vacuity_kind(payload) == "empty_payload"


def test_an_informative_payload_is_not_vacuous():
    assert sig.result_vacuity_kind('{"events": [{"date": "2007-11-7"}]}') is None
    assert sig.result_vacuity_kind("2007-11-7") is None


# ================================================================================================
# 4. the two POST_GENERATION sub-cases, and TWO events per turn
# ================================================================================================
def test_the_two_post_generation_subcases_are_mutually_exclusive():
    for calls in ([], [_call()]):
        st = cctu.observable_state(_turn(calls=calls))
        term = cctu.evaluate_signal("terminal_response_proposed", st)
        tool = cctu.evaluate_signal("proposed_tool_action", st)
        assert term != tool


def test_absence_of_a_generation_is_not_a_terminal_response():
    """The prototype's `no_tool_call_yet` was inverted in exactly this way."""
    st = cctu.observable_state(cctu.normalize_event({"has_generation": False}))
    assert cctu.evaluate_signal("terminal_response_proposed", st) is False


def test_a_turn_expands_into_two_boundary_tagged_events():
    """ONE event per turn derives ONE boundary, and a residual with one boundary can never be
    observed to move earlier. `check_adapter.py` check 3 warns on precisely that."""
    msgs = [{"role": "assistant", "content": "look", "tool_calls": [_call()]},
            {"role": "tool", "tool_call_id": "0", "content": "{}"},
            {"role": "assistant", "content": "**done**"}]
    rows = cctu.raw_rows_from_messages(msgs)
    events = cctu.events_from_messages(msgs)
    assert len(rows) == len(events) == 4
    assert [cctu.boundary_key(e) for e in events] == [
        "post_generation_pre_exec", "post_execution"] * 2

    from anchoropt.learning.structured_search import localize
    boundaries = localize(events, runtime=cctu)
    assert len(boundaries) == 2, "WHERE must be a searchable coordinate on a real trajectory"


def test_the_post_execution_event_is_emitted_even_for_a_clean_turn():
    """Skipping it would make 'the validator ran and reported nothing' invisible, leaving the
    successes with no observation to contrast the failures against."""
    events = cctu.events_from_messages([{"role": "assistant", "content": "**done**"}])
    assert [cctu.boundary_key(e) for e in events] == [
        "post_generation_pre_exec", "post_execution"]
    assert events[1]["violation_class"] is None


def test_committing_to_an_answer_is_a_decision():
    """The most-repeated porting lesson: requiring a CALL here meant an episode that answered
    without acting contributed no boundary at all."""
    e = _turn(content="**answer**", calls=[])
    assert cctu.commits_to_answer(e) is True
    assert cctu.is_decision(e) is True


def test_results_place_an_event_past_the_gate():
    e = _turn(calls=[_call()], boundary=None,
              feedback=[{"role": "tool", "tool_call_id": "0", "content": "{}"}])
    assert cctu.boundary_key({k: v for k, v in e.items() if k != "boundary"}) == "post_execution"


# ================================================================================================
# 5. the signal surface
# ================================================================================================
def test_unknown_signal_raises_rather_than_returning_false():
    """A missing evaluator returning False is indistinguishable from 'did not fire' -- how an arm
    silently becomes the control arm."""
    with pytest.raises(KeyError, match="no evaluator"):
        cctu.evaluate_signal("does_not_exist", {})
    with pytest.raises(KeyError, match="no evaluator"):
        cctu.signal_boundaries("does_not_exist")


def test_an_optional_param_narrows_and_an_unknown_one_is_refused():
    st = cctu.observable_state(_turn(boundary="post_execution", feedback=[
        {"role": "user", "content": "INSTRUCTION FOLLOWING ERROR: MIN ROUND NOT FOLLOWED!"}]))
    assert cctu.evaluate_signal("constraint_violation_reported", st) is True
    assert cctu.evaluate_signal("constraint_violation_reported", st,
                                {"violation_class": "min_round"}) is True
    assert cctu.evaluate_signal("constraint_violation_reported", st,
                                {"violation_class": "max_length"}) is False
    with pytest.raises(ValueError, match="unknown param"):
        cctu.evaluate_signal("constraint_violation_reported", st, {"klass": "min_round"})


def test_every_declared_signal_is_evaluable_where_it_is_declared_observable():
    for name in cctu.declared_signals():
        at = cctu.signal_boundaries(name)
        assert at, f"{name} is declared but observable nowhere"
        for point in at:
            cctu.evaluate_signal(name, {"boundary": point.value}, cctu.probe_params(name))


def test_a_signal_can_be_added_without_touching_any_action_implementation():
    before = dict(cctu.apply_action(Action.NOOP, IncisionPoint.PRE_GENERATION, {}, {}))
    try:
        cctu.register_signal("smoke_only_signal", lambda s, p: True, {},
                             boundaries=("post_execution",))
        assert "smoke_only_signal" in cctu.declared_signals()
        assert cctu.evaluate_signal("smoke_only_signal", {}) is True
        assert dict(cctu.apply_action(Action.NOOP, IncisionPoint.PRE_GENERATION, {}, {})) == before
    finally:
        sig.SIGNALS.pop("smoke_only_signal", None)
        sig.SIGNAL_PARAMS.pop("smoke_only_signal", None)
        dict(sig.SIGNAL_BOUNDARIES).pop("smoke_only_signal", None)


def test_a_frozen_signal_cannot_be_silently_replaced():
    with pytest.raises(ValueError, match="already registered"):
        cctu.register_signal("proposed_tool_action", lambda s, p: False, {})


def test_an_expanded_signal_retains_the_boundary_it_was_validated_at():
    """Dropping it left `signal_boundaries` with nothing to answer for exactly the signals expansion
    creates, and the caller read that as 'observable nowhere'."""
    try:
        cctu.install_signal("expanded_probe", lambda st: True,
                            boundary="post_execution", provenance="test")
        assert cctu.signal_boundaries("expanded_probe") == {IncisionPoint.POST_EXECUTION}
        assert "expanded_probe" in cctu.declared_signals()
        assert cctu.expanded_signals() == ("expanded_probe",)
    finally:
        cctu.reset_expanded_signals()


def test_expansion_may_not_shadow_a_shipped_signal():
    with pytest.raises(ValueError, match="shipped signal"):
        cctu.install_signal("proposed_tool_action", lambda st: True)


def test_probe_params_ship_empty_so_no_measured_value_is_exported():
    assert dict(sig.SIGNAL_PROBE_PARAMS) == {}
    assert all(cctu.probe_params(n) == {} for n in sig.SIGNALS)


# ================================================================================================
# 6. U_H(l) -- declared only where an executor exists
# ================================================================================================
def test_every_declared_cell_has_an_executor():
    """Admissibility is not materializability. A cell declared with no executor is a candidate that
    reports grounded and cannot run."""
    for point in IncisionPoint:
        for action in cctu.feasible_actions(point):
            if action is Action.NOOP:
                continue
            assert cctu.executor_for(point, action) is not None, f"{point.value}/{action.value}"


def test_no_executor_claims_to_have_been_driven_yet():
    """Every cell is directive-level until a live run fires it. Reading a cell as live because it is
    declared is the mistake HOST.notes exists to prevent."""
    assert all(spec.get("driven") is False for spec in cctu.EXECUTORS.values())


def test_reroute_is_withheld_at_post_execution_and_suppress_is_structurally_excluded():
    assert Action.REROUTE not in cctu.feasible_actions(IncisionPoint.POST_EXECUTION)
    assert Action.SUPPRESS not in cctu.feasible_actions(IncisionPoint.POST_EXECUTION)
    assert Action.SUPPRESS in cctu.feasible_actions(IncisionPoint.POST_GENERATION_PRE_EXEC)


def test_the_two_rejections_stay_distinguishable():
    """'structurally impossible' and 'this host cannot' lead to different next steps, so a candidate
    pruned for the wrong reason would be unattributable."""
    with pytest.raises(ActionNotExecutable, match="structurally inadmissible"):
        cctu.apply_action(Action.SUPPRESS, IncisionPoint.POST_EXECUTION, {"reason": "r"}, {})
    with pytest.raises(ActionNotExecutable, match="cannot execute"):
        cctu.apply_action(Action.REROUTE, IncisionPoint.POST_EXECUTION,
                          {"destination": "t", "drop_args": ["a"]}, {})


def test_a_withheld_action_family_reports_its_missing_requirements():
    """`ground_transforms` returning [] must surface as a named infeasibility, not a shrunken grid."""
    from anchoropt.learning.action_contract import Operator, instantiate
    arms, failure = instantiate(Operator.TRANSFORM, signal="proposed_tool_action",
                                boundary=IncisionPoint.POST_GENERATION_PRE_EXEC, runtime=cctu)
    assert not arms and failure is not None
    assert set(failure.missing) == {"target_surface", "operator", "preservation"}


def test_grounding_does_not_whitelist_by_signal_name():
    """A two-name whitelist meant no synthesized signal could ever compose with SUPPRESS, so the
    search produced zero arms while the action was admissible and an executor existed."""
    try:
        cctu.install_signal("synth_probe", lambda st: True, boundary="post_generation_pre_exec")
        for grounder in (cctu.ground_reprompt, cctu.ground_suppress,
                         cctu.ground_substitute_destinations):
            assert grounder("synth_probe", IncisionPoint.POST_GENERATION_PRE_EXEC), grounder.__name__
    finally:
        cctu.reset_expanded_signals()


def test_the_executor_refuses_a_conflicting_parameter_rather_than_coercing_it():
    ok, _ = cctu.executor_supports(IncisionPoint.POST_EXECUTION, Action.REPROMPT,
                                   "constraint_violation_reported", {"charges_round": True})
    assert ok
    bad, why = cctu.executor_supports(IncisionPoint.POST_EXECUTION, Action.REPROMPT,
                                      "constraint_violation_reported", {"charges_round": False})
    assert not bad and "fixed_param_conflict" in why


def test_a_reprompt_charges_a_round_after_execution_and_not_at_the_gate():
    """The whole reason the commitment gate is preferable: the validator has not run there."""
    gate = cctu.apply_action(Action.REPROMPT, IncisionPoint.POST_GENERATION_PRE_EXEC,
                             {"text": "x"}, {})
    after = cctu.apply_action(Action.REPROMPT, IncisionPoint.POST_EXECUTION, {"text": "x"}, {})
    assert gate["charges_round"] is False and gate["request_redecision"] is True
    assert after["charges_round"] is True


def test_dropping_the_only_call_is_refused():
    """It would convert a tool turn into a final answer, and the terminal handlers would then judge
    content written for a tool turn. The precondition belongs in the directive, not in the executor's
    memory."""
    with pytest.raises(ValueError, match="more than one call"):
        cctu.apply_action(Action.SUPPRESS, IncisionPoint.POST_GENERATION_PRE_EXEC,
                          {"reason": "r"}, {"n_tool_calls": 1})
    kept = cctu.apply_action(Action.SUPPRESS, IncisionPoint.POST_GENERATION_PRE_EXEC,
                             {"reason": "r", "keep_call": True}, {"n_tool_calls": 1})
    assert kept["executed"] is True and kept["charges_budget"] is True


def test_a_reroute_that_changes_nothing_is_refused():
    with pytest.raises(ValueError, match="drop_args"):
        cctu.apply_action(Action.REROUTE, IncisionPoint.POST_GENERATION_PRE_EXEC,
                          {"destination": "t"}, {"n_tool_calls": 1})


def test_the_theta_contract_is_readable_and_matches_apply_action():
    """A contract the caller must satisfy but cannot read is a defect in the interface: a proposer
    once wrote a thoughtful reprompt into theta['guidance'] where the runtime requires theta['text']."""
    schema = cctu.action_theta_schema()
    assert set(schema) == {a.value for a in Action}
    with pytest.raises(ValueError, match=re.escape("theta['text']")):
        cctu.apply_action(Action.REPROMPT, IncisionPoint.PRE_GENERATION, {}, {})
    assert "text" in schema["reprompt"]["required"]


# ================================================================================================
# 7. THE LABEL RULE
# ================================================================================================
_LABEL_FIELDS = ("unsolved_set", "answer")


def test_no_declared_field_is_a_scoring_label():
    for name in caps.all_fields():
        assert name not in _LABEL_FIELDS
        assert "answer" not in name and "unsolved" not in name


def test_observable_state_cannot_carry_a_label_out_of_a_sample():
    """A discovery made with the answer key in the room is a lookup, however unused it looks."""
    event = cctu.normalize_event({
        "boundary": "post_execution", "has_generation": True,
        "message": {"content": "x", "tool_calls": []},
        # a caller handing the whole sample in by mistake
        "unsolved_set": '{"t": ["2007-11-7"]}', "answer": "2007-11-7"})
    state = cctu.observable_state(event)
    for key in _LABEL_FIELDS:
        assert key not in state, f"{key} reached observable state"
    assert "2007-11-7" not in json.dumps(state, default=str)


def test_the_adapter_never_names_a_label_field():
    src = (CCTU / "cctu_adapter.py").read_text() + (CCTU / "cctu_state.py").read_text()
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    for key in _LABEL_FIELDS:
        assert f'"{key}"' not in code and f"'{key}'" not in code, (
            f"{key} is a scoring label and must not be read by the adapter")


# ================================================================================================
# 8. no infinity, and the predictive path is pure
# ================================================================================================
def test_an_unconstrained_budget_is_none_never_inf():
    """`signal_grammar` would turn inf into `x < inf` (fires on everything finite) and `x > inf`
    (never fires), and shift every other quantile on the field."""
    snap = state_mod.CheckerSnapshot(max_call_times=math.inf, max_round=20,
                                     max_call_times_per_tool={"t": math.inf})
    carried = state_mod.carried_state(
        snap, {"tool_calls_raw": [_call()], "proposes_tool_call": True, "tool_name": "t",
               "content": "x"}, round_index=0)
    assert carried["call_times_remaining"] is None
    assert carried["calls_remaining_this_tool"] is None
    assert not any(isinstance(v, float) and not math.isfinite(v) for v in carried.values())


def test_the_inf_hazard_is_real_so_the_none_rule_is_not_decoration():
    """Verified rather than asserted: an inf in the observed values yields a degenerate atom pair."""
    from anchoropt.learning.signal_grammar import atoms_for

    class _F:
        type, enum, boundaries, doc = float, (), ("post_execution",), ""

    with_inf = atoms_for({"remaining": _F()}, [{"remaining": 1.0}, {"remaining": math.inf}])
    with_none = atoms_for({"remaining": _F()}, [{"remaining": 1.0}, {"remaining": None}])
    assert any(not math.isfinite(a.value) for a in with_inf if isinstance(a.value, float))
    assert all(math.isfinite(a.value) for a in with_none if isinstance(a.value, float))


def test_every_carried_key_is_declared():
    snap = state_mod.CheckerSnapshot()
    carried = state_mod.carried_state(snap, {"tool_calls_raw": [], "proposes_tool_call": False,
                                             "content": ""}, round_index=0)
    undeclared = set(carried) - set(caps.all_fields())
    assert not undeclared, f"carried_state produces undeclared field(s) {sorted(undeclared)}"


def test_rounds_remaining_is_zero_on_the_last_permitted_turn():
    """An off-by-one here is the difference between firing on the last turn and firing too late."""
    snap = state_mod.CheckerSnapshot(max_round=20)
    base = {"tool_calls_raw": [], "proposes_tool_call": False, "content": ""}
    assert state_mod.carried_state(snap, base, round_index=18)["rounds_remaining"] == 1
    assert state_mod.carried_state(snap, base, round_index=19)["rounds_remaining"] == 0


@needs_corpus
def test_asking_would_this_violate_does_not_charge_the_budget():
    """The stand-in must absorb every mutation. If the live checker moves, a pre-execution question
    has charged the budget it was asking about."""
    from utils.constraint_checker import DialogueConstraintChecker

    with open(DATA / "input_data.jsonl", encoding="utf-8") as fh:
        sample = json.loads(fh.readline())
    checker = DialogueConstraintChecker(sample={**sample, "id": f"{sample['id']}_0"},
                                        max_turns=20, validators_dir=str(DATA))
    before = (checker.round, checker.callTimes, dict(checker.callTimesPerTool),
              checker.accum_max_parallelCallTypes, dict(checker.earliest_callTurnPerTool),
              checker.first_tool_name)
    tool = min(checker.max_callTimesPerTool)
    calls = [_call(str(i), tool) for i in range(3)]
    snap = state_mod.snapshot_of(checker)
    for _ in range(3):
        state_mod.would_violate(snap, tool_calls=calls, content="x")
        state_mod.args_invalid(snap, calls)
    after = (checker.round, checker.callTimes, dict(checker.callTimesPerTool),
             checker.accum_max_parallelCallTypes, dict(checker.earliest_callTurnPerTool),
             checker.first_tool_name)
    assert before == after
    assert not state_mod.HANDLER_ERRORS, state_mod.HANDLER_ERRORS


@needs_corpus
def test_the_predictive_verdict_is_upstreams_verdict():
    """Not a mirror: the prediction runs upstream's own handler on a copy of the state."""
    from utils.constraint_checker import DialogueConstraintChecker

    with open(DATA / "input_data.jsonl", encoding="utf-8") as fh:
        sample = json.loads(fh.readline())
    checker = DialogueConstraintChecker(sample={**sample, "id": f"{sample['id']}_0"},
                                        max_turns=20, validators_dir=str(DATA))
    tool = min(checker.max_callTimesPerTool)
    checker.max_callTimesPerTool[tool] = 1
    snap = state_mod.snapshot_of(checker)
    two = [_call("0", tool), _call("1", tool)]
    assert "max_calls_per_tool" in state_mod.would_violate(snap, tool_calls=two)
    assert "max_calls_per_tool" not in state_mod.would_violate(snap, tool_calls=two[:1])


@needs_corpus
def test_a_validator_crash_is_contained_and_recorded_not_raised():
    """A tool name the episode does not offer makes upstream's own handler raise `KeyError`.

    `handlers/tool.py:40` does `checker.callTimesPerTool[name] += 1` on a plain dict built from the
    episode's tool list, so a hallucinated name raises. In the LIVE path that propagates out of
    `get_feedback`, `sample_process` retries it, and the episode is **dropped from the denominator**
    rather than scored -- which is worse than a wrong score, because a missing episode is invisible.

    Measured on the shipped granite baselines: it never fires (0 hallucinated calls across 140
    episodes), because the tool list travels with the request. So this is LATENT, and it is pinned
    here rather than fixed in the validator, which stays upstream's.

    What must hold is that the PREDICTIVE path never inherits the crash: `would_violate` records the
    failure and answers "no violation predicted" instead of raising. A guard that dies on a
    hallucinated tool name would take the episode with it.
    """
    from utils.constraint_checker import DialogueConstraintChecker

    with open(DATA / "input_data.jsonl", encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh]
    sample = next(s for s in samples
                  if any(str(c[1]).lower() == "specific tool call count"
                         for c in s["constraints_list"]))
    checker = DialogueConstraintChecker(sample={**sample, "id": f"{sample['id']}_0"},
                                        max_turns=20, validators_dir=str(DATA))
    hallucinated = [_call("0", "tool_that_does_not_exist")]

    # the live validator raises ...
    with pytest.raises(KeyError):
        checker.get_feedback_if(is_final=False, content="x", tool_calls=hallucinated)

    # ... and the predictive path does not.
    snap = state_mod.snapshot_of(checker)
    assert state_mod.would_violate(snap, tool_calls=hallucinated) == frozenset()
    assert any("KeyError" in e for e in state_mod.HANDLER_ERRORS), (
        "the failure must be RECORDED, not swallowed -- a guard that looks clean because it crashed "
        "is worse than no guard")


def test_only_the_declared_handlers_are_driven_predictively():
    """An explicit allowlist, so a handler added upstream does not silently start executing
    per-episode Python on every proposed answer."""
    names = {h.__name__ for h in state_mod.PREDICTIVE_HANDLERS()}
    assert not {"ResponseFormatHandler", "ResponsePunctuationHandler",
                "ResponseIdentifiersHandler"} & names


# ================================================================================================
# 9. the case-id grammar
# ================================================================================================
def test_the_case_id_grammar_raises_rather_than_guessing():
    assert cctu.parse_case_id("12_0") == ("12_0", 12, 0)
    assert cctu.query_id_of("7_1") == 7
    for bad in ("", "12", "abc_0", "12_0_0", None):
        with pytest.raises(cctu.UnknownCaseId):
            cctu.parse_case_id(bad)


# ================================================================================================
# 10. isolation
# ================================================================================================
def test_the_core_contains_no_cctu_vocabulary():
    banned = ("constraint_checker", "DialogueConstraintChecker", "callTimesPerTool",
              "accum_max_parallelCallTypes", "responseLength_unit", "unsolved_set",
              "constraints_list", "solve_rate_is_one", "func_set_timeout",
              "INSTRUCTION FOLLOWING ERROR", "mid_tool_call", "cctu")
    offenders = {}
    for path in (REPO / "anchoropt").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        hits = [b for b in banned if b in path.read_text()]
        if hits:
            offenders[str(path.relative_to(REPO))] = hits
    assert not offenders, f"CCTU vocabulary leaked into the core: {offenders}"


def test_importing_the_cctu_adapter_does_not_hijack_the_active_adapter():
    """One global slot: taking it on import would silently repoint BFCL's core call sites."""
    from anchoropt.attribution import adapter_or_none
    active = adapter_or_none()
    if active is not None:
        assert getattr(active, "NAME", None) != cctu.NAME
        assert active is not sys.modules["cctu_adapter"]


def test_the_adapter_satisfies_the_declared_hook_contract():
    from adapters.adapter_template import OPTIONAL_HOOKS, REQUIRED_HOOKS
    missing = [h for h in REQUIRED_HOOKS if getattr(cctu, h, None) is None]
    assert not missing, f"missing required hook(s) {missing}"
    grounders = [h for h in OPTIONAL_HOOKS
                 if h.startswith("ground_") and getattr(cctu, h, None) is not None]
    assert grounders, "no eta grounder: no candidate could ever be built"
    assert getattr(cctu.ADAPTER, "HOST", None) is cctu.HOST
