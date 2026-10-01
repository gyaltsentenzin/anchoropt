"""The CCTU middleware: the persisted format, the load-time refusals, and thread isolation.

Every regression here exists because the failure it catches is SILENT:

  * a spec that cannot run producing a middleware that no-ops -- a candidate arm quietly becoming the
    control arm, which is the most expensive measurement error available here;
  * per-episode state living on the shared middleware, so four concurrent episodes share one turn
    counter and one one-shot guard. `response_generator.py` runs a `ThreadPoolExecutor`, so this is
    not hypothetical, and no single-episode test can reveal it;
  * a suppressed call vanishing from the trace, which erases the evidence needed to evaluate the
    suppression. `docs/GENERALIZABILITY.md` records that this recurred four times on the other
    benchmark.

Unlike `tests/test_tb2_middleware.py` these need no agent framework: CCTU's middleware is a plain
object the episode loop calls at three points, so everything here runs in the core suite.
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CCTU = REPO / "benchmarks" / "cctu"
for _p in (str(REPO), str(CCTU)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cctu_adapter as cctu  # noqa: E402
import cctu_apply as apply_mod  # noqa: E402
import cctu_middleware as mw  # noqa: E402
import cctu_state as state_mod  # noqa: E402
from anchoropt.runtime import ActionNotExecutable  # noqa: E402

TEXT = "Plan the calls you still need before acting again."


@pytest.fixture(autouse=True)
def _clean_hook_registry():
    """`anchoropt.runtime_hook` holds ONE global installed-controller table.

    That is correct for a run -- one arm, one process -- and it means a test installing a controller
    would otherwise leak it into every later test, including the ones asserting the control is inert.
    """
    from anchoropt.runtime_hook import reset
    reset()
    yield
    reset()


def _spec(**kw):
    d = dict(controller_id="A1", boundary="post_generation_pre_exec", action="reprompt",
             signal="proposed_tool_action", theta={"text": TEXT})
    d.update(kw)
    return mw.ControllerSpec(**d)


def _call(cid="0", name="t", args="{}"):
    return {"id": cid, "function": {"name": name, "arguments": args}}


def _turn(calls=(), content="x", feedback=(), boundary="post_generation_pre_exec"):
    return cctu.normalize_event({"boundary": boundary, "has_generation": True,
                                 "message": {"content": content, "tool_calls": list(calls)},
                                 "feedback": list(feedback)})


# ================================================================================================
# 1. the persisted format
# ================================================================================================
def test_spec_roundtrips_through_disk(tmp_path):
    spec = _spec()
    (got,) = mw.ControllerSpec.load(spec.write(tmp_path / "c.json"))
    assert got == spec


def test_spec_rejects_an_unknown_format(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"format": "something.else", "controller_id": "x", '
                 '"boundary": "pre_generation", "action": "noop"}')
    with pytest.raises(ValueError, match=mw.SPEC_FORMAT):
        mw.ControllerSpec.load(p)


def test_the_format_is_the_same_one_the_other_adapter_emits():
    """Two runtimes emitting one spec format is what makes a cross-benchmark claim stateable."""
    assert mw.SPEC_FORMAT == "anchoropt.controller.v1"


def test_a_bundle_of_controllers_loads(tmp_path):
    """`--controllers` promises 'one spec, or {"controllers": [...]}'."""
    a, b = _spec(controller_id="A"), _spec(controller_id="B")
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps({"controllers": [json.loads(a.to_json()),
                                             json.loads(b.to_json())]}))
    assert [s.controller_id for s in mw.ControllerSpec.load(p)] == ["A", "B"]


def test_the_spec_carries_no_benchmark_vocabulary():
    fields = set(mw.ControllerSpec.__dataclass_fields__)
    assert fields == {"controller_id", "boundary", "action", "theta", "signal", "predicate",
                      "signal_params", "provenance"}


# ================================================================================================
# 2. a spec that cannot run fails at LOAD, not mid-episode
# ================================================================================================
def test_an_undeclared_signal_is_refused():
    with pytest.raises(ValueError, match="not declared"):
        _spec(signal="no_such_signal").validate()


def test_a_signal_installed_where_it_is_not_observable_is_refused():
    """A condition evaluated where its facts do not exist answers False for a structural reason
    indistinguishable from 'it did not hold'."""
    with pytest.raises(ValueError, match="not observable"):
        _spec(boundary="post_execution", signal="proposed_tool_action").validate()


def test_an_action_outside_U_H_is_refused():
    with pytest.raises(ActionNotExecutable):
        _spec(boundary="post_execution", signal="constraint_violation_reported", action="reroute",
              theta={"destination": "t", "drop_args": ["a"]}).validate()


def test_a_theta_missing_a_required_key_is_refused():
    """The dry run proves the KEYS are present; the live precondition is re-checked at firing time."""
    with pytest.raises(ValueError, match="non-empty"):
        _spec(theta={}).validate()


def test_exactly_one_of_signal_or_predicate_is_required():
    with pytest.raises(ValueError, match="exactly one"):
        _spec(predicate={"field": "n_tool_calls", "op": "gt", "value": 1}).validate()
    with pytest.raises(ValueError, match="exactly one"):
        mw.ControllerSpec(controller_id="x", boundary="post_execution", action="reprompt",
                          theta={"text": TEXT}).validate()


def test_a_previous_generation_mechanism_policy_is_refused_not_ignored(tmp_path):
    """Ignoring it would run the CONTROL under an arm's name."""
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"enable": {"enable_reroute": True}}))
    with pytest.raises(ValueError, match="previous-generation"):
        mw.refuse_legacy_policy(p)


def test_an_unrecognisable_policy_shape_is_refused_rather_than_guessed(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"something": 1}))
    with pytest.raises(ValueError, match="refusing rather than guessing"):
        mw.refuse_legacy_policy(p)


# ================================================================================================
# 3. a SYNTHESIZED predicate can be persisted -- what the TB2 format cannot do
# ================================================================================================
def test_a_synthesized_predicate_roundtrips_and_fires(tmp_path):
    """A condition the search synthesized this session must survive a fresh process, or it can be
    installed and never persisted."""
    spec = mw.ControllerSpec(
        controller_id="syn", boundary="post_generation_pre_exec", action="reprompt",
        theta={"text": TEXT},
        predicate={"name": "gate_pressure",
                   "all": [{"field": "call_times_over_cap", "op": "is_true"},
                           {"field": "proposes_tool_call", "op": "is_true"}]})
    (loaded,) = mw.ControllerSpec.load(spec.write(tmp_path / "c.json"))
    loaded.validate()
    controller = mw.InstalledController(loaded)
    assert controller.fires_on({"call_times_over_cap": True, "proposes_tool_call": True}) is True
    assert controller.fires_on({"proposes_tool_call": True}) is False, \
        "a missing field is NOT a firing"


def test_a_predicate_over_an_undeclared_field_is_refused():
    from anchoropt.learning.signal_lang import SignalSpecError
    with pytest.raises(SignalSpecError, match="not declared"):
        mw.ControllerSpec(controller_id="x", boundary="post_execution", action="reprompt",
                          theta={"text": TEXT},
                          predicate={"name": "z", "field": "made_up", "op": "is_true"}).validate()


def test_a_proposal_only_predicate_cannot_be_installed_before_generation():
    """The guard that actually bites on this alphabet: every CCTU field is readable at
    POST_EXECUTION, so `compile_signal`'s empty-intersection rejection can never fire here."""
    expr = {"name": "p", "field": "proposes_tool_call", "op": "is_true"}
    with pytest.raises(ValueError, match="not observable"):
        mw.ControllerSpec(controller_id="x", boundary="pre_generation", action="reprompt",
                          theta={"text": TEXT}, predicate=expr).validate()
    mw.ControllerSpec(controller_id="x", boundary="post_generation_pre_exec", action="reprompt",
                      theta={"text": TEXT}, predicate=expr).validate()


# ================================================================================================
# 4. the control arm is inert BY CONSTRUCTION
# ================================================================================================
def test_no_spec_means_nothing_installed_and_every_hook_returns_none(tmp_path):
    m = mw.AnchorOptMiddleware.from_path(None, trace_path=tmp_path / "t.jsonl")
    assert m.is_control is True
    hooks = m.episode("0_0")
    norm = _turn(calls=[_call()])
    assert hooks.pre_generation() is None
    assert hooks.post_generation_pre_exec(normalized=norm) is None
    assert hooks.post_execution(normalized=norm) is None
    assert m.telemetry_snapshot()["interventions_executed"] == 0


def test_the_control_still_records_the_denominator(tmp_path):
    """A firing count with no exposure count is not a rate."""
    trace = tmp_path / "t.jsonl"
    m = mw.AnchorOptMiddleware.from_path(None, trace_path=trace)
    hooks = m.episode("0_0")
    hooks.post_generation_pre_exec(normalized=_turn(calls=[_call()]))
    rows = [json.loads(x) for x in trace.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["fired"] is False and rows[0]["controller"] is None


# ================================================================================================
# 5. firing, the one-shot guard, and the trace
# ================================================================================================
def test_a_controller_fires_and_the_trace_records_the_proposal_before_the_intervention(tmp_path):
    trace = tmp_path / "t.jsonl"
    m = mw.AnchorOptMiddleware([_spec()], trace_path=trace)
    directive = m.episode("0_0").post_generation_pre_exec(normalized=_turn(calls=[_call(), _call("1")]))
    assert directive["kind"] == "reprompt" and directive["executed"] is True
    row = json.loads(trace.read_text().splitlines()[0])
    assert row["fired"] and row["executed"]
    assert row["proposed"]["n_tool_calls"] == 2, "the proposal must be recorded as PROPOSED"
    assert row["signal"] == "proposed_tool_action"


def test_one_shot_bounds_the_intervention_and_records_why(tmp_path):
    trace = tmp_path / "t.jsonl"
    m = mw.AnchorOptMiddleware([_spec()], trace_path=trace)
    hooks = m.episode("0_0")
    norm = _turn(calls=[_call()])
    assert hooks.post_generation_pre_exec(normalized=norm) is not None
    assert hooks.post_generation_pre_exec(normalized=norm) is None
    t = m.telemetry_snapshot()
    assert t["signal_firings"] == 2 and t["interventions_executed"] == 1
    assert t["loop_prevention_events"] == 1
    assert any("one_shot" in json.loads(x)["detail"] for x in trace.read_text().splitlines())


def test_a_live_precondition_declines_and_records_rather_than_raising(tmp_path):
    """SUPPRESS/drop_call on a turn proposing one call: refused, traced, episode continues."""
    trace = tmp_path / "t.jsonl"
    spec = _spec(action="suppress", theta={"reason": "over the cap"})
    m = mw.AnchorOptMiddleware([spec], trace_path=trace)
    assert m.episode("0_0").post_generation_pre_exec(normalized=_turn(calls=[_call()])) is None
    t = m.telemetry_snapshot()
    assert t["signal_firings"] == 1 and t["interventions_executed"] == 0
    assert any("more than one call" in e for e in t["errors"])
    assert any("declined" in json.loads(x)["detail"] for x in trace.read_text().splitlines())


def test_a_raising_predicate_is_not_a_trigger(tmp_path):
    """`runtime_hook.decide` catches it and records the error. A predicate that throws must never be
    read as a fire."""
    from anchoropt.runtime_hook import errors, install

    class _Boom:
        name = "boom"

        def __init__(self):
            self.eta = {}

        def fires_on(self, state):
            raise RuntimeError("predicate exploded")

    install("post_generation_pre_exec", _Boom())
    m = mw.AnchorOptMiddleware([], trace_path=tmp_path / "t.jsonl")
    assert m.episode("0_0").post_generation_pre_exec(normalized=_turn(calls=[_call()])) is None
    assert any("predicate exploded" in e for e in errors())


# ================================================================================================
# 6. THREAD ISOLATION -- the guarantee with no precedent in the other integrations
# ================================================================================================
def test_per_episode_state_is_per_episode_object_not_a_middleware_attribute():
    m = mw.AnchorOptMiddleware([])
    a, b = m.episode("0_0"), m.episode("1_0")
    a.round_index = 7
    a.seen_calls.append(("t", "{}"))
    assert b.round_index == 0 and b.seen_calls == []
    assert a is not b


def test_concurrent_episodes_keep_their_own_counters_and_one_shot_guard(tmp_path):
    """`max_workers=4` is the default. A middleware attribute here would be four episodes' counters
    added together -- silently, in a way no single-episode test can show."""
    trace = tmp_path / "t.jsonl"
    m = mw.AnchorOptMiddleware([_spec()], trace_path=trace)
    turns, episodes = 5, 24
    norm = _turn(calls=[_call(), _call("1")])

    def run(i):
        hooks = m.episode(f"{i}_0")
        fired = 0
        for _ in range(turns):
            if hooks.post_generation_pre_exec(normalized=norm) is not None:
                fired += 1
            hooks.observe_turn(norm)
        return hooks.round_index, len(hooks.seen_calls), fired

    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(run, range(episodes)))

    assert {r[0] for r in got} == {turns}, "an episode counted another episode's turns"
    assert {r[1] for r in got} == {turns * 2}, "an episode saw another episode's calls"
    assert {r[2] for r in got} == {1}, "the one-shot guard is not per episode"

    t = m.telemetry_snapshot()
    assert t["boundary_exposures"] == turns * episodes
    assert t["interventions_executed"] == episodes
    rows = trace.read_text().splitlines()
    assert len(rows) == t["trace_rows"] == turns * episodes, "the trace lost lines under concurrency"
    assert len({json.loads(x)["case_id"] for x in rows}) == episodes
    for line in rows:
        json.loads(line)          # no torn lines


def test_the_trace_writer_is_serialised():
    """Four threads appending to one JSONL without a lock is how a trace loses lines -- and a trace
    missing lines UNDERSTATES firings, which is the number that gates reading any accuracy."""
    m = mw.AnchorOptMiddleware([])
    assert isinstance(m._lock, type(threading.Lock()))


# ================================================================================================
# 7. the recorded-result cache feeds only from calls that actually executed
# ================================================================================================
def test_only_executed_calls_populate_the_replay_cache():
    """Invariant 4. Recording a substituted observation would let the cache feed itself; recording a
    violation message would replay a refusal as though it were a result."""
    m = mw.AnchorOptMiddleware([])
    hooks = m.episode("0_0")
    norm = _turn(calls=[_call("0", "t", '{"a":1}'), _call("1", "t", '{"a":2}')])
    hooks.observe_turn(norm, executed={"0": "REAL RESULT"})
    assert hooks.results_by_call == {("t", '{"a":1}'): "REAL RESULT"}


def test_the_violation_streak_resets_on_a_clean_turn():
    """It measures a stuck self-refinement loop, not a running total."""
    hooks = mw.AnchorOptMiddleware([]).episode("0_0")
    dirty = _turn(boundary="post_execution", feedback=[
        {"role": "user", "content": "INSTRUCTION FOLLOWING ERROR: MIN ROUND NOT FOLLOWED!"}])
    clean = _turn(boundary="post_execution")
    hooks.observe_turn(dirty)
    hooks.observe_turn(dirty)
    assert hooks.consecutive_violation_turns == 2
    assert hooks.last_violation_class == "min_round"
    hooks.observe_turn(clean)
    assert hooks.consecutive_violation_turns == 0
    assert hooks.last_violation_class == "min_round", "the last class is history, not a streak"


# ================================================================================================
# 8. the mechanisms' proof obligations
# ================================================================================================
def test_a_withheld_call_costs_zero_executions():
    """`docs/EFFICIENCY_CLASS.md` invariant 1, and the only thing that distinguishes a saving from a
    behaviour change: the first implementation of this class memoised AFTER the executor ran and saved
    nothing while passing every predicate test."""
    import utils.utils as uu

    calls = [_call("0", "t", '{"a":1}'), _call("1", "t", '{"a":2}')]
    invoked = []
    real = uu.call_function

    def counting(name, arguments, code, **kw):
        invoked.append(json.dumps(arguments, sort_keys=True))
        return '{"ok": true}'

    uu.call_function = counting
    try:
        applied = apply_mod.apply_directive(
            {"kind": "suppress", "executed": True, "keep_call": True, "reason": "r"},
            {"content": "x", "tool_calls": calls},
            recorded_results={("t", '{"a":1}'): "RECORDED VERBATIM"})
        to_run = apply_mod.execute_ids(calls, applied.withheld_ids)
        executed = uu.get_feedback_tools([], to_run, {"t": "def t(a=None):\n    return {'ok': 1}\n"})
        merged = apply_mod.merge_feedback(calls, executed, applied.substitutions)
    finally:
        uu.call_function = real

    assert applied.withheld_ids == {"0"} and not applied.declined
    assert '{"a": 1}' not in invoked, "the executor RAN a withheld call: a saving that saves nothing"
    assert [m["tool_call_id"] for m in merged] == ["0", "1"], "spliced out of proposed order"
    assert merged[0]["content"] == "RECORDED VERBATIM", "the replay must be VERBATIM"


def test_withholding_declines_without_a_recorded_result():
    applied = apply_mod.apply_directive(
        {"kind": "suppress", "executed": True, "keep_call": True, "reason": "r"},
        {"content": "x", "tool_calls": [_call()]}, recorded_results={})
    assert applied.declined and "recorded" in applied.detail


def test_drop_call_verifies_its_invariant_or_declines():
    """propose OR DECLINE, then verify -- `constraint_repair.py`'s shape. A suppression that drops a
    call without checking whether the violation clears has spent a call and guessed."""
    snap = state_mod.CheckerSnapshot(max_call_times_per_tool={"t": 1}, call_times_per_tool={"t": 0})
    four = [_call(str(i), "t") for i in range(4)]
    d = {"kind": "suppress", "executed": True, "keep_call": False, "reason": "r",
         "target_violation": "max_calls_per_tool"}

    ok = apply_mod.apply_directive(d, {"content": "x", "tool_calls": four}, snapshot=snap)
    assert not ok.declined and len(ok.message["tool_calls"]) == 1
    assert "verified clear" in ok.detail
    assert "max_calls_per_tool" not in state_mod.would_violate(
        snap, tool_calls=ok.message["tool_calls"])

    hopeless = state_mod.CheckerSnapshot(max_call_times_per_tool={"t": 0},
                                         call_times_per_tool={"t": 0})
    no = apply_mod.apply_directive(d, {"content": "x", "tool_calls": four}, snapshot=hopeless)
    assert no.declined and "declining rather than spending a call on a guess" in no.detail


def test_an_unverifiable_invariant_is_not_a_satisfied_one():
    d = {"kind": "suppress", "executed": True, "keep_call": False, "reason": "r",
         "target_violation": "max_call_times"}
    got = apply_mod.apply_directive(d, {"content": "x", "tool_calls": [_call(), _call("1")]},
                                    snapshot=None)
    assert got.declined and "cannot be verified" in got.detail


def test_argument_repair_deletes_only_what_the_schema_rejects():
    """Pure deletion: nothing synthesized, so a valid call cannot be turned into an invalid one.

    `drop_unknown_args` is EXPLICIT. Deriving the deletions whenever no names were given would mean a
    spec that asked for nothing still rewrote the call, and the grounded variant needs to say which
    behaviour it wants -- the names are a per-call fact a persisted spec cannot enumerate.
    """
    tools_doc = {"t": {"properties": {"keep": {"type": "string"}}, "required": []}}
    call = _call("0", "t", json.dumps({"keep": "a", "bogus": 1}))
    got = apply_mod.apply_directive(
        {"kind": "reroute", "executed": True, "destination": "t", "drop_args": (),
         "args_patch": {}, "drop_unknown_args": True},
        {"content": "x", "tool_calls": [call]}, tools_doc=tools_doc)
    assert not got.declined
    assert json.loads(got.message["tool_calls"][0]["function"]["arguments"]) == {"keep": "a"}


def test_argument_repair_does_nothing_unless_it_was_asked_to():
    """Neither explicit names nor `drop_unknown_args` means no rewrite, and a decline says so."""
    tools_doc = {"t": {"properties": {"keep": {"type": "string"}}, "required": []}}
    call = _call("0", "t", json.dumps({"keep": "a", "bogus": 1}))
    got = apply_mod.apply_directive(
        {"kind": "reroute", "executed": True, "destination": "t", "drop_args": (), "args_patch": {}},
        {"content": "x", "tool_calls": [call]}, tools_doc=tools_doc)
    assert got.declined and "no argument would change" in got.detail


def test_a_reroute_that_would_change_nothing_declines():
    """Reporting a successful rewrite that changed nothing is the `executed=True` on an empty payload
    shape that `docs/CONSUMER_BOUNDARY_RULE.md` exists to prevent."""
    tools_doc = {"t": {"properties": {"keep": {"type": "string"}}, "required": []}}
    call = _call("0", "t", json.dumps({"keep": "a"}))
    got = apply_mod.apply_directive(
        {"kind": "reroute", "executed": True, "destination": "t", "drop_args": (),
         "args_patch": {}, "drop_unknown_args": True},
        {"content": "x", "tool_calls": [call]}, tools_doc=tools_doc)
    assert got.declined and "no argument would change" in got.detail


def test_an_unparseable_argument_string_is_left_for_the_validator():
    """Deletion cannot repair it, and inventing a replacement is what this must never do."""
    tools_doc = {"t": {"properties": {}, "required": []}}
    call = _call("0", "t", "{not json")
    got = apply_mod.apply_directive(
        {"kind": "reroute", "executed": True, "destination": "t", "drop_args": (), "args_patch": {}},
        {"content": "x", "tool_calls": [call]}, tools_doc=tools_doc)
    assert got.declined


# ================================================================================================
# 9. the redecide budget at the commitment gate
# ================================================================================================
#
# A granted redecide discards the generation and regenerates WITHOUT charging a round
# (`response_generator.py`'s l2 branch: `redecide = True; break`, then `continue`). `times` is only
# incremented in the exception handler, so nothing in the episode loop bounds that cycle. The
# contract has always declared `retry_budget` and `evolve_memory/primitives.py` says it is "FIXED at
# 1 by the executor" -- these tests are what make that true on this runtime.
#
# It was not live before only because `one_shot` caps a controller at one intervention per episode.
# That bound disappears the moment a per-turn condition is measured, which is exactly what a
# duplicate-call gate needs, so these tests pin the budget with `one_shot=False`.
def test_the_redecide_budget_bounds_regeneration_when_one_shot_is_off(tmp_path):
    trace = tmp_path / "t.jsonl"
    m = mw.AnchorOptMiddleware([_spec()], one_shot=False, trace_path=trace)
    hooks = m.episode("0_0")
    norm = _turn(calls=[_call()])

    first = hooks.post_generation_pre_exec(normalized=norm)
    assert first is not None and first["request_redecision"] is True
    # The SAME turn asks again: the generation was discarded, so nothing reset the budget.
    assert hooks.post_generation_pre_exec(normalized=norm) is None, \
        "a second redecide on one turn is the unbounded loop"
    t = m.telemetry_snapshot()
    assert t["signal_firings"] == 2 and t["interventions_executed"] == 1
    assert t["redecide_budget_declines"] == 1
    assert t["loop_prevention_events"] == 0, "this is the budget, not the one-shot guard"
    assert any("redecide budget spent" in json.loads(x)["detail"]
               for x in trace.read_text().splitlines())


def test_a_completed_turn_restores_the_redecide_budget(tmp_path):
    """`observe_turn` runs once per COMPLETED turn and a redecide never reaches it, which is what
    makes the counter a per-turn budget rather than a per-episode one."""
    m = mw.AnchorOptMiddleware([_spec()], one_shot=False, trace_path=tmp_path / "t.jsonl")
    hooks = m.episode("0_0")
    norm = _turn(calls=[_call()])
    assert hooks.post_generation_pre_exec(normalized=norm) is not None
    assert hooks.post_generation_pre_exec(normalized=norm) is None
    hooks.observe_turn(_turn(calls=[_call()], boundary="post_execution"))
    assert hooks.post_generation_pre_exec(normalized=norm) is not None, \
        "the next turn must get a fresh budget"
    assert m.telemetry_snapshot()["interventions_executed"] == 2


def test_a_zero_budget_disables_the_gate_reprompt_while_leaving_it_wired(tmp_path):
    m = mw.AnchorOptMiddleware([_spec()], one_shot=False, redecide_budget=0,
                               trace_path=tmp_path / "t.jsonl")
    assert m.episode("0_0").post_generation_pre_exec(normalized=_turn(calls=[_call()])) is None
    t = m.telemetry_snapshot()
    assert t["signal_firings"] == 1 and t["interventions_executed"] == 0
    assert t["redecide_budget_declines"] == 1


def test_a_negative_redecide_budget_is_refused_at_construction():
    with pytest.raises(ValueError, match="redecide_budget"):
        mw.AnchorOptMiddleware([], redecide_budget=-1)


def test_the_budget_does_not_touch_post_execution(tmp_path):
    """`apply_action` sets `request_redecision` at POST_EXECUTION too, but the runner does not
    regenerate there -- it injects and clears `finish`. Widening the check would change the semantics
    cycle 1 measured against a frozen control."""
    spec = _spec(boundary="post_execution", signal="constraint_violation_reported")
    m = mw.AnchorOptMiddleware([spec], one_shot=False, trace_path=tmp_path / "t.jsonl")
    hooks = m.episode("0_0")
    violated = _turn(boundary="post_execution", calls=[_call()],
                     feedback=[{"role": "tool", "tool_call_id": "0",
                                "content": "INSTRUCTION FOLLOWING ERROR: MAX CALL TIMES NOT FOLLOWED!"}])
    for _ in range(3):
        assert hooks.post_execution(normalized=violated) is not None
    t = m.telemetry_snapshot()
    assert t["interventions_executed"] == 3
    assert t["redecide_budget_declines"] == 0


def test_the_budget_is_recorded_in_the_telemetry_contract(tmp_path):
    """A run whose firing regime is not recorded cannot be reproduced, only re-attempted."""
    m = mw.AnchorOptMiddleware([], one_shot=False, redecide_budget=2)
    assert m.one_shot is False and m.redecide_budget == 2
    assert "redecide_budget_declines" in m.telemetry_snapshot()
