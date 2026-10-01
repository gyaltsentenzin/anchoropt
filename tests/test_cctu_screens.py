"""The pre-arm screens: what they refuse, what they must NOT refuse, and the one they cannot decide.

Each test here corresponds to an arm cycle 1 actually spent.

  * three arms fired 7-8 times across 280 episodes on a class declared for EXCLUSION from the
    residual. That is the SUPPORT screen.
  * three arms fired 1500+ times and could not win, because `PSR = acc AND has_if_error == 0` and
    their trigger was `constraint_violation_reported` -- so the benefit metric was already 0 at the
    instant the signal fired and could not rise again. Every gain such an arm shows must land outside
    the episodes it acted in, which the ATTRIBUTION term vetoes after the compute is spent. That is
    the REACHABILITY screen.

THE MOST IMPORTANT TEST IN THIS FILE is `test_an_unresolvable_signal_is_unscreenable_not_vetoed`. A
screen that reports "fires in 0 episodes" for a predicate that does not exist yet is a confident
refusal computed from a number about the screen rather than about the candidate -- the same shape
`evaluate_signal` raises KeyError to avoid, and a screen is the last place to reintroduce it. A
synthesized condition is unscreenable until Phi-expansion registers it, and saying so is the whole
difference between a veto and a missing evaluator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CCTU = REPO / "benchmarks" / "cctu"
for _p in (str(REPO), str(CCTU)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cctu_adapter as cctu  # noqa: E402
import cctu_screens as screens  # noqa: E402


def _call(cid="0", name="t", args="{}"):
    return {"id": cid, "function": {"name": name, "arguments": args}}


def _state(*, boundary="post_execution", calls=(), feedback=(), content="x"):
    return cctu.observable_state(cctu.normalize_event(
        {"boundary": boundary, "has_generation": True,
         "message": {"content": content, "tool_calls": list(calls)},
         "feedback": list(feedback)}))


VIOLATION = [{"role": "tool", "tool_call_id": "0",
              "content": "INSTRUCTION FOLLOWING ERROR: MAX CALL TIMES NOT FOLLOWED!"}]


def _control(episodes: dict[str, tuple[list[dict], dict]]) -> dict:
    """`case_id -> (states, outcome)` -> the shape `load_control` returns.

    Built by hand rather than replayed, so these tests exercise the screens and not the checker.
    """
    return {
        "run": "synthetic", "split": "train",
        "states": {k: v[0] for k, v in episodes.items()},
        "detail": {k: {"id": k, **v[1]} for k, v in episodes.items()},
        "episodes": len(episodes),
        "replay_errors": [],
    }


def _won(**kw):
    return {"acc": True, "SR": True, "PSR": True, **kw}


def _lost(**kw):
    return {"acc": False, "SR": False, "PSR": False, **kw}


# ================================================================================================
# 1. SUPPORT -- the screen that would have saved three arms on tool_execution_error
# ================================================================================================
def test_support_counts_episodes_not_events():
    """A signal firing 500 times inside 2 episodes has support 2. Every metric is scored per
    episode, and `cycle1_propose.py` records what the event ordering costs: '512 events spread over
    far fewer episodes, and an anchor is judged per episode'."""
    busy = [_state(calls=[_call()], feedback=VIOLATION) for _ in range(250)]
    control = _control({"1_0": (busy, _lost()), "2_0": (busy, _lost())})
    got = screens.screen_support(control, "constraint_violation_reported",
                                 boundary="post_execution", min_episodes=10)
    assert got["episodes_firing"] == 2
    assert got["events_firing"] == 500
    assert got["verdict"] == "VETO"
    assert "under the floor" in got["detail"]


def test_support_passes_a_signal_with_real_breadth():
    eps = {f"{i}_0": ([_state(calls=[_call()], feedback=VIOLATION)], _lost()) for i in range(12)}
    got = screens.screen_support(_control(eps), "constraint_violation_reported",
                                 boundary="post_execution", min_episodes=10)
    assert got["verdict"] == "PASS" and got["episodes_firing"] == 12


def test_support_is_counted_at_the_candidate_s_own_boundary():
    """A condition credited with firings at a point the controller never evaluates is the
    '-4.95pp against +3.63pp for byte-identical text one boundary later' defect."""
    states = [_state(boundary="post_generation_pre_exec", calls=[_call()]),
              _state(boundary="post_execution", calls=[_call()], feedback=VIOLATION)]
    control = _control({f"{i}_0": (states, _lost()) for i in range(12)})
    at_gate = screens.screen_support(control, "proposed_tool_action",
                                     boundary="post_generation_pre_exec")
    assert at_gate["episodes_firing"] == 12
    # The same signal, asked about a boundary where its fact is not the one being decided.
    after = screens.screen_support(control, "constraint_violation_reported",
                                   boundary="post_generation_pre_exec")
    assert after["episodes_firing"] == 0 and after["verdict"] == "VETO"


# ================================================================================================
# 2. REACHABILITY -- the screen that would have saved arms 01-03
# ================================================================================================
def test_reachability_vetoes_a_metric_pinned_under_its_own_trigger():
    """The cycle 1 shape exactly: huge support, and zero attributable headroom.

    `PSR = acc AND has_if_error == 0`, and `constraint_violation_reported` is true exactly when such
    an error exists, so PSR is 0 in every episode the signal can fire in. 1512 firings over 198
    episodes bought a result that was arithmetically fixed before the run started.
    """
    eps = {f"{i}_0": ([_state(calls=[_call()], feedback=VIOLATION)], _lost()) for i in range(40)}
    got = screens.screen_reachability(_control(eps), "constraint_violation_reported",
                                      benefit="PSR", boundary="post_execution")
    assert got["verdict"] == "VETO"
    assert got["benefit_already_1"] == 0
    assert "PINNED under the trigger" in got["detail"]


def test_reachability_passes_when_the_metric_survives_the_trigger():
    """The same signal against a RECOVERABLE metric. `SR` reads `has_last_if_error`, which scans only
    past the last assistant message, so an episode can violate early and still score SR -- which is
    why the benefit metric is a choice the screen has an opinion about."""
    fires = [_state(calls=[_call()], feedback=VIOLATION)]
    eps = {f"{i}_0": (fires, _won() if i < 5 else _lost()) for i in range(40)}
    got = screens.screen_reachability(_control(eps), "constraint_violation_reported",
                                      benefit="SR", boundary="post_execution")
    assert got["verdict"] == "PASS"
    assert got["benefit_already_1"] == 5
    assert got["headroom"] == 35, "headroom is the arithmetic upper bound, not a prediction"


def test_reachability_is_restricted_to_the_episodes_the_signal_fires_in():
    """A metric that is 1 somewhere in the corpus but never where the signal fires is still pinned.
    Reading the corpus-wide rate instead is how the veto gets missed."""
    fires, quiet = [_state(calls=[_call()], feedback=VIOLATION)], [_state(calls=[_call()])]
    eps = {**{f"{i}_0": (fires, _lost()) for i in range(20)},
           **{f"9{i}_0": (quiet, _won()) for i in range(20)}}
    got = screens.screen_reachability(_control(eps), "constraint_violation_reported",
                                      benefit="PSR", boundary="post_execution")
    assert got["verdict"] == "VETO"
    assert got["episodes_firing_scored"] == 20


def test_reachability_refuses_a_metric_the_scorer_does_not_emit():
    with pytest.raises(ValueError, match="benefit must be one of"):
        screens.screen_reachability(_control({}), "proposed_tool_action", benefit="not_a_metric")


# ================================================================================================
# 3. the distinction the screens exist to keep -- UNSCREENABLE is not VETO
# ================================================================================================
def test_an_unresolvable_signal_is_unscreenable_not_vetoed():
    """`repeated_identical_call` is a declared FIELD, not a declared signal. A synthesized condition
    over it is screenable only once Phi-expansion registers it. Until then the screens have no
    opinion, and reporting one would be a refusal computed from a bug."""
    eps = {f"{i}_0": ([_state(calls=[_call()])], _lost()) for i in range(12)}
    with pytest.raises(screens.Unscreenable, match="no evaluator"):
        screens.screen_support(_control(eps), "repeated_identical_call")

    report = screens.screen_candidates(
        _control(eps), [{"controller_id": "c", "signal": "repeated_identical_call",
                         "boundary": "post_generation_pre_exec"}])
    assert report["results"][0]["verdict"] == "UNSCREENABLE"
    assert report["passed"] == 0
    assert report["vetoed"] == 1, "not passing is not the same as vetoed; the verdict says which"
    assert "EXPANDED_SIGNALS" in report["results"][0]["detail"]


def test_no_evaluable_state_is_unscreenable_rather_than_support_zero():
    """A systematic evaluation failure produces the same 0 a rare condition does. Only one of those
    is a statement about the candidate."""
    control = _control({"1_0": ([_state(boundary="post_execution", calls=[_call()])], _lost())})
    with pytest.raises(screens.Unscreenable, match="evaluable at 0"):
        screens.screen_support(control, "proposed_tool_action",
                               boundary="post_generation_pre_exec")


def test_a_candidate_naming_no_signal_is_skipped_not_screened():
    report = screens.screen_candidates(_control({}), [{"controller_id": "c", "signal": None}])
    assert report["results"][0]["verdict"] == "SKIP"


# ================================================================================================
# 4. the screens report, they do not select
# ================================================================================================
def test_every_candidate_is_reported_including_the_vetoed_ones():
    """A screen that silently dropped candidates would leave a reader unable to tell a vetoed
    proposal from one the search never made -- the reason `synthesis_reachable_classes()` exists."""
    fires = [_state(calls=[_call()], feedback=VIOLATION)]
    eps = {f"{i}_0": (fires, _lost()) for i in range(40)}
    cands = [{"controller_id": f"c{i}", "signal": "constraint_violation_reported",
              "boundary": "post_execution"} for i in range(3)]
    report = screens.screen_candidates(_control(eps), cands, benefit="PSR")
    assert len(report["results"]) == 3
    assert report["passed"] == 0 and report["vetoed"] == 3
    assert all(r["vetoed_by"] == ["reachability"] for r in report["results"])


def test_the_screens_preserve_candidate_order_and_add_no_ranking():
    """`cycle1_propose.py`'s rule: a screen may not choose WHERE to intervene, WHAT decides, or HOW
    to act. Reordering would be doing that under another name."""
    fires = [_state(calls=[_call()], feedback=VIOLATION)]
    eps = {f"{i}_0": (fires, _won() if i < 5 else _lost()) for i in range(40)}
    ids = ["z", "a", "m"]
    report = screens.screen_candidates(
        _control(eps),
        [{"controller_id": i, "signal": "constraint_violation_reported",
          "boundary": "post_execution"} for i in ids], benefit="SR")
    assert [r["controller_id"] for r in report["results"]] == ids
    assert all("rank" not in r and "score" not in r for r in report["results"])


# ================================================================================================
# 5. against the frozen control, if it is present
# ================================================================================================
CONTROL_DIR = CCTU / "results" / "granite" / "train_baseline"
CANDIDATES = REPO / "rounds" / "CCTU_CYCLE1" / "candidates"


@pytest.mark.skipif(not (CONTROL_DIR / "response.jsonl").exists() or not CANDIDATES.is_dir(),
                    reason="the granite train control is a local run artifact, not committed")
def test_the_screens_reproduce_cycle_1_s_verdicts_from_the_control_alone(tmp_path):
    """The claim this module is for: both vetoes were available BEFORE the six arms ran.

    Three vetoed on reachability, three on support, and the split must match the reasons
    `rounds/CCTU_CYCLE1/RESULT.md` recorded after spending the compute.

    RUN AS A SUBPROCESS, and that is not test hygiene for its own sake. `load_control` re-runs
    `cycle1_propose.assert_no_oracle_leak`, which refuses to proceed when an anchor fixture is in
    `sys.modules` -- and in a full pytest session another test has legitimately imported
    `fixtures.replay_anchors` by the time this one runs. Weakening the guard to make a test pass would
    remove the check that exists because "the fixture reached the runtime through an import nobody
    read as an import of the answer key". A fresh interpreter is the condition a real screening pass
    actually runs under, so this asserts against that instead, and covers the CLI's exit code while
    it is there.
    """
    import json
    import subprocess

    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(CCTU / "cctu_screens.py"),
         "--run", str(CONTROL_DIR), "--candidates", str(CANDIDATES),
         "--benefit", "PSR", "--json", str(out)],
        cwd=str(CCTU), capture_output=True, text=True, check=False)
    assert proc.returncode == 2, (
        f"a screening pass that refuses every candidate must exit non-zero.\n"
        f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}")
    report = json.loads(out.read_text())

    assert report["candidates"] == 6
    assert report["passed"] == 0, "all six failed in the real round; the screens must agree"
    by_screen = [r["vetoed_by"] for r in report["results"]]
    # MEMBERSHIP, not equality. Three arms must be refused for the reachability reason and three for
    # the support reason; a third screen agreeing with one of them is not a regression, and pinning
    # exact lists would make every future screen a test failure.
    assert sum(1 for v in by_screen if "reachability" in v) == 3
    assert sum(1 for v in by_screen if "support" in v) == 3
    assert all(v for v in by_screen), "every candidate must be refused by at least one term"
    # The reachability veto must be the STRONG form: PSR is 1 in none of the firing episodes.
    reach = [r["reachability"] for r in report["results"]
             if r["signal"] == "constraint_violation_reported"]
    assert all(x["benefit_already_1"] == 0 and x["episodes_firing"] > 150 for x in reach)
    # And the support veto must be the single-digit class, not a near miss.
    sup = [r["support"] for r in report["results"] if r["signal"] == "tool_execution_error"]
    assert all(x["episodes_firing"] < 10 for x in sup)


# ================================================================================================
# 6. the redecide budget, through the REAL episode loop
# ================================================================================================
#
# `tests/test_cctu_middleware.py` pins the budget where it is enforced. This drives the runner, which
# is where the unbounded loop actually lived: `response_generator.py`'s l2 branch sets
# `redecide = True`, breaks, and `continue`s the episode loop WITHOUT charging a round and WITHOUT
# incrementing `times`, so nothing there bounds the cycle.
#
# The replay client cannot reproduce it -- each regeneration consumes the next recorded turn, so the
# controller sees a new proposal every time and never re-triggers on one turn. A model that repeats
# itself does re-trigger, and that is not hypothetical: on the granite train control 60% of episodes
# contain a repeated identical call and 1349 turns carry one. So the client here returns the SAME
# proposal forever, which is the real failure mode reduced to its essentials.
class _StuckClient:
    """Always proposes the identical call. Counts generations so the bound is observable."""

    needs_episode_id = False

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, **_kw):
        self.calls += 1
        if self.calls > 400:                     # a guard on the TEST, so a regression fails loudly
            raise AssertionError("the redecide loop is unbounded: 400 generations on one episode")
        return {"choices": [{"message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "0", "type": "function",
                            "function": {"name": "t", "arguments": "{}"}}]}}]}


def _runner_args(middleware, **kw):
    import argparse as _ap
    d = dict(client=_StuckClient(), gen_kwargs={}, use_vllm=False, max_retries=3,
             middleware=middleware, analyze_tool_calls=False, input_dir=str(CCTU / "data"),
             allow_text_tool_calls=True)
    d.update(kw)
    return _ap.Namespace(**d)


@pytest.mark.skipif(not (CCTU / "data" / "input_data_train.jsonl").exists(),
                    reason="corpus not present")
def test_the_episode_loop_terminates_when_a_controller_always_redecides():
    """With `one_shot=False` the ONLY thing bounding regeneration is the redecide budget."""
    import json

    import cctu_middleware as mwmod
    import response_generator as rg

    sample = json.loads((CCTU / "data" / "input_data_train.jsonl").read_text().splitlines()[0])
    spec = mwmod.ControllerSpec(
        controller_id="always", boundary="post_generation_pre_exec", action="reprompt",
        signal="proposed_tool_action", theta={"text": "reconsider"})
    m = mwmod.AnchorOptMiddleware([spec], one_shot=False)
    try:
        args = _runner_args(m)
        case_id, messages, _analysis = rg.sample_process(dict(sample, id="50_0"), args)
        assert case_id == "50_0"
        # It TERMINATED, which is the property under test. Without the budget this never returns.
        t = m.telemetry_snapshot()
        assert t["interventions_executed"] > 0, "the arm must actually have intervened"
        assert t["redecide_budget_declines"] > 0, (
            "every turn after the first redecide must be declined; 0 declines means the budget "
            "never engaged and the loop was bounded by something else")
        assert args.client.calls < 400
    finally:
        m.reset()


@pytest.mark.skipif(not (CCTU / "data" / "input_data_train.jsonl").exists(),
                    reason="corpus not present")
def test_a_zero_budget_lets_the_episode_run_without_any_regeneration():
    """The cell wired and disabled: no redecide is granted, so generations equal turns."""
    import json

    import cctu_middleware as mwmod
    import response_generator as rg

    sample = json.loads((CCTU / "data" / "input_data_train.jsonl").read_text().splitlines()[0])
    spec = mwmod.ControllerSpec(
        controller_id="always", boundary="post_generation_pre_exec", action="reprompt",
        signal="proposed_tool_action", theta={"text": "reconsider"})
    m = mwmod.AnchorOptMiddleware([spec], one_shot=False, redecide_budget=0)
    try:
        args = _runner_args(m)
        rg.sample_process(dict(sample, id="50_0"), args)
        t = m.telemetry_snapshot()
        assert t["interventions_executed"] == 0
        assert t["redecide_budget_declines"] == t["signal_firings"] > 0
    finally:
        m.reset()


# ================================================================================================
# 7. a SYNTHESIZED candidate survives the process that invented it
# ================================================================================================
#
# This is the defect that made cycle 2 unrunnable. `spec_from` wrote `signal=<synthesized name>`;
# that validates in the proposer, where expansion has the name in `EXPANDED_SIGNALS`, and
# `ControllerSpec.validate` refuses it anywhere else. 312 of 330 granite candidates and 300 of 318
# qwen candidates were written that way, and the screens could only report them UNSCREENABLE.
_SPEC_LOADS = """
import json, sys
sys.path.insert(0, {cctu!r})
import cctu_middleware as mw
specs = mw.ControllerSpec.load(sys.argv[1])
for s in specs:
    s.validate()
c = mw.InstalledController(specs[0])
print(json.dumps({{"name": c.name,
                   "fires_on_hit": c.fires_on(json.loads(sys.argv[2])),
                   "fires_on_miss": c.fires_on(json.loads(sys.argv[3]))}}))
"""


def _fresh_process_check(spec_path, hit, miss):
    """Load, validate and FIRE the spec in an interpreter that never saw the expansion."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c", _SPEC_LOADS.format(cctu=str(CCTU)),
         str(spec_path), json.dumps(hit), json.dumps(miss)],
        cwd=str(CCTU), capture_output=True, text=True, check=False)
    return proc


import json  # noqa: E402  -- used by the helper above and the tests below


def test_a_synthesized_predicate_spec_loads_and_fires_in_a_fresh_process(tmp_path):
    import cctu_middleware as mwmod

    spec = mwmod.ControllerSpec(
        controller_id="c2_gate_last_round", boundary="post_generation_pre_exec", action="reprompt",
        theta={"text": "answer now"},
        predicate={"name": "rounds_remaining_lt_1p0", "field": "rounds_remaining",
                   "op": "lt", "value": 1.0},
        provenance="test")
    path = spec.write(tmp_path / "c2.json")

    proc = _fresh_process_check(path, {"rounds_remaining": 0}, {"rounds_remaining": 5})
    assert proc.returncode == 0, f"stdout:{proc.stdout}\nstderr:{proc.stderr}"
    got = json.loads(proc.stdout)
    assert got["fires_on_hit"] is True
    assert got["fires_on_miss"] is False
    assert "rounds_remaining_lt_1p0" in got["name"], "phi_name must report the condition, not the id"


def test_the_same_condition_written_as_a_bare_name_is_refused(tmp_path):
    """The old shape, pinned as a failure so the fix cannot silently regress.

    `ControllerSpec` is constructed directly here rather than through `spec_from`, because `spec_from`
    now refuses to produce this.
    """
    import cctu_middleware as mwmod

    spec = mwmod.ControllerSpec(
        controller_id="c2_bare_name", boundary="post_generation_pre_exec", action="reprompt",
        theta={"text": "answer now"}, signal="rounds_remaining_lt_1p0", provenance="test")
    path = spec.write(tmp_path / "bare.json")

    proc = _fresh_process_check(path, {"rounds_remaining": 0}, {"rounds_remaining": 5})
    assert proc.returncode != 0
    assert "not declared" in (proc.stderr + proc.stdout)


def test_spec_from_refuses_a_synthesized_signal_with_no_persistable_expression():
    """A signal that fires this session and cannot be written down is a real outcome. It must be
    refused at PROPOSE time, not discovered as an arm that will not start."""
    import cctu_adapter as runtime
    import cycle1_propose as propose

    class _Inst:
        operator = "reprompt"
        action = "reprompt"
        variant = "state_remaining_budget"
        eta = {"instruction": "x", "retry_budget": 1}
        detail = ""

    class _Cand:
        boundary = "post_generation_pre_exec"
        signal = "a_signal_with_no_expression"
        instantiated = _Inst()

    runtime.install_signal("a_signal_with_no_expression", lambda st: True,
                           boundary="post_generation_pre_exec", provenance="test")
    try:
        assert runtime.expanded_expression("a_signal_with_no_expression") is None
        # `Unpersistable`, not `SystemExit`. An earlier version of this raised SystemExit, which is a
        # BaseException and therefore slipped past `main`'s `except Exception` -- so one unpersistable
        # candidate killed the whole proposal instead of being recorded and skipped. The named
        # exception is the correct shape and this test pins it.
        with pytest.raises(propose.Unpersistable, match="no persistable expression"):
            propose.spec_from(_Cand(), 1, "test")
    finally:
        runtime.reset_expanded_signals()


def test_an_installed_expression_is_retained_and_reset_with_the_rest():
    import cctu_adapter as runtime

    expr = {"field": "rounds_remaining", "op": "lt", "value": 1.0}
    runtime.install_signal("rr_lt_1", lambda st: True, boundary="post_execution",
                           provenance="test", expr=expr)
    try:
        assert runtime.expanded_expression("rr_lt_1") == expr
        # A COPY, so a caller cannot mutate the registry through the accessor.
        runtime.expanded_expression("rr_lt_1")["op"] = "gt"
        assert runtime.expanded_expression("rr_lt_1")["op"] == "lt"
    finally:
        runtime.reset_expanded_signals()
    assert runtime.expanded_expression("rr_lt_1") is None


def test_the_screens_can_screen_a_predicate_spec(tmp_path):
    """Before the expression was persisted there was only a name, and a name is unresolvable in a
    process that did not synthesize it -- so every expanded candidate came back UNSCREENABLE."""
    import cctu_middleware as mwmod

    spec = mwmod.ControllerSpec(
        controller_id="c2_gate", boundary="post_generation_pre_exec", action="reprompt",
        theta={"text": "answer now"},
        predicate={"name": "n_tool_calls_gt_1p0", "field": "n_tool_calls", "op": "gt",
                   "value": 1.0},
        provenance="test")
    spec.write(tmp_path / "c2_gate.json")

    rows = screens._candidate_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["signal"] == "n_tool_calls_gt_1p0", "the name comes from the predicate block"
    assert rows[0]["predicate"]["field"] == "n_tool_calls"

    many = [_state(boundary="post_generation_pre_exec", calls=[_call("0"), _call("1")])]
    one = [_state(boundary="post_generation_pre_exec", calls=[_call("0")])]
    # `_lost(acc=True)` is deliberate: gain exposure requires `acc = 1`, because `judge` makes SR a
    # conjunction with it. A fixture of pure `_lost()` has no headroom at all and the loss-exposure
    # screen correctly refuses it -- which is a fact about the fixture, not about predicate reading.
    eps = {**{f"{i}_0": (many, _won() if i < 4 else _lost(acc=True)) for i in range(12)},
           **{f"9{i}_0": (one, _lost()) for i in range(6)}}
    report = screens.screen_candidates(_control(eps), rows, benefit="SR")
    assert report["results"][0]["verdict"] == "PASS"
    assert report["results"][0]["support"]["episodes_firing"] == 12, "only the 2-call episodes"
    assert report["results"][0]["reachability"]["benefit_already_1"] == 4


# ================================================================================================
# 8. LOSS EXPOSURE -- the screen that would have saved cycle 2's five arms
# ================================================================================================
#
# Support and reachability are both about the upside. Neither notices that a condition firing
# everywhere is also exposed to every episode it could BREAK. On the granite control the loss pool
# (37 episodes at SR=1) is LARGER than the gain pool (26 at acc=1 & SR=0), so the five arms that
# triggered on `proposes_tool_call` were betting 26 against 37 and came in at 8/10, 4/12, 5/4.
def test_loss_exposure_vetoes_a_condition_exposed_to_more_losses_than_gains():
    fires = [_state(boundary="post_generation_pre_exec", calls=[_call()])]
    eps = {**{f"{i}_0": (fires, _won()) for i in range(37)},
           **{f"9{i}_0": (fires, _lost(acc=True)) for i in range(26)}}
    got = screens.screen_loss_exposure(_control(eps), "proposed_tool_action", benefit="SR",
                                       boundary="post_generation_pre_exec")
    assert got["verdict"] == "VETO"
    assert got["loss_exposure"] == 37 and got["gain_exposure"] == 26
    assert "sank cycle 2" in got["detail"]


def test_loss_exposure_passes_a_condition_that_never_fires_where_it_could_break_something():
    """The property to design for, and it is reachable rather than aspirational: an episode that
    scores SR ends with a final answer, so it never spends its last round on a tool call."""
    hit = [_state(boundary="post_generation_pre_exec", calls=[_call()])]
    miss = [_state(boundary="post_generation_pre_exec", content="done")]
    eps = {**{f"{i}_0": (miss, _won()) for i in range(37)},
           **{f"9{i}_0": (hit, _lost(acc=True)) for i in range(18)}}
    got = screens.screen_loss_exposure(_control(eps), "proposed_tool_action", benefit="SR",
                                       boundary="post_generation_pre_exec")
    assert got["verdict"] == "PASS"
    assert got["loss_exposure"] == 0 and got["gain_exposure"] == 18


def test_gain_exposure_requires_acc_for_SR_and_PSR():
    """`judge` makes both a conjunction with `acc`, so an episode that never retrieved the answer
    cannot gain SR by any constraint fix and is not headroom."""
    fires = [_state(boundary="post_generation_pre_exec", calls=[_call()])]
    eps = {f"{i}_0": (fires, _lost()) for i in range(20)}          # acc=0 everywhere
    got = screens.screen_loss_exposure(_control(eps), "proposed_tool_action", benefit="SR",
                                       boundary="post_generation_pre_exec")
    assert got["gain_exposure"] == 0
    eps2 = {f"{i}_0": (fires, _lost(acc=True)) for i in range(20)}
    assert screens.screen_loss_exposure(_control(eps2), "proposed_tool_action", benefit="SR",
                                       boundary="post_generation_pre_exec")["gain_exposure"] == 20


def test_reachability_no_longer_vetoes_a_PREVENTIVE_trigger():
    """The correction cycle 2 forced. Before it, this function vetoed the one candidate on the corpus
    with zero loss exposure, and passed the five arms that fired in all 37 succeeding episodes."""
    fires = [_state(boundary="post_generation_pre_exec", calls=[_call()])]
    eps = {f"{i}_0": (fires, _lost(acc=True)) for i in range(18)}   # benefit 0 everywhere it fires
    gate = screens.screen_reachability(_control(eps), "proposed_tool_action", benefit="SR",
                                      boundary="post_generation_pre_exec")
    assert gate["verdict"] == "PASS"
    assert gate["pinned_after_trigger"] is False
    assert "BEFORE the validator" in gate["detail"]


def test_reachability_still_vetoes_a_metric_unrecoverable_after_the_validator():
    """The cycle 1 case must keep failing: `PSR = acc AND has_if_error == 0` over every message, so a
    violation-triggered arm has no attributable PSR headroom."""
    fires = [_state(boundary="post_execution", calls=[_call()], feedback=VIOLATION)]
    eps = {f"{i}_0": (fires, _lost()) for i in range(40)}
    after = screens.screen_reachability(_control(eps), "constraint_violation_reported",
                                       benefit="PSR", boundary="post_execution")
    assert after["verdict"] == "VETO"
    assert after["pinned_after_trigger"] is True
    assert "PINNED under the trigger" in after["detail"]


def test_all_three_screens_run_and_are_reported_per_candidate():
    fires = [_state(boundary="post_generation_pre_exec", calls=[_call()])]
    eps = {**{f"{i}_0": (fires, _won()) for i in range(37)},
           **{f"9{i}_0": (fires, _lost(acc=True)) for i in range(26)}}
    report = screens.screen_candidates(
        _control(eps), [{"controller_id": "c", "signal": "proposed_tool_action",
                         "boundary": "post_generation_pre_exec"}], benefit="SR")
    row = report["results"][0]
    assert set(("support", "reachability", "loss_exposure")) <= set(row)
    assert row["vetoed_by"] == ["loss_exposure"], "only the new term should refuse this one"


# ================================================================================================
# 9. REMOVABLE MASS -- the only one of the four that screens a COUNT benefit
# ================================================================================================
#
# The other three partition episodes by a BINARY metric, and the count risk does not live in that
# partition. Measured on the granite control: of arm 121's +1130 violations, +1050 landed in `PSR = 0`
# episodes and only +80 in `PSR = 1`; of arm 126's -548, -556 was in `PSR = 0`. A loss-exposure screen
# against PSR sees the 28-episode pool and misses where the count actually moves.
def _with_counts(episodes):
    """`_control`, with the per-class violation counts a count screen reads."""
    c = _control({k: (v[0], v[1]) for k, v in episodes.items()})
    for k, (_states, _outcome, counts) in episodes.items():
        c["detail"][k]["violations"] = dict(counts)
        c["detail"][k]["n_violations"] = sum(counts.values())
    return c


def test_removable_mass_vetoes_a_condition_that_reaches_none_of_the_target_classes():
    """The count analogue of a pinned metric: it cannot reduce these classes whatever it does."""
    fires = [_state(boundary="post_generation_pre_exec", calls=[_call()])]
    eps = {f"{i}_0": (fires, _lost(), {"tool_parallel": 9}) for i in range(12)}
    got = screens.screen_removable_mass(_with_counts(eps), "proposed_tool_action",
                                        classes=("max_calls_per_tool",),
                                        boundary="post_generation_pre_exec")
    assert got["verdict"] == "VETO"
    assert got["removable_mass"] == 0
    assert "cannot reduce these classes" in got["detail"]


def test_removable_mass_vetoes_more_places_to_add_than_to_remove():
    """Firing mostly where the target classes are absent is the count analogue of loss exposure."""
    fires = [_state(boundary="post_generation_pre_exec", calls=[_call()])]
    eps = {**{f"{i}_0": (fires, _lost(), {"max_calls_per_tool": 5}) for i in range(4)},
           **{f"9{i}_0": (fires, _lost(), {}) for i in range(9)}}
    got = screens.screen_removable_mass(_with_counts(eps), "proposed_tool_action",
                                        classes=("max_calls_per_tool",),
                                        boundary="post_generation_pre_exec")
    assert got["verdict"] == "VETO"
    assert got["episodes_with_mass"] == 4 and got["pure_downside_episodes"] == 9


def test_removable_mass_passes_a_condition_concentrated_on_the_target_mass():
    hit = [_state(boundary="post_generation_pre_exec", calls=[_call()])]
    miss = [_state(boundary="post_generation_pre_exec", content="done")]
    eps = {**{f"{i}_0": (hit, _lost(), {"max_calls_per_tool": 20}) for i in range(8)},
           **{f"9{i}_0": (miss, _lost(), {"max_calls_per_tool": 1}) for i in range(5)}}
    got = screens.screen_removable_mass(_with_counts(eps), "proposed_tool_action",
                                        classes=("max_calls_per_tool",),
                                        boundary="post_generation_pre_exec")
    assert got["verdict"] == "PASS"
    assert got["removable_mass"] == 160 and got["corpus_mass"] == 165
    assert got["pure_downside_episodes"] == 0


def test_removable_mass_sums_across_the_declared_classes_only():
    fires = [_state(boundary="post_generation_pre_exec", calls=[_call()])]
    eps = {f"{i}_0": (fires, _lost(), {"max_calls_per_tool": 3, "max_call_times": 4,
                                       "tool_parallel": 99}) for i in range(10)}
    got = screens.screen_removable_mass(_with_counts(eps), "proposed_tool_action",
                                        classes=("max_calls_per_tool", "max_call_times"),
                                        boundary="post_generation_pre_exec")
    assert got["removable_mass"] == 70, "tool_parallel is not a declared target here"


def test_a_count_screen_requires_named_classes():
    """Guessing the target would be guessing the round's benefit metric."""
    with pytest.raises(ValueError, match="at least one target class"):
        screens.screen_removable_mass(_control({}), "proposed_tool_action", classes=())


def test_a_control_without_the_count_fields_is_unscreenable_not_vetoed():
    """Re-scoring is required, and it needs no re-run -- saying so is the difference between a fixable
    gap and a refusal."""
    fires = [_state(boundary="post_generation_pre_exec", calls=[_call()])]
    plain = _control({f"{i}_0": (fires, _lost()) for i in range(10)})
    with pytest.raises(screens.Unscreenable, match="no `violations` field"):
        screens.screen_removable_mass(plain, "proposed_tool_action",
                                      classes=("max_calls_per_tool",),
                                      boundary="post_generation_pre_exec")


def test_the_count_screen_runs_only_when_the_round_declares_classes():
    fires = [_state(boundary="post_generation_pre_exec", calls=[_call()])]
    eps = {f"{i}_0": (fires, _lost(acc=True), {"max_calls_per_tool": 5}) for i in range(12)}
    cand = [{"controller_id": "c", "signal": "proposed_tool_action",
             "boundary": "post_generation_pre_exec"}]
    without = screens.screen_candidates(_with_counts(eps), cand, benefit="SR")
    assert "removable_mass" not in without["results"][0]
    with_ = screens.screen_candidates(_with_counts(eps), cand, benefit="SR",
                                      count_classes=("max_calls_per_tool",))
    assert with_["results"][0]["removable_mass"]["removable_mass"] == 60
