"""Every advertised tau-bench intervention, proven by BEHAVIOUR against the real mechanism.

"Telemetry says it fired" is not evidence. Each probe runs the real `ControlledLLMAgent` with the
controller ABSENT and PRESENT and asserts the two runs differ in the way the action claims -- comparing
what was dispatched and what the model was shown, never a firing flag. A telemetry-only mechanism fails
here, and must: a firing flag beside an unchanged trajectory is the inert-action defect.

The model is a deterministic script, so these need no GPU, no network and no corpus -- but they DO need
the tau-bench checkout for the real agent base class, the real Tool objects and the real message types.
They skip with a reason when it is absent.

TWO KINDS OF CLAIM, kept apart deliberately:

  * PARAMETER DELIVERY -- the eta value is present in what was handed to the model / returned to the
    orchestrator. Model-independent, and the assertion that would have caught BFCL's upstream reprompt
    writing its instruction to telemetry and never injecting it.
  * OBSERVABLE EFFECT -- the run changes. Needs a model that reads the instruction, which the script
    below does. A real model may or may not comply; that is a measurement question, not a contract one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "tau2"))
sys.path.insert(0, str(REPO / "tests"))

tau2 = pytest.importorskip(
    "tau2", reason="tau-bench checkout not importable: run with the tau2 venv and "
                   "PYTHONPATH=<anchoropt> (see docs/TAU2_ADAPTER.md)")

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage  # noqa: E402
from tau2.registry import registry                                          # noqa: E402

from test_executor_behavioral_contract import BoundaryProbe, run_boundary_contract  # noqa: E402

import tau2_mechanism as MECH                                               # noqa: E402
from tau2_fields import AFTER, GATE, TURN_START                             # noqa: E402
from tau2_capabilities import capability as ADAPTER_CAP                      # noqa: E402
from tau2_mechanism import Controller, ControlledLLMAgent, FiringSink       # noqa: E402

DOMAIN = "airline"
MUTATING_CALL = "book_reservation"
READ_CALL = "get_reservation_details"
REPROMPT_TEXT = "Verify the reservation with a read before changing it."
OTHER_REPROMPT_TEXT = "Re-read the policy section on cancellations first."


# ================================================================================ the scripted model
class ScriptedModel:
    """A deterministic stand-in for the agent's LLM, and the ONLY source of nondeterminism removed.

    It proposes the mutating call, and re-plans in the two ways a real agent would:

      * an instruction mentioning a read  -> it performs the read instead (so a REPROMPT that reached
        the model has an observable consequence);
      * a `NOT EXECUTED.` notice with no instruction -> it falls back to the read on its own, which is
        the SUPPRESS causal path: re-planning from the ABSENCE of its own call, with nothing suggested.

    Both branches are needed. If the script ignored the instruction, reprompt would look inert; if it
    ignored the notice, suppression could never be observed to work.
    """

    def __init__(self) -> None:
        self.prompts: list[list[dict]] = []

    def __call__(self, model=None, messages=(), tools=None, call_name=None, **kw):
        flat = [{"role": getattr(m, "role", ""), "content": str(getattr(m, "content", "") or "")}
                for m in messages]
        self.prompts.append(flat)
        text = " ".join(m["content"] for m in flat)
        if "read" in text.lower() and "verify" in text.lower():
            return AssistantMessage(
                role="assistant", content=None,
                tool_calls=[ToolCall(id=f"c{len(self.prompts)}", name=READ_CALL,
                                     arguments={"reservation_id": "ABC123"},
                                     requestor="assistant")])
        if MECH.SUPPRESS_NOTICE in text:
            return AssistantMessage(
                role="assistant", content=None,
                tool_calls=[ToolCall(id=f"c{len(self.prompts)}", name=READ_CALL,
                                     arguments={"reservation_id": "ABC123"},
                                     requestor="assistant")])
        return AssistantMessage(
            role="assistant", content=None,
            tool_calls=[ToolCall(id=f"c{len(self.prompts)}", name=MUTATING_CALL,
                                 arguments={"reservation_id": "ABC123"},
                                 requestor="assistant")])


@pytest.fixture
def tools():
    return registry.get_env_constructor(DOMAIN)().get_tools()


@pytest.fixture
def scripted(monkeypatch):
    """Patch the name the MECHANISM bound, not the module it came from.

    `tau2_mechanism` does `from tau2.utils.llm_utils import generate`, so the mechanism holds its own
    reference and patching `tau2.utils.llm_utils.generate` would leave it untouched -- the probe would
    then silently call the real gateway.
    """
    model = ScriptedModel()
    monkeypatch.setattr(MECH, "generate", model)
    return model


def _run(tools, controller, inbound=None):
    """One agent turn through the REAL mechanism. Returns an observable record."""
    sink = FiringSink()
    agent = ControlledLLMAgent(tools=tools, domain_policy="POLICY", llm="scripted", llm_args={},
                              controller=controller, sink=sink, case_id="case-1", max_replans=3)
    state = agent.get_init_state()
    msg = inbound if inbound is not None else UserMessage(
        role="user", content="Please change my reservation ABC123.")
    returned, state = agent.generate_next_message(msg, state)
    return {
        "dispatched": [{"name": tc.name, "arguments": dict(tc.arguments or {})}
                       for tc in (returned.tool_calls or ())],
        "content": str(returned.content or ""),
        "prompts": agent.observed_prompts,
        "sink": sink.snapshot(),
    }


def _fires_always(_state):
    return True


def _fires_on_mutating(state):
    return bool(state.get("proposed_tool_mutates_state"))


# ================================================================ post_generation_pre_exec / SUPPRESS
def test_suppress_removes_the_call_from_what_the_orchestrator_receives(tools, scripted):
    controller = Controller(boundary=GATE, action="suppress", predicate=_fires_on_mutating,
                            eta={"suppressed_operation": "the proposed call", "retry_budget": 1},
                            variant="cancel_proposed")
    probe = BoundaryProbe(
        name="tau2 suppress at the commitment gate",
        boundary="post_generation_pre_exec", action="suppress",
        run=lambda c: _run(tools, c)["dispatched"],
        controller=controller,
        effect_claim="the mutating call is absent from what the agent returned to the orchestrator",
        asserts_effect=lambda control, treated: (
            any(c["name"] == MUTATING_CALL for c in control)
            and all(c["name"] != MUTATING_CALL for c in treated)),
    )
    ok, detail = run_boundary_contract(probe)
    assert ok, detail


def test_suppress_supplies_no_instruction_so_it_is_not_secretly_a_reprompt(tools, scripted):
    """The toy host once read `hint or "Archive it instead"`, making suppression a reprompt.

    Two suppress arms differing only in `instruction` must show the model the SAME thing, and the
    reprompt's text must appear nowhere in what the model saw.
    """
    def run_with(instruction):
        c = Controller(boundary=GATE, action="suppress", predicate=_fires_on_mutating,
                       eta={"retry_budget": 1, "instruction": instruction},
                       variant="cancel_proposed")
        return _run(tools, c)

    a, b = run_with(REPROMPT_TEXT), run_with(OTHER_REPROMPT_TEXT)
    assert a["prompts"] == b["prompts"], (
        "the suppress branch read `instruction`, which makes it a reprompt")
    seen = " ".join(m["content"] for p in a["prompts"] for m in p)
    assert REPROMPT_TEXT not in seen
    assert MECH.SUPPRESS_NOTICE in seen, "the agent was not told its call did not run"


def test_suppress_never_lets_the_call_through_once_its_budget_is_spent(tools, scripted):
    """A suppression that eventually dispatches the call did not suppress.

    The scripted model here keeps proposing the mutating call, ignoring the absence notice.
    """
    class Stubborn(ScriptedModel):
        def __call__(self, **kw):
            super().__call__(**kw)
            return AssistantMessage(
                role="assistant", content=None,
                tool_calls=[ToolCall(id=f"s{len(self.prompts)}", name=MUTATING_CALL,
                                     arguments={"reservation_id": "ABC123"},
                                     requestor="assistant")])

    import unittest.mock as mock
    with mock.patch.object(MECH, "generate", Stubborn()):
        got = _run(tools, Controller(boundary=GATE, action="suppress",
                                    predicate=_fires_on_mutating,
                                    eta={"retry_budget": 1}, variant="cancel_proposed"))
    assert got["dispatched"] == [], got["dispatched"]
    assert got["content"], "a message with no tool calls must still carry content"


# =============================================================== post_generation_pre_exec / REPROMPT
def test_gate_reprompt_instruction_reaches_the_model(tools, scripted):
    """PARAMETER DELIVERY. The eta value must appear in what was handed to the model."""
    got = _run(tools, Controller(boundary=GATE, action="reprompt", predicate=_fires_on_mutating,
                                 eta={"instruction": REPROMPT_TEXT, "retry_budget": 1}))
    seen = " ".join(m["content"] for p in got["prompts"] for m in p)
    assert REPROMPT_TEXT in seen, "the instruction never entered the model's context"


def test_two_gate_reprompt_arms_differing_only_in_instruction_are_not_identical(tools, scripted):
    """The inert-action regression: BFCL's upstream reprompt produced byte-identical trajectories
    across 13/13 episodes because the instruction never reached the model."""
    def run_with(instruction):
        return _run(tools, Controller(boundary=GATE, action="reprompt",
                                      predicate=_fires_on_mutating,
                                      eta={"instruction": instruction, "retry_budget": 1}))

    a = run_with(REPROMPT_TEXT)
    b = run_with(OTHER_REPROMPT_TEXT)
    assert a["prompts"] != b["prompts"], "the two instructions produced identical model inputs"


def test_gate_reprompt_changes_what_is_dispatched(tools, scripted):
    controller = Controller(boundary=GATE, action="reprompt", predicate=_fires_on_mutating,
                            eta={"instruction": REPROMPT_TEXT, "retry_budget": 1})
    probe = BoundaryProbe(
        name="tau2 reprompt at the commitment gate",
        boundary="post_generation_pre_exec", action="reprompt",
        run=lambda c: _run(tools, c)["dispatched"],
        controller=controller,
        effect_claim="the agent re-plans and dispatches the read the instruction asked for",
        asserts_effect=lambda control, treated: (
            any(c["name"] == MUTATING_CALL for c in control)
            and any(c["name"] == READ_CALL for c in treated)),
    )
    ok, detail = run_boundary_contract(probe)
    assert ok, detail


# ================================================================ post_generation_pre_exec / REROUTE
def test_reroute_rewrites_the_dispatched_call_to_the_eta_destination(tools, scripted):
    controller = Controller(boundary=GATE, action="reroute", predicate=_fires_on_mutating,
                            eta={"destination": READ_CALL,
                                 "argument_mapping": "reservation_id -> reservation_id",
                                 "retry_semantics": "replace"},
                            variant=f"substitute_{READ_CALL}")
    probe = BoundaryProbe(
        name="tau2 reroute at the commitment gate",
        boundary="post_generation_pre_exec", action="reroute",
        run=lambda c: _run(tools, c)["dispatched"],
        controller=controller,
        effect_claim="the call the orchestrator receives names the destination, not the proposal",
        asserts_effect=lambda control, treated: (
            [c["name"] for c in control] == [MUTATING_CALL]
            and [c["name"] for c in treated] == [READ_CALL]),
    )
    ok, detail = run_boundary_contract(probe)
    assert ok, detail


def test_reroute_destination_parameter_reaches_the_execution_mechanism(tools, scripted):
    """Two reroute arms differing only in `destination` must dispatch different calls."""
    def run_with(dest):
        return _run(tools, Controller(boundary=GATE, action="reroute",
                                      predicate=_fires_on_mutating,
                                      eta={"destination": dest,
                                           "argument_mapping": "reservation_id -> reservation_id"}))

    a = run_with(READ_CALL)
    b = run_with("get_user_details")
    assert [c["name"] for c in a["dispatched"]] == [READ_CALL]
    # `get_user_details` requires a different argument, so the rewrite is correctly REFUSED and the
    # run is unaltered -- an arm this host cannot honour must not report a firing.
    assert [c["name"] for c in b["dispatched"]] == [MUTATING_CALL]
    assert b["sink"]["interventions_executed"] == 0, (
        "a refused reroute recorded a firing it did not perform")


def test_reroute_drops_arguments_the_destination_does_not_accept(tools, scripted):
    """The rewritten call must be one that EXECUTES, not merely one that validates.

    `list_all_airports` declares no parameters and pydantic ignores unknown ones, so forwarding the
    proposal's `reservation_id` validated cleanly while the real dispatch returned `unexpected keyword
    argument`. This probe found that; the mechanism now carries only the destination's own parameters.
    """
    got = _run(tools, Controller(boundary=GATE, action="reroute", predicate=_fires_on_mutating,
                                 eta={"destination": "list_all_airports",
                                      "argument_mapping": "reservation_id -> reservation_id"}))
    assert [c["name"] for c in got["dispatched"]] == ["list_all_airports"]
    assert got["dispatched"][0]["arguments"] == {}, (
        "a forwarded argument the destination rejects would error at dispatch")
    assert got["sink"]["interventions_executed"] == 1


def test_reroute_refuses_a_destination_that_does_not_exist(tools, scripted):
    got = _run(tools, Controller(boundary=GATE, action="reroute", predicate=_fires_on_mutating,
                                 eta={"destination": "no_such_tool"}))
    assert [c["name"] for c in got["dispatched"]] == [MUTATING_CALL]
    assert got["sink"]["interventions_executed"] == 0


# ================================================================ pre_generation / REPROMPT
def test_pre_generation_note_reaches_the_first_generation(tools, scripted):
    got = _run(tools, Controller(boundary=TURN_START, action="reprompt", predicate=_fires_always,
                                 eta={"instruction": REPROMPT_TEXT}))
    assert got["prompts"], "no generation was recorded"
    seen = " ".join(m["content"] for m in got["prompts"][0])
    assert REPROMPT_TEXT in seen, "the note did not reach the FIRST generation of the turn"
    assert got["sink"]["by_site"].get("pre_generation/reprompt") == 1


def test_pre_generation_note_is_not_persisted_into_the_returned_message(tools, scripted):
    """It reaches the generation and nothing else: the trajectory must not carry it."""
    got = _run(tools, Controller(boundary=TURN_START, action="reprompt", predicate=_fires_always,
                                 eta={"instruction": REPROMPT_TEXT}))
    assert REPROMPT_TEXT not in got["content"]


# ================================================================ post_execution / REPROMPT
def test_post_execution_guidance_reaches_the_model_without_touching_the_recorded_result(tools, scripted):
    """The recorded ToolMessage is the SAME OBJECT the orchestrator holds and already recorded.

    Mutating it corrupts the trajectory the evaluator replays -- the defect that turned 41/114 telecom
    simulations into infrastructure errors. So the guidance must be a separate message and the inbound
    object must come back unchanged.
    """
    inbound = ToolMessage(id="c1", role="tool", content="Error: Reservation ABC123 not found",
                          requestor="assistant", error=True)
    original = inbound.content
    got = _run(tools, Controller(boundary=AFTER, action="reprompt", predicate=_fires_always,
                                 eta={"instruction": REPROMPT_TEXT}), inbound=inbound)
    seen = " ".join(m["content"] for p in got["prompts"] for m in p)
    assert REPROMPT_TEXT in seen, "the guidance never reached the model"
    assert inbound.content == original, "the recorded ToolMessage was modified in place"
    assert got["sink"]["by_site"].get("post_execution/reprompt") == 1


def test_post_execution_reroute_is_declared_disabled_and_has_no_mechanism(tools, scripted):
    """The disabled cell must not quietly do something. Installing it changes nothing."""
    inbound = ToolMessage(id="c1", role="tool", content="Error: nope", requestor="assistant",
                          error=True)
    control = _run(tools, None, inbound=inbound)
    treated = _run(tools, Controller(boundary=AFTER, action="reroute", predicate=_fires_always,
                                     eta={"destination": READ_CALL}), inbound=inbound)
    assert control["dispatched"] == treated["dispatched"]
    assert treated["sink"]["interventions_executed"] == 0


# ================================================================ firings come from execution
def test_firings_are_counted_at_the_mechanism_not_from_registration(tools, scripted):
    """A controller that never matches must report zero, and a registry cannot know that."""
    never = Controller(boundary=GATE, action="suppress", predicate=lambda _s: False,
                       eta={"retry_budget": 1})
    got = _run(tools, never)
    assert got["sink"]["interventions_executed"] == 0
    assert got["sink"]["predicate_matches"] == 0
    assert got["sink"]["cases_fired"] == []
    assert [c["name"] for c in got["dispatched"]] == [MUTATING_CALL]


def test_a_matched_but_budget_blocked_intervention_is_not_counted_as_engagement(tools, scripted):
    """`predicate_matches` and `interventions_executed` are different facts.

    Counting a match that changed nothing would overstate engagement, and `train_objective` uses
    executions to decide whether an arm caused its delta.
    """
    class Stubborn(ScriptedModel):
        def __call__(self, **kw):
            super().__call__(**kw)
            return AssistantMessage(
                role="assistant", content="ok",
                tool_calls=[ToolCall(id=f"s{len(self.prompts)}", name=MUTATING_CALL,
                                     arguments={"reservation_id": "ABC123"},
                                     requestor="assistant")])

    import unittest.mock as mock
    with mock.patch.object(MECH, "generate", Stubborn()):
        got = _run(tools, Controller(boundary=GATE, action="suppress",
                                     predicate=_fires_on_mutating, eta={"retry_budget": 1}))
    snap = got["sink"]
    assert snap["predicate_matches"] > snap["interventions_executed"], snap


def test_firings_are_attributed_to_the_case_the_mechanism_ran_in(tools, scripted):
    got = _run(tools, Controller(boundary=GATE, action="reprompt", predicate=_fires_on_mutating,
                                 eta={"instruction": REPROMPT_TEXT, "retry_budget": 1}))
    assert got["sink"]["cases_fired"] == ["case-1"]
    assert got["sink"]["by_case"] == {"case-1": 1}


# ================================================================ every enabled cell has a probe
def test_every_enabled_capability_is_covered_by_a_behavioural_probe():
    """An enabled cell with no probe is a declaration nobody checked."""
    from tau2_capabilities import CAPABILITIES
    enabled = {f"{b}/{a}" for (b, a), c in CAPABILITIES.items() if c.is_bound and c.is_enabled}
    source = Path(__file__).read_text()
    covered = {cell for cell in enabled
               if cell.split("/")[1] in source and cell.split("/")[0].split("_")[0] in source}
    # Named explicitly so adding a cell without a probe fails loudly rather than passing by substring.
    probed = {"pre_generation/reprompt", "post_generation_pre_exec/reprompt",
              "post_generation_pre_exec/suppress", "post_generation_pre_exec/reroute",
              "post_execution/reprompt"}
    assert enabled == probed, f"unprobed: {enabled - probed}; stale: {probed - enabled}"
    assert covered == enabled


# ================================================================ the rewrite actually executes
def test_a_rerouted_call_executes_cleanly_against_the_real_environment(tools, scripted):
    """The strongest form of the materializability claim: dispatch the rewritten call for real.

    A rewrite that validates but errors at dispatch would make the arm measure its own error as the
    intervention's effect.
    """
    from tau2.data_model.message import ToolCall as TC
    env = registry.get_env_constructor(DOMAIN)()
    for dest in ("list_all_airports", READ_CALL):
        got = _run(tools, Controller(boundary=GATE, action="reroute", predicate=_fires_on_mutating,
                                     eta={"destination": dest,
                                          "argument_mapping": "reservation_id -> reservation_id"}))
        call = got["dispatched"][0]
        assert call["name"] == dest
        result = env.get_response(TC(id="probe", name=call["name"],
                                     arguments=call["arguments"], requestor="assistant"))
        assert "unexpected keyword argument" not in str(result.content), result.content
        assert "missing" not in str(result.content).lower() or not result.error, result.content


# ================================================================ installation through the registry
def test_the_emitted_controller_installs_through_the_registry_and_runs_an_episode(monkeypatch):
    """Task 3(6): the controller a round emits can be INSTALLED and EXECUTED, end to end.

    A full `run_simulation` against the real Orchestrator, real Environment and real evaluator, with
    both the agent's and the user's model scripted so it needs no network. This is what proves the
    round's output is installable rather than merely well-formed.
    """
    import tau2.user.user_simulator as US
    from tau2.data_model.message import UserMessage as UM
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.simulation import run_simulation
    from tau2.user.user_simulator import UserSimulator

    from tau2_mechanism import controller_spec, make_agent_factory

    env = registry.get_env_constructor(DOMAIN)()
    task = registry.get_tasks_loader(DOMAIN)()[0]

    agent_model = ScriptedModel()
    monkeypatch.setattr(MECH, "generate", agent_model)

    turns = {"n": 0}

    def scripted_user(model=None, messages=(), tools=None, call_name=None, **kw):
        """A user that says one thing and then stops, so the episode terminates deterministically."""
        turns["n"] += 1
        if turns["n"] >= 2:
            return UM(role="user", content="###STOP###")
        return UM(role="user", content="Please change my reservation ABC123.")

    monkeypatch.setattr(US, "generate", scripted_user)

    sink = FiringSink()
    controller = Controller(boundary=GATE, action="suppress", predicate=_fires_on_mutating,
                            eta={"suppressed_operation": "the proposed call", "retry_budget": 1},
                            variant="cancel_proposed",
                            label="post_generation_pre_exec/proposes_state_change/suppress")
    spec = controller_spec(controller)
    assert spec["boundary"] == GATE and spec["action"] == "suppress"

    factory = make_agent_factory(controller, sink)
    agent = factory(tools=env.get_tools(), domain_policy=env.get_policy(), llm="scripted",
                    llm_args={}, task=task)
    assert agent.case_id == task.id, "the factory did not key the agent to its own case"

    user = UserSimulator(llm="scripted", instructions=str(task.user_scenario),
                         tools=None, llm_args={})
    orch = Orchestrator(domain=DOMAIN, agent=agent, user=user, environment=env, task=task,
                        max_steps=8, max_errors=5)
    sim = run_simulation(orch)

    assert sim is not None and sim.messages, "the episode produced no trajectory"
    assert sim.reward_info is not None, "the evaluator did not run"
    # THE POINT OF THIS TEST: the mechanism ran inside the real loop and the suppressed call is absent
    # from the RECORDED trajectory, so the evaluator's replay never sees a call that did not execute.
    snap = sink.snapshot()
    assert snap["interventions_executed"] > 0, snap
    assert snap["cases_fired"] == [task.id], snap
    dispatched = [tc.name for m in sim.messages for tc in (getattr(m, "tool_calls", None) or ())]
    assert MUTATING_CALL not in dispatched, (
        f"the withheld call reached the trajectory and would be replayed: {dispatched}")


# ================================================================ retry_budget reaches execution
def _run_turns(tools, controller, n_turns: int, inbound_factory):
    """Drive one agent across several turns, so a per-episode budget can be observed."""
    sink = FiringSink()
    agent = ControlledLLMAgent(tools=tools, domain_policy="POLICY", llm="scripted", llm_args={},
                              controller=controller, sink=sink, case_id="case-1", max_replans=3)
    state = agent.get_init_state()
    for i in range(n_turns):
        _, state = agent.generate_next_message(inbound_factory(i), state)
    return sink.snapshot()


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_pre_generation_note_is_injected_at_most_retry_budget_times(tools, scripted, budget):
    """PARAMETER DELIVERY at a cell whose only other eta is the instruction.

    An uncapped note would fire on all four turns. Core's REPROMPT contract requires `retry_budget`
    precisely because a reprompt with no bound is a prompt edit; this asserts the executor honours it.
    """
    snap = _run_turns(
        tools,
        Controller(boundary=TURN_START, action="reprompt", predicate=_fires_always,
                   eta={"instruction": REPROMPT_TEXT, "retry_budget": budget}),
        n_turns=4,
        inbound_factory=lambda i: UserMessage(role="user", content=f"turn {i}"))
    assert snap["by_site"].get("pre_generation/reprompt") == budget, snap
    assert snap["predicate_matches"] == 4, "the predicate should match every turn"
    assert snap["interventions_executed"] == budget, (
        "a budget-blocked match was counted as an execution")


@pytest.mark.parametrize("budget", [1, 2])
def test_post_execution_guidance_is_injected_at_most_retry_budget_times(tools, scripted, budget):
    snap = _run_turns(
        tools,
        Controller(boundary=AFTER, action="reprompt", predicate=_fires_always,
                   eta={"instruction": REPROMPT_TEXT, "retry_budget": budget}),
        n_turns=4,
        inbound_factory=lambda i: ToolMessage(id=f"c{i}", role="tool", content="Error: nope",
                                              requestor="assistant", error=True))
    assert snap["by_site"].get("post_execution/reprompt") == budget, snap
    assert snap["interventions_executed"] == budget


def test_gate_suppress_budget_bounds_replanning(tools, scripted):
    """At the gate, retry_budget bounds re-planning attempts within one turn."""
    class Stubborn(ScriptedModel):
        def __call__(self, **kw):
            super().__call__(**kw)
            return AssistantMessage(
                role="assistant", content="ok",
                tool_calls=[ToolCall(id=f"s{len(self.prompts)}", name=MUTATING_CALL,
                                     arguments={"reservation_id": "ABC123"},
                                     requestor="assistant")])

    import unittest.mock as mock
    seen = {}
    for budget in (1, 2):
        with mock.patch.object(MECH, "generate", Stubborn()):
            got = _run(tools, Controller(boundary=GATE, action="suppress",
                                        predicate=_fires_on_mutating,
                                        eta={"retry_budget": budget}))
        seen[budget] = got["sink"]["by_site"].get("post_generation_pre_exec/suppress", 0)
    assert seen[2] > seen[1], seen


# ================================================================ suppressing an answer commitment
def test_suppressing_an_answer_commitment_tells_the_agent_its_reply_was_not_delivered(tools):
    """A candidate calling no tool is still a consequential decision: it commits to replying.

    Without a signal the model sees its own message followed by nothing and repeats it, so the
    suppression would be inert by construction rather than by measurement. The notice must stay
    non-directive, or this becomes a reprompt.
    """
    import unittest.mock as mock

    class Talker(ScriptedModel):
        def __call__(self, **kw):
            super().__call__(**kw)
            text = " ".join(m["content"] for m in self.prompts[-1])
            if MECH.SUPPRESS_REPLY_NOTICE in text:
                return AssistantMessage(
                    role="assistant", content=None,
                    tool_calls=[ToolCall(id=f"a{len(self.prompts)}", name=READ_CALL,
                                         arguments={"reservation_id": "ABC123"},
                                         requestor="assistant")])
            return AssistantMessage(role="assistant", content="Sorry, I cannot help.",
                                    tool_calls=None)

    controller = Controller(boundary=GATE, action="suppress",
                            predicate=lambda st: bool(st.get("commits_to_reply")),
                            eta={"retry_budget": 1}, variant="cancel_proposed")
    with mock.patch.object(MECH, "generate", Talker()):
        control = _run(tools, None)
        treated = _run(tools, controller)

    assert control["content"] and not control["dispatched"], control
    assert [c["name"] for c in treated["dispatched"]] == [READ_CALL], (
        "the withheld reply produced no re-plan, so the suppression was inert")
    seen = " ".join(m["content"] for p in treated["prompts"] for m in p)
    assert MECH.SUPPRESS_REPLY_NOTICE in seen
    # Still not a reprompt: no task guidance was supplied.
    assert REPROMPT_TEXT not in seen and OTHER_REPROMPT_TEXT not in seen


# ================================================================ retry_semantics reaches execution
def test_reroute_declines_semantics_this_host_cannot_execute(tools, scripted):
    """`retry_semantics` is in `consumes`, so the code must really read it.

    This host swaps the proposed call and has no path that keeps the original and adds a second: the
    agent returns one message and the orchestrator dispatches what it contains. An arm asking for other
    semantics must leave the run unaltered and record no firing.
    """
    ok = _run(tools, Controller(boundary=GATE, action="reroute", predicate=_fires_on_mutating,
                                eta={"destination": READ_CALL, "retry_semantics": "replace",
                                     "argument_mapping": "reservation_id -> reservation_id"}))
    assert [c["name"] for c in ok["dispatched"]] == [READ_CALL]
    assert ok["sink"]["interventions_executed"] == 1

    other = _run(tools, Controller(boundary=GATE, action="reroute", predicate=_fires_on_mutating,
                                   eta={"destination": READ_CALL, "retry_semantics": "append",
                                        "argument_mapping": "reservation_id -> reservation_id"}))
    assert [c["name"] for c in other["dispatched"]] == [MUTATING_CALL], (
        "the mechanism ignored retry_semantics, so the key is inert eta")
    assert other["sink"]["interventions_executed"] == 0


def test_the_reroute_cell_fixes_the_only_semantics_it_implements(tools):
    """Declared `fixed`, so core refuses a conflicting arm at CONSTRUCTION rather than letting the
    mechanism decline at run time and the arm measure as the control."""
    from anchoropt.anchor import Action as A, IncisionPoint as I
    cap = ADAPTER_CAP(I.POST_GENERATION_PRE_EXEC, A.REROUTE)
    assert "retry_semantics" in cap.consumes
    assert dict(cap.fixed).get("retry_semantics") == MECH.REROUTE_SEMANTICS
