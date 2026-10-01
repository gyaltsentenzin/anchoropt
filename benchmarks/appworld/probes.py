"""Behavioral proof that each declared `(boundary, action)` cell in `adapter.py`'s `_CAPS` actually
runs, per `tests/test_executor_behavioral_contract.py`'s reusable `BoundaryProbe`/`run_boundary_contract`
shape (that file's own docstring invites exactly this: "A TauBench or AppWorld adapter supplies a
`BoundaryProbe` for each capability it declares").

WHY EVERY PROBE HERE RUNS FULLY OFFLINE -- NO REAL MODEL SERVER, NO REAL APPWORLD TASK
--------------------------------------------------------------------------------------------------
`AppWorldReActAgent.__init__` builds a real `LanguageModel` client and `SimplifiedReActCodeAgent`'s own
`initialize()` needs a real `AppWorld` task to render the prompt template. Neither is needed to prove
what these six mechanisms do: every one of them is a pure transformation of `self.messages` / the
generated cell string, reachable by constructing a BARE instance (`__new__`, skipping `__init__`) with
only the attributes the mechanism under test touches, then calling the REAL, UNMODIFIED
`next_execution_inputs_usage_and_status` override -- the exact method AppWorld's own loop calls -- with
`self.language_model` substituted by `_FakeLanguageModel` (canned cells, no network) and `self.logger`
by a no-op. This is the same substitution `ToyDispatchHost`/`_Ctl` make in the reference file: the
INFRASTRUCTURE is faked, the MECHANISM LOGIC under test is the real, shipped code.

This resolves what was flagged as an open question in an earlier pass over this plan (whether
`pre_generation`/`post_execution` REPROMPT and `post_generation_pre_exec` SUPPRESS/REPROMPT could only
be proven with a real model call): for the four message-injection/reroute mechanisms
(`_inject_pre_generation_instruction`, `_inject_post_execution_instruction`,
`_reroute_to_api_docs_pre_exec`, `_reroute_after_execution`) the full claim IS the message/dispatch
mutation, independent of how a real model would respond to it. For the two mechanisms that regenerate
(`_inject_pre_exec_instruction`, `_withhold_proposed_call`), a `_FakeLanguageModel` returning two
DIFFERENT canned cells proves the pop-and-regenerate machinery genuinely runs (the treated run dispatches
the SECOND canned cell; control, which never regenerates, dispatches the first) without needing a real
model to decide what the second cell should be -- that decision is the model's competence, not the
executor's, and is not what this contract is testing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from appworld_agents.code.common.usage_tracker import Usage
from appworld_agents.code.simplified.agent import ExecutionIO

from anchoropt.anchor import IncisionPoint

from tests.test_executor_behavioral_contract import BoundaryProbe, run_boundary_contract

from adapter import ADAPTER
from agent_hooks import AppWorldReActAgent
from controller import Controller

_PRE = IncisionPoint.PRE_GENERATION.value
_PG = IncisionPoint.POST_GENERATION_PRE_EXEC.value
_PE = IncisionPoint.POST_EXECUTION.value


# ================================================================================================
# FAKES -- infrastructure only, never the mechanism logic under test
# ================================================================================================

class _FakeLanguageModel:
    """Returns one canned cell per call, in order. Running out is a test bug, not a host behavior,
    so it raises loudly rather than repeating the last cell and masking a miscounted probe."""

    def __init__(self, cells: list[str]) -> None:
        self._cells = list(cells)

    def generate(self, *, messages: Any, cache_control_at: int) -> dict[str, Any]:
        if not self._cells:
            raise AssertionError("_FakeLanguageModel: probe called generate() more times than expected")
        code = self._cells.pop(0)
        return {
            "content": f"```python\n{code}\n```\n",
            "reasoning_content": "",
            "standardized_usage": Usage(),
        }


class _FakeLogger:
    def show_message(self, **kwargs: Any) -> None:
        pass


def _bare_agent(*, task_id: str = "probe_task") -> AppWorldReActAgent:
    """A minimal `AppWorldReActAgent` with none of `Agent.__init__`'s real infrastructure (no real
    `LanguageModel`, no `AppWorld` world) but every attribute the six mechanisms and the wholesale
    `next_execution_inputs_usage_and_status` override actually read."""
    agent = AppWorldReActAgent.__new__(AppWorldReActAgent)
    agent.controller = None
    agent.budget_spent = {}
    agent.fired_tasks = set()
    agent.executed = 0
    agent._consecutive_errors = 0
    agent._last_error_kind = ""
    agent._last_target_app = ""
    agent._last_target_api = ""
    agent.messages = []
    agent.step_number = 3
    agent.max_output_length = None
    agent.ignore_multiple_calls = True
    agent.full_code_regex = r"```python\n(.*?)```"
    agent.partial_code_regex = r".*```python\n(.*)"
    agent.logger = _FakeLogger()
    agent.world = SimpleNamespace(task_id=task_id)
    return agent


def _run_step(
    agent: AppWorldReActAgent, controller: Controller | None, cells: list[str],
    last_execution_outputs: list[ExecutionIO] = (),
) -> dict[str, Any]:
    """One call to the REAL wholesale override, with `cells` loaded into a fresh fake model."""
    agent.controller = controller
    agent.language_model = _FakeLanguageModel(cells)
    outputs, _usage, status = agent.next_execution_inputs_usage_and_status(list(last_execution_outputs))
    return {
        "code": outputs[0].content if outputs else "",
        "failed": status.failed,
        "messages": [dict(m) for m in agent.messages],
    }


def _eta(action_method: str, boundary: str) -> dict[str, Any]:
    """The REAL grounded eta `adapter.py` would propose for this (boundary, action) -- never a
    hand-copied duplicate that could drift from what a real round actually installs."""
    grounded = getattr(ADAPTER, action_method)("execution_failed", boundary)
    return dict(grounded[0]["eta"]) if grounded else {}


_ALWAYS_FIRES = lambda state: True  # noqa: E731 -- probes test the EXECUTOR, not the signal grammar


# ================================================================================================
# THE SIX PROBES, ONE PER `adapter.py::_CAPS` ENTRY
# ================================================================================================

def _pre_reprompt_probe() -> BoundaryProbe:
    eta = _eta("ground_reprompt", _PRE)

    def run(controller):
        agent = _bare_agent()
        return _run_step(agent, controller, ["result = apis.phone.get_contacts()"])

    return BoundaryProbe(
        name="pre_generation/reprompt", boundary=_PRE, action="reprompt", run=run,
        controller=Controller(boundary=_PRE, action="reprompt", predicate=_ALWAYS_FIRES, eta=eta),
        effect_claim="the instruction is appended to the message list before generation",
        asserts_effect=lambda c, t: (
            len(t["messages"]) == len(c["messages"]) + 1
            and eta["instruction"] in t["messages"][0]["content"]
        ),
    )


def _pg_reprompt_probe() -> BoundaryProbe:
    eta = _eta("ground_reprompt", _PG)

    def run(controller):
        agent = _bare_agent()
        return _run_step(
            agent, controller,
            ["result = apis.phone.get_kontacts()", "result = apis.phone.get_contacts()"],
        )

    return BoundaryProbe(
        name="post_generation_pre_exec/reprompt", boundary=_PG, action="reprompt", run=run,
        controller=Controller(boundary=_PG, action="reprompt", predicate=_ALWAYS_FIRES, eta=eta),
        effect_claim="the first proposed cell is discarded and a second, regenerated cell is dispatched",
        asserts_effect=lambda c, t: (
            t["code"] == "result = apis.phone.get_contacts()"
            and c["code"] == "result = apis.phone.get_kontacts()"
            and any(eta["instruction"] in m.get("content", "") for m in t["messages"])
        ),
    )


def _pg_suppress_probe() -> BoundaryProbe:
    def run(controller):
        agent = _bare_agent()
        return _run_step(
            agent, controller,
            ["result = apis.phone.get_kontacts()", "result = apis.phone.get_contacts()"],
        )

    return BoundaryProbe(
        name="post_generation_pre_exec/suppress", boundary=_PG, action="suppress", run=run,
        controller=Controller(
            boundary=_PG, action="suppress", predicate=_ALWAYS_FIRES,
            eta={"suppressed_operation": "the proposed cell"},
        ),
        effect_claim="the proposed cell is withheld and a second, regenerated cell is dispatched, "
                     "with a neutral withheld-notice -- never the arm's own instruction text",
        asserts_effect=lambda c, t: (
            t["code"] == "result = apis.phone.get_contacts()"
            and c["code"] == "result = apis.phone.get_kontacts()"
            and any("withheld" in m.get("content", "") for m in t["messages"])
        ),
    )


def _pg_reroute_probe() -> BoundaryProbe:
    def run(controller):
        agent = _bare_agent()
        return _run_step(agent, controller, ["result = apis.phone.get_contacts()"])

    return BoundaryProbe(
        name="post_generation_pre_exec/reroute", boundary=_PG, action="reroute", run=run,
        controller=Controller(boundary=_PG, action="reroute", predicate=_ALWAYS_FIRES, eta={}),
        effect_claim="the proposed cell is replaced by a computed api_docs lookup, no regeneration",
        asserts_effect=lambda c, t: (
            c["code"] == "result = apis.phone.get_contacts()"
            and t["code"] == "print(apis.api_docs.show_api_doc(app_name='phone', "
                             "api_name='get_contacts'))"
        ),
    )


def _pe_reprompt_probe() -> BoundaryProbe:
    eta = _eta("ground_reprompt", _PE)
    failure = "Execution failed. Traceback:\nNo API named 'get_contacts' found in the phone app."

    def run(controller):
        agent = _bare_agent()
        return _run_step(
            agent, controller, ["result = apis.phone.get_contacts()"],
            last_execution_outputs=[ExecutionIO(content=failure)],
        )

    return BoundaryProbe(
        name="post_execution/reprompt", boundary=_PE, action="reprompt", run=run,
        controller=Controller(boundary=_PE, action="reprompt", predicate=_ALWAYS_FIRES, eta=eta),
        effect_claim="the instruction is appended before the NEXT generation, after a visible failure",
        asserts_effect=lambda c, t: (
            len(t["messages"]) == len(c["messages"]) + 1
            and any(eta["instruction"] in m.get("content", "") for m in t["messages"])
        ),
    )


def _pe_reroute_probe() -> BoundaryProbe:
    failure = "Execution failed. Traceback:\nNo API named 'get_contacts' found in the phone app."

    def run(controller):
        agent = _bare_agent()
        agent._last_target_app = "phone"
        agent._last_target_api = "get_contacts"
        # Control still reaches generation (no PE controller intercepts), so it needs a canned cell;
        # treated returns before generation is ever attempted, so it needs none.
        cells = ["result = apis.phone.retry()"] if controller is None else []
        return _run_step(
            agent, controller, cells, last_execution_outputs=[ExecutionIO(content=failure)]
        )

    return BoundaryProbe(
        name="post_execution/reroute", boundary=_PE, action="reroute", run=run,
        controller=Controller(boundary=_PE, action="reroute", predicate=_ALWAYS_FIRES, eta={}),
        effect_claim="dispatches a corrective api_docs lookup instead of asking the model to generate",
        asserts_effect=lambda c, t: (
            c["code"] == "result = apis.phone.retry()"
            and t["code"] == "print(apis.api_docs.show_api_doc(app_name='phone', "
                             "api_name='get_contacts'))"
        ),
    )


ADAPTER_PROBES: tuple[BoundaryProbe, ...] = (
    _pre_reprompt_probe(),
    _pg_reprompt_probe(),
    _pg_suppress_probe(),
    _pg_reroute_probe(),
    _pe_reprompt_probe(),
    _pe_reroute_probe(),
)


@pytest.mark.parametrize("probe", ADAPTER_PROBES, ids=lambda p: p.name)
def test_every_declared_capability_proves_itself_behaviorally(probe: BoundaryProbe) -> None:
    ok, detail = run_boundary_contract(probe)
    assert ok, detail


def test_every_CAPS_entry_has_a_registered_probe() -> None:
    """Ties this file to `adapter.py::_CAPS` directly: a capability added there with no probe here
    must fail loudly, not silently ship unproven."""
    declared = set(ADAPTER._CAPS)
    covered = {(p.boundary, p.action) for p in ADAPTER_PROBES}
    assert declared == covered, f"declared {declared - covered} has no probe; probed {covered - declared} is not declared"


# ================================================================================================
# NEGATIVE CONTROL -- proves the harness itself catches an inert (telemetry-only) mechanism
# ================================================================================================

def test_a_telemetry_only_reprompt_FAILS_the_behavioral_contract() -> None:
    """The same shape as the reference file's own negative test, reproduced against a bare agent: a
    mechanism whose `run()` never reads `controller` at all -- it calls `_generate_and_extract()`
    directly, bypassing the wholesale override's PRE_GENERATION check entirely -- dispatches the exact
    same cell and message list whether or not a controller is present. Control and treated are
    therefore byte-identical, proving this harness would have caught the exact defect (a mechanism
    that records telemetry but never touches the trajectory) it exists to rule out for the six probes
    above."""

    def run_telemetry_only(controller):
        agent = _bare_agent()
        agent.language_model = _FakeLanguageModel(["result = apis.phone.get_contacts()"])
        code, _usage = agent._generate_and_extract()
        return {"code": code, "messages": [dict(m) for m in agent.messages]}

    probe = BoundaryProbe(
        name="telemetry-only pre_generation/reprompt", boundary=_PRE, action="reprompt",
        run=run_telemetry_only,
        controller=Controller(boundary=_PRE, action="reprompt", predicate=_ALWAYS_FIRES,
                              eta={"instruction": "ignored", "retry_budget": 1}),
        effect_claim="the instruction changes what the agent runs or sees",
        asserts_effect=lambda c, t: c != t,
    )
    ok, detail = run_boundary_contract(probe)
    assert not ok, "a telemetry-only mechanism must FAIL the behavioral contract"
    assert "IDENTICAL" in detail


# ================================================================================================
# PER-EPISODE BUDGET -- persists across steps; exhaustion degrades to control, never drops the cell
# ================================================================================================

def test_the_per_episode_budget_persists_and_degrades_to_control() -> None:
    eta = _eta("ground_reprompt", _PG)
    ctl = Controller(boundary=_PG, action="reprompt", predicate=_ALWAYS_FIRES, eta=eta)
    agent = _bare_agent(task_id="ep1")

    first = _run_step(
        agent, ctl, ["result = apis.phone.get_kontacts()", "result = apis.phone.get_contacts()"]
    )
    assert first["code"] == "result = apis.phone.get_contacts()", "the first firing must regenerate"
    assert agent.budget_spent["ep1"] == 1

    second = _run_step(agent, ctl, ["result = apis.phone.get_kontacts_again()"])
    assert second["code"] == "result = apis.phone.get_kontacts_again()", (
        "on exhaustion the generated cell must DISPATCH unmodified -- the arm degrades to control"
    )

    # Same agent, same `budget_spent` dict (already `{"ep1": 1}`) -- a new task_id must still get its
    # own fresh budget, proving the dict is keyed per episode rather than a single corpus-wide counter.
    agent.world = SimpleNamespace(task_id="ep2")
    third = _run_step(
        agent, ctl, ["result = apis.phone.get_kontacts()", "result = apis.phone.get_contacts()"]
    )
    assert third["code"] == "result = apis.phone.get_contacts()", "the budget is PER EPISODE, not global"
    assert agent.budget_spent == {"ep1": 1, "ep2": 1}
