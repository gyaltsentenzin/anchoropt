"""The TB2/DeepAgents adapter, and the terminal-response smoke path.

The claim under test is the one the whole integration rests on: `POST_GENERATION_PRE_EXEC` covers
BOTH a proposed tool action and a proposed terminal response, so `verify_exact`-shaped control is
representable with NO fourth boundary and NO change to the core action implementations.

Also asserted here: no LangChain type and no TB2 task id crosses into `anchoropt/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "tb2_deepagents"))

import tb2_adapter as tb2  # noqa: E402
from anchoropt.anchor import Action, Anchor, IncisionPoint  # noqa: E402
from anchoropt.runtime import ActionNotExecutable  # noqa: E402

VERIFY_EXACT = (
    "Before concluding, verify the result with the most targeted command, file read, or test you "
    "can run. Explicitly check for exact string matches, required directory structures, and "
    "version outputs. Do not assume success based on partial matches or baseline artifacts; "
    "validate against the precise task requirements."
)


class _AIMessage:
    """Stand-in for a LangChain AIMessage -- attribute access, `tool_calls` list."""

    def __init__(self, tool_calls=None):
        self.tool_calls = list(tool_calls or [])


# ---- 1. event normalization -------------------------------------------------------------------

def test_terminal_response_is_detected_from_an_empty_tool_call_list():
    """LangChain's own loop-exit test is len(tool_calls) == 0 (factory.py:1744)."""
    rec = tb2.normalize_event({"boundary": IncisionPoint.POST_GENERATION_PRE_EXEC,
                               "messages": [_AIMessage(tool_calls=[])]})
    assert rec["has_generation"] is True
    assert rec["proposes_tool_call"] is False
    assert rec["n_tool_calls"] == 0


def test_proposed_tool_action_is_detected_and_its_command_extracted():
    rec = tb2.normalize_event({"boundary": IncisionPoint.POST_GENERATION_PRE_EXEC,
                               "messages": [_AIMessage(tool_calls=[
                                   {"name": "shell", "args": {"command": "ls -la /app"}}])]})
    assert rec["proposes_tool_call"] is True
    assert rec["tool_name"] == "shell"
    assert rec["proposed_command"] == "ls -la /app"


def test_normalization_accepts_dicts_as_well_as_objects():
    """Live middleware hands over objects; recorded transcripts are JSON. Both must work."""
    obj = tb2.normalize_event({"messages": [_AIMessage(tool_calls=[{"name": "shell", "args": {}}])]})
    dct = tb2.normalize_event({"messages": [{"tool_calls": [{"name": "shell", "args": {}}]}]})
    assert obj["proposes_tool_call"] == dct["proposes_tool_call"] is True


def test_no_generation_yet_is_not_a_terminal_response():
    """Absence of a generation must not read as 'proposed a final answer' -- that inversion is
    exactly the prototype's `no_tool_call_yet` defect."""
    state = tb2.observable_state(tb2.normalize_event({"messages": []}))
    assert tb2.evaluate_signal("terminal_response_proposed", state) is False


# ---- 2. signals ---------------------------------------------------------------------------------

def test_the_two_post_generation_subcases_are_mutually_exclusive():
    for calls in ([], [{"name": "shell", "args": {"command": "ls"}}]):
        state = tb2.observable_state(tb2.normalize_event({"messages": [_AIMessage(calls)]}))
        term = tb2.evaluate_signal("terminal_response_proposed", state)
        tool = tb2.evaluate_signal("proposed_tool_action", state)
        assert term != tool, "a generation is either a proposed call or a terminal response"


def test_unknown_signal_raises_rather_than_returning_false():
    """A missing evaluator returning False is indistinguishable from 'did not fire' -- how an arm
    silently becomes the control arm."""
    with pytest.raises(KeyError, match="no evaluator"):
        tb2.evaluate_signal("does_not_exist", {})


def test_a_required_param_is_enforced_not_defaulted():
    state = {"explore_streak": 5}
    with pytest.raises(ValueError, match="requires param"):
        tb2.evaluate_signal("explore_streak", state)
    assert tb2.evaluate_signal("explore_streak", state, {"at_least": 3}) is True


def test_bool_is_rejected_where_an_int_is_required():
    with pytest.raises(TypeError, match="must be int"):
        tb2.evaluate_signal("explore_streak", {"explore_streak": 5}, {"at_least": True})


def test_a_signal_can_be_added_without_touching_any_action_implementation():
    """Generality check: TB2 must be able to add a signal without editing action code."""
    before = dict(tb2.apply_action(Action.NOOP, IncisionPoint.PRE_GENERATION, {}, {}))
    tb2.register_signal("smoke_only_signal", lambda s, p: True, {})
    assert "smoke_only_signal" in tb2.declared_signals()
    assert tb2.evaluate_signal("smoke_only_signal", {}) is True
    assert dict(tb2.apply_action(Action.NOOP, IncisionPoint.PRE_GENERATION, {}, {})) == before


def test_a_frozen_signal_cannot_be_silently_replaced():
    with pytest.raises(ValueError, match="already registered"):
        tb2.register_signal("terminal_response_proposed", lambda s, p: False, {})


# ---- 3. U_H(l) --------------------------------------------------------------------------------

def test_the_host_declares_only_measured_executable_cells():
    """POST_EXECUTION observation-rewriting is withheld: the prototype's implementation read a
    field absent off POST_EXECUTION and reported success anyway."""
    assert tb2.feasible_actions(IncisionPoint.POST_EXECUTION) == {Action.NOOP}
    assert Action.REROUTE in tb2.feasible_actions(IncisionPoint.POST_GENERATION_PRE_EXEC)
    assert tb2.feasible_actions(IncisionPoint.PRE_GENERATION) == {Action.NOOP, Action.REPROMPT}


def test_an_unexecutable_action_raises_before_building_a_payload():
    """It must be impossible for an unexecutable cell to report success."""
    with pytest.raises(ActionNotExecutable):
        tb2.apply_action(Action.REROUTE, IncisionPoint.POST_EXECUTION,
                         {"tool_name": "x", "reason": "r"}, {})


# ---- 4. the terminal-response smoke path ------------------------------------------------------

def test_verify_exact_is_representable_at_the_existing_boundary():
    """The whole point: l = POST_GENERATION_PRE_EXEC (terminal case), phi =
    terminal_response_proposed, mu = REPROMPT, theta.text = the frozen instruction."""
    anchor = Anchor(
        name="verify_exact_localized",
        locus="verification/terminal_response/unverified",
        attribution="terminal response proposed without exact verification",
        incision_point=IncisionPoint.POST_GENERATION_PRE_EXEC,
        action=Action.REPROMPT,
        params={"signal": "terminal_response_proposed", "text": VERIFY_EXACT},
    )
    assert anchor.action in tb2.feasible_actions(anchor.incision_point)

    state = tb2.observable_state(tb2.normalize_event({
        "boundary": IncisionPoint.POST_GENERATION_PRE_EXEC,
        "messages": [_AIMessage(tool_calls=[])],
    }))
    assert tb2.evaluate_signal(anchor.params["signal"], state) is True

    directive = tb2.apply_action(anchor.action, anchor.incision_point,
                                {"text": anchor.params["text"]}, state)
    assert directive["executed"] is True
    # The directive is SEMANTIC: `text` is what the model must see, `request_redecision` says
    # another model decision is required. The middleware picks the delivery channel.
    assert directive["text"] == VERIFY_EXACT
    assert directive["request_redecision"] is True


def test_the_instruction_text_is_carried_verbatim():
    """AnchorOpt localizes; it never rewrites the provider's behavioural content."""
    d = tb2.apply_action(Action.REPROMPT, IncisionPoint.POST_GENERATION_PRE_EXEC,
                         {"text": VERIFY_EXACT}, {})
    assert d["text"] == VERIFY_EXACT


def test_reprompt_at_pre_generation_does_not_request_redecision():
    """At PRE_GENERATION the next generation has not happened; appending to the prompt suffices."""
    d = tb2.apply_action(Action.REPROMPT, IncisionPoint.PRE_GENERATION, {"text": "x"}, {})
    assert d["request_redecision"] is False


def test_empty_reprompt_text_is_refused():
    with pytest.raises(ValueError, match="non-empty"):
        tb2.apply_action(Action.REPROMPT, IncisionPoint.PRE_GENERATION, {"text": "  "}, {})


def test_suppress_and_reroute_smoke():
    s = tb2.apply_action(Action.SUPPRESS, IncisionPoint.POST_GENERATION_PRE_EXEC,
                         {"reason": "unproductive exploration"}, {})
    assert s["short_circuit"] is True
    r = tb2.apply_action(Action.REROUTE, IncisionPoint.POST_GENERATION_PRE_EXEC,
                         {"args_patch": {"overwrite": True}, "reason": "path exists"},
                         {"tool_name": "write_file", "tool_args": {"path": "/app/x"}})
    assert r["tool_name"] == "write_file"
    assert r["tool_args"] == {"path": "/app/x", "overwrite": True}


# ---- 5. isolation: no framework or benchmark vocabulary in the core ---------------------------

def test_the_core_contains_no_langchain_or_tb2_vocabulary():
    banned = ("after_model", "AIMessage", "jump_to", "langchain", "deepagents",
              "mailman", "qemu-startup", "build_verification_instruction")
    offenders = {}
    for path in (REPO / "anchoropt").rglob("*.py"):
        text = path.read_text()
        hits = [b for b in banned if b in text]
        if hits:
            offenders[str(path.relative_to(REPO))] = hits
    assert not offenders, f"framework/benchmark vocabulary leaked into the core: {offenders}"


def test_importing_the_tb2_adapter_does_not_hijack_the_active_adapter():
    """BFCL non-regression: importing TB2 must NOT register it, or BFCL's core call sites would
    start resolving against the wrong vocabulary."""
    from anchoropt.attribution import adapter_or_none
    active = adapter_or_none()
    # The TB2 adapter deliberately does NOT auto-register on import, unlike the BFCL adapter:
    # taking the single global slot would silently repoint BFCL's core call sites. Whatever is
    # registered here, it must not be this module.
    if active is not None:
        assert getattr(active, "NAME", None) != tb2.NAME
        assert active is not sys.modules.get("tb2_adapter")
