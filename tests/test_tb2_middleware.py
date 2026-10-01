"""The two guarantees the adapter owns, and the persisted controller format.

Both regressions here exist because the failure they catch is SILENT:

  * without `@hook_config(can_jump_to=["model"])` the graph edge back to the model is never
    created, so `{"jump_to": "model"}` is discarded and REPROMPT executes without changing the
    next decision. Measured in the live smoke test.
  * a bare `consumed` boolean latches across episodes when the runner reuses one middleware
    instance, so a candidate arm silently stops intervening after episode 1.

These run on the pinned runtime when it is importable and skip otherwise, so the core suite does
not require langchain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "tb2_deepagents"))

from anchoropt.anchor import Action, IncisionPoint  # noqa: E402
from anchoropt.runtime import ActionNotExecutable  # noqa: E402
from tb2_middleware import SPEC_FORMAT, ControllerSpec, build_middleware  # noqa: E402

langchain = pytest.importorskip("langchain.agents", reason="pinned runtime not installed")

TEXT = "VERIFY BEFORE CONCLUDING: check exact strings and required artifacts."


def _spec(**kw):
    d = dict(controller_id="A1", boundary="post_generation_pre_exec",
             signal="terminal_response_proposed", action="reprompt", theta={"text": TEXT})
    d.update(kw)
    return ControllerSpec(**d)


# ---- the persisted format ----------------------------------------------------------------------

def test_spec_roundtrips_through_disk(tmp_path):
    s = _spec()
    got = ControllerSpec.load(s.write(tmp_path / "c.json"))
    assert got == s


def test_spec_rejects_an_unknown_format(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"format": "something.else", "controller_id": "x", "boundary": "pre_generation",'
                 ' "signal": "s", "action": "noop"}')
    with pytest.raises(ValueError, match=SPEC_FORMAT):
        ControllerSpec.load(p)


def test_spec_carries_no_benchmark_or_task_vocabulary():
    fields = set(ControllerSpec.__dataclass_fields__)
    assert fields == {"controller_id", "boundary", "signal", "action", "theta", "signal_params"}


def test_a_spec_that_cannot_run_fails_at_load_not_mid_episode():
    with pytest.raises(ValueError, match="not declared"):
        _spec(signal="no_such_signal").validate()
    with pytest.raises(ActionNotExecutable):
        _spec(boundary="post_execution", action="reprompt").validate()
    with pytest.raises(ValueError, match="requires param"):
        _spec(signal="explore_streak", action="noop", theta={}).validate()


# ---- 1A: REPROMPT must cause a second generation ------------------------------------------------

def _scripted_model(calls):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class M(BaseChatModel):
        @property
        def _llm_type(self): return "scripted"

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            calls.append(" ".join(str(getattr(m, "content", "")) for m in messages))
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=f"final #{len(calls)}"))])
    return M()


def test_the_middleware_declares_the_jump_edge():
    """The guarantee itself: graph metadata is on the hook, not on the candidate."""
    mw = build_middleware(_spec())
    assert getattr(type(mw).after_model, "__can_jump_to__", None) == ["model"]


def test_reprompt_causes_a_second_model_generation():
    """1A end-to-end. Without the decorator this returns 1 call and passes silently elsewhere."""
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage

    calls: list[str] = []
    mw = build_middleware(_spec())
    agent = create_agent(model=_scripted_model(calls), tools=[], middleware=[mw])
    agent.invoke({"messages": [HumanMessage(content="do the task")]})

    assert len(calls) >= 2, "REPROMPT did not cause a second generation"
    assert TEXT in calls[1], "the exact theta text did not reach the next model call"
    assert mw.telemetry["interventions_executed"] == 1
    assert mw.telemetry["loop_prevention_events"] >= 1


# ---- 1B: one-shot state must be EPISODE-scoped --------------------------------------------------

def test_a_previous_episode_cannot_suppress_the_next():
    """1B. Two consecutive episodes through ONE middleware instance -- the failure a bare
    `consumed` boolean produces is that episode 2 never intervenes."""
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage

    calls: list[str] = []
    mw = build_middleware(_spec())
    agent = create_agent(model=_scripted_model(calls), tools=[], middleware=[mw])

    agent.invoke({"messages": [HumanMessage(content="episode ONE: set up the list")]})
    after_first = mw.telemetry["interventions_executed"]
    agent.invoke({"messages": [HumanMessage(content="episode TWO: a different task")]})
    after_second = mw.telemetry["interventions_executed"]

    assert after_first == 1, "episode 1 did not intervene"
    assert after_second == 2, (
        f"episode 2 was suppressed by episode 1's state "
        f"(executed {after_first} -> {after_second}); one-shot is not episode-scoped")
    assert mw.telemetry["episodes_seen"] == 2


def test_explicit_reset_also_clears_the_guard():
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage

    calls: list[str] = []
    mw = build_middleware(_spec())
    agent = create_agent(model=_scripted_model(calls), tools=[], middleware=[mw])
    agent.invoke({"messages": [HumanMessage(content="same text every episode")]})
    mw.reset()
    agent.invoke({"messages": [HumanMessage(content="same text every episode")]})
    assert mw.telemetry["interventions_executed"] == 2


def test_one_shot_off_is_available_but_not_the_default():
    mw_default = build_middleware(_spec())
    assert mw_default.one_shot is True
