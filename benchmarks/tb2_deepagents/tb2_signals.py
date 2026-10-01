"""Phi_TB2 -- TerminalBench's own signal evaluators. Registered THROUGH the adapter.

Deliberately separate from the generic core: `anchoropt/` must not learn what a terminal
response or a rebuild is. Each signal is a pure function of the adapter-normalized observable
state, so adding one requires NO change to any action implementation -- one of the generality
checks this integration has to satisfy.

Only the signals the terminal-response smoke path needs are implemented here. The category-B
signals derived from the H1 residual (`required_artifact_absent`, `source_edited_without_rebuild`)
are NOT pre-registered: they must come through the expansion step
(`anchoropt/learning/expand_attribution.py`), which scores coverage x precision and refuses a
condition that fires as often on successes as failures. Registering them by hand here would
bypass exactly the discipline that caught `single_read_then_answer` (precision 0.51).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable


def terminal_response_proposed(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """The model proposed a FINAL answer rather than a tool call.

    This is the observable the TB2 residual analysis kept needing and no PRE_GENERATION signal
    could supply: before generation the output does not exist yet. It is deterministic in the
    host runtime -- LangChain's own loop-exit test is `len(last_ai_message.tool_calls) == 0`
    (langchain 1.3.0 `agents/factory.py:1744`) -- and the adapter normalizes it into
    `proposes_tool_call` so this predicate never touches an AIMessage.

    Boundary: POST_GENERATION_PRE_EXEC. The generation has happened and nothing has run, which is
    what makes the terminal decision still changeable.
    """
    if not state.get("has_generation", False):
        return False
    return not bool(state.get("proposes_tool_call", False))


def proposed_tool_action(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """The complementary case: a concrete tool call is proposed and has not run.

    Present so the two POST_GENERATION_PRE_EXEC sub-cases are both first-class. The existing
    incision point covers proposed tool actions AND proposed terminal responses; nothing here
    introduces a fourth boundary.
    """
    return bool(state.get("has_generation", False)) and bool(state.get("proposes_tool_call", False))


def explore_streak(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """Consecutive read-only/exploratory calls, `at_least` of them.

    Included because it is the ONE signal the TB2 H1 residual could already express with no
    expansion (cluster 9, `qemu-startup`: 'steps 4, 6, 7 check KVM, architecture and strace
    unnecessarily'). Its threshold is NOT validated on TB2 -- it is here to exercise the path,
    not to claim a gain.
    """
    at_least = int(params["at_least"])
    return int(state.get("explore_streak", 0)) >= at_least


# Natural-language aliases per signal, used ONLY by expressibility matching -- never by the
# runtime, which dispatches on the canonical name alone.
#
# Why this exists: keyword matching compares a diagnosis's words against a SIGNAL NAME, and the
# two vocabularies genuinely differ. The `mailman` residual says "premature conclusion" and
# "concluded with a chat response"; the signal is called `terminal_response_proposed`. They
# describe the same observable and share only the weak token "response", so the canonical name
# alone marks the canonical case SIGNAL_BLOCKED.
#
# These aliases are a declared, reviewable part of the adapter's vocabulary. They are NOT a
# semantic model, and they do not make the matcher semantic -- a real expressibility test needs a
# mapping from `consequential_decision` to an observable, which is what Phi-expansion builds.
SIGNAL_ALIASES: Mapping[str, tuple[str, ...]] = {
    # Multi-word phrases only. Single verbs like "concluded" appear in prose about any failure --
    # they matched a DAG-edge reasoning error ("bn-fit-modify") on incidental wording, selecting a
    # SUPPRESS controller with no causal link. A phrase that names the EVENT does not.
    "terminal_response_proposed": ("premature conclusion", "premature termination",
                                   "chat response", "final answer", "without verifying",
                                   "before concluding", "terminated without"),
    "proposed_tool_action": ("tool call", "proposed command", "invoke"),
    "explore_streak": ("explore", "exploration", "unproductive", "unnecessarily", "diagnostic"),
}


SIGNALS: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], bool]] = {
    "terminal_response_proposed": terminal_response_proposed,
    "proposed_tool_action": proposed_tool_action,
    "explore_streak": explore_streak,
}

# Typed parameter schema per signal: name -> (type, required). Mirrors the BFCL adapter's habit of
# declaring rather than inferring, so a proposal omitting a required param is PRUNED and not run
# with a silent default -- a real failure mode recorded on this project.
SIGNAL_PARAMS: Mapping[str, Mapping[str, tuple[type, bool]]] = {
    "terminal_response_proposed": {},
    "proposed_tool_action": {},
    "explore_streak": {"at_least": (int, True)},
}
