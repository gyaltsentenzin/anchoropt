"""Phi_tau2 -- the conditions this host can already evaluate, and where each is observable.

DELIBERATELY COARSE. The shipped vocabulary names the facts the tau-bench runtime hands over directly:
a call was proposed, it mutates state, the turn committed to a reply, the last result errored. It
CANNOT express a thresholded condition ("the proposed arguments are unusually large", "this is the third
consecutive error"), because those have to be SYNTHESIZED from the typed alphabet at a boundary -- which
is half the algorithm and is not exercised if the adapter ships the answer.

Nothing here ranks or prefers a signal. `evaluate_signal` answers "does this condition hold on this
state"; which condition is worth conditioning on is core's question, decided by measurement.

RAISING ON AN UNKNOWN NAME IS INTENTIONAL. A signal that silently returns False is indistinguishable
from a condition that did not hold, which is the failure mode this whole contract exists to prevent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from anchoropt.anchor import IncisionPoint

_PRE = IncisionPoint.PRE_GENERATION
_PG = IncisionPoint.POST_GENERATION_PRE_EXEC
_PE = IncisionPoint.POST_EXECUTION

# ------------------------------------------------------------------------------------------------
# The shipped names, and the single locus set each is observable at.
# ------------------------------------------------------------------------------------------------
SIGNAL_BOUNDARIES: Mapping[str, frozenset] = {
    # POST_GENERATION_PRE_EXEC -- about the proposed call, before anything runs.
    "proposes_tool_call": frozenset({_PG}),
    "proposes_state_change": frozenset({_PG}),
    "proposes_unknown_tool": frozenset({_PG}),
    "commits_to_reply": frozenset({_PG}),
    # POST_EXECUTION -- about what came back.
    "tool_call_failed": frozenset({_PE}),
    # PRE_GENERATION -- carried history only; no call exists yet.
    "following_tool_error": frozenset({_PRE}),
}

SIGNALS: tuple[str, ...] = tuple(SIGNAL_BOUNDARIES)

# Human-readable phrases, used for expressibility matching only. They never steer a choice.
SIGNAL_ALIASES: Mapping[str, tuple[str, ...]] = {
    "proposes_tool_call": ("the agent proposed a tool call",),
    "proposes_state_change": ("the proposed call would change the database",),
    "proposes_unknown_tool": ("the proposed tool does not exist",),
    "commits_to_reply": ("the agent is about to answer instead of acting",),
    "tool_call_failed": ("the tool call returned an error",),
    "following_tool_error": ("this turn follows a failed tool call",),
}

# No tunable theta_phi: every shipped signal is DETERMINISTIC once installed. Thresholded conditions
# come from core's signal grammar over the typed fields, not from a hand-declared grid here.
SIGNAL_PARAM_DOMAINS: Mapping[str, tuple] = {}


def _b(state: Mapping[str, Any], key: str) -> bool:
    return bool(state.get(key))


def _i(state: Mapping[str, Any], key: str) -> int:
    try:
        return int(state.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def evaluate(signal: str, state: Mapping[str, Any], params: Mapping[str, Any] | None = None) -> bool:
    """Evaluate one shipped condition on one observable state. Unknown names RAISE."""
    if signal == "proposes_tool_call":
        return _b(state, "proposes_tool_call")
    if signal == "proposes_state_change":
        return _b(state, "proposed_tool_mutates_state")
    if signal == "proposes_unknown_tool":
        # A tool call was proposed AND the toolset does not contain it. Requiring the proposal keeps
        # this False on a reply-commitment state rather than True-by-vacuity.
        return _b(state, "proposes_tool_call") and not _b(state, "proposed_tool_is_known")
    if signal == "commits_to_reply":
        return _b(state, "commits_to_reply")
    if signal == "tool_call_failed":
        return _b(state, "result_is_error")
    if signal == "following_tool_error":
        return _b(state, "inbound_is_tool_result") and _i(state, "consecutive_tool_errors") > 0
    raise KeyError(f"tau2 adapter cannot evaluate signal {signal!r}")
