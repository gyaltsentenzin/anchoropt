"""The typed alphabet AnchorOpt may synthesize conditions over, and WHERE each fact is readable.

BOUNDARY-TRUTHFULNESS IS THE WHOLE POINT OF THIS FILE. A field declared at a boundary the tau-bench
runtime does not carry there makes every predicate over it answer False -- for a structural reason
indistinguishable from the condition not holding. So each field below is placed only where
`tau2_mechanism` can actually read it off live state at that point in `generate_next_message`.

The three loci, in tau-bench terms:

  PRE_GENERATION            the agent turn has begun; the inbound message is in hand. NO CALL EXISTS.
  POST_GENERATION_PRE_EXEC  `generate()` returned a candidate. The proposed call is visible; the
                            orchestrator has not been given it, so nothing has run.
  POST_EXECUTION            a ToolMessage came back from the orchestrator. The world has moved.

WHAT IS DELIBERATELY *NOT* AT PRE_GENERATION. No `proposed_*` field: the candidate does not exist yet.
WHAT IS DELIBERATELY *NOT* AT POST_GENERATION_PRE_EXEC. No `result_*` / `error_kind` field for THIS
turn: a result cannot exist before dispatch, and the guide is explicit that a result at a pre-dispatch
boundary is always a defect.

CARRIED HISTORY IS A DIFFERENT THING FROM A RESULT. `consecutive_tool_errors` and `tool_errors_so_far`
summarize turns that already finished. They are legitimately readable before this turn's generation --
the agent's own message list contains them -- and they are named `*_so_far` / `consecutive_*` rather
than `error_*` so no reader mistakes them for the pending call's outcome.
"""

from __future__ import annotations

from collections.abc import Mapping

from anchoropt.anchor import IncisionPoint

# ------------------------------------------------------------------------------------------------
# This adapter's OWN boundary keys. `tau2_runtime.boundary_from_key` is the only place the mapping to
# core loci lives -- core owns no key->locus table, which would be a stage map imposed from outside.
# ------------------------------------------------------------------------------------------------
TURN_START = "before_agent_turn"
GATE = "before_tool_dispatch"
AFTER = "after_tool_result"

KEY_TO_POINT: Mapping[str, IncisionPoint] = {
    TURN_START: IncisionPoint.PRE_GENERATION,
    GATE: IncisionPoint.POST_GENERATION_PRE_EXEC,
    AFTER: IncisionPoint.POST_EXECUTION,
}

POINT_TO_KEY: Mapping[str, str] = {p.value: k for k, p in KEY_TO_POINT.items()}

LABELS: Mapping[str, str] = {
    TURN_START: "the agent turn has begun and no call has been proposed",
    GATE: "a call has been proposed and nothing has run",
    AFTER: "a tool result came back",
}

_PRE = IncisionPoint.PRE_GENERATION.value
_PG = IncisionPoint.POST_GENERATION_PRE_EXEC.value
_PE = IncisionPoint.POST_EXECUTION.value

# Error classes read off the returned ToolMessage. These are OBSERVABLES, not recommendations: the
# adapter says what the runtime reports, and core decides whether any of it should drive an action.
ERROR_KINDS = ("", "tool_not_found", "invalid_arguments", "not_found", "domain_error")


class Field:
    """One typed observable. Core reads `.name`, `.type` and `.boundaries` when synthesizing Phi."""

    def __init__(self, name: str, type_: type, boundaries: tuple[str, ...], doc: str = "",
                 enum: tuple = ()):
        self.name, self.type, self.boundaries = name, type_, tuple(boundaries)
        self.doc, self.enum = doc, tuple(enum)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Field({self.name!r}, {self.type.__name__}, {self.boundaries})"


def _f(name: str, type_: type, boundaries: tuple[str, ...], doc: str = "", enum: tuple = ()) -> Field:
    return Field(name, type_, boundaries, doc, enum)


# ------------------------------------------------------------------------------------------------
# The alphabet, per locus.
# ------------------------------------------------------------------------------------------------
FIELDS_AT: Mapping[str, Mapping[str, Field]] = {
    _PRE: {
        "turn_index": _f("turn_index", int, (_PRE, _PG, _PE),
                         "position of this agent turn in the episode"),
        "assistant_turns_so_far": _f("assistant_turns_so_far", int, (_PRE,),
                                     "agent messages already emitted"),
        "user_turns_so_far": _f("user_turns_so_far", int, (_PRE,),
                                "user messages already received"),
        "inbound_is_tool_result": _f("inbound_is_tool_result", bool, (_PRE,),
                                     "this turn was triggered by a tool result rather than the user"),
        "consecutive_tool_errors": _f("consecutive_tool_errors", int, (_PRE, _PG),
                                      "errored tool results at the end of the history so far"),
        "tool_errors_so_far": _f("tool_errors_so_far", int, (_PRE, _PG, _PE),
                                 "errored tool results in the episode so far"),
    },
    _PG: {
        # THE PROPOSED CALL. Visible here and nowhere earlier.
        "proposes_tool_call": _f("proposes_tool_call", bool, (_PG,),
                                 "the candidate carries at least one tool call"),
        "commits_to_reply": _f("commits_to_reply", bool, (_PG,),
                               "the candidate is a message to the user and calls no tool -- the "
                               "answer-commitment decision"),
        "n_proposed_calls": _f("n_proposed_calls", int, (_PG,), "tool calls in the candidate"),
        "proposed_tool": _f("proposed_tool", str, (_PG,), "name of the first proposed tool"),
        "proposed_tool_mutates_state": _f(
            "proposed_tool_mutates_state", bool, (_PG,),
            "the proposed tool is declared mutating by the toolkit, so it would change the DB"),
        "proposed_tool_is_known": _f("proposed_tool_is_known", bool, (_PG,),
                                     "the proposed tool exists in this agent's toolset"),
        "n_proposed_args": _f("n_proposed_args", int, (_PG,), "arguments supplied to the call"),
        "proposed_args_chars": _f("proposed_args_chars", int, (_PG,),
                                  "serialized size of the proposed arguments"),
        "replan_attempt": _f("replan_attempt", int, (_PG,),
                             "how many times this turn has already re-planned"),
        # Carried facts. Both are summaries of FINISHED turns -- see the module docstring.
        "turn_index": _f("turn_index", int, (_PRE, _PG, _PE), "position of this agent turn"),
        "consecutive_tool_errors": _f("consecutive_tool_errors", int, (_PRE, _PG),
                                      "errored tool results at the end of the history so far"),
        "tool_errors_so_far": _f("tool_errors_so_far", int, (_PRE, _PG, _PE),
                                 "errored tool results in the episode so far"),
    },
    _PE: {
        # THE RESULT. Cannot exist before dispatch.
        "result_is_error": _f("result_is_error", bool, (_PE,),
                              "the returned ToolMessage carries error=True"),
        "error_kind": _f("error_kind", str, (_PE,), "classified error from the returned result",
                         enum=ERROR_KINDS),
        "result_chars": _f("result_chars", int, (_PE,), "size of the returned content"),
        "result_tool": _f("result_tool", str, (_PE,), "tool that produced this result"),
        "result_tool_mutates_state": _f("result_tool_mutates_state", bool, (_PE,),
                                        "the tool that ran is declared mutating"),
        "turn_index": _f("turn_index", int, (_PRE, _PG, _PE), "position of this agent turn"),
        "tool_errors_so_far": _f("tool_errors_so_far", int, (_PRE, _PG, _PE),
                                 "errored tool results in the episode so far"),
    },
}

# Keys every state carries regardless of boundary. `states_at` must let these through or a projected
# state loses its identity and cannot be paired with a case.
IDENTITY_KEYS = ("boundary", "case_id")


def fields_at(boundary) -> dict[str, Field]:
    """The alphabet at ONE locus. Accepts an IncisionPoint, its value, or an adapter key."""
    raw = str(getattr(boundary, "value", boundary))
    if raw in KEY_TO_POINT:
        raw = KEY_TO_POINT[raw].value
    return dict(FIELDS_AT.get(raw, {}))


def all_field_names() -> frozenset[str]:
    return frozenset(n for m in FIELDS_AT.values() for n in m)
