"""Live constraint state, and "would this turn violate anything" answered by UPSTREAM'S OWN HANDLERS.

WHAT THIS REPLACES, AND WHY THAT MATTERS
----------------------------------------
`cctu_capabilities.py` declares six per-turn pressure fields -- `call_times_over_cap`,
`per_tool_over_cap`, `parallel_over_cap`, `order_prereq_unmet`, `parallel_group_incomplete`,
`args_invalid` -- and its first draft described them as a SECOND implementation of upstream's rules,
needing a parity test each. That would have been the defect class `benchmarks/bfcl_v4/adapter.py`
exists to have ended: one regex implemented five times, three copies disagreeing, two silently
misfiling.

It turns out no mirror is needed. Every constraint handler reads and writes the checker through
PLAIN ATTRIBUTES and calls no method on it:

    RoundHandler              checker.round, min_round
    CallTimesHandler          checker.callTimes, min_callTimes, max_callTimes
    ParallelCallsHandler      checker.parallelCall_unit, accum_max_parallelCallTypes, min/max
    MaxCallsPerToolHandler    checker.callTimesPerTool, max_callTimesPerTool
    ToolOrderHandler          checker.tool_order, earliest_callTurnPerTool, first_tool_name, query_id
    ToolParallelHandler       checker.tool_parallel
    ResponseLengthHandler     checker.min/max_responseLength, responseLength_unit

So a duck-typed stand-in carrying those attributes, built from a snapshot of the live checker, can be
handed to upstream's real `check()` along with a throwaway `Feedback`. The handler then reports
exactly what it would report, the stand-in absorbs every mutation, and the live checker is untouched.

    the predicate IS upstream's logic, applied to a copy of the state

That is strictly better than a mirror, and it removes five parity risks rather than managing them. It
also inherits the awkward parts for free -- including `ToolOrderHandler`'s corpus-specific override
for `query_id == 96` (tool.py:77-88), where the effective order depends on which tool was called
first and is EMPTY unless that tool was `urban_area_identifier`. A hand-written mirror would have
disagreed on exactly one episode, which is the hardest kind of disagreement to notice and the easiest
to dismiss.

`args_invalid` needs no stand-in at all: `ToolArgsChecker.check` is already pure -- it reads
`tools_doc` and the proposed calls and mutates nothing -- so it is called directly.

WHAT IS DELIBERATELY NOT DRIVEN PREDICTIVELY
--------------------------------------------
`ResponseFormatHandler`, `ResponsePunctuationHandler` and `ResponseIdentifiersHandler` each load and
execute a per-episode Python file (`data/check_code/<query_id>/check_constraint_N.py`) through
`validator_loader.call_validator`. They are excluded here for the two reasons
`cctu_capabilities.py` records: executing arbitrary per-episode code on every proposed answer is a
cost to pay deliberately, and handing the anchor an exact validator where the model has only the
prose constraint is a capability asymmetry that should be reviewed rather than introduced quietly.

`PREDICTIVE_HANDLERS` is therefore an explicit allowlist, not "all of them minus whatever broke".
Adding the three is a one-line change plus a decision.

THE SNAPSHOT IS AS-OF-THE-BOUNDARY, AND THAT IS THE POINT
---------------------------------------------------------
`get_feedback_if` advances `round`, `callTimes`, `callTimesPerTool` and `earliest_callTurnPerTool`
mid-step. A snapshot taken at the commitment gate therefore differs from one taken after execution
within ONE turn, and the difference is exactly "still preventable" versus "already spent". Nothing
here caches: `snapshot_of` is called at the boundary where the answer is wanted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from cctu_signals import violation_class_of

INF = math.inf


# ------------------------------------------------------------------------------------------------
# 1. the snapshot
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckerSnapshot:
    """Everything upstream's constraint handlers read, captured at one boundary.

    A FROZEN, PLAIN-DATA record on purpose. It is the seam between the wrapper (which owns the live
    `DialogueConstraintChecker`) and everything here, so every function below is testable with no
    benchmark object, no episode and no corpus -- and a test can construct a state the corpus does
    not happen to contain.

    Defaults are upstream's own defaults from `DialogueConstraintChecker.__init__`, so a snapshot
    built from an episode with no constraints of a given kind behaves as that episode does.
    """

    # resource / interaction
    round: int = 0
    min_round: int = 0
    max_round: int = 20
    call_times: int = 0
    min_call_times: int = 0
    max_call_times: float = INF
    # per-tool
    call_times_per_tool: Mapping[str, int] = field(default_factory=dict)
    max_call_times_per_tool: Mapping[str, float] = field(default_factory=dict)
    # behavior / parallelism
    accum_max_parallel: int = 0
    min_parallel: int = 0
    max_parallel: float = INF
    parallel_unit: str = "type"
    tool_parallel: Sequence[Sequence[str]] = ()
    # behavior / ordering
    tool_order: Sequence[Sequence[str]] = ()
    earliest_call_turn: Mapping[str, int] = field(default_factory=dict)
    first_tool_name: str | None = None
    query_id: int = -1
    # response
    min_response_length: int = 0
    max_response_length: float = INF
    response_length_unit: str = "characters"
    # argument schema
    tools_doc: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def snapshot_of(checker: Any) -> CheckerSnapshot:
    """Capture the live checker. **The only function here that touches a benchmark object.**

    Mutable containers are COPIED, not referenced. A snapshot holding a live reference to
    `callTimesPerTool` would see the validator's own increments and report a pre-execution question
    using post-execution state -- which is the one distinction this module exists to preserve.
    """
    return CheckerSnapshot(
        round=int(getattr(checker, "round", 0)),
        min_round=_num(getattr(checker, "min_round", 0)),
        max_round=_num(getattr(checker, "max_round", 20)),
        call_times=int(getattr(checker, "callTimes", 0)),
        min_call_times=_num(getattr(checker, "min_callTimes", 0)),
        max_call_times=_num(getattr(checker, "max_callTimes", INF)),
        call_times_per_tool=dict(getattr(checker, "callTimesPerTool", {}) or {}),
        max_call_times_per_tool=dict(getattr(checker, "max_callTimesPerTool", {}) or {}),
        accum_max_parallel=int(getattr(checker, "accum_max_parallelCallTypes", 0)),
        min_parallel=_num(getattr(checker, "min_parallelCallTypes", 0)),
        max_parallel=_num(getattr(checker, "max_parallelCallTypes", INF)),
        parallel_unit=str(getattr(checker, "parallelCall_unit", "type")),
        tool_parallel=[list(g) for g in (getattr(checker, "tool_parallel", ()) or ())],
        tool_order=[list(o) for o in (getattr(checker, "tool_order", ()) or ())],
        earliest_call_turn=dict(getattr(checker, "earliest_callTurnPerTool", {}) or {}),
        first_tool_name=getattr(checker, "first_tool_name", None),
        query_id=int(getattr(checker, "query_id", -1)),
        min_response_length=_num(getattr(checker, "min_responseLength", 0)),
        max_response_length=_num(getattr(checker, "max_responseLength", INF)),
        response_length_unit=str(getattr(checker, "responseLength_unit", "characters")),
        tools_doc=dict(getattr(getattr(checker, "args_checker", None), "tools_doc", {}) or {}),
    )


def _num(v: Any) -> Any:
    """Keep `inf` as a float and everything else as an int, without going through `int(inf)`."""
    if isinstance(v, float) and not math.isfinite(v):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def _standin(snap: CheckerSnapshot) -> SimpleNamespace:
    """A duck-typed checker carrying only what the handlers touch, with FRESH mutable containers.

    The handlers write to `callTimes`, `callTimesPerTool`, `accum_max_parallelCallTypes`,
    `first_tool_name` and `earliest_callTurnPerTool`. Those writes land here and are discarded, which
    is what makes asking "would this turn violate anything" free of side effects.
    """
    return SimpleNamespace(
        round=snap.round, min_round=snap.min_round, max_round=snap.max_round,
        callTimes=snap.call_times, min_callTimes=snap.min_call_times,
        max_callTimes=snap.max_call_times,
        callTimesPerTool=dict(snap.call_times_per_tool),
        max_callTimesPerTool=dict(snap.max_call_times_per_tool),
        accum_max_parallelCallTypes=snap.accum_max_parallel,
        min_parallelCallTypes=snap.min_parallel, max_parallelCallTypes=snap.max_parallel,
        parallelCall_unit=snap.parallel_unit,
        tool_parallel=[list(g) for g in snap.tool_parallel],
        tool_order=[list(o) for o in snap.tool_order],
        earliest_callTurnPerTool=dict(snap.earliest_call_turn),
        first_tool_name=snap.first_tool_name, query_id=snap.query_id,
        min_responseLength=snap.min_response_length,
        max_responseLength=snap.max_response_length,
        responseLength_unit=snap.response_length_unit,
    )


# ------------------------------------------------------------------------------------------------
# 2. would this turn violate anything -- asked of upstream's own handlers
# ------------------------------------------------------------------------------------------------
#
# An explicit ALLOWLIST. The three validator-file handlers are absent by decision, not by accident
# (see the module docstring), and a handler that is added to the registry upstream does not silently
# start running here.
def _predictive_handlers():
    from utils.constraint_checker.handlers.interact import (
        CallTimesHandler, ParallelCallsHandler, RoundHandler,
    )
    from utils.constraint_checker.handlers.response import ResponseLengthHandler
    from utils.constraint_checker.handlers.tool import (
        MaxCallsPerToolHandler, ToolOrderHandler, ToolParallelHandler,
    )
    return (RoundHandler, CallTimesHandler, ParallelCallsHandler, MaxCallsPerToolHandler,
            ToolOrderHandler, ToolParallelHandler, ResponseLengthHandler)


PREDICTIVE_HANDLERS = _predictive_handlers


def would_violate(snap: CheckerSnapshot, *, tool_calls: Sequence[Mapping[str, Any]],
                  content: str = "", is_final: bool | None = None) -> frozenset[str]:
    """The canonical violation classes upstream's handlers WOULD report for this turn.

    Asked before the validator runs, so the answer is about what is still preventable. Every class
    comes back through `cctu_signals.violation_class_of`, which is also what classifies the
    violations that actually happen -- one owner, so a predicted class and an observed class are
    never two different vocabularies.

    `is_final` defaults to upstream's own test, `len(tool_calls) == 0` (`response_generator.py:129`),
    rather than to a parameter a caller could get wrong. Pass it explicitly only to ask a
    counterfactual: "if this call were suppressed, would the turn become terminal, and what would
    the terminal handlers then say?" -- which is a question a SUPPRESS candidate must ask, because
    dropping the last call of a turn silently converts a tool turn into a final answer.

    A handler that raises is treated as answering NOTHING, and the failure is recorded in
    `HANDLER_ERRORS` rather than swallowed. A predictive check is a convenience; it must never be
    able to kill an episode, and it must never look clean because it crashed.
    """
    from utils.constraint_checker.feedback import Feedback
    from utils.constraint_checker.handlers.base import TurnContext

    calls = list(tool_calls or [])
    if is_final is None:
        is_final = len(calls) == 0
    ctx = TurnContext(is_final=bool(is_final), content=str(content or ""), tool_calls=calls)

    seen: set[str] = set()
    for handler_cls in PREDICTIVE_HANDLERS():
        stand = _standin(snap)
        fb = Feedback()
        try:
            handler_cls().check(stand, ctx, fb)
        # BLE001 is silenced deliberately: a predictive check must never be able to kill an episode,
        # and narrowing this would mean guessing which exceptions a per-episode validator can raise.
        # It is NOT swallowed -- the failure is recorded below, because a guard that looks clean
        # because it crashed is worse than no guard.
        except Exception as exc:  # noqa: BLE001
            HANDLER_ERRORS.append(f"{handler_cls.__name__}: {type(exc).__name__}: {exc}")
            continue
        for msg in list(fb.user_msgs) + [m for v in fb.tool_msgs_by_callid.values() for m in v]:
            cls = violation_class_of(msg)
            if cls is not None:
                seen.add(cls)
    return frozenset(seen)


# Predictive-check failures, kept rather than discarded. A guard that swallows its own bugs is the
# defect `policy_tree._iter_result_collections` documents: a NameError vanished into a bare `except`
# and the detector silently returned None for every payload.
HANDLER_ERRORS: list[str] = []


def args_invalid(snap: CheckerSnapshot, tool_calls: Sequence[Mapping[str, Any]]) -> bool:
    """Would any proposed call fail the tool's own argument schema?

    Calls upstream's `ToolArgsChecker` DIRECTLY -- no stand-in and no copy -- because it is already
    pure: it reads `tools_doc` and the proposed calls and mutates nothing. Reusing it means unknown
    tools, unparseable JSON, missing required arguments, extra arguments and recursive schema
    failures are all classified exactly as the harness classifies them, including
    `schema_validate.validate_param_value`'s nested cases.
    """
    from utils.constraint_checker.args_checker import ToolArgsChecker

    if not tool_calls:
        return False
    return bool(ToolArgsChecker(dict(snap.tools_doc)).check([], list(tool_calls)))


# ------------------------------------------------------------------------------------------------
# 3. the carried state the adapter merges into `observable_state`
# ------------------------------------------------------------------------------------------------
def carried_state(snap: CheckerSnapshot, normalized: Mapping[str, Any], *,
                  round_index: int,
                  seen_calls: Sequence[tuple[str, str]] = (),
                  last_violation_class: str | None = None,
                  consecutive_violation_turns: int = 0) -> dict[str, Any]:
    """Build the carried half of the observable state: pressure, budgets, and episode history.

    Every key here is declared in `cctu_capabilities.py`, and every value is derived either from
    upstream's own verdict (`would_violate`, `args_invalid`) or from arithmetic over the snapshot.
    Nothing reads `unsolved_set` or `answer`.

    NO INFINITY LEAVES THIS FUNCTION. An episode that declares no cap has `max_call_times == inf`,
    and `signal_grammar` would turn that into the atoms `x < inf` (fires on everything finite) and
    `x > inf` (never fires) while shifting every other quantile on the field. So an unconstrained
    budget is reported as **None**, which `_numeric_values` skips -- no observation rather than a
    poisoned one. Verified by driving `atoms_for` both ways.

    `round_index` is the WRAPPER's 0-based turn counter, not `snap.round`, because `snap.round`
    advances inside the step and the same field must mean the same thing at all three boundaries.
    `rounds_remaining` is then turns available AFTER this one, so 0 means this is the last turn the
    budget allows -- agreeing with the harness's own `budget_exhausted` evaluated at the turn's end.
    """
    calls = list(normalized.get("tool_calls_raw") or ())
    tool_name = normalized.get("tool_name") or None
    is_final = not bool(normalized.get("proposes_tool_call", False))

    predicted = would_violate(snap, tool_calls=calls,
                              content=str(normalized.get("content") or ""),
                              is_final=is_final)

    state: dict[str, Any] = {
        # --- constraint pressure, per turn: upstream's own verdict, asked early ------------------
        "call_times_over_cap": "max_call_times" in predicted,
        "per_tool_over_cap": "max_calls_per_tool" in predicted,
        "parallel_over_cap": "max_parallel_calls" in predicted,
        "order_prereq_unmet": "tool_order" in predicted,
        "parallel_group_incomplete": "tool_parallel" in predicted,
        "args_invalid": args_invalid(snap, calls),

        # --- MIN-class and response-class: upstream only judges these on a terminal turn ---------
        "min_round_unmet": "min_round" in predicted,
        "min_call_times_unmet": "min_call_times" in predicted,
        "parallel_requirement_unmet": "min_parallel_calls" in predicted,
        "response_length_over_cap": "max_length" in predicted,
        "response_length_under_min": "min_length" in predicted,

        # --- turn context ------------------------------------------------------------------------
        "round_index": int(round_index),
        # An INT, because `cctu_capabilities` declares it as one and `max_round` is always finite:
        # upstream initialises it to `max_turns` and only ever narrows it with `min()`
        # (`constraint_checker/handlers/interact.py:29`). The None branch is defensive, not expected.
        "rounds_remaining": _as_int(_remaining(snap.max_round, int(round_index) + 1)),

        # --- budgets ----------------------------------------------------------------------------
        "call_times": int(snap.call_times),
        "call_times_remaining": _remaining(snap.max_call_times, snap.call_times),
        "calls_remaining_this_tool": _remaining(
            snap.max_call_times_per_tool.get(tool_name, INF) if tool_name else INF,
            snap.call_times_per_tool.get(tool_name, 0) if tool_name else 0,
        ) if tool_name else None,

        # --- episode history --------------------------------------------------------------------
        "repeated_identical_call": _repeats(calls, seen_calls),
        "last_violation_class": last_violation_class,
        "consecutive_violation_turns": int(consecutive_violation_turns),
    }
    return state


def _remaining(cap: Any, used: Any) -> float | None:
    """`cap - used`, or **None** when the cap is unbounded. Never `inf` -- see `carried_state`."""
    try:
        cap_f = float(cap)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cap_f):
        return None
    return float(cap_f - float(used))


def _as_int(v: float | None) -> int | None:
    """A finite remaining-count as an int, keeping None as None. Declared types must be honoured."""
    return None if v is None else int(v)


def _repeats(calls: Sequence[Mapping[str, Any]],
             seen: Sequence[tuple[str, str]]) -> bool:
    """Does this turn propose a (name, arguments) pair the episode has already proposed?

    A STRUCTURAL fact about the trajectory, not a judgement that repeating is wrong -- a legitimate
    retry after a violation is also a repeat, and deciding which repeats matter is the search's job.
    Arguments are compared as the raw JSON string the model emitted, because two spellings of one
    object are two different proposals as far as the harness is concerned.
    """
    if not calls or not seen:
        return False
    previous = {(str(n), str(a)) for n, a in seen}
    for call in calls:
        fn = call.get("function") or {}
        key = (str(fn.get("name") or ""), str(fn.get("arguments") or ""))
        if key in previous:
            return True
    return False


def call_keys(tool_calls: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    """The `(name, arguments)` keys of a turn's calls, for the wrapper's `seen_calls` history."""
    out = []
    for call in tool_calls or ():
        fn = call.get("function") or {}
        out.append((str(fn.get("name") or ""), str(fn.get("arguments") or "")))
    return tuple(out)
