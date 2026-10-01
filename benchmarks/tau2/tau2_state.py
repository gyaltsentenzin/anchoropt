"""Observable-state construction, ONE implementation used by two callers.

WHY THIS FILE EXISTS SEPARATELY. The live mechanism builds a state inside `generate_next_message` to
decide whether a controller fires; the offline reader builds states from a recorded `SimulationRun` so
core can localize and expand over them. If those were two implementations they would drift, and the
drift shows up as a controller that discriminates beautifully on mined states and never fires in the
run -- the defect class this contract is built around. So both go through the builders here, which take
plain data and return plain dicts, and neither imports tau2.

Every key produced here is declared in `tau2_fields.FIELDS_AT` at exactly the locus that produces it.
`tests/test_tau2_boundary_observability.py` asserts the two agree in both directions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from anchoropt.anchor import IncisionPoint

_PRE = IncisionPoint.PRE_GENERATION.value
_PG = IncisionPoint.POST_GENERATION_PRE_EXEC.value
_PE = IncisionPoint.POST_EXECUTION.value

# Classification of the runtime's own error text. Read off real `Environment.get_response` output
# (`Error: {exception}`), not invented: the four shapes below were sampled from the airline domain.
#
# THIS IS AN OBSERVABLE, NOT A RECOMMENDATION. It reports what the runtime said. Nothing here maps an
# error to a remedy -- that mapping is core's to learn, and putting it in the adapter would invalidate
# the experiment.
_ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tool_not_found", re.compile(r"Tool '[^']*' not found", re.I)),
    ("invalid_arguments", re.compile(
        r"(missing \d+ required positional argument|unexpected keyword argument"
        r"|validation error|invalid literal|takes \d+ positional)", re.I)),
    ("not_found", re.compile(r"\bnot found\b", re.I)),
)


def classify_error(content: Any, is_error: bool) -> str:
    """The runtime's error class for one returned result. '' when the result is not an error."""
    if not is_error:
        return ""
    text = "" if content is None else str(content)
    for kind, pat in _ERROR_PATTERNS:
        if pat.search(text):
            return kind
    return "domain_error"


def _args_chars(arguments: Mapping[str, Any] | None) -> int:
    if not arguments:
        return 0
    try:
        return len(json.dumps(arguments, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return len(str(arguments))


# ------------------------------------------------------------------------------------------------
# PRE_GENERATION -- the turn has begun, no candidate exists.
# ------------------------------------------------------------------------------------------------
def turn_start_state(*, case_id: str, turn_index: int, assistant_turns_so_far: int,
                     user_turns_so_far: int, inbound_is_tool_result: bool,
                     consecutive_tool_errors: int, tool_errors_so_far: int) -> dict[str, Any]:
    return {
        "boundary": _PRE,
        "case_id": str(case_id),
        "turn_index": int(turn_index),
        "assistant_turns_so_far": int(assistant_turns_so_far),
        "user_turns_so_far": int(user_turns_so_far),
        "inbound_is_tool_result": bool(inbound_is_tool_result),
        "consecutive_tool_errors": int(consecutive_tool_errors),
        "tool_errors_so_far": int(tool_errors_so_far),
    }


# ------------------------------------------------------------------------------------------------
# POST_GENERATION_PRE_EXEC -- the candidate exists; the orchestrator has not seen it.
# ------------------------------------------------------------------------------------------------
def gate_state(*, case_id: str, turn_index: int, proposed_calls: Sequence[Mapping[str, Any]],
               has_content: bool, known_tools: frozenset[str] | set[str],
               mutating_tools: frozenset[str] | set[str], replan_attempt: int,
               consecutive_tool_errors: int, tool_errors_so_far: int) -> dict[str, Any]:
    """One gate state. `proposed_calls` are {'name','arguments'} dicts in the candidate's order.

    `commits_to_reply` is the ANSWER-COMMITMENT decision and it must survive into the mined events:
    dropping it deletes the boundary the backward search needs, and localization then derives exactly
    one boundary (ADAPTER_GUIDE section 9).
    """
    calls = list(proposed_calls or ())
    first = calls[0] if calls else {}
    name = str(first.get("name") or "")
    args = first.get("arguments") or {}
    return {
        "boundary": _PG,
        "case_id": str(case_id),
        "turn_index": int(turn_index),
        "proposes_tool_call": bool(calls),
        "commits_to_reply": (not calls) and bool(has_content),
        "n_proposed_calls": len(calls),
        "proposed_tool": name,
        "proposed_tool_mutates_state": bool(name) and name in set(mutating_tools),
        "proposed_tool_is_known": bool(name) and name in set(known_tools),
        "n_proposed_args": len(args) if isinstance(args, Mapping) else 0,
        "proposed_args_chars": _args_chars(args if isinstance(args, Mapping) else None),
        "replan_attempt": int(replan_attempt),
        "consecutive_tool_errors": int(consecutive_tool_errors),
        "tool_errors_so_far": int(tool_errors_so_far),
    }


# ------------------------------------------------------------------------------------------------
# POST_EXECUTION -- a result came back. The world has moved.
# ------------------------------------------------------------------------------------------------
def result_state(*, case_id: str, turn_index: int, is_error: bool, content: Any,
                 tool_name: str, mutating_tools: frozenset[str] | set[str],
                 tool_errors_so_far: int) -> dict[str, Any]:
    name = str(tool_name or "")
    return {
        "boundary": _PE,
        "case_id": str(case_id),
        "turn_index": int(turn_index),
        "result_is_error": bool(is_error),
        "error_kind": classify_error(content, bool(is_error)),
        "result_chars": 0 if content is None else len(str(content)),
        "result_tool": name,
        "result_tool_mutates_state": bool(name) and name in set(mutating_tools),
        "tool_errors_so_far": int(tool_errors_so_far),
    }
