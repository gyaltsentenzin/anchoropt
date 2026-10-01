"""TerminalBench 2 / DeepAgents(LangChain) adapter -- supplies itself TO the core.

Same shape as `benchmarks/bfcl_v4/adapter.py`: module-level functions are the interface, a frozen
dataclass record exists for callers preferring injection, and `register()` installs it. Duck-typed,
no ABC -- the registry in `anchoropt/attribution/__init__.py` declines to pin one on purpose, and
this is the second adapter it said it was waiting for.

WHAT THIS ADAPTER OWNS, THAT THE CORE MUST NEVER SEE
----------------------------------------------------
  * LangChain specifics. `after_model`, `AIMessage`, `tool_calls`, `jump_to="model"` appear here
    and nowhere in `anchoropt/`. `jump_to="model"` is this runtime's IMPLEMENTATION of the generic
    semantic action REPROMPT -- request another model decision -- not a concept of its own.
  * TB2 task ids. `mailman`, `qemu-startup` etc. are opaque `case_id` tokens to the core.
  * Phi_TB2, in `signals.py`.

THE BOUNDARY MAPPING -- NO FOURTH BOUNDARY
------------------------------------------
LangChain's `after_model` hook fires between the model node and the routing decision
(`graph.add_edge("model", "<mw>.after_model")`, langchain 1.3.0 `agents/factory.py:1623`), so the
generated AIMessage exists and nothing has executed or been returned. That is precisely the
existing `POST_GENERATION_PRE_EXEC` incision point, which covers BOTH sub-cases:

    POST_GENERATION_PRE_EXEC
        |-- proposed tool action      tool_calls present
        |-- proposed terminal response  len(tool_calls) == 0   <- LangChain's own loop-exit test

`AnchorRuntime.post_generation_pre_exec(tool_name, tool_args)` in an earlier internal TB2
*prototype* accepted only tool calls, which is why the terminal case looked unobservable. That was
a prototype limitation, not a runtime one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.runtime import HostProfile

from tb2_signals import SIGNAL_ALIASES, SIGNAL_PARAMS, SIGNALS  # noqa: E402

NAME = "tb2_deepagents"

# ------------------------------------------------------------------------------------------------
# U_H(l) -- what THIS runtime can execute where
# ------------------------------------------------------------------------------------------------
#
# Narrower than the structural grid, and the narrowing is measured rather than assumed. Recorded in
# an earlier internal action-space matrix: every cell was driven with a real anchor and a real
# payload, and TRANSFORM-shaped observation rewriting only worked where a tool RESULT exists.
#
# REROUTE is declared at POST_GENERATION_PRE_EXEC only. Its `substitute` reading (rewrite the
# proposed call) is executable there. Its `transform` reading needs a returned observation, so it
# belongs at POST_EXECUTION -- deliberately NOT declared until a boundary-specific implementation
# exists, because the prototype's version reported executed=True while appending its payload to the
# empty string. See docs/CONSUMER_BOUNDARY_RULE.md.
HOST = HostProfile(
    name=NAME,
    executable={
        IncisionPoint.PRE_GENERATION: frozenset({Action.NOOP, Action.REPROMPT}),
        IncisionPoint.POST_GENERATION_PRE_EXEC: frozenset({
            Action.NOOP, Action.REPROMPT, Action.SUPPRESS, Action.REROUTE,
        }),
        IncisionPoint.POST_EXECUTION: frozenset({Action.NOOP}),
    },
    notes=("REROUTE-as-transform at POST_EXECUTION is withheld pending boundary-specific "
           "semantics; the prototype implementation read a field absent off POST_EXECUTION and "
           "reported success anyway."),
)

_EXPLORE_PREFIXES = ("ls", "find", "grep", "cat", "head", "tail", "tree", "which", "locate")


# ------------------------------------------------------------------------------------------------
# 1. trajectory / event normalization
# ------------------------------------------------------------------------------------------------
def normalize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """One host event -> a boundary-tagged, framework-neutral record.

    Accepts either a LangChain-shaped payload (`messages`, the last of which may carry
    `tool_calls`) or an already-flat record. The core consumes only the returned keys, so no
    LangChain type crosses this function.
    """
    boundary = event.get("boundary")
    ai = event.get("ai_message")
    messages: Sequence[Any] = event.get("messages") or ()

    if ai is None and messages:
        ai = messages[-1]

    tool_calls = _tool_calls_of(ai)
    has_generation = ai is not None

    rec: dict[str, Any] = {
        "boundary": boundary,
        "has_generation": has_generation,
        "proposes_tool_call": bool(tool_calls),
        "n_tool_calls": len(tool_calls),
        "message_count": len(messages),
    }
    if tool_calls:
        first = tool_calls[0]
        rec["tool_name"] = _get(first, "name") or ""
        rec["tool_args"] = _get(first, "args") or {}
        rec["proposed_command"] = _command_of(rec["tool_args"])
    else:
        rec["tool_name"] = ""
        rec["tool_args"] = {}
        rec["proposed_command"] = ""
    if "output" in event:
        rec["output"] = event.get("output") or ""
    return rec


def _tool_calls_of(ai: Any) -> list[Any]:
    if ai is None:
        return []
    calls = _get(ai, "tool_calls")
    return list(calls) if calls else []


def _get(obj: Any, key: str) -> Any:
    """Attribute-or-key access, so a LangChain object and a plain dict both work.

    Both shapes occur: live middleware hands over objects, recorded transcripts are JSON.
    """
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _command_of(tool_args: Mapping[str, Any]) -> str:
    for k in ("command", "cmd", "shell_command", "input"):
        v = (tool_args or {}).get(k)
        if isinstance(v, str) and v:
            return v
    return ""


# ------------------------------------------------------------------------------------------------
# 2. observable runtime state
# ------------------------------------------------------------------------------------------------
def observable_state(normalized: Mapping[str, Any],
                     carried: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The state Phi_TB2 may read at this boundary, plus any state carried forward.

    `carried` is explicit rather than implicit: a signal needing evidence from an earlier boundary
    must have it PASSED here, so cross-boundary dependence is visible in the call rather than
    hidden in a mutable accumulator.
    """
    state = dict(normalized)
    if carried:
        state.update(carried)
    state.setdefault("explore_streak", 0)
    return state


def is_exploratory(command: str) -> bool:
    """Read-only/exploratory shell command -- feeds `explore_streak`."""
    head = (command or "").strip().split()
    return bool(head) and head[0] in _EXPLORE_PREFIXES


# ------------------------------------------------------------------------------------------------
# 3. feasible actions -- U_H(l)
# ------------------------------------------------------------------------------------------------
def feasible_actions(boundary: IncisionPoint) -> frozenset[Action]:
    """What this runtime can execute at `boundary`. Never wider than the structural grid."""
    return HOST.executable_actions(boundary)


# ------------------------------------------------------------------------------------------------
# 4. signal evaluation
# ------------------------------------------------------------------------------------------------
def evaluate_signal(signal: str, state: Mapping[str, Any],
                    params: Mapping[str, Any] | None = None) -> bool:
    """Evaluate one Phi_TB2 signal. Unknown names RAISE rather than returning False.

    A missing evaluator that returns False is indistinguishable from a signal that did not fire,
    which is how an arm can silently become the control arm. The prototype had exactly this defect
    (`tool_args` declared with no evaluator) and it killed a run mid-cluster.
    """
    if signal not in SIGNALS:
        raise KeyError(
            f"{NAME}: no evaluator for signal {signal!r}; declared: {sorted(SIGNALS)}"
        )
    params = dict(params or {})
    schema = SIGNAL_PARAMS.get(signal, {})
    for key, (typ, required) in schema.items():
        if key not in params:
            if required:
                raise ValueError(f"{NAME}: signal {signal!r} requires param {key!r}")
            continue
        if typ is int and isinstance(params[key], bool):
            raise TypeError(f"{NAME}: signal {signal!r} param {key!r} must be int, got bool")
        if not isinstance(params[key], typ):
            raise TypeError(
                f"{NAME}: signal {signal!r} param {key!r} must be {typ.__name__}, "
                f"got {type(params[key]).__name__}"
            )
    unknown = set(params) - set(schema)
    if unknown:
        raise ValueError(f"{NAME}: signal {signal!r} unknown param(s) {sorted(unknown)}")
    return bool(SIGNALS[signal](state, params))


def declared_signals() -> tuple[str, ...]:
    return tuple(sorted(SIGNALS))


def signal_aliases(signal: str) -> tuple[str, ...]:
    """Natural-language phrases that describe `signal`, for expressibility matching only.

    Part of this benchmark's declared vocabulary, so the generic core never hard-codes any
    domain words -- it asks the runtime. Returns () for a signal with no declared aliases.
    """
    return tuple(SIGNAL_ALIASES.get(signal, ()))


def register_signal(name: str, fn, params: Mapping[str, tuple[type, bool]] | None = None) -> None:
    """Register a signal accepted by the expansion step.

    Adding a signal touches NO action implementation -- one of the integration's generality
    requirements. Refuses to overwrite: silently replacing a frozen signal would invalidate every
    measurement taken under the old definition.
    """
    if name in SIGNALS:
        raise ValueError(f"{NAME}: signal {name!r} already registered; frozen definitions "
                         f"must not be replaced")
    SIGNALS[name] = fn                                    # type: ignore[index]
    SIGNAL_PARAMS[name] = dict(params or {})              # type: ignore[index]


# ------------------------------------------------------------------------------------------------
# 5. applying a semantic action
# ------------------------------------------------------------------------------------------------
def apply_action(action: Action, boundary: IncisionPoint, theta: Mapping[str, Any],
                 state: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a SEMANTIC action into this runtime's directive.

    Returns a plain SEMANTIC dict the host middleware executes; this function performs no I/O, so
    it is testable without a live agent. It names WHAT must happen, never the channel: the
    `jump_to="model"` mechanism and the choice between prompt-append and conversation-message live
    in `tb2_middleware.py`.

    Raises `ActionNotExecutable` (via `HOST.require`) when the host cannot perform the action at
    the boundary -- BEFORE any payload is built, so an unexecutable cell can never report success.
    """
    HOST.require(boundary, action)

    if action is Action.NOOP:
        return {"kind": "noop", "executed": False}

    if action is Action.REPROMPT:
        text = str(theta.get("text", "")).strip()
        if not text:
            raise ValueError("REPROMPT requires non-empty theta['text']")
        # SEMANTIC, not channel-specific. `text` is what the model must see; `request_redecision`
        # says another model decision is required. WHICH channel carries the text is the
        # middleware's business, and the two differ by boundary:
        #   PRE_GENERATION            the next generation has not happened -> append to the prompt
        #   POST_GENERATION_PRE_EXEC  a decision already exists -> add a message and re-decide
        # The earlier key was named `append_to_system_message` at both boundaries, which
        # misdescribed the POST_GENERATION channel that the live smoke test actually used.
        return {"kind": "reprompt", "executed": True, "text": text,
                "request_redecision": boundary is IncisionPoint.POST_GENERATION_PRE_EXEC}

    if action is Action.SUPPRESS:
        reason = str(theta.get("reason", "")).strip()
        if not reason:
            raise ValueError("SUPPRESS requires non-empty theta['reason']")
        return {"kind": "suppress", "executed": True, "reason": reason,
                "replacement": theta.get("replacement"), "short_circuit": True}

    if action is Action.REROUTE:
        if not (theta.get("tool_name") or theta.get("args_patch")):
            raise ValueError("REROUTE requires theta['tool_name'] and/or theta['args_patch']")
        args = dict(state.get("tool_args") or {})
        args.update(dict(theta.get("args_patch") or {}))
        return {"kind": "reroute", "executed": True,
                "tool_name": theta.get("tool_name") or state.get("tool_name"),
                "tool_args": args, "reason": str(theta.get("reason", ""))}

    raise AssertionError(f"unhandled action {action!r}")


# ------------------------------------------------------------------------------------------------
# The adapter record + registration
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TB2Adapter:
    """The TB2/DeepAgents runtime vocabulary as one object, for injection-style callers."""

    name: str = NAME
    host: HostProfile = HOST

    normalize_event = staticmethod(normalize_event)
    observable_state = staticmethod(observable_state)
    feasible_actions = staticmethod(feasible_actions)
    evaluate_signal = staticmethod(evaluate_signal)
    apply_action = staticmethod(apply_action)
    declared_signals = staticmethod(declared_signals)
    register_signal = staticmethod(register_signal)
    is_exploratory = staticmethod(is_exploratory)


ADAPTER = TB2Adapter()


def register() -> None:
    """Install this adapter as the core's active vocabulary. Idempotent.

    Registering the MODULE, not `ADAPTER`, matching the BFCL adapter: module-level exception types
    and helpers are part of the interface and a dataclass instance would not expose them.
    """
    import sys as _sys

    from anchoropt.attribution import register_adapter
    register_adapter(_sys.modules[__name__])
