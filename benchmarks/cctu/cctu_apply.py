"""Turning a semantic directive into an edit of the turn. Pure functions, no I/O, no model.

`cctu_adapter.apply_action` says WHAT must happen; this says how the turn changes. Kept out of
`response_generator.py` on purpose: the upstream file's diff stays readable, and every decision here
is testable with a dict and no served model.

    directive (semantic)  ->  Applied (an edit of this turn)  ->  response_generator applies it

WHAT A MECHANISM OWES, and why these are not one-liners
------------------------------------------------------
`anchoropt/mechanisms/constraint_repair.py` fixes the shape every repair on this project follows:

    detect the constraint -> read live state -> propose OR DECLINE -> verify the invariant -> retry

The middle two are the ones that are tempting to skip. A suppression that drops a call without
checking whether the violation actually clears has not repaired anything -- it has spent a call and
guessed. So `drop_call` re-asks upstream's own handlers after each removal and **declines** if the
targeted class does not clear, and `withhold_execution` declines unless it holds a verbatim recorded
result for the identical call.

THE FOUR INVARIANTS FROM THE EFFICIENCY CLASS, PORTED RATHER THAN PARAPHRASED
----------------------------------------------------------------------------
`docs/EFFICIENCY_CLASS.md` states them for the anchor that withholds a redundant call, and each one
was learned from a defect:

  1. LOOK UP BEFORE EXECUTING, and execute the SHORTENED batch. The first implementation memoised
     after the executor ran and saved nothing -- a behaviour change dressed as an efficiency one, and
     it passed every predicate test. Here `execute_ids()` returns the calls that will actually be
     dispatched, and `tests` must prove the executor is invoked 0 times for a withheld call.
     *Fewer proposals is not a saving; only fewer executions is.*
  2. SPLICE THE RESULT BACK AT ITS ORIGINAL INDEX. CCTU keys feedback by `tool_call_id` rather than
     by position, which is safer than BFCL's positional zip -- but message ORDER still matters, so
     `merge_feedback` rebuilds the list in the proposed calls' own order.
  3. REPLAY THE TOOL'S VERBATIM STRING. Never author a new one. The predecessor that substituted a
     *new* message gained +0.33 pp and was REJECTED, because destructive calls rose 84 -> 90.
  4. RECORD ONLY FROM CALLS THAT ACTUALLY EXECUTED.

WHICH CALL GETS DROPPED, AND WHY TRAILING
-----------------------------------------
A turn-level condition fires on the turn, not on one call, so something has to decide which call to
remove. Trailing, and that is principled rather than arbitrary: `CallTimesHandler` and
`MaxCallsPerToolHandler` count calls **in the order they appear** and flag each one that takes the
running total past the cap, so the calls in violation are exactly the trailing ones. Dropping from
the front would remove a call the harness never objected to.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cctu_state as state_mod


@dataclass(frozen=True)
class Applied:
    """One directive, resolved against this turn.

    `message` is the assistant message as the harness should now see it -- possibly with calls
    removed or arguments rewritten. `inject` is appended to the conversation. `redecide` asks the
    loop to discard this generation and generate again.

    `declined` is a first-class outcome, not an error. A mechanism that cannot establish its
    invariant must say so and change nothing, and the reason must reach the trace: a decline that
    looks like a no-op is indistinguishable from a controller that never fired.
    """

    message: Mapping[str, Any]
    kind: str
    detail: str = ""
    inject: tuple[Mapping[str, Any], ...] = ()
    redecide: bool = False
    withheld_ids: frozenset[str] = frozenset()
    substitutions: Mapping[str, str] = field(default_factory=dict)
    declined: bool = False
    charges_round: bool = False

    @property
    def changed_calls(self) -> bool:
        return self.kind in ("suppress", "reroute") and not self.declined


def apply_directive(directive: Mapping[str, Any], message: Mapping[str, Any], *,
                    snapshot: Any = None,
                    recorded_results: Mapping[tuple[str, str], str] | None = None,
                    tools_doc: Mapping[str, Any] | None = None) -> Applied:
    """Resolve a semantic directive into an edit of `message`.

    `snapshot` lets `drop_call` verify its invariant against upstream's own handlers.
    `recorded_results` lets `withhold_execution` replay a verbatim earlier result.
    `tools_doc` lets `reroute` know which argument names a tool actually declares.

    Anything this function cannot establish comes back `declined=True` with a reason, never as a
    silent no-op and never as an exception -- the caller is an episode loop that must not die.
    """
    kind = str(directive.get("kind") or "")
    if kind in ("", "noop"):
        return Applied(message=message, kind="noop", detail="nothing to apply")

    if kind == "reprompt":
        text = str(directive.get("text") or "").strip()
        if not text:
            return Applied(message=message, kind=kind, declined=True,
                           detail="reprompt directive carried no text")
        return Applied(
            message=message, kind=kind,
            inject=({"role": "user", "content": text},),
            redecide=bool(directive.get("request_redecision", False)),
            charges_round=bool(directive.get("charges_round", False)),
            detail="instruction injected" + (" and the turn re-decided"
                                             if directive.get("request_redecision") else ""))

    if kind == "suppress":
        if directive.get("keep_call"):
            return _withhold(directive, message, recorded_results or {})
        return _drop_calls(directive, message, snapshot)

    if kind == "reroute":
        return _reroute(directive, message, tools_doc or {})

    return Applied(message=message, kind=kind, declined=True,
                   detail=f"no application is defined for directive kind {kind!r}")


# ------------------------------------------------------------------------------------------------
# SUPPRESS / drop_call -- remove trailing calls, and verify the violation clears
# ------------------------------------------------------------------------------------------------
def _drop_calls(directive: Mapping[str, Any], message: Mapping[str, Any],
                snapshot: Any) -> Applied:
    """Remove trailing calls until the targeted violation clears, or DECLINE.

    Three ways this declines, and each is a real state rather than a defensive branch:

      * fewer than two calls -- removing the only call converts a tool turn into a final answer, and
        the terminal handlers would judge content written for a tool turn. `apply_action` already
        refuses this, so reaching it here means the state disagreed with the directive.
      * a target class was named and no removal clears it -- the mechanism does not repair this turn,
        so spending a call on it would be a guess dressed as a repair.
      * a target class was named and there is no snapshot to check against -- the invariant is
        unverifiable, which is NOT the same as satisfied.

    With no `target_violation` named, exactly one trailing call is dropped and nothing is verified,
    which is honest: there is no invariant to check because none was declared.
    """
    calls = list(message.get("tool_calls") or ())
    if len(calls) < 2:
        return Applied(message=message, kind="suppress", declined=True,
                       detail=f"only {len(calls)} call proposed; dropping it would convert a tool "
                              f"turn into a final answer")

    target = directive.get("target_violation")
    content = str(message.get("content") or "")

    if target is None:
        kept = calls[:-1]
        return Applied(message=dict(message, tool_calls=kept), kind="suppress",
                       detail=f"dropped 1 trailing call ({len(calls)} -> {len(kept)}); no "
                              f"target_violation declared, so no invariant was verified")

    if snapshot is None:
        return Applied(message=message, kind="suppress", declined=True,
                       detail=f"target_violation={target!r} was declared but no live state was "
                              f"supplied, so the invariant cannot be verified; unverifiable is not "
                              f"satisfied")

    target = str(target)
    if target not in state_mod.would_violate(snapshot, tool_calls=calls, content=content):
        return Applied(message=message, kind="suppress", declined=True,
                       detail=f"target_violation={target!r} is not predicted for this turn; "
                              f"nothing to repair")

    kept = list(calls)
    while len(kept) > 1:
        kept.pop()
        if target not in state_mod.would_violate(snapshot, tool_calls=kept, content=content):
            return Applied(
                message=dict(message, tool_calls=kept), kind="suppress",
                detail=f"dropped {len(calls) - len(kept)} trailing call(s) ({len(calls)} -> "
                       f"{len(kept)}); {target} verified clear against the live constraint state")
    return Applied(message=message, kind="suppress", declined=True,
                   detail=f"no number of trailing removals clears {target} while keeping at least "
                          f"one call; declining rather than spending a call on a guess")


# ------------------------------------------------------------------------------------------------
# SUPPRESS / withhold_execution -- the efficiency shape
# ------------------------------------------------------------------------------------------------
def _withhold(directive: Mapping[str, Any], message: Mapping[str, Any],
              recorded: Mapping[tuple[str, str], str]) -> Applied:
    """Withhold execution of calls whose result is already known, replaying the verbatim string.

    The proof obligation, not a heuristic: a call is withheld only when THIS episode has already
    executed a byte-identical `(name, arguments)` pair and kept what it returned. That is
    `docs/EFFICIENCY_CLASS.md`'s pattern -- *known state + known action + deterministic tool -> reuse
    the recorded outcome* -- and it is a proof that the call is uninformative rather than a guess that
    it is unhelpful.

    The call STAYS in the message. That is what keeps the turn's kind and the validator's budget
    unchanged, so the arm is observationally inert outside the withheld executions -- which is the
    claim an efficiency anchor makes and the reason 0 pp is a pass for it.

    A WARNING THIS INHEthe hosted API: CCTU's tools are `exec`'d per episode and wrapped in a 10-second
    wall-clock timeout, so "deterministic" is an assumption about the corpus's tool code, not a
    property of the harness. If a tool reads the clock or a counter, replaying its result changes
    behaviour. The zero-flip bar this class is judged against is what would catch that -- and it
    depends on deterministic decoding, which under sampling makes the bar unobtainable.
    """
    calls = list(message.get("tool_calls") or ())
    withheld: dict[str, str] = {}
    for call in calls:
        fn = call.get("function") or {}
        key = (str(fn.get("name") or ""), str(fn.get("arguments") or ""))
        if key in recorded:
            cid = str(call.get("id") or "")
            if cid:
                withheld[cid] = recorded[key]
    if not withheld:
        return Applied(message=message, kind="suppress", declined=True,
                       detail="no proposed call repeats one this episode already executed, so no "
                              "recorded result can be replayed verbatim")
    return Applied(message=message, kind="suppress", withheld_ids=frozenset(withheld),
                   substitutions=dict(withheld), charges_round=False,
                   detail=f"withheld execution of {len(withheld)} call(s) whose verbatim result this "
                          f"episode already recorded; the calls stay in the message")


# ------------------------------------------------------------------------------------------------
# REROUTE -- mechanical argument repair
# ------------------------------------------------------------------------------------------------
def _reroute(directive: Mapping[str, Any], message: Mapping[str, Any],
             tools_doc: Mapping[str, Any]) -> Applied:
    """Rewrite the proposed calls' arguments. Deletion is derived from the schema, never guessed.

    `drop_args` may name argument names explicitly, but the useful form is empty: with no names given
    and a `tools_doc` available, the keys deleted are exactly those the tool's own schema does not
    declare -- the same set `ToolArgsChecker` reports as "extra argument(s)". Nothing is synthesized,
    so this cannot build an invalid call out of a valid one.

    It declines when there is nothing to delete and no patch to apply, rather than reporting a
    successful rewrite that changed nothing -- which is the `executed=True` on an empty payload shape
    that `docs/CONSUMER_BOUNDARY_RULE.md` exists to prevent.
    """
    import json

    calls = list(message.get("tool_calls") or ())
    named = tuple(directive.get("drop_args") or ())
    patch = dict(directive.get("args_patch") or {})
    derive = bool(directive.get("drop_unknown_args", False))
    out: list[dict[str, Any]] = []
    edits = 0

    for call in calls:
        fn = dict(call.get("function") or {})
        name = str(fn.get("name") or "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = None
        if not isinstance(args, dict):
            # An unparseable argument string, or one that is not an object, cannot be repaired by
            # deletion. Left untouched so the validator reports it, rather than replaced with
            # something invented.
            out.append(dict(call))
            continue

        declared = set((tools_doc.get(name) or {}).get("properties") or ())
        # Explicit names win; otherwise derive from the schema when asked to. With neither,
        # nothing is deleted and the caller gets a decline rather than a silent no-op.
        if named:
            remove = set(named)
        elif derive and declared:
            remove = set(args) - declared
        else:
            remove = set()
        new_args = {k: v for k, v in args.items() if k not in remove}
        new_args.update(patch)
        if new_args != args:
            edits += 1
            fn["arguments"] = json.dumps(new_args, ensure_ascii=False)
            out.append(dict(call, function=fn))
        else:
            out.append(dict(call))

    if not edits:
        return Applied(message=message, kind="reroute", declined=True,
                       detail="no argument would change: nothing the schema rejects and no patch")
    return Applied(message=dict(message, tool_calls=out), kind="reroute",
                   detail=f"repaired arguments on {edits} of {len(calls)} call(s) by deletion"
                          + (f" plus {len(patch)} patched key(s)" if patch else ""))


# ------------------------------------------------------------------------------------------------
# Execution planning -- invariant 1, made checkable
# ------------------------------------------------------------------------------------------------
def execute_ids(calls: Sequence[Mapping[str, Any]], withheld: frozenset[str]) -> list[dict]:
    """The calls that will ACTUALLY be dispatched -- the shortened batch.

    Exists as its own function so a test can assert the executor is invoked zero times for a withheld
    call. `docs/EFFICIENCY_CLASS.md`: the first implementation of this class memoised *after* the
    executor ran and saved nothing while passing every predicate test. A counting stub over this
    boundary is the only thing that distinguishes a saving from a behaviour change.
    """
    return [dict(c) for c in calls if str(c.get("id") or "") not in withheld]


def merge_feedback(calls: Sequence[Mapping[str, Any]],
                   executed_feedback: Sequence[Mapping[str, Any]],
                   substitutions: Mapping[str, str]) -> list[dict]:
    """Rebuild the feedback list in the PROPOSED calls' own order, splicing substitutions back in.

    CCTU delivers results keyed by `tool_call_id`, so a missing slot cannot shift a result onto the
    wrong call the way BFCL's positional zip could. Order still matters for the transcript, and a
    substituted observation must land where its call was -- so the list is rebuilt from the calls
    rather than concatenated.

    A substituted string is the tool's own recorded output, passed through unchanged. Authoring a
    replacement is what got the predecessor of this mechanism rejected.
    """
    by_id: dict[str, list[Mapping[str, Any]]] = {}
    extra: list[Mapping[str, Any]] = []
    for msg in executed_feedback or ():
        cid = str(msg.get("tool_call_id") or "")
        if cid:
            by_id.setdefault(cid, []).append(msg)
        else:
            extra.append(msg)

    out: list[dict] = []
    for call in calls or ():
        cid = str(call.get("id") or "")
        if cid in substitutions:
            out.append({"role": "tool", "tool_call_id": cid, "content": substitutions[cid]})
            continue
        for msg in by_id.get(cid, ()):
            out.append(dict(msg))
    # Anything not keyed to a call (a merged user-role violation message) keeps its place at the end,
    # which is where upstream puts it.
    out.extend(dict(m) for m in extra)
    return out
