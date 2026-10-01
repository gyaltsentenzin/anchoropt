#!/usr/bin/env python3
"""E1: is a proposed tool call PROVABLY uninformative, so that executing it is pure waste?

E stands for EFFICIENCY. A1-A4 raise accuracy; E1 lowers cost at identical accuracy. The objective
is accuracy per LLM call, and a 0 pp accuracy delta is a PASS for this class -- see
docs/EFFICIENCY_CLASS.md.

The decision is made PRE-EXECUTION -- post_generation_pre_execution. It needs the proposed call, so
pre_generation is structurally out; it must stop the call from running, so post_execution is out.
Exactly one admissible cell.

THE PATTERN

    known state + known action + deterministic tool  ->  reuse the recorded outcome

Not a heuristic that the call is probably unhelpful: a PROOF that it is uninformative. The tool is
deterministic, the state has not changed, and the model has already been told the answer. A second
identical call cannot return anything the first did not. It can only cost a step.

Instantiated for the one class whose determinism is verified against LIVE state: a removal whose
target is absent. Verified in the untouched control before this existed -- of 16 repeat-still-absent
removals, **0** later succeeded. Zero successful repeats is what licenses suppression; if any repeat
had succeeded the tool would not be deterministic for this class and the gate would block real work.

SUPPRESS, BUT THE OBSERVATION IS PRESERVED

E1 and A3 are both `suppress` at the same point, and they differ on what the model is left with:

    A3   call does not execute  +  call REMOVED from the record   -> model sees nothing
    E1   call does not execute  +  VERBATIM result REPLAYED       -> model's view is identical

Only the execution is saved. That is the whole acceptance argument: the trajectory outside the
withheld calls is identical, so +0.00 pp with ZERO FLIPS is the strongest available evidence rather
than a null result. The two rejected predecessors are the proof that the observation matters --
A5-v2 injected a user message (-1.65 pp) and A5-v3 substituted a NEW tool string (+0.33 pp but
destructive calls 84 -> 90). A5-v3 gained accuracy and was rejected: it said something new.

So: never author a string here. Replay what the tool itself returned, byte for byte.

FAIL OPEN, ALWAYS

Not a remove, unparseable target, no recorded outcome, unreadable live state -> MISS, and the tool
executes normally. A memoising gate that guesses is a correctness bug, not an efficiency win. The
cost of a wrong suppression is a legitimate removal silently dropped; the cost of a wrong miss is
one redundant call.

CALLER CONTRACT -- four invariants, each learned from a failure

    1  LOOK UP BEFORE EXECUTING, and execute the SHORTENED batch. The first implementation of this
       anchor memoised AFTER the executor ran and saved nothing -- a behaviour change dressed as an
       efficiency one. Verify with a counting stub that executor invocations for the targeted calls
       are 0. Fewer proposals is not a saving; only fewer EXECUTIONS is.
    2  SPLICE cached results back at their ORIGINAL INDEX. Delivery zips results against proposed
       calls POSITIONALLY, so a missing slot shifts every later result onto the wrong call. The
       length invariant is correctness, not bookkeeping.
    3  Replay VERBATIM (this module stores the tool's own string; do not post-process it).
    4  RECORD only from calls that ACTUALLY EXECUTED -- a withheld call has no fresh result, and
       recording from it would memoise the replay.

See docs/EFFICIENCY_CLASS.md for the acceptance class and rounds/E1_efficiency/ for the evidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

# The memoised action class. Deliberately narrow: only removals, because only their
# absent-target outcome has been shown deterministic against live state.
REMOVE = re.compile(r"\b(core|archival)_memory_remove\s*\(")
ADD = re.compile(r"\b(core|archival)_memory_add\s*\(")

# Target identifiers: a string key, or a vector id.
KEY_ARG = re.compile(r"key\s*=\s*['\"]([^'\"]*)['\"]")
VEC_ID = re.compile(r"vec_id\s*=\s*(\d+)")

# The native not-found errors, verbatim per backend:
#     kv       {"error": "Key not found."}
#     vector   {"error": "ID <n> not present in store."}
# Matching the ERROR STRING rather than a status code is deliberate -- it is what the model
# actually saw, and it is what will be replayed.
NOT_FOUND = re.compile(r"not present in store|Key not found", re.IGNORECASE)

MemoKey = tuple[str, str]


def container_of(call: object) -> str:
    """Which container the call addresses. Core and archival are separate namespaces."""
    return "core" if "core_memory" in str(call) else "archival"


def target_of(call: object) -> str | None:
    """The removal target: a quoted key, or a vector id. None when unparseable -> fail open."""
    m = KEY_ARG.search(str(call)) or VEC_ID.search(str(call))
    return m.group(1) if m else None


def is_remove(call: object) -> bool:
    return bool(REMOVE.search(str(call)))


def is_not_found(result: object) -> bool:
    return bool(NOT_FOUND.search(str(result)))


def live_container(instances: Any, which: str) -> dict[str, object] | None:
    """Read the CURRENT contents of a container from the live tool instances.

    Live state, never a reconstruction. Reconstructing state from a trajectory produced a wrong
    answer three separate times on this line before this anchor read the store directly; that is
    why determinism is re-confirmed here at every firing rather than trusted from the memo.

    Returns None when NO container could be read at all -- callers must treat that as "execute
    normally". An EMPTY dict is a valid, informative answer: the container is readable and holds
    nothing, so every target is absent.

    That distinction is deliberate and it is a fix, not a port detail. Conflating "unreadable"
    with "empty" (returning `{}` for both, then testing `if not live`) makes the guard fail open
    exactly when absence is most certain -- an empty container is the strongest possible evidence
    that the removal target is not there. The working repo's version has that conflation; it is
    harmless there only because its firing population never hit an empty container.
    """
    seen = False
    out: dict[str, object] = {}
    for inst in (instances or {}).values() if isinstance(instances, dict) else (instances or []):
        store = getattr(inst, f"{which}_memory", None)
        if store is None:
            continue
        # Two backend shapes: vector wraps its dict in `._store`, kv is a plain dict that
        # carries a `next_id` bookkeeping entry.
        inner = getattr(store, "_store", None)
        if isinstance(inner, dict):
            seen = True
            out.update({str(k): v for k, v in inner.items()})
        elif isinstance(store, dict):
            seen = True
            out.update({str(k): v for k, v in store.items() if k != "next_id"})
    return out if seen else None


def lookup(
    call: object,
    instances: Any,
    memo: Mapping[MemoKey, str],
) -> tuple[bool, str]:
    """(hit?, verbatim_result_or_miss_reason).

    A hit requires ALL FOUR:
      1. the call is a removal            (the only memoised action class)
      2. its target parses
      3. (container, target) has a RECORDED not-found outcome
      4. the target is STILL ABSENT from LIVE state, read now

    (3) says what to replay; (4) says the replay is still valid. Both are required -- a recorded
    outcome alone is a reconstruction, which is exactly the mistake this design avoids.

    Any uncertainty is a MISS, so the tool executes.
    """
    text = str(call)
    if not is_remove(text):
        return False, "not a memoised action class"

    which = container_of(text)
    target = target_of(text)
    if target is None:
        return False, "no target parsed -- execute normally"

    key = (which, target)
    if key not in memo:
        return False, "no recorded outcome -- execute normally"

    live = live_container(instances, which)
    if live is None:
        return False, "live container unreadable -- execute normally"
    if target in live:
        # The target came back. The recorded outcome no longer holds, so the caller must drop
        # this entry (see `invalidate`) and let the removal run -- it is now legitimate work.
        return False, "target present again -- entry stale"

    return True, memo[key]


def invalidate(call: object, instances: Any, memo: MutableMapping[MemoKey, str]) -> bool:
    """Drop a stale entry whose target has reappeared. Returns True if one was dropped.

    Separated from `lookup` so that `lookup` can take a read-only mapping: a predicate that
    mutates its input is a predicate that cannot be safely called twice, and this one is called
    once per proposed call per step.
    """
    text = str(call)
    if not is_remove(text):
        return False
    target = target_of(text)
    if target is None:
        return False
    key = (container_of(text), target)
    live = live_container(instances, container_of(text))
    if key in memo and live is not None and target in live:
        memo.pop(key, None)
        return True
    return False


def record(
    executed_calls: Sequence[object],
    executed_results: Sequence[object],
    memo: MutableMapping[MemoKey, str],
) -> int:
    """Memoise deterministic outcomes from calls that ACTUALLY EXECUTED. Returns entries written.

    Pass ONLY executed calls and their results (caller invariant 4). A withheld call has no fresh
    result; recording from it would memoise the replay of a replay.

    Records:      a removal that returned the native not-found error -> store the VERBATIM string.
    Invalidates:  a successful add of that target -> the entry exists again.

    KNOWN GAP: a container `clear` is not an explicit invalidation trigger. It is safe here only
    by DIRECTION -- clearing makes targets more absent, never less -- so a memoised not-found
    stays true. That is an argument about this tool set, not a general one. The live-state re-read
    in `lookup` is the actual safety net, and it covers clear too; this note exists so a port does
    not mistake the omission for an oversight.
    """
    written = 0
    for i, call in enumerate(executed_calls or []):
        text = str(call)
        result = str(executed_results[i]) if i < len(executed_results or []) else ""
        target = target_of(text)
        if target is None:
            continue
        key = (container_of(text), target)
        if is_remove(text) and is_not_found(result):
            memo[key] = result          # verbatim, so the replay is byte-identical
            written += 1
        elif ADD.search(text) and '"error"' not in result:
            memo.pop(key, None)         # the target exists again
    return written


def partition(
    proposed: Sequence[object],
    instances: Any,
    memo: MutableMapping[MemoKey, str],
) -> tuple[dict[int, str], list[object]]:
    """Split proposed calls into (withheld: index -> replay string, to_execute: list).

    The caller-facing entry point, and the one that makes invariants 1-2 hard to get wrong:

        withheld, to_execute = partition(proposed, instances, memo)
        results = execute(to_execute)            # <- SHORTENED batch; the saving lives here
        results = splice(withheld, results, len(proposed))
        record([c for i, c in enumerate(proposed) if i not in withheld], results_of_those, memo)

    Stale entries are invalidated as a side effect, so a target that reappeared is executed AND
    forgotten in one pass.
    """
    withheld: dict[int, str] = {}
    for i, call in enumerate(proposed or []):
        invalidate(call, instances, memo)
        hit, payload = lookup(call, instances, memo)
        if hit:
            withheld[i] = payload
    to_execute = [c for i, c in enumerate(proposed or []) if i not in withheld]
    return withheld, to_execute


def splice(withheld: Mapping[int, str], executed_results: Sequence[object], total: int) -> list[object]:
    """Reinsert withheld results at their ORIGINAL indices, restoring full length.

    Invariant 2, and it is correctness rather than tidiness: result delivery zips results against
    proposed calls POSITIONALLY, so one missing slot shifts every subsequent result onto the wrong
    call and mislabels the whole turn.

    Raises on a length mismatch. The working repo recorded this as a boolean flag and continued,
    which means a bad splice would be discovered only after the run; here it fails loudly.
    """
    it = iter(executed_results or [])
    out: list[object] = []
    for i in range(total):
        out.append(withheld[i] if i in withheld else next(it, ""))
    leftover = list(it)
    if leftover:
        raise ValueError(
            f"splice: {len(leftover)} executed result(s) left over -- "
            f"total={total}, withheld={len(withheld)}, executed={len(executed_results or [])}"
        )
    if len(out) != total:
        raise ValueError(f"splice: produced {len(out)} results for {total} proposed calls")
    return out
