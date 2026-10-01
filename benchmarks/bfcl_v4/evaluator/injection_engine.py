"""
AnchorOpt injection engine.

Combines state_extractor + tool_ranker to produce a compact context string
at each injection slot (pre-query anchor at step_n==0, post-step delta at step_n>=1).

No BFCL imports — pure Python.
"""

import json
import re
from pathlib import Path

from .state_extractor import delta as compute_delta, snapshot as _snapshot_fn
from .tool_ranker import _is_error, rank_tools, is_low_groundedness

# ── Runtime signal classification ──────────────────────────────────────────────
# Classifies the *previous* step's execution result into a coarse failure signal,
# using only the result string — no BFCL category knowledge. BFCL builds these
# strings in multi_turn_utils.py:107-108 as f"Error during execution: {str(e)}",
# so the discriminating text is the suffix (the raw Python exception message).
# Domain errors (e.g. file-not-found) return a JSON {"error": ...} dict instead
# and never carry the "Error during execution:" prefix.

SIG_HALLUCINATED      = "last_call_hallucinated"      # called a function name that does not exist
SIG_WRONG_ARGS        = "last_call_wrong_args"        # real function, bad/missing argument
SIG_DOMAIN_ERROR      = "last_call_domain_error"      # real call, valid args, function reported an error
SIG_REPEATED_SUCCESS  = "last_call_repeated_success"  # same (tool, args) called again on a non-error result
SIG_OVER_EXECUTED     = "last_turn_over_executed"     # prior turn made more calls than the task required
SIG_LOW_GROUNDEDNESS  = "low_groundedness"            # no top-K tool has all required params grounded in context

# Memory-specific domain-error sub-signals (v4 memory tasks).
# These narrow SIG_DOMAIN_ERROR by matching exact substrings from memory API error payloads.
SIG_DOMAIN_ERROR_CORE_FULL      = "last_call_domain_error_core_full"       # core/archival full → use archival
SIG_DOMAIN_ERROR_KEY_NOT_FOUND  = "last_call_domain_error_key_not_found"   # key lookup miss → list_keys fallback
SIG_DOMAIN_ERROR_TOO_LONG       = "last_call_domain_error_too_long"        # blob overflow → compact + re-append

# Successful retrieval signal — fires when a memory retrieval call returned a non-empty result.
# Used by on_retrieval_success_pre_answer to anchor the model before it commits to an answer.
SIG_RETRIEVAL_SUCCESS           = "last_call_retrieval_success"


def classify_last_result(last_result: str) -> str | None:
    """Map a tool-execution result string to one failure signal, or None.

    None means "no actionable failure signal" (success, empty, or unrecognized).
    Order matters: hallucination and wrong-args share the "Error during execution:"
    prefix and are disambiguated on the suffix; domain errors are checked separately
    because they do not carry that prefix.

    Note: SIG_REPEATED_SUCCESS cannot be detected from last_result alone — it
    requires call-history context. The evaluator sets it via the `loop_signal`
    parameter to get_injection() when it detects an identical (tool, args) repeat.
    """
    if not last_result:
        return None
    s = last_result

    # Hallucinated function name → NameError → "name '<NAME>' is not defined".
    # "is not allowed" is BFCL's blocked-builtin guard — also a wrong-target signal.
    if "is not defined" in s or "is not allowed" in s:
        return SIG_HALLUCINATED

    # Wrong/missing arguments → TypeError from Python's argument binding.
    if "unexpected keyword argument" in s or (
        "missing" in s and "required positional argument" in s
    ) or "positional argument" in s:
        return SIG_WRONG_ARGS

    # Domain error returned as a dict (not raised) — function exists and bound fine.
    # Sub-classify memory-specific errors before falling through to the generic bucket.
    if s.lstrip().startswith('{"error"') or "No such file or directory" in s or "No such directory" in s:
        if "is full" in s or "exceeds maximum size" in s or "Long term memory is full" in s or "Core memory is full" in s:
            return SIG_DOMAIN_ERROR_CORE_FULL
        if "Key not found." in s or "key not found" in s.lower():
            return SIG_DOMAIN_ERROR_KEY_NOT_FOUND
        if ("too long after appending" in s or "too long after updating" in s
                or "Entry is too long" in s or "Entry length exceeds" in s):
            return SIG_DOMAIN_ERROR_TOO_LONG
        return SIG_DOMAIN_ERROR

    return None


def is_retrieval_success(last_result: str) -> bool:
    """Return True when the last tool call was a successful memory retrieval.

    Matches result payloads from memory_kv (value/keys), memory_vector
    (similarity_score/text/ranked_results), and memory_rec_sum (memory_content).
    Excludes error dicts and empty-result payloads so the signal only fires when
    actual content was retrieved — not when the model called list_keys and got [].
    """
    if not last_result:
        return False
    if '"error"' in last_result:
        return False
    indicators = ('"value":', '"memory_content":', '"similarity_score":',
                  '"ranked_results":', '"keys":', '"text":')
    if not any(m in last_result for m in indicators):
        return False
    # Exclude empty-payload results that look like successes but contain nothing useful.
    empty_patterns = ('"keys": []', '"ranked_results": []', '"result": []', '"value": null', '"memory_content": ""')
    return not any(p in last_result for p in empty_patterns)


# ── Public API ────────────────────────────────────────────────────────────────

def load_policy(path) -> dict:
    with open(path) as f:
        return json.load(f)


# All-silent policy — every cell returns None (no injection). Used as the
# "no-injection baseline" arm to isolate AnchorOpt's contribution from the
# model's raw performance.
NULL_POLICY: dict = {"cells": {}, "rescue": {"streak_threshold": 9999, "max_tools": 0, "conditions": []}, "weights": {"state": 1.0, "error": 1.5, "recency": 0.25}}


def get_injection(
    turn_k: int,
    step_n: int,
    failure_streak: int,
    prev_snapshot,          # dict from state_extractor.snapshot(), or None (first step ever)
    curr_snapshot: dict,    # dict from state_extractor.snapshot()
    functions: list,        # BFCL function dicts
    recent_tools: list,     # tool names called in recent steps (most-recent last)
    last_result: str,       # raw string from last tool execution ("" if pre-query)
    policy: dict,
    error_streak: int = 0,              # consecutive steps with the same failure signal
    loop_signal: bool = False,          # evaluator sets True when (tool, args) just repeated on success
    user_message: str = "",             # current turn's user message (for plan mode + state relevance)
    over_execution_signal: bool = False, # evaluator sets True when prior turn made too many calls
    low_groundedness_signal: bool = False, # evaluator sets True when no top-K tool is param-grounded
) -> str | None:
    """
    Return a [Context] string to inject as a user message, or None if no injection.

    step_n == 0  → pre-query anchor slot (grounding before the model's first move this turn)
    step_n >= 1  → post-execution slot (delta of what the last step changed)
    """
    rescue_cfg = policy.get("rescue", {})
    streak_threshold = rescue_cfg.get("streak_threshold", 2)
    # Merge result-based signal with the evaluator-set flags.
    # Priority: loop > over_execution > low_groundedness > result-string signal.
    # low_groundedness only fires at step_n==0 (turn-start pre-action check).
    if loop_signal:
        signal = SIG_REPEATED_SUCCESS
    elif over_execution_signal and step_n == 0:
        signal = SIG_OVER_EXECUTED
    elif low_groundedness_signal and step_n == 0:
        signal = SIG_LOW_GROUNDEDNESS
    else:
        signal = classify_last_result(last_result)

    # 1. Rescue override — fires at any slot when model is stuck.
    if failure_streak >= streak_threshold:
        # A rescue condition may redirect to a signal-specific response (e.g.
        # tool_list on hallucination) instead of the default rescue render.
        rmode, rtools = _resolve_conditions(rescue_cfg, signal, error_streak)
        if rmode is not None:
            return _dispatch_mode(
                rmode, rtools, prev_snapshot, curr_snapshot, functions,
                recent_tools, last_result, policy.get("weights"),
                step_n=step_n, user_message=user_message,
            )
        return _render_rescue(
            curr_snapshot,
            last_result,
            functions,
            recent_tools,
            rescue_cfg.get("max_tools", 3),
            policy.get("weights"),
            user_message=user_message,
        )

    # 2. Cell lookup.
    key = _cell_key(turn_k, step_n)
    cell = policy.get("cells", {}).get(key, {"mode": "silent", "max_tools": 0})
    mode = cell.get("mode", "silent")
    max_tools = cell.get("max_tools", 0)

    # 2b. Signal-conditioned override — first matching condition wins, else base.
    cmode, ctools = _resolve_conditions(cell, signal, error_streak)
    if cmode is not None:
        mode, max_tools = cmode, ctools

    # 3. Dispatch.
    if mode == "silent":
        return None

    if mode == "snapshot":
        return _render_anchor(curr_snapshot, functions, recent_tools, max_tools,
                              policy.get("weights"), last_result, user_message=user_message)

    if mode == "delta":
        return _render_delta_injection(prev_snapshot, curr_snapshot, functions,
                                       recent_tools, max_tools, policy.get("weights"),
                                       last_result, require_fail=False,
                                       user_message=user_message)

    if mode == "delta_if_fail":
        if not _is_error(last_result):
            return None
        return _render_delta_injection(prev_snapshot, curr_snapshot, functions,
                                       recent_tools, max_tools, policy.get("weights"),
                                       last_result, require_fail=True,
                                       user_message=user_message)

    if mode == "minimal":
        return _render_minimal(curr_snapshot)

    if mode == "tool_list":
        return _render_tool_list(functions, recent_tools, max_tools,
                                 policy.get("weights"), last_result, curr_snapshot,
                                 user_message=user_message)

    if mode == "plan":
        return _render_plan(curr_snapshot, functions, recent_tools, max_tools,
                            policy.get("weights"), last_result, user_message,
                            with_check=False)

    if mode == "plan_with_check":
        return _render_plan(curr_snapshot, functions, recent_tools, max_tools,
                            policy.get("weights"), last_result, user_message,
                            with_check=True)

    if mode == "loop_break":
        return _render_loop_break(curr_snapshot, last_result)

    if mode == "turn_boundary":
        return _render_turn_boundary(curr_snapshot, functions, recent_tools, max_tools,
                                     policy.get("weights"), last_result, user_message)

    if mode == "gather":
        return _render_gather(curr_snapshot, functions, recent_tools, max_tools,
                              policy.get("weights"), last_result, user_message)

    return None


# ── Condition resolution ────────────────────────────────────────────────────────

def _resolve_conditions(cfg: dict, signal: str | None, error_streak: int):
    """Return (mode, max_tools) from the first matching condition, or (None, None).

    Conditions are an ordered list on a cell or the rescue config. First match wins.
    A condition matches when its `signal` equals the classified signal AND, if it
    carries a `count`, the consecutive-error streak meets that count. Editing is
    restricted to branch params (mode/max_tools/count) — order is fixed.
    """
    conditions = cfg.get("conditions")
    if not conditions:
        return None, None
    for cond in conditions:
        if cond.get("signal") != signal:
            continue
        need = cond.get("count")
        if need is not None and error_streak < need:
            continue
        return cond.get("mode", "silent"), cond.get("max_tools", 0)
    return None, None


def _dispatch_mode(mode, max_tools, prev_snapshot, curr_snapshot, functions,
                   recent_tools, last_result, weights,
                   step_n: int = 0, user_message: str = ""):
    """Render a single mode — shared by the main path and rescue conditions."""
    if mode == "silent":
        return None
    if mode == "snapshot":
        return _render_anchor(curr_snapshot, functions, recent_tools, max_tools,
                              weights, last_result, user_message=user_message)
    if mode == "delta":
        return _render_delta_injection(prev_snapshot, curr_snapshot, functions,
                                       recent_tools, max_tools, weights,
                                       last_result, require_fail=False,
                                       user_message=user_message)
    if mode == "delta_if_fail":
        if not _is_error(last_result):
            return None
        return _render_delta_injection(prev_snapshot, curr_snapshot, functions,
                                       recent_tools, max_tools, weights,
                                       last_result, require_fail=True,
                                       user_message=user_message)
    if mode == "minimal":
        return _render_minimal(curr_snapshot)
    if mode == "tool_list":
        return _render_tool_list(functions, recent_tools, max_tools, weights,
                                 last_result, curr_snapshot, user_message=user_message)
    if mode == "plan":
        return _render_plan(curr_snapshot, functions, recent_tools, max_tools,
                            weights, last_result, user_message, with_check=False)
    if mode == "plan_with_check":
        return _render_plan(curr_snapshot, functions, recent_tools, max_tools,
                            weights, last_result, user_message, with_check=True)
    if mode == "loop_break":
        return _render_loop_break(curr_snapshot, last_result)
    if mode == "turn_boundary":
        return _render_turn_boundary(curr_snapshot, functions, recent_tools, max_tools,
                                     weights, last_result, user_message)
    return None


# ── Cell key ──────────────────────────────────────────────────────────────────

def _cell_key(turn_k: int, step_n: int) -> str:
    if turn_k == 0:
        t = "0"
    elif turn_k == 1:
        t = "1"
    elif turn_k == 2:
        t = "2"
    elif turn_k == 3:
        t = "3"
    else:
        t = "4+"
    s = "0" if step_n == 0 else ("1" if step_n == 1 else "2+")
    return f"{t}_{s}"


# ── Render: snapshot / anchor ─────────────────────────────────────────────────

def _render_anchor(curr_snapshot, functions, recent_tools, max_tools, weights, last_result,
                   user_message: str = ""):
    state_lines = _render_state(curr_snapshot, user_message=user_message)
    if not state_lines:
        return None

    parts = ["[Context]", *state_lines]
    if max_tools > 0:
        ctx = _ranking_context(curr_snapshot, last_result)
        ranked = rank_tools(functions, ctx, last_result=last_result,
                            recent_tools=recent_tools, k=max_tools, weights=weights,
                            user_message=user_message)
        tool_line = _render_tools(ranked)
        if tool_line:
            parts.append(tool_line)

    return "\n".join(parts)


# ── Render: delta ─────────────────────────────────────────────────────────────

def _render_delta_injection(prev_snapshot, curr_snapshot, functions, recent_tools,
                             max_tools, weights, last_result, require_fail,
                             user_message: str = ""):
    # No previous state → fall back to snapshot.
    if prev_snapshot is None:
        return _render_anchor(curr_snapshot, functions, recent_tools, max_tools,
                               weights, last_result, user_message=user_message)

    d = compute_delta(prev_snapshot, curr_snapshot)
    if not d:
        return None

    change_lines = _render_delta(d)
    if not change_lines:
        return None

    parts = ["[Context]", *change_lines]
    if max_tools > 0:
        ranked = rank_tools(functions, d, last_result=last_result,
                            recent_tools=recent_tools, k=max_tools, weights=weights,
                            user_message=user_message)
        tool_line = _render_tools(ranked)
        if tool_line:
            parts.append(tool_line)

    return "\n".join(parts)


# ── Render: minimal ───────────────────────────────────────────────────────────

def _render_minimal(curr_snapshot) -> str | None:
    # Prefer GorillaFileSystem cwd (single most useful line for FS-heavy tasks).
    if "GorillaFileSystem" in curr_snapshot:
        fs = curr_snapshot["GorillaFileSystem"]
        cwd = fs.get("cwd", "")
        if cwd:
            return f"[Context]\ncwd: {cwd}"

    # Fallback: first scalar from any class.
    for cls, state in curr_snapshot.items():
        if not isinstance(state, dict):
            continue
        for k, v in state.items():
            if isinstance(v, (str, int, float, bool)):
                return f"[Context]\n{cls}.{k}: {v}"

    return None


# ── Render: tool_list ─────────────────────────────────────────────────────────

def _render_tool_list(functions, recent_tools, max_tools, weights, last_result,
                      curr_snapshot, user_message: str = "") -> str | None:
    """Exact available function names + one-line descriptions. No state block.

    This is the hallucination-recovery mode: when the model invents a function
    name, the missing information is the set of names that actually exist — not
    state. Showing state here is what regressed miss_func under snapshot mode, so
    this renderer deliberately omits it.
    """
    if not functions:
        return None
    k = max_tools if max_tools and max_tools > 0 else len(functions)
    ctx = _ranking_context(curr_snapshot, last_result)
    ranked = rank_tools(functions, ctx, last_result=last_result,
                        recent_tools=recent_tools, k=k, weights=weights,
                        user_message=user_message)
    if not ranked:
        return None

    # rank_tools returns {name, signature, score} — pull descriptions by name.
    desc_by_name = {f.get("name", ""): f.get("description", "") for f in functions}

    lines = ["[Available tools — call one of these exact names]"]
    for r in ranked:
        sig = r.get("signature", "")
        raw = desc_by_name.get(r.get("name", ""), "") or ""
        # BFCL descriptions are prefixed with "This tool belongs to..."; keep the
        # part after the "Tool description:" marker if present.
        marker = "Tool description:"
        if marker in raw:
            raw = raw[raw.index(marker) + len(marker):]
        desc = raw.strip().split("\n", 1)[0][:80]
        lines.append(f"- {sig}" + (f": {desc}" if desc else ""))
    return "\n".join(lines)


# ── Render: rescue ────────────────────────────────────────────────────────────

def _render_rescue(curr_snapshot, last_result, functions, recent_tools, max_tools,
                   weights, user_message: str = ""):
    parts = ["[Context — repeated failures, re-orient]"]

    # Show last error (truncated).
    if last_result:
        truncated = last_result[:200] + ("…" if len(last_result) > 200 else "")
        parts.append(f"Last result: {truncated}")

    # Compact state.
    state_lines = _render_state(curr_snapshot)
    if state_lines:
        parts.append("State now:")
        parts.extend(f"  {l}" for l in state_lines)

    # Recovery tools — re-ranked with error signal and user request.
    if max_tools > 0:
        ctx = _ranking_context(curr_snapshot, last_result)
        ranked = rank_tools(functions, ctx, last_result=last_result,
                            recent_tools=recent_tools, k=max_tools, weights=weights,
                            user_message=user_message)
        tool_line = _render_tools(ranked)
        if tool_line:
            parts.append(tool_line)

    return "\n".join(parts)


# ── State renderer ────────────────────────────────────────────────────────────

def _render_state(snapshot: dict, user_message: str = "") -> list:
    """Compact per-class summary. Returns list of lines.

    When user_message is provided and multiple API classes are present, the most
    relevant class (by token overlap with the user message) is rendered first.
    This prevents multi-API misfocus where e.g. TwitterAPI state is shown first
    when the turn is about VehicleControlAPI.
    """
    if not snapshot:
        return []

    classes = [(cls, state) for cls, state in snapshot.items()
               if isinstance(state, dict)]

    if len(classes) > 1 and user_message:
        msg_tokens = _tokenize_text(user_message)
        def _relevance(cls_state):
            cls, state = cls_state
            cls_tokens = _tokenize_text(cls)
            key_tokens = _tokenize_text(" ".join(str(k) for k in state.keys()))
            all_tokens = cls_tokens | key_tokens
            # Exact overlap + prefix overlap (handles "file" matching "files")
            exact = len(all_tokens & msg_tokens)
            prefix = sum(1 for ct in all_tokens
                         for mt in msg_tokens
                         if mt.startswith(ct) or ct.startswith(mt))
            return exact + 0.5 * prefix
        classes = sorted(classes, key=_relevance, reverse=True)

    lines = []
    for cls, state in classes:
        if cls == "GorillaFileSystem":
            lines.extend(_render_fs(state))
        else:
            lines.extend(_render_flat_class(cls, state))

    return lines


def _render_fs(fs_state: dict) -> list:
    cwd = fs_state.get("cwd", "")
    tree = fs_state.get("tree", {})

    # Navigate tree to cwd to list immediate contents.
    contents = _cwd_contents(cwd, tree)

    lines = [f"cwd: {cwd}"]
    if contents is not None:
        files = [k for k, v in contents.items() if not isinstance(v, dict)]
        dirs  = [k for k, v in contents.items() if isinstance(v, dict)]
        entries = files + [d + "/" for d in dirs]
        if entries:
            lines.append(f"  contents: {', '.join(entries)}")
        else:
            lines.append("  contents: (empty)")
    return lines


def _cwd_contents(cwd: str, tree: dict):
    """Navigate the serialized Directory tree to the cwd and return its contents dict."""
    if not tree:
        return None
    # tree = {"name": "...", "contents": {...}} — recursive Directory serialization.
    parts = [p for p in cwd.strip("/").split("/") if p]
    node = tree
    for part in parts:
        contents = node.get("contents", {})
        if part not in contents:
            return None
        child = contents[part]
        if not isinstance(child, dict):
            return None
        node = child
    return node.get("contents", {})


def _render_flat_class(cls: str, state: dict) -> list:
    """One compact line per class: scalars inline, collections summarized."""
    parts = []
    for k, v in state.items():
        if isinstance(v, (str, int, float, bool)):
            parts.append(f"{k}={v}")
        elif isinstance(v, list):
            if len(v) <= 5:
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={len(v)} entries")
        elif isinstance(v, dict):
            parts.append(f"{k}={len(v)} entries")
        elif isinstance(v, set):
            parts.append(f"{k}={sorted(v)}")
    if parts:
        return [f"{cls}: {', '.join(parts)}"]
    return []


# ── Delta renderer ────────────────────────────────────────────────────────────

_FS_PATH_RE = re.compile(
    r"^GorillaFileSystem\.tree(?:\.contents\.[^.]+)*\.contents\.([^.]+)$"
)


def _render_delta(d: dict) -> list:
    """One readable line per change in a delta dict."""
    lines = []
    for path, change in d.items():
        # Special-case: GorillaFileSystem cwd.
        if path == "GorillaFileSystem.cwd":
            old = change.get("old", "?")
            new = change.get("new", "?")
            lines.append(f"cwd: {old} → {new}")
            continue

        # Simplify filesystem tree paths to human-readable file paths.
        m = _FS_PATH_RE.match(path)
        if m:
            name = m.group(1)
            if "added" in change:
                lines.append(f"+ file: {name}")
            elif "removed" in change:
                lines.append(f"- file: {name}")
            continue

        # Generic leaf change.
        cls_field = path.split(".", 1)[1] if "." in path else path
        if "added" in change:
            lines.append(f"+ {cls_field}: {_compact(change['added'])}")
        elif "removed" in change:
            lines.append(f"- {cls_field}: {_compact(change['removed'])}")
        elif "old" in change and "new" in change:
            lines.append(f"{cls_field}: {_compact(change['old'])} → {_compact(change['new'])}")

    return lines


def _compact(val) -> str:
    """Short string representation of a value."""
    if isinstance(val, str):
        return val[:80] + ("…" if len(val) > 80 else "")
    if isinstance(val, (list, dict)):
        s = str(val)
        return s[:80] + ("…" if len(s) > 80 else "")
    return str(val)


# ── Text tokenizer (shared by relevance ranking + plan mode) ─────────────────

_RELEVANCE_STOPWORDS = {
    "the", "in", "is", "of", "to", "and", "or", "for", "with", "this", "that",
    "it", "its", "be", "are", "was", "were", "by", "at", "on", "from", "as",
    "into", "not", "no", "me", "my", "an", "a", "do", "please", "show", "list",
    "get", "set", "use", "make", "go", "let", "can", "would", "will", "should",
    "current", "currently", "status", "info", "information", "number", "now",
    "new", "all", "any", "some", "each", "first", "last", "next",
}


def _tokenize_text(text: str) -> set:
    """Lowercase split on non-alphanumeric; split camelCase; drop stopwords and short tokens."""
    if not text:
        return set()
    import re as _re
    text = _re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    tokens = _re.split(r"[^a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if t and len(t) > 2 and t not in _RELEVANCE_STOPWORDS}


# ── Render: plan / plan_with_check ────────────────────────────────────────────

def _render_plan(curr_snapshot, functions, recent_tools, max_tools, weights,
                 last_result, user_message: str, with_check: bool = False) -> str | None:
    """Pre-query anchor that includes a turn-goal restatement.

    Fires at step_n==0 only (enforced by the policy cell — the renderer itself
    does not check step_n). Injects state + a "Turn goal" line that re-anchors
    the model to THIS turn's specific task before it chooses a tool. Prevents:
      - Over-execution carry-over (model doing T1 work in T0)
      - Empty turns from turn-boundary confusion
      - Wrong-tool selection when context from prior turns dominates

    with_check=True appends an arithmetic verification reminder, targeting
    episodes where the model uses a wrong computed value (e.g. converting 10000
    RMB to USD but accidentally passing 10000 directly).
    """
    state_lines = _render_state(curr_snapshot, user_message=user_message)

    parts = ["[Context]"]
    if state_lines:
        parts.extend(state_lines)

    if max_tools > 0:
        ctx = _ranking_context(curr_snapshot, last_result)
        ranked = rank_tools(functions, ctx, last_result=last_result,
                            recent_tools=recent_tools, k=max_tools, weights=weights,
                            user_message=user_message)
        tool_line = _render_tools(ranked)
        if tool_line:
            parts.append(tool_line)

    if user_message:
        goal = user_message.strip().replace("\n", " ")[:150]
        parts.append(f"Turn goal: {goal}")
        parts.append(
            "Expected next action: call the tool that directly advances this goal"
            " — do NOT repeat work already done in prior turns."
        )

    if with_check:
        parts.append(
            "Note: if this turn requires a computed value (amount, distance, count),"
            " verify it matches the user's stated number — do not approximate or reuse a prior result."
        )

    return "\n".join(parts) if len(parts) > 1 else None


# ── Render: loop_break ────────────────────────────────────────────────────────

def _render_loop_break(curr_snapshot, last_result: str) -> str | None:
    """Inject when the model is repeating an identical successful call.

    The previous call returned a non-error result but the model called the same
    function with the same arguments again. This injection tells it to stop
    and advance to the next task step, without showing additional state that
    might confuse it.
    """
    parts = ["[Context — you already got this result]"]

    if last_result:
        truncated = last_result[:200] + ("…" if len(last_result) > 200 else "")
        parts.append(f"Previous call returned: {truncated}")

    parts.append(
        "This call already completed successfully. Do NOT call it again."
        " Move to the next step required by the task."
    )

    return "\n".join(parts)


# ── Render: turn_boundary ────────────────────────────────────────────────────

def _render_turn_boundary(curr_snapshot, functions, recent_tools, max_tools, weights,
                           last_result, user_message: str) -> str | None:
    """Inject when the prior turn likely over-executed (made more calls than needed).

    Fires via the `last_turn_over_executed` signal at step_n==0 only. Stronger
    boundary language than `plan` without the "Expected next action" phrase that
    regressed miss_param by making the model skip legitimate multi-step sequences.
    """
    state_lines = _render_state(curr_snapshot, user_message=user_message)
    parts = ["[Context — new turn]"]
    if state_lines:
        parts.extend(state_lines)

    if max_tools > 0:
        ctx = _ranking_context(curr_snapshot, last_result)
        ranked = rank_tools(functions, ctx, last_result=last_result,
                            recent_tools=recent_tools, k=max_tools, weights=weights,
                            user_message=user_message)
        tool_line = _render_tools(ranked)
        if tool_line:
            parts.append(tool_line)

    if user_message:
        goal = user_message.strip().replace("\n", " ")[:150]
        parts.append(f"Turn goal: {goal}")

    parts.append(
        "IMPORTANT: This is a NEW turn. The previous turn is complete. "
        "Only perform the action stated in this turn's message."
    )
    return "\n".join(parts)


# ── Render: gather ──────────────────────────────────────────────────────────────

def _render_gather(curr_snapshot, functions, recent_tools, max_tools, weights,
                   last_result, user_message: str) -> str | None:
    """Inject when no top-K tool has all required params grounded in context.

    Fires via the `low_groundedness` signal at step_n==0 only. Biases the
    model toward waiting/asking rather than guessing missing arguments.
    """
    state_lines = _render_state(curr_snapshot, user_message=user_message)
    parts = ["[Context]"]
    if state_lines:
        parts.extend(state_lines)

    if max_tools > 0:
        ctx = _ranking_context(curr_snapshot, last_result)
        ranked = rank_tools(functions, ctx, last_result=last_result,
                            recent_tools=recent_tools, k=max_tools, weights=weights,
                            user_message=user_message)
        tool_line = _render_tools(ranked)
        if tool_line:
            parts.append(tool_line)

    parts.append(
        "IMPORTANT: Only call a tool if all required arguments are present in the "
        "conversation or state above. If required information is missing, do NOT guess "
        "— either ask the user for it or wait for it to be provided."
    )
    return "\n".join(parts)


# ── Tool renderer ─────────────────────────────────────────────────────────────

def _render_tools(ranked: list) -> str:
    if not ranked:
        return ""
    sigs = [r["signature"] for r in ranked]
    return "Relevant tools: " + ", ".join(sigs)


# ── Ranking context synthesis ─────────────────────────────────────────────────

def _ranking_context(curr_snapshot: dict, last_result: str) -> dict:
    """
    Synthesize a delta-shaped context dict from the snapshot so that tool_ranker
    has vocabulary to overlap with even when there's no real delta.
    """
    ctx = {}
    for cls, state in curr_snapshot.items():
        if not isinstance(state, dict):
            continue
        for k, v in state.items():
            if isinstance(v, (str, int, float, bool)):
                ctx[f"{cls}.{k}"] = {"new": v}
    return ctx
