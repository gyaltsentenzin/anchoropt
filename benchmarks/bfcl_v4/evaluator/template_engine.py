"""
AnchorOpt template engine — runtime-grounded dynamic injection.

Replaces the mode-dispatch renderers in injection_engine.py with text templates
that contain {slot} placeholders filled from live episode state at inference time.

The optimizer edits only the static text around {slots}; the grounding content
(actual callable names, real state delta, this turn's goal) is always runtime-filled,
so templates trained on BFCL cases generalise to any episode.

Signal priority (identical to injection_engine.get_injection):
  on_loop > on_hallucinate > on_wrong_args > on_domain_error > turn_start (step_n==0)
  No signal + step_n > 0 → None (silent, same as NULL_POLICY floor on clean steps).

No BFCL imports — pure Python.
"""

import re
from typing import Optional

from .injection_engine import (
    SIG_HALLUCINATED,
    SIG_WRONG_ARGS,
    SIG_DOMAIN_ERROR,
    SIG_DOMAIN_ERROR_CORE_FULL,
    SIG_DOMAIN_ERROR_KEY_NOT_FOUND,
    SIG_DOMAIN_ERROR_TOO_LONG,
    classify_last_result,
    is_retrieval_success,
    _render_state,
    _render_delta,
    _ranking_context,
    _tokenize_text,
)
from .state_extractor import delta as compute_delta
from .tool_ranker import rank_tools

# ── Slot registry ──────────────────────────────────────────────────────────────
# Canonical set of placeholder names the template text may reference.
# The optimizer validator rejects any proposed template containing a name not in
# this set (unknown slots would silently produce literal "{unknown}" in output).

SLOT_NAMES: frozenset = frozenset({
    "turn_goal",           # current turn's user message (stripped to 150 chars)
    "pending_goal",        # last real user message before a holdout stub turn
    "state",               # compact multi-line state snapshot
    "delta",               # compact multi-line state delta since last step
    "tools",               # ranked callable tool signatures, one per line with "- " prefix
    "new_tools",           # signatures of tools newly introduced this turn (holdout function)
    "bad_name",            # invented function name extracted from NameError text
    "tool_sig",            # signature of the tool most recently called (from recent_tools)
    "last_result",         # raw last execution result, truncated to 200 chars
    "error",               # same as last_result (alias for on_domain_error readability)
    # Memory-specific slots (v4)
    "retrieved",           # pretty-printed payload of the last successful retrieval, truncated ~400 chars
    "failed_search_streak", # count of consecutive failed/empty retrieval calls this turn
})

# Template keys — must match the keys in templates_initial.json / templates_null.json.
# Adding a new key requires 3 coordinated changes: here, the routing ladder below,
# and templates_initial.json.
TEMPLATE_KEYS: frozenset = frozenset({
    # General signals (v3-compatible)
    "on_hallucinate",
    "on_loop",
    "on_wrong_args",
    "on_domain_error",
    "on_new_function_available",
    "on_turn_start_action",
    "on_ambiguous_target",
    "on_turn_boundary_orient",
    # Memory-specific signals (v4)
    "on_domain_error_core_full",       # core/archival full → archival_add
    "on_domain_error_key_not_found",   # key miss → key_search / list_keys
    "on_domain_error_too_long",        # blob overflow → compact + re-append (reactive only)
    "on_retrieval_success_pre_answer", # retrieval succeeded → verify answer before responding
    "on_idk_fallback",                 # N failed searches → try list_keys / retrieve_all
    # Gate-replacement keys (v4 run11): externalized G1/D3 text, AnchorOpt-tunable
    "on_core_clear_blocked",           # G1: core_memory_clear suppressed → use archival_add
    "on_premature_idk",                # D3: IDK before archival search → force search
    "on_memory_preamble",              # system-level proactive memory reminder (empty = no-op)
})

# Injection level per template key.
# system       — prepend to system message (proactive, per-episode)
# trailing_user — add as new user message AFTER tool results (step_n >= 1); beats recency
# turn_start_user — append to the turn-opening user message (step_n == 0, before any tools)
TEMPLATE_INJECTION_LEVELS: dict = {
    "on_memory_preamble":              "system",
    # reactive: all fire at step_n >= 1 (last_result / idk signal available)
    "on_domain_error_core_full":       "trailing_user",
    "on_domain_error_key_not_found":   "trailing_user",
    "on_domain_error_too_long":        "trailing_user",
    "on_retrieval_success_pre_answer": "trailing_user",
    "on_idk_fallback":                 "trailing_user",
    "on_hallucinate":                  "trailing_user",
    "on_loop":                         "trailing_user",
    "on_wrong_args":                   "trailing_user",
    "on_domain_error":                 "trailing_user",
    "on_core_clear_blocked":           "trailing_user",
    "on_premature_idk":                "trailing_user",
    # proactive at turn boundary (step_n == 0, no prior tool results)
    "on_turn_boundary_orient":         "turn_start_user",
    "on_new_function_available":       "turn_start_user",
    "on_turn_start_action":            "turn_start_user",
    "on_ambiguous_target":             "turn_start_user",
}


def get_injection_level(key: str) -> str:
    """Return the injection level for a template key (default: trailing_user)."""
    return TEMPLATE_INJECTION_LEVELS.get(key, "trailing_user")

# Default max tools ranked for the {tools} slot
_DEFAULT_MAX_TOOLS = 3


# ── Slot value builder ─────────────────────────────────────────────────────────

def _fmt_new_tools(funcs: list) -> str:
    """Render a list of BFCL function dicts as '- signature' lines."""
    lines = []
    for f in funcs:
        if isinstance(f, dict):
            from .tool_ranker import _signature
            lines.append(f"- {_signature(f)}")
        elif isinstance(f, str):
            lines.append(f"- {f}")
    return "\n".join(lines)


def _build_slot_values(
    needed: set,
    *,
    user_message: str,
    curr_snapshot: dict,
    prev_snapshot,
    functions: list,
    recent_tools: list,
    last_result: str,
    weights: Optional[dict],
    new_tools_list: list = None,
    pending_goal: str = "",
    **kwargs,
) -> dict:
    """Build only the slot values referenced by the chosen template.

    Lazy: only compute what the template actually uses, reusing the existing
    render helpers from injection_engine / tool_ranker.
    """
    vals = {}

    if "turn_goal" in needed:
        vals["turn_goal"] = user_message.strip().replace("\n", " ")[:150] if user_message else ""

    if "pending_goal" in needed:
        vals["pending_goal"] = (pending_goal or "").strip().replace("\n", " ")[:150]

    if "state" in needed:
        lines = _render_state(curr_snapshot, user_message=user_message)
        vals["state"] = "\n".join(lines) if lines else ""

    if "delta" in needed:
        if prev_snapshot is not None:
            d = compute_delta(prev_snapshot, curr_snapshot)
            lines = _render_delta(d) if d else []
        else:
            lines = _render_state(curr_snapshot, user_message=user_message)
        vals["delta"] = "\n".join(lines) if lines else ""

    if "tools" in needed:
        ctx = _ranking_context(curr_snapshot, last_result)
        ranked = rank_tools(
            functions, ctx,
            last_result=last_result,
            recent_tools=recent_tools,
            k=_DEFAULT_MAX_TOOLS,
            weights=weights,
            user_message=user_message,
        )
        if ranked:
            lines = [f"- {r['signature']}" for r in ranked]
            vals["tools"] = "\n".join(lines)
        else:
            vals["tools"] = ""

    if "new_tools" in needed:
        vals["new_tools"] = _fmt_new_tools(new_tools_list or [])

    if "bad_name" in needed:
        # NameError text: "name '<X>' is not defined"
        m = re.search(r"name '([^']+)' is not defined", last_result or "")
        vals["bad_name"] = m.group(1) if m else ""

    if "tool_sig" in needed:
        # Signature of the last called tool (may be empty if recent_tools is empty)
        last_tool = recent_tools[-1] if recent_tools else ""
        sig = ""
        if last_tool and functions:
            for f in functions:
                if f.get("name") == last_tool:
                    # Build a compact signature: name(param1, param2, ...)
                    params = list((f.get("parameters") or {}).get("properties", {}).keys())
                    required = (f.get("parameters") or {}).get("required", [])
                    param_strs = []
                    for p in params:
                        param_strs.append(p if p in required else f"{p}=...")
                    sig = f"{last_tool}({', '.join(param_strs)})"
                    break
        vals["tool_sig"] = sig

    if "last_result" in needed or "error" in needed:
        truncated = (last_result or "")[:200] + ("…" if len(last_result or "") > 200 else "")
        vals["last_result"] = truncated
        vals["error"] = truncated  # alias

    if "retrieved" in needed:
        # Pretty-print the retrieved memory payload for use in on_retrieval_success_pre_answer.
        # Attempt to parse as JSON and extract the most human-readable field.
        import json as _json
        raw = (last_result or "").strip()
        pretty = raw[:400] + ("…" if len(raw) > 400 else "")
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                # Prefer the most informative field in order of specificity.
                for key in ("value", "memory_content", "text", "ranked_results", "keys"):
                    if key in parsed and parsed[key] is not None:
                        field_val = parsed[key]
                        if isinstance(field_val, (list, dict)):
                            field_str = _json.dumps(field_val, ensure_ascii=False)
                        else:
                            field_str = str(field_val)
                        pretty = field_str[:400] + ("…" if len(field_str) > 400 else "")
                        break
        except Exception:
            pass
        vals["retrieved"] = pretty

    if "failed_search_streak" in needed:
        # Caller passes this via kwargs; default 0 if not provided.
        vals["failed_search_streak"] = str(kwargs.get("failed_search_streak", 0))

    return vals


# ── Template renderer ─────────────────────────────────────────────────────────

_SLOT_RE = re.compile(r"\{([a-z_]+)\}")


def _extract_slots(tmpl: str) -> set:
    """Return the set of slot names referenced in tmpl."""
    return {m.group(1) for m in _SLOT_RE.finditer(tmpl)}


def render_template(tmpl_str: str, slot_values: dict) -> Optional[str]:
    """Fill {slot} placeholders; drop lines whose referenced slot is empty.

    Rules:
    - A line is kept only if every {slot} it references has a non-empty value.
    - After filling all lines, strip surrounding whitespace.
    - Return None if the result is blank (nothing to inject).
    - Non-empty result is returned as-is (callers prepend '[Context]' if needed).
    """
    if not tmpl_str or not tmpl_str.strip():
        return None

    output_lines = []
    for line in tmpl_str.split("\n"):
        slots_in_line = {m.group(1) for m in _SLOT_RE.finditer(line)}
        # Drop the line if any referenced slot is empty
        if any(not slot_values.get(s, "") for s in slots_in_line):
            continue
        # Fill the slots that are present
        filled = line
        for s in slots_in_line:
            filled = filled.replace(f"{{{s}}}", slot_values[s])
        output_lines.append(filled)

    result = "\n".join(output_lines).strip()
    return result if result else None


def validate_template(tmpl_str: str) -> list:
    """Return a list of unknown slot names found in tmpl_str (empty = valid)."""
    return [s for s in _extract_slots(tmpl_str) if s not in SLOT_NAMES]


# ── Main injection entry point ────────────────────────────────────────────────

def get_injection_from_templates(
    turn_k: int,
    step_n: int,
    failure_streak: int,
    prev_snapshot,
    curr_snapshot: dict,
    functions: list,
    recent_tools: list,
    last_result: str,
    templates: dict,
    error_streak: int = 0,
    loop_signal: bool = False,
    user_message: str = "",
    over_execution_signal: bool = False,
    low_groundedness_signal: bool = False,
    new_functions_signal: bool = False,
    new_tools_list: list = None,
    turn_needs_action_signal: bool = False,
    ambiguous_target_signal: bool = False,
    pending_goal: str = "",
    # Memory-specific signals (v4)
    idk_fallback_signal: bool = False,   # N consecutive failed/empty searches this turn
    failed_search_streak: int = 0,       # count of consecutive failed searches (for {failed_search_streak} slot)
    return_key: bool = False,            # T1: also return which signal key fired (for telemetry/backend-scoping)
    report_trigger: bool = False,        # build-only: also expose the resolved trigger key even when text is empty
):
    """Return a grounded injection string, or None if no injection applies.

    Signal priority:
      loop > over_execution (kept for JSON-policy compat) >
      on_new_function_available (B) > on_turn_start_action (A) >
      on_ambiguous_target (C) > reactive (hallucinate/wrong_args/domain_error) >
      on_turn_boundary_orient (D, fallback at T>0 step 0)

    over_execution is retained in the argument list for backward compatibility
    with the JSON-policy path but no longer has a template key — it resolves to
    None silently. low_groundedness similarly removed.

    Proactive signals (B, A, C, D) only fire at step_n == 0.

    return_key: default False preserves the historical contract — returns
    Optional[str] (the injection, or None). When True, returns a
    (injection, key) tuple where `key` is the signal key that actually
    injected text (None when nothing was injected). The evaluator uses this
    for per-signal telemetry and backend-aware scoping; the string-only
    contract is what every existing caller/test relies on, so it stays default.
    This function remains BFCL-import-free: it only resolves and reports the
    key; any registry-backed backend scoping happens in the caller.

    report_trigger: default False. When True, returns a
    (injection, injected_key, trigger_key) 3-tuple where `trigger_key` is the
    signal key whose *condition* fired, reported regardless of whether its
    tunable text was empty (injected_key keeps the return_key semantics —
    None when nothing was injected). Used ONLY by the snapshot-store build to
    detect which storage-affecting signals fired during prereq episodes, so an
    empty→non-empty edit of an opt-in signal still invalidates the cached store.
    This path is orthogonal to the string-only / 2-tuple contracts above.
    """
    def _ret(injection, key):
        # report_trigger (build-only): expose the resolved trigger key even when
        # the template text is empty — an opt-in signal that ships empty still
        # fires its condition and could change storage behavior once filled.
        if report_trigger:
            return injection, (key if injection else None), key
        # Only report a key when text was actually injected — a resolved-but-empty
        # signal (e.g. the null arm, where every template is "") counts as silent.
        if return_key:
            return injection, (key if injection else None)
        return injection
    tmpl_dict = templates.get("templates", {})

    # ── Signal resolution ──────────────────────────────────────────────────────
    if loop_signal:
        key = "on_loop"
    elif new_functions_signal and step_n == 0:
        key = "on_new_function_available"
    elif turn_needs_action_signal and step_n == 0:
        key = "on_turn_start_action"
    elif ambiguous_target_signal and step_n == 0:
        key = "on_ambiguous_target"
    elif idk_fallback_signal and step_n >= 1:
        # N consecutive failed searches — nudge the model to try list_keys / retrieve_all.
        key = "on_idk_fallback"
    else:
        sig = classify_last_result(last_result)
        if sig == SIG_HALLUCINATED:
            key = "on_hallucinate"
        elif sig == SIG_WRONG_ARGS:
            key = "on_wrong_args"
        # W2/Phase 2 fidelity: core-full, key-not-found, and too-long are now handled
        # by the DETERMINISTIC G3/G4/G5 gates in memory_evaluator (mirroring
        # base_handler), using on_domain_error_core_full / on_domain_error_key_not_found
        # / on_domain_error_too_long as the tunable gate text. They are intentionally
        # NOT fired from the signal path here — doing so would double-inject on top of
        # the gate. (Same pattern as on_core_clear_blocked / on_premature_idk, which are
        # also gate-only keys.)
        elif sig == SIG_DOMAIN_ERROR:
            key = "on_domain_error"
        elif step_n >= 1 and is_retrieval_success(last_result):
            # A successful retrieval just completed — anchor the model before it answers.
            key = "on_retrieval_success_pre_answer"
        elif turn_k > 0 and step_n == 0:
            key = "on_turn_boundary_orient"
        else:
            return _ret(None, None)  # silent on clean mid-episode steps

    tmpl_str = tmpl_dict.get(key, "")
    if not tmpl_str or not tmpl_str.strip():
        return _ret(None, key)  # empty template == silent for this signal

    # ── Lazy slot building ─────────────────────────────────────────────────────
    needed = _extract_slots(tmpl_str)
    weights = None  # templates don't carry weight dicts; use tool_ranker defaults
    slot_values = _build_slot_values(
        needed,
        user_message=user_message,
        curr_snapshot=curr_snapshot,
        prev_snapshot=prev_snapshot,
        functions=functions,
        recent_tools=recent_tools,
        last_result=last_result,
        weights=weights,
        new_tools_list=new_tools_list or [],
        pending_goal=pending_goal,
        failed_search_streak=failed_search_streak,
    )

    return _ret(render_template(tmpl_str, slot_values), key)
