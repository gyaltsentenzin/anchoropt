"""
Tool ranker for AnchorOpt.

Scores available BFCL tool definitions by relevance to the current injection
context (state delta + last tool result + recent call history) and returns the
top-K most relevant tools.

No BFCL imports — operates purely on passed-in dicts/strings so it is
unit-testable in isolation and has zero inference overhead.
"""

import json
import re

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_K = 3

DEFAULT_WEIGHTS = {
    "state": 1.0,    # overlap with state delta
    "error": 1.5,    # overlap with error message (weighted higher — recovery context)
    "recency": 0.5,  # penalty for recently-used tools (targets blind repetition)
    "message": 2.0,  # overlap with user's current request (dominant — identifies requested tool)
}

# Tokens that appear in almost every tool schema but carry no discriminating
# signal — stripped before overlap scoring.
_STOPWORDS = {
    "a", "an", "the", "is", "in", "of", "to", "and", "or", "for", "with",
    "this", "that", "it", "its", "be", "are", "was", "were", "by", "at",
    "on", "from", "as", "into", "not", "no",
    # Schema boilerplate
    "tool", "belongs", "system", "api", "string", "integer", "boolean",
    "dict", "list", "type", "name", "description", "return", "returns",
    "current", "value", "default", "optional", "required", "object",
    "none", "true", "false",
}

# W3: structural error detection. BFCL memory backends return errors as a JSON
# object carrying an "error" key ({"error": "..."}) and the executor wraps raised
# exceptions as "Error during execution: ...". Successful results use
# status/value/keys/ranked_results and never carry a top-level error key. We match
# on that STRUCTURE, not on free words. The previous free-word regex
# (missing|cannot|not found|wrong|...) fired on legitimate stored content — e.g. a
# successful retrieval of "the invoice is missing" was misclassified as a failed
# search, inflating failed_search_streak and corrupting loop detection.
# Tight fallback for near-JSON strings that json.loads can't parse:
_ERROR_JSON_PREFIX = re.compile(r'^\{\s*[\'"]error[\'"]\s*:', re.IGNORECASE)


# ── Public API ─────────────────────────────────────────────────────────────────

def rank_tools(
    functions: list,
    state_delta: dict,
    last_result: str = "",
    recent_tools: list = None,
    k: int = DEFAULT_K,
    weights: dict = None,
    user_message: str = "",
) -> list:
    """
    Score and rank tool definitions by relevance to the current context.

    Args:
        functions:    List of BFCL function dicts {name, description, parameters}.
        state_delta:  Output of state_extractor.delta() — {path: {old, new, ...}}.
        last_result:  Raw string returned by the last tool execution (may be error).
        recent_tools: List of tool names called in recent steps (most recent last).
        k:            Number of top tools to return.
        weights:      Override default scoring weights.
        user_message: Current turn's user request — ranked separately at message weight
                      so the requested tool rises above tools only relevant to state.

    Returns:
        List of up to k dicts: [{"name": str, "signature": str, "score": float}, ...]
        Sorted descending by score.
    """
    if not functions:
        return []

    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    recent = set(recent_tools or [])
    ctx_tokens = _context_tokens(state_delta, last_result)
    msg_tokens = _tokenize(user_message) if user_message else set()
    is_err = _is_error(last_result)
    err_tokens = _tokenize(last_result) if is_err else set()

    scored = []
    for func in functions:
        tool_toks = _tool_tokens(func)

        state_score = _overlap(tool_toks, ctx_tokens) if ctx_tokens else 0.0
        msg_score   = _overlap(tool_toks, msg_tokens) if msg_tokens else 0.0
        error_score = _overlap(tool_toks, err_tokens) if err_tokens else 0.0
        recency_pen = 1.0 if func.get("name") in recent else 0.0

        score = (
            w["state"]   * state_score
            + w["message"] * msg_score
            + w["error"]   * error_score
            - w["recency"] * recency_pen
        )

        scored.append({
            "name": func.get("name", ""),
            "signature": _signature(func),
            "score": score,
        })

    # Sort descending by score; use name as tiebreaker for determinism.
    scored.sort(key=lambda x: (-x["score"], x["name"]))
    return scored[:k]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    """
    Lowercase, split camelCase/snake_case, remove stopwords.
    GorillaFileSystem → {gorilla, file, system}
    file_name         → {file, name}  (then 'name' is a stopword → {file})
    """
    if not text:
        return set()
    # Split camelCase: insert space before uppercase letters following lowercase.
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Split on anything non-alphanumeric.
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if t and t not in _STOPWORDS and len(t) > 1}


def _tool_tokens(func: dict) -> set:
    """
    Extract discriminating tokens from a function schema.
    Includes: tool name, parameter names, parameter descriptions.
    Skips the boilerplate "This tool belongs to..." prefix in the top description.
    """
    parts = []
    parts.append(func.get("name", ""))

    params = func.get("parameters", {}).get("properties", {})
    for param_name, param_info in params.items():
        parts.append(param_name)
        parts.append(param_info.get("description", ""))

    # Include the tail of the tool description (after "Tool description:") if present.
    desc = func.get("description", "")
    marker = "Tool description:"
    if marker in desc:
        parts.append(desc[desc.index(marker) + len(marker):])
    # Fallback: include whole description if no marker.
    elif desc:
        parts.append(desc)

    return _tokenize(" ".join(parts))


def _context_tokens(state_delta: dict, last_result: str) -> set:
    """
    Tokens from the state delta (keys + scalar values) + last result text.
    Delta keys look like "GorillaFileSystem.cwd" — split by dots too.
    """
    parts = []
    for path, change in state_delta.items():
        parts.append(path.replace(".", " "))
        if isinstance(change, dict):
            for val in change.values():
                if isinstance(val, (str, int, float)):
                    parts.append(str(val))
    parts.append(last_result or "")
    return _tokenize(" ".join(parts))


def _is_error(result) -> bool:
    """Structural check: is this tool result an error payload (not free-text content)?

    True iff the result is a JSON object with a top-level "error" key, or the
    executor's "Error during execution:" exception wrapper. Content that merely
    contains words like "missing" / "cannot" / "not found" is NOT an error.
    """
    if not result:
        return False
    s = result.strip() if isinstance(result, str) else str(result).strip()
    if not s:
        return False
    # Leading "error" marker: "Error:", "Error during execution:", "error occurred"…
    # A result that BEGINS with error is an error; free words like "missing"/"cannot"
    # mid-content are not. Success payloads start with "{" or plain text, never "error".
    if s.lower().startswith("error"):
        return True
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            return isinstance(obj, dict) and "error" in obj
        except (ValueError, TypeError):
            return bool(_ERROR_JSON_PREFIX.match(s))
    return False


def _overlap(tool_tokens: set, context_tokens: set) -> float:
    """Jaccard-style: intersection normalized by context size."""
    if not context_tokens:
        return 0.0
    return len(tool_tokens & context_tokens) / len(context_tokens)


def _signature(func: dict) -> str:
    """Compact one-line tool signature: name(required_param1, required_param2)."""
    name = func.get("name", "")
    required = func.get("parameters", {}).get("required", [])
    return f"{name}({', '.join(required)})"


# ── Proactive signals ─────────────────────────────────────────────────────────

def needs_action(ranked_tools: list) -> bool:
    """True when this turn plausibly requires a tool call.

    Uses a relative criterion: top-1 score must be meaningfully above the mean
    of all ranked tools. This is portable across models and task types regardless
    of schema verbosity — it does not depend on a fixed absolute threshold.

    Returns False when ranked_tools is empty, all scores are zero/negative, or
    the top score is not clearly dominant (flat distribution = ambiguous, not
    clearly action-requiring).
    """
    if not ranked_tools:
        return False
    top_score = ranked_tools[0].get("score", 0.0)
    if top_score <= 0:
        return False
    all_scores = [t.get("score", 0.0) for t in ranked_tools]
    mean_score = sum(all_scores) / len(all_scores)
    # Top tool must score at least 1.5× the mean — a clearly dominant candidate.
    return top_score >= 1.5 * mean_score if mean_score > 0 else True


def is_ambiguous_target(ranked_tools: list, margin: float = 0.15) -> bool:
    """True when the top-2 ranked tools score within `margin` of each other.

    Both tools must have positive scores (absolute-floor guard): two near-zero
    scores trivially satisfy the ratio and would cause a false-positive storm on
    low-signal turns where all tools are irrelevant.
    """
    if len(ranked_tools) < 2:
        return False
    s0 = ranked_tools[0].get("score", 0.0)
    s1 = ranked_tools[1].get("score", 0.0)
    if s0 <= 0 or s1 <= 0:
        return False
    return s1 >= (1.0 - margin) * s0


# ── Groundedness signal ───────────────────────────────────────────────────────

def _snapshot_tokens(snapshot: dict) -> set:
    """Tokens from every key and scalar value in the state snapshot."""
    parts = []
    for cls, state in snapshot.items():
        parts.append(cls)
        if isinstance(state, dict):
            for k, v in state.items():
                parts.append(str(k))
                if isinstance(v, (str, int, float, bool)):
                    parts.append(str(v))
    return _tokenize(" ".join(parts))


_VALUE_PATTERN = re.compile(
    r"\b([A-Z]{2,6}|\d{4}-\d{2}-\d{2}|\d+\.\d+|\w+_\w+|"
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+)\b"
)


def _has_value_tokens(text: str) -> bool:
    """Return True if the text contains concrete value tokens (codes, dates, IDs, emails).

    When a user message contains specific values like airport codes, dates, or IDs
    those values ground any string parameter even if they don't match description text.
    """
    return bool(_VALUE_PATTERN.search(text))


def is_low_groundedness(
    functions: list,
    snapshot: dict,
    user_message: str,
    recent_tools: list = None,
    k: int = 3,
    weights: dict = None,
) -> bool:
    """Return True if no top-K candidate tool has all required params grounded.

    A required param is 'grounded' if:
    - Its name/description tokens overlap with user message or state tokens, OR
    - The user message contains specific value tokens (dates, codes, IDs) — those
      broadly ground string params since the model can use them as arg values.
    Small-domain params (enum or bool) are always considered grounded.

    Scoped to top-K ranked tools — over all tools some tool is trivially fillable.
    """
    if not functions:
        return False

    from .injection_engine import _ranking_context
    ctx = _ranking_context(snapshot, "")
    ranked = rank_tools(functions, ctx, last_result="",
                        recent_tools=recent_tools or [], k=k, weights=weights,
                        user_message=user_message)
    if not ranked:
        return False

    avail_tokens = _tokenize(user_message) | _snapshot_tokens(snapshot)
    # If the message has concrete values (dates, codes, IDs), all string params
    # are considered potentially grounded — the user gave specific args.
    message_has_values = _has_value_tokens(user_message)

    func_by_name = {f.get("name", ""): f for f in functions}

    for r in ranked:
        func = func_by_name.get(r["name"])
        if not func:
            continue
        params = func.get("parameters", {})
        required = params.get("required", [])
        properties = params.get("properties", {})

        all_grounded = True
        for param_name in required:
            prop = properties.get(param_name, {})

            # Small-domain: enum or boolean — model can pick without external info.
            if prop.get("type") == "boolean" or prop.get("enum"):
                continue

            # String params: grounded if message has specific values OR token overlap.
            if prop.get("type") == "string" and message_has_values:
                continue

            param_tokens = _tokenize(param_name) | _tokenize(prop.get("description", ""))
            if not (param_tokens & avail_tokens):
                all_grounded = False
                break

        if all_grounded:
            return False  # At least one top-K tool is fully grounded → not low

    return True  # Every top-K tool has at least one ungrounded required param
