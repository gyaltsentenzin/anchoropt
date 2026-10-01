"""
AnchorOpt memory evaluator for BFCL v4 agentic memory tasks.

Runs one memory episode with AnchorOpt injection hooks and grades the final
answer using the BFCL agentic checker (deterministic word-boundary substring
match), NOT the v3 multi-turn state/response checker.

Design decisions vs the v3 AnchorOptEvaluator:
- No inheritance from GraniteMultiturnEvaluator / skillopt — that class is
  hardwired to the v3 multi_turn_checker, which mis-grades memory tasks.
- Memory tasks are "agentic" in BFCL v4: the model runs a multi-step tool loop
  within a single episode; the grade is on the final non-function-call answer.
- Snapshot continuity: memory APIs persist state on-disk across prereq entries.
  _reset_episode_state must NOT wipe the prereq snapshot folder between chained
  entries (memory_api_metaclass loads snapshots by test_id).
- The idk_fallback_signal is tracked per-episode as failed_search_streak, reset
  on any non-empty retrieval result.

Correct BFCL execution path (mirrors base_handler.py inference_multi_turn_prompting):
  1. execute_multi_turn_func_call([], ...)  — bootstrap: creates MemoryAPI instance
  2. add_memory_instruction_system_prompt(...)  — injects system prompt + core memory
  3. _pre_query_processing_prompting(test_entry)  — builds inference_data
  4. Episode loop: _add_first/next_turn_message → inject → query → decode → execute
  5. if is_memory_prereq: memory_instance._flush_memory_to_local_file()

Usage:
    evaluator = MemoryAnchorOptEvaluator(handler, vllm_url, served_model_name)
    checker_result, trajectory = evaluator.evaluate_episode(test_case, templates)
"""

import ast
import contextlib
import copy
import hashlib
import io
import os
import re
import sys
import tempfile
import threading
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Per-thread stdout capture for _quiet_snapshot_warnings.
# Replaces sys.stdout once (at import time) with a wrapper that routes writes to a
# per-thread StringIO buffer when that thread is inside _quiet_snapshot_warnings,
# and to the real stdout otherwise.  No global lock needed — each thread captures
# only its own output without affecting sibling threads.
_real_stdout = sys.stdout
_tl = threading.local()





# ── E1: archival-full duplicate eviction ─────────────────────────────────────
# Frozen in docs/N0_E1_CRITERIA_FROZEN.md BEFORE implementation. MECHANISM TEST: 15 firings on one
# chain, so it cannot pass use-case invariance. It isolates CONSTRAINED EXECUTION (the anchor performs
# a deterministic removal) from DELEGATED RECOVERY (B1's generic nudge, which cost -6.60pp because the
# model reinterpreted "do something else" as core_memory_clear).
E1_MAX_EVICTIONS = 3       # per episode. Unbounded eviction is a wholesale clear in slow motion.
_E1_SLOT_RE = re.compile(r"memory size exceeds maximum size", re.I)
_E1_ADD_RE = re.compile(r"archival_memory_add\s*\(")
_E1_VAL_RE = re.compile(r"(?:text|value|content)\s*=\s*(['\"])(.*?)\1", re.S)
_E1_ID_RE = re.compile(r'"?id"?\s*[:=]\s*(\d+)')


def _e1_norm(s) -> str:
    """Identity for duplicate detection: whitespace-normalized, lowercased. NO fuzzy comparator.

    A near-duplicate is NOT a duplicate. Fuzzy matching would break the information-preservation
    invariant, because two similar entries are not interchangeable.
    """
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _e1_pick_victim(tracked):
    """The LAST-WRITTEN copy of some value that appears >= 2 times, or None.

    Deterministic by construction. Returns (vec_id, normalized_value, total_copies).
    Preserves the EARLIEST instance, which later queries are likelier to have referenced.
    """
    counts = {}
    for _vid, val in tracked:
        counts[val] = counts.get(val, 0) + 1
    dup_vals = [v for v, n in counts.items() if n >= 2]
    if not dup_vals:
        return None
    # among duplicated values, take the one whose LAST copy was written most recently
    best = None
    for vid, val in tracked:
        if val in dup_vals:
            best = (vid, val, counts[val])
    return best



# ---------------------------------------------------------------------------------------------------
# KV KEY-PRESERVING CAPACITY MANAGEMENT -- OBSERVER ONLY, takes no action.
#
# A5 must NOT be ported to kv unchanged. Vector entries are anonymous, so evicting an exact duplicate
# is the whole policy. KV entries have an ADDRESS, and `core_memory_add` REFUSES a taken key
# ({"error": "Key name must be unique."}) rather than overwriting -- so the information the model
# wanted to store is DROPPED, not superseded. The remedy ladder must preserve addressability:
#
#   T1  same key + SAME info         -> SUPPRESS the write (store already says this)     LOSSLESS
#   T2  same key + COMPATIBLE info   -> MERGE into the existing value IF IT FITS         LOSSLESS
#   T3  same key + OVERLOADED info   -> SPLIT into finer keys                            LOSSLESS
#   T4  none of the above            -> destructive replace                              LAST RESORT
#
# Compression ranks BELOW T3 and is NOT assumed lossless -- W1 showed a fact-preserving rewrite still
# shifted retrieval.
#
# This block only CLASSIFIES, because the offline audit could resolve just 6 of 28 rejections: 22 keys
# were taken by PRE-SEEDED state that trace reconstruction cannot see. That is the same blindness that
# voided E1 v1 and invalidated the vector redundancy row, so the population is measured against the
# live store before any arm is designed.
_KVO_ADD_RE = re.compile(r"\b(core|archival)_memory_add\s*\(")
_KVO_KEY_RE = re.compile(r"key\s*=\s*(['\"])(.*?)\1", re.S)
_KVO_VAL_RE = re.compile(r"value\s*=\s*(['\"])(.*?)\1", re.S)
_KVO_UNIQ_RE = re.compile(r"key name must be unique", re.I)
_KVO_CAPS = {"core": 300, "archival": 2000}


def _kvo_norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _kvo_live_store(involved_instances, which):
    """The LIVE kv store as dict[key -> value], or {} if unavailable.

    kv holds plain dicts (`core_memory`, `archival_memory`) directly on the instance, so unlike
    vector's `archival_memory._store` no unwrapping is needed.
    """
    insts = ((involved_instances or {}).values()
             if isinstance(involved_instances, dict) else (involved_instances or []))
    for inst in insts:
        d = getattr(inst, "%s_memory" % which, None)
        if isinstance(d, dict) and d:
            return {str(k): str(v) for k, v in d.items()}
    return {}


def _kvo_units(s):
    return [x for x in re.split(r"(?<=[.;!?])\s+|\s*\|\s*|\s*;\s*", str(s or "").strip())
            if len(x.strip()) > 12]


def _kvo_classify(old, new, cap):
    """Deterministic taxonomy over (stored value, attempted value). Returns (class, reason)."""
    no, nn = _kvo_norm(old), _kvo_norm(new)
    if not nn:
        return "T1_suppress", "attempted value empty"
    if no == nn:
        return "T1_suppress", "identical to stored value"
    if nn in no:
        return "T1_suppress", "attempted value already contained in stored value"
    if no in nn:
        return "T2_merge", "stored value contained in new -> supersedes losslessly"
    merged = len(str(old)) + 2 + len(str(new))
    if merged <= cap:
        return "T2_merge", "disjoint but merged %d <= cap %d" % (merged, cap)
    if len(_kvo_units(old)) + len(_kvo_units(new)) >= 2:
        return "T3_split", "merged %d > cap %d, decomposable into finer keys" % (merged, cap)
    return "T4_destructive", "merged %d > cap %d and atomic" % (merged, cap)


def _e1_live_archival(involved_instances):
    """The LIVE archival store as [(vec_id, normalized_text)], newest last, or [] if unavailable.

    Reads `inst.archival_memory._store` -- a dict[int, str] of vec_id -> text on the vector backend --
    through the same `involved_instances` handle the constraint-state capture already uses. This is
    the state the TOOL SYSTEM itself holds, including entries written by earlier episodes on the same
    chain, which episode-local bookkeeping cannot see.

    Why this matters beyond fixing a bug: AnchorOpt claims state-aware LOCAL decision policies. A
    controller restricted to our own write log is not inspecting the state available at the decision
    point, so an arm built on it understates what a state-aware policy can do. E1 v1 was void for
    exactly this reason -- it saw a median of 4 entries where the store held ~50.

    Insertion order is preserved by dict ordering, and vec_ids are monotonically assigned by
    VectorStore.add, so "last-written" is well defined as the highest id among duplicates.
    """
    insts = ((involved_instances or {}).values()
             if isinstance(involved_instances, dict) else (involved_instances or []))
    for inst in insts:
        am = getattr(inst, "archival_memory", None)
        store = getattr(am, "_store", None) if am is not None else None
        if isinstance(store, dict) and store:
            return [(int(k), _e1_norm(v)) for k, v in sorted(store.items())]
    return []


# ── B1: futility escalation ───────────────────────────────────────────────────
# Fires when the model has demonstrably STOPPED recovering, not when an error occurs. 76% of the 948
# real error events already self-correct, so an error-triggered nudge is mostly risk -- the A4v1
# lesson. See docs/N0_B1_CRITERIA_FROZEN.md.
B1_FAIL_THRESHOLD = 3      # Nth failure of the SAME call. Measured: N=2 fires 59x (catches normal
                           # reformulation), N=3 45x, N=4 25x (waste already done).
B1_MAX_FIRINGS = 2         # per episode, shared by F1 and F2


def _b1_is_error(result) -> bool:
    """A REAL error: the payload carries a top-level "error" key.

    Substring matching on "error" is wrong and was measured wrong: it counted retrieved CONTENT that
    merely mentions the word, inflating 948 real errors to 1041 and true abandonment from 30 to 91.
    """
    s = str(result or "")
    try:
        o = _json.loads(s)
        return isinstance(o, dict) and "error" in o
    except Exception:
        return bool(re.match(r'^\s*\{\s*"error"', s))


def _b1_last_step_errored(history) -> bool:
    """Did the most recent executed step return a real error?

    Written as a function rather than inline: the inline version was a malformed comprehension
    (a trailing  inside the iterable), which is a syntax error rather than a logic bug --
    caught by the parse check before launch.
    """
    if not history:
        return False
    last = history[-1] or {}
    return any(_b1_is_error(r) for r in (last.get("tool_results") or []))


def _b1_threshold(templates) -> int:
    try:
        v = (templates or {}).get("b1_fail_threshold")
        return int(v) if v is not None else B1_FAIL_THRESHOLD
    except (TypeError, ValueError):
        return B1_FAIL_THRESHOLD


# ── A5: retrieval-similarity signal feature ──────────────────────────────────
# `s` = top similarity score returned by a read. Proposed by contrast against the A0 global-prompt
# teacher, validated on the full 303-case train set. See docs/N0_S6_SIGNAL_FEATURE.md and
# docs/N0_A5_CRITERIA_FROZEN.md.
#
# These live at module scope so the offline analysis scripts (s6_incision.py, s6_policy_grid.py,
# s6_feature_validate.py) and the live gate agree BY CONSTRUCTION on what counts as a read and what
# counts as its similarity. Divergence between an offline predicate and the live one is how the A3
# audit produced 237 phantom violations, and it is worth a shared function to avoid.

_A5_READ_VERBS = ("retrieve", "search", "get", "view", "lookup")
_A5_SIM_RE = re.compile(r'"similarity_score"\s*:\s*([0-9.eE+-]+)')
_A5_CALL_RE = re.compile(r"([a-z_][a-z_0-9]*)\s*\(")
A5_DEFAULT_THETA = 0.30          # FROZEN on train. docs/N0_A5_CRITERIA_FROZEN.md.


def _is_read_call(call_str) -> bool:
    """Does this call read from memory? Name-based, matching the offline scripts exactly."""
    m = _A5_CALL_RE.search(str(call_str or ""))
    if not m:
        return False
    name = m.group(1)
    return any(v in name for v in _A5_READ_VERBS)


def _top_similarity(result):
    """Max similarity_score in a tool result, or None if the backend reported none.

    None is NOT zero, and the distinction is load-bearing: kv and rec_sum return no similarity at
    all, so treating a missing score as 0.0 would make every kv/rec_sum read look maximally weak
    and fire the gate on backends that cannot expose this feature. The gate therefore requires a
    non-None score, and `_query_read_sims` preserves the None entries so "one read, unscored" is
    distinguishable from "one read, scored low".
    """
    vals = [float(x) for x in _A5_SIM_RE.findall(str(result or ""))]
    return max(vals) if vals else None


def _low_sim_theta(templates) -> float:
    """theta for the A5 gate. Policy may override; default is the frozen 0.30."""
    try:
        v = (templates or {}).get("low_similarity_theta")
        return float(v) if v is not None else A5_DEFAULT_THETA
    except (TypeError, ValueError):
        return A5_DEFAULT_THETA


class _PerThreadCapture:
    """sys.stdout wrapper: captures to a per-thread buffer when active, else passthrough."""

    def write(self, s):
        buf = getattr(_tl, "buf", None)
        if buf is not None:
            buf.write(s)
        else:
            _real_stdout.write(s)
        return len(s)

    def flush(self):
        _real_stdout.flush()

    def __getattr__(self, name):
        return getattr(_real_stdout, name)


sys.stdout = _PerThreadCapture()


@contextlib.contextmanager
def _quiet_snapshot_warnings():
    """Intercept memory_api_metaclass ⚠️*100 blocks and print one compact line instead.

    Thread-safe: each thread captures only its own stdout writes via _tl.buf.
    """
    _tl.buf = io.StringIO()
    try:
        yield
    finally:
        output = _tl.buf.getvalue()
        _tl.buf = None
    if not output:
        return
    # Collapse each ⚠️-block + warning text into a single prefixed line
    cleaned = re.sub(r"(⚠️)+\nWarning: ([^\n]+)\n(⚠️)+", r"⚠  \2", output)
    cleaned = re.sub(r"^(⚠️)+$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n", cleaned)
    _real_stdout.write(cleaned)
    _real_stdout.flush()

from .injection_engine import (
    _cell_key,
    classify_last_result,
    is_retrieval_success,
    _ranking_context,
)
from .template_engine import get_injection_from_templates
from .tool_ranker import _is_error

# ── BFCL agentic checker (read-only import) ───────────────────────────────────
try:
    from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker
except ImportError:
    agentic_checker = None

# ── BFCL multi-turn execution utilities ───────────────────────────────────────
# Correct import path: multi_turn_utils (not func_exec_utils which does not exist).
try:
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
        is_empty_execute_response,
    )
except ImportError:
    execute_multi_turn_func_call = None
    is_empty_execute_response = None

# ── BFCL memory system-prompt injection (read-only import) ────────────────────
try:
    from bfcl_eval.model_handler.utils import add_memory_instruction_system_prompt
except ImportError:
    add_memory_instruction_system_prompt = None

# ── Signal/gate registry (canonical data lives on the bfcl side; this package's
#    thin adapter degrades to inert stand-ins if bfcl_eval isn't importable) ──
# Canonical gate-config fingerprint (§8.3). Imported defensively: memory_gates
# degrades to inert stubs when bfcl_eval is unavailable, and a missing fingerprint
# must not break evaluation — but it MUST NOT silently return a constant either,
# or the cache-miss guarantee is void. The fallback hashes the policy's gate-facing
# keys directly so a gate change still splits the key.
try:
    from .memory_gates import gate_config_fingerprint as _gate_config_fingerprint
except ImportError:  # pragma: no cover
    def _gate_config_fingerprint(policy):
        import hashlib as _h, json as _jj
        rel = {k: v for k, v in (policy or {}).items()
               if k == "gate_enabled" or k.startswith("enable_")
               or k.endswith("_threshold") or k == "gate_default"}
        return _h.sha256(_jj.dumps(rel, sort_keys=True, default=str).encode()).hexdigest()[:16]

from .memory_gates import (
    RECENT_CALL_SIGNATURE_WINDOW,
    RETRIEVAL_TOOLS,
    IDK_FALLBACK_THRESHOLD,
    G1_USER_WANTS_CLEAR_PHRASES,
    REGISTRY_BY_KEY,
    archival_memory_searched,
    blob_pressure_threshold,
    disabled_gate_keys,
    enabled_explicit_gate_keys,
    find_failing_core_full_call,
    gate_applies,
    gate_enabled,
    gate_fallback,
    gate_match,
    gate_match_any,
    reprompt_specs,
    reroute_specs,
    find_failing_call_for_spec,
    synthesize_reroute_for_spec,
    suppress_specs,
    suppress_user_exempt,
    transform_specs,
    transform_instruction,
    transform_ok,
    is_error_result,
    is_empty_or_failed_retrieval,
    loop_repeat_detected,
    loop_repeat_threshold,
    memory_backend,
    remedy_enabled,
    seed_text,
    synthesize_core_full_reroute,
    synthesize_forced_key_search_call,
    synthesize_forced_retrieval_call,
    threshold_overrides,
)

# Single-sourced in bfcl_eval.model_handler.memory_gates (imported above as
# RETRIEVAL_TOOLS/IDK_FALLBACK_THRESHOLD) so base_handler.py's on_idk_fallback
# wiring reads the identical set/threshold. Local aliases kept so call sites
# below don't need to change.
_RETRIEVAL_TOOLS = RETRIEVAL_TOOLS
_IDK_FALLBACK_THRESHOLD = IDK_FALLBACK_THRESHOLD


def _last_non_fc_message(all_model_responses: List[List[str]], handler) -> str:
    """Return the last model response that could not be decoded as a function call.

    Uses is_empty_execute_response from multi_turn_utils (the official BFCL check)
    and passes has_tool_call_tag=False as base_handler does at line 572.
    """
    _is_empty = is_empty_execute_response if is_empty_execute_response else (lambda x: not x)
    for turn_responses in reversed(all_model_responses):
        for response in reversed(turn_responses):
            try:
                decoded = handler.decode_execute(response, has_tool_call_tag=False)
                if not _is_empty(decoded):
                    continue  # successfully decoded as tool call — skip
            except Exception:
                pass
            if response and response.strip():
                return response.strip()
    return ""


# _is_empty_or_failed_retrieval now single-sourced as is_empty_or_failed_retrieval
# in bfcl_eval.model_handler.memory_gates (imported above) — was a local def here,
# now called directly at its one call site below.


def _group_prereqs_by_chain(prereq_cases: List[Dict]) -> Dict[str, List[Dict]]:
    """Group prereq cases into (backend, scenario) chains, cases sorted by their
    trailing position number within each chain. ID format:
    memory_{backend}_prereq_{global_id}-{scenario}-{position}.

    Shared by build_snapshot_store (to resolve store_workers for the cache key)
    and _run_prereq_build (to actually run the chains), so the two never disagree
    on chain count.
    """
    def _chain_id(c):
        m = re.match(r"(memory_\w+)_prereq_\d+-(\w+)-\d+$", c["id"])
        return f"{m.group(1)}_{m.group(2)}" if m else c["id"]

    def _prereq_num(c):
        m = re.search(r"-(\d+)$", c["id"])
        return int(m.group(1)) if m else 0

    chains: Dict[str, List[Dict]] = defaultdict(list)
    for c in prereq_cases:
        chains[_chain_id(c)].append(c)
    for cases in chains.values():
        cases.sort(key=_prereq_num)
    return dict(chains)


def _resolve_store_workers(store_workers: int, n_chains: int) -> int:
    """<=0 means "no cap" (all chains in parallel) — an explicit spelling of the
    historical hardcoded min(n_chains, 15) behavior. A positive value caps the
    pool width; the new default of 1 makes prereq builds serial (deterministic)."""
    return n_chains if store_workers <= 0 else min(n_chains, store_workers)


# ── Part B: code-version fingerprint (base_key had no code-version salt; a
#    stale cross-commit store was once silently reused — see plan) ───────────

# The behavioural closure of a prereq episode, derived from the import graph
# above. One AST parse of these ~6 files per build call (cheap relative to a
# multi-minute prereq rebuild).
_STORE_CODE_MODULES = (
    "memory_evaluator.py",
    "memory_gates.py",
    "injection_engine.py",
    "template_engine.py",
    "tool_ranker.py",
    "state_extractor.py",
)

# Manual lever for an out-of-band change the AST fingerprint can't see (e.g. a
# vLLM version bump) — folded into base_key alongside the fingerprint so it
# actually invalidates stores when bumped, unlike the old anchoropt_store_schema
# field (written but read nowhere).
#
# Bumped 2 -> 3 (plan §105): the BM25 empty-corpus fix in bfcl_eval's memory_kv
# backend changed prereq behaviour, but _STORE_CODE_MODULES only covered THIS
# package, so no cached store was invalidated by it. Every snap_* dir built before
# that fix encodes a prereq world in which searching an empty memory raised
# ZeroDivisionError. The bump is belt-and-braces alongside _BACKEND_CODE_MODULES
# below, which closes the gap structurally for the next such change.
_ANCHOROPT_STORE_SCHEMA = 3

# The memory BACKEND implementations live in bfcl_eval, not in this package, so the
# closure above cannot see them -- yet a prereq episode executes them on every tool
# call, and they decide what the resulting store contains. A backend edit MUST split
# the cache key. Resolved by import location rather than a hardcoded path so a moved
# checkout cannot silently degrade to "unknown".
_BACKEND_CODE_MODULES = ("memory_api_metaclass.py", "memory_kv.py",
                         "memory_rec_sum.py", "memory_vector.py")


def _backend_pkg_dir() -> Optional[Path]:
    """Directory holding the bfcl_eval memory backend implementations, or None."""
    try:
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import (  # noqa: F401
            memory_kv as _mk,
        )
    except Exception:
        return None
    return Path(_mk.__file__).resolve().parent


def _ast_normalized_dump(source: str) -> str:
    """Parse-tree dump with docstrings stripped. Comments, blank lines, and
    formatting are already invisible to ast.parse; dropping docstrings too
    means a comment/docstring-only edit does not change the dump, while any
    real logic change (including a changed literal) does."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _store_code_fingerprint(pkg_dir: Optional[Path] = None) -> str:
    """AST-normalized digest of the behavioural module closure a prereq episode
    executes. `pkg_dir` is a test seam (defaults to this file's real package
    dir); production callers never pass it.

    ANCHOROPT_STORE_CODE_FINGERPRINT overrides the computed value (loud log) —
    the release valve for a known-irrelevant change. Falls back to "unknown"
    (with a warning) if source can't be read/parsed — never crash a 30-minute
    build over a fingerprint.
    """
    import os as _os

    override = _os.environ.get("ANCHOROPT_STORE_CODE_FINGERPRINT")
    if override:
        print(
            f"[snapshot_store] store code fingerprint OVERRIDDEN via "
            f"ANCHOROPT_STORE_CODE_FINGERPRINT={override!r}",
            flush=True,
        )
        return override

    _dir = pkg_dir if pkg_dir is not None else Path(__file__).resolve().parent
    dumps = []
    for name in _STORE_CODE_MODULES:
        try:
            dumps.append(_ast_normalized_dump((_dir / name).read_text()))
        except Exception as e:
            print(f"[snapshot_store] WARNING: could not fingerprint {name}: {e}", flush=True)
            return "unknown"
    # Backend implementations (bfcl_eval side). Only folded in when the test seam is
    # NOT in use, so `pkg_dir`-based unit tests keep their historical values.
    if pkg_dir is None:
        _bdir = _backend_pkg_dir()
        if _bdir is None:
            print("[snapshot_store] WARNING: backend package not importable; "
                  "cannot fingerprint memory backends", flush=True)
            return "unknown"
        for name in _BACKEND_CODE_MODULES:
            try:
                dumps.append(_ast_normalized_dump((_bdir / name).read_text()))
            except Exception as e:
                print(f"[snapshot_store] WARNING: could not fingerprint {name}: {e}", flush=True)
                return "unknown"
    return hashlib.sha256("\x00".join(dumps).encode()).hexdigest()[:16]



# ── Write-repair operator: adapter + accessor (plan T2) ───────────────────────
# The BACKEND ADAPTER lives here, deliberately: the operator must contain no benchmark
# facts, so the schema, the budget and the store state are assembled at the call site
# where a live memory instance exists. A different benchmark supplies a different adapter
# and reuses the operator unchanged.
_WRITE_TOOL_SCHEMA = {
    "kv": [
        {"tool": "core_memory_add", "args": {"key": "str", "value": "str"}, "writes_to": "core"},
        {"tool": "archival_memory_add", "args": {"key": "str", "value": "str"},
         "writes_to": "archival"},
    ],
    "vector": [
        {"tool": "core_memory_add", "args": {"text": "str"}, "writes_to": "core"},
        {"tool": "archival_memory_add", "args": {"text": "str"}, "writes_to": "archival"},
    ],
    "rec_sum": [
        {"tool": "memory_append", "args": {"text": "str"}, "writes_to": "blob"},
        {"tool": "memory_update", "args": {"text": "str"}, "writes_to": "blob"},
    ],
}


def _backend_caps():
    """Constraint constants read from the BACKEND MODULES, not redeclared here.

    They were referenced by name in a first draft of the adapter without being imported,
    which raised NameError into a bare `except` and silently produced a None budget -- a
    guard swallowing its own bug, the same shape as the §105 detector defect. Reading them
    from the source of truth also means a benchmark change to a cap needs no edit here.
    """
    caps = {"ENTRY": 300, "ARCH_ENTRY": 2000, "CORE_SIZE": 7, "ARCH_SIZE": 50, "BLOB": 10000}
    try:
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import memory_kv as _kv
        caps["ENTRY"] = _kv.MAX_CORE_MEMORY_ENTRY_LENGTH
        caps["ARCH_ENTRY"] = _kv.MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH
        caps["CORE_SIZE"] = _kv.MAX_CORE_MEMORY_SIZE
        caps["ARCH_SIZE"] = _kv.MAX_ARCHIVAL_MEMORY_SIZE
    except Exception:
        pass
    try:
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import (
            memory_rec_sum as _rs)
        caps["BLOB"] = _rs.MAX_MEMORY_ENTRY_LENGTH
    except Exception:
        pass
    # vector declares its OWN identical MAX_* constants. Read them rather than inheriting kv's:
    # the values agree today, and a benchmark change to one backend must not silently be reported
    # against another. Imported last and only to CONFIRM, so a divergence surfaces as a mismatch
    # here rather than as a wrong budget three layers down.
    try:
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import (
            memory_vector as _vec)
        caps["VECTOR_ENTRY"] = _vec.MAX_CORE_MEMORY_ENTRY_LENGTH
        caps["VECTOR_ARCH_ENTRY"] = _vec.MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH
        caps["VECTOR_CORE_SIZE"] = _vec.MAX_CORE_MEMORY_SIZE
        caps["VECTOR_ARCH_SIZE"] = _vec.MAX_ARCHIVAL_MEMORY_SIZE
    except Exception:
        pass
    return caps


def write_repair_adapter(backend: str) -> Dict[str, Any]:
    """The BACKEND ADAPTER, as data. Five things, per the frozen contract:

        constrained_object   what must be rewritten
        operation            the call template that applies the rewrite
        constraint           the schema-derived cap and its kind
        state_needed         which live fields compute the admissible budget
        validator_kind       how a candidate is checked mechanically

    The core never reads this dict's VALUES for meaning -- it passes them to the operator and
    to the validator. Adding a backend is adding an entry here; no core change.

    rec_sum is NOT a special case. It is the same Tier-3 mechanism with a different schema:
    the object is the blob rather than one field, the constraint is aggregate rather than
    per-entry, and the operation replaces rather than inserts. Same trigger class ("the store
    refused this write"), same action ("produce a valid write"), different extraction.
    """
    caps = _backend_caps()
    if backend == "rec_sum":
        return {
            "backend": backend,
            "constrained_object": "store",          # the whole blob
            "object_field": "memory",               # where to read it from mem_inst
            "operation": "memory_update(text={value})",
            "constraint": {"kind": "aggregate", "cap": caps["BLOB"], "unit": "characters"},
            "state_needed": ["memory"],
            "validator_kind": "length_and_numeric_facts",
        }
    return {
        "backend": backend,
        "constrained_object": "argument",           # one field of the failing call
        "object_field": "value" if backend == "kv" else "text",
        "operation": ("archival_memory_add(key={key}, value={value})" if backend == "kv"
                      else "archival_memory_add(text={value})"),
        "constraint": {"kind": "per_entry", "cap": caps["ARCH_ENTRY"], "unit": "characters"},
        "state_needed": ["core_memory", "archival_memory"],
        "validator_kind": "length_and_numeric_facts",
    }


def _parse_call_args(call_str):
    """`name(k='v', j="w")` -> {'k': 'v', 'j': 'w'}. Values may contain quotes and commas.

    Uses ast where the call parses as an expression, and falls back to a scanner otherwise.
    A regex was tried first and got this wrong twice: a non-greedy quote pattern truncated
    `text='User's schedule; 12 credits'` to `User`, and the greedy repair matched nothing.
    """
    s = (call_str or "").strip()
    try:
        node = ast.parse(s, mode="eval").body
        if isinstance(node, ast.Call):
            out = {}
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    out[kw.arg] = kw.value.value
                else:
                    out[kw.arg] = ast.unparse(kw.value)
            return out
    except Exception:
        pass
    # Scanner fallback: walk the argument list tracking the opening quote of each value.
    out, i = {}, s.find("(")
    if i < 0:
        return out
    i += 1
    while i < len(s):
        m = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])").match(s, i)
        if not m:
            break
        name, q = m.group(1), m.group(2)
        j = m.end()
        buf = []
        while j < len(s):
            if s[j] == "\\" and j + 1 < len(s):
                buf.append(s[j + 1]); j += 2; continue
            if s[j] == q:
                nxt = re.compile(r"\s*[,)]").match(s, j + 1)
                if nxt:
                    break
            buf.append(s[j]); j += 1
        out[name] = "".join(buf)
        k = s.find(",", j)
        if k < 0:
            break
        i = k + 1
    return out


def _capture_constraint_state(decoded, execution_results, involved_instances, backend):
    """Live state at a CONSTRAINT VIOLATION, as declared by the backend adapter.

    Returns None when no call in this step violated a constraint -- so the field appears only
    on the steps a repair would act on, rather than on every step of every episode.

    GENERIC BY CONSTRUCTION. The core does not know what "blob" or "core memory" mean: it reads
    the attribute names the adapter lists in `state_needed`, measures whatever it finds, and
    records it. Adding a backend is an adapter entry, not a change here. What is captured is
    exactly the four things a repair needs, plus the derived budget:

        constrained content + its size, the incoming item, the schema limit, remaining budget

    `remaining_budget` is reported BOTH ways because the two differ by whether the operation
    replaces or extends, and that distinction is the adapter's to make, not the core's:
    append-style writes get `limit - current`, replace-style writes get the whole `limit`.
    """
    # Locus classification lives in scripts/constraint_locus.py (stdlib-only, no eval stack).
    # Imported lazily and tolerantly: instrumentation must never be the reason a rollout dies,
    # and a missing classifier degrades to recording the raw error rather than nothing.
    _locus_of = None
    try:
        _cl_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if _cl_dir not in sys.path:
            sys.path.insert(0, _cl_dir)
        from constraint_locus import constraint_locus as _locus_of  # type: ignore
    except Exception:
        _locus_of = None

    caps = _backend_caps()
    adapter = write_repair_adapter(backend) or {}
    state_needed = list(adapter.get("state_needed") or ())
    out = []

    for idx, call in enumerate(decoded or []):
        result = execution_results[idx] if idx < len(execution_results or []) else ""
        rtext = str(result)
        if '"error"' not in rtext and "error" not in rtext.lower():
            continue
        args = _parse_call_args(str(call))
        # Locus classification is delegated; if unavailable, fall back to recording the raw
        # error so the capture is never silently empty on a real violation.
        locus = None
        if _locus_of is not None:
            try:
                locus = _locus_of(rtext, args)
            except Exception:
                locus = None

        # Measure every declared state field on every involved instance that has it.
        observed = {}
        for inst in (involved_instances or {}).values() if isinstance(involved_instances, dict) \
                else (involved_instances or []):
            for field in state_needed:
                if not hasattr(inst, field):
                    continue
                val = getattr(inst, field)
                # A STORE OBJECT is measured through the dict it holds, not skipped.
                #
                # This branch is the reason the vector cell emitted ZERO constraint_state over 629
                # prereq steps while raising 160 real capacity errors (kv 259 records, rec_sum 274).
                # `hasattr` PASSES on vector -- `core_memory` and `archival_memory` exist -- but they
                # are `VectorStore` OBJECTS whose dict lives at `._store`, so every isinstance branch
                # below missed, `observed` stayed empty, and the `if not observed: continue` at the
                # bottom of the loop dropped the record. Every signal reading `constrained_field`,
                # `schema_limit` or `current_size` was therefore blind on one of three cells and
                # presented as "no opportunity" -- which is exactly how A5 was nearly written off.
                #
                # Unwrapped FIRST and by attribute name, mirroring
                # `anchoropt.mechanisms.lossless_eviction._live_container`, which has handled both
                # shipped shapes correctly all along. `max_size` / `max_entry_length` are recorded
                # when the store carries them so the limit comes from the STORE rather than from a
                # sibling backend's constant.
                store_caps = {}
                inner = getattr(val, "_store", None)
                if isinstance(inner, dict):
                    for _cap_attr in ("max_size", "max_entry_length"):
                        _cv = getattr(val, _cap_attr, None)
                        if isinstance(_cv, int):
                            store_caps[_cap_attr] = _cv
                    val = inner
                if isinstance(val, str):
                    observed[field] = {"kind": "text", "size": len(val), "content": val}
                elif isinstance(val, dict):
                    observed[field] = {"kind": "mapping", "size": len(val),
                                       "keys": sorted(map(str, val.keys()))[:60]}
                elif isinstance(val, (list, tuple)):
                    observed[field] = {"kind": "sequence", "size": len(val)}
                if field in observed and store_caps:
                    observed[field]["store_caps"] = store_caps
        if not observed:
            continue

        incoming = max((v for v in args.values() if isinstance(v, str)),
                       key=len, default="")
        # The schema limit that applies: text-valued state is bounded by a character cap,
        # collection-valued state by an item-count cap. Both come from the adapter's caps.
        primary = next((f for f in state_needed if f in observed), None)
        obs = observed.get(primary) or {}
        if obs.get("kind") == "text":
            limit = caps.get("BLOB")
        else:
            limit = caps.get("CORE_SIZE") if "core" in (primary or "") else caps.get("ARCH_SIZE")
        # A store that declares its own cap is authoritative for itself. Without this the limit for
        # a vector container would be read off kv's module-level constants -- equal today, and a
        # silent misreport the moment one backend's schema moves.
        _sc = (obs.get("store_caps") or {})
        if _sc.get("max_size") is not None and obs.get("kind") != "text":
            limit = _sc["max_size"]
        cur = obs.get("size")
        rec = {
            "locus": list(locus) if locus else None,
            "error": rtext[:200],
            "call": str(call)[:160],
            "backend": backend,
            "constrained_field": primary,
            "current_size": cur,
            "incoming_size": len(incoming),
            "incoming": incoming[:400],
            "schema_limit": limit,
            # Two budgets, because the adapter -- not the core -- decides which applies.
            "remaining_budget_if_extends": (limit - cur) if (limit and cur is not None) else None,
            "remaining_budget_if_replaces": limit,
            "state": observed,
        }
        out.append(rec)
    return out or None


def build_write_repair_ctx(spec, failing_call, execution_results, mem_inst, backend,
                           mechanical=None):
    """Adapter: schema + schema-derived budget + live store state, read from mem_inst.

    Budgets come from the backend module's own constants, never from the error text -- for
    rec_sum the error text says "less than 10000 characters" while the entry is already 14,
    because the cap is on the AGGREGATE and the blob began at capacity.
    """
    err = ""
    for r in (execution_results or []):
        if gate_match(spec, str(r)):
            err = str(r)
            break
    schema = list(_WRITE_TOOL_SCHEMA.get(backend or "", ()))
    caps = _backend_caps()
    full, budget, alt, keys = [], None, [], []
    try:
        if backend == "rec_sum":
            blob = getattr(mem_inst, "memory", "") or ""
            budget = caps["BLOB"] - len(blob)
            if budget <= 0:
                full.append("blob")
                # memory_update REPLACES the blob, so a replacement write has the whole cap.
                alt.append("blob")
                budget = caps["BLOB"]
        else:
            core = getattr(mem_inst, "core_memory", None)
            arch = getattr(mem_inst, "archival_memory", None)
            if core is not None and len(core) >= caps["CORE_SIZE"]:
                full.append("core")
            if arch is not None and len(arch) >= caps["ARCH_SIZE"]:
                full.append("archival")
            else:
                alt.append("archival")
            budget = caps["ARCH_ENTRY"] if "core" in full else caps["ENTRY"]
            if isinstance(arch, dict):
                keys = sorted(arch.keys())[:40]
    except (AttributeError, TypeError, KeyError) as e:
        print(f"[write_repair] adapter could not read state: {type(e).__name__}: {e}",
              flush=True)
    ad = write_repair_adapter(backend or "")
    # The OBJECT to rewrite, chosen by the adapter -- an argument of the failing call, or the
    # whole store. This is the line that makes rec_sum ordinary rather than special.
    obj = None
    if ad["constrained_object"] == "store":
        obj = str(getattr(mem_inst, ad["object_field"], "") or "")
    else:
        _m = re.search(r"(?:%s)\s*=\s*(['\"])(.*?)\1" % ad["object_field"],
                       str(failing_call), re.S)
        obj = _m.group(2) if _m else None
    return {"failing_call": str(failing_call), "error_text": err, "schema": schema,
            "mechanical": mechanical, "adapter": ad, "object": obj,
            "state": {"budget": budget, "full_stores": full,
                      "alternative_stores": alt, "existing_keys": keys}}


class MemoryAnchorOptEvaluator:
    """Evaluator for BFCL v4 agentic memory tasks with AnchorOpt injection.

    Parameters
    ----------
    handler:
        BFCL model handler (e.g. GraniteHandler) already initialised with the
        model name and temperature.
    vllm_url:
        Base URL of the running vLLM server (e.g. "http://localhost:8080").
    served_model_name:
        Model ID as returned by GET /v1/models (used in the POST body).
    max_steps_per_turn:
        Maximum tool-call steps per turn before the episode is force-quit.
    """

    def __init__(
        self,
        handler,
        vllm_url: str,
        served_model_name: str,
        max_steps_per_turn: int = 20,  # W2 fidelity: match official MAXIMUM_STEP_LIMIT
        snapshot_cache_dir: Optional[Path] = None,
        disable_gates: bool = False,
        enable_reroute: bool = False,
        enable_forced_retrieval: bool = False,
        enable_forced_key_search: bool = False,
        store_workers: int = 1,  # W1 determinism: prereq store builds are SERIAL by
                                  # default (see _resolve_store_workers) — the store is
                                  # the state every decisional eval reads, and concurrent
                                  # continuous batching is not byte-stable even at
                                  # temperature 0. <=0 restores the old uncapped-parallel
                                  # behavior; a positive N caps the pool at N chains.
        seed: int = 42,
        # Decode params — hoisted out of _query so a --model-config can override
        # them without touching code. Defaults reproduce the previous inline
        # literals exactly (byte-identical stock).
        temperature: float = 0,
        max_tokens_cap: int = 4096,
        max_tokens_floor: int = 256,
        stop: Optional[list] = None,
    ):
        self.handler = handler
        self.vllm_url = vllm_url.rstrip("/")
        self.served_model_name = served_model_name
        self.max_steps_per_turn = max_steps_per_turn
        # 2x2-ablation flag (gates x templates), training-side only — base_handler.py
        # has no equivalent and never will, since it exists to isolate the templates-
        # only contribution during evaluation, not to describe a trained policy. Every
        # `not self.disable_gates` guard pairs with a `gate_enabled(templates, key)`
        # check for exactly one of MASKABLE_GATE_KEYS (on_loop via _loop_signal, which
        # already embeds its own gate_enabled term — see the loop_signal comment
        # below), so as of Phase 4's on_loop fix, disable_gates=True and a policy's
        # own gate_default=False (with no per-key override) now disable the identical
        # effect-set. That closes the exact divergence the plan flagged ("today
        # disable_gates suppresses the on_loop fallback and a mask would not"), but
        # the two remain independent mechanisms, not aliases: disable_gates is a hard,
        # policy-blind, all-or-nothing override that only exists here; gate_default is
        # part of the trained artifact itself, transfers to base_handler.py, and stays
        # overridable per key even when set. See
        # test_disable_gates_and_gate_default_off_cover_identical_gate_set for the
        # pinned equivalence.
        self.disable_gates = disable_gates
        # Phase 3/6 opt-in remedies (default off — byte-identical to Phase 0-2 stock).
        # Every firing site resolves the effective value via remedy_enabled(policy, name)
        # — policy alone, default=False, byte-identical to base_handler.py's call sites
        # (model_handler/base_handler.py:724,820,891, which pass no default either). These
        # attributes are NOT read by remedy_enabled; they are consulted only once, at
        # run start in run_memory_train.py/run_memory_eval.py, to bake the CLI-level
        # run-wide default into the templates/initial_templates dict itself (so a run
        # started with e.g. --enable-reroute is self-describing in templates_best.json
        # even if the gate-arm search never explicitly touches that arm). From then on
        # every candidate — including ones the arm search flips in EITHER direction per
        # batch — carries its own explicit key, so training and the leaderboard path
        # never disagree about what a trained policy means.
        self.enable_reroute = enable_reroute
        self.enable_forced_retrieval = enable_forced_retrieval
        # Phase 7: G4's forced remedy. Same policy-alone contract as the two above.
        self.enable_forced_key_search = enable_forced_key_search
        self.store_workers = store_workers
        # W1 determinism: fixed decode seed sent on every vLLM request so
        # accept/reject decisions are not made on run-to-run sampling noise.
        self.seed = seed
        self.temperature = temperature
        self.max_tokens_cap = max_tokens_cap
        self.max_tokens_floor = max_tokens_floor
        # Empty by default → no "stop" field is added to the request payload,
        # keeping the vLLM request byte-identical to the pre-hoist behavior.
        self.stop = list(stop) if stop else []
        self._snapshot_cache_dir = Path(snapshot_cache_dir) if snapshot_cache_dir else None
        self._snapshot_dir = Path(tempfile.mkdtemp(prefix="anchoropt_snap_"))
        self._store_diag: Dict = {}  # chain_id → {clear_calls, archival_adds, full_errors}

        import requests as _req
        try:
            models = _req.get(f"{self.vllm_url}/v1/models", timeout=5).json()
            self._max_model_len = models["data"][0].get("max_model_len", 32768)
        except Exception:
            self._max_model_len = 32768

    # ── vLLM query ────────────────────────────────────────────────────────────

    def _query(self, inference_data: dict) -> Tuple[object, float]:
        """POST to vLLM /v1/completions and return a response-like object + latency."""
        import requests
        import time

        messages = [
            {k: v for k, v in m.items() if k != "_anchoropt_ephemeral"}
            for m in inference_data["message"]
        ]
        formatted_prompt = self.handler._format_prompt(messages, inference_data["function"])
        approx_input_tokens = len(formatted_prompt) // 4
        max_tokens = max(self.max_tokens_floor,
                         min(self.max_tokens_cap, self._max_model_len - approx_input_tokens - 2))

        # W1 determinism: greedy decode + fixed seed. temperature 0.001 was
        # effectively-but-not-exactly greedy and left ±5-6pp noise that the
        # acceptance gate could not absorb. Values come from __init__ (defaults
        # reproduce the previous inline literals). "stop" is added only when
        # non-empty so the default payload is byte-identical to before.
        payload = {
            "model": self.served_model_name,
            "prompt": formatted_prompt,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": max_tokens,
        }
        if self.stop:
            payload["stop"] = self.stop

        t0 = time.time()
        resp = requests.post(
            f"{self.vllm_url}/v1/completions",
            json=payload,
            # Non-streaming POST: requests waits for the FULL completion. At this
            # server's ~17-25 tok/s, a long greedy decode (max_tokens up to 4096)
            # takes ~240s worst-case, so a 120s timeout deterministically aborted
            # long prereq steps (e.g. memory_kv_prereq_10), poisoning the chain and
            # failing the whole snapshot build. 600s gives ~2.5x margin.
            timeout=600,
        )
        latency = time.time() - t0

        if not resp.ok:
            if resp.status_code == 400 and ("input_tokens" in resp.text or "context" in resp.text.lower()):
                class _EmptyChoice:
                    text = ""
                class _EmptyUsage:
                    prompt_tokens = 0
                    completion_tokens = 0
                class _Empty:
                    choices = [_EmptyChoice()]
                    usage = _EmptyUsage()
                return _Empty(), latency
            raise RuntimeError(f"vLLM {resp.status_code}: {resp.text[:400]}")

        payload = resp.json()
        usage = payload.get("usage", {})
        content = payload["choices"][0].get("text", "")

        class _Usage:
            def __init__(self, p, c):
                self.prompt_tokens = p
                self.completion_tokens = c

        class _Choice:
            def __init__(self, t):
                self.text = t

        class _ApiResp:
            def __init__(self, text, p, c):
                self.choices = [_Choice(text)]
                self.usage = _Usage(p, c)

        return _ApiResp(content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)), latency

    # ── BFCL tool execution ───────────────────────────────────────────────────

    def _execute(
        self,
        decoded_calls: list,
        initial_config: dict,
        involved_classes: list,
        model_name: str,
        test_entry_id: str,
    ) -> Tuple[list, dict]:
        """Execute decoded tool calls via BFCL multi_turn_utils.

        execute_multi_turn_func_call handles singleton instantiation from
        initial_config and involved_classes — the same mechanism used by
        base_handler.py lines 421-428.
        """
        if execute_multi_turn_func_call is None:
            raise RuntimeError(
                "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils not importable. "
                "Ensure BFCL is in sys.path."
            )
        with _quiet_snapshot_warnings():
            return execute_multi_turn_func_call(
                func_call_list=decoded_calls,
                initial_config=initial_config,
                involved_classes=involved_classes,
                model_name=model_name,
                test_entry_id=test_entry_id,
                long_context=False,
                is_evaL_run=False,
            )

    def _write_repair_operator(self):
        """Lazily construct the operator. Returns None when unavailable, so a missing
        dependency degrades to the mechanical path rather than crashing an arm."""
        if getattr(self, "_wr_op", "unset") != "unset":
            return self._wr_op
        self._wr_op = None
        try:
            import importlib.util as _ilu
            _p = Path(__file__).resolve().parent.parent / "scripts" / "op_write_repair.py"
            _sp = _ilu.spec_from_file_location("_op_write_repair", _p)
            _m = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(_m)
            self._wr_op = _m.WriteRepairOperator(enabled=True)
        except Exception as e:
            print(f"[write_repair] operator unavailable: {type(e).__name__}: {e}", flush=True)
        return self._wr_op

    def _try_capacity_repair(
        self,
        decoded_calls: list,
        execution_results: list,
        involved_instances: dict,
        test_category: str,
        initial_config: dict,
        involved_classes: list,
        model_name: str,
        test_entry_id: str,
    ) -> Optional[Dict[str, Any]]:
        """POST-EXECUTION capacity repair for an AGGREGATE container (N0-A1).

        Same mined locus as the G3 reroute -- capacity/container/no_remaining_capacity --
        at the same incision point, on a backend whose container is an aggregate rather
        than a collection of items. G3 relocates the item into a second container; where
        no second container exists the repair REPLACES the aggregate with a version that
        admits the item.

        GENERIC. Everything backend-specific arrives through `capacity_repair.BACKENDS`:
        which error text signals the condition, which operation applies the replacement,
        which schema field bounds it, and how a candidate is validated. This method
        contributes control flow only -- it never names a tool, a limit, or a backend.

        Deliberately NOT a prompt-style nudge. It dispatches a mechanically validated
        call, so nothing depends on the model choosing to comply.

        Returns the payload dict (result + replaced + substituted + shed_chars), or None
        when no repair applies -- in which case the caller falls through untouched.
        """
        try:
            import sys as _sys
            _d = str(Path(__file__).resolve().parent.parent / "scripts")
            if _d not in _sys.path:
                _sys.path.insert(0, _d)
            import capacity_repair as _cr
        except Exception:
            return None

        backend = memory_backend(test_category)
        if not backend or not involved_instances:
            return None
        mem_inst = list(involved_instances.values())[0]
        caps = _backend_caps()

        # Find a call whose result reports this backend's capacity condition. The adapter
        # owns the matching, so an unfamiliar error text simply does not match.
        for idx, call in enumerate(decoded_calls or []):
            result = execution_results[idx] if idx < len(execution_results or []) else ""
            rtext = result.get("error", "") if isinstance(result, dict) else str(result)
            if not rtext:
                continue
            cond = _cr.match_condition(backend, rtext)
            if cond is None:
                continue

            args = _parse_call_args(str(call))
            # Live state for the constrained object, read by the name the adapter declares.
            store_text = None
            obj = cond.get("object")
            if obj == "store":
                for field in (write_repair_adapter(backend) or {}).get("state_needed") or ():
                    val = getattr(mem_inst, field, None)
                    if isinstance(val, str):
                        store_text = val
                        break
                if store_text is None:
                    continue

            repaired = _cr.repair(backend, args, rtext, caps, {}, store_text=store_text)
            # repair() returns {"ok": False, "reason": ...} when the constraint cannot be
            # satisfied (e.g. the fact-carrying content alone exceeds the cap). That is a
            # legitimate "no admissible repair" answer, not an error, and must not be read
            # as a truthy success -- the dict is truthy either way.
            if not repaired or not repaired.get("ok"):
                continue

            new_call = repaired.get("call")
            if not new_call:
                continue
            try:
                # _execute returns (results, state) -- every other call site unpacks the
                # pair. Treating the tuple as a list would have made `res` the results
                # LIST, so the error checks below would never match and a failed dispatch
                # would have been recorded as a successful repair.
                _results, _ = self._execute(
                    [new_call], initial_config, involved_classes, model_name, test_entry_id,
                )
            except Exception:
                return None
            res = _results[0] if _results else ""

            # A repair whose dispatch ERRORS is not a repair: report None so the caller
            # falls through rather than recording a substitution that did not take.
            if isinstance(res, dict) and res.get("error"):
                return None
            if isinstance(res, str) and '"error"' in res:
                return None
            # What the write WOULD have produced, versus what the repair produced. For a
            # replace-style repair the pre-repair size is store + incoming (the sum that
            # violated the cap), not the store alone -- using the store alone reported a
            # NEGATIVE shed when the replacement legitimately kept everything but the
            # overflow margin.
            _incoming = max((v for v in args.values() if isinstance(v, str)),
                            key=len, default="")
            before = len(store_text or "") + len(_incoming) if store_text is not None else \
                len(args.get(cond.get("reduce_arg") or "", "") or "")
            after = repaired.get("payload_len")
            return {"result": res, "replaced_call": str(call)[:300],
                    "substituted_call": str(new_call)[:300],
                    "shed_chars": (before - after) if (after is not None) else None}
        return None

    def _try_core_full_reroute(
        self,
        decoded_calls: list,
        execution_results: list,
        involved_instances: dict,
        test_category: str,
        initial_config: dict,
        involved_classes: list,
        model_name: str,
        test_entry_id: str,
    ) -> Optional[str]:
        """Phase 3 (opt-in via the enable_reroute policy flag, defaulting to
        self.enable_reroute): synthesize a corrected
        `archival_memory_add(...)` call from the failing `core_memory_add(...)` call
        that just returned a core/archival-full error, and re-dispatch it through
        self._execute (mirrors base_handler.py's _try_core_full_reroute, using the
        SAME executor so it mutates the same cached live instance).

        Returns the (stringified) execution result on success, or None if the
        reroute could not be synthesized or the re-dispatch itself failed — callers
        must fall back to the existing G3 reprompt text in that case.
        """
        failing_call = find_failing_core_full_call(decoded_calls, execution_results)
        if failing_call is None:
            return None

        backend = memory_backend(test_category)
        mem_inst = list(involved_instances.values())[0]
        synth_call = synthesize_core_full_reroute(failing_call, mem_inst, backend)
        if synth_call is None:
            return None

        reroute_results, _ = self._execute(
            [synth_call],
            initial_config,
            involved_classes,
            model_name,
            test_entry_id,
        )
        reroute_result = reroute_results[0] if reroute_results else ""
        if not reroute_result or is_error_result(reroute_result):
            return None
        # Return WHAT WAS SUBSTITUTED, not just the result (§37). Returning the result
        # alone discards the only evidence of what the gate actually did: on the frozen
        # G1 rerun the reroute arm recorded `reroute_gate` 284 times with no substitution
        # payload, so its R0 was INDETERMINATE rather than measurable -- the §24.5
        # "remedy worked, telemetry vanished" pattern, third occurrence.
        return {
            "result": reroute_result,
            "replaced_call": str(failing_call)[:200],
            "substituted_call": str(synth_call)[:200],
        }

    def _try_reroute(
        self,
        spec,
        decoded_calls: list,
        execution_results: list,
        involved_instances: dict,
        test_category: str,
        initial_config: dict,
        involved_classes: list,
        model_name: str,
        test_entry_id: str,
    ):
        """Generic, spec-driven reroute: the contract to match and the destination to
        substitute both come from `spec`, not from a hardcoded key.

        This method DID NOT EXIST while the G3 reroute arm ran (§52). The generic loop
        called `self._try_reroute` and the class only defined `_try_core_full_reroute`, so
        every vector prereq raised AttributeError during store construction and the arm
        produced no accuracy number at all. Two distinct defects, recorded separately
        because only the first is a typo:

          1. the method was never written, though a comment claimed it "now returns a dict";
          2. even present, delegating to the G1 helpers would have matched only the
             core-full contract, so it would have found no failing call for the G3 signal
             and returned None every time -- scored as "reroute never fires" rather than as
             a bug.

        Returns the §37 payload dict (result + what was replaced + what was substituted) so
        R0 is measurable, or None if no substitute could be synthesized -- in which case the
        caller falls through to the advisory family and the fact is not dropped.
        """
        failing_call = find_failing_call_for_spec(spec, decoded_calls, execution_results)
        if failing_call is None:
            return None

        backend = memory_backend(test_category)
        if not involved_instances:
            return None
        mem_inst = list(involved_instances.values())[0]
        synth_call = synthesize_reroute_for_spec(spec, failing_call, mem_inst, backend)
        # ── CONDITIONAL operator fallback (opt-in) ────────────────────────────
        # The mechanical synthesizer runs FIRST and, when it succeeds, the operator is never
        # invoked -- that ordering is what makes this a fallback rather than a first resort.
        # synthesize_reroute_for_spec is ~30 lines of per-pair glue (parse core_memory_add,
        # sanitise the key to kv's format, dedupe, check one hardcoded cap, build
        # archival_memory_add): it encodes benchmark facts as CODE. The operator takes the
        # same facts as runtime INPUTS -- the tool schema, the tool's own error text, the live
        # state -- so a backend with a different constraint shape needs no new code.
        #
        # Enabled only when the policy asks for it, so history-only and history+operator stay
        # a measurable comparison rather than a belief (llm_operator.py rule 7).
        if synth_call is None and remedy_enabled(
                getattr(self, "_active_templates", None) or {},
                "enable_write_repair_operator"):
            _op = self._write_repair_operator()
            if _op is not None:
                _ctx = build_write_repair_ctx(spec, failing_call, execution_results,
                                              mem_inst, backend, mechanical=None)
                _res = _op.run(_ctx)
                _ok = [c for c in _res.candidates if c.get("status") == "valid"]
                self._operator_log = getattr(self, "_operator_log", [])
                self._operator_log.append({"invoked": _res.invoked, "reason": _res.reason,
                                           "model": _res.model, "latency_s": _res.latency_s,
                                           "n_valid": len(_ok),
                                           "candidates": _res.candidates[:2]})
                if _ok:
                    synth_call = _ok[0]["call"]
        if synth_call is None:
            return None

        reroute_results, _ = self._execute(
            [synth_call],
            initial_config,
            involved_classes,
            model_name,
            test_entry_id,
        )
        reroute_result = reroute_results[0] if reroute_results else ""
        if not reroute_result or is_error_result(reroute_result):
            return None
        return {
            "result": reroute_result,
            "replaced_call": str(failing_call)[:200],
            "substituted_call": str(synth_call)[:200],
        }

    def _try_forced_archival_retrieval(
        self,
        test_category: str,
        initial_config: dict,
        involved_classes: list,
        model_name: str,
        test_entry_id: str,
    ) -> Optional[str]:
        """Phase 6 (opt-in via the enable_forced_retrieval policy flag, defaulting to
        self.enable_forced_retrieval): mirrors
        base_handler.py's _try_forced_archival_retrieval, using self._execute.
        Simpler than the reroute remedy — both target calls are static and
        zero-arg, no failing call to parse and no live-instance introspection.
        """
        backend = memory_backend(test_category)
        synth_call = synthesize_forced_retrieval_call(backend)
        if synth_call is None:
            return None

        results, _ = self._execute(
            [synth_call],
            initial_config,
            involved_classes,
            model_name,
            test_entry_id,
        )
        result = results[0] if results else ""
        if not result or is_error_result(result):
            return None
        return result

    def _try_forced_key_search(
        self,
        test_category: str,
        initial_config: dict,
        involved_classes: list,
        model_name: str,
        test_entry_id: str,
    ) -> Optional[str]:
        """Phase 7 (opt-in via the enable_forced_key_search policy flag, defaulting to
        self.enable_forced_key_search): mirrors base_handler.py's _try_forced_key_search,
        using self._execute. After a KV retrieve failed with "Key not found", run
        archival_memory_list_keys() on the model's behalf so it can see the real key
        names. kv only (G4's backend scope); None on an unsupported backend or a failed
        dispatch, in which case callers fall back to the existing G4 text.
        """
        synth_call = synthesize_forced_key_search_call(memory_backend(test_category))
        if synth_call is None:
            return None

        results, _ = self._execute(
            [synth_call],
            initial_config,
            involved_classes,
            model_name,
            test_entry_id,
        )
        result = results[0] if results else ""
        if not result or is_error_result(result):
            return None
        return result

    def _try_transform(self, spec, decoded, execution_results, inference_data,
                       templates) -> Optional[Dict]:
        """Generic GENERATIVE transform: rewrite a rejected call's argument.

        The first action family that is generative rather than structural. suppress
        deletes a call and reroute substitutes one; transform keeps the call and the
        intent, and rewrites the ARGUMENT so the same content satisfies a constraint
        the tool rejected.

        Generic over the spec: which errors trigger it comes from
        `spec.match_substrings`, and what the rewrite must achieve comes from
        `transform_instruction(spec.key, ...)`. Adding a transform gate for a
        different constraint therefore needs no execution code.

        Every rewrite is validated by `transform_ok()` before use. A generative
        action can fail in ways a structural one cannot — return prose, exceed the
        limit anyway, or drop the facts — and writing a corrupted value is worse than
        not acting, so a failed validation falls through to the model's own retry.
        """
        import re as _re
        # find the rejected call and its offending argument
        for call, res in zip(decoded or [], execution_results or []):
            rtext = str(res)
            if not any(m in rtext for m in (spec.match_substrings or ())):
                continue
            m = _re.search(r"(?:text|value)\s*=\s*'(.*)'\s*\)", str(call), _re.S) or \
                _re.search(r'(?:text|value)\s*=\s*"(.*)"\s*\)', str(call), _re.S)
            if not m:
                return None
            original = m.group(1)
            instr = transform_instruction(spec.key, rtext)
            if not instr:
                return None
            # Ask the model for the rewrite, on a scratch copy of the conversation so
            # the transform prompt never pollutes the episode's message history.
            scratch = {k: v for k, v in inference_data.items()}
            scratch["message"] = list(inference_data.get("message") or []) + [
                {"role": "user", "content": f"{instr}\n\nValue to rewrite:\n{original}"}
            ]
            try:
                api_response, _lat = self._query(scratch)
                parsed = self.handler._parse_query_response_prompting(api_response)
                rewritten = str(parsed.get("model_responses") or "")
            except Exception as e:
                # No `rewritten` exists, so this must NOT be recorded as a fixpoint:
                # a transient query failure says nothing about whether the branch is
                # exhausted. Signalled by omitting `rewritten` entirely.
                return {"ok": False, "reason": f"transform query failed: {type(e).__name__}",
                        "original": original, "error_text": rtext}
            ok, why = transform_ok(original, rewritten, rtext)
            # `original`, `rewritten` and the matched error text are returned on BOTH
            # paths because the §25.4 fixpoint key is (signal, gate, input, output) and
            # the REJECTED path is exactly where it is needed: the loop guard has to
            # recognise a rewrite it has already produced and refused. Returning them
            # only on success would make every key ("", gate, "", "") and collapse all
            # states into one.
            if not ok:
                return {"ok": False, "reason": why, "original_len": len(original),
                        "original": original, "rewritten": rewritten,
                        "error_text": rtext}
            new_val = rewritten.strip().strip('"').strip("'")
            # ESCAPING (defect found on the W1 arm, job 911257). The previous version did a bare
            #     str(call).replace(original, new_val)
            # which produces an UNPARSEABLE call whenever the rewritten value contains the quote
            # character delimiting the argument:
            #     core_memory_add(text="User is the Managing Director ...")  + a rewrite holding "
            #     -> "invalid syntax. Perhaps you forgot a comma? (<string>, line 1)"
            # and the write is then LOST -- strictly worse than not acting, because the original was
            # already rejected and the model no longer gets its own retry.
            #
            # Escape for the delimiter ACTUALLY used around the matched argument, and escape
            # backslashes first so an existing backslash cannot corrupt the literal either.
            _call_s = str(call)
            _i = _call_s.find(original)
            _delim = '"'
            if _i > 0:
                # the character immediately before the value is its opening quote
                _prev = _call_s[_i - 1]
                if _prev in ("'", '"'):
                    _delim = _prev
            _esc = new_val.replace("\\", "\\\\").replace(_delim, "\\" + _delim)
            new_call = _call_s.replace(original, _esc)
            # VERIFY IT PARSES before handing it to the executor. A refused rewrite falls through to
            # the model's own retry; a dispatched broken one loses the content outright.
            try:
                import ast as _ast
                _ast.parse(new_call.strip(), mode="eval")
            except SyntaxError as _se:
                return {"ok": False,
                        "reason": "synthesized call does not parse (%s); refusing rather than "
                                  "dispatching a broken write" % type(_se).__name__,
                        "original_len": len(original), "original": original,
                        "rewritten": rewritten, "error_text": rtext}
            return {"ok": True, "reason": why, "call": new_call,
                    "original_len": len(original), "new_len": len(new_val),
                    "original": original, "rewritten": rewritten,
                    "error_text": rtext}
        return None

    # ── Episode state reset ───────────────────────────────────────────────────

    def _reset_episode_state(
        self,
        model_name: str,
        test_entry_id: str,
        involved_classes: List[str],
    ) -> None:
        """Clear BFCL global instance singletons for this (model_name, test_entry_id).

        IMPORTANT: only wipe the exact (model_name, test_entry_id) singletons.
        Do NOT clear all memory singletons — prereq chains rely on instance
        persistence across entries within the same chain.

        multi_turn_utils stores instances as module-level globals under the key:
          re.sub(r'[-./:]', '_', f"{model_name}_{test_entry_id}_{class_name}_instance")
        There is no public delete API — manipulate __dict__ directly.
        """
        try:
            import sys
            mtu = sys.modules.get(
                "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils"
            )
            if mtu is None:
                return
            for class_name in involved_classes:
                key = re.sub(
                    r"[-./:]", "_",
                    f"{model_name}_{test_entry_id}_{class_name}_instance",
                )
                mtu.__dict__.pop(key, None)
        except Exception:
            pass

    # ── Snapshot store ────────────────────────────────────────────────────────

    def build_snapshot_store(self, prereq_cases: List[Dict], templates: Optional[Dict] = None) -> None:
        """Run prereq episodes upfront to populate the on-disk snapshot store.

        Chains (backend × scenario) run via a ThreadPoolExecutor whose width is
        self.store_workers (default 1 — serial; see _resolve_store_workers).
        Ordering is enforced within each chain (prereq_0 → prereq_1 → ...).
        Results are cached to disk keyed by prereq IDs + model name +
        on_memory_preamble text + store_workers — subsequent calls with the same
        inputs reuse saved snapshots.

        Args:
            prereq_cases: All prereq entries to run.
            templates: Template dict to apply during storage (on_memory_preamble
                is injected into the system message for each prereq episode).
        """
        import json as _json
        import os as _os
        import shutil as _shutil
        import uuid as _uuid

        templates = templates or {}
        tmpl_map = templates.get("templates") or {}
        preamble_text = tmpl_map.get("on_memory_preamble", "")
        _gate_off = list(disabled_gate_keys(templates))
        # Installed activation="explicit" gates. disabled_gate_keys cannot see these:
        # it reports gates turned OFF, which suffices only while every gate is on by
        # default. An explicit gate is off unless INSTALLED, so enabling it is the
        # behaviour-changing event, and without this term an installed learned gate
        # hashes identically to a policy that never mentions it -- reusing the
        # incumbent's store and being scored against the incumbent's memory state
        # (§Stage 5 snapshot trap). Appended CONDITIONALLY below so a policy that
        # installs nothing hashes exactly as before.
        _gate_on_explicit = list(enabled_explicit_gate_keys(templates))
        # EFFECTIVE remedy values, not the constructor defaults: both remedies are now
        # policy-overridable per candidate (see __init__), and both can fire DURING a
        # prereq/storage episode — reroute triggers on exactly the core-full error the
        # storage phase produces, and forced retrieval triggers on an empty/IDK step,
        # which a storage turn can also produce. So a candidate that flips either one
        # must MISS the incumbent's store, or the two arms get scored against
        # byte-identical memory state and the measured delta is not the effect of the
        # flip (the same hazard the gate_enabled mask had to fix).
        _reroute_on = remedy_enabled(templates, "enable_reroute")
        _forced_on = remedy_enabled(templates, "enable_forced_retrieval")
        # Phase 7, same reasoning: G4's forced key search fires on a "Key not found"
        # error, which a storage episode can produce whenever the model probes a key
        # before writing it; and both new NUMERIC thresholds change WHEN a gate that
        # can fire during prereqs (on_blob_pressure during rec_sum storage, on_loop in
        # any episode) fires. All three are appended CONDITIONALLY and LAST below, so a
        # policy that leaves them alone hashes exactly as before and every pre-existing
        # snap_* dir stays valid.
        _fks_on = remedy_enabled(templates, "enable_forced_key_search")
        _thresholds = list(threshold_overrides(templates))

        # Part A: resolved pool width the build will actually use (see
        # _run_prereq_build). Folded into base_key UNCONDITIONALLY below — not the
        # usual conditional-append idiom — because every pre-existing snap_* dir was
        # built in parallel with no such token at all; a conditional append would let
        # a new serial run silently HIT one of those, which is the exact hazard this
        # exists to prevent. Also written to the manifest so "was this store serial?"
        # is answerable from any storage_keys.json on disk.
        _n_chains_for_key = max(len(_group_prereqs_by_chain(prereq_cases)), 1)
        _resolved_store_workers = _resolve_store_workers(
            getattr(self, "store_workers", 1), _n_chains_for_key
        )

        # Part B: code-version fingerprint of the behavioural module closure a
        # prereq episode executes. Folded into base_key UNCONDITIONALLY below,
        # same rationale as store_workers — every pre-existing snap_* dir has no
        # such token, so a conditional append would let a stale cross-commit
        # store be silently reused (the 2026-07-31 granite incident). Also
        # enforced explicitly in the HIT-scan below (not just via base_key
        # partitioning) and recorded in the manifest.
        _code_fp = _store_code_fingerprint()

        # Part C: canonical fingerprint of the WHOLE gate configuration.
        # SIGNAL_POLICY_PLAN §8.3 prerequisite. The conditional tokens above cover
        # only the knobs that existed when each was written — `gate_off:`, `fks:`,
        # `thr:`, and the three positional bools. A NEWLY INSTALLED gate key
        # matches none of them, so an iteration that installs a gate while leaving
        # every template string byte-identical would silently cache-HIT and score
        # the new policy against the INCUMBENT's memory state. That is a
        # plausible-looking wrong accept, not a crash, and it would invalidate the
        # residual-learning experiment without any visible symptom.
        #
        # Folded in UNCONDITIONALLY, matching the store_workers/code_fp rationale:
        # every pre-existing snap_* dir predates this token, so a conditional
        # append would let a stale store be reused by exactly the runs this is
        # meant to protect. This DOES invalidate existing caches once — deliberate,
        # and cheaper than one wrong accept.
        _gate_cfg_fp = _gate_config_fingerprint(templates or {})

        # ── Base cache key ────────────────────────────────────────────────
        # Folds in disable_gates / enable_reroute / enable_forced_retrieval: all
        # three change whether the storage-side gates fire during prereqs, so two
        # arms differing only in those flags must NOT share a store. on_memory_preamble
        # is in here too (it seeds the prereq system prompt); every OTHER
        # teacher-editable text is handled by the per-variant layer below. gate_off
        # (per-gate enable mask) is appended conditionally and last so a policy with
        # no gate_enabled section hashes identically to before this was added.
        # The three bools keep their historical POSITIONS, so a policy that sets
        # neither remedy key hashes exactly as before (every pre-existing snap_* dir
        # stays valid); only a policy that actually flips one splits the key.
        base_key = hashlib.sha256(
            _json.dumps(
                sorted(c["id"] for c in prereq_cases)
                + [self.served_model_name, preamble_text,
                   bool(self.disable_gates), bool(_reroute_on),
                   bool(_forced_on)]
                + (["gate_off:" + ",".join(_gate_off)] if _gate_off else [])
                + (["gate_on:" + ",".join(_gate_on_explicit)] if _gate_on_explicit else [])
                + (["fks:1"] if _fks_on else [])
                + (["thr:" + ",".join(_thresholds)] if _thresholds else [])
                + [f"store_workers:{_resolved_store_workers}"]
                + [f"code_fp:{_code_fp}"]
                + [f"schema:{_ANCHOROPT_STORE_SCHEMA}"]
                + [f"gate_cfg:{_gate_cfg_fp}"]
            ).encode()
        ).hexdigest()[:16]

        # ── Cache provenance (§105) ────────────────────────────────────────
        # A reused store is an INPUT to every number a run produces, so the decision
        # to reuse must be as auditable as the result. Records identity, the verdict,
        # and — when it MISSES — the specific reason, because "rebuilt" and "rebuilt
        # for a reason I did not expect" look identical in a log otherwise.
        _prov = {
            "verdict": None,
            "base_key": base_key,
            "cache_dir": str(self._snapshot_cache_dir) if self._snapshot_cache_dir else None,
            "source_variant": None,
            "store_code_fingerprint": _code_fp,
            "gate_config_fingerprint": _gate_cfg_fp,
            "anchoropt_store_schema": _ANCHOROPT_STORE_SCHEMA,
            "served_model_name": self.served_model_name,
            "preamble_sha256": hashlib.sha256(preamble_text.encode()).hexdigest(),
            "n_prereqs": len(prereq_cases),
            "prereq_ids_sha256": hashlib.sha256(
                _json.dumps(sorted(c["id"] for c in prereq_cases)).encode()).hexdigest()[:16],
            "store_workers": _resolved_store_workers,
            "miss_reasons": [],
            "candidates_examined": 0,
        }
        self._cache_provenance = _prov

        def _emit_prov():
            """One structured line per run, plus a file beside the store."""
            print("[snapshot_store][provenance] " + _json.dumps(_prov, sort_keys=True),
                  flush=True)
            print(
                f"[snapshot_store][provenance] verdict={_prov['verdict']} "
                f"base_key={_prov['base_key']} variant={_prov['source_variant']} "
                f"code_fp={_prov['store_code_fingerprint']} "
                f"gate_cfg={_prov['gate_config_fingerprint']} "
                f"prereqs={_prov['n_prereqs']}({_prov['prereq_ids_sha256']}) "
                f"cache={_prov['cache_dir']}",
                flush=True,
            )
            for r in _prov["miss_reasons"]:
                print(f"[snapshot_store][provenance]   miss_reason: {r}", flush=True)

        self._emit_cache_provenance = _emit_prov

        # ── No-cache path: build straight into the ephemeral tempdir ───────
        # Byte-identical to the historical no-cache behavior (state lands in the
        # per-instance tempdir; no sentinel, no reuse across calls).
        if self._snapshot_cache_dir is None:
            _prov["verdict"] = "NO_CACHE"
            _prov["miss_reasons"].append("snapshot_cache_dir is None (caching disabled)")
            _emit_prov()
            self._run_prereq_build(prereq_cases, templates, record_observed=None)
            print("[snapshot_store] Done.", flush=True)
            return

        base_dir = self._snapshot_cache_dir / f"snap_{base_key}"

        # ── Fast-path cache check: scan sibling variant dirs ───────────────
        # A completed variant HITs iff, for every storage-trigger key it recorded
        # during its own build, this candidate's RAW template text is byte-equal.
        # Raw-text compare is conservative: at worst a needless rebuild, never an
        # unsafe reuse. An empty observed-key set (the common null-ish case) HITs
        # vacuously, so all such arms share one variant dir.
        if not base_dir.exists():
            _prov["miss_reasons"].append(
                f"no store exists for base_key={base_key} under {self._snapshot_cache_dir} "
                f"(first run of this exact corpus/model/policy/code combination)")
        if base_dir.exists():
            _vdirs = sorted(base_dir.glob("v_*"))
            _prov["candidates_examined"] = len(_vdirs)
            if not _vdirs:
                _prov["miss_reasons"].append(
                    f"base_key dir {base_dir.name} exists but holds no v_* variant")
            for vdir in _vdirs:
                manifest_f = vdir / "storage_keys.json"
                if not (manifest_f.exists() and (vdir / ".build_complete").exists()):
                    _prov["miss_reasons"].append(
                        f"{vdir.name}: incomplete (build_complete="
                        f"{(vdir / '.build_complete').exists()}, manifest="
                        f"{manifest_f.exists()}) -- a partial store is never reused")
                    continue
                try:
                    with open(manifest_f) as _f:
                        manifest = _json.load(_f)
                except Exception as _e:
                    _prov["miss_reasons"].append(f"{vdir.name}: manifest unreadable ({_e})")
                    continue
                # Defense-in-depth beyond base_key partitioning: a variant dir
                # can land on disk (hand-crafted, or from a schema this evaluator
                # predates) whose fingerprint doesn't match the current code even
                # though it happens to share base_key's hash bucket. Absent counts
                # as mismatch — this is what makes a stale store loud instead of
                # silently reused (the failure mode base_key alone can't catch).
                _manifest_fp = manifest.get("store_code_fingerprint")
                if _manifest_fp != _code_fp:
                    print(
                        f"[snapshot_store] Skipping {vdir.name}: code fingerprint "
                        f"mismatch (manifest={_manifest_fp!r}, current={_code_fp!r}).",
                        flush=True,
                    )
                    _prov["miss_reasons"].append(
                        f"{vdir.name}: store_code_fingerprint {_manifest_fp} != current "
                        f"{_code_fp} -- store predates a change to what a prereq build "
                        f"produces (this is the guard that refuses pre-BM25-fix stores)")
                    continue
                key_texts = manifest.get("key_texts", {})
                _diff = [k for k, txt in key_texts.items() if tmpl_map.get(k, "") != txt]
                if not _diff:
                    print(
                        f"[snapshot_store] Cache hit ({base_key[:8]}/{vdir.name}) — "
                        f"skipping {len(prereq_cases)} prereqs.",
                        flush=True,
                    )
                    _prov["verdict"] = "HIT"
                    _prov["source_variant"] = vdir.name
                    _prov["source_path"] = str(vdir)
                    _prov["prereqs_skipped"] = len(prereq_cases)
                    _prov["observed_storage_keys"] = manifest.get("observed_storage_keys", [])
                    _emit_prov()
                    self._snapshot_dir = vdir
                    diag_file = vdir / "store_diag.json"
                    if diag_file.exists():
                        with open(diag_file) as _f:
                            self._store_diag = _json.load(_f)
                    return
                _prov["miss_reasons"].append(
                    f"{vdir.name}: storage-trigger template text differs on "
                    f"{len(_diff)} key(s): {sorted(_diff)[:5]}")

        # ── MISS: build into a transient dir, then atomically publish ──────
        _prov["verdict"] = "MISS"
        _emit_prov()
        # variant_key is only knowable AFTER the build (it depends on which
        # storage triggers fired), so never name the final dir up front.
        base_dir.mkdir(parents=True, exist_ok=True)
        building = base_dir / f".building_{_uuid.uuid4().hex[:12]}"
        building.mkdir(parents=True, exist_ok=True)

        observed: set = set()
        ok = self._run_prereq_build(prereq_cases, templates,
                                    record_observed=observed, out_dir=building)
        if not ok:
            # Partial store — do NOT publish (no rename, no sentinel). The
            # unpublished .building_ dir is left on disk for inspection; the next
            # call re-MISSes and rebuilds cleanly. Raising here is deliberate: a
            # partial store must never be silently scored against.
            raise RuntimeError(
                f"[snapshot_store] Prereq build failed (base_key={base_key[:8]}); "
                f"store not published. Partial dir left at {building}."
            )

        # ── Compute variant_key + write manifest INTO the transient dir ────
        key_texts = {k: tmpl_map.get(k, "") for k in sorted(observed)}
        variant_key = hashlib.sha256(
            _json.dumps(sorted(key_texts.items())).encode()
        ).hexdigest()[:12]
        target = base_dir / f"v_{variant_key}"

        with open(building / "store_diag.json", "w") as _f:
            _json.dump(self._store_diag, _f, sort_keys=True)
        manifest = {
            "base_key": base_key,
            "variant_key": variant_key,
            "prereq_ids": sorted(c["id"] for c in prereq_cases),
            "served_model_name": self.served_model_name,
            "preamble_sha256": hashlib.sha256(preamble_text.encode()).hexdigest(),
            "disable_gates": bool(self.disable_gates),
            "enable_reroute": bool(_reroute_on),
            "enable_forced_retrieval": bool(_forced_on),
            "enable_forced_key_search": bool(_fks_on),
            "gate_off": _gate_off,
            "gate_on_explicit": _gate_on_explicit,
            "threshold_overrides": _thresholds,
            "observed_storage_keys": sorted(observed),
            "key_texts": key_texts,
            "store_workers": _resolved_store_workers,
            "store_code_fingerprint": _code_fp,
            "gate_config_fingerprint": _gate_cfg_fp,
            "anchoropt_store_schema": _ANCHOROPT_STORE_SCHEMA,
        }
        with open(building / "storage_keys.json", "w") as _f:
            _json.dump(manifest, _f, indent=2)
        # Provenance of the BUILD that produced this store, persisted so it outlives
        # the job log. A later HIT on this dir reports source_variant pointing here.
        _prov["variant_key"] = variant_key
        _prov["observed_storage_keys"] = sorted(observed)
        with open(building / "cache_provenance.json", "w") as _f:
            _json.dump(_prov, _f, indent=2, sort_keys=True)

        # ── Publish atomically (same parent → os.replace is atomic) ────────
        if target.exists():
            # A sibling arm published the identical variant first — adopt it.
            print(
                f"[snapshot_store] Variant already present "
                f"({base_key[:8]}/v_{variant_key}) — adopting, discarding transient build.",
                flush=True,
            )
            _shutil.rmtree(building, ignore_errors=True)
        else:
            try:
                _os.replace(building, target)
            except OSError:
                # Lost a publish race; adopt whatever landed there.
                _shutil.rmtree(building, ignore_errors=True)
                if not target.exists():
                    raise
        # Sentinel written LAST, into the published dir — a reader treats a dir
        # without .build_complete as incomplete and skips it.
        (target / ".build_complete").touch()
        self._snapshot_dir = target
        diag_file = target / "store_diag.json"
        if diag_file.exists():
            with open(diag_file) as _f:
                self._store_diag = _json.load(_f)
        print(f"[snapshot_store] Done. Cached → {target}", flush=True)

    def _run_prereq_build(
        self,
        prereq_cases: List[Dict],
        templates: Dict,
        record_observed: Optional[set] = None,
        out_dir: Optional[Path] = None,
    ) -> bool:
        """Run the prereq episodes that populate the on-disk snapshot store.

        Chains (backend × scenario) run at a pool width of
        _resolve_store_workers(self.store_workers, n_chains) — 1 (serial) by
        default. Ordering is enforced within each chain (prereq_0 → prereq_1 →
        ...). Episode state lands in self._snapshot_dir, so out_dir (when given)
        is assigned to it first.

        record_observed: if a set is passed, storage-trigger keys that fired
        during the build are merged into it (see evaluate_episode.record_triggers).

        Returns True iff every prereq episode completed without an exception. A
        False return means the store is PARTIAL and must not be published — the
        old in-eval prereq re-run used to self-heal this; filtering removes that.
        """
        import concurrent.futures

        if out_dir is not None:
            self._snapshot_dir = out_dir
        self._store_diag = {}  # fresh per build; a rejected candidate's diag must not linger

        # ── Group by chain, sort within each chain ────────────────────────
        # Chain = (backend, scenario); ordering within a chain is enforced by
        # _group_prereqs_by_chain. chain_items is sorted by chain id (not left in
        # dict/file order) so build order — and therefore _store_diag's insertion
        # order — is a function of the case ids, not of scheduling.
        chains = _group_prereqs_by_chain(prereq_cases)
        chain_items = sorted(chains.items())
        total = len(prereq_cases)
        n_chains = len(chain_items)
        if total == 0:
            return True
        max_workers = _resolve_store_workers(getattr(self, "store_workers", 1), n_chains)
        _mode = "serial" if max_workers <= 1 else f"parallel, {max_workers} workers"
        print(
            f"[snapshot_store] Building {total} prereqs across "
            f"{n_chains} chains ({_mode}) ...",
            flush=True,
        )

        completed = [0]
        progress_lock = threading.Lock()
        failures: List = []

        def _is_infra(e) -> bool:
            # Mirrors _eval_batch's infra-vs-genuine classifier (run_memory_train.py).
            _ename, _emsg = type(e).__name__, str(e)
            return (
                "Timeout" in _ename
                or "Connection" in _ename
                or _emsg.startswith("vLLM 5")
                or "timed out" in _emsg.lower()
                or "connection" in _emsg.lower()
            )

        def _run_chain(chain_id, cases):
            diag = {"clear_calls": 0, "archival_adds": 0, "full_errors": 0}
            chain_triggers = set() if record_observed is not None else None
            for case in cases:
                try:
                    _, traj = self.evaluate_episode(
                        case, templates, rollout_tag="snap",
                        record_triggers=chain_triggers,
                    )
                    # Stage 0 sidecar: persist the storage-phase trajectory. These
                    # episodes are where ~83% of kv/vector failures originate, and
                    # they never reach _eval_batch, so this is the only place they
                    # can be captured. No-op unless ANCHOROPT_TRAJ_DIR is set.
                    try:
                        from anchoropt.traj_sidecar import dump_episode
                        dump_episode(case, traj, phase="prereq",
                                     extra={"chain_id": chain_id})
                    except Exception:
                        pass
                    for step in traj:
                        decoded_str = str(step.get("decoded", ""))
                        tool_str = str(step.get("tool_results", ""))
                        if "core_memory_clear" in decoded_str:
                            diag["clear_calls"] += 1
                        if "archival_memory_add" in decoded_str:
                            diag["archival_adds"] += 1
                        if "is full" in tool_str or "exceeds" in tool_str:
                            diag["full_errors"] += 1
                except Exception as e:
                    _infra = _is_infra(e)
                    print(
                        f"  {'INFRA' if _infra else 'ERR'} snap {case['id'][:60]}: {e}",
                        flush=True,
                    )
                    with progress_lock:
                        failures.append((case["id"], _infra, str(e)))
            self._store_diag[chain_id] = diag
            with progress_lock:
                completed[0] += len(cases)
                if record_observed is not None and chain_triggers:
                    # In-place update, NOT |= : `record_observed |= …` rebinds the
                    # name, which Python would treat as local to this nested function
                    # (UnboundLocalError). .update() mutates the closed-over set.
                    record_observed.update(chain_triggers)
                print(
                    f"[snapshot_store] {completed[0]}/{total} ({chain_id} done)",
                    flush=True,
                )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as pool:
            futs = [pool.submit(_run_chain, cid, cases) for cid, cases in chain_items]
            concurrent.futures.wait(futs)
            for fut in futs:
                if fut.exception():
                    print(f"  ⚠ chain error: {fut.exception()}", flush=True)
                    failures.append(("<chain>", False, str(fut.exception())))

        return len(failures) == 0

    # ── Main evaluation entry point ───────────────────────────────────────────

    def evaluate_episode(
        self,
        test_case: Dict,
        templates: Dict,
        rollout_tag: str = "eval",
        record_triggers: Optional[set] = None,
    ) -> Tuple[Dict, List[Dict]]:
        """Run one memory episode with AnchorOpt injection and grade the result.

        Parameters
        ----------
        test_case:
            Fully-resolved BFCL v4 memory case dict as produced by
            process_memory_test_case + populate_test_cases_with_predefined_functions.
            Required keys: id, initial_config, involved_classes, question (list of
            turn message lists), possible_answer, function, scenario.
        templates:
            AnchorOpt template dict loaded from templates_initial.json.
        rollout_tag:
            Suffix appended to the model name for BFCL's instance registry.
        record_triggers:
            Build-only: if a set is passed, every storage-affecting template key
            whose *condition* fired during this episode is added to it, reported
            regardless of whether that key's tunable text was empty. Used by
            build_snapshot_store to compute a variant cache key so that an
            empty→non-empty edit of an opt-in storage signal (G5/blob) invalidates
            the cached store. None (default) = no recording = zero overhead on the
            normal scoring path.

        Returns
        -------
        checker_result : dict
        trajectory_history : list[dict]
        """
        test_case = copy.deepcopy(test_case)

        rollout_model_name = re.sub(
            r"[-./:]", "_",
            f"{self.handler.model_name_underline_replaced}_{rollout_tag}",
        )

        initial_config: Dict = dict(test_case.get("initial_config", {}))
        involved_classes: List[str] = test_case["involved_classes"]
        test_entry_id: str = test_case["id"]
        # test_category mirrors base_handler.py line 403
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        # Inject snapshot-system keys required by memory_api_metaclass._prepare_snapshot.
        # execute_multi_turn_func_call extracts initial_config.get(class_name, {}) and passes
        # that nested dict to class_instance._load_scenario → _prepare_snapshot.
        # Training cases have initial_config={} so the nested dicts are also empty.
        _ic0 = involved_classes[0] if involved_classes else ""
        _scenario = test_case.get("scenario") or (
            _ic0.split("MemoryAPI_", 1)[-1] if "MemoryAPI_" in _ic0 else "kv"
        )
        for _cls in involved_classes:
            _cls_cfg = initial_config.setdefault(_cls, {})
            _cls_cfg.setdefault("model_result_dir", self._snapshot_dir)
            _cls_cfg.setdefault("test_id", test_entry_id)
            _cls_cfg.setdefault("scenario", _scenario)

        # ── Step 1: bootstrap — creates the MemoryAPI singleton and loads snapshot ─
        # Mirrors base_handler.py lines 421-428.  Must happen BEFORE
        # add_memory_instruction_system_prompt (which reads _dump_core_memory_to_context).
        if execute_multi_turn_func_call is None:
            raise RuntimeError("BFCL multi_turn_utils not importable.")

        with _quiet_snapshot_warnings():
            _, involved_instances = execute_multi_turn_func_call(
                [],
                initial_config,
                involved_classes,
                rollout_model_name,
                test_entry_id,
                long_context=False,
                is_evaL_run=False,
            )

        # ── Step 2: inject memory system prompt (core memory contents + instructions) ─
        # Mirrors base_handler.py lines 431-442.  Modifies test_case["question"] in place.
        if add_memory_instruction_system_prompt is not None:
            memory_instance = list(involved_instances.values())[0]
            test_case["question"] = add_memory_instruction_system_prompt(
                test_case["question"],
                test_category,
                _scenario,
                memory_instance,
            )
        else:
            memory_instance = list(involved_instances.values())[0]

        # on_memory_preamble: prepend proactive instruction to the system message (empty = no-op).
        # Must operate on test_case["question"][0] here, before _pre_query_processing_prompting
        # below — that call returns an empty message list for every prompting-path handler in
        # use, so injecting into its output would silently never find a system-role message.
        _preamble = (templates.get("templates") or {}).get("on_memory_preamble", "")
        if _preamble and _preamble.strip():
            for _msg in test_case["question"][0]:
                if isinstance(_msg, dict) and _msg.get("role") == "system":
                    _msg["content"] = _preamble.strip() + "\n\n" + _msg["content"]
                    break

        # ── Step 3: build inference_data from the (now-modified) test_case ──────────
        # Mirrors base_handler.py line 465.
        inference_data = self.handler._pre_query_processing_prompting(test_case)

        # ── Per-episode injection state ───────────────────────────────────────
        prev_snapshot = None
        curr_snapshot: Dict = {}
        failure_streak = 0
        error_streak = 0
        prev_error_signal = None
        last_result = ""
        last_call_str = ""
        recent_tools: deque = deque(maxlen=5)
        # maxlen single-sourced from the registry (was a bare 4) so base_handler.py's
        # Phase 7 mirror of this window cannot drift from it.
        recent_call_signatures: deque = deque(maxlen=RECENT_CALL_SIGNATURE_WINDOW)
        failed_search_streak = 0

        all_model_responses: List[List[str]] = []
        trajectory_history: List[Dict] = []
        force_quit = False
        all_turn_messages: List[List[Dict]] = test_case["question"]
        # §25.4 layer 2 -- local outcome memory for outcome-aware escalation, keyed on
        # (signal, gate, input value, output value). EPISODE-scoped deliberately, not
        # per-turn: once a rewrite has been produced and rejected, an IDENTICAL output
        # cannot produce a different outcome, so the branch is provably exhausted for
        # the rest of the episode. Per-turn scope would re-run the same refuted rewrite
        # every turn, which is the §25.2 point that a latch alone still learns nothing
        # from having failed.
        #
        # Exact-output equality only, per §25.4: it is a logical fixpoint, needs no
        # threshold, and cannot overfit. The known incompleteness is accepted -- two
        # DIFFERENT outputs can represent the same exhausted branch (389 then 377, both
        # over 300), and this rule will not catch that. Widening requires a progress
        # metric and a stall threshold, deferred until a jittering-stall trace is
        # actually observed rather than guessed at now.
        _tx_refuted: set = set()

        for turn_idx, turn_messages in enumerate(all_turn_messages):
            # Skip system-prompt turns (role == "system"); only user turns drive the loop
            user_msgs = [m for m in turn_messages if isinstance(m, dict) and m.get("role") == "user"]
            if not user_msgs:
                continue

            if turn_idx == 0:
                inference_data = self.handler.add_first_turn_message_prompting(
                    inference_data, turn_messages
                )
            else:
                inference_data = self.handler._add_next_turn_user_message_prompting(
                    inference_data, turn_messages
                )

            user_message = user_msgs[-1].get("content", "")
            failed_search_streak = 0
            current_turn_responses: List[str] = []
            step_count = 0
            consecutive_decode_failures = 0
            # Per-turn gate applicability — each gate keyed off its OWN registry spec,
            # exactly as base_handler.py does (it calls gate_applies per gate). These are
            # all kv/vector today, but must NOT share a single flag: doing so couples the
            # gates' backend scope and would silently ignore a narrowed `backends` on any
            # one gate in the registry (breaking the single-source-of-truth invariant).
            _applies_premature_idk = gate_applies(REGISTRY_BY_KEY["on_premature_idk"], test_category)
            _applies_core_clear = gate_applies(REGISTRY_BY_KEY["on_core_clear_blocked"], test_category)
            _applies_core_full = gate_applies(REGISTRY_BY_KEY["on_domain_error_core_full"], test_category)
            _d3_archival_searched = False
            _d3_gate_fired = False
            _g3_fired = False   # W2 fidelity: core/archival-full redirect, once/turn
            _g4_fired = False   # W2 fidelity: KV key-not-found key-search fallback, once/turn
            # Reprompt-family latches, keyed by spec. Was a single `_g5_fired` bool
            # when the executor hardcoded one gate; a shared flag across gates would
            # couple their backend scope and let one gate's firing suppress another's.
            _rp_fired: set = set()
            # G3: per-spec latch for the generic recovery-substitution family.
            _rr_fired: set = set()
            _blob_pressure_fired = False  # Phase 2: proactive rec_sum compaction nudge, once/turn (opt-in)
            # Post-execution capacity repair (N0-A1). Separate latch from _blob_pressure_fired
            # because the two sit at DIFFERENT INCISION POINTS on the same locus: the blob
            # pressure gate is pre-generation (nudge before saturation), this is
            # post-execution (repair the rejected write). Sharing a latch would let a
            # pre-generation nudge suppress a post-execution repair.
            _capacity_repair_fired = False
            # A4v2: tool calls executed in this TURN, and a one-shot latch for the zero-call
            # intercept. These live in the per-turn scope, which is equivalent to per-query on
            # this corpus -- VERIFIED, not assumed: all 303 query episodes carry exactly one
            # turn. On a corpus with multi-turn queries the latch would reset mid-query and the
            # intercept could fire more than once per query, so this equivalence must be
            # re-checked rather than inherited.
            _query_tool_calls = 0
            _zero_call_reprompted = False
            # A5: the reads this query issued, and the best top-similarity seen. Recorded at
            # EXECUTION (like _query_tool_calls) so a call that was proposed but suppressed
            # pre-execution never counts as a retrieval that happened.
            #
            # `_query_read_sims` holds one entry per executed read: the max similarity_score in
            # its result, or None when the result carried no score (kv / rec_sum, which cannot
            # expose this feature at all).
            _query_read_sims = []
            _low_sim_reprompted = False
            # B1: consecutive-failure count per call text, and one shared per-episode budget.
            _b1_fail_counts = {}
            _b1_firings = 0
            # E1: archival contents observed this episode as (vec_id, normalized_value), in write
            # order, plus the eviction budget. The store is PRE-SEEDED by earlier episodes on the
            # chain, so this view is partial -- a duplicate living only in the pre-seeded portion is
            # invisible and E1 will not fire on it. That is consistent with the measured 23%
            # availability and is a stated limitation, not a defect.
            _e1_archival = []
            _e1_evictions = 0
            # §25.2 layer 1 -- the missing once-per-turn latch for the transform
            # family. Keyed by spec so several transform gates cannot starve each
            # other, and reset per turn like every other structural latch below.
            _tx_fired = set()

            while True:
                # ── AnchorOpt injection ───────────────────────────────────────
                # Phase 7: the membership test `sigs[-1] in sigs[:-1]` (fires on the very
                # first repeat) became a real COUNT with a policy-read threshold. The
                # default threshold of 1 reproduces that membership test exactly — see
                # loop_repeat_detected — so an unconfigured policy fires at the same step
                # as before. Detection now lives in the registry so base_handler.py's
                # Phase 7 loop gate reads the identical condition.
                #
                # Phase 3: a repeated retrieval call that keeps coming back empty
                # (e.g. list_keys() -> "keys": []) is not an error, so it both (a)
                # gets appended to recent_call_signatures and (b) trips
                # is_empty_or_failed_retrieval — meaning it used to satisfy on_loop's
                # "repeated successful call" signal AND on_idk_fallback's "repeated
                # failed search" signal on the same step. on_loop has ladder priority
                # (template_engine.py checks loop_signal before idk_fallback_signal),
                # so the model was always told "this already succeeded, don't repeat
                # it" on exactly the step where the honest message is "broaden your
                # search" — the opposite of what should happen. Excluding a failed/
                # empty retrieval from loop_signal (not from the deque itself — that
                # would blind a genuinely-looping model to its own failed calls, and
                # would silently change what RECENT_CALL_SIGNATURE_WINDOW means) lets
                # idk_fallback_signal win on that step instead.
                #
                # Phase 4: on_loop is the one soft key with a registry gate_fallback()
                # injected regardless of policy text, so it needs an explicit off
                # switch like the structural gates. Gating loop_signal itself (rather
                # than only the registry-fallback block below) turns off BOTH of
                # on_loop's injection surfaces at once: the soft-ladder path that
                # renders trained templates["on_loop"] text via get_injection_from_
                # templates, and the registry-fallback block that fires when that
                # resolves empty.
                _loop_signal = (
                    gate_enabled(templates, "on_loop")
                    and not _is_error(last_result)
                    and not is_empty_or_failed_retrieval(last_result, last_call_str)
                    and loop_repeat_detected(
                        recent_call_signatures, loop_repeat_threshold(templates)
                    )
                )
                idk_fallback_signal = failed_search_streak >= _IDK_FALLBACK_THRESHOLD

                _inj_result = get_injection_from_templates(
                    turn_idx,
                    step_count,
                    failure_streak,
                    prev_snapshot,
                    curr_snapshot,
                    inference_data.get("function", []),
                    list(recent_tools),
                    last_result,
                    templates,
                    error_streak=error_streak,
                    loop_signal=_loop_signal,
                    user_message=user_message,
                    idk_fallback_signal=idk_fallback_signal,
                    failed_search_streak=failed_search_streak,
                    # A4 (N0 line): the pre-generation signal for `no_tool_call_at_all`.
                    # `turn_needs_action_signal` existed in template_engine's ladder with a
                    # default of False and NO caller ever passed it -- declared but dead, so
                    # on_turn_start_action could never fire. The condition is simply "we are at
                    # the start of a turn and no tool has been called yet", which is exactly
                    # step_count == 0; the ladder already restricts the key to step_n == 0.
                    #
                    # Gated on the policy flag so this ships inert: without
                    # enable_turn_start_action the signal stays False and behaviour is
                    # byte-identical to before.
                    turn_needs_action_signal=(
                        step_count == 0
                        and remedy_enabled(templates, "enable_turn_start_action")),
                    return_key=True,
                    report_trigger=(record_triggers is not None),
                )
                if record_triggers is not None:
                    injection, injection_key, _trigger_key = _inj_result
                    # Record the soft-signal trigger key whose condition fired,
                    # independent of text emptiness, but only when it could actually
                    # inject for this backend — a backend-scoped-out key can't change
                    # this prereq's stored state, so its text needn't invalidate the
                    # store. (on_memory_preamble is already folded into the base key.)
                    if (
                        _trigger_key
                        and _trigger_key != "on_memory_preamble"
                        and (
                            _trigger_key not in REGISTRY_BY_KEY
                            or gate_applies(REGISTRY_BY_KEY[_trigger_key], test_category)
                        )
                    ):
                        record_triggers.add(_trigger_key)
                else:
                    injection, injection_key = _inj_result
                # T2: backend-aware soft-signal scoping. A soft template only injects
                # when its registry `backends` scope includes the live backend; keys
                # with backends=() (all) or domain="general" pass through unchanged, so
                # the null/unconfigured arm stays byte-identical stock. This is where
                # the registry-backed check lives (template_engine stays BFCL-free).
                if injection and injection_key and injection_key in REGISTRY_BY_KEY:
                    if not gate_applies(REGISTRY_BY_KEY[injection_key], test_category):
                        injection, injection_key = None, None

                # Phase 7: on_loop's registry fallback. The soft ladder only ever renders
                # POLICY text (empty template == silent), but on_loop now ships real
                # registry text and base_handler.py injects that text via
                # gate_fallback("on_loop") — so without this the two pipelines would
                # DISAGREE on the null arm, which is exactly the divergence class the
                # gate_fallback mirror tests exist to prevent. Reached only when the
                # ladder resolved on_loop and its text was empty: loop_signal has top
                # priority in the ladder, so `not injection and _loop_signal` implies the
                # resolved key was on_loop. A trained policy WITH on_loop text keeps the
                # ladder's slot-rendered output untouched. Suppressed under
                # disable_gates (this is registry gate text, and the 2x2's gates-off arm
                # must not receive it) and under gate_applies (the global env off-switch).
                if (
                    not injection
                    and _loop_signal
                    and not self.disable_gates
                    and gate_applies(REGISTRY_BY_KEY["on_loop"], test_category)
                ):
                    _loop_fb = gate_fallback("on_loop")
                    if _loop_fb.strip():
                        injection, injection_key = _loop_fb, "on_loop"

                if injection:
                    if step_count >= 1:
                        # trailing_user: add AFTER tool results from previous step (beats recency)
                        inference_data = self.handler._add_next_turn_user_message_prompting(
                            inference_data,
                            [{"role": "user", "content": injection}],
                        )
                    else:
                        # turn_start_user (step 0): ephemeral append to turn-opening user message
                        for msg in reversed(inference_data["message"]):
                            if msg.get("role") == "user":
                                msg["content"] = msg["content"] + "\n\n" + injection
                                msg["_anchoropt_ephemeral"] = True
                                break

                # ── Model query ───────────────────────────────────────────────
                api_response, latency = self._query(inference_data)
                model_response_data = self.handler._parse_query_response_prompting(api_response)
                model_response = model_response_data["model_responses"]

                inference_data = self.handler._add_assistant_message_prompting(
                    inference_data, model_response_data
                )

                # Strip ephemeral step-0 injection (trailing_user messages are permanent)
                if injection and step_count == 0:
                    for msg in inference_data["message"]:
                        if msg.get("_anchoropt_ephemeral") and msg.get("role") == "user":
                            msg["content"] = msg["content"].rsplit("\n\n" + injection, 1)[0]
                            del msg["_anchoropt_ephemeral"]

                current_turn_responses.append(model_response)

                step_record: Dict = {
                    "turn": turn_idx,
                    "step": step_count,
                    "assistant_response": model_response,
                    "latency": latency,
                    # Student token usage, as the handler already reports it. A handler that does
                    # not report usage yields None, which means NOT MEASURED -- never 0, because a
                    # missing measurement and a measured zero are different facts.
                    "input_token": model_response_data.get("input_token"),
                    "output_token": model_response_data.get("output_token"),
                    "injection": injection,
                    "injection_key": injection_key,  # T1: which signal key actually injected (None = silent)
                    "cell_key": _cell_key(turn_idx, step_count),
                    "loop_signal": _loop_signal,
                    "idk_fallback_signal": idk_fallback_signal,
                    "failed_search_streak": failed_search_streak,
                    "error_signal": prev_error_signal,
                    "error_streak": error_streak,
                }

                # ── Decode (has_tool_call_tag=False mirrors base_handler line 572) ──
                try:
                    decoded = self.handler.decode_execute(
                        model_response, has_tool_call_tag=False
                    )
                    model_response_data["model_responses_decoded"] = decoded
                    step_record["decoded"] = decoded

                    # D3 tracking: did this step access archival memory?
                    if _applies_premature_idk and archival_memory_searched(model_response):
                        _d3_archival_searched = True

                    # ── Pre-execution SUPPRESS executor, generic over GateSpecs ──
                    # DC ordering is signal -> policy -> execution. The signal is a
                    # spec's match_substrings and the policy is remedy_type="suppress";
                    # EXECUTION iterates whatever specs the registry+policy enable.
                    #
                    # Previously this block hardcoded "on_core_clear_blocked" at four
                    # points (enabled check, match check, filter, telemetry), mirrored
                    # in base_handler.py. A newly LEARNED suppress gate could then be
                    # correctly proposed and not installed -- the loop could not install
                    # what it could propose (AnchorOpt iteration 2 hit exactly this).
                    # Now a new suppress gate at this hook is pure registry data.
                    _sup_fired_text = None
                    if not self.disable_gates:
                        for _sup_spec in suppress_specs(templates, test_category):
                            if not gate_match_any(_sup_spec, decoded):
                                continue
                            # Per-gate user-intent exemption, keyed by gate: G1 skips
                            # when the user asked to forget everything. NOT inherited by
                            # other gates, whose semantics differ.
                            if suppress_user_exempt(_sup_spec.key, user_message):
                                continue
                            _n = len(decoded)
                            # ── A3: value-level redundancy predicate ──────────────────
                            # Existing suppress gates match on CALL NAME alone
                            # (core_memory_clear, archival_memory_remove) -- the call is
                            # unconditionally wrong. A3's call is not: an
                            # archival_memory_add is only wrong when the store ALREADY holds
                            # that exact key/value pair, which A1's reroute is what put
                            # there. So the name match selects candidates and this predicate
                            # decides, per call, using LIVE store state.
                            #
                            # Deliberately exact (normalised whitespace only), never
                            # semantic: 55/59 measured collisions are exact duplicates, and
                            # a fuzzy comparator would risk suppressing the 4 that carry
                            # genuinely new information. Those stay residual by design.
                            _pred = None
                            # ---- N0-R1: BOUNDED RECOVERY TRANSACTION -----------------------
                            # State machine per unresolved capacity failure on one container:
                            #   OPEN -> DIAGNOSE -> (<=1 authorized remove) -> VERIFY -> RETRY -> CLOSE
                            # The budget keys on the EPISODE, not (call, target): 10 of 12
                            # multi-proposal episodes switch verb or target, so a narrower key
                            # reproduces the substitution loophole.
                            #
                            # A refusal returns a STRUCTURED STATE OBSERVATION rather than a silent
                            # cancel. On the D1 population the tool already answers "ID 0 not present
                            # in store", so silence would give the model strictly LESS than it has
                            # today -- the measured mechanism that lengthens the re-proposal loop.
                            #
                            # NO new reclamation, merge, or rec_sum logic: D3 detects and refuses.
                            if _sup_spec.key == "on_destructive_authorization":
                                try:
                                    import sys as _s3
                                    _d3 = str(Path(__file__).resolve().parent.parent / "scripts")
                                    if _d3 not in _s3.path:
                                        _s3.path.insert(0, _d3)
                                    import recovery_txn as _rt
                                except Exception:
                                    _rt = None
                                if _rt is not None:
                                    if not hasattr(self, "_r1_txn"):
                                        self._r1_txn = {}
                                    _r1_obs = []

                                    def _pred(_call, _rt=_rt, _ii=involved_instances,
                                              _sr=step_record, _obs=_r1_obs):
                                        _cs = str(_call)
                                        _wh = _rt.container_of(_cs)
                                        _store = _rt.live_store(_ii, _wh)
                                        _v, _o = _rt.diagnose(_cs, _store, _wh)
                                        _txn = self._r1_txn.get(_wh)
                                        _sr.setdefault("recovery_txn_verdicts", []).append(_v)
                                        _sr["recovery_txn_live_size"] = len(_store)
                                        if _txn is None:
                                            # No open recovery episode: no unresolved failure, so no
                                            # authorization. Refuse and say why.
                                            _obs.append("No pending memory-capacity failure. " + _o)
                                            _sr.setdefault("recovery_txn_orphan", 0)
                                            _sr["recovery_txn_orphan"] += 1
                                            return True
                                        _txn.proposals += 1
                                        _txn.verdicts.append(_v)
                                        if _v in ("D1", "D2", "D3"):
                                            _obs.append(_o)
                                            _txn.refusals += 1
                                            return True
                                        # D4: capacity genuinely required.
                                        if "clear" in _cs:
                                            _obs.append(_o + " A clear would destroy every entry; "
                                                        "remove a single entry instead.")
                                            _txn.refusals += 1
                                            _sr["recovery_txn_clear_refused"] = True
                                            return True
                                        if _txn.authorized >= 1:
                                            _obs.append(_o + " One entry has already been removed for "
                                                        "this write; no further removal is authorized.")
                                            _txn.refusals += 1
                                            _sr["recovery_txn_budget_exhausted"] = True
                                            return True
                                        _txn.authorized += 1
                                        _sr["recovery_txn_authorized"] = True
                                        return False            # allow exactly this one

                            # C2: KV redundant-write suppression, as a PER-SPEC PREDICATE on the
                            # existing registry-driven executor rather than a parallel block. The
                            # comment above records why that matters: hardcoding a second suppress
                            # path is exactly the defect that once let the loop propose a gate it
                            # could not install.
                            #
                            # Measured on ground-truth kv state, A3 catches ZERO of the 24 T1 events:
                            #     6   call is core_memory_add, outside A3's match_substrings
                            #    18   value differs only in punctuation or clause order, which A3's
                            #         whitespace-only normalisation reads as different
                            # C2 closes both and NOTHING else. The single genuine MERGE case
                            # (lifestyle_changes) has different token content, is not matched, and its
                            # write proceeds -- merging would require composing a new value, a broader
                            # action space than suppression.
                            #
                            # A suppressed write is one the tool WOULD HAVE REJECTED with "Key name
                            # must be unique", so nothing that would otherwise have been stored is lost.
                            if (_sup_spec.key == "on_kv_redundant_write_suppressed"
                                    and remedy_enabled(templates,
                                                       "enable_kv_redundant_write_suppress")):
                                try:
                                    import sys as _s2
                                    for _d2 in (   # LOCAL FIX: see the A3 site below
                                        str(Path(__file__).resolve().parents[3] / "anchoropt" / "mechanisms"),
                                        str(Path(__file__).resolve().parent.parent / "scripts"),
                                    ):
                                        if _d2 not in _s2.path:
                                            _s2.path.insert(0, _d2)
                                    import redundant_write as _rw2
                                    _mi2 = (list(involved_instances.values())[0]
                                            if involved_instances else None)
                                    if _mi2 is not None:
                                        # Per-call REASONS, accumulated by the predicate so C2's
                                        # rationale stays separable from A3's. The generic executor
                                        # records the suppressed CALLS under suppress_removed_calls,
                                        # which is enough to re-score, but not enough to tell WHICH
                                        # mechanism fired -- canonical equivalence (C2's designed gap)
                                        # or container correctness (duplicates A3 misses because
                                        # store_lookup scans core first). Those are different findings
                                        # and must not be merged into one count.
                                        _c2_reasons = []

                                        def _pred(_call, _mi=_mi2, _m=_rw2, _acc=_c2_reasons):
                                            # Pass the TARGET CONTAINER. core and archival keys are
                                            # independent namespaces, so comparing across them would
                                            # suppress writes that LAND -- the dry run measured 264
                                            # such cases and refused to launch. The predicate itself
                                            # now refuses when the container is unknown.
                                            _cs = str(_call)
                                            _cont = ("core_memory" if "core_memory_add" in _cs
                                                     else "archival_memory"
                                                     if "archival_memory_add" in _cs else None)
                                            _ok, _w = _m.should_suppress_kv(
                                                _parse_call_args(_cs), _mi, container=_cont)
                                            if _ok:
                                                _acc.append("%s | %s" % (_cont, _w))
                                            return _ok
                                except Exception:
                                    _pred = None
                            elif remedy_enabled(templates, "enable_redundant_write_suppress"):
                                try:
                                    import sys as _s
                                    # LOCAL FIX (anchoropt repo): the mechanism is vendored to
                                    # anchoropt/mechanisms/, not <evaluator>/../scripts/. The original
                                    # path is kept as a fallback so this stays diffable upstream.
                                    # Getting this wrong is NOT a no-op: _pred stays None and
                                    # `_hits()` degenerates to `gate_match` alone, i.e. A3 suppresses
                                    # EVERY archival_memory_add instead of only exact duplicates.
                                    for _d in (
                                        str(Path(__file__).resolve().parents[3] / "anchoropt" / "mechanisms"),
                                        str(Path(__file__).resolve().parent.parent / "scripts"),
                                    ):
                                        if _d not in _s.path:
                                            _s.path.insert(0, _d)
                                    import redundant_write as _rwmod
                                    _minst = (list(involved_instances.values())[0]
                                              if involved_instances else None)
                                    if _minst is not None:
                                        def _pred(_call, _mi=_minst, _m=_rwmod):
                                            ok, _why = _m.should_suppress(
                                                _parse_call_args(str(_call)), _mi)
                                            return ok
                                except Exception:
                                    _pred = None

                            def _hits(_c, _spec=_sup_spec, _p=_pred):
                                if not gate_match(_spec, _c):
                                    return False
                                return _p(_c) if _p is not None else True

                            _removed = [c for c in decoded if _hits(c)]
                            decoded = [c for c in decoded if not _hits(c)]
                            if len(decoded) == _n:
                                continue
                            model_response_data["model_responses_decoded"] = decoded
                            step_record["decoded"] = decoded
                            # RECORD WHAT WAS SUPPRESSED (§28/§31). Overwriting
                            # step_record["decoded"] with the filtered list erases the
                            # offending call from the trajectory, so the record shows a
                            # gate that fired with no evidence of what it acted on. That
                            # is why G1's suppress arm could not be re-scored: A2 ("did
                            # the lever engage the target?") is underivable when the
                            # target call is gone. Keep both prefixed keys so the
                            # sidecar's suppress_ copy rule carries them.
                            step_record["suppress_removed_calls"] = [
                                str(c)[:200] for c in _removed]
                            step_record["suppress_removed_n"] = len(_removed)
                            # Per-spec telemetry flag; g1_gate stays the flag for G1 so
                            # every existing consumer and recorded result still reads.
                            step_record[_sup_spec.telemetry_flag or "suppress_gate"] = True
                            # C2 only: keep the per-call rationale under its own prefix, which is
                            # already in the sidecar whitelist.
                            if _sup_spec.key == "on_kv_redundant_write_suppressed":
                                try:
                                    if _c2_reasons:
                                        step_record["kv_redundant_reasons"] = _c2_reasons[:8]
                                        step_record["kv_redundant_removed_n"] = len(_removed)
                                        step_record["kv_redundant_removed_calls"] = [
                                            str(c)[:200] for c in _removed]
                                except NameError:
                                    pass
                            if record_triggers is not None:
                                record_triggers.add(_sup_spec.key)
                            _txt = ((templates.get("templates") or {}).get(
                                _sup_spec.key, "") or "").strip() or gate_fallback(_sup_spec.key)
                            if _sup_spec.key == "on_destructive_authorization":
                                try:
                                    if _r1_obs:
                                        step_record["recovery_txn_observations"] = _r1_obs[:6]
                                        _txt = " ".join(_r1_obs[:3])
                                except NameError:
                                    pass
                            if _txt:
                                _sup_fired_text = _txt
                    if _sup_fired_text is not None:
                        _ie = is_empty_execute_response if is_empty_execute_response else (lambda x: not x)
                        if _ie(decoded) and step_count < self.max_steps_per_turn:
                            inference_data = self.handler._add_next_turn_user_message_prompting(
                                inference_data,
                                [{"role": "user", "content": _sup_fired_text}],
                            )
                            # Record the SUPPRESSED step before continuing. Without
                            # this the `continue` discards the only step_record
                            # carrying the gate's telemetry flag, so a gate that
                            # demonstrably fires leaves NO trace: the trajectory
                            # just shows the call absent. Verified on the E2
                            # suppress arm, where 4 clears were eliminated across 2
                            # episodes with zero g1_gate flags recorded. The
                            # iterative loop needs this to confirm an installed
                            # gate is live (§8.3), and _eval_batch's gates_fired
                            # telemetry depends on it too.
                            step_record["status"] = "gate_suppressed"
                            trajectory_history.append(step_record)
                            step_count += 1
                            continue

                    _is_empty = (
                        is_empty_execute_response if is_empty_execute_response
                        else (lambda x: not x)
                    )
                    if _is_empty(decoded):
                        # D3 gate: force archival search before accepting IDK (mirrors base_handler.py:652-687)
                        _model_text = model_response if isinstance(model_response, str) else str(model_response)
                        # Computed once, independent of disable_gates/enable_forced_retrieval/
                        # once-per-turn state, so it's a labeled signal even when the gate
                        # itself is off or has already fired once this turn.
                        _gave_up_no_archival = (
                            _applies_premature_idk
                            and not _d3_archival_searched
                            and gate_match_any(REGISTRY_BY_KEY["on_premature_idk"], [_model_text.lower()])
                        )
                        if _gave_up_no_archival:
                            step_record["gave_up_without_archival_retrieval"] = True
                        if (
                            not self.disable_gates
                            and gate_enabled(templates, "on_premature_idk")
                            and _gave_up_no_archival
                            and not _d3_gate_fired
                            and step_count < self.max_steps_per_turn
                        ):
                            _d3_gate_fired = True
                            step_record["d3_gate"] = True
                            if record_triggers is not None:
                                record_triggers.add("on_premature_idk")  # frozen; defensive

                            # Phase 6 (opt-in via enable_forced_retrieval): instead of only
                            # asking the model to search (which it already ignores once, in
                            # the standing system prompt), execute the search for it.
                            _forced_result = None
                            if remedy_enabled(templates, "enable_forced_retrieval"):
                                _forced_result = self._try_forced_archival_retrieval(
                                    test_category, initial_config, involved_classes,
                                    rollout_model_name, test_entry_id,
                                )

                            if _forced_result is not None:
                                step_record["forced_retrieval_gate"] = True
                                step_record["tool_results"] = step_record.get("tool_results", []) + [_forced_result]
                                # Match G1/G3 here and base_handler's anchoropt_template():
                                # a present-but-blank policy template falls through to the
                                # registry fallback rather than injecting whitespace.
                                _d3_text = ((templates.get("templates") or {}).get(
                                    "on_forced_archival_retrieval", "") or "").strip() or gate_fallback("on_forced_archival_retrieval")
                            else:
                                _d3_text = ((templates.get("templates") or {}).get(
                                    "on_premature_idk", "") or "").strip() or gate_fallback("on_premature_idk")
                            inference_data = self.handler._add_next_turn_user_message_prompting(
                                inference_data,
                                [{"role": "user", "content": _d3_text}],
                            )
                            step_count += 1
                            continue
                        # ── A4v2: POST-GENERATION / PRE-EXECUTION commitment gate ─────────
                        # Incision point: the model has GENERATED its answer but nothing has been
                        # sent to the tool processor -- it chose to call nothing. So this is NOT
                        # post-execution: no tool ran, and there is no result to recover from.
                        # It is the same window A3 acts in (a generated decision not yet
                        # committed), which is why both are commitment gates rather than error
                        # recovery: A3 cancels a call before dispatch, this declines to commit an
                        # answer before it ends the turn.
                        #
                        # intercept answer_end_turn on a ZERO-TOOL-CALL query
                        # A4v1 fired at step_count == 0, i.e. the start of EVERY query, because
                        # "about to answer without calling a tool" is not observable before the
                        # model answers. It injected in 303/303 episodes instead of the ~48
                        # targeted and cost -4.95pp (rec_sum -24), even though the mechanism
                        # worked: no-call episodes went 60 -> 0. That was a specification error,
                        # not a failed hypothesis.
                        #
                        # Here the condition is observable: the model has produced its answer and
                        # made NO tool call this query. So intercept POST-GENERATION, reprompt
                        # once, and let it answer again. Bounded to one intervention per query by
                        # `_zero_call_reprompted` -- an unbounded version would loop on a model
                        # that keeps declining to call anything.
                        #
                        # Deliberately does NOT prescribe retrieval: 23 of the 48 measured no-call
                        # failures had nothing in the store, where pushing a read invites
                        # fabrication.
                        if (not self.disable_gates
                                and remedy_enabled(templates, "enable_zero_call_reprompt")
                                and not _zero_call_reprompted
                                and _query_tool_calls == 0
                                and step_count < self.max_steps_per_turn):
                            _zero_call_reprompted = True
                            step_record["zero_call_reprompt_gate"] = True
                            _zc_text = ((templates.get("templates") or {}).get(
                                "on_turn_start_action", "") or "").strip()
                            if _zc_text:
                                inference_data = self.handler._add_next_turn_user_message_prompting(
                                    inference_data, [{"role": "user", "content": _zc_text}],
                                )
                                trajectory_history.append(step_record)
                                step_count += 1
                                continue
                        # A5 CANDIDATE -- post_generation_pre_commit on LOW RETRIEVAL SIMILARITY.
                        #
                        # Signal feature `s` = top similarity score of a read's result. Frozen
                        # criteria: docs/N0_A5_CRITERIA_FROZEN.md (theta=0.30, exactly one read).
                        #
                        # Same window as A4v2 and for the same reason: the answer exists but
                        # nothing has been sent to the tool processor, so this is a COMMITMENT
                        # GATE, not error recovery -- the retrieval did not fail, it succeeded
                        # weakly, and the model is about to commit to it.
                        #
                        # Why not post_execution (i.e. the moment the weak result returns):
                        # measured on train, 54% of low-s first reads SELF-RECOVER -- the model
                        # reformulates unprompted -- and 43% of those end correct with no help.
                        # Intervening when the result returns would touch all of them. That is
                        # the A4v1 over-fire failure mode, and it shows up as a precision cap:
                        # ~70% flat across the whole theta range at post_execution vs 95.5% here.
                        #
                        # `reads == 1` is part of the signal, not an optimisation: it is what
                        # distinguishes "committed on one weak read" from "tried again already".
                        #
                        # Bounded to one injection per query by `_low_sim_reprompted`.
                        _a5_sims = [s for s in _query_read_sims if s is not None]
                        if (not self.disable_gates
                                and remedy_enabled(templates, "enable_low_similarity_reprompt")
                                and not _low_sim_reprompted
                                and len(_query_read_sims) == 1
                                and len(_a5_sims) == 1
                                and _a5_sims[0] < _low_sim_theta(templates)
                                and step_count < self.max_steps_per_turn):
                            _low_sim_reprompted = True
                            step_record["low_similarity_reprompt_gate"] = True
                            step_record["low_similarity_value"] = _a5_sims[0]
                            _ls_text = ((templates.get("templates") or {}).get(
                                "on_low_similarity_reprompt", "") or "").strip()
                            if not _ls_text:
                                _ls_text = (gate_fallback(
                                    "on_low_similarity_reprompt") or "").strip()
                            if _ls_text:
                                inference_data = self.handler._add_next_turn_user_message_prompting(
                                    inference_data, [{"role": "user", "content": _ls_text}],
                                )
                                trajectory_history.append(step_record)
                                step_count += 1
                                continue
                        # B1 / F2: the episode is about to END on an unresolved error. Only knowable
                        # once the terminal answer exists, so this is the same commitment window
                        # A4v2 uses -- detecting abandonment after the turn closes is too late to act.
                        # Measured at 30 true abandonments (NOT 91; that figure came from substring
                        # matching "error" against retrieved content).
                        if (not self.disable_gates
                                and remedy_enabled(templates, "enable_futility_escalate")
                                and _b1_firings < B1_MAX_FIRINGS
                                and step_count < self.max_steps_per_turn
                                and _b1_last_step_errored(trajectory_history)):
                            _b1_txt2 = ((templates.get("templates") or {}).get(
                                "on_repeated_failure_escalate", "") or "").strip()
                            if not _b1_txt2:
                                _b1_txt2 = (gate_fallback(
                                    "on_repeated_failure_escalate") or "").strip()
                            if _b1_txt2:
                                _b1_firings += 1
                                step_record["futility_escalate_gate"] = True
                                step_record["futility_kind"] = "F2_abandonment"
                                inference_data = (
                                    self.handler._add_next_turn_user_message_prompting(
                                        inference_data, [{"role": "user", "content": _b1_txt2}]))
                                trajectory_history.append(step_record)
                                step_count += 1
                                continue
                        step_record["status"] = "answer_end_turn"
                        trajectory_history.append(step_record)
                        break

                except Exception as e:
                    failure_streak += 1
                    consecutive_decode_failures += 1
                    step_record["status"] = "decode_failure"
                    step_record["error"] = str(e)
                    trajectory_history.append(step_record)

                    if consecutive_decode_failures >= 3:
                        force_quit = True
                        break
                    step_count += 1
                    if step_count > self.max_steps_per_turn:
                        force_quit = True
                        break
                    continue

                # ── Execute tools ─────────────────────────────────────────────
                execution_results, involved_instances = self._execute(
                    decoded,
                    initial_config,
                    involved_classes,
                    rollout_model_name,
                    test_entry_id,
                )

                step_record["status"] = "executed"
                # A4v2: count the tool calls this query actually made. Counted at EXECUTION, not
                # at decode, so a call the model proposed but that was suppressed pre-execution
                # (A3) does not count as "consulted memory" -- it never ran.
                _query_tool_calls += len(decoded or [])
                # A5: record one entry per executed READ -- the max similarity_score in its
                # result, or None when the backend returned no score. Recorded here, at
                # execution, for the same reason as _query_tool_calls: a read that was proposed
                # but suppressed pre-execution never happened, so it must not count as evidence
                # the model consulted memory.
                for _i, _c in enumerate(decoded or []):
                    if not _is_read_call(_c):
                        continue
                    _r = execution_results[_i] if _i < len(execution_results or []) else ""
                    _query_read_sims.append(_top_similarity(_r))
                step_record["tool_results"] = execution_results
                # ---- N0-R1: OPEN / CLOSE the recovery transaction ----------------------
                # The predicate above CONSUMES self._r1_txn; this is what populates it. Runs
                # post-execution, where the capacity failure and the retry outcome are both visible.
                #
                # OPEN  on a capacity-failed write, keyed by container.
                # CLOSE only on a successful retry of the ORIGINAL value -- NOT on any later
                #       successful write. Of 67 apparent "retry successes" offline only 40 were the
                #       same value re-attempted; the other 27 were the model moving on to a different
                #       write, which does not resolve the unresolved failure.
                if not self.disable_gates and gate_enabled(templates, "on_destructive_authorization"):
                    try:
                        import sys as _s4
                        _d4 = str(Path(__file__).resolve().parent.parent / "scripts")
                        if _d4 not in _s4.path:
                            _s4.path.insert(0, _d4)
                        import recovery_txn as _rt4
                        if not hasattr(self, "_r1_txn"):
                            self._r1_txn = {}
                        for _ri, _rc in enumerate(decoded or []):
                            _rcs = str(_rc)
                            if not _rt4.ADD_RE.search(_rcs):
                                continue
                            _rr = str(execution_results[_ri]) if _ri < len(execution_results or []) else ""
                            _rwh = _rt4.container_of(_rcs)
                            _rv = _rt4.VALARG.search(_rcs)
                            _rvc = _rt4.canon(_rv.group(2)) if _rv else ""
                            if "error" in _rr.lower() and _rt4.FULL_RE.search(_rr):
                                if _rwh not in self._r1_txn:
                                    _live = _rt4.live_store(involved_instances, _rwh)
                                    self._r1_txn[_rwh] = _rt4.RecoveryTransaction(
                                        _rwh, len(_live), _rvc)
                                    step_record["recovery_txn_opened"] = _rwh
                            elif "error" not in _rr.lower():
                                _t = self._r1_txn.get(_rwh)
                                if _t is not None and _rvc and _rvc == _t.pending_value:
                                    # VERIFY + CLOSE: the original write landed.
                                    step_record["recovery_txn_closed"] = "retry_succeeded"
                                    step_record["recovery_txn_authorized_total"] = _t.authorized
                                    step_record["recovery_txn_proposals_total"] = _t.proposals
                                    del self._r1_txn[_rwh]
                    except Exception as _e:
                        step_record["recovery_txn_error"] = str(_e)[:120]

                # KV OBSERVER: classify repeated-key rejections against the LIVE store. No action.
                for _ki, _kc in enumerate(decoded or []):
                    _kcs = str(_kc)
                    _km = _KVO_ADD_RE.search(_kcs)
                    if not _km:
                        continue
                    _kr = str(execution_results[_ki]) if _ki < len(execution_results or []) else ""
                    if not _KVO_UNIQ_RE.search(_kr):
                        continue
                    _which = _km.group(1)
                    _kk = _KVO_KEY_RE.search(_kcs)
                    _kv = _KVO_VAL_RE.search(_kcs)
                    if not _kk:
                        continue
                    _live_kv = _kvo_live_store(involved_instances, _which)
                    _old = _live_kv.get(_kk.group(2))
                    _rec = {"store": _which, "key": _kk.group(2),
                            "live_entries": len(_live_kv),
                            "attempted_len": len(_kv.group(2)) if _kv else 0}
                    if _old is None:
                        # Key rejected as taken, yet absent from the live store we can read. Record
                        # rather than guess -- an unexplained rejection is itself a finding.
                        _rec["kv_obs_class"] = "UNRESOLVED_live"
                        _rec["kv_obs_why"] = "key reported taken but not present in live store"
                    else:
                        _c, _w = _kvo_classify(_old, _kv.group(2) if _kv else "",
                                               _KVO_CAPS[_which])
                        _rec["kv_obs_class"] = _c
                        _rec["kv_obs_why"] = _w
                        _rec["stored_len"] = len(_old)
                    step_record.setdefault("kv_obs_events", []).append(_rec)
                # E1: track archival writes, then run the eviction state machine on a slot rejection.
                for _ei, _ec in enumerate(decoded or []):
                    _ecs = str(_ec)
                    if not _E1_ADD_RE.search(_ecs):
                        continue
                    _er = str(execution_results[_ei]) if _ei < len(execution_results or []) else ""
                    _ev = _E1_VAL_RE.search(_ecs)
                    _eval_norm = _e1_norm(_ev.group(2)) if _ev else ""
                    if "error" not in _er.lower():
                        continue
                    if not _E1_SLOT_RE.search(_er):
                        continue
                    # ARCHIVAL IS FULL. Frozen chain: evict ONE duplicate, then retry verbatim.
                    if (self.disable_gates
                            or not remedy_enabled(templates, "enable_archival_evict_duplicate")
                            or _e1_evictions >= E1_MAX_EVICTIONS):
                        continue
                    # LIVE STATE at the decision point -- not our episode-local log. v1 read the
                    # log and saw a median of 4 entries against a real store of ~50, which is why it
                    # never found a victim and the arm was void.
                    _live = _e1_live_archival(involved_instances)
                    step_record["e1_live_entries"] = len(_live)
                    _victim = _e1_pick_victim(_live)
                    if _victim is None:
                        step_record["e1_no_victim"] = True
                        step_record["e1_live_distinct"] = len({v for _i, v in _live})
                        continue
                    _vid, _vval, _vk = _victim
                    # INVARIANT, asserted at the point of action: a copy MUST remain. This is the
                    # property that makes E1 defensible, so it is enforced rather than audited.
                    if _vk < 2:
                        step_record["e1_invariant_refused"] = True
                        continue
                    _rm_call = "archival_memory_remove(vec_id=%d)" % _vid
                    _rm_res, _ = self._execute(
                        [_rm_call], initial_config, involved_classes,
                        rollout_model_name, test_entry_id)
                    _rm_blob = " ".join(str(x) for x in (_rm_res or []))
                    step_record["e1_evict_gate"] = True
                    step_record["e1_evicted_id"] = _vid
                    step_record["e1_evicted_copies_before"] = _vk
                    step_record["e1_evict_result"] = _rm_blob[:160]
                    if "error" in _rm_blob.lower():
                        step_record["e1_evict_failed"] = True
                        continue
                    _e1_evictions += 1
                    # VERIFY against live state that an identical copy REMAINS. This is the frozen
                    # invariant, and now it is checked against the store rather than inferred.
                    _after = _e1_live_archival(involved_instances)
                    _remaining = sum(1 for _i, _v in _after if _v == _vval)
                    step_record["e1_copies_remaining"] = _remaining
                    if _remaining < 1:
                        step_record["e1_invariant_violated"] = True
                    # retry the ORIGINAL call, verbatim. No rewriting: this arm tests eviction alone.
                    _rt_res, _ = self._execute(
                        [_ecs], initial_config, involved_classes,
                        rollout_model_name, test_entry_id)
                    _rt_blob = " ".join(str(x) for x in (_rt_res or []))
                    step_record["e1_retry_result"] = _rt_blob[:160]
                    step_record["e1_retry_landed"] = ("error" not in _rt_blob.lower())
                    execution_results = list(execution_results) + list(_rm_res or []) + list(_rt_res or [])
                # B1 / F1: count consecutive failures of the SAME call, and escalate at the Nth.
                # Counted here, at execution, because the error is what makes futility observable.
                if (not self.disable_gates
                        and remedy_enabled(templates, "enable_futility_escalate")
                        and _b1_firings < B1_MAX_FIRINGS):
                    _b1_thr = _b1_threshold(templates)
                    for _bi, _bc in enumerate(decoded or []):
                        _br = execution_results[_bi] if _bi < len(execution_results or []) else ""
                        _bkey = str(_bc)
                        if _b1_is_error(_br):
                            _b1_fail_counts[_bkey] = _b1_fail_counts.get(_bkey, 0) + 1
                            if _b1_fail_counts[_bkey] == _b1_thr:
                                _b1_txt = ((templates.get("templates") or {}).get(
                                    "on_repeated_failure_escalate", "") or "").strip()
                                if not _b1_txt:
                                    _b1_txt = (gate_fallback(
                                        "on_repeated_failure_escalate") or "").strip()
                                if _b1_txt:
                                    _b1_firings += 1
                                    step_record["futility_escalate_gate"] = True
                                    step_record["futility_kind"] = "F1_repeated_failure"
                                    step_record["futility_fail_count"] = _b1_fail_counts[_bkey]
                                    inference_data = (
                                        self.handler._add_next_turn_user_message_prompting(
                                            inference_data,
                                            [{"role": "user", "content": _b1_txt}]))
                        else:
                            # a SUCCESS resets the counter: the model recovered, so it is not futile
                            _b1_fail_counts.pop(_bkey, None)

                # ── Capacity-failure state capture (generic; opt-in) ──────────
                # Feasibility of a capacity repair cannot be judged from the logs alone: the
                # step schema records calls and results but never the live store, so the
                # admissible budget is unknowable offline. An earlier offline check tried to
                # RECONSTRUCT the rec_sum blob by summing appends, produced a median length of
                # 0, and reported "fits, loses nothing" -- a verdict about an empty string.
                #
                # This records only what the BACKEND ADAPTER declares in `state_needed`, so the
                # core stays free of per-backend policy: locus + live state in, repair spec out.
                # Off unless ANCHOROPT_CAPTURE_CONSTRAINT_STATE is set, so the default
                # trajectory schema and every fingerprint are unchanged.
                if os.environ.get("ANCHOROPT_CAPTURE_CONSTRAINT_STATE"):
                    try:
                        # Backend comes from the CASE ID, not the case dict: the corpus
                        # entries carry only id/question/involved_classes, so a
                        # `case["backend"]` lookup returns nothing. The first run of this
                        # instrumentation referenced an undefined `test_entry` and produced
                        # 0 captures across 337 real capacity errors.
                        _be = ""
                        for _cand in ("rec_sum", "vector", "kv"):
                            if ("memory_%s" % _cand) in test_entry_id:
                                _be = _cand
                                break
                        _cap = _capture_constraint_state(
                            decoded, execution_results, involved_instances, _be)
                        if _cap:
                            step_record["constraint_state"] = _cap
                    except Exception as _e:  # never let instrumentation kill a rollout
                        step_record["constraint_state_error"] = \
                            "%s: %s" % (type(_e).__name__, _e)

                trajectory_history.append(step_record)

                inference_data = self.handler._add_execution_results_prompting(
                    inference_data, execution_results, model_response_data
                )

                # ── Update AnchorOpt state after execution ────────────────────
                from .state_extractor import snapshot as _snapshot
                prev_snapshot = curr_snapshot
                curr_snapshot = _snapshot(involved_instances)
                last_result = execution_results[-1] if execution_results else ""

                step_errored = any(_is_error(r) for r in execution_results)
                failure_streak = failure_streak + 1 if step_errored else 0

                curr_error_signal = classify_last_result(last_result)
                if curr_error_signal is not None and curr_error_signal == prev_error_signal:
                    error_streak += 1
                elif curr_error_signal is not None:
                    error_streak = 1
                else:
                    error_streak = 0
                prev_error_signal = curr_error_signal

                last_call_str = decoded[0] if decoded else ""
                for call in decoded:
                    name = call.split("(")[0].split(".")[-1].strip()
                    recent_tools.append(name)
                    if not step_errored:
                        recent_call_signatures.append(call.strip())

                if is_empty_or_failed_retrieval(last_result, last_call_str):
                    failed_search_streak += 1
                elif is_retrieval_success(last_result):
                    failed_search_streak = 0
                    step_record["is_retrieval_success"] = True

                # ── G3 gate: core/archival-full redirect (mirrors base_handler.py:727-763) ──
                # Deterministic, once per turn, KV/Vector only. Fires on the official
                # "is full" / "exceeds maximum size" tool-error substrings. Text is taken
                # from the tunable on_domain_error_core_full template with the official
                # string as fallback, so the null arm still gets G3 exactly like official.
                # Phase 3 (opt-in via the enable_reroute policy flag, mirrors base_handler.py's
                # _try_core_full_reroute): re-dispatch a synthesized archival_memory_add
                # call instead of only reprompting; falls back to the reprompt text below
                # whenever the reroute can't be synthesized/dispatched.
                _tmpl_dict = (templates.get("templates") or {})
                if (
                    not self.disable_gates
                    and _applies_core_full
                    and gate_enabled(templates, "on_domain_error_core_full")
                    and not _g3_fired
                    and step_count < self.max_steps_per_turn
                    and gate_match_any(REGISTRY_BY_KEY["on_domain_error_core_full"], execution_results)
                ):
                    if record_triggers is not None:
                        record_triggers.add("on_domain_error_core_full")  # frozen; defensive
                    _g3_fired = True
                    step_record["a1_core_full_gate"] = True   # renamed from g3_gate (A1)

                    _reroute_result = None
                    # _try_core_full_reroute does not receive `templates`, so publish it for
                    # the operator seam inside that method. Without this the seam read an
                    # always-empty dict and the operator could never fire -- present in the
                    # code, inert in execution, which is the §24 defect class where the
                    # record and the behaviour disagree.
                    self._active_templates = templates
                    if remedy_enabled(templates, "enable_reroute"):
                        _reroute_result = self._try_core_full_reroute(
                            decoded,
                            execution_results,
                            involved_instances,
                            test_category,
                            initial_config,
                            involved_classes,
                            rollout_model_name,
                            test_entry_id,
                        )

                    if _reroute_result is not None:
                        # _try_reroute now returns a dict; accept the legacy string form
                        # so a stale checkout cannot silently drop the payload.
                        if isinstance(_reroute_result, dict):
                            _rr_res = _reroute_result.get("result", "")
                            step_record["reroute_replaced_call"] = _reroute_result.get("replaced_call")
                            step_record["reroute_substituted_call"] = _reroute_result.get("substituted_call")
                        else:
                            _rr_res = _reroute_result
                        step_record["reroute_gate"] = True
                        step_record["tool_results"] = step_record.get("tool_results", []) + [_rr_res]
                        _g3_text = (_tmpl_dict.get("on_core_full_rerouted") or "").strip() or gate_fallback("on_core_full_rerouted")
                    else:
                        _g3_text = (_tmpl_dict.get("on_domain_error_core_full") or "").strip() or gate_fallback("on_domain_error_core_full")
                    inference_data = self.handler._add_next_turn_user_message_prompting(
                        inference_data, [{"role": "user", "content": _g3_text}],
                    )
                    step_count += 1
                    continue

                # ── CAPACITY REPAIR, aggregate branch (N0-A1) ─────────────────────
                # The SAME mined locus as G3 above -- capacity/container/
                # no_remaining_capacity -- on a backend whose container is an aggregate
                # rather than a collection. G3 relocates an item to a second container;
                # here there is no second container, so the repair REPLACES the aggregate
                # with a version that admits the new item.
                #
                # Deliberately NOT the on_blob_pressure preempt: that nudges the model in
                # prose before saturation and is a prompt-style intervention, which this
                # line excludes. This fires on the post-execution capacity error, exactly
                # where G3 fires, and dispatches a mechanically validated call.
                #
                # Feasibility measured on 352 live captures before this was wired: budget
                # computable at every failure, 352/352 replacements constructed (median 1%
                # compression), 99.9% of query-relevant tokens preserved. The backend
                # supplies condition/operation/constraint/validator through the adapter;
                # this block contains no backend semantics of its own.
                if (
                    not self.disable_gates
                    and not _capacity_repair_fired
                    and remedy_enabled(templates, "enable_capacity_repair")
                    and step_count < self.max_steps_per_turn
                ):
                    _cr_res = self._try_capacity_repair(
                        decoded, execution_results, involved_instances, test_category,
                        initial_config, involved_classes, rollout_model_name, test_entry_id,
                    )
                    if _cr_res is not None:
                        _capacity_repair_fired = True
                        step_record["capacity_repair_gate"] = True
                        step_record["capacity_repair_replaced_call"] = _cr_res.get("replaced_call")
                        step_record["capacity_repair_substituted_call"] = _cr_res.get("substituted_call")
                        step_record["capacity_repair_shed_chars"] = _cr_res.get("shed_chars")
                        step_record["tool_results"] = (
                            step_record.get("tool_results", []) + [_cr_res.get("result", "")])
                        execution_results = list(execution_results) + [_cr_res.get("result", "")]
                        step_count += 1
                        continue

                # ── G4 gate: KV key-not-found key-search fallback (mirrors base_handler.py:765-796) ──
                if (
                    not self.disable_gates
                    and gate_applies(REGISTRY_BY_KEY["on_domain_error_key_not_found"], test_category)
                    and gate_enabled(templates, "on_domain_error_key_not_found")
                    and not _g4_fired
                    and step_count < self.max_steps_per_turn
                    and gate_match_any(REGISTRY_BY_KEY["on_domain_error_key_not_found"], execution_results)
                ):
                    if record_triggers is not None:
                        record_triggers.add("on_domain_error_key_not_found")  # frozen; defensive
                    _g4_fired = True
                    step_record["a2_key_not_found_gate"] = True   # renamed from g4_gate (A2)

                    # Phase 7 (opt-in via the enable_forced_key_search policy flag, mirrors
                    # base_handler.py's _try_forced_key_search): list the stored keys on the
                    # model's behalf instead of only telling it to; falls back to the G4
                    # reprompt text below whenever the dispatch can't be made.
                    _key_search_result = None
                    if remedy_enabled(templates, "enable_forced_key_search"):
                        _key_search_result = self._try_forced_key_search(
                            test_category, initial_config, involved_classes,
                            rollout_model_name, test_entry_id,
                        )

                    if _key_search_result is not None:
                        step_record["forced_key_search_gate"] = True
                        step_record["tool_results"] = step_record.get("tool_results", []) + [_key_search_result]
                        _g4_text = (_tmpl_dict.get("on_forced_key_search") or "").strip() or gate_fallback("on_forced_key_search")
                    else:
                        _g4_text = (_tmpl_dict.get("on_domain_error_key_not_found") or "").strip() or gate_fallback("on_domain_error_key_not_found")
                    inference_data = self.handler._add_next_turn_user_message_prompting(
                        inference_data, [{"role": "user", "content": _g4_text}],
                    )
                    step_count += 1
                    continue

                # ── Generic TRANSFORM family (§22) ────────────────────────────────
                # Generative remedy: rewrite the rejected ARGUMENT so the same content
                # satisfies the constraint, then re-execute the call. Ordered BEFORE
                # G5's reprompt deliberately: transform acts, whereas reprompt only
                # asks the model to try again. A failed or unsafe rewrite falls
                # through to G5 below, so the reprompt remains the safety net.
                #
                # Generic over the registry: match strings come from the spec and the
                # rewrite objective from transform_instruction(spec.key, ...), so a
                # transform gate for a different constraint needs no code here.
                for _tx_spec in transform_specs(templates, test_category):
                    if self.disable_gates or step_count >= self.max_steps_per_turn:
                        break
                    if not gate_match_any(_tx_spec, execution_results):
                        continue
                    # §25.2 layer 1: once-per-turn latch, honouring the spec's own
                    # `once_per_turn` rather than hardcoding it, so the registry stays
                    # the single source of truth. Without this the block re-fired
                    # unbounded and one episode spent 20 of its 21 steps on a rewrite
                    # fixpoint -- a correctness bug (the episode never got to do
                    # anything else), not a latency one.
                    if getattr(_tx_spec, "once_per_turn", False) and \
                            _tx_spec.key in _tx_fired:
                        step_record["transform_skipped_reason"] = "once_per_turn_latch"
                        continue
                    if record_triggers is not None:
                        record_triggers.add(_tx_spec.key)
                    _tx = self._try_transform(
                        _tx_spec, decoded, execution_results,
                        inference_data, templates,
                    )
                    if not _tx:
                        continue
                    _tx_fired.add(_tx_spec.key)
                    step_record[_tx_spec.telemetry_flag] = True

                    # §25.4 layer 2: outcome-aware escalation, exact-fixpoint rule.
                    # State key is (signal, gate, input value, output value), all four
                    # taken from trajectory STRUCTURE -- the matched error contract, the
                    # gate key, the original argument, and the produced rewrite. No
                    # reward label, so this cannot leak outcomes into the trigger.
                    #
                    # If this exact rewrite was already produced and refused here, the
                    # branch is provably exhausted: re-executing an identical output
                    # cannot yield a different result. Stop rather than spend another
                    # model call, and RECORD it, so a state whose every action is
                    # locally refuted becomes evidence for the next iteration's miner
                    # instead of being silently re-attempted each run.
                    # A transient query failure returns no `rewritten`, and must NOT be
                    # recorded as a fixpoint: it is evidence about the API, not about
                    # whether this branch is exhausted.
                    _tx_out = _tx.get("rewritten")
                    _tx_key = None
                    if _tx_out is not None:
                        _tx_key = (
                            str(_tx.get("error_text") or "")[:200],   # signal
                            _tx_spec.key,                            # gate
                            str(_tx.get("original") or "")[:400],    # input value
                            str(_tx_out)[:400],                      # output value
                        )
                    if _tx_key is not None and _tx_key in _tx_refuted:
                        step_record["transform_validated"] = False
                        step_record["transform_ok"] = False
                        step_record["transform_reason"] = "locally_refuted_fixpoint"
                        step_record["transform_locally_refuted"] = True
                        continue
                    if not _tx.get("ok"):
                        # A rejected rewrite is refuted FOR THIS (signal, gate, input,
                        # output). Recording it here is what stops the next turn from
                        # re-deriving the same failure.
                        if _tx_key is not None:
                            _tx_refuted.add(_tx_key)
                    # NAMING (§27.4). `transform_ok` records the verdict of
                    # transform_ok(), which is a PRE-EXECUTION validation of the
                    # candidate rewrite (fits the limit, kept the numeric facts). It
                    # does NOT mean the rewritten call then executed successfully --
                    # that outcome is `transform_executed_ok` below.
                    #
                    # The distinction is easy to lose because a step whose
                    # `transform_ok` is True still shows the ORIGINAL call's error in
                    # `tool_results` (the trigger), which reads as "validated yet
                    # still failing". On the first G2 corpus the two happened to agree
                    # 111/111, so conflating them would have been invisible here and
                    # would have silently overstated local efficacy on the next
                    # corpus. `transform_validated` is the honest name; `transform_ok`
                    # is kept as an alias so already-written trajectories and the
                    # smoke checker stay readable.
                    step_record["transform_validated"] = bool(_tx.get("ok"))
                    step_record["transform_ok"] = bool(_tx.get("ok"))  # deprecated alias
                    step_record["transform_reason"] = _tx.get("reason")
                    if not _tx.get("ok"):
                        # Writing a corrupted value is worse than not acting: record
                        # the attempt for telemetry and let the model's own retry (or
                        # G5) proceed on the ORIGINAL failure.
                        continue
                    step_record["transform_original_len"] = _tx.get("original_len")
                    step_record["transform_new_len"] = _tx.get("new_len")
                    # Re-execute the rewritten call through the SAME path as any other
                    # call, so state, telemetry and the snapshot stay consistent.
                    # Re-execute through self._execute -- the SAME wrapper the normal
                    # path uses (it holds the snapshot-warning suppression and the
                    # singleton wiring), so a transformed call cannot diverge from an
                    # ordinary one in state handling.
                    _tx_results, involved_instances = self._execute(
                        [_tx["call"]],
                        initial_config,
                        involved_classes,
                        rollout_model_name,
                        test_entry_id,
                    )
                    step_record["transform_retry_results"] = [
                        str(r)[:200] for r in (_tx_results or [])
                    ]
                    # The remedy's ACTUAL outcome, recorded explicitly rather than
                    # left to be re-derived downstream. Structural test only: did the
                    # re-executed call still report the error contract that triggered
                    # the gate? No reward label is consulted, so this stays usable as
                    # attribution evidence (§26.3) and not just as a score.
                    _tx_blob = " ".join(str(r) for r in (_tx_results or []))
                    # DEFECT FIXED (W1 arm, job 911257). This flag was
                    #     bool(_tx_results) and not any(m in _tx_blob for m in match_substrings)
                    # i.e. "the retry did not reproduce the GATE'S OWN trigger substring". For
                    # on_entry_too_long_transform that substring is 'exceeds maximum length', so a
                    # retry rejected for a DIFFERENT reason was recorded as a success:
                    #     {"error": "Memory size exceeds maximum size of 7 entries."} -> True
                    #     "Error during execution: invalid syntax..."                 -> True
                    # On W1 that produced 32/32 transform_executed_ok=True while only 4 writes
                    # actually landed. The flag survived the arm's own gating audit and made a null
                    # result look like clean execution, which cost a full round of misattribution
                    # (S4.9.29 -> S4.9.30).
                    #
                    # Three outcomes are now distinguished, because they imply different actions:
                    #   trigger_recurred   the rewrite did not satisfy the constraint -> remedy failed
                    #   downstream_error   the constraint IS satisfied and the write hit the NEXT one
                    #                      -> the sequential-repair case; the remedy worked
                    #   malformed          the synthesized call did not parse -> OUR bug, write LOST
                    _tx_recurred = bool(_tx_results) and any(
                        m in _tx_blob for m in (_tx_spec.match_substrings or ())
                    )
                    _tx_is_err = ("error" in _tx_blob.lower()) or ("Traceback" in _tx_blob)
                    _tx_malformed = ("invalid syntax" in _tx_blob) or (
                        "unexpected" in _tx_blob and "line 1" in _tx_blob)
                    if not bool(_tx_results):
                        _tx_outcome = "no_result"
                    elif _tx_malformed:
                        _tx_outcome = "malformed"
                    elif _tx_recurred:
                        _tx_outcome = "trigger_recurred"
                    elif _tx_is_err:
                        _tx_outcome = "downstream_error"
                    else:
                        _tx_outcome = "landed"
                    step_record["transform_retry_outcome"] = _tx_outcome
                    step_record["transform_retry_recurred_trigger"] = _tx_recurred
                    step_record["transform_write_landed"] = (_tx_outcome == "landed")
                    # deprecated alias, now meaning what its name says
                    step_record["transform_executed_ok"] = (_tx_outcome == "landed")
                    if _tx_malformed:
                        # A malformed synthesis LOSES the content: the original write was rejected
                        # and the rewrite never parsed, so nothing is stored and the model does not
                        # get its own retry. Loud, because it is our defect and not the model's.
                        step_record["transform_synthesis_malformed"] = True
                        print("[anchoropt] TRANSFORM SYNTHESIS MALFORMED -- write LOST: %s"
                              % _tx_blob[:160], flush=True)
                    execution_results = list(execution_results) + list(_tx_results or [])
                    break

                # ── G5 gate: rec_sum blob-overflow reprompt (mirrors base_handler.py G5) ──
                # Fires once per turn when a rec_sum memory_append/memory_update call just
                # failed because the resulting blob would exceed MAX_MEMORY_ENTRY_LENGTH.
                # Registry fallback is "" by design (opt-in) — seed_text() supplies the
                # on-enable default so unmasking this gate with no policy text is no
                # longer a guaranteed no-op (Phase 4 seed library).
                # ── GENERIC post-execution RECOVERY SUBSTITUTION (§40/G3) ─────────
                # Dispatch a MINED substitute call as the next recovery action. Ordered
                # BEFORE the reprompt family: substituting acts, whereas reprompting only
                # asks the model to try again, and a failed synthesis falls through to the
                # advisory path so the fact is never silently dropped.
                #
                # Reuses the SAME synthesizer the G1 incumbent uses. What differs per
                # signal is only WHICH substitute, and that came from the corpus
                # (node_recovery_action's attestation), not from this registry.
                # FLAG-GUARDED, mirroring base_handler.py: the synthesizer is the
                # opt-in reroute remedy, so a learned gate chooses WHEN it fires and never
                # whether the remedy is enabled at all.
                _rr_injected = False
                for _rr_spec in reroute_specs(templates, test_category):
                    if self.disable_gates or step_count >= self.max_steps_per_turn:
                        break
                    if _rr_spec.once_per_turn and _rr_spec.key in _rr_fired:
                        continue
                    if not gate_match_any(_rr_spec, execution_results):
                        continue
                    if record_triggers is not None:
                        record_triggers.add(_rr_spec.key)
                    _rr_res = None
                    # Policy ALONE, no constructor-default third argument -- the
                    # evaluator must resolve remedies exactly as base_handler.py does, or
                    # the two pipelines can diverge on a default nobody set deliberately.
                    if remedy_enabled(templates, "enable_reroute"):
                        _rr_res = self._try_reroute(
                            _rr_spec,
                            decoded, execution_results, involved_instances, test_category,
                            initial_config, involved_classes, rollout_model_name,
                            test_entry_id,
                        )
                    if _rr_res is None:
                        # Could not synthesize a substitute here; fall through to the
                        # advisory family below rather than dropping the write.
                        step_record["reroute_skipped_reason"] = "no_substitute_synthesized"
                        continue
                    _rr_fired.add(_rr_spec.key)
                    step_record[_rr_spec.telemetry_flag] = True
                    if isinstance(_rr_res, dict):
                        step_record["reroute_replaced_call"] = _rr_res.get("replaced_call")
                        step_record["reroute_substituted_call"] = _rr_res.get("substituted_call")
                        _rr_out = _rr_res.get("result", "")
                    else:
                        _rr_out = _rr_res
                    step_record["tool_results"] = (
                        step_record.get("tool_results", []) + [_rr_out])
                    execution_results = list(execution_results) + [_rr_out]
                    step_count += 1
                    _rr_injected = True
                    break
                if _rr_injected:
                    continue

                # GENERIC over the registry (§27.3). This block used to hardcode
                # "on_domain_error_too_long" at five points, which meant a newly
                # learned reprompt gate could be proposed, written into a policy, and
                # then never fire -- the §24 defect class where the record and the
                # behaviour disagree. `reprompt_specs()` now supplies the applicable
                # gates the same way transform_specs/suppress_specs do, so the vector
                # length candidate needs no execution code of its own.
                #
                # Latch is per-spec, not one shared flag: sharing would couple the
                # gates' backend scope and silently ignore a narrowed `backends` on any
                # one of them.
                _rp_injected = False
                for _rp_spec in reprompt_specs(templates, test_category):
                    if self.disable_gates or step_count >= self.max_steps_per_turn:
                        break
                    if _rp_spec.once_per_turn and _rp_spec.key in _rp_fired:
                        continue
                    if not gate_match_any(_rp_spec, execution_results):
                        continue
                    if record_triggers is not None:
                        # Editable opt-in gate: condition fired, so its text (even if
                        # currently empty) can alter storage once filled → record it
                        # BEFORE the emptiness guard so empty→non-empty edits MISS.
                        record_triggers.add(_rp_spec.key)
                    _rp_text = (
                        (_tmpl_dict.get(_rp_spec.key) or "").strip()
                        or seed_text(_rp_spec.key)
                        or gate_fallback(_rp_spec.key)
                    )
                    if _rp_text:
                        _rp_fired.add(_rp_spec.key)
                        step_record[_rp_spec.telemetry_flag] = True
                        inference_data = self.handler._add_next_turn_user_message_prompting(
                            inference_data, [{"role": "user", "content": _rp_text}],
                        )
                        step_count += 1
                        _rp_injected = True
                        break
                if _rp_injected:
                    # Same control flow as the hardcoded block this replaces: an
                    # injected reprompt ends the step and restarts the turn loop.
                    continue

                # ── Blob-pressure gate: proactive rec_sum compaction nudge (mirrors
                #    base_handler.py's blob-pressure block) ──────────────────────────
                # Fires once per turn when the live rec_sum blob is already close to
                # MAX_MEMORY_ENTRY_LENGTH, before any call has failed.
                # Phase 7: the trigger POINT is policy-read via blob_pressure_threshold()
                # (default = BLOB_PRESSURE_THRESHOLD), and the registry now ships real
                # fallback text, so this is no longer an inert opt-in no-op.
                if (
                    not self.disable_gates
                    and gate_applies(REGISTRY_BY_KEY["on_blob_pressure"], test_category)
                    and gate_enabled(templates, "on_blob_pressure")
                    and not _blob_pressure_fired
                    and step_count < self.max_steps_per_turn
                ):
                    _mem_inst = list(involved_instances.values())[0]
                    _blob_len = len(getattr(_mem_inst, "memory", "") or "")
                    if _blob_len >= blob_pressure_threshold(templates):
                        if record_triggers is not None:
                            # Editable opt-in gate: pressure condition fired, so its
                            # text (even if empty) can alter storage once filled →
                            # record BEFORE the emptiness guard.
                            record_triggers.add("on_blob_pressure")
                        _bp_text = (_tmpl_dict.get("on_blob_pressure") or "").strip() or gate_fallback("on_blob_pressure")
                        if _bp_text:
                            _blob_pressure_fired = True
                            step_record["blob_pressure_gate"] = True
                            inference_data = self.handler._add_next_turn_user_message_prompting(
                                inference_data, [{"role": "user", "content": _bp_text}],
                            )
                            step_count += 1
                            continue

                step_count += 1
                if step_count > self.max_steps_per_turn:
                    force_quit = True
                    break

            all_model_responses.append(current_turn_responses)
            if force_quit:
                break

        # ── Step 4: flush to disk for prereq entries ──────────────────────────
        # Mirrors base_handler.py lines 847-852.
        # "prereq" in test_entry_id is the same check as is_memory_prereq().
        if "prereq" in test_entry_id:
            try:
                mem_inst = list(involved_instances.values())[0]
                mem_inst._flush_memory_to_local_file()
            except Exception:
                pass

        # ── Grade with agentic checker ────────────────────────────────────────
        # Prereq entries are not graded — they exist solely to set up memory state.
        if "prereq" in test_entry_id:
            self._reset_episode_state(rollout_model_name, test_entry_id, involved_classes)
            return {"valid": None, "error_type": "prereq_entry"}, trajectory_history

        final_answer = _last_non_fc_message(all_model_responses, self.handler)
        possible_answer = test_case.get("possible_answer", [])

        if agentic_checker is None:
            checker_result = {
                "valid": False,
                "error_message": "agentic_checker not importable — check BFCL installation.",
                "error_type": "anchoropt:import_error",
            }
        elif not possible_answer:
            checker_result = {
                "valid": False,
                "error_message": "No possible_answer in test case.",
                "error_type": "anchoropt:no_answer",
                "final_answer": final_answer,
            }
        else:
            try:
                checker_result = agentic_checker(final_answer, possible_answer)
                checker_result["final_answer"] = final_answer
            except Exception as e:
                checker_result = {
                    "valid": False,
                    "error_message": f"agentic_checker raised: {e}",
                    "error_type": "anchoropt:checker_error",
                    "final_answer": final_answer,
                }

        # W2 fidelity: the official agentic scorer grades the last non-tool-call
        # message regardless of force-quit (eval_runner never hard-fails an
        # agentic entry on step-limit). Previously this block flipped a valid
        # answer to invalid on force_quit, producing false negatives on long-but-
        # correct trajectories that the leaderboard would have passed. We keep
        # force_quit only as a diagnostic annotation and no longer override validity.
        if force_quit:
            checker_result["force_quit"] = True

        # Clean up this episode's singleton (not prereq chain — only this entry).
        self._reset_episode_state(rollout_model_name, test_entry_id, involved_classes)
        return checker_result, trajectory_history
