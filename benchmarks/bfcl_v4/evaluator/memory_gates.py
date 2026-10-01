"""
Thin adapter over the canonical memory signal/gate registry.

The registry itself lives on the BFCL side
(bfcl_eval.model_handler.memory_gates) — see that module's docstring for why:
base_handler.py cannot import anchoropt without a dependency cycle
(anchoropt/__init__.py -> memory_evaluator -> bfcl) and without breaking bare
`bfcl generate` runs where anchoropt isn't on sys.path. So the registry is
stdlib-only and bfcl-side; this module re-exports it for anchoropt code and adds
anchoropt-only derivations (teacher prose, telemetry-flag helpers) that are free
to depend on the rest of anchoropt.

This module never redefines canonical data — it only imports and re-exports it,
plus adds small helpers that build ON TOP of the registry for anchoropt's own
consumers (e.g. run_memory_train.py's teacher).
"""

# Defensive import, matching the convention already used for every other bfcl_eval
# import in memory_evaluator.py: anchoropt (template_engine/injection_engine/etc.)
# must stay importable even in a v4-only or bfcl-not-on-sys.path context. Consumers
# that actually need the registry (memory_evaluator, run_memory_train) already
# require bfcl_eval to be importable for other reasons, so this only protects
# import-time robustness, not runtime correctness.
try:
    # ── Re-export the registry WITHOUT an explicit name list ──────────────────
    # This used to be a hand-maintained list of ~55 names, which is a footgun: every
    # new registry symbol had to be added here as well, and forgetting it fails at
    # IMPORT time in unrelated test modules. That happened twice in one session --
    # first `suppress_specs`, then `transform_specs`, the same mistake repeated after
    # having just fixed it. A list that must be updated in lockstep with another file
    # will eventually drift.
    #
    # Now every public name is re-exported programmatically, so a new action family
    # (or any registry addition) is available here immediately. The explicit
    # `_REQUIRED` check below still fails loudly if the upstream registry drops
    # something this package depends on, which is the one guarantee the manual list
    # actually provided.
    import bfcl_eval.model_handler.memory_gates as _reg

    _REEXPORT_ALL = [n for n in dir(_reg) if not n.startswith("_")]
    globals().update({n: getattr(_reg, n) for n in _REEXPORT_ALL})

    # Names this package calls directly. Kept as a runtime assertion rather than an
    # import list: it documents the dependency and fails with a clear message if
    # upstream removes one, without needing an edit when upstream ADDS one.
    _REQUIRED = (
        "GateSpec",
        "MEMORY_GATE_REGISTRY",
        "REGISTRY_BY_KEY",
        "G1_USER_WANTS_CLEAR_PHRASES",
        "D3_IDK_PHRASES",
        "KV_KEY_FORMAT_PATTERN",
        "MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH",
        "MAX_ARCHIVAL_MEMORY_SIZE",
        "MAX_MEMORY_ENTRY_LENGTH",
        "BLOB_PRESSURE_THRESHOLD",
        "RECENT_CALL_SIGNATURE_WINDOW",
        "LOOP_REPEAT_THRESHOLD",
        "THRESHOLD_DEFAULTS",
        "MASKABLE_GATE_KEYS",
        "REMEDY_FLAG_NAMES",
        "ARCHIVAL_LIST_KEYS_CALL",
        "ARCHIVAL_RETRIEVE_ALL_CALL",
        "RETRIEVAL_TOOLS",
        "IDK_FALLBACK_THRESHOLD",
        "memory_backend",
        "gate_applies",
        "gate_enabled",
        "reprompt_specs",
        "suppress_specs",
        "suppress_user_exempt",
        "transform_specs",
        "transform_instruction",
        "transform_ok",
        "disabled_gate_keys",
        "remedy_enabled",
        "blob_pressure_threshold",
        "loop_repeat_threshold",
        "threshold_overrides",
        "loop_repeat_detected",
        "gate_fallback",
        "SEED_TEXT_BY_KEY",
        "seed_text",
        "gate_match",
        "gate_match_any",
        "is_error_result",
        "is_retrieval_success",
        "is_empty_or_failed_retrieval",
        "archival_memory_searched",
        "gate_keys",
        "telemetry_flags",
        "injection_levels",
        "parse_core_memory_add_call",
        "sanitize_kv_key",
        "unique_kv_key",
        "build_archival_add_call",
        "synthesize_reroute_call",
        "introspect_archival_capacity",
        "synthesize_core_full_reroute",
        "find_failing_core_full_call",
        "synthesize_forced_retrieval_call",
        "synthesize_forced_key_search_call",
    )
    _missing = [n for n in _REQUIRED if n not in globals()]
    if _missing:
        raise ImportError(
            "bfcl_eval.model_handler.memory_gates is missing required names: %s"
            % ", ".join(_missing)
        )
except ImportError:
    GateSpec = None  # type: ignore
    MEMORY_GATE_REGISTRY = ()
    REGISTRY_BY_KEY = {}
    G1_USER_WANTS_CLEAR_PHRASES = ()
    D3_IDK_PHRASES = ()
    KV_KEY_FORMAT_PATTERN = r"^[a-z]+(_[a-z0-9]+)*$"
    MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH = 2000
    MAX_ARCHIVAL_MEMORY_SIZE = 50
    MAX_MEMORY_ENTRY_LENGTH = 10000
    BLOB_PRESSURE_THRESHOLD = 8000
    RECENT_CALL_SIGNATURE_WINDOW = 4
    LOOP_REPEAT_THRESHOLD = 1
    THRESHOLD_DEFAULTS = {
        "blob_pressure_threshold": (8000, 1, 10000),
        "loop_repeat_threshold": (1, 1, 3),
    }
    MASKABLE_GATE_KEYS = ()
    REMEDY_FLAG_NAMES = (
        "enable_forced_retrieval", "enable_forced_key_search", "enable_reroute",
    )
    ARCHIVAL_LIST_KEYS_CALL = "archival_memory_list_keys()"
    ARCHIVAL_RETRIEVE_ALL_CALL = "archival_memory_retrieve_all()"
    RETRIEVAL_TOOLS = frozenset()
    IDK_FALLBACK_THRESHOLD = 2

    def memory_backend(test_category):  # type: ignore
        return None

    def gate_applies(spec, test_category):  # type: ignore
        return False

    def reprompt_specs(policy=None, test_category=None):  # type: ignore
        return ()

    def suppress_specs(policy=None, test_category=None):  # type: ignore
        return ()

    def transform_specs(policy=None, test_category=None):  # type: ignore
        return ()

    def transform_instruction(gate_key, tool_error=""):  # type: ignore
        return ""

    def transform_ok(original, rewritten, tool_error=""):  # type: ignore
        return False, "registry unavailable"

    def suppress_user_exempt(gate_key, user_message):  # type: ignore
        return False

    def gate_enabled(policy, key):  # type: ignore
        # Mirrors the real gate_enabled's gate_default polarity marker exactly (see
        # bfcl_eval.model_handler.memory_gates.gate_enabled's docstring) rather than
        # hardcoding True, so a policy carrying "gate_default": false resolves
        # identically whether or not bfcl_eval happens to be importable. In THIS
        # branch MASKABLE_GATE_KEYS is always () (there is no registry to mask
        # without bfcl_eval), so `key not in MASKABLE_GATE_KEYS` is always true and
        # this always returns True regardless of gate_default — a key not in
        # REGISTRY_BY_KEY is not in this stub's REMEDY_FLAG_NAMES analog either.
        if not isinstance(policy, dict):
            return True
        if key not in MASKABLE_GATE_KEYS:
            return True
        section = policy.get("gate_enabled")
        if isinstance(section, dict) and key in section:
            value = section[key]
            if isinstance(value, bool):
                return value
        default = policy.get("gate_default")
        if isinstance(default, bool):
            return default
        return True

    def disabled_gate_keys(policy):  # type: ignore
        if not isinstance(policy, dict):
            return ()
        return tuple(sorted(k for k in MASKABLE_GATE_KEYS if not gate_enabled(policy, k)))

    def remedy_enabled(policy, name, default=False):  # type: ignore
        return default

    def blob_pressure_threshold(policy):  # type: ignore
        return BLOB_PRESSURE_THRESHOLD

    def loop_repeat_threshold(policy):  # type: ignore
        return LOOP_REPEAT_THRESHOLD

    def threshold_overrides(policy):  # type: ignore
        return ()

    def loop_repeat_detected(signatures, threshold=1):  # type: ignore
        seq = list(signatures)
        n = max(int(threshold), 1)
        return len(seq) >= n + 1 and seq.count(seq[-1]) >= n + 1

    def gate_fallback(key):  # type: ignore
        return ""

    SEED_TEXT_BY_KEY = {}

    def seed_text(key):  # type: ignore
        return ""

    def gate_match(spec, text):  # type: ignore
        return False

    def gate_match_any(spec, texts):  # type: ignore
        return False

    def is_error_result(result):  # type: ignore
        s = (result if isinstance(result, str) else str(result)).strip()
        return bool(s) and (s.lower().startswith("error") or '"error"' in s)

    def is_retrieval_success(last_result):  # type: ignore
        return False

    def is_empty_or_failed_retrieval(last_result, last_call):  # type: ignore
        return False

    def archival_memory_searched(text):  # type: ignore
        return False

    def gate_keys(domain=None):  # type: ignore
        return frozenset()

    def telemetry_flags():  # type: ignore
        return frozenset()

    def injection_levels():  # type: ignore
        return {}

    def parse_core_memory_add_call(call_str):  # type: ignore
        return None

    def sanitize_kv_key(key):  # type: ignore
        return str(key)

    def unique_kv_key(key, existing_keys, max_attempts=3):  # type: ignore
        return None

    def build_archival_add_call(*, key=None, value=None, text=None):  # type: ignore
        return ""

    def synthesize_reroute_call(call_str, *, max_entry_length, existing_keys=None, archival_full=False):  # type: ignore
        return None

    def introspect_archival_capacity(mem_inst, backend):  # type: ignore
        return True, None

    def synthesize_core_full_reroute(call_str, mem_inst, backend):  # type: ignore
        return None

    def find_failing_core_full_call(decoded_calls, execution_results):  # type: ignore
        return None

    def synthesize_forced_retrieval_call(backend):  # type: ignore
        return None

    def synthesize_forced_key_search_call(backend):  # type: ignore
        return None


# ── T3: teacher search constraint (anchoropt training-side policy) ────────────
# The teacher should spend its one-edit-per-batch budget on template keys that
# have REAL, gate-free headroom — not on rewriting text the deterministic gates
# already deliver for free (editing working gate text mostly regresses: lf1's G4
# edit cost kv −8pp) or on signals the evaluator never even fires. This is a
# training policy, not a gate semantic, so it lives in the adapter (on top of the
# registry) rather than in the canonical GateSpec data.
#
# FROZEN = do-not-edit. Two reasons a key is frozen:
#   (a) it's a WORKING deterministic gate — its text already works; edits regress:
#       on_core_clear_blocked (G1), on_premature_idk (D3),
#       on_domain_error_core_full (G3), on_domain_error_key_not_found (G4).
#   (b) it NEVER FIRES in the memory evaluator, so editing it is wasted budget.
#       Two distinct ways a key ends up here: the evaluator never passes the
#       triggering signal at all (on_new_function_available, on_turn_start_action,
#       on_ambiguous_target; plus the reroute key on_core_full_rerouted, already
#       net −9.33pp) — or the trigger exists but is structurally unreachable on the
#       scored split, which is 100% single-turn (turn_k is always 0):
#       on_turn_boundary_orient requires turn_k > 0 (template_engine.py) and so can
#       never fire on any scored case, only a hypothetical multi-turn one.
#
# NOT frozen (editable headroom): on_retrieval_success_pre_answer & on_idk_fallback
# (now backend-scoped by T2), on_memory_preamble (storage policy), the G5 rec_sum
# opt-in gate (on_domain_error_too_long, ships empty → any text is net-new rec_sum
# coverage), and the general reactive signals that do fire.
TEACHER_FROZEN_KEYS: frozenset = frozenset({
    "on_core_clear_blocked",
    "on_premature_idk",
    "on_domain_error_core_full",
    "on_domain_error_key_not_found",
    "on_new_function_available",
    "on_turn_start_action",
    "on_ambiguous_target",
    "on_turn_boundary_orient",
    "on_core_full_rerouted",
    "on_forced_archival_retrieval",
    # Phase 7: G4's forced remedy. Frozen for the same reason as the other two remedy
    # keys — it is only reachable behind an opt-in flag the GATE-SPACE arm queue
    # searches, so a teacher text edit would be scored against an arm that is off.
    "on_forced_key_search",
})


def teacher_editable_keys(template_keys) -> frozenset:
    """Editable template keys = the given template-key set minus TEACHER_FROZEN_KEYS.

    Caller passes template_engine.TEMPLATE_KEYS (the adapter stays BFCL-side/stdlib
    and must not import the anchoropt template engine). Returned set is what the
    training loop accepts an edit for.
    """
    return frozenset(template_keys) - TEACHER_FROZEN_KEYS


def teacher_key_doc_block(
    domain: str = None, indent: str = "  ", exclude: frozenset = frozenset(),
    include: frozenset = None,
) -> str:
    """Render "  key — trigger_desc" lines for the teacher system prompt.

    Used by run_memory_train.py (Phase 4) to replace the hand-maintained
    "Template keys and their purpose" block so new registry keys become
    teacher-visible for free. Filterable by domain so a general-signal-only
    consumer (no memory backends) can restrict itself to domain="general".

    exclude: keys to omit entirely (T3 — hide frozen keys so the teacher never
    proposes an edit that would just be rejected). Default empty = list all.

    include: if given, the key universe is restricted to exactly this set (applied
    after exclude). MEMORY_GATE_REGISTRY has keys outside TEMPLATE_KEYS entirely
    (e.g. on_blob_pressure) that `exclude=TEACHER_FROZEN_KEYS` alone does not hide —
    they are not frozen, just not teacher-editable, so they'd be advertised then
    rejected as unknown_key. Pass include=teacher_editable_keys(TEMPLATE_KEYS) to
    restrict the doc block to the teacher's real, editable key universe.
    """
    specs = MEMORY_GATE_REGISTRY
    if domain is not None:
        specs = [g for g in specs if g.domain == domain]
    if exclude:
        specs = [g for g in specs if g.key not in exclude]
    if include is not None:
        specs = [g for g in specs if g.key in include]
    return "\n".join(f"{indent}{g.key} — {g.trigger_desc}" for g in specs)


def error_signal_to_key_map(domain: str = None) -> dict:
    """error_signal -> template key, for the failure-context builder.

    Replaces the hardcoded error->template mapping in
    run_memory_train.py's _build_failure_context (Phase 4). Only includes
    specs with a non-None error_signal (most gates fire on structural
    conditions, not a classified signal, and are excluded here).
    """
    specs = MEMORY_GATE_REGISTRY
    if domain is not None:
        specs = [g for g in specs if g.domain == domain]
    return {g.error_signal: g.key for g in specs if g.error_signal}


# ── Canonical gate-configuration fingerprint ──────────────────────────────────

def _resolve_threshold(policy: dict, name: str):
    """Resolved value of a threshold, mirroring the real accessors.

    Uses the module's own accessor when available so the fingerprint agrees with
    what the run will actually execute (including clamping and malformed-value
    normalization); falls back to a bounds-checked read otherwise.
    """
    accessor = {"blob_pressure_threshold": globals().get("blob_pressure_threshold"),
                "loop_repeat_threshold": globals().get("loop_repeat_threshold")}.get(name)
    if callable(accessor):
        return accessor(policy)
    spec = THRESHOLD_DEFAULTS[name]
    if isinstance(spec, (tuple, list)):
        default, lo, hi = (list(spec) + [None, None])[:3]
    else:
        default, lo, hi = spec, None, None
    raw = (policy or {}).get(name)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return default
    if lo is not None and val < lo:
        return default
    if hi is not None and val > hi:
        return default
    return val


def gate_config_fingerprint(policy: dict) -> str:
    """Canonical hash of the ENTIRE gate configuration a run will execute.

    SIGNAL_POLICY_PLAN §8.3 makes this prerequisite infrastructure for the
    iterative loop. A newly installed gate changes the *transition dynamics* of
    the system, so reusing the incumbent's stored memory state invalidates the
    experiment — and the failure is silent: the run cache-HITs and the new policy
    is scored against the old system's memory. A plausible-looking wrong accept,
    not a crash.

    The store `base_key` upstream folds in specific KNOWN knobs one at a time
    (`disable_gates`, `enable_reroute`, `enable_forced_retrieval`,
    `enable_forced_key_search`, `gate_off:`, `thr:`). That enumeration is exactly
    why it cannot cover a *new* gate key: nothing in the list matches a gate the
    registry did not have when the list was written.

    This function is defined over the gate-config SURFACE rather than a fixed
    list, so a gate added later is covered without touching the cache logic:

      - every registry gate key and whether it is enabled under this policy
      - every remedy flag and its resolved value
      - every threshold and its resolved value
      - each gate's remedy_type and trigger_kind (a gate whose ACTION changed is
        a different system even at the same key and enabled state)

    Returns a short hex digest. Stable across runs: all iteration is over sorted
    keys, so the digest is a function of configuration only.
    """
    import hashlib
    import json as _j

    policy = policy or {}
    parts: list = []

    try:
        keys = sorted(gate_keys())
    except Exception:
        keys = []
    for k in keys:
        try:
            on = bool(gate_enabled(policy, k))
        except Exception:
            on = True
        spec = None
        try:
            spec = REGISTRY_BY_KEY.get(k)
        except Exception:
            spec = None
        # The TRIGGER CONTRACT is part of the system, not decoration: changing
        # match_substrings changes WHEN the gate fires. Omitting it meant the §77 casing
        # correction produced an IDENTICAL fingerprint, hence an identical store base_key,
        # so a fixed gate would reuse the broken gate's world -- the §4.3 snapshot trap
        # exactly. Also folds reroute_destination, since a reroute to a different tool is a
        # different action at the same key/family.
        _subs = ",".join(getattr(spec, "match_substrings", ()) or ()) if spec else ""
        _dest = getattr(spec, "reroute_destination", "") if spec else ""
        parts.append("gate:%s=%s:%s:%s:%s:%s" % (
            k, int(on),
            getattr(spec, "remedy_type", "?") if spec else "?",
            getattr(spec, "trigger_kind", "?") if spec else "?",
            _subs, _dest,
        ))

    try:
        for name in sorted(REMEDY_FLAG_NAMES):
            parts.append("remedy:%s=%s" % (name, int(bool(remedy_enabled(policy, name)))))
    except Exception:
        pass

    # Thresholds: record the RESOLVED value, not the raw policy entry. A threshold
    # explicitly set to its own default — or set to a malformed value that
    # normalizes back to the default — describes the same system as omitting it,
    # so it must hash identically. Recording raw values would split the key on
    # `"loop_repeat_threshold": "2"` (malformed -> default 1), invalidating every
    # pre-existing store for no behavioural difference.
    try:
        for name in sorted(THRESHOLD_DEFAULTS):
            spec = THRESHOLD_DEFAULTS[name]
            default = spec[0] if isinstance(spec, (tuple, list)) else spec
            try:
                resolved = _resolve_threshold(policy, name)
            except Exception:
                resolved = default
            parts.append("thr:%s=%s" % (name, resolved))
    except Exception:
        pass

    return hashlib.sha256(_j.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
