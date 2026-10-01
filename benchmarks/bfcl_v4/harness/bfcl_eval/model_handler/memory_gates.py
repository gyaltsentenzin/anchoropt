"""
Memory-task signal/gate registry — single source of truth for AnchorOpt's memory
injection points, consumed by BOTH the official BFCL handler
(bfcl_eval/model_handler/base_handler.py) and the AnchorOpt evaluator
(anchoropt/anchoropt/memory_evaluator.py, via the thin adapter at
anchoropt/anchoropt/memory_gates.py).

Stdlib-only (dataclasses/typing only — no anchoropt import, no heavy bfcl imports,
not even base_handler or multi_turn_utils) so importing this module is cheap and
cycle-free from either side. See anchoropt/README.md ("Signals vs. gates") and
anchoropt/docs/ for the design rationale that motivated this file.

STATUS: Phase 0 (registry as inert data) and Phase 1 (base_handler.py / memory_evaluator.py
routed through gate_fallback() / gate_applies() / gate_match_any() below, replacing their
former inline fallback literals and substring guards) are both complete. Phase 2 (rec_sum
gates: on_domain_error_too_long / on_blob_pressure, both still opt-in via empty default
fallback) is also wired into both call sites. Phase 3 (opt-in reroute remedy: see
synthesize_core_full_reroute / find_failing_core_full_call below) is wired into both call
sites' G3 blocks, gated behind enable_reroute (default off). Phase 6 (opt-in forced-retrieval
remedy: see synthesize_forced_retrieval_call below) is wired into both call sites' D3 blocks,
gated behind enable_forced_retrieval (default off) — targets Gemma/Qwen's dominant failure
(give up without ever calling archival retrieval), for which G1/G3 are inert (see
anchoropt/docs/cross_model_error_attribution.md). See anchoropt/test/test_memory_gates.py
for the consistency tests (golden literals, call-site presence, applicability matrix).

Phase 7 (2026-08-03) makes three previously-unsearchable gate levers real:
  - on_forced_key_search: G4's missing "forced" arm (opt-in via enable_forced_key_search,
    default off) — dispatches archival_memory_list_keys() on the model's behalf after a
    "Key not found" error, mirroring what forced retrieval does for D3.
  - on_blob_pressure: given real fallback text, and its trigger point moved from the
    hardcoded BLOB_PRESSURE_THRESHOLD constant to the policy-read
    blob_pressure_threshold() accessor.
  - on_loop: promoted from an evaluator-only soft signal (fired on the FIRST repeat,
    empty text, zero references in base_handler.py) to a real structural gate with a
    policy-read loop_repeat_threshold() (default 1 == the historical trigger point),
    real fallback text, and detection mirrored into base_handler.py for the first time.
The two flag-less text changes (on_blob_pressure, on_loop) are the ONLY places Phase 7
moves the unconfigured/null arm; every new lever (the flag and both thresholds)
defaults to the pre-Phase-7 behavior. See each constant's comment for the one-line
revert.

Two kinds of entries, per anchoropt/README.md "Signals vs. gates":
  - Gates (remedy_type in reprompt/suppress/reroute/preempt): a fixed, structural
    trigger condition mirrored from base_handler.py's hardcoded behavior. Only the
    *text* is tunable (via the AnchorOpt transfer contract / ANCHOROPT_POLICY_PATH).
  - Signals (remedy_type="soft"): a runtime classification made by
    injection_engine.py / template_engine.py's signal-priority ladder. Both the firing
    condition AND the text live only in the AnchorOpt evaluator/training world today —
    the official base_handler.py does not wire these up.
"""

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Optional


# Snapshot-dependency vocabulary (§65). See GateSpec.snapshot_anchor.
PRE_SNAPSHOT = "pre_snapshot"     # anchored at/before the storage boundary -> REBUILD
POST_SNAPSHOT = "post_snapshot"   # strictly downstream, cannot alter stored state -> REUSE
_SNAPSHOT_ANCHORS = (PRE_SNAPSHOT, POST_SNAPSHOT)


@dataclass(frozen=True)
class GateSpec:
    key: str                        # policy key == template key, e.g. "on_domain_error_core_full"
    domain: str                     # "memory" | "general"
    trigger_desc: str               # human-readable; feeds the teacher prose (Phase 4)
    backends: tuple = ()            # subset of ("kv","vector","rec_sum"); () = n/a (general) or all (memory)
    remedy_type: str = "soft"       # "reprompt" | "suppress" | "reroute" | "preempt" | "soft"
    injection_level: str = "trailing_user"  # "trailing_user" | "turn_start_user" | "system"
    once_per_turn: bool = True
    fallback: str = ""              # "" = opt-in-via-policy no-op (byte-identical stock)
    telemetry_flag: str = ""        # step_record key set when it fires, e.g. "a1_core_full_gate"
    trigger_kind: str = "soft_ladder"  # "pre_exec_decoded"|"post_exec_result"|"empty_idk"|"blob_pressure"|"soft_ladder"|"system_preamble"
    match_substrings: tuple = ()    # exact tool-error/decoded substrings (centralized)
    error_signal: object = None     # str label matching an injection_engine SIG_* value, for telemetry join; None if not string-classified
    # Substitute tool for remedy_type="reroute". Empty means "use the historical
    # archival_memory_add destination", which is what every pre-existing reroute spec
    # relies on, so adding this field leaves them byte-identical. A learned reroute gate
    # should DECLARE its destination: the value is attested from the corpus by
    # node_recovery_action (§40), and synthesize_reroute_for_spec refuses a destination
    # whose entry limit it does not know rather than defaulting to the wrong one.
    reroute_destination: str = ""

    # ── Snapshot dependency, expressed as the gate's ANCHOR relative to the
    #    snapshot boundary (§65) ──────────────────────────────────────────────
    #
    # The snapshot store is the cached state of the world AFTER the prerequisite/storage
    # phase. That phase boundary is the only thing the cache key needs to reason about:
    #
    #     PRE_SNAPSHOT   the gate is anchored at or before the storage boundary, or its
    #                    action can causally alter state that contributes to the stored
    #                    world. The snapshot must be REBUILT.
    #     POST_SNAPSHOT  the gate is anchored strictly downstream of the boundary and
    #                    cannot change what the storage phase produced. The snapshot is
    #                    REUSED, so control and candidate share one built world.
    #
    # Stated as a DEPENDENCY on the decision point, not as a gate-type heuristic. The
    # tempting shortcut -- "reroute/transform mutate, reprompt/soft do not", or "fires on
    # a read error => harmless" -- is unsound in both directions:
    #
    #   * on_forced_key_search triggers on the same 'Key not found' contract as G4 but
    #     DISPATCHES a call, changing the step sequence and hence what later turns store.
    #     Same trigger, opposite dependency.
    #   * a reprompt is "advisory" yet still alters the conversation the model conditions
    #     on, so if it can fire while storage is still being written it is PRE_SNAPSHOT.
    #
    # What settles it is whether the anchor lies upstream of the boundary and whether the
    # action can reach state on the storage side of it -- which is a property of the gate's
    # position in the pipeline, so it is declared per gate.
    #
    # ── STATUS: INERT. Documentational only; NOTHING reads this field. ──────────
    #
    # No snapshot reuse is implemented and none should be until a residual is found that
    # is EMPIRICALLY query-only. Measured on the incumbent corpus (§65), every gate in the
    # registry has its contract occur during prereq episodes -- including G4, whose
    # 'key not found' fires 19 times there because a storage episode still RETRIEVES keys
    # before updating them. So zero gates qualify for reuse today, and activating the
    # optimization now would ship machinery no case exercises.
    #
    # When a query-only residual does appear, the phase scope must be DERIVED FROM THE
    # MINED CORPUS and that evidence attached to the cache decision. It must NOT be a
    # hand-declared value that can contradict the observed phase distribution: a wrongly
    # reused snapshot scores a candidate against the wrong world and yields a plausible
    # wrong ACCEPT -- silent, not a crash. test_installing_an_explicit_gate_changes_the_
    # fingerprint exists for exactly that failure and caught a real instance of it in G4.
    #
    # Hence PRE_SNAPSHOT is the ONLY permitted value until the derivation exists, and
    # test_snapshot_anchor_is_inert pins that. The field is kept because the abstraction is
    # right; the optimization is not yet earned.
    snapshot_anchor: str = "pre_snapshot"
    # ── Activation semantics ─────────────────────────────────────────────────
    # "default_enabled" : unlisted in a policy's gate_enabled section -> ON.
    #                     This is the historical fail-open behaviour and the
    #                     default here, so every pre-existing spec and every
    #                     pre-existing policy file resolves byte-identically.
    # "explicit"        : unlisted -> OFF. A gate must be explicitly installed by
    #                     the learned policy/ledger to act.
    #
    # A property of the SPEC rather than an "original vs learned" special case, so
    # the rule is declarative and a hand-authored gate could also opt into explicit
    # activation. LEARNED gates use "explicit" so that adding one to the registry
    # cannot silently change the behaviour of any existing configuration — which
    # matters directly for incumbent-relative experiments: the G1 arm must NOT
    # acquire a later-learned gate just because the registry grew.
    activation: str = "default_enabled"


# ── Existing gate fallback text — copied VERBATIM from base_handler.py ─────────────
# (also byte-identical, by design, in memory_evaluator.py — see
#  anchoropt/test/test_memory_gates.py, which checks both files' literal source.)

_G1_FALLBACK = (
    "[Memory Gate] core_memory_clear is not allowed. "
    "Core memory is full — store the fact using archival_memory_add instead. "
    "If you must free a core slot, call core_memory_remove on the "
    "least-important key first, then archival_memory_add to preserve it."
)

_D3_FALLBACK = (
    "[Memory Gate] You have not searched archival memory this turn. You MUST call "
    "archival_memory_key_search or archival_memory_retrieve before concluding the "
    "answer is unknown."
)

_G3_FALLBACK = (
    "[Memory Gate] Core memory is full. Do NOT call core_memory_clear. "
    "Store the fact using archival_memory_add instead. "
    "If you must free a core slot, call core_memory_remove on the "
    "least-important key first, then archival_memory_add to preserve it."
)

# A4 (N0-A4), pre-generation. Deliberately NON-DIRECTIVE: it asks the model to CONSIDER whether
# memory should be consulted, and names both containers because the answer sits in archival-only
# in 10 of the 25 measured read-side cases (core-only 9, both 6) -- advice that says "check core"
# would reach at most 15. It does NOT force retrieval and does not name a specific call: the
# measured gap is one call, and the passing episodes reach it themselves (memory_retrieve 51,
# core_memory_retrieve 50). Forcing a call would also fire on the 23 no-fact cases, where pushing
# a read at an empty store is the harm this arm has to avoid.
_TURN_START_FALLBACK = (
    "[Memory Gate] Before answering, consider whether this question depends on information "
    "stored earlier. If it might, consult memory first -- core memory and archival memory are "
    "both available. If it does not, answer directly."
)

_G4_FALLBACK = (
    "[Memory Gate] The key you tried to retrieve was not found. "
    "Call archival_memory_key_search or archival_memory_list_keys to discover "
    "the correct stored key, then retrieve it with the exact key returned."
)

# New (Phase 3, not yet wired) — non-empty because it's only reachable behind the
# opt-in enable_reroute flag; on stock (enable_reroute unset/false) this text is
# never rendered, so it does not affect byte-identical-stock guarantees.
_REROUTE_FALLBACK = (
    "[Memory Gate] Core memory was full, so the fact was stored in archival memory "
    "instead. Do not retry core_memory_add — it is saved. Continue with the task."
)

# New (Phase 6, not yet wired) — non-empty for the same reason as _REROUTE_FALLBACK:
# only reachable behind the opt-in enable_forced_retrieval flag.
_FORCED_RETRIEVAL_FALLBACK = (
    "[Memory Gate] Archival memory was searched on your behalf (see the result "
    "above). If the answer is present there, use it. Only say the information is "
    "not available if archival memory is empty or does not contain it."
)

# New (Phase 7) — G4's forced remedy. Non-empty for the same reason as the two
# fallbacks above: only reachable behind the opt-in enable_forced_key_search flag,
# which defaults off, so this text is never rendered on stock.
_FORCED_KEY_SEARCH_FALLBACK = (
    "[Memory Gate] The stored keys were listed on your behalf (see the result "
    "above). Retrieve the fact again using one of those exact key names. Only say "
    "the information is not available if none of the listed keys holds it."
)

# Phrases the model's own message must contain for G1's "user explicitly asked to
# clear" carve-out, and for D3's "this looks like an IDK answer" detection.
# Centralized here so Phase 1 can replace the inline tuples in base_handler.py /
# memory_evaluator.py with these.
G1_USER_WANTS_CLEAR_PHRASES = ("forget everything", "clear everything", "reset memory")
D3_IDK_PHRASES = (
    "i do not know", "i don't know", "i don't have",
    "no information", "not available", "cannot find",
    "do not have information", "not in memory",
    # Evidenced in cross_model_error_attribution.md: Gemma's actual give-up phrasing
    # is "...but could not find..." (a near-miss on "cannot find" above), and Qwen
    # says "i cannot answer this question" (27x in the baseline logs) — neither
    # matched the original list, so D3 never fired for either model's real wording.
    "could not find", "couldn't find", "cannot answer this question",
)

# Key-format guard reused by the Phase 3 reroute AST synthesis — verbatim from
# memory_kv.py:_is_valid_key_format.
KV_KEY_FORMAT_PATTERN = r"^[a-z]+(_[a-z0-9]+)*$"
# memory_kv.py:MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH — reroute must fall back to G3 text
# (never silently drop the fact) if the value would exceed this once rerouted.
MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH = 2000
# memory_kv.py / memory_vector.py:MAX_ARCHIVAL_MEMORY_SIZE — identical in both backends;
# used only as a fallback when the live instance doesn't expose its own capacity
# (see introspect_archival_capacity below, which prefers live introspection for vector).
MAX_ARCHIVAL_MEMORY_SIZE = 50
# memory_rec_sum.py:MAX_MEMORY_ENTRY_LENGTH — used by the Phase 2 blob-pressure gate.
MAX_MEMORY_ENTRY_LENGTH = 10000
# DEFAULT blob-pressure trigger point (80% of MAX_MEMORY_ENTRY_LENGTH). Read through
# blob_pressure_threshold(policy) at both call sites so a training candidate can vary
# it; this constant is the value both sites see when the policy says nothing.
BLOB_PRESSURE_THRESHOLD = 8000

# Rolling window of recent non-errored (tool, args) call signatures used by the
# on_loop repetition gate. 4 is the historical deque(maxlen=4) in
# memory_evaluator.py; single-sourced here so base_handler.py's mirror can't drift.
# It is also the ceiling on LOOP_REPEAT_THRESHOLD: firing after N repeats needs N+1
# identical signatures to be visible at once, so N <= WINDOW - 1.
RECENT_CALL_SIGNATURE_WINDOW = 4
# DEFAULT number of REPEATS (not occurrences) of an identical successful call before
# the on_loop gate fires. 1 == "fires on the 2nd occurrence", which is byte-identical
# to the historical `sigs[-1] in sigs[:-1]` membership test. Read through
# loop_repeat_threshold(policy) at both call sites.
LOOP_REPEAT_THRESHOLD = 1

# New (Phase 7) — on_blob_pressure's real text. Declared here (not with the other
# fallbacks above) only because it interpolates MAX_MEMORY_ENTRY_LENGTH.
#
# UNLIKE _REROUTE_FALLBACK / _FORCED_RETRIEVAL_FALLBACK / _FORCED_KEY_SEARCH_FALLBACK,
# this one is NOT behind an opt-in flag: authoring it is what turns the proactive
# rec_sum gate from "condition fires, injects nothing" into a real gate, which is the
# whole point of making its threshold searchable — with an empty fallback the
# threshold arm would measure exactly 0.00pp. It therefore DOES change the
# null/unconfigured rec_sum arm (one proactive nudge per turn once the blob passes the
# threshold). Setting this back to "" restores the previous no-op behavior in one line;
# BFCL_DISABLE_ANCHOROPT_GATES=1 also still switches it (and every gate) off.
_BLOB_PRESSURE_FALLBACK = (
    "[Memory Gate] Your memory is approaching the "
    f"{MAX_MEMORY_ENTRY_LENGTH}-character limit and further appends will start "
    "failing. Call memory_update now to rewrite the whole memory as a compact, "
    "fact-dense summary — keep every number, name, date and identifier verbatim and "
    "drop only redundant wording — before you store anything else. Do not call "
    "memory_clear."
)

# New (Phase 7) — on_loop's real text. Same "not behind a flag" note as
# _BLOB_PRESSURE_FALLBACK: previously the key shipped empty AND base_handler.py never
# checked the condition at all, so this is net-new injection on the null arm in both
# pipelines (deliberately — the loop threshold is unsearchable otherwise).
#
# Slot-free ON PURPOSE: this text is rendered by base_handler.py's
# anchoropt_template(), which does not expand {slots}. (The trained policies' own
# on_loop text uses {last_result} and is slot-rendered only on the evaluator side —
# a pre-existing property shared with on_domain_error_core_full's {error}, not
# something introduced here.)
# A5 (LEARNED CANDIDATE, N0 line). Permissive by construction, per A4's lesson: A4v1 prescribed
# retrieval and cost -4.95pp. This must NOT name a tool, function, key or query string, must NOT
# assert the answer is wrong or that a fact exists, and must leave "the retrieval was sufficient"
# as an acceptable conclusion. Text frozen in docs/N0_A5_CRITERIA_FROZEN.md before the arm ran.
# B1 (BASELINE / LEARNED CANDIDATE, N0 line). Deliberately GENERIC: it names no tool, no destination
# and no remedy, because the whole point is to test whether ATTENTION alone recovers what the
# specific anchors recover. The model already SEES every error -- tool results are appended to the
# conversation as role=tool -- so this supplies salience and an explicit retry request, not
# information.
#
# Permissive per S4.10.1 (diagnostic certainty determines policy specificity): "repeated failure" is
# an ambiguous signal about WHY, so prescribing a fix would invent a cause. Every branch stays open.
_FUTILITY_FALLBACK = (
    "[Memory Gate] That attempt has now failed several times in a row, and the error above says "
    "why. Do not repeat the same call again. Reconsider what the error is telling you and take a "
    "different action -- adjust the arguments, use a different tool, or move on if the information "
    "cannot be stored as requested."
)

# E1 (N0 line). MECHANISM TEST, not an anchor-acceptance experiment: 15 firings on ONE chain
# (memory_vector-customer), so it cannot pass use-case invariance whatever its delta. It exists to
# isolate CONSTRAINED EXECUTION from DELEGATED RECOVERY with the locus held fixed -- B1 showed that
# handing the recovery choice to the model costs -6.60pp because it reinterprets "do something else"
# as core_memory_clear (11 -> 33).
#
# There is NO text: E1 is a structural remedy. The anchor removes one duplicate and retries. Nothing
# is asked of the model, which is the entire point of the arm.
_E1_FALLBACK = ""

_LOW_SIM_FALLBACK = (
    "The information retrieved may be a weak match for what was asked. Before answering, "
    "consider whether it is sufficient. If it is not, consider searching again with different "
    "wording. If it is sufficient, answer as normal."
)

_LOOP_FALLBACK = (
    "[Memory Gate] You just repeated a call that already succeeded — its result is "
    "above. Do not call it again. Take a different action: use the result you "
    "already have, or move on to the next step the task requires."
)


# ── The registry ─────────────────────────────────────────────────────────────────
# Order follows the plan doc's table: existing gates, new memory gates/remedies,
# memory-domain signals, then the general (v3-compatible) signal ladder.

MEMORY_GATE_REGISTRY: tuple = (
    # ═══════════════════════════════════════════════════════════════════════════════════════════
    # PRUNED FOR THIS REPO. Upstream this registry carried 34 gates; 30 were hand-authored in an
    # earlier fork (its G1/G3/G4/G5, on_turn_start_action, on_blob_pressure, on_low_similarity_*,
    # and so on) and are NOT part of the A1-A4 result. They have been removed rather than left
    # inert, so nothing here is anyone else's hand-written policy.
    #
    # What remains is exactly the dispatch A1-A4 fire through:
    #     on_domain_error_core_full        A1  opens the reroute block on a capacity error
    #     on_core_full_rerouted           A1  the reroute itself
    #     on_domain_error_key_not_found    A2  advise searching archival after a failed read
    #     on_redundant_write_suppressed    A3  cancel a normalised-identical re-write
    #
    # A1's repair and A4's trigger are code paths (enable_capacity_repair / enable_reroute /
    # enable_zero_call_reprompt), not registry entries, so they are unaffected.
    #
    # Consequence worth knowing: on_core_clear_blocked was the one inherited gate that was
    # DEFAULT-ON with no policy. Removing it means a bare run can no longer silently inherit it.
    # See benchmarks/bfcl_v4/VENDORED.md.
    # ═══════════════════════════════════════════════════════════════════════════════════════════
    GateSpec(
        key="on_redundant_write_suppressed",
        activation="explicit",
        domain="memory",
        trigger_desc=(
            "LEARNED (N0-A3): the model proposes an archival_memory_add whose key/value "
            "pair the store already holds -- typically the write A1 already rescued. "
            "Cancelled pre-execution."
        ),
        backends=("kv", "vector"),
        remedy_type="suppress",
        injection_level="trailing_user",
        once_per_turn=False,
        trigger_kind="pre_exec_decoded",
        match_substrings=("archival_memory_add",),
        error_signal=None,
        telemetry_flag="redundant_write_gate",
        fallback="",
    ),
    GateSpec(
        key="on_domain_error_core_full",
        domain="memory",
        trigger_desc=(
            "G3: a core_memory_add/core_memory_clear call just failed with a "
            "core/archival-full tool error — redirects to archival_memory_add."
        ),
        backends=("kv", "vector"),
        remedy_type="reprompt",
        injection_level="trailing_user",
        once_per_turn=True,
        fallback=_G3_FALLBACK,
        telemetry_flag="a1_core_full_gate",
        trigger_kind="post_exec_result",
        match_substrings=("is full", "exceeds maximum size"),
        error_signal="last_call_domain_error_core_full",
    ),
    GateSpec(
        key="on_domain_error_key_not_found",
        domain="memory",
        trigger_desc=(
            "G4: a KV retrieve call just failed with 'Key not found' — redirects "
            "to key_search/list_keys before retrying."
        ),
        backends=("kv",),
        remedy_type="reprompt",
        injection_level="trailing_user",
        once_per_turn=True,
        fallback=_G4_FALLBACK,
        telemetry_flag="a2_key_not_found_gate",
        trigger_kind="post_exec_result",
        match_substrings=("Key not found",),
        error_signal="last_call_domain_error_key_not_found",
    ),
    GateSpec(
        key="on_core_full_rerouted",
        domain="memory",
        trigger_desc=(
            "Reroute (new, opt-in via enable_reroute): instead of only telling the "
            "model to use archival_memory_add after a core-full error, the fact is "
            "automatically re-dispatched to archival on the model's behalf, then "
            "the model is informed of what happened."
        ),
        backends=("kv", "vector"),
        remedy_type="reroute",
        injection_level="trailing_user",
        once_per_turn=True,
        # Non-empty is safe: only reachable when enable_reroute=True, which defaults
        # off, so this text is never rendered on stock.
        fallback=_REROUTE_FALLBACK,
        telemetry_flag="reroute_gate",
        trigger_kind="post_exec_result",
        match_substrings=("is full", "exceeds maximum size"),
        error_signal="last_call_domain_error_core_full",
    ),
)

class _RetiredGates(dict):
    """Registry lookup that returns an INERT spec for the MANUALLY TUNED gates.

    PRIOR WORK THIS BUILDS ON -- WITH THANKS

    The ~30 manually tuned memory gates in this registry (G1/G3/G4/G5, on_turn_start_action,
    on_blob_pressure, on_premature_idk, and the rest), together with the whole gate-dispatch
    machinery -- the GateSpec abstraction, `suppress_specs`, `gate_match_any`, the signal ladder,
    and the generic executor that made a gate installable as pure registry data with no
    per-signal wiring -- are earlier hand-authored work this project builds on.
    **A1-A4 fire through that machinery.** It is not scaffolding we replaced; it is the substrate
    the learned anchors run on, and this line would not exist without it.

    That groundwork established the premise everything else rests on: **a local intervention at a
    specific decision point can move this benchmark**, and by a lot. Hand-tuned anchors reached
    58.67 % where a global prompt reached 41.33 % -- which is what made "locality is where the value
    is" a claim worth pursuing rather than a hunch. It also identified the right decision points and
    the right action vocabulary; the incision-point taxonomy used throughout this repo is a
    formalisation of distinctions that machinery already drew.

    **Those two numbers are on a DIFFERENT SPLIT and must not be differenced against A1-A4's.** They
    come from the earlier single-domain healthcare holdout (n=75); A1-A4's 29.04 % -> 42.24 % is the
    303-query train side of the balanced cell fold. Different corpora, different denominators --
    exactly the cross-job comparison this project forbids elsewhere (two controls here read 29.04 %
    and 24.36 % purely from cell composition while agreeing to 1 flip in 228 on shared cases). Read
    58.67 % as "locality can be worth a lot on the corpus it was measured on", never as a target
    A1-A4 fell short of.

    What hand-authoring could not settle is *how to find such interventions systematically* -- each
    gate came from a person reading trajectories and forming a hypothesis, so the method did not
    generalise past the author's attention. The A1-A4 line takes the same idea and makes it a
    procedure: mine the residual, attribute the failure to a decision, enumerate the admissible
    actions at that point, and install only what survives a paired counterfactual against
    pre-registered criteria. Four anchors, 29.04 % -> 42.24 %, no global prompt, nothing hand-authored.

    So the manual gates are RETIRED here, not repudiated. They are the motivating evidence and the
    engineering foundation; they are simply not part of *this* measurement. Keeping them enabled would
    confound the two questions -- a "control" carrying a hand-tuned gate is not a control for a learned
    one, and the delta would no longer isolate what the learning procedure contributed.

    HOW THE RETIREMENT WORKS

    Their definitions are gone from MEMORY_GATE_REGISTRY. But `base_handler.py` still contains ~26
    `REGISTRY_BY_KEY["on_premature_idk"]`-style bracket lookups at their old call sites, and a bare
    KeyError there would crash mid-episode. Surgically deleting 26 references from a 1854-line
    upstream file we do not own is the riskier option -- it edits the exact code path every frozen
    result depends on, for no behavioural gain. So a retired key resolves to a spec that can never
    match and can never be enabled:

        match_substrings = ()   gate_match_any(...) -> False, so it can never fire
        default = False         never enabled by any policy
        backends = ("__retired__",)  no real backend classifies to this

    Net effect: those call sites evaluate to "this gate did not fire", which is what we want -- the
    inherited gates are inert BY CONSTRUCTION, not merely switched off. Enabling one from a policy
    also cannot work, because it is not in MEMORY_GATE_REGISTRY at all.

    `gate_match_any() -> False` is the load-bearing guarantee: a gate whose trigger never matches
    never fires, whatever else is true of it.
    """

    def __missing__(self, key: str):
        import dataclasses as _dc

        template = MEMORY_GATE_REGISTRY[0]
        overrides = {"key": key, "default": False}
        for field in _dc.fields(template):
            if field.name in ("match_substrings", "phrases", "any_of", "all_of"):
                overrides[field.name] = ()
            elif field.name == "backends":
                overrides[field.name] = ("__retired__",)
        return _dc.replace(
            template, **{k: v for k, v in overrides.items() if hasattr(template, k)}
        )


REGISTRY_BY_KEY: dict = _RetiredGates({g.key: g for g in MEMORY_GATE_REGISTRY})


# ── Helpers (single source, consumed by both base_handler.py and memory_evaluator.py
#    once Phase 1 lands; unused by any call site as of Phase 0) ────────────────────

def memory_backend(test_category: str):
    """Classify a BFCL test_category string into a memory backend, or None.

    Mirrors bfcl_eval.utils.is_memory ("memory" in test_category) combined with the
    ad-hoc "rec_sum" / "kv" substring checks scattered across base_handler.py and
    memory_evaluator.py. Does not import bfcl_eval.utils to stay import-cheap and
    cycle-free; the substring is duplicated intentionally (it's one line and stable —
    ALL_AVAILABLE_MEMORY_BACKENDS in constants/category_mapping.py enumerates the
    same three backends: kv, vector, rec_sum).
    """
    if not test_category or "memory" not in test_category:
        return None
    if "rec_sum" in test_category:
        return "rec_sum"
    if "vector" in test_category:
        return "vector"
    if "kv" in test_category:
        return "kv"
    return None


def gate_applies(spec: GateSpec, test_category: str) -> bool:
    """Whether `spec` is active for `test_category`.

    domain="general" specs always apply (they don't gate on memory backend at all —
    matches how the v3-compatible signal ladder in template_engine.py behaves today).
    domain="memory" specs require test_category to actually be a memory category;
    if spec.backends is non-empty, the classified backend must be a member.

    Global off-switch: BFCL_DISABLE_ANCHOROPT_GATES=1 (or "true"/"yes") makes every
    gate a no-op regardless of domain/backend, for runs that need to be byte-for-byte
    vanilla BFCL (e.g. a clean baseline for a new model). Default is unset, i.e.
    gates behave exactly as before — this is purely opt-in and reversible by not
    setting the var, no code changes needed to "put it back".
    """
    if os.environ.get("BFCL_DISABLE_ANCHOROPT_GATES", "").strip().lower() in ("1", "true", "yes"):
        return False
    if spec.domain == "general":
        return True
    backend = memory_backend(test_category)
    if backend is None:
        return False
    if not spec.backends:
        return True
    return backend in spec.backends


def gate_enabled(policy: dict, key: str) -> bool:
    """Whether the gate `key` is enabled per `policy`'s `gate_enabled` section.

    Deliberately a DIFFERENT mechanism from enable_reroute/enable_forced_retrieval:
    those select a remedy *within* a gate that already fired; this masks the gate
    itself off entirely, before it ever fires. Composes as AND with gate_applies at
    the call site — `gate_applies` decides backend/category eligibility (and honors
    the global BFCL_DISABLE_ANCHOROPT_GATES switch), `gate_enabled` decides whether
    a training/search candidate has this specific gate turned on.

    `policy` is an explicit dict, never read from env or disk — callers pass
    whatever policy they have in hand (the evaluator's `templates` argument, or
    base_handler's `load_anchoropt_policy()`), so a per-candidate value during
    training is visible here without any file I/O.

    An explicit per-key bool in the "gate_enabled" section always wins. Absent
    that, the policy's own top-level "gate_default" bool sets the fallback: a
    policy that opts into `"gate_default": false` (e.g. a genuinely-empty null
    baseline) means an unlisted key is OFF, not on. Absent *that* too — every
    existing policy file predating this marker, and the `{}` policy returned by
    `load_anchoropt_policy()` when ANCHOROPT_POLICY_PATH is unset — falls back to
    the historical fail-OPEN default, so a plain `bfcl generate` and every already-
    trained policy resolve byte-identically to before this marker existed. A
    malformed policy silently disabling a gate that carries leaderboard points is
    a far worse failure mode than a probe that quietly doesn't run, hence the
    fail-open default rather than fail-closed.

    `gate_default` only ever governs MASKABLE_GATE_KEYS (the structural gates with
    a deterministic code-level trigger), never the full 20-key registry — a soft
    signal's off switch is empty text, not this mask (see MASKABLE_GATE_KEYS'
    docstring), and it has no `gate_enabled(...)` call site to read this function's
    answer anyway. A key outside MASKABLE_GATE_KEYS therefore always resolves to
    enabled here, regardless of `gate_default`, so the marker can't silently imply
    a soft key is "disabled" when nothing in either pipeline honors that.
    """
    if not isinstance(policy, dict):
        return True
    if key not in MASKABLE_GATE_KEYS:
        return True
    section = policy.get("gate_enabled")
    if isinstance(section, dict) and key in section:
        value = section[key]
        if isinstance(value, bool):
            return value
    # activation="explicit": absent from the policy means OFF, not fail-open. Checked
    # AFTER the explicit per-key bool above, so an installed gate still turns on the
    # normal way, and BEFORE gate_default so a learned gate is not swept on by a
    # policy-wide default it predates.
    _spec = REGISTRY_BY_KEY.get(key)
    if _spec is not None and getattr(_spec, "activation", "default_enabled") == "explicit":
        return False
    default = policy.get("gate_default")
    if isinstance(default, bool):
        return default
    return True


def disabled_gate_keys(policy: dict) -> tuple:
    """Sorted tuple of MASKABLE_GATE_KEYS entries disabled by `policy`, explicitly
    or by default.

    Delegates to gate_enabled per key so the two can never disagree. Mask-complete:
    iterates every MASKABLE_GATE_KEYS entry rather than only the ones present in
    the "gate_enabled" section, so a `"gate_default": false` policy with an empty
    (or absent) "gate_enabled" section still reports every maskable key as
    disabled. Without this, such a policy would fold an empty `_gate_off` into the
    snapshot-store cache key and silently hit the historical all-on store's cached
    prereqs. Deliberately scoped to MASKABLE_GATE_KEYS, not the full registry — see
    gate_enabled's docstring; a soft key can never appear here since nothing masks
    it. Used to fold the mask into that cache key (a disabled gate can change
    storage-phase behavior during prereqs) and available for eval-run logging.
    """
    if not isinstance(policy, dict):
        return ()
    # An activation="explicit" gate that a policy never mentions is NOT DISABLED --
    # it was never installed. Reporting it here would (a) fold a phantom key into
    # the snapshot-store cache key, invalidating every historical store the moment a
    # learned gate is added to the registry, and (b) make a stock {} policy look like
    # it had explicitly turned something off, breaking the byte-identical-stock
    # invariant. Only count a gate as disabled if it COULD have been on by default.
    def _explicitly_off(k: str) -> bool:
        spec = REGISTRY_BY_KEY.get(k)
        if spec is not None and getattr(spec, "activation", "default_enabled") == "explicit":
            section = policy.get("gate_enabled")
            listed_false = (isinstance(section, dict) and section.get(k) is False)
            return bool(listed_false)
        return not gate_enabled(policy, k)

    return tuple(sorted(k for k in MASKABLE_GATE_KEYS if _explicitly_off(k)))


# The structural gates — the ones with a deterministic, code-level trigger that both
# pipelines AND the gate_enabled mask apply to. Soft signals (remedy_type="soft") are
# deliberately excluded: they have no structural condition to mask off, only text —
# empty text already means silent, which is the polarity goal for those keys.
# on_loop is the one exception: unlike the other soft keys, it has a registry
# gate_fallback() injected regardless of policy text (both pipelines), so it needs
# an explicit off switch the way the structural gates do.
# Single source for (a) the training loop's mask-arm enumeration and (b) the
# call-site mirror test that proves every one of them is masked in BOTH pipelines.
MASKABLE_GATE_KEYS: tuple = (
    "on_core_clear_blocked",          # G1
    "on_archival_remove_blocked",     # LEARNED (iteration 2); empty fallback -> inert by default
    "on_entry_too_long_transform",    # LEARNED CANDIDATE (iteration 2); transform family
    "on_premature_idk",               # D3
    "on_domain_error_core_full",      # G3
    "on_domain_error_key_not_found",  # G4
    "on_domain_error_too_long",       # G5 (opt-in: empty fallback) — rec_sum-scoped
    "on_vector_entry_too_long_reprompt",  # LEARNED CANDIDATE (corrected G2); vector length
    "on_vector_entry_too_long_rerouted",  # LEARNED CANDIDATE (G3); vector length, reroute
    # G4 (§64). Membership is NOT optional for a learned gate: gate_default only governs
    # MASKABLE_GATE_KEYS, so a key outside this tuple resolves to the historical fail-OPEN
    # default and an activation="explicit" gate is ON for a policy that never mentions it.
    # Caught by test_installing_an_explicit_gate_changes_the_fingerprint, which is exactly
    # the snapshot-cache trap it was written for: the installed gate would have shared the
    # incumbent's store.
    "on_key_not_found_rerouted",          # LEARNED CANDIDATE (G4); kv key-miss, reroute
    "on_key_not_found_reprompt",          # LEARNED CANDIDATE (G4); kv key-miss, advisory
    # R2 (T1 rank 2). Same non-optional membership for the same reason: omitted, these two
    # hit the fail-OPEN branch above and would have been ON for every policy that never
    # mentions them -- i.e. shipped LIVE while their own trigger_desc claims inert. Caught
    # by test_installing_an_explicit_gate_changes_the_fingerprint on the first run, which is
    # the second time that test has caught precisely this omission.
    "on_zero_score_search_rerouted",      # R2 CANDIDATE; kv zero-score search, reroute
    "on_zero_score_search_reprompt",      # R2 CONTRAST; kv zero-score search, advisory
    # A5 (N0 line). Mandatory membership, same reason recorded twice above: a key outside this
    # tuple hits the fail-OPEN branch and would ship LIVE while its trigger_desc claims inert.
    "on_low_similarity_reprompt",         # A5 CANDIDATE; vector low-similarity, reprompt
    # B1 (N0 line). Mandatory membership for the reason recorded three times above: a key outside
    # this tuple hits the fail-OPEN branch and ships LIVE while its trigger_desc claims inert.
    "on_repeated_failure_escalate",       # B1 BASELINE; repeated-failure futility escalation
    # E1 (N0 line). Mandatory membership for the reason recorded four times above: a key outside this
    # tuple hits the fail-OPEN branch and ships LIVE while its trigger_desc claims inert.
    "on_archival_full_evict_duplicate",   # E1 MECHANISM TEST; archival-full duplicate eviction
    # A3 (N0 line). Membership is mandatory for the same reason recorded twice above: a key
    # outside this tuple takes the fail-OPEN default, so an activation="explicit" gate would be
    # ON for every policy that never mentions it -- shipped live while its own trigger_desc
    # claims inert. This is the third learned gate where that would have happened.
    "on_redundant_write_suppressed",  # LEARNED (N0-A3); pre-exec suppress of a duplicate write
    # C2. Membership MANDATORY: a key outside this tuple fail-OPENs, i.e. is active for every
    # policy that never mentions it -- which would silently change A1-A5 itself.
    "on_kv_redundant_write_suppressed",
    # N0-R1. Membership mandatory: outside this tuple an explicit gate is ON for every
    # policy that never names it, which would silently alter A1-A5 itself.
    "on_destructive_authorization",
    # A4 (N0 line). Pre-generation. Membership mandatory for the same fail-OPEN reason recorded
    # above: outside this tuple an explicit gate is ON for every policy that never mentions it.
    "on_turn_start_action",           # LEARNED (N0-A4); pre-generation "consider consulting memory"
    "on_blob_pressure",               # proactive rec_sum (opt-in: empty fallback)
    "on_loop",                        # repeated-call nudge (registry fallback text)
)

# The opt-in REMEDY selectors. Structurally different from MASKABLE_GATE_KEYS:
# a mask turns a gate off entirely; a remedy flag swaps which remedy a gate that
# already fired applies (reprompt-only vs. re-dispatch-a-call-then-reprompt). All
# are plain top-level policy booleans, read via remedy_enabled below.
REMEDY_FLAG_NAMES: tuple = (
    "enable_forced_retrieval",   # D3  -> on_forced_archival_retrieval
    "enable_forced_key_search",  # G4  -> on_forced_key_search      (Phase 7)
    "enable_reroute",            # G3  -> on_core_full_rerouted
)


def remedy_enabled(policy: dict, name: str, default: bool = False) -> bool:
    """Whether the opt-in remedy `name` is on, per `policy`, else `default`.

    The policy-dict counterpart of gate_enabled for the two REMEDY_FLAG_NAMES.
    Reads only the dict handed in — never env, never disk — so a per-candidate
    value during training is visible with no file I/O, exactly like gate_enabled.

    Falls back to `default` on anything that isn't an explicit bool: non-dict
    policy, absent key, or a non-bool value. That last case fixes a live footgun
    in the previous call sites, which used `bool(policy.get(name, False))`:
    a JSON policy with "enable_reroute": "false" (a string) is TRUTHY under bool(),
    so a hand-written policy meaning "off" silently turned the remedy ON. Requiring
    isinstance(v, bool) makes any non-bool read as the caller's default instead.

    `default` (rather than a hardcoded False) is what lets the evaluator keep its
    constructor flags as the run-level default while a candidate policy overrides
    them per-batch in either direction — needed for the training loop to search
    remedy-on vs remedy-off inside ONE run.
    """
    if not isinstance(policy, dict):
        return default
    value = policy.get(name)
    if not isinstance(value, bool):
        return default
    return value


# ── Phase 7: policy-read NUMERIC thresholds ───────────────────────────────────────
# Same contract as gate_enabled / remedy_enabled: read the explicit dict handed in
# (never env, never disk), type-check with isinstance, and fall back to the module
# default on anything malformed so the null/absent-key arm is byte-identical to the
# hardcoded constant these replace. bool is rejected explicitly because
# isinstance(True, int) is True in Python — a policy with "loop_repeat_threshold":
# true would otherwise read as the threshold 1 by accident rather than as malformed.

#: policy key -> (default value, minimum, maximum or None). Single source for the
#: accessors below AND for threshold_overrides()'s cache-key folding.
THRESHOLD_DEFAULTS: dict = {
    "blob_pressure_threshold": (BLOB_PRESSURE_THRESHOLD, 1, MAX_MEMORY_ENTRY_LENGTH),
    "loop_repeat_threshold": (LOOP_REPEAT_THRESHOLD, 1, RECENT_CALL_SIGNATURE_WINDOW - 1),
}


def _threshold(policy: dict, name: str) -> int:
    default, lo, hi = THRESHOLD_DEFAULTS[name]
    if not isinstance(policy, dict):
        return default
    value = policy.get(name)
    # bool first: isinstance(True, int) is True, and True is not a threshold.
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    if value < lo or (hi is not None and value > hi):
        return default
    return value


def blob_pressure_threshold(policy: dict) -> int:
    """Blob length (chars) at which the proactive rec_sum on_blob_pressure gate fires.

    Defaults to BLOB_PRESSURE_THRESHOLD (8000 = 80% of MAX_MEMORY_ENTRY_LENGTH) for
    any policy that doesn't set it, so the unconfigured arm keeps the exact trigger
    point the hardcoded constant gave. Out-of-range values (<=0, or above the hard
    MAX_MEMORY_ENTRY_LENGTH cap, past which the gate could never fire) fall back to
    the default rather than silently disabling the gate.
    """
    return _threshold(policy, "blob_pressure_threshold")


def loop_repeat_threshold(policy: dict) -> int:
    """Number of REPEATS of an identical successful call before on_loop fires.

    1 (the default) means "fires on the 2nd occurrence" — byte-identical to the
    historical `sigs[-1] in sigs[:-1]` membership test. Capped at
    RECENT_CALL_SIGNATURE_WINDOW - 1 because firing after N repeats requires N+1
    identical signatures to be simultaneously visible in the rolling window, so a
    larger N could never fire and would be an accidental off-switch.
    """
    return _threshold(policy, "loop_repeat_threshold")


def threshold_overrides(policy: dict) -> tuple:
    """Sorted ("<name>=<value>",) for every threshold `policy` moves off its default.

    The numeric counterpart of disabled_gate_keys: both on_blob_pressure and on_loop
    can fire during a prereq/storage episode, so a candidate that changes WHEN they
    fire must miss the incumbent's snapshot store or the two arms get scored against
    byte-identical memory state. Returns () for every policy that leaves both
    thresholds alone, which is what keeps the cache key of existing runs unchanged.
    """
    out = []
    for name in sorted(THRESHOLD_DEFAULTS):
        value = _threshold(policy, name)
        if value != THRESHOLD_DEFAULTS[name][0]:
            out.append(f"{name}={value}")
    return tuple(out)


def loop_repeat_detected(signatures, threshold: int = LOOP_REPEAT_THRESHOLD) -> bool:
    """True iff the most recent call signature has already been seen `threshold`
    times earlier in the rolling window (i.e. it is now its (threshold+1)-th
    occurrence).

    `signatures` is the rolling deque of recent NON-ERRORED (tool, args) call
    strings, oldest first; only the last RECENT_CALL_SIGNATURE_WINDOW entries are
    retained by the callers, so "occurrences" are counted within that window rather
    than over the whole episode.

    threshold=1 reduces exactly to the historical membership test
    `len(sigs) >= 2 and sigs[-1] in sigs[:-1]`: one earlier occurrence of the newest
    signature. threshold=3 needs 4 identical signatures, which is exactly what a
    window of RECENT_CALL_SIGNATURE_WINDOW=4 can hold — the boundary case, reachable
    without widening the window (which would also change the {last_result}-adjacent
    state any other reader of that deque sees).
    """
    seq = list(signatures)
    n = max(int(threshold), 1)
    if len(seq) < n + 1:
        return False
    return seq.count(seq[-1]) >= n + 1


def gate_fallback(key: str) -> str:
    """Return the registry's default fallback text for `key`, or "" if unknown."""
    spec = REGISTRY_BY_KEY.get(key)
    return spec.fallback if spec else ""


# ── Phase 4: on-enable seed library ───────────────────────────────────────────
# Structural gates carry their working text in GateSpec.fallback, so enabling
# them (gate_default) already yields real behavior for free. on_domain_error_
# too_long (G5) is the one MASKABLE_GATE_KEYS entry that does NOT: its
# fallback="" by design (opt-in, see the GateSpec comment above), so unmasking
# it with no explicit policy text is a guaranteed no-op. This dict is the
# on-enable default for exactly that gap — proven text from
# anchoropt/policies/templates_initial.json's batch-0 seed, minus the {error}
# slot (neither pipeline renders slots at G5's call site). Not a general
# per-key override of gate_fallback: only keys with no other proven-text
# source belong here. Deliberately NOT covering on_blob_pressure — Phase 7
# already gave it real fallback text, so it needs no seed.
SEED_TEXT_BY_KEY: dict = {
    "on_domain_error_too_long": (
        "[Context]\nMemory too long. Call memory_update to rewrite the full "
        "memory as a compact, fact-dense summary — preserving all exact "
        "numbers, names, dates, and identifiers verbatim — then re-append the "
        "new information."
    ),
    # Vector's per-entry length contract. Necessarily separate text from G5's, and not
    # just a scoping nicety: G5's seed says "call memory_update to rewrite the full
    # memory", which is rec_sum's whole-blob API. vector has no memory_update and no
    # single blob -- it stores discrete entries via core_memory_add. The same words
    # would name a tool that does not exist on this backend.
    #
    # The instruction targets ~60% of the limit on purpose. Measured on the first G2
    # corpus, the model's own retries undershoot: rejected rewrites clustered at
    # 341/350/344/379/389/405 against a 300 limit, i.e. it aims AT the limit and
    # overshoots. Asking for a stricter budget than the constraint is the §25.5
    # escalation rung expressed as the opening instruction.
    "on_vector_entry_too_long_reprompt": (
        "[Context]\nThat memory entry was rejected for exceeding the per-entry "
        "character limit. Retry the SAME write with a shorter entry — aim for about "
        "60% of the limit, since entries near the limit keep being rejected. "
        "Preserve every number, name, date and concrete fact verbatim; compress the "
        "surrounding prose. If the content cannot fit in one entry, split it across "
        "separate entries rather than truncating it."
    ),
    # G4 (§64) contrast arm. Lives HERE and not in the spec's fallback: fallback="" is what
    # keeps a learned gate inert on addition, while seed_text supplies text the moment a
    # policy installs it. Putting the text in fallback (my first attempt) made the registry
    # addition itself behaviour-changing -- caught by
    # test_every_learned_gate_is_explicit_activation -- and removing it without a seed made
    # the gate fire and inject nothing, caught by
    # test_every_generic_reprompt_gate_resolves_to_text. The two tests together pin the only
    # correct arrangement.
    "on_key_not_found_reprompt": (
        "[Context]\nThat key is not in core memory. The value may still be in archival "
        "memory.\n[Next]\nRetrieve it from archival memory using the same key."
    ),
    # R2 (T1 rank 2) contrast arm. Same fallback-vs-seed arrangement as G4 above, for the
    # same two reasons: fallback="" keeps the gate inert on registry addition, and a seed
    # keeps it from firing into silence once installed.
    #
    # The text names the OBSERVED failure rather than the constraint, because there is no
    # constraint here -- the search SUCCEEDED and returned nothing useful. The model's own
    # measured next move is another key_search (33 of 34 of which failed), so the
    # instruction has to redirect away from searching and toward READING a value: the only
    # follow-up the corpus attests resolving this locus (retrieve 16.7%, 1/6, vs another
    # search 2.9%, 1/34).
    #
    # It deliberately does NOT name a specific key. The whole failure is that the ranking
    # carried no information about which key is relevant, so suggesting one would be
    # inventing evidence the trajectory does not have -- list_keys is offered instead, and
    # note list_keys resolved 0/6 on its own, so it is a fallback, not the recommendation.
    "on_zero_score_search_reprompt": (
        "[Context]\nThat key search returned matches, but every relevance score was zero "
        "\u2014 none of those keys actually match the query, so the result is empty in "
        "substance.\n[Next]\nDo not search again with a reworded query. Read a value "
        "directly instead: retrieve the specific key you need, or list the available keys "
        "and then retrieve the one that matches."
    ),
}


def seed_text(key: str) -> str:
    """On-enable default text for `key`, or "" if it has none.

    Middle tier between explicit policy text and gate_fallback(key): a gate
    whose registry fallback is genuinely empty (G5 today) still needs SOME
    text the moment its gate_default mask is switched on with no per-key
    policy override, or the toggle is silently inert. Keys with real
    gate_fallback() text (structural gates, on_loop, on_blob_pressure) don't
    need an entry here — this only fills the gap gate_fallback leaves.
    """
    return SEED_TEXT_BY_KEY.get(key, "")


def gate_match(spec: GateSpec, text) -> bool:
    """True if any of spec.match_substrings appears in `text` (str or stringified)."""
    if not spec.match_substrings:
        return False
    s = text if isinstance(text, str) else str(text)
    return any(sub in s for sub in spec.match_substrings)


def gate_match_any(spec: GateSpec, texts) -> bool:
    """True if any of spec.match_substrings appears in any item of `texts`.

    Convenience for the common call-site shape
    `any(sub in str(r) for sub in subs for r in execution_results)`.
    """
    if not spec.match_substrings:
        return False
    return any(gate_match(spec, t) for t in texts)


# Matches a JSON object that begins with an "error" key even if the payload is
# not strictly parseable (single quotes, trailing junk). Mirrors the equivalent
# constant in anchoropt.tool_ranker._is_error.
_ERROR_JSON_PREFIX = re.compile(r'^\{\s*[\'"]error[\'"]\s*:', re.IGNORECASE)


def is_error_result(result) -> bool:
    """Structural check: did a tool call fail?

    True iff the result BEGINS with an error marker ("Error during execution: …",
    the executor's exception wrapper) or is a JSON object with a top-level "error"
    key. Content that merely contains the word "error" mid-text is NOT an error.

    Used by the reroute remedy at BOTH call sites (base_handler + memory_evaluator)
    to decide whether a re-dispatched archival_memory_add actually succeeded. This
    replaces a bare `'"error"' in result` substring test, which misses the executor's
    "Error during execution:" wrapper (no quoted "error" token) and would therefore
    report a failed reroute as success — telling the model a fact was stored when it
    was not. Deliberately kept here (stdlib-only, bfcl-side) rather than reusing
    anchoropt.tool_ranker._is_error, which must stay BFCL-import-free; the two share
    identical logic and are cross-checked in test_memory_gates.py.
    """
    if not result:
        return False
    s = result.strip() if isinstance(result, str) else str(result).strip()
    if not s:
        return False
    if s.lower().startswith("error"):
        return True
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            return isinstance(obj, dict) and "error" in obj
        except (ValueError, TypeError):
            return bool(_ERROR_JSON_PREFIX.match(s))
    return False


# Archival RETRIEVAL calls only — never a write. "archival_memory_retrieve" is a
# substring of "archival_memory_retrieve_all", so that variant matches for free.
ARCHIVAL_RETRIEVAL_SUBSTRINGS = (
    "archival_memory_retrieve",
    "archival_memory_key_search",
    "archival_memory_list_keys",
)


def archival_memory_searched(text) -> bool:
    """True iff `text` contains an archival RETRIEVAL call this step — never a
    write (archival_memory_add/_remove/_update/_clear).

    Replaces the blunt `"archival_memory" in text.lower()` check both call sites
    used for D3's "archival searched this turn" tracking, which incorrectly
    counted a turn where the model only called archival_memory_add (storage, not
    search) as having "searched".
    """
    s = (text if isinstance(text, str) else str(text)).lower()
    return any(sub in s for sub in ARCHIVAL_RETRIEVAL_SUBSTRINGS)


# Memory retrieval tool names — used to detect failed_search_streak for the
# on_idk_fallback soft key. Single-sourced here (was a private frozenset local to
# memory_evaluator.py only) so base_handler.py's new wiring reads the identical set.
RETRIEVAL_TOOLS = frozenset({
    "core_memory_retrieve",
    "archival_memory_retrieve",
    "memory_retrieve",
    "core_memory_key_search",
    "archival_memory_key_search",
    "core_memory_list_keys",
    "archival_memory_list_keys",
    "core_memory_retrieve_all",
    "archival_memory_retrieve_all",
})

# Consecutive failed/empty searches before on_idk_fallback fires. Read directly
# (not through a policy-overridable helper like loop_repeat_threshold) because,
# unlike on_loop's threshold, no training arm tunes this today.
IDK_FALLBACK_THRESHOLD = 2


def is_retrieval_success(last_result: str) -> bool:
    """Return True when the last tool call was a successful memory retrieval.

    Matches result payloads from memory_kv (value/keys), memory_vector
    (similarity_score/text/ranked_results), and memory_rec_sum (memory_content).
    Excludes error dicts and empty-result payloads so the signal only fires when
    actual content was retrieved — not when the model called list_keys and got [].

    Mirrors anchoropt.injection_engine.is_retrieval_success exactly (that module is
    deliberately BFCL-import-free, so it keeps its own copy; this one is for
    base_handler.py, which cannot import anchoropt). Cross-checked in
    test_memory_gates.py.
    """
    if not last_result:
        return False
    if '"error"' in last_result:
        return False
    indicators = ('"value":', '"memory_content":', '"similarity_score":',
                  '"ranked_results":', '"keys":', '"text":')
    if not any(m in last_result for m in indicators):
        return False
    empty_patterns = ('"keys": []', '"ranked_results": []', '"result": []', '"value": null', '"memory_content": ""')
    return not any(p in last_result for p in empty_patterns)


def is_empty_or_failed_retrieval(last_result: str, last_call: str) -> bool:
    """Return True when the last call was a retrieval tool but returned nothing useful.

    Byte-for-byte mirror of anchoropt.memory_evaluator._is_empty_or_failed_retrieval
    (single-sourced here so base_handler.py's on_idk_fallback wiring can never drift
    from the training path's streak logic — the two are NOT the same check as
    `not is_retrieval_success(...)` above, which recognizes a narrower set of
    "empty" payloads; this list is deliberately independent).
    """
    if not any(t in last_call for t in RETRIEVAL_TOOLS):
        return False
    if is_error_result(last_result):
        return True
    empty = ('"keys": []', '"ranked_results": []', '"result": []',
             '"value": null', '"memory_content": ""', '"memory_content": null')
    return any(p in last_result for p in empty)


def gate_keys(domain: str = None) -> frozenset:
    """All registry keys, optionally filtered to one domain ("memory" | "general")."""
    if domain is None:
        return frozenset(REGISTRY_BY_KEY.keys())
    return frozenset(k for k, g in REGISTRY_BY_KEY.items() if g.domain == domain)


def transform_specs(policy: dict = None, test_category: str = None) -> tuple:
    """Every ENABLED post-execution TRANSFORM gate, as data.

    `transform` is the first GENERATIVE action family: unlike suppress (delete a
    call) and reroute (substitute a call), it rewrites an ARGUMENT so the same
    intent satisfies a constraint the tool rejected. That needs a model call, not a
    string operation.

    Defined from evidence, not convenience. Recovery attribution over all 15
    self-corrected entry-length failures found semantic_compression 15/15 (100%) and
    literal truncation 0/15; content words were kept at median 53% while
    NUMBERS/FACTS were kept at median 100%. So the primitive is "compress to satisfy
    the constraint while preserving factual content" — truncation would cut
    mid-sentence and drop whichever facts fall past the cutoff, fixing the write
    while breaking the downstream query.

    Same accessor shape as suppress_specs so both call sites can iterate specs
    rather than hardcoding keys.
    """
    out = []
    for g in MEMORY_GATE_REGISTRY:
        if g.remedy_type != "transform":
            continue
        if test_category is not None and not gate_applies(g, test_category):
            continue
        if policy is not None and not gate_enabled(policy, g.key):
            continue
        out.append(g)
    return tuple(out)


# Per-gate transform instruction. The CONSTRAINT is data; the executor is generic.
# {limit} is filled from the tool error when a number is present.
TRANSFORM_INSTRUCTIONS = {
    "on_entry_too_long_transform": (
        "The previous memory write was rejected: the entry exceeded the {limit}-character "
        "limit. Rewrite the SAME content to fit within {limit} characters.\n"
        "Requirements:\n"
        "  - PRESERVE every number, amount, percentage, date, name and concrete fact.\n"
        "  - Compress prose: drop filler, use abbreviations and symbols.\n"
        "  - Do NOT truncate mid-sentence and do NOT invent content.\n"
        "Reply with ONLY the rewritten value, no quotes and no explanation."
    ),
}
TRANSFORM_DEFAULT_LIMIT = 300


def transform_instruction(gate_key: str, tool_error: str = "") -> str:
    """Prompt for a transform gate, with the observed constraint substituted in."""
    import re as _re
    tmpl = TRANSFORM_INSTRUCTIONS.get(gate_key, "")
    if not tmpl:
        return ""
    m = _re.search(r"(\d{2,})", tool_error or "")
    limit = m.group(1) if m else str(TRANSFORM_DEFAULT_LIMIT)
    return tmpl.replace("{limit}", limit)


def transform_ok(original: str, rewritten: str, tool_error: str = "") -> tuple:
    """Validate a transform output. Returns (accepted, reason).

    A generative action can fail in ways a structural one cannot — it may return
    prose, exceed the limit anyway, or drop the facts. Rejecting a bad rewrite and
    falling through is safer than writing corrupted state, so validation is part of
    the primitive rather than an afterthought.
    """
    import re as _re
    if not rewritten or not rewritten.strip():
        return False, "empty rewrite"
    r = rewritten.strip().strip('"').strip("'")
    m = _re.search(r"(\d{2,})", tool_error or "")
    limit = int(m.group(1)) if m else TRANSFORM_DEFAULT_LIMIT
    if len(r) > limit:
        return False, f"rewrite still {len(r)} chars > {limit}"
    if len(r) < 0.15 * len(original or ""):
        return False, "rewrite lost too much content (<15% of original length)"
    # fact preservation: numbers present in the original must survive
    def nums(t):
        return set(_re.findall(r"\d[\d,.]*", t or ""))
    lost = nums(original) - nums(r)
    if lost and len(lost) > 0.5 * len(nums(original)):
        return False, f"dropped {len(lost)} of {len(nums(original))} numeric facts"
    return True, f"ok ({len(original or '')} -> {len(r)} chars, facts preserved)"


def enabled_explicit_gate_keys(policy: dict = None) -> tuple:
    """activation="explicit" gates that a policy INSTALLS (turns on).

    Counterpart to disabled_gate_keys, and required for cache-key correctness.
    disabled_gate_keys reports only gates that are OFF, which is sufficient while
    every gate is on by default: turning one off is then the only way to change
    behaviour. An `explicit` gate inverts that — it is OFF unless installed, so the
    behaviour-changing event is ENABLING it, and disabled_gate_keys cannot see that.

    Without this term a policy that installs a learned gate hashes identically to one
    that never mentions it, so the installed gate reuses the incumbent's snapshot
    store and is scored against the incumbent's memory state. That is the §Stage 5
    snapshot-cache trap: a plausible-looking wrong accept, not a crash. Measured
    before the fix: "learned explicitly ON" and "learned never mentioned" both
    produced disabled_gate_keys() == () while enabling different suppress specs.
    """
    if not isinstance(policy, dict):
        return ()
    out = []
    for g in MEMORY_GATE_REGISTRY:
        if getattr(g, "activation", "default_enabled") != "explicit":
            continue
        if gate_enabled(policy, g.key):
            out.append(g.key)
    return tuple(sorted(out))


def suppress_specs(policy: dict = None, test_category: str = None) -> tuple:
    """Every ENABLED pre-execution suppress gate, as data.

    DC ordering is signal -> policy -> execution: the signal is a GateSpec's
    match_substrings, the policy is remedy_type="suppress", and EXECUTION should be
    generic over both. Before this accessor the executor hardcoded
    "on_core_clear_blocked" at four points in each of two mirrored call sites, so a
    newly learned suppress gate could be correctly proposed and then not installed
    -- the loop could not install what it could propose.

    Filtering here rather than at the call site keeps one source of truth for "which
    suppress gates apply right now", so the two mirrored executors cannot drift.
    """
    out = []
    for g in MEMORY_GATE_REGISTRY:
        if g.remedy_type != "suppress" or g.trigger_kind != "pre_exec_decoded":
            continue
        if test_category is not None and not gate_applies(g, test_category):
            continue
        if policy is not None and not gate_enabled(policy, g.key):
            continue
        out.append(g)
    return tuple(out)


# COMPATIBILITY LAYER, and a recorded piece of architecture debt (§27.8).
#
# TAXONOMY DEBT: `remedy_type` currently conflates two independent things -- the ACTION
# CATEGORY (what kind of intervention this is: reprompt / suppress / reroute /
# transform) and the EXECUTION PRIMITIVE (what code runs it). They coincide for
# suppress and transform, and they do NOT for reprompt: G3 synthesizes a reroute call,
# G4 forces a key-search call, while G5 and the vector length candidate simply inject
# text and retry. So "every gate with remedy_type=='reprompt' shares an executor" is
# false, and assuming it silently changed two DEFAULT-ENABLED gates that fire in the G1
# incumbent.
#
# The proper fix is to separate the two axes (e.g. an explicit `execution` field), so
# membership in an executor family is declared rather than inferred from the category.
# Deliberately NOT done now: it would touch every gate mid-experiment, and the
# corrected G2 run must not be preceded by an ontology refactor. This allow-list is the
# minimum that makes the candidate valid; treat it as scaffolding to delete once the
# taxonomy is split, not as the intended design.
#
# Gates whose execution IS the generic "inject text, spend a step, retry" path, and
# which are therefore driven by reprompt_specs() rather than by a block of their own.
#
# Deliberately an allow-list. G3/G4 share remedy_type="reprompt" but synthesize calls
# (reroute / forced key-search), so they must keep their bespoke executors; adding them
# here would change two DEFAULT-ENABLED gates that fire in the G1 incumbent. A new
# reprompt gate opts in by being added here, which is a one-line, reviewable change.
GENERIC_REPROMPT_KEYS: tuple = (
    "on_domain_error_too_long",           # G5, rec_sum blob overflow
    "on_vector_entry_too_long_reprompt",  # corrected G2 candidate, vector entry length
    "on_key_not_found_reprompt",          # G4 contrast arm, kv key-miss advisory
    "on_zero_score_search_reprompt",      # R2 contrast arm, kv zero-score search advisory
)


# Gates driven by the GENERIC post-execution recovery-substitution executor. Same
# allow-list discipline as GENERIC_REPROMPT_KEYS: membership is declared, because
# remedy_type alone does not determine execution semantics (§27.8 taxonomy debt).
#
# on_core_full_rerouted and on_forced_key_search keep their bespoke blocks -- they are
# DEFAULT-ENABLED and fire in the G1 incumbent, so routing them through a new executor
# would change incumbent behaviour mid-loop. The G3 candidate is explicit/opt-in and can
# use the generic path safely.
GENERIC_REROUTE_KEYS: tuple = (
    "on_vector_entry_too_long_rerouted",   # G3 candidate: vector length -> archival write
    "on_key_not_found_rerouted",           # G4 candidate: kv key-miss -> archival READ
    "on_zero_score_search_rerouted",       # R2 candidate: kv zero-score search -> value READ
)


def reroute_specs(policy: dict = None, test_category: str = None) -> tuple:
    """Every ENABLED post-execution RECOVERY-SUBSTITUTION gate, as data.

    Same accessor shape as reprompt_specs/transform_specs/suppress_specs, so a newly
    learned reroute gate is executed without new code. The substitute itself is MINED
    from trajectories (node_recovery_action); this only supplies the execution path.
    """
    out = []
    for g in MEMORY_GATE_REGISTRY:
        if g.key not in GENERIC_REROUTE_KEYS:
            continue
        if g.remedy_type != "reroute" or g.trigger_kind != "post_exec_result":
            continue
        if not g.match_substrings:
            continue
        if test_category is not None and not gate_applies(g, test_category):
            continue
        if policy is not None and not gate_enabled(policy, g.key):
            continue
        out.append(g)
    return tuple(out)


def reprompt_specs(policy: dict = None, test_category: str = None) -> tuple:
    """Every ENABLED post-execution REPROMPT gate, as data.

    Same accessor shape as transform_specs/suppress_specs, and added for the same
    reason those exist: the reprompt executor hardcoded "on_domain_error_too_long" at
    five points, so a newly learned reprompt gate could be proposed, installed in a
    policy, and then never fire -- present in the registry and inert in execution.
    That is the §24 defect class where the record and the behaviour disagree.

    This matters for the corrected G2 specifically. The vector-length candidate is the
    GENERIC reprompt family bound to a BACKEND-SPECIFIC trigger; it is not a new action
    type. Iterating specs here is what lets a second reprompt gate exist without a
    second copy of the executor block.

    `trigger_kind == "post_exec_result"` excludes proactive gates (e.g.
    on_blob_pressure), which fire on live state rather than on a tool error and so need
    their own condition rather than a match against execution_results.

    SCOPE, and why it is an explicit allow-list rather than "every reprompt gate".
    `remedy_type == "reprompt"` is necessary but NOT sufficient: G3
    (on_domain_error_core_full) and G4 (on_domain_error_key_not_found) also carry that
    remedy_type, are DEFAULT-ENABLED, and have bespoke executors that do more than
    inject text -- G3 synthesizes a reroute call via synthesize_core_full_reroute() and
    G4 forces a key-search call. Routing them through the generic inject-and-retry path
    would silently change two gates that fire in the G1 incumbent, which is precisely
    the kind of collateral the refactor must not cause.

    So membership is by explicit opt-in. A gate joins only when its execution genuinely
    IS "inject this text, spend a step, retry", which is true of G5 and of the vector
    length candidate. Anything with a specialized executor keeps its own block.
    """
    out = []
    for g in MEMORY_GATE_REGISTRY:
        if g.key not in GENERIC_REPROMPT_KEYS:
            continue
        if g.remedy_type != "reprompt" or g.trigger_kind != "post_exec_result":
            continue
        if not g.match_substrings:
            continue
        if test_category is not None and not gate_applies(g, test_category):
            continue
        if policy is not None and not gate_enabled(policy, g.key):
            continue
        out.append(g)
    return tuple(out)


# Per-gate exemption: a suppress gate may be skipped when the USER explicitly asked
# for the destructive action. Keyed by gate, because it is gate-specific reasoning
# that must NOT be silently inherited by a newly learned gate whose semantics
# differ. G1 exempts "forget everything"; on_archival_remove_blocked has no such
# exemption -- a user asking to forget things is not asking to prune archival
# storage mid-task.
SUPPRESS_USER_INTENT_EXEMPTIONS = {
    "on_core_clear_blocked": G1_USER_WANTS_CLEAR_PHRASES,
}


def suppress_user_exempt(gate_key: str, user_message: str) -> bool:
    """Did the user explicitly ask for this gate's destructive action?"""
    phrases = SUPPRESS_USER_INTENT_EXEMPTIONS.get(gate_key, ())
    if not phrases:
        return False
    u = (user_message or "").lower()
    return any(p in u for p in phrases)


def telemetry_flags() -> frozenset:
    """All non-empty telemetry_flag values across the registry (gates only — signals
    that haven't been given a dedicated step_record flag return "" and are excluded)."""
    return frozenset(g.telemetry_flag for g in MEMORY_GATE_REGISTRY if g.telemetry_flag)


def injection_levels() -> dict:
    """key -> injection_level for every registry entry."""
    return {g.key: g.injection_level for g in MEMORY_GATE_REGISTRY}


# ── Phase 3: opt-in reroute remedy — AST synthesis + bounded fallbacks ────────────
# Pure, stdlib-only, no live-instance access. Callers (base_handler.py /
# memory_evaluator.py) extract plain values (existing keys, capacity, entry-length
# cap) from the live MemoryAPI instance via introspect_archival_capacity below, then
# call synthesize_reroute_call. Returning None at any point means "abandon the
# reroute" — callers must fall back to the existing on_domain_error_core_full (G3)
# text so the fact is never silently dropped.

_KV_KEY_RE = re.compile(KV_KEY_FORMAT_PATTERN)


def parse_core_memory_add_call(call_str: str) -> Optional[dict]:
    """AST-parse a bare `core_memory_add(...)` call string into its literal args.

    Returns {"key": str, "value": str} for the KV shape (2 args) or {"text": str}
    for the vector shape (1 arg). Returns None if call_str is not syntactically a
    bare core_memory_add call with only string-literal arguments (defensive — an
    unparseable call means the reroute is abandoned, not guessed at).
    """
    try:
        node = ast.parse(call_str.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if not isinstance(node, ast.Call):
        return None
    func_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
    if func_name != "core_memory_add":
        return None
    if any(not isinstance(a, ast.Constant) for a in node.args):
        return None
    if any(not isinstance(kw.value, ast.Constant) for kw in node.keywords):
        return None

    positional = [a.value for a in node.args]
    keyword = {kw.arg: kw.value.value for kw in node.keywords}
    n = len(positional) + len(keyword)

    if n == 1:
        text = positional[0] if positional else keyword.get("text")
        return {"text": str(text)} if text is not None else None

    if n == 2:
        key = positional[0] if positional else keyword.get("key")
        value = keyword.get("value") if "value" in keyword else (
            positional[1] if len(positional) > 1 else None
        )
        if key is None or value is None:
            return None
        return {"key": str(key), "value": str(value)}

    return None


def sanitize_kv_key(key: str) -> str:
    """Best-effort snake_case coercion for a key that failed the KV format check
    (`^[a-z]+(_[a-z0-9]+)*$`); falls back to a short deterministic hash-based key
    if the coercion still doesn't produce a valid key."""
    candidate = re.sub(r"[^a-z0-9_]+", "_", str(key).strip().lower())
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    if candidate and _KV_KEY_RE.match(candidate):
        return candidate
    return "mem_" + hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:8]


def unique_kv_key(key: str, existing_keys, max_attempts: int = 3) -> Optional[str]:
    """Return `key` unchanged if not in `existing_keys`, else a `_2`/`_3`-suffixed
    variant not already present (bounded at `max_attempts`). None if every variant
    up to `max_attempts` collides — caller abandons the reroute."""
    if key not in existing_keys:
        return key
    for suffix in range(2, max_attempts + 1):
        candidate = f"{key}_{suffix}"
        if candidate not in existing_keys:
            return candidate
    return None


def build_archival_add_call(*, key: str = None, value: str = None, text: str = None) -> str:
    """Render a bare `archival_memory_add(...)` call string for re-dispatch through
    the executor. KV shape: key+value. Vector shape: text only."""
    if text is not None:
        return f"archival_memory_add({text!r})"
    return f"archival_memory_add(key={key!r}, value={value!r})"


def synthesize_reroute_call(
    call_str: str,
    *,
    max_entry_length: int,
    existing_keys=None,
    archival_full: bool = False,
) -> Optional[str]:
    """Given a failing bare `core_memory_add(...)` call string, synthesize an
    equivalent bare `archival_memory_add(...)` call string, or None if the reroute
    should be abandoned (archival already full, value too long, or the call
    couldn't be parsed/sanitized within bounds)."""
    if archival_full:
        return None
    parsed = parse_core_memory_add_call(call_str)
    if parsed is None:
        return None

    if "text" in parsed:
        text = parsed["text"]
        if len(text) > max_entry_length:
            return None
        return build_archival_add_call(text=text)

    key, value = parsed["key"], parsed["value"]
    if len(value) > max_entry_length:
        return None
    if not _KV_KEY_RE.match(key):
        key = sanitize_kv_key(key)
    key = unique_kv_key(key, existing_keys or ())
    if key is None:
        return None
    return build_archival_add_call(key=key, value=value)


def introspect_archival_capacity(mem_inst, backend: str):
    """Duck-typed read of the live archival store's fullness (+ KV existing keys),
    without importing memory_kv.py/memory_vector.py (keeps this module cheap and
    dependency-free — vector's real module pulls in faiss/sentence-transformers).

    Returns (archival_full: bool, existing_keys: frozenset | None). existing_keys is
    None for vector (no keys in that backend).
    """
    archival = getattr(mem_inst, "archival_memory", None)
    if backend == "kv":
        if not isinstance(archival, dict):
            return True, frozenset()
        return len(archival) >= MAX_ARCHIVAL_MEMORY_SIZE, frozenset(archival.keys())
    if backend == "vector":
        store = getattr(archival, "_store", None)
        if not isinstance(store, dict):
            return True, None
        max_size = getattr(archival, "max_size", MAX_ARCHIVAL_MEMORY_SIZE)
        return len(store) >= max_size, None
    return True, None


def synthesize_core_full_reroute(call_str: str, mem_inst, backend: str) -> Optional[str]:
    """End-to-end: given a failing bare `core_memory_add(...)` call string and the
    live memory instance, return a bare `archival_memory_add(...)` call string to
    re-dispatch through the executor, or None if the reroute should be abandoned
    (caller falls back to the existing G3 text — the fact is never silently dropped).
    """
    archival_full, existing_keys = introspect_archival_capacity(mem_inst, backend)
    return synthesize_reroute_call(
        call_str,
        max_entry_length=MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH,
        existing_keys=existing_keys,
        archival_full=archival_full,
    )


def synthesize_reroute_for_spec(spec, call_str: str, mem_inst, backend: str) -> Optional[str]:
    """Synthesize the substitute call for a generic reroute spec.

    Differs from synthesize_core_full_reroute only in WHERE the destination's limit comes
    from: that wrapper always passes MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH because its
    destination is fixed. Here the destination is whatever the spec declares, so the limit
    is looked up per destination and an unknown destination is refused rather than
    defaulted -- a wrong limit would synthesize a call that fails on arrival and be scored
    as "the substitute did not help" rather than as a bug.

    The destination itself is not invented here: it comes from the corpus, via
    node_recovery_action's attestation of successful alternatives (§40).
    """
    dest = getattr(spec, "reroute_destination", None) or "archival_memory_add"

    # WRITE destinations: the substitute stores content, so the destination's entry-length
    # limit must be checked or the synthesized call fails on arrival and is misscored as
    # "the substitute did not help" rather than as a bug.
    write_limits = {
        "archival_memory_add": MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH,
        "archival_memory_update": MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH,
    }
    # READ destinations (§64): the substitute RETRIEVES by key. There is no content to fit,
    # so there is no entry-length limit to check and no capacity question -- an archival
    # store being full does not prevent reading from it. Refusing these for "unknown limit"
    # is what the first G4 wiring did: synthesize_reroute_for_spec returned None for
    # archival_memory_retrieve because the limit table held only writes. Correct guard,
    # wrong table -- a read reroute is a genuinely different shape, not an unknown
    # destination.
    read_dests = ("archival_memory_retrieve", "archival_memory_key_search",
                  "core_memory_retrieve", "core_memory_retrieve_all")

    if dest in read_dests:
        return synthesize_read_reroute_call(call_str, dest)

    if dest not in write_limits:
        return None
    archival_full, existing_keys = introspect_archival_capacity(mem_inst, backend)
    return synthesize_reroute_call(
        call_str,
        max_entry_length=write_limits[dest],
        existing_keys=existing_keys,
        archival_full=archival_full,
    )


_KEY_ARG_RE = re.compile(r"""key\s*=\s*(['"])(.*?)\1""", re.S)

# READ destinations that take NO arguments. ACTIONABILITY (frozen rule, see
# docs/REROUTE_ACTIONABILITY_FROZEN.md): a reroute is actionable iff the destination call
# can be instantiated from information available at the intervention point WITHOUT
# inventing unsupported arguments. A zero-argument destination satisfies that mechanically
# the moment the destination itself is selected -- there is nothing to infer, so there is
# nothing to invent.
#
# This is an EXECUTION-CAPABILITY question, deliberately separate from evidence: membership
# here says a call can be emitted, never that it is worth emitting. Whether the destination
# helps is settled by the paired downstream comparison.
#
# Kept as an explicit allow-list rather than derived from a signature: a destination that
# merely happens to have optional-only arguments is NOT the same as one designed to take
# none, and guessing that from introspection is how a wrong call gets synthesized and then
# misscored as "the substitute did not help".
ZERO_ARG_READ_DESTS = ("core_memory_retrieve_all",)


# Argument name per READ destination. Exact-match lookups take `key`; similarity searches
# take `query`. Derived from the tool signatures, not assumed uniform.
_READ_DEST_ARG = {
    "archival_memory_retrieve": "key",
    "core_memory_retrieve": "key",
    "archival_memory_key_search": "query",
    "core_memory_key_search": "query",
}


def synthesize_read_reroute_call(call_str: str, dest: str) -> Optional[str]:
    """Re-dispatch a failed READ against a different store.

    Two shapes, and they are different in kind:

      * KEYED destination -- the mapping is COPY and nothing else: the key that missed in
        one store is looked up in another. Frozen for G4 from 6 observed pairs at 5/6 = 0.83
        consistency (docs/G4_ACCEPTANCE_FROZEN.md, 448e7164). Returns None when no key can
        be parsed from the failing call, because inventing one is exactly the failure mode
        the actionability rule forbids -- and a WRONG key is worse than no intervention,
        since the model at least keeps searching.

      * ZERO-ARGUMENT destination (ZERO_ARG_READ_DESTS) -- emitted directly. No argument is
        inferred from the corpus, so no argument can be wrong. The earlier version refused
        these on the grounds that a zero-arg destination "is not a COPY mapping at all",
        which is true but answers the wrong question: COPY-ness is about how arguments are
        DERIVED, and a call with no arguments derives none.
    """
    if dest in ZERO_ARG_READ_DESTS:
        return "%s()" % dest
    m = _KEY_ARG_RE.search(call_str or "")
    if m is None:
        # positional single-string form, e.g. core_memory_retrieve("k1")
        m2 = re.match(r"""\s*[a-z_][a-z0-9_]*\s*\(\s*(['"])(.*?)\1\s*\)\s*$""",
                      call_str or "", re.S)
        if m2 is None:
            return None
        key = m2.group(2)
    else:
        key = m.group(2)
    if not key:
        return None
    # The ARGUMENT NAME is per destination, not universal. archival_memory_retrieve takes
    # key= and demands an EXACT match; archival_memory_key_search takes query= and does a
    # fuzzy BM25 lookup. Emitting key= for the latter synthesizes an invalid call, which
    # would be scored as "the substitute did not help" rather than as a bug -- the same
    # misscoring this function's docstring warns about for entry limits.
    #
    # This matters for A2.R specifically: A1's write-side synthesizer SANITISES and DEDUPES
    # keys before archiving, so the archived key frequently differs from the one the reader
    # asks for. An exact-match retrieve therefore misses on the very cases A1 created --
    # measured, archival_memory_retrieve was tried once while key_search resolved 11/11.
    arg = _READ_DEST_ARG.get(dest, "key")
    return '%s(%s="%s")' % (dest, arg, key.replace('"', '\\"'))


def find_failing_call_for_spec(spec, decoded_calls, execution_results) -> Optional[str]:
    """Return the first decoded call whose paired result matches THIS SPEC's contract.

    The generic form of find_failing_core_full_call below, which hardcodes
    REGISTRY_BY_KEY["on_domain_error_core_full"] and therefore only ever matches the
    core-full contract. A generic reroute loop needs the contract to come from the spec it
    is currently evaluating, or it silently finds nothing for every other signal -- which is
    exactly what happened to the G3 reroute arm (§52).

    Index alignment is guaranteed by execute_multi_turn_func_call's contract: one result per
    input call string.
    """
    for call, result in zip(decoded_calls, execution_results):
        if gate_match(spec, result):
            return call
    return None


def find_failing_core_full_call(decoded_calls, execution_results) -> Optional[str]:
    """Return the first decoded call string whose paired execution result matches
    the G3 (core-full) substrings, or None if none did. decoded_calls and
    execution_results must be the same length and index-aligned (guaranteed by
    execute_multi_turn_func_call's contract: one result per input call string)."""
    spec = REGISTRY_BY_KEY["on_domain_error_core_full"]
    for call, result in zip(decoded_calls, execution_results):
        if gate_match(spec, result):
            return call
    return None


# ── Phase 6: opt-in forced-retrieval remedy ───────────────────────────────────────
# Simpler than the reroute remedy above: both target calls are static and zero-arg
# (no AST parsing of a failing call, no live-instance capacity introspection needed
# — they can't fail on content grounds the way a re-dispatched archival_memory_add
# could exceed a length/capacity limit).

# The two zero-arg archival dump calls, named once so the D3 forced-retrieval remedy
# and the Phase 7 G4 forced-key-search remedy cannot drift apart on the literal.
ARCHIVAL_LIST_KEYS_CALL = "archival_memory_list_keys()"
ARCHIVAL_RETRIEVE_ALL_CALL = "archival_memory_retrieve_all()"


def synthesize_forced_retrieval_call(backend: str) -> Optional[str]:
    """Deterministic, backend-appropriate "dump archival memory" call to execute on
    the model's behalf when it's about to give up without ever searching archival
    this turn. KV has no zero-arg archival dump, so list the keys instead (mirrors
    G4's key_search/list_keys fallback); vector's archival store supports a direct
    zero-arg dump of every entry. None for any other backend (rec_sum has no
    archival concept; D3 already scopes to kv/vector only)."""
    if backend == "kv":
        return ARCHIVAL_LIST_KEYS_CALL
    if backend == "vector":
        return ARCHIVAL_RETRIEVE_ALL_CALL
    return None


# ── Phase 7: opt-in forced-KEY-SEARCH remedy (G4's analogue of the above) ─────────
# G4 ("Key not found") previously had only two arms: allow (via gate_enabled) or
# reprompt (its existing nudge text). D3 and G3 each additionally have a "forced" arm
# that dispatches a call on the model's behalf. This is G4's.

def synthesize_forced_key_search_call(backend: str) -> Optional[str]:
    """Deterministic zero-arg call that shows the model which keys actually exist,
    executed on its behalf when a KV retrieve just failed with "Key not found".

    Returns the SAME call as synthesize_forced_retrieval_call's kv branch — that is
    intentional and the literal is shared (ARCHIVAL_LIST_KEYS_CALL): "list the stored
    keys" is the right recovery for both triggers, and G4 is kv-only, so there is no
    vector/rec_sum branch to diverge on. Kept as its own function rather than reusing
    synthesize_forced_retrieval_call directly so the kv-only scope is stated in code
    (a future vector arm for D3 must not silently become a G4 arm).
    """
    if backend == "kv":
        return ARCHIVAL_LIST_KEYS_CALL
    return None
