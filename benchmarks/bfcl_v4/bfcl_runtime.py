"""The BFCL v4 AnchorOpt RUNTIME surface -- what `candidate_search` requires, and nothing else.

`adapter.py` is the benchmark's VOCABULARY and is deliberately untouched by this module: ~294 tests
pin its behaviour, and it predates the runtime interface. It answers "what kind of call is this",
which is necessary and not sufficient -- `candidate_search.step()` also needs to know which
CONDITIONS are observable where, which actions this host can actually execute, and how to turn a
semantic action into a directive. Those are this module's job.

    declared_signals()    Phi_BFCL's names
    signal_aliases()      declared prose phrases, expressibility matching only
    evaluate_signal()     dispatch, with typed-param validation; unknown names RAISE
    feasible_actions()    U_H(l) for this runtime
    apply_action()        semantic action -> semantic directive (no I/O)
    normalize_event()     one host event -> boundary-tagged framework-neutral record
    observable_state()    what Phi may read here, plus explicitly-carried state
    HOST                  the HostProfile

Same shape as `benchmarks/tb2_deepagents/tb2_adapter.py`, which is the module this one is
deliberately analogous to. Duck-typed, no ABC -- the registry declines to pin one on purpose.

WHAT THIS MODULE OWNS THAT THE CORE MUST NEVER SEE
-------------------------------------------------
Memory-container concepts (`core`, `archival`), backend names (`kv`, `vector`, `rec_sum`), BFCL
error wording, and case-id shapes. `anchoropt/` contains none of these and
`tests/test_tb2_adapter.py` greps the core package to keep it that way.

WHY U_H(l) IS NARROWER THAN THE STRUCTURAL GRID
-----------------------------------------------
Declared from what the eight ACCEPTED anchors demonstrably execute (docs/ANCHORS.md), not from what
seems plausible. The rule this project already paid to learn is in docs/CONSUMER_BOUNDARY_RULE.md:
a cell that passes the structural grid can still do nothing at runtime while reporting success. So a
cell is declared here only where an accepted anchor actually ran in it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.learning.executor_capability import (
    ExecutorCapability, disabled_capabilities, unbound_capabilities,
)
from anchoropt.runtime import HostProfile

import re

import adapter as _vocab
from bfcl_capabilities import (
    all_fields, carried_fields, fields_at, is_known_tool, observable_fields, tool_schema,
    tools_of_kind, unknown_args,
)
from bfcl_signals import (
    SIGNAL_ALIASES, SIGNAL_BOUNDARIES, SIGNAL_PARAM_DOMAINS, SIGNAL_PARAMS,
    SIGNAL_PROBE_PARAMS, SIGNALS,
)
# NO FIXTURE IMPORT. `REROUTE_DESTINATIONS` was the accepted anchors' own repairs, and importing it
# here put the answer key in every process that touched the runtime (caught by the smoke run's leak
# detector). What EXISTS is `tool_schema()`; what WORKS is measured. Empty and mutable so the recovery
# test can inject the historical table explicitly.
REROUTE_DESTINATIONS: dict[str, Mapping[str, Any]] = {}

NAME = "bfcl_v4_memory"

# ------------------------------------------------------------------------------------------------
# U_H(l) -- what THIS runtime can execute where
# ------------------------------------------------------------------------------------------------
#
# Evidence per declared cell, all from the accepted stack:
#
#   POST_EXECUTION / REROUTE    repair a refused or vacuous result
#   POST_EXECUTION / REPROMPT   -- ask the model to retry against a reachable place
#   POST_GEN_PRE_EXEC / SUPPRESS  -- cancel a call before it commits
#   POST_GEN_PRE_EXEC / REPROMPT  -- ask for another decision before answering
#
# PRE_GENERATION declares NOOP only. That is a measured exclusion, not an oversight: an earlier version fired a
# byte-identical reprompt there and scored materially worse than the same text one boundary later, because the condition it
# needed ("about to answer without looking") does not exist before generation. Declaring the cell
# would let the search re-propose the arm the project already measured as harmful.
#
# POST_EXECUTION / SUPPRESS is absent for the structural reason: after execution there is nothing
# left to cancel. `anchor.exclusion_reason` enforces that independently.
HOST = HostProfile(
    name=NAME,
    executable={
        IncisionPoint.PRE_GENERATION: frozenset({Action.NOOP}),
        IncisionPoint.POST_GENERATION_PRE_EXEC: frozenset({
            Action.NOOP, Action.REPROMPT, Action.SUPPRESS,
        }),
        IncisionPoint.POST_EXECUTION: frozenset({
            Action.NOOP, Action.REPROMPT, Action.REROUTE,
        }),
    },
    notes=("PRE_GENERATION carries NOOP only: an earlier version measured materially worse firing a byte-identical "
           "reprompt there, because 'about to answer without looking' is not observable before "
           "generation. REROUTE at POST_GENERATION_PRE_EXEC is withheld -- no accepted anchor "
           "rewrites a proposed call in place; the accepted forms suppress it instead."),
)


# ------------------------------------------------------------------------------------------------
# 1. event normalization
# ------------------------------------------------------------------------------------------------
def normalize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """One host/trajectory event -> a boundary-tagged, framework-neutral record.

    Accepts a raw BFCL-shaped step (`tool_calls`/`calls` plus `tool_results`/`result`) or an
    already-flat record. Every benchmark-specific decode happens HERE, via `adapter.py`'s
    predicates, so no BFCL string shape crosses this function into the core.
    """
    calls = event.get("tool_calls")
    if calls is None:
        calls = event.get("calls")
    if isinstance(calls, str):
        calls = [calls]
    calls = [str(c) for c in (calls or [])]

    results = event.get("tool_results")
    if results is None:
        results = event.get("result")
    if isinstance(results, (str, bytes)) or isinstance(results, Mapping):
        results = [results]
    results = list(results or [])

    first_call = calls[0] if calls else ""
    # A step's results are a LIST. `str(list)` is not valid JSON, and stringifying it is exactly the
    # defect `policy_tree.any_vacuous` exists to document -- it made the vacuity detector return
    # None for every payload. So the payload handed on is the FIRST ELEMENT, never the list's repr.
    first_result = results[0] if results else None

    rec: dict[str, Any] = {
        "boundary": event.get("boundary"),
        "has_generation": bool(event.get("has_generation", bool(calls) or "response" in event)),
        "proposes_tool_call": bool(calls),
        "n_tool_calls": len(calls),
        "call": first_call,
        "result": first_result,
        "proposes_write": _vocab.is_write(first_call),
        "proposes_read": _vocab.is_read(first_call),
        "proposes_clear": _vocab.is_clear(first_call),
        "proposes_remove": _vocab.is_remove(first_call),
        "container": _vocab.container_of(first_call),
        "target_id": _vocab.target_id(first_call),
        "tool_calls_so_far": int(event.get("tool_calls_so_far", 0) or 0),
    }
    # error_kind is None for a result that is absent OR that does not look like an error. Those two
    # are different facts and the signals distinguish them: `no_informative_result` requires a
    # PRESENT result with no error marker.
    rec["error_kind"] = _vocab.error_kind(first_result) if first_result is not None else None
    for passthrough in ("best_similarity", "container_full", "is_relocation_target",
                        "case_id", "step"):
        if passthrough in event:
            rec[passthrough] = event[passthrough]
    return rec


# ------------------------------------------------------------------------------------------------
# 2. observable runtime state
# ------------------------------------------------------------------------------------------------
def observable_state(normalized: Mapping[str, Any],
                     carried: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The state Phi_BFCL may read at this boundary, plus any state carried forward.

    `carried` is EXPLICIT: a signal needing evidence from an earlier boundary must have it passed
    here, so cross-boundary dependence is visible in the call rather than hidden in a mutable
    accumulator that later reads cannot audit.
    """
    state = dict(normalized)
    if carried:
        state.update(carried)
    state.setdefault("tool_calls_so_far", 0)
    return state


# ------------------------------------------------------------------------------------------------
# 3. feasible actions -- U_H(l)
# ------------------------------------------------------------------------------------------------
#: WINDOWS WITHIN ONE BOUNDARY, and why this exists (R13).
#:
#: `post_generation_pre_exec` is not one place in this host. It is three, and they differ in what can
#: be executed:
#:
#:   zero_call       the step proposed NO tool call         -> a reprompt can act; there is no call
#:                                                             to cancel
#:   call_filtering  calls are queued for dispatch          -> a suppress can act: drop one before it
#:                                                             commits. A reprompt can too.
#:   answer_boundary the model has finished deciding and    -> a reprompt can act; there are NO
#:                   is about to answer                        pending calls, so SUPPRESS HAS NOTHING
#:                                                             TO CANCEL
#:
#: MEASURED, not reasoned: a suppress controller was installed at `answer_boundary`, fired, and then
#: executed nothing -- the window's only action is injecting a message, so it declined with
#: "controller has no eta.instruction". `HostProfile` is keyed by `IncisionPoint` alone and validates
#: that its keys ARE IncisionPoints, so a window cannot be expressed there without changing core. It
#: is expressed HERE, in the adapter, which is where host-specific structure belongs.
#:
#: The cost of not having this: a policy search offered `reprompt` and `suppress` as equally feasible
#: at this boundary, they tied exactly, and the winner was decided by enum sort order. A tie between a
#: real candidate and an inexecutable one is not a tie.
ACTION_WINDOWS: Mapping[str, frozenset[Action]] = {
    "zero_call": frozenset({Action.NOOP, Action.REPROMPT}),
    "call_filtering": frozenset({Action.NOOP, Action.REPROMPT, Action.SUPPRESS}),
    "answer_boundary": frozenset({Action.NOOP, Action.REPROMPT}),
}


def feasible_actions(boundary: IncisionPoint, site: str = "") -> frozenset[Action]:
    """What this runtime can execute at `boundary`, narrowed by `site` when one is named.

    `site` is the WINDOW within the boundary (see ACTION_WINDOWS). Omitted or unknown, the answer is
    the boundary's own set -- an unrecognised window must not silently narrow to nothing, because
    that would prune every action and read as "this host can do nothing here".

    Narrowing only. A window can never widen what the host declares, and the intersection enforces
    that structurally rather than by convention.
    """
    at_boundary = HOST.executable_actions(boundary)
    window = ACTION_WINDOWS.get(str(site or ""))
    return at_boundary & window if window is not None else at_boundary


# ------------------------------------------------------------------------------------------------
# 4. signal evaluation
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# SIGNAL EXPANSION -- the runtime side of Phi growth
# ------------------------------------------------------------------------------------------------
#
# Core validates a proposed signal (anchoropt/learning/signal_expansion.py) and then asks the RUNTIME
# to install it, because the runtime owns the state surface. Installed signals live in a separate
# table from the declared ones so that:
#   * `declared_signals()` reports both -- the search sees one Phi;
#   * an expanded signal is always distinguishable from a shipped one in any artifact, which matters
#     because a recovery claim made with an expanded Phi is a different claim from one made without;
#   * nothing is written to the frozen SIGNALS map, so a policy referencing a shipped signal cannot
#     change meaning between rounds.
#
# Empty at import. Expansion is a per-session act, never shipped state.
EXPANDED_SIGNALS: dict[str, Any] = {}
# The boundary each expanded signal was VALIDATED at, so `signal_boundaries` can answer for it.
EXPANDED_BOUNDARIES: dict[str, frozenset] = {}
EXPANDED_PROVENANCE: dict[str, str] = {}


def synthesis_fields(boundary=None) -> Mapping[str, Any]:
    """TYPED DECLARATIONS core may search over. Declarations, NOT candidate predicates.

    This replaced a hand-written list of four candidate signals. That list got the plumbing working and
    was the wrong final algorithm: if the adapter proposes the useful predicates, a reader can fairly ask
    what the learner is contributing. So the adapter now declares only what is observable and of what
    type, and `anchoropt/learning/signal_grammar.py` enumerates atoms, derives thresholds from the
    residual's own observed values, and composes conjunctions.

    `searched_other_container` is the one field added for this: it is a per-episode structural fact
    (did any call touch a second store) computed from the adapter's own container vocabulary. It is a
    FIELD, not a condition -- core decides whether "false" matters, at which threshold it matters, and
    whether it matters only in conjunction with something else.
    """
    from bfcl_capabilities import all_fields
    fields = dict(all_fields())
    fields["searched_other_container"] = _Declared(
        name="searched_other_container", type=bool,
        doc="whether any call in the episode so far touched a store other than the first one used")
    # PRE-DISPATCH facts about the call being proposed, declared so the grammar can compose over them.
    # Supplied-but-undeclared is invisible to synthesis (the space is declared x supplied), which is
    # why these need declaring and not merely populating.
    #
    # Named for what they MEASURE, never for the limit they might violate: the grammar derives any
    # threshold from the residual's own observed values, so naming a field after a cap would be the
    # adapter proposing the condition instead of declaring the observable.
    _pre = (IncisionPoint.POST_GENERATION_PRE_EXEC.value,)
    fields["proposed_payload_chars"] = _Declared(
        name="proposed_payload_chars", type=int, boundaries=_pre,
        doc="character length of the largest string argument of the call being proposed")
    fields["proposed_arg_count"] = _Declared(
        name="proposed_arg_count", type=int, boundaries=_pre,
        doc="how many keyword arguments the proposed call carries")
    fields["n_proposed_calls"] = _Declared(
        name="n_proposed_calls", type=int, boundaries=_pre,
        doc="how many calls were proposed together in this step")
    fields["call_index"] = _Declared(
        name="call_index", type=int, boundaries=_pre,
        doc="position of this call within the step's proposed batch")
    # EPISODE-SCOPED CAPACITY EVIDENCE, supplied at the commitment gate.
    #
    # Why it has to be episode-scoped: whether a store is full is established by an EARLIER step's
    # tool result, and the commitment-gate state is built per proposed call. Asking "is the container
    # full" of one call in isolation answers a different question, exactly as `searched_other_
    # container` above would if it were not accumulated.
    #
    # Measured gap this closes: `clear_proposed_at_capacity` reads `container_full`, the live gate
    # state carried neither it nor `error_kind`, and so the signal was UNSATISFIABLE at runtime while
    # passing every synthetic test that supplied the field. A capacity error appears in a prior step's
    # tool_results in 23 of 27 real storage episodes, so the evidence exists -- it was simply never
    # carried to the boundary that needs it.
    #
    # A FIELD, NOT A CONDITION. It says a limit was observed, never that acting on it is warranted:
    # core decides whether "full" matters, in conjunction with what, and at which boundary.
    fields["container_full"] = _Declared(
        name="container_full", type=bool, boundaries=_pre,
        doc=("whether any store in this episode has already reported a capacity limit "
             "(observed from an earlier step's tool result, not predicted)"))
    # ---- THE STORE THIS EPISODE RUNS AGAINST, and its structural consequences (R19) -------------
    #
    # WHY THESE ARE FIELDS AND NOT SCOPE. Backend-specificity used to be enforced three ways, none of
    # them learned: hardcoded `== "rec_sum"` gates in the evaluator, a transform "gated at dispatch",
    # and per-cell arm scoping by the experimenter. All three put applicability OUTSIDE phi, so the
    # learner could neither discover that a policy is backend-specific nor discover that one is
    # general. Declaring them makes applicability a learnable conjunct: the grammar already composes
    # `backend == X AND <mechanism>` (stage 3 pairs atoms over DIFFERENT fields), so nothing is
    # special-cased and no backend -> action table exists anywhere.
    #
    # MEASURED, before any of this shipped: on the real error-bearing residual (671 states, all three
    # backends) synthesis returns 12 backend-combining conjunctions, including
    # `backend == 'rec_sum' AND error_kind == 'blob_would_overflow'` (301/671) and
    # `backend == 'kv' AND tool_calls_so_far > 9.0` (72/671).
    #
    # SCENARIO IS DELIBERATELY ABSENT. `adapter.scenario_of` exists and is NOT declared here: a
    # scenario term would let a predicate memorise that one use case is favourable (kv/healthcare -8
    # vs kv/finance +1 is measured) and generalise to nothing. Backend is a property of the STORE's
    # structure, which is why its consequences below are derivable; a scenario label has no such
    # structure. Supply `backend` as its own field and never the raw case id, which contains both.
    #
    # BOTH LOCI. The backend is constant for the whole episode, so it is knowable before anything is
    # generated and after execution alike -- unlike a result-derived fact, declaring it at the
    # commitment gate leaks nothing about the call being judged.
    _both = (IncisionPoint.POST_GENERATION_PRE_EXEC.value, IncisionPoint.POST_EXECUTION.value)
    fields["backend"] = _Declared(
        name="backend", type=str, boundaries=_both, enum=tuple(_vocab.BACKENDS),
        doc="which memory store implementation this episode runs against")
    # STRUCTURAL PROPERTIES ARE DELIBERATELY *NOT* DECLARED HERE, and this is a measured decision.
    #
    # `has_second_container`, `is_single_blob` and `enforces_unique_keys` are derivable (see
    # `backend_facts`, which is kept: the EXECUTOR uses them for feasibility). But on this corpus the
    # structure -> backend map is BIJECTIVE -- kv (False,True,True), vector (False,True,False),
    # rec_sum (True,False,False) -- so as phi fields they are perfect aliases of `backend` and add
    # exactly zero information.
    #
    # They are not merely redundant, they are HARMFUL, and the cost was measured on the real
    # error-bearing residual (671 states): declaring them puts seven near-identical atoms (all firing
    # 335/336) into `keep[:12]`, which is the pool stage-3 conjunctions are drawn from, and the
    # backend x LENGTH pairs are then formed but truncated out of the 64 returned candidates.
    #
    #     declared (13 fields):   4 backend conjunctions, 0 of them over a length observable
    #     omitted  (10 fields):   5 backend conjunctions, 3 of them over a length observable
    #
    # So the alias crowds out the one class of conjunction that is NOT a restatement of the store's
    # name. Declare them only if a future store breaks the bijection -- at which point they carry
    # information that `backend` does not, and the trade reverses.
    # ---- HOW FAR INTO THE CONVERSATION THIS DECISION SITS ---------------------------------------
    #
    # Named for what it MEASURES (the turn index), never for a limit it might cross. Present on
    # 2443/2443 real states with real variance (max turn per episode 0..13, 308 episodes single-turn),
    # persisted by the trajectory sidecar all along and read only into teacher prose until now.
    #
    # This is the observable behind "some use cases are very long, hence need summarization" -- a
    # property of the DATA rather than a cell label, so it can transfer. The grammar derives any
    # threshold from this residual's own observed quantiles, so no length constant is written here.
    fields["turn"] = _Declared(
        name="turn", type=int, boundaries=_both,
        doc="0-based index of the conversation turn this decision point sits in")
    if boundary is None:
        return fields
    b = getattr(boundary, "value", str(boundary))
    declared_here = {n: f for n, f in fields.items()
                     if not getattr(f, "boundaries", ()) or b in f.boundaries}
    # DECLARED IS NOT ENOUGH: synthesis must only be offered fields the hook at this boundary
    # actually SUPPLIES. Measured -- 16 fields are declared at the commitment gate and the hook
    # supplies 5, so 11 of them cannot exist there. Offering them produced a predicate over
    # `searched_other_container` at that boundary: structurally valid, and inert at runtime.
    #
    # Rejecting the eventual winner is not a fix. A search whose space contains impossible points
    # wastes its budget on them and reports exhaustion it did not actually reach, so the constraint
    # belongs HERE, where the space is defined.
    #
    # A boundary with NO hook returns the declarations unchanged: an integration that supplies
    # nothing is not the same statement as a boundary whose fields are unknown, and silently
    # emptying the space would turn "not wired yet" into "nothing is observable here".
    supplied = hook_state_fields(b)
    if not supplied:
        # UNIMPLEMENTED FOR THIS HOST, which is NOT "exhausted". This host has no hook at
        # pre_generation, so nothing can be observed or acted on there however much is declared.
        # Returning the declarations would make the boundary look searchable and the eventual empty
        # result would read as SIGNAL_EXPANSION_EXHAUSTED -- reporting a search that never happened.
        # Returning {} makes core skip it, and `unimplemented_boundaries()` says why.
        return {}
    return {n: f for n, f in declared_here.items() if n in supplied}


class _Declared:
    """A minimal field declaration for observables the capability table does not already carry."""

    def __init__(self, name: str, type: type, doc: str = "", enum: tuple = (),
                 boundaries: tuple = ()):
        self.name, self.type, self.doc, self.enum, self.boundaries = name, type, doc, enum, boundaries


def _as_incision_point(b) -> "IncisionPoint | None":
    """Coerce an IncisionPoint, its `.value`, or a plain string to an IncisionPoint; else None.

    Tolerant by design: expansion may hand back any of the three, and a boundary we cannot recognise
    must degrade to "not recorded" rather than raising inside an installation that already validated.
    """
    if isinstance(b, IncisionPoint):
        return b
    try:
        return IncisionPoint(getattr(b, "value", b))
    except (ValueError, TypeError):
        return None


def install_signal(name: str, predicate, *, boundary=None, provenance: str = "") -> None:
    """Add a validated predicate to Phi. Refuses to shadow a shipped signal.

    `boundary` is the incision point expansion VALIDATED the predicate against, and it is now
    RETAINED. It used to be accepted and discarded, which left `signal_boundaries` with nothing to
    answer for an expanded signal -- so the observability question had no answer for exactly the
    signals expansion creates. Absent a boundary, the signal is recorded as observable nowhere,
    which is honest: it was never validated anywhere.
    """
    if name in SIGNALS:
        raise ValueError(f"{name!r} is a shipped signal; expansion may not redefine it")
    EXPANDED_SIGNALS[name] = predicate
    EXPANDED_PROVENANCE[name] = provenance or "(no provenance recorded)"
    pts = boundary if isinstance(boundary, (tuple, list, set, frozenset)) else \
        ((boundary,) if boundary is not None else ())
    EXPANDED_BOUNDARIES[name] = frozenset(
        p for p in (_as_incision_point(b) for b in pts) if p is not None)


def expanded_signals() -> tuple[str, ...]:
    """Signals added by expansion this session, distinguishable from the shipped ones."""
    return tuple(sorted(EXPANDED_SIGNALS))


def reset_expanded_signals() -> None:
    """Drop every expanded signal. Used between experiments so one round cannot inherit another's Phi."""
    EXPANDED_SIGNALS.clear()
    EXPANDED_PROVENANCE.clear()
    EXPANDED_BOUNDARIES.clear()


# ------------------------------------------------------------------------------------------------
# BOUNDARY IDENTIFICATION -- structural only; the ORDER comes from the trajectory
# ------------------------------------------------------------------------------------------------
#
# An earlier version declared STAGE_ORDER = ("read", "write") and a stage->locus map, and the scheduler
# took them as algorithm inputs. That was hand-engineering: the order was chosen by someone who already
# knew which failures mattered, and every new benchmark would have had to invent its own taxonomy before
# the optimizer could run. Removed. Core derives the order from the realized trajectory.
#
# What remains here is structural: which events are consequential decisions, and what to call the
# decision point. `label_for` is DESCRIPTIVE ONLY -- useful in a figure, never read by the search.
# Terminal statuses: the step where the agent stopped calling tools and committed to a final answer.
# Structural, from the trace's own status vocabulary -- no failure mode and no stage name is implied.
_ANSWER_COMMIT_STATUSES = frozenset({"answer_end_turn"})


def commits_to_answer(event: Mapping[str, Any]) -> bool:
    """Whether this event is the agent committing to a final answer rather than acting again."""
    if not isinstance(event, Mapping):
        return False
    return str(event.get("status") or "") in _ANSWER_COMMIT_STATUSES


def is_decision(event: Mapping[str, Any]) -> bool:
    """Whether this event is a point where the agent made a consequential choice.

    COMMITTING TO AN ANSWER IS A DECISION. This used to require a proposed or executed CALL, so an
    episode that answered without calling anything contributed no boundary at all -- and in the
    trajectories here that is 303 events per run (one per query), including every episode whose
    failure IS the commitment to answer unaided. Boundary derivation then yielded the single
    boundary `post_execution` for every residual, which no backward search can move earlier from.
    A decision point that the trace records but the adapter cannot name is indistinguishable from a
    trajectory that contains no such decision.
    """
    if not isinstance(event, Mapping):
        return False
    return bool(event.get("boundary")) or bool(event.get("has_generation")) or \
        bool(event.get("proposes_tool_call")) or bool(event.get("tool_calls")) or \
        bool(event.get("calls")) or bool(event.get("decoded")) or commits_to_answer(event)


def boundary_key(event: Mapping[str, Any]) -> str:
    """The decision point this event belongs to. Falls back to the declared boundary field."""
    b = event.get("boundary") if isinstance(event, Mapping) else None
    if b is not None:
        return getattr(b, "value", str(b))
    # An event with a generated call that has not been dispatched is at the commitment gate; one
    # carrying a result is past it. Both are facts about the event, not about any failure mode.
    if isinstance(event, Mapping):
        if event.get("tool_results") is not None or event.get("result") is not None:
            return IncisionPoint.POST_EXECUTION.value
        if event.get("has_generation") or event.get("decoded") or event.get("tool_calls"):
            return IncisionPoint.POST_GENERATION_PRE_EXEC.value
        # A commitment to answer is generated-but-undispatched: the answer can still be prevented,
        # which is the defining property of this incision point. Checked AFTER the two above so a
        # step that both answers and carries results is still placed by its results.
        if commits_to_answer(event):
            return IncisionPoint.POST_GENERATION_PRE_EXEC.value
    return ""


# Descriptive labels for analysis and figures. NOT an algorithm input: nothing in core reads these, and
# removing this map changes no search behaviour.
_DESCRIPTIVE_LABELS: Mapping[str, str] = {
    IncisionPoint.POST_GENERATION_PRE_EXEC.value: "commitment gate",
    IncisionPoint.POST_EXECUTION.value: "post-outcome",
    IncisionPoint.PRE_GENERATION.value: "pre-decision",
}


def actions_in(event: Mapping[str, Any]) -> tuple[str, ...]:
    """The actions this event took, in THIS trace format. Core asks; the adapter answers.

    Paired with `commits_to_answer`: together they let core ask the generic question "did the
    trajectory commit without acting first" without knowing that this runtime spells actions
    `decoded` and commitment `status == answer_end_turn`.
    """
    if not isinstance(event, Mapping):
        return ()
    return tuple(str(x) for x in (event.get("decoded") or []))


_PROPOSED_FACT_FIELDS = ("proposes_write", "proposes_read", "proposes_clear", "proposes_remove",
                         "container", "proposed_payload_chars", "proposed_arg_count")

# WHAT THE HOOK ACTUALLY SUPPLIES, per boundary. Distinct from `synthesis_fields`, which reports what
# this adapter DECLARES readable there. The two are allowed to differ -- a declaration is a statement
# about the runtime, a hook state is a fact about one integration -- but a controller may only be
# placed where its fields are actually supplied, so the difference has to be answerable.
#
# Measured on the live cluster evaluator: 5 declared fields at the commitment gate are populated in 0
# of 791 storage-phase step records. A search that trusted the declaration selected a controller that
# would have fired 0 times.
_HOOK_STATE_FIELDS: Mapping[str, frozenset[str]] = {
    IncisionPoint.POST_GENERATION_PRE_EXEC.value: frozenset({
        "proposed_call", "call_index", "phase", "step_index", "n_proposed_calls",
        # EPISODE-SCOPED, carried forward: the live hook accumulates it per episode and the offline
        # projection copies it, so a predicate over it validates and fires the same way. Listed here
        # because `synthesis_fields` intersects DECLARED with SUPPLIED -- a field the hook populates
        # but this set omits is invisible to synthesis, which is the mirror of the declared-but-
        # unsupplied gap and just as silent.
        "container_full",
        # EPISODE-STATIC STORE IDENTITY plus the turn index (R19). Constant for the whole episode
        # (backend) or strictly backward-looking (turn), so neither leaks anything about the call being
        # judged -- the discipline the gate's other fields follow. Structural properties are NOT here;
        # see synthesis_fields for the measured reason they are aliases and were dropped.
        "backend", "turn",
        # derived from the proposed call, pre-dispatch -- see proposed_call_facts
        *_PROPOSED_FACT_FIELDS}),
    IncisionPoint.POST_EXECUTION.value: frozenset({
        "best_similarity", "scored_entries", "searched_other_container", "query",
        # Whether the step proposed a READ. Backward-looking at this boundary (the call has already
        # run), and required by retrieval_similarity_below_threshold -- which was declared here and
        # could never fire, because nothing supplied this field.
        "proposes_read",
        # FIVE FIELDS THIS SET OMITTED WHILE EVERY REAL STATE CARRIED THEM (R19).
        #
        # Measured on all 2443 post-execution states of frozen H0 (3 backends x both phases): each of
        # these is present in 2443/2443, supplied by `observable_states`, and -- for the live side --
        # `error_kind` is patched into `_hook_state` by patches/bv/bv_supply_error_kind.py. Only THIS
        # declaration lagged, and it feeds TWO consumers, so the omission did not merely hide the
        # fields, it produced a confidently FALSE rejection:
        #
        #   synthesis_fields()        intersects declared x supplied -> the field is never OFFERED
        #   can_fire_at_boundary()    via boundary_state_shape -> hook_state_fields, REJECTS the
        #                             candidate with "the controller would fire 0 times"
        #
        # Measured verdicts before this line existed, against their ACTUAL firing counts:
        #
        #   error_kind == 'blob_would_overflow'            rejected, fires 301/2443
        #   error_kind == 'entry_too_long'                 rejected, fires 155/2443
        #   error_kind == 'no_capacity'                    rejected, fires 124/2443
        #   backend == 'kv' AND error_kind == 'no_capacity' rejected, fires  84/2443
        #
        # Six of the nine historical mechanisms act at this boundary, and `error_kind` is the field
        # that names WHICH refusal occurred -- the trigger class for all of them. With these five the
        # count of satisfiable declared signals here goes from 1 of 7 to 5 of 7; the remaining two
        # (`container_slots_exhausted`, `duplicate_identifier`) need `is_relocation_target` /
        # `identifier_present`, which are genuinely absent from states and are NOT added here.
        #
        # These are all POST-execution facts (this step's result exists), so the boundary discipline
        # holds: none of them is added to the commitment gate above, where the result does not yet
        # exist and where test_no_post_execution_fact_reaches_the_commitment_gate forbids them.
        "error_kind", "container", "proposes_write", "result", "tool_calls_so_far",
        # Same episode-static store identity and turn index as the gate above (R19), so a policy can
        # condition on the store it is acting against at the boundary where six of the nine historical
        # mechanisms act.
        "backend", "turn"}),
}


# TYPED FACTS DERIVED FROM A PROPOSED CALL, pre-dispatch by construction: this reads the call text and
# never its result. These are what makes the commitment gate searchable at all -- before them, synthesis
# was offered exactly one usable field there (step_index), so no write-side condition was expressible.
#
# Parsed with `ast`, not a regex. A regex over `text='...'` mis-measured the argument on real data and
# produced a length figure that was wrong on 17 of 161 calls; the structured parse is exact and reports
# nothing when it cannot parse rather than guessing.


#: EPISODE-STATIC facts a gate row inherits from the step it was projected from (R19).
#:
#: The gate projection REBUILDS each row rather than copying the step, which is deliberate -- it is how
#: post-execution facts are kept out of the pre-dispatch information set. But an episode-static fact
#: (which store this is, its structure, which turn we are on) is knowable before the call runs, so
#: dropping it would make a field DECLARED and SUPPLIED at this boundary yet absent from the states
#: synthesis validates on -- the `offered <= present` contract, broken silently.
_EPISODE_STATIC_FIELDS = ("backend", "turn")


def _carry_episode_static(row: dict, st: Mapping[str, Any]) -> dict:
    """Copy episode-static facts from a step onto a projected gate row, omitting what is absent.

    Absent stays absent: a missing `turn` must not become 0 (that asserts "the first turn" and feeds a
    false value into the quantiles the grammar derives thresholds from), and a missing backend must not
    become a guess.
    """
    for k in _EPISODE_STATIC_FIELDS:
        if st.get(k) is not None:
            row[k] = st[k]
    return row


def backend_facts(backend: str) -> Mapping[str, Any]:
    """Store identity plus the structural consequences a policy can depend on.

    DERIVED FROM THE ADAPTER'S OWN CONSTRAINT DECLARATIONS, never from a per-backend literal table.
    `bfcl_constraints` already records which quantity each store's refusal bounds, and that is exactly
    what the structure is: a store whose limit bounds CHARACTERS IN ONE AGGREGATE BLOB has no second
    container and no per-entry structure, while one whose limit bounds OCCUPIED SLOTS does. Writing
    `{"rec_sum": {...}}` here would hand the learner a lookup table and make every discovered
    "backend-specific" policy an artefact of this dict.

    So the only thing hardcoded is which constraint each store RAISES -- a fact about the host's
    schema, observable in its own error text -- and the properties follow from the declared quantity.

    Returns {} for an unknown backend rather than guessing: an unrecognised store yields no structural
    facts, and a predicate over an absent field simply does not fire, which is the correct semantics.
    """
    import bfcl_constraints as _bc
    b = str(backend or "").strip()
    if b not in _vocab.BACKENDS:
        return {}
    # Which declared constraint this store's writes are refused BY. One line per store, and it names
    # a constraint the adapter already declares -- not a remedy, not an action, not a threshold.
    raises = {"rec_sum": "blob_would_overflow",
              "kv": "no_capacity",
              "vector": "no_capacity"}.get(b, "")
    quantity = next((c.quantity for c in _bc.CONSTRAINTS if c.name == raises), "")
    aggregate = quantity == _bc.CHARS_IN_BLOB
    return {
        "backend": b,
        # An aggregate-bounded store is one blob: there is nowhere to relocate to and no entry to
        # shorten independently. A slot-bounded store has the second container relocation needs.
        "is_single_blob": aggregate,
        "has_second_container": not aggregate,
        # Uniqueness is a property of how the store ADDRESSES entries, which its tool signatures
        # record: a caller-supplied identifier can collide, an auto-assigned ordinal cannot, and a
        # blob has no identifier at all.
        "enforces_unique_keys": bool(_vocab.addresses_by_caller_key(b)),
    }


def proposed_call_facts(call: str) -> Mapping[str, Any]:
    """Structural facts about ONE proposed call, readable before it is dispatched.

    Returns {} when the call cannot be parsed -- an unparseable proposal yields no facts rather than
    facts that happen to be wrong.
    """
    import ast as _ast
    text = str(call or "").strip()
    if not text:
        return {}
    try:
        tree = _ast.parse(text, mode="eval")
        if not isinstance(tree.body, _ast.Call):
            return {}
        kwargs: dict[str, Any] = {}
        for kw in tree.body.keywords:
            if kw.arg is None:
                continue
            try:
                kwargs[kw.arg] = _ast.literal_eval(kw.value)
            except Exception:
                kwargs[kw.arg] = None
    except Exception:
        return {}
    strs = [v for v in kwargs.values() if isinstance(v, str)]
    return {
        "proposes_write": _vocab.is_write(text),
        "proposes_read": _vocab.is_read(text),
        "proposes_clear": _vocab.is_clear(text),
        "proposes_remove": _vocab.is_remove(text),
        "container": _vocab.container_of(text),
        # The largest string argument's length: the quantity a per-entry size limit is about. Named
        # for what it IS (a payload size) rather than for the limit it might violate, so the grammar
        # -- not this adapter -- decides whether and at what value it matters.
        "proposed_payload_chars": (max((len(s) for s in strs), default=0)),
        "proposed_arg_count": len(kwargs),
    }


def unimplemented_boundaries() -> Mapping[str, str]:
    """Boundaries this host DECLARES observables for but has no live hook at.

    Distinct from a boundary whose search was exhausted: nothing was ever searchable here. Kept as
    metadata so the declarations are not lost -- they describe the runtime, and a future integration
    can implement the hook without re-deriving them.
    """
    out = {}
    for point in IncisionPoint:
        b = point.value
        if hook_state_fields(b):
            continue
        from bfcl_capabilities import all_fields
        declared = [n for n, f in all_fields().items()
                    if not getattr(f, "boundaries", ()) or b in f.boundaries]
        if declared:
            out[b] = (f"no live hook at this boundary in this host; {len(declared)} observable(s) "
                      f"are declared but unreachable: {sorted(declared)[:4]}...")
    return out


_CALL_KEY_RE = re.compile(r"key\s*=\s*'([^']*)'")
_CALL_KEY_DQ = re.compile(r'key\s*=\s*"([^"]*)"')
_CALL_VAL_RE = re.compile(r"value\s*=\s*'([^']*)'")
_CALL_VAL_DQ = re.compile(r'value\s*=\s*"([^"]*)"')


def states_at(boundary, states):
    """Project observed states onto ONE boundary's information set. Called by core's WHAT search.

    The states core holds are assembled per STEP and carry post-execution facts. At the commitment
    gate that is the wrong information set: a predicate validated on a result that does not exist yet
    can look discriminating and then never fire. So this rebuilds, per PROPOSED CALL, only what is
    knowable before that call runs -- and returns nothing rather than something approximate when the
    input does not carry proposals, so core fails closed instead of validating on the wrong boundary.

    At post_execution the incoming states already ARE that boundary's information set, so they are
    returned unchanged and every existing result is unaffected.
    """
    want = str(getattr(boundary, "value", boundary))
    if want != IncisionPoint.POST_GENERATION_PRE_EXEC.value:
        return list(states or ())

    out = []
    for st in states or ():
        calls = [str(c) for c in (st.get("decoded") or [])]
        if not calls:
            # A STEP THAT PROPOSES NO CALL IS STILL A DECISION AT THIS BOUNDARY -- and it is the one
            # the answer-without-retrieval condition is about.
            #
            # `continue` here made the commitment gate structurally blind to it: 33 zero-tool-call
            # failing episodes projected to ZERO states, so `no_tool_call_at_all` -- the only declared
            # signal describing them -- could never be evaluated, and the fireability filter then
            # rejected every arm built for it as "fires on none of the observed states". True of those
            # states, and the reason there were none.
            #
            # The gate's question is "what is about to happen", and "a final answer with no tool call"
            # is a legitimate answer to it. So emit ONE row for the step, carrying the same
            # episode-scoped facts a call-bearing row gets, plus the three fields the zero-call signal
            # reads. `proposed_call` is None -- there is no call, and inventing one would be false.
            row = {"proposed_call": None, "call_index": 0, "n_proposed_calls": 0,
                   "step_index": st.get("step"), "boundary": want,
                   "phase": st.get("phase") or "query",
                   "container_full": bool(st.get("container_full") or False),
                   "prior_error_kind": st.get("prior_error_kind"),
                   # THE THREE FIELDS `no_tool_call_at_all` READS. A generation happened (the step
                   # exists in the trajectory), it proposes no tool call, and the count of calls made
                   # earlier in the episode is carried so "no call AT ALL" is distinguishable from
                   # "no call on this step after several earlier ones".
                   "has_generation": True,
                   "proposes_tool_call": False,
                   "tool_calls_so_far": int(st.get("tool_calls_so_far") or 0),
                   # Call-KIND facts are legitimately False: no call is proposed, so it proposes
                   # neither a read nor a write. These are answers, not absences.
                   "proposes_read": False, "proposes_write": False,
                   "proposes_clear": False, "proposes_remove": False,
                   # Nothing is proposed, so nothing proposed is redundant or previously refused.
                   "proposal_is_redundant": False, "proposal_refused_count": 0}
            # CALL-SHAPE MEASUREMENTS ARE ABSENT, NOT ZERO -- and the distinction is load-bearing.
            #
            # `proposed_payload_chars: 0` asserts "a call proposing 0 characters", which is false: no
            # call was proposed. Worse, it is a MEASUREMENT, so it enters the value distribution the
            # grammar synthesizes thresholds from. Measured: 376 injected zeros moved the quantiles far
            # enough that the archived 272-character threshold was no longer produced, and a guard
            # caught the historical controller becoming unreachable.
            #
            # Absent keeps the field's domain to real proposals. A predicate over an absent field
            # simply does not fire on this row, which is the correct semantics for "this row is not
            # about a proposed call's shape".
            _carry_episode_static(row, st)
            out.append(row)
            continue
        for i, call in enumerate(calls):
            row = {"proposed_call": call, "call_index": i, "n_proposed_calls": len(calls),
                   "step_index": st.get("step"), "boundary": want,
                   "phase": st.get("phase") or "query"}
            # EPISODE-SCOPED facts carried FORWARD to the commitment gate. A limit a store reported
            # earlier is still true when the next call is proposed, and the offline projection must
            # agree with the live hook field-for-field -- a divergence here is what makes a predicate
            # validate offline and fire zero times at runtime.
            #
            # ALWAYS PRESENT, defaulting to False. False here means "no limit has been OBSERVED in
            # this episode", which is a fact the boundary can state; leaving the key out would offer
            # synthesis a field some states do not carry, and a predicate over an absent field
            # silently answers False -- the same silent-miss this whole change exists to remove.
            row["container_full"] = bool(st.get("container_full") or False)
            # REDUNDANCY IS PER PROPOSED CALL, so it is computed HERE rather than carried from the
            # per-step state. The step-level flag is true if ANY of its calls repeats an earlier write;
            # attributing that to every call on the step would mark a genuinely new write redundant.
            # The episode-scoped (container, key) -> value map the assembler builds is passed through
            # as `store_before`, so each call is judged against what was stored BEFORE it.
            _seen = st.get("store_before") or {}
            _kk = (_CALL_KEY_RE.search(call) or _CALL_KEY_DQ.search(call))
            if _kk:
                _vv = (_CALL_VAL_RE.search(call) or _CALL_VAL_DQ.search(call))
                _cc = _vocab.container_of(call) or "unnamed"
                _pv = " ".join((_vv.group(1) if _vv else "").split())
                row["proposal_is_redundant"] = bool(
                    _pv and _seen.get((_cc, _kk.group(1))) == _pv)
            else:
                # No identifier in the call (text-only backend): nothing to compare, so not redundant.
                row["proposal_is_redundant"] = False
            # PER CALL: was THIS call already refused earlier in the episode? The assembler passes the
            # backward-looking map, so the row carries only what the live gate could know.
            row["proposal_refused_count"] = int(
                (st.get("refused_signatures") or {}).get(" ".join(call.split()), 0))
            # SAME THREE FIELDS ON THE CALL-BEARING ROWS, so the zero-call signal is EVALUABLE
            # everywhere rather than only where it fires. A predicate that can only be evaluated on
            # the states it fires on has no negative case, and the fireability check then cannot tell
            # "discriminates" from "unevaluable".
            row["has_generation"] = True
            row["proposes_tool_call"] = True
            row["tool_calls_so_far"] = int(st.get("tool_calls_so_far") or 0)
            # THE PRIOR STEP'S ERROR, UNDER ITS OWN NAME -- deliberately NOT `error_kind`.
            #
            # `error_kind` means "the result of THIS call", and this call has not run yet. Two guards
            # name it explicitly as a post-execution fact that may not appear here
            # (test_no_post_execution_fact_reaches_the_commitment_gate,
            # test_no_post_execution_fact_sneaks_in_with_it) and they are right: reusing the name
            # would let a predicate written for post-execution semantics silently read
            # pre-execution evidence, which is the offline/live divergence this projection exists to
            # prevent.
            #
            # So the episode-scoped, strictly backward-looking value is carried under a distinct name.
            # A pre-dispatch predicate that wants "an error has already been seen in this episode" can
            # read THIS; one that wants "this call failed" cannot be expressed here at all, which is
            # the honest answer. Always present; None means no error observed yet this episode.
            row["prior_error_kind"] = st.get("prior_error_kind")
            row.update(proposed_call_facts(call) or {})
            _carry_episode_static(row, st)
            out.append(row)
    return out


def hook_state_fields(locus) -> frozenset[str]:
    """Field names the host's hook populates at `locus`. Empty for a boundary with no hook."""
    return _HOOK_STATE_FIELDS.get(str(getattr(locus, "value", locus)), frozenset())


def boundary_from_key(key: str):
    """This adapter's own boundary key back to the incision point it denotes.

    The inverse of `boundary_key`, and it lives HERE for the same reason `boundary_key` does: core
    must not carry a key->locus table, because that table is the semantic stage map that was deleted.
    Core asks the adapter to convert its own vocabulary; an unrecognised key yields None and the
    boundary is simply not searchable.
    """
    try:
        return IncisionPoint(str(getattr(key, "value", key)))
    except (ValueError, TypeError):
        return None


def label_for(boundary_key_value: str) -> str:
    return _DESCRIPTIVE_LABELS.get(str(boundary_key_value), "")


def declared_signals() -> tuple[str, ...]:
    """Phi: the shipped signals plus any installed by expansion this session."""
    return tuple(sorted(set(SIGNALS) | set(EXPANDED_SIGNALS)))


def signal_aliases(signal: str) -> tuple[str, ...]:
    """Declared prose phrases describing `signal`, for expressibility matching only.

    Part of this benchmark's vocabulary, so the generic core never hard-codes a domain word -- it
    asks the runtime. Returns () for a signal with no declared aliases.
    """
    return tuple(SIGNAL_ALIASES.get(signal, ()))


def action_theta_schema() -> Mapping[str, Mapping[str, Any]]:
    """The theta keys each action REQUIRES, per `apply_action`. Declared so a proposer can be told.

    The live smoke run needed this. The proposer was given the fields, the actions and the tool
    schema, but never the PARAMETER NAMES -- so it wrote a thoughtful reprompt into
    `theta['guidance']` and the runtime, which requires `theta['text']`, declined all three
    proposals. Every decline was correct and none was informative about the proposal's substance.

    A contract the caller must satisfy but cannot read is a defect in the interface, not in the
    caller. This function is that contract, and it is generated from one place so it cannot drift
    from `apply_action`.
    """
    return {
        "reprompt": {"required": {"text": "str -- what the model must see"},
                     "optional": {}},
        "suppress": {"required": {"reason": "str -- why the call was cancelled"},
                     "optional": {"replacement": "the observation to return instead"}},
        "reroute": {"required": {"destination": "str -- a REAL tool name from the tool schema"},
                    "optional": {"args_patch": "dict of REAL argument names for that tool",
                                 "retry_original": "bool -- retry the original call afterwards",
                                 "reason": "str"}},
        "noop": {"required": {}, "optional": {}},
    }


def signal_param_schema() -> Mapping[str, Mapping[str, Any]]:
    """Per signal, the parameters it requires and their types -- the other half of the contract.

    Same reason as `action_theta_schema`: the smoke run's proposer supplied
    `signal_params={'threshold': 0.6}` for a signal whose declared parameter is `below`. It had no
    way to know the name, and guessing is not something the schema should reward or punish blindly.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, schema in SIGNAL_PARAMS.items():
        out[name] = {key: {"type": typ.__name__, "required": bool(req)}
                     for key, (typ, req) in schema.items()}
    return out


def parameter_domains(signal: str):
    """The searchable domain of each tunable parameter of `signal`, as `ParameterDomain` objects.

    Returns () for a signal with no tunable parameters -- which makes it a DETERMINISTIC policy, not
    a defective one. This is the seam the theta optimizer consumes; it is declared per parameter, so
    any signal with a range exposes it without the optimizer learning what the parameter means.
    """
    from anchoropt.learning.policy_class import ParameterDomain

    out = []
    for pname, spec in (SIGNAL_PARAM_DOMAINS.get(signal) or {}).items():
        out.append(ParameterDomain(
            name=pname, kind=spec.get("kind", float),
            low=spec.get("low"), high=spec.get("high"),
            values=tuple(spec.get("values", ())),
            observed_from=spec.get("observed_from", ""),
            monotone=spec.get("monotone", ""), doc=spec.get("doc", "")))
    return tuple(out)


def policy_class_for(signal: str):
    """PARAMETERIZED if the signal declares a tunable domain, else DETERMINISTIC.

    Derived rather than declared twice: a signal cannot disagree with itself about whether it has
    something to optimize.
    """
    from anchoropt.learning.policy_class import PolicyClass

    return (PolicyClass.PARAMETERIZED if parameter_domains(signal)
            else PolicyClass.DETERMINISTIC)


def probe_params(signal: str) -> Mapping[str, Any]:
    """Parameters the search may use to EVALUATE `signal` while enumerating candidates.

    Returns {} for a signal with no required parameters. These are measured values from the
    accepted stack, never invented ones -- see SIGNAL_PROBE_PARAMS. A signal needing a parameter
    this benchmark has not measured is deliberately left unconfigured, so the search prunes it with
    `signal_params_unconfigured` rather than firing it on a guess.
    """
    return dict(SIGNAL_PROBE_PARAMS.get(signal, {}))


def signal_fields(signal: str) -> frozenset[str]:
    """The STATE FIELDS `signal`'s predicate actually reads. Derived from source, never hand-listed.

    Core uses this to tell two different facts apart before it calls any arm inert:

        the predicate never fires here          real inertness -- no paired contrast exists
        this projection carries none of its     a SUPPLY GAP between signal and state record, which
        fields                                  says nothing about whether the condition ever holds

    Only the first justifies pruning. Without this distinction a declared signal whose evidence the
    adapter forgets to supply evaluates falsy everywhere and reads as inert -- the same error class as
    treating a raising predicate as all-False.

    It is DERIVED by reading `state.get(...)` / `state[...]` out of the predicate's own source, so it
    cannot drift from what the code reads. A hand-maintained list is exactly how `identifier_present`
    came to be read by two predicates and supplied by none.
    """
    import inspect as _inspect
    import re as _re

    fn = EXPANDED_SIGNALS.get(signal) or SIGNALS.get(signal)
    if fn is None:
        raise KeyError(f"{NAME}: no evaluator for signal {signal!r}")
    try:
        src = _inspect.getsource(fn)
    except (OSError, TypeError):
        # A synthesized predicate has no retrievable source. Returning empty makes core fail OPEN,
        # which is correct: an unknown field set is not evidence of a supply gap.
        return frozenset()
    fields = set(_re.findall(r"""state\.get\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""", src))
    fields |= set(_re.findall(r"""state\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*\]""", src))
    return frozenset(fields)


def unsupplied_signal_fields(boundary: str, states) -> dict[str, tuple[str, ...]]:
    """AUDIT: per declared signal at `boundary`, the fields its predicate reads and these states lack.

    A non-empty entry is a supply gap in THIS adapter -- the signal is declared observable here and
    the evidence it needs is absent, so it can only ever answer False. Run it before trusting a
    boundary's candidate set; it is how the `identifier_present` gap became visible.
    """
    have: set[str] = set()
    for st in states or ():
        have |= set(st)
    out: dict[str, tuple[str, ...]] = {}
    for sig in declared_signals():
        try:
            if not any(str(getattr(b, "value", b)) == str(boundary)
                       for b in signal_boundaries(sig)):
                continue
            need = signal_fields(sig)
        except Exception:
            continue
        missing = tuple(sorted(need - have))
        if need and missing:
            out[sig] = missing
    return out


def signal_boundaries(signal: str) -> frozenset[IncisionPoint]:
    """The incision points at which `signal` is observable. Unknown names RAISE."""
    # EXPANDED signals are dispatched first and separately. They carry no SIGNAL_BOUNDARIES entry and
    # no typed params -- validation already established that the predicate runs on this boundary's
    # state -- so falling through into the shipped path would raise on a boundary lookup that does not
    # exist for them, and an expanded signal would be undetectably unusable.
    if signal in EXPANDED_SIGNALS:
        # Return WHERE the predicate was validated -- not a truth value. This line used to read
        # `bool(EXPANDED_SIGNALS[signal](state or {}))`, copy-pasted from `evaluate_signal`, which
        # references a `state` this function does not take: every expanded signal raised NameError.
        # The only production caller (the proposer role) swallows exceptions and substitutes [], so
        # the proposer was silently told each synthesized signal is observable NOWHERE.
        return EXPANDED_BOUNDARIES.get(signal, frozenset())
    if signal not in SIGNALS:
        raise KeyError(f"{NAME}: no evaluator for signal {signal!r}; declared: "
                       f"{sorted(set(SIGNALS) | set(EXPANDED_SIGNALS))}")
    names = SIGNAL_BOUNDARIES.get(signal, frozenset())
    return frozenset(p for p in IncisionPoint if p.value in names)


def evaluate_signal(signal: str, state: Mapping[str, Any],
                    params: Mapping[str, Any] | None = None) -> bool:
    """Evaluate one Phi_BFCL signal. Unknown names RAISE rather than returning False.

    A missing evaluator that returns False is indistinguishable from a signal that did not fire,
    which is how an arm silently becomes the control arm. The TB2 prototype had exactly this defect
    (`tool_args` declared with no evaluator) and it crashed a run mid-cluster.

    Raises `KeyError` for an undeclared signal, `KeyError` for a signal not observable at the
    boundary named in `state`, `ValueError`/`TypeError` for bad params. `candidate_search` reads
    those three as distinct prune reasons, so they must stay distinguishable.
    """
    # EXPANDED signals are dispatched first and separately. They carry no SIGNAL_BOUNDARIES entry and
    # no typed params -- validation already established that the predicate runs on this boundary's
    # state -- so falling through into the shipped path would raise on a boundary lookup that does not
    # exist for them, and an expanded signal would be undetectably unusable.
    if signal in EXPANDED_SIGNALS:
        # Return WHERE the predicate was validated -- not a truth value. This line used to read
        # `bool(EXPANDED_SIGNALS[signal](state or {}))`, copy-pasted from `evaluate_signal`, which
        # references a `state` this function does not take: every expanded signal raised NameError.
        # The only production caller (the proposer role) swallows exceptions and substitutes [], so
        # the proposer was silently told each synthesized signal is observable NOWHERE.
        return EXPANDED_BOUNDARIES.get(signal, frozenset())
    if signal not in SIGNALS:
        raise KeyError(f"{NAME}: no evaluator for signal {signal!r}; declared: "
                       f"{sorted(set(SIGNALS) | set(EXPANDED_SIGNALS))}")

    boundary = state.get("boundary") if state else None
    if boundary is not None:
        value = boundary.value if isinstance(boundary, IncisionPoint) else str(boundary)
        allowed = SIGNAL_BOUNDARIES.get(signal, frozenset())
        if value not in allowed:
            # KeyError, not False: "cannot be observed here" is a STRUCTURAL fact about the
            # vocabulary and must prune the cell, whereas False means "observed, did not fire".
            # Conflating them is that measured defect.
            raise KeyError(
                f"{NAME}: signal {signal!r} is not observable at {value}; "
                f"declared observable at {sorted(allowed)}")

    params = dict(params or {})
    schema = SIGNAL_PARAMS.get(signal, {})
    for key, (typ, required) in schema.items():
        if key not in params:
            if required:
                raise ValueError(f"{NAME}: signal {signal!r} requires param {key!r}")
            continue
        val = params[key]
        if typ is float and isinstance(val, int) and not isinstance(val, bool):
            val = float(val)
            params[key] = val
        if typ in (int, float) and isinstance(val, bool):
            raise TypeError(f"{NAME}: signal {signal!r} param {key!r} must be "
                            f"{typ.__name__}, got bool")
        if not isinstance(val, typ):
            raise TypeError(f"{NAME}: signal {signal!r} param {key!r} must be {typ.__name__}, "
                            f"got {type(val).__name__}")
    unknown = set(params) - set(schema)
    if unknown:
        raise ValueError(f"{NAME}: signal {signal!r} unknown param(s) {sorted(unknown)}")
    return bool(SIGNALS[signal](state, params))


def register_signal(name: str, fn, params: Mapping[str, tuple[type, bool]] | None = None,
                    boundaries: Sequence[str] | None = None) -> None:
    """Register a signal accepted by the expansion step. Refuses to overwrite.

    Silently replacing a frozen definition would invalidate every measurement taken under the old
    one. Adding a signal touches NO action implementation.
    """
    if name in SIGNALS:
        raise ValueError(f"{NAME}: signal {name!r} already registered; frozen definitions must "
                         f"not be replaced")
    SIGNALS[name] = fn                                              # type: ignore[index]
    SIGNAL_PARAMS[name] = dict(params or {})                        # type: ignore[index]
    SIGNAL_BOUNDARIES[name] = frozenset(boundaries or ())           # type: ignore[index]


# ------------------------------------------------------------------------------------------------
# 4b. attested reroute destinations -- per SIGNAL, not per boundary
# ------------------------------------------------------------------------------------------------
#
# docs/GENERALIZABILITY.md states the rule this table exists to honour: "reroute must name a real
# destination tool with real argument names", and a destination must be attested on RESOLVING, not
# on not-erroring. a measured near-miss is the cautionary case -- a destination looked viable at 11/11
# "clean" where clean meant did-not-error, and 9 of those 11 returned nothing.
#
# So a reroute is only PROPOSABLE for a condition this benchmark has an attested destination for.
# Every entry below is the repair an accepted anchor actually performs (docs/ANCHORS.md); a
# condition absent from this table yields no REROUTE candidate at all, which is the honest state --
# not a reroute to an invented destination that would be misscored as "the substitute did not help".


def derive_reroute_destination(signal: str, boundary) -> Mapping[str, Any] | None:
    """DERIVE an executable reroute destination from this runtime's own tool capabilities.

    This is the honest replacement for a hardcoded table. It answers: given the CONDITION `signal`
    observes, does this runtime expose a tool that could supply what the condition says is missing?
    The derivation is structural -- it reads `tool_schema()`, matches on tool KIND and container, and
    returns None when nothing qualifies.

    NO INVENTED DESTINATIONS. docs/GENERALIZABILITY.md requires a reroute to name a real tool with
    real argument names, attested on RESOLVING rather than on not-erroring: a measured near-miss looked
    viable at 11/11 "clean" and 9 of those 11 returned nothing. A destination this function cannot
    derive is reported infeasible, never filled in with something plausible -- an invented
    destination gets misscored as "the substitute did not help", which is a measurement error
    dressed as a negative result.

    Returning None is therefore a FIRST-CLASS answer and it prunes the reroute cell.

    Note this is a CAPABILITY derivation, not an attestation. Whether the derived destination
    actually resolves is measured, and that measurement is the paired arm's job.
    """
    reads = tools_of_kind("read")
    # A read-side condition needs a read in a DIFFERENT container than the one that came back weak.
    # `container` on the failing call says which one that was; the alternative is the other.
    read_side = signal in {"retrieval_similarity_below_threshold", "identifier_not_found",
                           "no_informative_result"}
    if not read_side:
        return None
    core_reads = [t for t in reads if (tool_schema()[t].get("container") == "core")]
    arch_reads = [t for t in reads if (tool_schema()[t].get("container") == "archival")]
    if not arch_reads:
        return None
    # Prefer the archival read whose argument names MATCH a core read, so the query can be carried
    # over mechanically. docs/ANCHORS.md (an accepted anchor): a reroute is only deterministic if COPY/RENAME
    # suffices, and emitting `key=` universally built an invalid call.
    core_args = {a for t in core_reads for a in tool_schema()[t]["args"]}
    ranked = sorted(arch_reads,
                    key=lambda t: (-len(set(tool_schema()[t]["args"]) & core_args), t))
    dest = ranked[0]
    carried = sorted(set(tool_schema()[dest]["args"]) & core_args)
    if not carried:
        return None            # no argument can be carried mechanically -> not derivable
    return {"destination": dest, "carry_args": carried, "retry_original": False,
            "derivation": (f"read-side condition; {dest} reads the alternative container and shares "
                           f"argument(s) {carried} with the core read, so the query carries over "
                           f"mechanically")}



# ------------------------------------------------------------------------------------------------
# 4c. ACTION CONTRACT GROUNDING -- eta_mu per action family, enumerated from capabilities
# ------------------------------------------------------------------------------------------------
#
# One hook per canonical operator. Each returns a LIST of groundings; every element becomes a
# separate evaluation arm. Returning [] means the family is INFEASIBLE here, and the optimizer
# reports the missing requirement -- it never invents a tool, an argument or a surface, and it never
# downgrades an ungroundable substitute into a reprompt.

# ------------------------------------------------------------------------------------------------
# EXECUTOR REGISTRY -- what this host can ACTUALLY run, per (boundary, action)
# ------------------------------------------------------------------------------------------------
#
# U_H(l) declares which actions are admissible at a boundary. That is necessary and NOT sufficient:
# an action can be admissible while the host has no executor able to realize it, and a candidate
# reported "grounded" on the strength of the abstract declaration alone is not materializable. The
# targeted recovery experiment found exactly that -- three REPROMPT arms passed every abstract check
# and one of them declared a retry budget the executor cannot honour.
#
# So feasibility is derived from an EXECUTOR, and each entry names:
#   remedy_flag   the policy switch that arms it
#   eta_slot      where the action parameter is read from
#   trigger       the runtime condition it fires on
#   fixed         parameter values the executor imposes; a candidate declaring otherwise is REJECTED
#
# These executors are GENERIC over the signal: the post-generation reprompt window below is shared by
# more than one signal in this evaluator (a zero-tool-call condition and a low-similarity condition
# use the same window and the same injection mechanism), which is what makes it a boundary/action
# executor rather than a per-anchor gate.
EXECUTORS: Mapping[tuple[str, str], Mapping[str, Any]] = {
    ("post_generation_pre_exec", "reprompt"): {
        "remedy_flag": "enable_zero_call_reprompt",
        "eta_slot": "templates.on_turn_start_action",
        "trigger": "signal evaluated on the post-generation state, inside the empty-decode branch",
        "fixed": {"retry_budget": 1},
        "signals": ("no_tool_call_at_all",),
        "detail": ("intercepts a pending final-answer decision, injects the selected instruction as "
                   "a trailing user message, and regenerates once"),
    },
    ("post_execution", "reprompt"): {
        "remedy_flag": "enable_low_similarity_reprompt",
        "eta_slot": "templates.on_low_similarity_reprompt",
        "trigger": "signal evaluated on the returned result",
        "fixed": {"retry_budget": 1},
        "signals": ("retrieval_similarity_below_threshold", "identifier_not_found"),
        "detail": "injects after a weak or failed read and regenerates once",
        "telemetry_flag": "low_similarity_reprompt_gate",
    },
    # --------------------------------------------------------------------------------------------
    # REGISTERED FROM THE EVALUATOR'S ACTUAL FIRING SITES (anchoropt/memory_evaluator.py on the
    # cluster; the vendored copy under benchmarks/ mirrors it). Nothing here is new capability -- each
    # of these remedies already runs; they were simply undeclared, so `executor_supports()` pruned
    # most of the accepted stack's cells before evaluation, so a recovery study would have reported
    # a low rate for a REGISTRY reason rather than a search reason. Line numbers are the cluster copy at the time of writing.
    #
    # Every field below was read off the code, not inferred: `remedy_flag`/`gate` from the actual
    # condition, `telemetry_flag` from the `step_record[...]` assignment at the firing site, and
    # `also_requires` where the site demands something beyond the policy dict.
    # --------------------------------------------------------------------------------------------
    ("post_execution", "reroute"): {
        # Fires on the official core-full tool error, then re-dispatches a synthesized
        # archival_memory_add instead of only reprompting. memory_evaluator.py:5159-5195.
        "remedy_flag": "enable_reroute",
        "gate": "on_domain_error_core_full",
        "eta_slot": "templates.on_domain_error_core_full",
        "trigger": ("post-execution tool error matching the registry spec for on_domain_error_core_full "
                    "('is full' / 'exceeds maximum size'); once per turn, kv+vector only"),
        "fixed": {},
        "signals": ("container_at_capacity", "container_slots_exhausted",
                    "append_would_exceed_cap", "retrieval_similarity_below_threshold",
                    "identifier_not_found", "duplicate_identifier"),
        "detail": ("synthesizes and dispatches a write to the other container; falls back to the "
                   "reprompt text when the reroute cannot be synthesized"),
        "telemetry_flag": "g3_gate",
        # THE READ-MERGE PATH IMPOSES ITS OWN SCORE THRESHOLD, a module constant in the evaluator.
        # It was not declared here, so `executor_supports` had nothing to compare and reported a
        # candidate carrying a DIFFERENT threshold as materializable -- the same class as the
        # retry_budget conflict that correctly rejected an earlier candidate, but invisible because the
        # value was never surfaced. Declaring it makes the contract able to refuse the coercion instead
        # of measuring a trigger nobody proposed.
        "fixed": {"merge_score_threshold": 0.30},
        # The mechanism synthesizes a destination write from LIVE state, so it does not read the
        # signal's value -- an expanded condition can drive it. The reprompt cells are NOT marked
        # agnostic: their eta is a template slot chosen per signal.
        "signal_agnostic": True,
        # REROUTE has two operator readings (substitute / transform) and this host runs BOTH, through
        # different remedies. They are recorded HERE rather than as top-level ("post_execution",
        # "substitute"/"transform") keys because `executor_for` keys on Action.value and the Action
        # enum has no such members -- those keys would be unreachable dead weight.
        "operators": {
            "substitute": {
                # Replaces the failing call with a mechanically validated substitute.
                # memory_evaluator.py:5307-5325.
                "remedy_flag": "enable_capacity_repair",
                "signals": ("identifier_not_found", "append_would_exceed_cap",
                            "container_at_capacity"),
                "telemetry_flag": "capacity_repair_gate",
                "eta_is_computed": True,
                "detail": ("constructs a validated replacement call and re-dispatches it; the eta is "
                           "DERIVED from live state, so a candidate supplying its own "
                           "argument_mapping is not what this executor runs"),
            },
            # READ-SIDE MERGE IS ALREADY THIS OPERATOR. `ground_transforms` produces
            # merge_additional_read -- an extra read whose results are merged into the returned
            # observation, original retained, union ranked by score. I briefly registered a separate
            # `transform_read_merge` operator for it; that was redundant, and worse, a runtime-only
            # operator name can never become an arm because `action_contract.instantiate` enumerates
            # operators from the Operator ENUM, not from this dict.
            #
            # THE DESIGN PRINCIPLE THE ROUND EXPOSED STILL HOLDS: signal expansion is useless unless a
            # compatible primitive exists here. What was actually missing was not the primitive but the
            # SIGNAL-SIDE plumbing -- grounding classified read/write by fixed signal-name sets, so a
            # synthesized name grounded nothing and a controller on it could be built and never engage.
            "transform": {
                # Evicts one redundant archival copy, verifies a copy remains, retries VERBATIM.
                # memory_evaluator.py:4775-4800.
                "remedy_flag": "enable_archival_evict_duplicate",
                "gate": "on_archival_full_evict_duplicate",
                "signals": ("container_slots_exhausted", "duplicate_identifier"),
                "telemetry_flag": "e1_evict_gate",
                "eta_is_computed": True,
                "detail": "evicts one duplicate, verifies a copy remains, retries the write verbatim",
            },
        },
        "note": ("BOTH the gate and the remedy flag are required -- gate_enabled(on_domain_error_"
                 "core_full) AND remedy_enabled(enable_reroute). With only the gate it reprompts "
                 "instead of rerouting, which is a different intervention."),
    },
    ("post_generation_pre_exec", "suppress"): {
        # Filters a proposed call OUT of `decoded` before dispatch: one variant drops a redundant
        # write, the other drops a destructive clear and then evicts one duplicate and retries the
        # blocked write verbatim. memory_evaluator.py:3697-3860.
        "remedy_flag": "enable_redundant_write_suppress",
        "gate": "on_redundant_write_suppressed",
        "eta_slot": "(predicate-driven; no template slot)",
        "trigger": "a proposed call matching the suppression predicate, before dispatch",
        "fixed": {},
        "signals": ("duplicate_identifier", "clear_proposed_at_capacity"),
        "detail": ("removes the proposed call before it commits; for clear_proposed_at_capacity the "
                   "gate on_dedup_clear_recovery additionally evicts one duplicate and retries the "
                   "blocked write verbatim"),
        "telemetry_flag": "redundant_write_gate",
        "signal_agnostic": True,
        "variants": {"clear_proposed_at_capacity": {"gate": "on_dedup_clear_recovery",
                                                    "telemetry_flag": "dedup_clear_gate"}},
        "eta_is_computed": True,
    },
}


# ================================================================================================
# EXECUTOR CAPABILITY -- the BOUND form of the table above
# ================================================================================================
#
# WHY THIS EXISTS ALONGSIDE `EXECUTORS`. The dict above is a DECLARATION; it was accepted as proof
# that executing code existed, and for one cell it was not. `("post_generation_pre_exec","suppress")`
# named `remedy_flag="enable_redundant_write_suppress"` -- a flag that appears ZERO times in the hook
# that actually withholds the call. The hook (patches/upstream_preexec_hook_and_phase_scope.diff)
# gates on `anchoropt.runtime_hook.installed("post_generation_pre_exec")`, per-controller
# `_phase_eligible`, and `controller.fires_on(state)`. The flag was never consulted.
#
# So each cell below names:
#   binding   the code that RUNS, specific enough to grep for
#   consumes  the eta keys that code actually READS -- read off the source, not the declaration
#
# `consumes` is the field that closes inert eta. The upstream hook reads exactly two keys from a
# controller's eta: `retry_budget` (the withhold budget, keyed per episode) and `instruction` (which
# it writes to `step_record` and NOTHING ELSE -- see the disabled reprompt cell). It does not read
# `suppressed_operation` or `preservation`, so neither may be claimed as enforced.
_CAPABILITIES: tuple[ExecutorCapability, ...] = (
    ExecutorCapability(
        boundary="post_generation_pre_exec", action="suppress",
        # VERIFIED by reading the patch: this is the branch that removes the call from the dispatch
        # list. `_mg_exec_calls = [c for _i, c in _up_keep]` IS the suppression, and it is reached
        # only when an installed, phase-eligible controller's `fires_on` returns True.
        binding=("memory_evaluator.py upstream post_generation_pre_exec hook "
                 "(runtime_hook.installed + _phase_eligible + fires_on -> "
                 "_mg_exec_calls = [c for _i, c in _up_keep])"),
        # THE HOOK READS ONLY THIS. `suppressed_operation` and `preservation` are NOT here because the
        # hook never reads them -- which is exactly why `preservation` is no longer contract-required
        # and may not be claimed at this cell.
        consumes=("retry_budget",),
        # `redundant_proposed_write` added because this executor is SIGNAL_AGNOSTIC (declared below):
        # its hook builds the kept list from `_up_keep`, which is whatever the installed predicate did
        # not select -- it never reads the signal's name. So the list is a record of which conditions
        # have been SHOWN to work here, and a newly declared condition observable at this same boundary
        # belongs in it once its field supply is verified. That verification: the signal fires 408/1143
        # on real kv commitment-gate states with 735 negatives, and 0 of 39 same-key/different-value
        # UPDATES are reported redundant.
        #
        # This is a DECLARATION correction, not a new promise: nothing about the executing code changes,
        # and no anchor policy flag is read. Without it, grounding rejects the cell
        # `executor_signal_unsupported` -- correct under the contract, and the wrong answer here.
        signals=("duplicate_identifier", "clear_proposed_at_capacity",
                 "redundant_proposed_write", "proposal_already_refused"),
        fixed={},
        signal_agnostic=True,
        eta_is_computed=True,
        detail=("removes the proposed call from the dispatch list before self._execute; the model "
                "observes nothing in its place and proposes again next step. On budget "
                "exhaustion the withheld call DISPATCHES -- the arm degrades to control."),
    ),
    # THE WORKING INJECTOR AT THIS CELL. Declared BEFORE its inert sibling so a deterministic
    # first-match resolves to the executor that actually intervenes.
    #
    # This is not a new capability: the code has always run, and an earlier round measured a real
    # paired effect through it. It became unreachable when the cell's OTHER executor was correctly
    # disabled -- a single-capability lookup let the inert one mask this one, so a live mechanism was
    # pruned for a bookkeeping reason rather than a validity one.
    #
    # Read off the firing site, not inferred: the branch appends the selected instruction as a
    # trailing user message via `_add_next_turn_user_message_prompting` and then `continue`s, which
    # regenerates. That is an injection, and it is what distinguishes this cell from its sibling.
    ExecutorCapability(
        boundary="post_generation_pre_exec", action="reprompt",
        capability_id="inject_trailing_message_and_regenerate",
        binding=("memory_evaluator.py zero-decode branch: remedy gate -> "
                 "_add_next_turn_user_message_prompting(templates.on_turn_start_action) -> continue "
                 "(regenerates once, bounded by a once-per-turn latch)"),
        # CONSUMED VIA THE POLICY TEMPLATE, not from the candidate dict. That is a legitimate delivery
        # channel -- an earlier round measured a real paired effect through it, and varying the text
        # genuinely changes the trajectory -- but it has to be DECLARED, or two arms differing only in
        # `instruction` execute identically unless the runner happens to write it there.
        # BOTH are consumed, by DIFFERENT channels, and the declaration says which:
        #   instruction   -> delivered through the policy template the executor reads
        #   retry_budget  -> honoured by the once-per-turn latch, hence also `fixed` below
        consumes=("instruction", "retry_budget"),
        eta_delivered_via={"instruction": "templates.on_turn_start_action",
                           "retry_budget": "_zero_call_reprompted latch (fixed at 1)"},
        # THE LATCH FIXES THE BUDGET AT 1. `_zero_call_reprompted` allows exactly one regeneration per
        # turn, so `fixed` is the honest construct here rather than `unsupported_eta`: a candidate
        # asking for 1 is running exactly what it proposed, and one asking for anything else is
        # REFUSED with a PARAM_CONFLICT rather than silently coerced.
        #
        # Marking it unsupported outright was tried and was wrong -- the grounding emits
        # retry_budget=1, which is what the executor enforces, so refusing it removed a working
        # mechanism over a bookkeeping mismatch.
        fixed={"retry_budget": 1},
        signals=("no_tool_call_at_all",),
        detail=("intercepts a pending final answer, appends the candidate instruction as a user "
                "message, and regenerates once"),
    ),
    ExecutorCapability(
        boundary="post_generation_pre_exec", action="reprompt",
        capability_id="write_instruction_to_step_record",
        binding=("memory_evaluator.py upstream hook -- step_record['upstream_instruction'] "
                 "assignment ONLY"),
        consumes=(),
        signals=("no_tool_call_at_all",),
        # D3. THE ACTION IS INERT AND MUST NOT BE MEASURED.
        #
        # The hook assigns the instruction to `step_record["upstream_instruction"]` and
        # `step_record["upstream_reprompt_gate"]`. It never appends a message, never regenerates, and
        # never reaches `_mg_exec_calls`. Writing telemetry is not injecting an instruction.
        #
        # PROVEN, not inferred: two arms differing ONLY in `instruction` produced byte-identical
        # trajectories, 13/13 storage episodes. That is the strongest available evidence that the
        # field does not reach the agent -- if it did, the decode would differ.
        #
        # Disabled rather than deleted, because deleting it would make the cell look unconsidered and
        # a future porter would re-add it. Re-enable ONLY when the instruction demonstrably reaches
        # the model: a test showing two instruction variants produce DIFFERENT trajectories.
        disabled_reason=("the instruction is written to step_record and never injected into the "
                         "conversation -- two arms differing only in `instruction` were "
                         "byte-identical (13/13 episodes), so measuring this cell would measure the "
                         "control arm under a reprompt label"),
        detail="NOT materializable: telemetry-only. See D3 in docs/EXECUTOR_CONTRACT_NOTES.md.",
    ),
    ExecutorCapability(
        boundary="post_execution", action="reprompt",
        binding="memory_evaluator.py low_similarity_reprompt_gate firing site",
        consumes=("instruction", "retry_budget"),
        signals=("retrieval_similarity_below_threshold", "identifier_not_found"),
        fixed={"retry_budget": 1},
        detail="injects after a weak or failed read and regenerates once",
    ),
    # THE INFORMATION-PRESERVING TRANSFORM. Declared because the executing code exists, is generic,
    # and was invisible to the search purely for want of a declaration.
    #
    # Read off the implementation: `_try_capacity_repair` finds the call whose result reported this
    # backend's capacity condition, computes a character budget from the schema, builds a reduced
    # payload that PRESERVES the facts, validates it mechanically, and dispatches the replacement. Its
    # own docstring states the division of labour this relies on: "GENERIC. Everything backend-specific
    # arrives through `capacity_repair.BACKENDS` ... this method contributes control flow only -- it
    # never names a tool, a limit, or a backend."
    #
    # WHY A TRANSFORM AND NOT A SUPPRESSION. Two measured suppression arms at the commitment gate were
    # net-negative for the same structural reason: withholding a call DIVERTS the trajectory rather than
    # recovering the information the call was meant to store. This operation RECOVERS it -- the fact
    # still reaches the store, in a form that fits.
    #
    # eta is COMPUTED from live state (budget and reduced text are derived, not proposed), so
    # `consumes=()` is a positive claim, the same shape the reroute cell below declares.
    ExecutorCapability(
        # FILED UNDER REROUTE, and the reason is in core's own enum comment: there are four Action
        # values, and REROUTE covers BOTH `substitute` (replace the proposed call) and `transform`
        # (reshape what a later decision reads). `Action.TRANSFORM` does not exist, and
        # `feasible_actions(POST_EXECUTION)` is {noop, reroute, reprompt} -- so a capability declared
        # at a non-existent action is unreachable however well it is implemented. That is what the
        # first attempt did.
        #
        # This is a SECOND executor at the reroute cell, beside the additional-read-and-merge one,
        # which is exactly why capability identity matters here: the two do different things and an
        # arm must name which it means.
        boundary="post_execution", action="reroute",
        capability_id="reduce_payload_preserving_facts_and_replace",
        # A TRANSFORM: it reshapes the payload a later decision reads. Declared so a `substitute` arm
        # at this same cell cannot resolve to it.
        operators=("transform",),
        binding=("memory_evaluator.py _try_capacity_repair -> scripts/capacity_repair.repair(); "
                 "gated today by the enable_capacity_repair policy flag, which is why an "
                 "AnchorOpt-installed controller could not reach it"),
        consumes=(),
        # CHARACTER-BOUND SIGNALS ONLY. This executor reduces payload LENGTH, so it can relieve a
        # constraint counted in characters and NOT one counted in entries. `container_at_capacity` and
        # `container_slots_exhausted` were declared here and are now removed: on kv that refusal is
        # "exceeds maximum size of N entries", where a shorter value frees ZERO slots. Grounding on
        # them produced an arm that was refuted at feasibility, and the overclaim is what let it
        # ground. See bfcl_constraints.py and anchoropt/learning/constraint_feasibility.py.
        signals=("append_would_exceed_cap", "entry_exceeds_len_limit"),
        signal_agnostic=True,
        eta_is_computed=True,
        detail=("replaces an over-cap payload with a fact-preserving reduction sized to the schema's "
                "limit and dispatches it, so the write SUCCEEDS instead of being prevented; "
                "feasibility measured on 352 live captures -- 352/352 replacements constructed, "
                "median 1% compression, 99.9% of query-relevant tokens preserved"),
    ),
    ExecutorCapability(
        boundary="post_execution", action="reroute",
        capability_id="relocate_entry_preserving_information_then_retry",
        # Also a TRANSFORM: it changes the container's occupancy, not which call runs.
        operators=("transform",),
        binding=("memory_evaluator.py _try_capacity_relocate -> scripts/capacity_relocate."
                 "relocate_and_retry(); reached by an AnchorOpt-installed controller whose spec names "
                 "THIS capability_id"),
        # SLOT-BOUND SIGNALS ONLY, the mirror image of the reduction executor above. This one frees an
        # ENTRY; it does not shorten anything. Declaring the character signals here would be the same
        # overclaim in the other direction.
        signals=("container_at_capacity", "container_slots_exhausted"),
        consumes=(),
        signal_agnostic=True,
        eta_is_computed=True,
        detail=("frees ONE slot in the constrained store by relocating an entry to a strictly more "
                "permissive container (core 7 entries/300 chars -> archival 50/2000), VERIFYING the "
                "destination write against live state BEFORE removing the source, then retrying the "
                "refused call verbatim; bounded at 3 per episode. Measured premise: at all 218 recorded "
                "core-full refusals archival held <=1 of 50, so a destination slot existed every time, "
                "and 208 of those 218 were followed by a wholesale core_memory_clear -- the harm this "
                "recovers. If the destination write does not land, nothing is removed."),
    ),
    ExecutorCapability(
        boundary="post_execution", action="reroute",
        binding="memory_evaluator.py:5159-5195 on_domain_error_core_full + enable_reroute",
        # The mechanism synthesizes the destination write from LIVE state, so it consumes nothing the
        # candidate supplies. Declared explicitly rather than left blank: an empty `consumes` is a
        # POSITIVE claim that this executor reads no candidate eta, and it is what allows the
        # substitute/transform contracts' computed-eta groundings through.
        consumes=(),
        signals=("container_at_capacity", "container_slots_exhausted", "append_would_exceed_cap",
                 "retrieval_similarity_below_threshold", "identifier_not_found",
                 "duplicate_identifier"),
        fixed={"merge_score_threshold": 0.30},
        signal_agnostic=True,
        eta_is_computed=True,
        # WHAT THE LIVE HOOK WILL NOT ACT WITHOUT. Its first statement is
        #     if _prim != "additional_read_and_merge": return {"fired": False, ...}
        # so a candidate that carries no `primitive` reports not-fired and runs as the CONTROL. This
        # declaration is what turns that silent no-op into a refusal with a reason at construction.
        #
        # `retry_semantics` is listed because the two grounded variants -- replace_original and
        # retry_after -- are DIFFERENT interventions, and an executor that ignores the field makes
        # them byte-identical arms. Declaring it required means core refuses them until an executor
        # actually reads it, rather than measuring a 2x eta sweep that cannot differ in outcome.
        requires_eta={"primitive": ("additional_read_and_merge",),
                      "retry_semantics": ("replace_original", "retry_after")},
        detail=("synthesizes and dispatches a write to the other container from live state; requires "
                "BOTH gate_enabled(on_domain_error_core_full) AND remedy_enabled(enable_reroute), "
                "and eta.primitive naming a primitive this host implements"),
    ),
)

# A cell may declare MORE THAN ONE executor, so the index maps to a TUPLE in declaration order.
# It was a plain dict keyed on the cell, which silently kept only the last entry -- and at
# post_generation_pre_exec/reprompt that was the inert telemetry-only executor, masking the
# injector beside it.
_CAPABILITY_INDEX: dict[tuple[str, str], tuple[ExecutorCapability, ...]] = {}
for _c in _CAPABILITIES:
    _CAPABILITY_INDEX.setdefault((_c.boundary, _c.action), ())
    _CAPABILITY_INDEX[(_c.boundary, _c.action)] += (_c,)


def executor_capabilities(boundary, action) -> tuple[ExecutorCapability, ...]:
    """EVERY bound executor for (boundary, action), in declaration order.

    The hook core prefers. A cell is not one mechanism: this one names both an inject-and-regenerate
    executor and an inert telemetry-only one, and core must be able to reject the second WITHOUT
    losing the first. Core resolves which capability an arm is feasible against and carries its id
    into the arm, so the runner installs the executor core validated.
    """
    return _CAPABILITY_INDEX.get(
        (getattr(boundary, "value", str(boundary)), getattr(action, "value", str(action))), ())


def executor_capability(boundary, action):
    """The FIRST bound-and-enabled executor for this cell, or the first declared, or None.

    Kept for callers (and adapters) written against the singular form. Prefers an enabled capability
    so a legacy caller is not handed the disabled sibling, which is exactly the masking this hook's
    plural replacement exists to fix.
    """
    caps = executor_capabilities(boundary, action)
    for c in caps:
        if c.is_bound and c.is_enabled:
            return c
    return caps[0] if caps else None


def expanded_signal_names() -> tuple[str, ...]:
    """Signals installed by expansion. Core needs these to decide if one may reach an agnostic cell."""
    return tuple(EXPANDED_SIGNALS)


def capability_audit() -> dict:
    """Ghost and disabled cells, for a porter and for tests. The audit that would have caught D1."""
    return {"capabilities_per_cell": {f"{b}/{a}": [c.cid for c in caps]
                                     for (b, a), caps in sorted(_CAPABILITY_INDEX.items())},
            "unbound": list(unbound_capabilities(_CAPABILITIES)),
            "disabled": list(disabled_capabilities(_CAPABILITIES)),
            "bound_and_enabled": sorted(f"{c.boundary}/{c.action}" for c in _CAPABILITIES
                                        if c.is_bound and c.is_enabled)}

# Some executors additionally require an environment switch OUTSIDE the policy dict. A candidate that
# looks materializable but whose switch is unset runs as the CONTROL -- the silent-null defect class.
# The REQUIREMENT is declared here; the specific switch names are NOT, because naming one of them in
# this module would ship a per-anchor identifier into the learner-visible runtime (see
# tests/test_action_contract.py, which forbids exactly that). The run harness owns the mapping and is
# responsible for asserting it before submitting an arm.
ENV_GATED_CELLS: tuple[tuple[str, str], ...] = (
    ("post_execution", "reroute"),
)


def env_gated(boundary, action) -> bool:
    """Whether this cell needs an environment switch the policy dict cannot express."""
    key = (getattr(boundary, "value", str(boundary)), getattr(action, "value", str(action)))
    return key in ENV_GATED_CELLS


def executor_for(boundary, action) -> Mapping[str, Any] | None:
    """The executor that can realize (boundary, action) on this host, or None.

    None means NOT MATERIALIZABLE, whatever `U_H(l)` says about admissibility, and it prunes the
    candidate before evaluation rather than after.
    """
    key = (getattr(boundary, "value", str(boundary)), getattr(action, "value", str(action)))
    spec = EXECUTORS.get(key)
    return dict(spec) if spec else None


def executor_supports(boundary, action, signal: str, eta: Mapping[str, Any]
                      ) -> tuple[bool, str]:
    """Can the host's executor for (boundary, action) consume `eta` for `signal`?

    Five conditions, matching the feasibility rule this host is held to:
      1. an executor exists for (boundary, action)
      2. the executor covers this signal
      3/4. every parameter the executor FIXES matches what the candidate declares -- a candidate
           asking for a retry budget the executor cannot honour is rejected, not silently coerced
      5. the caller replays a qualifying state (see `replay_fires`), which this function does not do

    Silently coercing a mismatched parameter is the failure mode being prevented: the arm would run
    with different semantics than the ones proposed, under the proposal's name.
    """
    spec = executor_for(boundary, action)
    if spec is None:
        return False, (f"no_executor: this host has no executor for "
                       f"{getattr(boundary, 'value', boundary)}/{getattr(action, 'value', action)}, "
                       f"so the action is admissible but not materializable")
    covered = tuple(spec.get("signals") or ())
    if covered and signal not in covered:
        # AN EXPANDED SIGNAL CAN BE COVERED TOO. The `signals` tuple lists the SHIPPED conditions this
        # executor was declared for; a signal installed by expansion is not in it and would be refused
        # forever, so expansion and materializability would never compose -- Phi could grow while every
        # new cell stayed unreachable, which is a silent dead end rather than a reported one.
        #
        # An expanded signal is accepted at this cell only if the executor's mechanism does not depend
        # on WHICH condition triggered it. That is true here because these remedies read live state
        # rather than the signal's own value (see `eta_is_computed`), and it is asserted rather than
        # assumed: an executor whose eta comes from a template slot keyed on the signal is NOT
        # signal-agnostic and keeps the whitelist.
        if signal in EXPANDED_SIGNALS and spec.get("signal_agnostic"):
            # FALL THROUGH to the fixed-parameter check below -- do NOT return here. Returning early
            # exempted every expanded signal from parameter validation, which is the one population that
            # most needs it: a synthesized predicate carries a data-derived threshold, and if the
            # executor imposes its own the arm measures a trigger nobody proposed.
            expanded_ok = True
        else:
            return False, (f"executor_signal_unsupported: {spec['remedy_flag']} evaluates "
                           f"{list(covered)}, not {signal!r}")
    else:
        expanded_ok = False
    for key, want in dict(spec.get("fixed") or {}).items():
        if key in eta and eta[key] != want:
            return False, (f"executor_parameter_conflict: {spec['remedy_flag']} fixes {key}={want} "
                           f"but this eta declares {key}={eta[key]!r}; coercing it would run "
                           f"different semantics than the ones proposed")
    if expanded_ok:
        return True, (f"materializable via {spec['remedy_flag']} with the EXPANDED signal "
                      f"{signal!r} (mechanism reads live state, not the signal value)")
    return True, f"materializable via {spec['remedy_flag']}"


# ================================================================================================
# OPTIONAL APPLICABILITY: a grounded intervention may be offered CONDITIONED on the store it runs on
# ================================================================================================
#
# WHY THIS IS A GROUNDING CONCERN AND NOT A NEW SEARCH STAGE.
#
# R19 measured two mechanisms whose sign REVERSES across stores -- relocate kv +4 / vector -6, reprompt
# kv 0 / vector +4 -- and emitted 13 controller specs, every one a bare
# `{"declared_signal": ..., "params": {}}`. A loop that accepts or rejects a controller globally cannot
# express either result, so "where does this apply" was never a question measurement could answer.
#
# The whole of core's machinery for "one family, several groundings, one arm each" already exists:
# `instantiate` turns each grounding into its own `InstantiatedAction`, `PolicyArm.label` is derived
# from `instantiated.label` (so it is unique per variant), `arm_manifest` keys rows by that label, and
# `ExternalEvaluation` looks results up by it. Declaring applicability HERE therefore makes it
# selectable end to end with NO change to core. Declaring it anywhere else does not: a variant invented
# downstream of `build_arms` has no arm, so `select_on_measurement` cannot address it, its result reads
# as `missing`, and the round reports UNEVALUATED. That was measured before this was written.
#
# THE NAME OF A VARIANT IS NOT A GATE, which is the defect this closes. `relocate_entry_for_kv` is a
# grounding LABEL; the installed predicate reads only `error_kind` and `proposes_write`, and the executor
# dispatches on the RUNNING store. R19's telemetry proves the leak: that arm executed 32 relocations ON
# VECTOR, all 32 verified, and lost 7 vector cases. Only an explicit `backend == v` conjunct gates it.
#
# CONDITIONING IS OPTIONAL AND ADDITIVE. The unconditional grounding is always returned first and is
# never removed, so a mechanism that generalizes keeps its general arm and can win with it.
_APPLICABILITY_FIELD = "backend"


def _applicable_backends(signal: str, boundary) -> tuple[str, ...]:
    """Stores on which conditioning this signal could be INFORMATIVE, from declarations only.

    Three filters, each a declaration this adapter already makes:

      * the field must be DECLARED at this boundary -- otherwise the atom cannot be evaluated where the
        controller fires, and a predicate reading an absent field is not a firing;
      * a single-cell run yields NOTHING. With `ANCHOROPT_CELL` set the residual holds one store, so
        `backend == that_store` is constant on every state it could see: it carries zero information and
        `_separates`-style reasoning would discard it. Conditioning is only learnable corpus-wide, which
        is why R20 runs with ANCHOROPT_CELL="";
      * fewer than two candidate stores yields NOTHING -- a condition with no alternative is not a
        choice.

    Deliberately NOT filtered on which store the mechanism helps: that is the thing being measured, and
    encoding it here would be the backend->mechanism table this design exists to avoid.
    """
    import os as _os
    # OPT-IN, and this is a COST decision with a measured number. Conditioning multiplies the arm set
    # by 3.18x across this host's declared signals (17 unconditional arms -> 54), and every arm is a
    # paired GPU evaluation. Offering it unconditionally would silently triple the bill of every round,
    # including rounds that are not asking about applicability at all -- and it changed the arm counts
    # two existing contract tests assert, which is the same signal from a different direction.
    #
    # So the default is OFF and the experiment turns it on. A round that wants to learn applicability
    # sets ANCHOROPT_APPLICABILITY=backend; every other round is byte-identical to before.
    if str(_os.environ.get("ANCHOROPT_APPLICABILITY") or "").strip() != _APPLICABILITY_FIELD:
        return ()
    if _APPLICABILITY_FIELD not in (synthesis_fields(boundary) or {}):
        return ()
    if str(_os.environ.get("ANCHOROPT_CELL") or "").strip():
        return ()
    return tuple(_vocab.BACKENDS)


def with_applicability_variants(groundings, signal: str, boundary):
    """Each grounding, then one APPLICABILITY sibling per store the intervention can execute on.

    The sibling is the SAME intervention -- identical operator, eta, capability and detail -- carrying
    an `applicability` marker that the spec builder turns into the second conjunct of the predicate.
    Nothing about what the intervention DOES changes; only where it is allowed to fire.

    EXECUTABILITY IS RESPECTED, and this is the distinction that must not collapse. A grounding that
    already names the store it was derived for (`grounding["backend"]`) is conditioned ONLY on that
    store: offering `relocate_entry_for_kv` conditioned on rec_sum would be an arm whose action that
    store cannot perform, and `BACKENDS` declares no relocate condition there at all. A store-agnostic
    grounding (reprompt declares `backend: None` -- one instruction, executable anywhere) is offered on
    every candidate store, because for it applicability is exactly what is unknown.
    """
    cands = _applicable_backends(signal, boundary)
    if not cands:
        return list(groundings or ())
    out = []
    for g in (groundings or ()):
        out.append(g)
        if g.get("infeasible") or not str(g.get("variant") or ""):
            continue                       # nothing to condition: an infeasibility report is not an arm
        if g.get("applicability"):
            continue                       # ALREADY conditioned: `ground_transforms` re-labels a
            # relocation grounding as a transform sibling, so this runs over an already-wrapped list.
            # Conditioning a conditioned variant would nest `@backend=x@backend=x` and build an arm
            # whose predicate carries the same conjunct twice.
        own = str((g.get("grounding") or {}).get(_APPLICABILITY_FIELD) or "").strip()
        # a store-derived grounding is conditionable only on ITS OWN store (executability);
        # a store-agnostic one is conditionable on every candidate (applicability is open)
        scope = (own,) if own else cands
        for b in scope:
            if b not in cands:
                continue
            v = dict(g)
            v["variant"] = f"{g['variant']}@{_APPLICABILITY_FIELD}={b}"
            v["applicability"] = {"field": _APPLICABILITY_FIELD, "value": b}
            v["grounding"] = dict(g.get("grounding") or {})
            v["grounding"]["applicability"] = {"field": _APPLICABILITY_FIELD, "value": b}
            v["detail"] = (f"{g.get('detail') or ''} -- RESTRICTED to {_APPLICABILITY_FIELD} == {b!r}; "
                           f"the intervention is unchanged and the unconditional arm is measured "
                           f"alongside this one")
            out.append(v)
    return out


def ground_reprompt(signal: str, boundary):
    """eta_mu for REPROMPT: instruction content + retry budget.

    Several SEMANTICALLY DISTINCT instructions are returned, each its own arm, because which framing
    works is an empirical question. R2 measured one framing -- "distrust this result and verify" --
    net-negative at every threshold; that rejects the framing it tested, not the family, and the only
    way to tell the difference is to measure more than one.

    The instructions are written here as declared runtime content rather than generated per round, so
    an arm is reproducible. A generator may supply more, and measurement still chooses.
    """
    return [
        {"variant": "verify_before_answering",
         "eta": {"instruction": ("Before answering, check whether the retrieved content actually "
                                 "states the fact being asked for. If it does not, search again "
                                 "with different terms."),
                 "retry_budget": 1},
         "detail": "instructs verification of the result already in hand"},
        {"variant": "search_other_container",
         "eta": {"instruction": ("The container you searched may not hold this fact. Search the "
                                 "other memory container before answering."),
                 "retry_budget": 1},
         "detail": "redirects the model's own next read, without performing one"},
        {"variant": "state_absence_if_unfound",
         "eta": {"instruction": ("If no retrieved record states the requested value, say the stored "
                                 "memory does not contain it rather than inferring a value."),
                 "retry_budget": 0},
         "detail": "targets fabrication rather than under-retrieval"},
    ]


def ground_suppress(signal: str, boundary):
    """eta_mu for SUPPRESS: which operation is cancelled + what preserves safety.

    Boundary is part of the requirement: after execution there is nothing left to cancel, so this
    grounds only at the commitment gate. Suppression is offered only for signals that name a
    PROPOSED operation -- suppressing on a result-derived condition would cancel a call the condition
    never implicated.
    """
    if getattr(boundary, "value", str(boundary)) != "post_generation_pre_exec":
        return []
    # NAMED SIGNALS get a specific description of what is being cancelled, which reads better in the
    # eta and in any report.
    proposal_signals = {"clear_proposed_at_capacity": "the proposed destructive clear",
                        "duplicate_identifier": "the proposed duplicate write"}
    what = proposal_signals.get(signal)
    if what is None:
        # A SYNTHESIZED signal grounds too. This used to `return []` for any name outside the two
        # above, which meant no expanded signal could EVER compose with SUPPRESS: the action was
        # structurally admissible, an executor existed, and the search still produced zero arms. A
        # primitive that cannot be grounded for a new signal is indistinguishable from an absent
        # one, so the whole point of expanding Phi was lost at this line -- and the resulting "no
        # suppress candidate" read as a search result rather than a grounding limit.
        #
        # The two REAL invariants are kept, and they are what make suppression meaningful here:
        #   * the boundary must be the commitment gate (checked above) -- after execution there is
        #     nothing left to cancel;
        #   * the preservation clause stays, so a cancellation cannot silently lose the only copy.
        # Neither depends on the signal's NAME, which is why the whitelist was never the invariant.
        if signal not in EXPANDED_SIGNALS and signal not in SIGNALS:
            return []
        what = "the proposed operation this signal implicates"
    # NO `preservation` CLAIM. The hook at this boundary removes the call from the dispatch list and
    # puts nothing in its place -- the write never happens, so there is nothing to preserve and nothing
    # reads such a field. It was previously grounded as the string "verify an equivalent copy survives
    # before cancelling", which no code enforced; the capability check now refuses an arm that claims it,
    # and on the first real run that correctly rejected all 39 suppress candidates at this gate.
    #
    # A withhold-AND-REPLAY grounding is a different variant with a different contract: it must set
    # `variant` accordingly and carry `preservation`, backed by an executor that declares it consumes
    # the field. This host has no such executor, so it grounds only the remove-outright variant.
    return [{"variant": "cancel_proposed",
             "eta": {"suppressed_operation": what},
             "detail": f"cancels {what} at the commitment gate"}]


def _side_of(signal: str, read_side: set, write_side: set) -> str | None:
    """Whether a signal concerns the READ or WRITE side, for eta grounding.

    A SYNTHESIZED signal has a name no fixed set can contain, so grounding returned nothing for it and
    expansion could install a signal that no controller could then be built on -- the gap closes in Phi
    and reopens one step later. An expanded signal inherits its side from the declared signal whose
    condition it extends: the grammar builds conjunctions over DECLARED FIELDS, and a conjunction
    containing a read-side term is still about the read side.

    Falls through to None when nothing is identifiable, so an unclassifiable signal grounds nothing --
    the previous behaviour, rather than a guess.
    """
    if signal in read_side:
        return "read"
    if signal in write_side:
        return "write"
    if signal in EXPANDED_SIGNALS:
        # the conjunction's own name carries the fields it was built from
        low = signal.lower()
        if any(k in low for k in ("similarit", "retriev", "search", "found", "informative")):
            return "read"
        if any(k in low for k in ("capacit", "slot", "occupanc", "append", "duplicate")):
            return "write"
    return None


def ground_substitute_destinations(signal: str, boundary):
    """eta_mu for SUBSTITUTE: EVERY credible compatible destination, each as its own arm.

    Compatibility is structural and checked, not assumed: a destination must be a real tool of the
    same KIND as the operation being substituted, in a different container, whose required arguments
    can be carried MECHANICALLY from the original call. docs/ANCHORS.md records why the last clause
    matters -- emitting `key=` universally built an invalid call that was then misscored as "the
    substitute did not help".

    Both retry semantics are enumerated where both are executable, because replace-vs-retry is an
    action parameter and not a detail: it changes what the model ends up seeing.
    """
    read_side = {"retrieval_similarity_below_threshold", "identifier_not_found",
                 "no_informative_result"}
    write_side = {"container_at_capacity", "container_slots_exhausted"}
    schema = tool_schema()

    _side = _side_of(signal, read_side, write_side)
    if _side == "read":
        kind, origin_container = "read", "core"
    elif _side == "write":
        kind, origin_container = "write", "core"
    else:
        return []

    origin_args = {a for t, sp in schema.items()
                   if sp.get("kind") == kind and sp.get("container") == origin_container
                   for a in sp["args"]}
    out = []
    for tool in sorted(schema):
        spec = schema[tool]
        if spec.get("kind") != kind:
            continue
        if spec.get("container") in (origin_container,):
            continue                              # same container is not a substitution
        required = tuple(spec.get("args") or ())
        carried = sorted(set(required) & origin_args)
        ungrounded = sorted(set(required) - origin_args)
        if ungrounded:
            # A required argument that cannot be carried mechanically is not groundable. Skipped
            # here; the contract reports REROUTE_INFEASIBLE if nothing survives.
            continue
        if not carried:
            continue
        for retry in ("replace_original", "retry_after"):
            out.append({
                "variant": f"{tool}+{retry}",
                "eta": {"destination": tool,
                        "argument_mapping": {a: a for a in carried},
                        "retry_semantics": retry},
                "grounding": {"kind": kind, "container": spec.get("container"),
                              "required_args": list(required), "carried": carried},
                "detail": (f"{tool} is a {kind} on container {spec.get('container')}; arguments "
                           f"{carried} carry over unchanged; {retry}")})
    return out


def ground_capacity_relocations(signal: str, boundary):
    """eta_mu for a slot-freeing RELOCATION: move an existing entry out, then retry the refused write.

    WHY THIS IS NOT `ground_substitute_destinations`, and the distinction is the mechanism:

        substitute   write the NEW fact to a different container instead. The refused write never
                     happens, and the fact the model asked to put in core is somewhere else.
        relocate     move an EXISTING entry out of the constrained container, verify it survived,
                     then retry the ORIGINAL write verbatim. The refused write succeeds.

    Only the second frees a slot, and only the second is what the residual asks for: 208 of 218
    core-full refusals were followed by a wholesale clear, so the repair has to make room while keeping
    what is already stored.

    GROUNDED FROM THE HOST'S OWN DECLARATION. `capacity_repair.BACKENDS` already carries a `relocate`
    condition per backend -- which operation performs the move and which schema field bounds the
    destination. Nothing here invents a tool, a container or a limit.

    Feasibility is asked, not assumed: a relocation reduces `occupied_slots`, so it grounds only for a
    signal whose constraint is counted in slots. That is the mirror of the reduce test in
    `ground_transforms`, and it keeps one operator from claiming both constraints.
    """
    if getattr(boundary, "value", str(boundary)) != "post_execution":
        return []
    try:
        from bfcl_constraints import bfcl_constraint_contract, constraint_for_signal
    except Exception:                                      # pragma: no cover - packaged import
        from .bfcl_constraints import (  # type: ignore[no-redef]
            bfcl_constraint_contract, constraint_for_signal)
    cons = constraint_for_signal(signal)
    if not cons:
        return []
    verdict = bfcl_constraint_contract().check("relocate_entry", cons)
    if not verdict.ok:
        return [{"variant": "", "eta": {}, "grounding": {}, "infeasible": True,
                 "signal": signal, "constraint": cons,
                 "detail": verdict.detail, "remedy": verdict.remedy}]

    backends = {}
    for _imp in ("anchoropt.mechanisms.capacity_repair", "capacity_repair"):
        try:
            _cr = __import__(_imp, fromlist=["BACKENDS"])
            backends = getattr(_cr, "BACKENDS", {}) or {}
            if backends:
                break
        except Exception:
            continue
    import os as _os
    active = str(_os.environ.get("ANCHOROPT_CELL") or "").strip()
    out = []
    for backend in sorted(backends):
        if active and backend != active:
            continue                       # an alias for another backend is not a distinct arm
        for cond in (backends[backend].get("conditions") or []):
            if str(cond.get("kind", "")).lower() != "relocate":
                continue
            op = str(cond.get("operation") or "")
            if not op:
                continue
            out.append({
                "variant": f"relocate_entry_for_{backend}",
                "eta": {"destination_operation": op,
                        "budget_from": str(cond.get("budget_from") or ""),
                        "preservation": ("the relocated entry is verified present in the destination "
                                         "BEFORE it is removed from the source; if the destination "
                                         "write does not land, nothing is removed"),
                        "retry_semantics": "retry_original_verbatim_after_capacity_freed"},
                "grounding": {"backend": backend, "operation": op,
                              "budget_from": str(cond.get("budget_from") or ""),
                              "capability_id": "relocate_entry_preserving_information_then_retry"},
                "detail": (f"frees ONE slot on {backend} by moving an entry via {op} to a container "
                           f"bounded by {cond.get('budget_from')}, verifying the move against live "
                           f"state, then retrying the refused write verbatim; bounded per episode")})
    return out


def ground_transforms(signal: str, boundary):
    """eta_mu for TRANSFORM: a writable surface + operator + preservation constraint.

    TRANSFORM changes the STATE OR OBSERVATION a later decision reads, rather than which operation
    runs. Only grounded where a surface is actually writable at this boundary: the returned
    observation exists only post-execution.

    The `merge_additional_read` operator is derived structurally -- two scored read result sets share
    one ordering axis, so a union can be ranked exactly. A read exposing no scores has no common
    ordering with another, so no exact transform exists and nothing is returned rather than
    approximated.
    """
    if getattr(boundary, "value", str(boundary)) != "post_execution":
        return []
    read_side = {"retrieval_similarity_below_threshold", "identifier_not_found",
                 "no_informative_result"}

    # ---- WRITE-SIDE: a fact-preserving payload REDUCTION, grounded from the host's own primitive ----
    #
    # This function previously returned [] for every write-side signal, so the only transform it could
    # ground was a read-side merge. That is why the search kept reducing the capacity family to
    # "suppress the clear": suppression was the only expressible neighbour of a repair the teacher
    # stated as "size or condense the entry so it fits".
    #
    # The grounding is READ OFF `capacity_repair.BACKENDS`, which already declares -- per backend and
    # per condition -- which error text signals it, which operation applies the replacement, which
    # argument is reduced, and which schema field bounds it. So nothing here invents a limit, a tool or
    # a reduction rule; it reports what the host has already validated (352/352 replacements
    # constructed on live captures).
    #
    # One variant per (backend, condition) whose kind is REDUCE. RELOCATE conditions are deliberately
    # NOT grounded here: moving a payload to another container is the reroute cell's job, and two
    # operators for one effect is the ambiguity the action contract exists to remove.
    write_side = {"append_would_exceed_cap", "container_at_capacity", "container_slots_exhausted"}
    if _side_of(signal, set(), write_side) == "write":
        out = []
        # FEASIBILITY BEFORE GROUNDING. A REDUCE variant shortens a payload, so it can only relieve a
        # constraint counted in CHARACTERS. `container_at_capacity` on this host is
        # "exceeds maximum size of N entries" -- a slot count -- and a shorter value frees zero slots.
        #
        # Without this test the write_side set alone admitted it: the round emitted
        # post_execution/container_at_capacity/transform:reduce_value_for_kv, an arm that would have
        # measured a null indistinguishable from "the mechanism does not transfer". The backend's own
        # reduce condition is bound to `core_entry` (the 300-char per-entry limit), which is the
        # entry_too_long constraint -- not this one. Asking the declared contract is what separates them.
        try:
            from bfcl_constraints import bfcl_constraint_contract, constraint_for_signal
        except Exception:                                  # pragma: no cover - packaged import
            from .bfcl_constraints import (  # type: ignore[no-redef]
                bfcl_constraint_contract, constraint_for_signal)
        _cons = constraint_for_signal(signal)
        if _cons:
            _v = bfcl_constraint_contract().check("reduce_preserving_facts", _cons)
            if not _v.ok:
                # NO REDUCE VARIANT HERE -- but that is not the same as "no transform exists". A
                # RELOCATION is also a transform under the action contract: it changes the STATE a
                # later decision reads (the container's occupancy) rather than which operation runs.
                # So offer the feasible sibling instead of returning a bare refusal, which is what
                # collapsed this family to "suppress the clear" before.
                # THE *UNCONDITIONED* RELOCATIONS ONLY. `ground_capacity_relocations` is wrapped, so
                # its result already carries applicability siblings; passing those through here and
                # letting the outer wrapper skip them (its guard does) still put each conditioned
                # variant in the list TWICE -- once from the inner wrapper, once as a pass-through --
                # which produced two arms with the SAME label. Duplicate labels are how one arm's
                # measurement gets credited to another, so the inner call is read at its unwrapped
                # source and the outer wrapper is left to do the conditioning exactly once.
                _rel_src = getattr(ground_capacity_relocations, "__wrapped__",
                                   ground_capacity_relocations)
                _rel = _rel_src(signal, boundary)
                _rel_ok = [r for r in _rel
                           if not r.get("infeasible") and not r.get("applicability")]
                if _rel_ok:
                    return [{"variant": r["variant"],
                             # The contract's required TRANSFORM keys, carried explicitly.
                             "eta": {"target_surface": "constrained_container_occupancy",
                                     "operator": "relocate_entry_preserving_information",
                                     "preservation": r["eta"]["preservation"],
                                     **{k: v for k, v in r["eta"].items()
                                        if k != "preservation"}},
                             "grounding": r["grounding"], "detail": r["detail"]}
                            for r in _rel_ok]
                return [{"variant": "", "eta": {}, "grounding": {},
                         "infeasible": True, "signal": signal, "constraint": _cons,
                         "detail": _v.detail, "remedy": _v.remedy}]
        # The module ships in the repo as `anchoropt.mechanisms.capacity_repair`; on the cluster the
        # same file is importable as top-level `capacity_repair` from scripts/. Try both, because a
        # silent ImportError here returns [] and looks exactly like "no grounding exists" -- which is
        # how this grounding came back empty on the first attempt.
        backends = {}
        for _imp in ("anchoropt.mechanisms.capacity_repair", "capacity_repair"):
            try:
                _cr = __import__(_imp, fromlist=["BACKENDS"])
                backends = getattr(_cr, "BACKENDS", {}) or {}
                if backends:
                    break
            except Exception:
                continue
        # ONLY THE BACKEND BEING MINED. `capacity_repair.repair()` dispatches on the RUNNING backend,
        # so a grounding naming another one is not a distinct intervention: at best it is an ALIAS for
        # this backend's condition (3 arms measuring 1 mechanism, a 3x GPU bill for no information),
        # and at worst it selects a condition whose operation this backend does not have.
        #
        # The active backend is supplied by the adapter via ANCHOROPT_CELL, which the driver sets from
        # its own --cell. With no hint every backend is grounded, which is the previous behaviour and
        # the right default for a caller that has not said which cell it is mining.
        import os as _os
        _active = str(_os.environ.get("ANCHOROPT_CELL") or "").strip()
        _wanted = [b for b in sorted(backends) if not _active or b == _active]
        for backend in _wanted:
            for cond in (backends[backend].get("conditions") or []):
                if str(cond.get("kind", "")).lower() not in ("reduce", "1", "reduce_payload"):
                    continue
                arg = str(cond.get("reduce_arg") or "")
                op = str(cond.get("operation") or "")
                if not arg or not op:
                    continue
                out.append({
                    "variant": f"reduce_{arg}_for_{backend}",
                    "eta": {"target_surface": "proposed_payload",
                            "operator": f"reduce_preserving_facts:{arg}",
                            "preservation": ("every fact in the original payload is retained; only "
                                             "length is reduced, to the schema's own limit")},
                    "grounding": {"backend": backend, "operation": op, "reduce_arg": arg,
                                  "budget_from": str(cond.get("budget_from") or ""),
                                  "replaces": bool(cond.get("replaces"))},
                    "detail": (f"reduces `{arg}` to the {cond.get('budget_from')} budget and "
                               f"re-dispatches via {op}, so the write SUCCEEDS rather than being "
                               f"prevented")})
        return out

    if _side_of(signal, read_side, set()) != "read":
        return []
    schema = tool_schema()
    out = []
    for tool in sorted(schema):
        spec = schema[tool]
        if spec.get("kind") != "read" or spec.get("container") == "core":
            continue
        if "top_k" not in tuple(spec.get("args") or ()):
            continue                 # unscored: no common ordering, so no EXACT merge exists
        out.append({
            "variant": f"merge_with_{tool}",
            "eta": {"target_surface": "returned_observation",
                    "operator": f"merge_additional_read:{tool}",
                    "preservation": "the original result is retained; the union is ranked by score"},
            "grounding": {"additional_source": tool, "combine": "rank_union_by_score"},
            "detail": (f"performs an additional read via {tool}, keeps the original result, and "
                       f"hands back the score-ranked union")})
    return out


def reroute_destination(signal: str) -> Mapping[str, Any] | None:
    """The attested reroute destination for `signal`, or None if this benchmark has none.

    None is a first-class answer and it PRUNES the reroute cell. That is the point: a reroute whose
    destination is invented cannot be attested on resolving, and shipping one would produce a
    measurement that looks like "the substitute did not help" when the truth is that no substitute
    was ever named.
    """
    entry = REROUTE_DESTINATIONS.get(signal)
    return dict(entry) if entry else None


# ------------------------------------------------------------------------------------------------
# 5. applying a semantic action
# ------------------------------------------------------------------------------------------------
def apply_action(action: Action, boundary: IncisionPoint, theta: Mapping[str, Any],
                 state: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a SEMANTIC action into this runtime's directive.

    Returns a plain semantic dict; performs NO I/O, so it is testable without a live harness. It
    names WHAT must happen, never the channel -- which mechanism module executes it is the
    caller's business (`anchoropt/mechanisms/`).

    Raises `ActionNotExecutable` via `HOST.require` BEFORE building any payload, so an unexecutable
    cell can never report success. That ordering is the CONSUMER_BOUNDARY_RULE: the TB2 prototype
    built its payload first and reported `executed=True` while appending to the empty string.
    """
    HOST.require(boundary, action)

    if action is Action.NOOP:
        return {"kind": "noop", "executed": False}

    if action is Action.REPROMPT:
        text = str(theta.get("text", "")).strip()
        if not text:
            raise ValueError("REPROMPT requires non-empty theta['text']")
        return {"kind": "reprompt", "executed": True, "text": text,
                "request_redecision": boundary is IncisionPoint.POST_GENERATION_PRE_EXEC}

    if action is Action.SUPPRESS:
        reason = str(theta.get("reason", "")).strip()
        if not reason:
            raise ValueError("SUPPRESS requires non-empty theta['reason']")
        return {"kind": "suppress", "executed": True, "reason": reason,
                "replacement": theta.get("replacement"), "short_circuit": True}

    if action is Action.REROUTE:
        # A destination is MANDATORY. docs/ANCHORS.md (an accepted anchor): a destination was attested at 11/11
        # "clean" where clean meant did-not-error, and 9 of those 11 returned nothing. A reroute
        # with no named destination is unexecutable, so it must raise rather than no-op.
        destination = str(theta.get("destination", "")).strip()
        if not destination:
            raise ValueError("REROUTE requires theta['destination'] -- a reroute with no named "
                             "destination cannot be attested on RESOLVING")
        return {"kind": "reroute", "executed": True, "destination": destination,
                "args_patch": dict(theta.get("args_patch") or {}),
                "retry_original": bool(theta.get("retry_original", False)),
                "reason": str(theta.get("reason", ""))}

    raise AssertionError(f"unhandled action {action!r}")


# ---- APPLY THE APPLICABILITY WRAPPER ONCE, AT MODULE SCOPE ---------------------------------------
#
# Wrapped here rather than inside each hook: five hooks have six top-level `return` statements between
# them, and a per-return edit is a change that drifts -- one new early return and a family silently
# stops offering conditioned arms, which is indistinguishable from "conditioning did not help".
#
# Placed AFTER every hook is defined and BEFORE `BFCLRuntime` binds them as staticmethods, so the class
# and the module-level names are the same wrapped functions. `ground_transforms` calls
# `ground_capacity_relocations` internally and re-labels its result, so `with_applicability_variants`
# skips a grounding that already carries an `applicability` marker -- otherwise that path would nest the
# same conjunct twice.
def _wrap_with_applicability(fn):
    import functools as _ft

    @_ft.wraps(fn)
    def _wrapped(signal, boundary):
        return with_applicability_variants(fn(signal, boundary) or [], signal, boundary)

    _wrapped.__wrapped_for_applicability__ = True
    return _wrapped


for _name in ("ground_reprompt", "ground_suppress", "ground_substitute_destinations",
              "ground_capacity_relocations", "ground_transforms"):
    _fn = globals()[_name]
    if not getattr(_fn, "__wrapped_for_applicability__", False):
        globals()[_name] = _wrap_with_applicability(_fn)
del _name, _fn


# ------------------------------------------------------------------------------------------------
# The runtime record + registration
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BFCLRuntime:
    """The BFCL v4 runtime surface as one object, for injection-style callers."""

    name: str = NAME
    host: HostProfile = HOST

    normalize_event = staticmethod(normalize_event)
    observable_state = staticmethod(observable_state)
    feasible_actions = staticmethod(feasible_actions)
    evaluate_signal = staticmethod(evaluate_signal)
    apply_action = staticmethod(apply_action)
    declared_signals = staticmethod(declared_signals)
    signal_aliases = staticmethod(signal_aliases)
    signal_boundaries = staticmethod(signal_boundaries)
    probe_params = staticmethod(probe_params)
    parameter_domains = staticmethod(parameter_domains)
    policy_class_for = staticmethod(policy_class_for)
    action_theta_schema = staticmethod(action_theta_schema)
    signal_param_schema = staticmethod(signal_param_schema)
    reroute_destination = staticmethod(reroute_destination)
    derive_reroute_destination = staticmethod(derive_reroute_destination)
    ground_reprompt = staticmethod(ground_reprompt)
    executor_for = staticmethod(executor_for)
    executor_supports = staticmethod(executor_supports)
    executor_capability = staticmethod(executor_capability)
    expanded_signal_names = staticmethod(expanded_signal_names)
    capability_audit = staticmethod(capability_audit)
    ground_suppress = staticmethod(ground_suppress)
    ground_substitute_destinations = staticmethod(ground_substitute_destinations)
    ground_transforms = staticmethod(ground_transforms)
    # capability surface -- the alphabet a proposer may build new signals from
    observable_fields = staticmethod(observable_fields)
    carried_fields = staticmethod(carried_fields)
    all_fields = staticmethod(all_fields)
    fields_at = staticmethod(fields_at)
    tool_schema = staticmethod(tool_schema)
    tools_of_kind = staticmethod(tools_of_kind)
    unknown_args = staticmethod(unknown_args)
    register_signal = staticmethod(register_signal)


RUNTIME = BFCLRuntime()
