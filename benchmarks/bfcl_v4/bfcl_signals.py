"""Phi_BFCL -- the BFCL v4 signal vocabulary, as runtime evaluators.

SEPARATE FROM `adapter.py` ON PURPOSE. That module is the benchmark's VOCABULARY: ~25 predicates
(`is_write`, `container_of`, `error_kind`, `parse_case_id`) that answer "what kind of call/result is
this". ~294 tests depend on its exact behaviour, and nothing here modifies it -- this module IMPORTS
it and adds the runtime surface `candidate_search` needs on top.

    adapter.py        vocabulary   what kind of call is this, what kind of error is that
    bfcl_signals.py   Phi_BFCL     which CONDITIONS are observable at a boundary   <- new
    bfcl_runtime.py   the surface  U_H(l), evaluate_signal, apply_action, normalize <- new

WHY THE SIGNALS ARE THE ONES THEY ARE
-------------------------------------
Each declared signal is the trigger of an ALREADY-ACCEPTED anchor (docs/ANCHORS.md), re-expressed
as a predicate over normalized state. That is deliberate and it is the cheapest available
correctness test: if the search cannot recover the eight known controllers from this vocabulary,
the runtime is wrong and no BFCL evaluation is worth launching (scripts/anchor_recovery.py).

It is NOT a claim that this vocabulary is complete. It is the vocabulary the accepted stack
already proved is expressible; Phi-expansion is what grows it.

VACUITY IS NOT REDESIGNED HERE
------------------------------
`no_informative_result` delegates to `policy_tree.vacuous_result_kind`. A successful call returning
`{"ranked_results": []}` is a FAILURE with no syntactic cue, and docs/GENERALIZABILITY.md records
that this whole class was invisible until a human read trajectories by hand. Re-implementing the
detector here would fork it; there would then be two definitions of "vacuous" that could disagree,
which is precisely how this project has been bitten before.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import adapter as _vocab
from anchoropt.learning.policy_tree import vacuous_result_kind

# NO FIXTURE IMPORT HERE. An earlier version imported SIGNAL_ALIASES and SIGNAL_PROBE_PARAMS from
# `fixtures.replay_anchors`, which meant that merely importing the runtime pulled the answer key
# into the process -- and the live smoke run's leak detector caught it. A discovery claim made with
# the answer key in the room is a lookup, however unused it looks.
#
# So the runtime declares only what it can declare on its own account:
#
#   SIGNAL_ALIASES       EMPTY. Prose aliases were phrases lifted from each anchor's attribution text.
#                        They are calibration data; `scripts/anchor_recovery.py` injects them.
#   SIGNAL_PROBE_PARAMS  EMPTY. A probe value is a MEASURED number from an accepted anchor. Declaring
#                        one here would export a value measured on one backend to every future
#                        proposal. A proposer must supply its own parameters and have them validated.
#
# Both are module-level and mutable so the recovery test can inject the fixture explicitly, which
# keeps that dependency visible in the test rather than hidden in an import.
SIGNAL_ALIASES: dict[str, tuple[str, ...]] = {}
SIGNAL_PROBE_PARAMS: dict[str, Mapping[str, Any]] = {}

# ------------------------------------------------------------------------------------------------
# The signals. Each is a pure function of adapter-normalized state -- no I/O, no benchmark objects.
# ------------------------------------------------------------------------------------------------


def container_at_capacity(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """A write was refused because its container has no room.

    Keyed on the REFUSAL, never on the verb: `docs/ANCHORS.md` records that keying a remedy on the
    call kind proposed a suppression that inspection showed benign in 15 of 17 cases.
    """
    return state.get("error_kind") == "no_capacity" and bool(state.get("proposes_write", False))


def identifier_not_found(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """A read did not find what it asked for. this trigger (read side only).

    the attribution for this locus split ONE error string into 39 read-side and 5 write-side cases. The read-side
    test is part of the signal, not a downstream filter: without it the signal names two different
    failures with one remedy.
    """
    return state.get("error_kind") == "not_found" and bool(state.get("proposes_read", False))


def duplicate_identifier(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """A write whose identifier is already present in the store. this trigger.

    TWO OBSERVATION MODES, and the pre-execution one is the anchor's:

      post_generation_pre_exec  a write is PROPOSED whose key is already known present (from the
                                store's live state) -- the duplicate has not happened yet
      post_execution            the store has already refused it (`error_kind`)

    the accepted form fires at the FIRST. docs/ANCHORS.md: this is "the first anchor at the commitment gate, which
    is the only point where cancelling a call is free: after execution the duplicate has already
    been rejected and the step is spent."

    The first version of this predicate read `error_kind` alone, which exists only after execution.
    It therefore could not fire where the accepted form actually fires, while still being DECLARED observable
    pre-execution -- so the search reached the cell and the predicate silently answered False
    there. That is the defect shape the boundary declaration guards, and the reason
    `SIGNAL_BOUNDARIES` and the predicate must
    agree.
    """
    if state.get("error_kind") == "duplicate_identifier":
        return True
    return bool(state.get("proposes_write", False)) and bool(state.get("identifier_present", False))


def no_tool_call_at_all(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """The model is about to answer without having called any tool.

    Observable ONLY after generation: whether a decision is a tool call or a final answer does not
    exist before that decision is made, so evaluating this earlier would fire on every episode.
    """
    if not bool(state.get("has_generation", False)):
        return False
    return not bool(state.get("proposes_tool_call", False)) and int(state.get("tool_calls_so_far", 0)) == 0


def container_slots_exhausted(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """The relocation TARGET is out of slots, so an earlier relocation has nowhere to put its payload.

    this trigger. Distinct from `container_at_capacity` by which container refused: the deeper variant exists
    because a successful relocation eventually saturates its target, one level deeper in the same
    residual.
    """
    if state.get("error_kind") != "no_capacity":
        return False
    return state.get("container") == "archival" or bool(state.get("is_relocation_target", False))


def append_would_exceed_cap(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """A single-blob store refused an append that would overflow its cap. this trigger.

    The backend keeps memory as ONE string with no second container, so there is nowhere to
    relocate to -- which is why the remedy is to shorten the store rather than to reroute.
    """
    return state.get("error_kind") == "blob_would_overflow"


def clear_proposed_at_capacity(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """The model PROPOSES a destructive clear while the container is full. the destructive-clear trigger.

    Observable strictly pre-execution: after the clear runs there is nothing left to save. This is
    the one signal whose entire value depends on the boundary.
    """
    if not bool(state.get("proposes_clear", False)):
        return False
    return bool(state.get("container_full", False)) or state.get("error_kind") == "no_capacity"


def retrieval_similarity_below_threshold(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """A retrieve SUCCEEDED and returned nothing well-matched.

    The structural-signal tier the lexical vocabulary cannot reach: no error is raised at all, so a
    read that fails silently looks like a successful read. `below` is REQUIRED: a threshold measured
    on one backend must not be silently exported to backends whose scores are not comparable, and a
    default here would do exactly that. The optimizer supplies it from TRAIN.
    """
    below = float(params["below"])
    score = state.get("best_similarity")
    if score is None:
        return False
    return bool(state.get("proposes_read", False)) and float(score) < below


def no_informative_result(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """A syntactically successful call whose payload carries NO information.

    Delegates to `policy_tree.vacuous_result_kind` -- the existing detector, not a new one. It
    classifies `{"ranked_results": []}` as `empty_collection` and an all-0.0-scored ranking as
    `all_floor_scores`, and it never fires on a payload carrying an explicit error marker (that is
    `error_kind`'s job; double-counting one event as two conditions inflates support).

    This is the runtime condition the task description asks for: a vacuous observation is neither an
    error nor an ordinary informative success, and it must be representable as its own condition or
    the failure is invisible to an error-keyed search.
    """
    if state.get("error_kind") is not None:
        return False
    payload = state.get("result")
    if payload is None:
        return False
    kind = vacuous_result_kind(payload)
    if kind is None:
        return False
    want = params.get("kind")
    return True if want is None else kind == str(want)


# ------------------------------------------------------------------------------------------------
# Natural-language aliases -- expressibility matching ONLY, never dispatch.
# ------------------------------------------------------------------------------------------------
#
# Why this exists at all: the matcher compares a diagnosis's PROSE against a SIGNAL NAME, and the
# two vocabularies genuinely differ. One attribution says "a write was blocked by a full
# container"; the signal is `container_at_capacity`. They describe one observable and share no
# distinctive token, so the canonical name alone marks the canonical case SIGNAL_BLOCKED.
#
# Multi-word phrases only. Single generic verbs match incidental prose in unrelated failures -- the
# TB2 integration paid for that with a SUPPRESS controller selected for an arithmetic error.
#
# These are a DECLARED, reviewable part of the benchmark's vocabulary. They do not make the matcher
# semantic; the principled replacement is the consequential_decision -> observable mapping that
# Phi-expansion builds.


def redundant_proposed_write(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """A write is PROPOSED whose value is already stored under its key.

    PRE-EXECUTION by construction, and that is the point: at the commitment gate cancelling the call is
    free. After execution the store has already refused and the step is spent -- which is the DIFFERENT
    and far rarer condition `duplicate_identifier` covers. Measured on raw H0: a post-refusal duplicate
    occurs 4 / 0 / 0 times across kv / vector / rec_sum, while a redundant write PROPOSED before
    commitment occurs 449 / 104 / 178 times.

    The comparison itself is NOT made here. `proposal_is_redundant` is supplied by the adapter, which
    asks the host's own generic comparator (`redundant_write.is_redundant`) against the LIVE store named
    by the call: same key, normalised-identical value, nothing fuzzy. An absent key, a missing value, or
    any difference at all answers False -- so a genuine UPDATE (same key, DIFFERENT value) is never
    reported redundant, and that safety property belongs to the comparator rather than to a threshold.

    A state without the field answers False rather than guessing.
    """
    if not bool(state.get("proposes_write", False)):
        return False
    return bool(state.get("proposal_is_redundant", False))


def proposal_already_refused(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """This exact call was already REFUSED earlier in this episode.

    The condition the attributor named repeatedly and no signal expressed: "never emit a write whose
    arguments are byte-identical to one that has already failed in this episode". Distinct from
    `redundant_proposed_write`, and the distinction is what a measured
    suppression round exposed:

        redundant_proposed_write   the store ALREADY HOLDS this value -> the write is unnecessary
        proposal_already_refused   the store REFUSED this value       -> the write cannot succeed

    Measured on kv prereq: of 396 exact repeats of an earlier proposal, 199 repeat a COMMITTED write
    and 197 repeat a REFUSED one. Suppressing the first loses nothing; suppressing the second prevents
    a call that is guaranteed to fail again. They are different populations of comparable size, and
    only the first had a signal.

    Episode-scoped and strictly backward-looking: `refused_before` counts only refusals recorded
    BEFORE this call was proposed, so it carries no information the live gate could not have.
    """
    if not bool(state.get("proposes_tool_call", True)):
        return False
    return int(state.get("proposal_refused_count", 0) or 0) > 0


SIGNALS: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], bool]] = {
    "redundant_proposed_write": redundant_proposed_write,
    "proposal_already_refused": proposal_already_refused,
    "container_at_capacity": container_at_capacity,
    "identifier_not_found": identifier_not_found,
    "duplicate_identifier": duplicate_identifier,
    "no_tool_call_at_all": no_tool_call_at_all,
    "container_slots_exhausted": container_slots_exhausted,
    "append_would_exceed_cap": append_would_exceed_cap,
    "clear_proposed_at_capacity": clear_proposed_at_capacity,
    "retrieval_similarity_below_threshold": retrieval_similarity_below_threshold,
    "no_informative_result": no_informative_result,
}

# Typed parameter schema: name -> (type, required). DECLARED, never inferred -- a proposal omitting
# a required param is PRUNED rather than run with a silent default. A similarity threshold is
# required for exactly that reason: it is a measured, backend-specific number, not a constant.
SIGNAL_PARAMS: Mapping[str, Mapping[str, tuple[type, bool]]] = {
    "container_at_capacity": {},
    "identifier_not_found": {},
    "duplicate_identifier": {},
    "no_tool_call_at_all": {},
    "container_slots_exhausted": {},
    "append_would_exceed_cap": {},
    "clear_proposed_at_capacity": {},
    "retrieval_similarity_below_threshold": {"below": (float, True)},
    "no_informative_result": {"kind": (str, False)},
}

# Probe values for REQUIRED parameters, so a parameterized signal can be evaluated during search.
#
# Without these, `candidate_search` probes with {} and a signal whose parameter is required raises
# -- which excluded the parameterized trigger from the search entirely. A probe default is NOT a tuned
# hyperparameter, and this table is EMPTY in the shipped runtime: a probe value would export one
# backend's measured number into every future proposal. The calibration harness injects one; the
# autonomous path searches theta on TRAIN instead. A signal whose parameter has no measured value belongs in the
# Phi-expansion backlog, not in a probe default with an invented number.

# PARAMETER DOMAINS -- what the OPTIMIZER may search, per signal parameter.
#
# Requirement 1: this is declared generically, per parameter, so ANY signal with a tunable
# threshold or range exposes its domain. It is not a similarity special case -- `no_informative_result`
# declares its categorical domain here too, and a future signal declares its own.
#
# `observed_from` names the runtime field whose OBSERVED distribution seeds the grid. That is the
# substance of requirement 5: candidates come from values the data actually takes, so no arm can be
# a threshold that discriminates nothing. It also means no historical anchor number is privileged --
# if 0.30 appears in a grid it is because the observed distribution put it there.
#
# NOTE ON RANGES: `low`/`high` bound what is LEGAL, not what is sensible. The grid comes from the
# observations, and the bounds only reject a value that cannot mean anything.
SIGNAL_PARAM_DOMAINS: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    "retrieval_similarity_below_threshold": {
        "below": {"kind": float, "low": 0.0, "high": 1.0,
                  "observed_from": "best_similarity",
                  "monotone": "lower_fires_less",
                  "doc": "similarity floor; a read whose best match scores under this is weak"},
    },
    "no_informative_result": {
        # Categorical, and included precisely to show the domain layer is not threshold-only.
        "kind": {"kind": str, "values": ("empty_collection", "all_floor_scores"),
                 "doc": "which vacuity class to fire on; omitted means either"},
    },
}

# Which boundaries each signal is OBSERVABLE at. Declared rather than inferred, because getting it
# wrong is a measured defect class: a condition evaluated where the fact does not yet exist fires
# everywhere and pays the cost on episodes that did not need it.
#
#   PRE_GENERATION            nothing about this decision exists yet
#   POST_GENERATION_PRE_EXEC  a proposal exists, nothing has run
#   POST_EXECUTION            a result exists
SIGNAL_BOUNDARIES: Mapping[str, frozenset[str]] = {
    # Pre-execution ONLY. Declaring it at post_execution too would duplicate
              # `duplicate_identifier`'s job and invite two signals for one event.
    "redundant_proposed_write": frozenset({"post_generation_pre_exec"}),
    # Pre-execution only: the point is to stop the call BEFORE it fails again.
    "proposal_already_refused": frozenset({"post_generation_pre_exec"}),
    "container_at_capacity": frozenset({"post_execution"}),
    "identifier_not_found": frozenset({"post_execution"}),
    "duplicate_identifier": frozenset({"post_generation_pre_exec", "post_execution"}),
    "no_tool_call_at_all": frozenset({"post_generation_pre_exec"}),
    "container_slots_exhausted": frozenset({"post_execution"}),
    "append_would_exceed_cap": frozenset({"post_execution"}),
    "clear_proposed_at_capacity": frozenset({"post_generation_pre_exec"}),
    # BOTH, and the second was added only once an executor made it TRUE (R13).
    #
    # post_execution: the retrieve has just returned, so its score is this step's own fact.
    # post_generation_pre_exec: at the ANSWER boundary the score of a COMPLETED earlier retrieval is
    #   still a fact about this turn, read from `trajectory_history`. That is backward-looking, so it
    #   introduces no future information -- the discipline the call-filtering path states ("a field
    #   that does not exist yet must not be supplied") is about the FUTURE, not about the past. An
    #   existing gate in the evaluator already reads a completed step's retrieval telemetry in this
    #   same window on this same ground.
    #
    # This entry was `{post_execution}` alone until the answer-boundary seam
    # (patches/bv/bv_a2_answer_boundary_seam.py) actually supplied it there. Declaring a boundary
    # before an executor supplies it is the declared-vs-supplied defect this file's own comments warn
    # about; the order matters, and the seam came first.
    "retrieval_similarity_below_threshold": frozenset({"post_execution",
                                                       "post_generation_pre_exec"}),
    "no_informative_result": frozenset({"post_execution"}),
}

# Re-exported so callers need not import both modules for one question.
error_kind = _vocab.error_kind
