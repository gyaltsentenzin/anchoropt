"""The CCTU AnchorOpt RUNTIME -- supplies itself TO the core, and declares only what it can execute.

Third runtime adapter, after `benchmarks/bfcl_v4/` and `benchmarks/tb2_deepagents/`. Same shape as
both: module-level functions are the interface, a frozen dataclass record exists for injection-style
callers, and `register()` installs it. Duck-typed, no ABC -- the registry in
`anchoropt/attribution/__init__.py` declines to pin one on purpose.

    cctu_capabilities.py   the ALPHABET     which facts exist, of what type, at which boundary
    cctu_signals.py        Phi_CCTU         the four conditions, and the three classifiers
    cctu_state.py          LIVE STATE       "would this turn violate anything", via upstream's handlers
    cctu_adapter.py        the RUNTIME      U_H(l), boundary derivation, grounding, apply_action

WHICH HALF OF THE CONTRACT THIS IMPLEMENTS -- BOTH
--------------------------------------------------
The two existing adapters split the contract in `adapters/adapter_template.py` differently, and it is
worth knowing which one to read for what:

    tb2_adapter.py   the EXECUTION half only -- normalize_event, observable_state, evaluate_signal,
                     apply_action, HOST. It cannot run `optimize_residual`: it declares no
                     `is_decision`, no `boundary_key`, no `synthesis_fields`, no grounder.
    bfcl_runtime.py  the full SEARCH contract, plus a great deal of history.

This adapter implements both halves, because `ANCHOROPT.md` commits CCTU to the WHERE -> WHAT -> HOW
search. `scripts/check_adapter.py` is the arbiter; run it before spending anything on a run.

WHAT THIS MODULE OWNS, THAT THE CORE MUST NEVER SEE
---------------------------------------------------
Constraint-checker concepts, violation-class names, the `INSTRUCTION FOLLOWING ERROR` contract, the
`<query_id>_<repeat>` case-id shape, and upstream's handler classes. `anchoropt/` contains none of
these, and `tests/test_cctu_adapter.py` must grep the core to keep it that way -- the same guard
`tests/test_tb2_adapter.py` applies for its own framework vocabulary.

REGISTRATION IS EXPLICIT, AND IMPORTING THIS MODULE DOES NOT REGISTER IT
-----------------------------------------------------------------------
Following `tb2_adapter.py` rather than `adapter.py`: there is ONE global adapter slot, and taking it
on import would silently repoint BFCL's core call sites at the wrong vocabulary. Call `register()`
from a CCTU entry point.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.runtime import HostProfile

import cctu_capabilities as _caps
from cctu_signals import (
    SIGNAL_ALIASES, SIGNAL_BOUNDARIES, SIGNAL_PARAM_DOMAINS, SIGNAL_PARAMS, SIGNAL_PROBE_PARAMS,
    SIGNALS, dimensions_in, is_tool_fault, result_vacuity_kind, violation_classes_in,
)

NAME = "cctu_constraints"


# ------------------------------------------------------------------------------------------------
# U_H(l) -- what THIS runtime can execute where
# ------------------------------------------------------------------------------------------------
#
# Narrower than the structural grid, and every narrowing carries its reason. The rule this project is
# held to (`docs/CONSUMER_BOUNDARY_RULE.md`): a cell that passes the structural grid can still do
# nothing at runtime while reporting success, so a cell is declared only where a complete directive
# can be produced AND an executor is specified to consume it.
#
# EVIDENCE LEVEL, stated so it cannot be misread as a live claim: every cell below is
# DIRECTIVE-LEVEL -- `apply_action` produces a complete directive and `EXECUTORS` names what will run
# it. None has yet fired in a live episode, because the wrapper does not exist. `EXECUTORS[...]
# ["driven"]` is False everywhere and step 5 is what flips it.
#
# TWO CELLS ARE WITHHELD, and the reasons are findings rather than caution:
#
#   POST_EXECUTION / REROUTE
#     substitute-a-different-tool needs the new tool's ARGUMENTS, and on CCTU those are semantic and
#     per-tool -- exactly what `docs/GENERALIZABILITY.md` warns is not mechanically mappable: "a
#     reroute is only deterministic if COPY/RENAME suffices ... emitting `key=` universally would
#     have built an invalid call and been misscored as 'the substitute did not help'." And the one
#     mechanical repair that does exist (deleting arguments the schema rejects) is strictly better at
#     the commitment gate, where the budget has not been charged yet. Re-dispatching after the
#     validator has already counted the call is a worse version of the same intervention.
#
#   POST_EXECUTION / REROUTE-as-transform
#     rewriting the feedback string before it enters `messages` IS implementable here, unlike in TB2.
#     It is withheld until driven, because the TB2 prototype's version of this cell reported
#     `executed=True` while appending its payload to the empty string.
#
# PRE_GENERATION / REPROMPT is DECLARED, and that is a deliberate departure from BFCL, which carries
# NOOP only there. BFCL's exclusion is measured -- one anchor scored -4.95 pp at that boundary and
# +3.63 pp with byte-identical text one boundary later -- but that measurement is BFCL's. Withholding
# the cell here would import another benchmark's result as an assumption about this one. The cost of
# declaring it is at most one arm spent confirming a known-shaped failure; the cost of withholding it
# is never finding out. The note travels with the declaration so nobody reads the cell as endorsed.
HOST = HostProfile(
    name=NAME,
    executable={
        IncisionPoint.PRE_GENERATION: frozenset({Action.NOOP, Action.REPROMPT}),
        IncisionPoint.POST_GENERATION_PRE_EXEC: frozenset({
            Action.NOOP, Action.REPROMPT, Action.SUPPRESS, Action.REROUTE,
        }),
        IncisionPoint.POST_EXECUTION: frozenset({Action.NOOP, Action.REPROMPT}),
    },
    notes=("Every cell is directive-level, not live: no executor has fired in an episode yet. "
           "REROUTE at POST_EXECUTION is withheld -- substitution there needs semantic argument "
           "synthesis, and the one mechanical repair is strictly better at the commitment gate. "
           "REROUTE-as-transform at POST_EXECUTION is withheld pending a driven implementation. "
           "PRE_GENERATION/REPROMPT is declared rather than withheld: BFCL's -4.95 pp result at "
           "that boundary is BFCL's, and importing it would be an assumption, not a measurement."),
)


# ------------------------------------------------------------------------------------------------
# Case-id grammar
# ------------------------------------------------------------------------------------------------
#
# `response_generator.py:311` builds every episode id as `f"{input_sample['id']}_{i}"`, and
# `DialogueConstraintChecker.__init__` reads the query id back as `int(sample["id"].split("_")[0])`.
# So the grammar is `<query_id>_<repeat_index>` and nothing else.
#
# ONE DIFFERENCE FROM BFCL THAT AFFECTS THE RUN PROTOCOL: the shard axis is NOT in the id. BFCL
# encodes backend and scenario in the case id, so `cell_of` can derive the shard from the id alone.
# CCTU's `data_source` is a separate field on the sample, so a sharded runner must carry it
# alongside; there is deliberately no function here that pretends to recover it from the id.
_CASE_RE = re.compile(r"^(\d+)_(\d+)$")


class UnknownCaseId(ValueError):
    """Raised when an episode id does not match the corpus grammar.

    A distinct type so a caller with a tolerant contract can catch exactly this. It RAISES rather
    than guessing, which is the lesson `benchmarks/bfcl_v4/adapter.py` records: three of five
    implementations of one id parser silently filed an unrecognised id into a real backend, because
    the parse was a chain with no failure branch.
    """


class CaseId(NamedTuple):
    """A parsed episode id. `query_id` indexes `data/check_code/`; `repeat` is the replicate."""

    raw: str
    query_id: int
    repeat: int


def parse_case_id(case_id: str) -> CaseId:
    """Parse an episode id, or raise `UnknownCaseId`. The single place this grammar lives."""
    match = _CASE_RE.match(str(case_id or ""))
    if not match:
        raise UnknownCaseId(
            f"{case_id!r} does not match the CCTU episode grammar (<query_id>_<repeat>). Not "
            f"defaulting: an unrecognised id filed under a real query is worse than a loud failure."
        )
    return CaseId(raw=str(case_id), query_id=int(match.group(1)), repeat=int(match.group(2)))


def query_id_of(case_id: str) -> int:
    """The corpus query this episode replicates. Raises `UnknownCaseId` -- never guesses."""
    return parse_case_id(case_id).query_id


# ------------------------------------------------------------------------------------------------
# 1. event normalization
# ------------------------------------------------------------------------------------------------
def normalize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """One turn -> a boundary-tagged, framework-neutral record.

    Accepts a live turn (`message` plus the `feedback` list the harness built) or an already-flat
    record, so live middleware and a recorded `response.jsonl` both work. Every CCTU-specific decode
    happens HERE, via `cctu_signals`' classifiers, so no violation string crosses into the core.

    WHAT THIS CANNOT PRODUCE, and must not pretend to: the constraint-pressure fields. They are facts
    about (this proposal x the live checker), so they come from `cctu_state.carried_state` and are
    merged by `observable_state`. A `normalize_event` that guessed them from the turn alone would be
    reporting budget state it cannot see.

    `tool_calls_raw` is carried through deliberately. `cctu_state` needs the calls in the harness's
    own shape to hand them to upstream's handlers, and re-deriving them from the normalized fields
    would be a second decode of the same thing.
    """
    message = event.get("message")
    if message is None:
        message = {k: event.get(k) for k in ("content", "tool_calls") if k in event}
    calls = _tool_calls_of(message, event)
    content = _content_of(message, event)

    feedback = list(event.get("feedback") or ())
    classes = violation_classes_in(feedback)
    results = [m.get("content") for m in feedback
               if isinstance(m, Mapping) and m.get("role") == "tool"]

    has_error = any(is_tool_fault(r) for r in results)
    vacuity = None if has_error else next(
        (k for k in (result_vacuity_kind(r) for r in results) if k is not None), None)

    names = [str((c.get("function") or {}).get("name") or "") for c in calls]
    dims = dimensions_in(classes)

    rec: dict[str, Any] = {
        "boundary": event.get("boundary"),
        # the decision just made
        "has_generation": bool(event.get("has_generation", message is not None)),
        "proposes_tool_call": bool(calls),
        "n_tool_calls": len(calls),
        "n_distinct_tool_names": len({n for n in names if n}),
        "tool_name": names[0] if names else None,
        "content": content,
        "content_length": len(content),
        # the harness's own shapes, for cctu_state and for telemetry
        "tool_calls_raw": calls,
        "results": results,
        # what came back
        "violation_class": classes[0] if classes else None,
        "violation_classes": tuple(classes),
        "n_violations_this_turn": len(classes),
        "violation_is_resource": "resource" in dims,
        "violation_is_behavior": "behavior" in dims,
        "violation_is_response": "response" in dims,
        "violation_is_arguments": "arguments" in dims,
        "result_is_error": has_error,
        "result_is_empty": vacuity is not None,
        "result_vacuity_kind": vacuity,
    }
    return rec


def events_from_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A recorded episode's `messages` -> TWO boundary-tagged events per turn.

    WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL
    -------------------------------------------
    One iteration of `sample_process`'s loop produces one record containing the generation, the
    validator's verdict AND the tool results. Fed to `normalize_event` as a single event, it carries
    feedback, so `boundary_key` places it at POST_EXECUTION -- and an episode of such turns derives
    exactly ONE boundary. `scripts/check_adapter.py`'s check 3 then warns, correctly, that "WHERE is
    not a searchable coordinate for this trajectory", which is the same defect the template calls the
    most valuable lesson from the first port: a residual that derives one boundary can never be
    observed to move earlier.

    The two moments are real -- `ANCHOROPT.md` maps them to concrete lines -- so the trajectory has
    to record both. A live wrapper emits them as it passes each point; this function reconstructs
    them from the shipped `response.jsonl`, which is the only artifact a re-mine or a `--replay-from`
    run has.

    THE POST_EXECUTION EVENT IS EMITTED EVEN WHEN NOTHING CAME BACK. A clean final turn produces no
    feedback at all, and skipping it would make "the validator ran and reported nothing" invisible --
    leaving the successful turns with no post-execution observation to contrast the failing ones
    against. An absence that is never recorded cannot discriminate.
    """
    return [normalize_event(row) for row in raw_rows_from_messages(messages)]


def raw_rows_from_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The same expansion, stopping BEFORE normalization -- the canonical raw trajectory row.

    Split out so there is one declared row shape that both a live wrapper and a replay produce, and
    so `scripts/check_adapter.py --events` has something to feed `normalize_event` exactly once.
    Handing it already-normalized records would normalize them twice, and the second pass reads none
    of the keys the first pass wrote -- which would look like an adapter that cannot see its own
    trajectory.
    """
    turns: list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]] = []
    for msg in messages or ():
        if not isinstance(msg, Mapping):
            continue
        if msg.get("role") == "assistant":
            turns.append((msg, []))
        elif turns and msg.get("role") in ("tool", "user"):
            turns[-1][1].append(msg)

    rows: list[dict[str, Any]] = []
    for index, (message, feedback) in enumerate(turns):
        rows.append({"boundary": IncisionPoint.POST_GENERATION_PRE_EXEC.value,
                     "message": dict(message), "has_generation": True, "turn_index": index})
        rows.append({"boundary": IncisionPoint.POST_EXECUTION.value,
                     "message": dict(message), "feedback": [dict(f) for f in feedback],
                     "has_generation": True, "turn_index": index})
    return rows


def _tool_calls_of(message: Any, event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The turn's proposed calls, in the harness's own OpenAI-ish shape.

    Text-format extraction is NOT done here. `response_generator.extract_tool_calls_from_text` owns
    it, and duplicating it would fork a parser -- but note that on the text path the harness never
    executes the calls and never appends the feedback (`response_generator.py:147` and `:212`), so
    such an episode carries no tool results at all. `ANCHOROPT.md` records that as a defect to fix
    before cycle 0 rather than something for this function to paper over.
    """
    calls = None
    if isinstance(message, Mapping):
        calls = message.get("tool_calls")
    elif message is not None:
        calls = getattr(message, "tool_calls", None)
    if calls is None:
        calls = event.get("tool_calls")
    return [c for c in (calls or []) if isinstance(c, Mapping)]


def _content_of(message: Any, event: Mapping[str, Any]) -> str:
    """The turn's assistant text, normalized the way the response handlers score it.

    `_strip_think_keep_text` is upstream's, and it is what `get_feedback_if` applies before any
    response-length or format check (`constraint_checker/core.py:124`). Using the raw content instead
    would make `content_length` disagree with the constraint it exists to predict, on every episode
    from a thinking model.
    """
    raw = None
    if isinstance(message, Mapping):
        raw = message.get("content")
    elif message is not None:
        raw = getattr(message, "content", None)
    if raw is None:
        raw = event.get("content")
    try:
        from utils.constraint_checker.check_utils import _strip_think_keep_text
        return _strip_think_keep_text(raw if isinstance(raw, str) else "")
    except ImportError:              # readable without the benchmark package on the path
        return raw if isinstance(raw, str) else ""


# ------------------------------------------------------------------------------------------------
# 2. observable runtime state
# ------------------------------------------------------------------------------------------------
def observable_state(normalized: Mapping[str, Any],
                     carried: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The state Phi_CCTU may read at this boundary, plus any state carried forward.

    `carried` is EXPLICIT rather than implicit: a signal needing evidence from an earlier boundary, or
    from the live checker, must have it PASSED here. Cross-boundary dependence is then visible in the
    call instead of hidden in a mutable accumulator that a later read cannot audit -- and on this
    benchmark it matters more than usual, because the validator advances the budget mid-step, so the
    same field legitimately differs between the two POST_GENERATION sub-moments.

    Build `carried` with `cctu_state.carried_state(...)`. The defaults below exist so a state built
    without it is still EVALUABLE rather than raising -- but it will be missing every pressure field,
    which is why `cctu_state` is not optional in a real run.
    """
    state = dict(normalized)
    if carried:
        state.update(carried)
    state.setdefault("round_index", 0)
    state.setdefault("consecutive_violation_turns", 0)
    return state


# ------------------------------------------------------------------------------------------------
# 3. feasible actions -- U_H(l)
# ------------------------------------------------------------------------------------------------
def feasible_actions(boundary: IncisionPoint) -> frozenset[Action]:
    """What this runtime can execute at `boundary`. Never wider than the structural grid."""
    return HOST.executable_actions(boundary)


# ------------------------------------------------------------------------------------------------
# 4. boundary identification -- structural only; the ORDER comes from the trajectory
# ------------------------------------------------------------------------------------------------
#
# No stage taxonomy and no declared ordering. Core derives the order from the realized trajectory
# (`boundary_search.order_boundaries`), and a core module that shipped an ordering of named stages
# would have reintroduced the map whose removal is the reason boundary derivation exists.
def commits_to_answer(event: Mapping[str, Any]) -> bool:
    """Is this event the agent committing to a final answer rather than acting again?

    Upstream's own test, and it is exact rather than inferred: `is_final = (len(tool_calls) == 0)`
    at `response_generator.py:129`.
    """
    if not isinstance(event, Mapping):
        return False
    if not (event.get("has_generation") or "content" in event or "message" in event):
        return False
    return not bool(event.get("proposes_tool_call") or event.get("tool_calls"))


def is_decision(event: Mapping[str, Any]) -> bool:
    """Was this event a point where the agent made a consequential choice?

    **COMMITTING TO AN ANSWER IS A DECISION.** This is the most-repeated porting lesson in the repo
    and the one it is easiest to get wrong: requiring a proposed or executed CALL meant that an
    episode which answered without acting contributed no boundary at all, every residual derived
    exactly one boundary, and the backward search could never be observed to move. A decision point
    the trace records but the adapter cannot name is indistinguishable from a trajectory that
    contains no such decision.

    On CCTU that case is not marginal: an episode that answers immediately is a MIN-class failure,
    which is one of the largest classes in the corpus.
    """
    if not isinstance(event, Mapping):
        return False
    return bool(event.get("boundary")) or bool(event.get("has_generation")) or \
        bool(event.get("proposes_tool_call")) or bool(event.get("tool_calls")) or \
        bool(event.get("message")) or commits_to_answer(event)


def boundary_key(event: Mapping[str, Any]) -> str:
    """The decision point this event belongs to. Returns an `IncisionPoint` value, or "".

    Results are checked FIRST, so an event that both answers and carries results is placed by its
    results. A turn carrying feedback has been through the validator and the executor; a turn with a
    generation and nothing back is at the commitment gate.
    """
    declared = event.get("boundary") if isinstance(event, Mapping) else None
    if declared is not None:
        return getattr(declared, "value", str(declared))
    if not isinstance(event, Mapping):
        return ""
    if event.get("results") or event.get("feedback") or event.get("violation_class") is not None:
        return IncisionPoint.POST_EXECUTION.value
    if event.get("has_generation") or event.get("message") or event.get("tool_calls") or \
            event.get("proposes_tool_call"):
        return IncisionPoint.POST_GENERATION_PRE_EXEC.value
    # A commitment to answer is generated-but-undispatched: the answer can still be prevented, which
    # is the defining property of that incision point. Checked last so results win.
    if commits_to_answer(event):
        return IncisionPoint.POST_GENERATION_PRE_EXEC.value
    return ""


def boundary_from_key(key: str):
    """The inverse of `boundary_key`. Core owns no key->locus table, on purpose.

    An unrecognised key yields None and that boundary is simply not searched, rather than being
    coerced into a real one.
    """
    try:
        return IncisionPoint(str(getattr(key, "value", key)))
    except (ValueError, TypeError):
        return None


# Descriptive labels for reports and figures. NOT an algorithm input: nothing in core reads these,
# and deleting this map changes no search behaviour.
_DESCRIPTIVE_LABELS: Mapping[str, str] = {
    IncisionPoint.PRE_GENERATION.value: "pre-decision",
    IncisionPoint.POST_GENERATION_PRE_EXEC.value: "commitment gate",
    IncisionPoint.POST_EXECUTION.value: "post-outcome",
}


def label_for(key: str) -> str:
    return _DESCRIPTIVE_LABELS.get(str(getattr(key, "value", key)), "")


def actions_in(event: Mapping[str, Any]) -> tuple[str, ...]:
    """The tool names this event proposed, in this trace's shape. Core asks; the adapter answers.

    Paired with `commits_to_answer`, this lets core ask the generic question "did the trajectory
    commit without acting first" without knowing how this runtime spells a call.
    """
    if not isinstance(event, Mapping):
        return ()
    names = []
    for call in event.get("tool_calls_raw") or event.get("tool_calls") or ():
        if isinstance(call, Mapping):
            names.append(str((call.get("function") or {}).get("name") or ""))
    return tuple(n for n in names if n)


# ------------------------------------------------------------------------------------------------
# 5. the signal surface, and Phi expansion
# ------------------------------------------------------------------------------------------------
#
# Expanded signals live in a SEPARATE table from the shipped ones, so that `declared_signals()`
# reports one Phi to the search while any artifact can still tell a synthesized condition from a
# pre-registered one -- a recovery claim made with an expanded Phi is a different claim. Nothing is
# written into `SIGNALS`, so a policy referencing a shipped signal cannot change meaning between
# rounds. Empty at import: expansion is a per-session act, never shipped state.
EXPANDED_SIGNALS: dict[str, Any] = {}
EXPANDED_BOUNDARIES: dict[str, frozenset] = {}
EXPANDED_PROVENANCE: dict[str, str] = {}
# `name -> the signal_lang expression`, for the expanded signals that HAVE one. Separate from
# EXPANDED_SIGNALS because the two answer different questions: that one is "can this fire here",
# this one is "can this be written to disk and fire in another process". A name present in the first
# and absent from the second is a signal that works this session and cannot be persisted, which a
# caller must be able to detect rather than discover when an arm refuses to start.
EXPANDED_EXPRESSIONS: dict[str, dict[str, Any]] = {}


def declared_signals() -> tuple[str, ...]:
    """Phi: the shipped signals plus anything expansion installed this session."""
    return tuple(sorted(set(SIGNALS) | set(EXPANDED_SIGNALS)))


def signal_aliases(signal: str) -> tuple[str, ...]:
    """Declared prose phrases describing `signal`, for expressibility matching ONLY.

    Part of this benchmark's vocabulary, so the generic core never hard-codes a domain word -- it
    asks the runtime. Returns () for a signal with none.
    """
    return tuple(SIGNAL_ALIASES.get(signal, ()))


def signal_boundaries(signal: str) -> frozenset[IncisionPoint]:
    """The incision points at which `signal` is observable. Unknown names RAISE.

    Expanded signals are answered FIRST and separately: they carry no `SIGNAL_BOUNDARIES` entry, so
    falling through into the shipped path would raise on a lookup that does not exist for them and an
    expanded signal would be undetectably unusable. What is returned is WHERE THE PREDICATE WAS
    VALIDATED -- not a truth value. The BFCL version of this function once returned
    `bool(predicate(state))` for an expanded signal, referencing a `state` it does not take, so every
    expanded signal raised NameError into a caller that swallowed it and substituted [] -- silently
    telling the proposer each synthesized signal was observable nowhere.
    """
    if signal in EXPANDED_SIGNALS:
        return EXPANDED_BOUNDARIES.get(signal, frozenset())
    if signal not in SIGNALS:
        raise KeyError(f"{NAME}: no evaluator for signal {signal!r}; declared: "
                       f"{sorted(set(SIGNALS) | set(EXPANDED_SIGNALS))}")
    names = SIGNAL_BOUNDARIES.get(signal, frozenset())
    return frozenset(p for p in IncisionPoint if p.value in names)


def evaluate_signal(signal: str, state: Mapping[str, Any],
                    params: Mapping[str, Any] | None = None) -> bool:
    """Evaluate one Phi_CCTU signal. Unknown names RAISE rather than returning False.

    A missing evaluator that returns False is indistinguishable from a signal that did not fire,
    which is how an arm silently becomes the control arm. The TB2 prototype had exactly that defect
    -- a declared signal with no evaluator -- and it killed a run mid-cluster.

    Parameters are validated against the DECLARED schema, so a proposal omitting a required one is
    pruned rather than run with a silent default, and an unknown parameter name is refused rather
    than ignored. `bool` is rejected where an `int` is required, because `True` passes
    `isinstance(x, int)` and a boolean threshold is not the signal that was proposed.
    """
    if signal in EXPANDED_SIGNALS:
        return bool(EXPANDED_SIGNALS[signal](state or {}))
    if signal not in SIGNALS:
        raise KeyError(
            f"{NAME}: no evaluator for signal {signal!r}; declared: {sorted(declared_signals())}")
    params = dict(params or {})
    schema = SIGNAL_PARAMS.get(signal, {})
    for key, (typ, required) in schema.items():
        if key not in params:
            if required:
                raise ValueError(f"{NAME}: signal {signal!r} requires param {key!r}")
            continue
        if typ is int and isinstance(params[key], bool):
            raise TypeError(f"{NAME}: signal {signal!r} param {key!r} must be int, got bool")
        if not isinstance(params[key], typ):
            raise TypeError(f"{NAME}: signal {signal!r} param {key!r} must be {typ.__name__}, "
                            f"got {type(params[key]).__name__}")
    unknown = set(params) - set(schema)
    if unknown:
        raise ValueError(f"{NAME}: signal {signal!r} unknown param(s) {sorted(unknown)}")
    return bool(SIGNALS[signal](state, params))


SYNTH_POOL_ENV = "ANCHOROPT_CCTU_SYNTH_POOL"
SYNTH_POOL_STAGES = ("boolean", "all")


def synthesis_pool_stage() -> str:
    """Which stage of the atom ladder `synthesis_fields(boundary)` is offering. Recorded, not guessed.

    `boolean` (default) offers the boolean observables; `all` adds the numeric ones. A round states
    which stage it proposed from, the same way it states whether `--exhaust` was on, because the two
    stages do not offer the same candidate space.
    """
    import os
    stage = os.environ.get(SYNTH_POOL_ENV, "boolean").strip().lower()
    if stage not in SYNTH_POOL_STAGES:
        raise ValueError(f"{SYNTH_POOL_ENV}={stage!r} is not one of {SYNTH_POOL_STAGES}")
    return stage


def synthesis_fields(boundary: Any = None) -> Mapping[str, Any]:
    """TYPED DECLARATIONS core may search over at `boundary`. Declarations, NOT candidates.

    Filters by boundary, which is not cosmetic: `structured_search.MAX_SYNTHESIS_CANDIDATES` is 40
    and conjunctions are formed from the top 12 discriminating atoms, so a field declared where its
    fact does not exist spends budget a real candidate could have used. 7 fields are readable before
    generation, 25 at the commitment gate, all 33 after execution.

    THE ATOM BUDGET IS SPENT BY FIELD TYPE, NOT BY WHAT MATTERS, AND THAT IS WHY THERE IS A LADDER
    ------------------------------------------------------------------------------------------------
    `signal_grammar.atoms_for` emits TWO atoms for a boolean field and one PER DISTINCT OBSERVED
    VALUE for a numeric one, then `synthesize` pairs only the 12 atoms closest to an even split. One
    numeric field can therefore fill that window by itself, and the window is what decides whether a
    conjunction is ever offered. Measured at the commitment gate on `train_baseline`, 484 gate states:

        full pool (25 fields)      40 predicates,  0 conjunctions
        boolean pool (14 fields)   40 predicates, 24 conjunctions

    The conjunction `repeated_identical_call AND per_tool_over_cap` -- the only condition on this
    corpus measured at ZERO loss exposure, 0 of 37 already-succeeding episodes -- appears in the
    second and cannot appear in the first. Both atoms discriminate at the gate (36 and 37 of 223
    turns); nothing about the fact was missing. The threshold atoms crowded it out.

    So the pool is a LADDER and the default is its first rung: boolean observables first, numeric
    thresholds when no boolean policy remains. That is `docs/THE_LOOP.md`'s block-coordinate order --
    hold the coarser vocabulary and optimise the policy, refine only when the coarse one is exhausted
    -- applied to the atom pool rather than to Phi. Escalate with `ANCHOROPT_CCTU_SYNTH_POOL=all`,
    which a round records.

    A BOUNDARY IS THE SYNTHESIS PATH; NO BOUNDARY IS THE VALIDATION PATH, AND THEY DIFFER ON PURPOSE
    ------------------------------------------------------------------------------------------------
    `structured_search` asks with a boundary, to build candidates. `cctu_middleware.compile_signal`
    and `cctu_screens` ask WITHOUT one, to decide whether a predicate already written down refers to
    declared facts. The ladder must not narrow that second question: a spec from an earlier round
    references whatever was declared when it was proposed, and a round already on disk has to stay
    loadable and scoreable. So the no-boundary answer is always the full typed set.
    """
    fields = dict(_caps.all_fields())
    if boundary is None:
        return fields
    want = str(getattr(boundary, "value", boundary))
    here = {n: f for n, f in fields.items() if want in f.boundaries}
    if synthesis_pool_stage() == "all":
        return here
    rung = {n: f for n, f in here.items() if getattr(f, "type", None) is bool}
    return rung or here


def states_at(boundary: Any, states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project observed states onto ONE boundary's information set. Core FAILS CLOSED without this.

    `synthesis_fields` declares WHERE each fact exists; this is the runtime half of the same rule.
    Two things happen here, and only one of them filters rows:

      ROWS    `replay_episode` emits TWO states per turn -- the commitment gate and the post-outcome
              moment -- and `optimize_residual` is handed both, mixed, for the whole residual. A
              predicate synthesized at the gate but validated on post-execution rows can look
              discriminating on a result that does not exist yet, and then never fire at runtime.
              Rows are selected with `boundary_key`, the same function that derived the boundary, so
              the projection cannot disagree with the search about where a state belongs.

      FIELDS  a gate row still CARRIES the fields whose fact only exists after dispatch, because
              `observable_state` merges one turn's normalized record with its carried summaries.
              Dropping exactly the fields `cctu_capabilities` does not declare at `want` is what
              stops synthesis reading a result one boundary early. A field declared at BOTH -- a
              round index, a proposed payload -- is a carried fact and is KEPT; only the 8 that
              exist nowhere before dispatch are removed.

    IT FILTERS AND NEVER INVENTS: at most one row out per row in, and no key is ever added.
    `anchoropt.testing` fails an adapter whose projection grows the set, because a projection that
    invents states manufactures evidence for a boundary nobody observed.

    Returns [] for a boundary this trajectory never reaches -- `boundary_key` places every CCTU turn
    at the gate or after execution, so PRE_GENERATION is empty here. Core then refuses to search
    there rather than falling back to another boundary's states, which is the intended failure.
    """
    want = str(getattr(boundary, "value", boundary))
    here = frozenset(_caps.fields_at(want))
    if not here:
        return []
    # Declared SOMEWHERE but not at `want`: the fact does not exist yet at this boundary.
    not_yet = frozenset(_caps.all_fields()) - here
    out: list[dict[str, Any]] = []
    for st in states or ():
        if not isinstance(st, Mapping) or boundary_key(st) != want:
            continue
        out.append({k: v for k, v in st.items() if k not in not_yet})
    return out


# ------------------------------------------------------------------------------------------------
# Serializing a SYNTHESIZED predicate, so an expanded signal survives the process that invented it
# ------------------------------------------------------------------------------------------------
#
# Core hands `install_signal` a CLOSURE and no `expr`: `signal_expansion.expand_and_resume_predicates`
# calls `runtime.install_signal(prop.name, lambda st, _p=pred: bool(_p.evaluate(st)), ...)`, and there
# is no `expr=` anywhere in core. A closure cannot be written to a spec file, so without this the
# round is stuck: `expanded_expression` returns None for every synthesized signal and
# `cycle1_propose` correctly refuses to write a spec naming a signal a fresh process cannot resolve.
#
# So cctu wraps its own end of the seam and derives the expression from the predicate core wrapped.
# Two vocabularies have to meet, and neither is ours to change:
#
#   signal_grammar   Atom(field, op in {truthy, falsy, equals, lt, gt}, value) and Conjunction(terms)
#   signal_lang      {"field","op","value"} leaves over a CLOSED op set, combined with all/any/not
#
# The map below is the whole translation. An op absent from it yields None rather than a guess: a
# wrong expression would install a signal that measures one thing and ships another, which is worse
# than refusing, and the refusal path already exists and is honest.
_GRAMMAR_TO_LANG_OP: Mapping[str, str] = {
    "equals": "eq",
    "lt": "lt",
    "gt": "gt",
}

# Leaf ops that carry an operand. The truthiness and null-test ops take none, and
# `signal_lang` REFUSES a value on them, so emitting one would make the expression illegal.
_LANG_OPS_WITH_VALUE = frozenset({"eq", "lt", "gt"})


def _truthiness_leaf(field: str, *, negated: bool) -> dict[str, Any] | None:
    """`truthy`/`falsy` on `field` as a legal `signal_lang` leaf, or None if it cannot be one.

    The grammar's `truthy` is `bool(v)` over ANY field; `signal_lang` splits that intent across ops
    with declared type requirements, so the translation depends on what the field is declared to be.

      bool            `is_true` / `is_false`, which is `bool(actual) is True/False` -- the same test.
      str WITH enum   `is_none`, negated with the `not` combinator for the positive case. Exact, and
                      it is the DECLARATION that makes it so: the field's domain is its enum plus
                      None, and no enum member is falsy, so `not bool(v)` and `v is None` cannot
                      disagree on any value the field may take. Checked rather than assumed -- an
                      enum admitting a falsy member falls through to None below.

                      `not` over `is_none` rather than the op that spells the same thing: the op
                      table is core's vocabulary, and restating one of its names here is what
                      `scripts/check_adapter.py` check 7 exists to catch. The combinator is the
                      same expression with nothing duplicated.
      anything else   None. `is_true` on a non-bool field is refused by `signal_lang` anyway, and an
                      int whose falsiness means `0` would need a value-carrying op the grammar did
                      not ask for. Refusing leaves the signal usable this session and unshippable,
                      which `expanded_expression` already reports honestly.
    """
    declared = synthesis_fields().get(str(field))
    if declared is None:
        return None
    if declared.type is bool:
        return {"field": str(field), "op": "is_false" if negated else "is_true"}
    enum = tuple(getattr(declared, "enum", ()) or ())
    if declared.type is str and enum and not any(not bool(v) for v in enum):
        absent = {"field": str(field), "op": "is_none"}
        return absent if negated else {"not": absent}
    return None


def _leaf_of_atom(atom: Any) -> dict[str, Any] | None:
    """One `signal_grammar.Atom` -> one `signal_lang` leaf, or None if it does not translate."""
    field = getattr(atom, "field", None)
    if not field:
        return None
    grammar_op = str(getattr(atom, "op", ""))
    if grammar_op in ("truthy", "falsy"):
        return _truthiness_leaf(str(field), negated=grammar_op == "falsy")
    op = _GRAMMAR_TO_LANG_OP.get(grammar_op)
    if op is None:
        return None
    leaf: dict[str, Any] = {"field": str(field), "op": op}
    if op in _LANG_OPS_WITH_VALUE:
        value = getattr(atom, "value", None)
        if value is None:
            return None
        leaf["value"] = value
    return leaf


def _expr_of_predicate(predicate: Any) -> dict[str, Any] | None:
    """A synthesized predicate -> the `signal_lang` expression meaning the same thing, or None.

    A `Conjunction` carries `.terms`; an `Atom` carries `.field` directly. A single-term conjunction
    is emitted as a bare leaf rather than a one-element `all`, which compiles identically and keeps
    the persisted form the same shape the screens already read.
    """
    terms = getattr(predicate, "terms", None)
    if terms is not None:
        leaves = [_leaf_of_atom(t) for t in terms]
        if not leaves or any(leaf is None for leaf in leaves):
            return None
        return {"all": list(leaves)} if len(leaves) > 1 else dict(leaves[0])   # type: ignore[arg-type]
    return _leaf_of_atom(predicate)


def _predicate_behind(predicate: Any) -> Any | None:
    """The synthesized predicate itself, unwrapped from the closure core installs it as.

    Core's lambda keeps the predicate in a DEFAULT argument (`_p=pred`); a different wrapping would
    keep it in a closure cell. Both are checked, and the object is accepted only if it actually
    translates -- so this recognizes a predicate by its SHAPE rather than trusting a position.
    """
    if _expr_of_predicate(predicate) is not None:
        return predicate
    carried = list(getattr(predicate, "__defaults__", None) or ())
    for cell in getattr(predicate, "__closure__", None) or ():
        try:
            carried.append(cell.cell_contents)
        except ValueError:                       # an empty cell, mid-definition
            continue
    for candidate in carried:
        if _expr_of_predicate(candidate) is not None:
            return candidate
    return None


def _derived_expression(predicate: Any) -> dict[str, Any] | None:
    """The persistable form of a predicate core installed without one, or None if there is none.

    VALIDATED HERE, against the same compiler the runner uses. `signal_lang.compile_signal` is what
    `ControllerSpec._compile` calls at load time, so an expression that does not compile now is a
    spec that fails to start later -- and storing it would move the failure from proposal time, where
    the round can see it, to arm time, where it looks like a broken run. It also rejects a
    conjunction whose fields never coexist at one boundary, because `compile_signal` intersects the
    referenced fields' declared boundaries and an empty intersection is not observable anywhere.
    """
    found = _predicate_behind(predicate)
    if found is None:
        return None
    expr = _expr_of_predicate(found)
    if not expr:
        return None
    from anchoropt.learning.signal_lang import SignalSpecError, compile_signal
    try:
        compile_signal(str(getattr(found, "name", lambda: "probe")() or "probe"),
                       expr, fields=synthesis_fields())
    except SignalSpecError:
        return None
    except Exception:                            # a shape the compiler did not expect: refuse
        return None
    return expr


def install_signal(name: str, predicate, *, boundary: Any = None, provenance: str = "",
                   expr: Mapping[str, Any] | None = None) -> None:
    """Add a validated predicate to Phi. Refuses to shadow a shipped signal.

    `boundary` is the incision point expansion VALIDATED the predicate against, and it is RETAINED.
    An earlier version of this hook elsewhere accepted and discarded it, which left
    `signal_boundaries` with nothing to answer for exactly the signals expansion creates. Absent a
    boundary the signal is recorded as observable nowhere, which is honest: it was never validated
    anywhere.

    `expr` IS THE SERIALIZABLE FORM, and retaining it is what makes an expanded signal survive the
    process that invented it. A predicate is a closure: a spec naming it in `signal` validates in the
    proposer -- where this dict holds the name -- and is REFUSED by a fresh process, because
    `ControllerSpec.validate` checks the name against the declared vocabulary. That is not
    hypothetical: 312 of 330 expanded candidates on granite and 300 of 318 on qwen were written that
    way and could not be loaded, which would have surfaced as every arm failing to start.

    Optional, because a runtime may install a predicate that no expression can express. In that case
    the signal still works IN THIS PROCESS and `expanded_expression` returns None, so a caller that
    needs to persist it can refuse rather than write a spec that cannot be loaded.
    """
    if name in SIGNALS:
        raise ValueError(f"{name!r} is a shipped signal; expansion may not redefine it")
    EXPANDED_SIGNALS[name] = predicate
    # Core supplies no `expr`, so derive one from the predicate it wrapped. An explicit `expr` from a
    # caller that HAS one still wins: this fills a gap, it does not override a declaration.
    if not expr:
        expr = _derived_expression(predicate)
    if expr:
        EXPANDED_EXPRESSIONS[name] = dict(expr)
    EXPANDED_PROVENANCE[name] = provenance or "(no provenance recorded)"
    points = boundary if isinstance(boundary, (tuple, list, set, frozenset)) else \
        ((boundary,) if boundary is not None else ())
    EXPANDED_BOUNDARIES[name] = frozenset(
        p for p in (boundary_from_key(b) for b in points) if p is not None)


def expanded_signals() -> tuple[str, ...]:
    """Signals added by expansion this session, distinguishable from the shipped four."""
    return tuple(sorted(EXPANDED_SIGNALS))


def expanded_expression(name: str) -> dict[str, Any] | None:
    """The persistable expression for an expanded signal, or None if it has none.

    None means "this condition cannot be written down", which a caller that persists specs must treat
    as a refusal rather than a missing optional field.
    """
    expr = EXPANDED_EXPRESSIONS.get(str(name))
    return dict(expr) if expr else None


def reset_expanded_signals() -> None:
    """Drop every expanded signal, so one experiment cannot inherit another's Phi."""
    EXPANDED_SIGNALS.clear()
    EXPANDED_BOUNDARIES.clear()
    EXPANDED_PROVENANCE.clear()
    EXPANDED_EXPRESSIONS.clear()


def register_signal(name: str, fn, params: Mapping[str, tuple[type, bool]] | None = None,
                    *, boundaries: Sequence[str] = ()) -> None:
    """Register a signal as part of the DECLARED vocabulary, not as an expansion.

    Refuses to overwrite: silently replacing a frozen definition would invalidate every measurement
    taken under the old one. Adding a signal touches no action implementation, which is one of the
    integration's generality requirements.
    """
    if name in SIGNALS:
        raise ValueError(f"{NAME}: signal {name!r} already registered; frozen definitions must not "
                         f"be replaced")
    SIGNALS[name] = fn                                        # type: ignore[index]
    SIGNAL_PARAMS[name] = dict(params or {})                  # type: ignore[index]
    if boundaries:
        SIGNAL_BOUNDARIES[name] = frozenset(str(b) for b in boundaries)   # type: ignore[index]


def probe_params(signal: str) -> Mapping[str, Any]:
    """Parameters the search may use to EVALUATE `signal` while enumerating candidates.

    Empty for all four, and that costs nothing: none of them has a REQUIRED parameter. The table
    exists because the other benchmark shipped it non-empty and had to empty it -- a probe default
    exports one measured number into every future proposal, and a discovery claim made with that
    value in the room is a lookup.
    """
    return dict(SIGNAL_PROBE_PARAMS.get(signal, {}))


def parameter_domains(signal: str):
    """The searchable domain of each tunable parameter of `signal`, as `ParameterDomain` objects.

    Returns () for a signal with nothing to tune, which makes it a DETERMINISTIC policy rather than a
    defective one.
    """
    from anchoropt.learning.policy_class import ParameterDomain

    out = []
    for pname, spec in (SIGNAL_PARAM_DOMAINS.get(signal) or {}).items():
        out.append(ParameterDomain(
            name=pname, kind=spec.get("kind", str),
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

    return PolicyClass.PARAMETERIZED if parameter_domains(signal) else PolicyClass.DETERMINISTIC


# ------------------------------------------------------------------------------------------------
# 6. ACTION GROUNDING -- eta_mu per operator, enumerated from this runtime's real capabilities
# ------------------------------------------------------------------------------------------------
#
# One hook per canonical operator. Each returns a LIST of groundings; every element becomes a separate
# evaluation arm, because choosing among them is measurement's job and not the adapter's. Returning []
# means the family is INFEASIBLE here, and `action_contract.instantiate` reports the missing
# requirement -- it never invents a tool, an argument or a surface, and never downgrades an
# ungroundable substitute into a reprompt.
#
# GROUNDED ON STRUCTURE, NEVER ON A SIGNAL NAME. `adapters/adapter_template.py` records the bug: a
# two-name whitelist meant no synthesized signal could ever compose with SUPPRESS, so the search
# produced zero arms while the action was admissible and an executor existed -- indistinguishable
# from an absent primitive. The only signal-dependent check below is that the signal is DECLARED at
# all, because an undeclared name is not a signal.
def _known(signal: str) -> bool:
    return signal in SIGNALS or signal in EXPANDED_SIGNALS


def ground_reprompt(signal: str, boundary: Any) -> list[dict]:
    """eta_mu for REPROMPT: instruction content plus a retry budget.

    Three SEMANTICALLY DISTINCT instructions, each its own arm, because which framing works is an
    empirical question and measuring one framing rejects that framing rather than the family.

    THE FRAMINGS ARE CHOSEN TO SAY WHAT THE ENVIRONMENT DOES NOT. CCTU already reprompts on every
    violation -- a non-final turn's violations come back as tool messages, a final turn's as a user
    message, and the loop continues while any feedback exists. So a reprompt that restates the
    violation adds nothing. What the environment never volunteers is the state BEFORE a violation:
    how much budget is left, and which minimums are still unmet. Two of the three below carry that;
    the third is the pure self-check framing, included precisely because upstream measured
    self-refinement to be weak and a framing predicted to fail is still worth one arm.

    The text is declared runtime content rather than generated per round, so an arm is reproducible.
    A generator may supply more, and measurement still chooses.
    """
    if not _known(signal):
        return []
    return [
        {"variant": "state_remaining_budget",
         "eta": {"instruction": ("You have limited remaining interaction rounds and tool calls. "
                                 "Before acting again, decide which calls are still necessary and "
                                 "make them together rather than one at a time."),
                 "retry_budget": 1},
         "detail": "supplies budget pressure the environment only reports after a violation"},
        {"variant": "satisfy_pending_minimums",
         "eta": {"instruction": ("Some requirements of this task are not satisfied yet. Check the "
                                 "stated minimum number of rounds, tool calls and parallel calls, "
                                 "and satisfy them before giving a final answer."),
                 "retry_budget": 1},
         "detail": "names the MIN-class requirements, which are only reported once the answer commits"},
        {"variant": "recheck_constraints_before_answering",
         "eta": {"instruction": ("Before answering, re-read the constraints in the task description "
                                 "and check your drafted answer against each one."),
                 "retry_budget": 1},
         "detail": ("the pure self-check framing. Upstream measures self-refinement as weak, so this "
                    "is the arm most likely to fail -- which is why it is measured, not assumed")},
        # ADDED FOR THE RESPONSE-LENGTH RESIDUAL, and the gap it fills is concrete.
        #
        # The three framings above all tell the model to ACT MORE or to re-check; none of them asks
        # for a SHORTER answer. On the qwen control `max_length` is 420 violations per replicate and
        # `response_length_over_cap` reaches 100% of that mass with zero pure-downside episodes -- so
        # the condition was detectable and the repair was not expressible. Two of the existing texts
        # would have made it worse by pushing for more calls.
        #
        # NOTHING HERE IS PRIVILEGED INFORMATION. The response-length limit is published in the task
        # description the model already has, so restating that the draft is too long tells it nothing
        # the prompt did not. The cap itself is deliberately NOT interpolated: a template carrying a
        # number the anchor read from the checker would be a capability asymmetry, and the point is a
        # declared, reproducible instruction rather than a per-episode one.
        #
        # THIS TEXT WAS CHOSEN BY A HUMAN, which the round must record. `ground_reprompt` is declared
        # runtime content and this docstring already says "A generator may supply more, and
        # measurement still chooses" -- so adding a framing is within the established pattern, but
        # which framing was added is an input to the search and belongs in the round's FROZEN.md.
        #
        # OFFERED FOR EVERY SIGNAL, not only the length ones. Deciding per signal which instruction
        # "suits" it would be the grounding choosing the arm; the screens and the measurement decide.
        # The cost is one extra candidate per cell, against a screening pass that already narrows 279
        # candidates to 6.
        {"variant": "shorten_to_length_limit",
         "eta": {"instruction": ("Your drafted answer is longer than the response length this task "
                                 "allows. Give the same answer again, shorter: keep every fact the "
                                 "question asks for and remove preamble, restatement and "
                                 "explanation."),
                 "retry_budget": 1},
         "detail": ("asks for the one repair the other three cannot express. Two of them push toward "
                    "MORE calls, which is the opposite move on a response-length residual")},
        # ADDED FOR THE COST `shorten_to_length_limit` WAS MEASURED TO CARRY, which is the next
        # round's whole question.
        #
        # On `rounds/CCTU_QWEN1`, `shorten_to_length_limit` produced the largest reduction in the round
        # -- 156 `max_length` events, against a 60-event bar, and the largest falls on `identifiers`
        # and `punctuation` too -- and was REJECTED on harm: `acc` went 0 gains / 2 losses. It is the
        # only arm that lost an answer. The accepted arm separately introduced `min_length` violations
        # from a base of zero on `test`, so over-shortening is measured and not hypothetical.
        #
        # So the residual is reachable and the repair is not free. These two framings are the two
        # mechanisms by which a shortening instruction can destroy an answer, separated so measurement
        # can tell them apart rather than reject the family:
        #
        #   (a) the model REGENERATES and loses content it had already produced. Addressed by an EDIT
        #       framing -- keep the answer, delete around it -- which is a different operation from
        #       "say it again shorter", not a reworded version of it.
        #   (b) the model OVER-COMPRESSES, trading the answer away along with the preamble. Addressed
        #       by making the answer the part that cannot be cut, and naming a floor: being complete
        #       outranks being short.
        #
        # Neither interpolates a measured value, for the same reason `shorten_to_length_limit` does not:
        # the length limit is published in the task description the model already holds, and a template
        # carrying a number read from the checker would be a capability asymmetry.
        #
        # BOTH TEXTS WERE WRITTEN BY A HUMAN, from the measured failure of a previous arm. That is an
        # input to the search, not an output of it, and it belongs in the next round's FROZEN.md as
        # plainly as `shorten_to_length_limit`'s authorship belongs in CCTU_QWEN1's.
        {"variant": "trim_around_the_answer",
         "eta": {"instruction": ("Your drafted answer is longer than the response length this task "
                                 "allows. Keep your answer exactly as you wrote it and delete only "
                                 "the text around it -- preamble, restatement of the question, "
                                 "explanation of your reasoning. Do not rewrite the answer itself."),
                 "retry_budget": 1},
         "detail": ("an EDIT, not a regeneration. Tests whether the answers `shorten_to_length_limit` "
                    "lost were lost to rewriting rather than to shortening")},
        {"variant": "answer_first_completeness_floor",
         "eta": {"instruction": ("Your drafted answer is longer than the response length this task "
                                 "allows. Put the direct answer to the question first and stop there. "
                                 "Every fact the question asks for must still be present: if you "
                                 "cannot fit them all, keep them and cut everything else."),
                 "retry_budget": 1},
         "detail": ("shortens but names completeness as the term that outranks length. Tests whether "
                    "the lost answers came from over-compression rather than from rewriting")},
    ]


def ground_suppress(signal: str, boundary: Any) -> list[dict]:
    """eta_mu for SUPPRESS: which operation is cancelled, and what preserves safety.

    The boundary IS part of the requirement: after execution there is nothing left to cancel.

    TWO VARIANTS, AND THEY ARE NOT INTERCHANGEABLE -- they differ in who pays. This is the sharpest
    runtime-specific fact in this adapter:

      drop_call           removes the call from the proposed message BEFORE the validator sees it, so
                          `callTimes` is never charged. That is the only form that helps a budget
                          failure. Its hazard: if the turn proposed exactly one call, dropping it
                          makes `len(tool_calls) == 0`, the harness reads the turn as FINAL, and the
                          terminal handlers judge content that was written for a tool turn. So the
                          preservation clause is a precondition, not a reassurance.
      withhold_execution  leaves the call in place, so the turn stays non-terminal and the budget IS
                          charged, and substitutes the observation instead. This is the efficiency
                          shape (`docs/EFFICIENCY_CLASS.md`): behaviour-preserving, judged on work
                          removed rather than on accuracy, where 0 pp is a PASS.

    Offering both as arms is the contract working as intended. Picking one here would be the adapter
    deciding an empirical question, and they optimize different objectives.
    """
    if not _known(signal):
        return []
    if str(getattr(boundary, "value", boundary)) != IncisionPoint.POST_GENERATION_PRE_EXEC.value:
        return []
    return [
        {"variant": "drop_call",
         "eta": {"suppressed_operation": "the proposed tool call this signal implicates",
                 "preservation": ("the turn must retain at least one other proposed call, so that "
                                  "dropping this one cannot convert a tool turn into a final answer"),
                 "charges_budget": False},
         "detail": "removes the call before the validator counts it; the budget is preserved"},
        {"variant": "withhold_execution",
         "eta": {"suppressed_operation": "the execution of the proposed tool call",
                 "preservation": ("the call stays in the message so the turn's kind is unchanged, "
                                  "and a substituted observation is returned in its place"),
                 "charges_budget": True},
         "detail": ("withholds execution only; behaviour-preserving, so it is judged on the "
                    "efficiency objective where a 0 pp accuracy delta is a pass")},
    ]


def ground_substitute_destinations(signal: str, boundary: Any) -> list[dict]:
    """eta_mu for REROUTE-as-substitute: destination, argument mapping, retry semantics.

    ONE grounding, and the narrowness is a finding rather than an omission. `docs/GENERALIZABILITY.md`
    is explicit that "a reroute is only deterministic if COPY/RENAME suffices", and on CCTU a
    different tool needs different, semantic arguments -- there is no mechanical mapping from one
    tool's arguments to another's, and inventing one is how a substitute gets misscored as "did not
    help" when what actually happened is that an invalid call was built.

    What IS mechanical is ARGUMENT REPAIR: `ToolArgsChecker` reports extra arguments by name
    (`args_checker.py:90-96`), and deleting exactly those keys is a pure deletion with nothing
    synthesized. `Action.REROUTE` covers "the same function with different arguments" by its own
    definition, so this is squarely the family.

    ON THE DESTINATION NOT BEING A LITERAL NAME. The tool is the one the model already chose, which
    already resolves; nothing new is being attested. That distinction matters because the failure this
    project records is admitting a destination on "11/11 clean" where clean meant *did not error* and
    9 of the 11 returned nothing. There is no such risk in keeping a destination the model picked --
    the attestation question does not arise, because the destination is not changing.
    """
    if not _known(signal):
        return []
    if str(getattr(boundary, "value", boundary)) != IncisionPoint.POST_GENERATION_PRE_EXEC.value:
        return []
    return [
        {"variant": "drop_extra_arguments",
         "eta": {"destination": "the tool the model proposed, unchanged",
                 "argument_mapping": ("delete exactly the argument names the tool's own schema does "
                                      "not declare; every retained argument stays byte-identical"),
                 "retry_semantics": "replace"},
         "detail": ("repairs a schema-invalid call mechanically, before the validator charges the "
                    "budget for it. Nothing is synthesized -- only deletion")},
    ]


def ground_transforms(signal: str, boundary: Any) -> list[dict]:
    """eta_mu for REROUTE-as-transform. **Deliberately empty, and it is not an oversight.**

    The writable surface exists: the feedback string could be rewritten before it enters `messages`
    (between `response_generator.py:190` and `:213`). It is withheld until a boundary-specific
    implementation has been DRIVEN, because the TB2 prototype's version of this cell read a field
    that does not exist at earlier boundaries and reported `executed=True` while appending its
    payload to the empty string (`docs/CONSUMER_BOUNDARY_RULE.md`).

    Returning [] means `instantiate` reports TRANSFORM_INFEASIBLE with the missing requirements
    named, so the exclusion appears in the certificates rather than as a shrunken grid nobody can
    explain. Re-enabling it is a change here and nowhere in the core.
    """
    return []


# ------------------------------------------------------------------------------------------------
# 7. EXECUTOR REGISTRY -- what this host can ACTUALLY run, per (boundary, action)
# ------------------------------------------------------------------------------------------------
#
# `U_H(l)` declares which actions are admissible. That is necessary and NOT sufficient: an action can
# be admissible while no executor can realize it, and a candidate reported "grounded" on the strength
# of the abstract declaration alone is not materializable.
#
# NO SIGNAL WHITELIST. Every executor here is `signal_agnostic`, unlike BFCL's, whose per-cell
# `signals` tuples its own comments identify as the defect that stopped synthesized signals from
# grounding. An executor implements HOW; phi owns WHEN, exclusively.
#
# `driven` IS FALSE EVERYWHERE, and step 5 is what changes that. Reading a cell as live because it is
# declared here is the mistake `HOST.notes` exists to prevent.
#
# `fixed` names what the executor IMPOSES. A candidate declaring a different value for one of these
# keys is REJECTED rather than coerced: coercion runs different semantics than the ones proposed,
# under the proposal's name.
EXECUTORS: Mapping[tuple[str, str], Mapping[str, Any]] = {
    ("pre_generation", "reprompt"): {
        "eta_slot": "instruction",
        "site": "appended to `messages` before `args.client.chat` (response_generator.py:183)",
        "fixed": {"adds_turn": False, "charges_round": False},
        "signal_agnostic": True,
        "driven": False,
        "detail": ("shapes the context of the next generation. No extra generation and no extra "
                   "round, because it replaces nothing -- it only adds to what the model reads"),
    },
    ("post_generation_pre_exec", "reprompt"): {
        "eta_slot": "instruction",
        "site": "between response_generator.py:188 and :190 -- BEFORE get_feedback",
        "fixed": {"adds_turn": False, "charges_round": False},
        "signal_agnostic": True,
        "driven": False,
        "detail": ("discards the proposed decision, injects the instruction and re-decides. The "
                   "validator has NOT run on the discarded proposal, so no round is charged -- which "
                   "is the whole reason this cell is preferable to the POST_EXECUTION one"),
    },
    ("post_execution", "reprompt"): {
        "eta_slot": "instruction",
        "site": "after the feedback is built, before response_generator.py:213",
        "fixed": {"adds_turn": True, "charges_round": True},
        "signal_agnostic": True,
        "driven": False,
        "detail": ("injects after the outcome and lets the loop continue. THIS COSTS A ROUND: the "
                   "validator has already advanced `checker.round`, and the extra turn advances it "
                   "again. For an episode whose failure is round exhaustion, this cell makes the "
                   "failure worse -- so it must not be proposed for that residual without the "
                   "displacement check in anchoropt/learning/exposure.py"),
    },
    ("post_generation_pre_exec", "suppress"): {
        "eta_slot": "(predicate-driven; the eta names the operation, not a template)",
        "site": "edits message['tool_calls'] between response_generator.py:188 and :190",
        "fixed": {},
        "signal_agnostic": True,
        "driven": False,
        "detail": ("two variants with different cost: `drop_call` removes the call before the "
                   "validator counts it, `withhold_execution` keeps it and substitutes the "
                   "observation. Both must leave the proposed call in the trace telemetry -- a "
                   "suppressed call that simply vanishes erases the evidence needed to evaluate the "
                   "suppression, which recurred four times on the other benchmark"),
        "variants": {"drop_call": {"charges_budget": False},
                     "withhold_execution": {"charges_budget": True}},
    },
    ("post_generation_pre_exec", "reroute"): {
        "eta_slot": "argument_mapping",
        "site": "rewrites message['tool_calls'][i]['function']['arguments'] before the validator",
        "fixed": {"retry_semantics": "replace"},
        "signal_agnostic": True,
        "driven": False,
        "detail": ("deletes argument names the tool's schema rejects, then lets the repaired call "
                   "proceed. Pure deletion -- the executor synthesizes nothing"),
    },
}


def _value_of(x: Any) -> str:
    """An `IncisionPoint`/`Action`'s `.value`, or the plain string. Callers pass either."""
    return str(getattr(x, "value", x))


def executor_for(boundary: Any, action: Any) -> Mapping[str, Any] | None:
    """The executor that can realize `(boundary, action)` on this host, or None.

    None means NOT MATERIALIZABLE whatever `U_H(l)` says, and it prunes the candidate before
    evaluation rather than after.
    """
    spec = EXECUTORS.get((_value_of(boundary), _value_of(action)))
    return dict(spec) if spec else None


def executor_supports(boundary: Any, action: Any, signal: str,
                      eta: Mapping[str, Any]) -> tuple[bool, str]:
    """Can the host's executor for `(boundary, action)` consume this `eta` for `signal`?

    Three conditions:
      1. an executor exists for the cell;
      2. every parameter the executor FIXES matches what the candidate declares, where the candidate
         declares it at all -- silence is fine, disagreement is not;
      3. the executor is signal-agnostic, or it covers this signal.

    A MISMATCH IS REFUSED, NEVER COERCED. An executor that silently substitutes its own value runs
    different semantics under the candidate's name, and the measured delta is then not caused by what
    was proposed. On the other benchmark exactly this check correctly rejected a candidate declaring
    a retry budget the executor could not honour -- and failed to reject another, because the value
    the executor imposed had never been declared for comparison.
    """
    where = _value_of(boundary)
    what = _value_of(action)
    spec = executor_for(boundary, action)
    if spec is None:
        return False, (f"no_executor: {NAME} has no executor for ({where}, {what}); U_H "
                       f"admissibility is not materializability")
    if not spec.get("signal_agnostic", False):
        covered = tuple(spec.get("signals", ()))
        if covered and signal not in covered:
            return False, f"signal_not_covered: this executor covers {list(covered)}"
    for key, value in dict(spec.get("fixed") or {}).items():
        if key in (eta or {}) and eta[key] != value:
            return False, (f"fixed_param_conflict: the executor imposes {key}={value!r} and the "
                           f"candidate declares {eta[key]!r}; refusing rather than coercing")
    return True, f"materializable by the {what} executor at {where}"


def env_gated(boundary: Any, action: Any) -> bool:
    """Does this cell need a switch outside the controller spec? None does, on this runtime.

    Declared so the question has an answer. A candidate that looks materializable but whose switch is
    unset runs as the CONTROL -- the silent-null defect class -- and `--controllers` omitted being the
    only control is what `verify_plumbing.py` will assert.
    """
    return False


# ------------------------------------------------------------------------------------------------
# 8. the theta / eta contract, declared so a proposer can read it
# ------------------------------------------------------------------------------------------------
#
# TWO VOCABULARIES, and conflating them has a measured cost. `eta` is the SEARCH-side action
# parameter named by `anchoropt/learning/action_contract.py` (`instruction`, `suppressed_operation`,
# `destination`, ...). `theta` is the EXECUTION-side payload `apply_action` reads (`text`, `reason`,
# `destination`, ...). They are deliberately not merged -- the contract is generic, the directive is
# this runtime's -- but the mapping must be written down, because BFCL's live smoke run had a proposer
# write a thoughtful reprompt into `theta['guidance']` where the runtime requires `theta['text']`.
# Every decline was correct and none was informative. A contract the caller must satisfy but cannot
# read is a defect in the interface, not in the caller.
ETA_TO_THETA: Mapping[str, Mapping[str, str]] = {
    "reprompt": {"instruction": "text"},
    "suppress": {"suppressed_operation": "reason"},
    # `argument_mapping` is PROSE in the grounding ("delete exactly the names the schema does not
    # declare"), so it maps to the boolean that requests that behaviour rather than to a literal list.
    # Mapping prose into `drop_args` would have produced a spec asking to delete an argument called
    # "delete exactly the argument names...".
    "substitute": {"destination": "destination", "argument_mapping": "drop_unknown_args"},
}


def action_theta_schema() -> Mapping[str, Mapping[str, Any]]:
    """The theta keys each action REQUIRES, per `apply_action`. Generated from one place so it
    cannot drift from the function it documents."""
    return {
        "noop": {"required": {}, "optional": {}},
        "reprompt": {"required": {"text": "str -- what the model must see"},
                     "optional": {}},
        "suppress": {"required": {"reason": "str -- why the call was cancelled"},
                     "optional": {"replacement": "the observation to return in its place",
                                  "keep_call": "bool -- withhold execution but leave the call, so "
                                               "the turn's kind and the budget are unchanged"}},
        "reroute": {"required": {"destination": "str -- the tool to call; the proposed one, "
                                               "unchanged, is the supported case"},
                    "optional": {"args_patch": "dict of REAL argument names for that tool",
                                 "drop_args": "list of argument names to delete",
                                 "drop_unknown_args": "bool -- delete whatever the tool's schema "
                                                      "does not declare, derived per call",
                                 "reason": "str"}},
    }


def signal_param_schema() -> Mapping[str, Mapping[str, Any]]:
    """Per signal, the parameters it requires and their types -- the other half of the contract.

    Same reason as `action_theta_schema`: BFCL's smoke-run proposer supplied a parameter named
    `threshold` for a signal whose declared parameter is `below`. It had no way to know the name.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, schema in SIGNAL_PARAMS.items():
        out[name] = {key: {"type": typ.__name__, "required": bool(req)}
                     for key, (typ, req) in schema.items()}
    return out


# ------------------------------------------------------------------------------------------------
# 9. applying a semantic action
# ------------------------------------------------------------------------------------------------
def apply_action(action: Action, boundary: IncisionPoint, theta: Mapping[str, Any],
                 state: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a SEMANTIC action into this runtime's directive.

    Returns a plain semantic dict the wrapper executes; performs NO I/O, so it is testable without a
    live model or a corpus. It names WHAT must happen, never the channel -- which message list is
    edited is `cctu_middleware`'s business.

    `HOST.require` runs BEFORE any payload is built, so an unexecutable cell can never report
    success. That ordering is the CONSUMER_BOUNDARY_RULE: the TB2 prototype built its payload first
    and reported `executed=True` while appending it to the empty string.
    """
    HOST.require(boundary, action)

    if action is Action.NOOP:
        return {"kind": "noop", "executed": False}

    if action is Action.REPROMPT:
        text = str(theta.get("text", "")).strip()
        if not text:
            raise ValueError("REPROMPT requires non-empty theta['text']")
        # `request_redecision` says another model decision is required. At PRE_GENERATION the next
        # generation has not happened, so the text only shapes context; at the other two a decision
        # already exists and must be replaced. `charges_round` is the CCTU-specific half: a
        # re-decision at the commitment gate is free because the validator has not run, and one
        # after execution costs a round because it has.
        return {"kind": "reprompt", "executed": True, "text": text,
                "request_redecision": boundary is not IncisionPoint.PRE_GENERATION,
                "charges_round": boundary is IncisionPoint.POST_EXECUTION}

    if action is Action.SUPPRESS:
        reason = str(theta.get("reason", "")).strip()
        if not reason:
            raise ValueError("SUPPRESS requires non-empty theta['reason']")
        keep_call = bool(theta.get("keep_call", False))
        # THE PRECONDITION IS PART OF THE DIRECTIVE, not left to the executor to remember. Dropping
        # the only proposed call makes `len(tool_calls) == 0`, which the harness reads as a final
        # answer -- so the content written for a tool turn gets judged by the terminal handlers.
        # `withhold_execution` (keep_call) is exempt, because the call stays in the message.
        n_calls = int(state.get("n_tool_calls", 0) or 0)
        if not keep_call and n_calls <= 1:
            raise ValueError(
                "SUPPRESS/drop_call requires the turn to propose more than one call: removing the "
                "only call converts a tool turn into a final answer, and the terminal constraint "
                "handlers would then judge content written for a tool turn. Use keep_call=True to "
                "withhold execution instead.")
        return {"kind": "suppress", "executed": True, "reason": reason,
                "keep_call": keep_call,
                "replacement": theta.get("replacement"),
                "charges_budget": keep_call,
                "short_circuit": not keep_call}

    if action is Action.REROUTE:
        destination = str(theta.get("destination", "")).strip()
        if not destination:
            raise ValueError("REROUTE requires theta['destination'] -- a reroute with no named "
                             "destination cannot be attested on RESOLVING")
        drop_args = tuple(str(a) for a in (theta.get("drop_args") or ()))
        args_patch = dict(theta.get("args_patch") or {})
        # `drop_unknown_args` is the SCHEMA-DERIVED form, and it exists because the grounded variant
        # needs it: `ground_substitute_destinations` offers exactly one option -- delete the argument
        # names the tool's own schema does not declare -- and which names those are is a per-call fact,
        # not something a persisted spec can enumerate. Without this, that candidate could be built by
        # the search and not expressed as a controller, which is a gap that only appears when you try
        # to serialize one.
        derive = bool(theta.get("drop_unknown_args", False))
        if not drop_args and not args_patch and not derive:
            raise ValueError("REROUTE requires theta['drop_args'], theta['args_patch'] or "
                             "theta['drop_unknown_args'] -- a substitution that changes neither the "
                             "tool nor its arguments is a noop wearing a reroute's name")
        return {"kind": "reroute", "executed": True, "destination": destination,
                "drop_args": drop_args, "args_patch": args_patch,
                "drop_unknown_args": derive, "reason": str(theta.get("reason", ""))}

    raise AssertionError(f"unhandled action {action!r}")


# ------------------------------------------------------------------------------------------------
# The runtime record + registration
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CCTURuntime:
    """The CCTU runtime surface as one object, for injection-style callers.

    The module-level functions are the primary interface -- they are what the core call sites use --
    and this record exists so the required surface is enumerable and so a caller can pass the adapter
    in rather than import it. `scripts/check_adapter.py` reads `ADAPTER` if present.
    """

    name: str = NAME
    HOST: HostProfile = HOST

    # boundary identification
    normalize_event = staticmethod(normalize_event)
    is_decision = staticmethod(is_decision)
    boundary_key = staticmethod(boundary_key)
    boundary_from_key = staticmethod(boundary_from_key)
    label_for = staticmethod(label_for)
    commits_to_answer = staticmethod(commits_to_answer)
    actions_in = staticmethod(actions_in)

    # observables and signals
    observable_state = staticmethod(observable_state)
    synthesis_fields = staticmethod(synthesis_fields)
    states_at = staticmethod(states_at)
    declared_signals = staticmethod(declared_signals)
    signal_aliases = staticmethod(signal_aliases)
    signal_boundaries = staticmethod(signal_boundaries)
    evaluate_signal = staticmethod(evaluate_signal)
    install_signal = staticmethod(install_signal)
    expanded_signals = staticmethod(expanded_signals)
    reset_expanded_signals = staticmethod(reset_expanded_signals)
    register_signal = staticmethod(register_signal)
    probe_params = staticmethod(probe_params)
    parameter_domains = staticmethod(parameter_domains)
    policy_class_for = staticmethod(policy_class_for)

    # actions
    feasible_actions = staticmethod(feasible_actions)
    ground_reprompt = staticmethod(ground_reprompt)
    ground_suppress = staticmethod(ground_suppress)
    ground_substitute_destinations = staticmethod(ground_substitute_destinations)
    ground_transforms = staticmethod(ground_transforms)
    executor_for = staticmethod(executor_for)
    executor_supports = staticmethod(executor_supports)
    env_gated = staticmethod(env_gated)
    apply_action = staticmethod(apply_action)
    action_theta_schema = staticmethod(action_theta_schema)
    signal_param_schema = staticmethod(signal_param_schema)

    # corpus vocabulary
    parse_case_id = staticmethod(parse_case_id)
    query_id_of = staticmethod(query_id_of)


ADAPTER = CCTURuntime()
RUNTIME = ADAPTER


def register() -> None:
    """Install this adapter as the core's active vocabulary. Idempotent.

    Registering the MODULE, not `ADAPTER`: module-level exception types (`UnknownCaseId`) are part of
    the interface and a dataclass instance would not expose them.

    NOT CALLED ON IMPORT, unlike `benchmarks/bfcl_v4/adapter.py`. There is one global adapter slot,
    and taking it on import would silently repoint BFCL's core call sites at this vocabulary -- so a
    test that merely imports this module would change another benchmark's behaviour.
    """
    import sys as _sys

    from anchoropt.attribution import register_adapter
    register_adapter(_sys.modules[__name__])
