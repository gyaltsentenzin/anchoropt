"""Phi-CONSTRUCTION ALPHABET for CCTU: the observable fields a signal may name.

This is the module the PROPOSER and `signal_grammar.synthesize` see. It answers two questions and
deliberately no others:

    observable_fields()   which primitive facts exist, of what type, at which decision points
    carried_fields()      which summaries of HISTORY the wrapper maintains

WHY AN ALPHABET AND NOT A LIST OF SIGNALS
-----------------------------------------
`benchmarks/bfcl_v4/bfcl_capabilities.py` records what happened when that integration declared nine
signals, one per accepted anchor: "That makes the anchor library the design: a proposer restricted to
those names can only ever re-select what is already known." So the runtime declares what is
OBSERVABLE and of what TYPE, and a signal is an expression over it
(`anchoropt/learning/signal_lang.py`, enumerated by `anchoropt/learning/signal_grammar.py`).

WHY THERE ARE DERIVED BOOLEANS HERE, AND WHY THAT IS NOT HAND-ENGINEERING
------------------------------------------------------------------------
The fair objection to `call_times_over_cap` is that it looks like a finished condition rather than a
fact, and that declaring it is the adapter doing the search's job. The answer is structural, not
stylistic:

    the grammar is  phi ::= x | not x | x == c | x < theta | x > theta | phi_1 and phi_2

Every leaf compares ONE field to a CONSTANT. So a condition relating two live quantities -- "this
turn proposes more calls than the budget still allows" -- is **not reachable** from
`n_tool_calls` and `call_times_remaining` no matter how the search composes them, because no leaf
compares two fields. It has to be supplied as a field or it cannot be expressed at all. That is the
same reason BFCL declares `container_occupancy` (a ratio) rather than a numerator and a denominator.

Both forms are declared: the raw scalars AND the derived comparison. Which one the search selects is
then an observation rather than an assumption -- and a checkable one. **If cycle 1 only ever selects
the derived booleans and never a threshold on a raw scalar, that is evidence the adapter did the
work**, and it should be reported as such rather than discovered later by a reader.

THE PRESSURE FIELDS ARE NOT A SECOND IMPLEMENTATION -- THEY ARE UPSTREAM'S, RUN EARLY
------------------------------------------------------------------------------------
`call_times_over_cap`, `per_tool_over_cap`, `parallel_over_cap`, `order_prereq_unmet`,
`parallel_group_incomplete` and `args_invalid` predict what `utils/constraint_checker/` will decide,
because the point is to know BEFORE the validator runs and mutates the budget (see ANCHOROPT.md on
the two POST_GENERATION sub-moments).

An earlier draft of this docstring called them a SECOND implementation of upstream's rules and
required a parity test for each. **That turned out to be unnecessary, and the better answer is in
`cctu_state.py`:** every constraint handler reads and writes the checker through plain attributes and
calls no method on it, so a duck-typed stand-in built from a snapshot can be handed to upstream's real
`check()`. The handler reports exactly what it would report, the stand-in absorbs the mutations, and
the live checker is untouched.

    the predicate IS upstream's logic, applied to a copy of the state

So there is no mirror to keep in step, and five parity risks are removed rather than managed. It also
inherits the awkward parts for free -- including `ToolOrderHandler`'s corpus-specific override for
`query_id == 96` (tool.py:77-88), where the effective order depends on which tool was called first
and is EMPTY unless that tool was `urban_area_identifier`. A hand-written mirror would have disagreed
on exactly one episode, which is the hardest kind of disagreement to notice and the easiest to
dismiss. `args_invalid` needs no stand-in at all: `ToolArgsChecker.check` is already pure.

NO INFINITY MAY REACH A DECLARED FIELD
--------------------------------------
`check_utils.to_int(None)` returns `math.inf`, so an episode that declares no cap has
`max_callTimes == inf` and `max_callTimesPerTool[t] == inf`. `signal_grammar._numeric_values` accepts
that value, `_quantile` will return it, and `round(inf, 6)` is `inf` -- yielding the atoms
`x < inf` (fires on everything finite) and `x > inf` (fires never) and shifting every other quantile
on the field. A threshold that fires on everything is not a threshold; that exact failure is recorded
in `signal_grammar`'s own module docstring.

    So `call_times_remaining` and `calls_remaining_this_tool` are **None when unconstrained, never
    inf.** `_numeric_values` skips None, which is the behaviour wanted: the field contributes no
    observation rather than a poisoned one.

EVERY FIELD IS AS-OF-THE-BOUNDARY
---------------------------------
`get_feedback_if` advances `round`, `callTimes`, `callTimesPerTool` and `earliest_callTurnPerTool`
mid-step (see ANCHOROPT.md). So a budget field legitimately differs between POST_GENERATION_PRE_EXEC
and POST_EXECUTION within ONE turn, and that is not a bug to be smoothed over -- it is the difference
between "spending this is still preventable" and "this is already spent". The wrapper passes the
value observed at the boundary through `observable_state(normalized, carried=...)`; nothing here reads
a mutable accumulator.

`round_index` is the ONE exception, and deliberately: it is the wrapper's own 0-based turn counter,
NOT `checker.round`, so it means the same thing at all three boundaries. `rounds_remaining` is then
`max_round - (round_index + 1)` -- turns available AFTER this one -- which makes
`rounds_remaining == 0` mean exactly "this is the last turn the budget allows", agreeing with the
harness's own `budget_exhausted = (checker.round >= checker.max_round)` evaluated at the end of the
turn. An off-by-one here is the difference between an anchor that fires on the last turn and one that
fires after it is too late.

WHAT IS DELIBERATELY ABSENT
---------------------------
**The three response validators.** `FORMAT`, `PUNCTUATION` and `IDENTIFIERS` are per-episode Python
files under `data/check_code/<query_id>/`, invoked through `validator_loader.call_validator`, and each
is a pure function of the proposed content. So `format_check_fails` etc. COULD be declared, and they
would make those violations preventable at POST_GENERATION rather than merely observable afterwards --
91 to 103 episodes each, so this is not a marginal class.

They are withheld from cycle 0 for two reasons, both worth stating rather than quietly deciding:

  1. it means executing arbitrary per-episode Python on every proposed final answer, which is a
     runtime and a containment cost that should be paid deliberately;
  2. it is a **capability asymmetry**: the anchor would get an exact validator where the model has
     only the prose constraint. That is legitimate -- the environment publishes its constraints, and
     BFCL's accepted A8 does the same shape of thing by testing a proposed action against live
     container contents -- but an asymmetry that large should be declared and reviewed, not
     introduced as a field nobody looked at twice.

Adding them later requires no change to any action implementation, which is the property this
separation exists to preserve.

**Any field derived from `unsolved_set` or `answer`.** Those are the scoring labels. Nothing here may
read them, and `tests/test_cctu_adapter.py` must assert it. A softer version of this leak has already
cost this project a claim: importing the anchor fixtures put the answer key into every process that
merely touched the BFCL runtime. A discovery made with the answer key in the room is a lookup,
however unused it looks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Boundary names, as declared by the core's IncisionPoint values. Strings here so this module can be
# read and tested without importing the core.
PRE_GEN = "pre_generation"
POST_GEN = "post_generation_pre_exec"
POST_EXEC = "post_execution"

ALL_BOUNDARIES = (PRE_GEN, POST_GEN, POST_EXEC)

# Boundaries at which a PROPOSAL exists. Named because every per-turn field shares it and repeating
# the tuple invites one of them drifting.
PROPOSED = (POST_GEN, POST_EXEC)


@dataclass(frozen=True)
class Field:
    """One primitive observable. `boundaries` is where the fact EXISTS, not where it is useful.

    Same shape as the BFCL and template declarations, because `signal_grammar.atoms_for` reads
    `.type` and `.enum` and `synthesis_fields` filters on `.boundaries`. A field listed at a boundary
    must be evaluable there: a condition evaluated where its facts do not exist answers False for a
    structural reason indistinguishable from "the condition did not hold", and on the other benchmark
    that cost -4.95 pp against +3.63 pp for byte-identical text one boundary later.
    """

    name: str
    type: type
    boundaries: tuple[str, ...]
    doc: str
    enum: tuple[str, ...] = ()


# ------------------------------------------------------------------------------------------------
# The violation vocabulary -- upstream's own, canonicalized
# ------------------------------------------------------------------------------------------------
#
# The 14 classes the handlers emit, as `INSTRUCTION FOLLOWING ERROR: <CLASS> NOT FOLLOWED!`. This is
# a CLOSED set read off the handlers, not a guess:
#
#   handlers/interact.py   MIN ROUND · MIN/MAX CALL TIMES · MIN/MAX PARALLEL CALLS
#   handlers/tool.py       MAX CALLS PER TOOL · TOOL ORDER · TOOL PARALLEL
#   handlers/response.py   MIN/MAX LENGTH · FORMAT · PUNCTUATION · IDENTIFIERS
#   args_checker.py        TOOL ARGUMENTS
#
# ORDERING IS LOAD-BEARING, and not for readability. `signal_grammar.atoms_for` emits `equals` atoms
# for the first MAX_ATOMS_PER_FIELD (= 4) enum values only, plus one `falsy`. So 10 of these 14 are
# unreachable to synthesis through this field, and which 4 are reachable is decided HERE.
#
# The four chosen are the four that are PREVENTABLE at the commitment gate from live checker state
# alone -- the budget and ordering classes -- because those are the ones an intervention at that
# boundary could act on. The response classes are judged by per-episode validator files this module
# deliberately does not run (see the docstring), so promoting one of them into the reachable four
# would offer synthesis a condition no declared field can support.
#
# THIS ORDERING IS A PRIORI AND SHOULD BE REVISITED FROM CYCLE 0's MEASURED DISTRIBUTION. Re-ordering
# from observed frequency is legitimate -- it is the same discipline as deriving thresholds from
# observed values -- it simply cannot be done before there is a residual to observe. The four
# dimension booleans below exist so the search is not hostage to this choice in the meantime.
VIOLATION_CLASSES: tuple[str, ...] = (
    # reachable by synthesis (first four), and preventable at the commitment gate
    "max_call_times",
    "max_calls_per_tool",
    "max_parallel_calls",
    "tool_order",
    # expressible by an exact predicate, not reached by enum-atom enumeration
    "tool_parallel",
    "tool_arguments",
    "min_round",
    "min_call_times",
    "min_parallel_calls",
    "min_length",
    "max_length",
    "format",
    "punctuation",
    "identifiers",
)

# Which dimension each class belongs to, from `HANDLER_REGISTRY`'s own keys
# (`utils/constraint_checker/handlers/registry.py`). `tool_arguments` is its own dimension: it comes
# from `ToolArgsChecker`, which is not a registered constraint handler at all but a schema check over
# every proposed call.
VIOLATION_DIMENSION: Mapping[str, str] = {
    "min_round": "resource",
    "min_call_times": "resource",
    "max_call_times": "resource",
    "max_calls_per_tool": "resource",
    "min_parallel_calls": "behavior",
    "max_parallel_calls": "behavior",
    "tool_order": "behavior",
    "tool_parallel": "behavior",
    "min_length": "response",
    "max_length": "response",
    "format": "response",
    "punctuation": "response",
    "identifiers": "response",
    "tool_arguments": "arguments",
}

VIOLATION_DIMENSIONS: tuple[str, ...] = ("resource", "behavior", "response", "arguments")


# ------------------------------------------------------------------------------------------------
# The primitive alphabet
# ------------------------------------------------------------------------------------------------
_FIELDS = (
    # ---- what the model just decided: exists once a generation has happened ---------------------
    Field("has_generation", bool, PROPOSED,
          "the model has produced a decision for this turn"),
    Field("proposes_tool_call", bool, PROPOSED,
          "the decision is at least one tool call rather than a final answer; upstream's own "
          "loop-exit test is `is_final = (len(tool_calls) == 0)`"),
    Field("n_tool_calls", int, PROPOSED,
          "how many calls this turn proposes -- the `times` unit of the parallel-calls constraint"),
    Field("n_distinct_tool_names", int, PROPOSED,
          "how many DISTINCT tools this turn proposes -- the `type` unit of the same constraint. "
          "Both units occur in the corpus, so both are declared"),
    Field("tool_name", str, PROPOSED,
          "name of the first proposed call, or None for a terminal response. DELIBERATELY carries "
          "no enum: the toolset is per-episode, and offering synthesis one episode's tool names "
          "would let it fit a condition that cannot generalize past that episode"),
    Field("content_length", int, PROPOSED,
          "characters of the assistant content, after `_strip_think_keep_text` -- the same "
          "normalization the response-length handler scores"),
    Field("repeated_identical_call", bool, PROPOSED,
          "this turn proposes a (name, arguments) pair already proposed earlier in the episode. A "
          "structural fact about the trajectory, not a judgement that the repeat is wrong"),

    # ---- constraint pressure, PER TURN: preventable at the commitment gate -----------------------
    #
    # Each is a comparison between two live quantities, which no grammar leaf can form. Each mirrors
    # one upstream handler and each needs the parity test named in the module docstring.
    Field("call_times_over_cap", bool, PROPOSED,
          "this turn's calls would push the episode's total past `max_callTimes`"),
    Field("per_tool_over_cap", bool, PROPOSED,
          "this turn's calls would push some tool past its own `max_callTimesPerTool` entry"),
    Field("parallel_over_cap", bool, PROPOSED,
          "this turn's parallel count exceeds `max_parallelCallTypes`, in the episode's declared unit"),
    Field("order_prereq_unmet", bool, PROPOSED,
          "a proposed tool has a predecessor in `tool_order` that has not been called yet"),
    Field("parallel_group_incomplete", bool, PROPOSED,
          "a proposed tool belongs to a `parallel_groups` entry this turn does not satisfy"),
    Field("args_invalid", bool, PROPOSED,
          "a proposed call fails the tool's own schema: unknown tool, unparseable arguments, "
          "missing required, extra, or a value the schema rejects"),

    # ---- MIN-class and response-class pressure: judged only where the turn is terminal ----------
    #
    # Declared at the same boundaries as everything else, but each is False unless the turn proposes
    # a final answer, because that is when upstream checks them (`if not ctx.is_final: return`).
    # This asymmetry is the corpus's dominant structure, not an inconvenience: a MIN-class violation
    # becomes observable exactly when the answer is proposed, and the rounds needed to satisfy it may
    # already be gone.
    Field("min_round_unmet", bool, PROPOSED,
          "the turn proposes a final answer before `min_round` rounds have been used"),
    Field("min_call_times_unmet", bool, PROPOSED,
          "the turn proposes a final answer with fewer than `min_callTimes` calls made"),
    Field("parallel_requirement_unmet", bool, PROPOSED,
          "the turn proposes a final answer and the episode never reached `min_parallelCallTypes`"),
    Field("response_length_over_cap", bool, PROPOSED,
          "the proposed final answer exceeds `max_responseLength` in the declared unit"),
    Field("response_length_under_min", bool, PROPOSED,
          "the proposed final answer is under `min_responseLength` in the declared unit"),

    # ---- what came back: exists only after the validator and the tools have run ------------------
    Field("violation_class", str, (POST_EXEC,),
          "the canonical class of the first violation reported this turn, or None. Only the first "
          "four values are reachable by synthesis -- see VIOLATION_CLASSES",
          enum=VIOLATION_CLASSES),
    Field("violation_is_resource", bool, (POST_EXEC,),
          "a reported violation is in the resource dimension (rounds, call budget, per-tool budget)"),
    Field("violation_is_behavior", bool, (POST_EXEC,),
          "a reported violation is in the behavior dimension (parallelism, ordering)"),
    Field("violation_is_response", bool, (POST_EXEC,),
          "a reported violation is in the response dimension (length, format, punctuation, "
          "identifiers)"),
    Field("violation_is_arguments", bool, (POST_EXEC,),
          "a reported violation is a tool-argument schema failure"),
    Field("n_violations_this_turn", int, (POST_EXEC,),
          "how many violation messages this turn produced, across both delivery channels"),
    Field("result_is_error", bool, (POST_EXEC,),
          "an executed tool raised or timed out. NOT a model failure: it is the episode's own tool "
          "code or the 10s `func_set_timeout`, so it must stay distinguishable from a violation and "
          "must be run past `attribution/harness_guard.py` before it is mined"),
    Field("result_is_empty", bool, (POST_EXEC,),
          "an executed tool succeeded and returned nothing informative. False whenever "
          "`result_is_error` is true, so one event is never counted as two conditions"),

    # ---- turn context: exists before anything is generated ---------------------------------------
    Field("round_index", int, ALL_BOUNDARIES,
          "0-based index of the turn being decided. The WRAPPER's counter, not `checker.round`, so "
          "it means the same at all three boundaries"),
    Field("rounds_remaining", int, ALL_BOUNDARIES,
          "turns available AFTER this one: `max_round - (round_index + 1)`. 0 means this is the last "
          "turn the budget allows"),
)

# ------------------------------------------------------------------------------------------------
# Carried summaries -- HISTORY, maintained by the wrapper
# ------------------------------------------------------------------------------------------------
#
# Each is a summary of earlier turns, readable at ONE boundary like any other field. This is what
# keeps every signal single-locus: a condition that genuinely spans time ("the last turn was
# refused, and this one proposes the same call again") reads a carried fact and a present-tense one,
# both at the same point, so `U_H(l)` keeps meaning what it says. There is deliberately no way for an
# expression to reference another boundary.
_CARRIED = (
    Field("call_times", int, ALL_BOUNDARIES,
          "calls the episode has made so far, as of this boundary"),
    Field("call_times_remaining", float, ALL_BOUNDARIES,
          "`max_callTimes - call_times`, or **None when the episode declares no cap** -- never inf, "
          "see the module docstring"),
    Field("calls_remaining_this_tool", float, ALL_BOUNDARIES,
          "the same quantity for the tool this turn proposes, or None when uncapped or when no call "
          "is proposed"),
    Field("last_violation_class", str, ALL_BOUNDARIES,
          "canonical class of the most recent violation in this episode, or None",
          enum=VIOLATION_CLASSES),
    Field("consecutive_violation_turns", int, ALL_BOUNDARIES,
          "turns in a row that produced at least one violation. Resets on a clean turn, so it "
          "measures a stuck self-refinement loop rather than a total"),
)


def observable_fields() -> Mapping[str, Field]:
    """The primitive fields a proposed signal may name. Nothing else is nameable."""
    return {f.name: f for f in _FIELDS}


def carried_fields() -> Mapping[str, Field]:
    """History summaries the wrapper maintains. Ordinary fields, single-locus by construction."""
    return {f.name: f for f in _CARRIED}


def all_fields() -> Mapping[str, Field]:
    return {**observable_fields(), **carried_fields()}


def fields_at(boundary: str) -> tuple[str, ...]:
    """Field names readable at `boundary`, primitives and carried summaries together.

    Narrowing per boundary is not cosmetic. `structured_search.MAX_SYNTHESIS_CANDIDATES` is 40 and
    `signal_grammar.synthesize` forms conjunctions from the top 12 discriminating atoms only, so
    every field declared where its fact does not exist spends search budget that a real candidate
    could have used. PRE_GENERATION sees the turn context and the carried summaries; the proposal
    fields and the result fields are not there yet.
    """
    b = str(getattr(boundary, "value", boundary))
    return tuple(sorted(n for n, f in all_fields().items() if b in f.boundaries))


def dimension_of(violation_class: str | None) -> str | None:
    """Which constraint dimension a canonical violation class belongs to, or None.

    Unknown classes return None rather than a default. An unrecognised violation filed under a real
    dimension is the same defect as an unrecognised case id filed under a real backend, which
    `benchmarks/bfcl_v4/adapter.py` exists to have ended.
    """
    if violation_class is None:
        return None
    return VIOLATION_DIMENSION.get(str(violation_class))


def synthesis_reachable_classes() -> tuple[str, ...]:
    """The violation classes `signal_grammar.atoms_for` will actually enumerate atoms for.

    Exposed rather than left implicit: a reader comparing VIOLATION_CLASSES against what the search
    tried would otherwise conclude the search ignored ten conditions, when in fact the enum-atom cap
    never offered them. Reported, like every other pruning in this project.
    """
    from anchoropt.learning.signal_grammar import MAX_ATOMS_PER_FIELD

    return VIOLATION_CLASSES[:MAX_ATOMS_PER_FIELD]
