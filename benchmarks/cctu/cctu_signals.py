"""Phi_CCTU -- CCTU's own signal evaluators, and the three classifiers they read.

Deliberately separate from the generic core: `anchoropt/` must not learn what a constraint violation
or a round budget is. Each signal is a pure function of the adapter-normalized observable state, so
adding one requires NO change to any action implementation.

    cctu_capabilities.py   the ALPHABET      which facts exist, of what type, at which boundary
    cctu_signals.py        Phi_CCTU          which CONDITIONS are observable, and the classifiers
    cctu_adapter.py        the RUNTIME       U_H(l), normalize_event, apply_action, the search hooks

WHY ONLY FOUR SIGNALS
--------------------
Every one is either structurally forced by the harness's own control flow or handed to us by the
environment's published error contract. Nothing here is a candidate somebody thought would work.

`benchmarks/tb2_deepagents/tb2_signals.py` refuses to pre-register the conditions its residual
analysis suggested, because "registering them by hand here would bypass exactly the discipline that
caught `single_read_then_answer` (precision 0.51)". The same rule applies with more force here,
because CCTU's checker state makes *many* conditions cheap to write and cheapness is not evidence.
The interesting predicates must arrive through `anchoropt/learning/expand_attribution.py`, which
scores coverage x precision and refuses a condition that fires as often on successes as on failures.

So: four signals, and an alphabet of 33 fields for the search to build the rest from -- 7 readable
before generation, 25 at the commitment gate, all 33 after execution.

THE CLASSIFIERS ARE THE PART MOST LIKELY TO BE WRONG
----------------------------------------------------
`docs/GENERALIZABILITY.md` names signal extraction as the first thing a port gets wrong, and asks for
four outcome classes: errored / vacuous / resolving / **harness fault**. CCTU's three are below, and
the third exists because that document's warning is concrete: on the other benchmark an empty-memory
search raised `ZeroDivisionError` and was mined as a real error contract at support 13 before anyone
noticed.

    violation_class_of()   the environment REFUSED the turn -- a model failure, and actionable
    is_tool_fault()        the episode's tool code raised or timed out -- NOT a model failure
    result_vacuity_kind()  the call succeeded and returned nothing informative

The second is the one to be careful with. `utils/utils.py:132-137` turns both `FunctionTimedOut` and
any exception from the episode's own tool source into an ordinary `role: "tool"` message, so a harness
fault and a real tool result are the same shape in the transcript. And the timeout is a **10-second
wall clock** (`utils/utils.py:95`) under a `ThreadPoolExecutor`, so the same call can fault on a loaded
host and succeed on an idle one -- which makes this class a variance source as well as an attribution
trap. Run everything this classifier marks past `anchoropt/attribution/harness_guard.py` before mining
it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, Callable

from cctu_capabilities import VIOLATION_CLASSES, dimension_of

# ------------------------------------------------------------------------------------------------
# 1. the violation classifier
# ------------------------------------------------------------------------------------------------
#
# The SAME pattern as `evaluation.py:26`, which is what decides SR and PSR. It is written out here
# rather than imported because `evaluation.py` is a script with a `main()` and module-level `tqdm` /
# `numpy` imports, and because importing a module named `evaluation` from a shared sys.path is the
# package-name collision `COLLABORATORS.md` lists among the four bugs real hardware found.
#
# One owner plus a parity test is the precedent this repo already set: `benchmarks/bfcl_v4/adapter.py`
# MOVED five copies of one regex into one place and shipped a test asserting byte-identical outputs,
# because five copies of a regex is a maintenance problem and five that DISAGREE is a correctness one.
# So `tests/test_cctu_adapter.py` must pin this against `evaluation._ERR_RE` over the corpus's own
# strings. Two definitions of "which constraint was violated" that disagree would mean the signal
# fires on a different population than the metric scores.
_ERR_RE = re.compile(
    r"INSTRUCTION FOLLOWING ERROR:\s*([A-Z0-9 _]+?)\s*NOT FOLLOWED!",
    re.IGNORECASE,
)

# Canonical classes observed that this module does not declare. Accumulated rather than discarded --
# see `violation_class_of`.
UNDECLARED_CLASSES_SEEN: set[str] = set()


def violation_classes_of(text: str) -> tuple[str, ...]:
    """Every violation class reported in `text`, in order.

    Canonicalization is MECHANICAL -- lowercase, spaces to underscores -- not a lookup table. That is
    deliberate: `docs/GENERALIZABILITY.md` records that a hand-written cue table is a fast path and
    not the coverage story, because an unseen wording returns None and *vanishes from the ranking*.
    Here the set is closed and read off the handlers, so a mechanical rule covers it exactly and
    there is nothing to maintain.

    AN UNDECLARED CLASS IS STILL A VIOLATION. If upstream adds a handler, its class canonicalizes to
    a name this module does not list -- and dropping it would silently shrink the residual. So every
    canonical string is returned regardless, and the name is recorded in `UNDECLARED_CLASSES_SEEN` so
    a test can fail on the drift instead of a reader discovering it. This is a deliberate improvement
    on the other benchmark's None-and-vanish behaviour, which its own documentation flags as a
    weakness.

    ALL MATCHES, NOT JUST THE FIRST. A final turn's feedback arrives as one message concatenating
    every violated rule's line (`utils/constraint_checker/core.py:137-149`); a single-match extractor
    would silently drop every violation but the first from that message. `evaluation.violation_counts`
    scores with `_ERR_RE.finditer` for the same reason -- this must agree with it, not just with
    `evaluation.has_if_error_in_text`'s presence check.
    """
    out: list[str] = []
    for match in _ERR_RE.finditer(str(text or "")):
        canon = "_".join(match.group(1).strip().lower().split())
        if not canon:
            continue
        if canon not in VIOLATION_CLASSES:
            UNDECLARED_CLASSES_SEEN.add(canon)
        out.append(canon)
    return tuple(out)


def violation_class_of(text: str) -> str | None:
    """The canonical class of the first violation in `text`, or None if it reports none.

    A thin wrapper over `violation_classes_of` so there is one owner of the regex and the
    canonicalization rule. Kept because callers that only ever see one violation per message (the
    live predictive path in `cctu_state.would_violate`) want the plain optional-string shape.
    """
    classes = violation_classes_of(text)
    return classes[0] if classes else None


def violation_classes_in(messages: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """Every violation class reported across a turn's feedback messages, in order.

    Both delivery channels are read. Upstream splits them by turn kind
    (`utils/constraint_checker/core.py:137-149`): a non-final turn's violations come back as
    `role: "tool"` messages keyed by `tool_call_id`, a final turn's as ONE merged `role: "user"`
    message. A classifier that read only one channel would see none of the MIN-class and
    response-class violations, which are exactly the half that is not preventable. And within that
    merged message, `violation_classes_of` (not `violation_class_of`) is what actually reads every
    line rather than just the first.
    """
    out: list[str] = []
    for msg in messages or ():
        if not isinstance(msg, Mapping):
            continue
        out.extend(violation_classes_of(msg.get("content") or ""))
    return tuple(out)


def dimensions_in(classes: Iterable[str]) -> frozenset[str]:
    """The constraint dimensions a turn's violations span. Unknown classes contribute nothing."""
    return frozenset(d for d in (dimension_of(c) for c in classes or ()) if d is not None)


# ------------------------------------------------------------------------------------------------
# 2. the harness-fault classifier
# ------------------------------------------------------------------------------------------------
#
# `get_feedback_tools` wraps both failure modes in one message shape (`utils/utils.py:132-137`):
#
#     f"an error occured when call {tool_name}: {exc}"
#
# NOTE THE SPELLING. "occured" is upstream's, with one `r`. Matching the corrected spelling instead
# would make this classifier silently return False for every fault -- a detector that never fires is
# indistinguishable from a benchmark with no faults, and this whole class would then be invisible to
# mining rather than merely unhandled. Do not "fix" it here; fix it upstream and re-vendor.
_TOOL_FAULT_RE = re.compile(r"^\s*an error occured when call\s+\S", re.IGNORECASE)


def is_tool_fault(payload: Any) -> bool:
    """Did an executed tool raise or time out, rather than return a result?

    A HARNESS/CORPUS fault, never a model failure: the cause is the episode's own tool source under
    `exec` or the 10-second `func_set_timeout`. It must stay separable from a constraint violation
    (which the model caused and could avoid) and from a real result, because crediting an anchor for
    removing a timeout would be crediting it for the load on the host.
    """
    return bool(_TOOL_FAULT_RE.match(str(payload or "")))


# ------------------------------------------------------------------------------------------------
# 3. the vacuity classifier
# ------------------------------------------------------------------------------------------------
#
# Every literal below is a value `call_function` ITSELF produces: it `json.dumps` a dict or list and
# `str()`s anything else (`utils/utils.py:104-107`), so a tool returning None yields the four
# characters "None" and one returning [] yields "[]". These are facts about the harness, not guesses
# about phrasing.
#
# LEXICAL VACUITY IS DELIBERATELY ABSENT. A payload reading "no results found" is very likely vacuous
# and this module does not say so, because the wording would be invented rather than observed. Those
# cues must come from reading real payloads in cycle 0 -- the same reason the other benchmark's
# vacuity class "was invisible until a human read trajectories by hand".
_LITERAL_EMPTY = frozenset({"", "none", "null", "[]", "{}", "()"})


def result_vacuity_kind(payload: Any) -> str | None:
    """Classify a SUCCESSFUL call that carries no information, or None.

    Returns `"empty_collection"` / `"all_floor_scores"` from the shared core detector, or
    `"empty_payload"` for the two shapes that detector structurally cannot see.

    DELEGATES RATHER THAN FORKS. `policy_tree.vacuous_result_kind` is the existing definition and
    `benchmarks/bfcl_v4/bfcl_signals.py` records why re-implementing it here would be wrong: "there
    would then be two definitions of 'vacuous' that could disagree, which is precisely how this
    project has been bitten before."

    It is EXTENDED, though, and the reason is structural rather than a matter of taste:
    `_iter_result_collections` parses the payload and returns immediately unless it is a JSON
    **object** with list-valued fields. So a payload that IS a top-level empty array, or an empty
    string, or the literal "None", is not reachable by it at all. CCTU's tools return all three
    shapes. Extending at the shapes the shared detector cannot express keeps one definition of the
    case it does own.

    An erroring payload is never vacuous. Double-counting one event as two conditions inflates
    support, which is the same guard `no_informative_result` applies on the other benchmark.
    """
    if payload is None:
        return None
    if is_tool_fault(payload):
        return None

    from anchoropt.learning.policy_tree import vacuous_result_kind

    kind = vacuous_result_kind(payload)
    if kind is not None:
        return kind

    text = payload if isinstance(payload, str) else str(payload)
    if text.strip().lower() in _LITERAL_EMPTY:
        return "empty_payload"
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if isinstance(obj, (list, dict, str)) and len(obj) == 0:
        return "empty_payload"
    return None


VACUITY_KINDS: tuple[str, ...] = ("empty_collection", "all_floor_scores", "empty_payload")


# ------------------------------------------------------------------------------------------------
# 4. the signals. Each is a pure function of normalized state -- no I/O, no benchmark objects.
# ------------------------------------------------------------------------------------------------


def terminal_response_proposed(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """The model proposed a FINAL ANSWER rather than a tool call.

    Deterministic in this runtime: upstream's own loop-exit test is
    `is_final = (len(tool_calls) == 0)` (`response_generator.py:129`), and the adapter normalizes it
    into `proposes_tool_call` so this predicate never touches a message object.

    Boundary: POST_GENERATION_PRE_EXEC. The generation has happened and nothing has run, which is
    what makes the terminal decision still changeable -- and this is the ONLY point at which the
    MIN-class and response-class constraints are both observable and preventable, because upstream
    judges them exactly here (`if not ctx.is_final: return` in every such handler).

    SAME NAME AS THE TB2 ADAPTER'S, on purpose. Two runtimes that observe the same fact should not
    give it two names; a signal name is how a cross-benchmark claim is even stateable.

    ABSENCE OF A GENERATION IS NOT A TERMINAL RESPONSE. The TB2 prototype's `no_tool_call_yet` was
    inverted in exactly that way -- `tool_calls_so_far == 0` is true only at episode start -- so the
    guard is explicit rather than implied by field defaults.
    """
    if not state.get("has_generation", False):
        return False
    return not bool(state.get("proposes_tool_call", False))


def proposed_tool_action(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """The complementary case: at least one concrete tool call is proposed and has not run.

    Present so both POST_GENERATION_PRE_EXEC sub-cases are first-class. The existing incision point
    covers proposed tool actions AND proposed terminal responses; nothing here introduces a fourth
    boundary. This is also the sub-case where suppression is still free -- see ANCHOROPT.md on why it
    must act before the validator advances the budget.
    """
    return bool(state.get("has_generation", False)) and bool(state.get("proposes_tool_call", False))


def constraint_violation_reported(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """The environment refused this turn on at least one constraint.

    The lexical rung, and on this benchmark it is EXACT rather than approximate: the contract is
    `INSTRUCTION FOLLOWING ERROR: <CLASS> NOT FOLLOWED!` over a closed set of 14 classes, emitted by
    upstream's own handlers. `docs/GENERALIZABILITY.md`'s warning about error strings -- "ours include
    `core memory is full`, `Key name must be unique`" -- is about a vocabulary that had to be
    canonicalized by hand. Here the environment publishes it.

    `violation_class` is OPTIONAL. Omitted, the signal fires on any violation; supplied, it narrows
    to one class. The parameter is not required because "the turn was refused" is a real condition on
    its own, and a required parameter with no measured value would prune the signal from the search
    entirely rather than let it compete unparameterized.

    Boundary: POST_EXECUTION. The verdict exists only once the validator has run -- at which point
    the budget it charges has already been spent, which is why the preventable form of these
    conditions lives in the alphabet's per-turn booleans instead of in this signal.
    """
    seen = state.get("violation_class")
    if seen is None:
        return False
    want = params.get("violation_class")
    return True if want is None else str(seen) == str(want)


def tool_execution_error(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """An executed tool raised or timed out.

    DECLARED SO IT CAN BE EXCLUDED, not so it can be optimized. This is a harness/corpus fault, and
    an anchor that "fixed" it would be taking credit for host load or for a bug in the episode's own
    tool source. It is a first-class condition because the alternative -- leaving it unnamed -- is how
    it gets mined as a model failure, which has already happened once on this project at support 13.

    Keep it out of the accuracy residual, and note it separately as a variance source: the 10-second
    `func_set_timeout` is wall clock, so this condition is not deterministic across runs of the same
    policy. That is the first thing cycle 0's variance floor should be read against.
    """
    return bool(state.get("result_is_error", False))


# ------------------------------------------------------------------------------------------------
# Natural-language aliases -- expressibility matching ONLY, never dispatch
# ------------------------------------------------------------------------------------------------
#
# Why this exists: the matcher compares a diagnosis's PROSE against a SIGNAL NAME, and the two
# vocabularies genuinely differ. A residual note says "answered before making the required calls";
# the signal is called `terminal_response_proposed`. They describe one observable and share no
# distinctive token, so the canonical name alone marks the case SIGNAL_BLOCKED.
#
# MULTI-WORD PHRASES ONLY. Single generic verbs appear in prose about any failure -- on the TB2
# integration they matched an arithmetic reasoning error on incidental wording and selected a
# SUPPRESS controller with no causal link to it. A phrase that names the EVENT does not.
#
# These are a declared, reviewable part of the benchmark's vocabulary. They do NOT make the matcher
# semantic; the principled replacement is the `consequential_decision` -> observable mapping that
# Phi-expansion builds.
SIGNAL_ALIASES: Mapping[str, tuple[str, ...]] = {
    "terminal_response_proposed": ("premature conclusion", "premature termination", "final answer",
                                   "answered without", "concluded before", "stopped early",
                                   "without completing"),
    "proposed_tool_action": ("tool call", "proposed command", "invoke"),
    "constraint_violation_reported": ("constraint violation", "instruction following",
                                      "not followed", "violated the constraint",
                                      "exceeded the limit", "budget exceeded",
                                      "over the cap", "out of order"),
    "tool_execution_error": ("tool raised", "execution error", "timed out", "tool failed"),
}


SIGNALS: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], bool]] = {
    "terminal_response_proposed": terminal_response_proposed,
    "proposed_tool_action": proposed_tool_action,
    "constraint_violation_reported": constraint_violation_reported,
    "tool_execution_error": tool_execution_error,
}

# Typed parameter schema per signal: name -> (type, required). DECLARED, never inferred, so a
# proposal omitting a required parameter is PRUNED rather than run with a silent default. The TB2
# adapter records the cost of the other arrangement: a signal declared with no evaluator returned
# False everywhere, which is indistinguishable from a signal that did not fire, and an arm silently
# became the control arm mid-cluster.
SIGNAL_PARAMS: Mapping[str, Mapping[str, tuple[type, bool]]] = {
    "terminal_response_proposed": {},
    "proposed_tool_action": {},
    "constraint_violation_reported": {"violation_class": (str, False)},
    "tool_execution_error": {},
}

# PARAMETER DOMAINS -- what the OPTIMIZER may search, per signal parameter.
#
# Declared generically per parameter rather than as a special case, so any future signal with a
# tunable range exposes its domain the same way. CCTU's one parameter is CATEGORICAL, which is worth
# having in the first integration that ships: it keeps the domain layer from being read as
# threshold-only machinery.
SIGNAL_PARAM_DOMAINS: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    "constraint_violation_reported": {
        "violation_class": {"kind": str, "values": VIOLATION_CLASSES,
                            "doc": "which constraint class to fire on; omitted means any"},
    },
}

# Which boundaries each signal is OBSERVABLE at. DECLARED rather than inferred, because getting it
# wrong is a measured defect class and not a hypothetical one: a condition evaluated where its fact
# does not yet exist fires everywhere and pays the cost on the turns that needed nothing.
#
#   pre_generation            nothing about this turn's decision exists yet
#   post_generation_pre_exec  a proposal exists, nothing has run and nothing has been counted
#   post_execution            the validator has run and results are back
#
# Note what is NOT here: no signal is declared at `pre_generation`. That is not an oversight and not
# a policy about prompts -- none of these four conditions EXISTS before the model has decided.
SIGNAL_BOUNDARIES: Mapping[str, frozenset[str]] = {
    "terminal_response_proposed": frozenset({"post_generation_pre_exec"}),
    "proposed_tool_action": frozenset({"post_generation_pre_exec"}),
    "constraint_violation_reported": frozenset({"post_execution"}),
    "tool_execution_error": frozenset({"post_execution"}),
}

# Probe values for REQUIRED parameters, so a parameterized signal can be evaluated while candidates
# are enumerated. EMPTY, and mutable so a calibration harness can inject one explicitly.
#
# None of the four signals needs one -- `violation_class` is optional -- so this table being empty
# costs nothing today. It exists because the other benchmark shipped it non-empty and had to remove
# it: a probe default exports one measured number into every future proposal, and a discovery claim
# made with that value in the room is a lookup. A signal whose parameter has no measured value
# belongs in the Phi-expansion backlog, not in a probe default with an invented number.
SIGNAL_PROBE_PARAMS: dict[str, Mapping[str, Any]] = {}
