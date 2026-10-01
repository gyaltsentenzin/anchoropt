"""COPY THIS FILE to start a new benchmark adapter. Everything AnchorOpt needs is in one object.

    cp adapters/adapter_template.py adapters/my_benchmark.py     # then fill in the TODOs

AnchorOpt optimizes three coordinates over a residual failure -- WHERE to intervene, WHAT condition
decides, HOW to intervene -- and it learns all three. This adapter's job is only to expose YOUR
runtime: which events are decisions, what is observable at each, which actions exist, and how to
parameterize them. The adapter never chooses a locus, a signal or an action, and it never needs to
know what AnchorOpt is searching for.

WHAT YOU DO NOT NEED TO BUILD. If you already run this benchmark under your own harness you have
the episode loop, trajectories, scoring and model access. Reuse all of it: `normalize_event` adapts
your trajectory rows, and your existing runner becomes the `evaluate` callback. You do not need to
import anything from `benchmarks/bfcl_v4/`, and you should not read it as a spec -- it is one
instance of this interface carrying a lot of history.

THE 20-LINE USAGE, once this file is filled in:

    from anchoropt.learning.structured_search import optimize_residual
    from adapters.my_benchmark import ADAPTER

    events = [ADAPTER.normalize_event(r) for r in my_trajectory_rows]
    states = [ADAPTER.observable_state(e) for e in events]

    out = optimize_residual(residual, runtime=ADAPTER, host=ADAPTER.HOST,
                            events=events, states=states,
                            evaluate=my_paired_eval)      # or evaluate=None for structure only
    if out.validated:
        install(out.promoted)

REQUIRED vs OPTIONAL is marked on every hook below. With only the REQUIRED ones, `optimize_residual`
localizes boundaries, checks expressibility, and reports what it could not do -- it just cannot ground
an action, so it will report ACTION_UNAVAILABLE / ETA_UNSUPPORTED rather than crash. Add one grounder
and you get candidates; add `evaluate` and you get promotions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# IncisionPoint and Action are CORE vocabulary, not benchmark vocabulary: they describe any LLM call
# (before generation / after generation but before execution / after execution) and the four ways to
# intervene. Use them rather than inventing a parallel enum -- HostProfile is typed on them.
from anchoropt.anchor import Action, IncisionPoint
from anchoropt.runtime import HostProfile

# ================================================================================================
# 0. YOUR OBSERVABLE FIELDS  (REQUIRED)
# ================================================================================================
#
# The typed alphabet AnchorOpt may synthesize conditions over. Declare only facts your runtime really
# records, and declare WHERE each is readable: a field listed at a boundary must be evaluable there.
# Getting this wrong is the most common porting bug -- a condition evaluated where the fact does not
# exist answers False for a structural reason that looks exactly like "the condition did not hold".


class Field:
    """One observable. `boundaries` are the IncisionPoint VALUES where this field is readable."""

    def __init__(self, name: str, type_: type, boundaries: Sequence[str], doc: str = "",
                 enum: Sequence[Any] = ()):
        self.name, self.type, self.doc, self.enum = name, type_, doc, tuple(enum)
        self.boundaries = tuple(boundaries)


_PRE = IncisionPoint.POST_GENERATION_PRE_EXEC.value
_POST = IncisionPoint.POST_EXECUTION.value

FIELDS: dict[str, Field] = {
    # TODO replace with your own. Two illustrative shapes:
    #   a fact known once a call is PROPOSED but not yet run  -> readable at _PRE and _POST
    #   a fact that only exists once it RAN (a result, an error, a score) -> _POST only
    "proposes_action": Field("proposes_action", bool, (_PRE, _POST),
                            "the model proposed at least one tool call at this step"),
    "step_index": Field("step_index", int, (_PRE, _POST), "position in the episode"),
    "last_error_kind": Field("last_error_kind", str, (_POST,),
                             "classified error from the most recent result, or None",
                             enum=("not_found", "rejected", "timeout")),
    "result_score": Field("result_score", float, (_POST,),
                          "quality of what came back, if your runtime produces one"),
}


# ================================================================================================
# 1. BOUNDARY IDENTIFICATION  (REQUIRED)  -- WHERE AnchorOpt may intervene
# ================================================================================================


def normalize_event(row: Mapping[str, Any]) -> dict[str, Any]:
    """REQUIRED (by you, not by core): one row of YOUR trajectory -> a plain dict AnchorOpt reads.

    Core never parses your trace format; it consumes whatever this returns. Keep the keys the same as
    your FIELDS names wherever possible so `observable_state` is nearly a pass-through.
    """
    # TODO map your own trajectory row.
    return {
        "status": row.get("status"),
        "calls": list(row.get("tool_calls") or []),
        "results": list(row.get("tool_results") or []),
        "step_index": row.get("step"),
    }


def is_decision(event: Mapping[str, Any]) -> bool:
    """REQUIRED. Was this event a point where the agent made a CONSEQUENTIAL choice?

    Include COMMITTING TO A FINAL ANSWER, not only acting. That is the most valuable lesson from the
    first port: requiring a tool call here meant episodes that answered without acting contributed no
    boundary at all, every residual derived exactly one boundary, and the backward search could never
    be observed to move. If your traces mark termination with a status, treat it as a decision.
    """
    # TODO
    return bool(event.get("calls")) or str(event.get("status") or "") in {"final_answer", "done"}


def boundary_key(event: Mapping[str, Any]) -> str:
    """REQUIRED. Which decision point this event belongs to. Return an IncisionPoint VALUE.

    Rule of thumb: a result is present -> POST_EXECUTION; a call is proposed but not yet run, or the
    agent is committing to an answer -> POST_GENERATION_PRE_EXEC. Check results FIRST so an event that
    both answers and carries results is placed by its results.
    """
    # TODO
    if event.get("results"):
        return IncisionPoint.POST_EXECUTION.value
    if event.get("calls") or str(event.get("status") or "") in {"final_answer", "done"}:
        return IncisionPoint.POST_GENERATION_PRE_EXEC.value
    return ""


def boundary_from_key(key: str):
    """REQUIRED. The inverse of `boundary_key`. Core owns no key->locus table on purpose.

    Return None for a key you do not recognise; that boundary is then simply not searched.
    """
    try:
        return IncisionPoint(str(getattr(key, "value", key)))
    except (ValueError, TypeError):
        return None


def label_for(key: str) -> str:
    """OPTIONAL, descriptive only. Never read by the search -- useful in figures and logs."""
    return {_PRE: "commitment gate", _POST: "post-outcome"}.get(str(key), "")


def depends_on(key: str) -> Sequence[str]:
    """OPTIONAL. A generic dependency relation, if realized order does not reflect dependence.

    Omit it entirely and core orders boundaries by the order the events happened in, which is right for
    most runtimes. This is a DEPENDENCY relation, deliberately not a named stage taxonomy.
    """
    return ()


# ================================================================================================
# 2. OBSERVABLE STATE AND SIGNALS  (REQUIRED)  -- WHAT conditions can decide
# ================================================================================================


def observable_state(event: Mapping[str, Any]) -> dict[str, Any]:
    """REQUIRED (by you). The state your signals read, keyed by FIELDS names.

    EPISODE-LEVEL FACTS MUST ACCUMULATE. A field like "has the agent consulted the other source yet"
    is about the episode, not the step; computing it per step in isolation makes it constant and the
    discrimination check will correctly refuse every predicate built on it.
    """
    # TODO
    return {
        "proposes_action": bool(event.get("calls")),
        "step_index": event.get("step_index"),
        "last_error_kind": event.get("error_kind"),
        "result_score": event.get("score"),
        "boundary": boundary_key(event),
    }


def synthesis_fields(boundary: Any = None) -> dict[str, Field]:
    """REQUIRED. The typed fields readable AT `boundary`. Core synthesizes conditions over these.

    Must filter by boundary. Returning everything everywhere is the same defect as a mis-declared
    field: core will build a condition that cannot be evaluated where it is installed.
    """
    want = str(getattr(boundary, "value", boundary or ""))
    if not want:
        return dict(FIELDS)
    return {n: f for n, f in FIELDS.items() if want in f.boundaries}


# --- Phi: the conditions your runtime already knows how to test -----------------------------------
#
# Declared signals are your starting Phi. If you have none, declare an empty tuple: AnchorOpt will
# report SIGNAL_BLOCKED and synthesize conditions over FIELDS instead. Starting with none is a
# perfectly good way to port -- you lose nothing except a cheaper first search.

_DECLARED = {
    # TODO your own named conditions, each a predicate over `observable_state`.
    "no_action_proposed": lambda st: st.get("proposes_action") is False,
    "lookup_failed": lambda st: st.get("last_error_kind") == "not_found",
}
_SIGNAL_BOUNDARIES = {
    "no_action_proposed": frozenset({IncisionPoint.POST_GENERATION_PRE_EXEC}),
    "lookup_failed": frozenset({IncisionPoint.POST_EXECUTION}),
}

# Signals AnchorOpt synthesized this session. Keep the predicate AND the boundary it was validated
# at -- dropping the boundary is a real bug we shipped once: `signal_boundaries` then had nothing to
# answer for exactly the signals expansion creates, and the caller read that as "observable nowhere".
_EXPANDED: dict[str, Any] = {}
_EXPANDED_BOUNDARIES: dict[str, frozenset] = {}


def declared_signals() -> tuple[str, ...]:
    """REQUIRED. Current Phi: your own signals plus anything expansion installed."""
    return tuple(_DECLARED) + tuple(_EXPANDED)


def signal_boundaries(signal: str) -> frozenset:
    """REQUIRED. Where `signal` is observable. Return a frozenset of IncisionPoint."""
    if signal in _EXPANDED:
        return _EXPANDED_BOUNDARIES.get(signal, frozenset())
    return _SIGNAL_BOUNDARIES.get(signal, frozenset())


def evaluate_signal(signal: str, state: Mapping[str, Any],
                    params: Mapping[str, Any] | None = None) -> bool:
    """REQUIRED. Does `signal` hold in `state`? Raise KeyError for a name you do not know."""
    if signal in _EXPANDED:
        return bool(_EXPANDED[signal](state))
    if signal not in _DECLARED:
        raise KeyError(f"no evaluator for signal {signal!r}; declared: {sorted(_DECLARED)}")
    return bool(_DECLARED[signal](state))


def install_signal(name: str, predicate: Any, *, boundary: Any = None,
                   provenance: str = "") -> None:
    """REQUIRED for signal expansion (omit only if you never want Phi widened).

    RETAIN THE BOUNDARY. It is the incision point expansion validated the predicate at, and it is the
    only honest basis for later claiming the signal is observable anywhere.
    """
    if name in _DECLARED:
        raise ValueError(f"{name!r} is a declared signal; expansion may not redefine it")
    _EXPANDED[name] = predicate
    pt = boundary if isinstance(boundary, IncisionPoint) else boundary_from_key(boundary)
    _EXPANDED_BOUNDARIES[name] = frozenset({pt}) if pt is not None else frozenset()


def reset_expanded_signals() -> None:
    """OPTIONAL but strongly recommended: lets one experiment avoid inheriting another's Phi."""
    _EXPANDED.clear()
    _EXPANDED_BOUNDARIES.clear()


def probe_params(signal: str) -> dict[str, Any]:
    """OPTIONAL. Concrete parameter values so a parameterized signal is evaluable during search.

    A parameterized signal with no probe params is pruned as unevaluable -- which reads like "the
    signal did not apply". Supply something, even a midpoint.
    """
    return {}


def states_at(boundary: Any, states: Sequence[Mapping[str, Any]]) -> list:
    """REQUIRED once you declare synthesis fields. Project states onto ONE boundary's information set.

    Core FAILS CLOSED without this: it refuses to validate a predicate on states from a different
    boundary, because a predicate checked against a result that does not exist yet can look
    discriminating and then never fire at runtime. We shipped exactly that bug.

    If your states are already per-boundary, return them unchanged. If a boundary sees something
    NARROWER -- a commitment gate sees the proposed call but not its result -- rebuild the states from
    only what is knowable there, and return [] rather than something approximate if you cannot.
    """
    want = str(getattr(boundary, "value", boundary))
    if want != _PRE:
        return list(states or ())
    # TODO if your commitment gate sees proposed calls, emit one state per proposal carrying only
    # pre-dispatch facts. Returning [] makes core fail closed, which is the safe default.
    return list(states or ())


def parameter_domains(signal: str) -> tuple:
    """OPTIONAL. Tunable domains for a parameterized signal. Omit if none of yours are."""
    return ()


# ================================================================================================
# 3. ACTIONS AND ETA GROUNDING  (at least ONE grounder REQUIRED for candidates)  -- HOW
# ================================================================================================
#
# HOST declares which actions your runtime can really perform at each boundary. It narrows the
# structural grid and can never widen it: SUPPRESS is only meaningful before execution (afterwards
# there is nothing left to cancel), REROUTE only after.

HOST = HostProfile(
    name="my_benchmark_host",       # TODO name YOUR runtime, not the benchmark
    executable={
        IncisionPoint.POST_GENERATION_PRE_EXEC: frozenset({Action.REPROMPT, Action.SUPPRESS}),
        IncisionPoint.POST_EXECUTION: frozenset({Action.REPROMPT, Action.REROUTE}),
    })

# Each grounder returns a LIST of variants; each variant becomes its own evaluation arm, because
# choosing among them is measurement's job, not the adapter's. Return [] when your runtime genuinely
# cannot parameterize that action for this signal -- core reports *_INFEASIBLE with the reason and
# moves on, which is exactly what you want recorded.
#
# DO NOT whitelist by SIGNAL NAME. We shipped that bug: a two-name whitelist meant no synthesized
# signal could ever compose with SUPPRESS, so the search produced zero arms while the action was
# admissible and an executor existed -- indistinguishable from an absent primitive. Ground on the
# STRUCTURE of the boundary and the action, never on which signal asked.


def ground_reprompt(signal: str, boundary: Any) -> list[dict]:
    """OPTIONAL (required for REPROMPT arms). eta needs `instruction` and `retry_budget`."""
    return [{"variant": "retry_with_hint",
             "eta": {"instruction": "TODO an instruction your runtime can actually inject",
                     "retry_budget": 1},
             "detail": "ask the model to try again with guidance"}]


def ground_suppress(signal: str, boundary: Any) -> list[dict]:
    """OPTIONAL (required for SUPPRESS arms). eta needs `suppressed_operation` and `preservation`.

    The boundary IS part of the requirement -- suppression only makes sense before execution.
    """
    if str(getattr(boundary, "value", boundary)) != _PRE:
        return []
    return [{"variant": "cancel_proposed",
             "eta": {"suppressed_operation": "the operation this signal implicates",
                     "preservation": "TODO what guarantees nothing is lost by cancelling"},
             "detail": "cancel before dispatch"}]


def ground_substitute_destinations(signal: str, boundary: Any) -> list[dict]:
    """OPTIONAL (required for REROUTE arms). eta needs `destination`, `argument_mapping`,
    `retry_semantics`. Return EVERY credible destination -- each is a separate arm."""
    return []          # TODO your real alternative destinations, or leave empty


def ground_transforms(signal: str, boundary: Any) -> list[dict]:
    """OPTIONAL. eta needs `target_surface`, `operator`, `preservation`."""
    return []


def executor_supports(boundary: Any, action: Any, signal: str,
                      eta: Mapping[str, Any]) -> tuple[bool, str]:
    """REQUIRED once you ground anything. Can your runtime REALLY execute this (action, eta)?

    Return (False, reason) when it cannot. Say so rather than coercing a parameter: an executor that
    silently substitutes its own value runs different semantics under the candidate's name, and the
    measured delta is then not caused by what was proposed.

    A TRIGGER-FREE EXECUTOR IS THE CONTRACT. If your executor carries its own arming condition it
    silently overrides phi, two differently-parameterized controllers fire on identical states, and the
    whole abstraction is untestable. phi owns WHEN; the executor owns only HOW.
    """
    return True, "TODO check your executor really consumes this eta unchanged"


def policy_class_for(signal: str):
    """OPTIONAL. Only if you distinguish deterministic from parameterized policies."""
    from anchoropt.learning.policy_class import PolicyClass
    return PolicyClass.DETERMINISTIC


# ================================================================================================
# 4. OPTIONAL: commitment semantics, used by target-population accounting
# ================================================================================================


def commits_to_answer(event: Mapping[str, Any]) -> bool:
    """OPTIONAL. Did the agent commit to a final answer at this event, rather than act again?"""
    return str(event.get("status") or "") in {"final_answer", "done"}


def actions_in(event: Mapping[str, Any]) -> tuple:
    """OPTIONAL. The actions taken at this event, in your trace's own shape."""
    return tuple(event.get("calls") or ())


# ================================================================================================
# 5. THE EVALUATION CALLBACK -- YOUR EXISTING RUNNER, WRAPPED
# ================================================================================================
#
# NOT part of the adapter object: it is passed to optimize_residual as `evaluate=`. It receives one
# candidate arm and returns a comparable objective (higher is better), or None if unmeasurable.
#
# `evaluate=None` is a first-class mode: the search still localizes, expands and grounds, and stops at
# REALIZABLE_UNMEASURED -- structural rediscovery with no claim of benefit. Port in that mode first.


def make_evaluator(run_paired, *, baseline):
    """Wrap your existing benchmark runner into the callback AnchorOpt expects.

    `run_paired(arm) -> (control_correct, arm_correct)` over the SAME case set.

    THREE RULES, each of which has cost this project a run:
      1. compare over the SAME cases -- a delta across two case sets is not a paired result;
      2. read firings from your own per-episode telemetry, not from a registry of what was CONFIGURED:
         a configured-but-never-fired arm looks identical to a working one in a summary;
      3. do not let a query-time controller modify SETUP episodes. If your benchmark builds state
         before scoring, the controller must be inert during that build, or the arm's cases are scored
         against state the arm itself changed.
    """
    def evaluate(arm) -> Any | None:
        control, treated = run_paired(arm)
        if set(control) != set(treated):
            return None                    # denominator mismatch: not a paired comparison
        return sum(treated.values()) - sum(control.values())
    return evaluate


# ================================================================================================
# THE ADAPTER OBJECT -- pass this as `runtime=`
# ================================================================================================
#
# `optimize_residual` duck-types everything above, so a MODULE works as the adapter too:
#     import adapters.my_benchmark as ADAPTER
# This class exists for callers who prefer injection, and to make the required surface enumerable.


class Adapter:
    """The whole contract as one object."""

    name = "my_benchmark"
    HOST = HOST

    # boundary identification
    normalize_event = staticmethod(normalize_event)
    is_decision = staticmethod(is_decision)
    boundary_key = staticmethod(boundary_key)
    boundary_from_key = staticmethod(boundary_from_key)
    label_for = staticmethod(label_for)

    # observables and signals
    observable_state = staticmethod(observable_state)
    synthesis_fields = staticmethod(synthesis_fields)
    declared_signals = staticmethod(declared_signals)
    signal_boundaries = staticmethod(signal_boundaries)
    evaluate_signal = staticmethod(evaluate_signal)
    install_signal = staticmethod(install_signal)
    reset_expanded_signals = staticmethod(reset_expanded_signals)
    probe_params = staticmethod(probe_params)
    states_at = staticmethod(states_at)
    parameter_domains = staticmethod(parameter_domains)

    # actions
    ground_reprompt = staticmethod(ground_reprompt)
    ground_suppress = staticmethod(ground_suppress)
    ground_substitute_destinations = staticmethod(ground_substitute_destinations)
    ground_transforms = staticmethod(ground_transforms)
    executor_supports = staticmethod(executor_supports)
    policy_class_for = staticmethod(policy_class_for)

    # optional commitment semantics
    commits_to_answer = staticmethod(commits_to_answer)
    actions_in = staticmethod(actions_in)


ADAPTER = Adapter()

# The required surface, enumerated so `scripts/check_adapter.py` can verify a port without running it.
REQUIRED_HOOKS = (
    "is_decision", "boundary_key", "boundary_from_key",
    "synthesis_fields", "declared_signals", "signal_boundaries", "evaluate_signal",
    "HOST",
)
OPTIONAL_HOOKS = (
    "normalize_event", "observable_state", "label_for", "depends_on",
    "install_signal", "reset_expanded_signals", "probe_params", "parameter_domains",
    "states_at",
    "ground_reprompt", "ground_suppress", "ground_substitute_destinations", "ground_transforms",
    "executor_supports", "policy_class_for", "commits_to_answer", "actions_in",
)
