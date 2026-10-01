"""U_H(l) in its BOUND form: one cell per (boundary, action) this host can really run.

A declaration is not an executor. Each cell below names the code in `tau2_mechanism` that RUNS and the
eta keys that code actually READS -- verified against the source, not inferred from what the action
sounds like. An unbound cell is a ghost and core refuses it; a contract-required key outside `consumes`
is inert eta and core refuses that too.

`tests/test_tau2_executor_behavioral.py` proves each enabled cell by BEHAVIOUR: run the real mechanism
with the controller absent and present, and assert the runs differ in the way the action claims. A cell
that only sets a flag fails that test, and must.
"""

from __future__ import annotations

from collections.abc import Mapping

from anchoropt.anchor import IncisionPoint
from anchoropt.learning.executor_capability import ExecutorCapability

_PRE = IncisionPoint.PRE_GENERATION.value
_PG = IncisionPoint.POST_GENERATION_PRE_EXEC.value
_PE = IncisionPoint.POST_EXECUTION.value

_M = "tau2_mechanism.ControlledLLMAgent.generate_next_message"

CAPABILITIES: Mapping[tuple[str, str], ExecutorCapability] = {
    # --------------------------------------------------------------------------------------------
    # PRE_GENERATION / REPROMPT
    # --------------------------------------------------------------------------------------------
    (_PRE, "reprompt"): ExecutorCapability(
        boundary=_PRE, action="reprompt",
        binding=f"{_M}:inject_pre_generation_note",
        # `retry_budget` caps how many times the note may be injected in one episode. It is genuinely
        # read: an uncapped note on every turn is a prompt edit rather than a local intervention, and
        # core's REPROMPT contract requires the key precisely because a reprompt without a bound is not
        # one. An earlier version of this cell omitted it and every arm here was correctly refused
        # `executor_cannot_consume_eta`.
        consumes=("instruction", "retry_budget"),
        eta_delivered_via={"instruction": "the extra message appended to the list passed to generate()"},
        signals=(), signal_agnostic=True,
        detail=("appends the instruction to the messages handed to this turn's generate() call; it "
                "reaches the model and is persisted nowhere"),
    ),
    # --------------------------------------------------------------------------------------------
    # POST_GENERATION_PRE_EXEC / REPROMPT
    # --------------------------------------------------------------------------------------------
    (_PG, "reprompt"): ExecutorCapability(
        boundary=_PG, action="reprompt",
        binding=f"{_M}:inject_replan_instruction",
        consumes=("instruction", "retry_budget"),
        # An empty instruction makes this cell a no-op, and a candidate that supplies none is not the
        # arm that would run. Declaring the requirement lets core refuse it instead of measuring it.
        requires_eta={"instruction": ()},
        eta_delivered_via={"instruction": "a user-role message appended to the agent's own context"},
        signals=(), signal_agnostic=True,
        detail=("withholds the proposed call, appends the instruction to the agent's context, and "
                "regenerates, up to retry_budget times"),
    ),
    # --------------------------------------------------------------------------------------------
    # POST_GENERATION_PRE_EXEC / SUPPRESS
    # --------------------------------------------------------------------------------------------
    (_PG, "suppress"): ExecutorCapability(
        boundary=_PG, action="suppress",
        binding=f"{_M}:withhold_proposed_call",
        # The executor reads the budget and NOTHING else. Which call is cancelled comes from the
        # candidate's own tool_calls, not from the arm -- so `suppressed_operation` is the arm's
        # identity rather than executor input, declared via eta_is_computed instead of over-claimed.
        consumes=("retry_budget",),
        eta_is_computed=True,
        signals=(), signal_agnostic=True,
        # The replay variant is NOT declared: see `ground_suppress`. `preservation` has no executor
        # here and a remove-outright suppression preserves nothing, so it must not claim to.
        unsupported_eta={
            "preservation": ("nothing replays a withheld call's recorded result: the runtime keeps no "
                             "result store, so the clause would have no executor"),
        },
        detail=("the proposed call is never returned to the orchestrator, so it is absent from the "
                "trajectory and from the evaluator's replay; no instruction is supplied"),
    ),
    # --------------------------------------------------------------------------------------------
    # POST_GENERATION_PRE_EXEC / REROUTE  (substitute)
    # --------------------------------------------------------------------------------------------
    (_PG, "reroute"): ExecutorCapability(
        boundary=_PG, action="reroute",
        binding=f"{_M}:rewrite_proposed_call",
        # `retry_semantics` is in `consumes` because the code READS it and declines a value it cannot
        # execute. Omitting it refused all five grounded destinations with `executor_cannot_consume_eta`:
        # core's REROUTE/substitute contract requires the key, and an arm carrying eta no executor reads
        # measures a different intervention. `fixed` then makes the one implementable value explicit, so
        # an arm asking for other semantics is refused at construction instead of running as the control.
        consumes=("destination", "argument_mapping", "retry_semantics"),
        requires_eta={"destination": ()},
        fixed={"retry_semantics": "replace"},
        eta_delivered_via={"destination": "the rewritten ToolCall.name returned to the orchestrator"},
        signals=(), signal_agnostic=True,
        detail=("replaces the proposed call with the destination after validating the mapped arguments "
                "against the destination's own parameter model; the rewritten call is what executes "
                "and what is recorded, so the evaluator's replay stays consistent"),
    ),
    # --------------------------------------------------------------------------------------------
    # POST_EXECUTION / REPROMPT
    # --------------------------------------------------------------------------------------------
    (_PE, "reprompt"): ExecutorCapability(
        boundary=_PE, action="reprompt",
        binding=f"{_M}:append_post_result_guidance",
        # `retry_budget` bounds the injections per episode; see the pre_generation cell.
        consumes=("instruction", "retry_budget"),
        requires_eta={"instruction": ()},
        eta_delivered_via={"instruction": "a separate user-role message after the recorded result"},
        signals=(), signal_agnostic=True,
        detail=("appends guidance after the returned result as a SEPARATE message; the recorded "
                "ToolMessage is never modified"),
    ),
    # --------------------------------------------------------------------------------------------
    # POST_EXECUTION / REROUTE -- DECLARED AND DISABLED
    # --------------------------------------------------------------------------------------------
    #
    # Disabled rather than deleted, because the cell is not unconsidered: it was implemented once and
    # measured as harmful. Deleting it makes it look unexamined and the next porter re-adds it. Core
    # refuses arms here WITH this reason.
    (_PE, "reroute"): ExecutorCapability(
        boundary=_PE, action="reroute",
        binding=f"{_M}:append_post_result_guidance",
        consumes=(),
        eta_is_computed=True,
        disabled_reason=(
            "no sound post-execution re-dispatch exists in this runtime. The agent cannot dispatch "
            "tools -- only the orchestrator does -- so a repair would have to rewrite the returned "
            "ToolMessage, and the orchestrator holds that same object and has already recorded it. An "
            "in-place rewrite corrupts the trajectory the evaluator replays through "
            "Environment.set_state; the earlier hand-authored version of exactly this turned 41 of 114 "
            "telecom simulations into infrastructure errors."),
        detail="unavailable: see disabled_reason",
    ),
    # POST_EXECUTION / SUPPRESS is absent for the structural reason -- after execution there is nothing
    # left to cancel. `anchor.exclusion_reason` enforces that independently, so declaring it here would
    # be redundant rather than additive.
}


def capability(boundary, action) -> ExecutorCapability | None:
    return CAPABILITIES.get((str(getattr(boundary, "value", boundary)),
                             str(getattr(action, "value", action))))


def audit() -> dict:
    """What is bound, what is disabled, what is a ghost. Printed by the round driver."""
    from anchoropt.learning.executor_capability import (
        disabled_capabilities, unbound_capabilities,
    )
    caps = tuple(CAPABILITIES.values())
    return {
        "unbound": list(unbound_capabilities(caps)),
        "disabled": list(disabled_capabilities(caps)),
        "bound_and_enabled": sorted(f"{c.boundary}/{c.action}" for c in caps
                                    if c.is_bound and c.is_enabled),
    }
