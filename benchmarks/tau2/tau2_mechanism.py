"""THE EXECUTING CODE. Every `binding` string in `tau2_capabilities` names something in this file.

This is the only module in the adapter that imports tau2, so the contract and observability tests run
with no tau-bench checkout.

WHAT THIS IS. A tau-bench `LLMAgent` subclass carrying ONE installed controller -- the 4-tuple
(boundary, signal predicate, action, eta) that core selected. It executes; it does not decide. There is
no table here mapping a kind of failure to a remedy: the controller arrives fully specified and the
mechanism's only judgement is MATERIALIZABILITY (can this host actually run this?).

THE ONE STRUCTURAL FACT EVERYTHING HERE OBEYS. `Orchestrator.step` records only the message this agent
RETURNS; whatever the agent appends to its own `state.messages` never reaches `self.trajectory`. And the
evaluator replays the trajectory's mutating calls through `Environment.set_state`, raising if a replayed
result disagrees with the recorded one. So:

  * withholding a call = never returning it            -- sound, nothing to replay
  * injecting context  = appending to the agent's list -- sound, invisible to the replay
  * rewriting a proposed call BEFORE returning it      -- sound, the rewritten call is what is recorded
  * rewriting a RETURNED ToolMessage in place          -- CORRUPTS the record. Never done here; it is
    why post_execution/reroute is declared with a `disabled_reason` instead of implemented.

SUPPRESS AND REPROMPT ARE KEPT MECHANICALLY DISTINCT. The suppress branch never reads
`eta["instruction"]` and never manufactures one: the agent is told only that its call did not run, and
re-plans from that absence. The toy host shipped `hint or "Archive it instead"` once, which silently
made suppression a reprompt and would have made any comparison between the two families meaningless.
`tests/test_tau2_executor_behavioral.py` asserts the instruction text is absent from what the model saw.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (
    AssistantMessage, MultiToolMessage, ToolMessage, UserMessage,
)
from tau2.environment.toolkit import MUTATES_STATE_ATTR
from tau2.utils.llm_utils import generate

import tau2_state as _st
from tau2_fields import AFTER, GATE, TURN_START

# ------------------------------------------------------------------------------------------------
# Mechanism constants -- part of the EXECUTOR, never candidate parameters.
# ------------------------------------------------------------------------------------------------
#
# What a suppressed call looks like from inside the agent. Deliberately non-directive: it reports that
# the call did not run and says nothing about what to do instead. If this text ever carried task
# guidance, SUPPRESS would be a REPROMPT wearing a different label.
SUPPRESS_NOTICE = "NOT EXECUTED."
SUPPRESS_SIBLING_NOTICE = (
    "NOT EXECUTED. Another tool call in the same message was not executed, so nothing in this "
    "message was run."
)
# Used only when suppression has exhausted its budget and the model produced no words of its own. A
# message must carry content or tool calls to pass `validate_message`.
SUPPRESS_FALLBACK_REPLY = "Let me check what else I can do for you."
# What a withheld ANSWER COMMITMENT looks like from inside the agent. A candidate that calls no tool is
# still a consequential decision -- it commits to replying to the user -- and suppressing it must give
# the agent the same kind of signal a withheld call does, or the model sees its own message followed by
# nothing and simply repeats it. Non-directive for the same reason as SUPPRESS_NOTICE: it reports that
# the reply was not delivered and says nothing about what to do instead.
SUPPRESS_REPLY_NOTICE = "NOT DELIVERED. Your previous message was not sent."
# The only substitution semantics this host implements: the proposed call is REPLACED. Declared as the
# capability's `fixed` value so core refuses an arm asking for anything else at construction, rather than
# the mechanism silently declining at run time and the arm measuring as the control.
REROUTE_SEMANTICS = "replace"


# ------------------------------------------------------------------------------------------------
# FIRING TELEMETRY, counted where the mechanism actually altered the run
# ------------------------------------------------------------------------------------------------
class FiringSink:
    """Execution counts, per case, recorded AT THE SITE that changed the run.

    NOT a registry-derived flag. A registry cannot see whether the code ran, and `train_objective`
    requires `interventions_executed > 0` for an arm to count as engaged -- so a count inferred from
    installation makes every arm look identical and the argmax falls through to a tiebreak.

    `predicate_matches` is counted separately and is NOT engagement: a match that was then budget-
    blocked changed nothing, and folding it into `executed` would overstate what the arm did.

    Thread-safe because tau-bench's batch runner is a `ThreadPoolExecutor` in one process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.executed: int = 0
        self.predicate_matches: int = 0
        self.by_case: dict[str, int] = {}
        self.by_site: dict[str, int] = {}
        self.events: list[dict[str, Any]] = []

    def record(self, *, case_id: str, site: str, detail: Mapping[str, Any] | None = None) -> None:
        """One intervention that ACTUALLY altered the run."""
        with self._lock:
            self.executed += 1
            self.by_case[str(case_id)] = self.by_case.get(str(case_id), 0) + 1
            self.by_site[site] = self.by_site.get(site, 0) + 1
            self.events.append({"case_id": str(case_id), "site": site, **dict(detail or {})})

    def record_match(self) -> None:
        with self._lock:
            self.predicate_matches += 1

    @property
    def cases_fired(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self.by_case))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "interventions_executed": self.executed,
                "predicate_matches": self.predicate_matches,
                "cases_fired": sorted(self.by_case),
                "by_case": dict(self.by_case),
                "by_site": dict(self.by_site),
            }


# ------------------------------------------------------------------------------------------------
# The installed controller
# ------------------------------------------------------------------------------------------------
@dataclass
class Controller:
    """An installed policy: (boundary, predicate, action, eta). The 4-tuple core selects.

    `boundary` is this adapter's own key. `predicate` is the signal core installed, evaluated on the
    observable state this mechanism builds at that boundary. `capability_id` carries WHICH executor core
    judged feasible, so the runner installs that one rather than whichever resolves first.
    """

    boundary: str
    action: str
    predicate: Callable[[Mapping[str, Any]], bool]
    eta: Mapping[str, Any] = field(default_factory=dict)
    label: str = ""
    variant: str = ""
    capability_id: str = ""

    def fires_on(self, state: Mapping[str, Any]) -> bool:
        try:
            return bool(self.predicate(state))
        except Exception:
            # A predicate that raises has not fired. Never let it abort an episode: that would turn a
            # controller bug into an infrastructure error and score as a run the arm never had.
            return False


def _parse_argument_mapping(spec: Any) -> dict[str, str]:
    """`"a -> b, c -> d"` -> {'a': 'b', 'c': 'd'}. The eta key the reroute executor really reads."""
    out: dict[str, str] = {}
    for clause in str(spec or "").split(","):
        if "->" not in clause:
            continue
        src, _, dst = clause.partition("->")
        src, dst = src.strip(), dst.strip()
        if src and dst:
            out[src] = dst
    return out


class ControlledLLMAgent(LLMAgent):
    """An LLMAgent with ONE core-selected controller in front of its decisions.

    With `controller=None` this is stock `LLMAgent` behaviour and is the CONTROL run. That is what makes
    the behavioural probes meaningful: the same class, the same code path, one thing different.
    """

    def __init__(self, tools, domain_policy: str, llm: str, llm_args: Optional[dict] = None,
                 *, controller: Controller | None = None, sink: FiringSink | None = None,
                 case_id: str = "", max_replans: int = 3) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.controller = controller
        self.sink = sink if sink is not None else FiringSink()
        self.case_id = str(case_id)
        self.max_replans = int(max_replans)
        # Tool facts read off THIS agent's own toolset -- real runtime observability, not a table.
        self._known_tools: frozenset[str] = frozenset(t.name for t in (tools or ()))
        self._mutating_tools: frozenset[str] = frozenset(
            t.name for t in (tools or ())
            if getattr(getattr(t, "_func", None), MUTATES_STATE_ATTR, True))
        self._tool_by_name = {t.name: t for t in (tools or ())}
        # Episode-level carried history.
        self._turn_index = 0
        self._assistant_turns = 0
        self._user_turns = 0
        self._tool_errors = 0
        self._consecutive_tool_errors = 0
        self._call_names_by_id: dict[str, str] = {}
        # PER-SITE INJECTION BUDGET. `retry_budget` is a real parameter at every reprompt cell, not
        # only at the gate: an instruction re-injected on every turn of a 20-turn episode is a prompt
        # edit, not a local intervention. The executor reads the budget and stops, so two arms
        # differing only in `retry_budget` genuinely differ in run.
        self._budget_spent: dict[str, int] = {}
        # What the model was actually shown, per generation. The behavioural probes read this rather
        # than a firing flag: "telemetry says it fired" is not evidence.
        self.observed_prompts: list[list[dict[str, Any]]] = []
        self.returned: list[dict[str, Any]] = []

    # -------------------------------------------------------------------- state extraction
    def _fires(self, boundary: str, state: Mapping[str, Any]) -> bool:
        c = self.controller
        if c is None or c.boundary != boundary:
            return False
        if not c.fires_on(state):
            return False
        self.sink.record_match()
        return True

    def _ingest_inbound(self, message) -> bool:
        """Fold the inbound message into carried history. Returns whether it was a tool result."""
        subs = (list(message.tool_messages or []) if isinstance(message, MultiToolMessage)
                else [message] if isinstance(message, ToolMessage) else [])
        for sub in subs:
            if getattr(sub, "error", False):
                self._tool_errors += 1
                self._consecutive_tool_errors += 1
            else:
                self._consecutive_tool_errors = 0
        if isinstance(message, UserMessage):
            self._user_turns += 1
        return bool(subs)

    def _result_states(self, message) -> list[tuple[Any, dict[str, Any]]]:
        """(ToolMessage, post-execution state) for each result in the inbound message."""
        subs = (list(message.tool_messages or []) if isinstance(message, MultiToolMessage)
                else [message] if isinstance(message, ToolMessage) else [])
        out = []
        for sub in subs:
            out.append((sub, _st.result_state(
                case_id=self.case_id, turn_index=self._turn_index,
                is_error=bool(getattr(sub, "error", False)), content=getattr(sub, "content", None),
                tool_name=self._call_names_by_id.get(str(getattr(sub, "id", "")), ""),
                mutating_tools=self._mutating_tools, tool_errors_so_far=self._tool_errors)))
        return out

    def _gate_state(self, candidate: AssistantMessage, replan_attempt: int) -> dict[str, Any]:
        calls = [{"name": tc.name, "arguments": dict(tc.arguments or {})}
                 for tc in (candidate.tool_calls or ())]
        return _st.gate_state(
            case_id=self.case_id, turn_index=self._turn_index, proposed_calls=calls,
            has_content=bool((candidate.content or "").strip()),
            known_tools=self._known_tools, mutating_tools=self._mutating_tools,
            replan_attempt=replan_attempt, consecutive_tool_errors=self._consecutive_tool_errors,
            tool_errors_so_far=self._tool_errors)

    def _turn_start_state(self, inbound_is_tool_result: bool) -> dict[str, Any]:
        return _st.turn_start_state(
            case_id=self.case_id, turn_index=self._turn_index,
            assistant_turns_so_far=self._assistant_turns, user_turns_so_far=self._user_turns,
            inbound_is_tool_result=inbound_is_tool_result,
            consecutive_tool_errors=self._consecutive_tool_errors,
            tool_errors_so_far=self._tool_errors)

    # -------------------------------------------------------------------- generation
    def _generate(self, state, extra: Sequence[Any] = ()) -> AssistantMessage:
        """Call the model. `extra` reaches THIS generation only and is never persisted anywhere.

        Recording the exact prompt is what lets a probe assert an instruction reached the model rather
        than a flag being set beside an unchanged trajectory.
        """
        messages = list(state.system_messages) + list(state.messages) + list(extra)
        self.observed_prompts.append(
            [{"role": getattr(m, "role", ""), "content": str(getattr(m, "content", "") or "")}
             for m in messages])
        return generate(model=self.llm, tools=self.tools, messages=messages,
                        call_name="agent_response", **self.llm_args)

    # -------------------------------------------------------------------- the three boundaries
    def generate_next_message(self, message, state):
        self._turn_index += 1
        inbound_is_tool_result = self._ingest_inbound(message)

        # ---- BOUNDARY: after_tool_result (POST_EXECUTION) -------------------------------------
        # BINDING: append_post_result_guidance. Guidance is appended as a SEPARATE message in the
        # agent's own context. The recorded ToolMessage is NEVER touched: the orchestrator holds the
        # same object and already put it in the trajectory, so an in-place rewrite corrupts the record
        # (it turned 41/114 telecom simulations into infrastructure errors once).
        post_exec_guidance: list[Any] = []
        for sub, rstate in self._result_states(message):
            if self._fires(AFTER, rstate):
                text = str((self.controller.eta or {}).get("instruction") or "").strip()
                if text and self._claim_budget("post_execution/reprompt"):
                    post_exec_guidance.append(UserMessage(role="user", content=text))
                    self.sink.record(case_id=self.case_id, site="post_execution/reprompt",
                                     detail={"turn": self._turn_index,
                                             "result_tool": rstate.get("result_tool", "")})

        # Mirror the base class: the inbound message joins the agent's history exactly once.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        state.messages.extend(post_exec_guidance)

        # ---- BOUNDARY: before_agent_turn (PRE_GENERATION) --------------------------------------
        # BINDING: inject_pre_generation_note. No candidate exists yet, so this conditions on carried
        # context only and reaches the upcoming generation as an extra message.
        pre_gen_extra: list[Any] = []
        if self._fires(TURN_START, self._turn_start_state(inbound_is_tool_result)):
            text = str((self.controller.eta or {}).get("instruction") or "").strip()
            if text and self._claim_budget("pre_generation/reprompt"):
                pre_gen_extra.append(UserMessage(role="user", content=text))
                self.sink.record(case_id=self.case_id, site="pre_generation/reprompt",
                                 detail={"turn": self._turn_index})

        # ---- BOUNDARY: before_tool_dispatch (POST_GENERATION_PRE_EXEC) -------------------------
        extra = list(pre_gen_extra)
        candidate = self._generate(state, extra)
        budget = self._retry_budget()

        for attempt in range(max(1, self.max_replans)):
            gstate = self._gate_state(candidate, attempt)
            if not self._fires(GATE, gstate):
                break
            action = self.controller.action

            if action == "reroute":
                # BINDING: rewrite_proposed_call. The rewritten call is what is returned, executed and
                # recorded, so the evaluator's replay stays consistent.
                rewritten = self._rewrite_call(candidate, gstate)
                if rewritten is None:
                    break                     # not materializable here: the run is unaltered
                candidate = rewritten
                self.sink.record(case_id=self.case_id, site="post_generation_pre_exec/reroute",
                                 detail={"turn": self._turn_index,
                                         "from": gstate.get("proposed_tool", ""),
                                         "to": str((self.controller.eta or {}).get("destination", ""))})
                break                         # one substitution; the result is dispatched

            if attempt >= budget:
                # Budget spent. For SUPPRESS the call must still not execute -- a suppression that
                # eventually lets the call through did not suppress. So the tool calls are stripped and
                # the agent answers in words.
                if action == "suppress":
                    candidate = self._strip_tool_calls(candidate)
                break

            if action == "suppress":
                # BINDING: withhold_proposed_call. The candidate never reaches the orchestrator. NO
                # instruction is supplied and none is manufactured.
                self._record_withheld(state, candidate, instruction="")
                self.sink.record(case_id=self.case_id, site="post_generation_pre_exec/suppress",
                                 detail={"turn": self._turn_index, "attempt": attempt,
                                         "withheld_tool": gstate.get("proposed_tool", "")})
                candidate = self._generate(state)
                continue

            if action == "reprompt":
                # BINDING: inject_replan_instruction. The instruction genuinely enters the context the
                # model is given; two arms differing only in `instruction` therefore differ in run.
                instruction = str((self.controller.eta or {}).get("instruction") or "").strip()
                if not instruction:
                    break
                self._record_withheld(state, candidate, instruction=instruction)
                self.sink.record(case_id=self.case_id, site="post_generation_pre_exec/reprompt",
                                 detail={"turn": self._turn_index, "attempt": attempt})
                candidate = self._generate(state)
                continue

            break                              # noop, or an action with no mechanism here

        state.messages.append(candidate)
        self._assistant_turns += 1
        for tc in (candidate.tool_calls or ()):
            self._call_names_by_id[str(tc.id)] = tc.name
        self.returned.append({
            "turn": self._turn_index,
            "tool_calls": [{"name": tc.name, "arguments": dict(tc.arguments or {})}
                           for tc in (candidate.tool_calls or ())],
            "content": str(candidate.content or ""),
        })
        return candidate, state

    # -------------------------------------------------------------------- executor internals
    def _retry_budget(self) -> int:
        try:
            return max(1, int((self.controller.eta or {}).get("retry_budget", 1) or 1))
        except (TypeError, ValueError, AttributeError):
            return 1

    def _claim_budget(self, site: str) -> bool:
        """Spend one unit of this site's budget, or refuse. Refusing changes nothing and is not a firing."""
        spent = self._budget_spent.get(site, 0)
        if spent >= self._retry_budget():
            return False
        self._budget_spent[site] = spent + 1
        return True

    def _record_withheld(self, state, candidate: AssistantMessage, *, instruction: str) -> None:
        """Put the withheld attempt in the AGENT'S OWN context, well-formed.

        Every tool call needs a paired response or the chat history is malformed. `instruction` is the
        REPROMPT's text; the SUPPRESS path passes "" and gets the neutral non-directive notice.
        """
        state.messages.append(candidate)
        calls = list(candidate.tool_calls or ())
        for i, tc in enumerate(calls):
            body = SUPPRESS_NOTICE if i == 0 else SUPPRESS_SIBLING_NOTICE
            state.messages.append(ToolMessage(id=tc.id, role="tool", content=body,
                                              requestor="assistant", error=False))
        if not calls:
            # A withheld answer commitment. Without this the agent is told nothing at all and the
            # suppression is inert by construction rather than by measurement.
            state.messages.append(UserMessage(role="user", content=SUPPRESS_REPLY_NOTICE))
        if instruction:
            state.messages.append(UserMessage(role="user", content=instruction))

    def _strip_tool_calls(self, candidate: AssistantMessage) -> AssistantMessage:
        content = str(candidate.content or "").strip() or SUPPRESS_FALLBACK_REPLY
        return AssistantMessage(role="assistant", content=content, tool_calls=None)

    def _rewrite_call(self, candidate: AssistantMessage,
                      gstate: Mapping[str, Any]) -> AssistantMessage | None:
        """Substitute the first proposed call with the eta's destination, or decline.

        Declining returns None and records NOTHING: an arm whose destination this host cannot honour
        must measure as unaltered, not as a firing.
        """
        eta = dict(self.controller.eta or {})
        dest = str(eta.get("destination") or "")
        calls = list(candidate.tool_calls or ())
        if not dest or not calls or dest not in self._tool_by_name:
            return None
        # `retry_semantics` IS READ, and this mechanism can honour exactly one value: it swaps the
        # proposed call for the destination. There is no path here that keeps the original and adds a
        # second call -- the agent returns one message and the orchestrator dispatches what it contains.
        # Declining anything else leaves the run unaltered and records no firing, which is what stops an
        # arm asking for semantics this host does not implement from being measured as if it ran.
        if str(eta.get("retry_semantics") or REROUTE_SEMANTICS) != REROUTE_SEMANTICS:
            return None
        mapping = _parse_argument_mapping(eta.get("argument_mapping"))
        src_args = dict(calls[0].arguments or {})
        mapped = {mapping.get(k, k): v for k, v in src_args.items()}
        # SUBSTITUTION SEMANTICS: the rewritten call carries the destination's OWN parameters. An
        # argument the destination does not accept is dropped rather than forwarded -- forwarding it
        # produces `unexpected keyword argument` at dispatch, so the arm would measure an error it
        # caused rather than the repair it claims. Dropping is visible: the rewritten call is what the
        # trajectory records.
        accepted = self._accepted_params(dest)
        new_args = ({k: v for k, v in mapped.items() if k in accepted}
                    if accepted is not None else dict(mapped))
        if not self._satisfies_schema(dest, new_args):
            return None
        head = calls[0].model_copy(update={"name": dest, "arguments": new_args})
        return candidate.model_copy(update={"tool_calls": [head] + calls[1:]})

    def _accepted_params(self, tool_name: str) -> frozenset[str] | None:
        """Parameter names the destination accepts, or None when they cannot be determined."""
        tool = self._tool_by_name.get(tool_name)
        params = getattr(tool, "params", None)
        fields = getattr(params, "model_fields", None)
        return frozenset(fields) if fields is not None else None

    def _satisfies_schema(self, tool_name: str, arguments: Mapping[str, Any]) -> bool:
        """Would this call actually EXECUTE against the destination? Materializability, nothing more.

        Pydantic validation alone is not the test, and assuming it was is a bug this project's own
        behavioural probe caught: `list_all_airports.params` declares no fields and pydantic IGNORES
        unknown ones, so a stray argument validated cleanly -- while the real dispatch returned
        `unexpected keyword argument`. So the check is: every argument is one the destination accepts,
        AND every required parameter is supplied.
        """
        tool = self._tool_by_name.get(tool_name)
        params = getattr(tool, "params", None)
        if params is None:
            return False
        accepted = self._accepted_params(tool_name)
        if accepted is not None and not set(arguments) <= set(accepted):
            return False
        fields = getattr(params, "model_fields", {}) or {}
        required = {n for n, f in fields.items() if getattr(f, "is_required", lambda: False)()}
        if not required <= set(arguments):
            return False
        try:
            params(**dict(arguments))
            return True
        except Exception:
            return False


# ------------------------------------------------------------------------------------------------
# Installation -- the registry seam tau-bench actually provides
# ------------------------------------------------------------------------------------------------
def make_agent_factory(controller: Controller | None, sink: FiringSink,
                       *, max_replans: int = 3) -> Callable[..., ControlledLLMAgent]:
    """A `registry.register_agent_factory` factory carrying one controller.

    `build_agent` passes `task=task`, which is how each agent learns its own case id -- and therefore
    how firings are attributed per case rather than only in aggregate.
    """

    def factory(tools, domain_policy, llm=None, llm_args=None, task=None, **_kw):
        return ControlledLLMAgent(
            tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args,
            controller=controller, sink=sink,
            case_id=str(getattr(task, "id", "") or ""), max_replans=max_replans)

    return factory


def controller_spec(controller: Controller | None) -> dict[str, Any]:
    """The installable, serializable form of a controller. What a runner writes and re-reads."""
    if controller is None:
        return {}
    return {"boundary": controller.boundary, "action": controller.action,
            "eta": {k: (v if isinstance(v, (int, float, bool, str)) else str(v))
                    for k, v in dict(controller.eta).items()},
            "label": controller.label, "variant": controller.variant,
            "capability_id": controller.capability_id}
