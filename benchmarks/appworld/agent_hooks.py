"""The live executor: an `Agent` subclass that actually runs a controller against AppWorld.

WHY A WHOLESALE OVERRIDE, NOT A WRAPPER AROUND THE BASE METHOD
----------------------------------------------------------------
`SimplifiedReActCodeAgent.next_execution_inputs_usage_and_status` is the one method that touches
every incision point this adapter declares: it appends the previous cell's output (where POST_EXECUTION
state becomes visible), calls the language model (PRE_GENERATION is "before this call"), and extracts
the next cell (POST_GENERATION_PRE_EXEC is "after this, before `world.batch_execute`"). Wrapping it
from outside would mean re-deriving those three sites from what the method returns, which is exactly
the state the base method's own locals already have. So this file copies the method's body (verified
verbatim against `<appworld-checkout>/experiments/code/simplified/react_code_agent.py`) and weaves
the three checks in at the sites they actually occur, preserving 100% of the surrounding logging and
message-management logic.

ONE CONTROLLER PER RUN, NOT PER BOUNDARY
------------------------------------------
`run_round.py` measures one arm at a time (build controller -> run the corpus -> record the delta),
mirroring `examples/toy_host/demo.py`'s `Evaluator` and `toy_host.runtime.ToyRuntime.run_episode`, both
of which take a single `controller` argument. So `self.controller` here is one `Controller` (or
`None` for the incumbent/control run), never a collection keyed by boundary. Because an arm's
`(boundary, action)` pair is fixed for the whole run, at most one of the three boundary checks below
ever does anything non-trivial for a given instance -- the other two see `self.controller.boundary !=
<that locus>` and fall through exactly like the base class would.

BUDGET AND FIRING TELEMETRY, MIRRORING `toy_host.runtime.ToyRuntime`
-----------------------------------------------------------------------
`self.budget_spent` is keyed by `task_id` and is never reset between tasks -- it is corpus-level state,
like the toy host's own attribute of the same name. `self._consecutive_errors` / `self._last_error_kind`
/ `self._last_target_app` / `self._last_target_api` ARE reset per task, in `initialize`, because they
describe one episode's history. Firing telemetry (`fired_tasks`, `executed`) increments only where a
mechanism actually altered what ran -- never on a bare predicate match that budget then blocked -- so
`ThetaResult.interventions_executed` (populated from `executed` by `run_round.py`) can't overstate
engagement, the same rule the toy host's own comment states.

WHAT "WITHHOLD" MEANS FOR A REAL LANGUAGE MODEL, NOT A FIXED IF/ELSE
------------------------------------------------------------------------
The toy host's suppress mechanism removes a call from a fixed decision function and the agent
"re-plans" deterministically. There is no such hook into a real model's decoding, so the only lever
this host has for a genuine (not telemetry-only) SUPPRESS effect is: discard the generated cell before
it runs, tell the model only that nothing was accepted (never why, never what to do differently -- that
would make this REPROMPT wearing a different name), and let it generate again. This is a design
decision this plan's text did not spell out to the byte level; it is called out here, in case the
behavioral probe in `probes.py` needs to be read against it.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Sequence
from typing import Any

from appworld.common.path_store import path_store

from appworld_agents.code.common.usage_tracker import Usage
from appworld_agents.code.simplified.agent import Agent, ExecutionIO, Status
from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent

from anchoropt.anchor import IncisionPoint

from adapter import exec_fields_from_output, gate_fields_from_code
from controller import Controller

_PRE = IncisionPoint.PRE_GENERATION.value
_PG = IncisionPoint.POST_GENERATION_PRE_EXEC.value
_PE = IncisionPoint.POST_EXECUTION.value


def _api_docs_lookup_code(target_app: str, target_api: str) -> str:
    """The REROUTE substitute: query `apis.api_docs` instead of running the unresolved call.

    Parameter names verified against `data/api_docs/standard/api_docs.json`'s own entries for
    `show_api_doc` (`app_name`, `api_name`), `show_api_descriptions` (`app_name`), and
    `show_app_descriptions` (no parameters) -- narrowing to the most specific query the cached
    target actually supports, never guessing an API name that itself might not exist.
    """
    if target_app and target_api:
        return f"print(apis.api_docs.show_api_doc(app_name={target_app!r}, api_name={target_api!r}))"
    if target_app:
        return f"print(apis.api_docs.show_api_descriptions(app_name={target_app!r}))"
    return "print(apis.api_docs.show_app_descriptions())"


class _GenerationFailed(Exception):
    """Carries a model-call error out of `_generate_and_extract` for the caller to convert into
    the same `Status(failed=True, ...)` the base class returns -- on either the first attempt or a
    SUPPRESS/REPROMPT-triggered regeneration; an LM error is terminal regardless of which attempt."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@Agent.register("appworld_anchoropt_react_code_agent")
class AppWorldReActAgent(SimplifiedReActCodeAgent):
    """`SimplifiedReActCodeAgent`, with the three declared boundaries wired to a live controller."""

    def __init__(self, *args: Any, controller: Controller | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.controller = controller
        self.budget_spent: dict[str, int] = {}
        self.fired_tasks: set[str] = set()
        self.executed: int = 0
        self._fired_counts: dict[str, int] = {}
        self._consecutive_errors: int = 0
        self._last_error_kind: str = ""
        self._last_target_app: str = ""
        self._last_target_api: str = ""

    def initialize(self, world: Any) -> None:
        super().initialize(world)
        self._consecutive_errors = 0
        self._last_error_kind = ""
        self._last_target_app = ""
        self._last_target_api = ""

    # ------------------------------------------------------------------ boundary state, mirroring
    # ------------------------------------------------------------------ adapter.py's own field shapes
    def _pre_state(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "consecutive_errors": self._consecutive_errors,
            "last_error_kind": self._last_error_kind,
        }

    def _pg_state(self, code: str) -> dict[str, Any]:
        # Same `self.step_number` `_pre_state` reports, so the live gate state matches what
        # `residual.py` reconstructs offline for the same turn -- the two must stay identical or a
        # signal mined offline will not fire live.
        state = gate_fields_from_code(code, self.step_number)
        self._last_target_app = state["target_app"]
        self._last_target_api = state["target_api"]
        return state

    # ------------------------------------------------------------------ budget / telemetry, shared
    # ------------------------------------------------------------------ across all six mechanisms
    def _fired_marker_path(self, task_id: str) -> str | None:
        """`tasks/<task_id>/misc/anchoropt_fired`, mirroring AppWorld's own `misc/finished` marker."""
        experiment_name = getattr(self.world, "experiment_name", "") or ""
        if not experiment_name:
            return None
        return os.path.join(
            path_store.experiment_outputs, experiment_name, "tasks", task_id, "misc",
            "anchoropt_fired",
        )

    def _record_fired(self, task_id: str) -> None:
        """Records a firing in memory AND on disk, at the moment it happens.

        The on-disk marker exists because `skip_if_finished` makes a resumed run skip episodes it
        already completed: those episodes never re-enter this process, so an in-memory set alone would
        come back empty for them and a scorer filtering on it would discard every real gain and loss.
        Writing the marker at fire time -- per episode, from the executor itself -- keeps firing
        recoverable across a wall-clock kill and a resubmit, which is the normal path for a 90-task
        arm. It is also what makes the count a record of execution rather than of registration.

        THE MARKER HOLDS THIS EPISODE'S FIRING COUNT, NOT JUST ITS EXISTENCE
        ------------------------------------------------------------------------------------------
        `fired_tasks` was recoverable from these markers but `executed` was not: the marker was
        written once, empty, on the first fire, so a scorer could tell THAT an episode fired but not
        HOW MANY TIMES. Two consequences, both real:
          - A resumed arm under-reported `interventions_executed`, counting only firings that
            happened after the resubmit, while `gains`/`losses` correctly covered all 90 episodes.
          - Scoring an arm across several worker processes was impossible to tally at all, since no
            parent process sees any child's in-memory counter.
        So the count is written every time, and an episode's total survives both a kill and a fork.
        A pre-existing empty marker still reads as 1 (see `_fired_counts_from_markers`), which is the
        most that file was ever able to claim.
        """
        self.fired_tasks.add(task_id)
        self.executed += 1
        # Lazily, because probes.py reaches the executors on a BARE instance (`__new__`, no
        # `__init__`) by design -- so this method may not assume any field `__init__` would have set.
        counts = getattr(self, "_fired_counts", None)
        if counts is None:
            counts = self._fired_counts = {}
        counts[task_id] = counts.get(task_id, 0) + 1
        path = self._fired_marker_path(task_id)
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{counts[task_id]}\n")
        except OSError:
            # Telemetry must never take down a live episode; the in-memory set still covers this run.
            pass

    def _consume_budget(self, task_id: str) -> bool:
        """True if under budget (act now, and the caller must then call `_record_fired`); False if
        exhausted -- the caller degrades to control, exactly as `run_boundary_contract`'s per-episode
        budget test requires, and leaves the just-generated cell to dispatch unmodified."""
        assert self.controller is not None
        budget = int((self.controller.eta or {}).get("retry_budget", 1) or 1)
        spent = self.budget_spent.get(task_id, 0)
        if spent >= budget:
            return False
        self.budget_spent[task_id] = spent + 1
        return True

    # ------------------------------------------------------------------ one model call, factored out
    # ------------------------------------------------------------------ so SUPPRESS/REPROMPT can repeat it
    def _generate_and_extract(self) -> tuple[str, Usage]:
        """Verbatim body of the base class's generate-and-extract half, factored out so it can run
        twice in one step (the first, discarded attempt and a SUPPRESS/REPROMPT-triggered retry)
        without duplicating the error handling, message bookkeeping, or logging."""
        output = self.language_model.generate(messages=self.trimmed_messages, cache_control_at=-1)
        error_message = output.pop("error", None)
        if error_message:
            raise _GenerationFailed(error_message)
        standardized_usage = output.pop("standardized_usage")
        raw_message = copy.deepcopy(output)
        code, fixed_output_content = self.extract_code_and_fix_content(output["content"] or "")
        output["content"] = fixed_output_content + "\n\n"
        self.messages.append(output)  # NOTE: appended as-is, so the model's own thinking persists.
        reasoning_content = (output.get("reasoning_content") or "").strip()
        self.logger.show_message(
            role="agent",
            content=fixed_output_content,
            reasoning_content=reasoning_content,
            raw_message=raw_message,
            step_number=self.step_number,
        )
        return code, standardized_usage

    # ------------------------------------------------------------------ the six mechanisms, named to
    # ------------------------------------------------------------------ match adapter.py's `_CAPS` bindings
    def _inject_pre_generation_instruction(self, instruction: str) -> None:
        """PRE_GENERATION/REPROMPT. Appended before `generate()` is called for this step."""
        text = str(instruction or "").strip()
        if text:
            self.messages.append({"role": "user", "content": f"Note:\n{text}\n\n"})

    def _inject_post_execution_instruction(self, instruction: str) -> None:
        """POST_EXECUTION/REPROMPT. Same site as pre-generation injection; triggered by the error
        just observed in the previous cell's output rather than by accumulated step history."""
        text = str(instruction or "").strip()
        if text:
            self.messages.append({"role": "user", "content": f"Note:\n{text}\n\n"})

    def _withhold_proposed_call(self) -> tuple[str, Usage]:
        """POST_GENERATION_PRE_EXEC/SUPPRESS. The just-generated cell is discarded before it runs;
        the agent is told only that nothing was accepted, never why."""
        self.messages.pop()
        self.messages.append(
            {
                "role": "user",
                "content": "Your last cell was withheld before it ran. Provide a different cell.\n\n",
            }
        )
        return self._generate_and_extract()

    def _inject_pre_exec_instruction(self, instruction: str) -> tuple[str, Usage]:
        """POST_GENERATION_PRE_EXEC/REPROMPT. Unlike suppress, this carries the arm's own guidance
        text -- that is the entire difference between the two mechanisms."""
        self.messages.pop()
        text = str(instruction or "").strip()
        note = "Your last cell was withheld before it ran.\n" + (f"{text}\n\n" if text else "\n")
        self.messages.append({"role": "user", "content": note})
        return self._generate_and_extract()

    def _reroute_to_api_docs_pre_exec(self, target_app: str, target_api: str) -> str:
        """POST_GENERATION_PRE_EXEC/REROUTE. Replaces the just-generated cell with a docs lookup,
        computed from that cell itself -- `eta_is_computed`, no arm-supplied eta."""
        code = _api_docs_lookup_code(target_app, target_api)
        self.messages.pop()
        self.messages.append({"role": "assistant", "content": f"```python\n{code}\n```\n\n"})
        return code

    def _reroute_after_execution(self) -> str:
        """POST_EXECUTION/REROUTE. Fires before generation even starts for this step, using the
        target the PREVIOUS cell (whose error is now visible) was aimed at -- POST_EXECUTION declares
        no app/api field of its own, so this reads state cached at the site that cell was built,
        the same computed-eta contract REROUTE always has."""
        return _api_docs_lookup_code(self._last_target_app, self._last_target_api)

    # ------------------------------------------------------------------ the wholesale override
    def next_execution_inputs_usage_and_status(
        self, last_execution_outputs: Sequence[ExecutionIO]
    ) -> tuple[Sequence[ExecutionIO], Usage, Status]:
        task_id = self.world.task_id

        if last_execution_outputs:
            assert len(last_execution_outputs) == 1, "React expects exactly one last_execution_output."
            output_content = last_execution_outputs[0].content
            self.logger.show_message(
                role="environment", content=output_content, step_number=self.step_number
            )
            maybe_new_line = "\n" if not output_content.endswith("\n") else ""
            self.messages.append(
                {
                    "role": "user",
                    "content": "Output:\n```\n" + output_content + maybe_new_line + "```\n\n",
                }
            )

            pe_state = exec_fields_from_output(output_content)
            self._last_error_kind = pe_state["error_kind"]
            self._consecutive_errors = 0 if pe_state["succeeded"] else self._consecutive_errors + 1

            ctl = self.controller
            if ctl is not None and ctl.boundary == _PE and ctl.fires_on(pe_state):
                if ctl.action == "reprompt" and self._consume_budget(task_id):
                    self._record_fired(task_id)
                    self._inject_post_execution_instruction(str(ctl.eta.get("instruction") or ""))
                elif ctl.action == "reroute":
                    self._record_fired(task_id)
                    code = self._reroute_after_execution()
                    self.messages.append(
                        {"role": "assistant", "content": f"```python\n{code}\n```\n\n"}
                    )
                    return [ExecutionIO(content=code)], Usage(), Status(failed=False)

        ctl = self.controller
        if ctl is not None and ctl.boundary == _PRE and ctl.fires_on(self._pre_state()):
            if ctl.action == "reprompt" and self._consume_budget(task_id):
                self._record_fired(task_id)
                self._inject_pre_generation_instruction(str(ctl.eta.get("instruction") or ""))

        try:
            code, usage = self._generate_and_extract()
        except _GenerationFailed as exc:
            return [], Usage(), Status(failed=True, message=exc.message)

        # Cached unconditionally -- `_reroute_after_execution` needs THIS step's target on a LATER
        # step whose active controller is at POST_EXECUTION, not POST_GENERATION_PRE_EXEC, so the
        # cache update must not be gated behind a PG-boundary controller being the one installed.
        pg_state = self._pg_state(code)

        ctl = self.controller
        if ctl is not None and ctl.boundary == _PG and ctl.fires_on(pg_state):
            action = ctl.action
            if action == "suppress" and self._consume_budget(task_id):
                self._record_fired(task_id)
                code, extra_usage = self._withhold_proposed_call()
                usage = usage + extra_usage
            elif action == "reprompt" and self._consume_budget(task_id):
                self._record_fired(task_id)
                code, extra_usage = self._inject_pre_exec_instruction(
                    str(ctl.eta.get("instruction") or "")
                )
                usage = usage + extra_usage
            elif action == "reroute":
                self._record_fired(task_id)
                code = self._reroute_to_api_docs_pre_exec(
                    pg_state["target_app"], pg_state["target_api"]
                )

        return [ExecutionIO(content=code)], usage, Status(failed=False)
