"""Running tau-bench episodes, mining their trajectories, and comparing a run against ONE incumbent.

This is the host-side harness: the part that already existed as a benchmark runner and that AnchorOpt
reuses rather than replaces. It imports tau2, so the contract and observability tests do not import it.

THREE JOBS
  tool_catalog()        real tool facts, so REROUTE grounding is schema-driven rather than guessed
  trajectory_events()   a recorded SimulationRun -> the boundary-tagged events core localizes over
  run_corpus()          run every task under one policy and report per-case outcomes + firings

WHY `trajectory_events` USES THE SAME BUILDERS AS THE LIVE MECHANISM. If mining had its own state
construction it would drift from what the controller actually sees, and a predicate that discriminates
beautifully on mined states would never fire in the run. Both go through `tau2_state`.

WHAT A RECORDED TRAJECTORY DOES NOT CONTAIN: withheld attempts. That is by design -- they never reached
the orchestrator -- so mining an incumbent run yields exactly the decisions the agent actually faced,
which is the residual that incumbent leaves.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tau2.data_model.message import (
    AssistantMessage, MultiToolMessage, ToolMessage, UserMessage,
)
from tau2.environment.toolkit import MUTATES_STATE_ATTR
from tau2.registry import registry

import tau2_state as _st
from tau2_mechanism import Controller, FiringSink, make_agent_factory

# A task counts as SOLVED when it earns full reward. tau-bench's task reward is the PRODUCT of its
# reward_basis components, so any zero component zeroes the task -- there is no partial credit to
# recover. The raw reward is carried alongside so a reader can check this reduction.
SOLVED_THRESHOLD = 1.0

# How much void a run may carry and still be usable. A shared inference service drops the occasional
# connection -- litellm reports it as `AuthenticationError: ... All connection attempts failed` -- and an
# all-or-nothing rule discards 24 good episodes over one blip, which is how three arms were thrown away.
# The distinction that matters is not "any void" but "did enough run to answer the question": the scored
# cases become the round's case set, and every arm is measured on exactly that set.
VOID_TOLERANCE = float(os.environ.get("ANCHOROPT_TAU2_VOID_TOLERANCE", "0.2"))


# ------------------------------------------------------------------------------------------------
# 1. real tool facts
# ------------------------------------------------------------------------------------------------
def make_env(domain: str, env_kwargs: Mapping[str, Any] | None = None):
    """One environment. `env_kwargs` carries per-domain configuration.

    banking_knowledge needs it: its default `alltools` retrieval variant embeds 698 documents with
    `text-embedding-3-large`, and this LiteLLM gateway's allowlist contains NO embedding model, so
    construction fails with 403. `retrieval_variant="bm25_grep"` needs no embedding service.
    """
    return registry.get_env_constructor(domain)(**dict(env_kwargs or {}))


def tool_catalog(env) -> dict[str, dict[str, Any]]:
    """{tool: {mutates_state, required, params}} read off the live environment."""
    cat: dict[str, dict[str, Any]] = {}
    for tool in env.get_tools():
        func = getattr(tool, "_func", None)
        params = getattr(tool, "params", None)
        fields = getattr(params, "model_fields", {}) or {}
        cat[tool.name] = {
            "mutates_state": bool(getattr(func, MUTATES_STATE_ATTR, True)),
            "required": sorted(n for n, f in fields.items()
                               if getattr(f, "is_required", lambda: False)()),
            "params": sorted(fields),
        }
    return cat


def mutating_tool_names(env) -> frozenset[str]:
    return frozenset(n for n, v in tool_catalog(env).items() if v["mutates_state"])


# ------------------------------------------------------------------------------------------------
# 2. mining a recorded trajectory
# ------------------------------------------------------------------------------------------------
def trajectory_events(simulation, *, case_id: str = "", known_tools: frozenset[str] = frozenset(),
                      mutating_tools: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """One SimulationRun -> ordered, boundary-tagged events.

    Three per agent turn at most: the turn start (PRE_GENERATION), what it proposed or committed to
    (POST_GENERATION_PRE_EXEC), and each result that came back (POST_EXECUTION).

    THE ANSWER-COMMITMENT ROW IS KEPT. A turn that replies instead of acting emits a gate event with
    `commits_to_reply=True`. Dropping those deletes a boundary the backward search needs and
    localization then derives exactly one.
    """
    cid = str(case_id or getattr(simulation, "task_id", "") or "")
    events: list[dict[str, Any]] = []
    turn = 0
    assistant_turns = user_turns = tool_errors = consecutive = 0
    pending_inbound_is_tool_result = False
    call_names: dict[str, str] = {}

    for msg in (getattr(simulation, "messages", None) or ()):
        if isinstance(msg, UserMessage) and not getattr(msg, "tool_calls", None):
            user_turns += 1
            pending_inbound_is_tool_result = False
            continue

        if isinstance(msg, AssistantMessage):
            turn += 1
            events.append(dict(_st.turn_start_state(
                case_id=cid, turn_index=turn, assistant_turns_so_far=assistant_turns,
                user_turns_so_far=user_turns,
                inbound_is_tool_result=pending_inbound_is_tool_result,
                consecutive_tool_errors=consecutive, tool_errors_so_far=tool_errors),
                kind="turn_start"))
            calls = [{"name": tc.name, "arguments": dict(tc.arguments or {})}
                     for tc in (msg.tool_calls or ())]
            events.append(dict(_st.gate_state(
                case_id=cid, turn_index=turn, proposed_calls=calls,
                has_content=bool((msg.content or "").strip()), known_tools=known_tools,
                mutating_tools=mutating_tools, replan_attempt=0,
                consecutive_tool_errors=consecutive, tool_errors_so_far=tool_errors),
                kind="propose"))
            for tc in (msg.tool_calls or ()):
                call_names[str(tc.id)] = tc.name
            assistant_turns += 1
            pending_inbound_is_tool_result = False
            continue

        subs = (list(msg.tool_messages or []) if isinstance(msg, MultiToolMessage)
                else [msg] if isinstance(msg, ToolMessage) else [])
        for sub in subs:
            if getattr(sub, "error", False):
                tool_errors += 1
                consecutive += 1
            else:
                consecutive = 0
            events.append(dict(_st.result_state(
                case_id=cid, turn_index=turn, is_error=bool(getattr(sub, "error", False)),
                content=getattr(sub, "content", None),
                tool_name=call_names.get(str(getattr(sub, "id", "")), ""),
                mutating_tools=mutating_tools, tool_errors_so_far=tool_errors),
                kind="result"))
        if subs:
            pending_inbound_is_tool_result = True

    return events


# ------------------------------------------------------------------------------------------------
# 3. running a corpus under one policy
# ------------------------------------------------------------------------------------------------
@dataclass
class RunResult:
    """One policy measured over one corpus. The shape a paired comparison needs."""

    reward: dict[str, float] = field(default_factory=dict)
    solved: dict[str, bool] = field(default_factory=dict)
    events: dict[str, list] = field(default_factory=dict)
    termination: dict[str, str] = field(default_factory=dict)
    firings: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    @property
    def n(self) -> int:
        return len(self.solved)

    @property
    def n_solved(self) -> int:
        return sum(1 for v in self.solved.values() if v)

    @property
    def mean_reward(self) -> float:
        return (sum(self.reward.values()) / len(self.reward)) if self.reward else 0.0

    @property
    def harness_errors(self) -> int:
        """Episodes that never produced a scored outcome.

        A harness error is an ABSENCE, not a zero. `run_corpus` records reward 0.0 for one so the
        dict shapes stay uniform, but that number must never reach a paired comparison: an arm whose
        episodes did not run would be scored as losing every case the incumbent solved, which is the
        fabricated-measurement defect wearing its worst face. A gateway outage on 2026-09-22 06:31-07:54
        produced 27 such arm runs and every one was written into results.json as a measured loss.
        """
        return sum(1 for v in self.termination.values() if str(v).startswith("harness_error"))

    @property
    def truncated(self) -> dict[str, int]:
        """Premature endings, by kind. Each scores 0 by construction, so a large share means the run
        measured a budget rather than the model."""
        out: dict[str, int] = {}
        for v in self.termination.values():
            t = str(v)
            if "MAX_STEPS" in t or "TIMEOUT" in t or "TOO_MANY_ERRORS" in t:
                out[t.split(".")[-1]] = out.get(t.split(".")[-1], 0) + 1
        return out

    @property
    def scored_cases(self) -> tuple[str, ...]:
        return tuple(c for c, v in self.termination.items()
                     if not str(v).startswith("harness_error"))

    @property
    def interventions_executed(self) -> int:
        return int(self.firings.get("interventions_executed", 0) or 0)

    @property
    def cases_fired(self) -> list[str]:
        return list(self.firings.get("cases_fired") or ())

    def failing(self) -> list[str]:
        return sorted(c for c, ok in self.solved.items() if not ok)

    def to_json(self) -> dict:
        return {"label": self.label, "n": self.n, "n_solved": self.n_solved,
                "mean_reward": round(self.mean_reward, 4), "reward": self.reward,
                "solved": self.solved, "termination": self.termination, "firings": self.firings,
                "harness_errors": self.harness_errors}


def split_ids(domain: str, split: str) -> tuple[str, ...]:
    """The task ids of one of tau-bench's OWN splits, or () when the domain defines none."""
    loader = registry.get_task_splits_loader(domain)
    if loader is None:
        return ()
    got = loader() or {}
    return tuple(str(t) for t in (got.get(split) or ()))


def load_tasks(domain: str, *, limit: int | None = None, task_ids: Sequence[str] = (),
               split: str = "") -> list:
    """The tasks one round runs on.

    SELECT BY SPLIT, NOT BY PREFIX. An earlier version sorted ids as STRINGS and took the first N,
    which on airline meant tasks 0-3 and 10-30 -- because "10" < "2" -- and silently mixed tau-bench's
    official train and test sets: 11 of the 20 airline test tasks were optimized against, burning 55%
    of the held-out set. A lexicographic prefix is deterministic but it is not a sample of anything,
    and it cannot be validated on afterwards.

    `split` names one of the benchmark's own splits ("train"/"test"/"base"). `limit` subsamples WITHIN
    that split and should normally be left unset: a smaller split is a defensible reduction, a prefix
    of all tasks is not.
    """
    tasks = list(registry.get_tasks_loader(domain)())
    if task_ids:
        want = {str(t) for t in task_ids}
        by_id = {str(t.id): t for t in tasks}
        # Preserve the CALLER's order, so a recorded case set replays in the order it was recorded.
        return [by_id[i] for i in (str(x) for x in task_ids) if i in by_id] or [
            t for t in tasks if str(t.id) in want]
    if split:
        ids = split_ids(domain, split)
        if not ids:
            raise SystemExit(
                f"FATAL: domain {domain!r} defines no task splits, so --split {split!r} cannot be "
                f"honoured. Name the cases explicitly instead of falling back to an arbitrary subset.")
        keep = {i: n for n, i in enumerate(ids)}
        tasks = sorted((t for t in tasks if str(t.id) in keep), key=lambda t: keep[str(t.id)])
    else:
        tasks.sort(key=lambda t: str(t.id))
    return tasks[:limit] if limit else tasks


def run_corpus(controller: Controller | None, *, domain: str, tasks: Sequence[Any],
               agent_llm: str, user_llm: str, max_steps: int = 40, max_errors: int = 10,
               max_concurrency: int = 4, label: str = "",
               agent_llm_args: Mapping[str, Any] | None = None,
               user_llm_args: Mapping[str, Any] | None = None,
               env_kwargs: Mapping[str, Any] | None = None,
               sim_timeout: float | None = 3600.0,
               on_episode: Callable[[str, float], None] | None = None) -> RunResult:
    """Run every task under ONE policy. `controller=None` is the unmodified pipeline.

    FIRINGS COME FROM THE RUN. One `FiringSink` is shared across the corpus and written by the
    mechanism at the site that altered each episode -- never derived from the fact that a controller
    was installed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.simulation import run_simulation
    from tau2.user.user_simulator import UserSimulator

    sink = FiringSink()
    factory = make_agent_factory(controller, sink)
    out = RunResult(label=label or ("incumbent" if controller is None else "arm"))

    def one(task):
        env = make_env(domain, env_kwargs)
        catalog = tool_catalog(env)
        known = frozenset(catalog)
        mutating = frozenset(n for n, v in catalog.items() if v["mutates_state"])
        agent = factory(tools=env.get_tools(), domain_policy=env.get_policy(), llm=agent_llm,
                        llm_args=dict(agent_llm_args or {"temperature": 0.0}), task=task)
        # `get_user_tools` RAISES on a domain with no user tools (airline is one; telecom is not), so
        # it is guarded exactly as tau-bench's own `build_user` guards it. An earlier version let the
        # exception through and all 8 airline episodes were recorded as `harness_error` -- visible only
        # because a harness failure is kept distinct from a measured zero.
        try:
            user_tools = env.get_user_tools(include=task.user_tools) or None
        except Exception:
            user_tools = None
        user = UserSimulator(llm=user_llm, instructions=str(task.user_scenario),
                             tools=user_tools,
                             llm_args=dict(user_llm_args or {"temperature": 0.0}))
        # A PER-SIMULATION WALL CLOCK. tau-bench defaults to `timeout=None`, and the LSF `normal`
        # queue has no RUNLIMIT, so at max_steps=200 a single pathological episode could hold an arm
        # for 200 x the per-call timeout -- over a day -- with nothing to stop it. A TIMEOUT
        # termination scores 0 like any premature end, so this is set generously (a telecom episode
        # averages ~840s) and `RunResult.truncated` surfaces it: a cap that fires often is measuring
        # the cap, which is exactly the mistake max_steps=30 made.
        orch = Orchestrator(domain=domain, agent=agent, user=user, environment=env, task=task,
                            max_steps=max_steps, max_errors=max_errors, timeout=sim_timeout)
        # env_kwargs reaches the EVALUATOR too: it builds its own environment for the DB check, and
        # a banking evaluation under the default variant would 403 exactly as construction does.
        sim = run_simulation(orch, env_kwargs=dict(env_kwargs or {}) or None)
        return task, sim, known, mutating

    with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as pool:
        futures = {pool.submit(one, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            cid = str(task.id)
            try:
                task, sim, known, mutating = fut.result()
            except Exception as exc:                      # an infrastructure failure, not a score
                out.reward[cid] = 0.0
                out.solved[cid] = False
                out.termination[cid] = f"harness_error: {type(exc).__name__}: {exc}"[:200]
                out.events[cid] = []
                continue
            reward = float(getattr(getattr(sim, "reward_info", None), "reward", 0.0) or 0.0)
            out.reward[cid] = reward
            out.solved[cid] = reward >= SOLVED_THRESHOLD
            out.termination[cid] = str(getattr(sim, "termination_reason", "") or "")
            out.events[cid] = trajectory_events(sim, case_id=cid, known_tools=known,
                                                mutating_tools=mutating)
            if on_episode is not None:
                on_episode(cid, reward)

    out.firings = sink.snapshot()
    return out


# ------------------------------------------------------------------------------------------------
# 4. the paired comparison
# ------------------------------------------------------------------------------------------------
def paired(arm: RunResult, incumbent: RunResult) -> dict[str, Any]:
    """Per-case set difference against ONE frozen incumbent. Returns CASE IDS, not counts.

    The denominator must be identical or the comparison answers a different question; the caller is
    told rather than having the mismatch averaged away.

    CASES THE ARM COULD NOT RUN ARE EXCLUDED, and `harness_errors` reports how many. Including them
    would count an infrastructure failure as the intervention's effect. The caller decides what to do
    with a partial comparison -- `complete` says whether one is even available.
    """
    usable = set(arm.scored_cases) if arm.harness_errors else set(arm.solved)
    shared = sorted(set(arm.solved) & set(incumbent.solved) & usable)
    gains = tuple(c for c in shared if arm.solved[c] and not incumbent.solved[c])
    losses = tuple(c for c in shared if not arm.solved[c] and incumbent.solved[c])
    return {
        "gains": gains, "losses": losses, "net": len(gains) - len(losses),
        "n": len(shared),
        "denominator_identical": set(arm.solved) == set(incumbent.solved),
        "arm_solved": sum(1 for c in shared if arm.solved[c]),
        "incumbent_solved": sum(1 for c in shared if incumbent.solved[c]),
        "accuracy_delta_pp": (100.0 * (sum(1 for c in shared if arm.solved[c])
                                       - sum(1 for c in shared if incumbent.solved[c]))
                              / max(1, len(shared))),
        "interventions_executed": arm.interventions_executed,
        "cases_fired": arm.cases_fired,
        "harness_errors": arm.harness_errors,
        # A comparison is COMPLETE only when the arm ran every case the incumbent did. Anything less is
        # a different question, and the caller must not average the gap away.
        "complete": arm.harness_errors == 0 and set(arm.solved) == set(incumbent.solved),
    }


def save_run(result: RunResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.to_json(), indent=2, default=str) + "\n")
