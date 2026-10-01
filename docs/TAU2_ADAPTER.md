# tau-bench (τ²) AnchorOpt adapter — runtime audit and capability declaration

What the tau-bench runtime actually executes, where a controller can intervene, and which of the four
action families it can genuinely realize. Every row below names the code that runs it; nothing is
declared because the benchmark "conceptually supports" it.

Upstream pin: `sierra-research/tau2-bench` 1.0.1 at `e6609f3`, plus one patch
([`benchmarks/tau2/patches/tau2_config_nl_judge.patch`](../benchmarks/tau2/patches/tau2_config_nl_judge.patch));
referenced via a working checkout at `$ANCHOROPT_TAU2_REPO`. The harness is **referenced, not
vendored**: the corpus is 714 MB and every domain is load-bearing, so there is no subset to trim to.

This document is the audit that came *before* the implementation. The code and how to run it:
[`benchmarks/tau2/README.md`](../benchmarks/tau2/README.md). The first results narrative report is not
in this repo (see [`docs/RESULTS_POLICY.md`](../docs/RESULTS_POLICY.md)); the paper's reported
reliability numbers are in [`PAPER_RESULTS.md`](PAPER_RESULTS.md#tau2-bench--repeated-execution-reliability).
§5 records what running it changed.

> Facts cited below from `benchmarks/tau2/{VENDORED,RESULTS}.md` and `benchmarks/tau2/gates/` come from
> the **hand-authored-gates** work on branch `add-tau2-benchmark`, which is not an ancestor of this
> branch. They are quoted as prior evidence about the same runtime, not as files in this tree. That
> work intervenes with rules a person wrote; this adapter lets core *learn* where and how to intervene.

---

## 1. The agent execution loop

`Orchestrator.step()` — `src/tau2/orchestrator/orchestrator.py:823`. A half-duplex role machine:

```
USER/ENV -> AGENT :  agent_msg, agent_state = agent.generate_next_message(message, agent_state)
                     trajectory.append(agent_msg)            <-- ONLY the returned message is recorded
                     to_role = ENV if agent_msg.is_tool_call() else USER
AGENT    -> ENV   :  tool_results = _execute_tool_calls(message.tool_calls)
                     -> environment.get_response(tool_call)  <-- the world moves HERE
                     trajectory.extend(tool_results)
```

**The consequential decision is made entirely inside `agent.generate_next_message`.** The orchestrator
does not choose; it dispatches whatever the agent returned. So the agent subclass is the only
intervention seam, and `registry.register_agent_factory(...)` (`src/tau2/registry.py:106`) is how one is
installed. `build_agent` (`src/tau2/runner/build.py:100`) passes `task=task`, so **the mechanism can key
firings by `task.id`** — that is what makes per-case firing attribution possible.

### The two lists, and why the distinction is load-bearing

`self.trajectory` (orchestrator) and `agent_state.messages` (agent) are **different lists**. Only the
*returned* message enters the trajectory. Therefore:

* messages the mechanism appends to the agent's own context are **invisible** to the trajectory, the
  evaluator, and the replay;
* the `ToolMessage` objects the orchestrator hands the agent are **the same objects** it already put in
  `self.trajectory` — mutating one in place corrupts the record.

### The replay constraint that rules out two otherwise-plausible interventions

`Environment.set_state` (`src/tau2/environment/environment.py:293-412`) is called by the **evaluator**
(`evaluator_env.py:89,296`), not only at init. It pairs every `(ToolCall, ToolMessage)` in the recorded
trajectory and, for **mutating** tools (`_is_mutating_tool`, default `True`), re-executes the call
against a fresh environment and compares content — raising `ValueError` on mismatch when `strict=True`
(the live-evaluation default).

Two consequences, both binding:

1. **A fabricated tool result for a mutating call cannot enter the trajectory.** It would be replayed
   for real at evaluation time *and* fail the content comparison. So the only sound way to stop an
   action is to never emit it.
2. **Rewriting a recorded `ToolMessage` in place corrupts the trajectory.** This is not hypothetical:
   the earlier hand-authored gate work did exactly that and turned **41 of 114** telecom simulations
   into infrastructure errors (`benchmarks/tau2/gates/gated_agent.py` on `add-tau2-benchmark`,
   `_redirect_guidance` docstring).

---

## 2. Boundary / capability table

Adapter keys are this adapter's own; `boundary_from_key` owns the mapping to core loci.

| adapter key | locus | observable there | can prevent execution |
|---|---|---|---|
| `before_agent_turn` | `PRE_GENERATION` | turn index, assistant/user turn counts, whether inbound is a tool result, carried error history. **No proposed call exists** | everything downstream |
| `before_tool_dispatch` | `POST_GENERATION_PRE_EXEC` | the proposed call: tool name, arg count/size, `mutates_state`, whether the tool exists; or that the turn **commits to a reply** instead | the execution |
| `after_tool_result` | `POST_EXECUTION` | `error` flag, classified `error_kind`, result size, which tool returned | nothing — repair only |

### Executable actions — U_H(ℓ), and the code behind each

| boundary | action | mechanism (the code that runs) | status |
|---|---|---|---|
| `PRE_GENERATION` | `REPROMPT` | `inject_pre_generation_note` — extra message in the list passed to `generate()`, agent context only | **enabled** |
| `POST_GENERATION_PRE_EXEC` | `REPROMPT` | `inject_replan_instruction` — instruction added, candidate regenerated | **enabled** |
| `POST_GENERATION_PRE_EXEC` | `SUPPRESS` | `withhold_proposed_call` — the call is **never returned** to the orchestrator; agent re-plans from its absence, no instruction supplied | **enabled** (variant `cancel_proposed` only) |
| `POST_GENERATION_PRE_EXEC` | `REROUTE` | `rewrite_proposed_call` — the proposed call is replaced by a **schema-validated** destination before it is returned; the rewritten call is what executes and what is recorded, so replay stays consistent | **enabled** (substitute) |
| `POST_EXECUTION` | `REPROMPT` | `append_post_result_guidance` — guidance appended as a **separate** message, never overwriting the recorded `ToolMessage` | **enabled** |
| `POST_EXECUTION` | `REROUTE` | rewriting the returned observation | **DISABLED** — `disabled_reason`: in-place rewrite corrupts the recorded trajectory (41/114 telecom infra errors) and the agent cannot dispatch tools itself, so there is no sound post-execution re-dispatch path |
| `POST_EXECUTION` | `SUPPRESS` | — | structurally excluded by core (`anchor.exclusion_reason`); nothing left to cancel |

### Variants deliberately **not** grounded

`SUPPRESS / withhold_and_replay` (agent sees the tool's verbatim recorded result) requires a recorded-
result store the tau-bench runtime does not have, and its `preservation` clause would have no executor.
It is not grounded rather than grounded-and-inert. `ground_suppress` therefore emits `cancel_proposed`
only, and carries **no `preservation` key** — the remove-outright variant preserves nothing and must not
claim to.

---

## 3. Measurement

| question | answer |
|---|---|
| trajectories | `SimulationRun.messages` (`src/tau2/data_model/simulation.py:1247`) |
| correctness | `evaluate_simulation` → `RewardInfo.reward`; the task reward is the **product** over `reward_basis`, so any zero component zeroes the task. Premature termination ⇒ reward 0 |
| one episode, no registry | `run_simulation(orchestrator)` (`src/tau2/runner/simulation.py:19`) |
| firings | counted **inside the mechanism**, at each site that actually altered the run, into a thread-safe sink keyed by `task.id`. `ThreadPoolExecutor` (`runner/batch.py:810`) keeps runs in one process, so the sink is readable after the run |
| paired comparison | per-`task_id` reward against one frozen incumbent run (`select`); against the incumbent's **distribution** over replicated runs (`rescore`); and mean-vs-mean on the held-out split (`validate`) |
| task selection | tau-bench's own `train`/`test` splits via `registry.get_task_splits_loader` |

### The variance floor is not zero — and it bounds what a round may claim

Measured in this integration at `max_steps=200`: across five runs of the identical policy on the 30
airline train tasks, **17 of 30** cases flip outcome for qwen3.6 and **10 of 30** for granite, and qwen3.6's
totals ranged 16–23. (Earlier gate work on `add-tau2-benchmark` saw 8 of 50 airline tasks flip between
two baselines.) BFCL's "a single lost case is real" discipline does **not** transfer: a train net
measured against one incumbent run means little until it is checked against replicates and on held-out
data.

---

## 4. Blockers and constraints found

1. **Two interpreters.** tau-bench needs its own venv (py3.12 + `loguru`, `litellm`, …); the anchoropt
   venv has neither. Both import cleanly in one process via
   `PYTHONPATH=<path-to-anchoropt> <path-to-tau2-bench>/.venv/bin/python`. The adapter modules
   therefore **do not import tau2** — only the mechanism does — so the contract and observability tests
   run offline in the anchoropt venv.
2. **A real round needs a served model** for the agent plus a user-simulator/NL-judge endpoint. Episode
   execution is not offline; localization, expansion, grounding and selection are.
3. **`POST_EXECUTION` re-dispatch is genuinely unavailable** (item above), not merely unimplemented.

---

## 5. What running it changed

Each of these was found by running the adapter, not by reading tau-bench, and each is now fixed or
pinned by a test.

| found | consequence | now |
|---|---|---|
| a harness error was recorded as reward 0 | an endpoint outage read as the controller making the agent worse | void episodes are excluded; a >20% void baseline is refused |
| `max_steps=30` | 25/25 telecom episodes truncated and scored 0 by construction | 200, tau-bench's own default |
| tasks taken as a string-sorted prefix | tau-bench's train and test splits mixed; 11 of 20 airline test tasks optimized against | selection by the official split |
| `banking_knowledge` | not part of the original tau-2; no split; default retrieval needs an embedding model the gateway lacks | excluded |
| the reroute and reprompt cells under-declared the eta they read | core correctly refused every arm there (`executor_cannot_consume_eta`) | `retry_budget` / `retry_semantics` genuinely read and declared |
| a winner that executed 0 interventions | a variance flip promoted as a train win | flagged by `select`, refused by `validate` |
| one incumbent run as the reference | every measured arm looked positive against a low draw | `replicate`/`rescore`, and `validate` with replicated sides |
| local vLLM needs `--tool-call-parser qwen3_xml` / `granite4` | a wrong parser fails silently: no tool calls, exit 0, near-zero scores | preflight refuses an endpoint that parses no tool call |
