# Executor/eta contract notes

Three findings from an audit of the BFCL v4 adapter's executor bindings against what the upstream
hook actually does, kept because `benchmarks/bfcl_v4/bfcl_runtime.py` cites them directly.

## `Action.SUPPRESS` semantics — read `anchoropt/anchor.py:68-82` before touching this

Verbatim:

> **SUPPRESS** cancel the proposed call — it does not execute. What the model then OBSERVES is a
> second, independent choice, and both variants are in the accepted set:
> **A3** removes the call outright; the model sees nothing in its place.
> **E1** withholds the call but REPLAYS the tool's verbatim recorded result, so the observation is
> byte-identical and only the execution is saved.
> Both are suppression — the proposed action does not run.

**The upstream hook implements A3.** It removes the call from `_mg_exec_calls`, puts nothing in its
place, and the model proposes again next step.

**Therefore: do NOT add a `WITHHOLD` action.** It would duplicate an existing family.

### Executor/eta contract problems

| id | problem | where |
|---|---|---|
| **D1** | **Ghost executor.** `executor_supports(post_generation_pre_exec, suppress, …)` returns **True** via the `signal_agnostic` escape hatch, naming `enable_redundant_write_suppress` — which appears **0 times** in the upstream hook's code path. The check is a declaration lookup, not a binding to executing code. | core: `anchor_policy_opt.py:257`; adapter: `bfcl_runtime.EXECUTORS` |
| **D2** | **Inert eta.** The built arm declares `suppressed_operation` + `preservation`; nothing reads either. A3 needs no eta, so the contract demands fields the executor cannot consume. `preservation: "verify an equivalent copy survives"` is instruction text nothing enforces. | core contract + `ground_suppress` |
| **D3** | **Inert action.** `REPROMPT` upstream writes the instruction to `step_record` and **never injects it**. Proven: two arms differing only in `instruction` produced **byte-identical** trajectories (13/13). | adapter (evaluator patch) |
