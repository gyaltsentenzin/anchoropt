# AppWorld adapter

The reference implementation among AnchorOpt's five benchmark adapters — the only one with
headroom screening, behavioural probes, and a held-out validation driver. AppWorld is a multi-app
agentic environment: an agent calls real app APIs (email, calendar, files, …) across multi-step
tasks graded on whether the world ends up in the right state.

## What's here

| path | what it is |
|---|---|
| `adapter.py` | `U_H(ℓ)`, the capability bindings, the per-`(boundary, action)` instruction templates |
| `agent_hooks.py` | the three incision points woven into AppWorld's own agent loop |
| `residual.py` | mines the frozen incumbent's residual from on-disk transcripts — no model calls |
| `run_round.py` | the round driver: `propose` → score → `propose` again → promote |
| `evaluate.py` | identifies the frozen incumbent for a model/split pair |
| `controller.py` | this adapter's own `(incumbent_id, controllers.json)` serialization |
| `headroom.py` | screens whether a residual has any headroom before spending inference on it |
| `heldout_validation.py` | the held-out driver — scores a promoted arm against `sh_heldout` |
| `probes.py` | behavioural proof that each declared capability actually alters a run, not just telemetry |
| `metrics.py` | recomputes TGC/SGC baselines and per-arm deltas from on-disk results |
| `env.sh` | source this first — sets every path this adapter needs from env vars, no personal defaults |

## Setup and running a round

Full walkthrough, including the splits used, the baseline numbers, and the exact commands for each
step: [`../../docs/APPWORLD_SETUP.md`](../../docs/APPWORLD_SETUP.md). The short version:

```bash
source benchmarks/appworld/env.sh && appworld_env_check
python benchmarks/appworld/residual.py --evaluation <path>       # mine (free)
python benchmarks/appworld/run_round.py propose --evaluation <path> --out runs/x   # propose (free)
# score each arm in runs/x/controllers.json against your model (cost-bearing)
python benchmarks/appworld/metrics.py --results-dir runs --out runs/metrics.json   # record
```

Only scoring an arm costs inference; mining, proposing, and recording are offline.

## Verification gates, before spending anything

```bash
python -m pytest tests/test_toy_host_e2e.py tests/test_executor_behavioral_contract.py -q
python -c "from anchoropt.testing import check_adapter_contract; from benchmarks.appworld.adapter import ADAPTER; print(check_adapter_contract(ADAPTER))"
cd benchmarks/appworld && python -m pytest probes.py -q   # 9 passed: proof each capability alters the run
```

## Gotchas

Ordered by how much they cost you if you hit them.

1. **`APPWORLD_ROOT` is not optional.** `appworld.common.path_store` defaults its root to
   `os.getcwd()`. Omit it and you both mine the wrong tree and write a full AppWorld run tree
   (~100k log lines per 90-task split) wherever you happen to be.
2. **The `appworld` CLI ignores `APPWORLD_ROOT`.** `appworld run` resolves its root from its own
   `--root` option, defaulting to the CWD, and never consults the environment variable. This
   adapter's own scripts (`residual.py`, `run_round.py`, `metrics.py`) do honour it — this trap is
   specific to the upstream CLI.
3. **Mining and scoring use different roots.** A round cell holds the incumbent's evaluation and
   transcripts but ships no `experiments/configs`, so it cannot be the root for scoring. Read the
   evaluation from the cell, run the arm under your AppWorld checkout, and pass `--experiment`
   explicitly — it cannot be inferred from the evaluation's path alone.
4. **Reported net can include tasks the controller never touched.** `gains`/`losses` are computed
   over every scored task, not only those where the mechanism fired. A contaminated net can produce
   a false `NO_BENEFIT`. Verify firing per episode (recoverable from `logs/lm_calls.jsonl`'s final
   message list) before trusting a null result.
5. **Shared-endpoint contention is severe.** Two jobs against the same model endpoint can turn an
   episode from ~37s into ~17min with rate-limit errors. Stage concurrent work against the same
   endpoint, or budget for it.
6. **A killed job loses less than it looks.** Episodes persist: a completed episode writes a
   `misc/finished` marker, so a resubmit skips them and re-runs only the partial one. Arm results
   are written after every arm and merged into any existing file. Prefer resubmitting to restarting.
7. **Serving a thinking model locally needs its reasoning parser set.** Without it, `<think>` text
   can land in `content` instead of `reasoning_content` and corrupt the agent's code-block
   extraction.
8. **Put server logs on storage every host can read**, not a compute node's local `/tmp` — otherwise
   a startup stall is indistinguishable from a missing log.
