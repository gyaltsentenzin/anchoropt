# Stock-pipeline check (qwen3.6 · retail · test)

Does AnchorOpt's runner score the **unmodified** tau-bench pipeline the way tau-bench's own runner does, as
GEPA calls it? Motivated by GEPA's stock measurements averaging 3.5 points above AnchorOpt's across the
comparison matrix. The result is reported in
[`docs/TAU2_FRAMEWORK_COMPARISON.md`](../../../../docs/TAU2_FRAMEWORK_COMPARISON.md#is-stock-the-same-pipeline-in-every-framework).

| side | what runs |
|---|---|
| `ours` | `run_anchoropt_round.py validate --p0-only` on a scratch copy of the arm's `baseline.json`: shared gateway, 180 s call timeout, 1 retry |
| `tau2runner` | GEPA's `experiments/tau2/eval_prompt.py --stock` (tau-bench's `run_tasks`), seeds 301–304, a fresh artifacts directory so nothing is served from its cache |

Both sides run qwen3.6 · retail test (40 tasks) at concurrency 3, at the same time, on the same endpoint.

```bash
# one job per chain of two runs, however you submit work on your own cluster/scheduler;
# $DIR holds a copy of the arm's baseline.json for SIDE=ours
SIDE=ours DIR=$S/ours_a bash run_check.sh
SIDE=tau2runner DIR=$S/tau2runner SEEDS="301 302" bash run_check.sh
STOCKCHECK_DIR=$S python analyze.py
```
