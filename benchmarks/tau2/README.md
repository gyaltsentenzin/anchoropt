# tau-bench (τ²) AnchorOpt adapter

The adapter that lets AnchorOpt **learn** where and how to intervene in tau-bench. Core chooses the locus,
the condition and the action; the adapter only says what the tau-bench runtime can observe and execute.

* The runtime audit — execution loop, replay constraint, boundary/capability table:
  [`docs/TAU2_ADAPTER.md`](../../docs/TAU2_ADAPTER.md). Read that first.
* The first results — self-teach, round 1: the narrative report with every intermediate number is not
  in this repo (see [`docs/RESULTS_POLICY.md`](../../docs/RESULTS_POLICY.md)), but the **measured
  artifacts it was built from are** — [`rounds/SELFTEACH_R1/`](rounds/SELFTEACH_R1/) ships every arm
  directory, and `scripts/export_results.py --from-dest --markdown` recomputes the report's own table
  from them with no model and no GPU.
* Against GEPA and Self-Harness on the same matrix, pass^1 and pass^k: the paper's reported reliability
  table is in [`docs/PAPER_RESULTS.md`](../../docs/PAPER_RESULTS.md#tau2-bench--repeated-execution-reliability).
  The comparison script and per-trial data behind the full framework-comparison document are not in
  this repo (see [`docs/RESULTS_POLICY.md`](../../docs/RESULTS_POLICY.md)).

> Not to be confused with the **hand-authored gates** on branch `add-tau2-benchmark`. Those are rules a
> person wrote after reading failing trajectories; nothing here is hand-authored.

## Layout

| path | what it is |
|---|---|
| **the adapter** (no tau2 import — the contract and observability tests run offline) | |
| `tau2_runtime.py` | `ADAPTER`: every contract hook, `HOST` (U_H(ℓ)), grounding, and the injectable teacher eta |
| `tau2_fields.py` | the typed alphabet, declared per locus and boundary-truthful |
| `tau2_signals.py` | Φ_tau2: the shipped conditions and where each is observable |
| `tau2_state.py` | observable-state builders, **shared** by the live mechanism and trajectory mining |
| `tau2_capabilities.py` | `ExecutorCapability` per (boundary, action): the binding and the eta it reads |
| `tau2_attribution.py` | residual mining: failing episodes → diagnoses → ranked problems |
| `tau2_models.py` | addressing the agent (a shared gateway or local vLLM), the user simulator and the teacher |
| `tau2_teacher.py` | the teacher: an LLM operator that proposes REPROMPT instruction eta, mechanically validated |
| **the runtime side** (imports tau2) | |
| `tau2_mechanism.py` | the executing code: the `LLMAgent` subclass every capability binding names, and the firing sink |
| `tau2_episodes.py` | running episodes, task selection by split, mining trajectories, the paired comparison |
| **driving it** | |
| `run_anchoropt_round.py` | the phases: `baseline` `propose` `evaluate` `select` `replicate` `rescore` `validate` `refill` |
| `scripts/run_arm.sh` | one arm (model, teacher, domain) end to end; what your job scheduler should invoke per arm |
| `scripts/stage_rerun.sh` | quarantine void data and stage arms for a re-run |
| `scripts/compare_portal.py` | shared gateway vs local vLLM: per-call latency and the tool-call-parser gate |
| `scripts/export_results.py` | copy a round's artifacts into the repo and derive every reported number |
| `scripts/compare_frameworks.py` | AnchorOpt vs GEPA vs Self-Harness: every table of the comparison, from per-task trial outcomes |
| **data** | |
| `catalogs/{airline,retail,telecom}.json` | real tool schemas, so REROUTE grounding is schema-driven |
| `patches/tau2_config_nl_judge.patch` | the one change to tau-bench: a configurable NL-assertion judge |
| `rounds/SELFTEACH_R1/` | the committed artifacts behind the round-1 report |
| `rounds/TAU2_R1/` | an early smoke round (haiku, 20-task prefix); superseded, kept for history |

## Environment

The harness is **referenced, not vendored** (714 MB corpus). tau-bench 1.0.1 at upstream `e6609f3`, plus
`patches/tau2_config_nl_judge.patch`. Both packages run in one process under tau-bench's interpreter:

```bash
export ANCHOROPT_TAU2_REPO=/path/to/tau2-bench
set -a && . "$ANCHOROPT_TAU2_REPO/.env" && set +a     # HOSTED_API_KEY, OPENAI_BASE_URL, OPENAI_API_KEY
export TAU2_LLM_NL_ASSERTIONS=openai/claude-sonnet-5  # the upstream gpt-4.1 judge is not on this gateway
PY="env PYTHONPATH=$PWD $ANCHOROPT_TAU2_REPO/.venv/bin/python"
```

| setting | where | notes |
|---|---|---|
| agent model | `--agent-llm` | `granite-4.1-30b`, `minimax-m2.5`, `qwen3.6-35b-a3b` resolve to the shared gateway (`HOSTED_MODELS` in `tau2_models.py`); its auth is a `HOSTED_API_KEY` **header**, not Bearer |
| portal | `ANCHOROPT_TAU2_PORTAL=hosted\|vllm` | `vllm` reads `$ANCHOROPT_TAU2_SERVING_DIR/endpoint_<model>.txt` and **re-checks the port** (the file survives a dead server). Preflight refuses an endpoint that parses no tool call |
| user simulator | `--user-llm` | `claude-sonnet-5` on the gateway, constant across arms |
| split | `--split` | tau-bench's own `train`/`test`; `--limit 0` (default) uses the whole split. A domain with no split refuses |
| step budget | `--max-steps`, `--sim-timeout` | 200 (tau-bench's default) and 3600 s |

### Guards, and what each `run_arm.sh` exit means

| phase written to `PHASE` | exit | meaning |
|---|---|---|
| `FAILED_preflight` | 12 | an endpoint did not answer 3 probes (or, local portal, parsed no tool call). Nothing spent |
| `FAILED_baseline_void` | 11 | more than 20% of P0's episodes produced no outcome. Not a result; re-run |
| `FAILED_step_budget_mismatch` | 14 | a cached baseline was recorded with a different `MAX_STEPS`. Reset with `stage_rerun.sh --apply --all` |
| `OPEN_outage` | 13 | two consecutive arms ran nothing; genuine measurements kept, round left open |
| `DONE_no_arms` | 0 | P0 solved everything (no residual) — a legitimate end state |
| `DONE` | 0 | round complete |

Void episodes (harness failures) are excluded from scoring, never counted as 0. A void arm evaluation
is **omitted** from `results.json`. Void data is quarantined with `stage_rerun.sh`, not deleted.

### Known asymmetry, kept by decision

qwen3.6 runs with `enable_thinking: False` (inherited from an earlier adapter); minimax-m2.5 ignores
the switch and reasons regardless; granite emits no reasoning. All three produce valid tool calls
either way. The team decided to keep this configuration, so qwen-vs-minimax is not a controlled
comparison on this axis, and reports must say so. **Do not change `MODEL_EXTRA_BODY` while arms are
pending** — a running job imports it when it starts, so an edit mid-run splits the matrix across two
configurations.

## Tests

```bash
$PY -m pytest tests/test_tau2_*.py -q
```

| file | needs tau2 | what it pins |
|---|---|---|
| `test_tau2_adapter_contract.py` | no | the contract, the four closed action families, disabled-not-deleted cells, no failure→action mapping |
| `test_tau2_boundary_observability.py` | no | each boundary exposes only what the runtime carries there; `states_at` filters, never invents |
| `test_tau2_executor_behavioral.py` | yes | every enabled executor proven by **behaviour** against the real agent with a scripted model |
| `test_tau2_teacher.py` | no | the teacher's seven operator rules |
| `test_tau2_round_discipline.py` | yes | outcome classes kept apart; replicate/rescore; validate refusals |
| `test_tau2_void_measurements.py` | yes | a harness error is an absence, not a zero; split selection; submitter/runner budget agreement |
| `test_tau2_committed_results.py` | partly | committed artifacts reproduce the report and every anchor rebuilds |
| `test_tau2_framework_comparison.py` | no (needs GEPA's and Self-Harness's data) | the comparison document is the script's output; its pass^k equals the other frameworks' own hand-offs |

## One arm, end to end

```bash
D=benchmarks/tau2/run_anchoropt_round.py; OUT=/tmp/arm
$PY $D baseline  --domain airline --split train --agent-llm granite-4.1-30b \
                 --user-llm claude-sonnet-5 --max-steps 200 --concurrency 8 --out $OUT
$PY $D propose   --out $OUT --teacher self          # emits every arm; names no winner
$PY $D evaluate  --out $OUT --max-arms 8 --concurrency 8 --resume
$PY $D select    --out $OUT                          # core's argmax on what was measured
$PY $D replicate --out $OUT --k 4                    # P0 as a distribution, not one draw
$PY $D rescore   --out $OUT
$PY $D validate  --out $OUT --split test --k 4 --p0-only
$PY $D refill    --out $OUT --split test             # only if a held-out run has void episodes
```

`--max-arms N` truncates the evaluation budget **in manifest order**, never by firing rate. Arms left
unmeasured make the round `BUDGET_EXHAUSTED`: open, not a null.

**Cycle 2** (where Φ expansion happens): expansion is driven by *measured* exhaustion, so a first
`propose` legitimately does not expand. Feed cycle 1's measurements back with
`propose --results $OUT/results.json`, then `evaluate --results-name results_c2.json` and
`select --results-name results_c2.json`. Nothing is re-run and nothing is invented.

## Before trusting a number

* **One incumbent run is not a stable reference.** On qwen3.6 · airline, 17 of 30 train cases flip between
  runs of the identical policy, and `select` promotes the best of the measured arms against that one run.
  Use `replicate`/`rescore` and `validate` before reading a train net as an effect.
* **A winner that never fired is not a repair.** `select` flags it, and `validate` refuses it.
* **Attribution needs selective firing.** A controller that fires on every case makes every gain trivially
  "where it fired".
