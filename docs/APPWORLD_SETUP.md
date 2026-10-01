# AppWorld × AnchorOpt — setup and run guide

How to stand up the AppWorld adapter and run one optimization round, from a clean shell. Its
companion is `APPWORLD_RESULTS.md` (not in this branch, see [`RESULTS_POLICY.md`](RESULTS_POLICY.md)),
which records what has been measured; this file records how to get there.

Adapter in `benchmarks/appworld/`. For common pitfalls while running this, see the "gotchas" section
in [`../benchmarks/appworld/README.md`](../benchmarks/appworld/README.md).

Mining and proposing are free: offline log replay plus a pure search function, no model calls. Only
**scoring** issues inference.

---

## 1. What the round is

AppWorld is wired into the **existing** AnchorOpt optimization algorithm. The adapter supplies state
extraction, observability and execution; core keeps ownership of where/what/how the search runs.

> frozen incumbent → mine residual → localize boundary → ground arms → paired measurement → accept or expand

---

## 2. Components

| Component | Notes |
|---|---|
| AppWorld checkout | upstream clone + local patches, see §3. Not vendored — too large and under its own license |
| Python environment | needs `appworld` and this repo importable together — see [`../benchmarks/appworld/env.sh`](../benchmarks/appworld/env.sh) |
| Adapter | `benchmarks/appworld/`, tracked in this repo |
| Incumbent baselines + splits | **data, not code** — see §7. You generate your own by running the stock agent |

Source [`benchmarks/appworld/env.sh`](../benchmarks/appworld/env.sh) once per shell; it sets every
path below from environment variables with no personal defaults, and fails loudly naming what to set
if you skip one.

---

## 3. The AppWorld checkout

Upstream `https://github.com/StonyBrookNLP/appworld.git`, pinned at **`42b5bcf`** for the splits and
baselines below, with local patches:

1. **A robustness fix to `experiments/code/simplified/react_code_agent.py`**:
   `output.get("reasoning_content", "")` → `(output.get("reasoning_content") or "").strip()`.
   Some OpenAI-compatible endpoints return an explicit `None` here, which crashes the unpatched agent.
2. **Per-model generator entries and `.jsonnet` configs** for whichever models you serve — one entry
   per model pointing at its endpoint, plus per-split configs generated from it.

> A new model needs both an entry in the generator **and** per-split `.jsonnet` configs. The fastest
> safe route is to copy an existing model's config directory and change only `base_url` —
> regenerating wholesale can overwrite configs other runs depend on.

---

## 4. Model endpoints

Point each model at any OpenAI-compatible endpoint — a hosted API, your own vLLM server, or a
gateway in front of either. If your gateway needs a non-standard auth header (not bare
`Authorization: Bearer`), front it with a small local auth proxy and point AppWorld's `base_url` at
that instead.

The port (or endpoint) is baked into each model's `base_url` in the generator, so **do not change it
to dodge a collision** — a different endpoint renders a different agent config and the comparison
against the frozen incumbent stops being paired.

---

## 5. Environment

```bash
source benchmarks/appworld/env.sh
appworld_env_check   # fails loudly, naming what's missing, before you spend anything
```

`APPWORLD_ROOT` changes meaning by stage, which is the single most common mistake:

| Stage | `APPWORLD_ROOT` must be | Why |
|---|---|---|
| mine residual, propose | the round cell holding the incumbent's transcripts | that is where the data mining reads actually lives |
| score an arm | your AppWorld checkout | a round cell ships no `experiments/configs`, so an arm cannot run there |

---

## 6. Verification gates

Run these before anything expensive. Each has a known-good output.

```bash
# a. core regression guard -- the reference adapter still passes, unchanged
python -m pytest tests/test_toy_host_e2e.py tests/test_executor_behavioral_contract.py -q
#   => 28 passed

# b. declaration-level contract + capability audit
python -c "
from anchoropt.testing import check_adapter_contract
from benchmarks.appworld.adapter import ADAPTER
import json; print(check_adapter_contract(ADAPTER))
print(json.dumps(ADAPTER.capability_audit(), indent=2))"
#   => no FAIL, no unbound or ghost capability cells

# c. behavioral probes -- proof each declared capability actually alters the run
cd benchmarks/appworld && python -m pytest probes.py -q
#   => 9 passed   (6 capability probes, 1 negative control, 2 budget tests)
```

Gate (c) is the important one: it runs each real mechanism with the controller absent and present and
asserts they differ **in the way the action claims**. A telemetry-only mechanism fails it by design.

---

## 7. Splits and the frozen incumbent

Splits are drawn **by scenario, not by task**: SGC is a mean over scenarios of
`min(per-task success)`, so a task-level draw fragments scenarios and inflates it.

| Role | Dataset | Tasks | Scenarios | Provenance |
|---|---|---|---|---|
| held-in | `train` | 90 | 30 | AppWorld's own official train split |
| held-out | `sh_heldout` | 90 | 30 | scenario-stratified from `test_normal` + `test_challenge`, SEED=42 |
| round 1 | `dev` | 57 | 19 | AppWorld's official dev split |

The `train`/`sh_heldout` evaluations and their per-task transcripts live wherever you ran your own
baseline — **each such cell is itself a valid `APPWORLD_ROOT`** (own `data/` and
`experiments/outputs/`), which is what lets the adapter mine those splits with no code change and no
new episodes. These are data you generate yourself; they are not shipped (a single run's logs are
80–538 MB — see `benchmarks/appworld/env.sh`).

Baselines measured here (stock agent, mean of two repeats). TGC = per-task pass rate; SGC = fraction
of scenarios where every task passes:

| Model | Held-in TGC / SGC | Held-out TGC / SGC |
|---|---|---|
| MiniMax M2.5 | 65.55 / 53.30 | 54.45 / 36.65 |
| Qwen3.6-35B-A3B | 74.45 / 58.35 | 62.20 / 40.00 |
| granite-4.1-30b | 25.00 / 11.65 | 25.55 / 6.70 |

> **Noise floor — read before believing any delta.** Those repeats are the same harness, model and
> tasks, differing only in sampling, and they disagree by **2–5 tasks** on TGC. On MiniMax held-in, SGC
> was identical across repeats (spread 0.0pp), so SGC moves are more informative there. Any
> single-repeat delta smaller than this is not evidence.

---

## 8. Running a round

### Step 1 — mine the residual (free)

```bash
CELL=<path to your round-1 baseline cell for this model>
EV=$CELL/experiments/outputs/simplified_react_code_agent/<creator>/<model>/train/evaluations/train.json

APPWORLD_ROOT=$CELL python benchmarks/appworld/residual.py --evaluation "$EV"
```

Expected shape for a held-in run: `failing_task_ids` and `mined_task_ids` matching the number of
failing tasks, three times that many records across the three boundaries, and a
`post_execution_error_kinds` histogram. A non-zero `residual.py: no environment_io.md log for N/M
failing task(s)` warning on stderr means transcripts are missing — investigate before proceeding.

### Step 2 — propose (free)

```bash
APPWORLD_ROOT=$CELL python benchmarks/appworld/run_round.py propose \
  --evaluation "$EV" --out runs/<model>_heldin
```

This is **not** "pass 1 of N". Each call advances the search past whatever has already been measured.
Read the `state:` line:

| State | Meaning | Next |
|---|---|---|
| `REALIZABLE_UNMEASURED` | arms built, nothing measured yet | score them (step 3) |
| `NO_BENEFIT` | measured, did not help | core expands Φ to an earlier boundary and builds new arms |
| `IMPROVED` | an arm passed `improves` | promoted; validate on held-out (step 5) |
| `BUDGET_EXHAUSTED` | declined to spend more | distinct from `NO_BENEFIT`; do not collapse them |

It writes `arm_manifest.json` (core's work order) and `controllers.json` (how to build and run each
arm). Put these on storage every worker that will score an arm can see.

### Step 3 — score (COST-BEARING, real inference)

Run each arm in `controllers.json` against the model, however you submit work on your own
compute (a job scheduler, a local loop, whatever you have):

```bash
python benchmarks/appworld/run_round.py score \
  --controllers runs/<model>_heldin/controllers.json \
  --evaluation "$EV" --experiment <experiment_name> \
  --check-regressions --out runs/<model>_heldin/arm_results.json
```

- Which model is called comes from the `--experiment` config's own `base_url`, not a separate flag.
  `--experiment` defaults to a name inferred from `--evaluation`'s path, but pass it explicitly
  whenever that inference wouldn't name the right config.
- `--check-regressions` is **not optional in practice**: without it the arm runs only on the residual,
  `losses` is structurally empty, and the number is a floor rather than a measurement.
- Use a task-count limit for a smoke run first. A 2-task smoke validates the whole live path in
  minutes; do that before a 90-task arm.
- Already-measured arms are skipped and merged, so a resubmit is cheap (`--rescore` forces a redo).

### Step 4 — propose again (free)

Same as step 2, plus `--results runs/<model>_heldin/arm_results.json`. Repeat until core promotes or
reaches a terminal state.

### Step 5 — validate on held-out (COST-BEARING)

Build a `controllers.json` holding **only** the promoted arm, with `incumbent_token` set to the
held-out evaluation's absolute path, then score against the `sh_heldout` cell with
`--check-regressions`. This is the generalization number, and the only one that answers "does it
transfer".

### Step 6 — record (free)

```bash
python benchmarks/appworld/metrics.py --results-dir runs --out runs/appworld_anchoropt_metrics.json
```

Recomputes from disk: baselines, search state, and per-arm before/after TGC and SGC on both splits.
Raw artifacts are gitignored by design; put durable numbers in `APPWORLD_RESULTS.md`.

---

## 9. Reading results honestly

- **The four outcomes are distinct.** `IMPROVED`, `UNEVALUATED` (`REALIZABLE_UNMEASURED`),
  `NO_BENEFIT` and `BUDGET_EXHAUSTED` mean different things; a measured loss must never be collapsed
  into `NO_BENEFIT`.
- **SGC can move when TGC barely does.** One arm moved TGC −1.11pp and SGC −10.00pp: it fixed tasks in
  still-failing scenarios (worth nothing to SGC) while breaking tasks in all-passing ones (a whole
  scenario each).
- **Check `interventions_executed`.** An arm reporting zero executions has a delta it did not cause.
- **The remedy wording is authored, not discovered.** Core chooses the boundary, signal and action; the
  instruction text is a per-`(boundary, action)` template in `adapter.py`. A weak result may be a
  badly-worded instruction rather than a wrong boundary, and one round cannot separate those.
