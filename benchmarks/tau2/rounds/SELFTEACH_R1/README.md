# Artifacts: tau-bench, self-teach, round 1

Everything behind the round-1 report (withheld in this branch while the tau2 campaign is open — see
[`../../../../docs/RESULTS_POLICY.md`](../../../../docs/RESULTS_POLICY.md)), exported from the run
directory by [`scripts/export_results.py`](../../scripts/export_results.py). Logs and job-scheduler
bookkeeping were left out, and no file holds a credential (a test checks).

**Raw trajectories are not committed.** The repo's `.gitignore` keeps raw trajectories out so it stays
clonable, and they were ~85% of this round's bytes (13.4 MB → 2.2 MB). No reported number and no anchor
depends on them: `summary.json` recomputes byte-for-byte and all 7 anchors rebuild from these files alone,
which `tests/test_tau2_committed_results.py` enforces. Each `baseline.json` has `events: {}` and an
`events_stripped` record saying how many cases were removed and where the full set is.

The **full set, trajectories included**, is archived outside the repo:

| | |
|---|---|
| path | `<matrix-dir>/_archive/SELFTEACH_R1_full.tar.gz` |
| size | 0.4 MB compressed, 13.4 MB uncompressed, 190 files |
| sha256 | `53b3f971bd963e5e6c3d90a268ee0bb4faa01de24b8cd724e1a7e6a9f8c0fb93` |

It is needed only to re-run `propose` (rebuild the full candidate set) from files, or to read the
conversations themselves. Extracting it over this directory gives the complete arm directories.

```bash
# recompute every reported number from these files -- no model, no GPU
PYTHONPATH=$PWD python benchmarks/tau2/scripts/export_results.py --from-dest --markdown
```

## Top level

| file | what |
|---|---|
| `summary.json` | every number in the report, per arm, derived from the arm directories below |
| `anchors.json` | the 7 promoted controllers in installable form: boundary, signal, action, variant, eta, instruction source, and the adapter boundary key. `accepted` is false for all |
| `<model>__self__<domain>/` | one directory per arm (9) |

## Inside an arm directory

Produced by the phases of [`run_anchoropt_round.py`](../../run_anchoropt_round.py), in this order:

| file | phase | what it holds |
|---|---|---|
| `arm.json` | — | the arm's identity: agent, teacher, domain |
| `baseline.json` | `baseline` | **P0 on train**: per-case reward and termination, the tool catalog, `scored_task_ids`, models, portal (`agent_provider`) and budgets. Trajectories (`events`) stripped here; in the archive |
| `arm_manifest.json` | `propose` | every candidate arm core built (54–66), with boundary, signal, action, eta; the residual it targeted; the teacher record (accepted instructions, refusals, provenance). **Names no winner** |
| `arm_01.json` … `arm_08.json` | `evaluate` | each measured arm's run on train: per-case solved, termination, and firings counted at the mechanism (`interventions_executed`, `cases_fired`) |
| `results.json` | `evaluate` | the measured arms paired against P0 (gains, losses, executions). An arm not run is omitted, never zero |
| `selection.json` | `select` | core's outcome class, the best measured arm, `promoted` (null unless a train win), `winner_engaged`, and the installable `controller_spec` |
| `incumbent_rep_1..4.json` | `replicate` | four more P0 train runs (qwen3.6 · airline and granite · airline only) |
| `rescore.json` | `rescore` | every measured arm scored against the 5 P0 runs: expected net, z, above-best |
| `heldout_p0_test_rep1..4.json` | `validate` | P0 on the test split, four runs |
| `heldout_p1_test_rep1..4.json` | `validate` | the promoted controller on the test split, four runs (arms with a train winner only) |
| `validation_test.json` | `validate` | the held-out comparison: totals, mean difference, se, z, pass^1..pass^4 for each side, cases fired, and per-case improvement split by whether the controller fired |

A patched void episode is recorded under `refilled` in the run file it was patched into (three:
qwen3.6 · airline task 44 in P0 run 2 and P1 run 3; granite · retail task 45 in P0 run 4).

## Rebuilding a controller from these files

```python
import json, sys; sys.path.insert(0, "benchmarks/tau2")
import importlib.util
spec = importlib.util.spec_from_file_location("drv", "benchmarks/tau2/run_anchoropt_round.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
from pathlib import Path
d = Path("benchmarks/tau2/rounds/SELFTEACH_R1/granite_4_1_30b__self__airline")
base, man = json.loads((d/"baseline.json").read_text()), json.loads((d/"arm_manifest.json").read_text())
label = json.loads((d/"selection.json").read_text())["promoted"]
controller = drv._controller_for(d, base, man, label)   # a tau2_mechanism.Controller
```

This needs the tau-bench checkout on `PYTHONPATH`, because the mechanism subclasses tau2's `LLMAgent`.
`tests/test_tau2_committed_results.py` does it for every anchor.

## Re-measuring the anchors

A later driver revision adds a `replay` phase that reruns a committed arm's P0-test and P1-test from
scratch, with the models, without reading or writing the saved runs here. **That revision is not
in this branch** — the `run_anchoropt_round.py` shipped here has `baseline`, `propose`, `evaluate`,
`select`, `replicate`, `rescore`, `validate`, and `refill` only (see
[`../../README.md`](../../README.md)). To re-measure an anchor with what's here,
build a fresh round against the same domain/model/teacher and compare against this one's
`selection.json`.
