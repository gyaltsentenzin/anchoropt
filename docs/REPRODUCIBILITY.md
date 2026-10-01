# Reproducing AnchorOpt — five benchmarks, three tiers, and the parts you cannot get from this repo

Written for someone reviewing or reproducing AnchorOpt who has **no access to our cluster**.

The honest answer to "can I reproduce this?" is **partly**, and the parts divide cleanly. This guide
states the barrier first rather than letting you discover it on attempt three, then gives the three
tiers in increasing cost, then the per-benchmark specifics.

For the definitions of `net`, `Δpp`, the noise floor and the four acceptance criteria, see
[`METRICS.md`](METRICS.md). For what this branch deliberately does not contain, see
[`RESULTS_POLICY.md`](RESULTS_POLICY.md).

---

## The barrier, up front

**The frozen incumbents are data, and they are not in this repo.**

Every AnchorOpt round is measured against a *frozen incumbent*: one recorded run of the stock agent on
a split, **including per-task step logs**. `mine_residual` reads those logs to build the residual and
evaluate signals at incision points. Outcome files are not enough — a pass/fail table cannot be mined.

Those logs cannot be shipped. One 90-task AppWorld run is **80–538 MB** depending on the model, and a
single model × split × repeat set runs to gigabytes. They are also not regenerable from this repo
alone: they are the product of a specific model serving a specific config at a specific time.

So to reproduce the **search**, you must first create **your own incumbent**: run the stock agent on
your split with your own model, keeping the per-task logs, then point the adapter at it. Your absolute
numbers will differ from ours — different model, different serving stack — but the *method*
reproduces: mine, propose, screen, score, promote or reject.

The one exception is **CCTU**, whose transcripts are small enough to ship and do ship. It is the only
end-to-end reproduction here that needs nothing but a CPU.

---

## Tier 1 — runs anywhere. No model, no GPU, seconds

This is the tier that verifies the *method* and the *published BFCL numbers*, and it is not trivial.

```bash
pip install -e ".[dev]"

pytest -q                                   # the whole integrity suite
python scripts/verify_progression.py        # recomputes every published accuracy; non-zero exit on drift
python scripts/walk_framework.py            # steps through the method on the real artifacts
python scripts/run_pipeline.py --show       # the composed A1-A4 policy
```

`verify_progression.py` is the one to run first. It recomputes the nine published accuracies from the
shipped per-case results rather than reading them from prose, and on this branch it prints:

```
OK -- all 9 published accuracies reproduced from the shipped per-case results.
   RECOMPUTED HERE: 29.04 % -> 42.24 % (+13.20 pp) over 6 rounds / 4 anchors.
   FULL STACK (transcribed for T6, T7): 30.03 % -> 52.48 % (+22.44 pp), 8 anchors.
```

A figure in this repo is data in one file and prose never restates one, so if a document and
`rounds/anchors.py` ever disagree, `tests/test_progression_integrity.py` fails rather than letting the
document drift.

**What Tier 1 covers:** the search logic, the signal grammar, the policy class, the termination and
acceptance rules, the screens, and every offline analysis that is not an arm score.

### Checking an adapter without a model

```bash
python scripts/check_adapter.py <import.path.to.your_adapter> --events real_rows.json
```

Pass **real** trajectory rows. The contract checker exists to catch the porting bugs that otherwise
surface only after a scoring run: a declared-but-unexecutable action, a boundary whose observability
is asserted rather than probed, evidence that cannot be traced to the controller's own telemetry.
See [`ADAPTER_GUIDE.md`](ADAPTER_GUIDE.md).

---

## Tier 2 — needs a model endpoint you supply. No cluster

Every adapter talks to an **OpenAI-compatible `base_url`**. Ours happens to sit behind an internal
auth proxy on a fixed port per model; **nothing depends on that.** vLLM, Ollama, TGI or any hosted
OpenAI-compatible API works.

Two failure modes cost us real results. Both are cheap to avoid and neither is detectable after the
fact without the raw logs:

> ### ⚠ Pin the generation config on *both* sides
> Qwen arms once ran at `max_completion_tokens: 8192` against an **uncapped** incumbent — and the
> signal under test fires on the cap being hit. The treatment manufactured its own trigger. Verify
> what was actually *sent*, from `input.max_completion_tokens` in `lm_calls.jsonl`, not what a config
> file says.

> ### ⚠ Match concurrency to the incumbent's
> `--num-processes N` must equal the N the incumbent was run at. On a server you are the sole tenant
> of, a different N means different batch shapes — a second uncontrolled difference layered on top of
> the controller, which is no longer a paired comparison.

Then, in order, for any benchmark:

```bash
# 1. Mine your own incumbent's residual and let the search propose. Free -- no LLM calls.
# 2. Screen before spending episodes: an arm firing < --min-firings cannot promote, and an arm whose
#    headroom ceiling is <= the noise floor is UNRESOLVABLE before it is run.
# 3. Score. Real inference. Two passes, not one -- a single number cannot be told from the 2-5 task
#    spread between same-config baseline repeats.
# 4. Validate a promoted arm on the held-out split. Only an already-promoted arm goes there.
```

Step 4 is a discipline, not a suggestion: the held-out split scores **only** an arm already promoted
on train. Using it to choose between arms makes it a selection set, and its figure stops being a
generalization estimate.

---

## Tier 3 — our site, and what to substitute

| we use | why | substitute |
|---|---|---|
| LSF (`bsub`, `bjobs`) | batch scheduling | anything that runs a command with GPUs; only cluster job-submission wrappers care, and those are not shipped |
| GPFS (`mmlsquota`) | a quota guard, added after two runs died mid-episode on a full filesystem | drop it, or swap in `df` |
| a sibling repo's venv as `APPWORLD_PY` | historical: the two projects grew together | any interpreter that imports `appworld` and `anchoropt` |
| a per-model auth proxy on a fixed port | one proxy per model, in front of a gateway needing a non-standard header | any OpenAI-compatible endpoint's URL |

For AppWorld, all of it is configured in one place, and every variable falls back to our values — so
sourcing it is a no-op for us and a single edit for you:

```bash
source benchmarks/appworld/env.sh   # then:
appworld_env_check                  # fails loudly on a missing interpreter, root or incumbent
```

`appworld_env_check` failing early is the point. The alternative is discovering a missing incumbent
halfway through a scoring run.

---

## Per-benchmark specifics

Five adapters ship. They are at different maturity levels, and that is stated rather than averaged
away.

| benchmark | adapter | entry point | maturity |
|---|---|---|---|
| **AppWorld** | `benchmarks/appworld/` | `run_round.py` | **reference implementation** — richest, and the only one with headroom screening (`headroom.py`), behavioural probes (`probes.py`) and a held-out driver (`heldout_validation.py`) |
| **BFCL v4** | `benchmarks/bfcl_v4/` | `run.py` | the original; where the A1–A8 progression was measured. Vendored harness, Apache-2.0 |
| **CCTU** | `benchmarks/cctu/` | `cctu_run_arms.py` | complete, with shipped transcripts for CPU-only re-scoring |
| **τ² (tau2)** | `benchmarks/tau2/` | `run_anchoropt_round.py` | complete adapter with a capability audit; **runner parity against stock tau-bench is unresolved** (~1.5 tasks, z ≈ −1.0) — read `TAU2_ADAPTER.md` before trusting a τ² number |
| **TB2 / deepagents** | `benchmarks/tb2_deepagents/` | `tb2_adapter.py` | thinnest. An integration audit exists (`TB2_INTEGRATION_AUDIT.md`); treat it as a porting example |

### AppWorld — start here

The most complete path, and the one the guide above is written against.

```bash
source benchmarks/appworld/env.sh && appworld_env_check
python benchmarks/appworld/headroom.py --help     # screen first
python benchmarks/appworld/run_round.py --help    # mine, propose, score
python benchmarks/appworld/metrics.py --help      # recompute an arm's absolute score
```

Read [`APPWORLD_SETUP.md`](APPWORLD_SETUP.md) for the environment,
[`APPWORLD_ANCHORS.md`](APPWORLD_ANCHORS.md) for the anchors that worked *and the ones that did not*,
and [`CORE_FINDINGS_FROM_APPWORLD.md`](CORE_FINDINGS_FROM_APPWORLD.md) for what the round changed about
the core — including which findings are general and which are AppWorld's accidents.

Note the archiving rule: **episodes must be archived between repeat passes**, or the second pass reads
the first one's tree.

### CCTU — the only CPU-only end-to-end reproduction

```bash
python benchmarks/cctu/verify_plumbing.py         # check the wiring before spending anything
python benchmarks/cctu/cctu_run_arms.py --help
```

[`CCTU_REPRODUCE.md`](CCTU_REPRODUCE.md) Level 1 re-scores the control and the deployed anchor on both
splits **from the shipped transcripts**, with no GPU, and checks the invariant that matters: every
non-firing episode must reproduce the control exactly. Levels 2 and 3 re-run the control and the anchor.

CCTU is **vendored** third-party work — see [`../benchmarks/cctu/VENDORED.md`](../benchmarks/cctu/VENDORED.md)
for the upstream attribution (Ye et al., Apache-2.0) and the credential-handling history. Its client
reads `HOSTED_API_KEY` / `HOSTED_API_BASE_URL` from the environment and **hardcodes nothing**.

### BFCL v4

```bash
python benchmarks/bfcl_v4/run.py --compare --shard all --dry-run   # the commands a real run needs
```

`--dry-run` prints the concurrent commands without launching them. The harness is a vendored
Apache-2.0 fork; `benchmarks/bfcl_v4/VENDORED.md` records what was changed and why. The MEMORY task
needs far less than the harness's own 32 declared dependencies — `pip install -e ".[bfcl]"` covers
`memory_kv` and `memory_rec_sum`; `".[bfcl-vector]"` adds the embedding stack for `memory_vector`.

### τ² and TB2

```bash
python benchmarks/tau2/run_anchoropt_round.py --help
```

Read [`TAU2_ADAPTER.md`](TAU2_ADAPTER.md) first — specifically §2, which lists the variants
deliberately **not** grounded, and §3, which states the variance floor and therefore what a round may
claim. The τ² results from our own round are not in this branch; see
[`RESULTS_POLICY.md`](RESULTS_POLICY.md).

---

## What reproduces, and at what cost

| claim | reproducible without our data? | cost |
|---|---|---|
| the BFCL A1–A8 published accuracies | **yes** — from shipped per-case results | seconds, no GPU |
| search logic, decision rules, acceptance criteria, screens | **yes** | seconds, no GPU |
| an adapter's contract conformance | **yes** | seconds, no GPU |
| CCTU control + anchor re-score, both splits | **yes** — shipped transcripts | minutes, no GPU |
| the support and headroom screens on *your* incumbent | yes | minutes, no GPU |
| an arm's absolute score from an episode tree | yes, on your own tree | minutes, no GPU |
| **our specific AppWorld/τ² arm nets and promotions** | **no** — needs our incumbents | — |
| the method end to end on your model | yes | 1 baseline run per split, then ~90 episodes per arm |

**Budget, from our measurements.** A 90-task AppWorld pass is ~30 min at 4-way concurrency on a shared
endpoint, and ~1h15m–2h on a slower model. Evaluation wall-time is dominated by **filesystem
contention, not compute**: we measured `evaluate_tasks` at 0.6 s/task on a quiet filesystem and
**3.4 min/task against 11 concurrent jobs**. Serialise runs; do not parallelise the evaluator.

---

## Known issues

See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md). The suite is green except for nine tests in two files that
encode a pre-held-out-validation acceptance semantics; the diagnosis and the reason they are left
failing rather than quietly edited are recorded there.
