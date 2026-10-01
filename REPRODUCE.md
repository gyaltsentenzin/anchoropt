# Reproducing the A1–A4 result

Three tiers, from "works on a laptop in ten seconds" to "needs a GPU and the BFCL harness." Start at
Tier 1 — it verifies every published number without a model.

**What "reproduce" means here:** the four **accepted** anchors and the nine accuracies they produced.
That is the reproduction target.

Signals and policies the loop **deferred or rejected** are a different thing, and they are kept
deliberately — they are method content, not clutter. A method whose record shows only its successes is
not inspectable. So:

| | where it lives | in the replication path? |
|---|---|---|
| the 4 accepted anchors | `rounds/T{1,2,3,5}/` | **yes** — Tier 1 verifies all nine numbers |
| deferred signals & policies (A2.W, sub-floor loci, the 3 rejected detectors, bucket D) | [`docs/KEEP_OR_DEFER.md`](docs/KEEP_OR_DEFER.md) §4, with the rule that stopped each | documented, with evidence |
| two non-accepted **arms** (A4 v1, A2 reroute) | `rounds/*/result/`, printed under `--with-evidence` | no — evidence for the incision-point claim, not a target |

The last row is the only exclusion: re-deriving our failed arms is not something a collaborator should
have to do to trust the result.

---

## Setup (once, ~30 s)

Needs **Python 3.10+**. Two dependencies: `jsonschema` and `scipy` (plus `pytest`/`ruff` for dev).

<details open>
<summary><b>with <code>uv</code></b> (fastest)</summary>

```bash
git clone <this-repo> anchoropt && cd anchoropt
uv venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```
</details>

<details>
<summary><b>with stock <code>venv</code> + <code>pip</code></b></summary>

```bash
git clone <this-repo> anchoropt && cd anchoropt
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```
</details>

Confirm you are in the right interpreter before continuing — this is the one step worth checking,
because a shell with another project's venv active will silently use *that* Python:

```bash
python -c "import sys, anchoropt; print(sys.version.split()[0], anchoropt.__file__)"
# 3.12.x  /path/to/anchoropt/anchoropt/__init__.py     <- must be THIS repo
```

If you would rather not activate anything, call the interpreter by path instead and skip the check —
every command below works with `./.venv/bin/python` substituted for `python`.

## Tier 1 — verify every published number (no GPU, ~10 s)

Recomputes the whole progression from the shipped per-case results, including the paired sign tests.

```bash
python scripts/verify_progression.py
```

Expected output:

```
T0  native (no prompt, no anchors)     88/303 =  29.04 %
T1  + A1  reroute                     102/303 =  33.66 %   Δ +4.62 pp   42g/28l  p=0.1196
T2  + A2  reprompt                    109/303 =  35.97 %   Δ +2.31 pp   10g/ 3l  p=0.0923
T3  + A3  suppress                    117/303 =  38.61 %   Δ +2.64 pp   14g/ 6l  p=0.1153
T5  + A4  reprompt                    128/303 =  42.24 %   Δ +3.63 pp   16g/ 5l  p=0.0266
CUMULATIVE  T0 -> T5                  +13.20 pp   67g/27l  p=0.0000
```

**Exit code 0** means every published figure was reproduced; **exit 1** means at least one disagreed,
and the mismatch is named (e.g. `got 116/303, expected 117/303`). It is a real check, not a printout —
flipping a single case in 303 fails the run.

It also writes **`verification_report.json`**: every accuracy, delta, gain/loss count and p-value, with
the sha256 of each artifact it read. Read that rather than parsing stdout if you are pulling these
numbers into a table or a CI diff.

```bash
python scripts/verify_progression.py --with-evidence   # also prints the two non-accepted arms
echo $?                                                # 0 = all reproduced
```

Two things worth reading in that output:

- **Only A4's own step reaches p < 0.05.** The other three are directionally consistent but individually
  underpowered — the cumulative effect is significant (p ≈ 4.5e-05), the per-round steps mostly are not.
  Stated here because it is easy to over-read a table of positive deltas.
- **Held-out A4 is +3.57 pp at p = 0.5488** — the point estimate transfers almost exactly (+3.63 → +3.57)
  and significance does not. The pre-stated bar was directional consistency, not significance.

`--with-evidence` adds the two non-accepted arms (A4 v1 at −4.95 pp; the A2 reroute arm at 0 flips).
Those are **evidence for the incision-point claim, not reproduction targets**.

## Tier 2 — the tests (no GPU, ~5 s)

```bash
python -m pytest tests/ -q        # expect: 187 passed
```

Four suites, checking different things:

| suite | what it pins |
|---|---|
| `test_mechanisms.py` | A1's write repair and A3's suppression predicate. The **safety trades** are tested directly: A1 *refuses* rather than dropping a number; A3 declines reorderings and superstrings rather than risk suppressing real information |
| `test_verify_script.py` | the replication script's own failure paths — tampering with one case must exit 1; a missing file must fail |
| `test_progression_integrity.py` | the README table against `rounds/anchors.py`, and A4's frozen spec against the sha256 cited in its acceptance commit (`a204313cd34eb761`) — so a criterion cannot be edited after its result landed |
| `test_search_space_funnel.py` | the incision grid: 9 of 12 cells admissible, every exclusion carrying a stated reason |

Then read [`anchoropt/mechanisms/`](anchoropt/mechanisms/) — both files are short, and they are the
clearest statement of what an anchor mechanism actually is. To poke at them directly:

```bash
python -c "
from anchoropt.mechanisms import capacity_repair as cr, redundant_write as rw
print(cr.reduce_preserving_facts('No figures here. Reading was 7.2 units.', 40))
print(rw.is_redundant('value_a', 'value_a'))
print(rw.is_redundant('value_a plus more', 'value_a'))
"
```

### Reading the progression

```bash
python rounds/anchors.py     # regenerates the README delta table from the source of truth
ls rounds/                   # the five rounds, in order
```

Start at [`rounds/README.md`](rounds/README.md) for the causal chain between anchors, then read each
round's `README.md` beside its `FROZEN.md`.

## Tier 3 — re-run the arms end to end (GPU + BFCL v4 + cluster)

> ### ⚠️ Use the right split — there are two, and the default is the wrong one
>
> `benchmarks/bfcl_v4/data/` contains two leakage-free splits of the same corpus. **They are not
> interchangeable**, and running the wrong one produces numbers that look like a failed reproduction
> but are simply a different corpus:
>
> | case files | split | scored |
> |---|---|---|
> | `g8_balanced_{train,test}_cases.json` | **`balanced_one_fold`** — holds out customer/kv + finance/rec_sum + student/vector, one cell per backend | **303 train / 84 test** ← **the published numbers** |
> | `memory_{train,test}_cases.json` | W5 hybrid — holds out the whole healthcare scenario | 408 / 90 |
>
> So always pass the fold explicitly:
>
> ```bash
> --cases g8_balanced_train_cases.json     # 303 scored -> 29.04 % / 42.24 %
> --cases g8_balanced_test_cases.json      #  84 scored -> 16.67 % / 26.19 %
> ```
>
> Omitting `--cases` falls back to `memory_<split>_cases.json`, i.e. the 408-case W5 split. Differencing
> its result against the baseline would be exactly the cross-corpus comparison this project forbids
> everywhere else. `g8_balanced_manifest.json` records the fold's provenance (strategy, seed,
> dev cells, `corpus_sha256`).

### Shard by backend, and submit the shards concurrently

**This is the default protocol, not an optimisation.** A 3-backend × 2-arm contrast is **six
independent jobs**, one GPU each:

```bash
python benchmarks/bfcl_v4/run.py --compare --shard all --dry-run
```

That prints six commands, each with the environment it needs, ready to hand to a scheduler. Running
them in a loop works and is not the protocol.

**It buys determinism, not just wall-clock.** Whole-corpus runs are **not byte-reproducible** — two
runs of the same effective policy diverge even when the treatment fires zero times, because
whole-corpus execution adds cross-chain sequencing that is not byte-stable. Sharded pairs are
bit-identical except in the shard where the treatment actually fires. So a paired train number from a
whole-corpus run should not be quoted.

Three things each shard needs, and each has broken a real fleet when omitted:

| | why |
|---|---|
| a port per **(arm × backend)** | not per backend — two arms sharing a port were safe once only because the scheduler happened to place them on different hosts |
| its own snapshot cache | the cache key hashes the prerequisite id list, so shards necessarily differ. That is correct, and it means equivalence must be checked on store **content**, never on cache keys |
| the **same** store fingerprint within an arm | that is what makes the three shards' results comparable when concatenated |

`--shard` emits all three. It also **asserts the shard is self-contained** before writing it: sharding
is sound on this corpus only because no case dependency crosses a backend boundary, and if that ever
changes the shard would build stores from partial chains. It refuses rather than warning.

**Keep `--workers 1` in every shard.** Parallelism *across* isolated runs is measured at 0 flips;
parallelism *within* one store build is forbidden. Different axes — sharding is not permission to raise
`--workers`.

After the shards land, concatenate the results and verify the partition is complete before scoring.
Sharding is also how the **per-backend** figures are produced: the full stack's per-backend cells in
the README are currently *derived* rather than measured, and
`python scripts/derive_per_backend.py` cross-checks that derivation against the shipped artifacts and
reports the one-case drift it finds.

> ### The full 303-case run — measured twice, once with A3 broken and once fixed
>
> **One caveat bounds everything below: this is not the official BFCL pipeline.** `run.py` drives a
> reimplemented episode loop that borrows BFCL's tool executor, memory backends and official scorer but
> replaces its inference loop. Six divergences are documented file:line in
> [`docs/FIDELITY_AUDIT.md`](docs/FIDELITY_AUDIT.md) — five still present here, the highest-impact being
> that prereq seeding runs in a different environment than scoring. So these runs show **this pipeline
> and these artifacts are self-consistent**; they are not evidence about an official BFCL submission.
>
> | | control | A1–A4 | paired delta | p |
> |---|---|---|---|---|
> | **A3 fixed** (job 1003964) | 90/303 = 29.70 % | **129/303 = 42.57 %** | **+12.87 pp**, 68 g / 29 l | 9.3e-05 |
> | A3 mis-scoped (job 976709) | 90/303 = 29.70 % | 128/303 = 42.24 % | +12.54 pp, 62 g / 24 l | 5.1e-05 |
> | published | 88/303 = 29.04 % | 128/303 = 42.24 % | +13.20 pp, 67 g / 27 l | ≈4.5e-05 |
>
> **Every anchor was verified as firing** — the check that was missing the first time:
>
> | anchor | telemetry flag | firings | episodes |
> |---|---|---|---|
> | A1 | `a1_core_full_gate` / `reroute_gate` / `capacity_repair_gate` | 257 / 221 / 45 | 55 / 49 / 21 |
> | A2 | `a2_key_not_found_gate` | 43 | 43 |
> | A3 | `redundant_write_gate` | 103 *(110 calls removed)* | 27 |
> | A4 | `zero_call_reprompt_gate` | 126 | 68 |
>
> The **control arm fired zero gates**, confirmed from its own trajectories rather than assumed.
>
> #### What the two runs together establish
>
> **Fixing A3 moved the result by exactly one case** (128 → 129). That matches the prediction derivable
> from A3's frozen spec: the correct predicate fires on 55 of 59 candidates while the broken one fired on
> all matching calls, so the two differ on ~4 suppressions — three paraphrases of already-stored values
> plus one genuinely new fact — and at most one could touch a scored query. A real bug with a small blast
> radius on *this* corpus, not a harmless one: silent data loss is the worst failure mode, and a corpus
> with more novel archival writes would diverge further.
>
> **A3 is now demonstrably conditional**, not a name-matcher: 103 firings removing 110 calls, where the
> broken version would have dropped every `archival_memory_add`.
>
> #### What remains unexplained
>
> **The control reads 90, not the published 88 — twice, under two different stores.** I first attributed
> that to store variation; reproducing it exactly under a *different* store refutes that. It looks
> systematic to this repo's configuration. The leading candidate is fidelity divergence **#3**: the
> published runs used a step budget of **15**, and **this repo uses 20 everywhere** — the official
> `MAXIMUM_STEP_LIMIT`. Five extra steps would plausibly let two borderline episodes finish.
>
> **20 is the settled default and will not be lowered to chase 88.** It is the officially faithful value,
> so the two numbers are each correct for their own step budget and this repo is *more* faithful on that
> axis. Every default is 20 — `run.py`, `run_memory_eval.py`, `run_memory_train.py`, and both evaluator
> constructors — asserted against the vendored `MAXIMUM_STEP_LIMIT` by `tests/test_step_budget.py`, so a
> stray 15 fails the suite rather than silently producing incomparable numbers.
>
> The 90-vs-88 attribution itself is still **untested**: a one-off control at `--max-steps 15` would
> confirm the mechanism, and is worth running once as a diagnostic — but the default stays 20 either way.
>
> So the accurate claim stays narrow: **the pipeline runs the full corpus and reproduces the anchors arm
> to within one case, with every anchor verified as engaging.** It is not bit-exact, and not a statement
> about official BFCL.
>
> #### The lesson, recorded because it is the method's own rule
>
> On the first run I read the outcome (128/303, matching) and **skipped the engagement check** — exactly
> what `check_attribution.py` exists to run *first*. A number that matches expectation is not evidence
> the mechanism engaged. That is how a broken A3 passed unnoticed through a run reported as verified.
>
> The 18-case smoke corpus (`--cases memory_smoke_cases.json`) also runs and is useful as a fast path
> check, but it is **not a result**: 3 scored queries cannot measure anything, and `--check` correctly
> reports that its scores match no published arm.
>
> Worth knowing before you try it: **five submissions were needed to get there, and four of them found
> real bugs** — an argparse footgun that swallowed `--cases`, a package-name collision on `anchoropt`,
> another user's hardcoded path to a vLLM binary, and a bare-filename-vs-path assumption that failed
> *after* the model had loaded. All are fixed, and all had passed 178 tests and `--dry-run` beforehand.
> Run yours on real hardware early.

This regenerates the `eval_*.json` files that Tiers 1–2 consume. It is **not** required to verify the
result, and it is not a one-command operation.

**What you need:**

| | |
|---|---|
| model | `granite-4.1-8b`, `temperature=0.001` (the zero variance floor depends on this) |
| harness | BFCL v4 Agent Memory, with the `ZeroDivisionError` fix on empty-memory kv search |
| corpus | 387 scored + 111 prereq cases; the balanced fold split (leakage 0) |
| compute | ~40 min per arm on 1 GPU, plus ~35 min prereq store build unless cached |
| runner | the evaluator in the working repo (see below) |

**The policies are here.** Each round ships the exact learned artifact:

| round | policy | the stack it turns on |
|---|---|---|
| T1 | [`rounds/T1_A1_capacity/policy.json`](rounds/T1_A1_capacity/policy.json) | `enable_reroute`, `enable_capacity_repair` |
| T2 | [`rounds/T2_A2_not_found/policy.json`](rounds/T2_A2_not_found/policy.json) | + `on_domain_error_key_not_found` |
| T3 | [`rounds/T3_A3_duplicate/policy.json`](rounds/T3_A3_duplicate/policy.json) | + `enable_redundant_write_suppress` |
| T5 | [`rounds/T5_A4_no_tool_call/policy.json`](rounds/T5_A4_no_tool_call/policy.json) | + `enable_zero_call_reprompt` |

Each is self-contained JSON carrying its own rationale in a `_comment` field, and every
`on_memory_preamble` is **empty** — this line has no global prompt anywhere. You can read the whole
A1→A4 accumulation off the flags.

**The evaluator IS in this repo** — `benchmarks/bfcl_v4/evaluator/memory_evaluator.py`, 3,621 lines, vendored with the harness. It has no upstream counterpart (`bfcl generate` is never invoked) but calls BFCL for tool execution, backends, context assembly and official scoring. Six documented divergences from the official pipeline: [`docs/FIDELITY_AUDIT.md`](docs/FIDELITY_AUDIT.md). What you still supply is a **GPU and a served model**.

**Ordering matters.** Each round's control is the *previous* round's arm, not a re-run baseline. The
frozen specs say so explicitly (A3's names `results/n0clean_train/a1a2` as its control). Never
difference accuracies across jobs — two controls here read 29.04 % and 24.36 % purely from corpus
composition while agreeing to 1 flip in 228 on shared cases.

---

## Applying it to a different benchmark

**Read [`docs/GENERALIZABILITY.md`](docs/GENERALIZABILITY.md) first.** The algorithm is general; signal
extraction and policy-space trimming are benchmark-specific glue, and that document names the three
seams and what you will have to write yourself.

The one thing to do before anything else: **measure your own variance floor** on identical-policy
replicates. Ours is zero, which is why a single-case loss is treated as real here. If yours is not near
zero, every harm tolerance in this repo needs widening on your benchmark.

## Provenance of the shipped results

Every `rounds/*/result/*.json` was copied unmodified from the original run outputs on the cluster where
the arms executed:

| shipped as | original |
|---|---|
| `T1_A1_capacity/result/baseline_T0_train.json` | `results/g8_train/prompt/control/` |
| `T1_A1_capacity/result/eval_train.json` | `results/n0a1v2_train/anchor/` |
| `T2_A2_not_found/result/eval_train.json` | `results/n0clean_train/a1a2/` |
| `T3_A3_duplicate/result/eval_train.json` | `results/n0a3_train/suppress/` |
| `T5_A4_no_tool_call/result/eval_train.json` | `results/n0a4v2_train/zerocall/` |

The T0 baseline sits under a `g8_*` path because that experiment was retrospectively designated A0 and
**historical paths were deliberately never renamed** — a frozen artifact's hash must keep meaning what
it meant. Its *control* arm (prompt disabled) is the N0 native baseline used here.
