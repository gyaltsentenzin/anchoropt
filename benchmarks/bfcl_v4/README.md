# BFCL v4 Memory adapter

The original AnchorOpt integration — where the A1–A9 progression in the main
[`README.md`](../../README.md) was measured. Berkeley Function-Calling Leaderboard v4's agent
memory task: three backends (key-value, vector, recursive summarization), each exercised through
multi-turn tool use.

## What's here

| path | what it is |
|---|---|
| `adapter.py`, `bfcl_runtime.py` | the adapter: `U_H(ℓ)`, signal dispatch, the capability bindings |
| `bfcl_signals.py` | Φ_BFCL — the declared signals and where each is observable |
| `bfcl_capabilities.py`, `bfcl_constraints.py`, `bfcl_primitives.py` | the typed alphabet and the executor capabilities per boundary |
| `run.py` | the entry point: paired control-vs-anchors runs, dry-run mode, per-backend sharding |
| `run_memory_train.py`, `run_memory_eval.py` | lower-level training/eval drivers `run.py` wraps |
| `harness/`, `evaluator/` | **vendored** — a modified fork of the official BFCL v4 harness; see [`VENDORED.md`](VENDORED.md) for exactly what changed and why |
| `data/` | the memory-task corpus (trimmed to what the memory task actually reads — see `VENDORED.md`) |

## Reproducing the result

The main README's one-command check (`python scripts/verify_progression.py`) recomputes every
published accuracy from the per-case results already shipped in `rounds/` — no GPU, no model
needed for that. To actually re-run an arm end to end against a served model, see
[`REPRODUCE.md`](../../REPRODUCE.md) Tier 3, which covers setup, sharding by backend, and
submitting shards concurrently.

Quick sanity check before spending GPU time — this inspects the six concurrent commands a real run
needs without launching anything:

```bash
python benchmarks/bfcl_v4/run.py --compare --shard all --dry-run
```

## Dependencies

Core AnchorOpt has no BFCL-specific dependencies. Running this adapter needs:

```bash
pip install -e ".[bfcl]"          # key-value and recursive-summarization backends
pip install -e ".[bfcl-vector]"   # adds the embedding stack for the vector backend
```

See [`VENDORED.md`](VENDORED.md) for why these are much lighter than the vendored harness's own
`pyproject.toml` would suggest (32 declared dependencies; the memory task touches 4).

## Before you read the vendored harness as a spec

`harness/` and `evaluator/` are a **modified fork**, not stock BFCL v4 — the published numbers
depend on fixes made to it (documented in [`VENDORED.md`](VENDORED.md)). If you're porting this
adapter's approach to a different benchmark, read [`../../docs/ADAPTER_GUIDE.md`](../../docs/ADAPTER_GUIDE.md)
and [`adapters/adapter_template.py`](../../adapters/adapter_template.py) instead — this directory
carries a lot of BFCL-specific history that isn't the generic interface.

## Tests

```bash
python -m pytest tests/test_bfcl_*.py -q
```

Most run with no GPU and no served model; a few integration tests need a live run under
`/tmp/c2` and skip cleanly when it's absent.
