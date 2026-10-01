# CCTU adapter

AnchorOpt's integration with CCTU (Complex Constraint Tool Use): multi-turn tool-use tasks graded
by an executable constraint validator rather than a single pass/fail. CCTU itself — the benchmark,
its leaderboard, upstream's citation — is in [`README.md`](README.md); this file is the adapter.

## What's here

| path | what it is |
|---|---|
| `cctu_adapter.py` | the full runtime surface: `U_H(ℓ)`, declared signals, grounders for reprompt/suppress/substitute |
| `cctu_capabilities.py` | the 33-field typed alphabet |
| `cctu_signals.py` | Φ_CCTU — the declared signals and the three classifiers behind them |
| `cctu_state.py` | live constraint state, built from upstream's own constraint handlers |
| `cctu_middleware.py` | the three incision-point hooks, `ControllerSpec`, the firing trace |
| `cctu_apply.py` | directive → an edit of the turn, with the mechanisms that implement each action |
| `cctu_replay.py` | replay a frozen transcript through the middleware with no model |
| `response_generator.py` | the entry point: wires the three hooks, reads `--controllers`/`--anchor-trace`/`--replay-from` |
| `cctu_run_arms.py` | sweeps a set of candidate controllers against the model |
| `verify_plumbing.py` | 9 structural checks against real episodes — run this before anything live |
| `evaluation.py`, `analyze_residual.py`, `check_pairing.py` | scoring, residual mining, and paired-comparison integrity checks |

## One anchor is deployed

A2 (round 5) is accepted and deployed — a `reprompt` controller at the commitment gate that
reduces response-length violations with no regression on redundant tool calls. What it does, why
it's split into two branches, and the exact command to reproduce it:
[`../../docs/CCTU_ANCHORS.md`](../../docs/CCTU_ANCHORS.md). Full external reproduction, three tiers
from no-GPU re-scoring to a live re-run: [`../../docs/CCTU_REPRODUCE.md`](../../docs/CCTU_REPRODUCE.md).

## Setup

```bash
cd benchmarks/cctu
pip install -r requirements.txt
```

Download [the CCTU corpus](https://huggingface.co/datasets/Junjie-Ye/CCTU) and put
`input_data.jsonl` under `data/`. See [`ANCHOROPT.md`](ANCHOROPT.md)'s status table for the measured
corpus facts (residual size, variance floor, headroom) before running anything live.

## Verify the plumbing

```bash
python scripts/check_adapter.py benchmarks.cctu.cctu_adapter   # 7 structural checks, synthetic trajectory, no corpus needed
PYTHONPATH=benchmarks/cctu python benchmarks/cctu/verify_plumbing.py   # 9 checks against real episodes, needs the corpus
```

The structural check currently warns on check 3 ("backward search can move"): the synthetic smoke
trajectory derives only one boundary, so it cannot demonstrate the search moving earlier. That is a
property of the synthetic probe, not of the real adapter — real trajectories derive multiple
boundaries, which is what the deployed anchor's own round record demonstrates.

## Tests

```bash
cd benchmarks/cctu && pip install -r requirements.txt   # func_timeout is load-bearing for these
cd ../.. && python -m pytest tests/test_cctu_*.py -q
```

142 tests across `test_cctu_adapter.py`, `test_cctu_middleware.py`, `test_cctu_cycle1_thinness.py`,
and `test_cctu_screens.py`. None need a GPU or a served model; a handful skip without the real
corpus under `data/`.

## Honest gap

**`ANCHOROPT.md`'s header is stale.** It says "no anchor yet"; A2 has since been accepted and
deployed (see `CCTU_ANCHORS.md`). Read its status table for what each file does, not for whether
an anchor exists.
