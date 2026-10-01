# Results as reported in the paper

This page transcribes the tables and the algorithm from the paper this repo accompanies. They are
**reported figures**, not a claim that every cell is re-derivable from this repo's shipped data with no
model and no GPU — that stricter bar is the one described in
[`BFCL_PROGRESSION.md`](BFCL_PROGRESSION.md) and enforced by `scripts/verify_progression.py` for the
granite-4.1-8B BFCL v4 progression specifically. Where a row below overlaps with that reproducible line,
the two are consistent; where it doesn't (other backbones, AppWorld, τ²-bench, the GEPA-complementarity
study), treat this page as the citable source and the paper as the full account, including the
appendix provenance each caption points to.

$H_0$ is the native harness with no AnchorOpt controller installed. GEPA and Self-Harness are the two
baselines AnchorOpt is compared against — GEPA as a prompt-optimization baseline, Self-Harness as an
earlier internally-built harness.

---

## Algorithm: residual-driven optimization

The loop described in prose in [`THE_LOOP.md`](THE_LOOP.md) and in the README, formalized:

```
Algorithm 1: AnchorOpt — residual-driven optimization

Require: Native harness H0, initial signals Φ0, budget B
 1:  P ← ∅, Φ ← Φ0
 2:  T ← Execute(H0)
 3:  while budget remains do
 4:      H ← H0 ⊕ P
 5:      F ← DiagnoseResiduals(T)
 6:      if Stop(F) then
 7:          return P
 8:      L_c ← ConsequentialBoundaries(F, H)
 9:      A_c ← SearchFeasiblePolicies(L_c, Φ, H)        # Fix Φ
10:      A* ← EvaluateAndSelect(A_c, H)
11:      if A* ≠ ∅ then
12:          P ← P ⊕ A*
13:          T ← Execute(H0 ⊕ P)
14:          continue                                    # Re-mine residuals
15:      if budget exhausted then
16:          return P
17:      Φ' ← ExpandSignals(F, L_c, Φ)                   # Fix P
18:      if Φ' = Φ then
19:          return P
20:      Φ ← Φ'
21: return P
```

Reading guide, mapped onto the six-step loop and the shipped code:

| line | step | code |
|---|---|---|
| 5 | mine + attribute | `anchoropt/attribution/` |
| 8 | locate | `anchoropt/anchor.py`'s incision points |
| 9 | prune to feasible local policies, signals held fixed | `anchoropt/learning/` candidate search |
| 10–13 | measure a paired counterfactual arm; on acceptance, persist and re-mine | `anchoropt/mechanisms/`, `anchoropt/learning/round_runner.py` |
| 17 | signals expand only once the current signal vocabulary is measurably exhausted, policy held fixed | `anchoropt/learning/` signal expansion (see `THE_LOOP.md`'s "vocabulary gets exhausted" section for the empirical trigger, T4) |

The two expansions (line 9 fixes Φ; line 17 fixes P) never happen in the same iteration — that
block-coordinate structure is what T4 in the BFCL progression demonstrates concretely: a round where no
local policy survives search and the signal vocabulary expands instead of a controller being installed.

---

## BFCL v4 Agent Memory — strict task success (%)

Three backbones beyond the granite-4.1-8B progression this repo reproduces end to end
(see [`BFCL_PROGRESSION.md`](BFCL_PROGRESSION.md)). Parentheses show change relative to each method's
own native reference. Full provenance: paper Appendix "BFCL table provenance."

| Model | Split | $H_0$ | GEPA | Self-Harness | AnchorOpt |
|---|---|---:|---:|---:|---:|
| MiniMax M2.5 | train | 23.7% | 34.0% (+10.3pp) | $H_0$ retained | 26.0% (+2.2pp) |
| MiniMax M2.5 | held-out | 42.7% | 58.7% (+16.0pp) | $H_0$ retained | 46.7% (+4.0pp) |
| Qwen3.6-35B | train | 39.8% | 39.3% (−0.5pp) | $H_0$ retained | $H_0$ retained |
| Qwen3.6-35B | held-out | 19.1% | 35.7% (+16.7pp) | $H_0$ retained | $H_0$ retained |
| Opus → Granite 4.1-8B | train | 30.0% | 31.0% (+1.0pp) | 31.9% (+1.9pp) | 37.6% (+7.6pp) |
| Opus → Granite 4.1-8B | held-out | 16.7% | 19.1% (+2.4pp) | 19.8% (+3.1pp) | 21.4% (+4.8pp) |

`$H_0$ retained` means the method's search measured candidates and did not find one that beat the
incumbent on that split — a reported null, not an untried cell.

The last row (Opus → Granite 4.1-8B) is this repo's reproducible progression at its T9 endpoint, read
through a slightly different lens than `BFCL_PROGRESSION.md`'s table — see that doc for the per-round
walk and exact `rounds/anchors.py` provenance behind 37.6% / 21.4%.

## AppWorld — task-goal completion (TGC, %)

| Model | Split | $H_0$ | GEPA | Self-Harness | AnchorOpt |
|---|---|---:|---:|---:|---:|
| MiniMax M2.5 | train | 65.6% | 71.1% (+5.5pp) | 68.9% (+3.3pp) | 70.6% (+5.0pp) |
| MiniMax M2.5 | held-out | 54.4% / 55.2% | 62.8% (+8.3pp) | 61.7% (+7.2pp) | 60.6% (+5.4pp) |
| Qwen3.6-35B | train | 75.6% | 71.7% (−3.9pp) | $H_0$ retained | $H_0$ retained |
| Qwen3.6-35B | held-out | 58.9% | 51.1% (−7.8pp) | $H_0$ retained | $H_0$ retained |
| Granite 4.1-30B | train | 30.0% | $H_0$ retained | $H_0$ retained | 33.3% (+3.3pp) |
| Granite 4.1-30B | held-out | 26.7% | $H_0$ retained | $H_0$ retained | 30.0% (+3.3pp) |

MiniMax and Granite rows are consistent with the README's AppWorld table; this is the fuller version
including Qwen3.6 (no anchor promoted on that backbone) and the GEPA/Self-Harness comparison columns.
Full provenance: paper Appendix "AppWorld details."

## tau2-bench — repeated-execution reliability

Granite 4.1-30B, airline tasks (n=20). `pass^k` is success in all `k` executions of the same task — a
stricter reliability measure than mean accuracy.

| Metric | $H_0$ | AnchorOpt | Gain (pp) |
|---|---:|---:|---:|
| pass¹ | 0.188 | **0.350** | +16.2 |
| pass⁴ | 0.000 | **0.350** | +35.0 |

The `pass⁴ = 0` baseline is notable on its own: the native harness never reproduced a success across
four runs of the same task, which is the harness-reliability failure mode this metric is designed to
expose. See [`benchmarks/tau2/README.md`](../benchmarks/tau2/README.md) and
[`docs/TAU2_ADAPTER.md`](TAU2_ADAPTER.md) for what this adapter measures and the capability audit behind
it; the comparison against GEPA and Self-Harness on the full matrix is in the paper's tau2-frameworks
appendix.

## Interaction with a frozen GEPA-optimized prompt

A separate study: does a *local* AnchorOpt controller add anything on top of a *global* GEPA-optimized
prompt, rather than replacing it? MiniMax, BFCL. The read-side comparison is store-fixed; the write-side
results are from two separate fresh-build comparisons. The write-side controller is exploratory and was
**not** included in the final accepted portfolio — this table reports an interaction effect, not a
frozen held-out portfolio gain.

| Local controller | Targeted failure | Result on top of GEPA |
|---|---|---|
| Vector read-side | Insufficient retrieval | No net gain: 29/104 → 29/104 |
| Key-value capacity | Rejected memory writes | +9 successes on the targeted backend |
| Vector write-side | Relevant facts never stored | 29/104 → 47/104, and 25/104 → 48/104 (two fresh builds) |

This is the empirical argument behind [`GENERALIZABILITY.md`](GENERALIZABILITY.md)'s comparison to
global prompting: a global prompt and a local decision-boundary controller are not mutually exclusive
interventions, and composing them can help on some backends and do nothing on others, depending on
which failure mode the backend actually has.
