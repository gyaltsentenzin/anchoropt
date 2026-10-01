# The metrics, and the rules that decide what counts

Every number this project reports is one of a small set of quantities, and each one has a rule
attached that says when it may be quoted. This document defines them once so the adapters, the
results documents and the paper can refer to them rather than restate them.

The short version: **`net` is the unit of evidence, `Δpp` is the unit of reporting, and four
preregistered criteria decide whether an arm is installed.** Everything below elaborates one of those.

---

## 1. Vocabulary

| term | meaning |
|---|---|
| **incumbent** | one recorded run of the stock agent on a split, including per-task step logs. Every measurement is paired against it |
| **draw** | one repeat run of the incumbent at identical config. Repeats differ: on AppWorld/minimax, three same-config draws passed 58, 60 and 61 of 90 |
| **arm** | the incumbent plus exactly one candidate intervention, everything else byte-identical |
| **control** | an arm carrying a generic instruction instead of the treatment. It separates "the intervention worked" from "any extra text worked" |
| **firing** | one episode in which the controller's condition was true and it acted. An arm that never fires cannot have caused anything |
| **held-in / held-out** | the split the search ran on, and an independent split used only to validate an arm already promoted on held-in |

A note on the last row, because it is the easiest thing to get wrong: **the second split is a
validation set, not a clean generalization estimate**, whenever it was consulted during selection.
The BFCL dev figures in `README.md` are selected-on and labelled as such.

---

## 2. `net` — the unit of evidence

For one arm against one incumbent draw, over the **paired** cases:

```
net = |gains| - |losses|
```

where `gains` and `losses` are case-ID sets: episodes the arm passed and the incumbent failed, and
vice versa. `net` is a count of tasks, not a percentage, and it is computed only over cases both
sides actually scored. If the two runs did not score the same cases, `denominator_ok` is false and
the comparison is refused rather than reported — an unpaired comparison is not a weaker result, it is
a different quantity.

### `--net-from-draws`: judge an arm on the mean over every draw

`net` against a *single* draw is the dominant bias on small corpora. When arms are scored against the
highest draw, every arm reads systematically low; on episodes where the controller never fired, the
agent behaves identically to the incumbent apart from sampling, so those flips are resampling noise,
and pairing against one draw makes that noise one-directional.

`--net-from-draws` replaces the single-draw net with the **mean net across every incumbent draw**.
Three properties make it a pairing fix rather than a thumb on the scale, and all three are load-bearing:

1. **It moves verdicts in both directions.** On AppWorld/minimax the pagination treatment goes
   +4 → +5.33 *while its control goes −8 → −1.83* and `execution_failed` goes −3 → −1.33.
2. **It is applied to every arm it can be computed for**, never to a chosen one.
3. **It predates the round it affects** (commits `156ec3b`, `ea0e616`). A rule changed after seeing
   which arm it promotes is indistinguishable from p-hacking.

It is **off by default** so earlier rounds stay reproducible. **Anyone quoting a promotion that
depends on it must say so** — the AppWorld minimax cells do.

---

## 3. `Δpp` — the unit of reporting

```
Δpp = 100 × (arm_passes − incumbent_passes) / n_cases
```

**Quote `Δpp`, not relative percent.** Relative percent inflates on a weak baseline, and the effect is
large enough to invert an ordering: on AppWorld, granite's **+3.3 pp** reads as **+11.1%** while
minimax's *larger* **+5.0 pp** reads as **+7.6%**, purely because granite's baseline is 30.0 against
minimax's 65.6. Reporting the relative figure would rank the smaller improvement first.

Three reporting rules travel with it:

- **An unmeasured cell is left empty, never filled with the baseline.** A baseline value sitting in a
  treatment column reads as a measured null, which is a different claim.
- **A measured-then-rejected arm is not a result.** It is reported in the notes, never in the
  treatment column — printing it there would assert a deployment that never happened, and on a
  held-out split it can read as the method *lowering* the score.
- **Pool each arm against the draws it was actually scored against.** Where the two sides pool
  different counts, say so; several AppWorld cells are marked asymmetric for exactly this reason.

---

## 4. The noise floor, and screening before spending

A single 90-task number cannot be distinguished from the 2–5 task spread between same-config baseline
repeats. So:

- **Two passes, not one.** One pass measures the arm and the noise together.
- **Screen before scoring.** `headroom.py` computes the most an arm could possibly gain given where it
  fires. When that ceiling is at or below the noise floor, the arm is **UNRESOLVABLE before it is
  run** — no number of episodes can settle it, so the episodes are not spent.
- **An arm firing below `--min-firings` cannot promote**, and is screened out first. Free: no LLM calls.

A `+1` on a 90-task split is one task. It is inside the noise floor and is not evidence.

---

## 5. The four acceptance criteria

Preregistered, and implemented in
[`anchoropt/learning/acceptance_criteria.py`](../anchoropt/learning/acceptance_criteria.py) over the
same `accept()` seam the manual rounds used. Stated exactly as predefined — not restated, not
tightened:

| # | criterion |
|---|---|
| 1 | **train net > 0** on the paired cases |
| 2 | **no aggregate regression** on the independent dev split. A gain is *not* required: `dev_net >= 0` passes |
| 3 | **attribution + a causally validated mechanism**, from the controller's *own* telemetry |
| 4 | **safety**: the three-counter contract |

### Three verdicts, not two

This is the most important property in the module, because every way this project has previously
fooled itself reduces to **reading silence as success**:

| verdict | permits install? | meaning |
|---|---|---|
| `PASS` | yes | criterion met |
| `PASS_UNINFORMATIVE` | yes | met, but the split could not have detected a regression — the weakness is reported, never used to raise the bar |
| `PENDING_VALIDATION` | **no** | evidence *absent*. Blocks acceptance |
| `FAIL` | no | criterion violated |

`PENDING_VALIDATION` is not a pass. The failure modes it exists to catch are concrete: an arm that
never executed still produces a score, and a positive delta with zero firings is proof of
**non-attribution**; and a host's telemetry allowlist can silently drop undeclared keys, so a
controller may run correctly and leave no evidence — indistinguishable from never having run.

**Absent evidence and weak-but-satisfied evidence are not the same thing, and only the first blocks.**

### Criterion 2 is non-regression, and is not negotiable *downward* — or upward

Requiring a dev *gain* would be stricter than the preregistered rule. Substituting a stricter test
chosen after seeing the data is the same category of error as weakening one, and this project made
that mistake once and corrected it. When the control also scores 0 of N, the split could not have
detected a regression: the verdict becomes `PASS_UNINFORMATIVE`, carrying `informative: False` so the
weakness travels with the result.

### Criterion 4: three counters, asymmetric by design

| counter | rule |
|---|---|
| `clears_added` | categorically forbidden — any value > 0 **FAILS** |
| `information_losing_removes` | a removal not verified present elsewhere in live state **FAILS** |
| `verified_relocations` | reported and bounded, **never** a failure |

A removal whose copy was confirmed in live state before the source was touched preserves the
information. Counting it as destructive would forbid the only safe capacity recovery available;
counting it as free would forbid nothing. Hence three counters rather than one. Verified relocations
count on the **preservation** denominator, not the end-to-end one — the two differ whenever a
relocation preserves its entry but the retried operation then fails for an unrelated reason.

---

## 6. Why no p-value

The BFCL v4 benchmark is **deterministic** at fixed config, so there is no sampling distribution for
a null and a p-value would be a category error rather than a missing statistic. Where a benchmark
*is* stochastic, the repeat-draw spread in §4 is the variance estimate, and paired sign tests
(`scipy.binomtest`) are used on the paired case sets.

Omitting a p without saying why reads as cherry-picking, which is why this section exists and why
`tests/test_progression_integrity.py` fails if `README.md` stops saying it.

---

## 7. The four outcome classes

An external evaluation resolves to exactly one disjoint class, and the order matters:

| class | meaning |
|---|---|
| `TRAIN_IMPROVED_PENDING_VALIDATION` | train net > 0 — criterion 1 of four. A **provisional winner, not an accepted controller** |
| `INCOMPARABLE_INCUMBENT` | the only results were measured against a different incumbent. Establishes nothing; calling it a null would bank a null nobody measured |
| `UNEVALUATED` | nothing was measured at all, whatever else happened |
| `BUDGET_EXHAUSTED` | arms left unmeasured are still open, so this is checked *before* declaring no benefit |
| `NO_BENEFIT` | measured, and it did not help |

`INCOMPARABLE_INCUMBENT` is checked before the null classes deliberately: scoring an absent result as
zero is the same error class as reading silence as success.

---

## Where each of these is enforced

| rule | enforced by |
|---|---|
| published figures match the per-case data | `scripts/verify_progression.py`, `tests/test_progression_integrity.py` |
| the four criteria, three verdicts | `anchoropt/learning/acceptance_criteria.py`, `tests/test_round_ledger.py` |
| paired-denominator refusal | `anchoropt/learning/external_evaluation.py` (`denominator_ok`) |
| outcome classes are disjoint and ordered | `tests/test_external_evaluation.py` |
| screening before scoring | `benchmarks/appworld/headroom.py` |
| an adapter cannot fabricate evidence | `anchoropt/testing/adapter_contract.py` |
