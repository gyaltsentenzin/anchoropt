# BFCL integration — the supported path, verified end to end

What was run, on which artifacts, and what it found. Every number here was read from a real run; none is
synthetic.

---

## The artifacts

| | |
|---|---|
| frozen incumbent | `/tmp/c2/c2inc` — 56/89 vector, 89 query + 27 prereq trajectories |
| measured arm | `/tmp/c2/w2b` — 58/89 vector, same shape |
| cached attribution | `rounds/WRITE2/diagnoses.json` — **24 real diagnoses** |
| archived round | `rounds/WRITE2/cycle2.json` — 245 candidates, θ=272 chosen by the old heuristic |

> `/tmp` is ephemeral. The artifact-dependent tests in `tests/test_bfcl_integration_e2e.py` skip when
> `/tmp/c2` is gone and say so; the four defect regressions are source-level and **never** skip.

No GPU run was launched. The paired measurement comes from two runs that already existed.

---

## The path, stage by stage

```bash
# Stages 1-3 — mine, localize, emit. Names no winner.
python scripts/self_evolve_cycle2.py \
    --incumbent /tmp/c2/c2inc --out OUT --cell vector --phase prereq

# Stages 4-6 — feed the real measurement back; select on J_train.
python scripts/self_evolve_cycle2.py \
    --incumbent /tmp/c2/c2inc --out OUT2 --cell vector --phase prereq \
    --results results_real.json
```

| stage | observed |
|---|---|
| 1. mine the residual | 24 diagnoses → **5 ranked residual problems**, R1 support 8 |
| 2. localize + build | both boundaries visited, **`moves_earlier=1`**; Φ expanded to **63 signals**; 219 arms |
| 3. emit manifest | `arm_manifest.json`, 219 arms, `state=UNEVALUATED`, **no `selected_arm`**, no `next_arm` |
| 4. external results | `--results` with the **real** `c2inc` vs `w2b` comparison |
| 5. select | `IMPROVED`, net **+2**, via `AnchorPolicyOpt.optimize` under `train_objective` |
| 6. correspondence | the selected arm is in the manifest **and** has an executable controller spec |

### The real paired measurement

Read from the two runs' own `eval_train_results.json`, firings from `w2b/traj/prereq` sidecars:

```
incumbent 56/89   arm 58/89   +9 / -7   net +2   +2.25 pp
denominator identical: True
firings: 15 episodes, 155 interventions executed
```

These match the archived WRITE2 round exactly.

### The round's own discovery is rediscovered

```json
{
  "arm_label": "post_generation_pre_exec/proposed_payload_chars_gt_272p0/suppress:cancel_proposed",
  "signal": "proposed_payload_chars_gt_272p0",
  "action": "suppress", "variant": "cancel_proposed",
  "eta": {"suppressed_operation": "the proposed operation this signal implicates"},
  "fires_on_states": 127, "total_states": 711
}
```

`fires_on_states: 127/711` is the archived round's own figure. Its emitted spec
(`proposed_payload_chars > 272.0`) fires on **125/543** real pre-dispatch states — so it is *executable*,
not merely well-formed. Note the eta: no `preservation`.

---

## What running it found

Four defects. None was reachable by a synthetic fixture; all four are now regression-tested.

### Driver

1. **`NameError: 'fires' is not defined`.** A vestigial `next_arm` block still read the variable the
   firing-rate heuristic had owned. The manifest wrote *first*, so a partial run looked like progress
   before the crash. The field is removed outright — it named one arm as "the next one", the heuristic's
   last vestige.
2. **Every manifest arm reported `fires_on_states: 0`.** `fires_by_label` keyed on `instantiated.label`
   (the last path segment) while the manifest keys on `arm.label` (the full path), so the lookup matched
   nothing. Visibly false — the fireability filter had already excluded every all-zero arm — and it
   survived because nothing read the field back. Now **1–627** across 219 arms.
3. **The propose phase's `UNEVALUATED` was overwritten** by `AWAITING_EVALUATION` on the way out, and a
   `--results` round lost its outcome class entirely.

### Adapter

4. **`ground_suppress` still emitted `preservation`.** The commitment-gate executor does not consume it,
   so the new inert-eta rule correctly refused **all 39** suppress candidates there — including the
   controller WRITE2 discovered. A remove-outright suppression preserves nothing and must not claim to;
   nothing ever enforced the clause. Gate candidates: **0 → 39**.

Defect 4 is the one to note: the release's *new rule was right* and the *adapter had not been migrated*.
The rejection was correct and, without this run, would have silently removed the search's best cell.

---

## Misclassification guards, on real data

| input | state | `is_negative_result` |
|---|---|---|
| no results recorded | `UNEVALUATED` | `False` |
| denominator mismatch | `UNEVALUATED` | `False` |
| real result, `--eval-budget 0` | `UNEVALUATED` | `False` |
| real result **inverted** (net −2), 218 arms open | `BUDGET_EXHAUSTED` | `False` |

None is ever `NO_BENEFIT`. The last row is the sharpest: a genuinely measured *loss* still does not
settle a round that has arms left unmeasured, and nothing is selected.

`select_on_measurement` was confirmed to call `AnchorPolicyOpt.optimize` (by intercepting the method)
with `train_objective` as the objective.

---

## Backward compatibility

Five archived specs carry `preservation`. They are **not** rewritten — they record what ran.

The boundary that makes this safe, now pinned: the eta contract governs **arm construction**, not
**controller installation**. `preservation` is inert at install time. Each archived spec still installs at
its locus, and the WRITE2 predicate still fires as recorded (`>272` → fires at 400, not at 50). Every
frozen result remains reproducible.

---

## Reproducing this

```bash
mkdir -p OUT && cp rounds/WRITE2/diagnoses.json OUT/diagnoses.json
python scripts/self_evolve_cycle2.py --incumbent /tmp/c2/c2inc --out OUT \
       --cell vector --phase prereq
python -m pytest -q tests/test_bfcl_integration_e2e.py     # 21 tests
```

With `/tmp/c2` present, all 21 pass. Without it, the artifact-dependent ones skip and the source-level
defect regressions still run.
