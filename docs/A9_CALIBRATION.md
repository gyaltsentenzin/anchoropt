# A9 calibration: does the accepted controller still work on the current substrate?

Calibration-only. **Not part of autonomous discovery**, and isolated from the proposer/runtime
vocabulary by test (`test_policy_family.py::test_no_a9_identifier_appears_in_the_runtime_source`).

BV · Granite-4.1-8B · vector/train · n=89 · LSF 1761195 (ctl) / 1761196 (arm) · both `rc=0`

## The exact historical (ℓ, φ, μ, θ) replayed

| element | value |
|---|---|
| **ℓ** | `POST_EXECUTION` |
| **φ** | `core_memory_retrieve` whose best `similarity_score` < θ |
| **μ** | `REROUTE` (historical label preserved; its own evidence record calls the behaviour *augment-the-observation* — it substitutes no call) |
| **θ** | **0.30** (`_XCM_THRESHOLD`), tie-break toward core, vector-only |
| policy | `experiments/n0a9/n0a9_xcm.json` vs control `experiments/n0dcrf/n0dcrf_frozen.json` — differing **only** in `on_low_similarity_cross_container` |
| gates | policy gate **and** env `ANCHOROPT_XCM=1`; asserted in-log (`XCM=1` arm, `unset` control, `A7C=1` both) |

Runners are the **unmodified historical scripts**; only output paths were redirected so the
2026-08-26 artifacts survive as the reference.

## Result: A9 remains a valid known-good controller

**DENOMINATOR INTEGRITY: OK** — 89/89 both arms, two scorer paths, zero disagreements.

| | control (A1–A8) | arm (+A9) | Δ | gains/losses | firings |
|---|---|---|---|---|---|
| **today** | 39/89 = 43.82% | **55/89 = 61.80%** | **+17.98 pp** | 17 / 1 | 49 |
| published 2026-08-26 | 39/89 | 54/89 | +16.90 pp | 15 / 0 | 55 |

The controller is confirmed on the current substrate, slightly **stronger** than published.

## Why the arm differs and the control does not

| | old → new flips |
|---|---|
| control | **0** — byte-stable |
| arm | **5** (`vector_5, 8, 15, 20, 22 -customer`), firings 55 → 49 |

The control reproducing its 39/89 exactly rules out substrate drift, model change, or scoring change.
The variation is confined to the arm, which is expected and documented: A9 is the one anchor whose
repair **calls a model at inference time**, so its dispatch reorders the trajectory. `ANCHORS.md`
already records A9 as a source of non-determinism (1 of 26 compactions decoded differently for A7,
the same class of effect).

So: +17.98 pp vs +16.90 pp is one sample of a noisy arm against a stable control, **not** an
improvement over the published figure. The honest statement is that A9 reproduces within run-to-run
variation, and the single loss is inside that variation rather than a new regression.

## Bearing on residual boosting

A9 was fitted to the residual A1–A8 left, and it still engages (49 firings) and gains (+16 net) on
that same incumbent — so a learned anchor is not fragile to being re-measured. This says nothing yet
about whether A9 would help on the **native** incumbent, which is the interesting composition
question: R1's control is 19/89 native versus 39/89 here, and A9's trigger may be reached far less
often when the earlier anchors are absent. That is a separate arm and has not been run.

## Contrast with R1/R2, which is the useful part

Same signal family, same locus, opposite outcome:

| | ℓ | φ | execution | measured |
|---|---|---|---|---|
| **A9** | post_execution | low similarity | *supplies* archival evidence, merged | **+17.98 pp** |
| **R1/R2** | post_execution | low similarity | *instructs the model to distrust its result* | **−1 to −5 net at every θ** |

The condition is worth acting on. What R2 rejected was one **execution** for it, not the signal — and
that is precisely the distinction the action-contract work exists to make measurable.
