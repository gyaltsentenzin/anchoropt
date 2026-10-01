# Attributor prompts: frozen, with the A/B/C result

Three variants, all frozen as of 2026-09-19. **No further prompt revision before ICLR.**

| | file | status |
|---|---|---|
| **P0** | `attributor_P0.txt` | the original, as-run for A9 / cycle-2 / WRITE1 |
| **P1** | `attributor_P1.txt` | phase-aware outcome, evidence-grounded, uncertainty permitted |
| **P2** | `attributor_P2.txt` | P1 + explicit separation of OBSERVED failure from the consequential-decision HYPOTHESIS |

`SEMANTIC_PROPOSER_SYSTEM` is unchanged and **inactive** — it has no call site in the current loop.
It was not revised, because revising an inactive prompt to explain failures in the active algorithm
would be misleading.

## Replayed on the SAME frozen WRITE1 residuals (24 storage-phase cases, c2inc)

| metric | P0 | P1 | P2 |
|---|---|---|---|
| usable diagnoses | 24 | 24 | 24 |
| schema rejections | 0 | 0 | 0 |
| families (prose-keyed) | 6 | 24 | 24 |
| families (pooled by observable) | 5 | 9 | **6** |
| top support (prose) | 4 | 1 | 1 |
| **top support (pooled)** | 8 | 8 | **14** |
| **uncertainty admitted** | 0 | 0 | **2** |
| numbers quoted as evidence | 244 | **270** | 207 |
| search state / moves_earlier / expanded / candidates | NO_BENEFIT / 1 / 26 / 196 | identical | identical |

## What this does and does not show

**P2 is the best of the three on grouping and honesty.** It reaches the highest pooled support (14 vs
8), admits uncertainty where the evidence does not localize a decision (2 cases — the only variant to
do so), and its top residual is keyed on **three** capacity observables rather than two, adding
`append_would_exceed_cap`. Its top key names a mechanism rather than a category: *"applying the wrong
remedy to a rejected write — responding to a per-entry length violation by deleting stored entries and
resubmitting."*

**P1 is not worse than P0** — a claim made and retracted. All 24 diagnoses usable, 0 rejections,
identical search outcome, and it quoted the MOST numeric evidence (270). What it changed was
FRAGMENTATION under prose-keyed grouping (6 → 24 families), which observable pooling then absorbed.
That is a property of our grouping method, not of attribution quality.

**No variant changed anything downstream.** All three produce the identical search: `NO_BENEFIT`,
`moves_earlier=1`, 26 signals expanded, 196 candidates, same distribution over (boundary, action).

That is the finding worth carrying into the paper: on this residual, **prompt variation is not the
binding constraint — the upstream observable set is.** The commitment gate supplies 5 fields, of which
synthesis can use 1 (`step_index`), so no write-side controller is expressible there regardless of how
the failure is described. The next step is observable-field expansion, not prompting.

## Reproduce

```
python scripts/replay_prompts.py --incumbent /tmp/c2/c2inc --phase prereq --cell vector \
    --prompts prompts/attributor_P0.txt prompts/attributor_P2.txt --out rounds/PROMPT_AB2 --live
```

Artifacts: `rounds/PROMPT_AB/` (P0 vs P1), `rounds/PROMPT_AB2/` (P0 vs P2). Each keeps its raw
per-variant diagnoses, so the comparison is re-derivable without new API calls.
