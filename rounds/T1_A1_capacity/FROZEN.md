# N0-A1 acceptance criteria — FROZEN BEFORE LAUNCH

Prompt-free AnchorOpt line. Written before job submission; not editable after results land.

## Line definition

| | |
|---|---|
| **N0 native baseline** | `g7_a0_control` template — NO preamble, NO gates, NO reroute |
| baseline corpus | `results/g8_train/prompt/control`, balanced split, 84 prereq + 303 query |
| **baseline accuracy** | **88/303 = 29.04 %** |
| residual | 215 failing queries |

This line has **no global memory prompt of any kind**. `on_memory_preamble` is empty in every
template. The A0/global-prompt experiment is preserved untouched as a separate strong-baseline
comparison and is NOT part of this line.

## The candidate

Semantic miner rank 1 over the N0 residual, frozen artifact `ledgers/n0_locus.json`
fp `62295bdc0ce0c2ec`, model `claude-haiku-4-5`, prompt v2:

| rank | locus | events | linked | backends |
|---|---|---|---|---|
| **1** | `capacity/container/no_remaining_capacity` | 543 | **81 / 215 (38 %)** | kv 202, rec_sum 301, vector 40 |
| 2 | `size/item/exceeds_per_item_limit` | 154 | 59 | kv 33, vector 121 |
| 3 | `existence/identifier/not_found` | 31 | 40 | kv 31 |
| 4 | `format/field/malformed_value` | 60 | 13 | kv 60 |
| 5 | `permission/identifier/duplicate` | 62 | 13 | kv 62 |

Ranking is by **linked downstream loss**, never local error volume — rank 3 has fewer events
than ranks 4/5 yet outranks them.

## Instantiation (generic adapter interface only)

| backend | gate | family | mechanism |
|---|---|---|---|
| kv, vector | `on_core_full_rerouted` | reroute | mechanical arg copy `core_memory_add` → `archival_memory_add` |
| rec_sum | `on_blob_pressure` | preempt | compact BEFORE the blob saturates (threshold 8000/10000) |

`on_domain_error_core_full` + `enable_reroute` are required to open the G3 block the reroute
executes inside; enabling `on_core_full_rerouted` alone fires nothing.

## Predictions, stated before the run

1. The anchor fires on all three backends. If rec_sum shows 0 firings the arm is uninformative
   about rec_sum regardless of the aggregate.
2. Direction is positive. The prior A0-line measurement of the kv/vector half of this locus was
   +6.27 pp, but that was **on top of a global prompt that already instructed this behaviour**.
   Here there is no prompt, so the anchor must supply the behaviour itself — the effect could be
   LARGER (more headroom) or smaller (no prompt scaffolding).
3. rec_sum is where the largest untouched damage is: 301 events, and the feasibility study
   measured **21–22 consecutive lost writes** per saturated episode.

## Acceptance rule

**DO NO HARM FIRST, then gain.** Harm is checked before gain, per the standing rule.

- **ACCEPT** if paired gains > losses on the 303 scored cases and no backend degrades by more
  than 2 cases against the native baseline.
- **REJECT** if losses ≥ gains.
- **INCONCLUSIVE** only if the anchor is not demonstrably **live** — it never fired, or it fired
  somewhere other than the mined context. This is a *liveness* test, not a coverage threshold.

  **Coverage is reported as a DIAGNOSTIC, never as a gate.** Two earlier versions of this clause were
  wrong in opposite directions and both are corrected here:

  1. A **firing floor** ("< 10 firings ⇒ inconclusive") applies a FREQUENCY filter to a DAMAGE
     question, and on this corpus the two are inversely correlated — the rarest locus
     (`existence/identifier/not_found`, 31 events) does the MOST damage per occurrence (1.29 linked
     failures each) while the most common (`capacity/…`, 543 events) does the least (0.149). A floor
     discards the highest-value anchors first, and rare-but-costly is exactly what a local anchor is
     for: a global prompt cannot target it without paying on every turn.

  2. A **coverage fraction** ("≥ half its mined exposure") is not well defined. Measured on the
     completed run, the rec_sum branch reached **100 % of affected turns (24/24)** and **9 % of
     events (24/277)** — the same anchor, the same run, two denominators an order of magnitude
     apart. Any threshold on that is a choice of denominator dressed up as a criterion.

  What matters for a verdict is only that the arm tested the thing that was mined. Coverage numbers
  then describe the anchor's *reach*, which is an input to designing the successor, not to accepting
  this one.

Significance is reported, not required: this corpus cannot reach p<0.05 on a ~5 pp effect
(see `anchoropt-corpus-splits`). Directional consistency is the bar.

## On acceptance

Freeze this anchor as **N0-A1**, then re-mine the frozen incumbent for the next locus. Do NOT
carry the current rank 2–5 ordering forward: acceptance closes the iteration and the residual,
exposure sets and ranking are all recomputed.
