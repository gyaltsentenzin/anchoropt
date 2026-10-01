# T1 — A1: reroute a capacity-blocked write

**Result: 29.04 % → 33.66 %, +4.62 pp.** Installed.

| | |
|---|---|
| incumbent | native agent, no prompt, no anchors — 88/303 = 29.04 % |
| residual | 215 failing queries |
| signal | `capacity/container/no_remaining_capacity` |
| incision point | **post-execution** |
| action | **reroute** — the blocked write goes to archival memory |
| criteria | [`FROZEN.md`](FROZEN.md), written before launch |

## Why this signal

Miner rank 1, and the ranking rule is the lesson of this round: **loci are ordered by linked
downstream loss, not by how often they occur.**

| rank | locus | events | failing queries linked | linked loss / event |
|---|---|---|---|---|
| **1** | `capacity/container/no_remaining_capacity` | 543 | **81 / 215 (38 %)** | 0.149 |
| 2 | `size/item/exceeds_per_item_limit` | 154 | 59 | 0.383 |
| 3 | `existence/identifier/not_found` | **31** | **40** | **1.29** |
| 4 | `format/field/malformed_value` | 60 | 13 | 0.217 |
| 5 | `permission/identifier/duplicate` | 62 | 13 | 0.210 |

Note rank 3: **31 events but 40 linked failures** — it outranks ranks 4 and 5 despite having *half*
their event volume, because nearly every occurrence is fatal. An event-count ranking would have buried
it. (It becomes A2 in the next round.)

This is also why the frozen criteria refuse a **firing floor**: on this corpus frequency and damage are
*inversely* correlated, so a "must fire ≥ N times" rule discards the highest-value anchors first.
Rare-but-costly is exactly what a local anchor is for — a global prompt cannot target it without paying
on every turn.

## Why post-execution + reroute

The failure is observable only *after* the write is attempted: the store returns a capacity error. At
that point the call has run, so `suppress` is structurally unavailable — nothing can be
undone. What remains is to **substitute a destination that works**.

`reroute` was chosen over `reprompt` because the substitution is *mechanical*: the same arguments, a
different tool. There is no inference at runtime, no model involvement, and no chance the model invents
a different recovery. Per-backend instantiation:

| backend | mechanism |
|---|---|
| kv, vector | mechanical argument copy `core_memory_add` → `archival_memory_add` |
| rec_sum | compact *before* the blob saturates (threshold 8000/10000) |

## What was predicted, and what happened

The frozen doc committed to three predictions before launch:

1. **Fires on all three backends** — stated that zero rec_sum firings would make the arm uninformative
   about rec_sum regardless of the aggregate.
2. **Direction positive**, with magnitude explicitly uncertain: a prior measurement of the kv/vector half
   of this locus read +6.27 pp, but *on top of a global prompt that already instructed this behaviour*.
   Here there is no prompt, so the anchor must supply the behaviour itself — the effect could be larger
   (more headroom) or smaller (no scaffolding).
3. **rec_sum holds the largest untouched damage** — 301 events, and 21–22 consecutive lost writes per
   saturated episode.

Outcome: **+4.62 pp**, accepted on the do-no-harm-then-gain rule.

## Coverage is a diagnostic, never a gate

Worth reading in [`FROZEN.md`](FROZEN.md), because two earlier versions of this clause were wrong in
opposite directions. Measured on the completed run, A1's rec_sum branch reached **100 % of affected
turns (24/24)** and **9 % of events (24/277)** — the same anchor, the same run, two denominators an
order of magnitude apart. Any threshold on that is a choice of denominator dressed up as a criterion.

So the liveness test asks only whether the arm tested the thing that was mined. Coverage then describes
the anchor's *reach*, which is an input to designing the successor — not to accepting this one.

## What it changed for the next round

**A1 is the cause of the next two anchors.** It succeeds at its job and moves the failure elsewhere:

- Data now lives in archival, but reads still consult core → `not_found` at read time. **39 of those
  cases are self-inflicted** and become **[T2 / A2](../T2_A2_not_found/)**.
- A1 injects keys the model does not know exist, so the model later re-writes the same fact and collides.
  **All 59 collisions are against A1's own keys** → **[T3 / A3](../T3_A3_duplicate/)**.

Neither residual existed at T1. That is why acceptance closes the iteration: the ranking, the exposure
sets, and every downstream counterfactual are recomputed before the next candidate is chosen.
