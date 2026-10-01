# A6 — DEFERRED: a real signal, and a remedy that is not ready

**A6 is deferred — not in the accepted stack, and not closed.** The remedy fails the governing
acceptance rule on two criteria independently: **dev −1.19 pp**, and its own target backend is
**−2.50 pp**. The *signal* it was built for is real and still unaddressed, which is why this is a
deferral rather than a rejection.

Full record: [`DEFERRED.md`](DEFERRED.md).

| | |
|---|---|
| locus | `size/item/exceeds_per_item_limit` — **miner rank 1** on the post-A5 residual |
| remedy | relocate an over-long entry to a larger-cap container **+** raw-exact dedup |
| train | +1.65 pp (14 g / 9 l) |
| **dev** | **−1.19 pp** (8 g / 9 l, 17 discordant) |
| **on-target (vector)** | **−2.50 pp** — engages as designed and makes things worse |
| status | **DEFERRED** — remedy not ready, signal still open. Not refuted |
| ledger | [`ledger/n0t6_locus.json`](ledger/n0t6_locus.json), frozen, fp `869316d8730866c4` |
| mechanism code | [`../../anchoropt/mechanisms/entry_length_reroute.py`](../../anchoropt/mechanisms/entry_length_reroute.py) |

## Why this directory exists

Because the **signal is real** and the **remedy was mis-specified**, and those are separable. Deleting
the anchor would delete the measurement too, and the measurement is the most transferable result in
the project:

> On a saturated store, a capacity-shaped repair cannot add information. It can only reorder which
> entries win the fixed slots — a **selection** intervention wearing a capacity costume. A6 fires 44×
> on dev while **A1's own dispatches collapse 22 → 6**, and the final store shows 54 facts out / 53 in
> at constant size: the same topics in different phrasings.

That is why [`displacement_check`](../../anchoropt/learning/exposure.py) ships as code, and why
[`../../docs/SATURATION_AND_SELECTION.md`](../../docs/SATURATION_AND_SELECTION.md) exists. A6 paid for
both.

## What is still on the table

`entry_too_long` fires **55×** on train and **102×** on dev; vector stores sit at **62–85%** capacity;
**103** byte-identical duplicate writes remain; the paraphrase tier (**15.8%** of allowed writes ≥ 0.90
similar) is untouched. **With A6 removed those entries are rejected and lost again** — the 1.65 pp
train cost, priced in knowingly.

Two things to carry forward if it is reopened: **raw byte-exact dedup beat normalized on every axis**
(dev −8.33 → −1.19 pp, facts retained 56 vs 17), and **suppression is not monotonically good** —
informative variants matter. The standing instruction is **not** to simply re-enable the flags; four
reopening conditions are listed in [`DEFERRED.md`](DEFERRED.md).

## It cost A7 nothing

A6 never fires on `rec_sum` — 0 firings, and that shard is **bit-identical** with and without it. So
**A7 would have been mined identically had A6 never existed.** Not a stepping stone.

See **[T7 — A7](../T7_A7_blob_overflow/)** for the anchor that followed, and
[`../../docs/ACCEPTANCE_RULE.md`](../../docs/ACCEPTANCE_RULE.md) for the rule A6 failed.
