# T6 — A5: free a slot without losing a fact

**Result: train 42.24 % → 43.56 %, +1.32 pp · dev 26.19 % → 34.52 %, +8.33 pp** — the largest dev step
in the progression. Accepted; the mechanism is **causally validated** 30/30 on dev.

| | |
|---|---|
| incumbent | N0 + A1–A4 — 128/303 = 42.24 % |
| signal | `capacity/container/no_remaining_slots` |
| incision point | **post-execution** |
| action | **reroute** — evict one provable duplicate, then retry the original write verbatim |
| criteria | [`OVERRIDE.md`](OVERRIDE.md) — pre-registered, **and one clause failed** |
| dev mechanism | 30 evictions, blocked write landed **30/30**, lossless invariant **30/30**, 24 correct declines |
| mechanism | [`../../anchoropt/mechanisms/lossless_eviction.py`](../../anchoropt/mechanisms/lossless_eviction.py) |

## Why this signal

The third residual **A1 created**, and the deepest one. A1 answers "this container cannot hold the
write" by relocating the payload to archival. A2 and A3 cleaned up the read-side and duplicate-write
consequences of that. A5 is what happens when A1's *destination* runs out: archival hits its slot cap,
and A1's own repair now has nowhere to put anything.

> A1 rescues the failed write. A5 rescues A1's rescue.

## The mechanism, and the one condition that justifies it

Evicting from a store whose contents are the task's only memory is destructive, and this project has
measured how badly it goes when the victim choice is delegated or heuristic. A5 fires **only** where
eviction is provably lossless:

    the container holds the SAME value twice  ->  removing one copy destroys no information

That is the whole rule. It is **not** "evict the least useful entry" — no utility model exists, and
inventing one at a single decision point is the selection problem
[`../../docs/SATURATION_AND_SELECTION.md`](../../docs/SATURATION_AND_SELECTION.md) shows cannot be
solved there.

Four steps, each load-bearing:

| step | why it is not an implementation detail |
|---|---|
| **1. read the LIVE container** | the first version read its own episode-local write log, saw a median of **4** entries where the store held **~50**, never found a victim, and the arm was **void** |
| **2. deterministic victim** | last-written copy of a value with ≥ 2 copies; preserves the earliest, which later queries are likelier to reference. No model consulted about what to destroy. |
| **3. VERIFY a copy remains** | checked against the store *after* the removal. Reasoning that a copy must remain is not observing that one does. |
| **4. retry the ORIGINAL call verbatim** | a synthesized retry failed on every kv firing *after* the eviction succeeded — destroying a duplicate and losing the write, strictly worse than nothing |

Bounded at **3 evictions per episode**: unbounded, it empties the container one entry at a time while
every individual step passes the losslessness test.

## The integrity clauses all passed

Measured at the point of action, not audited afterwards:

| clause | result |
|---|---|
| live archival entries seen at the decision | min 50, median 50, max 50 |
| evictions where ≥ 2 copies existed | **43 / 43** |
| copies remaining after removal | min 1, max 2 — **never zero** |
| tool failures | **0** |
| retries that landed | **43 / 43** |
| archival slot rejections | 66 → **47** |

## And it still lost 3 cases — which is the point of this round

Frozen clause 2 required **zero losses**. The arm lost **3**, all on one cell, against a **zero
variance floor** where a lost case is a real loss rather than noise.

The losses were traced, and they are **not the eviction**: `core_memory_remove` calls rose **42 → 96**
once A5 began firing. The model, observing a store that changed under it, became *more destructive on
its own*.

> **Induced model drift.** A mechanism can be provably lossless at its own decision point and still
> lose cases, because the agent reacts to the changed world.

That is the general cost of intervening at all, and it is the reason this round ships an
`OVERRIDE.md` rather than a `FROZEN.md`. The 3 losses are **baked into the baseline of every later
arm** rather than excluded.

## How the acceptance was settled

**A reader who weights pre-registration absolutely was right to call this unaccepted — as a *process*
criticism.** That criticism stands and is not withdrawn.

What settled it is evidence, under the governing rule
([`../../docs/ACCEPTANCE_RULE.md`](../../docs/ACCEPTANCE_RULE.md)): positive train net, **+8.33 pp on
independent dev**, attribution confined to the single backend it engages (`vector` +17.50 pp; kv and
rec_sum exactly 0.00), and a mechanism **causally validated from A5's own decision-point telemetry** —
30 evictions, 30/30 landed, 30/30 lossless, 24 correct declines.

> The override was **procedurally wrong and substantively right.** Both halves are recorded, and the
> dev result does not erase the process failure.

Note also that a **zero-loss rule would have rejected A5** — which is precisely why the governing rule
says *net*, not *no losses*.

## What it changed for the next round

The re-mine on this residual ([`../A6_deferred/ledger/n0t6_locus.json`](../A6_deferred/ledger/n0t6_locus.json),
frozen, fingerprint `869316d8730866c4`) ranks **`size/item/exceeds_per_item_limit` first**. The miner
selected that locus; nobody picked it — and the ranking stands even though the remedy built for it
(A6) was later rejected.

More consequentially, A5 established that A1's relocation target saturating is not a one-off. On a
store at its cap every later repair is choosing *what survives* rather than *what fits* — the framing
A6 walked straight into.

That remedy was later **rejected** (see [`../A6_deferred/`](../A6_deferred/)); the locus it targeted is still open. The next accepted anchor is **[T7 — A7](../T7_A7_blob_overflow/)**.
