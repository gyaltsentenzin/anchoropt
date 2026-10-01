# A5 acceptance — an EXPLICIT OVERRIDE of a frozen clause

> **Read this header before the numbers.** A5's criteria *were* pre-registered
> (`docs/N0_A5_CRITERIA_FROZEN.md` in the working repo, written before launch), and the arm
> **failed one of them**. It was installed anyway, on a recorded override. The filename is
> `OVERRIDE.md` and not `FROZEN.md` for that reason: a directory of `FROZEN.md` files would imply
> every anchor cleared its own bar, and this one did not.
>
> Overriding a frozen clause after seeing the result is the single most dangerous move in this
> protocol — it is one step away from selecting on the outcome. The argument for it is below, in
> full, so a reader can disagree with it.

## The anchor

| | |
|---|---|
| locus | `capacity/container/no_remaining_slots` |
| attribution | the container A1 reroutes *into* has hit its slot cap |
| decision point | post-execution |
| action | reroute (evict one provable duplicate, then retry the original write verbatim) |
| policy diff | one key: `"enable_archival_evict_duplicate": true` |

A1 answers "this container cannot hold the write" by relocating the payload to archival. A5 is the
anchor for the state A1's own success eventually produces: archival is now at its slot cap, and the
pending write has nowhere left to go.

## The clause that failed

Frozen clause 2 required **zero losses** against the incumbent. The arm lost **3 cases**, all on one
cell (vector / customer), against a **zero variance floor** — at `temperature=0.001` a lost case is a
real loss, not noise, which is precisely why the clause was written that way.

| | control | A5 arm |
|---|---|---|
| accuracy | 128/303 = 42.24% | 132/303 = 43.56% |
| delta | — | **+1.32 pp**, 7 gains / **3 losses** |
| archival slot rejections | 66 | **47** |
| evictions performed | — | **43**, every retry landed |

## Why it was installed anyway

**The mechanism is provably information-preserving, and the losses are not the mechanism.**

Every integrity clause passed, measured at the point of action rather than audited afterwards:

| clause | result |
|---|---|
| live-state read at the decision point | archival entries seen: min 50, median 50, max 50 |
| evictions where ≥ 2 copies existed | **43 / 43** |
| copies remaining after removal, measured | min 1, max 2 — **never zero** |
| tool failures | **0** |
| retries landed | **43 / 43** |

So no eviction destroyed the last copy of anything. The 3 losses were then traced, and they come
from somewhere else: **`core_memory_remove` calls rose 42 → 96** once A5 began firing. The model,
observing a store that changed under it, became *more destructive on its own*.

That is **induced model drift** — a downstream consequence of intervening at all, not of what the
intervention did. It is the general cost this anchor documents: a mechanism can be provably lossless
at its own decision point and still lose cases, because the agent reacts to the changed world.

## What this override costs, and where the cost is carried

The 3 losses are **baked into the baseline of every later arm**. A7 was measured against an
incumbent that already contains them, as was the A6 attempt. They were not excluded, re-run away, or attributed to noise.

**The honest reading:** A5's delta is positive and its mechanism is sound, but this *particular
acceptance* rested on a judgement — that a traced, off-mechanism regression is tolerable — and not on
the criterion that was frozen. That process criticism stands.

What settled it afterwards is evidence, under the governing rule: **+8.33 pp on independent dev** and a
mechanism **causally validated 30/30** from A5's own decision-point telemetry. Note also that a
zero-loss rule would have rejected A5 outright — which is exactly why the governing rule says *net*.
See [`../../docs/ACCEPTANCE_RULE.md`](../../docs/ACCEPTANCE_RULE.md).

## A note on which A5 number is quoted

The progression uses **132/303 = 43.56% (+1.32 pp)** — A5's own acceptance arm. A **131/303 (43.23%)**
figure also appears in the working repo's later tables: that is the *post-repair* incumbent, one case
lower, and it is a different arm rather than a different reading of this one. See
`MEASUREMENT_GENERATIONS` in [`../anchors.py`](../anchors.py) for the related and more important point
that whole-corpus paired runs are not byte-reproducible while sharded ones are.

## The next locus this handed forward

Induced model drift, and — more consequentially — the observation that A1's relocation target
saturating is not a one-off. On a store at its cap, every later repair is choosing *what survives*
rather than *what fits*. That is the framing A6 then walked straight into.
