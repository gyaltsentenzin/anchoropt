# Saturation, selection, and the two diagnostics that separate them

This is the transferable half of the A5–A7 line. The anchors themselves are specific to one
benchmark's memory backends; the failure mode below is not, and neither are the two checks that catch
it. Read this before porting any capacity-shaped anchor to another benchmark.

---

## 1. The failure mode: a capacity intervention that is really a selection intervention

A store with a fixed number of slots has two distinct regimes, and an anchor can look identical in
both while doing opposite things:

| regime | what a repair does | does it add information? |
|---|---|---|
| **headroom** | frees or finds room the payload can occupy | **yes** |
| **saturated** | takes a slot from something already in the store | **no — it chooses** |

Under headroom, relocating a rejected write is a genuine capacity repair. Under saturation, the same
code cannot add anything: the slot count is fixed, so admitting one entry evicts another. The
intervention has silently changed job from *making room* to *deciding what survives* — and there is
no utility model at a single decision point that can make that call well.

**A6 is the worked example, and it was REJECTED on this evidence** — see
[`../rounds/A6_deferred/`](../rounds/A6_deferred/). It catches a per-entry length rejection and reroutes
the payload to a container with a larger per-entry cap. On train (many short prerequisite chains,
repeatedly fresh stores) it buys real headroom: **+1.65 pp**. On dev (one long chain per backend,
saturated early and staying saturated) it reads **−1.19 pp**, and on `vector` — its only target
backend — **−2.50 pp**. Every variant tried is negative on dev.

The mechanism is measured, not inferred:

| dev vector dispatch site | incumbent | + A6 |
|---|---|---|
| `a6_reroute` (A6's own) | 0 | **44** |
| **`a1_reroute`** (the incumbent anchor) | **22** | **6** |

A6 fires 44 times while A1 collapses 22 → 6. A6 catches the payload one error *earlier* and reroutes
it before the state A1 acts on can form. **It adds no capability — it pre-empts A1 on the same
payloads.** And the store shows what that buys:

| | incumbent | + A6 |
|---|---|---|
| final distinct facts | 57 | 56 |
| facts destroyed / added | — | **54 out / 53 in** |

A near-total content swap at constant saturated capacity — the same **topics** in different
**phrasings** (`…sparked early cs interest` vs `…inspired early cs interest`). Flips come out almost
balanced: 9 lost, 8 won. Whether the swap helps is a coin flip on how the surviving phrasing aligns
with the queries.

**No threshold fixes this.** The problem is not which duplicates get written; it is which facts
survive saturation.

### The corollary that kills the obvious next idea

The remaining pressure after byte-exact deduplication is the **paraphrase tier**: of allowed writes,
**15.8%** are ≥ 0.90-similar one-edit paraphrases and **73.7%** are < 0.50 genuinely distinct.
Merging paraphrases is the natural next lever, and it is the same coin flip — it changes *which
phrasing survives*. Byte-exact suppression is nearly exhausted as a lever, and what remains is a
selection decision wearing a deduplication costume.

Also measured, and counterintuitive: **suppression is not monotonically good.** A normalized
comparator (casefold, strip punctuation, collapse whitespace) *over*-suppressed and retained **17**
dev facts where raw byte-exact retained **56**. Informative variants matter — two phrasings of a
topic can answer different questions. Prefer the strict comparator.

---

## 2. Diagnostic one: displacement

Before believing any candidate's gain, compare **per-site dispatch counts** between the incumbent and
the candidate, each read from its own telemetry.

```python
from anchoropt.learning.exposure import displacement_check

report = displacement_check(
    incumbent_sites={"a1_reroute": 22, "controller_write_repair": 135},
    candidate_sites={"a6_reroute": 44, "a1_reroute": 6, "controller_write_repair": 114},
)
report.verdict   # "SELECTION"
```

| verdict | meaning |
|---|---|
| `ADDITIVE` | new dispatch sites opened, no incumbent site collapsed — the candidate adds capability |
| `SELECTION` | a new site rose **while an incumbent site collapsed** — the candidate pre-empts, and its train gain is unlikely to transfer |
| `INERT` | no dispatches at all; any delta is not attributable to the candidate |
| `NO CHANGE` | dispatch sites unchanged |

`SELECTION` is not automatically a rejection. It is a statement that the gain is conditioned on
something the split geometry controls, so a dev regression should be *expected* rather than treated
as a surprise.

---

## 3. Diagnostic two: exposure

Most anchors are gated to one backend, one domain, or one error contract. Everything outside that gate
is untouched **by construction**, which makes it a **free noise estimate on every paired run.** Use it.

```python
from anchoropt.learning.exposure import Cell, exposure_weighted_delta

report = exposure_weighted_delta([
    Cell("vector",  n=89,  control_correct=30, arm_correct=31, exposed=True,  firings=0),
    Cell("kv",      n=105, control_correct=30, arm_correct=35, exposed=False),
    Cell("rec_sum", n=109, control_correct=54, arm_correct=59, exposed=False),
])
print(report.summary())
```

This is the case that motivated writing it. A candidate scored **+3.63 pp train / +7.14 pp dev**,
both positive, no reversal — and was rejected:

| backend | n | delta | guard fired? |
|---|---|---|---|
| **vector** (exposed) | 89 | **+1.12 pp** | **0 times** |
| kv | 105 | +4.76 pp | never |
| rec_sum | 109 | +4.59 pp | never |

A noise floor of **+4.67 pp on 214 unexposed cases** against **+1.12 pp on the 89 exposed** ones. The
delta measured **store perturbation**, not the mechanism: any intervention in a prerequisite episode
perturbs the store, and perturbation alone moves scores everywhere.

> **Engagement of zero with a positive delta is proof of non-attribution, never evidence of a subtle
> mechanism.**

The same arithmetic governs *reporting*. An anchor reaching 36% of the corpus cannot be quoted at its
within-backend figure: **+7.34 pp on that backend is +2.64 pp corpus-wide.** Quote both, with the
exposure attached. `exposure_weighted_delta` emits that warning automatically.

### Spliced is not measured

When only the exposed backend was actually run, a corpus-wide figure assembled from *the incumbent's
other backends + the arm's exposed backend* is **spliced**. The splice is sound arithmetic over a
population the anchor cannot alter — but it is not an end-to-end measurement, and until the fleet runs
it must carry the word.

A7 is the worked example of why that matters. Its corpus figures *were* spliced, and the subsequent
three-backend run both confirmed them on train and **corrected the dev baseline**: the splice had used
a pre-A6 reference, so the incumbent's true dev figure was 33.33% rather than 34.52%, which had
*understated* A7's gain. A7's figures are now measured end to end (0 disagreements and byte-identical
logs on the 258 cases it cannot reach), so the label has been removed — but the correction is the
reason the label exists.

---

## 4. Why the dev fold produces reversals here

The dev fold is **one domain chain per backend**. So it measures **domain transfer**, not
within-distribution generalization, and any anchor whose effect depends on chain multiplicity or
saturation depth will appear to reverse on it:

| | `a1_reroute` | at-capacity fraction | entry-length errors |
|---|---|---|---|
| train: incumbent → +A6 | 84 → **110** (rises) | 85% → 81% (falls) | 55 |
| dev: incumbent → +A6 | 22 → **6** (collapses) | 62% → 70% (rises) | **102** |

Many short chains give repeatedly fresh stores, so a reroute buys headroom and the incumbent anchor
keeps working. One long chain saturates early, so the candidate displaces the incumbent and reshuffles
slots. Dev also carries ~2× the entry-length errors on a quarter the prerequisite count.

**A leave-one-chain-out protocol is the honest fix** when a verdict needs dev support. It is
deferred, not done, and that is stated rather than worked around.

---

## 5. Run protocol: shard by backend, and submit the shards CONCURRENTLY

**Every paired arm runs as one isolated job per backend, each serial internally.** This is required
for any paired train contrast, not merely preferred — and the concurrency is the point.

**Submit every shard at once, one GPU each. A 3-backend × 2-arm contrast is 6 simultaneous jobs, not
6 sequential ones.** Every sub-job is independent: separate process, own store, own snapshot-cache key
(the cache key hashes the prerequisite ID list), and chains are contiguous per backend so no chain can
observe another backend's store. There is nothing to serialise. The A7 line ran its arms sequentially
and paid roughly a 4× wall-clock penalty for a fleet that could have cost one shard's time.

### It buys determinism, not just wall-clock

This is the part that matters more than speed:

| | parallelism | measured | status |
|---|---|---|---|
| **within** one store build | intra-process | replicates diverge **reproducibly** | **forbidden** |
| **across** isolated runs | one process per backend | **0 flips**, stores hash-identical | **default** |

**Whole-corpus runs are not byte-reproducible.** Two runs of the *same effective policy* differ even
when the treatment's guard fires **zero times**, because whole-corpus execution adds cross-chain
sequencing that is not byte-stable even with store construction forced serial. Every *sharded* pair is
bit-identical except in the single shard where the treatment actually fires.

Consequence, applied in this repo: **a paired train number from a whole-corpus run must not be
quoted.** That demotes A5's own headline (`+1.32 pp`, whole corpus) in favour of the sharded reference
(`+0.99 pp`), which is the line A6 and A7 were both measured against. See `MEASUREMENT_GENERATIONS` in
`rounds/anchors.py`.

Two gotchas, both learned the expensive way:

- **Ports must be distinct per (arm × backend)**, not per backend. Two arms sharing a port were safe
  only because the scheduler happened to place them on different hosts; a co-location reproduces a
  port-collision bug.
- **Equivalence must be checked on store CONTENT, never on cache keys.** Each shard necessarily gets
  its own cache key, which is correct and tells you nothing about equivalence.

Sharding parallelises a **different axis** from store construction. It does not relax the
serial-store-build requirement.

---

## 6. The checklist this reduces to

Before quoting any capacity-anchor result:

1. **Engagement** — did the gate fire, measured on *this arm's own* telemetry? Zero firings with a
   positive delta is non-attribution.
2. **Exposure** — split the delta by exposed vs unexposed population. Is the exposed delta above the
   unexposed noise floor?
3. **Displacement** — did an incumbent dispatch site collapse as the new one rose? If so, expect a
   dev regression and say so in advance.
4. **Baseline** — is the comparison against the **incumbent stack**? An arm-vs-its-own-host-anchor
   delta answers "did my patch improve my patch". The same candidate reads **+7.14 pp** against its
   host anchor and **−1.19 pp** against the incumbent; the baseline choice flipped the sign of the
   conclusion.
5. **Generation** — is the number from a sharded run? If it is whole-corpus, it does not reproduce.
6. **Splice** — is the corpus-wide figure measured end to end, or assembled? Say which.
