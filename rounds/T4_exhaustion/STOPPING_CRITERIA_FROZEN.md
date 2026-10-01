# Stopping criteria for the anchor-learning loop — FROZEN

Written **before the A2.R verdicts landed** (jobs 834667/834668 were still RUN), so it cannot be a
post-hoc justification for stopping wherever the loop happens to be. That timing is the point: a
stopping rule chosen after seeing the next result is not a rule, it is a rationalisation.

## Why a stopping rule is needed at all

The loop — mine → rank → evaluate → freeze → re-mine — has no natural end. It will always produce a
rank 1. Three failure modes it must be protected against:

1. **Chasing exhausted loci.** An accepted anchor *settles* exposures. Rank 3 already carries
   `settled=372` of `raw_ev=724`; without a floor the loop keeps re-proposing work largely done.
2. **Chasing loci too small to measure.** At n=303 a 5 pp effect is ~15 cases. A locus linked to 12
   failing queries cannot produce a measurable, let alone significant, result even if the anchor is
   perfect.
3. **Accumulating ineffective anchors.** Each anchor is code, a gate, and a maintenance cost. Three
   anchors that each add +0.5 pp are worse than one that adds +4 pp, because the surface area grows
   while the effect does not.

## The rule

Stop the current tree line when **any** of the following holds.

### S1 — coverage floor: no candidate reaches 10 % of the residual

The top-ranked *unsettled* locus must be linked to **≥ 10 % of failing queries**. Below that, a
perfect anchor cannot move the aggregate enough to distinguish from noise on this corpus.

State as measured at N0-T2 (residual 201):

| rank | locus | linked | % residual | above floor? |
|---|---|---|---|---|
| 1 | `existence/identifier/not_found` | 47 | 23.4 % | yes |
| 2 | `permission/identifier/duplicate` | 40 | 19.9 % | yes |
| 3 | `capacity/container/no_remaining_capacity` | 34 | 16.9 % | yes |
| 4 | `size/item/exceeds_per_item_limit` | 23 | 11.4 % | yes |
| 5 | `format/field/malformed_value` | 12 | **6.0 %** | **no — below floor** |

So rank 5 is already excluded by S1 *today*, before any further acceptance. Every acceptance shrinks
the residual and settles exposures, so more loci fall below the floor each round.

**10 % is chosen from the corpus, not from taste:** 10 % of a ~200-case residual is ~20 cases; the
sign test needs ≥ 9 net at that n, so an anchor resolving even half its exposure is at the edge of
detectability. Below 10 % the measurement cannot answer the question being asked.

### S2 — settled saturation: the locus is mostly already handled

Stop considering a locus when **`settled / raw_events ≥ 0.5`** — the incumbent already acts on most
occurrences. Rank 3 is at **372/724 = 51 %**, so it is at this boundary now. What remains there is an
A1 *coverage* question (widen an existing anchor's reach) rather than a new anchor, and those are
cheaper: a parameter change, not a new mechanism.

### S3 — no admissible action: the live-cell set is empty

If the attribution miner's live cells for a locus are all excluded — structurally, or by this line's
constraints — the locus is **identified but not actionable** and is recorded, not pursued. Already
binding: **A2.W** (5 occurrences, only live cell `pre_generation + reprompt`, which the prompt-free
line excludes) and the 8 `unattributed` occurrences (empty admissible set by construction).

### S4 — diminishing returns: two consecutive rejections, or two accepts below +1 pp

* **Two consecutive REJECTS** at the top of the ranking ⇒ the ranking is no longer selecting
  actionable work. Stop and re-examine the *miner*, not the next locus.
* **Two consecutive ACCEPTS each below +1 pp on train** ⇒ the marginal anchor is not paying for its
  maintenance surface. Stop and consolidate.

### S5 — unmined residual dominates

At N0-T2 the five loci account for **156 of 201 = 78 %** of the residual. The remaining **22 % has no
mined locus at all** — no recurring tool error explains it. When the *unmined* share exceeds the
largest mined candidate, the bottleneck is the **signal vocabulary**, not the policy: the next move is
a new detector (as `vacuous_result` once was), not another anchor.

## What stopping does NOT mean

* Not "the method failed." S1/S2 firing means the addressable-by-this-signal-family work is done.
* Not "no headroom remains." It means no *anchorable* headroom remains **at the current signal
  resolution**. [[anchoropt-g8-heldout-confirmed]] records the post-prompt residual as 82 % "retrieved
  OK but answered wrong" — real headroom that no error-keyed miner can see, because nothing errors.
* Not permanent. A new detector, a new incision point (nothing has been learned at
  `pre_generation` yet), or a different backend re-opens the loop.

## Precedence

Check in order **S3 → S1 → S2 → S5 → S4**: cheapest first, and S3/S1 are free (no arm needed), while
S4 requires having run arms. This mirrors the evidence → actionability → downstream ordering used for
individual candidates.
