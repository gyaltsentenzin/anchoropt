# A6 — DEFERRED

> **A6 is DEFERRED: not in the accepted stack, and not closed.** The remedy fails the governing
> acceptance rule ([`../../docs/ACCEPTANCE_RULE.md`](../../docs/ACCEPTANCE_RULE.md)) on **two criteria
> independently** — chiefly a regression on the validation split. This directory is not numbered `T*`
> because it is not a round of the progression.
>
> "Deferred" rather than "rejected" is the accurate word, and the distinction is not softening: the
> *signal* is real, measured, and still unaddressed, while the *remedy built for it* is not ready. A
> rejection would imply the locus was a mistake to pursue. It was not — the miner ranked it first.
>
> **The signal is real and measured. The remedy was wrong.** Those are separable, and keeping them
> separable is the point of this file: dropping the anchor does not remove the errors it was
> responding to.

## Why it fails

| criterion | measurement | verdict |
|---|---|---|
| 1. train net > 0 | **+1.65 pp** (14 g / 9 l) | pass |
| **2. no dev regression** | **−1.19 pp** (8 g / 9 l, **17 discordant**) | **FAIL** |
| **3. attribution** | vector — its only target backend — is **−2.50 pp** | **FAIL** |
| 4. safety | 0 clears, 0 removes added | pass |

**Criterion 3 fails independently of criterion 2, and that is the cleaner statement.** The anchor
engages exactly where it was designed to and *makes things worse there*. An anchor whose on-target
effect is negative is not a tolerance question — no choice of ε rescues it.

The paired shape matters as much as the net. A6 churns **17 of 84 dev cases (20%)** to a net effect
of −1. Compare A7 at 5 gains / 0 losses. On net delta those are 7 pp apart; on shape they are not
comparable objects, and that is how "one stochastic flip decides an anchor" gets answered — by making
the shape visible rather than absorbing it into a threshold.

### A threshold was considered and rejected on process grounds

ε = 1 case = **1.19 pp** was considered. That is *exactly* A6's dev regression. Adopting it after
seeing the table would be choosing the threshold that admits the single anchor it decides —
defensible in the abstract, fitted in this instance. Recorded so the reasoning is auditable.

## What A6 was

| | |
|---|---|
| locus | `size/item/exceeds_per_item_limit` |
| attribution | the write is refused for the length of **one entry**, not for lack of room |
| decision point | post-execution |
| action | reroute (relocate to a container with a larger per-entry cap) **+** raw-exact dedup |

The locus was **chosen by the miner**, not by a person: the re-mine on the post-A5 residual
([`ledger/n0t6_locus.json`](ledger/n0t6_locus.json), frozen, fingerprint `869316d8730866c4`) ranks it
**first**. That ranking is not retracted by the deferral — the ledger ships here precisely so the
signal's standing is checkable independently of the remedy's fate.

## The mechanism — why this is a principled deferral, not a threshold call

Dev vector dispatch sites, incumbent vs A6:

| site | incumbent | + A6 |
|---|---:|---:|
| `a6_reroute` (A6's own) | 0 | **44** |
| **`a1_reroute`** | **22** | **6** |

**A6 fires 44 times while A1 collapses 22 → 6.** A6 catches the payload on the entry-length error and
reroutes it *before* the write reaches the core-full state where A1 would have acted. **A6 adds no
capability — it pre-empts A1 on the same payloads.**

What that does to the store:

| | incumbent | + A6 |
|---|---:|---:|
| final distinct facts | 57 | 56 |
| facts destroyed / added | — | **54 out / 53 in** |

A near-total **content swap at constant saturated capacity** — the same **topics** in different
**phrasings** (`…sparked early cs interest` vs `…inspired early cs interest`).

> **On a saturated store A6 cannot add information.** It can only reorder which entries win the fixed
> slots, and whether that helps is a coin flip on query-phrasing alignment.

That is the sign flip: **+1.98 pp train / −8.33 pp dev** for the bare variant. **A selection
intervention in a capacity costume.** No duplicate-suppression threshold reaches it, because the
problem is not which duplicates are written but which facts survive saturation. See
[`../../docs/SATURATION_AND_SELECTION.md`](../../docs/SATURATION_AND_SELECTION.md).

## Three variants, and the fact that none of them qualifies

| variant | train | dev | note |
|---|---:|---:|---|
| bare A6 (no dedup) | **+1.98 pp** | −8.33 pp | the train maximum, and 7× worse on dev |
| + normalized idempotence | +1.98 pp | −4.76 pp | **fired 0× on train**, so its train figure is bare A6's; all 34 dev firings from a *single* episode |
| **+ raw-exact dedup** (best) | +1.65 pp | **−1.19 pp** | the conservative variant — least harmful on dev, still negative |

The dedup half exists because relocation alone spent **85% of its granted archival slots on redundant
content** (29 exact + 5 near of 40), in a reroute → reroute ×29 pattern: the model retries the
over-long write, A6 relocates it again, and the relocation *manufactures* the duplicate saturation it
was built to relieve. **A repair that generates the pressure it relieves is not a repair.**

The conservative variant was the one carried forward, even though it has the *smallest* train delta.
That was the right instinct and it still was not enough.

## Worth keeping if this is ever reopened

**Raw byte-exact dedup beat normalized dedup on every axis:** dev −8.33 → **−1.19 pp**, facts retained
**56 vs 17** (net **+39**), **103 suppressions** against the normalized guard's **0**. If duplicate
suppression is revisited, **start from raw byte equality.**

The reason is counterintuitive and worth carrying: **suppression is not monotonically good.**
Informative variants matter — two phrasings of a topic can answer different questions — so an
over-eager comparator destroys information while looking like it is saving space.

## The signal that remains, measured and unaddressed

| signal | measurement | still present? |
|---|---|---|
| `entry_too_long` events | **55** train / **102** dev | yes |
| vector store saturation | **62–85%** at capacity | yes |
| byte-identical duplicate writes | **103** | yes |
| paraphrase tier | **15.8%** of allowed writes ≥ 0.90-similar; **73.7%** < 0.50 distinct | yes, untouched |

**With A6 removed, over-long vector entries are rejected and lost again.** That is the **1.65 pp train
cost**, priced in knowingly.

## Four conditions to reopen — and do NOT simply re-enable the flags

1. A **leave-one-chain-out protocol**, so a dev verdict is not one chain per backend.
2. A **selection-aware remedy** that decides which fact to *keep* on explicit grounds, rather than
   reordering slots as a side effect.
3. A corpus where vector stores are **not saturated** for most of the trajectory.
4. A **prose-capable fact detector** — the capitalisation-based lexical proxy is disqualified for
   exactly these claims.

## A7 is unaffected, and this was checked rather than assumed

A6 **never fires on rec_sum**: 0 firings across all runs, and the rec_sum shard is **bit-identical**
with and without A6 (0/109 train, 0/20 dev disagreements). Structurally necessary — A6 targets the
archival container and vector entries; rec_sum is a single string with neither.

**So A7 would have been mined identically had A6 never existed.** It was not a stepping stone, and
dropping A6 costs A7 nothing.

## The status history, recorded because it flipped twice

A6 was **frozen into the stack** under an earlier train-only rule, with the dev regression
reclassified as a "generalization diagnostic" — which is precisely the move the current rule forbids.
It was then dropped when that rule was replaced.

An earlier framing said the drop "deliberately overrides the train-based acceptance rule." **That was
wrong: there was nothing to override.** Under the governing rule A6 should never have been accepted.
The drop is rule-based, not a policy exception.

**Consequence to follow up:** any other anchor admitted on train-only evidence that dev contradicts
should be re-examined on the same grounds. A1–A5 were not audited that way in this pass.
