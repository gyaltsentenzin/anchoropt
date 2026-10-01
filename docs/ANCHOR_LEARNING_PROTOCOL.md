# The anchor-learning protocol — boosting iterations

The unit of the algorithm is a **boosting iteration**. Each iteration fits one increment
against the **current** incumbent's residual, which is boosting: every round re-weights toward what the
current model still gets wrong.

| | |
|---|---|
| **T0** | the incumbent — global prompt, **zero anchors** |
| **T1** | first policy-tree search over **T0's** residuals; inside it, ranked **candidates R1, R2, R3, …** evaluated in order |
| **T2** | fresh search over the residuals of **T0 + whatever T1 accepted** |

`R1, R2, …` are *ranked candidates* — proposals with evidence, no standing.
A candidate becomes an **anchor** only by passing measurement, and only then does it get an `A` number:

    T1 evaluates R1, R2, R3, ...   ->  R2 passes  ->  R2 becomes A1, the first anchor
    T2 evaluates R1, R2, ... (fresh ranking over the new residual) -> a pass becomes A2

So the incumbent after T1 is `T0 + A1`, where `A1` *is* whichever R won. Naming a candidate `A1`
before it has been measured pre-supposes the outcome, and this project has three recorded cases of a
mechanism-plausible candidate losing to measurement.

Per candidate, within an iteration:

1. prune `{noop, suppress, reroute, reprompt}` on **offline T0 counterfactual evidence**;
2. **reject offline** if no intervention is supported — costs minutes, not a GPU arm;
3. if viable, test **only on the cases exposed to that locus**;
4. accept only on **downstream counterfactual gain/harm**.

## The load-bearing invariant

**Acceptance CLOSES the iteration.** The accepted anchor changes the residual distribution, the
exposure sets, the ranking, and every downstream counterfactual — so the remaining `Rj` ranked in the
old world are **stale**, not merely pending. Learning one list from T0 and installing it sequentially
would treat rank-3 evidence gathered under T0 as if it described the T0+A1 world. It does not.

    new incumbent = old incumbent + accepted anchor
      -> replay the counterfactual world
      -> re-mine and re-rank residuals
      -> start T(n+1) with fresh R1, R2, ...

**Rejection does NOT close the iteration.** Rejecting leaves the incumbent unchanged, so the rest of
the ranking still holds and evaluation continues. `rank_candidates()` enforces both: `rejected=` filters
and continues, `accepted_this_iteration=` sets `closed`, drops `next` to `None`, and marks every
unevaluated row `stale`. This is the boosting invariant expressed as code — one increment per round,
re-weight, refit.

## Multiple anchors in one iteration — allowed, iff exposure sets are DISJOINT

Testing several anchors per round is compatible with boosting **only** when they cannot interfere.
Two anchors on **disjoint** exposure sets do not change the cases the other is measured on, so their
increments are independent and one arm can carry both. Overlapping ones cannot be co-measured: on a
shared case, the observed outcome is not attributable to either anchor.

This is decidable **offline**. Measured on A0 train:

| A | B | \|A\| | \|B\| | \|A∩B\| | verdict |
|---|---|---|---|---|---|
| rank 1 | rank 2 | 40 | 32 | **23** | **NOT co-measurable** |
| rank 1 | rank 4 | 40 | 14 | 0 | disjoint — co-measurable |
| rank 1 | rank 5 | 40 | 13 | 0 | disjoint — co-measurable |
| rank 2 | rank 4 | 32 | 14 | 0 | disjoint — co-measurable |
| rank 2 | rank 5 | 32 | 13 | **1** | **NOT co-measurable** |
| rank 4 | rank 5 | 14 | 13 | 0 | disjoint — co-measurable |

Maximal disjoint blocks: **{rank 1, rank 4, rank 5}** (67 exposed) and **{rank 2}** (32 exposed).

So an iteration may test one **block** per arm, with these rules:

* **Per-anchor accept/reject is evaluated on that anchor's own exposure set**, which is exactly why
  disjointness is required — otherwise the per-anchor attribution is unavailable.
* **The harm bound is evaluated on the FULL split**, once, for the block as a whole. Induced harm off
  the exposed sets is a property of the whole intervention.
* **A block with a mixed result is not "partially accepted."** Accepting the winners means installing a
  policy that was never measured as such; the winners become a **new block, measured fresh** in the
  next iteration. Cherry-picking within a measured block is selecting on the outcome.
* Rank 1 and rank 2 overlap on 23 episodes, so — even setting aside rank 1's offline rejection — they
  could never have shared an arm.

### Why batching matters: counterfactual generation is the cost

Strict boosting -- one anchor per iteration -- means **one counterfactual world per accepted anchor**.
Each new incumbent needs its trajectories replayed before the next round can be mined, and on this
corpus that is a paired job of ~40 min per arm plus a prereq store build. Ten anchors, strictly
sequential, is ten replays.

Batching disjoint anchors collapses that: **one replay per iteration regardless of how many anchors the
block contains.** That is the actual saving -- not the arms, the *replays*. And it is free of
attribution cost precisely because disjointness means no anchor changes the cases another is measured
on.

So the two ideas are complementary rather than in tension:

* **boost across TIME** (iterations) -- the incumbent updates once per round, and the next round's
  ranking is conditional on it;
* **batch within a ROUND** (disjoint blocks) -- several independent increments fitted against the *same*
  residual, because they provably cannot interact.

The honest limit: a block of *k* disjoint anchors is *k* increments fitted against one residual, so the
2nd and 3rd do not benefit from the 1st's correction. On disjoint exposure sets there is nothing to
benefit from -- but the moment exposure sets touch, that reasoning fails and the anchors must go in
different rounds.

**The cost is asymmetric between accept and reject.** A rejection costs no replay (the incumbent does
not move), so an iteration can evaluate many candidates cheaply and only pays a replay when something is
accepted. T1 rank 1 was rejected in minutes of trajectory analysis. That is why offline pruning is
step 1 of the per-candidate loop: the expensive step is *acceptance*, not evaluation.

**Where this trades against boosting purity:** a block of *k* disjoint anchors is *k* increments fitted
against the *same* residual, so the second and third do not benefit from the first's correction. That
is acceptable precisely because disjointness means they could not have benefited anyway. It buys GPU
arms — 1 instead of *k* — at zero cost in attribution, and that is the whole justification. It is not
licence to batch overlapping anchors for convenience.

---

## 1. The ranking rule (`policy_tree.rank_candidates`) — T1 over T0

Before this, "top-ranked" was **undefined**, for two concrete reasons:

* `support` counted **events** for `error_contract`/`reactive_call` but **episodes** for
  `vacuous_result` — one sort key over unlike units.
* `reactive_call` held the **top four** slots on the A0 corpus, but a reactive call is the model's
  **remedy**, not a failure locus. Ranking one as a target is the §94/§95 category error that produced
  a G1 gate matching `core_memory_clear` with **0 failures in 93 attempts**.

The rule, deliberately dull and auditable:

* **Rank only LOCI.** `reactive_call` is excluded with a stated reason, never silently dropped.
* **One unit: distinct FAILING episodes.** Tie-break on precision (a locus that is *always* fatal is a
  cleaner target than a noisy one), then signal name for determinism.
* **`RANK_MIN_FAILING_EPISODES = 5`** — an arm needs enough exposed cases to measure. **Not** a
  coverage bar; coverage is explicitly not a rejection criterion, because a local policy is allowed to
  be local.
* **Settled candidates are filtered AFTER ranking**, so ranks stay stable and history is not silently
  renumbered.

### Ranked loci on A0 (train, 303 query episodes)

| rank | kind | signal | failing eps | precision | status |
|---|---|---|---|---|---|
| **1** | vacuous_result | `empty_collection:archival_memory_key_search` | **40** | 1.00 | **T1: REJECTED offline (§2)** |
| 2 | vacuous_result | `all_floor_scores:core_memory_key_search` | 30 | 0.94 | **T1: next unresolved** |
| 3 | error_contract | `key not found` | 20 | 1.00 | pending |
| 4 | vacuous_result | `empty_collection:archival_memory_retrieve` | 14 | 1.00 | pending |
| 5 | vacuous_result | `all_floor_scores:archival_memory_key_search` | 11 | 0.85 | pending |

The rule selected rank 1 — **not** the candidate hand-picked as A1, which is rank 2. That disagreement
is the point: the protocol overrode a human choice, which is what makes it an algorithm.

---

## 2. T1 rank 1 — REJECTED OFFLINE, no arm run

`empty_collection:archival_memory_key_search`: an archival key-search returned
`{"ranked_results": []}` — a syntactically successful response containing nothing.

Offline evidence, pooled over **every corpus that exhibits the locus** (50 episodes):

| | |
|---|---|
| episodes | **50** |
| **resolved** | **0 (0.0 %)** |
| 38/40 on A0 | answered immediately, **no follow-up action at all** |
| follow-ups ever observed | `core_memory_retrieve_all` 2 → 0 resolved; `archival_memory_add` 2 → 0; `archival_memory_key_search` 1 → 0 |

Action space:

| action | verdict |
|---|---|
| `noop` | baseline **0.0 %** resolved |
| `suppress` | **PRUNED on mechanism** — the locus is a *read* returning nothing; blocking it removes nothing and supplies nothing. |
| `reroute` | **NOT ELIGIBLE** — the asymmetric bar (§0) requires **positive evidence for the replacement**. Zero destinations resolve, across 50 episodes and every corpus. |
| `reprompt` | **Only formally available**, and unattested: 95 % of episodes halt at this locus, so there is nothing to redirect *toward*. |

**Verdict: no actionable intervention has attested evidence of beating `noop`. Rank 1 is REJECTED
without spending a GPU arm — and because it is a rejection, T1 stays OPEN and evaluation moves to
rank 2.** That is the protocol working as intended: step 4 is reached
offline, and the cost of rejecting a candidate is a few minutes of trajectory analysis.

**Why this locus is probably not an anchor problem at all.** An empty archival search means *the fact
was never stored*, or was stored under a key this query cannot reach. No read-side intervention can
conjure it. The fault is upstream, in the write phase — which is a different locus class, and one the
miner should be pointed at (prereq-phase trajectories) rather than patched around on the read side.

---

## 3. T1 rank 2 = R2 — next unresolved (the hand-picked candidate's locus)

`all_floor_scores:core_memory_key_search`, 30 failing episodes, precision 0.94. Its evidence differs
from rank 1 in exactly the way that matters:

| | rank 1 | rank 2 |
|---|---|---|
| noop baseline | **0.0 %** (0/50) | **7.0 %** (3/43) |
| episodes that halt at the locus | 95 % | **0 %** |
| best attested follow-up | none resolves | `core_memory_retrieve` → **16.7 %** (1/6) |
| reroute eligible? | **no** | **yes** — a destination has positive evidence |

So rank 2 is the first locus where reroute clears the asymmetric bar. `docs/A1_ACCEPTANCE_FROZEN.md`
already holds its frozen design (written before this naming; it describes **R2**, which becomes **A1**
only if it passes); note its evidence table was built on the hand-picked conjunction
(28 exposed cases, `retrieve` at 21.1 %), whereas the protocol's own view of the same locus is 43
episodes at a 7.0 % baseline. **Both must be stated; they are the same locus counted under two
definitions**, and the frozen conjunction (which additionally requires "no value retrieved") is the
tighter one.

---

## 4. Recording rule

Every evaluated locus is recorded with **its iteration**, its rank, its evidence, and why it closed —
accepted, rejected-on-evidence (no arm), or rejected-on-measurement (arm run and lost). A rejected locus
is never silently re-tried: re-tuning it and re-running is selecting on the outcome, and a fixed variant
is a **new** candidate measured fresh.

**Rank numbers are only meaningful inside their iteration.** "Rank 2" means rank 2 *of T1*; the same
locus may rank anywhere in T2, or vanish entirely if the accepted anchor removed its failures. Never
compare ranks across iterations, and never carry a T1 ranking into T2.
