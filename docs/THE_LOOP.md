# The loop: counterfactual worlds, deferral, and when to expand

![AnchorOpt: decision-centric local interventions in the agent loop](img/anchoropt_agent_loop.png)

**Where the anchors attach.** The baseline loop (left) has one lever: the prompt. The AnchorOpt loop
(right) names the objects inside a step — model input, selected action, tool result — and each named
object is a point the system can act at. The three dashed boxes are the three incision points; every
accepted anchor in this repo is one of them. See
[`INCISION_POINTS.md`](INCISION_POINTS.md) for why the point matters more than the action, and
[`OBSERVABILITY_VS_DECISION_CONTEXT.md`](OBSERVABILITY_VS_DECISION_CONTEXT.md) for the one anchor (A9)
that acts on the tool result rather than on the instruction.

Three things that are easy to miss from the delta table, and that carry most of the method.

1. **Every installed anchor's world becomes the next round's no-op arm.** The baseline moves.
2. **The deferrals are where the acceptance criteria become legible.** A rule applied only to winners
   is not a rule.
3. **Expanding the signal vocabulary is a decision with a trigger**, not something you do when the
   ranking looks thin.

## Attribution ranks by linked downstream loss, not frequency

AnchorOpt does not rank errors by frequency alone. For each failed query, attribution walks the
trajectory in reverse to the earliest decision whose alternative would have made the answer reachable,
and credits the loss there — not to whatever error string appeared closest to the final answer. **A1
ranked rank 1 by linked loss — 81 of 215 residual failures — not by event volume**: a rejected write
silently removes a fact that a query several turns later needs. Frequency ranking would have surfaced
the downstream read failure instead, which is the symptom, not the cause.

## The residual measurably shrinks, and the vocabulary gets exhausted

Share of remaining failures covered by the mined loci, re-measured each round: **T0 78% → T2 61% → T3
13% → T5 3%**. By T5 the lexical vocabulary that found A1–A3 was exhausted — canonicalized error strings
(`no_remaining_capacity`, `not_found`, `duplicate`) stopped accounting for what still failed. T4 is that
exhaustion recorded in place: a real round where no locus survived and no anchor was installed, which is
exactly the trigger that forces signal expansion from lexical to structural predicates over live state —
A4's locus is `no_tool_call_at_all`, an absence rather than a string.

## Induced errors: necessary cost of a repair, or pathological?

Re-mining raises an obvious objection: if A1 caused the failures A2 and A3 fix, is AnchorOpt just
pushing errors downstream? The method cannot simply forbid induced errors — several accepted anchors
propagate to a successor, and A1 alone seeded three — so it classifies them instead, with a one-step
lookahead at acceptance time. An induced signal whose follow-on action is **forced** (`forced_rate ≥
0.8`) is the *necessary* cost of a real repair: A1 converted a fatal capacity error into a recoverable
retrieval problem, which is progress. One that merely invites a new failure mode is **pathological**,
and it is harm.

The rejected contrast makes the rule credible rather than rhetorical: W1's entry-condensing repair
validated 29 of 29 and was rejected, because **25 of those 29** were immediately blocked by the next
constraint — it had moved the failure one step, not fixed it. The guard is deliberately **one-step**,
not a multi-round search.

## The shape of the whole thing: coordinate descent over two blocks

Before any of the detail: the loop alternates between improving **two different things**, and holds one
fixed while it improves the other. That is block-coordinate descent, and naming it explains most of the
design decisions further down.

```
   ┌─────────────────────────── POLICY BLOCK ────────────────────────────┐
   │  signal vocabulary FIXED                                            │
   │  mine → attribute → rank → incise → measure → install → re-label    │
   │  → re-mine → …  until every locus hits a stopping rule              │
   └──────────────────────────────┬──────────────────────────────────────┘
                                  │  S1/S2 stop everything AND
                                  │  unexplained mass dominates (S5)
                                  v
   ┌─────────────────────────── SIGNAL BLOCK ────────────────────────────┐
   │  policy FIXED                                                       │
   │  propose step-local detectors → score coverage × precision          │
   │  → keep what discriminates → FREEZE the vocabulary                  │
   └──────────────────────────────┬──────────────────────────────────────┘
                                  │  vocabulary frozen
                                  └──────────► back to the policy block
```

On this line the alternation ran: **T1–T3 policy** (three anchors from the error-keyed vocabulary) →
**T4 signal** (nothing installable; expand) → **T5 policy** (A4, from the new vocabulary) → **signal
again** — which is where A5, A7 and A8 came from.

Three consequences that are not obvious until you name the structure:

- **Only one block moves at a time, and that is what makes attribution possible.** If you changed the
  signal vocabulary and the policy in the same round, a gain would be unattributable between them —
  you could not say whether you found a better lever or merely a better way to *see* the same lever.
  So the vocabulary is **frozen before** any policy is fitted against it.
- **Each block has its own objective, and they are not the same.** The policy block maximises paired
  task gain under a do-no-harm constraint. The signal block maximises *discriminating coverage* —
  precision against passing episodes, not against failures. A detector that fires on 77 % of the
  residual and on 89 of 117 successes is a good coverage number and a useless signal.
- **The stopping rules are the switching condition, not a shutdown.** S1/S2 firing on every locus does
  not mean "done" — it means *this block is converged*. Only S5 (unexplained mass exceeds the largest
  explained candidate) tells you which way to switch: if unexplained mass is small, you are genuinely
  finished; if it dominates, the bottleneck moved to the other block. Conflating those two is the
  failure `phase_switch.py` exists to prevent.

Like any coordinate-descent scheme this gives no global optimality guarantee — it descends on the
residual it can currently *see*, which is precisely why the signal block exists at all.

---

## 1. The installed anchor becomes the default no-op arm

After a candidate is accepted, its trajectories are **persisted as the incumbent world**, and the next
round is measured against *that* — not against the original baseline, and not against a freshly re-run
control.

```
   T1: candidate A1  vs  no-op = T0 native world          -> A1 accepted, 33.66 %
                                    │
                                    │  A1's own trajectories are persisted
                                    v
   T2: candidate A2  vs  no-op = the A1 world (33.66 %)   -> A2 accepted, 35.97 %
                                    │
                                    v
   T3: candidate A3  vs  no-op = the A1+A2 world (35.97 %)-> A3 accepted, 38.61 %
                                    │
                                    v
   T5: candidate A4  vs  no-op = the A1-A3 world (38.61 %)-> A4 accepted, 42.24 %
```

The frozen specs state this explicitly. A3's reads:

> | control for this arm | the incumbent's own results; **not re-run** |

Verifiable in the shipped artifacts — round N's control file *is* round N−1's arm file:

| arm | sha256 (12) | scored | role next round |
|---|---|---|---|
| T1 A1 | `f90db95bbcde` | 102/303 | control for T2 |
| T2 A1+A2 | `c1af359e9822` | 109/303 | control for T3 |
| T3 A1–A3 | `6d30bc6eae9e` | 117/303 | control for T5 |
| T5 A1–A4 | `f5d18b2c9ff0` | 128/303 | current incumbent |

### Why this matters rather than being bookkeeping

**It makes the procedure boosting.** Each round fits one increment against what the *current* policy
still gets wrong. The residual is re-weighted by the anchor that just landed, so the next candidate is
selected against a world that already contains its predecessor.

**It is also the experimental control.** The incumbent's *own recorded results* are the control for the
next round — not a freshly re-run baseline. Re-running the control separately would reintroduce variation
the design exists to remove, so **not re-running is a correctness property, not an optimisation.**

**One precision that is easy to get wrong, and I did.** "Both arms share the same factual prefix" is true
for an anchor that acts strictly *downstream* of the storage phase — but **A1-A4 are
`pre_snapshot`** (see `GateSpec.snapshot_anchor`): A1 reroutes writes, A3 suppresses them, so each can
change what the storage phase *produces*. The store cache key therefore includes
`gate_config_fingerprint`, and the two conditions legitimately build **different worlds**. Both full runs
show two distinct `base_key` values.

That is correct — scoring an anchors arm against a store built *without* the anchors would measure the
wrong counterfactual. But it means the comparison is paired on **case identity**, not on shared world
state: read it as "policy A vs policy B, each in the world it creates." For a storage-affecting anchor
that is the right question, and it is a weaker claim than a strict shared-prefix A/B. See
[`../benchmarks/bfcl_v4/VENDORED.md`](../benchmarks/bfcl_v4/VENDORED.md).

### The consequence that bites

**Never difference accuracies across rounds' jobs.** Two controls in this project read 29.04 % and
24.36 % purely from corpus composition, while agreeing to **1 flip in 228** on shared cases. Only
within-job, paired, control-vs-candidate comparisons are licensed — which is exactly what
`scripts/verify_progression.py` computes.

And it is why **A4 v1's raw 33.66 % is meaningless on its own.** It coincides with A1's accuracy by
accident; against its actual incumbent (A1–A3, 38.61 %) it is **−4.95 pp**.

---

## 2. Deferral is where the criteria become visible

A candidate that was declined tells you more about the standard than one that passed. Each deferral
below names the rule that stopped it — and every rule was **frozen before** the decision.

| deferred signal / policy | the number | rule | what the rule is protecting |
|---|---|---|---|
| **A2.W** write-side `not_found` | 5 occurrences | **S1** | An arm on 5 cases cannot produce a measurable result. Also: its mechanism is A1's own, so it is a *coverage* question for an existing anchor — not a new one with its own maintenance surface |
| **8 `unattributed` occurrences** | no write-side explanation | **S3** | The admissible action set is empty by construction. Identified ≠ actionable |
| `format/field/malformed_value` | 12 linked failures = **6.0 %** | **S1** | Below the 10 % floor: even a *perfect* anchor could not move the aggregate detectably |
| `capacity/…` re-proposal | **372/724 = 51 %** settled | **S2** | The incumbent already acts on most occurrences. Re-anchoring would re-solve solved work |
| `size/item/exceeds_per_item_limit` | fell **30 → 9** linked failures | **S1** after re-label | Occurrence-level re-labeling removed the repaired occurrences. Under the old chain-level count it looked actionable; it was not |
| `short_episode` | 77 % coverage, precision **0.57** | detector screen | Fires on 89 of 117 *successes*. High coverage, no discriminative power |
| `answered_after_read` | precision **0.50** | detector screen | Indistinguishable from a coin flip |
| `single_read_only` | precision **0.47** | detector screen | Fires on passes *more* than failures |
| **"retrieved OK but answered wrong"** | **82 %** of one residual | judgement, recorded | An observed *outcome class*, not an attributed *mechanism* |

### What each of these demonstrates about the standard

- **Support must be measurable, not merely real** (A2.W, `malformed_value`). The floor is derived from
  the corpus's own statistical power, so "the failure is genuine" is not sufficient.
- **Already-fixed work does not count** (`capacity` re-proposal, `size/item`). This is only visible
  *because* the incumbent world is re-mined and re-labeled — under a static ranking both still look
  like top candidates.
- **Coverage without precision is not a signal** (the three detectors). `short_episode` is the sharpest
  case: it covers 77 % of the residual and reads exactly like premature answering, and it carries almost
  no information. **Coverage alone never proposes a detector.**
- **Attribution is required, not optional** (bucket D). This is the hardest one to hold, because it is
  the *biggest* bucket. Wanting a signal is not having one; a detector keyed on "the answer was wrong"
  restates the label instead of naming a decision point.
- **Structural infeasibility is a legitimate terminus** (the unattributed 8). Recording "no admissible
  action" is a result.

**Deferred means recorded, not discarded.** Each stays in the ledger with its evidence, so if a later
round changes the world — a new detector, more support, a different incision point — it can be
reconsidered on its merits rather than rediscovered from scratch.

---

## 3. When to expand error attribution — the trigger, not the hunch

Expanding the signal vocabulary is the most tempting place to fool yourself: when nothing ranks well,
"we need better signals" is always available as an excuse. So the decision has a **quantitative
trigger** evaluated from already-frozen thresholds.

### The two terminal states, which look identical from the ranking

```
STOP                 no locus survives S1/S2   AND  unexplained residual is SMALL
                     -> the residual is genuinely addressed. Done.

EXPAND_ATTRIBUTION   no locus survives S1/S2   AND  unexplained residual DOMINATES
                     -> the bottleneck moved from POLICY to SIGNAL. Not done -- blind.
```

Conflating these is the failure the switch exists to prevent. "Nothing survives the gates" **alone**
means only that nothing survives *in the current representation*.

### What actually happened at T4

| | |
|---|---|
| every locus | **S1-STOP and S2-STOP simultaneously** |
| largest explained candidate | 8 % of residual — under S1's 10 % floor |
| unexplained residual | **97 %** |
| **S5** (unexplained > largest explained) | **fired** |
| verdict | `EXPAND_ATTRIBUTION` |

At 97 % unexplained, forcing a fifth error-keyed anchor would have meant optimising a vocabulary that
could not see almost anything remaining. And the mined-coverage trace shows the exhaustion arriving
rather than being asserted: **78 % → 61 % → 13 % → 3 %**.

### How to expand: three rungs, cheapest first

Expansion is not one move. There is an ordered ladder, and you climb it only as far as you have to —
each rung is more expensive and less precise than the one below it.

```
  1  LEXICAL           key on the error STRING the environment gave you
        │                  cheap, deterministic, auditable, high precision
        │  exhausted when: the failures no longer produce a distinct error string
        v
  2  SEMANTIC          canonicalize different wordings onto ONE constraint
        │                  removes benchmark-specific phrasing; makes support countable
        │  exhausted when: the failures produce NO error at all
        v
  3  STRUCTURAL        reason over trajectory SHAPE -- what was called, in what order,
                       what came back, and what the episode did next
```

**Rung 1 — lexical.** When the environment tells you what went wrong, use it. `core memory is full`,
`Key name must be unique`, `entry exceeds maximum length`: cheap to detect, deterministic, auditable,
and high precision. A1, A2 and A3 all came from here.

**Rung 2 — semantic.** Lexical strings fragment one constraint into several apparent candidates.
Canonicalization maps them onto a single locus — `core memory is full` and
`Memory size exceeds maximum size of 7` both become `capacity/container/no_remaining_capacity` — so
support is countable and the signal is not tied to one benchmark's phrasing. **The deterministic cue
table is a fast path, not the coverage story**: an unseen wording returns `None` and *vanishes* from the
ranking, which is why the LLM canonicalizer exists and why A3's locus came from it rather than the
table. Run `python scripts/walk_framework.py --step 1` to see that failure live.

**Rung 3 — structural.** Many consequential failures produce no error whatsoever. A write succeeds but
lands somewhere unreachable; a search returns `{"ranked_results": []}` — syntactically successful,
semantically empty; the model ignores a correct result. No string identifies these, so the detector has
to key on trajectory structure instead. A4's `no_tool_call_at_all` is a rung-3 signal.

These are **complementary, not substitutes.** Keep using rung 1 where the environment cooperates; climb
only for what it cannot see.

### Backward induction: start at the end state, then travel upstream

The three rungs above ask *what evidence do I key on*. This axis asks *where in the workflow do I look* —
and it is not a fallback for when the cheap rungs run dry. **It is the general procedure.** Start where
the reward is realised and work backwards toward the cause, exactly as you would solve any sequential
decision problem.

Failures are observed at **query** time, because that is where the task is scored. They are often
*caused* at **write** time — several turns, sometimes several episodes, earlier:

```
   observed here (scored)                       caused here
   ──────────────────────                       ───────────
   query: retrieval returns nothing      <──    write: the fact was never stored
   query: `not_found` on a key           <──    write: stored under a sanitised key
   query: answer is wrong                <──    write: stored, then cleared
```

**Why downstream-first, and not simply "go where the cause is."** Two reasons, and the second is the
load-bearing one:

1. **The end state is where the objective is defined.** Only at query time do you know a task was lost.
   An upstream event is a *candidate* cause; it earns that status by linking to a scored failure.
2. **Fixing upstream multiplies the downstream scenarios you then have to validate.** A read-side
   anchor changes one decision at the point of failure. A write-side anchor changes *what is in the
   store* — so every later read now runs against a different world, and the number of branches you must
   check grows with it. A1 is the worked example: rerouting writes was correct, and it spawned two new
   residuals (A2's `not_found` reads, A3's key collisions) that had to be learned in turn.

So the order is: **make the later stage well-behaved first, then move upstream.** Reaching upstream
before the downstream is settled means validating an intervention against a moving target — every
write-side change re-opens read-side questions you have not yet answered. That is also why the
`R1` lookahead term (§2.5 of the README) is a *hard* guard rather than a soft score: it is what stops
an upstream fix from silently expanding the branch set.

Two concrete findings from applying this ordering:

- **Rank 1 of one iteration was rejected precisely on this reasoning.** An archival search returning
  empty resolved in **0 of 50** episodes across every corpus that exhibited it. No read-side action can
  conjure a fact that was never written — the fault is upstream, in the write phase, and *that* is where
  the miner should have been pointed.
- **On train, 220/220 vector-length and 384/386 capacity events occur in prereq episodes**, ~0 in query.
  A query-only evaluation cannot exhibit them at all, which is why several early test-split numbers for
  those gates were uninformative.

The full escalation, then, is two-dimensional — **richer evidence** *and* **earlier in the workflow**:

```
                    lexical  ->  semantic  ->  structural
   query phase         A1          A2/A3          A4
   write phase         A5          A7/A8            .       <- where the later anchors landed
```

The bottom row is where the work went after A4, and it is now populated: **A5** (archival eviction),
**A7** (blob compaction) and **A8** (dedup-first clear recovery) are all write-side. The structural cell
on that row is still empty, and the capacity ladder names what belongs there — see
[`SATURATION_AND_SELECTION.md`](SATURATION_AND_SELECTION.md) and `CAPACITY_LADDER` in
`rounds/anchors.py`.

### The bar a new detector must clear

A backward-trace class (`never_stored`, `stored_then_cleared`, `stored_but_not_retrieved`) is a
**post-hoc label on a whole episode**. A detector must be a condition **checkable at a decision point**,
with no hindsight. Every candidate is scored on two numbers, and **precision is against successes**:

| candidate | coverage | precision | verdict |
|---|---|---|---|
| **`no_tool_call_at_all`** | 48 failing | **0.80** | accepted → **A4**, +3.63 pp |
| `short_episode` | 77 % | 0.57 | rejected |
| `answered_after_read` | — | 0.50 | rejected |
| `single_read_only` | — | 0.47 | rejected |

The expansion is then **frozen before** any policy is attached to it, so ranking and acceptance still
run against a fixed world. Proposing a signal and choosing a policy for it in the same step would make
the vocabulary a free parameter fitted to the outcome.

### The trajectory, end to end

```
  T3  A3 installed                          38.61 %
   │
   │  re-mine the A1-A3 world, re-label repaired occurrences
   v
  T4  every locus S1-STOP and S2-STOP; unexplained = 97 %
   │      -> STOP would be wrong: the residual is not addressed, it is invisible
   │      -> S5 fires; phase_switch returns EXPAND_ATTRIBUTION
   │
   │  propose step-local detectors; score coverage x precision-against-successes
   │  keep no_tool_call_at_all (0.80); reject three higher-coverage candidates
   │  FREEZE the expanded vocabulary
   v
  T5  A4 from the new vocabulary                42.24 %   (+3.63 pp)
   │
   │  re-mine again: explained coverage 13 % -> 3 %, no_tool_call_at_all is GONE (A4 consumed it)
   v
      S5 fires a second time -> EXPAND_ATTRIBUTION again  (which produced the write-side line)
   │
   v
  T6  A5 archival eviction                      43.56 %   (+1.32 pp)
  T7  A7 blob compaction                        45.87 %   (+2.31 pp)
  T8  A8 dedup-first clear recovery             47.52 %   (+1.65 pp)
  T9  A9 cross-container retrieval merge       52.48 %   (+4.95 pp)
```

The T5 step is the honest ending of the *first* expansion: the loop did not converge, it handed back
another expansion request — and answering it is what produced the write-side anchors. The loop still has
not converged, and the capacity ladder names where the next one belongs.

---

## See also

- [`KEEP_OR_DEFER.md`](KEEP_OR_DEFER.md) — the full S1–S5 definitions and their derivations
- [`INCISION_POINTS.md`](INCISION_POINTS.md) — why the action space is small before any arm runs
- [`GENERALIZABILITY.md`](GENERALIZABILITY.md) — which of this ports, and which is glue
- [`../rounds/`](../rounds/) — the five rounds, in order, with their frozen criteria
- `scripts/verify_progression.py` — recomputes every paired comparison above from the shipped results
