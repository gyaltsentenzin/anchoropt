# Keep, defer, or stop

How a candidate earns installation, and — far more often — how it is declined.

Of the candidates this line has evaluated, **four were installed**. Two rounds installed nothing, three
post-A4 candidates were rejected on measurement, and several loci were declined offline without ever
spending a GPU arm. **The declining is the method**, not an absence of results.

---

## 1. Acceptance: four questions, in this order

The order is load-bearing. Each stage can veto, and the cheap vetoes come first.

```
1. ENGAGEMENT    did it fire, in the mined context, on the target backend?
                 -> veto BEFORE any outcome is scored
2. ATTRIBUTION   are the observed changes caused by the intervention, not replay noise?
3. HARM          does any backend or subgroup degrade beyond tolerance?
4. BENEFIT       only now: do paired gains exceed losses?
```

### Why engagement is checked before scoring

Because a candidate can pass every outcome test while having nothing to do with the residual it claims to
address. That happened: a winner passed all four acceptance terms while firing **12× on a different
backend's error contract**, and its own target residual **grew**. The aggregate looked fine. The
mechanism was fictional.

So engagement is a precondition, evaluated from trajectory structure with no GPU:

- telemetry fired non-trivially;
- firings are in the target residual's backend and error contract;
- the target residual's frequency did not increase.

**A firing count is the first number to read.** 303/303 firings on a signal with 60 exposed episodes is a
6× over-fire, and it should be caught before accuracy is examined (see
[`INCISION_POINTS.md`](INCISION_POINTS.md)).

### A p-value is an input to the benefit term, not the arbiter

Nothing in the four terms above is a significance threshold, and that is deliberate.

**Per anchor, a p-value is not even a usable signal here.** This corpus has 15 indivisible
`(use case × backend)` cells; a single anchor's arm cannot reach `p < 0.05` on any split of it whatever
its true effect. So a per-anchor threshold read is uninformative in **both** directions — a pass would
be noise, and a fail would say nothing about the mechanism. Quoting one invites exactly the judgement
the design cannot support, which is why the README reports p for the **cumulative** result only and each
round's frozen spec carries its own statistics.

**What decides instead** is the ordering: whether the mechanism engaged where it claimed to, whether the
changes are attributable to it, whether anything degraded, and only then whether paired gains exceed
losses. A p-value informs that last term alongside the raw counts. It does not replace it.

This is not a licence to ignore statistics — it is why the design leans on **zero replay variance**
(0/303 outcome flips at `temperature = 0.001`), which makes each flip individually attributable and a
single loss a real loss. Determinism does the work that a larger n would otherwise have to.

### Why harm precedes benefit

**Do no harm first, then gain.** A candidate that improves the aggregate while degrading a subgroup is
not accepted. Concretely, the rule used on this line: paired gains must exceed losses **and** no backend
may degrade by more than 2 net cases against the incumbent.

This ordering exists because the alternative — accept on aggregate, investigate harm later — means
shipping a regression you have already measured and chosen not to look at.

### What "attribution" requires here

The measurement is a **paired counterfactual**: same factual prefix, one intervention, replay the causal
suffix. The replay variance floor was measured directly, twice, on identical-policy replicates:

> **0/12 store divergence, 0/303 call divergence, 0/303 outcome flips** at `temperature=0.001`.

Two consequences, and the second is the uncomfortable one:

- every flip is attributable to the intervention — there is no noise budget to hide behind;
- **a single loss is a real loss.** You cannot dismiss −1 case as churn.

A caveat worth stating: a measured floor bounds only what it measured. This floor covers
store construction and eval determinism under identical policies. It does **not** license treating
across-arm differences between *different* policies as noise-free in general.

---

## 2. Coverage is a diagnostic, never a gate

The most-corrected rule in the project. Two earlier versions were wrong in opposite directions.

**A firing floor** ("< 10 firings ⇒ inconclusive") applies a *frequency* filter to a *damage* question.
On this corpus the two are **inversely correlated**: the rarest locus (31 events) did the most damage per
occurrence (1.29 linked failures each), while the most common (543 events) did the least (0.149). A floor
discards the highest-value anchors first — and rare-but-costly is exactly what a local anchor is for,
since a global prompt cannot target it without paying on every turn.

**A coverage fraction** ("≥ half its mined exposure") is not well defined. Measured on one completed run,
the same anchor reached **100 % of affected turns (24/24)** and **9 % of events (24/277)** — same anchor,
same run, two denominators an order of magnitude apart. Any threshold on that is a choice of denominator
dressed up as a criterion.

So the liveness test asks only: **did the arm test the thing that was mined?** Coverage then describes the
anchor's reach, which informs the design of its *successor* — not the acceptance of this one.

---

## 3. Stopping: S1–S5

The loop has no natural end. It will always produce a rank 1. Three failure modes it must be protected
against: chasing exhausted loci, chasing loci too small to measure, and accumulating ineffective anchors.

Checked **cheapest-first: S3 → S1 → S2 → S5 → S4.** S3 and S1 are free; S4 requires having run arms.

| rule | fires when | rationale |
|---|---|---|
| **S3** no admissible action | every live cell for the locus is excluded — structurally, or by this line's constraints | the locus is **identified but not actionable**. Record it; do not pursue it |
| **S1** coverage floor | top-ranked *unsettled* locus links **< 10 %** of failing queries | 10 % of a ~200-case residual is ~20 cases, and a sign test needs ≥ 9 net. Below this a *perfect* anchor cannot produce a distinguishable result — **the measurement cannot answer the question being asked** |
| **S2** settled saturation | `settled / raw_events ≥ 0.5` | the incumbent already acts on most occurrences. What remains is a **coverage question for an existing anchor** — a parameter change, not a new mechanism with its own maintenance surface |
| **S5** unmined dominates | unmined residual exceeds the largest mined candidate | the bottleneck is the **signal vocabulary**, not the policy. The next move is a new detector, not another anchor |
| **S4** diminishing returns | 2 consecutive rejects at the top of the ranking, **or** 2 consecutive accepts each < +1 pp | rejects ⇒ the ranking is no longer selecting actionable work; fix the **miner**, not the next locus. Small accepts ⇒ each anchor is code, a gate, and maintenance; three +0.5 pp anchors are worse than one +4 pp |

**The 10 % floor is derived from the corpus, not from taste.** That derivation is the template to redo for
your own benchmark: take your residual size, ask what net gain a sign test needs at that n, and set the
floor where a perfect anchor would still be undetectable.

### What stopping does *not* mean

- Not "the method failed." S1/S2 firing means the addressable-by-this-signal-family work is done.
- Not "no headroom remains." It means no **anchorable** headroom *at the current signal resolution*.
- **Not permanent.** A new detector, a new incision point, or a different backend re-opens the loop.

That last point is not theoretical: it is exactly what happened at
[T4](../rounds/T4_exhaustion/) — every locus S1-STOP and S2-STOP, S5 fired, the vocabulary expanded, and
A4 followed at +3.63 pp.

### The two terminal states that must not be conflated

```
STOP                 nothing survives  AND  unexplained mass is small
                     -> the residual is genuinely addressed

EXPAND_ATTRIBUTION   nothing survives  AND  unexplained mass DOMINATES
                     -> the bottleneck moved from policy to signal; not done, blind
```

They look identical from the ranking and demand opposite responses. `phase_switch.py` decides between
them quantitatively, from the already-frozen thresholds, rather than by a human noticing the ranking
looks thin — because at that exact moment the temptation is to relax a threshold, and a threshold
relaxed after seeing the ranking is not a threshold.

---

## 4. What we deferred, and why

Shipped as a table rather than as silence. Deferred is **recorded in the ledger**, not discarded.

| deferred | evidence | rule | status |
|---|---|---|---|
| **A2.W** — write-side `not_found` | 5 occurrences; its mechanism is A1's own (A1 already rescues ~97 % of core-full events) | **S1** | in ledger. It is an A1 *coverage* question, not a new anchor |
| **8 `unattributed` `not_found` occurrences** | no write-side explanation; admissible set empty by construction | **S3** | in ledger |
| `format/field/malformed_value` (rank 5 at T2) | 12 linked failures = **6.0 %** of residual | **S1** | below floor |
| `capacity/…` re-proposal (rank 3 at T3) | **372/724 = 51 %** already settled by A1 | **S2** | saturated |
| `size/item/exceeds_per_item_limit` | fell **30 → 9** linked failures under occurrence-level re-labeling | **S1** | the re-label is what disqualified it |
| `short_episode` | 77 % coverage, precision **0.57** — fires on 89 of 117 successes | detector screen | rejected |
| `answered_after_read` | precision **0.50** | detector screen | rejected |
| `single_read_only` | precision **0.47** | detector screen | rejected |
| **"retrieved OK but answered wrong"** | **82 %** of one measured residual | — | **deliberately not anchored** — see below |
| relaxing support thresholds for high-precision/low-support fixes | rank 2 is exactly that shape | — | deferred **not mid-line**, or T1–T5 stop being comparable |

### The largest residual class is deliberately left alone

*"Retrieved OK but answered wrong"* is 82 % of one measured residual — by far the biggest single bucket.
It is **not anchored**, and the reason is the discipline in miniature:

> It is an observed **outcome class**, not an attributed **mechanism**.

No answer-formation signal gets added unless counterfactual attribution shows a **recurring decision
point** that the existing signals cannot represent. Wanting a signal for the biggest bucket is not the
same as having one, and a detector keyed on "the answer was wrong" is a restatement of the label, not a
condition checkable at a decision point.

### Rejected on measurement after A4

Recorded because rejections are results. The accepted anchors that followed — A5, A7 and A8 — are in
this repo; **A6 is deferred** and its record is in [`../rounds/A6_deferred/`](../rounds/A6_deferred/).

| candidate | verdict | why |
|---|---|---|
| low-similarity reprompt *(an early candidate that shared the "A5" name — not the accepted A5)* | **rejected** | mechanism fired 22/22 and reformulated every time — **2/22 converted**, ~5 % of the predicted ceiling. *A good detector is not a good anchor target.* |
| W1 entry-length condense | **rejected** | the repair worked (71 → 44 errors, 29/29 fact-preserving) and still lost 3 cases on one cell against a **zero** variance floor |
| B1 futility escalation | **rejected** | **−6.60 pp, p = 0.0352** |

**B1 is the most informative failure on this line.** It tried the *opposite* of the thesis: a generic
"reconsider and try again", delegating the choice of remedy to the model. Under a hard capacity
constraint the model's chosen remedy was **wholesale clearing** (`core_memory_clear` 11 → 33) rather than
targeted repair. The accepted anchors go the other way — A1 and A3 *execute* a fixed action with no model
involvement, and A4's nudge is safe only because its alternative space has two branches, one of which is
"answer directly".

> Learn a narrow signal and execute a constrained remedy. Do not ask the LLM to invent the recovery
> action.

---

## 5. Standing rules, and the failure each came from

Every one of these is a scar. The *why* matters more than the rule.

| rule | the failure it came from |
|---|---|
| **Attribution prunes the action space; measurement chooses the policy.** | Attribution proposed `suppress` for a locus by keying on the action *name*; inspecting all 17 events showed 0 unnecessary and 15/17 benign. The candidate was cancelled. |
| **Mechanism plausibility is not evidence.** | Three separate cases of a mechanism-plausible candidate losing — including one argued repeatedly for on mechanism grounds. State the mechanism as *motivation*, then let the arm decide. |
| **Passing the criterion is not enough — the lever must have engaged the target.** | A winner passed all four terms while firing on a different backend's contract; the target residual grew. |
| **Never fix a losing arm after seeing it lose.** | That is selecting on the outcome. Freeze the result as run; a fixed version is a **new candidate measured fresh**. (A4 v2 qualified because its correction came from a specification change, not from inspecting v1's loss.) |
| **Freeze the exposure RULE, not the exposed-case count.** | A confirmation was once launched whose detection threshold (±24.03 pp) exceeded the effect being tested (11.54 pp) — unfalsifiable before it started. Read `n_exposed` and judge adequacy *before* looking at any outcome. |
| **Report raw counts beside percentages.** | One episode firing 20× dragged apparent acceptance from 85 % to 59 %. Concentration hides in rates. |
| **Prefer behavioural verification of the live path over source inspection.** | Source scans passed throughout a period when a suppressed step was being silently dropped from the trajectory. Same class of bug recurred three times. |
| **Derive a learned anchor's scope from observed incidence, never assert it.** | An anchor was declared for three backends; it fires on one (228 events vs 0), and another backend's error text cannot even match its strings. |
| **A measured noise floor bounds only what it measured.** | A zero floor for identical policies was wrongly generalised to across-policy comparisons. |
| **Never difference raw accuracies across jobs.** | Two controls read 29.04 % and 24.36 % purely from cell composition, while agreeing to **1 flip in 228** on shared cases. Only within-job paired comparison is licensed. |

---

## 6. Applying this to your benchmark

The thresholds here are **derived from this corpus**, not universal. Port the derivations, not the
numbers:

1. **S1's floor** — measure your residual size, ask what net gain a sign test needs at that n, and set
   the floor where a perfect anchor would still be undetectable. On a 387-case corpus that was 10 %.
2. **Your variance floor** — run identical-policy replicates *before* any candidate. If it is not
   ~0, every "single loss is a real loss" statement here weakens, and your harm tolerance must widen.
3. **Freeze criteria before launch**, with a hash. Every round in [`../rounds/`](../rounds/) ships one.
   A stopping rule chosen after seeing the result is a rationalisation.
4. **Pre-state what would make the arm uninformative** — zero firings, wrong contract, inadequate
   exposure — so an inconclusive run is recognisable as inconclusive rather than read as a null.

## See also

- [`INCISION_POINTS.md`](INCISION_POINTS.md) — the 3 × 4 grid; where S3's structural exclusions come from
- [`../rounds/T4_exhaustion/`](../rounds/T4_exhaustion/) — S1/S2/S5 firing together, and what followed
- [`ANCHOR_LEARNING_PROTOCOL.md`](ANCHOR_LEARNING_PROTOCOL.md) — why acceptance closes an iteration
- [`../rounds/T4_exhaustion/STOPPING_CRITERIA_FROZEN.md`](../rounds/T4_exhaustion/STOPPING_CRITERIA_FROZEN.md) — the frozen original
