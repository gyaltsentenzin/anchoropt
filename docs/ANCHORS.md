# The anchors, in detail

One page per anchor's worth of detail, kept out of the README so the results there stay readable. The
README has the progression table and a one-line description of each anchor; this is what each one
actually does, what it cost, and what it taught.

Every anchor was judged by the same rule — [`ACCEPTANCE_RULE.md`](ACCEPTANCE_RULE.md). Numbers live in
[`../rounds/anchors.py`](../rounds/anchors.py) and are pinned to this file by test.

The train steps below are against the **measured** native baseline (30.03%). A long-quoted 29.04% had no
artifact behind it, and correcting it changed A1's step from +4.62 to +3.63 and A2's from +2.31 to +0.99.
Nothing above A1 shifted — every later delta was already measured against measured neighbours.

| | anchor | locus | decision point | action | train | dev |
|---|---|---|---|---|---:|---:|
| T1 | **A1** | `capacity/container/no_remaining_capacity` | post-execution | reroute | +3.63 | — |
| T2 | **A2** | `existence/identifier/not_found` | post-execution | reprompt | +0.99 | — |
| T3 | **A3** | `permission/identifier/duplicate` | post-gen / pre-exec | suppress | +3.96 | — |
| T5 | **A4** | `no_tool_call_at_all` | post-gen / pre-exec | reprompt | +3.63 | +9.52 |
| T6 | **A5** | `capacity/container/no_remaining_slots` | post-execution | reroute | +1.32 | +8.33 |
| T7 | **A7** | `size/blob/append_would_exceed_cap` | post-execution | reroute | +2.31 | +5.95 |
| T8 | **A8** | `capacity/container/clear_proposed_at_capacity` | post-gen / pre-exec | suppress | +1.65 | +0.00 |
| — | **E1** | `redundancy/deterministic_outcome/known_absent_target` | post-gen / pre-exec | suppress | 0.00 | — |
| — | **A6** | `size/item/exceeds_per_item_limit` | post-execution | reroute | *deferred* | *deferred* |

---

## A1 — reroute a capacity-blocked write

The core container is full and the write is refused. A1 re-addresses it to archival, which has room.

Rank 1 on the first mine by **linked downstream loss**, not event volume: 81 of 215 residual failures
traced back to it. That ranking choice matters — the most *frequent* error is often a symptom, and
ranking by frequency would have sent the first anchor somewhere else.

A1 is also the anchor that **created most of the later work**. Its rescue writes under a key the model
does not know about, which produces the read-side residual A2 addresses and the duplicate-write
residual A3 addresses, and eventually saturates the container A5 and A8 have to manage. That is the clearest
argument in the project for re-mining after every install rather than working down a fixed ranking.

Full record: [`../rounds/T1_A1_capacity/`](../rounds/T1_A1_capacity/).

---

## A2 — after a failed read, point the retry at a reachable destination

The value was stored but the read did not find it. A2 injects a reprompt telling the model to search
archival.

**One error string, two causes.** Attribution split `not_found` into 39 read-side cases (this anchor)
and 5 write-side cases (deferred below the support floor). Treating the error label as the anchor would
have merged two different failures into one remedy.

**All 39 are self-inflicted** — the value sits exactly where A1 put it.

The destination was attested on **resolving**, not on not-erroring: `archival_memory_retrieve` demands
an exact key match and A1 sanitises keys, so the exact-match retrieve *would not have errored and would
not have found anything*. The fuzzy key search is the reachable destination. "It didn't error" is not
evidence a repair works.

Full record: [`../rounds/T2_A2_not_found/`](../rounds/T2_A2_not_found/).

---

## A3 — suppress a redundant re-write before it executes

The model, unaware A1 already archived a value, issues its own write for it. The store answers *"Key
name must be unique."* A3 cancels that call before it runs.

**All 59 collisions are against keys A1 itself injected**, and 54 of 59 carry a token-identical value.
So this is not the model being wrong in general — it is the model being uninformed about a repair made
on its behalf.

**The predicate is deliberately exact, never semantic**: same key *and* normalised-identical value. It
fires on 55 of 59 and declines 4 — three paraphrases and one genuinely new fact — rather than buy 7%
more coverage at the risk of suppressing real information. When the cost of a false positive is silent
data loss, precision beats coverage.

This is also the first anchor at the **commitment gate** (post-generation, pre-execution), which is the
only point where cancelling a call is free: after execution the duplicate has already been rejected and
the step is spent.

Full record: [`../rounds/T3_A3_duplicate/`](../rounds/T3_A3_duplicate/).

---

## A4 — when the model is about to answer without looking, ask it to look

The fact was retrievable and the model did not consult memory. A4 injects a non-directive reprompt.

**The clearest single demonstration that the decision point is part of the anchor.** A4 v1 and v2 share
a signal, an action family, and a *byte-identical* injected text. They differ only in where they fire:

| | trigger | fired | delta |
|---|---|---|---|
| v1 | `step_count == 0`, pre-generation | 303/303 | **−4.95 pp** |
| v2 | zero tool calls, post-gen / pre-exec | 59/303 | **+3.63 pp** |

v1 could not observe "about to answer without calling a tool" — that fact does not exist before
generation — so it fired everywhere and paid the cost on 244 episodes that did not need it.

**Deliberately non-directive.** 23 of the 48 exposed episodes have *nothing* in the store, where telling
the model what to look for invites fabrication. It says "check memory", not "the answer is in memory".

The detector was chosen on **precision against passing queries** (0.80), beating three more intuitive
candidates that fired on successes about as often as failures: `short_episode` 0.57,
`answered_after_read` 0.50, `single_read_only` 0.47.

Full record: [`../rounds/T5_A4_no_tool_call/`](../rounds/T5_A4_no_tool_call/).

---

## A5 — free a slot without losing a fact

A1's relocation target eventually saturates, so A1's own repair has nowhere left to put a payload. A5
frees one slot, but **only where eviction is provably lossless**: the container holds the same value
twice, so removing one copy destroys nothing. Then it retries the original write **verbatim**.

It is not "evict the least useful entry." No utility model exists, and inventing one at a single
decision point is the selection problem that [`SATURATION_AND_SELECTION.md`](SATURATION_AND_SELECTION.md)
shows cannot be solved there.

**Mechanism causally validated** — this is what criterion 3 asks for, and it is telemetry rather than a
story. Dev, vector shard:

| check | result |
|---|---|
| evictions dispatched | **30** |
| blocked write **landed** afterwards | **30 / 30** |
| lossless invariant (`copies_remaining ≥ 1`) | **30 / 30** |
| eviction errors | **0** |
| correctly **declined**, no victim existed | **24** |

Four steps, each of which has a documented way to get it wrong: read the **live** container (an earlier
version read its own write log, saw a median of 4 entries against a real ~50, and the arm was void);
pick a **deterministic** victim; **verify** the invariant against the store after the removal; retry the
**original** call (a synthesized retry failed on every kv firing *after* the eviction had succeeded —
destroying a duplicate and losing the write).

**The process criticism, which stands.** A5 was originally installed on an **explicit override** of a
frozen zero-loss clause: it lost 3 cases against a zero variance floor. Those losses were traced and are
*not* the eviction — `core_memory_remove` calls rose 42 → 96 once A5 began firing, i.e. the model became
more destructive on its own once the store changed under it. That is **induced model drift**, the general
cost of intervening at all.

A reader weighting pre-registration absolutely was right to call that unaccepted *as a process
criticism*. Under the governing rule it is settled on evidence. **The override was procedurally wrong and
substantively right**, and the dev result does not erase the process failure. Note also that a zero-loss
rule would have rejected A5 outright — which is exactly why the rule says *net*.

Full record: [`../rounds/T6_A5_archival_full/`](../rounds/T6_A5_archival_full/) ·
mechanism: [`../anchoropt/mechanisms/lossless_eviction.py`](../anchoropt/mechanisms/lossless_eviction.py).

---

## A7 — rewrite a single-string store to admit a refused append

One backend keeps memory as a **single string** at a 10,000-character cap, with **no second container**.
When an append would overflow, the tool rejects it and the content is lost — so neither A1 nor A6 can
help, because there is nowhere to relocate to. The only way to admit the write is to make the store
shorter. A7 rewrites the blob, then retries the append verbatim.

**It is the one anchor that calls a model at inference time.** Every other one computes its repair. That
cost is real: an added LLM call reorders the trajectory, so coverage, state keys and downstream questions
all move, and per-decision reasoning cannot bound run-level behaviour. It is also the only source of
non-determinism in the stack — 1 of 26 compactions decoded differently.

**The simplest acceptance rule won, and every added guard scored below it:**

| variant | rule | rec_sum train |
|---|---|---:|
| **v3** | fits **and** shorter than the original | **+7.34 pp** |
| v5 | + rescue retry at a numeric target | +4.59 pp |
| v6 | fit only, no length rule | +3.67 pp |
| v4 | + reject rewrites under 60% of input | **−7.34 pp** |

### The transferable finding: constrain relative to the input, not to an absolute target

The obvious reading of *v3 beats v6* is "shorter output loses less information." **That is wrong**, and
one case pins it. On `35-healthcare-5`, v3's rewrite is **1,298** chars and v6's is **3,094** — v3 is
2.4× *smaller* and it is the one that keeps the needed fact, because v6 spent its length on a nested
markdown outline while v3 wrote flat prose.

> Constrained to be **shorter than the original**, the model **rephrases** and keeps content.
> Constrained only to **fit**, it **restructures** — and restructuring costs content at any length.

So the length rule is not a length rule; it keeps the model in rephrasing mode. A ratio also measures how
much *text* survived, not how much *information* did: v3's median accepted ratio is 0.899 against v6's
0.712, and v3 retains more.

Two corollaries, both measured: **a ratio floor is actively harmful** (rejection removes the compaction
without improving it — 5 of 8 cases could only emit sub-floor output and stayed permanently blocked), and
**a numeric target does not work on this model** (it obeys the qualitative instruction and overshoots
every stated ceiling by 400–800 chars, in one direction — so report the *signed* distance, not a
pass/fail band).

### The attribution trap

For 12 losses, an automated classifier built on a capitalisation-based fact proxy reported **0
compaction-induced / 12 upstream write failures.** Manual inspection of the actual blobs gave the **exact
inverse: 9 and 0.** The needed answers were `walk`, `avocado`, `lawn`, `seven` — lowercase prose the proxy
never emits — so its "lost facts" came from unrelated tokens and every case fell into the classifier's
first branch.

> **Never let a lexical proxy assign a causal class.** A capitalisation-based fact detector cannot work
> on prose.

The `specifics()` helper is kept for *description* and gates nothing, with that constraint in its
docstring.

### Exposure, and what the confirmation run corrected

A7 is gated at dispatch to `rec_sum`, so **+7.34 pp there is +2.31 pp corpus-wide** — quote both. The
three-backend confirmation showed it is a **bit-exact no-op** off-target: 0 disagreements and
byte-identical execution logs across the 258 cases it cannot reach, which is stronger than
accuracy-neutrality. It also **corrected a spliced dev baseline** that had used a pre-A6 reference and
thereby *understated* A7's gain.

Still open, and stated as open: **19 of 21 train gains are unexplained** by any measurement performed —
not attributed to chance, unexplained. A prose-capable fact detector is the prerequisite for closing it.

Full record: [`../rounds/T7_A7_blob_overflow/`](../rounds/T7_A7_blob_overflow/) ·
mechanism: [`../anchoropt/mechanisms/blob_compaction.py`](../anchoropt/mechanisms/blob_compaction.py).

---

## A8 — stop a destructive clear when a redundant copy exists

The container is full and the model proposes to **clear it**. A8 checks whether the live container holds
an exact duplicate; if it does, it suppresses the clear, evicts one redundant copy, verifies a copy
remains, and retries the blocked write verbatim. If it does not, it **falls through unchanged**.

**It is the first anchor that prevents a destructive action rather than repairing a failed one.** Every
earlier capacity anchor waits for a refusal: A1 relocates a blocked write, A5 evicts after an add is
rejected, A7 rewrites after an append overflows. A8 fires on a *proposal*, because after the clear
executes there is nothing left to save. This is the one anchor whose value depends entirely on the
decision point being pre-execution.

### Why it is not A5 with a wider trigger

Checked rather than assumed, since "you already have an eviction anchor" is the obvious objection:

| | A5 | A8 |
|---|---|---|
| fires on | a **blocked add**, post-execution | a **proposed clear**, pre-execution |
| trigger string | matches the **vector** error phrasing | every observed event is **kv** |
| what it prevents | nothing — the write already failed | the destruction of 50 entries |

A5's trigger cannot reach these states. **A8 reuses A5's validated action with a genuinely new
trigger** — which is a good outcome rather than a coincidence: the action was already known to be
lossless, and only the decision point was missing.

### The best-validated mechanism in the stack

| property | required | measured |
|---|---|---|
| firings with per-firing detail | all | **15/15** |
| `copies_before` | ≥ 2 | **2 on every firing** |
| `copies_remaining` | ≥ 1 | **1 on every firing — LOSSLESS** |
| container size | one slot freed | **50 → 49 on every firing** |
| retry landed | true | **15/15** |
| invariant violations / eviction failures | 0 | **0 / 0** |

Each surviving copy verified against **live** state. Exposure is exactly the **3 episodes the diagnosis
predicted**, and the whole firing set is **byte-reproducible across two independent runs**.

**The victims are self-documenting:** every one is a `*_unique`-suffixed key the model invented to work
around `"Key name must be unique"`, byte-identical to its unsuffixed partner. The model created the
redundancy itself while retrying, so these are true duplicates rather than near-duplicates.

Safety: archival clears executed **5 → 4** with **15 suppressed**, `force_quit` 0, and median prereq
steps **unchanged at 19** — the check that matters, since a guard whose suppressed action gets
re-proposed indefinitely shows up as a step explosion.

### And the narrowest support in the stack

Both things are true at once, and the acceptance rests on the mechanism while the support bounds what
may be claimed:

1. **The entire effect is one cell** — all 3 exposed episodes and all 13 flips are in `kv/student`. The
   (backend, scenario) cell is the indivisible unit, so this is **n = 1 cell**, narrower than A5's
   single-backend exposure.
2. **Dev cannot confirm it.** No `kv/student` cell exists in dev and the arm is a measured no-op there,
   with byte-identical exec logs. Criterion 2 is met **vacuously** — it establishes *no harm* and
   supplies zero independent evidence. Pre-registered as a limitation before any number existed.
3. **9 gains / 4 losses**, so the arm *reorders* within the cell rather than only adding.
4. **Dedup is exhaustible.** 4 clears still executed once duplicates ran out, and clear *proposals* went
   **5 → 19**: suppressing a clear does not end the pressure, and each cycle spends one duplicate.

### The ladder it confirmed

Asked **regardless of the verdict**, and measured from the live destructive-call audit rather than from
the gate's decline reasons or a replay. Of the archival clears that still execute on the accepted
incumbent, **5 of 5 are at 50/50 capacity with no exact duplicate** — so 100% of the remaining
clear-time residual is out of this layer's reach by construction. One of the five *had* duplicates,
which A8 spent, and the pressure returned.

> **dedup → safe concat / consolidation → destructive fallback.** Layer 2 belongs at exactly those 5
> states, where the container is full and holds nothing redundant, so the only lossless route to
> capacity is *combining* entries rather than removing one.

A8 is **dedup-only by design**, so its effect is attributable in isolation; bundling consolidation into
the same arm would have made the two indistinguishable.

Full record: [`../rounds/T8_A8_dedup_clear/`](../rounds/T8_A8_dedup_clear/) ·
mechanism: [`../anchoropt/mechanisms/dedup_clear_recovery.py`](../anchoropt/mechanisms/dedup_clear_recovery.py).

---

## A9 — search the other container when the retrieve comes back weakly matched

**A9 and A2 are both read-side, and comparing them is the clearest way to see what A9 adds.** They
share a diagnosis — *the value is in archival and the read did not consult archival* — and differ on
everything else:

| | A2 | A9 |
|---|---|---|
| trigger | the read **errored** (`not_found`) | the read **succeeded**, and merely returned nothing well-matched |
| signal type | an error **string** | a similarity **score** |
| action | **reprompt** — tell the model to search archival | **reroute** — search archival *itself*, merge, return |
| who acts next | the **model** | the **system**; the model is told nothing |

**The trigger is the deeper difference.** A2 needs something to have gone visibly wrong. A9 fires where
**nothing raises at all** — a retrieve that returns five weak matches is, to the harness, a completely
successful call. That is precisely the residual class the lexical vocabulary cannot reach, and it is why
this anchor needed a structural signal rather than another error label.

What A9 does *not* share with any earlier anchor is leaving the store byte-untouched while changing what
the model is **shown**: A1, A3, A5, A7 and A8 all act on writes or destructive actions, and A2 acts by
instructing.

The diagnosis inverted the obvious reading. A core retrieve returns nothing well-matched, and the model
concludes the store does not hold the answer. In **13 of 13** verified cases the information *was*
present — in archival — and the model **never issued an archival read at all**.

```
core_memory_retrieve, best similarity_score < 0.30
  → dispatch the SAME query to archival
  → rank the UNION of both containers globally, by the harness's own scores
  → replace the weak core result AT ITS OWN INDEX with the merged top-k
otherwise → fall through unchanged
```

The containers are lopsided, which is what makes merging worth doing: `vector-finance` holds **5 core
entries against 30 archival**, so a merged top-5 ranks over 35 candidates instead of 5.

| | vector train | vector dev |
|---|---|---|
| control | 43.8% (39/89) | 35.0% (14/40) |
| **+A9** | **60.7%** (54/89) | **52.5%** (21/40) |
| delta | **+16.9 pp**, 15 g / 0 l | **+17.5 pp**, 9 g / 2 l |
| mechanism | 15/15 | 9/9 |

> **These are vector-shard figures** — read as corpus numbers they overstate the anchor by roughly 3×.
> Whole-corpus, **measured on three shards**: +4.95 pp train (159/303) and +8.33 pp dev (41/84), with kv
> and rec_sum byte-identical between arms and zero off-target firings.

### The "additive by construction" safety claim is RETRACTED

A9 was accepted partly on this: the union is ranked, so a core entry that still ranks in the top k is
retained, therefore a false firing "costs one extra read, **never a displaced correct answer**."

**That last clause is false, and this anchor's own held-out losses falsified it.** Both had
`core_topk_before = 5` and `core_entries_retained = 0` — archival out-scored core ~2×, so global ranking
evicted *every* core entry. On `89-student-9` the control answered *"sailing"* correctly from a core entry
the merge dropped; the arm answered *"gaming"* from archival. The evidence was **absent, not overlooked**.

The reasoning error is the transferable part: it argued from a property of the **mechanism** (union
ranking preserves order) to an **outcome** (a used entry survives). Order preservation says nothing about
which entries fall below the cut. Same shape as the delivery defect below — reasoning about the gate
instead of measuring downstream. And displacement is the **expected** case, since the anchor only fires
when core scored weakly, which is when archival is likeliest to dominate.

> **Corrected:** the merge returns the globally best top-k of the union. When archival out-ranks core,
> core entries are dropped. Measured trade: **15/0 train, 9/2 dev — net-positive, not non-destructive.**

The acceptance stands because the rule requires **net** dev ≥ 0, not zero losses (+7 clears it) — the
same rule that deferred A6 for a negative net. Ties still break toward core, so an *equal*-scoring
archival entry cannot evict one the model had; that narrow guarantee is all that survives. **Do not reuse
"additive by construction"** for a future merge anchor: the obvious repair is a new mechanism needing its
own criteria, since a "retain one core entry" guard would have fixed both losses *and* changed the 24
wins.

**The rejected alternative was winner-takes-container**, and it was dropped on measurement rather than
taste: firing cases that already *pass* have `core_max` median 0.159 against 0.172 for the ones that
*fail* — the same band. A container-level switch cannot discriminate, so it would swap out correct results
about as often as wrong ones.

### The delivery defect — the most expensive error in the line

**A9's first two versions computed the correct merged payload on all 78 firings and delivered it on
none.** The gate ran ~190 lines below where the prompt is assembled and wrote to the telemetry sidecar;
its write to the real list was dead code, followed immediately by `continue`.

`xcm_replaced=True` was *true* — of the sidecar. So the existing arm-identity rule needs a second half:
**firing telemetry can be right about itself and wrong about the world.**

Four downstream experiments were voided: the "+0.00 pp → the model won't use the evidence" reading, a
retrieval positive measured on the wrong object, a comprehension limit that was a re-read of absent
evidence, a dilution ruling on a discarded payload, plus a two-model probe and a sysprompt localisation
built on the same void premise.

**The tell was visible.** The pre-fix arm scored *exactly* the control's 35.0%, to two decimals, across
two independent runs — the signature of a computationally inert anchor. And the probe found the model
answered correctly on **all ten** reconstruction cells while the real system always failed. *"My
reconstruction always succeeds and the real system always fails"* is a **delivery** finding.

The fix changed **where** the result is applied and nothing else; the merge is byte-identical, pinned by
a 22-check test that **fails 11 on the pre-fix file**. General rule adopted from it:
[`CONSUMER_BOUNDARY_RULE.md`](CONSUMER_BOUNDARY_RULE.md).

### What it says about decision context

A9 issues no instruction, yet it materially improves what the model knows before it answers — through the
**data channel** rather than the instruction channel. That is a third category the reprompt-vs-execution
split does not cover, and it is a first measured answer to the question
[`OBSERVABILITY_VS_DECISION_CONTEXT.md`](OBSERVABILITY_VS_DECISION_CONTEXT.md) closes on.

Full record: [`../rounds/T9_A9_xcontainer_merge/`](../rounds/T9_A9_xcontainer_merge/) ·
mechanism: [`../anchoropt/mechanisms/cross_container_merge.py`](../anchoropt/mechanisms/cross_container_merge.py).

---

## E1 — suppress a call whose outcome is already known

The model re-issues a removal for a target it has already been told is absent. The tool is
deterministic, so the second call cannot return anything new: E1 withholds the execution and **replays
the tool's verbatim recorded result**, leaving the model's view byte-identical.

**Accepted on a different objective**, accuracy per LLM call, so a **0 pp accuracy delta is a PASS** —
and a small *positive* delta would be a warning sign that the intervention leaked into behaviour.
Applying the accuracy rule here would be a category error, since `Δ_train > 0` is unsatisfiable for an
anchor whose defining property is that behaviour does not change.

Result: **+0.00 pp with zero flips, 16 of 16 redundant executions eliminated.**

**It contrasts with A3 on exactly one axis** — what the model is left with. Both are `suppress` at the
same decision point:

| | call executes? | what the model sees |
|---|---|---|
| A3 | no | nothing — the call is removed from the record |
| E1 | no | the tool's **verbatim** recorded result |

Two rejected predecessors prove the observation matters: one injected a *user* message (−1.65 pp), and
one substituted a *new* tool string (+0.33 pp, but destructive calls rose 84 → 90 — it gained accuracy
and was rejected anyway, because it said something new).

**Accepted but not currently active.** Its gate key is absent from the frozen policy, so it fires 0
times behind every accuracy number in this repo — none are affected either way. That looks like an
assembly oversight rather than a decision and is recorded as an **open item**, not asserted. Reactivating
it requires re-confirming the three efficiency requirements against the *current* stack.

**Scope, which must travel with the numbers:** all 16 firings fall in two `vector` prerequisite episodes,
so E1 fired in **zero scored episodes**. The 0 pp is *consistent with* behaviour preservation rather than
a test of it at scale. No pre-registered spec, no dev run, numbers transcribed.

Full record: [`../rounds/E1_efficiency/`](../rounds/E1_efficiency/) ·
[`EFFICIENCY_CLASS.md`](EFFICIENCY_CLASS.md) ·
mechanism: [`../anchoropt/mechanisms/memoize_guard.py`](../anchoropt/mechanisms/memoize_guard.py).

---

## A6 — DEFERRED: the signal is real, the remedy is not ready

An over-long *entry* is refused — a length limit, not a capacity one — and A6 re-addresses it to a
container with a larger per-entry cap, plus a byte-exact duplicate guard.

**It is deferred, not accepted.** On train it gains **+1.65 pp**; on the validation split it
**regresses −1.19 pp**, and on `vector` — its only target backend — it is **−2.50 pp**. An anchor that
engages exactly where designed and makes things worse there is not a threshold question.

**The mechanism is measured, and it is the most transferable result in the project.** A6 fires 44 times
on dev while **A1's own dispatches collapse 22 → 6**: it catches the payload one error earlier and
*pre-empts* A1 rather than adding a capability. On a store at its slot cap that cannot add information —
the final stores hold 57 vs 56 distinct facts with **54 destroyed and 53 added**, the same topics in
different phrasings.

> **A capacity-shaped intervention that is really a selection intervention.** Whether it helps is a coin
> flip on query-phrasing alignment.

That is why [`displacement_check`](../anchoropt/learning/exposure.py) ships as code and why
[`SATURATION_AND_SELECTION.md`](SATURATION_AND_SELECTION.md) exists. A6 paid for both.

**Why deferred rather than closed.** The signal has not gone away: `entry_too_long` fires **55×** on
train and **102×** on dev, vector stores sit at **62–85%** capacity, **103** byte-identical duplicate
writes remain, and the paraphrase tier (15.8% of allowed writes ≥ 0.90-similar) is untouched. With A6
removed, over-long entries are **rejected and lost again** — the 1.65 pp train cost, priced in
knowingly.

Worth carrying if it is reopened: **raw byte-exact dedup beat normalized on every axis** (dev −8.33 →
−1.19 pp, facts retained **56 vs 17**), because **suppression is not monotonically good** — informative
variants matter. Four reopening conditions, and the standing instruction *not* to simply re-enable the
flags, are in [`../rounds/A6_deferred/DEFERRED.md`](../rounds/A6_deferred/DEFERRED.md).

**It cost A7 nothing.** A6 never fires on `rec_sum` — 0 firings, and that shard is bit-identical with
and without it — so A7 would have been mined identically had A6 never existed. Not a stepping stone.
