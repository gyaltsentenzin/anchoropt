# The acceptance rule

One rule per **class**, applied to every anchor. There is no earlier "pre-registered" tier and later
"qualified" tier — every accuracy anchor was audited against the same four criteria, and the one that
failed them is not in the stack.

## Two acceptance classes

The stack holds anchors of two kinds, with **different criteria**. An efficiency anchor cannot satisfy
`Δ_train > 0` **by construction** — its whole point is that behaviour is unchanged — so applying the
accuracy rule to it is a **category error**.

| class | accepted when |
|---|---|
| **accuracy anchor** | the four criteria below: net train > 0, net dev ≥ 0, attribution + causally-validated mechanism, safety |
| **efficiency anchor** | **no task loss** + **provably useless work removed** + **no new harm**. A **0 pp delta is a PASS**, not a null result. |

This distinction is stated first because omitting it had a concrete cost: it is how the efficiency
anchor came to be dropped from the frozen policy **without anyone deciding to drop it.** See
[the efficiency class](#the-efficiency-class-and-why-accepted--active) below.

## The accuracy rule

Accept an accuracy anchor iff **all four** hold:

1. **Discovery / train** — positive **NET** effect versus the current incumbent, on the paired cases.
2. **Independent dev** — **no AGGREGATE regression** versus the same incumbent.
3. **Attribution** — the improvement is concentrated where the anchor can actually engage, and the
   intended mechanism is **causally validated** from the anchor's own decision-point telemetry.
4. **Safety / invariants** — no catastrophic or prohibited failure mode introduced.

Then **freeze it, re-evaluate, and re-mine residuals from scratch.**

### Criterion 4 made explicit: the three destructive counters

Added 2026-09-21, as a **clarification with its rationale recorded**, not a rewrite. The stack summary
below reads "0 clears and 0 removes added", and taken literally that contradicts the accepted A5, whose
own validation records **30 evictions dispatched**. `adapter.is_destructive` is a LEXICAL counter
(`is_clear or is_remove`) and has never been the criterion. What the rule has always counted is
**removals that lose information**. Three counters, reported separately:

| counter | rule |
|---|---|
| **clears added** | **CATEGORICALLY FORBIDDEN.** Any added `core_memory_clear` / `archival_memory_clear` fails criterion 4 outright. |
| **information-losing removes added** | A removal whose content is **not** verified present elsewhere in **live state at the moment of removal**. Any is a failure. |
| **verified relocations** | A removal preceded by a destination write **confirmed in live state**, with the surviving copy confirmed. **Reported and bounded, and not counted as a destructive remove.** |

A verified relocation and a destructive information-losing removal are **different acts**, and conflating
them is what made this ambiguous. The bound matters as much as the verification: an unbounded sequence of
verified relocations is a wholesale clear in slow motion, so a mechanism must cap them per episode (A5
uses 3, and so does the relocation primitive).

This forbids strictly **more** than the lexical reading in one direction — a removal without verified
preservation now fails explicitly instead of being argued about — and it leaves clears categorically
forbidden. **No accepted verdict changes:** A5 passes on the invariant it was already measured against
(`copies_remaining >= 1`, 30/30).

## What this is not

It is **not** `gains > 0 AND zero losses`. It is **net > 0 on train, net ≥ 0 on dev, with mechanism
validation.**

The distinction is the whole point: this gives **monotonic policy improvement at the AGGREGATE level,
not monotonicity on every individual case.** An anchor that fixes nine cases and breaks two is an
improvement. A zero-loss requirement rejects it — and would also have rejected **A5**, which is
measured at **+8.33 pp on dev with a 30/30 causally-validated mechanism.**

## Report the shape, not only the net

**Every verdict states the paired `(gains, losses)` beside the net delta.** Not as a gate — because the
*shape* is diagnostic, and a net figure cannot distinguish two very different objects:

| anchor | dev gains | losses | discordant | net | shape |
|---|---:|---:|---:|---:|---|
| **A6** | 8 | 9 | **17** | −1 | churns 20% of dev to no net effect |
| **A7** | 5 | **0** | 5 | +5 | strictly additive |

On net delta these sit about 7 pp apart and a tolerance would decide between them. On paired structure
they are not close, and no plausible ε changes either verdict.

**That is how "one stochastic flip decides an anchor" is answered — by making the shape visible, not by
absorbing it into a tolerance.**

## Why no ε, and why ε = 1 case was rejected on process grounds

ε = 0 is deliberately conservative and acknowledged as such: it will occasionally reject a genuinely
neutral anchor. That is the accepted cost of not accumulating anchors that quietly trade dev cases.

ε = 1 case = **1.19 pp** was considered — and that is *exactly* A6's dev regression. Adopting it after
seeing the table would be choosing the threshold that admits the single anchor it decides. Defensible
in the abstract; fitted in this instance. Recorded so the reasoning is auditable rather than implicit.

## p-values are diagnostic, never prescriptive — and here they are especially weak

The standing policy, now with the reason specific to this benchmark: **it is deterministic.**
Byte-identical execution logs were measured across independent sharded runs spanning days, jobs and
hosts. **There is no sampling process over which a null distribution is defined.** A sign test on 84
fixed cases asks "how surprising is this split if the flips were coin tosses" — but they are not coin
tosses. They are the deterministic consequence of a policy change on a fixed corpus.

The one genuine stochastic channel anywhere in the stack is **A7's compaction call** (1 of 26
compactions decoded differently). That is one anchor's LLM call, not a corpus-level sampling
distribution, and it licenses no inference from a p either.

**Report the paired counts and the mechanism.** A7's dev sign-test p = 0.062 is recorded as a
diagnostic only; the reason to trust 5 gains / 0 losses is that all five are gains with an identified
mechanism.

## The audit

Deltas versus the incumbent at each point. Train n = 303, dev n = 84.

| anchor | 1. train net (g/l) | 2. dev net (g/l) | 3. attribution | 4. safety | verdict |
|---|---:|---:|---|---|---|
| **A1** core-full reroute | stack floor | stack floor | — | — | in stack (no isolated delta) |
| **A2** failed-lookup reprompt | **+0.99 pp** (5/2) | **+0.00 pp** (1/1) | — | no new clears | **ACCEPT** |
| **A3** redundant-write suppress | **+3.96 pp** (18/6) | **+3.57 pp** (3/0) | — | no new clears | **ACCEPT** |
| **A4** zero-call reprompt | **+3.63 pp** (16/5) | **+3.57 pp** (7/4) | all 3 backends; on-target +9.17 / off-target 0.00 | no new clears | **ACCEPT** |
| **A5** archival evict + retry | **+1.32 pp** (7/3) | **+8.33 pp** (9/2) | **vector +17.50; kv 0.00, rec_sum 0.00** | 0 clears, 0 removes | **ACCEPT** |
| **A6** entry-length reroute | +1.65 pp (14/9) | **−1.19 pp** (8/9) | vector **−2.50** — on-target and NEGATIVE | 0 clears | **REJECT (2 and 3)** |
| **A7** rec_sum compaction | **+2.64 pp** (21/13) | **+5.95 pp** (5/0) | **rec_sum +25.00; kv 0.00, vector 0.00** | 0 clears | **ACCEPT** |

**The accepted stack is A1–A5 + A7.** Safety across the stack: **0 clears and 0 removes added** by any
accepted anchor.

**A6 fails criterion 3 independently of criterion 2** — its on-target backend is *negative*. The anchor
engages exactly where intended and makes things worse there, which is a cleaner statement of the
mechanism than the aggregate delta gives. See [`../rounds/A6_deferred/`](../rounds/A6_deferred/).

### A5's mechanism, causally validated

Criterion 3 asks for causal validation from the anchor's own telemetry, not a plausible story. Dev,
vector shard, from A5's own per-decision records:

| check | result |
|---|---|
| evictions dispatched | **30** |
| blocked write **landed** after eviction | **30 / 30** |
| `copies_remaining >= 1` (lossless invariant) | **30 / 30** |
| `copies_before` range | **2–3** (k ≥ 2, so one copy always survives) |
| eviction call errors | **0** |
| correctly **declined** when no victim existed | **24** |

A5 was originally installed on an **explicit override of a frozen zero-loss clause** (3 train losses,
net +4). A reader weighting pre-registration absolutely was right to call that unaccepted **as a process
criticism**. Under this rule it is settled on evidence. **The override was procedurally wrong and
substantively right** — both are recorded, and the dev result does not erase the process failure.

## Train and dev — and why the second split is called dev

| split | n | role |
|---|---:|---|
| **train** | 303 | discovery — mine loci, fit and select the remedy |
| **dev** | 84 | **the accept/reject criterion** |

**Using dev to accept and reject anchors makes it a VALIDATION set, not a test set.** The reported dev
figure is therefore partly selected-on and is **no longer a clean generalization estimate** — A6 was
dropped *on* dev evidence, which is exactly the selection.

An untouched final number would require a **third reserved split**, which does not exist. Stated so the
limitation is not silently forgotten.

Two further properties of the dev fold, both measured rather than assumed:

- It is **one domain chain per backend**, so it measures **domain transfer** rather than
  within-distribution generalization. Any anchor whose effect depends on chain multiplicity or on how
  deeply the store saturates will move on it. A leave-one-chain-out protocol is the honest fix and is
  deferred, not done.
- At n = 84, **one case is 1.19 pp.** That is why the paired shape is mandatory rather than optional.

## The efficiency class, and why ACCEPTED ≠ ACTIVE

**E1 is an accepted anchor on the efficiency objective**, and it is currently **not active**. Those are
different statements and conflating them is a mistake this project has already made.

### Three requirements specific to the class

Each was learned from a failure here, and none is a restatement of the accuracy rule:

1. **The saving must be in EXECUTED calls, not proposals.** The first implementation memoized *after*
   `_execute` and saved nothing — a behaviour change dressed as an efficiency one. Verify with a
   counting stub that `_execute` invocations for the targeted calls are **0**.
2. **Behaviour must be preserved by construction, not by hope.** Return the tool's **verbatim** result
   so the model's view is identical. Two rejected predecessors prove the point: one injected a trailing
   *user* message (−1.65 pp); another returned a *new* tool string (+0.33 pp, but destructive calls rose
   84 → 90). Both changed behaviour because both said something new. **The acceptance evidence is a
   0 pp delta with ZERO flips — which is stronger than a small positive delta,** because a positive
   delta on a behaviour-preserving change means the intervention leaked into behaviour.
3. **The redundancy metric must count what the gate changes.** Counting redundant calls in the decoded
   proposals reads **16 → 16**, because withheld calls still appear there by design. Counting
   **executed** redundant calls reads **16 → 0** on the same data.

**A behaviour-preserving anchor does not trigger a re-mine.** Re-mining looks for loci whose
trajectories shifted, and a 0 pp / zero-flip arm has none. Its locus leaves the **efficiency residual**,
which is tracked separately from the accuracy residual.

### Status: accepted, not active

E1's acceptance record is **8/8 criteria passed**: +0.00 pp with 0 gains / 0 losses, 16 calls withheld
and **0 sent**, executed redundancy **16 → 0**, splice integrity 16/16, destructive counts and A5's copy
invariant bit-identical.

But its gate key `on_memoized_deterministic_call` is **absent from the frozen policy** with
`gate_default = false`, so it is **off**, and it fires **0 times** in every run behind the accuracy
numbers in this document — **none of them are affected either way.**

**This looks like an assembly oversight rather than a decision** — a single missing gate key in the same
policy build that also carried an env-only-flag bug. It cannot be proven from artifacts, so it is
recorded as an open item rather than asserted. Restoring it requires **re-confirming the three
requirements above against the CURRENT stack**, because the last confirmation was against A1–A4 and
A5/A7 have since changed the trajectories it operates on. The opportunity still exists: 15–23
repeated-identical remove calls per shard, plus 49 not-found results on kv train.

### Why this matters for the call-count metric

A planned evaluation compares **total LLM calls** against baseline under the harness's
`max_steps_per_turn = 20` (the official `MAXIMUM_STEP_LIMIT`). **That metric is the efficiency class's
own acceptance criterion, so the efficiency anchors must be live for it to mean anything.**

Measured on the current stack (query-phase steps per case, budget never binding — `force_quit` = 0
everywhere), **the stack is not uniformly cheaper**:

| arm | split | n | total steps | vs control |
|---|---|---:|---:|---|
| **A7** rec_sum | train | 109 | **239** | **−22 steps (−8.4%)** vs the incumbent's 261, while gaining **+7.34 pp** |
| A7 kv / vector | train | — | identical to control | 0 — backend-gated |
| **A5** vector | dev | 40 | **105** | **+14 steps (+15%)** vs control's 91, to buy **+17.50 pp** |

A7 *saves* calls because landing a blocked append ends a retry loop; A5 *spends* calls (evict, then
retry) to buy accuracy. **Both facts belong in the metric** — reporting only the saving would be
selective.

Two caveats for that measurement: `steps` covers the **query phase only**, so the prereq phase — where
A5 and A7 mostly fire, and where E1's 16 withheld calls lived — needs separate counting; and the 20-step
ceiling is never reached, so call count is a free-running cost rather than a truncation artifact.

## The baseline rule

**Score every candidate against the ACCEPTED STACK, never against its own host anchor.** An
arm-vs-arm delta measures "did my patch improve my patch". Worked example: A6's best variant reads
**+7.14 pp** against bare-A6 and **−1.19 pp** against the incumbent — the baseline choice flipped the
sign of the conclusion.

## History of this rule

It replaced a **train-only** rule under which A6 was admitted, with A6's dev regression reclassified as
a "generalization diagnostic" — precisely the move this rule forbids. An intermediate formulation was
read as "zero losses", which would have rejected A5; the current wording says **net**, explicitly.

Dropping A6 was therefore **not** an override of the old rule, as an earlier note claimed. Under the
governing rule there was nothing to override: A6 should never have been accepted.

**Consequence to follow up:** any other anchor admitted on train-only evidence that dev contradicts
should be re-examined on the same grounds. A1–A5 were not audited that way in this pass.
