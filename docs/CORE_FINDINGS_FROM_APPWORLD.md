# Core findings from the AppWorld round

**One interface change to core (shipped), one integration gap, and a set of core gaps that no existing
gate closes.** Measured evidence for every claim is in `APPWORLD_RESULTS.md` (not in this branch, see [`docs/RESULTS_POLICY.md`](../docs/RESULTS_POLICY.md)); this
file records only what bears on core, so core can be changed without reading the benchmark record.

Everything here came out of scoring arms across two models on AppWorld `train`/`sh_heldout`, with repeats.
Two findings frame the rest:

* **Pairing against ONE incumbent draw was the dominant bias, and it pointed opposite ways on the two
  models.** Re-scoring every archived pass against every available draw: minimax's treatment averages
  **+5.33** (positive in all six arm×draw comparisons) against **+2.50** as reported versus the highest of
  its three draws, with its control at −1.83 and flipping sign — a 7.17-task separation, and the
  best-supported result here. qwen's treatment and control are **identical at +0.75**, and a second qwen
  baseline draw (67/90, SGC 19/30 against draw 1's 64/90, 14/30) retracts the +6.6pp SGC that had looked
  like the project's only primary-metric win: the arm's 16/30 sits inside the baseline's own spread. So one
  extra baseline run removed a false positive and uncovered a real effect that single-draw pairing had been
  hiding. SGC stability cannot be assumed either, and this is stronger than first reported: minimax scores
  53.3 on the three draws we used but **40.0–53.3 across all six that exist** (spread 13.3pp = 4 scenarios,
  identical configuration), and qwen-local swings 46.7 → 63.3 with 21 of 90 tasks flipping between two
  identical-config baseline runs. Our three draws were the joint SGC maximum, so every ΔSGC first reported
  here was against a best case.
* **Both models promoted, and neither promotion was supported by its evidence.** `net > 0` combined with
  firing-attribution structurally prefers arms that barely fire (§3, §4), and firing rate turned out to
  predict whether a verdict survives re-measurement.
* **Which grounding an action uses sometimes matters as much as where it fires, and core cannot tell when.**
  On minimax's `pagination_unbounded`, two variants at one coordinate — same boundary, signal, action and
  matched firing — average **+2.5** and **−5.0** over two passes each (§6). On qwen's `generation_truncated`
  the same comparison is **+2.0** vs **+1.0**, with the generic variant reaching *all three* of the
  treatment's stable gains: content contributes nothing there. The discriminator is the signal's shape — an
  **absence** the model can fix by retrying versus a **successful wrong action** that must be redirected —
  and core has an unused hook for exactly that declaration (C8). So variant choice is load-bearing on some
  coordinates and irrelevant on others, and core currently picks by enumeration order in both cases.

* **The best arm on the search objective does nothing on the benchmark's primary metric, and the reason is
  structural.** AnchorOpt's objective is a net over **episodes**; AppWorld's headline metric is a conjunction
  over **groups** of episodes (scenario goal completion = mean of `min(per-task success)`). The arm measured at
  attributable **+5.33** and raw **+4** moves SGC by **+0.0**, because at scenario level it completes 4 and
  breaks 4. Core has no grouping concept in the search at all, and no hook through which an adapter could
  supply one (C9, §7). The group-level version of core's existing headroom screen predicts this from the frozen
  incumbent alone — downside 7 intact scenarios against an upside ceiling of 6 completable ones — while the
  per-episode screen it already computes reads "worth trying".
* **A refinement can be expressible and still unreachable.** The adapter change that made turn position
  expressible at the gate boundary is verified working — 15 of 39 synthesizable predicates there are over
  `step_number`. The search proposes none of them, because Φ expansion is anchored to the boundary the
  schedule is *currently on*, and the gate boundary is reached only by the frontier sweep, which by design
  installs nothing (C10, §8). Adapter gap closed; core gap exposed underneath it. And the refinement the
  round actually needs is a *conjunction* — `targets_paginated_api ∧ step_number > k` — which is unreachable
  for a second, independent reason: the conjunction pool is ranked by balance, so the rare atom that names the
  situation (19.1% of states) is excluded, while declaring the new field simultaneously cut the conjunction
  budget from 17 slots to 7 (C11, §9). **Adding observability made the search cheaper in atoms and poorer in
  conjunctions**, and nothing reported that.

The single change made to core so far is §1, four lines in `signal_grammar.py`. Everything else in this
file is either a proposal or an adapter/protocol item, categorised in §10.

## 1. Interface change: a field declaration may raise its own atom cap

`anchoropt/learning/signal_grammar.py::atoms_for`:

```python
- for v in enum[:MAX_ATOMS_PER_FIELD]:
+ cap = getattr(decl, "max_atoms", None) or MAX_ATOMS_PER_FIELD
+ for v in enum[:cap]:
```

**What it was doing.** `MAX_ATOMS_PER_FIELD = 4` is a flat global cap on equality atoms per enum field,
whose stated purpose is *"keeps an enum field from dominating the grid."* It truncates by **position in
the declared tuple**.

**Why that is wrong for some fields.** The AppWorld adapter declares `error_kind` with a 13-value error
taxonomy. Under a flat 4, only the first four values could ever become atoms, and *which* four was
decided by nothing but the order they were typed in. Nine classes were unreachable, silently — no error,
no warning. For a field whose values **are** the discriminating vocabulary rather than incidental
categories, a flat cap truncates the vocabulary into uselessness.

**Why this shape rather than raising the constant.**

* The guard's original intent survives. An incidental enum — 200 app names — still cannot flood the grid.
  Raising the global constant would have weakened the protection for every field in every adapter.
* Core cannot know whether an enum's values are incidental or discriminating. The adapter can, so the
  adapter declares it. No benchmark knowledge enters core.
* Duck-typed via `getattr`, so every existing declaration is untouched.

**Verified.** `target_app` (enum longer than 4, declaring no cap) still yields exactly 4 equality atoms;
`error_kind` with `max_atoms=13` yields 13. 257 core tests pass (`test_signal_lang`,
`test_signal_expansion`, `test_candidate_search`, `test_boundary_search`, `test_search_space_funnel`,
`test_structured_search`, `test_core_genericity`, `test_adapter_parity`), plus 28 contract tests and 9
adapter probes.

## 2. Integration gap: the AppWorld driver hand-rolls the verdict

Core has two layers. The AppWorld driver only uses the first:

| level | entry point | returns |
|---|---|---|
| 1 — search | `structured_search.optimize_residual()` | search *states*: `IMPROVED`, `NO_BENEFIT`, `REALIZABLE_UNMEASURED` |
| 2 — verdict | `termination.classify_residual()` | earned *outcomes*: `PROMOTED`, `INSUFFICIENT_SUPPORT`, `EVALUATION_INVALID`, `NO_MEASURED_IMPROVEMENT`, `SEARCH_EXHAUSTED` |

`benchmarks/appworld/run_round.py` imports `ArmResult`, `next_coordinate` and `optimize_residual`, and
reads `outcome.state` directly as a verdict. It never imports `termination`.

**This is a departure from an established, test-enforced convention, not an open question.**
`scripts/self_evolve_cycle2.py:1408` does call `classify_residual`, and
`tests/test_cycle2_thinness.py:85` asserts *"the driver must derive the outcome, not hand-roll it."*

What bypassing it costs, independent of section 3:

* **`Validity` / `channel_integrity` never run.** For the promoted arm, `channel_integrity` would have
  reported *"2/5 gains on a fired case (40%)"* — a warning about attribution quality that nobody saw.
* **The negative-result vs non-result distinction is lost.** Core marks only `NO_MEASURED_IMPROVEMENT`
  and `SEARCH_EXHAUSTED` as negative results; everything else is explicitly *"a NON-result: the question
  stands."* An AppWorld arm that fired 5 times and produced **zero** attributable flips — no information
  at all — was recorded as `NO_BENEFIT`, i.e. as evidence the mechanism does not work.

## 3. The genuine core gap: nothing gates on the arm's own power

This is the correction that matters, and it is easy to get wrong. **Core's existing gates would also
have promoted the unsupported arm.** Measured against the real numbers:

```
promoted arm: 6 firings, raw gains 5 (2 on fired), raw losses 9 (0 on fired), attributable net +2

classify_residual(support=29, accepted=True)            -> PROMOTED
channel_integrity(gains=5, gains_on_fired=2, firings=6) -> True
    "2/5 gains on a fired case (40%) (no threshold declared; reported, not enforced)"
```

Two reasons neither fires:

* **`support` is the residual family's size, not the arm's firing count.** The existing driver passes
  `support=top.support`. Our residual was 29 cases, far above `MIN_SUPPORT = 5`, so the floor passes.
  `support=4` would return `INSUFFICIENT_SUPPORT` — but that describes a tiny residual, not a rarely
  firing arm.
* **`channel_integrity`'s `min_fraction_on_fired` defaults to `None` deliberately.** Its docstring is
  explicit: on a real round 11 of 22 gains sat off fired cases, so any threshold above 0.5 would
  reclassify a result already on the record — *"a call for whoever owns the claim, not a number to pick
  inside a helper."* With `None`, only the zero-firings check applies, and 6 > 0.

So core today has **no minimum-firings and no minimum-detectable-effect gate on the arm being scored**.
`improves` is a caller-supplied boolean, which cannot express "insufficient evidence" even when the
caller knows it — the information (`cases_fired`, `interventions_executed`) is already carried on
`ArmResult` and discarded at the interface.

**Proposed core change.** A verdict should be unobtainable without telling core what a resolvable effect
looks like: an arm-level support floor (minimum `cases_fired`) and a minimum `|net|` relative to a
caller-declared noise floor, returning `INSUFFICIENT_SUPPORT` — a non-result that leaves the frontier
open — rather than `IMPROVED`/`NO_BENEFIT`. This is benchmark-independent; the thresholds are the
caller's to declare, as `channel_integrity` already argues for its own.

## 4. Why it matters: `net > 0` plus firing-attribution prefers arms that barely fire

The mechanism is general, not an AppWorld artifact.

Attribution restricts gains and losses to episodes where the controller actually fired — correct, since
a flip on an episode it never touched cannot have been caused by it. But a rarely firing arm has almost
no fired episodes on which attributable **losses** can accumulate, so the filter removes nearly all its
losses while keeping some of its gains:

| arm | firings | raw net | attributable net | outcome |
|---|---|---|---|---|
| `execution_failed/reprompt` | **53** | −6 | **−3** | rejected |
| `unknown_api_targeted/reprompt` | **5** | −9 | **0** (nothing attributable) | rejected |
| `unknown_api_targeted/suppress` | **6** | −4 | **+2** | **promoted** |

The promoted arm kept 0 of 9 losses and 2 of 5 gains. Of its 6 firings, **2 helped, 0 hurt, 4 changed
nothing** — had two landed on regression tasks rather than residual tasks it reads −2 and does not
promote. The 53-firing arm, the only one with enough events to say anything, was the one rejected.

**Attribution makes the number honest without making it powered.** Any firing-filtered metric combined
with a `net > 0` rule inherits this bias.

Two compounding factors, both measured:

* **Single-draw incumbent.** minimax `train` repeats are 58, 60, **61**; arms are scored against the 61.
  All three arms are raw-negative largely because of that. Pairing against a mean over repeats rather
  than one draw would remove it.
* **A signal can be too precise to measure.** Correctly narrowing `unknown_api_targeted` to fire only on
  cells with an API call dropped it to 5–6 firings per 90 tasks, which puts its maximum conceivable
  effect below the 2–5 task noise floor. Precision and measurability trade off, and nothing in the loop
  currently notices.

## 5. Second core proposal: the divergence index, a strictly stronger admissibility filter

Episode-level analysis of the qwen `sh_heldout` arm (90 arm + 90 baseline transcripts) produced a test
that subsumes firing-attribution. The reasoning is general: a controller acting at boundary `k` can only
have caused a difference beginning at step `k+1`. Let `d` be the first step at which the arm's and the
incumbent's trajectories differ. Across all 14 outcome flips in that measurement:

| | count |
|---|---|
| `d < k` — already diverged before the controller could act | **11** |
| `d == k` — even the failing step differs | 2 |
| `d == k+1` — the signature the intervention would leave | **0** |
| `d > k+1` — acted-on step identical, diverged later | 1 |

**13 of 14 flips are not identified.** Today `run_round.py` admits a flip if the episode fired — a
necessary condition for causation. "Fired **and** the arm's prefix up to the firing step matches the
incumbent's" is strictly stronger, costs nothing (both transcripts are already on disk), and on this
measurement admits **0 of 14**.

This is unidentifiability, not proof of no effect: two trajectories can differ cosmetically while the
controller is still the operative difference at the decision point. But a verdict should not be built on
flips whose provenance cannot be established, and core is where that admissibility rule belongs — it is
benchmark-independent, needing only "which step did the controller act at" plus a trajectory prefix
comparison.

Why the filter matters here: the same measurement contains a **zero-treatment control**. Of the 17
episodes the controller never fired on, **2 (12%) flipped outcome anyway** — extrapolating to ~8.6 of the
14 observed flips from sampling alone.

## 6. Third core proposal: at a coordinate, promote the BEST variant rather than the first

**Measured first, then read against the code.** One arm, one boundary, one signal, one action — differing
only in the *grounding variant*, the text the REPROMPT injects:

| variant at `post_generation_pre_exec/pagination_unbounded/reprompt` | fired | gains | losses | net |
|---|---|---|---|---|
| `page_through_all_results` | 39, 42 | 6, 6 | 2, 5 | **+4, +1** → mean **+2.50** |
| `check_api_before_running` | 41, 39 | 0, 5 | 8, 7 | **−8, −2** → mean **−5.00** |

Firing is identical to within sampling (39/42/41/39), so the boundary, the signal and the trigger are held
fixed. The separation **across variants at a single coordinate is 7.5 tasks in the mean** — wider than any
spread this project measured across *coordinates*.

Two details matter for how strongly to read this. The variants reach **overlapping** gains: the control
reaches 3 of the treatment's 5 stable gain tasks, so the better variant is not unlocking something
unreachable. And each variant is self-inconsistent in the opposite place — the treatment's gains replicate
task-for-task (5 of 6) while its losses share nothing across passes; the control's losses replicate (6 of
8/7) while its gains share nothing. So what the variant governs is **collateral harm and its concentration**,
not access. That is still a property worth choosing on, and it is invisible to a single pass: one pass of
the control showed 0 gains and supported the stronger, wrong conclusion that the gains were variant-exclusive.

**And the same comparison on the other model says the opposite.** qwen's `generation_truncated` arm,
measured identically (two passes each, matched firing 50/47 vs 53/49):

| variant at `post_generation_pre_exec/generation_truncated/reprompt` | fired | net | mean |
|---|---|---|---|
| `emit_short_runnable_cell` | 50, 47 | +5, −1 | **+2.00** |
| `check_api_before_running` (the same generic text) | 53, 49 | +1, +1 | **+1.00** |

One task apart, with the generic variant reaching **all three** of the treatment's stable gains and proving
the *more* stable of the two. Here the argmax buys nothing and the boundary is the whole finding.

The discriminator is what the signal names. `generation_truncated` names an **absence** — no runnable cell
was produced — and any reprompt buys another turn to produce one, so the content is nearly irrelevant.
`pagination_unbounded` names a **successful wrong action** — the cell ran and returned page 1 — and a
reprompt that does not name the correction reproduces the same page. That is a property the adapter knows
and can declare (C8), and without it core cannot tell a coordinate where variant choice is load-bearing
from one where it is noise.

Core already represents variants as first-class: `action_contract.variant_required` keys required eta
*per variant*, and `external_evaluation.py:260` records the variant on the arm. The search enumerates
them — our manifest carried three variants per signal as separate arms. So this is not a supply gap.

The gap is in how one is chosen. `structured_search.py:499-513`:

```python
best = None
for arm in built:
    obj = _eval(arm)
    ...
    if best is None or _objective_key(obj) > _objective_key(best):
        best = obj
    if improves(obj):
        att.state, att.promoted, att.objective = IMPROVED, arm, obj
        ...  # promotes THIS arm and leaves the coordinate
```

`best` is tracked correctly as the argmax, and it is what the `NO_BENEFIT` path reports — so a bad variant
does **not** condemn a coordinate, which is right. But promotion fires on the **first** arm satisfying
`improves`, and `built`'s order is the grounder's emission order. Two consequences:

1. **Where several variants improve, which one promotes is decided by enumeration order**, not by measured
   objective. The argmax is already computed one line above; promotion does not consult it.
2. Because promotion also **halts** the search (C5), the first improver becomes the round's answer and the
   better variant is never scored.

On our data the harmful variant is negative on both passes and `improves` rejects it, so nothing went
wrong *here*. The finding is that nothing in the mechanism prevents it: a 7.5-task separation between
variants means the
difference between first-improving and best-improving is as large as the difference between a promotion and
a rejection.

**Proposal.** Evaluate all arms at a coordinate, then promote `best` if `improves(best)` — a change of
ordering, not of criterion, using state core already maintains. It costs nothing when one variant
improves, and it removes an order dependence that is currently silent. This composes with C2: the support
gate decides *whether* to promote, the argmax decides *which*.

**Why it generalises.** Any adapter with more than one grounding per action has this exposure, and the
more expressive the action space the wider the spread. It is also the part of the search a benchmark cannot
fix for itself — an adapter can order its groundings, but it cannot make core prefer the best one.

## 7. Fourth core proposal: the objective counts episodes; the benchmark counts groups of episodes

`improves(res)` receives `res.net` — gains minus losses over **episodes**. Grepping `anchoropt/` for
`scenario`, `group_key`, `episode_group` or `grouping` returns hits only in `attribution/` and
`evolve_memory/`, never in the search objective. Core has no notion that episodes might be grouped, and no
adapter can give it one: there is no hook to declare a grouping key.

AppWorld's primary metric is **scenario goal completion** — the mean over scenarios of `min(per-task
success)`. A conjunction over a group. The consequence, measured:

| | attributable net | raw passed | **SGC** |
|---|---|---|---|
| `pagination_unbounded` / `reprompt:page_through_all_results` | **+5.33** (mean of 6 draw comparisons, sign-consistent) | 65 vs 61 (**+4**) | **53.3 vs 53.3 (+0.0)** |

The arm is the strongest thing measured in this project on the search objective and does **nothing** on the
benchmark's own metric. Not because the objective is wrong about the episodes — it is right — but because
the two objectives are not monotone in each other. Counted at the scenario level the arm **completes 4
scenarios and breaks 4**. A gain moves SGC only if it completes a group; a loss moves SGC only if it breaks
an intact one. Everything else is invisible to the metric and fully visible to `net`.

**This is not an attribution artifact.** Raw and attributable agree (+4 vs +5.33); the disagreement is
between *episode-level* and *group-level* accounting, which no amount of better pairing or more draws fixes.
A claim previously made here — that SGC is **draw-invariant** — is **retracted**. It rested on three draws
all scoring 16/30; three further draws of the identical stock configuration on the same split score 14/30,
13/30, 14/30 and 12/30, so SGC swings **13.3pp (4 scenarios)** across six measurements, and our three draws
were the joint maximum. Rescored against the six-draw mean the treatment reads SGC **+4.5** and its matched
control **−0.5** — same direction as the attributable separation, still inside a noise band ~3× its size.
The retraction does not weaken C9: episode-level and group-level accounting still disagree (the arm is +4 raw
tasks and completes 4 scenarios while breaking 4). It changes only the claim that the group metric is the
quieter one. It is not — it is noisier, which makes a group-level *screen* more valuable, not less, because
30 scenarios cannot be measured precisely enough to discover the same fact by running arms.

**Proposal, in two parts.**

1. **Let the adapter declare an optional grouping**, e.g. `group_of(case_id) -> str | None`, exactly as
   `policy_class_for` (C8) lets it declare a signal's shape. Absent the hook, everything behaves as now.
2. **When a grouping exists, let the objective see it.** `ArmResult` already carries the per-case gains and
   losses needed to compute a group-level net; nothing new must be measured. Then `improves` can be given a
   group-level net, or both, and a caller can require that an arm not break an intact group.

**And screen it first, which is the cheap half.** §3's headroom screen already computes the per-episode risk
side — `headroom_task` (upside ceiling) against `loss_exposure` (episodes where the signal fires and the
incumbent passes) — and is marked *advisory, does not affect the search*. The group-level version of the same
computation, from the frozen incumbent's logs and no arm runs at all:

```
signal fires on 39 episodes
  17 of them in already-INTACT scenarios, spanning  7 of 16 intact scenarios   <- can only go down
  of the 14 broken scenarios, only 6 have EVERY failing task in the firing set <- only 6 can be completed
  => downside 7 : upside 6                    measured outcome: 4 broken, 4 completed
```

The screen predicts the measured result at the right granularity, before 90 episodes are spent. Its
per-episode counterpart does not: it reports ceiling 15 against exposure 23, which reads as "worth trying".

**Why it generalises.** Any benchmark whose headline metric is a conjunction over grouped episodes has this
gap — SWE-bench instances within a repo, BFCL multi-turn trajectories, τ-bench task families, WebArena
groups. The grouping is benchmark knowledge, so it belongs in the adapter; the aggregation and the screen are
benchmark-independent, so they belong in core. This is the same split C8 uses, and it keeps core general.

## 8. Fifth core proposal: Φ expansion is anchored to the current WHERE, so a swept boundary is never widened

`structured_search` orders boundaries and, when HOW is exhausted at the current one, does two things in
sequence. First the **grounded frontier sweep** — deliberately "a WHERE move under a FIXED Phi", which
installs no signals, because a grounded attribution is evidence and a synthesized predicate is a guess.
That ordering rule is right, and this document is not proposing to change it. Then, once the frontier is
exhausted, STEP 3 calls:

```python
names, preds = _expand_phi_at(boundary_arg, runtime=runtime, states=states, existing=...)
```

`boundary_arg` is the **current** boundary — not the frontier boundary that was just found wanting. So a
boundary reached only via the sweep gets its arms built from the seeded Φ and never gets its alphabet
widened at all.

Measured on round 4. The run exhausted HOW at `post_execution`, swept the gate boundary
`post_generation_pre_exec` (15 arms built, 10 measured, none improved), then widened Φ **at
`post_execution`** — producing 23 new signals, all `result_chars_*` and `error_kind_*`. Zero candidates over
the gate boundary's newly available fields, although a direct probe confirms
`_expand_phi_at(post_generation_pre_exec)` on the same 1254 records yields **39 predicates, 15 of them over
`step_number`**, including the exact conjunction the loss analysis motivated.

The schedule does reach the gate boundary eventually — `i -= 1` walks earlier — but only after the 23
`post_execution` refinements are all measured and rejected: ~2,070 episodes, ~17 hours of scoring. And it
widens at the boundary whose measured arm is the *worst* in this record (−1.33) and whose local ceiling is 4,
while the gate boundary's ceiling is 15.

**Proposal.** Two independent changes, either of which unblocks this:

1. **Expand at the boundary whose frontier was just exhausted**, not only at the anchor. The sweep already
   knows which boundaries it visited (`frontier_boundaries` records them precisely so provenance stays
   readable); expansion can follow the same list.
2. **Let the headroom screen order the schedule.** Core already computes per-signal ceilings and marks them
   advisory. Ordering boundaries by ceiling rather than by fixed position would have put the gate boundary
   first on this corpus, with no new measurement.

**Why it generalises.** The fixed latest-to-earliest walk is a core scheduling choice with no benchmark
content. Any adapter with a usable earlier boundary and a weak late one pays the same cost, and the more
boundaries an adapter declares, the longer the queue in front of the one that matters.

## 9. Sixth core proposal: the synthesis budget penalises observability, and excludes the atoms conjunctions need

`signal_grammar.synthesize` builds atoms, sorts them by **balance** — `abs(fired/total - 0.5)`, closest to
a 50/50 split first — then forms conjunctions from `keep[:12]` and returns `(atoms + conjunctions)[:cap]`,
with `MAX_SYNTHESIS_CANDIDATES = 40`.

Atoms are enumerated first and conjunctions get whatever is left. At the AppWorld gate boundary, measured on
the 418 mined states:

| | atoms | discriminating | conjunction slots left | conjunctions formed | **cut** |
|---|---|---|---|---|---|
| after A12 (`step_number` declared) | 35 | 33 | **7** | 43 | **36** |
| before A12 | 25 | 23 | **17** | 38 | 21 |

**Declaring a field more than halved the number of conjunctions the search can consider.** That is a perverse
incentive pointed at exactly the thing an adapter is supposed to do: every observable it adds consumes the
budget that conjunctions over its *existing* observables were using. A12 paid this cost to buy `step_number`
atoms, and the trade was invisible.

**And the seed pool excludes the atoms a conjunction is for.** Ranked by balance, the top 12 at this boundary
are nine `code_chars`/`step_number` threshold atoms, `target_app_is_api_docs`, and `not_proposes_write`. The
field that actually names the situation — `targets_paginated_api`, 80/418 states, 19.1% — ranks **16th** and
never enters the pool, so it appears in no conjunction at all. Five of the eight declared fields never reach
one:

```
in a surviving conjunction: code_chars, proposes_write, step_number
never in one:               api_path_unknown, passes_page_index, target_api, target_app, targets_paginated_api
```

Note that `target_app` *is* in the pool (as `target_app_is_api_docs`, rank 4) and still reaches no surviving
conjunction. That is a second, independent filter: conjunctions are sorted by **firing count descending**
before the cap applies, so the broadest conjunctions survive and any conjunction involving a selective atom
is cut even when the atom made the pool. Both orderings — pool entry by balance, survival by breadth — push
the same way, toward conjunctions that fire on most states.

The refinement this project actually wants — *fire only when the call targets a paginated API **and** the
episode has developed past turn k* — is therefore still inexpressible, even after A12 made turn position
available. Both fields are declared; their conjunction is unreachable.

**Why balance is the wrong criterion for the conjunction pool.** A conjunction's purpose is to *narrow* a
signal, and the atom worth narrowing with is a rare, situational one. Balance-ranking selects for atoms near
50/50, which are the least selective available, and systematically discards the rare atoms that identify
*which* situation is at hand. `code_chars_gt_97p0` fires on exactly 50.0% of states and means nothing;
`targets_paginated_api` fires on 19.1% and means the thing the signal is about.

**Proposal.**

1. **Budget atoms and conjunctions separately** instead of sharing one cap with atoms enumerated first, so
   declaring an observable cannot shrink the conjunction search.
2. **Seed the conjunction pool by field coverage, not by balance alone** — admit at least one atom per
   declared field before filling the remainder by rank, so no declared field is structurally unable to
   participate in a conjunction.
3. **Stop ranking surviving conjunctions by firing count**, which selects for the broadest and so undoes the
   narrowing a conjunction exists to perform.

All three are orderings over what the grammar already produces. No new synthesis, no benchmark knowledge.

**Why it generalises.** This is C1 one level up. C1 was a flat per-field atom cap truncating a discriminating
vocabulary by declaration order; this is a flat global candidate cap truncating the conjunction space by an
ordering that does not track usefulness. Any adapter declaring more than a handful of fields at one boundary
hits it, and the richer the adapter, the more of its conjunction space disappears.

## 10. Ownership split — every finding, categorised

### anchoropt-core (general, benchmark-independent)

| # | finding | status |
|---|---|---|
| C1 | flat `MAX_ATOMS_PER_FIELD` truncates a discriminating enum by declaration order | **fixed** — per-field opt-in, §1 |
| C2 | no arm-level power gate: `support` is the residual family's size, not the arm's firings; `channel_integrity`'s fraction threshold defaults to `None`. Core promotes a 6-firing arm. **Now demonstrated empirically: the 6-firing arm measured +2, then −1 on repeat, then 0 on held-out; the 62-firing arm measured +1 twice. Firing rate predicts whether a verdict survives re-measurement.** | **proposed**, §3 |
| C3 | `improves` is a boolean, so a caller cannot express "insufficient evidence" even when it knows — while `cases_fired` is already carried on `ArmResult` and discarded at the interface | **proposed**, §3 |
| C4 | attribution is necessary-condition-only; the divergence index is strictly stronger and belongs in core's admissibility rule | **proposed**, §5 |
| C5 | promotion halts the search, so an *unsupported* promotion terminates a round before refinement is reached — C2's cost compounds | consequence of C2 |
| C6 | `incumbent_token` is `os.path.abspath(evaluation_path)` — a path, not a content hash. Two different baseline runs at the same path share a token, so `load_results`' cross-incumbent guard cannot distinguish them. Same class as A5: the pairing identifier does not cover what makes runs comparable. A hash over the evaluation plus the rendered agent config closes both. | **proposed** |
| C7 | promotion fires on the **first** arm at a coordinate satisfying `improves`, in grounder emission order, while the argmax `best` is computed one line earlier and used only on the `NO_BENEFIT` path. Measured: two variants of one arm — same boundary, same signal, same action, matched firing (39/42 vs 41/39) — average **+2.5 and −5.0** over two passes each, a wider separation than anything measured across coordinates. Conditional: the same comparison on qwen's `generation_truncated` is +2.0 vs +1.0, so the argmax matters on some coordinates and not others, and which is which follows the signal's shape (C8). Combined with C5 (promotion halts), the first improver becomes the round's answer. | **proposed**, §6 |
| C9 | **the objective counts episodes; AppWorld's primary metric counts groups of episodes.** `improves` sees `res.net` over cases, and core has no grouping concept anywhere in the search (no `group_of` hook, no `scenario`/`group_key` in `anchoropt/learning`). Measured: the arm at attributable **+5.33** and raw **+4** moves SGC by **+0.0**, because at scenario level it completes 4 and breaks 4. Not an attribution artifact — raw and attributable agree; the gap is episode-level vs group-level accounting. The group-level headroom screen predicts this from the frozen incumbent alone (downside 7 intact scenarios : upside 6 completable) while the per-episode screen reads "worth trying" (ceiling 15 : exposure 23). Adapter declares the grouping, core aggregates and screens — the same split as C8. | **proposed**, §7 |
| C10 | **Φ expansion is anchored to the current WHERE, so a boundary reached only by the frontier sweep is never widened.** `_expand_phi_at(boundary_arg, ...)` is called with the anchor boundary, not the swept one. Measured: round 4 exhausted `post_execution`, swept the gate boundary under seeded Φ (15 built, 10 measured, none improved), then synthesized 23 new signals **at `post_execution`** — none over the gate boundary's fields, although a direct probe yields 39 predicates there, 15 over `step_number`. The gate's refinements sit behind ~2,070 episodes of scoring at the boundary with the worst measured arm (−1.33) and local ceiling 4, versus the gate's ceiling 15. Fix: expand at the boundaries `frontier_boundaries` already records, and/or let the advisory headroom screen order the schedule. | **proposed**, §8 |
| C11 | **the synthesis cap penalises observability, and the conjunction seed pool excludes the atoms conjunctions are for.** `synthesize` returns `(atoms + conjunctions)[:40]` with atoms enumerated first, so declaring a field consumes conjunction budget: at the AppWorld gate boundary A12 took the alphabet 25→35 atoms and the conjunction slots **17→7**, cutting 36 of 43 formed conjunctions. Separately, the pool is `keep[:12]` ranked by *balance* (`abs(fired/total − 0.5)`), so `targets_paginated_api` — 19.1% of states, and the field that names the actual situation — ranks 16th and appears in no conjunction at all; 5 of 8 declared fields never reach one. The refinement the round needs (`targets_paginated_api ∧ step_number > k`) is therefore *still* inexpressible although both fields are declared. This is C1 one level up: a flat cap truncating by an ordering that does not track usefulness. Fix: budget atoms and conjunctions separately, and admit one atom per declared field to the pool before filling by rank. | **proposed**, §9 |
| C8 | core declares an optional `policy_class_for(signal)` hook (`policy_family.py:134`, `anchor_policy_opt.py:233`) and no adapter here implements it. The distinction we had to discover by hand — a signal naming an **absence** (no runnable cell was produced) versus one naming a **successful wrong action** (the cell ran fine and returned the wrong thing) — is exactly a policy class, and it now predicts three separate things measured here: which headroom ceiling is valid (A10), whether the action's *content* carries any effect at all (§6: 7.5-task variant separation on the second kind, 1 task on the first), and therefore whether searching grounding variants at that coordinate is worth paying for. One declaration the adapter already knows the answer to. | **proposed**, strongest of the set |

### AppWorld adapter (benchmark-specific)

| # | finding | status |
|---|---|---|
| A1 | `error_kind` taxonomy written for error kinds this benchmark barely produces; the dominant classes all collapsed to `other_error`, making the field constant across every gain/loss case | **fixed** |
| A2 | `error_kind` declared with an empty enum, so the grammar saw only `truthy`/`falsy` — and `truthy` *is* `execution_failed`, making refinement inexpressible rather than unfound | **fixed** |
| A3 | **`execution_failed` fires on generation truncation.** `no_code` is `finish_reason: "length"` with `content: None` — produced *before* execution by the LM hitting its token cap. 22/73 firings; the injected "read the error above" refers to nothing. | **fixed** — split out as `generation_truncated` at the gate; firing opportunities fell ~60% on qwen (372→151) |
| A4 | `other_error` swallows HTTP 422, which mixes `bad_parameter`-shaped validation errors with `precondition_unmet` ("the payment card has expired") — opposite remedies, one of which no retry can fix | **open** |
| A5 | `_score_one_arm` validates the incumbent's *dataset* but not its `model_config`, which is how an arm ran at `max_completion_tokens: 8192` against a baseline measured at `None` | **open** |
| A6 | the firing marker records only a count, so recovering *where* and *on what* an arm fired requires re-deriving it from transcripts | **open** |
| A7 | the driver hand-rolls the verdict from level-1 search states instead of calling `classify_residual`, against a convention `test_cycle2_thinness.py:85` enforces for the other driver | **open**, §2 |
| A8 | signal precision vs measurability: correctly narrowing `unknown_api_targeted` dropped it to 5–6 firings per 90 tasks, below the noise floor | design tension, no fix |
| A9 | the arm itself was redundant — the baseline already recovers after 78% of first errors unprompted. Screenable from the baseline alone, before any arm is scored. | **fixed** — `headroom.py`, advisory, computed from the incumbent's own logs |
| A10 | the screen then judged every signal on the ceiling valid only for error-shaped signals, and printed UNRESOLVABLE for `pagination_unbounded` (local 1, task 15) — the one arm whose gains replicate. The dual ceiling existed in `headroom.py` and had not reached its caller. | **fixed** — verdict now takes `max(local, task)`; see C8 |
| A12 | **the only measured correlate of an arm's harm is inexpressible at the boundary that needs it.** `step_number` and `consecutive_errors` are declared only at `pre_generation`, so at POST_GENERATION_PRE_EXEC the grammar has just `api_path_unknown`, `code_chars`, `passes_page_index`, `proposes_write`, `target_api`, `target_app`, `targets_paginated_api`. Gains and losses separate on *first-firing position* -- gains at mean call 6.7/7.5, losses at 2.5/4.6, i.e. the arm fires EARLIER where it harms, the opposite of what total episode length suggested -- so the refinement is "fire only after the episode has developed a few turns", and it cannot be written at that boundary, never mind searched. Re-gating the measured episodes at k>=4 gives +5/+3 against the actual +4/+1 (post-hoc on 8-11 events; a reason to measure, not a result). The reason it is adapter-side rather than core: core searched every conjunction the adapter declared, and this term was never declared. **Not one line** — `step_number` was *declared* at `_PRE` but never populated, because `gate_fields_from_code(code)` saw only the cell text; the fix is four edits across `adapter.py`, `residual.py` and `agent_hooks.py`, adding `_PG` to the boundary tuple, declaring the field in the `_PG` block so `synthesis_fields` returns it, and threading the turn index from both call sites that already hold it. | **fixed and verified**: `synthesis_fields(_PG)` carries it, 418/418 mined gate states populate it, and `_expand_phi_at(_PG)` yields 39 predicates of which 15 are over `step_number` — including `step_number_gt_4p0__and__code_chars_gt_75p0`, the conjunction the loss analysis motivated. **The search still never proposes one**, for two reasons that are both core-side: the schedule never expands Φ at that boundary (C10), and the conjunction that would actually encode the refinement — `targets_paginated_api ∧ step_number > k` — is unreachable because the conjunction pool is balance-ranked and `targets_paginated_api` ranks 16th (C11). A12 delivered the atoms; the useful conjunction needs C11. |
| A11 | minimax is not error-shaped at all: 15 of 29 failures contain no erroring cell, 25 of 36 error cells are self-repaired `auth_error`, and failing episodes are *shorter* than passing ones (10 vs 13 median steps). Every signal declared before this named something that had already failed, so none of them had purchase on this model. | **fixed** — `pagination_unbounded`, the first completeness signal |

### Protocol / measurement (cross-cutting, neither core nor adapter)

| # | finding |
|---|---|
| P1 | arms are scored against a **single** incumbent draw -- now measured as the dominant bias in both directions: it suppressed minimax's treatment by 2.83 tasks (+2.50 reported vs +5.33 across draws) and inflated qwen's by 1.25, manufacturing the only apparent SGC win in the project. `runs/multi_draw_nets.py` re-scores against every draw and reports whether the sign holds. Original note follows. minimax `train` repeats are 58/60/**61**; scoring against the 61 makes every arm raw-negative. A mean over repeats removes it. |
| P2 | the model server is not reproducible: 34/90 first cells differ at `temperature 0` with a fixed seed (vLLM continuous-batching nonassociativity), so a paired design at n=90 is not identified. |
| P3 | ~~no arm has been measured with repeats~~ — superseded. Repeats are now built into the scoring job, which archives an arm's episodes between passes because `skip_if_finished` otherwise returns a byte-identical number that looks like perfect reproducibility. Measured consequence: firing rate predicts replication (6-firing arm: +2, −1, 0; 62-firing arm: +1, +1). |
| P4 | held-out validation must pin and verify the **full** model config, not just the dataset (see A5). |
| P5 | evaluation wall-clock is dominated by filesystem contention, not compute (0.6 s/task quiet vs 3.4 min/task under 11 concurrent jobs). Serialise jobs; do not parallelise the evaluator. |
| P8 | an arm's **absolute** score is not recorded anywhere: `arm_results_pass*.json` carries only the attributable net and `accuracy_delta_pp` (= net/n), no passed/TGC/SGC, and no arm-level evaluation file is written. Recomputing it from the archived episode trees changed the reading of the whole round -- the arm with the strongest attributable result moves SGC by −1.6 while the arm explained away by its control moves it by +6.6. Whatever a round reports, it should report the benchmark's own primary metric beside the attributable one. |
| P7 | the liveness signal for a running scoring job is the per-episode `lm_calls.jsonl` line count. Directory mtimes, the arm dir's mtime and the job's own stdout mtime all lag or track unrelated writes, and reading them produced two wrong "this job is hung" diagnoses on the same job in one night. LSF `stat` stays RUN and CPU time stays flat for a job blocked on inference, so neither distinguishes slow from dead either. |
| P6 | a measurement is tied to the GPU it was taken on. Qwen3.6-35B-A3B crashes vLLM's Sm90 TRT-LLM MoE GEMM on H100 (5 jobs of 5) and runs on A100-80GB (6 of 6); the fix is to pin the GPU model, not to switch MoE backend, because changing the kernel changes the arithmetic and every qwen number here was taken on A100. Same class as A5 and C6: what makes two runs comparable is wider than what the pairing guard checks. |

**Reading of the split.** One core defect was real and is fixed (C1). The proposals fall into two groups
that are worth keeping distinct, because they are answers to different questions:

* **C2–C4, C6 — refusing to issue a verdict the evidence cannot support.** An arm-level power floor, an
  `improves` that can say "insufficient evidence", the divergence index, a pairing token that covers what
  makes runs comparable. All are about *not over-claiming*, and they arose from both promotions being
  unsupported.
* **C7–C8 — choosing well among things the search already enumerates.** Promote the best grounding variant
  rather than the first; let an adapter declare a signal's shape so core stops applying an error-shaped
  bound to a completeness signal. These arose from the opposite direction — from the one arm that *did*
  work, and from nearly discarding it.

Everything else is adapter or protocol. Core turned out to be more complete than the adapter's use of it
— `variant`, `policy_class_for`, `channel_integrity` and `classify_residual` all already exist and all
went unused — but not complete enough to have blocked the unsupported promotions on its own. The honest
summary is that most of what we needed was already in core and not wired up, and the two places core is
genuinely missing something (C2, C7) are both about *how a verdict is earned* rather than about
representation.

## 11. Most "adapter" findings are generic failure modes with adapter-specific content

This is the part that bears on making AnchorOpt work across models and harnesses. Fixing A1–A9 inside
`benchmarks/appworld/` fixes AppWorld. It does not stop the next adapter making the same mistakes,
because **nothing in core detects any of them.** Separating the content from the failure mode:

| adapter finding | AppWorld-specific content | generic failure mode |
|---|---|---|
| A1, A2 | which error strings occur | a declared signal field with **no variance** on the residual population |
| A3 | `"No code available to execute."` | a signal **spanning two layers** — condition produced pre-execution, declared post-execution |
| A4 | HTTP 422 bodies | one class conflating **opposite remedies** |
| A5 | `max_completion_tokens` | an **unpaired comparison** that passes the dataset check |
| A9 | ReAct self-correction | an intervention whose **ceiling is below the noise floor**, knowable from the incumbent alone |

Four corresponding detectors, each using information core already has or could require:

1. **Degenerate-field check.** At `propose`, core holds the mined records. Test every declared enum field
   for variance across them and refuse when one is constant: *"`error_kind` takes a single value across all
   1254 records, so no predicate over it can discriminate."* This catches A1 and A2 **before any arm is
   scored** — the cheapest possible point.
2. **Boundary-truthfulness for signals.** Core already fails closed on exactly this question for Phi
   expansion (`CORE_SEAM_states_at.md`: a predicate validated on states from the wrong boundary "can look
   discriminating and then fire zero times at runtime"). The same question applies to a *signal*: is the
   field it reads produced at the boundary it is declared at? A3 is that defect, one level up.
3. **Config fingerprint in the paired-comparison contract.** The adapter supplies an opaque hash of
   whatever makes a run comparable; core compares it exactly as it already compares `incumbent_token`, and
   refuses on mismatch. A5 becomes unrepresentable.
4. **Declared headroom.** Core cannot compute "the incumbent already recovers after 78% of first errors" —
   that needs host logs. But it can **require** the adapter to report the trigger condition's
   self-resolution rate, and refuse to score an arm whose ceiling falls below the declared noise floor.

Detectors 3 and 4 are the same principle as C2 and C3: **core requires the adapter to declare what makes
a result believable, and refuses to issue a verdict without it.** That principle is what generalises
across models and harnesses; the per-adapter patches do not. It also costs adapters little — each is one
declaration, and the adapter is the only party that knows the answer.

A useful property of this framing: every detector fails *early and loudly* rather than producing a number
that looks like a result. A1/A2 cost a full round and two scoring jobs before anyone noticed the field was
constant; the degenerate-field check would have cost one `propose` call.

## 12. Consequence worth recording

Promotion halts the search. Both models promoted on unsupported arms at round 1–2, so **the 21 refined
`error_kind` candidates the taxonomy fix made expressible were never scored.** Reaching them requires the
underpowered arm *not* to promote — so the adapter fix cannot be evaluated until the core gate in §3
exists. The two are coupled, not independent workstreams.
