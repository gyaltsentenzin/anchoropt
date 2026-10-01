# What generalizes, and what is glue

Read this before planning to run AnchorOpt on your own benchmark. It is deliberately blunt about the
boundary, because the alternative — discovering the glue work after committing — is worse.

## The short version

**The algorithm is general. The signal extraction and the policy-space trimming are not.**

The loop — mine residuals, attribute causes, enumerate incision points, prune the action space, measure
one paired arm, re-label, re-mine — is benchmark-independent and is the transferable contribution. But
two of its steps bottom out in code that knows what your tools are called, what your errors look like,
and what your world state is:

| step | generalizes? | why |
|---|---|---|
| rank loci by linked downstream loss | **yes** | operates on a residual and a linkage, not on tool semantics |
| the 3 incision points | **yes** | a property of the generate → propose → execute loop, not of any benchmark |
| structural pruning of the action grid | **yes** at grid level | "suppress has nothing to cancel pre-generation" is universal |
| acceptance ordering (engagement → attribution → harm → benefit) | **yes** | a measurement discipline |
| S1–S5 stopping rules | **structure yes, thresholds no** | S1's 10 % floor is derived from *this* corpus's power; re-derive it |
| **signal extraction** | **NO** | needs to know which payloads are errors, which are vacuous, which are harness faults |
| **locus canonicalization** | **NO** | maps *your* error strings onto semantic loci |
| **policy-space trimming below the grid** | **partly** | locus-level feasibility depends on what *your* tools can do |
| **executing an action** | **NO** | reroute must name a real destination tool with real argument names |

## The three seams where the glue lives

### 1. Getting signals out of trajectories

The miner needs to classify each tool result into one of: *errored*, *vacuous* (syntactically fine,
semantically empty), *resolving*, or *harness fault*. Every one of those is benchmark-specific:

- **errored** — your error strings. Ours include `core memory is full`, `Key name must be unique`.
- **vacuous** — this category is easy to miss and it mattered enormously here. A search returning
  `{"ranked_results": []}` is a *successful* call containing no information. So is a BM25 result where
  every score is `0.0`. Without a vacuity detector these look like successes and the failure is
  invisible to an error-keyed miner.
- **harness fault** — an implementation crash is **not** a model failure and must never become a mined
  signal. On this benchmark an empty-memory search raised `ZeroDivisionError`, and it got mined as a
  real error contract with support 13 before being caught. Budget for this; assume your benchmark has
  one too.

**What you must write:** a classifier over your tool-result payloads. Expect it to be the first thing
you get wrong, and expect to revise it after reading real trajectories.

### 2. Trimming the policy space below the grid level

The grid-level pruning is universal (9 of 12 cells; see [`INCISION_POINTS.md`](INCISION_POINTS.md)).
Getting from ~10 to the 1–3 cells actually worth measuring needs benchmark knowledge:

- **Does a substitute destination exist, and does it *resolve*?** Not "does it avoid erroring" —
  attest on resolving. Here a destination looked viable at 11/11 "clean" where clean meant *did not
  error*, and 9 of those 11 returned nothing.
- **Can the arguments be mapped mechanically?** A reroute is only deterministic if `COPY`/`RENAME`
  suffices. Ours needed per-destination argument names, and emitting `key=` universally would have
  built an invalid call and been misscored as "the substitute did not help."
- **What does your configuration forbid?** A2 had a live `pre_generation + reprompt` cell that was
  excluded because this line bans unconditional per-turn prose. That is a project constraint, not a
  structural one.

**What you must write:** a feasibility check per locus, and a destination attestation over your own
successful trajectories.

### 3. Executing the action

Suppression must actually cancel a call *and leave telemetry behind*. That last part is the trap: an
intervention can destroy the evidence needed to evaluate it. If a suppressed call simply vanishes from
the trajectory, you cannot tell "the remedy worked" from "the remedy never ran." This exact failure
recurred **four times** here via a field allowlist silently dropping telemetry.

**What you must preserve:** `proposed action → intervention → executed action → outcome`, for every
firing, including the ones that changed nothing.

## What actually generalizes: the decision, not the action

This is the most useful thing the progression taught, and it took six anchors to see.

**The anchors get more specialized over time.** Early ones address system-wide behavioural failures;
later ones read like they exploit one backend's internals. Laid out in order:

| | what it exploits |
|---|---|
| **A1–A4** | broadly applicable failures — capacity handling, unnecessary writes, not consulting memory |
| **A5** | `vector` semantics: entries are **anonymous**, so a duplicate copy is safely removable |
| **A7** | `rec_sum` semantics: **one bounded text blob**, so it can be compressed in place |
| **A8** | `kv` semantics: **addressable slots**, so a duplicate key is evictable instead of clearing |

Read naively that is a slide into per-backend hacks, and it would be a fair criticism of the method if
that were what was happening. It is not.

**The kv row was a prediction when this was first written, and A8 subsequently confirmed it** — the
anchor that landed is dedup-on-addressable-slots, which is what the framing said the substrate would
admit. That is the closest thing here to the framing making a checkable forecast.

### The principle is general; the feasible action is conditional on the substrate

The three anchors above are **one decision instantiated three ways**:

> **When memory is under capacity pressure, preserve information before resorting to destructive recovery.**

| substrate | how it obeys the principle | why that action and not another |
|---|---|---|
| `vector` | remove a redundant copy | entries are anonymous, so an exact duplicate carries no address anyone can depend on |
| `rec_sum` | compress the blob | there is no second container to relocate into, so the only lever is length |
| `kv` | evict a duplicate instead of clearing (**A8**) | entries are **addressed**, and a taken key is *refused* rather than overwritten — so the model manufactures `*_unique` retries, and those suffixed copies are exactly what is safely evictable |
| any | **fall back** | when no information-preserving option exists, that is a real answer |

So the transferable output is the **decision locus plus the principle**. The action is a function of what
the representation admits — which is why porting means writing a `StoreAdapter`
([`../anchoropt/mechanisms/constraint_repair.py`](../anchoropt/mechanisms/constraint_repair.py)) rather
than porting A5 or A7 themselves.

**Do not port the action across substrates.** The single most expensive class of bug on this line came
from exactly that: A5's helpers are archival+vector-shaped, and a retry that synthesized `add(text=…)`
failed on **every** kv firing *after* the eviction had already succeeded — destroying a duplicate and
losing the write, strictly worse than doing nothing. kv and vector look interchangeable and are not.

### A6 is the counterexample that makes the principle falsifiable

The principle earns its keep by **ruling something out**, and A6 is the case it rules out.

A6 faced the same capacity pressure and reached for a relocation — an action that *looks*
information-preserving. On a store already at its slot cap it is not: admitting one entry evicts
another, so the "repair" only reorders which phrasings occupy the fixed slots. Measured: 54 facts
destroyed and 53 added at constant size, and A1's own dispatches collapsing 22 → 6 as A6 pre-empted it.

> **On a saturated substrate there was no information-preserving action available, and A6 supplied a
> selection intervention wearing a capacity costume instead.** It regressed the validation split.

So the principle is not a slogan that every anchor satisfies by construction. It has a discriminating
edge: A5 and A7 clear it because a redundant copy and a compressible blob are genuinely recoverable
slack; A6 does not, because a saturated slot table has none. When no information-preserving option
exists, **fall back** is the correct branch — and taking it is a result, not a failure to find an anchor.
See [`SATURATION_AND_SELECTION.md`](SATURATION_AND_SELECTION.md) and
[`../rounds/A6_deferred/`](../rounds/A6_deferred/).

### Why the specialization is expected rather than worrying

**Once the common errors are removed, the residual exposes substrate-specific bottlenecks.** That is what
iterative re-mining is *for*: it does not work down a fixed list, it recomputes what is left after each
install, and what is left gets progressively closer to the substrate. A progression whose anchors stayed
uniformly general after six rounds would more likely mean the re-mine was not resolving anything.

It also predicts where the method stops being useful without more machinery: when the residual is
entirely substrate-specific, each further anchor buys only the exposure of one backend — which is why
A7's `+7.34 pp` on `rec_sum` is `+2.31 pp` corpus-wide, and why every delta is reported with its
exposure attached.

## Why this repo does not ship an adapter framework

An earlier plan for this repo included a `TrajectoryAdapter`/`ActionAdapter` abstraction and a toy
`minimal_adapter`. **That was dropped deliberately.**

An interface generalized from **one** benchmark is a guess about the second one. The honest state is:
this loop has been run end-to-end on exactly one benchmark (BFCL v4 Agent Memory), so the seams above
are *described* rather than *abstracted*. Writing the abstraction now would encode this benchmark's
accidents into a base class and make the second port harder, not easier — and it would advertise a
portability that has never been tested.

The intended sequence is the reverse: run it on a second benchmark, see which seams actually recur,
*then* abstract. What ships in the meantime is narrower than a framework and more useful than nothing:
[`anchoropt/mechanisms/`](../anchoropt/mechanisms/) holds each anchor mechanism as **pure functions with
no benchmark coupling**, and `constraint_repair.py` factors the shared control flow — detect the
constraint, read live state, propose or decline, verify the invariant, retry verbatim — behind a single
`StoreAdapter` record. Register one adapter and the flow works; there is still no base class to inherit,
for the reason above.

## So what do you get from this repo today

1. **Six accepted anchors plus one deferred**, each with the criteria it was judged against, the mined
   ledgers it was selected from, and — for A1–A4 — per-case results that reproduce every published
   number offline (`python scripts/verify_progression.py`). A5 and A7's numbers are transcribed; A6's
   deferral is recorded in full, because a rejected candidate's evidence is still evidence.
2. **The algorithm, legibly** — five rounds you can read in order, including the two that installed
   nothing, and the reasoning at each decision point.
3. **Two mechanisms as reference implementations** — pure stdlib, no coupling, unit-tested.
4. **The methodology**, which is the part that ports: acceptance ordering, stopping rules with their
   derivations, and the standing rules with the specific failure behind each one.

What you do **not** get is a `pip install` that runs on your benchmark. That work is real, and this
document exists so you can scope it before starting.

## Estimating the port

For a benchmark with tool calls, error strings, and a task-level reward, the load-bearing pieces:

| piece | rough shape |
|---|---|
| trajectory capture (proposed/intervened/executed/outcome per step) | small, but must be **fail-closed** |
| result classifier (errored / vacuous / resolving / harness fault) | the first real design task |
| locus canonicalization for your error vocabulary | small; a cue table goes far before needing an LLM |
| linkage from a locus to downstream task failures | the piece that makes ranking meaningful |
| one action executor per family you intend to use | you likely need only 2–3 |
| your own S1 threshold, re-derived from your corpus power | an afternoon of arithmetic |
| your own variance floor, measured on identical-policy replicates | **do this first** — see below |

**Measure your variance floor before anything else.** Ours is *zero* (0/12 stores, 0/303 calls, 0/303
flips at `temperature=0.001`), which is why a single-case loss is treated as real here. If yours is not
near zero, every harm claim in this repo needs a wider tolerance on your benchmark, and small deltas
will not be interpretable at all.

## Comparison: global prompting and RL

AnchorOpt is complementary to both prompting and model learning, but operates at a different layer.

| | global prompt | RL / fine-tuning | **AnchorOpt** |
|---|---|---|---|
| what changes | prompt/context | model weights or parametric policy | local execution policy |
| where it acts | pre-generation | inside the model | any of three decision points |
| granularity | broad, fires repeatedly | global behavior change | sparse, condition-specific rules |
| can inspect proposed calls/results? | no | indirectly through learned behavior | **yes** |
| credit assignment | bundled across prompt clauses | return / training objective | explicit backward attribution + paired arm |
| inspectability | prompt text is visible, contribution is bundled | limited | **each rule and its evidence are explicit** |

A **global prompt** is one intervention at the pre-generation decision point. It can be effective, but
it cannot inspect a proposed call or tool result, and it pays the cost of firing broadly. In the A4
comparison ([`INCISION_POINTS.md`](INCISION_POINTS.md)), the same text used globally scored **−4.95
pp**, while firing only on the relevant post-generation condition scored **+3.63 pp**.

**RL or fine-tuning** is the better tool when the model itself must learn qualitatively different
behavior — for example, better long-horizon planning or reasoning. AnchorOpt instead leaves model
weights untouched and learns a small set of independently testable rules around the model. This makes
it useful when failures can be attributed to specific decisions in the execution loop.

The structural point behind the table: at pre-generation only `{noop, reprompt}` are feasible, so
**7 of the 9 admissible cells are unreachable by any prompt**, however well written — a prompt
cannot inspect a call that does not exist yet.

**Why this matters under distribution shift.** Because the policy is a set of independently switchable rules over an *unmodified* model, a change in workload is a **re-selection problem, not a retraining problem**: re-run the arms, then turn anchors on or off. Each is one boolean in `policy.json`, and the same paired-comparison machinery that selected an anchor re-validates it. There is no checkpoint to refit, and no risk of disturbing behaviour the anchors never touched. This works precisely because each anchor was accepted against its own incumbent, so each stays separately attributable.

**Where this approach stops.** Anchors act *around* a fixed model and never make it smarter. In this
line, **82% of one measured residual is "retrieved correctly, answered wrong"** — the information
reached the model and the model misused it. No `suppress` or `reroute` reaches that; it is recorded as
an outcome class rather than an attributed mechanism for exactly that reason. Interpretable rules also
do not mean assumption-free inference — the per-anchor power limit above applies to every rule in the
set. What the rules give you is that the assumptions are **inspectable, one rule at a time** — readable
rules, not assumption-free inference.

The approaches are not mutually exclusive: **a prompted or fine-tuned model can itself be the incumbent
that AnchorOpt improves.**

## See also

- [`KEEP_OR_DEFER.md`](KEEP_OR_DEFER.md) §6 — porting the *derivations* rather than the thresholds
- [`INCISION_POINTS.md`](INCISION_POINTS.md) — which pruning is structural and which is yours
- [`../REPRODUCE.md`](../REPRODUCE.md) — what runs offline today versus what needs a GPU
