# T9 — A9: the model searched one container, the answer was in the other

**Vector shard: train 43.8% → 60.7% (+16.9 pp, 15 g / 0 l) · dev 35.0% → 52.5% (+17.5 pp, 9 g / 2 l).**
Accepted. Mechanism validated **24/24** at case level.

> **These are VECTOR-SHARD figures.** The harness prints `Overall accuracy` per shard and this shard is
> vector-only, so read as a corpus number 60.7% overstates the anchor by about **3×** — the vector shard
> is 29.4% of train and 47.6% of dev.
>
> **Whole-corpus, now MEASURED on three shards:** train 47.52% → **52.48%** (+4.95 pp, +15/−0), dev
> 40.48% → **48.81%** (+8.33 pp, +9/−2). These cells were a projection from the vector shard for one
> commit; the confirmation run reproduced them **exactly**.

| | |
|---|---|
| incumbent | N0 + A1–A5 + A7 + A8 — 144/303 = 47.52% train, 34/84 = 40.48% dev |
| signal | `retrieval/similarity/core_max_below_threshold` |
| decision point | **post-execution** — the retrieve has returned and its scores are readable |
| action | **reroute** the read: dispatch the same query to archival, merge-rank both containers |
| exposure | **vector only** — enforced in code, not by policy |
| policy | [`policy.json`](policy.json), sha256 `c610e74fedf3…` — one key from T8 |
| mechanism | [`../../anchoropt/mechanisms/cross_container_merge.py`](../../anchoropt/mechanisms/cross_container_merge.py) |
| jobs | 1146571 (arm/train), 1146572 (arm/dev), 1146573 (ctl/train), 1146574 (ctl/dev) |

## What is new about this anchor

**It is not the first read-side anchor — A2 is — and the comparison is the point.** Both come from the
same diagnosis: the value is in archival and the read did not consult archival. They differ on the
trigger and on who acts.

| | A2 | A9 |
|---|---|---|
| trigger | the read **errored** (`not_found`) | the read **succeeded**, weakly matched |
| signal | an error **string** | a similarity **score** |
| action | **reprompt** the model to search archival | **search archival itself**, merge, return |
| who acts next | the model | the system; the model is told nothing |

A2 needs something to have gone visibly wrong. **A9 fires where nothing raises at all** — a retrieve
returning five weak matches is a fully successful call as far as the harness is concerned. That is the
residual class an error-keyed vocabulary cannot see.

What is new to A9: it leaves the store byte-untouched and changes what the model is **shown**. Every
other anchor either acts on a write or a destructive action, or (A2, A4) acts by instructing.

The diagnosis: a core retrieve comes back with every entry weakly matched, and the model concludes the
store does not hold the answer. In **13 of 13** verified cases the information *was* present — in
archival — and the model **never issued an archival read at all**.

The containers are lopsided, which is what makes merging worth doing rather than just tidy:
`vector-finance` holds **5 core entries against 30 archival**, so a merged top-5 ranks over 35 candidates
instead of 5.

```
core retrieve, best similarity < 0.30
  → dispatch the SAME query to archival
  → rank the UNION globally by the harness's own scores
  → replace the weak core result AT ITS OWN INDEX with the merged top-k
otherwise → fall through unchanged
```

## The safety argument this anchor was accepted on is RETRACTED

**A9 was argued to be "additive by construction": the union is ranked, so a core entry that still ranks
in the top k is retained, therefore a false firing "costs one extra read, never a displaced correct
answer."**

The last clause is **false**, and it was measured false on this anchor's own held-out losses.

| case | `core_topk_before` | `core_entries_retained` | core sim | archival sim |
|---|---:|---:|---|---|
| `123-student-43` | 5 | **0** | 0.197–0.256 | 0.405–0.516 |
| `89-student-9` | 5 | **0** | 0.217–0.287 | 0.400–0.515 |

Archival out-scored core by roughly **2×** on both, so global ranking evicted **every** core entry from
the returned top-5. On `89-student-9` the control answered *"sailing"* correctly from core id=4; the arm
answered *"gaming"* from archival. **The evidence was absent, not overlooked** — which is displacement,
not distraction.

**The reasoning error matters more than the correction.** It argued from a property of the *mechanism*
(union ranking preserves relative order) to an *outcome* (a used entry survives). Order preservation says
nothing about which entries fall below the cut. That is the same shape as the delivery defect below:
reasoning about the gate instead of measuring the thing downstream.

And displacement is not an edge case — it is the **expected** one, because this anchor only fires when
core scored *weakly*, which is exactly when archival is likely to dominate it.

> **Corrected:** the merge returns the globally best top-k of the union; when archival out-ranks core,
> core entries are dropped. Measured, that trades **15 gains / 0 losses on train and 9 / 2 on dev**. It
> is **net-positive, not non-destructive.**

**Why the acceptance stands.** The rule requires **net** dev ≥ 0, not zero losses, and +7 clears it — the
same rule that deferred A6 for a *negative* net. What changed is the rationale, not the verdict.

**What survives as a guarantee:** ties break toward core, so an *equal*-scoring archival entry cannot
evict one the model already had. That is narrow, and it is all it says. Displacement is now reported per
firing (`core_entries_retained` against `core_topk_before`, with dropped ids listed) rather than labelled
an invariant.

**Do not reuse "additive by construction"** to justify a future merge-style anchor. The obvious repair —
reserve k slots for core, or require archival to beat it by a margin — is a **new mechanism** needing its
own frozen criteria: a "retain one core entry" guard would have fixed both losses **and** changed the 24
wins, so its net effect is unknown, not positive.

**The rejected alternative was winner-takes-container** — use archival's list when it scores higher. It
was dropped on measurement, not taste: firing cases that already **pass** have `core_max` median 0.159
against 0.172 for the ones that **fail**. The same band. A container-level switch cannot discriminate, so
it would swap out correct results about as often as wrong ones.

## Criterion 3 — mechanism, 24/24 at case level

Each gain required **all three**, verified per case rather than inferred from the aggregate:

1. A9 fired **and the merged result was attested delivered** into the real message list
2. archival entries were **promoted** into the returned top-k
3. the case **fails in the paired control**

Representative jumps in best similarity:

| case | before | after |
|---|---:|---:|
| `43-healthcare-13` | 0.034 | **0.691** |
| `95-student-15` | 0.211 | **0.713** |
| `8-customer-8` | 0.278 | **0.709** |
| `108-student-28` | 0.140 | **0.617** |

**Train had zero losses.** Dev lost 2 (`123-student-43`, `89-student-9`); both fired with promoted
entries, so they are genuine **displacement by the merge**, not gate failures. Recorded rather than
absorbed into the net.

## The delivery defect — read this before trusting any anchor's telemetry

**A9's first two versions computed the correct merged payload on all 78 firings and delivered it on
none.**

The gate ran ~190 lines *below* the point where the model's prompt is assembled, and wrote to
`step_record["tool_results"]` — the telemetry sidecar. Its write to the real `execution_results` was
**dead code**: immediately followed by `step_count += 1; continue`, so the loop restarted and rebuilt the
list before anything consumed it.

Measured proof on `108-student-28`, from the byte-exact captured prompt:

| | |
|---|---|
| **computed** | id 59, similarity **0.617**, `"Family Europe trip: London(Tower, British Museum…)"` |
| **delivered** | friendship entries at 0.140 / 0.136 / 0.133; the token `London` appeared **zero times** in the 12,646-char prompt |

Granite's *"the core memory does not contain any information"* was **correct behaviour on the input it
actually received.**

`xcm_replaced=True` was *true* — of the sidecar. So the existing arm-identity rule needs a second half:
**firing telemetry can be right about itself and wrong about the world.**

### What it cost

| claim | status |
|---|---|
| A9-R +0.00 pp → "the model won't use the evidence" | **VOID** — evidence never delivered |
| A9-R retrieval positive, gold-in-context 31/78 → 50/78 | **superseded** — measured on `tool_results` |
| delivery audit: "verbatim in the actual model input" | **RETRACTED** — audited the wrong object |
| A9-U comprehension limit (10/11 re-denied) | **VOID** — a re-read of absent evidence |
| A9-R-top1: dilution ruled out | **VOID** — top-1 of a discarded payload |
| two-model probe: framing, not capability | **VOID PREMISE** — fed a passage the run never sent |
| sysprompt localisation: system prompt innocent | **VOID PREMISE** — same |

Four experiments, all downstream of one unverified substitution.

**The tell was visible.** The pre-fix arm scored **exactly the control's 35.0%**, to two decimals, across
two independent control runs — the signature of a computationally inert anchor. And the probe found the
model answered correctly on **all ten** reconstruction cells while the real system always failed. *"My
reconstruction always succeeds and the real system always fails"* is a **delivery** finding, not a framing
finding.

### The fix, and the general rule

v3 changes **where** the result is applied and nothing else. `_try_xcm_merge` is byte-identical — trigger,
threshold, dispatch, global ranking, tie-break, top-k policy all unchanged, pinned by a 22-check test that
**fails 11 on the pre-fix file**.

`execution_results` is now the single authoritative object and the telemetry list is **derived** from it.
The prior form ran two independent index-range checks against two lists that need not be the same length,
so `xcm_replaced=True` could be recorded from telemetry while the model input took an append fallback.

> **Validate an anchor at the boundary its CONSUMER reads, not at the gate boundary.** Gate-side
> telemetry establishes *intent*; it never establishes *effect*.

Full rule: [`../../docs/CONSUMER_BOUNDARY_RULE.md`](../../docs/CONSUMER_BOUNDARY_RULE.md).

## Gate 0 — evidence validity, checked before mechanism and accuracy

| condition | train | dev |
|---|---|---|
| every **attested** firing delivered | **55/55** | **23/23** |
| delivery violations | 0 | 0 |
| positional replacement, no append fallback | 0 appended | 0 appended |
| trigger unchanged (all `core_max` < 0.30) | ✓ | ✓ |
| arm/control distinct `base_key` | ✓ `9870c69d…` vs `526c02d2…` | ✓ |

**Now 0 unattested.** The earlier count of 11 train / 3 dev was **stale artifacts** from a previous
run, not a delivery gap — episodes carry a `run_id` and Gate 0 counts only the current run.

A first Gate 0 pass scored these as failures — "55/66 delivered, 0 violations", which is
self-inconsistent. Conflating *not measured* with *not delivered* is the original A9 error in mirror
image. They are now counted separately, and unattested is constrained to be prereq-only so a query-phase
gap cannot hide there.

## Limitations — part of the acceptance, not footnotes

1. **It is net-positive, not non-destructive** — the retracted claim above, and the most important entry
   here. Displacement is the expected mode, and 2 of 37 dev firings displaced a correct answer.
2. **Vector only.** The trigger needs per-entry similarity scores, which kv and rec_sum do not expose. So
   this is one backend's instantiation of a read-side principle, and the principle's reach on the other
   two is untested rather than established. Note this is a *scope* limit, not a *harm* limit: the two
   other shards are now measured byte-identical rather than assumed unaffected.
3. **n = 2 losses is enough to falsify, not to estimate.** One counterexample kills a universal claim,
   and both losses show the identical signature, so the failure *mode* is characterised. A displacement
   *rate* across domains is not.
4. **The two losses point at a real retrieval-quality signal, unaddressed.** Both are `student` cases
   whose core entries are thematically right but lexically weak (sailing-as-problem-solving) while the
   archival entries are lexically strong and thematically wrong. That is a mining target in its own
   right, recorded rather than acted on.
5. **A ~1 pp reproducibility floor.** A forced-rebuild run scored 61.8% on the same arm where the
   cache-HIT run scored 60.7% — store-build nondeterminism, not A9. Single-shard differences under ~1 pp
   are within noise, here and elsewhere in this repo.

**Two limitations from the first version of this document are now CLOSED**, and one closed by finding its
own premise was wrong:

- **Three-shard confirmation: done.** It reproduced the projected cells exactly, and measured
  non-interference (kv 35/35 and rec_sum 70/70 train, byte-identical, zero off-target firings) rather
  than inferring it from the backend guard in code.
- **Prereq-phase delivery: verified — and it was never a code gap.** The 11 "unattested" firings were
  **stale artifacts** from the previous day's pre-capture run, left in the sidecar directory by a
  snapshot-cache hit; the v3 run skipped all 27 prereqs, so no prereq episode had executed at all.
  Episodes now carry a `run_id` and Gate 0 counts only the current run: **55/55 and 23/23 attested, zero
  unattested.** A forced rebuild confirms the prereq path does capture (1 firing, 1 attested). The fix
  was to the *measurement*, not the code — and "unverified" was the right label for it either way.

## Reproducibility

Both splits ran under the sharding protocol — `--store-workers 1`, distinct ports, arm and control on
separate `base_key`s — so the figures are byte-reproducible. **Dev replicated 52.5% exactly across two
independent runs.**

---

See [`../../docs/ANCHORS.md`](../../docs/ANCHORS.md) for how A9 sits beside the capacity anchors, and
[`../../docs/ACCEPTANCE_RULE.md`](../../docs/ACCEPTANCE_RULE.md) for the rule it was judged against.
