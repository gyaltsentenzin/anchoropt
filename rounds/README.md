# The rounds — read them in order

Each directory is one iteration of the loop. They are numbered by round (`T1`…`T8`), not by anchor,
because **two rounds installed nothing** and dropping them would hide the most interesting part.

| round | what happened | train | Δ |
|---|---|---|---|
| [T1](T1_A1_capacity/) | **A1 installed** — reroute a capacity-blocked write to archival | 33.66 % | +4.62 |
| [T2](T2_A2_not_found/) | **A2 installed** — reprompt to search archival after a failed read | 35.97 % | +2.31 |
| [T3](T3_A3_duplicate/) | **A3 installed** — suppress a redundant re-write before it executes | 38.61 % | +2.64 |
| [T4](T4_exhaustion/) | **nothing installed** — every locus exhausted; expand the signal vocabulary | 38.61 % | +0.00 |
| [T5](T5_A4_no_tool_call/) | **A4 installed** — from the expanded vocabulary | 42.24 % | +3.63 |
| [T6](T6_A5_archival_full/) | **A5 installed** — evict a provable duplicate to free a slot | 43.56 % | +1.32 |
| [T7](T7_A7_blob_overflow/) | **A7 installed** — rewrite a single-string store to admit a refused append | 45.87 % | +2.31 |
| [T8](T8_A8_dedup_clear/) | **A8 installed** — cancel a destructive clear when a redundant copy can be evicted instead | 47.52 % | +1.65 |
| [T9](T9_A9_xcontainer_merge/) | **A9 installed** — search the other container when a retrieve comes back weakly matched. Vector shard **+16.9 / +17.5 pp** | 52.48 %* | +4.95* |
| [A6](A6_deferred/) | **DEFERRED** — dev −1.19 pp and on-target −2.50 pp, so the remedy is not ready. The signal is still open. | — | — |

\* **T9's cells are measured** — three-shard paired run, with kv and rec_sum byte-identical between
arms and zero off-target firings. They were a projection from the vector shard for one commit, and the
run reproduced them exactly. **T8 — 47.52 % train / 40.48 % dev — remains the last endpoint this repo
can recompute OFFLINE**, because A9's shard artifacts are still on the cluster
([`T9…/result/NOT_SHIPPED.md`](T9_A9_xcontainer_merge/result/NOT_SHIPPED.md)), so start a reproduction
there. Measured and recomputable-here are different properties, and only the second one is about T8.

Each round contains:

```
FROZEN.md    the acceptance criteria, written BEFORE launch and never edited afterward
README.md    why this signal, why this incision point, what it changed for the next round
ledger/      the mined loci, rankings and attribution artifacts the decision was made from
result/      the arm's outcome
```

---

## The chain: three of four anchors were caused by their predecessor

This is the reason the loop re-mines after every install instead of working down a ranked list
computed once.

```
        T1  A1  reroute blocked writes to archival
             │
             ├──── moves data to a tier reads don't consult
             │         └──> T2  A2   `not_found`     39/39 SELF-INFLICTED
             │
             └──── injects keys the model doesn't know exist
                       └──> T3  A3   `duplicate`     59/59 on A1's OWN keys

        T4  error-keyed vocabulary EXHAUSTED  (S1-STOP and S2-STOP on every locus)
             │
             └──── phase_switch -> EXPAND_ATTRIBUTION
                       └──> T5  A4   `no_tool_call_at_all`  (a detector, not an error)
```

A1 did not "cause failures" — it converted a fatal capacity error into a *recoverable* retrieval
problem, which is progress. But the residual it left was **not the residual it inherited**, and that
is the whole argument for re-mining:

- A ranking computed at T1 would still have listed `capacity/…` at the top for T2 and T3, because
  those events keep appearing in failing traces even after A1 repairs them.
- **Occurrence-level re-labeling** marks a repaired occurrence *corrected* and removes it from the
  support used to pick the next anchor. Applying it materially reordered the queue — a `size/item`
  candidate fell from 30 linked failures to 9.
- Without that step the loop optimizes an error taxonomy from a world that no longer exists.

## Telemetry flag names — old and new

Trajectory records carry one boolean per anchor. Two flags were renamed because the old names came from
an earlier gate numbering and read as *retired* gates rather than as A1 and A2 — I misreported them as
stray firings once, which is exactly the confusion worth removing.

| anchor | registry key | flag now | flag in the frozen `result/*.json` |
|---|---|---|---|
| **A1** | `on_domain_error_core_full` | `a1_core_full_gate` | `g3_gate` |
| **A1** | `on_core_full_rerouted` | `reroute_gate` | `reroute_gate` |
| **A1** | *(the computed repair)* | `capacity_repair_gate` | `capacity_repair_gate` |
| **A2** | `on_domain_error_key_not_found` | `a2_key_not_found_gate` | `g4_gate` |
| **A3** | `on_redundant_write_suppressed` | `redundant_write_gate` | `redundant_write_gate` |
| **A4** | *(code path)* | `zero_call_reprompt_gate` | `zero_call_reprompt_gate` |

**The frozen artifacts were deliberately NOT rewritten.** They are the published evidence; editing them
to match a later naming choice would break the diff against what was actually measured. So a record
under `rounds/*/result/` carries `g3_gate`, and a record from a run made today carries
`a1_core_full_gate`. Same event.

Renaming was safe because the flags are **write-only** — nothing reads them by literal name, and every
generic consumer resolves them from the registry (`telemetry_flags()`, `flag_to_key`). A test asserts
the legacy names cannot return and that each flag survives the sidecar filter, since a step_record key
matching no rule is dropped silently — a defect that has recurred seven times in this project.

## What each round teaches, in one line

| round | the transferable lesson |
|---|---|
| **T1** | Rank by **linked downstream loss**, not error frequency. Rank 3 outranked ranks 4–5 on *fewer* events, because its events were each ~9× more damaging. |
| **T2** | One error string can hold **two causes**. Attribution split `not_found` into a read-side anchor (39 cases, installed) and a write-side one (5 cases, deferred). Also: attest a reroute destination on **resolving**, not on not-erroring. |
| **T3** | Look for **self-inflicted** residuals. All 59 collisions were against keys A1 itself injected — an anchor auditing its predecessor. And the predicate stays **exact, never semantic**: it declines 4 of 59 rather than risk suppressing real information. |
| **T4** | **Installing nothing is a result.** Distinguish *policy exhausted* from *signal representation exhausted* — they look identical from the ranking and demand opposite responses. |
| **T5** | A detector needs **precision against successes**, not coverage. `no_tool_call_at_all` (0.80) beat three more intuitive candidates that fired on passes as often as failures (0.57 / 0.50 / 0.47). |
| **T6** | A repair can be **provably lossless at its own decision point and still lose cases**, because the agent reacts to the changed store (`core_memory_remove` 42 → 96). Intervening has a cost that is not the intervention. |
| **A6** | **Check for displacement before believing a gain.** A6 helps on train and REGRESSES on dev because it pre-empts A1 at constant capacity — a *selection* intervention in a capacity costume, and its own target backend goes negative. Deferred. |
| **T8** | **Prevention needs a pre-execution point.** A8 is the first anchor to stop a destructive action rather than repair a failed one, and it reuses A5's validated *action* with an entirely new *trigger* — the action was already known lossless; only the decision point was missing. It also has the stack's best mechanism record (15/15, lossless every time) and its narrowest support (one cell). |
| **T7** | Constrain an LLM repair **relative to its input**, not to an absolute target: told to be shorter, the model rephrases; told only to fit, it restructures and loses content at any length. Every added guard scored below the simplest rule. |

## Reading the numbers honestly

- **All train accuracies share one denominator** (n = 303) and come from within-job paired
  comparisons against that round's own incumbent. Never difference accuracies across jobs or corpora.
- **The replay variance floor is zero** (0/12 stores, 0/303 calls, 0/303 flips at
  `temperature=0.001`), so every flip is attributable — and a single loss is a real loss.
- **Dev is a separate, smaller corpus** (n = 84, disjoint domains): 16.67 % → 48.81 %. A8 is a measured **no-op** there, so its dev step is +0.00 and criterion 2 is satisfied vacuously. It is used to accept and reject anchors, which makes it a **validation** set — the figure is partly selected-on. A4's own
  step is +3.57 pp — the point estimate transfers almost exactly (+3.63 → +3.57), and the pre-stated
  bar was directional consistency. **A per-anchor p-value is deliberately not quoted here**: at this n,
  and with 15 indivisible cells, one anchor's arm cannot reach significance whatever its true effect,
  so a threshold read would be uninformative in either direction. Each round's frozen spec carries its
  own statistics for anyone who wants them.
- **Per-backend splits are descriptive only.** A 24-case cell cannot reach significance at any effect
  size, and across runs the backend decomposition has taken four different shapes. Report the
  aggregate.

## Rejected candidates, recorded because the rejections are part of the method

These were evaluated on this line and **not** accepted:

| candidate | verdict | why |
|---|---|---|
| low-similarity reprompt | **rejected** | mechanism fired 22/22 and reformulated every time, but only 2/22 converted — a good detector is not a good anchor target |
| W1 entry-length **condense** | **rejected** | repair worked and preserved every fact, yet still lost 3 cases on one cell against a zero variance floor. A6 later succeeded at the same locus by **relocating** instead of rewriting. |
| B1 futility escalation | **rejected** | −6.60 pp, p = 0.0352 — a generic "reconsider and try again" delegates the remedy to the model, which chose wholesale clearing |
| A6 normalized idempotence | **rejected** | fired **0×** on train, so its train figure was bare A6's with a guard that never engaged; over-suppressed where it did fire (17 dev facts retained vs raw-exact's 56) |
| A7 ratio floor (v4) | **rejected** | −7.34 pp. Rejecting a collapsed rewrite removes the compaction without improving it; 5 of 8 cases could only emit sub-floor output and stayed permanently blocked |

B1 is the sharpest of these: it is the experiment that tried the *opposite* of the thesis — asking the
model to invent its own recovery — and failed significantly. The accepted anchors execute a constrained
action instead.

## Two rounds whose acceptance is qualified, and one that failed a frozen clause

The filename in each directory states how that round was accepted, because the filename carries a claim:

| file | what it asserts |
|---|---|
| `FROZEN.md` | criteria written **before** the arm ran, never edited (T1–T3, T5) |
| `CRITERIA.md` | conditions **transcribed**, or variants swept with results visible (T7, E1) |
| `OVERRIDE.md` | a frozen clause was **failed** and overridden, with the argument recorded (T6) |
| `DEFERRED.md` | the remedy did not clear the rule and the **signal is left open** (A6) |

A directory of `FROZEN.md` files would imply every anchor cleared its own bar. A5 did not, so its file
is named for what actually happened. See [`../docs/SATURATION_AND_SELECTION.md`](../docs/SATURATION_AND_SELECTION.md)
for the diagnostics the T6–T8 line produced.
