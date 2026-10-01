# T2 — A2: reprompt to search archival after a failed read

**Result: 33.66 % → 35.97 %, +2.31 pp.** Installed.

| | |
|---|---|
| incumbent | N0 + A1 — 102/303 = 33.66 % |
| signal | `existence/identifier/not_found` |
| incision point | **post-execution** |
| action | **reprompt** — advise searching archival; do not synthesise the call |
| criteria | [`FROZEN.md`](FROZEN.md), written before launch |

## Why this signal

Rank 1 on the **post-A1** residual: 51 events, **47 of 201 failing queries linked** — nearly 1:1 loss
per occurrence, the tightest coupling of any locus. It was rank 3 at T1 and rose without changing,
because A1 consumed the mass above it.

## The lesson of this round: one error string, two causes

The semantic miner names a *symptom*. A second **attribution** stage then splits the locus by *first
consequential decision*:

| first consequential decision | occurrences | anchor |
|---|---|---|
| the value **was** archived; the read never consulted archive | **39** | **A2.R** — installed here |
| core filled and the value was **not** archived | 5 | A2.W — **deferred** |
| no write-side explanation | 8 | `unattributed` |

Same error text, two different causes, two different anchors. Without this split the arm would have
targeted a mixture and been unattributable either way.

**All 39 are self-inflicted:** the value sits exactly where A1 put it. A1 converted a fatal capacity
error into a recoverable retrieval problem — real progress — but the recovery has to be learned too.

### Why A2.W was deferred rather than pursued

5 occurrences is below any support floor (**S1**), and its mechanism is A1's own: A1 already rescues
~97 % of core-full events (finance 33/33, notetaker 28/28, student 145/146). The remaining leak
concentrates in `vector-customer` (125/133) and `vector-healthcare` (37/42) — that is a **coverage
question for A1**, which is a parameter change, not a new anchor with its own maintenance surface.

It stays in the ledger. Deferred is recorded, not discarded.

## Why post-execution + reprompt

The grid: 2 admissible incision points × 4 actions = 8 nominal cells, **3 live**. The exclusions are
structural and stated:

- `suppress` — has nothing to cancel pre-generation, and cannot undo a completed read.
- `reroute` — needs a call to substitute for, so unavailable pre-generation.
- `noop` — point-independent (it is the control).
- `pre_generation + reprompt` — live, but **not run**: it is a prose nudge on *every* turn, which this
  prompt-free line excludes by construction.

Two arms were run and **reprompt won**, against the standing 2/3 prior that substitution beats advice.
Recorded as a case where a mechanism-plausible expectation lost to measurement.

| arm | train | Δ vs the A1 incumbent (33.66 %) | note |
|---|---|---|---|
| **reprompt** | 105/303 = 34.65 % | **+0.99 pp** | selected |
| reroute | 102/303 = 33.66 % | +0.00 pp, **0 gains / 0 losses** | byte-identical to control — it never fired |

The reroute arm producing *zero* flips is the informative part: the substitution could not be
constructed on the cases that mattered (see the destination correction below), so the arm tested
nothing. Under the frozen rule that is **inconclusive-by-liveness**, not a null result — and it is why
`destination_attest.py` now exists.

### Two numbers for this round, and why they differ

| number | source | what it is |
|---|---|---|
| 34.65 % | `n0a2r_train/reprompt` | the **selection arm** — A2's two candidates measured against A1 |
| **35.97 %** | `n0clean_train/a1a2` | the **frozen incumbent** — a clean re-run of A1+A2 together |

The progression table uses **35.97 %**, because the frozen sequence requires a clean incumbent
re-measurement before the next round mines it (A3's spec names `n0clean_train/a1a2` explicitly as its
control). The selection arm chooses *which* candidate; the clean re-run establishes the world the next
round is mined from. Both are shipped in [`result/`](result/).

## The correction that saved this arm: attest destinations on *resolving*

The registry declared `archival_memory_retrieve` as the reroute destination. That demands an **exact**
key match — but A1 sanitises and dedupes keys before archiving, so an exact retrieve misses on precisely
the cases A1 created.

| destination | tried | resolved |
|---|---|---|
| `archival_memory_retrieve` | 1 | — |
| `archival_memory_key_search` (fuzzy BM25) | 11 | **11/11** |

A related trap in the same area: an earlier version admitted a destination at "11/11 clean", where
*clean* meant **did not error** — and 9 of those 11 were vacuous, returning nothing. Hence the standing
rule: **attest a destination on resolving, never on not-erroring.** `destination_attest.py` exists for
exactly this.

`synthesize_read_reroute_call` also had to learn per-destination argument names — it emitted `key=`
universally, which would have built an invalid call for `key_search` and been misscored as "the
substitute did not help."

## What was predicted

1. Fires on the kv chains carrying the locus. **Zero firings would be a wiring fault, not a null result.**
2. Direction positive but the effect **small**: 39 occurrences against a 201-case residual, and
   `key_search` returns *ranked keys*, not a value — so the model must still issue a follow-up retrieve.
   **This arm removes a dead end; it does not hand over the answer.**
3. reroute ≥ reprompt. **This one was wrong**, and the run decided it.

## What it changed for the next round

The `not_found` mass is consumed, and the next-ranked locus is `permission/identifier/duplicate` —
which, like this one, turns out to be **entirely A1-induced**. See
**[T3 / A3](../T3_A3_duplicate/)**.
