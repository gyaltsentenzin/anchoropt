# T8 — A8: stop a destructive clear when a redundant copy exists

**Result: train 45.87 % → 47.52 % (+1.65 pp, 9 g / 4 l) · dev 40.48 % → 40.48 % (+0.00 pp, a measured
no-op).** Accepted. Cumulative **+17.49 pp train / +23.81 pp dev** over the measured native baseline.

| | |
|---|---|
| incumbent | N0 + A1–A5 + A7 — 139/303 = 45.87 % train, 34/84 = 40.48 % dev |
| signal | `capacity/container/clear_proposed_at_capacity` |
| decision point | **post-generation, pre-execution** — the only point where the destruction is still preventable |
| action | **suppress** the clear, evict one provable duplicate, retry the blocked write verbatim |
| exposure | **kv only**, and within kv a single scenario cell |
| criteria | [`FROZEN.md`](FROZEN.md), written while the jobs were still running |
| policy | [`policy.json`](policy.json), sha256 `26792094baec5d50…` — one key from the incumbent |
| ledger | [`ledger/n0t10_locus.json`](ledger/n0t10_locus.json) |
| mechanism | [`../../anchoropt/mechanisms/dedup_clear_recovery.py`](../../anchoropt/mechanisms/dedup_clear_recovery.py) — registered as a strategy in `constraint_repair.py` |

## What is new about this anchor

**It is the first anchor that intercepts a destructive action instead of repairing a failed one.**

Every earlier capacity anchor waits for a refusal and then repairs: A1 relocates a blocked write, A5
evicts after an add is rejected, A7 rewrites after an append overflows. A8 fires on a *proposal* — the
model is about to call `clear` on a full container — and the whole point is that after execution there
is nothing left to save. This is the one anchor whose value depends on acting **before** the world
changes.

The predicate:

> the model proposes a clear **and** the container is at capacity **and** a write is known to be
> blocked **and** the live container holds an **exact** duplicate → suppress the clear, evict one
> redundant copy, verify a copy remains, retry the blocked write verbatim.
>
> Otherwise **fall through unchanged.**

That last clause is load-bearing. Where no duplicate exists, letting the destructive call proceed is
the correct branch, and this anchor takes it 5 times out of 5 on the remaining residual (see below).

## Why this is not A5 with a wider trigger

Checked rather than assumed, because "you already have an eviction anchor" is the obvious objection:

| | A5 | A8 |
|---|---|---|
| fires on | a **blocked add**, post-execution | a **proposed clear**, pre-execution |
| trigger string | matches the **vector** error phrasing | every observed event is **kv** |
| what it prevents | nothing — the write already failed | the destruction of 50 entries |

A5's trigger regex cannot reach these states: it matches the vector wording, and these are kv events.
**A8 reuses A5's validated action with a genuinely new trigger** — which is a good outcome, not a
coincidence. The action was already known to be lossless; only the decision point was missing.

## Criterion 3 — the strongest mechanism record in the stack

This is what the second run existed to produce; the first run's telemetry was dropped by a sidecar
whitelist (see below).

| property | required | measured |
|---|---|---|
| firings with per-firing detail | all | **15/15** |
| `copies_before` | ≥ 2 | **2 on every firing** |
| `copies_remaining` | ≥ 1 | **1 on every firing — LOSSLESS** |
| container size | one slot freed | **50 → 49 on every firing** |
| retry landed | true | **15/15** |
| invariant violations / eviction failures | 0 | **0 / 0** |

Each surviving copy was verified against **live** state, not inferred. Exposure is exactly the **3
episodes the diagnosis predicted**, with the 4/5/6 removable copies it measured, and the whole firing
set is **byte-reproducible across two independent runs**.

**The victims are self-documenting.** Every one is a `*_unique`-suffixed key the model invented to work
around `{"error": "Key name must be unique."}`, byte-identical to its unsuffixed partner. These are
true redundant copies, not near-duplicates — the model created the redundancy itself while retrying.

## Criterion 4 — safety

| | control | arm |
|---|---|---|
| archival clears **executed** | 5 | **4** |
| archival clears **suppressed** | 0 | **15** |
| `force_quit`, both splits | 0 | **0** |
| median prereq steps (kv/train) | 19 | **19 — unchanged** |

The median being flat is the check that matters: a guard that suppresses an action the model then
re-proposes indefinitely shows up as a step explosion, and that pattern is absent.

**The eviction cap never engaged** — max 6 per episode against a budget of 8. The duplicate *supply*
binds first, which is the intended ordering: the cap is a backstop, not a throttle. Worth noting that
A5's cap of 3 would have refused firings 4 and 5, declining capacity this arm can supply losslessly —
so a bound copied from a sibling anchor would have silently cost coverage.

## What makes the delta attributable at all

| check | result |
|---|---|
| control reproduces the frozen incumbent | **139/303, 0 disagreements across all 303 cases** |
| run 2's control reproduces run 1's | kv/train **30/105 identical**, 0 case-level disagreements |
| control purity | **0** gate dispatches in all six control shards |
| backend scope | **0** dispatches on vector and rec_sum arm shards |
| double-dispatch | 15 evictions, 15 distinct (case, victim) pairs, **0 repeats** |

The control identity is the strongest of these: an independently rebuilt store, on a different day,
reproduces the incumbent case for case. Any arm difference is therefore the arm's.

## Limitations — part of the acceptance, not footnotes

1. **The entire effect is one cell.** All 3 exposed episodes and all 13 flips are in **kv/student**.
   The (backend, scenario) cell is the indivisible unit here, so this is **n = 1 cell of support** —
   narrower than A5's single-backend exposure, which was already narrow.
2. **Dev cannot confirm it.** There is no kv/student cell in dev, and the arm is a measured no-op
   there with byte-identical exec logs. Criterion 2 is satisfied **vacuously**: it establishes *no
   harm* and supplies **zero** independent evidence for the gain. Pre-registered as a limitation
   before any number existed.
3. **9 gains / 4 losses, not 9/0.** The losses are real and in the same cell, so the arm **reorders**
   outcomes within kv/student rather than only adding to them. The rule permits this and requires the
   counts be reported beside the net.
4. **Dedup is exhaustible.** 4 archival clears still executed after the duplicates ran out, and clear
   *proposals* went **5 → 19**: suppressing a clear does not end the pressure — the model re-proposes,
   and each cycle spends one duplicate. This bounds what this layer can ever achieve.
5. **One observability defect is open.** The gate's decline reason is never recorded, because an early
   `continue` precedes the recording. No effect on any criterion, but the no-duplicate share has to be
   read from the destructive-call audit instead of from the gate.

## The ladder question, answered from live state

Asked **regardless of the verdict**, per the frozen criteria — and measured from the destructive-call
audit, which snapshots the live container. Not from the gate's decline reasons (stranded by the defect
above) and not by replay: an exec-log replay misses model-written content and produced impossible
`size = 0` readings at a clear.

**Of the archival clears that still execute on the accepted incumbent: 5 of 5 are at 50/50 capacity
with no exact duplicate.**

| case | size / cap | could dedup rescue it? |
|---|---|---|
| `prereq_12-healthcare-2` | 50 / 50 | no — no duplicate |
| `prereq_35-notetaker-3` | 50 / 50 | no — no duplicate |
| `prereq_24-student-2` | 50 / 50 | no — **duplicates spent** |
| `prereq_26-student-4` | 50 / 50 | no — no duplicate |
| `prereq_4-customer-4` (dev) | 50 / 50 | no — no duplicate |

**100 % of the remaining clear-time residual is out of this layer's reach by construction.** Two of the
five are the class-3 events the original diagnosis predicted would fall through. One is new:
`24-student-2` *had* duplicates, this layer spent all four, and the pressure returned — which is
limitation 4 observed directly.

> **So the ladder is confirmed by measurement: dedup → safe concat / consolidation → destructive
> fallback.** Layer 2 should be tested at exactly these 5 states, where the container is full and holds
> nothing redundant, so the only lossless route to capacity is *combining* entries rather than removing
> one.

This layer is **dedup-only by design**, so its effect is attributable in isolation. Bundling
consolidation into the same arm would have made the two indistinguishable.

## Process failures in this line, recorded

Both were caught before they corrupted a verdict, and both are now guarded.

1. **Double-dispatch.** The suppress hook evaluates its predicate twice per call, and the first
   implementation accumulated a plan — so every eviction and retry ran twice. State was unharmed (the
   repeat is a no-op) but a call was wasted and telemetry double-counted. Fixed by keying on the call
   string; a regression test now reproduces the double-evaluation.
2. **Dropped telemetry — the eighth instance of one defect class.** The new gate's key was missing
   from the sidecar's family whitelist, whose own comments document **seven prior instances** and
   state the rule: *any new step_record key needs a prefix here on the same commit that adds it.* That
   rule was not followed. Criterion 3 was unprovable from run 1 as a direct result, and the re-run cost
   12 GPU-shards.

A third, smaller: a cap test inserted identical content for every simulated retry, so the retries
became duplicates of each other and read as a cap leak. A test artifact — but it surfaced a real
property now asserted directly: the duplicate supply is whatever the **live** store holds at each
decision point, so repeated identical model writes are legitimately evictable.

---

See [`../../docs/ANCHORS.md`](../../docs/ANCHORS.md) for how A8 fits the capacity principle, and
[`../../docs/ACCEPTANCE_RULE.md`](../../docs/ACCEPTANCE_RULE.md) for the rule it was judged against.
