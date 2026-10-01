# T8 — A7: when the store is one string and there is nowhere to relocate to

**Result: 44.88 % → 47.52 %, +2.64 pp corpus-wide (spliced), from +7.34 pp on the one backend it
reaches.** Held out **+4.76 pp** corpus-wide, **+20.00 pp** on the backend. Accepted; tuning
**closed**. Cumulative **+18.48 pp** over the native baseline.

| | |
|---|---|
| incumbent | N0 + A1–A5 — 132/303 = 43.56 % train, 29/84 = 34.52 % dev |
| signal | `size/blob/append_would_exceed_cap` |
| incision point | **post-execution** |
| action | **reroute** — rewrite the blob to free room, then retry the append verbatim |
| exposure | **rec_sum only** — 109/303 train, 20/84 dev |
| criteria | [`CRITERIA.md`](CRITERIA.md) — variants swept with results visible |
| ledger | [`ledger/n0t7_locus.json`](ledger/n0t7_locus.json), frozen, fp `edea7bb67e43e48b` |
| mechanism | [`../../anchoropt/mechanisms/blob_compaction.py`](../../anchoropt/mechanisms/blob_compaction.py) |

## Why neither earlier capacity anchor applies

A1 and A6 both answer "this container cannot hold the write" by **relocating** the payload to a second
container. This backend stores memory as a **single string** at a 10,000-character cap and **has no
second container**. There is nowhere to put anything: when an append would overflow, the tool rejects it
and the content is simply lost.

So the only way to admit the pending write is to make the blob itself shorter — which makes A7 **the one
accepted anchor that calls a model at inference time.** Every other one computes its repair. That cost
is real and stated rather than buried: an added LLM call reorders the trajectory, so coverage, state
keys and downstream questions all move, and per-decision reasoning cannot bound run-level behaviour.

## The simplest acceptance rule won, and every added guard scored below it

| variant | acceptance rule | train (n=109) | vs incumbent |
|---|---|---|---|
| **v3** | **fits AND shorter than the original** | **64.22 %** | **+7.34 pp** (21 g / 13 l) |
| v5 | + rescue retry at a numeric target | 61.47 % | +4.59 pp |
| v6 | fit only, no length rule | 60.55 % | +3.67 pp |
| v4 | + reject rewrites under 60 % of input | 49.54 % | **−7.34 pp** |

v3 is also the only variant with **no train/dev regression** (+20.00 pp dev, 4 g / 0 l).

## The transferable finding: rephrasing vs restructuring

The obvious reading of *v3 beats v6* is "shorter output loses more information." **That is wrong**, and
one case pins it. On `35-healthcare-5`:

| | ratio | final blob | needed fact (`Notebook`) |
|---|---|---|---|
| v3 | 0.132 | **1,298** chars | **present** |
| v6 | 0.314 | **3,094** chars | **absent** |

v3's rewrite is **2.4× smaller** and it is the one that keeps the fact. v6 spent its extra length on a
nested markdown outline; v3 wrote flat prose.

> Constrained to be **shorter than the original**, the model **rephrases** and keeps content.
> Constrained only to **fit**, it **restructures** — and restructuring costs content at any length.

The "must be shorter" clause is therefore not a length rule. It keeps the model in rephrasing mode
instead of reformatting mode — work nobody designed it to do. **Generalisation: when an LLM repairs a
payload under a size constraint, constrain it relative to its input, not to an absolute target.**

A ratio also measures how much *text* survived, not how much *information* did: v3's median accepted
ratio is 0.899 against v6's 0.712, and v3 is the arm that retains more.

Two corollaries, both measured:

- **A ratio floor is actively harmful** (v4, −7.34 pp). Rejecting a collapsed rewrite removes the
  compaction without improving it; 5 of 8 cases could only ever emit sub-floor output and stayed
  permanently blocked. **Blocking a bad action is not the same as taking a good one.**
- **A numeric target does not work on this model** (v5). It obeys the qualitative instruction reliably
  and overshoots every stated ceiling by 400–800 chars in one direction. A pass/fail band test hides
  that — report the **signed** distance.

## The attribution trap — the reason to return to this round

For the 12 losses against the incumbent, an automated classifier built on a capitalisation-based fact
proxy reported **0 compaction-induced / 12 upstream write failures.** Manual inspection — locating the
gold answer string in the actual pre- and post-rewrite blobs — gave the **exact inverse**:

| classification | automated proxy | **manual** |
|---|---|---|
| compaction-induced | **0** | **9** |
| read / reasoning | 0 | 3 |
| upstream write failure | **12** | **0** |

The proxy emits only capitalised specifics. The needed answers were `walk`, `avocado`, `lawn`, `seven`,
`Backup` — ordinary lowercase prose it never produces — so its "lost facts" came from unrelated tokens
and every case fell into the classifier's first branch.

> **Never let a lexical proxy assign a causal class.** A capitalisation-based fact detector cannot work
> on prose.

Second time that property disqualified a proxy here. The `specifics()` helper in the ported mechanism is
kept for *description*, gates nothing, and says so in its docstring.

## Exposure — and why the corpus figure is labelled SPLICED

A7 dispatches only on rec_sum, so kv and vector are unchanged **by construction**.

| | train rec_sum | train overall | dev rec_sum | dev overall |
|---|---|---|---|---|
| incumbent A1–A5 | 56.88 % | **43.56 %** | 45.00 % | **34.52 %** |
| **+ A7 v3** | 64.22 % | **45.87 %** | 65.00 % | **40.48 %** |
| | **+7.34 pp** | **+2.64 pp** *(spliced)* | **+20.00 pp** | **+4.76 pp** *(spliced)* |

Both overall columns combine A7's measured rec_sum result with the incumbent's other two backends. The
splice is sound arithmetic over a population A7 cannot alter, but it is **not an end-to-end
measurement** and is never presented as one.

The outstanding three-backend run is **confirmatory, not a gate**: it converts "unchanged by
construction" into "measured, 0 disagreements on kv and vector". This project has a specific reason to
want that — A6's flags were env-only, and a templates-only run would have silently reproduced the
incumbent under the frozen stack's name.

**Standing rule: quote the exposed-backend delta only with the exposure beside it.**
[`exposure_weighted_delta`](../../anchoropt/learning/exposure.py) computes the split and emits the
warning, so the rule lives in code rather than in a reviewer's memory.

## Open, and honestly open

- **19 of the 21 train gains are unexplained** by any measurement performed. Not "explained by chance"
  — unexplained. A prose-capable fact detector is a prerequisite for closing this.
- **v3's known flaw:** 2 of its 13 train losses are destructive rewrites. Three attempts to prevent
  them (v4, v5, v6) each cost more than the flaw does.
- **Held out never exercised the destructive mode:** all 5 dev compactions retained 0.888–1.000,
  and at n=20 one case is 5 pp.
- **A distinct downstream locus exists:** 2 cases had the fact in the final blob and still answered
  wrong. Not A7's to fix.

## Disposition

**Accepted (2026-08-24); A7 closed to further tuning.** The accepted stack is **A1–A7 + E1**.

A7 is simultaneously the largest single-anchor gain on the backend it reaches and the
weakest-confirmed anchor in the stack: the winning rule was chosen with results visible, every
measurement is single-backend, the corpus figures are spliced, and most of its gains are unexplained.
`QUALIFICATIONS["A7"]` travels with the number wherever it is rendered.
