# The efficiency anchor: a second acceptance class

**E stands for efficiency.** A1–A4 were each accepted because they made the agent *more accurate*.
E1 was accepted because it made the agent *cheaper at identical accuracy* — and that requires a
different bar, because judged as an accuracy anchor E1 scores +0.00 pp and looks like a null result.

This document exists because the wrong bar produces the wrong verdict in both directions: an
efficiency anchor scored on accuracy is rejected for passing, and an accuracy anchor scored on cost
is accepted for doing nothing.

## The two classes

| class | objective | accepted when |
|---|---|---|
| **accuracy anchor** (A1–A4) | end-task accuracy | paired delta is positive **and** survives the harm clause — A1 +4.62, A2 +2.31, A3 +2.64, A4 +3.63 |
| **efficiency anchor** (E1) | **accuracy per LLM call** | **no task loss** + **provably useless work removed** + **no new harm**. A 0 pp delta is a **PASS**, not a null. |

The objective is a ratio, so there are two ways to move it. A1–A4 raise the numerator. E1 lowers the
denominator. Reporting only accuracy makes the second kind of progress invisible.

### Why 0 pp is the *strongest* result here, not the weakest

Counter-intuitive and worth stating plainly. For a *behaviour-preserving* anchor,
**+0.00 pp with zero flips is stronger evidence than a small positive delta would be.**

A small positive delta on a behaviour-preserving change means something moved that should not have —
the intervention was supposed to be observationally inert, so a gain indicates it leaked into
behaviour and got lucky. Zero flips means the trajectories are identical outside the withheld calls,
which is exactly the claim. The two variants that came before E1 make the point:

| variant | what it changed | task | side effect |
|---|---|---|---|
| A5-v2 | deleted the call, injected a trailing **user** message | **−1.65 pp** | native not-found errors 43 → 33 |
| A5-v3 | kept the call, substituted a **new** tool string | **+0.33 pp** | destructive `vector_core_remove` **84 → 90** |
| **E1** | kept the call, replayed the **verbatim** tool string | **+0.00 pp**, 0 flips | none measurable |

A5-v3 is the sharp case: it *gained* accuracy and was **rejected**, because it said something new to
the model and destruction rose. Saying nothing new is the whole mechanism.

## The pattern

> **known state + known action + deterministic tool → reuse the recorded outcome instead of
> executing again.**

The rationale is determinism, not heuristics: given the same state and the same action, a
deterministic tool returns the same outcome every time. A second identical call cannot inform
anything. It can only cost a step.

E1 instantiates this for the one class whose determinism is **verified against live state**: a
removal whose target is absent. First attempt executes and its result is memoised verbatim; a repeat
while the target is *still* absent replays that exact string.

## Where it sits in the taxonomy

**`suppress` at `post_generation_pre_exec`** — it suppresses a useless tool call.

It needs the proposed call (so not `pre_generation`) and must stop it from running (so not
`post_execution`). Exactly one admissible cell, like A3 and A4.

E1 and A3 are both suppression, and they differ on a second axis the action family does not encode:

```
                  does it EXECUTE?        what does the model OBSERVE?
   A3  suppress   no                      nothing -- the call is gone from the record
   E1  suppress   no                      the tool's VERBATIM result, replayed
```

Both cancel the proposed action. Only A3 also removes the observation. E1's whole acceptance
argument rests on *not* doing that — so if you port the taxonomy, keep "does it execute" and "what
does the model see" as separate questions.

**One consequence for implementation:** E1 spans two points. It suppresses at
`post_generation_pre_exec` and *records* at `post_execution` (it needs the result to memoise). The
recording half changes no behaviour, but a faithful port needs a hook at both.

## The four invariants — port these or it is not this anchor

Each was learned from a failure in the working repo.

**1. Look up strictly BEFORE execution; execute the shortened batch.**
The first implementation memoised *after* the executor ran and saved nothing — a behaviour change
dressed as an efficiency one. Verify with a counting stub that executor invocations for the targeted
calls are **0**. "Fewer proposals" is not a saving; only fewer *executions* is.

**2. Splice the cached result back at its original index.**
Delivery zips results against proposed calls **positionally**. If the withheld call's slot is not
refilled, every subsequent result shifts onto the wrong call and the whole turn is mislabelled. The
length invariant (`len(results) == len(proposed)`) is not bookkeeping; it is correctness.

**3. Replay the tool's VERBATIM result. Never author a new string.**
A5-v3 changed one message and moved destruction 84 → 90. Behaviour preservation must be *by
construction*, not by hope.

**4. Record only from calls that actually executed.**
A withheld call has no fresh result to learn from. Recording from it would memoise the replay.

## Measuring it: count what the gate changes

**The redundancy metric must count EXECUTED redundant calls, not proposed ones.** In the working
repo the first verdict counted redundant calls in the *decoded proposals*, where withheld calls
still appear by design — it read **16 → 16** and failed the anchor. Counting *executed* redundancy
reads **16 → 0** on the same data.

This is a metric correction made *after* seeing results, which is normally a red flag. It is
defensible here only because the old metric was arithmetically incapable of registering the change
it was meant to detect — the gate's defining behaviour is that proposals are preserved. Recorded
rather than hidden, because the reader should judge it.

## Two consequences for the loop

**A behaviour-preserving anchor does not trigger a re-mine.** Re-mining searches for loci whose
trajectories shifted; a 0 pp / zero-flip arm has none. There is nothing new to see.

**The efficiency residual is tracked separately from the accuracy residual.** E1's locus is removed
from the efficiency residual and leaves the accuracy residual untouched. In the working repo this is
so far a **design commitment, not implemented infrastructure** — the accuracy residual is a real
ledger artifact; the efficiency residual is not yet.

## What is not settled

Stated at the same prominence as the result, because the population is small and the provenance is
thinner than for A1–A4.

- **It fired in ZERO scored episodes.** All 16 firings are in two `vector` **prereq** episodes
  (`vector_prereq_13-healthcare` ×10, `vector_prereq_4-customer-4` ×6). The +0.00 pp is over the full
  303-case scored split, but the gate never fired there. So the result is **consistent with** safety,
  not evidence of safety at scale. Honest phrasing: *16 of 16 redundant executions eliminated in the
  population where it fires; scored accuracy unchanged.* Never *"16 calls saved across 303 cases"* —
  that implies a rate.
- **No frozen pre-registration.** Every A1–A4 round has a `FROZEN.md` written before launch. E1 has
  none; its criteria are transcribed from the verdict script that produced the numbers. A documented
  deviation from this project's own protocol.
- **No run artifacts and no job id** are recorded in the working repo, so the numbers are transcribed
  rather than recomputable.
- **`kv` is claimed but never observed firing.** The gate's metadata claims `kv` and `vector`;
  observed firings are vector-only. kv determinism is unit-tested, not measured live.
- **The zero-flip bar depends on deterministic decoding.** This line's variance floor is literally
  zero (temperature = 0.001; 0/303 outcome flips across replicates). Under sampling, a 0 pp /
  zero-flip arm is unobtainable and **this acceptance criterion does not transfer as written** — the
  single biggest portability caveat, and it is not stated in the working repo's own section.
- **Step budget.** This repo uses 20 (the official `MAXIMUM_STEP_LIMIT`); the published run used 15.
  A tighter budget makes a returned step scarcer and more valuable, so E1 plausibly scores *better* at
  15. **Untested**, and the mechanism is indirect: the firings are in unscored prereq episodes, so a
  returned step would have to prevent a prereq from force-quitting mid-write to show up in a score.
  20 remains the default regardless — see [`FIDELITY_AUDIT.md`](FIDELITY_AUDIT.md) divergence #3.
- **Invalidation covers reappearance, not clearing.** A successful add of the target invalidates the
  memo, and every hit re-reads live state. A container *clear* is not an explicit trigger — safe by
  direction only (clearing makes targets more absent), which is an argument about this corpus, not a
  general one.

## Applying this to your benchmark

1. **Find a deterministic tool and a repeated call against unchanged state.** The candidate is any
   `(state, action)` pair whose outcome cannot vary. Verify determinism from *live state at decision
   time*, not from a reconstruction — reconstruction produced a wrong answer three times in the
   working repo before this anchor read live state instead.
2. **Characterise the population in the untouched control first.** How many repeats, in how many
   episodes, and — critically — **how many later succeeded?** If any repeat succeeds, the tool is not
   deterministic for that class and suppression would block legitimate work. Here it was 0.
3. **Decide the bar before running.** Write down: no task loss, the executed-work metric, and the
   harm counters. Freeze it. (E1 did not, and it is the weakest thing about it.)
4. **Fail open on every uncertainty.** Unparseable target, unreadable state, no recorded outcome →
   execute normally. A memoising gate that guesses is a correctness bug, not an efficiency win.

## See also

- [`../rounds/E1_efficiency/`](../rounds/E1_efficiency/) — the round record and transcribed criteria
- [`INCISION_POINTS.md`](INCISION_POINTS.md) — why `post_generation_pre_exec` is the only cell
- [`KEEP_OR_DEFER.md`](KEEP_OR_DEFER.md) — the S1–S5 rules for the accuracy class
- `rounds/anchors.py` — `E1` and `E1_EVIDENCE`, the single source of truth for these numbers
