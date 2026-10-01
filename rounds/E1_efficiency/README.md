# E1 — suppress a useless tool call (efficiency)

**E stands for efficiency.** The first anchor in this line accepted on an objective other than
end-task accuracy.

|  |  |
|---|---|
| locus | `redundancy/deterministic_outcome/known_absent_target` |
| attribution | the model re-issues a removal for a target it has already been told is absent |
| incision point | `post_generation_pre_exec` |
| action | **`suppress`** — the call does not execute |
| objective | **accuracy per LLM call** |
| result | **+0.00 pp, zero flips, 16 of 16 redundant executions eliminated** |

## Why suppress it

The tool is **deterministic**. Same state, same action → same outcome, always. The model asks to
remove a target that is already absent, having been told so; the store has not changed; the second
call cannot return anything the first did not. Executing it buys strictly nothing and costs a step.

So this is not a heuristic guess that the call is unhelpful. It is a **proof** that the call is
uninformative, and suppression follows from the proof.

## What makes it different from A3

Both are `suppress` at the same incision point. They differ on what the model is left with:

```
   A3   suppress   call does not execute   +   call REMOVED from the record
                                              -> the model sees nothing in its place

   E1   suppress   call does not execute   +   the tool's VERBATIM result REPLAYED
                                              -> the model's view is byte-identical
```

E1 saves the execution and preserves the observation. That is the entire basis of its acceptance:
the trajectory outside the withheld calls is *identical*, which is why 0 pp with zero flips is the
**strongest** available evidence rather than a null result.

The two rejected predecessors show what happens when the observation changes:

| variant | delivery | task | side effect |
|---|---|---|---|
| A5-v2 | deleted the call, injected a trailing **user** message | **−1.65 pp** | native not-found 43 → 33 |
| A5-v3 | kept the call, substituted a **new** tool string | **+0.33 pp** | destruction **84 → 90** |
| **E1** | kept the call, replayed the **verbatim** string | **+0.00 pp**, 0 flips | none measurable |

A5-v3 *gained* accuracy and was **rejected**: it said something new, and destructive calls rose.
Saying nothing new is the mechanism.

## The mechanism, in four stages

Order is not incidental — the first implementation had it wrong and saved nothing.

```
  1  LOOK UP        for each proposed remove: is (container, target) recorded not-found,
     (pre-exec)     AND still absent from LIVE state?  -> withhold it
                       |
  2  EXECUTE        hand the executor the SHORTENED batch. The withheld call never runs.
     (shortened)       |
  3  SPLICE         reinsert each cached result at its ORIGINAL INDEX, restoring full length.
                    Delivery zips results against proposals POSITIONALLY -- get this wrong and
                    every later result lands on the wrong call.
                       |
  4  RECORD         memoise only from calls that ACTUALLY EXECUTED.
     (post-exec)
```

Memo key `(container, target)`; value the **verbatim** result string. Scope is per-case, in-process,
never persisted. Invalidated two ways: a hit **re-reads live state** every time (target present again
→ entry dropped, call executes), and a successful add of the target drops the entry.

**Fail-open on every uncertainty** — not a remove, unparseable target, no recorded outcome, unreadable
state → execute normally.

## Evidence

| # | condition | measured |
|---|---|---|
| 1 | no task loss | **+0.00 pp**, 0 gains / 0 losses |
| 2a | withheld from executor | 16 withheld, **0 sent** |
| 2b | executed redundancy | **16 → 0** |
| 2c | splice integrity | 16/16 length-ok |
| 3a | destruction | all five categories identical |
| 3b | episode length | median 3, max 69 — unchanged |
| 3c/3d | A5 canary | 43 vs 43 firings, 43/43 copy invariant |

Policy diff between arms: **one key**.

## Read the scope before quoting the numbers

**The gate fired in ZERO scored episodes.** All 16 firings are in two `vector` **prereq** episodes.
The +0.00 pp is over the full 303-case scored split, but the anchor never fired there.

- ✅ *"16 of 16 redundant executions eliminated in the population where it fires; scored accuracy
  unchanged, zero flips."*
- ❌ *"saved 16 calls across 303 cases"* — implies a rate that does not exist.
- ❌ *"0 pp proves it is safe"* — it is **consistent with** safety on a 16-call, 2-episode,
  single-backend population.

Further limits: no dev run; `kv` claimed but never observed firing; **no frozen
pre-registration** (see [`CRITERIA.md`](CRITERIA.md)); no job id or run artifacts in the working repo,
so the numbers are transcribed rather than recomputable here.

## A prediction worth testing

This repo runs at the official step budget of **20**; the published run used **15**. A tighter budget
makes a returned step scarcer, so **E1 plausibly scores better at 15** — a withheld call hands a step
back, and at 15 that step is likelier to be the one that lets a long episode finish.

**Untested, and the mechanism is indirect:** the firings are in unscored prereq episodes, so a
returned step would have to prevent a prereq from force-quitting mid-write before it could move a
score. Worth one diagnostic run. The default stays 20 either way — it is the officially faithful
value.

## What this round teaches

**Cost is a second axis, and an accuracy table hides it.** Accuracy per LLM call can improve two ways:
raise the numerator (A1–A4) or lower the denominator (E1). A pipeline reported only on accuracy makes
half the available progress invisible — and scores an efficiency anchor as a failure for passing.

## See also

- [`../../docs/EFFICIENCY_CLASS.md`](../../docs/EFFICIENCY_CLASS.md) — the class, the four porting invariants, the caveats
- [`CRITERIA.md`](CRITERIA.md) — the conditions, and why this file is not called `FROZEN.md`
- [`../T3_A3_duplicate/`](../T3_A3_duplicate/) — the other `suppress` anchor, for contrast
- `anchoropt/mechanisms/memoize_guard.py` — the predicate
- `rounds/anchors.py` — `E1` and `E1_EVIDENCE`, single source of truth
