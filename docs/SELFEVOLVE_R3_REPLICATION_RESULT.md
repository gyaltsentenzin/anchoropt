# R3 replication (seed 2) — the seed does not vary anything, and my drift diagnosis was WRONG

`DENOMINATOR INTEGRITY: OK` on all three arms, all `rc=0`, `provenance.seed = 2` on every arm.
The frozen scorer (`0270274`) and criterion (`91b3662`) were run unmodified.

## The result: bit-exact re-execution, not a replication

Every reported quantity is **identical** to seed 42, field for field:

| | seed 42 | seed 2 |
|---|---|---|
| control | 19/89 = 21.35% | **19/89 = 21.35%** |
| se3a1 | 25/89, net +6, 34 firings | **25/89, net +6, 34 firings** |
| se3a2 | 24/89, net +5, 28 firings | **24/89, net +5, 28 firings** |
| attributable net (a1 / a2) | +2 / +4 | **+2 / +4** |
| conversion (a1 / a2) | 8.3% / 16.7% | **8.3% / 16.7%** |
| reads per case (a1 / a2) | 1.96 / 1.25 | **1.96 / 1.25** |

Verified at the trajectory level, not just in aggregate: comparing every step field of all 89
episodes across the two seeds gives **0 differing episodes in all three arms**. Per-case outcome
flips: **0**. The only difference between the two runs anywhere on disk is `provenance.seed: 42 -> 2`.

## Why: decoding is greedy by design

`anchoropt/memory_evaluator.py` sends `temperature = 0` with the fixed seed, and its own comment
states the intent: *"W1 determinism: greedy decode + fixed seed. temperature 0.001 was
effectively-but-not-exactly greedy and left ±5-6pp noise that the acceptance gate could not absorb."*

So the seed is **inert**. It is plumbed correctly and recorded correctly (the scorer's seed check
confirmed `seed=2` in all three arms), and it cannot change a token. Varying it was never capable of
testing selection stability, and this round therefore **does not test what it was designed to test**.

## RETRACTED: "step-0 sampling drift of 11/89"

My seed-42 report attributed 11 of 89 cases to *pre-intervention sampling nondeterminism* and built
the drift-free subset on that premise. **That diagnosis was wrong, and decoding is deterministic, so
no such sampling drift exists.**

What I actually measured was a **trajectory-recording artifact**. When the gate fires, the executor
appends its own bookkeeping row as step 0 — `status=None`, `decoded=[]`, `zero_call_reprompt_gate=True`
— which shifts the model's real first decision to step 1. My step-0 comparison was therefore reading
the arm's *gate record* against the control's *first real decision*:

```
ctl: step0 status=executed  decoded=["core_memory_retrieve(...)"]
a1 : step0 status=None      decoded=[]   gate=True     <- bookkeeping, not a decision
     step1 status=executed  decoded=[...]              <- the real first decision
```

Excluding pure gate rows, the arm's first real decision differs from the control on 32-33 of 89
cases — as it must, because the injection changed the prompt. That is the treatment, not drift.

### What survives, and what does not

* **The `on_target_shared` partition survives** and is unaffected: it is defined from the CONTROL's
  trajectory alone (`qualifying_in_control`) plus whether the arm fired, and neither reads step-0
  alignment. So 26 target states, 25/25 per arm, 24 shared, and the attributable numbers
  (+2 vs +4, 8.3% vs 16.7%, 0 losses, 24/24 mechanism) all stand.
* **`opportunity_created` (9 vs 3) needs reinterpretation.** These are cases where the arm fired but
  the control was not a zero-call state. Under determinism that cannot be sampling; it is the
  injection's *downstream* effect — a later turn reaching a zero-call commitment that the control
  never reached. That is a real behavioural difference of the arm, not noise, and se3a1 produces 3x
  more of it.
* **The seed-42 "drift-free subset" language is wrong** wherever it says sampling. The subset itself
  was still the right population, for a different reason than I gave.

## Verdict: NOT PROMOTED

The frozen scorer prints `PROMOTE se3a2: YES`, and all four criteria pass. **I am not promoting**,
because the criteria were written to be evaluated on an *independent* replication and this run is a
re-execution of the same computation. Passing them twice on identical bytes is one observation, not
two, and the promotion rule's purpose was to establish stability.

**Nothing about se3a2 is contradicted** — it is positive, breaks nothing, and its mechanism holds
24/24. There is simply no new evidence.

## What an actual replication requires

The substrate is deterministic, so independent variation must come from somewhere other than the
decode seed. Options, none yet run:

1. **`temperature > 0`** for a genuinely stochastic repeat — but this deliberately reintroduces the
   ±5-6pp noise the greedy setting exists to remove, and 5-6pp swamps a 2-case effect.
2. **A different (backend × scenario) cell** — the held-out split is a different scenario, which
   tests generalization rather than seed stability, and is the harsher test the project already uses.
3. **Case-level resampling / bootstrap over the existing 24 on-target cases** — no GPU, but it
   quantifies sampling error in the *estimate*, not stability of the *selection*.
4. **Paraphrase-level variation of the two eta texts** — tests whether the depth asymmetry is a
   property of the instruction *class* rather than of two specific strings.

Given n=24 on-target and a 4-vs-2 conversion difference, option 2 or 3 is the honest next step.
Option 1 would produce a number that cannot resolve the effect.
