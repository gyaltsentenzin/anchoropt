# Cycle #2 protocol: where each leg runs, and what proves it closed

The cluster executes arms; the Mac runs the search. That split is forced, not chosen: the cluster
checkout has **no `anchoropt/learning/`**, and the search is CPU-only anyway.

```
  [BV/GPU]  regenerate incumbent          runner_c2inc_vector_train.sh          ~17 min
     |                                    ALSO the guard's behavioural proof
  [fetch]   scp traj + results  ->  /tmp/c2/c2inc
     |
  [Mac]     mine -> attribute -> optimize_residual -> next arm
     |      scripts/self_evolve_cycle2.py --incumbent /tmp/c2/c2inc --live
     |
  [Mac]     emit controller spec (JSON) + arm runner
     |
  [BV/GPU]  paired evaluation: arm vs the regenerated incumbent as control    ~17 min x2
     |
  [Mac]     score -> promote/reject     --score <arm dir>
     |      then RE-MINE and report the residual shift
```

## The guard's proof, which gates everything after it

`c2inc` is the promoted arm's own configuration (θ=0.487213) rerun under the guarded evaluator.
a9r487 had **2** prereq firings, each with a merge effect. `c2inc` must have:

| | required | why |
|---|---|---|
| `controller_fired_gate` in `traj/prereq/` | **0** | the guard works |
| `controller_skipped_prereq_gate` in `traj/prereq/` | **> 0** | it was reached and declined, not absent |
| `controller_fired_gate` in `traj/query/` | **~49** | queries are untouched |

All three matter. Zero prereq firings with zero skips would mean the hook never ran at all — a dead
channel masquerading as a fix.

## What makes this cycle #2 rather than cycle #1 again

It starts from the incumbent A9 was promoted **into**, so the residual it mines is the one A9's own
success created. If the loop works, the next intervention is not A9 and not a paraphrase of it. The
`[7] RESIDUAL SHIFT` section measures that directly: which residual keys disappeared, which are new.

## Honest limits, stated before the numbers exist

* **n=65 clean** (customer excluded while the guard is unproven; if `c2inc` shows 0 prereq firings
  then the arm side is clean and the full 89 becomes usable — decided by the telemetry, not by which
  denominator looks better).
* One cell of one split. No significance will be claimed.
* Every gain must sit on a fired case, or the delta has another cause and is reported as such.
