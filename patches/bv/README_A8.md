# The A8 executor: four patches, one mechanism, and the order they must be applied in

`docs/ANCHORS.md`'s A8 is *suppress a proposed destructive clear → evict ONE redundant copy → verify an
equivalent copy remains in live state → retry the blocked write verbatim*. Making that reachable to an
autonomous controller on BV took **four** patches, because three separate things had to be true that were
not: the mechanism had to be importable, it had to receive an input nobody supplied, it had to be
verified at the right moment, and its dispatched calls had to survive to dispatch.

Every one of these was diagnosed **from the mechanism's own telemetry, never from an accuracy delta**.
That is the point worth carrying forward: four defects in a row, and not one of them would have been
visible in the arm's score.

## Apply order (each is idempotent; all target the ISOLATED tree only)

| # | patch | what it fixes |
|---|---|---|
| 1 | `bv_dedup_clear_executor.py` | declares the capability and adds the branch at the commitment gate |
| 2 | `bv_a8_pending_supply.py` | populates `_a8_pending`, which patch 1 read and nothing wrote |
| 3 | `bv_a8_verify_after_dispatch.py` | moves `verify()` to after `self._execute()` |
| 4 | `bv_a8_sequence_precedence.py` | stops the withhold branch overwriting patch 1's dispatch list |

Prerequisite: `dedup_clear_lossless.py` must exist in the isolated runtime. **Deliberately not named
`dedup_clear_recovery`** — the shared tree ships an older namesake with a different API (`decide`, no
`plan_recovery`) that wins the import inside the evaluator. See
`bv_phi_module_isolation.py` for the same defect on the loader.

## The five attempts, and what each one taught

| attempt | telemetry observed | defect |
|---|---|---|
| R6a | 26 requested, **26 `AttributeError`** | shared-tree namesake won the import |
| R6b | 26 requested, **26 declines**, 0 errors | import fixed; `_a8_pending` never supplied |
| R6d | 50 requested, **8 evictions, 8 "violated"** | supply fixed; verification ran pre-dispatch |
| R6e | 68 requested, **8 evictions, 8 "violated"** | verification deferred; dispatch list overwritten |
| R6f | *(measuring)* | precedence fixed |

**The defects NESTED, and that is the most transferable lesson.** While the verification clock was wrong
(R6d), "nothing happened" was *fully explained* by reading too early — so there was no reason to suspect
a second cause. Only after fixing the clock did the unchanged verdict at R6e reveal that the calls were
never dispatched at all.

> **When a fix changes the symptom but not the verdict, suspect a second defect behind it rather than
> concluding the mechanism does not work.**

## Two diagnostic properties that made this tractable

**1. Every declining branch records a reason.** R6b's 26 declines all read *"no pending blocked write
recorded -- nothing to retry, not interfering"*. That sentence named the missing input directly. A
mechanism that declined silently would have been indistinguishable from one that had no opportunity.

**2. `verify()` returns its components, not a bare boolean.** R6d/R6e reported
`victim_removed=False, copies_remaining=2` with `copies_before=2` — together saying *"the store is
untouched"*, which is a completely different story from *"the eviction destroyed the last copy"*. A bare
`ok=False` would have read as a genuine safety violation and sent me hunting through the eviction logic
instead of my own call ordering.

## The `_mg_exec_calls` hazard

That variable is the dispatch boundary (`memory_evaluator.py`: *"THE BOUNDARY IS HERE, not at
`decoded`"*), and **three patches now write it**. A later unconditional assignment silently discards an
earlier one, and the control flow gives no hint that an earlier write is load-bearing.

**Before adding a fourth writer: grep every existing writer and state the new precedence explicitly.**

## What is NOT claimed here

That A8 helps. R6f is the first arm in which the mechanism can actually act, and its result is measured
separately with its own capability identity. The cancel-only arm R5 rejected (net −5) stays recorded as
measured — same trigger, different action, different controller.
