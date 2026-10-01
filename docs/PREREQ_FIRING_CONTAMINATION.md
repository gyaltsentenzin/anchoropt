# The gate fires during PREREQ episodes, and prereqs write the store queries read

Found while auditing the held-out round, then confirmed on vector/train. Documented and **not solved
generally** — per the v0.1 priority, this records the defect and its scope rather than fixing it.

## The mechanism

A cell's episodes split into **prereq** (setup: they write the memory store) and **query** (scored:
they read it). The A4 executor is armed for the whole run, so it fires on prereq episodes too
whenever a prereq turn ends without a tool call. Its injection then changes what the prereq writes,
so the arm's queries are scored against **a store the arm itself modified**.

That breaks the pairing: the control and the arm are no longer answering the same question.

## Measured

| run | prereq episodes | prereq gate firings | prereq trajectories |
|---|---|---|---|
| held-out **kv**-customer, control | 10 | 0 | — |
| held-out **kv**, se3a1 | 10 | **22** | differ from control |
| held-out **kv**, se3a2 | 10 | **12** | differ from control |
| held-out vector-student, all arms | 10 | **0** | byte-identical |
| held-out rec_sum-finance, all arms | 7 | **0** | byte-identical |
| **train** vector, control (`se3ctl`) | 27 | 0 | — |
| **train** vector, promoted (`se1inc`) | 27 | **6** | differ from control |

## What it invalidated, and what it did not

**Invalidated:** the kv-customer held-out cell as a paired comparison. It produced exactly the
symptom that exposed it — `memory_kv_23-customer-23` flipped to correct in se3a2 **on a case where
the arm never fired**, with all three arms issuing different step-0 calls. Pooled net was +4 while
only 3 gains sat on fired cases; that arithmetic gap is what led to the audit. kv is excluded from
the causal comparison, and a `[0b] PREREQ CONTAMINATION` check now reports firings and prereq-
trajectory signatures per cell so the next instance is detected rather than discovered.

**Not invalidated:** the vector-student and rec_sum-finance held-out cells (0 prereq firings, prereq
trajectories byte-identical across arms) — the promotion evidence rests on those.

**Partially affected:** the train rerun of the promoted incumbent, which has 6 prereq firings. The
before/after accuracy delta reported for it (19/89 → 24/89, +5.62 pp) therefore mixes the
controller's query-time effect with a small store difference, and should be read as approximate. It
does **not** affect the loop's ability to re-attribute and re-mine: the residual is computed from the
episodes that actually failed, whatever store they ran against.

## Why it is not fixed here

The fix is to arm remedies only for query episodes (or to build prereqs once, gate-free, and share
that store across arms). Both touch the evaluator's episode dispatch, which is the frozen substrate
every prior anchor was measured on — a change there would invalidate cross-round comparability for a
problem that affects one of three held-out cells. Deferred to v0.2 with the rest of the substrate
work.

**Scope for anyone reading a number:** any cell where the arm fires during prereqs is not a clean
paired comparison. Check `[0b]` output before quoting a delta.
