# E1 acceptance criteria — TRANSCRIBED, *not* pre-registered

> **Read this header before the table.** Every other round in this directory ships a `FROZEN.md`
> written **before** the arm was launched. **E1 has no such document.** The conditions below are
> transcribed from `eanchor_verdict.py`, the script that produced the verdict — whose docstring
> states the conditions were "fixed before results are read", but which was committed in the *same
> commit* as the results. That is self-attestation, not pre-registration.
>
> The filename is `CRITERIA.md` and not `FROZEN.md` for exactly that reason. It is a documented
> deviation from this project's own protocol, recorded here rather than smoothed over.
>
> One condition **was** changed after seeing results — the redundancy metric (see below). It is
> defensible, and it is the thing a reader should scrutinise hardest.

## Class

**Efficiency anchor.** E stands for efficiency. Objective: **accuracy per LLM call**. A 0 pp accuracy
delta is a **PASS**. See [`../../docs/EFFICIENCY_CLASS.md`](../../docs/EFFICIENCY_CLASS.md).

## Incumbent and control

| | |
|---|---|
| incumbent | A1–A4 **+ the working repo's A5** (archival-full duplicate eviction) |
| control | `results/a5v2ctl_train/run` — the incumbent with the guard off |
| arm | `results/eanchor_train/run` |
| policy diff | **exactly one key**: `"on_memoized_deterministic_call": true` |
| corpus | `g8_balanced_train_cases.json`, 387 cases = 84 prereq + **303 scored** |
| decode | temperature 0.001 (variance floor is literally zero on this line) |

The one-line policy diff is the strongest structural feature of this experiment: the two arms differ
by a single boolean and nothing else.

**Note on A5.** The incumbent includes the working repo's A5, which is *not* in this showcase repo's
accepted set — it was installed there on an **explicit override** of a frozen zero-loss clause (3
losses). A5 is held **fixed and identical in both arms**, so it differences out of E1's comparison.
But it means E1's measured world is not this repo's A1–A4 world. Stated because it bears on
transfer, not on the internal validity of the paired comparison.

## The conditions

| # | condition | required | measured |
|---|---|---|---|
| **1** | no task loss | `delta >= 0` | **+0.00 pp**, 0 gains / 0 losses |
| **2a** | calls withheld from the executor | `withheld > 0 and sent == 0` | **16 withheld, 0 sent** |
| **2b** | **executed** redundancy fell | `after < before` | **16 → 0** |
| **2c** | splice integrity | `len_bad == 0` | **16/16 length-ok, 0 bad** |
| **3a** | no more destruction | not worse in any category | all five categories identical |
| **3b** | no episode-length growth | `median_arm <= median_ctl` | median 3, max 69 — unchanged |
| **3c** | A5 firings preserved | `arm >= ctl` | **43 vs 43** |
| **3d** | A5 copy invariant | `copies_remaining == firings` | **43/43** |

Conditions 3c/3d use the incumbent's A5 as a **canary**: if E1 perturbed the world, A5's firing count
or its lossless-copy invariant would move. They did not.

### Condition 2b was corrected after seeing results

The verdict initially counted redundant calls in the **decoded proposals**, where withheld calls
still appear *by design* — E1's defining property is that the proposal record is untouched. It read
**16 → 16** and **failed** the anchor. Counting **executed** redundancy reads **16 → 0** on the same
data, with no re-run needed (it is a filter over already-recorded telemetry).

Defensible, because the old metric was arithmetically incapable of registering the change it was
meant to detect. Still a post-hoc metric change against a self-attested pre-registration — which is
why it is called out here rather than left in a commit message.

## What "provably useless" rests on

Determinism, verified in the **untouched control** before the mechanism was built:

| | |
|---|---|
| repeat-still-absent removals | **16**, all `vector` |
| repeats that later **SUCCEEDED** | **0** |
| concentration | `vector_prereq_13-healthcare` (10), `vector_prereq_4-customer-4` (6) |

Zero successful repeats is the load-bearing number: it is what licenses suppression. If any repeat
had succeeded, the tool would not be deterministic for this class and the gate would block legitimate
work.

## Scope limits, stated with the result

- **Zero firings in any SCORED episode.** All 16 are in prereq episodes, which build the store and are
  not scored. The +0.00 pp over 303 scored cases is therefore **consistent with** safety, not evidence
  of safety at scale.
- **One backend.** vector only; `kv` is claimed in metadata with zero observed live firings.
- **No dev run.** Train only. Given the firing population, a dev run would most likely fire
  0 times — itself worth reporting.
- **No job id or run artifacts** in the working repo; numbers are transcribed.

## Verdict

**ACCEPTED** on the efficiency class: no task loss, 16 of 16 redundant executions eliminated, no new
harm on any counter.

Accepted **with** the caveats above, all of which belong in any statement of the result.
