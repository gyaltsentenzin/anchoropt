# N0-DCR — acceptance criteria, FROZEN before results

Written while jobs 1115686–1115697 are still running, so the verdict cannot be shaded after seeing
numbers. Arm: `experiments/n0dcr/n0dcr_dedup_clear.json` (sha256
`ff1b985878831d26cbaeefffe9bc8f2c10bc9c4112c8c812b9bafa4bcc7dec1c`). Control: the frozen incumbent
A1–A5+A7, `experiments/n0a7f/n0a7f_frozen.json`. The two differ in **exactly one** `gate_enabled`
key, verified by policy diff.

Class: **accuracy anchor**, so all four terms of `ACCEPTANCE_RULE_FROZEN.md` apply. This is *not* an
efficiency anchor — it changes what the store contains, so a 0pp delta is a null result here, not a
pass.

---

## 0. Gate 0 — evidence validity (checked FIRST; failure voids the run)

A delta is uninterpretable unless the arm actually differs from the control in the intended way.
Each of these is a **void** condition, not a rejection:

| check | required |
|---|---|
| snapshot cache | `verdict=MISS` on **every** arm shard (`gate_cfg=c8bd210de30d2d22`) |
| A6 contamination | `elen_reroute` == 0 in **all 12** exec logs |
| control purity | `dcr_evict` / `dcr_retry` == 0 in all **6 control** shards |
| backend scope | `dcr_*` == 0 on vector and rec_sum arm shards (A7-exposed backend untouched) |
| double-dispatch | `dcr_evict` count == number of distinct (case, victim) pairs — the 1115318 bug must not recur |
| completion | all 12 shards `rc=0`, n=303 train / n=84 dev after merge |

## 1. Mechanism — causally validated from the gate's own telemetry

Required, and checked **before** any accuracy number is read:

| property | required |
|---|---|
| `dedup_clear_gate` firings | **> 0** on kv (else the arm is a no-op and nothing is attributable) |
| `dedup_clear_copies_before` | **>= 2** on every firing |
| `dedup_clear_copies_remaining` | **>= 1** on every firing — the lossless invariant |
| `dedup_clear_invariant_violated` | **0** |
| `dedup_clear_evict_failed` | 0 expected; any occurrence must be explained, not averaged away |
| `dedup_clear_retry_landed` | true on every firing where the eviction succeeded |
| clears executed in prereq | **strictly fewer** in the arm than in the control |

**Predicted exposure, from `docs/A5_CLEAR_COVERAGE_DIAGNOSIS.md`:** 3 class-1 events on kv/student
(`prereq_24-student-2`, `prereq_25-student-3`, `prereq_29-student-7`), and 2 class-3 events
(`prereq_12-healthcare-2`, `prereq_35-notetaker-3`) that must **fall through untouched**. A firing
count materially above 3 means the trigger is broader than diagnosed and the attribution claim needs
re-deriving before acceptance — the coverage-is-a-property-of-the-trajectory lesson from A7 v5.

## 2. Safety — the N0-R1 non-termination guard

This is the failure mode this arm exists to avoid, so it is a **hard** gate:

| check | required |
|---|---|
| `force_quit` count | **must not increase** vs control on either split |
| prereq step counts on class-3 episodes | unchanged (those episodes are untouched by construction) |
| total prereq steps | may rise slightly (evict+retry adds 2 calls per firing) but **must not** show the R1 signature of a 2×+ blow-up with unresolved writes |
| destroyed-content check | no case may lose content the control retained |

N0-R1 reference numbers: 85 clears refused, **0 of 14 episodes closed**, median kv prereq steps
19 → 42, max 43 → 189. Anything resembling that pattern is a rejection regardless of accuracy.

## 3. Accuracy — the four-term rule

1. **Train, net effect > 0** vs the incumbent on paired cases. Report `(gains, losses)` alongside
   the net, because the shape is diagnostic.
2. **Dev, no aggregate regression** (net >= 0).
3. **Attribution**: the gain must be concentrated on **kv**, the only backend where a clear was
   observed. vector and rec_sum are expected at **exactly 0.00pp** — they are byte-identical arms by
   construction, and any movement there means something leaked.
4. **Safety**: §2 above.

### Headroom, stated as a bound

The diagnosis measured **12** distinct failing kv/student queries whose gold was in memory pre-clear
and absent at query time → **12/303 = 3.96pp** upper bound on train. This is a **bound, not a
prediction**: one eviction frees one slot where the clear destroyed up to 50 entries, and "present
pre-clear" is necessary but not sufficient. A result well below 3.96pp is fully consistent with the
mechanism working.

### What I will NOT do with the numbers

- **No p-value as a gate.** This benchmark is deterministic (byte-identical exec logs across runs);
  report paired (gains, losses) + mechanism instead.
- **No per-cell cherry-picking.** All 3 class-1 events are in **kv/student**, so under
  `CELL_KFOLD_NOTE` this is effectively **n=1 cell** of support. A positive train result confined to
  kv/student must be reported as *possibly cell-specific*, and **dev has no kv/student cell**, so dev
  cannot confirm it. This limitation is stated up front rather than discovered afterwards.
- **No re-mine unless accepted.** Per the standing rule.

## 4. The ladder question (asked regardless of the verdict)

Independent of accept/reject: of the clears that **still execute** in the arm, what share decline for
**"no exact duplicate"**? Measured by `scripts/dcr_ladder_next.py` from the gate's own recorded
decline reasons, not by replay (a replay is what got the pre-clear sizes wrong on the first pass).

- If no-duplicate **dominates** → layer 2 (safe concat/consolidation) is the next test, at those states.
- If it does not → inspect the other decline reasons before adding a layer.

Ladder order is fixed: **dedup → safe concat/consolidation → destructive fallback.** Layer 1 is
deliberately dedup-only so its effect is attributable in isolation.

---

## AMENDMENT after run 1 (jobs 1115686–1115697), recorded as an amendment

Run 1 could not be scored on criterion 3 (its per-firing telemetry was dropped by the sidecar
whitelist), so it is being re-run. Two things in the criteria above were **wrong or missing**, and both
are corrected here rather than silently reinterpreted when run 2 lands. Everything else stands.

### A1. The clear-count check was under-specified — now ARCHIVAL-SCOPED

§1 said *"clears executed in prereq: **strictly fewer** in the arm than in the control"*. Measured on
run 1 that read **17 arm vs 16 control** — more in the arm — which looks like a failed suppression and
is not one:

| | arm | ctl |
|---|---|---|
| archival clears **suppressed** | **15** | 0 |
| archival clears **executed** | **4** | 5 |
| core clears executed | 13 | 11 |

Suppression worked where it fired (`25-student-3`, `29-student-7`). The extra clears are **different
calls at different steps in different episodes** (`26-student-4` archival, `29-student-7` **core**,
`31-student-9` **core**) — downstream trajectory divergence, and **core is not this arm's locus**.

This is rule 4 from the A7 line: *an intervention that adds an LLM call reorders the trajectory, so
per-decision reasoning cannot bound run-level behaviour.* A raw count across reordered trajectories is
not evidence about suppression.

**Corrected criterion:** *ARCHIVAL clears **executed** must be strictly fewer in the arm.* Core clears
and total clears are **reported, not gated** — this arm does not target `core_memory_clear`.

### A2. The eviction cap was an omission — now an explicit decision

A5 caps evictions at 3 per episode (`E1_MAX_EVICTIONS`, *"unbounded eviction is a wholesale clear in
slow motion"*). Run 1's gate had **no cap** and fired 4/5/6 times in single episodes. Not a rule
violation — that constant governs A5's own loop — but exactly the shape the comment warns about, and
leaving it unstated was an omission.

**Decision: `_DCR_MAX_PER_EPISODE = 8`.** Above 3 deliberately, because the measured need is 4–6 per
episode (each blocked write is its own decision point) and capping at 3 would decline capacity the arm
can supply losslessly. Capped at all, because run 1 measured **5 archival clear proposals becoming 19**
— the re-proposal loop is real, and "answer every re-proposal forever" is unbounded destruction
deferred. On budget exhaustion the gate **declines**, so the clear executes exactly as the incumbent's
would: the same fall-through as no-duplicate, for the same reason.

New telemetry: `dedup_clear_episode_count`, `dedup_clear_episode_budget`, plus a `budget spent` decline
reason.

### A3. What run 1 already established (carried forward, not re-litigated)

These do not depend on the dropped telemetry and will not be re-argued when run 2 lands:

- **Control fidelity.** `dcrctl` reproduces the frozen incumbent **exactly**: 139/303 = 45.87%, with
  **0 disagreements across all 303 cases** vs the independently-built `a7fstack_merged_train`.
- **Train, corpus-wide:** +1.65pp (144 vs 139, 9g/4l), entirely from kv **+4.76pp**; vector and
  rec_sum **exactly 0.00pp**.
- **Attribution is exact:** all 13 flips are in kv/student, the only cell where the gate fired.
- **Dev is a construction-level no-op:** all three exec logs byte-identical, 0 firings. Criterion 2 is
  satisfied *vacuously* and dev provides **no** independent confirmation.
- **Safety passes:** `force_quit` 0 in both arms; kv/train prereq **median steps unchanged** (19 vs 19);
  total +17.1%, the expected cost of 2 added calls × 15 firings. No N0-R1 pattern.
- **Dedup is EXHAUSTIBLE:** 4 archival clears still executed once duplicates ran out. Direct evidence
  for layer 2.

Run 2 exists to supply **criterion 3 only**. If its accuracy delta differs materially from +1.65pp
train, that is itself a finding — the cap is a real behaviour change, so a small shift is expected and
must be reported rather than explained away.
