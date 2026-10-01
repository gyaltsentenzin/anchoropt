# CCTU qwen round 5 — ACCEPTED. The split gate, fired once per episode, replaces the incumbent.

Judged against [`FROZEN.md`](FROZEN.md), sha256
`fba003d92ff666f384ac357c0ab62d690a7ddb2c91aa06e1bbf6377b35d550c3`, frozen before the arm ran and **whose
paste round-trips**:

```bash
H=fba003d92ff666f384ac357c0ab62d690a7ddb2c91aa06e1bbf6377b35d550c3
sed "0,/$H/s//PENDING/" rounds/CCTU_QWEN5/FROZEN.md | sha256sum   # -> $H
```

**Installed as [`controller.json`](controller.json)**, replacing
`cctu_c1_258_reprompt_state_remaining_budget` from an earlier round (not shipped). This is the first
anchor in this benchmark accepted under criteria every one of which was fixed before the arm was executed.

## The result, stated the way FROZEN.md requires

**188 `max_length` violations removed on train at the cost of 2 correct answers destroyed** — `0 gains / 2
losses`, not the net — against a replaced anchor costing the same 2 answers for 118 violations and eight
times the redundant calling. **A better trade than the incumbent's, not a free one.**

On `test`: 50 violations removed, **0 gains / 0 losses**.

## Every term, both splits

| step | train | test |
|---|---|---|
| 0 EXECUTION | ✅ `one_shot=True`, `max_workers=1`, endpoint recorded, decode matching | ✅ single job, single endpoint |
| 1 COMPARABILITY | ✅ 202/202 non-firing identical | ✅ 74/74 |
| 2 ENGAGEMENT | ✅ all four clauses | ✅ 0 two-controller turns, union 46 |
| 3 WITHDRAWAL | ✅ `assistant_turns` +30, `tool_results` +228 | ✅ +0 / +50 |
| 4 HARM `acc` | ✅ **0g / 2l** — equals the replaced anchor's 2, does not exceed | ✅ **0g / 0l** |
| 4 HARM `identifiers` / `punctuation` / `format` | ✅ −122 / −106 / −50 | ✅ −34 / −44 / −20 |
| 4 HARM `min_length` | ✅ +0 | ✅ +0 |
| 4 HARM `min_call_times` | ✅ −16 | ✅ +28 |
| 4 HARM `repeated_identical_calls` | ✅ 85 → 108 (**+23**), bar > 200 | ✅ 41 → 49 (**+8**) |
| 5 BENEFIT | ✅ **−188**, bar > 118 | — |
| 6 REPLICATION | — | ✅ **−50**, same direction and sign, zero `acc` losses |

## KNOWN LIMITATION: it loses to the anchor it replaces on the held-out split

| split | replaced anchor | this anchor | |
|---|---:|---:|---|
| train | −118 | **−188** | better by 70 |
| test | **−56** | −50 | **worse by 6** |

FROZEN.md step 6 required "same direction and sign" on `test` and **did not require beating the incumbent
there.** It has the sign; it does not have the margin. Proportionally the gap is wider than the raw numbers
suggest: train is 23.9% of its base, `test` only 9.3%.

**This is a defect in the bar, not a finding, and it is disclosed rather than repaired after the fact.**
The benefit bar was written relative to the incumbent *on train* and left absolute on `test`, so the rule as
frozen cannot distinguish this arm from one marginally worse than what is already deployed. The same class
of error as round 2's "zero losses" `acc` bar. The corrected form — benefit stated relative to the incumbent
**on each split** — is recorded in
[`../../benchmarks/cctu/ANCHOROPT.md`](../../benchmarks/cctu/ANCHOROPT.md) for round 6, and is **not**
applied to this round.

**Why it is installed anyway**, stated so a reader can disagree with the reasoning rather than guess at it:

* It passed every pre-registered term, including replication. Reversing that on a comparison the rule did
  not make is the post-hoc move the whole protocol exists to prevent.
* −6 on 60 test episodes is well inside what that split can resolve; −70 on train is not.
* The cost improvement replicates on **both** splits and is large: `repeated_identical_calls` +23 and +8
  against the replaced anchor's +214, and `min_length` +0 against its +6 on `test`. That is the part least
  likely to be noise.

A reader who weighs the held-out split more heavily than the training split would not install this, and
that position is defensible on this evidence.

## This round was a confirmation, and said so before it ran

[`FROZEN.md`](FROZEN.md) opened by disclosing that the configuration had already been executed in an
earlier round (not shipped) — `−188`, `+23`, `0g / 2l` — and voided there on two grounds, neither of
them its outcome: `one_shot=True` contradicted that round's own spec, and its ENGAGEMENT criterion was
unsatisfiable. Decode is deterministic and the control unchanged, so this round was **expected to
reproduce those numbers, and did, exactly.**

What it establishes is therefore not the number but that a configuration meeting this bar was measured under
criteria fixed beforehand. Every bar traces to a date before that round ran: benefit `> 118` and
`repeated_identical_calls > 200` from an earlier round's FROZEN.md (not shipped), the `acc` rule
written to `ANCHOROPT.md` at **2026-09-24 14:23:34 UTC** against q3's FROZEN.md at 15:28 and its run at 18:48.

## The engagement criterion that finally passed, and why rounds 3 and 4 could not

Both earlier executions required each controller to land on exactly its screened episode set, and both failed
identically: `...01a` on exactly its 66, `...01b` on **20** against a screened **18**. Round 5 replaced that
with four structural clauses, and all four hold:

1. `...01a` on exactly its screened 66 — **0 extra, 0 missed**
2. **0 turns** with two distinct controllers firing
3. union **78** on train, **46** on test — within the replaced anchor's footprint
4. `...01b`'s excess is `112_0` / `112_1`, and in both `...01a` fired at **turn 0** with `...01b` at
   **turn 2** — **attributable**, zero unexplained

Clause 4 is the substance. In episode 112 `...01a` fires while `call_times < 1`, the agent calls a tool in
response, the episode becomes committed, and `...01b` becomes eligible. **`...01a` acting is what moves the
episode across the partition boundary.** `cctu_screens.py` replays the *control*, where `...01a` never
fires, so 112 cannot appear in the screened 18. Exact-footprint matching is unsatisfiable by construction
for any split gate whose first controller changes the state the second partitions on — and an *unexplained*
excess still voids, so the replacement is narrower in the way that matters, not merely looser.

It is not a `one_shot` artefact: q3 (`one_shot=True`), q4 (`one_shot=False`) and this round all produce the
same 20 and the same two ids.

## Why once-per-episode and not per-turn

`one_shot` is the only lever on intervention frequency. It caps each controller at one intervention per
episode, keyed by controller name (`cctu_middleware.py:373`). θ's `retry_budget` is declared but **not
enforced** per-episode, and `redecide_budget` bounds redecisions per *turn*, so no intermediate cap is
expressible. The curve therefore has exactly two points, both now measured:

| firing | `max_length` train | `repeated_identical_calls` | verdict |
|---|---:|---:|---|
| once per episode (this round) | −188 | **+23** | **ACCEPTED** |
| every over-cap turn (earlier round, not shipped) | **−326** | **+409** | rejected, bar > 200 |

Per-turn firing produces by far the largest reduction ever measured here and buys it with redundant tool
calling — `tool_results` +1000 against this round's +228. That term did not exist before round 2; it was
added because round 1's arms ran 164–299 unwatched, and round 4 is the only arm it has ever rejected.

## What this round establishes

- **A state-conditioned split beats a single instruction**, −188 against −118 for the same accuracy cost,
  and at roughly a tenth of the loop cost.
- **Intervention frequency is a real lever with a real cost curve**: −188/+23 against −326/+409.
- **The engagement criterion for split policies has to be structural.** Two rounds were voided by a
  criterion that could not be met; the replacement is checkable and still catches divergence.
- **`cctu_screens.py` now records `firing_episode_ids`**, which is why this round could name its own excess
  when rounds 3 and 4 could not.

## What it does not establish

- **No `PSR` result.** `SR` and `PSR` are unchanged to the decimal on both splits, as in every qwen arm
  measured — **thirteen now, across five rounds.** A `max_length` reduction is progress toward the headline
  metric and is not that metric.
- **Nothing about intermediate firing frequencies**, which would need a per-controller cap in the middleware.
- **Nothing about the three withheld-validator classes**, which hold 1,226 of 1,935 violations per
  replicate. This is the fifth round on one declared residual.
- **Nothing that transfers to granite**, whose residual is round exhaustion rather than refusal at the answer.

## Caveats on the record

- The train control's `endpoint` predates the field and is unrecorded — the caveat every round here carries.
  `results_pinned/qwen/train_baseline` is the same control with `host21:8000` recorded and is
  byte-identical to it.
- The `test` arm's endpoint differs from the test control's. Per [`FROZEN.md`](FROZEN.md) that does not void
  anything — node-invariance is measured — and `check_pairing.py` cleared every non-firing episode on both
  splits.
- `repeated_identical_calls` figures are **replicate 0**, from `analyze_residual`'s loop-shape line, as
  FROZEN.md pins. Round 3's RESULT.md reported this term doubled against a spec pinned to replicate 0.
- `one_shot=True` is **part of the installed policy**, not a default, and is recorded in
  [`controller.json`](controller.json) under `requires`. Installing these specs at `one_shot=False`
  reproduces round 4, which was rejected.

## What comes next

1. **Fix the benefit bar** — relative to the incumbent on *each* split. Recorded in `ANCHOROPT.md`, to be
   adopted by round 6 and not applied retroactively here.
2. **The open lever is a per-controller firing cap.** −326 exists at +409; whether a cap of 2 or 3 keeps
   most of the benefit at an acceptable cost is untested and needs a middleware change.
3. **`PSR` has not moved in thirteen arms.** If the project's claim needs `PSR`, five rounds on this
   residual have produced two anchors and no movement in it, and the withheld-validator decision is now the
   larger lever by a wide margin.
