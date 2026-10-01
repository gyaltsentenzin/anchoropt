# CCTU qwen round 5 — pre-registered criteria

**sha256 `fba003d92ff666f384ac357c0ab62d690a7ddb2c91aa06e1bbf6377b35d550c3`** — of this file with this line reading exactly `PENDING`, the state it is hashed in.

```bash
H=$(sha256sum rounds/CCTU_QWEN5/FROZEN.md | cut -d' ' -f1)
sed "0,/$H/s//PENDING/" rounds/CCTU_QWEN5/FROZEN.md | sha256sum   # must equal $H
```

Freeze: confirm every queued run directory is absent, hash, paste, **then run the round-trip above.**

## What this round is, and the disclosure comes first

**The split gate, fired once per episode instead of once per turn.** One arm, two controllers, same
predicates and same θ as rounds 3 and 4, run with `one_shot=True`.

**This is a confirmation run, not a discovery.** The configuration has already been executed, in an
earlier round (not shipped), and its numbers are known: `max_length` **−188**,
`repeated_identical_calls` **+23**, `acc` **0g / 2l**, every class term falling. That run was **void** on
two grounds, neither of them its outcome:

1. `controllers.one_shot=True`, which q3's own FROZEN.md listed as a void condition because q3 specified
   per-turn firing and `--no-one-shot` was never passed. The run did not match its spec.
2. ENGAGEMENT: `...01b` fired in 20 episodes against a screened 18.

Round 4 then ran the same arm per-turn as specified, and was **rejected** on
`repeated_identical_calls` — **+409 against a 200 bar** — while producing the largest benefit on record
(−326). So the per-turn version is decided: it works and it costs too much. Decode is deterministic
(`temperature 0.0`, `seed 42`) and the corpus and control are unchanged, so **this round is expected to
reproduce q3's numbers.** Its purpose is to obtain a *judgeable* run of a configuration whose criteria were
fixed before it was ever measured, not to find out what the number is.

**Every criterion below predates q3's execution.** The benefit bar (`> 118`) was set in round 2. The
`repeated_identical_calls` bar (`> 200`) was set in round 2. The `acc` rule was written to
`benchmarks/cctu/ANCHOROPT.md` at **2026-09-24 14:23:34 UTC**, before q3's FROZEN.md (15:28) and its run
(18:48). Nothing here is calibrated to a result. The only thing this round changes is the ENGAGEMENT
criterion, and it is *replaced* rather than loosened — see below.

**If this round is accepted it replaces the incumbent**, which sits at −118 with `repeated_identical_calls`
+214. The candidate is −188 at +23: more repair, at roughly a tenth of the loop cost.

## ENGAGEMENT is replaced, because the old criterion is unsatisfiable by construction

Rounds 3 and 4 both required each controller to land on exactly its screened episode set, and both failed
it identically: `...01a` on exactly its 66 (0 extra, 0 missed), `...01b` on **20** against a screened
**18**, the excess being `112_0` and `112_1` — one episode, both replicates.

In episode 112, `...01a` fires at turn 0 (`call_times < 1`), the agent calls a tool in response, the episode
becomes committed, and `...01b` fires at turn 2. **`...01a` acting is what moves the episode into `...01b`'s
partition.** `cctu_screens.py` replays the *control*, where `...01a` never fires, so 112 cannot appear in
the screened 18. This is the intended mechanism, not a runtime divergence — and it is not a `one_shot`
artefact: q3 (`one_shot=True`) and q4 (`one_shot=False`) produce the same 20 and the same two ids.

So exact-footprint matching cannot be met by any split gate whose first controller changes the state the
second partitions on. The replacement is structural, derived from that mechanism and checkable without
reference to any arm's outcome:

> **ENGAGEMENT (round 5).** All four must hold:
> 1. `...01a` fires on **exactly** its screened episode set — 0 extra, 0 missed. Its partition is
>    `call_times < 1`, which an intervention can only move episodes *out of*, so there is no mechanism by
>    which it can gain one.
> 2. The two controllers' footprints are **disjoint per turn**: no turn on which two *distinct* controllers
>    fired. Count distinct controllers per turn, not trace rows — q4 had 36 turns with multiple rows and 0
>    with two controllers.
> 3. The **union** of the two footprints is a subset of the incumbent's 78 episodes on train (46 on test).
> 4. Every episode in `...01b`'s footprint that is **not** in its screened set must be one in which
>    `...01a` fired at an **earlier turn** of the same episode. An unexplained excess — one where `...01a`
>    never fired — is a runtime divergence and still voids the arm.

Clause 4 is what the old criterion was reaching for. `cctu_screens.py` now records `firing_episode_ids`, so
all four are set arithmetic rather than a count comparison; rounds 3 and 4 could not name their own excess
because the report stored `18` and not which 18.

## The arm

One arm, two controllers, bundled as one policy with a state-conditioned message. Both `reprompt` at
`post_generation_pre_exec`, both gating on `response_length_over_cap`, partitioned on `call_times`.
Spec: [`arms/cctu_q5_01_split_gate_once.json`](arms/cctu_q5_01_split_gate_once.json).

| controller_id | predicate | screened episodes | `max_length` reached | share |
|---|---|---:|---:|---:|
| `cctu_q5_01a_reprompt_call_tool_first` | `response_length_over_cap AND call_times < 1` | 66 | 722 | **92%** |
| `cctu_q5_01b_reprompt_state_remaining_budget` | `response_length_over_cap AND call_times >= 1` | 18 | 140 | 18% |

Both PASS `support`, `reachability`, `loss_exposure` and `removable_mass` on
[`screen_PSR.json`](screen_PSR.json), regenerated against this control with episode ids recorded. **Arm
budget is 1 and there is no selection to make.**

**`one_shot=True` is the arm**, not an accident: it is the single available lever on intervention frequency.
`one_shot` caps each controller at one intervention per episode, keyed by controller name
(`cctu_middleware.py:373`). θ's `retry_budget` is declared but **not enforced** per-episode, and
`redecide_budget` bounds redecisions per *turn*, so neither can express "cap this controller at N per
episode". The curve therefore has exactly two reachable points, and round 4 measured the other one.

### The two θ texts, unchanged from rounds 3 and 4

> **`...01a`** — "Your draft is already longer than this task allows, and you have not called any tool yet.
> Stop writing the answer directly. Decide which tool you need and call it now; write the final answer only
> once you have its result."

> **`...01b`** — round 1's accepted θ, unchanged: "You have limited remaining interaction rounds and tool
> calls. Before acting again, decide which calls are still necessary and make them together rather than one
> at a time."

`...01a` is human-authored, from round 2's measured loss (episode `112`) plus 8 sampled turn-0 firings in
the control. `...01b` is round 1's own text, reused. Neither interpolates a measured value.

## The measurement position

Unchanged. Control `benchmarks/cctu/results/qwen/train_baseline`, 280 episodes, `repeat 2`,
`max_workers 1`, `temperature 0.0`, `seed 42`, `top_p 1.0`, `max_tokens 1024`, vLLM. Within-job replicate
noise 0; between-job drift 0; the serving node does not move generations; `max_workers` must be 1 — the
measurements are in an earlier round's FROZEN.md (not shipped) and none is re-derived. The
control's own `endpoint` predates the field and is unrecorded; `results_pinned/qwen/train_baseline` is the
same control with `host21:8000` recorded and is byte-identical.

## The counts

`detail.jsonl`'s `violations` field, never `analyze_residual`'s `RESIDUAL` block, and only from a
`detail.jsonl` re-scored since the `violation_counts` fix
([`../../benchmarks/cctu/VENDORED.md`](../../benchmarks/cctu/VENDORED.md)).

| class | train control | test control |
|---|---:|---:|
| `max_length` | **786** | **538** |
| `identifiers` | 1044 | 432 |
| `punctuation` | 764 | 368 |
| `format` | 586 | 202 |
| `min_call_times` | 154 | 44 |
| `min_length` | **0** | **0** |

## The bar — unchanged from rounds 2, 3 and 4

| role | metric | bar |
|---|---|---|
| **benefit** | paired net reduction in `max_length` | **> 118** — must beat the installed anchor |
| **harm** | `acc` | the standing rule below |
| **harm** | `identifiers`, `punctuation`, `format` | net increase > 40 in any one rejects |
| **harm** | `min_length` | **any** net increase rejects. Base 0 on both splits |
| **harm** | `min_call_times` | net increase > 40 rejects |
| **harm** | `repeated_identical_calls` | net increase > 200 rejects. **Replicate-0**, from `analyze_residual`'s loop-shape line, base **85** train / **41** test. This is the term that rejected round 4 at +409 |
| **veto** | `assistant_turns`, `tool_results` | either falling makes a benefit a withdrawal. Both from `detail.jsonl`'s own fields |
| reported | `SR`, `PSR`, `died_of_exhaustion`, every other class | — |

### The `acc` rule

Quoted from [`../../benchmarks/cctu/ANCHOROPT.md`](../../benchmarks/cctu/ANCHOROPT.md), mtime
**2026-09-24 14:23:34 UTC**:

> An arm is rejected on `acc` if its losses exceed the INCUMBENT's losses on the same split, or if it has
> any loss at all when no anchor is installed for that residual. Report gains and losses separately, never
> only the net. An arm with equal losses and greater benefit is admissible and must say so explicitly,
> because it is trading the same accuracy cost for more repair — which is a defensible trade and not a
> free one.

The incumbent is **4g / 2l on train** and **0g / 0l on test**. So up to **2 losses is admissible on train**
and **any loss on test rejects**. That asymmetry means this arm can clear train and fail replication on one
lost answer.

### The trade, decided in advance

q3 measured this configuration at 0 gains / 2 losses. **The decision, made now: yes, that trade is
acceptable** — 188 violations removed at the cost of 2 correct answers destroyed, against an incumbent
costing the same 2 for 118 and eight times the redundant calling. It is a better trade than the incumbent's,
not a free one, and the acceptance must state it in those words. Reporting the benefit without the loss, or
the net (`−2`) in place of `0g / 2l`, is pre-registered as unacceptable.

## The acceptance rule

Each term vetoes before the next is computed.

0. **EXECUTION.** `run_manifest.json` has `controllers.one_shot == true` (**this round's arm** — note the
   inversion from rounds 3 and 4), `max_workers == 1`, `endpoints` recorded, `decode` matching the control.
   Assert before reading any outcome number.
1. **COMPARABILITY.** `check_pairing.py` reports COMPARABLE on both splits.
2. **ENGAGEMENT.** The four-clause criterion above.
3. **WITHDRAWAL.** Neither `assistant_turns` nor `tool_results` falls.
4. **HARM.** `acc` against the standing rule, then the five class terms — `repeated_identical_calls`
   included, and it is the one that killed round 4.
5. **BENEFIT.** Only then, `max_length` on train against **> 118**.
6. **REPLICATION.** Then on `test` against the clean control (base 538), same direction and sign, with
   **zero** `acc` losses.

## What would void this round

- `controllers.one_shot` not `true`, `max_workers` not 1, `endpoint` unrecorded, `decode` differing from the
  control's
- `...01a` firing outside its screened 66, or missing any of them
- two distinct controllers firing on the same turn of the same episode
- an episode in `...01b`'s footprint outside its screened set in which `...01a` never fired
- the union of footprints exceeding the incumbent's 78 (train) / 46 (test)
- an arm from more than one job **without** `endpoints` listing every server and `check_pairing.py`
  clearing the pairing
- scoring against a `detail.jsonl` not re-scored since the `violation_counts` fix
- a tag already on disk. `q5` is unused; q3's and q4's run directories exist and reusing either spec
  filename would make the runner skip this arm as already complete
- reading the benefit from `analyze_residual`'s `RESIDUAL` block

## What this round cannot establish

- **Nothing about intermediate firing frequencies.** `one_shot` is binary and is the only lever; a
  per-controller cap of N would need a middleware change. The curve has two points: −188 at +23 repeated
  calls, and −326 at +409.
- **Nothing new about the numbers**, if it reproduces q3 as expected. What it establishes is that a
  configuration meeting this bar was measured under criteria fixed beforehand — which q3 was not.
- **No `PSR` result.** `SR` and `PSR` have not moved in eleven arms across four rounds.
- **Nothing about the three withheld-validator classes**, which hold 1,226 of 1,935 violations per
  replicate. This is the fifth round on one declared residual.
