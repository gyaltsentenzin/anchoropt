# The AppWorld anchors that worked — published for reproduction

Two anchors cleared their model's promotion gate and held up on a held-out split. This file gives each
one **exactly** as it ran: boundary, the signal's executable firing condition, the action, and the literal
injected text. Everything is copied from the `controllers.json` the scoring job consumed, so what is
written here is what was measured.

For the earlier BFCL-era anchors A1–A8 see [`ANCHORS.md`](ANCHORS.md); this file is AppWorld only and
shares no anchors with it. Measured numbers and every caveat behind them live in
`APPWORLD_RESULTS.md` (not in this branch, see [`docs/RESULTS_POLICY.md`](../docs/RESULTS_POLICY.md)). Reproduction prerequisites — including the one thing this
repo cannot give you — are in [`REPRODUCING_EXTERNALLY.md`](REPRODUCING_EXTERNALLY.md).

**Read the two warnings before reusing either anchor.** One is about what the text says; the other is
about how the gain was selected.

---

## Where they attach

Both fire at **`post_generation_pre_exec`** — after the model has produced a cell, before that cell
executes. That is the only boundary in this project where anything worked. The `post_execution` boundary
("read the error above and retry") produced nothing on any model: minimax −3, qwen −3, granite −1.

| | model | signal | fires when | action | train | held-out |
|---|---|---|---|---|---:|---:|
| **AW1** | MiniMax M2.5 | `pagination_unbounded` | `targets_paginated_api` **and not** `passes_page_index` | `reprompt` | **+5.0pp** | **+5.4pp** |
| **AW2** | granite-4.1-30b | `unknown_api_targeted` | `api_path_unknown` | `reprompt` | **+3.3pp** | **+3.3pp** |

Held-out figures are pooled over **two** arm passes against pooled baseline repeats. A single pass is not
a result here: AW2's two passes are raw **+4 and 0**, so pass 1 alone would read +5.5pp and make held-out
look better than train, while pass 2 alone reads as nothing. Pooled, AW2 lands on its train figure to the
decimal. AW1's passes are +3 and +4, genuinely low-variance.

Both are **raw-positive** on both splits — they gain outcomes overall, not only on the episodes the
attribution filter credits them for. That distinguishes them from every other arm measured here, and from
the two arms the historical `net > 0` rule promoted, which were raw-*negative* and lowered the score.

---

## AW1 — MiniMax M2.5: page to exhaustion before using results

Fires when the proposed cell calls one of the 70-of-473 AppWorld APIs that declare `page_index` and does
not pass it. The cell then executes perfectly and returns page 1, and the episode goes on to fail the task
for missing data. **Nothing errors**, which is why no post-execution signal can see it.

```json
{
  "boundary": "post_generation_pre_exec",
  "signal": "pagination_unbounded",
  "action": "reprompt",
  "variant": "page_through_all_results",
  "eta": {
    "instruction": "This API is paginated. Loop over page_index from 0, collecting every page until one comes back empty, before using the results. Do not stop at the first page.",
    "retry_budget": 1
  }
}
```

Measured: **65.6 → 70.6** held-in (2 baseline repeats vs 2 arm passes, 118/180 → 127/180), **55.2 → 60.6**
held-out (3 baseline draws vs 2 arm passes, 149/270 → 109/180; passes raw +3 and +4). Fires on 39–42 of 90
held-in episodes and 64–67 of 90 held-out.

**The content carries the effect, and a matched control proves it.** The same signal, boundary, action and
firing set with a generic text (`check_api_before_running`) scores **−2.3pp** — a 7.3pp separation on
matched firing (39/42 vs 41/39 episodes). Exhaustive paging is load-bearing: a *bounded* variant
("page through at most a few pages") measures only +1.67 on the draw mean, and `suppress` — cancel the
call entirely — measures −2.00.

---

## AW2 — granite-4.1-30b: at an undocumented API call, fetch a little and proceed

Fires when the proposed cell targets an app/API pair that is not in the documentation
(`api_path_unknown`). On granite this is the dominant failure: its residual carries **94**
`api_not_found` errors against minimax's **1**.

```json
{
  "boundary": "post_generation_pre_exec",
  "signal": "unknown_api_targeted",
  "action": "reprompt",
  "variant": "page_first_pages_then_proceed",
  "eta": {
    "instruction": "This API is paginated and you have only the first page. Fetch a few more pages, then continue with what you have. Do not keep paging until the results run out.",
    "retry_budget": 1
  }
}
```

Measured: **30.0 → 33.3** held-in (1 draw vs 1 pass, 27/90 → 30/90, attributable +6 / raw +3, fires on 37
of 90), **26.7 → 30.0** held-out (2 baseline repeats vs 2 arm passes, 48/180 → 54/180; passes raw +4 and
**0**, fires on 49 then 44 of 90 — the variance is real and wider than AW1's). It cleared
`--min-net 5 --min-firings 10` on train *before* being sent to held-out, so its held-out number is a
protocol-clean validation.

### ⚠ The instruction does not describe the condition it fires on — do not reuse the text as written

**This is the most important caveat in this file.** The signal detects an **undocumented API path**. The
injected text talks about **pagination**. They are unrelated: the variant pool is shared across signals, so
the search tried every (signal × action × variant) combination and this pairing won on measurement, not on
meaning. `adapter.py` already notes the converse case — `check_api_before_running` "was written for
`unknown_api_targeted` and reads as nonsense" elsewhere.

So AW2 is **not** evidence that "a pagination instruction fixes unknown APIs." What the arm family
actually shows, on five arms firing on 37–42 episodes each and differing *only* in injected text:

| variant | what it tells the model to do | attributable |
|---|---|---:|
| `page_first_pages_then_proceed` | stop, get a little, **proceed with partial data** | **+6** |
| `suppress:cancel_proposed_call` | **cancel the call**, no text at all | **+5** |
| `check_api_before_running` | go look the API up first | +3 |
| `page_through_all_results` | page exhaustively | +2 |
| `emit_short_runnable_cell` | emit a short cell | +0 |

Ordered that way, the reading is mechanistic rather than semantic: on a weak model about to call an API
that does not exist, **interventions that abandon the doomed call score highest, and the one that also
licenses proceeding with what it already has scores highest of all.** The two top arms are the two that
stop the call; the bottom arm does not stop it. That `suppress` — which injects no text whatsoever —
reaches +5 is the strongest evidence that AW2's gain is not coming from the literal words.

**Consequence for anyone reusing this.** Rewrite the text to say what the mechanism is ("this API is not
documented; do not call it — continue with the information you already have") and re-measure. We have not
run that arm, so we cannot claim it performs the same. Publishing the tested text verbatim with this
warning is more honest than publishing a cleaned-up version we never measured.

---

## What did NOT work, stated because it bounds the claim

**qwen3.6-35B-A3B has no anchor, and scoring of its full candidate set is still in progress.** Of the arms
measured so far its best is the **generic control** (+1 — one task, inside the noise floor), not the
treatment (+0); scored on held-out anyway that control **reverses to −3.3pp**, which is evidence the gate
was right to reject it rather than a failed anchor. qwen's earlier reported +5 was a
[generation-config confound](APPWORLD_RESULTS.md#retracted-qwens-5-was-a-generation-config-confound-on-both-splits)
and is retracted. Treat the qwen row as an open null, not a closed one — see *Scoring still owed* below.

The structural reason matters more than the null. **AnchorOpt's headroom is inversely proportional to
baseline strength**: the upside ceiling is the residual, the downside exposure is the passing set. Screened
on its own residual, *every* signal proposed for qwen fires on 2–3× more passing episodes than failing
ones — best ratio **0.47**, against granite's **7.00**. qwen passes 74% of `train` (23-task residual, 67
episodes exposed); granite passes 30% (63 upside, 27 exposed).

So AW2 exists because granite is weak, not because its wording is better — and on a 74%-baseline model
every available signal has negative expected value before any instruction is written. Anyone reproducing
this on a strong model should expect the same, and should screen before spending episodes.

### Scoring still owed on qwen, and a second cap confound found while checking

qwen's three measured arms came from a round whose `incumbent_id` is `.../train` — **uncapped** — while
every qwen arm ran at `max_completion_tokens 8192` and was paired against `train_r2`. That is the
[cap confound](APPWORLD_RESULTS.md#retracted-qwens-5-was-a-generation-config-confound-on-both-splits)
again, on the *proposal* side this time: the signals were mined from a residual with a different error
distribution than the one they were tested against. The correctly-mined round (`incumbent_id = train_r2`,
30 arms) exists and had never been scored.

Two things keep this from overturning the null. The signals actually measured — `generation_truncated` and
`execution_failed` — **both appear in the correctly-mined round as well**, so the mis-mining did not cause
wrong signals to be tested; it caused signals to be *missed*. And screening the correct round changes no
verdict: 17 arms clear the firing floor, and the best ratio among them is `error_kind_is_auth_error` at
**0.47**. The typed error kinds that might have been sharper all fall below the floor (`name_error` 9,
`other_error` 5, `precondition_unmet` 3, `python_error` 2), and every `result_chars` threshold arm is a
twin of `not_error_kind` firing on all 90 episodes.

Eleven previously unscored arms from that round are being scored now, ordered so the most informative land
first: AW1's exact spec (`pagination_unbounded/reprompt:page_through_all_results`) as a **cross-model
transfer test**, its matched control, granite's winning variant text on qwen's best-firing signal, and the
two never-tried signals (`auth_error`, `bad_parameter`). AW2's signal cannot be transfer-tested here —
`unknown_api_targeted` does not fire on qwen at all. This section will be updated with those measurements;
until then the honest statement is that qwen's null rests on 4 of 14 scoreable arms, with a screen
predicting the rest.

**The best candidate the screen could offer has now been measured, and it is zero.** Narrowing truncation
to *late* truncation by hand — `code_chars == 0 ∧ step_number > 14` — was the only screened candidate on
qwen whose upside:downside ratio crossed 1.0, at **1.17**. Scored with the `emit_short_runnable_cell` text
(job 963858) it fires **16/90** — comfortably over the firing floor, and far more precisely than
`generation_truncated`'s 49 — for **2 attributable gains against 2 losses, net 0** (raw +1: 9 gains, 8
losses, of which the attribution filter discards 7 and 6 as flips on episodes the controller never fired).

The narrowing worked *as a narrowing* and bought nothing, which is the informative part. It also answers a
question about the search rather than about qwen: this predicate is one the search never proposes, because
Φ expands `step_number` at `post_execution` and not at this gate. Had it paid, that would have been
evidence the scheduler's reach is too short. It did not, so **not proposing it cost nothing measurable** —
and the strongest remaining statement about qwen's null is that its single most favourable candidate,
screened, narrowed and adequately fired, lands dead on the noise floor.

**The `post_execution` boundary produced nothing on any model.** `execution_failed/reprompt` — "read the
error above and issue a corrected call" — measured −3 (minimax), −3 (qwen), −1 (granite). An episode-level
audit found why: the incumbent already recovers unaided after **78%** of first errors, so the instruction
had almost no room. On minimax, 11 synthesized refinements of that boundary all measured −2 to −8.

**A signal that works on one model can be dead on another.** `unknown_api_targeted` is granite's best lever
and is **UNRESOLVABLE on minimax** — 3/90 firings, ceiling 1, below the noise floor. That asymmetry is the
argument for per-model search rather than one shared prompt fix.

---

## Installing and reproducing

An anchor is a `controllers.json` entry; the adapter builds and runs it with no code change.

```bash
source benchmarks/appworld/env.sh        # site paths; see REPRODUCING_EXTERNALLY.md
cd benchmarks/appworld

# 1. Mine your own incumbent's residual and let the search propose. Free, no LLM calls.
$APPWORLD_PY run_round.py propose \
    --evaluation <your incumbent>/evaluations/train.json \
    --out /tmp/round --min-net 5 --min-firings 10

# 2. Screen before spending episodes: an arm firing < min-firings cannot promote.
$APPWORLD_PY runs/support_screen.py \
    --controllers /tmp/round/controllers.json \
    --evaluation <your incumbent>/evaluations/train.json --min-firings 10

# 3. Score. REAL inference, ~90 episodes per arm. Match --num-processes to the
#    concurrency your incumbent was measured at, or it is not a paired comparison.
$APPWORLD_PY run_round.py score \
    --controllers /tmp/round/controllers.json \
    --evaluation <your incumbent>/evaluations/train.json \
    --arm "post_generation_pre_exec/pagination_unbounded/reprompt:page_through_all_results" \
    --num-processes 4 --check-regressions --out /tmp/round/arm_results.json

# 4. Two passes, not one. A single 90-task number cannot be told from the 2-5 task
#    spread between same-config baseline repeats.
PASSES="1 2" runs/score_with_repeat.sh <arm> <controllers> <evaluation> <experiment> <out_dir> 4
```

**You will not reproduce our absolute numbers, and that is expected.** The frozen incumbents are per-task
log data we cannot ship (80–538 MB per run), so you must create your own with your own model. The *method*
reproduces; the numbers are ours. See [`REPRODUCING_EXTERNALLY.md`](REPRODUCING_EXTERNALLY.md).

**Two traps that cost us real results**, both worth checking before you trust any number:

1. **Pin the generation config on both sides**, and verify from `input.max_completion_tokens` in
   `lm_calls.jsonl` what was actually *sent* — not what a config says. An 8192-capped arm against an
   uncapped incumbent invalidated a whole qwen round, because the signal under test fires on the cap being
   hit.
2. **Pair against the draw distribution, not one draw.** minimax has seven same-surface baseline draws
   spanning 53–61 passed and SGC 40.0–53.3. AW1 reads +4 against the highest draw and +5.33 against the
   mean — below the gate on one convention, above it on the other.

### ⚠ How AW1's promotion was selected, stated for review

AW1 clears the gate only under `--net-from-draws`, which judges an arm on the mean net over all
(arm pass × incumbent draw) comparisons instead of against the single draw it was scored on. That rule is
**off by default**, it predates the round it decides (commits `156ec3b`, `ea0e616`), and it is not a uniform
uplift — on the same recomputation AW1's own control moves −8 → −1.83 and three other arms stay rejected.
AW2 needs no such rule; it clears the plain gate.

Both anchors' held-out numbers validate a train-selected arm — search on `train`, take the best improving
arm, score it once on `sh_heldout`. The difference is only which rule promoted it: **AW2 clears the
default gate, AW1 needs `--net-from-draws`**, so quote that rule whenever you quote AW1. Neither regressed
on held-out (AW1 +5.0 → +5.4, AW2 +3.3 → +3.3), which is the check worth having, since a train figure is
always a selected maximum.
