# TAU2_R1 — the first AnchorOpt round on tau-bench

> **Superseded — kept for history, not a result.** This was the smoke round that proved the chain runs:
> claude-haiku-4-5 as both agent and user, 20 airline tasks taken as a *string-sorted prefix* (which mixes
> tau-bench's train and test splits), `max_steps=30`, and a single incumbent run. None of those survive in
> the current setup. The round-1 report that supersedes this is withheld in this branch while the tau2
> campaign is open — see [`../../../../docs/RESULTS_POLICY.md`](../../../../docs/RESULTS_POLICY.md).

**Outcome: `BUDGET_EXHAUSTED`. Nothing was promoted, and the round is open rather than settled.**

Deliberately **not** in [`../../../../rounds/`](../../../../rounds/): that directory is the BFCL measured
line, and this round is a single 20-task subset with no dev or held-out validation. It demonstrates that
the chain runs on real benchmark data; it licenses no effect claim.

## What ran

| | |
|---|---|
| domain / subset | `airline`, 20 tasks (first 20 by sorted id), `max_steps=30` |
| agent & user model | `openai/aws/claude-haiku-4-5`, temperature 0 |
| frozen incumbent P0 | **8/20 solved**, mean reward 0.4000 |
| terminations | 17 `USER_STOP`, 3 `MAX_STEPS`, **0 harness errors** |

## The chain, stage by stage

| stage | observed |
|---|---|
| 1. residual | 12 failing cases → 12 diagnoses → **2 ranked problems** |
| 2. target | R1, support **9**, coverage 0.75: *an answer committed to the user before the task's required change was made* |
| 3. trajectories | **262 events** over the 9 cases |
| 4. localization | **all three boundaries** present — 97 pre-generation, 97 gate, 68 post-execution states |
| 5. grounding | **36 arms** over 5 cells: gate reroute (20), gate reprompt (8), gate suppress (4), post-execution reprompt (2), pre-generation reprompt (2) |
| 6. manifest | `state=REALIZABLE_UNMEASURED`, **no `selected_arm`**, no `next_arm` |
| 7. paired evaluation | **3 of 36 measured** against the same frozen P0 |
| 8. selection | `BUDGET_EXHAUSTED`; best measured **net +0**; `promoted=None`, `controller_spec={}` |

`moves_earlier = 0`: the latest boundary yielded candidates immediately, so the search never had to move
earlier. Φ was **not** expanded, and that is the schedule working — expansion is driven by *measured*
exhaustion, and this pass had no evaluator. See `README.md` on cycle 2.

## The three measurements

| arm | net | gains/losses | Δpp | executed |
|---|---|---|---|---|
| `post_execution/tool_call_failed/reprompt:recheck_against_policy` | −1 | +0 / −1 | −5.00 | 2 |
| `post_execution/tool_call_failed/reprompt:require_verification_first` | **+0** | +0 / −0 | +0.00 | 2 |
| `post_generation_pre_exec/proposes_tool_call/reprompt:recheck_against_policy` | −1 | **+2 / −3** | −5.00 | **96** |

Arms were truncated **in manifest order**, never by firing rate — that is the heuristic core-v2 removed.

## Four things worth reading off this round

**The outcome class is the result.** 33 arms were never run, so the round cannot be `NO_BENEFIT`: that
would assert a measurement that never happened. `is_negative_result` is `False` and nothing is
installable. Complete it with `evaluate --resume` rather than classifying it.

**A firing count proves the mechanism ran.** The broad gate arm executed **96** interventions across
**20/20** cases; the narrow post-execution arms executed 2 each. On arm 1, `predicate_matches=3` against
`interventions_executed=2` — the retry budget blocked one match in a live run, and the two are counted
separately so a blocked match never inflates engagement.

**Every arm run replayed cleanly.** All 20 episodes under each installed controller were scored, with no
`harness_error`. Scoring calls `Environment.set_state`, which re-executes the trajectory's mutating calls
under `strict=True` — so a withheld call leaking into the record would have raised. It did not.

**A signal can fire outside the residual it was fitted to.** `tool_call_failed` fires on **0** of the
residual's 262 states (those 9 cases contain zero errored results) yet fired twice during evaluation,
because evaluation runs the whole 20-case corpus. That is a real property of the measurement design, and
the reason the comparison is per case.

### A correction worth recording

The round first emitted **16** arms, with **no REROUTE arm at all** — and that was a defect in this
adapter's own declaration, not a property of the host. The gate's reroute capability declared
`consumes=("destination", "argument_mapping")` while core's REROUTE/substitute contract requires
`retry_semantics`, so all five grounded destinations per signal were refused
`executor_cannot_consume_eta`: an arm carrying eta no executor reads measures a different intervention
than the one proposed. The mechanism now genuinely reads the key and declines a value it cannot execute
(this host only ever *replaces* a call), and the cell declares `fixed={"retry_semantics": "replace"}` so a
conflicting arm is refused at construction rather than running as the control. 16 → **36**.

Same class as the `retry_budget` omission that had silenced the pre-generation and post-execution reprompt
cells. Both were found by *building arms and reading the rejections* — nothing in the contract check or
the behavioural probes catches an executor that under-declares what it reads, because the probe proves
the mechanism works and says nothing about whether core can construct an arm for it.

## Reproducing

```bash
export ANCHOROPT_TAU2_REPO=/path/to/tau2-bench
set -a && . "$ANCHOROPT_TAU2_REPO/.env" && set +a
PY="PYTHONPATH=$PWD $ANCHOROPT_TAU2_REPO/.venv/bin/python"
OUT=benchmarks/tau2/rounds/TAU2_R1

$PY benchmarks/tau2/run_anchoropt_round.py baseline --domain airline --limit 20 \
      --max-steps 30 --concurrency 5 --out $OUT
$PY benchmarks/tau2/run_anchoropt_round.py propose  --out $OUT
$PY benchmarks/tau2/run_anchoropt_round.py evaluate --out $OUT --max-arms 3 --concurrency 5
$PY benchmarks/tau2/run_anchoropt_round.py select   --out $OUT
```

The model is sampled at temperature 0 but the benchmark's variance floor is **not** zero — two identical
airline baselines have differed by 8 of 50 tasks. A net of ±1 here is inside noise.
