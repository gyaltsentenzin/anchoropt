# Reproducing this outside our cluster — what you need, and what you cannot get from this repo

Written for someone reviewing or reproducing AnchorOpt who has no access to our machines.

This file exists because the honest answer to "can I reproduce the AppWorld numbers?" is **partly**, and
the parts divide cleanly. Everything in §1 runs anywhere. Everything in §2 needs a model endpoint you
supply. §3 is the one thing this repo cannot give you, and it is the real barrier — so it is stated first
rather than discovered on attempt three.

---

## The barrier, up front

**The frozen incumbents are data, and they are not in this repo.**

Every AnchorOpt round is measured against a *frozen incumbent*: one recorded run of the stock
`simplified_react_code_agent` on a split, including **per-task step logs** (`environment_io.md`,
`lm_calls.jsonl`). `mine_residual` reads those logs to build the residual and evaluate signals at
incision points. Outcome files are not enough — a pass/fail table cannot be mined.

Those logs cannot be shipped: one 90-task run is **80–538 MB** depending on the model, and a single model
× split × repeat set runs to gigabytes. They are also not regenerable from this repo alone, because they
are the product of a specific model serving a specific config at a specific time.

So to reproduce the *search*, you must first **create your own incumbent**: run stock AppWorld
`simplified_react_code_agent` on `train` (and `sh_heldout`, if you want held-out numbers) with your own
model, keeping the per-task logs. Then point the adapter at it. Your absolute numbers will differ from
ours — different model, different serving stack — but the *method* reproduces: mine, propose, score,
promote or reject.

What you can check without any of that is in §1, and it is not trivial: the search logic, the decision
rules, the screens and every offline analysis in `docs/APPWORLD_RESULTS.md` that is not an arm score.

---

## §1 Runs anywhere, no model, no GPU, seconds

```bash
python -m pytest tests/test_toy_host_e2e.py tests/test_executor_behavioral_contract.py -q
python scripts/walk_framework.py        # the framework on toy data, computing real numbers
python scripts/verify_progression.py    # recomputes published figures; non-zero exit on drift
```

These exercise `anchoropt/learning/` — `optimize_residual`, the signal grammar, the policy class, the
termination rules — which is where the method lives. No AppWorld, no endpoint.

## §2 Needs a model endpoint you supply; no cluster needed

The adapter talks to an **OpenAI-compatible** `base_url`. Ours happens to be an internal gateway behind
an auth proxy on a fixed port per model; nothing depends on that. Any of vLLM, Ollama, TGI or a hosted
OpenAI-compatible API works. Set it in the AppWorld model config
(`experiments/configs/_generator/models/*.py`), not in an environment variable — the adapter renders the
agent config through AppWorld's own generator so a controller-modified run and the incumbent are
byte-identical apart from the controller.

**Pin the generation config on both sides.** We lost a whole result to this: qwen arms ran at
`max_completion_tokens: 8192` against an uncapped incumbent, and the signal under test fires on the cap
being hit — the treatment manufactured its own trigger. Verify what was *sent*, from
`input.max_completion_tokens` in `lm_calls.jsonl`, not what a config says. See the retraction in
`APPWORLD_RESULTS.md`.

**Match concurrency to the incumbent's.** `score --num-processes N` must equal the N the incumbent was
run at. Against a server you are the sole tenant of, a different N means different batch shapes, which is
a second uncontrolled difference on top of the controller.

## §3 Our site, and what to substitute

| we use | why | substitute |
|---|---|---|
| LSF (`bsub`, `bjobs`) | batch scheduling | anything that runs a command with GPUs; only cluster job-submission wrappers care, and those are not shipped |
| GPFS (`mmlsquota`) | a quota guard, after two runs died mid-episode on a full filesystem | drop it, or swap in `df` |
| a sibling repo's venv as `APPWORLD_PY` | historical: the two projects grew together | any interpreter that imports `appworld` and `anchoropt` |
| a per-model auth proxy on a fixed port | one proxy per model, in front of a gateway needing a non-standard header | any OpenAI-compatible endpoint's URL |

Configure all of it in one place:

```bash
source benchmarks/appworld/env.sh   # then: appworld_env_check
```

Every variable falls back to our values, so sourcing it is a no-op for us and a single edit for you.
`appworld_env_check` fails loudly on a missing interpreter, root or incumbent rather than halfway through
a scoring run.

---

## What reproduces, and at what cost

| claim | reproducible without our data? | cost |
|---|---|---|
| search logic, decision rules, screens | **yes** | seconds, no GPU |
| the support screen and headroom screen on your own incumbent | yes | minutes, no GPU |
| an arm's *absolute* score from an episode tree | yes, on your own tree | minutes, no GPU |
| **our specific arm nets and promotions** | **no** — needs our incumbents | — |
| the method end to end on your model | yes | 1 baseline run per split, then ~90 episodes per arm |

Budget, from our measurements: a 90-task pass is ~30 min at 4-way concurrency on a shared endpoint and
~1h15m–2h on a slower model, and evaluation wall-time is dominated by filesystem contention rather than
compute — we measured `evaluate_tasks` at 0.6 s/task on a quiet filesystem and 3.4 min/task against 11
concurrent jobs. Serialise runs; do not parallelise the evaluator.
