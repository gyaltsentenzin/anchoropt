# Vendored code in this directory

Most of this directory is a **copy** of the CCTU benchmark, not original work. It is vendored so the
AnchorOpt integration can run against the real corpus and the real constraint validator instead of
pointing at a repo you may not have.

| origin | licence |
|---|---|
| [CCTU: A Benchmark for Tool Use under Complex Constraints](https://arxiv.org/abs/2603.15309) — Junjie Ye et al., Fudan University | Apache-2.0 ([`LICENSE`](LICENSE)) |
| [corpus on HuggingFace](https://huggingface.co/datasets/Junjie-Ye/CCTU) | as upstream |

## A hardcoded API key — removed, and the key itself has been rotated

`utils/hosted_client.py` carried a literal API key as a module constant, read at four call sites,
alongside the provider's internal base URL. **Both are gone.** The client now reads
`HOSTED_API_KEY` and `HOSTED_API_BASE_URL` from the environment through a `_require` helper that raises
a message naming the missing variable, rather than defaulting to anything — verified: constructing a
client with neither set raises instead of sending an unauthenticated request.

The provider also wanted a second, provider-specific auth header whose *name* was its product name. It
could not simply be renamed, because the provider matches the exact string, and renaming in place would
have left code that looks right and authenticates as nobody. Set `HOSTED_API_KEY_HEADER` to that header
name and it is sent; leave it unset and only standard bearer auth goes out, which is correct for any
OpenAI-compatible endpoint.

The key never reached a git commit (confirmed by searching the full history, not just the current
tree), and the live key has since been rotated with the provider — the value in the old working tree
is no longer valid.

BFCL hit the milder version of this and the precedent is recorded in
[`../bfcl_v4/VENDORED.md`](../bfcl_v4/VENDORED.md): a *dummy* credential literal for a mock API was
still replaced with a derived value, because "requesting a scanning exemption to publish unused
third-party fixtures would have been the wrong trade". This one was not a dummy.

## No upstream commit is pinned — and that is a real gap

`../bfcl_v4/VENDORED.md` can say exactly which upstream tree its fork diverges from and enumerate the
load-bearing differences. **This directory cannot.** There is no recorded upstream commit, tag or
archive URL, so the classification below is inferred from **evidence in the files themselves** rather
than from a diff:

| evidence | reading |
|---|---|
| `# Copyright 2026 Junjie Ye` + Apache header, no AnchorOpt or IBM references | upstream, probably unmodified |
| the same header **plus** AnchorOpt/vLLM/seed/controller vocabulary | upstream file, **locally modified** |
| no upstream header, or a different copyright | **ours** |

That is weaker than a diff and it should be replaced by one. **Pin the upstream source** — commit hash
or dated archive — the next time this directory is touched. Without it, "we did not change upstream's
scorer" is an assertion, and the whole point of `FIDELITY_AUDIT.md` on the other benchmark is that such
assertions get written down and checked.

## What is upstream, what is modified, what is ours

### Upstream, believed unmodified

| path | notes |
|---|---|
| `data/input_data.jsonl` | 200 episodes, balanced 50 per `data_source` |
| `data/check_code/` | 200 directories, one per `query_id` — the executable per-episode validators |
| `utils/constraint_checker/` | 12 files. **The constraint validator, which the integration must not touch.** Its verdicts are what SR and PSR mean |
| `utils/client.py` | the paper's own API clients (GPT / Kimi / Doubao). **Dead code** — nothing imports it; `response_generator.py` uses `hosted_client` and `vllm_client` instead. Its `BASE_URL` values are already sanitised to `https://example.com` upstream |
| `evaluation.sh` | upstream's driver. Superseded here by `run_hosted.py` / `run_vllm.py`, and it does not forward `--split`, `--trial` or any anchor flag |
| `__init__.py` | licence header only, no code |
| `requirements.txt` | `func_timeout`, `numpy`, `openai`, `tqdm` |

`utils/utils.py` sits on the boundary: it carries the upstream header and names nothing AnchorOpt-
specific, but `build_tool_call_id2name` has a text-format (`<tool_call>…</tool_call>`) branch that
exists only to serve the locally added vLLM/the hosted API path. Treat it as **upstream, possibly modified**
until a diff says otherwise.

### Upstream, locally modified

| path | what was added |
|---|---|
| `response_generator.py` | `--seed` / `--no-seed` (an unseeded run cannot support a variance-floor claim), vLLM serving + `SERVED_MODEL_NAMES`, `--max-tokens`, `extract_tool_calls_from_text`, `--split`, `--analyze-tool-calls` and the analysis sidecar. **Step 5** split `get_feedback` into `decode_tool_calls` / `validate_turn` / `execute_turn`, wired the three incision points, made `--controllers` / `--anchor-trace` / `--replay-from` live, added `--allow-text-tool-calls`, and fixed four defects (see below) |
| `evaluation.py` | `acc` as a first-class metric, `terminal_state`, `judge()` returning a dict, `METRICS`, the `_meta` block with the honest `spread_is` note, `--detail` and sorted `detail.jsonl`, and the fail-closed length check that names the missing episodes |
| `evaluation.py` | **`violation_counts` now skips `role: "assistant"` messages.** It counted every message regardless of role, and the model quotes the validator's error text back while reasoning about how to fix it — so its own echoes were scored as violations. Upstream delivers violations only as `role: "tool"` (non-final turns) and one merged `role: "user"` message (final turns), `core.py:137-149`; `assistant` is not a validator channel. Measured on `results/qwen/train_baseline`: episode `124_0` holds 29 `MAX LENGTH NOT FOLLOWED` strings of which **13 are the model's own**, and the control's `max_length` was 840 counted against 786 real. The inflation is **not uniform across arms** (0 on one, 44 on another, against a 60-event bar), so it moved per-arm deltas and reordered them. See below |

`evaluation.py`'s additions are the ones AnchorOpt depends on, and each has a reason recorded in its own
docstring: `acc` exists because PSR-failure is a circular ranking metric, `terminal_state` because an
episode that exhausts its round budget is short-circuited to `(1, 1)` and vanishes from an SR/PSR
aggregate, and the `_meta.spread_is` note because `±` over a single replicate is identically `0.00` and
means nothing. **`solve_rate_is_one`, `compute_if_flags` and the SR/PSR definitions are upstream's and
stay upstream's** — a scored case must keep meaning what CCTU says it means.

#### The `violation_counts` role filter, and what it invalidates

This one is not an addition, it is a **correction to a count every round has been judged on**, so the
consequence is stated plainly rather than left for a reader to work out.

`analyze_residual`'s mined counts were always BELOW `detail.jsonl`'s, and that gap was first read as the
miner undercounting — it walks the transcript through `cctu_signals.violation_classes_in`, which reads
only the two real validator channels. The miner was right. The scorer was counting the model's echoes.

**Every `detail.jsonl` on disk was written by the old behaviour.** Re-score with
`evaluation.py --detail --overload`, which needs no model and no GPU. Until then:

* frozen per-class figures move, including `rounds/CCTU_CYCLE0`'s floors and every round scored against
  them
* a delta computed between an OLD count and a NEW one is meaningless — re-score a control and its arms
  together or not at all
* `SR`, `PSR`, `acc` and `termination` are untouched: they come from `compute_if_flags` and
  `terminal_state`, not from this function

Measured effect on `rounds/CCTU_QWEN1`, whose verdict does **not** change — every arm still clears the
60-event benefit bar on real counts, and the accepted arm is still 258:

| run | counted | real | `max_length` delta, counted -> real |
|---|---:|---:|---|
| train control | 840 | 786 | — |
| 258 `state_remaining_budget` | 702 | 668 | −138 -> **−118** |
| 259 `satisfy_pending_minimums` | 714 | 704 | −126 -> −82 |
| 260 `recheck_constraints` | 702 | 684 | −138 -> −102 |
| 261 `shorten_to_length_limit` | 684 | 630 | −156 -> −156 |

The ranking changes: 258 and 260 were tied on the counted metric and are 16 events apart on the real one.
On `test` (control 568 counted / 538 real) 258 goes −84 -> **−56** and 260 −62 -> **−32**, which is below
a proportional bar — so the correction separates the two arms that had replicated and leaves one.

### Ours

| path | notes |
|---|---|
| `run_hosted.py` | the hosted API runner. Its module docstring is a copy of `run_vllm.py`'s and says "vLLM-served Granite" — **wrong, and worth fixing** |
| `run_vllm.py` | vLLM runner |
| `utils/hosted_client.py` | IBM the hosted API clients. Carries the upstream header although the file is ours — and carries the key in item 1 above |
| `utils/vllm_client.py` | `# Copyright 2026 <author>`; "matching anchoropt's pipeline" |
| `split_train_test.py` | the 140 / 60 split. **Positional, not randomized** — first 35 per category to train, next 15 to test |
| `analyze_response.py` | trajectory reader for the residual analysis |
| `ANCHOROPT.md`, `VENDORED.md` | this integration's documentation |
| `results/all_tool_code.txt` | generated (7,050 lines of the corpus's tool source, extracted for reading) |

## Not linted, not reformatted

`utils/constraint_checker/` and `data/` are upstream's. Linting or reformatting them would destroy the
diff against the source and edit code whose verdicts define the metric. **Fix bugs upstream and
re-vendor** rather than editing here.

**Four upstream defects were fixed in `response_generator.py`** — in scope because that file is already
locally modified. The validator and the scorer were not touched. Full detail in
[`ANCHOROPT.md`](ANCHOROPT.md):

| defect | was |
|---|---|
| the retry loop swallowed every exception | `except Exception as e: err = str(e)` — no `break`, no `raise`, `err` never read |
| `times` was never incremented in that loop | a persistently failing episode spun **forever** instead of being dropped |
| `tool_calls_from_text` was set unconditionally under `--use-vllm` | a plain final answer was flagged as text-format and **its feedback discarded**, so terminal violations were computed, never shown, and the episode looped to budget exhaustion |
| a text-extracted call is never executed, making the episode unscoreable | now raises with the endpoint fix named, rather than producing `acc = 0` for a harness reason |

The third is the one to know about: it produces plausible-looking output, it is `--use-vllm`-specific,
and the symptom is indistinguishable from a weak model.

**Four more, found while pinning the run layout and the decode settings:**

| where | defect |
|---|---|
| `run_vllm.py` | despite its name it passed neither `--use-vllm` nor `--model`, so it ran the **the hosted API** client against `response_generator`'s default model (`qwen3-6-35b`) |
| `utils/vllm_client.py` | `"seed": kwargs.get("seed", 42)` substituted a seed whenever the harness sent none, so `--no-seed` was a statement the run did not honour and an "unseeded" run was silently seeded at 42 |
| `run_hosted.py` | `--temperature` was declared `type=int` with a default of `0.01`, so every explicit value was coerced to an integer and `--temperature 0.7` ran as `0` |
| `requirements.txt` | `requests` was missing, and `utils/vllm_client.py` imports it — so `--use-vllm` failed at import |

`run_hosted.py`'s module docstring was also a verbatim copy of `run_vllm.py`'s, describing a vLLM server
and a `--vllm-url` flag it does not have. Fixed, and noted in the docstring itself, because a wrong
docstring costs a reader more than a missing one.

## Dependencies

Four, all light: `func_timeout`, `numpy`, `openai`, `tqdm`. No GPU is needed in-process — the agent
talks to a served model over HTTP, so `--dry-run` and the plumbing checks work on a laptop. Note that
`func_timeout` is load-bearing in a way that affects measurement: `call_function` is wrapped in
`@func_set_timeout(10)`, so tool execution is wall-clock bounded and the variance floor must be
measured rather than assumed to be zero.


## Patches applied to vendored upstream code

**None.** Every file that came from upstream is unmodified except for the AnchorOpt hook sites already
described above, and that is a deliberate boundary rather than an accident.

### The one crash we chose NOT to patch here

`utils/constraint_checker/handlers/tool.py` (`MaxCallsPerToolHandler.check`) indexes
`callTimesPerTool` and `max_callTimesPerTool` -- plain dicts built by `core.py:64-65` from the
episode's own tool list -- with a name taken straight from the model's proposal. A hallucinated or
truncated name raises `KeyError`.

`tests/test_cctu_adapter.py::test_a_validator_crash_is_contained_and_recorded_not_raised` pins that
raise as a **characterisation of upstream**, and states the choice: "this is LATENT, and it is pinned
here rather than fixed in the validator, which stays upstream's."

**The "latent" premise has since been falsified, and the response was to contain rather than patch.**
It was measured as never firing across 140 granite episodes, because a control's tool list travels with
the request. It fired twice on qwen arms: 2 episodes lost to `KeyError: 'earthquake'` (for
`earthquake_data_analyzer`) and 36 to `KeyError: 'histor'`. A reprompt at the commitment gate induces
the truncation, so no control can show it. Unwrapped, the exception escapes `validate_turn`, is retried
15 times to the same deterministic failure, and **drops the episode** -- leaving a short
`response.jsonl` that `evaluation.py` correctly refuses to score, so one bad tool name costs a whole
arm.

The fix is in `response_generator.py`'s phase B, which is already our code: the `KeyError` is caught,
recorded in the run's `anchoropt_errors`, and the turn proceeds with no constraint verdict. Execution
then reports `an error occured when call <name>`, so the model is still told and the episode stays in
the denominator. The vendored validator is untouched, the characterisation test still passes, and no
control figure moves.
