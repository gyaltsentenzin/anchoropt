# The meta-model: which model reads failures and proposes

The **meta-model** does attribution and proposal. The **model under test** produces the trajectories.
They are independent choices, and conflating them is easy because both can be the same family.

```bash
# default -- exactly the historical behaviour, no env needed
export ANCHOROPT_META_PROVIDER=claude
export ANCHOROPT_META_MODEL=claude-opus-5

# any OpenAI-compatible endpoint: Granite on vLLM, a hosted gateway, a local server
export ANCHOROPT_META_PROVIDER=openai_compat
export ANCHOROPT_META_MODEL=granite-4.1-8b
export ANCHOROPT_META_BASE_URL=http://localhost:8000/v1
export ANCHOROPT_META_API_KEY=EMPTY        # local servers often want a non-empty value
```

`ANCHOROPT_CLAUDE_MODEL` is still honoured, so existing runners keep working unchanged.

## What the switch does and does not touch

| unchanged | changed |
|---|---|
| role system prompts | the HTTP transport |
| required output schema and the refusal to salvage a truncated diagnosis | which endpoint and model name |
| JSON extraction, batch size, `MAX_TOKENS` | — |
| search, candidate selection, acceptance logic | — |

That separation is the point: **if the seam also changed the prompt, a difference in result would not
be attributable to the model.** `tests/test_meta_model.py` pins it — including that the prompt section
of `client.py` never mentions a provider.

## No silent failover

There are no retries and no "if the primary fails use the secondary". A meta-model that fails over
produces records attributed to a model that did not write them. Errors are returned as errors and
recorded on the `CallRecord`.

## Every record says which model

`CallRecord` carries `model`, `meta_provider` and `meta_base_url`, and `scripts/result_summary.py`
prints a **meta-model** column. Two providers can serve the same model name, so the provider and URL
are recorded too — a table row that cannot say which model produced its diagnosis cannot be compared
against a row produced with a different one.

## Running the ablation (not yet run)

Same corpus, same incumbent, same arms; only the meta-model differs. Everything downstream of
attribution is deterministic given the diagnoses, so the comparison isolates the meta-model's
contribution to *finding* the intervention — not to executing it.

```bash
ANCHOROPT_META_PROVIDER=claude        ... python scripts/self_evolve_cycle2.py ... --out rounds/ABL_claude
ANCHOROPT_META_PROVIDER=openai_compat ... python scripts/self_evolve_cycle2.py ... --out rounds/ABL_granite
python scripts/result_summary.py --control <ctl> --arm <claude_arm> <granite_arm>
```
