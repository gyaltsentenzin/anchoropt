# TerminalBench 2 / DeepAgents adapter

The second runtime adapter, and the one `anchoropt/attribution/__init__.py` was waiting for when
it declined to pin an ABC "to let the second benchmark inform that shape".

## What lives here, and why it is not in the core

| file | contents |
|---|---|
| `tb2_adapter.py` | event normalization, `U_H(ℓ)`, signal dispatch, semantic-action application, `register()` |
| `tb2_signals.py` | Φ_TB2 — this benchmark's signal evaluators |

Everything LangChain-specific is confined here: the `after_model` hook, `AIMessage`, `tool_calls`,
and `jump_to="model"`. That last one is *this runtime's implementation* of the generic semantic
action **REPROMPT** (request another model decision), not a concept of its own.
`tests/test_tb2_adapter.py` greps `anchoropt/` for that vocabulary and fails if any of it leaks.

## The boundary mapping — no fourth boundary

LangChain's `after_model` fires between the model node and the routing decision
(`graph.add_edge("model", "<mw>.after_model")`, langchain 1.3.0 `agents/factory.py:1623`), so the
generated message exists and nothing has run or been returned. That is the existing
`POST_GENERATION_PRE_EXEC` incision point, which covers both sub-cases:

```
POST_GENERATION_PRE_EXEC
    ├── proposed tool action        tool_calls present
    └── proposed terminal response  len(tool_calls) == 0   ← LangChain's own loop-exit test
```

`terminal_response_proposed` is extracted **here**, from the normalized event. No PRE_GENERATION
signal could supply it — before generation the output does not exist — which is why the earlier
prototype's `no_tool_call_yet` was inverted (`tool_calls_so_far == 0` is true only at episode start).

## `U_H(ℓ)` — declared from measurement, not assumption

`HOST` declares only cells driven with a real anchor and a real payload. Observation-rewriting
(`REROUTE`-as-`transform`) at `POST_EXECUTION` is **withheld**: the prototype implementation read a
tool-result field that does not exist at earlier boundaries and reported `executed=True` while
appending its payload to the empty string. See `docs/CONSUMER_BOUNDARY_RULE.md`. It is re-enabled
by declaring it here once boundary-specific semantics exist — no core change required.

## Registration

Importing this module does **not** register it, unlike the BFCL adapter. There is one global
adapter slot, and taking it on import would silently repoint BFCL's core call sites at the wrong
vocabulary. Call `register()` explicitly from a TB2 entry point.

## What is deliberately absent

The category-B signals derived from the H1 residual — `required_artifact_absent`,
`source_edited_without_rebuild` — are **not** pre-registered. They must arrive through
`anchoropt/learning/expand_attribution.py`, which scores coverage × precision and rejects a
condition that fires as often on successes as on failures. Hand-registering them would bypass the
discipline that caught `single_read_then_answer` at precision 0.51.
