# CCTU / qwen — reproducing these results externally

Three levels, cheapest first. Level 1 needs nothing but this repository. Level 2 and 3 need a served model.

| level | what it checks | cost | needs a GPU |
|---|---|---|---|
| **1. Re-score** | every number in `CCTU_RESULTS_QWEN.md` (not in this branch, see [`docs/RESULTS_POLICY.md`](../docs/RESULTS_POLICY.md)) from the shipped transcripts | seconds | no |
| **2. Re-run the control** | that the baseline is reproducible on your endpoint | ~4 h train | yes |
| **3. Re-run the anchor** | the anchor's effect end to end | ~4 h train, ~2 h test | yes |

## What the environment was

| | |
|---|---|
| model | `Qwen/Qwen3.6-35B-A3B`, served by vLLM over the OpenAI-compatible API |
| harness alias | `qwen-3-6-35b` (maps to the served id in `response_generator.SERVED_MODEL_NAMES`) |
| decode | `temperature 0.0`, `seed 42`, `top_p 1.0`, `max_tokens 1024`, `thinking` off |
| replicates | `--repeat 2` |
| concurrency | `--max-workers 1` — **load-bearing, see below** |
| train corpus | `data/input_data_train.jsonl`, 140 tasks, `sha256:6af950d0d7f0af144298a218102e6918d2035ed2bb739c696cb7434f17214440` |
| test corpus | `data/input_data_test.jsonl`, 60 tasks, `sha256:554e09d3352c92a6b75b9f1e03b2f440b14f7eb8d79ab11eb36659b4f47af041` |

**The endpoint must return structured `tool_calls`.** CCTU executes tool calls the harness parses out of
the API response; if vLLM returns them as text the harness does not execute them and `acc` is
structurally 0 for every episode — a whole run unscoreable in a way that looks exactly like a weak model.
Serve with `--enable-auto-tool-choice` and a `--tool-call-parser` appropriate to the model, and verify:

```bash
curl -s http://<NODE>:<PORT>/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model":"Qwen/Qwen3.6-35B-A3B",
 "messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
 "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object",
   "properties":{"city":{"type":"string"}},"required":["city"]}}}],
 "temperature":0.0}' | python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message'].get('tool_calls'))"
```

A non-`null` `tool_calls` array is the requirement. Note that `model_config_granite41.json` names the
**granite** parser — that file is the granite integration's, and its parser choice is wrong for Qwen. The
runs behind these results used an externally launched Qwen server verified by the call above.

**`--max-workers 1` is not conservatism.** At `--max-workers 8` on this setup the two replicates of a
single job stop agreeing (`acc` 23.57 ± 0.00 → 21.07 ± 1.79). That `±` is the spread *across replicates
inside one job*, so nothing about the serving node explains it — concurrency alone does. Every tolerance in
this work assumes a zero replicate floor, so a run at higher concurrency cannot be compared to these
numbers.

## Level 1 — re-score from the shipped transcripts (no GPU)

`response.jsonl` holds every generation. Scoring is deterministic and needs no model.

```bash
cd benchmarks/cctu

# the control, and the deployed anchor, on both splits
for d in results/qwen/train_baseline results/qwen/train_cctu_q5_01_split_gate_once; do
  python evaluation.py --input-dir data \
    --input-response-data "$d/response.jsonl" --split train --repeat 2 \
    --detail --overload --output-file "$d/scores.json"
done
for d in results/qwen/test_baseline results/qwen/test_cctu_q5_01_split_gate_once; do
  python evaluation.py --input-dir data \
    --input-response-data "$d/response.jsonl" --split test --repeat 2 \
    --detail --overload --output-file "$d/scores.json"
done
```

**Pass the right `--split`.** `evaluation.py` fails closed if the row count does not match the split, and
`--overload` deletes `detail.jsonl` before that check — so the wrong split costs you the scored artifact
(recoverable, since `response.jsonl` survives, but only by re-scoring correctly).

Then reproduce the tables:

```bash
python - <<'PY'
import json, collections
def counts(run):
    c = collections.Counter()
    for line in open(run + '/detail.jsonl'):
        for k, v in (json.loads(line).get('violations') or {}).items():
            c[k] += int(v)
    return c
for split in ('train', 'test'):
    b = counts(f'results/qwen/{split}_baseline')
    a = counts(f'results/qwen/{split}_cctu_q5_01_split_gate_once')
    tb, ta = sum(b.values()), sum(a.values())
    print(f'{split}: all classes {tb} -> {ta} = {ta-tb:+d} ({100*(ta-tb)/tb:+.1f}%)')
    print(f'       max_length {b["max_length"]} -> {a["max_length"]} = '
          f'{a["max_length"]-b["max_length"]:+d}')
PY
```

Expected: `train: all classes 3726 -> 3294 = -432 (-11.6%)`, `max_length 786 -> 598 = -188`;
`test: all classes 1866 -> 1702 = -164 (-8.8%)`, `max_length 538 -> 488 = -50`.

**Read counts from `detail.jsonl`'s `violations` field, not from `analyze_residual.py`'s `RESIDUAL`
block.** The two now agree, but only `detail.jsonl` is what the rounds were judged on.

Attribution and the paired rates:

```bash
# every non-firing episode must reproduce the control exactly
python check_pairing.py \
  results/qwen/train_baseline results/qwen/train_cctu_q5_01_split_gate_once

# acc / SR / PSR as per-episode gains and losses, not just the net
python analyze_residual.py --compare \
  results/qwen/train_baseline results/qwen/train_cctu_q5_01_split_gate_once
```

Expected: `COMPARABLE: all 202 non-firing episodes reproduce the control exactly`, and
`acc 23.57 -> 22.86 (-0.71 pp) 0g / 2l`.

## Level 2 — re-run the control

```bash
cd benchmarks/cctu
python run_vllm.py --evaluate \
  --model qwen-3-6-35b --split train --repeat 2 \
  --seed 42 --temperature 0.0 --top-p 1.0 --max-workers 1 \
  --data-dir data \
  --output-dir results_repro \
  --vllm-url http://<NODE>:<PORT>/v1
```

Compare against ours. `response.jsonl` will **not** match byte-for-byte even on an identical model — vLLM
generates a random `chatcmpl-tool-<hex>` id per call. Normalise those, or compare the scored artifacts:

```bash
sha256sum results_repro/qwen/train_baseline/detail.jsonl results/qwen/train_baseline/detail.jsonl
```

Three separate runs of this control on different cluster nodes produced **byte-identical**
`detail.jsonl` and `scores.json` here, so an exact match is the expected outcome on the same model build.
A mismatch most likely means a different quantisation, a different vLLM build, a different tool-call
parser, or concurrency above 1 — check those before concluding the result does not reproduce.

## Level 3 — re-run the anchor

As in [`ANCHORS.md`](CCTU_ANCHORS.md), with `--output-dir results_repro`, then repeat the Level 1 checks against
your own control. Two assertions before reading any number:

```bash
python -c "
import json
m=json.load(open('results_repro/qwen/train_cctu_q5_01_split_gate_once/run_manifest.json'))
assert m['controllers']['one_shot'] is True, 'one_shot=True is part of the policy'
assert m['max_workers'] == 1
print('OK')"
```

## Known caveats a reproducer will hit

* **The control's `endpoint` is unrecorded.** Our train and test baselines predate the field, so
  `analyze_residual --compare` reports `endpoint NOT RECORDED for one side` on every pairing.
  `results_pinned/qwen/train_baseline` is the same control with `host21:8000` recorded and is
  byte-identical to it, which is what licenses reading the pairings.
* **Arms were split across serving nodes.** Arms take ~4 h and cluster allocations are shorter, so several
  runs were finished on a second or third node. `run_manifest.json`'s `endpoints` lists every one, and
  `check_pairing.py` is what establishes comparability — node changes were measured not to move
  generations on this corpus.
* **`anchor_telemetry.json` is wrong on any resumed run.** It is per-invocation and overwritten. Read
  engagement from `anchor_trace.jsonl`.
* **Counts before 2026-09-23 are inflated.** `evaluation.violation_counts` was counting every message
  including the model's own, and the model quotes validator error text while reasoning — 840 counted
  against 786 real on the train control. Any `detail.jsonl` not re-scored since is not comparable. See
  [`VENDORED.md`](../benchmarks/cctu/VENDORED.md).
* **`repeated_identical_calls` is replicate-0** and comes from `analyze_residual.py`'s loop-shape line, not
  from `detail.jsonl` — it is not a violation class.
