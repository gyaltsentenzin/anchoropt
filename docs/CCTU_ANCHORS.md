# CCTU / qwen — the anchors

What was learned, what is deployed, and how to run it. Numbers are in
`CCTU_RESULTS_QWEN.md` (not in this branch, see [`docs/RESULTS_POLICY.md`](../docs/RESULTS_POLICY.md)); external
reproduction is [`REPRODUCE.md`](CCTU_REPRODUCE.md).

## One anchor is deployed

Two arms were accepted over five rounds, and **the second replaced the first.** They cannot coexist: the
second anchor's `01b` branch is the first anchor's instruction restricted to `call_times >= 1`, so
installing both fires two controllers on the same turn — which round 5's own criteria forbid.

| | accepted | spec | status |
|---|---|---|---|
| **A1** | round 1 | `rounds/CCTU_QWEN1/controller.json` | **superseded by A2** |
| **A2** | round 5 | [`../../rounds/CCTU_QWEN5/controller.json`](../rounds/CCTU_QWEN5/controller.json) | **deployed** |

## A2 — the deployed anchor

A single policy with a state-conditioned message. Two controllers, both `reprompt` at the **commitment
gate** (`post_generation_pre_exec` — after the model has drafted a turn, before anything executes), both
gating on the same condition and splitting on whether the agent has called a tool yet.

| | branch `01a` | branch `01b` |
|---|---|---|
| fires when | `response_length_over_cap` **and** `call_times < 1` | `response_length_over_cap` **and** `call_times >= 1` |
| control episodes | 66 | 18 |
| share of target mass | **92%** | 18% |
| instruction | "Your draft is already longer than this task allows, and you have not called any tool yet. Stop writing the answer directly. Decide which tool you need and call it now; write the final answer only once you have its result." | "You have limited remaining interaction rounds and tool calls. Before acting again, decide which calls are still necessary and make them together rather than one at a time." |

`response_length_over_cap` is a declared boolean field: the drafted final answer exceeds the task's
published `max_responseLength`. `call_times` is the number of tool calls made so far in the episode. The
two predicates are mutually exclusive by construction, so at most one fires on any turn.

**Why the split.** Round 1's single instruction reached the whole condition but treated two different
situations the same way. 92% of the target mass is turns where the agent has **not called a tool yet** and
is drafting a long answer from nothing — those answers are typically unfinished rather than verbose, and
telling the model to shorten is the wrong instruction. Branch `01a` tells it to go get data instead.
Branch `01b` keeps round 1's text for the case it actually describes.

### `one_shot=True` is part of the policy

Each controller fires **at most once per episode**. This is not a default — it is the accepted
configuration, and the same specs at `one_shot=False` are a **rejected** arm:

| firing | `max_length` train | redundant identical calls (rep 0) | verdict |
|---|---:|---:|---|
| **once per episode** | −188 | 85 → 108 (**+23**) | **accepted** |
| every over-cap turn | **−326** | 85 → 494 (**+409**) | rejected, tolerance was +200 |

Per-turn firing removes far more — the largest reduction measured anywhere in this work — and buys it by
driving the agent into repeated tool calling (`tool_results` +1000 against +228). The pre-registered harm
term on redundant calls rejected it. `controller.json` records the requirement under a `requires` key,
which `ControllerSpec.load` ignores because it keys off `"controllers"`.

`one_shot` is the *only* lever on intervention frequency: θ's `retry_budget` is declared but not enforced
per-episode, and `redecide_budget` bounds redecisions per *turn*. So the frequency curve has exactly two
points and both are measured.

## How to run it

The anchor is a spec file handed to `--controllers`. Nothing is patched into the harness.

```bash
cd benchmarks/cctu

# the endpoint must return STRUCTURED tool_calls, not text — verify first
curl -s http://<NODE>:<PORT>/v1/models        # expect Qwen/Qwen3.6-35B-A3B

python run_vllm.py --evaluate \
  --model qwen-3-6-35b --split train --repeat 2 \
  --seed 42 --temperature 0.0 --top-p 1.0 --max-workers 1 \
  --data-dir data \
  --output-dir results \
  --controllers ../../rounds/CCTU_QWEN5/controller.json \
  --anchor-trace \
  --vllm-url http://<NODE>:<PORT>/v1
```

**Do not pass `--no-one-shot`.** Omitting it gives `one_shot=True`, which is the anchor. Passing it
reproduces the rejected configuration.

Then assert the runtime actually matched the policy before reading any number:

```bash
python -c "
import json
m=json.load(open('results/qwen/train_cctu_q5_01_split_gate_once/run_manifest.json'))
c=m['controllers']
assert c['one_shot'] is True,  'one_shot must be True — this is part of the policy'
assert m['max_workers'] == 1,  'the variance floor is only measured at max_workers=1'
assert m.get('endpoints'),     'the serving endpoint must be on the record'
print('OK', c['installed'])"
```

`--max-workers 1` is not conservatism. At 8 the two replicates of a single job stop agreeing and every
tolerance the anchor was judged against becomes unusable.

## Deploying it elsewhere

The spec is portable; the *measurement* is not. To use this anchor on another model or corpus:

1. **Re-measure the control.** Every tolerance is relative to a measured baseline — the benefit bar, the
   accuracy rule, the redundant-call ceiling. None transfers.
2. **Check the condition exists.** `response_length_over_cap` needs a published response-length limit in
   the task description and a declared field that evaluates it. Without the limit in the prompt, the
   instruction is asking for something the model cannot know, which is a capability asymmetry.
3. **Re-screen the split.** The 92%/18% mass split is a property of this corpus. If nearly all mass is on
   the committed side, branch `01a` is doing no work and the split is not worth its complexity.
4. **Expect the accuracy cost to reappear.** Four independent instructions on this condition each lost
   exactly 2 answers out of 78 firing episodes, with the identity of the lost episodes moving between
   arms. That looks like a property of re-emission at the commitment gate rather than of any wording.

## A1 — the superseded anchor, kept for the record

`reprompt` at `post_generation_pre_exec` on bare `response_length_over_cap`, with A2's `01b` text applied to
the whole condition. Accepted in round 1 at −118 `max_length` on train and −56 on test, with `acc` at
**4 gains / 2 losses** on train and 0/0 on test. Superseded because A2 removes 70 more targeted violations
on train, 142 more violations overall on test, and causes a tenth of the redundant calling. Its spec is left
at `rounds/CCTU_QWEN1/controller.json` so the history is auditable, and it should not be loaded alongside A2.
