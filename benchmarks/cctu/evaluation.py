# Copyright 2026 Junjie Ye
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
import json
from tqdm import tqdm
from numpy import std
import re
from utils.utils import build_tool_call_id2name, answer_verify
from copy import deepcopy


_ERR_RE = re.compile(
    r"INSTRUCTION FOLLOWING ERROR:\s*([A-Z0-9 _]+?)\s*NOT FOLLOWED!",
    re.IGNORECASE,
)


def solve_rate_is_one(messages, unsolved_set):
    remain = {k: list(v) for k, v in (unsolved_set).items()}
    unsolved_cnt = sum(len(v) for v in remain.values())

    id2name = build_tool_call_id2name(messages)
    solved_cnt = 0

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id")
        if not call_id:
            continue
        tool_name = id2name.get(str(call_id))
        if not tool_name:
            continue
        answers = remain.get(tool_name, [])
        if not answers:
            continue

        content = msg.get("content") or ""
        for ans in list(answers):
            if answer_verify(content, ans):
                answers.remove(ans)
                solved_cnt += 1
                break

    return solved_cnt == unsolved_cnt


def has_if_error_in_text(text):
    return bool(text) and (_ERR_RE.search(text) is not None)


def compute_if_flags(messages):
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("messages must be a non-empty list")

    if messages[-1].get("role") == "tool":
        return 1, 1

    has_if_error = 0
    for msg in messages:
        if has_if_error_in_text(msg.get("content") or ""):
            has_if_error = 1
            break

    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx == -1:
        return 0, 0

    has_last_if_error = 0
    for msg in messages[last_assistant_idx + 1:]:
        if has_if_error_in_text(msg.get("content") or ""):
            has_last_if_error = 1
            break

    return has_if_error, has_last_if_error


def terminal_state(messages):
    """How the episode ENDED, independent of whether it scored.

    Reported because it is not derivable from SR/PSR and it dominates this corpus: an episode
    whose last message is a tool result exhausted its round budget mid-loop, and
    `compute_if_flags` short-circuits it to (1, 1) -- so SR and PSR are both forced to 0 before
    any other check runs. Aggregating only SR/PSR hides that entirely.
    """
    if not messages:
        return "empty"
    last = messages[-1]
    role = last.get("role")
    if role == "tool":
        return "mid_tool_call"          # round budget exhausted with tool results still pending
    if role == "assistant":
        return "truncated_with_tool_calls" if (last.get("tool_calls") or []) else "final_answer"
    return f"ended_on_{role}"           # e.g. injected feedback was the last thing appended


def violation_counts(messages):
    """Every violation in the episode, bucketed by class. COUNTS, not a flag.

    WHY A COUNT AND NOT A FLAG. `SR` and `PSR` are booleans over the whole episode, so an
    intervention that takes an episode from 12 violations to 2 scores identically to one that
    changes nothing -- and PSR requires exactly 0, so 12 -> 2 is measurable progress toward the
    benchmark's own headline metric that no existing metric can express. Two null rounds were scored
    without this number being available.

    BUCKETED BY CLASS, because the classes are not interchangeable and an unbundled total would be
    dominated by the two largest. Measured on the granite train control's own replicates, the
    per-class noise ranges from 0.8% of base (`max_calls_per_tool`, 601 events, replicate net -5) to
    31% (`min_length`, 16 events, net +5). A claim about the first is well-founded and the same claim
    about the last is not, so the floors are per class and several classes cannot support one at all.
    See `rounds/CCTU_CYCLE0/CYCLE0_COUNTS.md`, which freezes them from the control alone.

    THIS IS A LEADING INDICATOR OF PSR, NOT A SUBSTITUTE FOR IT. Reporting it as the benchmark's
    result would be replacing a hard metric with an easier one after two nulls; reporting it as
    progress toward PSR is what it is.

    ONLY THE VALIDATOR'S CHANNELS ARE COUNTED, and the role filter is the whole point.

    This counted EVERY message regardless of role, and the model quotes the validator back at itself.
    Measured on `results/qwen/train_baseline`: episode `124_0` carries 29 `MAX LENGTH NOT FOLLOWED`
    strings, of which **13 are in the model's own `assistant` turns** -- it echoes the error text while
    reasoning about how to fix it. Those are not violations; they are the model reading its feedback.
    Across that control the inflation is 54 events on `max_length` (840 counted, 786 real), and it is
    NOT uniform across arms -- 0 on one arm and 44 on another -- so it moved a per-arm benefit delta by
    up to 44 events against a 60-event bar, and it reordered the arms.

    Upstream delivers violations on two channels and neither is `assistant`
    (`utils/constraint_checker/core.py:137-149`): a non-final turn's come back as `role: "tool"`
    messages, a final turn's as one merged `role: "user"` message. `cctu_signals.violation_classes_in`
    already reads exactly those two, which is why `analyze_residual`'s mined counts were BELOW this
    function's and why the difference was misread as the miner undercounting.

    RE-SCORING IS REQUIRED AND IS NOT AUTOMATIC. Every `detail.jsonl` on disk was written by the old
    behaviour, and its counts are inflated by whatever its episodes echoed. Re-score with
    `evaluation.py --detail --overload`, which needs no model. **Frozen per-class figures move**,
    including `rounds/CCTU_CYCLE0`'s floors and every round scored against them, so re-score a control
    and its arms together or not at all -- a delta between an old count and a new one is meaningless.
    """
    out = {}
    for msg in messages or ():
        if (msg or {}).get("role") == "assistant":
            continue
        for match in _ERR_RE.finditer(str((msg or {}).get("content") or "")):
            key = "_".join(match.group(1).strip().lower().split())
            out[key] = out.get(key, 0) + 1
    return out


def work_done(messages):
    """`(assistant_turns, tool_results)` -- the two denominators a violation drop must be judged against.

    A violation count falls for two completely different reasons and the count alone cannot tell them
    apart: the agent made fewer mistakes, or the agent did less. The environment reprompts on every
    violation, so a stuck episode re-emits its class every turn until the budget dies -- which makes
    the raw count largely a function of episode LENGTH.

    So both denominators are recorded and a repair has to survive them: **a violation drop that comes
    with fewer rounds used or fewer tools actually run is a withdrawal, not a repair.**

    `tool_results` excludes the violation messages, which share the `role: "tool"` shape with real
    results (`utils/constraint_checker/core.py` returns a non-final turn's violations keyed by
    `tool_call_id`). Counting those as work done would make a violation drop lower its own
    denominator and mask exactly the confound this exists to catch.
    """
    turns = sum(1 for m in (messages or ()) if (m or {}).get("role") == "assistant")
    results = sum(1 for m in (messages or ())
                  if (m or {}).get("role") == "tool"
                  and not has_if_error_in_text((m or {}).get("content") or ""))
    return turns, results


def judge(messages, input_data):
    """Score one episode.

    Returns a dict rather than a tuple so `acc` and `termination` travel with SR/PSR. `acc` is
    the metric to rank loci against: PSR = acc AND no-error-anywhere, so ranking loci by
    PSR-failure is circular -- every locus scores precision 1.00 by construction.
    """
    unsolved_set = json.loads(input_data.get("unsolved_set"))
    acc = int(solve_rate_is_one(messages, unsolved_set))

    has_if_error, has_last_if_error = compute_if_flags(messages)

    counts = violation_counts(messages)
    turns, results = work_done(messages)
    termination = terminal_state(messages)

    return {
        "acc": bool(acc == 1),
        "SR": bool(acc == 1 and has_last_if_error == 0),
        "PSR": bool(acc == 1 and has_if_error == 0),
        "termination": termination,
        # ---- the count layer, per episode. NOT in METRICS -- see below ---------------------------
        "violations": counts,
        "n_violations": sum(counts.values()),
        # `termination` is categorical, so it cannot carry a paired delta. This is the one contrast
        # that matters and it is the residual cycle 2 aimed at: 195 of 280 control episodes end here,
        # and `compute_if_flags` short-circuits every one of them to SR = PSR = 0 before any other
        # check. `terminal_state`'s own docstring says aggregating only SR/PSR "hides that entirely";
        # this is what stops it being hidden.
        "died_of_exhaustion": termination == "mid_tool_call",
        "assistant_turns": turns,
        "tool_results": results,
    }


# THE RATE METRICS -- booleans, averaged per bucket, floors measured as episode FLIPS.
#
# `died_of_exhaustion` is deliberately NOT added here. Adding it would change `scores.json`'s shape
# and the variance-floor dict that `rounds/CCTU_CYCLE0/record_granite.json` froze, so a re-score of
# any existing run would read as drift against a record that is correct. It is scored as its own
# group instead, which costs one extra tuple and keeps every frozen figure valid.
METRICS = ("acc", "SR", "PSR")

# Booleans scored SEPARATELY from METRICS: same flip-based floor, own reporting block.
#
# A WARNING THAT TRAVELS WITH THE FIELD: `died_of_exhaustion` is only a benefit metric for an anchor
# whose mechanism does not act on termination. An anchor that forces the model to answer converts
# `mid_tool_call -> final_answer` BY CONSTRUCTION, so measuring it would restate whether the anchor
# fired -- the same shape as cycle 1's PSR veto with the sign flipped. For duplicate suppression,
# which frees rounds without touching the terminal decision, a drop here is a genuine effect.
EXTRA_BOOL_METRICS = ("died_of_exhaustion",)

# Per-episode COUNTS. Their floor is a paired net difference over the corpus, NOT a flip count -- a
# flip is the right statistic for a boolean and the wrong one for a count. Measured on the granite
# train control: 1276 total violations against a replicate net of -1.
COUNT_METRICS = ("n_violations",)

# Denominators a count delta must be read against, never optimised. See `work_done`.
WORK_METRICS = ("assistant_turns", "tool_results")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default="all", choices=["all", "train", "test"])
    parser.add_argument("--input-dir", type=str, default="data")
    parser.add_argument("--input-response-data", type=str, default="results/response.jsonl")
    parser.add_argument("--output-file", type=str, default="results/scores.json")
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument("--overload", action="store_true")
    parser.add_argument("--detail", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    out_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    if not args.overload:
        if os.path.exists(args.output_file):
            with open(args.output_file, 'r') as f:
                data = json.load(f)
                print('[SCORE]', data)
            return

    detail_path = os.path.join(out_dir, "detail.jsonl")
    if args.detail and os.path.exists(detail_path):
        os.remove(detail_path)

    if args.split == "all":
        input_file = "input_data.jsonl"
    else:
        input_file = f"input_data_{args.split}.jsonl"
    with open(os.path.join(args.input_dir, input_file), 'r', encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f]

    data = [
        {**deepcopy(input_sample), "id": f"{input_sample['id']}_{i}"}
        for i in range(args.repeat)
        for input_sample in input_data
    ]

    tmp_data = {item['id']: item for item in data}

    with open(args.input_response_data, 'r', encoding='utf-8') as f:
        response_data = [json.loads(line) for line in f.readlines()]

    # Fail closed, and name what is missing. `sample_process` drops an episode whose retries are
    # exhausted, so a short response file means an INCOMPLETE run -- not a scoring bug. The old
    # message reported only two integers, which said nothing about which episodes were lost.
    if len(data) != len(response_data):
        got = {r['id'] for r in response_data}
        missing = sorted(set(tmp_data) - got, key=lambda x: (int(x.split('_')[0]), x))
        extra = sorted(got - set(tmp_data))
        raise SystemExit(
            f"response file is not the expected size: {len(response_data)} rows for "
            f"{len(data)} episodes (split={args.split}, repeat={args.repeat}).\n"
            f"  missing ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}\n"
            f"  unexpected ({len(extra)}): {extra[:20]}{' ...' if len(extra) > 20 else ''}\n"
            f"Re-run response_generator.py without --overload to resume the missing episodes."
        )

    judges = {"Overall": {}}
    terminations = {}
    detail_rows = []

    for sample in tqdm(response_data, total=len(response_data)):
        entry = tmp_data[sample['id']]
        verdict = judge(sample['messages'], entry)
        epoch = entry['id'].split('_')[1]
        source = entry['data_source']

        terminations[verdict["termination"]] = terminations.get(verdict["termination"], 0) + 1

        for bucket in ("Overall", source):
            judges.setdefault(bucket, {})
            judges[bucket].setdefault(epoch, {m: [] for m in METRICS})
            for m in METRICS:
                judges[bucket][epoch][m].append(verdict[m])

        if args.detail:
            detail_rows.append({"id": sample['id'], "data_source": source, **verdict})

    if args.detail:
        # Sorted, so the file's bytes depend on the result and not on input line order.
        detail_rows.sort(key=lambda r: (int(r["id"].split('_')[0]), r["id"]))
        with open(detail_path, 'a', encoding='utf-8') as f:
            for row in detail_rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')

    scores = {}
    for ds, epochs in judges.items():
        scores[ds] = {}
        for m in METRICS:
            per_epoch = [sum(v[m]) / len(v[m]) * 100 for v in epochs.values()]
            scores[ds][m] = f"{sum(per_epoch)/len(per_epoch):.2f} ± {std(per_epoch):.2f}"

    # The `±` above is the spread ACROSS REPLICATES, which is the aggregate variance floor when
    # --repeat > 1. It is not a confidence interval, and with --repeat 1 it is identically 0.00
    # and means nothing at all -- hence the explicit note.
    n_epochs = len(judges["Overall"])
    scores["_meta"] = {
        "split": args.split,
        "repeat": args.repeat,
        "episodes_scored": len(response_data),
        "replicates": n_epochs,
        "spread_is": ("across-replicate spread" if n_epochs > 1
                      else "MEANINGLESS: single replicate, no variance measured"),
        "termination": dict(sorted(terminations.items(), key=lambda kv: -kv[1])),
        "response_data": os.path.basename(args.input_response_data),
    }

    print('[SCORE]', {k: v for k, v in scores.items() if k != "_meta"})
    print('[TERMINATION]', scores["_meta"]["termination"])

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
