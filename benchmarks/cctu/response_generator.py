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


import argparse
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from tqdm import tqdm

from utils.utils import get_feedback_tools
from cctu_replay import ReplayExhausted
from utils.constraint_checker import DialogueConstraintChecker

HERE = os.path.dirname(os.path.abspath(__file__))

# The name vLLM serve name
SERVED_MODEL_NAMES = {
    "granite-4-1-8b": "ibm-granite/granite-4.1-8b",
    "granite-4.1-8b": "ibm-granite/granite-4.1-8b",
    "qwen-3-6-35b": "Qwen/Qwen3.6-35B-A3B",
    "minimax-m2-5": "MiniMaxAI/MiniMax-M2.5",
    "deepseek-v3-2": "deepseek-ai/DeepSeek-V3.2",
}


def extract_tool_calls_from_text(text):
    """Extract tool calls from text format.

    Supports two formats:
    1. <tool_call>\n{JSON}\n</tool_call> (matching bfcl_v4 approach)
    2. <function_calls><invoke name="..."><parameter>...</parameter>...</invoke></function_calls> (DeepSeek/the hosted API format)

    Converts to OpenAI format: {"id": "...", "function": {"name": "...", "arguments": "JSON string"}}
    """
    if not text:
        return []

    tool_calls = []

    # Try format 1: <tool_call> with JSON
    pattern1 = r'<tool_call>\s*\n(.*?)\n\s*</tool_call>'
    matches1 = re.findall(pattern1, text, re.DOTALL)

    for match in matches1:
        try:
            tool_call = json.loads(match.strip())
            if "name" in tool_call and "arguments" in tool_call:
                args = tool_call["arguments"]
                if isinstance(args, dict):
                    args = json.dumps(args)

                openai_format = {
                    "id": f"tool_call_{len(tool_calls)}",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": args
                    }
                }
                tool_calls.append(openai_format)
        except (json.JSONDecodeError, KeyError):
            pass

    # Try format 2: <function_calls><invoke name="..."> (DeepSeek/the hosted API format)
    pattern2 = r'<invoke\s+name="([^"]+)">(.*?)</invoke>'
    matches2 = re.findall(pattern2, text, re.DOTALL)

    for func_name, params_text in matches2:
        try:
            # Extract parameters from <parameter> tags
            param_pattern = r'<parameter\s+name="([^"]+)"[^>]*>([^<]*)</parameter>'
            params = {}
            for param_name, param_value in re.findall(param_pattern, params_text, re.DOTALL):
                # Try to parse as JSON if it looks like JSON
                try:
                    params[param_name] = json.loads(param_value.strip())
                except (json.JSONDecodeError, ValueError):
                    params[param_name] = param_value.strip()

            openai_format = {
                "id": f"tool_call_{len(tool_calls)}",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(params)
                }
            }
            tool_calls.append(openai_format)
        except Exception:
            pass

    return tool_calls


# ================================================================================================
# THE TURN, SPLIT INTO ITS THREE PHASES
# ================================================================================================
#
# `get_feedback` used to fuse three separate things, and the fusion is what hid the only incision
# point at which a suppression is free. Split so an interception can stand between them:
#
#     A  decode_tool_calls   what the model proposed                  <- INTERVENE HERE (l2)
#     B  validate_turn       the constraint verdict. MUTATES the budget counters
#     C  execute_turn        the tools actually run
#                                                                     <- INTERVENE HERE (l3)
#
# The ordering of A and B is the whole point. `get_feedback_if` advances `round`, `callTimes`,
# `callTimesPerTool` and `earliest_callTurnPerTool`, so a call suppressed after B has already been
# charged to the budget -- strictly worse than not intervening. See ANCHOROPT.md.
#
# `get_feedback` is KEPT below as a thin composition with unchanged semantics, so any caller that
# does not need the phases is unaffected.


def decode_tool_calls(message, use_vllm):
    """Phase A: what did the model propose? Returns `(content, tool_calls, from_text)`.

    ONE BEHAVIOUR CHANGE FROM THE ORIGINAL, AND IT IS A BUG FIX. The original set
    `tool_calls_from_text = True` unconditionally on the `use_vllm` branch, whether or not extraction
    actually found any calls. So under `--use-vllm` a plain final answer -- no tool calls at all --
    came back flagged as text-format, and `sample_process`'s `if not has_text_tool_calls` then
    DISCARDED its feedback. The consequence was severe and silent: a terminal constraint violation
    (length, format, punctuation, identifiers) was computed, never shown to the model, and `finish`
    stayed False -- so the episode re-answered identically until the round budget ran out, and the
    transcript ended in the `mid_tool_call` / budget-exhausted state with no indication why.
    Response-class constraints are 91-103 episodes each, so this reached most of the corpus.

    The flag now means what its name says: text extraction PRODUCED calls.
    """
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    if not tool_calls and content and ('<function_calls>' in content or use_vllm):
        tool_calls = extract_tool_calls_from_text(content)
        return content, tool_calls, bool(tool_calls)
    return content, tool_calls, False


def validate_turn(checker: DialogueConstraintChecker, content, tool_calls, is_final):
    """Phase B: the constraint verdict. **This is what mutates the budget counters.**

    Returns `(if_fb, if_args_fb)`. Anything that wants to act before the budget is charged must run
    before this call, which is why it is its own function.
    """
    if_fb = checker.get_feedback_if(is_final=is_final, content=content, tool_calls=tool_calls)
    if_args_fb = checker.get_feedback_tool_arguments(if_feedback=if_fb, tool_calls=tool_calls)
    return if_fb, if_args_fb


def execute_turn(sample, if_args_fb, tool_calls, *, withheld=frozenset(), substitutions=None):
    """Phase C: run the tools. Returns `(feedback, executed_by_id)`.

    THE SHORTENED BATCH IS WHAT RUNS. `cctu_apply.execute_ids` removes withheld calls BEFORE
    `get_feedback_tools` sees them, so a withheld call costs zero executions --
    `docs/EFFICIENCY_CLASS.md`'s first invariant, whose violation ("memoise after the executor ran")
    saved nothing while passing every predicate test.

    `executed_by_id` carries only what actually ran, which is the fourth invariant: a substituted
    observation must never be recorded as a result, or the replay cache would feed itself.
    """
    import cctu_apply

    substitutions = dict(substitutions or {})
    codes = json.loads(sample['codes'])
    to_run = cctu_apply.execute_ids(tool_calls, frozenset(withheld))
    executed_feedback = get_feedback_tools(if_args_fb, to_run, codes)
    executed_by_id = {str(m.get("tool_call_id") or ""): (m.get("content") or "")
                      for m in executed_feedback if m.get("role") == "tool"}
    feedback = cctu_apply.merge_feedback(tool_calls, executed_feedback, substitutions)
    return feedback, executed_by_id


def finish_state(checker: DialogueConstraintChecker, is_final, feedback):
    """Upstream's own termination test, unchanged, named so both call sites share one copy."""
    budget_exhausted = (checker.round >= checker.max_round)
    return ((is_final and (len(feedback) == 0)) or budget_exhausted), budget_exhausted


def get_feedback(message, sample, checker: DialogueConstraintChecker, use_vllm):
    """The three phases composed. Semantics unchanged except for the `from_text` fix above."""
    content, tool_calls, tool_calls_from_text = decode_tool_calls(message, use_vllm)
    is_final = (len(tool_calls) == 0)
    if_fb, if_args_fb = validate_turn(checker, content, tool_calls, is_final)
    if tool_calls and not tool_calls_from_text:
        feedback, _ = execute_turn(sample, if_args_fb, tool_calls)
    else:
        # Text-extracted calls are NOT executed by this harness, so such an episode never receives a
        # tool result and `solve_rate_is_one` -- which reads only `role: "tool"` messages -- can never
        # score it. `sample_process` refuses that case by default rather than producing unscoreable
        # episodes; see `--allow-text-tool-calls`.
        feedback = list(if_fb)
    finish, budget_exhausted = finish_state(checker, is_final, feedback)
    return finish, feedback, tool_calls_from_text, budget_exhausted


def sample_process(sample, args):
    """Run one episode.

    Returns (id, messages, analysis_entry, status)
    """
    id = sample['id']
    messages = deepcopy(sample['messages'])
    tools = json.loads(sample['tools'])
    checker = DialogueConstraintChecker(
        sample=sample,
        max_turns=20,
        validators_dir=args.input_dir,
    )

    finish = False
    analysis_entry = {
        "sample": sample,
        "conversation": []
    }

    # AnchorOpt: one per-episode hook handle. None when --controllers was omitted, which IS the
    # control -- the middleware installs nothing and every hook returns None.
    hooks = args.middleware.episode(str(id)) if getattr(args, "middleware", None) else None
    errors = []

    def _snapshot():
        """Live constraint state AS OF THIS BOUNDARY. Re-taken each time, never cached.

        `validate_turn` advances the budget counters mid-turn, so a cached snapshot would answer a
        pre-execution question with post-execution state -- the one distinction the two
        POST_GENERATION sub-moments exist to preserve.
        """
        import cctu_state
        return cctu_state.snapshot_of(checker)

    while not finish:
        # ---- l1  PRE_GENERATION: context and state, no decision yet ----------------------------
        if hooks is not None:
            try:
                directive = hooks.pre_generation(snapshot=_snapshot())
                if directive:
                    import cctu_apply
                    applied = cctu_apply.apply_directive(directive, {"content": "", "tool_calls": []})
                    if not applied.declined:
                        messages.extend(deepcopy(list(applied.inject)))
            except Exception as e:  # noqa: BLE001  -- an intervention must never kill an
                # episode. The reason is RECORDED, not swallowed: a hook that looks clean
                # because it crashed is worse than no hook.
                errors.append(f"l1: {type(e).__name__}: {e}")

        times = 0
        redecide = False
        while times < args.max_retries:
            try:
                responses = None

                _gen = dict(args.gen_kwargs)
                if getattr(args.client, "needs_episode_id", False):
                    _gen["episode_id"] = str(id)
                responses = args.client.chat(messages=messages, tools=tools, **_gen)
                if not (responses.get("choices") and responses["choices"][0].get("message")):
                    raise ValueError(
                        "provider returned no assistant message: "
                        + json.dumps(responses, ensure_ascii=False)[:400])
                tmp_message = responses["choices"][0]["message"]

                # ---- phase A: what was proposed -------------------------------------------------
                content, tool_calls, has_text_tool_calls = decode_tool_calls(
                    tmp_message, args.use_vllm)

                if has_text_tool_calls and not getattr(args, "allow_text_tool_calls", False):
                    # FAIL LOUDLY. This harness never executes a text-extracted call, so the episode
                    # would receive no tool result and `solve_rate_is_one` -- which reads only
                    # `role: "tool"` messages -- could never score it. A whole run of unscoreable
                    # episodes looks exactly like a weak model, which is the worst available failure
                    # mode. The fix is on the serving side: give vLLM a tool-call parser so the model
                    # returns structured `tool_calls`.
                    raise SystemExit(
                        f"episode {id}: the provider returned tool calls as TEXT, not as structured "
                        f"`tool_calls`. This harness does not execute those, so the episode cannot "
                        f"be scored -- `acc` would be 0 for a harness reason, not a model one.\n"
                        f"Fix the endpoint (vLLM needs --tool-call-parser / --enable-auto-tool-choice), "
                        f"or pass --allow-text-tool-calls to record the run as unscoreable on purpose.")

                withheld, substitutions = frozenset(), {}

                # ---- l2  POST_GENERATION_PRE_EXEC: BEFORE the validator charges anything --------
                if hooks is not None:
                    try:
                        import cctu_adapter
                        import cctu_apply
                        proposed = {"content": content, "tool_calls": tool_calls}
                        norm = cctu_adapter.normalize_event({
                            "boundary": "post_generation_pre_exec", "message": proposed,
                            "has_generation": True})
                        directive = hooks.post_generation_pre_exec(
                            normalized=norm, snapshot=_snapshot())
                        if directive:
                            applied = cctu_apply.apply_directive(
                                directive, proposed, snapshot=_snapshot(),
                                recorded_results=hooks.results_by_call,
                                tools_doc=checker.args_checker.tools_doc)
                            if applied.declined:
                                pass                    # traced with its reason; change nothing
                            elif applied.redecide:
                                # Discard THIS generation and generate again. The validator has not
                                # run, so no round is charged -- which is why this cell is preferable
                                # to the post-execution one. The discarded proposal lives in
                                # anchor_trace.jsonl; the injected instruction is appended to the
                                # transcript so the scored artifact shows that something intervened.
                                messages.extend(deepcopy(list(applied.inject)))
                                redecide = True
                                break
                            else:
                                tool_calls = list(applied.message.get("tool_calls") or ())
                                tmp_message = dict(tmp_message, tool_calls=tool_calls)
                                withheld = applied.withheld_ids
                                substitutions = applied.substitutions
                                if applied.inject:
                                    messages.extend(deepcopy(list(applied.inject)))
                    except Exception as e:  # noqa: BLE001  -- see l1
                        errors.append(f"l2: {type(e).__name__}: {e}")

                # ---- phase B: the constraint verdict (mutates the budget) -----------------------
                is_final = (len(tool_calls) == 0)
                # CONTAINED HERE, NOT IN THE VALIDATOR, and the placement is the point.
                #
                # `handlers/tool.py` indexes `callTimesPerTool` / `max_callTimesPerTool` -- plain dicts
                # keyed by the episode's own tool list -- with a name taken from the model's proposal,
                # so a hallucinated or truncated name raises `KeyError`. Unwrapped, that escapes to
                # `sample_process`'s generic handler, is retried 15 times to the same deterministic
                # failure, and DROPS the episode: a short `response.jsonl` that `evaluation.py`
                # correctly refuses to score, so one bad tool name costs the whole arm.
                #
                # Measured: a four-arm qwen sweep lost 2 episodes to `KeyError: 'earthquake'` (for
                # `earthquake_data_analyzer`) and 36 to `KeyError: 'histor'`. A reprompt at the
                # commitment gate induces the truncation, which is why a control never shows it.
                #
                # `tests/test_cctu_adapter.py` pins the validator's raise as a CHARACTERISATION of
                # upstream and says so: "this is LATENT, and it is pinned here rather than fixed in the
                # validator, which stays upstream's." That boundary is kept. The fix belongs at this
                # layer, which is already ours, so the vendored tree stays byte-faithful and no control
                # figure moves.
                #
                # THE TURN IS NOT SKIPPED. With no verdict, the proposal proceeds to execution, where
                # `get_feedback_tools` reports `an error occured when call <name>` and `args_checker`
                # would have refused it as nonexistent -- so the model is still told, and the episode
                # stays in the denominator instead of vanishing from it.
                try:
                    if_fb, if_args_fb = validate_turn(checker, content, tool_calls, is_final)
                except KeyError as exc:
                    errors.append(
                        f"validator KeyError on a tool name the episode does not declare ({exc}); "
                        f"turn proceeds with no constraint verdict rather than dropping the episode")
                    if_fb, if_args_fb = [], []

                # ---- phase C: execution, on the SHORTENED batch ---------------------------------
                executed_by_id = {}
                if tool_calls:
                    feedback, executed_by_id = execute_turn(
                        sample, if_args_fb, tool_calls,
                        withheld=withheld, substitutions=substitutions)
                else:
                    feedback = list(if_fb)
                finish, _budget_exhausted = finish_state(checker, is_final, feedback)

                # ---- l3  POST_EXECUTION: the verdict and the results exist ----------------------
                if hooks is not None:
                    try:
                        import cctu_adapter
                        import cctu_apply
                        norm3 = cctu_adapter.normalize_event({
                            "boundary": "post_execution",
                            "message": {"content": content, "tool_calls": tool_calls},
                            "feedback": feedback, "has_generation": True})
                        directive = hooks.post_execution(normalized=norm3, snapshot=_snapshot())
                        if directive:
                            applied = cctu_apply.apply_directive(directive, tmp_message)
                            if not applied.declined and applied.inject:
                                feedback = list(feedback) + list(applied.inject)
                                finish = False          # an injected instruction needs an answer
                        hooks.observe_turn(norm3, executed=executed_by_id)
                    except Exception as e:  # noqa: BLE001  -- see l1
                        errors.append(f"l3: {type(e).__name__}: {e}")

                # ========== TOOL CALL ANALYSIS ==========
                if tool_calls:
                    # Extract feedback messages (contain error/constraint info)
                    feedback_text = []
                    for fb in feedback:
                        if isinstance(fb, dict) and "content" in fb:
                            feedback_text.append(fb["content"])

                    # Log for analysis
                    if args.analyze_tool_calls:
                        entry = {
                            "assistant_msg": tmp_message,
                            "tool_calls": tool_calls,
                            "feedback": feedback_text,
                        }
                        analysis_entry["conversation"].append(entry)

                messages.append(dict(tmp_message))
                messages.extend(deepcopy(feedback))

                if finish:
                    # ERRORS TRAVEL WITH A SUCCESSFUL EPISODE TOO. The tail of this function attaches
                    # `errors` to the analysis entry, but this early return skipped it -- so anything
                    # recorded during an episode that then COMPLETED was silently discarded. That hid
                    # exactly the cases worth seeing: a contained validator crash, or an intervention
                    # that failed at a hook and let the turn proceed. An error that only surfaces when
                    # the episode also dies is an error you learn about twice over or not at all.
                    if errors:
                        analysis_entry["anchoropt_errors"] = errors
                    return id, messages, analysis_entry
                break
            except SystemExit:
                raise                                   # a configuration failure, never a retry
            except ReplayExhausted as e:
                # DETERMINISTIC, so retrying it 15 times burns the run for no information. This is
                # the tb2_health.py distinction applied to the replay path: an infrastructure failure
                # should be retried, and a transcript that does not cover the trajectory is neither
                # infrastructure nor the task -- it is a statement about the arm.
                errors.append(f"replay: {e}")
                print(f"[drop] episode {id}: {e}", file=sys.stderr)
                times = args.max_retries
                break
            except Exception as e:  # noqa: BLE001
                # NOT SWALLOWED. The original bound `err` and never read it, with no `break` and no
                # `raise`, so any exception -- including a bug in our own code -- became "episode
                # dropped after 15 silent retries" and surfaced only as `evaluation.py`'s length
                # mismatch. `tb2_health.py` records why the distinction matters: an INFRASTRUCTURE
                # failure should be retried, a TASK failure never should, and conflating them let two
                # uninformative trials sit in a paired denominator as though the agent had tried.
                times += 1
                errors.append(f"attempt {times}/{args.max_retries}: {type(e).__name__}: {e}")
                if times >= args.max_retries:
                    print(f"[drop] episode {id} exhausted {args.max_retries} attempts; "
                          f"last error: {type(e).__name__}: {e}", file=sys.stderr)
        if redecide:
            continue                                    # regenerate; nothing was charged
        if times >= args.max_retries:
            break

    if errors:
        analysis_entry["anchoropt_errors"] = errors
    return id, None, analysis_entry


def parse_args():
    parser = argparse.ArgumentParser()
    # Support both API-based models and local vLLM
    parser.add_argument('--model', type=str, default="qwen3-6-35b", help='Model name: API models ')
    parser.add_argument('--user', type=str, default=None)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--base_url', type=str, default=None)
    parser.add_argument('--thinking', action='store_true')

    # ---- DECODE, PINNED -------------------------------------------------------------------------
    # Every one of these is SENT EXPLICITLY rather than left to a client or server default, and all
    # three are recorded in run_manifest.json. The distinction matters: a parameter the harness does
    # not send is whatever the endpoint happens to default to today, so a run that does not state it
    # cannot be reproduced -- only re-attempted. And two runs differing in an unrecorded parameter
    # look exactly like a variance floor.
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Decode temperature (default 0.0 = greedy). Pinned and recorded.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Decode seed sent with every request (default 42). A run without a '
                             'seed cannot support a variance-floor claim.')
    parser.add_argument('--top-p', type=float, default=1.0,
                        help='Nucleus cutoff (default 1.0 = disabled). Sent explicitly because a '
                             'server-side default of anything below 1.0 makes greedy decoding a '
                             'claim about the server rather than about this run.')
    parser.add_argument('--no-seed', action='store_true', default=False,
                        help='Do not send a seed. Use only if the endpoint rejects the '
                             'parameter; the run is then recorded as unseeded.')

    # vLLM-specific options
    parser.add_argument('--vllm-url', type=str, default='http://localhost:8080/v1',
                        help='vLLM server URL (for local model serving)')
    parser.add_argument('--use-vllm', action='store_true', default=False,
                        help='Use vLLM for model serving instead of API')
    parser.add_argument('--served-model-name', type=str, default=None,
                        help='vLLM only: the name the server serves this model under. Defaults to '
                             'the SERVED_MODEL_NAMES entry for --model; required if there is none.')
    parser.add_argument('--max-tokens', type=int, default=1024,
                        help='vLLM only: per-turn completion cap (default 1024). Raise it for a '
                             'thinking model -- a truncated turn is recorded as a short answer, not '
                             'as an error.')

    parser.add_argument('--input-dir', type=str, default='data')
    # ---- WHERE THE ARTIFACTS GO ------------------------------------------------------------------
    # Derived by `cctu_paths.run_dir` from the model, the split and the policy:
    #     <results-root>/<model>/<split>_<config>/response.jsonl
    # `--output-file` still overrides it outright, which is what verify_plumbing.py uses to write into
    # a scratch directory.
    parser.add_argument('--output-file', type=str, default=None,
                        help='Explicit response path. Omitted = derived from --model / --split / '
                             '--controllers under --results-root (see cctu_paths.py)')
    parser.add_argument('--results-root', '--output_dir', dest='results_root', type=str,
                        default='results',
                        help='Root under which per-model run directories are created')
    parser.add_argument('--split', type=str, default="all", choices=["all", "train", "test"])
    parser.add_argument('--repeat', type=int, default=1,
                        help='Replicates per episode, each with its own id `<query>_<i>`. THIS is '
                             'the replicate axis: evaluation.py aggregates across them and reports '
                             'the spread, so a variance floor is `--repeat 2` in ONE run rather than '
                             'two separately launched ones. (The removed --trial only renamed '
                             'directories and was never read.)')
    parser.add_argument("--overload", action="store_true")

    parser.add_argument('--max_workers', type=int, default=4)
    parser.add_argument('--max_retries', type=int, default=15)
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=-1)


    # Anchor FLAGS TODO work in progress
    parser.add_argument('--analyze-tool-calls', action='store_true', default=False,
                        help='Log tool calls and feedback for anchor analysis')
    parser.add_argument('--controllers', type=str, default=None,
                        help='Controller spec JSON: one spec, or {"controllers": [...]}. Omitted = '
                             'the control (nothing installed), which is inert by construction -- '
                             'see verify_plumbing.py')
    parser.add_argument('--policy', type=str, default=None,
                        help='Deprecated alias for --controllers. A mechanism policy from the '
                             'previous generation (one with an "enable" block) is REFUSED, not '
                             'ignored -- see ANCHOROPT.md')
    parser.add_argument('--anchor-trace', action='store_true', default=False,
                        help='Write anchor_trace.jsonl beside the response file: '
                             'proposed -> intervened -> executed, for every firing')
    parser.add_argument('--no-one-shot', dest='one_shot', action='store_false', default=True,
                        help='Let each controller intervene on EVERY qualifying turn instead of once '
                             'per episode. Required for a per-turn gate: a condition that recurs '
                             'within an episode cannot be measured at one firing per episode. Read '
                             '--redecide-budget before using it.')
    parser.add_argument('--redecide-budget', type=int, default=1,
                        help='Redecides GRANTED PER TURN at the commitment gate (default 1, the '
                             'value action_contract declares). A granted redecide discards the '
                             'generation and regenerates WITHOUT charging a round, so this is the '
                             'only thing bounding that loop once --no-one-shot is on. 0 disables '
                             'the cell while leaving it wired.')
    parser.add_argument('--replay-from', type=str, default=None,
                        help='Replay a frozen response.jsonl instead of calling a model. For '
                             'verifying the plumbing without an API arm.')
    parser.add_argument('--allow-text-tool-calls', action='store_true', default=False,
                        help='Proceed when the provider returns tool calls as TEXT rather than as '
                             'structured tool_calls. Such episodes are NOT executed and therefore '
                             'cannot score; use this only to record a run as unscoreable on purpose.')

    return parser.parse_args()


def build_middleware(args):
    """Install the controllers, or return None -- which is the control.

    `--controllers` omitted means nothing is installed, `anchoropt.runtime_hook.decide` returns the
    host's own decision unchanged, and every hook returns None. That is the control arm, inert by
    construction rather than by a flag somebody has to remember to check, and `verify_plumbing.py` is
    what asserts it.
    """
    import cctu_middleware

    path = args.controllers
    if args.policy:
        # `--policy` is the deprecated alias. A previous-generation MECHANISM policy (one with an
        # `enable` block) is REFUSED, not ignored: ignoring it would run the control under an arm's
        # name, which is the most expensive measurement error available here.
        cctu_middleware.refuse_legacy_policy(args.policy)
        if path:
            raise SystemExit("pass either --controllers or --policy, not both")
        print("[warn] --policy is a deprecated alias for --controllers", file=sys.stderr)
        path = args.policy

    trace_path = None
    if args.anchor_trace:
        # "beside the response file", as the flag's own help promises.
        trace_path = os.path.join(os.path.dirname(args.output_file) or ".",
                                  "anchor_trace.jsonl")

    if not path and not trace_path:
        return None
    mw = cctu_middleware.AnchorOptMiddleware.from_path(
        path, one_shot=getattr(args, "one_shot", True),
        redecide_budget=getattr(args, "redecide_budget", 1), trace_path=trace_path)
    print(f"[info] AnchorOpt: {'CONTROL (nothing installed)' if mw.is_control else mw.telemetry['controllers']}"
          + (f"  trace -> {trace_path}" if trace_path else "")
          + f"  one_shot={mw.one_shot} redecide_budget={mw.redecide_budget}")
    return mw


def main():
    args = parse_args()
    import cctu_paths

    # ---- where this run's artifacts go -----------------------------------------------------------
    if args.output_file:
        if not args.output_file.endswith('.jsonl'):
            os.makedirs(args.output_file, exist_ok=True)
            args.output_file = os.path.join(args.output_file, "response.jsonl")
        run_dir = os.path.dirname(args.output_file) or "."
    else:
        run_dir = str(cctu_paths.run_dir(root=args.results_root, model=args.model,
                                         split=args.split, controllers=args.controllers))
        args.output_file = str(cctu_paths.artifact(run_dir, "response.jsonl"))
    os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir

    # ---- the decode configuration, SENT EXPLICITLY ------------------------------------------------
    # Not left to a client or a server default. `utils/vllm_client.py` used to substitute `seed=42`
    # whenever the harness sent none, which made `--no-seed` a statement the run did not honour; that
    # is fixed there, and the harness now states every value it relies on.
    args.gen_kwargs = {"top_p": args.top_p}
    if not args.no_seed:
        args.gen_kwargs["seed"] = args.seed

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

    if args.end_id == -1:
        args.end_id = len(data)
    data = data[min(args.start_id, args.end_id): min(args.end_id, len(data))]
    print(f"Total number of data: {len(data)}")

    ids = []

    # Output and log files. `analysis_file` is derived unconditionally: deriving it only under
    # --analyze-tool-calls left the --overload branch below referencing an unbound name.
    #
    # Derived from the run directory rather than by `args.output_file.replace("response", "analysis")`,
    # which silently mangled any path whose DIRECTORIES contained "response" -- and one of them now
    # can, since the run directory is built from a model name.
    analysis_file = str(cctu_paths.artifact(run_dir, "analysis.jsonl")) \
        if args.output_file == str(cctu_paths.artifact(run_dir, "response.jsonl")) \
        else args.output_file.replace("response", "analysis")
    if os.path.exists(args.output_file):
        if args.overload:
            os.remove(args.output_file)
        else:
            with open(args.output_file, "r") as f:
                ids = [json.loads(line)["id"] for line in f.readlines()]
    if args.overload and os.path.exists(analysis_file):
        os.remove(analysis_file)

    process_data = [item for item in data if item["id"] not in ids]
    if len(process_data) == 0:
        print("Done for Generation! (nothing to do -- output already complete)")
        return

    if args.max_workers == -1:
        args.max_workers = len(process_data)

    args.middleware = build_middleware(args)

    if args.replay_from:
        # No model, no GPU: the recorded generations are handed back in order. A plumbing test, and
        # on an intervention arm it will legitimately exhaust -- see cctu_replay.py.
        from cctu_replay import ReplayClient
        args.client = ReplayClient(args.replay_from)
        print(f"[info] replaying {args.replay_from}: {len(args.client.episodes)} episodes, "
              f"no model will be called")
    # Use vLLM if explicitly requested or if model matches granite pattern
    elif args.use_vllm:
        from utils.vllm_client import vllm_client
        model_for_api = args.served_model_name or SERVED_MODEL_NAMES.get(args.model)
        if not model_for_api:
            raise SystemExit(
                f"--use-vllm needs the name vLLM serves {args.model!r} under, and there is no entry "
                f"for it in SERVED_MODEL_NAMES.\nPass --served-model-name <name>, matching the "
                f"--served-model-name you gave `vllm serve`.\nKnown: {sorted(SERVED_MODEL_NAMES)}")
        print(f"Using vLLM client with model '{model_for_api}' at {args.vllm_url}")
        args.client = vllm_client(model_name=model_for_api, base_url=args.vllm_url,
                                  temperature=args.temperature, thinking=bool(args.thinking),
                                  max_tokens=args.max_tokens)
    else:
        from utils.hosted_client import client
        print(f"Using the hosted API API client for {args.model}")
        args.client = client(model=args.model, thinking=args.thinking, temperature=args.temperature)

    # ---- the manifest, written BEFORE the run so a crashed run still says what it attempted -------
    manifest = {
        "model": args.model,
        "model_dir": cctu_paths.model_dir(args.model),
        "split": args.split,
        "repeat": args.repeat,
        "config": cctu_paths.config_tag(args.controllers),
        "episodes": len(process_data),
        "decode": {
            "temperature": args.temperature,
            "seed": None if args.no_seed else args.seed,
            "top_p": args.top_p,
            "thinking": bool(args.thinking),
            "max_tokens": args.max_tokens if args.use_vllm else None,
            "sent_as": dict(args.gen_kwargs),
            "deterministic": (not args.no_seed) and args.temperature == 0.0 and args.top_p == 1.0,
        },
        "provider": ("replay" if args.replay_from else ("vllm" if args.use_vllm else "hosted")),
        # WHICH SERVER SERVED THIS RUN. Recorded because `provider: "vllm"` does not identify it, and
        # a run whose endpoint is unknown cannot be paired with another.
        #
        # This is not hypothetical. A four-arm qwen sweep ran while the cluster node behind
        # `run_vllm.py`'s default moved three times (host21 -> host19 -> host11): two arms
        # reached one server, two reached nothing, and the control had been served by a third. Nothing
        # in the artifacts said so, and `analyze_residual.compare` could not catch it because it
        # checks the decode block and the endpoint was not in one.
        #
        # An endpoint is part of the decode contract on a served model: a different node may run a
        # different build, a different quantisation, or a different tool-call parser, any of which
        # changes generations without changing temperature or seed.
        "endpoint": (args.vllm_url if args.use_vllm else None),
        "replay_from": args.replay_from,
        # max_workers does not change any episode's CONTENT -- episodes share no state -- but it does
        # change how often the 10s func_set_timeout on tool execution fires under load, which is the
        # likeliest source of a nonzero variance floor. Recorded for that reason.
        "max_workers": args.max_workers,
        "corpus": {
            "input_dir": args.input_dir,
            "input_file": input_file,
            "sha256": cctu_paths.file_digest(os.path.join(args.input_dir, input_file)),
        },
        "controllers": {
            "path": args.controllers,
            "sha256": cctu_paths.file_digest(args.controllers) if args.controllers else None,
            "installed": (args.middleware.telemetry["controllers"]
                          if args.middleware is not None else []),
            # RECORDED because they change the policy without changing the spec file. Two arms with
            # the same controllers and different firing regimes are different arms, and an
            # unrecorded difference between two runs is read as a variance floor -- the failure mode
            # `write_manifest` exists to prevent.
            "one_shot": (args.middleware.one_shot if args.middleware is not None else None),
            "redecide_budget": (args.middleware.redecide_budget
                                if args.middleware is not None else None),
        },
    }
    cctu_paths.write_manifest(run_dir, manifest)

    print(f"Generating responses for {args.model}, length={len(process_data)}!")
    print(f"  run_dir={run_dir}")
    print(f"  temperature={args.temperature}  seed={args.gen_kwargs.get('seed')}  "
          f"top_p={args.top_p}  repeat={args.repeat}  max_workers={args.max_workers}")
    if not manifest["decode"]["deterministic"]:
        print("  [warn] decode is NOT pinned to a deterministic setting; this run cannot support a "
              "variance-floor claim", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(sample_process, sample, args)
                   for sample in process_data]
        for future in tqdm(as_completed(futures), total=len(futures)):
            id, messages, analysis_entry = future.result()
            if messages:
                with open(args.output_file, "a") as f:
                    f.write(json.dumps(
                        {"id": id, "messages": messages}, ensure_ascii=False) + "\n")
                    f.flush()
            if args.analyze_tool_calls:
                with open(analysis_file, "a") as f:
                    f.write(json.dumps(
                        {"id": id, "analysis_entry": analysis_entry}, ensure_ascii=False) + "\n")
                    f.flush()

    print("Done for Generation!")

    # AnchorOpt telemetry, written beside the response file and PRINTED. `COLLABORATORS.md`: nothing
    # in the benchmark can validate an anchor, because the controllers live outside it -- "set
    # ANCHOROPT_TRAJ_DIR and check firing counts BEFORE reading any accuracy". An arm whose
    # `interventions_executed` is 0 is the control wearing an arm's name.
    if args.middleware is not None:
        telemetry = args.middleware.telemetry_snapshot()
        out = os.path.join(args.run_dir, "anchor_telemetry.json")
        with open(out, "w") as f:
            json.dump(telemetry, f, indent=2, sort_keys=True, default=str)
        print(f"[ANCHOR] exposures={telemetry['boundary_exposures']} "
              f"firings={telemetry['signal_firings']} "
              f"executed={telemetry['interventions_executed']} "
              f"loop_prevented={telemetry['loop_prevention_events']} "
              f"episodes={telemetry['episodes_seen']}")
        for label, rows in (("errors", telemetry["errors"]),
                            ("hook_errors", telemetry["hook_errors"]),
                            ("predictive_handler_errors",
                             telemetry["predictive_handler_errors"])):
            if rows:
                print(f"[ANCHOR] {label} ({len(rows)}): {rows[:3]}")
        print(f"[ANCHOR] telemetry -> {out}")


if __name__ == '__main__':
    main()
