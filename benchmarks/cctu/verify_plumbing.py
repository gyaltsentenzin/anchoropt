#!/usr/bin/env python3
"""PROVE THE CONTROL IS INERT. Run this before spending anything on a real arm.

    python benchmarks/cctu/verify_plumbing.py
    python benchmarks/cctu/verify_plumbing.py --replay-from results/response.jsonl --episodes 8

Nine checks, no GPU and no model: the generations are replayed from a frozen transcript, synthesized
from the corpus if none is supplied.

WHY THIS SCRIPT EXISTS
----------------------
`response_generator.py`'s own `--controllers` help says that omitting it "is the control (nothing
installed), which is inert by construction -- see verify_plumbing.py". That is a claim, and an
unverified claim about the control arm invalidates every paired number measured against it. The
failure mode is not hypothetical: on the other benchmark an anchor shipped with a broken predicate
import and suppressed every archival write instead of only duplicates, and it was invisible to every
layer except trajectory telemetry, which that run had not captured
([`../../COLLABORATORS.md`](../../COLLABORATORS.md)).

The checks are ordered so that a failure tells you something the next one would not:

    1 THE HOOKS ARE REACHED        an "inert" control that never ran is not evidence of anything
    2 THE CONTROL IS INERT         hooked-but-empty == un-hooked, as a mapping of id -> messages
    3 THE SCORES AGREE             same verdicts, case by case, from upstream's own scorer
    4 REPEATED RUNS AGREE          the variance floor at THIS layer is zero
    5 AN ARM ACTUALLY FIRES        an arm with 0 executions is the control wearing an arm's name
    6 AN ARM ACTUALLY DIFFERS      ... and one that changes nothing is not an intervention
    7 THE TRACE IS COMPLETE        rows == exposures, or firing counts have no denominator
    8 WITHHELD CALLS COST NOTHING  fewer proposals is not a saving; only fewer executions is
    9 THE PREDICTIVE PATH IS PURE  asking "would this violate" must not charge the budget

ON "BYTE-IDENTICAL", WHICH NEEDS QUALIFYING
-------------------------------------------
The response file's ROW ORDER is thread-completion order under `ThreadPoolExecutor`, so two runs of
one policy legitimately differ as bytes while being identical as content. Check 2 therefore compares
`{id: messages}` mappings, and check 3 compares `detail.jsonl`, which `evaluation.py` already sorts so
that "the file's bytes depend on the result and not on input line order". Claiming byte equality on
the raw response file would be a claim about the scheduler.

THE CHECKS WERE CONFIRMED TO BITE
---------------------------------
A guard that silently matches nothing is worse than no guard, so checks 5 and 6 were driven against a
deliberately broken arm: a controller keyed on `terminal_response_proposed` that asks to drop a call
from a turn proposing none. It fired **76 times and executed 0**, and both checks failed with the
decline reason readable in the trace. That is precisely the shape
[`../../COLLABORATORS.md`](../../COLLABORATORS.md) warns about -- "a configured-but-never-fired arm
looks identical to a working one in a summary" -- and it is what check 5 exists to refuse.

WHAT THIS CANNOT TELL YOU
-------------------------
Whether an intervention HELPS. The replayed generations cannot respond to an injected instruction --
the model that produced them never saw it -- so every accuracy number here is meaningless by
construction and none is printed. This is a plumbing test. The variance floor it measures is the floor
of THIS layer only: it does not include model sampling, and it does not include the 10-second
`func_set_timeout` on tool execution firing differently under load, which is the likeliest source of a
nonzero floor on real hardware.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

W = 96
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


def _say(state: str, check: str, detail: str = "") -> bool:
    mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[state]
    print(f"[{mark}] {check}")
    for line in (detail or "").splitlines():
        if line.strip():
            print(f"          {line}")
    return state != FAIL


# ------------------------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------------------------
def synthesize_transcript(data_dir: Path, out: Path, episodes: int) -> Path:
    """A frozen transcript for `episodes` real corpus episodes, so this runs with no prior artifacts.

    The generations are deliberately crude -- one tool turn, then repeated short answers -- because the
    point is to exercise the wiring, not to resemble a model. The episodes and their constraint
    validators are the corpus's own, so the validator, the executor and the scorer all run for real.
    """
    rows = []
    with open(data_dir / "input_data.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= episodes:
                break
            sample = json.loads(line)
            tool = json.loads(sample["tools"])[0]["function"]["name"]
            turns = [{"role": "assistant", "content": "looking it up", "tool_calls": [
                {"id": "0", "function": {"name": tool, "arguments": "{}"}},
                {"id": "1", "function": {"name": tool, "arguments": "{}"}}]}]
            turns += [{"role": "assistant", "content": "**partial answer**"} for _ in range(24)]
            rows.append({"id": f"{sample['id']}_0", "messages": turns})
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return out


def subset_data(data_dir: Path, out_dir: Path, episodes: int) -> Path:
    """A `--input-dir` holding only the episodes under test, so the scorer's denominators match.

    `evaluation.py` fails closed when the response file is not the size the split implies -- correct
    behaviour, and it means a subset run needs a subset input rather than a relaxed scorer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "input_data.jsonl", encoding="utf-8") as src:
        kept = [next(src) for _ in range(episodes)]
    (out_dir / "input_data.jsonl").write_text("".join(kept))
    link = out_dir / "check_code"
    if not link.exists():
        os.symlink(data_dir / "check_code", link)
    return out_dir


def write_control_spec(path: Path) -> Path:
    """A controller that fires at the commitment gate on any proposed tool action.

    Deliberately trivial: checks 5-7 ask whether an intervention REACHES the loop and changes it, not
    whether any particular condition is worth intervening on. A narrow predicate would risk a check
    failing because the fixture did not happen to contain its trigger.
    """
    import cctu_middleware

    return cctu_middleware.ControllerSpec(
        controller_id="plumbing_probe", boundary="post_generation_pre_exec", action="reprompt",
        signal="proposed_tool_action",
        theta={"text": "Plan the calls you still need before acting again."},
        provenance="verify_plumbing.py: a probe, never an anchor").write(path)


# ------------------------------------------------------------------------------------------------
# running an arm
# ------------------------------------------------------------------------------------------------
def run_arm(*, tag: str, work: Path, data_dir: Path, frozen: Path, episodes: int,
            controllers: Path | None, trace: bool) -> dict:
    """One generation + scoring pass. Returns the artifacts as loaded data."""
    out_dir = work / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    response = out_dir / "response.jsonl"

    cmd = [sys.executable, str(HERE / "response_generator.py"),
           "--input-dir", str(data_dir), "--split", "all", "--overload",
           "--end_id", str(episodes), "--max_workers", "2",
           "--output-file", str(response), "--replay-from", str(frozen)]
    if controllers:
        cmd += ["--controllers", str(controllers)]
    if trace:
        cmd.append("--anchor-trace")
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"[{tag}] generation failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

    scores = out_dir / "scores.json"
    ev = subprocess.run(
        [sys.executable, str(HERE / "evaluation.py"), "--split", "all",
         "--input-dir", str(data_dir), "--input-response-data", str(response),
         "--output-file", str(scores), "--detail", "--overload"],
        cwd=HERE, capture_output=True, text=True, check=False)
    if ev.returncode != 0:
        raise RuntimeError(f"[{tag}] scoring failed:\n{ev.stdout[-2000:]}\n{ev.stderr[-2000:]}")

    def _jsonl(path: Path) -> list:
        if not path.exists():
            return []
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    telemetry_path = out_dir / "anchor_telemetry.json"
    return {
        "tag": tag,
        "transcripts": {r["id"]: r["messages"] for r in _jsonl(response)},
        "detail": {r["id"]: r for r in _jsonl(out_dir / "detail.jsonl")},
        "trace": _jsonl(out_dir / "anchor_trace.jsonl"),
        "telemetry": (json.loads(telemetry_path.read_text()) if telemetry_path.exists() else {}),
        "stdout": proc.stdout,
    }


# ------------------------------------------------------------------------------------------------
# the checks
# ------------------------------------------------------------------------------------------------
def check_hooks_reached(hooked: dict) -> bool:
    """1. An inert control that never reached a hook proves nothing about inertness."""
    n = int(hooked["telemetry"].get("boundary_exposures", 0))
    if n <= 0:
        return _say(FAIL, "1. the hooks are reached",
                    "the hooked control recorded 0 boundary exposures, so 'inert' here would mean "
                    "'never ran'. Check that response_generator builds a middleware when "
                    "--anchor-trace is passed, and that sample_process calls its three hooks.")
    return _say(PASS, "1. the hooks are reached",
                f"{n} boundary exposures in the control, across {hooked['telemetry'].get('episodes_seen')} "
                f"episodes")


def check_inert(unhooked: dict, hooked: dict) -> bool:
    """2. Hooked-but-empty must equal un-hooked, as a mapping of id -> messages."""
    a, b = unhooked["transcripts"], hooked["transcripts"]
    if set(a) != set(b):
        return _say(FAIL, "2. the control is inert",
                    f"different episodes completed: un-hooked {len(a)}, hooked {len(b)}; "
                    f"only in un-hooked: {sorted(set(a) - set(b))[:5]}; "
                    f"only in hooked: {sorted(set(b) - set(a))[:5]}")
    differing = [k for k in a if a[k] != b[k]]
    if differing:
        return _say(FAIL, "2. the control is inert",
                    f"{len(differing)} of {len(a)} transcripts differ with the hooks merely WIRED: "
                    f"{differing[:5]}\nWiring must change nothing until a controller is installed; "
                    f"`runtime_hook.decide` returns the host's own decision when none is.")
    return _say(PASS, "2. the control is inert",
                f"all {len(a)} transcripts identical between un-hooked and hooked-but-empty "
                f"(compared as id -> messages; row order is thread-completion order)")


def check_scores_agree(unhooked: dict, hooked: dict) -> bool:
    """3. Same verdicts case by case, from upstream's own scorer."""
    a, b = unhooked["detail"], hooked["detail"]
    if set(a) != set(b):
        return _say(FAIL, "3. the scores agree", "different episodes scored")
    keys = ("acc", "SR", "PSR", "termination")
    bad = [k for k in a if any(a[k].get(m) != b[k].get(m) for m in keys)]
    if bad:
        return _say(FAIL, "3. the scores agree",
                    f"{len(bad)} episodes scored differently: {bad[:5]}")
    return _say(PASS, "3. the scores agree",
                f"{len(a)} episodes, identical acc / SR / PSR / termination")


def check_repeatable(first: dict, second: dict) -> bool:
    """4. The variance floor at THIS layer. Not the floor of a real run."""
    a, b = first["detail"], second["detail"]
    if set(a) != set(b):
        return _say(FAIL, "4. repeated runs agree", "different episodes scored between replicates")
    flips = [k for k in a if any(a[k].get(m) != b[k].get(m) for m in ("acc", "SR", "PSR"))]
    if flips:
        return _say(FAIL, "4. repeated runs agree",
                    f"{len(flips)} outcome flips between two runs of the SAME policy: {flips[:5]}\n"
                    f"Every harm tolerance in this project assumes a floor near zero. If this fails, "
                    f"small deltas are not interpretable at all.")
    return _say(PASS, "4. repeated runs agree",
                f"0 outcome flips over {len(a)} episodes, two runs of one policy.\n"
                f"THIS LAYER ONLY: no model sampling, and no load-dependent tool timeouts.")


def check_arm_fires(arm: dict) -> bool:
    """5. An arm that executed nothing is the control wearing an arm's name.

    READ FROM THE ACCUMULATED TRACE, not from `anchor_telemetry.json`. That file is written once per
    INVOCATION and overwritten, so a RESUMED run reports only its last invocation: cycle 1 had an arm
    print `exposures=60, executed=0` while its trace held 9826 rows over 280 episodes with 6 real
    executions, and it looked broken rather than weak. `analyze_residual.py` was fixed for this; this
    check is the other reader of the same number, and a pre-flight that disagrees with the scorer is
    worse than no pre-flight. The trace APPENDS, so it is the only source that survives a resume.
    """
    trace = arm.get("trace") or []
    t = arm["telemetry"]
    if not trace:
        # No trace at all is a DIFFERENT failure from "did not fire", and saying so points at the
        # flag rather than at the controller.
        return _say(FAIL, "5. an arm actually fires",
                    "no anchor_trace.jsonl rows: engagement cannot be checked from the summary, "
                    "because that file is per-invocation and overwritten. Re-run with --anchor-trace.")
    fired = [r for r in trace if r.get("fired")]
    executed = [r for r in trace if r.get("executed")]
    episodes = len({r["case_id"] for r in executed})
    if not executed:
        declines = collections.Counter(r.get("detail", "") for r in fired)
        return _say(FAIL, "5. an arm actually fires",
                    f"trace has {len(trace)} rows and {len(fired)} firings, but 0 executions. An arm "
                    f"that never executes IS the control, and comparing them would measure nothing.\n"
                    f"  why each firing was declined: "
                    + ("; ".join(f"{n}x {d!r}" for d, n in declines.most_common(4)) or "(no detail)"))
    summary_exposures = int(t.get("boundary_exposures", 0) or 0)
    resumed = "" if summary_exposures >= len(trace) else (
        f"\n  NOTE: anchor_telemetry.json reports {summary_exposures} exposures against "
        f"{len(trace)} trace rows -- that summary is per-invocation and this run was RESUMED. "
        f"Engagement is the trace's number; do not quote the summary.")
    return _say(PASS, "5. an arm actually fires",
                f"trace: {len(trace)} rows, firings={len(fired)}, executed={len(executed)} "
                f"over {episodes} episode(s); loop_prevented={t.get('loop_prevention_events')}, "
                f"redecide_declines={t.get('redecide_budget_declines')}; "
                f"controllers={sorted({r['controller'] for r in trace if r.get('controller')})}"
                + resumed)


def check_arm_differs(unhooked: dict, arm: dict) -> bool:
    """6. ... and an arm that changes nothing observable is not an intervention."""
    a, b = unhooked["transcripts"], arm["transcripts"]
    shared = sorted(set(a) & set(b))
    differing = [k for k in shared if a[k] != b[k]]
    if not differing:
        return _say(FAIL, "6. an arm actually differs",
                    f"the arm executed an intervention and yet all {len(shared)} transcripts are "
                    f"identical to the control's. Either the directive never reached the message, or "
                    f"it was applied to a copy.")
    return _say(PASS, "6. an arm actually differs",
                f"{len(differing)} of {len(shared)} transcripts differ from the control")


def check_trace_complete(hooked: dict) -> bool:
    """7. Rows == exposures. A firing count with no denominator is not a rate."""
    rows = len(hooked["trace"])
    exposures = int(hooked["telemetry"].get("boundary_exposures", 0))
    if rows != exposures:
        return _say(FAIL, "7. the trace is complete",
                    f"{rows} trace rows against {exposures} boundary exposures. A trace missing "
                    f"lines UNDERSTATES firings, which is the number that gates reading any "
                    f"accuracy -- and four concurrent appenders is how lines go missing.")
    declined = sum(1 for r in hooked["trace"] if not r.get("fired"))
    return _say(PASS, "7. the trace is complete",
                f"{rows} rows == {exposures} exposures; {declined} recorded as evaluated-but-not-fired, "
                f"which is the denominator")


def check_withheld_costs_nothing() -> bool:
    """8. The efficiency class's first invariant, with a counting stub over the executor."""
    import cctu_apply
    import utils.utils as uu

    calls = [{"id": "0", "function": {"name": "t", "arguments": '{"a":1}'}},
             {"id": "1", "function": {"name": "t", "arguments": '{"a":2}'}}]
    invoked: list[str] = []
    real = uu.call_function

    def counting(name, arguments, code, **kw):
        invoked.append(json.dumps(arguments, sort_keys=True))
        return '{"ok": true}'

    uu.call_function = counting
    try:
        applied = cctu_apply.apply_directive(
            {"kind": "suppress", "executed": True, "keep_call": True, "reason": "probe"},
            {"content": "x", "tool_calls": calls},
            recorded_results={("t", '{"a":1}'): "RECORDED VERBATIM"})
        to_run = cctu_apply.execute_ids(calls, applied.withheld_ids)
        executed = uu.get_feedback_tools([], to_run, {"t": "def t(a=None):\n    return {'ok': True}\n"})
        merged = cctu_apply.merge_feedback(calls, executed, applied.substitutions)
    finally:
        uu.call_function = real

    problems = []
    if applied.declined:
        problems.append(f"the directive declined: {applied.detail}")
    if '{"a": 1}' in invoked:
        problems.append("the executor RAN the withheld call -- a saving that saves nothing")
    if not merged or merged[0].get("content") != "RECORDED VERBATIM":
        problems.append("the substituted observation is not the tool's verbatim recorded string")
    if [m.get("tool_call_id") for m in merged] != ["0", "1"]:
        problems.append(f"feedback is not in the proposed calls' order: "
                        f"{[m.get('tool_call_id') for m in merged]}")
    if problems:
        return _say(FAIL, "8. withheld calls cost nothing", "\n".join(problems))
    return _say(PASS, "8. withheld calls cost nothing",
                f"executor invoked {len(invoked)} time(s), 0 of them the withheld call; the replayed "
                f"string is verbatim and spliced back in the proposed order")


def check_predictive_is_pure(data_dir: Path) -> bool:
    """9. Asking "would this violate anything" must not charge the budget it is asking about."""
    import cctu_state
    from utils.constraint_checker import DialogueConstraintChecker

    with open(data_dir / "input_data.jsonl", encoding="utf-8") as fh:
        sample = json.loads(fh.readline())
    sample = {**sample, "id": f"{sample['id']}_0"}
    checker = DialogueConstraintChecker(sample=sample, max_turns=20, validators_dir=str(data_dir))
    before = (checker.round, checker.callTimes, dict(checker.callTimesPerTool),
              checker.accum_max_parallelCallTypes, dict(checker.earliest_callTurnPerTool),
              checker.first_tool_name)

    tool = min(checker.max_callTimesPerTool)
    calls = [{"id": str(i), "function": {"name": tool, "arguments": "{}"}} for i in range(3)]
    snap = cctu_state.snapshot_of(checker)
    for _ in range(3):
        cctu_state.would_violate(snap, tool_calls=calls, content="x")
        cctu_state.args_invalid(snap, calls)
        cctu_state.carried_state(snap, {"tool_calls_raw": calls, "proposes_tool_call": True,
                                        "tool_name": tool, "content": "x"}, round_index=0)

    after = (checker.round, checker.callTimes, dict(checker.callTimesPerTool),
             checker.accum_max_parallelCallTypes, dict(checker.earliest_callTurnPerTool),
             checker.first_tool_name)
    if before != after:
        return _say(FAIL, "9. the predictive path is pure",
                    f"the live checker changed:\n  before {before}\n  after  {after}\n"
                    f"The stand-in in cctu_state must absorb every mutation; if the real checker "
                    f"moves, a pre-execution question has charged the budget it was asking about.")
    if cctu_state.HANDLER_ERRORS:
        return _say(WARN, "9. the predictive path is pure",
                    f"the checker is untouched, but {len(cctu_state.HANDLER_ERRORS)} handler(s) "
                    f"raised during prediction: {cctu_state.HANDLER_ERRORS[:2]}")
    return _say(PASS, "9. the predictive path is pure",
                "3 rounds of would_violate / args_invalid / carried_state left round, callTimes, "
                "callTimesPerTool, parallel and ordering state byte-identical")


# ------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Prove the CCTU control arm is inert.")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--replay-from", default=None,
                    help="a frozen response.jsonl. Synthesized from the corpus if omitted.")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="keep the working directory")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    work = Path(tempfile.mkdtemp(prefix="cctu_plumbing_"))
    print("=" * W)
    print(f"CCTU PLUMBING VERIFICATION  ·  {args.episodes} episodes  ·  {work}")
    print("=" * W)

    try:
        subset = subset_data(data_dir, work / "data", args.episodes)
        frozen = (Path(args.replay_from).resolve() if args.replay_from
                  else synthesize_transcript(subset, work / "frozen.jsonl", args.episodes))
        if not args.replay_from:
            print(f"[info] no --replay-from given; synthesized {frozen.name} from the corpus")
        spec = write_control_spec(work / "probe.json")

        common = {"work": work, "data_dir": subset, "frozen": frozen,
                  "episodes": args.episodes}
        unhooked = run_arm(tag="unhooked", controllers=None, trace=False, **common)
        hooked = run_arm(tag="hooked", controllers=None, trace=True, **common)
        hooked2 = run_arm(tag="hooked_again", controllers=None, trace=True, **common)
        arm = run_arm(tag="arm", controllers=spec, trace=True, **common)

        results = [
            check_hooks_reached(hooked),
            check_inert(unhooked, hooked),
            check_scores_agree(unhooked, hooked),
            check_repeatable(hooked, hooked2),
            check_arm_fires(arm),
            check_arm_differs(unhooked, arm),
            check_trace_complete(hooked),
            check_withheld_costs_nothing(),
            check_predictive_is_pure(subset),
        ]
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"[info] kept {work}")

    print("=" * W)
    print("NOTE no accuracy is reported and none would mean anything: the replayed generations")
    print("     cannot respond to an injected instruction. This is a PLUMBING test.")
    bad = results.count(False)
    print(f"{'ALL CHECKS PASSED' if not bad else f'{bad} CHECK(S) FAILED'}")
    print("=" * W)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
