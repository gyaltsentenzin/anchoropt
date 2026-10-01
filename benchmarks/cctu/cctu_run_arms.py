#!/usr/bin/env python3
"""Run a ROUND's arms: one launcher invocation per candidate, screened and budgeted.

    python benchmarks/cctu/cctu_run_arms.py \\
        --round-dir ../../rounds/CCTU_CYCLE2 \\
        --screen-report ../../rounds/CCTU_CYCLE2/screen_SR.json \\
        --model granite-4-1-8b --split train --repeat 2 \\
        --max-arms 6 --no-one-shot --dry-run

THIS IS NOT A LOOP OVER EVERY CANDIDATE, AND THAT IS THE POINT
--------------------------------------------------------------
`cycle1_propose.py --exhaust` emits 330 behaviourally distinct candidates. Running all of them and
keeping the best is not a stronger round than cycle 1's six -- it is a weaker one, for a reason that
has nothing to do with compute:

    the benefit bar is a NET GAIN over a measured per-episode flip floor (granite train: acc 1,
    SR 3, PSR 2, plus 1 for between-job drift). A bar like that is calibrated for ONE comparison.
    Run 330 arms against it and arms will clear it by noise alone; keep the best and you have
    selected on the noise. `FROZEN.md` exists because "a stopping rule chosen after seeing the
    result is a rationalisation, not a rule", and "run everything, then pick the winner" is that
    rationalisation with a bigger denominator.

So `--max-arms` is REQUIRED and has no default. Choosing an arm budget is a pre-registration decision
that belongs in FROZEN.md next to the tolerances, not a flag with a convenient default. If the budget
is smaller than the screened set -- it will be -- the round must say in FROZEN.md which arms it
spends and why, before this script runs. This script refuses to decide that: it takes candidates in
the order it is given them, stops at the budget, and NEVER ranks.

WHAT IT DOES DO
---------------
  * reads a screen report and queues only `PASS` candidates (`--candidates` takes a raw directory
    instead, for a round that screens by hand);
  * PREFLIGHTS every spec by loading and validating it in a FRESH interpreter. This is the check
    worth the most: a synthesized signal persisted as a bare `signal` name validates inside the
    proposer -- where expansion registered it in memory -- and raises on load anywhere else. Without
    this preflight that surfaces as every arm refusing to start after the GPU time is already
    committed;
  * runs ONE LAUNCHER PROCESS PER ARM, because `runtime_hook` holds one global installed-controller
    table and `AnchorOptMiddleware.reset`'s docstring is explicit: "One arm, one process";
  * SKIPS an arm whose run directory is already complete, so an interrupted sweep resumes instead of
    re-running what it already paid for;
  * records every arm's status, exact command and duration in `<round-dir>/sweep.json`.

WHAT IT REFUSES TO DO
---------------------
It does not score, compare, rank or nominate a winner, and it prints no accuracy. The comparison is
`analyze_residual.py --compare`, judged against a FROZEN.md that was hashed before this ran; a sweep
driver that printed a leaderboard would be choosing the round's answer in the order it happened to
queue things. It prints the compare commands and stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cctu_paths  # noqa: E402

W = 96


# ------------------------------------------------------------------------------------------------
# which candidates
# ------------------------------------------------------------------------------------------------
def queue_from_screen(report_path: Path) -> list[Path]:
    """The `PASS` candidates, in the report's own order.

    Order is preserved rather than sorted: the report follows the proposer's order, which is
    deterministic and carries no rank. Sorting here would invent one.
    """
    report = json.loads(report_path.read_text())
    out: list[Path] = []
    seen: set[str] = set()
    for row in report.get("results", ()):
        if row.get("verdict") != "PASS":
            continue
        source = row.get("source")
        if not source:
            raise SystemExit(f"{report_path}: a PASS row has no `source` path; re-run cctu_screens.py")
        # DEDUPED BY PATH, order preserved. A report screened from a `propose.json` gives every row
        # the same source, and queueing one file N times would run one arm N times under one run
        # directory -- each invocation resuming the previous one's artifacts. `--candidates <dir>`,
        # where each spec is its own file, is the shape that maps one candidate to one arm.
        if source in seen:
            continue
        seen.add(source)
        out.append(Path(source))
    if len(out) == 1 and len(report.get("results", ())) > 1:
        print(f"[WARN] {report_path} screened a bundle, not a candidates/ directory: "
              f"{len(report['results'])} rows resolve to ONE spec file. Point cctu_screens.py at "
              f"the candidates/ directory so each arm has its own spec.", file=sys.stderr)
    return out


def queue_from_dir(candidates_dir: Path) -> list[Path]:
    return sorted(candidates_dir.glob("*.json"))


# ------------------------------------------------------------------------------------------------
# preflight
# ------------------------------------------------------------------------------------------------
_PREFLIGHT = """
import json, sys
sys.path.insert(0, {here!r})
import cctu_middleware as mw
specs = mw.ControllerSpec.load(sys.argv[1])
if not specs:
    raise SystemExit("spec file declares no controller")
for s in specs:
    s.validate()
print(json.dumps({{"controllers": [s.controller_id for s in specs],
                   "signals": [s.phi_name for s in specs]}}))
"""


def preflight(spec: Path) -> tuple[bool, str]:
    """Load and validate `spec` in a FRESH interpreter, the way the runner will.

    A separate process is the whole value here. In-process validation passes for a synthesized signal
    whose evaluator was registered by the proposer and never persisted; the runner is a different
    process and refuses it. See `cctu_middleware.ControllerSpec`'s note on why `predicate` exists:
    a synthesized name is "undeclared and refused" on load, so the EXPRESSION has to be persisted.
    """
    proc = subprocess.run([sys.executable, "-c", _PREFLIGHT.format(here=str(HERE)), str(spec)],
                          cwd=str(HERE), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, (tail[-1] if tail else f"exit {proc.returncode}")
    return True, (proc.stdout or "").strip()


# ------------------------------------------------------------------------------------------------
# is this arm already done
# ------------------------------------------------------------------------------------------------
def completion(run_dir: Path, *, expected_episodes: int | None) -> tuple[bool, str]:
    """Is this run finished? `(complete, why)`.

    An arm is complete only when its response file holds the expected number of episodes. A SHORT
    response file is an incomplete run, not a scoring bug -- `evaluation.py` fails closed on exactly
    this and names the missing ids -- and treating one as done is how `qwen/test_baseline` came to sit
    at 82 of 120 transcripts while looking like a scoreable control.
    """
    response = run_dir / "response.jsonl"
    if not response.exists():
        return False, "no response.jsonl"
    rows = sum(1 for line in response.read_text(encoding="utf-8").splitlines() if line.strip())
    if expected_episodes is not None and rows != expected_episodes:
        return False, f"{rows} of {expected_episodes} episodes"
    if not (run_dir / "detail.jsonl").exists():
        return False, f"{rows} episodes but no detail.jsonl (not scored)"
    return True, f"{rows} episodes, scored"


def corpus_episodes(data_dir: Path, split: str, repeat: int) -> int | None:
    name = "input_data.jsonl" if split == "all" else f"input_data_{split}.jsonl"
    path = data_dir / name
    if not path.exists():
        return None
    n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return n * int(repeat)


# ------------------------------------------------------------------------------------------------
def build_command(args, spec: Path) -> list[str]:
    """One launcher invocation. Every decode knob is forwarded EXPLICITLY.

    Nothing is left to a launcher default: `analyze_residual.compare` refuses a pairing whose decode
    block differs from the control's, and a default that changes between releases is exactly the
    unrecorded parameter that reads as a variance floor.
    """
    launcher = HERE / ("run_hosted.py" if args.provider == "hosted" else "run_vllm.py")
    cmd = [sys.executable, str(launcher), "--evaluate",
           "--model", args.model,
           "--split", args.split,
           "--repeat", str(args.repeat),
           "--seed", str(args.seed),
           "--temperature", str(args.temperature),
           "--top-p", str(args.top_p),
           "--max-workers", str(args.max_workers),
           "--data-dir", str(args.data_dir),
           "--output-dir", str(args.output_dir),
           "--controllers", str(spec),
           "--anchor-trace"]
    if args.provider == "vllm" and getattr(args, "vllm_url", None):
        cmd += ["--vllm-url", args.vllm_url]
    if not args.one_shot:
        cmd.append("--no-one-shot")
    if args.redecide_budget != 1:
        cmd += ["--redecide-budget", str(args.redecide_budget)]
    if args.overload:
        cmd.append("--overload")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a round's arms, one launcher process each. Screened, budgeted, resumable.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--screen-report", help="a cctu_screens.py --json report; queues PASS rows only")
    src.add_argument("--candidates", help="a candidates/ directory; queues every spec in it")
    ap.add_argument("--round-dir", required=True, help="where sweep.json and the arm logs are written")
    # THE ARM BUDGET. Required, no default -- see the module docstring.
    ap.add_argument("--max-arms", type=int, required=True,
                    help="how many arms this round spends. REQUIRED and deliberately without a "
                         "default: the number belongs in FROZEN.md beside the tolerances, because a "
                         "net-gain bar calibrated for one comparison does not survive many. "
                         "WHEN THE BUDGET IS SMALLER THAN THE PASS ROWS, WHICH ROWS RUN IS A ROUND "
                         "DECISION: a screen report is in the proposer's EMISSION order and the "
                         "screens do not rank it (see cctu_screens.py), so this flag takes the first "
                         "N of whatever order it is given. State the selection rule in FROZEN.md and "
                         "apply it yourself -- --candidates takes a directory, so a round can queue "
                         "exactly the set its rule names. Measured cost of not doing this: cycle 4's "
                         "first dry run spent all six arms on a family two earlier cycles had already "
                         "measured null, because the grammar emitted it first")
    ap.add_argument("--provider", choices=("hosted", "vllm"), default="hosted",
                    help="which launcher to drive (default hosted, which is what the granite controls "
                         "were run on)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="train", choices=("all", "train", "test"))
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-workers", type=int, default=8)
    # THE ENDPOINT, PINNED FOR THE WHOLE SWEEP.
    #
    # `run_vllm.py`'s default is a cluster node that moves. A four-arm qwen sweep ran while it went
    # host21 -> host19 -> host11: arms 258/259 reached one server, 260/261 reached nothing
    # and produced no response.jsonl at all, and the control had been served by a third. Each arm
    # inherited whatever the launcher's default was AT ITS OWN LAUNCH, so the sweep silently split
    # across servers mid-run.
    #
    # Resolved ONCE here and passed to every arm, so a sweep is served by one endpoint or fails
    # loudly. Defaults to the launcher's own default, which keeps a single-arm invocation unchanged.
    ap.add_argument("--vllm-url", default=None,
                    help="vLLM endpoint, resolved once and pinned for every arm in the sweep "
                         "(vllm provider only). Omitted takes run_vllm.py's default -- which is a "
                         "moving cluster node, so pass it explicitly for a round that must be paired")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--output-dir", default=str(HERE / "results"))
    ap.add_argument("--no-one-shot", dest="one_shot", action="store_false", default=True,
                    help="forwarded to the launcher; required for a per-turn condition")
    ap.add_argument("--redecide-budget", type=int, default=1, help="forwarded to the launcher")
    ap.add_argument("--control", default=None,
                    help="the control run directory, used only to print the compare commands")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after an arm fails. OFF by default: a launcher failure is usually "
                         "a configuration error that applies to every remaining arm too")
    ap.add_argument("--overload", action="store_true",
                    help="regenerate an arm from scratch, discarding whatever is in its run "
                         "directory. Without this a PARTIAL run directory is refused rather than "
                         "resumed onto -- see the stale-output check below")
    ap.add_argument("--resume", action="store_true",
                    help="finish a PARTIAL run directory instead of refusing it, keeping the episodes "
                         "already generated. Use this when an allocation ended mid-arm: an arm takes "
                         "longer than a node is held, so a split is unavoidable rather than careless. "
                         "The endpoints are recorded cumulatively and the pairing must then be cleared "
                         "by check_pairing.py, which measures comparability instead of assuming it")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight and print the commands, run nothing. Do this first")
    args = ap.parse_args()

    round_dir = Path(args.round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)

    # FROZEN.md, hashed and recorded. Not enforced as a gate -- a round may legitimately dry-run
    # before its criteria are written -- but a REAL sweep with no frozen criteria is announced loudly,
    # because that is the one defect this project's whole protocol is built to prevent.
    frozen = round_dir / "FROZEN.md"
    frozen_sha = None
    if frozen.exists():
        frozen_sha = "sha256:" + hashlib.sha256(frozen.read_bytes()).hexdigest()
    elif not args.dry_run:
        print(f"[WARN] no {frozen}. Acceptance criteria written after seeing a result are a "
              f"rationalisation, not a rule -- freeze and hash them before spending arms.",
              file=sys.stderr)

    queue = (queue_from_screen(Path(args.screen_report)) if args.screen_report
             else queue_from_dir(Path(args.candidates)))
    expected = corpus_episodes(Path(args.data_dir), args.split, args.repeat)

    print("=" * W)
    print(f"CCTU ARMS  round={round_dir}  model={args.model}  split={args.split}  "
          f"repeat={args.repeat}")
    print("=" * W)
    print(f"  provider={args.provider}  one_shot={args.one_shot}  "
          f"redecide_budget={args.redecide_budget}  max_workers={args.max_workers}")
    print(f"  decode: temperature={args.temperature} seed={args.seed} top_p={args.top_p}")
    print(f"  queued={len(queue)}  arm budget={args.max_arms}  episodes/arm={expected}")
    print(f"  FROZEN.md={frozen_sha or 'ABSENT'}")
    if len(queue) > args.max_arms:
        print(f"  [note] {len(queue)} candidates passed screening and the budget is {args.max_arms}. "
              f"The first {args.max_arms} in the report's order will run; FROZEN.md must say why "
              f"those are the arms this round spends.")
    print()

    records: list[dict[str, Any]] = []
    spent = failed = 0
    for spec in queue:
        if spent >= args.max_arms:
            records.append({"spec": str(spec), "status": "not_queued",
                            "detail": f"arm budget {args.max_arms} spent"})
            continue

        run_dir = cctu_paths.run_dir(root=args.output_dir, model=args.model, split=args.split,
                                     controllers=spec)
        done, why = completion(run_dir, expected_episodes=expected)
        if done:
            print(f"  [skip] {spec.stem}\n         already complete: {why}")
            records.append({"spec": str(spec), "run_dir": str(run_dir), "status": "skipped",
                            "detail": why})
            continue

        # A PARTIAL run directory is REFUSED, not resumed onto.
        #
        # `response_generator.py` resumes by episode id unless it is passed `--overload`, and this
        # runner never passed it. So an arm whose previous attempt died partway silently KEPT those
        # episodes and generated only the remainder -- across whatever endpoint happened to serve the
        # second attempt. Measured: one arm's directory held 8 episodes from `host7` while 272
        # were appended from `host9`, giving 280 rows that `completion()` calls complete and that
        # `analyze_residual.compare` cannot tell apart from one clean job, because `run_manifest.json`
        # records only the LAST endpoint. Nothing in the artifacts said so.
        #
        # Pinning `--vllm-url` closed this BETWEEN arms; it does nothing against output already on
        # disk. An arm is meant to be one job, so the two correct answers are to delete the directory
        # or to regenerate it, and neither can be chosen silently here.
        #
        # A COMPLETE but unscored directory is not stale: resuming it generates nothing and only
        # re-runs `evaluation.py`, which is the legitimate way to add a missing `detail.jsonl`.
        # `--resume` is the deliberate form of this, for the case the refusal cannot serve: an arm
        # takes longer to generate than an allocation lasts, so REFUSING every split makes the round
        # unrunnable rather than rigorous. What the refusal is actually protecting is the artifact --
        # `endpoint` used to be overwritten, so a split arm claimed one server and had several. With
        # `endpoints` accumulating (`cctu_paths.write_manifest`) the split is on the record, and
        # `check_pairing.py` decides whether it mattered by checking the non-firing episodes against
        # the control. That is a measurement; this check is only a default.
        response = run_dir / "response.jsonl"
        rows = (sum(1 for line in response.read_text(encoding="utf-8").splitlines() if line.strip())
                if response.exists() else 0)
        if rows and expected is not None and rows != expected and not (args.overload or args.resume):
            print(f"  [FAIL] {spec.stem}\n         partial output: {why}. Finishing it on another "
                  f"server would split the arm across nodes, and only the LAST is recorded as "
                  f"`endpoint`.\n         --resume to finish it (the split is then recorded in "
                  f"`endpoints` and must be cleared by check_pairing.py),\n         --overload to "
                  f"regenerate from scratch, or delete {run_dir}")
            records.append({"spec": str(spec), "run_dir": str(run_dir), "status": "stale_output",
                            "detail": why})
            failed += 1
            if not args.keep_going:
                print(f"\n  stopping: a dirty run directory is a state problem, not a transient "
                      f"one. Re-run with --keep-going to continue past it.")
                break
            continue

        ok, detail = preflight(spec)
        if not ok:
            print(f"  [FAIL] {spec.stem}\n         preflight: {detail}")
            records.append({"spec": str(spec), "run_dir": str(run_dir), "status": "preflight_failed",
                            "detail": detail})
            failed += 1
            if not args.keep_going:
                print(f"\n  stopping: a spec that does not load will not load for the other arms "
                      f"either. Re-run with --keep-going to override.")
                break
            continue

        cmd = build_command(args, spec)
        if args.dry_run:
            print(f"  [dry ] {spec.stem}\n         {' '.join(cmd)}")
            records.append({"spec": str(spec), "run_dir": str(run_dir), "status": "dry_run",
                            "command": cmd})
            spent += 1
            continue

        log = round_dir / "logs" / f"{spec.stem}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [run ] {spec.stem}  ({why})\n         -> {run_dir}\n         log {log}")
        started = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(" ".join(cmd) + "\n\n")
            fh.flush()
            proc = subprocess.run(cmd, cwd=str(HERE), stdout=fh, stderr=subprocess.STDOUT,
                                  check=False)
        elapsed = round(time.time() - started, 1)
        status = "ok" if proc.returncode == 0 else "failed"
        complete, detail_after = completion(run_dir, expected_episodes=expected)
        if status == "ok" and not complete:
            # A zero exit with a short response file is the failure mode that looks like success.
            status = "incomplete"
        print(f"         {status} in {elapsed}s -- {detail_after}")
        records.append({"spec": str(spec), "run_dir": str(run_dir), "status": status,
                        "returncode": proc.returncode, "seconds": elapsed,
                        "detail": detail_after, "command": cmd, "log": str(log)})
        spent += 1
        if status != "ok":
            failed += 1
            if not args.keep_going:
                print(f"\n  stopping after a {status} arm. Read {log}. "
                      f"Re-run with --keep-going to continue past it.")
                break

    # Every QUEUED candidate gets a row, including the ones a break never reached. A sweep record
    # that silently omits them cannot be told apart from a sweep that was never asked to run them --
    # the same reason `cctu_paths.artifact` refuses an undeclared name.
    accounted = {r["spec"] for r in records}
    for spec in queue:
        if str(spec) not in accounted:
            records.append({"spec": str(spec), "status": "not_run",
                            "detail": "the sweep stopped before this candidate"})

    sweep = {
        "round_dir": str(round_dir), "frozen_sha256": frozen_sha,
        "model": args.model, "split": args.split, "repeat": args.repeat,
        "provider": args.provider, "endpoint": getattr(args, "vllm_url", None),
        "one_shot": args.one_shot,
        "redecide_budget": args.redecide_budget,
        "decode": {"temperature": args.temperature, "seed": args.seed, "top_p": args.top_p},
        "max_arms": args.max_arms, "queued": len(queue), "arms_spent": spent, "failed": failed,
        "episodes_per_arm": expected, "dry_run": args.dry_run, "arms": records,
    }
    (round_dir / "sweep.json").write_text(json.dumps(sweep, indent=2, sort_keys=True, default=str)
                                          + "\n")

    print()
    print("-" * W)
    print(f"  arms spent={spent}/{args.max_arms}  failed={failed}  "
          f"skipped={sum(1 for r in records if r['status'] == 'skipped')}")
    print(f"  sweep -> {round_dir / 'sweep.json'}")
    ran = [r for r in records if r["status"] in ("ok", "skipped")]
    if ran and args.control:
        # NO SCORES AND NO RANKING. The comparison is judged against the frozen criteria, and a
        # driver that printed a leaderboard would be choosing the round's answer.
        print("\n  Score each arm against the control, then read the verdict against FROZEN.md:")
        for r in ran:
            print(f"    python analyze_residual.py --compare {args.control} {r['run_dir']}")
    elif ran:
        print("\n  Pass --control <control run dir> to have the compare commands printed.")
    print("=" * W)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
