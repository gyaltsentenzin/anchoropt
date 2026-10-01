#!/usr/bin/env python3
"""One-command CCTU runs against a the hosted API-served model: generate, then score.

    python benchmarks/cctu/run_hosted.py --model ibm-granite/granite-4.1-8b --evaluate --split train
    python benchmarks/cctu/run_hosted.py --model <m> --evaluate --split train \
        --controllers rounds/<round>/controller.json --anchor-trace

This docstring used to be a verbatim copy of `run_vllm.py`'s and described a vLLM server and a
`--vllm-url` flag this script does not have. Recorded because a wrong docstring costs a reader more
than a missing one.

WHERE THE ARTIFACTS GO. `cctu_paths.run_dir` decides, from the model, the split and the policy:
`results/<model>/<split>_<config>/`. This script does not format a path of its own -- three scripts
each doing that is how the previous layout ended up with three incompatible shapes.

DECODE IS PINNED AND RECORDED. `--temperature` / `--seed` / `--top-p` are forwarded explicitly and land
in `run_manifest.json` with the corpus and controller hashes, so a reproduction can be compared rather
than assumed. For a variance floor use `--repeat 2` (one run, two replicates, which `evaluation.py`
aggregates) and consider `--max-workers 1`, since the 10s tool-execution timeout is wall-clock.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _fail(msg: str, *detail):
    """Print failure message and exit."""
    print(f"[error] {msg}", file=sys.stderr)
    for line in detail:
        print(f"        {line}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="CCTU evaluation with the hosted API",
    )

    ap.add_argument(
        "--model",
        required=True,
        help="Model ID",
    )

    ap.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation (generate responses and score)",
    )

    # Data and output options
    ap.add_argument(
        "--data-dir",
        type=str,
        default=str(HERE / "data"),
        help="Path to evaluation data directory",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=str(HERE / "results/"),
        help="Path to output directory",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decode temperature. `type=int` here with a default of 0.01 silently coerced every "
             "explicit value to an integer, so --temperature 0.7 ran as 0",
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of evaluation repeats",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "test"]
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Decode seed, forwarded and recorded in run_manifest.json",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus cutoff, sent explicitly rather than left to the server's default",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Episodes in flight. Does not change any episode's content, but a loaded host fires the "
             "10s tool-execution timeout more often -- use 1 for variance-floor runs",
    )
    # ANCHOR ARMS. `--enable-anchor-a1` used to live here and was forwarded to a flag nothing read,
    # so an "a1" arm was byte-identical to the baseline. A controller is named by a persisted
    # `anchoropt.controller.v1` spec; that file is what response_generator VALIDATES at load, what
    # names the run directory, and what run_manifest.json hashes.
    ap.add_argument(
        "--controllers",
        type=str,
        default=None,
        help="Controller spec JSON. Omitted = the control, inert by construction "
             "(see verify_plumbing.py).",
    )
    ap.add_argument(
        "--anchor-trace",
        action="store_true",
        help="Write anchor_trace.jsonl beside the response file: proposed -> intervened -> "
             "executed, for every firing AND every declined evaluation",
    )
    # THE FIRING REGIME. The granite controls were run through THIS launcher, so a round that cannot
    # set these here cannot measure a condition that recurs inside an episode. Both land in
    # run_manifest.json, which is what makes two arms differing only here distinguishable rather than
    # a variance floor.
    ap.add_argument("--no-one-shot", dest="one_shot", action="store_false", default=True,
                   help="Let each controller intervene on EVERY qualifying turn instead of once per "
                        "episode. Required to measure a condition that recurs within an episode")
    ap.add_argument("--redecide-budget", type=int, default=1,
                   help="Redecides granted PER TURN at the commitment gate (default 1). With "
                        "--no-one-shot this is the only bound on that regeneration loop")

    ap.add_argument("--overload", action="store_true")

    args = ap.parse_args()

    if not args.evaluate:
        ap.print_help()
        return

    # THE LAYOUT HAS ONE OWNER. `cctu_paths` derives the run directory from the model, the split and
    # the policy, and `response_generator.py` and `analyze_response.py` resolve it the same way -- so
    # this script no longer builds a path of its own. Three scripts each formatting their own output
    # path is how the previous layout ended up with three mutually incompatible shapes.
    sys.path.insert(0, str(HERE))
    import cctu_paths

    run_dir = cctu_paths.run_dir(root=args.output_dir, model=args.model, split=args.split,
                                controllers=args.controllers)
    response_output = cctu_paths.artifact(run_dir, "response.jsonl")
    os.makedirs(run_dir, exist_ok=True)

    print("[info] Running CCTU evaluation...")
    print(f"[info] Data directory: {args.data_dir}")
    print(f"[info] Run directory:  {run_dir}")
    print(f"[info] Policy:         {cctu_paths.config_tag(args.controllers)}")

    cmd = [
        sys.executable,
        str(HERE / "response_generator.py"),
        "--model", args.model,
        "--input-dir", args.data_dir,
        "--output-file", str(response_output),
        "--repeat", str(args.repeat),
        # Decode, pinned and forwarded explicitly. Every one of these lands in run_manifest.json.
        "--temperature", str(args.temperature),
        "--seed", str(args.seed),
        "--top-p", str(args.top_p),
        "--split", str(args.split),
        "--max_workers", str(args.max_workers),
        "--analyze-tool-calls",
    ]
    if args.overload:
        cmd.append("--overload")
    if args.controllers:
        cmd += ["--controllers", args.controllers]
    if args.anchor_trace:
        cmd.append("--anchor-trace")
    if not args.one_shot:
        cmd.append("--no-one-shot")
    if args.redecide_budget != 1:
        cmd += ["--redecide-budget", str(args.redecide_budget)]

    # Run response generation
    print(f"[info] Generating responses...")
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        _fail("Response generation failed")

    # Run evaluation
    print(f"[info] Evaluating responses...")
    eval_cmd = [
        sys.executable,
        str(HERE / "evaluation.py"),
        "--split",
        args.split,
        "--input-dir",
        args.data_dir,
        "--input-response-data",
        str(response_output),
        "--output-file",
        str(cctu_paths.artifact(run_dir, "scores.json")),
        # BOTH are required and neither was forwarded. evaluation.py builds its expected episode
        # list from --repeat, so with --repeat > 1 it aborted on its own length assert; and
        # detail.jsonl is the only per-episode output the variance floor and the paired delta can
        # be computed from, so a run without it cannot be compared to anything.
        "--repeat",
        str(args.repeat),
        "--detail",
    ]
    if args.overload:
        eval_cmd.append("--overload")

    result = subprocess.run(eval_cmd, cwd=HERE)
    if result.returncode != 0:
        _fail("Evaluation failed")

    print(f"[info] Evaluation complete! Artifacts in {run_dir}")


if __name__ == "__main__":
    main()
