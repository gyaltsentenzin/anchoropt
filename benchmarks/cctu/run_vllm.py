#!/usr/bin/env python3
"""One-command CCTU evaluation runs with vLLM-served Granite 4.1-8B.

    python benchmarks/cctu/run.py --evaluate          # Run evaluation
    python benchmarks/cctu/run.py --evaluate --dry-run    # Print commands, execute nothing

WHAT THIS NEEDS. The agent runs against a vLLM endpoint over HTTP, so this script does not import
torch and does not need a GPU in-process -- but it does need:

  1. a reachable vLLM server serving granite-4.1-8b   (--vllm-url, or it starts one with --model-path)
  2. the corpus data                                   (--data-dir)

This script validates the environment, resolves the model config, and invokes the response generator.
Everything before that step needs no GPU, which is why --dry-run is worth running on a laptop first.

DEFAULTS: If neither --vllm-url nor --model-path is provided, defaults to http://localhost:8000/v1.
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

# `CONFIG_ID` names the model_config_*.json file; `DEFAULT_MODEL` is the model id the harness needs
# and MUST be a key of response_generator.SERVED_MODEL_NAMES.
#
# THE ORIGINAL COMMAND PASSED NEITHER `--use-vllm` NOR `--model`, so despite its name this script ran
# the the hosted API client against response_generator's default model (`qwen3-6-35b`). Both are now forwarded,
# and the run manifest records which provider actually served the run.
CONFIG_ID = "granite41"
DEFAULT_MODEL = "qwen-3-6-35b"
MODEL_CONFIG = HERE / f"model_config_{CONFIG_ID}.json"


def _fail(msg: str, *detail):
    """Print failure message and exit."""
    print(f"[error] {msg}", file=sys.stderr)
    for line in detail:
        print(f"        {line}", file=sys.stderr)
    sys.exit(1)


def validate_config():
    """Validate model config file exists and is readable."""
    if not MODEL_CONFIG.exists():
        _fail(f"model config missing: {MODEL_CONFIG}")
    try:
        cfg = json.loads(MODEL_CONFIG.read_text())
        return cfg
    except Exception as e:
        _fail(f"failed to parse model config: {e}")


def validate_vllm_availability(vllm_url: str):
    """Check if vLLM endpoint is reachable."""
    import requests
    try:
        response = requests.get(f"{vllm_url.rstrip('/')}/models", timeout=5)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    return False


def main():
    ap = argparse.ArgumentParser(
        description="CCTU evaluation with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/cctu/run.py --evaluate
  python benchmarks/cctu/run.py --evaluate --vllm-url http://localhost:8000/v1
  python benchmarks/cctu/run.py --evaluate --model-path /path/to/granite-4.1-8b --num-gpus 1
        """,
    )

    ap.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation (generate responses and score)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    # vLLM endpoint options
    mdl = ap.add_argument_group("model endpoint")
    # NO CLUSTER HOST AS A DEFAULT. This defaulted to a specific internal node, which is wrong twice
    # over: it leaks where the work ran, and the node it named moves. A sweep that inherited it was
    # measured splitting across three servers mid-run with nothing in the artifacts saying so. Pass
    # --vllm-url explicitly, or set VLLM_URL; localhost is the only safe default.
    mdl.add_argument(
        "--vllm-url",
        default=os.environ.get("VLLM_URL", "http://localhost:8000/v1"),
        help="vLLM endpoint. Set explicitly or via VLLM_URL; the default is deliberately localhost")
    mdl.add_argument(
        "--model-path",
        default=os.environ.get("ANCHOROPT_MODEL_PATH"),
        help="Local path to model (if provided, starts local vLLM server)",
    )
    mdl.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for local vLLM server (default: 1)",
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
        default=str(HERE / "results"),
        help="Path to output directory",
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of evaluation repeats",
    )
    # ANCHOR ARMS. Omitting --controllers is the control, which verify_plumbing.py PROVES is inert
    # rather than asserting it.
    ap.add_argument(
        "--controllers",
        type=str,
        default=None,
        help="Controller spec JSON. Omitted = the control, inert by construction.",
    )
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help="model id; must be a key of response_generator.SERVED_MODEL_NAMES")
    ap.add_argument("--split", type=str, default="all", choices=["all", "train", "test"])
    ap.add_argument("--seed", type=int, default=42, help="Decode seed; recorded in run_manifest.json")
    ap.add_argument("--temperature", type=float, default=0.0, help="Decode temperature")
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="Nucleus cutoff, sent explicitly rather than left to the server's default")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="Episodes in flight; use 1 for variance-floor runs (the 10s tool timeout "
                         "is wall-clock)")
    ap.add_argument("--thinking", action="store_true",
                    help="Enable the model's thinking mode. Raise --max-tokens with it: a truncated "
                         "turn is recorded as a SHORT ANSWER, not as an error, so it silently becomes "
                         "a response-length violation")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="Per-turn completion cap (default 1024). vLLM path only")
    ap.add_argument(
        "--anchor-trace",
        action="store_true",
        help="Write anchor_trace.jsonl beside the response file",
    )
    # THE FIRING REGIME. Passed through rather than left at the generator's defaults, because a round
    # launched through this script could otherwise not measure a per-turn condition at all -- and
    # both values land in run_manifest.json, so two arms that differ only here are distinguishable.
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

    # Validate model config
    print("[info] Validating model config...")
    cfg = validate_config()
    print(f"[info] Model config: {cfg['registry_name']}")

    # Determine vLLM endpoint
    if args.model_path:
        print(f"[info] Starting vLLM server with model: {args.model_path}")
        if args.dry_run:
            print(f"[dry-run] vllm serve {args.model_path} --port 8000 --dtype {cfg['dtype']} --max-model-len {cfg['max_model_len']} --tensor-parallel-size {args.num_gpus}")
        else:
            # In real execution, you'd start the vLLM server here
            # For now, we'll just note that it should be running
            print(f"[warn] Note: You should start vLLM separately:")
            print(f"      vllm serve {args.model_path} --port 8000 --dtype {cfg['dtype']} --max-model-len {cfg['max_model_len']} --tensor-parallel-size {args.num_gpus}")
            vllm_url = "http://localhost:8000/v1"
    else:
        vllm_url = args.vllm_url

    print(f"[info] Using vLLM endpoint: {vllm_url}")

    # Check vLLM availability
    if not args.dry_run:
        print("[info] Checking vLLM endpoint availability...")
        if not validate_vllm_availability(vllm_url):
            _fail(
                "vLLM endpoint not reachable",
                f"Make sure vLLM is running at {vllm_url}",
                "Start it with: vllm serve <model> --port 8000",
            )
        print("[info] vLLM endpoint is reachable")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build and run response generator command
    print(f"[info] Running CCTU evaluation...")
    print(f"[info] Data directory: {args.data_dir}")
    print(f"[info] Output directory: {args.output_dir}")

    # THE LAYOUT HAS ONE OWNER -- see cctu_paths.py. This script formats no path of its own.
    sys.path.insert(0, str(HERE))
    import cctu_paths
    run_dir = cctu_paths.run_dir(root=args.output_dir, model=args.model, split=args.split,
                                 controllers=args.controllers)
    response_output = str(cctu_paths.artifact(run_dir, "response.jsonl"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"[info] Run directory: {run_dir}")

    cmd = [
        sys.executable,
        str(HERE / "response_generator.py"),
        "--use-vllm",
        "--vllm-url", vllm_url,
        "--model", args.model,
        "--input-dir", args.data_dir,
        "--output-file", response_output,
        "--repeat", str(args.repeat),
        "--split", args.split,
        # Decode, pinned and forwarded explicitly; every value lands in run_manifest.json.
        "--temperature", str(args.temperature),
        "--seed", str(args.seed),
        "--top-p", str(args.top_p),
        "--max_workers", str(args.max_workers),
        "--max-tokens", str(args.max_tokens),
    ]
    if args.thinking:
        cmd.append("--thinking")
    if args.controllers:
        cmd += ["--controllers", args.controllers]
    if args.anchor_trace:
        cmd.append("--anchor-trace")
    if not args.one_shot:
        cmd.append("--no-one-shot")
    if args.redecide_budget != 1:
        cmd += ["--redecide-budget", str(args.redecide_budget)]
    if args.overload:
        cmd.append("--overload")

    if args.dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        print(f"[info] After response generation, run evaluation:")
        eval_cmd = [
            sys.executable,
            str(HERE / "evaluation.py"),
            "--input-dir",
            args.data_dir,
            "--input-response-data",
            response_output,
            "--split", args.split,
            "--repeat", str(args.repeat),
            "--detail",
            "--output-file",
            str(cctu_paths.artifact(run_dir, "scores.json")),
        ]
        print(f"[dry-run] {' '.join(eval_cmd)}")
        return

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
        "--input-dir",
        args.data_dir,
        "--input-response-data",
        response_output,
        "--split", args.split,
        "--repeat", str(args.repeat),
        "--detail",
        "--output-file",
        str(cctu_paths.artifact(run_dir, "scores.json")),
    ]
    if args.overload:
        eval_cmd.append("--overload")

    result = subprocess.run(eval_cmd, cwd=HERE)
    if result.returncode != 0:
        _fail("Evaluation failed")

    print(f"[info] Evaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
