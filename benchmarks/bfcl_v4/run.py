#!/usr/bin/env python3
"""One-command BFCL v4 Agent Memory runs, with the four learned anchors on or off.

    python benchmarks/bfcl_v4/run.py --compare          # BOTH conditions -- the default
    python benchmarks/bfcl_v4/run.py --compare --dry-run    # print commands, execute nothing

    python benchmarks/bfcl_v4/run.py --only control     # no anchors      -> expect 30.03 %
    python benchmarks/bfcl_v4/run.py --only a8          # full stack      -> expect 47.52 %
    python benchmarks/bfcl_v4/run.py --compare --against a7   # pair control with A1-A5+A7

FOUR CONDITIONS, and `--compare` pairs the control with `--against` (default `a8`):

    control    every gate off
    anchors    A1-A4, the waypoint whose per-backend cells are artifact-backed
    a7         A1-A5 + A7        -> 139/303
    a8         A1-A5 + A7 + A8   -> 144/303, the full accepted stack

A7 has no policy flag -- it is dispatch-gated and enabled by ANCHOROPT_A7C, which this script sets
for the arms that need it. Running an A7 arm without it silently reproduces the arm below it.

    python benchmarks/bfcl_v4/run.py --compare --shard kv        # ONE backend, for concurrency
    python benchmarks/bfcl_v4/run.py --compare --shard all --dry-run   # print all 6 shard commands

SHARD BY BACKEND, AND RUN THE SHARDS CONCURRENTLY. `--shard` restricts a run to one backend by
writing a filtered case file. It is how the per-backend numbers are produced, and it is the DEFAULT
protocol rather than an optimisation:

  * a 3-backend x 2-arm contrast is SIX INDEPENDENT JOBS, one GPU each -- not six sequential ones
  * it buys DETERMINISM, not just wall-clock: whole-corpus runs are not byte-reproducible, while
    sharded pairs are bit-identical except in the shard where the treatment fires
  * it is sound on this corpus because NO case dependency crosses a backend boundary (asserted at
    shard time, not assumed -- `--shard` refuses if that ever stops being true)

Use `--shard all --dry-run` to print the six commands, then submit them however your scheduler wants.
Ports must be distinct per (arm x backend) if you co-locate them.

TWO CONDITIONS, FOUR ANCHORS. `--compare` runs two experimental conditions, not two anchors:

    control    every gate off
    anchors    all FOUR anchors on at once (A1+A2+A3+A4 compose into one policy artifact)

`--compare` is the default because it is the only mode that produces a defensible number: both
conditions must run in the SAME job against the same store, since only within-job paired comparison
is licensed on this corpus. Running one condition today and the other next week is not a comparison.

WHAT THIS NEEDS. The agent runs against a vLLM endpoint over HTTP, so this script does not import
torch and does not need a GPU in-process -- but it does need:

  1. a reachable vLLM server serving granite-4.1-8b   (--vllm-url, or it starts one with --model-path)
  2. the BFCL v4 evaluator on PYTHONPATH             (--evaluator, see below)
  3. the corpus                                       (--cases-dir)

The evaluator and harness ARE vendored, under benchmarks/bfcl_v4/{evaluator,harness}/ -- so
`--evaluator benchmarks/bfcl_v4` works out of the box. This script resolves the policies, states the
expected numbers, validates the environment, and then invokes the runner. Everything before that step
needs no GPU, which is why --dry-run is worth running on a laptop first.

Use `--cases memory_smoke_cases.json` for an 18-case end-to-end shakedown (15 prereq + 3 scored).
That is a PATH test, not a result: 3 scored queries cannot measure anything.

Adding another benchmark: create `benchmarks/<name>/run.py` with the same three flags. Nothing here is
shared machinery yet, deliberately -- see docs/GENERALIZABILITY.md on why one example is not enough to
abstract from.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

MODEL_CONFIG = HERE / "model_config_granite41.json"

# Which case file each split uses when `--cases` is not given. Needed by `--shard`, which has to read
# the corpus to filter it -- the runner resolves the split itself otherwise.
DEFAULT_CASES = {
    "train": "g8_balanced_train_cases.json",
    "test": "g8_balanced_test_cases.json",
}

# The two experimental CONDITIONS -- not two anchors. The `anchors` condition is the composed A1-A4
# policy: one artifact with all four anchors on.
#
# `expect` is keyed by the value of --split as the evaluator understands it. In this corpus the
# held-out fold IS the `test` split (customer/kv + finance/rec_sum + student/vector, n=84), so both
# names map to the same published numbers -- keying only on "held-out" silently lost them.
CONDITIONS = {
    "control": {
        "policy": HERE / "policy_control.json",
        "label": "control (no anchors)",
        # The MEASURED native baseline. A long-quoted 88/303 had no artifact behind it; a six-shard
        # run of this policy produced 91/303, with dev reproducing exactly. See NATIVE_BASELINE in
        # rounds/anchors.py. The shipped T0 result file still reads 88 and is left alone -- it is a
        # faithful record of that older arm.
        "expect": {"train": (91, 303), "test": (14, 84)},
    },
    "anchors": {
        # The A1-A4 waypoint. Kept as its own condition because it is the only stack whose
        # per-backend cells are recomputable from shipped artifacts.
        "policy": REPO / "rounds" / "T5_A4_no_tool_call" / "policy.json",
        "label": "A1-A4 (the artifact-backed waypoint)",
        "expect": {"train": (128, 303), "test": (22, 84)},
    },
    "a7": {
        # A1-A5 + A7. A7 has NO policy flag -- it is gated at dispatch on the backend and enabled by
        # ANCHOROPT_A7C, which `shard_env` sets. Running this without that env silently reproduces
        # A1-A5 under A7's name, which is the recorded "silent duplicate" failure mode.
        "policy": REPO / "rounds" / "T7_A7_blob_overflow" / "policy.json",
        "label": "A1-A5 + A7",
        "expect": {"train": (139, 303), "test": (34, 84)},
        "env": {"ANCHOROPT_A7C": "1"},
    },
    "a8": {
        # The full accepted stack. A8 IS policy-flagged (gate_enabled.on_dedup_clear_recovery), so
        # this policy differs from a7's by exactly that one key -- plus the same A7 env.
        "policy": REPO / "rounds" / "T8_A8_dedup_clear" / "policy.json",
        "label": "A1-A5 + A7 + A8 (full stack)",
        "expect": {"train": (144, 303), "test": (34, 84)},
        "env": {"ANCHOROPT_A7C": "1"},
    },
}


def _fail(msg: str, hint: str = "") -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    if hint:
        print(f"       {hint}", file=sys.stderr)
    raise SystemExit(2)


def preflight(args: argparse.Namespace) -> list[str]:
    """Check what we can before spending a GPU-hour. Returns human-readable warnings."""
    warn: list[str] = []

    for arm in args.arms:
        pol = CONDITIONS[arm]["policy"]
        if not pol.exists():
            _fail(f"policy for arm '{arm}' not found: {pol}")
        try:
            body = json.loads(pol.read_text())
        except json.JSONDecodeError as e:
            _fail(f"policy for arm '{arm}' is not valid JSON: {e}")
        preamble = (body.get("templates") or {}).get("on_memory_preamble", "")
        if preamble.strip():
            _fail(f"policy for arm '{arm}' carries a global preamble",
                  "this line is prompt-free by construction; a preamble invalidates the comparison")

        # INHERITED GATES. The vendored harness ships a gate registry from an earlier fork, and with
        # NO policy `on_core_clear_blocked` (that fork's G1) is ENABLED BY DEFAULT. `gate_default:
        # false` is the one flag that turns the whole registry off; without it the "control" would
        # silently carry someone else's hand-written gate and every delta here would be measured
        # against the wrong baseline. Checked rather than trusted, for both conditions.
        if body.get("gate_default") is not False:
            _fail(f"policy for arm '{arm}' does not set gate_default=false",
                  "the vendored harness enables an inherited gate (on_core_clear_blocked) by "
                  "default; without gate_default=false this run measures against a contaminated "
                  "baseline. See benchmarks/bfcl_v4/VENDORED.md.")

    if not MODEL_CONFIG.exists():
        _fail(f"model config missing: {MODEL_CONFIG}")
    cfg = json.loads(MODEL_CONFIG.read_text())
    temp = (cfg.get("decode") or {}).get("temperature")
    if temp not in (0, 0.0, 0.001):
        warn.append(f"decode temperature is {temp}; the published counts assume ~0 "
                    "(the zero variance floor depends on it)")

    if not args.dry_run:
        ev = Path(args.evaluator).expanduser() if args.evaluator else None
        if ev is None or not (ev / "run_memory_eval.py").exists():
            _fail("the BFCL v4 evaluator was not found",
                  "the evaluator is VENDORED here -- pass --evaluator benchmarks/bfcl_v4 "
                  "(see REPRODUCE.md Tier 3)")
        cases = Path(args.cases_dir).expanduser()
        if not cases.exists():
            _fail(f"cases dir not found: {cases}", "pass --cases-dir")
        if not args.vllm_url and not args.model_path:
            _fail("no model endpoint", "pass --vllm-url http://host:8080 for a running server, "
                                      "or --model-path /path/to/granite-4.1-8b to start one")
    return warn


def build_cmd(arm: str, args: argparse.Namespace, shard_backend: str | None = None,
              shard_cases: str | None = None) -> list[str]:
    """The exact run_memory_eval.py invocation for one arm, optionally scoped to one backend.

    A shard gets its own out-dir suffix so six concurrent jobs cannot overwrite each other's results.
    """
    spec = CONDITIONS[arm]
    out = Path(args.out_dir) / (f"{arm}_{shard_backend}" if shard_backend else arm)
    cmd = [
        sys.executable, str(Path(args.evaluator or "scripts") / "run_memory_eval.py"),
        "--templates", str(spec["policy"]),
        "--out-dir", str(out),
        "--model-config", str(MODEL_CONFIG),
        "--split", args.split,
        "--cases-dir", str(args.cases_dir),
        "--seed", str(args.seed),
        "--workers", str(args.workers),
        "--max-steps", str(args.max_steps),
    ]
    cases_override = shard_cases or args.cases
    if cases_override:
        # The runner wants a PATH, not a bare filename -- passing `memory_smoke_cases.json` gives
        # "[error] Cases file not found" after the model has already loaded. Resolve a bare name
        # against --cases-dir so either form works.
        c = Path(cases_override).expanduser()
        if not c.is_absolute() and not c.exists():
            c = Path(args.cases_dir).expanduser() / cases_override
        cmd += ["--cases", str(c)]
    if args.vllm_url:
        cmd += ["--skip-server-setup", "--vllm-url", args.vllm_url]
    elif args.model_path:
        cmd += ["--model-path", args.model_path, "--num-gpus", str(args.num_gpus)]
    if args.snapshot_cache_dir:
        cmd += ["--snapshot-cache-dir", str(args.snapshot_cache_dir)]
    if arm == "control":
        # The control must have NOTHING on. --disable-gates is belt-and-braces beside the policy.
        cmd += ["--disable-gates"]
    extra = [a for a in args.passthrough if a != "--"]
    cmd += extra
    return cmd


BACKENDS = ("kv", "vector", "rec_sum")

# Base for the derived vLLM ports. See `shard_port`.
SHARD_PORT_BASE = 8600


def backend_of(case_id: str) -> str:
    """Which backend a case exercises. Delegates to the adapter, which owns the grammar.

    The previous local copy ended `else "rec_sum"`, so ANY unparseable id was filed under a real
    backend. The adapter raises instead.

    Importing the adapter here also REGISTERS it with the AnchorOpt core -- this file is the
    benchmark entry point, and supplying the vocabulary is its job. The core does not go looking.
    """
    import adapter
    return adapter.backend_of(case_id)


def write_shard(cases_path: Path, backend: str, out_dir: Path) -> Path:
    """Write a case file containing only `backend`, and assert the shard is self-contained.

    The assertion is the point. Sharding is only sound because prerequisite dependencies never cross
    a backend boundary -- if that changed, a shard would reference cases it does not contain and the
    stores would be built from an incomplete chain. Checked every time rather than trusted, because
    the failure mode is a silently wrong store rather than an error.
    """
    cases = json.loads(cases_path.read_text())
    if not isinstance(cases, list):
        raise SystemExit(f"[error] expected a list of cases in {cases_path}")

    keep = [c for c in cases if backend_of(c["id"]) == backend]
    if not keep:
        raise SystemExit(f"[error] no {backend} cases in {cases_path}")

    kept_ids = {c["id"] for c in keep}
    dangling = sorted(
        dep for c in keep for dep in (c.get("depends_on") or [])
        if dep not in kept_ids and any(d["id"] == dep for d in cases)
    )
    if dangling:
        raise SystemExit(
            f"[error] the {backend} shard is NOT self-contained: {len(dangling)} dependencies "
            f"point outside it (e.g. {dangling[:3]}). Sharding by backend is unsound on this "
            "corpus -- do not work around this, the stores would be built from partial chains."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / f"cases_{backend}.json"
    shard.write_text(json.dumps(keep, indent=1))
    return shard


def shard_port(arm: str, backend: str) -> int:
    """A distinct vLLM port per (ARM x BACKEND), derived so it cannot collide.

    NOT per backend. Two arms sharing a port were safe once only because the scheduler happened to
    place them on different hosts; a co-location would have collided. Derived from both coordinates
    rather than kept in a hand-maintained table, because a table plus a fallback is exactly how the
    first version of this produced control/kv and a8/vector on the same port.
    """
    return SHARD_PORT_BASE + 10 * (sorted(CONDITIONS).index(arm) + 1) + BACKENDS.index(backend)


def condition_env(arm: str) -> dict[str, str]:
    """Env a CONDITION needs beyond its policy file.

    A7 is gated at dispatch rather than by a policy flag, so it is enabled by ANCHOROPT_A7C. Omitting
    that silently reproduces the arm below it under A7's name -- the recorded "silent duplicate"
    failure mode, which is why this is returned as data and printed in --dry-run rather than left to
    a shell wrapper.
    """
    return dict(CONDITIONS[arm].get("env") or {})


def shard_env(arm: str, backend: str, out_dir: Path) -> dict[str, str]:
    """Environment a shard needs so six concurrent jobs do not collide.

    Three things, each of which has caused a real failure when omitted:

      port          distinct per (ARM x BACKEND). Two arms sharing a port were safe once only
                    because the scheduler placed them on different hosts.
      cache dir     distinct per shard. The snapshot cache key hashes the prereq id list, so shards
                    necessarily differ -- which is correct, and it means equivalence must be checked
                    on store CONTENT, never on cache keys.
      fingerprint   SHARED across shards of the same arm, so the store content is comparable.
    """
    port = shard_port(arm, backend)
    return {
        "ANCHOROPT_SHARD_PORT": str(port),
        "ANCHOROPT_SNAPSHOT_CACHE": str(out_dir / f"_cache_{arm}_{backend}"),
        # Same value for every shard of one arm: that is what makes the concatenation comparable.
        "ANCHOROPT_STORE_CODE_FINGERPRINT": f"shard_{arm}",
    }


def _shard_plan(args: argparse.Namespace) -> list[tuple[str | None, str | None]]:
    """`[(backend, cases_path)]` for this invocation, or `[(None, None)]` for a whole-corpus run.

    Writing the shard files here means `--dry-run` produces commands a scheduler can take verbatim,
    which is the point: the six jobs are meant to be submitted concurrently, not looped over.
    """
    if not args.shard:
        return [(None, None)]
    backends = list(BACKENDS) if args.shard == "all" else [args.shard]
    source = Path(args.cases).expanduser() if args.cases else None
    if source is None or not source.exists():
        name = args.cases or DEFAULT_CASES.get(args.split)
        if not name:
            raise SystemExit(f"[error] --shard needs a case file; none known for split {args.split!r}")
        # --cases-dir defaults to a RELATIVE "data", which resolves against the caller's cwd. Try it
        # as given, then against this benchmark's own directory, so the flag works from anywhere.
        base = Path(args.cases_dir).expanduser()
        for candidate in (base / name, HERE / base / name, HERE / "data" / name):
            if candidate.exists():
                source = candidate
                break
    if source is None or not source.exists():
        raise SystemExit(
            f"[error] --shard could not find a case file for split {args.split!r} "
            f"(looked for {DEFAULT_CASES.get(args.split)} under {args.cases_dir!r} and {HERE / 'data'})"
        )
    out = [(b, str(write_shard(source, b, Path(args.out_dir) / "_shards"))) for b in backends]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run BFCL v4 Agent Memory with the A1-A4 anchors on, off, or both.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Expected: control 30.03 %, full stack 47.52 % (train, n=303). "
               "See `python scripts/run_pipeline.py --expect`.",
    )
    # NAMING. An "arm" is an experimental condition, not an anchor -- there are 2 arms and 4 anchors,
    # so `--arm both` read as "both anchors" to a first-time reader. `--compare` says what it does.
    ap.add_argument("--compare", dest="mode", action="store_const", const="paired", default="paired",
                    help="run BOTH conditions (control vs A1-A4) in one job. The default, and the "
                         "only defensible comparison.")
    ap.add_argument("--only", dest="mode", choices=tuple(CONDITIONS),
                    help="run a single condition: 'control' (no anchors) or 'anchors' (all four on)")
    ap.add_argument("--arm", dest="legacy_arm", choices=("control", "anchors", "both"),
                    help=argparse.SUPPRESS)  # accepted for compatibility; --compare/--only preferred
    ap.add_argument("--split", default="train", choices=("train", "test", "val", "all"))
    ap.add_argument("--against", choices=tuple(k for k in CONDITIONS if k != "control"),
                    default="a8",
                    help="which arm --compare pairs the control with (default: a8, the full stack)")
    ap.add_argument("--shard", choices=(*BACKENDS, "all"), default=None,
                    help="restrict to ONE backend (or print every shard's command with 'all'). "
                         "This is the default run protocol -- see the module docstring.")
    ap.add_argument("--out-dir", default="results/bfcl_v4_run",
                    help="parent dir; each arm writes to <out-dir>/<arm>/")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and expected results, execute nothing")

    src = ap.add_argument_group("sources (vendored under benchmarks/bfcl_v4/)")
    src.add_argument("--evaluator", default=os.environ.get("ANCHOROPT_EVALUATOR"),
                     help="dir containing run_memory_eval.py (env: ANCHOROPT_EVALUATOR)")
    src.add_argument("--cases-dir", default=os.environ.get("ANCHOROPT_CASES", "data"),
                     help="BFCL v4 memory corpus (env: ANCHOROPT_CASES)")

    mdl = ap.add_argument_group("model endpoint")
    mdl.add_argument("--vllm-url", default=os.environ.get("ANCHOROPT_VLLM_URL"),
                     help="a already-running vLLM server (env: ANCHOROPT_VLLM_URL)")
    mdl.add_argument("--model-path", default=os.environ.get("ANCHOROPT_MODEL_PATH"),
                     help="start a server from these weights instead")
    mdl.add_argument("--num-gpus", type=int, default=1)

    run = ap.add_argument_group("run settings (defaults reproduce the published numbers)")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--workers", type=int, default=1,
                     help="store workers; 1 is what the published runs used")
    run.add_argument("--max-steps", type=int, default=20)
    run.add_argument("--snapshot-cache-dir", default=os.environ.get("ANCHOROPT_STORE_CACHE"),
                     help="reuse prereq stores across arms (~35 min saved per arm)")
    run.add_argument("--cases", default=None,
                     help="case-list JSON inside --cases-dir (e.g. memory_smoke_cases.json). "
                          "Omit to run the full split.")
    # Forwarded verbatim, but only AFTER a literal `--`. A bare nargs="*" positional silently
    # absorbs the value of any flag this script does not define -- which is exactly how a
    # `--cases memory_smoke_cases.json` invocation ended up setting cases_dir and failing on a
    # compute node. Requiring the separator makes a typo'd flag an argparse error instead.
    run.add_argument("passthrough", nargs=argparse.REMAINDER,
                     help="after a literal `--`, extra args forwarded verbatim to run_memory_eval.py")

    args = ap.parse_args()

    # --arm is the old spelling; map it on and say so once, rather than silently accepting both.
    if args.legacy_arm:
        args.mode = "paired" if args.legacy_arm == "both" else args.legacy_arm
        print(f"note: --arm {args.legacy_arm} is deprecated; use "
              f"{'--compare' if args.legacy_arm == 'both' else f'--only {args.legacy_arm}'}\n",
              file=sys.stderr)

    args.arms = ["control", args.against] if args.mode == "paired" else [args.mode]

    print("=" * 78)
    print(f"BFCL v4 Agent Memory  |  split: {args.split}")
    if args.mode == "paired":
        print("  PAIRED COMPARISON -- 2 conditions in one job (not 2 anchors; there are 4 anchors,")
        print("  and the 'anchors' condition turns all four on together).")
    print(f"  conditions to run: {', '.join(args.arms)}")
    print("=" * 78)

    warn = preflight(args)
    for w in warn:
        print(f"  WARNING: {w}")

    for arm in args.arms:
        spec = CONDITIONS[arm]
        exp = spec["expect"].get(args.split)
        print()
        print(f"  CONDITION: {spec['label']}")
        print(f"    policy   {spec['policy'].relative_to(REPO) if spec['policy'].is_relative_to(REPO) else spec['policy']}")
        if exp:
            print(f"    expect   {exp[0]}/{exp[1]} = {100.0 * exp[0] / exp[1]:.2f} %")
        else:
            print(f"    expect   (no published number for split '{args.split}')")
        for backend, shard_cases in _shard_plan(args):
            cmd = build_cmd(arm, args, backend, shard_cases)
            if backend is None:
                for key, value in condition_env(arm).items():
                    print(f"    export   {key}={value}")
                print(f"    command  {' '.join(cmd)}")
                continue
            env_pairs = {**condition_env(arm), **shard_env(arm, backend, Path(args.out_dir))}
            print(f"    [{backend}]")
            for key, value in env_pairs.items():
                print(f"        export {key}={value}")
            print(f"        {' '.join(cmd)}")

    if len(args.arms) == 2:
        a = CONDITIONS[args.arms[0]]["expect"].get(args.split)
        b = CONDITIONS[args.arms[1]]["expect"].get(args.split)
        if a and b:
            print()
            print(f"  PAIRED DELTA expected: {100.0*a[0]/a[1]:.2f} % -> {100.0*b[0]/b[1]:.2f} % "
                  f"= +{100.0*b[0]/b[1] - 100.0*a[0]/a[1]:.2f} pp")
            print(f"  ({CONDITIONS[args.arms[0]]['label']} -> {CONDITIONS[args.arms[1]]['label']})")
            if args.split == "test":
                print("  NOTE: this is the DEV split. It is the accept/reject criterion, so it is a")
                print("  validation set -- the figure is partly selected-on, not a clean")
                print("  generalization estimate. At n=84 one case is 1.19 pp.")

    if args.shard:
        print()
        print("  SUBMIT THESE CONCURRENTLY -- one GPU each. They are independent jobs: separate")
        print("  process, own store, own snapshot cache, own port. Running them in a loop is not")
        print("  the protocol and costs wall-clock for nothing.")
        print()
        print("  Each shard keeps --workers 1. Parallelism ACROSS isolated runs is measured at 0")
        print("  flips; parallelism WITHIN one store build is forbidden -- different axes, and")
        print("  sharding is not permission to raise --workers.")
        print()
        print("  Sound on this corpus because NO case dependency crosses a backend boundary,")
        print("  asserted at shard time. After the shards land, concatenate the results and verify")
        print("  the partition is complete before scoring.")

    if args.dry_run:
        print()
        print("  --dry-run: nothing executed.")
        print("  When the run completes, check it with:")
        print(f"      python scripts/run_pipeline.py --check {args.out_dir}")
        return 0

    # The vendored runner imports `anchoropt.eval_common`, `anchoropt.memory_evaluator`, and six more
    # -- names that belong to a DIFFERENT package in this repo. _evaluator_path/ holds a shim that
    # re-exports evaluator/ under that name; prepending it here (rather than rewriting ~14 vendored
    # files) keeps the vendored code byte-identical to its source. See VENDORED.md.
    env = dict(os.environ)
    shim = HERE / "_evaluator_path"
    parts = [str(shim), str(HERE / "harness"), str(HERE)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)

    # The vendored runner resolves the vLLM binary as
    #     $VLLM_BIN  or  shutil.which("vllm")  or  a hardcoded /u/<someone-else>/... fallback
    # In a batch job neither of the first two is set -- LSF does not source a login shell, so the
    # conda env's bin/ is not on PATH -- and it fell through to another user's home directory and
    # died with FileNotFoundError. Derive it from the interpreter running us, which is correct for
    # any environment rather than just ours.
    if not env.get("VLLM_BIN"):
        candidate = Path(sys.executable).with_name("vllm")
        if candidate.exists():
            env["VLLM_BIN"] = str(candidate)
        elif not shutil.which("vllm"):
            _fail("cannot find the vllm binary",
                  "set VLLM_BIN=/path/to/env/bin/vllm, or run with an interpreter whose bin/ "
                  "contains it, or pass --vllm-url to use an already-running server")

    print()
    plan = _shard_plan(args)
    if len(plan) > 1:
        print(f"NOTE: {len(plan)} shards x {len(args.arms)} arms will run SEQUENTIALLY here.")
        print("      That is not the intended protocol -- these jobs are independent and belong")
        print("      on separate GPUs concurrently. Use --dry-run to get the commands and submit")
        print("      them to your scheduler instead.")
        print()
    for arm in args.arms:
        for backend, shard_cases in plan:
            tag = arm if backend is None else f"{arm}/{backend}"
            cmd = build_cmd(arm, args, backend, shard_cases)
            shard_specific = (
                {**condition_env(arm), **shard_env(arm, backend, Path(args.out_dir))}
                if backend else condition_env(arm)
            )
            print(f"--- running arm '{tag}' ---", flush=True)
            rc = subprocess.call(cmd, env={**env, **shard_specific})
            if rc != 0:
                print(f"arm '{tag}' failed with exit code {rc}", file=sys.stderr)
                return rc

    print()
    print("Both arms complete. Verify against the published numbers:")
    print(f"    python scripts/run_pipeline.py --check {args.out_dir}")
    return 0


if __name__ == "__main__":
    if shutil.which("nvidia-smi") is None and "--dry-run" not in sys.argv:
        print("note: no nvidia-smi on PATH. If your model endpoint is remote via --vllm-url this is "
              "fine; otherwise you are probably on a login node.\n", file=sys.stderr)
    sys.exit(main())
