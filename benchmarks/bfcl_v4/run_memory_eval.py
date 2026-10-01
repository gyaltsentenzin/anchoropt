"""
AnchorOpt memory evaluation — standalone test/val set eval.

Loads a templates JSON, spins up vLLM, runs _eval_batch on the requested
split, and writes eval_results.json with per-category accuracy.

Usage:
    python scripts/run_memory_eval.py \\
        --templates results/memory_run9/templates_best.json \\
        --model-path /path/to/granite-4.1-8b \\
        --served-model-name granite41-8b \\
        --out-dir    results/memory_run9/eval_test

    # If vLLM already running:
    python scripts/run_memory_eval.py \\
        --skip-server-setup --vllm-url http://localhost:8080 \\
        --templates results/memory_run9/templates_best.json \\
        --out-dir   results/memory_run9/eval_test
"""

import argparse
import os as _os
import json
import os
import signal
import pathlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ── Path setup (mirrors run_memory_train.py) ──────────────────────────────────
_SCRIPT_DIR     = Path(__file__).resolve().parent
_ANCHOROPT_ROOT = _SCRIPT_DIR.parent
_BFCL_ROOT      = _ANCHOROPT_ROOT.parent / "berkeley-function-call-leaderboard"

for _p in [str(_ANCHOROPT_ROOT), str(_BFCL_ROOT), str(_SCRIPT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import helpers from training script
from run_memory_train import (  # noqa: E402
    _VllmServer,
    _eval_batch,
    _per_category_accuracy,
)
from anchoropt.eval_common import (  # noqa: E402
    load_model_config,
    apply_model_config,
    decode_params,
    split_prereqs,
    split_queries,
    prereq_union,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AnchorOpt memory eval — one split.")

    # Model config (single source of truth; optional — omitting it leaves every
    # hardcoded default below unchanged, i.e. byte-identical stock).
    parser.add_argument("--model-config", default=None,
                        help="Path to a model_config.json (see anchoropt/model_config.json). "
                             "Supplies defaults for the server/handler/decode args below; "
                             "explicit CLI flags still override it.")

    # Server
    srv = parser.add_argument_group("Server")
    srv.add_argument("--model-path",   default=None, help="Model path (required unless --skip-server-setup)")
    srv.add_argument("--port",         type=int, default=8080)
    srv.add_argument("--num-gpus",     type=int, default=1)
    srv.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    srv.add_argument("--max-model-len", type=int, default=0, help="0=auto")
    srv.add_argument("--dtype",        default="bfloat16")
    srv.add_argument("--skip-server-setup", action="store_true")
    srv.add_argument("--vllm-url",     default=None)
    srv.add_argument("--served-model-name", default=None)

    # Handler
    hnd = parser.add_argument_group("Handler")
    hnd.add_argument("--handler-module", default="bfcl_eval.model_handler.local_inference.granite_4")
    hnd.add_argument("--handler-class",  default="Granite4PromptHandler")
    hnd.add_argument("--registry-name",  default="ibm-granite/granite-4.1-8b-prompting")

    # Eval
    ev = parser.add_argument_group("Eval")
    ev.add_argument("--templates", required=True,
                    help="Path to templates JSON (e.g. results/run9/templates_best.json)")
    ev.add_argument("--cases",     default=None,
                    help="Path to cases JSON (default: <cases-dir>/memory_<split>_cases.json)")
    ev.add_argument("--cases-dir", default=str(_ANCHOROPT_ROOT / "data"))
    ev.add_argument("--split",     default="test", choices=["test", "val", "train", "all"],
                    help="Which split to evaluate. 'all' combines train+val+test queries (465 total, matches official BFCL count).")
    ev.add_argument("--out-dir",   required=True,
                    help="Directory to write eval_results.json")
    ev.add_argument("--max-steps", type=int, default=20)  # W2 fidelity: match official MAXIMUM_STEP_LIMIT
    ev.add_argument("--seed", type=int, default=42,
                    help="W1: decode seed sent to vLLM; keep identical to the train seed for comparable numbers")
    ev.add_argument("--workers", type=int, default=1,
                    help="Parallel eval workers (1=sequential, matches W1 determinism default). "
                         "Concurrent continuous batching is not byte-stable, so results may differ "
                         "from a serial run of the same split/templates.")
    ev.add_argument("--disable-gates", action="store_true",
                    help="Turn OFF the G1/D3/G3/G4 deterministic gates (for the gates-off arms of the 2x2 baseline)")
    ev.add_argument("--enable-reroute", action="store_true",
                    help="Turn ON the opt-in reroute remedy (Phase 3): core-full failures are "
                         "re-dispatched to archival instead of just reprompted. Default off, "
                         "matching the byte-identical-stock guarantee.")
    ev.add_argument("--enable-forced-retrieval", action="store_true",
                    help="Turn ON the opt-in forced archival retrieval remedy (Phase 6): when "
                         "the model is about to give up without searching archival memory this "
                         "turn, dispatch the search deterministically and show it the result "
                         "instead of only reprompting. Default off, matching the "
                         "byte-identical-stock guarantee.")
    ev.add_argument("--enable-forced-key-search", action="store_true",
                    help="Turn ON the opt-in forced key-search remedy: when the model is about "
                         "to give up without trying archival_memory_key_search this turn, "
                         "dispatch it deterministically and show it the result instead of only "
                         "reprompting. Default off, matching the byte-identical-stock guarantee. "
                         "Trainable via run_memory_train.py's gate-arm search; this flag brings "
                         "it into eval/leaderboard transfer alongside the other two remedies.")
    ev.add_argument("--snapshot-cache-dir", default=None,
                    help="Shared on-disk snapshot-store cache dir. Default: <out-dir-parent>/.snapshot_cache "
                         "so sibling arms (e.g. a 2x2/2x3 harness's OUT_ROOT/<arm>/) sharing templates and "
                         "preamble text reuse the same prereq build instead of rebuilding from scratch.")
    # ── Counterfactual evaluation scope (§76) ───────────────────────────────────
    # Names WHAT VARIES, not how fast it runs, and leaves room for further scopes
    # (e.g. affected_plus_neighbours) without another boolean.
    ev.add_argument("--counterfactual-eval-scope", choices=("full", "affected_only"),
                    default="full",
                    help="Which episodes the counterfactual arm is evaluated over. "
                         "'full' (default) re-evaluates every episode and is REQUIRED wherever "
                         "generation is stochastic, since untouched episodes then legitimately "
                         "differ run to run. 'affected_only' evaluates the episodes the anchor "
                         "can reach and copies the rest from --counterfactual-control, which is "
                         "admissible ONLY where determinism is measured -- for BFCL it is "
                         "(§66: 6/6 identical-policy pairs at 0 flips; §75.2: 253 untouched "
                         "cases, 0 flips).")
    ev.add_argument("--counterfactual-control", metavar="CONTROL_RESULTS_JSON", default=None,
                    help="Required by --counterfactual-eval-scope affected_only: the control "
                         "arm's results JSON, from which untouched outcomes are copied.")
    ev.add_argument("--counterfactual-audit-n", type=int, default=30,
                    help="With affected_only: re-run this many RANDOMLY CHOSEN copied episodes "
                         "and require 0 flips. Copying removes the check that caught the §70 "
                         "port collision (36 disagreeing cases, all untouched), so the audit "
                         "restores it at ~10%% of the eval. 0 disables it, which is NOT "
                         "recommended.")
    ev.add_argument("--allow-parallel-store", action="store_true",
                    help="REQUIRED to use --store-workers != 1. Parallel store "
                         "construction is NOT reproducible: measured at n=4, serial "
                         "replicates are byte-identical while parallel runs diverge on "
                         "all three backends with the SAME cases flipping every time "
                         "(§42). A policy-learning comparison built this way is invalid "
                         "against any serially-built number. This flag exists so the "
                         "choice is explicit and auditable, never accidental.")
    ev.add_argument("--store-workers", type=int, default=1, dest="store_workers",
                    help="W1 determinism: prereq store builds are SERIAL by default (1) — "
                         "concurrent vLLM continuous batching is not byte-stable even at "
                         "temperature 0. <=0 restores the old uncapped-parallel behavior; a "
                         "positive N caps the pool at N chains.")

    return parser


def _enforce_serial_store(args) -> None:
    """FROZEN (§42): store construction is serial for all policy-learning experiments.

    Parallelism is used only ACROSS isolated arms/servers (§35, measured at 0 flips), never
    WITHIN store construction (§42, measured at 3 reproducible flips per backend).

    Enforced here rather than documented, because the failure mode is someone speeding up a
    run and silently invalidating its comparability -- the original -3.85pp footgun. Opting
    out requires --allow-parallel-store, so it is explicit and shows up in the provenance.
    """
    w = getattr(args, "store_workers", 1)
    if w != 1 and not getattr(args, "allow_parallel_store", False):
        raise SystemExit(
            f"REFUSING --store-workers {w}: store construction must be SERIAL for "
            f"policy-learning runs (§42). Parallel builds are reproducibly different from "
            f"serial ones, so results are not comparable against any serially-built "
            f"baseline. Pass --allow-parallel-store if you genuinely intend a "
            f"non-comparable run (e.g. re-measuring the parallel floor itself)."
        )


def main():
    # Peek --model-config first (own tiny parser so required args aren't enforced yet).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model-config", default=None)
    pre_args, _ = pre.parse_known_args()

    parser = build_parser()
    cfg: Dict = {}
    if pre_args.model_config:
        cfg = load_model_config(pre_args.model_config)
        apply_model_config(parser, cfg)
    args = parser.parse_args()
    _enforce_serial_store(args)

    if not args.skip_server_setup and not args.model_path:
        parser.error("--model-path is required unless --skip-server-setup is set")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Server ────────────────────────────────────────────────────────────────
    server = None
    if not args.skip_server_setup:
        server = _VllmServer(
            model_path=args.model_path,
            port=args.port,
            num_gpus=args.num_gpus,
            gpu_mem_util=args.gpu_memory_utilization,
            max_model_len=args.max_model_len or None,
            dtype=args.dtype,
            served_model_name=args.served_model_name or None,
        )
        try:
            server.wait_ready()
        except Exception:
            server.shutdown()
            raise

    def _cleanup(signum=None, frame=None):
        if server:
            server.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    vllm_url = args.vllm_url or f"http://localhost:{args.port}"

    # ── Discover served model ─────────────────────────────────────────────────
    import requests
    served_name = args.served_model_name
    if not served_name:
        r = requests.get(f"{vllm_url}/v1/models", timeout=10)
        served_name = r.json()["data"][0]["id"]
        print(f"[init] Served model: {served_name}", flush=True)

    # ── Handler ───────────────────────────────────────────────────────────────
    # _HANDLER_MODEL_GUARD (§100): a model served under another model's parser runs to
    # completion and reports a meaningless number, because an empty tool-call parse is not an
    # error anywhere downstream. Granite 4.1 emits JSON tool calls, 4.2 emits XML, and each
    # handler returns [] on the other's format (measured). §91 lost three 4.6-hour jobs to this;
    # job 766681 then lost another 2.4 hours to the SAME defect because the §91 fix only
    # forwarded the handler when an env var happened to be set.
    #
    # Refuse the mismatch rather than guessing: silently picking a handler would repeat the
    # original error of a default applied to a case it was not written for.
    _mp = (args.model_path or "") + " " + (args.served_model_name or "")
    _is42 = ("4.2" in _mp) or ("4_2" in _mp) or ("granite42" in _mp.lower())
    _h42 = "4_2" in (args.handler_module or "") or "42" in (args.handler_class or "")
    if _is42 and not _h42:
        sys.exit(
            "HANDLER/MODEL MISMATCH (§100): model %r looks like Granite 4.2 (XML tool calls) but "
            "the handler is %s.%s, which parses 4.1 JSON and returns [] on 4.2 output. Every tool "
            "call would be silently dropped and the run would report a meaningless accuracy.\n"
            "Pass --handler-module bfcl_eval.model_handler.local_inference.granite_4_2 "
            "--handler-class Granite42PromptHandler (or Granite42ThinkingPromptHandler), and the "
            "matching --registry-name."
            % (_mp.strip(), args.handler_module, args.handler_class))
    if _h42 and not _is42:
        sys.exit(
            "HANDLER/MODEL MISMATCH (§100): handler %s.%s is for Granite 4.2 (XML) but model %r "
            "does not look like 4.2. A 4.1 checkpoint's JSON tool calls would be dropped."
            % (args.handler_module, args.handler_class, _mp.strip()))
    # _REGISTRY_MODEL_GUARD (§100.5): --registry-name defaults INDEPENDENTLY of the handler,
    # so a 4.2 job can clear the handler guard above and still load 4.1's prompt template. The
    # handler and the registry are two halves of the same mismatch; guarding one is not enough.
    _reg = args.registry_name or ""
    if _is42 and _reg and "4.2" not in _reg and "4_2" not in _reg:
        sys.exit(
            "REGISTRY/MODEL MISMATCH (§100.5): model looks like Granite 4.2 but --registry-name "
            "is %r, a 4.1 entry. The prompt template would not match the served model. Pass a 4.2 "
            "registry entry, e.g. "
            "'ibm-research/granite-4.2-8b-prerelease-r260622a-thinking-off' (or "
            "'…-thinking-truncated' with Granite42ThinkingPromptHandler)." % _reg)
    if _is42 and not _reg:
        sys.exit(
            "REGISTRY REQUIRED (§100.5): model looks like Granite 4.2, so --registry-name must be "
            "given explicitly -- the default is a 4.1 entry and would silently load the wrong "
            "prompt template. Use "
            "'ibm-research/granite-4.2-8b-prerelease-r260622a-thinking-off' for thinking-off.")
    print("[init] handler: %s.%s   registry: %s"
          % (args.handler_module, args.handler_class, args.registry_name or "<default>"),
          flush=True)
    mod = __import__(args.handler_module, fromlist=[args.handler_class])
    HandlerCls = getattr(mod, args.handler_class)
    try:
        handler = HandlerCls(
            model_name=served_name,
            temperature=0.001,
            registry_name=args.registry_name,
            is_fc_model=False,
        )
    except TypeError:
        handler = HandlerCls(model_name=served_name, temperature=0.001)

    # ── Evaluator ─────────────────────────────────────────────────────────────
    from anchoropt.memory_evaluator import MemoryAnchorOptEvaluator
    # Resolution order: explicit flag > ANCHOROPT_SNAPSHOT_CACHE > per-out-root default.
    # The env var is read HERE and not only in the submit scripts because every runner
    # otherwise has to remember to forward it, and one that forgets silently rebuilds the
    # entire prereq store -- 87-90% of arm wallclock (SS69.1) -- while reporting a
    # perfectly ordinary cache MISS. That is exactly what the first T1/R2 arm did.
    _env_cache = (_os.environ.get("ANCHOROPT_SNAPSHOT_CACHE") or "").strip()
    if args.snapshot_cache_dir:
        snapshot_cache_dir = Path(args.snapshot_cache_dir)
    elif _env_cache:
        snapshot_cache_dir = Path(_env_cache)
    else:
        snapshot_cache_dir = out_dir.parent / ".snapshot_cache"
    evaluator = MemoryAnchorOptEvaluator(
        handler=handler,
        vllm_url=vllm_url,
        served_model_name=served_name,
        max_steps_per_turn=args.max_steps,
        seed=args.seed,  # W1: fixed decode seed, matched to training
        disable_gates=args.disable_gates,
        enable_reroute=args.enable_reroute,
        enable_forced_retrieval=args.enable_forced_retrieval,
        enable_forced_key_search=args.enable_forced_key_search,
        snapshot_cache_dir=snapshot_cache_dir,
        store_workers=args.store_workers,
        # Decode params from --model-config (defaults reproduce the inline literals).
        **decode_params(cfg),
    )

    # ── Load templates ────────────────────────────────────────────────────────
    with open(args.templates) as f:
        templates = json.load(f)
    print(f"[init] Templates: {args.templates}", flush=True)

    # Transfer-contract fix: opt-in remedy flags (--enable-reroute,
    # --enable-forced-retrieval, --enable-forced-key-search) are baked directly
    # into the in-memory `templates` dict that drives THIS run's own
    # build_snapshot_store/_eval_batch calls below — not just a side file.
    # Previously the CLI flag only reached this script's own eval via the
    # evaluator constructor's default-parameter fallback inside remedy_enabled(),
    # while the exported policy_overrides.json was a separate dict built purely
    # for a downstream leaderboard run; the two could silently diverge (e.g. a
    # candidate scored here as "reroute on" would export as reroute-absent). An
    # explicit key in the policy already wins over any default, so this only
    # fills in a key the policy doesn't set. Still never mutates --templates
    # on disk (the input artifact, e.g. templates_best.json, is never modified
    # as a side effect of running an eval) — only the in-memory copy, plus one
    # merged sibling file so a downstream leaderboard run (ANCHOROPT_POLICY_PATH
    # pointed at that JSON) picks up the exact same policy via base_handler.py's
    # load_anchoropt_policy().get(...).
    _overrides = {}
    if args.enable_reroute and "enable_reroute" not in templates:
        _overrides["enable_reroute"] = True
    if args.enable_forced_retrieval and "enable_forced_retrieval" not in templates:
        _overrides["enable_forced_retrieval"] = True
    if args.enable_forced_key_search and "enable_forced_key_search" not in templates:
        _overrides["enable_forced_key_search"] = True
    if _overrides:
        templates.update(_overrides)
        policy_path = out_dir / "policy_overrides.json"
        with open(policy_path, "w") as f:
            json.dump(templates, f, indent=2)
        print(f"[init] {_overrides} — baked into in-process eval and written to {policy_path}", flush=True)

    # ── Load cases ────────────────────────────────────────────────────────────
    if args.cases:
        cases_path = Path(args.cases)
        if not cases_path.exists():
            print(f"[error] Cases file not found: {cases_path}", flush=True)
            sys.exit(1)
        with open(cases_path) as f:
            cases = json.load(f)
        print(f"[init] Cases ({args.split}): {len(cases)} entries  ({cases_path})", flush=True)
    elif args.split == "all":
        cases = []
        for s in ("train", "val", "test"):
            p = Path(args.cases_dir) / f"memory_{s}_cases.json"
            if not p.exists():
                print(f"[error] Cases file not found: {p}", flush=True)
                sys.exit(1)
            with open(p) as f:
                split_cases = json.load(f)
            queries = split_queries(split_cases)
            cases.extend(queries)
            print(f"[init]   {s:5s}: {len(queries)} queries", flush=True)
        print(f"[init] Cases (all): {len(cases)} queries total (train+val+test)", flush=True)
    else:
        cases_path = Path(args.cases_dir) / f"memory_{args.split}_cases.json"
        if not cases_path.exists():
            print(f"[error] Cases file not found: {cases_path}", flush=True)
            sys.exit(1)
        with open(cases_path) as f:
            cases = json.load(f)
        print(f"[init] Cases ({args.split}): {len(cases)} entries  ({cases_path})", flush=True)

    # ── Build snapshot store from this split's own prereqs ─────────────────────
    # Each split file is self-contained: train/val share the same 96 prereq ids
    # (same underlying scenario chains, different queries layered on top), and
    # test's 15 prereqs are its own fully-disjoint chains — confirmed via direct
    # id/chain inspection. So scoring a single split only ever needs that split's
    # own prereq cases, not a union across train+val+test (the old behavior,
    # which rebuilt all 111 prereqs for a 15-prereq test run). --split all is the
    # one case that legitimately needs the union, since it scores every split's
    # queries. W2 fidelity: seed with the SAME templates we score with.
    if args.split == "all":
        all_prereqs = prereq_union(args.cases_dir)
    else:
        all_prereqs = split_prereqs(cases)
    if all_prereqs:
        print(f"\n[snapshot_store] Running {len(all_prereqs)} prereqs ({args.split}) ...", flush=True)
        evaluator.build_snapshot_store(all_prereqs, templates)
    else:
        print(f"[warn] No prereqs found in {args.cases_dir} — memory may start empty", flush=True)

    # ── Run eval ──────────────────────────────────────────────────────────────
    try:
        print(f"\n[eval] Running {args.split}-set evaluation ...", flush=True)
        # Queries-only: the store was already built above from this split's own
        # prereqs (or the union for --split all), and query state is read from disk,
        # so re-running prereq episodes here is redundant serial work. Guarded on the
        # cache dir so the no-cache path keeps the historical full-list behavior.
        # (--split all already loaded queries-only; this is a no-op there.)
        if evaluator._snapshot_cache_dir is not None:
            cases = split_queries(cases)
        acc, results = _eval_batch(cases, templates, evaluator, tag_prefix=args.split, max_workers=args.workers)
        cat_acc = _per_category_accuracy(results)

        print(f"\n[eval] === Results ({args.split}) ===", flush=True)
        print(f"[eval] Overall accuracy: {acc:.1%}", flush=True)
        for cat, cat_a in cat_acc.items():
            print(f"[eval]   {cat:10s}: {cat_a:.1%}", flush=True)

        # ── PROVENANCE (§41) ─────────────────────────────────────────────────
        # Recorded so BASELINE EQUIVALENCE is a mechanical check rather than an
        # argument. Before this block, a results file carried only accuracy, split,
        # templates and three CLI flags -- so five of the seven equivalence criteria
        # (seed, model, temperature, store-workers, prefix-caching) could not be
        # verified from the corpus at all, only from the submit script that produced
        # it. That is what forced a redundant no-op arm for G3.
        #
        # Two disciplines encoded here:
        #   * EFFECTIVE policy, not the CLI flag (§24.2). Metadata recorded
        #     `enable_reroute=False` for a run whose policy set it true, because the
        #     flag is applied only when the key is ABSENT from the policy. Reading the
        #     flag alone tells you what was typed, not what ran.
        #   * The gate-config fingerprint, which folds in a code-version salt, so a
        #     cross-commit corpus is detectable rather than silently comparable.
        _prov = {
            "seed": getattr(args, "seed", None),
            "model_path": str(getattr(args, "model_path", "") or ""),
            "handler_module": args.handler_module,
            "handler_class": args.handler_class,
            "registry_name": args.registry_name,
            "served_model_name": str(getattr(args, "served_model_name", "") or ""),
            # Hardcoded in this script (see the handler construction above), not a CLI
            # arg -- recorded as the literal so a future change is visible in the corpus
            # rather than inferred. A getattr(args, "temperature") would have written
            # None here, which reads as "unknown" when the value is in fact fixed.
            "temperature": 0.001,
            "max_model_len": getattr(args, "max_model_len", None),
            "store_workers": getattr(args, "store_workers", None),
            "allow_parallel_store": bool(getattr(args, "allow_parallel_store", False)),
            # NOT KNOWABLE HERE: prefix caching is a vLLM SERVER flag set by the submit
            # script (--no-enable-prefix-caching), so this process cannot observe it.
            # Recorded as an explicit unknown rather than a default, so a baseline audit
            # reports INDETERMINATE instead of silently assuming a value.
            "enable_prefix_caching": "unknown_server_side",
            "server_url": str(getattr(args, "vllm_url", "") or ""),
            "n_scored_cases": len([r for r in results if not r.get("is_prereq")]),
        }
        # ARM IDENTITY. `gate_config_fingerprint` below covers the GATE registry only, so two runs
        # installing different controller SPECS were byte-identical in provenance (observed: the
        # +2.25pp pair, whose two runs differ only in their traj sidecars). Hash the templates
        # artifact itself, and record the repo commit, so a results file names the arm it ran.
        try:
            import hashlib as _hl
            import subprocess as _sp
            _tpath = pathlib.Path(str(args.templates))
            _prov["templates_sha256"] = (
                _hl.sha256(_tpath.read_bytes()).hexdigest()[:16] if _tpath.is_file() else None)
            _prov["git_sha"] = _sp.run(
                ["git", "-C", str(pathlib.Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10).stdout.strip() or None
        except Exception as _e:      # never fail a completed eval over provenance
            _prov["arm_identity_error"] = f"{type(_e).__name__}: {_e}"
        try:
            from anchoropt.memory_gates import (
                gate_config_fingerprint as _gcf,
                enabled_explicit_gate_keys as _eegk,
                disabled_gate_keys as _dgk,
                remedy_enabled as _re,
            )
            _prov["gate_config_fingerprint"] = _gcf(templates)
            _prov["explicit_gates_installed"] = list(_eegk(templates))
            _prov["gates_disabled"] = list(_dgk(templates))
            # EFFECTIVE remedy values -- what actually ran.
            _prov["effective_remedies"] = {
                f: bool(_re(templates, f)) for f in
                ("enable_reroute", "enable_forced_retrieval", "enable_forced_key_search")
            }
        except Exception as _e:      # never fail a completed eval over provenance
            _prov["provenance_error"] = f"{type(_e).__name__}: {_e}"

        out = {
            "split": args.split,
            "templates": str(args.templates),
            "disable_gates": args.disable_gates,
            # CLI flags, kept for backward compatibility. These are what was TYPED;
            # provenance.effective_remedies is what RAN.
            "enable_reroute": args.enable_reroute,
            "enable_forced_retrieval": args.enable_forced_retrieval,
            "enable_forced_key_search": args.enable_forced_key_search,
            "provenance": _prov,
            "accuracy": acc,
            "per_category": cat_acc,
            "results": results,
        }
        out_path = out_dir / f"eval_{args.split}_results.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[eval] Saved to {out_path}", flush=True)

    finally:
        if server:
            server.shutdown()


if __name__ == "__main__":
    main()
