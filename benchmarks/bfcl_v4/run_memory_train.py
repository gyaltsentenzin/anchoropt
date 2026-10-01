"""
AnchorOpt memory training — one-go script.

Combines:
  1. vLLM server lifecycle (optional launch + teardown)
  2. Memory train/val case loading
  3. Hill-climb template optimization loop
     - Batch evaluation on train cases with MemoryAnchorOptEvaluator
     - LLMOptimizer (Claude via LiteLLM) proposes one template edit per batch
     - Simple accuracy gate: accept if val score improves (no LLMaJ / verbatim check)
  4. Results + optimized templates written to --out-dir

No LLMaJ verbatim gate — memory grading uses agentic_checker (deterministic
word-boundary substring match). The gate is purely accuracy-based.

Usage:
    python scripts/run_memory_train.py \\
        --model-path /path/to/granite-4.1-8b \\
        --served-model-name granite41-8b \\
        --cases-dir  data/ \\
        --out-dir    results/memory_train_run1 \\
        --batches 20 --batch-size 12

    # Skip server launch if already running:
    python scripts/run_memory_train.py \\
        --skip-server-setup --vllm-url http://localhost:8080 \\
        ...

Environment:
    OPTIMIZER_LITELLM_MODEL  e.g. "anthropic/claude-opus-4-8"
    LITELLM_API_BASE / LITELLM_API_KEY  (optional proxy)
"""

import argparse
import difflib
import json
import math
import os
import re
import random
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
_detail_log = None  # file handle for per-case OK/FAIL lines; set in main()

_SCRIPT_DIR     = Path(__file__).resolve().parent
_ANCHOROPT_ROOT = _SCRIPT_DIR.parent
_BFCL_ROOT      = _ANCHOROPT_ROOT.parent / "berkeley-function-call-leaderboard"

for _p in [str(_ANCHOROPT_ROOT), str(_BFCL_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from anchoropt.eval_common import (  # noqa: E402
    load_model_config,
    apply_model_config,
    decode_params,
    split_prereqs,
    split_queries,
    prereq_union,
)
from anchoropt.teacher_client import make_teacher  # noqa: E402

# ── Load .env (API keys for LiteLLM/Anthropic teacher) ───────────────────────
_ENV_FILE = _ANCHOROPT_ROOT.parent / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())


# ── vLLM server lifecycle ─────────────────────────────────────────────────────

_VLLM_BIN = os.environ.get("VLLM_BIN") or shutil.which("vllm")


class _VllmServer:
    def __init__(self, model_path, port, num_gpus, gpu_mem_util, max_model_len, dtype,
                 served_model_name: Optional[str] = None):
        import requests
        if not _VLLM_BIN:
            raise RuntimeError(
                "vllm not found on PATH and VLLM_BIN is not set. Install vllm in this "
                "environment or export VLLM_BIN=/path/to/vllm.")
        cmd = [
            _VLLM_BIN, "serve", model_path,
            "--port", str(port),
            "--dtype", dtype,
            "--tensor-parallel-size", str(num_gpus),
            "--gpu-memory-utilization", str(gpu_mem_util),
            "--trust-remote-code",
            # Noise-floor measurement (2026-08-04): with --store-workers 1, automatic
            # prefix caching was the entire remaining f_null (11 -> 0 flips when off).
            # See anchoropt/results/08_methodology/noise_floor_granite41-8b_noprefixcache/.
            "--no-enable-prefix-caching",
        ]
        if max_model_len:
            cmd += ["--max-model-len", str(max_model_len)]
        if served_model_name:
            cmd += ["--served-model-name", served_model_name]

        print(f"[server] Launching: {' '.join(cmd)}", flush=True)
        self._stop_event = threading.Event()
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for pipe in (self._process.stdout, self._process.stderr):
            t = threading.Thread(
                target=self._log, args=(pipe,), daemon=True
            )
            t.start()
        self._base_url = f"http://localhost:{port}/v1"
        self._requests = requests

    def _log(self, pipe):
        for line in iter(pipe.readline, ""):
            if not self._stop_event.is_set():
                print(line, end="", flush=True)

    def wait_ready(self, timeout=3600):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with code {self._process.returncode}"
                )
            try:
                r = self._requests.get(f"{self._base_url}/models", timeout=5)
                if r.status_code == 200:
                    self._stop_event.set()
                    print("[server] Ready.", flush=True)
                    return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f"vLLM not ready after {timeout}s")

    def shutdown(self):
        if self._process and self._process.poll() is None:
            print("[server] Shutting down ...", flush=True)
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _eval_batch(
    cases: List[Dict],
    templates: Dict,
    evaluator,
    tag_prefix: str = "train",
    max_workers: int = 1,
) -> Tuple[float, List[Dict]]:
    """Run cases through the evaluator and return (accuracy, per_entry_results).

    Skips prereq entries (they're run for state setup but not counted in accuracy).
    """
    from anchoropt.memory_gates import telemetry_flags
    gate_flags = telemetry_flags()
    results = []

    def _run(i, case):
        is_prereq = "prereq" in case["id"]
        try:
            checker, traj = evaluator.evaluate_episode(
                case, templates, rollout_tag=f"{tag_prefix}_{i}"
            )
        except Exception as e:
            import traceback
            # W3: distinguish transient INFRA failures (vLLM down / timeout / 5xx /
            # connection reset) from genuine task/code failures. Infra failures are
            # noise that must NOT be scored as wrong answers — a transient error
            # during a gate eval could otherwise flip an accept/reject decision.
            # We flag them so the accuracy denominator can exclude them below.
            _ename = type(e).__name__
            _emsg = str(e)
            _infra = (
                "Timeout" in _ename
                or "Connection" in _ename
                or _emsg.startswith("vLLM 5")
                or "timed out" in _emsg.lower()
                or "connection" in _emsg.lower()
            )
            print(f"  {'INFRA' if _infra else 'ERR'} {case['id'][:60]}: {e}", flush=True)
            if not _infra:
                traceback.print_exc()
            return {
                "id": case["id"],
                "valid": False,
                "is_prereq": is_prereq,
                "infra_error": _infra,
                "error": str(e),
            }
        # Stage 0 sidecar: persist the full step list before it is reduced to the
        # booleans below. No-op unless ANCHOROPT_TRAJ_DIR is set.
        try:
            from anchoropt.traj_sidecar import dump_episode
            dump_episode(case, traj,
                         phase="prereq" if is_prereq else "query",
                         checker=checker)
        except Exception:
            pass

        # Scan trajectory for failure-classification signals
        g1_fired = any(s.get("g1_gate") for s in traj)
        d3_fired = any(s.get("d3_gate") for s in traj)
        # Registry-derived: any gate/remedy with a non-empty telemetry_flag (g1_gate,
        # d3_gate, g3_gate, g4_gate, g5_gate, blob_pressure_gate, reroute_gate, ...)
        # fires here for free as new gates are added to memory_gates.py.
        gates_fired = {flag: any(s.get(flag) for s in traj) for flag in gate_flags}
        # T1: which SOFT signal keys actually injected text this episode. The gates
        # above carry dedicated step_record flags (telemetry_flag); soft signals
        # don't, so their firing was previously invisible to the teacher. injection_key
        # is set per step by memory_evaluator only when a template actually injected
        # (None = silent), so this is the soft-signal analogue of gates_fired.
        signals_fired = sorted({
            s.get("injection_key") for s in traj if s.get("injection_key")
        })
        archival_searched = any(
            "archival_memory" in str(s.get("decoded", "")).lower() for s in traj
        )
        retrieval_succeeded = any(s.get("is_retrieval_success") for s in traj)
        retrieval_attempts = sum(
            1 for s in traj
            if any(t in str(s.get("decoded", ""))
                   for t in ("archival_memory_retrieve", "archival_memory_key_search",
                              "core_memory_retrieve", "memory_retrieve"))
        )
        return {
            "id": case["id"],
            "valid": checker.get("valid"),
            "is_prereq": is_prereq,
            "final_answer": checker.get("final_answer", ""),
            "error_type": checker.get("error_type", ""),
            "steps": len(traj),
            "g1_gate_fired": g1_fired,
            "d3_gate_fired": d3_fired,
            "gates_fired": gates_fired,
            "signals_fired": signals_fired,
            "archival_searched": archival_searched,
            "retrieval_succeeded": retrieval_succeeded,
            "retrieval_attempts": retrieval_attempts,
            "force_quit": bool(checker.get("force_quit", False)),
        }

    def _log_case(line: str):
        if _detail_log is not None:
            _detail_log.write(line + "\n")
            _detail_log.flush()
        else:
            print(line, flush=True)

    if max_workers <= 1:
        for i, case in enumerate(cases):
            r = _run(i, case)
            results.append(r)
            if not r["is_prereq"]:
                prefix = "OK" if r.get("valid") else "  "
                _log_case(f"  {prefix} {r['id'][:60]}")
    else:
        _print_lock = threading.Lock()

        def _run_and_report(i, case):
            r = _run(i, case)
            if not r["is_prereq"]:
                prefix = "OK" if r.get("valid") else "  "
                with _print_lock:
                    _log_case(f"  {prefix} {r['id'][:60]}")
            return r

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_run_and_report, i, c): i for i, c in enumerate(cases)}
            for fut in as_completed(futs):
                results.append(fut.result())

    query_results = [r for r in results if not r["is_prereq"]]
    # W3: exclude transient infra failures from the accuracy denominator so noise
    # is not scored as wrong. Genuine task failures (valid is False, no infra flag)
    # still count against accuracy as before.
    scored = [r for r in query_results if not r.get("infra_error")]
    n_infra = len(query_results) - len(scored)
    n_ok = sum(1 for r in scored if r.get("valid") is True)
    accuracy = n_ok / len(scored) if scored else 0.0
    if n_infra:
        rate = n_infra / len(query_results) if query_results else 0.0
        print(
            f"  [infra] {n_infra}/{len(query_results)} queries hit transient infra errors "
            f"({rate:.0%}) and were EXCLUDED from accuracy — scored over {len(scored)}. "
            f"If this rate is high, treat the gate result as unreliable and re-run.",
            flush=True,
        )
    return accuracy, results


def _score_split(evaluator, prereqs, cases, templates, tag, workers):
    """Score a split's QUERIES against a prebuilt on-disk snapshot store.

    When the evaluator has a snapshot cache dir, (re)build the store for
    `templates` from `prereqs` — a cheap cache HIT unless a storage-affecting
    template text changed, in which case the store is rebuilt so the score is
    measured against the correct stored state — then evaluate the split's
    QUERIES ONLY. Prereq episodes are redundant at score time (query state is
    read from disk) and are the expensive serial episodes, so skipping them is
    the ~2x per-eval win that makes a full serial run fit in walltime.

    When the evaluator has NO cache dir (_snapshot_cache_dir is None), this
    falls through to the historical full-list path — prereqs run inline and
    _eval_batch excludes them from scoring — so null/no-cache behavior is
    byte-identical to before this optimization.
    """
    if evaluator._snapshot_cache_dir is not None:
        evaluator.build_snapshot_store(prereqs, templates)
        cases = split_queries(cases)
    return _eval_batch(cases, templates, evaluator, tag_prefix=tag, max_workers=workers)


def _telemetry_rollup(results: List[Dict]) -> Dict[str, Dict]:
    """Per-flag/-signal fired/helped/hurt/convert_rate over query results.

    Covers TWO layers in one uniform table:
      - GATES: deterministic gates keyed by their telemetry_flag (g1_gate, …),
        read from each result's ``gates_fired`` dict {flag: bool}.
      - SOFT SIGNALS (T1): the tunable templates the teacher actually edits,
        keyed by their template key (on_retrieval_success_pre_answer, …), read
        from each result's ``signals_fired`` list. These carry NO telemetry_flag
        so they were previously invisible to the teacher — it was optimizing the
        soft layer blind. Each entry is tagged ``kind`` ("gate"|"signal").

    "helped" = fired on a case ultimately scored valid; "hurt" = fired on an
    invalid case. convert_rate = helped / fired. ``by_backend`` breaks the same
    counts down per kv/vector/rec_sum so the teacher can see a signal that helps
    one backend while poisoning another (e.g. on_retrieval_success_pre_answer
    converting worse on rec_sum than silence — the exact regression this exposes).
    """
    def _new():
        return {
            "kind": "", "fired": 0, "helped": 0, "hurt": 0,
            "by_backend": defaultdict(lambda: {"fired": 0, "helped": 0, "hurt": 0}),
        }
    rollup: Dict[str, Dict] = defaultdict(_new)

    def _bump(name: str, kind: str, valid: bool, backend: str):
        entry = rollup[name]
        entry["kind"] = kind
        entry["fired"] += 1
        entry["by_backend"][backend]["fired"] += 1
        bucket = "helped" if valid else "hurt"
        entry[bucket] += 1
        entry["by_backend"][backend][bucket] += 1

    for r in results:
        if r.get("is_prereq") or r.get("infra_error"):
            continue
        valid = r.get("valid") is True
        backend = _result_cat(r)
        for flag, fired in r.get("gates_fired", {}).items():
            if fired:
                _bump(flag, "gate", valid, backend)
        for key in r.get("signals_fired", []):
            _bump(key, "signal", valid, backend)

    for stats in rollup.values():
        stats["convert_rate"] = stats["helped"] / stats["fired"] if stats["fired"] else 0.0
        by_backend = {}
        for bk, d in stats["by_backend"].items():
            d["convert_rate"] = d["helped"] / d["fired"] if d["fired"] else 0.0
            by_backend[bk] = d
        stats["by_backend"] = by_backend
    return dict(rollup)


# ── T5: noise-robust acceptance helpers ──────────────────────────────────────

def _correct_vec(results: List[Dict]) -> Dict[str, bool]:
    """{case_id: bool} over scored (non-prereq, non-infra) query results.

    The per-case correctness vector the bootstrap resamples. Keyed by id so the
    incumbent and candidate vectors can be paired on the SAME cases even if a
    transient infra error drops one from either pass.
    """
    return {
        r["id"]: (r.get("valid") is True)
        for r in results
        if not r.get("is_prereq") and not r.get("infra_error")
    }


def _bootstrap_delta_lower_ci(
    incumbent: Dict[str, bool],
    candidate: Dict[str, bool],
    n_boot: int,
    alpha: float,
    rng: random.Random,
) -> Tuple[float, float, int]:
    """Paired case-bootstrap of the val delta (candidate − incumbent).

    Eval is deterministic (temp≈0), so per-case correctness is fixed; the variance
    that matters is which cases happen to be in the small val set. Resampling cases
    with replacement estimates how robust the delta is to val composition — the
    winner's-curse guard against accepting a 1–2-case fluke. PAIRED: each resample
    uses the same case indices for both arms so per-case win/loss correlation is kept.

    Vectorized with numpy (one (n_boot, n) integer draw) rather than a pure-Python
    per-resample loop: at the boot_n~50k scale needed for a Bonferroni-corrected
    alpha to have tail resolution, the Python loop's per-element rng.randrange call
    dominates; numpy draws all indices in one call. The numpy Generator is reseeded
    from `rng` (stdlib) so the caller's single `random.Random` stream still
    deterministically drives every batch's CI (needed for --resume reproducibility)
    without exposing numpy's internal RNG as a second checkpointed stream.

    Returns (observed_delta, lower_ci, n_paired). lower_ci is the one-sided lower
    bound at `alpha` (e.g. alpha=0.05 → 5th percentile). n_paired<... → CI is (−1,).
    """
    ids = [i for i in incumbent if i in candidate]
    n = len(ids)
    if n == 0:
        return 0.0, -1.0, 0
    inc = np.array([1.0 if incumbent[i] else 0.0 for i in ids])
    cand = np.array([1.0 if candidate[i] else 0.0 for i in ids])
    observed = float(cand.mean() - inc.mean())
    if n_boot <= 0:
        return observed, observed, n
    npy_rng = np.random.default_rng(rng.randrange(2**32))
    resample_idx = npy_rng.integers(0, n, size=(n_boot, n))
    deltas = np.sort(cand[resample_idx].mean(axis=1) - inc[resample_idx].mean(axis=1))
    idx = max(0, min(len(deltas) - 1, int(alpha * len(deltas))))
    return observed, float(deltas[idx]), n


# (2026-08-06) SkillOpt-style simplification: the strict regime (magnitude AND
# boot_lower>0) is kept as "bootstrap", opt-in, exactly as it worked before. The new
# default "simple" is greedy candidate-beats-incumbent (magnitude alone), matching
# SkillOpt's `score(candidate) > score(current)` accept rule, guarded by the same
# integer noise floor. See _accept_decision's docstring for the measured cost of
# this default: replaying run12's 47 CI-bearing decisions, 21 flip from reject to
# accept, including the RCA counterexample below tied to a real -9.33pp held-out
# regression. Restore old behavior with --gate-mode bootstrap.
_GATE_MODES = ("simple", "bootstrap")


def _accept_decision(
    delta: float,
    boot_lower: Optional[float],
    n_scored: int,
    min_effect: float,
    null_churn_cases: float,
    gate_mode: str,
) -> Tuple[bool, str, int, float]:
    """T8/(2026-08-06): the single accept rule, shared by both gate_mode regimes.

    Two questions, each owned by exactly one term — RCA 2026-08-04 found the prior
    rule asked both of ONE term and made the gate unreachable even at zero noise:
      - "is this worth adopting?" -> an INTEGER flip-count magnitude test. Owns the
        noise floor via `null_churn_cases` (so a 1-2 case fluke can't clear it) and
        is otherwise noise-free.
      - "is this real, i.e. would it survive out-of-sample?" -> `boot_lower > 0`, a
        SIGN check at the caller's (Bonferroni-corrected) alpha — not a second
        magnitude bar. Requiring boot_lower > min_effect double-counts the same
        noise the CI already widens for: full_pf1's one historically-successful,
        test-confirmed accept (net +11 of 78 val cases, zero measurement noise)
        fails `boot_lower > 3.85%` but passes `boot_lower > 0`.

    `net_flips = round(delta * n_scored)` replaces a bare float `delta > 3/n_scored`
    comparison — the float form is a coin flip on floating-point rounding for a
    delta of exactly 3 cases (observed: 3 different bit-patterns for "3/78" produced
    3 different verdicts across gemma/granite/qwen). `round()` makes the boundary a
    single well-defined integer instead.

    gate_mode has NO default — deliberately. This is the SkillOpt/bootstrap fork
    point; any relaxation of boot_lower must land in the same commit as its test
    coverage (see the RCA counterexample test), and a silent default would let a
    call site inherit a regime by omission instead of choosing one.
      - "simple": accept iff the magnitude test passes. boot_lower is ignored (it
        may still be computed by the caller and recorded as a diagnostic, but does
        not gate). This is SkillOpt's greedy candidate>incumbent rule.
      - "bootstrap": today's rule, verbatim — magnitude AND boot_lower>0.
    boot_lower is None in the gate_n_seeds>1 (no-CI) regime OR whenever the caller
    didn't bootstrap; the decision there is the magnitude test alone regardless of
    gate_mode.

    Returns (accepted, reason, net_flips, min_flips) — the caller needs net_flips/
    min_flips for logging and history without recomputing round()/ceil() a second
    time (the exact float-drift class this function's T8 predecessor exists to
    prevent). reason is empty on accept and names every failing clause on reject.
    """
    if gate_mode not in _GATE_MODES:
        raise ValueError(f"gate_mode must be one of {_GATE_MODES}, got {gate_mode!r}")
    n_scored = max(int(n_scored), 1)
    net_flips = round(delta * n_scored)
    min_flips = max(math.ceil(min_effect * n_scored), null_churn_cases)
    magnitude_ok = net_flips >= min_flips
    if gate_mode == "simple" or boot_lower is None:
        reason = "" if magnitude_ok else f"net_flips {net_flips} < min_flips {min_flips:g}"
        return magnitude_ok, reason, net_flips, min_flips
    ci_ok = boot_lower > 0.0
    if magnitude_ok and ci_ok:
        return True, "", net_flips, min_flips
    reasons = []
    if not magnitude_ok:
        reasons.append(f"net_flips {net_flips} < min_flips {min_flips:g}")
    if not ci_ok:
        reasons.append(f"boot_lower {boot_lower:+.2%} <= 0")
    return False, " and ".join(reasons), net_flips, min_flips


def _is_dup_proposal(
    history: List[Dict], key: str, new_text: str, threshold: float = 0.8, window: int = 8
) -> Tuple[bool, float]:
    """True if new_text is >threshold similar to a recently REJECTED edit for the SAME key.

    Stops the teacher from burning a gate eval re-proposing a near-identical edit it
    already rejected. Only compares within the same key (a shared phrase across
    different keys is legitimately different behavior). Uses difflib ratio (no
    similarity infra existed to reuse).
    """
    norm = " ".join((new_text or "").split())
    best = 0.0
    recent = [h for h in history if h.get("edit_key") == key and h.get("accepted") is False]
    for h in recent[-window:]:
        prev = " ".join((h.get("edit_text") or "").split())
        if not prev:
            continue
        ratio = difflib.SequenceMatcher(None, norm, prev).ratio()
        best = max(best, ratio)
    return best > threshold, best


def _stuck_batches(history: List[Dict]) -> int:
    """Consecutive most-recent batches since the last ACCEPTED edit.

    Every history entry already carries "accepted" (True/False) on every existing
    code path, so this needs no new state and naturally resets to 0 the batch after
    an accept. Used only to shape the teacher prompt (T7) — never gates anything.
    """
    n = 0
    for h in reversed(history):
        if h.get("accepted") is True:
            break
        n += 1
    return n


def _propose_gate_mask_edit(
    batch_idx: int,
    history: List[Dict],
    key: str = "on_premature_idk",
    probe_batch: int = 0,
    policy: Optional[Dict] = None,
) -> Optional[Tuple[str, bool]]:
    """Deterministic one-shot probe: propose flipping gate `key` away from its
    current incumbent value, once ever.

    The mask has exactly two arms and the incumbent already occupies one, so there
    is nothing for the teacher to search — this fires at `probe_batch` and never
    again, regardless of accept/reject, so `--resume` can't re-fire it. Draws no
    randomness so a probe run samples the same train batches as a non-probe run.

    `policy` defaults to None (empty dict), under which every key's incumbent is
    enabled (today's universal fail-open default), so the proposal is `False` —
    identical to this function's behavior before the `gate_default` polarity marker
    existed. Once a policy can start with a key already disabled (`gate_default:
    false`, or an explicit `gate_enabled` section), hardcoding `False` would silently
    propose the incumbent's own value — a no-op arm that burns a whole val pass to
    measure exactly zero. Reading the incumbent via `gate_enabled` and negating it
    keeps this in sync with `_arm_current_value`'s "negation of the current
    effective value" rule, the same toggle-away-from-incumbent logic the modern
    --probe-gates arms already use.

    SUPERSEDED by --probe-gates / _propose_gate_probe_edit below, which cycles the
    whole gate-space arm queue across a run instead of firing one arm once. Kept
    because --probe-gate-mask is still a documented flag and this is the exact
    degenerate case of it (one arm, one batch); prefer --probe-gates for new runs.
    """
    if batch_idx != probe_batch:
        return None
    edit_key = f"gate_enabled:{key}"
    if any(h.get("edit_key") == edit_key for h in history):
        return None
    from anchoropt.memory_gates import gate_enabled
    incumbent = gate_enabled(policy or {}, key)
    return key, not incumbent


# ── Gate-space search: enumerated single-lever arms ───────────────────────────
# Motivation (2026-08-03): the held-out 2x2 ablation showed the gates-vs-no-gates
# gap dwarfs every template-edit effect on all three models (Granite +26.6pp,
# Gemma +9.3pp, Qwen +8pp gates; templates -9.33pp / ~0 / ~0). But the only
# gate-side lever training could touch was a ONE-SHOT single-key mask probe that
# never once fired in a real run (probe_gate_mask: false in all three manifests).
# This makes the gate space a first-class, repeatedly-visited part of the search:
# a fixed, ordered queue of single-lever flips, one per scheduled batch, each
# scored through the SAME accept gate (delta > band AND bootstrap lower bound >
# band, at the Bonferroni-corrected alpha) that template edits ride.
#
# Why an enumerated queue rather than a teacher-proposed candidate type: the levers
# are booleans or short ordered value lists. There is no text for an LLM to author, no
# free-text failure mode to guard against, and enumerating is exhaustive where a
# teacher would only sample. The teacher keeps its full per-batch budget for the text
# it CAN write.
#
# Phase 7: an axis may now be CATEGORICAL (>2 candidate values) as well as boolean.
# A categorical axis occupies one scheduled batch PER candidate value, in the order
# listed in _GATE_ARM_VALUES, and each value is scored through the same accept gate as
# everything else — so the axis is a greedy coordinate search: the winner of each
# comparison becomes the incumbent the NEXT value is measured against. That is exactly
# how the boolean arms already behaved (each accept/reject updates the incumbent before
# the next arm is enumerated); the only change is that an axis can have more than one
# candidate to work through. Because arms are re-enumerated every batch against the
# live incumbent, an accepted value is dropped from its own axis's remaining candidates
# automatically (a value equal to the incumbent is not a change, so it is not an arm).
#
# Arm order = prior-evidence order, so a run that dies early still spends its arms on
# the most informative levers:
#   1. enable_forced_retrieval — best prior. The only lever that ADDS a capability with
#      no evidence against it, and it targets the documented dominant Gemma/Qwen
#      failure (give up without ever searching archival), for which G1/G3 are inert.
#   2-5. the mask ablations, in rough fire-frequency order (D3, G1, G3, G4). Weak
#      priors: the 2x2 says the gates are what carries these models, so ablating one
#      is unlikely to win — but "unlikely" per gate per model is exactly what a cheap
#      enumerated probe is for, and D3 in particular may be inert for a model whose
#      give-up wording misses D3_IDK_PHRASES.
#   6. enable_reroute — LAST, deliberately. Prior evidence is negative: enabling the
#      reroute remedy measured net -9.33pp (which is why on_core_full_rerouted sits in
#      TEACHER_FROZEN_KEYS). It still earns an arm because that measurement predates
#      the is_error_result fix, which stopped a FAILED reroute from being reported to
#      the model as a success ("the fact is saved, continue") — a bug that would by
#      itself explain a ~9pp regression. Retesting it costs one val pass; if it is
#      still bad the accept gate rejects it and the incumbent is untouched.
#   7-8. G5 / blob-pressure masks. G5 is normally skipped as inert (still ships empty
#      text); blob-pressure stopped being inert in Phase 7, which gave it real text.
#   9. enable_forced_key_search — G4's forced remedy (Phase 7). Placed with the other
#      remedies by kind but AFTER the masks it interacts with, since it is brand new and
#      has no prior evidence either way; it only fires on kv "Key not found" errors.
#   10. on_loop mask (Phase 4) — brand new like enable_forced_key_search, no prior
#      evidence either way, but placed BEFORE loop_repeat_threshold below so the
#      on/off lever is settled before its own tuning axis is probed (enable-then-tune,
#      the ordering principle Phase 5 generalizes to the rest of this tuple).
#   11-12. the two CATEGORICAL threshold axes (Phase 7), last because each consumes
#      several scheduled batches and both are speculative: nobody has measured either
#      trigger point, whereas every arm above it has at least a directional prior.
_GATE_ARM_ORDER: Tuple[Tuple[str, str], ...] = (
    ("remedy", "enable_forced_retrieval"),
    ("gate_enabled", "on_premature_idk"),              # D3
    ("gate_enabled", "on_core_clear_blocked"),         # G1
    ("gate_enabled", "on_domain_error_core_full"),     # G3
    ("gate_enabled", "on_domain_error_key_not_found"), # G4
    ("remedy", "enable_reroute"),
    ("gate_enabled", "on_domain_error_too_long"),      # G5   (skipped when inert)
    ("gate_enabled", "on_blob_pressure"),              # blob (real text since Phase 7)
    ("remedy", "enable_forced_key_search"),            # G4's forced remedy (Phase 7)
    ("gate_enabled", "on_loop"),                       # loop nudge mask (Phase 4)
    ("threshold", "blob_pressure_threshold"),          # categorical, 4 values
    ("threshold", "loop_repeat_threshold"),            # categorical, 3 values
)

# Candidate values for the CATEGORICAL axes, in the order they are tried. An axis
# absent from this map is boolean (its single arm is the negation of the incumbent).
# The values are the user-specified grids: blob pressure straddles the historical 8000
# in both directions; loop repetition covers "nudge on the first repeat" (1, today's
# behavior) through 3, which is the most a RECENT_CALL_SIGNATURE_WINDOW=4 window can
# ever detect (N repeats needs N+1 identical signatures visible at once).
_GATE_ARM_VALUES: Dict[Tuple[str, str], Tuple] = {
    ("threshold", "blob_pressure_threshold"): (4000, 6000, 8000, 10000),
    ("threshold", "loop_repeat_threshold"): (1, 2, 3),
}

# Registry key whose EFFECTIVE text an arm depends on. An arm listed here is inert
# whenever that key's text is empty (policy text, Phase 4 seed_text(), and registry
# gate_fallback() all blank), because the corresponding gate block bails on its
# emptiness guard before injecting anything — so masking it off, or moving the
# threshold at which it fires, is a guaranteed 0.00pp measurement. A template edit
# that fills the text un-inerts the arm mid-run (arms are re-enumerated every batch).
_ARM_TEXT_DEPENDENCY: Dict[Tuple[str, str], str] = {
    ("gate_enabled", "on_domain_error_too_long"): "on_domain_error_too_long",
    ("gate_enabled", "on_blob_pressure"): "on_blob_pressure",
    ("threshold", "blob_pressure_threshold"): "on_blob_pressure",
    ("threshold", "loop_repeat_threshold"): "on_loop",
}

# History marker for gate-space entries. Distinct per kind so a post-hoc reader can
# tell a mask ablation from a remedy flip from a threshold move, and so
# _build_failure_context can hide all of them from the teacher (none is a TEMPLATE_KEYS
# key — surfacing one would only tempt the teacher into an unknown_key rejection).
_GATE_EDIT_KINDS = ("gate_mask", "gate_remedy", "gate_threshold")

# arm kind -> history edit_kind. "gate_mask" for gate_enabled is unchanged from the
# one-shot probe so old histories/tests still read.
_GATE_EDIT_KIND_BY_ARM: Dict[str, str] = {
    "gate_enabled": "gate_mask",
    "remedy": "gate_remedy",
    "threshold": "gate_threshold",
}


def _gate_edit_key(kind: str, name: str, value=None) -> str:
    """Canonical history `edit_key` for a gate-space arm.

    "gate_enabled:<gate key>" is the pre-existing shape written by the one-shot mask
    probe, so old checkpoints/histories keep matching; remedies get their own
    "remedy:<flag>" namespace. This string is the ONLY spent-arm bookkeeping — no
    counter, no extra checkpoint field — which is what makes --resume decline to
    re-run an arm it already scored.

    A CATEGORICAL axis appends "=<value>" ("threshold:loop_repeat_threshold=2"), so
    each candidate value is spent independently and the axis walks its whole grid
    across successive scheduled batches instead of being retired after one try. The
    boolean kinds deliberately keep their value-free shape: a boolean axis has exactly
    one arm, and changing its key would orphan every existing history/checkpoint.
    """
    if kind == "threshold":
        return f"{kind}:{name}={value}"
    return f"{kind}:{name}"


def _gate_arm_is_inert(kind: str, name: str, policy: Dict) -> bool:
    """True if this arm provably cannot change any behavior, so scoring it would burn
    a whole val pass to measure exactly zero.

    Two independent inertness classes:

    1. Empty gate TEXT (see _ARM_TEXT_DEPENDENCY): a gate whose effective text is blank
       bails on its own emptiness guard, so neither masking it off nor moving its
       trigger point can change a single token. Must resolve text through the SAME
       three tiers as the actual gate call sites (explicit policy text → Phase 4
       seed_text() → registry gate_fallback()) or this goes stale the moment a key
       gains a seed: on_domain_error_too_long (G5) is exactly that case — its
       gate_fallback() is still "" by design, but seed_text() now supplies real text
       the instant its mask flips on, so it is no longer inert even with no explicit
       template edit.
    2. (Phase 5, threshold arms only) a DISABLED dependent gate. Phase 4 made
       "disabled" a first-class, checkable state (gate_default / gate_enabled) distinct
       from empty text — a gate can be masked off while still carrying real text (e.g.
       the seed library). Moving a threshold on a masked-off gate cannot change any
       behavior regardless of its text, so this rule only applies to THRESHOLD arms:
       a mask arm's whole purpose is to flip that same enabled state, so checking it
       against itself here would be circular.
    """
    dep = _ARM_TEXT_DEPENDENCY.get((kind, name))
    if dep is None:
        return False
    from anchoropt.memory_gates import gate_enabled, gate_fallback, seed_text
    if kind == "threshold" and not gate_enabled(policy, dep):
        return True
    text = ((policy.get("templates") or {}).get(dep) or "").strip()
    return not (text or seed_text(dep).strip() or gate_fallback(dep).strip())


def _arm_current_value(kind: str, name: str, policy: Dict, remedy_defaults: Dict[str, bool]):
    """The lever's CURRENT effective value under `policy` — the incumbent an arm is
    measured against. Reads through the same accessors the two pipelines use, so the
    arm direction can never disagree with what the gate blocks actually see."""
    from anchoropt.memory_gates import (
        blob_pressure_threshold, gate_enabled, loop_repeat_threshold, remedy_enabled,
    )
    if kind == "remedy":
        return remedy_enabled(policy, name, bool(remedy_defaults.get(name, False)))
    if kind == "gate_enabled":
        return gate_enabled(policy, name)
    if kind == "threshold":
        return {
            "blob_pressure_threshold": blob_pressure_threshold,
            "loop_repeat_threshold": loop_repeat_threshold,
        }[name](policy)
    raise ValueError(f"unknown gate arm kind: {kind!r}")


def _gate_probe_arms(
    policy: Dict, remedy_defaults: Dict[str, bool]
) -> List[Tuple[str, str, object]]:
    """Ordered (kind, name, new_value) arms: single-lever changes AWAY from `policy`.

    Every arm is a real change by construction. A BOOLEAN axis contributes one arm, the
    negation of the lever's current effective value, so a run launched with
    --enable-forced-retrieval searches "turn it off" while a stock run searches "turn it
    on". A CATEGORICAL axis contributes one arm per candidate value that differs from
    the incumbent, in _GATE_ARM_VALUES order. Inert arms are dropped (see
    _gate_arm_is_inert).

    `remedy_defaults` carries the evaluator's constructor flags, which are what a
    remedy falls back to when the policy dict does not mention it (mirrors
    remedy_enabled's `default` argument) — pass them in rather than reading the
    evaluator here so this stays a pure function.

    Accepted arms accumulate in the incumbent policy, so this is greedy coordinate
    descent: each later arm is a change relative to whatever was already adopted, NOT
    relative to the run's starting policy. For a categorical axis that means the value
    that won becomes the baseline for the axis's remaining values (and drops out of its
    own candidate list, since it is no longer a change). Coordinate descent does not
    revisit an arm rejected under an earlier incumbent — see the report's P1 note on a
    second pass.
    """
    arms: List[Tuple[str, str, object]] = []
    for kind, name in _GATE_ARM_ORDER:
        if _gate_arm_is_inert(kind, name, policy):
            continue
        current = _arm_current_value(kind, name, policy, remedy_defaults)
        values = _GATE_ARM_VALUES.get((kind, name))
        if values is None:
            arms.append((kind, name, not current))
            continue
        for value in values:
            if value != current:
                arms.append((kind, name, value))
    return arms


def _propose_gate_probe_edit(
    batch_idx: int,
    history: List[Dict],
    arms: List[Tuple[str, str, object]],
    every: int = 2,
    start: int = 0,
) -> Optional[Tuple[str, str, object]]:
    """Next unspent gate-space arm, on a scheduled batch only; None otherwise.

    Schedule: batches start, start+every, start+2*every, ... are gate batches; all
    others go to the teacher. `every=1` front-loads the whole queue (arms then
    exhaust and every later batch is the teacher's); the default 2 interleaves so
    gate arms and template edits share the run instead of one starving the other.

    "Spent" is derived purely from history edit_keys, so this is idempotent under
    --resume and cannot re-score an arm across a restart. Once the queue is
    exhausted it returns None forever and the run is pure template search.

    Categorical axes are spent PER VALUE (the edit_key carries the value), so an axis
    occupies as many scheduled batches as it has candidate values — walking its grid in
    order rather than being retired after one measurement.
    """
    if every <= 0 or not arms:
        return None
    if batch_idx < start or (batch_idx - start) % every != 0:
        return None
    spent = {h.get("edit_key") for h in history}
    for kind, name, value in arms:
        if _gate_edit_key(kind, name, value) not in spent:
            return kind, name, value
    return None


class _ShuffleCursorSampler:
    """T6: per-category shuffle-cursor sampler for even epoch coverage.

    The old per-batch ``rng.sample`` drew with replacement ACROSS batches, so over a
    run some train queries were seen many times and others never — the teacher's
    signal was biased toward whatever got sampled. This walks a freshly-shuffled
    permutation of each category, handing out the next ``k`` each batch and
    reshuffling only when a category is exhausted (epoch boundary). Every query is
    thus seen once per epoch before any is seen twice.

    State (per-category shuffled id order + cursor) is serializable so ``--resume``
    continues mid-epoch instead of restarting the permutation. The prereq filter is
    applied by the caller before handing us ``by_cat`` (prereqs must never enter a
    batch — they mutate the shared snapshot store).
    """

    def __init__(self, by_cat: Dict[str, List[Dict]], rng: random.Random):
        self._by_id = {
            cat: {c["id"]: c for c in cases} for cat, cases in by_cat.items()
        }
        self._rng = rng
        self._order: Dict[str, List[str]] = {}
        self._pos: Dict[str, int] = {}
        for cat in self._by_id:
            self._reshuffle(cat)

    def _reshuffle(self, cat: str) -> None:
        ids = list(self._by_id[cat].keys())
        self._rng.shuffle(ids)
        self._order[cat] = ids
        self._pos[cat] = 0

    def draw(self, cat: str, k: int) -> List[Dict]:
        """Next ``k`` cases for ``cat`` (capped at the category size), advancing the
        cursor and reshuffling across the epoch boundary as needed."""
        n = len(self._order.get(cat, ()))
        if n == 0:
            return []
        k = min(k, n)
        out: List[Dict] = []
        while len(out) < k:
            if self._pos[cat] >= len(self._order[cat]):
                self._reshuffle(cat)
            take = min(k - len(out), len(self._order[cat]) - self._pos[cat])
            chosen = self._order[cat][self._pos[cat]:self._pos[cat] + take]
            out.extend(self._by_id[cat][cid] for cid in chosen)
            self._pos[cat] += take
        return out

    def state(self) -> Dict:
        return {"order": {c: list(v) for c, v in self._order.items()},
                "pos": dict(self._pos)}

    def load_state(self, st: Optional[Dict]) -> None:
        if not st or not st.get("order"):
            return
        # Only restore ids that still exist in the current category (defensive
        # against a changed cases dir); reshuffle any category the checkpoint omits.
        for cat in self._by_id:
            saved = [cid for cid in st["order"].get(cat, []) if cid in self._by_id[cat]]
            if saved:
                self._order[cat] = saved
                self._pos[cat] = min(int(st.get("pos", {}).get(cat, 0)), len(saved))


def _per_category_accuracy(results: List[Dict]) -> Dict[str, float]:
    """Return per-category accuracy dict from _eval_batch results."""
    by_cat: Dict[str, Dict] = defaultdict(lambda: {"ok": 0, "total": 0})
    for r in results:
        if r.get("is_prereq"):
            continue
        cat = "kv" if "kv" in r["id"] else "vector" if "vector" in r["id"] else "rec_sum"
        by_cat[cat]["total"] += 1
        if r.get("valid") is True:
            by_cat[cat]["ok"] += 1
    return {
        cat: d["ok"] / d["total"] if d["total"] else 0.0
        for cat, d in sorted(by_cat.items())
    }


def _result_cat(result: Dict) -> str:
    """Derive backend category (kv/vector/rec_sum) from case id."""
    rid = result.get("id", "")
    if "kv" in rid:
        return "kv"
    if "vector" in rid:
        return "vector"
    return "rec_sum"


def _backend_deltas(
    incumbent_correct: Dict[str, bool], candidate_correct: Dict[str, bool]
) -> Dict[str, float]:
    """Per-backend accuracy delta (candidate − incumbent) from the per-case
    correctness vectors the gate already computed — zero additional evaluation.
    A backend absent from either side (e.g. an infra-dropped case) is skipped
    rather than zero-filled, so it never falsely reads as "no change"."""
    inc_acc = _per_category_accuracy(
        [{"id": cid, "valid": ok} for cid, ok in incumbent_correct.items()]
    )
    cand_acc = _per_category_accuracy(
        [{"id": cid, "valid": ok} for cid, ok in candidate_correct.items()]
    )
    return {
        cat: cand_acc[cat] - inc_acc[cat]
        for cat in sorted(set(inc_acc) & set(cand_acc))
    }


# ── Chain-correlation diagnostics ─────────────────────────────────────────────
# The accept gate's paired bootstrap (_bootstrap_delta_lower_ci) resamples INDIVIDUAL
# val cases, which assumes cases are independent. They are not: a memory case id is
# "memory_<backend>_<n>-<scenario>-<n>", and all cases sharing a (backend, scenario)
# read from ONE accumulating conversation store built by that chain's prereqs. The
# current val split has 78 queries in only 12 such chains (~6-7 cases each). One
# chain-level divergence — a single prereq storing a fact differently — can therefore
# flip 6 correlated cases at once, and a case-level bootstrap reads that as 6
# independent wins and reports a deceptively tight lower bound.
#
# These helpers do NOT change the accept condition (that was just re-derived today and
# is out of scope). They record, per gate decision, (a) how concentrated the measured
# effect is across chains and (b) what the lower bound would be under a CLUSTER
# bootstrap that resamples chains instead of cases — so an accept driven by one chain
# is visible in training_history.json and in the log instead of being invisible.


def _chain_key(case_id: str) -> str:
    """Correlated-cluster id for a case: "<backend>|<scenario>".

    Scenario is the middle dash-delimited segment ("memory_kv_13-customer-13" →
    "customer"); backend reuses _result_cat so the two never disagree. Backend is part
    of the key because kv/vector/rec_sum keep independent stores for the same
    scenario — clustering on scenario alone would merge three uncorrelated chains.
    """
    parts = str(case_id).split("-")
    scenario = parts[1] if len(parts) > 2 else (parts[0] if parts else "")
    return f"{_result_cat({'id': case_id})}|{scenario}"


def _chain_concentration(
    incumbent_correct: Dict[str, bool], candidate_correct: Dict[str, bool]
) -> Optional[Dict]:
    """Where the candidate-vs-incumbent flips live, grouped by chain.

    Returns None when nothing flipped. Otherwise:
      n_flips        — cases whose correctness changed either way
      n_chains       — distinct chains containing at least one flip
      top_chain      — chain with the largest |net| contribution
      top_chain_net  — that chain's net (gains-losses), in CASES not percent
      net            — overall net gains-losses, in cases
      single_chain   — True iff every flip is inside one chain

    Computed from the correctness vectors the gate already has — zero extra eval,
    exactly like _backend_deltas.
    """
    ids = [i for i in incumbent_correct if i in candidate_correct]
    per_chain: Dict[str, Dict[str, int]] = defaultdict(lambda: {"gain": 0, "loss": 0})
    n_flips = 0
    for cid in ids:
        inc, cand = bool(incumbent_correct[cid]), bool(candidate_correct[cid])
        if inc == cand:
            continue
        n_flips += 1
        per_chain[_chain_key(cid)]["gain" if cand else "loss"] += 1
    if not n_flips:
        return None
    nets = {c: d["gain"] - d["loss"] for c, d in per_chain.items()}
    top_chain = max(sorted(nets), key=lambda c: abs(nets[c]))
    return {
        "n_flips": n_flips,
        "n_chains": len(per_chain),
        "top_chain": top_chain,
        "top_chain_net": nets[top_chain],
        "net": sum(nets.values()),
        "single_chain": len(per_chain) == 1,
    }


def _bootstrap_delta_lower_ci_clustered(
    incumbent: Dict[str, bool],
    candidate: Dict[str, bool],
    n_boot: int,
    alpha: float,
    rng: random.Random,
) -> Tuple[float, float, int]:
    """Cluster (chain-level) counterpart of _bootstrap_delta_lower_ci.

    Resamples CHAINS with replacement and recomputes the delta over the union of the
    drawn chains' cases, so a within-chain run of correlated flips counts as the one
    event it is. Returns (observed_delta, lower_ci, n_chains).

    Vectorized the same way as the case-level version: per-chain (size, inc_sum,
    cand_sum) are enough to compute every resample's delta, since the statistic is a
    ratio of sums. Reported as a DIAGNOSTIC only — with 12 chains on the current val
    split, the Bonferroni-corrected tail this is evaluated at has ~1 chain of
    resolution, so wiring it into the accept condition would reject nearly everything.
    Surfacing the number lets that be a deliberate decision rather than an accident.
    """
    ids = [i for i in incumbent if i in candidate]
    if not ids:
        return 0.0, -1.0, 0
    groups: Dict[str, List[str]] = defaultdict(list)
    for cid in ids:
        groups[_chain_key(cid)].append(cid)
    chains = sorted(groups)
    size = np.array([len(groups[c]) for c in chains], dtype=float)
    inc_sum = np.array(
        [sum(1.0 for cid in groups[c] if incumbent[cid]) for c in chains]
    )
    cand_sum = np.array(
        [sum(1.0 for cid in groups[c] if candidate[cid]) for c in chains]
    )
    observed = float((cand_sum.sum() - inc_sum.sum()) / size.sum())
    n_chains = len(chains)
    if n_boot <= 0 or n_chains < 2:
        return observed, observed, n_chains
    npy_rng = np.random.default_rng(rng.randrange(2**32))
    idx = npy_rng.integers(0, n_chains, size=(n_boot, n_chains))
    totals = size[idx].sum(axis=1)
    deltas = np.sort((cand_sum[idx].sum(axis=1) - inc_sum[idx].sum(axis=1)) / totals)
    pos = max(0, min(len(deltas) - 1, int(alpha * len(deltas))))
    return observed, float(deltas[pos]), n_chains


def _classify_failure(result: Dict, store_diag: Optional[Dict] = None) -> str:
    """Classify why a query failed — used to guide the teacher's edit proposal."""
    if result.get("error_type") == "anchoropt:force_quit":
        return "force_quit"

    answer = (result.get("final_answer") or "").lower()
    if not answer.strip():
        return "empty_answer"

    cat = _result_cat(result)

    # Store-side detection for kv/vector: fact was lost during the prereq storage phase.
    # Signals: store diagnostics show destructive clears with no archival routing, OR
    # the model searched archival this turn and came up empty (thorough search, no content).
    if cat in ("kv", "vector"):
        store_botched = (
            store_diag is not None
            and store_diag.get("clear_calls", 0) > 0
            and store_diag.get("archival_adds", 0) == 0
        )
        clear_this_turn = result.get("g1_gate_fired", False)
        searched_empty = (
            result.get("archival_searched", False)
            and result.get("retrieval_attempts", 0) >= 1
            and not result.get("retrieval_succeeded", False)
        )
        if store_botched or clear_this_turn or searched_empty:
            return f"{cat}_store"

    from anchoropt.memory_gates import D3_IDK_PHRASES
    if any(p in answer for p in D3_IDK_PHRASES):
        return "false_IDK"
    return "wrong_answer"


def _enlarge_val_from_train(
    train_cases: List[Dict],
    val_cases: List[Dict],
    n: int,
    seed: int = 0,
) -> Tuple[List[Dict], List[Dict]]:
    """Move n stratified query cases from train into val.

    Stratifies by chain type (the test_category prefix, e.g. 'memory_kv_a') so the
    move preserves backend/scenario distribution. Returns (new_train, new_val).
    n=0 is a no-op.
    """
    if n <= 0:
        return list(train_cases), list(val_cases)

    rng = random.Random(seed)
    queries = split_queries(train_cases)
    prereqs = split_prereqs(train_cases)

    by_chain: Dict[str, List[Dict]] = defaultdict(list)
    for c in queries:
        chain = c["id"].rsplit("_", 1)[0]
        by_chain[chain].append(c)

    moved: List[Dict] = []
    total = len(queries)
    for chain, chain_cases in by_chain.items():
        k = max(1, round(n * len(chain_cases) / total))
        k = min(k, len(chain_cases))
        sampled = rng.sample(chain_cases, k)
        moved.extend(sampled)
        if len(moved) >= n:
            break

    moved_ids = {c["id"] for c in moved}
    new_train = prereqs + [c for c in queries if c["id"] not in moved_ids]
    new_val = list(val_cases) + moved
    print(
        f"[val_enlarge] Moved {len(moved)} queries train→val: "
        f"train={len([c for c in new_train if 'prereq' not in c['id']])} queries, "
        f"val={len(new_val)} entries",
        flush=True,
    )
    return new_train, new_val


def _chain_for_query(case_id: str) -> str:
    """Extract chain key from a query case id (e.g. memory_kv_97-student-17 → memory_kv_student)."""
    m = re.match(r"(memory_\w+?)_\d+-(\w+)-\d+$", case_id)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return ""


def _build_failure_context(
    results: List[Dict],
    cases: List[Dict],
    recent_history: Optional[List[Dict]] = None,
    store_diag_map: Optional[Dict] = None,
    editable_keys: Optional[frozenset] = None,
) -> str:
    """Build a compact failure summary for the LLM teacher.

    editable_keys: if given (Phase 6), telemetry lines for a non-editable key (a
    frozen working gate, or a registry-only key like on_blob_pressure that isn't
    even a TEMPLATE_KEYS member) are dropped rather than shown by name — the same
    leak as dumping frozen text in "Current templates": naming a key here tempts an
    edit proposal the training loop will just reject. None (the default) keeps the
    old unfiltered behavior for callers that don't pass it.
    """
    case_by_id = {c["id"]: c for c in cases}
    failed = [r for r in results if not r["is_prereq"] and not r.get("valid")]
    total_q = sum(1 for r in results if not r["is_prereq"])

    def _diag(r):
        if store_diag_map:
            return store_diag_map.get(_chain_for_query(r["id"]))
        return None

    # Error distribution for high-level teacher context
    by_type: Dict[str, int] = defaultdict(int)
    for r in failed:
        by_type[_classify_failure(r, store_diag=_diag(r))] += 1
    dist = "  ".join(f"{k}={v}" for k, v in sorted(by_type.items(), key=lambda x: -x[1]))

    store_failures = by_type.get("kv_store", 0) + by_type.get("vector_store", 0)
    store_banner = ""
    if store_failures > 0 and store_failures >= len(failed) // 2:
        store_banner = (
            f"\n⚠ STORE-SIDE FAILURES DOMINATE ({store_failures}/{len(failed)}): "
            "facts are not persisted — the model routes overflow to core_memory_clear "
            "instead of archival, or paraphrases instead of storing verbatim. "
            "This is a storage-policy problem — on_memory_preamble is the natural "
            "candidate, but if it's been tried and rejected recently (see recent "
            "edits below), don't just retry it; consider whether a different "
            "editable key addresses the same failures.\n"
        )

    # Telemetry: which gates AND soft signals fired this batch and whether firing
    # correlated with an ultimately-valid answer. Surfaces signals that fire often
    # but rarely convert (e.g. a rec_sum gate that nudges but doesn't fix), and —
    # crucially (T1) — the SOFT signals the teacher edits, per backend, so a
    # template that helps one backend while poisoning another is now visible.
    from anchoropt.memory_gates import MEMORY_GATE_REGISTRY
    flag_to_key = {g.telemetry_flag: g.key for g in MEMORY_GATE_REGISTRY if g.telemetry_flag}
    rollup = _telemetry_rollup(results)

    def _backend_convert_str(stats):
        # "kv 48%(21) vector 68%(19) rec_sum 56%(25)" — convert_rate(fired) per backend.
        parts = []
        for bk in ("kv", "vector", "rec_sum"):
            d = stats["by_backend"].get(bk)
            if d and d["fired"]:
                parts.append(f"{bk} {d['convert_rate']:.0%}({d['fired']})")
        return "  ".join(parts)

    gate_lines, signal_lines = [], []
    for flag, stats in sorted(rollup.items(), key=lambda kv: -kv[1]["fired"]):
        if stats["fired"] == 0:
            continue
        key = flag_to_key.get(flag, flag)
        # Phase 6: naming a non-editable key here (a frozen working gate, or a
        # registry-only key like on_blob_pressure with no TEMPLATE_KEYS slot at all)
        # is the same leak as dumping frozen text in "Current templates" — it tempts
        # a proposal the training loop only rejects. Drop it from the prompt entirely.
        if editable_keys is not None and key not in editable_keys:
            continue
        # For gates the rollup id is the telemetry_flag (kept visible so the teacher
        # can correlate with logs); for signals key == flag, so show it once.
        label = f"{key} ({flag})" if key != flag else key
        line = (
            f"  {label}: fired={stats['fired']} helped={stats['helped']} "
            f"hurt={stats['hurt']} convert_rate={stats['convert_rate']:.0%} "
            f"| per-backend {_backend_convert_str(stats)}"
        )
        (signal_lines if stats.get("kind") == "signal" else gate_lines).append(line)

    gate_telemetry_block = ""
    if gate_lines:
        gate_telemetry_block += (
            "\nGate telemetry this batch (deterministic gates — DON'T edit their text "
            "unless convert_rate is low; they mostly already work):\n"
            + "\n".join(gate_lines) + "\n"
        )
    if signal_lines:
        gate_telemetry_block += (
            "\nSoft-signal telemetry this batch (THESE are the tunable templates you "
            "edit; per-backend convert_rate shows where a template helps vs hurts — a "
            "signal converting WORSE than baseline on one backend is poisoning it and "
            "should be scoped away from that backend, not made stronger everywhere):\n"
            + "\n".join(signal_lines) + "\n"
        )

    # T7: batches since the last accepted edit — pure signal, no threshold/banner
    # logic here. The teacher's own rules (below) tell it what to do with a high
    # number; this line just makes the count visible every batch.
    stuck = _stuck_batches(recent_history or [])

    lines = [
        f"Failed {len(failed)}/{total_q} cases.\n"
        f"Error distribution: {dist}\n"
        f"Batches since last accepted edit: {stuck}\n"
        f"{store_banner}"
        f"{gate_telemetry_block}"
        "Known error → template mapping:\n"
        "  kv_store / vector_store → on_memory_preamble  (storage routing + verbatim storage)\n"
        "  false_IDK    → on_idk_fallback, on_retrieval_success_pre_answer\n"
        "  wrong_answer → on_retrieval_success_pre_answer\n"
        "  empty_answer → on_idk_fallback\n"
        # Phase 6: on_hallucinate/on_loop/on_wrong_args/on_domain_error/
        # on_domain_error_too_long fire off the LAST TOOL CALL mid-episode, not a
        # post-hoc classification of the final answer, so they have no natural row
        # above — point at where their firing is already visible instead of
        # fabricating one.
        "  on_hallucinate, on_loop, on_wrong_args, on_domain_error, "
        "on_domain_error_too_long → no error-type row (they trigger on the last "
        "tool call, not the final answer); check the telemetry block above for "
        "whether they fired this batch\n"
    ]
    for r in failed[:8]:
        case = case_by_id.get(r["id"], {})
        q = case.get("question", [[]])
        last_user = ""
        for turn in reversed(q):
            user_msgs = [m for m in turn if isinstance(m, dict) and m.get("role") == "user"]
            if user_msgs:
                last_user = user_msgs[-1].get("content", "")[:120]
                break
        ftype = _classify_failure(r, store_diag=_diag(r))
        storage_note = ""
        if ftype in ("kv_store", "vector_store"):
            d = _diag(r)
            if d:
                storage_note = (
                    f"\n    storage: chain had {d['clear_calls']} core_memory_clear calls, "
                    f"{d['full_errors']} 'full' errors, {d['archival_adds']} archival_memory_add"
                )
            if r.get("g1_gate_fired"):
                storage_note += "  [g1 fired this turn]"
        # T4: the gold answer is intentionally WITHHELD from the teacher. Showing
        # `expected: <gold>` let the teacher's reward-maximizing move be to bake the
        # answer into template text (the batch-8 memorization mechanism) — which does
        # not transfer to the unseen test split. The teacher gets the question, the
        # model's WRONG answer, the failure type, and the backend — enough to diagnose
        # a steering fix, nothing to memorize.
        lines.append(
            f"  id={r['id'][:50]}  [{ftype}] backend={_result_cat(r)}\n"
            f"    question: {last_user}\n"
            f"    model_answer(wrong): {r.get('final_answer', '')[:80]}"
            f"{storage_note}\n"
        )

    # Recent edit history
    if recent_history:
        recent_edits = [
            h for h in recent_history[-8:]
            if h.get("edit_key") and not h.get("reject_reason", "").startswith("bad_slots")
            and h.get("edit_kind") not in _GATE_EDIT_KINDS
        ]
        if recent_edits:
            lines.append("\nRecent edits (vary your target key if repeating a rejected one —")
            lines.append("  this applies to on_memory_preamble too; a repeated rejection there")
            lines.append("  means try a different key or a materially different text, not the same idea again):")
            for h in recent_edits:
                verdict = "ACCEPTED" if h.get("accepted") else "REJECTED"
                delta = h.get("val_score_after", 0) - h.get("val_score_before", 0)
                # (2026-08-06): net_flips/min_flips are the actual gate under
                # gate_mode=simple — surface them so the near-miss guidance below has
                # a live number to point at instead of the now-diagnostic-only boot_lower.
                net_flips, min_flips = h.get("net_flips"), h.get("min_flips")
                flips_str = f" net_flips={net_flips:+d}/{min_flips:g}" if net_flips is not None else ""
                boot_lower = h.get("boot_lower")
                boot_str = f" boot_lower={boot_lower:+.1%}" if boot_lower is not None else ""
                backend_delta = h.get("backend_delta") or {}
                backend_str = "  ".join(f"{k}={v:+.0%}" for k, v in sorted(backend_delta.items()))
                backend_str = f" | per-backend {backend_str}" if backend_str else ""
                lines.append(
                    f"  [{verdict} delta={delta:+.1%}{flips_str}{boot_str}] key={h['edit_key']!r}{backend_str}"
                )

    return "\n".join(lines)


# ── Teacher prompt for memory tasks ──────────────────────────────────────────

_TEACHER_SYSTEM = """\
You are an expert prompt engineer for LLM tool-use agents.
You are optimizing injection templates for a memory-agent task where the model:
1. Stores and retrieves facts using memory API tools (core_memory_add/retrieve, archival_memory_add/retrieve, etc.)
2. Must answer user questions by retrieving the right fact from memory
3. Returns a final natural-language answer graded by substring match

Template slots available: {slots}

Template keys and their purpose:
{key_docs}

  on_memory_preamble — SYSTEM message rule applied at every turn AND during fact
                       storage (prereq phase). The main slot that shapes storage
                       behavior. Use it for durable behavioral policy: overflow
                       routing (when core is full → archival_memory_add, never
                       core_memory_clear), verbatim storage (exact numbers/names/
                       dates, no paraphrasing). A reasonable candidate when the
                       failure summary shows kv_store/vector_store failures
                       dominating — but if recent edits show it was already tried
                       and rejected for this, don't just retry the same idea;
                       either propose materially different text or target a
                       different editable key instead.

Injection levels: trailing_user keys fire AFTER tool results (high recency, like gates).
  system keys are prepended to the system message once per episode.

Rules:
- Propose EXACTLY ONE edit (change one template key's text)
- Only use slot names from the provided list
- Keep injections short (≤5 lines): the model's context is limited
- Do NOT suggest fine-tuning, RAG, or architectural changes
- NEVER put a specific expected answer, fact value, name, number, date, or quoted
  phrase from any example into template text. Templates are GENERALIZABLE steering
  rules — what the model should DO (which tool to call, how to verify, what to store)
  — filled at runtime from live {{slots}}. A template that helps only because it
  contains the answer is memorization: it will PASS train/val and FAIL the unseen
  test split. Write instructions, not answer keys.
- Use the per-backend convert_rate telemetry: if a signal converts WORSE on one
  backend than the others, it is poisoning that backend — do not make it stronger
  everywhere; that backend is likely already scoped away from it.
- The failure summary reports "Batches since last accepted edit" and, per recent
  edit, its net_flips/min_flips and per-backend delta. The higher that count climbs,
  the less useful another small rewording of a recently-rejected idea is — a
  near-miss (positive net_flips just short of min_flips) is worth refining, but
  several rejections in a row for the same key mean the MECHANISM is wrong, not
  just the phrasing: change
  what the instruction actually tells the model to do, not just how it says it.
- Return JSON: {{"key": "<template_key>", "new_text": "<new template text>"}}
"""

_TEACHER_USER = """\
Current templates:
{templates}

Recent batch failure summary:
{failures}

Val accuracy before edit: {val_score:.1%}

Propose the single template edit most likely to increase accuracy.
Focus on the most common failure pattern visible above.
Return only JSON: {{"key": "...", "new_text": "..."}}
"""


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    evaluator,
    train_cases: List[Dict],
    val_cases: List[Dict],
    initial_templates: Dict,
    out_dir: Path,
    batches: int = 0,           # T6: >0 overrides epochs; 0 => epochs * steps_per_epoch
    epochs: int = 2,            # T6: primary loop control (full even coverage per epoch)
    batch_size: int = 12,
    seed: int = 42,
    llm_model: str = "anthropic/claude-opus-4-8",
    val_extra: int = 0,
    min_effect: float = 0.01,   # W4: min val improvement to ADOPT a candidate (noise band)
    null_churn_cases: float = 2.0,  # T8/(2026-08-06): floor on min_flips (integer case-flip
                                # count), so a 1-2 case fluke can't clear the bar even at a tiny
                                # min_effect. Lowered from 3.0: f_null=11 (the original floor's
                                # justification) was superseded by the prefix-caching fix, which
                                # measured f_null=0. Only raise if a re-measured noise floor exceeds this.
    gate_mode: str = "simple", # (2026-08-06): "simple" = SkillOpt-style magnitude-only accept
                                # (candidate>incumbent, guarded by null_churn_cases); "bootstrap" =
                                # the original stricter magnitude-AND-boot_lower>0 regime. See
                                # _accept_decision and _GATE_MODES.
    gate_n_seeds: int = 1,      # W4: >1 averages the candidate over paired seeds (stochastic regimes)
    workers: int = 1,
    resume: bool = False,       # W6: resume from out_dir/checkpoint.json if present
    boot_n: int = 50000,        # T5: paired case-bootstrap resamples for the accept-gate CI (0 disables).
                                # Raised from 2000 (RCA 2026-08-03): a Bonferroni-corrected alpha over a
                                # 20-batch run needs tail resolution 2000 resamples doesn't have.
    boot_alpha: float = 0.05,   # T5: one-sided lower-CI level, Bonferroni-spent across total_batches;
                                # accept requires boot_lower > 0.0 (pure sign check, T8) — NOT a second
                                # magnitude bar against effective_min_effect (that was the pre-T8 rule)
    dup_threshold: float = 0.8, # T5: difflib ratio above which a re-proposed rejected edit is skipped
    probe_gate_mask: bool = False,       # deterministic D3 on/off ablation probe (fires once, batch 0)
    probe_gate_mask_key: str = "on_premature_idk",
    probe_gates: bool = False,           # enumerated gate-space arm queue, revisited across the run
    gate_probe_every: int = 2,           # stride: gate arm on batches 0, every, 2*every, ... until exhausted
):
    from anchoropt.template_engine import TEMPLATE_KEYS, validate_template
    from anchoropt.memory_gates import (
        teacher_key_doc_block, telemetry_flags,
        TEACHER_FROZEN_KEYS, teacher_editable_keys,
    )
    # T3: the teacher may only edit keys with real gate-free headroom. Frozen keys
    # are working gate text (edits regress) or signals the evaluator never fires.
    editable_keys = teacher_editable_keys(TEMPLATE_KEYS)

    # Teacher seam: make_teacher builds a LiteLLMTeacher for a litellm model id
    # (the default) or a VLLMTeacher for a local endpoint. Request quirks (Bedrock
    # extra_body, max_tokens) live inside the client, not here.
    teacher = make_teacher(llm_model)
    if teacher is None:
        print("[warn] No teacher model configured — teacher edits disabled", flush=True)

    if gate_mode not in _GATE_MODES:
        raise ValueError(f"gate_mode must be one of {_GATE_MODES}, got {gate_mode!r}")

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "training_history.json"
    best_path    = out_dir / "templates_best.json"

    # Optionally enlarge val by moving stratified queries from train
    if val_extra > 0:
        train_cases, val_cases = _enlarge_val_from_train(train_cases, val_cases, val_extra, seed=seed)
        # Persist the updated split so subsequent eval jobs use the same partition
        split_data_dir = out_dir / "data"
        split_data_dir.mkdir(parents=True, exist_ok=True)
        with open(split_data_dir / "memory_train_cases.json", "w") as f:
            json.dump(train_cases, f, indent=2)
        with open(split_data_dir / "memory_val_cases.json", "w") as f:
            json.dump(val_cases, f, indent=2)
        print(f"[val_enlarge] Split saved to {split_data_dir}", flush=True)

    current_templates = initial_templates
    history = []

    # ── T6: epoch-based loop control + per-category shuffle-cursor sampler ─────
    # Build the query-only, category-stratified pools ONCE (prereqs excluded — they
    # mutate the shared snapshot store). steps_per_epoch covers every query exactly
    # once at the configured batch_size; total_batches = epochs * steps_per_epoch
    # unless an explicit --batches > 0 overrides it (back-compat for old scripts).
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for c in train_cases:
        if "prereq" in c["id"]:
            continue
        cat = "kv" if "kv" in c["id"] else "vector" if "vector" in c["id"] else "rec_sum"
        by_cat[cat].append(c)
    n_train_queries = sum(len(v) for v in by_cat.values())
    steps_per_epoch = max(1, math.ceil(n_train_queries / max(batch_size, 1)))
    if batches and batches > 0:
        total_batches = batches
        print(f"[loop] {total_batches} batches (explicit --batches override)", flush=True)
    else:
        total_batches = epochs * steps_per_epoch
        print(f"[loop] {epochs} epochs × {steps_per_epoch} steps/epoch = "
              f"{total_batches} batches ({n_train_queries} train queries, "
              f"batch_size={batch_size})", flush=True)
    # T5-Bonferroni: the accept-gate CI is evaluated once per batch against the SAME
    # fixed val set, so a nominal boot_alpha is really a family-wise error rate over
    # total_batches sequential tests (RCA 2026-08-03: ~53% chance of >=1 false accept
    # at boot_alpha=0.05 over 18-20 batches, by Monte-Carlo simulation of the observed
    # churn rate). Spend the configured alpha across the whole run instead of per-test.
    boot_alpha_effective = boot_alpha / max(total_batches, 1)
    print(f"[gate] Bonferroni-corrected boot_alpha = {boot_alpha_effective:.4%} "
          f"({boot_alpha:.2%} / {total_batches} batches)", flush=True)
    sampler = _ShuffleCursorSampler(by_cat, rng)

    # ── Gate-space search budget ──────────────────────────────────────────────
    # The evaluator's constructor flags are the run-level DEFAULT for each remedy;
    # a candidate policy overrides them per batch (see remedy_enabled). Coerced to
    # a real bool so a MagicMock evaluator in tests reads as False rather than
    # truthy — the arm direction must never depend on a test double.
    _remedy_defaults = {}
    for _kind, _name in _GATE_ARM_ORDER:
        if _kind != "remedy":
            continue
        _val = getattr(evaluator, _name, False)
        _remedy_defaults[_name] = _val if isinstance(_val, bool) else False
    if probe_gates:
        _arms0 = _gate_probe_arms(current_templates, _remedy_defaults)
        _last = (len(_arms0) - 1) * max(gate_probe_every, 1)
        print(f"[probe] Gate-space search ON: {len(_arms0)} arm(s), stride "
              f"{gate_probe_every} → last arm at batch {_last + 1}/{total_batches}"
              + ("  ** QUEUE DOES NOT FIT IN THIS RUN **"
                 if _last >= total_batches else ""), flush=True)
        for _k, _n, _v in _arms0:
            print(f"[probe]   {_gate_edit_key(_k, _n, _v)} → {_v}", flush=True)

    # ── W6: resume from checkpoint ───────────────────────────────────────────
    ckpt_path = out_dir / "checkpoint.json"
    start_batch = 0
    _resumed_best = None
    _resumed_correct = None
    if resume and ckpt_path.exists():
        with open(ckpt_path) as f:
            _ck = json.load(f)
        current_templates = _ck.get("current_templates", current_templates)
        _resumed_best = _ck.get("best_val_score")
        start_batch = int(_ck.get("next_batch", 0))
        history = _ck.get("history", [])
        # T6: restore the sampler cursor so resume continues mid-epoch.
        sampler.load_state(_ck.get("sampler_state"))
        # T5: restore the incumbent per-case vector so the bootstrap gate works on
        # the first post-resume batch (older checkpoints lack it → re-measured below).
        _bvc = _ck.get("best_val_correct")
        if isinstance(_bvc, dict):
            _resumed_correct = {k: bool(v) for k, v in _bvc.items()}
        try:
            _st = _ck.get("rng_state")
            if _st is not None:
                rng.setstate((_st[0], tuple(_st[1]), _st[2]))
        except Exception:
            pass
        print(f"[resume] Resuming at batch {start_batch} (best_val={_resumed_best})", flush=True)

    # ── P1: Build snapshot store from train prereqs (all 15 chains) ──────────
    train_prereqs = split_prereqs(train_cases)
    print(f"\n[snapshot_store] Running {len(train_prereqs)} train prereqs upfront ...", flush=True)
    evaluator.build_snapshot_store(train_prereqs, current_templates)

    # ── Baseline val score ────────────────────────────────────────────────────
    # W1 determinism: decisional evals (baseline / gate) run serially. vLLM greedy
    # decode is not byte-stable under concurrent continuous batching
    # (batch-composition-dependent float rounding), so accept/reject must not be
    # exposed to that noise. Exploratory train-batch eval stays parallel below.
    gate_workers = 1
    if _resumed_best is not None and _resumed_correct is not None:
        best_val_score = _resumed_best
        best_val_correct = _resumed_correct
        print(f"[resume] Using checkpointed baseline val: {best_val_score:.1%} "
              f"({len(best_val_correct)} scored cases)", flush=True)
    else:
        # Measure the incumbent's val accuracy AND per-case correctness vector. The
        # vector is what the T5 paired bootstrap resamples in the accept gate. We
        # re-measure here even on resume from an older checkpoint that lacks it.
        print("\n[baseline] Measuring val accuracy ...", flush=True)
        best_val_score, _base_res = _score_split(
            evaluator, train_prereqs, val_cases, current_templates, "val_baseline", gate_workers
        )
        best_val_correct = _correct_vec(_base_res)
        if _resumed_best is not None:
            best_val_score = _resumed_best  # keep the checkpointed score; vector is fresh
        print(f"[baseline] Val accuracy: {best_val_score:.1%} "
              f"({len(best_val_correct)} scored cases)", flush=True)

    # T5/T8: require the improvement to be worth at least `null_churn_cases` val-case
    # flips, so a 1-2 case fluke on a small val set can't clear the bar. Floor at the
    # configured min_effect. This scales the noise band with val size automatically.
    # Reporting-only: the actual gate recomputes this same floor via null_churn_cases
    # inside _accept_decision (min_flips); kept in sync here so prints/history match.
    n_val_scored = max(len(best_val_correct), 1)
    effective_min_effect = max(min_effect, null_churn_cases / n_val_scored)
    _min_flips_preview = max(math.ceil(min_effect * n_val_scored), null_churn_cases)
    print(f"[gate] effective min_effect = {effective_min_effect:.2%} "
          f"(max of configured {min_effect:.2%} and {null_churn_cases:.0f}/{n_val_scored} val cases) "
          f"-> min_flips = {_min_flips_preview:g} of {n_val_scored} val cases", flush=True)
    # (2026-08-06): announce the active regime up front — "simple" means boot_lower
    # below is a recorded diagnostic only, NOT a gating clause (see _accept_decision).
    print(f"[gate] gate_mode = {gate_mode!r} "
          f"({'magnitude only' if gate_mode == 'simple' else 'magnitude AND boot_lower>0'})",
          flush=True)
    if boot_n <= 0:
        print(f"[gate] ⚠ boot_n={boot_n}<=0: boot_lower recorded below is the raw point "
              f"estimate (candidate - incumbent), NOT a lower confidence bound — "
              f"{'harmless here since gate_mode=simple never gates on it' if gate_mode == 'simple' else 'this collapses the bootstrap clause to a bare sign check, delta>0'}.",
              flush=True)

    for batch_idx in range(start_batch, total_batches):
        print(f"\n{'='*60}", flush=True)
        print(f"Batch {batch_idx+1}/{total_batches} "
              f"(epoch {batch_idx // steps_per_epoch + 1})", flush=True)

        # ── Sample training batch ─────────────────────────────────────────────
        # T6: stratified per-category shuffle cursor (built once, pre-loop) — even
        # coverage each epoch instead of the old sample-with-replacement. Category
        # pools are QUERIES ONLY (prereqs were excluded when by_cat was built):
        # running a prereq episode flushes memory to the shared snapshot store (see
        # evaluate_episode), so one entering a batch would rewrite the store the val
        # gate reads from. The store is built once upfront.
        per_cat = batch_size // max(len(by_cat), 1)
        batch: List[Dict] = []
        for cat in by_cat:
            batch.extend(sampler.draw(cat, per_cat))
        # Snapshot store was built upfront — batches are queries only
        ordered_batch = batch

        # ── Evaluate batch ────────────────────────────────────────────────────
        print(f"[batch {batch_idx+1}] Running {len(ordered_batch)} entries ...", flush=True)
        train_score, train_results = _eval_batch(
            ordered_batch, current_templates, evaluator, tag_prefix=f"b{batch_idx}", max_workers=workers
        )
        print(f"[batch {batch_idx+1}] Train accuracy: {train_score:.1%}", flush=True)
        # Phase 5: computed unconditionally (not just on teacher batches, where
        # _build_failure_context already derived the same thing transiently) so every
        # batch's history entry carries which gates/signals actually fired — without
        # this, a firing-aware inertness rule has no data source to consult.
        batch_gate_telemetry = _telemetry_rollup(train_results)

        # ── Teacher edit, or a deterministic gate-space arm ───────────────────
        # A gate-space arm is a single-lever change enumerated by _gate_probe_arms (a
        # boolean flip, or the next value of a categorical axis), so it needs no LLM call
        # and none of the four teacher-specific validators below (TEMPLATE_KEYS /
        # TEACHER_FROZEN_KEYS / validate_template / _is_dup_proposal all assume a
        # template-text edit). --probe-gates walks the whole queue across the run at
        # stride gate_probe_every; --probe-gate-mask is the legacy one-arm one-batch
        # special case. Arms are re-enumerated EVERY batch against the current
        # incumbent, so an accepted change re-points every later arm on that axis (and a
        # template edit that fills G5's text un-inerts its mask arm).
        gate_edit = None
        if probe_gates:
            gate_edit = _propose_gate_probe_edit(
                batch_idx, history,
                _gate_probe_arms(current_templates, _remedy_defaults),
                every=gate_probe_every,
            )
        elif probe_gate_mask:
            _legacy = _propose_gate_mask_edit(
                batch_idx, history, key=probe_gate_mask_key, policy=current_templates,
            )
            if _legacy is not None:
                gate_edit = ("gate_enabled", _legacy[0], _legacy[1])

        if gate_edit is not None:
            gate_kind, key, gate_new_value = gate_edit
            # edit_text stays a plain string for every arm kind: _is_dup_proposal and
            # _build_failure_context only ever read it as text.
            new_text = (
                str(gate_new_value) if gate_kind == "threshold"
                else ("true" if gate_new_value else "false")
            )
            _lever = (f"gate_enabled[{key!r}]" if gate_kind == "gate_enabled" else key)
            print(f"[probe] Proposed gate-space arm: {_lever} = {gate_new_value}", flush=True)
        else:
            if teacher is None:
                history.append({
                    "batch": batch_idx, "train_score": train_score,
                    "val_score_before": best_val_score, "accepted": False,
                    "reject_reason": "no_teacher",
                    "gate_telemetry": batch_gate_telemetry,
                })
                continue

            failure_ctx = _build_failure_context(
                train_results, ordered_batch,
                recent_history=history,
                store_diag_map=evaluator._store_diag,
                editable_keys=editable_keys,
            )
            # Phase 6: only dump editable keys' text. Dumping every key (including
            # frozen working-gate text) put a frozen key's text in front of the teacher
            # right next to editable ones with no visual distinction — key_docs hid its
            # DESCRIPTION but not this, so it was still a live edit target in practice.
            editable_templates = {
                k: v for k, v in current_templates.get("templates", {}).items()
                if k in editable_keys
            }
            tmpl_summary = json.dumps(editable_templates, indent=2)[:3000]

            prompt = _TEACHER_USER.format(
                templates=tmpl_summary,
                failures=failure_ctx,
                val_score=best_val_score,
            )
            sys_prompt = _TEACHER_SYSTEM.format(
                slots=", ".join(sorted(
                    __import__("anchoropt.template_engine", fromlist=["SLOT_NAMES"]).SLOT_NAMES
                )),
                # Phase 6: include=editable_keys (not exclude=TEACHER_FROZEN_KEYS) — the
                # registry has keys outside TEMPLATE_KEYS entirely (on_blob_pressure),
                # which aren't frozen but also aren't teacher-editable; exclude alone let
                # them through, so the teacher would target one and get unknown_key'd.
                key_docs=teacher_key_doc_block(include=editable_keys),
            )

            try:
                raw = teacher.complete(sys_prompt, prompt)
                # Try clean parse first (handles slot names like {retrieved} inside new_text)
                try:
                    edit = json.loads(raw.strip())
                except (json.JSONDecodeError, ValueError):
                    # Fall back: greedy outer-brace extraction
                    m = re.search(r'\{.*\}', raw, re.DOTALL)
                    if not m:
                        raise ValueError(f"No JSON in teacher response: {raw[:200]}")
                    edit = json.loads(m.group(0))
                key = edit.get("key", "")
                new_text = edit.get("new_text", "")
            except Exception as e:
                print(f"[teacher] Error: {e}", flush=True)
                history.append({
                    "batch": batch_idx, "train_score": train_score,
                    "val_score_before": best_val_score, "accepted": False,
                    "reject_reason": f"teacher_error:{e}",
                })
                continue

            if key not in TEMPLATE_KEYS:
                print(f"[teacher] Unknown key '{key}' — skipping", flush=True)
                history.append({
                    "batch": batch_idx, "train_score": train_score,
                    "val_score_before": best_val_score, "accepted": False,
                    "reject_reason": "unknown_key", "edit_key": key,
                })
                continue

            # T3: reject edits to frozen keys (working gate text or never-fired signals).
            # These are hidden from the teacher prompt, so a proposal here is the teacher
            # ignoring instructions — never let it burn a batch or regress a working gate.
            if key in TEACHER_FROZEN_KEYS:
                print(f"[teacher] Frozen key '{key}' (working gate / never-fired) — skipping", flush=True)
                history.append({
                    "batch": batch_idx, "train_score": train_score,
                    "val_score_before": best_val_score, "accepted": False,
                    "reject_reason": "frozen_key", "edit_key": key,
                })
                continue

            bad_slots = validate_template(new_text)
            if bad_slots:
                print(f"[teacher] Unknown slots {bad_slots} in proposed edit — skipping", flush=True)
                history.append({
                    "batch": batch_idx, "train_score": train_score,
                    "val_score_before": best_val_score, "accepted": False,
                    "reject_reason": f"bad_slots:{bad_slots}",
                })
                continue

            # ── Duplicate-proposal guard (T5) ────────────────────────────────
            # Don't spend a gate eval re-testing an edit >dup_threshold similar to one
            # already rejected for the SAME key — the deterministic eval will reject it
            # again. Cheap difflib check on the rejected-edit history for this key.
            is_dup, dup_ratio = _is_dup_proposal(history, key, new_text, dup_threshold)
            if is_dup:
                print(f"[teacher] Duplicate proposal for {key!r} "
                      f"(ratio {dup_ratio:.2f} > {dup_threshold:.2f}) — skipping", flush=True)
                history.append({
                    "batch": batch_idx, "train_score": train_score,
                    "val_score_before": best_val_score, "accepted": False,
                    "reject_reason": f"dup_proposal:{dup_ratio:.2f}",
                    "edit_key": key, "edit_text": new_text,
                })
                continue

            print(f"[teacher] Proposed edit: key={key!r}  text={new_text[:80]!r}", flush=True)

        # ── Statistical acceptance gate (W4) ─────────────────────────────────
        # W1 made eval deterministic (temp 0 + seed + serial), so best_val_score is a
        # REPRODUCIBLE measurement of the incumbent — not a one-shot lucky high. The
        # winner's-curse fix: ADOPT the candidate only when it improves val by MORE
        # than a min-effect band (no accept-on-tie, no adopting within-noise changes),
        # and set best to the candidate's own measured score. The old code accepted on
        # delta>=0 and then averaged in a confirm eval keyed off the stale baseline,
        # which ratcheted best toward noise peaks. For stochastic regimes
        # (temperature>0), set gate_n_seeds>1 to average the candidate over paired seeds.
        candidate = json.loads(json.dumps(current_templates))
        if gate_edit is not None:
            if gate_kind == "gate_enabled":
                candidate.setdefault("gate_enabled", {})[key] = gate_new_value
            else:
                # Remedies AND thresholds are TOP-LEVEL policy values, matching how
                # base_handler.py reads them (load_anchoropt_policy().get("enable_reroute"),
                # blob_pressure_threshold(load_anchoropt_policy())) — so an accepted arm
                # lands in templates_best.json in the exact shape the official leaderboard
                # path already understands, with no extra translation step at eval time.
                candidate[key] = gate_new_value
        else:
            candidate.setdefault("templates", {})[key] = new_text

        # The store is (re)built for the candidate inside the scoring calls below via
        # the self-calibrating snapshot cache: any edit that changes a storage-affecting
        # template's text (on_memory_preamble, or an opt-in storage gate like G5/blob
        # whose condition fires during prereqs) MISSes and rebuilds; a query-only edit
        # HITs the incumbent's store. No per-key special-case is needed.
        print(f"[gate] Evaluating candidate on val set (n_seeds={gate_n_seeds}) ...", flush=True)
        # T5/T8/(2026-08-06): the accept decision depends on gate_mode, via
        # _accept_decision — see _GATE_MODES. In BOTH modes, gate_n_seeds==1
        # (deterministic, temp≈0) still gets a single reproducible val pass, and the
        # candidate-vs-incumbent per-case correctness vectors are still bootstrapped
        # below (boot_lower is always COMPUTED and RECORDED as a diagnostic, even
        # under gate_mode=simple where it does not gate):
        #  - gate_mode="simple": accept iff the integer net-flip magnitude test
        #    (min_flips) passes. SkillOpt-style greedy candidate>incumbent, guarded
        #    only by the noise floor. boot_lower is diagnostic-only.
        #  - gate_mode="bootstrap": magnitude AND a one-sided lower-CI sign check
        #    (boot_lower > 0, at a Bonferroni-corrected alpha spent across
        #    total_batches) — each term owns exactly one question (RCA 2026-08-04:
        #    requiring boot_lower to ALSO clear the magnitude band double-counted
        #    the same noise the CI already widens for, making the gate unreachable
        #    even at zero measurement noise).
        #  - gate_n_seeds>1 (stochastic, temp>0): average the score across paired
        #    seeds, then majority-vote each case's correctness across seeds into ONE
        #    vector, which feeds the exact same accept decision below (T8
        #    landmine fix 2026-08-04 — this regime used to silently skip the CI
        #    entirely and never update the incumbent vector on accept).
        if gate_n_seeds > 1:
            # Build the candidate's store ONCE (seed varies decode, not storage), then
            # score queries-only across seeds. Guard on the cache dir so the no-cache
            # path stays byte-identical: full val_cases (prereqs inline) as before.
            if evaluator._snapshot_cache_dir is not None:
                evaluator.build_snapshot_store(train_prereqs, candidate)
                _seed_cases = split_queries(val_cases)
            else:
                _seed_cases = val_cases
            _cand_scores = []
            _cand_correct_by_seed = []
            for _s in range(gate_n_seeds):
                try:
                    evaluator.seed = seed + _s
                except Exception:
                    pass
                _seed_score, _seed_res = _eval_batch(
                    _seed_cases, candidate, evaluator,
                    tag_prefix=f"gate_b{batch_idx}_s{_s}", max_workers=gate_workers
                )
                _cand_scores.append(_seed_score)
                _cand_correct_by_seed.append(_correct_vec(_seed_res))
            try:
                evaluator.seed = seed
            except Exception:
                pass
            new_val_score = sum(_cand_scores) / len(_cand_scores)
            # T8 landmine fix (2026-08-04): majority-vote each case's correctness across
            # seeds into ONE vector, so gate_n_seeds>1 feeds the same bootstrap-CI path
            # as the deterministic regime instead of silently falling back to a
            # magnitude-only test with no CI, and so an accept updates best_val_correct
            # below instead of leaving the incumbent vector stale. Ties (even seed
            # count split 50/50) default to "not correct" — no majority means no claim.
            _seed_ids = set().union(*_cand_correct_by_seed) if _cand_correct_by_seed else set()
            cand_correct = {
                cid: 2 * sum(v.get(cid, False) for v in _cand_correct_by_seed)
                     > sum(cid in v for v in _cand_correct_by_seed)
                for cid in _seed_ids
            }
        else:
            new_val_score, _cand_res = _score_split(
                evaluator, train_prereqs, val_cases, candidate, f"gate_b{batch_idx}", gate_workers
            )
            cand_correct = _correct_vec(_cand_res)

        delta = new_val_score - best_val_score
        val_score_before = best_val_score

        # T8 landmine fix: cand_correct is now populated in BOTH regimes above (the
        # gate_n_seeds>1 branch majority-votes per-seed vectors into one), so a single
        # accept rule — bootstrap CI + chain diagnostics + per-backend breakdown —
        # serves both instead of gate_n_seeds>1 silently losing its CI.
        # Bootstrap CI on a dedicated rng stream so the sampler's rng (checkpointed
        # for --resume) is never perturbed.
        boot_rng = random.Random(seed * 100003 + batch_idx)
        boot_observed, boot_lower, n_paired = _bootstrap_delta_lower_ci(
            best_val_correct, cand_correct, boot_n, boot_alpha_effective, boot_rng
        )
        # T8 (2026-08-04): magnitude (min_flips) and realness (boot_lower>0 sign
        # check at the Bonferroni-corrected alpha) are each other's job now, not
        # both the same term's — see _accept_decision. The RCA 2026-08-03 regression
        # is still caught: the Bonferroni correction alone drives its boot_lower
        # negative, without also requiring the CI to clear the magnitude bar.
        # (2026-08-06): raw min_effect, not effective_min_effect — _accept_decision
        # re-derives the same floor internally via null_churn_cases (min_flips), so
        # passing the effective value here would double-apply it (see the ceiling
        # identity in _accept_decision's docstring / the design note above it).
        accepted, reject_reason, net_flips, min_flips = _accept_decision(
            delta, boot_lower, n_paired, min_effect, null_churn_cases, gate_mode
        )
        # T7: per-backend breakdown from the correctness vectors already in hand —
        # fed back to the teacher next batch so a rejected edit's "why" (e.g. it
        # helped kv but poisoned vector) is visible, not just the aggregate delta.
        backend_delta = _backend_deltas(best_val_correct, cand_correct)
        print(f"[gate] incumbent={best_val_score:.1%}  candidate={new_val_score:.1%}  "
              f"delta={delta:+.2%}  min_effect={effective_min_effect:.2%}  "
              f"net_flips={net_flips:+d} (min_flips={min_flips:g})  "
              f"boot_lower={boot_lower:+.2%} (n_paired={n_paired})", flush=True)

        # Chain-correlation diagnostics (DIAGNOSTIC ONLY — they do not change the
        # accept condition above). The case-level bootstrap treats the ~6-7 cases of
        # one (backend, scenario) conversation chain as independent draws, so one
        # chain-level divergence can masquerade as a robust multi-case effect. Both
        # numbers come from the correctness vectors already in hand (no extra eval),
        # and both are recorded in training_history.json so a suspicious accept can
        # be identified after the fact instead of only from the aggregate delta.
        #
        # Deliberately NOT a hard veto (considered and rejected 2026-08-04):
        # `single_chain` measures SCOPE, not noise or reproducibility — a template
        # edit that only fires on one domain/scenario's tool-call pattern will
        # legitimately land 100% inside that one chain on every rerun, which is a
        # narrow-but-real fix, not overfitting. Auto-rejecting on scope alone would
        # let the training loop never adopt (and never build on top of) a genuine
        # narrow win. boot_lower/min_flips remain the sole accept authority; this
        # stays a loud, logged flag for post-hoc review (e.g. confirm on the 2x3
        # held-out test) rather than a gate.
        chain_conc = _chain_concentration(best_val_correct, cand_correct)
        _chain_rng = random.Random(seed * 100019 + batch_idx)
        _, boot_lower_chain, n_chains = _bootstrap_delta_lower_ci_clustered(
            best_val_correct, cand_correct, boot_n, boot_alpha_effective, _chain_rng
        )
        if chain_conc is not None:
            print(f"[gate] chain: {chain_conc['n_flips']} flips across "
                  f"{chain_conc['n_chains']}/{n_chains} chains, "
                  f"top={chain_conc['top_chain']} net={chain_conc['top_chain_net']:+d} "
                  f"of {chain_conc['net']:+d}; cluster boot_lower="
                  f"{boot_lower_chain:+.2%}", flush=True)
            if accepted and (
                chain_conc["single_chain"]
                or (chain_conc["net"] != 0
                    and abs(chain_conc["top_chain_net"]) >= abs(chain_conc["net"]))
            ):
                print(f"[gate] ⚠ CHAIN-CONCENTRATED ACCEPT: the entire net effect is "
                      f"carried by chain {chain_conc['top_chain']!r}. The case-level CI "
                      f"cannot distinguish that from {chain_conc['n_flips']} independent "
                      f"case wins — treat this accept as ONE observation and confirm on "
                      f"held-out test before trusting it.", flush=True)

        if accepted:
            best_val_score = new_val_score
            best_val_correct = cand_correct
            current_templates = candidate
            with open(best_path, "w") as f:
                json.dump(current_templates, f, indent=2)
            if boot_lower is None:
                _boot_note = ""
            elif gate_mode == "simple":
                _boot_note = f", boot_lower {boot_lower:+.2%} [diagnostic, not gating]"
            else:
                _boot_note = f", boot_lower {boot_lower:+.2%}"
            print(f"[gate] ✓ ACCEPTED gate_mode={gate_mode!r} "
                  f"(net_flips {net_flips:+d} >= min_flips {min_flips:g}{_boot_note})", flush=True)
        else:
            print(f"[gate] ✗ REJECTED gate_mode={gate_mode!r} ({reject_reason})", flush=True)
            # The gate pointed the store at the candidate's variant; on reject, rebuild
            # for the incumbent so the next batch's train/gate evals read the incumbent's
            # store, not the rejected candidate's. Unconditional now (any storage-affecting
            # edit could have switched the variant), but only meaningful with a snapshot
            # cache — the no-cache path re-runs prereqs inline in every _eval_batch, so
            # there is no stale store to restore. Cache HIT → cheap pointer reset.
            if evaluator._snapshot_cache_dir is not None:
                print(f"[gate] Restoring snapshot store to current templates ...", flush=True)
                evaluator.build_snapshot_store(train_prereqs, current_templates)

        _hist_entry = {
            "batch": batch_idx,
            "train_score": train_score,
            "val_score_before": val_score_before,
            "val_score_after": new_val_score,
            "delta": delta,
            "min_effect": effective_min_effect,
            "null_churn_cases": null_churn_cases,
            "net_flips": net_flips,
            "min_flips": min_flips,
            "gate_mode": gate_mode,
            "reject_reason": reject_reason,
            "boot_observed": boot_observed,
            "boot_lower": boot_lower,
            "boot_n": boot_n,
            "boot_n_paired": n_paired,
            "boot_alpha_effective": boot_alpha_effective,
            "boot_lower_chain": boot_lower_chain,
            "boot_n_chains": n_chains,
            "chain_concentration": chain_conc,
            "gate_n_seeds": gate_n_seeds,
            "backend_delta": backend_delta,
            "accepted": accepted,
            "edit_key": (
                key if gate_edit is None
                else _gate_edit_key(gate_kind, key, gate_new_value)
            ),
            "edit_text": new_text,
            "gate_telemetry": batch_gate_telemetry,
        }
        if gate_edit is not None:
            # "gate_mask" for a gate_enabled flip (unchanged from the one-shot probe, so
            # old histories/tests still read), "gate_remedy" for a remedy flip,
            # "gate_threshold" for a categorical threshold value.
            _hist_entry["edit_kind"] = _GATE_EDIT_KIND_BY_ARM[gate_kind]
        history.append(_hist_entry)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # W6: checkpoint after each gate decision so a killed run can --resume.
        with open(ckpt_path, "w") as f:
            json.dump({
                "next_batch": batch_idx + 1,
                "best_val_score": best_val_score,
                "best_val_correct": best_val_correct,
                "current_templates": current_templates,
                "rng_state": rng.getstate(),
                "sampler_state": sampler.state(),
                "history": history,
            }, f, indent=2)

    # Final history flush (covers batches that hit a continue before the gate write)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    n_accepted = sum(1 for h in history if h.get("accepted"))
    print(f"\n[done] Best val accuracy: {best_val_score:.1%}", flush=True)
    if n_accepted:
        print(f"[done] {n_accepted} edit(s) accepted — templates saved to {best_path}", flush=True)
    else:
        print(f"[done] 0 edits accepted this run — {best_path} written with the INITIAL "
              f"templates (a legitimate no-op outcome, not a crash)", flush=True)

    # best_path is written here UNCONDITIONALLY (not only inside `if accepted:` above)
    # so downstream consumers (the 2x2 ablation harness) can always open it — current_
    # templates equals the initial templates when nothing was ever accepted, same as
    # templates_final.json below.
    with open(best_path, "w") as f:
        json.dump(current_templates, f, indent=2)

    # Write final templates
    final_path = out_dir / "templates_final.json"
    with open(final_path, "w") as f:
        json.dump(current_templates, f, indent=2)

    return current_templates, best_val_score


# ── CLI ───────────────────────────────────────────────────────────────────────

def _write_run_manifest(out_dir: Path, args, initial_templates: Dict) -> None:
    """W6: write run_manifest.json (git SHA, resolved config, seeds, content hashes)
    so a run is reproducible from artifacts alone."""
    import hashlib as _hashlib
    import platform as _platform
    import subprocess as _subprocess

    def _git_sha():
        try:
            return _subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(_ANCHOROPT_ROOT),
                stderr=_subprocess.DEVNULL, text=True,
            ).strip()
        except Exception:
            return None

    def _file_sha(p):
        try:
            return _hashlib.sha256(Path(p).read_bytes()).hexdigest()
        except Exception:
            return None

    cfg_keys = ("seed", "epochs", "batches", "batch_size", "max_steps", "min_effect",
                "null_churn_cases", "gate_mode",
                "gate_seeds", "boot_n", "boot_alpha", "dup_threshold", "workers",
                "val_extra", "llm_model", "cases_dir", "initial_templates", "disable_gates",
                "enable_reroute", "enable_forced_retrieval", "enable_forced_key_search",
                "store_workers",
                "probe_gate_mask", "probe_gate_mask_key",
                "probe_gates", "gate_probe_every")
    manifest = {
        "git_sha": _git_sha(),
        "python": _platform.python_version(),
        "argv": sys.argv,
        "config": {k: getattr(args, k, None) for k in cfg_keys},
        "initial_templates_sha256": _hashlib.sha256(
            json.dumps(initial_templates, sort_keys=True).encode()).hexdigest(),
        "split_info_sha256": _file_sha(Path(args.cases_dir) / "split_info.json"),
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    _g = manifest["git_sha"]
    print(f"[manifest] git={_g[:12] if _g else 'n/a'} "
          f"tmpl={manifest['initial_templates_sha256'][:12]} → {out_dir / 'run_manifest.json'}",
          flush=True)


# submit_memory_train.sh always writes these into --out-dir BEFORE this script
# runs (runner.sh at submit time, lsf.log via bsub -oo), so their mere presence
# must not count as "existing results" when deciding whether --force/--resume
# is required — otherwise every fresh LSF submission would trip the guard.
_LAUNCH_BOOKKEEPING_FILES = {"runner.sh", "lsf.log"}


def _out_dir_has_existing_results(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    return any(p.name not in _LAUNCH_BOOKKEEPING_FILES for p in out_dir.iterdir())


def main():
    # Peek --model-config first so it can seed the argparse defaults below (a tiny
    # parser with add_help=False so required args aren't enforced during the peek).
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--model-config", default=None)
    _pre_args, _ = _pre.parse_known_args()
    model_cfg: Dict = {}
    if _pre_args.model_config:
        model_cfg = load_model_config(_pre_args.model_config)

    parser = argparse.ArgumentParser(description="AnchorOpt memory training — one-go.")
    parser.add_argument("--model-config", default=None,
                        help="Path to a model_config.json (see anchoropt/model_config.json). "
                             "Supplies defaults for the server/handler/decode args below; "
                             "explicit CLI flags still override it.")

    # Server
    srv = parser.add_argument_group("Server")
    # required=False (was True) so a --model-config can supply model_path; a manual
    # check below still errors when it's missing and the server isn't skipped.
    srv.add_argument("--model-path",   default=None, help="Model ID or local path")
    srv.add_argument("--port",         type=int,   default=8080)
    srv.add_argument("--num-gpus",     type=int,   default=1)
    srv.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    srv.add_argument("--max-model-len",type=int,   default=0, help="0=auto")
    srv.add_argument("--dtype",        default="bfloat16")
    srv.add_argument("--skip-server-setup", action="store_true")
    srv.add_argument("--vllm-url",     default=None)
    srv.add_argument("--served-model-name", default=None)

    # Handler
    hnd = parser.add_argument_group("Handler")
    hnd.add_argument("--handler-module", default="bfcl_eval.model_handler.local_inference.granite_4")
    hnd.add_argument("--handler-class",  default="Granite4PromptHandler")
    hnd.add_argument("--registry-name",  default="ibm-granite/granite-4.1-8b-prompting",
                     help="BFCL model registry key (used as registry_name in handler __init__)")

    # Cases
    cas = parser.add_argument_group("Cases")
    cas.add_argument("--cases-dir",
                     default=str(_ANCHOROPT_ROOT / "data"),
                     help="Directory containing memory_{train,val,test}_cases.json")
    cas.add_argument("--test-cases",
                     default=None,
                     help="Path to test cases JSON (default: <cases-dir>/memory_test_cases.json)")

    # Optimizer
    opt = parser.add_argument_group("Optimizer")
    opt.add_argument("--initial-templates",
                     default=str(_ANCHOROPT_ROOT / "policies" / "templates_cold_start.json"),
                     help="Cold-start policy the training loop seeds current_templates from "
                          "(:1406). Defaults to templates_cold_start.json — every template "
                          "slot starts empty and gate_default is unset (structural gates "
                          "stay enabled, self-seeding via gate_fallback()/seed_text() the "
                          "moment they fire). templates_initial.json's proven batch-0 text "
                          "is no longer the default: the cold-start mandate is that an edit "
                          "earns its way in through the accept gate, not that it starts "
                          "pre-applied. Pass templates_initial.json explicitly to reproduce "
                          "a pre-Phase-4 run.")
    opt.add_argument("--out-dir",
                     default=str(_ANCHOROPT_ROOT / "results" / "memory_train"))
    opt.add_argument("--epochs",     type=int, default=2,
                     help="T6: primary loop control — full even coverage of the train queries "
                          "per epoch. Effective batches = epochs × ceil(train_queries/batch_size).")
    opt.add_argument("--batches",    type=int, default=0,
                     help="T6: explicit batch-count override; >0 ignores --epochs "
                          "(back-compat for older submit scripts). 0 = derive from --epochs.")
    opt.add_argument("--batch-size", type=int, default=12)
    opt.add_argument("--max-steps",  type=int, default=20)  # W2 fidelity: match official MAXIMUM_STEP_LIMIT
    opt.add_argument("--workers",    type=int, default=1,
                     help="Parallel eval workers (1=sequential)")
    opt.add_argument("--seed",       type=int, default=42)
    opt.add_argument("--min-effect", type=float, default=0.01, dest="min_effect",
                     help="W4: minimum val improvement (fraction) required to ADOPT a candidate edit. "
                          "No accept-on-tie; guards against adopting within-noise changes.")
    opt.add_argument("--null-churn-cases", type=float, default=2.0, dest="null_churn_cases",
                     help="T8/(2026-08-06): floor on min_flips, the integer net-case-flip count "
                          "required to adopt a candidate (max of ceil(min_effect * n_scored) and this). "
                          "Keeps a 1-2 case fluke from clearing the bar even at a tiny --min-effect. "
                          "Lowered from 3.0: f_null=11 (its original justification) was superseded by "
                          "the vLLM prefix-caching fix, which measured f_null=0. Raise only if a "
                          "re-measured noise floor exceeds the current default.")
    opt.add_argument("--gate-mode", default="simple", dest="gate_mode", choices=list(_GATE_MODES),
                     help="(2026-08-06): accept-gate regime. 'simple' (default) = SkillOpt-style "
                          "greedy candidate>incumbent, magnitude only (net_flips>=min_flips), guarded "
                          "by --null-churn-cases. 'bootstrap' = the original stricter regime, ALSO "
                          "requiring boot_lower>0 at a Bonferroni-corrected alpha — restores the "
                          "pre-2026-08-06 behavior exactly. boot_lower is always computed and recorded "
                          "in training_history.json in both modes (diagnostic-only under 'simple').")
    opt.add_argument("--gate-seeds", type=int, default=1, dest="gate_seeds",
                     help="W4: number of paired seeds to average the candidate over at the gate "
                          "(1 = deterministic single eval; >1 for temperature>0 stochastic regimes).")
    opt.add_argument("--boot-n", type=int, default=50000, dest="boot_n",
                     help="T5: paired case-bootstrap resamples for the deterministic accept-gate CI "
                          "(0 disables the CI test, keeping only the min_flips band).")
    opt.add_argument("--boot-alpha", type=float, default=0.05, dest="boot_alpha",
                     help="T5: one-sided lower-CI level for the accept gate, Bonferroni-corrected by "
                          "dividing by total_batches before use; adopt only if the delta's lower bound "
                          "at the corrected level clears zero (a sign check, not a second magnitude bar "
                          "against min_flips/effective_min_effect).")
    opt.add_argument("--dup-threshold", type=float, default=0.8, dest="dup_threshold",
                     help="T5: difflib similarity above which a re-proposed edit for a key already "
                          "rejected is skipped without a gate eval.")
    opt.add_argument("--probe-gate-mask", action="store_true", dest="probe_gate_mask",
                     help="Fire a one-shot deterministic ablation probe at batch 0: propose disabling "
                          "--probe-gate-mask-key via the gate_enabled mask and score it through the "
                          "existing accept gate. No LLM call; fires at most once ever (checkpoint-safe).")
    opt.add_argument("--probe-gate-mask-key", default="on_premature_idk", dest="probe_gate_mask_key",
                     help="Registry gate key the --probe-gate-mask probe targets (default: on_premature_idk).")
    opt.add_argument("--probe-gates", action="store_true", dest="probe_gates",
                     help="Search the GATE SPACE alongside template edits: walk an enumerated queue "
                          "of single-lever arms, one per scheduled batch, each scored through the "
                          "same accept gate as a template edit. BOOLEAN arms: the three remedy flags "
                          "(enable_forced_retrieval, enable_reroute, enable_forced_key_search) plus "
                          "a gate_enabled mask per structural gate. CATEGORICAL (multi-valued) arms: "
                          "blob_pressure_threshold {4000,6000,8000,10000} and loop_repeat_threshold "
                          "{1,2,3} — each consumes ONE scheduled batch PER VALUE, greedily keeping "
                          "whichever value wins as the baseline for the next. Budget accordingly: "
                          "the full queue is ~13 arms, so a 20-batch run needs --gate-probe-every 1 "
                          "to reach the threshold axes at all. Supersedes --probe-gate-mask (which "
                          "fires one arm, once, ever). Accepted arms are written into "
                          "templates_best.json in the shape base_handler.py already reads "
                          "(gate_enabled section for masks; top-level bool/int for remedies and "
                          "thresholds), so they transfer to eval for free.")
    opt.add_argument("--gate-probe-every", type=int, default=2, dest="gate_probe_every",
                     help="Stride for --probe-gates: a gate arm is proposed on batches 0, N, 2N, ... "
                          "until the queue is exhausted; every other batch goes to the teacher. "
                          "1 = front-load all arms; 0 = disable. Default 2 (arms and template edits "
                          "share the batch budget) — note that with multi-valued arms the queue is "
                          "now longer than 10, so the default stride will NOT reach the end of it in "
                          "a 20-batch run.")
    opt.add_argument("--llm-model",
                     default=os.getenv("OPTIMIZER_LITELLM_MODEL", "openai/aws/claude-opus-4-8"))
    opt.add_argument("--eval-only", action="store_true",
                     help="Skip training; just build snapshot store and eval --eval-splits")
    opt.add_argument("--eval-splits", default="test,val",
                     help="Comma-separated splits to eval in --eval-only mode (default: test,val)")
    opt.add_argument("--disable-gates", action="store_true",
                     help="Disable G1 and D3 gates (diagnostic only).")
    opt.add_argument("--enable-reroute", action="store_true",
                     help="Run-level DEFAULT for the opt-in core-full reroute remedy (Phase 3): "
                          "re-dispatch the fact via archival_memory_add instead of only "
                          "reprompting. Default off. This flag had no counterpart here before, so "
                          "the remedy was dormant in training even as a global toggle; with "
                          "--probe-gates the search can flip it per candidate regardless.")
    opt.add_argument("--enable-forced-retrieval", action="store_true",
                     help="Turn ON the opt-in forced archival retrieval remedy (Phase 6): when "
                          "the model is about to give up without searching archival memory this "
                          "turn, dispatch the search deterministically and show it the result "
                          "instead of only reprompting. Default off, matching the "
                          "byte-identical-stock guarantee.")
    opt.add_argument("--enable-forced-key-search", action="store_true",
                     dest="enable_forced_key_search",
                     help="Run-level DEFAULT for the opt-in forced key-search remedy (Phase 7): "
                          "after a KV retrieve fails with 'Key not found', run "
                          "archival_memory_list_keys() on the model's behalf and show it the real "
                          "key names instead of only telling it to look them up. Default off; "
                          "--probe-gates searches the opposite direction from whatever this sets.")
    opt.add_argument("--val-extra", type=int, default=0,
                     help="Move this many stratified query cases from train into val before training "
                          "(enlarges val set to reduce overfitting; 0 = no change)")
    opt.add_argument("--store-workers", type=int, default=1, dest="store_workers",
                     help="W1 determinism: prereq store builds are SERIAL by default (1) — "
                          "concurrent vLLM continuous batching is not byte-stable even at "
                          "temperature 0, and the store is the state every decisional eval "
                          "reads. <=0 restores the old uncapped-parallel behavior; a positive "
                          "N caps the pool at N chains.")
    opt.add_argument("--force", action="store_true",
                     help="W6: overwrite a non-empty --out-dir instead of refusing (protects baselines).")
    opt.add_argument("--resume", action="store_true",
                     help="W6: resume training from <out-dir>/checkpoint.json if present.")

    if model_cfg:
        apply_model_config(parser, model_cfg)
    args = parser.parse_args()

    if not args.skip_server_setup and not args.model_path:
        parser.error("--model-path is required unless --skip-server-setup is set")

    # ── Server ───────────────────────────────────────────────────────────────
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
    handler_mod = args.handler_module or "bfcl_eval.model_handler.local_inference.granite_4"
    mod = __import__(handler_mod, fromlist=[args.handler_class])
    HandlerCls = getattr(mod, args.handler_class)
    # Granite4PromptHandler requires registry_name and is_fc_model in addition
    # to model_name and temperature.
    try:
        handler = HandlerCls(
            model_name=served_name,
            temperature=0.001,
            registry_name=args.registry_name,
            is_fc_model=False,
        )
    except TypeError:
        # Fallback for handlers that only take model_name + temperature
        handler = HandlerCls(model_name=served_name, temperature=0.001)

    # ── Load cases ────────────────────────────────────────────────────────────
    cases_dir = Path(args.cases_dir)

    # ── Evaluator ─────────────────────────────────────────────────────────────
    from anchoropt.memory_evaluator import MemoryAnchorOptEvaluator
    evaluator = MemoryAnchorOptEvaluator(
        handler=handler,
        vllm_url=vllm_url,
        served_model_name=served_name,
        max_steps_per_turn=args.max_steps,
        snapshot_cache_dir=cases_dir / "snapshot_cache",
        disable_gates=args.disable_gates,
        enable_reroute=args.enable_reroute,
        enable_forced_retrieval=args.enable_forced_retrieval,
        enable_forced_key_search=args.enable_forced_key_search,
        store_workers=args.store_workers,
        seed=args.seed,  # W1: fixed decode seed for reproducible gate evals
        # Decode params from --model-config (defaults reproduce the inline literals).
        **decode_params(model_cfg),
    )
    with open(cases_dir / "memory_train_cases.json") as f:
        train_cases = json.load(f)

    # ── Load templates ────────────────────────────────────────────────────────
    with open(args.initial_templates) as f:
        initial_templates = json.load(f)

    # Transfer-contract fix: bake the run-level --enable-reroute/-forced-retrieval/
    # -forced-key-search CLI defaults into initial_templates itself, before it
    # becomes current_templates (:1387) or reaches templates_best.json/
    # templates_final.json. Previously these CLI flags only reached the
    # evaluator via a constructor-level default-parameter fallback inside
    # remedy_enabled(); an explicit key was written to the exported policy only
    # if the gate-arm search happened to flip that remedy, so a run started
    # with e.g. --enable-reroute but whose search never touched that arm would
    # export a policy with the key silently absent — a leaderboard/eval run
    # loading that JSON with no matching flag would then resolve it as off,
    # diverging from what was actually measured throughout training. An
    # explicit key already present in the loaded policy still wins.
    for _flag_name, _flag_val in (
        ("enable_reroute", args.enable_reroute),
        ("enable_forced_retrieval", args.enable_forced_retrieval),
        ("enable_forced_key_search", args.enable_forced_key_search),
    ):
        if _flag_val and _flag_name not in initial_templates:
            initial_templates[_flag_name] = True

    out_dir = Path(args.out_dir)
    # W6: refuse to clobber a non-empty existing out-dir (protects baselines / prior
    # runs) unless --force or --resume. eval-only writes are additive, so exempt them.
    if _out_dir_has_existing_results(out_dir) and not (args.force or args.resume or args.eval_only):
        print(f"[error] --out-dir {out_dir} exists and is non-empty. Use a new dir, "
              f"--force to overwrite, or --resume to continue.", flush=True)
        if server:
            server.shutdown()
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        _write_run_manifest(out_dir, args, initial_templates)

    global _detail_log
    _detail_log = open(out_dir / "case_detail.log", "w", buffering=1)

    # ═══════════════════════════════════════════════════════════════════════════
    # EVAL-ONLY MODE: build snapshot store, eval requested splits, exit
    # ═══════════════════════════════════════════════════════════════════════════
    if args.eval_only:
        try:
            requested = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
            # "all" expands to train+val+test combined in one pass. Each split file is
            # self-contained: train/val share the same 96 prereq chains; test's 15 are
            # fully disjoint. So "all" needs the DEDUPED prereq UNION (not just train
            # prereqs — that would score test/val queries against a store missing their
            # setup), while a single split only needs its own prereqs.
            if requested == ["all"]:
                all_prereqs = prereq_union(cases_dir)
                all_queries = []
                for s in ("train", "val", "test"):
                    with open(cases_dir / f"memory_{s}_cases.json") as f:
                        queries = split_queries(json.load(f))
                    all_queries.extend(queries)
                    print(f"[eval]   {s:5s}: {len(queries)} queries", flush=True)
                print(f"\n[eval] Running all-splits eval ({len(all_queries)} queries total) ...", flush=True)
                # cases = prereqs + queries so the no-cache path runs prereqs inline;
                # with a cache, _score_split builds all_prereqs once and scores queries-only.
                acc, results = _score_split(
                    evaluator, all_prereqs, all_prereqs + all_queries,
                    initial_templates, "all", args.workers,
                )
                cat_acc = _per_category_accuracy(results)
                print(f"\n[eval] === all Results ===", flush=True)
                print(f"[eval] Overall: {acc:.1%}", flush=True)
                for cat, a in cat_acc.items():
                    print(f"[eval]   {cat:10s}: {a:.1%}", flush=True)
                out = {
                    "split": "all",
                    "templates": args.initial_templates,
                    "accuracy": acc,
                    "per_category": cat_acc,
                    "results": results,
                }
                out_path = out_dir / "eval_all_results.json"
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)
                print(f"[eval] Saved to {out_path}", flush=True)
            else:
                for split in requested:
                    split_path = cases_dir / f"memory_{split}_cases.json"
                    if not split_path.exists():
                        print(f"[eval] No {split} cases at {split_path} — skipping", flush=True)
                        continue
                    with open(split_path) as f:
                        split_cases = json.load(f)
                    prereqs = split_prereqs(split_cases)
                    print(f"\n[eval] Running {split}-set ({len(split_cases)} entries) ...", flush=True)
                    acc, results = _score_split(
                        evaluator, prereqs, split_cases, initial_templates, split, args.workers
                    )
                    cat_acc = _per_category_accuracy(results)
                    print(f"\n[eval] === {split} Results ===", flush=True)
                    print(f"[eval] Overall: {acc:.1%}", flush=True)
                    for cat, a in cat_acc.items():
                        print(f"[eval]   {cat:10s}: {a:.1%}", flush=True)
                    out = {
                        "split": split,
                        "templates": args.initial_templates,
                        "accuracy": acc,
                        "per_category": cat_acc,
                        "results": results,
                    }
                    out_path = out_dir / f"eval_{split}_results.json"
                    with open(out_path, "w") as f:
                        json.dump(out, f, indent=2)
                    print(f"[eval] Saved to {out_path}", flush=True)
        finally:
            if server:
                server.shutdown()
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING MODE
    # ═══════════════════════════════════════════════════════════════════════════
    with open(cases_dir / "memory_val_cases.json") as f:
        val_cases = json.load(f)
    print(
        f"[init] Train: {len(train_cases)} entries  Val: {len(val_cases)} entries",
        flush=True,
    )

    # ── Load test cases ───────────────────────────────────────────────────────
    test_cases_path = Path(args.test_cases) if args.test_cases else (
        cases_dir / "memory_test_cases.json"
    )
    test_cases: Optional[List[Dict]] = None
    if test_cases_path.exists():
        with open(test_cases_path) as f:
            test_cases = json.load(f)
        print(f"[init] Test:  {len(test_cases)} entries  ({test_cases_path})", flush=True)
    else:
        print(f"[init] No test cases found at {test_cases_path} — skipping test eval", flush=True)

    try:
        best_templates, best_val_score = train(
            evaluator=evaluator,
            train_cases=train_cases,
            val_cases=val_cases,
            initial_templates=initial_templates,
            out_dir=out_dir,
            batches=args.batches,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            llm_model=args.llm_model,
            val_extra=args.val_extra,
            min_effect=args.min_effect,
            null_churn_cases=args.null_churn_cases,
            gate_mode=args.gate_mode,
            gate_n_seeds=args.gate_seeds,
            boot_n=args.boot_n,
            boot_alpha=args.boot_alpha,
            dup_threshold=args.dup_threshold,
            workers=args.workers,
            resume=args.resume,
            probe_gate_mask=args.probe_gate_mask,
            probe_gate_mask_key=args.probe_gate_mask_key,
            probe_gates=args.probe_gates,
            gate_probe_every=args.gate_probe_every,
        )

        # ── Test eval (post-training) ─────────────────────────────────────────
        if test_cases:
            print("\n[test] Running test-set evaluation on best templates ...", flush=True)
            # W5 (train-side): test's own prereq chains live only in test_cases and were
            # never run through build_snapshot_store (only train_prereqs was, for training),
            # so the store must be (re)built from test's OWN prereqs before scoring. With a
            # snapshot cache, _score_split builds them once and scores test QUERIES only;
            # with no cache it falls through to the full-list path (prereqs inline).
            # RCA 2026-08-03: this eval must stay SERIAL (workers=1), matching the decisional
            # val/gate evals above and the standalone 2x2 harness (run_memory_eval.py defaults
            # --workers 1) — using args.workers (parallel, exploratory-batch setting) here made
            # this number incomparable to the 2x2's trained_gates arm on byte-identical templates
            # (observed: 4-5 case / ~5% divergence from parallel-batching decode nondeterminism,
            # not a real difference).
            test_prereqs = split_prereqs(test_cases)
            test_acc, test_results = _score_split(
                evaluator, test_prereqs, test_cases, best_templates, "test", 1
            )
            cat_acc = _per_category_accuracy(test_results)
            print(f"[test] Overall accuracy: {test_acc:.1%}", flush=True)
            for cat, acc in cat_acc.items():
                print(f"[test]   {cat:10s}: {acc:.1%}", flush=True)
            test_out = {
                "accuracy": test_acc,
                "per_category": cat_acc,
                "val_accuracy": best_val_score,
                "results": test_results,
            }
            test_path = out_dir / "test_results.json"
            with open(test_path, "w") as f:
                json.dump(test_out, f, indent=2)
            print(f"[test] Saved to {test_path}", flush=True)

    finally:
        if server:
            server.shutdown()


if __name__ == "__main__":
    main()
