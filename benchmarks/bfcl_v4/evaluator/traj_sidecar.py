"""Stage 0 of SIGNAL_POLICY_PLAN: persist raw episode trajectories to disk.

Both trajectory-producing call sites reduce `traj` to a handful of booleans and
drop the step records:

  - `_eval_batch` (run_memory_train.py) keeps ~12 `any(...)` summaries;
  - `_run_prereq_build` (memory_evaluator.py) keeps 3 integer counters.

The evidence a human used to find G1 — that `core_memory_clear` was called 66
times, 97 of them within 2 steps of a "core is full" tool error — lives in the
per-step `decoded` / `tool_results` strings, which neither summary retains. This
module writes them out so signal mining has a corpus.

Enabled by setting `ANCHOROPT_TRAJ_DIR`; a no-op otherwise, so it cannot change
the behaviour of any existing run. Writes one JSON per episode:

    <ANCHOROPT_TRAJ_DIR>/<phase>/<case_id>.json

⚠️ Prereq episodes matter more than query episodes here. ~83% of kv/vector
failures originate in the *storage* phase and surface in a different episode, so
a query-only corpus would not have found G1. `_run_prereq_build` is therefore
instrumented too, and its episodes are tagged phase="prereq".
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

# Step fields worth keeping. Deliberately explicit rather than dumping the whole
# record: `assistant_response` holds full model text, which would balloon the
# corpus for no mining value beyond what `decoded` already gives.
_STEP_FIELDS = (
    "turn", "step", "status", "decoded", "tool_results", "error",
    "error_signal", "error_streak", "injection_key", "injection",
    "loop_signal", "idk_fallback_signal", "failed_search_streak",
    "cell_key", "is_retrieval_success",
    # Cost fields. Already measured upstream (`latency` at the model call; `input_token` /
    # `output_token` from the handler's own usage block) and previously dropped by this
    # whitelist. None means NOT MEASURED, never 0.
    "latency", "input_token", "output_token",
)

_write_lock = threading.Lock()
_dir_cache: set = set()


def traj_dir() -> Optional[Path]:
    """Target directory, or None when the sidecar is disabled."""
    raw = os.environ.get("ANCHOROPT_TRAJ_DIR", "").strip()
    return Path(raw) if raw else None


def enabled() -> bool:
    return traj_dir() is not None


def _safe_name(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", case_id)[:180]


def _backend(case_id: str) -> str:
    m = re.match(r"memory_(kv|vector|rec_sum)_", case_id or "")
    return m.group(1) if m else "unknown"


def dump_episode(
    case: Dict,
    traj: List[Dict],
    phase: str,
    checker: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> Optional[Path]:
    """Persist one episode's trajectory. Returns the path written, or None.

    Never raises: a sidecar failure must not fail the eval that produced the
    data. Any error is reported once and swallowed.
    """
    base = traj_dir()
    if base is None:
        return None

    case_id = str(case.get("id", "unknown"))
    try:
        out_dir = base / phase
        # mkdir is cheap but not free at n=300+; cache which dirs exist.
        if out_dir not in _dir_cache:
            out_dir.mkdir(parents=True, exist_ok=True)
            _dir_cache.add(out_dir)

        steps = []
        for s in traj:
            rec = {k: s[k] for k in _STEP_FIELDS if k in s}
            # Gate/telemetry flags are dynamic (one per registry gate), so copy
            # any remaining truthy *_gate key rather than enumerating them.
            for k, v in s.items():
                if k.endswith("_gate") and v:
                    rec[k] = v
            # Action-family payloads: a remedy that DOES something records what it
            # did (accepted? why? how much did it compress?), and those keys match
            # neither _STEP_FIELDS nor the *_gate suffix rule. Dropping them makes a
            # gate that fired and acted indistinguishable from one that fired and
            # bailed -- the G1 `continue` failure mode again, and it cost a smoke
            # run: 18 real transform firings all reported ok=None/reason=None.
            # Copy every family payload key, INCLUDING falsey ones: transform_ok
            # False is the single most informative value here (validation rejected
            # the rewrite), so a truthiness filter would discard exactly the
            # evidence the checker needs.
            for k, v in s.items():
                # `constraint_state` and `constraint_state_error` are live-state captures,
                # not action payloads, so they are named explicitly rather than folded into
                # _FAMILY_PREFIXES. Both must be copied: a capture that RAISED has to stay
                # distinguishable from one that never fired.
                if k.startswith(_FAMILY_PREFIXES) or k.startswith("constraint_state"):
                    rec[k] = v
            steps.append(_stringify(rec))

        payload = {
            "case_id": case_id,
            "phase": phase,
            "backend": _backend(case_id),
            "n_steps": len(traj),
            "steps": steps,
        }
        if checker is not None:
            payload["outcome"] = {
                "valid": checker.get("valid"),
                "error_type": checker.get("error_type", ""),
                "error_message": checker.get("error_message", ""),
                "force_quit": bool(checker.get("force_quit", False)),
            }
        if extra:
            payload.update(extra)

        path = out_dir / f"{_safe_name(case_id)}.json"
        with _write_lock:
            with open(path, "w") as f:
                json.dump(payload, f, indent=1, default=str)
        return path
    except Exception as e:  # pragma: no cover - defensive
        _warn_once(f"[traj_sidecar] failed on {case_id}: {type(e).__name__}: {e}")
        return None


# Action-family payload prefixes. DERIVED from the action space plus the anchor procedures,
# not hand-listed: `capacity_repair_*` was added as a fourth field family and silently dropped
# because it matched none of the literals here, so a repair that FIRED recorded empty
# replaced_call / substituted_call / shed_chars. That is the same defect as the three before it
# (constraint_state, transform_ok, and the reroute payload), and the reason it keeps recurring is
# that the list is authored by hand while the fields are added elsewhere.
#
# Any new anchor mechanism must either use one of these prefixes or add its own HERE, in the same
# commit as the field. `test_sidecar_preserves_anchor_payloads` pins it.
_FAMILY_PREFIXES = (
    "transform_", "suppress_", "reroute_", "reprompt_",
    "capacity_repair_",   # post-execution aggregate-container repair (N0-A1)
    "preempt_",           # pre-generation anchors, for when a miner reaches that column
    # A5. FIFTH occurrence of this defect class: the gate set low_similarity_value and
    # low_similarity_n_reads, the whitelist did not carry them, and the values were silently
    # dropped while the *_gate boolean survived via another path. Firings counted, diagnostics
    # did not -- so the mechanism could be confirmed but not EXPLAINED. Prior instances:
    # constraint_state, capacity_repair_* (twice), and a tuple-vs-string concat that killed whole
    # episode dumps. Any new step_record key needs a prefix here on the same commit that adds it.
    "low_similarity_",
    # B1 and E1. Added after the SIXTH and SEVENTH occurrences of this defect: futility_kind was
    # dropped (so B1's F1-vs-F2 split is unrecoverable) and every e1_* key was dropped (so E1 v1's
    # "0 declines" could not be distinguished from "0 recorded"). Observability is a precondition for
    # an experiment, not a post-hoc convenience.
    "futility_",
    "e1_",
    # KV key-preserving observer. Added on the SAME COMMIT as the keys, per the rule above.
    "kv_obs",
    # C2. Same-commit rule again: the gate sets kv_redundant_* on step_record.
    "kv_redundant",
    # N0-R1. Same-commit rule: the controller sets recovery_txn_* on step_record.
    "recovery_txn",
    # CANONICAL ACCEPTANCE EVIDENCE. Criteria 3 and 4 are adjudicated over a fixed vocabulary
    # (mechanism_requested / mechanism_verified / mechanism_declined_with_reason /
    # mechanism_refused_by_guard, and the three safety counters). A PREFIX is used rather than a hand
    # listed set on purpose: the note above records that this allowlist has silently dropped
    # intervention telemetry at least seven times, and states the reason plainly -- the list is
    # authored by hand while the fields are added elsewhere. A prefix removes the coupling, so a
    # counter added to `mechanism_evidence.FAMILY_READINGS` cannot be dropped by forgetting this file.
    "mechanism_",
    # The safety counters do not share a prefix with each other, and renaming them would break the
    # core contract they satisfy, so these three are named. They are the only three; criterion 4's
    # contract is closed.
    "clears_added", "information_losing_removes", "verified_relocations",
)


def _stringify(rec: Dict) -> Dict:
    """Coerce non-JSON-native step values (tool result objects) to strings.

    `tool_results` and `decoded` carry whatever the handler produced; mining only
    ever substring-matches them, so a faithful repr is sufficient and cannot
    fail to serialise.
    """
    out = {}
    for k, v in rec.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif k in _STRUCTURED_FIELDS and _json_safe(v):
            # STRUCTURED payloads keep their shape. A faithful repr is fine for
            # tool_results/decoded because mining only substring-matches those, but a
            # consumer that READS FIELDS needs a dict, and `str(...)` on a list of dicts
            # yields Python repr (single quotes) that json.loads cannot parse. Verified
            # necessary: the constraint_state capture round-tripped as a repr string and
            # every downstream field lookup would have silently returned nothing.
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (str, int, float, bool, type(None))) else str(x)
                      for x in v]
        else:
            out[k] = str(v)
    return out


# Fields whose VALUES are read structurally downstream, not substring-matched.
_STRUCTURED_FIELDS = ("constraint_state",)


def _json_safe(v) -> bool:
    """True when `v` serialises as-is, so preserving shape cannot break the dump."""
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


_warned: set = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(msg, flush=True)
