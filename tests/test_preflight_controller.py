"""The pre-GPU preflight: every check here blocked a real wasted job, or would have.

The failure this prevents: an arm whose executor silently ignores its action runs as the CONTROL, so
its paired result is a measured zero against an intervention that never happened -- indistinguishable
from a real NO_BENEFIT on a results table. Six such defects were found in one session, five before any
GPU was spent, and the check that found them was always "will this EXECUTE what it claims", never
"does it install".
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "preflight_controller.py"
RUN = REPO / "results" / "h0_native" / "nativectl_rec_sum_train" / "run"

pytestmark = pytest.mark.skipif(
    not (RUN / "eval_train_results.json").exists(), reason="raw H0 rec_sum artifacts absent")


def _run(spec: dict, tmp_path: Path, *extra):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    out = tmp_path / "pf.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--spec", str(p), "--run", str(RUN), "--json", str(out),
         *[str(x) for x in extra]],
        capture_output=True, text=True, cwd=str(REPO), timeout=600)
    rec = json.loads(out.read_text()) if out.exists() else {}
    return proc, rec


def test_an_unsupplied_field_BLOCKS_the_job(tmp_path):
    """The `error_kind` class: a predicate missing one field answers False for that reason alone."""
    proc, rec = _run({"name": "clear_proposed_at_capacity", "locus": "post_generation_pre_exec",
                      "action": "suppress", "phase": "prereq",
                      "predicate": {"declared_signal": "clear_proposed_at_capacity"}}, tmp_path)
    assert proc.returncode == 1, "a controller that cannot fire must not reach GPU"
    assert not rec["passed"]
    assert any("unsupplied" in f for f in rec["failures"])


def test_a_signal_that_FIRES_passes_supply_and_fireability(tmp_path):
    """append_would_exceed_cap at post_execution: 0 firings before error_kind was supplied, 278 after."""
    proc, rec = _run({"name": "append_would_exceed_cap", "locus": "post_execution",
                      "action": "reprompt", "phase": "prereq",
                      "predicate": {"declared_signal": "append_would_exceed_cap"}}, tmp_path)
    assert rec["checks"]["supply"]["missing"] == [], rec["checks"]["supply"]
    assert rec["checks"]["fireability"]["fired"] > 0, "the supplied field must make it fire"
    assert rec["checks"]["negative"]["not_fired"] > 0, "it must also discriminate"


def test_an_UNDECLARED_phase_blocks(tmp_path):
    """Defaults to `query`, so a write-side controller is installed, reached, and SKIPPED."""
    _proc, rec = _run({"name": "append_would_exceed_cap", "locus": "post_execution",
                       "action": "reprompt",
                       "predicate": {"declared_signal": "append_would_exceed_cap"}}, tmp_path)
    assert any("phase" in f for f in rec["failures"]), rec["failures"]


def test_a_MULTI_executor_cell_demands_an_explicit_capability_id(tmp_path):
    """post_generation_pre_exec/reprompt has a live injector AND a disabled telemetry-only sibling.

    `capability_id` is emitted but NOT read by live dispatch, so a spec that omits it here is
    measuring whichever executor resolves first -- and one of the two does nothing at all.
    """
    _proc, rec = _run({"name": "no_tool_call_at_all", "locus": "post_generation_pre_exec",
                       "action": "reprompt", "phase": "query",
                       "predicate": {"declared_signal": "no_tool_call_at_all"}}, tmp_path)
    assert rec["checks"]["capability"]["n_executors"] > 1, "fixture assumes the multi-executor cell"
    assert any("names NONE" in f or "capability_id" in f for f in rec["failures"]), rec["failures"]


def test_naming_a_DISABLED_executor_blocks(tmp_path):
    """The inert sibling writes telemetry and never injects: two arms differing only in `instruction`
    produced byte-identical trajectories across 13/13 storage episodes."""
    _proc, rec = _run({"name": "no_tool_call_at_all", "locus": "post_generation_pre_exec",
                       "action": "reprompt", "phase": "query",
                       "capability_id": "write_instruction_to_step_record",
                       "predicate": {"declared_signal": "no_tool_call_at_all"}}, tmp_path)
    assert any("DISABLED" in f or "unbound" in f for f in rec["failures"]), rec["failures"]


def test_a_SYNTHESIZED_predicate_is_evaluated_STRUCTURALLY(tmp_path):
    """A signal from Phi expansion has no host evaluator outside the process that expanded it.

    `EXPANDED_SIGNALS` is in-memory, so looking the name up in a fresh interpreter raises KeyError on
    every state -- which reported all 37 arms of a real expansion round as "the predicate RAISED on
    every state", a fact about the preflight script and not about the arms. The spec carries its
    predicate structurally for exactly this reason, and `SpecPredicate` is the evaluator the live hook
    uses, so preflight what will actually be installed.
    """
    spec = {"name": "proposed_payload_chars_gt_252p0__and__step_index_gt_0p0",
            "locus": "post_generation_pre_exec", "action": "suppress", "operator": "suppress",
            "eta": {"suppressed_operation": "x"}, "phase": "prereq",
            "capability_id": "post_generation_pre_exec/suppress",
            "predicate": {"all": [{"field": "proposed_payload_chars", "op": "gt", "value": 252.0},
                                  {"field": "step_index", "op": "gt", "value": 0.0}]}}
    proc, rec = _run(spec, tmp_path)
    assert rec["checks"]["supply"]["missing"] == [], "the synthesized fields ARE supplied"
    assert rec["checks"]["fireability"]["eval_errors"] == 0, \
        "a structurally-carried predicate must evaluate without the host knowing its name"
    assert rec["checks"]["fireability"]["fired"] > 0
    assert rec["checks"]["negative"]["not_fired"] > 0
    assert proc.returncode == 0, rec["failures"]
