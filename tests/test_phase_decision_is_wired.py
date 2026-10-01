"""The phase decision must be made by AnchorOpt's rule, not by a human reading round output.

`phase_switch.decide` and `block_loop.residual_phase` both existed, were tested, and NOTHING CALLED
THEM -- so continue/expand/stop was still a person reading the output of a loop whose whole claim is
self-termination. These tests pin that the driver now calls it and records the verdict.

The distinction being protected is the reason the module exists: "nothing survives the gates" is NOT
"the work is done". Collapsing EXPAND_ATTRIBUTION into STOP declares a family scientifically exhausted
when in fact no declared signal can see it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "self_evolve_cycle2.py"
INCUMBENT = REPO / "results" / "h0_native" / "nativectl_rec_sum_train" / "run"
DIAGNOSES = REPO / "results" / "h0_rs_pe" / "diagnoses.json"

pytestmark = pytest.mark.skipif(
    not (INCUMBENT / "eval_train_results.json").exists() or not DIAGNOSES.exists(),
    reason="raw H0 rec_sum artifacts absent")


def _round(tmp_path: Path) -> dict:
    out = tmp_path / "r"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(DIAGNOSES, out / "diagnoses.json")
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--incumbent", str(INCUMBENT), "--out", str(out),
         "--cell", "rec_sum", "--phase", "prereq"],
        capture_output=True, text=True, cwd=str(REPO), timeout=900)
    assert proc.returncode == 0, proc.stderr[-1500:]
    return json.loads((out / "cycle2.json").read_text())


def test_the_round_RECORDS_a_phase_verdict(tmp_path):
    rec = _round(tmp_path)
    pd = rec.get("phase_decision")
    assert pd, "the round made no phase decision -- the switch is not wired"
    assert pd["phase"] in {"CONTINUE_POLICY", "EXPAND_ATTRIBUTION", "STOP"}, pd["phase"]


def test_every_THRESHOLD_is_shown_not_just_the_verdict(tmp_path):
    """A verdict without its gates is unauditable: the deferrals are where the criteria become legible."""
    rec = _round(tmp_path)
    reasons = " ".join(rec["phase_decision"]["reasons"])
    assert "S1-ok" in reasons or "S1-STOP" in reasons, reasons[:300]
    assert "S2-ok" in reasons or "S2-STOP" in reasons, reasons[:300]


def test_the_rule_is_IMPORTED_not_reimplemented_in_the_driver():
    """A second copy of a threshold is how two parts of a loop start disagreeing."""
    src = DRIVER.read_text()
    assert "residual_phase" in src, "the driver does not call core's phase rule"
    for literal in ("0.10", "0.50"):
        assert f"S1_COVERAGE_FLOOR = {literal}" not in src and \
               f"S2_SATURATION_CEIL = {literal}" not in src, \
            "the driver re-declares a frozen threshold instead of importing the rule"


def test_EXPAND_and_STOP_are_distinct_verdicts():
    """Directly on the rule: an inexpressible-dominated residual must not report STOP."""
    from anchoropt.learning.block_loop import residual_phase
    from anchoropt.runtime import ResidualDiagnosis

    def _d(decision: str, mech: str = "m"):
        return ResidualDiagnosis(case_id="c", mechanism=mech, evidence="e",
                                 consequential_decision=decision,
                                 proposed_behavior_change="p")

    # Nothing expressible, and the inexpressible mass dominates -> the REPRESENTATION is the limit.
    many = [_d(f"unreachable condition {i // 4}") for i in range(12)]
    phase, reasons = residual_phase(many, expressible=lambda _d: False)
    assert phase == "EXPAND_ATTRIBUTION", (phase, reasons[-4:])
    assert phase != "STOP", "an unreachable residual must never be reported as done"
