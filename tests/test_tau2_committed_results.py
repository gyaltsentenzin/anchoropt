"""The committed round-1 artifacts reproduce the report, and every committed anchor still installs.

`benchmarks/tau2/rounds/SELFTEACH_R1/` holds the arm artifacts behind docs/TAU2_SELFTEACH_ROUND1.md.
These tests make that claim checkable rather than asserted:

  * recomputing `summary.json` from the committed arm directories gives exactly the committed file, so
    no reported number can drift from the artifacts it was derived from;
  * every entry in `anchors.json` rebuilds, through the same `_controller_for` the phases use, into a
    controller whose boundary, action and eta match what the file records.

The first is offline. The second needs the tau-bench checkout (the mechanism subclasses tau2's agent)
and skips without it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "tau2" / "rounds" / "SELFTEACH_R1"
EXPORT = REPO / "benchmarks" / "tau2" / "scripts" / "export_results.py"


def test_summary_is_reproducible_from_the_committed_arm_artifacts(tmp_path):
    work = tmp_path / "SELFTEACH_R1"
    subprocess.run(["cp", "-r", str(RESULTS), str(work)], check=True)
    subprocess.run([sys.executable, str(EXPORT), "--from-dest", "--dest", str(work)],
                   check=True, capture_output=True)
    assert (work / "summary.json").read_text() == (RESULTS / "summary.json").read_text()
    assert (work / "anchors.json").read_text() == (RESULTS / "anchors.json").read_text()


def test_the_reported_table_matches_the_summary():
    """The main table in the report is the export's own rendering, not a hand-copied one.

    docs/TAU2_SELFTEACH_ROUND1.md is withheld in this branch (see docs/RESULTS_POLICY.md: the
    tau2 campaign is open, pending a runner-parity resolution), so this skips rather than fails
    when it is absent -- the committed arm artifacts this test's sibling checks are not withheld.
    """
    report_path = REPO / "docs" / "TAU2_SELFTEACH_ROUND1.md"
    if not report_path.exists():
        pytest.skip("docs/TAU2_SELFTEACH_ROUND1.md is withheld; see docs/RESULTS_POLICY.md")
    spec = importlib.util.spec_from_file_location("export_results", EXPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    table = mod.markdown(json.loads((RESULTS / "summary.json").read_text()))
    report = report_path.read_text()
    assert table in report, "docs/TAU2_SELFTEACH_ROUND1.md's main table differs from the artifacts"


def test_no_committed_artifact_carries_a_credential():
    import re

    key_shaped = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")      # a gateway key, not the word "task-"
    for f in RESULTS.rglob("*.json"):
        text = f.read_text()
        for needle in ('"api_key"', '"extra_headers"', "RITS_API_KEY"):
            assert needle not in text, f"{f.relative_to(REPO)} contains {needle!r}"
        assert not key_shaped.search(text), f"{f.relative_to(REPO)} contains a key-shaped string"


def test_every_committed_anchor_rebuilds_into_the_recorded_controller():
    pytest.importorskip("tau2", reason="the mechanism subclasses tau2's LLMAgent")
    sys.path.insert(0, str(REPO / "benchmarks" / "tau2"))
    spec = importlib.util.spec_from_file_location(
        "tau2_round_driver_anchors", REPO / "benchmarks" / "tau2" / "run_anchoropt_round.py")
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    from tau2_runtime import ADAPTER

    anchors = json.loads((RESULTS / "anchors.json").read_text())
    assert anchors, "no anchors committed"
    for a in anchors:
        d = RESULTS / a["arm"]
        base = json.loads((d / "baseline.json").read_text())
        man = json.loads((d / "arm_manifest.json").read_text())
        ADAPTER.reset_expanded_signals()
        ADAPTER.reset_instruction_proposals()
        ADAPTER.set_tool_catalog(base.get("tool_catalog") or {})
        c = drv._controller_for(d, base, man, a["arm_label"])
        assert c.boundary == a["install"]["adapter_boundary_key"], a["arm"]
        assert c.action == a["action"], a["arm"]
        assert dict(c.eta) == a["eta"], a["arm"]
        assert callable(c.predicate)


def test_committed_baselines_carry_no_raw_trajectories():
    """`.gitignore` keeps raw trajectories out of this repo so it stays clonable. The export strips them
    and records where the full archive is; this keeps a later re-export from quietly committing them."""
    for f in RESULTS.glob("*/baseline.json"):
        blob = json.loads(f.read_text())
        assert not blob.get("events"), f"{f.relative_to(REPO)} carries raw trajectories"
        assert blob.get("events_stripped", {}).get("full_archive"), f"{f.relative_to(REPO)}: no archive pointer"
