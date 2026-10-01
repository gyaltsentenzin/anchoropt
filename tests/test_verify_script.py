"""End-to-end tests for the replication script itself.

The script is the thing a collaborator runs first, so it needs its own tests. Two properties matter
more than the happy path:

  * it must FAIL, with a non-zero exit code, when a shipped result stops matching a published
    number -- a verifier that passes on corrupted data is worse than no verifier;
  * it must write its recomputed values to disk, including on failure, so a reviewer reads numbers
    rather than re-parsing stdout.

Tampering is done on a COPY of the repo so the real artifacts are never touched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = "scripts/verify_progression.py"
TAMPER_TARGET = "rounds/T3_A3_duplicate/result/eval_train.json"


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        cwd=cwd, capture_output=True, text=True, check=False,
    )


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A copy of everything the script reads. Never mutate the real repo in a test."""
    dst = tmp_path / "repo"
    for rel in ("scripts", "rounds", "anchoropt"):
        shutil.copytree(REPO / rel, dst / rel)
    return dst


def test_passes_on_the_shipped_results(sandbox: Path):
    r = _run(sandbox)
    assert r.returncode == 0, f"clean run should exit 0:\n{r.stdout}\n{r.stderr}"
    assert "OK --" in r.stdout
    for token in ("29.04", "42.24", "+13.20"):
        assert token in r.stdout


def test_writes_a_report_with_source_digests(sandbox: Path):
    _run(sandbox)
    report = json.loads((sandbox / "verification_report.json").read_text())
    assert report["verdict"] == "OK"
    assert report["failures"] == []
    assert len(report["train"]) == 5
    assert len(report["held_out"]) == 4
    for row in report["train"] + report["held_out"]:
        assert row["matches_published"] is True
        assert len(row["sha256"]) == 16, "each row must be traceable to the bytes it read"


def test_report_records_the_cumulative_effect(sandbox: Path):
    _run(sandbox)
    cum = json.loads((sandbox / "verification_report.json").read_text())["cumulative"]
    assert cum["delta_pp"] == pytest.approx(13.20, abs=0.01)
    assert cum["n_anchors"] == 4
    assert cum["p_value"] < 0.05


def test_report_records_the_held_out_step_as_NOT_significant(sandbox: Path):
    """The one claim most tempting to overstate is pinned in machine-readable form."""
    _run(sandbox)
    step = json.loads((sandbox / "verification_report.json").read_text())["held_out_a4_step"]
    assert step["delta_pp"] == pytest.approx(3.57, abs=0.01)
    assert step["significant_at_05"] is False
    assert step["p_value"] > 0.05


def test_evidence_is_opt_in_and_flagged_as_not_a_target(sandbox: Path):
    plain = _run(sandbox)
    assert "SUPPORTING EVIDENCE" not in plain.stdout

    with_ev = _run(sandbox, "--with-evidence")
    assert with_ev.returncode == 0
    assert "not a reproduction target" in with_ev.stdout
    # A4 v1's delta only makes sense against its OWN incumbent.
    assert "-4.95" in with_ev.stdout

    report = json.loads((sandbox / "verification_report.json").read_text())
    assert len(report["evidence"]) == 2
    for row in report["evidence"]:
        assert row["is_reproduction_target"] is False
        assert row["incumbent_source"], "an accuracy without a named control is meaningless"


def test_fails_loudly_when_a_result_is_tampered_with(sandbox: Path):
    """Flip a single case. One case in 303 must be enough to fail the run."""
    path = sandbox / TAMPER_TARGET
    data = json.loads(path.read_text())
    for row in data["results"]:
        if not row.get("is_prereq") and row.get("valid"):
            row["valid"] = False
            break
    path.write_text(json.dumps(data))

    r = _run(sandbox)
    assert r.returncode == 1, "a corrupted result must produce a non-zero exit code"
    assert "FAILED" in r.stdout
    assert "116/303, expected 117/303" in r.stdout


def test_report_is_written_even_when_verification_fails(sandbox: Path):
    """A failing verification is exactly when you want the numbers on disk."""
    path = sandbox / TAMPER_TARGET
    data = json.loads(path.read_text())
    for row in data["results"]:
        if not row.get("is_prereq") and row.get("valid"):
            row["valid"] = False
            break
    path.write_text(json.dumps(data))

    _run(sandbox)
    report = json.loads((sandbox / "verification_report.json").read_text())
    assert report["verdict"] == "FAIL"
    assert report["failures"], "the failure must be recorded, not just printed"
    assert any(not r["matches_published"] for r in report["train"])


def test_fails_when_a_result_file_is_missing(sandbox: Path):
    (sandbox / TAMPER_TARGET).unlink()
    r = _run(sandbox)
    assert r.returncode == 1
    assert "MISSING" in r.stdout
