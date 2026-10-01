"""The first fully automatic four-criterion ACCEPTANCE, on real measured artifacts.

Driven through the real driver by subprocess over a distilled copy of the actual BV pair
`r1_reloc_train` (job 1834414) vs `auto_ctl_train`, plus their held-out runs. The distillation keeps
every telemetry key the acceptance path reads and drops the prose; it was verified to reproduce the
full-size run's verdict exactly before being committed.

Why this test exists: acceptance was UNREACHABLE. The scoring-only path refused to adjudicate, and the
one path that did adjudicate sat behind five early exits. Closing that is the difference between a
library that records discoveries and a loop that installs them.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchoropt.learning.candidate_library import (            # noqa: E402
    ACCEPTED, PENDING_VALIDATION, CandidateLibrary)

FX = ROOT / "tests" / "fixtures" / "r1_accept_pair"
ARM, CTL = FX / "r1_reloc_train", FX / "auto_ctl_train"
DEV_ARM, DEV_CTL = FX / "r1_reloc_v4dev_test", FX / "auto_ctl_v4dev_test"

pytestmark = pytest.mark.skipif(not ARM.exists(), reason="real acceptance fixture absent")


def _run(tmp_path, *extra, lib=None):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "scripts")]),
               PYTHONHASHSEED="0")
    args = [sys.executable, str(ROOT / "scripts" / "self_evolve_cycle2.py"),
            "--incumbent", str(CTL), "--out", str(tmp_path / "out"), "--cell", "kv",
            "--score", str(ARM), "--baseline", str(CTL), *extra]
    if lib is not None:
        args += ["--library", str(lib)]
    r = subprocess.run(args, capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return r, json.loads((tmp_path / "out" / "cycle2.json").read_text())


DEV = ("--dev", str(DEV_ARM), "--dev-baseline", str(DEV_CTL), "--dev-cell", "kv")


# ---------------------------------------------------------------------------------------------------
# THE MEASUREMENT ITSELF -- it must reproduce the recorded numbers
# ---------------------------------------------------------------------------------------------------

def test_the_paired_train_result_reproduces_the_golden_registry(tmp_path):
    """18 -> 22 of 105, +5/-1, net +4 -- the numbers `rounds/GOLDEN/registry.json` records for
    kv_core_capacity_relocation. A scoring path that cannot reproduce them is not measuring the
    same thing."""
    _r, rec = _run(tmp_path)
    ev = rec["evaluation"]
    assert (ev["control"], ev["arm"], ev["n"]) == (18, 22, 105)
    assert (ev["gains"], ev["losses"], ev["net"]) == (5, 1, 4)
    assert ev["denominator_ok"] is True and ev["believable"] is True


# ---------------------------------------------------------------------------------------------------
# ACTING PHASE vs SCORING PHASE -- the defect that made C3/C4 unpassable
# ---------------------------------------------------------------------------------------------------

def test_mechanism_evidence_is_read_from_the_PREREQ_phase_where_it_acted(tmp_path):
    """The arm carries controller_fired_gate 84x and relocate_gate 61x in traj/prereq and ZERO gate
    keys in traj/query. Reading only the scored phase asked a prereq-acting controller to prove
    itself where it never runs, and both C3 and C4 reported PENDING with the evidence one directory
    over."""
    r, rec = _run(tmp_path)
    assert "prereq + " in r.stdout, "the evidence read must state which phases it covered"
    by_n = {c["number"]: c["verdict"] for c in rec["acceptance"]["criteria"]}
    assert by_n[3] == "PASS", f"C3 should pass on the acting phase's telemetry: {by_n}"
    assert by_n[4] == "PASS", f"C4 should pass on the acting phase's telemetry: {by_n}"


def test_the_SCORE_stays_query_only_even_though_the_evidence_widened(tmp_path):
    """Widening where EVIDENCE is found must not move where the REWARD is measured: n stays 105
    query cases, not 105+27."""
    _r, rec = _run(tmp_path)
    assert rec["evaluation"]["n"] == 105


# ---------------------------------------------------------------------------------------------------
# THE FOUR CRITERIA, END TO END
# ---------------------------------------------------------------------------------------------------

def test_without_a_dev_split_C2_BLOCKS_and_nothing_installs(tmp_path):
    """Criterion 2 is PENDING without held-out evidence, and PENDING blocks. A train-only win must
    not install."""
    _r, rec = _run(tmp_path)
    by_n = {c["number"]: c["verdict"] for c in rec["acceptance"]["criteria"]}
    assert by_n[2] == "PENDING_VALIDATION"
    assert rec["acceptance"]["installed"] is False


def test_with_the_dev_split_ALL_FOUR_pass_and_the_controller_is_ACCEPTED(tmp_path):
    """THE MILESTONE: a fully automatic four-criterion acceptance from measured artifacts."""
    r, rec = _run(tmp_path, *DEV)
    by_n = {c["number"]: c["verdict"] for c in rec["acceptance"]["criteria"]}
    assert by_n[1] == "PASS"
    # The kv dev split is one domain chain where the CONTROL also scores 0/24, so it cannot detect a
    # regression. The preregistered rule is NO AGGREGATE REGRESSION and none occurred, so this passes
    # while carrying `informative: False` -- the weakness is reported, not used to raise the bar.
    assert by_n[2] == "PASS_UNINFORMATIVE"
    assert by_n[3] == "PASS"
    assert by_n[4] == "PASS"
    assert rec["acceptance"]["installed"] is True
    assert "ACCEPTED" in r.stdout


def test_the_uninformative_dev_pass_is_FLAGGED_not_hidden(tmp_path):
    """An acceptance leaning on a zero-power split must say so, or the transfer claim is overstated."""
    _r, rec = _run(tmp_path, *DEV)
    c2 = next(c for c in rec["acceptance"]["criteria"] if c["number"] == 2)
    assert c2["evidence"]["informative"] is False
    assert c2["evidence"]["arm_score"] == 0 and c2["evidence"]["control_score"] == 0


# ---------------------------------------------------------------------------------------------------
# WHAT GETS PERSISTED
# ---------------------------------------------------------------------------------------------------

def test_an_ACCEPTED_controller_enters_the_library_with_its_measurement(tmp_path):
    lib_path = tmp_path / "cand.json"
    _r, _rec = _run(tmp_path, *DEV, lib=lib_path)
    lib = CandidateLibrary.load(lib_path)
    rec = lib.get("kv_core_capacity_relocation")
    assert rec.state == ACCEPTED
    assert rec.measurement is not None and rec.measurement.net == 4
    assert rec.measurement.n_scored == 105
    assert rec.provenance["criteria"]["C3"] == "PASS"
    # It must name the incumbent it was measured against, or the result is incomparable.
    assert rec.context.cell == "kv"
    assert rec.identity.capability_id == "relocate_entry_preserving_information_then_retry"


def test_a_BLOCKED_round_persists_as_PENDING_with_no_quotable_verdict(tmp_path):
    """Same arm, no dev split: C2 blocks, so this is missing evidence -- never a negative."""
    lib_path = tmp_path / "cand.json"
    _r, _rec = _run(tmp_path, lib=lib_path)
    rec = CandidateLibrary.load(lib_path).get("kv_core_capacity_relocation")
    assert rec.state == PENDING_VALIDATION
    assert rec.measurement is None
    assert rec.excludes_re_evaluation is False
    assert rec.provenance["paired_net"] == 4      # kept as provenance, not as a verdict


def test_the_accepted_controller_appears_in_the_installed_stack(tmp_path):
    """Acceptance is only real if the next round can SEE it as the incumbent."""
    lib_path = tmp_path / "cand.json"
    CandidateLibrary().save(lib_path)
    _r, _rec = _run(tmp_path, *DEV, lib=lib_path)
    lib = CandidateLibrary.load(lib_path)
    assert len(lib.installed_stack("kv")) == 1
    assert "relocate_entry_preserving_information_then_retry" in lib.installed_stack("kv")[0]


# ---------------------------------------------------------------------------------------------------
# VALIDITY PRECEDES ACCEPTANCE -- found UNPINNED by a neuter audit and pinned here
# ---------------------------------------------------------------------------------------------------

def test_four_PASSES_on_an_UNPAIRED_comparison_must_NOT_install(tmp_path):
    """A denominator mismatch means no paired comparison happened at all, so even four PASSes would be
    adjudicating a non-measurement. Validity is not one of the four criteria and is not replaced by
    them: the criteria ask 'does a believable measurement justify installing', validity asks 'is this
    a measurement'.

    A neuter audit found this guard unpinned -- deleting `and believable` from the install decision
    broke no test. It does now.
    """
    import shutil
    # An arm scored over a DIFFERENT case set than the control: no pairing exists.
    arm = tmp_path / "unpaired_arm"
    shutil.copytree(ARM, arm)
    res = next(iter((arm / "run").glob("eval_*results*.json")))
    payload = json.loads(res.read_text())
    rows = payload["results"]
    # Drop a third of the cases, and make every survivor correct so C1 would pass handsomely.
    keep = [r for i, r in enumerate(rows) if i % 3 != 0]
    for r in keep:
        if not r.get("is_prereq"):
            r["valid"] = True
    res.write_text(json.dumps({"results": keep}))

    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "scripts")]),
               PYTHONHASHSEED="0")
    lib_path = tmp_path / "cand.json"
    args = [sys.executable, str(ROOT / "scripts" / "self_evolve_cycle2.py"),
            "--incumbent", str(CTL), "--out", str(tmp_path / "out"), "--cell", "kv",
            "--score", str(arm), "--baseline", str(CTL), "--library", str(lib_path), *DEV]
    r = subprocess.run(args, capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((tmp_path / "out" / "cycle2.json").read_text())

    assert rec["evaluation"]["denominator_ok"] is False, "the fixture must produce a real mismatch"
    assert rec["evaluation"]["believable"] is False
    # The four criteria may well pass on the surviving cases -- that is the point. Installation
    # must still be refused, and the refusal must be stated rather than silent.
    assert rec["acceptance"]["installed"] is False, \
        "an unpaired comparison must never install, however the criteria read"
    assert "Validity precedes acceptance" in r.stdout
    # And it is PENDING, not a negative: the measurement was broken, not the controller.
    from anchoropt.learning.candidate_library import CandidateLibrary as _CL
    rec_lib = _CL.load(lib_path).get("kv_core_capacity_relocation")
    assert rec_lib.state == PENDING_VALIDATION
    assert rec_lib.measurement is None
