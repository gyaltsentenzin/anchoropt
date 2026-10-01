"""TWO SEQUENTIAL ROUNDS with a genuinely moving incumbent, end to end, offline.

This is the milestone that GPU access was blocking, reduced to the part that does NOT need a cluster:
whether the LOOP composes. Round 1 accepts a controller against H0; round 2 is then measured against
the round-1 incumbent, and the test proves the second round's control is the first round's arm rather
than H0 again.

WHAT IS AND IS NOT DEMONSTRATED HERE
------------------------------------
Demonstrated: acceptance -> library -> installed_stack -> the next round naming that stack -> a second
acceptance recorded against the MOVED fingerprint -> restart recovery of both. All through the real
driver and the real library, over the real distilled BFCL artifacts.

NOT demonstrated: that a real BFCL evaluator installed two controllers and measured their composition.
That needs the cluster, and the R4 attempt found a defect that made exactly that claim false while the
logs suggested otherwise (see rounds/AUTONOMY/R4_RESULT_INVALID.json). Round 2's arm here REUSES the
round-1 arm's artifacts with a different spec name, so its accuracy delta is not a new measurement --
the test asserts on the LIFECYCLE mechanics it can honestly check, and asserts nothing about the sign
of a delta it did not obtain.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchoropt.learning.candidate_library import (            # noqa: E402
    ACCEPTED, PENDING_VALIDATION, CandidateLibrary, stack_fingerprint)

FX = ROOT / "tests" / "fixtures" / "r1_accept_pair"
pytestmark = pytest.mark.skipif(not (FX / "r1_reloc_train").exists(),
                                reason="real acceptance fixture absent")


def _drive(tmp_path, *, out, score, baseline, lib, dev=True):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "scripts")]),
               PYTHONHASHSEED="0")
    args = [sys.executable, str(ROOT / "scripts" / "self_evolve_cycle2.py"),
            "--incumbent", str(baseline), "--out", str(out), "--cell", "kv",
            "--score", str(score), "--baseline", str(baseline), "--library", str(lib)]
    if dev:
        args += ["--dev", str(FX / "r1_reloc_v4dev_test"),
                 "--dev-baseline", str(FX / "auto_ctl_v4dev_test"), "--dev-cell", "kv"]
    r = subprocess.run(args, capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return r, json.loads((out / "cycle2.json").read_text())


def _rename_spec(src: pathlib.Path, dst: pathlib.Path, name: str, signal: str):
    """A second, DIFFERENT controller identity over the same artifacts.

    The accuracy delta is therefore not a new measurement, and this test never claims it is. What is
    real is that the library sees a second distinct identity and must place it beside the first.
    """
    shutil.copytree(src, dst, dirs_exist_ok=True)
    spec = json.loads((dst / "controller_spec.json").read_text())
    spec["name"] = name
    spec["signal"] = signal
    spec["capability_id"] = "second_capability_for_lifecycle_test"
    (dst / "controller_spec.json").write_text(json.dumps(spec))
    return dst


def test_two_sequential_rounds_move_the_incumbent(tmp_path):
    lib = tmp_path / "candidates.json"
    CandidateLibrary().save(lib)

    # ---- ROUND 1: accept against H0 ------------------------------------------------------------
    r1, rec1 = _drive(tmp_path, out=tmp_path / "r1", score=FX / "r1_reloc_train",
                      baseline=FX / "auto_ctl_train", lib=lib)
    assert rec1["stack_fingerprint"] == "H0", "round 1 must measure against bare H0"
    assert rec1["acceptance"]["installed"] is True, "round 1 should accept"
    after1 = CandidateLibrary.load(lib)
    stack1 = after1.installed_stack("kv")
    assert len(stack1) == 1, f"the incumbent must have moved: {stack1}"
    fp1 = stack_fingerprint(stack1)
    assert fp1 != "H0"

    # ---- ROUND 2: the SAME driver must now name the round-1 stack -------------------------------
    arm2 = _rename_spec(FX / "r1_reloc_train", tmp_path / "arm2",
                        "second_controller", "a_different_signal")
    r2, rec2 = _drive(tmp_path, out=tmp_path / "r2", score=arm2,
                      baseline=FX / "auto_ctl_train", lib=lib)

    # THE CLAIM THIS TEST EXISTS FOR: round 2 measured against the MOVED incumbent, not H0.
    assert rec2["stack_fingerprint"] == fp1, \
        f"round 2 must name round 1's stack, got {rec2['stack_fingerprint']} want {fp1}"
    assert rec2["incumbent_stack"] == list(stack1)
    assert "INCUMBENT STACK for cell 'kv': 1 controller(s)" in r2.stdout

    # ---- and the second controller lands BESIDE the first, not over it --------------------------
    after2 = CandidateLibrary.load(lib)
    assert len(after2) == 2, f"both controllers must be recorded: {[r.name for r in after2.all()]}"
    assert "kv_core_capacity_relocation" in after2
    assert "second_controller" in after2
    first = after2.get("kv_core_capacity_relocation")
    assert first.state == ACCEPTED, "round 1's acceptance must survive round 2"
    # The second record names the stack it was measured against -- which is NOT the empty stack.
    second = after2.get("second_controller")
    assert second.context.fingerprint == fp1, \
        "the second controller's evidence must name the incumbent it was measured against"


def test_the_two_rounds_are_distinguishable_in_the_record(tmp_path):
    """A controller accepted against H0 and one accepted against H0+A must not look alike."""
    lib = tmp_path / "candidates.json"
    CandidateLibrary().save(lib)
    _drive(tmp_path, out=tmp_path / "r1", score=FX / "r1_reloc_train",
           baseline=FX / "auto_ctl_train", lib=lib)
    fp1 = stack_fingerprint(CandidateLibrary.load(lib).installed_stack("kv"))
    arm2 = _rename_spec(FX / "r1_reloc_train", tmp_path / "arm2", "second_controller", "sig2")
    _drive(tmp_path, out=tmp_path / "r2", score=arm2, baseline=FX / "auto_ctl_train", lib=lib)

    lib2 = CandidateLibrary.load(lib)
    a = lib2.get("kv_core_capacity_relocation")
    b = lib2.get("second_controller")
    assert a.context.fingerprint == "H0"
    assert b.context.fingerprint == fp1
    assert a.context.fingerprint != b.context.fingerprint, \
        "two controllers accepted against different incumbents must be distinguishable"


def test_a_RESTART_recovers_both_rounds_from_disk_alone(tmp_path):
    """The yo-yo test: a fresh interpreter, reading only the file, must recover the whole state."""
    lib = tmp_path / "candidates.json"
    CandidateLibrary().save(lib)
    _drive(tmp_path, out=tmp_path / "r1", score=FX / "r1_reloc_train",
           baseline=FX / "auto_ctl_train", lib=lib)
    arm2 = _rename_spec(FX / "r1_reloc_train", tmp_path / "arm2", "second_controller", "sig2")
    _drive(tmp_path, out=tmp_path / "r2", score=arm2, baseline=FX / "auto_ctl_train", lib=lib)

    probe = f"""
import sys; sys.path.insert(0, {str(ROOT)!r})
from anchoropt.learning.candidate_library import CandidateLibrary, stack_fingerprint
lib = CandidateLibrary.load({str(lib)!r})
print("RECORDS", len(lib))
print("ACCEPTED", sorted(r.name for r in lib.accepted()))
print("STACK", len(lib.installed_stack("kv")))
print("FP", stack_fingerprint(lib.installed_stack("kv")))
"""
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "RECORDS 2" in r.stdout, r.stdout
    assert "kv_core_capacity_relocation" in r.stdout
    assert "STACK 2" in r.stdout or "STACK 1" in r.stdout, r.stdout


def test_a_second_round_that_BLOCKS_leaves_the_first_acceptance_intact(tmp_path):
    """A failed round must never un-install what an earlier round installed."""
    lib = tmp_path / "candidates.json"
    CandidateLibrary().save(lib)
    _drive(tmp_path, out=tmp_path / "r1", score=FX / "r1_reloc_train",
           baseline=FX / "auto_ctl_train", lib=lib)
    stack1 = CandidateLibrary.load(lib).installed_stack("kv")

    # Round 2 with NO dev split: C2 is PENDING, so it blocks.
    arm2 = _rename_spec(FX / "r1_reloc_train", tmp_path / "arm2", "blocked_controller", "sig3")
    _r, rec2 = _drive(tmp_path, out=tmp_path / "r2", score=arm2,
                      baseline=FX / "auto_ctl_train", lib=lib, dev=False)
    assert rec2["acceptance"]["installed"] is False

    after = CandidateLibrary.load(lib)
    assert after.get("blocked_controller").state == PENDING_VALIDATION
    assert after.get("kv_core_capacity_relocation").state == ACCEPTED
    assert after.installed_stack("kv") == stack1, "the incumbent must not regress on a blocked round"
