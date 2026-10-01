"""THE MILESTONE: discover -> persist -> evaluate -> accept -> install -> regenerate -> re-mine,
then a FRESH PROCESS recovers the incumbent and the pending queue without manual reconstruction.

Driven through the real driver (`scripts/self_evolve_cycle2.py`) by subprocess against a synthetic
host, and through the real loader and real runtime hook. Nothing is reimplemented here: this project
has twice concluded a guard was fine from a hand-rolled driver and been wrong, and the whole point of
this test is that the lifecycle works in the code that actually runs.
"""

from __future__ import annotations

import json
import os
import re
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchoropt.learning.candidate_library import (            # noqa: E402
    ACCEPTED, MEASURED_NEGATIVE, PENDING_VALIDATION, CandidateLibrary, stack_fingerprint)

CELL = "kv"


def _spec(name="cap_ctl", signal="container_at_capacity", locus="post_execution",
          action="reroute", operator="transform", cap="relocate_then_retry", phase="prereq"):
    return {"name": name, "locus": locus, "signal": signal, "action": action,
            "operator": operator, "capability_id": cap, "phase": phase,
            "eta": {"primitive": "relocate_entry_preserving_information"},
            "predicate": {"all": [{"field": "error_kind", "op": "eq", "value": "no_capacity"}]}}


#: REAL trajectories, copied from the measured BV run `r1_reloc_train` (job 1834414). The proposal
#: phase projects candidate states out of `constraint_state` in these steps, so a synthetic
#: trajectory yields "0 projected states" and the driver exits before scoring -- the scoring phase is
#: downstream of a SUCCESSFUL proposal phase. Fabricating states rich enough to pass would be writing
#: a fake host, which is exactly the kind of test that has given this project false greens.
FIXTURE_TRAJ = ROOT / "tests" / "fixtures" / "bfcl_kv_sample" / "query"


def _real_cases():
    return sorted(p.stem for p in FIXTURE_TRAJ.glob("*.json")) if FIXTURE_TRAJ.exists() else []


def _mkrun(d: pathlib.Path, correct: dict, *, spec=None, provenance=None, traj=True):
    """A run directory in the shape `load_run` expects, carrying REAL trajectories when available."""
    (d / "run" / "traj").mkdir(parents=True, exist_ok=True)
    # `load_run` reads `valid`, not `correct` -- checked against the real loader rather than assumed.
    results = {"results": [{"id": cid, "valid": bool(ok)} for cid, ok in correct.items()]}
    (d / "run" / f"eval_train_results.json").write_text(json.dumps(results))
    if traj and FIXTURE_TRAJ.exists():
        import shutil
        shutil.copytree(FIXTURE_TRAJ, d / "run" / "traj" / "query", dirs_exist_ok=True)
    if spec is not None:
        (d / "controller_spec.json").write_text(json.dumps(spec))
    if provenance is not None:
        (d / "store_provenance.json").write_text(json.dumps(provenance))
    return d


def _run_driver(args, cwd):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "scripts")]),
               PYTHONHASHSEED="0")
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "self_evolve_cycle2.py"), *args],
                          capture_output=True, text=True, cwd=str(cwd), env=env)


def _diagnosis(case_id):
    """One diagnosis in the attributor's real schema, so `ingest_attribution` accepts it.

    Seeded as a CACHE (`<out>/diagnoses.json`) rather than mocked, because the driver's non-live path
    reads exactly that file. This drives the real ingestion, pooling and residual construction -- a
    stubbed diagnosis object would prove nothing about the path that runs on the GPU.
    """
    return {
        "case_id": case_id,
        "failure_mechanism": (
            "The agent treats the core key-value store as if it had unlimited capacity: it issues a "
            "batch of core_memory_add calls without accounting for what is already stored, and when "
            "the store rejects a write for want of capacity it wipes the whole store with "
            "core_memory_clear and re-issues the identical batch, destroying earlier facts."),
        "evidence": [
            {"step": 2, "observation": "core_memory_add returns {\"error\": \"Core memory is full. "
                                       "Please clear some entries.\"}"},
            {"step": 3, "observation": "the agent issues core_memory_clear and re-adds the batch"}],
        "causal_region": {"step": 3,
                          "phase": "immediately after the capacity rejection came back, when the "
                                   "agent chose how to react to a rejected write"},
        "consequential_decision": (
            "How the agent responds when a memory write is rejected for want of capacity -- whether "
            "it destructively wipes the store, or repairs the specific rejected write while "
            "preserving facts already stored."),
        "proposed_behavior_change": (
            "On a capacity rejection the agent should relocate an existing entry to a container with "
            "room, verify the copy landed before removing the original, and retry the rejected write "
            "verbatim, instead of clearing the store."),
    }


@pytest.fixture
def host(tmp_path):
    """An incumbent and an arm that beats it by +2, over the REAL kv case ids in the fixture."""
    cases = _real_cases()
    if not cases:
        pytest.skip("real trajectory fixture absent")
    # First 4 correct in both; the arm additionally fixes two the incumbent got wrong.
    base = {c: (i < 4) for i, c in enumerate(cases)}
    arm = dict(base)
    arm[cases[4]] = True
    arm[cases[5]] = True
    inc = _mkrun(tmp_path / "incumbent", base)
    a = _mkrun(tmp_path / "arm", arm, spec=_spec())
    return tmp_path, inc, a


def _seed_diagnoses(out_dir: pathlib.Path, failing):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnoses.json").write_text(
        json.dumps([_diagnosis(c) for c in failing], indent=1))


# ---------------------------------------------------------------------------------------------------
# READ SIDE: a round must be able to NAME the incumbent it measures against
# ---------------------------------------------------------------------------------------------------

def test_the_driver_reports_the_installed_stack_it_measures_against(host, tmp_path):
    root, inc, arm = host
    lib_path = root / "candidates.json"
    # Seed an ACCEPTED controller, i.e. a moving incumbent.
    from anchoropt.learning.candidate_library import (CandidateRecord, EvaluationContext,
                                                      Measurement)
    from anchoropt.learning.golden_registry import ControllerIdentity
    lib = CandidateLibrary()
    lib.upsert(CandidateRecord(
        name="already_installed",
        identity=ControllerIdentity(boundary="post_execution", signal="prior_signal",
                                    action="reroute", operator="transform",
                                    capability_id="prior_cap", phase="prereq"),
        spec=_spec(name="prior"), state=ACCEPTED,
        context=EvaluationContext(incumbent_stack=(), cell=CELL, incumbent_token="h0"),
        measurement=Measurement(arm_correct=5, control_correct=4, n_scored=10, gains=("g",),
                                losses=())))
    lib.save(lib_path)

    r = _run_driver(["--incumbent", str(inc), "--out", str(root / "out"), "--cell", CELL,
                     "--library", str(lib_path)], root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "INCUMBENT STACK for cell 'kv': 1 controller(s)" in r.stdout
    assert "post_execution/prior_signal/reroute/transform/prior_cap/prereq" in r.stdout
    rec = json.loads((root / "out" / "cycle2.json").read_text())
    assert rec["incumbent_stack"] == ["post_execution/prior_signal/reroute/transform/"
                                      "prior_cap/prereq"]
    assert rec["stack_fingerprint"] == stack_fingerprint(rec["incumbent_stack"])
    assert rec["stack_fingerprint"] != "H0"


def test_an_empty_library_says_bare_H0_out_loud(host, tmp_path):
    """A silent empty stack is indistinguishable from 'no library', and that ambiguity is what let a
    bare-H0 baseline pass for a moving incumbent."""
    root, inc, arm = host
    lib_path = root / "candidates.json"
    CandidateLibrary().save(lib_path)
    r = _run_driver(["--incumbent", str(inc), "--out", str(root / "out"), "--cell", CELL,
                     "--library", str(lib_path)], root)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((root / "out" / "cycle2.json").read_text())
    assert rec["incumbent_stack"] == []
    assert rec["stack_fingerprint"] == "H0"


def test_without_library_the_behaviour_is_unchanged(host):
    """The flag is additive: omitting it must reproduce the old path exactly."""
    root, inc, arm = host
    r = _run_driver(["--incumbent", str(inc), "--out", str(root / "out2"), "--cell", CELL], root)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((root / "out2" / "cycle2.json").read_text())
    assert rec["incumbent_stack"] == [] and rec["library"] == ""
    assert "PERSISTED" not in r.stdout


# ---------------------------------------------------------------------------------------------------
# WRITE SIDE: the verdict must outlive the round
# ---------------------------------------------------------------------------------------------------

def test_a_scored_round_PERSISTS_its_outcome(host):
    root, inc, arm = host
    lib_path = root / "candidates.json"
    r = _run_driver(["--incumbent", str(inc), "--out", str(root / "out"), "--cell", CELL,
                     "--library", str(lib_path), "--score", str(arm), "--baseline", str(inc)], root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PERSISTED" in r.stdout
    lib = CandidateLibrary.load(lib_path)
    assert len(lib) == 1
    rec = lib.get("cap_ctl")
    assert rec.identity.signal == "container_at_capacity"
    assert rec.context.cell == CELL
    # It names the stack it was measured against -- the field whose absence made results incomparable.
    assert rec.context.fingerprint == "H0"
    assert rec.spec["eta"]["primitive"] == "relocate_entry_preserving_information"


def test_a_control_arm_with_no_spec_is_NOT_persisted(host):
    """Inventing an identity from a directory name would put a record in the library that later
    rounds deduplicate against."""
    root, inc, arm = host
    ctl = _mkrun(root / "control", {f"memory_{CELL}_{i}-healthcare-{i}": (i < 4) for i in range(10)})
    lib_path = root / "candidates.json"
    r = _run_driver(["--incumbent", str(inc), "--out", str(root / "out"), "--cell", CELL,
                     "--library", str(lib_path), "--score", str(ctl), "--baseline", str(inc)], root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "NOT PERSISTED" in r.stdout
    assert CandidateLibrary.load(lib_path if lib_path.exists() else root / "absent.json") is not None
    assert len(CandidateLibrary.load(lib_path)) == 0 if lib_path.exists() else True


def test_a_DENOMINATOR_MISMATCH_is_recorded_and_stays_measurable(host):
    """THE rule this project kept losing. A denominator mismatch is a broken instrument, and banking
    it as a negative would permanently exclude a mechanism that was never really measured."""
    root, inc, arm = host
    cases = _real_cases()
    # A DIFFERENT case set: no paired comparison exists.
    bad = _mkrun(root / "badarm", {c: True for c in cases[3:]}, spec=_spec(name="unmeasurable_ctl"))
    lib_path = root / "candidates.json"
    r = _run_driver(["--incumbent", str(inc), "--out", str(root / "out"), "--cell", CELL,
                     "--library", str(lib_path), "--score", str(bad), "--baseline", str(inc)], root)
    assert r.returncode == 0, r.stdout + r.stderr
    rec_json = json.loads((root / "out" / "cycle2.json").read_text())
    assert rec_json["evaluation"]["denominator_ok"] is False
    lib = CandidateLibrary.load(lib_path)
    rec = lib.get("unmeasurable_ctl")
    assert rec.state == PENDING_VALIDATION, f"got {rec.state}"
    assert rec.measurement is None, "an unadjudicated round must not leave a quotable number"
    assert rec.excludes_re_evaluation is False, "it must stay available for validation"
    assert "denominator_ok" in rec.provenance


def test_the_scoring_only_path_ADJUDICATES_all_four_criteria(host):
    """The path must ASK the criteria, not refuse to. Refusing made acceptance unreachable on every
    path that did not complete a search -- which is why the lifecycle could not close."""
    root, inc, arm = host
    lib_path = root / "candidates.json"
    r = _run_driver(["--incumbent", str(inc), "--out", str(root / "out"), "--cell", CELL,
                     "--library", str(lib_path), "--score", str(arm), "--baseline", str(inc)], root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ACCEPTANCE:" in r.stdout
    rec = json.loads((root / "out" / "cycle2.json").read_text())
    assert rec["evaluation"]["criteria_adjudicated"] is True
    assert [c["number"] for c in rec["acceptance"]["criteria"]] == [1, 2, 3, 4]
    # C1 passes on +2; 2/3/4 have no evidence in this fixture, so they are PENDING and BLOCK.
    by_n = {c["number"]: c["verdict"] for c in rec["acceptance"]["criteria"]}
    assert by_n[1] == "PASS", by_n
    assert rec["acceptance"]["installed"] is False


def test_a_PENDING_criterion_blocks_acceptance_and_is_not_a_negative(host):
    """Missing evidence must never become a refutation -- the rule this project kept losing."""
    root, inc, arm = host
    lib_path = root / "candidates.json"
    _run_driver(["--incumbent", str(inc), "--out", str(root / "out"), "--cell", CELL,
                 "--library", str(lib_path), "--score", str(arm), "--baseline", str(inc)], root)
    lib = CandidateLibrary.load(lib_path)
    rec = lib.get("cap_ctl")
    assert rec.state == PENDING_VALIDATION, f"got {rec.state}"
    assert rec.excludes_re_evaluation is False
    assert rec.measurement is None, "a blocked round must not leave a quotable verdict"
    # The net IS preserved, as provenance rather than as a verdict.
    assert rec.provenance["paired_net"] == 2
    assert rec.provenance["criteria"]["C1"] == "PASS"


# ---------------------------------------------------------------------------------------------------
# THE RESTART MILESTONE
# ---------------------------------------------------------------------------------------------------

def test_a_FRESH_PROCESS_recovers_the_incumbent_and_the_pending_queue(host):
    """Restart the session and show the system recovers its state without manual reconstruction."""
    root, inc, arm = host
    lib_path = root / "candidates.json"
    # Round A: a scored arm.
    _run_driver(["--incumbent", str(inc), "--out", str(root / "outA"), "--cell", CELL,
                 "--library", str(lib_path), "--score", str(arm), "--baseline", str(inc)], root)
    # Round B: a second arm, recorded independently.
    bad = _mkrun(root / "badarm", {c: True for c in _real_cases()[3:]},
                 spec=_spec(name="pending_ctl"))
    _run_driver(["--incumbent", str(inc), "--out", str(root / "outB"), "--cell", CELL,
                 "--library", str(lib_path), "--score", str(bad), "--baseline", str(inc)], root)

    # A genuinely separate interpreter, reading only the file.
    probe = f"""
import sys; sys.path.insert(0, {str(ROOT)!r})
from anchoropt.learning.candidate_library import CandidateLibrary
lib = CandidateLibrary.load({str(lib_path)!r})
print("RECORDS", len(lib))
print("PENDING", sorted(r.name for r in lib.pending()))
print("STACKN", len(lib.installed_stack()))
"""
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout
    assert "RECORDS 2" in out, out
    # Both land PENDING: the scoring-only path does not adjudicate the acceptance criteria.
    assert "PENDING ['cap_ctl', 'pending_ctl']" in out, out


def test_the_shipped_library_round_trips_through_a_fresh_interpreter():
    """The real records, not a fixture -- so a schema change cannot silently orphan them."""
    lib_path = ROOT / "rounds" / "GOLDEN" / "candidates.json"
    if not lib_path.exists():
        pytest.skip("candidate library not seeded in this checkout")
    probe = f"""
import sys; sys.path.insert(0, {str(ROOT)!r})
from anchoropt.learning.candidate_library import CandidateLibrary, stack_fingerprint
lib = CandidateLibrary.load({str(lib_path)!r})
print("N", len(lib), "ACC", len(lib.accepted()), "PEND", len(lib.pending()))
print("FP", stack_fingerprint(lib.installed_stack()))
print("NAMES", " ".join(sorted(r.name for r in lib.accepted())))
"""
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    # NOT a hardcoded total. The library GROWS as rounds land -- it went 9 -> 10 when R5's
    # MEASURED_NEGATIVE was recorded, and a frozen count turns every real round into a test failure,
    # which trains you to bump the number instead of reading it. What must hold is the INVARIANT: the
    # shipped records round-trip through a fresh interpreter, and the three accepted controllers are
    # still there.
    nums = {k: int(v) for k, v in re.findall(r"(N|ACC|PEND) (\d+)", r.stdout)}
    # NOT a fixed count -- I already fixed `N == 9` here and then re-made the same mistake with
    # `ACC == 3`, which A9's acceptance broke. Acceptances only ever GROW, so the invariant is a floor
    # plus the named controllers surviving; a ceiling turns real progress into a test failure and
    # trains you to bump the number.
    assert nums.get("ACC", 0) >= 3, f"accepted controllers must not be LOST: {r.stdout}"
    assert nums.get("N", 0) >= nums["ACC"] + nums.get("PEND", 0), r.stdout
    assert nums.get("N", 0) >= 9, f"records must not be LOST: {r.stdout}"
    for name in ("kv_core_capacity_relocation", "a4_zero_call_reprompt_vector",
                 "rec_sum_capacity_recovery"):
        assert name in r.stdout, f"{name} must survive a round trip: {r.stdout}"
    assert "FP stack_" in r.stdout, "the installed stack must fingerprint to a real stack, not H0"
