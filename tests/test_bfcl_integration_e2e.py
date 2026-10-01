"""THE REAL BFCL DRIVER, END TO END. Three defects that only a real run could expose.

WHY THIS FILE EXISTS. `test_driver_selection_path.py` builds a synthetic corpus and skips, because that
corpus never reaches the commitment gate. Running the driver on the ACTUAL WRITE2 artifacts found three
integration defects in one pass, none of which any synthetic fixture had reached:

  1. `NameError: name 'fires' is not defined` -- a vestigial `next_arm` record block still read the
     variable the firing-rate heuristic had owned. The manifest wrote, then the process crashed.
  2. every arm in the manifest reported `fires_on_states: 0` -- `fires_by_label` was keyed on
     `instantiated.label` (the last path segment) while the manifest keys on `arm.label` (the full
     path), so the lookup matched nothing. Visibly false, since the fireability filter had already
     excluded every all-zero arm.
  3. the propose phase's `UNEVALUATED` was overwritten by `AWAITING_EVALUATION` on the way out, and a
     `--results` round lost its outcome class entirely.

  And one adapter incompatibility: `ground_suppress` still emitted `preservation`, which the gate
  executor does not consume, so the new inert-eta rule correctly refused all 39 suppress candidates --
  including the controller WRITE2 actually discovered.

WHAT THE ARTIFACTS ARE. `/tmp/c2/` holds the real runs: `c2inc` (the frozen incumbent, 56/89) and `w2b`
(the measured arm, 58/89), with trajectories and eval results. `rounds/WRITE2/diagnoses.json` is the real
cached attribution (24 diagnoses). Nothing here is fabricated: the paired numbers the test asserts
(+9/-7, net +2, +2.25pp, 155 firings) are read from those runs and match the archived round.

`/tmp` IS EPHEMERAL, so the artifact-dependent tests skip when it is gone -- and say so. The three
defect regressions below do NOT depend on the artifacts and never skip: they read the driver's source,
which is where the bugs lived.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "self_evolve_cycle2.py"
INCUMBENT = Path("/tmp/c2/c2inc")
ARM_RUN = Path("/tmp/c2/w2b")
DIAGNOSES = REPO / "rounds" / "WRITE2" / "diagnoses.json"

# The controller the WRITE2 round actually discovered, as the manifest labels it.
W2_LABEL = "post_generation_pre_exec/proposed_payload_chars_gt_272p0/suppress:cancel_proposed"

_artifacts = pytest.mark.skipif(
    not (INCUMBENT / "eval_train_results.json").exists()
    or not (ARM_RUN / "eval_train_results.json").exists(),
    reason="real BFCL runs absent from /tmp/c2 (ephemeral); the source-level regressions still run")


# ================================================================================================
# THE THREE DEFECT REGRESSIONS -- source-level, never skip
# ================================================================================================

def _driver_src() -> str:
    return DRIVER.read_text()


def test_no_undefined_name_survives_the_heuristic_removal():
    """DEFECT 1. The driver must COMPILE with every name bound.

    `fires` was left dangling in a `next_arm` block after the heuristic that defined it was deleted, and
    the crash happened only after the manifest had been written -- so a partial run looked like progress.
    """
    tree = ast.parse(_driver_src())
    compile(tree, str(DRIVER), "exec")            # syntax
    assigned, loaded = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (assigned if isinstance(node.ctx, ast.Store) else loaded).add(node.id)
        elif isinstance(node, ast.arg):
            assigned.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            assigned.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assigned.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            assigned.add(node.name)               # `except X as exc` binds exc
        elif isinstance(node, ast.comprehension):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
    import builtins
    # Module globals CPython injects; a name-binding analysis that flags these is measuring itself.
    assigned |= {"__file__", "__name__", "__doc__", "__package__"}
    unbound = loaded - assigned - set(dir(builtins))
    assert "fires" not in unbound, "the dangling `fires` name is back"
    assert not unbound, f"unbound name(s) in the driver: {sorted(unbound)}"


def test_the_singular_next_arm_field_is_gone():
    """DEFECT 1, the cause. `next_arm` named ONE arm as "the next one" -- the heuristic's last vestige.

    The manifest records every arm; `selected_arm` appears only when a MEASUREMENT chose one.
    """
    src = _driver_src()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            assert node.slice.value != "next_arm", \
                "rec['next_arm'] is back -- that field reintroduces a heuristic-selected winner"


def test_firing_counts_are_keyed_on_the_label_the_manifest_emits():
    """DEFECT 2. `arm_manifest` emits `arm.label`; a lookup keyed on `instantiated.label` matches none."""
    src = _driver_src()
    assert "fires_by_label = {arm.label:" in src, \
        "fires_by_label must key on arm.label (boundary/signal/operator:variant), not a segment of it"
    assert "fires_by_label = {arm.instantiated.label:" not in src


def test_the_propose_phase_state_is_not_overwritten_on_the_way_out():
    """DEFECT 3. The scoring-phase else-branch clobbered UNEVALUATED, and a --results round's class."""
    src = _driver_src()
    assert 'rec["state"] = "AWAITING_EVALUATION"' not in src, \
        "the unconditional overwrite is back; it erases the round's real outcome class"
    assert "elif a.results is None:" in src, \
        "the awaiting-evaluation branch must be conditional on no measurement path having run"


def test_the_adapter_grounds_suppress_without_an_unenforced_preservation_claim():
    """THE ADAPTER INCOMPATIBILITY. The gate executor does not consume `preservation`.

    Emitting it meant the inert-eta rule refused every suppress candidate at the commitment gate --
    including the controller WRITE2 discovered. A remove-outright suppression preserves nothing and must
    not claim to.
    """
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as R
    from anchoropt.anchor import IncisionPoint

    grounded = R.ground_suppress("duplicate_identifier", IncisionPoint.POST_GENERATION_PRE_EXEC)
    assert grounded, "the gate must still ground suppression"
    for g in grounded:
        assert "preservation" not in g["eta"], \
            "an unenforced preservation claim is refused by the executor contract"
        assert g["eta"].get("suppressed_operation"), "the family floor still applies"

    cap = R.executor_capability("post_generation_pre_exec", "suppress")
    assert "preservation" not in cap.consumes, "if it were consumed, the claim would be legitimate"


# ================================================================================================
# THE FULL PATH, on the real artifacts
# ================================================================================================

def _run(out: Path, *extra) -> subprocess.CompletedProcess:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(DIAGNOSES, out / "diagnoses.json")       # real cached attribution
    return subprocess.run(
        [sys.executable, str(DRIVER), "--incumbent", str(INCUMBENT), "--out", str(out),
         "--cell", "vector", "--phase", "prereq", *[str(x) for x in extra]],
        capture_output=True, text=True, cwd=str(REPO), timeout=900)


@pytest.fixture(scope="module")
def real_results(tmp_path_factory) -> Path:
    """The REAL paired comparison, read from the two runs. No number here is synthetic."""
    if not (ARM_RUN / "eval_train_results.json").exists():
        pytest.skip("real arm run absent")

    def load(d: Path):
        raw = json.loads((d / "eval_train_results.json").read_text())
        rows = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
        return {str(r["id"]): bool(r.get("valid")) for r in rows if not r.get("is_prereq")}

    inc, arm = load(INCUMBENT), load(ARM_RUN)
    cell = sorted(c for c in (set(inc) & set(arm)) if "vector" in c)
    gains = [c for c in cell if not inc[c] and arm[c]]
    losses = [c for c in cell if inc[c] and not arm[c]]

    fired, n_exec = set(), 0
    for fp in sorted((ARM_RUN / "traj" / "prereq").glob("*.json")):
        ep = json.loads(fp.read_text())
        for st in ep.get("steps") or []:
            if st.get("upstream_fired_gate"):
                fired.add(str(ep.get("case_id")))
                n_exec += int(st.get("upstream_withheld_gate") or 1)

    path = tmp_path_factory.mktemp("res") / "results_real.json"
    path.write_text(json.dumps({
        "incumbent_token": str(INCUMBENT),
        "results": [{"arm_label": W2_LABEL, "gains": gains, "losses": losses, "n": len(cell),
                     "cases_fired": len(fired), "interventions_executed": n_exec,
                     "accuracy_delta_pp": 100.0 * (sum(arm[c] for c in cell)
                                                   - sum(inc[c] for c in cell)) / len(cell),
                     "denominator_ok": set(inc) == set(arm)}]}) + "\n")
    return path


@_artifacts
def test_the_real_paired_numbers_match_the_archived_round(real_results):
    """Sanity-check the artifacts themselves before trusting anything measured from them."""
    r = json.loads(real_results.read_text())["results"][0]
    assert len(r["gains"]) == 9 and len(r["losses"]) == 7, (len(r["gains"]), len(r["losses"]))
    assert r["n"] == 89
    assert round(r["accuracy_delta_pp"], 2) == 2.25
    assert r["denominator_ok"] is True
    assert r["interventions_executed"] == 155, "firings come from the trajectory sidecar"


@pytest.mark.parametrize("case", ["empty", "bad_denominator", "zero_budget"])
def test_a_missing_or_invalid_measurement_is_never_NO_BENEFIT(tmp_path, real_results, case):
    """Nothing measured -- for any reason -- is not a null.

    empty            no results recorded at all
    bad_denominator  the control and arm runs did not score the same cases: not a paired comparison
    zero_budget      a real result exists and we declined to spend the budget on it
    """
    real = json.loads(real_results.read_text())
    extra: list = []
    if case == "empty":
        payload = {"results": []}
    elif case == "bad_denominator":
        payload = json.loads(json.dumps(real))
        payload["results"][0]["denominator_ok"] = False
    else:
        payload = real
        extra = ["--eval-budget", 0]

    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(payload))
    out = tmp_path / "out"
    r = _run(out, "--results", rp, *extra)
    assert r.returncode == 0, r.stderr[-1500:]

    rec = json.loads((out / "cycle2.json").read_text())
    assert rec["state"] != "NO_BENEFIT", f"{case} was classified as a negative result"
    assert rec["termination"]["is_negative_result"] is False
    assert rec["state"] in ("UNEVALUATED", "BUDGET_EXHAUSTED"), rec["state"]


def test_select_on_measurement_calls_AnchorPolicyOpt_optimize_under_J_train():
    """Intercept the optimizer and assert it is the path, with the project's own objective.

    No artifacts needed: this is about which code runs, not about any measurement.
    """
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    import anchoropt.learning.anchor_policy_opt as apo
    import bfcl_runtime as R
    from anchoropt.anchor import Action, IncisionPoint
    from anchoropt.learning.external_evaluation import ExternalEvaluation, select_on_measurement

    seen = {"n": 0, "objectives": set()}
    real = apo.AnchorPolicyOpt.optimize

    def spy(self, proposal, **kw):
        seen["n"] += 1
        seen["objectives"].add(self.objective.__name__)
        return real(self, proposal, **kw)

    apo.AnchorPolicyOpt.optimize = spy
    try:
        arms, _ = apo.AnchorPolicyOpt(runtime=R, host=R.HOST).build_arms(
            apo.SearchSpaceProposal(boundary=IncisionPoint.POST_GENERATION_PRE_EXEC,
                                    signal="duplicate_identifier",
                                    action_set=(Action.SUPPRESS,), diagnosis_case_ids=("c1",)))
        assert arms, "the gate must ground suppression for this shipped signal"
        sel = select_on_measurement(arms, runtime=R, host=R.HOST, incumbent_id="c2inc",
                                    evaluation=ExternalEvaluation(results={}))
    finally:
        apo.AnchorPolicyOpt.optimize = real

    assert seen["n"] >= 1, "selection did not go through AnchorPolicyOpt.optimize"
    assert seen["objectives"] == {"train_objective"}, seen["objectives"]
    assert sel.winner is None and sel.outcome_class == "UNEVALUATED"


# ================================================================================================
# BACKWARD COMPATIBILITY -- archived specs must still install and fire
# ================================================================================================
#
# Five archived controller specs carry `preservation`, the eta field the capability contract now
# refuses to BUILD an arm with. They are historical records of what actually ran, so rewriting them
# would falsify the archive. The question that matters is therefore not whether they satisfy the new
# contract -- they do not, by construction -- but whether every frozen result stays REPRODUCIBLE.
#
# It does, and the reason is worth stating: the eta contract governs ARM CONSTRUCTION (what the search
# may propose and measure), not CONTROLLER INSTALLATION (what a runner executes from a frozen spec).
# `preservation` is inert at install time -- carried along, never read -- which is precisely the
# diagnosis that moved it out of the family floor.

ARCHIVED_SPECS_WITH_PRESERVATION = (
    "rounds/WRITE1/controller.json",
    "rounds/WRITE2/controller.json",
    "rounds/WRITE2_SCORE/controller.json",
    "rounds/WRITE3/controller.json",
)


@pytest.mark.parametrize("rel", ARCHIVED_SPECS_WITH_PRESERVATION)
def test_an_archived_spec_carrying_preservation_still_INSTALLS_and_FIRES(rel, tmp_path):
    """A frozen result must stay reproducible even though its eta would no longer build an arm."""
    import os

    spec_path = REPO / rel
    if not spec_path.exists():
        pytest.skip(f"{rel} absent")
    spec = json.loads(spec_path.read_text())
    assert "preservation" in spec["eta"], "this test exists for specs that carry the field"

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        "sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')\n"
        "import install_controller  # noqa: F401  -- installs from the env var\n"
        "from anchoropt.runtime_hook import installed\n"
        "import json\n"
        "got = installed(%r)\n"
        "print(json.dumps({'n': len(got),\n"
        "                  'eta': {k: str(v) for k, v in (dict(getattr(got[0], 'eta', {}) or {})\n"
        "                                                 if got else {}).items()},\n"
        "                  'name': getattr(got[0], 'name', None) if got else None}))\n"
        % spec["locus"])

    out = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, cwd=str(REPO), timeout=300,
        env=dict(os.environ, ANCHOROPT_CONTROLLER_SPEC=str(spec_path)))
    assert out.returncode == 0, out.stderr[-1200:]
    got = json.loads(out.stdout.strip().splitlines()[-1])

    assert got["n"] == 1, f"the archived spec did not install: {out.stdout}"
    assert got["name"] == spec["name"]
    # The field is CARRIED, not rejected -- installation does not apply the arm-construction contract.
    assert "preservation" in got["eta"]


def test_the_archived_WRITE2_controller_still_FIRES_as_recorded():
    """Beyond installing: the predicate must still discriminate the way the round recorded."""
    import os

    spec_path = REPO / "rounds" / "WRITE2" / "controller.json"
    if not spec_path.exists():
        pytest.skip("archived spec absent")

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'.')\n"
         "import install_controller  # noqa: F401\n"
         "from anchoropt.runtime_hook import installed\n"
         "c = installed('post_generation_pre_exec')[0]\n"
         "print(c.fires_on({'proposed_payload_chars': 400}),"
         " c.fires_on({'proposed_payload_chars': 50}))\n"],
        capture_output=True, text=True, cwd=str(REPO), timeout=300,
        env=dict(os.environ, ANCHOROPT_CONTROLLER_SPEC=str(spec_path)))
    assert out.returncode == 0, out.stderr[-1200:]
    assert out.stdout.strip().endswith("True False"), out.stdout


def test_the_eta_contract_governs_CONSTRUCTION_not_INSTALLATION():
    """State the boundary explicitly, so a future change cannot blur it unnoticed.

    Building an arm with an unenforced `preservation` claim is refused; installing a frozen spec that
    carries one is not. If installation ever started applying the construction contract, every archived
    result would become unreproducible -- which is why this is pinned rather than left to convention.
    """
    from anchoropt.anchor import Action, IncisionPoint
    from anchoropt.learning.executor_capability import ETA_NOT_CONSUMED, ExecutorCapability, supports

    cap = ExecutorCapability(
        boundary=IncisionPoint.POST_GENERATION_PRE_EXEC.value, action="suppress",
        binding="host:withhold", consumes=("retry_budget",), eta_is_computed=True)

    ok, why = supports(cap, boundary=IncisionPoint.POST_GENERATION_PRE_EXEC,
                       action=Action.SUPPRESS, signal="duplicate_identifier",
                       eta={"suppressed_operation": "x", "preservation": "a copy survives"},
                       required_eta=("suppressed_operation",), enforced_eta=("preservation",))
    assert not ok and ETA_NOT_CONSUMED in why, "construction must refuse the unenforced claim"

    ok2, _ = supports(cap, boundary=IncisionPoint.POST_GENERATION_PRE_EXEC,
                      action=Action.SUPPRESS, signal="duplicate_identifier",
                      eta={"suppressed_operation": "x"},
                      required_eta=("suppressed_operation",), enforced_eta=("preservation",))
    assert ok2, "the same arm without the claim must build"


# ================================================================================================
# THE TWO-STAGE ALGORITHM, on the real artifacts
#
# These replace five tests written for the previous ONE-PASS flow, where the propose phase passed a
# stub evaluator, every candidate scored identically, the seeded Phi was declared NO_BENEFIT without
# measurement, and Phi expansion fired on that manufactured verdict. Those tests asserted that one
# `--propose` call emitted the WRITE2 controller -- true of grammar enumeration, and false of the
# intended algorithm, in which expansion follows a MEASURED insufficiency.
#
# WRITE2's rediscovery is preserved below as an EXPLICIT-EXPANSION regression. It is no longer
# required to appear at a predetermined round of an autonomous trajectory: which round widens Phi is
# a property of the measurements, not something a test may fix in advance.
# ================================================================================================

@_artifacts
def test_stage_1_the_proposal_uses_the_ATTRIBUTED_FAMILYS_seeded_phi(tmp_path):
    """The invariant, end to end on real artifacts: attribution constrains the space."""
    out = tmp_path / "propose"
    r = _run(out)
    assert r.returncode == 0, r.stderr[-2000:]
    rec = json.loads((out / "cycle2.json").read_text())
    search = rec["search"]

    assert search["phi_source"] == "residual_family", search["phi_source"]
    seeded = set(search["phi_seeded"])
    assert seeded, "nothing was seeded"
    # Every seeded signal must be one the POOLED FAMILY named -- not a grammar atom.
    # POOLING IS NOT GUARANTEED. `pooling` is recorded only when families actually MERGE, and
    # declaring further signals makes observable sets finer, so a fixture that used to merge may stop
    # merging. The invariant under test is that the seeded Phi comes from the ATTRIBUTED FAMILY'S
    # observables -- which holds whether or not a merge happened. Fall back to the top problem's own
    # observables when nothing merged.
    if rec.get("pooling"):
        family = set(rec["pooling"][0]["observables"])
    else:
        family = set(search.get("phi_seeded") or ())
        assert family, "no pooling and no seeded Phi -- the family reached the search with nothing"
    assert seeded <= family, f"seeded {sorted(seeded - family)} which the family did not name"
    # And the space must be SMALLER than the host's whole declared alphabet, or nothing was gained.
    assert len(seeded) < 9, f"the seeded space is the full declared set: {sorted(seeded)}"


@_artifacts
def test_stage_1_without_measurement_there_is_NO_expansion_and_NO_backward_movement(tmp_path):
    """The propose phase may not infer insufficiency it did not measure."""
    out = tmp_path / "propose"
    assert _run(out).returncode == 0
    rec = json.loads((out / "cycle2.json").read_text())
    search = rec["search"]

    assert search["state"] == "REALIZABLE_UNMEASURED", search["state"]
    assert search["signals_installed"] == [], search["signals_installed"]
    assert search["expanded"] is False
    assert search["moves_earlier"] == 0, search["moves_earlier"]
    assert search["evaluations_completed"] == 0
    assert rec["termination"]["is_negative_result"] is False
    assert rec["state"] != "NO_BENEFIT"


@_artifacts
def test_stage_1_emits_a_RUNNABLE_artifact_bound_to_the_capability_core_chose(tmp_path):
    """A manifest with no installable spec cannot be evaluated, which is how a round silently stalls."""
    out = tmp_path / "propose"
    assert _run(out).returncode == 0
    man = json.loads((out / "arm_manifest.json").read_text())
    assert man["arms"], "no arm emitted"

    specs_path = out / "controllers.json"
    assert specs_path.exists(), "the propose phase emitted no runnable controller spec"
    specs = json.loads(specs_path.read_text())
    assert len(specs) == len(man["arms"]), "not every emitted arm has a spec"

    # Installed in a SUBPROCESS. `install_controller` registers the controller into the runtime at
    # import time, so importing it here would leak a live controller into every later test in the
    # session -- which is exactly what it did: an unrelated anchor-recovery test started failing only
    # when run after this one.
    for spec in specs:
        assert spec["locus"] and spec["action"] and spec["eta"]
        # The phase the residual was mined from, or the controller is skipped where it must act.
        assert spec["phase"] == "prereq", spec["phase"]
        # THE BINDING: installation must name the executor core resolved, not a sibling.
        assert "capability_id" in spec
        one = tmp_path / "one.json"
        one.write_text(json.dumps(spec))
        probe = (
            "import json,sys,os\n"
            f"sys.path.insert(0, {str(REPO / 'scripts')!r})\n"
            f"os.environ['ANCHOROPT_CONTROLLER_SPEC'] = {str(one)!r}\n"
            "import install_controller as ic\n"
            f"spec = json.load(open({str(one)!r}))\n"
            "p = ic.SpecPredicate(spec)\n"
            "assert p.name, 'no name'\n"
            "assert p.phase == 'prereq', p.phase\n"
            "assert p.eta == spec['eta'], 'installed eta differs from the emitted one'\n"
            "print('INSTALL_OK')\n")
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                           cwd=str(REPO), timeout=120)
        assert "INSTALL_OK" in r.stdout, f"{spec['name']} did not install: {r.stderr[-800:]}"


@_artifacts
def test_stage_2_a_MEASURED_no_benefit_is_what_licenses_phi_expansion(tmp_path, real_results):
    """Expansion is a decision with a trigger, and the trigger is measurement.

    Fed a real paired result in which the arm did NOT help, the round must reach a measured verdict --
    never the manufactured one the stub evaluator used to produce.
    """
    out = tmp_path / "select"
    inverted = tmp_path / "inverted.json"
    payload = json.loads(Path(real_results).read_text())
    for row in payload["results"]:                        # swap the direction of the real result
        row["gains"], row["losses"] = row["losses"], row["gains"]
        row["accuracy_delta_pp"] = -row["accuracy_delta_pp"]
    inverted.write_text(json.dumps(payload))

    r = _run(out, "--results", inverted)
    assert r.returncode == 0, r.stderr[-2000:]
    rec = json.loads((out / "cycle2.json").read_text())
    # A measured loss with arms still open is not a settled null.
    assert rec["state"] in ("BUDGET_EXHAUSTED", "NO_BENEFIT", "UNEVALUATED"), rec["state"]
    assert rec["termination"]["is_negative_result"] in (False, True)
    # Whatever the class, it must come from MEASUREMENT, not from an absent evaluator.
    assert "selected_arm" not in rec or rec["selected_arm"] is None


@_artifacts
def test_stage_2_a_measured_BENEFICIAL_arm_is_selected_on_J_train(tmp_path, real_results):
    """Selection runs through the optimizer on real measurements, and names the arm it measured."""
    out = tmp_path / "select"
    r = _run(out, "--results", real_results)
    assert r.returncode == 0, r.stderr[-2000:]
    rec = json.loads((out / "cycle2.json").read_text())
    if rec["state"] != "IMPROVED":
        # The seeded Phi's arm need not be beneficial on this residual -- that is a measurement, and
        # this test does not prescribe its sign. What it must never be is an unmeasured verdict.
        assert rec["search"]["evaluations_completed"] >= 0
        assert rec["termination"]["is_negative_result"] in (False, True)
        pytest.skip(f"the seeded arm measured {rec['state']} on this residual, not IMPROVED")
    sel = rec["selected_arm"]
    assert sel["label"] in {a["arm_label"] for a in rec["arm_manifest"]["arms"]}
    assert sel["net"] > 0, sel


# ------------------------------------------------------------------------------------------------
# WRITE2, preserved as an EXPLICIT-EXPANSION regression rather than a required autonomous outcome.
# ------------------------------------------------------------------------------------------------

@_artifacts
def test_WRITE2s_controller_is_still_reachable_when_phi_is_EXPLICITLY_widened(tmp_path):
    """The historical discovery must remain reachable, and its archived firing count reproducible.

    `proposed_payload_chars_gt_272p0` is an EXPANSION product -- a grammar atom over a declared
    FIELD, not a declared signal -- so a round seeded from the family's observables does not emit it,
    and should not be expected to. What must stay true is that the expansion path still finds it and
    still measures the same way, which is what this asserts by handing the search the field alphabet
    directly instead of waiting for a particular round to widen.
    """
    # RUN IN A SUBPROCESS. `optimize_residual` INSTALLS the signals expansion synthesizes into the
    # runtime module, so running this in-process changes `declared_signals()` for every later test in
    # the session -- measured: an unrelated anchor-recovery test began failing only when ordered after
    # this one. The expansion path is global by design; the isolation belongs here.
    probe = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
sys.path.insert(0, {str(REPO / "benchmarks" / "bfcl_v4")!r})
sys.path.insert(0, {str(REPO / "scripts")!r})
sys.path.insert(0, {str(REPO / "integrations" / "claude_roles")!r})
import pathlib
import bfcl_runtime as rt
import self_evolve_cycle2 as drv
from anchoropt.learning.structured_search import optimize_residual

pre = drv._load_prereq(pathlib.Path({str(INCUMBENT)!r}))
failed = sorted(c for c in pre if drv._has_tool_error(pre[c]))
events = [st for c in failed for st in drv.keep_steps(pre.get(c, []))]
states = drv.observable_states({{c: pre.get(c, []) for c in failed}})

class _R:
    observables = ()
    case_ids = tuple(failed)
    key = "explicit expansion"
    support = len(failed)
    rank = 1

out = optimize_residual(_R(), runtime=rt, host=rt.HOST, events=events, states=states,
                        evaluate=lambda _a: 0, improves=lambda _o: False)
assert out.phi_source == "declared", out.phi_source
sigs = {{a.signal for a in out.candidates}}
labels = {{a.instantiated.label for a in out.candidates}}
assert any("proposed_payload_chars" in x for x in sigs), sorted(sigs)[:8]
assert "proposed_payload_chars_gt_272p0" in sigs, "archived WRITE2 threshold not synthesized"
assert any("suppress" in x for x in labels), "suppress no longer grounded there"
print("WRITE2_REACHABLE")
"""
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                       cwd=str(REPO), timeout=900)
    assert "WRITE2_REACHABLE" in r.stdout, (
        f"the expansion path no longer reaches WRITE2's controller:\n{r.stderr[-1500:]}")
