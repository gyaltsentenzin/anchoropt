"""Every documented command must actually run.

A README command that errors is worse than an undocumented one: it costs a collaborator their first
ten minutes and their confidence in everything else. These tests run each entry point as a
subprocess -- the way a person would -- and assert on exit code and the substance of the output.

They also pin that the ported attribution/learning modules import, since those are what make the
framework walkthrough more than prose.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args], cwd=REPO, capture_output=True, text=True, check=False,
    )


# ---------------------------------------------------------------------------------------------
# scripts/run_pipeline.py -- compose the anchors, state expectations, check a run
# ---------------------------------------------------------------------------------------------

def test_pipeline_show_lists_all_four_anchors():
    r = run("scripts/run_pipeline.py", "--show")
    assert r.returncode == 0, r.stderr
    for name in ("A1", "A2", "A3", "A4"):
        assert name in r.stdout
    # The composed policy must show every anchor's switch as enabled.
    for switch in ("enable_capacity_repair", "enable_reroute",
                   "enable_redundant_write_suppress", "enable_zero_call_reprompt"):
        assert switch in r.stdout


def test_pipeline_show_flags_the_empty_global_preamble():
    """The defining property of this line: no global prompt anywhere."""
    r = run("scripts/run_pipeline.py", "--show")
    assert "EMPTY" in r.stdout
    assert "PRESENT (!)" not in r.stdout


def test_pipeline_show_explains_the_a4_v1_v2_switch_pair():
    """`on_turn_start_action=false` beside `enable_zero_call_reprompt=true` looks wrong; it is the fix."""
    r = run("scripts/run_pipeline.py", "--show")
    assert "on_turn_start_action" in r.stdout
    assert "enable_zero_call_reprompt" in r.stdout
    assert "-4.95" in r.stdout and "+3.63" in r.stdout


def test_pipeline_expect_states_arms_conditions_and_the_temperature_caveat():
    r = run("scripts/run_pipeline.py", "--expect")
    assert r.returncode == 0, r.stderr
    # The MEASURED native baseline (91/303), the A1-A4 waypoint, and the full stack.
    for token in ("91/303", "128/303", "144/303", "22/ 84", "granite-4.1-8b", "0.001"):
        assert token in r.stdout
    # Reproducing the exact counts depends on the near-zero temperature; say so.
    assert "will NOT reproduce" in r.stdout


def test_pipeline_check_fails_cleanly_on_a_directory_with_no_results(tmp_path: Path):
    r = run("scripts/run_pipeline.py", "--check", str(tmp_path))
    assert r.returncode == 1
    assert "no eval_" in r.stdout


def test_pipeline_check_recognises_a_real_arm(tmp_path: Path):
    """Point --check at a copy of a shipped arm; it must match a known expected score."""
    import shutil
    src = REPO / "rounds" / "T5_A4_no_tool_call" / "result" / "eval_train.json"
    shutil.copy(src, tmp_path / "eval_train_results.json")
    r = run("scripts/run_pipeline.py", "--check", str(tmp_path))
    assert r.returncode == 0, r.stdout
    assert "128/303" in r.stdout
    assert "PASS" in r.stdout


def test_pipeline_requires_a_mode():
    r = run("scripts/run_pipeline.py")
    assert r.returncode != 0, "no mode given should be an error, not a silent no-op"


# ---------------------------------------------------------------------------------------------
# scripts/walk_framework.py -- the six framework steps
# ---------------------------------------------------------------------------------------------

def test_walkthrough_runs_all_steps():
    r = run("scripts/walk_framework.py")
    assert r.returncode == 0, r.stderr
    for n in range(1, 7):
        assert f"STEP {n}" in r.stdout


@pytest.mark.parametrize("step", range(1, 7))
def test_each_step_runs_alone(step: int):
    r = run("scripts/walk_framework.py", "--step", str(step))
    assert r.returncode == 0, r.stderr
    assert f"STEP {step}" in r.stdout


def test_step1_shows_canonicalization_merging_and_its_failure_mode():
    """Two wordings must merge to one locus -- and an unseen wording must visibly return None."""
    r = run("scripts/walk_framework.py", "--step", "1")
    assert "capacity/container/no_remaining_capacity" in r.stdout
    assert "None" in r.stdout, "the cue-table gap must be shown, not hidden"
    assert "harness_fault=True" in r.stdout


def test_step4_recomputes_the_progression_from_shipped_results():
    r = run("scripts/walk_framework.py", "--step", "4")
    for token in ("88/303", "102/303", "117/303", "128/303"):
        assert token in r.stdout


def test_step5_names_the_rule_that_stopped_each_deferral():
    r = run("scripts/walk_framework.py", "--step", "5")
    for rule in ("S1", "S2", "S3", "S4", "S5"):
        assert rule in r.stdout
    assert "A2.W" in r.stdout
    # The biggest residual class is deliberately not anchored.
    assert "outcome class, not a mechanism" in r.stdout


def test_step6_shows_each_arm_becoming_the_next_control():
    r = run("scripts/walk_framework.py", "--step", "6")
    assert "NOT re-run" in r.stdout
    # RE-PINNED FOR THE ANONYMOUS RELEASE. Was f90db95bbcde, the sha256 of
    # rounds/T1_A1_capacity/result/eval_train.json as measured. That file records absolute cluster
    # paths containing real usernames, which a double-blind submission cannot ship, so the paths were
    # rewritten to placeholders and the digest moved with them.
    #
    # What this does NOT change: the per-case verdicts in that file, and so every accuracy derived
    # from it -- `scripts/verify_progression.py` still reproduces all nine published figures.
    # What it DOES cost: this digest can no longer be compared against the pre-scrub measurement.
    assert "4e0debbd0e1a" in r.stdout, "the arm->control chain is pinned by digest"


# ---------------------------------------------------------------------------------------------
# the ported framework modules
# ---------------------------------------------------------------------------------------------

PORTED = {
    "anchoropt/attribution": ["harness_guard", "constraint_locus", "llm_operator",
                              "op_canonicalize_locus", "attribution_miner", "trace_backward",
                              "destination_attest"],
    "anchoropt/learning": ["policy_tree", "remine_incumbent", "phase_switch",
                           "expand_attribution", "check_attribution", "target_spec",
                           "evidence_ledger"],
}


@pytest.mark.parametrize(
    "pkg,mod", [(p, m) for p, mods in PORTED.items() for m in mods],
    ids=[f"{p.split('/')[-1]}.{m}" for p, mods in PORTED.items() for m in mods],
)
def test_ported_module_imports(pkg: str, mod: str):
    """Import each ported module in isolation. They are stdlib-only, so this must not need extras."""
    code = (
        "import sys;"
        f"sys.path[:0]=[r'{REPO / 'anchoropt' / 'attribution'}', r'{REPO / 'anchoropt' / 'learning'}'];"
        f"import {mod}"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"{mod} failed to import:\n{r.stderr}"


def test_no_hardcoded_absolute_paths_in_ported_code():
    """125 of 149 scripts in the working repo carry /u/userB paths. None may survive here."""
    offenders = []
    for pkg in PORTED:
        for f in (REPO / pkg).glob("*.py"):
            text = f.read_text()
            if "/u/userB" in text or "/u/userX" in text:
                offenders.append(str(f.relative_to(REPO)))
    assert not offenders, f"hardcoded cluster paths remain: {offenders}"


# ---------------------------------------------------------------------------------------------
# benchmarks/bfcl_v4/run.py -- the one-flag benchmark runner
# ---------------------------------------------------------------------------------------------

BENCH = "benchmarks/bfcl_v4/run.py"


def test_bfcl_dry_run_shows_both_arms_and_the_expected_delta():
    r = run(BENCH, "--compare", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "control (no anchors)" in r.stdout
    # --compare now pairs the control with the FULL STACK by default, not the A1-A4 waypoint.
    assert "full stack" in r.stdout
    assert "91/303 = 30.03 %" in r.stdout, "the control expectation is the MEASURED baseline"
    assert "144/303 = 47.52 %" in r.stdout
    assert "+17.49 pp" in r.stdout


def test_bfcl_dry_run_resolves_the_held_out_split_too():
    """Keying expectations on the wrong split name silently loses the held-out numbers."""
    r = run(BENCH, "--compare", "--split", "test", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "14/84 = 16.67 %" in r.stdout
    assert "34/ 84 = 40.48 %" in r.stdout or "34/84 = 40.48 %" in r.stdout
    assert "+23.81 pp" in r.stdout
    # The dev split must not be presented as a clean generalization estimate. The wording moved
    # from "NOT significant" to naming WHY -- it is the accept/reject criterion, so it is
    # selected-on -- but the guard is the same: no overclaiming on n=84.
    low = r.stdout.lower()
    assert "validation set" in low and "selected-on" in low
    assert "1.19 pp" in r.stdout, "one case at n=84 must be quantified"


def test_bfcl_defaults_to_the_paired_comparison():
    """`both` is the only defensible mode, so it must be the default."""
    r = run(BENCH, "--dry-run")
    assert r.returncode == 0
    assert "conditions to run: control, a8" in r.stdout


@pytest.mark.parametrize("arm", ["control", "anchors"])
def test_bfcl_single_arm_runs(arm: str):
    r = run(BENCH, "--only", arm, "--dry-run")
    assert r.returncode == 0, r.stderr


def test_bfcl_builds_a_command_naming_the_right_policy():
    r = run(BENCH, "--only", "anchors", "--dry-run")
    assert "rounds/T5_A4_no_tool_call/policy.json" in r.stdout
    assert "--templates" in r.stdout and "--model-config" in r.stdout


def test_bfcl_control_arm_also_passes_disable_gates():
    """Belt and braces: the control policy is empty AND the flag is set."""
    r = run(BENCH, "--only", "control", "--dry-run")
    assert "--disable-gates" in r.stdout


def test_bfcl_real_run_without_an_evaluator_fails_with_a_hint():
    """The evaluator is not vendored; saying so is more useful than a traceback."""
    r = run(BENCH, "--only", "control")
    assert r.returncode == 2
    assert "evaluator" in (r.stdout + r.stderr).lower()
    assert "REPRODUCE.md" in r.stdout + r.stderr


def test_bfcl_control_policy_is_genuinely_empty():
    """If the 'control' ever gained a gate, every published delta would be wrong."""
    pol = json.loads((REPO / "benchmarks/bfcl_v4/policy_control.json").read_text())
    assert not [k for k, v in (pol.get("gate_enabled") or {}).items() if v]
    assert not [k for k, v in pol.items() if k.startswith("enable_") and v]
    assert not [v for v in (pol.get("templates") or {}).values() if v.strip()]


def test_bfcl_anchors_policy_enables_all_four():
    pol = json.loads((REPO / "rounds/T5_A4_no_tool_call/policy.json").read_text())
    for switch in ("enable_capacity_repair", "enable_reroute",
                   "enable_redundant_write_suppress", "enable_zero_call_reprompt"):
        assert pol.get(switch) is True, f"{switch} must be on in the composed policy"


def test_bfcl_cases_flag_is_not_swallowed_by_passthrough():
    """Regression: `--cases X` once set `--cases-dir` and the job died 2 s into a GPU allocation.

    A bare `nargs="*"` positional absorbs the value of any flag the script does not define, so a
    real run failed on a compute node with "cases dir not found: memory_smoke_cases.json". `--cases`
    is now a declared flag and passthrough requires a literal `--`.
    """
    r = run(BENCH, "--compare", "--dry-run", "--cases", "memory_smoke_cases.json")
    assert r.returncode == 0, r.stderr
    # Resolved against --cases-dir, because the downstream runner wants a PATH: passing a bare
    # filename produced "[error] Cases file not found" AFTER the model had already loaded.
    assert "--cases data/memory_smoke_cases.json" in r.stdout
    assert "--cases-dir memory_smoke_cases.json" not in r.stdout


def test_bfcl_rejects_an_unknown_flag_instead_of_forwarding_it():
    """A typo must be an error here, not silently handed to the downstream runner."""
    r = run(BENCH, "--dry-run", "--caes", "typo")
    assert r.returncode != 0
    assert "unrecognized arguments" in r.stderr


def test_bfcl_forwards_extra_args_only_after_a_separator():
    r = run(BENCH, "--only", "anchors", "--dry-run", "--", "--counterfactual-eval-scope", "full")
    assert r.returncode == 0, r.stderr
    assert "--counterfactual-eval-scope full" in r.stdout
    assert " -- " not in r.stdout.split("command")[-1], "the separator itself must not be forwarded"


# ---------------------------------------------------------------------------------------------
# the vendored evaluator's import shim
# ---------------------------------------------------------------------------------------------

def test_vendored_evaluator_imports_as_anchoropt_via_the_shim():
    """Regression: the runner died on a compute node with ModuleNotFoundError: 'anchoropt'.

    The vendored code expects a package named `anchoropt` (that is its name in the working repo),
    but here that name belongs to the anchor library. _evaluator_path/ re-exports evaluator/ under
    the expected name so the vendored files stay byte-identical to their source.
    """
    bench = REPO / "benchmarks" / "bfcl_v4"
    env_path = os.pathsep.join([
        str(bench / "_evaluator_path"), str(bench / "harness"), str(bench),
    ])
    # harness/ is required: evaluator/memory_gates.py re-exports the canonical registry from
    # bfcl_eval behind a defensive import, so without it the adapter degrades and
    # `enabled_explicit_gate_keys` goes missing. run.py sets the same three entries.
    code = "import anchoropt.eval_common, anchoropt.memory_evaluator, anchoropt.traj_sidecar"
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, text=True,
                       capture_output=True, check=False,
                       env={**os.environ, "PYTHONPATH": env_path})
    assert r.returncode == 0, f"shim failed:\n{r.stderr}"


def test_the_shim_does_not_shadow_the_real_anchoropt_package():
    """Outside the runner, `anchoropt` must still be OUR library, not the vendored evaluator."""
    r = subprocess.run(
        [sys.executable, "-c",
         "from anchoropt.anchor import Action; print(sorted(a.value for a in Action))"],
        cwd=REPO, text=True, capture_output=True, check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "reroute" in r.stdout and "transform" not in r.stdout


def test_the_runner_itself_imports_under_the_shim():
    """`run_memory_eval.py --help` is the exact path that failed on the GPU node."""
    bench = REPO / "benchmarks" / "bfcl_v4"
    env_path = os.pathsep.join([
        str(bench / "_evaluator_path"), str(bench / "harness"), str(bench),
    ])
    r = subprocess.run([sys.executable, "run_memory_eval.py", "--help"], cwd=bench, text=True,
                       capture_output=True, check=False,
                       env={**os.environ, "PYTHONPATH": env_path})
    assert r.returncode == 0, f"runner cannot import:\n{r.stderr[-1500:]}"
    assert "--templates" in r.stdout


def test_inherited_harness_gate_is_off_in_both_conditions():
    """The vendored harness ships an earlier fork's gate registry, and with NO policy
    `on_core_clear_blocked` (that fork's G1) is ENABLED BY DEFAULT.

    Both of our policies must set gate_default=false, or the "control" would silently carry
    someone else's hand-written gate and every published delta would be measured against a
    contaminated baseline. Asserted against the harness's OWN resolver, not by reading JSON.
    """
    bench = REPO / "benchmarks" / "bfcl_v4"
    code = f'''
import sys, json
sys.path.insert(0, {str(bench / "harness")!r})
from bfcl_eval.model_handler.memory_gates import suppress_specs
ctl = json.load(open({str(bench / "policy_control.json")!r}))
anc = json.load(open({str(REPO / "rounds/T5_A4_no_tool_call/policy.json")!r}))
print("CTL", sorted(s.key for s in suppress_specs(ctl, "memory_kv")))
print("ANC", sorted(s.key for s in suppress_specs(anc, "memory_kv")))
print("BARE", sorted(s.key for s in suppress_specs({{}}, "memory_kv")))
'''
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, text=True,
                       capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    out = {}
    for line in r.stdout.strip().splitlines():
        k, _, v = line.partition(" ")
        out[k] = v

    assert out["CTL"] == "[]", f"control must suppress NOTHING, got {out['CTL']}"
    assert "on_core_clear_blocked" not in out["ANC"], "the inherited gate must stay off"
    # It used to be DEFAULT-ON with no policy, which is why the guard above exists. It has since been
    # RETIRED from the registry entirely, so a bare policy now suppresses nothing at all -- a strictly
    # stronger position than "off by default".
    assert out["BARE"] == "[]", (
        f"a bare policy should suppress nothing after retirement, got {out['BARE']}"
    )


def test_runner_rejects_a_policy_without_gate_default_false(tmp_path: Path):
    """A policy missing gate_default=false must be refused, not run against a dirty baseline."""
    bad = tmp_path / "policy_control.json"
    bad.write_text(json.dumps({"templates": {"on_memory_preamble": ""}}))
    probe = (
        "import sys, pathlib;"
        "sys.path.insert(0, str(pathlib.Path('benchmarks/bfcl_v4')));"
        "import run as R;"
        "R.CONDITIONS['control']['policy'] = pathlib.Path(sys.argv[1]);"
        "R.preflight(type('N', (), {'arms': ['control'], 'dry_run': True})())"
    )
    r = subprocess.run(
        [sys.executable, "-c", probe, str(bad)],
        cwd=REPO, text=True, capture_output=True, check=False,
    )
    assert r.returncode == 2, f"expected refusal, got {r.returncode}\n{r.stdout}{r.stderr}"
    assert "gate_default" in r.stderr


def test_only_our_four_gates_remain_in_the_harness_registry():
    """The ~30 manually tuned gates from the earlier line are RETIRED, not merely disabled.

    They were the groundwork -- they established that a local intervention can move this benchmark,
    and A1-A4 fire through the same dispatch machinery. But they are not part of THIS measurement,
    and a control carrying a hand-tuned gate is not a control for a learned one.
    """
    bench = REPO / "benchmarks" / "bfcl_v4"
    code = f'''
import sys
sys.path.insert(0, {str(bench / "harness")!r})
from bfcl_eval.model_handler.memory_gates import (
    MEMORY_GATE_REGISTRY as REG, REGISTRY_BY_KEY as R, gate_applies, gate_match_any)
print("KEYS", ",".join(sorted(g.key for g in REG)))
fired = [k for k in ("on_premature_idk", "on_core_clear_blocked", "on_blob_pressure",
                     "on_idk_fallback", "on_loop")
         if gate_applies(R[k], "memory_kv")
         or gate_match_any(R[k], ["core memory is full", "i do not know", "Key not found"])]
print("FIREABLE", ",".join(fired))
'''
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, text=True,
                       capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    out = {}
    for line in r.stdout.strip().splitlines():
        k, _, v = line.partition(" ")
        out[k] = v

    assert sorted(out["KEYS"].split(",")) == [
        "on_core_full_rerouted",        # A1
        "on_domain_error_core_full",    # A1
        "on_domain_error_key_not_found",  # A2
        "on_redundant_write_suppressed",  # A3
    ]
    # A retired key must resolve to an inert spec rather than raising -- base_handler.py still has
    # ~26 bracket lookups at the old call sites, and a KeyError there would crash mid-episode.
    assert out.get("FIREABLE", "") == "", f"a retired gate can still fire: {out.get('FIREABLE')}"


def test_retired_gate_note_credits_the_prior_work_and_flags_the_split():
    """The note must credit the groundwork AND state that the hand-tuned numbers are on a
    different split, so they are not read as a target A1-A4 missed."""
    doc = (REPO / "benchmarks/bfcl_v4/harness/bfcl_eval/model_handler/memory_gates.py").read_text()
    note = doc[doc.index("class _RetiredGates"):doc.index("def __missing__")]
    assert "PRIOR WORK THIS BUILDS ON" in note, "the prior work must be credited"
    assert "A1-A4 fire through that machinery" in note
    assert "DIFFERENT SPLIT" in note and "58.67" in note
    assert "not repudiated" in note


def test_the_published_fold_case_files_ship_and_have_the_right_counts():
    """The g8 balanced fold IS the published split. Its case files must be present.

    The repo previously shipped g8_balanced_manifest.json (which merely POINTS at the fold) without
    the case files, so anyone following REPRODUCE.md silently ran the 408-case W5 split instead and
    would have read the mismatch as a failed reproduction.
    """
    data = REPO / "benchmarks" / "bfcl_v4" / "data"
    expected = {"train": (387, 84, 303), "test": (111, 27, 84)}
    for split, (n, n_pre, n_scored) in expected.items():
        f = data / f"g8_balanced_{split}_cases.json"
        assert f.exists(), f"the published fold's {split} cases are missing: {f.name}"
        ids = json.loads(f.read_text())

        def key(i):
            return i if isinstance(i, str) else str(i.get("id", ""))

        pre = sum(1 for i in ids if "prereq" in key(i))
        assert (len(ids), pre, len(ids) - pre) == (n, n_pre, n_scored), (
            f"{f.name}: got {len(ids)} total / {pre} prereq / {len(ids) - pre} scored, "
            f"expected {n}/{n_pre}/{n_scored}"
        )


def test_reproduce_warns_which_split_to_pass():
    """Two splits exist and the default is the wrong one; the doc must say so before the commands."""
    doc = (REPO / "REPRODUCE.md").read_text()
    assert "g8_balanced_train_cases.json" in doc
    assert "408" in doc, "the other split's size must be named so the mismatch is recognisable"
    # It belongs at the TOP of Tier 3 -- that is the tier where a wrong split actually costs GPU
    # hours and produces a plausible-looking wrong answer.
    tier3 = doc.index("## Tier 3")
    warn = doc.index("Use the right split")
    assert tier3 < warn < tier3 + 900, "the split warning must open Tier 3, not trail it"


def test_a3s_predicate_module_is_importable_from_its_call_site():
    """A3 suppresses UNCONDITIONALLY if its predicate module cannot be imported.

    memory_evaluator.py builds `_pred` inside a try/except and then does
    `return _p(_c) if _p is not None else True`. So a failed import is NOT a no-op: `_hits()`
    degenerates to `gate_match` alone and every `archival_memory_add` is dropped, including genuinely
    new facts -- the exact trade A3's frozen spec refuses (it declines 4 of 59 rather than risk data
    loss). This shipped broken once, because the module was vendored to anchoropt/mechanisms/ while the
    call site searched <evaluator>/../scripts/.
    """
    ev = REPO / "benchmarks" / "bfcl_v4" / "evaluator" / "memory_evaluator.py"
    code = f'''
import sys
from pathlib import Path
f = Path({str(ev)!r}).resolve()
for d in (str(f.parents[3] / "anchoropt" / "mechanisms"),
          str(f.parent.parent / "scripts")):
    if d not in sys.path:
        sys.path.insert(0, d)
import redundant_write as rw
dup, _ = rw.is_redundant("value_a", "value_a")
new, _ = rw.is_redundant("value_a plus new detail", "value_a")
print("DUP", dup)
print("NEW", new)
'''
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, text=True,
                       capture_output=True, check=False)
    assert r.returncode == 0, f"A3's predicate is not importable from its call site:\n{r.stderr}"
    out = dict(line.split(" ", 1) for line in r.stdout.strip().splitlines())
    assert out["DUP"] == "True", "an exact duplicate must be suppressed"
    assert out["NEW"] == "False", "NEW INFORMATION MUST NOT BE SUPPRESSED -- this is the data-loss bug"


def test_both_redundant_write_import_sites_search_the_ported_location():
    """There are two call sites (A3 and the C2 kv variant); fixing one and not the other would leave
    a live over-suppression path behind whichever flag reaches it first."""
    src = (REPO / "benchmarks/bfcl_v4/evaluator/memory_evaluator.py").read_text()
    assert src.count('"anchoropt" / "mechanisms"') >= 2, (
        "both redundant_write import sites must search the ported mechanism location"
    )


def test_telemetry_flags_are_anchor_named_and_survive_the_sidecar():
    """Flags must name the anchor, and must reach the trajectory record.

    Two failure modes, both of which have bitten this project:
      * `g3_gate` / `g4_gate` were the OLD gate numbering and read as retired gates rather than as
        A1 and A2 -- confusing enough that I misreported them as stray firings.
      * a step_record key that matches no sidecar rule is silently dropped. traj_sidecar's own comment
        counts SEVEN prior occurrences of that defect, so a rename must be checked against the filter,
        not assumed.
    """
    bench = REPO / "benchmarks" / "bfcl_v4"
    code = f'''
import sys
sys.path.insert(0, {str(bench / "evaluator")!r})
sys.path.insert(0, {str(bench / "harness")!r})
import traj_sidecar as ts
from bfcl_eval.model_handler.memory_gates import MEMORY_GATE_REGISTRY as R
for g in R:
    if not g.telemetry_flag:
        continue
    keeps = g.telemetry_flag.endswith("_gate") or g.telemetry_flag.startswith(ts._FAMILY_PREFIXES)
    print(f"{{g.telemetry_flag}} {{g.key}} {{keeps}}")
'''
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, text=True,
                       capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    rows = [ln.split() for ln in r.stdout.strip().splitlines()]
    assert rows, "no telemetry flags found"

    flags = {flag for flag, _key, _keep in rows}
    assert not flags & {"g1_gate", "g3_gate", "g4_gate", "g5_gate"}, (
        f"legacy gate-numbered flags are back: {sorted(flags)}"
    )
    for flag, key, keep in rows:
        assert keep == "True", f"{flag} would be DROPPED by the sidecar filter"


# ---------------------------------------------------------------------------------------------
# --shard: per-backend runs, which is how the per-backend numbers get produced
# ---------------------------------------------------------------------------------------------

def test_shard_all_emits_one_command_per_backend_per_arm():
    """A 3-backend x 2-arm contrast is SIX independent jobs, and --dry-run must print all six."""
    r = run(BENCH, "--compare", "--shard", "all", "--dry-run")
    assert r.returncode == 0, r.stderr
    for backend in ("kv", "vector", "rec_sum"):
        assert f"[{backend}]" in r.stdout
        assert f"cases_{backend}.json" in r.stdout
    # Distinct out-dirs, or six concurrent jobs overwrite each other.
    for arm in ("control", "a8"):
        for backend in ("kv", "vector", "rec_sum"):
            assert f"{arm}_{backend}" in r.stdout


def test_ports_are_distinct_across_every_condition_and_backend():
    """A port table plus a fallback is how control/kv and a8/vector once landed on the same port.

    Derived from both coordinates instead, and checked over the FULL cross product rather than only
    the two arms a default --compare happens to select.
    """
    import sys

    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    from run import BACKENDS, CONDITIONS, shard_port

    seen: dict[int, tuple[str, str]] = {}
    for arm in sorted(CONDITIONS):
        for backend in BACKENDS:
            port = shard_port(arm, backend)
            assert port not in seen, f"port {port}: {seen[port]} collides with {(arm, backend)}"
            seen[port] = (arm, backend)
    assert len(seen) == len(CONDITIONS) * len(BACKENDS)


def test_each_shard_gets_its_own_port_and_cache():
    """Three collisions this prevents, each of which has bitten a real fleet."""
    r = run(BENCH, "--compare", "--shard", "all", "--dry-run")
    ports = {ln.split("=")[-1].strip() for ln in r.stdout.splitlines()
             if "ANCHOROPT_SHARD_PORT" in ln}
    assert len(ports) == 6, f"a port per (arm x backend), not per backend; got {sorted(ports)}"

    caches = {ln.split("=")[-1].strip() for ln in r.stdout.splitlines()
              if "ANCHOROPT_SNAPSHOT_CACHE" in ln}
    assert len(caches) == 6, "each shard needs its own snapshot cache"

    # The store fingerprint is SHARED within an arm -- that is what makes the shards comparable.
    prints = [ln.split("=")[-1].strip() for ln in r.stdout.splitlines()
              if "ANCHOROPT_STORE_CODE_FINGERPRINT" in ln]
    assert len(set(prints)) == 2, "one fingerprint per arm, shared across its three shards"


def test_shards_keep_workers_at_one():
    """Parallelism ACROSS isolated runs is measured at 0 flips; WITHIN a store build it is
    forbidden. Sharding must not be mistaken for permission to raise --workers."""
    r = run(BENCH, "--compare", "--shard", "all", "--dry-run")
    assert "--workers 1" in r.stdout
    assert "--workers 2" not in r.stdout
    low = r.stdout.lower()
    assert "within one store build is forbidden" in low or "forbidden" in low


def test_shard_says_to_submit_concurrently():
    """Running them in a loop is not the protocol and the output must say so."""
    r = run(BENCH, "--compare", "--shard", "all", "--dry-run")
    assert "SUBMIT THESE CONCURRENTLY" in r.stdout


def test_the_shard_partition_is_complete_and_disjoint():
    """Union must equal the corpus and the shards must not overlap, or scoring is wrong."""
    import json
    import sys
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    from run import BACKENDS, backend_of

    source = REPO / "benchmarks" / "bfcl_v4" / "data" / "g8_balanced_train_cases.json"
    cases = json.loads(source.read_text())
    buckets = {b: [c for c in cases if backend_of(c["id"]) == b] for b in BACKENDS}
    assert sum(len(v) for v in buckets.values()) == len(cases), "the partition must be complete"
    ids = [c["id"] for v in buckets.values() for c in v]
    assert len(ids) == len(set(ids)), "the shards must be disjoint"


def test_sharding_is_sound_because_no_dependency_crosses_a_backend():
    """The property that licenses sharding at all -- asserted, not assumed."""
    import json
    import sys
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    from run import backend_of

    source = REPO / "benchmarks" / "bfcl_v4" / "data" / "g8_balanced_train_cases.json"
    cases = json.loads(source.read_text())
    known = {c["id"] for c in cases}
    crossing = [
        (c["id"], dep) for c in cases for dep in (c.get("depends_on") or [])
        if dep in known and backend_of(dep) != backend_of(c["id"])
    ]
    assert not crossing, f"sharding would build stores from partial chains: {crossing[:3]}"


def test_write_shard_refuses_a_shard_that_is_not_self_contained():
    """If the corpus ever gains a cross-backend dependency, this must fail loudly rather than
    silently produce a store built from an incomplete chain."""
    import json
    import sys
    import tempfile

    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    from run import write_shard

    source = REPO / "benchmarks" / "bfcl_v4" / "data" / "g8_balanced_train_cases.json"
    cases = json.loads(source.read_text())
    kv = next(c for c in cases if "memory_kv" in c["id"])
    vector = next(c for c in cases if "vector" in c["id"])

    forged = [dict(c) for c in cases]
    for case in forged:
        if case["id"] == kv["id"]:
            case["depends_on"] = [vector["id"]]

    tmp = Path(tempfile.mkdtemp())
    (tmp / "forged.json").write_text(json.dumps(forged))
    with pytest.raises(SystemExit) as excinfo:
        write_shard(tmp / "forged.json", "kv", tmp / "out")
    assert "NOT self-contained" in str(excinfo.value)


# ---------------------------------------------------------------------------------------------
# Everything the README tells a collaborator to run must actually run
# ---------------------------------------------------------------------------------------------

RUNNABLE = [
    ("scripts/verify_progression.py",),
    ("scripts/walk_framework.py", "--step", "4"),
    ("scripts/run_pipeline.py", "--show"),
    ("scripts/run_pipeline.py", "--expect"),
    ("scripts/derive_per_backend.py",),
    ("benchmarks/bfcl_v4/run.py", "--compare", "--shard", "all", "--dry-run"),
]


@pytest.mark.parametrize("argv", RUNNABLE, ids=lambda a: a[0].split("/")[-1])
def test_the_no_gpu_commands_exit_zero(argv):
    """A README that lists a command which errors is worse than one that lists nothing."""
    r = run(*argv)
    assert r.returncode == 0, f"{' '.join(argv)} exited {r.returncode}:\n{r.stderr[-800:]}"


def test_every_runnable_command_is_named_in_the_readme():
    """And the converse: the table must not drift from what is actually verified here."""
    readme = (REPO / "README.md").read_text()
    for argv in RUNNABLE:
        assert argv[0] in readme, f"{argv[0]} is verified but not documented in the README"
