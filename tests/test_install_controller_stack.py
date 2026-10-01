"""The loader must be able to install a STACK, or the moving incumbent is inexpressible.

Driven through the REAL module and the REAL runtime hook, by subprocess with the real environment
variables -- because this project has twice concluded a guard was fine from a hand-rolled driver and
been wrong. Nothing here reimplements the loader.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def spec(name, locus="post_execution", field="error_kind", value="no_capacity", **kw):
    return dict({"name": name, "locus": locus, "eta": {"primitive": name},
                 "predicate": {"all": [{"field": field, "op": "eq", "value": value}]}}, **kw)


def run_loader(tmp_path, env_extra, probe):
    """Import the loader exactly as the BV runner does, then run `probe` against the live hook."""
    script = f"""
import sys
sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / "scripts")!r}]
import install_controller           # noqa: F401  -- imported for its side effect
from anchoropt.runtime_hook import installed, decide
{probe}
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "scripts")]))
    env.update(env_extra)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                       cwd=str(tmp_path), env=env)
    assert r.returncode == 0, f"loader failed:\nSTDOUT{r.stdout}\nSTDERR\n{r.stderr}"
    return r.stdout


def write(tmp_path, name, s):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(s))
    return str(p)


# ---------------------------------------------------------------------------------------------------
# THE PRE-EXISTING BEHAVIOUR MUST NOT CHANGE
# ---------------------------------------------------------------------------------------------------

def test_no_env_installs_nothing_and_says_so(tmp_path):
    out = run_loader(tmp_path, {"ANCHOROPT_CONTROLLER_SPEC": "", "ANCHOROPT_CONTROLLER_STACK": ""},
                     "print('N=%d' % len(installed()))")
    assert "N=0" in out
    assert "nothing installed (control arm)" in out


def test_a_single_SPEC_still_installs_exactly_one(tmp_path):
    """The existing arm command must behave byte-identically: one variable, one controller."""
    p = write(tmp_path, "a", spec("a1"))
    out = run_loader(tmp_path, {"ANCHOROPT_CONTROLLER_SPEC": p, "ANCHOROPT_CONTROLLER_STACK": ""},
                     "print('N=%d' % len(installed()))")
    assert "N=1" in out
    assert "installed a1 at post_execution" in out


# ---------------------------------------------------------------------------------------------------
# THE STACK
# ---------------------------------------------------------------------------------------------------

def test_a_stack_of_three_installs_all_three(tmp_path):
    paths = [write(tmp_path, n, spec(n)) for n in ("c1", "c2", "c3")]
    sp = tmp_path / "stack.json"
    sp.write_text(json.dumps(paths))
    out = run_loader(tmp_path, {"ANCHOROPT_CONTROLLER_STACK": str(sp),
                                "ANCHOROPT_CONTROLLER_SPEC": ""},
                     "print('N=%d' % len(installed()))")
    assert "N=3" in out
    assert "STACK of 3 installed" in out
    for n in ("c1", "c2", "c3"):
        assert f"installed {n} at" in out


def test_stack_and_candidate_COMPOSE_with_the_candidate_last(tmp_path):
    """The paired shape a round needs: incumbent carried forward, candidate on top."""
    sp = tmp_path / "stack.json"
    sp.write_text(json.dumps([write(tmp_path, "inc", spec("incumbent_ctl"))]))
    cand = write(tmp_path, "cand", spec("candidate_ctl"))
    out = run_loader(tmp_path, {"ANCHOROPT_CONTROLLER_STACK": str(sp),
                                "ANCHOROPT_CONTROLLER_SPEC": cand},
                     "print('NAMES=%s' % [c.name for c in installed('post_execution')])")
    assert "NAMES=['incumbent_ctl', 'candidate_ctl']" in out
    assert "COMPOSITION n=2" in out


def test_the_stack_ORDER_is_preserved_verbatim(tmp_path):
    """`decide()` is first-fire-wins, so order is part of the arm and must not be sorted."""
    sp = tmp_path / "stack.json"
    sp.write_text(json.dumps([write(tmp_path, "z", spec("z_first")),
                              write(tmp_path, "a", spec("a_second"))]))
    out = run_loader(tmp_path, {"ANCHOROPT_CONTROLLER_STACK": str(sp),
                                "ANCHOROPT_CONTROLLER_SPEC": ""},
                     "print('NAMES=%s' % [c.name for c in installed('post_execution')])")
    assert "NAMES=['z_first', 'a_second']" in out


def test_controllers_at_DIFFERENT_loci_both_install(tmp_path):
    sp = tmp_path / "stack.json"
    sp.write_text(json.dumps([
        write(tmp_path, "p", spec("post_ctl", locus="post_execution")),
        write(tmp_path, "g", spec("gen_ctl", locus="post_generation_pre_exec"))]))
    out = run_loader(tmp_path, {"ANCHOROPT_CONTROLLER_STACK": str(sp),
                                "ANCHOROPT_CONTROLLER_SPEC": ""},
                     "print('POST=%d GEN=%d' % (len(installed('post_execution')),"
                     " len(installed('post_generation_pre_exec'))))")
    assert "POST=1 GEN=1" in out


def test_a_stacked_controller_ACTUALLY_FIRES_through_the_real_hook(tmp_path):
    """Installation is not execution. The stack is only real if `decide()` routes to it."""
    sp = tmp_path / "stack.json"
    sp.write_text(json.dumps([write(tmp_path, "c", spec("cap_ctl", value="no_capacity"))]))
    out = run_loader(
        tmp_path,
        {"ANCHOROPT_CONTROLLER_STACK": str(sp), "ANCHOROPT_CONTROLLER_SPEC": ""},
        "d = decide('post_execution', {'error_kind': 'no_capacity'}, host_default=False)\n"
        "print('FIRED=%s SRC=%s WHO=%s' % (d.proceed, d.source, getattr(d.controller,'name',None)))\n"
        "n = decide('post_execution', {'error_kind': 'other'}, host_default=False)\n"
        "print('NEG=%s' % n.proceed)")
    assert "FIRED=True SRC=controller WHO=cap_ctl" in out
    assert "NEG=False" in out


def test_the_SECOND_controller_fires_when_the_first_does_not(tmp_path):
    """First-fire-wins must not mean first-installed-shadows-everything."""
    sp = tmp_path / "stack.json"
    sp.write_text(json.dumps([write(tmp_path, "a", spec("first", value="never_matches")),
                              write(tmp_path, "b", spec("second", value="no_capacity"))]))
    out = run_loader(
        tmp_path, {"ANCHOROPT_CONTROLLER_STACK": str(sp), "ANCHOROPT_CONTROLLER_SPEC": ""},
        "d = decide('post_execution', {'error_kind': 'no_capacity'}, host_default=False)\n"
        "print('WHO=%s' % getattr(d.controller, 'name', None))")
    assert "WHO=second" in out


def test_an_object_form_stack_with_provenance_is_accepted(tmp_path):
    sp = tmp_path / "stack.json"
    sp.write_text(json.dumps({"incumbent": "R4", "stack_fingerprint": "stack_abc123",
                              "controllers": [write(tmp_path, "c", spec("obj_ctl"))]}))
    out = run_loader(tmp_path, {"ANCHOROPT_CONTROLLER_STACK": str(sp),
                                "ANCHOROPT_CONTROLLER_SPEC": ""},
                     "print('N=%d' % len(installed()))")
    assert "N=1" in out


def test_inline_specs_in_the_stack_are_installed(tmp_path):
    sp = tmp_path / "stack.json"
    sp.write_text(json.dumps({"controllers": [spec("inline_ctl")]}))
    out = run_loader(tmp_path, {"ANCHOROPT_CONTROLLER_STACK": str(sp),
                                "ANCHOROPT_CONTROLLER_SPEC": ""},
                     "print('N=%d' % len(installed()))")
    assert "N=1" in out and "installed inline_ctl" in out


def test_a_malformed_stack_fails_LOUDLY_rather_than_installing_nothing(tmp_path):
    """A silent no-op that reads as success is the defect class this project keeps paying for: the
    arm would become the control and report as the arm."""
    sp = tmp_path / "stack.json"
    sp.write_text(json.dumps({"controllers": "not-a-list"}))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "scripts")]),
               ANCHOROPT_CONTROLLER_STACK=str(sp), ANCHOROPT_CONTROLLER_SPEC="")
    r = subprocess.run([sys.executable, "-c", "import install_controller"],
                       capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert r.returncode != 0
    assert "must be a JSON list" in (r.stdout + r.stderr)


def test_a_missing_stack_file_fails_loudly(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "scripts")]),
               ANCHOROPT_CONTROLLER_STACK=str(tmp_path / "absent.json"),
               ANCHOROPT_CONTROLLER_SPEC="")
    r = subprocess.run([sys.executable, "-c", "import install_controller"],
                       capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert r.returncode != 0
    assert "FileNotFoundError" in (r.stdout + r.stderr)


# ---------------------------------------------------------------------------------------------------
# LOADER IDENTITY -- the defect that voided a 3-arm GPU round
# ---------------------------------------------------------------------------------------------------

def test_a_wrong_loader_location_is_FATAL_not_a_silent_control(tmp_path):
    """run_memory_eval.py is invoked by path from a SHARED tree, so sys.path[0] beats PYTHONPATH and
    `install_controller` resolved to the shared namesake -- which had no stack support. The arms
    installed only their candidate, the control installed nothing, and the log still said
    "STACK of 1 installed" because a PROBE process had imported the isolated copy. Zero relocate_*
    gates fired. An assertion here is the difference between a fatal error and a wrong measurement."""
    p = write(tmp_path, "a", spec("a1"))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "scripts")]),
               ANCHOROPT_CONTROLLER_SPEC=p, ANCHOROPT_CONTROLLER_STACK="",
               ANCHOROPT_EXPECT_LOADER_UNDER="/definitely/not/where/this/lives")
    r = subprocess.run([sys.executable, "-c", "import install_controller"],
                       capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert r.returncode != 0, "a loader in the wrong tree must be fatal"
    out = r.stdout + r.stderr
    assert "LOADER IDENTITY FAILED" in out
    assert "measured as the control" in out


def test_the_expectation_PASSES_when_the_loader_is_where_it_should_be(tmp_path):
    p = write(tmp_path, "a", spec("a1"))
    out = run_loader(tmp_path, {"ANCHOROPT_CONTROLLER_SPEC": p, "ANCHOROPT_CONTROLLER_STACK": "",
                                "ANCHOROPT_EXPECT_LOADER_UNDER": str(ROOT / "scripts")},
                     "print('N=%d' % len(installed()))")
    assert "N=1" in out


def test_no_expectation_set_means_no_assertion(tmp_path):
    """Callers that do not opt in must be unaffected."""
    p = write(tmp_path, "a", spec("a1"))
    out = run_loader(tmp_path, {"ANCHOROPT_CONTROLLER_SPEC": p, "ANCHOROPT_CONTROLLER_STACK": ""},
                     "print('N=%d' % len(installed()))")
    assert "N=1" in out
