"""The relocation patch must apply cleanly AND land somewhere reachable.

Two GPU rounds were lost to a branch placed after a turn-ending exit, certified by a marker-only
preflight. A marker proves text was inserted; it does not prove the interpreter can get there. So this
checks LINE ORDER and INDENTATION against a sibling gate that is known to fire.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PATCH = REPO / "patches" / "bv" / "bv_generic_capacity_relocate.py"
EVAL = REPO / "benchmarks" / "bfcl_v4" / "evaluator" / "memory_evaluator.py"
MARKER = "# [anchoropt-patch:generic_capacity_relocate]"


@pytest.fixture(scope="module")
def patched(tmp_path_factory):
    d = tmp_path_factory.mktemp("reloc")
    target = d / "memory_evaluator.py"
    shutil.copy(EVAL, target)
    r = subprocess.run([sys.executable, str(PATCH), str(target)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return target


def test_the_patch_applies_and_the_result_is_valid_python(patched):
    import ast
    ast.parse(patched.read_text())


def test_the_patch_is_idempotent(patched):
    r = subprocess.run([sys.executable, str(PATCH), str(patched)], capture_output=True, text=True)
    assert r.returncode == 0 and "already applied" in r.stdout


def test_the_patch_refuses_the_shared_tree(tmp_path):
    """The shared BV evaluator is imported by running GEPA workers."""
    d = tmp_path / "anchoropt-wei" / "anchoropt" / "anchoropt"
    d.mkdir(parents=True)
    t = d / "memory_evaluator.py"
    shutil.copy(EVAL, t)
    r = subprocess.run([sys.executable, str(PATCH), str(t)], capture_output=True, text=True)
    assert r.returncode == 3 and "REFUSING" in r.stdout


def _lines(p):
    return p.read_text().splitlines()


def test_the_bounded_counter_is_in_scope_before_the_guard(patched):
    src = _lines(patched)
    counter = next(i for i, l in enumerate(src, 1) if "_relocations_done = 0" in l)
    guard = next(i for i, l in enumerate(src, 1) if "and _reloc_requested" in l)
    assert counter < guard, "the episode counter must be declared before the branch that reads it"


def test_the_guard_sits_at_the_SAME_INDENTATION_as_a_gate_known_to_fire(patched):
    """The char-reduction gate at this cell fires 21 times in a measured round.

    Equal indentation in the same method body is the structural evidence that the new branch is
    reachable -- which a marker check cannot give.
    """
    src = _lines(patched)
    reduce_i = next(i for i, l in enumerate(src)
                    if 'and remedy_enabled(templates, "enable_capacity_repair")' in l)
    reloc_i = next(i for i, l in enumerate(src) if "and _reloc_requested" in l)
    ind = lambda s: len(s) - len(s.lstrip())
    assert ind(src[reloc_i]) == ind(src[reduce_i])
    assert reloc_i < reduce_i, "relocation is inserted as the preceding sibling"


def test_the_identity_gate_is_present_and_names_the_relocation_capability(patched):
    """This cell has more than one executor; an unrelated controller must not relocate."""
    txt = patched.read_text()
    assert "relocate_entry_preserving_information_then_retry" in txt
    assert "relocate_declined_identity_gate" in txt


def test_the_patch_adds_no_safety_logic_of_its_own(patched):
    """Safety lives in scripts/capacity_relocate.py, verified against live state there."""
    txt = patched.read_text()
    seg = txt[txt.index(MARKER):txt.index('and remedy_enabled(templates, "enable_capacity_repair")')]
    for invented in ("archival_memory_add(", "core_memory_remove("):
        assert invented not in seg, f"the patch builds its own store call: {invented!r}"
    assert "relocate_and_retry" in seg, "it must delegate to the verified primitive"


def test_the_patch_refuses_to_guess_when_an_anchor_is_missing(tmp_path):
    t = tmp_path / "memory_evaluator.py"
    t.write_text("print('nothing to anchor on')\n")
    r = subprocess.run([sys.executable, str(PATCH), str(t)], capture_output=True, text=True)
    assert r.returncode == 1 and "refusing to guess" in r.stdout
