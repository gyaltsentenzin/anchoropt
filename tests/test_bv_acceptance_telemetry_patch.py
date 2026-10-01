"""The BV acceptance-telemetry patch must be safe on THREE different trees, not one.

This patch exists because the live isolated BV evaluator is a DIFFERENT LINEAGE from the vendored
sidecar: its `_FAMILY_PREFIXES` continues for another nine prefixes and its own notes record the
allowlist-drop defect as having recurred twelve times, not seven. The first version of the patch
anchored on `"recovery_txn",\n)` -- the vendored tuple's terminator -- and refused to apply there.
Refusing was the correct behaviour; the fix is to locate the tuple STRUCTURALLY so the patch does
not depend on which lineage it is looking at.

The three trees, and what each must do:

  vendored / already-fixed   the source-side commit declares these keys directly, so the patch must
                             report SATISFIED and write NOTHING. A marker-only guard appends a
                             second redundant copy here, and a patch that edits an already-correct
                             file is a patch whose report cannot be trusted.
  pre-fix (the BV shape)     patch, idempotently, inside the tuple.
  the SHARED tree            refuse outright -- running GEPA workers import it.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PATCH = REPO / "patches" / "bv" / "bv_acceptance_telemetry.py"
SIDECAR = REPO / "benchmarks" / "bfcl_v4" / "evaluator" / "traj_sidecar.py"

REQUIRED = ("mechanism_", "clears_added", "information_losing_removes", "verified_relocations")


def _mod():
    spec = importlib.util.spec_from_file_location("bv_accept_patch", PATCH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(path: pathlib.Path):
    return subprocess.run([sys.executable, str(PATCH), str(path)],
                          capture_output=True, text=True)


def _prefix_tuple(path: pathlib.Path) -> tuple[str, ...]:
    """Read the tuple's literal members without importing the module (it has heavy deps)."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_FAMILY_PREFIXES" for t in node.targets):
            return tuple(e.value for e in node.value.elts if isinstance(e, ast.Constant))
    raise AssertionError("no _FAMILY_PREFIXES assignment found")


@pytest.fixture()
def prefix_tree(tmp_path):
    """A copy of the vendored sidecar with the required keys STRIPPED -- i.e. the BV shape."""
    kept = [l for l in SIDECAR.read_text().splitlines(keepends=True)
            if '"mechanism_",' not in l
            and '"clears_added", "information_losing_removes", "verified_relocations",' not in l]
    p = tmp_path / "traj_sidecar.py"
    p.write_text("".join(kept))
    assert not set(REQUIRED) & set(_prefix_tuple(p)), "fixture failed to strip the keys"
    return p


def test_an_already_fixed_tree_reports_SATISFIED_and_is_left_byte_identical(tmp_path):
    p = tmp_path / "traj_sidecar.py"
    before = SIDECAR.read_bytes()
    p.write_bytes(before)
    r = _run(p)
    assert r.returncode == 0
    assert "SATISFIED" in r.stdout
    assert p.read_bytes() == before, "the patch edited a tree that was already correct"
    assert not list(tmp_path.glob("*.pre_accepttelem")), "backed up a file it did not change"


def test_a_pre_fix_tree_gains_every_required_key_inside_the_tuple(prefix_tree):
    assert _run(prefix_tree).returncode == 0
    members = _prefix_tuple(prefix_tree)
    for key in REQUIRED:
        assert key in members, f"{key!r} did not reach _FAMILY_PREFIXES"


def test_patching_keeps_the_file_parseable_and_the_pre_existing_prefixes_intact(prefix_tree):
    before = _prefix_tuple(prefix_tree)
    _run(prefix_tree)
    after = _prefix_tuple(prefix_tree)
    ast.parse(prefix_tree.read_text())
    assert set(before) <= set(after), "the patch dropped a prefix that was already declared"


def test_a_second_run_changes_nothing(prefix_tree):
    _run(prefix_tree)
    once = prefix_tree.read_bytes()
    r = _run(prefix_tree)
    assert r.returncode == 0 and "already applied" in r.stdout
    assert prefix_tree.read_bytes() == once


def test_it_refuses_the_shared_tree_by_path_without_reading_it():
    """The refusal must not depend on the shared tree existing on this machine."""
    r = _run(pathlib.Path("/some/path/anchoropt-wei/anchoropt/anchoropt/traj_sidecar.py"))
    assert r.returncode == 3
    assert "SHARED" in r.stdout


def test_it_refuses_rather_than_guessing_when_the_tuple_is_absent(tmp_path):
    """A divergent tree that renamed the allowlist must produce a refusal, not a silent no-op.

    Refusing is what surfaced the lineage difference in the first place. A patch that quietly
    succeeds on a file it did not understand is the failure mode this whole class of defect is
    made of.
    """
    p = tmp_path / "traj_sidecar.py"
    p.write_text("_STEP_FIELDS = ('a',)\n_SOMETHING_ELSE = (\n    'x',\n)\n")
    r = _run(p)
    assert r.returncode == 1
    assert "refusing to guess" in r.stdout


def test_the_semantic_check_ignores_keys_that_appear_only_in_comments(tmp_path):
    """A key NAMED IN A COMMENT is not a declared key.

    Every prefix in this allowlist is discussed in a comment above it, so a substring check over the
    whole tuple body would read the discussion as the declaration and skip a tree that still drops
    the telemetry -- reading prose as evidence, which is the exact error class this patch serves.
    """
    m = _mod()
    commented = ("_FAMILY_PREFIXES = (\n"
                 "    # we should add mechanism_ and clears_added and\n"
                 "    # information_losing_removes and verified_relocations one day\n"
                 "    \"recovery_txn\",\n)\n")
    assert not m.already_admits_everything(commented)


def test_the_semantic_check_recognises_the_real_declaration():
    m = _mod()
    assert m.already_admits_everything(SIDECAR.read_text())
