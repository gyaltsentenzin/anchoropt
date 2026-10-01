"""The GUARANTEE: telemetry cannot affect a decision, enforced structurally.

A promise in a docstring is not a constraint. These tests fail if a future change imports
`anchoropt.telemetry` from a live decision path, so the separation survives people forgetting it.
"""

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
LIVE_DIRS = (
    REPO / "anchoropt" / "learning",
    REPO / "benchmarks" / "bfcl_v4",
    REPO / "integrations",
    REPO / "rounds",
)
TELEMETRY = REPO / "anchoropt" / "telemetry"


def _py_files(root: pathlib.Path):
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts] if root.exists() else []


def test_no_live_decision_path_imports_telemetry():
    """The load-bearing test. Telemetry is observational only if nothing live depends on it."""
    offenders = []
    for d in LIVE_DIRS:
        for f in _py_files(d):
            src = f.read_text()
            if re.search(r"^\s*(from\s+anchoropt\.telemetry|import\s+anchoropt\.telemetry)",
                         src, re.M):
                offenders.append(str(f.relative_to(REPO)))
    assert not offenders, (
        "telemetry is imported by a live decision path, so it is no longer observational: "
        f"{offenders}")


def test_telemetry_never_mutates_what_it_reads():
    """No assignment into a caller's mapping/sequence, and no policy/eta writing."""
    banned = (".pop(", ".setdefault(", ".clear(", ".update(")
    problems = []
    for f in _py_files(TELEMETRY):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#")[0]
            # dict(...) copies are fine; in-place mutation of an argument is not.
            for b in banned:
                if b in code and "dict(" not in code and "self." not in code:
                    problems.append(f"{f.name}:{i}: {line.strip()}")
    assert not problems, f"telemetry appears to mutate its inputs: {problems}"


def test_telemetry_does_not_import_an_llm_client_or_write_policies():
    """It records rounds; it must not call a model or emit a policy file."""
    for f in _py_files(TELEMETRY):
        src = f.read_text()
        assert "anthropic" not in src.lower(), f"{f.name} references an LLM SDK"
        assert "propose_semantic" not in src, f"{f.name} calls the proposer"
        assert "build_arms" not in src or "observe_build" in src, (
            f"{f.name} must only OBSERVE build_arms output, never call it")


def test_locality_reuses_the_frozen_eligibility_module():
    """It must not restate the frozen definitions -- restating is how two copies drift apart."""
    src = (TELEMETRY / "locality.py").read_text()
    assert "from anchoropt.learning.prestate_eligibility import" in src
    assert "engagement_split" in src
    # and it must NOT define its own version of them
    assert "def engagement_split" not in src
    assert "def qualifying_in_control" not in src


def test_roles_are_protocols_only():
    """Interfaces, not behaviour: every method body must be a docstring and/or `...`.

    Checked with AST, not by grepping lines -- the first version of this test matched the module
    docstring's own role diagram and failed on prose, which is exactly the kind of test that gets
    deleted instead of fixed.
    """
    import ast
    tree = ast.parse((TELEMETRY / "roles.py").read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert classes, "roles.py defines no protocols"
    for cls in classes:
        assert any(isinstance(b, ast.Name) and b.id == "Protocol"
                   or isinstance(b, ast.Attribute) and b.attr == "Protocol"
                   for b in cls.bases), f"{cls.name} is not a Protocol"
        for fn in [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for stmt in fn.body:
                is_doc = isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                    and isinstance(stmt.value.value, str)
                is_ellipsis = isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                    and stmt.value.value is Ellipsis
                assert is_doc or is_ellipsis, (
                    f"{cls.name}.{fn.name} has an implementation statement "
                    f"({type(stmt).__name__}) -- roles must be interfaces only")


def test_the_live_files_r3_depends_on_are_untouched_by_this_change():
    """These are the files the running seed-2 experiment resolves against."""
    for rel in ("anchoropt/learning/anchor_policy_opt.py",
                "anchoropt/learning/action_contract.py",
                "benchmarks/bfcl_v4/bfcl_runtime.py"):
        src = (REPO / rel).read_text()
        # The forbidden thing is a DEPENDENCY on the telemetry package, not the word. bfcl_runtime now
        # carries `telemetry_flag` keys naming where each executor records its firings -- that is a
        # fact about the host, needed to read a run, and it imports nothing.
        assert "anchoropt.telemetry" not in src, f"{rel} imports the telemetry package"
        assert "from anchoropt import telemetry" not in src, f"{rel} imports the telemetry package"
