"""EVERY CORE MODULE MUST IMPORT, with no benchmark directory and no ambient sys.path luck.

FOUND IN A CLEAN CHECKOUT, not in the dev tree. `anchoropt/attribution/destination_attest.py` did
`sys.path.insert(0, <its own dir>)` then `import policy_tree` -- and `policy_tree` lives in
anchoropt/learning/, so that import could only ever raise. Nothing caught it because no test imported
the module and its CLI was always run from the repo root, where the ambient path happened to resolve it.

That is the whole defect class this file closes: a core module that imports only by accident of where
you started Python. A collaborator who installs the package and imports it from their own tree gets a
`ModuleNotFoundError` on our code.

The second check is the release claim: core must not need `benchmarks/`. It is one instance of the
adapter interface, and a shared core that cannot be imported without it is not shared.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import pkgutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Modules that legitimately exit at import time because they are SCRIPTS with module-level argument
# handling. They are exercised through their CLI, not by import, and are listed here explicitly so the
# exemption is visible rather than a silent `except SystemExit`.
_SCRIPT_MODULES = frozenset({"anchoropt.learning.screen_ranked_loci"})

# The modules a collaborator actually needs. These are checked in a SUBPROCESS with benchmarks/ hidden,
# which is the only way to prove the independence claim rather than assert it.
RELEASE_CRITICAL = (
    "anchoropt.anchor",
    "anchoropt.runtime",
    "anchoropt.runtime_hook",
    "anchoropt.learning.structured_search",
    "anchoropt.learning.anchor_policy_opt",
    "anchoropt.learning.action_contract",
    "anchoropt.learning.executor_capability",
    "anchoropt.learning.external_evaluation",
    "anchoropt.learning.search_state",
    "anchoropt.learning.termination",
    "anchoropt.learning.boundary_search",
    "anchoropt.learning.residual_loop",
    "anchoropt.learning.learned_signal",
    "anchoropt.learning.policy_class",
    "anchoropt.testing",
)


def _all_core_modules() -> list[str]:
    """Modules that are REALLY part of the package -- i.e. backed by a file under anchoropt/.

    `pkgutil.walk_packages` reports whatever is importable under the package's `__path__`, and the
    ported `sys.path.insert` calls above add `benchmarks/bfcl_v4/evaluator/` to it. Without this filter
    the walk discovers `memory_evaluator` and `memory_gates` -- cluster-side evaluator files that are
    not core at all -- as `anchoropt.*`, and reports their import errors against core. That stray
    discovery is itself a symptom of the debt, and the ratchet above is what bounds it.
    """
    import anchoropt
    root = Path(anchoropt.__file__).resolve().parent
    names = []
    for m in pkgutil.walk_packages(anchoropt.__path__, "anchoropt."):
        rel = m.name[len("anchoropt."):].replace(".", "/")
        if (root / f"{rel}.py").exists() or (root / rel / "__init__.py").exists():
            names.append(m.name)
    return names


def test_every_core_module_imports():
    """No module may depend on ambient sys.path or on the directory Python started in."""
    failures = []
    for name in _all_core_modules():
        if name in _SCRIPT_MODULES:
            continue
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                importlib.import_module(name)
        except BaseException as exc:                      # SystemExit is not an Exception
            failures.append(f"{name}: {type(exc).__name__}: {str(exc)[:120]}")
    assert not failures, "core modules that do not import:\n  " + "\n  ".join(failures)


# EXISTING sys.path DEBT, frozen as a RATCHET rather than fixed in this release.
#
# These 7 modules were ported VERBATIM from the working repo, where their measured behaviour is what
# every frozen result rests on, and pyproject.toml records the policy: they stay byte-diffable against
# their source, so a cosmetic rewrite is a bad trade. The pattern is nonetheless a real hazard -- it is
# what made `destination_attest` import a module that was never there, and what lets
# `benchmarks/bfcl_v4/evaluator/` modules be discovered as if they were `anchoropt.*`.
#
# The ratchet: this list may SHRINK, never grow. A new core module doing this fails the test below.
_SYS_PATH_DEBT = frozenset({
    "anchoropt/attribution/attribution_miner.py",
    "anchoropt/attribution/op_canonicalize_locus.py",
    "anchoropt/attribution/trace_backward.py",
    "anchoropt/learning/expand_attribution.py",
    "anchoropt/learning/phase_switch.py",
    "anchoropt/learning/remine_incumbent.py",
    "anchoropt/learning/screen_ranked_loci.py",
})


def _sys_path_offenders() -> set[str]:
    out = set()
    for py in (REPO / "anchoropt").rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        for line in py.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue                      # a comment recording the pattern is not the pattern
            if "sys.path.insert" in stripped:
                out.add(str(py.relative_to(REPO)))
                break
    return out


def test_no_NEW_core_module_inserts_its_own_directory_on_sys_path():
    """The mechanism behind the bug, refused for anything new.

    A `sys.path.insert` inside core makes an import resolve differently depending on where the process
    started -- and when it points at the wrong directory, as `destination_attest` did, the import can
    never work at all. Use a package-relative import instead.

    The 7 verbatim-ported modules are grandfathered (see `_SYS_PATH_DEBT` above and the rationale in
    pyproject.toml). This is a RATCHET: the debt list may shrink, never grow.
    """
    offenders = _sys_path_offenders()
    new = offenders - _SYS_PATH_DEBT
    assert not new, (
        "NEW core module(s) manipulating sys.path -- use a package-relative import:\n  "
        + "\n  ".join(sorted(new)))


def test_the_sys_path_debt_list_does_not_go_stale():
    """A module that stopped doing this must leave the list, or the ratchet loosens silently."""
    stale = _SYS_PATH_DEBT - _sys_path_offenders()
    assert not stale, (
        "these no longer manipulate sys.path -- remove them from _SYS_PATH_DEBT so the ratchet "
        f"keeps tightening: {sorted(stale)}")


def test_no_release_critical_module_manipulates_sys_path():
    """The modules a collaborator actually imports must be clean, debt list or not."""
    files = {m.replace(".", "/") + ".py" for m in RELEASE_CRITICAL}
    files |= {m.replace(".", "/") + "/__init__.py" for m in RELEASE_CRITICAL}
    dirty = _sys_path_offenders() & files
    assert not dirty, f"release-critical modules manipulating sys.path: {sorted(dirty)}"


def test_the_release_critical_modules_import_with_NO_benchmarks_directory():
    """THE INDEPENDENCE CLAIM, proven rather than asserted.

    Run in a subprocess with a `benchmarks` import blocker installed, so any core module reaching for
    the one real adapter fails loudly here instead of in a collaborator's tree.
    """
    prog = textwrap.dedent(f"""
        import sys, importlib

        class _Blocker:
            def find_module(self, name, path=None):
                return self if name.split(".")[0] in ("benchmarks", "bfcl_runtime",
                                                      "bfcl_signals", "bfcl_capabilities") else None
            def load_module(self, name):
                raise ImportError("benchmarks/ is deliberately unavailable in this check")
        sys.meta_path.insert(0, _Blocker())

        for m in {RELEASE_CRITICAL!r}:
            importlib.import_module(m)
        print("OK", len({RELEASE_CRITICAL!r}))
    """)
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                         cwd=str(REPO), timeout=300)
    assert out.returncode == 0, f"core needs benchmarks/:\n{out.stderr[-1500:]}"
    assert out.stdout.startswith("OK"), out.stdout


def test_the_toy_host_example_needs_no_benchmark_adapter():
    """The reference implementation must stand alone -- it is what a porter copies."""
    prog = textwrap.dedent("""
        import sys
        class _Blocker:
            def find_module(self, name, path=None):
                return self if name.split(".")[0] in ("benchmarks", "bfcl_runtime") else None
            def load_module(self, name):
                raise ImportError("unavailable")
        sys.meta_path.insert(0, _Blocker())
        sys.path.insert(0, "examples/toy_host")
        import toy_adapter
        from toy_host.runtime import run_corpus
        from anchoropt.testing import check_adapter_contract
        rep = check_adapter_contract(toy_adapter.ADAPTER)
        assert rep.ok, rep.summary()
        assert run_corpus(None)["n_solved"] == 4
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                         cwd=str(REPO), timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    assert "OK" in out.stdout


def test_script_module_exemptions_are_still_accurate():
    """An exemption that stops being needed must be removed, not left as cover.

    If a listed module now imports cleanly, delete it from `_SCRIPT_MODULES` -- otherwise the list
    silently grows into a place where real breakage hides.
    """
    for name in _SCRIPT_MODULES:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                importlib.import_module(name)
        except BaseException:
            continue                       # still a script; the exemption is earned
        pytest.fail(f"{name} now imports cleanly -- remove it from _SCRIPT_MODULES")
