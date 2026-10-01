"""The step budget must be 20 everywhere, and 20 is not an arbitrary choice.

Official BFCL v4 sets MAXIMUM_STEP_LIMIT = 20. The published A1-A4 numbers were produced at 15 --
divergence #3 in docs/FIDELITY_AUDIT.md -- and that is the leading (untested) hypothesis for why the
control here reads 90/303 against a published 88/303. Every default in this repo is now 20, matching
the official harness.

These tests exist because "every runner defaults to 20" is invisible in review: a single
`default=15` in one of five files would silently produce numbers that look fine and are not
comparable to anything. The assertion is against the VENDORED HARNESS CONSTANT, not a literal, so
the repo tracks the official value rather than a number someone typed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BFCL = REPO / "benchmarks" / "bfcl_v4"


def official_step_limit() -> int:
    """MAXIMUM_STEP_LIMIT from the vendored official harness -- the source of truth."""
    src = (BFCL / "harness/bfcl_eval/constants/default_prompts.py").read_text()
    m = re.search(r"^MAXIMUM_STEP_LIMIT\s*=\s*(\d+)", src, re.MULTILINE)
    assert m, "MAXIMUM_STEP_LIMIT not found in the vendored harness"
    return int(m.group(1))


def test_official_step_limit_is_twenty():
    """Guards the premise. If upstream ever changed this, every assertion below should be revisited."""
    assert official_step_limit() == 20


def _argparse_default(path: Path, flag: str) -> int:
    """Extract the `default=` of an add_argument(flag, ...) call without importing the module."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == flag):
            for kw in node.keywords:
                if kw.arg == "default":
                    assert isinstance(kw.value, ast.Constant), f"{path.name}: {flag} default not a literal"
                    return kw.value.value
            raise AssertionError(f"{path.name}: {flag} has no default")
    raise AssertionError(f"{path.name}: no add_argument({flag!r})")


def test_every_runner_defaults_to_the_official_budget():
    """All three CLIs. A user who passes no --max-steps must get official behaviour."""
    expected = official_step_limit()
    for rel in ("run.py", "run_memory_eval.py", "run_memory_train.py"):
        got = _argparse_default(BFCL / rel, "--max-steps")
        assert got == expected, (
            f"{rel} defaults --max-steps to {got}, official is {expected}. "
            "A 15 here silently reproduces fidelity divergence #3."
        )


def _kwarg_default(path: Path, cls: str, param: str) -> int:
    """The default of a keyword-only/positional param on cls.__init__."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    a = fn.args
                    names = [x.arg for x in a.args[1:]]          # drop self
                    defaults = list(a.defaults)
                    pad = len(names) - len(defaults)
                    mapping = {n: d for n, d in zip(names[pad:], defaults)}
                    mapping.update({x.arg: d for x, d in zip(a.kwonlyargs, a.kw_defaults) if d})
                    assert param in mapping, f"{cls}.__init__ has no default for {param}"
                    v = mapping[param]
                    assert isinstance(v, ast.Constant), f"{cls}.{param} default not a literal"
                    return v.value
    raise AssertionError(f"{path.name}: class {cls} not found")


def test_evaluator_constructor_defaults_to_the_official_budget():
    """The evaluators are importable directly -- a library caller that omits the kwarg gets 20 too."""
    expected = official_step_limit()
    for rel, cls in (
        ("evaluator/memory_evaluator.py", "MemoryAnchorOptEvaluator"),
        ("evaluator/multiturn_evaluator.py", "AnchorOptEvaluator"),
    ):
        got = _kwarg_default(BFCL / rel, cls, "max_steps_per_turn")
        assert got == expected, f"{cls}.max_steps_per_turn defaults to {got}, official is {expected}"


def test_no_stray_fifteen_step_budget():
    """Belt-and-braces: no `max_steps... = 15` anywhere in our code (vendored harness excluded)."""
    ours = [
        BFCL / "run.py", BFCL / "run_memory_eval.py", BFCL / "run_memory_train.py",
        BFCL / "evaluator/memory_evaluator.py", BFCL / "evaluator/multiturn_evaluator.py",
    ]
    bad = re.compile(r"max[_-]steps[a-z_]*\s*[=:]\s*15\b")
    for p in ours:
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "#" in line:
                line = line.split("#", 1)[0]      # a comment may legitimately discuss 15
            assert not bad.search(line), f"{p.name}:{i} pins a step budget to 15"
