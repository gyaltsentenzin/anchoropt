"""Import shim: makes the vendored evaluator importable as `anchoropt.*`.

WHY THIS EXISTS
---------------
The vendored runner does `from anchoropt.eval_common import ...`, `from anchoropt.memory_evaluator
import ...`, and six more like them -- because in the working repo those modules live in a package
called `anchoropt`. In THIS repo that name is already taken by the anchor library (anchor.py,
attribution/, learning/, mechanisms/), which is a completely different thing.

Rewriting the vendored imports was the obvious alternative and is the wrong one: it would touch ~14
files whose measured behaviour every frozen result depends on, and destroy the diff against the
source (see ../VENDORED.md). So instead this directory is prepended to sys.path *only* when the
evaluator runs, and this package re-exports the vendored modules under the name they expect.

The two module sets are disjoint apart from `__init__.py`, so nothing is shadowed: the evaluator sees
eval_common / memory_evaluator / template_engine / traj_sidecar / ..., and the anchor library keeps
its own namespace for everyone else.

Discovered by running a real job -- it failed in 3 seconds on a compute node with
`ModuleNotFoundError: No module named 'anchoropt'`, which no static check or --dry-run had caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The real vendored modules live one level up, in `evaluator/`.
_VENDORED = Path(__file__).resolve().parent.parent.parent / "evaluator"

if not _VENDORED.is_dir():  # pragma: no cover - a broken checkout, not a runtime path
    raise ImportError(
        f"vendored evaluator not found at {_VENDORED}. This shim only makes sense inside "
        "benchmarks/bfcl_v4/; see benchmarks/bfcl_v4/VENDORED.md."
    )

# Make `anchoropt.<mod>` resolve to `evaluator/<mod>.py`. Using the package __path__ rather than
# copying or symlinking keeps ONE copy of each vendored file on disk, so the diff against the source
# stays exact.
__path__ = [str(_VENDORED)]

# ROBUSTNESS: if Python resolved `anchoropt` to the repo's OWN package instead of this shim -- which
# happens whenever the process runs with the repo root ahead of this directory on sys.path, e.g. cwd
# is the repo root -- then `anchoropt.eval_common` would not be found and the failure would look
# identical to the compute-node crash this shim fixes. Extend the search path in that case rather
# than depending on path ORDER, which callers control and get wrong.
_own = Path(__file__).resolve().parents[3] / "anchoropt"
if _own.is_dir() and str(_VENDORED) not in [str(Path(p).resolve()) for p in __path__[1:]]:
    __path__.append(str(_own))

# `evaluator/__init__.py` carries the package's own exports; run it in this namespace so anything it
# defines (version constants, re-exports) is visible as `anchoropt.X` too.
_init = _VENDORED / "__init__.py"
if _init.is_file():
    exec(compile(_init.read_text(), str(_init), "exec"), globals())  # noqa: S102

sys.modules.setdefault(__name__, sys.modules[__name__])
