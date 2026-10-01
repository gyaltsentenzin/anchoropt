"""AnchorOpt: decision-centric counterfactual policy learning for tool-using LLM agents.

The anchor library lives here -- `anchor` (the four-tuple and the incision points), `mechanisms/`,
`attribution/`, `learning/`.

ONE PIECE OF PATH PLUMBING, and it earns its place:

The vendored BFCL evaluator (benchmarks/bfcl_v4/evaluator/) does `from anchoropt.eval_common import
...` and seven more like it, because in the working repo those modules live in a package with this
same name. Rewriting those imports would touch ~14 files whose measured behaviour every frozen result
depends on, and would break the diff against their source (see benchmarks/bfcl_v4/VENDORED.md).

So when the vendored evaluator is present, this package extends its own search path to include it.
`anchoropt.anchor` still resolves here; `anchoropt.eval_common` resolves there. The two module sets
are disjoint, so nothing is shadowed either way.

Doing it here rather than only in a sys.path shim means it works regardless of path ORDER -- which
callers control and get wrong. The failure it prevents is not hypothetical: a real job died three
seconds into a GPU allocation with `ModuleNotFoundError: No module named 'anchoropt'`.

NOTE what this does NOT do. `anchoropt.memory_evaluator` additionally needs the vendored HARNESS on
sys.path, because `evaluator/memory_gates.py` is a thin adapter that re-exports the canonical gate
registry from `bfcl_eval.model_handler.memory_gates` behind a DEFENSIVE import -- upstream's design,
so that `bfcl generate` keeps working when anchoropt is absent. Without the harness the adapter
degrades silently and you get `ImportError: cannot import name 'enabled_explicit_gate_keys'`. That is
why `run.py` puts harness/ on PYTHONPATH too, and why the tests do the same.
"""

from __future__ import annotations

from pathlib import Path as _Path

_vendored = _Path(__file__).resolve().parent.parent / "benchmarks" / "bfcl_v4" / "evaluator"
if _vendored.is_dir() and str(_vendored) not in __path__:
    __path__.append(str(_vendored))

del _Path, _vendored
