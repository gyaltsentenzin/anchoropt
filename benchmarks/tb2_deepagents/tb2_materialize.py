"""Materialize an AnchorOpt controller as a TB2 evaluation surface.

The runner under test loads `repo_baseline.py` from a workspace directory. So installing a
controller means producing a candidate workspace whose `repo_baseline.py` is the incumbent's,
**plus one appended block** that builds the AnchorOpt middleware from a persisted ControllerSpec.

WHY A SINGLE APPENDED BLOCK, AND NOTHING ELSE
---------------------------------------------
The paired comparison is only interpretable if control and candidate differ by the controller and
nothing else. This function therefore never edits an existing line: the five prompt middlewares
inherited from the baseline this adapter was ported from, every compat fix, and the wrapper are
byte-identical between arms. `diff_summary()` reports added/removed counts so the claim is
checkable rather than asserted -- removed must be 0.

This is NOT a patch generator. The appended block is a fixed template; the only thing that varies
between candidates is the JSON spec it loads. An LLM never writes code here.
"""

from __future__ import annotations

import difflib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from tb2_middleware import ControllerSpec

# The one thing appended to a baseline surface. Fixed text -- only $ANCHOROPT_CONTROLLER varies.
_BLOCK = '''
    # --- AnchorOpt controller (anchoropt.controller.v1) -----------------------------
    # THE ONLY control-vs-candidate difference. The prompt middleware builders above are
    # untouched; this appends one middleware built by the TB2 adapter from
    # $ANCHOROPT_CONTROLLER, and writes per-episode telemetry to $ANCHOROPT_TELEMETRY so
    # firing/execution counts come from the anchor's own decision point.
    import os as _ao_os, sys as _ao_sys, json as _ao_json, atexit as _ao_atexit
    _ao_spec = _ao_os.environ.get("ANCHOROPT_CONTROLLER", "")
    if _ao_spec:
        for _p in (_ao_os.environ.get("ANCHOROPT_REPO", ""),
                   _ao_os.environ.get("ANCHOROPT_ADAPTER", "")):
            if _p and _p not in _ao_sys.path:
                _ao_sys.path.insert(0, _p)
        from tb2_middleware import ControllerSpec as _AOSpec, build_middleware as _ao_build
        _ao_mw = _ao_build(_AOSpec.load(_ao_spec))
        middleware.append(_ao_mw)

        def _ao_dump(mw=_ao_mw):
            out = _ao_os.environ.get("ANCHOROPT_TELEMETRY", "")
            if not out:
                return
            rec = dict(mw.telemetry)
            rec["controller_id"] = mw.spec.controller_id
            rec["task"] = _ao_os.environ.get("ANCHOROPT_TASK", "")
            try:
                with open(out, "a") as fh:
                    fh.write(_ao_json.dumps(rec) + "\\n")
            except Exception:
                pass
        _ao_atexit.register(_ao_dump)
    # -------------------------------------------------------------------------------

'''

_HOOK = '''    if middleware:
        kwargs["middleware"] = middleware
'''


@dataclass(frozen=True)
class MaterializedController:
    workspace: Path
    spec_path: Path
    added_lines: int
    removed_lines: int

    @property
    def is_clean(self) -> bool:
        """A candidate surface must ADD only. Any removal means the arms differ elsewhere."""
        return self.removed_lines == 0


def materialize(spec: ControllerSpec, *, incumbent_workspace: Path, out_dir: Path
                ) -> MaterializedController:
    """Build a candidate workspace installing `spec`, from `incumbent_workspace`.

    Writes `<out_dir>/repo_baseline.py`, copies the wrapper package unchanged, persists the spec,
    and writes the `manifest.json` the branch-promotion machinery this was ported from reads on
    promotion.
    """
    incumbent_workspace = Path(incumbent_workspace)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_src = (incumbent_workspace / "repo_baseline.py").read_text()
    if _HOOK not in base_src:
        raise ValueError(
            f"{incumbent_workspace}/repo_baseline.py has no middleware assembly block to append "
            f"after; cannot install a controller without editing existing lines")
    if base_src.count(_HOOK) != 1:
        raise ValueError("middleware assembly block is ambiguous (found more than once)")

    cand_src = base_src.replace(_HOOK, _BLOCK + _HOOK)
    (out_dir / "repo_baseline.py").write_text(cand_src)

    # the wrapper package must be IDENTICAL between arms -- copy, never regenerate
    wrapper = incumbent_workspace / "self_harness_harbor"
    if wrapper.is_dir():
        shutil.copytree(wrapper, out_dir / "self_harness_harbor", dirs_exist_ok=True)
        for junk in (out_dir / "self_harness_harbor").rglob("__pycache__"):
            shutil.rmtree(junk, ignore_errors=True)

    spec_path = spec.write(out_dir / "controller.json")
    (out_dir / "manifest.json").write_text(json.dumps(
        {"surface_files": {"baseline": "repo_baseline.py"},
         "controller_id": spec.controller_id}, indent=2) + "\n")

    added = removed = 0
    for line in difflib.unified_diff(base_src.splitlines(), cand_src.splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return MaterializedController(workspace=out_dir, spec_path=spec_path,
                                  added_lines=added, removed_lines=removed)


def diff_summary(incumbent_workspace: Path, candidate_workspace: Path) -> dict:
    """Added/removed line counts between two surfaces, for the paired-arm sanity claim."""
    a = (Path(incumbent_workspace) / "repo_baseline.py").read_text().splitlines()
    b = (Path(candidate_workspace) / "repo_baseline.py").read_text().splitlines()
    added = removed = 0
    for line in difflib.unified_diff(a, b, lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"added": added, "removed": removed, "clean": removed == 0}
