"""Make the BV runtime able to install a controller STACK, so the incumbent can move.

WHAT THIS FIXES
---------------
`install_controller.py` read one env var (`ANCHOROPT_CONTROLLER_SPEC`) and installed one predicate, so
every BV arm was structurally `H0 + exactly one controller`. The moving incumbent was therefore
inexpressible from the runner, and each round re-measured against bare H0. Measured consequence:
`rounds/AUTORUN/R1_RESULT.json` records `controllers_installed: 0` and `incumbent/correct: 18` on kv --
18 being exactly the CONTROL score of an already-accepted +4 controller.

`anchoropt/runtime_hook.py` needed no change: `_INSTALLED` is already `dict[locus, list]`, `install()`
appends, and `decide()` iterates every controller at a locus with first-fire-wins. Only the LOADER was
single-controller.

WHY THIS PATCHES THE ISOLATED TREE AND NOT THE SHARED ONE
---------------------------------------------------------
`install_controller.py` lives in the SHARED tree (`/path/to/remote-checkout/scripts/`), which
running GEPA workers import and which must stay at 0 patch markers. The runner's PYTHONPATH is
`$W:$R:$R/scripts` with `$W=/path/to/isolated-workspace`, so a copy at `$W/install_controller.py` wins by import
order without the shared file being touched. Verified: `importlib.util.find_spec` resolves to the
isolated copy, and the shared file's md5 is unchanged at 9540fad90ef22b2a1818a1d80df68fcb.

THE LINEAGE TRAP THIS AVOIDS
----------------------------
The BV shared copy had DIVERGED from the local tree: it imports `evaluate_signal` from
`anchoropt.bfcl_declared_signals`, while the local copy imports it from `anchoropt.memory_gates`.
Copying the local file over would have silently broken declared-signal evaluation for every arm whose
predicate names a host signal -- which is all four round-1 suppression arms. So the isolated copy is
built by grafting the new STACK tail onto **BV's own head**, and the graft asserts that BV's import
survived and the local one did not leak in. Structural patch, never an rsync -- the same rule the
sidecar lineages already follow.

ORDER IS DATA
-------------
`decide()` is first-fire-wins, so at a shared locus the list order is part of the arm. It is read
verbatim from the stack file and never sorted. `STACK` and `SPEC` COMPOSE, candidate LAST, so an
incumbent controller at the same locus keeps precedence and a candidate that only fires where the
incumbent does not stays measurable rather than shadowed.

RUNNER CHANGES (runner_auto.sh, backed up as .bak_prestack)
-----------------------------------------------------------
  * `STACK=<file>` exports `ANCHOROPT_CONTROLLER_STACK` and sets `ANCHOROPT_PHI_MODULE`, which alone
    is what makes a spec non-inert.
  * STACK with no SPEC is the INCUMBENT CONTROL -- the arm the next candidate must beat. Previously
    "no SPEC" could only mean bare H0.
  * The installed spec and stack are copied next to the results as `controller_spec.json` /
    `controller_stack.json`, because scoring must read what RAN rather than what a caller believed was
    installed (`_installed_spec` looks for exactly that filename).
  * SPEC-only behaviour is byte-identical to before.

Usage (idempotent; safe to re-run):
    python patches/bv/bv_controller_stack.py --check      # report only
    python patches/bv/bv_controller_stack.py --emit       # write the isolated loader to stdout
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

#: The shared file that must NOT change. Asserted by hash, not by intention.
SHARED_LOADER = "/path/to/remote-checkout/scripts/install_controller.py"
SHARED_MD5_EXPECTED = "9540fad90ef22b2a1818a1d80df68fcb"

#: Where the isolated (winning) copy goes.
ISOLATED_LOADER = "/path/to/isolated-workspace/install_controller.py"

#: The import that must survive the graft, and the one that must not appear.
MUST_KEEP = "anchoropt.bfcl_declared_signals import evaluate_signal"
MUST_NOT_LEAK = "memory_gates import evaluate_signal"

GRAFT_MARKER_LOCAL = 'def _install_one(spec, *, source=""):'
GRAFT_MARKER_BV = '_PATH = os.environ.get("ANCHOROPT_CONTROLLER_SPEC")'


def graft(bv_source: str, local_source: str) -> str:
    """BV's head + the local STACK tail. Refuses rather than producing a half-correct loader."""
    if GRAFT_MARKER_LOCAL not in local_source:
        raise SystemExit("local loader has no _install_one: it is not stack-capable")
    if GRAFT_MARKER_BV not in bv_source:
        raise SystemExit("BV loader has no _PATH block: unexpected shape, refusing to graft")
    out = bv_source[:bv_source.index(GRAFT_MARKER_BV)] + \
        local_source[local_source.index(GRAFT_MARKER_LOCAL):]
    if MUST_KEEP not in out:
        raise SystemExit(f"REFUSING: the graft lost BV's declared-signal import ({MUST_KEEP}). "
                         f"Every arm whose predicate names a host signal would silently stop "
                         f"evaluating it.")
    if MUST_NOT_LEAK in out:
        raise SystemExit(f"REFUSING: the local import leaked in ({MUST_NOT_LEAK}); BV resolves "
                         f"declared signals from a different module.")
    if "ANCHOROPT_CONTROLLER_STACK" not in out:
        raise SystemExit("REFUSING: the graft produced no stack support")
    compile(out, ISOLATED_LOADER, "exec")          # syntax, before it ever reaches the cluster
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bv", default="/tmp/ic_bv.py",
                    help="a local copy of BV's shared loader (scp it first)")
    ap.add_argument("--local", default=str(pathlib.Path(__file__).resolve().parents[2]
                                          / "scripts" / "install_controller.py"))
    ap.add_argument("--emit", action="store_true", help="print the grafted loader")
    ap.add_argument("--check", action="store_true", help="report the hashes and exit")
    a = ap.parse_args()

    bvp, lp = pathlib.Path(a.bv), pathlib.Path(a.local)
    if a.check:
        for p in (bvp, lp):
            print(f"{p}: {'MISSING' if not p.exists() else hashlib.md5(p.read_bytes()).hexdigest()}")
        print(f"shared expected md5: {SHARED_MD5_EXPECTED}  (must be unchanged on BV)")
        return 0
    if not bvp.exists():
        raise SystemExit(f"need BV's loader at {bvp}: "
                         f"scp <remote>:{SHARED_LOADER} {bvp}")
    got = hashlib.md5(bvp.read_bytes()).hexdigest()
    if got != SHARED_MD5_EXPECTED:
        print(f"WARNING: BV shared loader md5 is {got}, expected {SHARED_MD5_EXPECTED}. "
              f"It changed upstream -- re-read the diff before grafting.", file=sys.stderr)
    out = graft(bvp.read_text(), lp.read_text())
    if a.emit:
        sys.stdout.write(out)
    else:
        print(f"graft OK ({out.count(chr(10))} lines); declared-signal import preserved. "
              f"Use --emit and scp to {ISOLATED_LOADER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
