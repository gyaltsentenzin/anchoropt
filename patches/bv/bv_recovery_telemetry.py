#!/usr/bin/env python3
"""Let the sidecar keep the per-call RECOVERY telemetry, so preservation can be verified.

ISOLATED COPIES ONLY -- refuses the shared tree.

THE GAP. A recovery intervention must be checked differently from a suppression: the question is not
"did a call get withheld" but "did the information actually reach the store". The evaluator already
records that -- `capacity_repair_replaced_call`, `capacity_repair_substituted_call`,
`capacity_repair_shed_chars` -- and the sidecar DROPS all three: they match neither `_STEP_FIELDS` nor
the `*_gate` suffix rule. Only `capacity_repair_gate` survives, which says a repair fired and nothing
about what it preserved.

This is the same allowlist trap that has now cost three separate checks in this line of work
(`upstream_controller`, `upstream_withheld_calls`, and these). The sidecar's own comment predicts it:
"a gate could act correctly and leave no evidence -- the cause was always this list."

WHAT IT ADDS. The three recovery fields, plus the withheld-call list and the controller name whose
absence blocked two earlier audits. Nothing else: this is an allowlist, and over-declaring would
re-admit the unbounded-dump problem the list exists to prevent.
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "# [anchoropt-patch:recovery_telemetry]"

ANCHOR = '''    "cell_key", "is_retrieval_success",
'''

BLOCK = '''    "cell_key", "is_retrieval_success",
    ''' + MARKER + '''
    # RECOVERY EVIDENCE. A recovery must be verified on what it PRESERVED, not on whether it fired --
    # so the replaced call, the substituted call and the shed character count have to survive. They
    # match neither the allowlist nor the *_gate rule, so `capacity_repair_gate` was reaching the
    # sidecar alone and saying nothing about preservation.
    "capacity_repair_replaced_call", "capacity_repair_substituted_call",
    "capacity_repair_shed_chars",
    # The two keys whose absence blocked earlier audits: which controller fired, and which calls a
    # suppressor actually withheld. Both were dropped for the same reason.
    "upstream_controller", "upstream_withheld_calls",
'''


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_recovery_telemetry.py <path to traj_sidecar.py>")
        return 2
    p = pathlib.Path(sys.argv[1])
    if "/anchoropt-wei/anchoropt/anchoropt/" in str(p.resolve()):
        print("REFUSING: that is the SHARED tree, imported by running GEPA workers.")
        return 3
    src = p.read_text()
    if MARKER in src:
        print(f"already applied ({MARKER} present) -- no change")
        return 0
    if src.count(ANCHOR) != 1:
        print(f"FAIL: anchor appears {src.count(ANCHOR)} times; refusing to guess")
        return 1
    bak = p.with_suffix(p.suffix + ".pre_recovery")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    p.write_text(src.replace(ANCHOR, BLOCK, 1))
    print(f"patched {p}\n  recovery + suppression evidence now reaches the sidecar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
