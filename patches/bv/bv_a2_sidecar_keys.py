#!/usr/bin/env python3
"""Declare the A2 seam's step_record keys in the LIVE sidecar allowlist.

Same-commit rule, per `traj_sidecar`'s own note: that allowlist is authored by hand while the fields are
added elsewhere, and it has silently dropped intervention telemetry at least nine times
(constraint_state, capacity_repair_* twice, futility_kind, every e1_* key, the relocation family, a
tuple-vs-string concat, low_similarity_*, and relocate_victim_already_in_destination found in R12).

`a2_gate` survives on the `*_gate` suffix rule. NOTHING ELSE here would, and without them:

  * `a2_declined` missing  -> a refusing branch is silent, so "no controller fired" cannot be told from
    "the gate never ran" (anchoropt-unmeasurable-guard-paths)
  * `a2_state_*` missing   -> the round cannot show the predicate read the state it claims to read,
    which is criterion 3's whole content
  * `a2_controller` missing -> an arm is identified by position rather than name
    (anchoropt-identify-by-name-not-position)

IDEMPOTENCE. Guarded on the MARKER this patch inserts, never on the anchor.
Run from a private directory, NOT /tmp.
"""

from __future__ import annotations

import ast
import pathlib
import sys

MARKER = "# [anchoropt-patch:a2_sidecar_keys]"

ANCHOR = '''    "relocate_victim_already_in_destination",
'''

BLOCK = ANCHOR + '''    ''' + MARKER + '''
    # A2 answer-boundary seam. `a2_gate` rides the *_gate suffix rule; these do not.
    "a2_controller", "a2_injected_chars", "a2_declined", "a2_response_before",
    # THE STATE THE PREDICATE READ, recorded whether or not it fired -- this is what makes
    # "grounded in retrieval state" checkable rather than asserted, and what lets a non-firing
    # case be distinguished from an unobserved one.
    "a2_state_best_similarity", "a2_state_attempts", "a2_state_other_entries",
    "a2_state_searched_other",
    "answer_phase_error_gate", "a2_error_gate",
    # Which controllers this seam SKIPPED because it does not supply their signal. Without it, an
    # exclusion is silent -- and the defect that made this key necessary was the opposite case: the
    # seam dispatching to a neighbour whose signal it does not supply.
    "a2_skipped_other_signal",
'''


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_a2_sidecar_keys.py <path to anchoropt/traj_sidecar.py>")
        return 2
    p = pathlib.Path(sys.argv[1])
    src = p.read_text()
    if MARKER in src:
        print(f"already applied ({MARKER} present) -- no change")
        return 0
    if ANCHOR not in src:
        print("FAIL: anchor not found -- is bv_relocate_addressing_key applied?")
        return 1
    out = src.replace(ANCHOR, BLOCK, 1)
    try:
        ast.parse(out)
    except SyntaxError as exc:
        print(f"FAIL: patched file does not parse ({exc}); NOT written")
        return 1
    bak = p.with_suffix(p.suffix + ".pre_a2_keys")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    p.write_text(out)
    print(f"patched {p}\n  marker: {MARKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
