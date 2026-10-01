#!/usr/bin/env python3
"""Declare the CANONICAL ACCEPTANCE EVIDENCE keys in the trajectory sidecar allowlist.

ISOLATED COPIES ONLY -- refuses the shared tree.

WHY THIS PATCH EXISTS, and why it must land BEFORE any GPU round.

`scripts/self_evolve_cycle2.py --score` now adjudicates all four acceptance criteria instead of
`net > 0`. Criteria 3 and 4 are checked over a fixed vocabulary:

    mechanism_requested  mechanism_verified  mechanism_declined_with_reason
    mechanism_refused_by_guard
    clears_added  information_losing_removes  verified_relocations

`traj_sidecar._STEP_FIELDS` is an allowlist with one extra rule -- any truthy key ending `_gate` also
survives -- and everything else is DROPPED SILENTLY. Measured on the live isolated runtime before this
patch: ZERO occurrences of any canonical key. So a controller could execute correctly, the adapter
could translate its evidence faithfully, and criteria 3 and 4 would still read PENDING, which BLOCKS
installation. A correct arm would be indistinguishable from one that never ran.

This is the EIGHTH occurrence of this defect class in this project (constraint_state, capacity_repair_*
twice, transform_ok, the reroute payload, low_similarity_*, futility_/e1_*, relocate_*), and the file's
own notes state the reason plainly: the list is authored by hand while the fields are added elsewhere.

Hence a PREFIX for the mechanism counters rather than a hand-listed set: a counter added to
`mechanism_evidence.FAMILY_READINGS` cannot then be lost by forgetting this file. The three safety
counters do not share a prefix and renaming them would break the core contract they satisfy, so they
are named -- and they are the only three, because criterion 4's contract is closed.
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "# [anchoropt-patch:acceptance_telemetry]"

#: Inserted at the END of the _FAMILY_PREFIXES tuple, located STRUCTURALLY rather than by anchoring on
#: a neighbouring prefix name. The first version of this patch anchored on `"recovery_txn",\n)` and
#: refused on the live isolated runtime -- correctly, because that tree is a DIFFERENT LINEAGE whose
#: tuple continues for another nine prefixes (absent_target, repeat_guard, memoized_call, dedup_clear,
#: xcm_, a9u_ ...) and records this defect class as having recurred TWELVE times, not seven. That is
#: the standing lesson that BV's live evaluator is not the vendored one, so a patch must not assume
#: the local file's shape. Refusing beat guessing; locating the tuple beats both.
BLOCK_LINES = '''    ''' + MARKER + '''
    # CANONICAL ACCEPTANCE EVIDENCE for criteria 3 and 4. A PREFIX, not a hand-listed set: this
    # allowlist has silently dropped intervention telemetry many times before, always because the
    # list is maintained by hand while the fields are added elsewhere.
    "mechanism_",
    # The three safety counters, named because they share no prefix and the contract is closed.
    "clears_added", "information_losing_removes", "verified_relocations",
'''


#: The keys criteria 3 and 4 are adjudicated over. `mechanism_` is a prefix, so any member of
#: `mechanism_evidence`'s canonical vocabulary is admitted by it.
REQUIRED = ("mechanism_", "clears_added", "information_losing_removes", "verified_relocations")


def _tuple_bounds(src: str) -> tuple[int, int] | None:
    """(first line INSIDE the tuple, line holding its closing paren). None if not locatable."""
    lines = src.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if l.startswith("_FAMILY_PREFIXES = (")), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(")")), None)
    return None if end is None else (start + 1, end)


def already_admits_everything(src: str) -> bool:
    """True when the tuple ALREADY declares every required key, marker or no marker.

    This is the semantic idempotency check, and it is separate from the marker check on purpose.
    The marker only knows whether THIS patch ran. A tree can satisfy the requirement another way --
    the source-side fix committed upstream declares exactly these keys directly in the vendored
    file -- and on such a tree a marker-only guard appends a second, redundant copy. Harmless at
    runtime, but a patch that edits an already-correct file is a patch whose report cannot be
    trusted, so it reports SATISFIED and writes nothing.
    """
    b = _tuple_bounds(src)
    if b is None:
        return False
    body = src.splitlines(keepends=True)[b[0]:b[1]]
    literals = "".join(l for l in body if not l.lstrip().startswith("#"))
    return all(f'"{k}"' in literals for k in REQUIRED)


def _insert_at_tuple_end(src: str) -> str | None:
    """Add BLOCK_LINES just before the `)` that closes _FAMILY_PREFIXES. None if not locatable."""
    b = _tuple_bounds(src)
    if b is None:
        return None
    lines = src.splitlines(keepends=True)
    return "".join(lines[:b[1]]) + BLOCK_LINES + "".join(lines[b[1]:])


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_acceptance_telemetry.py <path to traj_sidecar.py>")
        return 2
    p = pathlib.Path(sys.argv[1])
    if "/anchoropt-wei/anchoropt/anchoropt/" in str(p.resolve()):
        print("REFUSING: that is the SHARED tree, imported by running GEPA workers.")
        return 3
    src = p.read_text()
    if MARKER in src:
        print(f"already applied ({MARKER} present) -- no change")
        return 0
    if already_admits_everything(src):
        print("SATISFIED: the tuple already declares every required key -- no change")
        print("  (this tree carries the source-side fix; patching would duplicate it)")
        return 0
    out = _insert_at_tuple_end(src)
    if out is None:
        print("FAIL: could not locate the _FAMILY_PREFIXES tuple; refusing to guess")
        return 1
    bak = p.with_suffix(p.suffix + ".pre_accepttelem")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    p.write_text(out)
    print(f"patched {p}\n  criterion 3/4 evidence will now reach the sidecar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
