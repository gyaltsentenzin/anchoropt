#!/usr/bin/env python3
"""Declare the RELOCATION evidence keys in the trajectory sidecar allowlist.

ISOLATED COPIES ONLY -- refuses the shared tree.

THE DEFECT THIS PREVENTS, for the fourth time. `traj_sidecar._STEP_FIELDS` is an allowlist with one
extra rule: any truthy key ending `_gate` also survives. Everything else is DROPPED SILENTLY. That has
already cost three separate audits -- `upstream_controller`, `upstream_withheld_calls` and the
`capacity_repair_*` preservation keys -- each time making a controller that ran correctly look like it
had left no evidence, which is indistinguishable from not having run.

`relocate_gate` would survive on the suffix rule. The keys that PROVE the recovery was safe would not:

    relocate_victim_key                      which entry was moved
    relocate_write_verified                  the destination write was confirmed in live state
    relocate_removed_verified                the source removal was confirmed in live state
    relocate_copy_survives_in_destination    THE INVARIANT: the fact still exists
    relocate_slot_freed                      capacity actually changed
    relocate_retry_landed                    the originally refused write succeeded
    relocate_declined                        why a firing did NOT act
    relocate_source_entries_before/after     the occupancy delta
    relocate_victim_chars                    how much content moved
    relocate_retried_key                     what the original call was storing

A recovery is verified on what it PRESERVED, not on whether it fired, so those are the load-bearing
fields. `relocate_declined` matters just as much: a declined repair with no reason recorded is the
silent no-op this project keeps rediscovering.
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "# [anchoropt-patch:relocate_telemetry]"

ANCHOR = '''    "upstream_controller", "upstream_withheld_calls",
'''

BLOCK = '''    "upstream_controller", "upstream_withheld_calls",
    ''' + MARKER + '''
    # RELOCATION EVIDENCE. `relocate_gate` survives on the *_gate suffix rule; none of these would,
    # and they are the ones that prove the recovery preserved the information rather than merely ran.
    # Fourth occurrence of this allowlist silently dropping intervention telemetry -- see the note
    # above for the previous three.
    "relocate_victim_key", "relocate_victim_chars", "relocate_write_verified",
    "relocate_removed_verified", "relocate_copy_survives_in_destination",
    "relocate_slot_freed", "relocate_retry_landed", "relocate_retried_key",
    "relocate_source_entries_before", "relocate_source_entries_after",
    "relocate_destination_entries_before", "relocate_declined",
    "relocate_write_result", "relocate_remove_result", "relocate_retry_result",
    "relocate_invariant_violated", "relocate_preserved_source_intact",
    "relocate_controller", "relocate_write_skipped", "relocate_no_failing_call",
'''


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_relocate_telemetry.py <path to traj_sidecar.py>")
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
    bak = p.with_suffix(p.suffix + ".pre_reloctelem")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    p.write_text(src.replace(ANCHOR, BLOCK, 1))
    print(f"patched {p}\n  relocation preservation evidence will now reach the sidecar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
