#!/usr/bin/env python3
"""Declare `relocate_addressing` in the LIVE sidecar allowlist.

WHY THIS IS ITS OWN PATCH, applied BEFORE the R12 GPU round
-----------------------------------------------------------
`traj_sidecar` keeps a HAND-AUTHORED literal allowlist, and its own comments record that the list has
silently dropped intervention telemetry at least seven times -- `constraint_state`, `capacity_repair_*`
(twice), `futility_kind`, every `e1_*` key, the relocation family, and a tuple-vs-string concat. The
stated reason it keeps recurring: **the list is authored by hand while the fields are added
elsewhere.** The standing rule is therefore that a new step_record key is declared in the SAME COMMIT
as the field that sets it.

`relocate_addressing` is the key that says WHICH declared pair executed -- `keyed` (kv) or `ordinal`
(vector). For R12 that is not decoration: it is the only per-step evidence distinguishing

    "A1 relocated on vector"   from   "A1 relocated on kv and vector was untouched again"

and without it a round measuring exactly that distinction would have to infer it from the cell name.
`relocate_gate` survives on the `*_gate` suffix rule; this one matches neither that nor any prefix.

IDEMPOTENCE
-----------
Guarded on the MARKER this patch inserts, never on the anchor.

Run from a private directory, NOT /tmp: `/tmp/inspect.py` on this cluster shadows stdlib `inspect`.
"""

from __future__ import annotations

import ast
import pathlib
import sys

MARKER = "# [anchoropt-patch:relocate_addressing_key]"

ANCHOR = '''    "relocate_controller", "relocate_write_skipped", "relocate_no_failing_call",
'''

BLOCK = '''    "relocate_controller", "relocate_write_skipped", "relocate_no_failing_call",
    ''' + MARKER + '''
    # WHICH declared pair executed: "keyed" (kv, plain dict stores) or "ordinal" (vector, VectorStore
    # objects addressed by auto-assigned vec_id). Declared on the same commit as the field, per the
    # rule in the note above. It is the only per-step evidence separating "A1 reached vector" from
    # "A1 relocated on kv again", which is precisely what R12 measures.
    "relocate_addressing",
    # And which addressings the process was ALLOWED to use -- so a control's own trajectory says why
    # it declined, rather than the reader having to trust the launch command.
    "relocate_enabled_addressings",
    # PRE-EXISTING OMISSION, found while auditing R12's victim selection and fixed here rather than
    # left for a ninth recurrence. `relocate_victim_already_in_destination` is written by the
    # primitive on EVERY firing (always a real bool, never None) and was dropped by this allowlist,
    # so an audit reading it back saw None for all 24 relocations and could not tell "rule 1 did not
    # apply" from "the key was never recorded". It is the flag for pick_victim's costless case -- the
    # victim already has a copy in the destination, so the write is skipped -- and without it the
    # skipped-write branch cannot be distinguished from a write that landed.
    "relocate_victim_already_in_destination",
'''


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_relocate_addressing_key.py <path to anchoropt/traj_sidecar.py>")
        return 2
    p = pathlib.Path(sys.argv[1])
    src = p.read_text()

    if MARKER in src:
        print(f"already applied ({MARKER} present) -- no change")
        return 0
    if ANCHOR not in src:
        print("FAIL: relocation-family anchor not found -- is bv_relocate_telemetry applied?")
        return 1

    out = src.replace(ANCHOR, BLOCK, 1)
    try:
        ast.parse(out)
    except SyntaxError as exc:
        print(f"FAIL: patched file does not parse ({exc}); NOT written")
        return 1

    bak = p.with_suffix(p.suffix + ".pre_reloc_addressing")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    p.write_text(out)
    print(f"patched {p}")
    print(f"  marker inserted: {MARKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
