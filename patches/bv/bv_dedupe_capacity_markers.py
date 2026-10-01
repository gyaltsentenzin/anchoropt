#!/usr/bin/env python3
"""Collapse the THREE identical `_CAPACITY_MARKERS`/`_capacity_seen` definitions into one.

ISOLATED COPIES ONLY. Do not run this against the shared tree: it is cosmetic, the three copies are
byte-identical so the last definition wins and behaviour is unaffected, and the shared evaluator is
imported by running GEPA workers.

WHY THE DUPLICATES EXIST -- the bug worth keeping in view
--------------------------------------------------------
An earlier patch guarded idempotence on the ANCHOR it inserted before rather than on a MARKER it
inserted itself. The anchor survives the insert, so every re-run inserted the block again. Three runs,
three copies. `patches/bv/README_error_kind.md` records the same lesson; this script is the cleanup and
is itself guarded on a marker, so it cannot repeat the mistake it fixes.

WHAT IT DOES
------------
Keeps the FIRST definition, deletes every later byte-identical duplicate. Refuses to touch anything if
the copies are not byte-identical -- divergent copies are a semantic question, not a formatting one,
and silently keeping the first would change behaviour.
"""

from __future__ import annotations

import pathlib
import re
import sys

MARKER = "# [anchoropt-patch:dedupe_capacity_markers]"

BLOCK_RE = re.compile(
    r'\n*_CAPACITY_MARKERS = \("is full", "exceeds maximum size", "at its entry limit", "no capacity"\)\n\n\n'
    r'def _capacity_seen\(results\) -> bool:\n'
    r'    """Did any tool result report a store at its limit\? The official substrings, not a guess\."""\n'
    r'    for r in \(results or \[\]\):\n'
    r'        low = str\(r\)\.lower\(\)\n'
    r'        for m in _CAPACITY_MARKERS:\n'
    r'            if m in low:\n'
    r'                return True\n'
    r'    return False\n')


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_dedupe_capacity_markers.py <path to memory_evaluator.py>")
        return 2
    p = pathlib.Path(sys.argv[1])
    if "/anchoropt-wei/anchoropt/anchoropt/" in str(p.resolve()):
        print("REFUSING: that is the SHARED tree, imported by running GEPA workers. "
              "Run this only against an isolated copy.")
        return 3
    src = p.read_text()

    if MARKER in src:
        print(f"already applied ({MARKER} present) -- no change")
        return 0

    hits = list(BLOCK_RE.finditer(src))
    print(f"found {len(hits)} definition block(s)")
    if len(hits) <= 1:
        print("nothing to collapse")
        return 0

    # Compare the CODE, not the surrounding blank lines. The regex swallows leading newlines, so the
    # first block carries two more than its successors -- a formatting artifact of where the duplicate
    # was spliced in, not a semantic difference. Stripping is correct here and ONLY here: the guard
    # exists to catch blocks whose BODIES diverge, which is a semantic question.
    bodies = {src[h.start():h.end()].strip() for h in hits}
    if len(bodies) != 1:
        print(f"REFUSING: the {len(hits)} blocks differ in CODE ({len(bodies)} distinct bodies). "
              "Divergent copies are a semantic question -- keeping the first would change behaviour.")
        return 1
    print(f"all {len(hits)} bodies are identical code (differing only in leading blank lines)")

    # Delete every block after the first, last-to-first so earlier offsets stay valid.
    out = src
    for h in reversed(hits[1:]):
        out = out[:h.start()] + "\n" + out[h.end():]
    # Mark the survivor.
    first = BLOCK_RE.search(out)
    out = out[:first.start()] + "\n\n" + MARKER + out[first.start():first.end()] + out[first.end():]

    bak = p.with_suffix(p.suffix + ".pre_dedupe")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    p.write_text(out)
    print(f"collapsed {len(hits)} -> 1 in {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
