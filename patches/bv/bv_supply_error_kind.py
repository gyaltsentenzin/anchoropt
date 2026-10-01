#!/usr/bin/env python3
"""Supply `error_kind` to the LIVE post-execution hook state on the cluster.

THE GAP THIS CLOSES, measured on the cluster itself (2026-09-20)
----------------------------------------------------------------
Six of the nine signals this host declares read `error_kind`:

    container_at_capacity        error_kind == "no_capacity"  and proposes_write
    identifier_not_found         error_kind == "not_found"    and proposes_read
    duplicate_identifier         error_kind == "duplicate_identifier" (or pre-exec form)
    container_slots_exhausted    error_kind == "no_capacity"  and archival/relocation target
    append_would_exceed_cap      error_kind == "blob_would_overflow"
    clear_proposed_at_capacity   container_full OR error_kind == "no_capacity"

`_hook_state` builds the post-execution information set as `dict(step_record)` plus
`best_similarity`, `scored_entries`, `searched_other_container` and `query`. **It never supplies
`error_kind`, and `step_record` never carries it** -- verified by grep on the live evaluator and by
dumping the keys of a real prereq step. So five of those six can only ever answer False on this host,
and the sixth (`clear_proposed_at_capacity`) survives solely on its `container_full` disjunct.

That is the declared-vs-supplied gap the evaluator's own comments describe twice already, one layer
further out: the commitment gate was fixed for `container_full` and for the typed call facts, and the
post-execution locus was left reading a field nobody writes.

WHAT IT DOES
------------
Adds, inside `_hook_state`, a classification of THIS step's own tool results into the same coarse
labels the offline classifier uses. Post-execution is exactly where that is legitimate: the result
exists. Nothing is carried forward and nothing pre-dispatch is touched, so the commitment gate's
information set is unchanged -- the boundary discipline the hook comments insist on is preserved.

`error_kind` is set to None for a clean result rather than omitted, so the key is always PRESENT: a
predicate over an absent field silently answers False, which is the whole defect class.

IDEMPOTENCE
-----------
Guarded on the MARKER this patch inserts, never on the anchor it inserts after. Guarding on the anchor
is why `_CAPACITY_MARKERS` and `_capacity_seen` each appear THREE times in the live evaluator: the
anchor survives the insert, so a second run inserts again. (Those three copies are byte-identical, so
the last one wins and behaviour is unaffected -- but the pattern is the bug.)

Run from a private directory, NOT /tmp: `/tmp/inspect.py` on this cluster shadows stdlib `inspect`.
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "# [anchoropt-patch:supply_error_kind]"

ANCHOR = '''    st = dict(step_record or {})
'''

BLOCK = '''    st = dict(step_record or {})
    ''' + MARKER + '''
    # ERROR KIND -- the field six declared signals read and nothing supplied.
    #
    # POST-EXECUTION ONLY, and that is why it is legitimate here: this step's result exists. The
    # commitment gate builds its own state a few thousand lines below and is untouched, so no
    # post-execution fact reaches a pre-dispatch predicate.
    #
    # ALWAYS PRESENT, None for a clean result. A predicate over an ABSENT field silently answers
    # False, which is the defect being closed -- so the key is written either way.
    try:
        _ek = None
        for _r in (execution_results or []):
            _t = str(_r or "").lower()
            if not any(_m in _t for _m in ("error", "cannot", "failed", "unable", "invalid",
                                           "is full", "exceeds maximum size", "at its entry limit",
                                           "no capacity", "not found", "no such", "too long")):
                continue
            if "unique" in _t:
                _ek = "duplicate_identifier"
            elif "not found" in _t or "no such" in _t:
                _ek = "not_found"
            elif "too long after appending" in _t:
                _ek = "blob_would_overflow"
            elif "entry length" in _t or "entry is too long" in _t:
                _ek = "entry_too_long"
            elif "full" in _t or "exceeds maximum size" in _t:
                _ek = "no_capacity"
            else:
                _ek = "other"
            if _ek:
                break
        st["error_kind"] = _ek
    except Exception:
        st["error_kind"] = None
    # THE OTHER FIELDS THE SAME SIGNALS READ, from this step's own decoded calls.
    # `proposes_write`/`proposes_read` gate container_at_capacity and identifier_not_found; without
    # them an error_kind match is necessary but not sufficient and the signal still cannot fire.
    try:
        _dec = [str(_c) for _c in (decoded or [])]
        st["proposes_write"] = any(
            _k in _c for _c in _dec
            for _k in ("_add(", "_append(", "_update(", "_insert("))
        st["proposes_read"] = any(
            _k in _c for _c in _dec
            for _k in ("_search(", "_retrieve(", "_get(", "_lookup(", "_key_search("))
        st["container"] = ("archival" if any("archival" in _c for _c in _dec)
                           else ("core" if any("core" in _c for _c in _dec) else None))
        for _r in (execution_results or []):
            st.setdefault("result", str(_r))
            break
    except Exception:
        pass
'''


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_supply_error_kind.py <path to anchoropt/memory_evaluator.py>")
        return 2
    p = pathlib.Path(sys.argv[1])
    src = p.read_text()

    if MARKER in src:
        print(f"already applied ({MARKER} present) -- no change")
        return 0

    # The anchor must be the one inside `_hook_state`, which is the FIRST occurrence after its def.
    i = src.find("def _hook_state(")
    if i < 0:
        print("FAIL: no `def _hook_state(` in this file -- wrong target?")
        return 1
    j = src.find(ANCHOR, i)
    if j < 0:
        print("FAIL: anchor not found inside _hook_state")
        return 1

    out = src[:j] + BLOCK + src[j + len(ANCHOR):]
    bak = p.with_suffix(p.suffix + ".pre_error_kind")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    p.write_text(out)
    print(f"patched {p}")
    print(f"  marker inserted: {MARKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
