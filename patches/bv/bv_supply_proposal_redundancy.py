#!/usr/bin/env python3
"""Supply `proposal_is_redundant` at the commitment gate, from the LIVE store.

ISOLATED COPIES ONLY -- refuses the shared tree.

The declared signal `redundant_proposed_write` reads one field the boundary does not yet carry. This
supplies it by calling the host's OWN generic comparator -- `redundant_write.should_suppress` -- against
`involved_instances`, which is already in scope at that gate (the anchor path reads the live store from
exactly there).

WHY THE COMPARATOR AND NOT A RULE OF MY OWN
-------------------------------------------
`should_suppress` is duck-typed over container names, takes the key/value field names as PARAMETERS,
refuses truncated telemetry outright, and returns False WITH A REASON whenever the answer is not
certain -- absent key, missing value, or any difference at all. So an UPDATE (same key, different
value) is never reported redundant, and that safety property belongs to the comparator rather than to a
threshold I picked.

NO ANCHOR POLICY IS READ. No `enable_*` flag, no threshold, no priority, no anchor identity. This adds
an OBSERVATION; whether to act on it is the installed predicate's decision and the action is whatever
core grounds.

FIELD NAMES ARE TRIED, NOT ASSUMED. kv writes carry `key=`/`value=`; vector and rec_sum write by `text=`
with no key at all, where the comparator correctly finds nothing to compare and answers False. That is
the honest result for those backends -- `identifier_present` is not merely unsupplied there, it is
undefined.
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "# [anchoropt-patch:supply_proposal_redundancy]"

ANCHOR = '''                        try:
                            _ust.update(_upstream_call_facts(str(_uc_call)))
'''

BLOCK = '''                        ''' + MARKER + '''
                        # IS THIS PROPOSED WRITE ALREADY IN THE STORE? Asked of the LIVE store, using
                        # the host's own generic comparator -- not a rule written here.
                        #
                        # `should_suppress` returns False with a reason whenever the answer is not
                        # certain (absent key, missing value, ANY difference), so a genuine UPDATE is
                        # never reported redundant. The safety property is the comparator's, not a
                        # threshold's. Always present, defaulting False: a predicate over an absent
                        # field silently answers False, which is the defect class this avoids.
                        _ust["proposal_is_redundant"] = False
                        try:
                            import sys as _rsys
                            _rdir = str(Path(__file__).resolve().parent.parent / "scripts")
                            if _rdir not in _rsys.path:
                                _rsys.path.insert(0, _rdir)
                            import redundant_write as _rwg
                            # THE DESTINATION STORE THE CALL NAMES, not the first arbitrary entry.
                            #
                            # `list(involved_instances.values())[0]` is whatever dict order yields. A
                            # `core_memory_add` compared against an ARCHIVAL instance asks the wrong
                            # store: it would miss a real duplicate (absent there -> "not a
                            # duplicate") and could call a genuinely new write redundant if the other
                            # store happens to hold that key. Resolve by the call's own container and
                            # fall back only when the backend names none.
                            _rdest = None
                            # Local pattern: `_CONTAINER_RE` is NOT defined in this module, and
                            # referencing it would raise into the handler below and leave the field
                            # silently False -- the failure mode this whole line of work keeps hitting.
                            try:
                                import re as _rre
                                _rcname = _rre.match(r"\s*(core|archival)_memory_", str(_uc_call))
                                _rwant = _rcname.group(1) if _rcname else None
                            except Exception:
                                _rwant = None
                            if _rwant:
                                for _rk, _rv in (involved_instances or {}).items():
                                    if _rwant.lower() in str(_rk).lower():
                                        _rdest = _rv
                                        break
                                if _rdest is None:
                                    for _rv in (involved_instances or {}).values():
                                        if isinstance(getattr(_rv, _rwant + "_memory", None), dict):
                                            _rdest = _rv
                                            break
                            # A backend whose calls name no container (text-only) has one store; the
                            # comparator then finds no key to compare and answers False anyway.
                            _rinst = _rdest if _rdest is not None else (
                                list(involved_instances.values())[0] if involved_instances else None)
                            step_record["redundancy_store_gate"] = str(_rwant or "unnamed")
                            if _rinst is not None:
                                _rargs = _parse_call_args(str(_uc_call))
                                # kv keys on (key, value); the text-only backends have no identifier,
                                # where the comparator finds nothing to compare and answers False.
                                _rok, _rwhy = _rwg.should_suppress(_rargs, _rinst)
                                _ust["proposal_is_redundant"] = bool(_rok)
                                step_record["redundancy_reason_gate"] = str(_rwhy)[:120]
                        except Exception as _rex:
                            step_record["redundancy_error_gate"] = "%s: %s" % (
                                type(_rex).__name__, _rex)
                        try:
                            _ust.update(_upstream_call_facts(str(_uc_call)))
'''


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_supply_proposal_redundancy.py <path to memory_evaluator.py>")
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
    bak = p.with_suffix(p.suffix + ".pre_redundancy")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    p.write_text(src.replace(ANCHOR, BLOCK, 1))
    print(f"patched {p}\n  supplies: proposal_is_redundant at the commitment gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
