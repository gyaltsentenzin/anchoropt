#!/usr/bin/env python3
"""Make `constraint_state` see STORE-OBJECT backends, on the LIVE cluster evaluator.

THE GAP THIS CLOSES, measured on the cluster itself (rounds/AUTONOMY/R12)
------------------------------------------------------------------------
`_capture_constraint_state` measures each field the backend adapter declares in `state_needed` by
walking `isinstance(val, str|dict|list|tuple)`. For vector, `state_needed` is
`["core_memory", "archival_memory"]` and BOTH attributes exist -- so `hasattr` PASSES -- but they are
`VectorStore` OBJECTS whose dict lives at `._store`. Every isinstance branch missed, `observed` stayed
empty, and the `if not observed: continue` dropped the record.

Measured: the vector cell emitted **zero** constraint_state over 629 prereq steps while raising 160
real capacity errors. kv emitted 259, rec_sum 274. So every signal reading `constrained_field`,
`schema_limit` or `current_size` was silently blind on one of three cells and presented as "no
opportunity" -- which is how A5's mechanism was nearly written off as structurally unrecoverable.

This is the same defect class as `bv_supply_error_kind` one layer further in: a field the signals read
that nothing supplies, versus a field the capture cannot see. Both report "no opportunity".

WHAT IT DOES
------------
Unwraps `._store` FIRST, by attribute name, mirroring `anchoropt.mechanisms.lossless_eviction
._live_container`, which has handled both shipped shapes correctly all along. Also records the store's
OWN `max_size`/`max_entry_length` when it carries them, and lets that take precedence over a sibling
backend's module constant -- the values agree today, and a schema change to one backend must not be
silently reported against another.

kv and rec_sum are untouched: a plain dict and a plain str have no `._store`, so the added branch is a
no-op for them. Verified against the real backends -- kv's capture is byte-equivalent (`store_caps`
absent), vector's goes from NOTHING to `current_size=7 schema_limit=7 remaining=0`.

IDEMPOTENCE
-----------
Guarded on the MARKER this patch inserts, never on the anchor it inserts at. Guarding on the anchor is
why `_CAPACITY_MARKERS` and `_capacity_seen` each appear THREE times in the live evaluator: the anchor
survives the insert, so a second run inserts again.

Run from a private directory, NOT /tmp: `/tmp/inspect.py` on this cluster shadows stdlib `inspect`.
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "# [anchoropt-patch:vector_store_observability]"
CAPS_MARKER = "# [anchoropt-patch:vector_caps]"

# ---- 1. the type dispatch that dropped every vector record -------------------------------------
ANCHOR = '''                val = getattr(inst, field)
                if isinstance(val, str):
'''

BLOCK = '''                val = getattr(inst, field)
                ''' + MARKER + '''
                # A STORE OBJECT is measured through the dict it holds, not skipped. `hasattr`
                # passes on vector (`core_memory`/`archival_memory` exist) but they are VectorStore
                # OBJECTS whose dict lives at `._store`, so every isinstance branch below missed and
                # the record was dropped -- 0 constraint_state over 629 vector prereq steps against
                # 160 real capacity errors. Unwrapped by attribute name, mirroring
                # anchoropt.mechanisms.lossless_eviction._live_container.
                _store_caps = {}
                _inner = getattr(val, "_store", None)
                if isinstance(_inner, dict):
                    for _cap_attr in ("max_size", "max_entry_length"):
                        _cv = getattr(val, _cap_attr, None)
                        if isinstance(_cv, int):
                            _store_caps[_cap_attr] = _cv
                    val = _inner
                if isinstance(val, str):
'''

# The store's own caps are attached INSIDE the per-field loop, where `field` and `_store_caps` are
# this iteration's. Anchoring on `primary =` instead would place it AFTER the loop, where both names
# hold only the last iteration's values -- correct by accident for a single-field backend and wrong
# for vector, which declares two.
PRIMARY_ANCHOR = '''                elif isinstance(val, (list, tuple)):
                    observed[field] = {"kind": "sequence", "size": len(val)}
'''
PRIMARY_BLOCK = '''                elif isinstance(val, (list, tuple)):
                    observed[field] = {"kind": "sequence", "size": len(val)}
                if field in observed and _store_caps:
                    observed[field]["store_caps"] = _store_caps
'''

# ---- 2. the store's OWN cap wins over a sibling backend's constant ------------------------------
LIMIT_ANCHOR = '''            limit = caps.get("CORE_SIZE") if "core" in (primary or "") else caps.get("ARCH_SIZE")
'''
LIMIT_BLOCK = '''            limit = caps.get("CORE_SIZE") if "core" in (primary or "") else caps.get("ARCH_SIZE")
        ''' + CAPS_MARKER + '''
        # A store that declares its own cap is authoritative for itself. Without this a vector
        # container's limit is read off kv's module constants -- equal today, a silent misreport the
        # moment one backend's schema moves.
        _sc = (obs.get("store_caps") or {})
        if _sc.get("max_size") is not None and obs.get("kind") != "text":
            limit = _sc["max_size"]
'''


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_vector_store_observability.py <path to anchoropt/memory_evaluator.py>")
        return 2
    p = pathlib.Path(sys.argv[1])
    src = p.read_text()

    if MARKER in src:
        print(f"already applied ({MARKER} present) -- no change")
        return 0

    # The anchor must be the one inside `_capture_constraint_state`.
    i = src.find("def _capture_constraint_state(")
    if i < 0:
        print("FAIL: no `def _capture_constraint_state(` -- wrong target?")
        return 1
    j = src.find(ANCHOR, i)
    if j < 0:
        print("FAIL: dispatch anchor not found inside _capture_constraint_state")
        return 1
    out = src[:j] + BLOCK + src[j + len(ANCHOR):]

    k = out.find(PRIMARY_ANCHOR, i)
    if k < 0:
        print("FAIL: `primary =` anchor not found")
        return 1
    out = out[:k] + PRIMARY_BLOCK + out[k + len(PRIMARY_ANCHOR):]

    m = out.find(LIMIT_ANCHOR, i)
    if m < 0:
        print("FAIL: schema-limit anchor not found")
        return 1
    out = out[:m] + LIMIT_BLOCK + out[m + len(LIMIT_ANCHOR):]

    bak = p.with_suffix(p.suffix + ".pre_vector_obs")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    p.write_text(out)
    # Fail loudly if the result does not parse: a patched evaluator that imports is the whole point.
    import ast
    try:
        ast.parse(out)
    except SyntaxError as exc:
        p.write_text(src)
        print(f"FAIL: patched file does not parse ({exc}); REVERTED")
        return 1
    print(f"patched {p}")
    print(f"  markers inserted: {MARKER} , {CAPS_MARKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
