#!/usr/bin/env python3
"""Issue 2: the generic capacity-repair path must check CAPABILITY IDENTITY, not merely "a controller fired".

ISOLATED COPIES ONLY -- refuses the shared tree.

THE DEFECT. My generic path fired the repair whenever ANY installed post_execution controller's
predicate returned True. That is too permissive in exactly the way capability identity exists to
prevent: a controller installed for a different mechanism at the same boundary -- say a
low-similarity reprompt -- would trigger a store-rewriting repair it never asked for, and the arm
would measure something other than what it declared.

THE FIX. The path runs only when the firing controller's spec NAMES this executor:

    capability_id == "reduce_payload_preserving_facts_and_replace"   AND   action/operator is the
    transform family

Both come off the installed spec, which `install_controller.SpecPredicate` already carries. A
controller that fires but names another executor is recorded as DECLINED with its identity, not
silently ignored -- "the hook chose not to act" and "the hook never saw it" are different facts.
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "# [anchoropt-patch:capability_identity_repair]"

ANCHOR = '''                        for _cc in _cr_ctls:
                            try:
                                if _cc.fires_on(_cr_state):
                                    _generic_capacity_repair_requested = True
'''

BLOCK = '''                        ''' + MARKER + '''
                        # CAPABILITY IDENTITY IS REQUIRED, not just a firing predicate.
                        #
                        # Without this, ANY installed post_execution controller whose phi happens to
                        # hold would trigger a store-rewriting repair it never asked for -- e.g. a
                        # low-similarity reprompt -- and the arm would measure a mechanism it did not
                        # declare. The cell legitimately hosts two executors, so "a controller fired"
                        # is not an identity.
                        _WANT_CAP = "reduce_payload_preserving_facts_and_replace"
                        for _cc in _cr_ctls:
                            _cc_spec = dict(getattr(_cc, "spec", {}) or {})
                            _cc_cap = str(_cc_spec.get("capability_id") or "")
                            _cc_op = str(_cc_spec.get("operator") or _cc_spec.get("action") or "")
                            if _cc_cap != _WANT_CAP or _cc_op not in ("transform", "reroute"):
                                step_record["capacity_repair_declined_gate"] = (
                                    "%s names capability %r/%s, not this executor" % (
                                        getattr(_cc, "name", "controller"), _cc_cap or "(none)",
                                        _cc_op or "(none)"))
                                continue
                            try:
                                if _cc.fires_on(_cr_state):
                                    _generic_capacity_repair_requested = True
'''


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_capability_identity_repair.py <path to memory_evaluator.py>")
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
    bak = p.with_suffix(p.suffix + ".pre_capidentity")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    p.write_text(src.replace(ANCHOR, BLOCK, 1))
    print(f"patched {p}\n  the repair now requires the installed spec to NAME this executor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
