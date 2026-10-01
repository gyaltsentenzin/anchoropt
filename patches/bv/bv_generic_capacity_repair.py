#!/usr/bin/env python3
"""Let a GENERIC installed controller reach the fact-preserving capacity repair.

ISOLATED COPIES ONLY -- refuses the shared tree.

The repair itself is already generic and already validated (352/352 replacements constructed on live
captures, median 1% compression, 99.9% of query-relevant tokens preserved). It is reachable today ONLY
via `remedy_enabled(templates, "enable_capacity_repair")` -- an anchor policy flag that a controller
installed through `runtime_hook` cannot set. So the capability exists, runs, and is invisible to
autonomous search.

This adds an OR: the branch also runs when an installed controller at post_execution fires on the
step's state. Whether to act is the INSTALLED PREDICATE's decision; the repair's own logic decides
what the replacement is, exactly as it does today. No anchor flag is read on this path, no threshold,
no trigger, no anchor identity.

The existing `_capacity_repair_fired` latch is reused, so the once-per-turn bound is the host's own
rather than a second copy.
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "# [anchoropt-patch:generic_capacity_repair]"

ANCHOR = '''                    and remedy_enabled(templates, "enable_capacity_repair")
'''

BLOCK = '''                    ''' + MARKER + '''
                    # EITHER the host's own policy flag, OR an installed controller that fires here.
                    #
                    # `remedy_enabled(...)` is an anchor policy flag; a controller AnchorOpt installs
                    # cannot set it, which is the only reason this validated repair was unreachable to
                    # the search. `_generic_capacity_repair_requested` is computed just above from the
                    # installed controllers' own predicates, so the decision to act stays theirs.
                    and (remedy_enabled(templates, "enable_capacity_repair")
                         or _generic_capacity_repair_requested)
'''

# Computed immediately before the guard, so it is in scope and reads only post-execution state.
PRE_ANCHOR = '''                if (
                    not self.disable_gates
                    and not _capacity_repair_fired
'''

PRE_BLOCK = '''                # DOES AN INSTALLED CONTROLLER ASK FOR A REPAIR HERE?
                #
                # Asked of the post-execution controllers on this step's own state, via the same
                # `_hook_state` the other post-execution gates use -- so the predicate sees exactly
                # what the host declares observable at this boundary, and phase eligibility is the
                # host's own check rather than a second copy.
                _generic_capacity_repair_requested = False
                try:
                    from anchoropt.runtime_hook import installed as _crinst
                    _cr_ctls = [c for c in _crinst("post_execution")
                                if _phase_eligible(c, str(rollout_tag) == "snap")]
                    if _cr_ctls:
                        _cr_state = _hook_state(step_record, decoded, execution_results,
                                                test_entry_id)
                        for _cc in _cr_ctls:
                            try:
                                if _cc.fires_on(_cr_state):
                                    _generic_capacity_repair_requested = True
                                    step_record["capacity_repair_requested_gate"] = True
                                    step_record["capacity_repair_controller"] = getattr(
                                        _cc, "name", "controller")
                                    break
                            except Exception as _cce:
                                step_record["capacity_repair_predicate_error_gate"] = "%s: %s" % (
                                    type(_cce).__name__, _cce)
                except Exception as _cre:
                    step_record["capacity_repair_hook_error_gate"] = "%s: %s" % (
                        type(_cre).__name__, _cre)

                if (
                    not self.disable_gates
                    and not _capacity_repair_fired
'''


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_generic_capacity_repair.py <path to memory_evaluator.py>")
        return 2
    p = pathlib.Path(sys.argv[1])
    if "/anchoropt-wei/anchoropt/anchoropt/" in str(p.resolve()):
        print("REFUSING: that is the SHARED tree, imported by running GEPA workers.")
        return 3
    src = p.read_text()
    if MARKER in src:
        print(f"already applied ({MARKER} present) -- no change")
        return 0
    for name, a in (("guard", ANCHOR), ("pre-guard", PRE_ANCHOR)):
        if src.count(a) != 1:
            print(f"FAIL: {name} anchor appears {src.count(a)} times; refusing to guess")
            return 1
    bak = p.with_suffix(p.suffix + ".pre_genericrepair")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    out = src.replace(PRE_ANCHOR, PRE_BLOCK, 1).replace(ANCHOR, BLOCK, 1)
    p.write_text(out)
    print(f"patched {p}\n  a generic controller can now request the fact-preserving repair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
