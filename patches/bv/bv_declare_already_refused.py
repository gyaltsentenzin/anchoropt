#!/usr/bin/env python3
"""Declare `proposal_already_refused` and supply `proposal_refused_count` in the isolated runtime.

ISOLATED COPIES ONLY -- refuses the shared tree.

The condition the attributor named twice and no signal expressed: this exact call was already REFUSED
earlier in this episode, so re-emitting it cannot succeed. Distinct from the committed-duplicate
condition, and on kv prereq the two populations are comparable in size (197 refused-repeats vs 199
committed-repeats) -- only the second had a signal, and suppressing it measured net -3.

Keyed on the normalised CALL TEXT, not on a key: "the same call" is the repair's subject, and keying on
the key would conflate an UPDATE with a resubmission. Episode-scoped and strictly backward-looking --
refusals are recorded from `trajectory_history`, which holds only COMPLETED steps, so the state carries
nothing about the call being judged.

No anchor policy flag is read, no threshold, no priority, no anchor identity.
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "# [anchoropt-patch:declare_already_refused]"

SIG_ANCHOR = '''    "redundant_proposed_write": redundant_proposed_write,'''
SIG_LINE = '''    "redundant_proposed_write": redundant_proposed_write,
    "proposal_already_refused": proposal_already_refused,'''

PRED = '''

''' + MARKER + '''
def proposal_already_refused(state, params):
    """This exact call was already REFUSED earlier in this episode.

    `proposal_refused_count` counts refusals of the SAME normalised call text recorded strictly
    BEFORE this call was proposed. A first attempt always passes.
    """
    if not bool(state.get("proposes_tool_call", True)):
        return False
    return int(state.get("proposal_refused_count", 0) or 0) > 0
'''

SUPPLY_ANCHOR = '''                        _ust["proposal_is_redundant"] = False
'''
SUPPLY_BLOCK = '''                        # WAS THIS EXACT CALL ALREADY REFUSED THIS EPISODE?
                        #
                        # From `trajectory_history`, which holds only COMPLETED steps -- so this cannot
                        # see the outcome of the call being judged. Keyed on normalised call text.
                        _ust["proposal_refused_count"] = 0
                        try:
                            import re as _rfre
                            _rf_want = " ".join(str(_uc_call).split())
                            _rf_n = 0
                            _ERRISH = _rfre.compile(
                                r"error|cannot|unable|must be|full|exceed|invalid|unique", _rfre.I)
                            for _pstep in (trajectory_history or []):
                                _pc = [str(x) for x in (_pstep.get("decoded") or [])]
                                _pr = [str(x) for x in (_pstep.get("tool_results") or [])]
                                for _pi, _pcall in enumerate(_pc):
                                    if " ".join(_pcall.split()) != _rf_want:
                                        continue
                                    _prt = _pr[_pi] if _pi < len(_pr) else ""
                                    if _prt and _ERRISH.search(_prt):
                                        _rf_n += 1
                            _ust["proposal_refused_count"] = _rf_n
                        except Exception as _rfe:
                            step_record["refused_count_error_gate"] = "%s: %s" % (
                                type(_rfe).__name__, _rfe)
                        _ust["proposal_is_redundant"] = False
'''


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: bv_declare_already_refused.py <bfcl_declared_signals.py> <memory_evaluator.py>")
        return 2
    sig, ev = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    for p in (sig, ev):
        if "/anchoropt-wei/anchoropt/anchoropt/" in str(p.resolve()):
            print("REFUSING: that is the SHARED tree, imported by running GEPA workers.")
            return 3

    s1 = sig.read_text()
    if MARKER not in s1:
        if s1.count(SIG_ANCHOR) != 1:
            print("FAIL: signal anchor appears %d times" % s1.count(SIG_ANCHOR))
            return 1
        bak = sig.with_suffix(sig.suffix + ".pre_refused")
        if not bak.exists():
            bak.write_text(s1)
        s1 = s1.replace("SIGNALS: Mapping[str, Callable",
                        PRED.strip("\n") + "\n\n\nSIGNALS: Mapping[str, Callable", 1)
        s1 = s1.replace(SIG_ANCHOR, SIG_LINE, 1)
        sig.write_text(s1)
        print("declared proposal_already_refused in %s" % sig.name)
    else:
        print("signal already declared -- no change")

    s2 = ev.read_text()
    if "proposal_refused_count" in s2:
        print("supply already present -- no change")
        return 0
    if s2.count(SUPPLY_ANCHOR) != 1:
        print("FAIL: supply anchor appears %d times" % s2.count(SUPPLY_ANCHOR))
        return 1
    bak2 = ev.with_suffix(ev.suffix + ".pre_refusedsupply")
    if not bak2.exists():
        bak2.write_text(s2)
    ev.write_text(s2.replace(SUPPLY_ANCHOR, SUPPLY_BLOCK, 1))
    print("supplies proposal_refused_count in %s" % ev.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
