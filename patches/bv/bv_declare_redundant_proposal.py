#!/usr/bin/env python3
"""Declare a PRE-COMMITMENT redundancy signal over information the boundary already has.

ISOLATED COPIES ONLY -- refuses the shared tree.

WHAT GAP THIS CLOSES
--------------------
This host declares 9 signals and NONE expresses "the write about to be dispatched is redundant with
what the store already holds". The only observable form is `duplicate_identifier` AFTER the store has
refused -- which occurs 4 / 0 / 0 times across kv / vector / rec_sum, while a redundant write PROPOSED
before commitment occurs 449 / 104 / 178 times. ~100x more common, at the boundary where cancelling is
free, and invisible to the search.

The attributor has repeatedly named this repair and the proposer recorded it UNREALIZABLE with exactly
that reason: "no declared signal observes this condition".

WHAT IT REUSES, AND WHAT IT REFUSES TO REUSE
--------------------------------------------
REUSES: `redundant_write.should_suppress(call_args, mem_inst, key_field, value_field)` -- a generic
store comparison already in the host. It is duck-typed over container names, takes the field names as
PARAMETERS, reads LIVE store state, refuses truncated telemetry outright, and returns False WITH A
REASON whenever the answer is not certain (absent key, missing value, any difference at all). It
decides one thing: is this proposed value already stored under this key.

REFUSES: every anchor policy around it. No `enable_*_redundant_write_suppress` flag is read, no
threshold, no priority, no trigger, and no anchor identity. Whether to act is the INSTALLED
PREDICATE's decision and the action is whatever core grounds for the signal.

SAFETY IS A PROPERTY OF THE COMPARATOR, not of a threshold I chose: an exact (key, value) repeat
provably carries no new information, and a same-key/different-value write is an UPDATE for which
`is_redundant` already returns False. Measured on raw H0: 408 exact repeats vs 41 updates in kv.
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "# [anchoropt-patch:declare_redundant_proposal]"

SIGNAL_SRC = '''

''' + MARKER + '''
def redundant_proposed_write(state, params):
    """A write is PROPOSED whose value is already stored under its key.

    Observable strictly PRE-EXECUTION: the call has not run, so cancelling it is free. After
    execution the store has already refused and the step is spent -- which is the DIFFERENT and far
    rarer condition `duplicate_identifier` covers.

    The comparison is the host's own generic one (`redundant_write.is_redundant`): same key,
    NORMALISED-IDENTICAL value, nothing fuzzy. An absent key, a missing value, or any difference at
    all answers False -- the default is to let the write through, so a genuine UPDATE
    (same key, different value) is never suppressed.

    Requires `proposal_is_redundant`, which the adapter supplies at the commitment gate by asking the
    live store. A state without it answers False rather than guessing.
    """
    if not bool(state.get("proposes_write", False)):
        return False
    return bool(state.get("proposal_is_redundant", False))
'''

# The BV module's registry line carries a type annotation, and it has NO SIGNAL_BOUNDARIES table --
# the installed spec carries its own locus there. Both were verified by reading the file rather than
# assumed, after a first draft guessed wrong on each.
REG_ANCHOR = '''    "container_at_capacity": container_at_capacity,'''
REG_LINE = '''    "redundant_proposed_write": redundant_proposed_write,
    "container_at_capacity": container_at_capacity,'''


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_declare_redundant_proposal.py <path to bfcl_declared_signals.py>")
        return 2
    p = pathlib.Path(sys.argv[1])
    if "/anchoropt-wei/anchoropt/anchoropt/" in str(p.resolve()):
        print("REFUSING: that is the SHARED tree, imported by running GEPA workers.")
        return 3
    src = p.read_text()
    if MARKER in src:
        print(f"already applied ({MARKER} present) -- no change")
        return 0

    if src.count(REG_ANCHOR) != 1:
        print("FAIL: registry anchor appears %d times; refusing to guess" % src.count(REG_ANCHOR))
        return 1

    bak = p.with_suffix(p.suffix + ".pre_redundant")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")

    # Define the predicate just before the registry, then register it.
    out = src.replace("SIGNALS: Mapping[str, Callable",
                      SIGNAL_SRC.strip("\n") + "\n\n\nSIGNALS: Mapping[str, Callable", 1)
    out = out.replace(REG_ANCHOR, REG_LINE, 1)
    p.write_text(out)
    print(f"patched {p}\n  declared: redundant_proposed_write (locus travels on the installed spec)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
