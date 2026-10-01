#!/usr/bin/env python3
"""The cancel-only withhold branch OVERWROTE A8's replacement sequence before dispatch.

THE DEFECT: TWO PATCHES WRITING ONE VARIABLE
--------------------------------------------
`bv_dedup_clear_executor.py` sets `_mg_exec_calls = list(_a8_seq)` -- the (remove, retry) sequence -- at
~line 5013. The PRE-EXISTING cancel-only withhold branch then runs at ~5059:

    _mg_exec_calls = [c for _i, c in _up_keep]

unconditionally, discarding A8's sequence before `self._execute()` ever saw it. So the eviction was never
dispatched, the store never changed, and the deferred verification CORRECTLY reported
`victim_removed=False, copies_remaining=2` on every firing. Two patches writing one variable, later one
wins -- and the later one was not the one that had decided to act.

This survived the previous fix (`bv_a8_verify_after_dispatch.py`) because that fix corrected WHEN the
verification ran, which was also wrong. Both were real; the second was hidden behind the first.

THE FIX
-------
The withhold branch now defers when A8 has staged a sequence (`_a8_verify_pending is not None`), and
records `dedup_clear_kept_sequence_gate` so the choice is visible in telemetry. When A8 did not act,
behaviour is byte-identical to before.

A8's sequence ALREADY excludes the suppressed clear -- it replaces it -- so deferring here is what
"suppress only if this gate can supply the capacity in the same action" means operationally. The
alternative (union the two lists) would re-dispatch the clear the mechanism just decided to prevent.

`dedup_clear_kept_sequence_gate` is covered by the sidecar's existing `dedup_clear` prefix, checked
rather than assumed.

WHAT THIS ADDS TO THE ORDERING RULE
-----------------------------------
`_mg_exec_calls` is the dispatch boundary, and by now THREE separate patches write it. Anything new that
writes it must state its precedence against the others explicitly, because the file's own control flow
gives no hint that an earlier assignment is load-bearing. Grep every writer before adding a fourth.

Backup at memory_evaluator.py.bak_preorder.
"""

import pathlib
p = pathlib.Path("/path/to/isolated-workspace/anchoropt/memory_evaluator.py")
s = p.read_text()

# THE DEFECT: my A8 block sets _mg_exec_calls = (remove, retry) at ~5013, and the EXISTING cancel-only
# withhold branch then OVERWRITES it at ~5059 with the kept-calls list. So the eviction sequence was
# discarded before self._execute ever saw it -- the store never changed, and the deferred verification
# correctly reported victim_removed=False. Two patches writing the same variable, later one wins.
old = """                        if _spent < _budget:
                            # WITHHOLD, then let the model propose again. The withheld call does not
                            # execute, so no error is produced downstream -- which is the whole
                            # difference from repairing after the fact.
                            _mg_exec_calls = [c for _i, c in _up_keep]"""
new = """                        if _spent < _budget:
                            # WITHHOLD, then let the model propose again. The withheld call does not
                            # execute, so no error is produced downstream -- which is the whole
                            # difference from repairing after the fact.
                            #
                            # UNLESS A8 ALREADY SUPPLIED A REPLACEMENT SEQUENCE. This assignment used to
                            # run unconditionally and overwrote the (remove, retry) sequence the A8
                            # executor had just placed on _mg_exec_calls, so the eviction was discarded
                            # before dispatch: the store never changed and the deferred verification
                            # correctly reported victim_removed=False on every firing. Two patches
                            # writing one variable, later one wins -- and the later one was not the one
                            # that had decided to act.
                            #
                            # A8's sequence ALREADY excludes the suppressed clear (it replaces it), so
                            # deferring to it here is what "suppress only if this gate can supply the
                            # capacity in the same action" means operationally. When A8 did not act,
                            # behaviour is byte-identical to before.
                            if getattr(_hook_state, "_a8_verify_pending", None) is not None:
                                step_record["dedup_clear_kept_sequence_gate"] = True
                            else:
                                _mg_exec_calls = [c for _i, c in _up_keep]"""
assert s.count(old) == 1, f"anchor count {s.count(old)}"
p.write_text(s.replace(old, new, 1))
compile(p.read_text(), str(p), "exec")
print("withhold branch now DEFERS to an A8 replacement sequence")
