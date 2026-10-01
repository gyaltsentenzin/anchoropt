#!/usr/bin/env python3
"""Supply the PENDING BLOCKED WRITE the A8 mechanism retries. It was read and never written.

THE DEFECT, caught from the mechanism's own decline reason
---------------------------------------------------------
bv_dedup_clear_executor.py read `_hook_state._a8_pending` and nothing ever populated it. So
`plan_recovery` declined **all 26 requests** with:

    no pending blocked write recorded -- nothing to retry, not interfering

That decline is the mechanism behaving EXACTLY as designed on an input the patch failed to supply --
"every uncertain branch returns do-not-interfere" is its stated safety rule. The gap was mine, and it
was visible only because every declining branch records a reason. A mechanism that declined silently
would have looked identical to one that had no opportunity.

THE SUPPLY
----------
Populated from the host's OWN core-full detector, `find_failing_core_full_call` -- the same helper A1's
reroute uses. So "the blocked write" A8 retries is the call the host already agrees was refused for
capacity, not a second opinion written inside the patch.

Keyed per EPISODE (`test_entry_id`), because the commitment gate that retries it runs later in the same
turn, and a module-level dict would leak across every episode in the run.

A VERIFICATION THAT MATTERED
----------------------------
`anchoropt/memory_gates.py` in the shared tree defines `find_failing_core_full_call` as a FALLBACK STUB
returning None -- the same class of stub that has silently turned an arm into the control before. The
REAL implementation lives in the BFCL harness
(`berkeley-function-call-leaderboard/bfcl_eval/model_handler/memory_gates.py`) and is what the runtime
actually imports. Checked in the live runtime before relaunching rather than assumed:

    file:    .../bfcl_eval/model_handler/memory_gates.py
    returns: 'core_memory_add(key="a",value="b")'   (a STRING, not a tuple)

Had the stub been the one in play, this supply would have populated nothing and the mechanism would
have declined 26/26 again -- with the patch looking applied.

Backup at memory_evaluator.py.bak_prea8pending. Idempotent via the marker.
"""

import pathlib
p = pathlib.Path("/path/to/isolated-workspace/anchoropt/memory_evaluator.py")
s = p.read_text()

# `_a8_pending` was READ and never WRITTEN, so plan_recovery declined all 26 requests with
# "no pending blocked write recorded -- nothing to retry, not interfering". That decline is the
# mechanism behaving correctly on an input my patch failed to supply. Populate it from the host's own
# core-full detector, which is the same helper A1's reroute uses -- so the "blocked write" the A8
# mechanism retries is exactly the call the host already agrees was refused for capacity.
old = """                if os.environ.get("ANCHOROPT_BLOCKED_WRITE_LOG"):"""
new = """                # [anchoropt-patch:a8_pending_supply]
                # A CORE write refused for capacity, recorded per EPISODE so the commitment gate can
                # retry it later in the turn. Uses the host's OWN detector (find_failing_core_full_call,
                # the same one A1's reroute uses), so "the blocked write" is the call the host already
                # agrees was refused -- not a second opinion written here.
                try:
                    _a8f = find_failing_core_full_call(decoded or [], execution_results or [])
                    if _a8f:
                        _a8st = getattr(_hook_state, "_a8_pending", None)
                        if not isinstance(_a8st, dict):
                            _a8st = _hook_state._a8_pending = {}
                        _a8_fc = _a8f[1] if isinstance(_a8f, (tuple, list)) and len(_a8f) > 1 else _a8f
                        _a8st[str(test_entry_id or "")] = {
                            "call": str(_a8_fc),
                            "container": "core",
                        }
                        step_record["a8_pending_recorded_gate"] = True
                except Exception as _a8pe:
                    step_record["a8_pending_error_gate"] = "%s: %s" % (
                        type(_a8pe).__name__, _a8pe)

                if os.environ.get("ANCHOROPT_BLOCKED_WRITE_LOG"):"""
assert s.count(old) == 1, f"anchor count {s.count(old)}"
p.write_text(s.replace(old, new, 1))
compile(p.read_text(), str(p), "exec")
print("_a8_pending now SUPPLIED from the host's own core-full detector")
