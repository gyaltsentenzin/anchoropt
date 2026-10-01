#!/usr/bin/env python3
"""The A8 verification ran BEFORE the dispatch, so it read pre-execution state.

THE DEFECT, caught from the mechanism's own telemetry
-----------------------------------------------------
bv_dedup_clear_executor.py set `_mg_exec_calls` to the (remove, retry) sequence and then called
`verify(live_after, plan)` in the same breath. But `_mg_exec_calls` is not dispatched there --
`self._execute()` consumes it further down. So the "live_after" state was the state BEFORE anything
ran. Measured on job 1859026:

    dedup_clear_gate                    2
    dedup_clear_invariant_violated_gate 2
    -> INVARIANT VIOLATED: victim_removed=False, copies_remaining=2

The store was simply untouched at the moment of the check. **The invariant was fine; the CLOCK was
wrong** -- the same class of error as scoring a controller in a phase it does not act in, one level
smaller.

Left unfixed, this would have failed criterion 3 on every firing (`mechanism_verified` reads
`dedup_clear_verified_gate`) and, worse, counted every eviction as an information-losing remove under
criterion 4 -- reporting a safety violation the mechanism had not committed.

THE FIX
-------
Stash `(plan, container)` on the hook state when an eviction is dispatched, then verify immediately
AFTER `self._execute()` returns, against `involved_instances` the host has actually mutated. The pending
tuple is cleared whether or not the check succeeds, so a verification can never be attributed to a later
step, and a raising check records `dedup_clear_verify_error_gate` rather than silently leaving the
previous step's verdict in place.

WHY THE TELEMETRY MADE THIS VISIBLE
-----------------------------------
`verify()` returns the components, not just ok/not-ok: `victim_removed=False` AND
`copies_remaining=2` (unchanged from `copies_before=2`) together say "nothing happened", which is a
different story from "the eviction destroyed the last copy". A bare boolean would have read as a real
safety violation and I would have gone looking for a bug in the eviction logic instead of in my own
call ordering.

Backup at memory_evaluator.py.bak_preverifytiming.
"""

import pathlib
p = pathlib.Path("/path/to/isolated-workspace/anchoropt/memory_evaluator.py")
s = p.read_text()

# 1) STASH the plan instead of verifying immediately. The old code set _mg_exec_calls and then verified
#    in the same breath -- BEFORE self._execute() at line ~5085 had dispatched anything. It was reading
#    PRE-execution state, so victim_removed was always False and copies_remaining never dropped:
#    measured 2 evictions, 2 "INVARIANT VIOLATED: victim_removed=False, copies_remaining=2".
old_verify = """                                # VERIFY AGAINST LIVE STATE, after the eviction and before trusting
                                # anything. ok=False means the last copy was destroyed -- recorded,
                                # never inferred from an accuracy delta later.
                                _a8_after = None
                                for _nm, _inst in (involved_instances or {}).items():
                                    _cand = getattr(_inst, (_a8_cont or "") + "_memory", None)
                                    if isinstance(_cand, dict):
                                        _a8_after = dict(_cand)
                                        break
                                _a8_v = _a8m.verify(_a8_after, _a8_plan)
                                step_record["dedup_clear_verified_gate"] = bool(_a8_v.get("ok"))
                                step_record["dedup_clear_copies_remaining"] = int(
                                    _a8_v.get("copies_remaining") or 0)
                                step_record["dedup_clear_victim_removed"] = bool(
                                    _a8_v.get("victim_removed"))
                                if not _a8_v.get("ok"):
                                    step_record["dedup_clear_invariant_violated_gate"] = str(
                                        _a8_v.get("reason") or "")[:200]"""
new_verify = """                                # DEFER THE VERIFICATION. `_mg_exec_calls` is not dispatched here --
                                # `self._execute` consumes it further down -- so verifying now reads
                                # PRE-execution state. Measured: 2 evictions, 2 "INVARIANT VIOLATED:
                                # victim_removed=False, copies_remaining=2", i.e. the store was simply
                                # untouched at the moment of the check. The invariant was fine; the
                                # CLOCK was wrong, which is the same class as scoring a controller in a
                                # phase it does not act in.
                                _hook_state._a8_verify_pending = (_a8_plan, _a8_cont)"""
assert s.count(old_verify) == 1, f"verify anchor count {s.count(old_verify)}"
s = s.replace(old_verify, new_verify, 1)

# 2) VERIFY AFTER DISPATCH, against state the host has actually mutated.
old_exec = """                execution_results, involved_instances = self._execute("""
new_exec = """                # [anchoropt-patch:a8_verify_after_dispatch]
                # Verify the A8 eviction against state the host HAS mutated. Placed immediately after
                # the dispatch below by construction: the pending tuple is set only when an eviction was
                # dispatched this step, and is cleared whether or not the check succeeds so it can never
                # be attributed to a later step.
                _a8_pend_v = getattr(_hook_state, "_a8_verify_pending", None)

                execution_results, involved_instances = self._execute("""
assert s.count(old_exec) == 1, f"exec anchor count {s.count(old_exec)}"
s = s.replace(old_exec, new_exec, 1)

# 3) The check itself, right after the dispatch call's closing paren. Anchor on the comment that
#    follows it so the insertion point is unambiguous.
old_after = """                # what _add_execution_results_prompting zips against. Verified by asserting the"""
new_after = """                if _a8_pend_v is not None:
                    try:
                        _hook_state._a8_verify_pending = None
                        _a8_plan_v, _a8_cont_v = _a8_pend_v
                        _a8_after = None
                        for _nm, _inst in (involved_instances or {}).items():
                            _cand = getattr(_inst, (_a8_cont_v or "") + "_memory", None)
                            if isinstance(_cand, dict):
                                _a8_after = dict(_cand)
                                break
                        import dedup_clear_lossless as _a8mv
                        _a8_v = _a8mv.verify(_a8_after, _a8_plan_v)
                        step_record["dedup_clear_verified_gate"] = bool(_a8_v.get("ok"))
                        step_record["dedup_clear_copies_remaining"] = int(
                            _a8_v.get("copies_remaining") or 0)
                        step_record["dedup_clear_victim_removed"] = bool(_a8_v.get("victim_removed"))
                        if not _a8_v.get("ok"):
                            # ok=False means this mechanism just destroyed the last copy of something --
                            # the one outcome its acceptance argument forbids. Recorded, never inferred
                            # later from an accuracy delta.
                            step_record["dedup_clear_invariant_violated_gate"] = str(
                                _a8_v.get("reason") or "")[:200]
                    except Exception as _a8ve:
                        step_record["dedup_clear_verify_error_gate"] = "%s: %s" % (
                            type(_a8ve).__name__, _a8ve)

                # what _add_execution_results_prompting zips against. Verified by asserting the"""
assert s.count(old_after) == 1, f"after anchor count {s.count(old_after)}"
s = s.replace(old_after, new_after, 1)

p.write_text(s)
compile(p.read_text(), str(p), "exec")
print("verification DEFERRED to after dispatch")
