#!/usr/bin/env python3
"""Let a GENERIC installed controller reach the zero-tool-call commitment decision.

ISOLATED COPIES ONLY -- refuses the shared tree.

THE DEFECT, AND WHY PLACEMENT IS THE WHOLE FIX
----------------------------------------------
A step that proposes no tool call sets `status="answer_end_turn"` and **BREAKS out of the step loop**
hundreds of lines before the commitment-gate hook. So that hook runs only on steps that DO propose a
call, and an installed controller for the no-call condition is never asked. Measured twice on real GPU
runs: 86 hook reaches, 0 firings, while the identical predicate fires 89/175 on the offline projection.

My first attempt patched the hook itself. That was the wrong place -- the hook is downstream of the
exit, so the branch was unreachable and the telemetry was unchanged. A source-marker check reported
"live parity OK" and proved nothing, which is exactly why execution must be read from trajectory
telemetry rather than from the presence of code.

This version inserts the generic path BESIDE the existing zero-call remedy branch, inside the same
scope and before that break.

WHAT IT REUSES
--------------
The host's own injector (`_add_next_turn_user_message_prompting`) and the host's own once-per-turn latch
(`_zero_call_reprompted`) -- the same two the anchor branch above uses. Retry budget is therefore
enforced by the existing mechanism, not by a second copy of it. `_phase_eligible` gates it, so a
query-phase controller stays inert during storage construction.

WHAT IT DOES NOT DO
-------------------
It reads no anchor policy flag, names no anchor, and encodes no trigger of its own: whether to act is
the INSTALLED PREDICATE's decision and the injected text is the controller's own `eta['instruction']`,
so two arms differing only in instruction execute differently. The anchor branch is untouched.
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "# [anchoropt-patch:generic_zero_call_gate]"

ANCHOR = "                        # A5 CANDIDATE -- post_generation_pre_commit on LOW RETRIEVAL SIMILARITY.\n"

block_body = '''                        %s
                        # GENERIC CONTROLLER PATH for the zero-tool-call commitment.
                        #
                        # Sits HERE, beside the anchor branch above, because a no-call step breaks out
                        # of the step loop long before the commitment-gate hook runs.
                        #
                        # Reuses the host's injector AND its once-per-turn latch, so the retry budget is
                        # the existing mechanism rather than a second copy. No anchor flag is read and no
                        # trigger is encoded: the predicate decides, and the text is the controller's eta.
                        if (not self.disable_gates and not _zero_call_reprompted
                                and _query_tool_calls == 0
                                and step_count < self.max_steps_per_turn):
                            try:
                                from anchoropt.runtime_hook import installed as _zinst
                                _zctls = [c for c in _zinst("post_generation_pre_exec")
                                          if _phase_eligible(c, str(rollout_tag) == "snap")]
                            except Exception as _ze0:
                                _zctls = []
                                step_record["upstream_phase_error_gate"] = "%%s: %%s" %% (
                                    type(_ze0).__name__, _ze0)
                            if _zctls:
                                step_record["upstream_hook_gate"] = True
                                _zst = {"proposed_call": None, "call_index": 0,
                                        "n_proposed_calls": 0,
                                        "step_index": step_record.get("step"),
                                        "boundary": "post_generation_pre_exec",
                                        "phase": ("prereq" if str(rollout_tag) == "snap" else "query"),
                                        "container_full": bool(
                                            (getattr(_hook_state, "_capacity_seen", None) or {}).get(
                                                str(test_entry_id or ""), False)),
                                        "has_generation": True, "proposes_tool_call": False,
                                        "tool_calls_so_far": int(_query_tool_calls or 0),
                                        "proposes_read": False, "proposes_write": False,
                                        "proposes_clear": False, "proposes_remove": False}
                                _zhit = None
                                for _zc in _zctls:
                                    try:
                                        if _zc.fires_on(_zst):
                                            _zhit = _zc
                                            break
                                    except Exception as _ze:
                                        step_record["upstream_error_gate"] = "%%s: %%s" %% (
                                            type(_ze).__name__, _ze)
                                if _zhit is not None:
                                    _zmsg = str((getattr(_zhit, "eta", {}) or {}).get(
                                        "instruction") or "").strip()
                                    if _zmsg:
                                        _zero_call_reprompted = True
                                        step_record["upstream_fired_gate"] = True
                                        step_record["upstream_zero_call_gate"] = True
                                        step_record["upstream_controller"] = getattr(
                                            _zhit, "name", "controller")
                                        step_record["upstream_injected_chars_gate"] = len(_zmsg)
                                        step_record["upstream_phase_gate"] = _zst["phase"]
                                        inference_data = (
                                            self.handler._add_next_turn_user_message_prompting(
                                                inference_data,
                                                [{"role": "user", "content": _zmsg}]))
                                        trajectory_history.append(step_record)
                                        step_count += 1
                                        continue
'''

BLOCK = (block_body % MARKER) + ANCHOR


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_generic_zero_call_gate.py <path to memory_evaluator.py>")
        return 2
    p = pathlib.Path(sys.argv[1])
    if "/anchoropt-wei/anchoropt/anchoropt/" in str(p.resolve()):
        print("REFUSING: that is the SHARED tree, imported by running GEPA workers.")
        return 3
    src = p.read_text()
    if MARKER in src:
        print("already applied (%s present) -- no change" % MARKER)
        return 0
    if src.count(ANCHOR) != 1:
        print("FAIL: anchor appears %d times; refusing to guess" % src.count(ANCHOR))
        return 1
    bak = p.with_suffix(p.suffix + ".pre_zerocall")
    if not bak.exists():
        bak.write_text(src)
        print("backup: %s" % bak)
    p.write_text(src.replace(ANCHOR, BLOCK, 1))
    print("patched %s\n  marker inserted: %s" % (p, MARKER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
