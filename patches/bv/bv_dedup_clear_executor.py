#!/usr/bin/env python3
"""Make A8's COMPLETE lossless recovery reachable to an installed controller at the commitment gate.

ISOLATED COPIES ONLY -- refuses the shared tree.

WHY THIS IS NEEDED, and it is now MEASURED rather than argued
-------------------------------------------------------------
The commitment gate already has a suppress executor, and it works: R5's arm withheld 67 proposed
clears and `core_memory_clear` calls fell 83 -> 67. But suppression is only HALF of A8. The hook
consumes `retry_budget` and explicitly does NOT read `suppressed_operation` or `preservation`, so a
controller cannot ask it to evict, verify and retry -- the repair half has no executor.

R5 measured what that costs: **net -5** (+7/-12) with the mechanism executing correctly. So this is a
successful intervention with a negative outcome, which is exactly what `docs/ANCHORS.md` predicts:

    a predecessor refused clears UNCONDITIONALLY -- 85 refusals, and 0 of 14 episodes closed. Median kv
    prereq steps went 19 -> 42, max 43 -> 189. Refusing destruction *without supplying capacity* leaves
    the model no route out, so it re-proposes rather than revises.

A8's design rule follows from that, and this patch exists to honour it: **suppress only when this gate
can itself supply the capacity in the same action.** Where it cannot, letting the destructive call
proceed is the CORRECT branch.

WHAT THIS PATCH DOES AND DOES NOT DO
------------------------------------
Adds ONE branch at the commitment gate, reached only by a controller whose spec names
`capability_id = "suppress_clear_evict_redundant_verify_and_retry"`. That branch:

  1. asks `dedup_clear_recovery.plan_recovery` whether capacity can be supplied instead of clearing;
  2. if `fire` is False -- for ANY reason -- FALLS THROUGH UNCHANGED, so the clear executes exactly as
     it would have. Every uncertain branch in the mechanism already returns "do not interfere";
  3. if `fire` is True, dispatches `dispatch_sequence(plan)` = (remove_call, retry_call) in that order,
     then calls `verify(live_after, plan)` against LIVE state and records the result.

It adds NO safety logic of its own. Every invariant lives in `dedup_clear_recovery.py` and is checked
there against live state: >= 2 copies before, >= 1 equivalent copy after, the victim actually gone, the
retry byte-identical to the blocked write. `verify(ok=False)` means the anchor just destroyed the last
copy of something -- the one outcome its acceptance argument forbids -- so it is written to telemetry
rather than inferred later from an accuracy delta.

CAPABILITY IDENTITY IS REQUIRED, not optional. The post_generation_pre_exec/suppress cell already has a
signal-agnostic cancel-only executor. Without an identity check, ANY suppression controller would start
evicting entries -- a different mechanism running under the accepted controller's name. Same discipline
as bv_capability_identity_repair and bv_generic_capacity_relocate.

NO TRIGGER, NO THRESHOLD, NO ANCHOR NAME is read on this path. The installed predicate decides whether
to act; the mechanism decides what to do.

WHY THIS IS A VERSIONED BEHAVIOURAL CHANGE
------------------------------------------
It changes execution semantics at a cell where results already exist. It must therefore be a NEW
capability_id and a NEW measurement, never a silent upgrade of the cancel-only arm: R5's
`clear_proposed_at_capacity` is recorded MEASURED_NEGATIVE at net -5 against stack_e50a3bb5dd37, and
that record stays exactly as measured. The full mechanism is a DIFFERENT controller with its own
identity, scored in its own round.

DEPENDENCY: `dedup_clear_recovery.py` must exist in the isolated runtime (stdlib-only; deployed to
/path/to/isolated-workspace/). This patch asserts it imports before touching the evaluator.

Usage:
    python patches/bv/bv_dedup_clear_executor.py --check          # report, change nothing
    python patches/bv/bv_dedup_clear_executor.py --emit           # print the branch it would insert
"""

from __future__ import annotations

import argparse
import pathlib
import sys

MARKER = "# [anchoropt-patch:dedup_clear_executor]"

#: The identity a controller's spec must name to reach this branch. Anything else at this cell keeps
#: the existing cancel-only behaviour.
CAPABILITY_ID = "suppress_clear_evict_redundant_verify_and_retry"

#: Inserted immediately before the existing withhold branch, so the two are visibly siblings: the
#: repair path is tried FIRST for a controller that names it, and cancel-only remains the fallback for
#: every other controller.
ANCHOR = """                    if _up_withheld:
                        _budget = 0
"""

BLOCK = '''                    ''' + MARKER + '''
                    # A8 COMPLETE: suppress the clear ONLY IF capacity can be supplied in the same
                    # action. Reached only by a controller whose spec names this capability_id,
                    # because this cell has more than one executor.
                    #
                    # Every declining branch falls through UNCHANGED and the clear executes as it
                    # would have. That is not a missed opportunity: R5 measured cancel-only
                    # suppression at net -5, and docs/ANCHORS.md records a predecessor that refused
                    # clears unconditionally closing 0 of 14 episodes. Prevention without supplied
                    # capacity leaves the model no route out.
                    _a8c_wants = False
                    for _uc in _up_ctls:
                        try:
                            _a8c_spec = getattr(_uc, "spec", {}) or {}
                            if str(_a8c_spec.get("capability_id") or "") == "%s":
                                _a8c_wants = True
                                break
                        except Exception:
                            pass
                    if _up_withheld and _a8c_wants:
                        step_record["dedup_clear_requested_gate"] = True
                        try:
                            import sys as _a8sys
                            _a8dir = "/path/to/isolated-workspace"
                            if _a8dir not in _a8sys.path:
                                _a8sys.path.insert(0, _a8dir)
                            import dedup_clear_recovery as _a8m

                            # LIVE container named by the proposed call, resolved the same way the
                            # redundancy comparator resolves it -- by the call's own container, never
                            # by dict order over involved_instances.
                            _a8_call = str(_up_withheld[0][1])
                            _a8_cont = _a8m.container_of(_a8_call)
                            _a8_live, _a8_cap = None, None
                            for _nm, _inst in (involved_instances or {}).items():
                                _cand = getattr(_inst, (_a8_cont or "") + "_memory", None)
                                if isinstance(_cand, dict):
                                    _a8_live = dict(_cand)
                                    _a8_cap = getattr(_inst, "_core_limit", None)
                                    break
                            # THE PENDING BLOCKED WRITE: the write that was refused for want of
                            # capacity and is waiting to be retried. Without it there is nothing to
                            # retry and the mechanism declines by its own rule.
                            _a8_pending = getattr(_hook_state, "_a8_pending", None) or {}
                            _a8_fired_n = int(getattr(_hook_state, "_a8_fired_n", 0) or 0)
                            _a8_plan = _a8m.plan_recovery(
                                _a8_call, _a8_live, _a8_pending.get(str(test_entry_id or "")),
                                _a8_cap, firings_so_far=_a8_fired_n)
                            step_record["dedup_clear_reason_gate"] = str(
                                _a8_plan.get("reason") or "")[:200]
                            if _a8_plan.get("fire"):
                                _a8_seq = _a8m.dispatch_sequence(_a8_plan)
                                step_record["dedup_clear_gate"] = True
                                step_record["dedup_clear_sequence"] = [s[:160] for s in _a8_seq]
                                step_record["dedup_clear_copies_before"] = int(
                                    _a8_plan.get("copies_before") or 0)
                                # The host executes the sequence in order; the retry MUST be last and
                                # byte-identical to the blocked write.
                                _mg_exec_calls = list(_a8_seq)
                                _hook_state._a8_fired_n = _a8_fired_n + 1
                                # VERIFY AGAINST LIVE STATE, after the eviction and before trusting
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
                                        _a8_v.get("reason") or "")[:200]
                        except Exception as _a8e:
                            # A raising mechanism must NOT silently become a plain suppression: that
                            # would run cancel-only under this capability's name, which is the arm
                            # identity defect this project has paid for repeatedly.
                            step_record["dedup_clear_error_gate"] = "%%s: %%s" %% (
                                type(_a8e).__name__, _a8e)
''' % CAPABILITY_ID


def check(path: pathlib.Path) -> int:
    src = path.read_text()
    print(f"{path}")
    print(f"  marker present : {MARKER in src}")
    print(f"  anchor found   : {src.count(ANCHOR)} occurrence(s) (need exactly 1)")
    return 0 if src.count(ANCHOR) == 1 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=pathlib.Path,
                    default=pathlib.Path("/path/to/isolated-workspace/anchoropt/memory_evaluator.py"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    if a.emit:
        sys.stdout.write(BLOCK)
        return 0
    if "/path/to/remote-checkout/" in str(a.target):
        raise SystemExit("REFUSING: that is the SHARED tree. Isolated copies only.")
    if not a.target.exists():
        raise SystemExit(f"target not found: {a.target}")
    if a.check:
        return check(a.target)
    if a.apply:
        src = a.target.read_text()
        if MARKER in src:
            print("already applied (idempotent)")
            return 0
        if src.count(ANCHOR) != 1:
            raise SystemExit(f"anchor found {src.count(ANCHOR)} times, need exactly 1 -- "
                             f"the evaluator's shape changed; re-read it before patching")
        a.target.with_suffix(".py.bak_prededuplclear").write_text(src)
        a.target.write_text(src.replace(ANCHOR, BLOCK + ANCHOR, 1))
        compile(a.target.read_text(), str(a.target), "exec")
        print(f"applied; backup at {a.target.with_suffix('.py.bak_prededuplclear')}")
        return 0
    print("nothing to do; pass --check, --emit or --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
