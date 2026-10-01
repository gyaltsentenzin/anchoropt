"""Supply episode-scoped capacity evidence at the commitment gate, live.

`clear_proposed_at_capacity` reads `container_full`. The gate state was built per proposed call and
carried neither it nor `error_kind`, so the signal was UNSATISFIABLE at runtime while passing every
synthetic test that supplied the field. The evidence exists -- a capacity error appears in an earlier
step's tool result in 23 of 27 real storage episodes -- it was simply never carried to the boundary.

Accumulated per EPISODE, keyed on test_entry_id, for the same reason the upstream retry budget is:
a module-level flag leaks across every episode in the run.
"""
import pathlib, re

p = pathlib.Path("anchoropt/memory_evaluator.py")
s = p.read_text()

# IDEMPOTENCE. The anchor this inserts BEFORE survives the insert, so a second run found it again and
# applied twice -- observed: the capacity block was duplicated in a live evaluator while arm jobs were
# running. Behaviourally a no-op (it recomputes the same value from the same inputs), but running code
# nobody intended is how a real defect hides. Guard on the inserted MARKER, never on the anchor.
if "EPISODE-SCOPED CAPACITY EVIDENCE" in s:
    print("already patched")
    raise SystemExit(0)


# 1. A helper that recognises a capacity limit in a tool result, using the same official strings
#    the on_domain_error_core_full registry spec matches.
helper = '''
_CAPACITY_MARKERS = ("is full", "exceeds maximum size", "at its entry limit", "no capacity")


def _capacity_seen(results) -> bool:
    """Did any tool result report a store at its limit? The official substrings, not a guess."""
    for r in (results or []):
        low = str(r).lower()
        for m in _CAPACITY_MARKERS:
            if m in low:
                return True
    return False

'''
anchor = "def _upstream_call_facts(call):"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor, 1)

# 2. Accumulate it per episode and put it on the gate state.
old = '''                        _ust = {"proposed_call": str(_uc_call), "call_index": _uc_i,
                                "phase": ("prereq" if str(rollout_tag) == "snap" else "query"),
                                "step_index": step_record.get("step"),
                                "n_proposed_calls": len(_mg_exec_calls or [])}'''
new = '''                        _ust = {"proposed_call": str(_uc_call), "call_index": _uc_i,
                                "phase": ("prereq" if str(rollout_tag) == "snap" else "query"),
                                "step_index": step_record.get("step"),
                                "n_proposed_calls": len(_mg_exec_calls or [])}
                        # EPISODE-SCOPED CAPACITY EVIDENCE. A limit a store reported at an earlier
                        # step is still true now, and this boundary is pre-dispatch so it cannot
                        # observe one itself. Without this the destructive-clear signal reads a field
                        # that is never present and answers False forever -- declared-vs-supplied,
                        # the same gap that made the payload-size predicate fire 0 times on an
                        # episode holding two over-cap writes.
                        _cf_store = getattr(_hook_state, "_capacity_seen", None)
                        if not isinstance(_cf_store, dict):
                            _cf_store = _hook_state._capacity_seen = {}
                        _cf_key = str(test_entry_id or "")
                        if _capacity_seen(execution_results):
                            _cf_store[_cf_key] = True
                        _ust["container_full"] = bool(_cf_store.get(_cf_key, False))'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
p.write_text(s)

import ast
ast.parse(s)
print("BV live hook patched: container_full supplied at the commitment gate, episode-scoped")
