"""Fix: the gate hook read `execution_results`, which is UNBOUND on the prereq/snapshot path.

My capacity patch assumed a local the commitment-gate function does not have -- it was the ONLY
reference to it there. The controller then raised UnboundLocalError on every prereq episode, the
snapshot store failed to publish, and the arm job died at rc=1 while the control ran fine (only the
arm installs a controller).

The fix reads the PREVIOUS step's results from the trajectory the episode is already accumulating,
which is the same information `execution_results` would have carried and is available on both paths.
Still strictly backward-looking: `trajectory_history` holds completed steps only.
"""
import pathlib

p = pathlib.Path("anchoropt/memory_evaluator.py")
s = p.read_text()
if "for _prev in (trajectory_history or [])" in s:
    print("already patched")
    raise SystemExit(0)

old = """                        if _capacity_seen(execution_results):
                            _cf_store[_cf_key] = True"""
new = """                        # PREVIOUS steps only. `execution_results` is not a local on this path --
                        # reading it raised UnboundLocalError on every prereq episode and took the
                        # snapshot store down with it. `trajectory_history` holds COMPLETED steps, so
                        # it carries the same backward-looking information on both paths and cannot
                        # see the call being decided now.
                        for _prev in (trajectory_history or []):
                            if _capacity_seen(_prev.get("tool_results") or []):
                                _cf_store[_cf_key] = True
                                break"""
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, new))
import ast
ast.parse(p.read_text())
print("patched: reads trajectory_history (completed steps), not an unbound local")
