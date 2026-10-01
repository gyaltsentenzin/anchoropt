"""Replay the evaluator's OWN withholding loop against real episodes, in the isolated runtime.

This is the difference between "the predicate returns True" and "the intervention happens": the
evaluator partitions each step's proposed calls into withheld/kept, and only the withheld ones are
never dispatched. Verifying the partition is verifying the mechanism.
"""
import glob
import json

import install_controller  # noqa: F401  installs from ANCHOROPT_CONTROLLER_SPEC
from anchoropt.runtime_hook import installed as _uinst

EVAL = "/path/to/isolated-workspace/anchoropt/memory_evaluator.py"
src = open(EVAL).read()
i = src.find("def _upstream_call_facts(")
j = src.find("\ndef ", i + 10)
ns = {}
exec(compile(src[i:j], "facts", "exec"), ns)
facts = ns["_upstream_call_facts"]

R = "/path/to/remote-checkout/results/nativectl_rec_sum_train/run/traj/prereq"
ctls = list(_uinst("post_generation_pre_exec"))
print("controllers at the gate:", [getattr(c, "name", "?")[:44] for c in ctls])
print("evaluator:", EVAL, "patched:", "supply_error_kind" in src)

tot_w = tot_k = eps = 0
per_ep = []
for p in sorted(glob.glob(R + "/*.json")):
    ep = json.load(open(p))
    w = k = 0
    for s in ep.get("steps") or []:
        calls = [str(c) for c in (s.get("decoded") or [])]
        if not calls:
            continue
        for idx, call in enumerate(calls):
            st = {"proposed_call": call, "call_index": idx, "n_proposed_calls": len(calls),
                  "step_index": s.get("step"), "phase": "prereq", "container_full": False}
            try:
                st.update(facts(call) or {})
            except Exception:
                pass
            if any(c.fires_on(st) for c in ctls):
                w += 1
            else:
                k += 1
    if w or k:
        eps += 1
        tot_w += w
        tot_k += k
        per_ep.append((ep.get("case_id"), w, k))

print("episodes=%d  WITHHELD=%d  KEPT=%d  (rate %.1f%%)"
      % (eps, tot_w, tot_k, 100 * tot_w / max(tot_w + tot_k, 1)))
print()
print("per-episode (first 8):")
for cid, w, k in per_ep[:8]:
    print("   %-44s withheld %3d  kept %3d" % (str(cid)[:44], w, k))
print()
print("episodes where EVERY call is withheld (would empty the store):",
      sum(1 for _c, w, k in per_ep if k == 0))
print("episodes where NOTHING is withheld (arm == control there):",
      sum(1 for _c, w, k in per_ep if w == 0))
