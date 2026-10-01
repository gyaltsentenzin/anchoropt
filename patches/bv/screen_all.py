"""Per-arm safety + engagement screen, through the isolated evaluator's own withholding loop."""
import glob
import json
import os
import sys

sys.path.insert(0, "/path/to/remote-checkout/scripts")
import install_controller as ic  # noqa: E402

EVAL = "/path/to/isolated-workspace/anchoropt/memory_evaluator.py"
src = open(EVAL).read()
i = src.find("def _upstream_call_facts(")
j = src.find("\ndef ", i + 10)
ns = {}
exec(compile(src[i:j], "facts", "exec"), ns)
facts = ns["_upstream_call_facts"]

R = "/path/to/remote-checkout/results/nativectl_rec_sum_train/run/traj/prereq"
EPS = sorted(glob.glob(R + "/*.json"))

print("%-56s %6s %6s %7s %7s %7s" % ("arm", "withh", "kept", "rate", "emptied", "inert_eps"))
print("-" * 98)
ok = []
for f in sorted(glob.glob("/path/to/isolated-workspace/specs/*.json")):
    spec = json.load(open(f))
    p = ic.SpecPredicate(spec)
    w = k = emptied = inert = 0
    for path in EPS:
        ep = json.load(open(path))
        ew = ek = 0
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
                if p.fires_on(st):
                    ew += 1
                else:
                    ek += 1
        if ew or ek:
            if ek == 0 and ew > 0:
                emptied += 1
            if ew == 0:
                inert += 1
        w += ew
        k += ek
    rate = w / max(w + k, 1)
    verdict = []
    if emptied > 0:
        verdict.append("EMPTIES %d ep(s)" % emptied)
    if rate > 0.34:
        verdict.append("withholds %.0f%%" % (100 * rate))
    print("%-56s %6d %6d %6.1f%% %7d %7d  %s"
          % (spec["name"][:56], w, k, 100 * rate, emptied, inert,
             "PASS" if not verdict else "FAIL: " + "; ".join(verdict)))
    if not verdict:
        ok.append(spec)
json.dump(ok, open("/path/to/isolated-workspace/passed.json", "w"), indent=2)
print()
print("PASSED the store-safety screen: %d of %d" % (len(ok), len(glob.glob("/path/to/isolated-workspace/specs/*.json"))))
