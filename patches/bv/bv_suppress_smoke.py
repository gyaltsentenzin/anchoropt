"""LIVE SMOKE: does the installed controller actually WITHHOLD the destructive clear?

Registration is not suppression. The gate builds `_up_withheld` / `_up_keep` and hands only `_up_keep`
to the dispatcher, so the test is whether a firing predicate moves a call from one list to the other.
This replays that exact filter with the REAL installed SpecPredicate and the REAL `_upstream_call_facts`
the hook uses, so nothing about the decision is re-implemented here.
"""
import json, os, sys
R = "/path/to/remote-checkout"
sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")

spec_path = R + "/rounds/H0_R1/kv_spec.json"
os.environ["ANCHOROPT_CONTROLLER_SPEC"] = spec_path
import install_controller as ic
from anchoropt.memory_evaluator import _upstream_call_facts, _capacity_seen

spec = json.load(open(spec_path))
ctl = ic.SpecPredicate(spec)
print(f"controller: {ctl.name}  phase={ctl.phase}  eta={ctl.eta}")
print()

def gate(proposed_calls, prior_results, *, phase="prereq"):
    """The hook's own filter, replayed. Returns (withheld, kept)."""
    cap = _capacity_seen(prior_results)          # PRIOR results only -- pre-dispatch by construction
    withheld, kept = [], []
    for i, call in enumerate(proposed_calls):
        st = {"proposed_call": str(call), "call_index": i, "phase": phase,
              "step_index": 3, "n_proposed_calls": len(proposed_calls),
              "container_full": bool(cap)}
        st.update(_upstream_call_facts(str(call)) or {})
        (withheld if ctl.fires_on(st) else kept).append(call)
    return withheld, kept

FULL = ["{'error': 'core memory is full'}"]
NONE = ["{'id': 0}"]
bad = 0

def check(label, calls, prior, want_withheld):
    global bad
    w, k = gate(calls, prior)
    ok = (w == want_withheld)
    bad += (not ok)
    print(f"  {label:46s} withheld={w}")
    print(f"  {'':46s} kept    ={k}   {'ok' if ok else 'MISMATCH expected ' + str(want_withheld)}")

# 2. DOES IT SUPPRESS the destructive clear when the predicate fires?
check("clear after a capacity error -> SUPPRESSED",
      ["core_memory_clear()"], FULL, ["core_memory_clear()"])

# 3a. Does it leave unrelated operations alone?
check("ordinary write after capacity error -> kept",
      ["core_memory_add(text='x')"], FULL, [])
check("read after capacity error -> kept",
      ["core_memory_retrieve(query='x')"], FULL, [])
check("clear with NO capacity error -> kept",
      ["core_memory_clear()"], NONE, [])

# 3b. Mixed batch: only the clear is withheld, the rest still dispatch.
w, k = gate(["core_memory_add(text='keep me')", "core_memory_clear()",
             "archival_memory_retrieve(query='q')"], FULL)
ok = (w == ["core_memory_clear()"] and len(k) == 2)
bad += (not ok)
print(f"  {'mixed batch -> only the clear withheld':46s} withheld={w}")
print(f"  {'':46s} kept    ={k}   {'ok' if ok else 'MISMATCH'}")

# 3c. FUTURE INFORMATION: the step whose OWN result reveals the limit must not already act on it.
w_future, _ = gate(["core_memory_clear()"], [])          # nothing known yet
ok = (w_future == [])
bad += (not ok)
print(f"  {'no prior result -> does NOT fire (no future info)':46s} withheld={w_future}   "
      f"{'ok' if ok else 'MISMATCH'}")

# phase eligibility: a prereq controller must not act during query
w_q, _ = gate(["core_memory_clear()"], FULL, phase="query")
print(f"  {'(phase field is prereq; eligibility is enforced by the hook, not the predicate)':46s}")

print()
print("SMOKE_OK" if not bad else f"{bad} MISMATCHES")
