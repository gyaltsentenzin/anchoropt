"""Do two reprompt candidates with DIFFERENT instructions reach the model differently?

The invariant: each candidate's actual policy parameters must reach the executor before evaluation.
The reprompt executor reads its text from templates.on_turn_start_action (the policy file), which is a
legitimate delivery channel -- but only if the runner WRITES each candidate's instruction there.

If it does not, two nominally different arms execute identically, which is the retry_semantics defect
in another cell.
"""
import json, os, sys, pathlib
R = "/path/to/remote-checkout"
sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
from anchoropt.memory_gates import remedy_enabled

BASE = json.load(open(f"{R}/policies/templates_null.json"))
A = "Search the other memory container before answering."
B = "State plainly that the information is absent."

def policy_for(instruction, budget=1):
    """What a runner must produce for a reprompt candidate: the eta written into its eta_slot."""
    pol = json.loads(json.dumps(BASE))
    pol.setdefault("templates", {})["on_turn_start_action"] = instruction
    pol["enable_zero_call_reprompt"] = True
    return pol

bad = 0
pols = {}
for label, instr in (("A", A), ("B", B)):
    pol = policy_for(instr)
    p = f"/tmp/pol_{label}.json"; json.dump(pol, open(p, "w"))
    pols[label] = pol
    armed = remedy_enabled(pol, "enable_zero_call_reprompt")
    got = (pol.get("templates") or {}).get("on_turn_start_action")
    ok = armed is True and got == instr
    bad += (not ok)
    print(f"  candidate {label}: armed={armed}  eta_slot text={got[:44]!r}  {'ok' if ok else 'FAIL'}")

print()
distinct = pols["A"]["templates"]["on_turn_start_action"] != pols["B"]["templates"]["on_turn_start_action"]
print(f"  the two policies deliver DISTINCT instructions: {distinct}")
bad += (not distinct)

# Does the live executor read exactly that slot?
src = pathlib.Path(f"{R}/anchoropt/memory_evaluator.py").read_text()
i = src.index('remedy_enabled(templates, "enable_zero_call_reprompt")')
w = src[i:i+1200]
reads_slot = '"on_turn_start_action"' in w
injects = "_add_next_turn_user_message_prompting" in w
print(f"  the live executor reads on_turn_start_action: {reads_slot}")
print(f"  and injects it into the conversation        : {injects}")
bad += (not reads_slot) + (not injects)

# retry_budget: is it delivered and ENFORCED anywhere on this path?
budget_read = "retry_budget" in src
zc_latch = "_zero_call_reprompted" in w
print(f"  retry_budget appears in the evaluator       : {budget_read}")
print(f"  the zero-call path bounds itself by a latch : {zc_latch} (_zero_call_reprompted)")

print()
print("DELIVERY_OK" if not bad else f"{bad} PROBLEM(S)")
