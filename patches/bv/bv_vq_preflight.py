"""Does the executable behaviour match the candidate spec, BEFORE any GPU time?

The post_execution/reprompt executor reads its text from templates.on_low_similarity_reprompt and is
gated by remedy_enabled(enable_low_similarity_reprompt). So a runner must (a) arm that remedy and
(b) write THIS candidate's instruction into that slot -- otherwise two distinct arms run identically,
or neither fires at all.
"""
import json, os, sys, pathlib
R = "/path/to/remote-checkout"
sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
from anchoropt.memory_gates import remedy_enabled

src = pathlib.Path(f"{R}/anchoropt/memory_evaluator.py").read_text()
SLOT = "on_low_similarity_reprompt"
FLAG = "enable_low_similarity_reprompt"
print(f"executor reads {SLOT}        : {SLOT in src}")
print(f"executor gated by {FLAG}: {FLAG in src}")
i = src.find(FLAG)
w = src[i:i+1400] if i >= 0 else ""
print(f"and it INJECTS (adds a message): {'_add_next_turn_user_message_prompting' in w or '_add_message' in w}")
print()

BASE = json.load(open(f"{R}/policies/templates_null.json"))
bad = 0
pols = {}
for tag in ("vq1", "vq2"):
    spec = json.load(open(f"{R}/rounds/H0_Q1/{tag}.json"))
    instr = spec["eta"]["instruction"]
    pol = json.loads(json.dumps(BASE))
    pol[FLAG] = True
    pol.setdefault("templates", {})[SLOT] = instr
    p = f"{R}/rounds/H0_Q1/policy_{tag}.json"
    json.dump(pol, open(p, "w"), indent=1)
    pols[tag] = pol
    armed = remedy_enabled(pol, FLAG)
    got = pol["templates"][SLOT]
    ok = armed is True and got == instr
    bad += (not ok)
    print(f"{tag}: armed={armed} slot_text={got[:52]!r} {'ok' if ok else 'FAIL'}")
    # and the predicate must install
    os.environ["ANCHOROPT_CONTROLLER_SPEC"] = f"{R}/rounds/H0_Q1/{tag}.json"
    sys.modules.pop("install_controller", None)
    import install_controller as ic
    pred = ic.SpecPredicate(spec)
    st = {"boundary": "post_execution", "proposes_read": True, "best_similarity": 0.15}
    fires = pred.fires_on(st)
    st2 = dict(st, best_similarity=0.95)
    print(f"      predicate: fires@0.15={fires}  fires@0.95={pred.fires_on(st2)}  phase={pred.phase}")
    bad += (pred.phase != "query")

distinct = pols["vq1"]["templates"][SLOT] != pols["vq2"]["templates"][SLOT]
print()
print(f"the two policies deliver DISTINCT instructions: {distinct}")
bad += (not distinct)
print("PREFLIGHT_OK" if not bad else f"{bad} PROBLEM(S)")
