import json, os, sys
R = "/path/to/remote-checkout"
sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
spec_path = R + "/rounds/H0_R1/kv_spec.json"
os.environ["ANCHOROPT_CONTROLLER_SPEC"] = spec_path
import install_controller as ic
spec = json.load(open(spec_path))
p = ic.SpecPredicate(spec)
print("installed:", p.name, "| phase:", p.phase, "| eta:", p.eta)
G = "post_generation_pre_exec"
tests = [
    ("clear + prior capacity error", {"proposes_clear": True, "container_full": True}, True),
    ("clear + error_kind no_capacity", {"proposes_clear": True, "container_full": False,
                                        "error_kind": "no_capacity"}, True),
    ("clear, nothing observed yet", {"proposes_clear": True, "container_full": False}, False),
    ("ordinary write at capacity", {"proposes_clear": False, "container_full": True}, False),
]
bad = 0
for label, st, want in tests:
    got = p.fires_on(dict(st, boundary=G))
    bad += (got is not want)
    print(f"  {label:32s} fires={got}  expected={want}  {'ok' if got is want else 'MISMATCH'}")

# Is the controller actually REGISTERED into the live dispatch path?
from anchoropt.runtime_hook import installed
reg = installed("post_generation_pre_exec")
print("registered controllers at the gate:", [getattr(c, "name", "?") for c in reg])
print("capability_id on the spec:", spec.get("capability_id"))
print("INSTALL_PROBE_OK" if not bad and reg else "PROBLEM")
