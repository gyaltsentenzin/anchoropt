"""Are the four reroute arms EXECUTABLE on BV, and do the two variants differ?"""
import json, os, sys
R = "/path/to/remote-checkout"
sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")

ARMS = [
    ("container_at_capacity", "archival_memory_add+replace_original", "replace_original"),
    ("container_at_capacity", "archival_memory_add+retry_after", "retry_after"),
    ("container_slots_exhausted", "archival_memory_add+replace_original", "replace_original"),
    ("container_slots_exhausted", "archival_memory_add+retry_after", "retry_after"),
]
for sig, variant, sem in ARMS:
    spec = {"name": sig, "locus": "post_execution", "action": "reroute", "operator": "substitute",
            "variant": variant, "phase": "query",
            "eta": {"destination": "archival_memory_add",
                    "argument_mapping": "{'text': 'text'}", "retry_semantics": sem},
            "predicate": {"declared_signal": sig}, "capability_id": "post_execution/reroute"}
    p = f"/tmp/probe_{sig}_{sem}.json"
    json.dump(spec, open(p, "w"))
    os.environ["ANCHOROPT_CONTROLLER_SPEC"] = p
    sys.modules.pop("install_controller", None)
    import install_controller as ic
    pred = ic.SpecPredicate(spec)
    eta = dict(pred.eta)
    prim = eta.get("primitive", "<ABSENT>")
    print(f"{sig:26s} {sem:16s} installs OK | eta keys={sorted(eta)} | primitive={prim}")
print()
print("The live post_execution hook requires eta['primitive'] == 'additional_read_and_merge'.")
print("None of these arms carries a `primitive` key, so each would report fired=False and run")
print("as the CONTROL. And `retry_semantics` is read nowhere, so the two variants are identical.")
