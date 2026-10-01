import sys
sys.path.insert(0, "/path/to/remote-checkout")
from anchoropt.bfcl_declared_signals import evaluate_signal as ev, declared_signals
print("declared on BV:", len(declared_signals()), "signals ->", declared_signals())
G = "post_generation_pre_exec"
SIG = "clear_proposed_at_capacity"
cases = [
    ("clear + container_full",          {"proposes_clear": True,  "container_full": True},  True),
    ("clear + error_kind no_capacity",  {"proposes_clear": True,  "container_full": False,
                                         "error_kind": "no_capacity"},                      True),
    ("clear, no limit at all",          {"proposes_clear": True,  "container_full": False}, False),
    ("write + container_full",          {"proposes_clear": False, "container_full": True},  False),
]
bad = 0
for label, st, want in cases:
    got = ev(SIG, dict(st, boundary=G), {})
    ok = (got is want)
    bad += (not ok)
    print(f"  {label:34s} -> {got}   expected {want}   {'ok' if ok else 'MISMATCH'}")
print("ALL_OK" if not bad else f"{bad} MISMATCHES")
