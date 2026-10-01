#!/usr/bin/env python3
"""Install a controller and prove it DISCRIMINATES, generically over any signal.

Two defects this replaced, both of the same shape -- a probe that understands one mechanism blocks
every other one:

  * it built two payload-SIZE states, so a redundancy predicate (which reads different fields)
    reported NOT DISCRIMINATING and its arm was killed before GPU;
  * it hardcoded the commitment gate, so a post_execution controller crashed it with IndexError.

So: the locus comes from the spec, the fields come from the PREDICATE'S OWN SOURCE, and the contrast is
found by flipping one field. Both boolean polarities are tried, because some predicates want a field
True and others want it False. String-valued fields (`error_kind`) are swept over the literal values
that appear in the predicate source, because each signal matches a different label.
"""

from __future__ import annotations

import inspect
import json
import re
import sys

import install_controller  # noqa: F401  registers from ANCHOROPT_CONTROLLER_SPEC
from anchoropt.runtime_hook import installed

spec = json.load(open(sys.argv[1]))
name = spec["name"]
locus = str(spec.get("locus") or "post_generation_pre_exec")

inst = list(installed(locus))
if not inst:
    sys.exit("INSTALL FAILED: no controller registered at %s" % locus)
c = inst[0]

# ---- fields AND their candidate literal values, from the predicate's own source ----------------
fields: set = set()
literals: list = []
try:
    try:
        from anchoropt import bfcl_declared_signals as sg
    except Exception:
        import bfcl_declared_signals as sg
    fn = (getattr(sg, "SIGNALS", {}) or {}).get(name)
    if fn is not None:
        src = inspect.getsource(fn)
        fields |= set(re.findall(r'state\.get\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']', src))
        fields |= set(re.findall(r'state\[\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*\]', src))
        # every string literal the predicate compares against
        literals = sorted(set(re.findall(r'==\s*["\']([a-z_]+)["\']', src)))
except Exception as exc:
    print("  (could not read predicate source: %s)" % exc)

print("fields the predicate reads:", sorted(fields) or "UNKNOWN")
if not fields:
    sys.exit("cannot derive the predicate's fields; refusing to guess a contrast")
if literals:
    print("string literals it compares against:", literals)

BASE = {"boundary": locus, "phase": str(spec.get("phase") or "prereq"), "call_index": 0,
        "n_proposed_calls": 1, "container_full": False, "container": "core", "step_index": 3,
        "proposed_call": "core_memory_add(key='k',value='v')", "proposed_payload_chars": 200}

KIND_FALLBACK = ["no_capacity", "blob_would_overflow", "not_found", "duplicate_identifier",
                 "entry_too_long", "other"]


def flip(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return 0 if v else 1
    if isinstance(v, str):
        return None
    return "x"


str_fields = [f for f in sorted(fields) if f.endswith("_kind") or f == "result"]
bool_fields = [f for f in sorted(fields) if f not in str_fields]
values = literals or KIND_FALLBACK

found = None
for sval in (values if str_fields else [None]):
    for polarity in (True, False):
        base = dict(BASE)
        for f in bool_fields:
            base[f] = polarity
        for f in str_fields:
            base[f] = sval
        a = bool(c.fires_on(base))
        for f in sorted(fields):
            probe = dict(base)
            probe[f] = flip(base.get(f))
            if bool(c.fires_on(probe)) != a:
                found = (f, a, bool(c.fires_on(probe)), polarity, sval)
                break
        if found:
            break
    if found:
        break

if not found:
    sys.exit("NOT DISCRIMINATING: no single-field flip, over polarities %s and string values %s, "
             "changes the verdict for fields %s" % ([True, False], values, sorted(fields)))

f, a, b, pol, sval = found
print("discriminates on %r (bool base %s, string base %r): %s -> %s" % (f, pol, sval, a, b))
if spec.get("phase") not in ("query", "prereq", "any"):
    sys.exit("PHASE: spec declares %r" % spec.get("phase"))
print("[iso] installed=%s at %s phase=%s" % (getattr(c, "name", "?")[:44], locus, spec.get("phase")))
