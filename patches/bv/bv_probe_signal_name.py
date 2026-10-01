#!/usr/bin/env python3
"""The pre-GPU probe looked up the CONTROLLER's name in the host's SIGNALS table.

`probe_install_discriminate.py` read `spec["name"]` and looked it up in
`bfcl_declared_signals.SIGNALS` to derive the fields a predicate reads. That works only while a
controller happens to be NAMED AFTER ITS TRIGGER, which every arm so far was.

The first spec that named the MECHANISM instead -- `dedup_clear_lossless_recovery` over the signal
`clear_proposed_at_capacity` -- derived no fields, and the probe refused:

    fields the predicate reads: UNKNOWN
    cannot derive the predicate's fields; refusing to guess a contrast

**The refusal was correct behaviour and the lookup was wrong.** A probe that cannot see what a
predicate reads must not invent a contrast, so it stopped the arm before it reached a GPU -- which is
exactly what it is for. It cost two jobs (1857252/1857253, EXIT after 3s) and no measurement.

THE FIX: read the signal identity from where it actually lives -- the predicate's own
`declared_signal` -- falling back to `spec["signal"]` and only then to `spec["name"]`, so every older
spec behaves byte-identically. When the two differ the probe now prints both, because a controller
name and a signal name silently coinciding is what hid this.

Verified after the fix on the A8 spec:
    signal 'clear_proposed_at_capacity' (controller 'dedup_clear_lossless_recovery')
    fields the predicate reads: ['container_full', 'error_kind', 'proposes_clear']
    discriminates on 'proposes_clear' (bool base True, string base 'no_capacity'): True -> False

GENERAL LESSON, and it is the same one as the A8 naming collision: a NAME is not an IDENTITY. Two
names that agree by convention will eventually disagree, and the code that relied on the coincidence
fails at the first controller whose name describes its mechanism rather than its trigger.

Backup at probe_install_discriminate.py.bak_prename. Idempotent.
"""

import pathlib
p = pathlib.Path("/path/to/isolated-workspace/probe_install_discriminate.py")
s = p.read_text()
old = '''spec = json.load(open(sys.argv[1]))
name = spec["name"]'''
new = '''spec = json.load(open(sys.argv[1]))
# THE SIGNAL NAME IS NOT THE CONTROLLER NAME. This read `spec["name"]`, which is the
# CONTROLLER's name, and looked it up in the host's SIGNALS table. That works only while the two
# happen to coincide -- as they did for every arm so far, whose controllers were named after their
# trigger. The first spec that named the MECHANISM instead ("dedup_clear_lossless_recovery" over
# signal "clear_proposed_at_capacity") derived no fields and the probe refused the arm before it
# reached the GPU. The refusal itself was correct behaviour; the lookup was wrong.
#
# Prefer the predicate's own `declared_signal`, which is where the signal identity actually lives,
# and fall back to `signal` and then `name` so older specs behave exactly as before.
_pred = spec.get("predicate") or {}
name = (str(_pred.get("declared_signal") or "").strip()
        or str(spec.get("signal") or "").strip()
        or spec["name"])
if name != spec.get("name"):
    print("signal %r (controller %r)" % (name, spec.get("name")))'''
assert s.count(old) == 1, f"anchor count {s.count(old)}"
p.write_text(s.replace(old, new, 1))
compile(p.read_text(), str(p), "exec")
print("probe now reads the PREDICATE's declared signal, not the controller name")
