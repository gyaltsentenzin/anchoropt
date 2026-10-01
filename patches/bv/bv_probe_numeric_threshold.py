#!/usr/bin/env python3
"""The pre-GPU probe could not test a NUMERIC-THRESHOLD predicate, and killed all six A9 arms.

WHAT HAPPENED
-------------
`probe_install_discriminate.py` establishes that a predicate DISCRIMINATES by flipping one field and
requiring the verdict to change. It partitioned fields into string-valued (`*_kind`, `result`) and
"boolean" -- everything else. A9's predicate reads `best_similarity`, a FLOAT compared against 0.30, so
the probe set `best_similarity = True` and flipped to `False`. Neither crosses 0.30, so it reported:

    NOT DISCRIMINATING: no single-field flip, over polarities [True, False] and string values [...]

and exited 1. All six R10 arms died in ~4 seconds.

**The refusal was correct behaviour for an under-powered test; the test was what was wrong.** Verified
independently against the real signal on the real hook state: A9 fires at 0.12 and 0.299, and does NOT
fire at 0.30, 0.81, on a non-read, or with no score at all.

THE FIX
-------
A field the predicate compares numerically gets NUMERIC probe values that STRADDLE the threshold, and
the threshold is read from the SPEC'S OWN `predicate.params` -- so the pair brackets the value the arm
will actually run with rather than a guess. Straddling is the point: a single value proves nothing about
a comparison.

    numeric fields (threshold-compared): ['best_similarity'] | thresholds from the spec: [0.3]
    discriminates on 'best_similarity' (0.19999999999999998 -> 0.4): True -> False

Numeric fields are identified by name hint (similarity, score, ratio, fraction, chars, size, count, len)
and are tried BEFORE the boolean flips, because their contrast is the straddle rather than a negation.

REGRESSION-CHECKED on every existing spec, all still discriminating:
  spec_kvreloc          error_kind
  spec_a4_vector        has_generation
  spec_a7_recsum        error_kind
  spec_a8_v2            proposes_clear
  r1_clear_proposed...  proposes_clear

THE TRANSFERABLE POINT: a guard that cannot express a candidate's contrast will report the candidate as
defective. That is the safe direction to fail, but it makes the guard a silent ceiling on what kinds of
controller can ever be evaluated -- here, every threshold-based structural signal. Worth checking
whenever a new signal TYPE is introduced, not just a new signal.

Backup at probe_install_discriminate.py.bak_prenumeric.
"""

import pathlib
p = pathlib.Path("/path/to/isolated-workspace/probe_install_discriminate.py")
s = p.read_text()

old = '''str_fields = [f for f in sorted(fields) if f.endswith("_kind") or f == "result"]
bool_fields = [f for f in sorted(fields) if f not in str_fields]
values = literals or KIND_FALLBACK'''
new = '''str_fields = [f for f in sorted(fields) if f.endswith("_kind") or f == "result"]

# NUMERIC-THRESHOLD FIELDS. A predicate that compares a SCORE against a threshold cannot be tested by
# boolean or string flipping: setting best_similarity=True and flipping to False never crosses 0.30, so
# the probe reported NOT DISCRIMINATING and killed all six R10 arms before GPU -- for a predicate that
# discriminates perfectly (verified separately: fires at 0.12 and 0.299, not at 0.30 or 0.81).
#
# The refusal was RIGHT for an under-powered test and the test was the thing that was wrong. A field the
# predicate reads numerically gets numeric probe values, taken from the SPEC'S OWN params where possible
# so the pair straddles the threshold the arm will actually run with, rather than a guess.
_params = dict((spec.get("predicate") or {}).get("params") or {})
_thresholds = [float(v) for v in _params.values()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
NUMERIC_HINTS = ("similarity", "score", "ratio", "fraction", "chars", "size", "count", "len")
num_fields = [f for f in sorted(fields)
              if f not in str_fields and any(h in f for h in NUMERIC_HINTS)]
bool_fields = [f for f in sorted(fields) if f not in str_fields and f not in num_fields]
values = literals or KIND_FALLBACK

if num_fields:
    print("numeric fields (threshold-compared):", num_fields,
          "| thresholds from the spec:", _thresholds or "(none declared)")'''
assert s.count(old) == 1, "value-partition anchor"
s = s.replace(old, new, 1)

old2 = '''found = None
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
        break'''
new2 = '''# For a numeric field, probe a value BELOW and a value ABOVE each declared threshold. Straddling is
# what makes the contrast meaningful; a single value proves nothing about a comparison.
def _numeric_pairs(thresholds):
    if not thresholds:
        return [(0.0, 1.0)]
    out = []
    for t in thresholds:
        out.append((max(t - 0.1, 0.0) if t > 0.1 else t / 2.0, t + 0.1))
    return out


found = None
for sval in (values if str_fields else [None]):
    for polarity in (True, False):
        for lo, hi in (_numeric_pairs(_thresholds) if num_fields else [(None, None)]):
            base = dict(BASE)
            for f in bool_fields:
                base[f] = polarity
            for f in str_fields:
                base[f] = sval
            for f in num_fields:
                base[f] = lo
            a = bool(c.fires_on(base))
            # numeric fields first: their contrast is the straddle, not a flip()
            for f in num_fields:
                probe = dict(base)
                probe[f] = hi
                if bool(c.fires_on(probe)) != a:
                    found = (f, a, bool(c.fires_on(probe)), polarity, "%s->%s" % (lo, hi))
                    break
            if not found:
                for f in sorted(fields):
                    if f in num_fields:
                        continue
                    probe = dict(base)
                    probe[f] = flip(base.get(f))
                    if bool(c.fires_on(probe)) != a:
                        found = (f, a, bool(c.fires_on(probe)), polarity, sval)
                        break
            if found:
                break
        if found:
            break
    if found:
        break'''
assert s.count(old2) == 1, "discrimination loop anchor"
s = s.replace(old2, new2, 1)
p.write_text(s)
compile(p.read_text(), str(p), "exec")
print("probe now tests NUMERIC thresholds by straddling the spec's own declared value")
