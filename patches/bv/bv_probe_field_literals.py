#!/usr/bin/env python3
"""The probe assigned ONE shared string literal to EVERY string field. A conjunction over two fields
was therefore unsatisfiable, and all six R11 arms died.

WHAT HAPPENED
-------------
`probe_install_discriminate.py` collected every string literal the predicate compares against --
`literals = re.findall(r'==\s*["\']([a-z_]+)["\']', src)` -- and threw away WHICH FIELD each belonged
to, then set every string field to the same value.

A5's predicate needs BOTH:

    state.get("error_kind") != "no_capacity"   -> return False
    state.get("container")  == "archival"

so it can only fire when `error_kind == "no_capacity"` AND `container == "archival"`. With one shared
literal (`'archival'`, the only one the regex found) the probe set `error_kind = "archival"`, the predicate
answered False for every flip, and it reported NOT DISCRIMINATING. Six arms, ~4 seconds each.

Verified independently before and after: A5 fires on archival-full and on `is_relocation_target`, and does
NOT fire on core-full (A1 keeps that), on blob overflow (A7's trigger), or with no error.

THE FIX
-------
Capture the PAIRING -- field -> the literals that field is compared against -- and give each string field
its own value, falling back to the shared pool only where the source did not pair them. Also: any field
the source pairs with a string literal IS a string field. `container` is neither `*_kind` nor `result`, so
it had been classed "boolean" and flipped True/False.

Regression-checked on all eight live specs, every one still discriminating: a5, a9, kvreloc, a4_vector,
a7_recsum, a8_v2, r1_clear_proposed_at_capacity, r1_redundant_proposed_write.

THE PATTERN, now four deep in this guard alone
----------------------------------------------
Each fix widened what the probe can EXPRESS, and each gap presented identically -- as the candidate being
defective:

  1. signal name taken from `spec["name"]`          -> "cannot derive the predicate's fields"
  2. numeric thresholds flipped as booleans          -> NOT DISCRIMINATING   (A9)
  3. candidate taken as `installed(locus)[0]`        -> NOT DISCRIMINATING   (A9, masking #2)
  4. one shared literal across all string fields     -> NOT DISCRIMINATING   (A5)

**A guard that cannot express a candidate's contrast reports the candidate as broken.** Failing closed is
right, but it makes the guard a silent ceiling on which KINDS of controller can ever be evaluated --
threshold signals, then multi-field conjunctions. Test the guard whenever a new signal SHAPE appears, not
just a new signal.

Backup at probe_install_discriminate.py.bak_prepairs.
"""

import pathlib
p = pathlib.Path("/path/to/isolated-workspace/probe_install_discriminate.py")
s = p.read_text()

# PER-FIELD LITERALS. Line 79 collected every string literal the predicate compares against but threw
# away WHICH FIELD each belonged to, then assigned the same value to every string field. A5's predicate
# needs error_kind == "no_capacity" AND container == "archival" simultaneously, so a single shared value
# can never satisfy it and the probe reported NOT DISCRIMINATING -- killing all six R11 arms for a
# predicate that discriminates perfectly (verified: fires on archival-full and on is_relocation_target,
# not on core-full, not on blob overflow, not without an error).
old = '''        literals = sorted(set(re.findall(r'==\\s*["\\']([a-z_]+)["\\']', src)))'''
new = '''        literals = sorted(set(re.findall(r'==\\s*["\\']([a-z_]+)["\\']', src)))
        # AND the field each literal is compared against, so a multi-field conjunction can be
        # satisfied. `state.get("error_kind") != "no_capacity"` and `state.get("container") ==
        # "archival"` are different fields needing different values; one shared literal satisfies
        # neither predicate that needs both.
        for _m in re.finditer(
                r'state\\.get\\(\\s*["\\']([A-Za-z_][A-Za-z0-9_]*)["\\']\\s*\\)\\s*[!=]=\\s*["\\']([a-z_]+)["\\']',
                src):
            field_literals.setdefault(_m.group(1), set()).add(_m.group(2))'''
assert s.count(old) == 1, f"literal anchor {s.count(old)}"
s = s.replace(old, new, 1)

s = s.replace('''fields: set = set()
literals: list = []''', '''fields: set = set()
literals: list = []
#: field -> the literals the predicate compares THAT field against. A shared pool loses the pairing.
field_literals: dict = {}''')

old2 = '''        for f in str_fields:
                base[f] = sval'''
new2 = '''        for f in str_fields:
                # Prefer the literal this FIELD is compared against; fall back to the shared value
                # only where the predicate source did not pair them.
                own = sorted(field_literals.get(f, ()))
                base[f] = own[0] if own else sval'''
if old2 in s:
    s = s.replace(old2, new2, 1)
else:
    # indentation differs; match the loop body directly
    old3 = """            for f in str_fields:
                base[f] = sval"""
    assert s.count(old3) == 1, f"str_fields loop anchor {s.count(old3)}"
    s = s.replace(old3, """            for f in str_fields:
                # Prefer the literal this FIELD is compared against; fall back to the shared value
                # only where the predicate source did not pair them.
                _own = sorted(field_literals.get(f, ()))
                base[f] = _own[0] if _own else sval""", 1)

# A5 reads `container`, which is not *_kind and not `result`, so it was classed boolean. Any field the
# source pairs with a string literal IS a string field.
old4 = '''str_fields = [f for f in sorted(fields) if f.endswith("_kind") or f == "result"]'''
new4 = '''str_fields = [f for f in sorted(fields)
              if f.endswith("_kind") or f == "result" or f in field_literals]'''
assert s.count(old4) == 1
s = s.replace(old4, new4, 1)
p.write_text(s)
compile(p.read_text(), str(p), "exec")
print("probe now pairs each string field with ITS OWN literals")
