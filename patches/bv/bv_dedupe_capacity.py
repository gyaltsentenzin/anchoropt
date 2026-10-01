"""Remove a duplicated capacity block, and make the patch that created it truly idempotent.

`bv_capacity_fact.py` asserted its anchor appeared exactly once and then inserted -- so a second run
found the anchor again (the insert does not remove it) and applied a second time. The duplicate is
behaviourally a no-op (it recomputes the same value from the same inputs and writes the same key), but
running code nobody intended is how a real defect hides, so it is removed.
"""
import ast, pathlib

p = pathlib.Path("anchoropt/memory_evaluator.py")
s = p.read_text()
MARK = "                        # EPISODE-SCOPED CAPACITY EVIDENCE."
END = '                        _ust["container_full"] = bool(_cf_store.get(_cf_key, False))\n'

n = s.count(MARK)
if n <= 1:
    print(f"no duplicate ({n} block present)")
else:
    while s.count(MARK) > 1:
        i1 = s.index(MARK)
        i2 = s.index(MARK, i1 + 1)
        e2 = s.index(END, i2) + len(END)
        s = s[:i2] + s[e2:]
    p.write_text(s)
    ast.parse(s)
    print(f"removed {n - 1} duplicate block(s); {s.count(MARK)} remains")
