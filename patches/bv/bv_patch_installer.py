import pathlib
old = '    "eq": lambda v, x: v == x,'
new = '''    "eq": lambda v, x: v == x,
    # signal_grammar.py (the structured-search path) spells equality "equals"; signal_lang.py
    # spells it "eq". Both are core's own vocabulary and this file evaluates whatever core
    # emits -- 6 of the 219 arms in the first live round carried "equals" and raised
    # `unknown op` at install time, so those arms could not be evaluated at all.
    "equals": lambda v, x: v == x,'''
for f in ("scripts/install_controller.py", "install_controller.py"):
    p = pathlib.Path(f)
    s = p.read_text()
    if '"equals"' in s:
        print(f, "already patched"); continue
    assert s.count(old) == 1, (f, s.count(old))
    p.write_text(s.replace(old, new, 1))
    print(f, "patched")
