import pathlib, sys
targets = sys.argv[1:] or ["scripts/install_controller.py"]
old = '''            return bool(_es(str(node["declared_signal"]), dict(state or {}), {}))'''
new = '''            # PARAMS TRAVEL WITH THE PREDICATE. Evaluating a parameterized signal with `{}` raises,
            # and "a raising phi is not a firing" then makes the controller silently never fire --
            # an installed arm that runs as the control and reports a measured zero.
            return bool(_es(str(node["declared_signal"]), dict(state or {}),
                            dict(node.get("params") or {})))'''
for f in targets:
    p = pathlib.Path(f)
    if not p.exists():
        print(f, "absent"); continue
    s = p.read_text()
    if 'node.get("params")' in s:
        print(f, "already patched"); continue
    if s.count(old) != 1:
        print(f, f"anchor count={s.count(old)} -- SKIPPED"); continue
    p.write_text(s.replace(old, new)); print(f, "patched")
