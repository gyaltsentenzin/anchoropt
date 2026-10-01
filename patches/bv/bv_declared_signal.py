"""Teach the BV installer to evaluate a `declared_signal` predicate."""
import pathlib
for f in ("scripts/install_controller.py", "install_controller.py"):
    p = pathlib.Path(f)
    if not p.exists():
        print(f, "absent"); continue
    s = p.read_text()
    if "declared_signal" in s:
        print(f, "already patched"); continue
    old = '''    @staticmethod
    def _check(node):
        if "all" in node:'''
    new = '''    @staticmethod
    def _check(node):
        # A DECLARED SIGNAL: the host ships the predicate, so the spec names it rather than
        # describing an atom. This is what a Phi seeded from the residual family's grounded
        # observables emits -- those signals are the host's own, not expansion products.
        if "declared_signal" in node:
            if not str(node["declared_signal"] or "").strip():
                raise ValueError("`declared_signal` must name a signal")
            return
        if "all" in node:'''
    assert s.count(old) == 1, (f, s.count(old))
    s = s.replace(old, new)
    old2 = '''    def _eval(self, node, state):
        if "all" in node:'''
    new2 = '''    def _eval(self, node, state):
        if "declared_signal" in node:
            # Evaluated BY THE HOST, by name, from the same module the run imports -- a second
            # implementation here is how an offline-validated predicate stops matching the one
            # that actually runs.
            from anchoropt.memory_gates import evaluate_signal as _es
            return bool(_es(str(node["declared_signal"]), dict(state or {}), {}))
        if "all" in node:'''
    assert s.count(old2) == 1, (f, "eval", s.count(old2))
    s = s.replace(old2, new2)
    p.write_text(s)
    print(f, "patched")
