import pathlib
for f in ("scripts/install_controller.py", "install_controller.py"):
    p = pathlib.Path(f); s = p.read_text()
    old = "            from anchoropt.memory_gates import evaluate_signal as _es"
    new = ("            # The host's OWN declared-signal implementations, copied verbatim from the\n"
           "            # adapter's bfcl_signals. One definition, evaluated by both the optimizer and\n"
           "            # the runner -- a second implementation here is how the two drift apart.\n"
           "            from anchoropt.bfcl_declared_signals import evaluate_signal as _es")
    if "bfcl_declared_signals" in s:
        print(f, "already points at the signal module"); continue
    assert s.count(old) == 1, (f, s.count(old))
    p.write_text(s.replace(old, new)); print(f, "repointed")
