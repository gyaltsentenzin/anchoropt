#!/usr/bin/env python3
"""THE DEFECT THAT VOIDED R4: `ANCHOROPT_PHI_MODULE=install_controller` loaded the SHARED loader.

WHAT HAPPENED, measured
-----------------------
`run_memory_eval.py` lives in the SHARED tree and is invoked BY PATH:

    python "$R/scripts/run_memory_eval.py" ...

Python puts a script's own directory at `sys.path[0]`, AHEAD of everything in PYTHONPATH. So when that
script does `importlib.import_module("install_controller")`, it resolves to
`$R/scripts/install_controller.py` -- the shared loader -- no matter that PYTHONPATH begins with the
isolated tree. The shared loader knows nothing of `ANCHOROPT_CONTROLLER_STACK`.

The symptom was ASYMMETRIC, which is what made it dangerous:

  * `probe_install_discriminate.py` lives in `$W`, so when the runner ran it (SPEC branch only) its
    `sys.path[0]` WAS `$W` and it imported the isolated loader -- printing
    "STACK of 1 installed" and "COMPOSITION n=2" into the log. Those lines are the PROBE.
  * The real eval process then imported the SHARED loader and installed only the candidate.
  * The CONTROL arm had no SPEC, so it never ran the probe, and printed the shared loader's
    "ANCHOROPT_CONTROLLER_SPEC unset -- nothing installed".

So the log APPEARED to show a 2-controller stack while the measurement ran with one controller (arms)
or none (control). Confirmed from the trajectories, not the log: ZERO `relocate_*` gates in either arm.
Jobs 1840434 / 1840435 / 1840498 are therefore INVALID as composed-stack measurements and are recorded
as such. This is exactly the "in the library is not the same as executing in the evaluator" failure, and
it was caught by `score_round.py`'s execution-evidence census rather than by reading accuracy.

THE FIX
-------
Name a module that exists ONLY in the isolated tree. `install_controller_stack.py` is the same grafted
loader under a distinct name, so import resolution cannot silently prefer a shared namesake:

    install_controller       -> /path/to/remote-checkout/scripts/install_controller.py  (shared)
    install_controller_stack -> /path/to/isolated-workspace/install_controller_stack.py                    (isolated)

Verified with `importlib.util.find_spec` after inserting the shared scripts dir at sys.path[0] -- the
exact condition that defeated the previous attempt.

GENERAL RULE THIS EARNS
-----------------------
A patch delivered by SHADOWING a module name is only as reliable as the import order, and import order
is decided by whoever invokes the interpreter. Deliver by a DISTINCT NAME instead, and assert the
resolved path from inside the process that will do the measuring.
"""

import pathlib
p = pathlib.Path("/path/to/isolated-workspace/runner_auto.sh")
s = p.read_text()
# THE FIX: PHI_MODULE must name a module that EXISTS ONLY in the isolated tree. run_memory_eval.py
# lives in the SHARED tree, and Python puts a script's own directory at sys.path[0] ahead of
# PYTHONPATH -- so `install_controller` always resolved to the SHARED copy, which knows nothing of
# _STACK. Measured: the eval process installed only the candidate and zero relocate_* gates fired.
s = s.replace("export ANCHOROPT_PHI_MODULE=install_controller\n",
              "export ANCHOROPT_PHI_MODULE=install_controller_stack\n")
s = s.replace("(PHI_MODULE=install_controller)", "(PHI_MODULE=install_controller_stack)")
p.write_text(s)
print("runner occurrences of install_controller_stack:", s.count("install_controller_stack"))

q = pathlib.Path("/path/to/isolated-workspace/probe_install_discriminate.py")
t = q.read_text().replace("import install_controller  # noqa: F401",
                          "import install_controller_stack  # noqa: F401")
q.write_text(t)
print("probe now imports:", "install_controller_stack" in t)
