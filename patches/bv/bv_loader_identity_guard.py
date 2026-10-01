#!/usr/bin/env python3
"""Export ANCHOROPT_EXPECT_LOADER_UNDER in the runner, so a shadowed loader CRASHES.

The R4 round completed successfully, printed an accuracy, and had not installed the controller stack
(see bv_phi_module_isolation.py and rounds/AUTONOMY/R4_RESULT_INVALID.json). A wrong measurement that
completes is worse than a crash, because it enters the record.

`install_controller_stack.py` refuses to import when its own resolved path is not under the declared
tree. The runner declares `/path/to/isolated-workspace`. Unset means no assertion, so callers that do not opt in are
unaffected.

Applied together with bv_phi_module_isolation.py; both are idempotent.
"""

import pathlib
p = pathlib.Path("/path/to/isolated-workspace/runner_auto.sh")
s = p.read_text()
old = 'export ANCHOROPT_CAPTURE_CONSTRAINT_STATE=1'
new = '''# LOADER IDENTITY, asserted from inside the measuring process. The previous round completed
# successfully with an accuracy number while the stack had NOT installed, because
# PHI_MODULE=install_controller resolved to the SHARED tree's namesake (run_memory_eval.py is invoked
# by path, so its own directory precedes PYTHONPATH). A wrong measurement that completes is worse than
# a crash, so this makes it a crash.
export ANCHOROPT_EXPECT_LOADER_UNDER=/path/to/isolated-workspace
export ANCHOROPT_CAPTURE_CONSTRAINT_STATE=1'''
assert s.count(old) == 1, f"anchor count {s.count(old)}"
p.write_text(s.replace(old, new, 1))
print("guard exported in runner")
