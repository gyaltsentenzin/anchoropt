import pathlib, re
p = pathlib.Path("/path/to/isolated-workspace/anchoropt/traj_sidecar.py")
s = p.read_text()
m = re.search(r"_FAMILY_PREFIXES\s*=\s*\(", s)
assert m, "could not find _FAMILY_PREFIXES"
# Declare the new key BEFORE a GPU round: an undeclared key is dropped by the allowlist, and this
# project has lost intervention telemetry that way repeatedly (12 times by the runner's own count).
ins = m.end()
s = s[:ins] + '\n    "a5_hook_source",   # A5 controller seam: which route enabled the eviction\n' + s[ins:]
p.write_text(s)
import ast
t = ast.parse(s)
for n in t.body:
    if isinstance(n, ast.Assign) and any(getattr(x, "id", None) == "_FAMILY_PREFIXES" for x in n.targets):
        d = tuple(e.value for e in n.value.elts if isinstance(e, ast.Constant))
        print("prefixes now:", len(d), "| a5_hook_source covered:",
              any("a5_hook_source".startswith(x) for x in d))
