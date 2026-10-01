#!/usr/bin/env python3
"""The pre-GPU probe tested `installed(locus)[0]` -- which, with a stack, is the INCUMBENT.

WHAT HAPPENED
-------------
`probe_install_discriminate.py` took `c = inst[0]`: the FIRST controller registered at the locus. Under
the runner's compose order the STACK installs first and the candidate last, so with the R9 incumbent
installed at `post_execution` the probe was testing **A1's** predicate while reporting on the candidate.

It then said NOT DISCRIMINATING -- correctly, for A1, which does not read `best_similarity` -- and killed
all six R10b arms. The log made it visible only in hindsight: `COMPOSITION n=4` in the failing run against
`n=1` in my standalone check, which is the whole difference.

LATENT SINCE STACKS EXISTED. Every earlier candidate happened to be the only controller at its locus, or
the first one. A9 is the first candidate to share `post_execution` with TWO incumbent controllers, so it
is the first to expose it. Notably this fix and the numeric-threshold fix were BOTH required, and the
first masked the second: with numeric probing broken, the wrong-controller selection produced the same
verdict for a different reason.

THE FIX
-------
Select by the spec's own `name`, falling back to the LAST-installed controller (the candidate under the
runner's compose order). When more than one controller is present it prints what it chose and what else
was there, so a future mis-selection is visible rather than inferred:

    candidate at post_execution: 'a9_cross_container_retrieval_merge'
      (of 3 installed: ['kv_core_capacity_relocation', 'rec_sum_capacity_recovery',
                        'a9_cross_container_retrieval_merge'])
    discriminates on 'best_similarity' (0.2 -> 0.4): True -> False

THE TRANSFERABLE POINT, and it is the third instance of the same shape this session: a check that
IDENTIFIES ITS SUBJECT BY POSITION breaks the moment the population grows. `inst[0]`, `spec["name"]` in
the SIGNALS table, and `_mg_exec_calls` precedence were all position- or convention-based, and all three
failed when a second thing arrived. Identify by NAME and print the choice.

Backup at probe_install_discriminate.py.bak_prepick.
"""

import pathlib
p = pathlib.Path("/path/to/isolated-workspace/probe_install_discriminate.py")
s = p.read_text()
old = '''inst = list(installed(locus))
if not inst:
    sys.exit("INSTALL FAILED: no controller registered at %s" % locus)
c = inst[0]'''
new = '''inst = list(installed(locus))
if not inst:
    sys.exit("INSTALL FAILED: no controller registered at %s" % locus)

# PICK THE CANDIDATE, NOT WHOEVER IS FIRST. `inst[0]` is the FIRST controller at this locus, which with
# an incumbent STACK installed is the incumbent -- so the probe was testing A1's predicate while claiming
# to test the candidate's, and reported NOT DISCRIMINATING because A1 does not read the candidate's
# fields. Latent since stacks existed; it surfaced on A9 because A9 is the first candidate to share
# post_execution with two incumbent controllers.
#
# Match on the spec's own name; fall back to the last-installed controller, which is the candidate under
# the runner's compose order (STACK first, SPEC last).
_want = str(spec.get("name") or "")
_by_name = [x for x in inst if str(getattr(x, "name", "")) == _want]
c = _by_name[0] if _by_name else inst[-1]
if len(inst) > 1:
    print("candidate at %s: %r (of %d installed: %s)"
          % (locus, getattr(c, "name", "?"), len(inst),
             [getattr(x, "name", "?") for x in inst]))'''
assert s.count(old) == 1, f"anchor count {s.count(old)}"
p.write_text(s.replace(old, new, 1))
compile(p.read_text(), str(p), "exec")
print("probe now selects the CANDIDATE by name, not inst[0]")
