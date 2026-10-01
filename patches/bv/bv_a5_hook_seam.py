#!/usr/bin/env python3
"""A5's eviction was reachable ONLY by a legacy remedy flag. Give it the seam A9 already has.

WHAT WAS AND WAS NOT MISSING
----------------------------
A5's executor is **already live** in the BV evaluator and is the careful version: it reads the LIVE
container (an earlier version read its own write log, saw a median of 4 entries against a real ~50, and
the arm was void), picks a deterministic victim, enforces `copies >= 2` **at the point of action**, and
retries the ORIGINAL call verbatim (a synthesized retry failed on every kv firing *after* the eviction had
succeeded -- destroying a duplicate and losing the write).

What was missing is only the seam: the branch was gated by
`remedy_enabled(templates, "enable_archival_evict_duplicate")` and nothing else, so an autonomous
controller could not ground it. A9 at the same locus already had `_hook_decide` sitting beside its legacy
`_xcm_enabled()` flag and OVERRIDING it, so this is that pattern applied, not a new mechanism.

THE SAFETY LINE, drawn deliberately
-----------------------------------
`E1_MAX_EVICTIONS = 3` and the `copies < 2` refusal stay OUTSIDE the seam. Whichever route asked, the
budget still bounds the loop and the invariant still refuses a lossy eviction.

    A controller may request the action; it may not relax the safety.

This is the lesson from A8, where suppression and repair were separately gated and the suppression
outlived the repair budget (R6f: 129 withheld, 8 repaired, 31 extra clears). Putting a budget behind a
controller-controllable flag is how that happens.

A NAMING COLLISION FOUND ON THE WAY, worth recording
----------------------------------------------------
`grep _a5_` in the evaluator returns 8 hits and **none of them is this A5**. `_da_a5_history` and
`_a5_sims` belong to a LOW-SIMILARITY mechanism; the lossless eviction is named `e1_*`/`E1_*` throughout
(`E1_MAX_EVICTIONS`, `e1_evict_gate`, `_e1_pick_victim`). Third naming collision in this codebase after
"A8" (dedup-clear vs cross-container merge) and the probe's controller-vs-signal name. **Check the locus
and the action, never the label.**

TELEMETRY: `a5_hook_source` was NOT in the sidecar's `_FAMILY_PREFIXES` and is declared by
bv_a5_sidecar_key.py in the same change -- an undeclared key is dropped by the allowlist, which has
swallowed intervention telemetry repeatedly here.

Backup at memory_evaluator.py.bak_prea5hook.
"""

import pathlib
p = pathlib.Path("/path/to/isolated-workspace/anchoropt/memory_evaluator.py")
s = p.read_text()

# A5's eviction is fully implemented and live-state-verified, but reachable ONLY through the legacy
# remedy flag `enable_archival_evict_duplicate`. An autonomous controller cannot ground it. This is the
# same shape as A9, where `_hook_decide` already sits beside the legacy `_xcm_enabled()` and OVERRIDES
# it -- so the fix is to give A5 the same seam rather than inventing a new one.
old = """                    if (self.disable_gates
                            or not remedy_enabled(templates, "enable_archival_evict_duplicate")
                            or _e1_evictions >= E1_MAX_EVICTIONS):
                        continue"""
new = """                    # [anchoropt-patch:a5_hook_seam]
                    # HOST ARM vs CONTROLLER. The legacy path is `remedy_enabled(...)`; an installed
                    # controller reaches the same executor through the hook, exactly as A9 does at
                    # post_execution. The hook's answer OVERRIDES the flag, so a controller can enable
                    # A5 without the frozen policy being edited -- and with no flag and no controller
                    # the behaviour is byte-identical to before.
                    #
                    # The budget and the invariant are NOT part of the seam: E1_MAX_EVICTIONS still
                    # bounds the loop and `_vk < 2` still refuses a lossy eviction, whichever route
                    # asked. A controller may request the action; it may not relax the safety.
                    _a5_host = (not self.disable_gates
                                and remedy_enabled(templates, "enable_archival_evict_duplicate"))
                    _a5_state = _hook_state(step_record, decoded, execution_results)
                    _a5_state["error_kind"] = "no_capacity"
                    _a5_state["container"] = "archival"
                    _a5_state["is_relocation_target"] = True
                    _a5_hk = _hook_decide("post_execution", _a5_state, _a5_host)
                    if _a5_hk is not None:
                        step_record["a5_hook_source"] = _a5_hk[1]
                    _a5_go = (_a5_hk[0] if _a5_hk is not None else _a5_host)
                    if self.disable_gates or not _a5_go or _e1_evictions >= E1_MAX_EVICTIONS:
                        if _a5_go and _e1_evictions >= E1_MAX_EVICTIONS:
                            step_record["e1_budget_spent_gate"] = True
                        continue"""
assert s.count(old) == 1, f"anchor count {s.count(old)}"
p.write_text(s.replace(old, new, 1))
compile(p.read_text(), str(p), "exec")
print("A5 eviction now reachable by a controller, same seam as A9")
