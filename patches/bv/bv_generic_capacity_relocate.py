#!/usr/bin/env python3
"""Make the core-capacity RELOCATION recovery reachable to an installed controller.

ISOLATED COPIES ONLY -- refuses the shared tree.

THE RESIDUAL, measured on the frozen raw-H0 kv runs (132 trajectories): 218 core writes refused with
"Core memory is full", and 208 of those 218 (95.4%) followed by a wholesale `core_memory_clear`. The
failure is not that the store is full; it is that the agent's own recovery destroys what it accumulated.

WHY THIS IS A SEPARATE PATCH from bv_generic_capacity_repair. That one reaches a CHARACTER reduction,
which cannot relieve an ENTRIES-count refusal -- shortening a value frees zero slots. Two different
constraints need two different executors, and the capability table now declares disjoint signal sets so
one cell cannot dispatch to the wrong mechanism.

CAPABILITY IDENTITY IS REQUIRED, not optional. The post_execution/reroute cell has more than one
executor, so this branch runs ONLY for a controller whose spec names
`relocate_entry_preserving_information_then_retry`. An unrelated post-execution controller -- including
the accepted rec_sum reduction -- must not be able to trigger a relocation. Same discipline as
bv_capability_identity_repair.

SAFETY IS IN scripts/capacity_relocate.py AND IS VERIFIED THERE AGAINST LIVE STATE: destination write
first and confirmed present, only then the source removal, retry the refused call verbatim only once a
slot is genuinely free, nothing removed if the write did not land, bounded at 3 per episode. This patch
adds no safety logic of its own -- it only routes the decision.

NO TRIGGER, NO THRESHOLD, NO ANCHOR IDENTITY is read on this path. The installed predicate decides
whether to act; the primitive decides what to do.
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "# [anchoropt-patch:generic_capacity_relocate]"

# Inserted immediately BEFORE the character-reduction gate, so the two are visibly siblings and the
# ordering is explicit rather than incidental. Both are latched independently.
ANCHOR = '''                if (
                    not self.disable_gates
                    and not _capacity_repair_fired
'''

BLOCK = '''                ''' + MARKER + '''
                # CORE-CAPACITY RELOCATION: free ONE slot by preserving an entry elsewhere first.
                #
                # Distinct from the character reduction below: that one shortens a payload and cannot
                # relieve an entries-count refusal. Reached only by a controller whose spec names THIS
                # capability, because this cell has more than one executor.
                _reloc_requested = False
                _reloc_ctl_name = ""
                try:
                    from anchoropt.runtime_hook import installed as _rlinst
                    _rl_ctls = [c for c in _rlinst("post_execution")
                                if _phase_eligible(c, str(rollout_tag) == "snap")]
                    _RL_WANT = "relocate_entry_preserving_information_then_retry"
                    for _rc in _rl_ctls:
                        _rc_spec = dict(getattr(_rc, "spec", {}) or {})
                        _rc_cap = str(_rc_spec.get("capability_id") or "")
                        _rc_op = str(_rc_spec.get("operator") or _rc_spec.get("action") or "")
                        if _rc_cap != _RL_WANT or _rc_op not in ("transform", "substitute", "reroute"):
                            if _rc_cap:
                                step_record["relocate_declined_identity_gate"] = (
                                    "%s/%s != %s" % (_rc_cap, _rc_op, _RL_WANT))
                            continue
                        _rl_state = _hook_state(step_record, decoded, execution_results,
                                                test_entry_id)
                        try:
                            if _rc.fires_on(_rl_state):
                                _reloc_requested = True
                                _reloc_ctl_name = getattr(_rc, "name", "controller")
                                step_record["relocate_requested_gate"] = True
                                step_record["relocate_controller"] = _reloc_ctl_name
                                break
                        except Exception as _rce:
                            step_record["relocate_predicate_error_gate"] = "%s: %s" % (
                                type(_rce).__name__, _rce)
                except Exception as _rle:
                    step_record["relocate_hook_error_gate"] = "%s: %s" % (
                        type(_rle).__name__, _rle)

                if (
                    not self.disable_gates
                    and _reloc_requested
                    and _relocations_done < 3
                    and step_count < self.max_steps_per_turn
                ):
                    # The refused call is the one whose result carried the capacity error.
                    _rl_failing = None
                    for _ri, _rc2 in enumerate(decoded or []):
                        _rr = str(execution_results[_ri]) if _ri < len(execution_results or []) else ""
                        _rlow = _rr.lower()
                        if "is full" in _rlow or "exceeds maximum size" in _rlow:
                            _rl_failing = str(_rc2)
                            break
                    if _rl_failing is None:
                        step_record["relocate_no_failing_call"] = True
                    else:
                        def _rl_exec(_calls):
                            return self._execute(_calls, initial_config, involved_classes,
                                                 rollout_model_name, test_entry_id)
                        try:
                            import scripts.capacity_relocate as _RL
                        except Exception:
                            import capacity_relocate as _RL
                        _rl_out = _RL.relocate_and_retry(
                            failing_call=_rl_failing, involved_instances=involved_instances,
                            execute=_rl_exec, relocations_so_far=_relocations_done)
                        for _k, _v in (_rl_out or {}).items():
                            if _k != "relocate_extra_results":
                                step_record[_k if _k.endswith("_gate") else _k] = _v
                        if _rl_out.get("relocate_ok"):
                            _relocations_done += 1
                            step_record["relocate_gate"] = True
                            _extra = list(_rl_out.get("relocate_extra_results") or [])
                            step_record["tool_results"] = (
                                step_record.get("tool_results", []) + [str(x) for x in _extra])
                            execution_results = list(execution_results) + [str(x) for x in _extra]
                            step_count += 1
                            continue

'''

# The episode-scoped counter. Declared beside the host's own latches so the bound is per episode.
COUNTER_ANCHOR = '''            _capacity_repair_fired = False
'''
COUNTER_BLOCK = '''            _capacity_repair_fired = False
            _relocations_done = 0          # [anchoropt-patch:generic_capacity_relocate] bounded
'''


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_generic_capacity_relocate.py <path to memory_evaluator.py>")
        return 2
    p = pathlib.Path(sys.argv[1])
    if "/anchoropt-wei/anchoropt/anchoropt/" in str(p.resolve()):
        print("REFUSING: that is the SHARED tree, imported by running GEPA workers.")
        return 3
    src = p.read_text()
    if MARKER in src:
        print(f"already applied ({MARKER} present) -- no change")
        return 0
    for name, a in (("gate", ANCHOR), ("counter", COUNTER_ANCHOR)):
        if src.count(a) != 1:
            print(f"FAIL: {name} anchor appears {src.count(a)} times; refusing to guess")
            return 1
    # ORDER CHECK, not just presence: the relocation branch must be reachable, i.e. the counter must be
    # declared BEFORE the gate. A branch placed after its own guard's scope cost two GPU rounds once.
    if src.index(COUNTER_ANCHOR) > src.index(ANCHOR):
        print("FAIL: the counter is declared AFTER the gate; the branch would not be in scope")
        return 1
    bak = p.with_suffix(p.suffix + ".pre_genericreloc")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    out = src.replace(COUNTER_ANCHOR, COUNTER_BLOCK, 1).replace(ANCHOR, BLOCK + ANCHOR, 1)
    p.write_text(out)
    print(f"patched {p}\n  a controller naming relocate_entry_preserving_information_then_retry can "
          f"now free ONE core slot by a verified relocation and retry the refused write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
