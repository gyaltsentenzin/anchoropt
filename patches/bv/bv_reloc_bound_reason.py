#!/usr/bin/env python3
"""Record WHY a relocation request that fired was never acted on.

ISOLATED COPIES ONLY -- refuses the shared tree.

THE DEFECT. The generated evaluator gates the relocation body on

    if (not self.disable_gates and _reloc_requested
        and _relocations_done < 3 and step_count < self.max_steps_per_turn):
        _rl_failing = <the call whose result carried the capacity error>
        if _rl_failing is None:
            step_record["relocate_no_failing_call"] = True
        else:
            <relocate_and_retry>

and THAT OUTER `if` has no `else`. A request that fires but loses the bound check writes
`relocate_requested_gate` + `relocate_controller` and then falls through in total silence. Measured on
job 1834414: 84 requests = 67 verified + 1 guard-declined + **16 carrying no reason at all**, so
criterion 3's population does not sum and the arm reports PENDING_VALIDATION -- 16 unattributed.

This is MISSING EVIDENCE, not safe behaviour. Key absence is not a recorded reason: the checker is
right to refuse, and the remedy belongs on the adapter's telemetry path, not in the criterion.

WHY v1 OF THIS PATCH CRASHED THE ARM (job 1835324, UnboundLocalError on `_rl_failing`)
-------------------------------------------------------------------------------------
v1 anchored on the INNER pair

    if _rl_failing is None:
        step_record["relocate_no_failing_call"] = True

and appended `else:` directly after it. But that `if` ALREADY HAS an `else` -- the one holding
`relocate_and_retry`. Inserting a new `else:` there bound it to the inner `if` and RE-PARENTED the
existing `else` one level deeper, so the relocation ran only when the bound check FAILED and read
`_rl_failing`, which is bound only on the branch that was taken. The mechanism inverted.

The textual diff really was "an else block alone", and the file really did compile. Neither fact can
see re-parenting: an `else:` inserted at a given column silently changes which `if` every FOLLOWING
line belongs to. Checking indent columns and `py_compile` both pass on the broken version.

SO THIS VERSION ANCHORS ON THE END OF THE OUTER BLOCK and asserts the structure it expects. The
`else` is emitted at the OUTER `if`'s own column (16), after the inner if/else pair has closed, and
the patch verifies post-application that `relocate_and_retry` is still reached at its original depth.

WHAT THIS CHANGES: nothing that executes. The conditions, the bound, and every action are byte
identical; only a previously-silent fall-through now records a reason.

WHY THE NAME ENDS `_gate`: `traj_sidecar._STEP_FIELDS` is an allowlist whose one extra rule is that a
truthy `*_gate` key survives. Naming the key `relocate_declined_bound_gate` and giving it a non-empty
reason string means it reaches the sidecar without a second patch to the allowlist -- fifth occurrence
of that allowlist silently eating intervention telemetry. The `or "condition_false"` fallback is
load-bearing: an empty reason is FALSY and would be dropped, reproducing the defect being fixed.

NOTE PRESERVED SEPARATELY, NOT FIXED HERE: all 16 refusals follow >=3 prior verified relocations in
the same trajectory, but 8 follow FIVE, which a strict 3-per-episode bound makes impossible. The
counter is per-turn, not per-episode. That is a BEHAVIOURAL defect in measured code and is deliberately
left alone -- changing it mid-experiment would rescue a result by editing the mechanism.
"""

from __future__ import annotations

import pathlib
import re
import sys

MARKER = "# [anchoropt-patch:reloc_bound_reason]"

#: The LAST lines of the outer bound block. The new `else` goes after them, at the outer `if`'s own
#: column, so it cannot capture the inner `else` that holds the relocation call.
#:
#: It is deliberately LONG. `step_count += 1 / continue` alone occurs 4 times in this file, and the
#: uniqueness guard below refuses rather than guessing -- so the anchor carries enough of the
#: relocation tail to identify exactly one site.
ANCHOR = '''                            step_record["relocate_gate"] = True
                            _extra = list(_rl_out.get("relocate_extra_results") or [])
                            step_record["tool_results"] = (
                                step_record.get("tool_results", []) + [str(x) for x in _extra])
                            execution_results = list(execution_results) + [str(x) for x in _extra]
                            step_count += 1
                            continue
'''

BLOCK = ANCHOR + '''                else:
                    ''' + MARKER + '''
                    # The request fired but the bound block was not entered. Record WHICH
                    # sub-condition refused, so criterion 3's 84-request population SUMS instead
                    # of leaving silent requests unattributed. Telemetry only: the condition
                    # above is unchanged, so this cannot alter what the arm did.
                    if _reloc_requested:
                        _rl_why = []
                        if self.disable_gates:
                            _rl_why.append("gates_disabled")
                        if not (_relocations_done < 3):
                            _rl_why.append("episode_bound_reached(%d/3)" % _relocations_done)
                        if not (step_count < self.max_steps_per_turn):
                            _rl_why.append("step_budget_exhausted(%d/%d)" % (
                                step_count, self.max_steps_per_turn))
                        # An EMPTY reason is falsy and the sidecar would drop it, which is the
                        # very defect this patch exists to fix. Never let it be empty.
                        step_record["relocate_declined_bound_gate"] = (
                            ",".join(_rl_why) or "condition_false")
'''


def _refuse_shared(path: pathlib.Path) -> None:
    if "anchoropt-wei" in str(path) or "ek_iso" in str(path):
        sys.exit(f"REFUSING: {path} is a shared/other-experiment tree")


def _structure_ok(text: str) -> tuple[bool, str]:
    """The outer bound `if` must exist, have no `else`, and wrap the inner if/else pair.

    This is what v1 failed to check. Column arithmetic and py_compile both pass on the broken
    version, so the only honest check is on the PARSE TREE: find the relocation call and confirm
    it is still reached through `if _rl_failing is None: ... else: <relocate>`.
    """
    import ast
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # the inner discriminator: `if _rl_failing is None:`
        src = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
        if src.replace(" ", "") != "_rl_failingisNone":
            continue
        if not node.orelse:
            return False, "inner `if _rl_failing is None` has no else -- structure unrecognised"
        body = "\n".join(ast.unparse(s) for s in node.orelse) if hasattr(ast, "unparse") else ""
        if "relocate_and_retry" not in body:
            return False, ("inner else no longer contains relocate_and_retry -- the relocation was "
                           "re-parented (this is the v1 defect)")
        return True, "relocate_and_retry is reached from the inner else, at its original depth"
    return False, "could not find `if _rl_failing is None` -- evaluator shape changed"


def main() -> int:
    if len(sys.argv) != 2:
        return int(bool(sys.stderr.write(f"usage: {sys.argv[0]} <memory_evaluator.py>\n")))
    path = pathlib.Path(sys.argv[1]).resolve()
    _refuse_shared(path)
    text = path.read_text()

    if MARKER in text:
        print(f"[skip] already patched: {path}")
        return 0

    # Refuse unless the PRE-state is the shape we think it is.
    ok, why = _structure_ok(text)
    if not ok:
        return int(bool(sys.stderr.write(f"REFUSING (pre-check): {why}\n")))

    n = text.count(ANCHOR)
    if n != 1:
        return int(bool(sys.stderr.write(
            f"REFUSING: anchor found {n} times, expected exactly 1\n")))

    # The anchor's `continue` must be the last line of the outer bound block: the next
    # non-blank line must be dedented to at most column 16.
    tail = text.split(ANCHOR, 1)[1]
    nxt = next((l for l in tail.splitlines() if l.strip()), "")
    indent = len(nxt) - len(nxt.lstrip())
    if indent > 16:
        return int(bool(sys.stderr.write(
            f"REFUSING: line after anchor is indented {indent} (>16), so the anchor is not the "
            f"end of the outer block: {nxt.strip()[:70]!r}\n")))

    backup = path.with_suffix(path.suffix + ".pre_boundreason")
    if not backup.exists():
        backup.write_text(text)

    new = text.replace(ANCHOR, BLOCK, 1)

    # POST-check on the parse tree, before writing: the relocation must still hang off the inner
    # else. This is the check whose absence let v1 ship an inverted mechanism.
    ok, why = _structure_ok(new)
    if not ok:
        return int(bool(sys.stderr.write(f"REFUSING (post-check): {why}\n")))
    compile(new, str(path), "exec")

    path.write_text(new)
    print(f"[ok] patched {path}\n     {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
