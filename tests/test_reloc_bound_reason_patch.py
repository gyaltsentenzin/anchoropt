"""The bound-reason patch must not RE-PARENT the relocation it is supposed to leave alone.

Why this file exists: v1 of the patch inserted `else:` after

    if _rl_failing is None:
        step_record["relocate_no_failing_call"] = True

which ALREADY had an `else` holding the relocation call. The insertion bound the new `else` to the
inner `if` and pushed the existing one a level deeper, so `relocate_and_retry` ran only when the
bound check FAILED, reading `_rl_failing` -- bound only on the other branch. Job 1835324 died with
`UnboundLocalError: cannot access local variable '_rl_failing'` on the first capacity event.

The two checks I ran before shipping v1 BOTH PASS on the broken file: it compiles, and the `else`
sits at column 16. Only the parse tree can see re-parenting. These tests are therefore written
against the AST, and one of them reconstructs the v1 output and asserts it is REJECTED.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PATCH = REPO / "patches" / "bv" / "bv_reloc_bound_reason.py"

#: The shape of the real evaluator region, reduced to what the patch reasons about. Indentation is
#: the production indentation (outer `if` at 16) because the patch is textual and depends on it.
PRISTINE = '''
class E:
    def run(self):
        for turn in turns:
            while True:
                if (
                    not self.disable_gates
                    and _reloc_requested
                    and _relocations_done < 3
                    and step_count < self.max_steps_per_turn
                ):
                    _rl_failing = None
                    for _ri, _rc2 in enumerate(decoded or []):
                        if "is full" in _rlow:
                            _rl_failing = str(_rc2)
                            break
                    if _rl_failing is None:
                        step_record["relocate_no_failing_call"] = True
                    else:
                        def _rl_exec(_calls):
                            return self._execute(_calls)
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

                if something_else:
                    pass
'''


@pytest.fixture(scope="module")
def patchmod():
    spec = importlib.util.spec_from_file_location("bound_patch", PATCH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _apply(patchmod, text):
    assert text.count(patchmod.ANCHOR) == 1
    return text.replace(patchmod.ANCHOR, patchmod.BLOCK, 1)


def _outer_bound_if(text):
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.If):
            t = ast.unparse(n.test)
            if "_reloc_requested" in t and "disable_gates" in t and "_relocations_done" in t:
                return n
    raise AssertionError("outer bound if not found")


# ------------------------------------------------------------------ the v1 defect, pinned

def test_the_v1_output_is_REJECTED_by_the_structure_check(patchmod):
    """The exact transformation that crashed job 1835324 must not be accepted again."""
    v1_anchor = ('                    if _rl_failing is None:\n'
                 '                        step_record["relocate_no_failing_call"] = True\n')
    v1 = PRISTINE.replace(v1_anchor, v1_anchor + '                else:\n'
                                                 '                    if _reloc_requested:\n'
                                                 '                        step_record["x"] = "y"\n', 1)
    ok, why = patchmod._structure_ok(v1)
    assert not ok
    assert "re-parent" in why or "no else" in why


def test_compiling_and_column_checks_CANNOT_catch_the_v1_defect():
    """Pins why the AST check is necessary: the cheap checks pass on the broken file.

    If this ever fails it means a cheap check DID catch it, and the AST check could be relaxed --
    but as long as it passes, removing the AST check would silently restore the v1 hazard.
    """
    v1_anchor = ('                    if _rl_failing is None:\n'
                 '                        step_record["relocate_no_failing_call"] = True\n')
    v1 = PRISTINE.replace(v1_anchor, v1_anchor + '                else:\n'
                                                 '                    if _reloc_requested:\n'
                                                 '                        step_record["x"] = "y"\n', 1)
    compile(v1, "<v1>", "exec")          # compiles fine
    line = next(l for l in v1.splitlines() if l.strip() == "else:")
    assert len(line) - len(line.lstrip()) == 16   # and the column looks right


def test_v1_moved_the_relocation_out_of_the_bound_body():
    """The actual semantic damage, stated as an assertion rather than prose."""
    v1_anchor = ('                    if _rl_failing is None:\n'
                 '                        step_record["relocate_no_failing_call"] = True\n')
    v1 = PRISTINE.replace(v1_anchor, v1_anchor + '                else:\n'
                                                 '                    if _reloc_requested:\n'
                                                 '                        step_record["x"] = "y"\n', 1)
    outer = _outer_bound_if(v1)
    body = "\n".join(ast.unparse(s) for s in outer.body)
    assert "relocate_and_retry" not in body, "v1 left the relocation in the body -- restate the test"


# ------------------------------------------------------------------ v2 correctness

def test_v2_attaches_the_else_to_the_OUTER_bound_if(patchmod):
    outer = _outer_bound_if(_apply(patchmod, PRISTINE))
    assert outer.orelse, "outer bound if still has no else"
    els = "\n".join(ast.unparse(s) for s in outer.orelse)
    assert "relocate_declined_bound_gate" in els
    assert "relocate_and_retry" not in els, "the else must NOT contain the mechanism"


def test_v2_leaves_the_mechanism_in_the_bound_BODY(patchmod):
    outer = _outer_bound_if(_apply(patchmod, PRISTINE))
    body = "\n".join(ast.unparse(s) for s in outer.body)
    assert "relocate_and_retry" in body
    assert "_rl_failing = None" in body, "the binding must stay on the same branch as its use"


def test_v2_is_an_ADD_ONLY_text_change(patchmod):
    """Every pristine line must survive verbatim and in order -- nothing edited, nothing moved."""
    before = PRISTINE.splitlines()
    after = _apply(patchmod, PRISTINE).splitlines()
    it = iter(after)
    for line in before:
        assert any(line == a for a in it), f"pristine line lost or reordered: {line!r}"


def test_v2_passes_its_own_structure_check(patchmod):
    ok, _ = patchmod._structure_ok(_apply(patchmod, PRISTINE))
    assert ok


def test_v2_output_compiles(patchmod):
    compile(_apply(patchmod, PRISTINE), "<v2>", "exec")


def test_the_patch_refuses_a_structure_it_does_not_recognise(patchmod):
    ok, why = patchmod._structure_ok("def f():\n    return 1\n")
    assert not ok and "_rl_failing" in why


# ------------------------------------------------------------------ the falsy-reason hazard

def test_the_emitted_reason_can_never_be_falsy(patchmod):
    """`relocate_declined_bound_gate` survives the sidecar only on the *_gate TRUTHY rule, so an
    empty reason string would be dropped -- reproducing the defect this patch fixes."""
    assert 'or "condition_false"' in patchmod.BLOCK


def test_every_reachable_branch_combination_yields_a_truthy_reason():
    """All 8 combinations of the three sub-conditions, including the one where none of them is the
    stated cause (e.g. _reloc_requested raced), must still write something non-empty."""
    for gates_disabled in (False, True):
        for done in (0, 3):
            for step, maxstep in ((0, 10), (10, 10)):
                why = []
                if gates_disabled:
                    why.append("gates_disabled")
                if not (done < 3):
                    why.append("episode_bound_reached")
                if not (step < maxstep):
                    why.append("step_budget_exhausted")
                assert ",".join(why) or "condition_false"


def test_the_key_name_keeps_the_gate_suffix(patchmod):
    """Renaming it without the suffix would make the sidecar drop it silently."""
    assert 'step_record["relocate_declined_bound_gate"]' in patchmod.BLOCK
    assert "relocate_declined_bound_gate".endswith("_gate")
