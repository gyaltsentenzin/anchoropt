"""The cycle-#2 driver must not decide WHERE, WHAT or HOW either.

A closed loop that picks its own answer is not evidence that the algorithm picks it. The recovery
harness is held to this and so is the loop: both supply evidence and score outcomes, and the search
belongs to `optimize_residual`.

Also pins the two things that make cycle #2 a SECOND cycle rather than a rerun of the first: it mines
the incumbent it was promoted INTO, and it reports whether the residual distribution moved.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "self_evolve_cycle2.py"


def _code() -> str:
    """Source with comments and the module docstring stripped.

    The driver NAMES the forbidden imports in a comment, to record that they are absent; a test that
    counted that comment as a violation would forbid documenting the property.
    """
    src = DRIVER.read_text().split('"""', 2)[-1]
    return "\n".join(line.split("#", 1)[0] for line in src.splitlines())


def test_the_driver_cannot_name_a_coordinate():
    """Checked against EXECUTABLE code: imports, calls and attribute access -- not docstrings.

    The driver's own docstrings legitimately name core machinery in order to record what it does NOT
    use and why (e.g. that clause pooling fires only on the SIGNAL_BLOCKED path, so it is the wrong
    tool for an expressible residual). A check that counted prose would forbid documenting the
    property it is trying to enforce -- the same trap as a mechanism term matching ordinary English.
    """
    import ast
    tree = ast.parse(DRIVER.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    for forbidden in ("IncisionPoint", "AnchorPolicyOpt", "SearchSpaceProposal",
                      "expand_and_resume", "expand_and_resume_predicates",
                      "group_blocked_by_clause", "build_arms", "synthesize"):
        assert forbidden not in names, (f"the cycle driver USES {forbidden!r} -- the search belongs "
                                        f"to core, so the loop cannot steer it")


def test_the_driver_delegates_to_optimize_residual():
    assert "optimize_residual(" in _code(), "the driver must call the core search"


def test_the_core_call_receives_evidence_only():
    """No locus, no signal, no action may cross the boundary into the search."""
    code = _code()
    call = code[code.index("optimize_residual("):]
    call = call[:call.index(")\n")]
    for leak in ("signals=", "boundaries=", "locus", "action="):
        assert leak not in call, f"the core call passes {leak!r}, which would steer the search"


def test_the_arm_is_chosen_by_firing_behaviour_not_by_name():
    """Selection among built controllers must be behavioural, so it cannot smuggle in a preference."""
    code = _code()
    assert "firing_vector(" in code, "arm selection must use the behavioural fingerprint"


def test_promotion_requires_a_VALID_evaluation_not_just_an_intact_denominator():
    """Stronger than the old assertion, which pinned the defect itself.

    The driver used to read `IMPROVED if (net > 0 and ok_denom) else NO_BENEFIT`, so a denominator
    mismatch -- no paired comparison at all -- was filed as NO_BENEFIT: a broken measurement banked as
    evidence against the candidate. Validity is now decided FIRST, over three independent channels
    (denominator, contamination, intervention channel), by anchoropt.learning.termination.
    """
    code = _code()
    assert "classify_residual(" in code, "the driver must derive the outcome, not hand-roll it"
    assert "Validity(" in code and "denominator_ok=ok_denom" in code
    assert "contamination_free=" in code, "prereq contamination must be measured per round"
    assert "channel_ok=" in code, "intervention-channel integrity must be checked"
    assert not re.search(r"IMPROVED if \(net > 0 and ok_denom\)", code), \
        "the defective line is back: an invalid round would be recorded as a negative result"
    # and promotion happens only on the derived PROMOTED outcome
    assert 'term.outcome == "PROMOTED"' in code


def test_an_invalid_round_does_not_trigger_promotion_or_remining():
    """Re-mining after an invalid round would build the next cycle on a measurement nobody trusts."""
    code = _code()
    i_guard = code.index('term.outcome == "PROMOTED"')
    i_remine = code.index("residual_shift")
    assert i_guard < i_remine, "re-mining is not gated on a genuine promotion"


def test_the_global_verdict_is_reported():
    """A single rejected arm must never read as the end of evolution."""
    code = _code()
    assert "global_verdict(" in code
    assert "GLOBAL" in code or "gv.verdict" in code


def test_firings_come_from_the_trajectory_sidecar_not_a_registry():
    """`gates_fired` is registry-derived and blind to remedy flags; the sidecar is the only source."""
    code = _code()
    assert "gates_fired" not in code, "firings must not be read from the registry-derived dict"
    assert "_gate" in code, "firings must be read from the *_gate sidecar keys"


def test_the_loop_remines_after_promoting_and_reports_the_shift():
    """The whole point: a promotion changes the trajectories, so the residual must be re-mined."""
    code = _code()
    assert "residual_shift" in code
    for field in ("disappeared", "new", "moved"):
        assert field in code, f"the residual shift must report {field!r}"


def test_answer_commitment_steps_are_kept_when_building_evidence():
    """Dropping them would derive a single boundary and make the backward search unobservable."""
    code = _code()
    assert "runtime.is_decision(s)" in code, \
        "step filtering must ask the runtime, or commitment boundaries vanish"


def test_the_oracle_guard_runs():
    code = _code()
    assert "assert_no_oracle_leak()" in code


# ---- boundary observability is enforced PROGRAMMATICALLY ----------------------------------------
#
# WRITE1's search selected a controller at the commitment gate whose phi read a field the hook only
# supplies post-execution. It would have fired 0 times, and the round would have measured nothing. No
# prompt revision catches that; only an invariant does.

def _c2():
    import importlib.util
    spec = importlib.util.spec_from_file_location("c2", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Arm:
    def __init__(self, locus):
        self.boundary = type("B", (), {"value": locus})()
        self.signal = "probe"


def test_a_controller_whose_field_the_boundary_does_not_supply_is_REJECTED():
    import sys
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    from anchoropt.learning.signal_grammar import Atom
    mod = _c2()
    states = [{"best_similarity": i / 10, "searched_other_container": i % 2 == 0}
              for i in range(10)]
    phi = Atom(field="searched_other_container", op="falsy")   # post-execution field
    ok, why = mod.can_fire_at_boundary(_Arm("post_generation_pre_exec"), phi,
                                      runtime=rt, states=states)
    assert not ok, "an unfireable controller was accepted"
    assert "does not supply" in why and "0 times" in why


def test_the_same_predicate_IS_accepted_where_the_field_is_supplied():
    """The check must discriminate by boundary, not reject the predicate outright."""
    import sys
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    from anchoropt.learning.signal_grammar import Atom
    mod = _c2()
    states = [{"best_similarity": i / 10, "searched_other_container": i % 2 == 0}
              for i in range(10)]
    phi = Atom(field="searched_other_container", op="falsy")
    ok, _ = mod.can_fire_at_boundary(_Arm("post_execution"), phi, runtime=rt, states=states)
    assert ok


def test_a_never_firing_predicate_is_rejected():
    import sys
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    from anchoropt.learning.signal_grammar import Atom
    mod = _c2()
    never = [{"searched_other_container": True} for _ in range(6)]
    ok, why = mod.can_fire_at_boundary(_Arm("post_execution"),
                                       Atom(field="searched_other_container", op="falsy"),
                                       runtime=rt, states=never)
    assert not ok and "none of" in why


def test_an_UNCONDITIONAL_policy_is_permitted_not_rejected():
    """"Always act here" is a real intervention, and the simplest one in the space.

    It must not be confused with a predicate that CANNOT be evaluated: the first is a policy choice,
    the second is a broken controller. Forbidding the former would rule out, for instance, reprompting
    on every proposed write -- which may well be the right upstream policy.
    """
    import sys
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    from anchoropt.learning.signal_grammar import Atom
    mod = _c2()
    always = [{"searched_other_container": False} for _ in range(6)]
    ok, why = mod.can_fire_at_boundary(_Arm("post_execution"),
                                       Atom(field="searched_other_container", op="falsy"),
                                       runtime=rt, states=always)
    assert ok, "an unconditional policy was rejected"
    assert "unconditional" in why.lower(), "it must still be FLAGGED as unconditional"


def test_unevaluable_and_unconditional_are_distinguished():
    import sys
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    from anchoropt.learning.signal_grammar import Atom
    mod = _c2()
    states = [{"best_similarity": 0.2} for _ in range(6)]
    ok, why = mod.can_fire_at_boundary(_Arm("post_generation_pre_exec"),
                                       Atom(field="best_similarity", op="lt", value=0.5),
                                       runtime=rt, states=states)
    assert not ok and "does not supply" in why


def test_synthesis_is_offered_only_fields_the_boundary_SUPPLIES():
    """The space must not contain impossible points -- rejecting the winner is not a fix.

    A search whose space contains predicates the boundary cannot evaluate spends its budget on them
    and then reports an exhaustion it never actually reached.
    """
    import sys
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    for locus in ("post_generation_pre_exec", "post_execution"):
        offered = set(rt.synthesis_fields(locus))
        supplied = set(rt.hook_state_fields(locus))
        assert offered <= supplied, (f"{locus}: synthesis offered {sorted(offered - supplied)} "
                                    f"which the hook does not supply")


def test_a_hookless_boundary_is_UNIMPLEMENTED_not_searchable():
    """A boundary with no live hook must not be searched, and must not read as EXHAUSTED either.

    Superseding an earlier version of this test, which asserted the declarations were KEPT in
    `synthesis_fields`. That was wrong in a way that mattered: offering them made the boundary look
    searchable, and the inevitable empty result surfaced as SIGNAL_EXPANSION_EXHAUSTED -- reporting an
    exhausted search for a boundary that was never searched at all. The declarations are preserved,
    but as METADATA on `unimplemented_boundaries()`, where they describe the runtime without claiming
    the boundary is usable.
    """
    import sys
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    assert rt.hook_state_fields("pre_generation") == frozenset(), "fixture changed: a hook appeared"
    assert not rt.synthesis_fields("pre_generation"), \
        "a hookless boundary is being offered to synthesis"
    unimpl = rt.unimplemented_boundaries()
    assert "pre_generation" in unimpl, "the boundary is unsearchable but not REPORTED as such"
    assert "declared but unreachable" in unimpl["pre_generation"]


def test_an_implemented_boundary_is_not_reported_unimplemented():
    import sys
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    unimpl = rt.unimplemented_boundaries()
    for live in ("post_generation_pre_exec", "post_execution"):
        assert live not in unimpl
        assert rt.synthesis_fields(live), f"{live} has a hook but offers no fields"


def test_the_adapter_answers_what_its_hook_actually_supplies():
    """Declared != populated. The adapter must be able to state the difference."""
    import sys
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    up = rt.hook_state_fields("post_generation_pre_exec")
    post = rt.hook_state_fields("post_execution")
    assert "proposed_call" in up and "searched_other_container" not in up
    assert "searched_other_container" in post and "proposed_call" not in post
    assert rt.hook_state_fields("pre_generation") == frozenset(), "a hookless boundary must be empty"


def test_fields_read_is_structural_not_name_parsing():
    from anchoropt.learning.signal_grammar import Atom, Conjunction
    mod = _c2()
    conj = Conjunction(terms=(Atom(field="a", op="truthy"), Atom(field="b", op="lt", value=1)))
    assert mod.fields_read(conj) == {"a", "b"}
    assert mod.fields_read(Atom(field="solo", op="truthy")) == {"solo"}


# ---------------------------------------------------------------- the four criteria are consulted

def test_the_driver_adjudicates_ALL_FOUR_criteria_not_just_net():
    """`net > 0` is criterion 1 alone and was the entire installation test before this.

    Pinned because `acceptance_criteria` sat in the tree with ZERO production consumers -- every
    reference to it outside its own module was a test fixture -- so the four-criterion rule existed in
    prose and in unit tests but never gated an install.
    """
    code = _code()
    assert "evaluate_criteria(" in code, "the driver must adjudicate the four criteria"
    assert '"accepted": bool(report.accepted)' in code, \
        "classify_residual must receive the FOUR-criterion verdict, not net > 0 alone"
    assert not re.search(r'"accepted":\s*bool\(net > 0\)', code), \
        "criterion 1 alone must no longer decide installation"


def test_the_driver_supplies_a_held_out_input_path_for_criterion_2():
    """Criterion 2 has no input unless the CLI can name a dev run. Absent => PENDING => blocked."""
    code = _code()
    assert '"--dev"' in code, "criterion 2 needs a held-out arm run"
    assert "dev=dev" in code


def test_criterion_3_and_4_evidence_comes_from_the_ARMS_OWN_sidecar():
    """Not from a registry-derived dict, and not from the spec. `asteps` is the arm's trajectory."""
    code = _code()
    assert "mechanism_evidence(asteps" in code, \
        "mechanism and safety evidence must be read from the arm's own trajectory steps"
    assert "control_steps=bsteps" in code, \
        "added clears are a PAIRED quantity, so the control's trajectory is required too"
    assert "safety=safety" in code and "telemetry=mech_tel" in code


def test_a_PENDING_criterion_is_routed_as_UNRESOLVED_not_as_a_refutation():
    """A criterion whose evidence was never gathered must not be banked against the candidate."""
    code = _code()
    assert "evidence_complete=False" in code, \
        "missing evidence must invalidate the round, not count as a measured negative result"
    assert "missing_evidence=" in code, "what was missing must travel with the verdict"
