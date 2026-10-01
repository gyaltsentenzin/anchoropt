"""EXTERNAL PAIRED EVALUATION: the seam that replaces the firing-rate heuristic.

WHAT WAS WRONG. The BFCL driver could not evaluate inline (evaluating one arm means a GPU job), so it
passed `evaluate=lambda _a: 0` and chose an arm by `abs(firing_rate - 0.25)`. That selects without
measuring, then reports NO_BENEFIT -- asserting a measurement that never happened.

WHAT REPLACES IT. Not another heuristic. Arms are built and EMITTED; an external process measures them;
the driver feeds those recorded results to the SAME `AnchorPolicyOpt.optimize`. With no measurement the
round is UNEVALUATED and no winner is named.

THE ONE RULE, tested from several angles: a missing result is None, never zero. A zero is a
measurement; None is the absence of one, and `train_objective` would happily rank a fabricated zero
against real results.
"""

from __future__ import annotations

import json

import pytest

from anchoropt.learning.external_evaluation import (
    BUDGET_EXHAUSTED, EVALUATED_NO_BENEFIT, IMPROVED, INCOMPARABLE_INCUMBENT,
    TRAIN_IMPROVED_PENDING_VALIDATION, UNEVALUATED, ArmResult, ExternalEvaluation,
    MissingEvaluation, arm_manifest,
)
from anchoropt.learning.policy_class import ThetaResult, train_objective


class _Arm:
    def __init__(self, label):
        self.label = label


def _res(label, gains=(), losses=(), **kw):
    kw.setdefault("n", 12)
    kw.setdefault("cases_fired", 4)
    kw.setdefault("interventions_executed", 4)
    return ArmResult(arm_label=label, gains=tuple(gains), losses=tuple(losses), **kw)


# ================================================================================================
# THE ONE RULE
# ================================================================================================

def test_an_arm_with_NO_recorded_result_returns_None_not_zero():
    """The defect class in one test. A zero would be ranked against real measurements."""
    ev = ExternalEvaluation(results={})
    assert ev(_Arm("a/b/c")) is None
    assert ev.n_completed == 0
    assert ev.missing == ["a/b/c"]


def test_a_fabricated_zero_would_have_OUTRANKED_a_real_negative_result():
    """WHY the rule matters, shown numerically rather than asserted.

    If a missing result were scored as net 0, it would beat a genuinely measured -2 arm under J_train --
    so an arm nobody ran would be selected over one that was measured and found harmful.
    """
    real_negative = ThetaResult(theta={}, gains=("g1",), losses=("l1", "l2", "l3"), firings=3,
                                cases_fired=3, n=12, interventions_executed=3)
    fabricated_zero = ThetaResult(theta={}, gains=(), losses=(), firings=0, cases_fired=0, n=12,
                                  interventions_executed=0)
    assert train_objective(fabricated_zero) > train_objective(real_negative)


def test_strict_mode_RAISES_rather_than_silently_returning_None():
    """A driver that believes every arm was evaluated should fail loudly when one was not."""
    ev = ExternalEvaluation(results={}, strict=True)
    with pytest.raises(MissingEvaluation) as exc:
        ev(_Arm("missing/arm"))
    assert "UNEVALUATED" in str(exc.value)


def test_a_broken_DENOMINATOR_is_not_a_result_either():
    """VALIDITY BEFORE ACCEPTANCE. Two runs that scored different cases are not a paired comparison.

    This was a real defect once: a denominator MISMATCH was recorded as NO_BENEFIT, banking a broken
    measurement as evidence against the candidate.
    """
    ev = ExternalEvaluation(results={"x": _res("x", gains=("c1", "c2"), denominator_ok=False)})
    assert ev(_Arm("x")) is None
    assert ev.n_completed == 0
    assert "x" in ev.missing


# ================================================================================================
# THE FOUR DISJOINT OUTCOME CLASSES
# ================================================================================================

def test_nothing_measured_is_UNEVALUATED_even_though_arms_existed():
    ev = ExternalEvaluation(results={})
    for lbl in ("a", "b", "c"):
        ev(_Arm(lbl))
    assert ev.outcome_class(improved=False) == UNEVALUATED
    assert ev.n_completed == 0


def test_measured_and_none_helped_is_NO_BENEFIT():
    ev = ExternalEvaluation(results={"a": _res("a", losses=("c1",)), "b": _res("b")})
    ev(_Arm("a"))
    ev(_Arm("b"))
    assert ev.n_completed == 2
    assert ev.outcome_class(improved=False) == EVALUATED_NO_BENEFIT


def test_budget_exhaustion_is_reported_SEPARATELY_from_no_benefit():
    """Arms left unmeasured mean the round is not settled -- requeue, do not classify."""
    ev = ExternalEvaluation(results={"a": _res("a"), "b": _res("b", gains=("c1",))}, budget=1)
    assert ev(_Arm("a")) is not None
    assert ev(_Arm("b")) is None, "the second arm is past the budget"
    assert ev.budget_exhausted
    assert ev.outcome_class(improved=False) == BUDGET_EXHAUSTED
    assert ev.outcome_class(improved=False) != EVALUATED_NO_BENEFIT


def test_a_PARTIALLY_evaluated_round_is_not_a_negative_result():
    """Some arms measured and some missing is budget/incompleteness, not a null."""
    ev = ExternalEvaluation(results={"a": _res("a")})
    ev(_Arm("a"))
    ev(_Arm("b"))                      # no result recorded
    assert ev.n_completed == 1
    assert ev.outcome_class(improved=False) == BUDGET_EXHAUSTED


def test_improvement_wins_outright():
    ev = ExternalEvaluation(results={"a": _res("a", gains=("c1", "c2"))})
    r = ev(_Arm("a"))
    assert r is not None and r.net == 2
    # A train win is PROVISIONAL: criterion 1 of four, never acceptance.
    assert ev.outcome_class(improved=True) == TRAIN_IMPROVED_PENDING_VALIDATION


def test_the_four_classes_are_distinct_strings():
    """Guard against a refactor collapsing two of them."""
    assert len({UNEVALUATED, EVALUATED_NO_BENEFIT, BUDGET_EXHAUSTED, IMPROVED,
                TRAIN_IMPROVED_PENDING_VALIDATION, INCOMPARABLE_INCUMBENT}) == 6


# ================================================================================================
# THE MANIFEST ROUND TRIP
# ================================================================================================

def test_the_manifest_emits_EVERY_arm_not_one_chosen_by_a_heuristic():
    """This is what replaces `abs(firing_rate - 0.25)`: all arms go out, measurement decides."""
    from anchoropt.anchor import Action, IncisionPoint
    from anchoropt.learning.action_contract import InstantiatedAction, Operator
    from anchoropt.learning.anchor_policy_opt import PolicyArm
    from anchoropt.learning.policy_class import PolicyClass

    arms = [
        PolicyArm(boundary=IncisionPoint.POST_GENERATION_PRE_EXEC, signal="s1",
                  instantiated=InstantiatedAction(operator=Operator.SUPPRESS, variant="cancel",
                                                  eta={"suppressed_operation": "the write"}),
                  policy_class=PolicyClass.DETERMINISTIC),
        PolicyArm(boundary=IncisionPoint.POST_GENERATION_PRE_EXEC, signal="s2",
                  instantiated=InstantiatedAction(operator=Operator.REPROMPT, variant="warn",
                                                  eta={"instruction": "x", "retry_budget": 1}),
                  policy_class=PolicyClass.DETERMINISTIC),
    ]
    man = arm_manifest(arms, incumbent_id="P3", incumbent_token="tok3")
    assert man["n_arms"] == 2 and len(man["arms"]) == 2
    assert {a["arm_label"] for a in man["arms"]} == {a.label for a in arms}
    assert "must be OMITTED, not reported as zero" in man["note"]
    for row in man["arms"]:
        assert row["boundary"] and row["signal"] and row["action"] and row["operator"]


def test_results_load_from_json_and_ignore_unknown_keys(tmp_path):
    """A richer external record must not have to be trimmed to fit."""
    p = tmp_path / "results.json"
    p.write_text(json.dumps({
        "incumbent_token": "tok9",
        "results": [
            {"arm_label": "a/b/c", "gains": ["c1", "c2"], "losses": [], "n": 12,
             "cases_fired": 5, "interventions_executed": 5, "accuracy_delta_pp": 16.7,
             "cluster_job_id": "12345", "wall_clock_s": 900},
        ]}))
    ev = ExternalEvaluation.from_json(p)
    assert ev.incumbent_token == "tok9"
    r = ev(_Arm("a/b/c"))
    assert r is not None and r.net == 2 and r.cases_fired == 5
    assert ev(_Arm("not/recorded")) is None


def test_a_bare_list_of_results_also_loads(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps([{"arm_label": "x", "gains": ["c1"], "losses": [], "n": 4}]))
    ev = ExternalEvaluation.from_json(p)
    assert ev(_Arm("x")) is not None


# ================================================================================================
# IT DRIVES THE REAL OPTIMIZER -- no second optimizer exists
# ================================================================================================

def test_optimize_selects_the_argmax_over_EXTERNALLY_supplied_results():
    """End to end through `AnchorPolicyOpt.optimize`, the same entry point the toy host uses inline.

    The winner is the J_train argmax of the recorded results, and the arm with NO result is not ranked
    at all -- it cannot win by default and cannot be reported as a null.
    """
    from anchoropt.anchor import Action, IncisionPoint
    from anchoropt.learning.anchor_policy_opt import (
        AnchorPolicyOpt, FrozenIncumbent, SearchSpaceProposal,
    )
    from anchoropt.runtime import HostProfile

    GATE = IncisionPoint.POST_GENERATION_PRE_EXEC

    class _RT:
        HOST = HostProfile(name="ext", executable={GATE: frozenset({Action.REPROMPT})})

        def declared_signals(self):
            return ("s1",)

        def evaluate_signal(self, signal, state, params=None):
            return True

        def ground_reprompt(self, signal, boundary):
            return [{"variant": "v_low", "eta": {"instruction": "a", "retry_budget": 1},
                     "detail": ""},
                    {"variant": "v_high", "eta": {"instruction": "b", "retry_budget": 1},
                     "detail": ""},
                    {"variant": "v_none", "eta": {"instruction": "c", "retry_budget": 1},
                     "detail": ""}]

    rt = _RT()
    prop = SearchSpaceProposal(boundary=GATE, signal="s1", action_set=(Action.REPROMPT,),
                               diagnosis_case_ids=("c1",))
    opt = AnchorPolicyOpt(runtime=rt, host=rt.HOST)
    arms, _rej = opt.build_arms(prop)
    labels = {a.instantiated.variant: a.label for a in arms}
    assert len(labels) == 3

    ev = ExternalEvaluation(results={
        labels["v_low"]: _res(labels["v_low"], gains=("c1",)),
        labels["v_high"]: _res(labels["v_high"], gains=("c1", "c2", "c3")),
        # v_none deliberately has NO recorded result.
    })
    res = opt.optimize(prop, incumbent=FrozenIncumbent("P0", "tok"), evaluate=ev)

    assert res.winner is not None
    assert res.winner.instantiated.variant == "v_high", "the measured argmax must win"
    assert ev.n_completed == 2, "only the two recorded arms were evaluated"
    assert any("v_none" in m for m in ev.missing)
    # A train win is PROVISIONAL: criterion 1 of four, never acceptance.
    assert ev.outcome_class(improved=True) == TRAIN_IMPROVED_PENDING_VALIDATION


def test_with_NO_external_results_optimize_names_NO_winner():
    """The honest outcome when nothing was measured: no winner, and UNEVALUATED.

    This is the case the 25% heuristic used to paper over -- it named a winner from firing behaviour
    and the round then reported NO_BENEFIT about arms nobody ran.
    """
    from anchoropt.anchor import Action, IncisionPoint
    from anchoropt.learning.anchor_policy_opt import (
        AnchorPolicyOpt, FrozenIncumbent, SearchSpaceProposal,
    )
    from anchoropt.runtime import HostProfile

    GATE = IncisionPoint.POST_GENERATION_PRE_EXEC

    class _RT:
        HOST = HostProfile(name="ext2", executable={GATE: frozenset({Action.REPROMPT})})

        def declared_signals(self):
            return ("s1",)

        def evaluate_signal(self, signal, state, params=None):
            return True

        def ground_reprompt(self, signal, boundary):
            return [{"variant": "v", "eta": {"instruction": "a", "retry_budget": 1}, "detail": ""}]

    rt = _RT()
    prop = SearchSpaceProposal(boundary=GATE, signal="s1", action_set=(Action.REPROMPT,),
                               diagnosis_case_ids=("c1",))
    ev = ExternalEvaluation(results={})
    res = AnchorPolicyOpt(runtime=rt, host=rt.HOST).optimize(
        prop, incumbent=FrozenIncumbent("P0", "tok"), evaluate=ev)

    assert res.arms, "arms must still be BUILT -- that is the structural result"
    assert res.winner is None, "no measurement means no winner may be named"
    assert ev.outcome_class(improved=False) == UNEVALUATED


# ================================================================================================
# THE 25% HEURISTIC IS GONE FROM THE SUPPORTED PATH
# ================================================================================================

def test_the_driver_no_longer_selects_by_firing_rate():
    """EXECUTABLE code must not rank arms by distance from a firing rate.

    Checked against the AST, not the text, so the comment recording WHY the heuristic was removed does
    not trip the guard -- the same reasoning `test_cycle2_thinness.py` uses. A prose-based check would
    forbid documenting the property it enforces.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "scripts" / "self_evolve_cycle2.py"
    tree = ast.parse(src.read_text())

    # The heuristic was `abs(n / len(fv) - 0.25)`. Any float literal near a firing-rate computation is
    # the shape to refuse; 0.25 specifically must not appear in executable code at all.
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert 0.25 not in literals, \
        "the firing-rate target is back in executable code -- selection must come from measurement"

    # And `abs()` must not be applied to anything involving a firing vector length.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "abs":
            seg = ast.dump(node)
            assert "firing" not in seg.lower() and "fv" not in seg, \
                f"a firing-rate distance heuristic is back: {seg[:160]}"


def test_the_driver_does_not_drive_the_optimizer_itself():
    """The search belongs to core. A driver that builds proposals can steer it.

    This duplicates `test_cycle2_thinness.py`'s rule deliberately, from the evaluation side: when the
    `--results` path was first wired it constructed `SearchSpaceProposal`s inline and that guard caught
    it. The selection logic moved into `select_on_measurement` instead.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "scripts" / "self_evolve_cycle2.py"
    names = set()
    for node in ast.walk(ast.parse(src.read_text())):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
    for forbidden in ("AnchorPolicyOpt", "SearchSpaceProposal", "FrozenIncumbent"):
        assert forbidden not in names, \
            f"the driver uses {forbidden!r} directly; selection belongs in core"
    assert "select_on_measurement" in names, "it must call core's selection entry point"


# ================================================================================================
# `select_on_measurement` -- core's selection entry point
# ================================================================================================

def _toy_selection_runtime():
    from anchoropt.anchor import Action, IncisionPoint
    from anchoropt.runtime import HostProfile

    GATE = IncisionPoint.POST_GENERATION_PRE_EXEC

    class _RT:
        HOST = HostProfile(name="sel", executable={GATE: frozenset({Action.REPROMPT})})

        def declared_signals(self):
            return ("s1",)

        def evaluate_signal(self, signal, state, params=None):
            return True

        def ground_reprompt(self, signal, boundary):
            return [{"variant": "lo", "eta": {"instruction": "a", "retry_budget": 1}, "detail": ""},
                    {"variant": "hi", "eta": {"instruction": "b", "retry_budget": 1}, "detail": ""}]

    return _RT(), GATE, Action


def _built_arms(rt, GATE, Action):
    from anchoropt.learning.anchor_policy_opt import AnchorPolicyOpt, SearchSpaceProposal
    prop = SearchSpaceProposal(boundary=GATE, signal="s1", action_set=(Action.REPROMPT,),
                               diagnosis_case_ids=("c1",))
    arms, _ = AnchorPolicyOpt(runtime=rt, host=rt.HOST).build_arms(prop)
    return arms


def test_select_on_measurement_picks_the_measured_argmax():
    rt, GATE, Action = _toy_selection_runtime()
    from anchoropt.learning.external_evaluation import select_on_measurement
    arms = _built_arms(rt, GATE, Action)
    labels = {a.instantiated.variant: a.label for a in arms}

    ev = ExternalEvaluation(results={
        labels["lo"]: _res(labels["lo"], gains=("c1",)),
        labels["hi"]: _res(labels["hi"], gains=("c1", "c2", "c3")),
    })
    sel = select_on_measurement(arms, runtime=rt, host=rt.HOST, incumbent_id="P0",
                               incumbent_token="tok", evaluation=ev, case_ids=("c1",))
    assert sel.improved and sel.winner.instantiated.variant == "hi"
    assert sel.outcome_class == TRAIN_IMPROVED_PENDING_VALIDATION
    # Selection establishes the TRAIN criterion and must never claim acceptance: criteria 2-4 run at
    # the protocol's checkpoint, on data this path may not see.
    assert sel.improved and not sel.accepted
    assert sel.as_dict()["accepted"] is False
    assert sel.is_negative_result is False
    assert sel.evaluations_completed == 2


def test_select_on_measurement_reports_UNEVALUATED_with_no_results():
    rt, GATE, Action = _toy_selection_runtime()
    from anchoropt.learning.external_evaluation import select_on_measurement
    arms = _built_arms(rt, GATE, Action)

    sel = select_on_measurement(arms, runtime=rt, host=rt.HOST, incumbent_id="P0",
                               incumbent_token="tok", evaluation=ExternalEvaluation(results={}))
    assert sel.winner is None, "no measurement means no winner"
    assert sel.outcome_class == UNEVALUATED
    assert sel.is_negative_result is False, "UNEVALUATED is NOT a negative result"
    assert "not evidence about them" in sel.detail()


def test_select_on_measurement_distinguishes_NO_BENEFIT_from_BUDGET():
    rt, GATE, Action = _toy_selection_runtime()
    from anchoropt.learning.external_evaluation import select_on_measurement
    arms = _built_arms(rt, GATE, Action)
    labels = [a.label for a in arms]

    # Every arm measured, none helps -> a real negative result.
    all_measured = ExternalEvaluation(results={lbl: _res(lbl, losses=("c9",)) for lbl in labels})
    sel = select_on_measurement(arms, runtime=rt, host=rt.HOST, incumbent_id="P0",
                               evaluation=all_measured)
    assert sel.outcome_class == EVALUATED_NO_BENEFIT
    assert sel.is_negative_result is True

    # One arm measured, one left open -> the round is not settled.
    partial = ExternalEvaluation(results={labels[0]: _res(labels[0], losses=("c9",))})
    sel2 = select_on_measurement(arms, runtime=rt, host=rt.HOST, incumbent_id="P0",
                                evaluation=partial)
    assert sel2.outcome_class == BUDGET_EXHAUSTED
    assert sel2.is_negative_result is False, "arms left unmeasured are not a null"


def test_the_selection_record_is_serializable_for_a_driver():
    rt, GATE, Action = _toy_selection_runtime()
    from anchoropt.learning.external_evaluation import select_on_measurement
    arms = _built_arms(rt, GATE, Action)
    sel = select_on_measurement(arms, runtime=rt, host=rt.HOST, incumbent_id="P0",
                               evaluation=ExternalEvaluation(results={}))
    blob = json.dumps(sel.as_dict(), default=str)
    back = json.loads(blob)
    assert back["outcome"] == UNEVALUATED and back["is_negative_result"] is False


# ================================================================================================
# INCUMBENT IDENTITY IS A PRECONDITION -- the cg272 defect class
# ================================================================================================
#
# `incumbent_token` was carried on every ArmResult and on every manifest, and nothing compared them.
# A real, internally sound paired run measured against a harness that already carried two controllers
# (55/89) was therefore reported as an improvement on a raw baseline of 19/89. The arm was not at
# fault and the pairing was not broken -- the measurement simply answered a question about a
# different incumbent. See rounds/CG272_AUDIT/audit.json.

def test_a_result_measured_against_ANOTHER_incumbent_is_not_served():
    ev = ExternalEvaluation(incumbent_token="raw_h0",
                            results={"a": _res("a", gains=("c1", "c2"),
                                               incumbent_token="already_controlled")})
    assert ev(_Arm("a")) is None, "a result from a different incumbent must not be scored"
    assert ev.incomparable == ["a"]
    assert ev.n_completed == 0


def test_incomparable_is_NOT_a_negative_result():
    """The distinction that matters: nothing was learned about this candidate."""
    ev = ExternalEvaluation(incumbent_token="raw_h0",
                            results={"a": _res("a", incumbent_token="other")})
    ev(_Arm("a"))
    assert ev.outcome_class(improved=False) == INCOMPARABLE_INCUMBENT
    assert ev.outcome_class(improved=False) != EVALUATED_NO_BENEFIT
    assert ev.outcome_class(improved=False) != UNEVALUATED


def test_a_matching_token_is_served_normally():
    ev = ExternalEvaluation(incumbent_token="raw_h0",
                            results={"a": _res("a", gains=("c1",), incumbent_token="raw_h0")})
    assert ev(_Arm("a")) is not None
    assert not ev.incomparable


def test_an_ABSENT_token_is_unverifiable_not_a_mismatch():
    """Older manifests predate the field. Voiding them would rewrite correctly-scored results."""
    ev = ExternalEvaluation(incumbent_token="raw_h0", results={"a": _res("a", gains=("c1",))})
    assert ev(_Arm("a")) is not None, "an empty token is unverifiable, not a mismatch"
    assert not ev.incomparable
    ev2 = ExternalEvaluation(results={"b": _res("b", incumbent_token="whatever")})
    assert ev2(_Arm("b")) is not None, "no round token means nothing to compare against"


def test_strict_mode_RAISES_on_a_mismatched_incumbent():
    ev = ExternalEvaluation(incumbent_token="raw_h0", strict=True,
                            results={"a": _res("a", incumbent_token="other")})
    with pytest.raises(MissingEvaluation):
        ev(_Arm("a"))


def test_a_train_win_is_never_reported_as_accepted():
    """Criterion 1 of four. Nothing in the selection path may assert acceptance."""
    ev = ExternalEvaluation(results={"a": _res("a", gains=("c1", "c2"))})
    ev(_Arm("a"))
    assert ev.outcome_class(improved=True) == TRAIN_IMPROVED_PENDING_VALIDATION
    assert "ACCEPT" not in TRAIN_IMPROVED_PENDING_VALIDATION.upper().replace("PENDING", "")
