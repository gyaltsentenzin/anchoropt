"""The round's bookkeeping: the four outcome classes, kept apart, on the real driver.

Collapsing any pair of UNEVALUATED / NO_BENEFIT / BUDGET_EXHAUSTED / IMPROVED is how a null gets
believed. These run the ACTUAL propose and select phases of `run_anchoropt_round.py` against a synthetic
baseline, so no model is called and nothing is mocked except the episodes themselves.

The baseline here is synthetic; the trajectories are built by the same `tau2_state` builders the live
mechanism uses, so the events are shaped exactly like a recorded run's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "tau2"))
sys.path.insert(0, str(REPO))

import tau2_state as S                                              # noqa: E402
from tau2_runtime import ADAPTER                                    # noqa: E402

import importlib.util                                               # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "tau2_round_driver", REPO / "benchmarks" / "tau2" / "run_anchoropt_round.py")
DRIVER = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DRIVER)

CATALOG = json.loads((REPO / "benchmarks" / "tau2" / "catalogs" / "airline.json").read_text())
KNOWN = frozenset(CATALOG)
MUTATING = frozenset(k for k, v in CATALOG.items() if v["mutates_state"])


def _episode(cid: str, *, failing: bool) -> list[dict]:
    ev, turn, terr, cons = [], 0, 0, 0
    for i in range(3):
        turn += 1
        ev.append(dict(S.turn_start_state(
            case_id=cid, turn_index=turn, assistant_turns_so_far=turn - 1, user_turns_so_far=1,
            inbound_is_tool_result=turn > 1, consecutive_tool_errors=cons,
            tool_errors_so_far=terr), kind="turn_start"))
        args = {"reservation_id": "ABC123"}
        if failing:
            args["payload"] = "x" * 60
        calls = [{"name": "update_reservation_flights", "arguments": args}] if i < 2 else []
        ev.append(dict(S.gate_state(
            case_id=cid, turn_index=turn, proposed_calls=calls, has_content=not calls,
            known_tools=KNOWN, mutating_tools=MUTATING, replan_attempt=0,
            consecutive_tool_errors=cons, tool_errors_so_far=terr), kind="propose"))
        if calls:
            is_err = failing and i < 2
            if is_err:
                terr += 1
                cons += 1
            else:
                cons = 0
            ev.append(dict(S.result_state(
                case_id=cid, turn_index=turn, is_error=is_err,
                content="Error: Reservation ABC123 not found" if is_err else '{"ok": true}',
                tool_name="update_reservation_flights", mutating_tools=MUTATING,
                tool_errors_so_far=terr), kind="result"))
    return ev


@pytest.fixture
def round_dir(tmp_path):
    """A synthetic frozen incumbent: 10 cases, 4 solved, 6 failing."""
    ADAPTER.reset_expanded_signals()
    ADAPTER.reset_instruction_proposals()
    events, reward, solved, term = {}, {}, {}, {}
    for i in range(10):
        cid = f"t{i}"
        failing = i < 6
        events[cid] = _episode(cid, failing=failing)
        reward[cid] = 0.0 if failing else 1.0
        solved[cid] = not failing
        term[cid] = "agent_stop"
    (tmp_path / "baseline.json").write_text(json.dumps({
        "domain": "airline", "agent_llm": "synthetic", "user_llm": "synthetic", "max_steps": 40,
        "task_ids": sorted(events),
        "run": {"label": "P0", "n": 10, "n_solved": 4, "mean_reward": 0.4, "reward": reward,
                "solved": solved, "termination": term, "firings": {}},
        "events": events, "tool_catalog": CATALOG, "adapter": "tau2"}, indent=2))
    yield tmp_path
    ADAPTER.reset_expanded_signals()
    ADAPTER.reset_instruction_proposals()


def _propose(round_dir, results=None, teacher="", teacher_n=3):
    rc = DRIVER.cmd_propose(SimpleNamespace(out=str(round_dir), rank=1, results=results,
                                            teacher=teacher, teacher_n=teacher_n))
    assert rc == 0
    return json.loads((round_dir / "arm_manifest.json").read_text())


def _select(round_dir, eval_budget=None, results_name="results.json"):
    rc = DRIVER.cmd_select(SimpleNamespace(out=str(round_dir), eval_budget=eval_budget,
                                           results_name=results_name))
    assert rc == 0
    return json.loads((round_dir / "selection.json").read_text())


def _results(round_dir, rows, token=""):
    (round_dir / "results.json").write_text(json.dumps({
        "incumbent_id": "P0", "incumbent_token": token, "results": rows}, indent=2))


# ============================================================================ propose names no winner
def test_propose_emits_every_arm_and_names_no_winner(round_dir):
    man = _propose(round_dir)
    assert man["state"] == "REALIZABLE_UNMEASURED"
    assert man["n_arms"] > 0
    assert "selected_arm" not in man, "the propose phase named a winner"
    assert "next_arm" not in man, "the firing-rate heuristic's last vestige is back"
    assert all(r["arm_label"] for r in man["arms"])


def test_propose_localizes_more_than_one_boundary(round_dir):
    """Exactly one boundary means the answer-commitment rows were dropped."""
    man = _propose(round_dir)
    assert len({r["boundary"] for r in man["arms"]}) > 1, man["arms"][:2]


def test_propose_builds_no_arm_at_the_disabled_cell(round_dir):
    man = _propose(round_dir)
    cells = {f"{r['boundary']}/{r['action']}" for r in man["arms"]}
    assert "post_execution/reroute" not in cells


def test_every_emitted_arm_reports_how_often_its_signal_fires(round_dir):
    """All-zero would mean the fireability figure is a lookup that matches nothing."""
    man = _propose(round_dir)
    fires = [r["fires_on_states"] for r in man["arms"]]
    assert any(f > 0 for f in fires), fires
    assert all(r["total_states"] > 0 for r in man["arms"])


# ============================================================================ the four outcome classes
def test_nothing_measured_is_UNEVALUATED_and_not_a_negative_result(round_dir):
    _propose(round_dir)
    _results(round_dir, [])
    sel = _select(round_dir)
    assert sel["outcome_class"] == "UNEVALUATED"
    assert sel["is_negative_result"] is False
    assert sel["best_measured_arm"] is None
    assert sel["promoted"] is None
    assert sel["controller_spec"] == {}


def test_arms_left_unmeasured_is_BUDGET_EXHAUSTED_not_NO_BENEFIT(round_dir):
    """The sharpest guard: a genuinely measured LOSS still does not settle a round with arms open."""
    man = _propose(round_dir)
    first = man["arms"][0]["arm_label"]
    _results(round_dir, [{"arm_label": first, "gains": [], "losses": ["t0", "t1"],
                          "firings": 6, "cases_fired": 6, "n": 10,
                          "interventions_executed": 6, "accuracy_delta_pp": -20.0}])
    sel = _select(round_dir)
    assert sel["outcome_class"] == "BUDGET_EXHAUSTED", sel
    assert sel["is_negative_result"] is False
    assert len(sel["unevaluated"]) == man["n_arms"] - 1
    # Core names the best MEASURED arm whatever its sign; that is not a promotion. Both facts must be
    # present and distinct, or a driver installs a controller the round measured as harmful.
    assert sel["best_measured_arm"] == first
    assert sel["best_measured_net"] == -2
    assert sel["improved"] is False
    assert sel["promoted"] is None
    assert sel["controller_spec"] == {}


def test_every_arm_measured_and_none_helping_is_NO_BENEFIT(round_dir):
    man = _propose(round_dir)
    _results(round_dir, [{"arm_label": r["arm_label"], "gains": [], "losses": ["t0"],
                          "firings": 6, "cases_fired": 6, "n": 10,
                          "interventions_executed": 6, "accuracy_delta_pp": -10.0}
                         for r in man["arms"]])
    sel = _select(round_dir)
    assert sel["outcome_class"] == "NO_BENEFIT", sel
    assert sel["is_negative_result"] is True
    assert sel["improved"] is False and sel["promoted"] is None


def test_a_measured_gain_is_a_train_win_pending_validation_and_is_installable(round_dir):
    man = _propose(round_dir)
    winner = man["arms"][0]["arm_label"]
    rows = [{"arm_label": r["arm_label"], "gains": [], "losses": ["t0"], "firings": 6,
             "cases_fired": 6, "n": 10, "interventions_executed": 6, "accuracy_delta_pp": -10.0}
            for r in man["arms"][1:]]
    rows.insert(0, {"arm_label": winner, "gains": ["t0", "t1", "t2"], "losses": [], "firings": 6,
                    "cases_fired": 6, "n": 10, "interventions_executed": 6,
                    "accuracy_delta_pp": 30.0})
    _results(round_dir, rows)
    sel = _select(round_dir)
    # Core's name for it, and the name matters: train net > 0 is criterion 1 of four, so the round has
    # a provisional winner rather than an accepted controller. `accepted` is always False here.
    assert sel["outcome_class"] == "TRAIN_IMPROVED_PENDING_VALIDATION", sel
    assert sel["improved"] is True
    assert sel["accepted"] is False
    assert sel["promoted"] == winner
    assert sel["best_measured_arm"] == winner
    spec = sel["controller_spec"]
    assert spec["boundary"] and spec["action"] and spec["signal"]
    assert spec["boundary"] in {"before_agent_turn", "before_tool_dispatch", "after_tool_result"}


# ============================================================================ an absent arm is not a zero
def test_an_unevaluated_arm_never_outranks_a_measured_one(round_dir):
    """A fabricated zero outranks a measured -2, so an arm nobody ran would be selected.

    Here EVERY measured arm is harmful. The correct answer is that no unmeasured arm is promoted --
    absence is not a measurement.
    """
    man = _propose(round_dir)
    measured = man["arms"][0]["arm_label"]
    _results(round_dir, [{"arm_label": measured, "gains": [], "losses": ["t0", "t1"],
                          "firings": 6, "cases_fired": 6, "n": 10,
                          "interventions_executed": 6, "accuracy_delta_pp": -20.0}])
    sel = _select(round_dir)
    assert sel["promoted"] is None, "a harmful arm was promoted"
    assert sel["improved"] is False
    assert sel["best_measured_arm"] == measured, (
        "the best measured arm must be the only one measured -- an omitted arm was scored")
    for label in sel["unevaluated"]:
        assert label != measured
    assert set(sel["served"]) == {measured}


def test_an_arm_with_zero_executions_is_not_promoted_over_one_that_ran(round_dir):
    """`train_objective` needs interventions_executed > 0 for an arm to count as engaged."""
    man = _propose(round_dir)
    inert, engaged = man["arms"][0]["arm_label"], man["arms"][1]["arm_label"]
    rows = [{"arm_label": r["arm_label"], "gains": [], "losses": ["t0"], "firings": 6,
             "cases_fired": 6, "n": 10, "interventions_executed": 6, "accuracy_delta_pp": -10.0}
            for r in man["arms"][2:]]
    rows += [
        {"arm_label": inert, "gains": ["t0"], "losses": [], "firings": 0, "cases_fired": 0,
         "n": 10, "interventions_executed": 0, "accuracy_delta_pp": 10.0},
        {"arm_label": engaged, "gains": ["t1"], "losses": [], "firings": 6, "cases_fired": 6,
         "n": 10, "interventions_executed": 6, "accuracy_delta_pp": 10.0},
    ]
    _results(round_dir, rows)
    sel = _select(round_dir)
    assert sel["promoted"] == engaged, (
        f"an arm that executed nothing was promoted over one that did: {sel['promoted']}")


# ============================================================================ propose/select agree
def test_select_rebuilds_exactly_the_arms_propose_emitted(round_dir):
    """The predicate that runs must be the one core installed, so both phases must agree."""
    man = _propose(round_dir)
    ADAPTER.reset_expanded_signals()
    base = json.loads((round_dir / "baseline.json").read_text())
    arms = DRIVER._rebuild_arms(round_dir, base, man, results=man.get("replayed_results"))
    assert {a.label for a in arms} == {r["arm_label"] for r in man["arms"]}


# ============================================================================ the key matches its evidence
def test_a_residual_key_never_claims_an_error_the_trajectory_does_not_contain():
    """The decision key is what the round is grouped and reported by, so it must match the evidence.

    An earlier version of the miner reported "while a tool call was still failing" for cases whose own
    evidence string said "none errored" -- the round's headline named a failing call on a residual with
    zero errored results, and the two arms conditioning on an error could never fire on it.
    """
    import tau2_attribution as ATTR

    clean = _episode("c1", failing=False)
    d = ATTR.diagnose_case("c1", clean, termination="agent_stop", reward=0.0)
    assert d is not None
    assert "failing" not in d.consequential_decision, d.consequential_decision
    assert "none errored" in d.evidence
    assert d.metadata["n_errored"] == 0

    failing = _episode("c2", failing=True)
    d2 = ATTR.diagnose_case("c2", failing, termination="agent_stop", reward=0.0)
    assert d2.metadata["n_errored"] > 0
    assert "failing" in d2.consequential_decision or "returned" in d2.consequential_decision


def test_every_diagnosis_mentioning_an_error_has_one_in_its_metadata():
    import tau2_attribution as ATTR
    for failing in (True, False):
        for term in ("agent_stop", "TerminationReason.MAX_STEPS"):
            ev = _episode("c", failing=failing)
            d = ATTR.diagnose_case("c", ev, termination=term, reward=0.0)
            claims_error = ("failing" in d.consequential_decision
                            or "returned" in d.consequential_decision)
            if claims_error:
                assert d.metadata["n_errored"] > 0, (d.consequential_decision, d.metadata)


# ============================================================================ cycle 2 expands Phi
def test_replaying_measurements_exhausts_the_shipped_vocabulary_and_expands_phi(round_dir):
    """Signal expansion is driven by MEASURED exhaustion, not by asking for it.

    Cycle 1 with no evaluator legitimately stops at REALIZABLE_UNMEASURED without expanding. Feeding
    cycle 1's real measurements back -- every shipped-Phi arm measured as harmful -- exhausts the
    boundary, and only then does core synthesize new conditions over the typed fields.
    """
    man1 = _propose(round_dir)
    assert man1["signals_installed"] == [], (
        "cycle 1 expanded Phi without any measurement to exhaust it")

    _results(round_dir, [{"arm_label": r["arm_label"], "gains": [], "losses": ["t0"],
                          "firings": 6, "cases_fired": 6, "n": 10,
                          "interventions_executed": 6, "accuracy_delta_pp": -10.0}
                         for r in man1["arms"]])

    ADAPTER.reset_expanded_signals()
    man2 = _propose(round_dir, results=str(round_dir / "results.json"))
    assert man2["signals_installed"], (
        "every shipped-Phi arm measured as harmful and Phi was still not expanded")
    shipped = set(ADAPTER.declared_signals()) - set(man2["signals_installed"])
    new_signal_arms = [r for r in man2["arms"] if r["signal"] not in shipped]
    assert new_signal_arms, "Phi expanded but no arm was built on a synthesized condition"
    # A synthesized condition must still be one the boundary can evaluate.
    for r in new_signal_arms[:5]:
        assert r["fires_on_states"] >= 0 and r["total_states"] > 0


def test_a_cycle_two_manifest_records_which_measurements_it_replayed(round_dir):
    """Provenance: a manifest built on replayed measurements must say so, or `select` rebuilds
    different arms than the ones that were emitted."""
    man1 = _propose(round_dir)
    _results(round_dir, [{"arm_label": r["arm_label"], "gains": [], "losses": ["t0"], "firings": 6,
                          "cases_fired": 6, "n": 10, "interventions_executed": 6,
                          "accuracy_delta_pp": -10.0} for r in man1["arms"]])
    ADAPTER.reset_expanded_signals()
    man2 = _propose(round_dir, results=str(round_dir / "results.json"))
    assert man2["replayed_results"] == str(round_dir / "results.json")
    ADAPTER.reset_expanded_signals()
    base = json.loads((round_dir / "baseline.json").read_text())
    rebuilt = DRIVER._rebuild_arms(round_dir, base, man2, results=man2["replayed_results"])
    assert {a.label for a in rebuilt} == {r["arm_label"] for r in man2["arms"]}


# ============================================================================ the teacher changes arms
def test_history_only_and_history_plus_teacher_produce_different_arm_sets(round_dir, monkeypatch):
    """Rule 7: the comparison must be measurable, which means the two configs cannot be identical.

    The teacher is injected here, so this needs no network.
    """

    import tau2_teacher as TEACH

    man_plain = _propose(round_dir)
    plain = {r["arm_label"] for r in man_plain["arms"]}
    assert man_plain["teacher"]["invoked"] is False
    assert man_plain["teacher"]["provenance"] == "history_only"

    proposals = [
        {"variant": "confirm_applied",
         "instruction": "Confirm the requested change is actually in the record before you tell the "
                        "customer it is done."},
        {"variant": "read_back",
         "instruction": "Read the updated details back to the customer before closing the request."},
    ]

    def fake_propose(problem, *, endpoint, provenance, boundary_fields, known_tools=frozenset(),
                     n=3, max_evidence=6, complete=None):
        return TEACH.TeacherResult(
            invoked=True, reason="injected", model="fake", provenance=provenance,
            proposals=[TEACH.TeacherProposal(variant=p["variant"], instruction=p["instruction"],
                                             model="fake", provenance=provenance)
                       for p in proposals])

    monkeypatch.setattr(TEACH, "propose_instructions", fake_propose)
    ADAPTER.reset_expanded_signals()
    man_teach = _propose(round_dir, teacher="claude-sonnet-5")
    taught = {r["arm_label"] for r in man_teach["arms"]}

    assert man_teach["teacher"]["invoked"] is True
    assert man_teach["teacher"]["provenance"] == "teacher:claude-sonnet-5"
    assert taught > plain, "the teacher added no arm, so the two configs are indistinguishable"
    added = taught - plain
    assert added and all("teacher_" in lbl for lbl in added), added
    # The teacher supplies eta, never a locus or an action: its arms must land on reprompt cells only.
    for row in man_teach["arms"]:
        if "teacher_" in row["arm_label"]:
            assert row["action"] == "reprompt", row


def test_a_teacher_arms_eta_is_restricted_to_what_that_cells_executor_reads(round_dir, monkeypatch):
    """A teacher proposal carrying a key the cell does not consume would be refused as inert eta."""
    import tau2_teacher as TEACH

    def fake_propose(problem, *, endpoint, provenance, boundary_fields, known_tools=frozenset(),
                     n=3, max_evidence=6, complete=None):
        return TEACH.TeacherResult(
            invoked=True, reason="injected", model="fake", provenance=provenance,
            proposals=[TEACH.TeacherProposal(
                variant="v", instruction="Confirm the change is in the record before you say it is "
                                         "done to the customer.", model="fake",
                provenance=provenance)])

    monkeypatch.setattr(TEACH, "propose_instructions", fake_propose)
    man = _propose(round_dir, teacher="claude-sonnet-5")
    from anchoropt.anchor import Action
    for row in man["arms"]:
        if "teacher_" not in row["arm_label"]:
            continue
        cap = ADAPTER.executor_capability(row["boundary"], Action.REPROMPT)
        assert set(row["eta"]) <= set(cap.consumes), (
            f"{set(row['eta']) - set(cap.consumes)} is inert eta at {row['boundary']}")


def test_a_teacher_rounds_arms_can_be_rebuilt_by_a_later_phase(round_dir, monkeypatch):
    """`evaluate` and `select` run in separate processes and rebuild the arm set.

    Without the teacher's groundings persisted in the manifest, the rebuilt set lacks every teacher arm
    -- so the manifest would list arms that could never be measured, and `select` would optimize over a
    different set than the one emitted.
    """
    import tau2_teacher as TEACH

    proposals = [TEACH.TeacherProposal(
        variant="confirm_applied",
        instruction="Confirm the requested change is in the record before you tell the customer it "
                    "is done.", model="fake", provenance="teacher:x")]

    def fake_propose(problem, *, endpoint, provenance, boundary_fields, known_tools=frozenset(),
                     n=3, max_evidence=6, complete=None):
        return TEACH.TeacherResult(invoked=True, reason="injected", model="fake",
                                   provenance=provenance, proposals=proposals)

    monkeypatch.setattr(TEACH, "propose_instructions", fake_propose)
    man = _propose(round_dir, teacher="claude-sonnet-5")
    assert man["teacher"]["groundings"], "the groundings were not persisted"

    ADAPTER.reset_expanded_signals()
    ADAPTER.reset_instruction_proposals()
    base = json.loads((round_dir / "baseline.json").read_text())
    rebuilt = DRIVER._rebuild_arms(round_dir, base, man, results=man.get("replayed_results"))
    assert {a.label for a in rebuilt} == {r["arm_label"] for r in man["arms"]}
    assert any("teacher_" in a.label for a in rebuilt)


# ============================================================ a winner that never fired
def test_a_positive_net_from_an_arm_that_never_fired_is_flagged_not_called_a_win(round_dir):
    """`train_objective` orders by net first and engagement second, so when few arms are measured a
    run-to-run flip can be promoted by an arm whose controller executed nothing -- there is nothing for
    the engagement tiebreak to outrank. Observed live: qwen/telecom measured net +1 with
    interventions_executed=0 on a 1-case residual, which is exactly the benchmark's variance floor.

    Core's ordering is core's business; what this driver must not do is call it a win.
    """
    man = _propose(round_dir)
    winner = man["arms"][0]["arm_label"]
    _results(round_dir, [{"arm_label": winner, "gains": ["t0"], "losses": [], "firings": 0,
                          "cases_fired": 0, "n": 10, "interventions_executed": 0,
                          "accuracy_delta_pp": 10.0}])
    sel = _select(round_dir)
    assert sel["improved"] is True, "core still reports a positive train net"
    assert sel["winner_interventions_executed"] == 0
    assert sel["winner_engaged"] is False, "an arm that executed nothing must not read as engaged"

    driver_src = (REPO / "benchmarks" / "tau2" / "run_anchoropt_round.py").read_text()
    assert "NEVER FIRED" in driver_src
    assert "run-to-run variance, not a repair" in driver_src


def test_an_engaged_winner_is_still_reported_as_engaged(round_dir):
    man = _propose(round_dir)
    winner = man["arms"][0]["arm_label"]
    _results(round_dir, [{"arm_label": winner, "gains": ["t0"], "losses": [], "firings": 6,
                          "cases_fired": 6, "n": 10, "interventions_executed": 6,
                          "accuracy_delta_pp": 10.0}])
    sel = _select(round_dir)
    assert sel["winner_engaged"] is True
    assert sel["winner_interventions_executed"] == 6


def test_results_are_flushed_after_every_arm_not_once_at_the_end():
    """Per-arm runs were durable but the round manifest was written only after the whole loop, so a
    killed evaluate discarded every measurement it had made -- and `select` and `--resume` both read
    that file. One arm had already measured a clean net +1 when this was found."""
    src = (REPO / "benchmarks" / "tau2" / "run_anchoropt_round.py").read_text()
    assert "def _flush_results()" in src
    assert src.count("_flush_results()") >= 3, "must flush per measured arm, per void, and at the end"


# ============================================================ the held-out validate phase
def test_validate_refuses_a_round_with_no_train_winner(round_dir):
    """Criterion 2 does not arise if criterion 1 was not met."""
    man = _propose(round_dir)
    _results(round_dir, [{"arm_label": r["arm_label"], "gains": [], "losses": ["t0"], "firings": 6,
                          "cases_fired": 6, "n": 10, "interventions_executed": 6,
                          "accuracy_delta_pp": -10.0} for r in man["arms"]])
    _select(round_dir)
    rc = DRIVER.cmd_validate(SimpleNamespace(out=str(round_dir), split="test", concurrency=2,
                                             portal="", resume=False, k=3, p0_only=False,
                                             call_timeout=180, retries=1))
    assert rc == 0
    assert not (round_dir / "validation_test.json").exists()


def test_validate_refuses_a_train_winner_that_never_fired(round_dir):
    """Validating a variance flip would launder it into a result."""
    man = _propose(round_dir)
    winner = man["arms"][0]["arm_label"]
    _results(round_dir, [{"arm_label": winner, "gains": ["t0"], "losses": [], "firings": 0,
                          "cases_fired": 0, "n": 10, "interventions_executed": 0,
                          "accuracy_delta_pp": 10.0}])
    sel = _select(round_dir)
    assert sel["improved"] is True and sel["winner_engaged"] is False
    rc = DRIVER.cmd_validate(SimpleNamespace(out=str(round_dir), split="test", concurrency=2,
                                             portal="", resume=False, k=3, p0_only=False,
                                             call_timeout=180, retries=1))
    assert rc == 5, "a zero-execution winner must not be validated"
    assert not (round_dir / "validation_test.json").exists()


def test_validate_uses_criterion_2s_actual_threshold_and_claims_no_acceptance():
    """docs/ACCEPTANCE_RULE.md: criterion 2 is `net dev >= 0` -- NO AGGREGATE REGRESSION -- not a
    positive net. And criteria 3 (in full) and 4 are not decided by this phase."""
    src = (REPO / "benchmarks" / "tau2" / "run_anchoropt_round.py").read_text()
    assert "no aggregate regression, net >= 0" in src
    assert 'pr["net"] >= 0' in src
    assert '"accepted": False' in src
    # criterion 3 evidence: gains must be split by whether the controller fired on that case
    assert "criterion_3_gains_attributable" in src
    assert "criterion_3_gains_not_attributable" in src


def test_validate_defaults_to_the_train_rounds_portal():
    """An incumbent and an arm on different portals attribute a portal effect to the controller; the
    same applies across splits."""
    src = (REPO / "benchmarks" / "tau2" / "run_anchoropt_round.py").read_text()
    assert 'portal=(a.portal or base.get("agent_provider") or "")' in src


# ============================================================ the incumbent is a distribution
def test_rescore_compares_arms_to_the_incumbent_distribution_not_one_draw(round_dir):
    """Every arm was paired against ONE incumbent run and the winner was the max of the measured arms.
    For qwen3.6/airline the incumbent drew 16/30 while repeats of the same policy averaged 76%, so all
    8 arms looked positive. Rescoring against replicates must expose that."""
    import json as _json

    man = _propose(round_dir)
    base = _json.loads((round_dir / "baseline.json").read_text())
    cases = base["task_ids"]
    # incumbent drew LOW (4/10); three replicates of the same policy score 7-8/10
    for k, solved_ids in enumerate((set(cases[:7]), set(cases[:8]), set(cases[1:8])), start=1):
        (round_dir / f"incumbent_rep_{k}.json").write_text(_json.dumps({
            "solved": {c: c in solved_ids for c in cases},
            "termination": {c: "TerminationReason.USER_STOP" for c in cases}}))
    # an arm that is merely a TYPICAL draw of the incumbent: 7/10
    arm_solved = {c: c in set(cases[:7]) for c in cases}
    (round_dir / "arm_01.json").write_text(_json.dumps({
        "label": man["arms"][0]["arm_label"], "solved": arm_solved,
        "termination": {c: "TerminationReason.USER_STOP" for c in cases},
        "firings": {"interventions_executed": 3, "cases_fired": cases[:3]}}))
    rc = DRIVER.cmd_rescore(SimpleNamespace(out=str(round_dir)))
    assert rc == 0
    rep = _json.loads((round_dir / "rescore.json").read_text())
    row = rep["arms"][0]
    assert row["old_net"] > 0, "against the single low draw the arm looks like a win"
    assert row["above_best_incumbent"] is False, "but it is inside the incumbent's own range"
    assert abs(row["expected_net"]) < row["old_net"], "and the expected net shrinks toward zero"


def test_rescore_refuses_without_enough_replicates(round_dir):
    _propose(round_dir)
    with pytest.raises(SystemExit):
        DRIVER.cmd_rescore(SimpleNamespace(out=str(round_dir)))


# ============================================================ pass^k, as tau-bench defines it
def test_pass_hat_k_matches_the_definition_at_its_endpoints():
    """pass^1 is the mean success rate over all runs; pass^n is the fraction of tasks solved every time."""
    cases = [f"t{i}" for i in range(5)]
    runs = [{"t0": 1, "t1": 1, "t2": 0, "t3": 1, "t4": 0},
            {"t0": 1, "t1": 0, "t2": 0, "t3": 1, "t4": 0},
            {"t0": 1, "t1": 1, "t2": 1, "t3": 1, "t4": 0},
            {"t0": 1, "t1": 1, "t2": 0, "t3": 0, "t4": 0}]
    got = DRIVER.pass_hat_k(runs, cases)
    assert set(got) == {1, 2, 3, 4}
    assert got[1] == pytest.approx(sum(sum(r.values()) for r in runs) / (4 * 5))
    assert got[4] == pytest.approx(1 / 5), "only t0 is solved in all four runs"
    assert got[1] >= got[2] >= got[3] >= got[4], "pass^k can only fall as k grows"


def test_pass_hat_k_equals_tau_benchs_own_function():
    tau2_metrics = pytest.importorskip("tau2.metrics.agent_metrics")
    import random

    rng = random.Random(7)
    cases = [str(i) for i in range(30)]
    runs = [{c: rng.random() < 0.55 for c in cases} for _ in range(4)]
    ours = DRIVER.pass_hat_k(runs, cases)
    for k in range(1, 5):
        theirs = sum(tau2_metrics.pass_hat_k(4, sum(r[c] for r in runs), k) for c in cases) / len(cases)
        assert ours[k] == pytest.approx(theirs, abs=1e-12)
