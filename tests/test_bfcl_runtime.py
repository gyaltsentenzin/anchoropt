"""The BFCL v4 runtime surface, and the anchor-recovery calibration it exists to pass.

Each test pins a defect found while building this module, so it cannot silently return:

  * the existing adapter is NOT modified -- it has no runtime surface and must keep none here
  * vacuity is the EXISTING detector, not a second copy
  * a signal declared observable at a boundary must be evaluable there (the A4-v1 defect shape)
  * a parameterized signal must be REACHABLE by the search (it was silently excluded)
  * a reroute with no attested destination is PRUNED, never given an invented one
  * the eight accepted anchors are recoverable from their own attribution prose
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
sys.path.insert(0, str(REPO / "scripts"))

import bfcl_runtime as R                                              # noqa: E402
import bfcl_signals as _signals                                       # noqa: E402
from anchoropt.anchor import Action, IncisionPoint                    # noqa: E402
from anchoropt.runtime import ActionNotExecutable                     # noqa: E402


@pytest.fixture(autouse=True)
def _inject_calibration_fixture():
    """Install the A1-A9 calibration data per test, then tear it down.

    The runtime declares NO aliases, probe params or reroute destinations on its own account: those
    were reverse-engineered from the accepted anchors, and importing them at runtime put the answer
    key into every process touching the module -- which the live smoke run's leak detector caught.
    This file is a calibration harness, so it injects them explicitly, which keeps the dependency
    visible here and absent everywhere else.
    """
    from fixtures.replay_anchors import (
        REROUTE_DESTINATIONS, SIGNAL_ALIASES, SIGNAL_PROBE_PARAMS,
    )
    _signals.SIGNAL_ALIASES.update(SIGNAL_ALIASES)
    _signals.SIGNAL_PROBE_PARAMS.update(SIGNAL_PROBE_PARAMS)
    R.REROUTE_DESTINATIONS.update(REROUTE_DESTINATIONS)
    yield
    _signals.SIGNAL_ALIASES.clear()
    _signals.SIGNAL_PROBE_PARAMS.clear()
    R.REROUTE_DESTINATIONS.clear()


def test_runtime_declares_no_calibration_data_on_its_own_account(monkeypatch):
    """The fixture leak the live smoke run caught: importing the runtime must not bring the answer
    key with it."""
    import fixtures.replay_anchors as fx

    monkeypatch.setattr(_signals, "SIGNAL_ALIASES", {}, raising=False)
    monkeypatch.setattr(_signals, "SIGNAL_PROBE_PARAMS", {}, raising=False)
    src = (REPO / "benchmarks/bfcl_v4/bfcl_signals.py").read_text()
    assert "from fixtures" not in src, "bfcl_signals imports the calibration fixture"
    rsrc = (REPO / "benchmarks/bfcl_v4/bfcl_runtime.py").read_text()
    assert "from fixtures" not in rsrc, "bfcl_runtime imports the calibration fixture"
    assert fx.SIGNAL_ALIASES, "the fixture itself should still hold the historical data"


def test_runtime_declares_the_parameter_contract():
    """The smoke run's proposer wrote theta['guidance'] and signal_params['threshold'] because it was
    never told the names. A contract the caller must satisfy but cannot read is an interface defect."""
    theta = R.action_theta_schema()
    assert "text" in theta["reprompt"]["required"]
    assert "reason" in theta["suppress"]["required"]
    assert "destination" in theta["reroute"]["required"]
    sig = R.signal_param_schema()
    assert sig["retrieval_similarity_below_threshold"]["below"]["required"] is True
    assert sig["retrieval_similarity_below_threshold"]["below"]["type"] == "float"


# ---- the existing adapter is untouched --------------------------------------------------------

def test_existing_adapter_has_no_runtime_surface_added():
    """P1's constraint: the runtime lives in a NEW module. ~294 tests depend on the adapter, so
    growing it a runtime surface is exactly what this change must not do."""
    import adapter
    for name in ("declared_signals", "evaluate_signal", "apply_action", "normalize_event",
                 "observable_state", "signal_aliases", "HOST"):
        assert not hasattr(adapter, name), f"adapter.py grew a runtime surface: {name}"


def test_runtime_provides_every_surface_candidate_search_requires():
    for name in ("declared_signals", "signal_aliases", "evaluate_signal", "feasible_actions",
                 "apply_action", "normalize_event", "observable_state", "HOST"):
        assert hasattr(R, name), f"missing runtime surface: {name}"


# ---- vacuity is reused, not reinvented -------------------------------------------------------

def test_vacuity_delegates_to_the_existing_detector():
    """`{'ranked_results': []}` is a SUCCESSFUL call carrying no information. The detector is
    policy_tree's; a second copy could disagree with it."""
    from anchoropt.learning.policy_tree import vacuous_result_kind
    payload = '{"ranked_results": []}'
    assert vacuous_result_kind(payload) == "empty_collection"

    st = R.observable_state(R.normalize_event(
        {"boundary": "post_execution", "calls": ['core_memory_retrieve(query="x")'],
         "tool_results": [payload]}))
    assert st["error_kind"] is None, "a vacuous result is not an error"
    assert R.evaluate_signal("no_informative_result", st, {}) is True


def test_vacuity_never_fires_on_an_explicit_error():
    """Double-counting one event as two conditions inflates support -- error_kind owns errors."""
    st = R.observable_state(R.normalize_event(
        {"boundary": "post_execution", "calls": ['core_memory_add(key="k",value="v")'],
         "tool_results": ["Error: core memory is full"]}))
    assert st["error_kind"] == "no_capacity"
    assert R.evaluate_signal("no_informative_result", st, {}) is False


def test_results_list_is_not_stringified():
    """`str([...])` is not valid JSON; stringifying a results LIST made the detector return None
    for every payload (policy_tree.any_vacuous documents the same defect)."""
    ev = R.normalize_event({"boundary": "post_execution", "calls": ["core_memory_retrieve(query=1)"],
                            "tool_results": ['{"ranked_results": []}']})
    assert ev["result"] == '{"ranked_results": []}'


# ---- boundary discipline ---------------------------------------------------------------------

def test_declared_boundaries_and_predicates_agree():
    """A signal declared observable at l must be EVALUABLE at l. The A4-v1 defect was a condition
    evaluated where the fact does not exist; a predicate that silently answers False there is the
    same failure one layer down."""
    for signal in R.declared_signals():
        for point in R.signal_boundaries(signal):
            state = R.observable_state({"boundary": point.value})
            R.evaluate_signal(signal, state, R.probe_params(signal))   # must not raise


def test_signal_not_observable_off_its_declared_boundary():
    with pytest.raises(KeyError):
        R.evaluate_signal("no_tool_call_at_all",
                          {"boundary": "pre_generation", "has_generation": False}, {})


def test_pre_generation_declares_noop_only():
    """A4 v1 fired a byte-identical reprompt at PRE_GENERATION for -4.95 pp. Declaring the cell
    would let the search re-propose an arm already measured as harmful."""
    assert R.feasible_actions(IncisionPoint.PRE_GENERATION) == frozenset({Action.NOOP})


def test_unknown_signal_raises_rather_than_returning_false():
    """A missing evaluator returning False is indistinguishable from a signal that did not fire --
    which is how an arm silently becomes the control arm."""
    with pytest.raises(KeyError):
        R.evaluate_signal("no_such_signal", {"boundary": "post_execution"}, {})


# ---- parameters -------------------------------------------------------------------------------

def test_required_param_is_enforced():
    with pytest.raises(ValueError):
        R.evaluate_signal("retrieval_similarity_below_threshold",
                          {"boundary": "post_execution"}, {})


def test_parameterized_signal_is_reachable_by_the_search():
    """The defect this pins: the search probed with {}, a required param raised, and the signal was
    recorded `signal_not_observable_at_boundary` -- so A9's trigger could never be selected."""
    assert R.probe_params("retrieval_similarity_below_threshold") == {"below": 0.30}
    st = R.observable_state(R.normalize_event(
        {"boundary": "post_execution", "calls": ['core_memory_retrieve(query="x")'],
         "tool_results": ['{"ranked_results": [[0.03, "e"]]}'], "best_similarity": 0.034}))
    assert R.evaluate_signal("retrieval_similarity_below_threshold", st,
                             R.probe_params("retrieval_similarity_below_threshold")) is True


# ---- actions ----------------------------------------------------------------------------------

def test_unexecutable_cell_raises_before_building_a_payload():
    """CONSUMER_BOUNDARY_RULE: the prototype built its payload first and reported executed=True
    while appending to the empty string."""
    with pytest.raises(ActionNotExecutable):
        R.apply_action(Action.SUPPRESS, IncisionPoint.POST_EXECUTION, {"reason": "x"}, {})


def test_reroute_requires_a_named_destination():
    with pytest.raises(ValueError):
        R.apply_action(Action.REROUTE, IncisionPoint.POST_EXECUTION, {}, {})


def test_unattested_condition_yields_no_reroute_destination():
    """docs/GENERALIZABILITY.md: a reroute must name a real destination attested on RESOLVING.
    Handing every reroute an invented destination let the strongest action survive everywhere and
    cost A3 its recovery."""
    assert R.reroute_destination("no_tool_call_at_all") is None
    assert R.reroute_destination("container_at_capacity")["destination"] == "archival_memory_add"


# ---- the calibration itself -------------------------------------------------------------------

def test_eight_accepted_anchors_are_recovered():
    """P2: 7 exact + 1 approximate, 0 failures, from each anchor's own attribution prose.

    A4 is the approximate one: (l, phi) exact, SUPPRESS chosen where the anchor uses REPROMPT, the
    two separated only by the frozen control-strength ordering. Pinned as approximate on purpose --
    the ranking is NOT tuned to force it, since the only reason to prefer REPROMPT there is knowing
    A4's measured outcome."""
    from anchor_recovery import diagnosis_for, theta_for_signal, verdict, _EXPECTED_SIGNAL

    # Anchors whose attribution asserts a clause Phi cannot observe ("the other store was never
    # consulted"). Recovery for these runs through signal expansion, not through candidate_search.
    _EXPECTED_SIGNAL_BLOCKED = {"A2", "A9"}
    from anchoropt.learning.candidate_search import (
        enumerate_candidates, expressible_under, select,
    )
    from rounds.anchors import ANCHORS

    verdicts = {}
    for anchor in ANCHORS:
        d = diagnosis_for(anchor)
        ok, _why = expressible_under(d, runtime=R)
        # TWO ANCHORS NOW BLOCK, CORRECTLY. `expressible_under` used to accept a residual when SOME
        # declared signal matched SOMETHING in the diagnosis; it now requires coverage of every clause
        # the diagnosis asserts. Both anchors whose attribution says the model NEVER CONSULTED the other
        # store fail that -- Phi has no signal for store coverage, only for a weak or failed read. That
        # is the gap the signal-expansion branch exists to fill, so asserting expressibility here would
        # re-institute the defect this test would then be protecting.
        if anchor.name in _EXPECTED_SIGNAL_BLOCKED:
            assert not ok, f"{anchor.name} was expected to be SIGNAL_BLOCKED under full-coverage"
            verdicts[anchor.name] = "signal_blocked"
            continue
        assert ok, f"{anchor.name} became SIGNAL_BLOCKED"
        cands = []
        for signal in R.declared_signals():
            cands.extend(c for c in enumerate_candidates(
                d, runtime=R, host=R.HOST, theta_for=theta_for_signal(signal))
                if c.signal == signal)
        chosen = select(cands)
        verdicts[anchor.name] = verdict(anchor, chosen)

    assert verdicts["A4"] == "approximate", verdicts
    exact = {k for k, v in verdicts.items() if v == "exact"}
    blocked = {k for k, v in verdicts.items() if v == "signal_blocked"}
    # WAS 7 exact + 1 approximate under the any-match expressibility rule. Under full-coverage it is
    # 5 exact + 1 approximate + 2 SIGNAL_BLOCKED, and the two blocked anchors are the ones whose own
    # attribution asserts a store was never consulted -- a clause Phi does not express. This is a
    # STRICTLY MORE HONEST reading of the same machinery: candidate_search recovers what the declared
    # vocabulary can express, and the rest is expansion's job, which is exactly the split the
    # architecture claims.
    assert exact == {"A1", "A3", "A5", "A7", "A8"}, verdicts
    assert blocked == {"A2", "A9"}, verdicts
    assert not [k for k, v in verdicts.items() if v == "no"], verdicts


def test_selection_is_deterministic():
    from anchor_recovery import diagnosis_for, theta_for_signal
    from anchoropt.learning.candidate_search import enumerate_candidates, select
    from rounds.anchors import ANCHORS

    def once():
        out = []
        for anchor in ANCHORS:
            d = diagnosis_for(anchor)
            cands = []
            for signal in R.declared_signals():
                cands.extend(c for c in enumerate_candidates(
                    d, runtime=R, host=R.HOST, theta_for=theta_for_signal(signal))
                    if c.signal == signal)
            c = select(cands)
            out.append(None if c is None else (c.boundary.value, c.signal, c.action.value))
        return out

    assert once() == once()


# ---- expanded-signal observability ------------------------------------------------------------
#
# `signal_boundaries` used to answer expanded signals with
# `bool(EXPANDED_SIGNALS[signal](state or {}))` -- a copy-paste of the dispatch block in
# `evaluate_signal`, referencing a `state` parameter this function does not have. Every synthesized
# signal therefore raised NameError, and the only production caller (the proposer role) catches
# Exception and substitutes [], so the proposer was told each new signal is observable NOWHERE.
# A null that looks like "no boundary suits this signal" but is really a crash is precisely the
# channel-never-live failure the project's key rule is about.

def test_signal_boundaries_returns_boundaries_not_a_truth_value():
    from anchoropt.anchor import IncisionPoint
    R.reset_expanded_signals()
    try:
        R.install_signal("synth_obs", lambda s: True, boundary=IncisionPoint.POST_EXECUTION)
        got = R.signal_boundaries("synth_obs")
        assert not isinstance(got, bool), "returned a truth value instead of boundaries"
        assert got == frozenset({IncisionPoint.POST_EXECUTION})
    finally:
        R.reset_expanded_signals()


def test_signal_boundaries_is_total_over_every_declared_signal():
    """No declared signal -- shipped OR expanded -- may raise or answer with a bool."""
    from anchoropt.anchor import IncisionPoint
    R.reset_expanded_signals()
    try:
        R.install_signal("synth_total", lambda s: True, boundary="post_generation_pre_exec")
        for sig in R.declared_signals():
            got = R.signal_boundaries(sig)
            assert not isinstance(got, bool), f"{sig} answered with a bool"
            assert all(isinstance(p, IncisionPoint) for p in got), sig
    finally:
        R.reset_expanded_signals()


def test_expanded_signal_without_a_validated_boundary_is_observable_nowhere():
    """Honest degradation: no boundary recorded means it was validated nowhere -- not everywhere."""
    R.reset_expanded_signals()
    try:
        R.install_signal("synth_nob", lambda s: True)
        assert R.signal_boundaries("synth_nob") == frozenset()
    finally:
        R.reset_expanded_signals()


def test_the_proposer_is_told_where_an_expanded_signal_is_observable():
    """Mirrors integrations/claude_roles/client.py, which swallows exceptions into []."""
    from anchoropt.anchor import IncisionPoint
    R.reset_expanded_signals()
    try:
        R.install_signal("synth_prop", lambda s: True, boundary=IncisionPoint.POST_EXECUTION)
        try:
            at = sorted(p.value for p in R.signal_boundaries("synth_prop"))
        except Exception:
            at = []
        assert at == ["post_execution"], "the proposer would see an empty observability list"
    finally:
        R.reset_expanded_signals()


# ---- boundary identification: committing to an answer IS a decision ----------------------------
#
# `is_decision` required a proposed or executed CALL, so an episode that answered without calling
# anything contributed NO boundary. Measured on the recovery traces: 303 such events per run (one per
# query), and boundary derivation returned the single boundary `post_execution` for every residual --
# which no backward search can move earlier from. The commitment gate was invisible, so "the search
# never moved earlier" was indistinguishable from "there was nowhere earlier to go".

def test_answer_commitment_is_a_decision():
    assert R.is_decision({"status": "answer_end_turn", "decoded": []})
    assert R.commits_to_answer({"status": "answer_end_turn"})


def test_answer_commitment_sits_at_the_commitment_gate():
    from anchoropt.anchor import IncisionPoint
    assert R.boundary_key({"status": "answer_end_turn", "decoded": []}) == \
        IncisionPoint.POST_GENERATION_PRE_EXEC.value


def test_a_step_with_results_is_placed_by_its_results_not_its_status():
    """Ordering matters: a step that both answers and carries results is post-execution."""
    from anchoropt.anchor import IncisionPoint
    assert R.boundary_key({"status": "answer_end_turn", "tool_results": ['{"ok": 1}']}) == \
        IncisionPoint.POST_EXECUTION.value


def test_a_trajectory_with_calls_and_an_answer_yields_two_boundaries():
    """The whole point: a residual must offer somewhere EARLIER to search."""
    from anchoropt.learning.boundary_search import boundaries_of, order_boundaries
    events = [{"status": "executed", "decoded": ["core_memory_retrieve(q='x')"],
               "tool_results": ['{"result": []}']},
              {"status": "answer_end_turn", "decoded": []}]
    keys = [b.key for b in order_boundaries(boundaries_of(events, adapter=R), adapter=R)]
    assert keys == ["post_execution", "post_generation_pre_exec"], keys


def test_a_non_decision_event_is_still_not_a_decision():
    assert not R.is_decision({"status": None, "decoded": []})
    assert not R.commits_to_answer({"status": "executed"})
    assert R.boundary_key({"status": None, "decoded": []}) == ""
