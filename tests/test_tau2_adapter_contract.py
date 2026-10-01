"""The tau-bench adapter against the contract, and against the rules the contract cannot express.

OFFLINE. Nothing here imports tau2, so these run with no tau-bench checkout and no model. The
behavioural half -- does the declared executor actually DO what it claims -- is
`test_tau2_executor_behavioral.py`, which does need the checkout and skips without it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "tau2"))

from anchoropt.anchor import Action, IncisionPoint            # noqa: E402
from anchoropt.testing import check_adapter_contract          # noqa: E402
from anchoropt.testing.adapter_contract import AdapterContract  # noqa: E402

import tau2_state as S                                        # noqa: E402
from tau2_capabilities import CAPABILITIES                     # noqa: E402
from tau2_runtime import ADAPTER                               # noqa: E402

_PRE = IncisionPoint.PRE_GENERATION.value
_PG = IncisionPoint.POST_GENERATION_PRE_EXEC.value
_PE = IncisionPoint.POST_EXECUTION.value


def _states() -> list[dict]:
    """One realistic state per boundary, built by the SAME builders the live mechanism uses."""
    return [
        S.turn_start_state(case_id="c1", turn_index=1, assistant_turns_so_far=0,
                           user_turns_so_far=1, inbound_is_tool_result=False,
                           consecutive_tool_errors=0, tool_errors_so_far=0),
        S.turn_start_state(case_id="c1", turn_index=2, assistant_turns_so_far=1,
                           user_turns_so_far=1, inbound_is_tool_result=True,
                           consecutive_tool_errors=1, tool_errors_so_far=1),
        S.gate_state(case_id="c1", turn_index=1,
                     proposed_calls=[{"name": "book_reservation",
                                      "arguments": {"reservation_id": "ABC", "cabin": "economy"}}],
                     has_content=False, known_tools={"book_reservation", "get_reservation_details"},
                     mutating_tools={"book_reservation"}, replan_attempt=0,
                     consecutive_tool_errors=0, tool_errors_so_far=0),
        S.gate_state(case_id="c2", turn_index=3, proposed_calls=[], has_content=True,
                     known_tools={"book_reservation"}, mutating_tools={"book_reservation"},
                     replan_attempt=0, consecutive_tool_errors=0, tool_errors_so_far=0),
        S.result_state(case_id="c1", turn_index=1, is_error=True,
                       content="Error: Reservation ZZZ not found",
                       tool_name="get_reservation_details", mutating_tools={"book_reservation"},
                       tool_errors_so_far=1),
        S.result_state(case_id="c2", turn_index=2, is_error=False, content='{"ok": true}',
                       tool_name="get_user_details", mutating_tools={"book_reservation"},
                       tool_errors_so_far=0),
    ]


def _events(states) -> list[dict]:
    kinds = {_PRE: "turn_start", _PG: "propose", _PE: "result"}
    return [dict(s, kind=kinds[s["boundary"]]) for s in states]


@pytest.fixture(autouse=True)
def _catalog():
    ADAPTER.load_tool_catalog("airline")
    ADAPTER.reset_expanded_signals()
    yield
    ADAPTER.reset_expanded_signals()


# ================================================================================ 1. the contract
def test_adapter_satisfies_the_contract():
    states = _states()
    rep = check_adapter_contract(ADAPTER, states=states, events=_events(states))
    assert rep.ok, rep.summary()


def test_no_mandatory_hook_is_missing():
    states = _states()
    rep = check_adapter_contract(ADAPTER, states=states, events=_events(states))
    assert rep.failures == [], [f"{v.hook}: {v.detail}" for v in rep.failures]


class TestTau2AdapterContract(AdapterContract):
    """The reusable pytest mixin, on this adapter."""

    adapter = ADAPTER
    states = _states()


# ============================================================ 2. the four families, and no fifth
def test_host_declares_only_the_four_closed_action_families():
    declared = {a for pt in IncisionPoint for a in (ADAPTER.HOST.executable_actions(pt) or ())}
    assert declared <= set(Action), declared - set(Action)


def test_every_declared_cell_is_bound_so_none_is_a_ghost():
    audit = ADAPTER.capability_audit()
    assert audit["unbound"] == [], audit["unbound"]


def test_every_capability_declares_consumed_eta_or_says_it_computes_it():
    for (boundary, action), cap in CAPABILITIES.items():
        assert cap.consumes or cap.eta_is_computed, (
            f"{boundary}/{action} declares neither consumed eta nor eta_is_computed")


def test_the_unavailable_cell_is_disabled_with_a_reason_not_deleted():
    """An inert intervention must be disabled with an explicit reason, never silently dropped."""
    cap = ADAPTER.executor_capability(IncisionPoint.POST_EXECUTION, Action.REROUTE)
    assert cap is not None, "post_execution/reroute was deleted, which makes it look unconsidered"
    assert not cap.is_enabled
    assert cap.disabled_reason.strip()
    assert "set_state" in cap.disabled_reason or "replay" in cap.disabled_reason


def test_post_execution_suppress_is_absent_because_core_excludes_it_structurally():
    assert Action.SUPPRESS not in (ADAPTER.HOST.executable_actions(IncisionPoint.POST_EXECUTION) or ())
    assert ADAPTER.executor_capability(IncisionPoint.POST_EXECUTION, Action.SUPPRESS) is None


# ============================================================ 3. grounding is materializability only
def test_suppress_grounds_only_at_the_gate_and_claims_no_preservation():
    """Remove-outright preserves nothing; a clause no executor reads is not a clause."""
    gate = ADAPTER.ground_suppress("proposes_state_change", IncisionPoint.POST_GENERATION_PRE_EXEC)
    assert gate and all(g["variant"] == "cancel_proposed" for g in gate)
    assert all("preservation" not in g["eta"] for g in gate), gate
    assert ADAPTER.ground_suppress("x", IncisionPoint.POST_EXECUTION) == []
    assert ADAPTER.ground_suppress("x", IncisionPoint.PRE_GENERATION) == []


def test_pre_generation_reprompt_does_not_carry_eta_its_executor_cannot_read():
    """No candidate exists before generation, so retry_budget has nothing to re-plan."""
    cap = ADAPTER.executor_capability(IncisionPoint.PRE_GENERATION, Action.REPROMPT)
    for g in ADAPTER.ground_reprompt("following_tool_error", IncisionPoint.PRE_GENERATION):
        assert set(g["eta"]) <= set(cap.consumes), (
            f"{set(g['eta']) - set(cap.consumes)} is inert eta at pre_generation")


def test_reroute_destinations_come_from_real_schemas_and_are_empty_without_a_catalog():
    got = ADAPTER.ground_substitute_destinations("x", IncisionPoint.POST_GENERATION_PRE_EXEC)
    assert got, "the airline catalog is installed, so destinations should be offered"
    catalog = ADAPTER.tool_catalog
    for g in got:
        dest = g["eta"]["destination"]
        assert dest in catalog, f"{dest} is not a real tool"
        assert not catalog[dest]["mutates_state"], (
            f"{dest} mutates state: rewriting onto it could corrupt the evaluator's replay")

    ADAPTER.set_tool_catalog({})
    assert ADAPTER.ground_substitute_destinations(
        "x", IncisionPoint.POST_GENERATION_PRE_EXEC) == [], (
        "with no catalog the adapter must offer nothing rather than guess a destination")


def test_transform_grounding_is_empty_and_the_reason_is_recorded():
    assert ADAPTER.ground_transforms("x", IncisionPoint.POST_EXECUTION) == []
    cap = ADAPTER.executor_capability(IncisionPoint.POST_EXECUTION, Action.REROUTE)
    assert not cap.is_enabled


# ============================================================ 4. the adapter never decides
def test_the_adapter_exposes_no_failure_to_action_mapping():
    """Grounding must not vary with the signal: choosing an action per failure is core's job.

    If a grounder returned different eta for different signals, the adapter would be recommending a
    remedy for a failure class -- the specific thing ADAPTER_GUIDE section 7 forbids.
    """
    boundary = IncisionPoint.POST_GENERATION_PRE_EXEC
    for grounder in (ADAPTER.ground_reprompt, ADAPTER.ground_suppress,
                     ADAPTER.ground_substitute_destinations, ADAPTER.ground_transforms):
        got = {s: grounder(s, boundary) for s in ADAPTER.declared_signals()}
        first = got[next(iter(got))]
        assert all(v == first for v in got.values()), (
            f"{grounder.__name__} varies with the signal, which is a failure->action mapping")


def test_no_tau2_identifier_reaches_core():
    """`tests/test_core_genericity.py` enforces this globally; this pins THIS adapter's vocabulary.

    Matched as identifiers with word boundaries, not as prose. An earlier version of this test grepped
    for "reservation" and flagged nine core files -- every hit was the substring inside
    "pre-servation", which is core's own safety-clause vocabulary. A leak test that cries wolf gets
    muted, so it matches only tokens that would indicate real coupling: this adapter's module names,
    its boundary keys, its mechanism class, and domain tool names.
    """
    tokens = ("tau2_runtime", "tau2_mechanism", "tau2_signals", "tau2_fields", "tau2_state",
              "tau2_capabilities", "tau2_episodes", "ControlledLLMAgent", "before_tool_dispatch",
              "after_tool_result", "before_agent_turn", "book_reservation",
              "get_reservation_details", "telecom_gates")
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b")
    leaked = []
    for path in (REPO / "anchoropt").rglob("*.py"):
        for m in pattern.finditer(path.read_text(errors="ignore")):
            leaked.append(f"{path.relative_to(REPO)}: {m.group(1)}")
    assert leaked == [], leaked
