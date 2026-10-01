"""Each declared boundary exposes ONLY what the tau-bench runtime carries at that point.

This is the porting bug that produces controllers which can never fire: a field declared where the
runtime does not have it makes every predicate over it answer False, for a structural reason
indistinguishable from the condition not holding. So the tests here check the declaration against the
STATE BUILDERS -- in both directions -- and check that `states_at` filters without inventing.

OFFLINE: no tau2 import.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "tau2"))

from anchoropt.anchor import IncisionPoint                    # noqa: E402

import tau2_fields as F                                       # noqa: E402
import tau2_signals as SIG                                    # noqa: E402
import tau2_state as S                                        # noqa: E402
from tau2_runtime import ADAPTER                              # noqa: E402

_PRE = IncisionPoint.PRE_GENERATION.value
_PG = IncisionPoint.POST_GENERATION_PRE_EXEC.value
_PE = IncisionPoint.POST_EXECUTION.value

# One state per locus, from the builders the LIVE mechanism uses. If these drift from the declaration,
# every test below fails -- which is the point.
TURN = S.turn_start_state(case_id="c1", turn_index=2, assistant_turns_so_far=1, user_turns_so_far=1,
                          inbound_is_tool_result=True, consecutive_tool_errors=1,
                          tool_errors_so_far=1)
GATE = S.gate_state(case_id="c1", turn_index=2,
                    proposed_calls=[{"name": "book_reservation", "arguments": {"reservation_id": "A"}}],
                    has_content=False, known_tools={"book_reservation"},
                    mutating_tools={"book_reservation"}, replan_attempt=0,
                    consecutive_tool_errors=1, tool_errors_so_far=1)
RESULT = S.result_state(case_id="c1", turn_index=2, is_error=True,
                        content="Error: Reservation A not found", tool_name="book_reservation",
                        mutating_tools={"book_reservation"}, tool_errors_so_far=2)

BY_LOCUS = {_PRE: TURN, _PG: GATE, _PE: RESULT}


# ==================================================== declaration <-> runtime, both directions
def test_every_key_the_runtime_produces_is_declared_at_that_locus():
    for locus, state in BY_LOCUS.items():
        declared = set(F.FIELDS_AT[locus])
        produced = set(state) - set(F.IDENTITY_KEYS)
        undeclared = produced - declared
        assert undeclared == set(), (
            f"{locus} produces {undeclared} which no field declares: a predicate could never be "
            f"synthesized over them")


def test_every_declared_field_is_actually_produced_at_that_locus():
    """The direction that catches the real defect: a field declared where nothing supplies it."""
    for locus, state in BY_LOCUS.items():
        declared = set(F.FIELDS_AT[locus])
        missing = declared - set(state)
        assert missing == set(), (
            f"{locus} declares {missing} but the runtime does not carry them there, so every "
            f"predicate over them answers False for a structural reason")


def test_each_field_lists_the_locus_it_is_declared_at():
    for locus, fields in F.FIELDS_AT.items():
        for name, field in fields.items():
            assert locus in field.boundaries, (
                f"{name} appears at {locus} but its own boundaries are {field.boundaries}")


# ==================================================== the asymmetry that forces backward search
def test_no_proposed_call_field_exists_before_generation():
    """PRE_GENERATION has no candidate: a `proposed_*` field there would be a fabrication."""
    leaked = [n for n in F.FIELDS_AT[_PRE] if n.startswith("proposed") or n.startswith("proposes")]
    assert leaked == [], leaked


def test_no_result_field_exists_at_or_before_the_commitment_gate():
    """A result cannot exist before dispatch. This is always a defect, never a carried fact."""
    for locus in (_PRE, _PG):
        leaked = [n for n in F.FIELDS_AT[locus]
                  if n.startswith("result_") or n == "error_kind"]
        assert leaked == [], f"{locus} declares result fields {leaked}"


def test_carried_history_is_named_so_it_cannot_be_read_as_this_turns_result():
    """`tool_errors_so_far` / `consecutive_tool_errors` appear pre-dispatch and legitimately so.

    They summarize FINISHED turns. The naming is load-bearing: called `error_kind` they would look like
    the pending call's outcome, which is exactly the confusion the previous test forbids.
    """
    carried = {n for n in F.FIELDS_AT[_PG]} & {n for n in F.FIELDS_AT[_PE]}
    assert carried == {"turn_index", "tool_errors_so_far"}, carried
    for name in carried:
        assert not name.startswith("result_") and name != "error_kind"


# ==================================================== states_at filters, never invents
def test_states_at_returns_only_states_from_that_boundary():
    everything = [TURN, GATE, RESULT]
    for point in IncisionPoint:
        got = ADAPTER.states_at(point, everything)
        assert all(s["boundary"] == point.value for s in got), got
        assert len(got) == 1, f"{point.value} projected {len(got)} states"


def test_states_at_never_invents_a_key():
    everything = [TURN, GATE, RESULT]
    for point in IncisionPoint:
        for projected in ADAPTER.states_at(point, everything):
            source = BY_LOCUS[point.value]
            assert set(projected) <= set(source), set(projected) - set(source)
            for k, v in projected.items():
                assert v == source[k]


def test_states_at_strips_fields_the_boundary_does_not_declare():
    """A post-execution field must not survive projection onto the gate, or a predicate over it
    could be validated where it cannot fire."""
    contaminated = dict(GATE, error_kind="not_found", result_is_error=True)
    got = ADAPTER.states_at(IncisionPoint.POST_GENERATION_PRE_EXEC, [contaminated])
    assert got and "error_kind" not in got[0] and "result_is_error" not in got[0]


def test_states_at_accepts_this_adapters_own_boundary_key_too():
    got = ADAPTER.states_at(F.GATE, [TURN, GATE, RESULT])
    assert len(got) == 1 and got[0]["boundary"] == _PG


# ==================================================== signals are only claimed where evaluable
def test_every_shipped_signal_evaluates_on_its_own_boundarys_state():
    for signal, points in SIG.SIGNAL_BOUNDARIES.items():
        for point in points:
            state = BY_LOCUS[point.value]
            assert isinstance(ADAPTER.evaluate_signal(signal, state), bool)


def test_every_shipped_signal_reads_only_fields_declared_at_its_boundary():
    """A signal that needs a field its boundary does not carry can never fire there."""
    for signal, points in SIG.SIGNAL_BOUNDARIES.items():
        for point in points:
            declared = set(F.FIELDS_AT[point.value]) | set(F.IDENTITY_KEYS)
            # An empty state makes every missing field falsy; if the signal still answers the same as
            # on a state restricted to declared fields, it reads nothing outside the alphabet.
            restricted = {k: v for k, v in BY_LOCUS[point.value].items() if k in declared}
            assert ADAPTER.evaluate_signal(signal, restricted) == ADAPTER.evaluate_signal(
                signal, BY_LOCUS[point.value])


def test_a_signal_is_not_declared_at_a_boundary_that_cannot_evaluate_it():
    """`tool_call_failed` is post-execution only: at the gate there is no result to inspect."""
    assert SIG.SIGNAL_BOUNDARIES["tool_call_failed"] == frozenset({IncisionPoint.POST_EXECUTION})
    assert IncisionPoint.POST_GENERATION_PRE_EXEC not in SIG.SIGNAL_BOUNDARIES["following_tool_error"]
    for s in ("proposes_tool_call", "proposes_state_change", "commits_to_reply",
              "proposes_unknown_tool"):
        assert SIG.SIGNAL_BOUNDARIES[s] == frozenset({IncisionPoint.POST_GENERATION_PRE_EXEC})


def test_unknown_signal_raises_rather_than_answering_false():
    """Silently answering False is indistinguishable from the condition not holding."""
    import pytest
    with pytest.raises(KeyError):
        ADAPTER.evaluate_signal("no_such_signal", GATE)
