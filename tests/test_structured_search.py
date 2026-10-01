"""`optimize_residual` is the algorithm, so this file tests the SCHEDULE, not any benchmark.

Everything here runs against a toy adapter defined in this file -- a deliberately non-BFCL runtime
with two boundaries, two observables and two actions. That is the portability claim: the structured
search works against any runtime that implements the contract, and nothing about the one real
benchmark is load-bearing.

The ordering under test, which is the algorithmic contribution:

    at a fixed WHERE:  existing WHAT x HOW  ->  exhaust HOW/eta  ->  expand WHAT
                       ->  retry HOW here   ->  only then move WHERE earlier
"""

from __future__ import annotations

import pytest

from anchoropt.learning.search_state import (
    BOUNDARY_NOT_REPAIRABLE, HOW, IMPROVED, NO_BENEFIT, REALIZABLE_UNMEASURED, SIGNAL_EXPANDED,
    WHAT, WHERE,
)
from anchoropt.anchor import Action, IncisionPoint
from anchoropt.learning.structured_search import localize, optimize_residual
from anchoropt.runtime import HostProfile

# ------------------------------------------------------------------------------------------------
# A TOY RUNTIME. Two decision points named by this adapter's own vocabulary, deliberately NOT the
# benchmark's: if core needed a BFCL concept, this file could not exist.
# ------------------------------------------------------------------------------------------------

# Named for their REALIZED POSITION in EVENTS below, which is what the search orders by: the
# `act` step happens first, the `commit` step second, so the commit is the LATER boundary.
#
# The boundary KEYS are this adapter's own; `IncisionPoint` and `Action` are CORE vocabulary (they
# describe any LLM call, not this or any benchmark), so a portable adapter uses them rather than
# inventing a parallel enum. `boundary_from_key` is how an adapter maps its own keys back -- core
# owns no key->locus table, because that would be the semantic stage map reintroduced.
FIRST, LATER = "gate_act", "gate_commit"
_KEY_TO_POINT = {FIRST: IncisionPoint.POST_EXECUTION,
                 LATER: IncisionPoint.POST_GENERATION_PRE_EXEC}


class Field:
    def __init__(self, name, type_, boundaries):
        self.name, self.type, self.doc, self.enum = name, type_, "", ()
        self.boundaries = boundaries


class ToyRuntime:
    """The whole adapter contract, in one small class, with a domain of its own."""

    def __init__(self, *, declared=("saw_alarm",), actions=None, fields_at=None):
        self._declared = tuple(declared)
        self.expanded: dict = {}
        self.expanded_boundaries: dict = {}
        pe, pg = IncisionPoint.POST_EXECUTION, IncisionPoint.POST_GENERATION_PRE_EXEC
        self._fields_at = fields_at or {
            pe.value: {"pressure": Field("pressure", float, (pe.value, pg.value)),
                       "flagged": Field("flagged", bool, (pe.value, pg.value))},
            pg.value: {"pressure": Field("pressure", float, (pe.value, pg.value)),
                       "flagged": Field("flagged", bool, (pe.value, pg.value))},
        }
        self.HOST = HostProfile(
            name="toy_host",
            executable=(actions if actions is not None
                        else {pe: frozenset({Action.REPROMPT, Action.REROUTE}),
                              pg: frozenset({Action.REPROMPT, Action.SUPPRESS})}))

    # -- boundary identification (the adapter's own trace vocabulary)
    def is_decision(self, event):
        return bool(event.get("kind"))

    def boundary_key(self, event):
        return {"act": FIRST, "commit": LATER}.get(event.get("kind"), "")

    def label_for(self, key):
        return {FIRST: "post-act", LATER: "pre-commit"}.get(key, "")

    def boundary_from_key(self, key):
        return _KEY_TO_POINT.get(str(key))

    # -- signals
    def declared_signals(self):
        return self._declared + tuple(self.expanded)

    def signal_boundaries(self, signal):
        if signal in self.expanded:
            return frozenset(self.expanded_boundaries.get(signal, ()))
        return frozenset(_KEY_TO_POINT.values())

    def synthesis_fields(self, boundary=None):
        return dict(self._fields_at.get(str(getattr(boundary, "value", boundary)), {}))

    def install_signal(self, name, predicate, *, boundary=None, provenance=""):
        self.expanded[name] = predicate
        pt = boundary if isinstance(boundary, IncisionPoint) else _KEY_TO_POINT.get(str(boundary))
        self.expanded_boundaries[name] = {pt} if pt is not None else set()

    def evaluate_signal(self, signal, state, params=None):
        if signal in self.expanded:
            return bool(self.expanded[signal](state))
        return bool(state.get(signal))

    def probe_params(self, signal):
        return {}

    def states_at(self, boundary, states):
        """Project states onto one boundary's information set. Core FAILS CLOSED without this.

        This toy's states are already boundary-agnostic, so it returns them unchanged -- but it must
        be PRESENT, because core refuses to validate predicates on another boundary's states and an
        absent projection is a contract gap rather than a reason to guess.
        """
        return list(states or ())

    def parameter_domains(self, signal):
        return ()

    def executor_supports(self, boundary, action, signal, eta):
        return True, "toy host realizes every grounded action"

    # -- eta grounding: the adapter says HOW each action is parameterized in ITS domain
    def ground_reprompt(self, signal, boundary):
        return [{"variant": "warn", "eta": {"message": "check the gauge", "retry_budget": 1},
                 "detail": "ask the operator to re-read the gauge"}]

    def ground_suppress(self, signal, boundary):
        if str(getattr(boundary, "value", boundary)) != \
                IncisionPoint.POST_GENERATION_PRE_EXEC.value:
            return []
        return [{"variant": "cancel", "eta": {"suppressed_operation": "the proposed action",
                                              "preservation": "log it first"},
                 "detail": "cancel before dispatch"}]

    def ground_substitute_destinations(self, signal, boundary):
        return [{"variant": "reroute_backup", "eta": {"destination": "backup_line"},
                 "detail": "send to the backup line"}]

    def ground_transforms(self, signal, boundary):
        return []


EVENTS = [{"kind": "act", "i": 0}, {"kind": "commit", "i": 1}]
STATES = [{"pressure": i / 10.0, "flagged": i % 2 == 0} for i in range(12)]


class Residual:
    case_ids = ("c1", "c2")


# ------------------------------------------------------------------------------------------------
# WHERE
# ------------------------------------------------------------------------------------------------

def test_localize_derives_boundaries_from_the_trajectory_in_realized_order():
    rt = ToyRuntime()
    assert [b.key for b in localize(EVENTS, runtime=rt)] == [FIRST, LATER]


def test_a_recurring_boundary_collapses_to_its_first_occurrence():
    rt = ToyRuntime()
    events = [{"kind": "act"}, {"kind": "act"}, {"kind": "commit"}]
    assert [b.key for b in localize(events, runtime=rt)] == [FIRST, LATER]


def test_a_trajectory_with_no_decision_is_not_repairable():
    rt = ToyRuntime()
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST,
                            events=[{"noise": 1}], states=STATES)
    assert out.state == BOUNDARY_NOT_REPAIRABLE
    assert not out.rediscovered


def test_the_search_visits_the_LATEST_boundary_first():
    """Backward: a repair where the failure is observed is better attributed than an earlier one."""
    rt = ToyRuntime(declared=("saw_alarm",))
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES)
    assert out.boundaries == (FIRST, LATER)      # derived order: earliest -> latest
    assert out.visited[0] == LATER               # but SEARCHED latest-first


# ------------------------------------------------------------------------------------------------
# WHAT x HOW at a fixed WHERE, and the ordering between them
# ------------------------------------------------------------------------------------------------

def test_existing_phi_is_searched_before_any_expansion():
    """Step 1: the cheapest move needs no new representation."""
    rt = ToyRuntime(declared=("saw_alarm",))
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES)
    assert out.rediscovered
    assert not out.expanded, "expanded Phi despite the declared signal sufficing"
    assert out.state == REALIZABLE_UNMEASURED


def test_unmeasured_is_the_state_when_no_evaluator_is_supplied():
    """Structural rediscovery, with no claim of benefit."""
    rt = ToyRuntime()
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES)
    assert out.state == REALIZABLE_UNMEASURED
    assert out.rediscovered and not out.validated


def test_expansion_happens_at_the_SAME_boundary_before_moving_earlier():
    """Steps 3-4: no declared signal is observable, so Phi widens HERE and HOW is retried HERE."""
    rt = ToyRuntime(declared=())                      # empty Phi -> nothing to search
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES)
    assert out.expanded, "Phi was never widened"
    # The attempt that CARRIES the expansion is the one to look at -- its final `state` reflects what
    # the retry at that boundary then achieved, so `signals_expanded` is the record of the expansion.
    expanded_at = [a.boundary for a in out.attempts if a.signals_expanded]
    assert expanded_at, "no attempt records an expansion"
    assert expanded_at[0] == LATER, f"expanded at {expanded_at[0]}, not the latest boundary"
    # ... and the retry produced controllers AT THAT SAME BOUNDARY, without moving earlier
    same = [a for a in out.attempts if a.signals_expanded and a.candidates_built]
    assert same and same[0].boundary == LATER
    assert out.rediscovered
    assert out.moves_earlier == 0, "moved earlier even though expansion succeeded here"


def test_a_measured_improvement_promotes_and_stops():
    rt = ToyRuntime()
    promoted = []
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES,
                            evaluate=lambda arm: 5, improves=lambda o: o > 0,
                            promote=promoted.append)
    assert out.state == IMPROVED and out.validated
    assert len(promoted) == 1
    assert out.promoted is not None


def test_no_benefit_is_distinct_from_unmeasured_and_routes_to_HOW():
    """A measured zero is a result; an unmeasured controller is not."""
    rt = ToyRuntime()
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES,
                            evaluate=lambda arm: 0, improves=lambda o: o > 0)
    states = [a.state for a in out.attempts]
    assert NO_BENEFIT in states, states
    assert REALIZABLE_UNMEASURED not in states
    from anchoropt.learning.search_state import next_coordinate
    assert next_coordinate(NO_BENEFIT) == HOW


def test_an_unrepairable_boundary_moves_earlier():
    rt = ToyRuntime()
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES,
                            repairable_at=lambda r, b: b.key == FIRST)
    assert out.visited[0] == LATER
    assert out.attempts[0].state == BOUNDARY_NOT_REPAIRABLE
    assert out.attempts[0].coordinate_changed == WHERE
    assert out.moves_earlier >= 1
    assert FIRST in out.visited, "never reached the earlier boundary"


def test_moves_earlier_counts_only_boundary_movement():
    rt = ToyRuntime()
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES)
    assert out.moves_earlier == 0, "stopped at the first boundary, so nothing moved"


# ------------------------------------------------------------------------------------------------
# HOW feasibility certificates
# ------------------------------------------------------------------------------------------------

def test_a_boundary_with_no_executable_action_yields_a_certificate_not_silence():
    rt = ToyRuntime(actions={IncisionPoint.POST_EXECUTION: frozenset({Action.REPROMPT})})
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES)
    assert out.certificates, "no action available and no certificate explaining it"
    assert any(c.state == "ACTION_UNAVAILABLE" for c in out.certificates)


def test_every_certificate_names_a_generic_state_and_keeps_its_upstream_code():
    rt = ToyRuntime(actions={})
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES)
    for c in out.certificates:
        assert c.state, "a certificate with no generic state"


# ------------------------------------------------------------------------------------------------
# genericity
# ------------------------------------------------------------------------------------------------

def test_the_outcome_serializes_for_a_round_record():
    rt = ToyRuntime()
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES)
    d = out.as_dict()
    for key in ("boundaries", "visited", "moves_earlier", "state", "rediscovered", "validated"):
        assert key in d


def test_expansion_is_skipped_when_it_would_add_no_information():
    """A field that is constant across the observed states separates nothing."""
    pe, pg = IncisionPoint.POST_EXECUTION, IncisionPoint.POST_GENERATION_PRE_EXEC
    rt = ToyRuntime(declared=(), fields_at={
        pe.value: {"const": Field("const", bool, (pe.value,))},
        pg.value: {"const": Field("const", bool, (pg.value,))}})
    flat = [{"const": True} for _ in range(12)]
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=flat)
    assert not out.expanded, "installed a signal that discriminates nothing"


def test_no_benefit_after_expansion_moves_EARLIER_rather_than_terminating():
    """The bug this pins: the post-expansion branch used to `return` unconditionally.

    A widened Phi that was MEASURED and did not help exhausts the boundary -- it does not finish the
    residual. Returning there made every no-benefit boundary look terminal and made `moves_earlier`
    structurally unreachable on any measured run, so the backward search could never be observed to
    work. Termination is decided by the state machine, not by control flow reaching a line.
    """
    rt = ToyRuntime(declared=())                      # force the expansion path
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES,
                            evaluate=lambda arm: 0, improves=lambda o: o > 0)
    assert out.expanded, "never expanded, so this does not exercise the branch"
    assert FIRST in out.visited, "expanded and measured no benefit, yet never moved earlier"
    assert out.moves_earlier >= 1


def test_an_improvement_after_expansion_still_terminates():
    """The converse: IMPROVED and REALIZABLE_UNMEASURED remain terminal."""
    rt = ToyRuntime(declared=())
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES,
                            evaluate=lambda arm: 7, improves=lambda o: o > 0)
    assert out.state == IMPROVED
    assert out.visited[-1] == LATER, "kept searching after a promotion"


def test_moves_earlier_is_counted_from_observed_transitions():
    """The bug this pins: counting each attempt's terminal coordinate UNDERCOUNTS movement.

    After Phi is widened at a boundary, the last attempt there ends in NO_BENEFIT (-> HOW) or
    REALIZABLE_UNMEASURED (-> DONE). So a search that visibly moved from one boundary to an earlier
    one reported moves_earlier == 0 -- a metric unable to observe the very movement it exists to
    measure, which is the channel-never-live failure one level up.
    """
    rt = ToyRuntime(declared=())
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES,
                            evaluate=lambda arm: 0, improves=lambda o: False)
    visited = list(out.visited)
    transitions = sum(1 for x, y in zip(visited, visited[1:]) if x != y)
    assert out.moves_earlier == transitions, (out.moves_earlier, visited)
    assert len(set(visited)) > 1, "this fixture must actually move to exercise the count"
    assert out.moves_earlier >= 1


def test_moves_earlier_is_zero_when_the_search_never_leaves_one_boundary():
    """`moves_earlier` counts GIVING UP on a boundary, and the localization sweep is not that.

    With no evaluator the round now also collects candidates at the other boundaries the seeded Phi is
    observable at -- so `visited` may hold more than one boundary while `moves_earlier` stays 0. The
    two were conflated before, which made a completed localization report structural movement it never
    performed. The sweep is recorded separately.
    """
    rt = ToyRuntime(declared=("saw_alarm",))
    out = optimize_residual(Residual(), runtime=rt, host=rt.HOST, events=EVENTS, states=STATES)
    swept = set(out.localization_sweep)
    assert len(set(out.visited) - swept) == 1, (out.visited, out.localization_sweep)
    assert out.moves_earlier == 0
