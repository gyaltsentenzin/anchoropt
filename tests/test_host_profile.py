"""`HostProfile` = U_H(l): the operational narrowing of the structural grid.

The distinction under test is the one the TB2 prototype got wrong at measurable cost: a cell can
pass the STRUCTURAL grid (`feasible_actions`) and still be unexecutable in a given runtime. Two of
ten nominally admissible cells did nothing there, and one reported success while appending its
payload to an empty string.
"""

from __future__ import annotations

import pytest

from anchoropt.anchor import (
    Action, CANONICAL_OPERATOR, IncisionPoint, canonical_operators, feasible_actions,
)
from anchoropt.runtime import (
    ActionNotExecutable, HostProfile, ResidualDiagnosis, diagnoses_from, structural_grid,
)


def test_a_host_may_narrow_the_structural_grid():
    host = HostProfile(name="h", executable={
        IncisionPoint.PRE_GENERATION: frozenset({Action.NOOP}),
    })
    assert host.executable_actions(IncisionPoint.PRE_GENERATION) == {Action.NOOP}
    assert Action.REPROMPT in feasible_actions(IncisionPoint.PRE_GENERATION)
    assert not host.can_execute(IncisionPoint.PRE_GENERATION, Action.REPROMPT)


def test_a_host_may_never_widen_the_structural_grid():
    """The exclusions are structural, so no runtime can authorise them."""
    with pytest.raises(ValueError, match="structurally inadmissible"):
        HostProfile(name="bad", executable={
            IncisionPoint.PRE_GENERATION: frozenset({Action.SUPPRESS}),
        })


def test_an_undeclared_point_declares_nothing_rather_than_everything():
    """An omission must read as 'not declared', never as 'everything works'."""
    host = HostProfile(name="h", executable={})
    for point in IncisionPoint:
        assert host.executable_actions(point) == frozenset()


def test_require_distinguishes_structural_from_host_limitation():
    """The two rejections lead to different next steps, so they must be distinguishable."""
    host = HostProfile(name="h", executable={
        IncisionPoint.POST_GENERATION_PRE_EXEC: frozenset({Action.NOOP}),
    })
    with pytest.raises(ActionNotExecutable, match="structurally inadmissible"):
        host.require(IncisionPoint.PRE_GENERATION, Action.SUPPRESS)
    with pytest.raises(ActionNotExecutable, match="cannot execute"):
        host.require(IncisionPoint.POST_GENERATION_PRE_EXEC, Action.SUPPRESS)


def test_unsupported_cells_are_reported_not_hidden():
    host = HostProfile(name="h", executable={
        p: frozenset({Action.NOOP}) for p in IncisionPoint
    })
    gaps = host.unsupported_cells()
    assert (IncisionPoint.PRE_GENERATION, Action.REPROMPT) in gaps
    assert all(a is not Action.NOOP for _, a in gaps)


def test_structural_grid_matches_feasible_actions():
    assert structural_grid() == {p: feasible_actions(p) for p in IncisionPoint}


# ---- the canonical operator mapping (paper 5 vs code 4) ----------------------------------------

def test_canonical_operator_covers_every_action():
    assert set(CANONICAL_OPERATOR) == set(Action)


def test_only_reroute_is_ambiguous():
    """REROUTE splits into substitute/transform; every other action names itself."""
    for action in Action:
        names = canonical_operators(action)
        if action is Action.REROUTE:
            assert names == {"substitute", "transform"}
        else:
            assert names == {action.value}


# ---- the generic diagnosis type ----------------------------------------------------------------

def test_residual_diagnosis_carries_no_provider_vocabulary():
    """No hook field: a prompt-heavy provider must not bias (l, phi, mu, theta) selection."""
    fields = ResidualDiagnosis.__dataclass_fields__
    for banned in ("hook", "prompt", "instruction", "middleware", "surface"):
        assert not any(banned in f for f in fields), f"{banned!r} leaked into the core type"


def test_residual_diagnosis_requires_the_load_bearing_fields():
    with pytest.raises(ValueError, match="mechanism"):
        ResidualDiagnosis(case_id="c", mechanism="  ", evidence="e",
                          consequential_decision="d", proposed_behavior_change="b")


def test_diagnoses_from_is_provider_agnostic():
    got = diagnoses_from("self_harness", [
        {"case_id": "mailman", "mechanism": "concluded instead of acting",
         "evidence": "assertion", "consequential_decision": "emit terminal response",
         "proposed_behavior_change": "verify before concluding"},
    ])
    assert len(got) == 1 and got[0].provider == "self_harness"
    # The core treats the case id as opaque -- it never parses it.
    assert got[0].case_id == "mailman"
