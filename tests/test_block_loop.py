"""The block-coordinate loop: POLICY / SIGNAL distinct, phase switch in AnchorOpt, re-mining live.

Mock seams throughout -- this tests SEQUENCING, which is all the module owns. Each test pins a way
the method could quietly degrade into something weaker:

  * a frozen error list instead of re-mining (A1's success created A2/A3/A5/A8's residuals)
  * Phi expanding during a policy round (the blocks collapse into one unconstrained call)
  * a tie-break choosing the controller before measurement
  * a corrupted denominator reaching the acceptance rule
  * "no candidate" reported as "done"
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

from anchoropt.anchor import Action, IncisionPoint                     # noqa: E402
from anchoropt.learning.block_loop import (                            # noqa: E402
    S1_COVERAGE_FLOOR, S2_SATURATION_CEIL, RoundRecord, residual_phase, run_blocks,
)
from anchoropt.learning.self_evolve import (                           # noqa: E402
    CONTINUE_POLICY, EXPAND_ATTRIBUTION, STOP, EvaluationResult, Incumbent,
    OUTCOME_ACCEPTED, OUTCOME_BLOCKED, OUTCOME_NO_CANDIDATE, OUTCOME_REJECTED,
)
from anchoropt.learning.proposal_seams import ProposedCandidate        # noqa: E402
from anchoropt.runtime import ResidualDiagnosis                        # noqa: E402


def _diag(case_id, decision="whether to search the other container"):
    return ResidualDiagnosis(case_id=case_id, mechanism="m", evidence="e",
                             consequential_decision=decision, proposed_behavior_change="b")


def _cand(name="phi_a", action=Action.REPROMPT):
    return ProposedCandidate(boundary=IncisionPoint.POST_EXECUTION, signal_name=name,
                             action=action, theta={"text": "check archival"},
                             diagnosis_case_ids=("c0",))


def _ev(gains=(), losses=(), ok=True, fired=5, executed=5):
    paired = {f"c{i}": True for i in range(10)}
    return EvaluationResult(control=paired, candidate=paired, gains=tuple(gains),
                            losses=tuple(losses), denominator_ok=ok,
                            telemetry={"signal_firings": fired, "interventions_executed": executed},
                            detail="mock")


# ---- the thresholds must not drift from phase_switch's ----------------------------------------

def test_mirrored_thresholds_match_phase_switch():
    """block_loop mirrors these because phase_switch pulls BFCL readers at import. Mirrored copies
    are only safe while something asserts they agree."""
    src = (REPO / "anchoropt/learning/phase_switch.py").read_text()
    assert f"S1_COVERAGE_FLOOR = {S1_COVERAGE_FLOOR}" in src
    assert f"S2_SATURATION_CEIL = {S2_SATURATION_CEIL}" in src


# ---- residual is re-mined every round ---------------------------------------------------------

def test_diagnose_is_called_against_the_current_incumbent_every_round():
    """The frozen-error-list failure: a loop that mines once cannot find the residuals its own
    accepted anchors create."""
    seen: list[str] = []

    def diagnose(incumbent):
        seen.append(incumbent.incumbent_id)
        return [_diag(f"c{i}") for i in range(10)]

    n = {"i": 0}

    def promote(inc, cand, ev):
        n["i"] += 1
        return Incumbent(workspace=inc.workspace, incumbent_id=f"P{n['i']}", depth=n["i"])

    trail = run_blocks(
        Incumbent(workspace=Path("."), incumbent_id="P0"),
        diagnose=diagnose,
        propose=lambda ds, allow: [_cand()],
        validate=lambda c: (None, "legal"),
        evaluate=lambda inc, c: _ev(gains=("c1", "c2")),
        accept=lambda ev: (ev.net > 0, "net"),
        promote=promote, max_rounds=3)

    assert seen == ["P0", "P1", "P2"], seen
    assert all(r.outcome == OUTCOME_ACCEPTED for r in trail)


# ---- POLICY and SIGNAL blocks stay distinct ---------------------------------------------------

def test_policy_rounds_never_receive_permission_to_author_a_new_signal():
    """`allow_new_phi=False` in every POLICY round. If a policy proposal could invent a signal, the
    block-coordinate structure is gone."""
    flags: list[bool] = []

    def propose(ds, allow_new_phi):
        flags.append(allow_new_phi)
        return [_cand()]

    run_blocks(Incumbent(workspace=Path("."), incumbent_id="P0"),
               diagnose=lambda inc: [_diag(f"c{i}") for i in range(10)],
               propose=propose, validate=lambda c: (None, "legal"),
               evaluate=lambda inc, c: _ev(),
               accept=lambda ev: (False, "no"), promote=lambda *a: None, max_rounds=2)
    assert flags == [False, False], flags


def test_signal_block_is_entered_only_when_the_inexpressible_mass_dominates():
    """S5. 'No candidate' means Phi is the limit, not that the work is done -- reporting the two as
    one hides the only signal that says grow the vocabulary."""
    # 10 diagnoses spread over 10 decisions. Every locus covers exactly the S1 floor, so
    # expressibility is what decides -- which is the point: an inexpressible locus must not clear
    # the gate into a POLICY round that has nothing legal to propose.
    ds = [_diag(f"c{i}", decision=f"decision {i}") for i in range(10)]

    phase, reasons = residual_phase(ds, expressible=lambda d: False)
    assert phase == EXPAND_ATTRIBUTION, reasons
    assert any("S5" in r for r in reasons)
    assert any("SIGNAL_BLOCKED" in r for r in reasons)

    # The same residual, now expressible: a policy round IS warranted, and reporting STOP here
    # would abandon addressable failures.
    phase2, _ = residual_phase(ds, expressible=lambda d: True)
    assert phase2 == CONTINUE_POLICY

    # STOP is the case where nothing survives the gates AND the inexpressible mass does not
    # dominate: one dominant locus, expressible, already saturated by the incumbent (S2).
    settled = [_diag(f"c{i}", decision="one dominant decision") for i in range(10)]
    key = "one dominant decision"
    phase3, reasons3 = residual_phase(settled, expressible=lambda d: True,
                                      engaged={key: 10})
    assert phase3 == STOP, reasons3
    assert any("S2-STOP" in r for r in reasons3)


def test_expansion_freezes_phi_and_returns_to_policy():
    blocks: list[str] = []

    def diagnose(inc):
        # first round: nothing expressible and fragmented -> EXPAND; later: one dominant locus
        if not blocks:
            return [_diag(f"c{i}", decision=f"d{i}") for i in range(10)]
        return [_diag(f"c{i}") for i in range(10)]

    def decide_phase(ds):
        phase, reasons = residual_phase(ds, expressible=lambda d: len(blocks) > 0)
        blocks.append(phase)
        return phase, reasons

    trail = run_blocks(Incumbent(workspace=Path("."), incumbent_id="P0"),
                       diagnose=diagnose, propose=lambda ds, allow: [_cand()],
                       validate=lambda c: (None, "legal"),
                       evaluate=lambda inc, c: _ev(gains=("c1",)),
                       accept=lambda ev: (ev.net > 0, "net"),
                       promote=lambda inc, c, ev: Incumbent(workspace=inc.workspace,
                                                            incumbent_id="P1", depth=1),
                       decide_phase=decide_phase,
                       expand=lambda ds: ("newly_validated_phi",), max_rounds=3)
    assert trail[0].block == EXPAND_ATTRIBUTION
    assert trail[0].new_signals == ("newly_validated_phi",)
    assert "frozen again" in trail[0].note
    assert trail[1].block == CONTINUE_POLICY


def test_expansion_proposing_nothing_is_a_terminal_state_not_success():
    trail = run_blocks(Incumbent(workspace=Path("."), incumbent_id="P0"),
                       diagnose=lambda inc: [_diag(f"c{i}", decision=f"d{i}") for i in range(10)],
                       propose=lambda ds, allow: [], validate=lambda c: (None, "legal"),
                       evaluate=lambda inc, c: _ev(), accept=lambda ev: (False, ""),
                       promote=lambda *a: None,
                       decide_phase=lambda ds: residual_phase(ds, expressible=lambda d: False),
                       expand=lambda ds: (), max_rounds=3)
    assert len(trail) == 1
    assert trail[0].outcome == EXPAND_ATTRIBUTION
    assert "not addressable" in trail[0].note
    assert trail[0].outcome != STOP


# ---- measurement chooses ----------------------------------------------------------------------

def test_top_k_candidates_are_all_evaluated_and_the_best_measured_one_wins():
    """The ranking screens for cost; it must not decide. Here the SECOND-ranked candidate is the
    better one on measurement and must win."""
    cands = [_cand("phi_a"), _cand("phi_b"), _cand("phi_c")]
    scores = {"phi_a": _ev(gains=("c1",)), "phi_b": _ev(gains=("c1", "c2", "c3")),
              "phi_c": _ev(gains=("c1", "c2"))}

    trail = run_blocks(Incumbent(workspace=Path("."), incumbent_id="P0"),
                       diagnose=lambda inc: [_diag(f"c{i}") for i in range(10)],
                       propose=lambda ds, allow: cands,
                       validate=lambda c: (None, "legal"),
                       evaluate=lambda inc, c: scores[c.signal_name],
                       accept=lambda ev: (ev.net > 0, "net"),
                       promote=lambda inc, c, ev: Incumbent(workspace=inc.workspace,
                                                            incumbent_id="P1", depth=1),
                       max_rounds=1, top_k=3)
    assert len(trail[0].evaluated) == 3, "all screened candidates must be measured"
    assert trail[0].accepted.signal_name == "phi_b"


def test_top_k_bounds_the_evaluation_budget():
    cands = [_cand(f"phi_{i}") for i in range(6)]
    trail = run_blocks(Incumbent(workspace=Path("."), incumbent_id="P0"),
                       diagnose=lambda inc: [_diag(f"c{i}") for i in range(10)],
                       propose=lambda ds, allow: cands,
                       validate=lambda c: (None, "legal"),
                       evaluate=lambda inc, c: _ev(), accept=lambda ev: (False, "no"),
                       promote=lambda *a: None, max_rounds=1, top_k=2)
    assert len(trail[0].evaluated) == 2


# ---- the guards -------------------------------------------------------------------------------

def test_denominator_failure_blocks_before_the_acceptance_rule():
    """A corrupted denominator previously produced wreckage that read as a clean null result."""
    reached = {"accept": False}

    def accept(ev):
        reached["accept"] = True
        return True, "should never be asked"

    trail = run_blocks(Incumbent(workspace=Path("."), incumbent_id="P0"),
                       diagnose=lambda inc: [_diag(f"c{i}") for i in range(10)],
                       propose=lambda ds, allow: [_cand()],
                       validate=lambda c: (None, "legal"),
                       evaluate=lambda inc, c: _ev(gains=("c1",), ok=False),
                       accept=accept, promote=lambda *a: None, max_rounds=1)
    assert trail[0].outcome == OUTCOME_BLOCKED
    assert not reached["accept"], "acceptance rule must not see an untrusted denominator"


def test_all_proposals_declined_is_a_policy_limit_not_completion():
    trail = run_blocks(Incumbent(workspace=Path("."), incumbent_id="P0"),
                       diagnose=lambda inc: [_diag(f"c{i}") for i in range(10)],
                       propose=lambda ds, allow: [_cand()],
                       validate=lambda c: (None, "action_not_executable: nope"),
                       evaluate=lambda inc, c: _ev(), accept=lambda ev: (True, ""),
                       promote=lambda *a: None, max_rounds=1)
    assert trail[0].outcome == OUTCOME_NO_CANDIDATE
    assert trail[0].outcome != STOP
    assert trail[0].declined[0][1].startswith("action_not_executable")


def test_rejected_round_continues_rather_than_stopping():
    trail = run_blocks(Incumbent(workspace=Path("."), incumbent_id="P0"),
                       diagnose=lambda inc: [_diag(f"c{i}") for i in range(10)],
                       propose=lambda ds, allow: [_cand()],
                       validate=lambda c: (None, "legal"),
                       evaluate=lambda inc, c: _ev(losses=("c1",)),
                       accept=lambda ev: (ev.net > 0, "net"),
                       promote=lambda *a: None, max_rounds=2)
    assert [r.outcome for r in trail] == [OUTCOME_REJECTED, OUTCOME_REJECTED]


def test_every_declined_proposal_keeps_its_reason():
    """A loop reporting only accepted anchors is unauditable."""
    trail = run_blocks(Incumbent(workspace=Path("."), incumbent_id="P0"),
                       diagnose=lambda inc: [_diag(f"c{i}") for i in range(10)],
                       propose=lambda ds, allow: [_cand("a"), _cand("b")],
                       validate=lambda c: ((None, "legal") if c.signal_name == "a"
                                           else (None, "signal_not_observable: b")),
                       evaluate=lambda inc, c: _ev(gains=("c1",)),
                       accept=lambda ev: (True, ""),
                       promote=lambda inc, c, ev: Incumbent(workspace=inc.workspace,
                                                            incumbent_id="P1", depth=1),
                       max_rounds=1)
    assert len(trail[0].declined) == 1
    assert trail[0].declined[0][0].signal_name == "b"
