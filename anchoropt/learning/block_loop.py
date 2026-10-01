"""The block-coordinate outer loop: POLICY and SIGNAL blocks, with the phase switch in ANCHOROPT.

WHAT THIS ADDS TO `self_evolve.py`
---------------------------------
`self_evolve.run()` already sequences the blocks and `self_evolve.step()` already implements one
policy round end-to-end. They were disconnected: `run()` had the phase logic and no evaluation,
`step()` had the evaluation and no phase logic, so neither could execute the method as specified.
This module is the join, and it adds no decision of its own beyond that sequencing.

THE TWO BLOCKS ARE DISTINCT, AND THE SWITCH IS OURS
--------------------------------------------------
    POLICY block   Phi FROZEN. Choose only (l, existing phi, mu, theta). A proposal carrying a new
                   signal expression is REJECTED here (`proposal_seams.ingest_proposal`).
    SIGNAL block   Propose a NEW declarative phi, validate it, add it, FREEZE Phi, return to POLICY.

`decide_phase` is injected but it is AnchorOpt's rule -- `phase_switch.decide`'s S1/S2/S5 thresholds,
or `residual_phase` below for callers without BFCL artifact dirs. It is never the proposer's call. If
a language model decides when to expand its own vocabulary, there is no block structure left: the
whole point of holding one block fixed is that the other cannot move to accommodate it.

RESIDUAL-GUIDED, NOT A FROZEN ERROR LIST
----------------------------------------
`diagnose(incumbent)` takes the CURRENT incumbent and is re-called every round. Working down a fixed
initial error list is the failure this project has the clearest evidence against: A1's own success
created the residuals that A2, A3, A5 and A8 later addressed, so a loop that mines once cannot find
its own consequences. `docs/ANCHORS.md` calls re-mining after every install the clearest argument in
the project.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anchoropt.learning.self_evolve import (
    CONTINUE_POLICY, EXPAND_ATTRIBUTION, STOP, EvaluationResult, Incumbent,
    OUTCOME_ACCEPTED, OUTCOME_BLOCKED, OUTCOME_NO_CANDIDATE, OUTCOME_REJECTED,
)
from anchoropt.runtime import ResidualDiagnosis

# Inherited from STOPPING_CRITERIA_FROZEN.md via `phase_switch`. MIRRORED rather than imported for
# the same reason `self_evolve` mirrors the phase verdict strings: `phase_switch` pulls in the BFCL
# artifact readers at import time, so importing it here would make the generic loop require a BFCL
# checkout. The values are asserted equal to `phase_switch`'s by test, which is what keeps the two
# copies from drifting -- a second copy of a threshold is otherwise how two parts of a loop start
# disagreeing about what was decided.
S1_COVERAGE_FLOOR = 0.10
S2_SATURATION_CEIL = 0.50


@dataclass(frozen=True)
class RoundRecord:
    """What one round did, and why. The audit trail IS the deliverable.

    A loop reporting only its accepted anchors is unauditable: the deferrals are where the
    acceptance criteria become legible, so every declined candidate keeps its reason code.
    """

    iteration: int
    block: str                                  # CONTINUE_POLICY | EXPAND_ATTRIBUTION | STOP
    incumbent_id: str
    n_diagnoses: int
    phase_reasons: tuple[str, ...] = ()
    proposals: tuple[Any, ...] = ()
    declined: tuple[tuple[Any, str], ...] = ()
    evaluated: tuple[tuple[Any, EvaluationResult], ...] = ()
    accepted: Any = None
    new_signals: tuple[str, ...] = ()
    outcome: str = ""
    note: str = ""


def residual_phase(diagnoses: Sequence[ResidualDiagnosis], *,
                   expressible: Callable[[ResidualDiagnosis], bool],
                   engaged: Mapping[str, int] | None = None,
                   group_key: Callable[[ResidualDiagnosis], str] | None = None
                   ) -> tuple[str, list[str]]:
    """A benchmark-neutral form of `phase_switch.decide`, over diagnoses rather than artifact dirs.

    The RULE is the imported one and the thresholds are the frozen ones; only the input shape
    differs, because `phase_switch.partition` reads BFCL trajectory directories and this must run for
    any benchmark.

        S1  a locus must cover at least S1_COVERAGE_FLOOR of the residual to be worth a round
        S2  a locus already SETTLED by the incumbent above S2_SATURATION_CEIL is refinement, not
            discovery, and does not justify a policy round
        S5  when nothing survives, the terminal case depends on whether the INEXPRESSIBLE mass
            dominates the expressible: if it does, the limit is the REPRESENTATION (expand Phi);
            otherwise the residual is addressed and the loop stops

    S5 is the distinction that matters for research: "no candidate" means Phi is the limit, not that
    the work is done, and reporting the two as one hides the only signal that says grow the
    vocabulary.
    """
    from anchoropt.learning.proposal_seams import compute_support

    # WHAT COUNTS AS ONE LOCUS MUST NOT DIFFER BETWEEN THIS RULE AND THE SEARCH.
    #
    # `compute_support` groups by normalized PROSE, so twelve rewordings of one decision are twelve
    # loci. Measured on a real round: pooling merged those twelve into FIVE problems with a top support
    # of 14 of 24 (58% of the residual), while this rule -- handed the unpooled diagnoses -- saw twelve
    # loci at 8.3% each, so every one failed S1 (>= 10%) and the verdict was EXPAND_ATTRIBUTION. The
    # representation was not the limit; the GROUPING was.
    #
    # `group_key` lets the caller supply the same equivalence the search uses (the observable condition
    # a diagnosis implicates). Default is unchanged, so no existing caller moves.
    n = max(len(diagnoses), 1)
    if group_key is not None:
        support: dict[str, int] = {}
        for d in diagnoses:
            support[str(group_key(d))] = support.get(str(group_key(d)), 0) + 1
    else:
        support = dict(compute_support(diagnoses))
    engaged = engaged or {}
    reasons: list[str] = []
    surviving: list[str] = []

    # Group diagnoses by decision so expressibility can be asked PER LOCUS, not per case.
    def _key(d) -> str:
        if group_key is not None:
            return str(group_key(d))
        return " ".join(str(d.consequential_decision).lower().split())

    by_key: dict[str, list[ResidualDiagnosis]] = {}
    for d in diagnoses:
        by_key.setdefault(_key(d), []).append(d)

    for key, count in sorted(support.items(), key=lambda kv: (-kv[1], kv[0])):
        cov = count / n
        settled = engaged.get(key, 0)
        sat = settled / max(count, 1)
        s1 = cov >= S1_COVERAGE_FLOOR
        s2 = sat < S2_SATURATION_CEIL
        # A locus no declared signal can OBSERVE cannot be addressed by a policy round, however
        # much of the residual it covers. Omitting this let an inexpressible locus clear S1 and
        # sent the loop into a POLICY round that had nothing legal to propose -- which is reported
        # as `no_candidate` and looks like policy exhaustion, hiding the fact that the real limit
        # is the REPRESENTATION. That is exactly the distinction S5 exists to preserve.
        s3 = any(expressible(d) for d in by_key.get(key, ()))
        if s1 and s2 and s3:
            surviving.append(key)
        reasons.append("%-46s cov %5.1f%% %-8s sat %3.0f%% %-8s %s"
                       % (key[:46], 100 * cov, "S1-ok" if s1 else "S1-STOP",
                          100 * sat, "S2-ok" if s2 else "S2-STOP",
                          "expressible" if s3 else "SIGNAL_BLOCKED"))

    if surviving:
        return CONTINUE_POLICY, reasons

    inexpressible = [d for d in diagnoses if not expressible(d)]
    largest_expressible = max((c for k, c in support.items()
                               if any(expressible(d) for d in diagnoses if _key(d) == k)),
                              default=0)
    dominates = len(inexpressible) > largest_expressible
    reasons += [
        "",
        "no locus survives the policy gates",
        "  largest expressible candidate : %d (%.0f%% of residual)"
        % (largest_expressible, 100 * largest_expressible / n),
        "  inexpressible residual        : %d (%.0f%% of residual)"
        % (len(inexpressible), 100 * len(inexpressible) / n),
        "  S5 -- inexpressible dominates : %s" % dominates,
    ]
    return (EXPAND_ATTRIBUTION if dominates else STOP), reasons


def run_blocks(incumbent: Incumbent, *, diagnose, propose, evaluate, accept, promote,
               validate, decide_phase=None, expressible=None, expand=None,
               max_rounds: int = 4, top_k: int = 3) -> list[RoundRecord]:
    """Drive the block-coordinate loop. Every decision function is INJECTED; order is decided here.

        diagnose(incumbent)               -> Sequence[ResidualDiagnosis]      re-mined EVERY round
        propose(diagnoses, allow_new_phi) -> Sequence[ProposedCandidate]
        validate(candidate)               -> (compiled_or_None, reason)       AnchorOpt's legality
        evaluate(incumbent, candidate)    -> EvaluationResult                 paired, denominator-gated
        accept(EvaluationResult)          -> (bool, reason)                   the acceptance rule
        promote(incumbent, candidate, ev) -> Incumbent                        moving incumbent
        expand(diagnoses)                 -> Sequence[str]                    newly ADDED phi names

    Returns the audit trail. Stops on STOP, on `max_rounds`, or when an expansion proposes nothing --
    the last being the honest terminal state for "no representation can see the rest of the
    residual", which is NOT the same as done and must not be reported as such.
    """
    trail: list[RoundRecord] = []
    expressible = expressible or (lambda _d: True)

    for i in range(1, max_rounds + 1):
        # RE-MINE against the CURRENT incumbent. Not a frozen list: A1's success created the
        # residuals A2/A3/A5/A8 addressed.
        diagnoses = tuple(diagnose(incumbent))
        if decide_phase is not None:
            phase, reasons = decide_phase(diagnoses)
        else:
            phase, reasons = residual_phase(diagnoses, expressible=expressible)

        if phase == STOP:
            trail.append(RoundRecord(
                iteration=i, block=STOP, incumbent_id=incumbent.incumbent_id,
                n_diagnoses=len(diagnoses), phase_reasons=tuple(reasons), outcome=STOP,
                note="residual addressed; no locus survives and the inexpressible mass is small"))
            return trail

        # ---- SIGNAL block: the ONLY place a new phi may be authored -----------------------------
        if phase == EXPAND_ATTRIBUTION:
            new_signals = tuple(expand(diagnoses)) if expand else ()
            trail.append(RoundRecord(
                iteration=i, block=EXPAND_ATTRIBUTION, incumbent_id=incumbent.incumbent_id,
                n_diagnoses=len(diagnoses), phase_reasons=tuple(reasons),
                new_signals=new_signals, outcome=EXPAND_ATTRIBUTION,
                note=(f"Phi expanded by {list(new_signals)}; now frozen again" if new_signals
                      else "expansion proposed nothing that discriminates; the residual is not "
                           "addressable in any currently proposable representation")))
            if not new_signals:
                return trail
            continue        # Phi is frozen again; the next iteration is a POLICY round

        # ---- POLICY block: Phi frozen, choose (l, mu, theta) ------------------------------------
        proposals = tuple(propose(diagnoses, False))
        legal: list[Any] = []
        declined: list[tuple[Any, str]] = []
        for cand in proposals:
            _compiled, why = validate(cand)
            (legal if why == "legal" else declined).append(cand if why == "legal" else (cand, why))

        if not legal:
            trail.append(RoundRecord(
                iteration=i, block=CONTINUE_POLICY, incumbent_id=incumbent.incumbent_id,
                n_diagnoses=len(diagnoses), phase_reasons=tuple(reasons), proposals=proposals,
                declined=tuple(declined), outcome=OUTCOME_NO_CANDIDATE,
                note="every proposal was declined; this is a POLICY limit under a frozen Phi, "
                     "not a completed optimization"))
            return trail

        # MEASUREMENT CHOOSES among the screened top-K. The ranking's job is cost control only: a
        # deterministic tie-break picking the controller before anything is measured is the failure
        # `select_top_k` exists to prevent.
        screened = legal[:max(1, int(top_k))]
        evaluated: list[tuple[Any, EvaluationResult]] = []
        blocked = False
        for cand in screened:
            ev = evaluate(incumbent, cand)
            # Denominator integrity is a PRECONDITION, not a result: an arm whose case coverage
            # cannot be trusted must never reach the acceptance rule. A corrupted denominator once
            # produced wreckage that read as a clean null result.
            if not ev.denominator_ok:
                trail.append(RoundRecord(
                    iteration=i, block=CONTINUE_POLICY, incumbent_id=incumbent.incumbent_id,
                    n_diagnoses=len(diagnoses), phase_reasons=tuple(reasons),
                    proposals=proposals, declined=tuple(declined),
                    evaluated=tuple(evaluated), outcome=OUTCOME_BLOCKED,
                    note=f"DENOMINATOR INTEGRITY FAILED: {ev.detail}"))
                blocked = True
                break
            evaluated.append((cand, ev))
        if blocked:
            return trail

        winners = [(c, ev) for c, ev in evaluated if accept(ev)[0]]
        if not winners:
            trail.append(RoundRecord(
                iteration=i, block=CONTINUE_POLICY, incumbent_id=incumbent.incumbent_id,
                n_diagnoses=len(diagnoses), phase_reasons=tuple(reasons), proposals=proposals,
                declined=tuple(declined), evaluated=tuple(evaluated), outcome=OUTCOME_REJECTED,
                note=f"{len(evaluated)} candidate(s) measured, none cleared the acceptance rule"))
            continue

        best, best_ev = max(winners, key=lambda ce: ce[1].net)
        new_incumbent = promote(incumbent, best, best_ev)
        trail.append(RoundRecord(
            iteration=i, block=CONTINUE_POLICY, incumbent_id=incumbent.incumbent_id,
            n_diagnoses=len(diagnoses), phase_reasons=tuple(reasons), proposals=proposals,
            declined=tuple(declined), evaluated=tuple(evaluated), accepted=best,
            outcome=OUTCOME_ACCEPTED,
            note=f"promoted to {new_incumbent.incumbent_id} on net +{best_ev.net}; "
                 f"next round re-mines against it"))
        incumbent = new_incumbent

    return trail
