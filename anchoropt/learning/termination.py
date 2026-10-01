"""WHEN TO STOP, at three levels -- and the distinction between a negative result and a non-result.

    level 1  local WHERE/WHAT/HOW      already terminal in the frozen search; read, never changed
    level 2  one residual family       ResidualOutcome, below
    level 3  global harness evolution  GlobalVerdict, below

THE DISTINCTION THIS MODULE EXISTS FOR. "We measured it and it did not help" and "we could not
measure it" are different facts with opposite consequences: the first settles a residual family, the
second returns it to the queue. Collapsing them retires work that was never tested, and -- worse -- it
banks a broken measurement as evidence.

The defect was live. `self_evolve_cycle2.py` computed

    state = IMPROVED if (net > 0 and ok_denom) else NO_BENEFIT

so a denominator MISMATCH -- two runs that scored different cases, i.e. no paired comparison at all --
was recorded as NO_BENEFIT. `EVALUATION_INVALID` is checked FIRST here for exactly that reason, and it
never counts as a negative.

WHAT IS NOT CHANGED. The frozen search's own states (`search_state.STATES`) and its transition table
are untouched; this module CONSUMES them. `NO_BENEFIT` remains a search-local state meaning "try
another eta at this boundary". `NO_MEASURED_IMPROVEMENT` is the family-level verdict. Same English,
deliberately different scope, and both are kept rather than overloading one word.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anchoropt.learning.search_state import (
    BOUNDARY_EXHAUSTED, BOUNDARY_NOT_REPAIRABLE, IMPROVED, NO_CANDIDATE, REALIZABLE_UNMEASURED,
    SIGNAL_EXPANSION_EXHAUSTED,
)

# ---- level 2: one residual family ---------------------------------------------------------------

PROMOTED = "PROMOTED"
SEARCH_EXHAUSTED = "SEARCH_EXHAUSTED"
NO_MEASURED_IMPROVEMENT = "NO_MEASURED_IMPROVEMENT"
INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
EVALUATION_INVALID = "EVALUATION_INVALID"

OUTCOMES = (PROMOTED, SEARCH_EXHAUSTED, NO_MEASURED_IMPROVEMENT, INSUFFICIENT_SUPPORT,
            BUDGET_EXHAUSTED, EVALUATION_INVALID)

# ONLY THESE TWO ARE NEGATIVE RESULTS -- i.e. evidence that this family does not repay intervention
# with the current action space. Everything else is a NON-result: the question stands.
NEGATIVE_RESULTS = (NO_MEASURED_IMPROVEMENT, SEARCH_EXHAUSTED)

# Outcomes that SETTLE a family for this pass. A settled family is skipped for the remainder of the
# pass; it is not permanently closed -- SEARCH_EXHAUSTED in particular reopens when the action space
# grows, which is the existing ledger's `unresolved_no_policy` semantics.
SETTLES_FAMILY = (PROMOTED, NO_MEASURED_IMPROVEMENT, SEARCH_EXHAUSTED, INSUFFICIENT_SUPPORT)

# Outcomes that return the family to the queue unchanged, because nothing was learned about it.
REQUEUES_FAMILY = (EVALUATION_INVALID, BUDGET_EXHAUSTED)

# Mapping onto the EXISTING experiment ledger, rather than a parallel scheme. The ledger already drew
# these distinctions for a different layer, and its semantics line up exactly:
#   learned                -> promoted
#   rejected_no_op         -> measured, did not help
#   unresolved_no_policy   -> real signal, no action in the current space; reopens when it grows
#   underpowered_candidate -> the corpus cannot resolve an effect this size, either way
#   unresolved             -> still on the work list
LEDGER_STATUS = {
    PROMOTED: "learned",
    NO_MEASURED_IMPROVEMENT: "rejected_no_op",
    SEARCH_EXHAUSTED: "unresolved_no_policy",
    INSUFFICIENT_SUPPORT: "underpowered_candidate",
    EVALUATION_INVALID: "unresolved",
    BUDGET_EXHAUSTED: "unresolved",
}

assert set(LEDGER_STATUS) == set(OUTCOMES), "every outcome needs a ledger status"

# Support below this cannot carry a family-level verdict either way. Pre-declared, not tuned: with
# fewer than this many failures a single flaky episode moves the sign.
MIN_SUPPORT = 5


@dataclass(frozen=True)
class Validity:
    """Whether a paired evaluation may be believed AT ALL, before asking what it showed.

    Three independent ways a round can be uninterpretable, each measured on this project:
      * denominator      -- the two runs did not score the same cases, so it is not paired;
      * contamination    -- the arm modified state its own scored cases were graded against;
      * channel integrity -- the intervention did not run where the effect appeared, so whatever
        moved the number, it was not the controller;
      * evidence complete -- a REQUIRED acceptance criterion was unresolved rather than refuted, so
        there is nothing to classify yet.

    THE FOURTH IS NOT A VERDICT ABOUT THE CANDIDATE. Criteria 2, 3 and 4 return PENDING_VALIDATION
    when their evidence was never gathered, and PENDING is neither a pass nor a refutation. Routing it
    through `channel_ok` would assert something false about the intervention channel; routing it
    through `accepted=False` would bank a NEGATIVE RESULT against a candidate nobody measured, which
    is precisely the "reading silence as a result" failure this lattice exists to prevent. So it gets
    its own flag: the round is UNINTERPRETABLE, the outcome is EVALUATION_INVALID, and the question
    about the candidate stands open.
    """

    denominator_ok: bool = True
    contamination_free: bool = True
    channel_ok: bool = True
    evidence_complete: bool = True
    detail: str = ""
    missing_evidence: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return (self.denominator_ok and self.contamination_free and self.channel_ok
                and self.evidence_complete)

    def reasons(self) -> tuple[str, ...]:
        out = []
        if not self.denominator_ok:
            out.append("denominator mismatch: the runs did not score the same cases")
        if not self.contamination_free:
            out.append("contamination: the arm altered state its own cases were graded against")
        if not self.channel_ok:
            out.append("channel integrity: the effect did not appear where the controller fired")
        if not self.evidence_complete:
            out.append("required acceptance evidence was never gathered, so the candidate is "
                       "UNRESOLVED rather than refuted: "
                       + ("; ".join(self.missing_evidence) or "unspecified"))
        return tuple(out)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "denominator_ok": self.denominator_ok,
                "contamination_free": self.contamination_free, "channel_ok": self.channel_ok,
                "evidence_complete": self.evidence_complete,
                "missing_evidence": list(self.missing_evidence),
                "reasons": list(self.reasons()), "detail": self.detail}


def channel_integrity(*, gains: int, gains_on_fired: int, firings: int,
                      min_fraction_on_fired: float | None = None) -> tuple[bool, str]:
    """Did the intervention plausibly CAUSE the gains?

    Two checks, and the first is not a judgement call:
      * gains with ZERO firings -- the controller cannot have caused an effect it never produced;
      * gains that sit off fired cases -- each one has a cause other than the controller.

    `min_fraction_on_fired` is the fraction of gains that must sit on a fired case. LEFT AS None BY
    DEFAULT, deliberately: on a real round 11 of 22 gains were off fired cases, so any threshold above
    0.5 reclassifies a result already on the record. That is a call for whoever owns the claim, not a
    number to pick inside a helper. With None, only the zero-firings check applies and the fraction is
    REPORTED so it cannot be overlooked.
    """
    if gains and firings == 0:
        return False, (f"{gains} gain(s) with ZERO firings -- the controller cannot have caused an "
                       f"effect it never produced")
    if not gains:
        return True, "no gains to attribute"
    frac = gains_on_fired / gains
    msg = f"{gains_on_fired}/{gains} gains on a fired case ({frac:.0%})"
    if min_fraction_on_fired is None:
        return True, msg + " (no threshold declared; reported, not enforced)"
    if frac < min_fraction_on_fired:
        return False, msg + f" -- below the declared {min_fraction_on_fired:.0%}"
    return True, msg + f" -- meets the declared {min_fraction_on_fired:.0%}"


@dataclass
class ResidualOutcome:
    """Why the loop stopped working on ONE residual family, and what that licenses."""

    family: str
    outcome: str
    support: int = 0
    candidates_built: int = 0
    candidates_evaluated: int = 0
    validity: Validity = field(default_factory=Validity)
    search_state: str = ""
    detail: str = ""
    novel_candidates_remaining: int = 0

    @property
    def is_negative_result(self) -> bool:
        """Evidence that this family does not repay intervention. NOT merely 'we stopped'."""
        return self.outcome in NEGATIVE_RESULTS

    @property
    def settles_family(self) -> bool:
        return self.outcome in SETTLES_FAMILY

    @property
    def ledger_status(self) -> str:
        return LEDGER_STATUS[self.outcome]

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, "outcome": self.outcome, "support": self.support,
                "candidates_built": self.candidates_built,
                "candidates_evaluated": self.candidates_evaluated,
                "validity": self.validity.as_dict(), "search_state": self.search_state,
                "is_negative_result": self.is_negative_result,
                "settles_family": self.settles_family, "ledger_status": self.ledger_status,
                "novel_candidates_remaining": self.novel_candidates_remaining,
                "detail": self.detail}


def classify_residual(*, family: str, search_outcome: Any, support: int,
                      evaluation: Mapping[str, Any] | None = None,
                      validity: Validity | None = None,
                      budget_remaining: bool = True,
                      min_support: int = MIN_SUPPORT,
                      novel_candidates_remaining: int = 0) -> ResidualOutcome:
    """Derive the family-level outcome. PRECEDENCE IS TOTAL and validity comes first.

        EVALUATION_INVALID -> INSUFFICIENT_SUPPORT -> PROMOTED -> BUDGET_EXHAUSTED
                           -> NO_MEASURED_IMPROVEMENT -> SEARCH_EXHAUSTED

    Validity first because an invalid round has no result to classify; support next because a verdict
    on too few cases is not a verdict; and SEARCH_EXHAUSTED last because it is the only outcome that
    requires nothing to have been measurable at all.
    """
    built = len(getattr(search_outcome, "candidates", ()) or ())
    state = str(getattr(search_outcome, "state", "") or "")
    n_eval = int((evaluation or {}).get("candidates_evaluated", 0) or 0)
    kw = dict(family=family, support=support, candidates_built=built,
              candidates_evaluated=n_eval, search_state=state,
              novel_candidates_remaining=novel_candidates_remaining)

    # 1. VALIDITY. An unbelievable measurement is not a negative result.
    if evaluation is not None and validity is not None and not validity.ok:
        return ResidualOutcome(outcome=EVALUATION_INVALID, validity=validity,
                               detail="; ".join(validity.reasons()), **kw)

    # 2. SUPPORT. Too few cases to carry a verdict either way.
    if support < min_support:
        return ResidualOutcome(outcome=INSUFFICIENT_SUPPORT,
                               validity=validity or Validity(),
                               detail=f"support {support} < declared floor {min_support}", **kw)

    # 3. A measured improvement.
    if evaluation is not None and bool(evaluation.get("accepted")):
        return ResidualOutcome(outcome=PROMOTED, validity=validity or Validity(),
                               detail="acceptance met on a valid paired evaluation", **kw)

    # 4. BUDGET. Stopped early with work left -- explicitly not a null.
    if not budget_remaining and (novel_candidates_remaining or not built):
        return ResidualOutcome(outcome=BUDGET_EXHAUSTED, validity=validity or Validity(),
                               detail=f"budget exhausted with {novel_candidates_remaining} novel "
                                      f"feasible candidate(s) unexplored", **kw)

    # 5. Something was measured and did not clear the bar. THE negative result.
    if n_eval:
        return ResidualOutcome(outcome=NO_MEASURED_IMPROVEMENT, validity=validity or Validity(),
                               detail=f"{n_eval} candidate(s) evaluated, none met acceptance", **kw)

    # 6. Nothing measurable could be built here, after the allowed expansion.
    exhausted = state in (SIGNAL_EXPANSION_EXHAUSTED, BOUNDARY_EXHAUSTED, BOUNDARY_NOT_REPAIRABLE,
                          NO_CANDIDATE, REALIZABLE_UNMEASURED, IMPROVED)
    return ResidualOutcome(
        outcome=SEARCH_EXHAUSTED, validity=validity or Validity(),
        detail=(f"no feasible candidate after boundary/signal/action expansion (search={state})"
                if exhausted else f"search ended in {state!r} with {built} candidate(s), none "
                                  f"evaluated"), **kw)


# ---- level 3: global harness evolution ----------------------------------------------------------

GLOBAL_CONTINUE = "CONTINUE"
GLOBAL_STOP_EXHAUSTED = "STOP_SEARCH_SPACE_EXHAUSTED"
GLOBAL_STOP_BUDGET = "STOP_BUDGET_EXHAUSTED"


@dataclass
class GlobalVerdict:
    """Whether harness evolution should continue, and on what basis it may stop.

    STOPPING REQUIRES A COMPLETE PASS. A single rejected arm must never terminate evolution -- that is
    the failure mode this level exists to prevent, and it is easy to hit because a rejection is the
    most visible thing a round produces.
    """

    verdict: str
    families_eligible: int = 0
    families_settled: int = 0
    promotions: int = 0
    novel_candidates_remaining: int = 0
    requeued: tuple[str, ...] = ()
    invalid: tuple[str, ...] = ()
    detail: str = ""

    @property
    def should_continue(self) -> bool:
        return self.verdict == GLOBAL_CONTINUE

    @property
    def pass_complete(self) -> bool:
        return self.families_eligible > 0 and self.families_settled >= self.families_eligible

    def as_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "should_continue": self.should_continue,
                "pass_complete": self.pass_complete,
                "families_eligible": self.families_eligible,
                "families_settled": self.families_settled, "promotions": self.promotions,
                "novel_candidates_remaining": self.novel_candidates_remaining,
                "requeued": list(self.requeued), "invalid": list(self.invalid),
                "detail": self.detail}


def global_verdict(outcomes: Sequence[ResidualOutcome], *, budget_remaining: bool = True,
                   families_eligible: int | None = None) -> GlobalVerdict:
    """Stop only on a COMPLETE pass with no promotion and nothing novel left, or on budget.

    Anything that merely requeues a family (an invalid evaluation, a budget stop) keeps the loop
    alive, because the family's question is still open.
    """
    eligible = families_eligible if families_eligible is not None else len(outcomes)
    settled = [o for o in outcomes if o.settles_family]
    promotions = [o for o in outcomes if o.outcome == PROMOTED]
    requeued = tuple(o.family for o in outcomes if o.outcome in REQUEUES_FAMILY)
    invalid = tuple(o.family for o in outcomes if o.outcome == EVALUATION_INVALID)
    novel = sum(o.novel_candidates_remaining for o in outcomes)
    kw = dict(families_eligible=eligible, families_settled=len(settled),
              promotions=len(promotions), novel_candidates_remaining=novel,
              requeued=requeued, invalid=invalid)

    if promotions:
        return GlobalVerdict(GLOBAL_CONTINUE,
                            detail=f"{len(promotions)} promotion(s): regenerate trajectories and "
                                   f"re-mine the changed residual distribution", **kw)
    if not budget_remaining:
        return GlobalVerdict(GLOBAL_STOP_BUDGET,
                            detail=f"predeclared budget exhausted with {novel} novel feasible "
                                   f"candidate(s) unexplored -- NOT a null result", **kw)
    if requeued:
        return GlobalVerdict(GLOBAL_CONTINUE,
                            detail=f"{len(requeued)} family requeued ({', '.join(requeued[:3])}) -- "
                                   f"their questions were never answered", **kw)
    if len(settled) < eligible:
        return GlobalVerdict(GLOBAL_CONTINUE,
                            detail=f"pass incomplete: {len(settled)}/{eligible} families settled",
                            **kw)
    if novel:
        return GlobalVerdict(GLOBAL_CONTINUE,
                            detail=f"pass complete but {novel} novel feasible candidate(s) remain",
                            **kw)
    return GlobalVerdict(GLOBAL_STOP_EXHAUSTED,
                        detail=f"complete pass over {eligible} eligible family(ies), no promotion, "
                               f"no novel feasible candidate remaining", **kw)
