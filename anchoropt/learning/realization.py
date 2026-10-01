"""Projecting a proposed policy onto what the runtime can actually realize -- verified, not asserted.

    LLM attribution -> LLM policy hypothesis -> STRUCTURED feasibility + realization checks
      -> measured evaluation -> promote/reject

A proposer may notice that a synthesized parameter is not settable and offer the nearest realizable
policy instead. That is a useful thing for a model to do: it is a mapping from a semantic hypothesis to
a runtime policy class, which is exactly the judgement a fixed rule table does badly.

WHAT THE MODEL MAY AND MAY NOT DO
    MAY   propose the realization, and give a rationale
    MAY NOT certify that the realization is equivalent

Equivalence is decided HERE, deterministically, by comparing TRIGGER SETS on observed states. The
project has already paid for the alternative: a candidate whose parameter the executor silently coerced
would have measured a trigger nobody proposed, and a model asked "is this close enough?" has no way to
answer except by sounding confident.

TOLERANCE IS PREDECLARED, before any proposal is seen, for the obvious reason: a threshold chosen after
looking at the disagreement is not a test. `MAX_TRIGGER_DISAGREEMENT` is the fraction of the UNION of
the two trigger sets that may differ. 0.15 admits a projection that moves at most a handful of states
out of a nine-state trigger and refuses one that reshapes it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# PREDECLARED. Do not tune this against a result.
MAX_TRIGGER_DISAGREEMENT = 0.15
MIN_OBSERVED_STATES = 8          # below this the comparison is not informative

PROJECTION_ACCEPTED = "PROJECTION_ACCEPTED"
PROJECTION_REFUSED_DIVERGENT = "PROJECTION_REFUSED_DIVERGENT"
PROJECTION_REFUSED_TOO_FEW_STATES = "PROJECTION_REFUSED_TOO_FEW_STATES"
PROJECTION_REFUSED_EMPTY_TRIGGER = "PROJECTION_REFUSED_EMPTY_TRIGGER"


@dataclass(frozen=True)
class ProjectionVerdict:
    """Whether a proposed realization may stand in for the policy it approximates."""

    state: str
    proposed_fires: int
    realizable_fires: int
    agree: int
    disagree: int
    total_states: int
    detail: str = ""
    rationale: str = ""          # the proposer's, recorded as provenance only

    @property
    def accepted(self) -> bool:
        return self.state == PROJECTION_ACCEPTED

    @property
    def disagreement(self) -> float | None:
        union = self.agree + self.disagree
        return (self.disagree / union) if union else None

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state, "proposed_fires": self.proposed_fires,
                "realizable_fires": self.realizable_fires, "agree": self.agree,
                "disagree": self.disagree, "total_states": self.total_states,
                "disagreement": self.disagreement,
                "tolerance": MAX_TRIGGER_DISAGREEMENT,
                "detail": self.detail, "proposer_rationale": self.rationale}


def verify_projection(proposed: Any, realizable: Any,
                      states: Sequence[Mapping[str, Any]], *,
                      rationale: str = "",
                      tolerance: float = MAX_TRIGGER_DISAGREEMENT) -> ProjectionVerdict:
    """Compare the TRIGGER SETS of a proposed policy and its proposed realization.

    Both arguments need only `.evaluate(state) -> bool`. Nothing about the proposer's reasoning enters
    the decision; `rationale` is carried for the record and never read.
    """
    if len(states) < MIN_OBSERVED_STATES:
        return ProjectionVerdict(PROJECTION_REFUSED_TOO_FEW_STATES, 0, 0, 0, 0, len(states),
                                 f"only {len(states)} observed states; need "
                                 f"{MIN_OBSERVED_STATES} for the comparison to mean anything",
                                 rationale)
    a = {i for i, s in enumerate(states) if proposed.evaluate(s)}
    b = {i for i, s in enumerate(states) if realizable.evaluate(s)}
    agree, disagree = len(a & b), len((a | b) - (a & b))
    if not (a or b):
        return ProjectionVerdict(PROJECTION_REFUSED_EMPTY_TRIGGER, 0, 0, 0, 0, len(states),
                                 "neither policy fires on any observed state, so they are trivially "
                                 "'equivalent' and both are inert", rationale)
    union = agree + disagree
    frac = disagree / union if union else 1.0
    if frac > tolerance:
        return ProjectionVerdict(PROJECTION_REFUSED_DIVERGENT, len(a), len(b), agree, disagree,
                                 len(states),
                                 f"trigger sets differ on {disagree}/{union} = {100*frac:.1f}% of "
                                 f"their union, above the predeclared {100*tolerance:.0f}%; the "
                                 f"realization would fire on a materially different population",
                                 rationale)
    return ProjectionVerdict(PROJECTION_ACCEPTED, len(a), len(b), agree, disagree, len(states),
                             f"trigger sets differ on {disagree}/{union} = {100*frac:.1f}% of their "
                             f"union, within the predeclared {100*tolerance:.0f}%", rationale)
