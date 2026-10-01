"""The optimizer's state vocabulary: what happened, and therefore which coordinate changes next.

AnchorOpt optimizes three coordinates over a residual failure:

    WHERE   the decision boundary to intervene at
    WHAT    the observable condition phi that decides when to act
    HOW     the action and its parameters eta, realized by a runtime primitive

A state here is not a label for a report. It is the input to a block-coordinate decision: each one
names the coordinate to move next, so the search schedule is a function of measured outcomes rather
than a fixed script. `NEXT_COORDINATE` is that function, and it is total over `STATES`.

WHY A SIXTH VOCABULARY WOULD HAVE BEEN WRONG. Core already had five overlapping schemes -- the
boundary states, the signal-expansion refusals, the arm-rejection codes, the policy outcomes and the
projection verdicts -- each owned by a different module and none of them answerable to "what do I try
next". This module does not replace them: every one of those codes is RETAINED verbatim as the
`detail`/`reason_code` of an Outcome, because they carry the specific fact (which eta field was
missing, which executor refused) that a generic state deliberately abstracts away. What is new is
exactly one thing: a state the scheduler can dispatch on.

UNMEASURED IS NOT UNSUCCESSFUL. `REALIZABLE_UNMEASURED` exists because a controller that was built
and grounded but never evaluated is a STRUCTURAL result and nothing more. Collapsing it into
`NO_BENEFIT` would report "the intervention did not help" about an intervention nobody ran, which is
the same class of error as believing a null from a channel that was never live.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# ---- coordinates -------------------------------------------------------------------------------

WHERE = "WHERE"          # the decision boundary
WHAT = "WHAT"            # the observable condition (Phi)
HOW = "HOW"              # the action + eta, and the primitive that realizes it
DONE = "DONE"            # nothing further to try on this residual
PROMOTE = "PROMOTE"      # an improvement was measured; leave the search and install it

COORDINATES = (WHERE, WHAT, HOW, DONE, PROMOTE)

# ---- states ------------------------------------------------------------------------------------

# WHERE-level
BOUNDARY_NOT_REPAIRABLE = "BOUNDARY_NOT_REPAIRABLE"
BOUNDARY_EXHAUSTED = "BOUNDARY_EXHAUSTED"

# WHAT-level
SIGNAL_BLOCKED = "SIGNAL_BLOCKED"
SIGNAL_EXPANDED = "SIGNAL_EXPANDED"
SIGNAL_EXPANSION_EXHAUSTED = "SIGNAL_EXPANSION_EXHAUSTED"

# HOW-level
ACTION_UNAVAILABLE = "ACTION_UNAVAILABLE"
ETA_UNSUPPORTED = "ETA_UNSUPPORTED"
PRIMITIVE_MISSING = "PRIMITIVE_MISSING"
NO_CANDIDATE = "NO_CANDIDATE"

# A boundary where every built arm's phi fires on NONE of that boundary's observed states. The arms
# are structurally sound and inert HERE: running one reproduces the control exactly, so there is no
# paired contrast to measure. Distinct from NO_CANDIDATE (nothing was built) and from NO_BENEFIT
# (something was measured and did not help) -- this is "not measurable at this boundary", which is a
# reason to move the schedule on, never a statement about the residual or about benefit.
UNFIREABLE_HERE = "UNFIREABLE_HERE"

#: Every arm this boundary can build names a controller identity an EARLIER round already promoted.
#: That is a fact about the SEARCH HISTORY, not about the representation, so it must advance the
#: boundary and must NOT trigger Phi expansion -- expansion's trigger is a MEASURED insufficiency,
#: and nothing was measured here.
ALL_CELLS_ALREADY_DECIDED = "ALL_CELLS_ALREADY_DECIDED"

# outcome-level
REALIZABLE_UNMEASURED = "REALIZABLE_UNMEASURED"
NO_BENEFIT = "NO_BENEFIT"
IMPROVED = "IMPROVED"
# The search spent its declared evaluation budget with candidates still unmeasured. DISTINCT from
# NO_BENEFIT, which claims every candidate was measured and none helped: one is a statement about the
# controllers, the other about how much we were willing to spend looking. Collapsing them turns an
# unfinished search into a negative result about an intervention nobody ran.
SEARCH_BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"

STATES = (
    BOUNDARY_NOT_REPAIRABLE, BOUNDARY_EXHAUSTED,
    SIGNAL_BLOCKED, SIGNAL_EXPANDED, SIGNAL_EXPANSION_EXHAUSTED,
    ACTION_UNAVAILABLE, ETA_UNSUPPORTED, PRIMITIVE_MISSING, NO_CANDIDATE, UNFIREABLE_HERE,
    REALIZABLE_UNMEASURED, NO_BENEFIT, IMPROVED, SEARCH_BUDGET_EXHAUSTED,
    ALL_CELLS_ALREADY_DECIDED,
)

# ---- the block-coordinate transition ------------------------------------------------------------
#
# THE ORDERING IS THE ALGORITHMIC CLAIM, so it is stated once, as data, and tested directly:
#
#   at a fixed WHERE:  optimize existing WHAT x HOW first
#                      -> if candidates are realizable but none improves, exhaust HOW/eta
#                      -> then expand WHAT
#                      -> then retry HOW at the SAME boundary
#   move WHERE earlier ONLY when the boundary is structurally unrepairable, or its WHAT/HOW search
#   is exhausted.
#
# Read the table as "this happened, so change this coordinate next":
NEXT_COORDINATE: Mapping[str, str] = {
    # The residual does not manifest here at all, or everything here has been tried.
    BOUNDARY_NOT_REPAIRABLE: WHERE,
    BOUNDARY_EXHAUSTED: WHERE,

    # Phi cannot express the condition AT THIS BOUNDARY -> expand the representation, do not move.
    SIGNAL_BLOCKED: WHAT,
    # Expansion produced a signal -> immediately retry the action search HERE, with the wider Phi.
    SIGNAL_EXPANDED: HOW,
    # Phi cannot be widened any further here, so this boundary has nothing left.
    SIGNAL_EXPANSION_EXHAUSTED: WHERE,

    # The action space at this boundary cannot host the intervention -> try another action/eta.
    ACTION_UNAVAILABLE: HOW,
    ETA_UNSUPPORTED: HOW,
    PRIMITIVE_MISSING: HOW,
    # Nothing could be built from the current Phi here -> widen Phi before giving up on the boundary.
    NO_CANDIDATE: WHAT,
    # Arms were built and grounded and every one is INERT at this boundary: no phi fires on any of its
    # observed states, so none has a paired contrast here. Same next move as NO_CANDIDATE and for the
    # same reason -- the limit is what the current Phi can OBSERVE here, so widen Phi rather than
    # abandoning the boundary. Deliberately NOT `DONE`: a boundary whose arms cannot fire has produced
    # no verdict, and halting the schedule on it once cost a round the three-quarters of its
    # localization that lived at another boundary.
    UNFIREABLE_HERE: WHAT,
    # Every arm here names a controller identity an EARLIER round already promoted. -> WHERE, and
    # emphatically NOT `WHAT`: the limit is the SEARCH HISTORY, not what Phi can express here, and
    # widening the representation because a previous round already succeeded would re-enumerate the
    # grammar off a verdict nobody produced. Also not `DONE` -- the other boundaries are untouched.
    ALL_CELLS_ALREADY_DECIDED: WHERE,

    # Built and grounded, but evaluation was unavailable: a structural result, full stop. Reporting
    # it as a failure to improve would assert a measurement that never happened.
    REALIZABLE_UNMEASURED: DONE,
    # Measured and it did not help -> exhaust the remaining HOW/eta choices at this boundary first.
    NO_BENEFIT: HOW,
    IMPROVED: PROMOTE,
    # The budget ran out with candidates still unmeasured. DONE, like REALIZABLE_UNMEASURED and for
    # the same reason: the search has no verdict, so advancing a coordinate would act on a conclusion
    # it never reached. The round is REQUEUED with more budget, not continued -- which is what makes
    # this state different from NO_BENEFIT, whose next move is a real consequence of a real result.
    SEARCH_BUDGET_EXHAUSTED: DONE,
}

assert set(NEXT_COORDINATE) == set(STATES), "the transition table must be total over STATES"


def next_coordinate(state: str) -> str:
    """Which coordinate the optimizer changes next. Unknown states are conservative: stop."""
    return NEXT_COORDINATE.get(str(state), DONE)


def is_terminal(state: str) -> bool:
    return next_coordinate(state) in (DONE, PROMOTE)


# ---- feasibility certificates -------------------------------------------------------------------
#
# A rejected (boundary, signal, action, eta) must say WHY in a form the optimizer can act on. The
# specific codes already exist upstream (REJECT_*, *_INFEASIBLE, SIGNAL_REFUSED_*) and are better at
# naming the fact than any generic word would be -- so they are carried through untouched and only
# CLASSIFIED here. An unrecognised code becomes ETA_UNSUPPORTED rather than being dropped, and
# `certificate.classified` records whether the mapping was known: a new upstream code must be
# visible as unclassified, never silently bucketed.
_CODE_TO_STATE: Mapping[str, str] = {
    # arm rejection (anchor_policy_opt)
    "not_executable_in_host": ACTION_UNAVAILABLE,
    "signal_not_observable_at_boundary": SIGNAL_BLOCKED,
    "signal_params_unconfigured": SIGNAL_BLOCKED,
    "not_exactly_groundable": ETA_UNSUPPORTED,
    "no_action_parameters": ETA_UNSUPPORTED,
    "no_parameter_grid": ETA_UNSUPPORTED,
    "not_materializable_by_host_executor": PRIMITIVE_MISSING,
    # action contract
    "REROUTE_INFEASIBLE": ETA_UNSUPPORTED,
    "REPROMPT_INFEASIBLE": ETA_UNSUPPORTED,
    "SUPPRESS_INFEASIBLE": ETA_UNSUPPORTED,
    "TRANSFORM_INFEASIBLE": ETA_UNSUPPORTED,
    # signal expansion
    "SIGNAL_REFUSED_COLLISION": SIGNAL_EXPANSION_EXHAUSTED,
    "SIGNAL_REFUSED_UNCOMPUTABLE": SIGNAL_EXPANSION_EXHAUSTED,
    "SIGNAL_REFUSED_NOT_OBSERVABLE": SIGNAL_EXPANSION_EXHAUSTED,
    "SIGNAL_REFUSED_CONSTANT": SIGNAL_EXPANSION_EXHAUSTED,
    "SIGNAL_REFUSED_NO_RUNTIME_SUPPORT": SIGNAL_EXPANSION_EXHAUSTED,
}


@dataclass(frozen=True)
class Certificate:
    """Why one (boundary, signal, action) could not be realized -- generically classified.

    `state` is what the optimizer dispatches on; `reason_code` and `detail` are the upstream fact,
    retained so the specific cause survives the abstraction.
    """

    state: str
    reason_code: str = ""
    boundary: str = ""
    signal: str = ""
    action: str = ""
    missing: tuple[str, ...] = ()
    detail: str = ""
    classified: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state, "reason_code": self.reason_code, "boundary": self.boundary,
                "signal": self.signal, "action": self.action, "missing": list(self.missing),
                "detail": self.detail, "classified": self.classified}


def classify(reason_code: str, *, boundary: str = "", signal: str = "", action: str = "",
             missing: Sequence[str] = (), detail: str = "") -> Certificate:
    """Map an upstream rejection code onto a generic state, keeping the code itself."""
    code = str(reason_code or "")
    known = code in _CODE_TO_STATE
    return Certificate(state=_CODE_TO_STATE.get(code, ETA_UNSUPPORTED), reason_code=code,
                       boundary=str(getattr(boundary, "value", boundary) or ""),
                       signal=str(signal or ""), action=str(getattr(action, "value", action) or ""),
                       missing=tuple(str(m) for m in missing), detail=str(detail or ""),
                       classified=known)


def certificate_from(rejected: Any, *, boundary: str = "", signal: str = "") -> Certificate:
    """Build a Certificate from a `RejectedArm`-shaped record. Duck-typed on purpose."""
    return classify(getattr(rejected, "reason_code", "") or "",
                    boundary=boundary, signal=signal,
                    action=getattr(rejected, "action", "") or "",
                    missing=tuple(getattr(rejected, "missing", ()) or ()),
                    detail=str(getattr(rejected, "detail", "") or ""))


# ---- what happened at one (boundary, Phi) visit -------------------------------------------------


@dataclass
class Attempt:
    """One visit to one coordinate assignment, in enough detail to explain the schedule after."""

    boundary: str
    state: str
    coordinate_changed: str = ""
    label: str = ""
    signals_expanded: tuple[str, ...] = ()
    candidates_built: int = 0
    candidates_realizable: int = 0
    candidates_evaluated: int = 0
    #: Arms NOT evaluated because an earlier round already promoted their (boundary, signal, action)
    #: identity. Reported rather than dropped: a round must be able to say what it declined to
    #: re-measure, and the alternative -- a silently smaller `candidates_evaluated` -- is
    #: indistinguishable from a search that built fewer arms.
    cells_already_decided: int = 0
    certificates: list[Certificate] = field(default_factory=list)
    objective: Any = None
    promoted: Any = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.coordinate_changed:
            self.coordinate_changed = next_coordinate(self.state)

    def as_dict(self) -> dict[str, Any]:
        return {"boundary": self.boundary, "state": self.state,
                "coordinate_changed": self.coordinate_changed, "label": self.label,
                "signals_expanded": list(self.signals_expanded),
                "candidates_built": self.candidates_built,
                "candidates_realizable": self.candidates_realizable,
                "candidates_evaluated": self.candidates_evaluated,
                "cells_already_decided": self.cells_already_decided,
                "certificates": [c.as_dict() for c in self.certificates],
                "objective": self.objective,
                "promoted": (self.promoted.as_dict()
                             if hasattr(self.promoted, "as_dict") else self.promoted),
                "detail": self.detail}
