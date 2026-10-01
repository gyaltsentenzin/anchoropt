"""WHICH candidate to evaluate next, decided by a rule fixed in advance.

WHY THIS MODULE EXISTS. A round used to end by printing "N MEASURABLE ARM(S) EMITTED -- none is preferred
here; selection requires measurement", which is correct about measurement being the judge but leaves the
CHOICE OF WHAT TO MEASURE outside the system. In the orchestration pilot a human then read the residual
ranking and picked. The reasoning was sound and it was still an external decision, so the run could not
be called unattended.

The fix is not to let the optimizer guess a winner -- measurement still decides that. It is to make the
question "which hypothesis do we spend the next paired evaluation on?" answerable by a deterministic,
pre-registered policy, so the same round always selects the same arm and the choice is auditable.

WHAT IS AND IS NOT A PRIOR HERE. Rules 1, 2, 4 and 5 are bookkeeping: eliminate what cannot run, use the
rank order the pipeline already computed, prefer the narrower hypothesis, break ties reproducibly.

Rule 3 is the ONE substantive prior -- recovery before suppression -- and it is carried with its evidence
rather than hidden: on this corpus, commitment-gate suppression measured -3 and -4 because withholding a
call diverts the trajectory, while the two recovery mechanisms measured +5 and +4. A prior stated with its
measurement can be argued with; an unstated one cannot. It orders arms only WITHIN one support level, so
it never overrides the residual's own ranking.

CORE STAYS BENCHMARK-FREE. Everything below reads declared fields -- operator name, support count,
projected firing count, label -- and nothing about what a signal observes or what a store does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# Operator families, coarsest first. The ORDER IS THE PRIOR: an action that repairs the situation and
# lets the original operation succeed is tried before one that cancels the operation.
RECOVERY_OPERATORS = ("transform", "substitute")
REPROMPT_OPERATORS = ("reprompt",)
SUPPRESSION_OPERATORS = ("suppress",)

# Lower sorts first.
_FAMILY_RANK = {**{o: 0 for o in RECOVERY_OPERATORS},
                **{o: 1 for o in REPROMPT_OPERATORS},
                **{o: 2 for o in SUPPRESSION_OPERATORS}}
_UNKNOWN_FAMILY_RANK = 3        # an undeclared operator family sorts last, never first

# Why an arm was excluded. Named so a skipped arm is auditable rather than silently absent.
SETTLED = "settled_already_installed_and_accepted"
MEASURED_LOSER = "excluded_measured_no_benefit_or_negative"
INFEASIBLE_FOR_CONSTRAINT = "operator_infeasible_for_the_constraint"
CAPABILITY_MISMATCH = "capability_id_does_not_match_the_resolved_executor"
NEVER_FIRES = "fires_on_zero_projected_states"

POLICY_EXHAUSTED = "POLICY_EXHAUSTED_CURRENT_SPACE"


#: How a `support` number was arrived at. The selector RANKS on support, so a number's provenance is
#: part of its meaning and must travel with it.
#:
#: MEASURED      counted against live host state, or against an executed run
#: PROJECTED     reconstructed offline from incomplete state -- an UPPER BOUND, not a count
#: UNKNOWN       the round did not say, which is treated as PROJECTED (the conservative direction)
SUPPORT_MEASURED = "measured"
SUPPORT_PROJECTED = "projected"
SUPPORT_UNKNOWN = "unknown"
_SUPPORT_PROVENANCE = (SUPPORT_MEASURED, SUPPORT_PROJECTED, SUPPORT_UNKNOWN)


@dataclass(frozen=True)
class Candidate:
    """One emitted arm, as the selector sees it. Every field is declared by the round, not inferred."""

    label: str
    operator: str
    signal: str
    boundary: str
    support: int = 0
    fires_on_states: int = 0
    total_states: int = 0
    capability_id: str = ""
    resolved_capability_id: str = ""
    feasible: bool = True
    spec: Mapping[str, Any] = field(default_factory=dict)
    #: WHERE `support` CAME FROM. Defaults to UNKNOWN, and UNKNOWN is treated as PROJECTED, so a caller
    #: that does not think about it cannot accidentally get measured-grade ranking.
    #:
    #: THE DEFECT THIS ANSWERS, measured on BFCL: an arm was ranked on support 8 and 201/590 projected
    #: firings; the host's LIVE comparator found exactly ONE qualifying case -- a 201x overstatement,
    #: because the offline reconstruction could not see state an earlier phase had already built. The
    #: selector ranked on that number as though it were a count, and a GPU round was spent on a
    #: one-case mechanism. Unknown live state must not convert into confident support.
    support_provenance: str = SUPPORT_UNKNOWN

    def __post_init__(self) -> None:
        if self.support_provenance not in _SUPPORT_PROVENANCE:
            raise ValueError(
                f"support_provenance {self.support_provenance!r} not in {_SUPPORT_PROVENANCE}; "
                f"an undeclared provenance must be UNKNOWN rather than invented")

    @property
    def support_is_measured(self) -> bool:
        """Only an explicitly MEASURED support may be treated as a count."""
        return self.support_provenance == SUPPORT_MEASURED

    @property
    def effective_support(self) -> int:
        """The support the selector may rank on.

        A PROJECTED (or UNKNOWN) number is an upper bound, so it cannot outrank a MEASURED one on its
        face value. It is not discarded -- that would make an unmeasured candidate unselectable and
        block every first-round arm -- it simply loses ties to measured evidence of the same size, via
        the sort key below. This method reports the number; `_sort_key` carries the demotion.
        """
        return int(self.support)

    @property
    def family_rank(self) -> int:
        return _FAMILY_RANK.get(str(self.operator).lower(), _UNKNOWN_FAMILY_RANK)


@dataclass(frozen=True)
class Exclusion:
    label: str
    reason: str
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.label}: {self.reason}" + (f" -- {self.detail}" if self.detail else "")


@dataclass
class Selection:
    """What the policy chose, and why every other arm was not chosen."""

    chosen: Candidate | None
    ranked: tuple[Candidate, ...] = ()
    excluded: tuple[Exclusion, ...] = ()
    terminal: str = ""
    # WHICH POLICY CHOSE. Recorded so a result cannot be quoted without the prior that produced it.
    policy: str = ""

    @property
    def ok(self) -> bool:
        return self.chosen is not None

    def rationale(self) -> str:
        """A human-readable audit trail. Printed into the round record."""
        lines: list[str] = [f"policy={self.policy or '<unset>'}"]
        if self.chosen is None:
            lines.append(f"NO ELIGIBLE CANDIDATE -> {self.terminal or POLICY_EXHAUSTED}")
        else:
            c = self.chosen
            lines.append(f"SELECTED {c.label}")
            lines.append(f"  support={c.support} [{c.support_provenance}] "
                         f"family={c.operator}(rank {c.family_rank}) "
                         f"fires={c.fires_on_states}/{c.total_states}")
            if not c.support_is_measured:
                # Said in the audit trail, not only in a docstring: a round that SELECTED on an
                # unmeasured number should be readable as having done so.
                lines.append("  NOTE: support is an UPPER BOUND from an offline projection, not a "
                             "count. One measured instance overstated by 201x.")
        for i, c in enumerate(self.ranked[1:], start=2):
            lines.append(f"  #{i} {c.label}  support={c.support} family_rank={c.family_rank} "
                         f"fires={c.fires_on_states}")
        for e in self.excluded:
            lines.append(f"  EXCLUDED {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------
# TWO POLICIES, AND WHY BOTH EXIST
#
# NEUTRAL is the scientific default: feasibility, the residual's own support, firing count, label. It
# encodes NO belief about which action family works, so a result under it is evidence about the search
# procedure itself rather than about our history with one corpus.
#
# HISTORY_INFORMED adds exactly one ordering term -- recovery before reprompt before suppression -- and
# that term comes from OUR MEASUREMENTS on the Granite/BFCL corpus: commitment-gate suppression measured
# -3 and -4 because withholding a call diverts the trajectory, while the two recovery mechanisms measured
# +5 and +4. It is a legitimate prior and it is NOT a blind rediscovery policy, so a run under it cannot
# be reported as evidence that the search found recovery on its own.
#
# Keeping both, each fixed before its own run, is what preserves that distinction. The policy name is
# recorded in every Selection so no result can be quoted without it.
NEUTRAL = "neutral"
HISTORY_INFORMED = "history_informed"
POLICIES = (NEUTRAL, HISTORY_INFORMED)


def _sort_key(c: Candidate, policy: str) -> tuple:
    """The pre-registered order, as one total function.

    Shared by both policies:
      1. higher residual support first        -- the pipeline's own ranking
      2. fewer projected firings             -- the narrower, more conditional hypothesis
      3. label, lexicographically            -- deterministic tie-break

    HISTORY_INFORMED inserts the operator-family term between 1 and 2. NEUTRAL omits it entirely -- it
    does not appear with a neutral weight, it is simply not part of the comparison.
    """
    # MEASURED SUPPORT OUTRANKS PROJECTED SUPPORT OF THE SAME SIZE. Inserted immediately after the
    # support term so it breaks ties without reordering genuinely larger residuals: a projected 201
    # still sorts above a measured 8, because the projection may be right and refusing to rank it at
    # all would make every first-round arm unselectable. What it must not do is win a tie on a number
    # nobody counted. `0` sorts before `1`, so measured comes first.
    measured_first = 0 if c.support_is_measured else 1
    if policy == HISTORY_INFORMED:
        return (-int(c.support), measured_first, c.family_rank, int(c.fires_on_states), str(c.label))
    return (-int(c.support), measured_first, int(c.fires_on_states), str(c.label))


def select_candidate(candidates: Iterable[Candidate], *,
                     settled: Sequence[str] = (),
                     measured_losers: Sequence[str] = (),
                     policy: str = NEUTRAL) -> Selection:
    """Apply a pre-registered policy. Deterministic: same input and policy, same choice.

    `policy` MUST be one of POLICIES and defaults to NEUTRAL, so a caller that does not think about it
    gets the assumption-free order rather than our corpus history.

    `settled` and `measured_losers` are label sets the caller carries across rounds -- accepted
    controllers, and arms a previous round measured as no-benefit or negative. A round is never re-spent
    on either.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown selection policy {policy!r}; expected one of {POLICIES}")
    settled_set = {str(s) for s in settled}
    loser_set = {str(s) for s in measured_losers}

    eligible: list[Candidate] = []
    excluded: list[Exclusion] = []
    for c in candidates:
        if c.label in settled_set:
            excluded.append(Exclusion(c.label, SETTLED))
            continue
        if c.label in loser_set:
            excluded.append(Exclusion(c.label, MEASURED_LOSER))
            continue
        if not c.feasible:
            excluded.append(Exclusion(c.label, INFEASIBLE_FOR_CONSTRAINT,
                                      "the operator cannot change the constrained quantity"))
            continue
        # THE PILOT'S DEFECT, caught here rather than at install time: a substitute arm carried the
        # relocation executor's capability_id, which the host's identity gate would refuse -- so the arm
        # would have measured the CONTROL while being reported as an intervention.
        if (c.resolved_capability_id and c.capability_id
                and c.resolved_capability_id != c.capability_id):
            excluded.append(Exclusion(
                c.label, CAPABILITY_MISMATCH,
                f"spec names {c.capability_id!r}, core resolved {c.resolved_capability_id!r}"))
            continue
        if int(c.fires_on_states) <= 0:
            excluded.append(Exclusion(c.label, NEVER_FIRES))
            continue
        eligible.append(c)

    ranked = tuple(sorted(eligible, key=lambda c: _sort_key(c, policy)))
    if not ranked:
        return Selection(None, (), tuple(excluded), terminal=POLICY_EXHAUSTED, policy=policy)
    return Selection(ranked[0], ranked, tuple(excluded), policy=policy)
