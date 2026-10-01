"""The OUTER loop: prioritized residual first, policy search strictly inside it.

    rank residuals -> freeze R1 -> propose+optimize for R1 ONLY -> measure
                   -> promote, or prove POLICY_EXHAUSTED_CURRENT_SPACE -> R2

WHAT THIS REPLACES
------------------
`block_loop.run_blocks` asked the proposer for candidates over the WHOLE residual, validated them,
and screened the survivors globally. The residual ranking was computed and then ignored, so a
candidate's survival depended on being instantiable rather than on addressing the top problem. The R1
audit measured the consequence: the top residual's only full-coverage candidate was infeasible, and
the budget moved to a candidate covering 1 of its 4 cases.

THE THREE INVARIANTS, each enforced rather than intended
--------------------------------------------------------
1. ONE RESIDUAL AT A TIME. The proposer sees only `problem.scoped_diagnoses()`, and every arm handed
   to `AnchorPolicyOpt` comes from that one problem. Candidates for R2 cannot compete with R1's --
   they are not in the same call.
2. AN INFEASIBLE ACTION DOES NOT DEMOTE THE RESIDUAL. R1 stays frozen while the loop tries the other
   action families at that locus, then any further locus the proposer offers for the SAME problem.
   Moving on requires exhaustion, not a single failure.
3. EXHAUSTION IS PROVEN AND SCOPED. `POLICY_EXHAUSTED_CURRENT_SPACE` names what was tried and is
   explicitly relative to the current Phi, host, proposer-offered loci and action contracts. A
   residual no signal can express is preserved as SIGNAL-block backlog instead -- a representation
   limit is not a solved problem.

Controller ranking still exists, and is confined: `top_k` bounds the arms measured WITHIN the frozen
residual. It can never advance a lower-priority residual.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anchoropt.learning.anchor_policy_opt import (
    AnchorPolicyOpt, FrozenIncumbent, PolicyOptResult, SearchSpaceProposal,
)
from anchoropt.learning.residual_problem import (
    POLICY_ACCEPTED, POLICY_EXHAUSTED_CURRENT_SPACE, SIGNAL_BLOCKED_RESIDUAL, ResidualProblem,
    ResidualSearchLedger, build_residual_problems,
)
from anchoropt.runtime import ResidualDiagnosis


@dataclass
class ResidualRoundResult:
    """One outer round: which residuals were searched, in what order, and what each concluded."""

    problems: tuple[ResidualProblem, ...]
    ledgers: list[ResidualSearchLedger] = field(default_factory=list)
    accepted_problem: ResidualProblem | None = None
    accepted_arm: Any = None
    accepted_theta: Mapping[str, Any] = field(default_factory=dict)
    accepted_result: Any = None
    outcome: str = ""

    @property
    def searched_keys(self) -> tuple[str, ...]:
        return tuple(l.problem.key for l in self.ledgers)

    def ledger_for(self, key: str) -> ResidualSearchLedger | None:
        for l in self.ledgers:
            if l.problem.key == key:
                return l
        return None


def run_residual_round(diagnoses: Sequence[ResidualDiagnosis], *,
                       runtime, host,
                       propose_for: Callable[[ResidualProblem], Sequence[SearchSpaceProposal]],
                       evaluate: Callable[..., Any],
                       accept: Callable[[Any], tuple[bool, str]],
                       incumbent: FrozenIncumbent,
                       expressible: Callable[[ResidualDiagnosis], bool] | None = None,
                       engaged: Mapping[str, int] | None = None,
                       observations: Mapping[str, Sequence[float]] | None = None,
                       top_k: int = 3,
                       max_problems: int | None = None) -> ResidualRoundResult:
    """Search residual problems in RANK ORDER, one frozen at a time.

        propose_for(problem)   -> Sequence[SearchSpaceProposal]   sees ONLY that problem's diagnoses
        evaluate(arm, theta)   -> ThetaResult                     one paired arm vs `incumbent`
        accept(ThetaResult)    -> (bool, reason)                  AnchorOpt's acceptance rule

    Returns on the FIRST accepted policy: one anchor is installed per round, and the outer residual
    boosting loop then re-runs the new incumbent and re-mines. Continuing to a lower-priority
    residual after an acceptance would measure the next anchor against a stale incumbent.
    """
    problems = build_residual_problems(diagnoses, expressible=expressible, engaged=engaged)
    result = ResidualRoundResult(problems=problems)
    opt = AnchorPolicyOpt(runtime=runtime, host=host)

    for problem in problems[:max_problems] if max_problems else problems:
        ledger = ResidualSearchLedger(problem=problem)
        result.ledgers.append(ledger)

        # A residual the vocabulary cannot express has no proposable policy. Preserve it for the
        # SIGNAL block rather than calling it exhausted -- different limit, different next step.
        if not problem.expressible:
            ledger.state = SIGNAL_BLOCKED_RESIDUAL
            continue

        # SCOPED: the proposer sees only this problem's diagnoses. This is invariant 1.
        proposals = tuple(propose_for(problem))
        if not proposals:
            ledger.state = POLICY_EXHAUSTED_CURRENT_SPACE
            continue

        accepted_here = None
        for proposal in proposals:
            # Guard invariant 1 mechanically: a proposal must cite only this problem's cases.
            alien = set(proposal.diagnosis_case_ids) - set(problem.case_ids)
            if alien:
                raise ValueError(
                    f"proposal for residual R{problem.rank} cites cases from another residual "
                    f"({sorted(alien)[:3]}...). Candidates for a lower-priority residual must not "
                    f"enter this optimization -- that is the defect this loop exists to prevent.")
            ledger.note_locus(proposal.boundary.value, proposal.signal)

            opt_result = opt.optimize(proposal, incumbent=incumbent, evaluate=evaluate,
                                      observations=observations)

            # Record every infeasible cell with its named missing requirement. Invariant 2: these do
            # NOT end the residual -- the loop continues to the next action family, then the next
            # locus this problem offers.
            for rj in opt_result.rejected:
                op = f"[{rj.operator.value}]" if rj.operator else ""
                ledger.note_infeasible(f"{proposal.boundary.value}/{proposal.signal}",
                                       f"{rj.action.value}{op}",
                                       f"{rj.reason_code}"
                                       + (f" missing={list(rj.missing)}" if rj.missing else ""))

            # Controller ranking is confined HERE, inside the frozen residual: top_k bounds how many
            # of this problem's arms are measured. It cannot promote another residual.
            ranked = sorted(opt_result.per_arm.items(),
                            key=lambda kv: -max(r.net for r in kv[1][1]))[:max(1, int(top_k))]
            for label, (theta_star, results) in ranked:
                best = max(results, key=lambda r: r.net)
                ledger.note_measured(label, best.net)
                ok, _why = accept(best)
                if ok and accepted_here is None:
                    accepted_here = (opt_result, theta_star, best)

            if accepted_here is not None:
                break                     # this residual is solved; stop proposing for it

        if accepted_here is not None:
            opt_result, theta_star, best = accepted_here
            ledger.state = POLICY_ACCEPTED
            ledger.accepted = opt_result.winner
            result.accepted_problem = problem
            result.accepted_arm = opt_result.winner
            result.accepted_theta = dict(theta_star)
            result.accepted_result = best
            result.outcome = POLICY_ACCEPTED
            return result                 # one anchor per round; the outer loop re-mines next

        # Nothing accepted for this problem. Exhaustion is PROVEN and SCOPED (invariant 3).
        ledger.state = POLICY_EXHAUSTED_CURRENT_SPACE

    result.outcome = (POLICY_EXHAUSTED_CURRENT_SPACE if result.ledgers
                      else "NO_RESIDUAL")
    return result


def summarize_round(result: ResidualRoundResult) -> str:
    lines = ["RANKED RESIDUAL PROBLEMS (metric unchanged: support, then key)"]
    for p in result.problems:
        mark = " <- FROZEN FIRST" if p.rank == 1 else ""
        lines.append(f"  {p}{mark}")
    lines.append("")
    lines.append("SEARCH, in rank order:")
    for l in result.ledgers:
        lines.append(l.summary())
        if l.state in (POLICY_EXHAUSTED_CURRENT_SPACE, SIGNAL_BLOCKED_RESIDUAL):
            lines.append(f"    -> {l.exhausted_reason()[:150]}")
    lines.append("")
    if result.accepted_problem is not None:
        lines.append(f"ACCEPTED for R{result.accepted_problem.rank}: "
                     f"{getattr(result.accepted_arm, 'label', '?')} "
                     f"theta*={dict(result.accepted_theta)} "
                     f"net {result.accepted_result.net:+d}")
        lines.append("  round ends here -- one anchor per round; the outer loop now re-runs the new "
                     "incumbent and re-mines its residual")
    else:
        lines.append(f"outcome: {result.outcome}")
    return "\n".join(lines)
