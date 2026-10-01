"""Expand ONE semantic proposal into the feasible policy family, then let measurement choose.

THE DECOMPOSITION CHANGE
------------------------
Before: the proposer emitted a COMPLETE controller (l, phi, mu, theta) and AnchorOpt checked whether
it was legal. Four optimization decisions in one shot, from a model with no measurement channel.

    R1: post_execution / retrieval_similarity_below_threshold / reprompt / below=0.75
        -> one committed controller. `below=0.75` exceeded the observed maximum (0.6984), so it fired
           on 76/76 retrieves -- an unconditional reprompt, not a threshold -- and measured -3.37 pp.
           REROUTE was never evaluated at all, because the proposer happened to name reprompt.

After: the proposer emits a SEMANTIC LOCALIZATION -- "weak retrieval at post-execution matters" --

    (l, phi, policy_class)

and AnchorOpt expands it into every feasible policy in Pi(l, phi):

    mu in U_H(l), grounded and executable      the DISCRETE part
    theta from the declared domain, on TRAIN   the CONTINUOUS part

Measurement then chooses. The LLM's action preference and threshold hint are recorded as
PROVENANCE -- hypotheses to compare against theta*, never constraints on the search.

WHY THIS IS STILL POLICY-BLOCK WORK
-----------------------------------
`retrieval_similarity_below_threshold` is already in Phi. Nothing here learns a signal, and no claim
about signal learning follows from it. The SIGNAL block is where Phi grows, and the planned
leave-one-signal-out test is what would speak to that.

NO INVENTED DESTINATIONS
------------------------
A REROUTE arm exists only if the runtime can GROUND a real executable destination from its own tool
schema. Where it cannot, the action is reported infeasible with a reason -- never filled in with a
plausible-looking tool name. docs/GENERALIZABILITY.md: a destination must be attested on RESOLVING,
and A2's near-miss (viable at 11/11 "clean", 9 of which returned nothing) is why "it did not error"
is not evidence. An invented destination would be misscored as "the substitute did not help".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.learning.policy_class import (
    ParameterDomain, PolicyClass, PolicySpec, PolicySpecError, ThetaResult, optimize_theta,
    train_objective,
)


@dataclass(frozen=True)
class SemanticProposal:
    """What the LLM proposes now: WHERE and WHAT MATTERS. Not the controller.

    `action_preference` and `theta_hint` are accepted and recorded, because comparing them to what
    measurement chose is how R1's failure became legible -- but neither narrows the search. That is
    the whole point of the change.
    """

    boundary: IncisionPoint
    signal: str
    policy_class: PolicyClass = PolicyClass.PARAMETERIZED
    rationale: str = ""
    diagnosis_case_ids: tuple[str, ...] = ()
    # provenance only, never constraints
    action_preference: str = ""
    theta_hint: Mapping[str, Any] = field(default_factory=dict)
    #: OPTIONAL window within the boundary, when the host distinguishes them. One incision point can
    #: contain several places with different executable actions -- a step that proposed no call, a set
    #: of calls queued for dispatch, and a finished decision about to be answered are all
    #: "post-generation, pre-execution" and only the middle one has a call to cancel. When a host
    #: declares windows, naming one here lets an action that cannot execute in THIS window be pruned
    #: with a reason instead of tying with one that can. Empty means "the boundary as a whole", which
    #: is the behaviour every existing caller already gets.
    site: str = ""


@dataclass(frozen=True)
class PolicyCandidate:
    """One feasible (mu, theta) policy in Pi(l, phi), ready to be measured."""

    boundary: IncisionPoint
    signal: str
    action: Action
    policy_class: PolicyClass
    theta: Mapping[str, Any] = field(default_factory=dict)          # action parameters
    signal_domains: tuple[ParameterDomain, ...] = ()
    grounding: Mapping[str, Any] = field(default_factory=dict)
    was_llm_preference: bool = False

    @property
    def label(self) -> str:
        return f"{self.boundary.value}/{self.signal}/{self.action.value}"


@dataclass(frozen=True)
class InfeasibleAction:
    """An action in the structural grid that this runtime cannot execute here, and why.

    Reported rather than dropped: "reroute was never tried" and "reroute is not executable here" are
    different facts, and a search that silently omits the second is unauditable.
    """

    action: Action
    reason: str


def expand_policy_family(proposal: SemanticProposal, *, runtime, host,
                         theta_for_action: Callable[[Action, SemanticProposal], Mapping[str, Any] | None] | None = None,
                         ) -> tuple[list[PolicyCandidate], list[InfeasibleAction]]:
    """Pi(l, phi): every DISCRETE action this runtime can actually execute at l for this signal.

    Four checks per action, each with its own reason so an omission is attributable:

      structural/host   mu must be in U_H(l)                     -- HostProfile.require
      observable        phi must be observable at l              -- the runtime is asked, not assumed
      grounded          a REROUTE needs a real destination       -- from the runtime's tool schema
      parameterizable   theta for the ACTION must be derivable   -- None prunes the cell

    The returned candidates are NOT ranked. Ranking here would reintroduce the thing this module
    exists to remove: a preference asserted before measurement.
    """
    feasible: list[PolicyCandidate] = []
    infeasible: list[InfeasibleAction] = []

    # phi must be observable at the proposed boundary at all.
    try:
        probe = dict(runtime.probe_params(proposal.signal)) if hasattr(runtime, "probe_params") else {}
    except Exception:
        probe = {}
    try:
        runtime.evaluate_signal(proposal.signal, {"boundary": proposal.boundary}, probe)
    except KeyError as exc:
        return [], [InfeasibleAction(Action.NOOP, f"signal_not_observable_at_boundary: {exc}")]
    except (ValueError, TypeError):
        pass          # parameter-shaped complaint; the domain layer handles it

    domains = tuple(runtime.parameter_domains(proposal.signal)) \
        if hasattr(runtime, "parameter_domains") else ()
    declared_class = (runtime.policy_class_for(proposal.signal)
                      if hasattr(runtime, "policy_class_for") else proposal.policy_class)

    for action in sorted(Action, key=lambda a: a.value):
        if action is Action.NOOP:
            continue                                  # the control arm, never a proposal
        try:
            host.require(proposal.boundary, action)
        except Exception as exc:
            infeasible.append(InfeasibleAction(action, f"not_executable_in_host: {exc}"))
            continue
        # WINDOW NARROWING, asked of the runtime rather than assumed by core.
        #
        # `HostProfile` is keyed by incision point, and one point can contain several windows with
        # different executable actions. Core does not know what a window IS -- it asks the runtime
        # whether this action survives at the named one, and prunes with a reason if not. A runtime
        # that declares no windows, or does not accept the argument at all, is unaffected: the
        # fallback answers the boundary's own set, so this cannot narrow an existing search.
        #
        # Why it matters: without it, an action that cannot execute in this window is offered as an
        # equal candidate, ties exactly with one that can, and the winner falls out of enum order.
        # A tie between a real candidate and an inexecutable one is not a tie.
        if proposal.site and hasattr(runtime, "feasible_actions"):
            try:
                allowed = runtime.feasible_actions(proposal.boundary, proposal.site)
            except TypeError:
                allowed = None          # runtime predates windows; boundary-level check stands
            if allowed is not None and action not in allowed:
                infeasible.append(InfeasibleAction(
                    action,
                    f"not_executable_in_window: the host declares {action.value} executable at "
                    f"{proposal.boundary.value} but not in its {proposal.site!r} window"))
                continue

        grounding: dict[str, Any] = {}
        if action is Action.REROUTE:
            # NO INVENTED DESTINATIONS. Ask the runtime to derive one from its own capabilities.
            derived = None
            if hasattr(runtime, "derive_reroute_destination"):
                derived = runtime.derive_reroute_destination(proposal.signal, proposal.boundary)
            if not derived:
                infeasible.append(InfeasibleAction(
                    action, "no_grounded_destination: the runtime derives no executable destination "
                            "for this signal from its tool schema, and a destination may not be "
                            "invented -- it must be attestable on RESOLVING"))
                continue
            grounding = dict(derived)

        theta = ({} if theta_for_action is None
                 else theta_for_action(action, proposal))
        if theta is None:
            infeasible.append(InfeasibleAction(
                action, "no_action_theta: nothing available to parameterize this action"))
            continue
        if action is Action.REROUTE and grounding.get("destination"):
            theta = {**dict(theta), "destination": grounding["destination"],
                     **({"retry_original": grounding["retry_original"]}
                        if "retry_original" in grounding else {})}

        feasible.append(PolicyCandidate(
            boundary=proposal.boundary, signal=proposal.signal, action=action,
            policy_class=declared_class, theta=theta, signal_domains=domains,
            grounding=grounding,
            was_llm_preference=(proposal.action_preference or "").strip().lower() == action.value))
    return feasible, infeasible


@dataclass(frozen=True)
class PolicySearchResult:
    """The measured outcome of searching Pi(l, phi). The whole table, not just the winner."""

    proposal: SemanticProposal
    per_action: Mapping[str, tuple[Mapping[str, Any], tuple[ThetaResult, ...]]]
    infeasible: tuple[InfeasibleAction, ...]
    winner_action: str = ""
    winner_theta: Mapping[str, Any] = field(default_factory=dict)
    winner_result: ThetaResult | None = None

    @property
    def llm_preference_won(self) -> bool | None:
        """Did the LLM's named action turn out to be the measured best? None if it named none.

        Reported because it is the honest measure of how much the one-shot preference was worth --
        across rounds this is the statistic that says whether the decomposition is earning anything.
        """
        pref = (self.proposal.action_preference or "").strip().lower()
        if not pref:
            return None
        return pref == self.winner_action


def search_policy_family(proposal: SemanticProposal, *, runtime, host,
                         evaluate: Callable[[PolicyCandidate, Mapping[str, Any]], ThetaResult],
                         observations: Mapping[str, Sequence[float]] | None = None,
                         theta_for_action=None, n_grid: int = 9,
                         objective=train_objective) -> PolicySearchResult:
    """Expand, optimize theta per feasible action on TRAIN, and let measurement choose.

    `evaluate(candidate, theta) -> ThetaResult` runs one paired arm against the SAME incumbent, so
    every arm across every action and theta is comparable.

    Returns the full per-action sweep. Reporting only the winner would hide the shape, and the shape
    is what distinguishes "this family works at the right theta" from "this family cannot work" --
    the exact distinction R1 could not make.
    """
    feasible, infeasible = expand_policy_family(proposal, runtime=runtime, host=host,
                                                theta_for_action=theta_for_action)
    per_action: dict[str, tuple[Mapping[str, Any], tuple[ThetaResult, ...]]] = {}
    best_overall: tuple[str, Mapping[str, Any], ThetaResult] | None = None

    for cand in feasible:
        spec = PolicySpec(
            boundary=cand.boundary, signal=cand.signal, action=cand.action,
            policy_class=cand.policy_class, domains=cand.signal_domains, theta=cand.theta,
            llm_hint=dict(proposal.theta_hint), rationale=proposal.rationale)
        try:
            theta_star, results = optimize_theta(
                spec, evaluate=lambda th, c=cand: evaluate(c, th),
                observations=observations, n_grid=n_grid, objective=objective)
        except PolicySpecError as exc:
            infeasible.append(InfeasibleAction(cand.action, f"theta_search_failed: {exc}"))
            continue
        per_action[cand.action.value] = (theta_star, tuple(results))
        best = max(results, key=objective)
        if best_overall is None or objective(best) > objective(best_overall[2]):
            best_overall = (cand.action.value, theta_star, best)

    return PolicySearchResult(
        proposal=proposal, per_action=per_action, infeasible=tuple(infeasible),
        winner_action=best_overall[0] if best_overall else "",
        winner_theta=dict(best_overall[1]) if best_overall else {},
        winner_result=best_overall[2] if best_overall else None)


def summarize_policy_search(result: PolicySearchResult) -> str:
    """The full family table: every feasible action x theta, plus what was infeasible and why."""
    p = result.proposal
    lines = [f"Pi({p.boundary.value}, {p.signal})   policy_class={p.policy_class.value}",
             f"  LLM said: action_preference={p.action_preference or '(none)'}  "
             f"theta_hint={dict(p.theta_hint) or '(none)'}   [PROVENANCE ONLY]"]
    for act, (theta_star, results) in sorted(result.per_action.items()):
        lines.append(f"  -- mu={act} --   theta* = {dict(theta_star)}")
        for r in sorted(results, key=lambda x: str(x.theta)):
            tv = ", ".join(f"{k}={v}" for k, v in r.theta.items()) or "(none)"
            star = " *" if r.theta == theta_star else "  "
            lines.append(f"     {tv:>16}{star} net {r.net:>+3d}  +{len(r.gains):<3d} "
                         f"-{len(r.losses):<3d} fire {100*r.firing_rate:5.1f}%  "
                         f"dAcc {r.accuracy_delta_pp:+.2f} pp")
    for inf in result.infeasible:
        lines.append(f"  -- mu={inf.action.value} INFEASIBLE: {inf.reason[:96]}")
    if result.winner_result is not None:
        lines.append(f"  WINNER ON MEASUREMENT: mu={result.winner_action} "
                     f"theta*={dict(result.winner_theta)} net {result.winner_result.net:+d}")
        pref = result.llm_preference_won
        if pref is not None:
            lines.append(f"  LLM's action preference was {'CORRECT' if pref else 'WRONG'} "
                         f"-- recorded, and it constrained nothing")
    return "\n".join(lines)
