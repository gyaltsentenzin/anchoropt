"""AnchorPolicyOpt: the component that OPTIMIZES a policy. The proposer only narrows the space.

    Attributor  ->  Proposer  ->  AnchorPolicyOpt  ->  Executor  ->  Promote / Re-mine

THE DIVISION OF LABOUR
----------------------
    Attributor       what failed, where, on what evidence, which decision must change
    Proposer         (l, phi, U-hat) -- a locus, a condition, and a SMALL HIGH-RECALL set of
                     plausible actions. It reduces the space semantically; it does not pick a winner.
    AnchorPolicyOpt  builds every runtime-feasible (mu, mechanism, theta) arm in that space,
                     measures them against ONE frozen incumbent, and returns the argmax of J_train.
    Executor         runs the selected policy.

WHY THIS EXISTS: R2 WAS INCOMPLETE OPTIMIZATION
-----------------------------------------------
R2 swept theta for `retrieval_similarity_below_threshold` and found every threshold net-negative
(-1 to -5, monotone in firing rate). That correctly rejected ONE POLICY -- reprompt-at-this-signal --
and it was reported as if it bore on the family. It did not, because theta was searched CONDITIONAL
ON the proposer's action: the sweep asked "which threshold makes reprompt work" when the open question
was "which EXECUTION should respond to weak retrieval". A grounded observation-augmenting arm was
feasible at that locus and was never built, purely because the proposer named reprompt first.

So the optimization is over (mu, mechanism, theta) JOINTLY. Rejecting a policy is not rejecting a
signal, and this component is what makes the difference measurable rather than rhetorical.

INNER LOOP vs OUTER LOOP -- the distinction, in code
----------------------------------------------------
    INNER (this module)   one POLICY round. (l, phi) fixed; counterfactual (mu, mechanism, theta)
                          arms measured against the SAME frozen incumbent P_k. Nothing is promoted
                          here, and the incumbent never moves mid-search -- if it did, arms would
                          not be comparable and the argmax would be meaningless.
    OUTER (block_loop)    residual boosting. P_k -> P_{k+1} on acceptance, then RE-RUN the new
                          incumbent and RE-MINE its residual. Each anchor is fitted to the residual
                          its predecessors left, which is why a learned anchor should not need its
                          original predecessors to remain valid.

`FrozenIncumbent` below exists to make the inner-loop invariant checkable rather than assumed.

NOT A THRESHOLD SWEEP
---------------------
The optimizer consumes `PolicySpec` objects whose class is DETERMINISTIC or PARAMETERIZED, and calls
`candidate_thetas()`. A deterministic policy yields exactly one arm ({}), a parameterized one yields
its declared grid. CLASSIFIER and LLM_POLICY are declared-and-refused in `policy_class`, so adding
them is a new `candidate_thetas` implementation and nothing here changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.learning.action_contract import (
    CONTRACTS, ContractFailure, InstantiatedAction, Operator, instantiate, operators_of, validate,
)
from anchoropt.learning import executor_capability as exec_cap
from anchoropt.learning.policy_class import (
    ParameterDomain, PolicyClass, PolicySpec, PolicySpecError, ThetaResult, train_objective,
)


# ================================================================================================
# what the proposer now emits
# ================================================================================================

@dataclass(frozen=True)
class SearchSpaceProposal:
    """(l, phi, U-hat): a locus, a condition, and a small HIGH-RECALL action set.

    High-recall on purpose: the proposer's job is to exclude what is semantically irrelevant, not to
    pick the winner. Naming two or three plausible actions costs a few paired arms; naming one and
    being wrong costs the round, which is what R1 did.

    `preferred_action`, `theta_hint` and `rationale` are PROVENANCE. They are recorded, reported, and
    compared against what measurement chose -- and they constrain nothing. A test asserts that
    changing them does not change the arms built.
    """

    boundary: IncisionPoint
    signal: str
    action_set: tuple[Action, ...]
    rationale: str = ""
    diagnosis_case_ids: tuple[str, ...] = ()
    # ---- provenance only, never constraints ----
    preferred_action: str = ""
    theta_hint: Mapping[str, Any] = field(default_factory=dict)
    preferred_mechanism: str = ""

    def __post_init__(self) -> None:
        if not self.action_set:
            raise ValueError(
                "U-hat is empty: a proposal must name at least one plausible action, or it has "
                "localized nothing to optimize")
        if Action.NOOP in self.action_set:
            raise ValueError("NOOP is the control arm, never a proposed action")


@dataclass(frozen=True)
class FrozenIncumbent:
    """The ONE incumbent every arm in a round is measured against. Frozen for the round's duration.

    The inner-loop invariant: if the incumbent moved between arms, the arms would not be comparable
    and argmax J_train would be meaningless. `token` is whatever the caller uses to identify the
    incumbent build; `assert_same` is called before every arm so a drifted incumbent fails loudly
    instead of silently invalidating a sweep.
    """

    incumbent_id: str
    token: str = ""

    def assert_same(self, other_token: str) -> None:
        if self.token and other_token and self.token != other_token:
            raise RuntimeError(
                f"incumbent drifted mid-round: round was frozen at {self.token!r} but an arm ran "
                f"against {other_token!r}. Every arm in a POLICY round must share one incumbent, or "
                f"the comparison across arms is invalid.")


# ================================================================================================
# arms
# ================================================================================================

@dataclass(frozen=True)
class PolicyArm:
    """One fully instantiated (mu, eta_mu) policy, ready to be crossed with the signal grid.

    `instantiated` carries the operator and its grounded action parameters eta_mu. theta_phi is NOT
    stored here: it varies per evaluation and is supplied by the optimizer, which is exactly the
    separation that was missing when one `theta` dict held both.
    """

    boundary: IncisionPoint
    signal: str
    instantiated: InstantiatedAction
    policy_class: PolicyClass
    signal_domains: tuple[ParameterDomain, ...] = ()
    was_preferred: bool = False
    # WHICH executor core judged this arm feasible against, when the cell has more than one. Carried
    # so the runner installs THE capability the feasibility check accepted: proving "some executor
    # here works" and then executing a different one would measure a mechanism core never validated.
    capability_id: str = ""

    @property
    def action(self) -> Action:
        return self.instantiated.action

    @property
    def eta(self) -> Mapping[str, Any]:
        return self.instantiated.eta

    @property
    def label(self) -> str:
        return f"{self.boundary.value}/{self.signal}/{self.instantiated.label}"


@dataclass(frozen=True)
class RejectedArm:
    """A family that could not be instantiated, and why. RECORDED, never silently dropped.

    "reroute was not tried" and "reroute cannot be grounded here" are different facts, and a search
    that omits the second is unauditable. `missing` names the ungrounded requirement, so a rejection
    says what was absent rather than merely that something was.
    """

    action: Action
    operator: Operator | None
    reason_code: str
    missing: tuple[str, ...] = ()
    detail: str = ""


REJECT_NOT_IN_HOST = "not_executable_in_host"
REJECT_NOT_GROUNDED = "not_exactly_groundable"
REJECT_NO_ACTION_THETA = "no_action_parameters"
REJECT_SIGNAL_UNOBSERVABLE = "signal_not_observable_at_boundary"
REJECT_NO_THETA_GRID = "no_parameter_grid"
# An action can be ADMISSIBLE in U_H(l) while the host has no executor able to realize it.
# Reporting such a candidate "grounded" is what let three abstractly-legal REPROMPT arms
# reach materialization before one was found unrunnable.
REJECT_NO_EXECUTOR = "not_materializable_by_host_executor"


# ================================================================================================
# AnchorPolicyOpt
# ================================================================================================

@dataclass
class AnchorPolicyOpt:
    """Builds, measures, and selects among counterfactual policy arms. Measurement decides.

    `runtime` supplies feasibility, mechanisms, grounding and parameter domains. `host` supplies
    U_H(l). Neither the proposer nor any LLM is consulted after the proposal arrives -- which is the
    property that makes this an optimizer rather than a second opinion.
    """

    runtime: Any
    host: Any
    objective: Callable[[ThetaResult], tuple] = train_objective
    n_grid: int = 9

    # ---------------------------------------------------------------- build
    def build_arms(self, proposal: SearchSpaceProposal) -> tuple[list[PolicyArm], list[RejectedArm]]:
        """Instantiate every action family in U-hat into its grounded (mu, eta_mu) arms.

        THE OPTIMIZER'S BRIEF, stated so it cannot drift: this does not choose an action by semantic
        plausibility -- the proposer already narrowed the space. It takes each suggested family,
        reads its REQUIRED action parameters from the contract, grounds or enumerates them from the
        runtime's declared capabilities, and builds one arm per grounding. It never invents a tool,
        an argument or a state surface, and it never downgrades an ungroundable family into another.

        A family with several groundings yields several arms -- two destinations, two retry
        semantics, three reprompt instructions are six arms, not one guess.
        """
        arms: list[PolicyArm] = []
        rejected: list[RejectedArm] = []

        probe = {}
        if hasattr(self.runtime, "probe_params"):
            try:
                probe = dict(self.runtime.probe_params(proposal.signal) or {})
            except Exception:
                probe = {}
        try:
            self.runtime.evaluate_signal(proposal.signal, {"boundary": proposal.boundary}, probe)
        except KeyError as exc:
            return [], [RejectedArm(Action.NOOP, None, REJECT_SIGNAL_UNOBSERVABLE, (), str(exc))]
        except (ValueError, TypeError):
            pass

        domains = tuple(self.runtime.parameter_domains(proposal.signal)) \
            if hasattr(self.runtime, "parameter_domains") else ()
        pclass = (self.runtime.policy_class_for(proposal.signal)
                  if hasattr(self.runtime, "policy_class_for")
                  else (PolicyClass.PARAMETERIZED if domains else PolicyClass.DETERMINISTIC))
        pref = (proposal.preferred_action or "").strip().lower()

        for action in sorted(set(proposal.action_set), key=lambda a: a.value):
            try:
                self.host.require(proposal.boundary, action)
            except Exception as exc:
                rejected.append(RejectedArm(action, None, REJECT_NOT_IN_HOST, (), str(exc)))
                continue
            # REROUTE has TWO operator readings (substitute / transform) with different contracts.
            for operator in operators_of(action):
                built, failure = instantiate(operator, signal=proposal.signal,
                                             boundary=proposal.boundary, runtime=self.runtime)
                if failure is not None:
                    rejected.append(RejectedArm(action, operator, failure.code,
                                                tuple(failure.missing), failure.detail))
                    continue
                for inst in built:
                    ok, why = validate(inst)          # HARD validation, independent of any prompt
                    if not ok:
                        rejected.append(RejectedArm(action, operator, why.split(":")[0], (), why))
                        continue
                    # EXECUTOR-BACKED FEASIBILITY. Abstract legality is not materializability: the
                    # host must have an executor for (l, mu) that covers this signal and can consume
                    # this eta unchanged. A parameter the executor would have to coerce is REJECTED
                    # rather than silently adjusted -- coercion runs different semantics than the
                    # ones proposed, under the proposal's name.
                    can, detail, cid = self._materializable(proposal, action, operator, inst)
                    if not can:
                        rejected.append(RejectedArm(action, operator, REJECT_NO_EXECUTOR,
                                                    (), detail))
                        continue
                    arms.append(PolicyArm(
                        boundary=proposal.boundary, signal=proposal.signal, instantiated=inst,
                        policy_class=pclass, signal_domains=domains,
                        was_preferred=(pref == action.value), capability_id=cid))
        return arms, rejected

    # ------------------------------------------------------- executor-backed materializability
    def _materializable(self, proposal: SearchSpaceProposal, action: Action, operator: Operator,
                        inst: InstantiatedAction) -> tuple[bool, str, str]:
        """Is there EXECUTING CODE at this boundary that runs `action` and reads this eta?

        Two adapter shapes are supported, and the difference is the point of this method:

        * `executor_capability(boundary, action) -> ExecutorCapability` -- the checked form. Core
          validates the binding (no ghosts), the disabled flag, signal coverage, and that every
          CONTRACT-REQUIRED eta key is one the executor actually consumes (no inert eta).
        * `executor_supports(boundary, action, signal, eta) -> (bool, str)` -- the legacy form, kept
          so existing adapters keep working. Core cannot see a binding through it, so it can only
          pass the adapter's own verdict through. Adapters are migrated to the form above.

        An adapter with NEITHER is unconstrained here: admissibility already bounded it, and refusing
        every action for a host that has not yet declared its executors would make the first port
        impossible. `check_adapter.py` reports the absence.
        """
        expanded = frozenset(getattr(self.runtime, "expanded_signal_names", lambda: ())() or ())
        contract = CONTRACTS.get(operator)
        required = tuple(contract.required) if contract else ()
        enforced = tuple(getattr(contract, "enforced", ()) or ()) if contract else ()

        # PLURAL FORM FIRST. A cell is not one mechanism: on this project's own host
        # post_generation_pre_exec/reprompt names both an inject-and-regenerate executor and an
        # inert telemetry-only one. Under a single-capability lookup the inert one MASKED the
        # working one, and the mechanism was unreachable for a bookkeeping reason rather than a
        # validity one. `resolve` returns WHICH capability it accepted, and that id travels with
        # the arm -- accepting the cell and then executing an unnamed sibling is the arm-identity
        # defect from a new direction.
        if hasattr(self.runtime, "executor_capabilities"):
            caps = self.runtime.executor_capabilities(proposal.boundary, action)
            cap, why = exec_cap.resolve(
                caps, boundary=proposal.boundary, action=action, signal=proposal.signal,
                eta=inst.eta, expanded_signals=expanded, required_eta=required,
                enforced_eta=enforced,
                # THE OPERATOR READING, so a cell holding one executor per reading resolves the right
                # one. Without it both readings of REROUTE collapse to whichever capability is declared
                # first, and the arm carries an identity its operator does not implement.
                operator=operator.value if hasattr(operator, "value") else operator)
            return (cap is not None), why, (cap.cid if cap is not None else "")
        if hasattr(self.runtime, "executor_capability"):
            cap = self.runtime.executor_capability(proposal.boundary, action)
            ok, why = exec_cap.supports(
                cap, boundary=proposal.boundary, action=action, signal=proposal.signal,
                eta=inst.eta, expanded_signals=expanded, required_eta=required,
                enforced_eta=enforced)
            return ok, why, (cap.cid if (ok and cap is not None) else "")
        if hasattr(self.runtime, "executor_supports"):
            ok, why = self.runtime.executor_supports(
                proposal.boundary, action, proposal.signal, inst.eta)
            return ok, why, ""
        return True, "host declares no executor registry; materializability unchecked", ""

    # ---------------------------------------------------------------- optimize
    def optimize(self, proposal: SearchSpaceProposal, *,
                 incumbent: FrozenIncumbent,
                 evaluate: Callable[[PolicyArm, Mapping[str, Any]], ThetaResult],
                 observations: Mapping[str, Sequence[float]] | None = None,
                 ) -> "PolicyOptResult":
        """(mu*, theta*) = argmax J_train over every feasible arm x its parameter grid.

        `evaluate(arm, theta) -> ThetaResult` runs ONE paired arm against `incumbent`. Every arm in
        the round shares that incumbent, which is what makes the argmax across actions meaningful --
        this is the property R2 had within one action and lacked across actions.
        """
        arms, rejected = self.build_arms(proposal)
        per_arm: dict[str, tuple[Mapping[str, Any], tuple[ThetaResult, ...]]] = {}
        best: tuple[PolicyArm, Mapping[str, Any], ThetaResult] | None = None
        unevaluated: list[PolicyArm] = []

        for arm in arms:
            spec = PolicySpec(
                boundary=arm.boundary, signal=arm.signal, action=arm.action,
                policy_class=arm.policy_class, domains=arm.signal_domains,
                theta=arm.eta, llm_hint=dict(proposal.theta_hint),
                rationale=proposal.rationale)
            try:
                grid = spec.candidate_thetas(observations, n=self.n_grid)
            except PolicySpecError as exc:
                rejected.append(RejectedArm(arm.action, arm.instantiated.operator,
                                            REJECT_NO_THETA_GRID, (), str(exc)))
                continue
            # AN UNMEASURED THETA HAS NO RESULT, and `None` is how an evaluator says so. It must be
            # DROPPED, not scored: `self.objective` would raise on it (it did -- an honest external
            # evaluator crashed the optimizer, which is why the only usable callback was a stub that
            # always returned a number), and coercing it to zero would rank an arm nobody ran against
            # arms that were measured. A zero is a measurement; None is the absence of one.
            results = []
            for theta in grid:
                r = evaluate(arm, theta)
                if r is None:
                    continue
                incumbent.assert_same(getattr(r, "incumbent_token", "") or incumbent.token)
                results.append(r)
            if not results:
                # Every theta for this arm went unmeasured. The arm stays in `arms` -- it WAS built,
                # which is a structural fact -- but it contributes no per_arm row and cannot win.
                # Recording it as a rejection would be wrong too: nothing refused it.
                unevaluated.append(arm)
                continue
            theta_star = dict(max(results, key=self.objective).theta)
            per_arm[arm.label] = (theta_star, tuple(results))
            local_best = max(results, key=self.objective)
            if best is None or self.objective(local_best) > self.objective(best[2]):
                best = (arm, theta_star, local_best)

        return PolicyOptResult(
            proposal=proposal, incumbent=incumbent, arms=tuple(arms), rejected=tuple(rejected),
            per_arm=per_arm, unevaluated=tuple(unevaluated),
            winner=best[0] if best else None,
            winner_theta=dict(best[1]) if best else {},
            winner_result=best[2] if best else None)


@dataclass(frozen=True)
class PolicyOptResult:
    """The whole inner-loop table: every arm x theta, every rejection, and the measured winner."""

    proposal: SearchSpaceProposal
    incumbent: FrozenIncumbent
    arms: tuple[PolicyArm, ...]
    rejected: tuple[RejectedArm, ...]
    per_arm: Mapping[str, tuple[Mapping[str, Any], tuple[ThetaResult, ...]]]
    # ARMS THAT WERE BUILT AND NEVER MEASURED. Distinct from `rejected` (something refused them) and
    # from a per_arm row with a poor score (something measured them). Without this field the three
    # collapse, and "we did not run it" becomes indistinguishable from "it did not work".
    unevaluated: tuple[PolicyArm, ...] = ()
    winner: PolicyArm | None = None
    winner_theta: Mapping[str, Any] = field(default_factory=dict)
    winner_result: ThetaResult | None = None

    @property
    def n_arms_measured(self) -> int:
        return sum(len(rs) for _t, rs in self.per_arm.values())

    @property
    def preference_was_right(self) -> bool | None:
        """Did the proposer's preferred action win on measurement? None if it named none.

        The statistic worth tracking across rounds: it says what the one-shot preference is worth,
        and therefore whether this decomposition earns its extra arms.
        """
        pref = (self.proposal.preferred_action or "").strip().lower()
        if not pref or self.winner is None:
            return None
        return pref == self.winner.action.value

    @property
    def hint_was_right(self) -> bool | None:
        hint = dict(self.proposal.theta_hint)
        if not hint or self.winner_result is None:
            return None
        return all(self.winner_theta.get(k) == v for k, v in hint.items())


def summarize(result: PolicyOptResult) -> str:
    p = result.proposal
    lines = [
        f"AnchorPolicyOpt  l={p.boundary.value}  phi={p.signal}",
        f"  U-hat (proposer)      : {[a.value for a in p.action_set]}",
        f"  incumbent (frozen)    : {result.incumbent.incumbent_id}",
        f"  PROVENANCE ONLY       : preferred={p.preferred_action or '(none)'}  "
        f"theta_hint={dict(p.theta_hint) or '(none)'}",
        f"  arms built            : {len(result.arms)}   paired evaluations: {result.n_arms_measured}",
    ]
    for label, (theta_star, results) in sorted(result.per_arm.items()):
        lines.append(f"  -- {label}   theta* = {dict(theta_star)}")
        for r in sorted(results, key=lambda x: str(x.theta)):
            tv = ", ".join(f"{k}={v}" for k, v in r.theta.items()) or "(deterministic)"
            star = " *" if dict(r.theta) == dict(theta_star) else "  "
            lines.append(f"     {tv:>18}{star} net {r.net:>+3d}  +{len(r.gains):<3d} "
                         f"-{len(r.losses):<3d}  fire {100*r.firing_rate:5.1f}%  "
                         f"dAcc {r.accuracy_delta_pp:+.2f} pp")
    if result.rejected:
        lines.append("  REJECTED ARMS (recorded, not dropped):")
        for rj in result.rejected:
            op = f"[{rj.operator.value}]" if rj.operator else ""
            miss = f" missing={list(rj.missing)}" if rj.missing else ""
            lines.append(f"     {rj.action.value}{op}: {rj.reason_code}{miss} -- {rj.detail[:70]}")
    if result.winner is not None:
        lines.append(f"  WINNER ON MEASUREMENT : {result.winner.label} "
                     f"theta*={dict(result.winner_theta)} net {result.winner_result.net:+d}")
        for name, got in (("action preference", result.preference_was_right),
                          ("theta hint", result.hint_was_right)):
            if got is not None:
                lines.append(f"     proposer's {name} was {'RIGHT' if got else 'WRONG'} "
                             f"-- recorded; it constrained nothing")
    elif result.arms:
        lines.append(f"  NO WINNER: {len(result.arms)} arm(s) were BUILT and "
                     f"{len(result.unevaluated)} went UNMEASURED. This is not evidence about them -- "
                     f"an unevaluated arm has no result, and reporting one would assert a "
                     f"measurement that never happened.")
    else:
        lines.append("  NO ARM COULD BE BUILT -- see the rejection codes above. This is a POLICY or "
                     "RUNTIME limit, not evidence about the signal.")
    if result.unevaluated:
        lines.append("  UNEVALUATED ARMS (built, never measured):")
        for u in result.unevaluated[:6]:
            lines.append(f"     {u.label}")
    return "\n".join(lines)
