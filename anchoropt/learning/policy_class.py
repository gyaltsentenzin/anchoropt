"""POLICY as a first-class object: the decision rule and its parameter domain.

THE DECOMPOSITION THIS MODULE MAKES EXPLICIT
--------------------------------------------
    SIGNAL     what is OBSERVABLE at a decision point        Phi, chosen/expanded in the SIGNAL block
    POLICY     the DECISION RULE over that signal, and its parameters   <- this module
    EXECUTION  carrying out the selected action                          mu, applied by the runtime

The LLM proposer proposes the policy STRUCTURE -- (l, phi_theta, mu), a FAMILY. AnchorOpt then
optimizes the parameter:

    theta* = argmax_{theta in Theta} J_train(l, phi_theta, mu)

Self-Evolve R1 is the evidence for the split. A proposer chose `below = 0.75` from a single observed
score of 0.4302; measured, that fired on 52% of cases -- 11 of them already passing -- and cost
-3.37 pp. The right reading is NOT "the family is bad": the controller engaged, and the harm was
concentrated where it engaged, which is what an over-broad THRESHOLD looks like. One-shot numeric
selection by a language model is what failed. A family whose parameter has never been searched has
not been tested.

WHY "POLICY" AND NOT "THRESHOLD"
-------------------------------
`PolicyClass` is deliberately an open enum with a `parameter_domain()` seam. Thresholds are the only
class implemented here, and the abstraction exists so that DETERMINISTIC (no parameters),
PARAMETERIZED (this one), CLASSIFIER (learned) and LLM_POLICY (a model in the loop) can be added
without the optimizer, the search, or the block loop learning that a policy was ever a number.
Hard-coding `policy = threshold` now is the thing that would make those additions a rewrite.

The optimizer below therefore consumes an ABSTRACT parameter space -- a list of candidate theta
mappings -- and knows nothing about what any parameter means.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class PolicyClass(enum.Enum):
    """What KIND of decision rule a controller uses. Open by design; only two are implemented.

    The value is the wire name a runtime or proposal may use. Adding a member is a deliberate,
    reviewable act -- as with the action vocabulary, an open-ended set is how a closed space stops
    being closed.
    """

    DETERMINISTIC = "deterministic"      # fires on a fixed predicate; no free parameters
    PARAMETERIZED = "parameterized"      # a threshold/range family: theta is searched
    CLASSIFIER = "classifier"            # NOT IMPLEMENTED -- a learned decision function
    LLM_POLICY = "llm_policy"            # NOT IMPLEMENTED -- a model consulted at the decision point


IMPLEMENTED_CLASSES = frozenset({PolicyClass.DETERMINISTIC, PolicyClass.PARAMETERIZED})


class PolicySpecError(ValueError):
    """A policy whose class or parameter domain is not usable. Carries why, for the decline log."""


# ================================================================================================
# parameter domains
# ================================================================================================

@dataclass(frozen=True)
class ParameterDomain:
    """The values a single policy parameter may take. Declared by the RUNTIME, never inferred.

    Two ways to say it, and the distinction matters for how theta is searched:

        `values`        an explicit, ordered candidate list -- categorical or a fixed sweep
        `low`/`high`    a continuous range, from which candidates are drawn

    `observed_from` names the runtime field whose OBSERVED distribution should seed the grid. This is
    requirement 5's substance: for a 1-D threshold the meaningful candidates are the values the data
    actually takes, not a tidy arithmetic sweep and not a number recalled from a previous anchor. A
    grid over observed quantiles cannot propose a threshold that discriminates nothing.
    """

    name: str
    kind: type
    low: float | None = None
    high: float | None = None
    values: tuple[Any, ...] = ()
    observed_from: str = ""
    monotone: str = ""          # "lower_fires_less" | "higher_fires_less" | "" -- documentation only
    doc: str = ""

    def __post_init__(self) -> None:
        if not self.values and (self.low is None or self.high is None):
            raise PolicySpecError(
                f"parameter {self.name!r} declares neither an explicit `values` list nor a "
                f"low/high range; the optimizer cannot enumerate an undeclared domain")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise PolicySpecError(f"parameter {self.name!r} has low > high")

    def contains(self, value: Any) -> bool:
        if self.values:
            return value in self.values
        try:
            return float(self.low) <= float(value) <= float(self.high)
        except (TypeError, ValueError):
            return False

    def clamp(self, value: Any) -> Any:
        """Bring a value inside the domain, for an LLM HINT that overshoots.

        A hint outside the declared domain is not an error -- it is a hint. Clamping keeps it usable
        as an initialization while making the deployed value the runtime's business.
        """
        if self.values:
            return value if value in self.values else self.values[0]
        try:
            return min(max(float(value), float(self.low)), float(self.high))
        except (TypeError, ValueError):
            return self.low


def quantile_grid(observations: Sequence[float], *, n: int = 9,
                  domain: ParameterDomain | None = None) -> tuple[float, ...]:
    """A deterministic grid over OBSERVED values, at evenly spaced quantiles.

    Requirement 5, and the reason it is quantiles rather than a linear sweep: a linear sweep over
    [0, 1] spends most of its budget where no data lives, so most arms are byte-identical to the
    control or fire on everything. Quantiles of the observed distribution guarantee each candidate
    partitions the data differently.

    Deterministic: the same observations always give the same grid, so a rerun evaluates the same
    arms. No LLM is asked for a number, and no historical anchor value is privileged -- if a past
    threshold happens to fall in the grid it is there because the data put it there.
    """
    xs = sorted(float(x) for x in observations if x is not None)
    if not xs:
        return ()
    out: list[float] = []
    for i in range(1, n + 1):
        q = i / (n + 1)
        idx = min(int(q * (len(xs) - 1) + 0.5), len(xs) - 1)
        v = round(xs[idx], 4)
        if domain is not None and not domain.contains(v):
            continue
        if v not in out:
            out.append(v)
    return tuple(out)


# ================================================================================================
# the policy spec
# ================================================================================================

@dataclass(frozen=True)
class PolicySpec:
    """A controller FAMILY: structure fixed by the proposer, parameters left to the optimizer.

    `llm_hint` records what the proposer suggested, for provenance only. Requirement 2: it is an
    initialization, never the deployed value. Keeping it is worth doing -- comparing the hint to
    theta* is exactly how R1's failure became legible.
    """

    boundary: Any                              # IncisionPoint
    signal: str
    action: Any                                # Action
    policy_class: PolicyClass = PolicyClass.DETERMINISTIC
    domains: tuple[ParameterDomain, ...] = ()
    theta: Mapping[str, Any] = field(default_factory=dict)          # action parameters (fixed)
    llm_hint: Mapping[str, Any] = field(default_factory=dict)       # PROVENANCE ONLY
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.policy_class not in IMPLEMENTED_CLASSES:
            raise PolicySpecError(
                f"policy class {self.policy_class.value!r} is declared but NOT IMPLEMENTED; "
                f"implemented: {sorted(c.value for c in IMPLEMENTED_CLASSES)}")
        if self.policy_class is PolicyClass.PARAMETERIZED and not self.domains:
            raise PolicySpecError(
                "a PARAMETERIZED policy must declare at least one parameter domain, else there is "
                "nothing to optimize and it should be DETERMINISTIC")
        if self.policy_class is PolicyClass.DETERMINISTIC and self.domains:
            raise PolicySpecError(
                "a DETERMINISTIC policy declares parameter domains; that is a PARAMETERIZED policy")

    @property
    def is_parameterized(self) -> bool:
        return self.policy_class is PolicyClass.PARAMETERIZED

    def candidate_thetas(self, observations: Mapping[str, Sequence[float]] | None = None,
                         *, n: int = 9) -> tuple[Mapping[str, Any], ...]:
        """Enumerate the signal-parameter settings to evaluate. ONE dimension in v0.1.

        Multi-dimensional theta is deliberately refused rather than silently producing a product
        grid: a 2-D sweep is a different experiment with a different evaluation budget, and quietly
        multiplying the arm count is how a run becomes unaffordable without anyone deciding to.
        """
        if not self.is_parameterized:
            return ({},)
        if len(self.domains) != 1:
            raise PolicySpecError(
                f"v0.1 optimizes ONE parameter; {self.signal!r} declares {len(self.domains)}. A "
                f"product grid multiplies the evaluation budget and needs its own decision.")
        dom = self.domains[0]
        if dom.values:
            grid = tuple(dom.values)
        else:
            obs = (observations or {}).get(dom.observed_from or dom.name) or ()
            grid = quantile_grid(obs, n=n, domain=dom)
            if not grid:
                raise PolicySpecError(
                    f"no observed values for {dom.observed_from or dom.name!r}, so no data-driven "
                    f"grid can be built. Refusing to invent a sweep over an unobserved range.")
        return tuple({dom.name: v} for v in grid)

    def hint_in_grid(self, grid: Sequence[Mapping[str, Any]]) -> bool:
        """Would the proposer's own number have been evaluated? Provenance, not a gate."""
        if not self.llm_hint:
            return False
        return any(all(g.get(k) == v for k, v in self.llm_hint.items()) for g in grid)


# ================================================================================================
# the training objective and the search
# ================================================================================================

@dataclass(frozen=True)
class ThetaResult:
    """One theta's measured outcome on TRAIN. Every field the acceptance rule or a reader needs."""

    theta: Mapping[str, Any]
    gains: tuple[str, ...]
    losses: tuple[str, ...]
    firings: int
    cases_fired: int
    n: int
    interventions_executed: int = 0
    accuracy_delta_pp: float = 0.0
    cost_calls: int | None = None
    detail: str = ""

    @property
    def net(self) -> int:
        return len(self.gains) - len(self.losses)

    @property
    def firing_rate(self) -> float:
        return (self.cases_fired / self.n) if self.n else 0.0


def train_objective(r: ThetaResult) -> tuple:
    """J_train. Higher is better; returns a sortable tuple so ties break deterministically.

    Ordered lexicographically:

      1. net gains-minus-losses            the thing being optimized
      2. ENGAGEMENT non-zero               a theta with zero firings has a delta it did not cause,
                                           and `docs/ACCEPTANCE_RULE.md` treats that as proof of
                                           NON-attribution. So an inert theta can never outrank an
                                           engaged one at equal net -- R1's candidate 3 scored 0 pp
                                           with 0 firings and must not read as a tie with a real 0.
      3. FEWER firings                     at equal net and both engaged, the more selective rule is
                                           preferred: it achieves the same with less interference,
                                           and R1's -3.37 pp came from firing on 52% of cases,
                                           11 of which were already passing.
      4. the theta value itself            a stable final tiebreak so a rerun picks the same arm
    """
    engaged = 1 if r.interventions_executed > 0 else 0
    # The final tiebreak is a STRING of the theta items: it must be total and stable, and it must
    # never raise on an empty theta. An earlier version indexed a tuple comprehension and was both
    # unreadable and wrong for a deterministic policy.
    tiebreak = ";".join(f"{k}={v}" for k, v in sorted(r.theta.items()))
    return (r.net, engaged, -r.firing_rate, tiebreak)


def optimize_theta(spec: PolicySpec, *,
                   evaluate: Callable[[Mapping[str, Any]], ThetaResult],
                   observations: Mapping[str, Sequence[float]] | None = None,
                   n_grid: int = 9,
                   objective: Callable[[ThetaResult], tuple] = train_objective,
                   ) -> tuple[Mapping[str, Any], list[ThetaResult]]:
    """theta* = argmax J_train over the declared domain. TRAIN ONLY.

    `evaluate(theta) -> ThetaResult` runs one paired arm against the SAME incumbent. Every theta is
    measured against that one control, which is what makes the comparison across theta valid.

    Returns (theta*, every result). The full sweep is returned rather than only the winner because
    the SHAPE of J over theta is the finding -- a single peak, a plateau, or no engaged theta at all
    are three different conclusions, and reporting only argmax hides which one happened.

    Held-out data is never touched here. Requirement 4: theta is frozen after this and evaluated on
    validation ONCE, by the caller.
    """
    grid = spec.candidate_thetas(observations, n=n_grid)
    if not grid:
        raise PolicySpecError(f"empty theta grid for {spec.signal!r}")
    results = [evaluate(theta) for theta in grid]
    best = max(results, key=objective)
    return dict(best.theta), results


def summarize_sweep(spec: PolicySpec, results: Sequence[ThetaResult]) -> str:
    """A readable sweep table. Reports firing rate beside net, because one explains the other."""
    lines = [f"theta sweep for {spec.signal} @ {getattr(spec.boundary, 'value', spec.boundary)} "
             f"/ {getattr(spec.action, 'value', spec.action)}   (n={results[0].n if results else 0})",
             f"{'theta':>18}  {'net':>5} {'+':>4} {'-':>4}  {'fire%':>6} {'exec':>6}  {'dAcc pp':>8}"]
    for r in sorted(results, key=lambda x: str(x.theta)):
        tv = ", ".join(f"{k}={v}" for k, v in r.theta.items()) or "(none)"
        lines.append(f"{tv:>18}  {r.net:>+5d} {len(r.gains):>4d} {len(r.losses):>4d}  "
                     f"{100*r.firing_rate:>5.1f}% {r.interventions_executed:>6d}  "
                     f"{r.accuracy_delta_pp:>+8.2f}")
    if spec.llm_hint:
        hint = ", ".join(f"{k}={v}" for k, v in spec.llm_hint.items())
        lines.append(f"  LLM hint was {hint} -- provenance only, never the deployed value")
    return "\n".join(lines)
