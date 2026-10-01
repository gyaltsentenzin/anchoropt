"""EXTERNAL PAIRED EVALUATION: how a host whose evaluation is not inline supplies real results.

THE PROBLEM THIS SOLVES
-----------------------
`AnchorPolicyOpt.optimize(proposal, incumbent=, evaluate=)` needs a callback that returns a measured
`ThetaResult` per arm. A toy host can do that inline -- run the episodes, compare, return. A real
benchmark often cannot: evaluating one arm means submitting a GPU job, waiting, and scoring the
artifacts hours later, in a different process.

The BFCL driver's response was to pass `evaluate=lambda _a: 0` and then choose an arm by
`abs(firing_rate - 0.25)` -- nearest to firing on 25% of observed states. That is a heuristic standing
in for a measurement, and it has three separate problems:

  1. it SELECTS without measuring, so the "optimizer" never optimized anything;
  2. the round then reported `NO_BENEFIT`, which asserts a measurement that never happened;
  3. the heuristic has no theory behind it -- 25% is a guess about what a useful trigger rate is, and
     a controller firing on 24% of states is not thereby better than one firing on 60%.

WHAT REPLACES IT
----------------
Nothing replaces the heuristic with another heuristic. Selection requires measurement, so when no
measurement exists the honest outcome is UNEVALUATED -- the arms are built, recorded, and handed out
for evaluation, and NO winner is named.

`ExternalEvaluation` is the seam. A driver:

    1. builds arms through core (`build_arms`), emits the manifest, and stops at UNEVALUATED;
    2. an external process evaluates each arm -- GPU job, cluster, whatever -- and writes results;
    3. the driver loads those results and calls the SAME `AnchorPolicyOpt.optimize`, with an
       `evaluate` callback backed by the recorded results.

Step 3 is the point: the optimizer is the same code path the toy host exercises inline. There is no
second optimizer, and no path where an unmeasured arm gets a score.

THE ONE RULE THIS FILE ENFORCES
-------------------------------
An arm with no recorded result returns `None`, never a zero. A zero is a measurement ("we ran it and it
changed nothing"); `None` is the absence of one. Coercing the second into the first is how a null gets
banked against an intervention nobody ran -- and `train_objective` would rank that fabricated zero
against real results.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anchoropt.learning.policy_class import ThetaResult

# Outcome classes a round can report. These are DISJOINT and the distinction is the deliverable:
# collapsing any pair of them is how an unmeasured arm becomes a negative result.
UNEVALUATED = "UNEVALUATED"                  # arms built; nothing measured. Not a result about them.
EVALUATED_NO_BENEFIT = "NO_BENEFIT"          # arms measured; none beat the incumbent. A real null.
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"        # measurement stopped early; arms still open. Requeue.
IMPROVED = "IMPROVED"                        # a measured arm beat the incumbent on TRAIN only.
# TRAIN net > 0 is criterion 1 of FOUR (docs/ACCEPTANCE_RULE.md). It is not acceptance, and naming it
# IMPROVED invited exactly that read: a round reporting IMPROVED was taken as an accepted controller
# while criteria 2-4 (independent dev non-regression, causal attribution, safety) had not been applied.
# This alias names what was actually established, and `SelectionOutcome.accepted` stays False until the
# protocol runs at its designated checkpoint.
TRAIN_IMPROVED_PENDING_VALIDATION = "TRAIN_IMPROVED_PENDING_VALIDATION"
# A measurement that cannot be compared to this round's incumbent at all. NOT a null and NOT a win:
# the paired contrast is against a different harness configuration, so it says nothing about this
# round's candidate. Kept disjoint from NO_BENEFIT for the same reason UNEVALUATED is.
INCOMPARABLE_INCUMBENT = "INCOMPARABLE_INCUMBENT"


@dataclass(frozen=True)
class ArmResult:
    """One arm's externally measured paired outcome, against the round's frozen incumbent.

    `gains`/`losses` are CASE IDs, not counts, so the denominator can be checked and the same case
    cannot be counted twice. `cases_fired`/`interventions_executed` must come from the run's own
    trajectory record -- a registry-derived flag cannot see whether a mechanism actually ran, and an
    arm reporting 0 executions has a delta it did not cause.
    """

    arm_label: str
    gains: tuple[str, ...]
    losses: tuple[str, ...]
    n: int
    cases_fired: int = 0
    interventions_executed: int = 0
    accuracy_delta_pp: float = 0.0
    theta: Mapping[str, Any] = field(default_factory=dict)
    incumbent_token: str = ""
    denominator_ok: bool = True
    detail: str = ""

    def as_theta_result(self) -> ThetaResult:
        return ThetaResult(
            theta=dict(self.theta), gains=tuple(self.gains), losses=tuple(self.losses),
            firings=int(self.interventions_executed), cases_fired=int(self.cases_fired),
            n=int(self.n), interventions_executed=int(self.interventions_executed),
            accuracy_delta_pp=float(self.accuracy_delta_pp), detail=self.detail)


class MissingEvaluation(KeyError):
    """Raised only when a caller demands strictness. The default is to return None."""


@dataclass
class ExternalEvaluation:
    """An `evaluate` callback backed by externally recorded results.

    Pass it straight to `AnchorPolicyOpt.optimize(..., evaluate=ev)`. It is deliberately dumb: it looks
    a result up and converts it. Every decision about which arm wins stays in the optimizer.
    """

    results: dict[str, ArmResult] = field(default_factory=dict)
    incumbent_token: str = ""
    budget: int | None = None
    strict: bool = False

    # ---- observability, so a driver can report the right outcome class ----
    requested: list[str] = field(default_factory=list)
    served: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    budget_exhausted: bool = False
    # Arms whose recorded result was measured against a DIFFERENT incumbent than this round's.
    # Kept apart from `missing`: an absent result is unmeasured, whereas these were measured and the
    # measurement answers a different question.
    incomparable: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str | Path, **kw) -> "ExternalEvaluation":
        """Load a results manifest written by whatever evaluated the arms.

        Accepts either a list of records or {"results": [...]}. Unknown keys are ignored so a richer
        external record does not have to be trimmed to fit.
        """
        raw = json.loads(Path(path).read_text())
        rows = raw.get("results", raw) if isinstance(raw, Mapping) else raw
        fields = set(ArmResult.__dataclass_fields__)
        out: dict[str, ArmResult] = {}
        for row in rows or ():
            d = {k: v for k, v in dict(row).items() if k in fields}
            d["gains"] = tuple(d.get("gains") or ())
            d["losses"] = tuple(d.get("losses") or ())
            r = ArmResult(**d)
            out[r.arm_label] = r
        token = raw.get("incumbent_token", "") if isinstance(raw, Mapping) else ""
        return cls(results=out, incumbent_token=str(token or ""), **kw)

    # ---- the callback ----
    def __call__(self, arm, theta=None) -> ThetaResult | None:
        label = getattr(arm, "label", str(arm))
        self.requested.append(label)

        if self.budget is not None and len(self.served) >= self.budget:
            # BUDGET, not absence. The arm may well have a result; we are declining to spend more.
            self.budget_exhausted = True
            return None

        res = self.results.get(label)
        if res is None:
            # THE ONE RULE. No result means NO RESULT -- never a zero. A zero would be ranked by
            # `train_objective` against real measurements, banking a null nobody observed.
            self.missing.append(label)
            if self.strict:
                raise MissingEvaluation(
                    f"no external evaluation recorded for arm {label!r}. Arms with no result are "
                    f"UNEVALUATED; they must not be scored.")
            return None

        # INCUMBENT IDENTITY IS A PRECONDITION, exactly like the denominator.
        #
        # `incumbent_token` was carried on every record and on the manifest, and NOTHING EVER COMPARED
        # THEM. That is how a round measured against a harness already carrying a zero-call reprompt
        # and a cross-container-merge controller (55/89) was reported as an improvement on a raw
        # baseline of 19/89 -- a +37-case gap the candidate had no part in. The arm was real and the
        # pairing internally sound; it simply answered a question about a different incumbent.
        #
        # Comparison is exact-match on a non-empty token. An empty token on either side is NOT treated
        # as a mismatch: older manifests predate the field, and silently voiding them would rewrite
        # results that were correctly scored. It is recorded as unverifiable instead.
        want = str(self.incumbent_token or "")
        got = str(res.incumbent_token or "")
        if want and got and want != got:
            self.incomparable.append(label)
            if self.strict:
                raise MissingEvaluation(
                    f"arm {label!r} was measured against incumbent {got!r}, but this round's "
                    f"incumbent is {want!r}. A paired result against a different harness "
                    f"configuration is not a measurement of this candidate.")
            return None

        if not res.denominator_ok:
            # VALIDITY BEFORE ACCEPTANCE. Two runs that scored different cases are not a paired
            # comparison, so there is no result to rank -- not a negative one.
            self.missing.append(label)
            if self.strict:
                raise MissingEvaluation(
                    f"arm {label!r} reports denominator_ok=False: the control and arm runs did not "
                    f"score the same cases, so this is not a paired comparison and cannot be scored")
            return None

        self.served.append(label)
        return res.as_theta_result()

    # ---- what the driver reports ----
    @property
    def n_completed(self) -> int:
        """COMPLETED evaluations. Not attempts, and not arms built."""
        return len(self.served)

    def outcome_class(self, *, improved: bool) -> str:
        """Which disjoint state this round is in. Order matters.

        A train win wins outright, and is named TRAIN_IMPROVED_PENDING_VALIDATION because train net > 0
        is criterion 1 of four -- the round has a provisional winner, not an accepted controller.

        Otherwise, INCOMPARABLE_INCUMBENT is checked BEFORE the null classes. A round whose only
        results were measured against another incumbent has established nothing about its candidates,
        and calling that NO_BENEFIT would bank a null nobody measured -- the same error class as
        scoring an absent result as zero.

        Then: nothing measured at all is UNEVALUATED, whatever else happened. Budget exhaustion is
        checked before NO_BENEFIT because arms left unmeasured are still open.
        """
        if improved:
            return TRAIN_IMPROVED_PENDING_VALIDATION
        if self.n_completed == 0:
            return INCOMPARABLE_INCUMBENT if self.incomparable else UNEVALUATED
        if self.budget_exhausted or self.missing:
            return BUDGET_EXHAUSTED
        return EVALUATED_NO_BENEFIT

    def report(self) -> dict:
        return {"arms_requested": len(self.requested), "evaluations_completed": self.n_completed,
                "unevaluated_arms": sorted(set(self.missing)),
                "incomparable_arms": sorted(set(self.incomparable)),
                "budget": self.budget, "budget_exhausted": self.budget_exhausted,
                "incumbent_token": self.incumbent_token}


def arm_manifest(arms: Sequence[Any], *, incumbent_id: str, incumbent_token: str = "") -> dict:
    """The work order handed to whatever will evaluate these arms.

    Every field an external evaluator needs to BUILD the arm, and the labels it must report back
    under. Written by the propose phase; consumed by the scoring phase. This is what replaces choosing
    an arm by a firing-rate heuristic: all of them are emitted, and measurement decides.

    `fires_on_states` IS AN UPPER BOUND, NOT MEASURED SUPPORT, and the manifest now says so in its own
    `projection_caveat`. A host projects it by reconstructing state offline, and a reconstruction is
    blind to state it does not hold. Measured instance on one adapter: an arm projected 201 of 590
    firings while the host's LIVE comparator, reading real state, found exactly ONE qualifying case --
    a 201x overstatement, because the offline reconstruction could not see state an earlier phase had
    already built.

    This matters to CORE and not only to that host: `candidate_selection` ranks by `support`, which an
    arm inherits from the residual problem its signal addresses. So an overstated projection can send
    a deterministic policy to spend a GPU round on a one-case mechanism, with nothing in the record
    saying the support was never measured. The caveat travels with the number so a consumer cannot
    quote it as evidence.
    """
    return {
        "incumbent_id": incumbent_id,
        "incumbent_token": incumbent_token,
        "n_arms": len(arms),
        "note": ("Evaluate each arm against the SAME frozen incumbent and report results keyed by "
                 "`arm_label`. An arm you did not evaluate must be OMITTED, not reported as zero: "
                 "core treats a missing result as UNEVALUATED and a zero as a measurement."),
        "projection_caveat": (
            "`fires_on_states` is an UPPER BOUND from an offline projection, never measured support. A "
            "reconstruction is blind to state it does not hold -- one measured instance overstated by "
            "201x (201 projected, 1 qualifying case against live state). Do not treat a projected "
            "count as evidence that a residual is worth an evaluation."),
        "arms": [{
            "arm_label": a.label,
            "boundary": a.boundary.value,
            "signal": a.signal,
            "action": a.action.value,
            "operator": a.instantiated.operator.value,
            "variant": a.instantiated.variant,
            "eta": {k: (v if isinstance(v, (int, float, bool, str)) else str(v))
                    for k, v in a.eta.items()},
        } for a in arms],
    }


# ================================================================================================
# SELECTION ON MEASUREMENT -- core owns this, a driver only supplies the results
# ================================================================================================

@dataclass(frozen=True)
class SelectionOutcome:
    """What measurement chose among already-built arms, and which outcome class the round is in.

    `winner is None` is a legitimate and common answer: with nothing measured there is no winner to
    name, and `outcome_class` says UNEVALUATED rather than reporting a null about arms nobody ran.
    """

    outcome_class: str
    winner: Any = None
    winner_theta: Mapping[str, Any] = field(default_factory=dict)
    winner_result: ThetaResult | None = None
    per_group: tuple[Mapping[str, Any], ...] = ()
    evaluations_completed: int = 0
    unevaluated: tuple[str, ...] = ()
    report: Mapping[str, Any] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        """TRAIN criterion only: net > 0 on the paired train cases. NOT acceptance."""
        return self.winner_result is not None and self.winner_result.net > 0

    @property
    def accepted(self) -> bool:
        """ALWAYS FALSE HERE, and that is the point.

        Selection on measurement establishes criterion 1 of four. Criteria 2-4 -- independent dev
        non-regression, causal attribution from the controller's own telemetry, and safety -- are
        applied by the acceptance protocol at its designated checkpoint, on data this function must
        never see. A property that could return True here would put held-out evidence inside the
        selection path, which is the contamination the train/held-out separation exists to prevent.
        """
        return False

    @property
    def is_negative_result(self) -> bool:
        """ONLY an evaluated no-benefit is a negative result. The other classes are not."""
        return self.outcome_class == EVALUATED_NO_BENEFIT

    def detail(self) -> str:
        return {
            TRAIN_IMPROVED_PENDING_VALIDATION: (
                f"provisional TRAIN winner on measured J_train: net "
                f"{self.winner_result.net:+d}. Criterion 1 of 4 only -- dev non-regression, causal "
                f"attribution and safety are NOT yet applied, so this is not an accepted controller"
                if self.winner_result else TRAIN_IMPROVED_PENDING_VALIDATION),
            INCOMPARABLE_INCUMBENT: (
                "every recorded result was measured against a DIFFERENT incumbent than this round's, "
                "so nothing here is evidence about these candidates -- not a null, not a win"),
            UNEVALUATED: "arms were built and NONE was measured; this is not evidence about them",
            BUDGET_EXHAUSTED: ("some arms were left unmeasured when the budget ran out; the round "
                               "stays open -- requeue, do not classify"),
            EVALUATED_NO_BENEFIT: ("every arm was measured against the frozen incumbent and none "
                                   "improved on it: a real negative result"),
        }.get(self.outcome_class, self.outcome_class)

    def as_dict(self) -> dict:
        return {"outcome": self.outcome_class, "is_negative_result": self.is_negative_result,
                # Explicit, so no reader infers acceptance from a train win.
                "accepted": self.accepted,
                "train_criterion_only": self.improved,
                "detail": self.detail(),
                "winner": getattr(self.winner, "label", None),
                "winner_theta": dict(self.winner_theta),
                "net": self.winner_result.net if self.winner_result else None,
                "evaluations_completed": self.evaluations_completed,
                "unevaluated": list(self.unevaluated),
                "per_group": [dict(g) for g in self.per_group],
                "external_evaluation": dict(self.report)}


def select_on_measurement(arms: Sequence[Any], *, runtime, host, incumbent_id: str,
                          incumbent_token: str = "", evaluation: "ExternalEvaluation",
                          case_ids: Sequence[str] = (),
                          observations: Mapping[str, Sequence[float]] | None = None,
                          ) -> SelectionOutcome:
    """Choose among ALREADY-BUILT arms using `AnchorPolicyOpt.optimize` on real measurements.

    WHY THIS LIVES IN CORE. A driver must not construct proposals or call the optimizer itself -- the
    search belongs to core, and a loop that re-derives the search space can steer it (a property
    `tests/test_cycle2_thinness.py` enforces against the driver's executable code). So the driver hands
    over the arms it already has plus the measurements it loaded, and core does the grouping, the
    optimization and the outcome classification.

    Arms are grouped by (boundary, signal) because that is one POLICY ROUND: within a group, the
    optimizer measures counterfactual (mu, eta, theta) arms against ONE frozen incumbent, which is what
    makes the argmax across actions meaningful. Across groups the best measured net wins.

    NOTHING HERE RANKS ON FIRING BEHAVIOUR. That is the heuristic this function replaces; selection is
    the argmax of J_train over arms that were actually measured, and an unmeasured arm is not ranked at
    all.
    """
    from anchoropt.learning.anchor_policy_opt import (
        AnchorPolicyOpt, FrozenIncumbent, SearchSpaceProposal,
    )

    opt = AnchorPolicyOpt(runtime=runtime, host=host)
    frozen = FrozenIncumbent(str(incumbent_id), str(incumbent_token or ""))

    groups: dict[tuple, list] = {}
    for arm in arms or ():
        groups.setdefault((arm.boundary, arm.signal), []).append(arm)

    best: tuple[Any, Mapping[str, Any], ThetaResult] | None = None
    rows: list[dict] = []
    for (boundary, signal), group in sorted(groups.items(),
                                            key=lambda kv: (kv[0][0].value, str(kv[0][1]))):
        proposal = SearchSpaceProposal(
            boundary=boundary, signal=signal,
            action_set=tuple(sorted({g.action for g in group}, key=lambda x: x.value)),
            diagnosis_case_ids=tuple(case_ids),
            rationale="externally measured paired evaluation")
        # OBSERVATIONS ARE REQUIRED FOR A PARAMETERIZED ARM TO BE EVALUATED AT ALL.
        #
        # `candidate_thetas` builds a data-driven grid from observed values, and raises when it has
        # none -- so without them every parameterized arm was rejected `no_parameter_grid` and the
        # round reported `arms_built: 2, arms_measured: 0` with an EMPTY unevaluated list. Measured on
        # a real round: two arms with recorded paired results were classified UNEVALUATED, which is
        # the honest class for "nobody measured them" and the WRONG one for "we could not build their
        # grid". The propose path has always passed observations; this path could not accept them.
        res = opt.optimize(proposal, incumbent=frozen, evaluate=evaluation,
                          observations=observations)
        rows.append({"boundary": boundary.value, "signal": signal,
                     "arms_built": len(res.arms), "arms_measured": res.n_arms_measured,
                     "unevaluated": [u.label for u in res.unevaluated],
                     "winner": res.winner.label if res.winner else None,
                     "net": res.winner_result.net if res.winner_result else None})
        if res.winner_result is not None and (best is None
                                              or res.winner_result.net > best[2].net):
            best = (res.winner, dict(res.winner_theta), res.winner_result)

    improved = best is not None and best[2].net > 0
    return SelectionOutcome(
        outcome_class=evaluation.outcome_class(improved=improved),
        winner=best[0] if best else None,
        winner_theta=dict(best[1]) if best else {},
        winner_result=best[2] if best else None,
        per_group=tuple(rows),
        evaluations_completed=evaluation.n_completed,
        unevaluated=tuple(sorted(set(evaluation.missing))),
        report=evaluation.report())
