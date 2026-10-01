"""How much does structured localization shrink the search space? -- the funnel, measured.

    N_proposed -> N_(l,phi) -> N_admissible -> N_executable -> N_evaluated -> N_accepted

OBSERVATIONAL ONLY. This reads `AnchorPolicyOpt`'s existing outputs (`arms`, `rejected`) and counts
them; it never decides feasibility, and importing it cannot change an arm.

WHY THE STAGES ARE THESE AND NOT THE PAPER'S
--------------------------------------------
The stage boundaries mirror the checks the optimizer actually performs, in the order it performs
them, because a funnel whose stages do not correspond to real gates cannot attribute a drop:

  proposed      candidate (locus, signal, action-family) triples the proposer emitted
  localized     survived (l, phi) admission -- the signal is observable at that boundary
  admissible    survived U_H(l): the host DECLARES the action executable there
  executable    survived executor_supports(): an executor exists, covers the signal, and consumes
                eta UNCHANGED. This is the stage that matters -- admissible != materializable, and
                conflating them inflated a reported "113 counterfactuals" by 91 non-runnable arms.
  evaluated     arms x theta actually measured against the incumbent
  accepted      promoted by measurement

`localized` and `admissible` are separate because they fail for different reasons and the
distinction is the whole claim: localization prunes BEFORE the host is consulted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

FUNNEL_STAGES = ("proposed", "localized", "admissible", "executable", "evaluated", "accepted")

# The six reasons a candidate leaves the funnel. Keys are OUR vocabulary; `classify_rejection`
# maps the codes the optimizer and the action contract actually emit onto them, so a new code
# upstream shows up as "unclassified" rather than being silently bucketed.
REJECTION_REASONS = (
    "signal_not_observable",
    "action_not_admissible",
    "no_executor",
    "eta_incompatible",
    "duplicate_candidate",
    "evaluation_failure",
)

# Codes as emitted by anchoropt.learning.anchor_policy_opt / action_contract, verified against the
# source rather than guessed. `not_materializable_by_host_executor` covers TWO distinct facts --
# no executor at all, versus an executor that would have to coerce eta -- and the detail string is
# what separates them, so classify_rejection inspects it.
_CODE_MAP = {
    "signal_not_observable_at_boundary": "signal_not_observable",
    "signal_params_unconfigured": "signal_not_observable",
    "not_executable_in_host": "action_not_admissible",
    "not_exactly_groundable": "eta_incompatible",
    "no_action_parameters": "eta_incompatible",
    "no_parameter_grid": "evaluation_failure",
    "REROUTE_INFEASIBLE": "eta_incompatible",
    "REPROMPT_INFEASIBLE": "eta_incompatible",
    "SUPPRESS_INFEASIBLE": "eta_incompatible",
    "TRANSFORM_INFEASIBLE": "eta_incompatible",
}


def classify_rejection(reason_code: str, detail: str = "") -> str:
    """Map an emitted rejection code onto one of REJECTION_REASONS, or 'unclassified'.

    `not_materializable_by_host_executor` is deliberately split on the detail text: "no executor at
    this boundary" and "the executor would have to coerce your eta" are different findings about the
    host, and a funnel that merges them cannot say whether the gap is capability or parameters.
    """
    code = (reason_code or "").strip()
    det = (detail or "")
    if code == "not_materializable_by_host_executor":
        if "executor_parameter_conflict" in det:
            return "eta_incompatible"
        if "executor_signal_unsupported" in det:
            return "signal_not_observable"
        return "no_executor"
    if code in _CODE_MAP:
        return _CODE_MAP[code]
    for prefix, mapped in _CODE_MAP.items():
        if code.startswith(prefix):
            return mapped
    return "unclassified"


@dataclass
class FunnelReport:
    """Stage counts, per-reason rejection counts, and the shrink factors between stages."""

    stages: dict[str, int]
    rejections: dict[str, int]
    unclassified: tuple[str, ...] = ()

    def shrink(self, a: str, b: str) -> float | None:
        """Multiplicative change from stage `a` to stage `b` (1.0 = no pruning).

        CAN EXCEED 1.0, and that is not a bug: `proposed` and `localized` count ACTION FAMILIES,
        while `admissible` onward count grounded (mu, eta_mu) ARMS, and one family expands into
        several groundings (three reprompt instructions, six destination x retry pairs). So
        localized->admissible measures EXPANSION and admissible->executable measures PRUNING.
        `pruning_ratio` reports only the part that is genuinely a reduction.
        """
        n_a, n_b = self.stages.get(a, 0), self.stages.get(b, 0)
        return None if not n_a else n_b / n_a

    @property
    def pruning_ratio(self) -> float | None:
        """How much executor-backed feasibility removes: executable / admissible.

        This is the number that answers "how much does structured localization shrink the search
        space" on the arm-for-arm comparison; the earlier stages change units.
        """
        return self.shrink("admissible", "executable")

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"stages": dict(self.stages), "rejections": dict(self.rejections)}
        out["shrink"] = {f"{a}->{b}": self.shrink(a, b)
                         for a, b in zip(FUNNEL_STAGES, FUNNEL_STAGES[1:])}
        out["pruning_ratio_admissible_to_executable"] = self.pruning_ratio
        out["units_note"] = ("proposed/localized count ACTION FAMILIES; admissible onward count "
                             "grounded arms, so localized->admissible is expansion, not pruning")
        if self.unclassified:
            out["unclassified_codes"] = list(self.unclassified)
        return out


@dataclass
class SearchFunnel:
    """Accumulates one residual round's candidate funnel.

    Counts are ADDED explicitly by a caller that already has the optimizer's output; nothing is
    inferred, and the recorder never re-derives feasibility.
    """

    stages: dict[str, int] = field(default_factory=lambda: {s: 0 for s in FUNNEL_STAGES})
    rejections: dict[str, int] = field(default_factory=lambda: {r: 0 for r in REJECTION_REASONS})
    _unclassified: list[str] = field(default_factory=list)

    def record_stage(self, stage: str, n: int = 1) -> None:
        if stage not in self.stages:
            raise KeyError(f"unknown funnel stage {stage!r}; expected one of {FUNNEL_STAGES}")
        self.stages[stage] += int(n)

    def record_rejection(self, reason_code: str, detail: str = "", n: int = 1) -> str:
        bucket = classify_rejection(reason_code, detail)
        if bucket == "unclassified":
            self._unclassified.append(reason_code)
        else:
            self.rejections[bucket] += int(n)
        return bucket

    def observe_build(self, arms: Sequence[Any], rejected: Sequence[Any], *,
                      proposed: int | None = None, localized: int | None = None) -> None:
        """Read one `AnchorPolicyOpt.build_arms()` result.

        `arms` are the EXECUTABLE survivors (build_arms applies executor_supports before returning),
        so they land on `executable`. `admissible` is executable + everything rejected at or after
        the host check, i.e. every candidate the host declared legal.
        """
        n_exec = len(arms)
        self.record_stage("executable", n_exec)
        n_admissible = n_exec
        for rj in rejected:
            code = getattr(rj, "reason_code", "") or ""
            detail = getattr(rj, "detail", "") or ""
            bucket = self.record_rejection(code, detail)
            # A candidate rejected for a reason DOWNSTREAM of U_H(l) was admissible.
            if bucket in ("no_executor", "eta_incompatible", "evaluation_failure"):
                n_admissible += 1
        self.record_stage("admissible", n_admissible)
        if localized is not None:
            self.record_stage("localized", localized)
        if proposed is not None:
            self.record_stage("proposed", proposed)

    def report(self) -> FunnelReport:
        return FunnelReport(stages=dict(self.stages), rejections=dict(self.rejections),
                            unclassified=tuple(sorted(set(self._unclassified))))
