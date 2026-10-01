"""The four acceptance criteria, made MACHINE-CHECKABLE. No new rule, no new threshold.

WHAT THIS IS
------------
`docs/ACCEPTANCE_RULE.md` states the rule in prose and every acceptance so far has been adjudicated
by hand. `self_evolve.step()` already exposes an `accept(EvaluationResult) -> (bool, reason)` seam,
and `self_evolve._default_accept` implements only criterion 1 plus engagement -- it says so, and it
deliberately omits criterion 2 because v0.1 ran one split. This module implements all four over that
same seam so an unattended round decides for itself.

The criteria, exactly as predefined -- NOT restated, NOT tightened:

  1. train net > 0 on the paired cases
  2. NO AGGREGATE REGRESSION on the independent dev split.  A gain is NOT required: net 0 passes.
  3. attribution + a causally validated mechanism, from the CONTROLLER'S OWN telemetry
  4. safety: the three-counter contract

THREE VERDICTS, NOT TWO
-----------------------
A criterion can PASS, FAIL, or be unresolved. Unresolved is `PENDING_VALIDATION` and it is NOT a
pass -- `step()` receives False, so the round rejects rather than installs on absent evidence. This
is the single most important property here, because every way this project has previously fooled
itself reduces to reading silence as success:

  * an arm that never executed still scores, and a positive delta with zero firings is proof of
    NON-attribution (docs/ACCEPTANCE_RULE.md);
  * a host's per-step telemetry allowlist can DROP undeclared keys, so a controller may run
    correctly and leave no evidence -- indistinguishable from not having run at all;
  * criterion 2 on a split whose CONTROL also scores zero is arithmetically guaranteed to pass and
    carries no information.

So absent telemetry yields PENDING_VALIDATION, which BLOCKS acceptance. A degenerate dev split is
different: it yields PASS_UNINFORMATIVE, which ALLOWS acceptance but flags the weakness. Absent
evidence and weak-but-satisfied evidence are not the same thing, and only the first is a blocker.

CRITERION 2 IS NON-REGRESSION, AND THAT IS NOT NEGOTIABLE DOWNWARD
------------------------------------------------------------------
Requiring a dev GAIN would be a stricter rule than the one preregistered, and substituting a
stricter test of my own for the predefined one is a mistake this project has already made once and
corrected. `dev_net >= 0` passes. What is reported ALONGSIDE the pass is whether the split could
have detected a regression at all: when the control arm scores 0 of N, it could not, and the verdict
becomes PASS_UNINFORMATIVE -- still a pass, because the preregistered rule was met, but carrying
`informative: False` so the weakness travels with the result.

An uninformative split must be REPORTED, never used to retrospectively raise the threshold. Blocking
on it would be replacing a rule fixed before the measurement with a stricter one chosen after seeing
the data, which is the same category of error as weakening it.

CRITERION 4: THREE COUNTERS, ASYMMETRIC BY DESIGN
-------------------------------------------------
  clears_added                    categorically forbidden -- any value > 0 FAILS
  information_losing_removes      a removal NOT verified present elsewhere in LIVE state -- FAILS
  verified_relocations            reported, bounded, NOT destructive -- never a failure

A removal whose copy was confirmed in live state before the source was touched preserves the
information; counting it as destructive would forbid the only safe capacity recovery this project
has. Counting it as free would forbid nothing at all. Hence three counters.

Verified relocations must be counted on the PRESERVATION denominator, not the end-to-end one. Those
two differ whenever a relocation preserves its entry but the retried operation then fails for an
unrelated reason, and reporting the smaller number understates the safety evidence. A measured
instance of exactly that gap is recorded in rounds/AUTONOMY/RELOCATION_CHAIN_RECONCILED.json.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

PASS = "PASS"
FAIL = "FAIL"
PENDING = "PENDING_VALIDATION"

#: A PASS on evidence that could not have failed. It ALLOWS acceptance -- the preregistered rule is
#: what it is -- and exists so the weakness is reported explicitly instead of being absorbed into an
#: ordinary pass.
#:
#: This distinction is deliberate and was got wrong once: an uninformative split must be REPORTED,
#: not used to retrospectively raise the acceptance threshold. Substituting a stricter test of my
#: own for the preregistered one is the same error as weakening it -- both replace the rule that was
#: fixed before the measurement.
PASS_UNINFORMATIVE = "PASS_UNINFORMATIVE"

#: Verdicts that permit installation. `PENDING_VALIDATION` is absent because absent evidence is not
#: evidence; `PASS_UNINFORMATIVE` is present because a criterion that was satisfied as written has
#: been satisfied, whatever its power.
_ALLOWS_ACCEPT = frozenset({PASS, PASS_UNINFORMATIVE})


@dataclass(frozen=True)
class CriterionResult:
    """One criterion's verdict, the evidence it rested on, and what would resolve it."""

    number: int
    name: str
    verdict: str
    detail: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    remedy: str = ""

    @property
    def allows_accept(self) -> bool:
        return self.verdict in _ALLOWS_ACCEPT


@dataclass(frozen=True)
class AcceptanceReport:
    """All four verdicts. Accept only if EVERY criterion allows it."""

    criteria: tuple[CriterionResult, ...]

    @property
    def accepted(self) -> bool:
        return bool(self.criteria) and all(c.allows_accept for c in self.criteria)

    def by_number(self, n: int) -> CriterionResult:
        for c in self.criteria:
            if c.number == n:
                return c
        raise KeyError(f"no criterion {n}")

    @property
    def blocking(self) -> tuple[CriterionResult, ...]:
        return tuple(c for c in self.criteria if not c.allows_accept)

    def reason(self) -> str:
        if self.accepted:
            return "; ".join(f"C{c.number} {c.verdict}" for c in self.criteria)
        return "; ".join(f"C{c.number} {c.name} {c.verdict}: {c.detail}" for c in self.blocking)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason(),
            "criteria": [
                {"number": c.number, "name": c.name, "verdict": c.verdict, "detail": c.detail,
                 "evidence": dict(c.evidence), "remedy": c.remedy}
                for c in self.criteria
            ],
        }


# =================================================================================================
# criterion 1 -- train net > 0
# =================================================================================================

def check_train(evaluation) -> CriterionResult:
    net = int(evaluation.net)
    n = len(evaluation.control)
    ev = {"net": net, "gains": len(evaluation.gains), "losses": len(evaluation.losses),
          "n_paired": n}
    if n == 0:
        return CriterionResult(
            1, "train_net_gt_0", PENDING, "no paired cases -- nothing was measured", ev,
            remedy="run both arms over the intended case set")
    if net > 0:
        return CriterionResult(1, "train_net_gt_0", PASS,
                               f"net +{net} ({len(evaluation.gains)}g/{len(evaluation.losses)}l) "
                               f"on {n} paired cases", ev)
    return CriterionResult(1, "train_net_gt_0", FAIL,
                           f"net {net} on {n} paired cases", ev)


# =================================================================================================
# criterion 2 -- no AGGREGATE REGRESSION on dev.  A gain is not required.
# =================================================================================================

def check_dev(dev: Mapping[str, Any] | None) -> CriterionResult:
    """`dev` carries the dev split's own paired result: arm_score, control_score, n.

    Returns PASS when net >= 0. Returns PASS_UNINFORMATIVE -- which DOES allow acceptance, carrying
    `informative: False` -- when the control also scored zero, because the split then cannot detect
    a regression: the preregistered rule is met, and the weakness is reported rather than used to
    raise the bar after the fact.
    """
    if not dev:
        return CriterionResult(
            2, "dev_no_aggregate_regression", PENDING,
            "no dev evaluation supplied", {},
            remedy="run the candidate on the independent dev split")

    arm = int(dev.get("arm_score", 0) or 0)
    ctl = int(dev.get("control_score", 0) or 0)
    n = int(dev.get("n", 0) or 0)
    net = arm - ctl
    ev = {"arm_score": arm, "control_score": ctl, "n": n, "net": net,
          "firings": int(dev.get("firings", 0) or 0)}

    if n == 0:
        return CriterionResult(2, "dev_no_aggregate_regression", PENDING,
                               "dev split has no cases", ev,
                               remedy="supply a non-empty dev split")
    if net < 0:
        return CriterionResult(2, "dev_no_aggregate_regression", FAIL,
                               f"aggregate regression on dev: {arm} vs {ctl} of {n} (net {net})",
                               ev)
    if arm == 0 and ctl == 0:
        # The preregistered rule is NO AGGREGATE REGRESSION, and none occurred. It passes. The
        # split's zero power is reported as a flag rather than used to block -- an uninformative
        # split must not retrospectively raise the threshold.
        ev["informative"] = False
        return CriterionResult(
            2, "dev_no_aggregate_regression", PASS_UNINFORMATIVE,
            f"no aggregate regression ({arm} vs {ctl} of {n}, net 0), so the preregistered "
            f"criterion is satisfied. REPORTED WEAKNESS: both arms score 0, so this split could "
            f"not have detected a regression and the pass carries no transfer evidence", ev,
            remedy="for informative transfer evidence, evaluate on a split with non-zero control "
                   "accuracy (ACCEPTANCE_RULE.md names leave-one-chain-out as the honest fix and "
                   "marks it deferred). This does NOT gate the current acceptance.")
    ev["informative"] = True
    return CriterionResult(2, "dev_no_aggregate_regression", PASS,
                          f"no aggregate regression: {arm} vs {ctl} of {n} (net {net:+d}); "
                          f"a gain is not required by this criterion", ev)


# =================================================================================================
# criterion 3 -- attribution + a causally validated mechanism, from the controller's OWN telemetry
# =================================================================================================

def check_mechanism(telemetry: Mapping[str, Any] | None,
                    *, required_evidence_key: str = "mechanism_verified",
                    requested_key: str = "mechanism_requested") -> CriterionResult:
    """POSITIVE evidence is required, and it must come from the controller's own execution.

    Counts, in order:
      * `interventions_executed` -- a gain with zero executions is proof of non-attribution, so
        zero here is a FAIL, not a pending.
      * `mechanism_verified` -- how many executions were verified to have done what the mechanism
        claims. Absent => PENDING. Zero while executions occurred => FAIL.
      * unattributed requests -- requests that neither verified nor recorded a reason. These are
        MISSING EVIDENCE, and a partition that does not sum is itself the finding.
    """
    t = dict(telemetry or {})
    fired = int(t.get("signal_firings", 0) or 0)
    executed = int(t.get("interventions_executed", 0) or 0)
    ev: dict[str, Any] = {"signal_firings": fired, "interventions_executed": executed}

    if not t:
        return CriterionResult(
            3, "attribution_and_mechanism", PENDING,
            "no execution telemetry at all -- installation is not execution", ev,
            remedy="report firings from the host's per-step execution record. A registry-derived "
                   "firing count cannot see an intervention's own remedy flags, and per-step "
                   "telemetry allowlists silently drop undeclared keys -- so declare the keys "
                   "BEFORE the round, not after reading an empty result.")
    if executed == 0:
        return CriterionResult(
            3, "attribution_and_mechanism", FAIL,
            f"zero interventions executed ({fired} firings) -- any delta is not attributable", ev)

    if required_evidence_key not in t:
        return CriterionResult(
            3, "attribution_and_mechanism", PENDING,
            f"{executed} interventions executed but no {required_evidence_key!r} count -- firing "
            f"is not validation; a recovery is verified on what it PRESERVED", ev,
            remedy=f"record {required_evidence_key} from the controller's own per-step telemetry")

    verified = int(t.get(required_evidence_key, 0) or 0)
    ev[required_evidence_key] = verified
    if verified == 0:
        return CriterionResult(3, "attribution_and_mechanism", FAIL,
                               f"{executed} executions, 0 verified by the mechanism's own "
                               f"telemetry", ev)

    # Attribution completeness: every request must be accounted for, either verified or declined
    # with a recorded reason. Key ABSENCE is not a reason.
    requested = int(t.get(requested_key, 0) or 0)
    declined = int(t.get("mechanism_declined_with_reason", 0) or 0)
    refused = int(t.get("mechanism_refused_by_guard", 0) or 0)
    if requested:
        ev.update({requested_key: requested, "declined_with_reason": declined,
                   "refused_by_guard": refused})
        unattributed = requested - verified - declined - refused
        ev["unattributed"] = unattributed
        if unattributed > 0:
            return CriterionResult(
                3, "attribution_and_mechanism", PENDING,
                f"{requested} requested but only {verified + declined + refused} attributed "
                f"({verified} verified / {declined} declined / {refused} guard-refused); "
                f"{unattributed} unattributed", ev,
                remedy="classify the population exhaustively and check that it SUMS; a request "
                       "with no recorded reason is missing evidence, not safe behaviour")

    return CriterionResult(3, "attribution_and_mechanism", PASS,
                          f"{executed} executions, {verified} verified by the mechanism's own "
                          f"telemetry", ev)


# =================================================================================================
# criterion 4 -- safety: the three-counter contract
# =================================================================================================

def check_safety(safety: Mapping[str, Any] | None) -> CriterionResult:
    """clears_added > 0 FAILS. information_losing_removes > 0 FAILS. verified_relocations never fails.

    Absence of the two destructive counters is PENDING, not a pass: "no evidence of harm" and "no
    harm" are different claims, and only the second clears a safety criterion.
    """
    s = dict(safety or {})
    if not s:
        return CriterionResult(
            4, "safety_three_counters", PENDING,
            "no safety telemetry -- absence of evidence of harm is not evidence of no harm", {},
            remedy="record clears_added and information_losing_removes from LIVE state")

    missing = [k for k in ("clears_added", "information_losing_removes") if k not in s]
    if missing:
        return CriterionResult(
            4, "safety_three_counters", PENDING,
            f"missing required counter(s): {missing}", dict(s),
            remedy="both destructive counters must be measured against live state")

    clears = int(s.get("clears_added", 0) or 0)
    lossy = int(s.get("information_losing_removes", 0) or 0)
    relocs = int(s.get("verified_relocations", 0) or 0)
    ev = {"clears_added": clears, "information_losing_removes": lossy,
          "verified_relocations": relocs}

    problems = []
    if clears > 0:
        problems.append(f"{clears} clears added (categorically forbidden)")
    if lossy > 0:
        problems.append(f"{lossy} information-losing removes (a removal not verified present "
                        f"elsewhere in live state)")
    if problems:
        return CriterionResult(4, "safety_three_counters", FAIL, "; ".join(problems), ev)

    return CriterionResult(
        4, "safety_three_counters", PASS,
        f"0 clears added, 0 information-losing removes, {relocs} verified relocations "
        f"(reported, bounded, not destructive)", ev)


# =================================================================================================
# the seam
# =================================================================================================

def evaluate_criteria(evaluation, *, dev: Mapping[str, Any] | None = None,
                      safety: Mapping[str, Any] | None = None,
                      telemetry: Mapping[str, Any] | None = None) -> AcceptanceReport:
    """Adjudicate all four. `telemetry` defaults to the evaluation's own."""
    tel = telemetry if telemetry is not None else getattr(evaluation, "telemetry", {})
    return AcceptanceReport((
        check_train(evaluation),
        check_dev(dev),
        check_mechanism(tel),
        check_safety(safety),
    ))


def make_acceptance(*, dev_for: Callable[[Any], Mapping[str, Any] | None] | None = None,
                    safety_for: Callable[[Any], Mapping[str, Any] | None] | None = None,
                    on_report: Callable[[AcceptanceReport], None] | None = None,
                    ) -> Callable[[Any], tuple[bool, str]]:
    """Build the `accept(EvaluationResult) -> (bool, reason)` callable `self_evolve.step()` expects.

    `dev_for` / `safety_for` are looked up per evaluation so the dev arm and the live-state safety
    audit can be produced by the benchmark adapter. Supplying neither is legitimate and yields two
    PENDING_VALIDATION verdicts -- the round then rejects, which is the intended behaviour for a
    candidate whose dev and safety evidence was never gathered.
    """
    def accept(evaluation) -> tuple[bool, str]:
        report = evaluate_criteria(
            evaluation,
            dev=dev_for(evaluation) if dev_for else None,
            safety=safety_for(evaluation) if safety_for else None,
        )
        if on_report is not None:
            on_report(report)
        return report.accepted, report.reason()
    return accept
