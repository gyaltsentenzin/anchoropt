"""The attributor and proposer seams: two SEPARATE roles, with the separation enforced structurally.

THE ARCHITECTURAL IDEA, BORROWED DELIBERATELY
--------------------------------------------
An earlier internal harness separates the model doing ERROR ATTRIBUTION from the model doing
INTERVENTION PROPOSAL. Both may be the same vendor -- in this project both are Claude, in separate
prompts and separate contexts, while the model under optimization is Granite. What matters is that
neither role can quietly do the other's job.

    ATTRIBUTOR   why did this episode fail, and which decision must change
                 -> mechanism, evidence, causal REGION, consequential decision
    PROPOSER     given the residual and the runtime's capabilities, which controllers to try
                 -> a small set of structured (l, phi, mu, theta)
    ANCHOROPT    which of those is a LEGAL controller, and which one wins on measurement

WHY THE SEPARATION IS ENFORCED HERE RATHER THAN REQUESTED IN A PROMPT
--------------------------------------------------------------------
An attributor that also names the decision point has already made AnchorOpt's decision, and it will
make it with the habits of whatever search space it knows best -- a prompt-heavy miner biases
(l, phi, mu, theta) toward prompt edits exactly because that is the search space it knows. A prompt
asking it not to is not a constraint; a schema that has no field for it is.

So `ingest_attribution` REJECTS a record carrying an incision point, a signal name, an action, or a
support/loss count. The first three are AnchorOpt's decisions. The fourth is a measurement, and a
number produced by a language model is not one -- support and linked downstream loss are COMPUTED
from traces (`compute_support`), never accepted from a proposal. A model that invents a plausible
support figure is indistinguishable from one that measured it, which is how an unattributable gain
becomes a published result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.runtime import ResidualDiagnosis
from anchoropt.learning.signal_lang import CompiledSignal, SignalSpecError, compile_signal

# Keys an attribution record may NOT carry, and what each would usurp.
_FORBIDDEN_ATTRIBUTOR_KEYS: Mapping[str, str] = {
    "boundary": "the decision point is AnchorOpt's choice, not the attributor's",
    "incision_point": "the decision point is AnchorOpt's choice, not the attributor's",
    "locus": "the decision point is AnchorOpt's choice, not the attributor's",
    "signal": "phi is selected by the search from the declared vocabulary",
    "phi": "phi is selected by the search from the declared vocabulary",
    "action": "mu is selected from the closed semantic action vocabulary",
    "mu": "mu is selected from the closed semantic action vocabulary",
    "theta": "theta is grounded against the runtime, not proposed as prose",
    "support": "support is COMPUTED from traces, never accepted from a proposal",
    "linked_downstream_loss": "linked loss is COMPUTED from traces, never proposed",
    "n_cases": "counts are computed from traces",
    "confidence": "a model's self-reported confidence is not a measurement",
}

_REQUIRED_ATTRIBUTOR_KEYS = ("case_id", "failure_mechanism", "evidence",
                             "consequential_decision", "causal_region")


class AttributionSchemaError(ValueError):
    """An attribution record that oversteps the attributor's role, or is incomplete."""


class ProposalSchemaError(ValueError):
    """A proposal that is not a well-formed candidate. Recorded, so a decline is attributable."""


# ================================================================================================
# 1. the attributor seam
# ================================================================================================

@dataclass(frozen=True)
class CausalRegion:
    """WHERE the failure happened, in the attributor's terms -- deliberately not an IncisionPoint.

    `phase` is prose ("after the retrieve returned", "before it answered"). Mapping a region onto
    candidate boundaries is `candidate_search.candidate_loci()`'s job, and it keeps all three open
    on purpose: a diagnosis localizes a REGION, and three different boundaries can address a
    step-k failure. Letting the attributor emit an IncisionPoint collapses that distinction at the
    one place the architecture exists to keep it open.
    """

    step: int | None = None
    phase: str = ""
    call: str = ""


def ingest_attribution(record: Mapping[str, Any], *, provider: str) -> ResidualDiagnosis:
    """Validate one attributor record and convert it to a `ResidualDiagnosis`.

    Raises `AttributionSchemaError` if the record is incomplete or oversteps. The rejection is not
    politeness -- see the module docstring for what each forbidden key would usurp.
    """
    overstep = sorted(k for k in record if k in _FORBIDDEN_ATTRIBUTOR_KEYS)
    if overstep:
        why = "; ".join(f"{k}: {_FORBIDDEN_ATTRIBUTOR_KEYS[k]}" for k in overstep)
        raise AttributionSchemaError(
            f"attribution for {record.get('case_id', '?')!r} carries key(s) the attributor must "
            f"not decide -- {why}")
    missing = [k for k in _REQUIRED_ATTRIBUTOR_KEYS if k not in record or record[k] in ("", None)]
    if missing:
        raise AttributionSchemaError(
            f"attribution for {record.get('case_id', '?')!r} missing {missing}")

    region = record["causal_region"]
    if isinstance(region, Mapping):
        region = CausalRegion(step=region.get("step"), phase=str(region.get("phase", "")),
                             call=str(region.get("call", "")))
    evidence = record["evidence"]
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
        evidence_text = " | ".join(
            f"step {e.get('step')}: {e.get('call', '')} -> {str(e.get('result', ''))[:160]}"
            if isinstance(e, Mapping) else str(e) for e in evidence)
    else:
        evidence_text = str(evidence)

    return ResidualDiagnosis(
        case_id=str(record["case_id"]),
        mechanism=str(record["failure_mechanism"]),
        evidence=evidence_text,
        consequential_decision=str(record["consequential_decision"]),
        proposed_behavior_change=str(record.get("proposed_behavior_change", "")),
        provider=provider,
        metadata={"causal_region": {"step": region.step, "phase": region.phase,
                                    "call": region.call},
                  "raw_evidence": record["evidence"]},
    )


def compute_support(diagnoses: Sequence[ResidualDiagnosis], *,
                    trace_lookup: Callable[[str], Mapping[str, Any]] | None = None
                    ) -> Mapping[str, int]:
    """Support per consequential decision, COMPUTED by grouping diagnoses -- never proposed.

    v0.1 counts diagnoses sharing a normalized `consequential_decision`. That is a real count over
    real episodes, which is the property that matters: it cannot be inflated by a model asserting a
    number. `trace_lookup` is accepted for the linked-downstream-loss version, which needs the
    traces themselves and is deliberately not faked here -- an unimplemented measurement must look
    unimplemented rather than return a plausible integer.
    """
    counts: dict[str, int] = {}
    for d in diagnoses:
        key = " ".join(str(d.consequential_decision).lower().split())
        counts[key] = counts.get(key, 0) + 1
    return counts


def rank_by_support(diagnoses: Sequence[ResidualDiagnosis]) -> tuple[ResidualDiagnosis, ...]:
    """Order diagnoses by computed support, descending; stable within a tier by case_id.

    Ranking by linked downstream loss rather than event volume is A1's finding -- the most FREQUENT
    error is often a symptom, and ranking by frequency would have sent the first anchor elsewhere.
    v0.1 ranks by grouped support because that is what it can compute honestly; the docstring says
    so rather than implying the stronger measure.
    """
    support = compute_support(diagnoses)

    def key(d: ResidualDiagnosis):
        k = " ".join(str(d.consequential_decision).lower().split())
        return (-support.get(k, 0), d.case_id)

    return tuple(sorted(diagnoses, key=key))


# ================================================================================================
# 2. the proposer seam
# ================================================================================================

@dataclass(frozen=True)
class ProposedCandidate:
    """A structured proposal, BEFORE AnchorOpt decides whether it is a legal controller."""

    boundary: IncisionPoint
    signal_name: str
    action: Action
    theta: Mapping[str, Any] = field(default_factory=dict)
    signal_expr: Mapping[str, Any] | None = None      # set only in the SIGNAL block
    signal_params: Mapping[str, Any] = field(default_factory=dict)
    diagnosis_case_ids: tuple[str, ...] = ()
    rationale: str = ""


def ingest_proposal(record: Mapping[str, Any], *, allow_new_signal: bool,
                    declared_signals: Sequence[str]) -> ProposedCandidate:
    """Validate one proposer record into a `ProposedCandidate`.

    `allow_new_signal` is the POLICY/SIGNAL block switch as seen from here, and it is passed IN by
    AnchorOpt rather than chosen by the proposer:

        POLICY block  allow_new_signal=False -- phi must already be in the declared vocabulary, and
                      a record carrying `signal_expr` is REJECTED. Phi is frozen; only (l, mu, theta)
                      are in play.
        SIGNAL block  allow_new_signal=True  -- a new declarative expression may be authored, and it
                      still has to compile and pass the expansion screen before it joins Phi.

    Collapsing the two -- letting every proposal optionally invent a signal -- would dissolve the
    block-coordinate structure into one unconstrained call, which is the thing the method is built
    to avoid.
    """
    for key in ("boundary", "action"):
        if key not in record:
            raise ProposalSchemaError(f"proposal missing {key!r}")
    try:
        boundary = IncisionPoint(str(record["boundary"]))
    except ValueError as exc:
        raise ProposalSchemaError(
            f"unknown decision point {record['boundary']!r}; the three legal points are "
            f"{[p.value for p in IncisionPoint]}") from exc
    try:
        action = Action(str(record["action"]))
    except ValueError as exc:
        raise ProposalSchemaError(
            f"action {record['action']!r} is outside the closed semantic vocabulary "
            f"{[a.value for a in Action]}") from exc

    expr = record.get("signal_expr") or (record.get("signal") or {}).get("expr") \
        if isinstance(record.get("signal"), Mapping) else record.get("signal_expr")
    name = record.get("signal_name") or (
        record["signal"].get("name") if isinstance(record.get("signal"), Mapping) else record.get("signal"))
    if not name:
        raise ProposalSchemaError("proposal names no signal")
    name = str(name)

    if expr is not None and not allow_new_signal:
        raise ProposalSchemaError(
            f"proposal authors a new signal {name!r} during the POLICY block, where Phi is frozen; "
            f"a new phi may only be proposed in the SIGNAL block")
    if expr is None and name not in set(declared_signals):
        raise ProposalSchemaError(
            f"signal {name!r} is not in the declared vocabulary {sorted(declared_signals)} and no "
            f"expression was supplied")

    return ProposedCandidate(
        boundary=boundary, signal_name=name, action=action,
        theta=dict(record.get("theta") or {}),
        signal_expr=dict(expr) if expr else None,
        signal_params=dict(record.get("signal_params") or {}),
        diagnosis_case_ids=tuple(str(c) for c in (record.get("diagnosis_case_ids") or ())),
        rationale=str(record.get("rationale", "")))


def validate_proposal(candidate: ProposedCandidate, *, runtime, host, fields: Mapping[str, Any]
                      ) -> tuple[CompiledSignal | None, str]:
    """AnchorOpt's legality check. Returns (compiled_signal_or_None, reason).

    Five checks, each with its own reason so a decline is attributable rather than merely counted.
    Claude proposes; this function decides whether the proposal is a controller:

        action_not_executable   U_H(l) -- the operational check `admissible()` cannot make
        signal_does_not_compile the expression is not legal over the declared alphabet
        signal_not_observable   phi cannot be observed at the proposed l
        theta_not_grounded      a reroute destination that is not a real tool, or bad arg names
        no_diagnosis_linked     the proposal explains no episode, so engagement cannot be checked
    """
    try:
        host.require(candidate.boundary, candidate.action)
    except Exception as exc:
        return None, f"action_not_executable: {exc}"

    compiled: CompiledSignal | None = None
    if candidate.signal_expr is not None:
        try:
            compiled = compile_signal(candidate.signal_name, candidate.signal_expr, fields=fields)
        except SignalSpecError as exc:
            return None, f"signal_does_not_compile: {exc}"
        if not compiled.observable_at(candidate.boundary.value):
            return None, (f"signal_not_observable: {candidate.signal_name!r} is observable at "
                          f"{sorted(compiled.boundaries)}, not at {candidate.boundary.value}")
        missing = compiled.params_used - set(candidate.signal_params)
        if missing:
            return None, f"signal_params_unconfigured: missing {sorted(missing)}"
    else:
        try:
            runtime.evaluate_signal(candidate.signal_name,
                                    {"boundary": candidate.boundary}, candidate.signal_params)
        except KeyError as exc:
            return None, f"signal_not_observable: {exc}"
        except (ValueError, TypeError) as exc:
            return None, f"signal_params_unconfigured: {exc}"

    ok, why = ground_theta(candidate.action, candidate.theta, runtime=runtime)
    if not ok:
        return None, f"theta_not_grounded: {why}"
    if not candidate.diagnosis_case_ids:
        return None, ("no_diagnosis_linked: the proposal names no episode it explains, so its "
                      "engagement cannot be checked against the residual")
    return compiled, "legal"


def ground_theta(action: Action, theta: Mapping[str, Any], *, runtime) -> tuple[bool, str]:
    """Are this action's parameters real, against the runtime's own schema?

    REROUTE is the case that matters. docs/GENERALIZABILITY.md: "reroute must name a real
    destination tool with real argument names". A2's near-miss is why -- emitting `key=` universally
    would have built an invalid call and been misscored as "the substitute did not help", i.e. a
    measurement error dressed as a negative result.
    """
    if action is Action.NOOP:
        return True, "control arm"
    if action is Action.REPROMPT:
        return (bool(str(theta.get("text", "")).strip()),
                "REPROMPT requires non-empty theta['text']")
    if action is Action.SUPPRESS:
        return (bool(str(theta.get("reason", "")).strip()),
                "SUPPRESS requires non-empty theta['reason']")
    if action is Action.REROUTE:
        dest = str(theta.get("destination", "")).strip()
        if not dest:
            return False, "REROUTE requires theta['destination']"
        if hasattr(runtime, "is_known_tool") and not runtime.is_known_tool(dest):
            return False, f"destination {dest!r} is not a tool this runtime exposes"
        patch = theta.get("args_patch") or {}
        if hasattr(runtime, "unknown_args"):
            bad = runtime.unknown_args(dest, tuple(patch))
            if bad:
                return False, f"destination {dest!r} does not accept argument(s) {list(bad)}"
        return True, "grounded"
    return False, f"unhandled action {action!r}"
