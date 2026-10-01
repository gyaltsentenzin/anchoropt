"""Evaluating an intervention whose effect is DELAYED: it acts on state, and is scored later.

WHY THIS MODULE EXISTS
----------------------
A measured round passed all four acceptance criteria and was still uninterpretable. The controller
fired on 24 of 27 store-construction episodes and on 0 of 105 scored episodes; the +4 was measured
entirely in the scored phase. Every criterion was satisfied -- on telemetry from one phase -- while
the delta lived in another. `channel_integrity` caught it only after the fact, by asking for firings
on the gained cases and finding none.

The defect is not that the controller misbehaved. It behaved exactly as declared: it was mined from
storage episodes, declared `phase: prereq`, and the host correctly confined it there. The defect is
in the EVALUATION CONTRACT, which assumed the phase an intervention acts in is the phase its effect
is observed in. For any agent that changes state which later work depends on, that assumption is
false, and the resulting "0 firings on the gained case" is a measurement artifact rather than a
finding.

  acts at:      a state transition (a write, a migration, a config change, a cached artifact)
  scored at:    a later unit of work that READS the state that transition produced

THE GENERIC STRUCTURE, NOT ONE HOST'S SPECIAL CASE
-------------------------------------------------
Nothing below names a benchmark or a state-store technology. The contract needs three things from a
host,
and a host that cannot supply them gets an explicit UNSUPPORTED verdict rather than a wrong one:

  1. a PHASE label per observed unit (which units are state-building, which are scored);
  2. a declared DEPENDENCY GRAPH over unit ids -- who reads what was built by whom. Declared by the
     host, never inferred from id strings: a naming convention is not a dependency, and inferring one
     invents provenance the corpus does not assert;
  3. firing telemetry keyed by unit id.

Any host with those has this structure. A CI agent that edits a shared fixture consumed by later
tests; a build agent that populates a cache read by later compilations; a data agent that repairs a
table queried downstream; a config agent whose change takes effect on the next deploy. In every case
the intervention's evidence is not in the scored unit's own trace.

WHAT IT DOES, IN THE USER'S FOUR TERMS
--------------------------------------
  record the transition      `ActingSite` -- the phase and unit where the controller actually ran
  connect via the graph      `attribute_through_dependencies` -- credit a firing on the UPSTREAM
                             units a scored unit declares it depends on
  matched initial conditions `matched_initial_conditions` -- both arms must start from equivalent
                             state, while the intervention is ALLOWED to change downstream state
  verify where it executes   `phase_scoped_verification` -- route C3/C4 to the acting phase

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not weaken `channel_integrity`. The zero-firings check stays exactly as strict; what changes
is WHERE the firings are counted, from "the scored unit's own trace" to "the causal cone the host
itself declares." A gain whose upstream cone contains no firing is still unattributed, and this
module reports that as such. Two ways it can still fail:

  * a scored unit gains, its declared upstream cone was touched by nothing -> UNATTRIBUTED, as before;
  * the arm modified state that its own scored units are graded AGAINST, with no matched control
    construction -> CONFOUNDED, which is a validity failure and not a negative result.

The second is the trap the original round fell into, and it is reported separately from the first
because the remedies differ: the first needs a better mechanism, the second needs a better protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: The intervention's effect was observed in the phase it acted in. Ordinary, same-phase evaluation.
IMMEDIATE = "IMMEDIATE"
#: It acted in one phase and was scored in another, and the dependency graph connects the two.
DELAYED = "DELAYED"
#: It acted in one phase and was scored in another, and NOTHING connects them.
DISCONNECTED = "DISCONNECTED"
#: The host did not supply phases, a graph, or keyed firings. Not a verdict about the controller.
UNSUPPORTED = "UNSUPPORTED"
#: The controller declared a phase, the host CAN express phases, and that phase was never executed in
#: this run -- so the controller had no opportunity to act. Distinct from UNSUPPORTED (the host cannot
#: express delayed effects) and from DISCONNECTED (it acted, but nothing connects it to the score).
#:
#: This is the honest verdict for an evaluation that eliminated the acting phase in the course of
#: controlling for it. For an intervention that acts at a state transition, holding construction
#: constant and letting the controller act can be MUTUALLY EXCLUSIVE; when they are, the comparison
#: is the wrong instrument rather than a stricter one, and a net of 0 is the absence of a measurement
#: rather than a measured null. Reporting it as NO_OPPORTUNITY is what stops that 0 being banked.
NO_OPPORTUNITY = "NO_OPPORTUNITY"


@dataclass(frozen=True)
class ActingSite:
    """WHERE a controller actually ran: the phase, and the units whose state it changed.

    `phase` is the host's own label for the kind of work, `units` the ids it fired on, and
    `transitions` an optional per-unit note of what state moved. A controller that ran nowhere has
    empty `units`, which is a real and reportable outcome -- not missing data.

    `declared_phase` is what the controller's spec SAID. When it differs from the observed phase the
    controller ran somewhere it did not declare, which is a finding in its own right: the host's
    phase confinement and the spec disagree.
    """

    phase: str
    units: frozenset[str] = frozenset()
    transitions: Mapping[str, str] = field(default_factory=dict)
    declared_phase: str | None = None

    @property
    def fired(self) -> bool:
        return bool(self.units)

    @property
    def phase_as_declared(self) -> bool:
        """Did it run where its spec said it would? None declared means unconstrained, hence True."""
        return self.declared_phase is None or self.declared_phase in (self.phase, "any")

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase, "n_units": len(self.units), "units": sorted(self.units),
                "declared_phase": self.declared_phase, "phase_as_declared": self.phase_as_declared,
                "transitions": dict(self.transitions)}


def acting_site(firings_by_unit: Mapping[str, Any], phase_of: Mapping[str, str], *,
                declared_phase: str | None = None,
                transitions: Mapping[str, str] | None = None) -> ActingSite:
    """Reduce per-unit firing telemetry to the single phase the controller acted in.

    A controller is defined at one boundary and so acts in one phase; if telemetry says otherwise the
    MAJORITY phase is reported and the split is visible in `units` (callers comparing
    `len(units)` against a per-phase count will see the disagreement). Units with no phase label are
    labelled "unknown" rather than dropped -- silently discarding an unlabelled firing is how a
    controller comes to look inert.
    """
    fired = {str(u) for u, v in (firings_by_unit or {}).items() if v}
    if not fired:
        return ActingSite(phase=str(declared_phase or "unknown"), units=frozenset(),
                          declared_phase=declared_phase, transitions=dict(transitions or {}))
    counts: dict[str, int] = {}
    for u in fired:
        ph = str(phase_of.get(u, "unknown"))
        counts[ph] = counts.get(ph, 0) + 1
    # deterministic: most units first, then phase name, so a tie does not depend on dict order
    phase = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return ActingSite(phase=phase, units=frozenset(fired), declared_phase=declared_phase,
                      transitions=dict(transitions or {}))


# =================================================================================================
# connect the acting site to the scored units, THROUGH THE HOST'S DECLARED GRAPH
# =================================================================================================

def causal_cone(unit: str, graph: Mapping[str, Sequence[str]]) -> list[str]:
    """Transitive closure of what `unit` declares it depends on, nearest first. Cycle-safe.

    The graph is the host's assertion, read from its own corpus or manifest. An empty graph yields an
    empty cone, which makes every delayed attribution DISCONNECTED -- the correct answer for a host
    that never declared who reads whose output.
    """
    order: list[str] = []
    seen = {unit}
    frontier = list(graph.get(unit) or [])
    while frontier:
        nxt: list[str] = []
        for d in frontier:
            d = str(d)
            if d in seen:
                continue
            seen.add(d)
            order.append(d)
            nxt.extend(str(x) for x in (graph.get(d) or []))
        frontier = nxt
    return order


@dataclass(frozen=True)
class DelayedAttribution:
    """Per-unit attribution through the dependency graph, plus the aggregate verdict."""

    mode: str
    attributed: frozenset[str] = frozenset()
    unattributed: frozenset[str] = frozenset()
    cones: Mapping[str, list[str]] = field(default_factory=dict)
    detail: str = ""

    @property
    def n_attributed(self) -> int:
        return len(self.attributed)

    @property
    def fraction_attributed(self) -> float:
        total = len(self.attributed) + len(self.unattributed)
        return (len(self.attributed) / total) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "n_attributed": len(self.attributed),
                "n_unattributed": len(self.unattributed),
                "attributed": sorted(self.attributed), "unattributed": sorted(self.unattributed),
                "fraction_attributed": round(self.fraction_attributed, 4), "detail": self.detail,
                "cones": {k: list(v) for k, v in self.cones.items()}}


def attribute_through_dependencies(scored_units: Iterable[str], site: ActingSite,
                                   graph: Mapping[str, Sequence[str]]) -> DelayedAttribution:
    """Which scored units have a firing in their own declared causal cone?

    A scored unit counts as attributed when the controller fired on that unit itself (the immediate
    case) OR on any unit in its transitive `depends_on` closure (the delayed case). Anything else is
    unattributed, and stays unattributed: this widens WHERE evidence may be found, never what counts
    as evidence.
    """
    units = [str(u) for u in scored_units]
    if not site.fired:
        return DelayedAttribution(
            DISCONNECTED, unattributed=frozenset(units),
            detail="the controller fired on no unit in any phase, so nothing is attributable")
    if not graph:
        direct = {u for u in units if u in site.units}
        return DelayedAttribution(
            IMMEDIATE if direct else DISCONNECTED, attributed=frozenset(direct),
            unattributed=frozenset(u for u in units if u not in direct),
            detail=("no dependency graph declared, so only same-unit firings can be credited; "
                    f"{len(direct)} of {len(units)} scored units fired directly"))

    attributed, unattributed, cones, delayed_any = set(), set(), {}, False
    for u in units:
        cone = causal_cone(u, graph)
        hit_up = [c for c in cone if c in site.units]
        if u in site.units:
            attributed.add(u)
            cones[u] = []
        elif hit_up:
            attributed.add(u)
            cones[u] = hit_up
            delayed_any = True
        else:
            unattributed.add(u)
            cones[u] = []
    mode = DELAYED if delayed_any else (IMMEDIATE if attributed else DISCONNECTED)
    return DelayedAttribution(
        mode, frozenset(attributed), frozenset(unattributed), cones,
        detail=(f"{len(attributed)} of {len(units)} scored units have a firing in their declared "
                f"causal cone ({'delayed' if delayed_any else 'same-unit only'}); "
                f"{len(unattributed)} have none"))


# =================================================================================================
# matched initial conditions -- the check the original round actually failed
# =================================================================================================

@dataclass(frozen=True)
class MatchedConditions:
    """Did the two arms start from equivalent state, and was the state built comparably?

    Two distinct things, and conflating them is the confound:

      * `initial_equivalent` -- the two arms began from the same state. Required for a paired
        comparison to mean anything.
      * `construction_matched` -- the state-building phase was carried out under the same protocol
        in both arms. This is what the original round got wrong: the arm repaired the store during
        construction and the control never did, so the scored units were graded against DIFFERENT
        state -- and the delta measured the state, not the query-time mechanism.

    Downstream divergence is NOT a failure. An intervention that changes future state is supposed to
    change future state; forbidding that would forbid the whole class of controller this module
    exists for. What must match is where the arms START and how construction is CARRIED OUT, not
    where they end up.
    """

    initial_equivalent: bool
    construction_matched: bool
    initial_fingerprints: Mapping[str, str] = field(default_factory=dict)
    detail: str = ""
    remedy: str = ""

    @property
    def ok(self) -> bool:
        return self.initial_equivalent and self.construction_matched

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "initial_equivalent": self.initial_equivalent,
                "construction_matched": self.construction_matched,
                "initial_fingerprints": dict(self.initial_fingerprints), "detail": self.detail,
                "remedy": self.remedy}


def matched_initial_conditions(arm: Mapping[str, Any], control: Mapping[str, Any], *,
                               fingerprint_key: str = "initial_state_fingerprint",
                               construction_key: str = "construction_protocol") -> MatchedConditions:
    """Compare two arms' declared starting state and construction protocol.

    Absent fingerprints are NOT treated as matching. "Neither arm reported its initial state" and
    "both arms reported the same initial state" are different claims, and reading the first as the
    second is how an unmatched pair passes for a matched one.

    THE TWO FLAGS ARE MEASURED INDEPENDENTLY, and an earlier version of this function got that
    wrong: a differing fingerprint returned early with `construction_matched=False`, never comparing
    the protocols at all. That conflation matters because the two failures have different remedies --
    an unmatched start needs a shared initial state, an unmatched construction needs a shared
    building phase -- and because `contamination_free` reads ONLY `construction_matched`. Forcing it
    False on a fingerprint mismatch reports a construction confound that was not measured, which is
    the mirror image of the defect this module exists to prevent: it manufactures an invalidity
    rather than hiding one, and either way the reported reason is not the measured one.
    """
    a_fp, c_fp = arm.get(fingerprint_key), control.get(fingerprint_key)
    a_proto, c_proto = arm.get(construction_key), control.get(construction_key)
    fps = {"arm": str(a_fp), "control": str(c_fp)}

    # --- the two questions, asked separately -------------------------------------------------
    if a_fp is None or c_fp is None:
        init_ok = False
        init_detail = (
            f"missing {fingerprint_key!r} on "
            f"{'both arms' if a_fp is None and c_fp is None else ('the arm' if a_fp is None else 'the control')}"
            f" -- an unreported initial state is not a matched one")
        init_remedy = f"have each arm record {fingerprint_key} for the state it started from"
    elif str(a_fp) != str(c_fp):
        init_ok = False
        init_detail = (f"arms started from DIFFERENT initial state ({a_fp} vs {c_fp}); the paired "
                       f"delta cannot separate the intervention from the starting condition")
        init_remedy = "re-run both arms from one shared initial state"
    else:
        init_ok, init_detail, init_remedy = True, f"both arms started from {a_fp}", ""

    if a_proto is None or c_proto is None:
        cons_ok = False
        cons_detail = (f"{construction_key!r} is unreported, so it is unknown whether the state the "
                       f"scored units are graded against was built the same way in both arms")
        cons_remedy = f"have each arm record {construction_key} for the state-building phase"
    elif str(a_proto) != str(c_proto):
        cons_ok = False
        cons_detail = (f"construction protocol DIFFERS ({a_proto} vs {c_proto}): the scored units "
                       f"are graded against state built differently in each arm, so the delta is "
                       f"confounded with construction")
        cons_remedy = ("build the state under one protocol in both arms, then intervene; or score "
                       "the intervention in the phase it acts in")
    else:
        cons_ok = True
        cons_detail = f"both arms built state under {a_proto}"
        cons_remedy = ""

    if init_ok and cons_ok:
        return MatchedConditions(
            True, True, fps,
            detail=f"both arms started from {a_fp} and built state under {a_proto}; downstream "
                   f"divergence is expected and permitted")
    parts = [d for d, ok in ((init_detail, init_ok), (cons_detail, cons_ok)) if not ok]
    return MatchedConditions(
        init_ok, cons_ok, fps, detail="; ".join(parts),
        remedy="; ".join(r for r in (init_remedy, cons_remedy) if r))


# =================================================================================================
# verify the mechanism IN THE PHASE THE CONTROLLER EXECUTES
# =================================================================================================

def phase_scoped_verification(telemetry_by_phase: Mapping[str, Mapping[str, Any]],
                              site: ActingSite) -> tuple[dict[str, Any], str]:
    """Pick the telemetry slice C3/C4 must be judged on: the acting phase's, not the scored phase's.

    Returns `(telemetry, note)`. When the acting phase has no slice the return is an EMPTY mapping,
    not a merged or substituted one -- `check_mechanism({})` then yields PENDING_VALIDATION, which is
    the honest verdict for "the controller ran somewhere we did not instrument." Quietly falling back
    to another phase's telemetry is exactly the substitution that produced the original defect.
    """
    if not site.fired:
        return {}, ("the controller fired in no phase; there is no acting phase to verify in, so "
                    "mechanism evidence is absent rather than negative")
    slice_ = telemetry_by_phase.get(site.phase)
    if slice_ is None:
        return {}, (f"the controller acted in phase {site.phase!r} but no telemetry was recorded for "
                    f"that phase (recorded: {sorted(telemetry_by_phase)}); verification is PENDING, "
                    f"and another phase's telemetry must NOT be substituted")
    return dict(slice_), (f"mechanism and safety verified on phase {site.phase!r} telemetry -- the "
                          f"phase the controller actually executed in")


# =================================================================================================
# the whole contract, in one call
# =================================================================================================

@dataclass(frozen=True)
class DelayedEffectReport:
    """Everything the acceptance layer needs to judge a delayed-effect intervention."""

    site: ActingSite
    attribution: DelayedAttribution
    conditions: MatchedConditions
    verification_telemetry: Mapping[str, Any]
    verification_note: str
    supported: bool = True
    #: Phases the host reports as actually executed in this run. Empty means the host did not say,
    #: in which case no opportunity claim is made either way.
    phases_executed: frozenset[str] = frozenset()

    @property
    def had_opportunity(self) -> bool:
        """Did the run execute the phase the controller declared it acts in?

        True when it fired, or when the declared phase ran, or when the host reported nothing about
        which phases ran. False ONLY when the host affirmatively reports that the declared acting
        phase did not execute: absence of a report is not evidence of absence of opportunity.
        """
        if self.site.fired or not self.phases_executed:
            return True
        declared = self.site.declared_phase
        if declared is None or declared == "any":
            return True
        return declared in self.phases_executed

    @property
    def channel_ok(self) -> bool:
        """Is there a causal channel from where it acted to where it was scored?

        True when at least one scored unit of interest has a firing in its declared cone. This is the
        SAME question `channel_integrity` asks, evaluated over the cone instead of the bare unit.
        """
        return self.attribution.n_attributed > 0

    @property
    def contamination_free(self) -> bool:
        """Was the state the scored units are graded against built comparably in both arms?"""
        return self.conditions.construction_matched

    @property
    def interpretable(self) -> bool:
        return bool(self.supported and self.channel_ok and self.conditions.ok)

    def to_dict(self) -> dict[str, Any]:
        return {"supported": self.supported, "interpretable": self.interpretable,
                "had_opportunity": self.had_opportunity,
                "phases_executed": sorted(self.phases_executed),
                "channel_ok": self.channel_ok, "contamination_free": self.contamination_free,
                "acting_site": self.site.to_dict(), "attribution": self.attribution.to_dict(),
                "conditions": self.conditions.to_dict(),
                "verification_note": self.verification_note,
                "verification_phase_telemetry_keys": sorted(self.verification_telemetry)}


def evaluate_delayed_effect(*, gained_units: Iterable[str],
                            firings_by_unit: Mapping[str, Any],
                            phase_of: Mapping[str, str],
                            graph: Mapping[str, Sequence[str]],
                            telemetry_by_phase: Mapping[str, Mapping[str, Any]],
                            arm_conditions: Mapping[str, Any],
                            control_conditions: Mapping[str, Any],
                            declared_phase: str | None = None,
                            transitions: Mapping[str, str] | None = None,
                            phases_executed: Iterable[str] | None = None) -> DelayedEffectReport:
    """The full contract. Genuinely host-agnostic: unit ids are opaque strings throughout.

    A host that supplies no phase labels and no graph gets `supported=False` and an UNSUPPORTED
    attribution mode, so the caller can tell "this host cannot express delayed effects" apart from
    "this controller has no causal channel."

    `phases_executed` is the optional list of phases the run actually ran. Supplying it lets the
    contract separate a THIRD case from those two: the host can express delayed effects, and the
    controller declared a phase that this run never executed, so it had no opportunity to act. That
    is NO_OPPORTUNITY. It matters because an evaluation can eliminate the acting phase in the very
    act of controlling for it -- and the resulting zero delta looks exactly like a measured null
    while being the absence of a measurement.
    """
    site = acting_site(firings_by_unit, phase_of, declared_phase=declared_phase,
                       transitions=transitions)
    supported = bool(phase_of) and bool(graph)
    attribution = attribute_through_dependencies(gained_units, site, graph)
    if not supported:
        attribution = DelayedAttribution(
            UNSUPPORTED, attribution.attributed, attribution.unattributed, attribution.cones,
            detail="host declared "
                   + ("no phase labels" if not phase_of else "phase labels")
                   + " and "
                   + ("no dependency graph" if not graph else "a dependency graph")
                   + "; delayed effects cannot be evaluated, which is a statement about the host, "
                     "not about the controller. " + attribution.detail)
    executed = frozenset(str(p) for p in (phases_executed or ()))
    # A declared acting phase the run never executed is reported as such, and takes precedence over
    # UNSUPPORTED/DISCONNECTED: those describe a controller that had its chance, and this one did not.
    if (executed and not site.fired and declared_phase not in (None, "any")
            and str(declared_phase) not in executed):
        attribution = DelayedAttribution(
            NO_OPPORTUNITY, attribution.attributed, attribution.unattributed, attribution.cones,
            detail=f"controller declared it acts in phase {declared_phase!r}, which this run did not "
                   f"execute (executed: {sorted(executed)}); it never had the opportunity to fire, so "
                   f"any delta measured here is not evidence about the controller. " +
                   attribution.detail)
    conditions = matched_initial_conditions(arm_conditions, control_conditions)
    tel, note = phase_scoped_verification(telemetry_by_phase, site)
    return DelayedEffectReport(site, attribution, conditions, tel, note, supported, executed)
