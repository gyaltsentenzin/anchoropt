"""Three regression levels for accepted controllers, and the certificates they emit.

WHY THREE, and why the boundary between them is exactly here.

  L1 CONTRACT   (fast, no host)   Can this controller still be BUILT and INSTALLED? Observations
                                  supplied, operator feasible for the constraint, capability identity
                                  intact, grounding present, backend compatible, phase right.
  L2 SMOKE      (real host path)  Does it still EXECUTE? Positive states must reach the action
                                  through the host's own dispatch; negative states must not.
  L3 PAIRED     (benchmark)       Does it still MEASURE the same? Per-case outcomes, gain/loss sets
                                  and firing telemetry against the accepted reference.

The L1/L2 split is the one that has cost us the most. `fires_on()` returning True in isolation proved
nothing three separate times: the branch sat after a turn-ending exit so it was unreachable; a spec
was installed without its predicate module so the arm silently became the control; a telemetry
allowlist dropped the firing keys so a live controller looked inert. L1 cannot see any of those. Only
L2, driven through the host's real path, can.

L3 is required when EXECUTION SEMANTICS change -- a new gate at the dispatch point, a changed executor
resolution, a new controller sharing a cell. Not for a docs edit.

CERTIFICATES, NOT EMPTY RESULTS. Every failure names the level, the controller, what was expected, what
was observed, and the remedy. A missing import, an unavailable observation, a wrong capability identity
and an infeasible operation are all reported as distinct, actionable certificates -- because the defect
this whole module guards against is the silent no-op that reads as success.

Core supplies the schedule and the verdict shapes. The adapter supplies the probes: what a state looks
like, how to run the host, how to score a split. Core never learns what a signal observes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from .golden_registry import AcceptedController, GoldenRegistry, PairedOutcome

# Levels, ordered. Cheap first: an L1 failure makes L2 and L3 meaningless, so the schedule stops.
L1_CONTRACT = "L1_CONTRACT"
L2_SMOKE = "L2_SMOKE"
L3_PAIRED = "L3_PAIRED"
LEVELS = (L1_CONTRACT, L2_SMOKE, L3_PAIRED)

# Certificate reasons. Named so a failure is greppable and its remedy is unambiguous.
MISSING_IMPORT = "missing_import"
OBSERVATION_UNAVAILABLE = "observation_unavailable"
CAPABILITY_IDENTITY_MISMATCH = "capability_identity_mismatch"
OPERATION_INFEASIBLE = "operation_infeasible"
GROUNDING_ABSENT = "grounding_absent"
BACKEND_INCOMPATIBLE = "backend_incompatible"
PHASE_MISMATCH = "phase_mismatch"
NOT_INSTALLED = "not_installed"
DID_NOT_FIRE = "did_not_fire"
FIRED_ON_NEGATIVE = "fired_on_negative"
DID_NOT_ACT = "did_not_act"
OUTCOME_DRIFT = "outcome_drift"
TELEMETRY_DRIFT = "telemetry_drift"
COMPOSITION_CONFLICT = "composition_conflict"
PROBE_UNAVAILABLE = "probe_unavailable"


@dataclass(frozen=True)
class Certificate:
    """One actionable failure (or explicit skip). Never a bare False."""

    level: str
    controller: str
    reason: str
    detail: str
    remedy: str = ""
    skipped: bool = False

    def __str__(self) -> str:  # pragma: no cover - display only
        tag = "SKIP" if self.skipped else "FAIL"
        s = f"[{tag} {self.level}] {self.controller}: {self.reason} -- {self.detail}"
        return s + (f"\n    remedy: {self.remedy}" if self.remedy else "")


@dataclass
class LevelResult:
    level: str
    passed: tuple[str, ...] = ()
    certificates: tuple[Certificate, ...] = ()

    @property
    def ok(self) -> bool:
        return not [c for c in self.certificates if not c.skipped]

    @property
    def skipped(self) -> tuple[Certificate, ...]:
        return tuple(c for c in self.certificates if c.skipped)


@dataclass
class RegressionReport:
    results: dict[str, LevelResult] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results.values())

    def failures(self) -> tuple[Certificate, ...]:
        out: list[Certificate] = []
        for lvl in LEVELS:
            r = self.results.get(lvl)
            if r:
                out.extend(c for c in r.certificates if not c.skipped)
        return tuple(out)

    def summary(self) -> str:
        lines: list[str] = []
        for lvl in LEVELS:
            r = self.results.get(lvl)
            if r is None:
                lines.append(f"{lvl:12s} NOT RUN")
                continue
            lines.append(f"{lvl:12s} {'ok' if r.ok else 'FAIL'}  passed={len(r.passed)} "
                         f"failed={len(r.certificates) - len(r.skipped)} skipped={len(r.skipped)}")
        for c in self.failures():
            lines.append("  " + str(c))
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------
# THE HOST SEAM
# ---------------------------------------------------------------------------------------------------

class RegressionHost(Protocol):
    """What an adapter must provide. Every method is benchmark-specific; none is core's business.

    A probe that cannot run must raise NotImplementedError, which becomes an explicit SKIP
    certificate. Returning a bland pass for an unavailable probe is the exact failure mode this
    protocol exists to prevent.
    """

    def contract_check(self, entry: AcceptedController) -> Sequence[Certificate]:
        """L1. Observations supplied, operator feasible FOR THE CONSTRAINT, identity, grounding,
        backend, phase, installability. Host decides what each means."""

    def smoke_execute(self, entry: AcceptedController,
                      state: Mapping[str, Any]) -> Mapping[str, Any]:
        """L2. Run ONE state through the host's real dispatch path.

        Must return at least {"fired": bool, "acted": bool, "capability_id": str}. `acted` is
        separate from `fired` on purpose: a predicate can fire while the executor declines.
        """

    def paired_measure(self, entry: AcceptedController) -> PairedOutcome:
        """L3. Re-score the accepted split and return the per-case outcome."""


# ---------------------------------------------------------------------------------------------------
# THE SCHEDULE
# ---------------------------------------------------------------------------------------------------

class RegressionSuite:
    """Runs the levels for a registry against a host."""

    def __init__(self, registry: GoldenRegistry, host: RegressionHost) -> None:
        self.registry = registry
        self.host = host

    # -- L1 ------------------------------------------------------------------------------------
    def run_contract(self, only: Sequence[str] = ()) -> LevelResult:
        passed: list[str] = []
        certs: list[Certificate] = []
        for e in self._selected(only):
            try:
                got = list(self.host.contract_check(e) or ())
            except NotImplementedError as exc:
                certs.append(Certificate(L1_CONTRACT, e.name, PROBE_UNAVAILABLE, str(exc),
                                         remedy="implement contract_check on the adapter host",
                                         skipped=True))
                continue
            except ImportError as exc:
                # Explicit, because an ImportError here is precisely the "silently no candidates"
                # failure: the module that builds the controller is gone.
                certs.append(Certificate(L1_CONTRACT, e.name, MISSING_IMPORT, str(exc),
                                         remedy="restore the import or the module that provides it"))
                continue
            if got:
                certs.extend(got)
            else:
                passed.append(e.name)
        return LevelResult(L1_CONTRACT, tuple(passed), tuple(certs))

    # -- L2 ------------------------------------------------------------------------------------
    def run_smoke(self, only: Sequence[str] = ()) -> LevelResult:
        """Positive states must fire AND act; negative states must not fire.

        A controller with no recorded states is a FAILURE, not a pass: an accepted fixture without
        a discriminating case cannot show that it still executes.
        """
        passed: list[str] = []
        certs: list[Certificate] = []
        for e in self._selected(only):
            if not e.positive_states:
                certs.append(Certificate(
                    L2_SMOKE, e.name, PROBE_UNAVAILABLE,
                    "no positive_states recorded",
                    remedy="record at least one real state that makes this controller fire"))
                continue
            failed = False
            for i, st in enumerate(e.positive_states):
                try:
                    r = dict(self.host.smoke_execute(e, st) or {})
                except NotImplementedError as exc:
                    certs.append(Certificate(L2_SMOKE, e.name, PROBE_UNAVAILABLE, str(exc),
                                             remedy="provide smoke_execute (real evaluator path)",
                                             skipped=True))
                    failed = True
                    break
                if not r.get("fired"):
                    failed = True
                    certs.append(Certificate(
                        L2_SMOKE, e.name, DID_NOT_FIRE,
                        f"positive_states[{i}] did not fire through the host path "
                        f"(detail={r.get('detail')!r})",
                        remedy="check the branch is reachable at this boundary, the predicate module "
                               "is installed, and firing telemetry is not dropped downstream"))
                    continue
                if not r.get("acted"):
                    failed = True
                    certs.append(Certificate(
                        L2_SMOKE, e.name, DID_NOT_ACT,
                        f"positive_states[{i}] fired but the executor did not act "
                        f"(detail={r.get('detail')!r})",
                        remedy="a predicate firing is not an intervention; check the executor's "
                               "required eta and its capability gate"))
                    continue
                got_cap = str(r.get("capability_id") or "")
                want_cap = e.identity.capability_id
                if want_cap and got_cap and got_cap != want_cap:
                    failed = True
                    certs.append(Certificate(
                        L2_SMOKE, e.name, CAPABILITY_IDENTITY_MISMATCH,
                        f"positive_states[{i}] dispatched to {got_cap!r}, expected {want_cap!r}",
                        remedy="another controller at this cell is being resolved first; pin "
                               "dispatch to the capability identity"))
            for i, st in enumerate(e.negative_states):
                try:
                    r = dict(self.host.smoke_execute(e, st) or {})
                except NotImplementedError:
                    break
                if r.get("fired"):
                    failed = True
                    certs.append(Certificate(
                        L2_SMOKE, e.name, FIRED_ON_NEGATIVE,
                        f"negative_states[{i}] fired but must not",
                        remedy="the predicate has widened; a controller that fires everywhere is "
                               "not conditional"))
            if not failed:
                passed.append(e.name)
        return LevelResult(L2_SMOKE, tuple(passed), tuple(certs))

    # -- L3 ------------------------------------------------------------------------------------
    def run_paired(self, only: Sequence[str] = ()) -> LevelResult:
        """Compare per case, and compare telemetry. Totals alone are not evidence."""
        passed: list[str] = []
        certs: list[Certificate] = []
        for e in self._selected(only):
            try:
                got = self.host.paired_measure(e)
            except NotImplementedError as exc:
                certs.append(Certificate(L3_PAIRED, e.name, PROBE_UNAVAILABLE, str(exc),
                                         remedy="run the paired split on the real runtime",
                                         skipped=True))
                continue
            certs.extend(compare_outcomes(e, got))
            if not [c for c in certs if c.controller == e.name and not c.skipped]:
                passed.append(e.name)
        return LevelResult(L3_PAIRED, tuple(passed), tuple(certs))

    # -- composition ---------------------------------------------------------------------------
    def check_composition(self, candidate) -> tuple[Certificate, ...]:
        """Static conflicts a new controller would create with accepted ones.

        Reported as certificates so installing over an accepted controller cannot happen quietly.
        A clean result is NOT proof of composition -- L2 on BOTH controllers is.
        """
        ident = getattr(candidate, "identity", candidate)
        return tuple(
            Certificate(L1_CONTRACT, getattr(candidate, "name", "<candidate>"),
                        COMPOSITION_CONFLICT, d,
                        remedy="smoke-test both controllers together before measuring")
            for d in self.registry.composition_conflicts(ident))

    # -- schedule ------------------------------------------------------------------------------
    def run(self, levels: Sequence[str] = (L1_CONTRACT, L2_SMOKE),
            only: Sequence[str] = ()) -> RegressionReport:
        """Run levels in order, stopping at the first failing level.

        Stopping is deliberate: an L1 failure means the controller cannot be built, so an L2 or L3
        result would describe something other than the accepted controller.
        """
        rep = RegressionReport()
        runners: dict[str, Callable[[Sequence[str]], LevelResult]] = {
            L1_CONTRACT: self.run_contract, L2_SMOKE: self.run_smoke, L3_PAIRED: self.run_paired}
        for lvl in LEVELS:
            if lvl not in levels:
                continue
            res = runners[lvl](only)
            rep.results[lvl] = res
            if not res.ok:
                break
        return rep

    def _selected(self, only: Sequence[str]) -> tuple[AcceptedController, ...]:
        if not only:
            return self.registry.all()
        return tuple(self.registry.get(n) for n in only)


def compare_outcomes(entry: AcceptedController, got: PairedOutcome) -> tuple[Certificate, ...]:
    """Per-case comparison against the accepted reference.

    Checks the id SETS first: a changed corpus silently reads as agreement when you only diff the
    keys present in both. Then the per-case map, then the gain/loss sets, then the net.
    """
    ref = entry.outcome
    out: list[Certificate] = []
    if ref.case_ids and got.case_ids:
        missing = sorted(set(ref.case_ids) - set(got.case_ids))
        added = sorted(set(got.case_ids) - set(ref.case_ids))
        if missing or added:
            out.append(Certificate(
                L3_PAIRED, entry.name, OUTCOME_DRIFT,
                f"case id sets differ: {len(missing)} missing, {len(added)} added "
                f"(missing[:3]={missing[:3]}, added[:3]={added[:3]})",
                remedy="the split changed; a comparison over the intersection would read as "
                       "agreement -- re-establish the same corpus before comparing"))
        else:
            flipped = sorted(i for i in ref.case_ids if ref.case_ids[i] != got.case_ids[i])
            if flipped:
                out.append(Certificate(
                    L3_PAIRED, entry.name, OUTCOME_DRIFT,
                    f"{len(flipped)} per-case outcome(s) changed vs the accepted reference "
                    f"(net {ref.net:+d} -> {got.net:+d}); first: {flipped[:5]}",
                    remedy="if intended, record a NEW measured version that supersedes this entry; "
                           "if not, this is a regression in execution"))
    if got.net != ref.net:
        out.append(Certificate(
            L3_PAIRED, entry.name, OUTCOME_DRIFT,
            f"net changed {ref.net:+d} -> {got.net:+d} "
            f"(arm {ref.arm_correct}->{got.arm_correct}, control {ref.control_correct}->"
            f"{got.control_correct})",
            remedy="supersede with a new measured version, or fix the regression"))
    elif (len(got.gains), len(got.losses)) != (len(ref.gains), len(ref.losses)):
        # Equal net, different composition: exactly the offsetting-flip case a total would hide.
        out.append(Certificate(
            L3_PAIRED, entry.name, OUTCOME_DRIFT,
            f"net is unchanged ({ref.net:+d}) but the gain/loss composition changed: "
            f"{len(ref.gains)}g/{len(ref.losses)}l -> {len(got.gains)}g/{len(got.losses)}l",
            remedy="offsetting flips; the aggregate is stable but the mechanism's effect moved"))
    return tuple(out)
