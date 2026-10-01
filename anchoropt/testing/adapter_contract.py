"""THE ADAPTER CONTRACT, as executable checks. Run these against your adapter before trusting a result.

    from anchoropt.testing import check_adapter_contract
    report = check_adapter_contract(MY_ADAPTER, states=my_observed_states)
    assert report.ok, report.summary()

WHY A CONTRACT CHECKER AND NOT JUST DOCUMENTATION
-------------------------------------------------
Every defect class this project spent time on was an adapter declaring something it did not deliver,
and each was invisible until a run came back wrong:

  * a field declared readable at a boundary that the boundary does not carry -- so a predicate over it
    silently answered False forever, and "the condition did not hold" was indistinguishable from "the
    fact does not exist here";
  * `states_at` missing, so predicates were validated against another boundary's states;
  * an executor capability naming a remedy flag that appears nowhere in the code that runs;
  * a contract-required eta field no executor reads, so an arm measured something it did not declare.

None of these is a crash. They all produce confident, wrong numbers, which is why they belong in a
checker rather than a README.

MANDATORY vs OPTIONAL
---------------------
MANDATORY hooks are what `optimize_residual` cannot run without: it must know which events are
decisions, which locus each belongs to, what is observable where, and what the host can execute.

OPTIONAL hooks unlock capability rather than correctness, and the checker says what each one buys.
Without any grounder you get localization and expressibility but no candidates; without
`executor_capability` materializability is unchecked; without `install_signal` Phi cannot be expanded.
A missing optional hook is reported as a NOTE, never a failure -- a first port should be able to run.

THE CHECKER NEVER GUESSES. Every check either reads a declaration and compares it against the
adapter's own behaviour, or reports that it could not check. It does not read your source, and it does
not accept a declaration as evidence of behaviour -- the executor BINDING in particular can only be
established by running the mechanism, which `behavioral_probe` is for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anchoropt.anchor import Action, IncisionPoint

# ================================================================================================
# WHAT AN ADAPTER MUST AND MAY PROVIDE
# ================================================================================================

MANDATORY_HOOKS: tuple[tuple[str, str], ...] = (
    ("is_decision", "which trajectory events are CONSEQUENTIAL decisions; localization runs over these"),
    ("boundary_key", "this adapter's own key for the decision point an event belongs to"),
    ("boundary_from_key", "map your key to a core IncisionPoint -- core owns no key->locus table"),
    ("synthesis_fields", "the TYPED alphabet readable AT ONE boundary; core fails closed if empty"),
    ("declared_signals", "the current Phi: conditions this host can already evaluate"),
    ("signal_boundaries", "where each signal is observable, so a predicate is not built off-boundary"),
    ("evaluate_signal", "evaluate one signal on one state"),
    ("HOST", "a HostProfile naming the executable actions per boundary (U_H(l))"),
)

OPTIONAL_HOOKS: tuple[tuple[str, str], ...] = (
    ("states_at", "project states onto ONE boundary's information set. STRONGLY RECOMMENDED: without "
                  "it core cannot validate a predicate against the right states"),
    ("normalize_event", "adapt your trajectory rows to the event shape the other hooks read"),
    ("observable_state", "derive an observable state from an event"),
    ("label_for", "a human label per boundary, for reports"),
    ("depends_on", "a dependency relation over events; used to order boundaries"),
    ("install_signal", "accept a synthesized predicate. WITHOUT IT Phi CANNOT BE EXPANDED"),
    ("reset_expanded_signals", "drop expanded signals, so a search can restart from shipped Phi"),
    ("expanded_signal_names", "which signals came from expansion; lets one reach an agnostic executor"),
    ("probe_params", "parameters for probing a signal's evaluability"),
    ("parameter_domains", "tunable theta_phi per signal; absent means DETERMINISTIC policies only"),
    ("policy_class_for", "declare a signal's policy class explicitly"),
    ("executor_capability", "BOUND executors: binding + consumed eta. Core can CHECK this form"),
    ("executor_supports", "legacy executor verdict; core can only pass it through, not check it"),
    ("ground_reprompt", "eta for REPROMPT -- instruction content and retry budget"),
    ("ground_suppress", "eta for SUPPRESS -- which operation is cancelled"),
    ("ground_substitute_destinations", "eta for REROUTE/substitute -- a real compatible destination"),
    ("ground_transforms", "eta for REROUTE/transform -- a writable surface"),
)

_GROUNDERS = ("ground_reprompt", "ground_suppress", "ground_substitute_destinations",
              "ground_transforms")


@dataclass(frozen=True)
class ContractViolation:
    """One thing the adapter declares but does not deliver. `hook` names where a porter looks."""

    severity: str              # "FAIL" blocks a trustworthy run; "NOTE" is a missing capability
    hook: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.hook}: {self.detail}"


@dataclass
class ContractReport:
    violations: list[ContractViolation] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[ContractViolation]:
        return [v for v in self.violations if v.severity == "FAIL"]

    @property
    def notes(self) -> list[ContractViolation]:
        return [v for v in self.violations if v.severity == "NOTE"]

    @property
    def ok(self) -> bool:
        """FAILures block; NOTEs do not. A first port with no grounders should still pass."""
        return not self.failures

    def summary(self) -> str:
        lines = [f"adapter contract: {len(self.checked)} check(s) run, "
                 f"{len(self.failures)} failure(s), {len(self.notes)} note(s)"]
        for v in self.violations:
            lines.append(f"  {v}")
        if self.skipped:
            lines.append(f"  not checked (needs more input): {', '.join(sorted(self.skipped))}")
        return "\n".join(lines)


# ================================================================================================
# THE CHECKS
# ================================================================================================

def _has(adapter, name: str) -> bool:
    return getattr(adapter, name, None) is not None


def check_adapter_contract(adapter: Any, *, states: Sequence[Mapping[str, Any]] = (),
                           events: Sequence[Mapping[str, Any]] = ()) -> ContractReport:
    """Run every check that the supplied inputs make possible. Never raises on a bad adapter.

    `states` and `events` are optional but buy real checks: several properties (boundary-truthful
    observability, `states_at` projection) can only be tested against data the adapter produced.
    """
    rep = ContractReport()

    # ---- 1. MANDATORY hooks exist -------------------------------------------------------------
    for hook, why in MANDATORY_HOOKS:
        rep.checked.append(f"mandatory:{hook}")
        if not _has(adapter, hook):
            rep.violations.append(ContractViolation(
                "FAIL", hook, f"MANDATORY and absent -- {why}"))

    for hook, why in OPTIONAL_HOOKS:
        if not _has(adapter, hook):
            rep.violations.append(ContractViolation("NOTE", hook, f"absent -- {why}"))

    if not _has(adapter, "install_signal"):
        rep.violations.append(ContractViolation(
            "NOTE", "install_signal",
            "Phi CANNOT be expanded without this, so a residual the shipped vocabulary cannot "
            "express will report SIGNAL_BLOCKED and stop rather than synthesizing a condition"))
    if not any(_has(adapter, g) for g in _GROUNDERS):
        rep.violations.append(ContractViolation(
            "NOTE", "ground_*",
            "no eta grounder at all: localization and expressibility work, but NO candidate can be "
            "built, so every round ends ACTION_UNAVAILABLE"))
    if not _has(adapter, "executor_capability") and not _has(adapter, "executor_supports"):
        rep.violations.append(ContractViolation(
            "NOTE", "executor_capability",
            "materializability is UNCHECKED: an action admissible in U_H(l) with no executing code "
            "behind it will be built and measured as if it ran"))

    if rep.failures:
        return rep                       # further checks would cascade meaninglessly

    # ---- 2. HOST declares real actions at real loci --------------------------------------------
    rep.checked.append("host:executable_actions")
    host = getattr(adapter, "HOST", None)
    try:
        any_action = False
        for point in IncisionPoint:
            acts = host.executable_actions(point)
            for a in acts or ():
                if not isinstance(a, Action):
                    rep.violations.append(ContractViolation(
                        "FAIL", "HOST", f"executable_actions({point.value}) yielded {a!r}, which is "
                                        f"not a core Action -- use the Action enum, not a string"))
                any_action = True
        if not any_action:
            rep.violations.append(ContractViolation(
                "FAIL", "HOST", "no executable action at ANY boundary, so nothing can ever be built"))
    except Exception as exc:
        rep.violations.append(ContractViolation(
            "FAIL", "HOST", f"executable_actions raised {type(exc).__name__}: {exc}"))

    # ---- 3. BOUNDARY-TRUTHFUL OBSERVABILITY ----------------------------------------------------
    #
    # THE MOST COMMON PORTING BUG. A field declared at a boundary the runtime does not carry makes
    # every predicate over it answer False, for a structural reason that looks exactly like the
    # condition not holding.
    rep.checked.append("synthesis_fields:per_boundary")
    fields_by_point: dict[str, set] = {}
    for point in IncisionPoint:
        try:
            got = adapter.synthesis_fields(point) or {}
        except Exception as exc:
            rep.violations.append(ContractViolation(
                "FAIL", "synthesis_fields",
                f"raised for {point.value}: {type(exc).__name__}: {exc}. It must return a mapping "
                f"(possibly empty) for every boundary, because core calls it per boundary"))
            continue
        fields_by_point[point.value] = set(got)
        for name, fld in dict(got).items():
            if not hasattr(fld, "type"):
                rep.violations.append(ContractViolation(
                    "FAIL", "synthesis_fields",
                    f"{point.value}/{name} has no `.type`; core synthesizes TYPED predicates and "
                    f"cannot build one over an untyped field"))
            declared = tuple(getattr(fld, "boundaries", ()) or ())
            if declared and point.value not in declared and \
                    not any(str(point.value) in str(d) for d in declared):
                # A field may legitimately name the adapter's own keys rather than locus values, so
                # this is a NOTE: the checker cannot tell a key from a wrong boundary.
                rep.violations.append(ContractViolation(
                    "NOTE", "synthesis_fields",
                    f"{name} is returned at {point.value} but its own `boundaries` say {list(declared)}"
                    f" -- if those are your own keys this is fine; if they are locus values, one of "
                    f"the two declarations is wrong"))

    if all(not v for v in fields_by_point.values()):
        rep.violations.append(ContractViolation(
            "FAIL", "synthesis_fields",
            "EVERY boundary returned an empty alphabet. Core fails closed here, so no condition can "
            "ever be synthesized. Declare the typed facts your runtime records at each boundary"))

    # Post-execution facts must not leak into a pre-dispatch boundary.
    pre = fields_by_point.get(IncisionPoint.POST_GENERATION_PRE_EXEC.value, set())
    post = fields_by_point.get(IncisionPoint.POST_EXECUTION.value, set())
    if pre and post:
        shared = pre & post
        if shared:
            rep.violations.append(ContractViolation(
                "NOTE", "synthesis_fields",
                f"{sorted(shared)} are declared readable at BOTH the commitment gate and after "
                f"execution. That is legitimate for a fact carried across (a step index, a proposed "
                f"payload), and a defect for a RESULT -- a result cannot exist before dispatch"))

    # ---- 4. SIGNALS ARE EVALUABLE WHERE THEY ARE DECLARED --------------------------------------
    rep.checked.append("declared_signals:evaluable")
    try:
        signals = tuple(adapter.declared_signals() or ())
    except Exception as exc:
        signals = ()
        rep.violations.append(ContractViolation(
            "FAIL", "declared_signals", f"raised {type(exc).__name__}: {exc}"))
    if not signals:
        rep.violations.append(ContractViolation(
            "NOTE", "declared_signals",
            "Phi is EMPTY. Nothing can be searched before expansion, and without `install_signal` "
            "nothing can be expanded either"))
    for sig in signals:
        try:
            bnds = adapter.signal_boundaries(sig)
        except Exception as exc:
            rep.violations.append(ContractViolation(
                "FAIL", "signal_boundaries", f"raised for {sig!r}: {type(exc).__name__}: {exc}"))
            continue
        if not bnds:
            rep.violations.append(ContractViolation(
                "NOTE", "signal_boundaries",
                f"{sig!r} is declared observable NOWHERE, so no controller can be built on it"))

    # ---- 5. states_at PROJECTS, and fails closed rather than guessing --------------------------
    if _has(adapter, "states_at"):
        rep.checked.append("states_at:projection")
        if states:
            for point in IncisionPoint:
                try:
                    projected = list(adapter.states_at(point, states) or ())
                except Exception as exc:
                    rep.violations.append(ContractViolation(
                        "FAIL", "states_at",
                        f"raised for {point.value}: {type(exc).__name__}: {exc}"))
                    continue
                if len(projected) > len(states):
                    rep.violations.append(ContractViolation(
                        "FAIL", "states_at",
                        f"{point.value} returned MORE states ({len(projected)}) than it was given "
                        f"({len(states)}); a projection may filter, never invent"))
                allowed = fields_by_point.get(point.value, set())
                if allowed:
                    for st in projected[:50]:
                        leaked = (set(st) - set(allowed)) & (
                            set().union(*[v for k, v in fields_by_point.items()
                                          if k != point.value]) if len(fields_by_point) > 1 else set())
                        # Only flag a leak of a field declared at ANOTHER boundary and not this one.
                        if leaked:
                            rep.violations.append(ContractViolation(
                                "NOTE", "states_at",
                                f"a state projected at {point.value} carries {sorted(leaked)}, which "
                                f"is declared at another boundary. If those are carried facts this is "
                                f"fine; if they are that boundary's results, the projection leaks"))
                            break
        else:
            rep.skipped.append("states_at:projection (pass `states=` to check it)")
    else:
        rep.violations.append(ContractViolation(
            "NOTE", "states_at",
            "absent. Core FAILS CLOSED without it, so a predicate cannot be validated against the "
            "states of its own boundary -- this is the hook that prevents selecting a controller "
            "which can never fire"))

    # ---- 6. EXECUTOR CAPABILITIES ARE BOUND ----------------------------------------------------
    if _has(adapter, "executor_capability"):
        rep.checked.append("executor_capability:bound")
        for point in IncisionPoint:
            for action in Action:
                try:
                    cap = adapter.executor_capability(point, action)
                except Exception as exc:
                    rep.violations.append(ContractViolation(
                        "FAIL", "executor_capability",
                        f"raised for {point.value}/{action.value}: {type(exc).__name__}: {exc}"))
                    continue
                if cap is None:
                    continue
                if not getattr(cap, "is_bound", False):
                    rep.violations.append(ContractViolation(
                        "FAIL", "executor_capability",
                        f"{point.value}/{action.value} is declared with NO binding -- a GHOST. A "
                        f"declaration is not an executor: name the code that runs, or remove the "
                        f"cell"))
                if not getattr(cap, "is_enabled", True):
                    # A DISABLED cell is exempt from the eta rules, and deliberately so: the honest
                    # declaration for an inert action is `consumes=()` plus a reason, because nothing
                    # is consumed -- that is what makes it inert. It can never be measured, so an arm
                    # cannot carry a parameter nothing reads. Reported as a NOTE so the cell stays
                    # visible: a silently absent executor reads as one nobody thought of.
                    rep.violations.append(ContractViolation(
                        "NOTE", "executor_capability",
                        f"{point.value}/{action.value} is declared but DISABLED, so no arm will be "
                        f"built there: {getattr(cap, 'disabled_reason', '')[:120]}"))
                    continue
                if not getattr(cap, "consumes", ()) and not getattr(cap, "eta_is_computed", False):
                    rep.violations.append(ContractViolation(
                        "FAIL", "executor_capability",
                        f"{point.value}/{action.value} declares neither `consumes` nor "
                        f"`eta_is_computed`. Say which eta keys the executing code READS, or declare "
                        f"that it derives its parameters from live state -- otherwise an arm may "
                        f"carry parameters nothing reads"))

    # ---- 7. EVENTS LOCALIZE ---------------------------------------------------------------------
    if events:
        rep.checked.append("localization:round_trip")
        try:
            keys = [adapter.boundary_key(e) for e in events if adapter.is_decision(e)]
        except Exception as exc:
            keys = []
            rep.violations.append(ContractViolation(
                "FAIL", "boundary_key", f"raised: {type(exc).__name__}: {exc}"))
        if not keys:
            rep.violations.append(ContractViolation(
                "FAIL", "is_decision",
                "no supplied event is a decision, so localization finds no boundary and every round "
                "ends BOUNDARY_NOT_REPAIRABLE. Check that `is_decision` recognizes your rows"))
        for k in set(keys):
            if not k:
                continue
            if adapter.boundary_from_key(k) is None:
                rep.violations.append(ContractViolation(
                    "FAIL", "boundary_from_key",
                    f"key {k!r} maps to no IncisionPoint, so core cannot tell WHERE it is"))
    else:
        rep.skipped.append("localization:round_trip (pass `events=` to check it)")

    return rep


class AdapterContract:
    """Pytest-friendly mixin. Subclass it in YOUR test file and every check below runs on your adapter.

        class TestMyAdapter(AdapterContract):
            adapter = MY_ADAPTER          # a module, a class, or an instance -- any is fine
            states = MY_STATES            # optional, but buys the projection checks
            events = MY_EVENTS            # optional, but buys the localization checks

    DELIBERATELY NOT A DATACLASS. A generated `__init__` makes pytest refuse to collect the subclass
    ("cannot collect test class ... because it has a __init__ constructor") -- it warns and silently
    runs NOTHING, which is worse than failing. Plain class attributes collect correctly.
    """

    adapter: Any = None
    states: Sequence[Mapping[str, Any]] = ()
    events: Sequence[Mapping[str, Any]] = ()

    def report(self) -> ContractReport:
        return check_adapter_contract(self.adapter, states=self.states, events=self.events)

    def test_the_contract_has_no_failures(self):
        rep = self.report()
        assert rep.ok, rep.summary()

    def test_every_mandatory_hook_is_present(self):
        missing = [h for h, _ in MANDATORY_HOOKS if getattr(self.adapter, h, None) is None]
        assert not missing, f"mandatory hooks absent: {missing}"

    def test_at_least_one_boundary_has_a_typed_alphabet(self):
        got = {p.value: set(self.adapter.synthesis_fields(p) or {}) for p in IncisionPoint}
        assert any(got.values()), \
            f"core fails closed on an empty alphabet at every boundary: {got}"
