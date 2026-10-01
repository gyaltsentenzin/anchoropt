"""The adapter-owned AnchorOpt middleware: three hooks, the persisted controller format, and the trace.

    Controller = (locus, phi, action, theta)
    at locus:  state = observable_state();  if phi(state): apply_action(action, theta, state)

Nothing here is controller-specific. The middleware is constructed from persisted `ControllerSpec`s
and works for any `(l, phi, mu, theta)` this host can execute, which is what makes it infrastructure
rather than tuning for one candidate. The alternative -- a bespoke gate per controller -- is how a host
ends up owning the trigger and silently overriding phi.

WHY THIS FILE EXISTS: FOUR GUARANTEES A CANDIDATE MUST NEVER BE RESPONSIBLE FOR
------------------------------------------------------------------------------

1. THREAD SAFETY, and this is the one with no precedent in the other two integrations.
   `response_generator.py` runs episodes in a `ThreadPoolExecutor` whose default is
   `max_workers=4`, so four episodes interleave in one process. Per-episode bookkeeping -- the turn
   index, the calls seen so far, the violation streak, the one-shot guard -- CANNOT live on the
   middleware, or four episodes would share one counter and every one of those fields would be
   wrong in a way no test on a single episode can show. So the shared object owns only the specs and
   the aggregate telemetry, and `episode()` hands out a per-episode `EpisodeHooks` that owns its own
   state. The aggregate counters and the trace file are guarded by one lock, because four threads
   appending to one JSONL is how a trace silently loses lines.

   TB2's middleware faced the weaker version of this ("nothing guarantees a fresh instance per
   episode") and keyed its guard on episode identity. Here the episodes are genuinely concurrent, so
   identity-keying alone is not enough -- the state has to be separate objects.

2. THE CONTROL ARM IS INERT BY CONSTRUCTION. With no spec loaded, nothing is installed, and
   `anchoropt.runtime_hook.decide` returns the host's own decision UNCHANGED -- so every hook returns
   None and the harness behaves exactly as it does un-hooked. Omitting `--controllers` IS the
   control, and `verify_plumbing.py` is what asserts it rather than assuming it.

3. TELEMETRY SURVIVES THE INTERVENTION. `docs/GENERALIZABILITY.md` names this as the third porting
   seam and records that it broke four times on the other benchmark: "a suppressed call that simply
   vanishes from the trajectory erases the evidence needed to evaluate the suppression." So the trace
   records `proposed -> intervened -> executed` for every firing AND every declined evaluation, which
   is what `--anchor-trace`'s own help text already promises. Firing counts must be read BEFORE any
   accuracy: nothing in CCTU can validate an anchor, because the controllers live outside it.

4. A SPEC THAT CANNOT RUN FAILS AT LOAD, NOT MID-EPISODE. `validate()` checks the signal is
   observable at the declared boundary, the action is executable there, an executor exists for the
   cell, and `theta` carries what `apply_action` requires. A spec that would no-op silently must not
   produce a middleware at all -- a candidate arm that quietly becomes the control arm is the defect
   this project has paid for most often.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anchoropt.anchor import Action, IncisionPoint

import cctu_adapter as cctu
import cctu_state as state_mod

SPEC_FORMAT = "anchoropt.controller.v1"


# ------------------------------------------------------------------------------------------------
# The persisted controller format
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ControllerSpec:
    """A saved AnchorOpt controller: exactly `(l, phi, mu, theta)` plus what is needed to load it.

    THE SAME `anchoropt.controller.v1` FORMAT AS `benchmarks/tb2_deepagents/tb2_middleware.py`, and
    deliberately so: two runtimes emitting one spec format is what makes a cross-benchmark claim
    stateable at all, and it means a spec synthesized by `optimize_residual` needs no per-benchmark
    translation. It carries no task id, no benchmark concept, and nothing CCTU-specific -- `theta` is
    opaque payload.

    ONE ADDITION OVER THE TB2 FORM, and it closes a real gap. TB2's spec names a DECLARED signal, so a
    condition the search synthesized this session can be installed but not PERSISTED: a fresh process
    loading that spec finds the name undeclared and refuses. So `predicate` is accepted as an
    alternative -- an expression tree over the runtime's declared FIELDS, compiled by
    `anchoropt/learning/signal_lang.py`. Exactly one of `signal` / `predicate` must be given.

    Using `signal_lang` rather than a local grammar is the point: it validates that every field is
    declared and typed, it forbids `eval` and code strings, and it DERIVES the boundary set as the
    intersection of the referenced fields' boundaries -- so a spec cannot claim a locus at which its
    own facts never coexist. `scripts/install_controller.py` carries a smaller private grammar for the
    same job; this uses the shared one instead of forking it.
    """

    controller_id: str
    boundary: str                                   # l, an IncisionPoint value
    action: str                                     # mu, an Action value
    theta: Mapping[str, Any] = field(default_factory=dict)
    signal: str = ""                                # phi, from the host's declared vocabulary
    predicate: Mapping[str, Any] = field(default_factory=dict)   # ... OR an inline expression
    signal_params: Mapping[str, Any] = field(default_factory=dict)
    provenance: str = ""

    def to_json(self) -> str:
        return json.dumps({"format": SPEC_FORMAT, "controller_id": self.controller_id,
                           "boundary": self.boundary, "action": self.action,
                           "theta": dict(self.theta), "signal": self.signal,
                           "predicate": dict(self.predicate),
                           "signal_params": dict(self.signal_params),
                           "provenance": self.provenance}, indent=2, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n")
        return path

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ControllerSpec:
        if d.get("format") != SPEC_FORMAT:
            raise ValueError(f"controller spec format must be {SPEC_FORMAT!r}, "
                             f"got {d.get('format')!r}")
        return cls(controller_id=str(d["controller_id"]), boundary=str(d["boundary"]),
                   action=str(d["action"]), theta=dict(d.get("theta") or {}),
                   signal=str(d.get("signal") or ""), predicate=dict(d.get("predicate") or {}),
                   signal_params=dict(d.get("signal_params") or {}),
                   provenance=str(d.get("provenance") or ""))

    @classmethod
    def load(cls, path: str | Path) -> tuple[ControllerSpec, ...]:
        """Load one spec, or a `{"controllers": [...]}` bundle. Both shapes, as the CLI promises."""
        d = json.loads(Path(path).read_text())
        if isinstance(d, Mapping) and "controllers" in d:
            return tuple(cls.from_dict(x) for x in d["controllers"])
        if isinstance(d, list):
            return tuple(cls.from_dict(x) for x in d)
        return (cls.from_dict(d),)

    # -- resolution into typed objects, validated against the host ------------------------------
    @property
    def l(self) -> IncisionPoint:
        return IncisionPoint(self.boundary)

    @property
    def mu(self) -> Action:
        return Action(self.action)

    @property
    def phi_name(self) -> str:
        return self.signal or str(self.predicate.get("name") or self.controller_id)

    def validate(self) -> None:
        """Fail loudly at LOAD time, not mid-episode. Five checks, in dependency order.

        A mechanism policy from the previous generation -- one with an `enable` block -- is refused
        here rather than ignored, which is what `response_generator.py`'s `--policy` help promises.
        Ignoring it would run the control under an arm's name.
        """
        if bool(self.signal) == bool(self.predicate):
            raise ValueError(
                f"controller {self.controller_id!r} must declare exactly one of `signal` (a name "
                f"this host declares) or `predicate` (an expression over its declared fields); "
                f"got signal={self.signal!r}, predicate={'set' if self.predicate else 'unset'}")

        # 1/2. phi is observable AT the declared boundary
        if self.signal:
            if self.signal not in cctu.declared_signals():
                raise ValueError(f"signal {self.signal!r} not declared by {cctu.NAME}; declared: "
                                 f"{list(cctu.declared_signals())}")
            cctu.evaluate_signal(self.signal, {}, self.signal_params)      # param-schema check
            at = {p.value for p in cctu.signal_boundaries(self.signal)}
        else:
            at = set(self._compile().boundaries)
        if self.boundary not in at:
            raise ValueError(
                f"controller {self.controller_id!r} installs at {self.boundary!r}, where its "
                f"condition is not observable (observable at: {sorted(at) or 'nowhere'}). A "
                f"condition evaluated where its facts do not exist answers False for a structural "
                f"reason indistinguishable from 'it did not hold'.")

        # 3. mu is executable there -- U_H(l)
        cctu.HOST.require(self.l, self.mu)

        # 4. an executor exists for the cell. Admissibility is not materializability.
        if cctu.executor_for(self.l, self.mu) is None:
            raise ValueError(f"controller {self.controller_id!r} names ({self.boundary}, "
                             f"{self.action}), which {cctu.NAME} declares admissible but has no "
                             f"executor for")

        # 5. theta carries what apply_action requires. A DRY RUN, with a synthetic state chosen so
        #    the payload checks run: `n_tool_calls=2` satisfies SUPPRESS/drop_call's precondition,
        #    which is a LIVE condition and is therefore re-checked at firing time -- this only
        #    proves the theta KEYS are present, which is exactly what a load-time check can prove.
        cctu.apply_action(self.mu, self.l, self.theta, {"n_tool_calls": 2, "tool_name": "_probe"})

    def _compile(self):
        from anchoropt.learning.signal_lang import compile_signal

        expr = dict(self.predicate)
        name = str(expr.pop("name", "") or self.controller_id)
        return compile_signal(name, expr, fields=cctu.synthesis_fields())


# ------------------------------------------------------------------------------------------------
# The installed controller -- what `anchoropt.runtime_hook` sees
# ------------------------------------------------------------------------------------------------
class InstalledController:
    """Duck-typed to the hook contract: `.fires_on(state)`, `.eta`, `.name`.

    `eta` carries `theta` so the hook's decision travels with its payload. The two vocabularies are
    genuinely different -- `eta` is the search-side action parameter named by
    `anchoropt/learning/action_contract.py`, `theta` is the execution-side payload `apply_action`
    reads, and `cctu_adapter.ETA_TO_THETA` maps between them -- so this is the one place they meet,
    named rather than merged.
    """

    def __init__(self, spec: ControllerSpec) -> None:
        self.spec = spec
        self.name = f"{spec.action}@{spec.boundary}[{spec.phi_name}]"
        self.eta = dict(spec.theta)
        self._compiled = None if spec.signal else spec._compile()

    def fires_on(self, state: Mapping[str, Any]) -> bool:
        """WHEN. The only place that question is answered.

        A MISSING FIELD IS NOT A FIRING: `signal_lang` evaluates a comparison against an absent field
        as False, and the declared signals read through `.get`. That keeps "the condition did not
        hold" distinguishable from "the state never carried what phi reads", which this project has
        paid for more than once. A raising predicate is not a trigger either -- `runtime_hook.decide`
        catches it and records the error rather than reading it as a fire.
        """
        if self._compiled is not None:
            return bool(self._compiled.predicate(state, self.spec.signal_params))
        return bool(cctu.evaluate_signal(self.spec.signal, state, self.spec.signal_params))


# ------------------------------------------------------------------------------------------------
# Per-episode hooks
# ------------------------------------------------------------------------------------------------
class EpisodeHooks:
    """One episode's three hooks and its own bookkeeping. **Not shared between episodes.**

    Every mutable field here is per-episode, which is the whole reason this class exists: with
    `max_workers=4` a middleware attribute would be four episodes' turn counters added together.

    The bookkeeping is what `cctu_state.carried_state` needs and what no single turn can supply:
    the turn index, the `(name, arguments)` pairs already proposed, the most recent violation class,
    and the streak of consecutive violating turns.
    """

    def __init__(self, parent: AnchorOptMiddleware, case_id: str) -> None:
        self._parent = parent
        self.case_id = str(case_id)
        self.round_index = 0
        self.seen_calls: list[tuple[str, str]] = []
        self.last_violation_class: str | None = None
        self.consecutive_violation_turns = 0
        self._intervened: set[str] = set()
        # Redecides REQUESTED AND GRANTED on the turn being decided. Reset by `observe_turn`, which
        # runs once per COMPLETED turn -- a redecide never reaches it, because the runner breaks out
        # of the turn before the validator runs. See `_at` for why the executor owns this budget.
        self._redecides_this_turn = 0
        # `(name, arguments) -> the tool's VERBATIM returned string`, for the one mechanism that
        # replays a recorded result. Recorded ONLY from calls that actually executed
        # (`docs/EFFICIENCY_CLASS.md` invariant 4) -- a withheld call contributes nothing, or the
        # cache would start feeding itself.
        self.results_by_call: dict[tuple[str, str], str] = {}

    # -- the three incision points ---------------------------------------------------------------
    def pre_generation(self, *, normalized: Mapping[str, Any] | None = None,
                       snapshot: Any = None) -> dict[str, Any] | None:
        """Before the model is called. Context and state are visible; no decision exists yet."""
        return self._at(IncisionPoint.PRE_GENERATION, normalized or {"has_generation": False},
                        snapshot)

    def post_generation_pre_exec(self, *, normalized: Mapping[str, Any],
                                 snapshot: Any = None) -> dict[str, Any] | None:
        """The commitment gate. **Call this BEFORE `get_feedback`.**

        That ordering is the whole value of this boundary: the validator has not run, so nothing has
        been counted and a suppression is free. Calling it after `get_feedback` would cancel the
        execution while leaving the budget charged for a call that never ran -- strictly worse than
        doing nothing. See ANCHOROPT.md on the two POST_GENERATION sub-moments.
        """
        return self._at(IncisionPoint.POST_GENERATION_PRE_EXEC, normalized, snapshot)

    def post_execution(self, *, normalized: Mapping[str, Any],
                       snapshot: Any = None) -> dict[str, Any] | None:
        """After the validator's verdict and the tool results exist. The world has already moved."""
        return self._at(IncisionPoint.POST_EXECUTION, normalized, snapshot)

    # -- bookkeeping ------------------------------------------------------------------------------
    def observe_turn(self, normalized: Mapping[str, Any], *,
                     executed: Mapping[str, str] | None = None) -> None:
        """Record a completed turn. Call once per turn, AFTER the post-execution hook.

        The violation streak counts turns that produced at least one violation and RESETS on a clean
        one, so it measures a stuck self-refinement loop rather than a running total -- which matters
        on this benchmark, where the environment reprompts on every violation and a model can sit in
        that loop until the round budget is gone.

        `executed` maps `tool_call_id -> the verbatim string the tool returned`, for calls that
        ACTUALLY RAN. It is the only thing allowed to populate `results_by_call`: recording a
        substituted observation would let the replay cache feed itself, and recording a violation
        message would replay a refusal as though it were a result.
        """
        calls = list(normalized.get("tool_calls_raw") or ())
        self.seen_calls.extend(state_mod.call_keys(calls))
        if executed:
            for call in calls:
                cid = str(call.get("id") or "")
                if cid not in executed:
                    continue
                fn = call.get("function") or {}
                key = (str(fn.get("name") or ""), str(fn.get("arguments") or ""))
                self.results_by_call.setdefault(key, executed[cid])
        cls = normalized.get("violation_class")
        if cls is not None:
            self.last_violation_class = str(cls)
            self.consecutive_violation_turns += 1
        else:
            self.consecutive_violation_turns = 0
        self.round_index += 1
        # A turn COMPLETED, so the next one gets a fresh redecide budget. This is the only reset:
        # a granted redecide abandons the turn before the validator runs and never arrives here,
        # which is exactly the property that makes this counter bound the regeneration loop.
        self._redecides_this_turn = 0

    # -- the shared path --------------------------------------------------------------------------
    def _carried(self, normalized: Mapping[str, Any], snapshot: Any) -> dict[str, Any]:
        """The carried half of the observable state.

        Without a snapshot the pressure fields are simply ABSENT rather than defaulted, and a
        predicate reading one evaluates to False. That is the honest degradation -- a default would
        make "the budget was not near its cap" indistinguishable from "nobody looked" -- but it means
        a run without snapshots cannot fire a pressure-based controller, so the wrapper must supply
        one. `verify_plumbing.py` is where that gets asserted rather than hoped for.
        """
        if snapshot is None:
            return {"round_index": self.round_index,
                    "last_violation_class": self.last_violation_class,
                    "consecutive_violation_turns": self.consecutive_violation_turns}
        return state_mod.carried_state(
            snapshot, normalized, round_index=self.round_index,
            seen_calls=tuple(self.seen_calls),
            last_violation_class=self.last_violation_class,
            consecutive_violation_turns=self.consecutive_violation_turns)

    def _at(self, boundary: IncisionPoint, normalized: Mapping[str, Any],
            snapshot: Any) -> dict[str, Any] | None:
        from anchoropt.runtime_hook import decide

        parent = self._parent
        parent._count("boundary_exposures")

        try:
            carried = self._carried(normalized, snapshot)
            st = cctu.observable_state(dict(normalized, boundary=boundary.value), carried)
        # BLE001 silenced deliberately: building the observable state must never be able to kill
        # an episode, and narrowing this would mean guessing what a per-episode validator raises.
        # The failure is RECORDED, not swallowed -- a hook that looks clean because it crashed is
        # worse than no hook.
        except Exception as exc:  # noqa: BLE001
            parent._error(f"observable_state@{boundary.value}: {type(exc).__name__}: {exc}")
            return None

        decision = decide(boundary.value, st, host_default=False)
        if not (decision.proceed and decision.by_controller):
            # A declined evaluation is TRACED, not dropped. Firing counts are only auditable if the
            # denominator is recorded too, and `--anchor-trace`'s help promises both.
            parent._trace(self, boundary, None, normalized, fired=False, executed=False,
                          detail="no controller fired")
            return None

        controller = decision.controller
        parent._count("signal_firings")

        if parent.one_shot and controller.name in self._intervened:
            parent._count("loop_prevention_events")
            parent._trace(self, boundary, controller, normalized, fired=True, executed=False,
                          detail="one_shot: already intervened in this episode")
            return None

        try:
            directive = cctu.apply_action(controller.spec.mu, boundary,
                                          controller.spec.theta, st)
        except Exception as exc:  # noqa: BLE001
            # A live precondition failing here is NORMAL, not a bug: SUPPRESS/drop_call refuses a
            # turn proposing only one call, because removing it would convert a tool turn into a
            # final answer. Recorded with its reason so the decline is attributable.
            parent._error(f"apply_action@{boundary.value}: {type(exc).__name__}: {exc}")
            parent._trace(self, boundary, controller, normalized, fired=True, executed=False,
                          detail=f"declined: {type(exc).__name__}: {exc}")
            return None

        if not directive.get("executed"):
            parent._trace(self, boundary, controller, normalized, fired=True, executed=False,
                          detail=f"directive kind={directive.get('kind')!r} is not executable")
            return None

        # THE REDECIDE BUDGET. `action_contract` declares `retry_budget` on every reprompt and
        # `evolve_memory/primitives.py` states it is "FIXED at 1 by the executor" -- but on this
        # runtime NOTHING enforced it. `response_generator.py`'s l2 branch sets `redecide = True`,
        # breaks, and `continue`s the episode loop without charging a round and without incrementing
        # `times`, so a controller that fires again on the regenerated proposal regenerates forever.
        #
        # It is not live TODAY only because `one_shot` caps each controller at one intervention per
        # episode. That is the wrong guarantee to rely on: it is a property of a different setting,
        # and a per-turn gate (a duplicate-call detector fires ~8x per affected episode on this
        # corpus) has to run with `one_shot=False` to engage at all. So the bound is made explicit
        # and independent of `one_shot`.
        #
        # SCOPED TO THE COMMITMENT GATE, deliberately. `apply_action` sets `request_redecision` at
        # POST_EXECUTION too, but the runner does not regenerate there -- it injects and clears
        # `finish`. Widening this check would change the semantics cycle 1 measured, and a frozen
        # control must keep meaning what it meant.
        if (boundary is IncisionPoint.POST_GENERATION_PRE_EXEC
                and directive.get("request_redecision")):
            if self._redecides_this_turn >= parent.redecide_budget:
                parent._count("redecide_budget_declines")
                parent._trace(self, boundary, controller, normalized, fired=True, executed=False,
                              detail=(f"redecide budget spent: {self._redecides_this_turn} granted "
                                      f"on this turn, budget {parent.redecide_budget}"))
                return None
            self._redecides_this_turn += 1

        parent._count("interventions_executed")
        self._intervened.add(controller.name)
        parent._trace(self, boundary, controller, normalized, fired=True, executed=True,
                      detail="", directive=directive)
        return directive


# ------------------------------------------------------------------------------------------------
# The middleware
# ------------------------------------------------------------------------------------------------
class AnchorOptMiddleware:
    """Shared across episodes: the specs, the aggregate telemetry, and the trace file.

    Holds NO per-episode state. `episode(case_id)` returns an `EpisodeHooks` that does.
    """

    def __init__(self, specs: Sequence[ControllerSpec] = (), *, one_shot: bool = True,
                 redecide_budget: int = 1,
                 trace_path: str | Path | None = None) -> None:
        from anchoropt.runtime_hook import install

        self.specs = tuple(specs)
        self.one_shot = one_shot
        # Granted redecides per TURN, at the commitment gate. 1 is the contract's own value; 0 turns
        # the gate's reprompt cell into a noop and is accepted so a run can be scored with the cell
        # disabled rather than un-wired. Negative is a typo, not a policy.
        if int(redecide_budget) < 0:
            raise ValueError(f"redecide_budget must be >= 0, got {redecide_budget!r}")
        self.redecide_budget = int(redecide_budget)
        self.trace_path = Path(trace_path) if trace_path else None
        self._lock = threading.Lock()
        self.telemetry: dict[str, Any] = {
            "boundary_exposures": 0, "signal_firings": 0, "interventions_executed": 0,
            "loop_prevention_events": 0, "redecide_budget_declines": 0,
            "episodes_seen": 0, "errors": [],
            "controllers": [], "trace_rows": 0,
        }
        self.controllers: list[InstalledController] = []
        for spec in self.specs:
            spec.validate()                            # loud, at LOAD time
            controller = InstalledController(spec)
            install(spec.boundary, controller)         # the ONE generic runtime hook
            self.controllers.append(controller)
            self.telemetry["controllers"].append(controller.name)
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    # -- construction -----------------------------------------------------------------------------
    @classmethod
    def from_path(cls, path: str | Path | None, *, one_shot: bool = True,
                  redecide_budget: int = 1,
                  trace_path: str | Path | None = None) -> AnchorOptMiddleware:
        """Build from a spec file, or **the control** when `path` is None.

        `path=None` installs nothing, so `runtime_hook.decide` returns the host's own decision
        unchanged and every hook returns None. That is the control arm, inert by construction rather
        than by a flag nobody checks.
        """
        specs = ControllerSpec.load(path) if path else ()
        return cls(specs, one_shot=one_shot, redecide_budget=redecide_budget,
                   trace_path=trace_path)

    @property
    def is_control(self) -> bool:
        """True when nothing is installed. The property `verify_plumbing.py` asserts against."""
        return not self.controllers

    def episode(self, case_id: str) -> EpisodeHooks:
        """A fresh per-episode handle. Thread-safe: the returned object is not shared."""
        with self._lock:
            self.telemetry["episodes_seen"] += 1
        return EpisodeHooks(self, case_id)

    def reset(self) -> None:
        """Drop every installed controller, so one arm cannot inherit another's policy.

        `runtime_hook.reset()` is global, which is correct and is also why constructing two
        middlewares in one process is unsupported: the second would share the first's installed
        controllers. One arm, one process.
        """
        from anchoropt.runtime_hook import reset

        reset()
        self.controllers.clear()

    # -- telemetry and trace, both under the lock -------------------------------------------------
    def _count(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.telemetry[key] = self.telemetry.get(key, 0) + n

    def _error(self, message: str) -> None:
        with self._lock:
            self.telemetry["errors"].append(message)

    def _trace(self, hooks: EpisodeHooks, boundary: IncisionPoint,
               controller: InstalledController | None, normalized: Mapping[str, Any], *,
               fired: bool, executed: bool, detail: str = "",
               directive: Mapping[str, Any] | None = None) -> None:
        """One `proposed -> intervened -> executed` row. Written for EVERY evaluation.

        The proposed calls are recorded BEFORE any intervention touches them, which is the point:
        `docs/GENERALIZABILITY.md` records that a suppressed call vanishing from the trajectory
        erases the evidence needed to evaluate the suppression, and that it recurred four times on
        the other benchmark via a field allowlist silently dropping telemetry.

        Appends under the lock. Four threads appending to one JSONL without it is how a trace loses
        lines -- and a trace with missing lines understates firings, which is the number that gates
        reading any accuracy at all.
        """
        if self.trace_path is None:
            return
        row = {
            "case_id": hooks.case_id,
            "turn": hooks.round_index,
            "boundary": boundary.value,
            "controller": controller.name if controller else None,
            "signal": controller.spec.phi_name if controller else None,
            "action": controller.spec.action if controller else None,
            "fired": bool(fired),
            "executed": bool(executed),
            "detail": detail,
            "proposed": {
                "n_tool_calls": normalized.get("n_tool_calls"),
                "tool_names": list(cctu.actions_in(normalized)),
                "violation_class": normalized.get("violation_class"),
            },
            "directive": dict(directive) if directive else None,
        }
        line = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            with open(self.trace_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
            self.telemetry["trace_rows"] += 1

    def telemetry_snapshot(self) -> dict[str, Any]:
        """A copy of the aggregate counters, safe to read while episodes are running."""
        with self._lock:
            snap = dict(self.telemetry)
        snap["errors"] = list(snap.get("errors", ()))
        snap["controllers"] = list(snap.get("controllers", ()))
        from anchoropt.runtime_hook import errors as hook_errors
        snap["hook_errors"] = list(hook_errors())
        snap["predictive_handler_errors"] = list(state_mod.HANDLER_ERRORS)
        return snap


# ------------------------------------------------------------------------------------------------
# The refusal the CLI already promises
# ------------------------------------------------------------------------------------------------
def refuse_legacy_policy(path: str | Path) -> None:
    """Raise if `path` is a previous-generation mechanism policy. **Refused, not ignored.**

    `response_generator.py`'s `--policy` help says a policy with an `enable` block is refused rather
    than ignored, and that is not pedantry: silently ignoring it would run the CONTROL under an arm's
    name, which is the single most expensive kind of measurement error available here. The BFCL
    integration has real files of that shape, and pointing `--policy` at one is an easy mistake.
    """
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: not readable as JSON ({exc})") from exc
    if not isinstance(d, Mapping):
        return
    if d.get("format") == SPEC_FORMAT or "controllers" in d:
        return
    if any(k in d for k in ("enable", "gates", "gate_default", "templates", "remedies")):
        raise ValueError(
            f"{path} looks like a previous-generation mechanism policy (it declares "
            f"{sorted(k for k in ('enable', 'gates', 'gate_default', 'templates', 'remedies') if k in d)}). "
            f"This integration takes an {SPEC_FORMAT!r} controller spec via --controllers. Refusing "
            f"rather than ignoring: an ignored policy would run the control under an arm's name.")
    raise ValueError(f"{path} is not an {SPEC_FORMAT!r} controller spec and declares no "
                     f"recognisable policy shape; refusing rather than guessing")
