"""EXECUTOR CAPABILITY: the check that an action can actually RUN at a boundary.

THE DEFECT THIS FIXES
---------------------
`executor_supports()` used to be a DECLARATION LOOKUP. An adapter published a dict of cells, and a
cell's presence was accepted as proof that executing code existed. It is not proof. The BFCL adapter
declared `("post_generation_pre_exec", "suppress")` with `remedy_flag="enable_redundant_write_
suppress"` -- and that flag appears ZERO times in the hook that actually withholds the call. The hook
gates on installed controllers and their `fires_on`, never on the flag. So the contract validated a
name, the runtime ran something else, and both reported success.

Two distinct facts were being conflated, and this module separates them:

    ADMISSIBILITY     `feasible_actions(l)` -- could this action exist at this boundary at all?
                      Structural, benchmark-independent, already in `anchor.py`.
    MATERIALIZABILITY is there EXECUTING CODE at this boundary that will run this action, and can it
                      consume this eta unchanged? Host-specific, and what this module checks.

WHAT AN ADAPTER MUST NOW DECLARE
--------------------------------
A capability is only accepted if it names, for that (boundary, action):

    binding     the executing site -- a module path, function, patch, or gate name. A capability with
                no binding is a GHOST: rejected, because nothing has been shown to run.
    consumes    the eta keys the executing code actually READS. This is the fix for inert eta: a
                required contract field that no executor consumes is a field the proposal cannot
                influence, so an arm carrying it measures something other than what it declares.

`consumes` is deliberately an allowlist rather than a denylist. An adapter that forgets a key it does
read gets a loud rejection naming the key; an adapter that over-declares would silently re-admit the
inert-eta class, which is the defect being closed.

WHY `verified` IS NOT A BOOLEAN THE ADAPTER SETS FREELY
-------------------------------------------------------
An adapter could declare `binding="anything"` and pass. That is accepted and deliberate: core cannot
read the host's source. What core CAN do is refuse a capability that names nothing, force the eta
consumption to be written down where a test can pin it against the real hook, and make the
declaration auditable -- `unbound_capabilities()` lists every cell whose binding is absent. The
BFCL ghost was invisible precisely because no field existed to be wrong.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Rejection codes. Distinct per cause, because "no executor" and "the executor cannot read this
# parameter" lead to different next steps for a porter.
NO_CAPABILITY = "no_executor_capability"
GHOST_CAPABILITY = "executor_capability_unbound"
ETA_NOT_CONSUMED = "executor_cannot_consume_eta"
SIGNAL_UNSUPPORTED = "executor_signal_unsupported"
PARAM_CONFLICT = "executor_parameter_conflict"
DISABLED = "executor_disabled_by_host"


@dataclass(frozen=True)
class ExecutorCapability:
    """One (boundary, action) cell an adapter claims its host can actually run.

    `binding` and `consumes` are the two fields whose absence made the ghost possible. `disabled_
    reason`, when set, is a host declaring that the cell exists in code but must NOT be measured --
    the honest state for an action whose instruction never reaches the agent.
    """

    boundary: str
    action: str
    # WHICH executor this is, when a cell has more than one. Opaque to core: it is carried into the
    # arm and the emitted spec so the runner installs THE capability core judged feasible, not
    # whichever one the host happens to resolve first. Must describe the MECHANISM, never name an
    # accepted controller -- an id like "a4_zero_call" would ship the answer key to the learner.
    # Defaults to the cell name, which is correct while a cell has exactly one capability.
    capability_id: str = ""
    # WHICH OPERATOR READING THIS EXECUTOR IMPLEMENTS, when its action has more than one.
    #
    # `Action.REROUTE` covers both `substitute` (replace the proposed call) and `transform` (reshape the
    # state a later decision reads). Those are different mechanisms, and a cell can hold one executor for
    # each. Without this field `resolve` sees only the ACTION, so both readings collapse to whichever
    # capability is declared first and supports the signal.
    #
    # MEASURED: on this host, post_execution/REROUTE declares a reduce executor, a relocate executor and
    # a signal-agnostic default. A substitute arm and a transform arm on `container_at_capacity` BOTH
    # resolved to `relocate_entry_preserving_information_then_retry`, so two emitted specs carried an
    # identity their operator does not implement. Under the host's identity gate those would be refused
    # at dispatch -- the arm silently becoming the control while being reported as an intervention.
    #
    # Empty means "this executor does not distinguish operator readings", which is correct for a cell
    # with one mechanism and preserves every existing adapter unchanged.
    operators: tuple[str, ...] = ()
    binding: str = ""
    consumes: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()
    fixed: Mapping[str, Any] = field(default_factory=dict)
    signal_agnostic: bool = False
    eta_is_computed: bool = False
    # ETA THE EXECUTOR DEMANDS, and the values it accepts. `consumes` says which keys the code READS;
    # this says which it will REFUSE TO RUN WITHOUT.
    #
    # The gap it closes, measured: this project's post_execution hook begins
    #     if _prim != "additional_read_and_merge": return {"fired": False, ...}
    # while its capability declared `eta_is_computed=True, consumes=()` -- a positive claim that the
    # executor derives everything from live state. Four grounded arms therefore passed every
    # feasibility check, installed cleanly, and would each have reported fired=False and run as the
    # CONTROL: a measured zero against an intervention that never ran.
    #
    # `eta_is_computed` does NOT exempt this. That flag says a candidate cannot STEER the executor's
    # parameters; it says nothing about a key the executor requires before it will act at all.
    #
    #     requires_eta={"primitive": ("additional_read_and_merge",)}   one of these values
    #     requires_eta={"primitive": ()}                               any non-empty value
    requires_eta: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    # WHERE each eta key is DELIVERED, when the executor does not read it from the candidate directly.
    #
    # A policy-template channel is legitimate: this host's reprompt executor reads its instruction
    # from `templates.on_turn_start_action`, and an earlier round measured a real effect through it.
    # What is NOT legitimate is leaving the channel undeclared -- then two arms differing only in
    # `instruction` execute identically unless the runner happens to write that text into the policy,
    # which is the inert-eta defect wearing a different hat.
    #
    # Declaring it makes the delivery CHECKABLE: a preflight can assert the emitted spec's value
    # actually reached the slot before any GPU time is spent.
    #
    #     eta_delivered_via={"instruction": "templates.on_turn_start_action"}
    eta_delivered_via: Mapping[str, str] = field(default_factory=dict)
    # Eta keys this executor CANNOT honour, with the reason. A candidate carrying one is REFUSED
    # rather than run with the parameter quietly ignored -- an arm that asks for a retry budget the
    # executor fixes by a latch is not the arm that would run.
    unsupported_eta: Mapping[str, str] = field(default_factory=dict)
    detail: str = ""
    disabled_reason: str = ""

    @property
    def cid(self) -> str:
        """The stable identifier for this executor. Falls back to the cell when unset."""
        return str(self.capability_id or f"{self.boundary}/{self.action}")

    @property
    def is_bound(self) -> bool:
        """Does this capability name executing code? An unbound cell is a ghost."""
        return bool(str(self.binding).strip())

    @property
    def is_enabled(self) -> bool:
        return not str(self.disabled_reason).strip()


def _v(x: Any) -> str:
    return str(getattr(x, "value", x))


def supports(cap: ExecutorCapability | None, *, boundary: Any, action: Any, signal: str,
             eta: Mapping[str, Any], expanded_signals: frozenset[str] = frozenset(),
             required_eta: tuple[str, ...] = (),
             enforced_eta: tuple[str, ...] = ()) -> tuple[bool, str]:
    """Can `cap` actually run (boundary, action) for `signal` with this `eta`?

    Ordered so the rejection names the FIRST thing missing, which is the one a porter fixes:
      1. a capability exists for the cell
      2. it is BOUND to executing code                          <- closes the ghost class
      3. the host has not disabled it                           <- closes the inert-action class
      4. it covers this signal (expansion may reach it)
      5. it CONSUMES every eta key the contract requires        <- closes the inert-eta class
      5b. it CONSUMES every ENFORCED key this arm actually carries
      6. every parameter it fixes matches what the candidate declares
    """
    b, a = _v(boundary), _v(action)
    if cap is None:
        return False, (f"{NO_CAPABILITY}: this host declares no executor for {b}/{a}, so the action "
                       f"is admissible but not materializable")
    if not cap.is_bound:
        return False, (f"{GHOST_CAPABILITY}: the capability for {b}/{a} names no executing site. A "
                       f"declaration is not an executor -- give it a `binding` naming the code that "
                       f"runs, or remove the cell")
    if not cap.is_enabled:
        return False, f"{DISABLED}: {b}/{a} is declared but not measurable -- {cap.disabled_reason}"

    expanded_ok = False
    covered = tuple(cap.signals or ())
    if covered and signal not in covered:
        # An EXPANDED signal may reach a cell whose mechanism reads live state rather than the
        # signal's value. Falls THROUGH to the checks below -- an early return here once exempted
        # every expanded signal from parameter validation.
        if signal in expanded_signals and cap.signal_agnostic:
            expanded_ok = True
        else:
            return False, (f"{SIGNAL_UNSUPPORTED}: the executor at {b}/{a} evaluates "
                           f"{list(covered)}, not {signal!r}")

    # THE INERT-ETA CHECK. A contract-required field that the executor does not read cannot be
    # influenced by the proposal, so an arm carrying it measures something it does not declare.
    #
    # `eta_is_computed` IS a real exemption, and getting this wrong is instructive: treating it as no
    # exemption rejected every REROUTE arm on this project's own host, because that executor
    # synthesizes its replacement call from LIVE STATE and therefore reads none of the candidate's
    # destination/argument_mapping/retry_semantics. Those arms are not inert -- the intervention
    # genuinely runs, and the contract eta is the arm's IDENTITY (which grounding produced it) rather
    # than executor input.
    #
    # The distinction that matters, and it is a declaration the adapter must make deliberately:
    #     consumes=(...)             the executor reads these keys FROM THE CANDIDATE
    #     eta_is_computed=True       the executor DERIVES its parameters from runtime state; contract
    #                                eta is provenance, and a candidate cannot steer it
    # A cell with neither is the ghost-eta case and is refused. What `eta_is_computed` must NOT
    # exempt is an `enforced` claim -- see the enforcement check below, which applies regardless.
    if not cap.eta_is_computed:
        unconsumed = tuple(k for k in required_eta if k not in tuple(cap.consumes or ()))
        if unconsumed:
            return False, (f"{ETA_NOT_CONSUMED}: the executor at {b}/{a} reads "
                           f"{list(cap.consumes)} but the action contract requires "
                           f"{list(unconsumed)}, which nothing at this boundary consumes. An arm "
                           f"carrying an eta field no executor reads measures a different "
                           f"intervention than the one proposed. If this executor derives its "
                           f"parameters from live state instead, declare eta_is_computed=True")

    # THE ENFORCEMENT CHECK. An `enforced` field is optional to GROUND but, once an arm carries it,
    # only means something if the executor reads it. The withhold-and-replay variant is the case this
    # closes: a remove-outright arm omits `preservation` and is fine; an arm that DECLARES it is
    # claiming a copy survives, and that claim needs an executor behind it rather than a sentence in a
    # dict.
    #
    # `eta_is_computed` does NOT exempt this. That flag says the executor chooses its own parameters,
    # which is compatible with running the intervention -- it says nothing about enforcing a safety
    # clause, and a computed-eta executor that does enforce one can simply list it in `consumes`.
    claimed = tuple(k for k in enforced_eta
                    if k in eta and eta[k] not in (None, "")
                    and k not in tuple(cap.consumes or ()))
    if claimed:
        return False, (f"{ETA_NOT_CONSUMED}: this arm declares {list(claimed)}, which the executor at "
                       f"{b}/{a} does not consume (it reads {list(cap.consumes)}). An unenforced "
                       f"safety or preservation clause is not a constraint -- either back it with an "
                       f"executor that reads the field, or do not claim it")

    # THE UNSUPPORTED-ETA CHECK. A parameter the executor cannot honour must not be silently ignored:
    # the arm that would run is not the arm that was proposed.
    for key, why_not in dict(cap.unsupported_eta or {}).items():
        if key in eta and eta[key] not in (None, ""):
            return False, (
                f"{ETA_NOT_CONSUMED}: this arm declares {key}={eta[key]!r}, which the executor at "
                f"{b}/{a} cannot honour -- {why_not}. Running it would measure a different "
                f"intervention than the one proposed")

    # THE REQUIRED-ETA CHECK. An executor that will not act without a key is not materializable for a
    # candidate that does not carry it -- and saying so HERE, at arm construction, is the difference
    # between a refusal with a reason and a silent no-op measured as a null.
    for key, allowed in dict(cap.requires_eta or {}).items():
        have = eta.get(key)
        if have in (None, ""):
            return False, (
                f"{ETA_NOT_CONSUMED}: the executor at {b}/{a} will not act without eta {key!r} "
                f"(it accepts {list(allowed) or 'any non-empty value'}); this arm does not carry it, "
                f"so the mechanism would report not-fired and run as the CONTROL")
        if allowed and have not in allowed:
            return False, (
                f"{PARAM_CONFLICT}: the executor at {b}/{a} accepts {key}={list(allowed)} but this "
                f"arm declares {key}={have!r}; it would report not-fired and run as the CONTROL")

    for key, want in dict(cap.fixed or {}).items():
        if key in eta and eta[key] != want:
            return False, (f"{PARAM_CONFLICT}: the executor at {b}/{a} fixes {key}={want!r} but this "
                           f"eta declares {key}={eta[key]!r}; coercing it would run different "
                           f"semantics than the ones proposed")

    why = f"materializable at {b}/{a} via {cap.binding}"
    if expanded_ok:
        why += f" with the EXPANDED signal {signal!r} (mechanism reads live state)"
    return True, why


def unbound_capabilities(caps) -> tuple[str, ...]:
    """Every declared cell that names no executing site. The audit that would have caught the ghost."""
    out = []
    for cap in caps or ():
        if not cap.is_bound:
            out.append(f"{cap.boundary}/{cap.action}: no binding")
    return tuple(sorted(out))


def disabled_capabilities(caps) -> tuple[str, ...]:
    """Every cell a host declares present-but-not-measurable, with its stated reason."""
    return tuple(sorted(f"{c.boundary}/{c.action}: {c.disabled_reason}"
                        for c in (caps or ()) if c.is_bound and not c.is_enabled))


# ================================================================================================
# MORE THAN ONE EXECUTOR PER CELL
# ================================================================================================
#
# A (boundary, action) cell is not one mechanism. On the BFCL host, `post_generation_pre_exec/
# reprompt` names TWO: one that appends a user message and regenerates, and one that writes the
# instruction to the step record and does nothing else. They have different bindings, different eta,
# and opposite validity -- the first is a real intervention, the second is inert and must not be
# measured.
#
# While the lookup was a dict keyed on the cell, the inert one MASKED the working one and a whole
# mechanism was unreachable for a bookkeeping reason rather than a validity one.
#
# TWO PROPERTIES THIS FUNCTION MUST HAVE, and the second is the one that makes it safe:
#
#   1. feasibility is per-CAPABILITY, not per-cell: a cell is usable if ANY of its executors can run
#      this (signal, eta), and a disabled sibling does not veto a working one;
#   2. it RETURNS THE CAPABILITY IT ACCEPTED. Answering "yes, something here works" and then letting
#      the host pick an executor would evaluate a different mechanism than the one core judged
#      feasible -- the arm-identity defect class, arrived at from a new direction. The caller carries
#      the returned `cid` into the arm and the emitted spec.


def resolve(caps, *, boundary: Any, action: Any, signal: str, eta: Mapping[str, Any],
            expanded_signals: frozenset[str] = frozenset(),
            required_eta: tuple[str, ...] = (),
            enforced_eta: tuple[str, ...] = (),
            operator: Any = None
            ) -> tuple[ExecutorCapability | None, str]:
    """The FIRST capability at this cell that can run (signal, eta), and why -- or None and why not.

    `caps` may be a single ExecutorCapability, an iterable of them, or None; a host that declares one
    executor per cell behaves exactly as before.

    Deterministic: capabilities are tried in the order the adapter declared them, so two runs of the
    same round resolve the same executor. When none qualifies, the returned reason is the FIRST
    rejection -- the one a porter fixes -- with the others appended, because "disabled" on one
    executor and "signal unsupported" on its sibling are different problems.
    """
    if caps is None:
        items: list[ExecutorCapability] = []
    elif isinstance(caps, ExecutorCapability):
        items = [caps]
    else:
        items = [c for c in caps if isinstance(c, ExecutorCapability)]

    if not items:
        return None, supports(None, boundary=boundary, action=action, signal=signal, eta=eta)[1]

    reasons: list[str] = []
    want_op = str(operator or "").strip().lower()

    # IS THIS CELL AMBIGUOUS? A cell is ambiguous when its executors do not agree on one operator
    # reading -- i.e. more than one distinct reading is declared across them, or some declare a reading
    # and others declare none. In an ambiguous cell an UNDECLARED executor cannot be shown to implement
    # the requested reading, so accepting it is a guess.
    #
    # WHY FAIL CLOSED HERE. The first fix filtered executors that declared a DIFFERENT reading, which
    # closed the measured defect but left the dual: a capability with `operators=()` beside one that
    # declares `("transform",)` remained a candidate for `substitute`, purely because it said nothing.
    # Constructed and confirmed: resolve() returned the undeclared executor for substitute at a cell whose
    # only other executor was transform-only. That is the same arm-identity failure wearing the opposite
    # hat -- an executor silently accepting a reading it may not implement.
    #
    # BACKWARD COMPATIBILITY IS PRESERVED EXACTLY WHERE IT IS SOUND. A cell whose executors declare
    # nothing at all is UNAMBIGUOUS by construction (one mechanism, or a host that has not yet declared
    # readings), and every such adapter keeps working untouched. Only a cell that is demonstrably mixed
    # requires each executor to say which reading it implements.
    _declared_sets = [tuple(sorted(str(o).lower() for o in (getattr(c, "operators", ()) or ())))
                      for c in items]
    _nonempty = {d for d in _declared_sets if d}
    ambiguous = bool(_nonempty) and (len(_nonempty) > 1 or any(not d for d in _declared_sets))

    for cap in items:
        # OPERATOR FILTER. An executor that names its readings is not a candidate for another reading,
        # even when it supports the signal and the eta.
        declared = tuple(str(o).lower() for o in (getattr(cap, "operators", ()) or ()))
        if want_op and declared and want_op not in declared:
            reasons.append(f"{cap.cid}: implements operator(s) {list(declared)}, not {want_op!r}")
            continue
        # FAIL CLOSED on an undeclared executor in an ambiguous cell.
        if want_op and not declared and ambiguous:
            reasons.append(
                f"{cap.cid}: declares no operator reading, and this cell is ambiguous "
                f"(readings declared here: {sorted(_nonempty)}); cannot establish that it implements "
                f"{want_op!r} -- declare `operators` on it")
            continue
        ok, why = supports(cap, boundary=boundary, action=action, signal=signal, eta=eta,
                           expanded_signals=expanded_signals, required_eta=required_eta,
                           enforced_eta=enforced_eta)
        if ok:
            return cap, f"{why} [capability {cap.cid}]"
        reasons.append(f"{cap.cid}: {why}")
    head = reasons[0]
    if len(reasons) > 1:
        head += "  (also tried -- " + "; ".join(reasons[1:]) + ")"
    return None, head
