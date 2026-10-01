"""EXECUTOR CAPABILITY: the negative tests. A declaration is not an executor.

THE DEFECT THESE PIN
--------------------
`executor_supports()` was a declaration lookup, so a cell's PRESENCE was accepted as proof that
executing code existed. On this project's own host it was not: the commitment-gate suppress cell named
a remedy flag that appears ZERO times in the hook that actually withholds the call. The contract
validated a name; the runtime ran something else; both reported success.

Three failure classes are closed here, and each gets a test that FAILS LOUDLY rather than one that
confirms the happy path:

    GHOST      a capability naming no executing site
    INERT ETA  a contract-required parameter no executor reads
    INERT ACTION  a cell whose parameter provably does not reach the agent

EVERYTHING HERE RUNS ON A TOY HOST defined in this file. That is deliberate and it is the portability
claim: the rules are core rules, so they must be demonstrable without the one real benchmark. A
separate test (`test_upstream_attribution.py`) pins the real adapter's declarations against the real
hook.
"""

from __future__ import annotations

import pytest

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.learning.action_contract import CONTRACTS, Operator
from anchoropt.learning.anchor_policy_opt import (
    REJECT_NO_EXECUTOR, AnchorPolicyOpt, SearchSpaceProposal,
)
from anchoropt.learning.executor_capability import (
    DISABLED, ETA_NOT_CONSUMED, GHOST_CAPABILITY, NO_CAPABILITY, PARAM_CONFLICT,
    SIGNAL_UNSUPPORTED, ExecutorCapability, disabled_capabilities, supports,
    unbound_capabilities,
)
from anchoropt.runtime import HostProfile

GATE = IncisionPoint.POST_GENERATION_PRE_EXEC
POST = IncisionPoint.POST_EXECUTION


def _cap(**kw):
    base = dict(boundary=GATE.value, action="suppress", binding="toy.hook:withhold",
                consumes=("retry_budget",), signals=("saw_alarm",))
    base.update(kw)
    return ExecutorCapability(**base)


# ================================================================================================
# 1. GHOST EXECUTORS
# ================================================================================================

def test_a_capability_with_no_binding_is_rejected_as_a_ghost():
    """The exact shape of the real defect: a declared cell naming no executing code.

    This is the test whose absence made the original bug invisible. There was no field that could be
    wrong, so nothing could be checked.
    """
    ghost = _cap(binding="")
    ok, why = supports(ghost, boundary=GATE, action=Action.SUPPRESS, signal="saw_alarm", eta={})
    assert not ok
    assert GHOST_CAPABILITY in why
    assert "binding" in why, "the rejection must name the missing field, not just refuse"


def test_a_binding_of_only_whitespace_is_still_a_ghost():
    """A porter satisfying the field with a space has not named an executor."""
    ok, why = supports(_cap(binding="   "), boundary=GATE, action=Action.SUPPRESS,
                       signal="saw_alarm", eta={})
    assert not ok and GHOST_CAPABILITY in why


def test_an_absent_capability_is_distinguished_from_a_ghost_one():
    """`no_executor` and `unbound` are different facts and lead to different fixes.

    Absent: this host cannot do it -- a porter implements the mechanism.
    Ghost:  someone declared it without naming the code -- a porter finds the site or deletes the cell.
    """
    ok, why = supports(None, boundary=GATE, action=Action.SUPPRESS, signal="saw_alarm", eta={})
    assert not ok and NO_CAPABILITY in why
    assert GHOST_CAPABILITY not in why


def test_unbound_capabilities_audits_every_ghost_in_one_call():
    """The audit a porter runs, and the one that would have caught the real ghost."""
    caps = (_cap(), _cap(action="reprompt", binding=""), _cap(action="reroute", binding=""))
    assert unbound_capabilities(caps) == (
        f"{GATE.value}/reprompt: no binding", f"{GATE.value}/reroute: no binding")


# ================================================================================================
# 2. INERT ETA
# ================================================================================================

def test_a_contract_required_field_no_executor_reads_is_rejected():
    """The inert-eta class: an arm carrying a parameter nothing consumes measures something else.

    REPROMPT requires `instruction`. An executor that reads only `retry_budget` cannot be steered by
    the instruction, so an arm claiming to test an instruction would in fact test the control.
    """
    cap = _cap(action="reprompt", consumes=("retry_budget",), signals=("saw_alarm",))
    ok, why = supports(cap, boundary=GATE, action=Action.REPROMPT, signal="saw_alarm",
                       eta={"instruction": "try again", "retry_budget": 1},
                       required_eta=("instruction", "retry_budget"))
    assert not ok
    assert ETA_NOT_CONSUMED in why
    assert "instruction" in why, "the rejection must name the unconsumed field"


def test_eta_is_computed_is_a_DECLARED_exemption_not_an_assumed_one():
    """An executor that derives its parameters from live state consumes no candidate eta.

    This exemption is load-bearing and its absence was a bug I introduced and caught here: without it,
    every REROUTE arm on the real host was rejected, because that executor synthesizes its replacement
    call from runtime state and reads none of destination/argument_mapping/retry_semantics. Those arms
    are NOT inert -- the intervention runs -- so the contract eta is the arm's identity, not executor
    input. The adapter must SAY so; a cell that declares neither `consumes` nor `eta_is_computed` is
    still refused.
    """
    computed = _cap(action="reroute", consumes=(), eta_is_computed=True)
    ok, why = supports(computed, boundary=GATE, action=Action.REROUTE, signal="saw_alarm",
                       eta={"destination": "backup", "argument_mapping": "same",
                            "retry_semantics": "replace"},
                       required_eta=("destination", "argument_mapping", "retry_semantics"))
    assert ok, why

    undeclared = _cap(action="reroute", consumes=(), eta_is_computed=False)
    ok2, why2 = supports(undeclared, boundary=GATE, action=Action.REROUTE, signal="saw_alarm",
                         eta={"destination": "backup"}, required_eta=("destination",))
    assert not ok2 and ETA_NOT_CONSUMED in why2
    assert "eta_is_computed" in why2, "the rejection must tell a porter the exemption exists"


def test_an_ENFORCED_claim_is_refused_when_no_executor_enforces_it():
    """`preservation` was the string "verify an equivalent copy survives" that NOTHING read.

    A safety clause no executor enforces is not a clause. An arm may omit it; an arm that DECLARES it
    is making a runtime claim and needs an executor behind it.
    """
    # The executor DOES consume the required field, so the only thing left unbacked is the enforced
    # claim -- otherwise the earlier required-eta check fires and the test proves nothing about
    # enforcement.
    cap = _cap(consumes=("suppressed_operation", "retry_budget"))
    ok, why = supports(cap, boundary=GATE, action=Action.SUPPRESS, signal="saw_alarm",
                       eta={"suppressed_operation": "the write",
                            "preservation": "verify a copy survives"},
                       required_eta=("suppressed_operation",), enforced_eta=("preservation",))
    assert not ok
    assert ETA_NOT_CONSUMED in why
    assert "preservation" in why, f"must name the unenforced claim, got: {why}"
    assert "not a constraint" in why or "do not claim it" in why


def test_the_same_arm_WITHOUT_the_unenforced_claim_is_accepted():
    """The remove-outright variant preserves nothing and needs to preserve nothing.

    This is the pair that shows the rule targets the CLAIM, not the family: drop the clause nobody
    enforces and the identical arm is materializable.
    """
    cap = _cap(consumes=("retry_budget",), eta_is_computed=True)
    ok, why = supports(cap, boundary=GATE, action=Action.SUPPRESS, signal="saw_alarm",
                       eta={"suppressed_operation": "the write"},
                       required_eta=("suppressed_operation",), enforced_eta=("preservation",))
    assert ok, why


def test_eta_is_computed_does_NOT_exempt_an_enforced_claim():
    """Deriving parameters from state says nothing about enforcing a safety clause."""
    cap = _cap(consumes=(), eta_is_computed=True)
    ok, why = supports(cap, boundary=GATE, action=Action.SUPPRESS, signal="saw_alarm",
                       eta={"suppressed_operation": "the write", "preservation": "a copy survives"},
                       required_eta=("suppressed_operation",), enforced_eta=("preservation",))
    assert not ok and ETA_NOT_CONSUMED in why


# ================================================================================================
# 3. INERT ACTIONS, SIGNAL COVERAGE, FIXED PARAMETERS
# ================================================================================================

def test_a_host_disabled_cell_is_refused_WITH_its_reason():
    """An action whose parameter does not reach the agent must not be measured, and must say why.

    Silently pruning it is not good enough: an absent cell is indistinguishable from one nobody
    thought of, which is how an inert action gets re-added by the next porter.
    """
    cap = _cap(action="reprompt", consumes=("instruction",),
               disabled_reason="the instruction is written to telemetry and never injected")
    ok, why = supports(cap, boundary=GATE, action=Action.REPROMPT, signal="saw_alarm",
                       eta={"instruction": "x"}, required_eta=("instruction",))
    assert not ok
    assert DISABLED in why and "never injected" in why


def test_disabled_capabilities_audit_lists_the_reason():
    caps = (_cap(), _cap(action="reprompt", disabled_reason="inert: telemetry only"))
    assert disabled_capabilities(caps) == (f"{GATE.value}/reprompt: inert: telemetry only",)


def test_an_expanded_signal_reaches_an_agnostic_cell_but_NOT_a_keyed_one():
    """Expansion must compose with materializability, or Phi grows into unreachable cells.

    And the fall-through matters: an early return here once exempted every expanded signal from
    parameter validation -- the one population that most needs it, since a synthesized predicate
    carries a data-derived threshold.
    """
    agnostic = _cap(signals=("saw_alarm",), signal_agnostic=True, eta_is_computed=True)
    ok, why = supports(agnostic, boundary=GATE, action=Action.SUPPRESS, signal="synth_1",
                       eta={}, expanded_signals=frozenset({"synth_1"}))
    assert ok and "EXPANDED" in why

    keyed = _cap(signals=("saw_alarm",), signal_agnostic=False)
    ok2, why2 = supports(keyed, boundary=GATE, action=Action.SUPPRESS, signal="synth_1",
                         eta={}, expanded_signals=frozenset({"synth_1"}))
    assert not ok2 and SIGNAL_UNSUPPORTED in why2


def test_an_expanded_signal_is_STILL_parameter_validated():
    """The fall-through, pinned: a fixed-parameter conflict must bite an expanded signal too."""
    cap = _cap(signals=("saw_alarm",), signal_agnostic=True, eta_is_computed=True,
               fixed={"retry_budget": 1})
    ok, why = supports(cap, boundary=GATE, action=Action.SUPPRESS, signal="synth_1",
                       eta={"retry_budget": 7}, expanded_signals=frozenset({"synth_1"}))
    assert not ok and PARAM_CONFLICT in why


def test_a_parameter_the_executor_fixes_may_not_be_silently_coerced():
    cap = _cap(fixed={"retry_budget": 1}, eta_is_computed=True)
    ok, why = supports(cap, boundary=GATE, action=Action.SUPPRESS, signal="saw_alarm",
                       eta={"retry_budget": 3})
    assert not ok and PARAM_CONFLICT in why
    ok2, _ = supports(cap, boundary=GATE, action=Action.SUPPRESS, signal="saw_alarm",
                      eta={"retry_budget": 1})
    assert ok2, "the honoured value must pass"


# ================================================================================================
# 4. THE A3/E1 SPLIT -- preserved, per-variant, no new action family
# ================================================================================================

def test_SUPPRESS_keeps_ONE_family_with_TWO_variant_contracts():
    """Both accepted variants stay valid and neither is weakened. No WITHHOLD action is added."""
    assert {a.value for a in Action} == {"noop", "reprompt", "suppress", "reroute"}, \
        "no new action family may be introduced -- that was the `transform` mistake"

    c = CONTRACTS[Operator.SUPPRESS]
    # remove-outright: the family floor only. It preserves nothing and is not asked to.
    assert c.required_for("cancel_proposed") == ("suppressed_operation",)
    # withhold-and-replay: preservation is the DEFINING property, so it is mandatory.
    assert "preservation" in c.required_for("withhold_and_replay")
    assert "preservation" in c.enforced, "and it must be executor-backed, not merely present"


def test_the_replay_variant_cannot_omit_the_field_that_distinguishes_it():
    """Weakening SUPPRESS globally would have let a replay arm drop its own defining claim."""
    from anchoropt.learning.action_contract import InstantiatedAction, validate

    bad = InstantiatedAction(operator=Operator.SUPPRESS, variant="withhold_and_replay",
                             eta={"suppressed_operation": "the write"})
    ok, why = validate(bad)
    assert not ok and "preservation" in why

    good = InstantiatedAction(operator=Operator.SUPPRESS, variant="withhold_and_replay",
                              eta={"suppressed_operation": "the write",
                                   "preservation": "the recorded result is replayed verbatim"})
    assert validate(good)[0]


# ================================================================================================
# 5. THE CHECK IS REACHED FROM `build_arms`, on a TOY host
# ================================================================================================

class _ToyHostRuntime:
    """A minimal adapter exercising the capability hook. Deliberately not the real benchmark."""

    def __init__(self, cap):
        self._cap = cap
        self.HOST = HostProfile(name="toy_cap_host",
                                executable={GATE: frozenset({Action.SUPPRESS})})

    def declared_signals(self):
        return ("saw_alarm",)

    def evaluate_signal(self, signal, state, params=None):
        return bool(state.get(signal))

    def executor_capability(self, boundary, action):
        return self._cap

    def expanded_signal_names(self):
        return ()

    def ground_suppress(self, signal, boundary):
        return [{"variant": "cancel_proposed",
                 "eta": {"suppressed_operation": "the proposed write"},
                 "detail": "remove the call before dispatch"}]


def _prop():
    return SearchSpaceProposal(boundary=GATE, signal="saw_alarm",
                               action_set=(Action.SUPPRESS,), diagnosis_case_ids=("c1",))


def test_build_arms_refuses_a_ghost_and_RECORDS_the_rejection():
    """End to end: a ghost cell yields zero arms and a named rejection, before any evaluation."""
    rt = _ToyHostRuntime(_cap(binding="", eta_is_computed=True))
    arms, rejected = AnchorPolicyOpt(runtime=rt, host=rt.HOST).build_arms(_prop())
    assert arms == []
    assert [r.reason_code for r in rejected] == [REJECT_NO_EXECUTOR]
    assert GHOST_CAPABILITY in rejected[0].detail


def test_build_arms_accepts_the_same_cell_once_it_names_its_executor():
    """The paired positive: the ONLY difference is a real binding."""
    rt = _ToyHostRuntime(_cap(binding="toy.hook:withhold", eta_is_computed=True))
    arms, rejected = AnchorPolicyOpt(runtime=rt, host=rt.HOST).build_arms(_prop())
    assert len(arms) == 1, rejected
    assert arms[0].action is Action.SUPPRESS


def test_an_adapter_with_NO_capability_hook_still_works_via_the_legacy_one():
    """COMPATIBILITY. Existing adapters implement `executor_supports` and must keep working.

    Core prefers `executor_capability` because it can CHECK that form -- binding, consumed eta, the
    disabled flag. Through the legacy hook it can only pass the adapter's own verdict through, which is
    exactly the weakness being retired; the fallback exists so a port is not broken by the upgrade, not
    because the two are equivalent.
    """

    class _LegacyOnly:
        HOST = HostProfile(name="legacy_host", executable={GATE: frozenset({Action.SUPPRESS})})

        def declared_signals(self):
            return ("saw_alarm",)

        def evaluate_signal(self, signal, state, params=None):
            return bool(state.get(signal))

        def ground_suppress(self, signal, boundary):
            return [{"variant": "cancel_proposed",
                     "eta": {"suppressed_operation": "the proposed write"},
                     "detail": "remove the call before dispatch"}]

        def executor_supports(self, boundary, action, signal, eta):
            return True, "legacy adapter verdict"

    lo = _LegacyOnly()
    assert not hasattr(lo, "executor_capability")
    arms, rejected = AnchorPolicyOpt(runtime=lo, host=lo.HOST).build_arms(_prop())
    assert len(arms) == 1, rejected


def test_the_legacy_hook_can_still_REFUSE_and_the_refusal_is_recorded():
    """A legacy adapter's own rejection must still reach the ledger with the executor reason code."""

    class _LegacyRefuses:
        HOST = HostProfile(name="legacy_no", executable={GATE: frozenset({Action.SUPPRESS})})

        def declared_signals(self):
            return ("saw_alarm",)

        def evaluate_signal(self, signal, state, params=None):
            return bool(state.get(signal))

        def ground_suppress(self, signal, boundary):
            return [{"variant": "cancel_proposed",
                     "eta": {"suppressed_operation": "the proposed write"}, "detail": "x"}]

        def executor_supports(self, boundary, action, signal, eta):
            return False, "no_executor: this legacy host cannot run it"

    lr = _LegacyRefuses()
    arms, rejected = AnchorPolicyOpt(runtime=lr, host=lr.HOST).build_arms(_prop())
    assert arms == []
    assert [r.reason_code for r in rejected] == [REJECT_NO_EXECUTOR]


def test_an_adapter_declaring_NEITHER_hook_is_not_blocked_from_porting():
    """A first port has declared no executors yet; admissibility already bounded it."""

    class _Bare:
        HOST = HostProfile(name="bare", executable={GATE: frozenset({Action.SUPPRESS})})
        declared_signals = _ToyHostRuntime.declared_signals
        evaluate_signal = _ToyHostRuntime.evaluate_signal
        ground_suppress = _ToyHostRuntime.ground_suppress

    b = _Bare()
    arms, _rej = AnchorPolicyOpt(runtime=b, host=b.HOST).build_arms(_prop())
    assert len(arms) == 1
