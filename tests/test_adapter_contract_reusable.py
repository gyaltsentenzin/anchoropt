"""THE REUSABLE ADAPTER CONTRACT CHECKER -- tested on adapters that deliberately violate it.

A checker nobody has seen fail is a checker nobody should trust. Each negative case below builds an
adapter with exactly ONE defect and asserts the checker names it. The defects are the real ones this
project hit, not hypotheticals:

  * a boundary that declares no observable alphabet (core fails closed, so nothing can be synthesized)
  * `states_at` absent, so a predicate is validated against another boundary's states
  * an executor capability naming no executing code -- a GHOST
  * a capability that declares neither which eta it reads nor that it computes its own
  * a boundary key that maps to no locus, so core cannot tell WHERE it is

The checker is also run against BOTH shipped adapters, which must pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "examples" / "toy_host"), str(REPO / "benchmarks" / "bfcl_v4")):
    if p not in sys.path:
        sys.path.insert(0, p)

from anchoropt.anchor import Action, IncisionPoint                            # noqa: E402
from anchoropt.learning.executor_capability import ExecutorCapability         # noqa: E402
from anchoropt.runtime import HostProfile                                     # noqa: E402
from anchoropt.testing import (                                              # noqa: E402
    MANDATORY_HOOKS, OPTIONAL_HOOKS, AdapterContract, check_adapter_contract,
)

GATE = IncisionPoint.POST_GENERATION_PRE_EXEC
POST = IncisionPoint.POST_EXECUTION


class _Field:
    def __init__(self, name, type_, boundaries=()):
        self.name, self.type, self.boundaries = name, type_, tuple(boundaries)
        self.doc, self.enum = "", ()


class _Good:
    """A minimal adapter that satisfies the whole contract. Each negative case subclasses and breaks
    exactly one thing, so a failure can only be attributed to that one difference."""

    HOST = HostProfile(name="good", executable={GATE: frozenset({Action.SUPPRESS})})

    def __init__(self):
        self.expanded = {}

    def is_decision(self, event):
        return bool(event.get("kind"))

    def boundary_key(self, event):
        return {"propose": "gate"}.get(str(event.get("kind")), "")

    def boundary_from_key(self, key):
        return {"gate": GATE}.get(str(key))

    def synthesis_fields(self, boundary=None):
        if str(getattr(boundary, "value", boundary)) == GATE.value:
            return {"size": _Field("size", int, (GATE.value,))}
        return {}

    def declared_signals(self):
        return ("big",)

    def signal_boundaries(self, signal):
        return frozenset({GATE})

    def evaluate_signal(self, signal, state, params=None):
        return int(state.get("size") or 0) > 10

    def states_at(self, boundary, states):
        want = set(self.synthesis_fields(boundary))
        return [{k: v for k, v in st.items() if k in want} for st in states or ()]

    def install_signal(self, name, predicate, *, boundary=None, provenance=""):
        self.expanded[name] = predicate

    def expanded_signal_names(self):
        return tuple(self.expanded)

    def ground_suppress(self, signal, boundary):
        return [{"variant": "cancel", "eta": {"suppressed_operation": "the write"}, "detail": ""}]

    def executor_capability(self, boundary, action):
        if (str(getattr(boundary, "value", boundary)), str(getattr(action, "value", action))) == \
                (GATE.value, "suppress"):
            return ExecutorCapability(boundary=GATE.value, action="suppress",
                                      binding="host.step:withhold", consumes=("retry_budget",),
                                      signals=("big",), eta_is_computed=True)
        return None


_STATES = [{"size": 5}, {"size": 50}, {"size": 200}]
_EVENTS = [{"kind": "propose", "size": 200}, {"kind": "propose", "size": 5}]


# ================================================================================================
# THE POSITIVE CASES
# ================================================================================================

def test_a_complete_adapter_passes():
    rep = check_adapter_contract(_Good(), states=_STATES, events=_EVENTS)
    assert rep.ok, rep.summary()
    assert not rep.failures


def test_the_toy_host_reference_adapter_passes():
    """The example a collaborator copies must itself satisfy the contract."""
    import toy_adapter
    rep = check_adapter_contract(toy_adapter.ADAPTER)
    assert rep.ok, rep.summary()


def test_the_bfcl_adapter_passes():
    """The one real benchmark adapter, checked by the same generic rules.

    It is passed as the MODULE, which is how the driver passes it -- an adapter may be a module, a
    class or an instance, and the contract does not care which.
    """
    import bfcl_runtime
    rep = check_adapter_contract(bfcl_runtime)
    assert rep.ok, rep.summary()


def test_the_report_is_readable_and_names_the_hook():
    rep = check_adapter_contract(_Good(), states=_STATES, events=_EVENTS)
    text = rep.summary()
    assert "adapter contract:" in text
    for v in rep.violations:
        assert v.hook and v.severity in ("FAIL", "NOTE")


# ================================================================================================
# THE NEGATIVE CASES -- one defect each
# ================================================================================================

def test_a_missing_MANDATORY_hook_fails():
    class _NoDecision(_Good):
        is_decision = None

    rep = check_adapter_contract(_NoDecision())
    assert not rep.ok
    assert any(v.hook == "is_decision" for v in rep.failures), rep.summary()


def test_an_EMPTY_alphabet_at_every_boundary_fails():
    """Core fails closed here, so no condition can ever be synthesized."""

    class _NoFields(_Good):
        def synthesis_fields(self, boundary=None):
            return {}

    rep = check_adapter_contract(_NoFields())
    assert not rep.ok
    assert any("EVERY boundary returned an empty alphabet" in v.detail for v in rep.failures), \
        rep.summary()


def test_an_UNTYPED_field_fails():
    """Core synthesizes TYPED predicates; a field with no `.type` cannot host one."""

    class _Untyped(_Good):
        def synthesis_fields(self, boundary=None):
            if str(getattr(boundary, "value", boundary)) == GATE.value:
                return {"size": object()}
            return {}

    rep = check_adapter_contract(_Untyped())
    assert not rep.ok
    assert any("no `.type`" in v.detail for v in rep.failures), rep.summary()


def test_a_GHOST_executor_capability_fails():
    """THE REAL DEFECT. A declared cell naming no executing code."""

    class _Ghost(_Good):
        def executor_capability(self, boundary, action):
            cap = _Good.executor_capability(self, boundary, action)
            if cap is None:
                return None
            return ExecutorCapability(boundary=cap.boundary, action=cap.action, binding="",
                                      consumes=cap.consumes, signals=cap.signals,
                                      eta_is_computed=True)

    rep = check_adapter_contract(_Ghost())
    assert not rep.ok
    assert any("GHOST" in v.detail for v in rep.failures), rep.summary()


def test_a_capability_declaring_NEITHER_consumes_NOR_computed_fails():
    """Otherwise an arm may carry parameters nothing reads -- the inert-eta class."""

    class _Silent(_Good):
        def executor_capability(self, boundary, action):
            cap = _Good.executor_capability(self, boundary, action)
            if cap is None:
                return None
            return ExecutorCapability(boundary=cap.boundary, action=cap.action,
                                      binding=cap.binding, consumes=(), signals=cap.signals,
                                      eta_is_computed=False)

    rep = check_adapter_contract(_Silent())
    assert not rep.ok
    assert any("neither `consumes` nor" in v.detail for v in rep.failures), rep.summary()


def test_a_DISABLED_capability_is_exempt_and_reported_as_a_note():
    """An inert action's honest declaration is `consumes=()` plus a reason.

    It can never be measured, so no arm can carry an unread parameter -- but it must stay VISIBLE,
    because a silently absent executor reads as one nobody thought of.
    """

    class _Disabled(_Good):
        def executor_capability(self, boundary, action):
            cap = _Good.executor_capability(self, boundary, action)
            if cap is None:
                return None
            return ExecutorCapability(boundary=cap.boundary, action=cap.action,
                                      binding=cap.binding, consumes=(), signals=cap.signals,
                                      disabled_reason="the instruction is never injected")

    rep = check_adapter_contract(_Disabled())
    assert rep.ok, rep.summary()
    assert any("DISABLED" in v.detail for v in rep.notes), rep.summary()


def test_a_boundary_key_that_maps_to_NO_locus_fails():
    class _BadKey(_Good):
        def boundary_from_key(self, key):
            return None

    rep = check_adapter_contract(_BadKey(), events=_EVENTS)
    assert not rep.ok
    assert any(v.hook == "boundary_from_key" for v in rep.failures), rep.summary()


def test_events_that_are_never_decisions_fail():
    """Localization would find no boundary and every round would end BOUNDARY_NOT_REPAIRABLE."""

    class _NeverDecides(_Good):
        def is_decision(self, event):
            return False

    rep = check_adapter_contract(_NeverDecides(), events=_EVENTS)
    assert not rep.ok
    assert any(v.hook == "is_decision" for v in rep.failures), rep.summary()


def test_states_at_may_FILTER_but_never_INVENT():
    class _Inventing(_Good):
        def states_at(self, boundary, states):
            return list(states) + [{"size": 999}]

    rep = check_adapter_contract(_Inventing(), states=_STATES)
    assert not rep.ok
    assert any("never invent" in v.detail for v in rep.failures), rep.summary()


def test_a_HOST_declaring_string_actions_fails():
    """`Action` is core vocabulary; a parallel string enum silently matches nothing."""

    class _Strings(_Good):
        HOST = HostProfile(name="strings", executable={GATE: frozenset()})

        def __init__(self):
            super().__init__()
            object.__setattr__(self, "HOST", _Strings.HOST)

    class _FakeHost:
        name = "fake"

        def executable_actions(self, point):
            return {"suppress"} if point is GATE else set()

    class _BadHost(_Good):
        HOST = _FakeHost()

    rep = check_adapter_contract(_BadHost())
    assert not rep.ok
    assert any("not a core Action" in v.detail for v in rep.failures), rep.summary()


def test_a_HOST_with_no_executable_action_anywhere_fails():
    class _Empty(_Good):
        HOST = HostProfile(name="empty", executable={})

    rep = check_adapter_contract(_Empty())
    assert not rep.ok
    assert any("no executable action at ANY boundary" in v.detail for v in rep.failures), \
        rep.summary()


# ================================================================================================
# MANDATORY vs OPTIONAL is a declared, testable distinction
# ================================================================================================

def test_missing_OPTIONAL_hooks_are_notes_and_never_failures():
    """A FIRST PORT must be able to pass. Optional hooks unlock capability, not correctness."""

    class _Bare:
        HOST = HostProfile(name="bare", executable={GATE: frozenset({Action.SUPPRESS})})
        is_decision = _Good.is_decision
        boundary_key = _Good.boundary_key
        boundary_from_key = _Good.boundary_from_key
        synthesis_fields = _Good.synthesis_fields
        declared_signals = _Good.declared_signals
        signal_boundaries = _Good.signal_boundaries
        evaluate_signal = _Good.evaluate_signal

    rep = check_adapter_contract(_Bare())
    assert rep.ok, rep.summary()
    missing = {v.hook for v in rep.notes}
    # The consequential ones must be SPELLED OUT, not merely listed as absent.
    assert any("Phi CANNOT be expanded" in v.detail for v in rep.notes)
    assert any("NO candidate can be built" in v.detail for v in rep.notes)
    assert any("materializability is UNCHECKED" in v.detail for v in rep.notes)
    assert "states_at" in missing


def test_the_two_hook_lists_are_disjoint_and_documented():
    m = {h for h, _ in MANDATORY_HOOKS}
    o = {h for h, _ in OPTIONAL_HOOKS}
    assert not (m & o), f"a hook cannot be both mandatory and optional: {m & o}"
    for hook, why in MANDATORY_HOOKS + OPTIONAL_HOOKS:
        assert why and len(why) > 20, f"{hook} needs a real explanation, got {why!r}"


# ================================================================================================
# THE PYTEST-FRIENDLY WRAPPER a collaborator subclasses
# ================================================================================================

def _toy():
    import toy_adapter
    return toy_adapter.ADAPTER


class TestToyHostViaWrapper(AdapterContract):
    """The shape a TauBench or AppWorld porter writes: two lines and the checks come free.

    This class existing and being COLLECTED is itself the test. An earlier version of
    `AdapterContract` was a dataclass, so pytest refused to collect the subclass and silently ran
    none of these checks -- a green run that verified nothing.
    """

    adapter = _toy()
    states = [{"payload_chars": 200, "boundary": "post_generation_pre_exec"},
              {"payload_chars": 5, "boundary": "post_generation_pre_exec"}]


def test_the_wrapper_subclass_is_actually_collected_by_pytest():
    """Guard the trap above: assert the mixin has no generated __init__."""
    assert "__init__" not in AdapterContract.__dict__, \
        "a generated __init__ makes pytest SKIP COLLECTION of every subclass, verifying nothing"
    names = [n for n in dir(TestToyHostViaWrapper) if n.startswith("test_")]
    assert len(names) >= 3, names
