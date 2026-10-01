"""BEHAVIORAL contract tests: a declared binding is NOT proof that the action runs.

WHY THIS FILE EXISTS SEPARATELY FROM `test_executor_capability.py`
-----------------------------------------------------------------
That file tests core's RULES -- ghosts, inert eta, disabled cells -- and core cannot do better than
check that an adapter names a binding. A porter can satisfy `binding="anything"` and pass. Core must
not try to close that hole by reading the host's source: it would couple core to one benchmark's hook
names, which is precisely the dependency the two-tier genericity guard forbids.

So the hole is closed at the ADAPTER level, by BEHAVIOUR, and this file is the reusable shape:

    1. run the host mechanism with the controller ABSENT   -> record what happened
    2. run it with the controller INSTALLED and firing     -> record what happened
    3. assert the two differ IN THE WAY THE ACTION CLAIMS

Step 3 is the whole point. "Telemetry says it fired" is not step 3 -- the inert-action defect (D3) had
a firing flag AND byte-identical trajectories. What distinguishes a real executor is an observable
change in what the agent runs or sees.

COLLABORATORS: `run_boundary_contract` below is the reusable harness. A TauBench or AppWorld adapter
supplies a `BoundaryProbe` for each capability it declares, and the parametrized test at the bottom
runs the same assertions against it. See docs/ADAPTER_GUIDE.md section "Proving your executor runs".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.learning.executor_capability import ExecutorCapability, supports

GATE = IncisionPoint.POST_GENERATION_PRE_EXEC


# ================================================================================================
# THE REUSABLE HARNESS
# ================================================================================================

@dataclass
class BoundaryProbe:
    """One executable claim about a host: "this action, at this boundary, does THIS".

    `run(controller)` must execute the host's real boundary mechanism and return an OBSERVABLE
    record -- what was dispatched, what the agent saw. `controller=None` is the control.

    `asserts_effect(control, treated)` states the claimed difference. It must compare OBSERVABLE
    consequences, never a telemetry flag: a firing flag next to an unchanged trajectory is the exact
    shape of the inert-action defect this harness exists to catch.
    """

    name: str
    boundary: str
    action: str
    run: Callable[[Any], Any]
    asserts_effect: Callable[[Any, Any], bool]
    controller: Any = None
    effect_claim: str = ""


def run_boundary_contract(probe: BoundaryProbe) -> tuple[bool, str]:
    """Execute the probe's control and treated runs and check the claimed effect.

    Returns (ok, detail). `ok=False` with "no observable difference" is the finding that matters: the
    action is declared, it may even report firing, and it changes nothing.
    """
    control = probe.run(None)
    treated = probe.run(probe.controller)
    if control == treated:
        return False, (f"{probe.boundary}/{probe.action}: the trajectory is IDENTICAL with and "
                       f"without the controller, so nothing the action claims reached the host. "
                       f"A declared binding and a firing flag are both consistent with this.")
    if not probe.asserts_effect(control, treated):
        return False, (f"{probe.boundary}/{probe.action}: the run DIFFERED but not as claimed "
                       f"({probe.effect_claim}); a difference in some other respect is not evidence "
                       f"for this action")
    return True, f"{probe.boundary}/{probe.action}: {probe.effect_claim} -- observed"


# ================================================================================================
# A TOY HOST WITH A REAL DISPATCH LIST. Suppression here is a genuine mechanism, not a declaration.
# ================================================================================================

class ToyDispatchHost:
    """A host with a commitment gate: proposed calls are filtered, then dispatched.

    Small on purpose, and it mirrors the one structural fact that made the real hook correct: the
    controller reads the DISPATCH LIST (what will actually run), not the decode.
    """

    def __init__(self):
        self.dispatched: list[str] = []
        self.observed: list[str] = []
        self.budget_spent: dict[str, int] = {}

    def step(self, episode: str, proposed: Sequence[str], controller=None) -> dict:
        """One turn. Returns an OBSERVABLE record: what ran and what the agent saw."""
        keep, withheld = [], []
        for call in proposed:
            fired = bool(controller and controller.fires_on({"proposed_call": call}))
            (withheld if fired else keep).append(call)

        if withheld and controller is not None:
            # PER-EPISODE BUDGET, the semantics the real executor imposes. It must persist across
            # steps of one episode -- reading it from per-step state means it never binds, and an
            # unbounded loop looks bounded. On exhaustion the withheld call DISPATCHES: the arm
            # degrades to the control rather than silently dropping the write.
            budget = int((getattr(controller, "eta", {}) or {}).get("retry_budget", 1) or 1)
            spent = self.budget_spent.get(episode, 0)
            if spent < budget:
                self.budget_spent[episode] = spent + 1
            else:
                keep = list(proposed)
                withheld = []

        for call in keep:
            self.dispatched.append(call)
            self.observed.append(f"RESULT({call})")
        return {"dispatched": list(keep), "observed": list(self.observed)}


class _Ctl:
    """A controller that fires on a payload-size predicate, with declared eta."""

    def __init__(self, threshold=10, eta=None, name="toy_ctl"):
        self.threshold, self.eta, self.name = threshold, dict(eta or {}), name

    def fires_on(self, state):
        return len(str(state.get("proposed_call", ""))) > self.threshold


def _suppress_probe(**eta):
    def run(controller):
        """ONE step per episode. A second step on the same episode would exhaust the budget and
        dispatch -- correct executor behaviour, but it would make the probe measure the degradation
        path instead of the suppression it claims. The budget semantics get their own test below."""
        h = ToyDispatchHost()
        long_call = "write(payload=" + "x" * 40 + ")"
        return h.step("ep1", [long_call, "read(k)"], controller=controller)

    return BoundaryProbe(
        name="toy suppress", boundary=GATE.value, action="suppress",
        run=run,
        controller=_Ctl(eta=eta or {"retry_budget": 1}),
        effect_claim="the proposed call is absent from what the host dispatched",
        asserts_effect=lambda c, t: (len(t["dispatched"]) < len(c["dispatched"])
                                     and not any("payload" in d for d in t["dispatched"])))


# ================================================================================================
# THE TESTS
# ================================================================================================

def test_a_real_suppress_executor_PASSES_the_behavioral_contract():
    """The positive: the call is genuinely absent from the dispatch list."""
    ok, detail = run_boundary_contract(_suppress_probe())
    assert ok, detail
    assert "observed" in detail


def test_a_TELEMETRY_ONLY_executor_FAILS_the_behavioral_contract():
    """D3's shape, reproduced on the toy host: the flag is set and nothing changes.

    This is the test that a declaration-based check cannot express. The capability names a binding,
    the controller fires, a flag is recorded -- and the trajectory is byte-identical, because the
    mechanism only writes telemetry. That was true of the real upstream reprompt cell for 13/13
    episodes, and it must FAIL here.
    """

    class TelemetryOnlyHost(ToyDispatchHost):
        def step(self, episode, proposed, controller=None):
            self.flags = {}
            for call in proposed:
                if controller and controller.fires_on({"proposed_call": call}):
                    # The entire "intervention": a flag. Exactly what the inert cell did.
                    self.flags["fired"] = True
                    self.flags["instruction"] = (getattr(controller, "eta", {}) or {}).get(
                        "instruction", "")
            for call in proposed:
                self.dispatched.append(call)
                self.observed.append(f"RESULT({call})")
            return {"dispatched": list(proposed), "observed": list(self.observed)}

    def run(controller):
        h = TelemetryOnlyHost()
        return h.step("ep1", ["write(payload=" + "x" * 40 + ")"], controller=controller)

    probe = BoundaryProbe(
        name="telemetry-only reprompt", boundary=GATE.value, action="reprompt", run=run,
        controller=_Ctl(eta={"instruction": "please reconsider", "retry_budget": 1}),
        effect_claim="the instruction changes what the agent runs or sees",
        asserts_effect=lambda c, t: c != t)

    ok, detail = run_boundary_contract(probe)
    assert not ok, "a telemetry-only mechanism must FAIL the behavioral contract"
    assert "IDENTICAL" in detail
    assert "firing flag" in detail


def test_a_run_that_differs_in_the_WRONG_way_also_fails():
    """A difference is necessary, not sufficient. The claimed effect is what is under test."""

    def run(controller):
        h = ToyDispatchHost()
        # The controller changes something irrelevant: it appends a note, and dispatches everything.
        out = h.step("ep1", ["write(payload=" + "x" * 40 + ")"], controller=None)
        if controller is not None:
            out = dict(out, observed=out["observed"] + ["NOTE(controller was here)"])
        return out

    probe = BoundaryProbe(
        name="wrong-effect suppress", boundary=GATE.value, action="suppress", run=run,
        controller=_Ctl(eta={"retry_budget": 1}),
        effect_claim="the proposed call is absent from what the host dispatched",
        asserts_effect=lambda c, t: not any("payload" in d for d in t["dispatched"]))
    ok, detail = run_boundary_contract(probe)
    assert not ok and "not as claimed" in detail


def test_the_PER_EPISODE_budget_is_the_semantics_the_capability_declares():
    """The budget must persist across steps of one episode, and exhaustion degrades to control.

    Both halves matter and both were once wrong. Read from per-step state, the budget resets every
    step and never binds -- an unbounded withhold loop that looks bounded. And on exhaustion the
    withheld call must DISPATCH, or the arm silently drops the operation instead of ceding to control.
    """
    h = ToyDispatchHost()
    ctl = _Ctl(eta={"retry_budget": 1})
    long_call = "write(payload=" + "x" * 40 + ")"

    first = h.step("ep1", [long_call], controller=ctl)
    assert first["dispatched"] == [], "the first firing must withhold"
    assert h.budget_spent["ep1"] == 1

    second = h.step("ep1", [long_call], controller=ctl)
    assert second["dispatched"] == [long_call], \
        "on exhaustion the withheld call must DISPATCH -- the arm degrades to control"

    other = h.step("ep2", [long_call], controller=ctl)
    assert other["dispatched"] == [], "the budget is PER EPISODE, not global"


def test_a_capability_declaring_a_budget_it_does_not_consume_is_refused():
    """Ties the behaviour above back to the declaration: the budget is part of the contract.

    The toy executor's semantics depend on `retry_budget`, so the capability must list it in
    `consumes`. One that does not is claiming budget-free suppression while running a budget.
    """
    honest = ExecutorCapability(
        boundary=GATE.value, action="suppress", binding="ToyDispatchHost.step:withhold",
        consumes=("retry_budget",), signals=("saw_alarm",), eta_is_computed=True)
    ok, _ = supports(honest, boundary=GATE, action=Action.SUPPRESS, signal="saw_alarm",
                     eta={"retry_budget": 1}, required_eta=())
    assert ok

    lying = ExecutorCapability(
        boundary=GATE.value, action="suppress", binding="ToyDispatchHost.step:withhold",
        consumes=(), signals=("saw_alarm",), fixed={"retry_budget": 1})
    bad, why = supports(lying, boundary=GATE, action=Action.SUPPRESS, signal="saw_alarm",
                        eta={"retry_budget": 5}, required_eta=())
    assert not bad and "retry_budget" in why


# ================================================================================================
# THE PARAMETRIZED SHAPE COLLABORATORS EXTEND
# ================================================================================================

ADAPTER_PROBES: tuple[BoundaryProbe, ...] = (
    _suppress_probe(),
)


@pytest.mark.parametrize("probe", ADAPTER_PROBES, ids=lambda p: p.name)
def test_every_declared_capability_proves_itself_behaviorally(probe):
    """Each capability an adapter declares must have a probe that PASSES.

    A TauBench or AppWorld porter appends their probes to `ADAPTER_PROBES` (or parametrizes this test
    over their own list) and gets the same assertions. A capability with no probe is a capability whose
    binding nobody checked.
    """
    ok, detail = run_boundary_contract(probe)
    assert ok, detail
