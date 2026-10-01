"""BFCL's RegressionHost: drives accepted controllers through the real dispatch logic.

WHY THIS FILE EXISTS. `RegressionSuite` against a mock host proves the protocol works; it proves
nothing about whether an accepted controller still executes in BFCL. The gap is exactly where our
failures lived:

  * a branch placed after a turn-ending exit, so the predicate was never reached (2 GPU rounds lost)
  * `ANCHOROPT_CONTROLLER_SPEC` set without `ANCHOROPT_PHI_MODULE`, so the arm silently became control
  * a spec whose capability_id named a different executor than the one that would run
  * a trajectory sidecar allowlist that dropped the firing keys, making a live controller look inert

WHAT IS REAL HERE, AND WHAT IS NOT. Real: the installer from `scripts/install_controller.py` (the same
code the GPU runner uses), the host's own declared predicates from `bfcl_signals`, the phase-eligibility
rule, the capability-identity gate, and the dispatch GUARD CONDITIONS transcribed from the patches that
run on the cluster. Not real: the model, the store objects and the evaluator process -- those need the
isolated runtime, so L2 here is a faithful replay of the DECISION, and L3 remains the GPU's job.

That division is deliberate and is the honest claim: this closes "can the predicate still be built,
installed, reached and dispatched to the right executor", which is what regressed on us repeatedly. It
does not claim to prove an end-to-end store mutation.

THE GUARD CONDITIONS ARE THE POINT. Transcribing them is what catches an unreachable branch: the A4
gate requires `_query_tool_calls == 0` AND `not _zero_call_reprompted` AND `step_count <
max_steps_per_turn`, and the capacity gate requires `not _capacity_repair_fired`. A test that calls
`fires_on` directly skips all of them, which is precisely how a controller can pass L1 and be dead in
the evaluator.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from anchoropt.learning.golden_registry import AcceptedController, PairedOutcome
from anchoropt.learning.regression_protocol import (
    BACKEND_INCOMPATIBLE, CAPABILITY_IDENTITY_MISMATCH, GROUNDING_ABSENT, L1_CONTRACT,
    OBSERVATION_UNAVAILABLE, OPERATION_INFEASIBLE, PHASE_MISMATCH, Certificate)
from anchoropt.learning.constraint_feasibility import INFEASIBLE, UNKNOWN

_REPO = Path(__file__).resolve().parent.parent.parent
_HERE = Path(__file__).resolve().parent

# FLAT IMPORTS, matching this adapter's own convention: bfcl_signals does `import adapter`, so its
# directory must be on sys.path. tests/test_bfcl_runtime.py does the same. Importing it as a package
# submodule fails on that inner import.
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bfcl_signals                                                   # noqa: E402
from bfcl_constraints import bfcl_constraint_contract, constraint_for_signal   # noqa: E402


def _load_installer():
    """The REAL installer module, by path -- not a reimplementation.

    `scripts/` is not a package, so it is loaded by spec. If this import fails the certificate says so
    rather than the suite quietly passing: a missing installer means no controller can be built at all.
    """
    p = _REPO / "scripts" / "install_controller.py"
    spec = importlib.util.spec_from_file_location("_anchoropt_install_controller_regr", p)
    if spec is None or spec.loader is None:      # pragma: no cover - defensive
        raise ImportError(f"cannot load the real installer at {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _phase_eligible(ctl: Any, is_prereq: bool) -> bool:
    """The live evaluator's rule, mirrored. SYMMETRIC -- see tests/test_prereq_guard.py."""
    ph = str(getattr(ctl, "phase", "query")).lower()
    if ph == "any":
        return True
    return ph == ("prereq" if is_prereq else "query")


# The dispatch guards, transcribed from the patches that run on the cluster. Keyed by boundary.
#
#   post_generation_pre_exec  patches/bv/bv_generic_zero_call_gate.py
#   post_execution            patches/bv/bv_generic_capacity_repair.py
def _gate_open(boundary: str, state: Mapping[str, Any]) -> tuple[bool, str]:
    """Would the host even REACH the predicate on this state? Returns (open, why_not)."""
    if state.get("disable_gates"):
        return False, "self.disable_gates is set: the host runs no gates at all"
    if boundary == "post_generation_pre_exec":
        if state.get("_zero_call_reprompted"):
            return False, "the host's once-per-turn latch is already spent (_zero_call_reprompted)"
        if int(state.get("tool_calls_so_far", 0) or 0) != 0:
            return False, ("the host requires _query_tool_calls == 0 at this branch; a step that made "
                           "a call never reaches it")
        if int(state.get("step_count", 0) or 0) >= int(state.get("max_steps_per_turn", 1) or 1):
            return False, "step_count >= max_steps_per_turn: the turn is over"
        return True, ""
    if boundary == "post_execution":
        if state.get("_capacity_repair_fired"):
            return False, "the host's per-episode latch is already spent (_capacity_repair_fired)"
        return True, ""
    return True, ""


class BfclRegressionHost:
    """Adapter-side RegressionHost for the golden registry.

    L1 contract, and L2 smoke through the real installer + real predicates + real guard conditions.
    L3 raises NotImplementedError -> an explicit SKIP, because a paired measurement needs the GPU
    runtime. A bland pass there would be the exact dishonesty this protocol exists to prevent.
    """

    def __init__(self, installer=None) -> None:
        self.seam = _ensure_declared_signal_seam()
        self._installer = installer or _load_installer()

    # -- L1 -----------------------------------------------------------------------------------
    def contract_check(self, entry: AcceptedController) -> Sequence[Certificate]:
        out: list[Certificate] = []
        ident = entry.identity
        spec = dict(entry.spec or {})

        # 1. the declared signal must still exist and still be legal at this boundary
        sig = ident.signal
        if sig not in getattr(bfcl_signals, "SIGNALS", {}):
            out.append(Certificate(
                L1_CONTRACT, entry.name, OBSERVATION_UNAVAILABLE,
                f"signal {sig!r} is no longer declared by the adapter",
                remedy="restore the predicate in bfcl_signals.SIGNALS or supersede this entry"))
        else:
            allowed = getattr(bfcl_signals, "SIGNAL_BOUNDARIES", {}).get(sig)
            if allowed and ident.boundary not in allowed:
                out.append(Certificate(
                    L1_CONTRACT, entry.name, OBSERVATION_UNAVAILABLE,
                    f"signal {sig!r} is no longer legal at {ident.boundary!r} (declared: {allowed})",
                    remedy="a boundary change invalidates the acceptance; revalidate or supersede"))

        # 2. the operator must still be feasible FOR THE CONSTRAINT (the KV lesson)
        op = str(spec.get("eta", {}).get("operator") or ident.operator or "")
        base_op = op.split(":")[0] if op else ""
        cons = constraint_for_signal(sig)
        if cons and base_op:
            v = bfcl_constraint_contract().check(base_op, cons)
            if v.status == INFEASIBLE:
                out.append(Certificate(
                    L1_CONTRACT, entry.name, OPERATION_INFEASIBLE, v.detail, remedy=v.remedy))
            elif v.status == UNKNOWN and base_op in ("reduce_preserving_facts",):
                out.append(Certificate(
                    L1_CONTRACT, entry.name, OPERATION_INFEASIBLE,
                    f"feasibility of {base_op!r} against {cons!r} is undeclared: {v.detail}",
                    remedy=v.remedy))

        # 3. grounding: eta must still be present for an action that needs it
        if ident.action in ("reroute",) and not spec.get("eta"):
            out.append(Certificate(
                L1_CONTRACT, entry.name, GROUNDING_ABSENT,
                "a reroute with no grounded eta cannot execute",
                remedy="restore the grounded eta in the spec"))
        if ident.action == "reprompt":
            if not str((spec.get("eta") or {}).get("instruction") or "").strip():
                out.append(Certificate(
                    L1_CONTRACT, entry.name, GROUNDING_ABSENT,
                    "a reprompt with no instruction injects nothing",
                    remedy="restore eta.instruction"))

        # 4. capability identity must be intact and must match the spec
        if spec.get("capability_id") and spec["capability_id"] != ident.capability_id:
            out.append(Certificate(
                L1_CONTRACT, entry.name, CAPABILITY_IDENTITY_MISMATCH,
                f"spec names {spec['capability_id']!r} but identity says {ident.capability_id!r}",
                remedy="an identity mismatch means the wrong executor may run; fix the spec"))

        # 5. the phase must survive the round trip through the REAL installer
        try:
            ctl = self._build(entry)
        except Exception as exc:
            out.append(Certificate(
                L1_CONTRACT, entry.name, GROUNDING_ABSENT,
                f"the real installer refused this spec: {type(exc).__name__}: {exc}",
                remedy="the spec no longer installs; fix it or supersede the entry"))
            return out
        if ident.phase and str(getattr(ctl, "phase", "")) != ident.phase:
            out.append(Certificate(
                L1_CONTRACT, entry.name, PHASE_MISMATCH,
                f"installed phase {getattr(ctl, 'phase', None)!r} != accepted {ident.phase!r}",
                remedy="phase travels with the controller; a drift changes when it may fire"))

        # 6. backend: the cell the acceptance was measured on must still be the grounded one
        cell = str((entry.provenance.env or {}).get("ANCHOROPT_CELL") or "")
        if cell and cell not in str(entry.provenance.split or ""):
            out.append(Certificate(
                L1_CONTRACT, entry.name, BACKEND_INCOMPATIBLE,
                f"ANCHOROPT_CELL={cell!r} does not match split {entry.provenance.split!r}",
                remedy="a grounding measured on one backend must not be reported on another"))
        return out

    # -- L2 -----------------------------------------------------------------------------------
    def smoke_execute(self, entry: AcceptedController,
                      state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Install for real, apply the host's guards, then ask the real predicate.

        `acted` requires the capability identity to match, mirroring bv_capability_identity_repair:
        a predicate firing at a cell whose executor is someone else's is NOT this intervention.
        """
        ctl = self._build(entry)
        st = dict(state)
        is_prereq = str(st.get("phase") or entry.identity.phase) == "prereq"

        if not _phase_eligible(ctl, is_prereq):
            return {"fired": False, "acted": False, "capability_id": "",
                    "detail": f"phase {ctl.phase!r} not eligible (is_prereq={is_prereq})"}
        open_, why = _gate_open(entry.identity.boundary, st)
        if not open_:
            return {"fired": False, "acted": False, "capability_id": "",
                    "detail": f"host gate closed: {why}"}
        try:
            fired = bool(ctl.fires_on(st))
        except Exception as exc:
            # A predicate that throws must never read as a trigger -- the hook's own contract.
            return {"fired": False, "acted": False, "capability_id": "",
                    "detail": f"predicate raised {type(exc).__name__}: {exc}"}
        if not fired:
            # DISTINGUISH "false" FROM "raised". The real `fires_on` returns False on an exception --
            # correct at runtime ("a raising phi is NOT a trigger") but in a verification path it
            # hides a broken predicate as a clean non-firing. So re-evaluate unguarded.
            try:
                ctl._eval(ctl._pred, dict(st))
            except Exception as exc:
                return {"fired": False, "acted": False, "capability_id": "",
                        "detail": (f"predicate RAISED {type(exc).__name__}: {exc} -- the real "
                                   f"fires_on swallows this and reports False")}
            return {"fired": False, "acted": False, "capability_id": "",
                    "detail": "the declared predicate did not fire on this state"}

        cap = str((entry.spec or {}).get("capability_id") or entry.identity.capability_id or "")
        acted, detail = self._would_act(entry, st)
        return {"fired": True, "acted": acted, "capability_id": cap, "detail": detail}

    def _would_act(self, entry: AcceptedController, st: Mapping[str, Any]) -> tuple[bool, str]:
        """Would the executor actually do something, given what it requires?

        Mirrors the two real executors' own preconditions. A firing predicate whose executor then
        declines is the `DID_NOT_ACT` case, and it is a real failure mode: measured once as 32/32
        "transform_executed_ok" while only 4 writes landed.
        """
        eta = dict((entry.spec or {}).get("eta") or {})
        if entry.identity.action == "reprompt":
            msg = str(eta.get("instruction") or "").strip()
            return (bool(msg), f"injects {len(msg)} chars" if msg
                    else "no instruction: nothing would be injected")
        if entry.identity.action == "reroute":
            want = entry.identity.capability_id
            got = str((entry.spec or {}).get("capability_id") or "")
            if want and got and want != got:
                return False, f"capability identity gate would refuse: {got!r} != {want!r}"
            if not eta:
                return False, "no grounded eta: the executor has nothing to apply"
            return True, f"would apply {eta.get('operator') or entry.identity.operator!r}"
        return True, "no executor precondition declared"

    # -- L3 -----------------------------------------------------------------------------------
    def paired_measure(self, entry: AcceptedController) -> PairedOutcome:
        raise NotImplementedError(
            "a paired measurement needs the isolated BV runtime, the model and the benchmark split; "
            "run the arm and control on GPU and compare with compare_outcomes()")

    # -- internals ----------------------------------------------------------------------------
    def _build(self, entry: AcceptedController):
        """Build the controller through the REAL installer, exactly as the runner would."""
        spec = dict(entry.spec or {})
        spec.setdefault("predicate", {"declared_signal": entry.identity.signal, "params": {}})
        spec.setdefault("phase", entry.identity.phase or "query")
        return self._installer.SpecPredicate(spec)


def _ensure_declared_signal_seam() -> str:
    """Make `anchoropt.memory_gates.evaluate_signal` resolvable from THIS adapter's predicates.

    THE DEFECT THIS CLOSES, found by L2 the first time it ran. `install_controller` resolves a declared
    signal with `from anchoropt.memory_gates import evaluate_signal`. On the cluster that module is
    installed by `patches/bv/bfcl_declared_signals.py`. In this repository it does not exist in core --
    correctly, since it is benchmark-specific -- and `anchoropt.memory_gates` instead resolves to
    `benchmarks/bfcl_v4/evaluator/memory_gates.py`, which has no `evaluate_signal`. The import raises,
    and the real `fires_on` turns that into `False`: an installed controller that can never fire and
    reports a clean non-firing while doing it.

    So the ADAPTER supplies the seam from its own `bfcl_signals.SIGNALS` -- the same predicates the BV
    patch mirrors -- rather than leaving the meaning of a declared signal to module-name luck. An
    unknown name RAISES, because "this host does not implement that signal" and "the signal did not
    fire" must never look the same.

    Returns a short description of what was wired, for the record.
    """
    import types

    import anchoropt as _core

    class SignalUnavailable(Exception):
        pass

    def evaluate_signal(name, state, params=None):
        fn = getattr(bfcl_signals, "SIGNALS", {}).get(str(name))
        if fn is None:
            raise SignalUnavailable(
                f"{name!r} is not a declared signal here; known: "
                f"{sorted(getattr(bfcl_signals, 'SIGNALS', {}))}")
        return bool(fn(dict(state or {}), dict(params or {})))

    mod = sys.modules.get("anchoropt.memory_gates")
    if mod is not None and hasattr(mod, "evaluate_signal"):
        return f"already resolvable: {getattr(mod, '__file__', '<synthetic>')}"
    shim = types.ModuleType("anchoropt.memory_gates")
    shim.evaluate_signal = evaluate_signal
    shim.SignalUnavailable = SignalUnavailable
    shim.SIGNALS = dict(getattr(bfcl_signals, "SIGNALS", {}))
    shim.__doc__ = ("declared-signal seam supplied by the BFCL adapter for offline regression; "
                    "on the cluster this module comes from patches/bv/bfcl_declared_signals.py")
    sys.modules["anchoropt.memory_gates"] = shim
    setattr(_core, "memory_gates", shim)
    return f"wired {len(shim.SIGNALS)} adapter predicates into anchoropt.memory_gates"
