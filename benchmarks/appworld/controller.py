"""The installed controller, and the (de)serialization that lets it cross a process boundary.

WHY THIS FILE EXISTS SEPARATELY FROM `adapter.py`
--------------------------------------------------
`run_round.py` runs pass 1 (build arms, emit a manifest, exit) and pass 2 (load results, select) as
SEPARATE process invocations, per the plan's measurement seam. Whatever candidate signal a boundary's
Phi expansion synthesized in pass 1 -- an `Atom` or a `Conjunction` over this adapter's own declared
fields -- lives only in that process's `ADAPTER.expanded` dict and is gone when it exits. Scoring an
arm for real (running AppWorld episodes with the candidate installed, in `agent_hooks.py`) happens in
YET ANOTHER process. So the predicate structure has to be written to disk and rebuilt, twice: once for
the scoring run, once for pass 2's own re-derivation of the same search state.

THE `falsy` DIVERGENCE, AND WHY IT IS RESOLVED HERE RATHER THAN IN `scripts/install_controller.py`
----------------------------------------------------------------------------------------------------
That script rebuilds a predicate from `{field, op, value}` JSON via its own `_OPS` dispatch table,
and `_OPS["falsy"]` is `lambda v, x: v is False` -- exactly `False`, never merely falsy. For a string
field like `error_kind` (empty string `""` when nothing failed), `"" is False` is `False`: a falsy
atom over `error_kind` built through that script can never fire, silently. `signal_grammar.Atom`'s own
`falsy` branch is `not bool(v)`, which is correct for the string/enum fields this adapter declares.

Both this adapter and `agent_hooks.py`'s agent subclass run in the same Python process as each other
(never in the same process as core's search, across the pass-1/pass-2 boundary, but always in the same
process as EACH OTHER at scoring time). So there is no need for a second operator table at all: a
spec's `{field, op, value}` is handed straight to the REAL `anchoropt.learning.signal_grammar.Atom`,
and `.evaluate(state)` is core's own method, not a re-derived copy of it. This is the resolution the
plan's pre-implementation verification settled on, and it is a reportable finding about
`install_controller.py`, not a defect in this file to work around by duplicating its bug.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from anchoropt.learning.signal_grammar import Atom, Conjunction

from adapter import ADAPTER

# ================================================================================================
# ARM -> SPEC (JSON-safe), never parsed from the signal's own name
# ================================================================================================
#
# A synthesized signal's name is truncated (the expansion path caps it at 60 chars), so parsing intent
# back out of it would be lossy. `Atom` carries `(field, op, value)` structurally and `Conjunction`
# carries `.terms`, so translating the OBJECT is exact where parsing the name would not be. This
# mirrors `scripts/self_evolve_cycle2.py:controller_spec`'s own translation, so a spec this adapter
# writes and one core's own reference driver writes are structurally identical.

def _recover_predicate_object(name: str) -> Any | None:
    """The `Atom`/`Conjunction` behind an EXPANDED signal's installed callable, recovered without any
    adapter-side bookkeeping.

    Core installs `lambda st, _p=pred: bool(_p.evaluate(st))` when a synthesized predicate is accepted
    (`anchoropt/learning/signal_expansion.py:277`) -- the object survives as that lambda's own default
    argument. Returns `None` for the two shipped (declared) signals, which are adapter-hardcoded logic
    in `evaluate_signal`, not grammar objects, and for anything else with no such closure.
    """
    fn = ADAPTER.expanded.get(name)
    if fn is None:
        return None
    obj = (getattr(fn, "__defaults__", None) or (None,))[0]
    return obj if obj is not None and hasattr(obj, "evaluate") else None


def _atom_to_spec(a: Atom) -> dict:
    out = {"field": a.field, "op": a.op}
    if a.value is not None:
        out["value"] = a.value
    return out


def predicate_to_spec(pred_obj: Any) -> dict:
    """An `Atom` or `Conjunction` object, translated to JSON-safe data."""
    terms = getattr(pred_obj, "terms", None)
    return {"all": [_atom_to_spec(t) for t in terms]} if terms else _atom_to_spec(pred_obj)


def spec_for_arm(arm: Any) -> dict:
    """Everything a fresh process needs to install and run one candidate arm live.

    A DECLARED signal has no grammar object -- `predicate` names itself and is resolved by NAME
    through `ADAPTER.evaluate_signal` on reload. An EXPANDED signal's `predicate` is the recovered
    STRUCTURE. Either way this is the single record `run_round.py` writes into `controllers.json` and
    the scoring process reads back through `controller_from_spec` below.
    """
    obj = _recover_predicate_object(arm.signal)
    predicate = predicate_to_spec(obj) if obj is not None else {"declared_signal": arm.signal}
    return {
        "arm_label": arm.label,
        "boundary": arm.boundary.value,
        "action": arm.action.value,
        "signal": arm.signal,
        "operator": arm.instantiated.operator.value,
        "variant": arm.instantiated.variant,
        "eta": {k: (v if isinstance(v, (int, float, bool, str)) else str(v))
                for k, v in arm.eta.items()},
        "capability_id": getattr(arm, "capability_id", "") or "",
        "predicate": predicate,
    }


# ================================================================================================
# SPEC -> live predicate, built from core's REAL grammar classes
# ================================================================================================

def _atom_from_spec(d: Mapping[str, Any]) -> Atom:
    return Atom(field=str(d["field"]), op=str(d["op"]), value=d.get("value"))


def predicate_from_spec(spec: Mapping[str, Any]) -> Any:
    """The inverse of `predicate_to_spec`. Builds a real `Atom`/`Conjunction` -- never a hand-copied
    operator table -- so `.evaluate` is exactly core's own method, `falsy` branch included."""
    if "all" in spec:
        return Conjunction(terms=tuple(_atom_from_spec(t) for t in spec["all"]))
    return _atom_from_spec(spec)


@dataclass
class Controller:
    """The installed policy an `agent_hooks.py` mechanism fires against live state.

    Shape matches `examples/toy_host/toy_host/runtime.py:Controller` exactly: `.boundary`/`.action`
    are locus/action VALUES (strings), `.eta` is what a `ground_*` method proposed, and `.fires_on`
    fails closed on a raising predicate rather than treating an error as a firing.
    """

    boundary: str
    action: str
    predicate: Callable[[Mapping[str, Any]], bool]
    eta: Mapping[str, Any] = field(default_factory=dict)
    label: str = ""

    def fires_on(self, state: Mapping[str, Any]) -> bool:
        try:
            return bool(self.predicate(state))
        except Exception:
            return False


def controller_from_spec(spec: Mapping[str, Any]) -> Controller:
    """Reconstruct a live `Controller` from one `spec_for_arm` record.

    The ONLY reconstruction path, whether the spec was just built in this process or loaded from
    `controllers.json` in a fresh one -- so there is exactly one way a spec becomes a runnable
    predicate, and the `falsy` fix above applies uniformly to both.
    """
    pred_spec = dict(spec.get("predicate") or {})
    declared = pred_spec.get("declared_signal")
    if declared:
        predicate = lambda st, _n=declared: bool(ADAPTER.evaluate_signal(_n, st))  # noqa: E731
    else:
        obj = predicate_from_spec(pred_spec)
        predicate = lambda st, _p=obj: bool(_p.evaluate(st))  # noqa: E731
    return Controller(
        boundary=str(spec.get("boundary") or ""),
        action=str(spec.get("action") or ""),
        predicate=predicate,
        eta=dict(spec.get("eta") or {}),
        label=str(spec.get("arm_label") or spec.get("label") or ""))


def controller_for_arm(arm: Any) -> Controller:
    """Same-process shortcut: a live `Controller` straight from a just-built arm, no JSON round trip.
    Used by `probes.py` and any local (non-cross-process) scoring, mirroring
    `examples/toy_host/demo.py:Evaluator._controller_for`."""
    return controller_from_spec(spec_for_arm(arm))
