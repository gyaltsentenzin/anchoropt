"""The intervention primitives this host can ACTUALLY materialize. A registry, not a synthesizer.

Shown to the proposer so it selects from what exists instead of inventing an implementation mechanism
the runtime then rejects. Deliberately NOT code synthesis: each entry describes an executor that is
already implemented and measured.

SOURCE OF TRUTH IS THE RUNTIME, NOT THIS FILE. `available()` reads `bfcl_runtime.EXECUTORS` and
`HOST.executable_actions` and reports what it finds; the descriptions here add the human-facing
constraint notes that a registry dict cannot carry. If an executor is removed from the runtime it
disappears from this listing -- a second hand-maintained list of what is executable is exactly how
`admissible` and `materializable` drifted apart in the first place.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Primitive:
    """One executable intervention pattern."""

    name: str
    action_family: str
    supported_loci: tuple[str, ...]
    required_signal: tuple[str, ...]          # empty = any signal the executor covers
    eta_schema: Mapping[str, str] = field(default_factory=dict)
    executor: str = ""
    known_constraints: tuple[str, ...] = ()

    def as_prompt_line(self) -> str:
        loci = ", ".join(self.supported_loci)
        sig = ", ".join(self.required_signal) or "(any covered signal)"
        eta = ", ".join(f"{k}: {v}" for k, v in dict(self.eta_schema).items()) or "(none)"
        out = [f"- {self.name} [{self.action_family}]",
               f"    loci      : {loci}",
               f"    signal    : {sig}",
               f"    eta       : {eta}",
               f"    executor  : {self.executor}"]
        for c in self.known_constraints:
            out.append(f"    CONSTRAINT: {c}")
        return "\n".join(out)


# Constraint notes are MEASURED facts, each traceable to a round, not general advice.
_CONSTRAINTS: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("post_generation_pre_exec", "reprompt"): (
        "retry_budget is FIXED at 1 by the executor; a candidate declaring another value is rejected, "
        "not coerced (se3a3 was rejected for exactly this).",
        "bounded to one injection per query; an unbounded version would loop on a model that keeps "
        "declining to call anything.",
        "fires on the model's FIRST decision, so there is no pre-intervention prefix to compare -- "
        "comparability must come from the target population, not from prestate matching.",
        "the executor is armed for the WHOLE run and also fires on PREREQ episodes, which write the "
        "store queries read; check prereq firings before quoting a paired delta.",
    ),
    ("post_execution", "reprompt"): (
        "retry_budget is FIXED at 1 by the executor.",
        "MEASURED NET-NEGATIVE at every theta for the low-similarity signal (R2, 9 thresholds): this "
        "framing tells the model to distrust a retrieval that was adequate.",
    ),
    ("post_execution", "reroute"): (
        "NO EXECUTOR on this host. Admissible in U_H(l) but not materializable -- 6 substitute arms "
        "plus 1 transform arm were never runnable, which inflated a reported '113 counterfactuals' "
        "by 91 non-materializable arms.",
    ),
    ("post_generation_pre_exec", "suppress"): (
        "no executor: grounding succeeds for proposal-naming signals but nothing can run it here.",
    ),
}

_ETA_SCHEMA: Mapping[str, Mapping[str, str]] = {
    "reprompt": {"instruction": "the text injected as a trailing user message",
                 "retry_budget": "int; the executor fixes this at 1"},
    "suppress": {"suppressed_operation": "which proposed operation is cancelled",
                 "preservation": "what guarantees safety before cancelling"},
    "substitute": {"destination": "a real tool of the same kind in another container",
                   "argument_mapping": "how the original call's args carry over",
                   "retry_semantics": "replace_original | retry_after"},
    "transform": {"target_surface": "which surface is rewritten",
                  "operator": "the rewrite performed",
                  "preservation": "what is retained"},
}


def available(runtime: Any) -> list[Primitive]:
    """Primitives the runtime can materialize, read FROM the runtime."""
    out: list[Primitive] = []
    executors = getattr(runtime, "EXECUTORS", {}) or {}
    for (locus, action), spec in sorted(executors.items()):
        out.append(Primitive(
            name=f"{action}@{locus}",
            action_family=action,
            supported_loci=(locus,),
            required_signal=tuple(spec.get("signals") or ()),
            eta_schema=_ETA_SCHEMA.get(action, {}),
            executor=str(spec.get("remedy_flag") or ""),
            known_constraints=_CONSTRAINTS.get((locus, action), ())))
    return out


def unavailable(runtime: Any) -> list[Primitive]:
    """Cells that are ADMISSIBLE in U_H(l) but have no executor.

    Reported to the proposer as explicitly unavailable, because "the host declares this legal" and
    "the host can run this" are different facts and the gap between them is where a round's evaluation
    budget goes to die.
    """
    from anchoropt.anchor import Action, IncisionPoint
    out: list[Primitive] = []
    host = getattr(runtime, "HOST", None)
    executors = getattr(runtime, "EXECUTORS", {}) or {}
    if host is None:
        return out
    for boundary in IncisionPoint:
        try:
            acts = host.executable_actions(boundary)
        except Exception:
            continue
        for a in sorted(acts, key=lambda x: x.value):
            if a is Action.NOOP:
                continue
            if (boundary.value, a.value) in executors:
                continue
            out.append(Primitive(
                name=f"{a.value}@{boundary.value}", action_family=a.value,
                supported_loci=(boundary.value,), required_signal=(),
                eta_schema=_ETA_SCHEMA.get(a.value, {}), executor="(NONE)",
                known_constraints=_CONSTRAINTS.get((boundary.value, a.value),
                                                   ("no executor on this host",))))
    return out


def prompt_block(runtime: Any) -> str:
    """The compact primitive listing shown to a role."""
    lines = ["AVAILABLE RUNTIME PRIMITIVES (select from these; do not invent a mechanism)"]
    for p in available(runtime):
        lines.append(p.as_prompt_line())
    un = unavailable(runtime)
    if un:
        lines.append("")
        lines.append("ADMISSIBLE BUT NOT MATERIALIZABLE (no executor -- proposing these wastes a round)")
        for p in un:
            lines.append(f"- {p.name}: {p.known_constraints[0] if p.known_constraints else 'no executor'}")
    return "\n".join(lines)
