"""Retain an UNREALIZABLE repair as an explicit observation/action REQUIREMENT, benchmark-independent.

THE GAP THIS CLOSES
-------------------
A repair the proposer cannot express today is recorded as prose and then lost: the round moves on, the
next round re-derives the same sentence, and nothing accumulates. Measured across this experiment's
rounds: **12 distinct unrealizable repairs, 11 of them blocked by "no declared signal observes this
condition"** -- a missing OBSERVATION, not a missing action. Yet the search kept reducing the capacity
family to "suppress the clear", which is a different and weaker intervention than the one asked for.

So the loss is not in the teacher's reasoning. It is that a repair hypothesis has no durable
representation between "prose the attributor wrote" and "a declared signal the host supplies".

WHAT THIS MODULE IS
-------------------
A typed record and a small set of rules. It holds, per repair:

    what must be OBSERVED        the condition, as a requirement on state fields
    WHERE it is observable       the decision boundary at which those fields can exist
    what OPERATION would apply   the state change the repair asks for
    which primitive could realize it   or None, with the reason
    the SAFETY contract          what must remain true, so a repair cannot be adopted without one

It contains no benchmark vocabulary, no signal names, no anchor identity, and no thresholds. It does
not decide anything: `unmet_requirements()` reports which requirements a host does not yet satisfy, and
core's existing feasibility checks and measured selection do the rest.

DIVISION OF LABOUR, unchanged
-----------------------------
The LLM supplies semantics -- which condition a repair implicates, at which boundary, and what operation
would implement it. This module records that in a checkable form and asks the adapter whether the host
can supply it. Feasibility, policy search and selection stay in core; nothing here scores or ranks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Why a repair is not realizable yet. Distinct causes lead to different next steps, which is the whole
# reason to keep them apart rather than counting "blocked".
NEEDS_OBSERVATION = "needs_observation"      # the condition cannot be seen at the boundary
NEEDS_OPERATION = "needs_operation"          # the state change is not an action the host can take
NEEDS_BOTH = "needs_both"
SATISFIED = "satisfied"                      # the host already supplies everything it asks for


@dataclass(frozen=True)
class AffordanceRequirement:
    """One repair hypothesis, expressed as what a controller would need in order to attempt it.

    `observation_fields` is the load-bearing field: a repair is not a signal name, it is a claim that
    some condition is decidable from state at some boundary. Writing that down as FIELD NAMES is what
    turns prose into something an adapter can answer.
    """

    repair: str                               # the attributor's own words, kept verbatim
    boundary: str                             # where the decision is taken
    observation_fields: tuple[str, ...] = ()  # what must be readable in state there
    operation: str = ""                       # the state change the repair asks for
    candidate_primitive: str = ""             # a host primitive that might realize it, or ""
    safety_contract: tuple[str, ...] = ()     # what must remain true for this to be adoptable
    evidence_case_ids: tuple[str, ...] = ()   # the failures that motivated it
    provider: str = ""                        # who proposed it (never trusted as authority)
    notes: str = ""

    def status(self, *, supplied_fields: Sequence[str], available_operations: Sequence[str]) -> str:
        """Which requirement this host does not yet meet. A pure function of declared capability."""
        missing_obs = [f for f in self.observation_fields if f not in set(supplied_fields)]
        has_op = (not self.operation) or self.operation in set(available_operations)
        if missing_obs and not has_op:
            return NEEDS_BOTH
        if missing_obs:
            return NEEDS_OBSERVATION
        if not has_op:
            return NEEDS_OPERATION
        return SATISFIED

    def missing_observations(self, supplied_fields: Sequence[str]) -> tuple[str, ...]:
        return tuple(f for f in self.observation_fields if f not in set(supplied_fields))

    def as_dict(self) -> dict:
        return {"repair": self.repair, "boundary": self.boundary,
                "observation_fields": list(self.observation_fields),
                "operation": self.operation, "candidate_primitive": self.candidate_primitive,
                "safety_contract": list(self.safety_contract),
                "evidence_case_ids": list(self.evidence_case_ids),
                "provider": self.provider, "notes": self.notes}


@dataclass
class AffordanceLedger:
    """Repair hypotheses that outlive the round that produced them.

    A round records what it could not express; a later round -- or a porter -- reads the ACCUMULATED
    requirements and can see which single observation would unblock the most evidence. Without this the
    same sentence is re-derived every round and nothing is ever unblocked on purpose.

    Deliberately not a framework: a list, a merge rule, and two queries.
    """

    requirements: list[AffordanceRequirement] = field(default_factory=list)

    def add(self, req: AffordanceRequirement) -> None:
        """Merge by (boundary, observation_fields, operation) -- the IDENTITY of the requirement.

        Two attributors wording the same repair differently must not create two requirements: what
        makes them the same is needing the same observation at the same boundary for the same
        operation, not saying it the same way. Evidence accumulates on the merged record, which is
        exactly the quantity a porter needs.
        """
        key = (req.boundary, tuple(sorted(req.observation_fields)), req.operation)
        for i, existing in enumerate(self.requirements):
            if (existing.boundary, tuple(sorted(existing.observation_fields)),
                    existing.operation) == key:
                merged_cases = tuple(dict.fromkeys(existing.evidence_case_ids
                                                   + req.evidence_case_ids))
                self.requirements[i] = AffordanceRequirement(
                    repair=existing.repair, boundary=existing.boundary,
                    observation_fields=existing.observation_fields,
                    operation=existing.operation,
                    candidate_primitive=existing.candidate_primitive or req.candidate_primitive,
                    safety_contract=tuple(dict.fromkeys(existing.safety_contract
                                                        + req.safety_contract)),
                    evidence_case_ids=merged_cases, provider=existing.provider,
                    notes=existing.notes)
                return
        self.requirements.append(req)

    def unmet(self, *, supplied_fields: Sequence[str],
              available_operations: Sequence[str]) -> list[tuple[AffordanceRequirement, str]]:
        """Every requirement this host does not satisfy, with its cause."""
        out = []
        for r in self.requirements:
            st = r.status(supplied_fields=supplied_fields,
                          available_operations=available_operations)
            if st != SATISFIED:
                out.append((r, st))
        return out

    def ranked_by_evidence(self, *, supplied_fields: Sequence[str],
                           available_operations: Sequence[str]) -> list[tuple[Any, str, int]]:
        """Unmet requirements ordered by HOW MUCH EVIDENCE they would unblock.

        Cost control only, and stated as such: it decides what a porter looks at first, never what is
        adopted. Adoption still requires a declared signal, a grounded action, and a measurement.
        """
        rows = [(r, st, len(r.evidence_case_ids))
                for r, st in self.unmet(supplied_fields=supplied_fields,
                                        available_operations=available_operations)]
        return sorted(rows, key=lambda t: (-t[2], t[0].boundary, t[0].repair[:40]))

    def as_dict(self) -> dict:
        return {"requirements": [r.as_dict() for r in self.requirements]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AffordanceLedger":
        led = cls()
        for r in (data or {}).get("requirements", []):
            led.requirements.append(AffordanceRequirement(
                repair=str(r.get("repair", "")), boundary=str(r.get("boundary", "")),
                observation_fields=tuple(r.get("observation_fields") or ()),
                operation=str(r.get("operation", "")),
                candidate_primitive=str(r.get("candidate_primitive", "")),
                safety_contract=tuple(r.get("safety_contract") or ()),
                evidence_case_ids=tuple(r.get("evidence_case_ids") or ()),
                provider=str(r.get("provider", "")), notes=str(r.get("notes", ""))))
        return led
