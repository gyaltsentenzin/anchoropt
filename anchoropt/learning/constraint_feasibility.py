"""Is the OPERATOR feasible for the CONSTRAINT the signal observed?

THE MEASURED DEFECT. Autonomous discovery localized a real refusal correctly -- a store rejecting a
write for lack of room, observed on 84 genuine states -- and then attached an operator that shrinks
the payload. On that backend the limit counted ENTRIES, so shrinking a payload frees nothing. The
signal was right; the operator could not address the constraint. Feasibility checks up to that point
asked "can the host run this operator at this boundary?" (yes) and never "can this operator change the
quantity the constraint is expressed in?" (no).

Had it reached GPU it would have returned a null indistinguishable from "the mechanism does not
transfer" -- the most expensive kind of wrong answer, because it retires a good idea.

THE ABSTRACTION, and the line core must not cross. A constraint limits some QUANTITY. An operator
changes some set of quantities. Feasible iff they intersect. Core knows only that quantities are
opaque tokens compared for equality; it never learns that one spelling means characters in a blob and
another means occupied slots, nor which store raised it. The adapter declares both sides.

WHY A DECLARATION AND NOT AN INFERENCE. Two refusals can share an error class and differ in unit; one
signal's remedy can be right on one backend and meaningless on another. The host is the only component
that knows its own units, so it states them, and core enforces the intersection. A host that declares
nothing gets UNKNOWN -- never a silent pass, because a silent pass here is what produced the null.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

FEASIBLE = "FEASIBLE"
INFEASIBLE = "INFEASIBLE"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Constraint:
    """A limit the host can hit, named by the QUANTITY it bounds.

    `quantity` is an opaque token. `unit` and `raised_by` are carried for the report only -- core
    compares nothing but `quantity`.
    """

    name: str
    quantity: str
    unit: str = ""
    raised_by: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        u = f" ({self.unit})" if self.unit else ""
        return f"{self.name}: bounds {self.quantity}{u}"


@dataclass(frozen=True)
class OperatorEffect:
    """What an operator can actually change, as a set of opaque quantity tokens.

    `reduces` is the honest field name: an operator that reduces a quantity can relieve a constraint
    bounding it. `unaffected` is optional and documentation-only -- it makes an intended non-effect
    reviewable instead of implicit.
    """

    operator: str
    reduces: tuple[str, ...] = ()
    unaffected: tuple[str, ...] = ()

    def addresses(self, constraint: Constraint) -> bool:
        return constraint.quantity in self.reduces


@dataclass(frozen=True)
class FeasibilityVerdict:
    status: str
    operator: str
    constraint: str
    detail: str
    remedy: str = ""

    @property
    def ok(self) -> bool:
        return self.status == FEASIBLE

    def __str__(self) -> str:  # pragma: no cover - display only
        s = f"[{self.status}] {self.operator} vs {self.constraint}: {self.detail}"
        return s + (f"\n    remedy: {self.remedy}" if self.remedy else "")


class ConstraintContract:
    """A host's declaration of its constraints and what its operators change.

    Deliberately tiny. It exists so the question "can this operator relieve this constraint?" has an
    answer that is DECLARED by the component that knows, checkable by the component that decides, and
    greppable when it fails.
    """

    def __init__(self, constraints: Sequence[Constraint] = (),
                 effects: Sequence[OperatorEffect] = ()) -> None:
        self._constraints: dict[str, Constraint] = {c.name: c for c in constraints}
        self._effects: dict[str, OperatorEffect] = {e.operator: e for e in effects}

    # -- declaration ---------------------------------------------------------------------------
    def declare_constraint(self, c: Constraint) -> None:
        self._constraints[c.name] = c

    def declare_effect(self, e: OperatorEffect) -> None:
        self._effects[e.operator] = e

    def constraint(self, name: str) -> Constraint | None:
        return self._constraints.get(name)

    def effect(self, operator: str) -> OperatorEffect | None:
        return self._effects.get(operator)

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        return tuple(self._constraints[k] for k in sorted(self._constraints))

    # -- the check -----------------------------------------------------------------------------
    def check(self, operator: str, constraint_name: str) -> FeasibilityVerdict:
        """Feasible iff the operator reduces the quantity the constraint bounds.

        An undeclared side returns UNKNOWN with the missing declaration named. UNKNOWN is not a
        pass: a caller that treats it as one has reintroduced the defect.
        """
        c = self._constraints.get(constraint_name)
        e = self._effects.get(operator)
        if c is None and e is None:
            return FeasibilityVerdict(
                UNKNOWN, operator, constraint_name,
                "neither the constraint nor the operator effect is declared",
                remedy="declare both on the adapter's ConstraintContract")
        if c is None:
            return FeasibilityVerdict(
                UNKNOWN, operator, constraint_name, "constraint is not declared",
                remedy=f"declare Constraint(name={constraint_name!r}, quantity=...) naming the "
                       f"quantity it bounds")
        if e is None:
            return FeasibilityVerdict(
                UNKNOWN, operator, constraint_name, f"operator {operator!r} declares no effect",
                remedy=f"declare OperatorEffect(operator={operator!r}, reduces=(...)) naming the "
                       f"quantities it can reduce")
        if e.addresses(c):
            return FeasibilityVerdict(
                FEASIBLE, operator, constraint_name,
                f"operator reduces {c.quantity!r}, which is the quantity the constraint bounds")
        return FeasibilityVerdict(
            INFEASIBLE, operator, constraint_name,
            f"constraint bounds {c.quantity!r}"
            + (f" (unit: {c.unit})" if c.unit else "")
            + f" but the operator reduces {list(e.reduces)!r}; it cannot change that quantity",
            remedy=f"find an operator that reduces {c.quantity!r}, or drop this arm -- measuring it "
                   f"would yield a null that reads as 'the mechanism does not transfer'")

    def feasible_operators(self, constraint_name: str) -> tuple[str, ...]:
        """Which declared operators COULD relieve this constraint. Names the gap when none can."""
        c = self._constraints.get(constraint_name)
        if c is None:
            return ()
        return tuple(sorted(op for op, e in self._effects.items() if e.addresses(c)))
