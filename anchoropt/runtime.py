"""Host/runtime declarations: which actions are EXECUTABLE where, and a generic diagnosis.

WHY THIS MODULE EXISTS
----------------------
`anchor.feasible_actions` answers a STRUCTURAL question -- can this action legally exist at
this point -- and its exclusions hold for every host, because they follow from what exists in
a generate -> propose -> execute loop. That is necessary and not sufficient.

A second, independent question is whether the host RUNTIME can actually perform the action at
that point. It is a different question with a different answer per host, and conflating the two
has a measured cost on this project: in the TB2 prototype `TRANSFORM` passed the structural
grid at all three boundaries and was executable at exactly ONE, because its implementation read
a tool-result field that does not exist earlier in the cycle. Two of ten nominally admissible
cells did nothing, and one of the two reported `executed=True` while appending its payload to
the empty string. See `docs/CONSUMER_BOUNDARY_RULE.md` -- the same rule, one layer down.

So:

    feasible_actions(point)            structural   -- universal, this repo's original grid
    HostProfile.executable_actions(l)  operational  -- per host/runtime, NARROWS the above

`U_H(l)` in the paper's notation. A host may only ever narrow; it can never authorise a cell the
structural grid excludes, because that would mean suppressing a call that does not exist yet.

WHAT THIS MODULE MUST NOT CONTAIN
---------------------------------
No benchmark or framework vocabulary: no task ids, no memory-container concepts, no named hooks
of any agent framework, no filesystem paths, no tool names. A host declares its own subset from its
own adapter; the core consumes the declaration and never learns what produced it.

That rule is enforced mechanically, not by good intentions -- `tests/test_tb2_adapter.py` greps this
package for framework and benchmark vocabulary. It caught a violation in this very docstring, which
named the framework hook it was prohibiting.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from anchoropt.anchor import Action, IncisionPoint, exclusion_reason, feasible_actions


class ActionNotExecutable(ValueError):
    """Raised when an anchor names an action its host cannot perform at that point.

    Distinct from the structural `ValueError` in `Anchor.__post_init__`: that one means the cell
    cannot exist in ANY runtime, this one means THIS runtime cannot execute it. Keeping the two
    rejections distinguishable is the point -- a candidate pruned for the wrong reason is
    unattributable, and this project already paid for that once.
    """


@dataclass(frozen=True)
class HostProfile:
    """One host/runtime's executable action subset, per incision point.

    `name` identifies the runtime, not the benchmark: several benchmarks can share a host (any
    two LangChain agents), and one benchmark could be run on two hosts. Keeping them separate is
    what stops `U_H` from silently becoming `U_benchmark`.

    `executable` omits nothing implicitly. A point absent from the mapping declares NO executable
    actions there rather than defaulting to the structural grid -- an omission should read as
    "not declared", never as "everything works", which is exactly the failure mode this module
    exists to prevent.
    """

    name: str
    executable: Mapping[IncisionPoint, frozenset[Action]] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        for point, actions in self.executable.items():
            if not isinstance(point, IncisionPoint):
                raise TypeError(f"{self.name}: {point!r} is not an IncisionPoint")
            for action in actions:
                if not isinstance(action, Action):
                    raise TypeError(f"{self.name}: {action!r} is not an Action")
                reason = exclusion_reason(point, action)
                if reason is not None:
                    # A host narrows the structural grid; it cannot widen it.
                    raise ValueError(
                        f"host {self.name!r} declares {action.value} executable at "
                        f"{point.value}, but that cell is structurally inadmissible -- {reason}"
                    )

    def executable_actions(self, point: IncisionPoint) -> frozenset[Action]:
        """`U_H(l)` -- what this host can actually perform at `point`.

        Always a subset of `feasible_actions(point)`, enforced at construction.
        """
        return frozenset(self.executable.get(point, frozenset()))

    def can_execute(self, point: IncisionPoint, action: Action) -> bool:
        return action in self.executable_actions(point)

    def require(self, point: IncisionPoint, action: Action) -> None:
        """Raise unless this host can execute `action` at `point`.

        The message distinguishes 'structurally impossible' from 'this host cannot', because the
        two lead to different next steps: the first prunes a candidate permanently, the second
        is a gap a host implementation could close.
        """
        reason = exclusion_reason(point, action)
        if reason is not None:
            raise ActionNotExecutable(
                f"{action.value} at {point.value} is structurally inadmissible -- {reason}"
            )
        if not self.can_execute(point, action):
            declared = sorted(a.value for a in self.executable_actions(point))
            raise ActionNotExecutable(
                f"host {self.name!r} cannot execute {action.value} at {point.value}; "
                f"declared executable there: {declared or '(none)'}"
            )

    def unsupported_cells(self) -> tuple[tuple[IncisionPoint, Action], ...]:
        """Structurally admissible cells this host does NOT declare executable.

        Reported rather than hidden, for the same reason `exclusion_reason` is exposed: a
        shrunken grid with no explanation is indistinguishable from an oversight.
        """
        gaps = []
        for point in IncisionPoint:
            for action in sorted(feasible_actions(point), key=lambda a: a.value):
                if action not in self.executable_actions(point):
                    gaps.append((point, action))
        return tuple(gaps)


def structural_grid() -> dict[IncisionPoint, frozenset[Action]]:
    """The universal structural grid, as a mapping -- the widest any host may declare."""
    return {point: feasible_actions(point) for point in IncisionPoint}


@dataclass(frozen=True)
class ResidualDiagnosis:
    """One residual failure, in terms the core understands, from ANY diagnosis provider.

    An earlier internal harness was the first provider; it is deliberately not the only
    conceivable one, so its vocabulary does not appear here. In particular there is no field for
    a prompt hook: a provider whose own search space is prompt-heavy will propose prompt edits,
    and letting that reach the core would bias `(l, phi, mu, theta)` selection toward the
    provider's habits rather than the evidence. `proposed_behavior_change` carries WHAT should
    change; AnchorOpt decides WHERE and HOW.

    `case_id` is an opaque token. The core compares and groups it; it never parses it -- that is
    the adapter's job (see `benchmarks/bfcl_v4/adapter.py:parse_case_id`).
    """

    case_id: str
    mechanism: str                  # what the agent does wrong
    evidence: str                   # the observed boundary evidence it is inferred from
    consequential_decision: str     # which decision must change for it not to recur
    proposed_behavior_change: str   # the behavioural content, NOT a hook or a patch
    provider: str = ""              # which miner produced this, for attribution of the claim
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("case_id", "mechanism", "consequential_decision"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"ResidualDiagnosis.{name} must be non-empty")

    @property
    def failure_mechanism(self) -> str:
        """Alias for `mechanism`.

        The four-field seam schema names this `failure_mechanism`; the stored field predates that
        and is `mechanism`. Aliasing rather than renaming keeps the provider, its tests, and the
        frozen round-0 artifacts valid while letting new code use the documented name.
        """
        return self.mechanism


def diagnoses_from(provider: str, records: Iterable[Mapping[str, object]]
                   ) -> tuple[ResidualDiagnosis, ...]:
    """Build diagnoses from a provider's normalized records.

    A provider adapter does the translation and calls this; the core does not know any
    provider's field names.
    """
    out = []
    for r in records:
        out.append(ResidualDiagnosis(
            case_id=str(r["case_id"]),
            mechanism=str(r.get("mechanism", "")),
            evidence=str(r.get("evidence", "")),
            consequential_decision=str(r.get("consequential_decision", "")),
            proposed_behavior_change=str(r.get("proposed_behavior_change", "")),
            provider=provider,
            metadata=dict(r.get("metadata") or {}),
        ))
    return tuple(out)
