"""ACTION CONTRACTS: what each action family REQUIRES before it can become an evaluation arm.

THE ERROR THIS FIXES
--------------------
An action label was being treated as if it were already an executable policy. It is not. `REROUTE`
names an intent -- "achieve the same thing a different way" -- and says nothing about which
destination, which argument mapping, or whether the original call is retried or replaced. A proposal
naming REROUTE is INCOMPLETE, and completing it is the optimizer's job, not the proposer's.

So each family declares a CONTRACT: the action-specific parameters eta_mu that must be grounded from
the runtime before an arm exists. Two variables, kept apart on purpose:

    theta_phi   SIGNAL parameters   -- e.g. a similarity threshold. Tunable, searched on TRAIN.
    eta_mu      ACTION parameters   -- e.g. destination, argument mapping, retry semantics, retry
                                       budget, transformation operator. GROUNDED from capabilities,
                                       and where several groundings exist, each is its own ARM.

Collapsing both into one `theta` dict obscures which is which, and it obscured the real defect in
R2: theta was swept while eta was frozen at whatever the proposer happened to name.

THE PAPER'S FOUR FAMILIES, AND HOW THEY MAP
-------------------------------------------
`anchor.CANONICAL_OPERATOR` already records this: the stored enum has four values and the paper has
five operators, because REROUTE covers both `substitute` (replace the proposed call) and `transform`
(reshape what the model observes). The enum is deliberately NOT renamed -- those values are written
into frozen policy artifacts. So contracts are keyed on the CANONICAL OPERATOR, which is the level
where the requirements actually differ:

    suppress    which operation is suppressed, and what preserves safety
    reprompt    the instruction text, and a retry budget if applicable
    substitute  a compatible destination, an argument mapping, retry-vs-replace   (Action.REROUTE)
    transform   a writable target surface, an operator, preservation constraints  (Action.REROUTE)

`transform` is distinct from `substitute` and the distinction is operational: substitute changes WHICH
OPERATION RUNS; transform changes THE STATE OR OBSERVATION a later decision reads. The historical A9
is filed REROUTE and its own evidence record calls it augment-the-observation -- i.e. a `transform`
reading. The historical label is PRESERVED; only its requirements are now stated explicitly.

WHAT THIS MODULE MUST NEVER DO
------------------------------
Invent a tool, an argument, or a state surface. Every requirement is either grounded from the
runtime's declared capabilities or the family is reported INFEASIBLE with the missing requirement
named. Silently converting an ungroundable REROUTE into a REPROMPT is specifically prohibited: it
would measure a different controller under the first one's name.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anchoropt.anchor import Action, IncisionPoint, canonical_operators


class Operator(enum.Enum):
    """The paper's action operators. `substitute` and `transform` are both stored as REROUTE."""

    SUPPRESS = "suppress"
    REPROMPT = "reprompt"
    SUBSTITUTE = "substitute"
    TRANSFORM = "transform"

    @property
    def action(self) -> Action:
        return {"suppress": Action.SUPPRESS, "reprompt": Action.REPROMPT,
                "substitute": Action.REROUTE, "transform": Action.REROUTE}[self.value]


def operators_of(action: Action) -> tuple[Operator, ...]:
    """The operator readings of a stored action. REROUTE yields two; they are different contracts."""
    names = canonical_operators(action)
    return tuple(sorted((o for o in Operator if o.value in names), key=lambda o: o.value))


# Infeasibility codes. Distinct per missing requirement, so a rejection names what was absent.
REROUTE_INFEASIBLE = "REROUTE_INFEASIBLE"
REPROMPT_INFEASIBLE = "REPROMPT_INFEASIBLE"
SUPPRESS_INFEASIBLE = "SUPPRESS_INFEASIBLE"
TRANSFORM_INFEASIBLE = "TRANSFORM_INFEASIBLE"


@dataclass(frozen=True)
class ActionContract:
    """The eta_mu a family needs. `required` names them; `grounded_by` names the runtime hook."""

    operator: Operator
    required: tuple[str, ...]
    grounded_by: str
    infeasible_code: str
    doc: str
    # ETA THAT ONLY MEANS SOMETHING IF AN EXECUTOR ENFORCES IT.
    #
    # `required` is the family floor: every arm needs these or it does not exist. Fields listed in
    # `enforced` are different -- once an arm CARRIES one, it is making a claim about runtime
    # behaviour, and the claim is checked against the executor's declared `consumes`.
    #
    # `preservation` on SUPPRESS was grounded as the string "verify an equivalent copy survives
    # before cancelling" and NOTHING read it. A safety constraint no executor enforces is not a
    # constraint. It is not enough to drop the field, though: that would weaken E1, where preservation
    # is the DEFINING property. See `variant_required` -- the requirement is per-variant, because A3
    # and E1 are genuinely different contracts inside one family.
    enforced: tuple[str, ...] = ()

    # PER-VARIANT REQUIREMENTS. The A3/E1 case, stated in code rather than prose.
    #
    # `anchor.Action` already records that SUPPRESS has two accepted variants:
    #     A3  removes the call outright; the model sees nothing in its place
    #     E1  withholds the call but REPLAYS the tool's verbatim recorded result
    # They do not have the same requirements. A3 preserves nothing and needs to preserve nothing --
    # the write never happened. E1's whole claim is that the observation survives byte-identically,
    # so for E1 `preservation` is MANDATORY and must be executor-backed.
    #
    # Keying on the grounding's `variant` keeps both accepted anchors valid without weakening either:
    # an A3 arm is not asked for a field its executor cannot read, and an E1 arm cannot omit the field
    # that distinguishes it from A3.
    variant_required: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def required_for(self, variant: str) -> tuple[str, ...]:
        """The eta this family needs for a specific grounding variant. Family floor plus its extras."""
        extra = tuple(dict(self.variant_required or {}).get(str(variant or ""), ()))
        return tuple(self.required) + tuple(r for r in extra if r not in self.required)


CONTRACTS: Mapping[Operator, ActionContract] = {
    Operator.SUPPRESS: ActionContract(
        operator=Operator.SUPPRESS,
        required=("suppressed_operation",),
        enforced=("preservation",),
        variant_required={
            # E1 -- withhold but REPLAY the recorded result. Preservation is the defining property,
            # so it is mandatory here and (being `enforced`) must also be executor-backed.
            "replay_recorded_result": ("preservation",),
            "withhold_and_replay": ("preservation",),
        },
        grounded_by="ground_suppress",
        infeasible_code=SUPPRESS_INFEASIBLE,
        doc=("must identify a suppressible operation AT THIS BOUNDARY. No downstream destination is "
             "needed. After execution there is nothing left to cancel, so the boundary is part of "
             "the requirement.\n\n"
             "THE A3/E1 SPLIT IS A CONTRACT SPLIT, NOT A NEW FAMILY. `anchor.Action` records both "
             "variants as accepted suppression. A3 removes the call and the model sees nothing in its "
             "place: it preserves NOTHING and needs to preserve nothing, because the write never "
             "happened -- demanding `preservation` made every A3 arm ground a safety string no "
             "executor read. E1 withholds but replays the verbatim recorded result, so preservation "
             "is exactly what distinguishes it, and for E1 the field is REQUIRED via "
             "`variant_required` AND must be consumed by the executor via `enforced`. Neither "
             "variant is weakened and no action family is added.")),
    Operator.REPROMPT: ActionContract(
        operator=Operator.REPROMPT,
        required=("instruction", "retry_budget"),
        grounded_by="ground_reprompt",
        infeasible_code=REPROMPT_INFEASIBLE,
        doc=("must carry executable instruction CONTENT and a retry budget. Several semantically "
             "distinct instructions may be supplied -- each is its own ARM and measurement chooses "
             "among them; an LLM may author them but never selects the winner.")),
    Operator.SUBSTITUTE: ActionContract(
        operator=Operator.SUBSTITUTE,
        required=("destination", "argument_mapping", "retry_semantics"),
        grounded_by="ground_substitute_destinations",
        infeasible_code=REROUTE_INFEASIBLE,
        doc=("must name a REAL compatible destination, a mechanical argument mapping, and whether "
             "the original call is retried or replaced. EVERY credible grounded destination becomes "
             "a separate arm -- choosing one arbitrarily is what this contract exists to prevent. "
             "With no compatible executable destination: REROUTE_INFEASIBLE, never a silent "
             "downgrade to REPROMPT.")),
    Operator.TRANSFORM: ActionContract(
        operator=Operator.TRANSFORM,
        required=("target_surface", "operator", "preservation"),
        grounded_by="ground_transforms",
        infeasible_code=TRANSFORM_INFEASIBLE,
        doc=("must name a WRITABLE state or observation surface, the transformation operator, and "
             "the preservation constraint. Distinct from substitute: this changes the state a later "
             "decision reads, rather than which operation runs.")),
}


@dataclass(frozen=True)
class InstantiatedAction:
    """One fully grounded (operator, eta_mu). A distinct evaluation arm.

    `variant` distinguishes sibling groundings of one family -- two destinations, or two reprompt
    instructions -- so the sweep table can name them apart.
    """

    operator: Operator
    eta: Mapping[str, Any] = field(default_factory=dict)
    variant: str = ""
    grounding: Mapping[str, Any] = field(default_factory=dict)
    detail: str = ""

    @property
    def action(self) -> Action:
        return self.operator.action

    @property
    def label(self) -> str:
        return f"{self.operator.value}" + (f":{self.variant}" if self.variant else "")


@dataclass(frozen=True)
class ContractFailure:
    """A family that could not be instantiated, with the MISSING REQUIREMENT named."""

    operator: Operator
    code: str
    missing: tuple[str, ...]
    detail: str = ""


def instantiate(operator: Operator, *, signal: str, boundary: IncisionPoint, runtime,
                ) -> tuple[list[InstantiatedAction], ContractFailure | None]:
    """Complete one action family into every grounded arm, or report what is missing.

    The optimizer's contract, stated plainly: this does NOT judge semantic plausibility -- the
    proposer already narrowed the space. It asks the runtime for each required eta_mu, enumerates
    every credible grounding, and reports the missing requirement when it cannot.
    """
    contract = CONTRACTS[operator]
    hook = getattr(runtime, contract.grounded_by, None)
    if hook is None:
        return [], ContractFailure(
            operator, contract.infeasible_code, contract.required,
            f"this runtime declares no {contract.grounded_by}(), so {operator.value} cannot be "
            f"instantiated here; requirements {list(contract.required)} are ungrounded")

    try:
        groundings = hook(signal, boundary) or []
    except Exception as exc:                       # a runtime that raises is a runtime that cannot
        return [], ContractFailure(operator, contract.infeasible_code, contract.required,
                                   f"{contract.grounded_by}() raised {type(exc).__name__}: {exc}")

    out: list[InstantiatedAction] = []
    incomplete: list[tuple[str, tuple[str, ...]]] = []
    for g in groundings:
        eta = dict(g.get("eta") or {})
        # PER-VARIANT, not per-family: an E1 grounding must carry `preservation`, an A3 one must not
        # be asked for it. `required_for` falls back to the family floor for any unknown variant.
        need = contract.required_for(str(g.get("variant") or ""))
        missing = tuple(r for r in need if r not in eta or eta[r] in (None, ""))
        if missing:
            incomplete.append((str(g.get("variant") or "?"), missing))
            continue
        out.append(InstantiatedAction(
            operator=operator, eta=eta, variant=str(g.get("variant") or ""),
            grounding=dict(g.get("grounding") or {}), detail=str(g.get("detail") or "")))
    if out:
        return out, None
    if incomplete:
        variant, missing = incomplete[0]
        return [], ContractFailure(
            operator, contract.infeasible_code, missing,
            f"every grounding was incomplete; e.g. variant {variant!r} lacks {list(missing)}")
    return [], ContractFailure(
        operator, contract.infeasible_code, contract.required,
        f"the runtime grounded no {operator.value} option for signal {signal!r} at "
        f"{boundary.value}; requirements are {list(contract.required)}")


def validate(action: InstantiatedAction) -> tuple[bool, str]:
    """HARD validation, independent of any prompt. An arm failing this must never be evaluated.

    The four rules requested, enforced in code rather than requested in text:
      substitute  needs at least one grounded compatible target
      reprompt    needs executable instruction content
      suppress    needs an identified suppressible operation
      transform   needs a named writable surface
    """
    c = CONTRACTS[action.operator]
    need = c.required_for(action.variant)
    missing = [r for r in need if r not in action.eta or action.eta[r] in (None, "")]
    if missing:
        return False, f"{c.infeasible_code}: missing {missing}"
    if action.operator is Operator.SUBSTITUTE and not str(action.eta["destination"]).strip():
        return False, f"{REROUTE_INFEASIBLE}: no grounded compatible target"
    if action.operator is Operator.REPROMPT and not str(action.eta["instruction"]).strip():
        return False, f"{REPROMPT_INFEASIBLE}: no executable reprompt content"
    if action.operator is Operator.SUPPRESS and not str(action.eta["suppressed_operation"]).strip():
        return False, f"{SUPPRESS_INFEASIBLE}: no suppressible operation identified"
    if action.operator is Operator.TRANSFORM and not str(action.eta["target_surface"]).strip():
        return False, f"{TRANSFORM_INFEASIBLE}: no writable target surface named"
    return True, "grounded"
