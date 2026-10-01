"""The `Anchor` four-tuple, and the three incision points inside a single LLM call.

WHY THIS MODULE EXISTS
----------------------
An LLM call is usually treated as one atomic step, which leaves exactly one lever: the prompt.
This project decomposes every call into THREE points at which the *system* -- not the model --
can act. That decomposition is what turns "prompt engineering" into a search space:

    3 incision points x 4 action families = 12 cells per locus

and it is why the same signal can help or hurt depending only on WHERE it is applied.

The cost of leaving the incision point implicit is measured, not hypothetical. A4 v1 and v2
share a signal, an action family, and a byte-identical injected text; they differ only in the
incision point:

    v1  pre_generation           trigger `step_count == 0`     fired 303/303   -4.95 pp
    v2  post_generation_pre_exec trigger `zero tool calls`     fired  59/303   +3.63 pp

v1 could not observe "about to answer without calling a tool" -- that fact does not exist
before generation -- so it fired everywhere and paid the cost on 244 episodes that did not
need it. Making the incision point a required, typed field is the fix: it cannot be forgotten
or buried in a trigger string.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field


class IncisionPoint(enum.Enum):
    """Where in one LLM call the system may intervene.

    Ordered by execution time. The three differ along two axes that trade off monotonically:
    how much the system KNOWS, and how much is still PREVENTABLE.

        point                     knows                          can still prevent
        ------------------------  -----------------------------  ------------------
        PRE_GENERATION            context/state, not the call    everything
        POST_GENERATION_PRE_EXEC  the proposed call + arguments  the execution
        POST_EXECUTION            the result/error, world moved  nothing
    """

    PRE_GENERATION = "pre_generation"
    POST_GENERATION_PRE_EXEC = "post_generation_pre_exec"
    POST_EXECUTION = "post_execution"

    @property
    def sees_proposed_call(self) -> bool:
        """Is a concrete proposed call visible here?

        False pre-generation: no call exists yet. This is exactly the fact A4 v1 needed and
        could not have.
        """
        return self is not IncisionPoint.PRE_GENERATION

    @property
    def can_prevent_execution(self) -> bool:
        """Can the intervention stop the world from changing?

        False post-execution: the tool already ran. Repair is possible; prevention is not.
        """
        return self is not IncisionPoint.POST_EXECUTION


class Action(enum.Enum):
    """The action families. Deliberately small, explicit, and closed.

        NOOP      let the ORIGINAL PIPELINE RUN UNMODIFIED at this point -- no interception.
                  Not 'do nothing' in the abstract: the agent still acts, it just acts as it
                  would have without us. That is what makes noop the CONTROL ARM, and it is a
                  legitimate verdict: two of six rounds here selected it.
        REPROMPT  inject text at a trigger; the model chooses what to do next
        SUPPRESS  cancel the proposed call -- it does not execute. What the model then OBSERVES
                  is a second, independent choice, and both variants are in the accepted set:
                    A3  removes the call outright; the model sees nothing in its place.
                    E1  withholds the call but REPLAYS the tool's verbatim recorded result, so
                        the observation is byte-identical and only the execution is saved.
                  Both are suppression -- the proposed action does not run. Do not read the
                  family as implying the model is left uninformed.
        REROUTE   SUBSTITUTE the call -- a different function, OR the same function with
                  different arguments

    There is deliberately no separate `transform` family. An earlier version split "rewrite the
    arguments" out from "change the destination", and the split does not survive contact with the
    code: both are one operation, *replace the proposed call with a better one*, and they share the
    same requirement -- positive evidence that the substitute RESOLVES. `capacity_repair` already
    implements both under one anchor (RELOCATE moves the call, REDUCE rewrites its payload), which is
    the concrete argument for merging them.

    Empirically it also never earned its keep: no accepted anchor used `transform`, and the one arm
    that did (W1, condensing an over-long entry) was rejected -- the rewrite was mechanically perfect,
    29/29 validated, and 25 of those 29 were immediately blocked by the NEXT constraint while 3 cases
    regressed against a zero variance floor. Fixing the payload had moved the failure one step.
    """

    NOOP = "noop"
    REPROMPT = "reprompt"
    SUPPRESS = "suppress"
    REROUTE = "reroute"


# Structural feasibility of the 3 x 4 = 12 cells.
#
# These exclusions are STRUCTURAL -- they follow from what exists at each point, not from any
# measurement on any benchmark, so they hold for every adapter. Each carries its reason,
# because a silently pruned cell is indistinguishable from one nobody thought of.
_STRUCTURAL_EXCLUSIONS: Mapping[tuple, str] = {
    (IncisionPoint.PRE_GENERATION, Action.SUPPRESS):
        "nothing to cancel: no call has been proposed yet",
    (IncisionPoint.PRE_GENERATION, Action.REROUTE):
        "no call to substitute for: neither its target nor its arguments exist until generation",
    (IncisionPoint.POST_EXECUTION, Action.SUPPRESS):
        "cannot undo a completed execution; the step is already spent",
}


def feasible_actions(point: IncisionPoint) -> frozenset[Action]:
    """The structurally admissible actions at `point`.

    This only ever NARROWS the space. Which of the survivors to install is a question for
    measurement, never for this function -- attribution prunes, measurement chooses.
    """
    return frozenset(
        a for a in Action if (point, a) not in _STRUCTURAL_EXCLUSIONS
    )


# The canonical (paper) operator names for the four stored Action values.
#
# The paper uses FIVE operators where this enum has four: REROUTE covers both `substitute`
# (replace the proposed call) and `transform` (reshape what the model observes). The split is the
# better description -- A9 is filed REROUTE and substitutes no call, which
# `A9_EVIDENCE["action_family_caveat"]` says should really be called augment-the-observation --
# but the enum is NOT renamed, because these four values are written into frozen policy.json
# artifacts and renaming them would edit historical records for no measurement gain.
#
# So the mapping is written down instead, here and in `tests/test_paper_consistency.py`, which
# pins it per anchor. REROUTE maps to a SET because resolving it needs the anchor, not just the
# action: see `rounds/anchors.py` / `PAPER_OPERATOR` for the per-anchor resolution.
CANONICAL_OPERATOR: Mapping[Action, frozenset[str]] = {
    Action.NOOP: frozenset({"noop"}),
    Action.REPROMPT: frozenset({"reprompt"}),
    Action.SUPPRESS: frozenset({"suppress"}),
    Action.REROUTE: frozenset({"substitute", "transform"}),
}


def canonical_operators(action: Action) -> frozenset[str]:
    """The canonical operator name(s) for a stored `Action`.

    A single-element set means the name is unambiguous. REROUTE returns two, because which one
    applies is a property of the individual anchor rather than of the action family.
    """
    return CANONICAL_OPERATOR[action]


def exclusion_reason(point: IncisionPoint, action: Action) -> str | None:
    """Why this cell is inadmissible, or None if it is admissible.

    Exposed so a report can state every exclusion rather than presenting a shrunken grid with
    no explanation.
    """
    return _STRUCTURAL_EXCLUSIONS.get((point, action))


@dataclass(frozen=True)
class Anchor:
    """A local intervention: `(failure context, attribution, incision point, action)`.

    All four are required. An anchor is not defined by its error label -- two anchors can share
    a locus and differ in cause (see A2, where one error string split into a read-side and a
    write-side anchor), or share signal and action and differ only in incision point (A4 v1
    vs v2, opposite signs).
    """

    name: str
    locus: str          # canonical semantic locus, e.g. "existence/identifier/not_found"
    attribution: str    # the traced cause, NOT the symptom
    incision_point: IncisionPoint
    action: Action
    params: Mapping[str, object] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        reason = exclusion_reason(self.incision_point, self.action)
        if reason is not None:
            raise ValueError(
                f"anchor {self.name!r}: {self.action.value} is structurally inadmissible at "
                f"{self.incision_point.value} -- {reason}"
            )

    @property
    def kind(self) -> str:
        """The intervention kind, which follows from the incision point rather than the signal.

        Post-execution anchors repair something that already failed; pre-execution anchors
        intercept a call before it commits. A1/A2 are the former, A3/A4 the latter.
        """
        if self.incision_point is IncisionPoint.POST_EXECUTION:
            return "error_recovery"
        if self.incision_point is IncisionPoint.POST_GENERATION_PRE_EXEC:
            return "commitment_gate"
        return "context_shaping"
