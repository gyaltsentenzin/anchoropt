"""Residual mining for tau-bench: failing episodes -> `ResidualDiagnosis` -> `ResidualProblem`.

THE LINE THIS FILE WALKS. A diagnosis provider says WHAT went wrong, in terms of evidence the runtime
actually recorded. It must not say WHERE to intervene or WHICH action to use -- core owns the locus, the
condition and the action, and a miner that names a remedy has already chosen the answer the round is
supposed to measure.

So every field below is derived from the recorded trajectory by a deterministic rule, and
`proposed_behavior_change` describes a BEHAVIOUR ("do not commit to an answer while the requested change
is outstanding"), never a mechanism ("suppress the call", "reprompt at the gate").

`consequential_decision` is the grouping key, and it is phrased in the runtime's own vocabulary -- the
kind of decision, not its repair. Cases sharing a key form one `ResidualProblem`, which is the unit the
outer loop iterates over.

No import of tau2: this reads the events `tau2_episodes.trajectory_events` already produced.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from anchoropt.learning.residual_problem import ResidualProblem
from anchoropt.runtime import ResidualDiagnosis

PROVIDER = "tau2_trajectory_miner"

# The decision kinds this miner can distinguish from a recorded trajectory. Deliberately few: each is
# a shape the events really carry, not a hypothesis about intent.
D_TOOL_ERROR = "a tool call that returned {kind}"
D_ANSWER_NO_ACTION = "an answer committed to the user with no tool call in the episode"
D_ANSWER_AFTER_ERROR = "an answer committed to the user while a tool call was still failing"
# The no-error counterpart. Keeping it separate is not cosmetic: an earlier version of this miner
# reported D_ANSWER_AFTER_ERROR for cases whose own evidence string said "none errored", so the round's
# headline named a failing tool call on a residual with ZERO errored results -- and the two arms that
# condition on an error (`tool_call_failed`, `following_tool_error`) could never fire on it.
D_ANSWER_BEFORE_DONE = "an answer committed to the user before the task's required change was made"
D_UNKNOWN_TOOL = "a call proposed to a tool the toolset does not contain"
D_TERMINATED = "the episode ended before the agent finished ({reason})"
D_LAST_CALL = "the last call the agent proposed"


def _events_by_kind(events: Sequence[Mapping[str, Any]], kind: str) -> list[Mapping[str, Any]]:
    return [e for e in events or () if str(e.get("kind") or "") == kind]


def diagnose_case(case_id: str, events: Sequence[Mapping[str, Any]], *,
                  termination: str = "", reward: float = 0.0) -> ResidualDiagnosis | None:
    """One failing episode -> one diagnosis, or None when the trajectory shows nothing to attribute.

    Read BACKWARD in spirit: the classification keys on the LAST consequential decision the episode
    contains, because that is the one whose repair core will consider first before moving earlier.
    """
    events = list(events or ())
    if not events:
        return None

    proposals = _events_by_kind(events, "propose")
    results = _events_by_kind(events, "result")
    errored = [r for r in results if r.get("result_is_error")]
    unknown = [p for p in proposals if p.get("proposes_tool_call")
               and not p.get("proposed_tool_is_known")]
    any_call = any(p.get("proposes_tool_call") for p in proposals)
    last_proposal = proposals[-1] if proposals else {}

    normal_end = str(termination or "").lower() in {"", "agent_stop", "user_stop",
                                                    "terminationreason.agent_stop",
                                                    "terminationreason.user_stop"}
    if not normal_end:
        decision = D_TERMINATED.format(reason=str(termination))
        mechanism = "the episode did not reach a normal end, so no outcome was scored"
        evidence = (f"termination={termination}; {len(proposals)} proposals, "
                    f"{len(errored)} errored results")
        change = "reach a normal end within the step and error budget"
    elif unknown:
        decision = D_UNKNOWN_TOOL
        mechanism = "the agent proposed a tool that does not exist in its toolset"
        evidence = (f"{len(unknown)} proposal(s) named an unknown tool, first="
                    f"{unknown[0].get('proposed_tool', '')!r}")
        change = "act only through tools the toolset actually contains"
    elif errored:
        kind = Counter(str(r.get("error_kind") or "domain_error") for r in errored).most_common(1)[0][0]
        evidence = (f"{len(errored)} of {len(results)} results errored; dominant kind={kind}; "
                    f"tools={sorted({str(r.get('result_tool') or '') for r in errored})}")
        if last_proposal.get("commits_to_reply"):
            decision = D_ANSWER_AFTER_ERROR
            mechanism = "the agent answered the user while a tool call was still failing"
            change = "not answer while the step the task requires is still outstanding"
        else:
            decision = D_TOOL_ERROR.format(kind=kind)
            mechanism = f"a tool call returned {kind} and the episode did not recover"
            change = "obtain a usable result for the step the task requires"
    elif not any_call:
        decision = D_ANSWER_NO_ACTION
        mechanism = "the agent answered without taking any action"
        evidence = f"{len(proposals)} agent turn(s), none proposing a tool call"
        change = "take the action the task requires rather than only replying"
    elif last_proposal.get("commits_to_reply"):
        # NO ERRORED RESULT HERE -- this branch is only reached when `errored` was empty, so the key
        # must not mention a failing call.
        decision = D_ANSWER_BEFORE_DONE
        mechanism = "the agent answered before the task's required state was reached"
        evidence = (f"{len(proposals)} proposals, {len(results)} results, none errored; "
                    f"the last turn replied instead of acting")
        change = "not answer while the step the task requires is still outstanding"
    else:
        decision = D_LAST_CALL
        mechanism = "the episode ran to completion and the final state did not satisfy the task"
        evidence = (f"{len(proposals)} proposals, {len(results)} results, none errored; "
                    f"last proposed tool={last_proposal.get('proposed_tool', '')!r}")
        change = "produce the end state the task specifies"

    return ResidualDiagnosis(
        case_id=str(case_id), mechanism=mechanism, evidence=evidence,
        consequential_decision=decision, proposed_behavior_change=change, provider=PROVIDER,
        metadata={"reward": float(reward), "termination": str(termination),
                  "n_proposals": len(proposals), "n_results": len(results),
                  "n_errored": len(errored)})


def mine(run) -> tuple[ResidualDiagnosis, ...]:
    """Every failing case in a `RunResult` -> its diagnosis."""
    out = []
    for cid in run.failing():
        d = diagnose_case(cid, run.events.get(cid) or (),
                          termination=run.termination.get(cid, ""),
                          reward=float(run.reward.get(cid, 0.0)))
        if d is not None:
            out.append(d)
    return tuple(out)


def problems(diagnoses: Sequence[ResidualDiagnosis], *, n_total: int) -> tuple[ResidualProblem, ...]:
    """Group diagnoses into ranked residual problems. Support = how many cases share the decision.

    Ranking is by SUPPORT, which is a property of the evidence, not a preference over repairs. Core
    still decides where and how to intervene for whichever problem it is handed.
    """
    groups: dict[str, list[ResidualDiagnosis]] = {}
    for d in diagnoses or ():
        groups.setdefault(d.consequential_decision, []).append(d)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    out = []
    for i, (key, members) in enumerate(ordered, start=1):
        out.append(ResidualProblem(
            key=key, diagnoses=tuple(members), rank=i,
            coverage=(len(members) / max(1, len(diagnoses))),
            saturation=0.0, expressible=True,
            phase_note=f"{len(members)} of {len(diagnoses)} residual cases; n={n_total}"))
    return tuple(out)
