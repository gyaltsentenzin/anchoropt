"""Common-prestate eligibility: which cases may be compared CAUSALLY across arms.

FROZEN BEFORE THE REPLICATION RUN (see docs/SELFEVOLVE_R3_REPLICATION_PROTOCOL.md). Written here,
in the package, rather than inline in a scoring script, so the criterion cannot drift between the
run that motivated it and the run that tests it.

THE RULE
--------
A case is eligible for causal comparison only if the arm and the control share the same trajectory
UP TO the point where the intervention could first fire. After that point the arm is *supposed* to
diverge -- that divergence is the treatment. Before it, any divergence is sampling noise, and a case
whose prestate differs was not the same experiment in both arms.

WHY THIS IS NOT THE `step-0 decoded count` PROXY I USED FIRST
------------------------------------------------------------
R3's first analysis compared `len(steps[0]["decoded"])`. That is necessary but too weak in two ways:

  1. it reads only step 0, so for an arm that fires at a LATER step it ignores every step in
     between -- two arms could agree at step 0 and diverge at step 3 before firing at step 4.
  2. it compares only the COUNT of decoded calls, so `core_memory_retrieve(...)` and
     `archival_memory_retrieve(...)` at the same index look identical.

Both were harmless in R3 as it happened (every episode was single-turn and every firing was on turn
0), but a criterion that is only accidentally correct is not frozen -- it is waiting to be wrong. So
eligibility here compares the full prefix, call TEXT included, up to the firing boundary.

THE FIRING BOUNDARY IS SHARED, NOT PER-ARM
-----------------------------------------
The boundary must be the SAME index in every arm, and it is the EARLIEST step at which any arm
fires. Truncating each arm at its own firing index is wrong and the tests caught it: the control
never fires, so its "prefix" would be its whole trajectory while a treated arm's stops at the
injection -- and then every genuinely treated case compares a full trajectory against a truncated
one and reads as drift. That would have discarded exactly the cases the experiment is about.

So: `common_fire_index` = min over arms of `first_fire_index`, or None if no arm fired. Every arm is
then truncated at that one index. Steps at or after it are the treatment window and are not compared.
When no arm fires anywhere, the whole trajectory is compared -- nothing was treated, so any
difference is drift.

Deliberately NOT here: any notion of "correct". Eligibility is decided on the prestate alone, never
on the outcome -- selecting comparable cases using the outcome is how a subset analysis becomes a
way to choose the answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# The step key the post-generation reprompt executor writes at its firing site. Remedy-flag
# executors do NOT appear in `gates_fired` (that dict is registry-derived), so the per-step
# trajectory record is the only source -- see docs/SELFEVOLVE_R3_RESULT.md.
FIRE_KEY = "zero_call_reprompt_gate"


def first_fire_index(steps: Sequence[Mapping[str, Any]], fire_key: str = FIRE_KEY) -> int | None:
    """Index of the first step where the intervention fired, or None if it never did."""
    for i, s in enumerate(steps):
        if isinstance(s, Mapping) and s.get(fire_key):
            return i
    return None


def step_signature(step: Mapping[str, Any]) -> tuple:
    """The part of a step that must match for two trajectories to be the same experiment.

    Includes the decoded call TEXT, not just how many there were: a different tool or a different
    argument at the same index is a different prestate, and comparing counts alone would call them
    equal. `status` is included because it distinguishes "executed" from a terminal answer.

    Excludes anything downstream of the intervention (injection keys, gate flags) and anything about
    correctness.
    """
    decoded = step.get("decoded") or []
    return (
        step.get("turn"),
        step.get("step"),
        step.get("status"),
        tuple(str(c) for c in decoded),
    )


def prestate(steps: Sequence[Mapping[str, Any]], cut: int | None) -> tuple:
    """The comparable prefix of one episode, truncated at the SHARED boundary `cut`.

    `cut` is a single index applied to every arm (see `common_fire_index`), never each arm's own
    firing index -- comparing a non-firing control's full trajectory against a treated arm's
    truncated one turns the treatment itself into apparent drift.
    """
    prefix = steps if cut is None else steps[:cut]
    return tuple(step_signature(s) for s in prefix if isinstance(s, Mapping))


def common_fire_index(steps_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
                      fire_key: str = FIRE_KEY) -> int | None:
    """The earliest step index at which ANY arm fired for this case, or None if none did.

    The earliest, because everything from that point on is inside some arm's treatment window and is
    therefore not evidence about comparability.
    """
    idx = [i for i in (first_fire_index(st, fire_key) for st in steps_by_arm.values())
           if i is not None]
    return min(idx) if idx else None


def eligible_cases(episodes_by_arm: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
                   *, control: str, fire_key: str = FIRE_KEY) -> tuple[set[str], dict[str, set[str]]]:
    """(eligible case ids, per-arm drifted case ids).

    `episodes_by_arm[arm][case_id] -> steps`. A case is eligible when every arm's prestate equals
    the CONTROL's prestate. Cases absent from any arm are not eligible -- a missing case is a
    denominator problem, reported separately, never silently dropped into the comparison.
    """
    arms = list(episodes_by_arm)
    if control not in episodes_by_arm:
        raise ValueError(f"control arm {control!r} not among {arms}")
    common = set(episodes_by_arm[control])
    for arm in arms:
        common &= set(episodes_by_arm[arm])

    eligible, drifted = set(), {arm: set() for arm in arms}
    for c in common:
        per_arm = {arm: episodes_by_arm[arm][c] for arm in arms}
        cut = common_fire_index(per_arm, fire_key)
        base = prestate(per_arm[control], cut)
        ok = True
        for arm in arms:
            if prestate(per_arm[arm], cut) != base:
                drifted[arm].add(c)
                ok = False
        if ok:
            eligible.add(c)
    return eligible, drifted


def qualifying_in_control(steps: Sequence[Mapping[str, Any]], *, runtime: Any = None) -> bool:
    """Whether the CONTROL episode is a true target state: it reached the terminal COMMITMENT
    without having taken the relevant prior action.

    Read from the control only, so the target population is defined independently of any arm.

    THE QUESTION IS GENERIC; THE ENCODING IS THE ADAPTER'S. Core asks "did this trajectory commit
    without acting first"; which status string marks a commitment, and which field records the
    actions, are facts about one runtime's traces. A `runtime` exposing `commits_to_answer(event)`
    and/or `actions_in(event)` answers it in its own vocabulary.

    The fallback below is the ORIGINAL BFCL-shaped reading, retained deliberately and unchanged:
    several measured results (the SE1/R3 target populations) are defined by this predicate, so
    changing what counts as comparable would silently redefine those populations. It is a
    compatibility shim, not the contract -- a new runtime should supply the hooks.
    """
    if not steps:
        return False
    commits = getattr(runtime, "commits_to_answer", None)
    actions_in = getattr(runtime, "actions_in", None)
    if callable(commits):
        reached_answer = any(isinstance(s, Mapping) and commits(s) for s in steps)
    else:
        reached_answer = any(isinstance(s, Mapping) and s.get("status") == "answer_end_turn"
                             for s in steps)
    if callable(actions_in):
        acted = sum(len(actions_in(s) or ()) for s in steps if isinstance(s, Mapping))
    else:
        acted = sum(len(s.get("decoded") or []) for s in steps if isinstance(s, Mapping))
    return acted == 0 and reached_answer

# ------------------------------------------------------------------------------------------------
# THE STEP-0 PROBLEM, and the criterion that works when the prefix is empty
# ------------------------------------------------------------------------------------------------
#
# Measured on the seed-42 R3 data: `common_fire_index` is 0 in all 35 cases where any arm fires, and
# None in the other 54. So the compared prefix is EMPTY for every treated case, and
# `eligible_cases` returns 85/89 while actually having checked nothing on the cases that matter. A
# criterion that passes vacuously on the treatment group is not a safeguard.
#
# For a signal evaluated on the model's FIRST decision, comparability has to be defined on the
# TARGET POPULATION instead: a case is comparable iff the CONTROL was a genuine target state (a
# terminal answer with no tool call) and the arm therefore had the opportunity to intervene on it.
# Cases where the arm fired but the control was NOT a zero-call state are cases where sampling
# created the opportunity -- an arm cannot take credit for a state the control never presented -- and
# cases where the control qualified but the arm never fired are opportunities the arm declined.
#
# Both are reported, never merged, because they answer different questions:
#   on_target_shared   control qualified AND arm fired  -> the causal comparison
#   opportunity_created arm fired, control did not qualify -> sampling, excluded from attribution
#   opportunity_missed control qualified, arm did not fire -> engagement failure, kept in the
#                      denominator (an arm is not rewarded for skipping hard cases)

def target_population(control_steps_by_case: Mapping[str, Sequence[Mapping[str, Any]]]) -> set[str]:
    """The cases the signal is ABOUT, defined from the control alone."""
    return {c for c, st in control_steps_by_case.items() if qualifying_in_control(st)}


def engagement_split(control_steps: Mapping[str, Sequence[Mapping[str, Any]]],
                     arm_steps: Mapping[str, Sequence[Mapping[str, Any]]],
                     fire_key: str = FIRE_KEY) -> dict[str, set[str]]:
    """Partition the cases by (control qualified?) x (arm fired?).

    `on_target_shared` is the only cell from which a causal claim may be made: the control presented
    a genuine target state and the arm acted on it.
    """
    common = set(control_steps) & set(arm_steps)
    target = {c for c in common if qualifying_in_control(control_steps[c])}
    fired = {c for c in common if first_fire_index(arm_steps[c], fire_key) is not None}
    return {
        "on_target_shared": target & fired,
        "opportunity_created": fired - target,
        "opportunity_missed": target - fired,
        "off_target_quiet": common - target - fired,
    }
