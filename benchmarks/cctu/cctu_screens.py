#!/usr/bin/env python3
"""PRE-ARM SCREENS: refuse a candidate the control already proves cannot be measured.

    python benchmarks/cctu/cctu_screens.py --run results/granite/train_baseline \\
                                           --candidates rounds/CCTU_CYCLE2/candidates \\
                                           --benefit PSR

No model, no GPU, no network. Reads a CONTROL run's artifacts, replays each episode's state
trajectory, and answers two questions about every proposed candidate BEFORE an arm is spent on it:

    SUPPORT        does this signal fire often enough in the control to produce a delta at all?
    REACHABILITY   at post_execution, is the benefit metric already unrecoverable where it fires?
    LOSS EXPOSURE  does it fire in episodes that are ALREADY SUCCEEDING?

All three are vetoes, all three are cheap, and every one of them was paid for in arms rather than in
compute: support cost three arms in cycle 1, reachability three more, and loss exposure cost all five
scoreable arms of cycle 2.

WHY THESE THREE, AND WHY BEFORE THE ARMS
----------------------------------------
Cycle 1 spent six arms and installed nothing. The two failure shapes were different and neither
needed a GPU to see:

  * arms 04-06 fired 7-8 times across 280 episodes on `tool_execution_error` -- a class
    `cctu_signals.py` declares so it can be EXCLUDED from the residual, whose support in the control
    is 5 tool faults in 140 episodes. `rounds/CCTU_CYCLE1/RESULT.md`: "Next round should not spend
    arms on a signal whose support in the control is single digits, and that is a cheap screen to add
    before the arms rather than after." That is SUPPORT.

  * arms 01-03 fired 1500+ times and still could not win, for a reason that is arithmetic rather
    than statistical. `evaluation.judge` defines `PSR = acc AND has_if_error == 0`, and
    `compute_if_flags` scans EVERY message for `INSTRUCTION FOLLOWING ERROR`. The trigger of all
    three arms was `constraint_violation_reported`, which is true exactly when such a message
    exists. So at the instant the signal fires, PSR for that episode is already 0 and cannot rise
    again -- the benefit metric is pinned under the trigger. Measured on the frozen control: the
    controller executed in 198 episodes and PSR=1 in 0 of them, in the arm AND in the control, and
    across all seven scored runs there is no episode with PSR=1 that contains a violation.
    Arm 01's ATTRIBUTION veto ("all 3 PSR gains are in episodes it never touched") was therefore not
    bad luck: no other outcome was available. That is REACHABILITY.

  * cycle 2 then spent five scoreable arms on conditions that fire in EVERY episode. Its control has
    37 episodes at `SR = 1` (which can only go down) and 26 at `acc = 1 AND SR = 0` (which can go up),
    so those arms were betting 26 reachable gains against 37 exposed losses. They came in at 8 gains /
    10 losses, 4 / 12, 5 / 4 -- a null with real `acc` regressions, and four of five arms harming
    `acc` beyond its floor. That is LOSS EXPOSURE, and it is the term the other two do not ask: both
    of them are about the upside.

REACHABILITY IS SCOPED TO POST_EXECUTION, and the scoping is the correction cycle 2 forced. "The
metric is 0 in every episode this fires in" means two opposite things. After the validator it means
UNRECOVERABLE -- a violation is already in the transcript and no later action raises PSR. At the
commitment gate it means the condition fires only where the metric is currently failing, which is
zero loss exposure and the property to design for. An earlier version of this module vetoed
`rounds_remaining < 1 AND proposes_tool_call` (gain 18, loss 0) on that count while passing the five
arms that fired in all 37 already-succeeding episodes -- exactly backwards.

REACHABILITY IS THE GENERALIZABLE ONE, and it is not CCTU-specific. Any runtime can ask "in the
control, restricted to the states where this condition holds, does the benefit metric ever take the
value I want to move it to?" and refuse the cell when the answer is no. It belongs beside the support
screen in the core; it lives here because this is where the metric definitions are.

WHAT A SCREEN MAY NOT DO
------------------------
It may not rank, choose or rewrite a candidate. `cycle1_propose.py`'s docstring is the rule: the
proposer "must not choose WHERE to intervene, WHAT condition decides, or HOW to act", and a screen
that reordered candidates would be doing that under another name. Every function here returns a
VERDICT on a candidate somebody else built, with the number it was refused on. A screen is a veto and
an explanation, never a selection.

It also may not read a label the runtime cannot read. The screens do read `detail.jsonl` -- which is
scored output -- and that is legitimate and bounded: they read it to characterise the CONTROL's own
outcome distribution, which is what every tolerance in this project is already derived from. They
never hand a metric value to a predicate, and `assert_no_oracle_leak` is re-run here so a screening
pass cannot be the thing that imports the answer key.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cctu_adapter as runtime  # noqa: E402

W = 96

# The boundary at which the validator has already run. Named because two screens turn on it: after it,
# a metric the trigger implies is lost stays lost; before it, nothing has been counted.
POST_EXEC = "post_execution"

# The metrics `evaluation.judge` emits, and the ONE this module treats as a benefit by default.
# Declared rather than defaulted in a signature so a reader can see that the choice is a parameter.
METRICS: tuple[str, ...] = ("acc", "SR", "PSR")

# SUPPORT FLOOR. Deliberately NOT derived here, and deliberately not 0.
#
# The honest floor is the control's own per-episode flip count: an arm cannot demonstrate a net gain
# smaller than the noise between two runs of one policy. On granite train that is PSR 2 / SR 3 / acc
# 1, plus 1 for un-measured between-job drift (`rounds/CCTU_CYCLE1/FROZEN.md`). A signal that fires
# in fewer episodes than the floor cannot clear the floor even at 100% conversion, so the floor IS
# the support bar and there is nothing extra to invent.
#
# 10 is the DEFAULT only because it is above every measured floor on this corpus while staying well
# under the smallest residual class worth an arm. Pass --min-episodes to use a floor you measured.
DEFAULT_MIN_EPISODES = 10


# ------------------------------------------------------------------------------------------------
# reading a control run
# ------------------------------------------------------------------------------------------------
def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_control(run: Path, *, data_dir: Path | None = None) -> dict[str, Any]:
    """A control run directory -> the per-episode outcomes and replayed state trajectories.

    The states come from `cycle1_propose.replay_episode`, which advances a REAL
    `DialogueConstraintChecker` through the recorded turns rather than simulating one. Reusing it is
    the point: a screen that reconstructed state its own way could refuse a candidate on a
    trajectory the proposer never saw, and the disagreement would look like a screening bug rather
    than two definitions of the state.
    """
    import cycle1_propose as propose

    propose.assert_no_oracle_leak()

    detail = {r["id"]: r for r in _jsonl(run / "detail.jsonl")}
    transcripts = {r["id"]: r["messages"] for r in _jsonl(run / "response.jsonl")}
    if not detail:
        raise SystemExit(f"{run}: no detail.jsonl. Score the run first (evaluation.py --detail).")
    if not transcripts:
        raise SystemExit(f"{run}: no response.jsonl, so there are no trajectories to replay.")

    manifest = {}
    if (run / "run_manifest.json").exists():
        manifest = json.loads((run / "run_manifest.json").read_text())
    split = (manifest.get("split")
             or (manifest.get("corpus") or {}).get("input_file", "").replace(
                 "input_data_", "").replace(".jsonl", "")
             or "train")
    data = data_dir or (HERE / "data")
    samples = {}
    for row in _jsonl(data / f"input_data_{split}.jsonl"):
        samples[str(row["id"])] = row
    if not samples:
        raise SystemExit(f"no corpus at {data}/input_data_{split}.jsonl")

    states_by_case: dict[str, list[dict]] = {}
    replay_errors: list[str] = []
    for case_id in sorted(transcripts, key=lambda k: (int(str(k).split("_")[0]), str(k))):
        base = str(case_id).split("_")[0]
        sample = samples.get(base)
        if sample is None:
            replay_errors.append(f"{case_id}: no corpus row for id {base}")
            continue
        try:
            _events, states = propose.replay_episode(dict(sample, id=case_id),
                                                     transcripts[case_id])
        except Exception as exc:                                            # noqa: BLE001
            # A replay that dies must not take the screen with it, or one bad episode silences the
            # veto for every other candidate. Recorded and reported, never swallowed.
            replay_errors.append(f"{case_id}: {type(exc).__name__}: {exc}")
            continue
        states_by_case[case_id] = states

    return {
        "run": str(run),
        "split": split,
        "detail": detail,
        "states": states_by_case,
        "episodes": len(states_by_case),
        "replay_errors": replay_errors + list(propose.REPLAY_ERRORS),
    }


# ------------------------------------------------------------------------------------------------
# firing sets
# ------------------------------------------------------------------------------------------------
class Unscreenable(RuntimeError):
    """The signal could not be evaluated at all, so neither screen has an opinion about it.

    A DISTINCT OUTCOME FROM A VETO, and keeping them apart is the whole point of this class. An
    unresolvable signal that fell through as "fires in 0 episodes" would be reported as a support
    veto -- a confident refusal on a number that describes the screen rather than the candidate. That
    is the same defect `evaluate_signal` raises KeyError to avoid ("a missing evaluator that returns
    False is indistinguishable from a signal that did not fire, which is how an arm silently becomes
    the control arm"), and a screen is the last place to reintroduce it.

    The common cause is benign and expected: a SYNTHESIZED condition is screenable only once
    `Phi`-expansion has registered it in `cctu_adapter.EXPANDED_SIGNALS`. Screening a candidate whose
    predicate does not exist yet should say so and stop, not veto it.
    """


def _predicate(signal: str, expr: Mapping[str, Any] | None = None
               ) -> Callable[[Mapping[str, Any]], bool]:
    """A declared signal OR a persisted expression -> a callable on one observable state.

    Routed through `cctu_adapter.evaluate_signal` so the screen fires the SAME predicate the runtime
    will. Two definitions of when a signal holds is the defect this project has already paid for
    five times in one regex.

    `expr` is the `predicate` block of a spec carrying a SYNTHESIZED condition. It is compiled with
    `signal_lang.compile_signal` -- the same function `ControllerSpec._compile` uses, not a second
    reading of the same expression -- so a candidate screened here fires on exactly the states it will
    fire on in the runner. This is the path that makes an expanded candidate screenable at all: before
    the expression was persisted there was only a name, and a name is not resolvable in a process that
    did not synthesize it.

    Resolved ONCE, here, so an unresolvable candidate raises before any counting starts.
    """
    if expr:
        from anchoropt.learning.signal_lang import SignalSpecError, compile_signal

        payload = dict(expr)
        name = str(payload.pop("name", "") or signal or "predicate")
        try:
            compiled = compile_signal(name, payload, fields=runtime.synthesis_fields())
        except SignalSpecError as exc:
            raise Unscreenable(f"predicate for {name!r} is not a legal expression: {exc}") from exc
        return lambda state: bool(compiled.predicate(state, {}))

    if signal not in runtime.declared_signals() and signal not in getattr(
            runtime, "EXPANDED_SIGNALS", {}):
        raise Unscreenable(
            f"no evaluator for signal {signal!r}. Declared: "
            f"{sorted(runtime.declared_signals())}. A synthesized condition becomes screenable once "
            f"Phi-expansion registers it in cctu_adapter.EXPANDED_SIGNALS")
    try:
        params = runtime.probe_params(signal)
    except Exception as exc:                                                # noqa: BLE001
        raise Unscreenable(f"cannot build probe params for {signal!r}: "
                           f"{type(exc).__name__}: {exc}") from exc

    def _fire(state: Mapping[str, Any]) -> bool:
        return bool(runtime.evaluate_signal(signal, state, params))

    return _fire


def firing_episodes(control: Mapping[str, Any], signal: str,
                    *, boundary: str | None = None,
                    expr: Mapping[str, Any] | None = None) -> dict[str, int]:
    """`case_id -> how many boundaries of that episode the signal holds at`.

    Restricted to `boundary` when given. A candidate is installed AT a boundary, so support counted
    across all of them would credit a condition with firings at a point the controller never
    evaluates -- the "condition evaluated where its facts do not exist" defect
    `cctu_capabilities.py` records at -4.95pp against +3.63pp for byte-identical text one boundary
    later.

    Raises `Unscreenable` when the signal cannot be resolved, or when EVERY state it was offered
    raised. The second case is the one worth being strict about: a systematic evaluation failure
    produces the same 0 a genuinely rare condition does, and reporting it as support 0 would be a
    veto written by a bug.
    """
    fire = _predicate(signal, expr)
    out: dict[str, int] = {}
    evaluated = errors = 0
    first_error = ""
    for case_id, states in control["states"].items():
        n = 0
        for state in states:
            # `runtime.boundary_key` and NOT `state["boundary"]`. `observable_state` passes the
            # normalized event through without adding a boundary field, so a replayed state may
            # carry none -- and a string compare against a missing key would silently match nothing
            # and report support 0 for every candidate, which is a screen that vetoes everything for
            # a reason that has nothing to do with the candidates.
            if boundary is not None and runtime.boundary_key(state) != str(boundary):
                continue
            try:
                fired = fire(state)
            except Exception as exc:                                        # noqa: BLE001
                errors += 1
                first_error = first_error or f"{type(exc).__name__}: {exc}"
                continue
            evaluated += 1
            if fired:
                n += 1
        if n:
            out[case_id] = n
    if evaluated == 0:
        raise Unscreenable(
            f"signal {signal!r} was evaluable at 0 of the {errors} state(s) offered"
            + (f" at boundary {boundary!r}" if boundary else "")
            + (f"; first error: {first_error}" if first_error else
               ". No state matched that boundary, so there is nothing to screen against"))
    return out


# ------------------------------------------------------------------------------------------------
# screen 1: SUPPORT
# ------------------------------------------------------------------------------------------------
def screen_support(control: Mapping[str, Any], signal: str, *, boundary: str | None = None,
                   min_episodes: int = DEFAULT_MIN_EPISODES,
                   expr: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Does this signal fire in enough CONTROL episodes to produce a measurable delta?

    The unit is EPISODES, not events, because every metric in this project is scored per episode. A
    signal firing 500 times inside 3 episodes has support 3, and `cycle1_propose.py`'s docstring
    records what the other ordering costs: "512 events spread over far fewer episodes, and an anchor
    is judged per episode."
    """
    fires = firing_episodes(control, signal, boundary=boundary, expr=expr)
    episodes, events = len(fires), sum(fires.values())
    verdict = "PASS" if episodes >= min_episodes else "VETO"
    return {
        "screen": "support",
        "signal": signal,
        "boundary": boundary,
        "verdict": verdict,
        "episodes_firing": episodes,
        "events_firing": events,
        # THE IDS, NOT JUST THE COUNT. A round's ENGAGEMENT check asks whether the arm fired on the
        # episodes the screen said it would, and a count cannot answer that -- `rounds/CCTU_QWEN3` and
        # `rounds/CCTU_QWEN4` were both voided by a controller firing in 20 episodes against a screened
        # 18, and neither round could name the two because this report stored only `18`. With the ids
        # the check is set arithmetic and the diagnosis is immediate.
        "firing_episode_ids": sorted(fires),
        "episodes_scanned": control["episodes"],
        "min_episodes": min_episodes,
        "detail": (
            f"fires in {episodes} of {control['episodes']} control episodes ({events} events)"
            if verdict == "PASS" else
            f"fires in only {episodes} of {control['episodes']} control episodes ({events} events), "
            f"under the floor of {min_episodes}. An arm here cannot clear its own variance floor "
            f"even at 100% conversion -- this is the shape that cost cycle 1 three arms on "
            f"tool_execution_error"),
    }


# ------------------------------------------------------------------------------------------------
# screen 2: METRIC REACHABILITY
# ------------------------------------------------------------------------------------------------
def screen_reachability(control: Mapping[str, Any], signal: str, *, benefit: str = "PSR",
                        boundary: str | None = None,
                        expr: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Is `benefit` ever 1 in a control episode where this signal fires?

    THE QUESTION THIS ANSWERS. A controller can only be credited for a gain in an episode it acted
    in (the ATTRIBUTION term). So if the benefit metric is 0 in every control episode the signal can
    fire in, the arm has no attributable headroom -- every gain it ever shows will be outside the
    fired set by construction, and the attribution term will veto it after the compute is spent
    rather than before.

    A 0 here is NOT "this intervention is weak". It is "this (signal, metric) pair cannot express a
    win", which is a different and stronger statement, and it is why this screen is worth running
    even on a candidate with enormous support: cycle 1's arm 01 had 1512 firings over 198 episodes
    and a reachable headroom of exactly 0.

    `headroom` is the count that matters for sizing: control episodes where the signal fires, the
    benefit is 0, and it is therefore at least ARITHMETICALLY possible for the arm to convert one.
    It is an upper bound and nothing more -- it says a win is expressible, never that it is likely.
    """
    if benefit not in METRICS:
        raise ValueError(f"benefit must be one of {METRICS}, got {benefit!r}")
    fires = firing_episodes(control, signal, boundary=boundary, expr=expr)
    detail = control["detail"]
    scored = [c for c in fires if c in detail]
    already = [c for c in scored if bool(detail[c].get(benefit))]
    headroom = [c for c in scored if not bool(detail[c].get(benefit))]

    # THE VETO APPLIES ONLY WHERE THE WORLD HAS ALREADY MOVED, and getting this wrong made this
    # screen reject the best candidate on the corpus.
    #
    # `already == 0` has two completely different meanings and the count cannot tell them apart:
    #
    #   * AT POST_EXECUTION it means the metric is UNRECOVERABLE given the trigger. `PSR = acc AND
    #     has_if_error == 0` over every message, so once a violation is in the transcript no later
    #     action can raise it -- the gain has to land outside the fired set, and ATTRIBUTION vetoes.
    #     That is the real defect, measured in cycle 1: 198 episodes intervened, PSR=1 in none.
    #
    #   * AT THE COMMITMENT GATE it means the anchor fires only where the metric is currently 0 --
    #     which is a VIRTUE, not a defect. Nothing has been counted yet, the intervention controls
    #     what happens next, and firing only in failing episodes is exactly zero exposure to the loss
    #     pool. `rounds_remaining < 1 AND proposes_tool_call` fires in 195 episodes, every one of them
    #     `SR = 0` because every one ends `mid_tool_call` -- and an earlier version of this function
    #     VETOED it while cycle 2's five arms, which fired in all 37 already-succeeding episodes,
    #     passed.
    #
    # The distinction is whether the metric-killing event is in the past or the future when the
    # signal fires, which is the two-sub-moment distinction `cctu_middleware` already turns on: after
    # the validator the budget is spent, before it nothing is. So the veto is scoped to
    # POST_EXECUTION and the earlier boundaries report the same number as loss exposure instead.
    after_the_fact = str(boundary or "") == POST_EXEC
    if already:
        verdict = "PASS"
    elif after_the_fact:
        verdict = "VETO"
    else:
        verdict = "PASS"
    return {
        "screen": "reachability",
        "signal": signal,
        "boundary": boundary,
        "benefit": benefit,
        "verdict": verdict,
        "pinned_after_trigger": bool(after_the_fact and not already),
        "episodes_firing": len(fires),
        "episodes_firing_scored": len(scored),
        "benefit_already_1": len(already),
        "headroom": len(headroom),
        "detail": (
            f"{benefit}=0 in ALL {len(scored)} control episodes this signal fires in, and the signal "
            f"is at {boundary} where the validator has already run. The metric is PINNED under the "
            f"trigger: any gain must fall outside the episodes it acted in, which ATTRIBUTION "
            f"vetoes. Choose a benefit metric recoverable after the trigger, or a signal that fires "
            f"before the metric is lost"
            if verdict == "VETO" else
            f"{benefit}=1 in {len(already)} of the {len(scored)} control episodes this signal fires "
            f"in; {len(headroom)} are at {benefit}=0 and are the arithmetic headroom"
            if already else
            f"{benefit}=0 in ALL {len(scored)} control episodes this signal fires in, but the signal "
            f"fires at {boundary} -- BEFORE the validator -- so the metric is not yet lost and this "
            f"is zero exposure to the loss pool, not a pinned metric. See screen_loss_exposure"),
    }


# ------------------------------------------------------------------------------------------------
# screen 3: LOSS EXPOSURE
# ------------------------------------------------------------------------------------------------
def screen_loss_exposure(control: Mapping[str, Any], signal: str, *, benefit: str = "SR",
                         boundary: str | None = None,
                         expr: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Does this condition fire in episodes that are ALREADY SUCCEEDING?

    THE TERM THAT DECIDED CYCLE 2, and the one neither other screen asks. Support asks whether a
    candidate fires enough; reachability asks whether the metric can rise where it fires. Both are
    about the upside. Neither notices that a condition firing everywhere is also exposed to every
    episode it could BREAK.

    The arithmetic, on the granite train control:

        loss pool   37 episodes at SR = 1        -- can only go down
        gain pool   26 episodes at acc = 1, SR = 0 -- can go up

    Cycle 2's five arms triggered on `proposes_tool_call`, which fires in all 280 episodes. They were
    betting 26 reachable gains against 37 exposed losses, and the results came in at 8 gains / 10
    losses, 4 / 12, 5 / 4 -- a null with real `acc` regressions. **No conversion rate wins that bet
    unless it exceeds the destruction rate**, and that was computable before any arm ran.

    So the veto is the comparison itself: a candidate whose loss exposure is at least its gain
    exposure cannot show a net gain except by firing more accurately inside the loss pool than inside
    the gain pool, which is a claim no control supports in advance.

    ZERO LOSS EXPOSURE IS THE PROPERTY TO DESIGN FOR, and it is reachable rather than aspirational:
    `rounds_remaining < 1 AND proposes_tool_call` fires at 195 turns in 195 episodes, ALL of which
    terminate `mid_tool_call` and are therefore already `SR = 0`. Not luck -- an episode that scores
    SR ends with a final answer, so it never spends its last round on a tool call. Its net is then
    bounded below by zero and it needs only to convert 5 of the 18 it can reach.

    `gain_exposure` requires `acc = 1` for `SR` and `PSR`, because `judge` makes both conjunctions
    with `acc`: an episode that never retrieved the answer cannot gain SR by any constraint fix.
    """
    if benefit not in METRICS:
        raise ValueError(f"benefit must be one of {METRICS}, got {benefit!r}")
    fires = firing_episodes(control, signal, boundary=boundary, expr=expr)
    detail = control["detail"]
    scored = [c for c in fires if c in detail]

    def _loses(row: Mapping[str, Any]) -> bool:
        return bool(row.get(benefit))

    def _gains(row: Mapping[str, Any]) -> bool:
        if bool(row.get(benefit)):
            return False
        return True if benefit == "acc" else bool(row.get("acc"))

    loss = [c for c in scored if _loses(detail[c])]
    gain = [c for c in scored if _gains(detail[c])]
    pool_loss = sum(1 for r in detail.values() if _loses(r))
    pool_gain = sum(1 for r in detail.values() if _gains(r))
    verdict = "VETO" if len(loss) >= max(1, len(gain)) else "PASS"
    return {
        "screen": "loss_exposure",
        "signal": signal,
        "boundary": boundary,
        "benefit": benefit,
        "verdict": verdict,
        "loss_exposure": len(loss),
        "gain_exposure": len(gain),
        "episodes_firing_scored": len(scored),
        "corpus_loss_pool": pool_loss,
        "corpus_gain_pool": pool_gain,
        "detail": (
            f"fires in {len(loss)} of the {pool_loss} already-succeeding ({benefit}=1) episodes and "
            f"{len(gain)} of the {pool_gain} convertible ones"
            + ("" if verdict == "PASS" else
               f" -- loss exposure is not smaller than gain exposure, so a net gain requires firing "
               f"more accurately inside the loss pool than the gain pool, which no control supports "
               f"in advance. This is the shape that sank cycle 2's five arms (26 reachable against 37 "
               f"exposed)")),
    }


# ------------------------------------------------------------------------------------------------
# screen 4: REMOVABLE MASS -- the count round's own structural screen
# ------------------------------------------------------------------------------------------------
def screen_removable_mass(control: Mapping[str, Any], signal: str, *,
                          classes: Sequence[str], boundary: str | None = None,
                          expr: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """For a COUNT benefit: how much of the target mass is in reach, and where can it only add?

    THE OTHER THREE SCREENS CANNOT SCREEN A COUNT ROUND, and the reason is measured rather than
    argued. They all partition episodes by a BINARY metric, and the count risk does not live in that
    partition. On the granite control:

        arm 121   total +1130   of which +80 in PSR=1 episodes,  +1050 in PSR=0
        arm 126   total  -548   of which  +8 in PSR=1 episodes,   -556 in PSR=0

    93% of arm 121's damage and essentially all of arm 126's benefit landed in `PSR = 0` episodes --
    episodes that ALREADY violate, violating more or less. A loss-exposure screen against `PSR`
    catches only the 28 episodes with nothing to remove, so for a count benefit it is a weak filter
    and not a sound one. This screen asks the two questions that are structural for a count:

        removable_mass     target-class violations summed over the episodes the signal fires in.
                           Zero means the candidate cannot reduce these classes at all, whatever it
                           does -- the count analogue of a pinned metric.
        pure_downside      firing episodes carrying NONE of the target classes. There the anchor has
                           nothing to remove and can only add, which is the count analogue of loss
                           exposure.

    `classes` is REQUIRED and has no default. A count claim is per class -- several classes on granite
    cannot carry one at all (`min_length` 31% replicate noise) while all twelve can on qwen -- so a
    screen that guessed the target would be guessing the round's benefit metric.

    WHAT IT STILL CANNOT SEE. Whether the intervention ADDS violations is behavioural, not structural:
    no control can answer it in advance. `analyze_residual`'s per-class net plus its up/down sign test
    is what answers it afterwards. This screen bounds the upside and names the pure-downside
    population; it does not predict the sign.
    """
    classes = tuple(classes)
    if not classes:
        raise ValueError("screen_removable_mass requires at least one target class")
    fires = firing_episodes(control, signal, boundary=boundary, expr=expr)
    detail = control["detail"]
    scored = [c for c in fires if c in detail]

    def _mass(row: Mapping[str, Any]) -> int:
        counts = row.get("violations") or {}
        return sum(int(counts.get(c, 0)) for c in classes)

    if scored and "violations" not in detail[scored[0]]:
        raise Unscreenable(
            "the control's detail.jsonl carries no `violations` field, so no count screen can run. "
            "Re-score it with `evaluation.py --detail --overload` -- this needs no re-run, only a "
            "re-score of the existing response.jsonl")

    reachable = sum(_mass(detail[c]) for c in scored)
    corpus = sum(_mass(r) for r in detail.values())
    downside = [c for c in scored if _mass(detail[c]) == 0]
    with_mass = [c for c in scored if _mass(detail[c]) > 0]
    if reachable == 0:
        verdict = "VETO"
    elif len(downside) > len(with_mass):
        verdict = "VETO"
    else:
        verdict = "PASS"
    return {
        "screen": "removable_mass",
        "signal": signal,
        "boundary": boundary,
        "classes": list(classes),
        "verdict": verdict,
        "removable_mass": reachable,
        "corpus_mass": corpus,
        "mass_share": round(reachable / corpus, 4) if corpus else 0.0,
        "episodes_with_mass": len(with_mass),
        "pure_downside_episodes": len(downside),
        # Same reason as `screen_support`'s `firing_episode_ids`. Split here because the two sets answer
        # different questions: `with_mass` is what an arm can remove from, `downside` is where it can
        # only add.
        "episodes_with_mass_ids": sorted(with_mass),
        "pure_downside_episode_ids": sorted(downside),
        "detail": (
            f"reaches {reachable} of {corpus} {'+'.join(classes)} events "
            f"({100 * reachable / corpus:.0f}%) across {len(with_mass)} episodes, and fires in "
            f"{len(downside)} episode(s) carrying none of them"
            if verdict == "PASS" else
            f"reaches NONE of the {corpus} {'+'.join(classes)} events -- it cannot reduce these "
            f"classes whatever it does"
            if reachable == 0 else
            f"fires in {len(downside)} episodes with none of {'+'.join(classes)} against only "
            f"{len(with_mass)} carrying some: more places to ADD than to remove, which is the count "
            f"analogue of loss exposure exceeding gain exposure"),
    }


# ------------------------------------------------------------------------------------------------
# screening a candidate set
# ------------------------------------------------------------------------------------------------
def _candidate_rows(path: Path) -> list[dict[str, Any]]:
    """Candidates from a `candidates/` directory, a propose.json, or one spec file.

    Reads only `controller_id`, `signal` and `boundary` -- the three fields a screen needs. It does
    not validate a spec; `ControllerSpec.validate` owns that and runs at load time in the runner.
    """
    files: list[Path]
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    else:
        files = [path]
    rows: list[dict[str, Any]] = []
    for f in files:
        payload = json.loads(f.read_text())
        blocks = (payload.get("candidates") if isinstance(payload, dict) and "candidates" in payload
                  else payload.get("controllers") if isinstance(payload, dict)
                  and "controllers" in payload else [payload])
        for block in blocks or ():
            if not isinstance(block, Mapping):
                continue
            # A spec carries EITHER a declared `signal` name OR a `predicate` expression, never
            # both -- `ControllerSpec.validate` enforces that. Reading only `signal` would report
            # every synthesized candidate as "no signal named" and SKIP it, which is how a
            # persistable condition gets mistaken for an unnameable one.
            predicate = dict(block.get("predicate") or {})
            name = (block.get("signal") or block.get("phi") or block.get("phi_name")
                    or predicate.get("name"))
            rows.append({
                "controller_id": block.get("controller_id") or block.get("id") or f.stem,
                "signal": name,
                "predicate": predicate,
                "boundary": block.get("boundary"),
                "source": str(f),
            })
    return rows


def screen_candidates(control: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], *,
                      benefit: str = "PSR",
                      min_episodes: int = DEFAULT_MIN_EPISODES,
                      count_classes: Sequence[str] = ()) -> dict[str, Any]:
    """All three screens over every candidate: SUPPORT, REACHABILITY, LOSS EXPOSURE. Each vetoes alone.

    Reported per candidate rather than as a filtered list, because a screen that silently dropped
    candidates would leave a reader unable to tell a vetoed proposal from one the search never made
    -- the same reason `synthesis_reachable_classes()` exists.
    """
    results = []
    for cand in candidates:
        signal, boundary = cand.get("signal"), cand.get("boundary")
        expr = dict(cand.get("predicate") or {}) or None
        if not signal and not expr:
            results.append({**dict(cand), "verdict": "SKIP",
                            "detail": "no signal named; nothing to screen"})
            continue
        try:
            support = screen_support(control, signal, boundary=boundary,
                                     min_episodes=min_episodes, expr=expr)
            reach = screen_reachability(control, signal, benefit=benefit, boundary=boundary,
                                        expr=expr)
            loss = screen_loss_exposure(control, signal, benefit=benefit, boundary=boundary,
                                        expr=expr)
            # Only when the round declares target classes -- a count screen with no named class
            # would be guessing the benefit metric.
            mass = (screen_removable_mass(control, signal, classes=count_classes,
                                          boundary=boundary, expr=expr)
                    if count_classes else None)
        except Unscreenable as exc:
            # NOT a veto. Reported as its own verdict so a reader can tell "the control refuses this
            # candidate" from "this predicate does not exist yet".
            results.append({**dict(cand), "verdict": "UNSCREENABLE", "detail": str(exc)})
            continue
        terms = [t for t in (support, reach, loss, mass) if t]
        vetoes = [t["screen"] for t in terms if t["verdict"] == "VETO"]
        row = {**dict(cand), "verdict": "VETO" if vetoes else "PASS",
               "vetoed_by": vetoes, "support": support, "reachability": reach,
               "loss_exposure": loss}
        if mass:
            row["removable_mass"] = mass
        results.append(row)
    # THE REPORT IS IN THE PROPOSER'S EMISSION ORDER, and the screens deliberately do not rank it.
    #
    # A `_report_order` key was added here and is removed again. Its motivation was real: the driver
    # queues PASS rows in report order under an arm budget, and cycle 4's first dry run spent all six
    # arms on the `constraint_violation_reported` family that cycles 1 and 3 had already measured null,
    # purely because the grammar emitted it first. But ranking the report IS choosing which arms the
    # round spends, and `tests/test_cctu_screens.py` holds the screens to not doing that: a screen may
    # not choose WHERE to intervene, WHAT decides, or HOW to act. "It changes no verdict" is true and
    # beside the point when a budget truncates the list.
    #
    # The selection rule belongs to the ROUND, pre-registered in its FROZEN.md next to the arm budget,
    # where a reader can see it was chosen before the results. `cctu_run_arms.py --candidates` takes a
    # directory, so a round applies its stated rule to this report and queues the result -- mechanical,
    # reproducible, and on the record. See that file's `--max-arms` help.
    passed = [r for r in results if r["verdict"] == "PASS"]
    return {
        "run": control["run"], "split": control["split"], "benefit": benefit,
        "min_episodes": min_episodes, "count_classes": list(count_classes),
        "episodes_scanned": control["episodes"],
        "candidates": len(results), "passed": len(passed),
        "vetoed": len(results) - len(passed),
        "results": results,
        "replay_errors": control["replay_errors"][:20],
    }


# ------------------------------------------------------------------------------------------------
# printing
# ------------------------------------------------------------------------------------------------
def print_report(report: Mapping[str, Any]) -> None:
    print("=" * W)
    print(f"PRE-ARM SCREENS  control={report['run']}  split={report['split']}  "
          f"benefit={report['benefit']}")
    print("=" * W)
    print(f"  episodes scanned={report['episodes_scanned']}  support floor="
          f"{report['min_episodes']} episodes")
    if report["replay_errors"]:
        print(f"  [warn] {len(report['replay_errors'])} replay error(s); first: "
              f"{report['replay_errors'][0]}")
    print()
    for r in report["results"]:
        print(f"  [{r['verdict']}] {r.get('controller_id')}")
        print(f"         signal={r.get('signal')}  boundary={r.get('boundary')}")
        if r.get("detail") and not r.get("support"):
            print(f"         {r['detail']}")
        for key in ("support", "reachability", "loss_exposure", "removable_mass"):
            s = r.get(key)
            if not s:
                continue
            print(f"         {s['verdict']:<4} {key:<13} {s['detail']}")
        print()
    print("-" * W)
    print(f"  {report['passed']} of {report['candidates']} candidates pass ALL THREE screens; "
          f"{report['vetoed']} vetoed")
    if report["vetoed"]:
        by = collections.Counter(s for r in report["results"] for s in r.get("vetoed_by", ()))
        print("  vetoed by: " + ", ".join(f"{k}={v}" for k, v in by.most_common()))
    print("=" * W)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pre-arm screens for CCTU candidates. No model, no GPU, no network.")
    ap.add_argument("--run", required=True, help="the CONTROL run directory to screen against")
    ap.add_argument("--candidates", required=True,
                    help="a candidates/ directory, a propose.json, or one controller spec")
    ap.add_argument("--benefit", default="PSR", choices=METRICS,
                    help="the metric the round intends to MOVE (default PSR)")
    ap.add_argument("--classes", default="",
                    help="comma-separated violation classes the round intends to REDUCE. Enables the "
                         "removable-mass screen, which is the only one of the four that screens a "
                         "COUNT benefit. No default: a count claim is per class, and guessing the "
                         "target would be guessing the round's benefit metric")
    ap.add_argument("--min-episodes", type=int, default=DEFAULT_MIN_EPISODES,
                    help=f"support floor in EPISODES (default {DEFAULT_MIN_EPISODES}); pass a floor "
                         f"you measured from the control's own flip count")
    ap.add_argument("--data-dir", default=None, help="corpus directory (default benchmarks/cctu/data)")
    ap.add_argument("--json", dest="json_out", default=None, help="write the full report here")
    args = ap.parse_args()

    control = load_control(Path(args.run),
                           data_dir=Path(args.data_dir) if args.data_dir else None)
    candidates = _candidate_rows(Path(args.candidates))
    if not candidates:
        raise SystemExit(f"no candidates found at {args.candidates}")
    report = screen_candidates(control, candidates, benefit=args.benefit,
                               min_episodes=args.min_episodes,
                               count_classes=tuple(c.strip() for c in args.classes.split(",")
                                                   if c.strip()))
    print_report(report)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True, default=str)
                                       + "\n")
        print(f"  report -> {args.json_out}")
    # EXIT NON-ZERO WHEN EVERY CANDIDATE IS VETOED. A screening pass that refuses the whole set is a
    # result the round must act on, not a warning to scroll past.
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
