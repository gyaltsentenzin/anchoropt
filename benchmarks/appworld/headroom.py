"""How much room does a signal actually leave for an intervention? Answered from the incumbent's own
logs, before any arm is scored.

WHY THIS EXISTS
==================================================================================================
The `post_execution/execution_failed/reprompt` arm injects "the last cell failed; read the error above
and issue a corrected call." It measured net +1 on one split and 0/-3 elsewhere, and an episode-level
audit found the reason: **the agent already does that, unprompted.** Across the qwen `sh_heldout`
incumbent, the cell after a failing cell already succeeded 208/371 times (56%), and after the FIRST
failure of an episode -- the exact condition the signal fires on -- 54/69 times (78%). No next cell was
ever a byte-identical retry. The instruction had almost no room to add anything.

That was discovered after two scoring rounds and ~360 real episodes. It is computable from the
incumbent's logs alone, for free, in seconds. This file does that.

WHAT IT COMPUTES, AND WHAT THE NUMBER MEANS
==================================================================================================
For each declared signal, on the INCUMBENT's own trajectories:

  firing episodes      how many episodes the signal would fire on at all
  self-resolved        of those, how many the incumbent ALREADY recovers from at the next step
  headroom             firing episodes - self-resolved  <- the arm's ceiling, in episodes

`headroom` is an upper bound and a loose one: it counts every episode where the incumbent did NOT
immediately recover as one the intervention could theoretically rescue. A mechanism cannot beat it.
So when headroom is below the measurement's noise floor, the arm is unresolvable BEFORE it is run, and
no amount of signal refinement inside that boundary changes it.

This is a screen, not a verdict. High headroom does not predict success; low headroom does predict
unmeasurability. Used the cheap way round, it is the least expensive filter available.

CAVEATS, STATED RATHER THAN BURIED
==================================================================================================
* "Self-resolved" is "the next cell succeeded", which is necessary but not sufficient for the EPISODE
  to succeed. It measures local recovery, which is what a per-step reprompt competes with.
* The incumbent is one sampled run. On a nondeterministic server these rates carry the same sampling
  error as everything else; treat a difference of a few points as noise.
* Firing is evaluated with the adapter's own `evaluate_signal` against the adapter's own mined records,
  so it agrees with what the live executor will do by construction -- not by a reimplementation.
* Per-episode budget is not modelled beyond `--first-firing-only`, which matches `retry_budget: 1`.

Usage:
  headroom.py --evaluation <evaluations/<split>.json> [--experiment <name>]
              [--signal execution_failed] [--noise-floor 5] [--all-firings]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

# `mine_task` resolves logs through `path_store` itself, so this module needs no AppWorld import of its
# own -- and must not add one, since importing path_store here would fix the root at import time.
from adapter import ADAPTER, _SIGNAL_BOUNDARIES
from residual import _infer_experiment_name, mine_task

_PE = "post_execution"


_PG = "post_generation_pre_exec"


class HeadroomUnavailable(Exception):
    """This screen cannot produce a meaningful number for this signal.

    Raised rather than exiting, because `run_round.py propose` calls `screen()` for every proposed arm
    and must carry on when one of them is not screenable.
    """


# WHAT "ALREADY RECOVERED" MEANS, AND WHY IT IS THE SAME TEST AT BOTH BOUNDARIES
# --------------------------------------------------------------------------------------------------
# A step in `mine_task`'s output is a (PRE, PG, PE) triple, emitted in order, so the i-th PG record and
# the i-th PE record describe the same step. Recovery is always "did the execution of the NEXT step
# succeed?", i.e. `pe[i + 1]`:
#
#   POST_EXECUTION signal at step i  -- the condition IS pe[i] (a failure that already happened), and
#                                       the incumbent's own next attempt is pe[i+1].
#   GATE signal at step i            -- the condition is pg[i], observed BEFORE pe[i] runs. An
#                                       intervention here rewrites step i, so what it competes against
#                                       is the incumbent burning step i and repairing itself at
#                                       pe[i+1]. Same look-ahead, same denominator.
#
# Keeping one definition is deliberate: two boundary-specific notions of "recovered" would not be
# comparable, and the whole point of the number is to compare ceilings across candidate boundaries.
_LOOKAHEAD_NOTE = "recovery = the next step's execution succeeded"


def _all_task_ids(evaluation_path: str) -> list[str]:
    """Every task in the split, not just the failing ones. `mine_residual` deliberately mines only the
    residual; a signal fires on passing episodes too, and those are exactly where a loss can appear."""
    with open(evaluation_path, encoding="utf-8") as f:
        payload = json.load(f)
    individual = payload.get("individual") or {}
    if not individual:
        raise SystemExit(f"{evaluation_path} has no 'individual' block to enumerate tasks from")
    return sorted(individual)


def _passed(evaluation_path: str) -> dict[str, bool]:
    with open(evaluation_path, encoding="utf-8") as f:
        individual = (json.load(f).get("individual") or {})
    return {t: bool(v.get("success")) for t, v in individual.items()}


def screen(evaluation_path: str, experiment_name: str, signal: str,
           first_firing_only: bool = True) -> dict[str, Any]:
    # `_SIGNAL_BOUNDARIES` holds raw `IncisionPoint` members while `mine_task` keys its records by
    # `IncisionPoint.*.value`. Normalise to the string form rather than assuming either.
    boundaries = {getattr(b, "value", str(b)) for b in (_SIGNAL_BOUNDARIES.get(signal) or frozenset())}
    if _PE in boundaries:
        fires_at = _PE
    elif _PG in boundaries:
        fires_at = _PG
    else:
        # PRE_GENERATION signals condition on step_number / consecutive_errors / last_error_kind, which
        # describe the run so far rather than a repairable event at this step. "Already recovered" has
        # no counterpart there, so refuse rather than report a number whose meaning does not match its
        # name.
        raise HeadroomUnavailable(
            f"signal {signal!r} is declared at {sorted(boundaries) or ['nothing']}; this screen covers "
            f"{_PE} and {_PG} signals, where '{_LOOKAHEAD_NOTE}' is meaningful"
        )

    task_ids = _all_task_ids(evaluation_path)
    passed = _passed(evaluation_path)

    fired_episodes: list[str] = []
    resolved_episodes: list[str] = []
    per_class: Counter[str] = Counter()
    per_class_resolved: Counter[str] = Counter()
    opportunities = 0
    missing: list[str] = []

    for task_id in task_ids:
        try:
            records = mine_task(experiment_name, task_id)
        except FileNotFoundError:
            missing.append(task_id)
            continue

        pe = [r for r in records if r.get("kind") == _PE]
        # The i-th record of each boundary describes the same step, so one index serves both.
        trigger = pe if fires_at == _PE else [r for r in records if r.get("kind") == _PG]
        fired_here = False
        for i, rec in enumerate(trigger):
            if not ADAPTER.evaluate_signal(signal, rec):
                continue
            opportunities += 1
            if first_firing_only and fired_here:
                continue
            # Classify by the error this step actually produced, whichever boundary fired. For a gate
            # signal that is the outcome of the cell the gate saw -- e.g. `generation_truncated` should
            # land on `no_code`, which doubles as a check that the two describe the same condition.
            same_step_pe = pe[i] if i < len(pe) else {}
            kind = str((rec if fires_at == _PE else same_step_pe).get("error_kind") or "")
            nxt = pe[i + 1] if i + 1 < len(pe) else None
            # No successor at all: the episode ended on this failure, so the incumbent did NOT
            # locally recover. Counted as headroom, which is the conservative direction for a ceiling.
            recovered = bool(nxt and nxt.get("succeeded"))
            if not fired_here:
                fired_episodes.append(task_id)
                per_class[kind] += 1
                if recovered:
                    resolved_episodes.append(task_id)
                    per_class_resolved[kind] += 1
            fired_here = True

    # TWO CEILINGS, BECAUSE ONE OF THEM IS ONLY VALID FOR ERROR-SHAPED SIGNALS
    # ----------------------------------------------------------------------------------------------
    # `headroom_local` = firings - locally self-resolved. It assumes the condition IS a failure and
    # that repairing it shows up as the next step succeeding. True for `execution_failed`. FALSE for a
    # completeness signal: a cell calling a paginated API without `page_index` executes perfectly and
    # returns page 1, so 97% of `pagination_unbounded` firings look "self-resolved" while the episode
    # silently goes on to fail the task. Reported here as a false negative that this screen produced
    # on its first real use, not smoothed away.
    #
    # `headroom_task` = firings on episodes the incumbent FAILS. An intervention cannot gain on a task
    # already passing, so this bounds gains for ANY signal shape, error or completeness. It is looser
    # where `headroom_local` is valid, and it is the only one of the two that is always meaningful.
    #
    # Read them as: headroom_task is the ceiling; headroom_local tightens it only when the signal names
    # something that actually errors.
    fired_and_passed = sum(1 for t in fired_episodes if passed.get(t))
    return {
        "signal": signal,
        "fires_at": fires_at,
        "experiment": experiment_name,
        "headroom_task": len(fired_episodes) - fired_and_passed,
        "loss_exposure": fired_and_passed,
        "tasks": len(task_ids),
        "logs_missing": missing,
        "firing_opportunities": opportunities,
        "firing_episodes": len(fired_episodes),
        "self_resolved": len(resolved_episodes),
        "headroom": len(fired_episodes) - len(resolved_episodes),   # == headroom_local
        "per_class": dict(per_class),
        "per_class_resolved": dict(per_class_resolved),
        "fired_and_incumbent_passed": sum(1 for t in fired_episodes if passed.get(t)),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", required=True, help="the incumbent's evaluations/<split>.json")
    p.add_argument("--experiment", default=None, help="override the inferred experiment_name")
    p.add_argument("--signal", default="execution_failed", help="declared signal to screen")
    p.add_argument("--noise-floor", type=int, default=5,
                   help="task-count noise floor for this corpus; the screen calls an arm "
                        "unresolvable when its ceiling does not exceed this (default 5, the spread "
                        "measured between same-config baseline repeats on 90 AppWorld tasks)")
    p.add_argument("--all-firings", action="store_true",
                   help="count every firing rather than the first per episode; the default matches "
                        "retry_budget=1")
    p.add_argument("--json", dest="as_json", default=None, help="also write the report here")
    args = p.parse_args(argv)

    experiment = args.experiment or _infer_experiment_name(args.evaluation)
    rep = screen(args.evaluation, experiment, args.signal,
                 first_firing_only=not args.all_firings)

    n, fired, res, head = rep["tasks"], rep["firing_episodes"], rep["self_resolved"], rep["headroom"]
    print(f"signal      : {rep['signal']}  @ {rep['fires_at']}   ({_LOOKAHEAD_NOTE})")
    print(f"experiment  : {experiment}")
    print(f"tasks       : {n}" + (f"   ({len(rep['logs_missing'])} without logs, skipped)"
                                  if rep["logs_missing"] else ""))
    print(f"firings     : {fired}/{n} episodes ({rep['firing_opportunities']} total opportunities)")
    if fired:
        print(f"self-resolved: {res}/{fired} ({100.0*res/fired:.0f}%) -- the incumbent ALREADY "
              f"recovered at the next step")
        print(f"HEADROOM    : local {head}   task {rep['headroom_task']}   "
              f"(loss exposure {rep['loss_exposure']})")
        print( "              local = firings - locally self-resolved; VALID ONLY if this signal names")
        print( "                      something that errors. A completeness signal executes fine and")
        print( "                      fails the task silently, so local reads ~0 and is meaningless.")
        print( "              task  = firings on episodes the incumbent FAILS; bounds gains for ANY")
        print( "                      signal shape, and is the ceiling to trust when in doubt.")
        print()
        print("  per error class (first firing per episode):")
        for k, c in sorted(rep["per_class"].items(), key=lambda kv: -kv[1]):
            r = rep["per_class_resolved"].get(k, 0)
            print(f"    {k:<20} {c:>3} firings, {r:>3} self-resolved ({100.0*r/c:.0f}%), "
                  f"headroom {c-r}")
        print()
        print(f"  {rep['fired_and_incumbent_passed']}/{fired} firing episodes are ones the incumbent "
              f"PASSES -- where a loss, not a gain, is what can appear")

    print()
    if not fired:
        print("VERDICT: the signal never fires on this incumbent. Nothing to measure.")
    elif max(head, rep["headroom_task"]) <= args.noise_floor:
        print(f"VERDICT: UNRESOLVABLE. Both ceilings <= noise floor {args.noise_floor}. Even a "
              f"mechanism that rescued every non-self-resolving episode could not produce an effect "
              f"this corpus can distinguish from sampling. Refining the signal at this boundary does "
              f"not help -- refinement only ever shrinks the firing set.")
    else:
        print(f"VERDICT: resolvable in principle. Ceiling (local {head}, task "
              f"{rep['headroom_task']}) > noise floor {args.noise_floor}. "
              f"Note this is an upper bound and a loose one; it does not predict that the mechanism "
              f"works.")

    if args.as_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.as_json)), exist_ok=True)
        with open(args.as_json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
            f.write("\n")
        print(f"\nwrote {args.as_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
