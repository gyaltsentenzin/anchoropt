"""The round driver: `propose` advances the search by exactly one step and, when it produces new
candidates, writes the manifest an out-of-process scorer must run; `score` is that scorer.

WHY "PROPOSE" IS A SINGLE STEP, NOT A FIXED TWO-PASS SCRIPT
--------------------------------------------------------------------------------------------------
The plan originally described this driver as "pass 1: cold-cache sweep the whole frontier" then
"pass 2: warm-cache detects NO_BENEFIT and expands" -- two fixed invocations. Direct measurement
against `optimize_residual` (`evaluate.py`'s module docstring has the full derivation) falsified
that: a cold `ExternalEvaluation` and a populated one both halt at the FIRST realizable-but-
unmeasured boundary, whichever call encounters it. So `propose` is not "pass N" of a fixed script --
it is "run the search once more against whatever has been measured so far, and see how far it gets
this time." Some invocations will resolve a previously-unmeasured arm into `IMPROVED` or `NO_BENEFIT`
and then, WITHIN THE SAME CALL, continue sweeping or expanding until they hit the next thing nobody
has measured yet; others will make no progress beyond restating the same unmeasured frontier because
no new results were supplied since the last call. The caller (a human running this repeatedly, or a
future outer loop) drives the cadence; this file does not manufacture a fixed pass count.

WHY "SCORE" IS A SEPARATE COMMAND, NOT SOMETHING "PROPOSE" CALLS
--------------------------------------------------------------------------------------------------
`propose` is free: it mines an already-written log and calls a pure search function. `score` is not
-- it drives `AppWorldReActAgent` through real episodes against a real `ibm-granite/granite-4.1-30b`
server (see the model's `dev.jsonnet` config's `base_url`), which means real inference
cost against a real model-serving process for every task_id in scope. Keeping it a separate,
explicitly-invoked command means nothing in this file ever triggers that cost as a side effect of
proposing arms; a caller reads the manifest, decides how much of it to afford to measure, and invokes
`score` deliberately (optionally with `--limit` for a cheap partial measurement).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from typing import Any

from appworld.common.io import jsonnet_load
from appworld.common.path_store import path_store
from appworld.evaluator import evaluate_tasks

from anchoropt.learning.external_evaluation import ArmResult
from anchoropt.learning.search_state import next_coordinate
from anchoropt.learning.structured_search import optimize_residual

from adapter import ADAPTER
from agent_hooks import AppWorldReActAgent
from controller import controller_from_spec
from evaluate import (
    incumbent_identity,
    load_evaluation,
    load_results,
    write_manifest,
    write_results,
)
from residual import _infer_experiment_name, failing_task_ids, mine_residual, passing_task_ids


def _default_out_dir(evaluation_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(evaluation_path)), "..", "anchoropt_round")


# ================================================================================================
# propose -- one step of search against whatever has been measured so far
# ================================================================================================

def cmd_propose(args: argparse.Namespace) -> int:
    out_dir = args.out or _default_out_dir(args.evaluation)
    results_path = args.results or os.path.join(out_dir, "arm_results.json")

    experiment_name = args.experiment or _infer_experiment_name(args.evaluation)
    residual, records = mine_residual(args.evaluation, experiment_name)
    incumbent_id, incumbent_token = incumbent_identity(args.evaluation, experiment_name)

    evaluation = load_evaluation(results_path, incumbent_token=incumbent_token, budget=args.budget)

    # `improves` IS ADAPTER-SUPPLIED, SO POWER CAN BE REQUIRED WITHOUT TOUCHING CORE
    # ----------------------------------------------------------------------------------------------
    # `net > 0` alone promoted, on this benchmark: an arm that fired 6/90 times with 2 gains and 0
    # losses (+2, which measured -1 on a repeat and 0 on held-out), and an arm at +1 against a
    # same-config baseline spread of 2-5 tasks. It also REWARDS sparse firing -- a rarely-firing arm has
    # almost no fired episodes on which attributable losses can accumulate, so the 6-firing arm beat a
    # 53-firing arm measured at -3.
    #
    # And promotion HALTS the round. So an unsupported promotion does not merely mislabel itself, it
    # stops the search: on qwen held-in the +1 promotes at `post_execution` and the gate boundary is
    # never visited, even though `generation_truncated` there has 46 firings and a ceiling of 25 against
    # `execution_failed`'s 11. The cost of the missing gate is the arm we never got to try.
    #
    # Both thresholds default to 0, i.e. OFF, so every result measured before this remains reproducible
    # and no existing round changes meaning. They are declared per-invocation rather than hardcoded
    # because only the caller knows its corpus's noise floor -- the same reasoning
    # `termination.channel_integrity` gives for leaving `min_fraction_on_fired` unset.
    min_net, min_fired = args.min_net, args.min_firings

    # `--net-from-draws`: JUDGE AN ARM ON THE MEAN OVER EVERY DRAW, NOT ON ONE
    # ----------------------------------------------------------------------------------------------
    # `ArmResult.net` is gains-minus-losses against the ONE incumbent draw the arm was scored against.
    # On this corpus that is the dominant bias: minimax has three same-config draws (58, 60, 61 passed)
    # and arms were scored against the 61, the maximum, so every arm reads systematically low. On an
    # episode the controller never fired the agent behaves identically to the incumbent apart from
    # sampling, so those flips are resampling noise, and pairing against one draw instead of the
    # distribution makes that noise one-directional.
    #
    # This is A DECISION RULE, NOT A CORRECTION, and it is off by default so every prior round stays
    # reproducible. It changes verdicts in BOTH directions -- on minimax the pagination treatment goes
    # +4 -> +5.33 while its control goes -8 -> -1.83 and execution_failed -3 -> -1.33 -- which is the
    # evidence that it is a pairing fix rather than a thumb on the scale. Applied to every arm it can be
    # computed for, never to a chosen one.
    #
    # Provenance matters here: a rule changed after seeing which arm it promotes is indistinguishable
    # from p-hacking. This one predates the round it affects (commits 156ec3b and ea0e616). Anyone
    # reporting a promotion that depends on it should say so.
    #
    # Imported lazily and inside this branch only, so `score` -- which runs in a job that re-invokes this
    # file once per arm -- never imports the analysis stack and cannot be broken by it.
    # KEYED ON gains/losses, NOT ON arm_label, AND THAT IS FORCED BY CORE'S SEAM
    # ----------------------------------------------------------------------------------------------
    # `improves` does not receive the `ArmResult`. `ArmResult.as_theta_result()` projects it to a
    # `ThetaResult` (policy_class.py:225), which carries theta/gains/losses/cases_fired/n but drops
    # `arm_label` -- which is why this function's own diagnostics have always printed the arm as "?".
    # So an override table keyed by arm_label can never be looked up here.
    #
    # `gains`/`losses` ARE carried, and they are case-ID tuples measured against one specific incumbent,
    # so (gains, losses, cases_fired) identifies the recorded result uniquely. The join therefore goes:
    # arm_label -> mean net (from the draw analysis) and arm_label -> recorded gains/losses (from the
    # results file this same call is reading), giving a fingerprint the seam can actually match on.
    net_override: dict[tuple, dict] = {}
    if getattr(args, "net_from_draws", None):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
        from draw_mean_nets import mean_nets  # noqa: PLC0415
        draw_nets = mean_nets(args.net_from_draws)
        if not draw_nets:
            raise SystemExit(
                f"--net-from-draws {args.net_from_draws!r} produced no arms. Either no incumbent draw is "
                f"on disk for that model, or no arm tree is cached: run runs/multi_draw_nets.py first."
            )
        recorded = load_results(results_path, incumbent_token=incumbent_token)
        print(f"[net-from-draws] model {args.net_from_draws!r}: mean net over every "
              f"(arm pass x incumbent draw) comparison, replacing the single-draw net")
        matched = 0
        for label, r in sorted(draw_nets.items(), key=lambda kv: -kv[1]["mean_net"]):
            rec = recorded.get(label)
            flag = ""
            if rec is None:
                flag = "   [not in this results file -- override will not apply]"
            else:
                key = (tuple(sorted(rec.gains)), tuple(sorted(rec.losses)), int(rec.cases_fired))
                net_override[key] = {"mean_net": r["mean_net"], "n": r["n_comparisons"], "label": label}
                matched += 1
                # `net` is a ThetaResult property, not an ArmResult field -- compute it from the lists.
                flag = f"   (single-draw was {len(rec.gains) - len(rec.losses):+d})"
            print(f"  {r['mean_net']:+6.2f} over {r['n_comparisons']} comparison(s) "
                  f"(range {r['min_net']:+d}..{r['max_net']:+d}, "
                  f"{'sign-consistent' if r['sign_consistent'] else 'SIGN FLIPS'})  {label}{flag}")
        print(f"[net-from-draws] {matched} of {len(draw_nets)} arm(s) matched into the override table")

    def _net_of(res: Any) -> tuple[float, str]:
        """The net the gate should test, and where it came from."""
        key = (tuple(sorted(getattr(res, "gains", ()) or ())),
               tuple(sorted(getattr(res, "losses", ()) or ())),
               int(getattr(res, "cases_fired", 0) or 0))
        hit = net_override.get(key)
        if hit is None:
            return float(res.net), "single draw"
        return float(hit["mean_net"]), f"mean of {hit['n']} draw comparison(s), arm {hit['label']}"

    def improves(res: Any) -> bool:
        if res is None:
            return False
        net, src = _net_of(res)
        label = getattr(res, "arm_label", "?")
        if net <= 0:
            return False
        if min_fired and int(getattr(res, "cases_fired", 0) or 0) < min_fired:
            print(f"  [improves] {label}: net {net:+.2f} ({src}) but fired "
                  f"{getattr(res, 'cases_fired', 0)} < {min_fired} -- too few cases to carry a verdict")
            return False
        if min_net and net < min_net:
            print(f"  [improves] {label}: net {net:+.2f} ({src}) < required "
                  f"{min_net} -- inside the declared noise floor")
            return False
        if src != "single draw":
            print(f"  [improves] {label}: net {net:+.2f} ({src}) >= {min_net} -- IMPROVES")
        return True

    outcome = optimize_residual(
        residual, runtime=ADAPTER, host=ADAPTER.HOST, events=records, states=records,
        evaluate=evaluation, improves=improves,
    )

    print(f"residual: {len(residual.case_ids)} failing task(s), {len(records)} mined record(s)")
    print(f"incumbent_id={incumbent_id!r}")
    print(f"state: {outcome.state}   next_coordinate: {next_coordinate(outcome.state)}")
    print(f"boundaries: {outcome.boundaries}   visited: {outcome.visited}")
    print(f"candidates this step: {len(outcome.candidates)}")
    print(
        f"evaluation: requested={len(evaluation.requested)} served={len(evaluation.served)} "
        f"missing={len(evaluation.missing)} incomparable={len(evaluation.incomparable)} "
        f"budget_exhausted={evaluation.budget_exhausted}"
    )
    for att in outcome.attempts:
        print(
            f"  attempt @ {att.boundary:24s} state={att.state:24s} "
            f"built={att.candidates_built} evaluated={att.candidates_evaluated} "
            f"-> {att.coordinate_changed}"
        )

    if outcome.state == "IMPROVED" and outcome.promoted is not None:
        arm = outcome.promoted
        print(f"\nPROMOTED: {arm.label}")
        print(f"  boundary={arm.boundary.value} action={arm.action.value} signal={arm.signal}")
        print(f"  eta={dict(arm.eta)}")
        # The promotion path is where this warning matters most: a promotion is the one outcome that
        # HALTS the round, so an arm promoted on too few firings both mislabels itself and stops the
        # search before anything else is tried. Advisory only -- the promotion above already happened
        # and is not altered here.
        _report_headroom([arm], args.evaluation, experiment_name, out_dir,
                         noise_floor=args.noise_floor, enabled=not args.no_headroom)
        print(
            "\nA promotion is a NEW incumbent. Re-mining the residual under it (and proposing a "
            "further round against it) requires actually deploying this controller and running a "
            "fresh AppWorld dev evaluation -- that is a separate, later round, not automated here."
        )
        return 0

    if outcome.candidates:
        manifest_path, controllers_path = write_manifest(
            outcome.candidates, out_dir, incumbent_id=incumbent_id, incumbent_token=incumbent_token
        )
        print(f"\nwrote manifest: {manifest_path}")
        print(f"wrote controllers: {controllers_path}")
        _report_headroom(outcome.candidates, args.evaluation, experiment_name, out_dir,
                         noise_floor=args.noise_floor, enabled=not args.no_headroom)
        print(
            f"\nnext: measure these against the frozen incumbent (`run_round.py score "
            f"--controllers {controllers_path} --evaluation {args.evaluation}`), write results to "
            f"{results_path}, then run propose again."
        )
        return 0

    print(
        f"\nno new candidates this step. state={outcome.state} is not IMPROVED and produced "
        f"nothing to measure -- report this state rather than treating it as an error."
    )
    return 0


# ================================================================================================
# score -- REAL AppWorld episodes against a REAL granite-4.1-30b server. Cost-bearing.
# ================================================================================================

def _load_agent_config(experiment_name: str) -> tuple[dict[str, Any], str]:
    """The SAME rendered config the real harness would build for `experiment_name`, via the SAME
    `jsonnet_load`/`path_store` path `appworld.cli`'s own `run` command uses -- so a controller-
    modified run and the frozen incumbent's own run are, apart from the controller, byte-identical
    in every model/decoding/prompt setting. Returns `(agent_config, dataset_name)`.
    """
    config_path = path_store.experiment_config_file_path(experiment_name)
    rendered = jsonnet_load(
        config_path,
        APPWORLD_EXPERIMENT_PROMPTS_PATH=path_store.experiment_prompts,
        APPWORLD_EXPERIMENT_CONFIGS_PATH=path_store.experiment_configs,
        APPWORLD_EXPERIMENT_CODE_PATH=path_store.experiment_code,
    )
    runner_config = rendered["config"]
    agent_config = dict(runner_config["agent"])
    agent_config.pop("type", None)  # a registry-dispatch key, not a constructor kwarg
    return agent_config, runner_config["dataset"]


def _fired_counts_from_markers(experiment_name: str, task_ids: list[str]) -> dict[str, int]:
    """Per-task firing counts read from the `anchoropt_fired` markers `AppWorldReActAgent` writes.

    Read INSTEAD OF the live agent's own counters whenever the episodes were not all run by this
    process -- a resumed arm (finished episodes are skipped and never re-fire here) or a parallel arm
    (each worker's counters die with the worker). The markers are the only record that spans both.

    A marker whose body is empty or unparseable counts as 1: markers written before the count was
    recorded held an empty string, and "it fired at least once" is all such a file ever asserted.
    """
    root = path_store.experiment_outputs
    counts: dict[str, int] = {}
    for task_id in task_ids:
        path = os.path.join(root, experiment_name, "tasks", task_id, "misc", "anchoropt_fired")
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                counts[task_id] = max(1, int(f.read().strip()))
        except (OSError, ValueError):
            counts[task_id] = 1
    return counts


def _fired_from_markers(experiment_name: str, task_ids: list[str]) -> set[str]:
    """Task ids whose episode left an `anchoropt_fired` marker (written by `AppWorldReActAgent`).

    Read alongside the live agent's own set so a resumed run, whose finished episodes are skipped and
    therefore never re-fire in this process, still attributes their outcomes correctly.
    """
    return set(_fired_counts_from_markers(experiment_name, task_ids))


def _solve_chunk(
    spec: dict[str, Any], experiment_name: str, arm_experiment_name: str,
    all_task_ids: list[str], num_processes: int, process_index: int,
) -> None:
    """One worker's share of an arm's episodes. Module-level so it survives being handed to a child.

    `solve_tasks` does the chunking itself, from `(num_processes, process_index)` -- so every worker is
    given the WHOLE task list and takes its own slice. Handing a worker a pre-sliced list as well would
    slice twice and silently score a fraction of the corpus.

    Each worker builds its own agent rather than inheriting one: the controller is stateless per
    episode, but an agent carries per-episode world state, and two processes sharing one would
    interleave it. Firing telemetry is not returned -- it goes to the on-disk markers, which is the
    only channel a parent can read back (see `_fired_counts_from_markers`).
    """
    agent_config, _ = _load_agent_config(experiment_name)
    agent = AppWorldReActAgent(controller=controller_from_spec(spec), **agent_config)
    agent.solve_tasks(
        task_ids=all_task_ids,
        experiment_name=arm_experiment_name,
        num_processes=num_processes,
        process_index=process_index,
    )


def _solve_in_parallel(
    spec: dict[str, Any], *, experiment_name: str, arm_experiment_name: str,
    all_task_ids: list[str], num_processes: int,
) -> None:
    """Runs an arm's episodes across `num_processes` worker processes, and fails if any worker does.

    WHY PROCESSES, AND WHY THIS IS NOT A THROUGHPUT KNOB
    ----------------------------------------------------------------------------------------------
    `SimplifiedReActCodeAgent.solve_tasks` is single-threaded: its `num_processes` argument only
    selects which chunk to run, so calling it with `num_processes=16` in ONE process scores 1/16 of
    the corpus and reports the rest as absent. Real concurrency needs real processes, which is how
    AppWorld's own `run` CLI does it -- and that CLI is how every baseline here was measured.

    That is the reason this exists. A baseline measured at `--num-processes 16` against a vLLM server
    we are the sole tenant of, compared with an arm scored one episode at a time, differs in
    server-side batch shapes as well as in the controller -- so it is not the paired comparison the
    round claims to be. Matching the baseline's concurrency is what makes the comparison honest; the
    wall-clock saving is a side effect.

    `fork` is explicit rather than inherited from the platform default, which is changing: the whole
    point is that a child inherits the already-rendered config and the APPWORLD_ROOT-derived
    `path_store`, and no AppWorld world or database is open yet at this point.
    """
    ctx = multiprocessing.get_context("fork")
    workers = [
        ctx.Process(
            target=_solve_chunk,
            args=(spec, experiment_name, arm_experiment_name, all_task_ids, num_processes, i),
            name=f"arm-worker-{i}",
        )
        for i in range(num_processes)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    failed = [(w.name, w.exitcode) for w in workers if w.exitcode != 0]
    if failed:
        # Never fall through to evaluation: a dead worker's chunk has no episodes, and the arm would
        # be scored as though those tasks had simply failed, turning a crash into a measurement.
        raise RuntimeError(
            f"{len(failed)} of {num_processes} arm workers failed: "
            + ", ".join(f"{name} exit={code}" for name, code in failed)
        )


def _score_one_arm(
    spec: dict[str, Any], *, experiment_name: str, task_ids: list[str],
    regression_task_ids: list[str] = (), incumbent_token: str, incumbent_dataset: str,
    num_processes: int = 1,
) -> ArmResult:
    """Scores one arm against `task_ids` (the residual, or a subset of it) plus, when
    `regression_task_ids` is non-empty, every task named there too -- the SAME controller, applied
    uniformly; a task already passing under the incumbent is where a `loss` can actually be observed.
    Omitting `regression_task_ids` (the default) preserves the original behaviour exactly: `losses`
    comes back empty, because nothing outside the residual was run.
    """
    agent_config, dataset_name = _load_agent_config(experiment_name)
    if dataset_name != incumbent_dataset:
        raise ValueError(
            f"{experiment_name!r}'s config targets dataset {dataset_name!r}, but the frozen "
            f"incumbent this round measures against was evaluated on {incumbent_dataset!r} -- "
            f"scoring an arm against a different dataset is not a paired comparison."
        )

    arm_experiment_name = f"{experiment_name}/arms/{spec['arm_label']}"
    all_task_ids = list(task_ids) + list(regression_task_ids)

    # `agent` stays None in the parallel path on purpose: there is no parent-process agent whose
    # counters mean anything, and reading them would report only what the parent itself ran -- zero.
    agent: AppWorldReActAgent | None = None
    if num_processes > 1:
        _solve_in_parallel(
            spec, experiment_name=experiment_name, arm_experiment_name=arm_experiment_name,
            all_task_ids=all_task_ids, num_processes=min(num_processes, len(all_task_ids)),
        )
    else:
        agent = AppWorldReActAgent(controller=controller_from_spec(spec), **agent_config)
        agent.solve_tasks(task_ids=all_task_ids, experiment_name=arm_experiment_name)

    metrics = evaluate_tasks(task_ids=all_task_ids, experiment_name=arm_experiment_name)
    individual = metrics.get("individual") or {}
    succeeded = lambda tid: bool((individual.get(tid) or {}).get("success"))  # noqa: E731

    # ATTRIBUTION: only an episode the mechanism actually FIRED on can have been changed by it.
    #
    # This used to count every outcome flip in scope. Measured consequence on a minimax held-in arm:
    # it reported 4 gains and 5 losses, net -1 -- but 4 of those 5 losses were episodes where the
    # controller never fired at all, so its attributable result was 4 gains against 1 loss, net +3.
    # The incumbent's own two repeats of this split differ by 2-5 tasks at temperature 0, which is
    # where those phantom flips come from. Because core's `improves` is `net > 0`, feeding it the
    # unfiltered number turned a working arm into a NO_BENEFIT and spent a boundary expansion on an
    # artifact. Firing is a NECESSARY condition for causation, not a sufficient one -- this makes the
    # number honest about what the arm cannot have caused, and claims nothing more.
    # Union of this process's firings and the on-disk markers, so episodes that `skip_if_finished`
    # skipped on a resumed run still count as fired. Without the markers a resubmit -- the normal path
    # after a wall-clock kill -- would report an empty `fired` set and discard every real gain.
    # The markers are authoritative for BOTH the fired set and the firing count, because they are the
    # only record that covers episodes this process did not run -- ones `skip_if_finished` skipped on a
    # resume, and ones a worker process ran. The live agent's set is unioned in (serial path only) so a
    # marker write that failed with OSError cannot erase a firing this process witnessed.
    marker_counts = _fired_counts_from_markers(arm_experiment_name, all_task_ids)
    fired = set(marker_counts)
    if agent is not None:
        fired |= set(agent.fired_tasks)
    executed = sum(marker_counts.values())
    if agent is not None:
        executed = max(executed, agent.executed)
    raw_gains = tuple(tid for tid in task_ids if succeeded(tid))
    raw_losses = tuple(tid for tid in regression_task_ids if not succeeded(tid))
    gains = tuple(tid for tid in raw_gains if tid in fired)
    losses = tuple(tid for tid in raw_losses if tid in fired)
    unattributed_gains = tuple(tid for tid in raw_gains if tid not in fired)
    unattributed_losses = tuple(tid for tid in raw_losses if tid not in fired)
    denominator_ok = set(individual) == set(all_task_ids)

    # `n` stays the whole scored corpus on purpose: the arm's effect is a rate over everything it was
    # applied to, not over the subset it happened to touch. Narrowing the denominator to `fired` would
    # inflate every delta.
    detail = (
        f"unfiltered gains={len(raw_gains)} losses={len(raw_losses)} "
        f"(net {len(raw_gains) - len(raw_losses):+d}); "
        f"non-attributable flips discarded -- gains {list(unattributed_gains)}, "
        f"losses {list(unattributed_losses)}"
    )

    return ArmResult(
        arm_label=spec["arm_label"],
        gains=gains,
        losses=losses,
        n=len(all_task_ids),
        cases_fired=len(fired),
        interventions_executed=executed,
        accuracy_delta_pp=100.0 * (len(gains) - len(losses)) / len(all_task_ids) if all_task_ids else 0.0,
        theta={},
        incumbent_token=incumbent_token,
        denominator_ok=denominator_ok,
        detail=detail,
    )


def _report_headroom(candidates: Any, evaluation_path: str, experiment_name: str, out_dir: str,
                     *, noise_floor: int, enabled: bool = True) -> None:
    """ADVISORY ONLY. Reports each proposed arm's ceiling; changes nothing about the search.

    WHY THIS IS REPORTING AND NOT A GATE
    ----------------------------------------------------------------------------------------------
    `optimize_residual` has already run and its candidates are already written by the time this is
    called. Nothing here filters a candidate, alters a search state, or feeds back into core -- so the
    algorithm's behaviour is bit-identical whether this runs or not, and `--no-headroom` skips it
    entirely. Deciding what to afford is the caller's job; this only makes the cost/ceiling visible
    before the caller spends real inference on it.

    WHAT IT ANSWERS
    ----------------------------------------------------------------------------------------------
    "How many episodes could this arm change at most?" — measured on the incumbent's OWN logs. Two
    ceilings are reported and the verdict is taken against the LOOSER of them, because each is valid
    for a different shape of signal:

      local  firings minus the ones the incumbent already recovers from unaided. Valid only when the
             signal names something that errors.
      task   firings on episodes the incumbent FAILS. Valid for any signal, including a completeness
             signal whose cell executes successfully and fails the task silently.

    Measured motivation for the screen: the `post_execution/execution_failed/reprompt` arm was scored on
    four incumbents before anyone noticed the agent already self-corrects after 73-91% of first errors,
    leaving ceilings of 4 to 17 episodes against a 2-5 task noise floor. minimax held-in's local ceiling
    was **4** — unresolvable before a single episode ran.

    Measured motivation for using the max: `pagination_unbounded` screens at local **1** because 37 of
    its 38 firing cells execute fine, and at task **15**. It then measured +4 and +1 across two passes
    with 5 of 6 gain tasks identical. Judged on `local` it would have been discarded. `headroom.py` has
    the full derivation and caveats.

    Failures here are swallowed deliberately: an advisory screen must never be able to break a round.
    """
    if not enabled:
        return
    try:
        from headroom import HeadroomUnavailable, screen
    except ImportError as exc:                                  # pragma: no cover - defensive
        print(f"\n[headroom] unavailable ({exc}); skipping the advisory screen")
        return

    signals = []
    for arm in candidates:
        sig = str(getattr(arm, "signal", "") or "")
        if sig and sig not in signals:
            signals.append(sig)
    if not signals:
        return

    print("\nheadroom (ADVISORY -- computed from the incumbent's logs; does not affect the search):")
    report: dict[str, Any] = {"incumbent_evaluation": os.path.abspath(evaluation_path),
                              "noise_floor": noise_floor, "signals": {}}
    for sig in signals:
        try:
            rep = screen(evaluation_path, experiment_name, sig, first_firing_only=True)
        except HeadroomUnavailable as exc:
            print(f"  {sig:<24} not screenable: {exc}")
            report["signals"][sig] = {"available": False, "reason": str(exc)}
            continue
        except Exception as exc:                                 # pragma: no cover - defensive
            print(f"  {sig:<24} screen failed ({type(exc).__name__}: {exc}); continuing")
            report["signals"][sig] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
            continue

        fired, res, head = rep["firing_episodes"], rep["self_resolved"], rep["headroom"]
        # THE CEILING IS THE LOOSER OF THE TWO, AND GETTING THIS WRONG COST US THE ONE ARM THAT WORKED
        # ------------------------------------------------------------------------------------------
        # `headroom` (local) = firings - locally self-resolved. It is only meaningful when the signal
        # names something that ERRORS; a completeness signal executes fine and fails the task silently,
        # so local reads ~0 for it. `headroom_task` = firings on episodes the incumbent FAILS, which
        # bounds gains for any signal shape. `headroom.py` derives both; the verdict must therefore be
        # taken against `max(...)`, or a completeness signal is condemned by the ceiling that does not
        # apply to it.
        #
        # This line previously used `head` alone, and did exactly that: `pagination_unbounded` screened
        # at local **1**, task **15**, and was printed UNRESOLVABLE -- then measured +4 and +1 over two
        # passes with 5 of 6 gain tasks IDENTICAL across them, the only replicating gain in the record.
        # The screen stayed advisory, so nothing was lost; had it been a gate it would have killed the
        # one mechanism that worked. Kept as a comment because the failure mode is not the arithmetic,
        # it is applying an error-shaped bound to a signal that does not error.
        ceiling = max(head, int(rep.get("headroom_task") or 0))
        pct = f"{100.0 * res / fired:.0f}%" if fired else "n/a"
        verdict = "UNRESOLVABLE" if fired and ceiling <= noise_floor else "resolvable"
        print(f"  {sig:<24} fires on {fired}/{rep['tasks']} episodes, incumbent self-resolves "
              f"{res} ({pct}) -> ceiling {ceiling} (local {head}, task {rep.get('headroom_task')}) "
              f"vs noise floor {noise_floor}: {verdict}")
        if fired and ceiling <= noise_floor:
            print(f"  {'':<24} ^ BOTH ceilings are at or below the floor, so no mechanism at this "
                  f"boundary can produce an effect this corpus distinguishes from sampling; refining "
                  f"the signal only shrinks the firing set")
        elif fired and head <= noise_floor:
            # Report the fact, NOT a cause. Two different situations produce it and this screen cannot
            # tell them apart: a COMPLETENESS signal, whose cells execute successfully so local recovery
            # is undefined for it; or an error-shaped signal the incumbent simply repairs most of the
            # time (`execution_failed` self-resolves 91%, giving local 4 against task 14). An earlier
            # version of this line asserted the first, and mislabelled the second.
            print(f"  {'':<24} ^ local ceiling {head} is at/below the floor but task ceiling "
                  f"{rep.get('headroom_task')} is not, so this is judged on the task ceiling. Either "
                  f"the signal's cells do not error (local recovery undefined for it) or the incumbent "
                  f"already repairs most of what it names -- this screen cannot distinguish those.")
        rep["available"] = True
        rep["ceiling"] = ceiling          # recorded so the verdict in the JSON is reproducible from it
        rep["verdict"] = verdict
        report["signals"][sig] = rep

    # The promotion path reaches here without `write_manifest` having created out_dir.
    path = os.path.join(out_dir, "headroom.json")
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
        print(f"wrote headroom: {path}")
    except OSError as exc:                                      # pragma: no cover - defensive
        print(f"[headroom] could not write {path}: {exc}")


def cmd_score(args: argparse.Namespace) -> int:
    print(
        "SCORE MODE: this issues REAL LLM calls -- whichever model the incumbent's own experiment "
        "config names, reached through that config's base_url -- for every task_id in scope, once "
        "per arm. It is cost/quota-bearing. Confirm scope before running.",
        file=sys.stderr,
    )

    with open(args.controllers, encoding="utf-8") as f:
        payload = json.load(f)
    specs = payload["controllers"]
    if args.arm:
        specs = [s for s in specs if s["arm_label"] == args.arm]
        if not specs:
            print(f"no arm labeled {args.arm!r} in {args.controllers}", file=sys.stderr)
            return 1

    experiment_name = args.experiment or _infer_experiment_name(args.evaluation)
    # AppWorld names an evaluation file after the split it evaluated
    # (`<experiment_name>/evaluations/<split>.json`), so this is the incumbent's own dataset --
    # what an arm's config must agree with for the comparison to be paired.
    incumbent_dataset = os.path.splitext(os.path.basename(args.evaluation))[0]
    # `failing_task_ids`, not `mine_residual`: scoring needs only WHICH tasks are in the residual,
    # never their mined records. Mining here would re-parse every failing task's environment_io.md
    # for records this function discards -- and those logs live under the root that produced the
    # evaluation, which is not the root an arm must RUN under when the two differ (a frozen
    # round cell holds the incumbent's logs but no experiments/configs). Reading the evaluation JSON
    # alone keeps `score` independent of where the incumbent's transcripts happen to sit.
    task_ids = failing_task_ids(args.evaluation)
    if args.limit:
        task_ids = task_ids[: args.limit]

    regression_task_ids: list[str] = []
    if args.check_regressions:
        regression_task_ids = passing_task_ids(args.evaluation)

    incumbent_token = payload.get("incumbent_token", "")
    out_path = args.out or os.path.join(os.path.dirname(args.controllers), "arm_results.json")

    # Arms measured by an EARLIER invocation against this same incumbent. Loading them means this
    # command merges into the file rather than replacing it, and re-running after an interruption
    # does not pay again for arms that already have an answer.
    measured = load_results(out_path, incumbent_token=incumbent_token)
    if measured:
        print(f"{out_path} already holds {len(measured)} measured arm(s); merging into it")

    scored_now = 0
    for spec in specs:
        label = spec["arm_label"]
        if label in measured and not args.rescore:
            print(f"skipping (already measured; --rescore to redo): {label}")
            continue
        suffix = f" + {len(regression_task_ids)} regression task(s)" if regression_task_ids else ""
        print(f"scoring arm: {label}  ({len(task_ids)} residual task(s){suffix})")
        measured[label] = _score_one_arm(
            spec, experiment_name=experiment_name, task_ids=task_ids,
            regression_task_ids=regression_task_ids, incumbent_token=incumbent_token,
            incumbent_dataset=incumbent_dataset, num_processes=args.num_processes,
        )
        scored_now += 1
        # Persisted after EVERY arm, not once at the end. An arm costs real inference over every task
        # in scope, so a later arm failing -- or the whole job being killed at its wall-clock limit --
        # must not discard the arms that already finished. This does NOT rescue an arm interrupted
        # part-way through its own task list: that arm has no result yet by definition. What protects
        # THAT work is AppWorld's own per-episode `skip_if_finished` marker, which lets a resubmitted
        # run skip episodes already completed and re-run only the partial one.
        write_results(list(measured.values()), out_path, incumbent_token=incumbent_token)
        print(f"  recorded -> {out_path}  ({len(measured)} arm(s) in file)")

    if scored_now == 0:
        print(f"no arms scored this invocation; {out_path} left as it was ({len(measured)} arm(s))")
    else:
        print(f"scored {scored_now} arm(s) this invocation; {out_path} holds {len(measured)}")
    return 0


# ================================================================================================
# CLI
# ================================================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("propose", help="advance the search one step; write a manifest if there is one")
    p.add_argument("--evaluation", required=True, help="path to the frozen incumbent's evaluations/*.json")
    p.add_argument("--experiment", default=None, help="override the inferred experiment_name")
    p.add_argument("--out", default=None, help="directory for arm_manifest.json/controllers.json")
    p.add_argument("--results", default=None, help="path to already-measured ArmResult JSON, if any")
    p.add_argument("--budget", type=int, default=None, help="cap on evaluations this evaluator will serve")
    p.add_argument(
        "--noise-floor", type=int, default=5,
        help="task-count noise floor for this corpus, used ONLY to label the advisory headroom report "
             "(default 5: the spread measured between same-config baseline repeats on 90 tasks)",
    )
    p.add_argument(
        "--min-net", type=int, default=0,
        help="an arm must reach at least this net to count as an improvement. 0 (default) keeps the "
             "historical `net > 0` rule, so prior rounds stay reproducible. Set it to the corpus's "
             "noise floor -- 5 for 90 AppWorld tasks -- to stop promoting inside sampling error",
    )
    p.add_argument(
        "--min-firings", type=int, default=0,
        help="an arm must have fired on at least this many cases to count as an improvement, no matter "
             "its net. 0 (default) is off. Guards the failure this benchmark actually hit: an arm that "
             "fired 6/90 promoted at +2, then measured -1 on a repeat and 0 on held-out",
    )
    p.add_argument(
        "--net-from-draws", default=None, metavar="MODEL",
        help="judge each arm on the MEAN net over every (arm pass x incumbent draw) comparison instead "
             "of its net against the single draw it was scored against. MODEL is a key of "
             "runs/multi_draw_nets.DRAWS (minimax, qwen). OFF by default: it is a decision rule, it "
             "changes verdicts in both directions, and a promotion that depends on it must say so",
    )
    p.add_argument(
        "--no-headroom", action="store_true",
        help="skip the advisory headroom screen. It replays the incumbent's logs for every task, which "
             "is free but not instant; the search is unaffected either way",
    )
    p.set_defaults(func=cmd_propose)

    s = sub.add_parser("score", help="REAL AppWorld episodes against a REAL model server -- cost-bearing")
    s.add_argument("--controllers", required=True, help="controllers.json written by `propose`")
    s.add_argument("--evaluation", required=True, help="path to the frozen incumbent's evaluations/*.json")
    s.add_argument("--experiment", default=None, help="override the inferred experiment_name")
    s.add_argument("--arm", default=None, help="score only this arm_label")
    s.add_argument("--limit", type=int, default=None, help="cap the number of task_ids scored, for a smoke run")
    s.add_argument(
        "--num-processes", type=int, default=1,
        help="worker processes to spread this arm's episodes over. MATCH THE CONCURRENCY THE FROZEN "
             "INCUMBENT WAS MEASURED AT (baselines here were run via `appworld run --num-processes`): "
             "against a server we are the sole tenant of, a different concurrency means different "
             "server-side batch shapes, so the arm and the incumbent stop being a paired comparison. "
             "Default 1 preserves the original single-process behaviour.",
    )
    s.add_argument(
        "--check-regressions", action="store_true",
        help="also score every task the frozen incumbent already passes, so `losses` reflects a "
             "real full-corpus regression check instead of being unconditionally empty -- multiplies "
             "the cost of this command by (1 + passing_tasks/residual_tasks)",
    )
    s.add_argument("--out", default=None, help="path to write ArmResult JSON (default: alongside --controllers)")
    s.add_argument(
        "--rescore", action="store_true",
        help="re-measure arms that already have a result in --out instead of skipping them; only "
             "needed when an arm's recorded result is known stale, since re-running otherwise pays "
             "full inference cost for an answer already on disk",
    )
    s.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
