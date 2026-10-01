"""Mines the frozen incumbent's residual into `events`/`states`, offline and model-free.

WHERE THIS SITS IN THE PIPELINE
--------------------------------
`run_round.py` needs a `(residual, events, states)` triple to hand to
`anchoropt.learning.structured_search.optimize_residual`. Every fact that triple is built from
already exists on disk from the granite-4.1-30b `dev` run: the per-task correctness vector in
`evaluations/dev.json` (`individual[task_id]["success"]`), and each task's own cell-by-cell transcript
in `tasks/<task_id>/logs/environment_io.md`, readable back via `AppWorld.parse_environment_io_log`
(`<appworld-checkout>/src/appworld/environment.py:1246-1277`) -- a plain file read, no `AppWorld()`
instance, no world, no language model. So mining the residual costs zero LLM calls and needs no new
episode; it replays a run AppWorld already ran.

`_execute_preamble` (`environment.py:880-903`) calls `self.shell.run_cell` directly and never reaches
`execute()`/`_save_environment_io_log`, so `environment_io.md` contains only agent-issued cells, one
per step -- there is no task-setup contamination to filter out here.

ONE RECORD SERVES AS BOTH AN EVENT AND A STATE -- A DELIBERATE DEVIATION FROM PORTING.MD'S GENERIC
EXAMPLE, WORTH FLAGGING
--------------------------------------------------------------------------------------------------
`docs/PORTING.md` §8's canonical shape builds `events` and `states` as two separate lists
(`events = [normalize_event(r) ...]`, `states = [observable_state(ev) ...]`), because in general a
benchmark's "event" and its "observable state" are different projections of the same raw row. Here
they collapse to the same dict: core's only read of an event (`boundaries_of`,
`anchoropt/learning/boundary_search.py:63-79`) is `event.get("kind")`, and `AppWorldHost.states_at`
(`adapter.py:357-367`) only reads `st.get("boundary")` -- both keys, set to the same `IncisionPoint`
value, sit on one dict without conflict, and `states_at`'s own field whitelist drops everything else
(including the redundant `"kind"` key) when it projects. Building two parallel lists here would be
duplication with no behavioural difference, so this file emits one list and passes it for both
`events=` and `states=`. Flagged here because it diverges from the doc's own example, not because it
is a shortcut that skips a real requirement.

WHY PRE_GENERATION RECORDS ARE MINED TOO, NOT JUST POST_GENERATION_PRE_EXEC/POST_EXECUTION
--------------------------------------------------------------------------------------------
An earlier draft of this plan mined only two records per cell (the gate and the post-execution
result), leaving PRE_GENERATION unrepresented in `states`. That is a live crash risk, not a harmless
omission: `AppWorldHost.synthesis_fields(PRE_GENERATION)` is non-empty (`step_number`,
`consecutive_errors`, `last_error_kind`), so `_expand_phi_at` (`structured_search.py:196-238`) always
calls `states_at(PRE_GENERATION, states)` when Phi expansion reaches that boundary, and raises
`RuntimeError` -- uncaught -- if the projection comes back empty ("states_at(...) produced no states
... this is a contract gap, not an exhausted search"). PRE_GENERATION is a real, declared boundary
with a real executor (`agent_hooks.AppWorldReActAgent._inject_pre_generation_instruction`); giving it
no states because its fields do not need `gate_fields_from_code`/`exec_fields_from_output` would be
under-describing the host to avoid a few lines of bookkeeping. So all three boundaries are mined per
step, replaying `consecutive_errors`/`last_error_kind` forward exactly as
`agent_hooks.AppWorldReActAgent` accumulates them live.

STEP NUMBERING
--------------
`AppWorldReActAgent` (`agent_hooks.py`) always returns exactly one `ExecutionIO` per step, so
`Agent.solve_task`'s `world.batch_execute([...])` is always called with a single-element list; AppWorld
only sub-numbers an interaction (`"3.1"`, `"3.2"`) when one `batch_execute` call carries more than one
code string (`environment.py:1100-1141`). For this agent's logs, cell number and step number coincide,
so `enumerate(cells, start=1)` reconstructs `self.step_number` exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from appworld import AppWorld
from appworld.common.path_store import path_store

from anchoropt.anchor import IncisionPoint

from adapter import exec_fields_from_output, gate_fields_from_code

_PRE = IncisionPoint.PRE_GENERATION.value
_PG = IncisionPoint.POST_GENERATION_PRE_EXEC.value
_PE = IncisionPoint.POST_EXECUTION.value


@dataclass(frozen=True)
class AppWorldResidual:
    """The residual, as core sees it. `case_ids` is the only field `optimize_residual` reads off it
    (`getattr(residual, "case_ids", ())`, `structured_search.py:384`) -- provenance for
    `SearchSpaceProposal.diagnosis_case_ids`, never a constraint core evaluates."""

    case_ids: tuple[str, ...]


def _infer_experiment_name(evaluation_path: str) -> str:
    """`<experiment_outputs>/<experiment_name>/evaluations/<file>.json` -> `<experiment_name>`.

    Avoids a second, redundant CLI argument that could drift from the evaluation file's own location;
    `--experiment` remains available to override this for a differently-laid-out evaluation file.
    """
    root = os.path.abspath(path_store.experiment_outputs)
    path = os.path.abspath(evaluation_path)
    if not (path == root or path.startswith(root + os.sep)):
        raise ValueError(f"{evaluation_path!r} is not under experiment_outputs ({root})")
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if len(parts) < 3 or parts[-2] != "evaluations":
        raise ValueError(
            f"{evaluation_path!r} is not of the form "
            f"<experiment_name>/evaluations/<file>.json under {root}"
        )
    return "/".join(parts[:-2])


def failing_task_ids(evaluation_path: str) -> list[str]:
    """The frozen incumbent's residual, as a set of task_ids: `individual[task_id]["success"]`
    is False. Purely caller-side filtering, per `docs/PORTING.md` §8's own convention -- core never
    sees a success/failure flag, so labeling happens here, before any event/state is built."""
    with open(evaluation_path, encoding="utf-8") as f:
        payload = json.load(f)
    individual = payload.get("individual") or {}
    return sorted(task_id for task_id, record in individual.items() if not record.get("success"))


def passing_task_ids(evaluation_path: str) -> list[str]:
    """Complement of `failing_task_ids`: task_ids the frozen incumbent already solves. A `loss` can
    only be observed here -- a task already failing has nothing left to lose -- so this is the set
    `run_round.py --check-regressions` scores an arm against, in addition to the residual."""
    with open(evaluation_path, encoding="utf-8") as f:
        payload = json.load(f)
    individual = payload.get("individual") or {}
    return sorted(task_id for task_id, record in individual.items() if record.get("success"))


def mine_task(experiment_name: str, task_id: str) -> list[dict[str, Any]]:
    """One task's ordered PRE/PG/PE records, replayed from its already-written `environment_io.md`.

    Raises `FileNotFoundError` if AppWorld never wrote a log for this task_id (propagated, not
    swallowed here, so `mine_residual` can decide whether a missing log is worth reporting).
    """
    cells = AppWorld.parse_environment_io_log(experiment_name=experiment_name, task_id=task_id)
    cells = sorted(cells, key=lambda c: float(c["number"]))

    consecutive_errors = 0
    last_error_kind = ""
    records: list[dict[str, Any]] = []
    for step_number, cell in enumerate(cells, start=1):
        records.append({
            "kind": _PRE, "boundary": _PRE, "task_id": task_id, "case_id": task_id,
            "step_number": step_number,
            "consecutive_errors": consecutive_errors,
            "last_error_kind": last_error_kind,
        })

        # `step_number` comes from the same enumerate() that stamps the _PRE record, so the gate
        # state and the pre-generation state agree on which turn this is.
        pg_fields = gate_fields_from_code(cell["input"], step_number)
        records.append(
            {"kind": _PG, "boundary": _PG, "task_id": task_id, "case_id": task_id, **pg_fields}
        )

        pe_fields = exec_fields_from_output(cell["output"])
        records.append(
            {"kind": _PE, "boundary": _PE, "task_id": task_id, "case_id": task_id, **pe_fields}
        )

        last_error_kind = pe_fields["error_kind"]
        consecutive_errors = 0 if pe_fields["succeeded"] else consecutive_errors + 1

    return records


def mine_residual(
    evaluation_path: str, experiment_name: str | None = None
) -> tuple[AppWorldResidual, list[dict[str, Any]]]:
    """The frozen incumbent's residual, and every PRE/PG/PE record replayed from the failing tasks'
    own logs. The returned list is passed as BOTH `events=` and `states=` to `optimize_residual` --
    see the module docstring for why that is not a shortcut.

    A missing log for a failing task_id is reported to stderr and that task is skipped, not raised
    through -- one task's incomplete log should not block mining the other 40.
    """
    experiment_name = experiment_name or _infer_experiment_name(evaluation_path)
    task_ids = failing_task_ids(evaluation_path)

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for task_id in task_ids:
        try:
            records.extend(mine_task(experiment_name, task_id))
        except FileNotFoundError:
            missing.append(task_id)

    if missing:
        print(
            f"residual.py: no environment_io.md log for {len(missing)}/{len(task_ids)} "
            f"failing task(s), skipped: {missing}",
            file=sys.stderr,
        )

    return AppWorldResidual(case_ids=tuple(task_ids)), records


def _summarize(task_ids: list[str], records: list[dict[str, Any]]) -> dict[str, Any]:
    by_boundary: dict[str, int] = {}
    error_kinds: dict[str, int] = {}
    for rec in records:
        by_boundary[rec["boundary"]] = by_boundary.get(rec["boundary"], 0) + 1
        if rec["boundary"] == _PE:
            kind = rec["error_kind"] or "(succeeded)"
            error_kinds[kind] = error_kinds.get(kind, 0) + 1
    mined_task_ids = sorted({rec["task_id"] for rec in records})
    return {
        "failing_task_ids": len(task_ids),
        "mined_task_ids": len(mined_task_ids),
        "records_by_boundary": by_boundary,
        "post_execution_error_kinds": error_kinds,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluation", required=True, help="path to the frozen evaluations/*.json")
    parser.add_argument(
        "--experiment", default=None,
        help="experiment_name for parse_environment_io_log; inferred from --evaluation if omitted",
    )
    args = parser.parse_args(argv)

    residual, records = mine_residual(args.evaluation, args.experiment)
    print(json.dumps(_summarize(list(residual.case_ids), records), indent=2))


if __name__ == "__main__":
    main()
