"""The measurement seam: builds the pass-N manifest and loads pass-N results, using core's OWN
`external_evaluation` module rather than a hand-rolled disk cache.

WHY THIS FILE IS THIN, NOT THE "CACHED EVALUATOR" THE PLAN ORIGINALLY DESCRIBED
--------------------------------------------------------------------------------------------------
The plan (`tingly-tinkering-river.md`, "The measurement seam") describes a hand-rolled
`evaluate(arm, theta=None)` reading a disk cache of `ArmResult` JSON and returning `None` on a miss,
built specifically to avoid `evaluate=None` halting the search at the first realizable boundary.
`anchoropt.learning.external_evaluation.ExternalEvaluation` already IS that evaluator -- the same
`results: dict[str, ArmResult]` / None-on-miss contract, the same `from_json` loader, the same
`arm_manifest()` serializer for the work order an external process runs. Reimplementing it here would
be a second, adapter-specific copy of code core already ships in full; this file wraps it instead.

THE PLAN'S OWN MOTIVATING CLAIM FOR THE DISK CACHE IS FALSIFIED BY DIRECT MEASUREMENT -- A FINDING TO
REPORT, NOT A DEFECT TO WORK AROUND
--------------------------------------------------------------------------------------------------
The plan states that a cached evaluator returning `None` on a miss "does NOT return early" the way
`evaluate=None` does, and that "the grounded frontier is swept across all boundaries" on the cold
pass. Verified directly against `optimize_residual` (`anchoropt/learning/structured_search.py:499-
557`) using this adapter's own mined residual (41 failing tasks, 1413 records) and the real
`ADAPTER.HOST`: an `evaluate=None` call and an `evaluate=<a callable that always returns None>` call
produce BYTE-IDENTICAL `SearchOutcome`s -- same state (`REALIZABLE_UNMEASURED`), same single boundary
visited (`post_execution`, the latest), same one candidate arm, same empty `frontier_boundaries`.
Neither "sweeps all boundaries": the frontier-sweep gates at `:580`/`:642` only trigger on
`att.state in (NO_CANDIDATE, NO_BENEFIT)`, and a `None` evaluation result -- whether from
`evaluate=None` or from a real evaluator that simply has not measured this arm yet -- lands
`att.state = REALIZABLE_UNMEASURED` either way, which is NEITHER of those two states, so the function
returns immediately with only the current boundary's candidates in both cases.

So the real design is not "cold cache sweeps the whole frontier, warm cache detects no-benefit and
expands" -- it is genuinely incremental: one call to `optimize_residual` advances the search by
exactly one boundary (or one Phi-expansion) past whatever has already been measured, regardless of
which evaluator is passed. Multiple rounds are required either way; `ExternalEvaluation`'s actual job
is letting a LATER call recognize an arm it already knows the answer for, so the search can move past
`REALIZABLE_UNMEASURED` into `NO_BENEFIT` (and then legitimately sweep/expand) or `IMPROVED`. This
corrects the plan's own text about what the disk-cache evaluator buys; it does not change what
`ExternalEvaluation` itself is for, which is exactly this file's job below.

WHAT THIS FILE ACTUALLY ADDS ON TOP OF `external_evaluation.py`
--------------------------------------------------------------------------------------------------
Two AppWorld-specific facts core's generic module cannot know: (1) where the manifest and its
`controllers.json` companion (this adapter's own serialization for crossing the process boundary
described in `controller.py`) land on disk, and (2) what identifies "the frozen incumbent" for this
benchmark -- the granite-4.1-30b `dev` run's evaluation file and the experiment name
`residual.py` already derives from it. Everything else -- `ArmResult`, `arm_manifest`, the
None-on-miss contract, the incumbent-token mismatch check -- is used verbatim from core.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from anchoropt.learning.external_evaluation import ArmResult, ExternalEvaluation, arm_manifest

from controller import spec_for_arm


def incumbent_identity(evaluation_path: str, experiment_name: str) -> tuple[str, str]:
    """`(incumbent_id, incumbent_token)` for the frozen granite-4.1-30b `dev` run, mirroring
    `scripts/self_evolve_cycle2.py`'s own `(name, full-identity)` split -- a short label for display,
    and a stable, unambiguous token for the exact-match comparison `ExternalEvaluation.__call__`
    does against every recorded `ArmResult`."""
    return experiment_name, os.path.abspath(evaluation_path)


def write_manifest(
    arms: Any, out_dir: str, *, incumbent_id: str, incumbent_token: str = ""
) -> tuple[str, str]:
    """The pass's work order: `arm_manifest.json` (what core says must be measured, from
    `arm_manifest()` verbatim) and `controllers.json` (how to actually BUILD and RUN each arm --
    this adapter's own `spec_for_arm`, since core's manifest carries no executable predicate, only
    JSON-safe descriptive fields). Returns `(manifest_path, controllers_path)`.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest = arm_manifest(arms, incumbent_id=incumbent_id, incumbent_token=incumbent_token)
    manifest_path = os.path.join(out_dir, "arm_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    controllers_path = os.path.join(out_dir, "controllers.json")
    with open(controllers_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "incumbent_id": incumbent_id,
                "incumbent_token": incumbent_token,
                "controllers": [spec_for_arm(a) for a in arms],
            },
            f,
            indent=2,
        )
    return manifest_path, controllers_path


def load_evaluation(
    results_path: str | None,
    *,
    incumbent_token: str = "",
    budget: int | None = None,
    strict: bool = False,
) -> ExternalEvaluation:
    """The `evaluate=` callback for the NEXT `optimize_residual` call, whether or not measurement has
    happened yet. A missing/absent `results_path` returns a COLD evaluator (empty `results`), never
    `None` -- see the module docstring for why a cold `ExternalEvaluation` and `evaluate=None` are
    NOT the shortcut around the search halting early that the plan once took them for; the cold
    evaluator is still worth passing because a LATER call with the same object populated is how the
    search ever learns an arm's answer.

    `incumbent_token` is used only for the cold, no-file case. `from_json` deliberately reads the
    comparison token off the FILE's own `"incumbent_token"` field instead (`external_evaluation.py`
    :145) -- the round a results file was produced under, not whatever this call happens to pass --
    so it is never forwarded into `from_json`'s kwargs; doing so would collide with the value
    `from_json` already supplies positionally and (per its own comment at :169-179) is exactly the
    class of bug that check exists to prevent.
    """
    if results_path and os.path.exists(results_path):
        return ExternalEvaluation.from_json(results_path, budget=budget, strict=strict)
    return ExternalEvaluation(incumbent_token=incumbent_token, budget=budget, strict=strict)


def load_results(results_path: str, *, incumbent_token: str = "") -> dict[str, ArmResult]:
    """Arms already measured in `results_path`, keyed by `arm_label`, so a new measurement can be
    MERGED into that file instead of overwriting it. Returns `{}` when the file does not exist yet.

    The parse is `ExternalEvaluation.from_json` verbatim, so what this reads can never drift from what
    core will read back off the same file.

    A file recorded against a DIFFERENT incumbent raises rather than merging. Mixing arms measured
    against two different frozen incumbents into one results file yields numbers that are not
    comparable to each other -- precisely what `ExternalEvaluation`'s own incumbent-token check exists
    to catch -- so the caller is told to choose a different `--out` rather than being handed a
    silently-blended file.
    """
    if not (results_path and os.path.exists(results_path)):
        return {}
    loaded = ExternalEvaluation.from_json(results_path)
    if incumbent_token and loaded.incumbent_token and loaded.incumbent_token != incumbent_token:
        raise ValueError(
            f"{results_path!r} holds arms measured against incumbent {loaded.incumbent_token!r}, "
            f"but this round's incumbent is {incumbent_token!r}. Merging them would put "
            f"non-comparable numbers in one file -- pass a different --out path instead."
        )
    return dict(loaded.results)


def write_results(results: list[ArmResult], out_path: str, *, incumbent_token: str = "") -> None:
    """Writes an `ArmResult` list back in the exact shape `ExternalEvaluation.from_json` reads --
    for whatever process (this repo's own scoring run, or a hand-written one) records measurements
    against the arms named in a manifest this file wrote.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {"incumbent_token": incumbent_token, "results": [asdict(r) for r in results]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
