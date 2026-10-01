#!/usr/bin/env python3
"""One AnchorOpt self-evolve round on TerminalBench 2. Collaborator entry point.

    python examples/self_evolve_tb2.py --cases cases.txt --rounds 1
    python examples/self_evolve_tb2.py --cases cases.txt --dry-run     # no containers, no model

WHAT ONE ROUND DOES
    incumbent -> execute -> failed trajectories -> ResidualDiagnosis
      -> AnchorOpt selects (l, phi, mu, theta) from its typed space
      -> materialize the controller as a candidate harness surface
      -> paired evaluation vs the unmodified incumbent
      -> AnchorOpt acceptance rule -> promote if accepted

`--dry-run` exercises everything except containers and the model: it feeds diagnoses through the
real search and prints what the optimizer WOULD install. Start there -- it needs no Docker, no
endpoint, and no TB2 clone, and it is the fastest way to see the structured space at work. By
default it reads the small sample fixture shipped with this repo
(`examples/fixtures/sample_diagnoses.json`); pass `--diagnoses` to point it at your own harness's
diagnosis export instead -- a JSON list of objects with the same fields (see
`anchoropt.runtime.ResidualDiagnosis`).

See docs/SELF_EVOLVE_V0.md for the architecture and what v0.1 deliberately omits (no automatic
Phi-expansion, no clustering).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "tb2_deepagents"))

import tb2_adapter as tb2                                            # noqa: E402
from anchoropt.learning.self_evolve import (                         # noqa: E402
    Incumbent, OUTCOME_SELECTED_ONLY, step,
)
from anchoropt.anchor import Action                                  # noqa: E402
from anchoropt.runtime import ResidualDiagnosis                      # noqa: E402

DEFAULT_DIAGNOSES = REPO / "examples" / "fixtures" / "sample_diagnoses.json"

# The frozen instruction the round-0/round-1 experiments used. theta payload only -- AnchorOpt
# chooses WHERE and WHEN it applies; it never rewrites the text.
VERIFY_EXACT = (
    "Before concluding, verify the result with the most targeted command, file read, or test you "
    "can run. Explicitly check for exact string matches, required directory structures, and "
    "version outputs. Do not assume success based on partial matches or baseline artifacts; "
    "validate against the precise task requirements."
)


def theta_for(action: Action, diagnosis) -> dict | None:
    """Typed parameters per semantic action, or None to prune that action.

    v0.1 supplies theta for the two actions whose parameters can be derived from a diagnosis
    without inventing anything. Returning None is how the search learns an action is unusable
    here, rather than being handed empty parameters and reporting a hollow success.
    """
    if action is Action.REPROMPT:
        return {"text": diagnosis.proposed_behavior_change or VERIFY_EXACT}
    if action is Action.SUPPRESS:
        return {"reason": f"suppressed by AnchorOpt: {diagnosis.failure_mechanism[:80]}"}
    return None          # REROUTE needs a concrete destination; v0.1 does not synthesize one


def load_diagnoses(path: pathlib.Path) -> tuple[ResidualDiagnosis, ...]:
    """Load a JSON list of diagnosis records -- the shipped sample fixture, or your own export.

    Each record needs the fields `anchoropt.runtime.ResidualDiagnosis` declares: `case_id`,
    `mechanism`, `evidence`, `consequential_decision`, `proposed_behavior_change`, and optionally
    `provider`. Translate your own harness's diagnosis format into this shape once; nothing else
    in this example needs to change.
    """
    if not path.exists():
        sys.exit(f"diagnosis file not found: {path}\n"
                  f"Pass --diagnoses pointing at your own harness's diagnosis export, in the same "
                  f"shape as examples/fixtures/sample_diagnoses.json.")
    records = json.loads(path.read_text())
    return tuple(ResidualDiagnosis(**r) for r in records)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=pathlib.Path,
                    help="file with one TB2 task name per line (live mode)")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="search only: no containers, no model, no TB2 clone needed")
    ap.add_argument("--workspace", type=pathlib.Path,
                    help="incumbent workspace containing repo_baseline.py (live mode)")
    ap.add_argument("--diagnoses", type=pathlib.Path, default=DEFAULT_DIAGNOSES,
                    help=f"JSON diagnosis export to search over (default: the shipped sample "
                         f"fixture, {DEFAULT_DIAGNOSES.relative_to(REPO)})")
    a = ap.parse_args()

    if not a.dry_run and not (a.cases and a.workspace):
        sys.exit("live mode needs --cases and --workspace; try --dry-run first")

    diagnoses = load_diagnoses(a.diagnoses)
    incumbent = Incumbent(workspace=a.workspace or pathlib.Path("."), incumbent_id="P0")

    for rnd in range(1, a.rounds + 1):
        print("=" * 78)
        print(f"ROUND {rnd}   incumbent={incumbent.incumbent_id}")
        print("=" * 78)

        result = step(incumbent, runtime=tb2, host=tb2.HOST,
                      diagnose=lambda: diagnoses, theta_for=theta_for)

        print(f"\ndiagnoses            : {len(result.diagnoses)}")
        print(f"candidates considered: {len(result.candidates_considered)}")
        print(f"  survived U_H(l) + Phi : {len(result.survived_candidates)}")
        print(f"SIGNAL_BLOCKED       : {len(result.blocked_diagnoses)}")
        for d, why in result.blocked_diagnoses[:5]:
            print(f"    {d.case_id:24s} {why[:88]}")

        sc = result.selected_candidate
        if sc is None:
            print("\nno candidate: every diagnosis was SIGNAL_BLOCKED or every cell pruned.")
            print("That is a REPRESENTATION limit, not a policy one -- the Phi-expansion backlog.")
            return 0

        print(f"\nSELECTED CONTROLLER")
        print(f"  l     = {sc.boundary.value}")
        print(f"  phi   = {sc.signal}")
        print(f"  mu    = {sc.action.value}")
        print(f"  theta = {{{', '.join(f'{k}: {str(v)[:48]!r}' for k, v in sc.theta.items())}}}")
        print(f"  score = {sc.score}   ({sc.rationale})")

        print(f"\ntop survivors considered:")
        for c in sorted(result.survived_candidates, key=lambda c: -c.score)[:5]:
            print(f"  {c.score:5.1f}  {c.boundary.value:26s} {c.signal:26s} {c.action.value}")

        if result.outcome == OUTCOME_SELECTED_ONLY:
            print(f"\noutcome: {result.outcome} -- selection ran; no evaluator wired.")
            print("Live paired evaluation needs Docker, a TB2 clone, an endpoint, and an "
                  "`evaluate=` callback wired into step() -- see docs/SELF_EVOLVE_V0.md.")
            return 0

        print(f"\noutcome: {result.outcome}  accepted={result.accepted}")
        if result.new_incumbent:
            incumbent = result.new_incumbent
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
