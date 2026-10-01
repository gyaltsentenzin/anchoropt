"""Build the `controllers.json` that validates an ALREADY-PROMOTED arm on a held-out split.

WHY THIS IS NOT `propose`
==================================================================================================
`propose` mines the residual of whatever evaluation it is given. Pointing it at the held-out
evaluation would mine the held-out residual and let the search select against it -- and an arm chosen
using held-out information, then measured on held-out, produces a number that is not a generalization
result. `docs/APPWORLD_RESULTS.md` records that protocol decision, and the earlier held-out search
states in that table are kept only as a record of what was run.

So held-out validation is not a search step at all. It takes an arm the TRAIN search already promoted,
changes nothing about it, and measures it once against the held-out incumbent. That is a pure
transformation of provenance, which is all this file does:

  * the controller spec is copied VERBATIM from the train round's controllers.json -- same boundary,
    action, signal, eta. Any edit here would mean validating a different arm than the one selected.
  * `incumbent_id` / `incumbent_token` are re-pointed at the held-out evaluation, because
    `run_round.py score` reads the token from THIS file (not from --evaluation) and stamps it onto
    every ArmResult, and `evaluate.load_results` raises when a results file's token disagrees. Leaving
    the train token in place would silently record a held-out measurement as a train one.

Usage:
  heldout_validation.py --from <train controllers.json> --arm <arm_label> \
      --heldout-evaluation <path to sh_heldout.json> \
      --heldout-experiment <experiment_name ending in /sh_heldout> \
      --out <directory>
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="src", required=True,
                   help="controllers.json from the TRAIN round that promoted the arm")
    p.add_argument("--arm", required=True, help="arm_label to validate (must exist in --from)")
    p.add_argument("--heldout-evaluation", required=True,
                   help="held-out evaluations/<split>.json -- the frozen held-out incumbent")
    p.add_argument("--heldout-experiment", required=True,
                   help="experiment_name for the held-out split")
    p.add_argument("--out", required=True, help="directory to write controllers.json into")
    args = p.parse_args(argv)

    with open(args.src, encoding="utf-8") as f:
        src = json.load(f)

    matches = [c for c in src.get("controllers", []) if c.get("arm_label") == args.arm]
    if not matches:
        have = [c.get("arm_label") for c in src.get("controllers", [])]
        print(f"arm {args.arm!r} not found in {args.src}. Present: {have}", file=sys.stderr)
        return 2
    if len(matches) > 1:
        print(f"arm {args.arm!r} appears {len(matches)} times in {args.src}", file=sys.stderr)
        return 2

    if not os.path.isfile(args.heldout_evaluation):
        print(f"held-out incumbent not found: {args.heldout_evaluation}", file=sys.stderr)
        return 2

    # Must match `evaluate.incumbent_identity`, which is `os.path.abspath(evaluation_path)`. A
    # relative or symlinked path here would not compare equal to what a later run computes.
    token = os.path.abspath(args.heldout_evaluation)

    # Guard against the mistake this file exists to prevent: validating against the split the arm was
    # selected on is not held-out validation, it is re-measuring train.
    if os.path.abspath(src.get("incumbent_token", "")) == token:
        print("--heldout-evaluation is the SAME incumbent the arm was selected against; "
              "that is not held-out validation", file=sys.stderr)
        return 2

    payload = {
        "incumbent_id": args.heldout_experiment,
        "incumbent_token": token,
        "controllers": [dict(matches[0])],          # verbatim, single arm
        "provenance": {
            "selected_on": src.get("incumbent_id", ""),
            "selected_against_token": src.get("incumbent_token", ""),
            "note": "arm promoted by the train search; copied unchanged for held-out validation. "
                    "No held-out residual was mined to produce this file.",
        },
    }

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "controllers.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")

    print(f"wrote {dest}")
    print(f"  arm            : {args.arm}")
    print(f"  selected on    : {src.get('incumbent_id','')}")
    print(f"  validating vs  : {args.heldout_experiment}")
    print(f"  incumbent_token: {token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
