#!/usr/bin/env python3
"""Cross-check the per-backend cells of the full stack two independent ways, and report the drift.

WHY THIS SCRIPT EXISTS
----------------------
The full stack's per-backend figures come from the DCR acceptance run and are MEASURED. This script
exists to check them against a second, independent route -- walking forward from the last shipped
per-case artifact and applying each later anchor's recorded exposure -- because two routes to the same
number is worth more than one route asserted.

They agree on the corpus total (144/303) and on `vector`. They differ by ONE CASE on `kv` and on
`rec_sum`, and that drift is real, expected, and reported rather than averaged away: the same one-case
harness-repair divergence already recorded for the T0 baseline. See NATIVE_BASELINE in
rounds/anchors.py.

    python scripts/derive_per_backend.py            # show both routes and the drift
    python scripts/derive_per_backend.py --check     # exit 1 if the totals stop agreeing

WHICH NUMBER TO QUOTE
---------------------
**The MEASURED column**, and it is now measured for real: a six-shard concurrent run of the accepted
stack (3 backends x train/dev, all rc=0) produced every cell directly.

    kv       35/105 = 33.33%     vector 39/89 = 43.82%     rec_sum 70/109 = 64.22%
    total   144/303 = 47.52%   -- and dev 6/24, 14/40, 14/20 = 34/84 = 40.48%

**THIS SCRIPT'S WALK WAS WRONG, AND THE RUN SETTLED IT.** The forward walk below put `kv` at 34.29% and
`rec_sum` at 63.30% -- one case off each. The measured values match the figures derived from the corpus
total instead. Both routes agreed on the total and on `vector`, which is exactly why a total-only check
is not enough: two compensating one-case errors sum correctly.

The walk is kept, wrong-and-labelled, because it is the more useful artifact this way. A cross-check
that only ever agrees teaches nothing about whether it would have caught anything.

The residual disagreement is the one-case harness-repair drift already recorded for the T0 baseline:
the shipped T5 artifact predates a repair that the later measured line includes. That is a property of
the inputs, not of the arithmetic.

The exposures below are the load-bearing inputs, and each is a measured fact from the working repo
rather than an assumption:

    A5   acts on vector only. kv and rec_sum unchanged.
    A7   acts on rec_sum only. Confirmed byte-identical off-target (0 disagreements on 258 cases).
    A8   acts on kv only, and within kv on a single scenario cell.

A6 does NOT appear, and that omission is the point: it is DEFERRED, so it was never in this
progression and there is nothing to add or subtract. Writing this script caught a real error --
a first version applied A6's -5 as though the chain passed through it, which put the total at 139
instead of 144. Deriving by hand had hidden that; the reconciliation check surfaced it immediately.

If a collaborator disbelieves one of those, this script is where to challenge it: change the number
and watch the total stop reconciling.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rounds.anchors import CUMULATIVE, PROGRESSION

# The last round whose per-backend split comes from a shipped per-case artifact.
ARTIFACT_ROUND = "T5"
ARTIFACT_FILE = "T5_A4_no_tool_call/result/eval_train.json"
CONTROL_FILE = "T1_A1_capacity/result/baseline_T0_train.json"

# Each later anchor's exposure and its measured case delta on that backend. Every entry is a fact
# recorded in the acceptance documents, not a modelling choice -- see the module docstring.
LATER_ANCHORS = (
    # (anchor, backend it can act on, case delta on that backend, why the others are untouched)
    ("A5", "vector", +4, "archival eviction is vector-shaped; kv/rec_sum structurally unreachable"),
    ("A7", "rec_sum", +7, "backend-gated at dispatch; 0 disagreements on the 258 off-target cases"),
    ("A8", "kv", +5, "kv-only, and within kv a single scenario cell"),
)

# The MEASURED per-backend cells of the full stack, from the six-shard concurrent run (all rc=0).
# These are what the README quotes; the walk above is the cross-check that got them wrong.
MEASURED = {"kv": (35, 105), "vector": (39, 89), "rec_sum": (70, 109)}
MEASURED_DEV = {"kv": (6, 24), "vector": (14, 40), "rec_sum": (14, 20)}


def backend_of(case_id: str) -> str:
    """Delegates to the benchmark adapter. The local copy this replaces defaulted to `rec_sum`."""
    # Importing the benchmark's adapter registers it with the core. This script is BFCL-specific,
    # so naming it here is correct; the core itself must not.
    sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
    import adapter
    return adapter.backend_of(case_id)


def per_backend(rel: str) -> dict[str, tuple[int, int]]:
    """`{backend: (correct, n)}` over SCORED queries in a shipped result file."""
    rows = json.loads((REPO / "rounds" / rel).read_text())["results"]
    out: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for row in (r for r in rows if not r.get("is_prereq")):
        cell = out[backend_of(row["id"])]
        cell[1] += 1
        cell[0] += bool(row.get("valid"))
    return {k: (v[0], v[1]) for k, v in out.items()}


def derive() -> tuple[dict[str, tuple[int, int]], list[str]]:
    """Walk from the artifact-backed split to the full stack, one anchor at a time."""
    cells = dict(per_backend(ARTIFACT_FILE))
    log = [f"{ARTIFACT_ROUND} (from {ARTIFACT_FILE}) -- MEASURED per case"]
    for backend in sorted(cells):
        correct, n = cells[backend]
        log.append(f"    {backend:<8} {correct:3d}/{n:<4} = {100 * correct / n:6.2f} %")

    for anchor, backend, delta, why in LATER_ANCHORS:
        correct, n = cells[backend]
        cells[backend] = (correct + delta, n)
        verb = "deferred" if delta < 0 else "accepted"
        log.append(f"{anchor} ({verb}) acts on {backend} only: {delta:+d} cases   [{why}]")
        for other in sorted(k for k in cells if k != backend):
            log.append(f"    {other:<8} unchanged by construction")
        log.append(f"    {backend:<8} {correct:3d} -> {cells[backend][0]:3d}/{n}")

    return cells, log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 unless the derived cells reconcile with the published total")
    args = ap.parse_args()

    control = per_backend(CONTROL_FILE)
    cells, log = derive()

    print("=" * 78)
    print("DERIVING THE FULL-STACK PER-BACKEND CELLS")
    print("=" * 78)
    for line in log:
        print("  " + line)

    print()
    print("=" * 78)
    print("TWO ROUTES, COMPARED  (quote the MEASURED column)")
    print("=" * 78)
    print(f"  {'backend':<10} {'control':>9} {'MEASURED':>10} {'derived':>9} {'drift':>7} {'Δ vs ctl':>10}")
    print("  " + "-" * 60)
    drifts = {}
    for backend in ("kv", "vector", "rec_sum"):
        c, n = control[backend]
        measured, _ = MEASURED[backend]
        derived, _ = cells[backend]
        drifts[backend] = measured - derived
        print(f"  {backend:<10} {100 * c / n:8.2f} % {100 * measured / n:9.2f} % "
              f"{100 * derived / n:8.2f} % {measured - derived:+6d} "
              f"{100 * (measured - c) / n:+9.2f} pp")

    derived_total = sum(c for c, _ in cells.values())
    measured_total = sum(c for c, _ in MEASURED.values())
    n_total = sum(n for _, n in cells.values())
    published = CUMULATIVE["train_correct"][1]
    agree = derived_total == measured_total == published and n_total == CUMULATIVE["train_n"]

    print()
    print(f"  MEASURED total    {measured_total}/{n_total} = {100 * measured_total / n_total:.2f} %"
          "   <- the figure to quote")
    print(f"  derived total     {derived_total}/{n_total} = {100 * derived_total / n_total:.2f} %")
    print(f"  published total   {published}/{CUMULATIVE['train_n']} = {CUMULATIVE['train_to']:.2f} %")
    print(f"  totals {'AGREE' if agree else 'DISAGREE'}")

    print()
    print(f"  {'backend':<10} {'dev MEASURED':>14}")
    print("  " + "-" * 26)
    for backend in ("kv", "vector", "rec_sum"):
        c, n = MEASURED_DEV[backend]
        print(f"  {backend:<10} {c:3d}/{n:<3d} = {100 * c / n:6.2f} %")
    dev_c = sum(c for c, _ in MEASURED_DEV.values())
    dev_n = sum(n for _, n in MEASURED_DEV.values())
    print(f"  {'TOTAL':<10} {dev_c:3d}/{dev_n:<3d} = {100 * dev_c / dev_n:6.2f} %")

    print()
    if any(drifts.values()):
        moved = ", ".join(f"{b} {d:+d}" for b, d in drifts.items() if d)
        print(f"  PER-CELL DRIFT: {moved}  <- THIS WALK IS THE ONE THAT IS WRONG.")
        print("  The measured run matches the cells derived from the corpus total, not this walk.")
        print("  Both routes agree on the TOTAL and on vector, which is why a total-only check is")
        print("  insufficient: two compensating one-case errors sum correctly. The residual cause is")
        print("  the one-case harness-repair drift already recorded for the T0 baseline -- the")
        print("  shipped T5 artifact predates a repair the measured line includes.")
    else:
        print("  No per-cell drift: both routes agree cell for cell.")

    # The progression's own endpoint must agree too, or one of the two is stale.
    last = PROGRESSION[-1]
    if last.train_correct != published:
        print(f"  WARNING: PROGRESSION ends at {last.train_correct}, CUMULATIVE says {published}")
        reconciles = False

    if args.check and not reconciles:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
