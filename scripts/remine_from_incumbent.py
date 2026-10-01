"""Re-mine the residual from the NEW incumbent's own trajectories, and prove they are new.

WHY THIS IS A SEPARATE STEP
--------------------------
Re-mining lives inside `self_evolve_cycle2.optimize_residual`'s promotion branch, which is reachable
only when a full search ran and promoted. But re-mining is not a sub-step of searching: it is the act of
reading the residual that an INSTALLED controller left behind, and after this sprint a controller can be
accepted on a path that never searched. Bolting re-mine onto the acceptance path would duplicate the
mining logic; making it its own step keeps one implementation and lets the round say, explicitly, which
trajectories it mined.

THE CLAIM THIS EXISTS TO SUBSTANTIATE
-------------------------------------
"The next round used the new incumbent's trajectories." That is the claim the yo-yo kept breaking, and
it is checkable rather than assertable: the trajectories a round mines have a content hash, and the
hash of the incumbent-stack run must DIFFER from the hash of the H0 run. If they match, the round
re-mined the same failures and the incumbent did not really move, whatever the library says.

So this reports three things and refuses to hand-wave any of them:

  * `trajectory_digest`   -- a content hash over the mined episodes, per run
  * `same_as`             -- which other run (if any) produced an identical digest
  * `residual_shift`      -- which failure keys disappeared and which are new

A digest collision between the H0 run and the incumbent run is the failure mode. It is reported as
`REMINE_INVALID`, not as an empty shift.

Usage:
  python scripts/remine_from_incumbent.py --incumbent <run> --baseline-h0 <run> --cell kv --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


#: WHERE THE AGENT'S BEHAVIOUR ACTUALLY LIVES in a BFCL step, read off a real trajectory rather than
#: guessed. A first version hashed ("decoded", "status", "error", "result") and produced IDENTICAL
#: digests for the relocation arm and its control -- because this schema has no top-level `error` or
#: `result` at all. Tool outcomes are in `tool_results`, and the constraint that refused a call is in
#: `constraint_state` (whose nested `error` carries the store's message). Hashing three absent keys and
#: one present one made two visibly different runs look byte-equivalent, and the detector then reported
#: REMINE_INVALID -- a manufactured failure, which is the same class of error as a manufactured pass.
#:
#: `status` and `decoded` alone are NOT enough: the agent can issue the same calls and get different
#: results once a controller repairs the store, which is exactly the effect being detected.
BEHAVIOUR_KEYS = ("decoded", "status", "tool_results", "constraint_state", "error_signal")

#: The store messages that mark a residual failure. Lexical on purpose -- see `failure_keys`.
#:
#: THE CELL-SHAPED BLIND SPOT THIS LIST HAD (rounds/AUTONOMY/R12). Every signature below is a kv or
#: rec_sum message, so NONE of vector's three capacity refusals matched and the remine step could not
#: see a single vector failure as a residual:
#:
#:     "Memory size exceeds maximum size of 7 entries."         core slots, 41 occurrences
#:     "Entry length exceeds maximum length of 300 characters." per-entry chars, 121 occurrences
#:     "Memory size exceeds maximum size of 50 entries."        archival slots
#:
#: That is a SECOND, independent blindness on the same cell as the `constraint_state` one, at a
#: different layer: even with live state supplied, mining could not classify the failure. The adapter
#: already declares these correctly -- `benchmarks/bfcl_v4/bfcl_constraints.py` maps
#: "is full" / "exceeds maximum size" -> no_capacity and "entry length" -> entry_too_long -- so the
#: defect was this list being maintained separately from that declaration.
#:
#: Kept lexical (the docstring explains why) but now covering all three backends. Substrings are
#: chosen to be UNIT-DISTINGUISHING, because "exceeds maximum size" bounds ENTRIES while "entry
#: length" bounds CHARACTERS and the remedy that can work differs -- shortening a value frees zero
#: slots. Do not collapse them into one "exceeds maximum" signature.
FAILURE_SIGNATURES = ("core memory is full", "key not found", "must be unique",
                      "too long after appending", "archival memory is full",
                      "name must be unique",
                      # vector (object-backed stores; same constraints, different wording)
                      "exceeds maximum size",      # ENTRIES -- kv/archival and vector core/archival
                      "entry length exceeds",      # CHARACTERS in one entry -- vector
                      "memory size exceeds")       # ENTRIES -- explicit vector form


def _outcome_text(step) -> list[str]:
    """Every string in this step that could carry a store outcome, flattened.

    `constraint_state` is a LIST OF DICTS with a nested `error`, so a str() of the step would find the
    text but a `.get("error")` would not. Flattening the declared containers keeps the search explicit
    instead of stringifying the whole step and matching on incidental text.
    """
    out: list[str] = []
    for key in ("tool_results", "constraint_state"):
        v = (step or {}).get(key)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    out.append(json.dumps(item, default=str))
        elif isinstance(v, dict):
            out.append(json.dumps(v, default=str))
    sig = (step or {}).get("error_signal")
    if sig:
        out.append(str(sig))
    return out


def _episodes(run: pathlib.Path, phase: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for sub in (run / "run" / "traj" / phase, run / "traj" / phase):
        if not sub.exists():
            continue
        for fp in sorted(sub.glob("*.json")):
            try:
                ep = json.loads(fp.read_text())
            except Exception:
                continue
            out[str(ep.get("case_id") or fp.stem)] = ep.get("steps") or []
    return out


def trajectory_digest(run: pathlib.Path, phase: str) -> tuple[str, int]:
    """A content hash over the mined episodes. Order-stable, content-sensitive.

    Hashes the DECODED CALLS and the error/status fields -- what the agent did and what came back --
    rather than the whole step dict, so telemetry keys a controller adds do not by themselves make two
    runs look different. Two runs whose agent behaviour is identical must hash identically even if one
    carries extra gate flags, or the check would report "new trajectories" for every arm trivially.
    """
    eps = _episodes(run, phase)
    h = hashlib.sha256()
    for cid in sorted(eps):
        h.update(cid.encode())
        for st in eps[cid]:
            for key in BEHAVIOUR_KEYS:
                v = (st or {}).get(key)
                if v is not None:
                    h.update(json.dumps(v, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16], len(eps)


def failure_keys(run: pathlib.Path, phase: str, cell: str) -> collections.Counter:
    """A coarse residual signature: which error texts remain, and how often.

    Deliberately lexical and deliberately coarse -- this is a SHIFT detector, not an attributor. The
    teacher does attribution; this only answers "did the distribution move".
    """
    c: collections.Counter = collections.Counter()
    for cid, steps in _episodes(run, phase).items():
        if cell and cell not in cid:
            continue
        for st in steps or ():
            for blob in _outcome_text(st):
                low = blob.lower()
                for sig in FAILURE_SIGNATURES:
                    if sig in low:
                        c[sig] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", type=pathlib.Path, required=True,
                    help="the NEW incumbent's run (stack installed)")
    ap.add_argument("--baseline-h0", type=pathlib.Path, default=None,
                    help="the run the previous round mined, to prove the trajectories are new")
    ap.add_argument("--cell", default="kv")
    ap.add_argument("--phase", default="prereq", choices=("prereq", "query"))
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()

    runs = {"incumbent": a.incumbent}
    if a.baseline_h0:
        runs["previous"] = a.baseline_h0

    digests, counts, resid = {}, {}, {}
    for name, run in runs.items():
        d, n = trajectory_digest(run, a.phase)
        digests[name], counts[name] = d, n
        resid[name] = dict(failure_keys(run, a.phase, a.cell))

    print(f"RE-MINE  cell={a.cell}  phase={a.phase}")
    for name in runs:
        print(f"  {name:10s} {counts[name]:>4} episodes  digest={digests[name]}  {runs[name]}")

    same = (len(digests) > 1 and len(set(digests.values())) == 1)
    valid = not same
    if same:
        print("\n  *** REMINE_INVALID: the incumbent run's trajectories are BYTE-EQUIVALENT to the "
              "run the previous round mined. Whatever the library says, this round would re-mine the "
              "SAME failures -- the incumbent did not really move in the evaluator.")
    elif len(digests) > 1:
        print("\n  trajectories DIFFER from the previously mined run -- this round mines new ones")

    shift = {}
    if "previous" in resid:
        before, after = resid["previous"], resid["incumbent"]
        keys = sorted(set(before) | set(after))
        print(f"\n  {'failure signature':34s} {'before':>7s} {'after':>7s} {'delta':>7s}")
        for k in keys:
            b, af = before.get(k, 0), after.get(k, 0)
            print(f"  {k:34s} {b:>7} {af:>7} {af - b:>+7}")
        shift = {"before": before, "after": after,
                 "disappeared": sorted(k for k in before if k not in after),
                 "new": sorted(k for k in after if k not in before),
                 "moved": before != after}
        print(f"\n  RESIDUAL DISTRIBUTION MOVED: {shift['moved']}")

    out = {"cell": a.cell, "phase": a.phase, "runs": {k: str(v) for k, v in runs.items()},
           "trajectory_digests": digests, "episode_counts": counts,
           "residual_by_run": resid, "residual_shift": shift,
           "trajectories_are_new": valid if len(digests) > 1 else None,
           "state": "REMINE_INVALID" if same else "OK"}
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "REMINE.json").write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {a.out / 'REMINE.json'}")
    return 1 if same else 0


if __name__ == "__main__":
    raise SystemExit(main())
