#!/usr/bin/env python3
"""Attest a candidate REROUTE DESTINATION on whether it RESOLVES, not on whether it returns.

Written after A2.R was rejected at +0.00 pp with 44/44 firings. `archival_memory_key_search` had
been admitted as "attested 11/11 clean", where *clean* was implemented as **the call did not
error**. It returns RANKED KEYS, not values -- so every one of those 11 outcomes was a
syntactically successful, semantically empty payload. The anchor then did exactly what it was
designed to do, 44 times, and changed nothing.

That is the `vacuous_result` trap applied to a candidate DESTINATION rather than to a mined
signal, and this repo already shipped the detector for it (`policy_tree.vacuous_result_kind`).
The actionability gate simply did not use it.

THE RULE THIS ENFORCES

    A destination is attested only if it is observed to SUPPLY WHAT THE FAILING READ NEEDED --
    not merely to return without an error marker.

Three outcome classes, and only the first counts:

    resolving   payload carries usable content (a value, a non-empty non-floor collection)
    vacuous     payload is successful-looking but empty / all-floor  -> NOT attested
    errored     payload carries an explicit error marker             -> NOT attested

A destination whose observations are mostly `vacuous` is a *dead end that reports success*, which
is strictly worse than an error: the error at least keeps the model searching.

Usage:
    python scripts/destination_attest.py --traj-dir <dir> --trigger "Key not found"
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# PACKAGE-RELATIVE, not a sys.path hack. This read `sys.path.insert(0, <this file's dir>)` followed by
# `import policy_tree`, and `policy_tree` lives in anchoropt/learning/ -- not here -- so the import
# could only ever raise ModuleNotFoundError. Nothing caught it because no test imported this module and
# its CLI was always run from the repo root, where the ambient path happened to resolve it. Found by
# importing every core module in a clean checkout.
from anchoropt.learning import policy_tree as pt

_CALL = re.compile(r"([a-z_][a-z_0-9]*)\s*\(")


def classify_outcome(payload) -> str:
    """resolving | vacuous | errored -- the three-way split the old gate collapsed to two."""
    s = payload if isinstance(payload, str) else str(payload)
    if pt._canon_error(s) is not None:
        return "errored"
    if pt.vacuous_result_kind(s) is not None:
        return "vacuous"
    # A payload with no extractable content is vacuous even when the detector has no
    # collection to inspect: "returned an empty string" is not a resolution either.
    if not re.search(r"[A-Za-z0-9]", s):
        return "vacuous"
    return "resolving"


def attest(traj_dir: Path, trigger: str, window: int = 3) -> Dict[str, Dict[str, int]]:
    """For each call tried within `window` steps after `trigger`, tally its outcome classes."""
    tally: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in sorted(Path(traj_dir).rglob("*.json")):
        try:
            ep = json.load(open(p))
        except Exception:
            continue
        steps = ep.get("steps") or []
        for i, s in enumerate(steps):
            if not any(trigger in str(r) for r in (s.get("tool_results") or [])):
                continue
            for s2 in steps[i + 1:i + 1 + window]:
                dec = s2.get("decoded") or []
                res = s2.get("tool_results") or []
                for j, c in enumerate(dec):
                    m = _CALL.search(str(c))
                    if not m:
                        continue
                    r = res[j] if j < len(res) else ""
                    tally[m.group(1)][classify_outcome(r)] += 1
                if dec:
                    break
    return {k: dict(v) for k, v in tally.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True, type=Path)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--min-resolving", type=int, default=10,
                    help="attested only at this many RESOLVING observations")
    a = ap.parse_args()

    tally = attest(a.traj_dir, a.trigger)
    if not tally:
        print("no calls observed after %r" % a.trigger, file=sys.stderr)
        return 1
    print("candidate destinations after %r\n" % a.trigger)
    print("%-34s %10s %9s %8s   %s" % ("call", "resolving", "vacuous", "errored", "verdict"))
    attested = []
    for call, t in sorted(tally.items(), key=lambda kv: -sum(kv[1].values())):
        res, vac, err = t.get("resolving", 0), t.get("vacuous", 0), t.get("errored", 0)
        ok = res >= a.min_resolving
        if ok:
            attested.append((call, res))
        note = "ATTESTED" if ok else (
            "vacuous -- returns success with no content" if vac >= max(res, 1)
            else "too few resolving observations")
        print("%-34s %10d %9d %8d   %s" % (call, res, vac, err, note))
    print()
    if attested:
        print("attested destinations: %s" % ", ".join("%s (%d)" % x for x in attested))
    else:
        print("NO destination is attested to RESOLVE this locus.")
        print("  -> a single-call reroute cannot fix it; reject at the actionability gate")
        print("     rather than spending an arm to discover it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
