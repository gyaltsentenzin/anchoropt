#!/usr/bin/env python3
"""Is this (control, arm) pair actually comparable? Checked, not assumed.

    python check_pairing.py <control_run_dir> <arm_run_dir>

WHAT THIS REPLACES. `rounds/CCTU_QWEN1/FROZEN.md` used to void an arm whose `endpoint` differed from
its control's, on the ground that a served model's node is part of its decode contract. That rule
discards arms over a node change the allocation schedule makes unavoidable, and three control jobs
spanning a moving default node agree byte-for-byte -- so the rule was costing arms without buying
comparability.

What buys comparability is a property of the two runs in hand. Episodes share no state
(`response_generator.py:736`) and decode is pinned, so on every episode where the controller NEVER
FIRED the arm must reproduce the control exactly. If it does not, those two runs disagree for a reason
that has nothing to do with the intervention -- a different build, a different quantisation, a tool
timeout that fired under load -- and no delta between them is attributable, whatever their endpoints
say. If it does, the pair is comparable and the endpoints are beside the point.

This is also FROZEN.md's ATTRIBUTION step made exact. "The reduction is concentrated in episodes the
controller executed in" is an argument about concentration; "every non-firing episode is identical" is
the same claim with no room in it.

CALL IDS ARE NORMALISED FIRST, and skipping that step is the trap. `chatcmpl-tool-<hex>` ids are
generated per request by the server, so two BIT-IDENTICAL runs produce different `response.jsonl`
digests -- measured on three jobs of one control, which agree on `detail.jsonl` and `scores.json` with
no normalisation and disagree on all three raw transcripts. A checker that diffed the raw file would
fail every pair it was given and prove nothing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CALL_ID = re.compile(r"chatcmpl-tool-[0-9a-f]+")


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _transcripts(run: Path) -> dict[str, str]:
    """id -> canonical transcript text, call ids normalised away."""
    out = {}
    for row in _jsonl(run / "response.jsonl"):
        blob = json.dumps(row.get("messages"), sort_keys=True, ensure_ascii=False)
        out[row["id"]] = CALL_ID.sub("<call-id>", blob)
    return out


def _fired_episodes(run: Path) -> set[str]:
    """Episode ids where a controller fired OR executed. Absent trace is fatal, not empty.

    An arm with no `anchor_trace.jsonl` cannot be checked and must not be reported as clean: every
    episode would look non-firing and the comparison would silently become "are these two runs the
    same run", which is a different question.
    """
    trace = run / "anchor_trace.jsonl"
    if not trace.exists():
        raise SystemExit(f"no anchor_trace.jsonl in {run}: re-run the arm with --anchor-trace. "
                         f"Without it, firing and non-firing episodes cannot be told apart")
    fired = set()
    for row in _jsonl(trace):
        if row.get("fired") or row.get("executed"):
            fired.add(str(row.get("case_id")))
    return fired


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip().splitlines()[2].strip())
    control, arm = Path(sys.argv[1]), Path(sys.argv[2])

    c_t, a_t = _transcripts(control), _transcripts(arm)
    fired = _fired_episodes(arm)

    problems: list[str] = []

    missing = sorted(set(c_t) ^ set(a_t))
    if missing:
        problems.append(f"the two runs do not cover the same episodes ({len(missing)} differ, "
                        f"e.g. {missing[:5]}): they are not a pair")

    shared = sorted(set(c_t) & set(a_t))
    quiet = [i for i in shared if i not in fired]
    loud = [i for i in shared if i in fired]
    drifted = [i for i in quiet if c_t[i] != a_t[i]]

    if not quiet:
        problems.append("the controller fired in EVERY episode, so nothing in this pair is a held-out "
                        "comparison and comparability cannot be checked this way")
    if drifted:
        problems.append(
            f"{len(drifted)} of {len(quiet)} non-firing episodes DIFFER from the control "
            f"(e.g. {drifted[:5]}): the two runs disagree where the intervention was never applied, "
            f"so no delta between them is attributable to it")

    # Timeouts are the one harness-side coupling `max_workers` is known to change. Counted here
    # because a pair can pass the identity check and still have been served under load that a later,
    # larger run would not survive -- worth reporting even when it changes no verdict.
    timeouts = sum(len(re.findall(r"timed out", json.dumps(t))) for t in a_t.values())

    print(f"  control : {control}")
    print(f"  arm     : {arm}")
    print(f"  episodes: {len(shared)} shared   firing: {len(loud)}   non-firing: {len(quiet)}")
    print(f"  tool-execution timeouts in the arm: {timeouts}")
    if timeouts:
        print("            (a wall-clock timeout fired, so this arm's tool results are load-dependent)")

    if problems:
        print("\n  NOT COMPARABLE")
        for p in problems:
            print(f"    - {p}")
        return 1
    print(f"\n  COMPARABLE: all {len(quiet)} non-firing episodes reproduce the control exactly, so a "
          f"difference on the remaining {len(loud)} is attributable to the intervention.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
