#!/usr/bin/env python3
"""Shared harness-fault guard for every trajectory consumer.

`policy_tree.load()` gates the mining path, but six scripts read sidecars with their
own `rglob`/`glob` and so bypass it -- including the attribution tools the residual
phase depends on (trace_backward, error_attribution_study, obligation_trace,
obligation_readside). A guard that only covers one entry point is not a guard.

This module is deliberately dependency-free (stdlib only, no bfcl_eval, no anchoropt
package) so a script that runs without the eval stack can still assert cleanliness.

Contract, matching policy_tree:
  * a HARNESS fault is a crash inside a tool implementation. It says nothing about the
    policy under test, but reaches the sidecar as an ordinary error payload -- so it can
    be canonicalised into a mineable "error contract" and a gate learned against a
    stack-trace string. That already happened once: "error during execution division by
    zero" ranked with support=13 in the corpus all prior mining used (plan §105).
  * a DOMAIN error ("Key not found", "Core memory is full", "exceeds maximum length") is
    exactly what mining exists to find and must pass through untouched.

Usage:
    from harness_guard import assert_clean, fault_report, is_harness_fault
    assert_clean(episodes, source=str(traj_dir))          # raises on contamination
    assert_clean(episodes, source=..., allow=True)        # warn + drop, returns clean list
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

# Keep in sync with policy_tree._HARNESS_FAULTS.
HARNESS_FAULTS: Tuple[str, ...] = (
    "division by zero",
    "zerodivisionerror",
    "error during execution:",
    "traceback (most recent call last)",
)


def is_harness_fault(text) -> bool:
    """True if a tool payload is an implementation crash rather than a domain error."""
    low = str(text).lower()
    return any(sig in low for sig in HARNESS_FAULTS)


def episode_has_fault(ep: Dict) -> bool:
    return any(is_harness_fault(t)
               for s in (ep.get("steps") or [])
               for t in (s.get("tool_results") or []))


def fault_report(eps: Iterable[Dict]) -> Dict:
    """Counts and per-signature/per-backend breakdown, for a loud message."""
    eps = list(eps)
    n_eps = n_steps = 0
    sigs: Counter = Counter()
    backends: Counter = Counter()
    cases: List[str] = []
    for e in eps:
        hit = False
        for s in e.get("steps") or []:
            for t in (s.get("tool_results") or []):
                if is_harness_fault(t):
                    n_steps += 1
                    hit = True
                    low = str(t).lower()
                    for sig in HARNESS_FAULTS:
                        if sig in low:
                            sigs[sig] += 1
        if hit:
            n_eps += 1
            backends[e.get("backend") or "?"] += 1
            cases.append(e.get("case_id") or "?")
    return {"episodes": n_eps, "steps": n_steps, "signatures": dict(sigs),
            "backends": dict(backends), "cases": cases}


def assert_clean(eps: Iterable[Dict], source: str = "<episodes>",
                 allow: bool = False) -> List[Dict]:
    """Refuse contaminated trajectories. Returns the episode list to use.

    Default raises: a contaminated corpus silently produces a plausible-looking gate,
    which is strictly worse than a stopped analysis. `allow=True` acknowledges the
    contamination, warns loudly, and drops the affected episodes.
    """
    eps = list(eps)
    rep = fault_report(eps)
    if not rep["episodes"]:
        return eps
    msg = (f"[HARNESS FAULT] {rep['episodes']} episode(s) / {rep['steps']} step(s) in "
           f"{source} carry a harness crash, not a model failure: "
           f"signatures={rep['signatures']} backends={rep['backends']}. "
           f"Attributing or mining this corpus would credit a crash string. "
           f"Fix the harness and re-run, or pass allow=True to acknowledge and drop them.")
    if not allow:
        raise RuntimeError(msg)
    print("[warn] " + msg, file=sys.stderr)
    clean = [e for e in eps if not episode_has_fault(e)]
    print(f"[warn] dropped {len(eps) - len(clean)} episode(s); {len(clean)} remain",
          file=sys.stderr)
    return clean


def add_cli_flag(parser) -> None:
    """Standard opt-out flag, so every consumer spells it the same way."""
    parser.add_argument("--allow-harness-faults", action="store_true",
                        help="acknowledge harness-fault episodes and DROP them instead of "
                             "aborting (default: abort, because a contaminated corpus "
                             "yields a plausible-looking but meaningless gate)")
