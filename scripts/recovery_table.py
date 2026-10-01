#!/usr/bin/env python3
"""Print the anchor-recovery table from whatever rounds/REC_* results exist. Read-only."""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
ORDER = ["A1", "A2", "A3", "A4", "A5", "A7", "A8", "A9"]
# Verdicts that count as a REDISCOVERY (structure reconstructed). COINCIDENTAL deliberately does NOT:
# locus+action agreeing while the attribution describes a different failure is not rediscovery
# (see REC_A3).
COUNTS = {"EQUIVALENT", "EXACT"}

# ------------------------------------------------------------------------------------------------
# NOT TESTABLE != THE ALGORITHM FAILED
# ------------------------------------------------------------------------------------------------
#
# "A zero must be distinguishable from a channel that was never live." This table used to define
# `scored` as everything except NO_RESIDUAL, so NO_DIAGNOSES -- and any absent-evidence verdict --
# landed in the DENOMINATOR and depressed the headline as though the search had failed. Three of the
# eight anchors turn out to have no observable trigger in the failures the learner is shown, each for
# a different reason, so this distinction decides the headline number:
#
#   NO_RESIDUAL              the trigger is absent from the prior incumbent's trajectories entirely
#   NO_RESIDUAL_IN_FAILURES  the trigger occurs ONLY in recoverable/passing episodes
#   INSUFFICIENT_SUPPORT     present, but on too few episodes to support any verdict
#   NO_DIAGNOSES             attribution returned nothing usable -- a channel fault, not a result
#
# These are reported in their own column, never inside `recovered / testable`.
NOT_TESTABLE = {"NO_RESIDUAL", "NO_RESIDUAL_IN_FAILURES", "INSUFFICIENT_SUPPORT", "NO_DIAGNOSES"}

# Why each untestable verdict means, for the report. Keyed by verdict, not by anchor.
WHY_UNTESTABLE = {
    "NO_RESIDUAL": "trigger absent from the prior incumbent's trajectories",
    "NO_RESIDUAL_IN_FAILURES": "trigger occurs only in recoverable/passing episodes",
    "INSUFFICIENT_SUPPORT": "trigger present on too few episodes to support a verdict",
    "NO_DIAGNOSES": "attribution returned nothing usable (channel fault)",
}


def main() -> int:
    rows = {}
    for name in ORDER:
        p = REPO / f"rounds/REC_{name}/recovery.json"
        if p.exists():
            rows[name] = json.load(open(p))
    if not rows:
        print("no rounds/REC_*/recovery.json found")
        return 1

    W = 118
    print("=" * W)
    print("ANCHOR RECOVERY BENCHMARK  ·  can Self-Evolve rediscover the historically accepted anchors?")
    print("  learner sees prior-incumbent failures as RUNTIME FACTS only; the anchor is loaded after "
          "the search, to score")
    print("=" * W)
    hdr = (f"{'anchor':7s} {'supp':5s} {'recovered locus':27s} {'recovered signal':30s} "
           f"{'act':9s} {'via':10s} {'verdict':13s}")
    print(hdr)
    print("-" * W)
    for name in ORDER:
        r = rows.get(name)
        if r is None:
            print(f"{name:7s} {'-':5s} {'(not run)':27s}")
            continue
        v = r.get("verdict", "?")
        match = r.get("matching_candidates") or []
        loci = sorted({c["locus"] for c in match}) or r.get("recovered_loci") or []
        sigs = sorted({c["signal"] for c in match}) or r.get("recovered_signals") or []
        acts = sorted({c["action"] for c in match}) or r.get("recovered_actions") or []
        via = "expansion" if any(c.get("via") == "expansion" for c in match) else "shipped Phi"
        print(f"{name:7s} {str(r.get('failure_support','-')):5s} "
              f"{(loci[0] if loci else '-')[:27]:27s} {(sigs[0] if sigs else '-')[:30]:30s} "
              f"{(acts[0] if acts else '-')[:9]:9s} {via:10s} {v:13s}")
        if r.get("mechanism_terms_found"):
            print(f"{'':7s} mechanism agreement: {r['mechanism_terms_found']}")
        if v == "COINCIDENTAL":
            print(f"{'':7s} NOT counted: {r.get('why','')[:96]}")
        if r.get("expansions"):
            ins = [e for e in r["expansions"] if e["state"] == "SIGNAL_INSTALLED"]
            print(f"{'':7s} expansion: {len(ins)} installed of {len(r['expansions'])} proposed"
                  + (f" -> {[e['signal'] for e in ins]}" if ins else ""))

    untestable = {n: r for n, r in rows.items() if r.get("verdict") in NOT_TESTABLE}
    scored = [r for n, r in rows.items()
              if r.get("verdict") is not None and n not in untestable]
    recovered = [r for r in scored if r.get("verdict") in COUNTS]
    print("-" * W)

    # REDISCOVERED, not "recovered": this benchmark runs with evaluation disabled (`evaluate -> None`)
    # and no-op promote/remine, so it can establish that the WHERE/WHAT/HOW structure was reconstructed
    # and NOT that it helps. Only an anchor with a paired measured result is `validated`/`promoted`.
    denom = len(scored)
    print(f"REDISCOVERY RATE: {len(recovered)} / {denom} TESTABLE "
          f"({', '.join(sorted(n for n, r in rows.items() if r.get('verdict') in COUNTS)) or 'none'})")
    print("  'rediscovered' = WHERE/WHAT/HOW structure reconstructed from residual trajectories.")
    print("  It is NOT 'validated' (paired evaluation shows benefit) and NOT 'promoted' (accepted")
    print("  into the moving incumbent) -- evaluation is disabled in this offline benchmark.")
    validated = sorted(n for n, r in rows.items() if r.get("measured_delta_pp") is not None)
    if validated:
        for n in validated:
            d = rows[n]["measured_delta_pp"]
            print(f"    VALIDATED + PROMOTED: {n} ({d:+.2f}pp, paired)")
    for v in ("PARTIAL", "COINCIDENTAL", "NO", "EXHAUSTED"):
        got = sorted(n for n, r in rows.items() if r.get("verdict") == v)
        if got:
            print(f"  {v:13s}: {got}")

    if untestable:
        print("-" * W)
        print(f"NOT TESTABLE ({len(untestable)}) -- OUTSIDE the denominator above. A null from a "
              f"channel that was never")
        print("  live is not a result about the algorithm.")
        for n in sorted(untestable):
            v = untestable[n].get("verdict", "?")
            why = untestable[n].get("untestable_reason") or WHY_UNTESTABLE.get(v, "")
            print(f"  {n:7s} {v:24s} {why}")
    print(f"\n  of 8 known anchors: {len(recovered)} rediscovered, {denom} testable, "
          f"{len(untestable)} untestable, {8 - len(rows)} not yet run")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
