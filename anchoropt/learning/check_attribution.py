#!/usr/bin/env python3
"""Attribution precondition for candidate acceptance (SIGNAL_POLICY_PLAN §26.3).

WHY THIS EXISTS

G2's winning arm passed all four terms of the lookahead criterion and was still the
wrong policy. `reprompt` scored +2.24pp while its gate fired 12 times on a *rec_sum*
error contract, the target residual was 187+ *vector* events, and that residual
GREW 188 -> 220. The criterion is entirely about outcomes (local efficacy, global
safety, guardrail, direction); nothing in it asked whether the lever engaged the
residual it was proposed for. So a policy that demonstrably did not act on its
target was accepted on churn.

This runs BEFORE outcome scoring and can veto an arm without spending confirmation
budget. Three preconditions, all from §26.3:

  A1  the candidate's own telemetry fired a non-trivial number of times
  A2  those firings occur on the target residual's backend / error contract
  A3  the target residual's own frequency did not INCREASE

DISCIPLINE

Everything here is derived from trajectory STRUCTURE only -- case_id, per-step
tool_results text, and `*_gate` telemetry flags. No reward label, accuracy figure or
`valid` field is read, for the same reason the exposure rule forbids it (§14.10): if
outcomes leak into the precondition, the precondition stops being independent
evidence and becomes a second, worse scorer.

Attribution PRUNES; measurement CHOOSES (§21). A PASS here is permission to spend
scoring budget, never evidence that the candidate is good.

Usage:
    python scripts/check_attribution.py \\
        --results-root results/g2_arms_train \\
        --baseline noop --candidate reprompt \\
        --telemetry-flag g5_gate \\
        --target-substring "Entry length exceeds maximum length" \\
        --target-backend vector
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# A1: fewer firings than this is indistinguishable from incidental activity. G2's
# rejected arm fired 12 times against a 187-event residual, so a bare ">0" test
# would have passed it; the threshold is expressed as coverage of the residual
# instead of a bare count (see --min-coverage).
MIN_FIRINGS = 5
# A1: firings must also cover a real share of the target residual. 12/187 = 6%.
MIN_COVERAGE = 0.25
# A3: allow measurement jitter, but not growth. Expressed as a ratio of baseline.
MAX_RESIDUAL_GROWTH = 1.00
# Reported for every arm, never used as a pass/fail term (§27.2). A length remedy can
# in principle "fix" its target by trading it for a different contract, and a
# single-residual view cannot tell that apart from a real fix.
#
# The two capacity caps are listed SEPARATELY and on purpose. A bare
# "exceeds maximum size" matches BOTH vector's 7-entry core cap and its 50-entry
# archival cap, and on the G2 corpus that conflation was actively misleading: the
# aggregate read +86 for the transform arm, which looked like core-memory pressure
# induced by writing more short entries. Split apart it is core 145->139 (-6) and
# archival 50->142 (+92) -- i.e. essentially none of it was the cap a length remedy
# could plausibly shift, and nearly all of it was an unlatched archival retry loop
# whose event count scales with the episode's step budget (transform ran 1137 steps
# vs noop's 1025). Same lesson as the concentration rule: an aggregate that spans two
# mechanisms will be attributed to the wrong one.
DEFAULT_COMPANIONS = (
    "exceeds maximum size of 7 entries",    # vector core cap
    "exceeds maximum size of 50 entries",   # vector archival cap (different store)
)

def backend_of(case_id: str):
    """Backend from the case id. Structural: the corpus encodes it in the name.

    Delegates to the benchmark adapter, which owns the grammar. Returns None on an unrecognised id
    rather than raising -- this scans arm directories that can contain foreign files, and that was
    already this function's contract.
    """
    from anchoropt.attribution import active_adapter
    adapter = active_adapter()
    try:
        return adapter.backend_of(case_id)
    except adapter.UnknownCaseId:
        return None


def scan_arm(traj_dir: Path, target_substring: str, telemetry_flag: str,
             companions=(), target_backend_hint=None):
    """Count target-contract events and gate firings per backend. Outcome-blind.

    Returns (residual, firings, firings_on_target_contract, episodes_fired,
    companion_counts). The first three are Counters keyed by backend;
    `companion_counts` maps each companion substring -> Counter by backend.

    COMPANION CONTRACTS. A remedy can reduce its target residual by converting it
    into a DIFFERENT failure rather than by fixing the task, and a single-residual
    view cannot see that. vector enforces two independent limits -- per-entry length
    (300 chars) and total size (7 entries) -- so compressing one over-long entry into
    a shorter entry that now succeeds raises the entry COUNT and can trade a length
    error for a size error. Counting both is what distinguishes "fixed" from
    "moved", so it is reported for every arm whether or not it is being tested.
    """
    residual: Counter = Counter()
    firings: Counter = Counter()
    on_contract: Counter = Counter()
    episodes = set()
    comp = {c: Counter() for c in companions}
    n_steps = 0
    if not traj_dir.exists():
        return residual, firings, on_contract, episodes, comp, n_steps
    for phase in ("prereq", "query"):
        pdir = traj_dir / phase
        if not pdir.exists():
            continue
        for f in sorted(pdir.glob("*.json")):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            # Prefer the recorded backend; fall back to the case id so this also
            # works on corpora written before that field existed.
            b = j.get("backend") or backend_of(j.get("case_id", ""))
            if b is None:
                continue
            if b == target_backend_hint or target_backend_hint is None:
                n_steps += len(j.get("steps", []))
            for s in j.get("steps", []):
                blob = str(s.get("tool_results") or "")
                hit = target_substring in blob
                if hit:
                    residual[b] += 1
                for c in comp:
                    if c in blob:
                        comp[c][b] += 1
                if s.get(telemetry_flag):
                    firings[b] += 1
                    episodes.add(f.name)
                    # A2's core question: did the gate fire on a step whose OWN
                    # error is the target contract? A gate firing next to an
                    # unrelated error is not engagement with the residual.
                    if hit:
                        on_contract[b] += 1
    return residual, firings, on_contract, episodes, comp, n_steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--baseline", required=True,
                    help="arm subdir treated as the incumbent/no-op")
    ap.add_argument("--candidate", required=True, help="arm subdir under test")
    ap.add_argument("--telemetry-flag", required=True,
                    help="the candidate gate's telemetry_flag, e.g. g5_gate")
    ap.add_argument("--target-substring", required=True,
                    help="exact error text of the target residual's contract")
    ap.add_argument("--target-backend", required=True,
                    help="backend the target residual lives on")
    ap.add_argument("--companion-substring", action="append", default=None,
                    help="additional error contract to REPORT per arm (repeatable). "
                         "Defaults to vector's capacity contract, which is the "
                         "constraint a length remedy can convert its target into.")
    ap.add_argument("--backend-scope", action="append", default=None,
                    help="Backends the gate under test declares (repeatable). Read from "
                         "GateSpec.backends by the runner, not typed. The A2 "
                         "backend-concentration clause is applied ONLY when exactly one "
                         "is given: a gate scoped to several backends legitimately splits "
                         "its firings across them (§56).")
    ap.add_argument("--min-firings", type=int, default=MIN_FIRINGS)
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    companions = tuple(a.companion_substring
                       if a.companion_substring is not None
                       else DEFAULT_COMPANIONS)

    root = Path(a.results_root)
    base_res, _, _, _, base_comp, base_steps = scan_arm(
        root / a.baseline / "traj", a.target_substring, a.telemetry_flag,
        companions, a.target_backend)
    cand_res, cand_fire, cand_on, cand_eps, cand_comp, cand_steps = scan_arm(
        root / a.candidate / "traj", a.target_substring, a.telemetry_flag,
        companions, a.target_backend)

    tb = a.target_backend
    base_n = base_res.get(tb, 0)
    cand_n = cand_res.get(tb, 0)
    fired_total = sum(cand_fire.values())
    fired_tb = cand_fire.get(tb, 0)
    on_contract = cand_on.get(tb, 0)

    print(f"ATTRIBUTION  {a.candidate}  vs  {a.baseline}")
    print(f"  target: {tb} :: {a.target_substring!r}")
    print(f"  residual on {tb}:            {base_n} -> {cand_n}")
    print(f"  firings ({a.telemetry_flag}):      {fired_total} total, "
          f"{fired_tb} on {tb}, across {len(cand_eps)} episodes")
    print(f"  firings on target contract:  {on_contract}")
    if fired_total:
        other = {b: n for b, n in cand_fire.items() if b != tb}
        if other:
            print(f"  firings on OTHER backends:   {other}")

    # Companion contracts: REPORTED, never scored. A target residual that falls
    # while a companion residual rises is a constraint SHIFT, and it is the first
    # thing to check when an arm helps locally but hurts globally.
    if companions:
        print("  companion contracts (reported, not scored):")
        for c in companions:
            b0 = base_comp.get(c, Counter()).get(tb, 0)
            b1 = cand_comp.get(c, Counter()).get(tb, 0)
            print(f"    {c!r} on {tb}: {b0} -> {b1} ({b1 - b0:+d})")
        shifted = [
            c for c in companions
            if cand_comp.get(c, Counter()).get(tb, 0)
            > base_comp.get(c, Counter()).get(tb, 0)
        ]
        if cand_n < base_n and shifted:
            # Deliberately phrased as a prompt to investigate, not a finding. On the
            # G2 corpus this pattern held and constraint-shifting was still the WRONG
            # explanation: the rise was in a different store's cap and tracked the
            # step budget. Establishing a shift needs the case-level check below --
            # aggregate co-movement is not evidence of it.
            print(f"    NOTE: target fell {base_n - cand_n} while {shifted} rose.")
            print("      Do NOT call this a constraint shift on these counts alone. "
                  "Check, per episode: (a) is the companion error already present "
                  "BEFORE the first accepted remedy, (b) is the rise concentrated in "
                  "a few episodes, (c) does it track total step count instead.")
    print(f"  steps: {base_steps} -> {cand_steps} ({cand_steps - base_steps:+d})"
          "   [retry-loop event counts scale with this]")

    # A zero baseline residual is MISSING EVIDENCE, not a failed candidate.
    #
    # base_n comes from the BASELINE traj dir (see the scan_arm call above). When a control
    # cache-hits a reused world it can skip whole phases -- e.g. the 96 prereq episodes where
    # the G1 write-phase residual lives -- leaving base_n == 0. Coverage then reads 0/0 = 0%
    # and on-contract firings read 0, so A1/A2 fail BY CONSTRUCTION and the veto looks like a
    # finding about the candidate (SIGNAL_POLICY_PLAN 92.1). Refuse to score instead.
    if base_n == 0:
        print()
        print("  INDETERMINATE: baseline residual is 0 on the target contract.")
        print("  Coverage and on-contract firings have no denominator, so no verdict is")
        print("  possible. This is ABSENT BASELINE EVIDENCE -- it is NOT evidence that the")
        print("  candidate failed to engage.")
        print("  Likely cause: the baseline arm reused a cached world and skipped the phase")
        print("  where this residual occurs. Re-run the baseline with the full phase set, or")
        print("  point --baseline at a control that has it.")
        print()
        print("ATTRIBUTION INDETERMINATE: no verdict (baseline evidence absent).")
        return 2

    checks = []

    # A1 -- did the lever engage at all, at a scale that could matter?
    cov = (on_contract / base_n) if base_n else 0.0
    a1 = fired_total >= a.min_firings and cov >= a.min_coverage
    checks.append((
        "A1 fired non-trivially",
        a1,
        f"{fired_total} firings (need >={a.min_firings}), "
        f"target-contract coverage {on_contract}/{base_n} = {cov:.0%} "
        f"(need >={a.min_coverage:.0%})",
    ))

    # A2 -- were the firings on the target's error contract?
    #
    # The BACKEND-CONCENTRATION clause (>=50% of firings on the target backend) applies
    # ONLY when the gate under test is scoped to a single backend. It was written for G3,
    # whose gate declares backends=("vector",), and it is near-unsatisfiable BY
    # CONSTRUCTION for a gate scoped to two: on_domain_error_core_full declares
    # backends=("kv","vector") and split its 45 healthcare firings 21 kv / 24 vector, so
    # the 50% test vetoed a lever that had engaged its target correctly (§56).
    #
    # The frozen G1 criterion defines A2 for a reactive intervention as
    # `coincident_firings > 0` -- it declares NO concentration or dominance clause. Adding
    # one here imported a threshold the predeclaration never contained, which is exactly
    # the re-fitting the frozen-criterion discipline exists to prevent. So the scope now
    # comes from the SPEC (§37: declare the target once, derive the tests from it) rather
    # than being assumed.
    n_declared = len(a.backend_scope) if a.backend_scope else 0
    single_backend = (n_declared == 1)
    a2 = fired_total > 0 and on_contract > 0
    conc_note = ""
    if single_backend:
        a2 = a2 and on_contract == max(cand_on.values() or [0]) \
            and fired_tb >= fired_total * 0.5
        conc_note = (f"; {fired_tb}/{fired_total} on {tb} "
                     f"(concentration required: gate is {tb}-only)")
    elif n_declared > 1:
        conc_note = (f"; {fired_tb}/{fired_total} on {tb} -- concentration NOT required, "
                     f"gate declares {n_declared} backends {tuple(a.backend_scope)}")
    else:
        conc_note = (f"; {fired_tb}/{fired_total} on {tb} -- backend scope not declared, "
                     f"concentration not tested (pass --backend-scope to enable)")
    checks.append((
        "A2 firings on target contract",
        a2,
        f"{on_contract} firings on {tb}'s target contract{conc_note}",
    ))

    # A3 -- the residual the candidate was convened to fix must not grow.
    a3 = cand_n <= base_n * MAX_RESIDUAL_GROWTH
    delta = cand_n - base_n
    checks.append((
        "A3 target residual did not grow",
        a3,
        f"{base_n} -> {cand_n} ({delta:+d})",
    ))

    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    verdict = all(ok for _, ok, _ in checks)
    print()
    print(f"ATTRIBUTION {'PASS' if verdict else 'FAIL'} -- "
          + ("proceed to outcome scoring"
             if verdict else "do NOT score; the lever did not engage its target"))

    if a.json_out:
        Path(a.json_out).write_text(json.dumps({
            "candidate": a.candidate, "baseline": a.baseline,
            "target_backend": tb, "target_substring": a.target_substring,
            "residual_baseline": base_n, "residual_candidate": cand_n,
            "firings_total": fired_total, "firings_target_backend": fired_tb,
            "firings_on_target_contract": on_contract,
            "episodes_fired": len(cand_eps),
            "steps_baseline": base_steps, "steps_candidate": cand_steps,
            "companions": {
                c: {"baseline": base_comp.get(c, Counter()).get(tb, 0),
                    "candidate": cand_comp.get(c, Counter()).get(tb, 0)}
                for c in companions
            },
            "checks": {n: ok for n, ok, _ in checks},
            "verdict": "PASS" if verdict else "FAIL",
        }, indent=2))

    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
