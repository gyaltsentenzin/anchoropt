#!/usr/bin/env python3
"""Convert the UNREALIZABLE repairs every round recorded into explicit affordance requirements.

The proposer already writes, per round, the repairs it could not express and why. Those records are
prose and die with the round. This reads them all, expresses each as (boundary, observation fields,
operation), asks the adapter what it supplies today, and reports which single observation would unblock
the most evidence.

The MAPPING from prose to fields is the one semantic step, and it is kept explicit and auditable here
rather than hidden: each rule names the phrase it matches and the field it implies. No anchor name, no
threshold, no policy.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "benchmarks" / "bfcl_v4", REPO / "scripts"):
    sys.path.insert(0, str(_p))

from anchoropt.learning.affordance_requirements import (      # noqa: E402
    AffordanceLedger, AffordanceRequirement,
)

# ---- the one semantic mapping, stated as data so it can be argued with -------------------------
# Each rule: phrases that identify the repair family -> what a controller would have to OBSERVE, at
# which boundary, to attempt it, and what operation it asks for. Derived by READING the teacher's
# sentences, not from any anchor.
RULES = [
    {"family": "already-refused repeat",
     "phrases": ("already been rejected", "already failed in this episode", "already refused",
                 "byte-identical", "effectively unchanged"),
     "boundary": "post_generation_pre_exec",
     "fields": ("proposal_was_refused_before",),
     "operation": "suppress",
     "primitive": "the existing suppress executor withholds the call; only the observation is new",
     "safety": ("a call never refused in this episode must not be suppressed",
                "the first attempt is always allowed through")},
    {"family": "per-entry sizing",
     "phrases": ("per-entry", "sized to the known", "length limit at composition"),
     "boundary": "post_generation_pre_exec",
     "fields": ("proposed_payload_chars", "per_entry_limit"),
     "operation": "transform_payload",
     "primitive": "",
     "safety": ("no content may be dropped; splitting must preserve every fact",)},
    {"family": "capacity budget",
     "phrases": ("running budget", "remaining capacity", "persist across turn"),
     "boundary": "post_generation_pre_exec",
     "fields": ("remaining_capacity", "proposed_payload_chars"),
     "operation": "suppress_or_resize",
     "primitive": "",
     "safety": ("a write that fits must not be blocked",)},
    {"family": "consolidate / merge",
     "phrases": ("merge", "condense", "consolidat"),
     "boundary": "post_execution",
     "fields": ("stored_entries", "proposed_payload"),
     "operation": "rewrite_store_entries",
     "primitive": "",
     "safety": ("explicitly without discarding stored content",
                "0 clears and 0 removes added")},
    {"family": "entailment / entity check",
     "phrases": ("containment", "entailment", "entity-level", "literally state"),
     "boundary": "post_execution",
     "fields": ("question_entities", "retrieved_entities"),
     "operation": "reprompt",
     "primitive": "the existing reprompt executor; only the observation is new",
     "safety": ("must not fire when the answer IS present in the retrieved text",)},
    {"family": "attempt counting",
     "phrases": ("counts failed", "second attempt", "abandon"),
     "boundary": "post_execution",
     "fields": ("failed_attempts_this_episode",),
     "operation": "reprompt",
     "primitive": "the existing reprompt executor; only the observation is new",
     "safety": ("a first failure must still be retried once",)},
]


def classify(repair: str):
    t = repair.lower()
    for r in RULES:
        if any(p in t for p in r["phrases"]):
            return r
    return None


def main() -> int:
    led = AffordanceLedger()
    unmatched = []
    for f in sorted(glob.glob(str(REPO / "results/*/cycle2.json"))
                    + glob.glob(str(REPO / "rounds/*/cycle2.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cases = [p.get("key", "")[:40] for p in (d.get("problems") or [])][:1]
        rnd = pathlib.Path(f).parent.name
        for u in (d.get("unrealizable_repairs") or []):
            rule = classify(u.get("repair", ""))
            if rule is None:
                unmatched.append((rnd, u.get("repair", "")[:90]))
                continue
            led.add(AffordanceRequirement(
                repair=u["repair"], boundary=rule["boundary"],
                observation_fields=tuple(rule["fields"]), operation=rule["operation"],
                candidate_primitive=rule["primitive"],
                safety_contract=tuple(rule["safety"]),
                evidence_case_ids=(f"{rnd}:{cases[0] if cases else '?'}",),
                provider="claude-attributor-v1", notes=rule["family"]))

    import bfcl_runtime as rt
    import self_evolve_cycle2 as c2
    # WHAT THE HOST SUPPLIES TODAY, measured rather than declared: the union of fields present on real
    # projected states at each boundary.
    supplied = {}
    for b in ("post_generation_pre_exec", "post_execution"):
        fields = set()
        for cell in ("kv", "vector", "rec_sum"):
            run = REPO / f"results/h0_native/nativectl_{cell}_train/run"
            if not (run / "eval_train_results.json").exists():
                continue
            st = c2.observable_states(c2._load_prereq(run))
            for s in rt.states_at(b, st):
                fields |= set(s)
        supplied[b] = fields
    ops = {"suppress", "reprompt", "substitute", "transform", "reroute"}

    print("=" * 96)
    print("AFFORDANCE AUDIT -- what the teacher asked for, and what the host cannot yet supply")
    print("=" * 96)
    print("\n%-26s %-30s %-18s %s" % ("family", "missing observation(s)", "operation", "evidence"))
    print("-" * 96)
    rows = []
    for r in led.requirements:
        miss = r.missing_observations(sorted(supplied.get(r.boundary, ())))
        st = r.status(supplied_fields=sorted(supplied.get(r.boundary, ())),
                      available_operations=sorted(ops))
        rows.append((len(r.evidence_case_ids), r, miss, st))
    for n, r, miss, st in sorted(rows, key=lambda t: -t[0]):
        print("%-26s %-30s %-18s x%d  [%s]"
              % (r.notes[:26], ",".join(miss)[:30] or "(none)", r.operation[:18], n, st))
    print()
    print("THE OPERATION each family needs, and whether a primitive exists:")
    for _n, r, _m, _s in sorted(rows, key=lambda t: -t[0]):
        print("  %-26s %-22s %s" % (r.notes[:26], r.operation[:22],
                                    r.candidate_primitive[:60] or "NO PRIMITIVE -- new capability"))
    if unmatched:
        print("\nUNMATCHED repair texts (no rule): %d" % len(unmatched))
        for rnd, t in unmatched[:3]:
            print("   [%s] %s" % (rnd, t))
    out = REPO / "rounds/H0_SESSION/affordance_ledger.json"
    out.write_text(json.dumps(led.as_dict(), indent=2) + "\n")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
