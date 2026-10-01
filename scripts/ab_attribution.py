#!/usr/bin/env python3
"""A/B the two attributors on the SAME residual. Attribution is the only variable.

    python scripts/ab_attribution.py --cases 8 --out /tmp/ab_attr

WHAT IS HELD FIXED
------------------
Same residual cases, same proposer (`client.propose`), same Phi, same runtime/host feasibility
checks, same validation machinery, same top-K rule. No Granite evaluation is launched. The only
difference between the two columns is which attributor produced the diagnoses -- which is what makes
this an attribution A/B and not a comparison of two systems.

WHAT IS MEASURED
----------------
  * diagnoses accepted / rejected by each schema
  * distinct consequential decisions (V1's grouping) vs canonical signatures (V2's)
  * fragmentation: does canonicalization merge semantically identical diagnoses?
  * expressible vs not_expressible_under_current_phi
  * actionable vs recovered_friction (V2 only -- V1 cannot express it)
  * the phase verdict each residual representation produces
  * the resulting proposer candidates, and how many survive AnchorOpt validation

The fixtures are never imported: `assert_no_fixture_leak` runs at the end, as in the smoke script.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
sys.path.insert(0, str(REPO / "integrations" / "claude_roles"))
sys.path.insert(0, str(REPO / "scripts"))

import attributor_v2 as v2                                              # noqa: E402
import bfcl_runtime as runtime                                          # noqa: E402
import client as roles                                                  # noqa: E402
from anchoropt.learning.block_loop import residual_phase                 # noqa: E402
from anchoropt.learning.candidate_search import expressible_under        # noqa: E402
from anchoropt.learning.proposal_seams import (                          # noqa: E402
    AttributionSchemaError, ProposalSchemaError, ingest_attribution, ingest_proposal,
    rank_by_support, validate_proposal,
)
from live_smoke_bfcl import as_attributor_input, assert_no_fixture_leak, load_residual  # noqa: E402


def phi_expressible(text: str) -> bool:
    """Is a mechanism expressible under the CURRENT Phi? Uses the same test the search uses."""
    from anchoropt.runtime import ResidualDiagnosis
    probe = ResidualDiagnosis(case_id="probe", mechanism=text, evidence=text,
                             consequential_decision=text, proposed_behavior_change="")
    ok, _why = expressible_under(probe, runtime=runtime)
    return ok


def run_proposer(diagnoses, log, tag: str):
    """Identical proposer call for both arms. Returns (legal, declined, call_record)."""
    ranked = rank_by_support(diagnoses)
    raw, rec = roles.propose(ranked, runtime=runtime, host=runtime.HOST,
                             allow_new_signal=False, max_proposals=4)
    log.write("role_call", role=f"proposer:{tag}", error=rec.error, raw=rec.raw, usage=rec.usage)
    legal, declined = [], []
    if rec.error:
        return legal, declined, rec
    fields = runtime.all_fields()
    for r in raw:
        try:
            cand = ingest_proposal(r, allow_new_signal=False,
                                   declared_signals=runtime.declared_signals())
        except ProposalSchemaError as exc:
            declined.append((json.dumps(r)[:70], f"schema: {exc}"))
            continue
        compiled, why = validate_proposal(cand, runtime=runtime, host=runtime.HOST, fields=fields)
        what = f"{cand.boundary.value}/{cand.signal_name}/{cand.action.value}"
        (legal if why == "legal" else declined).append((cand, compiled) if why == "legal"
                                                       else (what, why))
    return legal, declined, rec


class Log:
    def __init__(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a")
        self.path = path

    def write(self, kind, **payload):
        self.fh.write(json.dumps({"t": time.time(), "kind": kind, **payload}, default=str) + "\n")
        self.fh.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/ab_attr"))
    a = ap.parse_args()

    log = Log(a.out / "ab_audit.jsonl")
    rows = load_residual(a.cases)
    cases = [as_attributor_input(r) for r in rows]
    log.write("residual", n=len(cases), case_ids=[c["case_id"] for c in cases])

    print("=" * 104)
    print(f"ATTRIBUTION A/B  ·  {len(cases)} identical residual cases  ·  downstream held fixed")
    print("=" * 104)

    # ------------------------------------------------------------------ V1
    print("\n### V1 (current attributor)")
    raw1, recs1 = roles.attribute_batched(cases)
    for r in recs1:
        log.write("role_call", role="attributor_v1", error=r.error, raw=r.raw, usage=r.usage)
    d1, rej1 = [], []
    for r in raw1:
        try:
            d1.append(ingest_attribution(r, provider="claude-attributor-v1"))
        except AttributionSchemaError as exc:
            rej1.append((r.get("case_id"), str(exc)))
    print(f"  accepted {len(d1)} / rejected {len(rej1)}")
    groups1 = collections.Counter(
        " ".join(d.consequential_decision.lower().split()) for d in d1)
    print(f"  distinct consequential decisions: {len(groups1)}")
    for k, v in groups1.most_common():
        print(f"    {v:3d}  {k[:86]}")
    exp1 = [d for d in d1 if phi_expressible(d.mechanism)]
    print(f"  expressible under current Phi: {len(exp1)}/{len(d1)}")
    ph1, rs1 = residual_phase(rank_by_support(d1),
                              expressible=lambda d: phi_expressible(d.mechanism))
    print(f"  phase verdict: {ph1}")

    # ------------------------------------------------------------------ V2
    print("\n### V2 (Self-Harness-inspired attributor)")
    raw2, recs2 = v2.attribute_v2(cases)
    for r in recs2:
        log.write("role_call", role="attributor_v2", error=r.error, raw=r.raw, usage=r.usage)
    d2, rej2 = [], []
    for r in raw2:
        try:
            d2.append(v2.ingest_attribution_v2(r, provider="claude-attributor-v2",
                                               expressible=phi_expressible))
        except v2.AttributionV2SchemaError as exc:
            rej2.append((r.get("case_id"), str(exc)))
            log.write("v2_rejected", case_id=r.get("case_id"), reason=str(exc), raw=r)
    print(f"  accepted {len(d2)} / rejected {len(rej2)}")
    for cid, why in rej2[:4]:
        print(f"    REJECTED {cid}: {why[:100]}")

    sigs = v2.group_by_signature(d2)
    print(f"  canonical signatures (terminal_cause/criticality/agent_mechanism): {len(sigs)}")
    for k, v in sorted(sigs.items(), key=lambda kv: -len(kv[1])):
        print(f"    {len(v):3d}  {k[:86]}")
    groups2raw = collections.Counter(
        " ".join(d.consequential_decision.lower().split()) for d in d2)
    print(f"  distinct consequential decisions (raw prose): {len(groups2raw)}")
    print(f"  -> canonicalization merged {len(groups2raw)} prose groups into {len(sigs)} signatures")

    crit = collections.Counter((d.metadata or {}).get("criticality") for d in d2)
    print(f"  criticality: {dict(crit)}")
    act = v2.actionable(d2)
    print(f"  actionable (root_cause/contributor): {len(act)}/{len(d2)}")
    qual = collections.Counter((d.metadata or {}).get("evidence_quality") for d in d2)
    print(f"  evidence quality (DERIVED, not asserted): {dict(qual)}")
    nonexp = [d for d in d2 if (d.metadata or {}).get("not_expressible_under_current_phi")]
    print(f"  not_expressible_under_current_phi: {len(nonexp)}/{len(d2)}  (retained as SIGNAL backlog)")
    for d in nonexp[:3]:
        print(f"    · {d.case_id}: {d.mechanism[:78]}")
    links = [d for d in d2 if (d.metadata or {}).get("upstream_link")]
    print(f"  upstream linkage asserted: {len(links)}/{len(d2)}")
    ph2, rs2 = residual_phase(rank_by_support(act or d2),
                              expressible=lambda d: not (d.metadata or {}).get(
                                  "not_expressible_under_current_phi", False))
    print(f"  phase verdict: {ph2}")

    # ------------------------------------------------------------------ proposer, both arms
    print("\n### PROPOSER on each representation (same proposer, same Phi, same U_H)")
    legal1, dec1, _ = run_proposer(d1, log, "v1")
    print(f"  from V1: {len(legal1)} legal / {len(dec1)} declined")
    for cand, _c in legal1:
        print(f"    + {cand.boundary.value} / {cand.signal_name} / {cand.action.value}"
              f"   ({len(cand.diagnosis_case_ids)} cases)")
    for what, why in dec1:
        print(f"    - {what}  ::  {str(why)[:88]}")

    legal2, dec2, _ = run_proposer(act or d2, log, "v2")
    print(f"  from V2: {len(legal2)} legal / {len(dec2)} declined")
    for cand, _c in legal2:
        print(f"    + {cand.boundary.value} / {cand.signal_name} / {cand.action.value}"
              f"   ({len(cand.diagnosis_case_ids)} cases)")
    for what, why in dec2:
        print(f"    - {what}  ::  {str(why)[:88]}")

    assert_no_fixture_leak()

    same = {(c.boundary.value, c.signal_name, c.action.value) for c, _ in legal1} == \
           {(c.boundary.value, c.signal_name, c.action.value) for c, _ in legal2}
    summary = {
        "cases": len(cases),
        "v1": {"accepted": len(d1), "rejected": len(rej1),
               "distinct_decisions": len(groups1), "expressible": len(exp1),
               "phase": ph1, "legal_candidates": len(legal1)},
        "v2": {"accepted": len(d2), "rejected": len(rej2),
               "prose_groups": len(groups2raw), "canonical_signatures": len(sigs),
               "criticality": dict(crit), "actionable": len(act),
               "evidence_quality": dict(qual), "not_expressible": len(nonexp),
               "upstream_links": len(links), "phase": ph2,
               "legal_candidates": len(legal2)},
        "same_controllers_proposed": same,
        "granite_evaluation_launched": False,
    }
    (a.out / "ab_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print("\n" + "=" * 104)
    print(f"V1: {len(groups1)} decision groups -> {len(legal1)} legal   |   "
          f"V2: {len(sigs)} signatures ({len(act)}/{len(d2)} actionable) -> {len(legal2)} legal")
    print(f"same controllers proposed: {same}")
    print(f"no Granite evaluation launched.  audit: {log.path}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
