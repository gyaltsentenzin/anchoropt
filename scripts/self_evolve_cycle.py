#!/usr/bin/env python3
"""SELF-EVOLVE v0.1 -- one command for a full residual-boosting cycle.

    python scripts/self_evolve_cycle.py \
        --before results/se3ctl_vector_train --after results/se1inc_vector_train \
        --out rounds/SE1_A4_promoted --live

Starts from an incumbent's trajectories and runs:

    failed trajectories -> attribution -> residual ranking -> localize (l, phi)
      -> runtime-constrained feasible actions -> concrete candidate policies
      -> AnchorPolicyOpt evaluates/selects -> [promotion happened upstream]
      -> rerun incumbent -> re-attribute / re-mine residuals -> NEXT residual

The GPU rerun is done by scripts/cluster/runner_se1inc_vector_train.sh and passed in as `--after`;
everything else in the loop runs here. `--live` calls the real attributor and proposer; without it the
attribution step reuses a cached audit so the loop can be exercised without API calls.

WHAT THIS DEMONSTRATES, precisely: that after promoting a controller the system REACHES THE NEXT
OPTIMIZATION PROBLEM automatically. It does NOT claim the next anchor is accepted -- that is the next
round's measurement, and this script stops at the point where that round is fully specified.
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

import bfcl_runtime as runtime                                              # noqa: E402
from anchoropt.anchor import Action, IncisionPoint                          # noqa: E402
from anchoropt.learning.anchor_policy_opt import (                          # noqa: E402
    AnchorPolicyOpt, SearchSpaceProposal,
)
from anchoropt.learning.candidate_search import expressible_under           # noqa: E402
from anchoropt.learning.prestate_eligibility import FIRE_KEY, first_fire_index  # noqa: E402
from anchoropt.learning.residual_problem import build_residual_problems     # noqa: E402
from anchoropt.runtime import ResidualDiagnosis                             # noqa: E402
from anchoropt.telemetry import (                                          # noqa: E402
    CandidateEta, ResidualEntry, RoundLogger, RoundRecord, SearchFunnel, residual_shape,
)

FIXTURES = ("replay_anchors", "replay_exprs", "fixtures.replay_anchors", "fixtures.replay_exprs")


def assert_no_oracle_leak() -> None:
    leaked = sorted(m for m in sys.modules if any(f in m for f in FIXTURES))
    if leaked:
        raise SystemExit(f"ORACLE LEAK: anchor fixtures imported ({leaked})")


def load_run(d: pathlib.Path):
    """(correct_by_case, steps_by_case) for one run directory."""
    res = next(iter(sorted((d / "run").glob("eval_*results*.json"))), None)
    if res is None:
        raise SystemExit(f"no eval results under {d}/run")
    payload = json.load(open(res))
    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    correct = {str(r["id"]): bool(r.get("valid")) for r in rows if not r.get("is_prereq")}
    steps = {}
    for fp in sorted((d / "run" / "traj" / "query").glob("*.json")):
        try:
            ep = json.load(open(fp))
        except Exception:
            continue
        steps[str(ep.get("case_id"))] = ep.get("steps") or []
    return correct, steps


def real_steps(steps):
    """Model decisions only -- gate bookkeeping rows are not decisions."""
    return [s for s in steps if not (s.get("status") is None and not (s.get("decoded") or []))]


def case_facts(cid: str, steps, backend: str) -> dict:
    """One failed case as RUNTIME FACTS. No label, no error class, no anchor hint.

    Same shape live_smoke_bfcl.as_attributor_input uses, derived from trajectories instead of the
    ledger so it works on any rerun.
    """
    rs = real_steps(steps)
    calls = [str(x) for s in rs for x in (s.get("decoded") or [])]
    reads = [c for c in calls if "retrieve" in c or "search" in c]
    writes = [c for c in calls if "add" in c or "insert" in c or "write" in c]
    results = [s.get("result") for s in rs if s.get("result") is not None]
    return {"case_id": cid, "backend": backend, "runtime_facts": {
        "steps": len(rs),
        "read_calls": len(reads),
        "write_calls": len(writes),
        "results_returned": len(results),
        "made_no_tool_call": len(calls) == 0,
        "intervened": any(s.get(FIRE_KEY) for s in steps),
        "outcome": "the episode produced a final answer that was scored incorrect",
    }}


def to_diagnoses(raw) -> list[ResidualDiagnosis]:
    """Validate raw attributor records through the EXISTING seam, not a second parser.

    `proposal_seams.ingest_attribution` owns this: it requires `failure_mechanism` (not
    `mechanism` -- my first version of this function looked for the wrong key and would have
    silently dropped every live diagnosis) and it REJECTS a record that names a locus, signal,
    action or parameter. Re-implementing the parse here would have bypassed that guard.
    """
    from anchoropt.learning.proposal_seams import AttributionSchemaError, ingest_attribution
    out = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        try:
            out.append(ingest_attribution(r, provider="claude-attributor-v1"))
        except AttributionSchemaError as exc:
            print(f"    diagnosis REJECTED by the schema: {exc}")
    return out


def cached_diagnoses(path: pathlib.Path) -> list[ResidualDiagnosis]:
    out = []
    for line in open(path):
        rec = json.loads(line)
        if rec.get("kind") != "diagnosis_accepted":
            continue
        raw = rec.get("raw") or {}
        out.append(ResidualDiagnosis(
            case_id=rec["case_id"], mechanism=rec["mechanism"],
            evidence=str(raw.get("evidence", "")),
            consequential_decision=rec["consequential_decision"],
            proposed_behavior_change=str(raw.get("proposed_behavior_change", "")),
            provider="claude-attributor-v1",
            metadata={"causal_region": rec.get("causal_region")}))
    return out


def rank(diagnoses):
    return build_residual_problems(
        diagnoses, expressible=lambda d: expressible_under(d, runtime=runtime)[0])


def entries(problems):
    return tuple(ResidualEntry(rank=p.rank, key=p.key, support=p.support,
                               expressible=p.expressible) for p in problems)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=pathlib.Path, required=True)
    ap.add_argument("--after", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--backend", default="vector")
    ap.add_argument("--live", action="store_true", help="call the real attributor/proposer")
    ap.add_argument("--before-audit", type=pathlib.Path,
                    default=REPO / "results/selfevolve_r1/audit.jsonl")
    ap.add_argument("--round-id", default="SE1")
    ap.add_argument("--incumbent-id", default="native+A4prime")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    W = 100
    log = RoundLogger(a.out / "round_log.jsonl")

    print("=" * W)
    print("SELF-EVOLVE v0.1  ·  full residual-boosting cycle")
    print("=" * W)

    # ---------------------------------------------------------------- 3. score
    bc, bs = load_run(a.before)
    ac, as_ = load_run(a.after)
    common = sorted(set(bc) & set(ac))
    b_ok, a_ok = sum(bc[c] for c in common), sum(ac[c] for c in common)
    gains = [c for c in common if ac[c] and not bc[c]]
    losses = [c for c in common if bc[c] and not ac[c]]
    fired = [c for c in common if first_fire_index(as_.get(c, [])) is not None]
    print(f"\n[3] SCORE THE PROMOTED INCUMBENT  (n={len(common)})")
    print(f"  before (native)      {b_ok}/{len(common)} = {100*b_ok/len(common):.2f}%")
    print(f"  after  (native+A4')  {a_ok}/{len(common)} = {100*a_ok/len(common):.2f}%   "
          f"delta {100*(a_ok-b_ok)/len(common):+.2f} pp   +{len(gains)}/-{len(losses)}")
    print(f"  firings: {len(fired)}")

    # ---------------------------------------------------------------- 4. re-attribute
    fails_after = [c for c in common if not ac[c]]
    print(f"\n[4] RE-ATTRIBUTE the remaining {len(fails_after)} failures")
    if a.live:
        import client as roles
        cases = [case_facts(c, as_.get(c, []), a.backend) for c in fails_after]
        raw, recs = roles.attribute_batched(cases)
        errs = [r.error for r in recs if r.error]
        print(f"  attributor: {len(raw)} diagnoses from {len(cases)} cases, "
              f"{len(recs)} calls, {len(errs)} errors")
        for e in errs[:3]:
            print(f"    ERROR: {e}")
        after_diag = to_diagnoses(raw)
        json.dump(list(raw), open(a.out / "diagnoses_after_raw.json", "w"),
                  indent=2, default=str)
        json.dump([{"case_id": d.case_id, "failure_mechanism": d.mechanism,
                    "consequential_decision": d.consequential_decision,
                    "evidence": d.evidence,
                    "causal_region": (d.metadata or {}).get("causal_region") or {"phase": ""},
                    "proposed_behavior_change": d.proposed_behavior_change}
                   for d in after_diag], open(a.out / "diagnoses_after.json", "w"), indent=2)
    else:
        print("  --live not set: reusing the cached AFTER diagnoses if present")
        p = a.out / "diagnoses_after.json"
        after_diag = to_diagnoses(json.load(open(p))) if p.exists() else []
    print(f"  usable diagnoses: {len(after_diag)}")

    # ---------------------------------------------------------------- 5/6. re-mine + re-rank
    before_diag = cached_diagnoses(a.before_audit)
    before_problems = rank(before_diag)
    after_problems = rank(after_diag)
    print(f"\n[5]/[6] RE-MINE AND RE-RANK  (metric unchanged: support, then key)")
    print(f"  BEFORE  shape {residual_shape(entries(before_problems))}")
    for p in before_problems:
        print(f"    R{p.rank} support={p.support} cov={100*p.coverage:5.1f}% "
              f"{'expressible' if p.expressible else 'SIGNAL_BLOCKED'}  {p.key[:64]}")
    print(f"  AFTER   shape {residual_shape(entries(after_problems))}")
    for p in after_problems:
        print(f"    R{p.rank} support={p.support} cov={100*p.coverage:5.1f}% "
              f"{'expressible' if p.expressible else 'SIGNAL_BLOCKED'}  {p.key[:64]}")

    # ---------------------------------------------------------------- 7. compare
    bk = {p.key for p in before_problems}
    akey = {p.key for p in after_problems}
    print(f"\n[7] RESIDUAL SHIFT")
    print(f"  before -> after shape : {residual_shape(entries(before_problems))} -> "
          f"{residual_shape(entries(after_problems))}")
    print(f"  DISAPPEARED ({len(bk-akey)}):")
    for k in sorted(bk - akey):
        print(f"    - {k[:88]}")
    print(f"  PERSISTED ({len(bk & akey)}):")
    for k in sorted(bk & akey):
        print(f"    = {k[:88]}")
    print(f"  NEW ({len(akey-bk)}):")
    for k in sorted(akey - bk):
        print(f"    + {k[:88]}")

    # ---------------------------------------------------------------- 8. next round
    print(f"\n[8] NEXT OPTIMIZATION PROBLEM -- selected automatically")
    nxt = next((p for p in after_problems if p.expressible), None)
    record_kw = {}
    if nxt is None:
        print("  no expressible residual remains: the loop terminates here, which is a RESULT")
        print("  (POLICY_EXHAUSTED_CURRENT_SPACE scoped to this Phi/host), not a failure.")
    else:
        print(f"  R1' = support {nxt.support}, coverage {100*nxt.coverage:.1f}%")
        print(f"        {nxt.key[:88]}")
        print(f"  cases: {[d.case_id for d in nxt.scoped_diagnoses()][:6]}")
        funnel = SearchFunnel()
        if a.live:
            import client as roles
            raw, call = roles.propose_semantic(nxt.scoped_diagnoses(), runtime=runtime,
                                               host=runtime.HOST, max_proposals=4)
            if call.error:
                print(f"  proposer FAILED: {call.error}")
                raw = []
            print(f"  proposer returned {len(raw)} proposal(s)")
            json.dump(raw, open(a.out / "proposals_next.json", "w"), indent=2, default=str)
        else:
            p = a.out / "proposals_next.json"
            raw = json.load(open(p)) if p.exists() else []
            print(f"  cached proposals: {len(raw)}")
        opt = AnchorPolicyOpt(runtime=runtime, host=runtime.HOST)
        cands = []
        for r in raw:
            # INGEST VIA THE CANONICAL RULE, not a second copy of it. `propose_semantic` emits
            # `action_preference` (provenance only) and NO `action_set`; my first version looked for
            # action_set/action, found neither, and silently produced zero candidates from three
            # perfectly good proposals -- the funnel read proposed=0. The established rule
            # (scripts/live_policy_recovery.to_search_space) is that an absent or single-action set
            # WIDENS U-hat to every host-executable action at that boundary, because the proposer
            # naming one action must not narrow the search to it.
            try:
                b = IncisionPoint(str(r.get("locus") or r.get("boundary")).strip().lower())
                phi = str(r.get("signal") or r.get("signal_name") or "").strip()
                names = r.get("action_set") or ([r.get("action")] if r.get("action") else [])
                acts = []
                for n in names:
                    try:
                        av = Action(str(n).strip().lower())
                    except ValueError:
                        continue
                    if av is not Action.NOOP:
                        acts.append(av)
                if len(acts) <= 1:
                    acts = sorted((x for x in runtime.HOST.executable_actions(b)
                                   if x is not Action.NOOP), key=lambda x: x.value)
                aset = tuple(acts)
            except Exception as exc:
                print(f"    proposal rejected at ingest: {type(exc).__name__}: {exc}")
                continue
            if not aset:
                print("    proposal carried no executable action at that boundary")
                continue
            funnel.record_stage("proposed", len(aset))
            funnel.record_stage("localized", len(aset))
            sp = SearchSpaceProposal(boundary=b, signal=phi, action_set=aset,
                                     diagnosis_case_ids=tuple(d.case_id for d in nxt.scoped_diagnoses()),
                                     rationale=str(r.get("rationale", ""))[:200])
            arms, rejected = opt.build_arms(sp)
            funnel.observe_build(arms, rejected)
            print(f"\n    l={b.value}  phi={phi}  U-hat={[x.value for x in aset]}")
            print(f"      U_H(l) = {sorted(x.value for x in runtime.feasible_actions(b))}")
            print(f"      MATERIALIZABLE ARMS: {len(arms)}")
            for arm in arms:
                print(f"        + {arm.instantiated.label}  eta={ {k: str(v)[:44] for k, v in arm.eta.items()} }")
                cands.append(CandidateEta(variant=arm.instantiated.label, eta=dict(arm.eta),
                                          materializable=True))
            for rj in rejected:
                print(f"        - {rj.action.value}: {rj.reason_code}")
                cands.append(CandidateEta(variant=f"{rj.action.value}", eta={},
                                          materializable=False, rejection_reason=rj.reason_code))
        rep = funnel.report()
        print(f"\n  SEARCH FUNNEL: " + " -> ".join(f"{k}={v}" for k, v in rep.stages.items()))
        print(f"  rejections: {dict(sorted(k for k in rep.rejections.items() if k[1]))}")
        record_kw = {"selected_residual": nxt.key, "candidate_etas": tuple(cands),
                     "funnel": rep.as_dict()}
        json.dump({"next_residual": nxt.key, "support": nxt.support,
                   "cases": [d.case_id for d in nxt.scoped_diagnoses()],
                   "materializable_candidates": [c.variant for c in cands if c.materializable],
                   "funnel": rep.as_dict()},
                  open(a.out / "next_round.json", "w"), indent=2)

    rec = RoundRecord(
        round_id=a.round_id, incumbent_id=a.incumbent_id,
        residuals_before=entries(before_problems), residuals_after=entries(after_problems),
        selected_locus="post_generation_pre_exec", selected_signal="no_tool_call_at_all",
        feasible_actions=tuple(sorted(x.value for x in runtime.feasible_actions(
            IncisionPoint("post_generation_pre_exec")))),
        winner="se3a2:search_other_container",
        objective_delta=100 * (a_ok - b_ok) / len(common),
        notes="Self-Evolve v0.1 first full residual-boosting cycle", **record_kw)
    log.append(rec)
    assert_no_oracle_leak()
    print(f"\n{'=' * W}")
    print(f"CYCLE COMPLETE. round log -> {a.out / 'round_log.jsonl'}")
    print(f"  residual shift: {residual_shape(entries(before_problems))} -> "
          f"{residual_shape(entries(after_problems))}")
    print(f"  next problem  : {'(none expressible)' if nxt is None else nxt.key[:70]}")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
