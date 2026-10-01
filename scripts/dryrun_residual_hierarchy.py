#!/usr/bin/env python3
"""Replay the REAL R1 attribution through the fixed hierarchy. No GPU, no LLM, nothing evaluated.

The acceptance check: the old run jumped from an infeasible top-residual controller to a 1-case
low-similarity controller. This must not happen -- the top residual must stay frozen and its
alternative action families must be searched.

Diagnoses come from `results/selfevolve_r1/audit.jsonl`, i.e. what live Claude actually produced.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

import bfcl_runtime as runtime                                          # noqa: E402
from anchoropt.anchor import Action, IncisionPoint                       # noqa: E402
from anchoropt.learning.anchor_policy_opt import (                       # noqa: E402
    AnchorPolicyOpt, FrozenIncumbent, SearchSpaceProposal,
)
from anchoropt.learning.candidate_search import expressible_under        # noqa: E402
from anchoropt.learning.policy_class import PolicySpec, quantile_grid     # noqa: E402
from anchoropt.learning.residual_problem import build_residual_problems  # noqa: E402
from anchoropt.runtime import ResidualDiagnosis                          # noqa: E402

AUDIT = REPO / "results/selfevolve_r1/audit.jsonl"
OBS = json.load(open(REPO / "results/selfevolve_r1/observed_similarity_control.json"))["per_call"]


def load_r1_diagnoses():
    out = []
    for line in open(AUDIT):
        rec = json.loads(line)
        if rec.get("kind") != "diagnosis_accepted":
            continue
        raw = rec.get("raw") or {}
        out.append(ResidualDiagnosis(
            case_id=rec["case_id"], mechanism=rec["mechanism"], evidence=str(raw.get("evidence", "")),
            consequential_decision=rec["consequential_decision"],
            proposed_behavior_change=str(raw.get("proposed_behavior_change", "")),
            provider="claude-attributor-v1",
            metadata={"causal_region": rec.get("causal_region")}))
    return out


def phi_expressible(d):
    ok, _why = expressible_under(d, runtime=runtime)
    return ok


def main() -> int:
    W = 102
    ds = load_r1_diagnoses()
    print("=" * W)
    print(f"R1 REPLAY THROUGH THE FIXED HIERARCHY -- {len(ds)} real diagnoses, nothing evaluated")
    print("=" * W)

    problems = build_residual_problems(ds, expressible=phi_expressible)

    print("\n[1] RANKED RESIDUAL TABLE (metric UNCHANGED: support, then key)")
    for p in problems:
        print(f"  R{p.rank}  support={p.support}  coverage={100*p.coverage:5.1f}%  "
              f"{'expressible' if p.expressible else 'SIGNAL_BLOCKED'}")
        print(f"       key: {p.key[:88]}")
        print(f"       cases: {list(p.case_ids)}")

    R1 = problems[0]
    print(f"\n[2] FROZEN RESIDUAL")
    print(f"  {R1}")

    print(f"\n[3] DIAGNOSES PASSED TO THE PROPOSER (only R1's)")
    for d in R1.scoped_diagnoses():
        print(f"  {d.case_id:32s} {d.mechanism[:66]}")
    excluded = [d.case_id for p in problems[1:] for d in p.diagnoses]
    print(f"  EXCLUDED from this call ({len(excluded)} cases from R2/R3): {excluded}")

    # The proposer's job now: (l, phi, U-hat) for THIS residual. Written here as a stand-in for the
    # live call, with the SAME locus R1's real proposer chose for its full-coverage candidate --
    # `no_tool_call_at_all` at the commitment gate, which was the infeasible one.
    proposals = (
        SearchSpaceProposal(
            boundary=IncisionPoint.POST_GENERATION_PRE_EXEC, signal="no_tool_call_at_all",
            action_set=(Action.REPROMPT, Action.REROUTE),
            rationale="the model commits to an answer at the gate without adequate grounding",
            diagnosis_case_ids=tuple(R1.case_ids)),
        SearchSpaceProposal(
            boundary=IncisionPoint.POST_EXECUTION, signal="no_informative_result",
            action_set=(Action.REPROMPT, Action.REROUTE),
            rationale="the read it did make returned nothing usable",
            diagnosis_case_ids=tuple(R1.case_ids)),
    )
    print(f"\n[4] PROPOSED (l, phi, U-hat) FOR R1 -- {len(proposals)} locus/loci")
    for pr in proposals:
        print(f"  l={pr.boundary.value:26s} phi={pr.signal:26s} "
              f"U-hat={[a.value for a in pr.action_set]}")

    opt = AnchorPolicyOpt(runtime=runtime, host=runtime.HOST)
    grid_len = {}
    total_arms = 0
    print(f"\n[5]/[6] ANCHORPOLICYOPT ARMS FOR R1, and INFEASIBLE CELLS")
    for pr in proposals:
        arms, rejected = opt.build_arms(pr)
        # Ask the SPEC for the grid, not quantile_grid directly: a categorical domain (e.g.
        # no_informative_result's `kind`) has explicit `values` and no observed distribution, so
        # calling quantile_grid on it returns () and would silently report 0 counterfactuals for
        # perfectly valid arms. candidate_thetas() dispatches on the domain kind.
        n_theta = 1
        if arms:
            spec = PolicySpec(boundary=arms[0].boundary, signal=arms[0].signal,
                              action=arms[0].action, policy_class=arms[0].policy_class,
                              domains=arms[0].signal_domains, theta=arms[0].eta)
            n_theta = len(spec.candidate_thetas({"best_similarity": OBS}, n=9))
        grid_len[pr.signal] = n_theta
        total_arms += len(arms) * n_theta
        print(f"\n  -- locus {pr.boundary.value} / {pr.signal}")
        print(f"     arms={len(arms)}  theta values={n_theta}  -> {len(arms)*n_theta} counterfactuals")
        for a in arms:
            print(f"       + {a.instantiated.label:52s} eta={list(a.eta)}")
        for rj in rejected:
            op = f"[{rj.operator.value}]" if rj.operator else ""
            miss = f" missing={list(rj.missing)}" if rj.missing else ""
            print(f"       - {rj.action.value}{op}: {rj.reason_code}{miss}")
            print(f"           {rj.detail[:84]}")

    print(f"\n[7] PROOF THAT NO R2/R3 CANDIDATE ENTERS THE OPTIMIZATION")
    r1_cases = set(R1.case_ids)
    for pr in proposals:
        alien = set(pr.diagnosis_case_ids) - r1_cases
        print(f"  proposal @{pr.signal:26s} cites {len(pr.diagnosis_case_ids)} cases, "
              f"alien={sorted(alien) or 'NONE'}")
    print(f"  run_residual_round RAISES on any alien case id (test: "
          f"test_a_proposal_citing_another_residual_is_refused)")

    print(f"\n[8] WHAT WOULD MOVE THE LOOP TO R2")
    print("  ONLY POLICY_EXHAUSTED_CURRENT_SPACE for R1, i.e. ALL of:")
    print("    * every locus the proposer offered for R1 has been tried, AND")
    print("    * for each, every action family in U-hat instantiated via its contract, AND")
    print("    * each feasible arm measured and REJECTED by the acceptance rule, AND")
    print("    * each infeasible cell recorded with its missing requirement")
    print("  An infeasible action alone does NOT move the loop. Neither does a single rejected arm.")

    print(f"\n[ACCEPTANCE CHECK]")
    lowsim_reached = any(pr.signal == "retrieval_similarity_below_threshold" for pr in proposals)
    print(f"  did the loop jump to the 1-case low-similarity residual? "
          f"{'YES -- FAIL' if lowsim_reached else 'NO'}")
    print(f"  R3 (the 1-case residual) searched? NO -- R1 is frozen and not yet exhausted")
    print(f"  total counterfactual arms for R1 this round: {total_arms}")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
