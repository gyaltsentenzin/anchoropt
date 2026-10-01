#!/usr/bin/env python3
"""DRY RUN: expand one proposer suggestion into the complete feasible policy space. No GPU.

Validates the INNER optimizer before the outer loop is reorganized around it. Shows, for the
low-similarity example that R1/R2 measured:

  1. the action families the proposer suggested (U-hat)
  2. the action-specific parameters eta_mu each family REQUIRES, per its contract
  3. every grounded destination the runtime derives -- not one hardcoded choice
  4. the resulting action x eta x theta_phi arms
  5. infeasible cells with the exact missing requirement
  6. that no historical A9 destination, threshold or config entered the search

Nothing is evaluated. The point is to see the space R1 never enumerated.
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
from anchoropt.learning.action_contract import CONTRACTS, operators_of   # noqa: E402
from anchoropt.learning.anchor_policy_opt import (                       # noqa: E402
    AnchorPolicyOpt, FrozenIncumbent, SearchSpaceProposal,
)
from anchoropt.learning.policy_class import ThetaResult, quantile_grid   # noqa: E402

SIG = "retrieval_similarity_below_threshold"
OBS = json.load(open(REPO / "results/selfevolve_r1/observed_similarity_control.json"))["per_call"]


def main() -> int:
    W = 100
    print("=" * W)
    print("DRY RUN -- AnchorPolicyOpt on the low-similarity example (no GPU, nothing evaluated)")
    print("=" * W)

    # ---- 1. what the proposer suggests: a locus, a condition, a SMALL action set --------------
    proposal = SearchSpaceProposal(
        boundary=IncisionPoint.POST_EXECUTION, signal=SIG,
        action_set=(Action.REPROMPT, Action.REROUTE),        # U-hat: high-recall, not a choice
        rationale="a core retrieve returned nothing well-matched and the model answered anyway",
        diagnosis_case_ids=("memory_vector_10-customer-10",),
        preferred_action="reprompt",                          # R1's one-shot pick -- PROVENANCE
        theta_hint={"below": 0.75})                           # R1's one-shot number -- PROVENANCE

    print(f"\n[1] PROPOSER SUGGESTED")
    print(f"  l      = {proposal.boundary.value}")
    print(f"  phi    = {proposal.signal}")
    print(f"  U-hat  = {[a.value for a in proposal.action_set]}")
    print(f"  PROVENANCE ONLY: preferred_action={proposal.preferred_action!r}  "
          f"theta_hint={dict(proposal.theta_hint)}")

    # ---- 2. contracts: what each family REQUIRES ---------------------------------------------
    print(f"\n[2] ACTION-SPECIFIC PARAMETERS eta_mu REQUIRED PER FAMILY")
    for action in proposal.action_set:
        for op in operators_of(action):
            c = CONTRACTS[op]
            print(f"  {action.value:9s} -> operator {op.value:11s} requires {list(c.required)}")
            print(f"  {'':9s}    grounded by runtime.{c.grounded_by}()   "
                  f"infeasible code: {c.infeasible_code}")

    # ---- 3. every grounding the runtime derives ----------------------------------------------
    print(f"\n[3] GROUNDINGS THE RUNTIME DERIVES (enumerated, not chosen)")
    for hook in ("ground_reprompt", "ground_substitute_destinations", "ground_transforms",
                 "ground_suppress"):
        got = getattr(runtime, hook)(SIG, proposal.boundary) or []
        print(f"  runtime.{hook}(): {len(got)} grounding(s)")
        for g in got:
            eta = {k: (str(v)[:44] + "..." if len(str(v)) > 44 else v) for k, v in g["eta"].items()}
            print(f"     {g['variant']:44s} {eta}")

    # ---- 4/5. the arms, and what was infeasible ---------------------------------------------
    opt = AnchorPolicyOpt(runtime=runtime, host=runtime.HOST)
    arms, rejected = opt.build_arms(proposal)

    theta_grid = quantile_grid(OBS, n=9, domain=runtime.parameter_domains(SIG)[0])
    print(f"\n[4] ARMS = action x eta_mu x theta_phi")
    print(f"  theta_phi grid (9 quantiles of {len(OBS)} OBSERVED control-arm similarities): "
          f"{list(theta_grid)}")
    print(f"  arms instantiated: {len(arms)}    "
          f"paired evaluations implied: {len(arms)} x {len(theta_grid)} = "
          f"{len(arms) * len(theta_grid)}")
    for a in arms:
        print(f"    {a.label}")
        print(f"        eta_mu = { {k: (str(v)[:52]) for k, v in a.eta.items()} }")
        print(f"        theta_phi searched over {len(theta_grid)} values   "
              f"policy_class={a.policy_class.value}  llm_preferred={a.was_preferred}")

    print(f"\n[5] INFEASIBLE CELLS (recorded with the missing requirement)")
    if not rejected:
        print("  (none)")
    for r in rejected:
        op = f"[{r.operator.value}]" if r.operator else ""
        miss = f"  missing={list(r.missing)}" if r.missing else ""
        print(f"  {r.action.value}{op}: {r.reason_code}{miss}")
        print(f"      {r.detail[:92]}")

    # ---- 6. no A9 leakage --------------------------------------------------------------------
    print(f"\n[6] A9 ISOLATION CHECK")
    blob = json.dumps([{"label": a.label, "eta": {k: str(v) for k, v in a.eta.items()}}
                       for a in arms], sort_keys=True).lower()
    for token in ("0.30", "0.3,", "xcm", "cross_container", "tie_break", "a9"):
        assert token not in blob, f"A9 specific leaked into the arms: {token!r}"
    import bfcl_signals as sig
    assert dict(sig.SIGNAL_PROBE_PARAMS) == {} and dict(sig.SIGNAL_ALIASES) == {}
    assert dict(runtime.REROUTE_DESTINATIONS) == {}
    print("  no A9 destination, threshold, alias or config in the arms or the runtime  -> CLEAN")
    print(f"  theta grid is data-derived; 0.30 in grid: {0.30 in theta_grid} "
          f"(present only if the observed distribution puts it there)")

    print(f"\n{'=' * W}")
    print(f"R1 evaluated 1 controller at 1 theta. This space is {len(arms)} arms x "
          f"{len(theta_grid)} theta = {len(arms) * len(theta_grid)} counterfactuals.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
