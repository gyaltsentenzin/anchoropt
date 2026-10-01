#!/usr/bin/env python3
"""First prototype of the history-aware LLM optimizer role, tested on one case.

    residual + synthesized candidates + executor capabilities + prior history
      -> LLM proposes the closest REALIZABLE policy, with a rationale
      -> verify_projection decides equivalence DETERMINISTICALLY on trigger sets
      -> (if accepted) paired evaluation decides promote/reject

The model maps a semantic hypothesis onto the realizable policy class. It does not certify success and
it does not declare equivalence; the tolerance was predeclared in anchoropt/learning/realization.py
before this script was written, and the comparison is over observed states.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
sys.path.insert(0, str(REPO / "integrations" / "claude_roles"))

import bfcl_runtime as runtime                                          # noqa: E402
from anchoropt.evolve_memory import ExperimentLedger, experience_block  # noqa: E402
from anchoropt.learning.realization import (                            # noqa: E402
    MAX_TRIGGER_DISAGREEMENT, verify_projection,
)
from anchoropt.learning.signal_grammar import Atom, Conjunction         # noqa: E402

SYSTEM = """\
You map a SEMANTIC POLICY HYPOTHESIS onto the policy class a runtime can actually realize.

You are given: a residual failure mode, the representation gap that blocked it, candidate predicates a
constrained grammar synthesized from observed data, what the host's executors can run, and the
parameters those executors FIX and therefore cannot be asked to vary.

Your job is to propose the CLOSEST REALIZABLE policy and say why. You may conclude that a synthesized
parameter is not settable on this host and that an existing controller is the nearest realization.

You must NOT claim the realization is equivalent to what it approximates. Equivalence is decided
downstream by comparing trigger sets on observed states against a tolerance fixed before you were
asked. State your reasoning; do not state a verdict.

Reply with JSON only:
{"semantic_hypothesis": "...", "proposed_realization": {"signal_form": "...", "threshold": <number>,
 "action": "...", "operator": "..."}, "rationale": "...", "what_would_make_this_wrong": "..."}
"""


def main() -> int:
    out_dir = REPO / "rounds/REC_A9"
    rec = json.load(open(out_dir / "recovery.json"))
    sel = json.load(open(out_dir / "selection.json"))
    led = ExperimentLedger(REPO / "results/self_evolve/memory")

    spec = runtime.EXECUTORS[("post_execution", "reroute")]
    fixed = dict(spec.get("fixed") or {})
    user = f"""RESIDUAL (failure mode being repaired)
  {sel['step1_clause']}
  support {rec['clause_residuals'][0]['support']} cases

REPRESENTATION GAP
  A declared signal observes part of this condition (a weak returned score). Nothing declared observes
  the other part (a store in scope was never consulted this episode), so the residual was SIGNAL_BLOCKED
  and a predicate was synthesized over declared typed fields.

SYNTHESIZED CANDIDATE PREDICATES (thresholds derived from observed values, not chosen)
{chr(10).join('  ' + s for s in sel['surviving_signals'])}

HOST EXECUTOR CAPABILITY at this boundary/action
  mechanism : {spec['operators']['transform']['detail'] if 'transform' in spec.get('operators', {}) else spec['detail']}
  read-merge: one additional read against an unconsulted store, results merged by score, returned
              payload replaced in place
  PARAMETERS THE EXECUTOR FIXES (cannot be varied by a policy): {fixed}

{experience_block(led, residual=sel['step1_clause'], locus='post_execution') or '(no prior experience)'}
QUESTION
  What is the closest policy this host can actually realize for this residual, and why?
"""
    (out_dir / "realization_prompt.txt").write_text(SYSTEM + "\n\n" + user)
    print("=" * 96)
    print("A9 REALIZATION PROBE  ·  LLM proposes; structure verifies")
    print("=" * 96)
    print(f"\npredeclared tolerance: {100*MAX_TRIGGER_DISAGREEMENT:.0f}% of the trigger-set union")
    print(f"prompt: {len(SYSTEM) + len(user)} chars -> {out_dir / 'realization_prompt.txt'}")

    import client as roles
    call = roles._call(SYSTEM, user, role="realizer")
    if call.error:
        print(f"\nproposer FAILED: {call.error}")
        return 1
    proposal = call.parsed if isinstance(call.parsed, dict) else {}
    json.dump(proposal, open(out_dir / "realization_proposal.json", "w"), indent=2)
    print(f"\n[1] LLM PROPOSAL  ({call.usage.get('input_tokens')} in / "
          f"{call.usage.get('output_tokens')} out)")
    print(f"  semantic hypothesis : {str(proposal.get('semantic_hypothesis'))[:150]}")
    pr = proposal.get("proposed_realization") or {}
    print(f"  proposed realization: signal={str(pr.get('signal_form'))[:70]}")
    print(f"                        threshold={pr.get('threshold')}  action={pr.get('action')} "
          f"operator={pr.get('operator')}")
    print(f"  rationale           : {str(proposal.get('rationale'))[:220]}")
    print(f"  self-stated risk    : {str(proposal.get('what_would_make_this_wrong'))[:180]}")

    # ------------------------------------------------------------------ deterministic verification
    import importlib.util
    s2 = importlib.util.spec_from_file_location("rr", REPO / "scripts/anchor_recovery_replay.py")
    rr = importlib.util.module_from_spec(s2); sys.modules["rr"] = rr; s2.loader.exec_module(rr)
    _cases, traj = rr.load_failures("n0e1_train", "vector", 16, want_traj=True)
    states = []
    for cid in sorted(traj):
        states.extend(rr._states_for_case(cid, traj))

    # THE PROJECTION IS SYNTHESIZED-THETA vs WHAT THE EXECUTOR ACTUALLY ARMS ON, not vs whatever
    # number the proposal echoes. My first version compared the proposal's threshold field against
    # itself -- the model returned 0.294 and the check reported 0% disagreement, which is vacuous: a
    # policy always agrees with itself. What must be tested is the value the HOST will use.
    #
    # Resolved from the evaluator: the threshold appears exactly once, as the ARMING condition
    # (decline when the observed score is at or above it). There is no separate post-merge filter, so
    # it IS the parameter the synthesized theta varies, and it is a module constant -- the proposer's
    # own first branch, which it flagged explicitly rather than assuming away.
    synth_theta = 0.2940          # the candidate the grammar produced, from selection.json
    realized_theta = float(next(iter(fixed.values())))
    print(f"\n  NOTE the proposal's threshold field said {pr.get('threshold')}; the projection is "
          f"tested against the executor's actual arming value {realized_theta}, not that echo.")
    def policy(theta):
        return Conjunction((Atom("searched_other_container", "falsy"),
                            Atom("best_similarity", "lt", theta)))

    print(f"\n[2] DETERMINISTIC VERIFICATION  (the LLM does not decide this)")
    v = verify_projection(policy(synth_theta), policy(realized_theta), states,
                          rationale=str(proposal.get("rationale", "")))
    print(f"  proposed theta {synth_theta} fires on {v.proposed_fires} of {v.total_states} states")
    print(f"  realizable theta {realized_theta} fires on {v.realizable_fires}")
    print(f"  agree {v.agree}  disagree {v.disagree}  "
          f"= {100*(v.disagreement or 0):.1f}% of the union")
    print(f"  VERDICT: {v.state}")
    print(f"           {v.detail}")
    json.dump({"proposal": proposal, "verification": v.as_dict(),
               "synth_theta": synth_theta, "realized_theta": realized_theta},
              open(out_dir / "realization_verdict.json", "w"), indent=2)
    print(f"\n{'=' * 96}")
    print("PROJECTION ACCEPTED -- paired evaluation may proceed" if v.accepted
          else "PROJECTION REFUSED -- no evaluation; the realization is a different policy")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
