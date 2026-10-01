#!/usr/bin/env python3
"""LIVE proposer on the frozen top residual, then AnchorPolicyOpt. Dry-run: no GPU, no evaluation.

    python scripts/live_policy_recovery.py --out /tmp/recovery_dry

Tests whether the POLICY-selection loop works end to end with a real proposer:

    residual -> attribution -> ranking -> R1 (frozen) -> LIVE (l, phi, U-hat)
             -> AnchorPolicyOpt arms -> [evaluation happens later, on GPU]

phi may already exist in the vocabulary; this is not a signal-invention test. The historical anchor
is an EVALUATION ORACLE ONLY and is never shown to the learner -- `assert_no_oracle_leak` fails the
run if any accepted-anchor identifier reached the process.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
sys.path.insert(0, str(REPO / "integrations" / "claude_roles"))

import bfcl_runtime as runtime                                          # noqa: E402
import client as roles                                                  # noqa: E402
from anchoropt.anchor import Action, IncisionPoint                        # noqa: E402
from anchoropt.learning.anchor_policy_opt import (                        # noqa: E402
    AnchorPolicyOpt, FrozenIncumbent, SearchSpaceProposal,
)
from anchoropt.learning.candidate_search import expressible_under         # noqa: E402
from anchoropt.learning.policy_class import PolicySpec                    # noqa: E402
from anchoropt.learning.residual_problem import build_residual_problems   # noqa: E402
from anchoropt.runtime import ResidualDiagnosis                           # noqa: E402

AUDIT = REPO / "results/selfevolve_r1/audit.jsonl"
OBS = json.load(open(REPO / "results/selfevolve_r1/observed_similarity_control.json"))["per_call"]
FIXTURES = ("replay_anchors", "replay_exprs", "fixtures.replay_anchors", "fixtures.replay_exprs")


def assert_no_oracle_leak() -> None:
    leaked = sorted(m for m in sys.modules if any(f in m for f in FIXTURES))
    if leaked:
        raise SystemExit(f"ORACLE LEAK: historical anchor fixtures imported ({leaked})")
    import bfcl_signals as sig
    assert dict(sig.SIGNAL_ALIASES) == {}, "anchor-derived aliases are live"
    assert dict(sig.SIGNAL_PROBE_PARAMS) == {}, "a measured threshold is live"
    assert dict(runtime.REROUTE_DESTINATIONS) == {}, "historical repairs are live"


def load_diagnoses():
    out = []
    for line in open(AUDIT):
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


def phi_expressible(d):
    ok, _why = expressible_under(d, runtime=runtime)
    return ok


def to_search_space(raw, problem):
    """One live proposal -> SearchSpaceProposal. U-hat is the proposer's ACTION SET.

    Accepts either `action_set` (the decomposed schema) or a single `action` (older schema), and in
    the latter case widens U-hat to every action the host can execute at that boundary -- the
    proposer having named one action must not silently narrow the search to it.
    """
    boundary = IncisionPoint(str(raw["boundary"]))
    names = raw.get("action_set") or ([raw["action"]] if raw.get("action") else [])
    acts = []
    for n in names:
        try:
            a = Action(str(n).strip().lower())
        except ValueError:
            continue
        if a is not Action.NOOP:
            acts.append(a)
    if len(acts) <= 1:
        acts = sorted((a for a in runtime.HOST.executable_actions(boundary) if a is not Action.NOOP),
                      key=lambda a: a.value)
    return SearchSpaceProposal(
        boundary=boundary, signal=str(raw.get("signal") or raw.get("signal_name")),
        action_set=tuple(acts), rationale=str(raw.get("rationale", "")),
        diagnosis_case_ids=tuple(c for c in (raw.get("diagnosis_case_ids") or ())
                                 if c in set(problem.case_ids)),
        preferred_action=str(raw.get("action") or raw.get("preferred_action") or ""),
        theta_hint=_coerce_hint(raw, runtime))


def _coerce_hint(raw, rt):
    """A hint is PROVENANCE. It must never cost a proposal.

    The live proposer emitted `theta_hint: 0.45` -- a bare float rather than a mapping -- and a
    `dict()` on it raised TypeError, silently dropping an otherwise legitimate proposal at ingest.
    Losing a candidate to the shape of a field that constrains nothing is the wrong failure: the
    hint is normalized onto the signal's declared parameter name when there is exactly one, and
    discarded otherwise.
    """
    hint = raw.get("signal_params") or raw.get("theta_hint")
    if isinstance(hint, dict):
        return dict(hint)
    if isinstance(hint, (int, float)) and not isinstance(hint, bool):
        sig = str(raw.get("signal") or raw.get("signal_name") or "")
        doms = rt.parameter_domains(sig) if hasattr(rt, "parameter_domains") else ()
        if len(doms) == 1:
            return {doms[0].name: float(hint)}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/recovery_dry"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "recovery.jsonl", "a")

    def rec(kind, **kw):
        log.write(json.dumps({"t": time.time(), "kind": kind, **kw}, default=str) + "\n")
        log.flush()

    W = 104
    ds = load_diagnoses()
    print("=" * W)
    print(f"LIVE POLICY RECOVERY (dry-run)  ·  {len(ds)} real diagnoses  ·  nothing evaluated")
    print("=" * W)

    problems = build_residual_problems(ds, expressible=phi_expressible)
    print("\n[2] RANKED RESIDUALS (metric unchanged: support, then key)")
    for p in problems:
        print(f"  R{p.rank} support={p.support} cov={100*p.coverage:5.1f}% "
              f"{'expressible' if p.expressible else 'SIGNAL_BLOCKED'}  {p.key[:70]}")
    rec("ranked", problems=[{"rank": p.rank, "support": p.support, "key": p.key} for p in problems])

    R1 = problems[0]
    print(f"\n[3] FROZEN R1\n  {R1}")

    print(f"\n[4] DIAGNOSES THE PROPOSER SEES (only R1's {R1.support})")
    for d in R1.scoped_diagnoses():
        print(f"  {d.case_id:32s} {d.mechanism[:64]}")
    print(f"  withheld: {[c for p in problems[1:] for c in p.case_ids]}")

    print(f"\n[5] LIVE PROPOSER on R1 ...")
    raw, call = roles.propose_semantic(R1.scoped_diagnoses(), runtime=runtime, host=runtime.HOST,
                                       max_proposals=4)
    rec("proposer_call", error=call.error, raw=call.raw, usage=call.usage)
    if call.error:
        print(f"  FAILED: {call.error}")
        return 1
    print(f"  returned {len(raw)} proposal(s)  "
          f"({call.usage.get('input_tokens')} in / {call.usage.get('output_tokens')} out)")

    proposals = []
    for r in raw:
        try:
            sp = to_search_space(r, R1)
        except Exception as exc:
            print(f"  rejected at ingest: {type(exc).__name__}: {exc}")
            continue
        proposals.append(sp)
        print(f"\n  l={sp.boundary.value}   phi={sp.signal}")
        print(f"     U-hat = {[x.value for x in sp.action_set]}")
        print(f"     cites {len(sp.diagnosis_case_ids)} of R1's {R1.support} cases")
        print(f"     PROVENANCE ONLY: preferred={sp.preferred_action or '(none)'} "
              f"hint={dict(sp.theta_hint) or '(none)'}")
        print(f"     rationale: {sp.rationale[:96]}")

    opt = AnchorPolicyOpt(runtime=runtime, host=runtime.HOST)
    total = 0
    print(f"\n[6]/[7]/[8]/[9] ANCHORPOLICYOPT ARMS, eta_mu, theta_phi, and INFEASIBLE CELLS")
    for sp in proposals:
        arms, rejected = opt.build_arms(sp)
        n_theta = 1
        if arms:
            spec = PolicySpec(boundary=arms[0].boundary, signal=arms[0].signal,
                              action=arms[0].action, policy_class=arms[0].policy_class,
                              domains=arms[0].signal_domains, theta=arms[0].eta)
            n_theta = len(spec.candidate_thetas({"best_similarity": OBS}, n=9))
        total += len(arms) * n_theta
        print(f"\n  -- {sp.boundary.value} / {sp.signal}:  {len(arms)} arm(s) x {n_theta} theta "
              f"= {len(arms)*n_theta} counterfactuals   (policy_class="
              f"{arms[0].policy_class.value if arms else 'n/a'})")
        for arm in arms:
            print(f"     + {arm.instantiated.label}")
            print(f"         eta_mu = { {k: str(v)[:56] for k, v in arm.eta.items()} }")
        for rj in rejected:
            op = f"[{rj.operator.value}]" if rj.operator else ""
            print(f"     - {rj.action.value}{op}: {rj.reason_code}"
                  + (f" missing={list(rj.missing)}" if rj.missing else ""))
        rec("arms", locus=f"{sp.boundary.value}/{sp.signal}",
            arms=[{"label": x.instantiated.label, "eta": {k: str(v) for k, v in x.eta.items()}}
                  for x in arms],
            rejected=[{"action": r.action.value, "code": r.reason_code,
                       "missing": list(r.missing)} for r in rejected],
            n_theta=n_theta)

    print(f"\n[10] ORACLE LEAKAGE AUDIT")
    assert_no_oracle_leak()
    blob = json.dumps([{"l": s.boundary.value, "phi": s.signal,
                        "U": [x.value for x in s.action_set]} for s in proposals]).lower()
    import re
    hits = re.findall(r"\ba[0-9]\b", blob)
    assert not hits, f"an anchor name reached the proposal: {hits}"
    print("  no historical fixture imported; no live alias/threshold/repair; "
          "no anchor name in any proposal  -> CLEAN")

    print(f"\n{'=' * W}")
    print(f"total counterfactual arms for R1: {total}")
    print("=" * W)
    json.dump({"problems": [{"rank": p.rank, "support": p.support, "key": p.key}
                            for p in problems],
               "frozen": R1.key, "n_proposals": len(proposals), "total_arms": total,
               "evaluated": False}, open(a.out / "summary.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
