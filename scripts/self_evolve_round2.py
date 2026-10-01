#!/usr/bin/env python3
"""SELF-EVOLVE v0.2 -- the next round, with memory. Additive: v0.1 driver untouched.

    python scripts/self_evolve_round2.py --diagnoses rounds/SE1_A4_promoted/diagnoses_after.json \
        --out rounds/SE2 --live

    re-mine -> CANONICAL residual family -> retrieve memory -> propose (memory in context)
      -> screen candidates (INSTALLED / REUSE / EVALUATE) -> build arms for the EVALUATE set
      -> record every verdict back into the ledger

The difference from v0.1 is entirely in what it refuses to do twice. v0.1's step 8 re-proposed the
promoted A4' controller and both previously-measured low-similarity candidates and would have paid for
all of them on GPU.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
sys.path.insert(0, str(REPO / "integrations" / "claude_roles"))

import bfcl_runtime as runtime                                              # noqa: E402
from anchoropt.anchor import Action, IncisionPoint                          # noqa: E402
from anchoropt.evolve_memory import (                                      # noqa: E402
    EVALUATE, INSTALLED, REUSE, Experiment, ExperimentLedger, context_key,
    experience_block, prompt_block, reorder_residuals, screen_candidate,
)
from anchoropt.evolve_memory.families import (                             # noqa: E402
    family_label, family_of, is_canonical, unmatched_report,
)
from anchoropt.learning.anchor_policy_opt import (                          # noqa: E402
    AnchorPolicyOpt, SearchSpaceProposal,
)
from anchoropt.learning.candidate_search import expressible_under           # noqa: E402
from anchoropt.learning.proposal_seams import (                             # noqa: E402
    AttributionSchemaError, ingest_attribution,
)
from anchoropt.learning.residual_problem import build_residual_problems     # noqa: E402


class FamilyDiagnosis:
    """A diagnosis whose `consequential_decision` is the CANONICAL FAMILY, not raw prose.

    This is the whole of priority 1: `build_residual_problems` groups on
    `consequential_decision`, so canonicalizing that ONE field makes the existing ranking
    family-aware without touching the ranking code. The original prose is kept for the record.
    """

    def __init__(self, inner, family_id: str):
        self._inner = inner
        self.consequential_decision = family_id
        self.raw_decision = inner.consequential_decision

    def __getattr__(self, name):
        return getattr(self._inner, name)


def load_diagnoses(path: pathlib.Path):
    out = []
    for r in json.load(open(path)):
        try:
            out.append(ingest_attribution(r, provider="claude-attributor-v1"))
        except AttributionSchemaError as exc:
            print(f"  diagnosis REJECTED by the schema: {exc}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnoses", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--memory", type=pathlib.Path, default=REPO / "results/self_evolve/memory")
    ap.add_argument("--backend", default="vector")
    ap.add_argument("--split", default="train")
    ap.add_argument("--round-id", default="SE2")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    led = ExperimentLedger(a.memory)
    CTX = context_key(host=runtime.NAME, split=a.split, backend=a.backend)
    W = 100
    report: dict = {"round_id": a.round_id, "context": CTX}

    print("=" * W)
    print(f"SELF-EVOLVE v0.2  ·  round {a.round_id}  ·  incumbent = native + A4'")
    print("=" * W)

    diags = load_diagnoses(a.diagnoses)
    print(f"\n[1] RE-MINE  ({len(diags)} failures from the promoted incumbent)")
    raw_keys = {d.consequential_decision for d in diags}
    print(f"  raw attributor prose keys : {len(raw_keys)}")

    # ---------------------------------------------------------------- canonical families
    fam_counts = collections.Counter(family_of(d.consequential_decision) for d in diags)
    print(f"  CANONICAL families        : {len(fam_counts)}")
    for fid, n in fam_counts.most_common():
        tag = "" if is_canonical(fid) else "  [FALLBACK: no declared family]"
        print(f"    {n:3d}  {fid}{tag}")
        if is_canonical(fid):
            print(f"         \"{family_label(fid)}\"")
    unmatched = unmatched_report(diags)
    if unmatched:
        print(f"  unmatched (audit surface) : {unmatched}")
    report["raw_prose_keys"] = len(raw_keys)
    report["canonical_families"] = {k: v for k, v in fam_counts.items()}

    canon = [FamilyDiagnosis(d, family_of(d.consequential_decision)) for d in diags]
    problems = build_residual_problems(
        canon, expressible=lambda d: expressible_under(d, runtime=runtime)[0])
    print(f"\n[2] RANK  (support on the FAMILY, not on prose)")
    for p in problems:
        print(f"  R{p.rank} support={p.support:3d} cov={100*p.coverage:5.1f}% "
              f"{'expressible' if p.expressible else 'SIGNAL_BLOCKED'}  {p.key}")

    # ---------------------------------------------------------------- downweighting
    print(f"\n[3] DOWNWEIGHT families already tried and failed  (never removed)")
    ordered = reorder_residuals(led, problems)
    for p, w, why in ordered:
        print(f"  weight {w:5.3f}  support {p.support:3d} -> effective {p.support * w:6.2f}   "
              f"{p.key[:44]}  ({why})")
    report["ranking"] = [{"family": p.key, "support": p.support, "weight": w,
                          "effective": p.support * w, "why": why} for p, w, why in ordered]

    target = next((p for p, _, _ in ordered if p.expressible), None)
    if target is None:
        print("\n  no expressible family remains -- POLICY_EXHAUSTED_CURRENT_SPACE")
        return 0
    print(f"\n  SELECTED: {target.key}  (support {target.support})")

    # ---------------------------------------------------------------- retrieve memory
    print(f"\n[4] RETRIEVE RELEVANT PRIOR EXPERIENCE  (bounded, not the full history)")
    block = experience_block(led, residual=target.key, locus="", signal=None)
    prim = prompt_block(runtime)
    print(f"  experience block : {len(block)} chars, "
          f"{sum(1 for l in block.splitlines() if l.startswith('- '))} items")
    print(f"  primitive listing: {len(prim)} chars")
    (a.out / "memory_context.txt").write_text(block + "\n\n" + prim + "\n")
    print(f"  -> {a.out / 'memory_context.txt'}")

    # ---------------------------------------------------------------- propose
    print(f"\n[5] PROPOSE for the selected family")
    if a.live:
        import client as roles
        raw, call = roles.propose_semantic(target.scoped_diagnoses(), runtime=runtime,
                                          host=runtime.HOST, max_proposals=4)
        if call.error:
            print(f"  proposer FAILED: {call.error}")
            return 1
        json.dump(raw, open(a.out / "proposals.json", "w"), indent=2, default=str)
        print(f"  live proposer returned {len(raw)} proposal(s)  "
              f"({call.usage.get('input_tokens')} in / {call.usage.get('output_tokens')} out)")
    else:
        p = a.out / "proposals.json"
        raw = json.load(open(p)) if p.exists() else []
        print(f"  cached proposals: {len(raw)}")

    # ---------------------------------------------------------------- screen + build
    print(f"\n[6] SCREEN CANDIDATES against experiment memory")
    opt = AnchorPolicyOpt(runtime=runtime, host=runtime.HOST)
    tally = collections.Counter()
    to_evaluate, screened = [], []
    for r in raw:
        try:
            b = IncisionPoint(str(r.get("locus") or r.get("boundary")).strip().lower())
        except ValueError as exc:
            print(f"  proposal rejected at ingest: {exc}")
            continue
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
            acts = sorted((x for x in runtime.HOST.executable_actions(b) if x is not Action.NOOP),
                          key=lambda x: x.value)
        sp = SearchSpaceProposal(
            boundary=b, signal=phi, action_set=tuple(acts),
            diagnosis_case_ids=tuple(d.case_id for d in target.scoped_diagnoses()),
            rationale=str(r.get("rationale", ""))[:200])
        arms, rejected = opt.build_arms(sp)
        print(f"\n  l={b.value}  phi={phi}   materializable arms: {len(arms)}")
        for arm in arms:
            v = screen_candidate(led, residual=target.key, locus=b.value, signal=phi,
                                 action=arm.action.value, eta=arm.eta, context=CTX)
            tally[v.decision] += 1
            mark = {INSTALLED: "SKIP    ", REUSE: "REUSE   ", EVALUATE: "EVALUATE"}[v.decision]
            print(f"    [{mark}] {arm.instantiated.label}")
            print(f"               {v.reason[:104]}")
            screened.append({"locus": b.value, "signal": phi, "action": arm.action.value,
                             "label": arm.instantiated.label, "decision": v.decision,
                             "fingerprint": v.fingerprint, "reason": v.reason,
                             "eta": {k: str(x) for k, x in arm.eta.items()}})
            if v.decision == EVALUATE:
                to_evaluate.append((arm, b, phi))
        for rj in rejected:
            tally["infeasible"] += 1
            screened.append({"locus": b.value, "signal": phi, "action": rj.action.value,
                             "decision": "infeasible", "reason": rj.reason_code})

    print(f"\n[7] SCREENING RESULT")
    print(f"  INSTALLED (skip, already the incumbent)   : {tally[INSTALLED]}")
    print(f"  REUSE     (prior result, no GPU)          : {tally[REUSE]}")
    print(f"  EVALUATE  (genuinely new)                 : {tally[EVALUATE]}")
    print(f"  infeasible (pruned before screening)      : {tally['infeasible']}")
    saved = tally[INSTALLED] + tally[REUSE]
    print(f"\n  GPU EVALUATIONS AVOIDED BY MEMORY: {saved}"
          f"   (v0.1 would have run all {saved + tally[EVALUATE]})")
    report["screening"] = {"installed": tally[INSTALLED], "reuse": tally[REUSE],
                           "evaluate": tally[EVALUATE], "infeasible": tally["infeasible"],
                           "avoided": saved}
    report["candidates"] = screened

    # ---------------------------------------------------------------- record verdicts
    print(f"\n[8] RECORD back into the experiment ledger")
    n_new = 0
    for arm, b, phi in to_evaluate:
        led.record(Experiment(
            round_id=a.round_id, residual=target.key, locus=b.value, signal=phi,
            action=arm.action.value, eta=dict(arm.eta), status="rejected",
            diagnosis=f"{len(target.scoped_diagnoses())} failures in family {target.key}",
            reason="PROPOSED in this round and queued for evaluation; status is a placeholder "
                   "until the paired run reports -- overwritten by a second row, never edited.",
            context=CTX))
        n_new += 1
    print(f"  queued-for-evaluation rows written: {n_new}")
    print(f"  ledger now holds {len(led.experiments())} experiments, {len(led.lessons())} lessons")

    json.dump({"next_candidates": [
        {"locus": b.value, "signal": phi, "action": arm.action.value,
         "label": arm.instantiated.label, "eta": {k: str(v) for k, v in arm.eta.items()}}
        for arm, b, phi in to_evaluate]}, open(a.out / "to_evaluate.json", "w"), indent=2)
    json.dump(report, open(a.out / "round_report.json", "w"), indent=2, default=str)
    print(f"\n{'=' * W}")
    print(f"ROUND {a.round_id} PREPARED. {len(to_evaluate)} candidate(s) to evaluate; "
          f"{saved} avoided.")
    print(f"  -> {a.out / 'to_evaluate.json'}")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
