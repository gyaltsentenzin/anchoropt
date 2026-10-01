#!/usr/bin/env python3
"""Live smoke test: real Claude in two roles, real BFCL residual, AnchorOpt's real validation.

    python scripts/live_smoke_bfcl.py --cases 8 --out /tmp/anchoropt_smoke

WHAT THIS DOES AND DOES NOT PROVE
---------------------------------
It answers ONE question: given real residual failures, does live Claude produce diagnoses and
proposals that survive AnchorOpt's validation machinery? That is a plumbing-and-vocabulary check.

It does NOT evaluate anything. No Granite call, no paired arm, no acceptance decision -- so it
cannot produce an efficacy claim, and the summary says so. Evaluation is the expensive half and it
comes after this passes.

THE FIXTURES ARE NOT IMPORTED
-----------------------------
`fixtures/replay_anchors.py` and `fixtures/replay_exprs.py` hold the A1-A9 answer key and are
CALIBRATION ONLY. This script must never import them, directly or transitively, or the "discovery"
would be a lookup. `assert_no_fixture_leak()` checks `sys.modules` after the run and fails loudly --
a claim about discovery is worth nothing if the answer key was in the room.

WHAT IS LOGGED
--------------
Everything, verbatim, as JSONL: the raw text of every role call, every diagnosis (accepted or
rejected with its schema reason), every proposal, every decline reason from AnchorOpt, and the phase
verdict with its per-locus S1/S2/S3 numbers. A run whose reasoning cannot be audited afterwards has
not demonstrated anything.
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

import bfcl_runtime as runtime                                          # noqa: E402
import client as roles                                                 # noqa: E402
from anchoropt.learning.block_loop import residual_phase                # noqa: E402
from anchoropt.learning.proposal_seams import (                         # noqa: E402
    AttributionSchemaError, ProposalSchemaError, ingest_attribution, ingest_proposal,
    rank_by_support, validate_proposal,
)

# Residual source: the a10 ledger's silent-failure cases, which carry per-case RUNTIME FACTS
# (read/write counts, empty results, best similarity). Raw per-step trajectories were purged from
# this repo by design, so these ledgers are the real evidence available -- and they are real
# measurements, not reconstructions.
WORKING_REPO = REPO.parent / "mellea-playground" / "decision_ops_agentic"
SILENT_CASES = WORKING_REPO / "ledgers" / "a10" / "a10_silent_cases.json"

FIXTURE_MODULES = ("replay_anchors", "replay_exprs", "fixtures.replay_anchors",
                   "fixtures.replay_exprs")


def assert_no_fixture_leak() -> None:
    leaked = sorted(m for m in sys.modules if any(f in m for f in FIXTURE_MODULES))
    if leaked:
        raise SystemExit(f"FIXTURE LEAK: the A1-A9 answer key was imported ({leaked}). A discovery "
                         f"claim from this run would be a lookup.")


def load_residual(limit: int, split: str = "train") -> list[dict]:
    """Real failed cases, as the runtime facts recorded for them. No prose, no labels, no anchors."""
    if not SILENT_CASES.exists():
        raise SystemExit(f"residual ledger not found: {SILENT_CASES}")
    rows = json.load(open(SILENT_CASES))
    rows = rows if isinstance(rows, list) else list(rows.values())[0]
    rows = [r for r in rows if r.get("split") == split]
    # Deterministic selection: sort by case id, take a spread across shards so one backend's failure
    # mode cannot masquerade as the whole residual.
    by_shard: dict[str, list[dict]] = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: str(r.get("case"))):
        by_shard[str(r.get("shard"))].append(r)
    out: list[dict] = []
    while len(out) < limit and any(by_shard.values()):
        for shard in sorted(by_shard):
            if by_shard[shard] and len(out) < limit:
                out.append(by_shard[shard].pop(0))
    return out


def as_attributor_input(row: dict) -> dict:
    """One failed case as RUNTIME FACTS the attributor may reason from.

    Deliberately field-named the way the runtime names things, and deliberately free of any label,
    error class, or anchor hint: the attributor's job is to explain the failure, not to recognize a
    category we already named.
    """
    facts = {
        "steps": row.get("nsteps"),
        "read_calls": row.get("nread"),
        "write_calls": row.get("nwrite"),
        "results_returned": row.get("nres"),
        "empty_results": row.get("empty"),
        "made_no_tool_call": row.get("zero_call"),
        "outcome": "the episode produced a final answer that was scored incorrect",
    }
    if row.get("maxsim") is not None:
        facts["best_similarity_score"] = round(float(row["maxsim"]), 4)
        facts["scored_entries_returned"] = row.get("nsim")
    return {"case_id": row.get("case"), "backend": row.get("shard"), "runtime_facts": facts}


class Log:
    """JSONL audit log. Append-only, flushed per record, so a crash keeps what happened."""

    def __init__(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a")
        self.path = path

    def write(self, kind: str, **payload) -> None:
        self.fh.write(json.dumps({"t": time.time(), "kind": kind, **payload},
                                 default=str) + "\n")
        self.fh.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, default=8, help="residual cases to attribute")
    ap.add_argument("--max-proposals", type=int, default=4)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("/tmp/anchoropt_smoke"))
    ap.add_argument("--model", default=roles.DEFAULT_MODEL)
    a = ap.parse_args()

    log = Log(a.out / "audit.jsonl")
    rows = load_residual(a.cases)
    cases = [as_attributor_input(r) for r in rows]
    log.write("residual_loaded", n=len(cases), source=str(SILENT_CASES),
              case_ids=[c["case_id"] for c in cases])

    print("=" * 100)
    print(f"LIVE SMOKE  model={a.model}  cases={len(cases)}  log={log.path}")
    print("=" * 100)
    print(f"\nresidual: {len(cases)} real failed cases "
          f"({collections.Counter(c['backend'] for c in cases)})")

    # ---------------------------------------------------------------- ROLE 1: attributor
    print("\n[1/4] ATTRIBUTOR -- diagnosing real failures ...")
    raw_diagnoses, recs = roles.attribute_batched(cases, model=a.model)
    for rec in recs:
        log.write("role_call", role="attributor", model=rec.model,
                  prompt_chars=rec.prompt_chars, usage=rec.usage, error=rec.error, raw=rec.raw)
    failed = [r for r in recs if r.error]
    for r in failed:
        print(f"  batch FAILED: {r.error}")
    tok_in = sum(int(r.usage.get("input_tokens") or 0) for r in recs)
    tok_out = sum(int(r.usage.get("output_tokens") or 0) for r in recs)
    if not raw_diagnoses:
        print("  no diagnoses returned; stopping.")
        return 1
    print(f"  returned {len(raw_diagnoses)} diagnoses over {len(recs)} batch(es), "
          f"{len(failed)} failed  ({tok_in} in / {tok_out} out tokens)")

    diagnoses, rejected = [], []
    for r in raw_diagnoses:
        try:
            d = ingest_attribution(r, provider=f"claude-attributor:{a.model}")
            diagnoses.append(d)
            log.write("diagnosis_accepted", case_id=d.case_id, mechanism=d.mechanism,
                      consequential_decision=d.consequential_decision,
                      causal_region=(d.metadata or {}).get("causal_region"), raw=r)
        except AttributionSchemaError as exc:
            rejected.append((r.get("case_id"), str(exc)))
            log.write("diagnosis_rejected", case_id=r.get("case_id"), reason=str(exc), raw=r)
    print(f"  schema: {len(diagnoses)} accepted, {len(rejected)} rejected")
    for cid, why in rejected[:4]:
        print(f"    REJECTED {cid}: {why[:110]}")
    if not diagnoses:
        print("  no usable diagnosis; stopping.")
        return 1

    print("\n  what the attributor actually said (first 3):")
    for d in diagnoses[:3]:
        region = (d.metadata or {}).get("causal_region") or {}
        print(f"    [{d.case_id}]")
        print(f"      mechanism : {d.mechanism[:150]}")
        print(f"      region    : step={region.get('step')} phase={region.get('phase')!r}")
        print(f"      decision  : {d.consequential_decision[:130]}")

    ranked = rank_by_support(diagnoses)
    groups = collections.Counter(
        " ".join(d.consequential_decision.lower().split()) for d in diagnoses)
    print(f"\n  computed support (grouped by consequential decision, NOT model-asserted):")
    for k, v in groups.most_common(5):
        print(f"    {v:3d}  {k[:88]}")
    log.write("support_computed", groups=dict(groups))

    # ---------------------------------------------------------------- AnchorOpt: phase switch
    print("\n[2/4] ANCHOROPT PHASE SWITCH -- ours, not Claude's ...")

    def expressible(d):
        from anchoropt.learning.candidate_search import expressible_under
        ok, _why = expressible_under(d, runtime=runtime)
        return ok

    phase, reasons = residual_phase(ranked, expressible=expressible)
    log.write("phase_decided", phase=phase, reasons=list(reasons))
    print(f"  verdict: {phase}")
    for line in reasons[:8]:
        print(f"    {line}")

    allow_new_signal = (phase == "EXPAND_ATTRIBUTION")
    print(f"  -> block = {'SIGNAL (new phi permitted)' if allow_new_signal else 'POLICY (Phi frozen)'}")

    # ---------------------------------------------------------------- ROLE 2: proposer
    print("\n[3/4] PROPOSER -- structured candidates ...")
    raw_props, prec = roles.propose(
        ranked, runtime=runtime, host=runtime.HOST, allow_new_signal=allow_new_signal,
        max_proposals=a.max_proposals, model=a.model)
    log.write("role_call", role="proposer", model=prec.model, prompt_chars=prec.prompt_chars,
              usage=prec.usage, error=prec.error, raw=prec.raw,
              allow_new_signal=allow_new_signal)
    if prec.error:
        print(f"  FAILED: {prec.error}")
        return 1
    print(f"  returned {len(raw_props)} proposals "
          f"({prec.usage.get('input_tokens')} in / {prec.usage.get('output_tokens')} out tokens)")

    # ---------------------------------------------------------------- AnchorOpt: legality
    print("\n[4/4] ANCHOROPT VALIDATION -- Claude proposes, AnchorOpt decides legality ...")
    fields = runtime.all_fields()
    legal, declined = [], []
    for r in raw_props:
        try:
            cand = ingest_proposal(r, allow_new_signal=allow_new_signal,
                                   declared_signals=runtime.declared_signals())
        except ProposalSchemaError as exc:
            declined.append((json.dumps(r)[:80], f"schema: {exc}"))
            log.write("proposal_rejected_at_ingest", reason=str(exc), raw=r)
            continue
        compiled, why = validate_proposal(cand, runtime=runtime, host=runtime.HOST, fields=fields)
        what = f"{cand.boundary.value}/{cand.signal_name}/{cand.action.value}"
        if why == "legal":
            legal.append((cand, compiled))
            log.write("proposal_legal", what=what, boundary=cand.boundary.value,
                      signal=cand.signal_name, action=cand.action.value, theta=dict(cand.theta),
                      signal_expr=cand.signal_expr, signal_params=dict(cand.signal_params),
                      observable_at=sorted(compiled.boundaries) if compiled else None,
                      diagnosis_case_ids=list(cand.diagnosis_case_ids), rationale=cand.rationale,
                      raw=r)
        else:
            declined.append((what, why))
            log.write("proposal_declined", what=what, reason=why, raw=r)

    print(f"\n  LEGAL: {len(legal)} of {len(raw_props)}")
    for cand, compiled in legal:
        print(f"    + {cand.boundary.value} / {cand.signal_name} / {cand.action.value}")
        if cand.signal_expr:
            from anchoropt.learning.signal_lang import describe
            print(f"        phi  = {describe(cand.signal_expr)}")
            print(f"        obs@ = {sorted(compiled.boundaries)}")
        print(f"        theta= {dict(cand.theta)}")
        print(f"        for  = {len(cand.diagnosis_case_ids)} case(s)")
        print(f"        why  = {cand.rationale[:120]}")
    print(f"\n  DECLINED: {len(declined)}")
    for what, why in declined:
        print(f"    - {what}")
        print(f"        {why[:150]}")

    assert_no_fixture_leak()

    summary = {
        "model": a.model, "residual_cases": len(cases),
        "diagnoses_accepted": len(diagnoses), "diagnoses_rejected": len(rejected),
        "phase": phase, "block": "SIGNAL" if allow_new_signal else "POLICY",
        "proposals_returned": len(raw_props), "proposals_legal": len(legal),
        "proposals_declined": len(declined),
        "fixture_leak": False,
        "evaluated": False,
        "caveat": ("no Granite call, no paired arm, no acceptance decision -- this is a "
                   "plumbing/vocabulary check and cannot support an efficacy claim"),
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.write("summary", **summary)

    print("\n" + "=" * 100)
    print(f"SMOKE RESULT: {len(diagnoses)} diagnoses accepted, {len(legal)} legal controller(s), "
          f"{len(declined)} declined")
    print(f"  fixtures leaked : NO (A1-A9 answer key never imported)")
    print(f"  evaluated       : NO -- no Granite call, no paired arm. Not an efficacy result.")
    print(f"  audit log       : {log.path}")
    print("=" * 100)
    return 0 if legal else 2


if __name__ == "__main__":
    raise SystemExit(main())
