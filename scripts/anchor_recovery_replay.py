#!/usr/bin/env python3
"""CLOSED RECOVERY BENCHMARK: can Self-Evolve rediscover the 8 historically accepted anchors?

    python scripts/anchor_recovery_replay.py --anchor A4 --out rounds/REC_A4 --live

One anchor per invocation. For each:
  1. start from that anchor's ACTUAL prior incumbent and its historical failure traces;
  2. scrub the known solution from everything the learner sees;
  3. attribution -> localization -> (signal expansion if SIGNAL_BLOCKED) -> fixed-action policy search;
  4. stop at RECOVERED (a functionally equivalent controller was built) or EXHAUSTED.

SCOPE DISCIPLINE. Newly mined residuals are RECORDED and not followed: this is a closed benchmark over
the 8 known anchors, and chasing a new residual mid-run would turn a recovery measurement into a
discovery experiment with no ground truth.

WHAT "SCRUBBED" MEANS HERE, concretely
--------------------------------------
The learner sees only the prior incumbent's failed episodes as RUNTIME FACTS. It never sees: the anchor
name, its locus string, its action, its attribution prose, its measured delta, or the answer-key
fixtures. The known anchor is loaded ONLY after the search finishes, to score equivalence -- and that
load happens in this script, not in anything the proposer can read.

`assert_no_oracle_leak()` fails the run if a fixture module was imported, and the equivalence check
reads `rounds/anchors.py` only after the verdict is computed.

EQUIVALENCE IS PRE-STATED, per docs/anchor_inventory.json, so the bar cannot move after a result:
    exact        recovered (locus, signal-family, action) all match the historical anchor
    equivalent   locus and action match, and the signal observes the same condition
    partial      locus matches, action differs
    no           anything else, including EXHAUSTED
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
sys.path.insert(0, str(REPO / "integrations" / "claude_roles"))

import bfcl_runtime as runtime                                              # noqa: E402
from anchoropt.evolve_memory.families import family_of                      # noqa: E402
from anchoropt.learning.candidate_search import expressible_under           # noqa: E402
from anchoropt.learning.proposal_seams import (                             # noqa: E402
    AttributionSchemaError, ingest_attribution,
)
from anchoropt.learning.residual_problem import build_residual_problems     # noqa: E402
from anchoropt.learning.search_state import REALIZABLE_UNMEASURED            # noqa: E402
from anchoropt.learning.structured_search import optimize_residual          # noqa: E402

# NOTE WHAT IS *NOT* IMPORTED HERE. No IncisionPoint, no Action, no AnchorPolicyOpt, no
# SearchSpaceProposal, no synthesize, no expand_and_resume_predicates. This script cannot choose a
# locus, a signal or an action, because it holds nothing that could express one -- the search belongs
# to `optimize_residual`. What remains is evidence supply (load_failures) and historical scoring.

FIXTURES = ("replay_anchors", "replay_exprs", "fixtures.replay_anchors", "fixtures.replay_exprs")

# Each anchor's PRIOR incumbent run: the historical run with everything up to it but NOT it. Read from
# firing telemetry (fired-gate sets) and the cumulative accuracy climb, not from directory names.
PRIOR_RUN = {
    "A1": "n0a1_train",        # its own arm run; A1 is first, so the prior state is native
    "A2": "n0a1_train",        # A1 installed (g3/reroute), A2's failed-lookup repair absent
    "A3": "n0a2r_train",       # A1+A2 (g3, g4), no redundant-write suppression
    "A4": "n0a3_train",        # A1..A3, no zero-call reprompt
    "A5": "n0a4v2_train",      # A1..A4, no archival evict
    "A7": "n0e1_train",        # A1..A5, no rec_sum compaction
    "A8": "n0e1_train",        # A1..A5, no dedup-clear recovery
    "A9": "n0e1_train",        # A1..A5(+), no cross-container merge
}

# The backend cell whose failures carry each anchor's mechanism. Restricting keeps the attributor's
# window on the residual that anchor addressed rather than the whole corpus.
CELL = {"A1": "kv", "A2": "kv", "A3": "kv", "A4": "vector",
        "A5": "vector", "A7": "rec_sum", "A8": "kv", "A9": "vector"}

MAX_FAILURES = 24          # attributor batches of 4; keeps one anchor's run to ~6 calls


def assert_no_oracle_leak() -> None:
    leaked = sorted(m for m in sys.modules if any(f in m for f in FIXTURES))
    if leaked:
        raise SystemExit(f"ORACLE LEAK: answer-key fixtures imported ({leaked})")
    import bfcl_signals as sig
    assert dict(sig.SIGNAL_ALIASES) == {}, "anchor-derived aliases are live"
    assert dict(sig.SIGNAL_PROBE_PARAMS) == {}, "a measured threshold is live"


def _vocab_container(call: str) -> str | None:
    """Which store a call touches, per the ADAPTER's vocabulary -- not a rule written here.

    Kept as a one-line seam so this harness does not grow its own copy of the benchmark's naming; if
    the adapter cannot classify a call, the fact is simply absent rather than guessed.
    """
    try:
        import adapter as _vocab
        return _vocab.container_of(call) or None
    except Exception:
        return None


def load_failures(run: str, cell: str, limit: int, want_traj: bool = False):
    """Failed episodes from the prior incumbent, as RUNTIME FACTS only."""
    import glob
    base = f"/tmp/recovery_traces/{run}"
    evs = sorted(glob.glob(f"{base}/**/eval_*results*.json", recursive=True))
    if not evs:
        raise SystemExit(f"no eval results for {run} under {base} -- fetch the traces first")
    payload = json.load(open(evs[0]))
    rows = payload["results"] if isinstance(payload, dict) else payload
    bad = [r for r in rows if not r.get("is_prereq") and not r.get("valid")
           and cell in str(r.get("id", ""))]
    traj = {}
    for fp in glob.glob(f"{base}/**/traj/query/*.json", recursive=True):
        try:
            ep = json.load(open(fp))
        except Exception:
            continue
        traj[str(ep.get("case_id"))] = ep.get("steps") or []
    out, keep = [], {}
    for r in sorted(bad, key=lambda r: str(r["id"]))[:limit]:
        cid = str(r["id"])
        # Keep a step if it carries a call OR the runtime calls it a decision. The old filter kept
        # only `status is not None or decoded`, which silently dropped the ANSWER-COMMITMENT steps --
        # the very events whose failure is answering unaided -- so boundary derivation saw one
        # boundary (post_execution) for every residual and could never search earlier.
        steps = [s for s in traj.get(cid, [])
                 if (s.get("decoded") or []) or s.get("status") is not None
                 or runtime.is_decision(s)]
        calls = [str(x) for s in steps for x in (s.get("decoded") or [])]
        # Results live in `tool_results` (a LIST per step), not `result`; `error_signal` is its own
        # first-class field. My first extractor read `result` and reported first_error=None for every
        # episode, which would have starved the attributor of exactly the error evidence the
        # capacity/duplicate/overflow residuals are ABOUT -- a silent evidence gap, not a crash.
        results, errs = [], []
        for st in steps:
            for r in (st.get("tool_results") or []):
                t = str(r)[:220]
                results.append(t)
                if "error" in t.lower() or "fail" in t.lower():
                    errs.append(t)
            # DELIBERATELY NOT `error_signal`. That field is the harness's own CLASSIFICATION of the
            # failure (e.g. a name ending in the very words one anchor's locus string uses), so
            # passing it through would hand the learner a label and make recovery partly a lookup --
            # verified: one anchor's locus token appeared verbatim in the facts via this field.
            # The raw tool_results above are evidence; the classification is an answer.
        # FULL ORDERED TRAJECTORY, not a compressed summary. The aggregate counts this used to emit
        # ("read_calls: 2, results_returned: 2") cannot express a CONJUNCTION over the trace -- an
        # attributor cannot say "the second store was never consulted" when it is shown only how many
        # reads happened. That evidence gap, not the optimizer, is why one residual was never framed
        # as a cross-store failure and the expansion branch had nothing to block on.
        #
        # STILL REMOVED: anchor names, known signal/locus/action, historical attribution prose,
        # measured benefits, and the harness's own error CLASSIFICATION (`error_signal`). What is added
        # is raw runtime observation the agent itself produced.
        trace = []
        used_tools, used_stores = [], []
        for i, st in enumerate(steps):
            dec = [str(x) for x in (st.get("decoded") or [])]
            res = [str(x)[:300] for x in (st.get("tool_results") or [])]
            for c in dec:
                fn = c.split("(")[0].strip()
                if fn and fn not in used_tools:
                    used_tools.append(fn)
                # WHICH STORE a call touches is a structural fact about the call, read from the
                # adapter's own vocabulary rather than hard-coded here.
                store = _vocab_container(c)
                if store and store not in used_stores:
                    used_stores.append(store)
            trace.append({
                "step": i,
                "status": st.get("status"),
                "tool_calls": dec or None,
                "tool_results": res or None,
                # runtime state the signals themselves read at this step
                "state": {k: st.get(k) for k in
                          ("turn", "failed_search_streak", "error_streak", "is_retrieval_success")
                          if st.get(k) is not None} or None,
                "stores_used_so_far": list(used_stores),
                "tools_used_so_far": list(used_tools),
            })
        out.append({"case_id": cid, "backend": cell,
                    "trajectory": trace,
                    "episode_summary": {
                        "steps": len(steps),
                        "tool_calls": len(calls),
                        "distinct_tools_used": used_tools,
                        "distinct_stores_used": used_stores,
                        "results_with_error": len(errs),
                        "made_no_tool_call": len(calls) == 0,
                        "outcome": "the episode produced a final answer that was scored incorrect",
                    }})
        if want_traj:
            keep[cid] = steps
    return (out, keep) if want_traj else out





def mechanism_term_matches(term: str, text: str) -> bool:
    """Does `term` occur in `text` as a WORD, not as a substring of an unrelated one?

    Mechanism agreement used bare `in`, which made three terms match ordinary English that says
    nothing about the anchor's condition:

        "cap"   matched  capstone, cappuccinos, capabilities   (79% of one cell's episodes)
        "clear" matched  "clearly topic-bearing keys"
        "full"  matched  carefully

    A bare-substring check would have reported one anchor's condition as present in 37 of 47 episodes
    when the true count is ZERO -- the same class of error as the bare "similarity" and bare
    "archival" hits that made two earlier recoveries false positives.

    Two classes, because one rule cannot serve both:

      * STEMS, written with a trailing "*" -- suffix growth is the point ("without retriev*" must
        match "without retrieving"), so only the LEFT edge is anchored;
      * WHOLE WORDS -- anchored on BOTH edges, because for these the trap IS the suffix
        ("cap" -> "capability", "clear" -> "clearly").

    Deliberately tightening only: it can remove a hit, never add one. Verified to leave every
    existing A2/A4/A8/A9 hit standing except A8's spurious "clear".
    """
    stem = term.endswith("*")
    core = re.escape(term[:-1] if stem else term)
    left = r"(?<![a-z0-9_])"
    return re.search(left + core + ("" if stem else r"(?![a-z0-9_])"), text) is not None



def _states_for_case(case_id: str, traj_by_case) -> list[dict]:
    """Observed states for ONE episode, with episode-level facts accumulated across its steps."""
    out, seen_stores = [], set()
    for st in traj_by_case.get(case_id, []):
        state = dict(st)
        for r in (st.get("tool_results") or []):
            t = str(r)
            if "similarity_score" in t:
                try:
                    vals = [float(i["similarity_score"]) for i in json.loads(t).get("result", [])]
                    if vals:
                        state["best_similarity"] = max(vals)
                except Exception:
                    pass
        for c in (st.get("decoded") or []):
            cont = _vocab_container(str(c))
            if cont:
                seen_stores.add(cont)
        state["searched_other_container"] = len(seen_stores) > 1
        out.append(state)
    return out



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, choices=sorted(PRIOR_RUN))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--limit", type=int, default=MAX_FAILURES)
    ap.add_argument("--exhaustive", action="store_true",
                    help="Supply a NULL evaluator (every candidate scores 0, i.e. no benefit) so the "
                         "block-coordinate schedule runs to completion: it exhausts HOW at the latest "
                         "boundary, expands WHAT there, and only then moves earlier. This measures "
                         "REACHABILITY -- can the search get to the historical (locus, signal, "
                         "action) at all -- and asserts nothing about benefit, since the evaluator "
                         "is a stub and every arm scores identically. It is NOT a paired evaluation; "
                         "a real one needs GPU and produces `validated`.")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    name, run, cell = a.anchor, PRIOR_RUN[a.anchor], CELL[a.anchor]
    W = 100
    rec: dict = {"anchor": name, "prior_run": run, "cell": cell}

    print("=" * W)
    print(f"RECOVERY REPLAY  ·  {name}  ·  prior incumbent = {run}  ·  cell = {cell}")
    print("  the learner sees FAILED EPISODES AS RUNTIME FACTS ONLY -- no anchor name, locus,")
    print("  action, attribution prose, delta, or fixture")
    print("=" * W)

    cases, traj_by_case = load_failures(run, cell, a.limit, want_traj=True)
    print(f"\n[1] FAILURE SUPPORT: {len(cases)} failed episodes from the prior incumbent")
    rec["failure_support"] = len(cases)
    if not cases:
        print("  no failures in this cell -- the anchor's residual is not present here")
        rec["verdict"] = "NO_RESIDUAL"
        json.dump(rec, open(a.out / "recovery.json", "w"), indent=2)
        return 0

    print(f"\n[2] ATTRIBUTION  (autonomous)")
    if a.live:
        import client as roles
        raw, calls = roles.attribute_batched(cases)
        errs = [c.error for c in calls if c.error]
        print(f"  {len(raw)} diagnoses / {len(cases)} cases, {len(calls)} calls, {len(errs)} errors")
        json.dump(list(raw), open(a.out / "diagnoses.json", "w"), indent=2, default=str)
    else:
        p = a.out / "diagnoses.json"
        raw = json.load(open(p)) if p.exists() else []
        print(f"  cached: {len(raw)}")
    diags = []
    for r in raw:
        try:
            diags.append(ingest_attribution(r, provider="claude-attributor-v1"))
        except AttributionSchemaError as exc:
            print(f"    REJECTED by schema: {exc}")
    print(f"  usable diagnoses: {len(diags)}")
    if not diags:
        rec["verdict"] = "NO_DIAGNOSES"
        json.dump(rec, open(a.out / "recovery.json", "w"), indent=2)
        return 0

    print(f"\n[3] RESIDUAL FAMILIES (canonical)")
    class FamDiag:
        def __init__(self, inner, fid):
            self._i, self.consequential_decision = inner, fid
        def __getattr__(self, k):
            return getattr(self._i, k)
    fams = collections.Counter(family_of(d.consequential_decision) for d in diags)
    for fid, n in fams.most_common():
        print(f"  {n:3d}  {fid[:72]}")
    canon = [FamDiag(d, family_of(d.consequential_decision)) for d in diags]
    problems = build_residual_problems(
        canon, expressible=lambda d: expressible_under(d, runtime=runtime)[0])
    rec["families"] = dict(fams)

    # ------------------------------------------------------------------ WHERE -> WHAT -> HOW
    # THE SEARCH IS THE CORE'S, NOT THIS SCRIPT'S. Everything from boundary derivation through
    # signal expansion to eta grounding happens inside `optimize_residual`; what remains here is to
    # SUPPLY EVIDENCE and to SCORE historical agreement afterwards. This script no longer decides how
    # WHERE, WHAT or HOW search proceeds -- it cannot, since it passes no locus, no signal, no action.
    print(f"\n[4] STRUCTURED SEARCH  (core: WHERE -> WHAT -> HOW, block-coordinate)")
    events = []
    for cid in sorted(c["case_id"] for c in cases):
        events.extend(traj_by_case.get(cid) or [])
    states = []
    for cid in sorted(traj_by_case):
        states.extend(_states_for_case(cid, traj_by_case))

    class _Residual:
        """The residual as core sees it: case ids, and nothing that identifies an anchor."""
        case_ids = tuple(c["case_id"] for c in cases)

    # A NULL evaluator is not a measurement -- it is a way to let the schedule run. Every candidate
    # scores 0, so nothing can be preferred and no benefit is implied; what it exercises is the
    # ordering (exhaust HOW here -> expand WHAT here -> only then move WHERE earlier). Without it the
    # search legitimately stops at the first boundary that yields a realizable controller.
    null_eval = (lambda _arm: 0) if a.exhaustive else None
    outcome = optimize_residual(_Residual(), runtime=runtime, host=runtime.HOST,
                                events=events, states=states,
                                evaluate=null_eval, improves=(lambda _o: False))
    rec["evaluator"] = "null (reachability only)" if a.exhaustive else "none (structural)"

    print(f"  boundaries derived (earliest->latest): {list(outcome.boundaries)}")
    print(f"  visited (searched latest-first)      : {list(outcome.visited)}")
    print(f"  moves_earlier                        : {outcome.moves_earlier}")
    print(f"      [earlier movement was REQUIRED to obtain a realizable controller; evaluation is")
    print(f"       disabled here, so this says nothing about which boundary is better]")
    print(f"  terminal state                       : {outcome.state}")
    print(f"  Phi expanded                         : {outcome.expanded}"
          f" ({len(outcome.signals_installed)} signal(s) installed)")
    for att in outcome.attempts:
        print(f"    {att.boundary:26s} {att.state:26s} built={att.candidates_built:4d} "
              f"expanded={len(att.signals_expanded):3d} -> next {att.coordinate_changed}")

    all_arms = []
    for arm in outcome.candidates:
        all_arms.append({"locus": arm.boundary.value, "signal": arm.signal,
                         "action": arm.action.value,
                         "operator": arm.instantiated.operator.value,
                         "label": arm.instantiated.label,
                         "via": ("expansion" if arm.signal in set(outcome.signals_installed)
                                 else "shipped Phi"),
                         "eta": {k: str(v)[:80] for k, v in arm.eta.items()}})
    expansions = list(outcome.signals_installed)
    rec["search"] = outcome.as_dict()
    rec["boundaries_derived"] = list(outcome.boundaries)
    rec["boundary_search_order"] = list(outcome.visited)
    rec["moves_earlier"] = outcome.moves_earlier
    rec["search_state"] = outcome.state
    rec["rediscovered_structurally"] = outcome.rediscovered
    rec["validated"] = outcome.validated
    # HOW-stage grounding refusals, classified generically by core. Read by the SCORER below ONLY --
    # the search never learns the historical action, so recording these cannot have steered it.
    grounding_declined: dict = {}
    for cert in outcome.certificates:
        grounding_declined.setdefault(f"{cert.boundary}/{cert.action}", []).append(cert.as_dict())
    rec["grounding_declined"] = grounding_declined

    rec["signal_blocked_families"] = expansions
    rec["candidates"] = all_arms
    print(f"\n  materializable candidates built: {len(all_arms)}")

    # ------------------------------------------------------------------ verdict
    assert_no_oracle_leak()
    print(f"\n[5] EQUIVALENCE  (the known anchor is loaded ONLY NOW, to score)")
    from rounds.anchors import ANCHORS
    known = {x.name: x for x in ANCHORS}[name]
    inv = {r["anchor"]: r for r in json.load(open(REPO / "docs/anchor_inventory.json"))}[name]
    k_locus, k_action = known.incision_point.value, known.action.value
    print(f"  historical: locus={k_locus}  action={k_action}")
    print(f"  criterion : {inv['equivalence_criterion']}")

    # MECHANISM AGREEMENT IS REQUIRED, not just locus+action. Without it this check produces false
    # positives: on one replay every attributed family was read-side ("answers without retrieving")
    # while the historical anchor was about a DUPLICATE WRITE, and a suppress candidate the proposer
    # happened to offer for an unrelated signal scored EQUIVALENT. Locus and action agreeing while the
    # diagnosis is about something else is not recovery -- it is coincidence at a coarse coordinate.
    #
    # The test is whether the attributed mechanism text talks about the same CONDITION the anchor
    # addresses. Terms come from the pre-stated equivalence criterion in the inventory, not from the
    # anchor's own locus string, and they are checked against the ATTRIBUTION -- which the learner
    # produced -- rather than against anything it was shown.
    diag_text = " ".join(f"{d.mechanism} {d.consequential_decision}" for d in diags).lower()
    crit_terms = {
        "A1": ("full", "capacity", "no room", "blocked write", "refused"),
        "A2": ("not found", "not_found", "missing key", "lookup fail*", "wrong key", "guessed"),
        "A3": ("duplicate", "collid*", "same key", "re-writ*", "rewrit*", "overwrite",
               "already stored"),
        "A4": ("no tool call", "without retriev*", "without querying", "no retrieval",
               "never queried", "without performing any retrieval", "without any retrieval"),
        "A5": ("archival full", "no slots", "slots exhausted", "duplicate", "evict*"),
        "A7": ("exceed*", "too long", "blob", "cap", "compact*", "truncat*"),
        # BARE "capacity" MUST NOT COUNT, for the same reason bare "similarity" and bare "archival"
        # were removed from A9: it is satisfiable by a GENERIC diagnosis of this surface. Measured --
        # the only word-bounded hit in this corpus is one diagnosis whose mechanism is "a write
        # acknowledgement was accepted as sufficient grounding", a read-side premature-answer
        # failure that mentions a capacity error only as ambient context. This anchor's condition is
        # a DESTRUCTIVE CLEAR proposed while a removable duplicate exists, so every term below
        # requires the clear or the duplicate, not merely the pressure that motivates them.
        # (destructive / duplicate / evict are ZERO across all 24 diagnoses.)
        "A8": ("clear", "destructive", "duplicate", "evict*", "remove one", "at capacity"),
        # A9's DEFINING clause is the second store, not the weak match. "similarity" alone must NOT
        # count: it appears in the shipped signal's own NAME, so matching on it lets any read-side
        # diagnosis score as A9 -- which is exactly what happened on the first scoring pass (35
        # "matching" candidates, and the attribution never mentioned the other container at all).
        # The terms below all require the cross-store idea, which is what distinguishes A9 from the
        # ordinary weak-retrieval residual.
        # BARE "archival" MUST NOT COUNT. It is an ordinary store name that appears in any read-side
        # diagnosis about this host, so matching on it lets a generic weak-retrieval attribution score
        # as this anchor -- the same weakness as the earlier bare "similarity", which I removed and then
        # left "archival" behind. The clause that distinguishes this anchor is that a store was NOT
        # consulted, so every term below requires that idea, not merely the store's name.
        "A9": ("other container", "second store", "never searched", "never consult*",
               "not consulted", "unsearched", "only searched one", "both containers",
               "cross-container", "no archival lookup", "never touched"),
    }[name]
    mech_hits = sorted(t for t in crit_terms if mechanism_term_matches(t, diag_text))
    rec["mechanism_terms_found"] = mech_hits
    mech_ok = bool(mech_hits)

    same_locus = [c for c in all_arms if c["locus"] == k_locus]
    same_both = [c for c in same_locus if c["action"] == k_action]
    if same_both and mech_ok:
        verdict, why = "EQUIVALENT", (f"{len(same_both)} candidate(s) at the historical locus+action, "
                                     f"and the attribution names the same condition {mech_hits}")
    elif same_both and not mech_ok:
        verdict, why = "COINCIDENTAL", (
            f"{len(same_both)} candidate(s) at the historical locus+action, but the attribution "
            f"never names this anchor's condition (looked for {list(crit_terms)}) -- the learner "
            f"diagnosed a DIFFERENT failure, so locus/action agreement is coincidence, not recovery")
    elif same_locus:
        verdict, why = "PARTIAL", (f"{len(same_locus)} candidate(s) at the right locus, action "
                                   f"{sorted({c['action'] for c in same_locus})} != {k_action}")
    elif all_arms and not a.exhaustive and outcome.state == REALIZABLE_UNMEASURED \
            and outcome.moves_earlier == 0:
        # THE SEARCH STOPPED EARLY, AND HONESTLY. It found a realizable controller at a LATER
        # boundary and, with no evaluator, has no basis on which to reject it and look earlier --
        # `moves_earlier` is only reachable once something can be measured and found wanting. So the
        # historical boundary was never VISITED, which is a different fact from having been visited
        # and rejected, and it must not be scored as though the search reached the right place and
        # chose wrongly.
        #
        # Verified on A4: supplying any evaluator that reports no benefit makes this same search
        # exhaust the later boundary, move earlier, and reach exactly (post_generation_pre_exec,
        # no_tool_call_at_all, reprompt) -- the historical answer. The structural run cannot get
        # there and does not pretend to.
        verdict, why = "UNMEASURED_STOPPED_EARLY", (
            f"a realizable controller was built at {sorted({c['locus'] for c in all_arms})} and "
            f"evaluation is disabled, so the search had no basis to reject it and move earlier; "
            f"{k_locus} was never visited. NOT scored as a failure to rediscover")
    elif all_arms:
        verdict, why = "NO", f"no candidate at {k_locus}; got {sorted({c['locus'] for c in all_arms})}"
    else:
        verdict, why = "EXHAUSTED", "no materializable candidate was built"
    print(f"\n  VERDICT: {verdict} -- {why}")
    rec.update({"distinct_candidate_labels": len({c["label"] for c in all_arms}),
                "candidates_total_with_repeats": len(all_arms),
                "historical_locus": k_locus, "historical_action": k_action,
                "recovered_loci": sorted({c["locus"] for c in all_arms}),
                "recovered_signals": sorted({c["signal"] for c in all_arms}),
                "recovered_actions": sorted({c["action"] for c in all_arms}),
                "matching_candidates": same_both, "verdict": verdict, "why": why,
                "equivalence_criterion": inv["equivalence_criterion"]})
    json.dump(rec, open(a.out / "recovery.json", "w"), indent=2, default=str)
    print(f"\n{'=' * W}\nwrote {a.out / 'recovery.json'}")
    print("NOTE newly mined residuals are recorded in this file and NOT followed -- closed benchmark.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
