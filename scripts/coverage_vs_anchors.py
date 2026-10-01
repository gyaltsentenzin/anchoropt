"""Retrospective coverage: which of the historical eight anchors does the recovered library hold?

REPORTING ONLY, AND STRICTLY AFTER THE FACT. This never runs before or inside a discovery round and
nothing it prints is fed to a proposer. Its whole purpose is to let a human ask "did we already have
this?" -- the question whose absence let round 1 re-search a mechanism worth +4 that was already
recorded with full telemetry.

WHY MATCHING IS ON THREE FIELDS AND NOT ON `identity_key`
--------------------------------------------------------
The two vocabularies are genuinely different and pretending otherwise would manufacture both false
hits and false misses. `rounds/anchors.py` names an anchor by a SEMANTIC locus path
(`capacity/container/no_remaining_capacity`) with no capability_id at all, because capability identity
postdates it. The candidate library names a controller by the executable
(boundary, signal, action, operator, capability, phase) tuple. So a strict identity_key comparison
would report 0/8 recovered no matter what we had -- a number that says nothing about coverage and
everything about the schema.

What the two DO share, and what a rediscovery claim actually rests on, is: the same incision point, the
same action family, and the same failure condition. Those three are compared. The signal token is
matched against the LAST segment of the anchor's locus path, which is where that vocabulary puts the
condition (`no_remaining_capacity`, `append_would_exceed_cap`, `clear_proposed_at_capacity`).

A match on these three is reported as MECHANISM MATCH, not as proof of equivalence: the eta, the
executor and the thresholds may still differ, and any such difference is exactly what the per-anchor
note is for. This is deliberately the weaker claim.

Usage:  python scripts/coverage_vs_anchors.py [--library rounds/GOLDEN/candidates.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchoropt.learning.candidate_library import (            # noqa: E402
    ACCEPTED, CandidateLibrary, stack_fingerprint)
from rounds import anchors as ANCH                            # noqa: E402

#: The mechanism families the user asked to be tracked by name, mapped to the anchor that realises
#: each. Kept here rather than inferred from prose so the report cannot drift from the question.
FAMILY = {
    "A1": "capacity repair (reroute a blocked write)",
    "A2": "zero-result recovery (identifier not found)",
    "A3": "duplicate-write suppression",
    "A4": "zero-call recovery / reprompting",
    "A5": "capacity repair at the destination (evict+retry)",
    "A7": "capacity repair (blob size cap)",
    "A8": "clear prevention (destructive clear at capacity)",
    "A9": "retrieval reroute (similarity below threshold)",
}


def anchor_rows():
    out = []
    for name in ("A1", "A2", "A3", "A4", "A5", "A7", "A8", "A9"):
        a = getattr(ANCH, name)
        locus = str(a.locus)
        out.append({"anchor": name, "family": FAMILY.get(name, ""),
                    "incision": a.incision_point.value, "action": a.action.value,
                    "locus": locus, "condition": locus.rsplit("/", 1)[-1]})
    return out


#: The signal tokens that realise each anchor's CONDITION in the executable vocabulary.
#:
#: This mapping is declared rather than inferred, because inferring it was measurably wrong. A first
#: version matched on any shared word token after dropping stop-words, and it made BOTH mistakes at
#: once on the same record: A3's condition is spelled `duplicate` while the candidate that realises it
#: is `redundant_proposed_write` -- no shared token, so A3 reported NOT RECOVERED although the
#: mechanism was sitting in the library. And that very candidate shares `capacity`/`proposed` with A8,
#: so it was credited to A8 instead. A loose matcher does not fail by being vague; it fails by being
#: confidently wrong in two directions, and a coverage report is the last place that is affordable.
#:
#: Each entry is the set of signal tokens that mean the same FAILURE CONDITION as the anchor's locus
#: tail. Extending this table is the correct way to record a new spelling -- not loosening the test.
CONDITION_SIGNALS = {
    "A1": {"container_at_capacity", "no_remaining_capacity", "no_capacity"},
    "A2": {"identifier_not_found", "not_found", "zero_results", "empty_result"},
    "A3": {"duplicate", "redundant_proposed_write", "duplicate_write", "already_present"},
    "A4": {"no_tool_call_at_all", "zero_call", "no_tool_call"},
    "A5": {"no_remaining_slots", "archival_at_capacity", "destination_at_capacity"},
    "A7": {"append_would_exceed_cap", "blob_would_overflow", "size_cap_exceeded"},
    "A8": {"clear_proposed_at_capacity", "destructive_clear_proposed"},
    "A9": {"core_max_below_threshold", "similarity_below_threshold", "best_similarity"},
}

#: What the anchor's ACTION actually does, beyond firing on the right condition. Read off
#: `docs/ANCHORS.md` and the frozen round records, not inferred from the action family.
#:
#: This exists because condition-matching alone overstated a recovery. `docs/ANCHORS.md` on A8: "it
#: suppresses the clear, EVICTS one redundant copy, VERIFIES a copy remains, and RETRIES the blocked
#: write verbatim", and it "reuses A5's validated action with a genuinely new trigger" -- with 15/15
#: firings evidencing copies_remaining >= 1 and a freed slot. A candidate spelled
#: `suppress:cancel_proposed` cancels and stops. Same boundary, same action family, same condition, and
#: still a WEAKER mechanism: it prevents the destruction but performs none of the repair that made A8
#: lossless. Reporting that as "A8 recovered" would be the coverage report flattering the search.
#:
#: A3 by contrast IS cancel-only by design -- "the predicate is deliberately exact, never semantic" --
#: so for A3 a cancel_proposed candidate is a full mechanism match.
#:
#: UPDATE 2026-09-22: A8's repair half is now a REAL declared capability
#: (`suppress_clear_evict_redundant_verify_and_retry`, patches/bv/bv_dedup_clear_executor.py), whose id
#: contains evict/verify/retry -- so a candidate naming it closes the gap through the existing word
#: match, with no change to this table. The cancel-only arm measured net -5 in R5 and stays TRIGGER
#: ONLY, which is the correct verdict for it: same trigger, materially weaker action.
REQUIRED_ETA = {
    "A3": (),                                   # cancel-only by design; nothing further required
    "A8": ("evict_redundant_copy", "verify_copy_remains", "retry_blocked_write"),
    "A1": ("relocate_or_reroute", "verify_before_remove"),
    "A5": ("evict", "retry"),
    "A7": ("reduce_payload", "replace"),
    "A2": ("reprompt_toward_reachable_destination",),
    "A4": ("inject_and_regenerate",),
    "A9": ("dispatch_to_other_container", "merge_and_rank"),
}

#: Backend scope the anchor was measured on, where the record states one. A3's collisions are against
#: keys A1 injected; A8: "every observed event is kv" (docs/ANCHORS.md).
ANCHOR_BACKEND = {"A8": "kv", "A5": "vector", "A9": "vector"}


def match(row, lib):
    """Candidates whose incision, action family and failure condition all agree with this anchor.

    The condition test is a DECLARED equivalence (`CONDITION_SIGNALS`), plus the substring cases where
    the two vocabularies genuinely spell one condition at two lengths. Anything matched is printed with
    both spellings, so an equated pair is always visible and never taken on trust.

    A hit here is a TRIGGER match only. `mechanism_gap` then asks the separate question of whether the
    candidate also does what the anchor's action does -- because the two came apart in practice.
    """
    accepted_signals = CONDITION_SIGNALS.get(row["anchor"], set())
    cond = row["condition"]
    hits = []
    for r in lib.all():
        if r.identity.boundary != row["incision"] or r.identity.action != row["action"]:
            continue
        sig = str(r.identity.signal or "")
        if not sig:
            continue
        if sig in accepted_signals or (cond and (cond == sig or cond in sig or sig in cond)):
            hits.append(r)
    return hits


def mechanism_gap(anchor, record):
    """Which parts of the anchor's ACTION this candidate does not evidence. Empty tuple = full match.

    Deliberately conservative: it reports a gap whenever the candidate's own spec does not SAY it does
    the step. That can flag a candidate whose executor performs the step without naming it -- which is
    the right way round for a coverage report, because a gap invites a look at the spec while a false
    "recovered" ends the enquiry.
    """
    required = REQUIRED_ETA.get(anchor, ())
    if not required:
        return ()
    blob = json.dumps({"spec": dict(record.spec or {}),
                       "cap": record.identity.capability_id,
                       "op": record.identity.operator}).lower()
    gap = []
    for step in required:
        # A step counts as present if any of its distinctive words appears in the spec or capability.
        words = [w for w in step.split("_") if len(w) > 3]
        if not any(w in blob for w in words):
            gap.append(step)
    return tuple(gap)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=pathlib.Path,
                    default=ROOT / "rounds" / "GOLDEN" / "candidates.json")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    lib = CandidateLibrary.load(a.library)
    stack = lib.installed_stack()
    print(f"library: {len(lib)} records   installed stack: {len(stack)} "
          f"({stack_fingerprint(stack)})\n")

    rows, report = [], []
    for row in anchor_rows():
        anchor = row["anchor"]
        hits = match(row, lib)
        accepted = [h for h in hits if h.state == ACCEPTED]
        gaps = {h.name: mechanism_gap(anchor, h) for h in hits}
        full = [h for h in hits if not gaps[h.name]]
        if accepted and any(not gaps[h.name] for h in accepted):
            verdict, detail = "IN THE INCUMBENT", ", ".join(
                f"{h.name} (net {h.measurement.net:+d})" if h.measurement else h.name
                for h in accepted if not gaps[h.name])
        elif full:
            verdict, detail = "HELD, NOT ACCEPTED", ", ".join(
                f"{h.name} [{h.state}]" for h in full)
        elif hits:
            # Right trigger, weaker action. Reported as its own verdict rather than folded into
            # either "recovered" or "not recovered": both would be misleading.
            verdict, detail = "TRIGGER ONLY (weaker mechanism)", "; ".join(
                f"{h.name} [{h.state}] missing: {', '.join(gaps[h.name])}" for h in hits)
        else:
            verdict, detail = "NOT RECOVERED", ""
        rows.append((row, verdict, detail))
        report.append({**row, "verdict": verdict, "matches": detail,
                       "backend_scope": ANCHOR_BACKEND.get(anchor, "unrestricted/unrecorded"),
                       "required_eta": list(REQUIRED_ETA.get(anchor, ())),
                       "mechanism_gaps": {k: list(v) for k, v in gaps.items() if v}})

    w = max(len(r[0]["family"]) for r in rows)
    print(f"{'anchor':7s} {'family':{w}s}  verdict")
    print("-" * (9 + w + 20))
    for row, verdict, detail in rows:
        print(f"{row['anchor']:7s} {row['family']:{w}s}  {verdict}")
        if detail:
            print(f"{'':7s} {'':{w}s}    {detail}")

    n_inc = sum(1 for _, v, _ in rows if v == "IN THE INCUMBENT")
    n_held = sum(1 for _, v, _ in rows if v == "HELD, NOT ACCEPTED")
    n_trig = sum(1 for _, v, _ in rows if v.startswith("TRIGGER ONLY"))
    print(f"\n{n_inc}/8 in the installed incumbent · {n_held}/8 held pending validation · "
          f"{n_trig}/8 trigger-only · {8 - n_inc - n_held - n_trig}/8 not recovered")
    print("\nNOTE: a match requires the same incision point, action family AND failure condition, and "
          "then the anchor's own action steps (REQUIRED_ETA, read off docs/ANCHORS.md). A candidate "
          "with the right trigger and a weaker action is reported TRIGGER ONLY -- not recovered. "
          "Even a full match is not proven equivalence of threshold or executor.")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(
            {"library": str(a.library), "n_records": len(lib),
             "installed_stack": list(stack), "stack_fingerprint": stack_fingerprint(stack),
             "in_incumbent": n_inc, "held_pending": n_held,
             "not_recovered": 8 - n_inc - n_held, "rows": report,
             "matching_rule": "incision point + action family + failure condition; NOT identity_key, "
                              "because rounds/anchors.py predates capability_id and names a semantic "
                              "locus path instead"}, indent=2) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
