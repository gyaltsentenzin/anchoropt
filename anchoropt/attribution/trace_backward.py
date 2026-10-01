#!/usr/bin/env python3
"""Backward causal trace across a multi-stage trajectory (SIGNAL_POLICY_PLAN A2).

The per-episode proximity view in `error_attribution_study.py --mine-traj-dir`
answers "which call reacts to a tool error 1-2 steps earlier". That is enough to
recover G1's hook, but it is blind to the failure mode that actually matters here:

  a *storage-phase* action corrupts memory state, and the damage surfaces in a
  *different, later episode* of the same chain, as a query that cannot be answered.

Nothing local to either episode reveals that link. This script reconstructs it by
walking the chain backward from the failing query:

  1. group episodes into chains (backend x scenario), ordered prereq_0..prereq_N then queries
  2. for each FAILING query, extract the fact/key it needed
  3. walk backward through the chain's storage episodes to the step that last
     touched that fact — the *divergence step*
  4. classify what happened there (never stored / stored then cleared / rejected
     by a tool contract / stored under a mangled key)

Step 4's classes are the actionable output: each names a different remedy, and
"stored then cleared" is precisely the G1 cascade.

Chain grouping mirrors `_group_prereqs_by_chain` in memory_evaluator.py, but is
reimplemented over the sidecar JSON so this needs no eval stack.

Usage:
    python scripts/trace_backward.py --traj-dir <dir> [--json out.json] [--show N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness_guard as _hg  # noqa: E402  shared fault guard (plan §105)

_KEY_RE = re.compile(r"key\s*=\s*['\"]([^'\"]+)['\"]")
# KEYLESS writes. Only kv writes carry key=; vector and rec_sum write content directly, so a
# key-based accounting is structurally blind to them -- 837 write calls on the A0 corpus, and
# every failing vector/rec_sum query was therefore mislabelled `never_stored` (plan §T2).
#
#   rec_sum  memory_append(text=...)            / memory_update(text=...)
#   vector   core_memory_add(text=...)          -> {"id": N}
#            archival_memory_add(text=...)      -> {"id": N}
#            core_memory_update(vec_id=N, new_text=...)
#
# Identity for these is the CONTENT, plus the vec_id handle where the tool returns one.
_TEXT_RE = re.compile(r"(?:new_text|text)\s*=\s*(['\"])(.*?)\1", re.S)
_RETID_RE = re.compile(r'"id"\s*:\s*(\d+)')
_VAL_RE = re.compile(r"value\s*=\s*['\"]([^'\"]*)['\"]")
_CALL_RE = re.compile(r"(?:^|[\[\(\"',;]|\s)\s*([a-z][a-z0-9_]*)\s*\(")
# Error-ish markers and the case-id grammar both live in the benchmark adapter now. Imported lazily
# so `anchoropt/` keeps no import-time dependency on `benchmarks/` -- the core must stay loadable
# without a benchmark on the path.
def _adapter():
    """The active benchmark adapter, supplied by a benchmark entry point.

    The core does not know which benchmark it is running against, nor where that adapter lives.
    Raises if none is registered rather than guessing at one.
    """
    from anchoropt.attribution import active_adapter
    return active_adapter()


_ERRISH = ("error", "not found", "full", "cannot", "fail", "too long",
           "exceed", "invalid", "unable", "must be")

UNKNOWN_CHAIN = "unknown"


def _chain_of(case_id: str) -> Tuple[str, int]:
    """(chain key, ordinal). Chain = backend x scenario; prereqs sort before queries.

    The case-id grammar lives in the benchmark adapter, not here -- it was reimplemented in five
    places and two copies silently filed unknown ids under `rec_sum`.

    TOLERANT ON PURPOSE, AND BOUNDED. This returns `(UNKNOWN_CHAIN, 0)` rather than raising, because
    it also SORTS trajectory files and a directory can legitimately contain non-corpus artifacts.
    But an unparseable id must never reach a statistic: `_group_into_chains` refuses to report one,
    so the tolerance is confined to ordering. `adapter.chain_of_strict` is the raising form for
    call sites where a foreign id is a bug.
    """
    return _adapter().chain_of(case_id)


def _group_into_chains(episodes) -> Tuple[dict, list]:
    """`(chains, skipped)` -- and the split is the point.

    Anything whose case id does not parse is SEPARATED OUT rather than bucketed under
    `UNKNOWN_CHAIN` and carried into `trace_chain`. Before this guard existed, a stray file would
    have been grouped, traced, and folded into the reported per-chain survival numbers with nothing
    in the output saying so. Tolerating an unreadable id while ORDERING files is reasonable;
    tolerating one while COUNTING is how a statistic quietly stops meaning what it says.
    """
    from collections import defaultdict

    chains = defaultdict(list)
    skipped = []
    for episode in episodes:
        key = _chain_of(episode.get("case_id", ""))[0]
        if key == UNKNOWN_CHAIN:
            skipped.append(episode.get("case_id", "<no case_id>"))
            continue
        chains[key].append(episode)
    return chains, skipped


def load(traj_dir: Path, allow_harness_faults: bool = False) -> List[Dict]:
    eps = []
    for p in sorted(traj_dir.rglob("*.json")):
        try:
            eps.append(json.load(open(p)))
        except Exception as e:
            print(f"[warn] skip {p.name}: {e}", file=sys.stderr)
    # A crash inside a tool implementation is not a model failure. Backward tracing
    # would otherwise nominate the crashing call as the divergence step and blame a
    # harness bug for the query it "caused" to fail (plan §105).
    return _hg.assert_clean(eps, source=str(traj_dir), allow=allow_harness_faults)


def _steps_with_calls(ep: Dict):
    """Yield (idx, call_name, key, value, tool_result, errored)."""
    for i, s in enumerate(ep.get("steps", [])):
        dec = str(s.get("decoded") or "")
        tr = str(s.get("tool_results") or "")
        errored = any(k in tr.lower() for k in _ERRISH)
        for m in _CALL_RE.finditer(dec):
            name = m.group(1)
            if "_" not in name:
                continue
            key = _KEY_RE.search(dec)
            val = _VAL_RE.search(dec)
            # Fall back to the keyless shapes so vector/rec_sum writes are visible at all.
            # `key` becomes a synthetic handle (vec:N or ret:N) when the tool supplies one,
            # and `val` carries the content either way -- which is what token matching needs.
            _txt = _TEXT_RE.search(dec)
            _vid = _adapter().target_id(dec)          # entry handle, backend-specific
            _rid = _RETID_RE.search(tr)
            _k = key.group(1) if key else (
                ("vec:%s" % _vid) if _vid else
                ("ret:%s" % _rid.group(1)) if _rid else None)
            _v = val.group(1) if val else (_txt.group(2) if _txt else None)
            yield (i, name, _k, _v, tr, errored)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _tokens(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]{4,}", (s or "").lower())}


def trace_chain(chain_key: str, eps: List[Dict], show: int = 0) -> List[Dict]:
    """For each failing query in the chain, walk backward to the divergence step."""
    storage = sorted([e for e in eps if e.get("phase") == "prereq"],
                     key=lambda e: _chain_of(e["case_id"])[1])
    queries = sorted([e for e in eps if e.get("phase") == "query"],
                     key=lambda e: _chain_of(e["case_id"])[1])

    # Flatten the chain's storage history once: what was written, cleared, rejected.
    writes: List[Dict] = []
    clears: List[Dict] = []
    rejects: List[Dict] = []
    for ep in storage:
        for (i, name, key, val, tr, errored) in _steps_with_calls(ep):
            rec = {"episode": ep["case_id"], "step": i, "call": name,
                   "key": key, "value": (val or "")[:120], "errored": errored,
                   "result": tr[:120]}
            if "clear" in name or "remove" in name:
                clears.append(rec)
            elif errored:
                rejects.append(rec)
            elif any(v in name for v in ("add", "append", "update", "insert")):
                writes.append(rec)

    # ── What the chain's memory actually looked like AT QUERY TIME ────────────
    # This is the measurement that makes the trace conclusive, and it needs no
    # key matching at all: the model's own list/retrieve calls during the query
    # phase report the surviving state. Comparing that to what storage wrote
    # gives the survival rate directly.
    visible: set = set()
    archival_listings = archival_empty = 0
    for q in queries:
        for i, s in enumerate(q.get("steps", [])):
            dec = str(s.get("decoded") or "")
            tr = str(s.get("tool_results") or "")
            if _adapter().is_read_all(dec, "core"):
                visible |= set(re.findall(r'"([a-z][a-z0-9_]{3,})"\s*:', tr))
                m = re.search(r'"keys"\s*:\s*\[(.*?)\]', tr)
                if m:
                    visible |= set(re.findall(r'"([a-z][a-z0-9_]{3,})"', m.group(1)))
            if _adapter().is_list_keys(dec, "archival"):
                archival_listings += 1
                if re.search(r'"keys"\s*:\s*\[\s*\]', tr):
                    archival_empty += 1
    written_keys = {w["key"] for w in writes if w["key"]}
    survived = written_keys & visible
    lost = written_keys - visible
    # CONTENT-based survival, for backends whose stores have no key names. `visible` above is
    # built from key listings, which keyless backends never produce, so it stays empty for
    # them and every write would read as "lost". Instead compare the tokens the storage phase
    # WROTE against the tokens any query-phase tool result RETURNED.
    q_result_tokens: set = set()
    for q in queries:
        for s2 in q.get("steps", []):
            q_result_tokens |= _tokens(str(s2.get("tool_results") or ""))
    written_tokens: set = set()
    for w in writes:
        written_tokens |= _tokens(w.get("value") or "")
    content_survived = written_tokens & q_result_tokens
    content_lost = written_tokens - q_result_tokens

    findings = []
    for q in queries:
        if (q.get("outcome") or {}).get("valid") is not False:
            continue
        # What did the query try to retrieve / talk about?
        #
        # Query-time search strings are natural language ('diabetes diagnosis'),
        # while stored keys are snake_case identifiers (diabetes_management_habits),
        # so exact key equality almost never fires and an exact-match classifier
        # mislabels every failure as never_stored. Match on token overlap instead,
        # and pull tokens from the search QUERY argument too, not just key=/value=.
        want_keys, want_tok = set(), set()
        for (i, name, key, val, tr, errored) in _steps_with_calls(q):
            if key:
                want_keys.add(key)
            want_tok |= _tokens(val or "")
        for s in q.get("steps", []):
            dec = str(s.get("decoded") or "")
            for m in re.finditer(r"query\s*=\s*['\"]([^'\"]+)['\"]", dec):
                want_tok |= _tokens(m.group(1))
        ans = (q.get("outcome") or {}).get("error_message", "") or ""
        want_tok |= _tokens(ans)

        # Walk backward: most recent relevant storage event first.
        def relevant(rec):
            if rec["key"] and want_keys:
                if any(_norm(rec["key"]) == _norm(k) for k in want_keys):
                    return 3
            score = 0
            if rec["key"] and want_tok and _tokens(rec["key"]) & want_tok:
                score = 2
            elif want_tok and _tokens(rec["value"]) & want_tok:
                score = 1
            return score

        cand = []
        for pool, kind in ((clears, "cleared"), (rejects, "rejected"), (writes, "stored")):
            for rec in pool:
                r = relevant(rec)
                if r:
                    cand.append((r, kind, rec))
        cand.sort(key=lambda t: (-t[0],))

        if not cand:
            cls = "never_stored"
            evidence = None
        else:
            _, kind, rec = cand[0]
            # Classify against MEASURED survival, not against a guess. A key that
            # storage wrote successfully but which is absent from the model's own
            # query-time listing was destroyed between the phases — and an
            # unqualified core_memory_clear in the chain is the mechanism.
            was_written = rec["key"] in written_keys if rec["key"] else False
            is_gone = rec["key"] in lost if rec["key"] else False
            if kind == "rejected":
                cls = "rejected_by_tool_contract"
            elif was_written and is_gone and clears:
                cls = "stored_then_cleared"
            elif was_written and is_gone:
                cls = "stored_then_lost"
            elif kind == "cleared":
                cls = "stored_then_cleared"
            else:
                cls = "stored_but_not_retrieved"
            evidence = rec

        findings.append({"query": q["case_id"], "chain": chain_key,
                         "class": cls, "divergence": evidence,
                         "wanted_keys": sorted(want_keys)[:4]})

    # Chain-level survival summary — the headline number, independent of any
    # per-query classification.
    survival = {"keys_written": len(written_keys), "keys_visible_at_query": len(survived),
                "keys_lost": len(lost), "clear_calls": len(clears),
                "writes_rejected": len(rejects),
                "archival_listings": archival_listings,
                "archival_empty": archival_empty,
                "lost_sample": sorted(lost)[:10],
                # Keyless backends: report CONTENT survival and say so, rather than
                # reporting 0 keys written and letting a reader infer nothing was stored.
                "write_calls": len(writes),
                "keyed": bool(written_keys),
                "content_tokens_written": len(written_tokens),
                "content_tokens_seen_at_query": len(content_survived),
                "content_survival": (round(len(content_survived) / len(written_tokens), 3)
                                     if written_tokens else None)}
    return findings, survival


REMEDY_HINT = {
    "stored_then_cleared":      "suppress the clearing call (G1 shape)",
    "rejected_by_tool_contract": "fix the contract violation at write time "
                                 "(reprompt/reroute with a conforming key)",
    "never_stored":             "storage never attempted — upstream planning failure",
    "stored_but_not_retrieved": "retrieval/synthesis failure — no storage-phase remedy",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--show", type=int, default=6, help="print N example traces")
    _hg.add_cli_flag(ap)
    a = ap.parse_args()

    eps = load(a.traj_dir, allow_harness_faults=a.allow_harness_faults)
    if not eps:
        print(f"No episodes under {a.traj_dir}", file=sys.stderr)
        return 1

    chains, skipped = _group_into_chains(eps)
    if skipped:
        # Loud, and excluded from every number below. Silence here would mean the survival table
        # includes episodes whose backend and scenario are unknown.
        print(f"   ⚠ {len(skipped)} episode(s) EXCLUDED -- case id does not parse: "
              f"{skipped[:4]}{' ...' if len(skipped) > 4 else ''}", file=sys.stderr)

    npre = sum(1 for e in eps if e.get("phase") == "prereq")
    print(f"corpus: {len(eps)} episodes (prereq={npre}, query={len(eps)-npre}) "
          f"in {len(chains)} chains")

    all_f = []
    surv_by_chain = {}
    for ck in sorted(chains):
        f, sv = trace_chain(ck, chains[ck])
        all_f.extend(f)
        surv_by_chain[ck] = sv

    # ── Survival: the cross-episode measurement, independent of classification ──
    print("\n-- storage -> query survival (per chain) --")
    print(f"   {'chain':26s} {'written':>7} {'visible':>7} {'lost':>5} "
          f"{'clears':>6} {'rejects':>7}  archival_empty")
    tw = tv = tl = tc = tr_ = 0
    for ck, sv in sorted(surv_by_chain.items()):
        tw += sv["keys_written"]; tv += sv["keys_visible_at_query"]
        tl += sv["keys_lost"]; tc += sv["clear_calls"]; tr_ += sv["writes_rejected"]
        ae = f"{sv['archival_empty']}/{sv['archival_listings']}"
        print(f"   {ck:26s} {sv['keys_written']:7d} {sv['keys_visible_at_query']:7d} "
              f"{sv['keys_lost']:5d} {sv['clear_calls']:6d} {sv['writes_rejected']:7d}  {ae}")
    rate = (tv / tw * 100) if tw else 0.0
    print(f"   {'TOTAL':26s} {tw:7d} {tv:7d} {tl:5d} {tc:6d} {tr_:7d}")
    print(f"\n   survival rate: {tv}/{tw} = {rate:.1f}% of successfully-written keys "
          f"are still visible at query time")
    if tl and tc:
        print(f"   ⚠ {tl} keys written OK then lost, with {tc} core_memory_clear calls "
              f"in the storage phase — the G1 cascade, across episodes.")

    if not all_f:
        print("\nNo failing query episodes with a traceable chain yet.")
        return 0

    from collections import Counter
    cc = Counter(f["class"] for f in all_f)
    print(f"\n-- backward trace over {len(all_f)} failing queries --")
    for cls, n in cc.most_common():
        print(f"   {n:4d}  {cls:28s} -> {REMEDY_HINT.get(cls,'?')}")

    # Per-chain concentration: a class confined to one backend is a backend-scoped gate.
    by_chain = defaultdict(Counter)
    for f in all_f:
        by_chain[f["chain"]][f["class"]] += 1
    print("\n-- by chain --")
    for ch in sorted(by_chain):
        parts = " ".join(f"{k}={v}" for k, v in by_chain[ch].most_common())
        print(f"   {ch:26s} {parts}")

    shown = 0
    for f in all_f:
        if shown >= a.show or not f["divergence"]:
            continue
        d = f["divergence"]
        print(f"\n   QUERY {f['query']}  [{f['class']}]")
        print(f"     wanted keys: {f['wanted_keys']}")
        print(f"     divergence : {d['episode']} step {d['step']} "
              f"{d['call']}(key={d['key']!r})")
        print(f"     result     : {d['result'][:110]}")
        shown += 1

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_findings": len(all_f), "classes": dict(cc),
                   "survival_by_chain": surv_by_chain,
                   "findings": all_f}, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
