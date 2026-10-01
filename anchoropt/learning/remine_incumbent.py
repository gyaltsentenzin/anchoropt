#!/usr/bin/env python3
"""Re-mine the residual of a FROZEN incumbent, for the next tree iteration.

Three rules, all of which change the ranking and none of which the baseline miner had.

RULE 1 — SUPPRESS SETTLED EXPOSURES.
    An exposure the incumbent already handles is not residual. Ranking on raw support
    re-selects work already done: on the A0 line, locus L-A carried 799 historical events of
    which 280 were already covered by the accepted anchor, so raw support would have proposed
    it again at full weight. A step is settled when the incumbent's own telemetry shows an
    anchor acted on it (`*_gate` flag or an action payload), so this is read from the
    incumbent's trajectories rather than predicted.

RULE 2 — COLLAPSE REPEATED IDENTICAL FAILURES WITHIN AN EPISODE/TURN.
    A retry loop is ONE decision that went wrong, observed many times. The model retries the
    same rejected write up to the step ceiling (21 observed), so counting each attempt as
    independent support lets a single stuck turn outrank a locus that fails once in each of
    twenty different episodes. Support therefore counts DISTINCT (episode, turn, failure
    identity) triples, where failure identity is the semantic locus plus the normalised call
    shape -- so a genuine second, different failure in the same turn still counts.

RULE 3 — RETRIES ARE SEVERITY, NOT SUPPORT.
    Collapsing must not throw the retry count away: a locus that traps a turn for 21 steps is
    worse than one the model recovers from immediately. Retry depth and linked downstream loss
    are carried as SEVERITY fields, used to break ties and to describe cost, and are never
    added back into support.

Ranking stays LINKED DOWNSTREAM LOSS first, as frozen. Severity only orders ties.

Usage:
    python scripts/remine_incumbent.py \
        --incumbent results/n0a1v2_train/anchor \
        --artifact  ledgers/n1_locus.json --iteration N0-T2
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import constraint_locus as det  # noqa: E402
import op_canonicalize_locus as oc  # noqa: E402

_ARG = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])(.*?)\2", re.S)
_CALL = re.compile(r"([a-z_][a-z_0-9]*)\s*\(")
_TOK = re.compile(r"[a-z0-9]{4,}")
# Steps to look ahead for a successful write of the same content.
_RESOLVE_WINDOW = 4

# Telemetry that means "an anchor acted on this step". Suffix + prefixes, mirroring the
# sidecar's own families so a new anchor mechanism is settled-aware without editing this.
_ACTED_PREFIXES = ("transform_", "suppress_", "reroute_", "reprompt_", "capacity_repair_",
                   "preempt_")


def chain_of(cid: str) -> str:
    """Chain key (`memory_kv-healthcare`), or "?" if the id does not parse.

    Delegates to the active benchmark adapter -- this was a SEVENTH copy of the same grammar. The "?"
    sentinel is preserved because it is used as a grouping label in this module's reports; note it is
    a third distinct convention (beside `None` and the old silent `rec_sum`), which is exactly the
    inconsistency the shared adapter exists to end. Do not add a fourth.
    """
    from anchoropt.attribution import active_adapter
    return active_adapter().chain_of(cid)[0].replace("unknown", "?")


def toks(s: str) -> set:
    return set(_TOK.findall((s or "").lower()))


def _acted(step: Dict) -> bool:
    """Did an incumbent anchor act on this step? (RULE 1)"""
    for k, v in step.items():
        if k.endswith("_gate") and v:
            return True
        if k.startswith(_ACTED_PREFIXES) and v not in (None, "", False):
            return True
    return False


def _call_shape(call: str) -> str:
    """Normalised call identity: tool name + sorted arg names, values dropped.

    Values are dropped deliberately. A retry loop re-sends *slightly different* text each
    attempt, so keying on values would treat 21 attempts at the same decision as 21 distinct
    failures -- exactly what RULE 2 exists to prevent. Arg NAMES are kept so
    `memory_append(text=)` and `memory_update(text=)` stay distinguishable.
    """
    m = _CALL.search(str(call) or "")
    name = m.group(1) if m else "?"
    args = sorted({k for k, _q, _v in _ARG.findall(str(call))})
    return "%s(%s)" % (name, ",".join(args))



def _occurrence_resolved(steps, i, call, args) -> bool:
    """Was THIS occurrence resolved -- i.e. did its content actually get stored?

    RULE 1 previously asked `_acted(step)`: did ANY gate fire anywhere in this step. That is
    step-level, and it mislabels in both directions -- a resolved occurrence stays counted as
    unresolved support when the telemetry flag lands on a neighbouring step, and a trajectory
    that still fails keeps being attributed to the ORIGINAL locus even after an anchor fixed it.
    Measured on the clean incumbent: 336 of 718 `capacity` occurrences (47%) had their content
    successfully stored, yet all 718 were still labelled `capacity`, while the actual remaining
    cause in those episodes was elsewhere (size/item 73, duplicate 59, format 22, not_found 18).
    That is the stale-label defect.

    Occurrence-level test, deliberately evidence-based rather than telemetry-based: the content
    counts as stored if an anchor substitution in this step carries it, or if any successful
    write within a short window carries it. Content match, not call match, because the whole
    point of a reroute is that the call changes while the content does not.
    """
    want = toks(args.get("value") or args.get("text") or "")
    if not want:
        return False
    s = steps[i]
    for fld in ("reroute_substituted_call", "capacity_repair_substituted_call"):
        sub = s.get(fld)
        if sub:
            sa = {k: v for k, _q, v in _ARG.findall(str(sub))}
            if toks(sa.get("value") or sa.get("text") or "") & want:
                return True
    for s2 in steps[i:i + _RESOLVE_WINDOW]:
        dec2 = s2.get("decoded") or []
        res2 = s2.get("tool_results") or []
        for j2, c2 in enumerate(dec2):
            rr = str(res2[j2]) if j2 < len(res2) else ""
            if "error" in rr.lower():
                continue
            m = _CALL.search(str(c2))
            if not m or not any(v in m.group(1) for v in ("add", "append", "update")):
                continue
            a2 = {k: v for k, _q, v in _ARG.findall(str(c2))}
            if toks(a2.get("value") or a2.get("text") or "") & want:
                return True
    return False


def mine(incumbent: Path, artifact: Path, iteration: str,
         use_llm: bool = True, overwrite: bool = False) -> Tuple[List[Dict], Dict]:
    traj = incumbent / "traj"
    results = incumbent / "eval_train_results.json"
    valid = {r["id"]: bool(r.get("valid"))
             for r in json.load(open(results))["results"] if not r.get("is_prereq")}
    n_fail = sum(1 for v in valid.values() if not v)

    # Freeze the semantic locus artifact for THIS world (own fingerprint, own iteration).
    obs = oc.observations_from_trajectories(traj)
    art = oc.canonicalize(obs, artifact, iteration, enabled=use_llm, overwrite=overwrite)

    # Failing-query tokens per chain, for linked downstream loss.
    fails: Dict[str, List] = collections.defaultdict(list)
    for p in sorted((traj / "query").glob("*.json")):
        try:
            ep = json.load(open(p))
        except Exception:
            continue
        if valid.get(ep.get("case_id"), True):
            continue
        w = set()
        for s in ep.get("steps") or []:
            for c in s.get("decoded") or []:
                for m in re.finditer(r"(?:query|key|text)\s*=\s*['\"]([^'\"]+)['\"]", str(c)):
                    w |= toks(m.group(1))
        fails[chain_of(ep.get("case_id"))].append((ep.get("case_id"), w))

    # ── the three rules ──────────────────────────────────────────────────────
    support: Dict[Tuple, set] = collections.defaultdict(set)   # distinct (ep,turn,identity)
    raw_events: Dict[Tuple, int] = collections.Counter()       # every occurrence
    settled: Dict[Tuple, int] = collections.Counter()          # RULE 1
    retry_depth: Dict[Tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    blocked: Dict[Tuple, Dict[str, set]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    backends: Dict[Tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    # Per-trajectory ledger: how many errors of each locus this episode had CORRECTED
    # by the incumbent. Answers "how many errors have we fixed in this trajectory".
    corrected: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    for phase in ("prereq", "query"):
        for p in sorted((traj / phase).glob("*.json")):
            try:
                ep = json.load(open(p))
            except Exception:
                continue
            cid = ep.get("case_id")
            ch = chain_of(cid)
            be = ep.get("backend")
            _steps = ep.get("steps") or []
            for _si, s in enumerate(_steps):
                dec = s.get("decoded") or []
                res = s.get("tool_results") or []
                for j, call in enumerate(dec):
                    r = res[j] if j < len(res) else ""
                    args = {k: v for k, _q, v in _ARG.findall(str(call))}
                    k = art.locus_of(r, args)
                    if not k:
                        continue
                    raw_events[k] += 1
                    # RULE 1, occurrence-level. `initial_locus` (k) is retained for lineage and
                    # saturation accounting; `status` is resolved vs unresolved HERE; and only
                    # unresolved occurrences contribute support, so ranking reflects the current
                    # policy world rather than stale occurrences of an already-handled locus.
                    if _occurrence_resolved(_steps, _si, call, args) or _acted(s):
                        settled[k] += 1
                        corrected[cid][ "/".join(k) ] += 1   # per-trajectory correction ledger
                        continue
                    ident = (cid, s.get("turn"), _call_shape(call))
                    support[k].add(ident)        # RULE 2: collapse retries
                    retry_depth[k][ident] += 1   # RULE 3: keep depth as severity
                    backends[k][be] += 1
                    blocked[k][ch] |= toks(" ".join(args.values()))

    rows = []
    # Iterate every locus SEEN, not only those with residual support. A locus the incumbent
    # fully resolved has no `support` entry, and keying the loop on support.items() dropped it
    # from the output entirely -- so `settled`/`raw_events` for a completely-fixed locus became
    # unreportable and saturation accounting silently lost it. The initial_locus must survive
    # for lineage even at support=0.
    for k in sorted(set(raw_events) | set(support), key=lambda x: tuple(x)):
        idents = support.get(k, set())
        linked = set()
        for ch, bt in blocked[k].items():
            for qid, w in fails.get(ch, []):
                if w & bt:
                    linked.add(qid)
        depths = sorted(retry_depth[k].values(), reverse=True)
        # SUPPRESS is a candidate remedy wherever the model REPEATS an identical failing
        # action after the same error. Collapsing retries is the mining half of that
        # observation; this is the POLICY half, and it was missing. BFCL allows up to 21
        # attempts per turn, so a known-failing retry does not merely waste compute -- it
        # consumes an attempt a feasible alternative could have used. Suppression is only
        # worth testing where BOTH hold: the repeats are frequent, AND an attested
        # alternative was left untried in those same episodes (freeing attempts is useless
        # if nothing can use them).
        _rep = [dep for dep in retry_depth[k].values() if dep >= 2]
        wasted = sum(dep - 1 for dep in _rep)
        rows.append({
            "locus": list(k),
            "support": len(idents),                  # collapsed, RULE 2
            "raw_events": raw_events[k],
            "settled_by_incumbent": settled[k],      # RULE 1
            "linked_loss": len(linked),              # primary ranking key
            "max_retry_depth": depths[0] if depths else 0,   # RULE 3 severity
            "repeat_runs": len(_rep),          # decisions the model retried identically
            "wasted_attempts": wasted,         # attempts a suppression would reclaim
            "suppress_candidate": len(_rep) >= 3 and wasted >= 10,
            "mean_retry_depth": round(sum(depths) / len(depths), 1) if depths else 0.0,
            "backends": dict(backends[k]),
        })
    # linked loss first (frozen); severity only breaks ties.
    rows.sort(key=lambda r: (-r["linked_loss"], -r["max_retry_depth"], -r["support"]))
    # Per-trajectory correction counts, so "how many errors did we fix here" is answerable
    # per episode rather than only in aggregate.
    ledger = {cid: dict(c) for cid, c in corrected.items() if c}
    meta = {"n_failing_queries": n_fail, "artifact": str(artifact),
            "corrections_by_episode": ledger,
            "episodes_with_corrections": len(ledger),
            "total_corrections": sum(sum(c.values()) for c in corrected.values()),
            "fingerprint": art.data.get("fingerprint"), "model": art.data.get("model"),
            "iteration": iteration}
    return rows, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", required=True, type=Path,
                    help="the FROZEN incumbent arm dir (its results are the no-op control)")
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--iteration", default="N0-T2")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    rows, meta = mine(a.incumbent, a.artifact, a.iteration,
                      use_llm=not a.no_llm, overwrite=a.overwrite)
    print("incumbent : %s" % a.incumbent)
    print("residual  : %d failing queries" % meta["n_failing_queries"])
    print("artifact  : %s  fp=%s  model=%s\n"
          % (a.artifact.name, meta["fingerprint"], meta["model"]))
    print("%-4s %-42s %7s %7s %8s %8s %7s  %s"
          % ("rank", "semantic locus", "support", "linked", "settled", "raw_ev",
             "maxdep", "backends"))
    for i, r in enumerate(rows, 1):
        print("%-4d %-42s %7d %7d %8d %8d %7d  %s"
              % (i, "/".join(r["locus"])[:42], r["support"], r["linked_loss"],
                 r["settled_by_incumbent"], r["raw_events"], r["max_retry_depth"],
                 ",".join("%s:%d" % kv for kv in sorted(r["backends"].items()))))
    print("\nsupport = DISTINCT (episode, turn, call-shape); retries collapsed.")
    print("settled = occurrences an incumbent anchor already acted on (excluded from support).")
    print("maxdep  = deepest retry loop, carried as SEVERITY only -- never added to support.")
    if rows:
        print("\n=> MINER'S A2 CANDIDATE: %s" % "/".join(rows[0]["locus"]))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"meta": meta, "ranked": rows}, open(a.json, "w"), indent=2)
        print("wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
