#!/usr/bin/env python3
"""Stage 2 of mining: given a failure FAMILY, find the decision point that could have changed it.

    semantic miner      -> WHICH failure family   (constraint_locus / op_canonicalize_locus)
    attribution miner   -> WHERE it was caused    (this module)
    policy search       -> WHICH action, at that point

The semantic miner names the family from the observed error, which is a statement about the
SYMPTOM. That is not where an anchor goes. Measured case that forced this stage into existence:

    locus `existence/identifier/not_found` is raised 43/44 times by core_memory_retrieve --
    a READ symptom. But 39 of those 44 failed reads are for facts that ARE present in
    archival, put there by the accepted A1 anchor when core filled up. The write succeeded;
    the read looked in the wrong container. Anchoring on the symptom would attach a remedy to
    the reader without ever asking whether the writer or the reader was wrong.

So a failure family decomposes into ATTRIBUTION CLASSES, each naming a different stage and
decision point, and each admitting different actions:

    unreachable_but_stored   the fact exists in another container -> READ side, redirect the
                             lookup. Often SELF-INFLICTED by an earlier accepted anchor.
    never_stored             no write was attempted -> WRITE side, upstream planning
    write_rejected           the write was attempted and refused -> WRITE side, repair
    stored_then_destroyed    written, then cleared/removed -> WRITE side, suppress
    unattributed             no write-side explanation found -> read/synthesis, or a gap in
                             this classifier; reported, never silently dropped

The stage is DERIVED (which phase the causing step is in), not declared, so a family that
turns out to be caused on the other side of the pipeline than its symptom says is visible
rather than assumed.

Benchmark-agnostic: containers, write verbs and read verbs are supplied as configuration, not
hardcoded. What is generic is the question -- "was the needed value ever written, is it
somewhere else, and which step decided that".

Usage:
    python scripts/attribution_miner.py --incumbent results/n0a1v2_train/anchor \
        --artifact ledgers/n0t2_locus.json [--locus existence/identifier/not_found]
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

import op_canonicalize_locus as oc  # noqa: E402

_ARG = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])(.*?)\2", re.S)
_CALL = re.compile(r"([a-z_][a-z_0-9]*)\s*\(")
_TOK = re.compile(r"[a-z0-9]{4,}")

# ── Benchmark configuration. The ONLY place tool vocabulary appears. ─────────
# `containers` maps a container name to the calls that write into it, so "is the value
# somewhere else" is answerable without the classifier knowing what a container means.
CONFIG = {
    "write_verbs": ("add", "append", "update", "insert", "replace"),
    "read_verbs": ("retrieve", "search", "list", "get"),
    "destroy_verbs": ("clear", "remove", "delete"),
    "containers": {
        "primary": ("core_memory_add", "core_memory_update", "memory_append", "memory_update"),
        "secondary": ("archival_memory_add",),
    },
}

STAGE_WRITE, STAGE_READ = "write", "read"


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


def _call_name(c) -> Optional[str]:
    m = _CALL.search(str(c) or "")
    return m.group(1) if m else None


def _kind(name: str) -> Optional[str]:
    if not name:
        return None
    for v in CONFIG["destroy_verbs"]:
        if v in name:
            return "destroy"
    for v in CONFIG["write_verbs"]:
        if v in name:
            return "write"
    for v in CONFIG["read_verbs"]:
        if v in name:
            return "read"
    return None


def _container_of(name: str) -> Optional[str]:
    for cont, calls in CONFIG["containers"].items():
        if name in calls:
            return cont
    return None


def attribute(incumbent: Path, artifact: Path,
              only_locus: Optional[Tuple] = None) -> Tuple[List[Dict], Dict]:
    """For each occurrence of a locus, name the decision point that could have changed it."""
    art = oc.LocusArtifact.load(artifact)
    if art is None:
        raise SystemExit("no locus artifact at %s -- run the semantic miner first" % artifact)
    traj = incumbent / "traj"

    # Pass 1: per chain, what was written where, and what was destroyed.
    written: Dict[str, Dict[str, set]] = collections.defaultdict(
        lambda: collections.defaultdict(set))          # chain -> container -> tokens
    written_keys: Dict[str, Dict[str, set]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    destroyed: Dict[str, int] = collections.Counter()
    anchor_wrote: Dict[str, int] = collections.Counter()   # writes an anchor caused

    for phase in ("prereq", "query"):
        for p in sorted((traj / phase).glob("*.json")):
            try:
                ep = json.load(open(p))
            except Exception:
                continue
            ch = chain_of(ep.get("case_id"))
            for s in ep.get("steps") or []:
                acted = any(k.endswith("_gate") and v for k, v in s.items())
                dec = s.get("decoded") or []
                res = s.get("tool_results") or []
                for j, c in enumerate(dec):
                    name = _call_name(c)
                    kind = _kind(name or "")
                    r = str(res[j]) if j < len(res) else ""
                    errored = '"error"' in r or "not found" in r.lower()
                    args = {k: v for k, _q, v in _ARG.findall(str(c))}
                    if kind == "destroy":
                        destroyed[ch] += 1
                    elif kind == "write" and not errored:
                        cont = _container_of(name) or "primary"
                        written[ch][cont] |= toks(" ".join(args.values()))
                        if args.get("key"):
                            written_keys[ch][cont].add(args["key"])
                        if acted:
                            anchor_wrote[ch] += 1

    # Pass 2: classify each locus occurrence.
    rows: List[Dict] = []
    for phase in ("prereq", "query"):
        for p in sorted((traj / phase).glob("*.json")):
            try:
                ep = json.load(open(p))
            except Exception:
                continue
            cid = ep.get("case_id")
            ch = chain_of(cid)
            for s in ep.get("steps") or []:
                dec = s.get("decoded") or []
                res = s.get("tool_results") or []
                for j, c in enumerate(dec):
                    r = res[j] if j < len(res) else ""
                    args = {k: v for k, _q, v in _ARG.findall(str(c))}
                    k = art.locus_of(r, args)
                    if not k or (only_locus and tuple(k) != tuple(only_locus)):
                        continue
                    name = _call_name(c) or "?"
                    kind = _kind(name)
                    want = toks(" ".join(args.values()))
                    here = _container_of(name)

                    # Where else could the wanted value be?
                    elsewhere = [cont for cont, tk in written[ch].items()
                                 if cont != here and want and (want & tk)]
                    if kind == "write":
                        cls, stage = "write_rejected", STAGE_WRITE
                    elif elsewhere:
                        cls, stage = "unreachable_but_stored", STAGE_READ
                    elif destroyed[ch] and want and any(
                            want & tk for tk in written[ch].values()):
                        cls, stage = "stored_then_destroyed", STAGE_WRITE
                    elif want and not any(want & tk for tk in written[ch].values()):
                        cls, stage = "never_stored", STAGE_WRITE
                    else:
                        cls, stage = "unattributed", "?"
                    rows.append({
                        "locus": list(k), "case_id": cid, "chain": ch, "phase": phase,
                        "failing_call": name, "class": cls, "stage": stage,
                        "value_found_in": elsewhere,
                        # Self-inflicted = the value sits where an ANCHOR put it.
                        "self_inflicted": bool(elsewhere) and anchor_wrote[ch] > 0,
                        "candidate_incision_points": admissible_points(cls)[0],
                        "live_cells": ["%s+%s" % c for c in feasible_cells(cls, stage)[0]],
                        "incision_feasibility": admissible_points(cls)[1],
                    })
    meta = {"artifact": str(artifact), "fingerprint": art.data.get("fingerprint"),
            "n_occurrences": len(rows)}
    return rows, meta


REMEDY = {
    "unreachable_but_stored": "READ side: redirect the lookup to the container holding it",
    "never_stored":           "WRITE side: no write attempted -- upstream planning",
    "write_rejected":         "WRITE side: repair the refused write",
    "stored_then_destroyed":  "WRITE side: suppress the destroying call",
    "unattributed":           "no write-side explanation -- read/synthesis, or a classifier gap",
}

PRE_GEN, PRE_EXEC, POST_EXEC = "pre_generation", "post_generation_pre_execution", "post_execution"

# ── CANDIDATE INCISION POINTS: admissible given the attribution class ────────
# ATTRIBUTION CONSTRAINS THE SEARCH SPACE; IT DOES NOT SOLVE THE OPTIMISATION.
#
# An anchor is a choice on three axes (stage x incision point x action family). Attribution can
# say WHERE a failure arose and therefore which points are *logically capable* of changing it.
# It must not say which point to use -- that is a policy question answered by counterfactual
# measurement, and an earlier version of this table got it wrong twice over: it called the
# points "derived from the attribution class" and stored them as ORDERED tuples, so any
# consumer reading element [0] would inherit a preference attribution never earned.
#
# The three points are a GENERIC INTERVENTION COORDINATE SYSTEM, not labels attached per error:
#
#   pre_generation             the model has not chosen a call yet; the only lever is context.
#   post_generation_pre_exec   a call exists but has not run; it can still be CANCELLED, which
#                              is the only point where suppression is free.
#   post_execution             the call ran; the actual error and live state are available.
#
# Admissibility is a FEASIBILITY claim with a stated reason, so a reader can check it:
ADMISSIBLE_POINTS = {
    # The value exists elsewhere. A pre-generation nudge could send the reader to the right
    # container from the start; a post-execution redirect could act on the miss. Both are
    # feasible and they trade differently (every-turn context cost vs per-miss latency), which
    # is exactly the comparison the policy learner exists to make.
    "unreachable_but_stored": ({PRE_GEN, POST_EXEC},
                               "value is in another container: reader can be forewarned "
                               "(pre-gen) or redirected after the miss (post-exec)"),
    # No call was ever made, so there is nothing to cancel and nothing to repair. Only the
    # information the model held when it planned can change the outcome.
    "never_stored": ({PRE_GEN},
                     "no call exists to intervene on -- only the planning context"),
    # The write was attempted and refused. It can be repaired from the error (post-exec) or
    # avoided by acting on state pressure before the attempt (pre-gen).
    "write_rejected": ({PRE_GEN, POST_EXEC},
                       "a call exists and its refusal is observable: pre-empt on state "
                       "pressure (pre-gen) or repair the refusal (post-exec)"),
    # Destruction is only preventable BEFORE it runs; post-execution the state is gone.
    "stored_then_destroyed": ({PRE_EXEC},
                              "irreversible once executed -- cancellable only pre-execution"),
    "unattributed": (set(), "no write-side explanation; incision points not constrained"),
}

ACTIONS = ("noop", "suppress", "reroute", "reprompt")


def feasible_cells(cls: str, stage: str) -> Tuple[List[Tuple[str, str]], List[Dict]]:
    """The (incision point x action) cells that are STRUCTURALLY live, plus why each is not.

    The grid is a COORDINATE SYSTEM, not a work list. Most cells rule out from what is
    *available* at the point -- a property of the point itself, not of the benchmark -- so the
    nominal |points| x 4 search is implicit and collapses before any arm is run. Measured on
    `unreachable_but_stored`: 8 nominal cells, **3 live arms**.

    The rules, stated once and generically:

      * `noop` is POINT-INDEPENDENT. It is the incumbent, already measured, and costs nothing.
        Emitting it per point would triple-count one control.
      * At `pre_generation` no call exists yet, so `suppress` (nothing to cancel) and `reroute`
        (nothing to substitute for) are both vacuous. `reprompt` -- context injection -- is the
        ONLY pre-generation lever.
      * At `post_generation_pre_execution` a call exists but has not run, which is the only
        point where `suppress` is free. `reroute` can also swap it before dispatch.
      * At `post_execution` the call has run. `suppress` cannot undo it; for a READ that is
        total (the value was returned or not), for a WRITE the state is already changed.
        `reroute` and `reprompt` remain.

    Returns (live_cells, excluded) where each exclusion carries its reason, so a reader can
    check that a cell was ruled out structurally rather than forgotten.
    """
    pts, _why = admissible_points(cls)
    live: List[Tuple[str, str]] = []
    excluded: List[Dict] = []
    for pt in pts:
        for act in ACTIONS:
            ok, why = _cell_feasible(pt, act, stage)
            if ok:
                live.append((pt, act))
            else:
                excluded.append({"point": pt, "action": act, "reason": why})
    return live, excluded


def _cell_feasible(point: str, action: str, stage: str) -> Tuple[bool, str]:
    if action == "noop":
        return False, "no-op is point-independent: one shared control, already measured"
    if point == PRE_GEN:
        if action == "suppress":
            return False, "nothing to suppress -- no call has been generated yet"
        if action == "reroute":
            return False, "nothing to reroute -- no call exists to substitute for"
        return True, "context injection is the only pre-generation lever"
    if point == PRE_EXEC:
        if action == "reprompt":
            return False, "a call is already pending; advising instead of acting wastes the step"
        return True, ("the call exists but has not run -- this is the only point where "
                      "suppression is free")
    if point == POST_EXEC:
        if action == "suppress":
            return False, ("the call already ran and cannot be undone"
                           + (" -- a read has already returned" if stage == "read"
                              else " -- the state is already changed"))
        return True, "the failing call, its args and the live state are all in hand"
    return False, "unknown incision point"


def admissible_points(cls: str) -> Tuple[List[str], str]:
    """Unordered admissible set (sorted for stable output) plus the feasibility reason."""
    pts, why = ADMISSIBLE_POINTS.get(cls, (set(), ""))
    return sorted(pts), why


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", required=True, type=Path)
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--locus", default=None,
                    help="restrict to one locus, e.g. existence/identifier/not_found")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    only = tuple(a.locus.split("/")) if a.locus else None
    rows, meta = attribute(a.incumbent, a.artifact, only)
    print("incumbent: %s" % a.incumbent)
    print("artifact : %s  fp=%s" % (a.artifact.name, meta["fingerprint"]))
    print("occurrences attributed: %d\n" % len(rows))

    by_locus = collections.defaultdict(collections.Counter)
    for r in rows:
        by_locus["/".join(r["locus"])][(r["class"], r["stage"])] += 1
    for lk in sorted(by_locus, key=lambda x: -sum(by_locus[x].values())):
        tot = sum(by_locus[lk].values())
        print("%s   (%d occurrences)" % (lk, tot))
        for (cls, stage), n in by_locus[lk].most_common():
            print("   %5d  %-24s stage=%-6s %s" % (n, cls, stage, REMEDY[cls]))
        si = sum(1 for r in rows if "/".join(r["locus"]) == lk and r["self_inflicted"])
        if si:
            print("   %5d  of these are SELF-INFLICTED: the value sits where an accepted "
                  "anchor put it" % si)
        print()

    # The (point x action) grid, and how far it COLLAPSES before any arm is run.
    print("=== search space handed to the policy learner ===")
    seen = set()
    for r in rows:
        key = (r["class"], r["stage"])
        if key in seen or r["class"] == "unattributed":
            continue
        seen.add(key)
        live, excl = feasible_cells(r["class"], r["stage"])
        pts = admissible_points(r["class"])[0]
        n = sum(1 for x in rows if (x["class"], x["stage"]) == key)
        print("  %-24s stage=%-5s %4d occurrences" % (r["class"], r["stage"], n))
        print("     nominal grid %d points x %d actions = %d cells -> %d LIVE"
              % (len(pts), len(ACTIONS), len(pts) * len(ACTIONS), len(live)))
        for pt, act in live:
            print("        LIVE  %-30s + %s" % (pt, act))
        for e in excl:
            print("        --    %-30s + %-9s %s" % (e["point"], e["action"], e["reason"][:58]))
        print("        plus ONE shared no-op (the incumbent, already measured)")
    print()

    # The headline the policy stage needs: which STAGE owns this failure family.
    print("=== stage attribution (what the policy search should target) ===")
    for lk in sorted(by_locus):
        st = collections.Counter()
        for (cls, stage), n in by_locus[lk].items():
            st[stage] += n
        tot = sum(st.values())
        top, n = st.most_common(1)[0]
        print("  %-44s -> %-5s stage (%d/%d = %.0f%%)" % (lk, top, n, tot, 100 * n / tot))

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"meta": meta, "rows": rows}, open(a.json, "w"), indent=2)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
