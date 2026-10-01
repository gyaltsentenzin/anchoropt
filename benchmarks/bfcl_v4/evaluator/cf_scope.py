"""Counterfactual evaluation scope (§76).

Which episodes a candidate arm must actually be evaluated on, and which may be copied from the
control. Two safety requirements from §75.3 are enforced here rather than left to the caller:

  (a) exposure is derived from the gate's DECLARED BACKEND SCOPE plus its match_substrings,
      never from observed control incidence. Predicting from the control under-covered on G1
      (kv predicted, kv+vector actual) and would have skipped a case that DID flip. The rule
      is over-include, never under-include.

  (b) an AUDIT SAMPLE of copied episodes is re-run and must show 0 flips. Copying removes the
      check that caught the §70 port collision -- two identical controls disagreed on 36
      cases, ALL of them untouched -- so the audit restores it.

`full` scope is the default everywhere and is REQUIRED for any benchmark whose generation is
stochastic: there, untouched episodes legitimately differ run to run and a copied outcome is
simply wrong. `affected_only` is admissible for BFCL because determinism is MEASURED
(§66: 6/6 identical-policy pairs at 0 flips; §75.2: 253 untouched cases, 0 flips).
"""

from __future__ import annotations

import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

FULL = "full"
AFFECTED_ONLY = "affected_only"
SCOPES = (FULL, AFFECTED_ONLY)


def backend_of(case_id: str) -> Optional[str]:
    for b in ("rec_sum", "vector", "kv"):          # rec_sum first: 'kv' is not a substring
        if "memory_%s" % b in str(case_id):
            return b
    return None


def chain_of(case_id: str) -> Optional[str]:
    """backend+domain key shared by a prereq chain and the queries that read it."""
    m = re.match(r"(memory_(?:kv|vector|rec_sum))_(?:prereq_)?\d+-([a-z]+)-\d+$", str(case_id))
    return "%s_%s" % (m.group(1), m.group(2)) if m else None


def declared_backends(specs: Iterable) -> Set[str]:
    """Union of the candidate gates' DECLARED backends. Empty tuple means all backends.

    Declared, not observed: §75.3(a). A gate scoped ("kv","vector") is treated as able to act
    on vector even if the control corpus only showed kv exposure -- which is exactly the case
    that broke the observed-incidence version.
    """
    out: Set[str] = set()
    for sp in specs:
        bks = tuple(getattr(sp, "backends", ()) or ())
        if not bks:
            return {"kv", "vector", "rec_sum"}
        out.update(bks)
    return out


def hot_chains_from_traj(traj_dir, substrings: Sequence[str]) -> Set[str]:
    """Chains whose CONTROL trajectories contain the target contract.

    Read from the trajectory sidecars, NOT from the results JSON: error text lives in
    steps[].tool_results, while the results JSON carries only summary booleans. An earlier
    version read the results JSON, found nothing, and would therefore have declared every case
    copyable -- reporting the control's accuracy as the candidate's and erasing all 15 (G3) /
    9 (G1) real flips. Caught by the offline replay validation.
    """
    import glob
    import json as _json
    subs = tuple(x.lower() for x in (substrings or ()))
    out: Set[str] = set()
    if not subs or not traj_dir:
        return out
    for f in glob.glob(str(traj_dir) + "/*/*.json"):
        try:
            t = _json.load(open(f))
        except Exception:
            continue
        for st in t.get("steps", []):
            blob = str(st.get("tool_results") or "").lower()
            if any(x in blob for x in subs):
                c = chain_of(t.get("case_id"))
                if c:
                    out.add(c)
                break
    return out


def affected_cases(cases: Sequence[Dict], specs: Iterable,
                   control_traj_dir=None,
                   hot_chains: Optional[Set[str]] = None,
                   backends: Optional[Set[str]] = None) -> Tuple[Set[str], Set[str]]:
    """-> (must_evaluate_ids, copyable_ids).

    A case must be evaluated if it is on a declared backend AND its chain shows the target
    contract in the control TRAJECTORIES. Chain-level, because a prereq firing changes the
    stored world that later queries read.

    With no exposure evidence the function is deliberately conservative: everything on a
    declared backend must be evaluated (§75.3(a): over-include, never under-include).
    """
    specs = list(specs)
    bks = set(backends) if backends else declared_backends(specs)
    # EMPTY specs would make `bks` empty, so every case fails the backend test below and is
    # COPIED -- an arm that evaluates nothing and reports +0.00pp because it never ran. A
    # predicate-based anchor legitimately has no specs, so it must pass `backends` explicitly
    # rather than relying on spec declarations.
    if not bks:
        raise ValueError(
            "affected_cases: no declared backends. A substring-based anchor must pass its "
            "GateSpecs; a predicate-based anchor must pass `backends=` explicitly. Proceeding "
            "would copy every case and measure nothing.")

    subs: List[str] = []
    for sp in specs:
        subs.extend(x.lower() for x in (getattr(sp, "match_substrings", ()) or ()))

    if hot_chains is None:
        hot_chains = hot_chains_from_traj(control_traj_dir, subs) if control_traj_dir else set()
    have_evidence = bool(subs) and bool(hot_chains)

    must, copyable = set(), set()
    for c in cases:
        cid = c["id"]
        if "prereq" in cid:                 # prereqs build the world; never copied
            must.add(cid)
            continue
        if backend_of(cid) not in bks:      # gate cannot act on this backend at all
            copyable.add(cid)
            continue
        if not have_evidence:
            must.add(cid)                   # unknown exposure -> evaluate
            continue
        (must if chain_of(cid) in hot_chains else copyable).add(cid)
    return must, copyable


def zero_call_chains(control_traj_dir) -> Set[str]:
    """Chains containing an episode that executed NO tool calls.

    Exposure evidence for a PREDICATE-based anchor rather than a substring-based one. A gate whose
    trigger is a property of the trajectory (A4v2: "this query called nothing") has no
    `match_substrings`, so `affected_cases` finds no evidence and conservatively evaluates
    everything -- correct but maximally slow.

    Measured on the A1+A2+A3 world: 5 of 84 prereq episodes make zero calls, and they sit in
    3 of 12 chains. So 9 chains provably cannot be reached by a zero-call gate in the write phase,
    and their stores are byte-identical to the incumbent's.
    """
    import glob
    import json
    hot: Set[str] = set()
    if not control_traj_dir:
        return hot
    # PREREQ ONLY. Scanning every phase was wrong and would have destroyed the saving: 60 QUERY
    # episodes make zero calls and they touch 10 of 12 chains, so a phase-blind scan marks almost
    # everything hot (12% reusable instead of 75%). What determines whether a chain's STORE can be
    # reused is only whether the gate can fire while that store is being BUILT -- i.e. in prereqs.
    for f in glob.glob("%s/prereq/*.json" % control_traj_dir):
        try:
            ep = json.load(open(f))
        except Exception:
            continue
        cid = ep.get("case_id") or ""
        n = sum(len(s.get("decoded") or []) for s in ep.get("steps") or [])
        if n == 0:
            hot.add(chain_of(cid))
    return hot


def reusable_prereq_chains(cases: Sequence[Dict], hot_chains: Set[str]) -> Tuple[Set[str], Set[str]]:
    """Split PREREQ ids into must-rebuild vs store-reusable, by chain.

    `affected_cases` never copies a prereq -- deliberately, because prereqs build the world every
    later query reads. That is the right default and it is also the whole cost: the store build is
    87-90% of arm wallclock. But when a gate provably cannot fire in a chain's prereqs, that
    chain's store is unchanged and rebuilding it is pure waste.

    NOT SAFE BY DEFAULT. The caller must supply hot_chains from real control evidence, and the
    result must be audited: reusing a store built under different code is exactly how the
    `500a799e` fingerprint pin silently scored two accepted anchors against a stale world. Treat
    this as an optimisation that requires a 0-flip audit, not as a free win.
    """
    rebuild, reuse = set(), set()
    for c in cases:
        cid = c["id"]
        if "prereq" not in cid:
            continue
        (rebuild if chain_of(cid) in hot_chains else reuse).add(cid)
    return rebuild, reuse


def audit_sample(copyable: Set[str], n: int, seed: int = 42) -> Set[str]:
    """Randomly choose copied episodes to RE-RUN as a corruption check (§75.3b)."""
    if n <= 0 or not copyable:
        return set()
    rnd = random.Random(seed)
    ids = sorted(copyable)
    return set(rnd.sample(ids, min(n, len(ids))))


def check_audit(audited: Dict[str, bool], copied: Dict[str, bool]) -> Tuple[bool, List[str]]:
    """0 flips required. Returns (ok, disagreeing_ids)."""
    bad = [k for k, v in audited.items() if k in copied and copied[k] != v]
    return (not bad), bad
