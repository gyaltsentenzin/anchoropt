#!/usr/bin/env python3
"""LLM canonicalizer for semantic locus identity, with a FROZEN per-iteration artifact.

WHY AN LLM HERE, AND ONLY HERE
------------------------------
`constraint_locus.py` decides locus identity from a hand-written cue table (`_CUES`). That
table was authored by reading six error strings from one benchmark, so it is structurally
incomplete: an unrecognized wording returns None and the locus silently VANISHES from mining
rather than raising. That is the exact failure that hid 436 events behind the registry's
"exceeds maximum length" match string -- a closed vocabulary standing in for an open one.

Deciding "do these two failures violate the same decision constraint?" is open-vocabulary
semantic judgment. It is the one step in the pipeline that is genuinely uncertain, so it is
the one step that gets a model. Everything downstream stays deterministic:

    uncertain semantic observation      <- THIS MODULE (LLM, build time only)
      -> FROZEN structured signal       <- LocusArtifact, cached + fingerprinted
      -> deterministic evidence/ranking <- policy_tree
      -> deterministic/validated policy <- adapters
      -> counterfactual acceptance      <- paired arms, ground truth

FREEZING IS THE LOAD-BEARING PART
---------------------------------
The LLM may be nondeterministic while PROPOSING. Once written, the artifact is the fixed world
every downstream stage reads, so ranking, exposure sets and acceptance are reproducible even
though their input came from a sampler. A run either reuses a frozen artifact or creates one;
it never re-asks mid-iteration. `freeze()` refuses to overwrite.

The cache key folds in MODEL ID and PROMPT VERSION, not just the error text. Without that a
model swap or a prompt edit would silently re-group history while the file name stayed the
same -- the `_store_code_fingerprint` lesson, where a fingerprint that covered only part of
the inputs let a stale cache masquerade as a clean rerun.

THE LLM DOES NOT GET THE LAST WORD
----------------------------------
Every proposal is validated against the deterministic classifier's INVARIANTS, not against its
answer:
  * the vocabulary is closed -- constraint_type/resource/violation_mode must be known values;
  * no remedy vocabulary may appear in a key (diagnosis must not encode intervention);
  * when the deterministic classifier also has an opinion, disagreement is RECORDED as a
    logged event, never silently resolved in either direction.
Agreement raises confidence; disagreement is the interesting signal and is surfaced for review.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The repo ROOT too, not just scripts/: LLMOperator._client() resolves its gateway via
# `from anchoropt.teacher_client import ...`, which needs the package parent importable.
# Without this the operator silently reports "no client for model ..." and every run falls
# back to the deterministic table -- a working gateway looking exactly like a missing one.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_operator import LLMOperator  # noqa: E402
import constraint_locus as det  # noqa: E402

# v2: surface the STATED BOUND and each argument's size RELATIVE to it. v1 showed only
# absolute argument sizes, so the model had no way to tell "236 chars against a 300 limit"
# (the item is the violator) from "236 chars against a 10000 limit" (the item is fine and the
# aggregate is exhausted). It misfiled 341 events as item violations on its first live run --
# the same arithmetic mistake the deterministic classifier made with an absolute 200-char
# floor. The judgment is comparative, so the comparison has to be in the prompt.
PROMPT_VERSION = "v2"

# Closed vocabulary. The LLM may only emit these; anything else is rejected in validate().
# Deliberately domain-neutral -- no memory/tool/backend words -- so the same vocabulary
# describes a constraint violation in any agent benchmark.
CONSTRAINT_TYPES = ("capacity", "size", "format", "permission", "existence", "rate")
RESOURCES = ("container", "item", "field", "identifier", "quota")
VIOLATION_MODES = (
    "no_remaining_capacity",     # the container cannot admit another item
    "exceeds_per_item_limit",    # the item itself is larger than a per-item bound
    "malformed_value",
    "not_permitted",
    "not_found",
    "duplicate",
    "rate_exceeded",
)
_REMEDY_WORDS = ("reroute", "transform", "relocate", "reduce", "suppress", "reprompt",
                 "retry", "tool", "argument", "fix", "repair")

SYSTEM = """You classify tool-call failures by the DECISION CONSTRAINT they violate.

You are given error messages produced by an agent's tool calls, together with the arguments of
the call that failed. Group them by whether they violate the SAME constraint.

Two failures share a constraint if and only if an intervention addressing one would address the
other in kind. Judge the constraint, NOT the wording: messages phrased differently often violate
the same constraint, and messages phrased alike sometimes do not.

The critical distinction you must get right:
  - The CONTAINER has no room left. The item being added is perfectly valid; the store is
    exhausted. Signals: a named resource is reported full, at capacity, or out of space. Also
    this case when a message blames the incoming item but the item is SMALL relative to the
    stated bound -- then the bound belongs to the aggregate, and the item merely triggered the
    check.
  - The ITEM violates a per-item bound. The container is fine; this specific payload is too
    large or malformed. Signals: a stated per-item limit, AND an argument in the failing call
    actually near or over that limit.

DO THE ARITHMETIC. Each input states its bound and each argument's size as a PERCENTAGE of
that bound. An argument at 5%% of the stated bound is nowhere near it and CANNOT be what
violated it -- however much the message blames the entry. In that case the bound belongs to
the aggregate and the correct answer is the container case. Only treat the item as the
violator when an argument is at roughly half the bound or more.

Emit ONLY a JSON array. One object per DISTINCT constraint you identify:

[{"constraint_type": "...", "resource": "...", "violation_mode": "...",
  "member_ids": [0, 3, 4], "reason": "one sentence, about the constraint"}]

constraint_type must be one of: %s
resource must be one of: %s
violation_mode must be one of: %s

Every input id must appear in exactly one group. Do not describe fixes or remedies -- you are
diagnosing what was violated, not deciding what to do about it.""" % (
    ", ".join(CONSTRAINT_TYPES), ", ".join(RESOURCES), ", ".join(VIOLATION_MODES))


def _fingerprint(observations: List[Dict], model: str) -> str:
    """Identity of (inputs, model, prompt). Any change must split the cache."""
    h = hashlib.sha256()
    h.update(PROMPT_VERSION.encode())
    h.update(b"\0")
    h.update((model or "").encode())
    for o in sorted(observations, key=lambda x: x.get("error", "")):
        h.update(b"\0")
        h.update(str(o.get("error", "")).encode())
        h.update(b"\1")
        h.update(json.dumps(o.get("args") or {}, sort_keys=True).encode())
    return h.hexdigest()[:16]


class LocusArtifact:
    """A frozen semantic-locus grouping. Immutable once written.

    Downstream stages read THIS, never the LLM. Reproducibility of ranking therefore does not
    depend on the sampler being deterministic -- only on the artifact being fixed.
    """

    def __init__(self, path: Path, data: Dict[str, Any]):
        self.path = Path(path)
        self.data = data

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: Path) -> Optional["LocusArtifact"]:
        p = Path(path)
        if not p.exists():
            return None
        return cls(p, json.load(open(p)))

    @classmethod
    def freeze(cls, path: Path, iteration: str, observations: List[Dict],
               groups: List[Dict], model: str, disagreements: List[Dict],
               overwrite: bool = False) -> "LocusArtifact":
        p = Path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(
                "%s already frozen. An iteration's locus artifact is written ONCE; "
                "re-asking the LLM mid-iteration would change the world that ranking, "
                "exposure and acceptance are all computed against. Pass overwrite=True "
                "only to start a NEW iteration." % p)
        data = {
            "iteration": iteration,
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "fingerprint": _fingerprint(observations, model),
            "n_observations": len(observations),
            "groups": groups,
            "disagreements": disagreements,
            "frozen": True,
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(data, open(p, "w"), indent=2, sort_keys=True)
        return cls(p, data)

    # ── the downstream read path ────────────────────────────────────────────
    def locus_of(self, error_text: str, args: Optional[Dict] = None) -> Optional[Tuple]:
        """Frozen lookup. Falls back to the deterministic classifier for unseen text.

        A miss is EXPECTED on a later corpus and is not an error -- but it is recorded by
        `misses()` so a caller can see how much of the ranking rests on the fallback.
        """
        norm = _norm_err(error_text)
        for g in self.data.get("groups", []):
            if norm in g.get("members", []):
                return (g["constraint_type"], g["resource"], g["violation_mode"])
        self._miss.add(norm)
        return det.constraint_locus(error_text, args)

    _miss: set = set()

    def misses(self) -> List[str]:
        return sorted(self._miss)

    def matches_inputs(self, observations: List[Dict]) -> bool:
        """True when this artifact was frozen over exactly these inputs and model."""
        return self.data.get("fingerprint") == _fingerprint(
            observations, self.data.get("model", ""))


def _norm_err(text) -> str:
    """Digit-masked normal form, used only as a CACHE KEY (never as locus identity).

    `text` may be a dict: live tool results arrive as objects, sidecar results as strings.
    `_error_message` handles both, but its None return fell through `or text` to the RAW DICT,
    which then reached `re.sub` and raised TypeError. Coerce to str at the boundary so the
    fallback is always a string -- the same dict-vs-str mismatch that earlier made every live
    capture record locus=None.
    """
    msg = det._error_message(text)
    if msg is None:
        msg = text if isinstance(text, str) else ("" if text is None else str(text))
    return det._norm(msg)[:90]


class LocusCanonicalizer(LLMOperator):
    """Proposes semantic locus groupings; validated against closed vocabulary + invariants."""

    name = "canonicalize_locus"

    def gate(self, ctx) -> Optional[str]:
        """Invoke only when there is something a closed table cannot settle.

        Two triggers, both cheap to check:
          * the deterministic classifier returned None for some observation (unseen wording);
          * or it produced groups so fine-grained that lexical fragmentation is likely.
        If the table covered everything and produced one group, there is nothing to gain.
        """
        obs = ctx.get("observations") or []
        if not obs:
            return None
        unclassified = [o for o in obs
                        if det.constraint_locus(o.get("error", ""), o.get("args")) is None]
        keys = {det.constraint_locus(o.get("error", ""), o.get("args")) for o in obs}
        keys.discard(None)
        if unclassified:
            return ("%d of %d observations unclassified by the deterministic cue table"
                    % (len(unclassified), len(obs)))
        if len(keys) > 1 and len(obs) >= 4:
            return ("deterministic table produced %d distinct keys over %d observations; "
                    "checking for lexical fragmentation" % (len(keys), len(obs)))
        return None

    def build_prompt(self, ctx):
        obs = ctx["observations"]
        lines = []
        for i, o in enumerate(obs):
            args = o.get("args") or {}
            bound = det._stated_bound(str(o.get("error", "")))
            # Argument SIZES, never full payloads -- and each expressed as a PERCENTAGE of the
            # stated bound, because the judgment is comparative. Absolute sizes alone made the
            # model call a 236-char item a violation of a 10000-char bound.
            parts = []
            for k, v in sorted(args.items(), key=lambda kv: -len(str(kv[1])))[:4]:
                n = len(str(v))
                parts.append("%s=%d chars%s" % (
                    k, n, (" (%d%% of bound)" % round(100.0 * n / bound)) if bound else ""))
            lines.append("id=%d  error=%r\n    stated_bound: %s\n    call_args: %s\n"
                         "    observed: %sx"
                         % (i, str(o.get("error", ""))[:200],
                            bound if bound else "none stated",
                            "; ".join(parts) or "none", o.get("count", 1)))
        return SYSTEM, "\n\n".join(lines)

    def validate(self, item, ctx) -> Dict[str, Any]:
        obs = ctx["observations"]
        if not isinstance(item, dict):
            return {"status": "rejected", "reason": "not an object"}
        ct = str(item.get("constraint_type", "")).strip().lower()
        rs = str(item.get("resource", "")).strip().lower()
        vm = str(item.get("violation_mode", "")).strip().lower()
        for val, allowed, field in ((ct, CONSTRAINT_TYPES, "constraint_type"),
                                    (rs, RESOURCES, "resource"),
                                    (vm, VIOLATION_MODES, "violation_mode")):
            if val not in allowed:
                return {"status": "rejected",
                        "reason": "%s=%r outside closed vocabulary" % (field, val)}
        # Diagnosis must not encode intervention.
        for w in _REMEDY_WORDS:
            if re.search(r"\b%s\b" % w, "%s %s %s" % (ct, rs, vm)):
                return {"status": "rejected", "reason": "remedy vocabulary %r in key" % w}
        ids = item.get("member_ids")
        if not isinstance(ids, list) or not ids:
            return {"status": "rejected", "reason": "member_ids missing or empty"}
        try:
            ids = sorted({int(i) for i in ids})
        except Exception:
            return {"status": "rejected", "reason": "member_ids not integers"}
        if any(i < 0 or i >= len(obs) for i in ids):
            return {"status": "rejected", "reason": "member_ids out of range"}

        members = [_norm_err(obs[i].get("error", "")) for i in ids]
        events = sum(int(obs[i].get("count", 1)) for i in ids)
        # Where the deterministic classifier HAS an opinion, record agreement/disagreement.
        det_keys = {det.constraint_locus(obs[i].get("error", ""), obs[i].get("args"))
                    for i in ids}
        det_keys.discard(None)
        proposed = (ct, rs, vm)
        return {"status": "accepted",
                "constraint_type": ct, "resource": rs, "violation_mode": vm,
                "member_ids": ids, "members": members, "events": events,
                "reason": str(item.get("reason", ""))[:300],
                "det_keys": [list(k) for k in sorted(det_keys)],
                "agrees_with_det": det_keys == {proposed} if det_keys else None}


def canonicalize(observations: List[Dict], artifact_path: Path, iteration: str,
                 model: Optional[str] = None, endpoint: Optional[str] = None,
                 enabled: bool = True, overwrite: bool = False) -> LocusArtifact:
    """Freeze a semantic-locus artifact for one iteration.

    Reuses an existing artifact when its fingerprint matches the inputs -- so re-running a
    mine is free and cannot re-group the world underneath a ranking already computed.
    """
    existing = LocusArtifact.load(artifact_path)
    if existing is not None and not overwrite:
        if existing.matches_inputs(observations):
            return existing
        raise FileExistsError(
            "%s exists but was frozen over different inputs/model (fingerprint %s). "
            "Ranking must not mix worlds: start a new iteration artifact instead."
            % (artifact_path, existing.data.get("fingerprint")))

    op = LocusCanonicalizer(model=model, enabled=enabled, endpoint=endpoint)
    res = op.run({"observations": observations})
    groups, disagreements = [], []
    for rec in (res.candidates or []):
        if rec.get("status") != "accepted":
            disagreements.append({"kind": "rejected_proposal",
                                  "reason": rec.get("reason"), "raw": rec.get("raw_item")})
            continue
        # COALESCE identical keys. The model may emit two groups carrying the same
        # (type, resource, mode) -- on the first v2 run it split the container case into
        # "container full" and "entry count exceeds maximum", which are the same violated
        # constraint under different wording. Left alone this breaks the artifact contract:
        # locus_of() returns on FIRST match, so the later group's events would resolve to the
        # earlier group silently. Merging is safe precisely because the key IS the identity --
        # if two groups share it, the model has asserted they are one locus.
        _key = (rec["constraint_type"], rec["resource"], rec["violation_mode"])
        _prior = next((g for g in groups
                       if (g["constraint_type"], g["resource"],
                           g["violation_mode"]) == _key), None)
        if _prior is not None:
            _prior["member_ids"] = sorted(set(_prior["member_ids"]) | set(rec["member_ids"]))
            _prior["members"] = sorted(set(_prior["members"]) | set(rec["members"]))
            _prior["events"] += rec["events"]
            _prior["reason"] = (_prior.get("reason", "") + " | " + rec.get("reason", ""))[:600]
            _prior.setdefault("coalesced", 0)
            _prior["coalesced"] += 1
            disagreements.append({"kind": "coalesced_duplicate_key",
                                  "locus": list(_key), "members": rec.get("members"),
                                  "reason": rec.get("reason")})
            continue
        if rec.get("agrees_with_det") is False:
            disagreements.append({"kind": "llm_vs_deterministic",
                                  "llm": [rec["constraint_type"], rec["resource"],
                                          rec["violation_mode"]],
                                  "deterministic": rec.get("det_keys"),
                                  "members": rec.get("members"),
                                  "reason": rec.get("reason")})
        groups.append(rec)

    # FALLBACK: no usable LLM output (disabled, no client, unparseable) -> freeze the
    # deterministic grouping instead of nothing, and say so in the artifact.
    used_model = res.model if res.invoked else "deterministic-fallback"
    if not groups:
        by_key: Dict[Tuple, Dict] = {}
        for i, o in enumerate(observations):
            k = det.constraint_locus(o.get("error", ""), o.get("args"))
            if k is None:
                continue
            g = by_key.setdefault(k, {"constraint_type": k[0], "resource": k[1],
                                      "violation_mode": k[2], "member_ids": [],
                                      "members": [], "events": 0,
                                      "reason": "deterministic cue table",
                                      "status": "accepted", "agrees_with_det": True})
            g["member_ids"].append(i)
            g["members"].append(_norm_err(o.get("error", "")))
            g["events"] += int(o.get("count", 1))
        groups = list(by_key.values())
        disagreements.append({"kind": "llm_unavailable", "reason": res.reason})
        used_model = "deterministic-fallback"

    return LocusArtifact.freeze(artifact_path, iteration, observations, groups,
                                used_model, disagreements, overwrite=overwrite)


def observations_from_trajectories(traj_dir: Path, min_count: int = 1) -> List[Dict]:
    """Distinct (normalized error, argument shape) observations with event counts.

    Deduplicated so the LLM sees ~10 distinct wordings rather than ~800 events, and so the
    cache key is stable under re-running the same corpus.
    """
    import collections
    ARG = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])(.*?)\2", re.S)
    seen: Dict[str, Dict] = {}
    counts: collections.Counter = collections.Counter()
    for p in sorted(Path(traj_dir).rglob("*.json")):
        try:
            ep = json.load(open(p))
        except Exception:
            continue
        for s in ep.get("steps") or []:
            dec = s.get("decoded") or []
            res = s.get("tool_results") or []
            for j, call in enumerate(dec):
                raw = str(res[j]) if j < len(res) else ""
                msg = det._error_message(raw)
                if msg is None:
                    continue
                norm = det._norm(msg)[:90]
                counts[norm] += 1
                if norm not in seen:
                    args = {k: v for k, _q, v in ARG.findall(str(call))}
                    seen[norm] = {"error": msg, "args": args}
    out = []
    for norm, rec in seen.items():
        if counts[norm] < min_count:
            continue
        rec = dict(rec)
        rec["count"] = counts[norm]
        out.append(rec)
    return sorted(out, key=lambda r: -r["count"])


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traj-dir", required=True, type=Path)
    ap.add_argument("--artifact", required=True, type=Path,
                    help="where to freeze the locus grouping for this iteration")
    ap.add_argument("--iteration", default="T3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--endpoint", default=None,
                    help="OpenAI-compatible base URL for a locally served model")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--no-llm", action="store_true",
                    help="freeze the deterministic grouping (reference baseline)")
    ap.add_argument("--overwrite", action="store_true",
                    help="start a NEW iteration artifact, discarding the frozen one")
    a = ap.parse_args()

    obs = observations_from_trajectories(a.traj_dir, a.min_count)
    print("distinct error observations: %d (%d events)"
          % (len(obs), sum(o["count"] for o in obs)))
    if not obs:
        print("nothing to canonicalize", file=sys.stderr)
        return 1

    art = canonicalize(obs, a.artifact, a.iteration, model=a.model,
                       endpoint=a.endpoint, enabled=not a.no_llm, overwrite=a.overwrite)
    d = art.data
    print("\nartifact %s  model=%s  fingerprint=%s  frozen=%s"
          % (art.path, d["model"], d["fingerprint"], d["frozen"]))
    print("\n%-46s %8s %7s  %s" % ("semantic locus", "events", "wordings", "agrees?"))
    for g in sorted(d["groups"], key=lambda x: -x["events"]):
        key = "%s/%s/%s" % (g["constraint_type"], g["resource"], g["violation_mode"])
        print("%-46s %8d %7d  %s" % (key[:46], g["events"], len(g["member_ids"]),
                                     g.get("agrees_with_det")))
    if d["disagreements"]:
        print("\ndisagreements / notes (%d):" % len(d["disagreements"]))
        for x in d["disagreements"][:8]:
            print("   [%s] %s" % (x.get("kind"), json.dumps(x)[:150]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
