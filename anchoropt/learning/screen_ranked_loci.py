"""Cheap screen of the next ranked loci, so a queue is ready when A3 resolves.

Per locus: coverage, saturation by the incumbent, stage attribution, self-inflicted vs native,
attested destinations (RESOLVING, not merely not-erroring), and whether a plausible policy exists.

All offline. Every one of these tests has already changed a decision this session:
  * coverage/saturation      -> S1/S2 excluded ranks 4-5 and stopped rank 1
  * stage attribution        -> split `not_found` into a read anchor and a write anchor
  * self-inflicted           -> revealed the whole duplicate locus was A1-induced
  * attested destinations    -> would have saved the A2.R arm (1 resolving / 9 vacuous)
"""
import collections
import glob
import json
import re
import sys

import argparse                    # noqa: E402
import os                          # noqa: E402
from pathlib import Path           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "attribution"))
import op_canonicalize_locus as oc  # noqa: E402
import policy_tree as pt          # noqa: E402  vacuity detector

ARG = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])(.*?)\2", re.S)
CALL = re.compile(r"([a-z_][a-z_0-9]*)\s*\(")

# Paths are arguments, not constants. This script was written as a one-off against one incumbent;
# the defaults below reproduce that invocation when ANCHOROPT_ROOT points at a tree with those
# results, and every path can be overridden.
_ROOT = Path(os.environ.get("ANCHOROPT_ROOT", Path(__file__).resolve().parents[2]))

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--incumbent", type=Path, default=_ROOT / "results/n0clean_train/a1a2",
                 help="incumbent run directory (needs traj/ and eval_train_results.json)")
_ap.add_argument("--artifact", type=Path, default=_ROOT / "ledgers/n0t3clean_locus.json",
                 help="frozen locus artifact for this iteration")
_ap.add_argument("--ranking", type=Path, default=Path("/tmp/n0t3clean_rank.json"),
                 help="ranked-loci JSON emitted by policy_tree.rank_candidates")
_args, _ = _ap.parse_known_args()

INC, ART = _args.incumbent, _args.artifact
for _p, _what in ((INC, "incumbent dir"), (ART, "locus artifact"), (_args.ranking, "ranking")):
    if not _p.exists():
        raise SystemExit(
            f"{_what} not found: {_p}\n"
            "This screen runs against a completed incumbent run. Pass --incumbent/--artifact/"
            "--ranking, or set ANCHOROPT_ROOT. See docs/GENERALIZABILITY.md."
        )

rank = json.load(open(_args.ranking))["ranked"]
nfail = json.load(open(_args.ranking))["meta"]["n_failing_queries"]
art = oc.LocusArtifact.load(ART)


def classify(payload):
    s = payload if isinstance(payload, str) else str(payload)
    if pt._canon_error(s) is not None:
        return "errored"
    if pt.vacuous_result_kind(s) is not None:
        return "vacuous"
    return "resolving" if re.search(r"[A-Za-z0-9]", s) else "vacuous"


# Per locus: what the model tried next and how it turned out; and whether the failing
# call was one an incumbent ANCHOR injected (self-inflicted) or the model's own (native).
nxt = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
origin = collections.defaultdict(collections.Counter)

for ph in ("prereq", "query"):
    for f in glob.glob("%s/traj/%s/*.json" % (INC, ph)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        steps = d.get("steps") or []
        injected = set()
        for i, s in enumerate(steps):
            for fld in ("reroute_substituted_call", "capacity_repair_substituted_call"):
                sub = s.get(fld)
                if sub:
                    a = {k: v for k, _q, v in ARG.findall(str(sub))}
                    if a.get("key"):
                        injected.add(a["key"])
            dec = s.get("decoded") or []
            rs = s.get("tool_results") or []
            for j, c in enumerate(dec):
                r = rs[j] if j < len(rs) else ""
                a = {k: v for k, _q, v in ARG.findall(str(c))}
                k = art.locus_of(r, a)
                if not k:
                    continue
                lk = "/".join(k)
                origin[lk]["anchor-induced" if a.get("key") in injected else "native"] += 1
                for s2 in steps[i + 1:i + 4]:
                    d2 = s2.get("decoded") or []
                    r2 = s2.get("tool_results") or []
                    for j2, c2 in enumerate(d2):
                        m2 = CALL.search(str(c2))
                        if not m2:
                            continue
                        nxt[lk][m2.group(1)][classify(r2[j2] if j2 < len(r2) else "")] += 1
                    if d2:
                        break

print("SCREEN of the ranked queue -- incumbent N0+A1+A2, residual %d\n" % nfail)
for i, r in enumerate(rank, 1):
    lk = "/".join(r["locus"])
    frac = 100 * r["linked_loss"] / nfail
    sat = r["settled_by_incumbent"] / max(r["raw_events"], 1)
    s1 = "PASS" if frac >= 10 else "S1-STOP"
    s2 = "PASS" if sat < 0.5 else "S2-STOP"
    att = [(t, c["resolving"]) for t, c in nxt[lk].items() if c.get("resolving", 0) >= 10]
    vac = [(t, c["vacuous"]) for t, c in nxt[lk].items() if c.get("vacuous", 0) > c.get("resolving", 0)]
    o = origin[lk]
    tot_o = sum(o.values()) or 1
    print("rank %d  %s" % (i, lk))
    print("   coverage  %5.1f%% of residual (%d linked)          %s" % (frac, r["linked_loss"], s1))
    print("   saturation %4.0f%% settled (%d/%d)                  %s"
          % (100 * sat, r["settled_by_incumbent"], r["raw_events"], s2))
    print("   provenance anchor-induced %d (%.0f%%) / native %d"
          % (o.get("anchor-induced", 0), 100 * o.get("anchor-induced", 0) / tot_o,
             o.get("native", 0)))
    print("   attested   %s" % (att or "NONE -- no single-call destination resolves"))
    if vac:
        print("   VACUOUS    %s  <- reports success, returns nothing" % vac)
    verdict = ("queue" if s1 == "PASS" and s2 == "PASS" and att else
               "skip: " + (s1 if s1 != "PASS" else s2 if s2 != "PASS" else "no attested destination"))
    print("   -> %s\n" % verdict)
