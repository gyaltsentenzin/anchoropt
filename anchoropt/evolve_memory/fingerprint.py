"""Canonical candidate fingerprints, so a round cannot re-buy a result it already owns.

    (residual_family, locus, signal, action_family, normalized_eta)

CONSERVATIVE BY DESIGN. Exact/normalized matching only -- no embeddings, no semantic similarity. A
false MATCH is far more expensive than a false miss here: a false match silently skips a candidate
that was never actually evaluated and reports someone else's outcome under its name, which is the
same class of error as coercing eta. A false miss just costs a GPU run we would have spent anyway.

WHY eta IS NORMALIZED AND NOT HASHED RAW
----------------------------------------
Two proposals that differ only in whitespace, capitalization, or trailing punctuation are the same
intervention, and the R3 arms proved the opposite matters: se3a1 and se3a2 differ in the instruction
TEXT and measured differently, so text is load-bearing and must stay in the key. Normalization
therefore collapses only formatting, never wording.

Numeric eta values are normalized to a canonical repr so `retry_budget=1` and `retry_budget=1.0`
agree -- the executor treats them as one value, so the ledger must too.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

# Fields that describe the SAME intervention regardless of value, so they are excluded from the key.
# Kept explicit: silently dropping a field that turns out to matter is how two different arms collapse
# into one ledger row.
_NON_IDENTIFYING_ETA_KEYS = frozenset({"detail", "variant", "rationale", "note", "_comment"})


def normalize_text(s: Any) -> str:
    """Collapse formatting only. Wording, ordering and punctuation-that-changes-meaning survive."""
    t = unicodedata.normalize("NFKC", str(s))
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t.rstrip(" .")


def normalize_eta(eta: Mapping[str, Any] | None) -> dict[str, str]:
    """eta as a canonical, order-independent dict of normalized strings."""
    out: dict[str, str] = {}
    for k, v in dict(eta or {}).items():
        key = normalize_text(k)
        if key in _NON_IDENTIFYING_ETA_KEYS:
            continue
        if isinstance(v, bool):
            out[key] = "true" if v else "false"
        elif isinstance(v, (int, float)):
            # 1 and 1.0 are the same budget to the executor, so they must be the same key here.
            out[key] = str(int(v)) if float(v).is_integer() else repr(float(v))
        elif isinstance(v, Mapping):
            out[key] = json.dumps({normalize_text(a): normalize_text(b) for a, b in v.items()},
                                  sort_keys=True)
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps([normalize_text(x) for x in v])
        else:
            out[key] = normalize_text(v)
    return dict(sorted(out.items()))


def residual_family(residual_key: str) -> str:
    """The CANONICAL family id for a residual.

    The residual key is the attributor's own `consequential_decision` prose, which varies between
    rounds even for the same failure mode -- SE1's re-mining produced 65 diagnoses under 17 different
    prose keys, all one failure. So this resolves through the declared family map
    (`evolve_memory.families`), and falls back to normalized prose when no family matches, so an
    unrecognized mode keeps its own identity rather than joining a catch-all.

    This is the single chokepoint: candidate fingerprints, ledger retrieval and residual
    downweighting all call it, so all three key on the family id and never on raw prose.
    """
    from anchoropt.evolve_memory.families import family_of
    return family_of(residual_key)


def candidate_fingerprint(*, residual: str, locus: str, signal: str, action: str,
                          eta: Mapping[str, Any] | None = None) -> str:
    """The canonical key. Two candidates share it iff they are the same intervention on the same
    residual family at the same place on the same condition with the same parameters."""
    payload = {
        "residual_family": residual_family(residual),
        "locus": normalize_text(locus),
        "signal": normalize_text(signal),
        "action_family": normalize_text(action),
        "eta": normalize_eta(eta),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def fingerprint_parts(*, residual: str, locus: str, signal: str, action: str,
                      eta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The same key's components, for logging and for human-readable skip reasons."""
    return {"residual_family": residual_family(residual), "locus": normalize_text(locus),
            "signal": normalize_text(signal), "action_family": normalize_text(action),
            "eta": normalize_eta(eta),
            "fingerprint": candidate_fingerprint(residual=residual, locus=locus, signal=signal,
                                                 action=action, eta=eta)}


def context_key(*, host: str, split: str, backend: str) -> str:
    """What makes two evaluations COMPARABLE.

    A result is only reusable if it was measured on the same host, split and backend. Skipping a
    candidate because it failed on a DIFFERENT cell would be exactly the error the corpus-splits work
    warned about: cells are indivisible and a rec_sum result says nothing about vector.
    """
    return f"{normalize_text(host)}/{normalize_text(split)}/{normalize_text(backend)}"
