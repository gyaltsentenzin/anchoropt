"""RELEVANT PRIOR EXPERIENCE -- a compact, retrieved block. Never the full history.

Three sections, in the order a role needs them:

    Structural lessons          methodological constraints that apply regardless of residual
    Previously tried            concrete attempts RETRIEVED for relevance, with outcomes
    Known invalid approaches    things that produced numbers which cannot be believed

RETRIEVAL IS DELIBERATELY NARROW. Relevance = same residual family, OR same locus (optionally same
signal). Locus matters across residuals because whether an executor can realize something at a
boundary is a property of the host, not of the failure. Everything else is withheld: a role given
fifty prior attempts will pattern-match on the list instead of reasoning about the diagnosis, and the
whole point of the attributor's schema is that it reasons from runtime facts.

WHAT IS NEVER PUT IN THIS BLOCK: a locus, signal or action for the CURRENT residual. The lessons and
prior attempts describe what happened elsewhere; suggesting where to intervene here would make the
memory a proposer, and the forbidden-key guard in `proposal_seams.ingest_attribution` exists precisely
to stop the attributor from deciding that.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from anchoropt.evolve_memory.ledger import Experiment, ExperimentLedger

MAX_LESSONS = 6
MAX_ATTEMPTS = 6


def _dedup_keep_order(items: Sequence[Experiment]) -> list[Experiment]:
    seen, out = set(), []
    for e in items:
        if e.fingerprint in seen:
            continue
        seen.add(e.fingerprint)
        out.append(e)
    return out


def relevant_experiments(ledger: ExperimentLedger, *, residual: str = "", locus: str = "",
                         signal: str | None = None, limit: int = MAX_ATTEMPTS) -> list[Experiment]:
    """Prior attempts worth showing: same residual family first, then same locus."""
    hits: list[Experiment] = []
    if residual:
        hits += ledger.for_residual(residual, limit=limit)
    if locus:
        hits += ledger.for_locus(locus, signal, limit=limit)
    return _dedup_keep_order(hits)[:limit]


def experience_block(ledger: ExperimentLedger, *, residual: str = "", locus: str = "",
                     signal: str | None = None, max_lessons: int = MAX_LESSONS,
                     max_attempts: int = MAX_ATTEMPTS) -> str:
    """The compact block handed to the Attributor / Localizer / Proposer.

    Returns "" when the ledger holds nothing relevant, so a first round gets no empty scaffolding.
    """
    lessons = ledger.structural_lessons()
    if residual or locus:
        scoped = [l for l in lessons if not l.applies_to
                  or any(t in (residual + " " + locus).lower() for t in l.applies_to)]
        lessons = scoped or lessons
    lessons = lessons[:max_lessons]
    attempts = relevant_experiments(ledger, residual=residual, locus=locus, signal=signal,
                                    limit=max_attempts)
    invalid = ledger.invalid_approaches()
    if not (lessons or attempts or invalid):
        return ""

    out = ["RELEVANT PRIOR EXPERIENCE", ""]
    if lessons:
        out.append("Structural lessons:")
        for l in lessons:
            out.append(f"- {l.text}")
        out.append("")
    if attempts:
        out.append("Previously tried for similar residuals:")
        for e in attempts:
            out.append(f"- {e.one_line()}")
        out.append("")
    if invalid:
        out.append("Known invalid approaches:")
        for l in invalid:
            out.append(f"- {l.text}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
