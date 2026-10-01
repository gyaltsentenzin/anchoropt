"""Should this candidate be evaluated, reused, or skipped? -- and how should a tried residual rank?

THREE VERDICTS, NOT TWO. The SE1 round produced one of each and they need different handling:

    EVALUATE  never validly tried in a comparable context -> spend the GPU
    REUSE     tried and measured here -> report the prior outcome, spend nothing
    INSTALLED already ACCEPTED and in the incumbent -> evaluating it measures the incumbent against
              itself, which is not a null result, it is a meaningless one

Collapsing INSTALLED into REUSE would report the promoted anchor's original delta as if it were a new
finding. SE1's third proposal was exactly the promoted A4' controller -- same locus, same signal, same
eta -- so this is not hypothetical.

RESIDUAL DOWNWEIGHTING. After a promotion, re-mining surfaces the same residual families again; the
ones already attacked and failed should not keep winning the ranking on support alone. `residual_penalty`
returns a multiplicative weight from the ledger, and `rank_key` orders residuals by penalized support.
It DOWNWEIGHTS rather than excludes on purpose: an exhausted residual is not refuted -- the search space
offered for it was, and a later host or signal may reopen it. Excluding it would lose that.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from anchoropt.evolve_memory.fingerprint import candidate_fingerprint, residual_family
from anchoropt.evolve_memory.ledger import Experiment, ExperimentLedger

EVALUATE, REUSE, INSTALLED = "evaluate", "reuse", "installed"


@dataclass(frozen=True)
class Verdict:
    """What to do with one candidate, and why -- the reason is shown to the proposer."""

    decision: str
    fingerprint: str
    prior: Experiment | None = None
    reason: str = ""

    @property
    def spend_gpu(self) -> bool:
        return self.decision == EVALUATE


def _intervention_key(locus: str, signal: str, action: str,
                      eta: Mapping[str, Any] | None) -> str:
    """The candidate WITHOUT its residual: what would actually be installed and run.

    Two residuals can motivate the same controller. Whether it is already in the incumbent, or was
    already measured on this cell, is a property of the INTERVENTION -- so those checks must not be
    scoped to the residual that happened to propose it.
    """
    return candidate_fingerprint(residual="", locus=locus, signal=signal, action=action, eta=eta)


def screen_candidate(ledger: ExperimentLedger, *, residual: str, locus: str, signal: str,
                     action: str, eta: Mapping[str, Any] | None = None,
                     context: str | None = None) -> Verdict:
    """Conservative screen: only an EXACT normalized match in a COMPARABLE context can block.

    Matching happens on the INTERVENTION (locus, signal, action, eta), not on (residual + those).
    SE1 re-proposed the already-promoted A4' controller under a DIFFERENT residual family; a
    residual-scoped screen reports that as "no comparable prior evaluation" and buys it again. The
    residual still determines RELEVANCE for what is shown to the proposer -- it just cannot make an
    installed controller look new.
    """
    fp = candidate_fingerprint(residual=residual, locus=locus, signal=signal, action=action, eta=eta)
    ikey = _intervention_key(locus, signal, action, eta)
    hits = [e for e in ledger.experiments()
            if e.status != "invalid"
            and _intervention_key(e.locus, e.signal, e.action, e.eta) == ikey]
    if context is not None:
        hits = [e for e in hits if not e.context or e.context == context]
    if not hits:
        return Verdict(EVALUATE, fp, None, "no comparable prior evaluation of this intervention")
    accepted = [e for e in hits if e.status == "accepted"]
    if accepted:
        e = max(accepted, key=lambda x: x.ts)
        return Verdict(INSTALLED, fp, e,
                       f"already ACCEPTED in {e.round_id} and installed in the incumbent; "
                       f"evaluating it would measure the incumbent against itself")
    e = max(hits, key=lambda x: x.ts)
    same_family = residual_family(e.residual) == residual_family(residual)
    scope = "same residual family" if same_family else f"proposed then for a different family"
    return Verdict(REUSE, fp, e, f"already evaluated in {e.round_id} ({scope}): {e.one_line()}")


def screen_all(ledger: ExperimentLedger, candidates: Sequence[Mapping[str, Any]], *,
               residual: str, context: str | None = None) -> list[tuple[Mapping[str, Any], Verdict]]:
    """Screen a whole candidate list. Order is preserved so a caller's ranking survives."""
    out = []
    for c in candidates:
        v = screen_candidate(ledger, residual=residual, locus=str(c.get("locus", "")),
                             signal=str(c.get("signal", "")), action=str(c.get("action", "")),
                             eta=c.get("eta") or {}, context=context)
        out.append((c, v))
    return out


# ------------------------------------------------------------------------------------------------
# residual downweighting
# ------------------------------------------------------------------------------------------------
#
# Weights are deliberately coarse and stated here rather than tuned: this is an MVP ordering nudge,
# not a learned prior. A residual that has absorbed evaluation effort and produced nothing should fall
# behind an untouched one of equal support -- but must stay in the queue.
W_UNTRIED = 1.0
W_PER_FAILED_ATTEMPT = 0.75      # multiplicative, compounding per failed attempt
W_FLOOR = 0.25                   # never fully suppressed
W_HAS_ACCEPTED = 0.5             # an accepted anchor already claimed part of this family


def residual_penalty(ledger: ExperimentLedger, residual: str) -> tuple[float, str]:
    """(weight, human-readable why) for a residual family, from the ledger."""
    fam = residual_family(residual)
    hits = [e for e in ledger.experiments() if residual_family(e.residual) == fam]
    if not hits:
        return W_UNTRIED, "untried"
    failed = [e for e in hits if e.status in ("rejected", "infeasible")]
    accepted = [e for e in hits if e.status == "accepted"]
    w = W_UNTRIED * (W_PER_FAILED_ATTEMPT ** len(failed))
    if accepted:
        w *= W_HAS_ACCEPTED
    w = max(w, W_FLOOR)
    why = f"{len(failed)} failed attempt(s)" + (f", {len(accepted)} accepted" if accepted else "")
    return w, why


def rank_key(ledger: ExperimentLedger, residual: str, support: int) -> tuple:
    """Sort key for re-mining: penalized support DESC, then raw support DESC, then the key.

    Deterministic, so a rerun searches the same problem first.
    """
    w, _ = residual_penalty(ledger, residual)
    return (-(support * w), -support, residual_family(residual))


def reorder_residuals(ledger: ExperimentLedger,
                      problems: Sequence[Any]) -> list[tuple[Any, float, str]]:
    """(problem, weight, why) ordered by penalized support. Nothing is dropped."""
    scored = [(p, *residual_penalty(ledger, p.key)) for p in problems]
    scored.sort(key=lambda t: rank_key(ledger, t[0].key, t[0].support))
    return scored
