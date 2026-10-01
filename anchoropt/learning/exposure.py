#!/usr/bin/env python3
"""Two diagnostics every anchor must pass before its delta means anything.

Both exist because a positive number was believed on this project and turned out to measure something
else. They are cheap, they need no GPU, and they read from artifacts a paired run already produces.

    exposure_weighted_delta   is the gain INSIDE the population the anchor can act on?
    displacement_check        did the anchor ADD capability, or take work from an incumbent anchor?

WHY EXPOSURE
------------
Most anchors here are gated to one backend, so the other backends are untouched BY CONSTRUCTION --
which makes them a free noise estimate on every paired run. Use it. One candidate scored +3.63 pp
train and +7.14 pp held out, both positive, no reversal, and was rejected: its guard fired ZERO times,
and the entire gain sat in the two backends it cannot touch.

    backend            n     delta      guard fires?
    vector (exposed)   89    +1.12 pp   0 times
    kv                105   +4.76 pp    never
    rec_sum           109   +4.59 pp    never

A noise floor of +4.67 pp on 214 unexposed cases against +1.12 pp on the 89 exposed ones. The delta
measured store perturbation, not the mechanism -- any intervention in a prerequisite episode perturbs
the store, and perturbation alone moves scores everywhere. **Engagement of zero with a positive delta
is proof of NON-attribution, never evidence of a subtle effect.**

The same arithmetic runs the other way for reporting. An anchor gated to a backend that is 36% of the
corpus cannot be quoted at its within-backend figure: +7.34 pp on that backend is +2.64 pp corpus-wide.
Quote both, always, with the exposure beside them.

WHY DISPLACEMENT
----------------
An anchor can improve the arm while ADDING NOTHING, by pre-empting an incumbent anchor on the same
payloads. A6 fires 44 times held-out while A1's dispatches collapse 22 -> 6: it catches the payload one
error earlier and reroutes it before the state A1 acts on can form. On a saturated store that cannot add
information -- it only reorders which entries win a fixed number of slots, which is a coin flip on
phrasing alignment and is exactly why A6 reverses sign between splits.

So compare per-site dispatch counts between the incumbent and the candidate. If an incumbent site
COLLAPSES while the new site rises, the candidate is a SELECTION intervention wearing a capacity
costume, and its train gain will not transfer. See `docs/SATURATION_AND_SELECTION.md`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cell:
    """One (population, outcome) pair from a paired run: the same cases under control and arm."""

    name: str
    n: int
    control_correct: int
    arm_correct: int
    exposed: bool          # can the anchor act on this population AT ALL?
    firings: int = 0       # measured on the ARM's own telemetry, never on another arm's

    @property
    def delta_pp(self) -> float:
        if not self.n:
            return 0.0
        return 100.0 * (self.arm_correct - self.control_correct) / self.n


@dataclass(frozen=True)
class ExposureReport:
    """The verdict inputs, separated so an unexposed gain can never be read as an effect."""

    exposed_delta_pp: float
    unexposed_delta_pp: float
    corpus_delta_pp: float
    exposed_n: int
    unexposed_n: int
    corpus_n: int
    firings: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def attributable(self) -> bool:
        """Is the delta attributable to the mechanism at all?

        Requires that the gate actually fired. It deliberately does NOT require the exposed delta to
        beat the unexposed one -- that is a judgement for the reader with the warnings in hand, not a
        threshold this function should quietly apply.
        """
        return self.firings > 0

    def summary(self) -> str:
        lines = [
            f"exposed    n={self.exposed_n:<4} delta={self.exposed_delta_pp:+.2f} pp  "
            f"firings={self.firings}",
            f"unexposed  n={self.unexposed_n:<4} delta={self.unexposed_delta_pp:+.2f} pp  "
            "(free noise estimate; the anchor cannot act here)",
            f"corpus     n={self.corpus_n:<4} delta={self.corpus_delta_pp:+.2f} pp  "
            "<- the figure to quote, with exposure stated",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def exposure_weighted_delta(cells: Sequence[Cell]) -> ExposureReport:
    """Split a paired result by exposure and flag the patterns that mean "not your effect".

    The corpus figure is the one to publish; the exposed/unexposed split is what licenses calling it
    an effect. Reporting only the corpus number hides non-attribution, and reporting only the exposed
    number overstates the anchor's reach.
    """
    exposed = [c for c in cells if c.exposed]
    unexposed = [c for c in cells if not c.exposed]

    def pooled(group: Sequence[Cell]) -> tuple[float, int]:
        n = sum(c.n for c in group)
        if not n:
            return 0.0, 0
        gained = sum(c.arm_correct - c.control_correct for c in group)
        return 100.0 * gained / n, n

    exposed_delta, exposed_n = pooled(exposed)
    unexposed_delta, unexposed_n = pooled(unexposed)
    corpus_delta, corpus_n = pooled(cells)
    firings = sum(c.firings for c in exposed)

    warnings: list[str] = []
    if firings == 0 and corpus_delta > 0:
        warnings.append(
            "the gate fired 0 times and the delta is positive -- this is proof of "
            "NON-attribution, not evidence of a subtle mechanism"
        )
    if unexposed_n and unexposed_delta >= max(exposed_delta, 0.0):
        warnings.append(
            f"the unexposed noise floor ({unexposed_delta:+.2f} pp on n={unexposed_n}) is at or "
            f"above the exposed delta ({exposed_delta:+.2f} pp on n={exposed_n}) -- the arm is "
            "most likely measuring store perturbation"
        )
    if any(c.firings and not c.exposed for c in cells):
        warnings.append(
            "a cell marked unexposed recorded firings -- the exposure rule is wrong, "
            "so fix it before reading the delta"
        )
    if exposed_n and corpus_n > exposed_n:
        share = 100.0 * exposed_n / corpus_n
        warnings.append(
            f"the anchor reaches {share:.0f}% of the corpus, so the within-exposure figure "
            f"({exposed_delta:+.2f} pp) must never be quoted without that exposure beside it"
        )
    return ExposureReport(
        exposed_delta_pp=round(exposed_delta, 4),
        unexposed_delta_pp=round(unexposed_delta, 4),
        corpus_delta_pp=round(corpus_delta, 4),
        exposed_n=exposed_n,
        unexposed_n=unexposed_n,
        corpus_n=corpus_n,
        firings=firings,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class DisplacementReport:
    """Per-site dispatch movement between the incumbent and a candidate."""

    new_sites: Mapping[str, int]
    collapsed_sites: Mapping[str, tuple[int, int]]
    grew_sites: Mapping[str, tuple[int, int]]
    verdict: str
    detail: str

    @property
    def displaces_incumbent(self) -> bool:
        return bool(self.collapsed_sites) and bool(self.new_sites)


def displacement_check(
    incumbent_sites: Mapping[str, int],
    candidate_sites: Mapping[str, int],
    collapse_ratio: float = 0.5,
) -> DisplacementReport:
    """Did the candidate ADD capability, or take work away from an incumbent anchor?

    `*_sites` map a dispatch-site name to how many times it fired -- read from each arm's OWN
    telemetry. A site that appears only in the candidate is the new anchor's own work. A site that
    falls to `collapse_ratio` or less of its incumbent count has been PRE-EMPTED.

    Both together is the signature to worry about: on a store at its capacity the candidate cannot be
    adding information, so it is choosing which entries survive. That gain is conditioned on phrasing
    alignment and will not transfer.
    """
    new_sites = {
        site: count for site, count in candidate_sites.items()
        if count > 0 and incumbent_sites.get(site, 0) == 0
    }
    collapsed: dict[str, tuple[int, int]] = {}
    grew: dict[str, tuple[int, int]] = {}
    for site, before in incumbent_sites.items():
        after = candidate_sites.get(site, 0)
        if before > 0 and after <= before * collapse_ratio:
            collapsed[site] = (before, after)
        elif after > before:
            grew[site] = (before, after)

    if new_sites and collapsed:
        verdict = "SELECTION"
        detail = (
            "the candidate fires where an incumbent anchor previously acted ("
            + ", ".join(f"{s}: {b} -> {a}" for s, (b, a) in sorted(collapsed.items()))
            + "). On a saturated store it cannot add information, only reorder which entries "
              "survive, so treat the train gain as unlikely to transfer."
        )
    elif new_sites:
        verdict = "ADDITIVE"
        detail = (
            "the candidate opened new dispatch sites ("
            + ", ".join(f"{s}: {c}" for s, c in sorted(new_sites.items()))
            + ") without collapsing any incumbent site."
        )
    elif not candidate_sites or sum(candidate_sites.values()) == 0:
        verdict = "INERT"
        detail = "the candidate recorded no dispatches; any delta is not attributable to it."
    else:
        verdict = "NO CHANGE"
        detail = "dispatch sites are unchanged; the candidate is not doing anything new."

    return DisplacementReport(
        new_sites=dict(sorted(new_sites.items())),
        collapsed_sites=dict(sorted(collapsed.items())),
        grew_sites=dict(sorted(grew.items())),
        verdict=verdict,
        detail=detail,
    )
