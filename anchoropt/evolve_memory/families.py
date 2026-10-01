"""Canonical residual FAMILY ids, so paraphrases stop fragmenting the ranking.

THE DEFECT THIS FIXES, measured in SE1
--------------------------------------
Re-mining after the A4' promotion produced **65 diagnoses -> 17 "distinct" residuals**, every one a
paraphrase of the same failure: *the model answers although retrieval returned nothing useful.* The
grouping key was the attributor's raw `consequential_decision` prose, so the residual shape read
`4/4/4/.../1` -- sixteen ties at support 4, none of which is really support 4. The support ranking then
picked whichever paraphrase sorted first, and coverage collapsed from 50% to 6.2% for what is
structurally ONE problem with support 64.

So: collapse to a family id BEFORE ranking. Ranking, experiment memory and downweighting all key on
the family, never on the prose.

HOW THE MATCHING WORKS, and why it is not embeddings
----------------------------------------------------
A family is a small ordered list of **required-term groups**: a decision matches if it contains at
least one term from every group. That is deterministic, inspectable, and cheap to audit -- and a
mis-grouping is visible as a rule rather than hidden in a vector. Order matters: the first matching
family wins, so narrower families are declared first.

CONSERVATISM RUNS THE OTHER WAY HERE than in candidate dedup. A false candidate match silently skips a
real experiment, so there the rule is exact-only. A false FAMILY match merely groups two failure modes
that should have been separate, which shows up as a family whose diagnoses disagree -- recoverable, and
`unmatched_report` exists to surface it. Anything that matches no family keeps its own normalized prose
as its id, so nothing is ever silently merged into a catch-all.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anchoropt.evolve_memory.fingerprint import normalize_text


@dataclass(frozen=True)
class Family:
    """One canonical residual family.

    `require` is a list of alternative-groups; ALL groups must hit, any term within a group suffices.
    """

    family_id: str
    label: str
    require: tuple[tuple[str, ...], ...]
    exclude: tuple[str, ...] = ()
    note: str = ""

    def matches(self, text: str) -> bool:
        t = normalize_text(text)
        if any(x in t for x in self.exclude):
            return False
        return all(any(term in t for term in group) for group in self.require)


# Declared narrowest-first. Every family here is grounded in diagnoses actually observed in a round.
FAMILIES: tuple[Family, ...] = (
    Family(
        family_id="answer_without_any_tool_call",
        label="commits to a final answer having made no tool call at all",
        require=(("no tool call", "without calling", "without querying memory", "never queried",
                  "no memory query", "without any retrieval", "without consulting"),
                 ("answer", "respond", "final")),
        exclude=("returned no", "empty retrieval", "zero results"),
        note="the A4 family: the trigger is that NOTHING was called, so there is no result to judge.",
    ),
    Family(
        family_id="answer_on_weak_first_retrieval",
        label="accepts the first weak-but-non-empty retrieval as sufficient evidence",
        require=(("first", "single", "one"),
                 ("retrieval", "retrieved", "result"),
                 ("sufficient", "adequate", "accept", "enough", "without checking",
                  "without verifying")),
        exclude=("zero results", "no results", "returned no", "zero supporting"),
        note="the low-similarity family (R1/R2): the retrieval SUCCEEDED weakly, which is a different "
             "condition from returning nothing, and the two need different interventions. NOTE the "
             "exclude list must not contain a bare 'empty' -- these decisions say 'non-empty', which "
             "CONTAINS 'empty', and a substring test would send them to the empty-result family.",
    ),
    Family(
        family_id="answer_despite_unhelpful_retrieval",
        label="commits to a final answer although retrieval returned nothing usable",
        require=(("empty", "no result", "no results", "zero result", "zero results",
                  "no supporting", "no usable", "unhelpful", "no retrieved", "nothing useful",
                  "returned no", "zero supporting", "no evidence", "no matching",
                  "nothing was retrieved", "without any retrieved"),
                 ("answer", "respond", "final", "commit")),
        exclude=("non-empty", "nonempty", "non empty"),
        note="SE1 re-mining produced 17 prose variants of exactly this; support 64 as one family.",
    ),
    Family(
        family_id="answer_from_context_without_memory",
        label="answers an episode-specific question from context instead of querying memory",
        require=(("from context", "context alone", "from the prompt", "without first querying",
                  "instead of first querying"),
                 ("answer", "respond")),
    ),
)


def family_of(consequential_decision: str) -> str:
    """The canonical family id for one diagnosis's decision text.

    Falls back to the normalized prose when nothing matches, so an unrecognized failure mode keeps its
    own identity instead of being merged into a catch-all bucket.
    """
    for fam in FAMILIES:
        if fam.matches(consequential_decision):
            return fam.family_id
    return normalize_text(consequential_decision)


def family_label(family_id: str) -> str:
    for fam in FAMILIES:
        if fam.family_id == family_id:
            return fam.label
    return family_id


def is_canonical(family_id: str) -> bool:
    """Whether this id came from a declared family rather than from fallback prose."""
    return any(f.family_id == family_id for f in FAMILIES)


def group_diagnoses(diagnoses: Sequence[Any]) -> dict[str, list[Any]]:
    """Group diagnosis objects by canonical family, preserving input order within a family."""
    out: dict[str, list[Any]] = {}
    for d in diagnoses:
        key = family_of(getattr(d, "consequential_decision", "") or "")
        out.setdefault(key, []).append(d)
    return out


def unmatched_report(diagnoses: Sequence[Any]) -> list[tuple[str, int]]:
    """(fallback id, count) for diagnoses that matched NO declared family, commonest first.

    This is the audit surface: a large unmatched group means a family is missing, and a family whose
    members disagree means one is too broad. Both are visible here rather than buried in a ranking.
    """
    counts: dict[str, int] = {}
    for d in diagnoses:
        fid = family_of(getattr(d, "consequential_decision", "") or "")
        if not is_canonical(fid):
            counts[fid] = counts.get(fid, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])
