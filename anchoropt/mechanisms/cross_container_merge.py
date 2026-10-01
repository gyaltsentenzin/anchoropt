#!/usr/bin/env python3
"""A9: the model searched ONE container, and the answer is in the other one.

The vector backend keeps two containers -- `core` and `archival` -- and the model chooses which to
search. When it retrieves from `core` and every entry comes back weakly matched, the usual reading is
"the store does not hold this". Measured, that reading was wrong: in 13 of 13 verified cases the
information WAS present, in archival, and the model never issued an archival read at all.

    a core retrieve comes back with max similarity < 0.30
      -> also dispatch the SAME query against archival
      -> rank the UNION of both containers globally by the harness's own similarity scores
      -> return the top k, replacing the weak core result AT ITS OWN POSITION

The containers are lopsided, which is what makes this worth doing: `vector-finance` holds 5 core
entries against 30 archival, so a merged top-5 ranks over 35 candidates instead of 5.

THE "ADDITIVE BY CONSTRUCTION" SAFETY CLAIM IS RETRACTED -- READ THIS BEFORE REUSING THE PATTERN
-----------------------------------------------------------------------------------------------
An earlier version of this docstring, and of A9's acceptance rationale, said:

    "ADDITIVE by construction: the union is ranked, so a core entry that still ranks in the top k is
     retained. A false firing costs one extra read, NEVER A DISPLACED CORRECT ANSWER."

**The last clause is false, and it was measured false.** Both held-out losses had
`core_topk_before = 5` and `core_entries_retained = 0`: archival out-scored core by roughly 2x, so
global ranking evicted EVERY core entry from the returned top-k. On `89-student-9` the control answered
correctly from a core entry that the merge dropped; the arm answered from archival and was wrong. The
model did not overlook the evidence -- it was not there.

The reasoning error is worth naming because it is the same shape as the delivery defect below: it argued
from a property of the MECHANISM (union ranking preserves relative order) to an OUTCOME (a used entry
survives). Order preservation says nothing about which entries fall below the cut. And displacement is
not the exceptional case -- it is the EXPECTED one, since this anchor only fires when the searched store
scored weakly, which is exactly when the sibling is likely to dominate it.

    CORRECTED, and this is what the anchor is accepted on:
    the merge returns the globally best top-k of the union. When the sibling out-ranks the primary,
    primary entries are DROPPED. Measured, that trades 15 gains / 0 losses on train and 9 gains / 2
    losses on dev. It is NET-POSITIVE, NOT NON-DESTRUCTIVE.

Two consequences for anyone extending this. Do not reuse "additive by construction" to justify a
merge-style anchor -- the phrase is retracted. And the obvious repair (reserve k slots for the primary
store, or require the sibling to beat it by a margin) is a NEW MECHANISM needing its own frozen criteria
and its own paired run: a "retain at least one primary entry" guard would have addressed both losses,
but it would also have changed the 24 wins, so its net effect is unknown rather than positive.

Ties still break toward the primary store, which is a real and narrow guarantee: an EQUAL-scoring
sibling entry cannot evict one the model already had. That is all it says.

The rejected alternative was winner-takes-container -- use the sibling's list if it scores higher. It was
dropped for a measurable reason rather than an aesthetic one: it cannot discriminate. Firing cases that
already PASS have core_max median 0.159 against 0.172 for the ones that fail, the same band, so a
container-level switch would swap out correct results as often as wrong ones.

NO PROXY SCORES
---------------
Both containers return similarity on the same scale, which is what makes a global re-rank well-defined
at all. The bag-of-words ranking used offline for diagnosis was only 73% top-1 faithful and is
deliberately NOT used here -- the same rule that appears throughout this project: a lexical proxy may
propose, it may never decide.

THE DELIVERY DEFECT THIS ANCHOR IS NAMED FOR
--------------------------------------------
Read this before trusting any anchor's telemetry, including this one's.

The first two versions of A9 computed the correct merged payload on **all 78 firings and delivered it on
none**. The gate ran ~190 lines below the point where the model's prompt is assembled, and wrote to the
telemetry sidecar rather than to the list the prompt is built from. Its write to the real list was dead
code -- immediately followed by `continue`, so the loop restarted and rebuilt the list before anything
read it.

Every downstream conclusion was consistent and worthless. The arm scored EXACTLY the control, to two
decimals, across two independent runs -- the signature of a computationally inert anchor -- and that
+0.00pp was read as "the model won't use the evidence". Four further experiments were built on top of
that reading and all four are void.

    The fix changed WHERE the result is applied, and nothing else.

The merge itself is byte-identical: same trigger, same threshold, same dispatch, same global ranking,
same tie-break, same top-k. `execution_results` is now the single authoritative object and the telemetry
list is DERIVED from it -- the prior form ran two independent index checks against two lists that need
not even be the same length, so the telemetry could record a positional replacement while the model
input silently took an append fallback.

Hence `verify_delivery` below, and the general rule in `docs/CONSUMER_BOUNDARY_RULE.md`: **validate an
anchor at the boundary its CONSUMER reads, not at the gate boundary.** Gate-side telemetry establishes
intent. It never establishes effect.

WHAT IS GENERAL HERE, AND WHAT IS NOT
-------------------------------------
The DECISION is substrate-independent and is stated that way in code:

    when a search of one store comes back weakly matched, search the SIBLING store with the same
    query and rank the union, preferring the incumbent on ties.

Nothing in that sentence mentions core, archival, similarity_score, or BFCL. Those live in
`RetrievalAdapter`, which holds the five substrate-specific facts: what the searched store is called,
what its sibling is called, how a search appears in a call, how to parse scored entries, and what
counts as weak. Every function below is generic over it, and `tests/test_cross_container_merge.py`
demonstrates that by running the same logic on an invented substrate with different store names, a
different call syntax, a different threshold and a CSV payload -- adapter only, no edits.

THE PRECONDITION THAT ACTUALLY LIMITS REUSE is not naming, it is comparability: both stores must score
on ONE SCALE, or a global re-rank compares incomparable numbers and the additivity argument collapses.
That is why this is a structural no-op on kv and rec_sum rather than a tuning decision.

ON THE ACTION FAMILY, recorded rather than glossed: A9 is filed as REROUTE, and it substitutes no
call. The searched retrieve already RAN and SUCCEEDED; this adds one call and rewrites the RESULT of
the first before the model reads it. A1/A5/A7 all end by retrying the original call, which is what
REROUTE actually describes. The honest name for what happens here is AUGMENT-THE-OBSERVATION. Not
fixed by adding a fifth family on the strength of one anchor -- see `A9_EVIDENCE`.

MEASURED: vector train 60.7% vs 43.8% control (+16.9 pp, 15 gained / 0 lost, mechanism 15/15); vector
dev 52.5% vs 35.0% (+17.5 pp, 9 / 2, mechanism 9/9). Both are VECTOR-SHARD figures -- see `A9_EVIDENCE`
in rounds/anchors.py, because reading 60.7% as a corpus number overstates this anchor by about 3x.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

# The trigger: a core retrieve whose best match is weak. Fixed at 0.30 by a rule frozen BEFORE the
# threshold was selected, so it is not a value tuned against the outcome.
THRESHOLD = 0.30

CORE_RETRIEVE_RE = re.compile(r"\bcore_memory_retrieve\s*\(")
QUERY_RE = re.compile(r"query\s*=\s*(['\"])(.*?)\1", re.DOTALL)
TOPK_RE = re.compile(r"top_k\s*=\s*([0-9]+)")

# The harness's own payload shape. Parsed rather than recomputed: these are the scores the store
# assigned, and the whole merge depends on both containers being on one scale.
_ENTRY_RE = re.compile(
    r'"id"\s*:\s*([0-9]+)\s*,\s*"similarity_score"\s*:\s*([0-9.]+)\s*,'
    r'\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"'
)

DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class RetrievalAdapter:
    """Everything substrate-specific about a two-store retrieval surface. The only thing a port writes.

    THE DECISION IS GENERAL; THE VOCABULARY IS NOT
    ----------------------------------------------
    The principle A9 implements has nothing to do with BFCL:

        when a search of one store comes back weakly matched, search the SIBLING store with the same
        query and rank the union, preferring the incumbent on ties.

    What is substrate-specific is only: what the searched store is called, what its sibling is called,
    how to phrase a search call, how to parse scored entries out of a result, and what counts as weak.
    Those five facts live here, so the functions below never name a container.

    THE PRECONDITION, and it is the one that actually restricts reuse: `primary` and `sibling` must
    return scores ON THE SAME SCALE. A global re-rank across two stores is meaningless otherwise. This
    is exactly why A9 is a structural no-op on the kv and rec_sum backends -- not a tuning choice, and
    not something a port can wave through: if the two stores score differently, a merged ranking is
    comparing incomparable numbers and the additivity argument collapses with it.
    """

    name: str
    primary: str                    # the store the model searched (BFCL: "core")
    sibling: str                    # the store it did not search (BFCL: "archival")
    threshold: float = 0.30
    # How a search of `primary` appears in a decoded call, and how to phrase one against `sibling`.
    search_re: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"\bcore_memory_retrieve\s*\(")
    )
    sibling_call_template: str = "{store}_memory_retrieve(query='{query}',top_k={top_k})"
    # Scored entries out of a result payload. Replaced wholesale for a store with another wire format.
    parse: Callable[[Any], list[Entry]] | None = None
    notes: str = ""

    def entries_of(self, result: Any) -> list[Entry]:
        return (self.parse or parse_entries)(result)

    def sibling_call(self, query: str, top_k: int) -> str:
        """A search of the sibling store, reusing the model's OWN query verbatim.

        Rewriting the query would confound a retrieval effect with a query-reformulation effect, so
        the template interpolates it unchanged and only escapes the quote character.
        """
        return self.sibling_call_template.format(
            store=self.sibling, query=str(query).replace("'", "\\'"), top_k=int(top_k)
        )

    def is_weak(self, best_score: float) -> bool:
        """Strictly below threshold. At or above, the store already answered."""
        return best_score < self.threshold


# The default adapter, named for the SUBSTRATE SHAPE it describes rather than for a benchmark -- the
# mechanism needs only "two stores that score on one scale", and `core`/`archival` happen to be what
# the shipped vector backend calls them. Kept as a VALUE, not baked into the logic below, so a second
# substrate needs a new adapter and no edit to any function here.
#
# (The core must not name a benchmark in executable code; tests/test_adapter_parity.py enforces that,
# and it caught an earlier draft of this constant called CORE_ARCHIVAL.)
CORE_ARCHIVAL = RetrievalAdapter(
    name="core_archival_scored_pair",
    primary="core",
    sibling="archival",
    threshold=THRESHOLD,
    notes=(
        "Both containers return the harness's own similarity_score on one scale, which is what makes "
        "the global re-rank well-defined. kv and rec_sum expose no per-entry scores at all, so there "
        "is no adapter for them and the trigger is undefined there."
    ),
)


class Entry(NamedTuple):
    """One retrieved entry. `source` is which container it came from, and it is load-bearing."""

    score: float
    source: str          # "core" | "archival"
    entry_id: int
    text: str


def parse_entries(result: Any) -> list[Entry]:
    """`[(score, source, id, text)]` from a vector retrieve payload, best first.

    `source` is filled in by the caller via `tag`; this returns them untagged as core-shaped tuples
    because the parse is identical for both containers.
    """
    out = [
        Entry(float(m.group(2)), "", int(m.group(1)), m.group(3))
        for m in _ENTRY_RE.finditer(str(result or ""))
    ]
    out.sort(key=lambda e: -e.score)
    return out


def tag(entries: Iterable[Entry], source: str) -> list[Entry]:
    """Stamp a container name onto parsed entries."""
    return [e._replace(source=source) for e in entries]


def merge(
    primary_entries: Sequence[Entry],
    sibling_entries: Sequence[Entry],
    top_k: int,
    *,
    primary: str = "core",
    sibling: str = "archival",
) -> list[Entry]:
    """Rank the UNION globally by score and return the best `top_k`.

    Two properties, both required by the acceptance argument, and neither of them substrate-specific:
      * TIES BREAK TOWARD THE INCUMBENT -- a sibling entry scoring exactly EQUAL cannot evict one the
        model already had. Ordering is otherwise by score alone, regardless of which store it came from.
        This is the only guarantee here, and it is narrow.
      * NOT "additive" -- that claim is RETRACTED (see the module docstring). A primary entry that
        falls below the cut IS dropped, and when the sibling dominates on score the whole primary set
        can be evicted. Both of A9's held-out losses are exactly that, with
        `core_entries_retained = 0`. Callers must treat displacement as expected and measure the
        trade, not assume non-destructiveness.

    `primary`/`sibling` are the store LABELS, defaulted to the BFCL pair so existing callers are
    unaffected. They appear only in the tags and the tie-break.
    """
    tagged = list(tag(primary_entries, primary)) + list(tag(sibling_entries, sibling))
    tagged.sort(key=lambda e: (-e.score, 0 if e.source == primary else 1))
    return tagged[: max(1, int(top_k or DEFAULT_TOP_K))]


def archival_query_for(query: str, top_k: int) -> str:
    """The synthetic archival read, in the backend's own call syntax.

    The SAME query the model already wrote. Rewriting it would make this a query-reformulation anchor
    and confound the retrieval effect with a rewriting effect.
    """
    escaped = str(query).replace("'", "\\'")
    return f"archival_memory_retrieve(query='{escaped}',top_k={int(top_k)})"


def plan_merge(
    calls: Sequence[Any] | None,
    results: Sequence[Any] | None,
    *,
    is_vector: bool = True,
    looks_like_error=None,
    adapter: RetrievalAdapter = CORE_ARCHIVAL,
) -> dict[str, Any]:
    """Decide whether to merge. Reads state and dispatches nothing.

    Returns `{"fire": False, "declined": <reason>}` on every non-firing path. Each decline is a
    distinct named reason rather than a bare False, because "the trigger did not hold" and "archival
    came back empty" are different facts about the store and only one of them is a ladder candidate.
    """
    if not is_vector:
        # Structural, not a tuning choice: the trigger needs comparable per-entry scores from BOTH
        # stores, and the kv and rec_sum containers expose none. This is also the precondition a port
        # must check -- see RetrievalAdapter. It is why A9 is a measured no-op on the other backends.
        return {"fire": False, "declined": "backend has no per-entry similarity scores"}

    for index, (call, result) in enumerate(zip(calls or (), results or ())):
        text = str(call)
        if not adapter.search_re.search(text):
            continue
        match = QUERY_RE.search(text)
        if not match:
            continue                     # retrieve_all carries no query; a different locus
        if looks_like_error is not None and looks_like_error(result):
            continue                     # the error branch belongs to the capacity anchors
        core = adapter.entries_of(result)
        if not core:
            continue

        core_max = core[0].score
        if not adapter.is_weak(core_max):
            return {"fire": False, "declined": "core_confident", "core_max": core_max}

        topk_match = TOPK_RE.search(text)
        top_k = int(topk_match.group(1)) if topk_match else DEFAULT_TOP_K
        return {
            "fire": True,
            "reason": (
                f"{adapter.primary}'s best match is {core_max:.3f} (< {adapter.threshold}); search "
                f"{adapter.sibling} with the same query and rank both stores together"
            ),
            "query": match.group(2),
            "top_k": top_k,
            "core_max": core_max,
            "core_index": index,         # WHERE the result goes back. v1 appended and left the
                                         # weak original ahead of the improvement.
            "archival_call": adapter.sibling_call(match.group(2), top_k),
            "core_entries": core,
            "adapter": adapter.name,
        }

    return {"fire": False, "declined": f"no weak {adapter.primary} search in this turn"}


def complete_merge(
    plan: Mapping[str, Any],
    archival_result: Any,
    *,
    adapter: RetrievalAdapter = CORE_ARCHIVAL,
) -> dict[str, Any]:
    """Finish a fired plan once the archival read has returned.

    Split from `plan_merge` so the dispatch itself stays in the harness: this module decides and
    ranks, the caller executes. That also makes the whole decision testable with no store at all.
    """
    if not plan.get("fire"):
        return {"fire": False, "declined": "plan did not fire"}
    if not archival_result:
        return {"fire": False, "declined": "archival_dispatch_failed"}

    archival = adapter.entries_of(archival_result)
    if not archival:
        return {"fire": False, "declined": "archival_empty", "core_max": plan.get("core_max")}

    core: Sequence[Entry] = plan["core_entries"]
    top_k = int(plan["top_k"])
    merged = merge(core, archival, top_k,
                   primary=adapter.primary, sibling=adapter.sibling)

    # NON-DESTRUCTIVENESS, measured rather than asserted: of the core entries that would still have
    # ranked in the top k, how many survived. Reported per firing so a violation is visible in
    # telemetry instead of being inferred later from an accuracy delta.
    kept = {e.entry_id for e in merged if e.source == adapter.primary}
    would_rank = {e.entry_id for e in core[:top_k]}

    return {
        "fire": True,
        "core_max": plan["core_max"],
        "archival_max": archival[0].score,
        "new_max": merged[0].score if merged else None,
        "core_n": len(core),
        "archival_n": len(archival),
        "archival_promoted": sum(1 for e in merged if e.source == adapter.sibling),
        "core_entries_retained": len(kept),
        "core_topk_before": len(would_rank),
        "core_dropped": sorted(would_rank - kept),
        "merged_returned": len(merged),
        "core_index": plan["core_index"],
        "payload": json.dumps(
            {"result": [{"id": e.entry_id, "similarity_score": e.score, "text": e.text}
                        for e in merged]}
        ),
        # NOT called an "invariant": what survives is whatever still ranks, which can be nothing.
        # Reported so displacement is VISIBLE per firing rather than inferred from an accuracy delta.
        "displacement": (
            f"{len(would_rank) - len(kept)} of {len(would_rank)} primary entries that would have "
            f"ranked in the top {top_k} were dropped"
        ),
    }


def verify_delivery(model_facing_text: Any, outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Was the merged payload actually delivered to the object the MODEL reads?

    This function exists because of a real defect, not as belt-and-braces. A9 computed the right
    answer on 78 firings and delivered it on 0: the gate wrote to the telemetry sidecar while the
    prompt was built from a different list, and `replaced=True` was *true of the sidecar*. Four
    downstream experiments were voided.

    So the check is deliberately dumb and end-of-pipeline: take the top merged entry's text and look
    for it in the text the model was actually sent. A structural assertion about indices is exactly
    what failed before.

    `attested=False` means NOT MEASURED, which is not the same as not delivered -- conflating those
    two is the original error in mirror image. Report them separately.
    """
    if not outcome.get("fire"):
        return {"attested": False, "reason": "no firing to attest"}
    if model_facing_text is None:
        return {"attested": False, "reason": "prompt not captured -- abstaining, NOT a failure"}

    try:
        entries = json.loads(outcome["payload"])["result"]
    except (KeyError, ValueError, TypeError):
        return {"attested": False, "delivered": False, "reason": "payload unreadable"}
    if not entries:
        return {"attested": False, "delivered": False, "reason": "payload empty"}

    probe = str(entries[0].get("text") or "")[:80]
    haystack = str(model_facing_text)
    delivered = bool(probe) and probe in haystack
    return {
        "attested": True,
        "delivered": delivered,
        "probe": probe[:40],
        "reason": (
            "the merged top entry is present in the model-facing text" if delivered
            else "DELIVERY VIOLATION: computed but absent from what the model was sent"
        ),
    }


def dispatch_sequence(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """The one call this anchor adds. Stated as data so a port can assert it."""
    return (str(plan["archival_call"]),) if plan.get("fire") else ()
