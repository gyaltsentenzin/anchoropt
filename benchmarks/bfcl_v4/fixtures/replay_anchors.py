"""BOOTSTRAP / REPLAY FIXTURES -- the historical A1-A9 vocabulary. NOT the autonomous Phi.

WHY THIS MODULE IS SEPARATE, AND WHY THAT SEPARATION IS THE POINT
----------------------------------------------------------------
Everything here was reverse-engineered FROM the eight accepted anchors: the prose aliases are
phrases lifted from each anchor's own `attribution=` text, the reroute destinations are the repairs
those anchors perform, and the expected-signal map is the answer key for the recovery test.

That makes this data a FIXTURE, not a vocabulary. It is fitted to the answer and it would help
nothing that is not already known. Keeping it in `bfcl_runtime.py` -- where the first version of
this work put it -- would have made the accepted anchor library the design of the autonomous
search, which is exactly backwards: the runtime is supposed to expose CAPABILITIES so that
discovery can construct signals we have never seen.

So:

    bfcl_runtime.py         capabilities   observable fields, tool schema, U_H(l), execution
    bfcl_predicates.py      the closed operator set signals are BUILT from
    fixtures/replay_anchors.py   THIS FILE -- historical answers, for calibration only

The autonomous loop must never import this module. `scripts/anchor_recovery.py` and
`tests/test_bfcl_runtime.py` inject it explicitly, which is what keeps the dependency visible.

WHAT IT IS LEGITIMATELY FOR
---------------------------
Calibration. The eight anchors are the only known-good controllers available at zero GPU cost, so
"can the machinery re-select them" is the cheapest correctness test that exists. It is a test of
the RUNTIME AND SEARCH, and it answers exactly one question -- whether the representation can hold
controllers we already trust. It is not evidence that the search can discover new ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Prose phrases per signal, taken from each anchor's own attribution text.
SIGNAL_ALIASES: Mapping[str, tuple[str, ...]] = {
    "container_at_capacity": (
        "full container", "container is full", "blocked by a full", "no remaining capacity",
        "capacity-blocked", "core memory is full", "write was blocked",
    ),
    "identifier_not_found": (
        "not found", "did not find", "never consulted", "stored-but-unreachable",
        "value was archived", "failed read",
    ),
    "duplicate_identifier": (
        "must be unique", "duplicate", "already exists", "re-write", "redundant write",
        "collision",
    ),
    "no_tool_call_at_all": (
        "did not look", "without looking", "no tool call", "never called", "did not consult",
        "answer without", "retrievable and the model did not",
    ),
    "container_slots_exhausted": (
        "slot cap", "no remaining slots", "reached its slot", "nowhere to put",
        "relocation target", "saturat",
    ),
    "append_would_exceed_cap": (
        "would exceed", "append would", "too long after appending", "single string",
        "single blob", "overflow", "10,000-character", "character cap",
    ),
    "clear_proposed_at_capacity": (
        "proposes to clear", "destructive clear", "proposes a destructive", "clear proposed",
        "wholesale clear", "proposes a clear",
    ),
    "retrieval_similarity_below_threshold": (
        "similarity", "nothing well-matched", "below threshold", "weak match", "low score",
        "best match", "never searched",
    ),
    "no_informative_result": (
        "returns nothing", "returned nothing", "empty result", "no information",
        "vacuous", "carries no information", "successful call containing no",
    ),
}


# Measured parameter values from the accepted stack (A9's threshold is vector-only).
SIGNAL_PROBE_PARAMS: Mapping[str, Mapping[str, Any]] = {
    "retrieval_similarity_below_threshold": {"below": 0.30},
}


# The repair each capacity/retrieval anchor actually performs.
REROUTE_DESTINATIONS: Mapping[str, Mapping[str, Any]] = {
    # A1: the blocked core write is re-addressed to archival, which has room.
    "container_at_capacity": {
        "destination": "archival_memory_add", "retry_original": False,
        "attested": "A1 -- rank 1 by linked downstream loss, 81 of 215 residual failures",
    },
    # A5: free one provably-redundant slot in the saturated target, then retry the ORIGINAL call.
    "container_slots_exhausted": {
        "destination": "archival_memory_remove+retry", "retry_original": True,
        "attested": "A5 -- 30/30 evictions, blocked write landed 30/30, copies_remaining >= 1",
    },
    # A7: no second container exists, so the store itself is rewritten shorter, then the append retried.
    "append_would_exceed_cap": {
        "destination": "memory_rewrite+retry_append", "retry_original": True,
        "attested": "A7 v3 -- fits AND shorter than the original; +7.34 pp on the target backend",
    },
    # A9: the same query is dispatched to archival and the union re-ranked globally.
    "retrieval_similarity_below_threshold": {
        "destination": "archival_memory_retrieve+merge_rank", "retry_original": False,
        "attested": "A9 -- 24/24 case-level, archival promoted into top-k",
    },
}


# The Phi_BFCL signal each accepted anchor's trigger corresponds to. Pure answer key: it exists so
# the recovery verdict is checked against a DECLARED expectation rather than one chosen after
# seeing the output.
EXPECTED_SIGNAL: Mapping[str, str] = {
    "A1": "container_at_capacity",
    "A2": "identifier_not_found",
    "A3": "duplicate_identifier",
    "A4": "no_tool_call_at_all",
    "A5": "container_slots_exhausted",
    "A7": "append_would_exceed_cap",
    "A8": "clear_proposed_at_capacity",
    "A9": "retrieval_similarity_below_threshold",
}

# Declared sibling signals: pairs observing the SAME condition at different specificity. Recorded
# explicitly so an "approximate" verdict is auditable rather than a loose string comparison.
SIBLINGS = (
    frozenset({"container_at_capacity", "container_slots_exhausted"}),
    frozenset({"identifier_not_found", "no_informative_result"}),
)
