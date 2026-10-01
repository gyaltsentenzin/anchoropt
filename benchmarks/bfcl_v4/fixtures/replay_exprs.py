"""The eight historical triggers, re-expressed in the DECLARATIVE signal language.

WHAT THIS DEMONSTRATES, AND WHY IT IS A STRONGER TEST THAN THE NAMED SIGNALS
---------------------------------------------------------------------------
`bfcl_signals.py` declares nine hand-written Python predicates, one per accepted anchor. That proves
only that the anchors can be typed in. These expression trees prove something the autonomous loop
actually depends on: that the SAME controllers are constructible from the runtime's declared
alphabet using the closed operator set -- i.e. that a proposer restricted to
`bfcl_capabilities.all_fields()` could have reached them without any of them being pre-registered.

So the historical vocabulary becomes EXPRESSIBLE rather than PRIVILEGED.

HISTORY WITHOUT A CROSS-BOUNDARY DSL
------------------------------------
A3, A5 and A8 all need facts from earlier steps. None of them reference another boundary: they read
CARRIED SUMMARY fields the wrapper maintains (`last_error_kind`, `pending_blocked_write`,
`container_occupancy`), each an ordinary field readable at one decision point. The anchor therefore
still fires at exactly one explicit locus, and the compiler's boundary intersection confirms it.

These remain FIXTURES. A discovery run must not import them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Each entry: the anchor it reproduces, the expression, and the params it needs.
REPLAY_EXPRS: Mapping[str, Mapping[str, Any]] = {
    # A1 -- a write refused because its container has no room. Keyed on the REFUSAL, not the verb.
    "A1": {
        "signal": "write_refused_no_capacity",
        "expr": {"all": [{"field": "proposes_write", "op": "is_true"},
                         {"field": "error_kind", "op": "eq", "value": "no_capacity"}]},
        "params": {},
    },
    # A2 -- a read that did not find what it asked for. The read-side test is PART of the signal:
    # one error string covered 39 read-side and 5 write-side cases, and merging them merges two
    # different failures into one remedy.
    "A2": {
        "signal": "read_found_nothing",
        "expr": {"all": [{"field": "proposes_read", "op": "is_true"},
                         {"field": "error_kind", "op": "eq", "value": "not_found"}]},
        "params": {},
    },
    # A3 -- a proposed write whose key is already present. Fires at the COMMITMENT GATE, which is
    # the only point where cancelling is free; `last_error_kind` carries the duplicate refusal the
    # model is about to repeat.
    "A3": {
        "signal": "duplicate_write_proposed",
        "expr": {"all": [{"field": "proposes_write", "op": "is_true"},
                         {"field": "last_error_kind", "op": "eq", "value": "duplicate_identifier"}]},
        "params": {},
    },
    # A4 -- about to answer without ever having called a tool. Observable only after generation.
    "A4": {
        "signal": "answering_without_looking",
        "expr": {"all": [{"field": "has_generation", "op": "is_true"},
                         {"field": "proposes_tool_call", "op": "is_false"},
                         {"field": "tool_calls_so_far", "op": "eq", "value": 0}]},
        "params": {},
    },
    # A5 -- the relocation TARGET is out of slots, so A1's own repair has nowhere to put the payload.
    "A5": {
        "signal": "relocation_target_saturated",
        "expr": {"all": [{"field": "error_kind", "op": "eq", "value": "no_capacity"},
                         {"field": "container", "op": "eq", "value": "archival"},
                         {"field": "container_occupancy", "op": "gte", "param": "occupied_at"}]},
        "params": {"occupied_at": 1.0},
    },
    # A7 -- a single-blob store refused an append that would overflow. Nowhere to relocate to.
    "A7": {
        "signal": "append_overflowed_blob",
        "expr": {"field": "error_kind", "op": "eq", "value": "blob_would_overflow"},
        "params": {},
    },
    # A8 -- a destructive clear PROPOSED while a refused write is still pending and the container is
    # full. Three facts, all readable at the commitment gate; two of them carried.
    "A8": {
        "signal": "destructive_clear_at_capacity",
        "expr": {"all": [{"field": "proposes_clear", "op": "is_true"},
                         {"field": "pending_blocked_write", "op": "is_true"},
                         {"field": "container_occupancy", "op": "gte", "param": "occupied_at"}]},
        "params": {"occupied_at": 1.0},
    },
    # A9 -- a retrieve that SUCCEEDED and returned nothing well-matched. No error is raised at all,
    # which is why the lexical tier cannot reach it: a silent read failure looks like a success.
    "A9": {
        "signal": "retrieval_weak_or_vacuous",
        "expr": {"all": [{"field": "proposes_read", "op": "is_true"},
                         {"field": "error_kind", "op": "is_none"},
                         {"any": [{"field": "result", "op": "is_vacuous"},
                                  {"field": "best_similarity", "op": "lt", "param": "below"}]}]},
        "params": {"below": 0.30},
    },
}

# The boundary each replayed signal MUST compile to, from docs/ANCHORS.md. Asserted by test: if the
# compiler's field-intersection disagrees with the accepted anchor's decision point, either the
# alphabet's declarations or the anchor's record is wrong, and both are worth knowing.
EXPECTED_BOUNDARY: Mapping[str, str] = {
    "A1": "post_execution",
    "A2": "post_execution",
    "A3": "post_generation_pre_exec",
    "A4": "post_generation_pre_exec",
    "A5": "post_execution",
    "A7": "post_execution",
    "A8": "post_generation_pre_exec",
    "A9": "post_execution",
}
