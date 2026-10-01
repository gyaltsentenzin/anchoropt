"""Phi-CONSTRUCTION ALPHABET for BFCL v4: the observable fields and tools a signal may name.

This is the module the PROPOSER sees. It answers three questions and deliberately no others:

    observable_fields()   which primitive facts exist, of what type, at which decision points
    carried_fields()      which summaries of HISTORY the runtime maintains
    tool_schema()         which destinations a reroute can name, with real argument names

WHY THIS REPLACES A LIST OF FINISHED SIGNALS
-------------------------------------------
The first version of this integration declared nine signals, one per accepted anchor. That makes the
anchor library the design: a proposer restricted to those names can only ever re-select what is
already known. Here the runtime declares an ALPHABET instead, and a signal is a declarative
expression over it (`anchoropt/learning/signal_lang.py`). The nine historical triggers are then
EXPRESSIBLE rather than PRIVILEGED -- and `fixtures/replay_anchors.py` keeps them only as
calibration.

HISTORY IS RUNTIME STATE, NOT A CROSS-BOUNDARY EXPRESSION
--------------------------------------------------------
the destructive-clear trigger genuinely spans time: a write was refused earlier, and the model NOW proposes a clear.
The tempting generalization -- letting an expression reference another boundary -- is wrong, because
a signal that straddles two decision points is not observable at either, and the anchor would then
claim a locus it does not have.

So history enters as CARRIED SUMMARIES the wrapper maintains (`carried_fields()`): `last_error_kind`,
`pending_blocked_write`, `container_occupancy`. Each is an ordinary field readable AT one boundary,
so the signal stays single-locus and `U_H(l)` keeps meaning what it says. The wrapper owns the
summarizing; the expression language never learns what a "previous step" is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Boundary names, as declared by the core's IncisionPoint values. Strings here so this module can be
# read without importing the core.
PRE_GEN = "pre_generation"
POST_GEN = "post_generation_pre_exec"
POST_EXEC = "post_execution"

ALL_BOUNDARIES = (PRE_GEN, POST_GEN, POST_EXEC)


@dataclass(frozen=True)
class Field:
    """One primitive observable. `boundaries` is where the fact EXISTS, not where it is useful.

    Getting that distinction wrong is a measured defect class: a condition evaluated where its
    facts do not yet exist fires on every episode and pays the cost on the ones that did not need
    it, and the same text one boundary later can be positive instead. Declaring existence
    per field -- and INTERSECTING those declarations when an expression combines fields -- makes
    that class of error structurally impossible instead of something a human must notice.
    """

    name: str
    type: type
    boundaries: tuple[str, ...]
    doc: str
    enum: tuple[str, ...] = ()


# ------------------------------------------------------------------------------------------------
# The primitive alphabet
# ------------------------------------------------------------------------------------------------
_FIELDS = (
    # ---- generation / proposal facts: exist once the model has produced a decision -------------
    Field("has_generation", bool, (POST_GEN, POST_EXEC),
          "the model has produced a decision for this step"),
    Field("proposes_tool_call", bool, (POST_GEN, POST_EXEC),
          "the decision is a tool call rather than a final answer"),
    Field("n_tool_calls", int, (POST_GEN, POST_EXEC),
          "how many calls this decision proposes"),
    Field("proposes_write", bool, (POST_GEN, POST_EXEC),
          "the proposed call puts something into the store"),
    Field("proposes_read", bool, (POST_GEN, POST_EXEC),
          "the proposed call gets something out of the store"),
    Field("proposes_clear", bool, (POST_GEN, POST_EXEC),
          "the proposed call is a wholesale destructive clear"),
    Field("proposes_remove", bool, (POST_GEN, POST_EXEC),
          "the proposed call is a targeted single-entry removal"),
    Field("container", str, (POST_GEN, POST_EXEC),
          "which container the proposed call addresses, or None for a single-blob backend",
          enum=("core", "archival")),
    Field("target_id", str, (POST_GEN, POST_EXEC),
          "the entry handle the call addresses, or None"),

    # ---- result facts: exist only after execution -----------------------------------------------
    Field("result", str, (POST_EXEC,),
          "the returned payload; readable by is_vacuous / matches_kind, never string-compared"),
    Field("error_kind", str, (POST_EXEC,),
          "coarse refusal class, or None when the result carries no error marker",
          enum=("duplicate_identifier", "not_found", "blob_would_overflow", "entry_too_long",
                "no_capacity", "other")),
    Field("best_similarity", float, (POST_EXEC,),
          "best per-entry similarity score in a retrieval result; None where the backend "
          "exposes no scores"),

    # ---- step context: exists before anything is generated ---------------------------------------
    Field("step_index", int, ALL_BOUNDARIES, "0-based index of this step in the episode"),
    Field("tool_calls_so_far", int, ALL_BOUNDARIES,
          "how many tool calls this episode has made before this step"),
)

# ------------------------------------------------------------------------------------------------
# Carried summaries -- HISTORY, maintained by the wrapper
# ------------------------------------------------------------------------------------------------
#
# Each is a summary of earlier steps, readable at ONE boundary like any other field. This is what
# lets the destructive-clear trigger ("a write was refused, and a clear is now proposed") be a single-locus signal:
# `last_error_kind` and `pending_blocked_write` are carried facts, and `proposes_clear` is a
# present-tense one, all readable at POST_GENERATION_PRE_EXEC.
_CARRIED = (
    Field("last_error_kind", str, ALL_BOUNDARIES,
          "refusal class of the most recent failed call in this episode, or None",
          enum=("duplicate_identifier", "not_found", "blob_would_overflow", "entry_too_long",
                "no_capacity", "other")),
    Field("pending_blocked_write", bool, ALL_BOUNDARIES,
          "a write was refused earlier in this episode and has not since succeeded"),
    Field("container_occupancy", float, ALL_BOUNDARIES,
          "occupied fraction of the addressed container's capacity, from LIVE state; None when "
          "unknown. Live state, never a write log -- an earlier version read its own log, saw a "
          "median of 4 entries against a real ~50, and the arm was void"),
    Field("consecutive_read_failures", int, ALL_BOUNDARIES,
          "reads that returned nothing useful in a row, counting vacuous results as failures"),
)


def observable_fields() -> Mapping[str, Field]:
    """The primitive fields a proposed signal may name. Nothing else is nameable."""
    return {f.name: f for f in _FIELDS}


def carried_fields() -> Mapping[str, Field]:
    """History summaries the wrapper maintains. Ordinary fields, single-locus by construction."""
    return {f.name: f for f in _CARRIED}


def all_fields() -> Mapping[str, Field]:
    return {**observable_fields(), **carried_fields()}


def fields_at(boundary: str) -> tuple[str, ...]:
    """Field names readable at `boundary`, primitives and carried summaries together."""
    return tuple(sorted(n for n, f in all_fields().items() if boundary in f.boundaries))


# ------------------------------------------------------------------------------------------------
# Tool schema -- grounded reroute destinations
# ------------------------------------------------------------------------------------------------
#
# docs/GENERALIZABILITY.md: "reroute must name a real destination tool with real argument names",
# and a destination must be attested on RESOLVING rather than on not-erroring -- a destination once
# looked viable at 11/11 "clean" where clean meant did-not-error, and 9 of those 11 returned nothing.
#
# So this table declares what EXISTS and how to call it. It does NOT declare what works: attestation
# is a MEASURED property the loop records per (locus, destination), never a literal typed here. That
# separation is the fix for the first version's `REROUTE_DESTINATIONS`, which hardcoded the four
# repairs the accepted anchors happen to perform and thereby smuggled the answer key into the
# runtime.
_TOOLS: Mapping[str, Mapping[str, Any]] = {
    "core_memory_add":            {"args": ("key", "value"), "container": "core", "kind": "write"},
    "core_memory_remove":         {"args": ("key",), "container": "core", "kind": "remove"},
    "core_memory_retrieve":       {"args": ("query", "top_k"), "container": "core", "kind": "read"},
    "core_memory_list_keys":      {"args": (), "container": "core", "kind": "read"},
    "core_memory_clear":          {"args": (), "container": "core", "kind": "clear"},
    "archival_memory_add":        {"args": ("key", "value"), "container": "archival", "kind": "write"},
    "archival_memory_remove":     {"args": ("vec_id",), "container": "archival", "kind": "remove"},
    "archival_memory_retrieve":   {"args": ("query", "top_k"), "container": "archival", "kind": "read"},
    "archival_memory_key_search": {"args": ("query",), "container": "archival", "kind": "read"},
    "archival_memory_clear":      {"args": (), "container": "archival", "kind": "clear"},
    "memory_append":              {"args": ("content",), "container": None, "kind": "write"},
    "memory_retrieve":            {"args": ("query",), "container": None, "kind": "read"},
}


def tool_schema() -> Mapping[str, Mapping[str, Any]]:
    """Available tools with their real argument names, for grounding a reroute destination."""
    return {name: dict(spec) for name, spec in _TOOLS.items()}


def tools_of_kind(kind: str) -> tuple[str, ...]:
    return tuple(sorted(n for n, s in _TOOLS.items() if s.get("kind") == kind))


def is_known_tool(name: str) -> bool:
    return name in _TOOLS


def unknown_args(tool: str, args) -> tuple[str, ...]:
    """Argument names `tool` does not accept. A reroute naming one is unexecutable, not merely odd.

    docs/ANCHORS.md (an accepted anchor): emitting `key=` universally would have built an invalid call and been
    misscored as "the substitute did not help" -- a measurement error dressed as a negative result.
    """
    spec = _TOOLS.get(tool)
    if spec is None:
        return tuple(sorted(args or ()))
    return tuple(sorted(set(args or ()) - set(spec["args"])))
