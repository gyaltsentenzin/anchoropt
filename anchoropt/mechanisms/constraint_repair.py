#!/usr/bin/env python3
"""The core algorithm: ONE control flow for every "the store refused this write" anchor.

A1, A5, A6 and A7 look like four different mechanisms. They are four instances of one algorithm with
four different repair strategies plugged into it. This module is that algorithm; the strategies are
data.

    detect      does this result mean a constraint was violated, and WHICH constraint?
    observe     read the LIVE state the repair depends on
    propose     compute an admissible repair, or decline
    verify      check the repair's own invariant BEFORE committing to it
    retry       replay the ORIGINAL call, verbatim

Writing it once matters because every step has a documented way to get it wrong, and each wrong way
cost a real experiment on this project. Factoring the flow out means a new anchor inherits the fixes
instead of rediscovering them:

    detect    keying on the ACTION NAME instead of the constraint proposed a suppression for a call
              that was benign in 15 of 17 cases. Detection is a property of the REFUSAL, not the verb.
              Backends also phrase the same constraint differently -- one anchor's trigger regex matched
              vector's wording only and never fired on kv at all.
    observe   reading an episode-local write log instead of the store saw a median of 4 entries against
              a real ~50 and voided an entire arm. And "unreadable" must be distinguishable from
              "readable and empty", or the guard fails open exactly when absence is most certain.
    propose   a repair that cannot be shown admissible must DECLINE, not guess. Declining is a
              legitimate and frequent verdict.
    verify    reasoning that an invariant holds is not observing that it holds. Check it against state,
              at the point of action.
    retry     NEVER synthesize a replacement call. A synthesized retry failed on every firing of one
              backend AFTER its eviction had already succeeded -- destroying a duplicate and losing the
              write, strictly worse than doing nothing. The recorded call is already correct for its
              own backend.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not execute anything. `plan_repair` reads state and returns a decision; the caller dispatches.
That is deliberate and it is the fix for the most expensive bug class here: two separate anchors placed
their guard AFTER execution, so the call had already mutated the store and "prevention" was theatre.
A pure planner cannot make that mistake -- there is nothing to execute inside it.

Ordering is likewise the caller's to enforce, and `RepairPlan.dispatch_sequence` states it as data so a
port can assert the order rather than reimplement it.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


class Constraint(enum.Enum):
    """What the store refused, independent of how any backend words it.

    The anchor is keyed on this, never on the error string or the tool name. Two backends state the
    same constraint in different words, and the same words can mean different constraints -- so the
    canonical constraint is the thing worth naming.
    """

    NO_SLOTS = "capacity/container/no_remaining_slots"
    ENTRY_TOO_LONG = "size/item/exceeds_per_item_limit"
    BLOB_WOULD_OVERFLOW = "size/blob/append_would_exceed_cap"
    DUPLICATE_IDENTIFIER = "permission/identifier/duplicate"
    NOT_FOUND = "existence/identifier/not_found"
    # Not an ERROR at all: a proposed DESTRUCTIVE action on a container at capacity. Detected from
    # the proposal rather than from a refusal, which is why `detect_constraint` cannot see it -- see
    # `detect_proposed_destruction`.
    CLEAR_PROPOSED_AT_CAPACITY = "capacity/container/clear_proposed_at_capacity"


class Repair(enum.Enum):
    """The repair strategies, and what each one requires to be admissible.

        RELOCATE   the payload is fine, the destination is not -> re-address it
                   requires: another container whose limits ACCEPT the payload verbatim
        REDUCE     the destination is fine, the payload is not -> shrink it
                   requires: a shortening that preserves the facts (computed, or model-generated
                             under a relative constraint -- see blob_compaction)
        EVICT      the destination is full -> free a slot
                   requires: PROOF the freed slot held redundant information
        DECLINE    no admissible repair exists here
                   requires: nothing. This is a real answer, not a failure to find one.

    EVICT covers two triggers that look alike and are not: a write REFUSED for lack of slots (A5,
    post-execution) and a destructive clear PROPOSED on a full container (A8, pre-execution). Same
    action, same invariant, different decision points -- which is why the (point, action) grid
    under-determines an anchor and the CONSTRAINT is what separates them.
    """

    RELOCATE = "relocate"
    REDUCE = "reduce"
    EVICT = "evict"
    DECLINE = "decline"


# ------------------------------------------------------------------------------------------------
# THE PORTING SEAM
# ------------------------------------------------------------------------------------------------
#
# Everything benchmark-specific lives in a StoreAdapter. To port this to another benchmark you write
# ONE adapter and register it -- you should not need to edit anything else in this module.
#
#     MY_STORE = StoreAdapter(
#         name="my_store",
#         error_patterns={Constraint.NO_SLOTS: r"capacity exceeded"},
#         entry_caps={"main": 512, "overflow": 4096},
#         relocation_target="overflow",
#         write_verbs=("store_add",),
#         id_kwarg="item_id",
#     )
#     register_adapter(MY_STORE)
#
# Then `plan_repair(call, result, state, backend="my_store")` works. The four things an adapter
# supplies are exactly the four that are NOT portable, and each one is a place a port has silently
# broken before:
#
#   error_patterns      how THIS store words each constraint. Two backends of the same benchmark
#                       phrase "container full" differently, and an anchor keyed on one string
#                       simply never fires on the other -- measured, not hypothetical.
#   entry_caps          per-container limits, which is what makes RELOCATE admissible at all. If no
#                       container has a bigger cap, relocation cannot help and the strategy declines.
#   write_verbs         which calls are writes. Needed to find the payload, never to decide the
#                       repair -- keying a remedy on the ACTION NAME instead of the refusal proposed
#                       a suppression that inspection showed was benign in 15 of 17 cases.
#   read_state          how to read live state off the runtime object. Container shapes differ even
#                       within one benchmark (a dict on the instance vs a dict under `._store` vs a
#                       bare string), and reading it wrong logged zero events for a whole line.
#
# `BFCL_V4_ADAPTERS` below is a worked example, not a base class to inherit. Abstracting an interface
# from one benchmark encodes that benchmark's accidents; the adapter is deliberately a plain record.


@dataclass(frozen=True)
class StoreAdapter:
    """Everything benchmark-specific about one memory backend. The only thing a port must write.

    `read_state` is optional: the default walk handles the two common shapes (a plain dict on the
    instance, and a dict under `._store`). Supply your own for anything else -- notably a store whose
    state is a bare string rather than a container, which the default cannot see.
    """

    name: str
    error_patterns: Mapping[Constraint, str]
    entry_caps: Mapping[str, int] = field(default_factory=dict)
    # Where an over-long entry may be MOVED TO. `None` means this store has no second container, so
    # relocation is structurally impossible and only a REDUCE can admit the write. That is not a
    # missing value -- it is the fact that makes a blob-compaction anchor necessary, so it must be
    # stated rather than inferred from the caps table (a single-container store still has a cap).
    relocation_target: str | None = None
    single_container: bool = False
    # Per-container SLOT counts, distinct from `entry_caps` (which is per-entry LENGTH). Needed to
    # answer "is this container at capacity", which is what makes a proposed clear interceptable.
    slot_caps: Mapping[str, int] = field(default_factory=dict)
    write_verbs: tuple[str, ...] = ()
    id_kwarg: str = "id"
    read_state: Callable[[Any, str], StateView] | None = None
    notes: str = ""

    def compiled(self) -> Mapping[Constraint, re.Pattern[str]]:
        return {c: re.compile(p, re.IGNORECASE) for c, p in self.error_patterns.items()}

    def cap_of(self, container: str) -> int | None:
        return self.entry_caps.get(container)

    def slots_of(self, container: str) -> int | None:
        """How many entries `container` holds at capacity, or None if not declared."""
        return self.slot_caps.get(container)

    def relocation_destination(self) -> str | None:
        """Where to move an over-long entry, or None if this store has nowhere to move it.

        Explicit `relocation_target` wins. Otherwise the largest-capped container is used -- but a
        `single_container` store never has one, however many caps it declares.
        """
        if self.single_container:
            return None
        if self.relocation_target:
            return self.relocation_target
        others = {k: v for k, v in self.entry_caps.items() if v is not None}
        return max(others, key=lambda k: others[k]) if len(others) > 1 else None


# Attribute names a store instance may expose its entries under. Declared here, beside the other
# store facts, so a mechanism that needs to read live state does not carry its own literal tuple --
# `redundant_write.store_lookup` used to. First readable match wins.
STATE_ATTRS: tuple[str, ...] = ("core_memory", "archival_memory", "memory")

_ADAPTERS: dict[str, StoreAdapter] = {}


def register_adapter(adapter: StoreAdapter) -> StoreAdapter:
    """Make `adapter` available to `plan_repair(..., backend=adapter.name)`."""
    _ADAPTERS[adapter.name] = adapter
    return adapter


def adapter_for(backend: str) -> StoreAdapter | None:
    return _ADAPTERS.get(backend)


def registered_backends() -> frozenset[str]:
    return frozenset(_ADAPTERS)


# ------------------------------------------------------------------------------------------------
# Worked example: the three BFCL v4 Agent Memory backends.
# ------------------------------------------------------------------------------------------------
#
# Read these as "what an adapter looks like when the numbers come from a real benchmark", not as
# defaults to keep. Every string and cap here was observed from the shipped backends.
BFCL_V4_ADAPTERS = (
    register_adapter(StoreAdapter(
        name="vector",
        error_patterns={
            Constraint.NO_SLOTS: r"memory size exceeds maximum size",
            Constraint.ENTRY_TOO_LONG: r"entry length exceeds maximum length|entry is too long",
        },
        entry_caps={"core": 300, "archival": 2000},
        slot_caps={"core": 7, "archival": 50},
        relocation_target="archival",
        write_verbs=("core_memory_add", "archival_memory_add"),
        id_kwarg="vec_id",
        notes="entries are anonymous; archival state lives under `.archival_memory._store`",
    )),
    register_adapter(StoreAdapter(
        name="kv",
        error_patterns={
            # NOTE the phrasing difference from vector for the SAME constraint. This is the trap.
            Constraint.NO_SLOTS: r"long term memory is full",
            Constraint.ENTRY_TOO_LONG: r"entry length exceeds maximum length|entry is too long",
            Constraint.DUPLICATE_IDENTIFIER: r"key name must be unique",
        },
        entry_caps={"core": 300, "archival": 2000},
        slot_caps={"core": 7, "archival": 50},
        relocation_target="archival",
        write_verbs=("core_memory_add", "archival_memory_add"),
        id_kwarg="key",
        notes=(
            "entries are ADDRESSED by key and a taken key is REFUSED rather than overwritten, so a "
            "remedy here must preserve addressability -- do not port vector's anonymous-eviction "
            "logic to kv unchanged"
        ),
    )),
    register_adapter(StoreAdapter(
        name="rec_sum",
        error_patterns={Constraint.BLOB_WOULD_OVERFLOW: r"too long after appending"},
        entry_caps={"blob": 10_000},
        relocation_target=None,       # there IS no second container -- hence A7 exists
        single_container=True,
        write_verbs=("memory_append", "memory_update", "memory_replace"),
        notes=(
            "state is a BARE STRING on the instance, not a container of entries. The default state "
            "walk cannot see it, which is why this backend logged 0 events for a whole line."
        ),
    )),
)


def detect_constraint(result: Any, backend: str) -> Constraint | None:
    """Which constraint this result reports, or None if it is not a constraint violation.

    Keyed on the refusal, scoped to the backend, resolved through the registered adapter. A result
    that is not an error is never a violation, and a backend with no adapter returns None rather than
    guessing -- an unregistered backend should read as "unknown", not as "fine".
    """
    text = str(result or "")
    if "error" not in text.lower():
        return None
    adapter = adapter_for(backend)
    if adapter is None:
        return None
    for constraint, pattern in adapter.compiled().items():
        if pattern.search(text):
            return constraint
    return None


def detect_proposed_destruction(call: Any, live_size: int | None, capacity: int | None) -> Constraint | None:
    """Is this a DESTRUCTIVE proposal on a container at capacity?

    A separate entry point from `detect_constraint` because the signal is structurally different: it
    comes from a PROPOSED CALL plus live state, not from an error string. Nothing has failed yet, and
    that is exactly the point -- after a clear executes there is nothing left to save.

    Returns None when the container is not at capacity. A clear on a container with room is probably
    the user asking to forget something, which is not an anchor's business.
    """
    from anchoropt.mechanisms.dedup_clear_recovery import container_of

    if container_of(call) is None:
        return None
    if live_size is None or capacity is None:
        return None
    if live_size < capacity:
        return None
    return Constraint.CLEAR_PROPOSED_AT_CAPACITY


def backends_for(constraint: Constraint) -> frozenset[str]:
    """Which registered backends can even produce this constraint. The anchor's exposure set.

    Use this to build the exposure split -- an anchor cannot be credited with movement on a backend
    whose error contract it can never match. Derive exposure from THIS rather than declaring it: one
    anchor's backend list was asserted for three backends and fires on exactly one.
    """
    return frozenset(
        name for name, adapter in _ADAPTERS.items() if constraint in adapter.error_patterns
    )


@dataclass(frozen=True)
class StateView:
    """What was actually observed at the decision point, and whether it could be observed at all.

    `readable=False` is NOT the same as an empty container, and conflating them is a real bug this
    field exists to prevent: a guard tested with `if not state` fails OPEN exactly when absence is
    most certain. Every strategy must check `readable` before trusting `entries`.
    """

    readable: bool
    entries: Mapping[Any, str] = field(default_factory=dict)
    blob: str | None = None
    container: str | None = None
    # The write that is currently blocked, if any: `{"call": ..., "container": ...}`. Required by the
    # proposed-destruction strategy, because freeing a slot with nothing to retry accomplishes
    # nothing -- and a clear on a container nobody is waiting to write to may be a legitimate
    # request to forget.
    pending_write: Mapping[str, Any] | None = None

    @property
    def count(self) -> int:
        return len(self.entries)

    @property
    def distinct(self) -> int:
        return len(set(self.entries.values()))

    def copies_of(self, value: str) -> int:
        return sum(1 for v in self.entries.values() if v == value)


@dataclass(frozen=True)
class RepairPlan:
    """The decision. Data only -- nothing here has executed, and nothing here executes."""

    fire: bool
    constraint: Constraint | None
    repair: Repair
    reason: str
    calls: tuple[str, ...] = ()          # what to dispatch, in order
    retry_call: str | None = None        # the ORIGINAL call, verbatim
    invariant: str | None = None         # what verify() must confirm before the retry
    telemetry: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dispatch_sequence(self) -> tuple[str, ...]:
        """Repair calls first, then the verbatim retry. Stated as data so a port can assert it."""
        if not self.fire:
            return ()
        return (*self.calls, self.retry_call) if self.retry_call else self.calls

    def declined(self, reason: str) -> RepairPlan:
        return RepairPlan(False, self.constraint, Repair.DECLINE, reason, telemetry=self.telemetry)


# A strategy takes the refused call, the observed state and the backend, and returns a plan.
# Registering one is how a new anchor joins this flow.
Strategy = Callable[[str, StateView, str], RepairPlan]

_STRATEGIES: dict[Constraint, Strategy] = {}


def register(constraint: Constraint) -> Callable[[Strategy], Strategy]:
    """Register the repair strategy for a constraint.

    One strategy per constraint, deliberately. Two anchors on the same constraint that differ on the
    repair are two candidates to MEASURE against each other, not two things to run at once -- and
    letting both fire is how anchors silently pre-empt each other (see `displacement_check`).
    """

    def wrap(fn: Strategy) -> Strategy:
        _STRATEGIES[constraint] = fn
        return fn

    return wrap


def strategy_for(constraint: Constraint) -> Strategy | None:
    return _STRATEGIES.get(constraint)


def plan_repair(
    call: Any,
    result: Any,
    state: StateView,
    backend: str,
    budget_used: int = 0,
    budget: int = 3,
) -> RepairPlan:
    """The core algorithm. Detect, then delegate to the registered strategy, then bound it.

    Returns a plan; executes nothing. The caller dispatches `plan.dispatch_sequence` and calls
    `verify_invariant` between the repair and the retry where the plan names an invariant.

    TWO KINDS OF SIGNAL, and the second is easy to miss. Most anchors key on a REFUSAL, so `result`
    carries the evidence. One keys on a PROPOSAL plus live state -- a destructive call about to run on
    a container at capacity -- where nothing has failed yet and `result` is empty by construction.
    Detection tries the refusal first and falls back to the proposal, so a caller passing
    `result=None` for a proposed call gets the right answer rather than "no violation".

    `budget` is not optional in spirit: unbounded repair is the failure mode where a mechanism frees
    one slot per pending write and empties the container while every individual step passes its own
    losslessness test.
    """
    constraint = detect_constraint(result, backend)
    if constraint is None:
        # No refusal. Is this a destructive PROPOSAL on a full container?
        adapter = adapter_for(backend)
        container = state.container or "archival"
        constraint = detect_proposed_destruction(
            call,
            live_size=state.count if state.readable else None,
            capacity=adapter.slots_of(container) if adapter else None,
        )
    if constraint is None:
        return RepairPlan(False, None, Repair.DECLINE, "not a constraint violation")

    if budget_used >= budget:
        return RepairPlan(
            False, constraint, Repair.DECLINE,
            f"episode repair budget spent ({budget})",
            telemetry={"budget": budget, "budget_used": budget_used},
        )

    if not state.readable:
        # Cannot observe the state the repair depends on => cannot show the repair is admissible.
        # Decline. Never fail open here: this is precisely the case where a wrong guess destroys data.
        return RepairPlan(
            False, constraint, Repair.DECLINE,
            "state unreadable at the decision point, so no repair can be shown admissible",
        )

    strategy = strategy_for(constraint)
    if strategy is None:
        return RepairPlan(
            False, constraint, Repair.DECLINE,
            f"no repair strategy registered for {constraint.value}",
        )

    plan = strategy(str(call), state, backend)
    if plan.fire and plan.retry_call is None:
        # Every repair here exists to admit a write that was refused. A plan that repairs the state
        # and never retries has fixed the symptom and dropped the payload.
        return plan.declined("strategy produced no verbatim retry, so the refused write would be lost")
    return plan


def verify_invariant(plan: RepairPlan, state_after: StateView) -> dict[str, Any]:
    """Confirm the plan's own invariant against state OBSERVED AFTER the repair, before the retry.

    Only `EVICT` carries an information-preservation invariant, and it is the one repair that can
    destroy data, so it is the one that must be checked against the store rather than argued.
    """
    if not plan.fire or plan.repair is not Repair.EVICT:
        return {"ok": True, "checked": False, "reason": "no invariant to verify"}
    value = str(plan.telemetry.get("victim_value", ""))
    if not state_after.readable:
        return {"ok": False, "checked": True, "reason": "state unreadable after the repair"}
    remaining = state_after.copies_of(value)
    return {
        "ok": remaining >= 1,
        "checked": True,
        "remaining": remaining,
        "reason": "a copy remains" if remaining >= 1 else "INVARIANT VIOLATED: no copy remains",
    }


# ------------------------------------------------------------------------------------------------
# The registered strategies. Each is a few lines, because the flow above carries the discipline.
# ------------------------------------------------------------------------------------------------

_VALUE_RE = re.compile(r"(?:text|value|content)\s*=\s*(['\"])(.*?)\1", re.DOTALL)
_POSITIONAL_RE = re.compile(r"\(\s*(['\"])(.*)\1\s*\)\s*$", re.DOTALL)


def payload_of(call: str) -> str | None:
    """The verbatim payload of a write call, or None when it cannot be parsed.

    None must mean "pass the call through untouched". A parser that guesses is a parser that
    silently drops writes it did not understand.
    """
    for pattern in (_VALUE_RE, _POSITIONAL_RE):
        match = pattern.search(str(call or ""))
        if match:
            return match.group(2)
    return None


@register(Constraint.NO_SLOTS)
def _evict_a_provable_duplicate(call: str, state: StateView, backend: str) -> RepairPlan:
    """A5. The container is full; free a slot ONLY where doing so is provably lossless.

    Victim: the last-written copy of a value with >= 2 copies. Deterministic, so no model decides
    what to destroy, and the earliest copy -- likelier to have been referenced -- is preserved.
    """
    counts: dict[str, int] = {}
    for value in state.entries.values():
        counts[value] = counts.get(value, 0) + 1
    duplicated = {v for v, n in counts.items() if n >= 2}

    base = RepairPlan(
        False, Constraint.NO_SLOTS, Repair.EVICT, "",
        telemetry={"live_entries": state.count, "live_distinct": state.distinct},
    )
    if not duplicated:
        return base.declined("no duplicate exists, so no eviction is lossless")

    victim_id, victim_value = None, None
    for entry_id, value in state.entries.items():
        if value in duplicated:
            victim_id, victim_value = entry_id, value
    total = counts[str(victim_value)]

    # The removal call is built from the ADAPTER's id keyword, not a hardcoded verb: kv addresses
    # entries by key and vector by numeric id, and synthesizing the wrong signature fails AFTER the
    # eviction has already succeeded -- destroying a duplicate and losing the write.
    adapter = adapter_for(backend)
    id_kwarg = adapter.id_kwarg if adapter else "id"
    container = state.container or "archival"
    remove_call = f"{container}_memory_remove({id_kwarg}={victim_id!r})"

    return RepairPlan(
        fire=True,
        constraint=Constraint.NO_SLOTS,
        repair=Repair.EVICT,
        reason=f"{total} copies of the victim value exist; removing one destroys no information",
        calls=(remove_call,),
        retry_call=call,                      # VERBATIM
        invariant="at least one copy of the victim value remains",
        telemetry={
            "victim_id": victim_id,
            "victim_value": victim_value,
            "copies_before": total,
            "live_entries": state.count,
            "live_distinct": state.distinct,
        },
    )


@register(Constraint.ENTRY_TOO_LONG)
def _relocate_to_a_larger_container(call: str, state: StateView, backend: str) -> RepairPlan:
    """A6. One entry is too long; re-address it to a container whose per-entry cap accepts it.

    Declines when the payload would not fit the destination either -- that needs a REDUCE, which has
    a much weaker guarantee, and silently switching repair kind would smuggle a lossy operation in
    behind a lossless one's evidence.
    """
    base = RepairPlan(False, Constraint.ENTRY_TOO_LONG, Repair.RELOCATE, "")
    adapter = adapter_for(backend)
    if adapter is None:
        return base.declined(f"no adapter registered for backend {backend!r}")

    destination = adapter.relocation_destination()
    if destination is None:
        # No second container to relocate into. That is A7's situation, and it needs a REDUCE.
        return base.declined(
            "this store has no alternative container, so relocation is impossible -- "
            "reducing the payload is a different repair with a weaker guarantee"
        )
    cap = adapter.cap_of(destination)
    payload = payload_of(call)

    if payload is None:
        return base.declined("payload unparseable; pass the call through untouched")
    if cap is not None and len(payload) > cap:
        return base.declined(
            f"payload {len(payload)} exceeds the {destination} per-entry cap {cap}; "
            "relocation cannot help and reducing is a different repair"
        )
    return RepairPlan(
        fire=True,
        constraint=Constraint.ENTRY_TOO_LONG,
        repair=Repair.RELOCATE,
        reason=f"{destination} per-entry cap {cap} accepts {len(payload)} chars verbatim",
        calls=(),                             # the relocation IS the retry
        retry_call=f"{destination}_memory_add('{payload}')",
        telemetry={"payload_len": len(payload), "destination": destination},
    )


@register(Constraint.BLOB_WOULD_OVERFLOW)
def _reduce_the_blob(call: str, state: StateView, backend: str) -> RepairPlan:
    """A7. The store is one string at its cap with no second container; shorten it, then retry.

    The only strategy that needs a model call, so it returns the prompt and the blob and lets the
    caller run the generation -- keeping this module free of an inference dependency. The acceptance
    rule the caller must then apply is `blob_compaction.validate`: FIT and SHORTER THAN THE ORIGINAL.
    """
    from anchoropt.mechanisms.blob_compaction import COMPACTION_PROMPT, HARD_CAP, overflow_state_key

    base = RepairPlan(False, Constraint.BLOB_WOULD_OVERFLOW, Repair.REDUCE, "")
    adapter = adapter_for(backend)
    cap = (adapter.cap_of("blob") if adapter else None) or HARD_CAP
    if not isinstance(state.blob, str):
        # This backend keeps its state as a bare string, not a container of entries. Walking it as a
        # dict returns nothing and logs zero events -- a backend-shape trap that has bitten here.
        return base.declined("blob unreadable: this backend's state is a string, not a container")

    payload = payload_of(call) or ""
    return RepairPlan(
        fire=True,
        constraint=Constraint.BLOB_WOULD_OVERFLOW,
        repair=Repair.REDUCE,
        reason=f"blob {len(state.blob)} + pending {len(payload)} exceeds cap {cap}",
        calls=(),                             # the caller generates, validates, then writes
        retry_call=call,                      # VERBATIM, after the rewrite lands
        invariant="the rewrite FITS and is SHORTER than the original",
        telemetry={
            "state_key": overflow_state_key(state.blob, payload),
            "prompt": COMPACTION_PROMPT,
            "blob_len": len(state.blob),
            "pending_len": len(payload),
            "requires_generation": True,
        },
    )


@register(Constraint.CLEAR_PROPOSED_AT_CAPACITY)
def _supply_capacity_instead_of_clearing(call: str, state: StateView, backend: str) -> RepairPlan:
    """A8. A destructive clear is proposed on a full container; free ONE slot losslessly instead.

    The only strategy here that fires on a PROPOSAL rather than a refusal, and the only one whose
    value depends on prevention: after the clear runs there is nothing left to save.

    It declines in every uncertain case, and that is the safety argument rather than caution. A
    predecessor refused clears unconditionally -- 85 refusals, 0 of 14 episodes closed -- because
    refusing destruction WITHOUT supplying capacity leaves the model no route out. This fires only
    when it can supply the capacity in the same action.

    `state.pending_write` is required: without a blocked write to retry, freeing a slot accomplishes
    nothing and the clear may be a legitimate request to forget.
    """
    from anchoropt.mechanisms.dedup_clear_recovery import plan_recovery

    adapter = adapter_for(backend)
    container = state.container or "archival"
    plan = plan_recovery(
        call,
        live=state.entries if state.readable else None,
        pending_write=state.pending_write,
        capacity=adapter.slots_of(container) if adapter else None,
        id_kwarg=adapter.id_kwarg if adapter else "key",
    )

    base = RepairPlan(
        False, Constraint.CLEAR_PROPOSED_AT_CAPACITY, Repair.EVICT, "",
        telemetry={"live_entries": state.count, "live_distinct": state.distinct},
    )
    if not plan["fire"]:
        declined = base.declined(plan["reason"])
        if plan.get("ladder_layer_2_candidate"):
            # Full, and nothing redundant. Dedup cannot reach this state by construction -- it is
            # what the next ladder layer (consolidation) exists for. Flagged so the population is
            # countable rather than merely declined.
            return RepairPlan(
                False, declined.constraint, Repair.DECLINE, declined.reason,
                telemetry={**base.telemetry, "ladder_layer_2_candidate": True},
            )
        return declined

    return RepairPlan(
        fire=True,
        constraint=Constraint.CLEAR_PROPOSED_AT_CAPACITY,
        repair=Repair.EVICT,
        reason=plan["reason"],
        calls=(plan["remove_call"],),
        retry_call=plan["retry_call"],           # the blocked write, VERBATIM
        invariant=plan["invariant"],
        telemetry={
            "victim_key": plan["victim_key"],
            "copies_before": plan["copies_before"],
            "preserved_keys": plan["preserved_keys"],
            "live_size": plan["live_size"],
            "capacity": plan["capacity"],
            "observation": plan["observation"],
            "suppressed_call": str(call),        # the clear that did NOT run
        },
    )
