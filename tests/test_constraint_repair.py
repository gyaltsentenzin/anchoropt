"""The core algorithm: one control flow, three repair strategies.

Every test here corresponds to a way this was got wrong at least once on a real experiment. The
point of factoring the flow out of four separate mechanisms is that a new anchor inherits these
fixes; the point of these tests is that it cannot lose them.
"""

from __future__ import annotations

from anchoropt.mechanisms.constraint_repair import (
    Constraint,
    Repair,
    RepairPlan,
    StateView,
    StoreAdapter,
    adapter_for,
    backends_for,
    detect_constraint,
    payload_of,
    plan_repair,
    register_adapter,
    registered_backends,
    strategy_for,
    verify_invariant,
)

FULL_VECTOR = '{"error": "memory size exceeds maximum size"}'
FULL_KV = '{"error": "Long term memory is full"}'
TOO_LONG = '{"error": "Entry length exceeds maximum length"}'
BLOB_OVERFLOW = '{"error": "Memory is too long after appending"}'


# ---------------------------------------------------------------------------------------------
# detect: keyed on the CONSTRAINT, per backend -- never on the tool name
# ---------------------------------------------------------------------------------------------

def test_a_successful_result_is_never_a_violation():
    assert detect_constraint('{"ok": true}', "vector") is None
    assert detect_constraint("", "vector") is None
    assert detect_constraint(None, "vector") is None


def test_backends_word_the_same_constraint_differently():
    """The trap that made one anchor fire on vector and never once on kv.

    Both mean "the container is full", and neither string matches the other's backend.
    """
    assert detect_constraint(FULL_VECTOR, "vector") is Constraint.NO_SLOTS
    assert detect_constraint(FULL_KV, "kv") is Constraint.NO_SLOTS
    assert detect_constraint(FULL_VECTOR, "kv") is None
    assert detect_constraint(FULL_KV, "vector") is None


def test_an_unregistered_backend_reads_as_unknown_not_as_fine():
    assert detect_constraint(FULL_VECTOR, "graph_store") is None


def test_exposure_set_is_derived_from_the_pattern_table():
    """An anchor's backends must come from observed incidence, never be asserted."""
    assert backends_for(Constraint.NO_SLOTS) == frozenset({"vector", "kv"})
    assert backends_for(Constraint.BLOB_WOULD_OVERFLOW) == frozenset({"rec_sum"})
    assert backends_for(Constraint.NOT_FOUND) == frozenset(), "no patterns registered yet"


def test_every_registered_adapter_declares_at_least_one_constraint():
    for name in registered_backends():
        adapter = adapter_for(name)
        assert adapter.error_patterns, f"{name} declares no constraints, so it can never fire"


# ---------------------------------------------------------------------------------------------
# observe: unreadable is NOT empty
# ---------------------------------------------------------------------------------------------

def test_unreadable_state_declines_and_does_not_fail_open():
    """The bug this distinction exists to prevent: a guard tested with `if not state` treats
    "cannot see the store" as "the store is empty" and acts when it is least entitled to."""
    plan = plan_repair("archival_memory_add(text='x')", FULL_VECTOR, StateView(readable=False), "vector")
    assert not plan.fire
    assert plan.repair is Repair.DECLINE
    assert "unreadable" in plan.reason


def test_readable_but_empty_is_a_different_state_than_unreadable():
    empty = StateView(readable=True, entries={})
    plan = plan_repair("archival_memory_add(text='x')", FULL_VECTOR, empty, "vector")
    assert not plan.fire
    # It declines for lack of a duplicate, NOT for unreadability -- a different reason.
    assert "unreadable" not in plan.reason
    assert "duplicate" in plan.reason


def test_state_view_counts_are_computed_not_stored():
    state = StateView(readable=True, entries={1: "a", 2: "b", 3: "a"})
    assert state.count == 3
    assert state.distinct == 2
    assert state.copies_of("a") == 2
    assert state.copies_of("zzz") == 0


# ---------------------------------------------------------------------------------------------
# propose: nothing executes inside the planner
# ---------------------------------------------------------------------------------------------

def test_the_planner_returns_data_and_dispatches_nothing():
    """The most expensive bug class here: a guard placed AFTER execution 'prevents' nothing.

    A pure planner structurally cannot make that mistake -- there is nothing to execute in it.
    """
    state = StateView(readable=True, entries={1: "dup", 2: "dup"})
    plan = plan_repair("archival_memory_add(text='new')", FULL_VECTOR, state, "vector")
    assert isinstance(plan, RepairPlan)
    assert isinstance(plan.dispatch_sequence, tuple)
    assert all(isinstance(c, str) for c in plan.dispatch_sequence)


def test_budget_bounds_the_repair():
    """Unbounded eviction empties the container while every single step passes its own test."""
    state = StateView(readable=True, entries={1: "dup", 2: "dup"})
    ok = plan_repair("archival_memory_add(text='n')", FULL_VECTOR, state, "vector", budget_used=0, budget=3)
    assert ok.fire
    spent = plan_repair("archival_memory_add(text='n')", FULL_VECTOR, state, "vector", budget_used=3, budget=3)
    assert not spent.fire
    assert "budget" in spent.reason


def test_a_plan_that_repairs_without_retrying_is_refused():
    """A repair that fixes the state and drops the payload has fixed the symptom only."""
    plan = RepairPlan(
        fire=True, constraint=Constraint.NO_SLOTS, repair=Repair.EVICT,
        reason="test", calls=("archival_memory_remove(vec_id=1)",), retry_call=None,
    )
    assert plan.dispatch_sequence == ("archival_memory_remove(vec_id=1)",)
    # plan_repair enforces the retry; construct the failure case through it.
    import anchoropt.mechanisms.constraint_repair as cr

    original = cr._STRATEGIES[Constraint.NO_SLOTS]
    try:
        cr._STRATEGIES[Constraint.NO_SLOTS] = lambda call, state, backend: plan
        got = plan_repair("archival_memory_add(text='x')", FULL_VECTOR,
                          StateView(readable=True, entries={1: "a", 2: "a"}), "vector")
        assert not got.fire
        assert "verbatim retry" in got.reason
    finally:
        cr._STRATEGIES[Constraint.NO_SLOTS] = original


def test_every_constraint_with_a_strategy_is_reachable():
    for constraint in (Constraint.NO_SLOTS, Constraint.ENTRY_TOO_LONG, Constraint.BLOB_WOULD_OVERFLOW):
        assert strategy_for(constraint) is not None, f"{constraint} has no registered strategy"


# ---------------------------------------------------------------------------------------------
# A5 / EVICT
# ---------------------------------------------------------------------------------------------

def test_evict_picks_the_last_written_duplicate_and_preserves_the_earliest():
    state = StateView(readable=True, entries={1: "fact a", 2: "fact b", 3: "fact a"})
    plan = plan_repair("archival_memory_add(text='new')", FULL_VECTOR, state, "vector")
    assert plan.fire and plan.repair is Repair.EVICT
    assert plan.telemetry["victim_id"] == 3, "the LAST copy goes; the earliest is preserved"
    assert plan.telemetry["copies_before"] == 2


def test_evict_retries_the_original_call_verbatim():
    """Never synthesize. A synthesized retry failed on every kv firing AFTER the eviction had
    already succeeded, destroying a duplicate and losing the write."""
    call = "archival_memory_add(text='new fact', key='k')"
    state = StateView(readable=True, entries={1: "d", 2: "d"})
    plan = plan_repair(call, FULL_VECTOR, state, "vector")
    assert plan.retry_call == call
    assert plan.dispatch_sequence[-1] == call, "the verbatim retry is dispatched last"


def test_evict_declines_when_no_duplicate_exists():
    """"No lossless eviction exists here" is a legitimate and frequent answer."""
    state = StateView(readable=True, entries={1: "a", 2: "b", 3: "c"})
    plan = plan_repair("archival_memory_add(text='n')", FULL_VECTOR, state, "vector")
    assert not plan.fire
    assert "lossless" in plan.reason


def test_evict_invariant_is_verified_against_observed_state():
    state = StateView(readable=True, entries={1: "dup", 2: "other", 3: "dup"})
    plan = plan_repair("archival_memory_add(text='n')", FULL_VECTOR, state, "vector")
    after_ok = StateView(readable=True, entries={1: "dup", 2: "other"})
    result = verify_invariant(plan, after_ok)
    assert result["ok"] and result["checked"] and result["remaining"] == 1

    after_bad = StateView(readable=True, entries={2: "other"})
    violated = verify_invariant(plan, after_bad)
    assert not violated["ok"]
    assert "VIOLATED" in violated["reason"]


def test_invariant_check_fails_closed_when_state_is_unreadable_afterwards():
    state = StateView(readable=True, entries={1: "dup", 2: "dup"})
    plan = plan_repair("archival_memory_add(text='n')", FULL_VECTOR, state, "vector")
    result = verify_invariant(plan, StateView(readable=False))
    assert not result["ok"]


def test_non_evict_repairs_have_no_information_invariant_to_check():
    state = StateView(readable=True, entries={})
    plan = plan_repair(f"core_memory_add(text='{'x' * 400}')", TOO_LONG, state, "vector")
    result = verify_invariant(plan, state)
    assert result["ok"] and result["checked"] is False


# ---------------------------------------------------------------------------------------------
# A6 / RELOCATE
# ---------------------------------------------------------------------------------------------

def test_relocate_moves_an_over_long_entry_to_the_larger_container():
    payload = "x" * 400            # over core's 300, under archival's 2000
    state = StateView(readable=True, entries={}, container="core")
    plan = plan_repair(f"core_memory_add(text='{payload}')", TOO_LONG, state, "vector")
    assert plan.fire and plan.repair is Repair.RELOCATE
    assert plan.retry_call.startswith("archival_memory_add")
    assert plan.telemetry["payload_len"] == 400
    assert plan.calls == (), "the relocation IS the retry; there is no separate repair call"


def test_relocate_declines_when_the_store_has_no_second_container():
    """rec_sum has nowhere to relocate TO. The strategy must decline, not invent a destination.

    That absence is the whole reason A7 exists, so getting it wrong here would hide the locus.
    """
    state = StateView(readable=True, entries={}, container="blob")
    plan = plan_repair("memory_append(content='x')", TOO_LONG, state, "rec_sum")
    # rec_sum does not even declare ENTRY_TOO_LONG, so this is not its constraint at all.
    assert not plan.fire


def test_relocate_declines_rather_than_silently_switching_to_a_reduce():
    """If the payload will not fit the destination either, that needs a REDUCE -- which has a much
    weaker guarantee. Switching repair kind silently would smuggle a lossy operation in behind a
    lossless one's evidence."""
    payload = "y" * 2500          # over archival's 2000 too
    state = StateView(readable=True, entries={}, container="core")
    plan = plan_repair(f"core_memory_add(text='{payload}')", TOO_LONG, state, "vector")
    assert not plan.fire
    assert "reducing is a different repair" in plan.reason


def test_relocate_declines_on_an_unparseable_payload():
    """An unparseable call is passed through, never dropped."""
    state = StateView(readable=True, entries={}, container="core")
    plan = plan_repair("core_memory_add(**kwargs)", TOO_LONG, state, "vector")
    assert not plan.fire
    assert "unparseable" in plan.reason


# ---------------------------------------------------------------------------------------------
# A7 / REDUCE
# ---------------------------------------------------------------------------------------------

def test_reduce_fires_on_a_blob_and_requires_generation():
    state = StateView(readable=True, blob="z" * 9900)
    plan = plan_repair("memory_append(content='new')", BLOB_OVERFLOW, state, "rec_sum")
    assert plan.fire and plan.repair is Repair.REDUCE
    assert plan.telemetry["requires_generation"] is True
    assert plan.telemetry["blob_len"] == 9900
    assert plan.invariant and "SHORTER" in plan.invariant


def test_reduce_declines_when_the_blob_cannot_be_read_as_a_string():
    """Backend-shape trap: this backend's state is a bare string, not a container of entries.
    Walking it as a dict finds nothing and logs zero events."""
    state = StateView(readable=True, entries={1: "a"}, blob=None)
    plan = plan_repair("memory_append(content='n')", BLOB_OVERFLOW, state, "rec_sum")
    assert not plan.fire
    assert "string, not a container" in plan.reason


def test_reduce_keys_on_the_overflow_state_not_the_call():
    """~18.7x append-retry amplification: a per-call gate would recompact on every retry."""
    blob = "z" * 9900
    a = plan_repair("memory_append(content='same')", BLOB_OVERFLOW,
                    StateView(readable=True, blob=blob), "rec_sum")
    b = plan_repair("memory_append(content='same')", BLOB_OVERFLOW,
                    StateView(readable=True, blob=blob), "rec_sum")
    assert a.telemetry["state_key"] == b.telemetry["state_key"], "same state -> same key"
    c = plan_repair("memory_append(content='different payload')", BLOB_OVERFLOW,
                    StateView(readable=True, blob=blob), "rec_sum")
    assert c.telemetry["state_key"] != a.telemetry["state_key"], "different pending -> different key"


# ---------------------------------------------------------------------------------------------
# payload parsing
# ---------------------------------------------------------------------------------------------

def test_payload_parsing_handles_both_shapes_and_admits_failure():
    assert payload_of("core_memory_add(text='hello')") == "hello"
    assert payload_of('core_memory_add(value="hi there")') == "hi there"
    assert payload_of("archival_memory_add('positional form')") == "positional form"
    assert payload_of("core_memory_add(**kwargs)") is None, "None means pass through, not empty"
    assert payload_of("") is None


# ---------------------------------------------------------------------------------------------
# The porting seam -- the contract a collaborator relies on
# ---------------------------------------------------------------------------------------------
#
# These tests are the promise that porting means WRITING ONE ADAPTER, not editing the core. If they
# start needing core changes to pass, the abstraction has leaked and the promise is broken.

def _toy_adapter(name: str = "toy_store") -> StoreAdapter:
    return StoreAdapter(
        name=name,
        error_patterns={
            Constraint.NO_SLOTS: r"capacity exceeded",
            Constraint.ENTRY_TOO_LONG: r"item too large",
        },
        entry_caps={"main": 512, "overflow": 4096},
        relocation_target="overflow",
        write_verbs=("store_add",),
        id_kwarg="item_id",
    )


def test_a_new_benchmark_needs_only_an_adapter():
    """The whole point: register one record and all three repair strategies work."""
    register_adapter(_toy_adapter())
    state = StateView(readable=True, entries={7: "dup", 8: "other", 9: "dup"}, container="main")

    evict = plan_repair("store_add(text='new')", '{"error": "capacity exceeded"}', state, "toy_store")
    assert evict.fire and evict.repair is Repair.EVICT
    assert evict.retry_call == "store_add(text='new')", "the original call, verbatim"

    relocate = plan_repair(
        f"store_add(text='{'x' * 1000}')", '{"error": "item too large"}', state, "toy_store"
    )
    assert relocate.fire and relocate.repair is Repair.RELOCATE
    assert "overflow" in relocate.reason and "4096" in relocate.reason


def test_the_adapter_supplies_the_id_keyword_so_the_removal_signature_is_right():
    """A synthesized removal with the wrong signature fails AFTER the eviction has succeeded --
    destroying a duplicate and losing the write, which is worse than doing nothing."""
    register_adapter(_toy_adapter())
    state = StateView(readable=True, entries={7: "dup", 9: "dup"}, container="main")
    plan = plan_repair("store_add(text='n')", '{"error": "capacity exceeded"}', state, "toy_store")
    assert plan.calls == ("main_memory_remove(item_id=9)",)

    # The same core, a different backend, a different keyword -- kv addresses entries by key.
    kv_state = StateView(readable=True, entries={1: "dup", 2: "dup"}, container="archival")
    kv_plan = plan_repair("archival_memory_add(value='n')", FULL_KV, kv_state, "kv")
    assert "key=" in kv_plan.calls[0], "kv is key-addressed, vector is id-addressed"


def test_adapter_caps_bound_relocation_rather_than_a_hardcoded_constant():
    register_adapter(_toy_adapter())
    state = StateView(readable=True, entries={}, container="main")
    too_big = plan_repair(
        f"store_add(text='{'x' * 9000}')", '{"error": "item too large"}', state, "toy_store"
    )
    assert not too_big.fire
    assert "4096" in too_big.reason, "the DECLINE must cite the adapter's cap, not a BFCL constant"


def test_a_store_with_no_second_container_cannot_relocate():
    """The structural fact that makes A7 necessary, expressed as data rather than a special case."""
    single = StoreAdapter(
        name="single_container_store",
        error_patterns={Constraint.ENTRY_TOO_LONG: r"item too large"},
        entry_caps={},                      # nothing to relocate into
        relocation_target=None,
        single_container=True,
    )
    register_adapter(single)
    state = StateView(readable=True, entries={}, container="only")
    plan = plan_repair("add(text='x')", '{"error": "item too large"}', state,
                       "single_container_store")
    assert not plan.fire
    assert "no alternative container" in plan.reason


def test_registering_an_adapter_extends_the_exposure_set():
    """Exposure is DERIVED from what is registered, so it cannot drift from the adapters."""
    before = backends_for(Constraint.NO_SLOTS)
    register_adapter(_toy_adapter("another_store"))
    after = backends_for(Constraint.NO_SLOTS)
    assert "another_store" in after
    assert before <= after


def test_the_shipped_bfcl_adapters_are_a_worked_example_not_a_default():
    """All three ship registered, each with the notes that explain why it is shaped that way."""
    for name in ("vector", "kv", "rec_sum"):
        adapter = adapter_for(name)
        assert adapter is not None, f"{name} adapter must ship"
        assert adapter.notes, f"{name} must record WHY its shape differs -- that is the portable part"
    # And the shape differences that actually bit, pinned:
    assert adapter_for("vector").id_kwarg == "vec_id"
    assert adapter_for("kv").id_kwarg == "key"
    assert adapter_for("rec_sum").relocation_destination() is None, "no second container -- hence A7"


# ---------------------------------------------------------------------------------------------
# A8 / proposed destruction -- a signal that is NOT an error
# ---------------------------------------------------------------------------------------------
#
# Every other strategy keys on a refusal. This one keys on a PROPOSAL plus live state, because after
# the destructive call runs there is nothing left to save. That makes it the one place where
# `result=None` must still produce a verdict.

CLEAR = "archival_memory_clear()"
PENDING = {"call": "archival_memory_add(key='new', value='v')", "container": "archival"}


def _full_with_duplicate() -> dict:
    """A kv archival container at 50/50 holding one self-inflicted duplicate pair."""
    live = {f"fact_{i}": f"value {i}" for i in range(48)}
    live["diet_focus"] = "cutting out sugary snacks"
    live["diet_focus_unique"] = "cutting out sugary snacks"   # the model's own *_unique retry
    return live


def _full_all_distinct() -> dict:
    return {f"fact_{i}": f"value {i}" for i in range(50)}


def test_a_proposed_clear_at_capacity_is_detected_without_any_error():
    """`result` is None here BY CONSTRUCTION -- nothing has failed yet."""
    state = StateView(readable=True, entries=_full_with_duplicate(), container="archival",
                      pending_write=PENDING)
    plan = plan_repair(CLEAR, None, state, "kv")
    assert plan.fire and plan.repair is Repair.EVICT
    assert plan.constraint is Constraint.CLEAR_PROPOSED_AT_CAPACITY


def test_it_evicts_the_self_inflicted_retry_and_preserves_the_original():
    state = StateView(readable=True, entries=_full_with_duplicate(), container="archival",
                      pending_write=PENDING)
    plan = plan_repair(CLEAR, None, state, "kv")
    assert plan.telemetry["victim_key"] == "diet_focus_unique", (
        "the LAST-written copy goes; the original the model first stored is preserved"
    )
    assert plan.telemetry["preserved_keys"] == ["diet_focus"]
    assert plan.telemetry["copies_before"] == 2


def test_the_suppressed_clear_is_recorded_and_the_retry_is_verbatim():
    state = StateView(readable=True, entries=_full_with_duplicate(), container="archival",
                      pending_write=PENDING)
    plan = plan_repair(CLEAR, None, state, "kv")
    assert plan.telemetry["suppressed_call"] == CLEAR, "what was NOT run must be recorded"
    assert plan.retry_call == PENDING["call"], "the blocked write, byte for byte"
    assert plan.dispatch_sequence == (
        "archival_memory_remove(key='diet_focus_unique')", PENDING["call"],
    ), "evict first, then retry -- and the retry is last"


def test_the_removal_uses_the_backends_own_address_space():
    """kv removes by key, vector by id. A synthesized call in the wrong shape is rejected by the
    tool, which the gate would then report as a failed eviction rather than as a bug."""
    live = _full_with_duplicate()
    kv = plan_repair(CLEAR, None, StateView(readable=True, entries=live, container="archival",
                                            pending_write=PENDING), "kv")
    assert "key=" in kv.calls[0]

    vec_live = {i: f"value {i}" for i in range(48)}
    vec_live[90] = "same text"
    vec_live[91] = "same text"
    vec = plan_repair(CLEAR, None, StateView(readable=True, entries=vec_live, container="archival",
                                             pending_write=PENDING), "vector")
    assert "vec_id=" in vec.calls[0]


def test_at_capacity_with_no_duplicate_declines_and_flags_the_next_ladder_layer():
    """Dedup CANNOT create room here. Declining is correct, and the population is countable."""
    state = StateView(readable=True, entries=_full_all_distinct(), container="archival",
                      pending_write=PENDING)
    plan = plan_repair(CLEAR, None, state, "kv")
    assert not plan.fire
    assert "no exact duplicate" in plan.reason
    assert plan.telemetry.get("ladder_layer_2_candidate") is True, (
        "these are the states consolidation exists for; they must be counted, not just declined"
    )


def test_a_clear_on_a_container_with_room_is_left_alone():
    """It may be a legitimate request to forget. Not an anchor's business."""
    state = StateView(readable=True, entries={"a": "1", "b": "2"}, container="archival",
                      pending_write=PENDING)
    assert not plan_repair(CLEAR, None, state, "kv").fire


def test_without_a_pending_write_freeing_a_slot_accomplishes_nothing():
    state = StateView(readable=True, entries=_full_with_duplicate(), container="archival")
    plan = plan_repair(CLEAR, None, state, "kv")
    assert not plan.fire
    assert "no pending blocked write" in plan.reason


def test_a_pending_write_for_a_different_container_does_not_justify_the_eviction():
    state = StateView(readable=True, entries=_full_with_duplicate(), container="archival",
                      pending_write={"call": "core_memory_add(key='k', value='v')",
                                     "container": "core"})
    plan = plan_repair(CLEAR, None, state, "kv")
    assert not plan.fire
    assert "would not admit that write" in plan.reason


def test_the_model_is_told_that_capacity_was_supplied():
    """The predecessor that refused clears without supplying capacity stalled: 85 refusals, 0 of 14
    episodes closed. The observation has to explain the remedy or the model just re-proposes."""
    state = StateView(readable=True, entries=_full_with_duplicate(), container="archival",
                      pending_write=PENDING)
    observation = plan_repair(CLEAR, None, state, "kv").telemetry["observation"]
    assert "at capacity" in observation
    assert "equivalent copy remains" in observation
    assert "re-applied the blocked write" in observation
    assert "no other entry was destroyed" in observation


def test_the_lossless_invariant_is_verified_against_state_after_the_eviction():
    from anchoropt.mechanisms.dedup_clear_recovery import verify

    live = _full_with_duplicate()
    state = StateView(readable=True, entries=live, container="archival", pending_write=PENDING)
    plan = plan_repair(CLEAR, None, state, "kv")

    after_ok = {k: v for k, v in live.items() if k != "diet_focus_unique"}
    good = verify(after_ok, plan.telemetry)
    assert good["ok"] and good["copies_remaining"] == 1 and good["size_after"] == 49

    # Both copies gone: the one outcome the acceptance argument forbids.
    after_bad = {k: v for k, v in live.items() if not k.startswith("diet_focus")}
    bad = verify(after_bad, plan.telemetry)
    assert not bad["ok"]
    assert "INVARIANT VIOLATED" in bad["reason"]


def test_the_equivalence_rule_is_shared_with_a5_not_reinvented():
    """Two anchors claiming the same information-preservation invariant must mean the same thing by
    'duplicate', or one of them is quietly weaker than its evidence."""
    from anchoropt.mechanisms import dedup_clear_recovery, lossless_eviction

    for raw in ("  Cutting  Out   SUGARY snacks ", "cutting out sugary snacks"):
        assert dedup_clear_recovery.normalize(raw) == lossless_eviction.normalize(raw)


def test_the_episode_budget_bounds_it_but_the_duplicate_supply_binds_first():
    """The cap is a backstop, not a throttle -- and a sibling anchor's cap of 3 would have refused
    firings this gate can supply losslessly."""
    from anchoropt.mechanisms.dedup_clear_recovery import MAX_PER_EPISODE, plan_recovery

    assert MAX_PER_EPISODE == 8
    spent = plan_recovery(CLEAR, _full_with_duplicate(), PENDING, capacity=50,
                          firings_so_far=MAX_PER_EPISODE)
    assert not spent["fire"] and "budget spent" in spent["reason"]
