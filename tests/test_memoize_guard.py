"""E1 (efficiency anchor): the memoising suppression guard.

Two things are tested, and the second is the one that matters:

  1. the PREDICATE is correct -- fires on a provably-uninformative repeat, never otherwise
  2. the SAVING IS IN EXECUTIONS, not proposals

(2) is checked with a COUNTING STUB standing in for the executor. It exists because the first
implementation of this anchor in the working repo memoised AFTER execution and saved nothing --
a behaviour change dressed as an efficiency one, which passed every predicate test. A test that
only checks the predicate cannot catch that.
"""

from __future__ import annotations

import pytest

from anchoropt.mechanisms.memoize_guard import (
    lookup,
    partition,
    record,
    splice,
)

KV_NOT_FOUND = '{"error": "Key not found."}'
VEC_NOT_FOUND = '{"error": "ID 7 not present in store."}'


class _KV:
    """kv backend shape: a plain dict with a `next_id` bookkeeping entry."""

    def __init__(self, **entries):
        self.core_memory = {"next_id": 3, **entries}
        self.archival_memory = {}


class _Vector:
    """vector backend shape: the dict is wrapped in `._store`."""

    class _S:
        def __init__(self, d):
            self._store = d

    def __init__(self, **entries):
        self.core_memory = self._S(dict(entries))
        self.archival_memory = self._S({})


# --------------------------------------------------------------------------------------------
# The predicate
# --------------------------------------------------------------------------------------------

def test_first_attempt_is_a_miss_so_the_tool_executes():
    """Nothing is recorded yet: there is no proof the call is uninformative."""
    memo = {}
    hit, why = lookup("core_memory_remove(key='absent')", {"a": _KV()}, memo)
    assert hit is False
    assert "no recorded outcome" in why


def test_repeat_after_a_recorded_not_found_is_a_hit_and_replays_verbatim():
    call = "core_memory_remove(key='absent')"
    insts = {"a": _KV()}
    memo = {}

    assert record([call], [KV_NOT_FOUND], memo) == 1

    hit, payload = lookup(call, insts, memo)
    assert hit is True
    assert payload == KV_NOT_FOUND, "the replay must be BYTE-IDENTICAL to the tool's own string"


def test_vector_replay_preserves_the_id():
    """Two different absent ids must replay two different strings -- not a generic message."""
    memo = {}
    record(["core_memory_remove(vec_id=7)"], [VEC_NOT_FOUND], memo)
    hit, payload = lookup("core_memory_remove(vec_id=7)", {"a": _Vector()}, memo)
    assert hit is True
    assert payload == VEC_NOT_FOUND
    assert "7" in payload


def test_a_present_target_is_never_suppressed():
    """The load-bearing safety property: a legitimate removal must always execute."""
    memo = {("core", "alive"): KV_NOT_FOUND}          # a stale record
    insts = {"a": _KV(alive="still here")}            # ...but the target is present
    hit, why = lookup("core_memory_remove(key='alive')", insts, memo)
    assert hit is False
    assert "present again" in why


def test_stale_entry_is_dropped_when_the_target_reappears():
    memo = {("core", "back"): KV_NOT_FOUND}
    insts = {"a": _KV(back="returned")}
    withheld, to_execute = partition(["core_memory_remove(key='back')"], insts, memo)
    assert withheld == {}, "must not suppress"
    assert len(to_execute) == 1, "must execute"
    assert ("core", "back") not in memo, "and must forget the stale record"


def test_successful_add_invalidates():
    memo = {}
    record(["core_memory_remove(key='k')"], [KV_NOT_FOUND], memo)
    assert ("core", "k") in memo
    record(["core_memory_add(key='k', value='v')"], ['{"id": 1}'], memo)
    assert ("core", "k") not in memo


def test_unreadable_state_fails_open():
    """No live container -> execute normally. Guessing is a correctness bug, not a saving."""
    memo = {("core", "x"): KV_NOT_FOUND}
    hit, why = lookup("core_memory_remove(key='x')", {}, memo)
    assert hit is False
    assert "unreadable" in why


@pytest.mark.parametrize("call", [
    "core_memory_clear()",
    "archival_memory_retrieve(key='k')",
    "core_memory_add(key='k', value='v')",
])
def test_other_action_classes_are_never_memoised(call):
    """Only removals. `clear` in particular must never be suppressed."""
    memo = {("core", "k"): KV_NOT_FOUND}
    hit, _ = lookup(call, {"a": _KV()}, memo)
    assert hit is False


def test_unparseable_target_fails_open():
    memo = {("core", "k"): KV_NOT_FOUND}
    hit, why = lookup("core_memory_remove(something_else=1)", {"a": _KV()}, memo)
    assert hit is False
    assert "no target parsed" in why


def test_record_only_stores_actual_not_found_results():
    """A removal that SUCCEEDED is not a memoisable outcome -- the target existed."""
    memo = {}
    assert record(["core_memory_remove(key='k')"], ['{"status": "removed"}'], memo) == 0
    assert memo == {}


# --------------------------------------------------------------------------------------------
# The saving is in EXECUTIONS -- the counting stub
# --------------------------------------------------------------------------------------------

class _CountingExecutor:
    """Stands in for the real executor and records exactly which calls reached it."""

    def __init__(self, responses=None):
        self.calls: list[str] = []
        self._responses = responses or {}

    def __call__(self, batch):
        self.calls.extend(str(c) for c in batch)
        return [self._responses.get(str(c), '{"status": "removed"}') for c in batch]


def _step(proposed, insts, memo, execute):
    """The caller contract, in order: partition -> execute SHORTENED -> splice -> record."""
    withheld, to_execute = partition(proposed, insts, memo)
    executed_results = execute(to_execute)
    results = splice(withheld, executed_results, len(proposed))
    record(to_execute, executed_results, memo)
    return withheld, results


def test_the_withheld_call_never_reaches_the_executor():
    """THE efficiency assertion. Everything else is a predicate detail."""
    call = "core_memory_remove(key='absent')"
    insts, memo = {"a": _KV()}, {}
    ex = _CountingExecutor({call: KV_NOT_FOUND})

    # First attempt: must execute, and its outcome is recorded.
    _step([call], insts, memo, ex)
    assert ex.calls == [call], "the first attempt must execute -- nothing is known yet"

    before = len(ex.calls)
    for _ in range(3):
        withheld, results = _step([call], insts, memo, ex)
        assert withheld, "a repeat against unchanged state must be withheld"
        assert results == [KV_NOT_FOUND], "and must replay the verbatim result"

    assert len(ex.calls) - before == 0, (
        "3 repeats proposed, and the executor was invoked 0 times for them. "
        "If this is nonzero the anchor saves nothing, whatever the predicate does."
    )


def test_ordering_is_preserved_when_a_hit_sits_between_live_calls():
    """Delivery zips POSITIONALLY, so a spliced result must land on its own call."""
    absent = "core_memory_remove(key='gone')"
    memo = {("core", "gone"): KV_NOT_FOUND}
    insts = {"a": _KV(other="x")}
    ex = _CountingExecutor()

    proposed = [
        "core_memory_add(key='a', value='1')",
        absent,                                     # <- withheld, index 1
        "core_memory_add(key='b', value='2')",
    ]
    withheld, results = _step(proposed, insts, memo, ex)

    assert set(withheld) == {1}
    assert len(results) == len(proposed), "full length restored"
    assert results[1] == KV_NOT_FOUND, "the replay landed at its ORIGINAL index"
    assert absent not in ex.calls, "and never executed"
    assert len(ex.calls) == 2, "only the two adds executed"


def test_splice_raises_rather_than_silently_mislabelling():
    """A length mismatch corrupts every later result. Fail loudly, not into a flag."""
    with pytest.raises(ValueError, match="left over"):
        splice({}, ["r1", "r2"], 1)


def test_splice_fills_short_executor_output():
    assert splice({1: "cached"}, ["live"], 3) == ["live", "cached", ""]


def test_a_step_with_no_hits_is_byte_identical_to_no_guard():
    """The noop property: with an empty memo the guard must change nothing at all."""
    proposed = ["core_memory_add(key='a', value='1')", "core_memory_remove(key='k')"]
    insts, memo = {"a": _KV(k="present")}, {}
    ex = _CountingExecutor()
    withheld, results = _step(proposed, insts, memo, ex)
    assert withheld == {}
    assert ex.calls == [str(c) for c in proposed]
    assert len(results) == 2
