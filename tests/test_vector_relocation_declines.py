"""The relocation primitive on VECTOR: reachable, addressing-aware, and safe in that order.

HISTORY OF THIS FILE, kept because it is the finding.

It used to assert the opposite -- that the primitive must DECLINE on vector -- on the stated reason
that "a vector pair needs value-based victim selection first". The structural facts it recorded were
all correct: vector's stores are `VectorStore` OBJECTS whose dict lives at `._store`, entries are
addressed by an AUTO-ASSIGNED integer `vec_id`, and `core_memory_add(text)` takes no key.

The CONCLUSION was wrong, and measurement is what showed it (rounds/AUTONOMY/R12):

  * `VectorStore._store` is `dict[int, str]` in INSERTION ORDER with monotonic ids, so
    `pick_victim`'s "first-inserted" rule is already deterministic there -- nothing new was needed.
  * `anchoropt/mechanisms/lossless_eviction._live_container` had ALREADY unwrapped both shapes for
    exactly this reason, so the repo contained the working pattern the whole time.
  * On the live corpus A1 was requested 39 times per vector arm and declined all 39 with
    `relocate_declined = "live stores unavailable: ..."`, against 41 core-size refusals that ALL had
    archival room (median 49 free slots of 50). The opportunity was real and unreachable by
    construction.

And because these three tests ERROR on a missing `faiss` in most environments, the assertion that
froze the ceiling was never actually executed. A decline recorded as intended behaviour, pinned by a
test that never ran, is how a correct local judgement becomes an invisible cap on a whole cell.

What is asserted now: the primitive reaches BOTH shapes, picks identity by the pair's DECLARED
addressing (value-based for ordinal stores, because ids are per-store and a coincidental id collision
would otherwise skip the destination write and still remove the source entry), and preserves the
same order-of-operations safety property on vector that it has on kv.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "scripts", REPO / "benchmarks" / "bfcl_v4" / "harness"):
    sys.path.insert(0, str(p))

import capacity_relocate as CR                                                   # noqa: E402


# --------------------------------------------------------------------------------------------------
# Shape-level facts that need no faiss: the pair table and the identity rule.
# --------------------------------------------------------------------------------------------------

def test_both_addressing_shapes_are_declared():
    """kv and vector are two DECLARED pairs, not two code paths."""
    shapes = [p["addressing"] for p in CR.RELOCATION_PAIRS]
    assert shapes == ["keyed", "ordinal"], shapes
    for pair in CR.RELOCATION_PAIRS:
        assert pair["source"] == "core_memory"
        assert pair["destination"] == "archival_memory"
        # Caps are the same on both; the strict-dominance safety argument is shared.
        assert (pair["source_cap"], pair["destination_cap"]) == (7, 50)


def test_the_two_call_templates_match_each_backends_real_signature():
    keyed, ordinal = CR.RELOCATION_PAIRS
    assert keyed["add_template"].format(add_verb="archival_memory_add", key="role",
                                        value="MD") == "archival_memory_add(key='role', value='MD')"
    assert keyed["remove_template"].format(remove_verb="core_memory_remove", key="role",
                                           value="MD") == "core_memory_remove(key='role')"
    # vector: add takes only `text`, remove takes an integer `vec_id`.
    assert ordinal["add_template"].format(add_verb="archival_memory_add", key=3,
                                          value="MD") == "archival_memory_add(text='MD')"
    assert ordinal["remove_template"].format(remove_verb="core_memory_remove", key=3,
                                              value="MD") == "core_memory_remove(vec_id=3)"


def test_ordinal_identity_is_value_based_and_that_prevents_data_loss():
    """The bug this rule exists to prevent: a coincidental id collision must NOT read as preserved.

    If it did, `relocate_and_retry` would skip the destination write ("already preserved") and still
    remove the source entry -- destroying the fact. Core id 3 and archival id 3 are unrelated.
    """
    destination = {3: "a totally different fact"}
    # keyed: id 3 present with a DIFFERENT value -> not preserved (correct either way)
    assert CR.already_preserved(3, "the victim fact", destination, "keyed") is False
    # ordinal: the id collides but the VALUE is absent -> must be False
    assert CR.already_preserved(3, "the victim fact", destination, "ordinal") is False
    # ordinal: same value under a DIFFERENT id -> genuinely preserved
    assert CR.already_preserved(99, "a totally different fact", destination, "ordinal") is True
    # keyed: same value under a different key is NOT the same entry
    assert CR.already_preserved(99, "a totally different fact", destination, "keyed") is False


def test_unreachable_stores_still_decline_with_a_reason():
    """Reachability, not backend names. An object exposing neither shape must decline, not guess."""
    class Opaque:
        core_memory = object()
        archival_memory = object()

    out = CR.relocate_and_retry(failing_call="core_memory_add(text='x')",
                                involved_instances=[Opaque()],
                                execute=lambda calls: ([None], None))
    assert out["relocate_ok"] is False
    assert "live stores unavailable" in out["relocate_declined"]


def test_ordinal_pair_is_selected_for_a_vector_shaped_instance_without_faiss():
    """`_select_pair` chooses by what the instance exposes, so a stub is sufficient here."""
    class Store:
        def __init__(self, d): self._store = d

    class VectorShaped:
        def __init__(self):
            self.core_memory = Store({0: "oldest", 1: "newer"})
            self.archival_memory = Store({})

    inst = VectorShaped()
    pair, (src, dst) = CR._select_pair([inst])
    assert pair["addressing"] == "ordinal"
    # Returned BY REFERENCE -- every verification step reads live state back through these.
    assert src is inst.core_memory._store
    assert dst is inst.archival_memory._store
    # First-inserted victim, deterministically.
    assert CR.pick_victim(src, dst, addressing="ordinal") == (0, "oldest")


def test_keyed_pair_is_still_selected_for_a_kv_shaped_instance():
    """The kv path must be untouched: it is an ACCEPTED, measured controller."""
    class KvShaped:
        def __init__(self):
            self.core_memory = {"role": "MD"}
            self.archival_memory = {}

    inst = KvShaped()
    pair, (src, dst) = CR._select_pair([inst])
    assert pair["addressing"] == "keyed"
    assert src is inst.core_memory


# --------------------------------------------------------------------------------------------------
# End-to-end against the REAL backend. Skipped, never silently passed, when faiss is absent.
# --------------------------------------------------------------------------------------------------

faiss_required = pytest.importorskip


@pytest.fixture(scope="module")
def full_vector_core():
    faiss_required("faiss")
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_vector import (
        MemoryAPI_vector)
    api = MemoryAPI_vector()
    api._load_scenario({"model_result_dir": pathlib.Path(tempfile.mkdtemp()),
                        "test_id": "memory_vector_0-x-0", "scenario": "x", "long_context": False})
    for i in range(7):
        assert "error" not in api.core_memory_add(text="fact number %d about something" % i)
    return api


def test_vector_really_is_slot_limited_with_the_same_refusal_string(full_vector_core):
    r = full_vector_core.core_memory_add(text="one more")
    assert r["error"] == "Memory size exceeds maximum size of 7 entries."


def test_vector_stores_are_objects_not_dicts(full_vector_core):
    """The structural fact. It is why `store_attr` exists -- not why the primitive declines."""
    assert not isinstance(full_vector_core.core_memory, dict)
    assert type(full_vector_core.core_memory).__name__ == "VectorStore"
    assert isinstance(full_vector_core.core_memory._store, dict)


def test_the_primitive_now_RELOCATES_on_vector_and_the_blocked_write_lands(full_vector_core):
    """The four steps, against the real store, in the order that IS the safety property."""
    api = full_vector_core
    before_core = dict(api.core_memory._store)
    assert len(before_core) == 7

    out = CR.relocate_and_retry(
        failing_call="core_memory_add(text='the eighth fact that cannot fit')",
        involved_instances=[api],
        execute=lambda calls: ([api._execute_one(c) for c in calls], None)
        if hasattr(api, "_execute_one") else (_dispatch(api, calls), None))

    assert out["relocate_addressing"] == "ordinal"
    assert out["relocate_ok"] is True, out
    assert out["relocate_write_verified"] is True
    assert out["relocate_removed_verified"] is True
    assert out["relocate_copy_survives_in_destination"] is True
    assert out.get("relocate_invariant_violated") is not True
    assert out["relocate_slot_freed"] is True
    assert out["relocate_retry_landed"] is True
    # Nothing was lost: the victim's TEXT is in archival, and core is back under its cap.
    victim_text = before_core[min(before_core)]
    assert victim_text in api.archival_memory._store.values()
    assert len(api.core_memory._store) == 7          # freed one, then the retry filled it
    assert "the eighth fact that cannot fit" in api.core_memory._store.values()


def _dispatch(api, calls):
    """Execute `name(kwargs)` against the instance, the way the evaluator's executor does."""
    import ast
    results = []
    for c in calls:
        node = ast.parse(str(c), mode="eval").body
        fn = getattr(api, node.func.id)
        kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        results.append(fn(**kw))
    return results


# --------------------------------------------------------------------------------------------------
# The paired-control gate. R12's arm and control differ in exactly ONE declared way.
# --------------------------------------------------------------------------------------------------

def test_the_addressing_gate_can_only_narrow_never_invent(monkeypatch):
    """A typo must yield a decline with a reason, not a silently different mechanism."""
    monkeypatch.delenv("ANCHOROPT_RELOCATE_ADDRESSING", raising=False)
    assert CR.enabled_addressings() == ("keyed", "ordinal")
    monkeypatch.setenv("ANCHOROPT_RELOCATE_ADDRESSING", "keyed")
    assert CR.enabled_addressings() == ("keyed",)
    monkeypatch.setenv("ANCHOROPT_RELOCATE_ADDRESSING", "ordinal,keyed")
    assert CR.enabled_addressings() == ("keyed", "ordinal")      # declared order, not env order
    monkeypatch.setenv("ANCHOROPT_RELOCATE_ADDRESSING", "sideways")
    assert CR.enabled_addressings() == ()                        # cannot invent a pair


def test_keyed_only_reproduces_the_pre_R12_decline_on_a_vector_shaped_store(monkeypatch):
    """The CONTROL arm's behaviour, pinned: identical to what the 39 live declines recorded."""
    class Store:
        def __init__(self, d): self._store = d

    class VectorShaped:
        def __init__(self):
            self.core_memory = Store({0: "oldest", 1: "newer"})
            self.archival_memory = Store({})

    inst = VectorShaped()
    monkeypatch.setenv("ANCHOROPT_RELOCATE_ADDRESSING", "keyed")
    assert CR._select_pair([inst]) is None
    out = CR.relocate_and_retry(failing_call="core_memory_add(text='x')",
                               involved_instances=[inst],
                               execute=lambda calls: ([None], None))
    assert out["relocate_ok"] is False
    assert "live stores unavailable" in out["relocate_declined"]
    assert out["relocate_enabled_addressings"] == "keyed"
    assert inst.core_memory._store == {0: "oldest", 1: "newer"}, "a declined repair changed the store"
    assert inst.archival_memory._store == {}, "a declined repair wrote somewhere"

    # ...and with the gate open, the SAME instance is reachable. One declared difference.
    monkeypatch.setenv("ANCHOROPT_RELOCATE_ADDRESSING", "keyed,ordinal")
    pair, _ = CR._select_pair([inst])
    assert pair["addressing"] == "ordinal"


def test_keyed_only_does_not_disturb_a_kv_shaped_store(monkeypatch):
    """The control must still relocate on kv -- A1 is an ACCEPTED controller there."""
    class KvShaped:
        def __init__(self):
            self.core_memory = {"role": "MD"}
            self.archival_memory = {}

    monkeypatch.setenv("ANCHOROPT_RELOCATE_ADDRESSING", "keyed")
    pair, (src, _) = CR._select_pair([KvShaped()])
    assert pair["addressing"] == "keyed"
