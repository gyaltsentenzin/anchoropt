"""`constraint_state` must be emitted for STORE-OBJECT backends, not only dict/str ones.

THE DEFECT THIS PINS (rounds/AUTONOMY/R12), and it is the eighth-ish recurrence of one shape.

`_capture_constraint_state` measures every field the backend adapter declares in `state_needed` by
walking `isinstance(val, str|dict|list|tuple)`. On vector, `state_needed` is
`["core_memory", "archival_memory"]` and BOTH attributes exist -- so `hasattr` passes -- but they are
`VectorStore` OBJECTS whose dict lives at `._store`. Every isinstance branch missed, `observed` stayed
empty, and the `if not observed: continue` dropped the record.

Measured consequence on the live corpus: the vector cell emitted ZERO constraint_state records over
629 prereq steps while raising 160 real capacity errors, against kv 259 and rec_sum 274. Every signal
reading `constrained_field`, `schema_limit` or `current_size` was silently blind on one of three
cells, and presented as "no opportunity" -- which is how A5's mechanism was nearly written off as
structurally unrecoverable.

The generic lesson, and the reason this test exists rather than a comment: a guard that cannot SEE a
backend reports the same thing as a backend with no opportunity. Those must never be indistinguishable.

These tests need the real backends, so they SKIP (never silently pass) without faiss.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
EVAL = REPO / "benchmarks/bfcl_v4/evaluator/memory_evaluator.py"
for p in (REPO, REPO / "benchmarks" / "bfcl_v4" / "harness"):
    sys.path.insert(0, str(p))


def _extracted():
    """The real function bodies, execed standalone.

    The evaluator module uses package-relative imports and cannot be imported on its own, so the
    functions under test are extracted by source. That keeps this a test of the SHIPPED code rather
    than of a copy.
    """
    src = EVAL.read_text()
    tree = ast.parse(src)
    want = {"_capture_constraint_state", "_backend_caps", "write_repair_adapter",
            "_parse_call_args"}
    segs = [ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in want]
    assert len(segs) == len(want), "a function under test was renamed"
    ns: dict = {"re": re, "sys": sys, "Path": pathlib.Path, "ast": ast, "Dict": dict, "Any": object}
    exec(compile("\n\n".join(segs), "<extracted>", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def ns():
    return _extracted()


def test_vector_caps_are_read_from_vectors_own_module(ns):
    """Not inherited from kv. They agree today; a divergence must surface here, not as a bad budget."""
    pytest.importorskip("faiss")
    caps = ns["_backend_caps"]()
    assert caps["VECTOR_CORE_SIZE"] == caps["CORE_SIZE"] == 7
    assert caps["VECTOR_ARCH_SIZE"] == caps["ARCH_SIZE"] == 50
    assert caps["VECTOR_ENTRY"] == caps["ENTRY"] == 300
    assert caps["VECTOR_ARCH_ENTRY"] == caps["ARCH_ENTRY"] == 2000


def _full_core(mod_name, api_name, add):
    from importlib import import_module
    api = import_module(
        f"bfcl_eval.eval_checker.multi_turn_eval.func_source_code.{mod_name}")
    inst = getattr(api, api_name)()
    inst._load_scenario({"model_result_dir": pathlib.Path(tempfile.mkdtemp()),
                         "test_id": f"{mod_name}_0-x-0", "scenario": "x", "long_context": False})
    for i in range(7):
        add(inst, i)
    return inst


def test_vector_at_its_core_cap_EMITS_constraint_state(ns):
    """The regression. Before the fix this returned None and the cell looked opportunity-free."""
    pytest.importorskip("faiss")
    inst = _full_core("memory_vector", "MemoryAPI_vector",
                      lambda o, i: o.core_memory_add(text=f"fact number {i} about something"))
    rec = ns["_capture_constraint_state"](
        ["core_memory_add(text='the eighth fact')"],
        ['{"error": "Memory size exceeds maximum size of 7 entries."}'],
        [inst], "vector")
    assert rec, "vector emitted NOTHING at a real capacity violation -- the blindness is back"
    r = rec[0]
    assert r["constrained_field"] == "core_memory"
    assert r["current_size"] == 7
    assert r["schema_limit"] == 7
    assert r["remaining_budget_if_extends"] == 0
    # The limit came from the STORE, not from a sibling backend's module constant.
    assert r["state"]["core_memory"]["store_caps"] == {"max_size": 7, "max_entry_length": 300}


def test_kv_at_its_core_cap_still_emits_unchanged(ns):
    """kv carries an ACCEPTED controller; its capture must be byte-equivalent to before."""
    pytest.importorskip("faiss")      # same env gate, so the two run or skip together
    inst = _full_core("memory_kv", "MemoryAPI_kv",
                      lambda o, i: o.core_memory_add(key=f"fact_{i}", value=f"v{i}"))
    rec = ns["_capture_constraint_state"](
        ["core_memory_add(key='fact_seven', value='v7')"],
        ['{"error": "Core memory is full. Please clear some entries."}'],
        [inst], "kv")
    assert rec
    r = rec[0]
    assert r["constrained_field"] == "core_memory"
    assert r["current_size"] == 7
    assert r["schema_limit"] == 7
    # kv stores are plain dicts, so there is no store-declared cap to record.
    assert r["state"]["core_memory"].get("store_caps") is None
