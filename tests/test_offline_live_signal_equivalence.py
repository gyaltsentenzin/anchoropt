"""The cluster's signal copy and the adapter's original must agree, on the REAL emitted spec.

The cluster checkout has no `bfcl_signals`, so a controller naming a declared signal had nothing to
evaluate there. Two ways to fix that, and only one is safe:

  * hand-translate each controller into a field/op atom the installer already understood. TRIED AND
    REJECTED: the first attempt wrote `proposes_clear AND container_full` and silently dropped the
    `error_kind == "no_capacity"` alternative the real signal carries. A translation re-derived per
    controller is a place for exactly that drift, and it makes one failure mechanism a special case.
  * copy the host's own implementations verbatim and evaluate them by name. That is what ships, and
    this file is what keeps the copy honest.

The `error_kind == "no_capacity"` row is the one this file exists for.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

CLUSTER_COPY = REPO / "patches" / "bv" / "bfcl_declared_signals.py"
GATE = "post_generation_pre_exec"

# States that separate the two implementations if they ever diverge. The third row is the one the
# rejected hand-translation got wrong.
CASES = [
    ("clear proposed, a limit was observed earlier",
     {"proposes_clear": True, "container_full": True}),
    ("clear proposed, no limit observed",
     {"proposes_clear": True, "container_full": False}),
    ("clear proposed, container_full FALSE but error_kind no_capacity",
     {"proposes_clear": True, "container_full": False, "error_kind": "no_capacity"}),
    ("ordinary write while at capacity",
     {"proposes_clear": False, "container_full": True}),
    ("read while at capacity",
     {"proposes_read": True, "proposes_clear": False, "container_full": True}),
    ("nothing set at all",
     {}),
]


def _cluster():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_bv_signals", CLUSTER_COPY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _adapter():
    import bfcl_signals
    return bfcl_signals


def test_the_cluster_copy_exists_and_is_tracked():
    """A cluster-only patch is not reproducible. It must live on the branch."""
    assert CLUSTER_COPY.exists(), "the cluster signal module is not tracked in patches/bv/"


@pytest.mark.parametrize("label,state", CASES, ids=[c[0][:40] for c in CASES])
def test_offline_and_cluster_agree_on_the_destructive_clear_signal(label, state):
    a = _adapter().SIGNALS["clear_proposed_at_capacity"]
    b = _cluster().SIGNALS["clear_proposed_at_capacity"]
    st = dict(state, boundary=GATE)
    assert a(st, {}) == b(st, {}), f"the two implementations disagree on: {label}"


def test_the_error_kind_ALTERNATIVE_is_carried_not_dropped():
    """THE ROW THIS FILE EXISTS FOR. `container_full` False must still fire on no_capacity."""
    st = {"proposes_clear": True, "container_full": False, "error_kind": "no_capacity",
          "boundary": GATE}
    for name, mod in (("adapter", _adapter()), ("cluster", _cluster())):
        assert mod.SIGNALS["clear_proposed_at_capacity"](st, {}) is True, (
            f"{name} dropped the error_kind alternative -- this is the hand-translation defect")


def test_every_stdlib_signal_body_is_VERBATIM():
    """Not 'equivalent': identical. A reimplementation is a second definition that can drift."""
    import re

    def body(src, name):
        m = re.search(rf"\ndef {name}\(state.*?(?=\ndef |\nSIGNALS|\Z)", src, re.S)
        return m.group(0).strip() if m else None

    a = (REPO / "benchmarks" / "bfcl_v4" / "bfcl_signals.py").read_text()
    b = CLUSTER_COPY.read_text()
    checked = 0
    for name in ("container_at_capacity", "identifier_not_found", "duplicate_identifier",
                 "no_tool_call_at_all", "container_slots_exhausted", "append_would_exceed_cap",
                 "clear_proposed_at_capacity", "retrieval_similarity_below_threshold"):
        x, y = body(a, name), body(b, name)
        assert x and y, name
        assert x == y, f"{name} is not verbatim in the cluster copy"
        checked += 1
    assert checked == 8


def test_the_one_unportable_signal_RAISES_rather_than_answering_False():
    """A signal that quietly answers False is a controller that quietly never fires."""
    mod = _cluster()
    with pytest.raises(Exception) as ei:
        mod.SIGNALS["no_informative_result"]({"boundary": "post_execution"}, {})
    assert "policy_tree" in str(ei.value), str(ei.value)


def test_an_unknown_signal_name_RAISES():
    mod = _cluster()
    with pytest.raises(Exception):
        mod.evaluate_signal("not_a_signal", {}, {})


def test_the_REAL_emitted_spec_names_a_signal_the_cluster_implements():
    """The end-to-end link: whatever the round emitted must be evaluable where it will run."""
    emitted = sorted(REPO.glob("results/h0_*_focus/controllers.json")) + \
              sorted(REPO.glob("results/h0v/controllers.json"))
    if not emitted:
        pytest.skip("no emitted spec on disk")
    known = set(_cluster().SIGNALS)
    for path in emitted:
        for spec in json.loads(path.read_text()):
            pred = spec.get("predicate") or {}
            if "declared_signal" not in pred:
                continue
            assert pred["declared_signal"] in known, (
                f"{path.name} names {pred['declared_signal']!r}, which the cluster cannot evaluate")
