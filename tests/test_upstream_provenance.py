"""A failing case must be able to reach the decision that caused it, even in another episode.

THE STRUCTURAL PROBLEM. A query episode reads a store an earlier storage episode built. If the fact
it needed was destroyed during construction, the query's own trajectory shows only a read that found
nothing -- the consequential decision is not in that episode at all. Attribution over one episode can
see that a read failed but never that a write destroyed what it wanted, so an entire class of
consequential decision is unreachable from downstream evidence.

This is the general provenance mechanism, not a memory-benchmark special case: any host whose
episodes share state through a declared dependency graph has the same shape, and a host that declares
no graph gets no upstream block and behaves exactly as before.

Read from the corpus's declared `depends_on`, never inferred from id strings -- a naming convention is
not a dependency, and inferring one would invent provenance the corpus does not assert.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "integrations" / "claude_roles"))


def _m():
    import self_evolve_cycle2
    return self_evolve_cycle2


CORPUS = [
    {"id": "q1", "depends_on": ["p2"]},
    {"id": "p2", "depends_on": ["p1"]},
    {"id": "p1", "depends_on": []},
    {"id": "lonely", "depends_on": []},
]

STEPS = {
    "q1": [{"status": "executed", "decoded": ["core_memory_retrieve(query='x')"],
            "tool_results": ["{'ranked_results': []}"]}],
    "p2": [{"status": "executed", "decoded": ["core_memory_clear()"], "tool_results": ["{}"]}],
    "p1": [{"status": "executed", "decoded": ["core_memory_add(text='the fact')"],
            "tool_results": ["{'id': 0}"]}],
}


def _graph(tmp_path):
    f = tmp_path / "cases.json"
    f.write_text(json.dumps(CORPUS))
    return _m().dependency_graph(f)


def test_the_graph_is_read_from_the_corpus(tmp_path):
    g = _graph(tmp_path)
    assert g["q1"] == ["p2"] and g["p1"] == []


def test_a_missing_or_unreadable_cases_file_yields_no_graph(tmp_path):
    m = _m()
    assert m.dependency_graph(None) == {}
    assert m.dependency_graph(tmp_path / "nope.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert m.dependency_graph(bad) == {}


def test_the_closure_is_transitive_and_nearest_first(tmp_path):
    """q1 depends on p2 which depends on p1: the cause may be two hops upstream."""
    m = _m()
    assert m.upstream_of("q1", _graph(tmp_path)) == ["p2", "p1"]


def test_a_cycle_terminates(tmp_path):
    m = _m()
    g = {"a": ["b"], "b": ["a"]}
    assert set(m.upstream_of("a", g)) == {"b", "a"} - {"a"} or "b" in m.upstream_of("a", g)


def test_the_failing_case_carries_the_upstream_trajectory(tmp_path):
    """The point: the destructive call is visible to attribution though it is in another episode."""
    m = _m()
    facts = m.case_facts("q1", STEPS["q1"], "vector", "query")
    assert "upstream_episodes" not in facts, "one-episode facts must not invent provenance"
    linked = m.with_upstream(facts, "q1", _graph(tmp_path), STEPS)
    blocks = linked["upstream_episodes"]
    assert [b["case_id"] for b in blocks] == ["p2", "p1"]
    calls = [c for b in blocks for s in b["trajectory"] for c in (s["tool_calls"] or [])]
    assert any("core_memory_clear" in c for c in calls), \
        "the destructive upstream call must be reachable from the failing query"


def test_upstream_evidence_is_LABELLED_not_merged(tmp_path):
    """An upstream write must not be mistakable for something the failing episode did."""
    m = _m()
    linked = m.with_upstream(m.case_facts("q1", STEPS["q1"], "vector", "query"),
                             "q1", _graph(tmp_path), STEPS)
    own = [c for s in linked["trajectory"] for c in (s["tool_calls"] or [])]
    assert not any("core_memory_clear" in c for c in own), "upstream leaked into the episode's own trace"
    for b in linked["upstream_episodes"]:
        assert b["case_id"] and b["relation"], "each block must name its case and its relation"
    assert "upstream_note" in linked


def test_a_case_with_no_dependencies_is_unchanged(tmp_path):
    m = _m()
    facts = m.case_facts("lonely", STEPS["q1"], "vector", "query")
    assert m.with_upstream(facts, "lonely", _graph(tmp_path), STEPS) == facts


def test_no_graph_means_no_change(tmp_path):
    """A host declaring no dependency graph must behave exactly as before."""
    m = _m()
    facts = m.case_facts("q1", STEPS["q1"], "vector", "query")
    assert m.with_upstream(facts, "q1", {}, STEPS) == facts


def test_the_upstream_block_is_bounded(tmp_path):
    """A full closure is 10 episodes of ~18 steps on the real corpus; the point is the decision."""
    m = _m()
    corpus = [{"id": "q", "depends_on": [f"p{i}" for i in range(9)]}] + \
             [{"id": f"p{i}", "depends_on": []} for i in range(9)]
    f = tmp_path / "big.json"
    f.write_text(json.dumps(corpus))
    steps = {f"p{i}": [{"status": "executed", "decoded": [f"core_memory_add(text='{i}')"],
                        "tool_results": ["{}"]}] for i in range(9)}
    linked = m.with_upstream(m.case_facts("q", [], "vector", "query"), "q",
                             m.dependency_graph(f), steps, max_episodes=3)
    assert len(linked["upstream_episodes"]) == 3


def test_the_driver_exposes_the_flag():
    """Provenance is opt-in via the shard file the adapter already ships."""
    src = (REPO / "scripts" / "self_evolve_cycle2.py").read_text()
    assert '"--cases"' in src
    assert "with_upstream(case_facts(" in src, "the attribution path does not attach provenance"
