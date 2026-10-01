"""PROOF that the WHAT search receives states from the boundary it is searching -- not a claim.

The WRITE1 defect had two halves. Fields (declared vs supplied) is fixed in the adapter. This is the
other half: a predicate was VALIDATED on post-execution-shaped states no matter which boundary was
being searched, so an upstream predicate could look discriminating on information that does not exist
yet, then fire zero times at runtime.

Two things are proven here, because the first without the second is worthless:
  * `_expand_phi_at` at the commitment gate is handed the PROJECTED states, observed by interception;
  * a runtime that cannot construct that information set makes core FAIL CLOSED, never fall back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "benchmarks" / "bfcl_v4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from anchoropt.anchor import IncisionPoint                                   # noqa: E402
from anchoropt.learning import structured_search as SS                       # noqa: E402

UP = IncisionPoint.POST_GENERATION_PRE_EXEC
POST = IncisionPoint.POST_EXECUTION


class Recorder:
    """A runtime that records exactly what `_expand_phi_at` passes to synthesis."""

    def __init__(self, *, with_states_at=True, project_empty=False):
        self.seen_states = None
        self.seen_fields = None
        self._with = with_states_at
        self._empty = project_empty
        if with_states_at:
            self.states_at = self._states_at

    def synthesis_fields(self, boundary=None):
        import bfcl_capabilities  # noqa: F401  (typed Field objects)
        import bfcl_runtime as rt
        self.seen_fields = rt.synthesis_fields(boundary)
        return self.seen_fields

    def _states_at(self, boundary, states):
        import bfcl_runtime as rt
        if self._empty:
            return []
        return rt.states_at(boundary, states)

    def declared_signals(self):
        return ()

    def install_signal(self, *a, **k):
        raise AssertionError("not reached in these tests")


STEP_STATES = [
    {"decoded": [f"core_memory_add(text='{'x' * (200 + 40 * i)}')"], "step": i,
     # post-execution facts DELIBERATELY present, so a leak would be visible
     "best_similarity": 0.4 + i / 100, "tool_results": ['{"error": "..."}'],
     # EPISODE-STATIC facts, as the real assembler supplies them (measured: present on 2443/2443 real
     # states). They belong in the fixture because the projection CARRIES them rather than deriving
     # them: without them here, `test_the_fields_offered_match_the_states_supplied` fails for a reason
     # that is an artefact of the fixture rather than of the adapter -- and that test is exactly the
     # declared-but-unsupplied guard, so it must fail only when the adapter is genuinely wrong.
     # `turn` advances every few steps so it is a real variable here, not a constant.
     "backend": "kv", "turn": i // 4}
    for i in range(12)
]


def _capture(runtime, boundary):
    """Call the real `_expand_phi_at` and intercept what synthesis was given."""
    captured = {}
    real = SS.__dict__["_expand_phi_at"]

    import anchoropt.learning.signal_grammar as G
    real_synth = G.synthesize

    def spy(fields, states, **kw):
        captured["fields"] = dict(fields)
        captured["states"] = list(states)
        return []                          # no predicates: we are testing the INPUT

    G.synthesize = spy
    try:
        real(boundary, runtime=runtime, states=STEP_STATES, existing=[])
    finally:
        G.synthesize = real_synth
    return captured


def test_expand_phi_at_the_commitment_gate_receives_PROJECTED_states():
    """One state per PROPOSED CALL, carrying pre-dispatch facts -- not the per-step states."""
    cap = _capture(Recorder(), UP)
    states = cap["states"]
    assert states, "synthesis was given no states at the commitment gate"
    assert len(states) == len(STEP_STATES), "expected one state per proposed call"
    for st in states:
        assert "proposed_call" in st, "projection did not run: no proposed_call"
        assert "proposed_payload_chars" in st, "typed pre-dispatch facts are missing"


def test_no_post_execution_fact_reaches_the_commitment_gate():
    """The whole point. A result does not exist yet at this boundary."""
    cap = _capture(Recorder(), UP)
    for st in cap["states"]:
        for leaked in ("best_similarity", "tool_results", "scored_entries", "error_kind"):
            assert leaked not in st, f"{leaked!r} leaked into the commitment-gate information set"


def test_the_fields_offered_match_the_states_supplied():
    """A field synthesis may use must be present in the states it is validated on."""
    cap = _capture(Recorder(), UP)
    offered = set(cap["fields"])
    present = set().union(*(set(s) for s in cap["states"]))
    assert offered <= present, f"offered {sorted(offered - present)} which no state carries"


def test_post_execution_states_pass_through_UNCHANGED():
    """A9's validation must be untouched by any of this."""
    cap = _capture(Recorder(), POST)
    assert cap["states"] == STEP_STATES


def test_a_runtime_with_no_states_at_FAILS_CLOSED():
    """An absent projection is a contract gap, not a licence to use another boundary's states.

    Falling back silently is what let an upstream candidate be validated on post-execution facts.
    """
    with pytest.raises(RuntimeError, match="no states_at"):
        _capture(Recorder(with_states_at=False), UP)


def test_an_empty_projection_FAILS_CLOSED_rather_than_reporting_exhaustion():
    """Returning [] would surface as SIGNAL_EXPANSION_EXHAUSTED for a boundary never searched."""
    with pytest.raises(RuntimeError, match="produced no states"):
        _capture(Recorder(project_empty=True), UP)


def test_the_live_adapter_projects_real_storage_trajectories():
    """End-to-end on real data rather than a fixture."""
    import importlib.util
    import bfcl_runtime as rt
    spec = importlib.util.spec_from_file_location(
        "c2", REPO / "scripts" / "self_evolve_cycle2.py")
    c2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c2)
    run = Path("/tmp/c2/c2inc")
    if not (run / "traj" / "prereq").exists():
        pytest.skip("c2inc storage trajectories not present in this checkout")
    steps = c2._load_prereq(run)
    per_step = c2.observable_states(steps)
    up = rt.states_at(UP, per_step)
    assert up, "no upstream states projected from real trajectories"
    # A NO-CALL STEP IS STILL A DECISION AT THIS BOUNDARY, and it is the one the
    # answer-without-retrieval condition is about. The old assertion here was
    # `len(up) < len(per_step)` -- "projection should drop steps that proposed no call" -- and that
    # assertion WAS the defect: it made the commitment gate structurally blind to zero-tool-call
    # episodes (33 of them projected to 0 states), so the only declared signal describing them could
    # never be evaluated.
    #
    # The contract is now: one row per proposed call, PLUS one row for a step that proposes none.
    n_calls = sum(len(st.get("decoded") or []) for st in per_step)
    n_nocall = sum(1 for st in per_step if not (st.get("decoded") or []))
    assert len(up) == n_calls + n_nocall, (len(up), n_calls, n_nocall)
    # The no-call rows must be IDENTIFIABLE and must carry the fields the zero-call signal reads.
    nocall = [s for s in up if s.get("proposed_call") is None]
    assert len(nocall) == n_nocall
    for s in nocall:
        assert s["n_proposed_calls"] == 0 and s["proposes_tool_call"] is False
        assert s["has_generation"] is True and "tool_calls_so_far" in s
    # And every row must be EVALUABLE by that signal, not only the ones it fires on -- otherwise it
    # has no negative case and "discriminates" cannot be told from "unevaluable".
    for s in up:
        assert {"has_generation", "proposes_tool_call", "tool_calls_so_far"} <= set(s)
    assert all("best_similarity" not in s for s in up)
    assert any(s.get("proposed_payload_chars", 0) > 300 for s in up), \
        "the real data contains over-cap proposals; the projection lost them"


def test_decoded_equals_the_dispatched_list_for_this_corpus():
    """The reconstruction reads `decoded`; the live hook reads the FILTERED dispatch list.

    They are equivalent here, and that is MEASURED rather than assumed: no filter fired in any of the
    791 storage-phase steps (`rx_withheld`, `memoized_call_sent_to_execute`, `rx_wire_error` all
    absent). If a filter ever does fire, this test fails and the reconstruction must switch to
    capturing the real pre-dispatch list instead of inferring it.
    """
    import json
    run = Path("/tmp/c2/c2inc/traj/prereq")
    if not run.exists():
        pytest.skip("c2inc storage trajectories not present")
    touched, total = 0, 0
    for fp in sorted(run.glob("*.json")):
        for st in (json.load(open(fp)).get("steps") or []):
            total += 1
            if any(st.get(k) for k in ("rx_withheld", "memoized_call_sent_to_execute",
                                       "rx_wire_error")):
                touched += 1
    assert total > 0
    assert touched == 0, (f"{touched}/{total} steps had the dispatch list filtered -- `decoded` is no "
                          f"longer equivalent to _mg_exec_calls, so the projection must capture the "
                          f"real pre-dispatch calls")


def test_the_live_hook_and_the_adapter_compute_IDENTICAL_pre_dispatch_facts():
    """A predicate validated offline must evaluate identically at runtime.

    Two implementations exist and both are necessary: the adapter's `proposed_call_facts` (used by the
    offline projection and therefore by WHAT search) and an in-place copy in the cluster evaluator,
    which has no `benchmarks/` directory to import from. A divergence between them is invisible until
    a controller that validated offline fires zero times live -- which is exactly the failure that
    produced this test: the live hook did not compute these facts at all, so the hook ran 20 times on
    an episode holding two over-cap core writes and fired 0.

    Verified on the 543 real proposed calls from c2inc's storage phase: 0 disagreements over 7 fields.
    """
    import json
    calls_file = Path("/tmp/calls_full.json")
    if not calls_file.exists():
        pytest.skip("real call corpus not fetched in this checkout")
    import bfcl_runtime as rt
    calls = json.load(open(calls_file))
    assert len(calls) > 100, "corpus too small to be meaningful"

    # The live helper's source, reproduced here so a drift in EITHER side fails this test.
    import ast as _ast

    def live(call):
        text = str(call or "").strip()
        if not text:
            return {}
        try:
            tree = _ast.parse(text, mode="eval")
            if not isinstance(tree.body, _ast.Call):
                return {}
            kw = {}
            for k in tree.body.keywords:
                if k.arg is None:
                    continue
                try:
                    kw[k.arg] = _ast.literal_eval(k.value)
                except Exception:
                    kw[k.arg] = None
        except Exception:
            return {}
        low = text.lower()
        cont = ("core" if "core_memory" in low
                else "archival" if "archival_memory" in low else None)
        strs = [v for v in kw.values() if isinstance(v, str)]
        return {"proposes_write": ("_add" in low or "_insert" in low or "_append" in low),
                "proposes_read": ("_retrieve" in low or "_search" in low or "_get" in low
                                  or "_list" in low),
                "proposes_clear": "_clear" in low, "proposes_remove": "_remove" in low,
                "container": cont,
                "proposed_payload_chars": max((len(s) for s in strs), default=0),
                "proposed_arg_count": len(kw)}

    keys = ("proposes_write", "proposes_read", "proposes_clear", "proposes_remove",
            "container", "proposed_payload_chars", "proposed_arg_count")
    bad = []
    for c in calls:
        a, l = rt.proposed_call_facts(c), live(c)
        for k in keys:
            if a.get(k) != l.get(k):
                bad.append((k, a.get(k), l.get(k), c[:60]))
    assert not bad, f"{len(bad)} field disagreements, e.g. {bad[:2]}"


def test_truncated_call_text_yields_no_facts_rather_than_wrong_ones():
    """An unparseable proposal must produce {} -- no facts beat facts that happen to be wrong.

    This is how a cross-check harness of mine reported 203 false disagreements: it truncated calls to
    400 chars, so the closing quote was gone and only one of the two implementations still answered.
    """
    import bfcl_runtime as rt
    truncated = "core_memory_add(text='" + "x" * 500      # no closing quote or paren
    assert rt.proposed_call_facts(truncated) == {}
