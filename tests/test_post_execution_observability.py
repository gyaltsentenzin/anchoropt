"""The post-execution boundary must OFFER the facts its own states carry, and say so truthfully.

WHY THIS FILE EXISTS (R19). `_HOOK_STATE_FIELDS[post_execution]` listed 5 field names while
`observable_states` wrote 10 into every state -- measured 2443/2443 on frozen H0, three backends, both
phases. That one stale frozenset feeds TWO consumers, so the omission did not merely hide the fields:

    synthesis_fields()        intersects declared x supplied  -> the field is never OFFERED
    can_fire_at_boundary()    via boundary_state_shape        -> REJECTS the candidate outright,
                                                                 saying "would fire 0 times"

Measured verdicts before the fix, beside the firing counts those same predicates achieved on the same
real states:

    error_kind == 'blob_would_overflow'   rejected, fires 301/2443
    error_kind == 'entry_too_long'        rejected, fires 155/2443
    error_kind == 'no_capacity'           rejected, fires 124/2443

The reason was FALSE, not merely unhelpful, and a false rejection is worse than a missing field: it
leaves a confident explanation in the search record, so the loop reports a closed question. Six of the
nine historical mechanisms act at this boundary and `error_kind` is the field naming which refusal
occurred -- the trigger class for all of them.

These tests pin the CONTRACT (a supplied field is declared, and a declared field is offered), not a
field count, because the next field added to the assembler would reproduce the defect exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "benchmarks" / "bfcl_v4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

POST = "post_execution"
GATE = "post_generation_pre_exec"

#: Fields the assembler writes into EVERY post-execution state, verified against the real corpus.
#: `error_kind` is the load-bearing one: it names WHICH refusal the store returned.
_SUPPLIED_AT_POST_EXEC = ("error_kind", "container", "proposes_write", "result", "tool_calls_so_far")


def _rt():
    import bfcl_runtime as rt
    return rt


def test_the_fields_every_real_state_carries_are_declared_as_supplied():
    """The regression. Supplied-but-undeclared is invisible to synthesis AND falsely rejected."""
    supplied = set(_rt().hook_state_fields(POST))
    missing = [f for f in _SUPPLIED_AT_POST_EXEC if f not in supplied]
    assert not missing, (
        f"{missing} are written into every post-execution state but absent from "
        f"_HOOK_STATE_FIELDS, so synthesis cannot see them and can_fire_at_boundary "
        f"rejects any predicate over them as 'would fire 0 times'")


def test_error_kind_is_offered_to_synthesis_at_post_execution():
    """Without this, no capacity or existence mechanism is expressible where it actually acts."""
    assert "error_kind" in _rt().synthesis_fields(POST)


def test_the_capacity_and_existence_signals_are_SATISFIABLE_here():
    """A declared signal whose fields the boundary does not supply can only ever answer False.

    Two are expected to remain unsatisfiable and are named explicitly rather than tolerated silently:
    their missing fields are genuinely absent from states, not merely undeclared.
    """
    rt = _rt()
    supplied = set(rt.hook_state_fields(POST))
    unsatisfiable = set()
    for sig in rt.declared_signals():
        at = {str(getattr(b, "value", b)) for b in rt.signal_boundaries(sig)}
        if POST in at and not set(rt.signal_fields(sig)) <= supplied:
            unsatisfiable.add(sig)
    assert unsatisfiable == {"container_slots_exhausted", "duplicate_identifier"}, (
        f"unexpected satisfiability set {sorted(unsatisfiable)}; these two are the known real supply "
        f"gaps (is_relocation_target / identifier_present are absent from states, not undeclared)")


def test_a_predicate_over_error_kind_is_ACCEPTED_with_its_true_firing_count():
    """End to end through the gate that used to lie. Verdict True, and the count is the real one."""
    import importlib.util
    import types

    rt = _rt()
    spec = importlib.util.spec_from_file_location("c2_obs", REPO / "scripts" / "self_evolve_cycle2.py")
    c2 = importlib.util.module_from_spec(spec)
    sys.modules["c2_obs"] = c2
    spec.loader.exec_module(c2)
    from anchoropt.learning.signal_grammar import Atom

    # States of the shape the assembler produces: error_kind ALWAYS present, None where clean.
    states = ([{"error_kind": "no_capacity", "container": "core", "proposes_write": True,
                "result": "{}", "tool_calls_so_far": 3, "backend": "kv", "turn": 1}] * 7
              + [{"error_kind": None, "container": "core", "proposes_write": True,
                  "result": "{}", "tool_calls_so_far": 1, "backend": "kv", "turn": 0}] * 13)
    arm = types.SimpleNamespace(boundary=types.SimpleNamespace(value=POST))
    ok, why = c2.can_fire_at_boundary(arm, Atom("error_kind", "equals", "no_capacity"),
                                      runtime=rt, states=states)
    assert ok, f"a predicate over a supplied field was refused: {why}"
    assert "7/20" in why, f"the accepted verdict must report the true firing count, got: {why}"


def test_no_post_execution_fact_was_added_to_the_COMMITMENT_GATE():
    """Widening post-execution must not widen the gate: the result does not exist there yet."""
    gate = set(_rt().hook_state_fields(GATE))
    for leaked in ("error_kind", "result", "best_similarity", "scored_entries"):
        assert leaked not in gate, f"{leaked} reached the pre-dispatch information set"


def test_backend_is_observable_but_scenario_is_NOT():
    """Applicability must be learnable from structure, never memorised from a use-case label."""
    rt = _rt()
    for locus in (GATE, POST):
        offered = set(rt.synthesis_fields(locus))
        assert "backend" in offered, f"backend not observable at {locus}"
        assert not [f for f in offered if "scenario" in f or "topic" in f], (
            f"a scenario-identity field reached phi at {locus}: it would memorise which use case is "
            f"favourable and generalise to nothing")


@pytest.mark.parametrize("backend,single_blob,second_container,unique_keys", [
    ("kv", False, True, True),
    ("vector", False, True, False),
    ("rec_sum", True, False, False),
])
def test_backend_facts_are_DERIVED_and_stay_available_to_the_executor(
        backend, single_blob, second_container, unique_keys):
    """Structural facts are not phi fields, but the executor still needs them for feasibility.

    Derived from the adapter's declared constraints, so adding a backend does not mean editing a
    per-backend table here.
    """
    f = _rt().backend_facts(backend)
    assert f["is_single_blob"] is single_blob
    assert f["has_second_container"] is second_container
    assert f["enforces_unique_keys"] is unique_keys


def test_an_unknown_backend_yields_NO_structural_claim():
    """Absent beats guessed: a predicate over an absent field does not fire."""
    assert _rt().backend_facts("not_a_store") == {}
