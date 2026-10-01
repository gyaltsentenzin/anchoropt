"""An already-grounded intervention may be offered CONDITIONED on the store it runs on -- optionally.

THE QUESTION R20 ASKS: can the loop learn WHERE an already-discovered controller applies?

R19 measured two mechanisms whose sign REVERSES across stores (relocate kv +4 / vector -6; reprompt
kv 0 / vector +4) and emitted 13 specs, every one a bare `{"declared_signal": ..., "params": {}}`. So
applicability was never something measurement could decide.

WHY THE VARIANT IS DECLARED AT GROUNDING. Core already turns one family's several groundings into one
arm each: `instantiate` builds an `InstantiatedAction` per grounding, `PolicyArm.label` derives from
`instantiated.label` (unique per variant), `arm_manifest` keys rows by that label, and
`ExternalEvaluation` looks results up by it. So a conditioned grounding is selectable end to end with
NO core change. A variant synthesized downstream of `build_arms` is NOT: it has no arm, so
`select_on_measurement` cannot address it, its result reads as `missing`, and the round reports
UNEVALUATED. That was measured before this design was chosen.

WHY IT IS OPT-IN. Conditioning multiplies this host's arm set 3.18x (17 -> 54) and every arm is a
paired GPU evaluation, so the default is off and the experiment sets ANCHOROPT_APPLICABILITY=backend.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "benchmarks" / "bfcl_v4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def rt_on(monkeypatch):
    """The adapter with conditioning ENABLED and corpus-wide scope."""
    monkeypatch.setenv("ANCHOROPT_APPLICABILITY", "backend")
    monkeypatch.delenv("ANCHOROPT_CELL", raising=False)
    import bfcl_runtime
    return bfcl_runtime


@pytest.fixture
def rt_off(monkeypatch):
    monkeypatch.delenv("ANCHOROPT_APPLICABILITY", raising=False)
    monkeypatch.delenv("ANCHOROPT_CELL", raising=False)
    import bfcl_runtime
    return bfcl_runtime


def _c2():
    spec = importlib.util.spec_from_file_location("c2t", REPO / "scripts/self_evolve_cycle2.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["c2t"] = m
    spec.loader.exec_module(m)
    return m


def _arms(rt, signal, locus, action):
    from anchoropt.learning.anchor_policy_opt import AnchorPolicyOpt, SearchSpaceProposal
    from anchoropt.runtime import IncisionPoint
    opt = AnchorPolicyOpt(runtime=rt, host=rt.HOST)
    proposal = SearchSpaceProposal(boundary=IncisionPoint(locus), signal=signal,
                                   action_set=(action,), diagnosis_case_ids=(), rationale="test")
    return opt.build_arms(proposal)[0]


# ---- the default must be untouched ---------------------------------------------------------------

def test_conditioning_is_off_by_default(rt_off):
    """Every other round must be byte-identical to before: 3.18x the arms is a real GPU bill."""
    from anchoropt.runtime import Action
    arms = _arms(rt_off, "no_tool_call_at_all", "post_generation_pre_exec", Action.REPROMPT)
    assert arms, "the unconditional family must still ground"
    assert not any("@backend=" in a.label for a in arms)


def test_a_single_cell_run_offers_nothing(monkeypatch):
    """Sharded by store, `backend == that_store` is constant: zero information, so no variant."""
    monkeypatch.setenv("ANCHOROPT_APPLICABILITY", "backend")
    monkeypatch.setenv("ANCHOROPT_CELL", "kv")
    import bfcl_runtime as rt
    g = rt.ground_reprompt("no_tool_call_at_all", "post_generation_pre_exec") or []
    assert g, "the unconditional groundings must survive"
    assert not [d for d in g if d.get("applicability")]


# ---- with conditioning on ------------------------------------------------------------------------

def test_the_unconditional_arm_is_always_kept(rt_on):
    """Conditioning is an OPTION for measurement to take or leave, never a replacement."""
    from anchoropt.runtime import Action
    arms = _arms(rt_on, "no_tool_call_at_all", "post_generation_pre_exec", Action.REPROMPT)
    plain = [a for a in arms if "@backend=" not in a.label]
    cond = [a for a in arms if "@backend=" in a.label]
    assert plain, "a mechanism that generalizes must keep its general arm"
    assert cond, "conditioning was enabled and produced nothing"


def test_every_arm_label_is_unique(rt_on):
    """A duplicate label is how one arm's measurement gets credited to another."""
    from anchoropt.runtime import Action
    seen = []
    for sig, loc, act in (("container_at_capacity", "post_execution", Action.REROUTE),
                          ("append_would_exceed_cap", "post_execution", Action.REROUTE),
                          ("no_tool_call_at_all", "post_generation_pre_exec", Action.REPROMPT),
                          ("clear_proposed_at_capacity", "post_generation_pre_exec", Action.SUPPRESS)):
        labels = [a.label for a in _arms(rt_on, sig, loc, act)]
        assert len(labels) == len(set(labels)), f"duplicate labels for {sig}: {labels}"
        seen += labels
    assert len(seen) == len(set(seen)), "labels collide across signals"
    assert not any(l.count("@backend=") > 1 for l in seen), "a nested condition was built"


def test_executability_is_respected_not_collapsed(rt_on):
    """A store-derived grounding is conditioned ONLY on its own store.

    `capacity_repair.BACKENDS` declares a relocate condition for kv and vector and NONE for rec_sum,
    so `relocate@backend=rec_sum` would be an arm whose action that store cannot perform. Applicability
    (where does it help) must not be allowed to invent executability (can it run at all).
    """
    g = rt_on.ground_capacity_relocations("container_at_capacity", "post_execution") or []
    cond = {d["applicability"]["value"] for d in g if d.get("applicability")}
    assert cond == {"kv", "vector"}, f"relocate conditioned on {cond}; rec_sum cannot relocate"


def test_a_store_agnostic_grounding_is_offered_on_every_store(rt_on):
    """reprompt declares backend None -- one instruction, executable anywhere -- so applicability is open."""
    g = rt_on.ground_reprompt("no_tool_call_at_all", "post_generation_pre_exec") or []
    for base in ("verify_before_answering", "search_other_container"):
        vals = {d["applicability"]["value"] for d in g
                if d.get("applicability") and d["variant"].startswith(base + "@")}
        assert vals == {"kv", "vector", "rec_sum"}, f"{base} conditioned on {vals}"


# ---- the spec the runner installs ----------------------------------------------------------------

def test_the_installed_predicate_carries_the_conjunct(rt_on):
    """The whole point: the predicate, not just the label, must confine the arm."""
    from anchoropt.runtime import Action
    c2 = _c2()
    arms = _arms(rt_on, "no_tool_call_at_all", "post_generation_pre_exec", Action.REPROMPT)
    cond = next(a for a in arms if a.label.endswith("@backend=vector"))
    plain = next(a for a in arms if "@backend=" not in a.label)

    assert c2._predicate_for(plain, rt_on) == {"declared_signal": "no_tool_call_at_all", "params": {}}
    assert c2._predicate_for(cond, rt_on) == {
        "all": [{"declared_signal": "no_tool_call_at_all", "params": {}},
                {"field": "backend", "op": "equals", "value": "vector"}]}


def test_conditioning_does_not_change_the_intervention(rt_on):
    """WHERE a mechanism applies is not WHAT it does; changing both would confound them."""
    from anchoropt.runtime import Action
    arms = _arms(rt_on, "no_tool_call_at_all", "post_generation_pre_exec", Action.REPROMPT)
    plain = next(a for a in arms if a.label.endswith("reprompt:search_other_container"))
    cond = next(a for a in arms if a.label.endswith("search_other_container@backend=kv"))
    assert dict(cond.eta) == dict(plain.eta), "eta must be identical"
    assert cond.action == plain.action and cond.signal == plain.signal
    assert cond.instantiated.operator == plain.instantiated.operator
    assert cond.capability_id == plain.capability_id


def test_the_spec_records_applicability_for_the_reader(rt_on):
    from anchoropt.runtime import Action
    c2 = _c2()
    arms = _arms(rt_on, "no_tool_call_at_all", "post_generation_pre_exec", Action.REPROMPT)
    cond = next(a for a in arms if a.label.endswith("@backend=rec_sum"))
    assert c2._applicability_of(cond) == {"field": "backend", "value": "rec_sum"}
    assert c2._applicability_of(next(a for a in arms if "@backend=" not in a.label)) == {}


def test_the_rule_names_no_store_and_no_mechanism():
    """A generic composition rule, not a backend-to-mechanism table."""
    src = (REPO / "benchmarks/bfcl_v4/bfcl_runtime.py").read_text()
    start = src.index("def with_applicability_variants")
    body = src[start:src.index("\ndef ", start + 10)]
    code = "\n".join(l for l in body.splitlines() if not l.strip().startswith("#"))
    code = code.split('"""')[0] + "".join(code.split('"""')[2:])
    for bad in ("'kv'", '"kv"', "'vector'", '"vector"', "'rec_sum'", '"rec_sum"',
                "relocate", "reprompt", "container_at_capacity"):
        assert bad not in code, f"{bad!r} must not appear in the composition rule"
