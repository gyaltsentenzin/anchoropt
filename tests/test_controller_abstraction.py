"""Controller = (locus, phi, action, eta). phi owns WHEN; a primitive only knows HOW."""

import pathlib

from anchoropt.learning.controller import Controller, firing_set, run_controller
from anchoropt.learning.signal_grammar import Atom, Conjunction


def ctl(theta, primitive="merge"):
    return Controller(locus="post_execution",
                      phi=Conjunction((Atom("flag", "falsy"), Atom("score", "lt", theta))),
                      action="reroute",
                      eta={"primitive": primitive, "destination": "other"})


STATES = [{"flag": False, "score": s} for s in (0.05, 0.12, 0.2, 0.25, 0.35, 0.5)]


def test_two_thresholds_fire_the_SAME_primitive_on_DIFFERENT_states():
    """THE property under test. It failed before this abstraction existed: the host's read-merge path
    hard-coded its own arming condition, so the executor decided WHEN and two predicates with different
    thresholds fired on identical states -- which makes the abstraction untestable, because the thing
    being tested was decided elsewhere."""
    a, b = firing_set(ctl(0.30), STATES), firing_set(ctl(0.15), STATES)
    assert a != b, "different thresholds must select different states"
    assert b < a, "a lower threshold must be strictly more selective here"
    assert len(a ^ b) == 2


def test_the_primitive_is_invoked_with_no_knowledge_of_why():
    seen = []

    def merge(state, destination, top_k=None):
        seen.append(dict(state))
        return {"fired": True, "destination": destination}

    res = run_controller(ctl(0.30), STATES, {"merge": merge})
    assert sum(1 for r in res if r.fired) == 4
    assert all(r.effect["fired"] for r in res if r.fired)
    # the primitive received STATE and PARAMETERS only -- no threshold, no signal, no reason
    assert seen and "threshold" not in seen[0] and "phi" not in seen[0]


def test_a_missing_primitive_is_REPORTED_not_simulated():
    """An unimplemented HOW must not look like a controller that chose not to act."""
    res = run_controller(ctl(0.30), STATES, {})
    fired = [r for r in res if r.fired]
    assert fired and all(r.effect is None for r in fired)
    assert all("no primitive named" in r.reason for r in fired)


def test_dry_run_separates_WHEN_from_HOW():
    res = run_controller(ctl(0.30), STATES, {}, dry_run=True)
    assert sum(1 for r in res if r.fired) == 4
    assert all("primitive not invoked" in r.reason for r in res if r.fired)


def test_core_controller_module_names_no_benchmark_concept():
    """The portability criterion: the same core must run on a different benchmark unchanged."""
    src = (pathlib.Path(__file__).resolve().parent.parent /
           "anchoropt" / "learning" / "controller.py").read_text().lower()
    for word in ("archival", "core_memory", "kv", "rec_sum", "bfcl", "container", "similarity",
                 "retrieve", "0.3", "threshold"):
        assert word not in src, f"core controller names a benchmark concept: {word!r}"


def test_the_same_core_runs_against_a_DIFFERENT_adapter_surface():
    """Portability, demonstrated rather than asserted: different observables, different primitive,
    same Controller and same run_controller."""
    states = [{"latency_ms": v, "retried": False} for v in (10, 250, 900)]
    c = Controller(locus="after_call",
                   phi=Atom("latency_ms", "gt", 200),
                   action="reprompt",
                   eta={"primitive": "escalate", "channel": "slow_path"})
    calls = []

    def escalate(state, channel):
        calls.append(channel)
        return {"fired": True, "channel": channel}

    res = run_controller(c, states, {"escalate": escalate})
    assert sum(1 for r in res if r.fired) == 2
    assert calls == ["slow_path", "slow_path"]
