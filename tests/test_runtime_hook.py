"""The generic runtime hook: one wiring per locus, then every future controller reaches it."""

import pathlib

import pytest

from anchoropt import runtime_hook as H
from anchoropt.learning.controller import Controller
from anchoropt.learning.signal_grammar import Atom


@pytest.fixture(autouse=True)
def clean():
    H.reset()
    yield
    H.reset()


def ctl(theta, name="c"):
    return Controller(locus="L", phi=Atom("score", "lt", theta), action="reroute",
                      eta={"primitive": "merge", "destination": "other"})


def test_with_nothing_installed_the_host_decision_passes_through_UNCHANGED():
    """Adoption must be a no-op until a controller exists, or wiring the hook would itself change
    every existing measurement -- which is not something a host can be asked to accept."""
    for default in (True, False):
        d = H.decide("L", {"score": 0.1}, host_default=default)
        assert d.proceed is default and d.source == "host" and d.controller is None


def test_an_installed_phi_decides_and_overrides_the_host_default():
    H.install("L", ctl(0.3))
    assert H.decide("L", {"score": 0.1}, host_default=False).proceed is True
    assert H.decide("L", {"score": 0.9}, host_default=True).proceed is False


def test_two_thresholds_produce_DIFFERENT_decisions_at_the_SAME_locus():
    """The property the hook exists for. Without it the host's own constant decides and two
    parameterisations are indistinguishable."""
    states = [{"score": s} for s in (0.05, 0.2, 0.5)]
    H.install("L", ctl(0.3))
    a = [H.decide("L", s, host_default=False).proceed for s in states]
    H.reset()
    H.install("L", ctl(0.1))
    b = [H.decide("L", s, host_default=False).proceed for s in states]
    assert a == [True, True, False]
    assert b == [True, False, False]
    assert a != b


def test_the_hook_carries_eta_so_the_host_can_execute_HOW():
    H.install("L", ctl(0.3))
    d = H.decide("L", {"score": 0.1}, host_default=False)
    assert d.eta["primitive"] == "merge" and d.eta["destination"] == "other"


def test_a_RAISING_predicate_is_not_a_trigger_and_is_recorded():
    class Bad:
        name, eta = "bad", {}
        def fires_on(self, state):
            raise ValueError("boom")

    H.install("L", Bad())
    assert H.decide("L", {}, host_default=True).proceed is False
    assert any("boom" in e for e in H.errors())


def test_controllers_at_OTHER_loci_do_not_fire_here():
    H.install("OTHER", ctl(0.9))
    assert H.decide("L", {"score": 0.1}, host_default=False).proceed is False


def test_reset_prevents_one_arm_inheriting_anothers_policy():
    H.install("L", ctl(0.9))
    assert H.installed("L")
    H.reset()
    assert not H.installed("L")
    assert H.decide("L", {"score": 0.1}, host_default=False).source == "host"


def test_the_hook_names_no_benchmark_concept():
    src = (pathlib.Path(__file__).resolve().parent.parent /
           "anchoropt" / "runtime_hook.py").read_text().lower()
    for word in ("archival", "kv", "rec_sum", "bfcl", "container", "similarity", "retrieve",
                 "memory_", "0.3"):
        assert word not in src, f"the generic hook names a benchmark concept: {word!r}"
