"""Signal expansion: the validation gate, and that core stays free of benchmark vocabulary."""

import pathlib
import re

import pytest

from anchoropt.anchor import IncisionPoint
from anchoropt.learning.signal_expansion import (
    SIGNAL_INSTALLED, SIGNAL_REFUSED_COLLISION, SIGNAL_REFUSED_CONSTANT,
    SIGNAL_REFUSED_NOT_OBSERVABLE, SIGNAL_REFUSED_NO_RUNTIME_SUPPORT,
    SIGNAL_REFUSED_UNCOMPUTABLE, SignalProposal, expand_and_resume, install_signal,
    validate_signal,
)

POST = IncisionPoint.POST_EXECUTION


class FakeRuntime:
    """A minimal host. Core must work against this, with no benchmark knowledge."""

    def __init__(self, declared=("existing_signal",), fields=None, installable=True):
        self._declared = list(declared)
        self._fields = fields
        self.installed = {}
        self._installable = installable

    def declared_signals(self):
        return tuple(self._declared)

    def observable_fields(self, boundary=None):
        return self._fields

    if True:
        def install_signal(self, name, predicate, *, boundary=None, provenance=""):
            if not self._installable:
                raise RuntimeError("this host cannot install signals")
            self.installed[name] = predicate
            self._declared.append(name)


def prop(**kw):
    base = dict(name="new_signal", boundary=POST, observable="flag", comparison="truthy")
    base.update(kw)
    return SignalProposal(**base)


# ---------------------------------------------------------------- the four checks
def test_a_discriminating_signal_is_installed():
    rt = FakeRuntime()
    states = [{"flag": True}, {"flag": False}, {"flag": False}]
    r = validate_signal(prop(), runtime=rt, observed_states=states)
    assert r.state == SIGNAL_INSTALLED and r.fired_on == 1 and r.total_states == 3
    assert r.firing_rate == pytest.approx(1 / 3)


def test_redefining_an_existing_signal_is_REFUSED_not_merged():
    """Silently merging would change the meaning of every policy referencing that name."""
    rt = FakeRuntime(declared=("existing_signal",))
    r = validate_signal(prop(name="existing_signal"), runtime=rt, observed_states=[{"flag": True}])
    assert r.state == SIGNAL_REFUSED_COLLISION
    assert "already declared" in r.detail


def test_a_signal_that_is_NEVER_true_is_refused():
    """Installing it would add a name to Phi that can never fire -- coverage that isn't."""
    rt = FakeRuntime()
    r = validate_signal(prop(), runtime=rt, observed_states=[{"flag": False}, {"flag": False}])
    assert r.state == SIGNAL_REFUSED_CONSTANT
    assert "never true" in r.detail


def test_a_signal_that_is_ALWAYS_true_is_refused():
    """This is the measured failure mode: an unconditional trigger fired on 303/303 episodes
    instead of the ~48 targeted, because the condition was evaluated where it is true of everything."""
    rt = FakeRuntime()
    r = validate_signal(prop(), runtime=rt, observed_states=[{"flag": True}, {"flag": True}])
    assert r.state == SIGNAL_REFUSED_CONSTANT
    assert "EVERY observed state" in r.detail


def test_an_unobservable_field_is_refused():
    rt = FakeRuntime(fields={"other_field"})
    r = validate_signal(prop(observable="flag"), runtime=rt, observed_states=[{"flag": True}])
    assert r.state == SIGNAL_REFUSED_NOT_OBSERVABLE
    assert "not observable" in r.detail


def test_a_nameless_proposal_is_refused():
    r = validate_signal(prop(name="  "), runtime=FakeRuntime())
    assert r.state == SIGNAL_REFUSED_UNCOMPUTABLE


def test_missing_observed_states_installs_but_SAYS_the_rate_is_unknown():
    """Without states the discrimination check cannot run; that must be stated, not hidden."""
    r = validate_signal(prop(), runtime=FakeRuntime(), observed_states=[])
    assert r.state == SIGNAL_INSTALLED
    assert "NOT checked for discrimination" in r.detail
    assert r.firing_rate is None


# ---------------------------------------------------------------- comparisons
@pytest.mark.parametrize("cmp_,value,state,want", [
    ("truthy", None, {"flag": 1}, True),
    ("truthy", None, {"flag": 0}, False),
    ("equals", "x", {"flag": "x"}, True),
    ("equals", "x", {"flag": "y"}, False),
    ("lt", 5, {"flag": 3}, True),
    ("lt", 5, {"flag": 7}, False),
    ("gt", 5, {"flag": 7}, True),
    ("present", None, {"flag": None}, True),
    ("absent", None, {}, True),
    ("absent", None, {"flag": 1}, False),
])
def test_comparison_vocabulary(cmp_, value, state, want):
    from anchoropt.learning.signal_expansion import _predicate_for
    assert _predicate_for(prop(comparison=cmp_, value=value))(state) is want


def test_a_non_numeric_value_under_a_numeric_comparison_is_FALSE_not_a_crash():
    from anchoropt.learning.signal_expansion import _predicate_for
    assert _predicate_for(prop(comparison="lt", value=5))({"flag": "not a number"}) is False


# ---------------------------------------------------------------- installation
def test_installation_goes_through_the_RUNTIME():
    rt = FakeRuntime()
    r = validate_signal(prop(), runtime=rt, observed_states=[{"flag": True}, {"flag": False}])
    ok, why = install_signal(r, runtime=rt)
    assert ok and "new_signal" in rt.installed
    assert "new_signal" in rt.declared_signals()


def test_a_host_that_cannot_install_leaves_the_residual_BLOCKED():
    """An adapter without expansion support must decline, not appear to have solved the residual."""
    rt = FakeRuntime(installable=False)
    r = validate_signal(prop(), runtime=rt, observed_states=[{"flag": True}, {"flag": False}])
    results, installed = expand_and_resume([prop()], runtime=rt,
                                           observed_states=[{"flag": True}, {"flag": False}])
    assert installed == []
    assert results[0].state == SIGNAL_REFUSED_NO_RUNTIME_SUPPORT


def test_a_refused_proposal_is_never_installed():
    rt = FakeRuntime()
    r = validate_signal(prop(), runtime=rt, observed_states=[{"flag": True}])   # constant
    ok, why = install_signal(r, runtime=rt)
    assert not ok and rt.installed == {}


def test_expand_and_resume_reports_each_verdict():
    rt = FakeRuntime()
    states = [{"flag": True}, {"flag": False}]
    good, bad = prop(name="good"), prop(name="existing_signal")
    results, installed = expand_and_resume([good, bad], runtime=rt, observed_states=states)
    assert installed == ["good"]
    assert [r.state for r in results] == [SIGNAL_INSTALLED, SIGNAL_REFUSED_COLLISION]


# ---------------------------------------------------------------- no domain leakage
def test_core_signal_expansion_carries_NO_benchmark_vocabulary():
    """Expansion is generic; a benchmark word here would make core benchmark-specific."""
    src = (pathlib.Path(__file__).resolve().parent.parent /
           "anchoropt" / "learning" / "signal_expansion.py").read_text().lower()
    for word in ("archival", "core_memory", "kv", "rec_sum", "bfcl", "container",
                 "memory_retrieve", "similarity"):
        assert word not in src, f"core signal_expansion names a benchmark concept: {word!r}"


def test_the_action_space_is_NOT_expanded():
    """Expansion changes what is OBSERVABLE. The action space stays the fixed small host-declared set."""
    src = (pathlib.Path(__file__).resolve().parent.parent /
           "anchoropt" / "learning" / "signal_expansion.py").read_text()
    assert "Action" not in src, "signal expansion must not touch the action space"


# ---------------------------------------------------------------- composition with the executor rule
def test_an_EXPANDED_signal_can_reach_a_signal_agnostic_executor():
    """Expansion and materializability must COMPOSE, or Phi grows while every new cell is unreachable.

    The executor's `signals` tuple is the shipped whitelist; a signal installed by expansion is not in
    it. Without this path the two mechanisms are a silent dead end: validation installs the signal,
    every candidate using it is then pruned `executor_signal_unsupported`, and the round reports no
    materializable arms for a reason that looks like a host limitation.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    from anchoropt.anchor import Action

    rt.reset_expanded_signals()
    try:
        states = [{"error_kind": "x"}, {"error_kind": None}]
        p = SignalProposal(name="probe_expanded_cond", boundary=POST,
                           observable="error_kind", comparison="equals", value="x")
        _, installed = expand_and_resume([p], runtime=rt, observed_states=states)
        assert installed == ["probe_expanded_cond"]

        ok, why = rt.executor_supports(POST, Action.REROUTE, "probe_expanded_cond", {})
        assert ok, why
        assert "EXPANDED" in why

        # a NON-agnostic executor keeps its whitelist: its eta is a template slot chosen per signal,
        # so an unknown condition has no text to inject and must not be reported materializable.
        ok2, why2 = rt.executor_supports(POST, Action.REPROMPT, "probe_expanded_cond", {})
        assert not ok2 and "executor_signal_unsupported" in why2
    finally:
        rt.reset_expanded_signals()


def test_expanded_signals_are_DISTINGUISHABLE_from_shipped_ones():
    """A recovery claim made with an expanded Phi is a different claim from one made without."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt

    rt.reset_expanded_signals()
    try:
        shipped = set(rt.declared_signals())
        p = SignalProposal(name="probe_distinguishable", boundary=POST,
                           observable="error_kind", comparison="present")
        expand_and_resume([p], runtime=rt,
                          observed_states=[{"error_kind": "x"}, {}])
        assert "probe_distinguishable" in rt.expanded_signals()
        assert "probe_distinguishable" not in shipped
        assert set(rt.declared_signals()) == shipped | {"probe_distinguishable"}
    finally:
        rt.reset_expanded_signals()


def test_expansion_cannot_shadow_a_shipped_signal_in_the_runtime():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt

    shipped = rt.declared_signals()[0]
    with pytest.raises(ValueError):
        rt.install_signal(shipped, lambda s: True)


def test_reset_clears_expansion_between_experiments():
    """One round must not inherit another's Phi -- an expanded signal is session state, not shipped."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt

    rt.reset_expanded_signals()
    before = len(rt.declared_signals())
    expand_and_resume([SignalProposal(name="probe_reset", boundary=POST,
                                      observable="error_kind", comparison="present")],
                      runtime=rt, observed_states=[{"error_kind": "x"}, {}])
    assert len(rt.declared_signals()) == before + 1
    rt.reset_expanded_signals()
    assert len(rt.declared_signals()) == before


# ---------------------------------------------------------------- the expressibility gate itself
def test_english_filler_in_a_signal_NAME_does_not_make_a_residual_expressible():
    """MEASURED DEFECT: `append_would_exceed_cap` was reported expressible for a retrieval residual
    on the strength of the word "would", and `no_informative_result` on "informative".

    Auxiliary verbs and generic adjectives appear in almost any diagnosis prose, so a match on them is
    grammar, not evidence. This is why nothing was ever SIGNAL_BLOCKED across five recovery replays and
    the expansion branch never fired: the gate OVER-reported expressibility, hiding real gaps in Phi
    behind an accidental word match.
    """
    from anchoropt.learning.candidate_search import _WEAK_TOKENS, _signal_matches
    for filler in ("would", "informative", "before", "another", "returned"):
        assert filler in _WEAK_TOKENS, f"{filler!r} must not carry a match on its own"
    text = ("the agent would have answered from the first retrieval without checking whether the "
            "returned content was informative")
    assert not _signal_matches("append_would_exceed_cap", text)
    assert not _signal_matches("no_informative_result", text)


def test_the_gate_BLOCKS_a_condition_phi_genuinely_cannot_observe():
    """A gate that never blocks is not a gate. These must block; the last must not."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "bfcl_v4"))
    import bfcl_runtime as rt
    from anchoropt.learning.candidate_search import expressible_under
    from anchoropt.runtime import ResidualDiagnosis

    def diag(mech):
        return ResidualDiagnosis(case_id="x", mechanism=mech, evidence="",
                                 consequential_decision=mech, proposed_behavior_change="",
                                 provider="p")

    # the cross-store clause: the shipped signal sees a weak match, NOT "the other store was skipped"
    blocked = diag("the agent read only one store and never consulted the second one, so evidence "
                   "sitting in the other place was never seen")
    assert not expressible_under(blocked, runtime=rt)[0]
    # conditions plainly outside this Phi
    assert not expressible_under(diag("the session expired before it finished"), runtime=rt)[0]
    assert not expressible_under(diag("the operations ran in the wrong sequence"), runtime=rt)[0]
    # and an ordinary weak-retrieval residual must STILL pass -- the fix must not block everything
    ok = diag("the retrieval returned a low similarity match and the agent answered from it anyway")
    assert expressible_under(ok, runtime=rt)[0]
