"""The three regression levels, and the certificates they must emit.

The L1/L2 boundary is the point of this file: a predicate that answers True in isolation proved
nothing three separate times (unreachable branch, spec installed without its predicate module,
telemetry allowlist dropping the firing keys). L1 cannot see any of those; L2 can.
"""

from __future__ import annotations

import pytest

from anchoropt.learning.golden_registry import (
    AcceptedController, ControllerIdentity, ExecutionTelemetry, GoldenRegistry, PairedOutcome,
    RuntimeProvenance)
from anchoropt.learning.regression_protocol import (
    CAPABILITY_IDENTITY_MISMATCH, COMPOSITION_CONFLICT, DID_NOT_ACT, DID_NOT_FIRE,
    FIRED_ON_NEGATIVE, L1_CONTRACT, L2_SMOKE, L3_PAIRED, MISSING_IMPORT, OUTCOME_DRIFT,
    PROBE_UNAVAILABLE, Certificate, RegressionSuite, compare_outcomes)


def _entry(name="c", **kw):
    base = dict(
        name=name,
        identity=ControllerIdentity("post_execution", "sig", "reroute", "transform", "cap", "prereq"),
        spec={}, provenance=RuntimeProvenance(split="train", model="m"),
        telemetry=ExecutionTelemetry(requested=9, fired=3, episodes=2, acted=3),
        outcome=PairedOutcome(arm_correct=5, control_correct=3, n_scored=4,
                              gains=("a", "b"), losses=(),
                              case_ids={"a": True, "b": True, "c": False, "d": True}),
        positive_states=({"k": 1},), negative_states=({"k": 0},))
    base.update(kw)
    return AcceptedController(**base)


class Host:
    """Minimal adapter stand-in. Benchmark-free on purpose: core must not need more."""

    def __init__(self, fires=True, acts=True, cap="cap", neg_fires=False,
                 contract=(), outcome=None, raise_on=None):
        self.fires, self.acts, self.cap, self.neg_fires = fires, acts, cap, neg_fires
        self.contract_certs, self.outcome, self.raise_on = contract, outcome, raise_on
        self.smoke_calls = 0

    def contract_check(self, entry):
        if self.raise_on == "contract":
            raise ImportError("no module named install_controller")
        if self.raise_on == "contract_ni":
            raise NotImplementedError("no contract probe on this host")
        return self.contract_certs

    def smoke_execute(self, entry, state):
        self.smoke_calls += 1
        if self.raise_on == "smoke":
            raise NotImplementedError("no live evaluator available here")
        positive = state.get("k") == 1
        fired = self.fires if positive else self.neg_fires
        return {"fired": fired, "acted": self.acts if fired else False, "capability_id": self.cap}

    def paired_measure(self, entry):
        if self.raise_on == "paired":
            raise NotImplementedError("paired measurement needs the real runtime")
        return self.outcome


# -- L1 ----------------------------------------------------------------------------------------

def test_a_missing_import_is_an_explicit_certificate_not_zero_candidates():
    """The defect: the module that builds the controller vanishes and the run reports 'no candidates'."""
    rep = RegressionSuite(GoldenRegistry([_entry()]), Host(raise_on="contract")).run([L1_CONTRACT])
    cert = rep.failures()[0]
    assert cert.reason == MISSING_IMPORT and "install_controller" in cert.detail
    assert cert.remedy and not rep.ok


def test_an_unavailable_probe_is_a_SKIP_not_a_pass():
    res = RegressionSuite(GoldenRegistry([_entry()]),
                          Host(raise_on="contract_ni")).run_contract()
    assert res.ok                      # skip does not fail the level
    assert res.skipped and res.skipped[0].reason == PROBE_UNAVAILABLE
    assert not res.passed              # but it is NOT recorded as a pass


# -- L2 ----------------------------------------------------------------------------------------

def test_a_controller_that_stops_firing_fails_with_a_reachability_remedy():
    rep = RegressionSuite(GoldenRegistry([_entry()]), Host(fires=False)).run([L1_CONTRACT, L2_SMOKE])
    cert = [c for c in rep.failures() if c.reason == DID_NOT_FIRE][0]
    assert "reachable" in cert.remedy and "telemetry" in cert.remedy


def test_firing_without_acting_is_its_own_failure():
    """A predicate firing is not an intervention."""
    rep = RegressionSuite(GoldenRegistry([_entry()]), Host(acts=False)).run([L1_CONTRACT, L2_SMOKE])
    assert [c for c in rep.failures() if c.reason == DID_NOT_ACT]


def test_dispatch_to_the_wrong_executor_is_caught():
    """The two-executor cell: right signal, wrong mechanism, under the accepted name."""
    rep = RegressionSuite(GoldenRegistry([_entry()]), Host(cap="OTHER")).run([L1_CONTRACT, L2_SMOKE])
    cert = [c for c in rep.failures() if c.reason == CAPABILITY_IDENTITY_MISMATCH][0]
    assert "OTHER" in cert.detail and "cap" in cert.detail


def test_a_widened_predicate_is_caught_by_the_negative_state():
    rep = RegressionSuite(GoldenRegistry([_entry()]), Host(neg_fires=True)).run([L1_CONTRACT, L2_SMOKE])
    assert [c for c in rep.failures() if c.reason == FIRED_ON_NEGATIVE]


def test_an_entry_with_no_positive_state_cannot_pass_L2():
    rep = RegressionSuite(GoldenRegistry([_entry(positive_states=())]), Host()).run([L2_SMOKE])
    assert not rep.ok and rep.failures()[0].reason == PROBE_UNAVAILABLE


def test_a_healthy_controller_passes_L1_and_L2():
    rep = RegressionSuite(GoldenRegistry([_entry()]), Host()).run([L1_CONTRACT, L2_SMOKE])
    assert rep.ok and rep.results[L2_SMOKE].passed == ("c",)


def test_the_schedule_stops_at_the_first_failing_level():
    """An L1 failure makes L2 describe something other than the accepted controller."""
    host = Host(raise_on="contract")
    RegressionSuite(GoldenRegistry([_entry()]), host).run([L1_CONTRACT, L2_SMOKE])
    assert host.smoke_calls == 0


# -- L3 ----------------------------------------------------------------------------------------

def test_offsetting_flips_are_caught_even_though_the_total_matches():
    """THE measured reason this compares per case: 59 vs 59 can hide a gain cancelling a loss."""
    e = _entry()
    got = PairedOutcome(arm_correct=5, control_correct=3, n_scored=4,
                        gains=("a", "c"), losses=("b",),
                        case_ids={"a": True, "b": False, "c": True, "d": True})
    certs = compare_outcomes(e, got)
    assert certs and any(c.reason == OUTCOME_DRIFT for c in certs)
    assert any("per-case outcome" in c.detail for c in certs)


def test_equal_net_with_changed_gain_loss_composition_is_reported():
    e = _entry()
    got = PairedOutcome(arm_correct=5, control_correct=3, n_scored=4,
                        gains=("a", "b", "c"), losses=("d",), case_ids=dict(e.outcome.case_ids))
    assert any("composition changed" in c.detail for c in compare_outcomes(e, got))


def test_a_changed_corpus_does_not_read_as_agreement():
    """Diffing only shared keys would silently pass; the id SETS are compared first."""
    e = _entry()
    got = PairedOutcome(arm_correct=5, control_correct=3, n_scored=3,
                        gains=("a", "b"), losses=(), case_ids={"a": True, "b": True, "z": True})
    certs = compare_outcomes(e, got)
    assert any("case id sets differ" in c.detail for c in certs)
    assert any("re-establish the same corpus" in c.remedy for c in certs)


def test_an_identical_remeasurement_passes_L3():
    e = _entry()
    assert compare_outcomes(e, e.outcome) == ()


def test_a_net_change_names_both_sides():
    e = _entry()
    got = PairedOutcome(arm_correct=4, control_correct=3, n_scored=4,
                        gains=("a",), losses=(), case_ids=dict(e.outcome.case_ids))
    d = " ".join(c.detail for c in compare_outcomes(e, got))
    assert "+2" in d and "+1" in d


def test_paired_needs_the_real_runtime_and_says_so():
    res = RegressionSuite(GoldenRegistry([_entry()]), Host(raise_on="paired")).run_paired()
    assert res.skipped and "real runtime" in res.skipped[0].remedy


# -- composition ------------------------------------------------------------------------------

def test_installing_over_an_accepted_cell_emits_a_composition_certificate():
    suite = RegressionSuite(GoldenRegistry([_entry("acc")]), Host())
    cand = _entry("new", identity=ControllerIdentity(
        "post_execution", "other", "reroute", "transform", "NEW_CAP", "prereq"))
    certs = suite.check_composition(cand)
    assert certs and certs[0].reason == COMPOSITION_CONFLICT
    assert "smoke-test both" in certs[0].remedy


def test_report_summary_lists_failures_actionably():
    rep = RegressionSuite(GoldenRegistry([_entry()]), Host(fires=False)).run([L1_CONTRACT, L2_SMOKE])
    s = rep.summary()
    assert "L2_SMOKE" in s and "FAIL" in s and "remedy" in s


def test_core_module_names_no_benchmark_concept():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "anchoropt" / "learning"
    for mod in ("regression_protocol.py", "golden_registry.py"):
        src = (root / mod).read_text()
        for bad in ("archival_memory", "core_memory", "rec_sum", "memory_kv", "bfcl", "granite"):
            assert bad not in src, f"{mod} names the benchmark identifier {bad!r}"
