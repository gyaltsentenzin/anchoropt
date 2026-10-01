"""The stage-3 discovery gate is LEXICAL, and this pins the measured consequence.

R14's finding: `candidate_search.expressible_under` decides whether a new signal is needed by matching
signal NAMES against diagnosis PROSE. Returning False is the ONLY route to signal expansion, so this
gate is the whole of "the system discovers it needs a new observation".

Measured: it would demand a new signal on 1 of 4 historical anchors, and the bias is correlated with
SIGNAL TIER -- upper-tier names are quantitative (similarity/threshold) while failure prose is phenomenal
("returned nothing well matched"), so the vocabularies do not intersect.

These tests EXIST TO FAIL when the gate is fixed. They are the baseline V2 is measured against, and a
failure here means someone changed V1's discovery behaviour -- which must be a deliberate, recorded act,
not a side effect. See rounds/AUTONOMY/R14/SIGNAL_DISCOVERY_AUDIT.md and PROPOSED_STAGE3_GATE.json.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

from anchoropt.learning.candidate_search import _signal_matches      # noqa: E402


def _rt():
    import bfcl_runtime
    return bfcl_runtime


#: Prose describing each signal's OWN condition, in the phenomenal terms a failure report uses.
SELF_DESCRIPTIONS = {
    "container_at_capacity": "the core container is full and the write is refused",
    "redundant_proposed_write": "the model issues a redundant write whose value is already stored",
    "identifier_not_found": "the read did not find the value under that identifier",
    "duplicate_identifier": "the store reports a duplicate identifier for the key",
    "no_tool_call_at_all": "the model answers with no tool call at all",
    "append_would_exceed_cap": "appending would exceed the blob cap",
    "clear_proposed_at_capacity": "a clear is proposed while the container is at capacity",
    "container_slots_exhausted": "the relocation target container has no slots left",
    "proposal_already_refused": "the proposal was already refused earlier",
    "retrieval_similarity_below_threshold": "a read succeeded but returned nothing well matched",
    "no_informative_result": "the result came back empty and uninformative",
}


def test_the_evidence_quality_signal_cannot_be_FOUND_from_its_own_description():
    """The load-bearing case: A9's signal, the most valuable single historical discovery.

    Its name is quantitative (`similarity`, `threshold`); a description of the failure it detects uses
    neither word. So a name-vs-prose matcher can never surface it, which is why the historical climb to
    rung 3 could not have been made by this gate.
    """
    rt = _rt()
    sig = "retrieval_similarity_below_threshold"
    assert not _signal_matches(sig, SELF_DESCRIPTIONS[sig], rt), (
        "the evidence-quality signal is now findable from prose describing its own condition -- if this "
        "is intentional, update rounds/AUTONOMY/R14/SIGNAL_DISCOVERY_AUDIT.md, because it changes the "
        "audit's central claim")


def test_a_retrieval_residual_falsely_matches_CAPACITY_signals_on_the_word_container():
    """The A9 false positive, reproduced exactly: one shared English word routes a retrieval residual
    onto two capacity signals."""
    rt = _rt()
    prose = ("a read succeeded and returned nothing well matched. in 13 of 13 verified cases the "
             "information was present in the second container.")
    for wrong in ("container_at_capacity", "container_slots_exhausted"):
        assert _signal_matches(wrong, prose, rt), (
            f"{wrong} no longer matches a retrieval residual -- the gate may have been fixed; see "
            f"PROPOSED_STAGE3_GATE.json")
    assert not _signal_matches("retrieval_similarity_below_threshold", prose, rt), (
        "the CORRECT signal is unmatched while two wrong ones match -- this asymmetry IS the finding")


def test_the_lexical_bias_is_tier_correlated_not_uniform():
    """9 of 11 signals self-match; the misses are the upper-tier ones. That correlation is the finding:
    a uniform error rate would be noise, a tier-correlated one is a structural limit."""
    rt = _rt()
    misses = sorted(s for s, prose in SELF_DESCRIPTIONS.items() if not _signal_matches(s, prose, rt))
    assert misses == ["identifier_not_found", "retrieval_similarity_below_threshold"], (
        f"the set of signals unfindable from their own description changed: {misses}")


@pytest.mark.parametrize("sig", sorted(SELF_DESCRIPTIONS))
def test_every_declared_signal_still_has_a_self_description_on_record(sig):
    """Guards the test above: a new signal added to the vocabulary must be classified here too, or the
    tier-correlation claim quietly stops covering the whole vocabulary."""
    assert sig in _rt().declared_signals()


def test_no_declared_signal_is_missing_from_this_records_coverage():
    declared = set(_rt().declared_signals())
    assert declared == set(SELF_DESCRIPTIONS), (
        f"vocabulary drift: declared-but-unclassified={sorted(declared - set(SELF_DESCRIPTIONS))}, "
        f"classified-but-undeclared={sorted(set(SELF_DESCRIPTIONS) - declared)}")
