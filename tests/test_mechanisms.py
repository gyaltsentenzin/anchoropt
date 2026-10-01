"""The A1 and A3 mechanisms, tested in isolation.

Both are pure functions over stdlib -- no benchmark, no model, no GPU -- so what an anchor actually
*does* is inspectable without the harness. These tests pin the properties the frozen specs relied on
when the anchors were accepted, especially the ones that trade coverage away for safety.

A1  capacity_repair    shrink an over-long write while keeping every number, or REFUSE
A3  redundant_write    suppress a re-write only on a normalised-EXACT duplicate

Test data here is deliberately DOMAIN-NEUTRAL (`value_a`, `<fact> is 7.2`, ...). The mechanisms know
nothing about memory stores, healthcare records, or any other corpus, and the tests should not
smuggle that knowledge back in -- otherwise they read as benchmark-specific when the code is not.
The one thing the A1 tests do assume is that facts are carried by NUMBERS, which is a property of
`reduce_preserving_facts` itself, not of any benchmark.
"""

from __future__ import annotations

import pytest

from anchoropt.mechanisms import capacity_repair as cr
from anchoropt.mechanisms import redundant_write as rw

# ---------------------------------------------------------------------------------------------
# A1 -- fact-preserving reduction
#
# The anchor's job: a write was rejected for being too long. Shrink it to fit WITHOUT losing a
# number, or refuse and let the caller fall through. Deterministic; no model involved.
# ---------------------------------------------------------------------------------------------


def test_text_within_budget_passes_through_unchanged():
    assert cr.reduce_preserving_facts("measurement is 7.2", 100) == "measurement is 7.2"


def test_drops_non_fact_sentences_and_keeps_every_number():
    text = (
        "This sentence carries no figures at all. "
        "First measurement was 7.2 units. "
        "Another sentence with only commentary. "
        "Second measurement was 130 units."
    )
    out = cr.reduce_preserving_facts(text, 80)
    assert out is not None
    assert len(out) <= 80
    for fact in ("7.2", "130"):
        assert fact in out, f"reduction dropped the number {fact}"
    assert "commentary" not in out, "a fact-free sentence should be dropped before a fact"


def test_refuses_rather_than_dropping_a_fact():
    """The load-bearing property: a lossy repair is worse than no repair.

    If the fact-carrying sentences alone exceed the budget, return None so the caller falls through
    instead of silently writing a value with a number missing. Coverage is traded for safety here,
    deliberately.
    """
    text = "Value 1 is 11. Value 2 is 22. Value 3 is 33. Value 4 is 44. Value 5 is 55."
    assert cr.reduce_preserving_facts(text, 20) is None


def test_over_budget_with_no_facts_is_never_silently_truncated():
    text = "Sentence one has no figures. Sentence two has none either. Nor does sentence three."
    out = cr.reduce_preserving_facts(text, 20)
    assert out is None or len(out) <= 20


@pytest.mark.parametrize("budget", range(10, 200, 7))
def test_output_never_exceeds_budget(budget: int):
    text = (
        "Reading 1.5 recorded. Note without figures. Value 22 logged. "
        "Commentary only. Measure 333 confirmed."
    )
    out = cr.reduce_preserving_facts(text, budget)
    if out is not None:
        assert len(out) <= budget


def test_reduction_is_deterministic():
    """No LLM, no sampling -- the same input must always give the same repair."""
    text = "First fact 11 here. Some prose. Second fact 22 there. More prose."
    assert len({cr.reduce_preserving_facts(text, 60) for _ in range(5)}) == 1


def test_none_input_is_handled():
    assert cr.reduce_preserving_facts(None, 50) is None


# ---------------------------------------------------------------------------------------------
# A3 -- normalised-exact duplicate suppression
#
# `is_redundant` returns (verdict, reason). Every decision carries its own justification -- the same
# discipline as the incision grid's stated exclusions: a suppression with no recorded reason is
# indistinguishable from a bug.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "proposed,stored",
    [
        ("value_a", "value_a"),          # exact duplicate
        ("value_a", "value_b"),          # different value
        ("value_a", None),               # key absent from the store
    ],
)
def test_verdict_always_comes_with_a_reason(proposed, stored):
    verdict, reason = rw.is_redundant(proposed, stored)
    assert isinstance(verdict, bool)
    assert reason and len(reason) > 10, "every verdict must state why"


def test_identical_value_is_redundant():
    verdict, reason = rw.is_redundant("value_a", "value_a")
    assert verdict is True
    assert "identical" in reason


def test_whitespace_only_difference_is_still_redundant():
    """Normalisation collapses whitespace. That is the ONLY latitude the predicate has."""
    verdict, _ = rw.is_redundant("value_a  part_b\n", " value_a part_b ")
    assert verdict is True


def test_reordered_value_is_NOT_redundant():
    """The frozen spec declines 4 of 59 rather than risk suppressing real information.

    Three of those four are reorderings/paraphrases. A fuzzy or semantic comparator would buy ~7%
    more coverage at the risk of silently destroying a fact; that trade was refused.
    """
    verdict, reason = rw.is_redundant("part_b value_a", "value_a part_b")
    assert verdict is False
    assert "new information" in reason


def test_unrelated_value_is_NOT_redundant():
    verdict, _ = rw.is_redundant("entirely different content", "value_a")
    assert verdict is False


def test_superstring_is_NOT_redundant():
    """A value that merely CONTAINS the stored one still carries new information."""
    verdict, _ = rw.is_redundant("value_a plus additional detail", "value_a")
    assert verdict is False


def test_absent_key_is_NOT_redundant():
    """No stored value means nothing to be redundant against -- never suppress."""
    verdict, reason = rw.is_redundant("value_a", None)
    assert verdict is False
    assert "not present" in reason


def test_suppression_is_conservative_by_construction():
    """Across a spread of near-misses, only the normalised-exact match is suppressed."""
    stored = "value_a part_b"
    near_misses = [
        "value_a part_b extra",   # superstring
        "part_b value_a",         # reordered
        "value_a",                # prefix
        "value_a part_c",         # one token changed
        "",                       # empty
    ]
    assert rw.is_redundant("value_a  part_b", stored)[0] is True
    for v in near_misses:
        assert rw.is_redundant(v, stored)[0] is False, f"must not suppress {v!r}"


def test_normalize_is_idempotent():
    for v in ("a  b", " a b ", "a\tb\n", "a b"):
        assert rw.normalize(rw.normalize(v)) == rw.normalize(v)


def test_truncated_telemetry_raises_instead_of_comparing():
    """A truncated value compared as 'different' produced a wrong verdict TWICE.

    Telemetry caps substituted calls at ~300 chars, so comparing against it can report a real
    duplicate as novel. The guard raises rather than letting that happen silently -- and the message
    names the correct source, because an error that does not say what to do instead gets worked around.
    """
    with pytest.raises(ValueError, match="truncated"):
        rw.assert_untruncated("x" * rw.TELEMETRY_CAP, "substituted_call_telemetry")


def test_untruncated_value_passes_the_guard():
    assert rw.assert_untruncated("short value", "decoded") == "short value"
