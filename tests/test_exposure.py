"""The two diagnostics that decide whether a delta means anything.

Both encode a case where a positive number was believed on this project and turned out to measure
something else. The tests reproduce those measured cases, so the diagnostic and the historical record
have to keep agreeing.
"""

from __future__ import annotations

import pytest

from anchoropt.learning.exposure import (
    Cell,
    displacement_check,
    exposure_weighted_delta,
)

# ---------------------------------------------------------------------------------------------
# exposure
# ---------------------------------------------------------------------------------------------

def test_the_rejected_candidate_reads_as_non_attributable():
    """The measured case: +3.63 pp corpus-wide, and the gate fired ZERO times.

    vector (exposed) +1.12 pp on n=89, against a +4.67 pp noise floor on 214 unexposed cases.
    """
    report = exposure_weighted_delta([
        Cell("vector", n=89, control_correct=30, arm_correct=31, exposed=True, firings=0),
        Cell("kv", n=105, control_correct=30, arm_correct=35, exposed=False),
        Cell("rec_sum", n=109, control_correct=54, arm_correct=59, exposed=False),
    ])
    assert report.corpus_delta_pp == pytest.approx(3.63, abs=0.01)
    assert report.exposed_delta_pp == pytest.approx(1.12, abs=0.01)
    assert report.unexposed_delta_pp == pytest.approx(4.67, abs=0.01)
    assert not report.attributable, "0 firings must never read as attributable"

    joined = " ".join(report.warnings).lower()
    assert "non-attribution" in joined
    assert "perturbation" in joined


def test_zero_firings_with_a_positive_delta_is_always_flagged():
    report = exposure_weighted_delta([
        Cell("exposed", n=100, control_correct=40, arm_correct=50, exposed=True, firings=0),
    ])
    assert not report.attributable
    # Case-insensitive: the claim is pinned, not the capitalisation.
    assert any("non-attribution" in w.lower() for w in report.warnings)


def test_firings_make_a_delta_attributable():
    report = exposure_weighted_delta([
        Cell("exposed", n=100, control_correct=40, arm_correct=50, exposed=True, firings=12),
    ])
    assert report.attributable
    assert not any("non-attribution" in w.lower() for w in report.warnings)


def test_a7s_exposure_dilution_is_computed_not_asserted():
    """+7.34 pp on rec_sum (109/303) must come out as +2.64 pp corpus-wide."""
    report = exposure_weighted_delta([
        Cell("rec_sum", n=109, control_correct=62, arm_correct=70, exposed=True, firings=21),
        Cell("kv", n=105, control_correct=37, arm_correct=37, exposed=False),
        Cell("vector", n=89, control_correct=37, arm_correct=37, exposed=False),
    ])
    assert report.exposed_delta_pp == pytest.approx(7.34, abs=0.01)
    assert report.corpus_delta_pp == pytest.approx(2.64, abs=0.01)
    assert report.unexposed_delta_pp == 0.0, "unexposed backends are untouched by construction"
    assert report.attributable


def test_partial_exposure_always_warns_to_quote_the_exposure():
    report = exposure_weighted_delta([
        Cell("rec_sum", n=109, control_correct=62, arm_correct=70, exposed=True, firings=21),
        Cell("kv", n=194, control_correct=70, arm_correct=70, exposed=False),
    ])
    assert any("must never be quoted without" in w for w in report.warnings)


def test_a_cell_marked_unexposed_that_fires_is_a_broken_exposure_rule():
    report = exposure_weighted_delta([
        Cell("exposed", n=50, control_correct=20, arm_correct=25, exposed=True, firings=5),
        Cell("oops", n=50, control_correct=20, arm_correct=21, exposed=False, firings=3),
    ])
    assert any("exposure rule is wrong" in w for w in report.warnings)


def test_fully_exposed_run_has_no_noise_estimate_and_says_nothing_false():
    report = exposure_weighted_delta([
        Cell("all", n=303, control_correct=88, arm_correct=128, exposed=True, firings=59),
    ])
    assert report.unexposed_n == 0
    assert report.unexposed_delta_pp == 0.0
    assert report.corpus_delta_pp == report.exposed_delta_pp
    # With nothing unexposed there is no "reaches x% of the corpus" warning to make.
    assert not any("must never be quoted without" in w for w in report.warnings)


def test_summary_names_the_corpus_figure_as_the_one_to_quote():
    report = exposure_weighted_delta([
        Cell("a", n=100, control_correct=40, arm_correct=45, exposed=True, firings=5),
        Cell("b", n=100, control_correct=40, arm_correct=40, exposed=False),
    ])
    text = report.summary()
    assert "the figure to quote" in text
    assert "free noise estimate" in text


def test_empty_input_does_not_explode():
    report = exposure_weighted_delta([])
    assert report.corpus_n == 0
    assert report.corpus_delta_pp == 0.0
    assert not report.attributable


# ---------------------------------------------------------------------------------------------
# displacement
# ---------------------------------------------------------------------------------------------

def test_a6s_measured_sites_classify_as_selection():
    """The measured case: A6's own site 0 -> 44 while A1's collapses 22 -> 6."""
    report = displacement_check(
        incumbent_sites={"a1_reroute": 22, "controller_write_repair": 135},
        candidate_sites={"a6_reroute": 44, "a1_reroute": 6, "controller_write_repair": 114},
    )
    assert report.verdict == "SELECTION"
    assert report.displaces_incumbent
    assert "a1_reroute" in report.collapsed_sites
    assert report.collapsed_sites["a1_reroute"] == (22, 6)
    assert "a6_reroute" in report.new_sites
    assert "cannot add information" in report.detail


def test_a_new_site_without_a_collapse_is_additive():
    report = displacement_check(
        incumbent_sites={"a1_reroute": 22},
        candidate_sites={"a1_reroute": 24, "a7_compact": 21},
    )
    assert report.verdict == "ADDITIVE"
    assert not report.displaces_incumbent
    assert "a7_compact" in report.new_sites


def test_no_dispatches_at_all_is_inert():
    report = displacement_check({"a1_reroute": 22}, {})
    assert report.verdict == "INERT"
    assert "not attributable" in report.detail


def test_zero_count_candidate_sites_are_inert_too():
    report = displacement_check({"a1_reroute": 22}, {"a6_reroute": 0})
    assert report.verdict == "INERT"


def test_unchanged_sites_are_reported_as_no_change():
    report = displacement_check({"a1_reroute": 22}, {"a1_reroute": 22})
    assert report.verdict == "NO CHANGE"
    assert not report.new_sites


def test_growth_is_tracked_separately_from_collapse():
    report = displacement_check(
        incumbent_sites={"a1_reroute": 10, "other": 5},
        candidate_sites={"a1_reroute": 20, "other": 5},
    )
    assert report.grew_sites["a1_reroute"] == (10, 20)
    assert not report.collapsed_sites


def test_collapse_ratio_is_tunable_and_bounds_the_verdict():
    sites_before = {"a1": 20}
    sites_after = {"a6": 10, "a1": 12}
    # 12/20 = 0.6, so it is not a collapse at the default 0.5 threshold.
    assert displacement_check(sites_before, sites_after).verdict == "ADDITIVE"
    # Raise the threshold and the same movement counts as displacement.
    assert displacement_check(sites_before, sites_after, collapse_ratio=0.7).verdict == "SELECTION"
