"""`support` is RANKED ON, so where it came from is part of its meaning.

Pins the measured defect: a BFCL arm was ranked on support 8 with 201/590 projected firings, and the
host's LIVE comparator found exactly ONE qualifying case -- a 201x overstatement, because the offline
reconstruction could not see state an earlier phase had already built. The selector ranked on that
number as though it were a count, and a GPU round went to a one-case mechanism.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest                                                            # noqa: E402

from anchoropt.learning.candidate_selection import (                      # noqa: E402
    NEUTRAL, SUPPORT_MEASURED, SUPPORT_PROJECTED, SUPPORT_UNKNOWN, Candidate, select_candidate)


def cand(label, support, prov=SUPPORT_UNKNOWN, fires=10, op="transform"):
    return Candidate(label=label, operator=op, signal="s", boundary="post_execution",
                     support=support, fires_on_states=fires, total_states=100,
                     capability_id="cap", support_provenance=prov)


def test_provenance_defaults_to_UNKNOWN_and_is_not_measured():
    """A caller that does not think about it must not get measured-grade ranking by omission."""
    c = cand("a", 8)
    assert c.support_provenance == SUPPORT_UNKNOWN
    assert c.support_is_measured is False


def test_an_INVENTED_provenance_is_refused():
    with pytest.raises(ValueError, match="not in"):
        cand("a", 8, prov="probably_fine")


def test_MEASURED_support_wins_a_TIE_against_projected():
    """The core rule: a projected number must not win a tie on a count nobody took.

    THE LABELS ARE DELIBERATELY ADVERSARIAL. A first version used "a_measured" vs "z_projected", which
    the LEXICOGRAPHIC tie-break also resolves the same way -- so the test passed with the provenance
    term deleted and proved nothing. A neuter audit caught it. Here the measured candidate sorts LAST
    by label and by firing count, so only the provenance term can put it first.
    """
    proj = cand("a_projected", 8, SUPPORT_PROJECTED, fires=5)
    meas = cand("z_measured", 8, SUPPORT_MEASURED, fires=90)
    sel = select_candidate([proj, meas], policy=NEUTRAL)
    assert sel.chosen.label == "z_measured", sel.rationale()


def test_a_LARGER_projected_support_still_outranks_a_smaller_measured_one():
    """Deliberately NOT discarded. A projection may be right, and refusing to rank it at all would make
    every first-round arm unselectable -- which would block discovery entirely."""
    proj = cand("big_projected", 201, SUPPORT_PROJECTED)
    meas = cand("small_measured", 8, SUPPORT_MEASURED)
    sel = select_candidate([proj, meas], policy=NEUTRAL)
    assert sel.chosen.label == "big_projected", sel.rationale()


def test_UNKNOWN_is_treated_like_PROJECTED_in_a_tie():
    unk = cand("a_unknown", 8, SUPPORT_UNKNOWN, fires=5)
    meas = cand("z_measured", 8, SUPPORT_MEASURED, fires=90)
    assert select_candidate([unk, meas], policy=NEUTRAL).chosen.label == "z_measured"


def test_the_rationale_STATES_the_provenance_and_warns_on_an_unmeasured_pick():
    """A round that selected on an unmeasured number should be readable as having done so."""
    sel = select_candidate([cand("only", 201, SUPPORT_PROJECTED)], policy=NEUTRAL)
    r = sel.rationale()
    assert "[projected]" in r, r
    assert "UPPER BOUND" in r and "201x" in r, r


def test_a_measured_pick_carries_NO_warning():
    r = select_candidate([cand("only", 8, SUPPORT_MEASURED)], policy=NEUTRAL).rationale()
    assert "[measured]" in r
    assert "UPPER BOUND" not in r


def test_provenance_does_not_disturb_the_existing_tie_breaks():
    """Same support and same provenance -> the pre-registered order still decides (fewer firings first)."""
    wide = cand("a_wide", 8, SUPPORT_PROJECTED, fires=90)
    narrow = cand("z_narrow", 8, SUPPORT_PROJECTED, fires=5)
    assert select_candidate([wide, narrow], policy=NEUTRAL).chosen.label == "z_narrow"
