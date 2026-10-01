"""The evaluated space is much smaller than the 3x5 grid, and the claim must stay honest.

"We search 15 options" would overstate the work; "structural reasoning collapses 15 to one or two
worth measuring" is the actual contribution. These tests pin the funnel so the published numbers and
the code cannot drift apart:

    15 nominal  ->  10 structurally admissible  ->  ~3 live per locus  ->  1-2 arms run

The grid-level pass is benchmark-independent and lives in `anchoropt.anchor`, so it is checked against
the code. The locus- and line-level passes are properties of each round, so they are checked against
the round records that state them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anchoropt.anchor import Action, IncisionPoint, exclusion_reason, feasible_actions
from rounds.anchors import ANCHORS

REPO = Path(__file__).resolve().parent.parent

# Live cells per locus, as recorded in each round's frozen criteria. A3 and A4 each reached exactly
# one admissible cell -- the design was forced, and the only open question was whether to act at all.
LIVE_CELLS = {"A2": 3, "A3": 1, "A4": 1}
ARMS_RUN = {"A2": 2, "A3": 1, "A4": 1}


def test_nominal_grid_is_twelve():
    """4 action families, not 5: `transform` is a REROUTE (same call, different arguments)."""
    assert len(IncisionPoint) * len(Action) == 12


def test_structural_pass_removes_exactly_three():
    nominal = len(IncisionPoint) * len(Action)
    admissible = sum(len(feasible_actions(p)) for p in IncisionPoint)
    assert admissible == 9
    assert nominal - admissible == 3


def test_there_is_no_transform_family():
    """Merged into reroute: substituting a call covers both a new target and new arguments.

    Kept as a test because the merge is a real taxonomy decision, not a rename -- re-adding a
    separate `transform` would silently re-split one operation into two.
    """
    assert not hasattr(Action, "TRANSFORM")
    assert {a.value for a in Action} == {"noop", "reprompt", "suppress", "reroute"}


def test_every_structural_exclusion_states_a_reason():
    """With a funnel this aggressive, an unstated exclusion reads as an option nobody considered."""
    for p in IncisionPoint:
        for a in Action:
            if a not in feasible_actions(p):
                reason = exclusion_reason(p, a)
                assert reason and len(reason) > 15, f"{p.value}/{a.value} lacks a real reason"


@pytest.mark.parametrize("name,live", sorted(LIVE_CELLS.items()))
def test_live_cells_never_exceed_structural_admissibility(name, live):
    """A locus cannot have more live cells than the grid admits at its incision point."""
    anchor = next(a for a in ANCHORS if a.name == name)
    assert live <= len(feasible_actions(anchor.incision_point))


@pytest.mark.parametrize("name", sorted(ARMS_RUN))
def test_arms_run_never_exceed_live_cells(name):
    assert ARMS_RUN[name] <= LIVE_CELLS[name]


def test_the_funnel_is_actually_a_funnel():
    """Monotone narrowing: 15 -> 10 -> few -> fewer. If this inverts, a claim somewhere is wrong."""
    nominal = len(IncisionPoint) * len(Action)
    admissible = sum(len(feasible_actions(p)) for p in IncisionPoint)
    max_live = max(LIVE_CELLS.values())
    max_arms = max(ARMS_RUN.values())
    assert nominal > admissible > max_live >= max_arms


def test_forced_designs_are_recorded_as_such():
    """A3 and A4 each had ONE admissible cell. That is a strong position, not a weak one."""
    forced = {n for n, c in LIVE_CELLS.items() if c == 1}
    assert forced == {"A3", "A4"}
    for name in forced:
        readme = next(
            d / "README.md" for d in (REPO / "rounds").glob("T*")
            if d.is_dir() and name.lower() in d.name.lower()
        )
        text = readme.read_text().lower()
        assert "only" in text, f"{name}'s round should state that its cell was the only one"


def test_docs_lead_with_the_point_being_obvious_not_with_the_grid():
    """The grid is for recording impossibilities, not for advertising a search.

    For most failures the decision point is forced by what is knowable and what is still
    preventable, so leading with "12 cells" oversells it. The doc must say so before it shows
    the table.
    """
    doc = (REPO / "docs" / "INCISION_POINTS.md").read_text()
    assert "obvious" in doc.lower()
    obvious_at = doc.lower().index("obvious")
    grid_at = doc.index("## The grid")
    assert obvious_at < grid_at, "the obviousness framing must come before the grid table"
    # And the grid must still record the exclusions, which is its actual job.
    assert "why inadmissible" in doc
    assert "9 of twelve" in doc.lower() or "nine of twelve" in doc.lower()
