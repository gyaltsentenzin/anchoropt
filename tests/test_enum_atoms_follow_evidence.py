"""Every OBSERVED declared enum value must be nameable. Cost is bounded downstream, not by omission.

WHY THIS FILE EXISTS, and why it was rewritten once. `atoms_for` twice restricted which enum values
could appear in a predicate, and both restrictions deleted valid points from the search space:

  1. A per-field cap applied by SLICING the declaration tuple, so expressibility depended on the order
     someone typed the values in. Measured on one host's 6-value refusal-class field: it kept the value
     observed 4 times and dropped the one observed 124 times.
  2. The same cap applied after ranking by observed FREQUENCY. Better -- but still wrong, because a
     rare value can be the most consequential one. On that same field the 4-occurrence value was linked
     to 50 downstream failures, MORE than the 301-occurrence value's 43. Frequency at the decision
     boundary says nothing about how much loss the condition explains.

The deeper point is about search RECORDS, not recall: a value that cannot be named is absent from the
record entirely, so the loop reports a closed question it never asked -- a silent omission looks exactly
like an exhausted search. A rejected candidate leaves a reason; an unnameable one leaves nothing.

So the alphabet is complete over observed values, and cost is controlled where it belongs: the
constant-predicate filter, MAX_CANDIDATES, the conjunction window, and residual-driven selection.

These tests pin the PROPERTY (completeness over observed values, unobserved ones excluded as constants)
rather than any constant, because that is what the next regression would violate.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anchoropt.learning.signal_grammar import atoms_for, synthesize  # noqa: E402


class _Decl:
    """Minimal field declaration: what the adapter says exists, and of what type."""

    def __init__(self, type_, enum=()):
        self.type = type_
        self.enum = tuple(enum)


# More values than the old cap of 4, ordered so the RAREST is declared first -- the shape that made the
# first defect invisible. Generic names: core must not know a benchmark's vocabulary.
_VALUES = ("v_rarest", "v_rare", "v_common", "v_frequent", "v_second", "v_tail")

#: Deliberately spans three orders of magnitude, including a value seen FOUR times -- the real case that
#: motivated removing the cap (4 occurrences, 50 linked downstream failures).
_OBSERVED = {"v_frequent": 301, "v_second": 155, "v_common": 124,
             "v_tail": 49, "v_rare": 38, "v_rarest": 4}


def _states(counts):
    return [{"kind": v} for v, n in counts.items() for _ in range(n)]


def _offered(fields, states):
    return [a.value for a in atoms_for(fields, states) if a.op == "equals"]


def test_EVERY_observed_value_is_nameable_however_rare():
    """The headline. Six declared values, all observed, all offered -- no cap."""
    offered = _offered({"kind": _Decl(str, _VALUES)}, _states(_OBSERVED))
    assert set(offered) == set(_VALUES), f"missing {sorted(set(_VALUES) - set(offered))}"


def test_the_FOUR_occurrence_value_survives():
    """The specific regression: rarity must not decide expressibility.

    A value seen 4 times can be linked to more downstream loss than one seen 301 times, so frequency
    is not a proxy for importance and must not gate the alphabet.
    """
    assert "v_rarest" in _offered({"kind": _Decl(str, _VALUES)}, _states(_OBSERVED))


def test_a_frequent_value_declared_LAST_is_nameable():
    """The original defect, kept as a guard against reintroducing a declaration-order slice."""
    values = ("a", "b", "c", "d", "e", "hot")
    states = _states({"hot": 500, "a": 1, "b": 1, "c": 1, "d": 1, "e": 1})
    assert "hot" in _offered({"kind": _Decl(str, values)}, states)


def test_an_UNOBSERVED_value_is_NOT_offered():
    """Evidence still bounds the alphabet: `x == c` is constant-False when no state carries `c`.

    This is not a budget -- it keeps a predicate from validating on a value this residual cannot
    exhibit, which is the boundary-discipline error in a different guise.
    """
    offered = _offered({"kind": _Decl(str, _VALUES)}, _states({"v_common": 10, "v_tail": 3}))
    assert set(offered) == {"v_common", "v_tail"}


def test_values_are_ordered_MOST_OBSERVED_first():
    """So that when a DOWNSTREAM budget truncates, what survives is what discriminates here."""
    offered = _offered({"kind": _Decl(str, _VALUES)}, _states(_OBSERVED))
    assert offered == sorted(offered, key=lambda v: -_OBSERVED[v])


def test_a_field_with_NO_observed_values_falls_back_to_the_declaration():
    """A synthetic or probe state set must still be searchable rather than silently empty."""
    fields = {"kind": _Decl(str, _VALUES)}
    assert _offered(fields, [{}]) == list(_VALUES)


def test_the_falsy_atom_survives():
    """"this field carries nothing" is a distinct condition, not a value of the enum."""
    atoms = atoms_for({"kind": _Decl(str, _VALUES)}, _states(_OBSERVED))
    assert any(a.field == "kind" and a.op == "falsy" for a in atoms)


def test_a_predicate_over_a_RARE_value_can_be_SYNTHESIZED():
    """End to end: nameable is only useful if synthesis returns it.

    The rare value must also DISCRIMINATE, so pair it with a second field and check the condition
    reaches the candidate list rather than merely the atom pool.
    """
    fields = {"kind": _Decl(str, _VALUES), "flag": _Decl(bool)}
    states = [{"kind": v, "flag": (v == "v_rarest")}
              for v, n in _OBSERVED.items() for _ in range(n)]
    described = " | ".join(p.describe() for p in synthesize(fields, states, declared=()))
    assert "v_rarest" in described, "the rare value never reached a synthesized predicate"
