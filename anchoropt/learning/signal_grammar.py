"""A constrained signal grammar, searched by CORE over fields the adapter merely DECLARES.

    phi ::= x | not x | x == c | x < theta | x > theta | phi_1 and phi_2

The adapter's job is to say what is observable and of what type. Choosing which predicate to try is
the ALGORITHM's job -- otherwise the benchmark is supplying the useful predicates and it is fair to ask
what Self-Evolve is learning. So there are no hand-written candidate signals here: atoms are ENUMERATED
from typed field declarations, thresholds come from the residual's OWN observed values, and conjunctions
are formed only when every simpler form has already failed.

STAGED BY CONSTRUCTION, cheapest first:
    stage 1  atoms          bool x, not x, x == c per declared enum value
    stage 2  thresholds     x < theta, x > theta at quantiles of the OBSERVED distribution
    stage 3  conjunctions   phi_1 and phi_2 over stage-1/2 survivors, and only if 1 and 2 found nothing

Stage 3 exists for the case that motivated this module: a residual whose condition IS partly
representable -- "the retrieval was weak" -- but whose distinguishing clause is a second fact, "and the
other store was never consulted". Neither atom alone separates the failures; the conjunction does. A
grammar that stops at atoms cannot express that, and reports the residual expressible when it is not.

WHY THRESHOLDS ARE DATA-DERIVED. A hand-picked constant is a hidden hyperparameter, and one measured
round chose 0.75 for a quantity whose observed maximum was 0.6984 -- a threshold that fires on
everything is not a threshold. Quantiles of the observed values cannot do that.

NO BENCHMARK VOCABULARY. Field NAMES flow through from the adapter as opaque strings; this module never
mentions one. A test greps for domain words.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# NO PER-FIELD ATOM CAP. There was one (`MAX_ATOMS_PER_FIELD = 4`, "keeps an enum field from
# dominating the grid") and it removed valid predicates from the alphabet: a declared value became
# unnameable, which is indistinguishable from an exhausted search. Enumeration cost is bounded by
# MAX_CANDIDATES on the returned list, the conjunction window in `synthesize`, and residual-driven
# candidate selection upstream -- all of which bound WORK without shrinking what is expressible.
MAX_CANDIDATES = 64              # a search budget, not a semantic limit
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


@dataclass(frozen=True)
class Atom:
    """One primitive predicate over a declared field."""

    field: str
    op: str                      # truthy | falsy | equals | lt | gt
    value: Any = None

    def name(self) -> str:
        if self.op == "truthy":
            return self.field
        if self.op == "falsy":
            return f"not_{self.field}"
        if self.op == "equals":
            return f"{self.field}_is_{_slug(self.value)}"
        return f"{self.field}_{self.op}_{_slug(self.value)}"

    def describe(self) -> str:
        return {"truthy": f"{self.field} is true",
                "falsy": f"{self.field} is false",
                "equals": f"{self.field} == {self.value!r}",
                "lt": f"{self.field} < {self.value}",
                "gt": f"{self.field} > {self.value}"}[self.op]

    def evaluate(self, state: Mapping[str, Any]) -> bool:
        if self.field not in state:
            return False
        v = state.get(self.field)
        if self.op == "truthy":
            return bool(v)
        if self.op == "falsy":
            return not bool(v)
        if self.op == "equals":
            return v == self.value
        if v is None:
            return False
        try:
            return float(v) < float(self.value) if self.op == "lt" else float(v) > float(self.value)
        except (TypeError, ValueError):
            return False


@dataclass(frozen=True)
class Conjunction:
    """phi_1 and phi_2. Two terms only: the point is to add ONE missing clause, not to fit."""

    terms: tuple[Atom, ...]

    def name(self) -> str:
        return "__and__".join(t.name() for t in self.terms)

    def describe(self) -> str:
        return " AND ".join(t.describe() for t in self.terms)

    def evaluate(self, state: Mapping[str, Any]) -> bool:
        return all(t.evaluate(state) for t in self.terms)


def _slug(v: Any) -> str:
    s = str(v).strip().lower().replace(".", "p").replace("-", "_")
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s)[:24]


def _numeric_values(states: Sequence[Mapping[str, Any]], fname: str) -> list[float]:
    out = []
    for s in states:
        v = s.get(fname)
        if isinstance(v, bool) or v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def _quantile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    i = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[i]


def atoms_for(fields: Mapping[str, Any], states: Sequence[Mapping[str, Any]]) -> list[Atom]:
    """Enumerate stage-1 and stage-2 atoms from TYPED declarations plus observed values.

    `fields` maps name -> a declaration carrying `.type` and optionally `.enum`. Anything the adapter
    did not declare is not nameable, which is what keeps the grammar inside the host's real surface.
    """
    out: list[Atom] = []
    for fname, decl in sorted(fields.items()):
        ftype = getattr(decl, "type", None)
        enum = tuple(getattr(decl, "enum", ()) or ())
        if ftype is bool:
            out += [Atom(fname, "truthy"), Atom(fname, "falsy")]
            continue
        if enum:
            # EVERY OBSERVED VALUE IS NAMEABLE. Cost is controlled downstream, never by deleting
            # candidates from the alphabet.
            #
            # Two earlier versions were both wrong, for the same underlying reason. The first sliced
            # `enum` directly, so the budget selected by the order someone typed the values in: on one
            # host's 6-value refusal-class field it kept the value observed 4 times and dropped the one
            # observed 124 times. Ranking by observed frequency fixed that particular inversion but
            # kept the deeper error -- a RARE value can be the most consequential one. On that same
            # field the 4-occurrence value was linked to 50 downstream failures, more than the
            # 301-occurrence value's 43, because frequency at this boundary says nothing about how much
            # loss the condition explains.
            #
            # A value that is unnameable is not rejected-with-a-reason; it is absent from the search
            # record entirely, so the loop reports a closed question it never asked. That asymmetry --
            # a silent omission looks exactly like an exhausted search -- is why this is the one place
            # the alphabet must be complete.
            #
            # `states` is still consulted, for two reasons that are about EVIDENCE, not budget:
            #   * an UNOBSERVED value is dropped -- `x == c` where no state carries `c` is constant
            #     False on this residual, which `synthesize` would discard anyway; and offering it
            #     would let a predicate validate on a value this residual cannot exhibit;
            #   * the order is most-observed first, so when a downstream budget truncates, the values
            #     that survive are the ones this residual can actually discriminate on.
            #
            # Where cost IS controlled: `synthesize`'s own `_separates` filter (constants are dropped),
            # MAX_CANDIDATES on the returned list, the `keep[:12]` conjunction window, and -- the real
            # lever -- residual-driven candidate selection and the evaluation budget upstream. A wide
            # alphabet costs CPU in enumeration; a missing atom costs a mechanism.
            seen = Counter(s.get(fname) for s in states if s.get(fname) is not None)
            observed = [v for v in enum if seen.get(v, 0)]
            for v in sorted(observed, key=lambda v: (-seen[v], enum.index(v))):
                out.append(Atom(fname, "equals", v))
            if not observed:
                # NOTHING OBSERVED: the states cannot tell us which values matter, so fall back to the
                # declaration rather than emitting no equality atom at all. A synthetic or probe state
                # set that carries the field but none of its values must still be searchable.
                out += [Atom(fname, "equals", v) for v in enum]
            # "this field carries nothing" is a distinct, often load-bearing condition
            out.append(Atom(fname, "falsy"))
            continue
        if ftype in (int, float):
            vals = _numeric_values(states, fname)
            if not vals:
                continue
            seen = set()
            for q in QUANTILES:
                t = round(_quantile(vals, q), 6)
                if t in seen:
                    continue
                seen.add(t)
                out += [Atom(fname, "lt", t), Atom(fname, "gt", t)]
            continue
        # unknown/opaque type: presence is still meaningful
        out += [Atom(fname, "truthy"), Atom(fname, "falsy")]
    return out


def _separates(pred, states: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    fired = sum(1 for s in states if pred.evaluate(s))
    return fired, len(states)


def synthesize(fields: Mapping[str, Any], states: Sequence[Mapping[str, Any]], *,
               declared: Sequence[str] = (), max_candidates: int = MAX_CANDIDATES,
               allow_conjunctions: bool = True) -> list[Any]:
    """Staged search over the grammar. Returns DISCRIMINATING predicates, simplest first.

    A predicate that is constant on `states` is dropped here rather than passed on: it carries no
    information, and installing one adds a name to Phi that fires everywhere or nowhere.

    `declared` names signals Phi already has, so their exact single-field equivalents are not re-offered
    -- expansion should add what is missing, not duplicate what exists.
    """
    if not states:
        return []
    atoms = atoms_for(fields, states)
    keep: list[Any] = []
    for a in atoms:
        fired, total = _separates(a, states)
        if 0 < fired < total:
            keep.append(a)
    keep.sort(key=lambda a: (abs(_separates(a, states)[0] / len(states) - 0.5), a.name()))

    out: list[Any] = [a for a in keep if a.name() not in set(declared)][:max_candidates]
    if not allow_conjunctions:
        return out

    # STAGE 3. Conjunctions are formed from atoms that individually discriminate but over DIFFERENT
    # fields -- pairing two predicates on one field yields a narrower band, not a new condition.
    pairs: list[Conjunction] = []
    for a, b in itertools.combinations(keep[:12], 2):
        if a.field == b.field:
            continue
        c = Conjunction((a, b))
        fired, total = _separates(c, states)
        if 0 < fired < total:
            pairs.append(c)
    pairs.sort(key=lambda c: (-_separates(c, states)[0], c.name()))
    return (out + pairs)[:max_candidates]
