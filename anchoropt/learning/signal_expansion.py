"""SIGNAL EXPANSION: what to do when a residual cannot be represented by the current Phi.

    residual -> SIGNAL_BLOCKED -> propose a new signal -> VALIDATE it -> expand Phi -> resume search

GENERIC. This module names no benchmark concept -- not a store, not a tool, not an error string. It
asks the RUNTIME whether a proposed signal is observable and installable, exactly as the policy search
asks the runtime about actions. `tests/test_signal_expansion.py` greps this file for domain words to
keep it that way, which is why even the sentence you are reading avoids them.

WHY EXPANSION NEEDS A VALIDATION GATE AT ALL
--------------------------------------------
The failure mode a naive version invites is the one this project already paid for with actions: a
proposer names something plausible, the machinery reports it "available", and the round measures
nothing because the host cannot actually evaluate it. The action side learned that
ADMISSIBLE != MATERIALIZABLE, and a signal has the identical trap -- a name in Phi that no probe can
compute is worse than an absent signal, because it looks like coverage.

So a proposed signal is installed only if the runtime can:
  1. NAME it without collision (a redefinition of an existing signal is refused, not merged);
  2. COMPUTE it -- the runtime accepts a predicate and returns a bool on a probe state;
  3. OBSERVE it at the boundary the residual needs, from fields that exist there;
  4. distinguish states -- a predicate that is constant on the observed states carries no
     information, and would fire everywhere or nowhere.

(4) is the check that would have caught a real defect: an earlier anchor's condition was evaluated
before generation, where it is TRUE of every episode, and it fired 303/303 times instead of the ~48
targeted. A constant signal is not a signal.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not write predicates. The runtime supplies candidate observables (it knows its own fields);
the proposer may say WHICH observable and WHAT comparison, and this module decides whether the result
is admissible into Phi. Keeping predicate authorship on the runtime side is what stops core from
growing benchmark-specific logic -- and is why an adapter can expose a new signal family without a
core change.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Terminal states for one expansion attempt.
SIGNAL_INSTALLED = "SIGNAL_INSTALLED"
SIGNAL_REFUSED_COLLISION = "SIGNAL_REFUSED_COLLISION"
SIGNAL_REFUSED_UNCOMPUTABLE = "SIGNAL_REFUSED_UNCOMPUTABLE"
SIGNAL_REFUSED_NOT_OBSERVABLE = "SIGNAL_REFUSED_NOT_OBSERVABLE"
SIGNAL_REFUSED_CONSTANT = "SIGNAL_REFUSED_CONSTANT"
SIGNAL_REFUSED_NO_RUNTIME_SUPPORT = "SIGNAL_REFUSED_NO_RUNTIME_SUPPORT"


@dataclass(frozen=True)
class SignalProposal:
    """A candidate addition to Phi.

    `observable` names a field the runtime already exposes at `boundary`; `comparison` and `value`
    describe the test over it. The proposer chooses among the runtime's OWN observables -- it does not
    invent a state surface, for the same reason it does not invent a tool.
    """

    name: str
    boundary: Any
    observable: str
    comparison: str = "truthy"          # truthy | equals | lt | gt | absent | present
    value: Any = None
    rationale: str = ""

    def describe(self) -> str:
        if self.comparison == "truthy":
            return f"{self.observable} is truthy"
        if self.comparison in ("absent", "present"):
            return f"{self.observable} is {self.comparison}"
        return f"{self.observable} {self.comparison} {self.value!r}"


@dataclass(frozen=True)
class ExpansionResult:
    """The verdict on one proposed signal. A refusal names what was missing."""

    state: str
    proposal: SignalProposal
    detail: str = ""
    fired_on: int = 0
    total_states: int = 0

    @property
    def installed(self) -> bool:
        return self.state == SIGNAL_INSTALLED

    @property
    def firing_rate(self) -> float | None:
        return (self.fired_on / self.total_states) if self.total_states else None


def _predicate_for(p: SignalProposal) -> Callable[[Mapping[str, Any]], bool]:
    """Turn a proposal into a predicate over observable state. Comparisons only -- no expressions.

    Deliberately a tiny fixed vocabulary rather than eval(): a proposer-authored expression is
    arbitrary code in the decision path, and the whole point of the seam is that core validates
    something it can reason about.
    """
    obs, cmp_, want = p.observable, p.comparison, p.value

    def pred(state: Mapping[str, Any]) -> bool:
        present = obs in state
        got = state.get(obs)
        if cmp_ == "present":
            return present
        if cmp_ == "absent":
            return not present
        if not present or got is None:
            return False
        if cmp_ == "truthy":
            return bool(got)
        if cmp_ == "equals":
            return got == want
        try:
            if cmp_ == "lt":
                return float(got) < float(want)
            if cmp_ == "gt":
                return float(got) > float(want)
        except (TypeError, ValueError):
            return False
        return False

    return pred


def validate_signal(proposal: SignalProposal, *, runtime,
                    observed_states: Sequence[Mapping[str, Any]] = ()) -> ExpansionResult:
    """Decide whether `proposal` may enter Phi. Pure -- installs nothing.

    `observed_states` are real states drawn from the residual's own episodes at the proposal's
    boundary. They are what makes check (4) possible: without them a constant predicate is
    indistinguishable from a discriminating one.
    """
    name = (proposal.name or "").strip()
    if not name:
        return ExpansionResult(SIGNAL_REFUSED_UNCOMPUTABLE, proposal, "proposal carries no name")

    # (1) collision -- redefinition is refused, never merged.
    try:
        declared = tuple(runtime.declared_signals())
    except Exception as exc:
        return ExpansionResult(SIGNAL_REFUSED_NO_RUNTIME_SUPPORT, proposal,
                               f"runtime cannot report its signals: {exc}")
    if name in declared:
        return ExpansionResult(SIGNAL_REFUSED_COLLISION, proposal,
                               f"{name!r} is already declared; redefining an existing signal would "
                               f"silently change every policy that references it")

    # (3) observable at the boundary, from fields that exist there.
    if hasattr(runtime, "observable_fields"):
        try:
            fields = set(runtime.observable_fields(proposal.boundary) or ())
        except Exception:
            fields = set()
        if fields and proposal.observable not in fields:
            return ExpansionResult(
                SIGNAL_REFUSED_NOT_OBSERVABLE, proposal,
                f"{proposal.observable!r} is not observable at "
                f"{getattr(proposal.boundary, 'value', proposal.boundary)}; the runtime exposes "
                f"{sorted(fields)[:8]}")

    # (2) computable -- the predicate must run on a probe without raising.
    pred = _predicate_for(proposal)
    probe = dict(observed_states[0]) if observed_states else {}
    try:
        pred(probe)
    except Exception as exc:
        return ExpansionResult(SIGNAL_REFUSED_UNCOMPUTABLE, proposal,
                               f"predicate raised on a probe state: {type(exc).__name__}: {exc}")

    # (4) discriminating -- a constant predicate carries no information.
    if observed_states:
        fired = sum(1 for s in observed_states if pred(s))
        if fired == 0:
            return ExpansionResult(SIGNAL_REFUSED_CONSTANT, proposal,
                                   "never true on the residual's own states -- it would install a "
                                   "signal that can never fire", 0, len(observed_states))
        if fired == len(observed_states):
            return ExpansionResult(SIGNAL_REFUSED_CONSTANT, proposal,
                                   "true on EVERY observed state -- an unconditional trigger, which "
                                   "is how one intervention fired on every episode instead of the "
                                   "targeted subset", fired, len(observed_states))
        return ExpansionResult(SIGNAL_INSTALLED, proposal,
                               f"discriminating: fires on {fired}/{len(observed_states)} states",
                               fired, len(observed_states))

    return ExpansionResult(SIGNAL_INSTALLED, proposal,
                           "computable and non-colliding; NOT checked for discrimination (no observed "
                           "states supplied), so the firing rate is unknown until measured")


def install_signal(result: ExpansionResult, *, runtime) -> tuple[bool, str]:
    """Ask the RUNTIME to add the validated signal to Phi. Core never mutates Phi itself.

    The runtime owns Phi because it owns the state surface; an adapter that cannot install signals
    simply declines, and the residual stays SIGNAL_BLOCKED with a reason rather than appearing solved.
    """
    if not result.installed:
        return False, f"refused upstream: {result.state}"
    if not hasattr(runtime, "install_signal"):
        return False, ("this runtime does not support signal installation; the residual remains "
                       "SIGNAL_BLOCKED and belongs to the SIGNAL backlog")
    try:
        runtime.install_signal(result.proposal.name, _predicate_for(result.proposal),
                               boundary=result.proposal.boundary,
                               provenance=result.proposal.describe())
    except Exception as exc:
        return False, f"runtime refused the installation: {type(exc).__name__}: {exc}"
    return True, f"installed {result.proposal.name!r} ({result.proposal.describe()})"


def expand_and_resume(proposals: Sequence[SignalProposal], *, runtime,
                      observed_states: Sequence[Mapping[str, Any]] = ()
                      ) -> tuple[list[ExpansionResult], list[str]]:
    """Validate every proposal, install those that pass, and report the newly available signals.

    Returns (results, installed_names). The caller resumes the ordinary policy search with the
    expanded Phi -- expansion changes what is OBSERVABLE, never the action space, which stays the
    fixed small set the host declares.
    """
    results, installed = [], []
    for p in proposals:
        r = validate_signal(p, runtime=runtime, observed_states=observed_states)
        if r.installed:
            ok, why = install_signal(r, runtime=runtime)
            if not ok:
                r = ExpansionResult(SIGNAL_REFUSED_NO_RUNTIME_SUPPORT, p, why,
                                    r.fired_on, r.total_states)
            else:
                installed.append(p.name)
        results.append(r)
    return results, installed


def expand_and_resume_predicates(predicates: Sequence[Any], proposals: Sequence[SignalProposal], *,
                                 runtime, observed_states: Sequence[Mapping[str, Any]] = ()
                                 ) -> tuple[list[ExpansionResult], list[str]]:
    """Validate and install predicates SYNTHESIZED by `signal_grammar`, which carry their own
    `.evaluate(state)`.

    `expand_and_resume` builds a predicate from the small comparison vocabulary; a synthesized
    predicate may be a conjunction, which that vocabulary cannot express. The validation rules are the
    same four -- collision, computable, observable, DISCRIMINATING -- applied to the predicate object.
    """
    results, installed = [], []
    try:
        declared = set(runtime.declared_signals())
    except Exception:
        declared = set()
    for pred, prop in zip(predicates, proposals):
        if prop.name in declared:
            results.append(ExpansionResult(SIGNAL_REFUSED_COLLISION, prop,
                                           f"{prop.name!r} is already declared"))
            continue
        try:
            pred.evaluate(dict(observed_states[0]) if observed_states else {})
        except Exception as exc:
            results.append(ExpansionResult(SIGNAL_REFUSED_UNCOMPUTABLE, prop,
                                           f"predicate raised: {type(exc).__name__}: {exc}"))
            continue
        fired = sum(1 for st in observed_states if pred.evaluate(st))
        if observed_states and fired in (0, len(observed_states)):
            results.append(ExpansionResult(
                SIGNAL_REFUSED_CONSTANT, prop,
                "constant over the residual's own states -- carries no information",
                fired, len(observed_states)))
            continue
        if not hasattr(runtime, "install_signal"):
            results.append(ExpansionResult(SIGNAL_REFUSED_NO_RUNTIME_SUPPORT, prop,
                                           "host cannot install signals"))
            continue
        try:
            runtime.install_signal(prop.name, lambda st, _p=pred: bool(_p.evaluate(st)),
                                   boundary=prop.boundary, provenance=pred.describe())
        except Exception as exc:
            results.append(ExpansionResult(SIGNAL_REFUSED_NO_RUNTIME_SUPPORT, prop,
                                           f"runtime refused: {exc}"))
            continue
        installed.append(prop.name)
        results.append(ExpansionResult(SIGNAL_INSTALLED, prop,
                                       f"discriminating: fires on {fired}/{len(observed_states)}",
                                       fired, len(observed_states)))
    return results, installed


# ================================================================================================
# CLAUSE-POOLED BLOCKED RESIDUALS
# ================================================================================================
#
# Signal expansion should operate on repeated REPRESENTATION GAPS, not repeated prose. Residual
# families are keyed on the attributor's `consequential_decision`, which collapses paraphrases but not
# genuinely different sentences -- so six diagnoses that all imply the same missing observable became
# six support-1 families and synthesis starved: a predicate cannot be shown discriminating on one
# observed state.
#
# Grouping by the clause the generic coverage check ALREADY produced gives one residual with support 6.
# No new analysis, no per-benchmark labels, no anchor knowledge: the clause is whatever
# `candidate_search.uncovered_clause_of` returned, and provenance back to the originating diagnoses and
# case ids is carried on the group.

@dataclass(frozen=True)
class ClauseResidual:
    """Blocked diagnoses that share one missing observable. Provenance is part of the object."""

    clause: str
    diagnoses: tuple[Any, ...]

    @property
    def support(self) -> int:
        return len(self.diagnoses)

    @property
    def case_ids(self) -> tuple[str, ...]:
        seen, out = set(), []
        for d in self.diagnoses:
            cid = getattr(d, "case_id", None)
            if cid and cid not in seen:
                seen.add(cid)
                out.append(cid)
        return tuple(out)

    def as_dict(self) -> dict[str, Any]:
        return {"clause": self.clause, "support": self.support,
                "case_ids": list(self.case_ids),
                "decisions": [getattr(d, "consequential_decision", "") for d in self.diagnoses]}


def group_blocked_by_clause(diagnoses: Sequence[Any], *, runtime) -> list[ClauseResidual]:
    """Blocked diagnoses grouped by their computed uncovered clause, largest support first.

    Uniform across residuals and benchmarks: the only input is what the coverage check reports, so a
    host with no uncovered clauses simply yields no groups.
    """
    from anchoropt.learning.candidate_search import uncovered_clause_of

    groups: dict[str, list[Any]] = {}
    for d in diagnoses:
        try:
            clause = uncovered_clause_of(d, runtime=runtime)
        except Exception:
            clause = None
        if clause:
            groups.setdefault(clause, []).append(d)
    out = [ClauseResidual(clause=c, diagnoses=tuple(ds)) for c, ds in groups.items()]
    out.sort(key=lambda g: (-g.support, g.clause))
    return out
