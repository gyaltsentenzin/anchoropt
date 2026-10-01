"""Block-coordinate optimization over decision boundary, observable representation, intervention.

    residual failure
      -> WHERE  localize: the consequential decision boundaries the trajectory CONTAINS
      -> WHAT   is the condition expressible under Phi here? if not, expand Phi HERE
      -> HOW    which action + eta can a runtime primitive actually realize?
      -> evaluate (when an evaluator is available)
      -> promote -> regenerate trajectories -> re-mine the residual

This module is the algorithm. It owns the SCHEDULE and nothing else: every step it takes is an
existing core function, and every fact about the benchmark comes through the runtime contract. The
only thing that is new here is the order in which the three coordinates move, which is the claim.

THE ORDERING, which is the contribution
---------------------------------------
At a fixed WHERE:

  1. optimize the EXISTING WHAT x HOW first -- the cheapest move is the one that needs no new
     representation;
  2. if candidates are realizable but none improves the incumbent, exhaust the remaining HOW/eta
     choices before touching anything else;
  3. only then EXPAND WHAT, at this same boundary;
  4. retry HOW with the widened Phi, still at this boundary;
  5. move WHERE earlier ONLY when the boundary is structurally unrepairable, or its WHAT/HOW search
     is exhausted.

Backward over WHERE (latest boundary first) because a repair applied where the failure is actually
observed is better attributed than one applied earlier: the earlier intervention has to be right
about everything that happens after it, while the later one sees the outcome. `search_state.
NEXT_COORDINATE` encodes the whole schedule as data, and `backward_boundary_search` is the driver --
this module supplies its hooks rather than re-implementing the loop.

UNMEASURED IS NOT UNSUCCESSFUL. With `evaluate=None` the search still localizes, expands and grounds,
and stops at `REALIZABLE_UNMEASURED`. That state is structural rediscovery: a controller was built and
a primitive can run it. It is deliberately NOT `NO_BENEFIT`, which asserts a measurement.

WHAT THE ADAPTER OWES, and nothing more
---------------------------------------
    is_decision(event) / boundary_key(event)   which events are decisions, and which point they are
    declared_signals()                        the current Phi
    synthesis_fields(boundary)                typed observables readable AT that boundary
    install_signal(name, pred, boundary=, provenance=)
    HOST.executable_actions(boundary)         the action space there
    evaluate_signal / probe_params            signal evaluation
    (via AnchorPolicyOpt) eta grounding and executor feasibility

No benchmark name, field, tool, status string or cell appears in this file, and none may.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anchoropt.learning.boundary_search import (
    BOUNDARY_EXHAUSTED as _BS_EXHAUSTED,
    Boundary, boundaries_of, order_boundaries,
)
from anchoropt.learning.policy_class import ThetaResult, train_objective
from anchoropt.learning.learned_signal import (
    adds_information, dedupe_by_behaviour, learned_from,
)
from anchoropt.learning.search_state import (
    ACTION_UNAVAILABLE, BOUNDARY_NOT_REPAIRABLE, DONE, HOW, IMPROVED, NO_BENEFIT, NO_CANDIDATE,
    PROMOTE, REALIZABLE_UNMEASURED, SIGNAL_BLOCKED, SIGNAL_EXPANDED, SIGNAL_EXPANSION_EXHAUSTED,
    WHAT, WHERE, Attempt, Certificate, certificate_from, next_coordinate,
    SEARCH_BUDGET_EXHAUSTED, UNFIREABLE_HERE, ALL_CELLS_ALREADY_DECIDED,
)

MAX_SYNTHESIS_CANDIDATES = 40      # must leave room for CONJUNCTIONS, which stage after atoms


@dataclass
class SearchOutcome:
    """Everything the schedule did, in enough detail to explain why it stopped where it did."""

    boundaries: tuple[str, ...] = ()
    attempts: list[Attempt] = field(default_factory=list)
    candidates: list[Any] = field(default_factory=list)
    certificates: list[Certificate] = field(default_factory=list)
    signals_installed: list[str] = field(default_factory=list)
    promoted: Any = None
    state: str = ""
    restarts: int = 0
    # WHICH REPRESENTATION THIS ROUND SEARCHED, and where it came from. A round that searched the
    # residual family's own observables and one that enumerated the whole grammar are different
    # experiments, and a result table that cannot tell them apart cannot support a claim about
    # attribution constraining the space.
    #   "residual_family"   the family's grounded observables (the invariant)
    #   "caller"            an explicit `signals=` argument, e.g. an ablation
    #   "declared"          the family named none, so the host's declared set
    #   "declared_fallback" the family named some, but the host declares NONE of them
    phi_source: str = ""
    phi_seeded: tuple[str, ...] = ()
    # Observables the family named that the runtime does not declare. Never silently dropped: an
    # attributor naming a condition the adapter cannot evaluate is a finding about the adapter.
    phi_dropped: tuple[str, ...] = ()
    # COMPLETED evaluations this search spent, and the budget it was given. Reported so a reader can
    # tell a search that measured nothing from one that measured everything and found nothing.
    evaluations_completed: int = 0
    budget: int | None = None
    # Boundaries swept under the SEEDED Phi after a measured no-benefit, before any expansion. The
    # provenance of "why this candidate": a WHERE move under a fixed representation, not a new one.
    frontier_boundaries: tuple[str, ...] = ()
    # Boundaries visited by the NO-EVALUATOR localization sweep: the seeded Phi is observable there and
    # the round collected its candidates. Kept distinct from `moves_earlier`, which counts the schedule
    # GIVING UP on a boundary -- sweeping a boundary the proposer localized is not giving up on
    # anything, and conflating the two would make a completed localization look like an exhausted one.
    localization_sweep: tuple[str, ...] = ()

    @property
    def visited(self) -> tuple[str, ...]:
        return tuple(a.boundary for a in self.attempts)

    @property
    def moves_earlier(self) -> int:
        """How many times the schedule gave up on a boundary and moved to an earlier one.

        COUNTED FROM THE OBSERVED TRANSITIONS, not from each attempt's terminal state. Counting
        `coordinate_changed == WHERE` undercounts: after Phi is widened at a boundary the last attempt
        there ends in NO_BENEFIT (-> HOW) or REALIZABLE_UNMEASURED (-> DONE), so a search that visibly
        moved from one boundary to an earlier one reported 0. A metric that cannot observe the movement
        it exists to measure is the same defect, one level up, as a null from a channel that was never
        live -- so this reads the boundary sequence the attempts actually record.

        With evaluation disabled this counts STRUCTURAL movement -- "earlier movement was required to
        obtain a realizable controller" -- and says nothing about which boundary is better.
        """
        # The localization sweep is EXCLUDED. Those attempts were not the schedule giving up on a
        # boundary; they are the round collecting candidates at boundaries the proposer had already
        # localized. Including them made a completed localization report structural movement it never
        # performed -- and `moves_earlier` is the algorithmic claim, so it must keep its meaning.
        swept = set(self.localization_sweep)
        seq = [a.boundary for a in self.attempts if a.boundary not in swept]
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y)

    @property
    def expanded(self) -> bool:
        return bool(self.signals_installed)

    @property
    def rediscovered(self) -> bool:
        """A controller was built AND a primitive can realize it. Structure only, never benefit."""
        return bool(self.candidates)

    @property
    def validated(self) -> bool:
        return self.state == IMPROVED

    def as_dict(self) -> dict[str, Any]:
        return {"boundaries": list(self.boundaries), "visited": list(self.visited),
                "moves_earlier": self.moves_earlier, "state": self.state,
                "attempts": [a.as_dict() for a in self.attempts],
                "candidates": len(self.candidates),
                "certificates": [c.as_dict() for c in self.certificates],
                "signals_installed": list(self.signals_installed),
                # WHICH REPRESENTATION WAS SEARCHED. Recorded because a round seeded from the
                # residual family and one that enumerated the whole grammar are different
                # experiments, and the artifact must be able to say which one ran.
                "phi_source": self.phi_source, "phi_seeded": list(self.phi_seeded),
                "phi_dropped": list(self.phi_dropped),
                "evaluations_completed": self.evaluations_completed, "budget": self.budget,
                "frontier_boundaries": list(self.frontier_boundaries),
                "expanded": self.expanded, "rediscovered": self.rediscovered,
                "validated": self.validated, "restarts": self.restarts,
                "promoted": (self.promoted.as_dict()
                             if hasattr(self.promoted, "as_dict") else self.promoted)}


def localize(events: Sequence[Mapping[str, Any]], *, runtime) -> list[Boundary]:
    """WHERE: the ordered decision boundaries this trajectory contains, earliest to latest.

    Derived, never declared. The adapter says which events are decisions and what point each belongs
    to; core orders them -- by the adapter's dependency relation if it offers one, otherwise by the
    order they actually happened in. A recurring boundary collapses to its first occurrence.
    """
    return order_boundaries(boundaries_of(events, adapter=runtime), adapter=runtime)


def _boundary_arg(boundary: Boundary, runtime) -> Any:
    """The boundary in whatever form this runtime's own API accepts.

    The adapter's `boundary_key` produced the key, so the adapter is also what converts it back.
    Core does not own a boundary->locus table: that would be the semantic stage map, reintroduced.
    """
    coerce = getattr(runtime, "boundary_from_key", None)
    if callable(coerce):
        try:
            got = coerce(boundary.key)
            if got is not None:
                return got
        except Exception:
            pass
    return boundary.key


def _actions_at(boundary: Any, *, host) -> tuple[Any, ...]:
    try:
        acts = tuple(host.executable_actions(boundary))
    except Exception:
        return ()
    return tuple(a for a in acts if str(getattr(a, "value", a)).lower() != "noop")


def _expand_phi_at(boundary: Any, *, runtime, states: Sequence[Mapping[str, Any]],
                   existing: Sequence[Any]) -> tuple[list[str], list[Any]]:
    """WHAT: synthesize predicates over the fields readable AT THIS BOUNDARY, install what is new.

    Two filters, both behavioural rather than syntactic:
      * `dedupe_by_behaviour` -- a threshold grid yields many names and few distinct firing sets, and
        searching the duplicates spends evaluations that cannot differ in outcome;
      * `adds_information` -- a constant predicate, or one whose firing set an existing signal
        already realizes, does not widen Phi no matter what it is called.
    """
    from anchoropt.learning.signal_expansion import expand_and_resume_predicates
    from anchoropt.learning.signal_grammar import synthesize
    from anchoropt.learning.signal_expansion import SignalProposal

    fields = {}
    if hasattr(runtime, "synthesis_fields"):
        fields = runtime.synthesis_fields(boundary) or {}
    # STATES MUST BELONG TO THIS BOUNDARY, AND THIS FAILS CLOSED.
    #
    # A predicate validated on states from a LATER boundary can look discriminating on information
    # that does not exist yet -- which is exactly how an upstream candidate came to be selected on a
    # post-execution field. So when a runtime declares that a boundary has its own information set
    # (`states_at`), a failure to construct it is NOT a reason to fall back to whatever states the
    # caller happened to pass: falling back silently reintroduces the defect and looks like success.
    #
    # `fields` is non-empty only for a boundary the runtime declares searchable, so an absent
    # `states_at` on such a boundary is a contract gap and is raised. A runtime that declares no
    # fields here returns [] below and never reaches this.
    project = getattr(runtime, "states_at", None)
    if fields and not callable(project):
        raise RuntimeError(
            f"runtime declares {len(fields)} synthesis field(s) at {boundary} but no states_at(): "
            f"the boundary's information set cannot be constructed, and validating predicates on "
            f"another boundary's states is the defect this check exists to prevent")
    if fields:
        projected = project(boundary, states)          # exceptions propagate: fail closed
        if not projected:
            raise RuntimeError(
                f"states_at({boundary}) produced no states: the boundary's information set is empty, "
                f"so no predicate can be validated there. This is a contract gap, not an exhausted "
                f"search -- returning [] here would report SIGNAL_EXPANSION_EXHAUSTED for a boundary "
                f"that was never actually searched")
        states = projected
    if not fields or not states:
        return [], []
    declared = tuple(runtime.declared_signals()) if hasattr(runtime, "declared_signals") else ()
    preds = synthesize(fields, states, declared=declared,
                       max_candidates=MAX_SYNTHESIS_CANDIDATES)
    preds = dedupe_by_behaviour(preds, states)
    preds = [p for p in preds if adds_information(p, existing, states)]
    # Prefer predicates spanning MORE fields: expansion was triggered because a declared signal
    # already covers part of the condition, so a bare atom on the covered dimension does not close
    # the gap while a conjunction across two fields can. An ordering over what synthesis returned --
    # nothing is added, and an atom-only result is still returned unchanged.
    preds.sort(key=lambda p: (-len({t.field for t in getattr(p, "terms", ())} or {1}), p.name()))
    if not preds:
        return [], []
    props = [SignalProposal(name=p.name()[:60], boundary=boundary,
                            observable=getattr(p, "field", "(composite)"),
                            comparison="synthesized", value=None, rationale=p.describe())
             for p in preds]
    _results, installed = expand_and_resume_predicates(
        preds, props, runtime=runtime, observed_states=states)
    by_name = {p.name()[:60]: p for p in preds}
    return list(installed), [by_name[n] for n in installed if n in by_name]



def _cell_of_arm(arm: Any) -> tuple[str, str, str]:
    """THE CONTROLLER IDENTITY of a built arm: (boundary, signal, action).

    The same triple `candidate_search.cell_of` names for a scored candidate, read off a `PolicyArm`.
    Theta and eta are deliberately NOT part of it: a re-parameterization of a cell is a REVISION of
    one controller, governed by the registry's supersession rule, not a second controller to measure.

    Spelled here rather than imported from `candidate_search` because these are two independent
    selection paths -- this module's block-coordinate schedule and that module's scored enumeration --
    and coupling them would make a change to one silently alter the other. The identity is the same by
    DEFINITION, and `tests/test_restart_terminates.py` pins that the two agree.
    """
    return (arm.boundary.value, str(arm.signal), arm.action.value)


def _objective_key(obj: Any) -> Any:
    """A SORTABLE key for whatever the evaluator returned. Total over the shapes callers actually use.

    THE DEFECT THIS FIXES. These two lines read `if best is None or obj > best`, which works for a
    scalar and raises `TypeError` for a `ThetaResult` -- the very type the real optimizer produces and
    `train_objective` is written to rank. So the only evaluator the schedule could run end to end was a
    scalar stub, and the toy-host example crashed on its first REAL paired evaluation.

    `ThetaResult` is ranked by `train_objective`, which is the project's J_train (net, then ENGAGEMENT,
    then selectivity, then a stable tiebreak) -- reusing it rather than inventing a second ordering is
    what keeps "which arm is best" a single definition. Anything else is compared as itself, so an
    integer- or float-returning evaluator behaves exactly as before.
    """
    if isinstance(obj, ThetaResult):
        return train_objective(obj)
    return obj

def optimize_residual(residual: Any, *, runtime, host, events: Sequence[Mapping[str, Any]],
                      states: Sequence[Mapping[str, Any]] = (),
                      signals: Sequence[str] = (),
                      evaluate: Callable[[Any], Any] | None = None,
                      improves: Callable[[Any], bool] | None = None,
                      promote: Callable[[Any], None] | None = None,
                      remine: Callable[[], Any] | None = None,
                      repairable_at: Callable[[Any, Boundary], bool] | None = None,
                      max_restarts: int = 3,
                      decided_cells: Iterable[tuple[str, str, str]] = (),
                      eval_budget: int | None = None) -> SearchOutcome:
    """Run the WHERE -> WHAT -> HOW block-coordinate search over one residual.

    `events` are the residual's realized trajectory steps, as the adapter's normalized events.
    `states` are observable states for the discrimination check during expansion.
    `signals` is the current Phi to search before expanding; defaults to the runtime's declared set.

    `evaluate=None` means no evaluator: the search still localizes, expands and grounds, and the
    outcome stops at REALIZABLE_UNMEASURED -- structural rediscovery with no claim about benefit.
    """
    from anchoropt.learning.anchor_policy_opt import AnchorPolicyOpt, SearchSpaceProposal

    opt = AnchorPolicyOpt(runtime=runtime, host=host)
    out = SearchOutcome()
    # THE EVALUATION BUDGET, counted in COMPLETED evaluations and shared across every boundary and
    # every expansion in this search. Declining to spend more is not a finding about a controller, so
    # a refusal returns None -- the same value an evaluator uses for "I did not measure this" -- and
    # the outcome is classified BUDGET_EXHAUSTED rather than NO_BENEFIT.
    spent = {"n": 0, "exhausted": False}

    def _eval(arm):
        if evaluate is None:
            return None
        if eval_budget is not None and spent["n"] >= int(eval_budget):
            spent["exhausted"] = True
            return None
        obj = evaluate(arm)
        # Count COMPLETED evaluations only: an evaluator declining to score an arm has not spent
        # budget on it, and counting the attempt would let a run of unmeasurable arms exhaust a
        # budget without measuring anything.
        if obj is not None:
            spent["n"] += 1
        return obj

    def _finish(o: SearchOutcome) -> SearchOutcome:
        """Stamp the budget facts on the way out. One place, so no exit can forget them.

        BUDGET IS NOT A NULL. If spending stopped with candidates still unmeasured, the outcome is
        BUDGET_EXHAUSTED -- a statement about what we were willing to spend -- and never NO_BENEFIT,
        which claims every candidate was measured and none helped.
        """
        o.evaluations_completed = spent["n"]
        o.budget = eval_budget
        if spent["exhausted"] and o.state != IMPROVED:
            o.state = SEARCH_BUDGET_EXHAUSTED
        return o

    ordered = localize(events, runtime=runtime)
    out.boundaries = tuple(b.key for b in ordered)
    if not ordered:
        out.state = BOUNDARY_NOT_REPAIRABLE
        out.attempts.append(Attempt(boundary="", state=BOUNDARY_NOT_REPAIRABLE,
                                    detail="the trajectory contains no consequential decision"))
        return _finish(out)

    # ------------------------------------------------------------------------------------------
    # THE INVARIANT: ATTRIBUTION CONSTRAINS THE DECISION SPACE.
    #
    # If the residual family names grounded observables, those ARE the initial representation. The
    # search starts from what attribution established about the failure, not from every atom the
    # host's grammar can spell.
    #
    # Measured on a real round before this held: attribution pooled 24 vector diagnoses into a family
    # whose observables were [append_would_exceed_cap, clear_proposed_at_capacity,
    # container_at_capacity, container_slots_exhausted] -- and the search then enumerated 51 grammar
    # atoms over every declared field, emitting 117 arms of which exactly ONE named an observable the
    # family had identified, and the family's top observable produced NO arm at all. The grouping was
    # computed and then discarded, so the search optimized a space the residual had not implicated.
    #
    # `signals=` still wins when a caller passes it explicitly: an ablation must be able to hand the
    # search a representation of its choosing. And EXPANSION IS UNTOUCHED -- a seeded Phi that yields
    # no beneficial controller falls through to the same expand-Phi step as before, which is what
    # keeps a seed from becoming a cage. `phi_seeded` is recorded so a reader can tell which
    # representation a round actually searched.
    family_phi = tuple(str(o) for o in (getattr(residual, "observables", ()) or ()))
    declared = tuple(runtime.declared_signals()) if hasattr(runtime, "declared_signals") else ()
    if signals:
        base_phi, out.phi_source = tuple(signals), "caller"
    elif family_phi:
        # Intersect with what the host declares: an observable the family named but the runtime
        # cannot evaluate is not a signal, and carrying it would fail at the first evaluate_signal.
        seeded = tuple(o for o in family_phi if not declared or o in declared)
        dropped = tuple(o for o in family_phi if declared and o not in declared)
        if seeded:
            base_phi, out.phi_source = seeded, "residual_family"
            out.phi_dropped = dropped
        else:
            # The family named observables and NONE is declared. That is a real finding about the
            # adapter, not a reason to silently search everything: fall back, and say so.
            base_phi, out.phi_source = declared, "declared_fallback"
            out.phi_dropped = dropped
    else:
        base_phi, out.phi_source = declared, "declared"
    out.phi_seeded = tuple(base_phi)
    case_ids = tuple(getattr(residual, "case_ids", ()) or ())
    improves = improves or (lambda obj: bool(obj) and getattr(obj, "net", obj) > 0)

    def _build_and_ground(boundary_arg, phi_names, att) -> tuple[list[Any], bool]:
        """HOW at a fixed (WHERE, WHAT): build arms, keep certificates, report if any is realizable.

        `build_arms` already emits a typed rejection for every refusal, so nothing here invents a
        second rejection mechanism -- the certificates are CLASSIFIED and kept.
        """
        actions = _actions_at(boundary_arg, host=host)
        if not actions:
            att.certificates.append(Certificate(state=ACTION_UNAVAILABLE, boundary=str(boundary_arg),
                                                detail="no executable action at this boundary"))
            return [], False
        built: list[Any] = []
        for phi in phi_names:
            try:
                sp = SearchSpaceProposal(boundary=boundary_arg, signal=phi,
                                         action_set=tuple(actions),
                                         diagnosis_case_ids=case_ids,
                                         rationale="block-coordinate search")
            except ValueError:
                continue
            arms, rejected = opt.build_arms(sp)
            for r in rejected:
                att.certificates.append(certificate_from(
                    r, boundary=str(getattr(boundary_arg, "value", boundary_arg)), signal=phi))
            built.extend(arms)
        att.candidates_built += len(built)
        att.candidates_realizable += len(built)

        # ---- STRUCTURAL FIREABILITY: measurable here, or not measurable here at all ---------------
        #
        # An arm whose phi fires on NONE of this boundary's own states has no paired contrast: running
        # it reproduces the control exactly. That is knowable WITHOUT an evaluator -- it is a property
        # of the predicate and the observed states, not of any outcome -- so checking it here is not
        # manufacturing a verdict, which is the thing this schedule must never do.
        #
        # WHY IT MATTERS FOR THE SCHEDULE. `elif realizable:` below halts the whole search at the
        # first boundary that produced a structurally realizable arm. Measured on a real round: the
        # proposer localized four grounded signals, three of them at a LATER boundary, and the single
        # arm built at the earlier one fired on 0 of 311 states. The round halted there, emitted
        # nothing, and never visited the boundary carrying three quarters of the localization. The
        # search reported REALIZABLE_UNMEASURED, which was true of that arm and false of the residual.
        #
        # FAIL OPEN. With no states supplied there is nothing to check and every arm is kept: a caller
        # that cannot project states must not have its candidates silently dropped.
        if states and built:
            fireable: list[Any] = []
            for arm in built:
                try:
                    if _fires_on_any(arm.signal, boundary_arg, runtime=runtime, states=states,
                                     params=dict(getattr(arm, "signal_params", {}) or {})):
                        fireable.append(arm)
                    else:
                        att.certificates.append(Certificate(
                            state=UNFIREABLE_HERE, boundary=str(boundary_arg), signal=arm.signal,
                            detail=("phi fires on none of this boundary's observed states, so the arm "
                                    "has no paired contrast and reproduces the control")))
                except Exception:
                    # NOT-EVALUABLE IS NOT NOT-FIREABLE. A predicate this core cannot evaluate here is
                    # kept: dropping it would silently prune on an infrastructure failure.
                    fireable.append(arm)
            if not fireable:
                # Every arm here is inert. The BOUNDARY is unmeasurable, not the residual.
                att.state = UNFIREABLE_HERE
                att.detail = (f"{len(built)} arm(s) built and grounded; none fires on any of this "
                              f"boundary's observed states, so none is measurable HERE. Not a "
                              f"statement about the residual or about benefit.")
                return [], False
            built = fireable
            att.candidates_realizable = len(built)
        return built, bool(built)

    # ---- the schedule: latest boundary first, moving earlier only when this one is finished ------
    i = len(ordered) - 1
    restarts = 0
    current = residual
    #: Controller identities (boundary, signal, action) this search has already DECIDED -- promoted
    #: OR measured-and-rejected. A cell is one controller however it is reached, so re-measuring it
    #: spends budget on a question already answered against this incumbent. This is the same contract
    #: `candidate_search.select_top_k` states within a round, held across the restart, and the same
    #: distinction `round_ledger` draws between a MEASURED_LOSER and a RETRYABLE: an arm the evaluator
    #: DECLINED to score is not in here, because no verdict was produced about it.
    #:
    #: Seeded by `decided_cells` so a caller can carry the memory between separate calls.
    #:
    #: TWO POPULATIONS, and conflating them disabled block-coordinate reoptimization. `installed`
    #: holds cells some round PROMOTED: one controller however it is reached, already in the
    #: incumbent, never re-measured. `rejected_here` holds cells this sweep MEASURED and did not
    #: promote -- true only against the incumbent they were scored against, which is why it is
    #: cleared on a re-mine. Keeping losers permanently made every restart inert: measured on the
    #: toy host as restarts=1 with ZERO arms evaluated against the new residual, because both
    #: boundaries reported ALL_CELLS_ALREADY_DECIDED from the previous incumbent's verdicts.
    #:
    #: This is the asymmetry `round_ledger` already draws between SETTLED and a within-round
    #: MEASURED_LOSER, held across the restart rather than re-derived.
    installed: set[tuple[str, str, str]] = {tuple(c) for c in decided_cells}
    rejected_here: set[tuple[str, str, str]] = set()
    installed_preds: list[Any] = []
    # SWEEP ONCE PER SEEDED Phi. Without this latch a NO_BENEFIT at each boundary would re-sweep the
    # others, cycling among boundaries and rejected controllers forever. A promotion resets it,
    # because a moved incumbent is a different residual and its frontier is genuinely new.
    frontier_swept = False
    #: Raised when this pass promoted an arm, so the sweep does not fall through into Phi expansion.
    #: NOT `out.state`: that is the verdict and must keep reporting IMPROVED across any number of
    #: restarts. One latch covers both cases -- a promotion carried OUT of the sweep, and a promotion
    #: that RESTARTS it -- because a restart resets `i` to the last boundary and the `not fresh` guard
    #: below then intercepts every already-decided boundary BEFORE the evaluation loop, so the resumed
    #: sweep never reaches expansion by that route either.
    promoted_this_pass = False

    #: EVERY promotion runs this, whichever sweep found the winner. There are three promotion sites
    #: -- the primary HOW sweep, the grounded FRONTIER sweep and the Phi-EXPANSION sweep -- and only
    #: the first used to consult `remine`. A winner found on either of the others returned
    #: immediately, so the incumbent moved, the trajectories changed, and the search never looked at
    #: the resulting residual. That silently collapsed the multi-round schedule to a single promotion
    #: exactly when the winner came from expansion, i.e. from a signal the core had to synthesize.
    #:
    #: Returns the re-mined residual to restart against, or None to finish. Recording the cell BEFORE
    #: `remine` runs is what stops a resumed sweep from re-measuring the controller it just installed.
    def _promoted(arm, obj) -> Any:
        nonlocal restarts, current, i, installed_preds, frontier_swept, promoted_this_pass
        installed.add(_cell_of_arm(arm))
        if promote is not None:
            promote(arm)
        out.promoted, out.state = arm, IMPROVED
        promoted_this_pass = True
        nxt = remine() if remine is not None else None
        if nxt is None or restarts >= max_restarts:
            return None
        # A promotion changed the trajectories, so every later decision now faces a different
        # residual: restart at the LATEST boundary rather than continue here.
        current, restarts = nxt, restarts + 1
        out.restarts = restarts
        i = len(ordered) - 1
        installed_preds = []
        frontier_swept = False
        # THE NEW INCUMBENT IS A NEW QUESTION. A cell this sweep measured and declined was declined
        # against the residual that has just been replaced, so it becomes eligible again -- its policy
        # or grounding may improve the incumbent that now exists. Cells already INSTALLED stay
        # excluded: those are settled, and re-measuring one would re-install what is already there.
        rejected_here.clear()
        return nxt

    while i >= 0:
        b = ordered[i]
        boundary_arg = _boundary_arg(b, runtime)

        if repairable_at is not None and not repairable_at(current, b):
            out.attempts.append(Attempt(boundary=b.key, state=BOUNDARY_NOT_REPAIRABLE, label=b.label,
                                        detail="the residual does not manifest at this decision"))
            i -= 1
            continue

        phi_here = [s for s in base_phi if _observable_at(s, boundary_arg, runtime=runtime)]
        att = Attempt(boundary=b.key, state=NO_CANDIDATE, label=b.label)

        # ---- STEP 1: existing WHAT x HOW ---------------------------------------------------------
        built, realizable = _build_and_ground(boundary_arg, phi_here, att)

        if att.state == UNFIREABLE_HERE:
            # ADVANCE THE BOUNDARY -- and do NOT fall through to Phi expansion.
            #
            # Every arm built here is inert, which is a fact about this boundary's information set and
            # NOT a measured insufficiency of the representation. Expansion's trigger is MEASURED
            # insufficiency; letting an unfireable boundary reach it re-enumerates the grammar off a
            # verdict nobody produced -- the failure that once turned 4 grounded observables into 117
            # arms. So the schedule tries the next boundary, where the localization's other signals
            # live, and the representation is left untouched.
            out.attempts.append(att)
            out.certificates.extend(att.certificates)
            i -= 1
            continue

        # ---- STEP 2/3: measured-but-no-benefit exhausts HOW; then expand WHAT --------------------
        if realizable and evaluate is not None:
            # Arms whose identity an EARLIER round already promoted are not re-measured: the cell is
            # one controller however it is reached, and it is already installed. Partitioned BEFORE
            # the loop so "everything here is already decided" is a state the schedule can dispatch
            # on rather than an empty loop that silently looks like an unmeasured boundary.
            fresh = [a for a in built
                     if _cell_of_arm(a) not in installed
                     and _cell_of_arm(a) not in rejected_here]
            att.cells_already_decided = len(built) - len(fresh)
            if not fresh:
                # ADVANCE THE BOUNDARY, and do NOT fall through to Phi expansion. Nothing was
                # measured here, so there is no measured insufficiency to expand on -- the same
                # discipline UNFIREABLE_HERE follows. Falling through instead rebuilt these identical
                # arms forever: measured as a non-terminating search once the restart resumed
                # correctly, with zero evaluations, so an evaluation cap could not even detect it.
                att.state = ALL_CELLS_ALREADY_DECIDED
                att.detail = (f"all {len(built)} arms name controller identities an earlier round "
                              f"already promoted")
                out.attempts.append(att)
                out.certificates.extend(att.certificates)
                i -= 1
                continue
            best = None
            for arm in fresh:
                obj = _eval(arm)
                att.candidates_evaluated += 1
                if obj is None:
                    # NOT decided: an evaluator that declined to score has produced no verdict, so
                    # the cell stays eligible. Recording it here would turn a budget refusal into a
                    # permanent rejection -- the same error as reading silence as a result.
                    continue
                # MEASURED, and therefore decided AGAINST THIS INCUMBENT. Without this a restart
                # re-measures every arm it already rejected -- measured as a 3-arm cycle repeating
                # forever once the promoted cell alone was excluded. Cleared on a re-mine, because a
                # different incumbent is a different question about the same cell.
                rejected_here.add(_cell_of_arm(arm))
                if best is None or _objective_key(obj) > _objective_key(best):
                    best = obj
                if improves(obj):
                    att.state, att.promoted, att.objective = IMPROVED, arm, obj
                    out.candidates.extend(built)
                    out.attempts.append(att)
                    out.certificates.extend(att.certificates)
                    # Records the cell, promotes, re-mines, and resets the sweep. Shared with the
                    # frontier and expansion sweeps so a promotion means the same thing everywhere.
                    if _promoted(arm, obj) is None:
                        return _finish(out)
                    # RESUME THE SWEEP against the re-mined residual. No separate latch is needed:
                    # `i` is reset above, and on re-entry the `not fresh` guard sees that this cell --
                    # added to `decided` BEFORE `remine` ran -- leaves the boundary with nothing to
                    # measure, so it advances the boundary rather than falling through to expansion.
                    # Verified by removing an earlier second latch here: the toy host produces the
                    # identical 4 evaluations over 3 cells and 3 DISTINCT Attempt objects either way.
                    #
                    # `out.state` is deliberately not used to carry this. Overloading it WAS the
                    # original defect: the carry guard then fired on every later iteration and
                    # continued WITHOUT decrementing `i`, sweeping one boundary forever, and
                    # `max_restarts` could not bound that because `restarts` advances only on a
                    # promotion -- and the re-measured arm scores net 0 against the moved incumbent,
                    # so no promotion ever happens again.
                    break
            else:
                att.objective = best
                att.state = NO_BENEFIT if best is not None else REALIZABLE_UNMEASURED
            if promoted_this_pass:
                # A promotion, whether or not a restart followed it. This attempt is already recorded,
                # and everything below is Phi expansion, whose trigger is a MEASURED insufficiency of
                # the representation at this boundary. A promotion is the opposite of that, so falling
                # through would both re-append the same Attempt object -- double-counting the round's
                # own work -- and expand Phi off a verdict nobody produced.
                continue
        elif realizable:
            # ------------------------------------------------------------------------------------
            # NO EVALUATOR MEANS NO VERDICT -- SO THE SCHEDULE STOPS HERE.
            #
            # This used to fall through to the expansion step below, which reads a state of
            # NO_CANDIDATE/NO_BENEFIT as "this representation is insufficient". With no evaluator
            # that state is manufactured, not measured: a stub returning a constant makes every
            # candidate score the same, `improves` is never true, and the round declares the seeded
            # Phi a failure WITHOUT RUNNING ANYTHING. Expansion then fires and re-enumerates the
            # grammar, which is how a search seeded with 4 grounded observables emitted 117 arms.
            #
            # Expanding the representation is a decision with a trigger, and the trigger is
            # MEASURED insufficiency. Backward movement has the same requirement: a boundary is
            # exhausted when its candidates were measured and none helped, not when nobody looked.
            #
            # So: keep the boundary, keep the seeded Phi, keep the candidate set, and report that
            # they are built and unmeasured. The caller evaluates them and comes back.
            att.state = REALIZABLE_UNMEASURED
            att.detail = ("controller built and grounded; NO evaluator supplied, so this "
                          "representation has not been shown insufficient -- neither Phi expansion "
                          "nor backward movement is warranted")
            out.candidates.extend(built)
            out.attempts.append(att)
            out.certificates.extend(att.certificates)
            out.state = REALIZABLE_UNMEASURED

            # VISIT THE OTHER BOUNDARIES THE SEEDED Phi IS OBSERVABLE AT, BEFORE STOPPING.
            #
            # `return` here stopped the whole search at the LATEST boundary. That conflated two very
            # different things:
            #
            #   expanding Phi, or declaring a boundary exhausted, on a verdict nobody measured
            #       -- forbidden, and still forbidden below
            #   collecting the candidates at the OTHER boundaries the proposer already localized
            #       -- not a verdict at all; it is completing the localization it was handed
            #
            # Measured on a real round: the proposer localized three signals, one of them
            # `no_tool_call_at_all` at an EARLIER boundary, and the search reported
            # `boundaries=[earlier, later]` with `visited=[later]` only. It stopped at the later
            # boundary and never built a single arm for the mechanism behind 63 failing queries.
            #
            # Scope is deliberately narrow: only boundaries where a SEEDED signal is observable, only
            # while no evaluator exists, and Phi is never widened. Nothing here scores or ranks -- every
            # boundary's arms are emitted and measurement decides.
            remaining = [k for k in range(i - 1, -1, -1)
                         if any(_observable_at(sig, _boundary_arg(ordered[k], runtime),
                                               runtime=runtime) for sig in base_phi)]
            if not remaining:
                return _finish(out)
            for k in remaining:
                b2 = ordered[k]
                arg2 = _boundary_arg(b2, runtime)
                if repairable_at is not None and not repairable_at(current, b2):
                    out.attempts.append(Attempt(boundary=b2.key, state=BOUNDARY_NOT_REPAIRABLE,
                                                label=b2.label,
                                                detail="the residual does not manifest at this decision"))
                    continue
                phi2 = [sig for sig in base_phi
                        if _observable_at(sig, arg2, runtime=runtime)]
                att2 = Attempt(boundary=b2.key, state=NO_CANDIDATE, label=b2.label)
                built2, realizable2 = _build_and_ground(arg2, phi2, att2)
                if att2.state == UNFIREABLE_HERE:
                    out.attempts.append(att2)
                    out.certificates.extend(att2.certificates)
                    continue
                if realizable2:
                    att2.state = REALIZABLE_UNMEASURED
                    att2.detail = ("controller built and grounded at a boundary the seeded Phi is "
                                   "also observable at; NO evaluator supplied")
                    out.candidates.extend(built2)
                out.attempts.append(att2)
                out.certificates.extend(att2.certificates)
                out.localization_sweep = out.localization_sweep + (b2.key,)
            return _finish(out)

        out.candidates.extend(built)

        # ------------------------------------------------------------------------------------------
        # EXHAUST THE GROUNDED FRONTIER BEFORE SYNTHESIZING A NEW REPRESENTATION.
        #
        # The ordering rule, stated generally: for one residual family, search every feasible
        # WHAT x HOW that its ATTRIBUTED Phi already supports, across all consequential boundaries,
        # before widening Phi through the grammar. A signal attribution grounded is evidence about
        # this failure; a synthesized predicate is a guess that happens to discriminate. Spending the
        # speculative move first is backwards.
        #
        # Measured on a real round: the seeded suppress arm at the latest boundary was measured at
        # net -3, and the schedule immediately expanded Phi into 40 synthesized atoms AT THE SAME
        # boundary -- never reaching four already-grounded arms at an earlier one, reachable from the
        # same seeded Phi with no synthesis at all. HOW was exhausted there; the representation was
        # not the thing that had run out.
        #
        # This is a WHERE move under a FIXED Phi, which is different from expansion in kind: no new
        # signal is installed, nothing is invented, and the arms it finds were admissible all along.
        # `frontier_boundaries` records which boundaries were swept this way, so the provenance of
        # "why this candidate" stays readable.
        if (att.state in (NO_CANDIDATE, NO_BENEFIT) or not realizable) and not frontier_swept:
            remaining = [k for k in range(i - 1, -1, -1)]
            if remaining:
                out.attempts.append(att)
                out.certificates.extend(att.certificates)
                frontier_swept = True
                swept_any = False
                restart_from_frontier = False
                for k in remaining:
                    b2 = ordered[k]
                    barg2 = _boundary_arg(b2, runtime)
                    phi2 = [sig for sig in base_phi
                            if _observable_at(sig, barg2, runtime=runtime)]
                    if not phi2:
                        continue
                    att2 = Attempt(boundary=b2.key, state=NO_CANDIDATE, label=b2.label,
                                   detail="grounded frontier: the SEEDED Phi, at an earlier boundary")
                    built2, realizable2 = _build_and_ground(barg2, phi2, att2)
                    out.frontier_boundaries = tuple(out.frontier_boundaries) + (b2.key,)
                    if not realizable2:
                        att2.state = NO_CANDIDATE
                        att2.detail = ("grounded frontier: no action here can be grounded for the "
                                       "seeded Phi")
                        out.attempts.append(att2)
                        out.certificates.extend(att2.certificates)
                        continue
                    swept_any = True
                    out.candidates.extend(built2)
                    best2 = None
                    for arm in built2:
                        obj = _eval(arm)
                        att2.candidates_evaluated += 1
                        if obj is None:
                            continue
                        if best2 is None or _objective_key(obj) > _objective_key(best2):
                            best2 = obj
                        if improves(obj):
                            att2.state, att2.promoted, att2.objective = IMPROVED, arm, obj
                            out.attempts.append(att2)
                            out.certificates.extend(att2.certificates)
                            # A frontier promotion is a promotion: re-mine and resume, rather than
                            # returning on a residual this arm has just made stale.
                            if _promoted(arm, obj) is None:
                                return _finish(out)
                            restart_from_frontier = True
                            break
                    if restart_from_frontier:
                        # `att2` is already recorded and `_promoted` has reset the sweep to the latest
                        # boundary. Falling through would append it a SECOND time and bank a
                        # NO_BENEFIT verdict over a boundary that had just promoted -- the same
                        # double-count the primary sweep's `promoted_this_pass` latch prevents.
                        break
                    # MISSING MEASUREMENT IS NOT A NULL. An arm nobody scored leaves the frontier
                    # UNMEASURED, which is what the caller must go and measure -- reporting NO_BENEFIT
                    # here would bank a verdict on arms that never ran.
                    att2.objective = best2
                    att2.state = NO_BENEFIT if best2 is not None else REALIZABLE_UNMEASURED
                    out.attempts.append(att2)
                    out.certificates.extend(att2.certificates)
                if restart_from_frontier:
                    # The frontier promoted and `_promoted` re-mined and reset `i`. Resume the outer
                    # sweep against the NEW residual. Neither the unmeasured-frontier report nor Phi
                    # expansion applies: both describe a representation that ran out, and a promotion
                    # is the opposite of that.
                    continue
                if swept_any and any(a.state == REALIZABLE_UNMEASURED for a in out.attempts):
                    # The frontier holds grounded arms nobody has measured. That is the deliverable:
                    # expansion stays unreached, because the representation has not been shown
                    # insufficient -- only one controller in it has.
                    out.state = REALIZABLE_UNMEASURED
                    return _finish(out)
                # Everything on the frontier was measured and none helped. NOW the representation is
                # the thing that ran out, and expansion is warranted. Fall through.

        # HOW is exhausted at this boundary under the CURRENT Phi. Per the ordering, widen Phi here
        # before moving earlier -- both NO_CANDIDATE and NO_BENEFIT route to a further attempt at
        # this same boundary, never straight to WHERE.
        if att.state in (NO_CANDIDATE, NO_BENEFIT) or not realizable:
            if att.state == NO_CANDIDATE and not phi_here:
                att.state, att.detail = SIGNAL_BLOCKED, "no declared signal is observable here"
            out.attempts.append(att)
            out.certificates.extend(att.certificates)

            # ---- STEP 3: expand WHAT at THIS boundary -------------------------------------------
            names, preds = _expand_phi_at(boundary_arg, runtime=runtime, states=states,
                                          existing=list(installed_preds))
            if not names:
                out.attempts.append(Attempt(boundary=b.key, state=SIGNAL_EXPANSION_EXHAUSTED,
                                            label=b.label,
                                            detail="no predicate over this boundary's fields both "
                                                   "discriminates and adds information"))
                out.state = SIGNAL_EXPANSION_EXHAUSTED
                i -= 1
                continue
            installed_preds.extend(preds)
            out.signals_installed.extend(names)
            exp = Attempt(boundary=b.key, state=SIGNAL_EXPANDED, label=b.label,
                          signals_expanded=tuple(names))
            restart_after_expansion = False

            # ---- STEP 4: retry HOW at the SAME boundary with the widened Phi --------------------
            built2, realizable2 = _build_and_ground(boundary_arg, names, exp)
            out.candidates.extend(built2)
            if realizable2:
                exp.state = REALIZABLE_UNMEASURED if evaluate is None else exp.state
                if evaluate is not None:
                    best = None
                    for arm in built2:
                        obj = _eval(arm)
                        exp.candidates_evaluated += 1
                        if obj is None:
                            continue
                        if best is None or _objective_key(obj) > _objective_key(best):
                            best = obj
                        if improves(obj):
                            exp.state, exp.promoted, exp.objective = IMPROVED, arm, obj
                            # An expansion promotion is a promotion. This is the path that installs a
                            # controller over a signal the core SYNTHESIZED, so it is the last place
                            # that should be unable to hand the caller a fresh residual.
                            restart_after_expansion = _promoted(arm, obj) is not None
                            break
                    else:
                        exp.objective = best
                        exp.state = NO_BENEFIT if best is not None else REALIZABLE_UNMEASURED
                out.attempts.append(exp)
                out.certificates.extend(exp.certificates)
                if out.state != IMPROVED:
                    out.state = exp.state
                # TERMINATION IS DECIDED BY THE STATE MACHINE, not by having reached this line.
                # IMPROVED leaves to promote; REALIZABLE_UNMEASURED is a structural stop (there is no
                # evaluator, so nothing here can be shown wrong). NO_BENEFIT is neither: the widened
                # Phi was MEASURED and did not help, which exhausts this boundary -- so the schedule
                # must move earlier instead of reporting the residual as finished. Returning here
                # unconditionally made every no-benefit boundary look terminal and `moves_earlier`
                # structurally unreachable on measured runs.
                if restart_after_expansion:
                    # Promoted here AND `remine` handed back a fresh residual, so the schedule
                    # continues against it. `_promoted` has already reset `i` to the latest boundary
                    # and cleared the frontier latch; returning instead would end the run on a
                    # residual this very promotion made stale, which is what made a synthesized-signal
                    # discovery terminal no matter how much residual was left.
                    continue
                if next_coordinate(out.state) in (DONE, PROMOTE):
                    return _finish(out)
                i -= 1
                continue
            exp.state = _BS_EXHAUSTED
            exp.detail = "Phi widened, but no action here can be grounded for the new signals"
            out.attempts.append(exp)
            out.certificates.extend(exp.certificates)
            out.state = _BS_EXHAUSTED
            i -= 1
            continue

        out.attempts.append(att)
        out.certificates.extend(att.certificates)
        out.state = att.state
        return _finish(out)

    if not out.state:
        out.state = _BS_EXHAUSTED
    return _finish(out)


def _fires_on_any(signal: str, boundary: Any, *, runtime, states: Sequence[Mapping[str, Any]],
                  params: Mapping[str, Any] | None = None) -> bool:
    """Does `signal` fire on at least one of `boundary`'s OWN projected states?

    Structural, not outcome-based: it asks whether a paired contrast can exist at all, which is a
    property of the predicate and the states. Nothing here scores or ranks -- a predicate firing on
    one state and one firing on ninety are equally measurable, and measurement decides between them.

    THE STATES MUST BE THE BOUNDARY'S OWN. A pre-dispatch predicate cannot fire on a state carrying no
    proposed call, so testing it against per-step states rejects it for the wrong reason -- the same
    defect that once rejected all 37 commitment-gate candidates. So project first, and fail OPEN on
    any projection or evaluation error: an infrastructure failure must not read as an inert arm.
    """
    project = getattr(runtime, "states_at", None)
    here = states
    if callable(project):
        try:
            here = project(getattr(boundary, "value", boundary), states) or ()
        except Exception:
            return True
    if not here:
        return True
    ev = getattr(runtime, "evaluate_signal", None)
    if not callable(ev):
        return True
    probe = dict(params or {})
    if not probe:
        try:
            probe = dict(getattr(runtime, "probe_params", lambda _s: {})(signal) or {})
        except Exception:
            probe = {}

    # OBSERVABILITY FIRST: "never fires" and "the states cannot express this signal" are DIFFERENT
    # facts, and only the first is inertness.
    #
    # A signal whose fields are absent from every projected state evaluates falsy everywhere -- but
    # that is a gap between the signal and the state RECORD, not evidence that the condition never
    # holds in the run. Calling it inert here would prune a legitimately seeded declared signal
    # because the adapter does not surface its field in this projection, which is the same class of
    # error as reading a raising predicate as all-False. So an UNOBSERVABLE signal is kept, and only a
    # signal this projection can actually express is ever judged inert.
    observed_fields: set[str] = set()
    for st in here:
        try:
            observed_fields |= set(st)
        except Exception:
            return True
    fields_fn = getattr(runtime, "signal_fields", None)
    if callable(fields_fn):
        try:
            need = set(fields_fn(signal) or ())
        except Exception:
            need = set()
        # EVERY field, not any. A signal is judgeable here only if this projection carries ALL the
        # evidence its predicate reads: a conjunctive predicate missing ONE field answers False for
        # that reason alone, which is a supply gap, not inertness. Requiring only a non-empty
        # intersection was measured wrong on a real round -- a capacity signal reading
        # (container_full, proposes_clear, error_kind) had the first two supplied and the third
        # missing, so it evaluated False on all 543 states and its boundary was pruned as inert while
        # the actual defect was the unsupplied field.
        if need - observed_fields:
            return True
    elif signal not in observed_fields:
        # No field declaration available. The conservative proxy is the signal's own name: a runtime
        # whose states never carry it may still evaluate it structurally, so keep the arm.
        return True

    fired = False
    for st in here:
        try:
            if ev(signal, st, probe):
                fired = True
                break
        except Exception:
            # A signal that cannot be evaluated on this state has not fired HERE, but the inability is
            # not evidence of inertness -- so a predicate that never evaluates anywhere is kept.
            return True
    return fired


def _observable_at(signal: str, boundary: Any, *, runtime) -> bool:
    """Is `signal` readable at `boundary`, per the runtime's own declaration?

    Tolerant: a runtime that cannot answer is taken to permit the signal, so the arm-building step
    (which asks the same question authoritatively and emits a certificate) makes the real decision.
    """
    fn = getattr(runtime, "signal_boundaries", None)
    if not callable(fn):
        return True
    try:
        at = fn(signal)
    except Exception:
        return True
    if not at:
        return False
    want = str(getattr(boundary, "value", boundary))
    return any(str(getattr(p, "value", p)) == want for p in at)
