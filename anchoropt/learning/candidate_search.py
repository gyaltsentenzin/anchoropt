"""Structured candidate search: ResidualDiagnosis -> ranked (l, phi, mu, theta).

This is the piece that makes the loop AUTONOMOUS. Everything it needs already existed --
`feasible_actions` (structural), `HostProfile.executable_actions` (operational), the runtime's
declared signal vocabulary -- but nothing joined them into "given this failure, what controllers
could address it, and which is best?"

WHAT THIS IS NOT
----------------
Not `policy_tree.py`. That module is the mature BFCL search: ~1300 lines whose nodes consume
corpus-mined records (`c["signal"]`, `c["support"]`, attested recovery alternatives) and whose
data loading reads BFCL trajectory dirs and memory-container error strings. Its DECISION RULE is
general; its INPUTS are not. Porting it to TB2 would mean porting its corpus assumptions, so v0.1
uses this smaller enumerator over the same typed space and leaves `policy_tree` untouched as the
BFCL path.

The shared discipline is what matters and is preserved here:

  * the space is CLOSED -- no candidate can name a boundary, signal or action outside the
    declared vocabulary, so an LLM cannot smuggle in an arbitrary patch;
  * attribution PRUNES, measurement CHOOSES -- ranking narrows the set, it does not decide;
  * every rejection carries a named reason, because a silently pruned candidate is
    indistinguishable from one nobody thought of.

THE CENTRAL DECISION: a causal step is NOT a locus
--------------------------------------------------
A diagnosis localizes a REGION -- "the failure is around step k". Three boundaries can address a
step-k failure, and they differ in what is still changeable:

    PRE_GENERATION(k)            shape the decision that produces step k
    POST_GENERATION_PRE_EXEC(k)  block or rewrite the action step k proposed
    POST_EXECUTION(k-1)          reshape what step k is about to observe

Choosing among them is AnchorOpt's job. The diagnoser must never do it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from anchoropt.anchor import Action, IncisionPoint, feasible_actions
from anchoropt.runtime import HostProfile, ResidualDiagnosis

#: Pruned because an EARLIER ROUND already decided this controller identity -- installed it, or
#: measured it and rejected it. Not a defect in the candidate: a statement that the budget for it
#: was already spent. Recorded rather than dropped so the round can say what it declined to re-run.
ALREADY_DECIDED_IN_AN_EARLIER_ROUND = "already_decided_in_an_earlier_round"

SIGNAL_BLOCKED = "SIGNAL_BLOCKED"

# Ordering over actions by how much control they assert. Used ONLY to rank a surviving set, never
# to decide -- the vocabulary's control strength is a fixed property, as in the frozen A2 ranking.
_CONTROL_STRENGTH = {Action.SUPPRESS: 3, Action.REROUTE: 3, Action.REPROMPT: 2, Action.NOOP: 0}


@dataclass(frozen=True)
class ScoredCandidate:
    """One point in the space, with why it survived or was pruned."""

    boundary: IncisionPoint
    signal: str
    action: Action
    theta: Mapping[str, Any] = field(default_factory=dict)
    signal_params: Mapping[str, Any] = field(default_factory=dict)
    score: float = 0.0
    evidence: int = 0
    rationale: str = ""
    pruned_reason: str = ""

    @property
    def survived(self) -> bool:
        return not self.pruned_reason


def candidate_loci(step_hint: int | None = None) -> tuple[IncisionPoint, ...]:
    """The boundaries that could address a failure localized at a step.

    All three, always: the diagnosis says WHERE the failure showed, not where it is fixable, and
    narrowing here on intuition is exactly the conflation this function exists to prevent.
    `step_hint` is accepted and deliberately unused -- it documents that the caller may know the
    step without that changing which boundaries are candidates.
    """
    return tuple(IncisionPoint)


def expressible_under(diagnosis: ResidualDiagnosis, *, runtime) -> tuple[bool, str]:
    """Can ANY declared signal observe the condition this diagnosis names?

    v0.1 is deliberately shallow: it asks whether the runtime declares at least one signal whose
    name shares a meaningful token with the diagnosis text. That is a keyword heuristic, not
    semantics, and it is the honest v0.1 answer -- a real expressibility test needs a mapping from
    `consequential_decision` to an observable, which is exactly what Phi-expansion will build.

    Returning `(False, reason)` marks the case SIGNAL_BLOCKED and the round moves on.
    """
    text = " ".join((diagnosis.failure_mechanism or "",
                     diagnosis.consequential_decision or "",
                     diagnosis.evidence or "")).lower()
    if not text.strip():
        return False, f"{SIGNAL_BLOCKED}: diagnosis carries no mechanism or decision text"

    declared = tuple(runtime.declared_signals())
    hits = [sig for sig in declared if _signal_matches(sig, text, runtime)]
    if not hits:
        return False, (f"{SIGNAL_BLOCKED}: no declared signal in {list(declared)} observes "
                       f"the condition this diagnosis names")

    # PARTIAL COVERAGE IS NOT COVERAGE. The question is not "does SOME signal observe SOMETHING this
    # diagnosis mentions" but "does some signal observe EVERYTHING it identifies as consequential".
    #
    # MEASURED: one residual's diagnoses state a conjunction -- the retrieval came back weak AND a
    # second store was never consulted. `retrieval_similarity_below_threshold` matches on
    # "retrieval"/"similarity" and the residual was scored expressible, silently discarding the second
    # clause. The search then proceeded on a signal covering HALF the condition -- and that half is the
    # one already measured net-negative in three separate rounds. So the defect does not merely skip
    # expansion; it routes the round into a known-bad arm.
    residual = _uncovered_clauses(text, hits, runtime)
    if residual:
        return False, (f"{SIGNAL_BLOCKED}: {hits} observe part of this condition but not all of it -- "
                       f"uncovered: {residual}. A signal covering only part of a conjunction would "
                       f"fire on a superset of the intended states.")
    return True, f"expressible via {hits}"


# Clause markers the declared vocabulary does NOT express, each paired with the evidence phrases that
# indicate the diagnosis is asserting it. Deliberately small and explicit: this is a keyword test like
# the rest of v0.1's expressibility, and pretending otherwise would be worse than saying so. The point
# is not to detect every unexpressed clause -- it is to stop scoring a KNOWN-unexpressed one as covered.
_UNCOVERED_CLAUSE_MARKERS: dict[str, tuple[str, ...]] = {
    "a store/surface was never consulted at all": (
        "never consult", "never touch", "never queried", "never searched", "does not query",
        "did not query", "no archival lookup", "performs no", "without consulting",
        "never looked", "never checked the other",
    ),
    # DELIBERATELY NOT "no follow-up / never re-queried". That was in the first version of this list
    # and it broke a CONFIRMED recovery: one anchor's own historical attribution says the model did not
    # retry after a failed lookup -- but retrying is that anchor's REMEDY, and the condition it triggers
    # on (the lookup failed) is fully expressed by a declared signal. Marking the remedy's absence as an
    # unexpressed CONDITION blocked a residual the vocabulary covers.
    #
    # The distinction that matters: "a surface was never consulted" is part of the TRIGGERING STATE and
    # nothing declared observes it. "no retry happened" describes the missing INTERVENTION, which is
    # what the action space is for, not the signal.
}


def _uncovered_clauses(text: str, hits: Sequence[str], runtime) -> list[str]:
    """Clauses the diagnosis asserts that no matched signal can observe.

    A clause counts as uncovered only when the diagnosis asserts it AND no matched signal's own name
    tokens relate to it -- so a signal that genuinely observes the clause suppresses the finding.
    """
    joined = " ".join(hits).lower()
    out = []
    for clause, markers in _UNCOVERED_CLAUSE_MARKERS.items():
        if not any(m in text for m in markers):
            continue
        # crude relatedness: does any matched signal name share a content word with the clause?
        clause_words = {w for w in clause.replace("/", " ").split() if len(w) > 4}
        if any(w in joined for w in clause_words):
            continue
        out.append(clause)
    return out


def _implicated_signals(diagnosis: ResidualDiagnosis, runtime) -> set[str]:
    """The declared signals whose name tokens appear in the diagnosis text.

    Same keyword test `expressible_under` uses, exposed so ranking can prefer a signal the
    EVIDENCE named over one that merely happens to carry a stronger action. Without this the
    ranker maximises control strength alone and picks, say, SUPPRESS on an unrelated signal --
    an action with no attributable link to the failure, which is the "engagement of zero"
    failure mode this project has already paid for.
    """
    text = " ".join((diagnosis.failure_mechanism or "",
                     diagnosis.consequential_decision or "",
                     diagnosis.evidence or "")).lower()
    return {sig for sig in runtime.declared_signals() if _signal_matches(sig, text, runtime)}


# Tokens too generic to carry evidence on their own. A single-token match on any of these is a
# coincidence, not expressibility: "proposed" appears in almost every diagnosis of a tool-using
# agent, so matching `proposed_tool_action` on it alone selected a SUPPRESS controller for a
# primer-design arithmetic failure -- an intervention with no causal link to the failure.
_WEAK_TOKENS = frozenset({
    "proposed", "action", "response", "tool", "call", "step", "state", "result", "output",
    "value", "error", "failed", "check", "count", "index", "size", "text", "name", "path",
    # ENGLISH FILLER that happened to be a signal-name token. Measured across five recovery replays:
    # `append_would_exceed_cap` was reported expressible for a retrieval residual on the strength of
    # the word "would", and `no_informative_result` on "informative" -- auxiliary verbs and generic
    # adjectives appear in almost any diagnosis prose, so a match on them is grammar, not evidence.
    # This is why nothing was ever SIGNAL_BLOCKED and the expansion branch never fired: the heuristic
    # OVER-reports expressibility, so real gaps in Phi are hidden behind an accidental word match.
    "would", "could", "should", "informative", "identifier", "before", "after", "without",
    "another", "other", "first", "each", "same", "given", "actual", "content", "returned",
})


def _diagnosis_text(diagnosis: ResidualDiagnosis) -> str:
    """The prose a diagnosis offers as evidence. One definition, used by every matcher here."""
    return " ".join((diagnosis.failure_mechanism or "",
                     diagnosis.consequential_decision or "",
                     diagnosis.evidence or "")).lower()


def evidence_strength(signal: str, diagnosis: ResidualDiagnosis, runtime) -> int:
    """HOW STRONGLY this diagnosis names `signal` -- the count of distinct matching phrases.

    Attribution is already a GATE (`signal not in implicated` prunes). This is the same evidence,
    used for the one thing the gate cannot do: choose among candidates that all pass it. Several
    declared signals routinely observe overlapping conditions -- a capacity refusal and a
    slot-cap refusal share most of their vocabulary -- so a diagnosis frequently implicates two or
    three, and the gate admits them all.

    Without this the survivors were ordered by `_CONTROL_STRENGTH` alone and then ALPHABETICALLY,
    which is not a decision rule: the alphabetically-first boundary and signal won. Measured on the
    eight accepted BFCL anchors, that selected an intervention on `container_at_capacity` for a
    retrieval-similarity failure -- an unattributable pick of exactly the kind the gate exists to
    prevent, arrived at one step later.

    Counting DISTINCT declared aliases (plus distinctive name tokens) keeps the ranking a function
    of the evidence rather than of the vocabulary's spelling. It is still not semantics: it cannot
    tell a strongly-worded coincidence from a real match, which is what Phi-expansion addresses.
    """
    text = _diagnosis_text(diagnosis)
    hits = set()
    if runtime is not None and hasattr(runtime, "signal_aliases"):
        for alias in runtime.signal_aliases(signal):
            if alias in text:
                hits.add(alias)
    for token in signal.split("_"):
        if len(token) > 3 and token not in _WEAK_TOKENS and token in text:
            hits.add(token)
    return len(hits)


def _signal_matches(signal: str, text: str, runtime=None) -> bool:
    """Does `signal` plausibly observe what `text` describes?

    Requires either a match on a DISTINCTIVE token, or >=2 matching tokens. Both guard the same
    failure: one generic token in common is not evidence. This remains a keyword heuristic -- the
    real test needs a mapping from `consequential_decision` to an observable, which is what
    Phi-expansion will build. Being explicit about that is better than a heuristic that looks
    semantic and is not.
    """
    # A declared alias is a direct hit: the adapter asserted this phrase describes the signal.
    if runtime is not None and hasattr(runtime, "signal_aliases"):
        for alias in runtime.signal_aliases(signal):
            if alias in text:
                return True
    tokens = [t for t in signal.split("_") if len(t) > 3]
    matched = [t for t in tokens if t in text]
    if not matched:
        return False
    distinctive = [t for t in matched if t not in _WEAK_TOKENS]
    return bool(distinctive) or len(matched) >= 2


def enumerate_candidates(diagnosis: ResidualDiagnosis, *, runtime, host: HostProfile,
                         theta_for) -> tuple[ScoredCandidate, ...]:
    """Every (l, phi, mu) the diagnosis could use, each marked survived or pruned-with-reason.

    `theta_for(action, diagnosis)` supplies typed parameters; it returns None when the caller has
    no theta for that action, which prunes the cell rather than inventing parameters.
    """
    out: list[ScoredCandidate] = []
    declared = set(runtime.declared_signals())
    implicated = _implicated_signals(diagnosis, runtime)

    for boundary in candidate_loci():
        structural = feasible_actions(boundary)
        operational = host.executable_actions(boundary)
        for signal in sorted(declared):
            # phi must be observable AT this boundary -- ask the runtime, do not assume.
            #
            # The probe passes the signal's DECLARED probe parameters, not {}. A signal with a
            # required parameter (a measured threshold, say) raises ValueError on a bare probe, and
            # treating that as "not observable here" silently excluded every parameterized signal
            # from the search -- while reporting the misleading reason
            # `signal_not_observable_at_boundary`. The two failures are different facts and are now
            # kept apart: a signal the runtime cannot observe at l is pruned permanently, whereas
            # one that merely needs configuration is a gap the runtime can close by declaring a
            # probe default.
            probe = {}
            if hasattr(runtime, "probe_params"):
                probe = dict(runtime.probe_params(signal) or {})
            observable_here = True
            unconfigured = ""
            try:
                runtime.evaluate_signal(signal, {"boundary": boundary}, probe)
            except KeyError:
                observable_here = False
            except (ValueError, TypeError) as exc:
                observable_here = False
                unconfigured = str(exc)

            for action in sorted(structural, key=lambda a: a.value):
                if action is Action.NOOP:
                    continue                      # the control arm, never a proposal
                theta = theta_for(action, diagnosis)
                rationale = (f"{boundary.value} + {signal} + {action.value} for "
                             f"{diagnosis.case_id}")
                if action not in operational:
                    out.append(ScoredCandidate(
                        boundary=boundary, signal=signal, action=action, theta=theta or {},
                        rationale=rationale,
                        pruned_reason=f"unexecutable_in_host: {action.value} not in U_H({boundary.value})"))
                    continue
                if not observable_here:
                    reason = (f"signal_params_unconfigured: {signal} is observable at "
                              f"{boundary.value} but has no usable probe parameters -- {unconfigured}"
                              if unconfigured else
                              f"signal_not_observable_at_boundary: {signal} at {boundary.value}")
                    out.append(ScoredCandidate(
                        boundary=boundary, signal=signal, action=action, theta=theta or {},
                        rationale=rationale, pruned_reason=reason))
                    continue
                if theta is None:
                    out.append(ScoredCandidate(
                        boundary=boundary, signal=signal, action=action, theta={},
                        rationale=rationale,
                        pruned_reason=f"no_theta_available: nothing to parameterize {action.value}"))
                    continue
                # ATTRIBUTION IS A GATE, NOT A WEIGHT. An intervention on a signal the
                # diagnosis never implicated has no causal link to this failure, however
                # forceful the action -- so it is PRUNED, not merely ranked lower. Scoring it
                # instead let a 3.0 unimplicated candidate win a round in which 13.0
                # well-attributed candidates existed.
                if signal not in implicated:
                    out.append(ScoredCandidate(
                        boundary=boundary, signal=signal, action=action, theta=theta,
                        rationale=rationale,
                        pruned_reason=(f"signal_not_implicated: {signal} is not named by this "
                                       f"diagnosis, so an intervention on it is unattributable")))
                    continue
                strength = evidence_strength(signal, diagnosis, runtime)
                out.append(ScoredCandidate(
                    boundary=boundary, signal=signal, action=action, theta=theta,
                    signal_params=probe,
                    score=float(_CONTROL_STRENGTH.get(action, 0)), evidence=strength,
                    rationale=rationale + f" [named by the diagnosis, {strength} matching phrase(s)]"))
    return tuple(out)


def cell_of(candidate: ScoredCandidate) -> tuple[str, str, str]:
    """The CONTROLLER IDENTITY: (boundary, signal, action). Two candidates with the same cell are
    the same controller reached by different routes.

    Named here because three separate things depend on it and previously each spelled it inline:
    `select_top_k`'s within-round de-duplication, cross-round exclusion of controllers already
    settled or measured, and the composition checks in the golden registry. Theta is deliberately
    NOT part of the identity -- a different parameterization of the same cell is a revision of one
    controller, not a second one, and the registry's supersession rule is what governs it.
    """
    return (candidate.boundary.value, candidate.signal, candidate.action.value)


def select(candidates: Sequence[ScoredCandidate]) -> ScoredCandidate | None:
    """Pick one survivor. Ranking narrows; it does not claim the choice is optimal.

    Order:
      1. EVIDENCE strength  -- how strongly THIS diagnosis names the signal
      2. control strength   -- a fixed property of the vocabulary, as in the frozen A2 ranking
      3. lexicographic      -- a stable tiebreak so a rerun selects the same candidate

    WHY EVIDENCE OUTRANKS CONTROL STRENGTH. Both orderings were run against the eight accepted BFCL
    anchors (`scripts/anchor_recovery.py`). With control strength first, a candidate matching ONE
    incidental phrase outranked the correctly-attributed controller matching THREE, purely because
    its action asserts more control -- it cost A2 its recovery. Control strength describes the
    VOCABULARY; evidence describes THIS FAILURE. Letting a fixed weight override the actual evidence
    is the "forceful action on a barely-implicated signal" mode that the attribution gate exists to
    prevent, arriving one step later through the ranking instead.

    This does not make the ranking a decision procedure. Attribution still PRUNES and measurement
    still CHOOSES: everything here does is order the survivors handed to evaluation.

    Step 1 was absent entirely at first, and ties fell through to the ALPHABET -- so the
    alphabetically-first boundary and signal won regardless of evidence. Determinism still matters
    (step 3 keeps identical input selecting identically), but a deterministic rule may not be an
    arbitrary one.
    """
    survivors = [c for c in candidates if c.survived]
    if not survivors:
        return None
    return sorted(survivors,
                  key=lambda c: (-c.evidence, -c.score, c.boundary.value, c.signal,
                                 c.action.value))[0]


# The default screening budget. Small on purpose: each surviving candidate costs one paired
# evaluation, which is the expensive resource the whole search exists to spend carefully.
DEFAULT_TOP_K = 3


def select_top_k(candidates: Sequence[ScoredCandidate], k: int = DEFAULT_TOP_K
                 ) -> tuple[ScoredCandidate, ...]:
    """Screen to the k most plausible survivors. MEASUREMENT chooses among them; this does not.

    WHY THIS EXISTS AND `select` IS NO LONGER THE DECIDER. Ranking heuristics were doing a job they
    cannot do: when several genuinely plausible candidates survive attribution, returning ONE means
    a deterministic tie-break picked the controller before anything was measured. Making the
    tie-break smarter does not fix that -- it makes the heuristic a better optimizer, which is the
    opposite of "attribution prunes, measurement chooses" (docs/ACCEPTANCE_RULE.md).

    So the ranking's only job is COST CONTROL: narrow the field to a budget the evaluator can
    afford, paired-evaluate all of them, and let the acceptance rule decide. `k=1` reproduces the
    old behaviour and is the wrong default.

    Candidates are de-duplicated on (l, phi, mu) -- the same cell reached through two diagnoses is
    one controller, and evaluating it twice would spend the budget on a duplicate.
    """
    survivors = [c for c in candidates if c.survived]
    if not survivors:
        return ()
    ordered = sorted(survivors, key=lambda c: (-c.evidence, -c.score, c.boundary.value, c.signal,
                                               c.action.value))
    seen: set[tuple[str, str, str]] = set()
    out: list[ScoredCandidate] = []
    for c in ordered:
        cell = cell_of(c)
        if cell in seen:
            continue
        seen.add(cell)
        out.append(c)
        if len(out) >= max(1, int(k)):
            break
    return tuple(out)


def search(diagnoses: Sequence[ResidualDiagnosis], *, runtime, host: HostProfile,
           theta_for, exclude_cells: Iterable[tuple[str, str, str]] = ()
           ) -> tuple[ScoredCandidate | None, tuple[ScoredCandidate, ...],
                      tuple[tuple[ResidualDiagnosis, str], ...]]:
    """Full v0.1 search: (selected, all_considered, blocked).

    Diagnoses are taken in order; the first that is expressible AND yields a survivor decides the
    round. v0.1 installs one controller per round, so there is no reason to enumerate the rest --
    but every blocked diagnosis is still recorded, because that list IS the Phi-expansion backlog.

    `exclude_cells` removes controller identities (see `cell_of`) that a PREVIOUS round already
    settled or measured. Within one round `select_top_k` de-duplicates cells, and that was the only
    de-duplication there was -- so across rounds the search re-selected an identity it had already
    installed, and an unattended loop spent every round re-measuring its own incumbent. An excluded
    candidate is recorded in `considered` with a pruned_reason rather than dropped, because "already
    decided" is an auditable outcome and silence is not.
    """
    considered: list[ScoredCandidate] = []
    blocked: list[tuple[ResidualDiagnosis, str]] = []
    selected: ScoredCandidate | None = None
    excluded = {tuple(c) for c in exclude_cells}

    for d in diagnoses:
        ok, why = expressible_under(d, runtime=runtime)
        if not ok:
            blocked.append((d, why))
            continue
        for c in enumerate_candidates(d, runtime=runtime, host=host, theta_for=theta_for):
            if c.survived and cell_of(c) in excluded:
                c = replace(c, pruned_reason=ALREADY_DECIDED_IN_AN_EARLIER_ROUND)
            considered.append(c)
    # Select across EVERY diagnosis's candidates, not from whichever diagnosis happened to come
    # first: ordering in the residual is arbitrary, and letting it decide made the round's outcome
    # depend on brief order rather than on evidence.
    selected = select(considered)
    return selected, tuple(considered), tuple(blocked)


def uncovered_clause_of(diagnosis: ResidualDiagnosis, *, runtime) -> str | None:
    """The clause this diagnosis asserts that no declared signal observes, or None.

    Exposed because it is the right GROUPING KEY for a signal gap. Residual families are keyed on the
    attributor's prose, which collapses paraphrases but not genuinely different sentences -- so the six
    A9 diagnoses that all assert "a store was never consulted" scattered into families of support 1-2,
    and signal synthesis starved: a predicate cannot be shown discriminating on one observed state.
    Grouping by the clause pools exactly the diagnoses that share the same missing observable, which is
    what synthesis needs and what the residual key cannot express.

    No new analysis: this returns the value `expressible_under` already computes.
    """
    text = _diagnosis_text(diagnosis).lower()
    if not text.strip():
        return None
    try:
        declared = tuple(runtime.declared_signals())
    except Exception:
        return None
    hits = [sig for sig in declared if _signal_matches(sig, text, runtime)]
    if not hits:
        return None                       # nothing matched at all: a different kind of block
    clauses = _uncovered_clauses(text, hits, runtime)
    return clauses[0] if clauses else None
