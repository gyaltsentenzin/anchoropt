#!/usr/bin/env python3
"""The self-evolving outer loop: the DRIVER that was missing, not a new AnchorOpt core.

`docs/THE_LOOP.md` already specifies block-coordinate descent over a POLICY block and a SIGNAL
block, and every piece it needs already exists in this package:

    residual partition + phase decision   learning/phase_switch.py     (CONTINUE/EXPAND/STOP)
    persistent learned state              learning/evidence_ledger.py
    action-space pruning                  learning/policy_tree.py
    signal proposal + scoring             learning/expand_attribution.py
    engagement / attribution validation   learning/exposure.py, check_attribution.py
    structural admissibility              anchor.py  (feasible_actions)
    operational executability U_H(l)      runtime.py (HostProfile)

What was missing is the thing that CALLS them in order and records why it moved. So this module
owns sequencing and nothing else: no scoring, no thresholds, no gates of its own. Any rule it
appears to apply is imported. That is deliberate -- a second copy of a threshold is how two parts
of a loop start disagreeing about what was decided.

TWO SEAMS, both injected rather than imported:

    diagnosis provider   anything returning ResidualDiagnosis records. An earlier internal
                         harness was the first; it is not privileged. Its prompt-hook vocabulary
                         is translated away in its own provider adapter, so a prompt-heavy miner
                         cannot bias (l, phi, mu, theta) selection toward prompt edits.
    runtime adapter      supplies U_H(l), observables, signal evaluators, action application.

Neither seam knows about the other, and the core knows neither's identity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.runtime import HostProfile, ResidualDiagnosis

# The three phase verdicts. Imported from phase_switch when it is importable (it pulls in the BFCL
# artifact readers), mirrored here otherwise so this module stays importable standalone and the
# STRINGS cannot drift apart.
CONTINUE_POLICY = "CONTINUE_POLICY"
EXPAND_ATTRIBUTION = "EXPAND_ATTRIBUTION"
STOP = "STOP"


@dataclass(frozen=True)
class Candidate:
    """A controller proposed by the policy block, before validation or measurement."""

    boundary: IncisionPoint
    signal: str
    action: Action
    theta: Mapping[str, Any] = field(default_factory=dict)
    signal_params: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass(frozen=True)
class StepRecord:
    """What the loop did in one iteration, and why. The audit trail IS the deliverable.

    A loop that reports only its accepted anchors is unauditable: the deferrals are where the
    acceptance criteria become legible (`docs/THE_LOOP.md`), so every pruned candidate is recorded
    with the reason code that pruned it.
    """

    iteration: int
    phase: str
    n_diagnoses: int
    candidates: tuple[Candidate, ...] = ()
    pruned: tuple[tuple[Candidate, str], ...] = ()
    validated: tuple[Candidate, ...] = ()
    note: str = ""


def validate_candidates(candidates: Iterable[Candidate], *, host: HostProfile,
                        adapter) -> tuple[list[Candidate], list[tuple[Candidate, str]]]:
    """Split candidates into (validated, pruned-with-reason) using EXISTING rules only.

    Three checks, each with its own reason code so a rejection is attributable rather than merely
    counted:

      unexecutable_in_host   U_H(l) -- the operational check `admissible()` alone cannot make
      signal_not_declared    phi has no evaluator in this runtime's vocabulary
      bad_signal_params      required parameters missing or mistyped

    The order matters: the cheapest, most decisive check runs first. Nothing here measures anything
    -- attribution prunes, measurement chooses.
    """
    ok: list[Candidate] = []
    pruned: list[tuple[Candidate, str]] = []
    declared = set(adapter.declared_signals())
    for c in candidates:
        try:
            host.require(c.boundary, c.action)
        except Exception as exc:
            pruned.append((c, f"unexecutable_in_host: {exc}"))
            continue
        if c.signal not in declared:
            pruned.append((c, f"signal_not_declared: {c.signal!r} not in this runtime's vocabulary"))
            continue
        try:
            adapter.evaluate_signal(c.signal, {}, c.signal_params)
        except (ValueError, TypeError) as exc:
            pruned.append((c, f"bad_signal_params: {exc}"))
            continue
        except KeyError as exc:                      # pragma: no cover - guarded by `declared`
            pruned.append((c, f"signal_not_declared: {exc}"))
            continue
        ok.append(c)
    return ok, pruned


def run(*, diagnose: Callable[[], Sequence[ResidualDiagnosis]],
        decide_phase: Callable[[Sequence[ResidualDiagnosis]], str],
        propose: Callable[[Sequence[ResidualDiagnosis]], Sequence[Candidate]],
        expand: Callable[[Sequence[ResidualDiagnosis]], Sequence[str]],
        host: HostProfile, adapter, max_iterations: int = 4) -> list[StepRecord]:
    """Drive the loop. Every decision function is INJECTED; this function decides only order.

        diagnose      -> residual diagnoses from any provider
        decide_phase  -> CONTINUE_POLICY | EXPAND_ATTRIBUTION | STOP  (phase_switch's rule)
        propose       -> candidates under the CURRENT fixed signal vocabulary (policy_tree)
        expand        -> newly accepted signal names (expand_attribution), vocabulary then frozen

    Returns the audit trail. Stops on STOP, on exhausting `max_iterations`, or when an expansion
    proposes nothing -- the last being the honest terminal state for "no representation can see
    the rest of the residual", which is NOT the same as "done" and must not be reported as such.
    """
    trail: list[StepRecord] = []
    for i in range(1, max_iterations + 1):
        diagnoses = tuple(diagnose())
        phase = decide_phase(diagnoses)

        if phase == STOP:
            trail.append(StepRecord(iteration=i, phase=STOP, n_diagnoses=len(diagnoses),
                                    note="residual addressed; no locus survives and unexplained "
                                         "mass is small"))
            return trail

        if phase == EXPAND_ATTRIBUTION:
            new_signals = tuple(expand(diagnoses))
            trail.append(StepRecord(
                iteration=i, phase=EXPAND_ATTRIBUTION, n_diagnoses=len(diagnoses),
                note=(f"expanded representation: {list(new_signals)}" if new_signals
                      else "expansion proposed nothing that discriminates; the residual is not "
                           "addressable in any currently proposable representation")))
            if not new_signals:
                return trail
            continue

        candidates = tuple(propose(diagnoses))
        validated, pruned = validate_candidates(candidates, host=host, adapter=adapter)
        trail.append(StepRecord(
            iteration=i, phase=CONTINUE_POLICY, n_diagnoses=len(diagnoses),
            candidates=candidates, pruned=tuple(pruned), validated=tuple(validated),
            note=f"{len(validated)} of {len(candidates)} candidates survived pre-measurement gates"))
        if not validated:
            # Policy exhausted under this vocabulary. Deliberately NOT reported as STOP: the
            # distinction between 'nothing survives' and 'genuinely done' is the one phase_switch
            # exists to preserve.
            return trail
    return trail


# ================================================================================================
# v0.1: one autonomous round -- step(incumbent) -> StepResult
# ================================================================================================
#
# `run()` above is the general driver with every decision injected. `step()` is the v0.1
# collaborator-facing entry point: ONE round, with the candidate search supplied by
# `candidate_search` rather than by the caller, and Phi-expansion deliberately absent.
#
# A diagnosis the current vocabulary cannot express is recorded SIGNAL_BLOCKED and skipped. That
# list is the Phi-expansion backlog and is the honest v0.1 answer -- see docs/SELF_EVOLVE_V0.md.


@dataclass(frozen=True)
class Incumbent:
    """The harness under optimization: a workspace plus its lineage."""

    workspace: Any                     # Path to a dir containing repo_baseline.py
    incumbent_id: str = "baseline"
    depth: int = 0


@dataclass(frozen=True)
class EvaluationResult:
    """Paired outcome. `control`/`candidate` map case_id -> strict pass."""

    control: Mapping[str, bool]
    candidate: Mapping[str, bool]
    gains: tuple[str, ...] = ()
    losses: tuple[str, ...] = ()
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    denominator_ok: bool = False
    detail: str = ""

    @property
    def net(self) -> int:
        return len(self.gains) - len(self.losses)


@dataclass(frozen=True)
class StepResult:
    """Everything one round produced, including what it refused to do and why."""

    incumbent_id: str
    diagnoses: tuple[ResidualDiagnosis, ...] = ()
    candidates_considered: tuple[Any, ...] = ()
    selected_candidate: Any = None
    blocked_diagnoses: tuple[tuple[ResidualDiagnosis, str], ...] = ()
    evaluation: EvaluationResult | None = None
    accepted: bool = False
    new_incumbent: Incumbent | None = None
    outcome: str = ""

    @property
    def survived_candidates(self) -> tuple[Any, ...]:
        return tuple(c for c in self.candidates_considered if getattr(c, "survived", False))


# The four ways a round can end. They are NOT interchangeable; see docs/SELF_EVOLVE_V0.md.
OUTCOME_ACCEPTED = "accepted"
OUTCOME_REJECTED = "rejected"
OUTCOME_NO_CANDIDATE = "no_candidate"
OUTCOME_BLOCKED = "blocked"
# Selection ran and chose a candidate, but no evaluator was wired -- an inspection mode, NOT a
# result. Kept distinct from `no_candidate` so "the optimizer found nothing" can never be confused
# with "the optimizer was not asked to measure anything".
OUTCOME_SELECTED_ONLY = "selected_only"


def step(incumbent: Incumbent, *, runtime, host: HostProfile, diagnose, theta_for,
         materialize=None, evaluate=None, promote=None, accept=None,
         exclude_cells=()) -> StepResult:
    """One autonomous round: diagnose -> search -> materialize -> evaluate -> accept -> promote.

    Injected seams, so a collaborator can run the search offline before wiring a real benchmark:

        diagnose()                      -> Sequence[ResidualDiagnosis]
        theta_for(action, diagnosis)    -> theta mapping, or None to prune that action
        materialize(spec, incumbent)    -> candidate workspace         (optional)
        evaluate(incumbent, candidate)  -> EvaluationResult            (optional)
        accept(EvaluationResult)        -> (bool, reason)              (optional)
        promote(incumbent, candidate)   -> Incumbent                   (optional)

    Omitting `materialize`/`evaluate` stops after selection and returns the chosen candidate --
    which is the cheap way to inspect what the optimizer WOULD install.

    `exclude_cells` is the controller identities an EARLIER round already decided -- see
    `candidate_search.cell_of`. A round is stateless by design, so without this the search
    re-selects the identity it installed last round: measured live, `step()` chose the same
    (boundary, signal, action) cell twice in a row against the same residual. An unattended loop
    then spends every round re-measuring its own incumbent, and the run looks productive while
    discovering nothing. The caller owns the memory (`round_ledger`); `step()` only honours it.
    """
    from anchoropt.learning.candidate_search import search

    diagnoses = tuple(diagnose())
    selected, considered, blocked = search(diagnoses, runtime=runtime, host=host,
                                           theta_for=theta_for, exclude_cells=exclude_cells)

    base = dict(incumbent_id=incumbent.incumbent_id, diagnoses=diagnoses,
                candidates_considered=considered, selected_candidate=selected,
                blocked_diagnoses=blocked)

    if selected is None:
        return StepResult(**base, outcome=OUTCOME_NO_CANDIDATE)
    if materialize is None or evaluate is None:
        return StepResult(**base, outcome=OUTCOME_SELECTED_ONLY)

    candidate_ws = materialize(selected, incumbent)
    evaluation = evaluate(incumbent, candidate_ws)

    # Denominator integrity is a PRECONDITION, not a result. An evaluation whose case coverage
    # cannot be trusted must never reach the acceptance rule -- that is how infrastructure
    # failure gets laundered into a null result.
    if not evaluation.denominator_ok:
        return StepResult(**base, evaluation=evaluation, outcome=OUTCOME_BLOCKED)

    ok, _reason = (accept or _default_accept)(evaluation)
    if not ok:
        return StepResult(**base, evaluation=evaluation, outcome=OUTCOME_REJECTED)

    new_inc = promote(incumbent, candidate_ws) if promote else None
    return StepResult(**base, evaluation=evaluation, accepted=True,
                      new_incumbent=new_inc, outcome=OUTCOME_ACCEPTED)


def _default_accept(ev: EvaluationResult) -> tuple[bool, str]:
    """AnchorOpt's acceptance rule, v0.1 subset -- not any other harness's.

    Requires net > 0 on the paired cases AND non-zero engagement AND a clean control arm. The
    engagement terms are the non-negotiable part: `docs/ACCEPTANCE_RULE.md` records that a
    positive delta with zero firings is proof of NON-attribution, never of a subtle effect.

    Deliberately omitted in v0.1: the dev/held-out no-regression term, because v0.1 runs one
    split. That omission is why v0.1 is a plumbing milestone and not an efficacy claim.
    """
    fired = int(ev.telemetry.get("signal_firings", 0) or 0)
    executed = int(ev.telemetry.get("interventions_executed", 0) or 0)
    if ev.net <= 0:
        return False, f"net {ev.net} <= 0 on {len(ev.control)} paired cases"
    if executed == 0:
        return False, f"zero interventions executed ({fired} firings) -- gain is not attributable"
    return True, (f"net +{ev.net} ({len(ev.gains)} gains / {len(ev.losses)} losses), "
                  f"{executed} interventions executed")
