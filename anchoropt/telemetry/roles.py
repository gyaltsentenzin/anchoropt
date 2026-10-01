"""Role interfaces for the eventual architecture. TYPES ONLY -- no behaviour, no wiring.

    Attributor        -> ResidualDiagnosis
    Localizer         -> (locus, signal)
    Runtime           -> feasible_actions(locus, signal)
    CandidateGenerator-> eta_mu candidates
    AnchorPolicyOpt   -> measured winner
    Executor          -> faithful realization

DELIBERATELY NOT DONE HERE, per the brief: `action_set` is NOT removed from the existing proposer,
candidate generation is unchanged, AnchorPolicyOpt is untouched, executors are untouched. These are
`Protocol`s -- structural types. Nothing in the live path implements them by name, nothing imports
them, and a class satisfies one by having the right shape, so adopting them later is a type-checking
change rather than a runtime one.

NOTE ON `Runtime.feasible_actions(locus, signal)`
------------------------------------------------
The eventual signature takes BOTH locus and signal. The current runtime's `feasible_actions(boundary)`
takes only the boundary, and the signal-dependent part lives in `executor_supports(...)`. That gap is
the actual finding from R3 -- admissibility is per-boundary, materializability is per (boundary,
action, signal, eta) -- so the protocol is written the way the architecture SHOULD be, and the
mismatch with today's runtime is intentional and recorded rather than papered over by defining the
protocol to match the current code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Attributor(Protocol):
    """Failure -> a structured diagnosis. May NOT name a locus, signal, action or parameter."""

    def attribute(self, evidence: Mapping[str, Any]) -> Any:
        """Returns a ResidualDiagnosis-shaped object (see anchoropt.runtime.ResidualDiagnosis)."""
        ...


@runtime_checkable
class Localizer(Protocol):
    """Diagnosis -> WHERE to intervene and on WHAT condition. Emits no action."""

    def localize(self, diagnoses: Sequence[Any]) -> tuple[Any, str]:
        """Returns (locus, signal)."""
        ...


@runtime_checkable
class RuntimeCapabilities(Protocol):
    """The host's own account of what it can do where.

    `feasible_actions` is ADMISSIBILITY. `executor_supports` is MATERIALIZABILITY, and the two are
    not the same question -- keeping them as separate members is the point.
    """

    def feasible_actions(self, locus: Any, signal: str | None = None) -> frozenset[Any]:
        ...

    def executor_supports(self, locus: Any, action: Any, signal: str,
                          eta: Mapping[str, Any]) -> tuple[bool, str]:
        ...


@runtime_checkable
class CandidateGenerator(Protocol):
    """(locus, signal, action) -> concrete eta_mu realizations, each independently evaluable."""

    def candidates(self, locus: Any, signal: str, action: Any) -> Sequence[Mapping[str, Any]]:
        ...


@runtime_checkable
class PolicyOptimizer(Protocol):
    """Candidates -> the MEASURED winner. Invents nothing; selects by downstream outcome."""

    def select(self, candidates: Sequence[Any], *, incumbent: Any,
               evaluate: Any) -> Any:
        ...


@runtime_checkable
class Executor(Protocol):
    """Faithful realization: runs eta UNCHANGED or declines.

    `supports` must reject a parameter it would have to coerce -- coercion runs different semantics
    than the ones proposed, under the proposal's name.
    """

    def supports(self, signal: str, eta: Mapping[str, Any]) -> tuple[bool, str]:
        ...

    def realize(self, signal: str, eta: Mapping[str, Any]) -> Any:
        ...


ROLE_PROTOCOLS = (
    Attributor, Localizer, RuntimeCapabilities, CandidateGenerator, PolicyOptimizer, Executor,
)
