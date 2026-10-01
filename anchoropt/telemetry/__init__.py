"""Observational telemetry. Nothing here is imported by a live decision path.

Every module in this package RECORDS what already happened; none of it decides anything, and none of
it is referenced from `anchoropt/learning/` or `benchmarks/`. That separation is enforced by
`tests/test_telemetry_is_observational.py`, which greps the live packages for imports of this one --
so a future change that wires telemetry into a decision fails a test rather than silently changing
an arm.

    search_funnel     proposed -> (l,phi) -> admissible -> executable -> evaluated -> accepted
    runtime_cost      per-episode / per-arm calls, tokens, steps, latency
    locality          how LOCAL a controller is: firing rate, target coverage, off-target changes
    round_record      the residual-round summary object (schema + logger)
    roles             role interfaces for the eventual refactor (typing only)
"""

from anchoropt.telemetry.locality import LocalityReport, locality_report
from anchoropt.telemetry.round_record import (
    CandidateEta, ResidualEntry, RoundLogger, RoundRecord, residual_shape,
)
from anchoropt.telemetry.roles import ROLE_PROTOCOLS
from anchoropt.telemetry.runtime_cost import ArmCost, EpisodeCost, arm_cost, cost_deltas
from anchoropt.telemetry.search_funnel import (
    FUNNEL_STAGES, REJECTION_REASONS, FunnelReport, SearchFunnel, classify_rejection,
)

__all__ = [
    "ArmCost", "CandidateEta", "EpisodeCost", "FUNNEL_STAGES", "FunnelReport", "LocalityReport",
    "REJECTION_REASONS", "ROLE_PROTOCOLS", "ResidualEntry", "RoundLogger", "RoundRecord",
    "SearchFunnel", "arm_cost", "classify_rejection", "cost_deltas", "locality_report",
    "residual_shape",
]
