"""Self-Evolve v0.2: persistent experiment memory, candidate dedup, and the primitive registry.

    ledger       append-only experiments.jsonl + lessons.jsonl
    fingerprint  canonical (residual_family, locus, signal, action_family, normalized_eta)
    dedup        evaluate / reuse / installed, plus residual downweighting after a failed attempt
    primitives   what this host can actually materialize, read FROM the runtime
    context      the compact RELEVANT PRIOR EXPERIENCE block for the semantic roles

ADDITIVE. Nothing here is imported by the v0.1 cycle driver or by anchoropt/learning; wiring is a
v0.2 step and `tests/test_evolve_memory.py` pins the separation.
"""

from anchoropt.evolve_memory.context import experience_block, relevant_experiments
from anchoropt.evolve_memory.dedup import (
    EVALUATE, INSTALLED, REUSE, Verdict, rank_key, reorder_residuals, residual_penalty,
    screen_all, screen_candidate,
)
from anchoropt.evolve_memory.fingerprint import (
    candidate_fingerprint, context_key, fingerprint_parts, normalize_eta, residual_family,
)
from anchoropt.evolve_memory.ledger import (
    DEFAULT_ROOT, STATUSES, Experiment, ExperimentLedger, Lesson,
)
from anchoropt.evolve_memory.primitives import Primitive, available, prompt_block, unavailable

__all__ = [
    "DEFAULT_ROOT", "EVALUATE", "INSTALLED", "Experiment", "ExperimentLedger", "Lesson",
    "Primitive", "REUSE", "STATUSES", "Verdict", "available", "candidate_fingerprint",
    "context_key", "experience_block", "fingerprint_parts", "normalize_eta", "prompt_block",
    "rank_key", "relevant_experiments", "reorder_residuals", "residual_family",
    "residual_penalty", "screen_all", "screen_candidate", "unavailable",
]
