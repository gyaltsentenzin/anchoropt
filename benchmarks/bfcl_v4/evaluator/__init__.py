from .state_extractor import snapshot, delta
from .tool_ranker import rank_tools
from .injection_engine import get_injection, load_policy
from .memory_evaluator import MemoryAnchorOptEvaluator

# Signal/gate registry adapter — canonical data lives on the bfcl side
# (bfcl_eval.model_handler.memory_gates); this re-exports it. Degrades to inert
# stand-ins (see memory_gates.py) rather than raising if bfcl_eval isn't importable.
from .memory_gates import (
    GateSpec,
    MEMORY_GATE_REGISTRY,
    gate_fallback,
    gate_applies,
    gate_keys,
)

# multiturn adapter — may not be present in a v4-only installation
try:
    from .multiturn_evaluator import AnchorOptEvaluator
except ImportError:
    AnchorOptEvaluator = None  # type: ignore

__all__ = [
    "snapshot", "delta", "rank_tools", "get_injection", "load_policy",
    "MemoryAnchorOptEvaluator",
    "AnchorOptEvaluator",
    "GateSpec", "MEMORY_GATE_REGISTRY", "gate_fallback", "gate_applies", "gate_keys",
]
