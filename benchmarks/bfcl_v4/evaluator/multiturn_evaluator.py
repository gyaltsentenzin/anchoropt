"""
AnchorOptEvaluator — adapter exposing the legacy v3 evaluator interface on top of
the proven v4 memory evaluator (`MemoryAnchorOptEvaluator`).

Background: the original v3 optimizer stack was written against an evaluator class
named `AnchorOptEvaluator` exposing
`evaluate_stateful_episode(episode, policy, rollout_tag=...) -> (checker, trajectory)`.
That v3 class went missing. Rather than resurrect the v3 rollout engine, this adapter
presents that exact interface on top of `MemoryAnchorOptEvaluator.evaluate_episode`,
which is the fidelity-checked v4 path (official tool execution + `agentic_checker` +
snapshot store). Current consumer: `scripts/error_attribution_study.py`.

Handler + evaluator are built LAZILY (on first episode / store build) so the adapter
can be imported and constructed without a running vLLM server — only actually running
an episode requires the server.
"""

from typing import Dict, List, Optional, Tuple


# Defaults for the Granite-4.1-8B prompting handler used throughout this project.
_DEFAULT_HANDLER_MODULE = "bfcl_eval.model_handler.local_inference.granite_4"
_DEFAULT_HANDLER_CLASS = "Granite4PromptHandler"
_DEFAULT_REGISTRY_NAME = "ibm-granite/granite-4.1-8b-prompting"


class AnchorOptEvaluator:
    """v3-compatible evaluator facade over MemoryAnchorOptEvaluator (v4 memory)."""

    def __init__(
        self,
        granite_url: Optional[str] = None,
        model_url: Optional[str] = None,
        model_handler_path: Optional[str] = None,
        gate_enabled: bool = True,
        *,
        vllm_url: Optional[str] = None,
        served_model_name: Optional[str] = None,
        handler_module: str = _DEFAULT_HANDLER_MODULE,
        handler_class: str = _DEFAULT_HANDLER_CLASS,
        registry_name: str = _DEFAULT_REGISTRY_NAME,
        max_steps_per_turn: int = 20,
        seed: int = 42,
        snapshot_cache_dir=None,
        store_workers: int = 1,
        **_ignored,
    ):
        # Accept the several URL aliases the SkillOpt call sites use.
        self.vllm_url = (model_url or granite_url or vllm_url or "http://localhost:8080").rstrip("/")
        self.served_model_name = served_model_name
        # model_handler_path may be "module.Class", "module:Class", or just a module.
        if model_handler_path:
            if ":" in model_handler_path:
                handler_module, handler_class = model_handler_path.split(":", 1)
            elif "." in model_handler_path.rsplit(".", 1)[-1][:1] or model_handler_path[0].islower():
                # dotted path whose last segment is the class, e.g. a.b.Class
                parts = model_handler_path.rsplit(".", 1)
                if len(parts) == 2 and parts[1][:1].isupper():
                    handler_module, handler_class = parts
        self._handler_module = handler_module
        self._handler_class = handler_class
        self._registry_name = registry_name
        # gate_enabled (SkillOpt) → disable_gates (MemoryAnchorOptEvaluator)
        self._disable_gates = not gate_enabled
        self._max_steps_per_turn = max_steps_per_turn
        self._seed = seed
        self._snapshot_cache_dir = snapshot_cache_dir
        self._store_workers = store_workers
        self._mem = None  # built lazily

    # ── Lazy construction ────────────────────────────────────────────────────
    def _ensure(self):
        if self._mem is not None:
            return self._mem
        import requests
        from anchoropt.memory_evaluator import MemoryAnchorOptEvaluator

        served = self.served_model_name
        if not served:
            r = requests.get(f"{self.vllm_url}/v1/models", timeout=10)
            served = r.json()["data"][0]["id"]
            self.served_model_name = served

        mod = __import__(self._handler_module, fromlist=[self._handler_class])
        HandlerCls = getattr(mod, self._handler_class)
        try:
            handler = HandlerCls(
                model_name=served, temperature=0.001,
                registry_name=self._registry_name, is_fc_model=False,
            )
        except TypeError:
            handler = HandlerCls(model_name=served, temperature=0.001)

        self._mem = MemoryAnchorOptEvaluator(
            handler=handler,
            vllm_url=self.vllm_url,
            served_model_name=served,
            max_steps_per_turn=self._max_steps_per_turn,
            snapshot_cache_dir=self._snapshot_cache_dir,
            disable_gates=self._disable_gates,
            seed=self._seed,
            store_workers=self._store_workers,
        )
        return self._mem

    # ── v3-compatible interface used by the SkillOpt optimizers ────────────────
    def evaluate_stateful_episode(
        self, episode: Dict, policy: Dict, rollout_tag: str = "",
    ) -> Tuple[Dict, List]:
        """Delegate to the v4 memory evaluator. `policy` is the AnchorOpt templates dict."""
        return self._ensure().evaluate_episode(episode, policy, rollout_tag=rollout_tag)

    # Passthroughs so memory-specific driving code can use the adapter directly.
    def evaluate_episode(self, episode: Dict, templates: Dict, rollout_tag: str = "") -> Tuple[Dict, List]:
        return self._ensure().evaluate_episode(episode, templates, rollout_tag=rollout_tag)

    def build_snapshot_store(self, prereq_cases: List[Dict], templates: Optional[Dict] = None) -> None:
        return self._ensure().build_snapshot_store(prereq_cases, templates)

    @property
    def _store_diag(self):
        return self._ensure()._store_diag
