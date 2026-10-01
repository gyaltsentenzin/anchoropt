"""LLM operators: narrow, controlled, validated. NOT autonomous agents.

The pattern every LLM use in AnchorOpt must follow, so that adding a second or third use
does not accumulate into an agent with discretion:

    1. GATED       an operator runs only when a cheaper method has provably failed to
                   produce an adequate result. It is a FALLBACK, never a first resort.
    2. NARROW      one question, fixed inputs, no access to the search state, the ranking,
                   the acceptance criteria, or prior outcomes.
    3. STRUCTURED  output is parsed into the pipeline's own representation. Prose is a
                   parse failure, not a result.
    4. VALIDATED   every proposal is checked mechanically before it is eligible for
                   evaluation. An unvalidated proposal is not a candidate.
    5. NON-DECIDING the operator never accepts, ranks or installs. Rollout and the frozen
                   acceptance criteria remain authoritative.
    6. ATTRIBUTED  output carries provenance (source, model, inputs shown) so a
                   history-only run and a history+LLM run stay distinguishable forever.
    7. OPTIONAL    off by default, configurable model, so `history-only` vs
                   `history+teacher` is a measurable comparison rather than a belief.

`LLMOperator` below is the base contract. Subclasses supply the gate, the prompt, and the
validator; they must not supply judgement about which candidate is better.

Deliberately NOT assuming a frontier model. `model` is a plain string resolved by the
existing teacher_client, so cheaper/smaller models can be benchmarked on the axes that
matter -- candidate validity, useful-candidate recall, downstream gain, cost and latency --
rather than assumed inadequate.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OperatorResult:
    """What an operator returns. `candidates` are already validated and tagged."""
    invoked: bool
    reason: str                      # why it ran, or why it was skipped
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    model: Optional[str] = None
    latency_s: Optional[float] = None
    raw: Optional[str] = None        # kept for audit; never parsed downstream

    def as_dict(self):
        return {"invoked": self.invoked, "reason": self.reason, "model": self.model,
                "latency_s": self.latency_s, "n_candidates": len(self.candidates),
                "candidates": self.candidates}


class LLMOperator:
    """Base contract. Subclasses override `gate`, `build_prompt`, `parse`, `validate`."""

    name = "llm_operator"

    # Default operator model. Haiku, not a frontier model: measured at parity with Opus-5
    # on candidate validity (2 valid / 0 rejected, both) at ~3x lower latency (2.9s vs
    # 8.5s) -- see §49.3. Candidate PROPOSAL for a bounded mapping is a simple task, so the
    # default should be cheap and the expensive option opt-in.
    DEFAULT_MODEL = "anthropic/aws/claude-haiku-4-5"

    def __init__(self, model: Optional[str] = None, enabled: bool = False,
                 endpoint: Optional[str] = None):
        """model    : any id the resolver understands (hosted gateway or local endpoint)
        endpoint : OpenAI-compatible base URL; when set, the model is served LOCALLY and
                   `model` is the served-model-name. This is the swap seam -- changing
                   which LLM runs an operator is CONFIGURATION, never a code edit.
        """
        # OFF by default: enabling an operator must be an explicit act, not a default a
        # future reader has to remember to disable.
        self.model = model or self.DEFAULT_MODEL
        self.endpoint = endpoint
        self.enabled = bool(enabled)

    # ── the four things a subclass supplies ─────────────────────────────────

    def gate(self, ctx) -> Optional[str]:
        """Return None to SKIP, or a reason string to invoke.

        This is the load-bearing method: an operator that always returns a reason is a
        first resort, not a fallback, and violates the pattern.
        """
        raise NotImplementedError

    def build_prompt(self, ctx):
        raise NotImplementedError                # -> (system, user)

    def parse(self, raw: str) -> List[Any]:
        """Structured extraction. Prose -> []."""
        m = re.search(r"\[.*\]", str(raw or ""), re.S)
        if not m:
            return []
        try:
            return list(json.loads(m.group(0)))
        except Exception:
            return []

    def validate(self, item, ctx) -> Dict[str, Any]:
        """-> a candidate record with a `status` field. Never raises."""
        raise NotImplementedError

    # ── the fixed driver; subclasses do not override this ───────────────────

    def run(self, ctx) -> OperatorResult:
        if not self.enabled:
            return OperatorResult(False, "operator disabled (default)", model=self.model)
        why = self.gate(ctx)
        if why is None:
            return OperatorResult(False, "gate closed: a cheaper method sufficed",
                                  model=self.model)
        try:
            system, user = self.build_prompt(ctx)
            client = self._client()
            if client is None:
                return OperatorResult(False, f"no client for model {self.model!r}",
                                      model=self.model)
            t0 = time.time()
            raw = client.complete(system, user)
            dt = round(time.time() - t0, 2)
        except Exception as e:
            return OperatorResult(False, f"call failed: {type(e).__name__}: {e}",
                                  model=self.model)

        out = []
        for item in self.parse(raw):
            try:
                rec = self.validate(item, ctx)
            except Exception as e:
                rec = {"status": "rejected",
                       "reason": f"validation error: {type(e).__name__}: {e}",
                       "raw_item": str(item)[:200]}
            rec.setdefault("provenance", {})
            rec["provenance"].update({"source": "llm_operator", "operator": self.name,
                                      "model": self.model,
                                      "endpoint": self.endpoint or "gateway"})
            out.append(rec)
        return OperatorResult(True, why, candidates=out, model=self.model,
                              latency_s=dt, raw=str(raw)[:2000] if raw else None)

    def _client(self):
        """Resolve a client. Two paths, both already supported by teacher_client:

          endpoint set -> VLLMTeacher against an OpenAI-compatible local server
          otherwise    -> LiteLLMTeacher against the configured gateway

        Kept behind one method so an operator never knows which backend it is talking to,
        and a future backend (another gateway, a different local server) is one branch
        here rather than a change in every operator.
        """
        try:
            from anchoropt.teacher_client import (LiteLLMTeacher, VLLMTeacher,
                                                  make_teacher)
        except ImportError:
            try:
                from teacher_client import (LiteLLMTeacher, VLLMTeacher, make_teacher)
            except ImportError:
                return None
        if self.endpoint:
            return VLLMTeacher(url=self.endpoint, model=self.model)
        return make_teacher(self.model)
