"""
TeacherClient — the swap seam for the optimizer's "teacher" model.

The training loop (run_memory_train.py) asks a teacher LLM to propose one template
edit per batch. Previously that call was a hand-rolled ``litellm.completion(...)``
inlined in the loop, hardcoding IBM-Bedrock-specific request fields — so pointing
the teacher at a non-Bedrock model silently sent the wrong request shape.

This module isolates "how to call the teacher" behind one method::

    teacher = make_teacher(model_id)          # or None → local-vLLM teacher
    text = teacher.complete(system, user)     # -> Optional[str]

so a different teacher model can be swapped in without touching the training loop.
Each backend owns its own request quirks:

* ``LiteLLMTeacher`` — the default. Owns the Bedrock ``extra_body``
  (``{thinking: adaptive, output_config: {effort: high}}``) and ``max_tokens=8192``.
  Reproduces the previous inline request EXACTLY (byte-identical default run).
* ``VLLMTeacher`` — an OpenAI-compatible local endpoint (temperature 0.25,
  max_tokens 1500), mirroring the old skillopt ``LLMOptimizer._call_vllm`` path.

Kept ``bfcl_eval``-free; ``litellm`` / ``requests`` are imported lazily inside
``complete()`` so importing this module (e.g. for tests) is cheap and dep-free.
"""

from __future__ import annotations

import os
from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class TeacherClient(Protocol):
    """A teacher that turns a (system, user) prompt pair into text (or None on failure)."""

    def complete(self, system: str, user: str) -> Optional[str]:  # pragma: no cover - protocol
        ...


def _flatten_content(content) -> Optional[str]:
    """Collapse a LiteLLM response body to plain text.

    Thinking-enabled models (IBM Bedrock Claude) return a list of blocks; keep only
    the ``text`` blocks and drop thinking blocks. Returns None if nothing remains.
    """
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return " ".join(text_parts).strip() or None
    return content


class LiteLLMTeacher:
    """Default teacher: a LiteLLM-backed model (e.g. openai/aws/claude-opus-4-8).

    Owns the Bedrock-specific request fields so the training loop stays generic.
    The emitted request is byte-identical to the previous inlined ``_call_teacher``.
    """

    # Bedrock only supports adaptive thinking; max_tokens=8192 keeps text output
    # from being squeezed out by thinking blocks. These are impl details of THIS
    # backend, not the training loop's concern.
    MAX_TOKENS = 8192
    EXTRA_BODY = {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }

    # Both EXTRA_BODY fields are Bedrock/gateway-only: the DIRECT Anthropic API rejects
    # "adaptive" thinking (it wants {"type": "enabled", "budget_tokens": N}) and has no
    # output_config field at all, so sending them 400s the request.
    #
    # The discriminator is the "aws/" SEGMENT, not the leading provider. Every gateway route
    # we use carries it -- "openai/aws/...", "anthropic/aws/..." -- while a direct route does
    # not ("anthropic/claude-opus-4-8"). An earlier version of this keyed on the "anthropic/"
    # PREFIX, which is wrong in both directions: it stripped extra_body from the
    # "anthropic/aws/..." gateway routes -- including routes used by recorded runs, breaking
    # the byte-identical guarantee this class promises -- while a direct route under some
    # other provider would still have received it. Caught by
    # test_litellm_teacher_request_shape, which pins the recorded request shape.
    GATEWAY_SEGMENT = "aws/"

    def _is_direct_anthropic(self) -> bool:
        """True when the request goes straight to the Anthropic API, which accepts neither
        adaptive thinking nor output_config."""
        return self.GATEWAY_SEGMENT not in self.model

    def __init__(self, model: str, api_base: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key

    def complete(self, system: str, user: str) -> Optional[str]:
        from litellm import completion

        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.MAX_TOKENS,
        )
        if not self._is_direct_anthropic():
            kwargs["extra_body"] = self.EXTRA_BODY
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        resp = completion(**kwargs)
        return _flatten_content(resp.choices[0].message.content)


class VLLMTeacher:
    """Teacher backed by an OpenAI-compatible local endpoint (e.g. a served vLLM).

    Mirrors the old skillopt ``LLMOptimizer._call_vllm`` request shape.
    """

    TEMPERATURE = 0.25
    MAX_TOKENS = 1500
    TIMEOUT = 120

    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model

    def complete(self, system: str, user: str) -> Optional[str]:
        import requests

        try:
            resp = requests.post(
                f"{self.url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": self.TEMPERATURE,
                    "max_tokens": self.MAX_TOKENS,
                },
                timeout=self.TIMEOUT,
            )
            if resp.status_code != 200:
                print(f"  Error: vLLM teacher returned {resp.status_code}: {resp.text}")
                return None
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001 - transient teacher failures skip the batch
            import traceback
            print(f"  Error calling vLLM teacher: {e}\n{traceback.format_exc()}")
            return None


def make_teacher(model_id: Optional[str]) -> Optional[TeacherClient]:
    """Build a teacher from a model id.

    - A truthy ``model_id`` (the default: OPTIMIZER_LITELLM_MODEL, e.g.
      ``openai/aws/claude-opus-4-8``) → :class:`LiteLLMTeacher`. This is the
      byte-identical default path. api_base/api_key come from the same env vars
      the old skillopt ``LLMOptimizer`` read.
    - A falsy ``model_id`` → :class:`VLLMTeacher` pointed at a local endpoint from
      OPTIMIZER_URL / OPTIMIZER_MODEL_PATH (mirrors LLMOptimizer's non-litellm branch).

    Returns None only if no teacher can be constructed (no model id and no
    OPTIMIZER_URL) — the caller then disables teacher edits, as before.
    """
    if model_id:
        return LiteLLMTeacher(
            model=model_id,
            api_base=os.getenv("LITELLM_API_BASE"),
            api_key=os.getenv("LITELLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        )
    url = os.getenv("OPTIMIZER_URL")
    if url:
        return VLLMTeacher(url=url, model=os.getenv("OPTIMIZER_MODEL_PATH", ""))
    return None
