# Copyright 2026 <author>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
vLLM client for local model serving, matching anchoropt's pipeline.
Uses HTTP requests (like bfcl_v4) to communicate with vLLM endpoint.
"""

import requests
import json


class VLLMClient:
    """Client for vLLM-served models via HTTP API (no SDK dependency)."""

    def __init__(self, base_url="http://localhost:8080/v1", model_name="granite-4.1-8b", timeout=3600,
                 temperature=0.0, thinking=False, max_tokens=1024):
        """
        Initialize vLLM client.

        Args:
            base_url: vLLM server base URL (e.g., http://compute-042:8000 or http://compute-042:8000/v1)
            model_name: Model name as registered in vLLM -- must match --served-model-name
            timeout: Request timeout in seconds
            temperature: decode temperature. Held on the CLIENT, as the the hosted API clients do, because the
                harness only forwards `seed` in gen_kwargs -- reading temperature from **kwargs meant
                `--temperature 0.7 --use-vllm` silently ran at 0 while the hosted API honoured it.
            thinking: send chat_template_kwargs={"enable_thinking": ...}. The the hosted API clients pass this
                through extra_body; this client dropped it entirely, so `--thinking` was a no-op over
                vLLM -- which matters because qwen-3-6-35b and minimax-m2-5 are thinking models.
            max_tokens: was hardcoded to 1024. A thinking model routinely needs more than that for a
                single turn, and a truncated turn is recorded as an ordinary short answer.
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.temperature = temperature
        self.thinking = thinking
        self.max_tokens = max_tokens

        # vLLM can serve at either /v1/chat/completions or /chat/completions
        # Try to construct the endpoint intelligently
        if self.base_url.endswith('/v1'):
            # If base_url already includes /v1, append directly
            self.api_endpoint = f"{self.base_url}/chat/completions"
        elif '/v1' in self.base_url:
            # If /v1 is somewhere in the URL, use as-is
            self.api_endpoint = f"{self.base_url}/chat/completions"
        else:
            # Otherwise, try /v1/chat/completions first (try /v1 prefix)
            self.api_endpoint = f"{self.base_url}/v1/chat/completions"

    def chat(self, messages, tools=None, **kwargs):
        """
        Send a chat message to vLLM.

        Args:
            messages: List of message dicts with role and content
            tools: Optional list of tool definitions
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            dict: Response in OpenAI format with choices, usage, etc.
        """
        # Build request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            # NO DEFAULT. This used to be `kwargs.get("seed", 42)`, which substituted a seed
            # whenever the harness sent none -- so `--no-seed` was a statement the run did not
            # honour, and an "unseeded" run was silently seeded at 42. The harness now states
            # every decode value it relies on, and omitting one here means it was omitted.
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]

        # Only sent when explicitly requested, so a non-thinking run's payload is byte-identical to
        # before. vLLM takes this through chat_template_kwargs, not the extra_body the the hosted API clients
        # use.
        if self.thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}

        # Add tools if provided
        if tools:
            payload["tools"] = tools

        # Add any additional parameters
        for key in ["top_p", "top_k", "frequency_penalty", "presence_penalty"]:
            if key in kwargs:
                payload[key] = kwargs[key]

        # Make HTTP request to vLLM
        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Provide helpful error message
            print(f"[vLLM Error] Status: {e.response.status_code}")
            print(f"[vLLM Error] Endpoint tried: {self.api_endpoint}")
            print(f"[vLLM Error] Base URL: {self.base_url}")
            print(f"[vLLM Error] Model: {self.model_name}")
            print(f"[vLLM Error] Response: {e.response.text[:500]}")
            raise

        # Return as dict in OpenAI format
        return response.json()


def vllm_client(model_name="granite-4.1-8b", base_url="http://localhost:8080/v1",
                temperature=0.0, thinking=False, max_tokens=1024):
    """Factory function to create a vLLM client matching anchoropt's pipeline."""
    return VLLMClient(base_url=base_url, model_name=model_name, temperature=temperature,
                      thinking=thinking, max_tokens=max_tokens)
