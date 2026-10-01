# Copyright 2026 Junjie Ye
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


import os

from openai import OpenAI


# CREDENTIALS AND ENDPOINT COME FROM THE ENVIRONMENT, NEVER FROM THIS FILE.
#
# This module previously carried a literal API key and the provider's internal base URL as
# module-level constants. A key in a source file is readable by anyone who can read the file, survives
# copies of the tree, and cannot be rotated without editing code. Set these instead:
#
#     export HOSTED_API_KEY=...
#     export HOSTED_API_BASE_URL=https://<provider-endpoint>
#
# `_require` raises at import-with-use rather than sending an unauthenticated request, because a 401
# from a provider is much harder to read back to a missing variable than this message is.
def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. This client reads its credentials and endpoint from the "
            f"environment; nothing is hardcoded. Set HOSTED_API_KEY and HOSTED_API_BASE_URL before use."
        )
    return value


def _base_url() -> str:
    return _require("HOSTED_API_BASE_URL")


def _api_key() -> str:
    return _require("HOSTED_API_KEY")


# THE EXTRA AUTH HEADER'S NAME IS THE PROVIDER'S, NOT OURS, so it is configuration rather than a
# literal. This client sent a second, provider-specific header alongside the bearer token, and that
# header's name was the provider's product name — which the anonymisation pass had to remove and cannot
# simply rename, because the provider matches on the exact string. Renaming it in place would have left
# code that looks right and authenticates as nobody.
#
# So: set HOSTED_API_KEY_HEADER to whatever header the provider expects and it is sent; leave it unset
# and only standard bearer auth goes out, which is correct for any OpenAI-compatible endpoint that does
# not want a second header.
def _extra_headers() -> dict[str, str] | None:
    name = os.environ.get("HOSTED_API_KEY_HEADER")
    return {name: _api_key()} if name else None


class Qwen:
    def __init__(self, model=None, thinking=True, temperature=0.0, timeout=3600):
        assert model in ["Qwen/Qwen3.6-35B-A3B"], f"model {model} is not supported"
        self.client = OpenAI(
            api_key=_api_key(),
            base_url=f"{_base_url()}/qwen3-6-35b-a3b-a100/v1",
            default_headers=_extra_headers(),
        )
        self.model = model
        self.thinking = thinking
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages, tools=None, **kwargs):
        if self.thinking:
            enable_thinking = True
        else:
            enable_thinking = False
        responses = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            extra_body={"enable_thinking": enable_thinking},
            temperature=self.temperature,
            timeout=self.timeout,
            **kwargs
        )
        responses = responses.model_dump()
        return responses


class Deepseek:
    def __init__(self, model=None, thinking=True, temperature=0.0, timeout=3600):
        assert model in ["deepseek-ai/DeepSeek-V3.2"], f"model {model} is not supported"
        self.client = OpenAI(
            api_key=_api_key(),
            base_url=f"{_base_url()}/deepseek-v3-2/v1",
            default_headers=_extra_headers(),
        )
        self.model = model
        self.thinking = thinking
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages, tools=None, **kwargs):
        if self.thinking:
            enable_thinking = True
        else:
            enable_thinking = False
        responses = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            extra_body={"enable_thinking": enable_thinking},
            temperature=self.temperature,
            timeout=self.timeout,
            **kwargs
        )
        responses = responses.model_dump()
        return responses


class MiniMax:
    def __init__(self, model=None, thinking=True, temperature=0.0, timeout=3600):
        assert model in ["MiniMaxAI/MiniMax-M2.5"], f"model {model} is not supported"
        self.client = OpenAI(
            api_key=_api_key(),
            base_url=f"{_base_url()}/minimax-m2-5/v1",
            default_headers=_extra_headers(),
        )
        self.model = model
        self.thinking = thinking
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages, tools=None, **kwargs):
        if self.thinking:
            enable_thinking = True
        else:
            enable_thinking = False
        responses = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            extra_body={"enable_thinking": enable_thinking},
            temperature=self.temperature,
            timeout=self.timeout,
            **kwargs
        )
        responses = responses.model_dump()
        return responses


class Granite:
    def __init__(self, model=None, thinking=True, temperature=0.0, timeout=3600):
        assert model in ["ibm-granite/granite-4.1-8b"], f"model {model} is not supported"
        self.client = OpenAI(
            api_key=_api_key(),
            base_url=f"{_base_url()}/granite-4-1-8b/v1",
            default_headers=_extra_headers(),
        )
        self.model = model
        self.thinking = thinking
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages, tools=None, **kwargs):
        if self.thinking:
            enable_thinking = True
        else:
            enable_thinking = False
        responses = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            extra_body={"enable_thinking": enable_thinking},
            temperature=self.temperature,
            timeout=self.timeout,
            **kwargs
        )
        responses = responses.model_dump()
        return responses


def client(model, thinking=True, temperature=0.0):
    match model:
        case 'qwen-3-6-35b':
            return Qwen(model='Qwen/Qwen3.6-35B-A3B', thinking=thinking, temperature=temperature)
        case 'deepseek-v3-2':
            return Deepseek(model='deepseek-ai/DeepSeek-V3.2', thinking=thinking, temperature=temperature)
        case 'minimax-m2-5':
            return MiniMax(model='MiniMaxAI/MiniMax-M2.5', thinking=thinking, temperature=temperature)
        case 'granite-4-1-8b':
            return Granite(model='ibm-granite/granite-4.1-8b', thinking=thinking, temperature=temperature)
        case _:
            raise ValueError(f"model {model} is not supported")
