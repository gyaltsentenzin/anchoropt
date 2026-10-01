"""Addressing the models a benchmark arm needs: the agent under test, the user simulator, the teacher.

THREE ROLES, THREE HOSTS, and keeping them apart is the point:

  agent    the model being benchmarked. Served through a shared research inference gateway.
  user     the user simulator. One model for the whole matrix, so the simulated customer is a
           CONSTANT and not a second variable -- an arm that looks better because its user was
           easier is not a result about the agent.
  teacher  the LLM operator's model. `self` means the agent model teaches itself; anything else is
           an external teacher on a separate LiteLLM gateway. Recorded either way, so a history-only
           run and a history+teacher run stay distinguishable forever.

THE SHARED GATEWAY'S AUTH IS NOT BEARER AUTH. It authenticates with a bespoke `HOSTED_API_KEY`
**header**; `Authorization: Bearer` is rejected with "Authentication parameters missing", so
`api_key` plumbing alone is not enough and every call must carry `extra_headers`. The slug/id/context
table below is taken from a verified table rather than guessed, and should be pinned by a test if
you port this to your own gateway.

REASONING MODELS NEED HEADROOM. All three models were smoke-probed at `max_tokens=12` and two
returned `content=None`: the reasoning tokens consumed the whole budget. qwen3.6 accepts the thinking
switch and is turned off; minimax-m2.5 ignores it and reasons regardless, so nothing here caps
`max_tokens` and the arm relies on the endpoint's own native limit.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# No default. The gateway we happened to serve these models through was an internal host, and baking
# its name in as a fallback both leaks it and lets a misconfigured run silently target the wrong
# endpoint. Any OpenAI-compatible base URL works -- vLLM, Ollama, TGI, a hosted API. Same `_require`
# shape as benchmarks/cctu/utils/hosted_client.py, for the same reason.
HOSTED_API_BASE = os.environ.get("HOSTED_API_BASE", "").rstrip("/")


def require_api_base() -> str:
    """The configured endpoint, or a message naming what to set."""
    if not HOSTED_API_BASE:
        raise RuntimeError(
            "HOSTED_API_BASE is not set. Point it at any OpenAI-compatible base URL, e.g. "
            "export HOSTED_API_BASE=http://localhost:8000/v1 -- nothing is hardcoded.")
    return HOSTED_API_BASE

# name -> (url slug, model id the endpoint expects, native max_model_len)
HOSTED_MODELS: Mapping[str, tuple[str, str, int]] = {
    "granite-4.1-30b": ("granite-4-1-30b", "ibm-granite/granite-4.1-30b", 131072),
    "minimax-m2.5": ("minimax-m2-5", "MiniMaxAI/MiniMax-M2.5", 196608),
    "qwen3.6-35b-a3b": ("qwen3-6-35b-a3b-a100", "Qwen/Qwen3.6-35B-A3B", 262144),
}

# Per-model request extras. qwen3.6 honours the thinking switch; minimax does not (measured).
MODEL_EXTRA_BODY: Mapping[str, Mapping[str, Any]] = {
    "qwen3.6-35b-a3b": {"chat_template_kwargs": {"enable_thinking": False}},
}

SELF_TEACH = "self"

# WHERE A MODEL IS SERVED FROM. Same weights, different portal: the shared gateway is a research
# inference service, `vllm` is a model we serve ourselves on our own GPUs. Latency and throughput are
# free to differ -- that is the point of having the choice -- but four things are OURS to keep equal,
# because getting them wrong means we sent different input rather than the model behaving
# differently: the chat template, the tool-call parser, the quantization, and max_model_len.
#
# Defaults to the shared gateway from the environment, so a process that sets nothing behaves exactly
# as before. That matters while a matrix is mid-flight: pending arms import this module fresh when
# they start, and a change of default would silently split the run across two portals.
PORTAL_HOSTED, PORTAL_VLLM = "hosted", "vllm"
DEFAULT_PORTAL = os.environ.get("ANCHOROPT_TAU2_PORTAL", PORTAL_HOSTED)

# Written by the serving job only after /v1/models answers -- but it also SURVIVES the job's death, so
# it is evidence of a past success, not of current liveness. `vllm_endpoint` re-checks the port.
# No default: the previous one was a specific user's project directory on our filesystem, which is
# both a path nobody else has and information about our layout. Set it to wherever your runner writes
# its `endpoint_<name>.txt` files.
SERVING_DIR = os.environ.get("ANCHOROPT_TAU2_SERVING_DIR", "")

# NOTHING IS HARDCODED ABOUT THE SERVED NAME. vLLM advertises the model under whatever
# `--served-model-name` it was launched with -- here the short key, not the weights path. Guessing the
# path produced `NotFoundError: The model ... does not exist` on every call, so the id is read from
# /v1/models instead, which is correct whatever the serving job chose.


@dataclass(frozen=True)
class Endpoint:
    """Everything needed to address one model through tau-bench's `generate`."""

    model: str                              # what litellm is called with
    llm_args: Mapping[str, Any] = field(default_factory=dict)
    provider: str = ""
    label: str = ""

    def args_with(self, **extra: Any) -> dict[str, Any]:
        out = dict(self.llm_args)
        out.update(extra)
        return out


def _gateway_env() -> tuple[str, str]:
    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or ""
    key = os.environ.get("OPENAI_API_KEY") or ""
    if not base or not key:
        raise SystemExit(
            "OPENAI_BASE_URL / OPENAI_API_KEY are not set. They live in the tau-bench checkout's "
            ".env; export them with `set -a && . $ANCHOROPT_TAU2_REPO/.env && set +a`.")
    return base, key


def hosted_endpoint(name: str, *, temperature: float = 0.0, timeout: int = 600) -> Endpoint:
    if name not in HOSTED_MODELS:
        raise SystemExit(f"{name!r} is not a known hosted model; known: {sorted(HOSTED_MODELS)}")
    slug, served, _max_len = HOSTED_MODELS[name]
    key = os.environ.get("HOSTED_API_KEY") or ""
    if not key:
        raise SystemExit(
            f"{name} is served through the shared gateway but HOSTED_API_KEY is not set. It "
            f"lives in the tau-bench checkout's .env.")
    args: dict[str, Any] = {
        "api_base": f"{require_api_base()}/{slug}/v1",
        "api_key": "dummy",
        # NOT Authorization: Bearer -- the shared gateway rejects that outright.
        "extra_headers": {"HOSTED_API_KEY": key},
        "temperature": temperature,
        "timeout": timeout,
    }
    if name in MODEL_EXTRA_BODY:
        args["extra_body"] = dict(MODEL_EXTRA_BODY[name])
    return Endpoint(model=f"openai/{served}", llm_args=args, provider=PORTAL_HOSTED, label=name)


def gateway_endpoint(model: str, *, temperature: float = 0.0, timeout: int = 600) -> Endpoint:
    """A model on the IBM LiteLLM gateway (the user simulator, the NL judge, an external teacher)."""
    base, key = _gateway_env()
    name = model if model.startswith("openai/") else f"openai/{model}"
    return Endpoint(model=name,
                    llm_args={"api_base": base, "api_key": key, "temperature": temperature,
                              "timeout": timeout},
                    provider="gateway", label=model)


def vllm_endpoint(name: str, *, temperature: float = 0.0, timeout: int = 600) -> Endpoint:
    """A model we serve ourselves. Raises if the endpoint is not actually answering.

    Failing here is the point: a stale endpoint file pointed a client at a dead server for a day, and
    the alternative to raising is every episode failing at the harness and being recorded as void.
    """
    import json as _json
    import urllib.request
    from pathlib import Path as _Path

    if not SERVING_DIR:
        raise SystemExit(
            "FATAL: ANCHOROPT_TAU2_SERVING_DIR is not set. It is the directory your serving job "
            "writes `endpoint_<name>.txt` into; there is no default.")
    f = _Path(SERVING_DIR) / f"endpoint_{name}.txt"
    if not f.exists():
        raise SystemExit(f"FATAL: {name} has no endpoint file at {f}. Submit the serving job first.")
    parts = f.read_text().split()
    if len(parts) < 2:
        raise SystemExit(f"FATAL: {f} is malformed: {parts!r}")
    host, port = parts[0], int(parts[1])
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=20) as r:
            listing = _json.loads(r.read())
        served = str((listing.get("data") or [{}])[0].get("id") or "")
    except Exception as exc:
        raise SystemExit(
            f"FATAL: {name}'s endpoint file names {host}:{port} but nothing answers there "
            f"({type(exc).__name__}). The file survives the serving job's death, so it is not "
            f"evidence of liveness. Re-submit the serving job.") from exc
    if not served:
        raise SystemExit(f"FATAL: {host}:{port} answered /v1/models but advertised no model id.")
    args: dict[str, Any] = {"api_base": f"http://{host}:{port}/v1", "api_key": "dummy",
                            "temperature": temperature, "timeout": timeout}
    if name in MODEL_EXTRA_BODY:
        args["extra_body"] = dict(MODEL_EXTRA_BODY[name])
    return Endpoint(model=f"openai/{served}", llm_args=args, provider=PORTAL_VLLM, label=name)


def agent_endpoint(name: str, *, portal: str = "", **kw) -> Endpoint:
    """The model under test. `portal` selects where it is served from; default is the shared gateway."""
    which = (portal or DEFAULT_PORTAL).lower()
    if which == PORTAL_VLLM:
        return vllm_endpoint(name, **kw)
    return hosted_endpoint(name, **kw) if name in HOSTED_MODELS else gateway_endpoint(name, **kw)


def teacher_endpoint(spec: str, *, agent_model: str, temperature: float = 0.0,
                     timeout: int = 600) -> tuple[Endpoint | None, str]:
    """Resolve the teacher. Returns (endpoint, provenance); (None, "history_only") when disabled.

    `self` reaches the agent model the same way the agent does, shared gateway included, so
    self-teach is a real configuration rather than a relabelled external call.
    """
    spec = str(spec or "").strip()
    if not spec or spec.lower() in {"none", "off", "history_only"}:
        return None, "history_only"
    if spec.lower() == SELF_TEACH:
        if not agent_model:
            raise SystemExit("--teacher self requires --agent-model")
        return agent_endpoint(agent_model, temperature=temperature, timeout=timeout), \
            f"self_teach:{agent_model}"
    return gateway_endpoint(spec, temperature=temperature, timeout=timeout), f"teacher:{spec}"
