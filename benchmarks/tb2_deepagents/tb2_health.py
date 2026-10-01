"""Fail fast when the model returns nothing actionable, and report it as INFRASTRUCTURE.

WHY THIS IS A MIDDLEWARE, NOT A WRAPPER PATCH
---------------------------------------------
The failure being caught: a reasoning model whose output budget is consumed by reasoning tokens
returns an AIMessage with `finish_reason=length`, **empty content, and no tool_calls**. LangChain
has nothing to dispatch, loops, and the agent emits `event_count=0` until the agent timeout. Two
TB2 control trials burned 30 and 60 minutes that way, and the reward/exception columns record it
as `AgentTimeoutError` -- indistinguishable from an agent that genuinely ran out of time.

`after_model` sees every generation, so the check lives here rather than inside the wrapper's
frozen streaming loop. It is also purely observational until the threshold trips.

WHY THE DISTINCTION MATTERS FOR THE EXPERIMENT
----------------------------------------------
`AgentTimeoutError` is a TASK outcome and must never be retried -- retrying it would launder a
real failure into a better one. "The model never returned usable content" is an INFRASTRUCTURE
outcome and SHOULD be retried. Conflating them is what let two uninformative trials sit in a
paired denominator as though the agent had tried and failed.
"""

from __future__ import annotations

from typing import Any


class EmptyGenerationError(RuntimeError):
    """The model returned no content and no tool calls, repeatedly.

    Deliberately NOT a subclass of anything Harbor treats as an agent failure: the run harness
    should surface this as its own class so Guard 2 can retry it.
    """


def build_health_middleware(*, max_consecutive_empty: int = 2):
    """Middleware that raises after `max_consecutive_empty` unusable generations.

    2 by default: one empty generation can be a transient truncation, two in a row on the same
    episode is the configuration failure. The counter RESETS on any usable generation, so a single
    blip cannot accumulate across an otherwise healthy episode.
    """
    from langchain.agents.middleware import AgentMiddleware

    class HealthMiddleware(AgentMiddleware):
        def __init__(self) -> None:
            super().__init__()
            self.consecutive_empty = 0
            self.empty_total = 0
            self.generations = 0

        def after_model(self, state, runtime=None):
            msgs = list((state or {}).get("messages") or ())
            if not msgs:
                return None
            ai = msgs[-1]
            if type(ai).__name__ != "AIMessage":
                return None
            self.generations += 1

            content = getattr(ai, "content", None)
            text = ""
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                # content_blocks form: join any text parts
                parts = []
                for b in content:
                    if isinstance(b, dict) and isinstance(b.get("text"), str):
                        parts.append(b["text"])
                text = " ".join(parts).strip()
            tool_calls = list(getattr(ai, "tool_calls", []) or [])

            if text or tool_calls:
                self.consecutive_empty = 0
                return None

            self.consecutive_empty += 1
            self.empty_total += 1
            if self.consecutive_empty >= max_consecutive_empty:
                raise EmptyGenerationError(
                    f"{self.consecutive_empty} consecutive generations returned no content and no "
                    f"tool calls (of {self.generations} total). This is the reasoning-budget "
                    f"exhaustion signature, not an agent failure: raise max_tokens "
                    f"(SH_MAX_TOKENS) rather than retrying the task.")
            return None

    return HealthMiddleware()
