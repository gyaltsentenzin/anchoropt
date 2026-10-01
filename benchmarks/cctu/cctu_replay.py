"""Replay a frozen `response.jsonl` instead of calling a model. For verifying plumbing with no GPU.

Satisfies the same interface the live clients do -- `.chat(messages=, tools=, **gen)` returning an
OpenAI-shaped payload -- so `sample_process` needs no branch for it. `needs_episode_id` is True
because a replay must know WHICH episode's recorded turns to hand back, and `sample_process` already
passes `episode_id` to any client that asks for it.

WHAT THIS IS FOR, AND WHAT IT CANNOT TELL YOU
---------------------------------------------
It exercises the wiring: the three hooks fire, a directive is applied, the trace is written, the
scorer runs. What it cannot do is tell you whether an intervention HELPS, because the recorded
generations do not respond to it -- the model that produced them never saw the injected instruction.

So a replay is a plumbing test, and the useful arm to replay is the **control**, where the trajectory
should not diverge at all. `verify_plumbing.py` is what turns that into an assertion.

DIVERGENCE IS THE POINT, NOT A PROBLEM
--------------------------------------
On an intervention arm the replay WILL run out of recorded turns, because the intervention changes how
many turns the episode takes. That exhaustion is the correct outcome and it is raised loudly:
`ReplayExhausted`. The alternative -- handing back the last recorded message again, or synthesizing a
final answer -- would fabricate a trajectory and let a plumbing test report success on an episode that
never happened. A run that exhausts is a run whose behaviour changed, which is information.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ReplayExhausted(RuntimeError):
    """An episode asked for more turns than the frozen transcript recorded.

    Expected on an intervention arm, where the trajectory legitimately diverges. On a CONTROL arm it
    means the plumbing is not inert -- which is exactly what `verify_plumbing.py` is looking for, so
    this must stay a loud failure rather than a recoverable one.
    """


class ReplayClient:
    """Hands back the recorded assistant turns of each episode, in order.

    Thread-safe: `response_generator` runs episodes in a `ThreadPoolExecutor`, so the per-episode
    cursors live in a dict behind a lock. Each episode reads only its own cursor, so contention is
    limited to the increment.
    """

    needs_episode_id = True

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._turns: dict[str, list[dict[str, Any]]] = {}
        self._cursor: dict[str, int] = {}
        self._lock = threading.Lock()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = str(row.get("id"))
            self._turns[episode_id] = [
                dict(m) for m in (row.get("messages") or ())
                if isinstance(m, Mapping) and m.get("role") == "assistant"
            ]

    @property
    def episodes(self) -> tuple[str, ...]:
        return tuple(sorted(self._turns))

    def turns_for(self, episode_id: str) -> int:
        return len(self._turns.get(str(episode_id), ()))

    def chat(self, messages: Sequence[Mapping[str, Any]] | None = None,
             tools: Any = None, *, episode_id: str = "", **_: Any) -> dict[str, Any]:
        """The next recorded assistant turn for `episode_id`, in the live client's payload shape.

        `messages` is accepted and IGNORED, which is the whole nature of a replay: the recorded
        generation cannot respond to a conversation it never saw. Ignoring it silently would be the
        problem; it is documented here and asserted by `verify_plumbing.py`, which compares a control
        replay against the frozen transcript it replays.
        """
        key = str(episode_id)
        if key not in self._turns:
            raise ReplayExhausted(
                f"{self.path} has no recorded episode {key!r}. Replay needs the SAME corpus split "
                f"and --repeat as the run that produced it; known episodes: {len(self._turns)}")
        with self._lock:
            index = self._cursor.get(key, 0)
            self._cursor[key] = index + 1
        recorded = self._turns[key]
        if index >= len(recorded):
            raise ReplayExhausted(
                f"episode {key!r} asked for turn {index + 1} and the transcript records "
                f"{len(recorded)}. On an intervention arm this is EXPECTED -- the arm changed how "
                f"many turns the episode takes. On a control arm it means the plumbing is not inert.")
        return {"choices": [{"message": dict(recorded[index])}]}
