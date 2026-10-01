"""Three-channel cost ledger for the GEPA global-prompt baseline.

WHY THREE CHANNELS AND NOT ONE NUMBER. The preregistered ceiling governs *scored-query
evaluations*, because that is the quantity the teacher-loop comparison was derived from
(240 train + 20 x 96.36% x 78 val = 1743). But two other costs are real and must not be
folded in or silently omitted:

  scored_queries   -- the CONTROLLED budget. Train + validation. Ceiling applies here.
  prereq_execs     -- prerequisite / snapshot-store construction episodes. NOT scored, but
                      they run the model and they are expensive. A preamble change
                      invalidates the store (the preamble is injected during storage too),
                      so a naive implementation rebuilds the store per candidate. Omitting
                      this from the report would understate GEPA's true cost by more than
                      the scored budget itself.
  reflection       -- optimizer-side LM calls, tokens, wall. Reported, never capped, never
                      matched across arms.

`scored_queries` is the only channel with a ceiling. `would_exceed` is asked BEFORE running
a batch so the ceiling truncates rather than overshoots.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


class BudgetExhausted(RuntimeError):
    """Raised when a scored-query batch would cross the preregistered ceiling."""


@dataclass
class Ledger:
    max_scored_queries: int

    scored_queries: int = 0
    scored_train: int = 0
    scored_val: int = 0
    prereq_execs: int = 0
    prereq_store_builds: int = 0
    candidate_evaluations: int = 0
    cache_hits: int = 0

    reflection_calls: int = 0
    reflection_prompt_tokens: int = 0
    reflection_completion_tokens: int = 0
    reflection_wall_s: float = 0.0

    scored_wall_s: float = 0.0
    prereq_wall_s: float = 0.0
    started_at: float = field(default_factory=time.time)
    history: list[dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------------- scored queries
    @property
    def remaining(self) -> int:
        return max(0, self.max_scored_queries - self.scored_queries)

    def would_exceed(self, n: int) -> bool:
        return self.scored_queries + n > self.max_scored_queries

    def spend_scored(self, n: int, *, kind: str, label: str,
                     wall_s: float = 0.0, tasks: list[str] | None = None) -> None:
        """Record n scored-query evaluations. `kind` is 'train' or 'val'."""
        if kind not in ("train", "val"):
            raise ValueError(f"kind must be 'train' or 'val', got {kind!r}")
        if self.would_exceed(n):
            raise BudgetExhausted(
                f"{n} scored queries would exceed the ceiling "
                f"({self.scored_queries}/{self.max_scored_queries}); "
                f"{self.remaining} remain")
        self.scored_queries += n
        if kind == "train":
            self.scored_train += n
        else:
            self.scored_val += n
        self.scored_wall_s += wall_s
        self.history.append({"t": time.time(), "event": f"scored_{kind}", "label": label,
                             "n": n, "tasks": tasks or [], "wall_s": round(wall_s, 1),
                             "counted_to_ceiling": True})

    # ------------------------------------------------------------------- prerequisites
    def spend_prereq(self, n_episodes: int, *, label: str, wall_s: float = 0.0,
                     store_build: bool = True) -> None:
        """Record prerequisite/store-building episodes. NOT capped, ALWAYS reported."""
        self.prereq_execs += n_episodes
        if store_build:
            self.prereq_store_builds += 1
        self.prereq_wall_s += wall_s
        self.history.append({"t": time.time(), "event": "prereq_exec", "label": label,
                             "n": n_episodes, "wall_s": round(wall_s, 1),
                             "counted_to_ceiling": False})

    def note_cache_hit(self, *, label: str) -> None:
        self.cache_hits += 1
        self.history.append({"t": time.time(), "event": "store_cache_hit",
                             "label": label, "counted_to_ceiling": False})

    # ---------------------------------------------------------------------- reflection
    def spend_reflection(self, *, prompt_tokens: int | None,
                         completion_tokens: int | None, wall_s: float) -> None:
        self.reflection_calls += 1
        self.reflection_prompt_tokens += int(prompt_tokens or 0)
        self.reflection_completion_tokens += int(completion_tokens or 0)
        self.reflection_wall_s += wall_s
        self.history.append({"t": time.time(), "event": "reflection",
                             "prompt_tokens": prompt_tokens,
                             "completion_tokens": completion_tokens,
                             "wall_s": round(wall_s, 1), "counted_to_ceiling": False})

    # -------------------------------------------------------------------------- report
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["_channels"] = {
            "environment_scored": {
                "_note": "THE controlled budget: scored-query evaluations (train + val). "
                         "Ceiling derived from historical Granite teacher runs "
                         "(240 train + 20 x 96.36% x 78 val = 1743). This is a "
                         "PREREGISTERED CEILING, not a verified match to any observed "
                         "Self-Harness or AnchorOpt run.",
                "max": self.max_scored_queries,
                "spent": self.scored_queries,
                "train": self.scored_train,
                "val": self.scored_val,
                "remaining": self.remaining,
            },
            "environment_prereq": {
                "_note": "Prerequisite / snapshot-store episodes. NOT scored and NOT capped, "
                         "but real model executions. Reported so GEPA's total environment "
                         "cost is never understated.",
                "episodes": self.prereq_execs,
                "store_builds": self.prereq_store_builds,
                "store_cache_hits": self.cache_hits,
                "wall_s": round(self.prereq_wall_s, 1),
            },
            "optimizer": {
                "_note": "Reflection LM cost. Reported transparently; never capped, never "
                         "matched across arms.",
                "calls": self.reflection_calls,
                "prompt_tokens": self.reflection_prompt_tokens,
                "completion_tokens": self.reflection_completion_tokens,
                "wall_s": round(self.reflection_wall_s, 1),
            },
        }
        d["wall_clock"] = {"started_at": self.started_at,
                           "elapsed_s": round(time.time() - self.started_at, 1)}
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_or_new(cls, path: Path, *, max_scored_queries: int) -> "Ledger":
        """Resume an interrupted run without double-spending the ceiling."""
        if not path.exists():
            return cls(max_scored_queries=max_scored_queries)
        d = json.loads(path.read_text())
        for k in ("_channels", "wall_clock"):
            d.pop(k, None)
        led = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        led.max_scored_queries = max_scored_queries
        led.history.append({"t": time.time(), "event": "ledger_restored",
                            "scored_queries": led.scored_queries,
                            "remaining": led.remaining})
        return led
