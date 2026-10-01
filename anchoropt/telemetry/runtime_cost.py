"""Per-episode and per-arm inference cost. MEASURABLE, not yet claimed.

Explicitly NOT asserting that AnchorOpt is cheaper than anything. The point is to make the question
answerable: a controller that buys accuracy with extra generations has a cost, and an evaluation
that reports only accuracy cannot see it.

WHAT IS AND IS NOT AVAILABLE IN THIS SUBSTRATE
---------------------------------------------
Read from the trajectory sidecar, whose per-step record carries `status`, `decoded`, `injection_key`
and the executor's `*_gate` flags. So steps, tool calls, retrieval calls, interventions and the
fired flag are DIRECTLY observable.

Token counts and per-episode latency are NOT in the sidecar on this substrate. Every field for them
is therefore Optional and defaults to None -- never to 0, because a missing measurement and a
measured zero are different facts and averaging them together is how a cost claim becomes wrong.
`ArmCost.tokens_available` says which is the case, and any per-token summary returns None when the
underlying numbers are absent rather than reporting a confident 0.

LLM CALLS is derived, with its assumption stated: one generation per step. That holds for this
evaluator (each step is one decode, and a reprompt injection adds a step), so llm_calls == steps.
It is recorded as its own field rather than aliased, so a substrate where that is false can override
it without every downstream ratio silently changing meaning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_READ_MARKERS = ("retrieve", "search")


@dataclass(frozen=True)
class EpisodeCost:
    """One episode under one arm. None means NOT MEASURED, never zero."""

    case_id: str
    steps: int
    llm_calls: int
    tool_calls: int
    retrieval_calls: int
    interventions: int
    fired: bool
    correct: bool | None = None
    input_tokens: int | None = None
    generated_tokens: int | None = None
    latency_s: float | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.generated_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.generated_tokens or 0)


def episode_cost(case_id: str, steps: Sequence[Mapping[str, Any]], *,
                 fire_key: str = "zero_call_reprompt_gate",
                 correct: bool | None = None,
                 usage: Mapping[str, Any] | None = None) -> EpisodeCost:
    """Cost of one episode, read from its sidecar steps.

    `usage` is an optional injection point for a substrate that DOES report tokens/latency; absent
    it, those fields stay None.
    """
    n_steps = 0
    tool_calls = retrieval = interventions = 0
    for s in steps:
        if not isinstance(s, Mapping):
            continue
        n_steps += 1
        decoded = [str(c) for c in (s.get("decoded") or [])]
        tool_calls += len(decoded)
        retrieval += sum(1 for c in decoded if any(m in c for m in _READ_MARKERS))
        if s.get(fire_key):
            interventions += 1
    u = dict(usage or {})
    return EpisodeCost(
        case_id=str(case_id), steps=n_steps, llm_calls=n_steps, tool_calls=tool_calls,
        retrieval_calls=retrieval, interventions=interventions, fired=interventions > 0,
        correct=correct,
        input_tokens=u.get("input_tokens"), generated_tokens=u.get("generated_tokens"),
        latency_s=u.get("latency_s"))


@dataclass
class ArmCost:
    """Aggregated cost for one arm, split by whether the controller touched the episode."""

    arm: str
    episodes: tuple[EpisodeCost, ...] = ()

    # ---- basic totals
    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def fired(self) -> tuple[EpisodeCost, ...]:
        return tuple(e for e in self.episodes if e.fired)

    @property
    def untouched(self) -> tuple[EpisodeCost, ...]:
        return tuple(e for e in self.episodes if not e.fired)

    @property
    def tokens_available(self) -> bool:
        return any(e.total_tokens is not None for e in self.episodes)

    def _sum(self, attr: str, subset: Sequence[EpisodeCost] | None = None) -> int:
        return sum(getattr(e, attr) for e in (self.episodes if subset is None else subset))

    def _mean(self, attr: str, subset: Sequence[EpisodeCost] | None = None) -> float | None:
        pool = self.episodes if subset is None else subset
        return (sum(getattr(e, attr) for e in pool) / len(pool)) if pool else None

    # ---- the four derived quantities the brief asks for
    def overhead_on_fired(self, baseline: "ArmCost") -> dict[str, float | None]:
        """Extra cost per episode the controller TOUCHED, versus the same episodes in `baseline`."""
        return self._overhead({e.case_id for e in self.fired}, baseline)

    def overhead_on_untouched(self, baseline: "ArmCost") -> dict[str, float | None]:
        """Extra cost on episodes it did NOT touch. Should be ~0; a non-zero value means the arm
        perturbed episodes it never fired on, which is a locality violation, not a cost finding."""
        return self._overhead({e.case_id for e in self.untouched}, baseline)

    def _overhead(self, ids: set[str], baseline: "ArmCost") -> dict[str, float | None]:
        base = {e.case_id: e for e in baseline.episodes}
        pairs = [(e, base[e.case_id]) for e in self.episodes
                 if e.case_id in ids and e.case_id in base]
        if not pairs:
            return {"n": 0, "steps": None, "tool_calls": None, "retrieval_calls": None,
                    "total_tokens": None}
        out: dict[str, float | None] = {"n": len(pairs)}
        for attr in ("steps", "tool_calls", "retrieval_calls"):
            out[attr] = sum(getattr(a, attr) - getattr(b, attr) for a, b in pairs) / len(pairs)
        if self.tokens_available and baseline.tokens_available:
            out["total_tokens"] = sum((a.total_tokens or 0) - (b.total_tokens or 0)
                                      for a, b in pairs) / len(pairs)
        else:
            out["total_tokens"] = None
        return out

    def cost_per_successful_episode(self) -> dict[str, float | None]:
        """Cost divided by episodes that ended CORRECT. None when correctness was not supplied."""
        wins = [e for e in self.episodes if e.correct]
        if not wins:
            return {"n_correct": 0, "steps": None, "total_tokens": None}
        out: dict[str, float | None] = {"n_correct": len(wins)}
        out["steps"] = self._sum("steps") / len(wins)
        out["total_tokens"] = ((sum(e.total_tokens or 0 for e in self.episodes) / len(wins))
                              if self.tokens_available else None)
        return out

    def cost_per_corrected_failure(self, baseline: "ArmCost") -> dict[str, float | None]:
        """Total cost of the arm divided by the failures it actually CONVERTED.

        The denominator is conversions relative to `baseline`, not raw successes: the interesting
        cost is what a fix costs, and an arm inherits the baseline's wins for free.
        """
        base = {e.case_id: e for e in baseline.episodes}
        conv = [e for e in self.episodes
                if e.correct and e.case_id in base and base[e.case_id].correct is False]
        if not conv:
            return {"n_converted": 0, "steps": None, "total_tokens": None}
        out: dict[str, float | None] = {"n_converted": len(conv)}
        extra_steps = sum(e.steps - base[e.case_id].steps
                          for e in self.episodes if e.case_id in base)
        out["steps"] = extra_steps / len(conv)
        out["total_tokens"] = None
        if self.tokens_available and baseline.tokens_available:
            out["total_tokens"] = sum((e.total_tokens or 0) - (base[e.case_id].total_tokens or 0)
                                      for e in self.episodes if e.case_id in base) / len(conv)
        return out

    def as_dict(self, baseline: "ArmCost | None" = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "arm": self.arm, "n_episodes": self.n, "n_fired": len(self.fired),
            "tokens_available": self.tokens_available,
            "totals": {a: self._sum(a) for a in
                       ("steps", "llm_calls", "tool_calls", "retrieval_calls", "interventions")},
            "mean_per_episode": {a: self._mean(a) for a in
                                 ("steps", "llm_calls", "tool_calls", "retrieval_calls")},
            "cost_per_successful_episode": self.cost_per_successful_episode(),
        }
        lat = [e.latency_s for e in self.episodes if e.latency_s is not None]
        d["latency_s"] = {"n": len(lat), "total": sum(lat) if lat else None,
                          "mean": (sum(lat) / len(lat)) if lat else None}
        if baseline is not None:
            d["overhead_on_fired"] = self.overhead_on_fired(baseline)
            d["overhead_on_untouched"] = self.overhead_on_untouched(baseline)
            d["cost_per_corrected_failure"] = self.cost_per_corrected_failure(baseline)
        return d


def arm_cost(arm: str, steps_by_case: Mapping[str, Sequence[Mapping[str, Any]]], *,
             correct_by_case: Mapping[str, bool] | None = None,
             usage_by_case: Mapping[str, Mapping[str, Any]] | None = None,
             fire_key: str = "zero_call_reprompt_gate") -> ArmCost:
    correct_by_case = correct_by_case or {}
    usage_by_case = usage_by_case or {}
    eps = tuple(episode_cost(c, st, fire_key=fire_key, correct=correct_by_case.get(c),
                             usage=usage_by_case.get(c))
                for c, st in sorted(steps_by_case.items()))
    return ArmCost(arm=arm, episodes=eps)


def cost_deltas(arm: ArmCost, baseline: ArmCost) -> dict[str, Any]:
    """The cost side of a paired comparison, in one object."""
    return {"arm": arm.arm, "baseline": baseline.arm,
            "overhead_on_fired": arm.overhead_on_fired(baseline),
            "overhead_on_untouched": arm.overhead_on_untouched(baseline),
            "cost_per_corrected_failure": arm.cost_per_corrected_failure(baseline),
            "tokens_measured": arm.tokens_available and baseline.tokens_available}
