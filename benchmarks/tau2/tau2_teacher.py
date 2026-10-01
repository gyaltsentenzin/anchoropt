"""The teacher: an LLM operator that proposes REPROMPT instruction eta for one frozen residual.

WHAT IT IS FOR. `ground_reprompt` ships two deliberately generic instructions. They are honest but say
nothing about the residual actually being repaired, so "does a teacher help?" cannot be asked at all
while the instruction text is a constant. This operator supplies additional eta CANDIDATES, and
measurement decides among them exactly as it decides among the shipped two.

IT FOLLOWS THE SEVEN RULES in `anchoropt/attribution/llm_operator.py`, and here is how:

  1. GATED        off by default. There is no cheaper method that produces residual-specific wording --
                  the deterministic grounder emits fixed strings on purpose -- so this is not an LLM
                  standing in for something that already works; it supplies what the deterministic path
                  cannot. It is still refused when the residual carries no evidence to read.
  2. NARROW       it sees the frozen problem's decision key and its diagnoses' evidence, plus the field
                  names readable at the boundary. It does NOT see the search state, the candidate list,
                  the arms already built, the acceptance rule, prior outcomes, the incumbent's score,
                  or the tool catalog.
  3. STRUCTURED   a JSON array of {variant, instruction}. Prose is a parse failure, not a result.
  4. VALIDATED    every proposal is checked mechanically -- shape, length, uniqueness, no invented tool
                  name, no imperative naming a boundary or an action -- before it can become an arm.
  5. NON-DECIDING it proposes text. It never ranks, never orders, never says which is better, and its
                  output is appended AFTER the shipped variants so position carries no preference.
  6. ATTRIBUTED   every accepted proposal records the model and the provenance string, so a
                  history-only arm and a history+teacher arm are distinguishable in the manifest.
  7. OPTIONAL     `--teacher` selects it; absent, the round runs on shipped Φ eta alone.

WHAT IT MUST NEVER DO, and rule 4 enforces it mechanically: name a locus, an action family, or a tool.
An instruction reading "suppress the call" would be the adapter choosing the action, which is core's
job; one naming a tool the domain does not have is an instruction the agent cannot follow.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Words that would make an instruction a decision about WHERE or HOW to intervene rather than about
# what the agent should do. Rejected mechanically -- see rule 4.
_FORBIDDEN = (
    "suppress", "reprompt", "reroute", "noop", "boundary", "locus", "incision",
    "pre_generation", "post_generation", "post_execution", "anchor", "controller",
)

MAX_INSTRUCTION_CHARS = 320
MIN_INSTRUCTION_CHARS = 20
# One short completion, once per arm. Overridable, but not from the episode timeout: those are sized
# for a multi-turn simulation and would let a silent teacher hold an arm open for many minutes.
TEACHER_TIMEOUT_S = int(os.environ.get("ANCHOROPT_TAU2_TEACHER_TIMEOUT", "240"))

_PROMPT = """\
You are given evidence about how a customer-service agent failed on a set of tasks.

Propose {n} DIFFERENT one-sentence instructions that could be shown to the agent at the moment it is
about to make the decision described below. Each instruction must tell the agent what to do
differently in its own work.

Decision that recurred:
{key}

Observed evidence from {n_cases} failing episodes:
{evidence}

Facts readable at that moment (you may refer to these conditions in plain words, not as field names):
{fields}

Rules for your output:
- Reply with a JSON array only. No prose before or after.
- Each element: {{"variant": "<short_snake_case_label>", "instruction": "<one sentence>"}}
- Each instruction is at most {maxc} characters and addresses the agent directly.
- Do NOT name any tool, function, or API.
- Do NOT mention suppressing, rerouting, reprompting, boundaries, loci, or controllers.
- Do NOT say which of your instructions is best, and do not rank them.
"""


@dataclass
class TeacherProposal:
    """One validated instruction candidate, carrying its own provenance."""

    variant: str
    instruction: str
    model: str = ""
    provenance: str = ""

    def as_grounding(self, *, retry_budget: int = 1, include_budget: bool = True) -> dict[str, Any]:
        eta: dict[str, Any] = {"instruction": self.instruction}
        if include_budget:
            eta["retry_budget"] = retry_budget
        return {"variant": f"teacher_{self.variant}", "eta": eta,
                "detail": f"instruction proposed by {self.provenance or self.model or 'teacher'}",
                "provenance": self.provenance, "model": self.model}


@dataclass
class TeacherResult:
    """What the operator did, in enough detail to audit the claim afterwards."""

    invoked: bool
    reason: str
    proposals: list[TeacherProposal] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    provenance: str = ""
    raw: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"invoked": self.invoked, "reason": self.reason, "model": self.model,
                "provenance": self.provenance,
                "accepted": [{"variant": p.variant, "instruction": p.instruction}
                             for p in self.proposals],
                "rejected": list(self.rejected)}


def _validate(raw_items: Sequence[Any], *, known_tools: frozenset[str],
              ) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Mechanical validation. An unvalidated proposal is not a candidate."""
    ok: list[tuple[str, str]] = []
    bad: list[dict[str, Any]] = []
    seen: set[str] = set()
    tool_words = {t.lower() for t in known_tools}
    for item in raw_items or ():
        if not isinstance(item, Mapping):
            bad.append({"item": str(item)[:80], "why": "not an object"})
            continue
        variant = re.sub(r"[^a-z0-9_]+", "_", str(item.get("variant") or "").strip().lower()).strip("_")
        text = " ".join(str(item.get("instruction") or "").split())
        if not variant:
            bad.append({"item": text[:60], "why": "no usable variant label"})
            continue
        if not (MIN_INSTRUCTION_CHARS <= len(text) <= MAX_INSTRUCTION_CHARS):
            bad.append({"item": text[:60], "why": f"length {len(text)} outside bounds"})
            continue
        low = text.lower()
        hit = next((w for w in _FORBIDDEN if w in low), "")
        if hit:
            bad.append({"item": text[:60], "why": f"names the mechanism ({hit!r}), which is core's choice"})
            continue
        named = next((w for w in tool_words if w and w in low.replace(" ", "_")), "")
        if named:
            bad.append({"item": text[:60], "why": f"names a tool ({named!r})"})
            continue
        if low in seen:
            bad.append({"item": text[:60], "why": "duplicate instruction"})
            continue
        seen.add(low)
        ok.append((variant, text))
    return ok, bad


def _extract_json_array(text: str) -> list[Any] | None:
    """Parse the operator's reply. Prose is a parse failure, not a result."""
    s = str(text or "")
    # Tolerate a fenced block, because a reasoning model often wraps its answer.
    fence = re.search(r"```(?:json)?\s*(.+?)```", s, re.S)
    if fence:
        s = fence.group(1)
    start, end = s.find("["), s.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        got = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None
    return got if isinstance(got, list) else None


def propose_instructions(problem, *, endpoint, provenance: str, boundary_fields: Sequence[str],
                         known_tools: frozenset[str] = frozenset(), n: int = 3,
                         max_evidence: int = 6, complete=None) -> TeacherResult:
    """Ask the teacher for `n` instruction candidates for ONE frozen residual problem.

    `complete(model, messages, **llm_args) -> str` is injectable so the operator can be tested with no
    network. It defaults to litellm, imported lazily so this module stays importable offline.
    """
    diagnoses = list(getattr(problem, "diagnoses", ()) or ())
    evidence = [str(getattr(d, "evidence", "") or "").strip() for d in diagnoses]
    evidence = [e for e in evidence if e][:max_evidence]
    if not evidence:
        # Rule 1: refused rather than run on nothing. A teacher with no evidence to read would be
        # inventing, and an invented instruction is not attributable to the residual.
        return TeacherResult(invoked=False, reason="the residual carries no evidence to read",
                             model=getattr(endpoint, "label", ""), provenance=provenance)

    prompt = _PROMPT.format(
        n=n, key=str(getattr(problem, "key", "") or "(unnamed decision)"),
        n_cases=len(diagnoses),
        evidence="\n".join(f"- {e}" for e in evidence),
        fields=", ".join(sorted(boundary_fields)) or "(none declared)",
        maxc=MAX_INSTRUCTION_CHARS)

    if complete is None:
        def complete(model, messages, **llm_args):           # pragma: no cover - network path
            import litellm
            # BOUNDED, because this runs once per arm inside a 24-job batch and a teacher that never
            # answers must not hold an arm open. The episode timeout (600 s, sized for a long
            # multi-turn simulation) is far too generous for a single short completion, and with
            # retries on top it could stall an arm for half an hour. One retry, a tighter deadline,
            # and a refusal that degrades to history-only rather than raising.
            args = dict(llm_args)
            args["timeout"] = min(int(args.get("timeout") or TEACHER_TIMEOUT_S), TEACHER_TIMEOUT_S)
            r = litellm.completion(model=model, messages=messages, num_retries=1, **args)
            msg = r.choices[0].message
            return str(getattr(msg, "content", "") or "")

    try:
        raw = complete(endpoint.model, [{"role": "user", "content": prompt}],
                       **dict(getattr(endpoint, "llm_args", {}) or {}))
    except Exception as exc:
        return TeacherResult(invoked=False, reason=f"teacher call failed: {type(exc).__name__}: {exc}",
                             model=getattr(endpoint, "label", ""), provenance=provenance)

    items = _extract_json_array(raw)
    if items is None:
        return TeacherResult(invoked=True, reason="reply was not a JSON array (prose is a parse failure)",
                             model=getattr(endpoint, "label", ""), provenance=provenance,
                             raw=str(raw)[:600])
    accepted, rejected = _validate(items, known_tools=known_tools)
    label = getattr(endpoint, "label", "")
    return TeacherResult(
        invoked=True,
        reason=(f"{len(accepted)} of {len(items)} proposals validated" if accepted
                else "no proposal survived validation"),
        proposals=[TeacherProposal(variant=v, instruction=t, model=label, provenance=provenance)
                   for v, t in accepted],
        rejected=rejected, model=label, provenance=provenance, raw=str(raw)[:600])
