"""The editable surface: exactly one template key, enforced structurally.

THE EXPERIMENTAL CLAIM THIS FILE DEFENDS. The GEPA arm differs from H0 in exactly one way:
the text of `on_memory_preamble`. Every other template key, every gate, every remedy flag,
every memory backend and all execution logic is byte-identical to H0. If that is not true,
the arm measures something other than global-prompt optimization and the ladder comparison
is void.

WHY on_memory_preamble IS THE ONLY FAIR CHOICE. BFCL's runtime is already a template-
injection engine with 18 keys. Of those, exactly one has injection level `system` and is
applied UNCONDITIONALLY, once per episode, before any state transition is observed
(memory_evaluator.py: the preamble is prepended to the system message of question[0]).
Every other key is conditional -- `trailing_user` keys fire after a tool result,
`turn_start_user` keys at a turn boundary. Conditional firing is the mechanism AnchorOpt
claims; lending it to the baseline would make any result unattributable.

This is the GEPA GLOBAL-PROMPT BASELINE. It tests global prompt optimization against
structured runtime optimization. It does NOT claim to exhaust GEPA's possible applications
to this benchmark.

H0'S PREAMBLE IS EMPTY. templates_initial.json ships `on_memory_preamble: ""`, so H0's
effective behaviour is "no preamble". We do NOT substitute the full H0 system message as
the seed: that text is already in the system message via BFCL's own prompt construction,
and seeding with it would duplicate every instruction the model already receives, which is
a different (and confounded) intervention. GEPA therefore optimizes from an empty seed, and
`assert_nonempty_candidate` makes a degenerate empty proposal a loud failure rather than a
silent no-op arm.
"""
from __future__ import annotations

import copy
from typing import Any

COMPONENT = "on_memory_preamble"


class IsolationError(AssertionError):
    """The arm is not what it claims to be. Never downgraded to a warning."""


def frozen_h0_templates(templates_initial: dict) -> dict:
    """Deep-copy H0's template dict and assert the preamble really is the empty seed."""
    t = copy.deepcopy(templates_initial)
    inner = t.get("templates")
    if inner is None:
        raise IsolationError(
            "templates dict has no 'templates' member; refusing to guess the schema")
    seed = inner.get(COMPONENT, None)
    if seed is None:
        raise IsolationError(
            f"H0 templates do not define {COMPONENT!r}. The editable surface must exist in "
            "H0 or the arm is not a one-key edit of H0.")
    if seed.strip():
        # Not fatal by itself, but it changes what the seed means, so it must be explicit.
        raise IsolationError(
            f"H0's {COMPONENT!r} is NOT empty (got {seed[:80]!r}). This plan was written for "
            "an empty seed; if H0 changed, revisit the seeding decision explicitly rather "
            "than silently optimizing from unexpected text.")
    return t


def seed_candidate() -> dict[str, str]:
    """GEPA's seed candidate: the empty H0 preamble. One component, always."""
    return {COMPONENT: ""}


def assert_candidate_shape(candidate: dict[str, Any]) -> str:
    """Return the candidate's preamble text, or raise if the candidate is off-contract."""
    if not isinstance(candidate, dict):
        raise IsolationError(f"candidate must be a dict, got {type(candidate).__name__}")
    if COMPONENT not in candidate:
        raise IsolationError(
            f"candidate must contain {COMPONENT!r}; got {sorted(candidate)}")
    extra = sorted(set(candidate) - {COMPONENT})
    if extra:
        raise IsolationError(
            f"GEPA proposed components outside the contract: {extra}. Only {COMPONENT!r} is "
            "editable; any other key would mean the arm edits conditional injection logic.")
    text = candidate[COMPONENT]
    if not isinstance(text, str):
        raise IsolationError(f"{COMPONENT} must be a str, got {type(text).__name__}")
    return text


def assert_nonempty_candidate(text: str, *, label: str) -> None:
    """A non-seed candidate must actually say something.

    GEPA seeds from "" here, and a reflector that returns an empty string would produce an
    arm byte-identical to the control while being labelled a GEPA candidate -- the exact
    'silently the control' failure that has voided arms in this project before.
    """
    if not text or not text.strip():
        raise IsolationError(
            f"candidate {label} has an EMPTY {COMPONENT}. An empty preamble is byte-identical "
            "to H0, so this arm would be mislabelled as a GEPA candidate. Refusing.")


def apply_candidate(h0_templates: dict, candidate: dict[str, Any]) -> dict:
    """Build the run-time template dict for one candidate: H0 with ONE key replaced.

    Returns a deep copy; the caller's H0 dict is never mutated (a mutated H0 would leak the
    previous candidate's text into the next evaluation, making every arm after the first a
    silent compound).
    """
    text = assert_candidate_shape(candidate)
    out = copy.deepcopy(h0_templates)
    out["templates"][COMPONENT] = text
    return out


def diff_against_h0(h0_templates: dict, applied: dict) -> list[str]:
    """Every difference between H0 and an applied candidate, as dotted paths.

    Used by the isolation test: the ONLY acceptable result is
    ['templates.on_memory_preamble'].
    """
    diffs: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k, _MISSING), b.get(k, _MISSING), f"{path}.{k}" if path else k)
        elif a != b:
            diffs.append(path)

    walk(h0_templates, applied, "")
    return diffs


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Missing)


_MISSING = _Missing()
