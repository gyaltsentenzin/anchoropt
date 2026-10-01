#!/usr/bin/env python3
"""TargetSpec — one declaration of what a candidate targets, consumed by every term.

WHY THIS EXISTS (§37)

The same defect appeared three times, in three different places, because the target's
SHAPE was re-assumed at every call site instead of declared once:

  §28  the error-contract attribution checker was pointed at G1's CALL target and
       reported "residual 0 -> 0" -- reading the wrong field, not measuring a policy.
  §33  Stage 0 was fixed for call targets by adding a second checker
       (check_attribution_call.py) with its own preventive/reactive logic.
  §36  R0 and the guardrail were STILL error-contract-only, so on the G1 rerun
       target_recovery() found 0 occurrences of `core_memory_clear` in tool_results
       (it lives in `decoded`), benefit computed as 0, and `harm < 0.5 x benefit`
       then failed automatically for any harm. Both arms printed "reject" on a
       measurement that never happened.

Fixing the third instance the way I fixed the second -- another special case -- would
guarantee a fourth. So the target is declared ONCE, here, and attribution / R0 / R1 /
guardrail all read the same object.

WHAT IS DECLARED

  shape          CALL (a generated tool call) vs ERROR_CONTRACT (a returned tool error)
  source_field   which trajectory field carries it: `decoded` vs `tool_results`
  decision_point pre_generation | pre_execution | post_execution  (§34)
  evidence_form  preventive | reactive | exposure_matched          (§33, §34.4)
  effect_horizon step | turn | episode                             (§34.7)

`source_field` is DERIVED from `shape`, not supplied separately: a call is what the model
emitted, an error contract is what the tool returned. Letting a caller set them
independently would reintroduce exactly the mismatch this module exists to prevent.

`evidence_form` defaults from `decision_point` (§33: the decision point fixes the FORM of
the evidence) but stays overridable, because §34.7 established that the decision point is
silent about the WINDOW -- so horizon is separate and explicit.

DISCIPLINE

Everything here is structural: field names, call names, error text, step and episode
positions. No reward label, no accuracy figure. A target spec that could read an outcome
would contaminate the very terms it feeds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ── shapes ───────────────────────────────────────────────────────────────────
CALL = "call"
ERROR_CONTRACT = "error_contract"
SHAPES = (CALL, ERROR_CONTRACT)

# ── decision points (§34) ────────────────────────────────────────────────────
PRE_GENERATION = "pre_generation"
PRE_EXECUTION = "pre_execution"
POST_EXECUTION = "post_execution"
DECISION_POINTS = (PRE_GENERATION, PRE_EXECUTION, POST_EXECUTION)

# ── evidence forms (§33, §34.4) ──────────────────────────────────────────────
PREVENTIVE = "preventive"
REACTIVE = "reactive"
EXPOSURE_MATCHED = "exposure_matched"
EVIDENCE_FORMS = (PREVENTIVE, REACTIVE, EXPOSURE_MATCHED)

# ── effect horizons (§34.7) ──────────────────────────────────────────────────
STEP = "step"
TURN = "turn"
EPISODE = "episode"
HORIZONS = (STEP, TURN, EPISODE)

# shape -> the ONLY field that can carry it. Derived, never caller-supplied.
_FIELD_BY_SHAPE = {
    CALL: "decoded",
    ERROR_CONTRACT: "tool_results",
}

# decision point -> default evidence form. The point fixes the FORM (§33); the horizon
# is separate because the point is silent about the window (§34.7).
_FORM_BY_POINT = {
    PRE_GENERATION: EXPOSURE_MATCHED,
    PRE_EXECUTION: PREVENTIVE,
    POST_EXECUTION: REACTIVE,
}

_CALL_RE = re.compile(r"""(?:^|[\[\("'`,;]|\s)\s*([a-z][a-z0-9_]*)\s*\(""")


@dataclass(frozen=True)
class TargetSpec:
    """What a candidate targets, and how every term must look for it."""

    signal: str                      # the call name, or the error substring
    shape: str                       # CALL | ERROR_CONTRACT
    decision_point: str              # §34
    evidence_form: Optional[str] = None      # defaults from decision_point
    effect_horizon: str = STEP               # §34.7 — declared, not inferred
    partition: Optional[str] = None          # e.g. "vector"; None = all
    telemetry_flag: str = ""                 # the gate's own flag
    # Extra fields a preventive gate writes when it removes/substitutes a call. Read by
    # attribution and R0 so a prevented call is still visible after being erased from
    # `decoded` (the §28 dropped-step failure).
    removal_fields: Tuple[str, ...] = ("suppress_removed_calls",
                                       "reroute_substituted_call")

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, got {self.shape!r}")
        if self.decision_point not in DECISION_POINTS:
            raise ValueError(f"decision_point must be one of {DECISION_POINTS}, "
                             f"got {self.decision_point!r}")
        if self.effect_horizon not in HORIZONS:
            raise ValueError(f"effect_horizon must be one of {HORIZONS}, "
                             f"got {self.effect_horizon!r}")
        form = self.evidence_form or _FORM_BY_POINT[self.decision_point]
        if form not in EVIDENCE_FORMS:
            raise ValueError(f"evidence_form must be one of {EVIDENCE_FORMS}, got {form!r}")
        object.__setattr__(self, "evidence_form", form)
        # A CALL target cannot be measured at post-execution, and an ERROR_CONTRACT
        # cannot be measured before the tool has run. Catching this here is the whole
        # point: it is the class of mismatch that produced §28, §33 and §36.
        if self.shape == CALL and self.decision_point == POST_EXECUTION:
            raise ValueError(
                "a CALL target at post_execution is a category error: the call has "
                "already run, so preventing it is undefined. Did you mean an "
                "ERROR_CONTRACT target?")
        if self.shape == ERROR_CONTRACT and self.decision_point == PRE_EXECUTION:
            raise ValueError(
                "an ERROR_CONTRACT target at pre_execution is a category error: no "
                "tool error exists yet. Did you mean a CALL target?")

    # ── the single source of truth every term reads ──────────────────────────

    @property
    def source_field(self) -> str:
        """The trajectory field carrying this target. DERIVED from shape."""
        return _FIELD_BY_SHAPE[self.shape]

    def occurs_in(self, step: dict) -> bool:
        """Does the target occur at this step? Shape-correct by construction.

        This one method replaces the per-call-site `signal in str(step["tool_results"])`
        that produced the three failures above.
        """
        if self.signal in str(step.get(self.source_field) or ""):
            return True
        # A preventive gate erases the call from `decoded`; the removal record is where
        # the evidence survives. Only meaningful for CALL targets.
        if self.shape == CALL:
            for f in self.removal_fields:
                if self.signal in str(step.get(f) or ""):
                    return True
        return False

    def removed_at(self, step: dict) -> bool:
        """Was the target REMOVED or SUBSTITUTED by a gate at this step?

        Requires a positive record. A preventive gate erases the call from `decoded`, so
        absence-from-decoded is NOT evidence of prevention -- it is indistinguishable
        from the model never emitting it. Only an explicit removal/substitution payload
        counts.
        """
        if self.shape != CALL:
            return False
        return any(self.signal in str(step.get(f) or "")
                   for f in self.removal_fields)

    def prevention_is_measurable(self, steps) -> bool:
        """Does this corpus record WHAT a preventive gate did?

        Measured on the frozen G1 rerun: the `suppress` arm writes
        `suppress_removed_calls` (161 records) and is measurable; the `reroute` arm
        writes only `reroute_gate` with NO substitution payload, so its R0 is UNKNOWN
        rather than zero. Reporting 0.0 there would repeat the §36 mistake -- a missing
        field read as a measured failure -- in a new place.

        Callers must branch on this and report INDETERMINATE, never a rate.
        """
        if self.shape != CALL:
            return True
        flag = self.telemetry_flag
        fired = any(s.get(flag) for s in steps) if flag else False
        recorded = any(any(s.get(f) for f in self.removal_fields) for s in steps)
        # A gate that never fired here is vacuously fine; one that fired without
        # recording what it did is the defect.
        return recorded or not fired

    def intended_at(self, step: dict) -> bool:
        """Did the model still INTEND the target at this step (survived into decoded)?"""
        return self.signal in str(step.get(self.source_field) or "")

    def applies_to(self, backend: Optional[str]) -> bool:
        return self.partition is None or backend == self.partition

    def as_dict(self) -> dict:
        return {
            "signal": self.signal, "shape": self.shape,
            "source_field": self.source_field,
            "decision_point": self.decision_point,
            "evidence_form": self.evidence_form,
            "effect_horizon": self.effect_horizon,
            "partition": self.partition,
            "telemetry_flag": self.telemetry_flag,
        }

    # ── constructors for the two shapes actually in use ──────────────────────

    @classmethod
    def for_call(cls, call_name: str, decision_point: str = PRE_EXECUTION, **kw):
        """G1's shape: a generated call to be prevented before it executes."""
        return cls(signal=call_name, shape=CALL,
                   decision_point=decision_point, **kw)

    @classmethod
    def for_error_contract(cls, error_substring: str,
                           decision_point: str = POST_EXECUTION, **kw):
        """G2's shape: a returned tool error to be reacted to."""
        return cls(signal=error_substring, shape=ERROR_CONTRACT,
                   decision_point=decision_point, **kw)


def calls_in(step: dict):
    """Call names emitted at this step. Shared so every term parses calls identically."""
    return [m.group(1) for m in _CALL_RE.finditer(str(step.get("decoded") or ""))]
