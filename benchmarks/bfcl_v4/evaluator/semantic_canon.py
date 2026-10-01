"""Semantic canonicalization: mechanical first, teacher optional, AnchorOpt validates (§85).

The problem with regex: deciding whether a literal is STATE-DEFINING or NUISANCE was drifting into a
hand-maintained list of BFCL phrasings ("maximum size of", "less than", ...). Three iterations in,
that list is benchmark-specific rule accretion -- the thing the project forbids in the learning
layer.

The division of labour here mirrors §49's operator contract:

  MECHANICAL FIRST  a small, general structural parse runs first and is preferred whenever it is
                    confident. "A number adjacent to a constraint word" is general across tool
                    APIs; the specific phrasings are not.
  TEACHER FALLBACK  only when the mechanical parse is NOT confident. It returns a STRUCTURED
                    abstraction -- event_type, store, constraint_type, state literals, nuisance
                    literals, canonical signal -- and nothing else.
  ANCHOROPT VALIDATES  a proposed grouping is accepted only if it holds up empirically: support,
                    within-group recovery coherence, actionability. The teacher never chooses or
                    installs a gate and never judges an outcome.

Off by default. `--canon-teacher-model` enables the fallback; with it unset the module is pure
mechanical and byte-identical to a regex-only run for every case the mechanical parse handles.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Mechanical tier ────────────────────────────────────────────────────────────
# General structure, not BFCL phrasings: a number is STATE-DEFINING when it is the object of a
# constraint expression. The cue words below are ordinary English constraint vocabulary, not tool
# names or benchmark strings, and the same parse works for "at most 8 tabs" or "limit 2 retries".
CONSTRAINT_CUES = (
    "maximum", "minimum", "max", "min", "limit", "at most", "at least",
    "less than", "greater than", "no more than", "up to", "exceeds", "capacity",
)
# A number is NUISANCE when it sits in obvious content positions.
NUISANCE_CUES = ("score", "similarity", "id", "timestamp", "minutes", "hours", "seconds",
                 "date", "index", "rank")

_NUM = re.compile(r"\d+(?:\.\d+)?")


def _cue_hit(ctx: str, cues) -> bool:
    """Word-boundary cue match.

    Bare substring matching made 'min' fire inside 'minutes', classifying a duration as a
    constraint. Multi-word cues ("at most") are matched as phrases.
    """
    for c in cues:
        if " " in c:
            if c in ctx:
                return True
        elif re.search(r"\b%s\b" % re.escape(c), ctx):
            return True
    return False


@dataclass
class SemanticParse:
    """The structured abstraction. Produced mechanically or by the teacher."""
    event_type: str = ""          # e.g. "capacity_exceeded", "key_missing"
    store: str = ""               # e.g. "core", "archival"
    constraint_type: str = ""     # e.g. "entry_count", "entry_length"
    state_literals: List[str] = field(default_factory=list)
    nuisance_literals: List[str] = field(default_factory=list)
    canonical: str = ""
    source: str = "mechanical"    # "mechanical" | "teacher"
    confident: bool = True
    note: str = ""

    def as_dict(self):
        return {"event_type": self.event_type, "store": self.store,
                "constraint_type": self.constraint_type,
                "state_literals": self.state_literals,
                "nuisance_literals": self.nuisance_literals,
                "canonical": self.canonical, "source": self.source,
                "confident": self.confident, "note": self.note}


def _window(text: str, start: int, end: int, w: int = 28) -> str:
    return text[max(0, start - w):end + w].lower()


def mechanical_parse(text: str, failed_action: str = "", backend: str = "") -> SemanticParse:
    """Structural parse. Confident only when every literal is classifiable by position."""
    t = str(text)
    state, nuisance, unknown = [], [], []
    spans = []
    for m in _NUM.finditer(t):
        lit = m.group(0)
        ctx = _window(t, m.start(), m.end())
        # NUISANCE FIRST: its cues ('score', 'minutes', 'id') are more specific than generic
        # constraint words, so checking constraint first misclassified content values.
        if _cue_hit(ctx, NUISANCE_CUES):
            nuisance.append(lit)
            spans.append((m.start(), m.end(), None))
        elif _cue_hit(ctx, CONSTRAINT_CUES):
            state.append(lit)
            spans.append((m.start(), m.end(), lit))
        else:
            unknown.append(lit)
            spans.append((m.start(), m.end(), None))

    # rebuild: keep state literals verbatim, normalize the rest
    out, last = [], 0
    for s, e, keep in spans:
        out.append(re.sub(r"\d+", "N", t[last:s]))
        out.append(keep if keep is not None else "N")
        last = e
    out.append(re.sub(r"\d+", "N", t[last:]))
    canon = "".join(out).lower()
    canon = re.sub(r"[^a-z0-9 ]", " ", canon)
    canon = " ".join(canon.split())

    # store inferred from the failed action when available -- mechanical, not a guess
    store = ""
    fa = (failed_action or "").lower()
    if "archival" in fa:
        store = "archival"
    elif "core" in fa:
        store = "core"

    low = t.lower()
    if "not found" in low or "not present" in low:
        ev, ct = "lookup_miss", "key_presence"
    elif any(c in low for c in ("maximum size", "is full", "capacity")):
        ev, ct = "capacity_exceeded", "entry_count"
    elif any(c in low for c in ("maximum length", "too long")):
        ev, ct = "capacity_exceeded", "entry_length"
    elif "unique" in low:
        ev, ct = "constraint_violation", "key_uniqueness"
    else:
        ev, ct = "", ""

    return SemanticParse(
        event_type=ev, store=store, constraint_type=ct,
        state_literals=state, nuisance_literals=nuisance,
        canonical=canon, source="mechanical",
        # not confident if a literal could not be positioned, or the event is unrecognised
        confident=(not unknown) and bool(ev),
        note="" if (not unknown and ev) else
             "unclassified literals %s; event_type=%r" % (unknown, ev))


# ── Teacher tier (optional fallback) ──────────────────────────────────────────
SYSTEM = """You parse ONE tool-error string into a structured semantic abstraction. You do not \
choose policies, propose gates, or judge outcomes.

Return ONLY JSON with these keys:
  event_type        short snake_case category, e.g. capacity_exceeded, lookup_miss
  store             which store the event concerns, e.g. core, archival, or "" if not applicable
  constraint_type   what is constrained, e.g. entry_count, entry_length, key_presence, or ""
  state_literals    numbers that IDENTIFY the state (a capacity, a limit) -- changing one means a
                    DIFFERENT state with possibly different remedies
  nuisance_literals numbers that do NOT identify the state (timestamps, scores, content values)
  canonical         the error text with nuisance numbers replaced by N and state literals KEPT

A limit of 7 entries and a limit of 50 entries are DIFFERENT states. A similarity score of 0.83 and
a duration of 3 minutes are the SAME state."""

USER = """backend/store : {backend}
failed action : {failed_action}
tool schema   : {schema}
error text    : {text}

JSON only."""


def teacher_parse(text: str, failed_action: str = "", backend: str = "",
                  schema: str = "", model: Optional[str] = None,
                  endpoint: Optional[str] = None) -> Optional[SemanticParse]:
    """Fallback only. Returns None on any failure -- the caller keeps the mechanical parse."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from anchoropt.llm_operator import LLMOperator
        except ImportError:
            from llm_operator import LLMOperator
    except Exception:
        return None

    class _CanonOperator(LLMOperator):
        name = "semantic_canonicalizer"

        def gate(self, ctx):
            # FALLBACK ONLY: never runs when the mechanical parse was confident.
            if ctx.get("mechanical_confident"):
                return None
            return "mechanical parse not confident: %s" % ctx.get("mechanical_note", "")

        def build_prompt(self, ctx):
            return SYSTEM, USER.format(**ctx)

        def parse(self, raw):
            m = re.search(r"\{.*\}", str(raw or ""), re.S)
            if not m:
                return []
            try:
                return [json.loads(m.group(0))]
            except Exception:
                return []

        def validate(self, item, ctx):
            if not isinstance(item, dict):
                return {"status": "rejected", "reason": "not a JSON object"}
            need = ("event_type", "canonical")
            if not all(item.get(k) for k in need):
                return {"status": "rejected", "reason": "missing %s" % list(need)}
            # a state literal the teacher claims must actually occur in the text
            for lit in (item.get("state_literals") or []):
                if str(lit) not in str(ctx["text"]):
                    return {"status": "rejected",
                            "reason": "claimed state literal %r absent from the text" % lit}
            return {"status": "eligible", "parse": item}

    op = _CanonOperator(model=model, enabled=True, endpoint=endpoint)
    mech = mechanical_parse(text, failed_action, backend)
    res = op.run({"text": text, "failed_action": failed_action, "backend": backend,
                  "schema": schema or "(not supplied)",
                  "mechanical_confident": mech.confident,
                  "mechanical_note": mech.note})
    if not res.invoked:
        return None
    good = [c for c in res.candidates if c.get("status") == "eligible"]
    if not good:
        return None
    p = good[0]["parse"]
    return SemanticParse(
        event_type=p.get("event_type", ""), store=p.get("store", ""),
        constraint_type=p.get("constraint_type", ""),
        state_literals=[str(x) for x in (p.get("state_literals") or [])],
        nuisance_literals=[str(x) for x in (p.get("nuisance_literals") or [])],
        canonical=str(p.get("canonical", "")).lower(), source="teacher",
        confident=True, note="teacher fallback (%s)" % (model or "default"))


def canonicalize(text: str, failed_action: str = "", backend: str = "",
                 schema: str = "", teacher_model: Optional[str] = None,
                 teacher_endpoint: Optional[str] = None) -> SemanticParse:
    """MECHANICAL FIRST; teacher only when the mechanical parse is not confident AND enabled."""
    mech = mechanical_parse(text, failed_action, backend)
    if mech.confident or not teacher_model:
        return mech
    got = teacher_parse(text, failed_action, backend, schema,
                        model=teacher_model, endpoint=teacher_endpoint)
    return got or mech


def state_key(text: str, failed_action: str, backend: str, **kw) -> Tuple[str, str, str]:
    """The DECISION-STATE identity requested: contract + backend/store + failed_action."""
    p = canonicalize(text, failed_action, backend, **kw)
    store = p.store or backend
    return (p.canonical, store, failed_action or "?")
