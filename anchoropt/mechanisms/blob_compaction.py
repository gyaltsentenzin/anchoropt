#!/usr/bin/env python3
"""A7: the store is a SINGLE STRING at its length cap, and the next append is refused.

The recursive-summarization backend keeps memory as one blob, not a container of entries. When an
append would push it past the cap the tool rejects the write and the content is simply LOST. Neither
earlier capacity anchor applies: A1 and A6 both relocate a payload to a second container, and this
backend has no second container. There is nowhere to put anything. The only way to admit the pending
write is to make the blob itself shorter.

    at the first append-overflow: REWRITE the blob to free room, then RETRY the append verbatim.

This is the one anchor in the accepted set that calls a model at inference time. Every other one
computes its repair. That is a real cost -- an added LLM call reorders the trajectory, so coverage,
state keys and downstream questions all move, and a per-decision-point argument cannot bound run-level
behaviour. It is accepted because losing the write outright is worse, and because the measured delta
is the largest of any anchor on the backend it reaches.

THE ACCEPTANCE RULE IS THE WHOLE ANCHOR, AND THE SIMPLEST ONE WON
----------------------------------------------------------------
Four variants were measured against the same incumbent. Every added guard scored BELOW the plain
mechanical rule:

    v3  shorter-than-original AND fits         +7.34 pp   <- accepted
    v5  + rescue retry at a numeric target     +4.59 pp
    v6  fit only, no length rule at all        +3.67 pp
    v4  + reject rewrites under 60% of input   -7.34 pp

WHY "MUST BE SHORTER" DOES WORK NOBODY DESIGNED IT TO DO

The obvious reading of v3-beats-v6 is "shorter output loses more information". That reading is WRONG,
and one case pins it: on `35-healthcare-5` the v3 rewrite is 1,298 chars and the v6 rewrite is 3,094 --
v3 is 2.4x SMALLER and it is the one that keeps the needed fact. v6 spent its extra length on a nested
markdown outline of headers; v3 wrote flat prose.

    Constrained to be SHORTER than the original, the model REPHRASES and keeps content.
    Constrained only to FIT, it RESTRUCTURES -- and restructuring costs content at any length.

So the length rule is not a length rule. It keeps the model in rephrasing mode instead of reformatting
mode. This generalises past this backend and past this benchmark: when an LLM repairs a payload under a
size constraint, constrain it RELATIVE TO THE INPUT, not to an absolute target. A ratio also measures
how much TEXT survived, not how much INFORMATION did -- v3's median accepted ratio is 0.899 against
v6's 0.712 while v3 is the arm that retains more.

Two corollaries, both measured:
  * A floor on the ratio (v4) is actively harmful. Rejecting a collapsed rewrite removes the
    compaction without improving it, and 5 of 8 cases could only ever emit sub-floor output, so they
    stayed permanently blocked. Blocking a bad action is not the same as taking a good one.
  * Asking for a numeric target (v5) does not work on this model: it obeys the qualitative
    instruction reliably and overshoots every stated numeric ceiling by 400-800 chars, one direction.
    A pass/fail band test hides that -- report the SIGNED distance.

ONE DECISION PER STATE, NOT PER CALL
------------------------------------
This locus has ~18.7x append-retry amplification: the model reissues the refused append many times, so
per-call counting inflates support about 8x and a per-call gate would recompact on every retry. The
gate keys on the OVERFLOW STATE and fires once per distinct state, bounded per episode.

THE FACT PROXY BELOW GATES NOTHING, AND THAT IS DELIBERATE
----------------------------------------------------------
`specifics` extracts numbers, dates, acronyms and capitalised nouns. It is recorded for description and
it MUST NOT decide anything. On 12 losses an automated classifier built on this proxy reported "0
compaction-induced, 12 upstream write failures"; manual inspection -- locating the actual gold answer
string in the actual pre- and post-rewrite blobs -- gave 9 and 0, the exact inverse. The needed answers
were `walk`, `avocado`, `lawn`, `seven`: ordinary lowercase prose the proxy never emits, so its "lost
facts" came from unrelated tokens and every case fell into the classifier's first branch.

**A capitalisation-based fact detector cannot work on prose, and no lexical proxy may assign a causal
class.** Locate the required information in the real texts instead. This is the second time this
property disqualified a proxy on this project.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# The only length limit that is REAL rather than requested: the tool enforces it.
HARD_CAP = 10_000

# Bounded: a compaction loop must not run away, and each firing is an extra LLM call.
MAX_COMPACTIONS_PER_EPISODE = 3

APPEND_OVERFLOW_RE = re.compile(r"too long after appending", re.IGNORECASE)

# The accepted (v3) prompt. Qualitative, and relative to the input -- see the module docstring for
# why a numeric target measurably does not work on this model.
COMPACTION_PROMPT = (
    "Compact the following memory so that it is SHORTER than the original while preserving every "
    "fact needed to answer future questions: names, numbers, dates, preferences, commitments, "
    "constraints, and relationships. Rewrite verbose phrasing more concisely and remove repetition "
    "and conversational narration. Keep the same flat prose style -- do not reorganise the text into "
    "sections, headings, or outlines. Do not invent or infer anything not present. Return only the "
    "compacted memory text."
)

# Preambles models add unprompted. Stripped before measuring, or the length test scores the wrapper.
_PREAMBLE_RE = re.compile(
    r"^(here (is|are)[^\n:]*:|compacted memory:?|compressed memory:?|\*\*compact(ed)? memory:?\*\*:?)\s*",
    re.IGNORECASE,
)

# Structural markers whose appearance means the model REFORMATTED instead of rephrasing -- the
# failure mode that costs content at any length. Observed, and reported; see `restructured`.
_STRUCTURE_RE = re.compile(r"^\s*(#{1,6}\s|[-*+]\s+\*\*|\*\*[A-Z][^*]{2,40}:\*\*)", re.MULTILINE)

_DISCOURSE = frozenset(["Added", "Acknowledged", "Acknowledges", "Appreciates", "Asked", "Asks", "Assistant", "Concerned", "Confirmed", "Confirms", "Discussed", "Emphasized", "Emphasizes", "Explained", "Expressed", "Expresses", "Greeted", "Indicated", "Indicates", "Inquired", "Inquires", "Mentioned", "Mentions", "Needs", "Noted", "Notes", "Offered", "Offers", "Prefers", "Provided", "Provides", "Reported", "Reports", "Requested", "Requests", "Responded", "Responds", "Said", "Says", "Seeks", "Shared", "Shares", "States", "Stated", "Suggests", "Thanks", "User", "Wants", "Wanted", "Customer", "Client"])


def is_append_overflow(result: Any) -> bool:
    """Does this result mean the append was refused for blob length?"""
    return bool(APPEND_OVERFLOW_RE.search(str(result or "")))


def specifics(text: str) -> set[str]:
    """Capitalisation-based "specifics" in `text`. DESCRIPTIVE ONLY -- never gate on this.

    Emits numbers, currency, acronyms and capitalised non-discourse nouns. It cannot see lowercase
    prose facts (`walk`, `avocado`, `lawn`), which is why it inverted a 12-case causal attribution
    when someone let it decide. Kept because the retention figures it produces are useful for
    DESCRIBING a distribution after the fact.
    """
    body = str(text or "")
    found: set[str] = set()
    found |= set(re.findall(r"\b\d[\d,.:/-]*\b", body))
    found |= set(re.findall(r"\$\s?\d[\d,.]*", body))
    found |= {w.lower() for w in re.findall(r"\b[A-Z]{2,}\b", body)}
    found |= {w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", body) if w not in _DISCOURSE}
    return found


def strip_preamble(rewritten: str) -> str:
    return _PREAMBLE_RE.sub("", str(rewritten or "").strip()).strip()


def restructured(original: str, rewritten: str) -> bool:
    """Did the rewrite introduce outline structure the original did not have?

    The measured failure mode: reformatting costs content at any length. Reported as a diagnostic
    beside every acceptance decision, so the mode is visible in telemetry rather than inferred later
    from an accuracy delta.
    """
    return bool(_STRUCTURE_RE.search(rewritten or "")) and not bool(_STRUCTURE_RE.search(original or ""))


def validate(original: str, rewritten: str, pending: int = 0, cap: int = HARD_CAP) -> dict[str, Any]:
    """The v3 acceptance rule. TWO mechanical conditions, both computed, no semantic judgement.

        1. FITS      len(rewrite) + len(pending append) <= cap
        2. SHORTER   len(rewrite) < len(original)

    Condition 2 is the one that carries the anchor, for the reason in the module docstring: it keeps
    the model rephrasing rather than reformatting. It is NOT a ratio floor -- a floor (v4) was measured
    at -7.34 pp because rejection removes the compaction without improving it.

    Retention statistics are computed and returned for description. They gate NOTHING. The benchmark
    decides whether information was lost.
    """
    body = strip_preamble(rewritten)
    original_text = str(original or "")

    original_facts, new_facts = specifics(original_text), specifics(body)
    lost = original_facts - new_facts
    stats: dict[str, Any] = {
        "original_len": len(original_text),
        "new_len": len(body),
        "ratio": round(len(body) / len(original_text), 4) if original_text else None,
        "pending": pending,
        "room_after": cap - (len(body) + pending),
        "fits": (len(body) + pending) <= cap,
        "shorter": len(body) < len(original_text),
        "restructured": restructured(original_text, body),
        # DESCRIPTIVE ONLY. See `specifics`.
        "specifics_original": len(original_facts),
        "specifics_kept": len(original_facts & new_facts),
        "specifics_lost": len(lost),
        "retention": (round(len(original_facts & new_facts) / len(original_facts), 4)
                      if original_facts else None),
        "lost_sample": sorted(lost)[:12],
    }

    if not body:
        return {"accept": False, "reason": "empty rewrite", "text": body, **stats}
    if not stats["fits"]:
        return {
            "accept": False,
            "reason": f"does not fit ({len(body)} + {pending} > {cap})",
            "text": body, **stats,
        }
    if not stats["shorter"]:
        # A verbatim echo. The append still has nowhere to go, so this is a dead state -- but
        # accepting it would let the model reformat freely, which measured worse.
        return {
            "accept": False,
            "reason": f"not shorter than the original ({len(body)} >= {len(original_text)})",
            "text": body, **stats,
        }
    return {"accept": True, "reason": "ok", "text": body, **stats}


def overflow_state_key(blob: str, pending: str) -> str:
    """Identity of an overflow STATE, so the gate fires once per state and not once per retry.

    Keyed on the blob length plus the pending payload: the model reissues the same refused append
    many times (~18.7x amplification here), and a per-call gate would recompact on each one.
    """
    import hashlib

    digest = hashlib.sha256(f"{len(blob or '')}|{pending or ''}".encode()).hexdigest()
    return digest[:16]


def plan_compaction(
    result: Any,
    blob: str | None,
    pending: str,
    seen_states: Mapping[str, Any] | None = None,
    compactions_so_far: int = 0,
    cap: int = HARD_CAP,
) -> dict[str, Any]:
    """Decide whether to spend an LLM call compacting this blob. Pure decision logic, no dispatch.

    Returns `fire` plus a `state_key` the caller records, so the once-per-state rule is enforced by
    the caller's own bookkeeping rather than by hoping the gate is not re-entered.
    """
    if not is_append_overflow(result):
        return {"fire": False, "reason": "not an append-overflow rejection"}
    if not isinstance(blob, str):
        # `_live`-style container walks cannot see this backend: its state is a bare `str` on the
        # instance, not a dict. Reading it wrong is how this backend logged zero events for a while.
        return {"fire": False, "reason": "blob unreadable"}
    if compactions_so_far >= MAX_COMPACTIONS_PER_EPISODE:
        return {"fire": False, "reason": f"episode budget spent ({MAX_COMPACTIONS_PER_EPISODE})"}

    key = overflow_state_key(blob, pending)
    if seen_states is not None and key in seen_states:
        return {"fire": False, "reason": "this overflow state was already compacted", "state_key": key}

    return {
        "fire": True,
        "reason": f"blob {len(blob)} + pending {len(pending or '')} exceeds cap {cap}",
        "state_key": key,
        "prompt": COMPACTION_PROMPT,
        "blob": blob,
        "pending_len": len(pending or ""),
        # The retry is the ORIGINAL append, replayed verbatim after the rewrite lands.
        "retry_verbatim": True,
    }
