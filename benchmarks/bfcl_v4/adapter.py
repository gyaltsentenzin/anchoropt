#!/usr/bin/env python3
"""BFCL v4 Agent Memory: case-ID grammar and tool vocabulary, in ONE place.

WHY THIS FILE EXISTS
--------------------
The AnchorOpt core is already substantially benchmark-agnostic -- the policy search, the stopping
rules, the re-mine, the exposure and displacement diagnostics contain no BFCL vocabulary in code at
all. What was NOT localized is the small amount of benchmark knowledge the core genuinely needs:
how to read a case id, and how to recognise a read/write/clear call.

That knowledge had been reimplemented **five times** -- in `check_attribution`, `trace_backward`,
`benchmarks/bfcl_v4/run.py`, `scripts/derive_per_backend.py`, and twice in tests. Five copies of one
regex is a maintenance problem; five copies that DISAGREE is a correctness problem, and they did:

    backend_of("weird_id_no_backend")
        check_attribution -> None          (correct: unrecognised)
        run.py            -> "rec_sum"    (WRONG: silent misfiling)
        derive_per_backend-> "rec_sum"    (WRONG: silent misfiling)

Two of those returned `rec_sum` for anything they could not parse, because the implementation was
`"kv" if ... else "vector" if ... else "rec_sum"` -- a chain with no failure branch. An unknown id was
filed into a real backend rather than flagged. **Everything here raises on an unrecognised id.**

DELIBERATELY THIN
-----------------
This is a parsing-and-vocabulary adapter, not an event model. The regexes are the ORIGINALS, moved
rather than rewritten, and there is a parity test asserting the outputs are byte-identical to the
five implementations it replaces. No behaviour change, no redesign.

A richer common abstraction (a canonical `DecisionEvent`, say) may well be warranted once a second
benchmark is integrated -- but that decision should be made from two examples, not one. Until then,
the honest boundary is: **the core stays as it is; BFCL string-handling lives here.**

WHAT IS NOT HERE, ON PURPOSE
----------------------------
`StoreAdapter` in `anchoropt/mechanisms/constraint_repair.py` already owns the *repair-time* facts --
per-container caps, slot counts, error phrasings, the id keyword. That seam works and is tested; this
file does not duplicate it. The split is by lifecycle stage, which is the one that has held up:

    adapter.py      reading trajectories and case ids   (attribution / mining time)
    StoreAdapter    acting on a store                   (repair time)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

# ------------------------------------------------------------------------------------------------
# Case-ID grammar
# ------------------------------------------------------------------------------------------------
#
# The corpus encodes backend, phase, ordinal and scenario in the id itself:
#
#     memory_kv_30-healthcare-0            query,   kv,      healthcare, ordinal 30
#     memory_kv_prereq_11-healthcare-1     prereq,  kv,      healthcare, ordinal 11
#     memory_rec_sum_82-student-2          query,   rec_sum, student,    ordinal 82
#
# This regex is `trace_backward._CHAIN_RE` verbatim. `rec_sum` contains an underscore, which is why
# the backend alternation is explicit rather than `[a-z_]+`.
_CASE_RE = re.compile(r"^(memory_(?:kv|vector|rec_sum))_(?:prereq_)?(\d+)-([a-z_]+)")
_PREREQ_RE = re.compile(r"^memory_(?:kv|vector|rec_sum)_prereq_")

BACKENDS = ("kv", "vector", "rec_sum")


class UnknownCaseId(ValueError):
    """Raised when a case id does not match the corpus grammar.

    A distinct exception type on purpose: callers that legitimately tolerate foreign ids can catch
    exactly this, and everyone else gets a loud failure instead of a case filed under the wrong
    backend. The silent-`rec_sum` fallback this replaces was a real defect.
    """


class CaseId(NamedTuple):
    """A parsed case id. `backend` is bare (`kv`), `chain_key` is prefixed (`memory_kv-healthcare`)."""

    raw: str
    backend: str
    scenario: str
    ordinal: int
    is_prereq: bool

    @property
    def chain_key(self) -> str:
        """Chain = backend x scenario, the indivisible unit for splitting this corpus.

        Prefixed form (`memory_kv-healthcare`) to match what `trace_backward` already emitted.
        """
        return f"memory_{self.backend}-{self.scenario}"


def parse_case_id(case_id: str) -> CaseId:
    """Parse a case id, or raise `UnknownCaseId`.

    The single place this grammar is implemented. Everything else in this module delegates here.
    """
    match = _CASE_RE.match(str(case_id or ""))
    if not match:
        raise UnknownCaseId(
            f"{case_id!r} does not match the BFCL v4 case grammar "
            "(memory_<backend>_[prereq_]<n>-<scenario>-<n>). Not defaulting to a backend: an "
            "unrecognised id filed under a real backend is worse than a loud failure."
        )
    prefixed, ordinal, scenario = match.groups()
    return CaseId(
        raw=str(case_id),
        backend=prefixed[len("memory_"):],
        scenario=scenario,
        ordinal=int(ordinal),
        is_prereq=bool(_PREREQ_RE.match(str(case_id))),
    )


def backend_of(case_id: str) -> str:
    """`kv` / `vector` / `rec_sum`. Raises `UnknownCaseId` -- never guesses."""
    return parse_case_id(case_id).backend


def scenario_of(case_id: str) -> str:
    """The scenario segment: `healthcare`, `student`, `customer`, `finance`, `notetaker`."""
    return parse_case_id(case_id).scenario


def phase_of(case_id: str) -> str:
    """`prereq` or `query`.

    The distinction matters more than it looks: prerequisite episodes WRITE the store that query
    episodes read, which is why a split boundary may not fall inside a chain, and why several anchors
    fire almost entirely in the prereq phase while only query cases are scored.
    """
    return "prereq" if parse_case_id(case_id).is_prereq else "query"


def is_prereq(case_id: str) -> bool:
    return parse_case_id(case_id).is_prereq


def chain_of(case_id: str) -> tuple[str, int]:
    """`(chain_key, ordinal)`, sortable. Prereqs sort before queries within a chain.

    Replaces `trace_backward._chain_of`, including its behaviour on unparseable ids: that returned
    `("unknown", 0)` rather than raising, because it sorts trajectory files that may include
    non-corpus artifacts. Preserved deliberately -- see `chain_of_strict` for the raising form.
    """
    try:
        parsed = parse_case_id(case_id)
    except UnknownCaseId:
        return ("unknown", 0)
    return (parsed.chain_key, parsed.ordinal)


def chain_of_strict(case_id: str) -> tuple[str, int]:
    """`chain_of` that raises. Use where a foreign id is a bug rather than a stray file."""
    parsed = parse_case_id(case_id)
    return (parsed.chain_key, parsed.ordinal)


def cell_of(case_id: str) -> tuple[str, str]:
    """`(backend, scenario)` -- the indivisible unit of this corpus.

    15 such cells, and no case dependency crosses one. That is what licenses both the cell-level
    split and sharding by backend, and it is asserted at shard time rather than assumed.
    """
    parsed = parse_case_id(case_id)
    return (parsed.backend, parsed.scenario)


# ------------------------------------------------------------------------------------------------
# Tool vocabulary
# ------------------------------------------------------------------------------------------------
#
# Small predicates over a decoded call string. These are the patterns the core was matching inline;
# they are moved, not redesigned. Note they answer "what KIND of call is this", never "is this call
# wrong" -- keying a remedy on the verb rather than on the refusal proposed a suppression that
# inspection showed was benign in 15 of 17 cases.
_WRITE_RE = re.compile(
    r"\b(?:core|archival)_memory_(?:add|insert|update|replace)\s*\(|\bmemory_(?:append|update|replace)\s*\("
)
_READ_RE = re.compile(
    r"\b(?:core|archival)_memory_(?:retrieve|retrieve_all|list_keys|key_search|search)\s*\(|"
    r"\bmemory_(?:retrieve|get)\s*\("
)
_CLEAR_RE = re.compile(r"\b(core|archival)_memory_clear\s*\(")
_REMOVE_RE = re.compile(r"\b(?:core|archival)_memory_remove\s*\(")
_CONTAINER_RE = re.compile(r"\b(core|archival)_memory_")

# Substrings that mean "this result is an error", from the shipped backends' own wording. Kept as a
# tuple rather than a regex because that is how `trace_backward._ERRISH` had it.
ERRISH = ("error", "not found", "full", "cannot", "fail", "too long",
          "exceed", "invalid", "unable", "must be")


def is_write(call: str) -> bool:
    """Does this call attempt to put something into the store?"""
    return bool(_WRITE_RE.search(str(call or "")))


def is_read(call: str) -> bool:
    """Does this call attempt to get something out?"""
    return bool(_READ_RE.search(str(call or "")))


def is_clear(call: str) -> bool:
    """Is this a wholesale destructive clear? The action A8 intercepts before it runs."""
    return bool(_CLEAR_RE.search(str(call or "")))


def is_remove(call: str) -> bool:
    """A targeted single-entry removal -- NOT a clear.

    Worth separating: a remove forced by a capacity limit is legitimate recovery, while a clear
    destroys state gratuitously. Measured 17/17 removes followed a capacity error, so gating them
    breaks the repair that A1 depends on.
    """
    return bool(_REMOVE_RE.search(str(call or "")))


def is_destructive(call: str) -> bool:
    return is_clear(call) or is_remove(call)


def container_of(call: str) -> str | None:
    """`core` / `archival`, or None for a backend whose calls carry no container name.

    rec_sum is the None case: its state is a single blob and its verbs (`memory_append`) name no
    container. That returned None is a fact about the backend, not a parse failure -- and treating it
    as one is why that backend logged zero admission events for a whole line of work.
    """
    match = _CONTAINER_RE.search(str(call or ""))
    return match.group(1) if match else None


# The three predicates the trajectory scan needs, with the ORIGINAL matching logic preserved
# exactly -- substring tests, not new regexes, because the point is parity rather than tidiness.
_LIST_KEYS_TMPL = "%s_memory_list_keys"
_RETRIEVE_ALL_TMPL = "%s_memory_retrieve_all"
_TARGET_ID_RE = re.compile(r"vec_id\s*=\s*(\d+)")


def is_list_keys(call: str, container: str | None = None) -> bool:
    """Does this call enumerate the keys of a container?

    `container=None` means either. Substring matching, identical to what the scan did inline.
    """
    text = str(call or "")
    targets = (container,) if container else ("core", "archival")
    return any(_LIST_KEYS_TMPL % c in text for c in targets)


def is_read_all(call: str, container: str | None = None) -> bool:
    """Does this call read a container's full contents -- by listing keys OR retrieving all?

    The two are one question for attribution: either makes the store's contents VISIBLE in the
    trajectory, which is what "did this written key survive" is measured from.
    """
    text = str(call or "")
    targets = (container,) if container else ("core", "archival")
    return any((_LIST_KEYS_TMPL % c) in text or (_RETRIEVE_ALL_TMPL % c) in text for c in targets)


def target_id(call: str) -> str | None:
    """The entry handle a call addresses, or None.

    Named generically because the handle is backend-specific: vector addresses by numeric `vec_id`,
    kv by `key`. Only the numeric form is parsed here, which is exactly what the previous inline
    regex did -- widening it would be a behaviour change, not a refactor.
    """
    match = _TARGET_ID_RE.search(str(call or ""))
    return match.group(1) if match else None


#: Stores whose write takes a CALLER-SUPPLIED identifier, which is therefore capable of colliding.
#: Read off the tool signatures, not invented: a keyed add is `..._add(key=, value=)` while the
#: ordinal stores assign the handle themselves (`vec_id`) and the aggregate store takes only text.
#: This is the addressing fact `target_id` above already describes in prose -- recorded as data so a
#: structural observable can be derived from it rather than from a per-backend literal table.
_ADDRESSES_BY_CALLER_KEY = {"kv": True, "vector": False, "rec_sum": False}


def addresses_by_caller_key(backend: str) -> bool:
    """Whether `backend` addresses entries by an identifier the CALLER chooses.

    False for an unknown store: an unrecognised backend gets no structural claim made about it.
    """
    return bool(_ADDRESSES_BY_CALLER_KEY.get(str(backend or "").strip(), False))


def error_kind(result: str) -> str | None:
    """A coarse label for a failed result, or None if it does not look like an error.

    Deliberately coarse. The canonical semantic locus is `attribution/constraint_locus.py`'s job;
    this only answers "did the store refuse, and roughly why", which is what the trajectory scan
    needs before attribution runs.
    """
    text = str(result or "").lower()
    if not any(marker in text for marker in ERRISH):
        return None
    if "unique" in text:
        return "duplicate_identifier"
    if "not found" in text or "no such" in text:
        return "not_found"
    if "too long after appending" in text:
        return "blob_would_overflow"
    if "entry length" in text or "entry is too long" in text:
        return "entry_too_long"
    if "full" in text or "exceeds maximum size" in text:
        return "no_capacity"
    return "other"


def looks_like_error(result: str) -> bool:
    """The `_ERRISH` test, preserved verbatim in behaviour."""
    return error_kind(result) is not None


# ------------------------------------------------------------------------------------------------
# The adapter record
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BFCLAdapter:
    """The BFCL v4 vocabulary as one object, for callers that prefer injection to imports.

    The module-level functions are the primary interface -- they are what the core call sites use,
    and they keep the diff small. This record exists so a future second benchmark can be passed in
    rather than imported, without anyone having to restructure the core first.
    """

    name: str = "bfcl_v4_memory"
    backends: tuple[str, ...] = BACKENDS

    parse_case_id = staticmethod(parse_case_id)
    backend_of = staticmethod(backend_of)
    scenario_of = staticmethod(scenario_of)
    phase_of = staticmethod(phase_of)
    is_prereq = staticmethod(is_prereq)
    chain_of = staticmethod(chain_of)
    cell_of = staticmethod(cell_of)

    is_read = staticmethod(is_read)
    is_list_keys = staticmethod(is_list_keys)
    is_read_all = staticmethod(is_read_all)
    target_id = staticmethod(target_id)
    is_write = staticmethod(is_write)
    is_clear = staticmethod(is_clear)
    is_remove = staticmethod(is_remove)
    is_destructive = staticmethod(is_destructive)
    container_of = staticmethod(container_of)
    error_kind = staticmethod(error_kind)
    looks_like_error = staticmethod(looks_like_error)


ADAPTER = BFCLAdapter()


# ------------------------------------------------------------------------------------------------
# Registration -- this module supplies itself TO the core
# ------------------------------------------------------------------------------------------------
#
# Importing this module is the opt-in. The core never looks for it: it consumes whatever was
# registered, and raises if nothing was. That keeps the dependency pointing one way --
#
#     benchmarks/bfcl_v4/run.py  ->  this adapter  ->  register_adapter()  ->  core
#
# -- so the core carries no knowledge of this benchmark's name or file location.
#
# Registering the MODULE rather than `ADAPTER` is deliberate: `UnknownCaseId` is part of the
# interface (callers with a tolerant contract catch it), and a dataclass instance would not expose
# the exception type.
def register() -> None:
    """Install this adapter as the core's active benchmark vocabulary. Idempotent."""
    import sys as _sys

    from anchoropt.attribution import register_adapter
    register_adapter(_sys.modules[__name__])


try:
    register()
except ImportError:      # pragma: no cover - allows this module to be read without the core present
    pass
