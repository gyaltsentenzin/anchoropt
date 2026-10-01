"""AnchorOpt core-v2 adapter for AppWorld's `SimplifiedReActCodeAgent` loop.

WHAT THIS DESCRIBES, AND WHY THE BOUNDARIES ARE WHERE THEY ARE
----------------------------------------------------------------
One decision cycle of `Agent.solve_task`
(`<appworld-checkout>/experiments/code/simplified/agent.py`) is: the model is asked to generate a
cell, the cell (a Python code string) is handed to `AppWorld.execute`, and the result -- a plain
string, success or `"Execution failed. Traceback:\n..."` -- comes back. That gives three genuinely
distinct incision points, and each is declared here because AppWorld's own loop actually has a seam
there, not because the search would be more interesting if it did:

    PRE_GENERATION            before `generate()` is called; context/history, not the call
    POST_GENERATION_PRE_EXEC  the cell string exists; nothing has run
    POST_EXECUTION            the cell ran; its result or error string is visible

`PRE_GENERATION` is declared REPROMPT-only because that is structurally all AppWorld's loop supports
there (`anchor.py`'s `_STRUCTURAL_EXCLUSIONS` already rules out SUPPRESS/REROUTE at this locus --
there is no proposed call yet to cancel or replace). `POST_EXECUTION` omits SUPPRESS for the same
structural reason (the call already ran). Nothing here is trimmed for convenience; every declared
cell has a `BoundaryProbe` in `probes.py` proving the binding in `agent_hooks.py` actually runs.

THE RESIDUAL ASYMMETRY THIS ADAPTER EXISTS TO EXPRESS
-------------------------------------------------------
A live instance (`0d8a4ee_1`): the first cell is `apis.phone.get_contacts()`, and AppWorld reports
"No API named 'get_contacts' found in the phone app." at POST_EXECUTION. But whether `(phone,
get_contacts)` is a documented pair is knowable BEFORE that cell ever runs, from the same static
corpus (`data/api_docs/standard/*.json`) that seeds AppWorld's own api_docs app. So `api_path_unknown`
is declared at the GATE, `error_kind` only after execution -- the exact asymmetry that makes backward
localization necessary rather than decorative.

NO BENCHMARK VOCABULARY LEAKS PAST THIS FILE. Field and signal names are opaque strings to core;
`controller.py`/`agent_hooks.py` are the only other files that know what `api_path_unknown` means.
"""

from __future__ import annotations

import functools
import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.learning.executor_capability import ExecutorCapability
from anchoropt.runtime import HostProfile

from appworld.common.path_store import path_store

# ================================================================================================
# 1. BOUNDARY KEYS == LOCUS VALUES
# ================================================================================================
#
# Using the locus's own `.value` as this adapter's boundary key (rather than a parallel vocabulary
# like the toy host's `before_filing`/`after_filing`) keeps `boundary_key`/`boundary_from_key` and
# every `_Field.boundaries` declaration mechanically consistent -- there is no separate table that
# could drift from `IncisionPoint` itself.

_PRE = IncisionPoint.PRE_GENERATION.value
_PG = IncisionPoint.POST_GENERATION_PRE_EXEC.value
_PE = IncisionPoint.POST_EXECUTION.value

_KEY_TO_POINT = {p.value: p for p in IncisionPoint}

_LABELS = {
    _PRE: "before the model is asked to generate the next cell",
    _PG: "the cell exists as text; nothing has executed",
    _PE: "the cell ran; its result or error is visible",
}


class _Field:
    """One typed observable. Core reads `.name`, `.type`, `.enum`, `.boundaries` and the optional
    `.max_atoms` when synthesizing Phi (`anchoropt/learning/signal_grammar.py:atoms_for`).

    `max_atoms` raises core's per-field equality-atom cap for THIS field only. It exists because
    `MAX_ATOMS_PER_FIELD` defaults to 4, which silently truncates an enum to its first four declared
    values -- fine for an incidental category, fatal for a field whose values ARE the vocabulary the
    search has to discriminate on. Left unset, the protective default applies.
    """

    def __init__(self, name: str, type_: type, boundaries: tuple[str, ...], enum: tuple = (),
                 max_atoms: int | None = None):
        self.name, self.type, self.boundaries, self.enum = name, type_, boundaries, enum
        self.max_atoms = max_atoms
        self.doc = ""


# ================================================================================================
# 2. STATIC API-DOC CORPUS -- the offline, world-free source for gate-side observability
# ================================================================================================
#
# `data/api_docs/standard/<app>.json` is a dict keyed by api_name -> doc (app_name, api_name, path,
# method, description, parameters, response_schemas). It is the SAME corpus that seeds AppWorld's
# api_docs DB (`appworld/apps/api_docs/models.py`), read directly so this needs no live AppWorld
# instance. `path_store.data` is AppWorld's own path-resolution convention (respects `APPWORLD_ROOT`,
# else cwd) -- reused rather than inventing a parallel one.

@functools.lru_cache(maxsize=1)
def _api_docs_index() -> Mapping[str, Mapping[str, str]]:
    """app_name -> {api_name: HTTP method}. Empty if the corpus directory is not found."""
    root = os.path.join(path_store.data, "api_docs", "standard")
    index: dict[str, dict[str, str]] = {}
    if not os.path.isdir(root):
        return index
    for fname in sorted(os.listdir(root)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(root, fname), encoding="utf-8") as f:
            docs = json.load(f)
        app_name = fname[: -len(".json")]
        index[app_name] = {
            api_name: str((doc or {}).get("method") or "GET").upper()
            for api_name, doc in docs.items()
        }
    return index


@functools.lru_cache(maxsize=1)
def _paginated_apis() -> frozenset:
    """`(app_name, api_name)` pairs whose doc declares a `page_index` parameter — 70 of 473 APIs.

    A SECOND loader rather than an extension of `_api_docs_index`, whose contract (`app -> {api:
    method}`) is what `_lookup_call` consumes; widening that return type to carry parameters would
    touch the `api_path_unknown` / `proposes_write` path for a fact unrelated to it.

    This reads the same corpus AppWorld's own `api_docs` app is seeded from, so "is this API
    paginated" is the benchmark's own declaration, not a name heuristic. That matters: a heuristic on
    names like `show_*`/`search_*` would catch unpaginated APIs and miss paginated ones.
    """
    root = os.path.join(path_store.data, "api_docs", "standard")
    if not os.path.isdir(root):
        return frozenset()
    out: set[tuple[str, str]] = set()
    for fname in sorted(os.listdir(root)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(root, fname), encoding="utf-8") as f:
            docs = json.load(f)
        app_name = fname[: -len(".json")]
        for api_name, doc in (docs or {}).items():
            params = (doc or {}).get("parameters") or []
            if any((p or {}).get("name") == "page_index" for p in params if isinstance(p, dict)):
                out.add((app_name, api_name))
    return frozenset(out)


def known_apps() -> tuple[str, ...]:
    """The app names the static corpus documents -- an `.enum` for `target_app`."""
    return tuple(sorted(_api_docs_index()))


_CALL_RE = re.compile(r"apis\.(\w+)\.(\w+)\s*\(")


def _first_call(code: str) -> tuple[str, str]:
    """The first `apis.<app>.<api>(` occurrence in the cell text.

    A syntactic best-effort scan of the whole cell, not an AST walk: cells are short and almost
    always one call, and a cell proposing several calls is scored on its first one only. This is a
    real limitation, not a curation -- documented rather than hidden.
    """
    m = _CALL_RE.search(code or "")
    return (m.group(1), m.group(2)) if m else ("", "")


def _lookup_call(app: str, api: str) -> tuple[bool, bool]:
    """(api_path_unknown, proposes_write).

    A NAMED pair that is absent from the docs is `api_path_unknown=True, proposes_write=False` --
    absence of the pair is the only fact established; asserting it writes would be a guess this
    adapter has no basis for. `proposes_write` otherwise comes straight from the doc's own HTTP
    `method` (GET => read), which is AppWorld's own classification, not a name-heuristic invented here.

    A cell that names NO api call at all is `api_path_unknown=False`, NOT True. This distinction was
    measured, not reasoned about: the earlier version returned True here, which made
    `unknown_api_targeted` fire on any cell that merely did local computation -- a set
    intersection over already-fetched songs, a loop over an already-fetched password list. Three of
    four firings inspected on a minimax held-in arm were cells like that, and one of them destroyed
    valid work the model had just set up. "No API was targeted" and "an API was targeted and it does
    not exist" are different states, and only the second is what this signal names; reporting them as
    one made the signal fire ~2x as often as the real phenomenon occurs while telling the search
    nothing true about those extra cells.
    """
    if not app and not api:
        return False, False  # no api call in this cell -- nothing to be unknown ABOUT
    if not app or not api:
        return True, False  # a half-parsed call: something was named and could not be resolved
    apis = _api_docs_index().get(app)
    if apis is None or api not in apis:
        return True, False
    return False, apis[api] != "GET"


def gate_fields_from_code(code: str, step_number: int = 0) -> dict[str, Any]:
    """The POST_GENERATION_PRE_EXEC state a proposed cell yields. Shared by `agent_hooks.py` (live)
    and `residual.py` (offline, from a logged cell's `input`) so both compute it identically.

    `step_number` IS OBSERVABLE HERE, AND NOT DECLARING IT COST US THE ONE DISCRIMINATOR WE FOUND
    ----------------------------------------------------------------------------------------------
    It was previously declared at _PRE only, so no predicate over turn position existed at this
    boundary. Measured consequence on the minimax `pagination_unbounded` arm: gains and losses separate
    on WHERE the controller first fires -- gains at mean call 6.7 and 7.5 across two passes, losses at
    2.5 and 4.6. The arm fires EARLIER on the episodes it harms, which is the opposite of what total
    episode length suggested, so the refinement is "fire only after the episode has developed a few
    turns" and it could not be written at all. Nothing was wrong with the search: core enumerates every
    conjunction the adapter declares, and this term was never declared.

    Legitimate rather than leakage, by this adapter's own information-set rule: at this boundary the
    cell exists, so the turn that produced it is known. Contrast `error_kind`, deliberately absent here
    because the execution has not happened -- and contrast episode LENGTH, which correlates with the
    harm even more strongly (+3.9 LM calls on losses against +0.1 on unchanged episodes) and is
    unusable as a signal precisely because it is only knowable after the fact.

    Defaulted rather than required so a caller that has no turn index still gets a valid gate state;
    both real call sites pass it.
    """
    app, api = _first_call(code)
    path_unknown, proposes_write = _lookup_call(app, api)
    return {
        "step_number": int(step_number or 0),
        "code_chars": len(code or ""),
        "target_app": app,
        "target_api": api,
        "api_path_unknown": path_unknown,
        "proposes_write": proposes_write,
        # PAGINATION COMPLETENESS, statically knowable before the cell runs.
        #
        # `page_index` membership comes from the benchmark's own api-doc corpus, and the second field
        # is a substring test over the WHOLE cell, not just the first call -- a `while` loop that
        # advances `page_index` several lines below the call still counts as bounded, which is the
        # shape a correct solution takes.
        #
        # Why this pair exists: minimax's failures are not error-shaped (52% of its failing tasks have
        # no erroring cell at all), so no signal keyed on `error_kind` can reach them. They fail by
        # stopping early with the right shape of work -- median 10 steps against 13 for passing tasks.
        # Independently, an earlier internal harness's accepted minimax edits were completeness
        # policies, one of them verbatim "Paginated APIs: Always process all results, looping through
        # the page_index. Don't stop at the first page." Measured on minimax held-in, `targets_paginated_api and not
        # passes_page_index` fires on 15 of 29 failing tasks (52%) against 23 of 61 passing (38%).
        "targets_paginated_api": (app, api) in _paginated_apis(),
        "passes_page_index": "page_index" in (code or ""),
    }


# ================================================================================================
# 3. ERROR CLASSIFICATION -- AppWorld returns a plain string, never a structured error
# ================================================================================================
#
# Every failure path in `AppWorld.execute` (syntax error, disallowed import, timeout, max-interactions,
# or any runtime exception including an API dispatch error) shares one prefix, `"Execution failed.
# Traceback:\n"` (`appworld/environment.py:970-1098`, read directly to confirm this rather than
# assumed). Sub-kinds are classified by substring match against the traceback body, the same general
# approach `benchmarks/bfcl_v4/adapter.py:error_kind` uses for its own (differently-shaped) host.

_FAILED_PREFIX = "Execution failed."

# ERROR_KINDS is ORDERED BY MEASURED FREQUENCY, AND THAT ORDER IS LOAD-BEARING
# ------------------------------------------------------------------------------------------------
# Core truncates an enum to its first `max_atoms` values, so declaration order decides which classes
# the search can name at all. Frequency order means a truncated grammar still keeps the classes that
# actually occur, instead of whichever ones happened to be typed first.
#
# The distribution is measured, not guessed: 103 failing cells across the 53 episodes the
# post_execution/execution_failed arm fired on (minimax held-in, train).
#   auth_error           43 + 4 "Invalid credentials"   ~46%
#   bad_parameter        30                              29%
#   name_error           15                              15%
#   precondition_unmet    5                              ~5%
#   api_not_found         5                              ~5%
#   python_error          1
# Before this, EVERY one of those except api_not_found classified as `other_error` -- so `error_kind`
# was CONSTANT across all 5 gains and all 5 losses of that arm, and no predicate over it could
# separate them. The taxonomy was not wrong about AppWorld, it was written for error kinds that this
# benchmark barely produces.
#
# The classes are cut where the REMEDY differs, which is what a controller acts on: an expired token
# needs a re-login, not a corrected call; a wrong kwarg needs the call rewritten (the message even
# lists the allowed parameters); a NameError needs earlier state re-established; an unmet
# precondition means the request was already satisfied or is impossible, where retrying is wasted.
ERROR_KINDS: tuple[str, ...] = (
    "auth_error",
    "bad_parameter",
    "name_error",
    "precondition_unmet",
    "api_not_found",
    "app_not_found",
    "python_error",
    "syntax_error",
    "unsafe_syntax",
    "max_interactions",
    "timeout",
    "no_code",
    "other_error",
)

_PY_BUILTIN_ERRORS = (
    "KeyError", "TypeError", "IndexError", "AttributeError", "ValueError", "ZeroDivisionError",
)


def _classify_error_kind(output: str) -> str:
    text = output or ""
    if text.startswith("No code available to execute."):
        return "no_code"
    if not text.startswith(_FAILED_PREFIX):
        return ""
    # Structural host errors first: these are decided by the harness before any app logic runs, so a
    # later app-level match on the same text would be reading the wrong layer.
    if "No API named" in text:
        return "api_not_found"
    if "No app named" in text:
        return "app_not_found"
    if "is not allowed" in text and "module" in text:
        return "unsafe_syntax"
    if "Maximum number of executions" in text:
        return "max_interactions"
    if "Syntax error in line" in text:
        return "syntax_error"
    if "Execution timed out" in text:
        return "timeout"
    # Auth before the other app-level responses: an expired token surfaces as an ordinary API
    # response, so whichever check runs first wins, and auth is the one with a distinct remedy.
    if ("not authorized" in text
            or "access token is missing" in text
            or "Invalid credentials" in text):
        return "auth_error"
    if "Unexpected parameter" in text or "Missing required parameter" in text:
        return "bad_parameter"
    if "NameError" in text:
        return "name_error"
    if any(e in text for e in _PY_BUILTIN_ERRORS):
        return "python_error"
    # An app rejecting a well-formed call on its own terms -- already done, absent, or out of range.
    # Kept last of the app-level classes so a more specific match above always takes precedence.
    if ("already" in text
            or "is not available" in text
            or "must be one of" in text):
        return "precondition_unmet"
    return "other_error"


def exec_fields_from_output(output: str) -> dict[str, Any]:
    """The POST_EXECUTION state a cell's return string yields. Shared the same way as the gate
    fields above."""
    kind = _classify_error_kind(output)
    succeeded = kind == ""
    return {
        "error_kind": kind,
        "succeeded": succeeded,
        "result_chars": len(output or "") if succeeded else 0,
    }


# ================================================================================================
# 4. WHAT IS OBSERVABLE, AND WHERE -- boundary-truthful by construction
# ================================================================================================
#
# `target_app` gets a non-empty `.enum`: app identity is genuinely categorical (no natural ordering),
# so equality is the right comparison per `atoms_for`'s own branch, despite
# `MAX_ATOMS_PER_FIELD = 4` meaning only the first four apps in this tuple are ever tried as atoms --
# a real grammar limit, reported as an open item rather than a defect to work around here.
# `target_api` stays opaque `str` (empty `.enum`): too many values for equality atoms to be the
# discriminating comparison, so only presence (`truthy`/`falsy`) is offered, which is still an honest
# declaration of what it carries.
#
# `error_kind` USED TO BE OPAQUE ON THAT SAME REASONING, AND THAT WAS THE DEFECT
# ------------------------------------------------------------------------------------------------
# Declaring it opaque offers core only `truthy`/`falsy` on it -- and `truthy` is EXACTLY what the
# `execution_failed` signal already means, so the declaration made a refinement of that signal
# inexpressible rather than merely unfound. Measured cost: the post_execution arm fired on 53 of 90
# episodes and returned 5 gains against 5 losses, net 0, with nothing in the declared surface able to
# tell the two groups apart. Core was never at fault there; it cannot refine on a field whose values
# it was never allowed to name.
#
# `max_atoms` is set to the full taxonomy length because these values ARE the discriminating
# vocabulary; core's default cap of 4 would silently drop most of it.
#
# `last_error_kind` (at _PRE) deliberately stays opaque for now. It carries the same vocabulary one
# boundary earlier, so giving it an enum too is a real capability -- but it is speculative until a
# measurement shows the pre-generation boundary needs it, and it would multiply the atom grid. Left
# as a follow-on rather than bundled into a change whose effect we want to read cleanly.

_FIELDS_AT: Mapping[str, Mapping[str, _Field]] = {
    _PRE: {
        # turn index; a threshold ("stuck past step N") is the natural comparison -> int, no enum.
        # Declared at BOTH boundaries: it is equally observable once the cell exists, and the gate is
        # where it turned out to matter (see `gate_fields_from_code`).
        "step_number": _Field("step_number", int, (_PRE, _PG)),
        # a run of immediately-preceding failures; also threshold-shaped -> int, no enum
        "consecutive_errors": _Field("consecutive_errors", int, (_PRE,)),
        # the kind of the immediately preceding failure, opaque presence only
        "last_error_kind": _Field("last_error_kind", str, (_PRE,)),
    },
    _PG: {
        # `synthesis_fields()` returns _FIELDS_AT[boundary], so a field absent from THIS dict is absent
        # from the gate's alphabet no matter what its own `boundaries` tuple says.
        "step_number": _Field("step_number", int, (_PRE, _PG)),
        "code_chars": _Field("code_chars", int, (_PG,)),
        "target_app": _Field("target_app", str, (_PG,), enum=known_apps()),
        "target_api": _Field("target_api", str, (_PG,)),
        # statically knowable before anything executes -- see module docstring
        "api_path_unknown": _Field("api_path_unknown", bool, (_PG,)),
        "proposes_write": _Field("proposes_write", bool, (_PG,)),
        "targets_paginated_api": _Field("targets_paginated_api", bool, (_PG,)),
        "passes_page_index": _Field("passes_page_index", bool, (_PG,)),
    },
    _PE: {
        # only exists after execution -- NOT declared at _PG, which is the whole point
        "error_kind": _Field("error_kind", str, (_PE,), enum=ERROR_KINDS,
                             max_atoms=len(ERROR_KINDS)),
        "succeeded": _Field("succeeded", bool, (_PE,)),
        "result_chars": _Field("result_chars", int, (_PE,)),
    },
}

# THE SHIPPED VOCABULARY. Deliberately hand-written and coarse, per the toy adapter's precedent --
# not withheld to force Phi expansion, and not curated to make either boundary look more or less
# sufficient than it is. If one of these already separates most of the residual, that is a real
# result to accept, not a reason to hide the signal.
_DECLARED = ("execution_failed", "unknown_api_targeted", "generation_truncated",
             "pagination_unbounded")

# `generation_truncated` IS DECLARED AT THE GATE, NOT AFTER EXECUTION -- AND THAT IS THE FIX
# ------------------------------------------------------------------------------------------------
# `execution_failed` used to fire on `no_code`, which is not an execution failure at all. Measured on
# the qwen sh_heldout round: 22 of 73 firings (30%), and 197 of 373 failing cells. What actually
# happened in those episodes is that the LM call hit `max_completion_tokens` with reasoning enabled --
# `finish_reason: "length"`, `content: None`, in one case 33k characters of looping reasoning -- so NO
# CELL WAS EVER PRODUCED. AppWorld then reports "No code available to execute." and the old predicate
# read that as an error to correct. The injected instruction ("read the error above and issue a
# corrected call") referred to something that did not exist, and on 9 of those 22 the next cell was
# empty again.
#
# Two layers were being conflated, which is why no refinement over `error_kind` could separate gains
# from losses: the field spanned generation truncation, authentication state, parameter errors and
# unsatisfiable preconditions -- four disjoint remedies.
#
# The condition is genuinely observable at the GATE: `gate_fields_from_code` already yields
# `code_chars`, and a truncated turn produces no extractable code block, so `code_chars == 0` names it
# before anything executes. Declaring it at POST_GENERATION_PRE_EXEC is therefore boundary-truthful in
# the sense `docs/CORE_SEAM_states_at.md` requires, and it keeps the condition NAMEABLE -- it is a real
# and frequent failure that may deserve its own mechanism (a reprompt for a short, runnable cell is a
# different remedy from "read the error"), not something to discard.
_SIGNAL_BOUNDARIES: Mapping[str, frozenset] = {
    "execution_failed": frozenset({IncisionPoint.POST_EXECUTION}),
    "unknown_api_targeted": frozenset({IncisionPoint.POST_GENERATION_PRE_EXEC}),
    "generation_truncated": frozenset({IncisionPoint.POST_GENERATION_PRE_EXEC}),
    "pagination_unbounded": frozenset({IncisionPoint.POST_GENERATION_PRE_EXEC}),
}

_SIGNAL_FIELDS: Mapping[str, frozenset] = {
    "execution_failed": frozenset({"error_kind"}),
    "unknown_api_targeted": frozenset({"api_path_unknown"}),
    "generation_truncated": frozenset({"code_chars"}),
    "pagination_unbounded": frozenset({"targets_paginated_api", "passes_page_index"}),
}

# `no_code` stays in ERROR_KINDS: it is a truthful description of what POST_EXECUTION observed, and
# removing it would make `error_kind` silently empty for those cells, i.e. indistinguishable from a
# successful execution. It is excluded from `execution_failed` instead -- see `evaluate_signal`.
_NOT_AN_EXECUTION_ERROR = frozenset({"no_code"})


def _predicate_fields(predicate: Any) -> frozenset:
    """Field names an installed `Atom`/`Conjunction` reads, for `signal_fields` below. Duck-typed
    against `signal_grammar.Atom`/`Conjunction` rather than importing them, so an adapter-supplied
    non-grammar predicate (a bare lambda) degrades to an empty set instead of raising."""
    if hasattr(predicate, "field"):
        return frozenset({predicate.field})
    if hasattr(predicate, "terms"):
        return frozenset(t.field for t in predicate.terms if hasattr(t, "field"))
    return frozenset()


# ================================================================================================
# 5. THE ADAPTER
# ================================================================================================

class AppWorldHost:
    """The adapter. Every method is a contract hook -- see
    `anchoropt/testing/adapter_contract.py` for exactly what core does with each answer."""

    name = "appworld_react_code_agent"

    def __init__(self) -> None:
        self.expanded: dict[str, Any] = {}
        self.expanded_boundaries: dict[str, set] = {}
        self.expanded_fields: dict[str, frozenset] = {}
        self.HOST = HostProfile(
            name=self.name,
            executable={
                IncisionPoint.PRE_GENERATION: frozenset({Action.REPROMPT}),
                IncisionPoint.POST_GENERATION_PRE_EXEC: frozenset(
                    {Action.REPROMPT, Action.SUPPRESS, Action.REROUTE}),
                IncisionPoint.POST_EXECUTION: frozenset({Action.REPROMPT, Action.REROUTE}),
            })

    # ---------------------------------------------------------------------------- WHERE
    def is_decision(self, event: Mapping[str, Any]) -> bool:
        return bool(event.get("kind"))

    def boundary_key(self, event: Mapping[str, Any]) -> str:
        return str(event.get("kind") or "")

    def boundary_from_key(self, key: str):
        return _KEY_TO_POINT.get(str(key))

    def label_for(self, key: str) -> str:
        return _LABELS.get(str(key), "")

    # ---------------------------------------------------------------------------- WHAT
    def declared_signals(self) -> tuple[str, ...]:
        return _DECLARED + tuple(self.expanded)

    def signal_boundaries(self, signal: str) -> frozenset:
        if signal in self.expanded:
            return frozenset(self.expanded_boundaries.get(signal, ()))
        return _SIGNAL_BOUNDARIES.get(signal, frozenset())

    def synthesis_fields(self, boundary=None) -> dict:
        return dict(_FIELDS_AT.get(str(getattr(boundary, "value", boundary)), {}))

    def signal_fields(self, signal: str) -> frozenset:
        """UNDOCUMENTED core hook, read at `structured_search.py:767` to tell a signal whose fields
        this projection cannot supply apart from one that is genuinely inert. Both `MANDATORY_HOOKS`
        and `OPTIONAL_HOOKS` in `adapter_contract.py` omit it -- flagged as an open item to report."""
        if signal in self.expanded:
            return self.expanded_fields.get(signal, frozenset())
        return _SIGNAL_FIELDS.get(signal, frozenset())

    def install_signal(self, name: str, predicate, *, boundary=None, provenance: str = "") -> None:
        self.expanded[name] = predicate
        pt = boundary if isinstance(boundary, IncisionPoint) else _KEY_TO_POINT.get(str(boundary))
        self.expanded_boundaries[name] = {pt} if pt is not None else set()
        self.expanded_fields[name] = _predicate_fields(predicate)

    def reset_expanded_signals(self) -> None:
        self.expanded.clear()
        self.expanded_boundaries.clear()
        self.expanded_fields.clear()

    def expanded_signal_names(self) -> tuple[str, ...]:
        return tuple(self.expanded)

    def evaluate_signal(self, signal: str, state: Mapping[str, Any], params=None) -> bool:
        if signal in self.expanded:
            return bool(self.expanded[signal](state))
        if signal == "execution_failed":
            # Non-empty AND an actual execution failure. `no_code` is excluded because the cell never
            # ran -- the generation was truncated before producing one. See _NOT_AN_EXECUTION_ERROR.
            kind = str(state.get("error_kind") or "")
            return kind != "" and kind not in _NOT_AN_EXECUTION_ERROR
        if signal == "unknown_api_targeted":
            return bool(state.get("api_path_unknown"))
        if signal == "pagination_unbounded":
            # Both fields must be PRESENT: at a non-gate boundary they are absent, and `absent` must
            # not read as False-and-therefore-unbounded, which would fire on every such record.
            if "targets_paginated_api" not in state or "passes_page_index" not in state:
                return False
            return bool(state.get("targets_paginated_api")) and not bool(state.get("passes_page_index"))
        if signal == "generation_truncated":
            # A gate-side condition: the turn yielded no runnable cell. `code_chars` is absent from a
            # state that is not a gate state, and `absent` must not read as `0` -- that would make this
            # fire on every post-execution record. So require the field to be present.
            if "code_chars" not in state:
                return False
            return int(state.get("code_chars") or 0) == 0
        raise KeyError(f"{self.name} cannot evaluate signal {signal!r}")

    def probe_params(self, signal: str) -> dict:
        return {}

    def parameter_domains(self, signal: str) -> tuple:
        """No tunable theta_phi in round 1: every signal here is deterministic once installed."""
        return ()

    def states_at(self, boundary, states: Sequence[Mapping[str, Any]]) -> list:
        """Project states onto ONE boundary's information set. Fails closed: a state whose own
        `boundary` field does not match is dropped, never invented from another boundary's fields."""
        want = set(self.synthesis_fields(boundary))
        key = str(getattr(boundary, "value", boundary))
        out = []
        for st in states or ():
            if str(st.get("boundary") or "") == key:
                out.append({k: v for k, v in st.items()
                           if k in want or k in ("boundary", "case_id", "task_id")})
        return out

    # ---------------------------------------------------------------------------- HOW: eta grounding
    #
    # Every `ground_*` branches on `boundary` (structure), never on `signal` (identity) -- a
    # signal-name whitelist is the shipped bug that once produced zero groundable arms from four real
    # observables, indistinguishable from an absent primitive.

    def ground_reprompt(self, signal: str, boundary) -> list:
        b = str(getattr(boundary, "value", boundary))
        if b == _PRE:
            return [{
                "variant": "review_before_generating",
                "eta": {"instruction": "The previous cell failed. Review the error before "
                                       "proposing the next one.", "retry_budget": 1},
                "detail": "nudge the model to address the last error before it generates again",
            }]
        if b == _PG:
            # FOUR variants, and every one is offered to EVERY gate signal on purpose.
            #
            # `check_api_before_running` was written for `unknown_api_targeted` and reads as nonsense
            # for a turn that produced no code at all, which is what `generation_truncated` names. The
            # tempting fix -- branch on `signal` and hand each one its own text -- is the signal-name
            # whitelist this class of method must not become (see the note above `ground_reprompt`):
            # the shipped bug there produced ZERO groundable arms from four real observables, which is
            # indistinguishable from the primitive being absent.
            #
            # Adding a variant keeps the structure boundary-keyed: every gate signal is offered both, no
            # signal can yield an empty list, and WHICH text suits WHICH condition is settled by
            # measurement rather than by a hardcoded pairing. The cost is a slightly larger candidate
            # set; `headroom.py` is the cheap way to decide which of them is worth scoring.
            return [{
                "variant": "check_api_before_running",
                "eta": {"instruction": "This call does not match a documented app/api pair. Call "
                                       "apis.api_docs.show_api_descriptions(app_name=...) to check "
                                       "the correct name, then propose the corrected call.",
                        "retry_budget": 1},
                "detail": "regenerate the cell before it executes",
            }, {
                "variant": "page_through_all_results",
                "eta": {"instruction": "This API is paginated. Loop over page_index from 0, "
                                       "collecting every page until one comes back empty, before "
                                       "using the results. Do not stop at the first page.",
                        "retry_budget": 1},
                "detail": "regenerate the cell to page through all results; the text mirrors the "
                          "completeness policy an earlier internal harness's accepted minimax edit used",
            }, {
                "variant": "emit_short_runnable_cell",
                "eta": {"instruction": "Your last turn produced no runnable code. Emit a single short "
                                       "Python code block that makes one API call. Keep any reasoning "
                                       "brief.",
                        "retry_budget": 1},
                "detail": "regenerate the cell when the turn yielded nothing executable; 'short' is "
                          "load-bearing, since the observed cause is reasoning running to the token "
                          "cap and leaving no code block behind",
            }, {
                # A BOUNDED counterpart to `page_through_all_results`, added because the unbounded text
                # is the likeliest cause of that arm's own losses.
                #
                # Measured: on the episodes the unbounded variant loses, it inflates LM calls by +3.9
                # against +0.1 on the episodes it does not flip, and its two sharpest losses nearly
                # double (18 -> 34 calls, 27 -> 34). Its GAINS meanwhile replicate task-for-task across
                # passes (5 of 6), so the gain does not obviously require exhausting every page -- which
                # is exactly the hypothesis a bounded text tests and the unbounded one cannot.
                #
                # Deliberately no numeric page cap in the text: a cap would be a theta to tune, and this
                # signal declares no parameter domain (`parameter_domains` returns ()). "A few pages,
                # then use what you have" is the policy; picking N is a separate question.
                "variant": "page_first_pages_then_proceed",
                "eta": {"instruction": "This API is paginated and you have only the first page. Fetch a "
                                       "few more pages, then continue with what you have. Do not keep "
                                       "paging until the results run out.",
                        "retry_budget": 1},
                "detail": "regenerate the cell to page a bounded number of times; the bounded/unbounded "
                          "pair is what separates 'page through' from 'page exhaustively' as policies",
            }]
        if b == _PE:
            return [{
                "variant": "fix_after_visible_error",
                "eta": {"instruction": "The last cell failed; read the error above and issue a "
                                       "corrected call.", "retry_budget": 1},
                "detail": "guide the next generation after a visible failure",
            }]
        return []

    def ground_suppress(self, signal: str, boundary) -> list:
        if str(getattr(boundary, "value", boundary)) != _PG:
            return []
        return [{
            "variant": "cancel_proposed_call",
            "eta": {"suppressed_operation": "the proposed cell"},
            "detail": "cancel the proposed cell before it reaches world.execute",
        }]

    def ground_substitute_destinations(self, signal: str, boundary) -> list:
        b = str(getattr(boundary, "value", boundary))
        if b == _PG:
            return [{
                "variant": "reroute_to_api_docs",
                "eta": {},
                "detail": "replace an unresolvable cell with an api_docs lookup for the intended "
                          "app, before anything executes",
            }]
        if b == _PE:
            return [{
                "variant": "reroute_to_api_docs",
                "eta": {},
                "detail": "after an api/app-not-found failure, dispatch an api_docs lookup and "
                          "surface ITS result in place of the bare traceback",
            }]
        return []

    def ground_transforms(self, signal: str, boundary) -> list:
        return []

    # ---------------------------------------------------------------------------- executors
    #
    # `binding` names methods on `agent_hooks.AppWorldReActAgent`, the executing code.
    # `probes.py` proves each one behaviorally, per `tests/test_executor_behavioral_contract.py`.
    _CAPS: Mapping[tuple[str, str], ExecutorCapability] = {
        (_PRE, "reprompt"): ExecutorCapability(
            boundary=_PRE, action="reprompt",
            binding="agent_hooks.AppWorldReActAgent._inject_pre_generation_instruction",
            consumes=("instruction", "retry_budget"), signals=(), signal_agnostic=True,
            detail="appends the instruction to the message list before the model is called"),
        (_PG, "reprompt"): ExecutorCapability(
            boundary=_PG, action="reprompt",
            binding="agent_hooks.AppWorldReActAgent._inject_pre_exec_instruction",
            consumes=("instruction", "retry_budget"), signals=(), signal_agnostic=True,
            detail="appends the instruction and regenerates the cell before it executes"),
        (_PG, "suppress"): ExecutorCapability(
            boundary=_PG, action="suppress",
            binding="agent_hooks.AppWorldReActAgent._withhold_proposed_call",
            consumes=("retry_budget",), signals=(), signal_agnostic=True, eta_is_computed=True,
            detail="withholds the cell from world.execute; the agent observes a neutral marker"),
        (_PG, "reroute"): ExecutorCapability(
            boundary=_PG, action="reroute",
            binding="agent_hooks.AppWorldReActAgent._reroute_to_api_docs_pre_exec",
            consumes=(), signals=(), signal_agnostic=True, eta_is_computed=True,
            detail="dispatches an api_docs lookup instead of the proposed cell"),
        (_PE, "reprompt"): ExecutorCapability(
            boundary=_PE, action="reprompt",
            binding="agent_hooks.AppWorldReActAgent._inject_post_execution_instruction",
            consumes=("instruction", "retry_budget"), signals=(), signal_agnostic=True,
            detail="appends the instruction before the NEXT generation, after a visible failure"),
        (_PE, "reroute"): ExecutorCapability(
            boundary=_PE, action="reroute",
            binding="agent_hooks.AppWorldReActAgent._reroute_after_execution",
            consumes=(), signals=(), signal_agnostic=True, eta_is_computed=True,
            detail="dispatches a corrective api_docs lookup and surfaces its result instead"),
    }

    def executor_capability(self, boundary, action):
        """Singular form, for `anchoropt.testing.check_adapter_contract` (which does not yet know
        about the plural form below -- an open item to report, not to patch)."""
        return self._CAPS.get((str(getattr(boundary, "value", boundary)),
                               str(getattr(action, "value", action))))

    def executor_capabilities(self, boundary, action):
        """Plural form. `_materializable` (`anchor_policy_opt.py:274-321`) tries this FIRST; every
        cell here has exactly one mechanism, so it is a single-element list, but declaring the
        plural form keeps this adapter on the checked, non-masking dispatch path."""
        cap = self.executor_capability(boundary, action)
        return [cap] if cap is not None else []

    def capability_audit(self) -> dict:
        from anchoropt.learning.executor_capability import (
            disabled_capabilities, unbound_capabilities,
        )
        caps = tuple(self._CAPS.values())
        return {"unbound": list(unbound_capabilities(caps)),
                "disabled": list(disabled_capabilities(caps)),
                "bound_and_enabled": sorted(f"{c.boundary}/{c.action}" for c in caps
                                            if c.is_bound and c.is_enabled)}


ADAPTER = AppWorldHost()
