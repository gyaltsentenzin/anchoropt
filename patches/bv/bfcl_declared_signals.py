"""BFCL declared signals, for the cluster runner. VERBATIM COPIES, not reimplementations.

WHY THIS FILE EXISTS. A controller emitted from a Phi seeded on the residual family's grounded
observables names a DECLARED signal -- the host ships the predicate, and the spec says which one. The
cluster checkout has no `bfcl_signals`/`bfcl_runtime` (those live in the repo's benchmarks/ tree, which
the runner does not import), so such a spec had nothing to evaluate.

The alternative was hand-translating one controller into a field/op atom in the installer. That was
rejected, and rightly: the first attempt at it wrote `proposes_clear AND container_full` and silently
dropped the `error_kind == "no_capacity"` alternative the real signal carries. A translation that has
to be re-derived per controller is a place for exactly that drift, and it would also have made one
historical failure mechanism a special case in the installer.

So the function bodies below are COPIED FROM bfcl_signals.py CHARACTER FOR CHARACTER. The runner and
the optimizer evaluate the same code, and `test_offline_live_signal_equivalence.py` pins that they
agree on real gate states -- including the case this note is about, container_full False with
error_kind "no_capacity".

ONE SIGNAL IS DEGRADED, DELIBERATELY AND LOUDLY. `no_informative_result` delegates to
`anchoropt.learning.policy_tree.vacuous_result_kind`, which this checkout does not have. It raises
here rather than returning a plausible False: a signal that silently answers False is a controller
that silently never fires, which is the defect class this whole line has paid for most.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable


class SignalUnavailable(RuntimeError):
    """This checkout cannot evaluate that signal. Raised, never answered False."""


def container_at_capacity(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """A write was refused because its container has no room.

    Keyed on the REFUSAL, never on the verb: `docs/ANCHORS.md` records that keying a remedy on the
    call kind proposed a suppression that inspection showed benign in 15 of 17 cases.
    """
    return state.get("error_kind") == "no_capacity" and bool(state.get("proposes_write", False))


def identifier_not_found(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """A read did not find what it asked for. this trigger (read side only).

    the attribution for this locus split ONE error string into 39 read-side and 5 write-side cases. The read-side
    test is part of the signal, not a downstream filter: without it the signal names two different
    failures with one remedy.
    """
    return state.get("error_kind") == "not_found" and bool(state.get("proposes_read", False))


def duplicate_identifier(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """A write whose identifier is already present in the store. this trigger.

    TWO OBSERVATION MODES, and the pre-execution one is the anchor's:

      post_generation_pre_exec  a write is PROPOSED whose key is already known present (from the
                                store's live state) -- the duplicate has not happened yet
      post_execution            the store has already refused it (`error_kind`)

    the accepted form fires at the FIRST. docs/ANCHORS.md: this is "the first anchor at the commitment gate, which
    is the only point where cancelling a call is free: after execution the duplicate has already
    been rejected and the step is spent."

    The first version of this predicate read `error_kind` alone, which exists only after execution.
    It therefore could not fire where the accepted form actually fires, while still being DECLARED observable
    pre-execution -- so the search reached the cell and the predicate silently answered False
    there. That is the defect shape the boundary declaration guards, and the reason
    `SIGNAL_BOUNDARIES` and the predicate must
    agree.
    """
    if state.get("error_kind") == "duplicate_identifier":
        return True
    return bool(state.get("proposes_write", False)) and bool(state.get("identifier_present", False))


def no_tool_call_at_all(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """The model is about to answer without having called any tool.

    Observable ONLY after generation: whether a decision is a tool call or a final answer does not
    exist before that decision is made, so evaluating this earlier would fire on every episode.
    """
    if not bool(state.get("has_generation", False)):
        return False
    return not bool(state.get("proposes_tool_call", False)) and int(state.get("tool_calls_so_far", 0)) == 0


def container_slots_exhausted(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """The relocation TARGET is out of slots, so an earlier relocation has nowhere to put its payload.

    this trigger. Distinct from `container_at_capacity` by which container refused: the deeper variant exists
    because a successful relocation eventually saturates its target, one level deeper in the same
    residual.
    """
    if state.get("error_kind") != "no_capacity":
        return False
    return state.get("container") == "archival" or bool(state.get("is_relocation_target", False))


def append_would_exceed_cap(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """A single-blob store refused an append that would overflow its cap. this trigger.

    The backend keeps memory as ONE string with no second container, so there is nowhere to
    relocate to -- which is why the remedy is to shorten the store rather than to reroute.
    """
    return state.get("error_kind") == "blob_would_overflow"


def clear_proposed_at_capacity(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """The model PROPOSES a destructive clear while the container is full. the destructive-clear trigger.

    Observable strictly pre-execution: after the clear runs there is nothing left to save. This is
    the one signal whose entire value depends on the boundary.
    """
    if not bool(state.get("proposes_clear", False)):
        return False
    return bool(state.get("container_full", False)) or state.get("error_kind") == "no_capacity"


def retrieval_similarity_below_threshold(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """A retrieve SUCCEEDED and returned nothing well-matched.

    The structural-signal tier the lexical vocabulary cannot reach: no error is raised at all, so a
    read that fails silently looks like a successful read. `below` is REQUIRED: a threshold measured
    on one backend must not be silently exported to backends whose scores are not comparable, and a
    default here would do exactly that. The optimizer supplies it from TRAIN.
    """
    below = float(params["below"])
    score = state.get("best_similarity")
    if score is None:
        return False
    return bool(state.get("proposes_read", False)) and float(score) < below


def no_informative_result(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """NOT AVAILABLE in this checkout: it delegates to anchoropt.learning.policy_tree, which is not
    present here. Raises rather than returning False -- a signal that quietly answers False is a
    controller that quietly never fires."""
    raise SignalUnavailable(
        "no_informative_result needs anchoropt.learning.policy_tree.vacuous_result_kind, which this "
        "checkout does not provide; install it or do not emit a controller on this signal")


SIGNALS: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], bool]] = {
    "container_at_capacity": container_at_capacity,
    "identifier_not_found": identifier_not_found,
    "duplicate_identifier": duplicate_identifier,
    "no_tool_call_at_all": no_tool_call_at_all,
    "container_slots_exhausted": container_slots_exhausted,
    "append_would_exceed_cap": append_would_exceed_cap,
    "clear_proposed_at_capacity": clear_proposed_at_capacity,
    "retrieval_similarity_below_threshold": retrieval_similarity_below_threshold,
    "no_informative_result": no_informative_result,
}


def evaluate_signal(name: str, state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """Evaluate one DECLARED signal by name. The seam the installed controller calls.

    An unknown name raises: a controller naming a signal this host does not implement has not
    "not fired", it cannot be evaluated at all, and the two must not look the same.
    """
    fn = SIGNALS.get(str(name))
    if fn is None:
        raise SignalUnavailable(
            f"{name!r} is not a declared signal here; known: {sorted(SIGNALS)}")
    return bool(fn(dict(state or {}), dict(params or {})))


def declared_signals() -> tuple[str, ...]:
    return tuple(sorted(SIGNALS))
