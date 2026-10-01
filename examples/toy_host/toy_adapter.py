"""A COMPLETE, RUNNABLE AnchorOpt adapter for a toy host. ~200 lines, no GPU, no benchmark.

WHAT THIS IS FOR
----------------
Two audiences, one file:

  * A COLLABORATOR porting TauBench or AppWorld reads this as the reference implementation. Every
    hook the contract requires appears here, with its real signature and a comment saying what core
    does with the answer. Copy it, rename the vocabulary, keep the shape.
  * The CORE RELEASE uses it as the deterministic proof that the algorithm runs end to end. There is
    no sampling, no model and no network here, so `demo.py` produces byte-identical output on every
    run -- which is what makes it usable as an acceptance test rather than a demonstration.

THE TOY DOMAIN, chosen to exercise the algorithm rather than to be realistic
---------------------------------------------------------------------------
An agent files reports into a fixed-size cabinet.

    file_report(text=...)   fails when `text` is longer than the drawer accepts
    archive_report(text=..) always succeeds; no length limit
    read_report(id=...)     a lookup that may miss

The residual: the agent files long reports into the drawer, they are rejected, and the information is
lost. The repair a search SHOULD find is to stop the over-long filing before it commits -- suppression
at the commitment gate, conditioned on the proposed payload size.

Nothing about that answer is declared anywhere. `demo.py` mines the residual, localizes the boundary
backward, discovers it cannot express "the payload is too long" under the shipped vocabulary, expands
the representation at that boundary, grounds the action, MEASURES the candidates, and accepts one.

VOCABULARY NOTE, and it is the portability claim: `IncisionPoint` and `Action` are CORE vocabulary --
they describe any tool-using LLM call, not any benchmark. `drawer`, `file_report`, `payload_chars` are
THIS host's. Core never sees the latter except as opaque strings it hands back to this adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.learning.executor_capability import ExecutorCapability
from anchoropt.runtime import HostProfile

# ================================================================================================
# 1. THE HOST'S OWN BOUNDARY KEYS, and how they map to core's loci
# ================================================================================================
#
# An adapter names its own decision points. `boundary_from_key` is the ONLY place the mapping lives;
# core owns no key->locus table, because that would be a semantic stage map imposed from outside.

GATE = "before_filing"          # a call has been proposed, nothing has run yet
AFTER = "after_filing"          # the call ran; its result (or error) is visible

_KEY_TO_POINT = {
    GATE: IncisionPoint.POST_GENERATION_PRE_EXEC,
    AFTER: IncisionPoint.POST_EXECUTION,
}

DRAWER_LIMIT = 80               # the host's real constraint; the learner is never told this number


class _Field:
    """One typed observable. Core reads `.name`, `.type` and `.boundaries` when synthesizing Phi."""

    def __init__(self, name: str, type_: type, boundaries: tuple[str, ...]):
        self.name, self.type, self.boundaries = name, type_, boundaries
        self.doc, self.enum = "", ()


# ================================================================================================
# 2. WHAT IS OBSERVABLE, AND WHERE. The single most important declaration in an adapter.
# ================================================================================================
#
# `synthesis_fields(boundary)` is the alphabet a new signal may be built from AT THAT BOUNDARY, and it
# must be boundary-truthful: a field that only exists after execution must not appear at the gate.
# Getting this wrong is the defect class that produces controllers which can never fire -- they are
# built on a field their boundary does not carry, and they silently answer False forever.
#
# Note `payload_chars` IS available at the gate (the proposed call is visible) while `error_kind` is
# NOT (nothing has run). That asymmetry is the whole reason the search must localize backward.

# A field's own `boundaries` are LOCUS VALUES, matching the keys of this dict. They may equally be an
# adapter's own boundary keys -- `anchoropt.testing.check_adapter_contract` reports a mismatch as a NOTE
# rather than a failure, because it cannot tell your key from a wrong locus. Using locus values here
# keeps the two declarations verifiably consistent.
_PG = IncisionPoint.POST_GENERATION_PRE_EXEC.value
_PE = IncisionPoint.POST_EXECUTION.value

_FIELDS_AT: Mapping[str, Mapping[str, _Field]] = {
    _PG: {
        "payload_chars": _Field("payload_chars", int, (_PG,)),
        "target": _Field("target", str, (_PG,)),
        "proposes_write": _Field("proposes_write", bool, (_PG,)),
    },
    _PE: {
        "error_kind": _Field("error_kind", str, (_PE,)),
        "result_chars": _Field("result_chars", int, (_PE,)),
        "succeeded": _Field("succeeded", bool, (_PE,)),
    },
}

# The SHIPPED vocabulary. Deliberately too coarse to express "the payload is too long" -- if it could,
# the demo would never exercise signal expansion, which is half the algorithm.
_DECLARED = ("filing_failed", "nothing_proposed")

_SIGNAL_BOUNDARIES: Mapping[str, frozenset] = {
    # observable only AFTER the call ran
    "filing_failed": frozenset({IncisionPoint.POST_EXECUTION}),
    # observable at the gate
    "nothing_proposed": frozenset({IncisionPoint.POST_GENERATION_PRE_EXEC}),
}


class ToyReportHost:
    """The adapter. Every method is a contract hook; nothing here is optional decoration."""

    name = "toy_report_host"

    def __init__(self) -> None:
        # Expanded signals live on the ADAPTER, not in core: core hands back a predicate and a name,
        # and the adapter owns evaluation. `reset_expanded_signals` exists so a test can re-run the
        # search from a clean vocabulary.
        self.expanded: dict[str, Any] = {}
        self.expanded_boundaries: dict[str, set] = {}
        self.HOST = HostProfile(
            name="toy_report_host",
            executable={
                # SUPPRESS is available at the gate (a proposed call can still be cancelled) and
                # structurally impossible after execution -- core's `_STRUCTURAL_EXCLUSIONS` already
                # knows the second half, so declaring it here would be redundant, not additive.
                IncisionPoint.POST_GENERATION_PRE_EXEC: frozenset({Action.SUPPRESS, Action.REPROMPT}),
                IncisionPoint.POST_EXECUTION: frozenset({Action.REPROMPT, Action.REROUTE}),
            })

    # ---------------------------------------------------------------------------- WHERE
    def is_decision(self, event: Mapping[str, Any]) -> bool:
        """Which trajectory events are CONSEQUENTIAL decisions. Core localizes over these only."""
        return bool(event.get("kind"))

    def boundary_key(self, event: Mapping[str, Any]) -> str:
        return {"propose": GATE, "execute": AFTER}.get(str(event.get("kind")), "")

    def boundary_from_key(self, key: str):
        return _KEY_TO_POINT.get(str(key))

    def label_for(self, key: str) -> str:
        return {GATE: "before the report is filed", AFTER: "after the filing returned"}.get(key, "")

    # ---------------------------------------------------------------------------- WHAT
    def declared_signals(self) -> tuple[str, ...]:
        return _DECLARED + tuple(self.expanded)

    def signal_boundaries(self, signal: str) -> frozenset:
        if signal in self.expanded:
            return frozenset(self.expanded_boundaries.get(signal, ()))
        return _SIGNAL_BOUNDARIES.get(signal, frozenset())

    def synthesis_fields(self, boundary=None) -> dict:
        """The typed alphabet at ONE boundary. Core FAILS CLOSED if this is empty."""
        return dict(_FIELDS_AT.get(str(getattr(boundary, "value", boundary)), {}))

    def install_signal(self, name: str, predicate, *, boundary=None, provenance: str = "") -> None:
        self.expanded[name] = predicate
        pt = boundary if isinstance(boundary, IncisionPoint) else _KEY_TO_POINT.get(str(boundary))
        self.expanded_boundaries[name] = {pt} if pt is not None else set()

    def reset_expanded_signals(self) -> None:
        self.expanded.clear()
        self.expanded_boundaries.clear()

    def expanded_signal_names(self) -> tuple[str, ...]:
        """Core needs these to decide whether an expanded signal may reach a signal-agnostic cell."""
        return tuple(self.expanded)

    def evaluate_signal(self, signal: str, state: Mapping[str, Any], params=None) -> bool:
        if signal in self.expanded:
            return bool(self.expanded[signal](state))
        if signal == "filing_failed":
            return str(state.get("error_kind") or "") != ""
        if signal == "nothing_proposed":
            return not bool(state.get("proposes_write"))
        raise KeyError(f"{self.name} cannot evaluate signal {signal!r}")

    def probe_params(self, signal: str) -> dict:
        return {}

    def parameter_domains(self, signal: str) -> tuple:
        """No tunable theta_phi in this toy: every signal is DETERMINISTIC once installed."""
        return ()

    def states_at(self, boundary, states: Sequence[Mapping[str, Any]]) -> list:
        """Project states onto ONE boundary's information set. Core fails closed without this.

        THIS IS NOT OPTIONAL POLISH. A predicate over `payload_chars` cannot be validated on a
        post-execution state, and a predicate over `error_kind` cannot fire at the gate. Returning
        every state regardless is how a controller gets selected that can never fire.
        """
        want = set(self.synthesis_fields(boundary))
        out = []
        for st in states or ():
            if str(st.get("boundary") or "") == str(getattr(boundary, "value", boundary)):
                out.append({k: v for k, v in st.items() if k in want or k in ("boundary", "case_id")})
        return out

    # ---------------------------------------------------------------------------- HOW: eta grounding
    def ground_suppress(self, signal: str, boundary) -> list:
        """eta_mu for SUPPRESS. The boundary IS part of the requirement: after execution there is
        nothing left to cancel, so this grounds only at the gate."""
        if str(getattr(boundary, "value", boundary)) != \
                IncisionPoint.POST_GENERATION_PRE_EXEC.value:
            return []
        return [{
            # The remove-outright variant: the proposal is cancelled and the agent sees nothing in its
            # place. It preserves NOTHING and is not asked to -- see `variant_required` in
            # action_contract: `preservation` is mandatory only for the replay variant.
            "variant": "cancel_proposed",
            "eta": {"suppressed_operation": "the proposed filing"},
            "detail": "cancel the proposed filing before it commits",
        }]

    def ground_reprompt(self, signal: str, boundary) -> list:
        """Two SEMANTICALLY DISTINCT instructions -- each becomes its own arm, and measurement picks.

        Naming one and being wrong costs the round; naming two costs two paired arms.
        """
        return [
            {"variant": "advise_archive",
             "eta": {"instruction": "The drawer rejects long reports. Archive it instead.",
                     "retry_budget": 1},
             "detail": "redirect the agent's own next choice"},
            {"variant": "advise_shorten",
             "eta": {"instruction": "Shorten the report before filing it.", "retry_budget": 1},
             "detail": "ask for a smaller payload"},
        ]

    def ground_substitute_destinations(self, signal: str, boundary) -> list:
        if str(getattr(boundary, "value", boundary)) != IncisionPoint.POST_EXECUTION.value:
            return []
        return [{"variant": "reroute_to_archive",
                 "eta": {"destination": "archive_report", "argument_mapping": "text -> text",
                         "retry_semantics": "replace"},
                 "detail": "re-file into the archive, which has no length limit"}]

    def ground_transforms(self, signal: str, boundary) -> list:
        return []

    # ---------------------------------------------------------------------------- executors
    #
    # THE BOUND FORM. Each cell names the code that RUNS and the eta keys that code READS. A cell with
    # no binding is a ghost and core refuses it; a contract-required key outside `consumes` is inert
    # eta and core refuses that too. `toy_host.runtime` below is the executing code, and
    # `tests/test_executor_behavioral_contract.py` is where the claim is proven by behaviour rather
    # than by this declaration.
    _CAPS = {
        (IncisionPoint.POST_GENERATION_PRE_EXEC.value, "suppress"): ExecutorCapability(
            boundary=IncisionPoint.POST_GENERATION_PRE_EXEC.value, action="suppress",
            binding="toy_host.runtime.ToyRuntime.step:withhold_proposed_call",
            # The executor reads the budget and nothing else: it decides WHETHER to withhold from the
            # predicate, and the identity of the cancelled call comes from the dispatch list, not from
            # the candidate. So `suppressed_operation` is the arm's identity, not executor input --
            # declared via eta_is_computed rather than over-claimed in `consumes`.
            consumes=("retry_budget",),
            signals=(),                 # covers any signal: the mechanism reads live state
            signal_agnostic=True,
            eta_is_computed=True,
            detail="removes the proposed call from the dispatch list before it runs"),
        (IncisionPoint.POST_GENERATION_PRE_EXEC.value, "reprompt"): ExecutorCapability(
            boundary=IncisionPoint.POST_GENERATION_PRE_EXEC.value, action="reprompt",
            binding="toy_host.runtime.ToyRuntime.step:inject_instruction",
            consumes=("instruction", "retry_budget"),
            signals=(), signal_agnostic=True,
            detail="appends the instruction to the agent's context and regenerates once"),
        (IncisionPoint.POST_EXECUTION.value, "reroute"): ExecutorCapability(
            boundary=IncisionPoint.POST_EXECUTION.value, action="reroute",
            binding="toy_host.runtime.ToyRuntime.step:redispatch_to_archive",
            consumes=(), eta_is_computed=True,
            signals=(), signal_agnostic=True,
            detail="re-dispatches the failed filing into the archive"),
        # POST_EXECUTION/reprompt is DELIBERATELY ABSENT, not disabled: this host has no
        # post-execution regeneration path at all. Core reports `no_executor_capability`, which is a
        # different fact from "declared but inert" and leads a porter to a different fix.
    }

    def executor_capability(self, boundary, action):
        return self._CAPS.get((str(getattr(boundary, "value", boundary)),
                               str(getattr(action, "value", action))))

    def capability_audit(self) -> dict:
        from anchoropt.learning.executor_capability import (
            disabled_capabilities, unbound_capabilities,
        )
        caps = tuple(self._CAPS.values())
        return {"unbound": list(unbound_capabilities(caps)),
                "disabled": list(disabled_capabilities(caps)),
                "bound_and_enabled": sorted(f"{c.boundary}/{c.action}" for c in caps
                                            if c.is_bound and c.is_enabled)}


ADAPTER = ToyReportHost()
