"""THE ADAPTER. Every contract hook AnchorOpt requires for tau-bench, and nothing else.

WHAT THIS OBJECT ANSWERS: questions about the tau-bench RUNTIME. Which trajectory events are
consequential decisions, what is observable at each, which actions this host can execute where, and how
to parameterize one.

WHAT IT NEVER ANSWERS: questions about the FAILURE. There is no "for this kind of error, use action X"
anywhere in this file. Choosing a locus, a condition and an action is core's job, decided by
measurement; putting that logic here would invalidate the experiment. The only filtering done here is
MATERIALIZABILITY -- can this host actually run the thing?

No tau2 import: `tau2_mechanism` is the executing code and is loaded only when episodes run, so the
contract and observability tests need no tau-bench checkout and no model.

THE TOOL CATALOG. Grounding a REROUTE destination requires real tool schemas, which belong to the
benchmark rather than to core. They arrive through `set_tool_catalog` -- populated from a live
environment by `tau2_episodes.tool_catalog`, or from a recorded fixture offline. With no catalog,
`ground_substitute_destinations` returns [] and says why: an empty grounding is an honest "this host
cannot offer a destination", whereas a guessed destination is an arm that would error rather than repair.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from anchoropt.anchor import Action, IncisionPoint
from anchoropt.runtime import HostProfile

import tau2_capabilities as _caps
import tau2_fields as _flds
import tau2_signals as _sig
from tau2_fields import AFTER, GATE, IDENTITY_KEYS, KEY_TO_POINT, TURN_START

NAME = "tau2"

CATALOG_DIR = Path(__file__).resolve().parent / "catalogs"

# ------------------------------------------------------------------------------------------------
# U_H(l) -- the abstract admissibility grid, narrowed to what this host has an executor for.
# ------------------------------------------------------------------------------------------------
#
# Evidence per cell is in `tau2_capabilities` (the binding) and proven by behaviour in
# `tests/test_tau2_executor_behavioral.py`. POST_EXECUTION carries REROUTE because the cell EXISTS and
# is declared DISABLED with its reason -- core must be able to refuse arms there and say why, which it
# cannot do for a cell that was simply left out.
HOST = HostProfile(
    name=NAME,
    executable={
        IncisionPoint.PRE_GENERATION: frozenset({Action.NOOP, Action.REPROMPT}),
        IncisionPoint.POST_GENERATION_PRE_EXEC: frozenset({
            Action.NOOP, Action.REPROMPT, Action.SUPPRESS, Action.REROUTE,
        }),
        IncisionPoint.POST_EXECUTION: frozenset({
            Action.NOOP, Action.REPROMPT, Action.REROUTE,
        }),
    },
    notes=("POST_EXECUTION/REROUTE is declared and DISABLED: the agent cannot dispatch tools, and "
           "rewriting the returned ToolMessage corrupts the trajectory the evaluator replays. "
           "POST_EXECUTION/SUPPRESS is omitted for the structural reason core already enforces. "
           "SUPPRESS grounds the remove-outright variant only -- the runtime keeps no recorded-result "
           "store, so withhold-and-replay has no executor."),
)


class Tau2Runtime:
    """The adapter. Every method is a contract hook; nothing here is decoration."""

    name = NAME
    HOST = HOST

    def __init__(self) -> None:
        # Expanded signals live on the ADAPTER: core hands back a predicate and a name, and the adapter
        # owns evaluation. `reset_expanded_signals` lets a test re-run from the shipped vocabulary.
        self.expanded: dict[str, Any] = {}
        self.expanded_boundaries: dict[str, set] = {}
        self._catalog: dict[str, dict[str, Any]] = {}
        # TEACHER-PROPOSED eta, injected rather than imported: the adapter stays pure and the teacher
        # stays optional. These are APPENDED after the shipped variants, so position carries no
        # preference -- the adapter must never rank, and measurement decides among all of them.
        self._proposed_instructions: list[dict[str, Any]] = []

    # ============================================================================ tool catalog
    def set_tool_catalog(self, catalog: Mapping[str, Mapping[str, Any]]) -> None:
        """Install real tool facts: {name: {"mutates_state": bool, "required": [...], "params": [...]}}."""
        self._catalog = {str(k): dict(v) for k, v in dict(catalog or {}).items()}

    def load_tool_catalog(self, domain: str) -> bool:
        path = CATALOG_DIR / f"{domain}.json"
        if not path.exists():
            return False
        self.set_tool_catalog(json.loads(path.read_text()))
        return True

    @property
    def tool_catalog(self) -> Mapping[str, Mapping[str, Any]]:
        return dict(self._catalog)

    # ============================================================================ teacher eta
    def set_instruction_proposals(self, groundings: Sequence[Mapping[str, Any]]) -> None:
        """Install teacher-proposed REPROMPT eta. Empty (the default) means history-only."""
        self._proposed_instructions = [dict(g) for g in (groundings or ())]

    def instruction_proposals(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(g) for g in self._proposed_instructions)

    def reset_instruction_proposals(self) -> None:
        self._proposed_instructions = []

    # ============================================================================ WHERE
    def is_decision(self, event: Mapping[str, Any]) -> bool:
        """Which trajectory events are CONSEQUENTIAL decisions. Localization runs over these only.

        ANSWER-COMMITMENT COUNTS. A turn that replies to the user instead of acting is a decision: in
        tau-bench an agent fails by talking as readily as by acting. Requiring a tool call here would
        delete that boundary, and localization would then derive exactly one -- the specific bug
        ADAPTER_GUIDE section 9 names.
        """
        kind = str(event.get("kind") or "")
        if kind in {"turn_start", "propose", "result"}:
            return True
        return bool(event.get("boundary"))

    def boundary_key(self, event: Mapping[str, Any]) -> str:
        kind = str(event.get("kind") or "")
        by_kind = {"turn_start": TURN_START, "propose": GATE, "result": AFTER}
        if kind in by_kind:
            return by_kind[kind]
        return _flds.POINT_TO_KEY.get(str(event.get("boundary") or ""), "")

    def boundary_from_key(self, key: str):
        """THE ONLY key->locus mapping. Core owns no such table by design."""
        return KEY_TO_POINT.get(str(key))

    def label_for(self, key: str) -> str:
        return _flds.LABELS.get(str(key), "")

    def depends_on(self, key: str) -> tuple[str, ...]:
        """Realized order within one agent turn: turn start -> proposal -> result."""
        return {TURN_START: (), GATE: (TURN_START,), AFTER: (GATE,)}.get(str(key), ())

    # ============================================================================ WHAT
    def declared_signals(self) -> tuple[str, ...]:
        return tuple(_sig.SIGNALS) + tuple(self.expanded)

    def signal_boundaries(self, signal: str) -> frozenset:
        if signal in self.expanded:
            return frozenset(self.expanded_boundaries.get(signal, ()))
        return _sig.SIGNAL_BOUNDARIES.get(signal, frozenset())

    def signal_aliases(self, signal: str) -> tuple[str, ...]:
        return _sig.SIGNAL_ALIASES.get(signal, ())

    def synthesis_fields(self, boundary=None) -> dict:
        """The typed alphabet at ONE boundary. Core FAILS CLOSED if this is empty."""
        return _flds.fields_at(boundary)

    def install_signal(self, name: str, predicate, *, boundary=None, provenance: str = "") -> None:
        """Accept a synthesized predicate. Without this hook Phi CANNOT be expanded."""
        self.expanded[name] = predicate
        pt = boundary if isinstance(boundary, IncisionPoint) else KEY_TO_POINT.get(str(boundary))
        if pt is None:
            raw = str(getattr(boundary, "value", boundary))
            pt = next((p for p in IncisionPoint if p.value == raw), None)
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
        return _sig.evaluate(signal, state, params)

    def probe_params(self, signal: str) -> dict:
        return {}

    def parameter_domains(self, signal: str) -> tuple:
        """No tunable theta_phi: shipped signals are DETERMINISTIC once installed.

        Thresholded conditions come from core's grammar over the typed fields -- declaring a grid here
        would be this adapter pre-choosing the cut points, which is the search's job.
        """
        return _sig.SIGNAL_PARAM_DOMAINS.get(signal, ())

    def states_at(self, boundary, states: Sequence[Mapping[str, Any]]) -> list:
        """Project states onto ONE boundary's information set. FILTERS, NEVER INVENTS.

        This is the hook that stops a controller being selected that can never fire. A predicate over
        `proposed_args_chars` cannot be validated on a post-execution state, and one over `error_kind`
        cannot fire before dispatch.
        """
        want = set(self.synthesis_fields(boundary))
        target = str(getattr(boundary, "value", boundary))
        if target in KEY_TO_POINT:
            target = KEY_TO_POINT[target].value
        out = []
        for st in states or ():
            if str(st.get("boundary") or "") != target:
                continue
            out.append({k: v for k, v in st.items() if k in want or k in IDENTITY_KEYS})
        return out

    # ============================================================================ events
    def normalize_event(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """One recorded trajectory row -> the boundary-tagged record the other hooks read.

        `tau2_episodes.trajectory_events` already emits this shape from a SimulationRun, so this is a
        pass-through that also accepts a row carrying only a locus value.
        """
        rec = dict(row)
        if not rec.get("boundary"):
            key = self.boundary_key(rec)
            pt = self.boundary_from_key(key)
            if pt is not None:
                rec["boundary"] = pt.value
        if not rec.get("kind"):
            rec["kind"] = {
                IncisionPoint.PRE_GENERATION.value: "turn_start",
                IncisionPoint.POST_GENERATION_PRE_EXEC.value: "propose",
                IncisionPoint.POST_EXECUTION.value: "result",
            }.get(str(rec.get("boundary") or ""), "")
        return rec

    def observable_state(self, normalized: Mapping[str, Any]) -> dict[str, Any]:
        """What Phi may read at this event's own boundary, plus the identity keys."""
        rec = dict(normalized)
        want = set(_flds.fields_at(rec.get("boundary")))
        return {k: v for k, v in rec.items() if k in want or k in IDENTITY_KEYS}

    # ============================================================================ HOW: grounding
    #
    # Each grounder returns the eta VARIANTS this host can execute at that boundary. More than one
    # variant means more than one arm, and measurement picks. Naming one and being wrong costs the
    # round; naming two costs two paired evaluations.
    def ground_reprompt(self, signal: str, boundary) -> list:
        """eta_mu for REPROMPT. Available at all three loci; the mechanism differs per locus.

        The two instructions are SEMANTICALLY DISTINCT and neither is preferred here. They are phrased
        over the runtime's own vocabulary (a call, the policy, the result) and name no failure class.
        """
        locus = self._locus(boundary)
        if locus not in (IncisionPoint.PRE_GENERATION.value,
                        IncisionPoint.POST_GENERATION_PRE_EXEC.value,
                        IncisionPoint.POST_EXECUTION.value):
            return []
        out = [
            {"variant": "recheck_against_policy",
             "eta": {"instruction": ("Before continuing, re-read the operative section of the policy "
                                     "and confirm this step is the one it prescribes."),
                     "retry_budget": 1},
             "detail": "ask for one policy re-check before the next decision"},
            {"variant": "require_verification_first",
             "eta": {"instruction": ("Verify the details you need with a read before taking any step "
                                     "that changes records."),
                     "retry_budget": 1},
             "detail": "ask for a confirming read first"},
        ]
        # `retry_budget` is carried at EVERY locus because every reprompt executor here reads it: at
        # the gate it bounds re-planning, and at the other two it bounds how often the instruction may
        # be injected in one episode. Stripping it at pre_generation (an earlier version did) made every
        # grounding there incomplete against core's REPROMPT contract.
        #
        # TEACHER ETA LAST, and only the keys this cell's executor reads. A proposal is one more arm to
        # measure, never a preferred one: the adapter appends it and says nothing about its quality.
        cap = self.executor_capability(boundary, Action.REPROMPT)
        allowed = set(getattr(cap, "consumes", ()) or ())
        for extra in self._proposed_instructions:
            eta = {k: v for k, v in dict(extra.get("eta") or {}).items() if k in allowed}
            if "instruction" not in eta:
                continue
            out.append({"variant": str(extra.get("variant") or "teacher"), "eta": eta,
                        "detail": str(extra.get("detail") or "teacher-proposed instruction")})
        return out

    def ground_suppress(self, signal: str, boundary) -> list:
        """eta_mu for SUPPRESS. Grounds ONLY where a proposed call can still be cancelled.

        ONE VARIANT, deliberately. `cancel_proposed` removes the call outright: the agent observes that
        it did not happen and re-plans, and nothing is preserved -- so NO `preservation` key. The
        withhold-and-replay variant would need a recorded-result store this runtime does not have, and a
        `preservation` clause no executor reads is not a clause.
        """
        if self._locus(boundary) != IncisionPoint.POST_GENERATION_PRE_EXEC.value:
            return []
        return [{
            "variant": "cancel_proposed",
            "eta": {"suppressed_operation": "the proposed call this signal implicates",
                    "retry_budget": 1},
            "detail": "withhold the proposed call before it is dispatched; the agent re-plans",
        }]

    def ground_substitute_destinations(self, signal: str, boundary) -> list:
        """eta_mu for REROUTE/substitute: REAL destinations, filtered by the real schemas.

        MATERIALIZABILITY ONLY. A destination is offered when the catalog says it exists and its
        required parameters are covered by a plausible argument set; it is not offered because it looks
        like a good idea. With no catalog installed this returns [] rather than guessing.
        """
        if self._locus(boundary) != IncisionPoint.POST_GENERATION_PRE_EXEC.value:
            return []
        if not self._catalog:
            return []
        out = []
        for name in sorted(self._catalog):
            spec = self._catalog[name]
            # A read is a safe substitute target in the sense that matters here: it is executable
            # without changing state, so a rewritten call that lands on it cannot corrupt the replay.
            if spec.get("mutates_state"):
                continue
            required = tuple(spec.get("required") or ())
            if len(required) > 1:
                continue
            out.append({
                "variant": f"substitute_{name}",
                "eta": {"destination": name,
                        "argument_mapping": (f"{required[0]} -> {required[0]}" if required
                                             else "(no arguments)"),
                        "retry_semantics": "replace"},
                "detail": f"rewrite the proposed call to {name}, whose arguments are validated first",
            })
        return out

    def ground_transforms(self, signal: str, boundary) -> list:
        """eta_mu for REROUTE/transform. EMPTY, and that is a finding rather than an omission.

        Reshaping what the model observes would mean rewriting the returned ToolMessage, which the
        orchestrator holds and has already recorded. See the POST_EXECUTION/REROUTE disabled_reason.
        """
        return []

    # ============================================================================ executors
    def executor_capability(self, boundary, action):
        return _caps.capability(boundary, action)

    def capability_audit(self) -> dict:
        return _caps.audit()

    # ============================================================================ helpers
    @staticmethod
    def _locus(boundary) -> str:
        raw = str(getattr(boundary, "value", boundary))
        if raw in KEY_TO_POINT:
            return KEY_TO_POINT[raw].value
        return raw


ADAPTER = Tau2Runtime()
