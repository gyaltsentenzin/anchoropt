"""THE EXECUTING CODE. This is what the adapter's `binding` strings name, and what makes the toy
example a real end-to-end proof rather than a mock.

A deterministic agent files reports into a fixed-size drawer. It is intentionally a BAD agent in one
specific way -- it files long reports into the drawer, which rejects them -- because that is the
residual the search has to find and repair.

DETERMINISM IS THE POINT. No model, no sampling, no clock, no network: the same episodes produce the
same trajectories and the same measured deltas on every run, on every machine. That is what lets the
full algorithm (localize -> expand -> ground -> MEASURE -> accept -> re-mine) be an acceptance test
instead of a demonstration whose output you have to squint at.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

DRAWER_LIMIT = 80


@dataclass
class Episode:
    """One task. `reports` are what the agent will try to file, in order."""

    case_id: str
    reports: tuple[str, ...]
    # The task is solved if every report is retrievable at the end -- from the drawer OR the archive.
    # A report the drawer rejected and that went nowhere else is LOST, and the episode fails.


@dataclass
class Store:
    drawer: list[str] = field(default_factory=list)
    archive: list[str] = field(default_factory=list)

    def holds(self, text: str) -> bool:
        return text in self.drawer or text in self.archive


class ToyRuntime:
    """The host. `step` is the boundary mechanism every capability binding refers to."""

    def __init__(self) -> None:
        self.budget_spent: dict[str, int] = {}
        # FIRING TELEMETRY, counted at the site where the intervention actually runs. `train_objective`
        # requires interventions_executed > 0 for an arm to count as ENGAGED: an arm with a delta it
        # did not cause must never outrank one that did. Counting a predicate match that was then
        # budget-blocked would overstate engagement, so `executed` increments only where the mechanism
        # really altered the run.
        self.fired_cases: set[str] = set()
        self.executed: int = 0

    # ------------------------------------------------------------------ the agent's fixed policy
    @staticmethod
    def _propose(report: str, *, hint: str = "", blocked: tuple[str, ...] = ()) -> dict:
        """The agent's decision. Deterministic, and deliberately wrong for long reports.

        TWO INDEPENDENT INPUTS, and keeping them separate is what makes the two action families
        genuinely different interventions rather than one mechanism with two labels:

          `hint`     text that ACTUALLY ENTERED the agent's context. Only a REPROMPT controller
                     supplies this. The agent reads it and complies, which is what makes that
                     executor real rather than telemetry -- an agent that ignored the hint would make
                     reprompt inert, and the behavioral contract test would catch it.

          `blocked`  tools whose call is ABSENT from the conversation, because it was withheld. This
                     is all a SUPPRESS controller provides: no instruction, no explanation. The agent
                     observes that its filing did not happen and re-plans on its own.

        WHY THIS MATTERS. An earlier version of this file read `hint or "Archive it instead"` at the
        retry site, so a suppression that supplied NO instruction was handed the reprompt's text
        anyway. That made SUPPRESS secretly a REPROMPT: the two families would have measured
        identically for a reason invented by the host. It is the same error the project already
        retracted once (R10) -- attributing a repair to a corrective instruction when the instruction
        was inert and the real cause was OMISSION.
        """
        if "Archive it instead" in hint:
            return {"tool": "archive_report", "text": report}
        if "Shorten" in hint:
            return {"tool": "file_report", "text": report[:DRAWER_LIMIT]}
        if "file_report" in blocked:
            # NO INSTRUCTION WAS GIVEN. The agent re-plans from the absence of its own filing: the
            # drawer route did not happen, so it tries the other destination it knows about. This is
            # the suppression mechanism's entire causal path.
            return {"tool": "archive_report", "text": report}
        return {"tool": "file_report", "text": report}

    # ------------------------------------------------------------------ one episode
    def run_episode(self, ep: Episode, controller=None) -> dict:
        """Run one episode and return its trajectory + outcome.

        The trajectory is the adapter's event vocabulary: `propose` events carry the proposed call,
        `execute` events carry its result. That is what `is_decision`/`boundary_key` read.
        """
        store = Store()
        events: list[dict] = []
        hint = ""

        for report in ep.reports:
            call = self._propose(report, hint=hint)
            hint = ""

            # ---- BOUNDARY 1: post_generation_pre_exec. The call exists; nothing has run. ----------
            gate_state = {
                "boundary": "post_generation_pre_exec",
                "case_id": ep.case_id,
                "payload_chars": len(str(call.get("text") or "")),
                "target": str(call.get("tool") or ""),
                "proposes_write": str(call.get("tool") or "").endswith("_report"),
            }
            events.append({"kind": "propose", "case_id": ep.case_id, "call": dict(call),
                           **{k: v for k, v in gate_state.items() if k != "boundary"}})

            intervened = False
            blocked: tuple[str, ...] = ()
            if controller is not None and controller.boundary == "post_generation_pre_exec":
                if controller.fires_on(gate_state):
                    budget = int(controller.eta.get("retry_budget", 1) or 1)
                    spent = self.budget_spent.get(ep.case_id, 0)
                    if spent < budget:
                        self.budget_spent[ep.case_id] = spent + 1
                        self.fired_cases.add(ep.case_id)
                        self.executed += 1
                        if controller.action == "suppress":
                            # BINDING: withhold_proposed_call. The call is removed from the dispatch
                            # list. NO instruction is supplied and none is manufactured -- the agent
                            # is told nothing and only observes that its filing is absent.
                            intervened = True
                            blocked = (str(call.get("tool") or ""),)
                        elif controller.action == "reprompt":
                            # BINDING: inject_instruction. The instruction genuinely enters the
                            # agent's context -- `_propose` reads it. If it did not, two instruction
                            # variants would give identical trajectories, which is the inert-action
                            # defect (D3).
                            intervened = True
                            hint = str(controller.eta.get("instruction") or "")

            if intervened:
                # The agent decides again on the SAME report, once. It receives EXACTLY what its
                # controller supplied: a hint (reprompt) or the knowledge that a tool call did not
                # happen (suppress). Never both, and never a hint nobody supplied.
                call = self._propose(report, hint=hint, blocked=blocked)
                hint = ""

            # ---- execute ---------------------------------------------------------------------------
            error = ""
            if call["tool"] == "file_report":
                if len(call["text"]) > DRAWER_LIMIT:
                    error = "drawer_rejected_oversize"
                else:
                    store.drawer.append(call["text"])
            elif call["tool"] == "archive_report":
                store.archive.append(call["text"])

            # ---- BOUNDARY 2: post_execution. The world moved. -------------------------------------
            exec_state = {
                "boundary": "post_execution",
                "case_id": ep.case_id,
                "error_kind": error,
                "result_chars": 0 if error else len(str(call.get("text") or "")),
                "succeeded": not error,
            }
            events.append({"kind": "execute", "case_id": ep.case_id, "call": dict(call),
                           **{k: v for k, v in exec_state.items() if k != "boundary"}})

            if error and controller is not None and controller.boundary == "post_execution":
                if controller.fires_on(exec_state) and controller.action == "reroute":
                    self.fired_cases.add(ep.case_id)
                    self.executed += 1
                    # BINDING: redispatch_to_archive. Derived from live state, reads no candidate eta.
                    store.archive.append(call["text"])
                    events[-1] = dict(events[-1], succeeded=True, error_kind="")

        solved = all(store.holds(r) or store.holds(r[:DRAWER_LIMIT]) for r in ep.reports)
        return {"case_id": ep.case_id, "events": events, "solved": solved,
                "store": {"drawer": list(store.drawer), "archive": list(store.archive)}}


# ================================================================================================
# THE CORPUS. Fixed, small, and deterministic.
# ================================================================================================
#
# Half the episodes file an over-long report (the residual) and half do not (so a controller that
# fires everywhere is measurably worse than one that discriminates -- which is what makes MEASUREMENT
# do real work here instead of rubber-stamping the only candidate).

_LONG = "A detailed incident report. " * 6          # > DRAWER_LIMIT
_SHORT = "Brief note."                              # < DRAWER_LIMIT

CORPUS: tuple[Episode, ...] = (
    Episode("c01", (_LONG,)),
    Episode("c02", (_LONG,)),
    Episode("c03", (_LONG,)),
    Episode("c04", (_LONG,)),
    Episode("c05", (_LONG,)),
    Episode("c06", (_LONG,)),
    Episode("c07", (_SHORT,)),
    Episode("c08", (_SHORT,)),
    Episode("c09", (_SHORT,)),
    Episode("c10", (_SHORT,)),
    Episode("c11", (_SHORT, _LONG)),
    Episode("c12", (_LONG, _SHORT)),
)


@dataclass
class Controller:
    """An installed policy: (boundary, predicate, action, eta). The 4-tuple core selects."""

    boundary: str
    action: str
    predicate: Any
    eta: Mapping[str, Any] = field(default_factory=dict)
    label: str = ""

    def fires_on(self, state: Mapping[str, Any]) -> bool:
        try:
            return bool(self.predicate(state))
        except Exception:
            return False


def run_corpus(controller=None, corpus: Sequence[Episode] = CORPUS) -> dict:
    """Run every episode under one policy. THIS IS THE EVALUATION the optimizer calls.

    Returns per-case solved flags, so a PAIRED comparison against an incumbent is a set difference --
    the same shape as a real benchmark's paired run, with the sampling noise removed.
    """
    rt = ToyRuntime()
    results = {ep.case_id: rt.run_episode(ep, controller) for ep in corpus}
    return {
        "solved": {cid: r["solved"] for cid, r in results.items()},
        "n_solved": sum(r["solved"] for r in results.values()),
        "n": len(results),
        "events": {cid: r["events"] for cid, r in results.items()},
        # FIRINGS COME FROM THE RUN, never from a registry or a declared flag. This is the toy
        # equivalent of reading the trajectory sidecar rather than a gates_fired dict: only what the
        # mechanism did at its own site counts.
        "cases_fired": sorted(rt.fired_cases),
        "interventions_executed": rt.executed,
    }
