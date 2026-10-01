"""THE COMPLETE ALGORITHM, end to end, on a deterministic host. Run: python examples/toy_host/demo.py

    residual  ->  backward WHERE localization  ->  WHAT expansion  ->  HOW grounding
              ->  EMPIRICAL paired evaluation  ->  acceptance  ->  moving incumbent  ->  RE-MINE

WHAT MAKES THIS AN ACCEPTANCE TEST AND NOT A DEMO
-------------------------------------------------
Every number printed is measured by actually running the toy host's episodes under each candidate
policy and comparing per-case outcomes against the frozen incumbent. There is no model and no
sampling, so the output is byte-identical on every run -- `tests/test_toy_host_e2e.py` asserts the
specific transitions rather than "it printed something".

THREE DISTINCTIONS THIS DEMONSTRATES, each of which was once collapsed
---------------------------------------------------------------------
  UNEVALUATED  (`REALIZABLE_UNMEASURED`)  a controller was built and grounded, and NOBODY RAN IT.
               Reporting this as "did not help" asserts a measurement that never happened. Run with
               `--no-evaluate` to see the search stop here legitimately.
  NO_BENEFIT   arms were BUILT AND MEASURED and none beat the incumbent. A real negative result.
  BUDGET       the search stopped because its evaluation budget ran out, with arms left unmeasured.
               Neither a negative result nor a structural one. `--budget N` demonstrates it.

WHAT SELECTS THE WINNER: `AnchorPolicyOpt.optimize` over measured J_train. Not a firing-rate
heuristic, not the proposer's preference, and not this file -- the arms are built by core, measured by
the callback below, and the argmax is core's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from anchoropt.anchor import Action, IncisionPoint                            # noqa: E402
from anchoropt.learning.policy_class import ThetaResult                       # noqa: E402
from anchoropt.learning.search_state import (                                 # noqa: E402
    IMPROVED, NO_BENEFIT, REALIZABLE_UNMEASURED,
)
from anchoropt.learning.structured_search import localize, optimize_residual  # noqa: E402
from toy_adapter import ADAPTER                                               # noqa: E402
from toy_host.runtime import CORPUS, Controller, run_corpus                   # noqa: E402

W = 96


def rule(title=""):
    print("=" * W if not title else f"\n{'=' * W}\n{title}\n{'=' * W}")


# ================================================================================================
# THE EVALUATION CALLBACK -- this is the "actual downstream empirical evaluation"
# ================================================================================================

class Evaluator:
    """Runs a real paired comparison of one arm against the FROZEN incumbent.

    The incumbent is frozen for the whole round: every arm is compared against the SAME baseline, or
    the argmax across arms is meaningless. `n_evaluations` is the budget counter -- exhausting it is a
    third outcome, distinct from both "measured and no benefit" and "never measured".
    """

    def __init__(self, incumbent_controller=None, budget: int | None = None):
        self.incumbent_controller = incumbent_controller
        self.baseline = run_corpus(incumbent_controller)
        self.budget = budget
        self.n_evaluations = 0
        self.exhausted = False
        self.log: list[dict] = []

    def _controller_for(self, arm):
        """Materialize a core arm as an installed toy-host policy.

        The predicate comes from the adapter's expanded-signal store, so what runs is the signal core
        actually installed -- not a re-derived guess.
        """
        pred = ADAPTER.expanded.get(arm.signal)
        if pred is None:
            pred = lambda st, _s=arm.signal: bool(ADAPTER.evaluate_signal(_s, st))  # noqa: E731
        return Controller(boundary=arm.boundary.value, action=arm.action.value,
                          predicate=pred, eta=dict(arm.eta), label=arm.label)

    def __call__(self, arm, theta=None):
        if self.budget is not None and self.n_evaluations >= self.budget:
            self.exhausted = True
            return None                      # NOT a zero score: an unmeasured arm has no score
        self.n_evaluations += 1

        got = run_corpus(self._controller_for(arm))
        gains = tuple(c for c, ok in got["solved"].items() if ok and not self.baseline["solved"][c])
        losses = tuple(c for c, ok in got["solved"].items() if not ok and self.baseline["solved"][c])
        res = ThetaResult(
            theta=dict(theta or {}), gains=gains, losses=losses,
            # FIRINGS AND EXECUTIONS COME FROM THE RUN. `train_objective` uses
            # interventions_executed to decide ENGAGEMENT, and an arm that never executed must not
            # tie with one that did -- reporting 0 here would have made every arm look inert and the
            # argmax would have fallen through to the string tiebreak.
            firings=got["interventions_executed"],
            cases_fired=len(got["cases_fired"]), n=got["n"],
            interventions_executed=got["interventions_executed"],
            accuracy_delta_pp=100.0 * (got["n_solved"] - self.baseline["n_solved"])
            / max(1, got["n"]))
        self.log.append({"arm": arm.label, "net": res.net, "gains": len(gains),
                         "losses": len(losses), "delta_pp": round(res.accuracy_delta_pp, 2),
                         "fired": len(got["cases_fired"]),
                         "executed": got["interventions_executed"]})
        return res


# ================================================================================================
# THE ROUND
# ================================================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-evaluate", action="store_true",
                    help="supply NO evaluator: the search must stop at REALIZABLE_UNMEASURED")
    ap.add_argument("--budget", type=int, default=None,
                    help="cap paired evaluations, to demonstrate budget exhaustion")
    ap.add_argument("--no-late-repair", action="store_true",
                    help="remove the post-execution reroute executor, so the LATER boundary has no "
                         "viable repair. Forces the full schedule: exhaust HOW here -> expand WHAT "
                         "here -> retry HOW -> only THEN move WHERE earlier, where the fix requires a "
                         "SYNTHESIZED signal over the proposed payload.")
    ap.add_argument("--json", type=Path, default=None, help="write the machine-readable record here")
    a = ap.parse_args()

    ADAPTER.reset_expanded_signals()
    record: dict = {}

    if a.no_late_repair:
        # The host loses its post-execution repair path. This is the interesting case for the
        # ALGORITHM: the residual is observed at the later boundary but cannot be repaired there, so a
        # correct search must exhaust that boundary, widen Phi at it, fail again, and only then move
        # earlier -- where the condition ("the proposed payload is too long") is not in the shipped
        # vocabulary and has to be SYNTHESIZED from the gate's typed fields.
        ADAPTER._CAPS = {k: v for k, v in ADAPTER._CAPS.items() if k[0] != "post_execution"}

    rule("AnchorOpt on a deterministic toy host")
    base = run_corpus(None)
    print(f"  host      : {ADAPTER.name}   corpus: {len(CORPUS)} episodes")
    print(f"  incumbent : P0 = the unmodified pipeline (no controller)")
    print(f"  baseline  : {base['n_solved']}/{base['n']} solved")
    failing = sorted(c for c, ok in base["solved"].items() if not ok)
    print(f"  residual  : {len(failing)} failing cases  {failing}")
    record["baseline"] = {"n_solved": base["n_solved"], "n": base["n"], "failing": failing}

    # ---- the residual, and the trajectories it actually produced --------------------------------
    events = [e for cid in failing for e in base["events"][cid]]
    states = [dict(e, boundary=("post_generation_pre_exec" if e["kind"] == "propose"
                                else "post_execution"))
              for e in events]

    class _Residual:
        key = "the filing is lost"
        case_ids = tuple(failing)
        rank = 1

    rule("[1] WHERE -- backward localization over the boundaries the trajectory CONTAINS")
    ordered = localize(events, runtime=ADAPTER)
    for i, b in enumerate(ordered):
        print(f"  {i}: {b.key:24s} {ADAPTER.label_for(b.key)}")
    print(f"  the search starts at the LATEST boundary and moves earlier only on exhaustion")
    record["boundaries"] = [b.key for b in ordered]

    # ---- the evaluator (or deliberately none) ---------------------------------------------------
    ev = None if a.no_evaluate else Evaluator(incumbent_controller=None, budget=a.budget)
    if a.no_evaluate:
        print("\n  --no-evaluate: NO evaluator is supplied. The search may localize, expand and")
        print("  ground, but it must NOT claim anything about benefit.")

    rule("[2-4] WHAT expansion, HOW grounding, and EMPIRICAL evaluation")
    outcome = optimize_residual(
        _Residual(), runtime=ADAPTER, host=ADAPTER.HOST, events=events, states=states,
        evaluate=ev,
        # ACCEPTANCE: net positive on the paired comparison. The rule lives with the caller, not in
        # the search -- core measures, the acceptance rule decides.
        improves=(lambda res: res is not None and res.net > 0) if ev else None)

    for att in outcome.attempts:
        print(f"  {att.boundary:24s} {att.state:24s} built={att.candidates_built:3d} "
              f"evaluated={att.candidates_evaluated:3d} -> next {att.coordinate_changed}")
    if outcome.signals_installed:
        print(f"\n  Phi EXPANDED at the boundary ({len(outcome.signals_installed)} installed):")
        for s in outcome.signals_installed[:6]:
            print(f"     {s}")
    if outcome.frontier_boundaries:
        print(f"\n  GROUNDED FRONTIER swept before any expansion: "
              f"{list(outcome.frontier_boundaries)}")
        print("     the SEEDED Phi, re-searched at an earlier boundary -- a WHERE move under a "
              "fixed\n     representation, which is a different move from synthesizing a new one")
    print(f"\n  moves_earlier = {outcome.moves_earlier}    state = {outcome.state}")
    record["search"] = {"state": outcome.state, "moves_earlier": outcome.moves_earlier,
                        "signals_installed": list(outcome.signals_installed),
                        "candidates": len(outcome.candidates),
                        "attempts": [{"boundary": x.boundary, "state": x.state,
                                      "built": x.candidates_built,
                                      "evaluated": x.candidates_evaluated}
                                     for x in outcome.attempts]}

    # ---- the three distinct non-improvement outcomes ---------------------------------------------
    rule("[5] THE OUTCOME, and which of the three non-improvement states applies")
    if ev is not None:
        print(f"  paired evaluations run : {ev.n_evaluations}"
              + (f" / budget {ev.budget}" if ev.budget is not None else " (unbudgeted)"))
        for row in ev.log[:10]:
            print(f"     {row['arm'][:52]:52s} net {row['net']:+3d}  "
                  f"+{row['gains']}/-{row['losses']}  fired {row['fired']:2d}  "
                  f"{row['delta_pp']:+.2f}pp")
        record["evaluations"] = ev.log

    if outcome.state == IMPROVED and outcome.promoted is not None:
        arm = outcome.promoted
        print(f"\n  ACCEPTED   {arm.label}")
        print(f"     locus  : {arm.boundary.value}")
        print(f"     phi    : {arm.signal}")
        print(f"     action : {arm.action.value} ({arm.instantiated.operator.value})")
        print(f"     eta    : {dict(arm.eta)}")
        record["accepted"] = {"label": arm.label, "boundary": arm.boundary.value,
                              "signal": arm.signal, "action": arm.action.value,
                              "eta": {k: str(v) for k, v in arm.eta.items()}}

        # ---- [6] MOVING INCUMBENT + RE-MINE -----------------------------------------------------
        rule("[6] MOVING INCUMBENT -- P0 -> P1, then RE-MINE the residual P1 leaves")
        p1 = Evaluator(incumbent_controller=ev._controller_for(arm))
        print(f"  P0 : {base['n_solved']}/{base['n']}")
        print(f"  P1 : {p1.baseline['n_solved']}/{p1.baseline['n']}   "
              f"(+{p1.baseline['n_solved'] - base['n_solved']})")
        still = sorted(c for c, ok in p1.baseline["solved"].items() if not ok)
        print(f"\n  RE-MINED residual under P1 : {len(still)} failing  {still}")
        if still:
            print("  the next round fits an anchor to THIS residual -- the one P1 leaves, not P0's.")
        else:
            print("  the residual is EMPTY under P1: nothing left for a further anchor to fit.")
            print("  a next round would report no residual rather than search for one.")
        record["incumbent_update"] = {"p0_solved": base["n_solved"],
                                      "p1_solved": p1.baseline["n_solved"],
                                      "remined_residual": still}

    elif outcome.state == REALIZABLE_UNMEASURED:
        print("\n  UNEVALUATED (REALIZABLE_UNMEASURED)")
        print("     A controller was BUILT AND GROUNDED, and no evaluator ran it. This is a")
        print("     STRUCTURAL result: the loop can express and realize an intervention here.")
        print("     It is deliberately NOT `NO_BENEFIT` -- that would assert a measurement that")
        print("     never happened, which is the same error class as believing a null from a")
        print("     channel that was never live.")
        record["outcome_class"] = "UNEVALUATED"

    elif outcome.state == NO_BENEFIT:
        print("\n  NO_BENEFIT -- arms were built AND MEASURED, and none beat the incumbent.")
        print("     A real negative result: the measurement happened and came back negative.")
        record["outcome_class"] = "NO_BENEFIT"

    if ev is not None and ev.exhausted:
        print("\n  BUDGET EXHAUSTED -- arms were left UNMEASURED when the evaluation budget ran out.")
        print("     This is neither a negative result nor a structural one: the search stopped")
        print("     early, and the unmeasured arms are still open. Requeue, do not classify.")
        record["outcome_class"] = "BUDGET_EXHAUSTED"
        record["budget_exhausted"] = True

    rule()
    if a.json:
        a.json.write_text(json.dumps(record, indent=2, default=str) + "\n")
        print(f"record written: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
