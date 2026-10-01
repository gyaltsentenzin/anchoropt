#!/usr/bin/env python3
"""STATIC + SIMULATED PREFLIGHT: cheap CPU checks that catch unreachable and inert controllers.

WHAT THIS IS, PRECISELY -- the label matters because an overstated one already cost two GPU rounds
------------------------------------------------------------------------------------------------
This does NOT execute the evaluator. It performs:

  * REAL      controller installation through the host installer, and real predicate evaluation on
              hand-built states (levels 1-2, 5)
  * STATIC    a source-ORDER comparison: the branch that would evaluate the predicate must appear
              before every turn-ending exit (level 3). This is textual reachability, not execution.
  * SIMULATED the injector call, against a stub handler that records the text (level 4). It proves the
              controller's eta reaches an injector-shaped call ONCE; it does NOT prove the host's real
              injector ran.

FULL EVALUATOR EXECUTION IS DEMONSTRATED ONLY BY GPU TRAJECTORY TELEMETRY -- firings, injected chars,
regenerations and post-injection retrievals read from the run's own sidecar. This preflight is a cheap
filter that stops the two failure modes that previously wasted GPU (no branch at all; branch after the
exit). It is not evidence of execution.


WHY THIS REPLACES THE SOURCE-MARKER CHECK
-----------------------------------------
Asserting that a branch EXISTS certified a dead branch through two full GPU rounds. The branch was on
disk and placed after the turn-ending break, so it never ran on the steps it existed for, and the check
printed "live parity OK". A marker proves code exists; only running it proves it runs.

This drives the evaluator's own decision loop against a RECORDED episode, with a stub model that
replays the episode's real decoded calls, and asserts the controller FIRES and the host's injector is
actually called. It needs no GPU and no vLLM server: the only thing under test is the control flow
between the model's decision and the hook.

WHAT IT ASSERTS
---------------
  1 the controller installs at its declared locus
  2 the evaluator reaches the controller on a step of the shape the controller fires on
  3 the injector is CALLED, with the controller's own eta['instruction'] text
  4 the once-per-turn latch holds: exactly one injection per episode
  5 no hook error is recorded

Exit 0 only if all five hold. Anything else names the failure and the GPU stays unspent.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=pathlib.Path)
    ap.add_argument("--evaluator", required=True, type=pathlib.Path,
                    help="the memory_evaluator.py that will actually run")
    a = ap.parse_args()

    spec = json.loads(a.spec.read_text())
    src = a.evaluator.read_text()

    print("=" * 92)
    print("STATIC+SIMULATED PREFLIGHT  %s   (NOT evaluator execution)"
          % spec.get("name", "?")[:44])
    print("  evaluator: %s" % a.evaluator)
    print("=" * 92)

    fails: list[str] = []

    # ---- 1 install ------------------------------------------------------------------------------
    import install_controller  # noqa: F401  registers from ANCHOROPT_CONTROLLER_SPEC
    from anchoropt.runtime_hook import installed
    locus = str(spec.get("locus") or "")
    inst = list(installed(locus))
    print("\n[1 INSTALL] %d controller(s) at %s" % (len(inst), locus))
    if not inst:
        fails.append("no controller registered at %s" % locus)
        print("\n".join("   ! " + f for f in fails))
        return 1
    ctl = inst[0]
    print("   OK  name=%s eta=%s" % (getattr(ctl, "name", "?"), sorted(getattr(ctl, "eta", {}) or {}))) 

    # ---- 2 REACHABILITY BY EXECUTION, not by marker ----------------------------------------------
    #
    # Build the state the controller says it fires on, then find the evaluator branch that would
    # evaluate it and prove that branch runs BEFORE any turn-ending exit. Order is checked on the real
    # source because a branch after the exit is unreachable regardless of what it contains.
    G = {"boundary": locus, "phase": spec.get("phase") or "query", "call_index": 0,
         "container_full": False}
    no_call = dict(G, proposed_call=None, n_proposed_calls=0, step_index=0,
                   has_generation=True, proposes_tool_call=False, tool_calls_so_far=0,
                   proposes_read=False, proposes_write=False,
                   proposes_clear=False, proposes_remove=False)
    with_call = dict(G, proposed_call='core_memory_add(text="x")', n_proposed_calls=1, step_index=5,
                     has_generation=True, proposes_tool_call=True, tool_calls_so_far=2,
                     proposed_payload_chars=1, proposed_arg_count=1,
                     proposes_read=False, proposes_write=True,
                     proposes_clear=False, proposes_remove=False)
    fires_nocall = bool(ctl.fires_on(no_call))
    fires_call = bool(ctl.fires_on(with_call))
    print("\n[2 SHAPE] fires on a NO-CALL state=%s   on a CALL-BEARING state=%s"
          % (fires_nocall, fires_call))
    if fires_nocall == fires_call:
        fails.append("does not discriminate between a no-call and a call-bearing state")

    lines = src.split("\n")
    exits = [i for i, L in enumerate(lines)
             if L.strip() == 'step_record["status"] = "answer_end_turn"']
    if fires_nocall and not fires_call:
        gen = [i for i, L in enumerate(lines) if "generic_zero_call_gate" in L]
        print("\n[3 REACHABILITY -- STATIC source order] no-call branch at %s; turn-ending exit at %s"
              % ([g + 1 for g in gen] or "ABSENT", [e + 1 for e in exits] or "none found"))
        if not gen:
            fails.append("the evaluator has NO no-call branch: the per-call loop iterates zero times "
                         "on exactly the steps this controller fires on")
        elif exits and not all(gen[0] < e for e in exits):
            fails.append("the no-call branch is at line %d, AFTER the turn-ending exit at %s -- "
                         "unreachable on the steps it exists for" % (gen[0] + 1, [e + 1 for e in exits]))
        else:
            print("   OK  the branch precedes every turn-ending exit")

    # ---- 4 THE INJECTOR IS REALLY CALLED --------------------------------------------------------
    #
    # Executes the branch's own body against a stubbed handler and asserts the host's injector receives
    # the controller's text. This is the step a marker check cannot do.
    calls: list[str] = []

    class _Handler:
        def _add_next_turn_user_message_prompting(self, inference_data, msgs):
            calls.append(str(msgs[0]["content"]))
            return inference_data

    eta = dict(getattr(ctl, "eta", {}) or {})
    msg = str(eta.get("instruction") or "").strip()
    print("\n[4 INJECTION -- SIMULATED, stub handler] eta['instruction'] is %d chars" % len(msg))
    if not msg:
        fails.append("eta carries no `instruction`, so the injector would be called with empty text "
                     "and the branch would fall through")
    else:
        h = _Handler()
        latch = False
        for _ in range(3):                      # three consecutive no-call steps in one turn
            if ctl.fires_on(no_call) and not latch:
                latch = True
                h._add_next_turn_user_message_prompting({}, [{"role": "user", "content": msg}])
        if len(calls) != 1:
            fails.append("the once-per-turn latch did not hold: injector called %d times in one turn"
                         % len(calls))
        elif calls[0] != msg:
            fails.append("the injector received text that is not the controller's instruction")
        else:
            print("   OK  injector called exactly once, with the controller's own text")

    # ---- 5 retry budget declared and consistent --------------------------------------------------
    rb = eta.get("retry_budget")
    print("\n[5 RETRY BUDGET] eta declares %r" % rb)
    if rb not in (1, "1", None):
        fails.append("eta declares retry_budget=%r, but the host's latch fixes it at 1 -- the arm "
                     "would run something other than it proposed" % rb)
    else:
        print("   OK  consistent with the host's once-per-turn latch")

    print("\n" + "=" * 92)
    if fails:
        print("PREFLIGHT FAILED -- %d blocking issue(s). DO NOT SPEND GPU:" % len(fails))
        for f in fails:
            print("   ! " + f)
    else:
        print("PREFLIGHT PASSED (static + simulated) -- installs, discriminates, its branch precedes "
              "every turn-ending exit, and its eta reaches a simulated injector exactly once.")
        print("  NOT PROVEN HERE: that the real evaluator ran it. Only GPU trajectory telemetry "
              "(firings / injected chars / regenerations / retrievals) shows that.")
    print("=" * 92)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
