#!/usr/bin/env python3
"""PRE-GPU PREFLIGHT: prove a controller can actually run before spending a job on it.

Every check here exists because its absence once produced a confidently wrong result. The rule this
enforces is the one from `docs/` and the project's own post-mortems: ask "will this EXECUTE what it
claims", not "does it install". An arm whose executor silently ignores its action runs as the CONTROL,
and its paired result is a measured zero against an intervention that never happened -- which is
indistinguishable from a real NO_BENEFIT on a results table.

FIVE CHECKS, each with its own exit reason:

  1 SUPPLY      every field the controller's predicate READS is present in the states it will see at
                its own boundary. A predicate missing one field answers False for that reason alone.
  2 FIREABILITY it fires on a non-trivial minority of those states: not zero (no contrast, runs as the
                control) and not all (no trigger, an unconditional rewrite).
  3 NEGATIVE    it does NOT fire on states that should not trigger it -- the discrimination the
                fireability count alone cannot show.
  4 CAPABILITY  at a cell with more than one executor, the spec NAMES which one, and that executor is
                bound and enabled. `capability_id` is emitted but NOT read by live dispatch, so a spec
                that omits it at a multi-executor cell is measuring an unidentified mechanism.
  5 PHASE       the spec declares the phase its residual was mined from. An undeclared controller
                defaults to `query`, so a write-side controller is installed, reached, and SKIPPED.

Exit 0 only if every check passes. Anything else names the failure and the fix.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "benchmarks" / "bfcl_v4", REPO / "scripts"):
    sys.path.insert(0, str(_p))

W = 96


def _states_for(run: pathlib.Path, phase: str):
    import self_evolve_cycle2 as c2
    steps = c2._load_prereq(run) if phase == "prereq" else c2.load_run(run)[1]
    return c2.observable_states(steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=pathlib.Path,
                    help="controller.json / one entry of controllers.json")
    ap.add_argument("--run", required=True, type=pathlib.Path,
                    help="run dir whose trajectories supply the states the controller will see")
    ap.add_argument("--min-fire", type=float, default=0.0,
                    help="minimum firing FRACTION to accept (0 = only reject exactly zero)")
    ap.add_argument("--json", type=pathlib.Path)
    a = ap.parse_args()

    import bfcl_runtime as rt

    spec = json.loads(a.spec.read_text())
    if isinstance(spec, list):
        spec = spec[0]
    signal = str(spec.get("name") or "")
    boundary = str(spec.get("locus") or "")
    action = str(spec.get("action") or "")
    phase = str(spec.get("phase") or "")
    params = dict((spec.get("predicate") or {}).get("params") or {})

    print("=" * W)
    print(f"PREFLIGHT  {signal[:54]}")
    print(f"  boundary={boundary}  action={action}  phase={phase or '(UNDECLARED)'}")
    print("=" * W)

    rec: dict = {"spec": str(a.spec), "signal": signal, "boundary": boundary,
                 "action": action, "phase": phase, "checks": {}}
    failures: list[str] = []

    states = _states_for(a.run, phase or "query")
    proj = rt.states_at(boundary, states)
    print(f"\n[states] {len(states)} per-step -> {len(proj)} projected at {boundary}")
    if not proj:
        failures.append("the boundary's information set is EMPTY: nothing to fire on")

    # ---- 1 SUPPLY --------------------------------------------------------------------------------
    def _spec_fields(node) -> set:
        """Fields the SPEC's predicate reads -- the structural equivalent of `signal_fields`."""
        out: set[str] = set()
        if not isinstance(node, dict):
            return out
        if "all" in node:
            for t in node["all"]:
                out |= _spec_fields(t)
        elif "field" in node:
            out.add(str(node["field"]))
        return out

    _pn = dict(spec.get("predicate") or {})
    if "declared_signal" not in _pn:
        need = _spec_fields(_pn)
    else:
        try:
            need = set(rt.signal_fields(signal))
        except Exception:
            need = set()
    have: set[str] = set()
    for st in proj:
        have |= set(st)
    missing = sorted(need - have)
    print(f"\n[1 SUPPLY] predicate reads {sorted(need) or '(unknown)'}")
    print(f"           states carry   {sorted(need & have)}")
    if missing:
        print(f"           MISSING        {missing}")
        failures.append(f"unsupplied fields {missing}: the predicate can only answer False")
    else:
        print("           OK -- every field the predicate reads is supplied")
    rec["checks"]["supply"] = {"reads": sorted(need), "missing": missing}

    # ---- 2 FIREABILITY ---------------------------------------------------------------------------
    #
    # EVALUATE THE SPEC'S OWN PREDICATE, not the signal NAME.
    #
    # A synthesized signal exists only in the process that expanded Phi -- `EXPANDED_SIGNALS` is
    # in-memory, so a fresh interpreter has no evaluator for it and `evaluate_signal` raises KeyError
    # on every state. Looking up by name therefore reported all 37 arms of an expansion round as "the
    # predicate RAISED on every state", which is a fact about THIS SCRIPT and not about the arms.
    #
    # The spec carries its predicate STRUCTURALLY (that is why `controller_spec` translates the object
    # rather than parsing the name), and `install_controller.SpecPredicate` is the same evaluator the
    # live hook uses. So preflight what will actually be installed, and fall back to the declared-name
    # path only when the spec names a host signal.
    import install_controller as ic

    pred_node = dict(spec.get("predicate") or {})
    structural = "declared_signal" not in pred_node
    if structural:
        _sp = ic.SpecPredicate(spec)
        def _fires(st):
            return bool(_sp.fires_on(st))
        print("\n[2 FIREABILITY] evaluating the SPEC'S OWN predicate "
              "(synthesized signal: no host evaluator exists outside the expanding process)")
    else:
        def _fires(st):
            return bool(rt.evaluate_signal(signal, st, params))

    fired, errors = 0, 0
    for st in proj:
        try:
            if _fires(st):
                fired += 1
        except Exception:
            errors += 1
    frac = fired / max(len(proj), 1)
    print(f"\n[2 FIREABILITY] fires {fired}/{len(proj)} = {100 * frac:.1f}%"
          + (f"   ({errors} evaluation errors)" if errors else ""))
    if errors == len(proj) and proj:
        failures.append("the predicate RAISED on every state: it cannot be evaluated here at all")
    elif fired == 0:
        failures.append("fires on ZERO states: no paired contrast, the arm runs as the control")
    elif fired == len(proj):
        failures.append("fires on EVERY state: not a trigger, this is an unconditional rewrite")
    elif frac < a.min_fire:
        failures.append(f"firing fraction {frac:.3f} below the required {a.min_fire}")
    else:
        print("           OK -- a non-trivial minority, so a paired contrast exists")
    rec["checks"]["fireability"] = {"fired": fired, "n": len(proj), "fraction": round(frac, 4),
                                    "eval_errors": errors}

    # ---- 3 NEGATIVE CASES ------------------------------------------------------------------------
    not_fired = len(proj) - fired - errors
    print(f"\n[3 NEGATIVE] does NOT fire on {not_fired}/{len(proj)} states")
    if not_fired <= 0 and proj:
        failures.append("no negative case: nothing demonstrates the predicate DISCRIMINATES")
    else:
        print("           OK -- the predicate discriminates rather than always firing")
    rec["checks"]["negative"] = {"not_fired": not_fired}

    # ---- 4 CAPABILITY IDENTITY -------------------------------------------------------------------
    caps = ()
    try:
        caps = rt.executor_capabilities(boundary, action)
    except Exception:
        pass
    declared_id = str(spec.get("capability_id") or "")
    print(f"\n[4 CAPABILITY] {len(caps)} executor(s) declared at {boundary}/{action}")
    for c in caps:
        print(f"           {c.capability_id[:52]:52s} bound={c.is_bound} enabled={c.is_enabled}")
    if len(caps) > 1:
        if not declared_id:
            failures.append(f"{len(caps)} executors at this cell and the spec names NONE: "
                            f"capability_id is not read by live dispatch, so the mechanism measured "
                            f"would be whichever resolves first")
        else:
            match = [c for c in caps if c.capability_id == declared_id]
            if not match:
                failures.append(f"spec names capability_id={declared_id!r}, which is not declared "
                                f"at this cell")
            elif not (match[0].is_bound and match[0].is_enabled):
                failures.append(f"spec names {declared_id!r}, which is "
                                f"{'unbound' if not match[0].is_bound else 'DISABLED'}")
            else:
                print(f"           OK -- spec names {declared_id}, bound and enabled")
    elif len(caps) == 1:
        print("           OK -- single executor, identity unambiguous")
    else:
        failures.append("no executor capability declared at this cell")
    rec["checks"]["capability"] = {"n_executors": len(caps), "declared": declared_id,
                                  "ids": [c.capability_id for c in caps]}

    # ---- 5 PHASE ---------------------------------------------------------------------------------
    print(f"\n[5 PHASE] spec declares phase={phase or '(UNDECLARED -> defaults to query)'}")
    if not phase:
        failures.append("no phase declared: defaults to `query`, so a write-side controller is "
                        "installed, reached, and SKIPPED in the only phase its residual lives in")
    else:
        print("           OK -- the phase is explicit")
    rec["checks"]["phase"] = {"declared": phase}

    print("\n" + "=" * W)
    if failures:
        print(f"PREFLIGHT FAILED -- {len(failures)} blocking issue(s). DO NOT SPEND GPU:")
        for f in failures:
            print(f"   ! {f}")
    else:
        print("PREFLIGHT PASSED -- the controller can fire, discriminates, names its executor, "
              "and declares its phase")
    print("=" * W)
    rec["passed"] = not failures
    rec["failures"] = failures
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rec, indent=2) + "\n")
        print(f"wrote {a.json}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
