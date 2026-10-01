#!/usr/bin/env python3
"""SMOKE-TEST A NEW ADAPTER. Run this before spending any GPU on a port.

    python scripts/check_adapter.py adapters.my_benchmark
    python scripts/check_adapter.py adapters.my_benchmark --events my_traj.json

Seven checks, in dependency order, each answering a question that has cost this project a run when it
was assumed instead of measured:

    1 REQUIRED HOOKS PRESENT      -- a missing hook degrades silently to "nothing was possible here"
    2 A DECISION BOUNDARY EXISTS  -- a trajectory that yields none cannot be optimized at all
    3 BACKWARD SEARCH CAN MOVE    -- one boundary means WHERE is not a real coordinate for you
    4 A SIGNAL IS OBSERVABLE      -- and evaluable where it is declared, not merely declared
    5 AN ACTION CAN BE GROUNDED   -- an ungroundable action is indistinguishable from an absent one
    6 A CANDIDATE IS REALIZABLE   -- optimize_residual(evaluate=None) reaches a controller
    7 CORE STAYS GENERIC          -- your benchmark's identifiers did not leak into anchoropt/

Exit code 0 only if every check passes. A FAIL prints what to fix and where.

WITHOUT `--events` the checker synthesizes a two-step trajectory from your declared fields. That
exercises the plumbing but NOT your real trace shape, so it reports check 2/3 as SYNTHETIC. Pass real
trajectory rows once you have them -- that is the difference between "the interface is wired" and
"my benchmark works".
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

W = 96
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


def _say(state: str, check: str, detail: str = "") -> bool:
    mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[state]
    print(f"[{mark}] {check}")
    for line in (detail or "").splitlines():
        if line.strip():
            print(f"          {line}")
    return state != FAIL


def check_hooks(ad) -> bool:
    from adapters.adapter_template import OPTIONAL_HOOKS, REQUIRED_HOOKS
    missing = [h for h in REQUIRED_HOOKS if getattr(ad, h, None) is None]
    if missing:
        return _say(FAIL, "1. required hooks present",
                    f"missing: {missing}\nsee adapters/adapter_template.py for each one's contract")
    have_opt = [h for h in OPTIONAL_HOOKS if getattr(ad, h, None) is not None]
    grounders = [h for h in have_opt if h.startswith("ground_")]
    detail = f"all {len(REQUIRED_HOOKS)} required present; {len(have_opt)} optional"
    if not grounders:
        return _say(WARN, "1. required hooks present",
                    detail + "\nNO eta grounder -- checks 5 and 6 will fail, so no candidate can be "
                             "built. Add at least one ground_* for the actions your runtime supports.")
    return _say(PASS, "1. required hooks present", detail + f"; grounders: {grounders}")


def check_boundaries(ad, events) -> tuple[bool, list]:
    from anchoropt.learning.structured_search import localize
    bs = localize(events, runtime=ad)
    if not bs:
        return _say(FAIL, "2. a decision boundary exists",
                    "localize() derived NOTHING from these events.\n"
                    "is_decision() returned False for every event, or boundary_key() returned \"\".\n"
                    "Remember: COMMITTING TO A FINAL ANSWER is a decision, not only acting."), bs
    keys = [b.key for b in bs]
    return _say(PASS, "2. a decision boundary exists",
                f"{len(bs)} derived (earliest->latest): {keys}"), bs


def check_search_can_move(ad, bs) -> bool:
    from anchoropt.learning.boundary_search import backward_boundary_search
    if len(bs) < 2:
        return _say(WARN, "3. backward search can move",
                    f"only {len(bs)} boundary derived, so WHERE is not a searchable coordinate for "
                    f"this trajectory. Usually means is_decision() misses one event KIND -- e.g. the "
                    f"answer-commitment step, or the proposed-but-not-executed step.")
    hooks = {"repairable_at": lambda r, b: True, "expressible": lambda r, b: (True, ""),
             "build": lambda r, b, s: ([] if b is bs[-1] else ["c"]),
             "realizable": lambda c: (True, "p"), "evaluate": lambda c: None,
             "improves": lambda o: False, "promote": lambda c: None}
    res = backward_boundary_search("R", boundaries=bs, hooks=hooks)
    if res.moves_earlier < 1:
        return _say(FAIL, "3. backward search can move",
                    f"visited {list(res.visited)} but never moved earlier")
    return _say(PASS, "3. backward search can move",
                f"visited {list(res.visited)}, moves_earlier={res.moves_earlier}")


def check_signals(ad, states) -> bool:
    sigs = tuple(ad.declared_signals())
    if not sigs:
        return _say(WARN, "4. a declared signal is observable",
                    "Phi is EMPTY. Not fatal -- AnchorOpt will synthesize conditions over your "
                    "declared fields instead -- but the first search is cheaper with one or two.")
    ok, bad = [], []
    for s in sigs:
        at = ad.signal_boundaries(s)
        if isinstance(at, bool):
            bad.append(f"{s}: signal_boundaries returned a BOOL, not a set of IncisionPoint")
            continue
        if not at:
            bad.append(f"{s}: declared but observable NOWHERE")
            continue
        for p in at:
            st = dict(states[0]) if states else {}
            st["boundary"] = getattr(p, "value", p)
            try:
                ad.evaluate_signal(s, st, (ad.probe_params(s) if hasattr(ad, "probe_params") else {}))
                ok.append(f"{s}@{getattr(p, 'value', p)}")
            except Exception as exc:
                bad.append(f"{s}@{getattr(p, 'value', p)}: evaluate_signal raised "
                           f"{type(exc).__name__}: {exc}")
    if bad:
        return _say(FAIL, "4. a declared signal is observable",
                    "A signal declared observable at l MUST be evaluable at l:\n" + "\n".join(bad))
    return _say(PASS, "4. a declared signal is observable", f"evaluable: {ok[:6]}")


def check_grounding(ad, bs) -> bool:
    """Ask the ACTUAL contract machinery, with the signature it actually has.

    `instantiate` returns (arms, failure): a ContractFailure names the missing eta_mu, which is far
    more useful to a porter than "grounding returned nothing".
    """
    from anchoropt.anchor import Action
    from anchoropt.learning.action_contract import operators_of
    grounded, failures = [], []
    sigs = tuple(ad.declared_signals()) or ("probe_signal",)
    for b in bs:
        locus = ad.boundary_from_key(b.key)
        if locus is None:
            continue
        for act in sorted(ad.HOST.executable_actions(locus), key=lambda a: a.value):
            if act is Action.NOOP:
                continue
            for op in operators_of(act):
                for sig in sigs[:2]:
                    try:
                        from anchoropt.learning.action_contract import instantiate
                        arms, failure = instantiate(op, signal=sig, boundary=locus, runtime=ad)
                    except Exception as exc:
                        failures.append(f"{locus.value}/{op.value}: raised "
                                        f"{type(exc).__name__}: {exc}")
                        continue
                    if arms:
                        grounded.append(f"{locus.value}/{op.value}")
                    elif failure is not None:
                        failures.append(f"{locus.value}/{op.value}: "
                                        f"{getattr(failure, 'code', '')} "
                                        f"missing={list(getattr(failure, 'missing', ()) or ())}")
    if not grounded:
        return _say(FAIL, "5. an action can be grounded",
                    "No (boundary, action) could be parameterized:\n"
                    + "\n".join(sorted(set(failures))[:8])
                    + "\nDO NOT whitelist by signal NAME -- ground on the STRUCTURE of the boundary "
                      "and action, or no synthesized signal will ever compose with an action.")
    detail = f"grounded: {sorted(set(grounded))}"
    if failures:
        detail += f"\nnot grounded (expected if your runtime lacks these): " \
                  f"{sorted(set(failures))[:3]}"
    return _say(PASS, "5. an action can be grounded", detail)


def check_realizable(ad, events, states) -> bool:
    from anchoropt.learning.structured_search import optimize_residual

    class R:
        case_ids = ("smoke",)

    out = optimize_residual(R(), runtime=ad, host=ad.HOST, events=events, states=states)
    if not out.candidates:
        certs = {c.state for c in out.certificates}
        return _say(FAIL, "6. a candidate is realizable",
                    f"optimize_residual built NO controller. state={out.state}\n"
                    f"certificate states seen: {sorted(certs) or 'none'}\n"
                    f"Each certificate names why one (boundary, signal, action) was refused -- read "
                    f"out.certificates rather than guessing.")
    return _say(PASS, "6. a candidate is realizable",
                f"{len(out.candidates)} controller(s); state={out.state}; "
                f"visited={list(out.visited)}; expanded={out.expanded}")


def check_core_generic(module_name: str) -> bool:
    """Your benchmark's OWN identifiers must not appear in anchoropt/.

    Deliberately narrow. An earlier version of this check globbed every quoted string in the adapter
    and flagged ordinary English that core legitimately uses (`not_found`, `rejected`,
    `observable_state`) -- a guard that cries wolf teaches porters to ignore it. What actually breaks
    portability is core naming YOUR runtime: its module name, its host name, its tool and store names.
    So the candidates are the adapter's own proper nouns, and generic protocol words are excluded by
    construction rather than by a blocklist.
    """
    mod = sys.modules[module_name]
    src = pathlib.Path(mod.__file__).read_text()
    candidates: set[str] = set()
    # the adapter's own identity
    for pat in (r'^\s*name\s*=\s*"([A-Za-z0-9_]{3,})"', r'name="([A-Za-z0-9_]{3,})"'):
        candidates |= set(re.findall(pat, src, re.M))
    # the module's own leaf name, e.g. adapters.taubench -> taubench
    candidates.add(module_name.rsplit(".", 1)[-1])
    # anything that looks like a tool/store call the adapter names: foo_bar_baz with 2+ underscores
    candidates |= {n for n in re.findall(r'"([a-z][a-z0-9]*(?:_[a-z0-9]+){2,})"', src)}
    # PROTOCOL VOCABULARY, excluded by construction. Two kinds, and both matter:
    #   * the CONTRACT'S OWN HOOK NAMES -- core MUST name these, that is what a contract is. An earlier
    #     version flagged `boundary_from_key` in structured_search.py as a leak, which would have told
    #     a porter to remove the very call that makes their adapter reachable;
    #   * eta field names and locus values, which the action contract defines, not the benchmark.
    from adapters.adapter_template import OPTIONAL_HOOKS, REQUIRED_HOOKS
    protocol = set(REQUIRED_HOOKS) | set(OPTIONAL_HOOKS) | {
        "adapter_template", "post_generation_pre_exec", "post_execution", "pre_generation",
        "my_benchmark_host", "my_benchmark",
        "suppressed_operation", "retry_semantics", "argument_mapping", "target_surface",
        "last_error_kind", "proposes_action", "result_score", "step_index",
        "no_action_proposed", "lookup_failed", "cancel_proposed", "retry_with_hint",
        "reset_expanded_signals", "parameter_domains", "signal_boundaries", "declared_signals",
        "synthesis_fields", "evaluate_signal", "install_signal", "probe_params",
        "policy_class_for", "executor_supports", "commits_to_answer", "normalize_event",
        "observable_state", "boundary_from_key", "boundary_key", "is_decision", "label_for",
        "depends_on", "actions_in", "ground_reprompt", "ground_suppress", "ground_transforms",
        "ground_substitute_destinations",
    }
    suspect = sorted(c for c in candidates if c and c not in protocol and len(c) > 4)
    leaked = []
    for py in (REPO / "anchoropt").rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        text = py.read_text()
        for n in suspect:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", text):
                leaked.append(f"{py.relative_to(REPO)}: {n}")
    if leaked:
        return _say(FAIL, "7. core stays generic",
                    "your adapter's identifiers appear in core -- core must ask the adapter "
                    "instead:\n" + "\n".join(sorted(set(leaked))[:8]))
    return _say(PASS, "7. core stays generic",
                f"checked {len(suspect)} benchmark-specific identifier(s) against anchoropt/"
                + (f": {suspect[:6]}" if suspect else " (none declared yet)"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("module", help="import path of your adapter, e.g. adapters.my_benchmark")
    ap.add_argument("--events", type=pathlib.Path, default=None,
                    help="JSON list of REAL trajectory rows (strongly recommended)")
    a = ap.parse_args()

    mod = importlib.import_module(a.module)
    ad = getattr(mod, "ADAPTER", mod)

    print("=" * W)
    print(f"ADAPTER SMOKE TEST  ·  {a.module}")
    print("=" * W)

    synthetic = a.events is None
    if synthetic:
        # Minimal two-event trajectory: one that acted, one that committed. Exercises plumbing only.
        rows = [{"status": "executed", "tool_calls": ["probe()"], "tool_results": ["{}"], "step": 0},
                {"status": "final_answer", "tool_calls": [], "tool_results": [], "step": 1}]
    else:
        rows = json.load(open(a.events))
    norm = getattr(ad, "normalize_event", None)
    events = [norm(r) for r in rows] if callable(norm) else list(rows)

    obs = getattr(ad, "observable_state", None)
    if callable(obs):
        base = [obs(e) for e in events]
        states = [dict(base[i % len(base)], step_index=i) for i in range(12)] if base else []
    else:
        states = list(events)

    results = [check_hooks(ad)]
    ok2, bs = check_boundaries(ad, events)
    results.append(ok2)
    if bs:
        results.append(check_search_can_move(ad, bs))
        results.append(check_signals(ad, states))
        results.append(check_grounding(ad, bs))
        results.append(check_realizable(ad, events, states))
    results.append(check_core_generic(a.module))

    print("=" * W)
    if synthetic:
        print("NOTE checks 2-3 ran on a SYNTHETIC trajectory: plumbing only, not your trace shape.")
        print("     Re-run with --events <real rows>.json before trusting this.")
    bad = results.count(False)
    print(f"{'ALL CHECKS PASSED' if not bad else f'{bad} CHECK(S) FAILED'}")
    print("=" * W)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
