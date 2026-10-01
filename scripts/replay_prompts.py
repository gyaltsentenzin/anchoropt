#!/usr/bin/env python3
"""Replay the SAME residuals under two attributor prompts, and compare what changes downstream.

    python scripts/replay_prompts.py --incumbent /tmp/c2/c2inc --phase prereq \
        --prompts prompts/attributor_P0.txt prompts/attributor_P1.txt --out rounds/PROMPT_AB

A prompt revision is a change to the EVIDENCE CHANNEL, so it is measured the way any other change to
that channel is: same input, same everything downstream, one thing different. The comparison reports
what a prompt can actually be held responsible for --

    diagnoses that survive the schema guard        (a prompt that oversteps loses cases)
    families, before and after observable pooling  (prose-keyed grouping is prompt-sensitive)
    uncertainty admitted                           (null causal step: honest, and newly permitted)
    numbers quoted from the input                  (evidence, now explicitly allowed)
    the candidate the frozen search then builds    (the only thing that reaches a GPU)

-- and NOT accuracy, which no prompt change can be credited with on its own.

The prompts are read from FILES, never edited in place, so the as-run text of each round stays
recoverable. `client.ATTRIBUTOR_SYSTEM` is monkeypatched for the duration of one call and restored;
nothing in the frozen core is touched.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
sys.path.insert(0, str(REPO / "integrations" / "claude_roles"))

import bfcl_runtime as runtime                                              # noqa: E402
from anchoropt.learning.candidate_search import expressible_under           # noqa: E402
from anchoropt.learning.proposal_seams import (                             # noqa: E402
    AttributionSchemaError, ingest_attribution,
)
from anchoropt.learning.residual_problem import build_residual_problems     # noqa: E402
from anchoropt.learning.structured_search import optimize_residual          # noqa: E402

W = 100


def _driver():
    """The cycle-2 driver, imported for its evidence builders so both arms see identical input."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "c2", REPO / "scripts" / "self_evolve_cycle2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attribute_with(prompt_text: str, cases, *, cache: pathlib.Path | None):
    """One attribution pass under `prompt_text`. Cached by prompt+case digest so a rerun is free."""
    import client as roles
    if cache is not None and cache.exists():
        return json.load(open(cache)), "cached"
    original = roles.ATTRIBUTOR_SYSTEM
    try:
        roles.ATTRIBUTOR_SYSTEM = prompt_text
        raw, calls = roles.attribute_batched(cases)
        errs = [c.error for c in calls if c.error]
        if errs:
            print(f"    call errors: {errs[:2]}")
    finally:
        roles.ATTRIBUTOR_SYSTEM = original          # restore even on failure
    raw = list(raw)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump(raw, open(cache, "w"), indent=2, default=str)
    return raw, f"{len(calls)} calls"


_NUM = re.compile(r"(?<![A-Za-z0-9_])\d{2,}(?![A-Za-z0-9_])")


def describe(raw, label: str, *, mod) -> dict:
    """What a prompt can be held responsible for, measured rather than asserted."""
    kept, rejected = [], []
    for r in raw:
        try:
            kept.append(ingest_attribution(r, provider=f"attributor-{label}"))
        except AttributionSchemaError as exc:
            rejected.append(str(exc)[:120])

    problems = build_residual_problems(
        kept, expressible=lambda d: expressible_under(d, runtime=runtime)[0])
    pooled = mod.pool_by_observable(problems, runtime=runtime)

    # Uncertainty admitted: a null causal step is an honest answer P1 explicitly permits.
    null_step = sum(1 for r in raw
                    if isinstance(r.get("causal_region"), dict)
                    and r["causal_region"].get("step") is None)
    # Numbers QUOTED from the input are evidence. Counted in evidence only, where quoting belongs.
    quoted = sum(len(_NUM.findall(json.dumps(r.get("evidence", "")))) for r in raw)
    return {
        "label": label, "raw": len(raw), "usable": len(kept), "rejected": len(rejected),
        "rejections": rejected[:3],
        "families_prose": len(problems), "families_pooled": len(pooled),
        "top_support_prose": max((p.support for p in problems), default=0),
        "top_support_pooled": max((p.support for p in pooled), default=0),
        "null_causal_step": null_step, "numbers_quoted_in_evidence": quoted,
        "observables": [list(getattr(p, "observables", ())) for p in pooled[:3]],
        "top_key": (pooled[0].key[:150] if pooled else ""),
        "_kept": kept, "_pooled": pooled,
    }


def search_from(pooled, *, mod, steps, cell: str) -> dict:
    """Run the FROZEN search on the top pooled residual -- the only thing that reaches a GPU."""
    if not pooled:
        return {"state": "NO_RESIDUAL"}
    top = pooled[0]
    events = []
    for cid in sorted(top.case_ids):
        events.extend(mod.keep_steps(steps.get(cid, [])))
    states = mod.observable_states({c: steps.get(c, []) for c in steps})
    runtime.reset_expanded_signals()
    out = optimize_residual(top, runtime=runtime, host=runtime.HOST, events=events,
                            states=states, evaluate=lambda _a: 0, improves=lambda _o: False)
    arms = collections.Counter((a.boundary.value, a.action.value) for a in out.candidates)
    res = {"state": out.state, "boundaries": list(out.boundaries),
           "visited": list(out.visited), "moves_earlier": out.moves_earlier,
           "expanded": len(out.signals_installed), "candidates": len(out.candidates),
           "by_cell": {f"{b}/{a}": n for (b, a), n in arms.most_common()}}
    runtime.reset_expanded_signals()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", type=pathlib.Path, required=True)
    ap.add_argument("--prompts", type=pathlib.Path, nargs=2, required=True,
                    help="P0 then P1, as files so the as-run text stays recoverable")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--cell", default="vector")
    ap.add_argument("--phase", choices=("query", "prereq"), default="prereq")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--live", action="store_true", help="omit to use cached attributions only")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    mod = _driver()

    # IDENTICAL INPUT FOR BOTH ARMS. Built once, so nothing but the prompt differs.
    correct, steps = mod.load_run(a.incumbent)
    if a.phase == "prereq":
        steps = mod._load_prereq(a.incumbent)
        failed = sorted(c for c, st in steps.items()
                        if a.cell in c and mod._has_tool_error(st))
    else:
        failed = sorted(c for c, ok in correct.items() if not ok and a.cell in c)
    cases = [mod.case_facts(c, steps.get(c, []), a.cell, a.phase) for c in failed[:a.limit]]

    print("=" * W)
    print(f"ATTRIBUTOR PROMPT A/B  ·  {a.incumbent.name}  ·  phase={a.phase}  ·  {len(cases)} cases")
    print("  identical input to both arms; only the system prompt differs")
    print("=" * W)

    rows = []
    for path in a.prompts:
        label = path.stem.replace("attributor_", "")
        text = path.read_text()
        cache = a.out / f"diagnoses_{label}.json"
        print(f"\n[{label}] {path}  ({len(text)} chars)")
        if not a.live and not cache.exists():
            print("    no cache and --live not given -- skipped")
            continue
        raw, how = attribute_with(text, cases, cache=cache)
        print(f"    {how}: {len(raw)} raw records")
        d = describe(raw, label, mod=mod)
        d["search"] = search_from(d.pop("_pooled"), mod=mod, steps=steps, cell=a.cell)
        d.pop("_kept", None)
        rows.append(d)
        print(f"    usable {d['usable']}/{d['raw']}  rejected {d['rejected']}")
        print(f"    families {d['families_prose']} prose -> {d['families_pooled']} pooled"
              f"   top support {d['top_support_prose']} -> {d['top_support_pooled']}")
        print(f"    uncertainty admitted (null causal step): {d['null_causal_step']}")
        print(f"    numbers quoted in evidence: {d['numbers_quoted_in_evidence']}")
        print(f"    observables of the top residual: {d['observables'][:1]}")
        s = d["search"]
        print(f"    SEARCH -> {s.get('state')}  visited={s.get('visited')}  "
              f"moves_earlier={s.get('moves_earlier')}  expanded={s.get('expanded')}")
        print(f"             candidates by (boundary/action): {s.get('by_cell')}")

    if len(rows) == 2:
        p0, p1 = rows
        print("\n" + "=" * W)
        print(f"{'metric':38s} {'P0':>14s} {'P1':>14s}")
        print("-" * W)
        for k, name in (("usable", "usable diagnoses"), ("rejected", "schema rejections"),
                        ("families_prose", "families (prose-keyed)"),
                        ("families_pooled", "families (pooled by observable)"),
                        ("top_support_prose", "top support (prose)"),
                        ("top_support_pooled", "top support (pooled)"),
                        ("null_causal_step", "uncertainty admitted"),
                        ("numbers_quoted_in_evidence", "numbers quoted as evidence")):
            print(f"{name:38s} {str(p0[k]):>14s} {str(p1[k]):>14s}")
        for k, name in (("state", "search terminal state"),
                        ("moves_earlier", "moves_earlier"), ("expanded", "signals expanded"),
                        ("candidates", "candidates built")):
            print(f"{'search: ' + name:38s} {str(p0['search'].get(k)):>14s} "
                  f"{str(p1['search'].get(k)):>14s}")
        print("=" * W)
        print("NOT COMPARED HERE: accuracy. No prompt change can be credited with that on its own;")
        print("it takes a paired GPU run, and both arms would need one.")
    json.dump(rows, open(a.out / "prompt_ab.json", "w"), indent=2, default=str)
    print(f"wrote {a.out / 'prompt_ab.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
