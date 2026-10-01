#!/usr/bin/env python3
"""ONE PAPER TABLE ROW per arm, from any benchmark's paired run. Benchmark-agnostic.

    python scripts/result_summary.py --control results/x/ctl --arm results/x/arm1 results/x/arm2
    python scripts/result_summary.py --control ... --arm ... --exclude-prefix customer --tsv

Emits exactly the columns a results table needs, and refuses to emit a delta it cannot defend:

    baseline accuracy | optimized accuracy | delta pp | gains/losses | firings
    gains on a fired case | denominator integrity | meta-model

WHY EACH COLUMN IS HERE, rather than "it looked useful":

  * DENOMINATOR INTEGRITY -- a delta computed over two different case sets is not a paired result.
    Reported per arm, never assumed, and a MISMATCH is printed instead of a number.
  * FIRINGS, read from per-episode telemetry -- a controller that was CONFIGURED but never fired is
    indistinguishable from one that worked, if you only look at accuracy. This has happened here.
  * GAINS ON A FIRED CASE -- the arithmetic that has exposed three separate wiring bugs in this
    project. If the gains are not on the cases the controller touched, the delta has another cause.
  * META-MODEL -- which model did the attribution/proposal. A table row that cannot say this cannot be
    compared against a row produced with a different one.

Trajectory and telemetry shapes are read generically: any `*_gate` key that is truthy counts as a
firing, which is the convention the sidecar keeps. Nothing here knows any benchmark's field names.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

W = 128


def load_run(run: pathlib.Path) -> tuple[dict, dict]:
    """(correct_by_case, steps_by_case). Accepts <run>/ or <run>/run/, flat or nested traj dirs."""
    res = None
    for base in (run, run / "run"):
        if not base.exists():
            continue
        res = next(iter(sorted(base.glob("eval_*result*.json"))), None) or \
            next(iter(sorted(base.glob("*result*.json"))), None)
        if res is not None:
            break
    if res is None:
        raise SystemExit(f"no results JSON under {run} (or {run}/run)")
    payload = json.load(open(res))
    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    # `is_prereq` is one benchmark's word for a setup episode; absent elsewhere and simply ignored.
    correct = {str(r.get("id")): bool(r.get("valid", r.get("correct")))
               for r in rows if not r.get("is_prereq")}
    steps: dict[str, list] = {}
    for sub in (run / "traj" / "query", run / "run" / "traj" / "query",
                run / "traj", run / "run" / "traj"):
        if not sub.exists():
            continue
        for fp in sorted(sub.glob("*.json")):
            try:
                ep = json.load(open(fp))
            except Exception:
                continue
            cid = str(ep.get("case_id") or ep.get("id") or fp.stem)
            steps.setdefault(cid, ep.get("steps") or [])
        if steps:
            break
    return correct, steps


# VISITING A HOOK IS NOT FIRING. The sidecar records both, and conflating them overcounts badly: on a
# real round this reported 65/65 cases "fired" where the truth was 38-49, because `*_hook_gate` marks
# that the hook was REACHED. It also makes "every gain sits on a fired case" vacuously true, which
# destroys the one check that has caught three wiring bugs here. So visited/skipped/error markers are
# excluded by name-shape, and a run that records only visits reports ZERO firings -- which is the
# honest answer, and loud enough to fix.
_NOT_A_FIRING = ("_hook_gate", "_skipped", "_error_gate", "_source_gate", "_visited")


def is_firing_key(key: str, value) -> bool:
    """Generic: a truthy `*_gate` key that is not a visit/skip/error marker."""
    if not (key.endswith("_gate") and value):
        return False
    return not any(marker in key for marker in _NOT_A_FIRING)


def fired_cases(steps_by_case: dict) -> set[str]:
    """Cases where a controller/gate actually FIRED, per the sidecar convention."""
    out = set()
    for cid, steps in steps_by_case.items():
        for s in steps:
            if isinstance(s, dict) and any(is_firing_key(k, v) for k, v in s.items()):
                out.add(cid)
                break
    return out


def exposed_cases(run: pathlib.Path, scored: set) -> tuple[set, str]:
    """Which scored cases were EXPOSED to the intervention -- by DEPENDENCY, not by co-location.

    EXPOSURE, NOT CAUSATION. A dependency link means the controller acted somewhere upstream of this
    case, so the case COULD have been affected. It does not establish that this particular gain was
    caused by the intervention -- per-case causation would need a counterfactual this design does not
    run. "9/9 exposed" is the honest claim; "9/9 attributed" overstates it.

    "Did this scored case fire?" is the wrong question for an UPSTREAM intervention. A controller that
    acts while state is being BUILT never fires during the scored episode at all -- by design -- so a
    naive on-fired count reports 0/N and reads as unattributable.

    Measured instance: an upstream storage controller showed 7/9 gains on fired cases, which looked
    like 2 gains with another cause. By dependency it is 9/9 -- every gain depends on a setup episode
    the controller fired in. The 2 were an artefact of the question, not of the effect.

    So: if a setup phase exists and fired, attribution follows the dependency graph. Otherwise it falls
    back to same-episode firing. The returned string says which was used, because the two answer
    different questions and a reader must know which one they are given.
    """
    setup = {}
    for sub in (run / "traj" / "prereq", run / "run" / "traj" / "prereq"):
        if not sub.exists():
            continue
        for fp in sorted(sub.glob("*.json")):
            try:
                ep = json.load(open(fp))
            except Exception:
                continue
            setup[str(ep.get("case_id"))] = ep.get("steps") or []
    fired_setup = fired_cases(setup)
    if not fired_setup:
        return set(), "same-episode firing"

    # The dependency graph is the ADAPTER's; read it from the shard the run used if present.
    shard = None
    for cand in (run / "cases.json", run.parent / "cases.json",
                 pathlib.Path("/tmp/shard_vector_train_cases.json")):
        if cand.exists():
            try:
                shard = {str(c["id"]): c for c in json.load(open(cand))}
                break
            except Exception:
                shard = None
    if not shard:
        return set(), "same-episode firing (no dependency graph available)"

    def deps(cid, seen=None):
        seen = seen if seen is not None else set()
        for d in (shard.get(cid, {}).get("depends_on") or []):
            d = str(d)
            if d in seen:
                continue
            seen.add(d)
            deps(d, seen)
        return seen

    return {c for c in scored if deps(c) & fired_setup}, "setup-phase dependency"


def meta_model_of(run: pathlib.Path) -> str:
    """The meta-model recorded with this run, if anything recorded one."""
    for name in ("cycle2.json", "recovery.json", "round.json", "selection.json", "meta.json"):
        for cand in (run / name, run.parent / name):
            if not cand.exists():
                continue
            try:
                blob = json.load(open(cand))
            except Exception:
                continue
            for key in ("meta_model", "meta_provider", "model"):
                got = blob.get(key) if isinstance(blob, dict) else None
                if got:
                    return str(got)
    try:
        from integrations.claude_roles import meta_model as mm
        return str(mm.configured())
    except Exception:
        return "(not recorded)"


def summarize(control: pathlib.Path, arm: pathlib.Path, *, exclude_prefix: str = "") -> dict:
    cc, _cs = load_run(control)
    ac, asteps = load_run(arm)
    common = sorted(set(cc) & set(ac))
    scored = [c for c in common if not (exclude_prefix and exclude_prefix in c)]
    fired = fired_cases(asteps) & set(scored)
    exposed, attr_basis = exposed_cases(arm, set(scored))
    if exposed:
        fired = exposed               # upstream intervention: EXPOSURE by dependency
    gains = [c for c in scored if not cc[c] and ac[c]]
    losses = [c for c in scored if cc[c] and not ac[c]]
    base, opt = sum(cc[c] for c in scored), sum(ac[c] for c in scored)
    n = max(1, len(scored))
    return {
        "arm": arm.name,
        "n": len(scored),
        "baseline": base,
        "optimized": opt,
        "baseline_acc": 100.0 * base / n,
        "optimized_acc": 100.0 * opt / n,
        "delta_pp": 100.0 * (opt - base) / n,
        "gains": len(gains),
        "losses": len(losses),
        "firings": len(fired),
        "gains_exposed": len([c for c in gains if c in fired]),
        "denominator_ok": set(cc) == set(ac),
        "excluded": len(common) - len(scored),
        "meta_model": meta_model_of(arm),
        "exposure_basis": attr_basis,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=pathlib.Path, required=True)
    ap.add_argument("--arm", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--exclude-prefix", default="",
                    help="drop cases whose id contains this (e.g. a contaminated cell). Applied to "
                         "control AND arm, so it cannot favour either.")
    ap.add_argument("--tsv", action="store_true", help="tab-separated, for pasting into a table")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()

    rows = [summarize(a.control, arm, exclude_prefix=a.exclude_prefix) for arm in a.arm]

    if a.tsv:
        cols = ("arm", "n", "baseline_acc", "optimized_acc", "delta_pp", "gains", "losses",
                "firings", "gains_on_fired", "denominator_ok", "meta_model")
        print("\t".join(cols))
        for r in rows:
            print("\t".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])
                            for c in cols))
    else:
        print("=" * W)
        print(f"PAIRED RESULT SUMMARY   control = {a.control.name}"
              + (f"   excluding ids containing {a.exclude_prefix!r}" if a.exclude_prefix else ""))
        print("=" * W)
        print(f"{'arm':12s} {'n':>4s} {'baseline':>9s} {'optimized':>10s} {'delta':>8s} "
              f"{'+/-':>8s} {'fired':>6s} {'exposed':>9s} {'denom':>6s}  meta-model")
        print("-" * W)
        for r in rows:
            denom = "OK" if r["denominator_ok"] else "MISMATCH"
            delta = ("%+.2f" % r["delta_pp"]) if r["denominator_ok"] else "  n/a"
            # Built with % rather than nested f-string quotes: the latter needs Python 3.12, and this
            # has to run on the 3.11 cluster env as well as locally.
            gl = "+%d/-%d" % (r["gains"], r["losses"])
            onf = "%d/%d" % (r["gains_exposed"], r["gains"])
            base = "%d/%d" % (r["baseline"], r["n"])
            opt = "%d/%d" % (r["optimized"], r["n"])
            print("%-12s %4d %9s %10s %8s %8s %6d %9s %6s  %s"
                  % (r["arm"][:12], r["n"], base, opt, delta, gl, r["firings"], onf, denom,
                     r["meta_model"]))
        print("-" * W)
        for r in rows:
            if not r["denominator_ok"]:
                print(f"  {r['arm']}: DENOMINATOR MISMATCH -- no delta reported. The two runs did "
                      f"not score the same cases, so this is not a paired comparison.")
            if r["firings"] == 0 and r["gains"]:
                print(f"  {r['arm']}: {r['gains']} gains with ZERO firings -- the delta cannot be "
                      f"caused by the controller. Check the telemetry channel is live.")
            elif r["gains"] and r["gains_exposed"] < r["gains"]:
                miss = r["gains"] - r["gains_exposed"]
                print("  %s: %d of %d gains are not attributable to the controller (basis: %s)."
                      % (r["arm"], miss, r["gains"], r.get("exposure_basis", "?")))
            if r["excluded"]:
                print(f"  {r['arm']}: {r['excluded']} case(s) excluded by --exclude-prefix.")
        print("=" * W)

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
