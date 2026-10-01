"""Score one round's arms against the incumbent-stack control, and prove which stack was installed.

WHAT THIS ADDS OVER CALLING THE DRIVER DIRECTLY
-----------------------------------------------
Two things the lifecycle needs and a single `--score` invocation does not give:

1. SEVERAL arms against ONE control. A round evaluates a budget of candidates against the same
   incumbent, and they must all be scored against the same control run or the comparisons are not
   commensurable. Running the driver once per arm does that correctly but reports each in isolation.

2. EXECUTION EVIDENCE, separated from library membership. "The controller is in the library" and "the
   controller ran in the evaluator" are different claims, and conflating them is how a stack becomes
   bookkeeping. So this reads each run's OWN sidecar for the firing keys of every controller the stack
   declares, and reports per-controller firing counts by phase. A stack member with zero firings
   anywhere is named, loudly: it is either inert or the arm silently became the control.

It does not re-implement scoring. Each arm goes through `self_evolve_cycle2.py --score`, which owns the
paired evaluation, the four criteria and the library write. This orchestrates and cross-checks.

Usage:
  python scripts/score_round.py --round R4 --results /tmp/r4 --control r4_inc_ctl \
      --arms r4_clear_arm r4_redun_arm --cell kv --library rounds/GOLDEN/candidates.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchoropt.learning.candidate_library import (            # noqa: E402
    ACCEPTED, CandidateLibrary, stack_fingerprint)


def gate_census(run: pathlib.Path) -> dict:
    """Every *_gate key that fired, by phase, read from the run's own trajectory sidecar.

    The sidecar is the only thing that identifies an arm: flags, fingerprints and filenames have all
    agreed and all been wrong on this project. A phase breakdown is required rather than a total,
    because a prereq-acting controller firing 0 times in query is CORRECT and a query-acting one doing
    the same is a dead arm -- one number cannot distinguish them.
    """
    out: dict[str, collections.Counter] = {}
    for phase in ("prereq", "query"):
        c: collections.Counter = collections.Counter()
        for sub in (run / "run" / "traj" / phase, run / "traj" / phase):
            if not sub.exists():
                continue
            for fp in sorted(sub.glob("*.json")):
                try:
                    ep = json.loads(fp.read_text())
                except Exception:
                    continue
                for step in (ep.get("steps") or []):
                    for k, v in (step or {}).items():
                        if k.endswith("_gate") and v:
                            c[k] += 1
        out[phase] = c
    return {p: dict(c) for p, c in out.items()}


def installed_names(run: pathlib.Path) -> list[str]:
    names = []
    for rel, key in (("controller_stack.json", "controllers"), ("controller_spec.json", None)):
        p = run / rel
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if key:
            for e in (d.get(key) if isinstance(d, dict) else d) or ():
                names.append(pathlib.Path(e).stem if isinstance(e, str)
                             else str(e.get("name") or "inline"))
        else:
            names.append(str(d.get("name") or "spec"))
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--results", type=pathlib.Path, required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--cell", default="kv")
    ap.add_argument("--split", default="train")
    ap.add_argument("--library", type=pathlib.Path, default=None)
    ap.add_argument("--dev", default=None,
                    help="tag SUFFIX appended to each arm tag to find its held-out run, e.g. "
                         "'_v4dev_test'. Convenient when the tags are parallel; use --dev-run when "
                         "they are not.")
    ap.add_argument("--dev-control", default=None,
                    help="suffix for the CONTROL's held-out run (defaults to --dev)")
    # EXPLICIT held-out paths, because derived ones silently resolve to nothing.
    #
    # `--dev` builds `<results>/<arm_tag><suffix>`, which only works while the train and held-out tags
    # are parallel. They stopped being parallel the moment a round was relaunched: the R6 train arm is
    # `r6d_a8full_arm` while its held-out run is `r6cho_a8full_arm_test`, because the train arm was
    # resubmitted after a vLLM checksum failure and the held-out one was not. A derived path then points
    # at a directory that does not exist, the dev block is skipped, and criterion 2 reports
    # PENDING_VALIDATION -- an arm BLOCKED for want of evidence that was sitting on disk.
    ap.add_argument("--dev-run", default=None,
                    help="explicit path to the ARM's held-out run (overrides --dev). Use when the "
                         "train and held-out tags are not parallel, e.g. after a relaunch.")
    ap.add_argument("--dev-control-run", default=None,
                    help="explicit path to the CONTROL's held-out run (overrides --dev-control)")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()

    ctl = a.results / f"{a.control}_{a.split}"
    if not ctl.exists():
        raise SystemExit(f"control run missing: {ctl}")
    lib_before = CandidateLibrary.load(a.library) if a.library else CandidateLibrary()
    stack_before = lib_before.installed_stack(a.cell)

    print("=" * 96)
    print(f"ROUND {a.round}  cell={a.cell}  control={a.control}")
    print(f"  incumbent stack BEFORE: {len(stack_before)} "
          f"[{stack_fingerprint(stack_before)}]")
    for k in stack_before:
        print(f"    {k}")
    print("=" * 96)

    # ---- EXECUTION EVIDENCE, before any score is quoted -------------------------------------------
    print("\n[A] DID THE STACK ACTUALLY EXECUTE? (per-run gate census from the sidecar)")
    census = {}
    for tag in [a.control, *a.arms]:
        run = a.results / f"{tag}_{a.split}"
        if not run.exists():
            print(f"  {tag:18s} RUN MISSING at {run}")
            continue
        g = gate_census(run)
        names = installed_names(run)
        census[tag] = {"installed": names, "gates": g}
        tot = sum(sum(c.values()) for c in g.values())
        print(f"  {tag:18s} installed={names}")
        for phase, c in g.items():
            if c:
                top = ", ".join(f"{k}={v}" for k, v in sorted(c.items(), key=lambda x: -x[1])[:5])
                print(f"    {phase:7s} {top}")
        # A run that installed NOTHING is a bare control, and zero firings is the correct
        # observation for it -- warning there would cry wolf on every round. The warning is for a run
        # that DID install something and still never fired: that one is inert, or silently became the
        # control, and its delta is not evidence about the controller.
        if tot == 0 and names:
            print(f"    *** ZERO FIRINGS ANYWHERE despite installing {names} -- this arm is inert "
                  f"or silently became the control. Do NOT read its delta as evidence about the "
                  f"controller.")
            census[tag]["inert"] = True
        elif tot == 0:
            print(f"    (no controller installed, no firings -- as expected for a bare control)")

    # ---- SCORE each arm against the SAME control --------------------------------------------------
    results = {}
    for tag in a.arms:
        run = a.results / f"{tag}_{a.split}"
        if not run.exists():
            continue
        out_dir = (a.out or (ROOT / "rounds" / "AUTONOMY" / a.round)) / tag
        cmd = [sys.executable, str(ROOT / "scripts" / "self_evolve_cycle2.py"),
               "--incumbent", str(ctl), "--out", str(out_dir), "--cell", a.cell,
               "--score", str(run), "--baseline", str(ctl)]
        if a.library:
            cmd += ["--library", str(a.library)]
        dev_arm = pathlib.Path(a.dev_run) if a.dev_run else (
            a.results / f"{tag}{a.dev}" if a.dev else None)
        dev_ctl = pathlib.Path(a.dev_control_run) if a.dev_control_run else (
            a.results / f"{a.control}{a.dev_control or a.dev}" if a.dev else None)
        if dev_arm is not None and dev_ctl is not None:
            if dev_arm.exists() and dev_ctl.exists():
                cmd += ["--dev", str(dev_arm), "--dev-baseline", str(dev_ctl),
                        "--dev-cell", a.cell]
            else:
                # SAID OUT LOUD. A missing held-out run makes criterion 2 PENDING, which BLOCKS
                # acceptance -- so silently skipping it turns a path typo into "not accepted" and
                # invites the wrong conclusion about the controller.
                for label, pth in (("arm", dev_arm), ("control", dev_ctl)):
                    if not pth.exists():
                        print(f"   HELD-OUT {label} run NOT FOUND at {pth} -- criterion 2 will be "
                              f"PENDING and the arm cannot install. Pass --dev-run / "
                              f"--dev-control-run if the tags are not parallel.")
        print(f"\n[B] SCORING {tag}")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        for line in r.stdout.splitlines():
            if any(m in line for m in ("PAIRED EVALUATION", "control ", "DENOMINATOR", "ACCEPTANCE",
                                       "  C1 ", "  C2 ", "  C3 ", "  C4 ", "PERSISTED",
                                       "HELD-OUT", "mechanism evidence", "NOT INSTALLED")):
                print("   " + line.strip())
        if r.returncode != 0:
            print(f"   DRIVER FAILED rc={r.returncode}\n{r.stderr[-800:]}")
            continue
        rec_p = out_dir / "cycle2.json"
        if rec_p.exists():
            rec = json.loads(rec_p.read_text())
            results[tag] = {"evaluation": rec.get("evaluation"),
                            "acceptance": {c["number"]: c["verdict"]
                                           for c in (rec.get("acceptance") or {}).get("criteria", [])},
                            "installed": (rec.get("acceptance") or {}).get("installed"),
                            "stack_fingerprint": rec.get("stack_fingerprint")}

    # ---- WHAT MOVED --------------------------------------------------------------------------------
    lib_after = CandidateLibrary.load(a.library) if a.library else CandidateLibrary()
    stack_after = lib_after.installed_stack(a.cell)
    print("\n" + "=" * 96)
    print(f"ROUND {a.round} SUMMARY")
    print(f"{'arm':20s} {'n':>4s} {'ctl':>4s} {'arm':>4s} {'net':>5s}  criteria            installed")
    for tag, r in results.items():
        ev = r["evaluation"] or {}
        cr = "".join(f"{v[0]}" for _, v in sorted((r["acceptance"] or {}).items()))
        print(f"{tag:20s} {ev.get('n', 0):>4} {ev.get('control', 0):>4} {ev.get('arm', 0):>4} "
              f"{ev.get('net', 0):>+5}  {cr:<19s} {r['installed']}")
    print(f"\nincumbent stack AFTER: {len(stack_after)} [{stack_fingerprint(stack_after)}]")
    moved = stack_fingerprint(stack_after) != stack_fingerprint(stack_before)
    print(f"INCUMBENT MOVED: {moved}")
    if moved:
        for k in sorted(set(stack_after) - set(stack_before)):
            print(f"  + {k}")

    out = {"round": a.round, "cell": a.cell, "control": a.control,
           "stack_before": list(stack_before), "fingerprint_before": stack_fingerprint(stack_before),
           "stack_after": list(stack_after), "fingerprint_after": stack_fingerprint(stack_after),
           "incumbent_moved": moved, "execution_evidence": census, "results": results}
    dest = (a.out or (ROOT / "rounds" / "AUTONOMY" / a.round)) / "ROUND_RESULT.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
