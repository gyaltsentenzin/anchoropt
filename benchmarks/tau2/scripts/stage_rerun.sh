#!/bin/bash
# Stage a re-run of the arms that produced no measurement, without destroying what did.
#
#   ./stage_rerun.sh                  # show the plan; move nothing
#   ./stage_rerun.sh --apply          # quarantine void data, leaving those arms ready to re-run
#   ./stage_rerun.sh --apply --all    # ALSO reset the 8 arms that have data (for a MAX_STEPS change)
#
# THREE CLASSES OF ARM, and they must not be treated alike:
#
#   VOID INCUMBENT   every baseline episode failed at the harness. There is no incumbent, so nothing
#                    in the arm is a result. The whole arm is quarantined and re-run from scratch.
#   VOID ROWS        the incumbent is clean but some recorded "measurements" were arms whose episodes
#                    did not run. The baseline is KEPT -- re-running it would move the incumbent every
#                    arm is compared against, which is the one thing a round may not do -- and only the
#                    void rows are dropped so `evaluate --resume` re-measures exactly those.
#   CLEAN            left alone.
#
# Quarantine, not deletion: void data is evidence about the outage. It moves to
# `_quarantine/<timestamp>/` so it can never be read back as a measurement but is still there.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ANCHOROPT_TAU2_MATRIX:?set ANCHOROPT_TAU2_MATRIX to your matrix output directory}"
PY="${ANCHOROPT_TAU2_REPO:?set ANCHOROPT_TAU2_REPO to your tau2-bench checkout}/.venv/bin/python"
[ -x "$PY" ] || PY=python3

APPLY=0; ALL=0
for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;;
    --all) ALL=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

[ -d "$ROOT" ] || { echo "no matrix at $ROOT"; exit 1; }
STAMP="$(date +%Y%m%dT%H%M%S)"
QUAR="$ROOT/_quarantine/$STAMP"

APPLY="$APPLY" ALL="$ALL" QUAR="$QUAR" "$PY" - "$ROOT" <<'STAGEPY'
import json, os, re, shutil, sys
from pathlib import Path

root = Path(sys.argv[1])
apply_ = os.environ.get("APPLY") == "1"
do_all = os.environ.get("ALL") == "1"
quar = Path(os.environ["QUAR"])
pat = re.compile(r"^arm_\d+\.json$")


def void_count(blob):
    return sum(1 for v in (blob.get("termination") or {}).values()
               if str(v).startswith("harness_error"))


plan = {"void_incumbent": [], "void_rows": [], "clean": [], "no_data": []}

for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")
                and p.name != "logs"):
    b = d / "baseline.json"
    if not b.exists():
        plan["no_data"].append((d, 0)); continue
    base = json.loads(b.read_text())
    if void_count(base["run"]):
        plan["void_incumbent"].append((d, void_count(base["run"]))); continue
    he = {}
    for f in sorted(x for x in d.glob("arm_*.json") if pat.match(x.name)):
        a = json.loads(f.read_text())
        he[a.get("label", "")] = void_count(a)
    rj = d / "results.json"
    rows = json.loads(rj.read_text()).get("results") or [] if rj.exists() else []
    bad = [r for r in rows if he.get(r["arm_label"], 0) > 0]
    (plan["void_rows"] if bad else plan["clean"]).append((d, len(bad)))

print(f"matrix: {root}")
print(f"mode  : {'APPLY' if apply_ else 'PLAN ONLY (pass --apply to act)'}"
      f"{'  +ALL (reset arms that have data too)' if do_all else ''}\n")

print(f"VOID INCUMBENT -- re-run from scratch ({len(plan['void_incumbent'])} arms)")
for d, n in plan["void_incumbent"]:
    print(f"   {d.name:50s} {n}/25 baseline episodes produced no outcome")
print(f"\nVOID ROWS -- keep the incumbent, re-measure the void arms ({len(plan['void_rows'])} arms)")
for d, n in plan["void_rows"]:
    print(f"   {d.name:50s} {n} recorded rows were never measured")
print(f"\nCLEAN ({len(plan['clean'])} arms)")
for d, _ in plan["clean"]:
    print(f"   {d.name}")
if plan["no_data"]:
    print(f"\nNO DATA ({len(plan['no_data'])} arms)")
    for d, _ in plan["no_data"]:
        print(f"   {d.name}")

if not apply_:
    print("\nnothing moved.")
    raise SystemExit(0)

quar.mkdir(parents=True, exist_ok=True)
moved = 0

# 1. void incumbent -> the whole arm directory is quarantined; the next submit re-runs it.
for d, _ in plan["void_incumbent"]:
    shutil.move(str(d), str(quar / d.name))
    moved += 1
    print(f"  quarantined whole arm: {d.name}")

# 2. void rows -> keep baseline + manifest, drop the void measurements and their run files.
for d, _ in plan["void_rows"]:
    he = {}
    for f in sorted(x for x in d.glob("arm_*.json") if pat.match(x.name)):
        a = json.loads(f.read_text())
        he[a.get("label", "")] = void_count(a)
    rj = d / "results.json"
    blob = json.loads(rj.read_text())
    rows = blob.get("results") or []
    keep = [r for r in rows if he.get(r["arm_label"], 0) == 0]
    drop = [r for r in rows if he.get(r["arm_label"], 0) > 0]
    tgt = quar / d.name
    tgt.mkdir(parents=True, exist_ok=True)
    (tgt / "results_voided_rows.json").write_text(json.dumps(drop, indent=2) + "\n")
    shutil.copy2(rj, tgt / "results.json.before")
    for f in sorted(x for x in d.glob("arm_*.json") if pat.match(x.name)):
        a = json.loads(f.read_text())
        if void_count(a):
            shutil.move(str(f), str(tgt / f.name))
    blob["results"] = keep
    blob["n_evaluated"] = len(keep)
    blob["restaged"] = {"at": os.environ["QUAR"], "dropped_void_rows": len(drop)}
    rj.write_text(json.dumps(blob, indent=2) + "\n")
    # selection.json was computed over the contaminated set; it must not survive. JOBID goes too, or
    # the status script reads LSF state from the FINISHED job and shows a re-staged arm as "DONE".
    for stale in ("selection.json", "PHASE", "JOBID"):
        f = d / stale
        if f.exists():
            shutil.move(str(f), str(tgt / stale))
    moved += 1
    print(f"  {d.name}: dropped {len(drop)} void row(s), kept {len(keep)}; baseline preserved")

# 3. --all: reset the clean arms too, for a change that invalidates their baselines.
if do_all:
    for d, _ in plan["clean"] + plan["void_rows"]:
        tgt = quar / d.name
        tgt.mkdir(parents=True, exist_ok=True)
        for name in ("baseline.json", "arm_manifest.json", "results.json", "selection.json", "PHASE"):
            f = d / name
            if f.exists():
                shutil.move(str(f), str(tgt / f"reset_{name}"))
        for f in sorted(x for x in d.glob("arm_*.json") if pat.match(x.name)):
            shutil.move(str(f), str(tgt / f.name))
        print(f"  reset (baseline invalidated): {d.name}")

print(f"\nquarantine: {quar}")
print(f"{moved} arm(s) staged for re-run.")
STAGEPY
