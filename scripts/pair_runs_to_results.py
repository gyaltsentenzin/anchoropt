#!/usr/bin/env python3
"""Turn measured run directories into the `--results` manifest core's selection consumes.

    python scripts/pair_runs_to_results.py --control results/cg_r1/cgctl \
        --arm cg272=results/cg_r1/cg272 --arm cg212=results/cg_r1/cg212 \
        --manifest results/cg_r1/arm_manifest.json --out results/cg_r1/results.json

WHY THIS EXISTS AND WHAT IT REFUSES TO DO. `self_evolve_cycle2.py --results` wants paired outcomes
keyed by `arm_label` exactly as `arm_manifest.json` emits them. The runs on the cluster are keyed by
a short tag. That mapping is the only thing this script owns.

It does NOT score, rank, or choose. Four rules it enforces instead:

  * `gains`/`losses` are CASE IDs, not counts, so the denominator is auditable and no case is
    counted twice.
  * A DENOMINATOR MISMATCH is recorded as `denominator_ok=False`, not silently intersected. Core
    routes such an arm to UNEVALUATED rather than scoring it.
  * FIRINGS come from the run's own trajectory sidecar -- a truthy `*_gate` key -- never from a
    policy flag. A registry-derived flag cannot see whether the mechanism ran, and this project has
    been bitten by that four times.
  * An arm whose run is MISSING or incomplete is OMITTED from the manifest entirely. It is then
    UNEVALUATED, which is the honest class. Emitting a zero for it would let a fabricated null
    outrank a genuinely measured loss under J_train.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def load_scored(run: pathlib.Path) -> dict[str, bool]:
    """case_id -> correct, for the SCORED (non-prereq) episodes of one run."""
    hits = sorted(run.glob("eval_*result*.json")) or sorted(run.glob("*result*.json"))
    if not hits:
        return {}
    payload = json.load(open(hits[0]))
    rows = payload.get("results") if isinstance(payload, dict) else payload
    out = {}
    for r in rows or ():
        if r.get("is_prereq"):
            continue
        out[str(r["id"])] = bool(r.get("valid"))
    return out


def firing_telemetry(run: pathlib.Path) -> tuple[int, int]:
    """(episodes that fired, interventions executed) from the sidecar. The ONLY firing source."""
    traj = run / "traj"
    eps, n = set(), 0
    if not traj.is_dir():
        return 0, 0
    for phase in sorted(p for p in traj.iterdir() if p.is_dir()):
        for fp in sorted(phase.glob("*.json")):
            try:
                ep = json.load(open(fp))
            except (ValueError, OSError):
                continue
            fired = False
            for s in ep.get("steps") or ():
                for k, v in s.items():
                    # The sidecar's own convention: a truthy key ending `_gate` is a firing.
                    if k.endswith("_gate") and v is True:
                        n += 1
                        fired = True
            if fired:
                eps.add(f"{phase.name}/{ep.get('case_id')}")
    return len(eps), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, type=pathlib.Path)
    ap.add_argument("--arm", action="append", default=[],
                    help="TAG=path/to/run, repeatable")
    ap.add_argument("--manifest", required=True, type=pathlib.Path,
                    help="arm_manifest.json, to resolve each TAG to its full arm_label")
    ap.add_argument("--map", type=pathlib.Path,
                    help="JSON {signal: tag}; defaults to tags.json beside the manifest")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()

    ctl = load_scored(a.control)
    if not ctl:
        print(f"FATAL: no scored results in the control run {a.control}", file=sys.stderr)
        return 2
    ctl_fired, ctl_iv = firing_telemetry(a.control)
    print(f"control {a.control.name}: {sum(ctl.values())}/{len(ctl)} "
          f"({100*sum(ctl.values())/len(ctl):.2f}%)  fired_eps={ctl_fired} interventions={ctl_iv}")

    manifest = json.load(open(a.manifest))
    tagmap = json.load(open(a.map)) if a.map else {}
    # KEY -> arm_label, from the manifest core emitted. Never reconstructed by string surgery.
    #
    # Keyed on SIGNAL alone this collapsed two arms that share a signal and differ only in VARIANT --
    # exactly the read-side pair (search_other_container vs verify_before_answering), which would have
    # reported one measurement for two distinct interventions. So a map value may name either the
    # signal or the full arm_label, and the label wins when both could match.
    label_by_key = {}
    for r in manifest["arms"]:
        label_by_key[r["arm_label"]] = r["arm_label"]
        label_by_key.setdefault(r["signal"], r["arm_label"])
        v = r.get("variant")
        if v:
            label_by_key.setdefault(f"{r['signal']}:{v}", r["arm_label"])
    # A signal shared by several arms is AMBIGUOUS on its own: drop the bare-signal shortcut for it
    # rather than silently picking the first, which is how two arms become one measurement.
    import collections as _c
    _shared = {sig for sig, n in _c.Counter(r["signal"] for r in manifest["arms"]).items() if n > 1}
    for sig in _shared:
        label_by_key.pop(sig, None)

    rows, omitted = [], []
    for spec in a.arm:
        tag, _, path = spec.partition("=")
        run = pathlib.Path(path)
        arm = load_scored(run)
        if not arm:
            omitted.append((tag, "no scored results -- run missing or incomplete"))
            continue
        key = next((s for s, t in tagmap.items() if t == tag), None)
        label = label_by_key.get(key or "")
        if not label:
            hint = (" -- that signal is carried by several arms, so name the full arm_label or "
                    "'signal:variant' in the map") if (key in _shared) else ""
            omitted.append((tag, f"no unambiguous arm_label for {key!r}{hint}"))
            continue

        same = set(ctl) == set(arm)
        shared = sorted(set(ctl) & set(arm))
        gains = [c for c in shared if not ctl[c] and arm[c]]
        losses = [c for c in shared if ctl[c] and not arm[c]]
        fired, iv = firing_telemetry(run)
        delta = 100.0 * (sum(arm.values()) / len(arm) - sum(ctl.values()) / len(ctl))
        rows.append({
            "arm_label": label, "gains": gains, "losses": losses, "n": len(shared),
            "cases_fired": fired, "interventions_executed": iv,
            "accuracy_delta_pp": round(delta, 4), "denominator_ok": same,
            "incumbent_token": str(a.control),
            "_tag": tag, "_key": key,
            "_arm_correct": sum(arm.values()), "_arm_n": len(arm),
            "_ctl_correct": sum(ctl.values()), "_ctl_n": len(ctl),
        })
        flag = "" if same else "   ** DENOMINATOR MISMATCH -> denominator_ok=False **"
        print(f"  {tag:10s} {sum(arm.values()):3d}/{len(arm)}  "
              f"net {len(gains)-len(losses):+d} (+{len(gains)}/-{len(losses)})  "
              f"{delta:+.2f}pp  fired_eps={fired} interventions={iv}{flag}")

    for tag, why in omitted:
        # OMITTED, not zeroed. An arm with no result is UNEVALUATED.
        print(f"  {tag:10s} OMITTED -- {why}")

    json.dump({"incumbent_token": str(a.control), "results": rows}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}  ({len(rows)} measured, {len(omitted)} omitted as UNEVALUATED)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
