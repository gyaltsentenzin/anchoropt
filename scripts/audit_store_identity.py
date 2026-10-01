"""Did any arm read another arm's constructed state? Answered from the runs' own provenance.

WHY A DETECTOR AND NOT A PATCH (for now)
----------------------------------------
`anchoropt/learning/state_cache_identity.py` exists and is correct, but wiring it into the LIVE BV
evaluator changes how state is keyed, which changes which runs are cache hits, which changes measured
execution semantics. Every result measured under the old keying would need re-measurement. That is a
versioned behavioural change, not a bug fix, and it must not be slipped in underneath results that are
already being collected.

So the immediate need is different: PROVE that the runs we have are uncontaminated. The blind key is
only dangerous when two arms with different controllers actually SHARE a store, and the harness gives
each arm its own cache directory, which is what has been saving us. This script checks that claim
against the artifacts instead of trusting it -- measured on the round-2 arms:

    base_key=0b94ed894bbddc41 for the incumbent control, the clear candidate AND the redundant-write
    candidate -- three different controller stacks, one identical key.

The key is blind. The DIRECTORIES are what differ. That is a real finding either way, and this script
is how it stays checked rather than remembered.

WHAT COUNTS AS CONTAMINATION
----------------------------
Two arms whose controller stacks DIFFER resolving to the same store INSTANCE (same cache dir + same
base_key + same variant). Two arms sharing a store DELIBERATELY (the matched-construction protocol) is
not contamination -- it is a declared design, and `state_cache_identity.collision_report` keeps the two
apart. A run that rebuilt (`verdict=MISS`) cannot have read anyone else's state at all.

Usage:
    python scripts/audit_store_identity.py --results /path/to/results --tags r4_inc_ctl r4_clear_arm
    python scripts/audit_store_identity.py --local tests/fixtures/r1_accept_pair
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchoropt.learning.state_cache_identity import (        # noqa: E402
    StateKey, StateKeyInputs, collision_report, state_cache_key)


def _provenance(run: pathlib.Path) -> dict:
    """The store provenance the harness wrote, from wherever it landed."""
    for rel in ("cache_provenance.json", "store_provenance.json"):
        for cand in (run / rel, run / "run" / rel):
            if cand.exists():
                try:
                    return json.loads(cand.read_text())
                except Exception:
                    return {}
    # A cache directory keeps it one or two levels down, under snap_*/v_*/.
    for cand in sorted(run.glob("snap_*/v_*/cache_provenance.json")):
        try:
            return json.loads(cand.read_text())
        except Exception:
            continue
    return {}


def _installed(run: pathlib.Path) -> tuple[str, str | None]:
    """(controller identity, declared phase) for what this run installed -- read from the run."""
    names: list[str] = []
    phase: str | None = None
    for rel in ("controller_stack.json", "controller_spec.json"):
        p = run / rel
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if rel == "controller_stack.json":
            ents = d.get("controllers") if isinstance(d, dict) else d
            for e in ents or ():
                names.append(pathlib.Path(e).stem if isinstance(e, str)
                             else str(e.get("name") or "inline"))
        else:
            names.append(str(d.get("name") or "spec"))
            phase = str(d.get("phase") or "") or phase
    return ("+".join(names) if names else "", phase)


def audit(runs: dict[str, pathlib.Path]) -> dict:
    keys: dict[str, StateKey] = {}
    rows = []
    for tag, run in runs.items():
        prov = _provenance(run)
        ident, phase = _installed(run)
        base = {"base_key": prov.get("base_key"), "variant": prov.get("variant_key"),
                "code_fp": prov.get("store_code_fingerprint"),
                "gate_cfg": prov.get("gate_config_fingerprint")}
        sk = state_cache_key(StateKeyInputs(base=base, controller=ident or None,
                                           controller_phase=phase))
        keys[tag] = sk
        rows.append({"tag": tag, "run": str(run), "controller": ident or "(control: none)",
                     "declared_phase": phase, "harness_base_key": prov.get("base_key"),
                     "harness_verdict": prov.get("verdict"),
                     "cache_dir": prov.get("cache") or prov.get("cache_dir"),
                     "would_be_key": sk.key, "covers_controller": sk.covers_controller})
    rep = collision_report(keys)

    # The harness's OWN key, independent of what core would have computed. This is the blind one.
    by_harness: dict[str, list[str]] = {}
    for r in rows:
        if r["harness_base_key"]:
            by_harness.setdefault(r["harness_base_key"], []).append(r["tag"])
    harness_shared = {k: v for k, v in by_harness.items() if len(v) > 1}

    # Contamination requires a shared store INSTANCE, not merely a shared key: different cache
    # directories mean different stores however equal the keys are.
    real = []
    for key, tags in harness_shared.items():
        dirs = {(next(r["cache_dir"] for r in rows if r["tag"] == t) or t) for t in tags}
        ctls = {next(r["controller"] for r in rows if r["tag"] == t) for t in tags}
        if len(dirs) == 1 and len(ctls) > 1:
            real.append({"harness_base_key": key, "tags": sorted(tags),
                         "shared_cache_dir": sorted(dirs)[0], "distinct_controllers": sorted(ctls)})
    rebuilt = [r["tag"] for r in rows if str(r["harness_verdict"]).upper() == "MISS"]
    # NO DATA IS NOT A CLEAN BILL. A first version of this script reported
    # "CLEAN -- no arm read another arm's state" over three arms whose provenance it had failed to
    # locate and whose controllers it had read as "(control: none)". Every input was empty and the
    # verdict was reassuring, which is the exact defect class this project keeps paying for. An audit
    # must be able to say "I could not tell".
    unreadable = [r["tag"] for r in rows if not r["harness_base_key"]]
    no_controller = [r["tag"] for r in rows if r["controller"] == "(control: none)"]
    undetermined = bool(unreadable) or len(no_controller) == len(rows)
    return {"rows": rows, "core_key_report": rep,
            "undetermined": undetermined,
            "arms_with_no_readable_provenance": sorted(unreadable),
            "arms_with_no_readable_controller": sorted(no_controller),
            "harness_keys_shared_by_multiple_arms": harness_shared,
            "harness_key_is_controller_blind": bool(
                harness_shared and any(
                    len({next(r["controller"] for r in rows if r["tag"] == t) for t in tags}) > 1
                    for tags in harness_shared.values())),
            "actual_shared_store_instances": real,
            "arms_that_rebuilt_their_own_store": sorted(rebuilt),
            "contaminated": bool(real),
            "verdict": (f"CONTAMINATED -- {len(real)} shared store instance(s) across differing "
                        f"controllers" if real else
                        f"UNDETERMINED -- provenance unreadable for {sorted(unreadable)}"
                        f"{' and no arm reported a controller' if len(no_controller) == len(rows) else ''}"
                        f"; the store may not have finished building. This is NOT a clean bill."
                        if undetermined else
                        "CLEAN -- no arm read another arm's state")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, default=None,
                    help="a results root holding <tag>_<split> run dirs and <tag>_cache dirs")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--split", default="train")
    ap.add_argument("--local", type=pathlib.Path, default=None,
                    help="a directory whose immediate children are run dirs")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    runs: dict[str, pathlib.Path] = {}
    if a.local:
        for d in sorted(a.local.iterdir()):
            if d.is_dir():
                runs[d.name] = d
    for t in a.tags:
        base = a.results or pathlib.Path(".")
        run = base / f"{t}_{a.split}"
        cache = base / f"{t}_cache"
        runs[t] = run if run.exists() else cache
    if not runs:
        raise SystemExit("nothing to audit: pass --tags with --results, or --local")

    out = audit(runs)
    print(f"{'tag':22s} {'controller':34s} {'phase':8s} {'verdict':6s} harness_base_key")
    print("-" * 100)
    for r in out["rows"]:
        print(f"{r['tag']:22s} {str(r['controller'])[:34]:34s} "
              f"{str(r['declared_phase'] or '-'):8s} {str(r['harness_verdict'] or '-'):6s} "
              f"{r['harness_base_key']}")
    print()
    if out["harness_key_is_controller_blind"]:
        print("FINDING: the harness base_key is CONTROLLER-BLIND -- arms running different "
              "controllers computed the same key. Only the per-arm cache DIRECTORY separates them.")
    for key, tags in out["harness_keys_shared_by_multiple_arms"].items():
        print(f"  {key} <- {sorted(tags)}")
    print()
    print(f"arms that rebuilt their own store (verdict=MISS): "
          f"{out['arms_that_rebuilt_their_own_store']}")
    if out["undetermined"]:
        print(f"UNDETERMINED: no readable provenance for "
              f"{out['arms_with_no_readable_provenance']}; "
              f"no readable controller for {out['arms_with_no_readable_controller']}")
    print(f"VERDICT: {out['verdict']}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {a.json}")
    if out["contaminated"]:
        return 1
    return 2 if out["undetermined"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
