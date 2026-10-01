#!/usr/bin/env python3
"""Export one round's artifacts into the repo and derive every reported number from them.

    python benchmarks/tau2/scripts/export_results.py \\
        --matrix /path/to/anchoropt_tau2_matrix \\
        --dest   benchmarks/tau2/rounds/SELFTEACH_R1

Three outputs under --dest:

  <arm>/         the arm's own artifacts (whitelist below). Logs and LSF bookkeeping are left out: they are
                 large, host-specific, and no number depends on them.

RAW TRAJECTORIES ARE STRIPPED BY DEFAULT. `.gitignore` keeps raw trajectories out of this repo so it stays
clonable, and they are ~85% of a round's bytes. No reported number and no anchor depends on them -- the
summary recomputes byte-for-byte and every anchor rebuilds without them, which a test enforces. What they
ARE needed for is re-running `propose` from the committed files, so `--archive` writes the full set,
trajectories included, to a tarball outside the repo, and each stripped baseline.json records where it is.
  summary.json   every number docs/TAU2_SELFTEACH_ROUND1.md reports, per arm, recomputed from <arm>/.
  anchors.json   the promoted controllers ("anchors") in installable form: boundary, signal, action,
                 variant, eta, and where the instruction text came from.

`--from-dest` skips the copy and recomputes from an existing --dest, which is how a reader checks the
report against the committed artifacts without access to the original matrix directory.

`--markdown` prints the report's tables from summary.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

# The files an arm's numbers depend on. Anything else in an arm dir is a log or bookkeeping.
WHITELIST = re.compile(
    r"^(baseline|arm_manifest|results|selection|rescore|validation_test|arm)\.json$"
    r"|^arm_\d+\.json$"
    r"|^incumbent_rep_\d+\.json$"
    r"|^heldout_p[01]_test_rep\d+\.json$")
ARM_FILE = re.compile(r"^arm_\d+\.json$")

MODELS = {"qwen3_6_35b_a3b": "qwen3.6-35b-a3b", "minimax_m2_5": "minimax-m2.5",
          "granite_4_1_30b": "granite-4.1-30b"}
TEACHERS = {"self": "self", "claude_sonnet_5": "sonnet"}
ORDER = [(m, d) for m in MODELS for d in ("airline", "retail", "telecom")]


def _void(term) -> bool:
    return str(term or "").startswith("harness_error")


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def copy_arms(matrix: Path, dest: Path, teacher: str, *, keep_trajectories: bool = False,
              archive_note: str = "") -> list[str]:
    copied = []
    for m, dom in ORDER:
        name = f"{m}__{teacher}__{dom}"
        src = matrix / name
        if not (src / "baseline.json").exists():
            continue
        out = dest / name
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if not (f.is_file() and WHITELIST.match(f.name)):
                continue
            if f.name == "baseline.json" and not keep_trajectories:
                blob = json.loads(f.read_text())
                n = len(blob.get("events") or {})
                blob["events"] = {}
                blob["events_stripped"] = {"n_cases": n, "full_archive": archive_note,
                                           "why": "raw trajectories are kept out of the repo (.gitignore)"}
                (out / f.name).write_text(json.dumps(blob, indent=2) + "\n")
            else:
                shutil.copy2(f, out / f.name)
        copied.append(name)
    return copied


def summarise_arm(d: Path) -> dict:
    name = d.name
    m, teacher, dom = name.split("__")
    base = _load(d / "baseline.json")
    man = _load(d / "arm_manifest.json") or {}
    res = _load(d / "results.json") or {}
    sel = _load(d / "selection.json") or {}
    run = base["run"]
    scored = list(base.get("scored_task_ids") or base["task_ids"])
    inc = {c: bool(run["solved"][c]) for c in scored}
    rec = {
        "arm": name, "model": MODELS[m], "domain": dom, "teacher": TEACHERS.get(teacher, teacher),
        "split": base.get("split"), "portal": base.get("agent_provider"),
        "agent_model_id": base.get("agent_model_id"), "user_model": base.get("user_llm"),
        "max_steps": base.get("max_steps"), "sim_timeout": base.get("sim_timeout"),
        "train_tasks": len(scored),
        "p0_train": {"solved": run["n_solved"], "n": len(scored),
                     "void": sum(_void(t) for t in (run.get("termination") or {}).values()),
                     "truncated": run.get("truncated") or {}},
        "residual_cases": len(man.get("residual_case_ids") or []),
        "residual_key": man.get("residual_key"),
        "arms_emitted": man.get("n_arms", 0),
        "arms_measured": len(res.get("results") or []),
        "arms_voided": len(res.get("voided") or []),
        "outcome_class": sel.get("outcome_class") or ("DONE_no_arms" if not man.get("n_arms") else None),
        "teacher_invoked": (man.get("teacher") or {}).get("invoked"),
        "teacher_accepted": [(g.get("variant"), g.get("instruction"))
                             for g in ((man.get("teacher") or {}).get("accepted") or [])],
    }
    # every measured arm's train net, for the "best of N" context
    per_arm = []
    for f in sorted(x for x in d.iterdir() if ARM_FILE.match(x.name)):
        a = json.loads(f.read_text())
        ok = [c for c in scored if not _void((a.get("termination") or {}).get(c))]
        g = [c for c in ok if a["solved"].get(c) and not inc[c]]
        lo = [c for c in ok if not a["solved"].get(c) and inc[c]]
        fired = set(a["firings"].get("cases_fired") or [])
        per_arm.append({"label": a["label"], "solved": sum(bool(a["solved"][c]) for c in ok), "n": len(ok),
                        "gains": len(g), "losses": len(lo), "net": len(g) - len(lo),
                        "executed": a["firings"].get("interventions_executed", 0),
                        "cases_fired": len(fired), "gains_where_fired": len(set(g) & fired)})
    rec["measured_arms"] = per_arm

    promoted = sel.get("promoted") if sel.get("improved") else None
    rec["p1_train"] = None
    if promoted:
        row = next(r for r in man["arms"] if r["arm_label"] == promoted)
        pa = next(x for x in per_arm if x["label"] == promoted)
        rec["p1_train"] = {**pa, "arm_label": promoted, "boundary": row["boundary"],
                           "signal": row["signal"], "action": row["action"], "operator": row.get("operator"),
                           "variant": row.get("variant"), "eta": row.get("eta"),
                           "instruction_source": ("teacher" if str(row.get("variant", "")).startswith("teacher_")
                                                  else "shipped")}
    rs = _load(d / "rescore.json")
    if rs:
        pr = next((x for x in rs["arms"] if x["promoted"]), None)
        rec["train_replicates"] = {"p0_totals": rs["incumbent_totals"], "p0_mean": rs["incumbent_mean"],
                                   "p0_sd": rs["incumbent_sd"], "flip_cases": rs["flip_cases"],
                                   "n_cases": rs["n_cases"],
                                   "promoted_total": pr and pr["total"], "promoted_n": pr and pr["n"],
                                   "promoted_expected_net": pr and pr["expected_net"],
                                   "promoted_z": pr and pr["z"],
                                   "promoted_above_best_p0": pr and pr["above_best_incumbent"]}
    v = _load(d / "validation_test.json")
    if v:
        t = {"n": v["n"], "k": v["k"], "portal": v.get("agent_provider"),
             "p0_totals": v["p0_totals"], "p0_pooled": [sum(v["p0_totals"]), v["n"] * len(v["p0_totals"])],
             "p0_void": v.get("p0_void"),
             # tau-bench's pass^k over the held-out runs (keys are k; JSON stores them as strings)
             "p0_pass_hat_k": {int(k): x for k, x in (v.get("p0_pass_hat_k") or {}).items()}}
        if v.get("tested_p1"):
            t.update({"p1_totals": v["p1_totals"],
                      "p1_pooled": [sum(v["p1_totals"]), v["n"] * len(v["p1_totals"])],
                      "p1_void": v.get("p1_void"), "p1_executed": v.get("p1_executed"),
                      "p1_pass_hat_k": {int(k): x for k, x in (v.get("p1_pass_hat_k") or {}).items()},
                      "mean_diff": v["mean_diff"], "se": v["se"], "z": v.get("z"),
                      "criterion_2_pass": v.get("criterion_2_no_aggregate_regression"),
                      "cases_fired_any_run": len(v.get("cases_fired_any_run") or []),
                      "improved_where_fired": len(v.get("criterion_3_gains_attributable") or []),
                      "improved_where_not_fired": len(v.get("criterion_3_gains_not_attributable") or []),
                      "worsened_where_fired": len(v.get("criterion_3_collateral_losses") or [])})
        else:
            t["why_no_p1"] = v.get("why_not_p1")
        refilled = []
        for f in sorted(d.glob("heldout_p*_test_rep*.json")):
            for r in (json.loads(f.read_text()).get("refilled") or []):
                refilled.append({"file": f.name, **r})
        t["refilled"] = refilled
        rec["test"] = t
    return rec


def anchors_from(summary: list[dict]) -> list[dict]:
    out = []
    for r in summary:
        p = r.get("p1_train")
        if not p:
            continue
        out.append({"arm": r["arm"], "model": r["model"], "domain": r["domain"], "teacher": r["teacher"],
                    "portal_train": r["portal"], "portal_test": (r.get("test") or {}).get("portal"),
                    "arm_label": p["arm_label"], "boundary": p["boundary"], "signal": p["signal"],
                    "action": p["action"], "operator": p["operator"], "variant": p["variant"],
                    "eta": p["eta"], "instruction_source": p["instruction_source"],
                    "install": {"module": "benchmarks/tau2/tau2_mechanism.py",
                                "rebuild": ("run_anchoropt_round._controller_for(<arm dir>, baseline, "
                                            "manifest, arm_label)"),
                                "adapter_boundary_key": {"post_generation_pre_exec": "before_tool_dispatch",
                                                         "post_execution": "after_tool_result",
                                                         "pre_generation": "before_agent_turn"}[p["boundary"]]},
                    "accepted": False})
    return out


def _frac(pair):
    return f"{pair[0]}/{pair[1]}"


def markdown(summary: list[dict]) -> str:
    lines = ["| model | domain | teaching-mode | P0 train | P1 train | P0 test | P1 test |",
             "|---|---|---|---|---|---|---|"]
    for r in summary:
        p1 = r.get("p1_train")
        t = r.get("test") or {}
        lines.append(f"| {r['model']} | {r['domain']} | {r['teacher']} | "
                     f"{r['p0_train']['solved']}/{r['p0_train']['n']} | "
                     f"{(str(p1['solved']) + '/' + str(p1['n'])) if p1 else '—'} | "
                     f"{_frac(t['p0_pooled']) if t else '—'} | "
                     f"{_frac(t['p1_pooled']) if t.get('p1_pooled') else '—'} |")
    return "\n".join(lines)


def markdown_pass_hat_k(summary: list[dict]) -> str:
    """pass^k on the test split, P0 and P1, for every k the held-out runs support."""
    ks = sorted({k for r in summary for k in ((r.get("test") or {}).get("p0_pass_hat_k") or {})})
    if not ks:
        return ""
    head = "| model | domain | teaching-mode | side | " + " | ".join(f"pass^{k}" for k in ks) + " |"
    lines = [head, "|---|---|---|---|" + "---|" * len(ks)]
    for r in summary:
        t = r.get("test") or {}
        for side in ("p0", "p1"):
            vals = t.get(f"{side}_pass_hat_k") or {}
            if not vals:
                continue
            cells = " | ".join(f"{vals[k]:.3f}" if k in vals else "—" for k in ks)
            lines.append(f"| {r['model']} | {r['domain']} | {r['teacher']} | {side.upper()} | {cells} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, default=None,
                    help="your round's matrix output directory -- required unless --from-dest")
    ap.add_argument("--dest", type=Path, default=Path("benchmarks/tau2/rounds/SELFTEACH_R1"))
    ap.add_argument("--keep-trajectories", action="store_true",
                    help="copy baseline.json with its trajectories (do not commit the result)")
    ap.add_argument("--archive", type=Path, default=None,
                    help="also write the FULL artifact set, trajectories included, to this .tar.gz")
    ap.add_argument("--teacher", default="self")
    ap.add_argument("--from-dest", action="store_true", help="recompute from --dest; copy nothing")
    ap.add_argument("--markdown", action="store_true", help="print the report's main table")
    a = ap.parse_args()
    if not a.from_dest and a.matrix is None:
        ap.error("--matrix is required unless --from-dest is set")

    a.dest.mkdir(parents=True, exist_ok=True)
    if not a.from_dest:
        if a.archive:
            import hashlib
            import tarfile
            a.archive.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(a.archive, "w:gz") as tar:
                for m, dom in ORDER:
                    src = a.matrix / f"{m}__{a.teacher}__{dom}"
                    for f in sorted(src.iterdir()) if src.is_dir() else ():
                        if f.is_file() and WHITELIST.match(f.name):
                            tar.add(f, arcname=f"{a.dest.name}/{src.name}/{f.name}")
            digest = hashlib.sha256(a.archive.read_bytes()).hexdigest()
            print(f"archive: {a.archive}  sha256 {digest}")
        note = str(a.archive) if a.archive else ""
        names = copy_arms(a.matrix, a.dest, a.teacher, keep_trajectories=a.keep_trajectories,
                          archive_note=note)
        print(f"copied {len(names)} arm(s) into {a.dest}"
              + ("" if a.keep_trajectories else " (trajectories stripped)"))
    arms = [a.dest / f"{m}__{a.teacher}__{d}" for m, d in ORDER]
    summary = [summarise_arm(d) for d in arms if (d / "baseline.json").exists()]
    (a.dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (a.dest / "anchors.json").write_text(json.dumps(anchors_from(summary), indent=2) + "\n")
    print(f"wrote {a.dest / 'summary.json'} and {a.dest / 'anchors.json'} ({len(summary)} arm(s))")
    if a.markdown:
        print()
        print(markdown(summary))
        pk = markdown_pass_hat_k(summary)
        if pk:
            print()
            print(pk)
    # sanity: every tested arm's pooled P0/P1 must equal the sum of its runs
    for r in summary:
        t = r.get("test") or {}
        if t and sum(t["p0_totals"]) != t["p0_pooled"][0]:
            raise SystemExit(f"inconsistent P0 pooling for {r['arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
