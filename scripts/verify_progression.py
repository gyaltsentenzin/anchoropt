#!/usr/bin/env python3
"""Recompute the whole A1-A4 progression from the shipped per-case results.

This is Tier 1 replication: no GPU, no model, no cluster account. It reads the same
`eval_*.json` files the original arms wrote, recomputes every published accuracy, and runs the
paired sign tests. If a number in the README is wrong, this script says so.

    python scripts/verify_progression.py

What it does NOT do: re-run the agent. Regenerating these files requires a GPU and the BFCL v4
harness -- see REPRODUCE.md, Tier 3.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rounds.anchors import CUMULATIVE, RECOMPUTABLE

# (label, path, expected_correct, expected_n)
TRAIN = [
    ("T0  native (no prompt, no anchors)", "T1_A1_capacity/result/baseline_T0_train.json", 88, 303),
    ("T1  + A1  reroute",                  "T1_A1_capacity/result/eval_train.json",       102, 303),
    ("T2  + A2  reprompt",                 "T2_A2_not_found/result/eval_train.json",      109, 303),
    ("T3  + A3  suppress",                 "T3_A3_duplicate/result/eval_train.json",      117, 303),
    ("T5  + A4  reprompt",                 "T5_A4_no_tool_call/result/eval_train.json",   128, 303),
]

HELD = [
    ("A1",       "T1_A1_capacity/result/eval_test.json",     16, 84),
    ("A1+A2",    "T2_A2_not_found/result/eval_test.json",    16, 84),
    ("A1-A3",    "T3_A3_duplicate/result/eval_test.json",    19, 84),
    ("A1-A4",    "T5_A4_no_tool_call/result/eval_test.json", 22, 84),
]

# Rejected arms, each with the incumbent it must be differenced against. Getting this wrong is
# the standing hazard: an arm's raw accuracy is meaningless without naming its control, and A4 v1
# is the trap -- it reads 102/303, which coincides with A1's accuracy but has nothing to do with it.
# NOT part of the reproduction target. These two arms are shipped as EVIDENCE for a specific
# methodological claim -- that the incision point decides the outcome -- and are printed under
# `--with-evidence` only. Replication means the four accepted anchors above; a collaborator does not
# need to re-derive our failed attempts, and presenting them as reproduction targets would confuse
# what the result actually is.
EVIDENCE = [
    ("A4 v1  pre-generation",
     "T5_A4_no_tool_call/result/eval_train_v1_rejected.json",
     "T3_A3_duplicate/result/eval_train.json",           # v1 sat on top of A1+A2+A3
     "same signal/action/text as A4, one incision point earlier"),
    ("A2  reroute arm",
     "T2_A2_not_found/result/eval_train_losing_arm_reroute.json",
     "T1_A1_capacity/result/eval_train.json",            # both A2 arms sat on top of A1
     "0 flips: the substitution could not be built, so the arm tested nothing"),
]


def load(rel: str) -> dict[str, bool]:
    """Per-case validity for scored queries only. Prereq episodes build the store; they are not scored."""
    path = REPO / "rounds" / rel
    if not path.exists():
        raise FileNotFoundError(path)
    rows = json.loads(path.read_text())["results"]
    return {r["id"]: bool(r.get("valid")) for r in rows if not r.get("is_prereq")}


def sha(rel: str) -> str:
    """Digest of the source artifact, recorded so a report is traceable to the exact bytes read."""
    return hashlib.sha256((REPO / "rounds" / rel).read_bytes()).hexdigest()[:16]


def sign_test(before: dict[str, bool], after: dict[str, bool]) -> tuple[int, int, float | None]:
    """Paired two-sided sign test over the discordant pairs."""
    shared = before.keys() & after.keys()
    gains = sum(1 for k in shared if after[k] and not before[k])
    losses = sum(1 for k in shared if before[k] and not after[k])
    p = None
    if gains + losses:
        try:
            from scipy.stats import binomtest
            p = binomtest(gains, gains + losses, 0.5, alternative="two-sided").pvalue
        except ImportError:
            pass
    return gains, losses, p


def main() -> int:
    failures: list[str] = []
    # Every computed quantity is collected here and written to disk, so a downstream consumer (a
    # paper table, a CI diff, a reviewer) reads the recomputed values rather than re-parsing stdout.
    report: dict = {"train": [], "held_out": [], "evidence": [], "cumulative": {}, "failures": []}

    print("=" * 78)
    print("TRAIN  (n = 303 scored queries, within-job paired against the prior incumbent)")
    print("=" * 78)

    arms, prev = [], None
    for label, rel, exp_ok, exp_n in TRAIN:
        try:
            v = load(rel)
        except FileNotFoundError:
            print(f"  {label:36s}  MISSING {rel}")
            failures.append(f"missing {rel}")
            continue
        ok, n = sum(v.values()), len(v)
        acc = 100.0 * ok / n
        flag = ""
        if (ok, n) != (exp_ok, exp_n):
            flag = f"  <-- MISMATCH, expected {exp_ok}/{exp_n}"
            failures.append(f"{label}: got {ok}/{n}, expected {exp_ok}/{exp_n}")
        row = {
            "label": label.strip(), "source": rel, "correct": ok, "n": n,
            "accuracy_pct": round(acc, 4), "sha256": sha(rel),
            "expected_correct": exp_ok, "matches_published": (ok, n) == (exp_ok, exp_n),
        }
        line = f"  {label:36s}  {ok:3d}/{n:3d} = {acc:6.2f} %"
        if prev is not None:
            g, l, p = sign_test(prev, v)
            prev_acc = 100.0 * sum(prev.values()) / len(prev)
            row |= {"delta_pp": round(acc - prev_acc, 4), "gains": g, "losses": l,
                    "p_value": None if p is None else round(p, 6)}
            line += f"   Δ {acc - prev_acc:+5.2f} pp   {g:2d}g/{l:2d}l"
            if p is not None:
                line += f"  p={p:.4f}"
        print(line + flag)
        report["train"].append(row)
        arms.append((label, v))
        prev = v

    if len(arms) >= 2:
        first, last = arms[0][1], arms[-1][1]
        a0 = 100.0 * sum(first.values()) / len(first)
        a1 = 100.0 * sum(last.values()) / len(last)
        g, l, p = sign_test(first, last)
        print("  " + "-" * 74)
        print(f"  {'RECOMPUTED  T0 -> T5':36s}  {a1 - a0:+5.2f} pp   {g:2d}g/{l:2d}l"
              + (f"  p={p:.4f}" if p is not None else ""))
        # Checked against RECOMPUTABLE, not CUMULATIVE: only T0-T5 ship result artifacts, so this
        # script can verify that subset. T6/T7 are transcribed and are reported, never "verified".
        if abs((a1 - a0) - RECOMPUTABLE["train_delta_pp"]) > 0.01:
            failures.append(f"cumulative {a1 - a0:.2f} != {RECOMPUTABLE['train_delta_pp']}")
        report["cumulative"] = {
            "from_pct": round(a0, 4), "to_pct": round(a1, 4), "delta_pp": round(a1 - a0, 4),
            "gains": g, "losses": l, "p_value": None if p is None else round(p, 8),
            "n_anchors": len(arms) - 1,
        }

    print()
    print("=" * 78)
    print("DEV  (n = 84, disjoint domains -- the accept/reject criterion, so a VALIDATION set)")
    print("=" * 78)
    held = []
    for label, rel, exp_ok, exp_n in HELD:
        try:
            v = load(rel)
        except FileNotFoundError:
            print(f"  {label:36s}  MISSING {rel}")
            failures.append(f"missing {rel}")
            continue
        ok, n = sum(v.values()), len(v)
        flag = ""
        if (ok, n) != (exp_ok, exp_n):
            flag = f"  <-- MISMATCH, expected {exp_ok}/{exp_n}"
            failures.append(f"{label} held-out: got {ok}/{n}, expected {exp_ok}/{exp_n}")
        print(f"  {label:36s}  {ok:3d}/{n:3d} = {100.0 * ok / n:6.2f} %{flag}")
        report["held_out"].append({
            "label": label.strip(), "source": rel, "correct": ok, "n": n,
            "accuracy_pct": round(100.0 * ok / n, 4), "sha256": sha(rel),
            "expected_correct": exp_ok, "matches_published": (ok, n) == (exp_ok, exp_n),
        })
        held.append((label, v))

    # A4's own held-out step is the one significance claim that must NOT be overstated.
    if len(held) >= 2:
        (_, a3), (_, a4) = held[-2], held[-1]
        g, l, p = sign_test(a3, a4)
        step = 100.0 * sum(a4.values()) / len(a4) - 100.0 * sum(a3.values()) / len(a3)
        print("  " + "-" * 74)
        print(f"  {'A4 step (A1-A3 -> A1-A4)':36s}  {step:+5.2f} pp   {g:2d}g/{l:2d}l"
              + (f"  p={p:.4f}" if p is not None else ""))
        if p is not None and p < 0.05:
            failures.append("A4 held-out step reads significant; the repo claims it is NOT")
        print("       NOT significant -- direction transfers, significance does not. And note the")
        print("       benchmark is DETERMINISTIC, so a p here is a diagnostic, not a test.")
        report["held_out_a4_step"] = {
            "delta_pp": round(step, 4), "gains": g, "losses": l,
            "p_value": None if p is None else round(p, 6),
            "significant_at_05": bool(p is not None and p < 0.05),
        }

    print()
    if "--with-evidence" in sys.argv:
        print("=" * 78)
        print("SUPPORTING EVIDENCE  (not a reproduction target)")
        print("=" * 78)
        print("  Two non-accepted arms, shipped only because they are the evidence for the claim")
        print("  that the incision point decides the outcome. Each is differenced against ITS OWN")
        print("  incumbent -- a raw accuracy without a named control is meaningless.")
        print()
        for label, rel, ctrl_rel, why in EVIDENCE:
            try:
                v, ctrl = load(rel), load(ctrl_rel)
            except FileNotFoundError:
                print(f"  {label:24s}  MISSING")
                continue
            acc = 100.0 * sum(v.values()) / len(v)
            c_acc = 100.0 * sum(ctrl.values()) / len(ctrl)
            g, l, p = sign_test(ctrl, v)
            print(f"  {label:24s} {acc:6.2f} % vs {c_acc:6.2f} %   Δ {acc - c_acc:+5.2f} pp"
                  f"   {g:2d}g/{l:2d}l" + (f"  p={p:.4f}" if p is not None else ""))
            print(f"  {'':24s} {why}")
            report["evidence"].append({
                "label": label.strip(), "source": rel, "incumbent_source": ctrl_rel,
                "accuracy_pct": round(acc, 4), "incumbent_accuracy_pct": round(c_acc, 4),
                "delta_pp": round(acc - c_acc, 4), "gains": g, "losses": l,
                "p_value": None if p is None else round(p, 6), "note": why,
                "is_reproduction_target": False,
            })
        print()

    # Persist the recomputed values. A downstream consumer -- a paper table, a CI diff, a reviewer
    # checking a claim -- should read these rather than re-parse stdout. Written on FAILURE too:
    # a verification that fails is exactly when you want the numbers on disk.
    report["failures"] = failures
    report["verdict"] = "FAIL" if failures else "OK"
    out = REPO / "verification_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print()
    if failures:
        print("FAILED -- the shipped results do not match the published numbers:")
        for f in failures:
            print(f"  - {f}")
        print(f"\n  details written to {out.relative_to(REPO)}")
        return 1

    print(f"OK -- all {len(TRAIN) + len(HELD)} published accuracies reproduced from the "
          f"shipped per-case results.")
    print(f"   RECOMPUTED HERE: {RECOMPUTABLE['train_from']:.2f} % -> "
          f"{RECOMPUTABLE['train_to']:.2f} % (+{RECOMPUTABLE['train_delta_pp']:.2f} pp) over "
          f"{len(RECOMPUTABLE['rounds'])} rounds / {len(RECOMPUTABLE['anchors'])} anchors.")
    print(f"   FULL STACK (transcribed for {', '.join(RECOMPUTABLE['transcribed_rounds'])}): "
          f"{CUMULATIVE['train_from']:.2f} % -> {CUMULATIVE['train_to']:.2f} % "
          f"(+{CUMULATIVE['train_delta_pp']:.2f} pp), {CUMULATIVE['n_anchors']} anchors.")
    print(f"   recomputed values written to {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
