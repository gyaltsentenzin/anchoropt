#!/usr/bin/env python3
"""AnchorOpt vs GEPA vs Self-Harness on tau-bench: held-out pass^1 and pass^k, from each framework's
own artifacts.

NOT RUNNABLE STANDALONE: this reads GEPA's and Self-Harness's own result artifacts (set
GEPA_TAU2_MATRIX and SELF_HARNESS_TAU2_DIR below to your own copies if you have them). It is
shipped as the documented derivation behind the comparison in docs/TAU2_FRAMEWORK_COMPARISON.md.

    python benchmarks/tau2/scripts/compare_frameworks.py [--json out.json]

WHAT IS COMPARED. For every (model, domain, teacher) cell: the stock pipeline and the framework's
candidate, both on tau-bench's official TEST split. Every number is computed here from PER-TASK outcomes of
every test trial the framework ran -- not copied from a report -- so pass^1 and pass^k come from one set of
data and one estimator (the driver's `pass_hat_k`, tested equal to tau-bench's own). Each framework is
compared against ITS OWN stock measurement: the pipelines are identical, but each framework measured stock
separately, and pairing within a framework keeps the comparison matched.

WHERE EACH TRIAL COMES FROM (read-only):
  AnchorOpt     benchmarks/tau2/rounds/SELFTEACH_R1/<arm>/heldout_p{0,1}_test_rep{1..4}.json  (`solved`)
  GEPA          $GEPA_TAU2_MATRIX/<domain>__<model>__<teacher>/
                  stock      eval_stock_test.json
                  candidate  passk/pass_seed{300..303}.json where GEPA ran them (the six self-teach cells
                             whose prompt changed; seed 300 is eval_gepa_test.json), else eval_gepa_test.json
  Self-Harness  $SELF_HARNESS_TAU2_DIR/<domain>_<self|claude>_<tag>/
                  stock      baseline_eval/{fragments,cells}/test_r{1,2}
                  candidate  candidates/<promoted>/promote_eval/{fragments,cells}/test_r{1,2}, plus
                             passk/{fragments,cells}/test_r{3,4} where SH ran them (its two accepted
                             self-teach cells)

POOLING. Trials are pooled only when they measured the same thing: same prompt hash (GEPA), same surface
hash and model (Self-Harness), and every trial scoring the same task set. Anything else is refused.

CONFIGURATION MATCHING. qwen3.6 runs with reasoning OFF in AnchorOpt and GEPA. Self-Harness's main
18-arm grid ran qwen3.6 with reasoning ON; its six `qwen3.6nothink` arms are the matched configuration, so
those are the qwen rows used here. Every cell's config is asserted, not assumed.

METRIC. GEPA and Self-Harness score strict pass (reward == 1 AND a clean termination, infra errors
counted as failures); Self-Harness's is re-derived here from its raw simulations and checked against its
fragments. AnchorOpt counts reward == 1 and excludes void episodes; tau-bench assigns reward 0 to any
unclean termination, so the two agree wherever there are no void episodes. The script refuses a trial with
any infra error / void episode rather than mixing the two conventions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
AOPT = REPO / "benchmarks" / "tau2" / "rounds" / "SELFTEACH_R1"
GEPA = Path(os.environ["GEPA_TAU2_MATRIX"]) if os.environ.get("GEPA_TAU2_MATRIX") else None
SH = Path(os.environ["SELF_HARNESS_TAU2_DIR"]) if os.environ.get("SELF_HARNESS_TAU2_DIR") else None

MODELS = ("qwen3.6-35b-a3b", "minimax-m2.5", "granite-4.1-30b")
DOMAINS = ("airline", "retail", "telecom")
TEACHERS = ("self", "sonnet")
FRAMEWORKS = (("anchoropt", "AnchorOpt"), ("gepa", "GEPA"), ("self_harness", "Self-Harness"))
MAX_K = 4

AOPT_TAG = {"qwen3.6-35b-a3b": "qwen3_6_35b_a3b", "minimax-m2.5": "minimax_m2_5",
            "granite-4.1-30b": "granite_4_1_30b"}
GEPA_TAG = {"qwen3.6-35b-a3b": "qwen36", "minimax-m2.5": "minimax", "granite-4.1-30b": "granite"}
# qwen: the no-think arms are the configuration-matched ones (see module docstring)
SH_TAG = {"qwen3.6-35b-a3b": "qwen3.6nothink", "minimax-m2.5": "minimax", "granite-4.1-30b": "granite"}
SH_TEACH = {"self": "self", "sonnet": "claude"}
SH_CLEAN = frozenset({"user_stop", "agent_stop"})
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
CEILING = 0.988          # a stock rate at or above this leaves no room to gain

_spec = importlib.util.spec_from_file_location("tau2_round_driver", REPO / "benchmarks" / "tau2" /
                                               "run_anchoropt_round.py")
_driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_driver)
pass_hat_k = _driver.pass_hat_k


def _j(p: Path):
    return json.loads(p.read_text())


def _n(x: float, spec: str) -> str:
    """A number for a table, with a typographic minus. Rounded from its exact decimal value, so a float
    error (0.6875 - 0.65 = 0.03749...) cannot print 3.75 as 3.7 in one table and 3.8 in another."""
    return format(round(x, 9), spec).replace("-", "−")


def _side(trials: list[dict[str, bool]], where: str) -> dict:
    """One side of a cell from its trials ({task_id: success}, one map per independent trial)."""
    tasks = sorted(trials[0])
    for t in trials:
        if sorted(t) != tasks:
            raise SystemExit(f"{where}: trials score different task sets -- not poolable")
    per_run = [sum(t.values()) for t in trials]
    passed, total = sum(per_run), len(tasks) * len(trials)
    succ = {c: sum(bool(t[c]) for t in trials) for c in tasks}
    return {"passed": passed, "total": total, "runs": len(trials), "rate": passed / total,
            "per_run": per_run, "pass_hat_k": pass_hat_k(trials, tasks), "task_successes": succ}


# ---------------------------------------------------------------------------------------------------------
# the three frameworks
# ---------------------------------------------------------------------------------------------------------

def anchoropt(model: str, domain: str, teacher: str) -> dict:
    if teacher != "self":
        return {"status": "not run"}
    d = AOPT / f"{AOPT_TAG[model]}__self__{domain}"
    v = _j(d / "validation_test.json")

    def trials(side: str) -> list[dict[str, bool]]:
        out = []
        for r, want in enumerate(v[f"{side}_totals"], 1):
            rep = _j(d / f"heldout_{side}_test_rep{r}.json")
            if rep["harness_errors"] or any(v.get(f"{side}_void") or []):
                raise SystemExit(f"AnchorOpt {d.name}: void episodes present; refill before comparing")
            solved = {str(c): bool(s) for c, s in rep["solved"].items()}
            if sum(solved.values()) != want:
                raise SystemExit(f"AnchorOpt {d.name} {side} rep{r}: per-task outcomes disagree with totals")
            out.append(solved)
        return out

    out = {"stock": _side(trials("p0"), d.name), "portal": v.get("agent_provider")}
    if v.get("tested_p1"):
        out["candidate"] = _side(trials("p1"), d.name)
        out["status"] = "promoted"
    else:
        out["candidate"] = out["stock"]          # nothing promoted: the framework ships stock
        out["status"] = "no candidate: " + str(v.get("why_not_p1") or "")
    return out


def _gepa_trial(e: dict, where: str, model: str) -> dict[str, bool]:
    if e["n_infra_errors"] or e.get("n_missing_simulations"):
        raise SystemExit(f"GEPA {where}: infra errors / missing simulations -- not a usable trial")
    if model.startswith("qwen"):
        assert e["agent_extra_body"] == NO_THINK, (where, e["agent_extra_body"])
    out = {str(t["task_id"]): bool(t["strict_pass"]) for t in e["tasks"]}
    if len(out) != e["total"] or sum(out.values()) != e["passed"]:
        raise SystemExit(f"GEPA {where}: per-task outcomes disagree with the file's totals")
    return out


def gepa(model: str, domain: str, teacher: str) -> dict:
    d = GEPA / f"{domain}__{GEPA_TAG[model]}__{teacher}"
    s, g = _j(d / "eval_stock_test.json"), _j(d / "eval_gepa_test.json")
    passk = sorted((d / "passk").glob("pass_seed*.json")) if (d / "passk").is_dir() else []
    cand = [_j(p) for p in passk] or [g]
    if {e["instruction_sha256"] for e in cand} != {g["instruction_sha256"]}:
        raise SystemExit(f"GEPA {d.name}: pass^k trials ran a different prompt than eval_gepa_test")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()           # noqa: E731
    out = {"stock": _side([_gepa_trial(s, d.name, model)], d.name),
           "portal": _j(d / "manifest.json").get("provider")}
    if sha(d / "best_prompt.txt") == sha(d / "seed_prompt.txt"):
        # GEPA shipped the seed. Its "gepa" test evaluation is then the stock one served from cache
        # (sub-second wall clock, identical per task) -- the same measurement, not a second one.
        if _gepa_trial(g, d.name, model) != _gepa_trial(s, d.name, model):
            raise SystemExit(f"GEPA {d.name}: seed returned, but its test evaluation differs from stock's")
        out["candidate"] = out["stock"]
        out["status"] = "returned the seed prompt unchanged"
    else:
        out["candidate"] = _side([_gepa_trial(e, d.name, model) for e in cand], d.name)
        out["status"] = "evolved prompt"
    return out


def _sh_trials(frag_dirs: list[Path], where: str) -> tuple[list[dict[str, bool]], dict]:
    """Strict pass per task, re-derived from each fragment's raw simulations and checked against it."""
    frags = []
    for fd in frag_dirs:
        for f in sorted(fd.glob("test_r*.json")) if fd.is_dir() else []:
            frags.append((f, _j(f)))
    if not frags:
        raise SystemExit(f"Self-Harness {where}: no test fragments")
    if len({fr["surface_sha256"] for _, fr in frags}) > 1 or len({fr["model_name"] for _, fr in frags}) > 1:
        raise SystemExit(f"Self-Harness {where}: trials disagree on surface or model -- not poolable")
    trials = []
    for f, fr in frags:
        if fr["n_infra_errors"]:
            raise SystemExit(f"Self-Harness {f}: infra errors -- not a usable trial")
        sims = _j(f.parent.parent / "cells" / f"test_r{fr['repeat']}" / "results.json")["simulations"]
        out: dict[str, bool] = {}
        for sim in sims:
            tid = str(sim["task_id"])
            if tid in out:
                raise SystemExit(f"Self-Harness {f}: task {tid} simulated twice in one trial")
            # the reward is in reward_info; the top-level field is None in these files (DATA_MAP.md §1.1)
            out[tid] = (sim.get("reward_info") or {}).get("reward") == 1.0 and \
                sim.get("termination_reason") in SH_CLEAN
        if len(out) != fr["total"] or sum(out.values()) != fr["passed"]:
            raise SystemExit(f"Self-Harness {f}: re-derived outcomes disagree with the fragment")
        trials.append(out)
    return trials, frags[0][1]


def self_harness(model: str, domain: str, teacher: str) -> dict:
    d = SH / f"{domain}_{SH_TEACH[teacher]}_{SH_TAG[model]}"
    summ = _j(d / "round_summary.json")
    base, frag = _sh_trials([d / "baseline_eval" / "fragments"], d.name)
    if model.startswith("qwen"):
        assert frag["agent_extra_body"] == NO_THINK, (d.name, frag["agent_extra_body"])
    decision = summ.get("decision")
    out = {"stock": _side(base, d.name), "status": decision, "portal": frag.get("model_name")}
    promoted = summ.get("promoted_candidate_id")
    if decision == "declined" or not promoted:
        out["candidate"] = out["stock"]            # the proposer declined: the framework ships stock
    else:
        cand, _ = _sh_trials([d / "candidates" / promoted / "promote_eval" / "fragments",
                              d / "passk" / "fragments"], d.name)
        out["candidate"] = _side(cand, d.name)
        out["promoted"] = promoted
    return out


def build() -> list[dict]:
    rows = []
    for m in MODELS:
        for dom in DOMAINS:
            for t in TEACHERS:
                rows.append({"model": m, "domain": dom, "teacher": t,
                             "anchoropt": anchoropt(m, dom, t), "gepa": gepa(m, dom, t),
                             "self_harness": self_harness(m, dom, t)})
    return rows


# ---------------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------------

def per_task_pass_k(side: dict, k: int) -> dict[str, float] | None:
    """Each task's C(c, k) / C(n, k); None when the side has fewer than k trials."""
    n = side["runs"]
    if n < k:
        return None
    return {t: math.comb(c, k) / math.comb(n, k) for t, c in side["task_successes"].items()}


def paired(a: dict, b: dict, k: int) -> tuple[float, float] | None:
    """pass^k(b) - pass^k(a) on the same tasks, with its standard error over tasks.

    Tasks are the sampling unit: each task's pass^k estimate already carries that task's trial-to-trial
    noise, so the spread of the per-task differences covers both sources. At k = n there is one size-n
    subset of trials, so this is the only standard error pass^n admits."""
    ua, ub = per_task_pass_k(a, k), per_task_pass_k(b, k)
    if ua is None or ub is None:
        return None
    if sorted(ua) != sorted(ub):
        raise SystemExit("paired pass^k over different task sets")
    diffs = [ub[t] - ua[t] for t in ua]
    return statistics.fmean(diffs), statistics.stdev(diffs) / math.sqrt(len(diffs))


# ---------------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------------

def _marks(key: str, fw: dict) -> str:
    m = ""
    if key == "anchoropt" and fw["status"].startswith("no candidate"):
        m += "ⁿ"
    if key == "gepa" and fw["status"] == "returned the seed prompt unchanged":
        m += "ᵍ"
    if key == "self_harness" and fw["status"] == "declined":
        m += "ᵈ"
    if key != "anchoropt" and fw["candidate"]["runs"] > fw["stock"]["runs"]:
        m += "ᵏ"
    if fw["stock"]["rate"] >= CEILING:
        m += "ᶜ"
    return f" {m}" if m else ""


def _delta(fw: dict) -> float:
    return fw["candidate"]["rate"] - fw["stock"]["rate"]


def _fmt(key: str, fw: dict) -> str:
    if not fw or "stock" not in fw:
        return "not run"
    s, c = fw["stock"], fw["candidate"]
    return (f"{_n(100 * s['rate'], '.1f')} → {_n(100 * c['rate'], '.1f')} ({_n(100 * _delta(fw), '+.1f')})"
            f"{_marks(key, fw)}")


def markdown(rows: list[dict]) -> str:
    """The pass^1 comparison: stock → candidate (Δ, percentage points), per cell and framework."""
    out = ["| model | domain | teacher | AnchorOpt | GEPA | Self-Harness |",
           "|---|---|---|---|---|---|"]
    for teacher in TEACHERS:
        for r in (r for r in rows if r["teacher"] == teacher):
            out.append(f"| {r['model']} | {r['domain']} | {r['teacher']} | "
                       + " | ".join(_fmt(k, r[k]) for k, _ in FRAMEWORKS) + " |")
    return "\n".join(out)


def _pts(x: float, spec: str = "+.1f") -> str:
    """A difference in percentage points; an exact zero prints unsigned."""
    return "0.0" if abs(x) < 1e-9 else _n(100 * x, spec)


def _summ(rows: list[dict], key: str) -> str:
    fws = [r[key] for r in rows if "stock" in r[key]]
    if not fws:
        return "not run | – | – | –"
    d = [_delta(f) for f in fws]
    g, z = sum(x > 1e-9 for x in d), sum(abs(x) <= 1e-9 for x in d)
    below = [_delta(f) for f in fws if f["stock"]["rate"] < 1.0]
    return (f"{_pts(statistics.fmean(d))} | {_pts(statistics.median(d))} | {g} / {z} / {len(d) - g - z} | "
            f"{_pts(statistics.fmean(below))} (n = {len(below)})")


def markdown_summary(rows: list[dict]) -> str:
    out = ["| framework | teacher | mean Δ | median Δ | gains / zeros / losses | "
           "mean Δ excluding cells where stock = 100% |",
           "|---|---|---:|---:|---|---:|"]
    for teacher in TEACHERS:
        sel = [r for r in rows if r["teacher"] == teacher]
        for key, name in FRAMEWORKS:
            out.append(f"| {name} | {teacher} | {_summ(sel, key)} |")
    return "\n".join(out)


def _pk(side: dict, k: int) -> str:
    v = side["pass_hat_k"].get(k)
    return "–" if v is None else f"{v:.3f}"


def _own(fw: dict, k: int) -> tuple[float, float] | None:
    """Δpass^k against the framework's own stock, with its se; a cell that ships stock is an exact 0."""
    if fw["candidate"] is fw["stock"]:
        return (0.0, 0.0) if k in fw["stock"]["pass_hat_k"] else None
    return paired(fw["stock"], fw["candidate"], k)


def _span(xs: list[int]) -> str:
    return str(min(xs)) if min(xs) == max(xs) else f"{min(xs)}–{max(xs)}"


def markdown_headline_k(rows: list[dict]) -> str:
    """Mean Δpass^k against each framework's own stock, over the cells of one teacher (points, se)."""
    out = ["| framework | teacher | test trials, stock / candidate | Δpass^1 | Δpass^2 | Δpass^3 | Δpass^4 |",
           "|---|---|---|---:|---:|---:|---:|"]
    for teacher in TEACHERS:
        sel = [r for r in rows if r["teacher"] == teacher]
        for key, name in FRAMEWORKS:
            fws = [r[key] for r in sel if "stock" in r[key]]
            if not fws:
                out.append(f"| {name} | {teacher} | not run | – | – | – | – |")
                continue
            cells = []
            for k in range(1, MAX_K + 1):
                got = [_own(f, k) for f in fws]
                if any(g is None for g in got):
                    cells.append("–")
                    continue
                se = math.sqrt(sum(g[1] ** 2 for g in got)) / len(got)
                cells.append(f"{_pts(statistics.fmean(g[0] for g in got))} ({100 * se:.1f})")
            trials = (f"{_span([f['stock']['runs'] for f in fws])} / "
                      f"{_span([f['candidate']['runs'] for f in fws])}")
            out.append(f"| {name} | {teacher} | {trials} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def markdown_pass_hat_k(rows: list[dict]) -> str:
    """pass^1..pass^4 per self-teach cell and framework, stock → candidate at every k both sides reach."""
    out = ["| model | domain | framework | test trials, stock / candidate | pass^1 | pass^2 | pass^3 | pass^4 | "
           "Δ at the largest k both sides reach (se) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in (r for r in rows if r["teacher"] == "self"):
        for i, (key, name) in enumerate(FRAMEWORKS):
            fw = r[key]
            s, c = fw["stock"], fw["candidate"]
            ships = c is s
            cells = []
            for k in range(1, MAX_K + 1):
                if k not in c["pass_hat_k"]:
                    cells.append("–")
                elif k in s["pass_hat_k"] and not ships:
                    cells.append(f"{_pk(s, k)} → {_pk(c, k)}")
                else:
                    cells.append(_pk(c, k))
            kmax = min(s["runs"], c["runs"])
            delta = "–" if ships else (lambda d: f"k = {kmax}: {_pts(d[0])} ({100 * d[1]:.1f})")(
                paired(s, c, kmax))
            head = f"| {r['model']} | {r['domain']} " if i == 0 else "| | "
            out.append(f"{head}| {name}{' (ships stock)' if ships else ''} | {s['runs']} / {c['runs']} | "
                       + " | ".join(cells) + f" | {delta} |")
    return "\n".join(out)


def matched(rows: list[dict], keys: tuple[str, ...], k: int) -> list[dict]:
    """Self-teach cells where every framework in `keys` has at least k candidate trials."""
    return [r for r in rows if r["teacher"] == "self"
            and all(k in r[key]["candidate"]["pass_hat_k"] for key in keys)]


def markdown_matched(rows: list[dict]) -> str:
    """Mean pass^k of two frameworks' candidates over the cells where both have k = 4, on the same tasks."""
    out = ["| comparison | side | pass^1 | pass^2 | pass^3 | pass^4 | pass^1 − pass^4 |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for other in ("gepa", "self_harness"):
        sel = matched(rows, ("anchoropt", other), MAX_K)
        name = dict(FRAMEWORKS)[other]
        own = _span([r[other]["stock"]["runs"] for r in sel])
        entries = [("stock, as AnchorOpt measured it (4 runs)", [r["anchoropt"]["stock"] for r in sel]),
                   (f"stock, as {name} measured it ({own} run{'s' if own != '1' else ''})",
                    [r[other]["stock"] for r in sel]),
                   ("AnchorOpt", [r["anchoropt"]["candidate"] for r in sel]),
                   (name, [r[other]["candidate"] for r in sel])]
        label = f"AnchorOpt vs {name} ({len(sel)} cells)"
        for j, (side, sides) in enumerate(entries):
            ks = [statistics.fmean(s["pass_hat_k"][k] for s in sides) if all(k in s["pass_hat_k"] for s in sides)
                  else None for k in range(1, MAX_K + 1)]
            decline = f"{ks[0] - ks[-1]:.3f}" if ks[-1] is not None else "–"
            out.append((f"| {label} | " if j == 0 else "| | ") + f"{side} | "
                       + " | ".join("–" if v is None else f"{v:.3f}" for v in ks) + f" | {decline} |")
        diffs = []
        for k in range(1, MAX_K + 1):
            got = [paired(r[other]["candidate"], r["anchoropt"]["candidate"], k) for r in sel]
            se = math.sqrt(sum(g[1] ** 2 for g in got)) / len(got)
            diffs.append(f"{_pts(statistics.fmean(g[0] for g in got))} ({100 * se:.1f})")
        out.append(f"| | AnchorOpt − {name}, points (se) | " + " | ".join(diffs) + " | |")
    return "\n".join(out)


def markdown_reliability(rows: list[dict]) -> str:
    """Test tasks solved in all 4 trials / in some / in none, for every side with 4 trials."""
    def hist(side: dict | None) -> str:
        if side is None:
            return "ships stock"
        if side["runs"] != MAX_K:
            return "–"
        c = Counter(side["task_successes"].values())
        return f"{c[MAX_K]} / {sum(c[i] for i in range(1, MAX_K))} / {c[0]}"

    out = ["| model | domain | stock (AnchorOpt) | AnchorOpt | GEPA | Self-Harness |",
           "|---|---|---|---|---|---|"]
    for r in (r for r in rows if r["teacher"] == "self"):
        cand = [None if r[k]["candidate"] is r[k]["stock"] else r[k]["candidate"] for k, _ in FRAMEWORKS]
        out.append(f"| {r['model']} | {r['domain']} | {hist(r['anchoropt']['stock'])} | "
                   + " | ".join(hist(c) for c in cand) + " |")
    return "\n".join(out)


def _rates(sides: list[dict]) -> dict[str, float]:
    """Per-task success rate pooled over several sides measured on the same tasks."""
    n = sum(s["runs"] for s in sides)
    return {t: sum(s["task_successes"][t] for s in sides) / n for t in sorted(sides[0]["task_successes"])}


def _diff(a: dict[str, float], b: dict[str, float]) -> list[float]:
    if sorted(a) != sorted(b):
        raise SystemExit("stock comparison over different task sets")
    return [b[t] - a[t] for t in a]


def _mse(diffs: list[float]) -> str:
    return f"{_pts(statistics.fmean(diffs))} ({100 * statistics.stdev(diffs) / math.sqrt(len(diffs)):.1f})"


def markdown_stock_agreement(rows: list[dict]) -> str:
    """Every independent stock measurement of each cell, and each framework's against AnchorOpt's.

    The stock pipeline is meant to be identical in all three frameworks. GEPA measured it once per arm (its
    self-teach and sonnet-teach arms are separate runs); Self-Harness once per cell (its sonnet arms reuse
    the self-teach baseline). Differences are paired over tasks, in points (se)."""
    out = ["| model | domain | AnchorOpt (4 runs) | GEPA, self-teach arm | GEPA, sonnet-teach arm | "
           "Self-Harness (2 runs) | GEPA − AnchorOpt | Self-Harness − AnchorOpt |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    all_g, all_s = [], []
    for r in (r for r in rows if r["teacher"] == "self"):
        son = next(x for x in rows if (x["model"], x["domain"], x["teacher"]) == (r["model"], r["domain"], "sonnet"))
        ao = _rates([r["anchoropt"]["stock"]])
        dg = _diff(ao, _rates([r["gepa"]["stock"], son["gepa"]["stock"]]))
        ds = _diff(ao, _rates([r["self_harness"]["stock"]]))
        all_g += dg
        all_s += ds
        lv = [r["anchoropt"]["stock"], r["gepa"]["stock"], son["gepa"]["stock"], r["self_harness"]["stock"]]
        out.append(f"| {r['model']} | {r['domain']} | " + " | ".join(_n(100 * x["rate"], ".1f") for x in lv)
                   + f" | {_mse(dg)} | {_mse(ds)} |")
    out.append(f"| **all 9 cells** | **{len(all_g)} tasks** | | | | | **{_mse(all_g)}** | **{_mse(all_s)}** |")
    return "\n".join(out)


TABLES = (("pass^1", markdown), ("pass^1 summary", markdown_summary),
          ("mean Δpass^k against each framework's own stock", markdown_headline_k),
          ("matched cells", markdown_matched), ("always / sometimes / never", markdown_reliability),
          ("pass^k per cell", markdown_pass_hat_k), ("stock agreement", markdown_stock_agreement))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    rows = build()
    for title, fn in TABLES:
        print(f"\n## {title}\n\n{fn(rows)}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
