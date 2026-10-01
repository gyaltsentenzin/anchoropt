#!/usr/bin/env python3
"""Recompute the cycle-0 record from the run artifacts. No model, no GPU, no network.

    python benchmarks/cctu/analyze_residual.py                          # every run under results/
    python benchmarks/cctu/analyze_residual.py --run results/granite/train_baseline
    python benchmarks/cctu/analyze_residual.py --compare results/granite/train_baseline \\
                                                         results/granite/train_a1
    python benchmarks/cctu/analyze_residual.py --json rounds/CYCLE0/record.json
    python benchmarks/cctu/analyze_residual.py --expect rounds/CYCLE0/record.json   # ratchet

WHY THIS EXISTS
---------------
The cycle-0 numbers -- the residual, the variance floor, the headroom ceiling -- were first produced by
throwaway analysis at a prompt. Numbers that ground a round spec cannot live that way:
`rounds/anchors.py` is the single source of truth on the other benchmark precisely so that "prose
never restates a figure", and `scripts/verify_progression.py` recomputes every published accuracy from
per-case results and fails loudly on drift. This is that, for CCTU's control runs.

    compute     read the artifacts, derive every figure, print and emit
    --expect    compare against a frozen record and EXIT NON-ZERO on drift

WHAT IT REFUSES TO DO
---------------------
  * quote `scores.json`. Every accuracy here is recomputed from `detail.jsonl`, so the script is a
    CHECK on the scorer's aggregate rather than a re-print of it.
  * compare two runs whose decode settings or corpus differ. `COLLABORATORS.md` is explicit that
    accuracies must never be differenced across jobs, and two of that project's controls read 29.04 %
    and 24.36 % purely from corpus composition while agreeing to 1 flip in 228 on shared cases.
  * compare two models. `analyze_response.py` already records the reason: "anchors are mined from one
    model's residual and do not transfer as artifacts."
  * report a variance floor from a single replicate. `evaluation.py` says it plainly -- with
    `--repeat 1` the spread "is identically 0.00 and means nothing at all".

THE PAIRED COMPARISON IS NOT REIMPLEMENTED HERE. `--compare` calls
`scripts/verify_progression.sign_test`, which is the shape `COLLABORATORS.md` says to read rather than
rewrite. A second implementation of a sign test is a second thing that can disagree.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for _p in (str(HERE), str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cctu_adapter as cctu  # noqa: E402
import cctu_signals as sig  # noqa: E402
import cctu_state as state_mod  # noqa: E402

W = 96
METRICS = ("acc", "SR", "PSR")

# The two terminal states that both mean "the round budget ran out", and they are NOT
# interchangeable. `mid_tool_call` is the budget expiring on a TOOL turn -- the last message is a tool
# result, and `compute_if_flags` short-circuits to (1, 1), forcing SR and PSR to 0 before any other
# check runs. `ended_on_user` is the budget expiring on a FINAL-ANSWER turn, with the validator's
# feedback last. An earlier decomposition keyed on `mid_tool_call` alone and therefore filed most of
# one model's budget failures into the wrong classes.
BUDGET_TERMINALS = ("mid_tool_call", "ended_on_user")


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def _scoring_disagrees(run: Path, manifest: dict, transcripts: dict, detail: list) -> str | None:
    """Does `detail.jsonl` actually describe THESE transcripts? Returns a reason, or None.

    Authoritative test: re-derive each verdict with `evaluation.judge` -- upstream's own scorer, loaded
    by path so a module named `evaluation` on some other sys.path cannot be picked up instead -- and
    compare. The corpus comes from the manifest, which is why the manifest records it.

    Falls back to comparing mtimes when the corpus is unreachable. That is weaker (a file copy
    reorders mtimes) but it is better than answering "looks fine" for a reason that amounts to not
    having looked.
    """
    corpus = manifest.get("corpus") or {}
    data_dir, input_file = corpus.get("input_dir"), corpus.get("input_file")
    samples: dict[str, dict] = {}
    if data_dir and input_file and (Path(data_dir) / input_file).exists():
        with open(Path(data_dir) / input_file, encoding="utf-8") as fh:
            samples = {json.loads(line)["id"]: json.loads(line) for line in fh}

    if not samples:
        a, b = run / "detail.jsonl", run / "response.jsonl"
        if a.exists() and b.exists() and a.stat().st_mtime < b.stat().st_mtime - 1:
            return (f"STALE SCORING (by mtime; the corpus at {data_dir!r} was not reachable so the "
                    f"verdicts could not be re-derived): detail.jsonl predates response.jsonl. "
                    f"Re-score with evaluation.py --detail --overload before trusting any figure.")
        return None

    import importlib.util
    spec = importlib.util.spec_from_file_location("cctu_evaluation", HERE / "evaluation.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    stored = {r["id"]: r for r in detail}
    compared = disagree = 0
    for episode_id, messages in transcripts.items():
        query = episode_id.split("_")[0]
        if query not in samples or episode_id not in stored:
            continue
        compared += 1
        fresh = ev.judge(messages, samples[query])
        if any(fresh[m] != stored[episode_id][m] for m in (*METRICS, "termination")):
            disagree += 1
    if disagree:
        return (f"STALE SCORING: the stored verdict disagrees with upstream's own judge() on "
                f"{disagree} of {compared} episodes ({_pct(disagree, compared)}%). detail.jsonl "
                f"describes a PREVIOUS job -- a completed re-run that was never re-scored, which an "
                f"id-set check cannot see because the ids are identical. Nothing below is a result; "
                f"re-score with evaluation.py --detail --overload.")
    return None


# ------------------------------------------------------------------------------------------------
# one run
# ------------------------------------------------------------------------------------------------
def analyze_run(run: Path) -> dict:
    """Every cycle-0 figure for one run directory, derived from its own artifacts."""
    out: dict = {"run": str(run), "problems": []}

    manifest_path = run / "run_manifest.json"
    if not manifest_path.exists():
        out["problems"].append("no run_manifest.json: the run's decode settings, corpus and policy "
                               "are unrecorded, so it cannot be reproduced or compared")
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text())
    out["manifest"] = {k: manifest.get(k) for k in
                       ("model", "model_dir", "split", "repeat", "config", "provider",
                        "max_workers", "episodes", "endpoint", "endpoints")}
    out["decode"] = manifest.get("decode", {})
    out["corpus_sha256"] = (manifest.get("corpus") or {}).get("sha256")
    out["controllers"] = manifest.get("controllers", {})
    if out["decode"] and not out["decode"].get("deterministic"):
        out["problems"].append("decode is not pinned to a deterministic setting, so this run cannot "
                               "support a variance-floor claim")

    detail = _jsonl(run / "detail.jsonl")
    transcripts = {r["id"]: r["messages"] for r in _jsonl(run / "response.jsonl")}
    out["episodes_scored"] = len(detail)
    out["transcripts"] = len(transcripts)
    planned = manifest.get("episodes")
    if planned and len(transcripts) < planned:
        out["problems"].append(
            f"{planned - len(transcripts)} of {planned} episodes produced no transcript. A dropped "
            f"episode leaves the DENOMINATOR, which is worse than a wrong score because it is "
            f"invisible -- see ANCHOROPT.md on the validator's KeyError path")
    if not detail:
        out["problems"].append("no detail.jsonl: re-score with evaluation.py --detail. Nothing "
                               "per-episode can be computed without it, including the floor and any "
                               "paired delta")
        return out

    # THE SCORING MUST DESCRIBE THIS RUN'S TRANSCRIPTS, and a partial `response.jsonl` beside a
    # complete `detail.jsonl` means it does not. That is what an IN-PROGRESS re-run looks like on
    # disk: `--overload` truncates the transcripts and rebuilds them while the previous scoring is
    # still sitting there, so every figure below would mix a stale aggregate with a partial residual.
    # Caught here rather than reported as drift, because "the numbers moved" and "the numbers are
    # from two different runs" call for completely different responses -- and only one of them is a
    # result.
    orphans = sorted(r["id"] for r in detail if r["id"] not in transcripts)
    if orphans:
        out["problems"].append(
            f"STALE SCORING: {len(orphans)} scored episode(s) have no transcript in this run "
            f"(e.g. {orphans[:3]}). detail.jsonl describes a DIFFERENT run than response.jsonl -- "
            f"the signature of a run still in progress under --overload. Nothing below is a result; "
            f"wait for the run to finish and re-score before freezing or comparing.")
        out["stale_scoring"] = True
        return out

    # AN ID-SET CHECK IS NOT ENOUGH, and this is the second version of this guard.
    #
    # The first compared the scored ids against the transcript ids, which catches a run still IN
    # PROGRESS and misses the more dangerous case: a re-run that COMPLETED without being re-scored.
    # The episode ids are then identical, the subset test passes, and every figure silently describes
    # the previous job -- which is exactly what happened here. A re-run's variance floor came back
    # bit-for-bit identical to the run it was meant to improve on, and only the implausibility of that
    # coincidence gave it away.
    #
    # So the verdicts are re-derived with UPSTREAM'S OWN `judge()` and compared. Not a fork: the same
    # function `evaluation.py` scores with, called on the transcripts actually on disk.
    stale = _scoring_disagrees(run, manifest, transcripts, detail)
    if stale:
        out["problems"].append(stale)
        out["stale_scoring"] = True
        return out

    # ---- replicates -----------------------------------------------------------------------------
    by_query: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for row in detail:
        query, _, replicate = row["id"].rpartition("_")
        by_query[query][replicate] = row
    out["queries"] = len(by_query)
    out["replicates"] = max((len(v) for v in by_query.values()), default=0)

    # ---- accuracies, RECOMPUTED per replicate ---------------------------------------------------
    per_replicate: dict[str, dict[str, float]] = {}
    for rep in sorted({r for v in by_query.values() for r in v}):
        rows = [v[rep] for v in by_query.values() if rep in v]
        per_replicate[rep] = {m: _pct(sum(1 for r in rows if r[m]), len(rows)) for m in METRICS}
        per_replicate[rep]["n"] = len(rows)
    out["per_replicate"] = per_replicate
    out["mean"] = {m: round(sum(v[m] for v in per_replicate.values()) / len(per_replicate), 2)
                   for m in METRICS} if per_replicate else {}

    # ---- THE VARIANCE FLOOR, per episode --------------------------------------------------------
    paired = {q: v for q, v in by_query.items() if len(v) >= 2}
    if len(paired) == 0 or out["replicates"] < 2:
        out["variance_floor"] = {"measured": False,
                                 "why": "a single replicate measures nothing; re-run with --repeat 2"}
    else:
        reps = sorted({r for v in paired.values() for r in v})[:2]
        flips = {m: sorted(q for q, v in paired.items() if v[reps[0]][m] != v[reps[1]][m])
                 for m in (*METRICS, "termination")}
        out["variance_floor"] = {
            "measured": True, "paired_queries": len(paired), "replicates_compared": reps,
            **{m: {"flips": len(v), "rate_pct": _pct(len(v), len(paired)), "queries": v[:12]}
               for m, v in flips.items()},
            "zero": all(not v for m, v in flips.items() if m in METRICS),
        }
        if not out["variance_floor"]["zero"]:
            worst = max(_pct(len(v), len(paired)) for m, v in flips.items() if m in METRICS)
            out["problems"].append(
                f"the variance floor is NOT zero (worst metric {worst}% of episodes flip between two "
                f"runs of one policy). Every harm tolerance widens accordingly, and a delta smaller "
                f"than this is noise rather than an effect")

    # ---- failure classes, on the first replicate ------------------------------------------------
    first = min({r for v in by_query.values() for r in v})
    rows = [v[first] for v in by_query.values() if first in v]
    classes = collections.Counter()
    terminations = collections.Counter()
    for r in rows:
        terminations[r["termination"]] += 1
        if r["termination"] in BUDGET_TERMINALS:
            classes[f"3_budget_exhausted_{r['termination']}"] += 1
        elif not r["acc"]:
            classes["1_answer_not_obtained"] += 1
        elif not r["PSR"]:
            classes["2_answered_but_violated"] += 1
        else:
            classes["0_clean_psr"] += 1
    out["replicate_analysed"] = first
    out["failure_classes"] = {k: {"n": v, "pct": _pct(v, len(rows))}
                              for k, v in sorted(classes.items())}
    out["terminations"] = dict(terminations.most_common())

    # ---- THE HEADROOM CEILING -------------------------------------------------------------------
    # Budget-exhausted episodes whose answer WAS retrieved are the only ones a budget-shaped
    # intervention can convert: SR and PSR are forced to 0 for them, and nothing else is wrong. The
    # rest have a deeper problem that no suppression reaches, so this is a CEILING on that class of
    # anchor and it belongs in a round spec before an arm runs.
    budget = [r for r in rows if r["termination"] in BUDGET_TERMINALS]
    retrieved = [r for r in budget if r["acc"]]
    psr_now = sum(1 for r in rows if r["PSR"])
    out["headroom"] = {
        "budget_exhausted": len(budget),
        "of_which_answer_retrieved": len(retrieved),
        "psr_now": psr_now, "psr_now_pct": _pct(psr_now, len(rows)),
        "psr_ceiling_if_all_landed": psr_now + len(retrieved),
        "psr_ceiling_pct": _pct(psr_now + len(retrieved), len(rows)),
        "gain_pp": round(_pct(psr_now + len(retrieved), len(rows)) - _pct(psr_now, len(rows)), 2),
    }

    # ---- the residual, via the adapter's own classifiers ----------------------------------------
    out.update(_mine(transcripts, {r["id"]: r for r in rows}, first))
    return out


def _repeated_calls_in(post_events: list[dict]) -> int:
    """Repeated identical tool calls across one episode's `post_execution` events.

    The one owner of this count: `_mine` reads it for one replicate of one run, and `compare`
    (below) reads it per episode over `shared` -- CCTU_QWEN1 read this as five single-run
    aggregates by eye, because nothing diffed it control vs arm.
    """
    seen_calls: set[tuple[str, str]] = set()
    repeats = 0
    for event in post_events:
        for key in state_mod.call_keys(event["tool_calls_raw"]):
            if key in seen_calls:
                repeats += 1
            seen_calls.add(key)
    return repeats


def _repeated_calls_by_episode(transcripts: dict[str, list]) -> dict[str, int]:
    """`_repeated_calls_in`, for every episode in `transcripts` -- no replicate filter.

    `_mine` restricts to one replicate because a single-run survey does not need the rest; a paired
    delta over `shared` needs every episode id `detail.jsonl` might carry, from any replicate.
    """
    out: dict[str, int] = {}
    for eid, messages in transcripts.items():
        post = [e for e in cctu.events_from_messages(messages)
                if cctu.boundary_key(e) == "post_execution"]
        out[eid] = _repeated_calls_in(post)
    return out


def _mine(transcripts: dict[str, list], scored: dict[str, dict], replicate: str) -> dict:
    """The lexical residual and the loop shape, read through `cctu_adapter`'s classifiers.

    Deliberately routed through the adapter rather than a local regex: the classifier that names a
    violation for the RESIDUAL must be the one that names it for a SIGNAL, or a candidate would fire
    on a different population than the mining ranked.

    THESE COUNTS ARE FEEDBACK-ONLY, AND THAT IS CORRECT -- do not "fix" them upward to match a
    `detail.jsonl` written before 2026-09-23. This block reads violations through
    `cctu_signals.violation_classes_in`, which reads upstream's two real validator channels
    (`role: "tool"` on non-final turns, one merged `role: "user"` on final turns). `evaluation.py`'s
    `violation_counts` used to read EVERY message, including the model's own turns, and the model quotes
    the validator's error text back while reasoning -- so a `detail.jsonl` from before that fix carries
    the model's echoes as violations. On `results/qwen/train_baseline` that is 840 counted against 786
    real. `evaluation.py` now skips `assistant`; see `VENDORED.md`. A gap remaining between this block
    and a re-scored `detail.jsonl` is a real defect and should be chased, but the gap against an
    un-re-scored one is expected.
    """
    violations = collections.Counter()
    violations_in_budget = collections.Counter()
    turns_hist = collections.Counter()
    faults = vacuous = 0
    episodes_with_violation = 0
    first_violation_turn: list[int] = []
    repeats = 0
    episodes_with_repeat = collections.Counter()
    at_cap = 0

    sig.UNDECLARED_CLASSES_SEEN.clear()
    for eid, messages in transcripts.items():
        if not eid.endswith(f"_{replicate}") or eid not in scored:
            continue
        row = scored[eid]
        budget = row["termination"] in BUDGET_TERMINALS
        post = [e for e in cctu.events_from_messages(messages)
                if cctu.boundary_key(e) == "post_execution"]
        turns_hist[len(post)] += 1
        at_cap += int(len(post) >= 20)

        had_violation = False
        for index, event in enumerate(post):
            for cls in event["violation_classes"]:
                violations[cls] += 1
                if budget:
                    violations_in_budget[cls] += 1
                if not had_violation:
                    first_violation_turn.append(index)
                    had_violation = True
            faults += int(event["result_is_error"])
            vacuous += int(event["result_is_empty"])
        episodes_with_violation += int(had_violation)
        episode_repeats = _repeated_calls_in(post)
        repeats += episode_repeats
        if episode_repeats:
            episodes_with_repeat[row["termination"]] += 1

    total = sum(violations.values())
    n = sum(turns_hist.values())
    budget_repeats = sum(v for k, v in episodes_with_repeat.items() if k in BUDGET_TERMINALS)
    return {
        "residual": {
            "episodes": n,
            "episodes_with_a_violation": episodes_with_violation,
            "violations_total": total,
            "by_class": {c: {"n": k, "pct": _pct(k, total),
                             "in_budget_exhausted_episodes": violations_in_budget[c]}
                         for c, k in violations.most_common()},
            # A class upstream emits that this build does not declare. Non-empty means upstream drift,
            # and it must not be read as "no such violations occurred".
            "undeclared_classes_seen": sorted(sig.UNDECLARED_CLASSES_SEEN),
            "median_first_violation_turn": (sorted(first_violation_turn)[len(first_violation_turn) // 2]
                                            if first_violation_turn else None),
        },
        "loop_shape": {
            # STRING keys: JSON has no integer keys, so an int-keyed histogram would
            # come back from `--expect` as strings and every bucket would read as drift.
            "turns_per_episode": {str(k): v for k, v in sorted(turns_hist.items())},
            "episodes_at_the_round_cap": at_cap,
            "repeated_identical_calls": repeats,
            "episodes_with_a_repeat_by_termination": dict(episodes_with_repeat.most_common()),
            "budget_exhausted_episodes_with_a_repeat": budget_repeats,
        },
        # Kept separate from the residual ON PURPOSE. A tool that raised or timed out is a
        # harness/corpus fault, not a model failure, and `attribution/harness_guard.py` exists because
        # one got mined as a real error contract at support 13 on the other benchmark.
        "harness_noise": {"tool_faults": faults, "vacuous_results": vacuous},
    }


# ------------------------------------------------------------------------------------------------
# two runs
# ------------------------------------------------------------------------------------------------
def compare(control: dict, arm: dict, control_dir: Path, arm_dir: Path) -> dict:
    """A paired delta over the SAME episodes, refusing every comparison that is not licensed."""
    from verify_progression import sign_test

    out: dict = {"control": str(control_dir), "arm": str(arm_dir), "problems": []}

    for field, why in (("model_dir", "two models' residuals do not transfer as artifacts"),
                       ("split", "a delta across splits is not a delta")):
        a, b = control["manifest"].get(field), arm["manifest"].get(field)
        if a != b:
            out["problems"].append(f"{field} differs ({a!r} vs {b!r}): {why}")
    if control.get("corpus_sha256") != arm.get("corpus_sha256"):
        out["problems"].append(
            "the corpus differs between the arms, so any delta is partly composition. Two controls "
            "on the other benchmark read 29.04% and 24.36% for exactly this reason while agreeing to "
            "1 flip in 228 on shared cases")
    for key in ("temperature", "seed", "top_p"):
        a, b = control.get("decode", {}).get(key), arm.get("decode", {}).get(key)
        if a != b:
            out["problems"].append(f"decode.{key} differs ({a!r} vs {b!r}): not a paired comparison")
    # THE SERVING ENDPOINT IS PART OF THE DECODE CONTRACT on a served model, and it was the one part
    # nothing checked. A qwen sweep split across three cluster nodes while `run_vllm.py`'s default
    # moved, and this function passed it: the decode block was identical because the endpoint was not
    # in one. A different node can run a different build, quantisation or tool-call parser, which
    # changes generations without changing temperature or seed.
    #
    # `None` on either side means the run predates the field, which is UNKNOWN rather than equal --
    # reported as a caveat instead of a veto, because the frozen controls were recorded before it
    # existed and silently passing them would be the same defect one level up.
    ep_a, ep_b = control["manifest"].get("endpoint"), arm["manifest"].get("endpoint")
    if ep_a is None or ep_b is None:
        if (control["manifest"].get("provider") == "vllm"
                or arm["manifest"].get("provider") == "vllm"):
            out["problems"].append(
                f"endpoint NOT RECORDED for one side (control={ep_a!r}, arm={ep_b!r}): this pairing "
                f"cannot be shown to have used one server. Re-run the missing side, or state the "
                f"caveat in the round")
    elif ep_a != ep_b:
        out["problems"].append(
            f"endpoint differs ({ep_a} vs {ep_b}): not a paired comparison. A served model's node is "
            f"part of its decode contract")
    # A SPLIT RUN is reported even when the last endpoints agree, because `endpoint` names only the
    # invocation that wrote the manifest. An arm finished on a second node after an allocation ended is
    # a legitimate artifact -- episodes share no state -- but it is not a single-server run, and a
    # reader comparing the two `endpoint` fields would conclude that it was. Reported rather than
    # vetoed: whether the split moved anything is what `check_pairing.py` measures, on the episodes
    # where the controller never fired.
    for side, m in (("control", control["manifest"]), ("arm", arm["manifest"])):
        served = list(m.get("endpoints") or [])
        if len(served) > 1:
            out["problems"].append(
                f"the {side} was served by {len(served)} endpoints ({' -> '.join(served)}): a split "
                f"run, so `endpoint` names only the last. Clear it with check_pairing.py before "
                f"reading any delta from this pairing")
    if arm.get("controllers", {}).get("sha256") is None:
        out["problems"].append("the arm installed no controller, so it IS the control under another "
                               "name; nothing here would measure an intervention")

    detail_a = {r["id"]: r for r in _jsonl(control_dir / "detail.jsonl")}
    detail_b = {r["id"]: r for r in _jsonl(arm_dir / "detail.jsonl")}
    shared = sorted(set(detail_a) & set(detail_b))
    out["episodes_shared"] = len(shared)
    if set(detail_a) != set(detail_b):
        out["problems"].append(
            f"the denominators differ: {len(set(detail_a) - set(detail_b))} only in the control, "
            f"{len(set(detail_b) - set(detail_a))} only in the arm. A delta over different case sets "
            f"is not a paired result")
    if not shared:
        return out

    out["metrics"] = {}
    for metric in METRICS:
        before = {k: bool(detail_a[k][metric]) for k in shared}
        after = {k: bool(detail_b[k][metric]) for k in shared}
        gains, losses, p = sign_test(before, after)
        out["metrics"][metric] = {
            "control_pct": _pct(sum(before.values()), len(shared)),
            "arm_pct": _pct(sum(after.values()), len(shared)),
            "delta_pp": round(_pct(sum(after.values()), len(shared))
                              - _pct(sum(before.values()), len(shared)), 2),
            "gains": gains, "losses": losses, "net": gains - losses,
            "p_two_sided": p,
        }

    # ---- the COUNT layer, and the two denominators that decide whether a drop is a repair --------
    #
    # `SR` and `PSR` are booleans, so an intervention that takes an episode from 12 violations to 2
    # scores identically to one that does nothing -- and PSR needs exactly 0, so that is measurable
    # progress toward the benchmark's own metric with no way to state it. Added here as a PAIRED NET
    # DIFFERENCE, which is the right statistic for a count; a flip count is right for a boolean and
    # wrong for this.
    #
    # The denominators are reported in the same block and not separately, because the number is
    # uninterpretable without them: the environment reprompts on every violation, so a stuck episode
    # re-emits its class every turn and the raw count is largely episode LENGTH. A drop that arrives
    # with fewer rounds used or fewer tools run is a WITHDRAWAL, not a repair.
    if all("n_violations" in detail_a[k] for k in shared[:1]) and \
       all("n_violations" in detail_b[k] for k in shared[:1]):
        classes = sorted({c for k in shared
                          for c in (*(detail_a[k].get("violations") or {}),
                                    *(detail_b[k].get("violations") or {}))})
        counts: dict = {}
        for name in ("n_violations", *(f"violations.{c}" for c in classes)):
            def _val(row, field=name):
                if field == "n_violations":
                    return int(row.get("n_violations") or 0)
                return int((row.get("violations") or {}).get(field.split(".", 1)[1], 0))
            before = sum(_val(detail_a[k]) for k in shared)
            after = sum(_val(detail_b[k]) for k in shared)
            moved = [k for k in shared if _val(detail_a[k]) != _val(detail_b[k])]
            counts[name] = {
                "control": before, "arm": after, "net": after - before,
                "episodes_changed": len(moved),
                # Sign test over the episodes that moved at all, so a corpus total driven by one
                # runaway episode is distinguishable from a broad shift.
                "episodes_down": sum(1 for k in moved if _val(detail_b[k]) < _val(detail_a[k])),
                "episodes_up": sum(1 for k in moved if _val(detail_b[k]) > _val(detail_a[k])),
            }
        work = {}
        for field in ("assistant_turns", "tool_results"):
            before = sum(int(detail_a[k].get(field) or 0) for k in shared)
            after = sum(int(detail_b[k].get(field) or 0) for k in shared)
            work[field] = {"control": before, "arm": after, "net": after - before}
        out["counts"] = counts
        out["work"] = work

        # `repeated_identical_calls`: `_mine`'s own version is a single-run aggregate for one
        # replicate, never diffed control vs arm. CCTU_QWEN1 read five such aggregates by eye across
        # five run directories (control 85, arms 164-299) because nothing paired them over `shared`.
        # Same treatment as `counts`, above, and the same denominator-integrity reason for it.
        repeats_a = _repeated_calls_by_episode(
            {r["id"]: r["messages"] for r in _jsonl(control_dir / "response.jsonl")})
        repeats_b = _repeated_calls_by_episode(
            {r["id"]: r["messages"] for r in _jsonl(arm_dir / "response.jsonl")})
        if all(k in repeats_a for k in shared) and all(k in repeats_b for k in shared):
            before = sum(repeats_a[k] for k in shared)
            after = sum(repeats_b[k] for k in shared)
            moved = [k for k in shared if repeats_a[k] != repeats_b[k]]
            out["repeated_identical_calls"] = {
                "control": before, "arm": after, "net": after - before,
                "episodes_changed": len(moved),
                "episodes_down": sum(1 for k in moved if repeats_b[k] < repeats_a[k]),
                "episodes_up": sum(1 for k in moved if repeats_b[k] > repeats_a[k]),
            }
        else:
            out["problems"].append(
                "repeated_identical_calls could not be computed for every shared episode (an id in "
                "detail.jsonl has no matching transcript in response.jsonl on one side); skipped")

        # THE WITHDRAWAL VETO. Stated as a problem rather than left for a reader to notice, because a
        # violation drop is the most flattering number in this block and the cheapest way to get it
        # is to do less.
        dropped = counts["n_violations"]["net"] < 0
        less_work = [f for f, v in work.items() if v["net"] < 0]
        if dropped and less_work:
            out["problems"].append(
                f"WITHDRAWAL, not repair: violations fell by {-counts['n_violations']['net']} but "
                f"{', '.join(less_work)} also fell "
                f"({', '.join(f'{f} {work[f]['net']:+d}' for f in less_work)}). A count that drops "
                f"because the agent did less is not a repair; judge the RATE, and check `acc`.")

        # `died_of_exhaustion`: a boolean, so a flip delta is correct for it.
        if all("died_of_exhaustion" in detail_a[k] for k in shared[:1]):
            gains = [k for k in shared if detail_a[k]["died_of_exhaustion"]
                     and not detail_b[k]["died_of_exhaustion"]]
            losses = [k for k in shared if detail_b[k]["died_of_exhaustion"]
                      and not detail_a[k]["died_of_exhaustion"]]
            out["termination_delta"] = {
                "control_died": sum(1 for k in shared if detail_a[k]["died_of_exhaustion"]),
                "arm_died": sum(1 for k in shared if detail_b[k]["died_of_exhaustion"]),
                "recovered": len(gains), "newly_exhausted": len(losses),
                "net_recovered": len(gains) - len(losses),
                # The caveat travels with the number, because it is the difference between an effect
                # and a restatement of the mechanism.
                "valid_as_benefit_only_if": ("the anchor's mechanism does not act on termination; an "
                                             "anchor that forces an answer converts mid_tool_call by "
                                             "construction and this number then measures whether it "
                                             "fired, not whether it helped"),
            }

    # The floor is the interpretability bar for every delta above it, so it is carried into the
    # comparison rather than left in the run report for a reader to remember.
    floor = control.get("variance_floor", {})
    out["control_variance_floor"] = {
        m: floor.get(m, {}).get("flips") for m in METRICS} if floor.get("measured") else None
    if floor.get("measured"):
        for metric, cell in out["metrics"].items():
            bar = floor.get(metric, {}).get("flips", 0)
            if abs(cell["net"]) <= bar:
                out["problems"].append(
                    f"{metric}: net {cell['net']:+d} is within the control's own variance floor "
                    f"({bar} flips), so it is not distinguishable from noise")

    # ENGAGEMENT, READ FROM THE ACCUMULATED TRACE rather than from the summary.
    #
    # `anchor_telemetry.json` is written once per INVOCATION and overwritten, so a resumed run reports
    # only its last invocation. Cycle 1 hit this: an arm reported `exposures=60, executed=0` and looked
    # broken, while its trace held 9826 rows over 280 episodes with 6 executions. The trace APPENDS, so
    # it is the only source that survives a resume -- the same reason the other benchmark's loop is
    # required to read firings from the sidecar and not from a summary dict.
    trace = _jsonl(arm_dir / "anchor_trace.jsonl")
    summary_path = arm_dir / "anchor_telemetry.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if trace:
        executed = [r for r in trace if r.get("executed")]
        fired = [r for r in trace if r.get("fired")]
        out["arm_engagement"] = {
            "source": "anchor_trace.jsonl (accumulated)",
            "trace_rows": len(trace), "episodes_seen": len({r["case_id"] for r in trace}),
            "fired": len(fired), "executed": len(executed),
            "episodes_intervened": len({r["case_id"] for r in executed}),
            "controllers": sorted({r["controller"] for r in trace if r.get("controller")}),
        }
        if summary and summary.get("boundary_exposures", 0) < len(trace):
            out["problems"].append(
                f"anchor_telemetry.json reports {summary.get('boundary_exposures')} exposures against "
                f"{len(trace)} trace rows: the summary is per-invocation and this run was RESUMED. "
                f"Engagement is read from the trace; do not quote the summary.")
        if not executed:
            out["problems"].append("the arm executed 0 interventions. READ THIS BEFORE THE ACCURACY: "
                                   "an arm that never fires is the control wearing an arm's name")
    else:
        out["problems"].append("no anchor_trace.jsonl for the arm: engagement must be checked before "
                               "any accuracy, and nothing here can check it. Re-run with "
                               "--anchor-trace")

    # ATTRIBUTION. Gains outside the intervened population are not the intervention's gains -- the
    # defect shape `COLLABORATORS.md` records as "a winner passed all four terms while firing 12x on a
    # different backend's error contract". Cycle 1's best arm had 3 of 3 PSR gains in episodes it never
    # touched, and only the harm was attributable. Computed here so the veto cannot be skipped.
    if trace and out.get("metrics"):
        intervened = {r["case_id"] for r in trace if r.get("executed")}
        out["attribution"] = {}
        for metric in METRICS:
            gains = [k for k in shared if detail_b[k][metric] and not detail_a[k][metric]]
            losses = [k for k in shared if detail_a[k][metric] and not detail_b[k][metric]]
            out["attribution"][metric] = {
                "gains": len(gains), "gains_in_intervened": sum(1 for k in gains if k in intervened),
                "losses": len(losses),
                "losses_in_intervened": sum(1 for k in losses if k in intervened),
            }
        psr = out["attribution"]["PSR"]
        if psr["gains"] and psr["gains_in_intervened"] == 0:
            out["problems"].append(
                f"ATTRIBUTION VETO: all {psr['gains']} PSR gain(s) are in episodes the controller "
                f"never executed in. A gain outside the intervened population is job-to-job noise, "
                f"not an effect -- and engagement passing does not cover this.")
    return out


# ------------------------------------------------------------------------------------------------
# printing
# ------------------------------------------------------------------------------------------------
def print_run(r: dict) -> None:
    m = r["manifest"]
    print("=" * W)
    print(f"{r['run']}")
    print("=" * W)
    print(f"  model={m.get('model')}  split={m.get('split')}  repeat={m.get('repeat')}  "
          f"config={m.get('config')}  provider={m.get('provider')}  workers={m.get('max_workers')}")
    d = r.get("decode") or {}
    print(f"  decode: temperature={d.get('temperature')} seed={d.get('seed')} top_p={d.get('top_p')}"
          f"  deterministic={d.get('deterministic')}")
    print(f"  corpus={str(r.get('corpus_sha256'))[:23]}  transcripts={r['transcripts']}  "
          f"scored={r['episodes_scored']}")
    if not r.get("queries"):
        for p in r["problems"]:
            print(f"  [!] {p}")
        return

    print(f"\n  ACCURACY (recomputed from detail.jsonl, {r['queries']} queries x "
          f"{r['replicates']} replicates)")
    for metric in METRICS:
        cells = "  ".join(f"rep{k}={v[metric]:>6.2f}" for k, v in sorted(r["per_replicate"].items()))
        print(f"    {metric:4} mean={r['mean'][metric]:>6.2f}   {cells}")

    floor = r["variance_floor"]
    print("\n  VARIANCE FLOOR (per-episode flips between two runs of ONE policy)")
    if not floor.get("measured"):
        print(f"    NOT MEASURED -- {floor['why']}")
    else:
        for metric in (*METRICS, "termination"):
            cell = floor[metric]
            print(f"    {metric:12} {cell['flips']:3d} flips  ({cell['rate_pct']:5.2f}% of "
                  f"{floor['paired_queries']})")
        print(f"    -> {'ZERO' if floor['zero'] else 'NON-ZERO: a delta below this is noise'}")

    print("\n  FAILURE CLASSES (replicate " + r["replicate_analysed"] + ")")
    for name, cell in r["failure_classes"].items():
        print(f"    {name:36} {cell['n']:4d}  ({cell['pct']:5.2f}%)")

    h = r["headroom"]
    print("\n  HEADROOM for a budget-shaped intervention")
    print(f"    budget-exhausted={h['budget_exhausted']}  of which the answer WAS retrieved="
          f"{h['of_which_answer_retrieved']}")
    print(f"    PSR now={h['psr_now']} ({h['psr_now_pct']}%)  ceiling if those landed cleanly="
          f"{h['psr_ceiling_if_all_landed']} ({h['psr_ceiling_pct']}%)  -> +{h['gain_pp']} pp max")

    res, loop = r["residual"], r["loop_shape"]
    print(f"\n  RESIDUAL: {res['violations_total']} violations over {res['episodes']} episodes; "
          f"{res['episodes_with_a_violation']} episodes carry at least one")
    for cls, cell in list(res["by_class"].items())[:8]:
        print(f"    {cls:24} {cell['n']:5d}  ({cell['pct']:5.2f}%)   in budget-exhausted: "
              f"{cell['in_budget_exhausted_episodes']}")
    if res["undeclared_classes_seen"]:
        print(f"    [!] UNDECLARED classes seen: {res['undeclared_classes_seen']} -- upstream drift")
    print(f"    median first-violation turn={res['median_first_violation_turn']}   "
          f"episodes at the round cap={loop['episodes_at_the_round_cap']}")
    print(f"    repeated identical calls={loop['repeated_identical_calls']}   "
          f"budget-exhausted episodes with a repeat="
          f"{loop['budget_exhausted_episodes_with_a_repeat']}")
    noise = r["harness_noise"]
    print(f"    harness noise (NOT residual): tool_faults={noise['tool_faults']} "
          f"vacuous={noise['vacuous_results']}")
    for p in r["problems"]:
        print(f"\n  [!] {p}")


def print_comparison(c: dict) -> None:
    print("=" * W)
    print(f"PAIRED: {c['control']}  ->  {c['arm']}")
    print("=" * W)
    print(f"  episodes shared: {c['episodes_shared']}")
    for metric, cell in (c.get("metrics") or {}).items():
        p = "n/a" if cell["p_two_sided"] is None else f"{cell['p_two_sided']:.4f}"
        print(f"    {metric:4} {cell['control_pct']:6.2f} -> {cell['arm_pct']:6.2f}  "
              f"({cell['delta_pp']:+.2f} pp)   {cell['gains']}g / {cell['losses']}l  "
              f"net={cell['net']:+d}  p={p}")
    if c.get("arm_telemetry"):
        t = c["arm_telemetry"]
        print(f"  arm telemetry: exposures={t['boundary_exposures']} firings={t['signal_firings']} "
              f"executed={t['interventions_executed']}")
    for p in c["problems"]:
        print(f"  [!] {p}")


# ------------------------------------------------------------------------------------------------
def _drift(expected, got, path="") -> list[str]:
    """Every leaf that changed, named by its path. Volatile keys are excluded by the caller."""
    if isinstance(expected, dict) and isinstance(got, dict):
        out = []
        for k in sorted(set(expected) | set(got), key=str):
            if k not in expected:
                continue                      # a new key is an addition, not drift
            if k not in got:
                out.append(f"{path}{k}: MISSING")
                continue
            out += _drift(expected[k], got[k], f"{path}{k}.")
        return out
    if isinstance(expected, float) and isinstance(got, (int, float)):
        return [] if math.isclose(expected, got, rel_tol=0, abs_tol=0.005) else \
            [f"{path.rstrip('.')}: {expected} -> {got}"]
    return [] if expected == got else [f"{path.rstrip('.')}: {expected!r} -> {got!r}"]


# Excluded from the drift check because they are properties of WHERE the analysis ran, not of the
# result. A frozen record that fails because someone moved the checkout is a ratchet nobody keeps.
_VOLATILE = ("run", "control", "arm")


def main() -> int:
    ap = argparse.ArgumentParser(description="Recompute CCTU's cycle-0 record from run artifacts.")
    ap.add_argument("--results-root", default=str(HERE / "results"))
    ap.add_argument("--run", action="append", default=[],
                    help="a run directory; repeatable. Omitted = discover every run under the root")
    ap.add_argument("--compare", nargs=2, metavar=("CONTROL", "ARM"),
                    help="paired delta over the shared episodes, with every licence checked")
    ap.add_argument("--json", dest="json_out", default=None, help="write the record here")
    ap.add_argument("--expect", default=None,
                    help="compare against a frozen record and exit non-zero on drift")
    args = ap.parse_args()

    runs = [Path(p) for p in args.run]
    if not runs:
        root = Path(args.results_root)
        runs = sorted(p.parent for p in root.glob("*/*/run_manifest.json")) if root.is_dir() else []
    if not runs and not args.compare:
        print(f"[error] no runs found under {args.results_root}; pass --run <dir>", file=sys.stderr)
        return 1

    record: dict = {"runs": [], "comparisons": []}
    for run in runs:
        report = analyze_run(run)
        record["runs"].append(report)
        print_run(report)
        print()

    if args.compare:
        a, b = Path(args.compare[0]), Path(args.compare[1])
        comparison = compare(analyze_run(a), analyze_run(b), a, b)
        record["comparisons"].append(comparison)
        print_comparison(comparison)
        print()

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
        print(f"[record] {out}")

    problems = sum(len(r["problems"]) for r in record["runs"]) \
        + sum(len(c["problems"]) for c in record["comparisons"])

    if args.expect:
        expected = json.loads(Path(args.expect).read_text())
        drift = []
        for want, got in zip(expected.get("runs", []), record["runs"]):
            drift += _drift({k: v for k, v in want.items() if k not in _VOLATILE},
                            {k: v for k, v in got.items() if k not in _VOLATILE})
        print("=" * W)
        if drift:
            print(f"DRIFT against {args.expect} ({len(drift)}):")
            for line in drift[:40]:
                print(f"  {line}")
            print("=" * W)
            return 1
        print(f"NO DRIFT against {args.expect}")
        print("=" * W)

    print("=" * W)
    print(f"{len(record['runs'])} run(s) analysed, {problems} problem(s) reported")
    if problems:
        print("A problem is not necessarily a defect -- a non-zero variance floor is a FACT about the")
        print("run, not a bug. It is reported so a round spec states it rather than assuming it away.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
