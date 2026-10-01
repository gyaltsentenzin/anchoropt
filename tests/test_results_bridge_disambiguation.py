"""Two arms that share a signal and differ only in variant must not collapse to one measurement.

The read-side family emitted exactly this pair -- `retrieval_similarity_below_threshold` with
`reprompt:search_other_container` and with `reprompt:verify_before_answering`. They carry DIFFERENT
instructions (102 vs 146 chars), delivered through the executor's own eta_slot, so they are different
interventions with different trajectories.

The bridge keyed `signal -> arm_label`, so both tags resolved to whichever arm the manifest listed
first: one measurement reported for two arms, and the other silently omitted. That is the
byte-identical-arm defect from the reroute family arriving at the scoring boundary instead of the
construction boundary.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
BRIDGE = REPO / "scripts" / "pair_runs_to_results.py"

SIG = "retrieval_similarity_below_threshold"
L1 = f"post_execution/{SIG}/reprompt:search_other_container"
L2 = f"post_execution/{SIG}/reprompt:verify_before_answering"

MANIFEST = {"incumbent_id": "h0", "incumbent_token": "h0", "n_arms": 2, "note": "",
            "arms": [
                {"arm_label": L1, "boundary": "post_execution", "signal": SIG,
                 "action": "reprompt", "operator": "reprompt",
                 "variant": "search_other_container", "eta": {}},
                {"arm_label": L2, "boundary": "post_execution", "signal": SIG,
                 "action": "reprompt", "operator": "reprompt",
                 "variant": "verify_before_answering", "eta": {}},
            ]}


def _run(tmp_path, tagmap):
    """Two runs that differ, so a collapse shows up as a wrong or missing row."""
    def mk(name, correct_ids):
        d = tmp_path / name
        (d / "traj" / "query").mkdir(parents=True)
        rows = [{"id": f"c{i}", "valid": i in correct_ids, "is_prereq": False} for i in range(4)]
        (d / "eval_train_results.json").write_text(json.dumps({"results": rows}))
        return d

    ctl = mk("ctl", {0})
    a1 = mk("a1", {0, 1})          # net +1
    a2 = mk("a2", {0, 1, 2})       # net +2  -- distinguishable from a1
    man = tmp_path / "manifest.json"; man.write_text(json.dumps(MANIFEST))
    tm = tmp_path / "tags.json"; tm.write_text(json.dumps(tagmap))
    out = tmp_path / "results.json"
    r = subprocess.run(
        [sys.executable, str(BRIDGE), "--control", str(ctl),
         "--arm", f"vq1={a1}", "--arm", f"vq2={a2}",
         "--manifest", str(man), "--map", str(tm), "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO), timeout=120)
    assert r.returncode == 0, r.stderr[-1500:]
    return json.loads(out.read_text()), r.stdout


def test_full_arm_labels_resolve_to_DISTINCT_arms(tmp_path):
    payload, _ = _run(tmp_path, {L1: "vq1", L2: "vq2"})
    rows = payload["results"]
    assert len(rows) == 2, f"two arms collapsed to {len(rows)} measurement(s)"
    labels = {r["arm_label"] for r in rows}
    assert labels == {L1, L2}, labels
    nets = {r["arm_label"]: len(r["gains"]) - len(r["losses"]) for r in rows}
    assert nets[L1] != nets[L2], (
        f"both arms reported the same net -- the runs were not told apart: {nets}")


def test_signal_plus_variant_also_resolves(tmp_path):
    payload, _ = _run(tmp_path, {f"{SIG}:search_other_container": "vq1",
                                 f"{SIG}:verify_before_answering": "vq2"})
    assert {r["arm_label"] for r in payload["results"]} == {L1, L2}


def test_a_BARE_SHARED_signal_is_refused_not_guessed(tmp_path):
    """THE DEFECT. A signal carried by several arms must not silently pick the first."""
    payload, stdout = _run(tmp_path, {SIG: "vq1"})
    assert payload["results"] == [], (
        "an ambiguous bare signal produced a measurement instead of being omitted")
    assert "OMITTED" in stdout
    assert "several arms" in stdout, stdout[-400:]


def test_a_bare_signal_carried_by_ONE_arm_still_works(tmp_path):
    """The shortcut stays available where it is unambiguous -- this is not a tightening for its own sake."""
    man = dict(MANIFEST); man["arms"] = [MANIFEST["arms"][0]]; man["n_arms"] = 1
    m = tmp_path / "manifest.json"; m.write_text(json.dumps(man))
    (tmp_path / "ctl" / "traj").mkdir(parents=True)
    (tmp_path / "ctl" / "eval_train_results.json").write_text(
        json.dumps({"results": [{"id": "c0", "valid": False, "is_prereq": False}]}))
    (tmp_path / "a1" / "traj").mkdir(parents=True)
    (tmp_path / "a1" / "eval_train_results.json").write_text(
        json.dumps({"results": [{"id": "c0", "valid": True, "is_prereq": False}]}))
    tm = tmp_path / "tags.json"; tm.write_text(json.dumps({SIG: "vq1"}))
    out = tmp_path / "r.json"
    r = subprocess.run(
        [sys.executable, str(BRIDGE), "--control", str(tmp_path / "ctl"),
         "--arm", f"vq1={tmp_path / 'a1'}", "--manifest", str(m), "--map", str(tm),
         "--out", str(out)], capture_output=True, text=True, cwd=str(REPO), timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert json.loads(out.read_text())["results"], "an unambiguous bare signal stopped working"
