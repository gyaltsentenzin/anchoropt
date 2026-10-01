"""The WHOLE-CORPUS recovery table must refuse to manufacture a total.

Pins the failure this script exists to prevent: "104/303 = 34.32%" was quoted as a recovery figure when
the composed stack had never been run. It was three separately measured per-cell deltas, added. There
must be no code path that produces a total from anything other than three cells of one executed stack.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("rt", ROOT / "scripts" / "corpus_recovery_table.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


def mkrun(d: pathlib.Path, correct: int, n: int, *, effects=None, prereq=False):
    (d / "run" / "traj" / "prereq").mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"c{i}", "valid": i < correct, "is_prereq": False} for i in range(n)]
    if prereq:
        rows += [{"id": "p0", "valid": True, "is_prereq": True}]
    (d / "run" / "eval_train_results.json").write_text(json.dumps({"results": rows}))
    if effects:
        (d / "run" / "traj" / "prereq" / "e.json").write_text(
            json.dumps({"case_id": "e", "steps": [dict.fromkeys(effects, True)]}))
    return d


# ---------------------------------------------------------------------------------------------------
# THE CANONICAL FACTS
# ---------------------------------------------------------------------------------------------------

def test_the_canonical_split_and_H0_match_the_measured_record():
    """rounds/H0_SESSION/FINDINGS.md: kv 18/105, vector 19/89, rec_sum 54/109 = 91/303 = 30.03%."""
    assert rt.CANONICAL == {"kv": 105, "vector": 89, "rec_sum": 109}
    assert rt.CORPUS_N == 303
    assert rt.H0 == {"kv": 18, "vector": 19, "rec_sum": 54}
    assert rt.H0_TOTAL == 91
    assert abs(100 * 91 / 303 - 30.03) < 0.01


def test_the_historical_target_and_gain_are_recorded_with_the_table():
    """A recovery percentage must always carry its denominator."""
    assert rt.HIST_TARGET == 52.48
    assert abs(rt.HIST_GAIN - 22.45) < 0.01


def test_effect_keys_exclude_hook_bookkeeping():
    """`relocate_requested_gate` in vector means the hook was consulted, not that a relocation happened.
    Counting it reports cross-cell activity that did not occur."""
    assert "relocate_gate" in rt.EFFECT_KEYS
    assert "relocate_requested_gate" not in rt.EFFECT_KEYS
    assert not any(k.endswith("_requested_gate") for k in rt.EFFECT_KEYS)


# ---------------------------------------------------------------------------------------------------
# THE REFUSALS
# ---------------------------------------------------------------------------------------------------

def test_a_two_cell_total_is_REFUSED(tmp_path, monkeypatch):
    mkrun(tmp_path / "a_kv_train", 22, 105, effects=["relocate_gate"])
    mkrun(tmp_path / "a_vector_train", 23, 89, effects=["upstream_zero_call_gate"])
    monkeypatch.setattr(sys, "argv", ["rt", "--results", str(tmp_path),
                                      "--arm-tags", "kv=a_kv,vector=a_vector",
                                      "--json", str(tmp_path / "o.json")])
    rc = rt.main()
    out = json.loads((tmp_path / "o.json").read_text())
    assert rc == 2, "an incomplete table must exit non-zero"
    assert out["complete"] is False
    assert "arm_total" not in out, "no total may be reported from two cells"


def test_a_WRONG_DENOMINATOR_blocks_the_total(tmp_path, monkeypatch):
    """A different n is a different population, not a rounding issue."""
    mkrun(tmp_path / "a_kv_train", 22, 105, effects=["relocate_gate"])
    mkrun(tmp_path / "a_vector_train", 23, 80, effects=["upstream_zero_call_gate"])   # 80 != 89
    mkrun(tmp_path / "a_rec_sum_train", 59, 109, effects=["capacity_repair_gate"])
    monkeypatch.setattr(sys, "argv", ["rt", "--results", str(tmp_path),
                                      "--arm-tags", "kv=a_kv,vector=a_vector,rec_sum=a_rec_sum",
                                      "--json", str(tmp_path / "o.json")])
    rt.main()
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["complete"] is False
    assert any("DIFFERENT POPULATION" in p for p in out["problems"]), out["problems"]
    assert "arm_total" not in out


def test_a_cell_with_NO_FIRINGS_blocks_the_total_when_required(tmp_path, monkeypatch):
    """A cell where nothing fired contributes an unattributed delta."""
    mkrun(tmp_path / "a_kv_train", 22, 105, effects=["relocate_gate"])
    mkrun(tmp_path / "a_vector_train", 23, 89)                      # no effects
    mkrun(tmp_path / "a_rec_sum_train", 59, 109, effects=["capacity_repair_gate"])
    monkeypatch.setattr(sys, "argv", ["rt", "--results", str(tmp_path), "--require-firings",
                                      "--arm-tags", "kv=a_kv,vector=a_vector,rec_sum=a_rec_sum",
                                      "--json", str(tmp_path / "o.json")])
    rt.main()
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["complete"] is False
    assert any("unattributed" in p for p in out["problems"]), out["problems"]


def test_an_UNPAIRED_control_blocks_the_total(tmp_path, monkeypatch):
    mkrun(tmp_path / "a_kv_train", 22, 105, effects=["relocate_gate"])
    mkrun(tmp_path / "a_vector_train", 23, 89, effects=["upstream_zero_call_gate"])
    mkrun(tmp_path / "a_rec_sum_train", 59, 109, effects=["capacity_repair_gate"])
    mkrun(tmp_path / "c_kv_train", 18, 100)                          # control n != arm n
    monkeypatch.setattr(sys, "argv", ["rt", "--results", str(tmp_path),
                                      "--arm-tags", "kv=a_kv,vector=a_vector,rec_sum=a_rec_sum",
                                      "--control-tags", "kv=c_kv",
                                      "--json", str(tmp_path / "o.json")])
    rt.main()
    out = json.loads((tmp_path / "o.json").read_text())
    assert any("not a paired comparison" in p for p in out["problems"]), out["problems"]
    assert out["complete"] is False


# ---------------------------------------------------------------------------------------------------
# THE ONE CASE THAT IS ALLOWED
# ---------------------------------------------------------------------------------------------------

def test_three_complete_cells_of_ONE_executed_stack_DO_produce_a_total(tmp_path, monkeypatch):
    """The projection's own numbers: 22+23+59 = 104/303 = 34.32%, +4.29pp = 19.1% of +22.45.
    Allowed ONLY because all three come from run directories of the same stack."""
    mkrun(tmp_path / "a_kv_train", 22, 105, effects=["relocate_gate"])
    mkrun(tmp_path / "a_vector_train", 23, 89, effects=["upstream_zero_call_gate"])
    mkrun(tmp_path / "a_rec_sum_train", 59, 109, effects=["capacity_repair_gate"])
    monkeypatch.setattr(sys, "argv", ["rt", "--results", str(tmp_path), "--require-firings",
                                      "--arm-tags", "kv=a_kv,vector=a_vector,rec_sum=a_rec_sum",
                                      "--json", str(tmp_path / "o.json")])
    rc = rt.main()
    out = json.loads((tmp_path / "o.json").read_text())
    assert rc == 0 and out["complete"] is True
    assert out["arm_total"] == 104
    assert out["control_total"] == 91, "defaults to the MEASURED H0 when no control tag is given"
    assert abs(out["arm_pct"] - 34.32) < 0.01
    assert abs(out["gain_pp"] - 4.29) < 0.01
    assert abs(out["pct_of_historical_gain"] - 19.1) < 0.2


def test_prereq_rows_are_excluded_from_the_scored_denominator(tmp_path, monkeypatch):
    """Only SCORED query cases count; a prereq row must not inflate n."""
    d = mkrun(tmp_path / "a_kv_train", 22, 105, effects=["relocate_gate"], prereq=True)
    got = rt.load_cell(d)
    assert got == (22, 105)
