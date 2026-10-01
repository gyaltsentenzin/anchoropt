"""The four cost fields the audit found measured-then-discarded, pinned at their drop sites.

Each test fails if a field is dropped again, and each asserts the SAME rule: a value the
substrate did not report stays None, never 0 -- averaging a missing measurement as zero is how a
cost claim becomes wrong.

Source-level where the field's fate is decided by a literal (a whitelist tuple, a dict literal),
because that is where all four regressions lived. No GPU, no model, no network.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
EVALUATOR = REPO / "benchmarks" / "bfcl_v4" / "evaluator" / "memory_evaluator.py"
SIDECAR = REPO / "benchmarks" / "bfcl_v4" / "evaluator" / "traj_sidecar.py"
RUN_EVAL = REPO / "benchmarks" / "bfcl_v4" / "run_memory_eval.py"
DRIVER = REPO / "scripts" / "self_evolve_cycle2.py"


# ---------------------------------------------------------------- student tokens + latency

def test_step_record_carries_student_tokens_and_latency():
    """The handler already returns usage; the step record must keep it.

    Before: `model_response_data` was read for `model_responses` only, so `input_token` /
    `output_token` -- which every BFCL handler returns from the provider's own usage block --
    were discarded at the point of use.
    """
    src = EVALUATOR.read_text()
    m = re.search(r"step_record: Dict = \{(.*?)\n                \}", src, re.S)
    assert m, "step_record literal not found -- did it move?"
    body = m.group(1)
    for field in ("latency", "input_token", "output_token"):
        assert f'"{field}"' in body, f"step_record no longer records {field!r}"
    # Read from the handler's response, not invented.
    assert 'model_response_data.get("input_token")' in body
    assert 'model_response_data.get("output_token")' in body


def test_sidecar_whitelist_keeps_the_cost_fields():
    """`_STEP_FIELDS` is a whitelist: a field absent from it is silently dropped.

    This is exactly how `latency` -- computed at every model call -- never reached any artifact.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_traj_sidecar_t", SIDECAR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for field in ("latency", "input_token", "output_token"):
        assert field in mod._STEP_FIELDS, f"_STEP_FIELDS drops {field!r} again"


def test_sidecar_preserves_tokens_and_omits_unmeasured(tmp_path, monkeypatch):
    """End to end over the real writer: measured values survive, absent ones do not become 0."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_traj_sidecar_t2", SIDECAR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv("ANCHOROPT_TRAJ_DIR", str(tmp_path))
    mod._dir_cache.clear()

    traj = [
        {"turn": 0, "step": 0, "status": "executed", "decoded": ["f()"],
         "latency": 1.25, "input_token": 3100, "output_token": 42},
        # A step whose handler reported no usage: the keys are absent entirely.
        {"turn": 0, "step": 1, "status": "executed", "decoded": ["g()"]},
    ]
    out = mod.dump_episode({"id": "memory_vector_9-customer-9"}, traj, "query")
    assert out is not None
    steps = json.loads(out.read_text())["steps"]

    assert steps[0]["latency"] == 1.25
    assert steps[0]["input_token"] == 3100
    assert steps[0]["output_token"] == 42
    # NOT MEASURED must stay absent -- a 0 here would be averaged as a free call.
    for field in ("latency", "input_token", "output_token"):
        assert field not in steps[1], f"unmeasured {field!r} was materialized as a value"


# ---------------------------------------------------------------- teacher usage

def _driver_module():
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "integrations" / "claude_roles"))
    import self_evolve_cycle2 as mod
    return mod


class _Rec:
    def __init__(self, usage, error=""):
        self.role, self.model = "attributor", "claude-opus-5"
        self.meta_provider, self.meta_base_url = "claude", ""
        self.prompt_chars, self.usage, self.error = 1000, usage, error


def test_teacher_usage_is_written_and_summed(tmp_path):
    """Usage lived on CallRecord and reached no artifact, so a round could not state its cost."""
    mod = _driver_module()
    path = tmp_path / "teacher_usage.json"
    rec = mod.write_teacher_usage(
        path,
        [_Rec({"input_tokens": 10, "output_tokens": 2}),
         _Rec({"input_tokens": 5, "output_tokens": 1})],
        stage="attribution", elapsed_s=3.5)
    assert rec["calls"] == 2 and rec["errors"] == 0
    assert rec["input_tokens"] == 15 and rec["output_tokens"] == 3
    assert rec["model"] == "claude-opus-5" and rec["meta_provider"] == "claude"
    on_disk = json.loads(path.read_text())
    assert isinstance(on_disk, list) and len(on_disk) == 1
    assert len(on_disk[0]["calls_detail"]) == 2


def test_teacher_usage_absent_is_none_not_zero(tmp_path):
    """A provider reporting no usage must not be recorded as a free call."""
    mod = _driver_module()
    rec = mod.write_teacher_usage(tmp_path / "t.json", [_Rec({})],
                                 stage="attribution", elapsed_s=1.0)
    assert rec["input_tokens"] is None
    assert rec["output_tokens"] is None


def test_teacher_usage_records_a_failed_call(tmp_path):
    """A failed teacher call is part of discovery cost and must survive as an error, not vanish."""
    mod = _driver_module()
    rec = mod.write_teacher_usage(
        tmp_path / "t.json", [_Rec({}, error="APIError: 529")],
        stage="attribution", elapsed_s=1.0)
    assert rec["calls"] == 1 and rec["errors"] == 1
    assert rec["calls_detail"][0]["error"] == "APIError: 529"


def test_teacher_usage_appends_across_stages(tmp_path):
    """Several teacher stages in one round accumulate; the last must not overwrite the first."""
    mod = _driver_module()
    path = tmp_path / "t.json"
    mod.write_teacher_usage(path, [_Rec({"input_tokens": 1, "output_tokens": 1})],
                            stage="attribution", elapsed_s=1.0)
    mod.write_teacher_usage(path, [_Rec({"input_tokens": 2, "output_tokens": 2})],
                            stage="proposal", elapsed_s=2.0)
    rows = json.loads(path.read_text())
    assert [r["stage"] for r in rows] == ["attribution", "proposal"]


def test_live_attribution_persists_usage():
    """The live branch must call the writer -- capturing usage and not writing it was the defect."""
    src = DRIVER.read_text()
    live = src[src.index("def attribute("):]
    live = live[:live.index("\ndef ")]
    assert "write_teacher_usage" in live, "attribute() no longer persists teacher usage"


# ---------------------------------------------------------------- arm identity

def test_provenance_records_arm_identity():
    """Two runs installing DIFFERENT controllers had byte-identical provenance.

    `gate_config_fingerprint` covers the gate registry, not the templates artifact, so the
    results file could not name the arm that produced it.
    """
    src = RUN_EVAL.read_text()
    assert '"templates_sha256"' in src, "provenance no longer hashes the controller spec"
    assert '"git_sha"' in src, "provenance no longer records the repo commit"
