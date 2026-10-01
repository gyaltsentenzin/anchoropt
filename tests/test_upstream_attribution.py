"""Attribution for an UPSTREAM intervention follows the DEPENDENCY graph, not co-location.

"Did this scored case fire?" is the wrong question when the controller acts while state is being
BUILT: it never fires during the scored episode at all, by design. A naive on-fired count then reports
gains as unattributable when they are fully attributable.

Measured instance: an upstream storage controller scored 7/9 gains on fired cases, which read as "2
gains have another cause". By dependency it is 9/9 -- every gain depends on a setup episode the
controller fired in. The 2 were an artefact of the question.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _summary():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "result_summary", REPO / "scripts" / "result_summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_run(root: Path, *, setup_fires, query_ids, shard):
    """A run directory with a setup phase that fires and a query phase that does not."""
    (root / "traj" / "prereq").mkdir(parents=True, exist_ok=True)
    (root / "traj" / "query").mkdir(parents=True, exist_ok=True)
    for cid, fires in setup_fires.items():
        step = {"decoded": ["w()"], "step": 0}
        if fires:
            step["upstream_fired_gate"] = True
        json.dump({"case_id": cid, "steps": [step]},
                  open(root / "traj" / "prereq" / f"{cid}.json", "w"))
    for cid in query_ids:
        json.dump({"case_id": cid, "steps": [{"decoded": ["r()"], "step": 0}]},
                  open(root / "traj" / "query" / f"{cid}.json", "w"))
    json.dump([{"id": i, "valid": True} for i in query_ids],
              open(root / "eval_train_results.json", "w"))
    json.dump(shard, open(root / "cases.json", "w"))


def test_attribution_uses_setup_dependencies_when_a_setup_phase_fired(tmp_path):
    mod = _summary()
    shard = [{"id": "q1", "depends_on": ["p1"]}, {"id": "q2", "depends_on": ["p2"]},
             {"id": "p1", "depends_on": []}, {"id": "p2", "depends_on": []}]
    run = tmp_path / "arm"
    _mk_run(run, setup_fires={"p1": True, "p2": False}, query_ids=["q1", "q2"], shard=shard)
    got, basis = mod.exposed_cases(run, {"q1", "q2"})
    assert basis == "setup-phase dependency"
    assert got == {"q1"}, "a query depending on a fired setup episode must be attributable"


def test_a_query_that_never_fires_is_STILL_attributable_via_its_setup(tmp_path):
    """The whole point: query_fired=0 is expected for an upstream intervention."""
    mod = _summary()
    shard = [{"id": "q1", "depends_on": ["p1"]}, {"id": "p1", "depends_on": []}]
    run = tmp_path / "arm"
    _mk_run(run, setup_fires={"p1": True}, query_ids=["q1"], shard=shard)
    # the query episode carries NO firing key at all
    q = json.load(open(run / "traj" / "query" / "q1.json"))
    assert not any(k.endswith("_gate") for s in q["steps"] for k in s)
    got, _ = mod.exposed_cases(run, {"q1"})
    assert got == {"q1"}


def test_transitive_dependencies_are_followed(tmp_path):
    mod = _summary()
    shard = [{"id": "q1", "depends_on": ["p2"]}, {"id": "p2", "depends_on": ["p1"]},
             {"id": "p1", "depends_on": []}]
    run = tmp_path / "arm"
    _mk_run(run, setup_fires={"p1": True, "p2": False}, query_ids=["q1"], shard=shard)
    got, _ = mod.exposed_cases(run, {"q1"})
    assert got == {"q1"}, "attribution must follow the chain, not only direct parents"


def test_no_setup_phase_falls_back_to_same_episode_firing(tmp_path):
    """A query-time intervention must still be attributed the ordinary way."""
    mod = _summary()
    run = tmp_path / "arm"
    (run / "traj" / "query").mkdir(parents=True)
    json.dump({"case_id": "q1", "steps": [{"fired_gate": True}]},
              open(run / "traj" / "query" / "q1.json", "w"))
    got, basis = mod.exposed_cases(run, {"q1"})
    assert got == set() and "same-episode" in basis


def test_the_basis_is_always_reported(tmp_path):
    """The two bases answer different questions; a reader must know which they were given."""
    mod = _summary()
    run = tmp_path / "arm"
    (run / "traj" / "prereq").mkdir(parents=True)
    json.dump({"case_id": "p1", "steps": [{"upstream_fired_gate": True}]},
              open(run / "traj" / "prereq" / "p1.json", "w"))
    _, basis = mod.exposed_cases(run, {"q1"})
    assert basis, "attribution basis must never be empty"


# ================================================================================================
# THE ADAPTER'S DECLARATIONS vs THE REAL HOOK -- the audit that would have caught the ghost
# ================================================================================================
#
# Core cannot read the host's source without coupling itself to one benchmark, so the binding claim is
# checked HERE, in the adapter's own test file, against the captured patch. The patch is the cluster
# hook verbatim, so this is a check against the code that actually ran.

def _hook_source() -> str:
    return (REPO / "patches" / "upstream_preexec_hook_and_phase_scope.diff").read_text()


def _adapter():
    import importlib
    import sys as _sys
    bd = REPO / "benchmarks" / "bfcl_v4"
    if str(bd) not in _sys.path:
        _sys.path.insert(0, str(bd))
    return importlib.import_module("bfcl_runtime")


def test_no_declared_capability_names_a_flag_the_hook_never_reads():
    """THE GHOST TEST. The defect verbatim: the commitment-gate suppress cell declared
    `enable_redundant_write_suppress`, and that flag appears ZERO times in the hook that withholds
    the call. The hook gates on installed controllers and `fires_on`; the flag was never consulted.

    A capability's `binding` must name something findable in the hook that runs.
    """
    src = _hook_source()
    R = _adapter()
    cap = R.executor_capability("post_generation_pre_exec", "suppress")
    assert cap is not None and cap.is_bound

    assert "enable_redundant_write_suppress" not in cap.binding, \
        "the binding must not name the flag the hook never reads"
    # The binding names the real gating mechanism, and each part is present in the hook.
    for token in ("runtime_hook", "_phase_eligible", "fires_on", "_mg_exec_calls"):
        assert token in cap.binding, f"binding should name {token}"
        assert token in src, f"{token} must exist in the hook the binding claims"


def test_the_upstream_suppress_capability_declares_ONLY_eta_the_hook_reads():
    """INERT ETA, pinned against the source. The hook reads exactly `retry_budget` and `instruction`
    from a controller's eta. It never reads `suppressed_operation` or `preservation`, so the suppress
    capability may not claim them.
    """
    src = _hook_source()
    R = _adapter()
    cap = R.executor_capability("post_generation_pre_exec", "suppress")

    assert "retry_budget" in cap.consumes
    assert 'get(\n' in src or '"retry_budget"' in src, "the hook must actually read retry_budget"
    for never_read in ("suppressed_operation", "preservation"):
        assert never_read not in cap.consumes, \
            f"the hook does not read {never_read}; declaring it would re-open the inert-eta class"
        assert never_read not in src, f"{never_read} must be absent from the hook"


def test_the_upstream_reprompt_capability_is_DISABLED_with_a_reason():
    """D3, pinned per EXECUTOR rather than per cell.

    The cell holds TWO executors and only one is inert. Pinning the CELL as disabled is what made a
    live injector unreachable, so this asserts the inert executor specifically: it is still declared,
    still refused, and still names its reason. Re-enable it only when two instruction variants
    demonstrably produce DIFFERENT trajectories through it.
    """
    src = _hook_source()
    R = _adapter()
    caps = {c.cid: c for c in R.executor_capabilities("post_generation_pre_exec", "reprompt")}
    cap = caps.get("write_instruction_to_step_record")
    assert cap is not None, f"the inert executor must stay DECLARED, not deleted: {sorted(caps)}"
    assert not cap.is_enabled, "an action whose parameter never reaches the agent must not be measured"
    assert "never injected" in cap.disabled_reason
    assert cap.consumes == (), "it consumes nothing -- it only records"

    # The only thing the hook does with the instruction is write it to the step record.
    assert 'step_record["upstream_instruction"]' in src
    for injection in ("messages.append", "inference_data", "_add_message", "regenerate"):
        assert injection not in src, \
            f"{injection} in the hook would mean the instruction DOES reach the model -- re-enable it"


def test_the_capability_audit_reports_no_ghosts():
    """The one-call audit a porter runs."""
    R = _adapter()
    audit = R.capability_audit()
    assert audit["unbound"] == [], f"ghost capabilities: {audit['unbound']}"
    assert any("post_generation_pre_exec/reprompt" in d for d in audit["disabled"])
    assert "post_generation_pre_exec/suppress" in audit["bound_and_enabled"]
