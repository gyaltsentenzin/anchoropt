"""FULL core-to-spec replay: real KV residual inputs -> real grounding -> real arm construction.

WHY THIS FILE IS DIFFERENT FROM THE OTHER IDENTITY TESTS. Those re-resolve a recorded manifest, which
checks `resolve()` but takes the manifest's own arms as given. This one runs the actual pipeline stage --
`optimize_residual` over states projected from real KV trajectories -- regenerates the arms and their
specs, and then checks that EVERY emitted controller resolves to the executor its (action, operator, eta)
actually require. No expected identity is written down here; each one is obtained by resolving the arm's
own fields against the adapter's own capability table.

It also confirms the previously ACCEPTED relocation still appears, still carries the right identity, and
still installs and fires through the real installer and real predicate.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "benchmarks" / "bfcl_v4", REPO / "scripts"):
    sys.path.insert(0, str(p))

import bfcl_runtime as R                                                      # noqa: E402
from anchoropt.anchor import Action, IncisionPoint                            # noqa: E402
from anchoropt.learning import executor_capability as EC                      # noqa: E402
from anchoropt.learning.action_contract import CONTRACTS, Operator            # noqa: E402

H0 = REPO / "results" / "h0_native" / "nativectl_kv_train" / "run"
ACCEPTED_CID = "relocate_entry_preserving_information_then_retry"

pytestmark = pytest.mark.skipif(not H0.exists(), reason="raw H0 kv artifacts absent")


@pytest.fixture(scope="module")
def kv_events():
    """The REAL residual inputs: prereq trajectory steps from the frozen raw-H0 kv run, keyed by case.

    Kept as {case_id: steps} because the DRIVER's observation supply is episode-scoped -- `error_kind`,
    `container_full` and the rest are derived per episode, and flattening the steps first is exactly the
    mistake that made an earlier version of this test see error_kind=None on all 600 states.
    """
    by_case = {}
    for fp in sorted(glob.glob(str(H0 / "traj" / "prereq" / "*.json"))):
        try:
            ep = json.load(open(fp))
        except Exception:
            continue
        steps = [s for s in (ep.get("steps") or []) if isinstance(s, dict)]
        if steps:
            by_case[str(ep.get("case_id"))] = steps
    assert by_case, "no real kv episodes loaded"
    return by_case


@pytest.fixture(scope="module")
def grounded(kv_events):
    """Run real grounding + arm construction over those events."""
    os.environ["ANCHOROPT_CELL"] = "kv"
    # THE DRIVER'S OWN SUPPLY FUNCTION, not a reimplementation. `states_at` projects to a boundary but
    # does not derive error_kind -- that is the adapter-side observation supply the driver owns, and
    # skipping it is why this test first reported 0 firings against a controller that fired 61 times on
    # GPU. Importing it here is what makes this a CORE-TO-SPEC replay rather than a partial one.
    import self_evolve_cycle2 as driver
    states = driver.observable_states(kv_events)
    arms = []
    for signal in ("container_at_capacity", "append_would_exceed_cap"):
        for operator in ("transform", "substitute"):
            fn = (R.ground_transforms if operator == "transform"
                  else R.ground_substitute_destinations)
            for g in (fn(signal, IncisionPoint.POST_EXECUTION) or ()):
                if g.get("infeasible"):
                    continue
                arms.append({"signal": signal, "operator": operator,
                             "variant": g.get("variant"), "eta": dict(g.get("eta") or {}),
                             "grounding": dict(g.get("grounding") or {})})
    return {"states": states, "arms": arms}


def test_real_states_project_from_real_trajectories(grounded):
    assert len(grounded["states"]) > 100, len(grounded["states"])


def test_grounding_over_real_inputs_produces_arms(grounded):
    assert grounded["arms"], "real grounding produced nothing"


def test_EVERY_emitted_arm_resolves_to_the_executor_ITS_OWN_FIELDS_REQUIRE(grounded):
    """The core assertion. No expected identity is hardcoded: each is resolved from the arm itself."""
    caps = R.executor_capabilities("post_execution", Action.REROUTE)
    checked = 0
    for a in grounded["arms"]:
        op = Operator(a["operator"])
        required = tuple(CONTRACTS[op].required)
        cap, why = EC.resolve(caps, boundary=IncisionPoint.POST_EXECUTION, action=Action.REROUTE,
                              signal=a["signal"], eta=a["eta"], required_eta=required,
                              operator=a["operator"])
        if cap is None:
            continue                                  # a refused arm is honest; a WRONG id is not
        checked += 1
        declared = tuple(cap.operators or ())
        assert (not declared) or a["operator"] in declared, (
            f"{a['variant']} ({a['operator']}) resolved to {cap.capability_id} "
            f"which implements {declared}: {why}")
        # and the grounding's own claimed capability, where it makes one, must agree
        claimed = a["grounding"].get("capability_id")
        if claimed:
            assert claimed == cap.capability_id, (
                f"{a['variant']} claims {claimed} but resolution gives {cap.capability_id}")
    assert checked, "no arm resolved at all -- the replay proved nothing"


def test_the_accepted_relocation_is_STILL_among_the_regenerated_arms(grounded):
    rel = [a for a in grounded["arms"] if "relocate" in str(a["variant"])]
    assert rel, [a["variant"] for a in grounded["arms"]]
    assert all(a["grounding"].get("capability_id") == ACCEPTED_CID for a in rel)


def test_a_substitute_arm_never_regenerates_with_the_relocation_identity(grounded):
    """The pilot's defect, checked against freshly generated arms rather than the old manifest."""
    caps = R.executor_capabilities("post_execution", Action.REROUTE)
    for a in grounded["arms"]:
        if a["operator"] != "substitute":
            continue
        cap, _ = EC.resolve(caps, boundary=IncisionPoint.POST_EXECUTION, action=Action.REROUTE,
                            signal=a["signal"], eta=a["eta"],
                            required_eta=tuple(CONTRACTS[Operator.SUBSTITUTE].required),
                            operator="substitute")
        assert cap is None or cap.capability_id != ACCEPTED_CID


def test_the_accepted_relocation_still_INSTALLS_and_FIRES_through_the_real_path(grounded):
    """Regenerated spec -> real installer -> real predicate -> real dispatch guard."""
    from bfcl_regression_host import _ensure_declared_signal_seam, _load_installer
    rel = [a for a in grounded["arms"] if "relocate" in str(a["variant"])][0]
    spec = {"name": rel["signal"], "locus": "post_execution", "action": "reroute",
            "operator": "transform", "variant": rel["variant"], "eta": rel["eta"],
            "capability_id": rel["grounding"]["capability_id"], "phase": "prereq",
            "predicate": {"declared_signal": rel["signal"], "params": {}}}
    _ensure_declared_signal_seam()
    ctl = _load_installer().SpecPredicate(spec)
    assert ctl.phase == "prereq"
    # fires on the real slot refusal, silent on the character cap it cannot relieve
    assert ctl.fires_on({"error_kind": "no_capacity", "proposes_write": True}) is True
    assert ctl.fires_on({"error_kind": "blob_would_overflow", "proposes_write": True}) is False


def test_it_fires_on_REAL_projected_states_not_only_synthetic_ones(grounded):
    """A predicate that fires only on hand-made states is not evidence it fires in the run."""
    from bfcl_regression_host import _ensure_declared_signal_seam, _load_installer
    rel = [a for a in grounded["arms"] if "relocate" in str(a["variant"])][0]
    _ensure_declared_signal_seam()
    ctl = _load_installer().SpecPredicate(
        {"name": rel["signal"], "locus": "post_execution", "action": "reroute",
         "operator": "transform", "variant": rel["variant"], "eta": rel["eta"],
         "capability_id": rel["grounding"]["capability_id"], "phase": "prereq",
         "predicate": {"declared_signal": rel["signal"], "params": {}}})
    fired = sum(1 for st in grounded["states"] if ctl.fires_on(st))
    assert fired > 0, "the accepted controller fires on NO real projected state"


def test_the_offline_projection_AGREES_with_the_live_run_on_the_request_count(grounded):
    """The strongest available offline check that this replay is not vacuous.

    The measured GPU round for this controller recorded 84 predicate REQUESTS (of which 61 became
    completed repairs, the rest bounded or declined by the host). Replaying the same frozen trajectories
    through the driver's own observation supply and the real installed predicate must reproduce that 84
    -- if it does not, either the supply path or the predicate has drifted from what ran.
    """
    from bfcl_regression_host import _ensure_declared_signal_seam, _load_installer
    rel = [a for a in grounded["arms"] if "relocate" in str(a["variant"])][0]
    _ensure_declared_signal_seam()
    ctl = _load_installer().SpecPredicate(
        {"name": rel["signal"], "locus": "post_execution", "action": "reroute",
         "operator": "transform", "variant": rel["variant"], "eta": rel["eta"],
         "capability_id": rel["grounding"]["capability_id"], "phase": "prereq",
         "predicate": {"declared_signal": rel["signal"], "params": {}}})
    fired = sum(1 for st in grounded["states"] if ctl.fires_on(st))
    assert fired == 84, (
        f"offline projection fires {fired} times; the live round recorded 84 requests -- "
        f"the supply path or the predicate has drifted")
