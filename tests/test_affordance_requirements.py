"""An unrealizable repair must survive the round that produced it, as a CHECKABLE requirement.

The defect this closes: a repair the proposer could not express was recorded as prose and died with
the round, so the next round re-derived the same sentence and nothing accumulated. Measured across one
experiment: 12 distinct unrealizable repairs, 11 blocked by "no declared signal observes this
condition" -- while the search kept reducing the capacity family to a weaker expressible neighbour.
"""

from __future__ import annotations

from anchoropt.learning.affordance_requirements import (
    NEEDS_BOTH, NEEDS_OBSERVATION, NEEDS_OPERATION, SATISFIED,
    AffordanceLedger, AffordanceRequirement,
)


def _req(**kw):
    base = dict(repair="r", boundary="gate", observation_fields=("f",), operation="suppress")
    base.update(kw)
    return AffordanceRequirement(**base)


def test_status_separates_a_missing_OBSERVATION_from_a_missing_ACTION():
    """The two lead to different next steps, which is the whole reason to keep them apart."""
    r = _req()
    assert r.status(supplied_fields=[], available_operations=["suppress"]) == NEEDS_OBSERVATION
    assert r.status(supplied_fields=["f"], available_operations=[]) == NEEDS_OPERATION
    assert r.status(supplied_fields=[], available_operations=[]) == NEEDS_BOTH
    assert r.status(supplied_fields=["f"], available_operations=["suppress"]) == SATISFIED


def test_a_repair_needing_no_operation_is_satisfied_by_the_observation_alone():
    r = _req(operation="")
    assert r.status(supplied_fields=["f"], available_operations=[]) == SATISFIED


def test_missing_observations_names_the_FIELDS_not_a_count():
    """The field list is the work item: it is what an adapter can actually answer."""
    r = _req(observation_fields=("a", "b"))
    assert r.missing_observations(["a"]) == ("b",)
    assert r.missing_observations([]) == ("a", "b")


def test_requirements_MERGE_on_identity_not_on_wording():
    """Two attributors wording one repair differently must not create two requirements."""
    led = AffordanceLedger()
    led.add(_req(repair="stop re-issuing a refused write", evidence_case_ids=("c1",)))
    led.add(_req(repair="never re-emit a write that already failed", evidence_case_ids=("c2",)))
    assert len(led.requirements) == 1, "the same (boundary, fields, operation) is ONE requirement"
    assert set(led.requirements[0].evidence_case_ids) == {"c1", "c2"}, \
        "evidence must accumulate -- that is the quantity a porter needs"


def test_a_DIFFERENT_boundary_is_a_different_requirement():
    led = AffordanceLedger()
    led.add(_req(boundary="gate"))
    led.add(_req(boundary="post_exec"))
    assert len(led.requirements) == 2


def test_ranking_is_by_EVIDENCE_and_is_cost_control_only():
    """It decides what a porter looks at first, never what is adopted."""
    led = AffordanceLedger()
    led.add(_req(observation_fields=("rare",), evidence_case_ids=("c1",)))
    led.add(_req(observation_fields=("common",), evidence_case_ids=("c1", "c2", "c3")))
    rows = led.ranked_by_evidence(supplied_fields=[], available_operations=["suppress"])
    assert [r.observation_fields[0] for r, _s, _n in rows] == ["common", "rare"]
    assert [n for _r, _s, n in rows] == [3, 1]


def test_a_SATISFIED_requirement_leaves_the_unmet_list():
    """Once the host supplies the field, the requirement stops being a work item."""
    led = AffordanceLedger()
    led.add(_req(observation_fields=("f",)))
    assert led.unmet(supplied_fields=[], available_operations=["suppress"])
    assert not led.unmet(supplied_fields=["f"], available_operations=["suppress"])


def test_round_trips_through_json():
    """It has to outlive the process, or it is not a ledger."""
    led = AffordanceLedger()
    led.add(_req(safety_contract=("a first attempt must pass",), evidence_case_ids=("c1",)))
    back = AffordanceLedger.from_dict(led.as_dict())
    assert len(back.requirements) == 1
    r = back.requirements[0]
    assert r.observation_fields == ("f",) and r.operation == "suppress"
    assert r.safety_contract == ("a first attempt must pass",)


def test_core_module_names_no_benchmark_vocabulary():
    """Generic reasoning stays in core: no signal names, no anchor identity, no thresholds."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "anchoropt/learning/affordance_requirements.py").read_text()
    for tok in ("core_memory", "archival", "rec_sum", "bfcl", "A1", "A3", "A4", "A9"):
        assert tok not in src, f"core module names {tok!r}"
