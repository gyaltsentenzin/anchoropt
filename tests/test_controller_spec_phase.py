"""An emitted controller must declare the PHASE its residual was mined from.

`_phase_eligible` in the BFCL evaluator confines a controller to one phase, and an UNDECLARED
controller defaults to "query". A round mined with `--phase prereq` therefore emitted controllers
that were installed, reached, and then skipped in the only phase their residual lives in.

Measured on the first live Claude->Granite round, before the fix:
  * prereq: `upstream_skipped_prereq_gate` 408, controller firings 0
  * query:  hook reached 85-108 times, fired 1-2 times -- query episodes are ~99% reads, so a
    write-side predicate has almost nothing to act on there

The historical WRITE2 arm carried `"phase": "prereq"` hand-added to its spec. That is the
human-in-the-loop step that emitting data as core's own output is meant to remove, so the driver --
which already knows which phase it mined -- says so itself.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "integrations" / "claude_roles"))
DRIVER_SRC = (REPO / "scripts" / "self_evolve_cycle2.py").read_text()


def _driver():
    import self_evolve_cycle2
    return self_evolve_cycle2


class _Atom:
    def __init__(self, field, op, value=None):
        self.field, self.op, self.value = field, op, value


def _arm():
    return types.SimpleNamespace(
        signal="proposed_payload_chars_gt_272p0",
        boundary=types.SimpleNamespace(value="post_generation_pre_exec"),
        action=types.SimpleNamespace(value="suppress"),
        instantiated=types.SimpleNamespace(operator=types.SimpleNamespace(value="suppress")),
        eta={"suppressed_operation": "the proposed operation this signal implicates"})


def test_spec_carries_the_mined_phase():
    spec = _driver().controller_spec(_arm(), _Atom("proposed_payload_chars", "gt", 272.0),
                                    phase="prereq")
    assert spec["phase"] == "prereq", "a prereq-mined controller must act in prereq"
    assert "phase=prereq" in spec["provenance"]


def test_spec_defaults_to_query_when_unspecified():
    """The evaluator's own default, stated explicitly rather than left to be inferred."""
    spec = _driver().controller_spec(_arm(), _Atom("proposed_payload_chars", "gt", 272.0))
    assert spec["phase"] == "query"


def test_the_driver_passes_its_own_phase_argument():
    """A spec emitted with the wrong phase measures a mechanism in a phase it cannot act in.

    Both exits must build specs through `_spec_for` AND hand it the phase the round mined. Checked as
    the two call shapes plus the builder's own pass-through, so a refactor that drops the argument
    fails here even if the call text moves.
    """
    assert "controller_spec(arm, pred_obj, phase=phase)" in DRIVER_SRC, (
        "_spec_for no longer passes the phase through for a synthesized signal")
    for call in ("_spec_for(arm, _SYNTH_PREDICATES, a.phase, runtime) for arm in outcome.candidates",
                 "_spec_for(arm, _SYNTH_PREDICATES, a.phase, runtime) for _n, arm in scored"):
        assert call in DRIVER_SRC, f"an exit stopped passing its mined phase: {call!r}"
    assert DRIVER_SRC.count("_spec_for(arm, _SYNTH_PREDICATES, a.phase, runtime)") >= 2, (
        "not every exit builds its specs through the shared builder")


def test_the_installer_reads_phase_from_the_spec(tmp_path, monkeypatch):
    """The two halves must agree: core writes `phase`, the installer must honour it."""
    import json
    spec = {"name": "n", "locus": "post_generation_pre_exec", "action": "suppress",
            "operator": "suppress", "eta": {"suppressed_operation": "x"},
            "phase": "prereq",
            "predicate": {"field": "proposed_payload_chars", "op": "gt", "value": 272.0}}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    monkeypatch.setenv("ANCHOROPT_CONTROLLER_SPEC", str(p))
    sys.modules.pop("install_controller", None)
    import install_controller as ic
    assert ic.SpecPredicate(spec).phase == "prereq"
