"""The installer must evaluate every operator core emits.

Core carries two predicate vocabularies -- `signal_lang.py` spells equality `eq`, and
`signal_grammar.py` (the structured-search path that `optimize_residual` uses) spells it `equals`.
The installer knew only `eq`, so an arm carrying `equals` raised `unknown op` at install time.

Found by a real round: 6 of the 219 arms in the first live Claude->Granite propose phase used
`equals`, and one of them was in the selected candidate set. It fails loudly rather than
mis-firing, so nothing silently ran the wrong arm -- but the arm could not be evaluated at all.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
GRAMMAR = REPO / "anchoropt" / "learning" / "signal_grammar.py"


def _installer(tmp_path, spec):
    """Import the installer against `spec`. It reads its spec at import time."""
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    import os
    os.environ["ANCHOROPT_CONTROLLER_SPEC"] = str(p)
    sys.path.insert(0, str(REPO / "scripts"))
    for mod in ("install_controller",):
        sys.modules.pop(mod, None)
    import install_controller
    return install_controller


_SPEC = {
    "name": "container_is_core__and__proposed_payload_chars_gt_57p0",
    "locus": "post_generation_pre_exec",
    "action": "suppress",
    "operator": "suppress",
    "eta": {"suppressed_operation": "the proposed operation this signal implicates"},
    "predicate": {"all": [{"field": "container", "op": "equals", "value": "core"},
                          {"field": "proposed_payload_chars", "op": "gt", "value": 57.0}]},
}


def test_equals_is_a_known_operator(tmp_path):
    """`equals` must evaluate, not raise -- it is core's own spelling on this path."""
    ic = _installer(tmp_path, _SPEC)
    assert "equals" in ic._OPS, "the installer cannot evaluate an `equals` predicate core emits"


def test_equals_discriminates_like_eq(tmp_path):
    ic = _installer(tmp_path, _SPEC)
    pred = ic.SpecPredicate(_SPEC)
    assert pred.fires_on({"container": "core", "proposed_payload_chars": 400}) is True
    assert pred.fires_on({"container": "core", "proposed_payload_chars": 40}) is False
    assert pred.fires_on({"container": "archival", "proposed_payload_chars": 400}) is False


def test_a_missing_field_is_not_a_firing(tmp_path):
    """The rule the project has paid for before: absent != condition satisfied."""
    ic = _installer(tmp_path, _SPEC)
    pred = ic.SpecPredicate(_SPEC)
    assert pred.fires_on({"proposed_payload_chars": 400}) is False
    assert pred.fires_on({"container": "core"}) is False


def test_grammar_still_emits_equals():
    """If core ever renames this, the alias is dead weight and this test says so."""
    assert '"equals"' in GRAMMAR.read_text(), (
        "signal_grammar no longer emits `equals` -- re-check whether the installer alias is needed")


def test_an_unknown_operator_still_fails_loudly(tmp_path):
    """The alias must not become a general fallback: an unknown op is still refused."""
    bad = dict(_SPEC, predicate={"field": "container", "op": "approximately", "value": "core"})
    ic = _installer(tmp_path, _SPEC)
    with pytest.raises(ValueError, match="unknown op"):
        ic.SpecPredicate(bad).fires_on({"container": "core"})
