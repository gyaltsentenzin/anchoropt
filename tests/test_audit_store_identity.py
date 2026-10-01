"""The store-identity audit must be able to say "I could not tell".

Pins the defect this script shipped with for ten minutes: over three arms whose provenance it had
failed to locate and whose controllers it read as "(control: none)", it printed
"CLEAN -- no arm read another arm's state". Every input was empty and the verdict was reassuring.
An audit that cannot distinguish "no contamination" from "no data" is worse than no audit.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("asi", ROOT / "scripts" / "audit_store_identity.py")
asi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asi)


def mkrun(d: pathlib.Path, *, stack=None, spec=None, prov=None):
    d.mkdir(parents=True, exist_ok=True)
    if stack is not None:
        (d / "controller_stack.json").write_text(json.dumps({"controllers": stack}))
    if spec is not None:
        (d / "controller_spec.json").write_text(json.dumps(spec))
    if prov is not None:
        (d / "cache_provenance.json").write_text(json.dumps(prov))
    return d


def prov(base="k0", verdict="MISS", cache="/c/a", variant=None):
    return {"base_key": base, "verdict": verdict, "cache": cache, "variant_key": variant,
            "store_code_fingerprint": "fp", "gate_config_fingerprint": "gc"}


# ---------------------------------------------------------------------------------------------------
# NO DATA IS NOT A CLEAN BILL
# ---------------------------------------------------------------------------------------------------

def test_missing_provenance_is_UNDETERMINED_not_clean(tmp_path):
    runs = {t: mkrun(tmp_path / t, stack=["/s/a.json"]) for t in ("a", "b")}
    out = asi.audit(runs)
    assert out["undetermined"] is True
    assert out["contaminated"] is False
    assert "UNDETERMINED" in out["verdict"]
    assert "NOT a clean bill" in out["verdict"]


def test_a_run_with_no_controller_at_all_is_UNDETERMINED(tmp_path):
    """If nothing reports a controller, the audit has no differing stacks to compare."""
    runs = {t: mkrun(tmp_path / t, prov=prov()) for t in ("a", "b")}
    out = asi.audit(runs)
    assert out["undetermined"] is True


def test_full_data_with_separate_cache_dirs_is_CLEAN(tmp_path):
    runs = {
        "ctl": mkrun(tmp_path / "ctl", stack=["/s/inc.json"], prov=prov(cache="/c/ctl")),
        "arm": mkrun(tmp_path / "arm", stack=["/s/inc.json"], spec={"name": "cand",
                                                                   "phase": "prereq"},
                     prov=prov(cache="/c/arm")),
    }
    out = asi.audit(runs)
    assert out["undetermined"] is False
    assert out["contaminated"] is False
    assert out["verdict"].startswith("CLEAN")


# ---------------------------------------------------------------------------------------------------
# THE ACTUAL FINDING: the harness key is controller-blind
# ---------------------------------------------------------------------------------------------------

def test_a_shared_harness_key_across_DIFFERENT_controllers_is_reported(tmp_path):
    """Measured on the real round-2 arms: one base_key for three different controller stacks."""
    runs = {
        "ctl": mkrun(tmp_path / "ctl", stack=["/s/inc.json"], prov=prov(cache="/c/ctl")),
        "arm": mkrun(tmp_path / "arm", stack=["/s/inc.json"], spec={"name": "cand",
                                                                   "phase": "prereq"},
                     prov=prov(cache="/c/arm")),
    }
    out = asi.audit(runs)
    assert out["harness_key_is_controller_blind"] is True
    assert sorted(out["harness_keys_shared_by_multiple_arms"]["k0"]) == ["arm", "ctl"]
    # ...but NOT contamination, because the cache directories differ.
    assert out["contaminated"] is False


def test_a_SHARED_cache_dir_across_different_controllers_IS_contamination(tmp_path):
    """Same key AND same store instance: one arm would be scored against the other's state."""
    runs = {
        "ctl": mkrun(tmp_path / "ctl", stack=["/s/inc.json"], prov=prov(cache="/c/SHARED")),
        "arm": mkrun(tmp_path / "arm", stack=["/s/inc.json"], spec={"name": "cand",
                                                                   "phase": "prereq"},
                     prov=prov(cache="/c/SHARED")),
    }
    out = asi.audit(runs)
    assert out["contaminated"] is True
    assert out["verdict"].startswith("CONTAMINATED")
    assert out["actual_shared_store_instances"][0]["shared_cache_dir"] == "/c/SHARED"


def test_identical_controllers_sharing_a_store_is_CORRECT_reuse(tmp_path):
    """Same controller, same key, same dir: that is caching working, not contamination."""
    runs = {
        "a": mkrun(tmp_path / "a", stack=["/s/inc.json"], prov=prov(cache="/c/SHARED")),
        "b": mkrun(tmp_path / "b", stack=["/s/inc.json"], prov=prov(cache="/c/SHARED")),
    }
    out = asi.audit(runs)
    assert out["contaminated"] is False


# ---------------------------------------------------------------------------------------------------
# WHAT THE CORE KEY WOULD HAVE DONE
# ---------------------------------------------------------------------------------------------------

def test_the_core_key_DOES_cover_a_prereq_controller(tmp_path):
    """state_cache_identity is the fix; this records that it would have separated these arms."""
    runs = {
        "ctl": mkrun(tmp_path / "ctl", stack=["/s/inc.json"], prov=prov(cache="/c/ctl")),
        "arm": mkrun(tmp_path / "arm", stack=["/s/inc.json"], spec={"name": "cand",
                                                                   "phase": "prereq"},
                     prov=prov(cache="/c/arm")),
    }
    out = asi.audit(runs)
    rows = {r["tag"]: r for r in out["rows"]}
    assert rows["arm"]["covers_controller"] is True
    assert rows["arm"]["would_be_key"] != rows["ctl"]["would_be_key"], \
        "the core key must separate arms the harness key conflates"


def test_arms_that_rebuilt_are_named(tmp_path):
    """A MISS cannot have read anyone else's state -- the strongest single piece of evidence."""
    runs = {"a": mkrun(tmp_path / "a", stack=["/s/x.json"], prov=prov(verdict="MISS"))}
    out = asi.audit(runs)
    assert out["arms_that_rebuilt_their_own_store"] == ["a"]
