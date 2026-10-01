"""A controller seam must supply every field its boundary's signals read, and dispatch BY SIGNAL.

THE THREE DEFECTS THIS PINS (rounds/AUTONOMY/R13/VOIDED_FIRST_LAUNCH.json). I added a generic seam at
the BFCL answer boundary and it voided six GPU arms before producing a number:

  1. MISSING FIELD DEFAULTS TO A REAL VALUE. The seam's state omitted `tool_calls_so_far`. The
     neighbouring `no_tool_call_at_all` reads `int(state.get("tool_calls_so_far", 0)) == 0`, so the
     ABSENT field became a genuine zero and that signal fired at the seam on every episode -- even
     after two retrievals. Declared-vs-supplied inverted: not a field nobody writes, but one whose
     absence is indistinguishable from a real observation.

  2. DISPATCH BY ITERATION ORDER IS DISPATCH BY POSITION. The seam looped `installed(locus)` and took
     the first controller that fired, so a neighbour at the same locus answered for a condition the
     seam was not built to give it -- and its firing was recorded under the seam's own telemetry keys.

  3. THE FIX EXPOSED A THIRD. With an allowlist in place nothing fired: the loaded controller is a
     `SpecPredicate` exposing only (name, phase, eta, spec, fires_on), so `getattr(c, "signal")` is
     ABSENT and returned "" for everything. The signal lives at `c.spec["signal"]`.

These are seam defects, not predicate defects, so no probe over hand-built states can catch them --
which is exactly why they reached the cluster.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

PATCH = REPO / "patches/bv/bv_a2_answer_boundary_seam.py"


def _patch_src() -> str:
    return PATCH.read_text()


def _seam_supplied_fields() -> set[str]:
    src = _patch_src()
    block = src[src.index('_ast = {'):src.index('step_record["a2_state_best_similarity"]')]
    return set(re.findall(r'"([a-z_]+)":', block))


def test_the_seam_supplies_every_field_ITS_OWN_signal_reads():
    """Defect 1, stated precisely. The seam owes fields to the signals it DISPATCHES to.

    Not to every signal at the boundary: four others are declared here and the seam deliberately does
    NOT supply their evidence, which is why it must skip them (defect 2) rather than let them answer on
    defaults. The two invariants are complements -- this test and the next are one rule split in half.
    """
    import bfcl_runtime as RT

    src = _patch_src()
    owned = set(re.findall(r'"([a-z_]+)"', src[src.index("_A2_SEAM_SIGNALS = frozenset("):
                                              src.index("_A2_SEAM_SIGNALS = frozenset(") + 200]))
    assert owned, "could not read the seam's own signal allowlist"
    supplied = _seam_supplied_fields()
    missing = {sig: sorted(set(RT.signal_fields(sig)) - supplied) for sig in owned
               if set(RT.signal_fields(sig)) - supplied}
    assert not missing, (
        f"the seam dispatches to these signals but does not supply fields they READ: {missing}")


def test_every_OTHER_signal_at_this_boundary_is_either_supplied_or_skipped():
    """The complement, and the one the voided launch actually violated.

    A signal declared here whose fields the seam does NOT supply must be excluded by the allowlist. If
    it is neither supplied nor skipped, it evaluates on defaults -- which is exactly how the zero-call
    signal fired on every episode at `tool_calls_so_far` absent.
    """
    import bfcl_runtime as RT
    from bfcl_signals import SIGNAL_BOUNDARIES

    src = _patch_src()
    owned = set(re.findall(r'"([a-z_]+)"', src[src.index("_A2_SEAM_SIGNALS = frozenset("):
                                              src.index("_A2_SEAM_SIGNALS = frozenset(") + 200]))
    supplied = _seam_supplied_fields()
    here = [s for s, b in SIGNAL_BOUNDARIES.items() if "post_generation_pre_exec" in b]
    assert here, "no signal is declared at this boundary -- has the table moved?"

    unsafe = []
    for sig in here:
        if sig in owned:
            continue                       # dispatched to, and covered by the test above
        if not (set(RT.signal_fields(sig)) - supplied):
            continue                       # fully supplied anyway, so answering here is sound
        unsafe.append(sig)
    # Every remaining signal must be skipped by the allowlist -- which it is, by construction, since
    # the dispatch consults `_A2_SEAM_SIGNALS`. Assert the mechanism is present rather than trusting it.
    assert "_asig not in _A2_SEAM_SIGNALS" in src, (
        f"these signals are declared here and NOT fully supplied {unsafe}, and the seam has no "
        f"allowlist check -- they would answer on defaults")
    # And the specific field whose absence caused the void must now be supplied, because the
    # zero-call signal shares this window through the OTHER seam and a future edit must not drop it.
    assert "tool_calls_so_far" in supplied, (
        "tool_calls_so_far is not supplied; an absent value defaults to 0 and reads as a real "
        "zero-call step")


def test_the_seam_dispatches_by_declared_signal_not_by_iteration_order():
    """Defect 2. An allowlist must exist, and the loop must consult it."""
    src = _patch_src()
    assert "_A2_SEAM_SIGNALS" in src, "the seam has no signal allowlist -- it dispatches blindly"
    assert "_asig not in _A2_SEAM_SIGNALS" in src, (
        "the allowlist exists but the dispatch loop does not consult it")
    # And a skipped controller must be RECORDED, not silently dropped.
    assert "a2_skipped_other_signal" in src, (
        "a skipped controller leaves no trace -- a silent exclusion is unauditable")


def test_the_seam_reads_the_signal_from_the_spec():
    """Defect 3. `SpecPredicate` has no `.signal`; reading it there matches nothing."""
    src = _patch_src()
    assert 'getattr(_ac, "spec", None)' in src, (
        "the seam does not read the controller's spec -- getattr(c, 'signal') is ABSENT on "
        "SpecPredicate and returns '' for every controller, so the allowlist matches nothing")
    assert '_aspec.get("signal")' in src


def test_the_loaded_controller_really_has_no_signal_attribute():
    """The fact defect 3 rests on, asserted against the real loader rather than remembered."""
    loader = REPO / "scripts"
    sys.path.insert(0, str(loader))
    try:
        from install_controller_stack import _install_one          # type: ignore
    except Exception:
        pytest.skip("install_controller_stack not importable in this checkout")
    spec = {"name": "t", "signal": "retrieval_similarity_below_threshold",
            "locus": "post_generation_pre_exec", "action": "reprompt",
            "predicate": {"declared_signal": "retrieval_similarity_below_threshold",
                          "params": {"below": 0.25}},
            "eta": {"instruction": "x"}, "phase": "query"}
    ctl = _install_one(spec, source="test")
    if ctl is None:
        pytest.skip("loader declined the synthetic spec")
    assert not hasattr(ctl, "signal"), (
        "a loaded controller now exposes .signal -- if this became true, the spec-reading fallback is "
        "no longer load-bearing and this test should be updated deliberately")
    assert (getattr(ctl, "spec", {}) or {}).get("signal") == \
        "retrieval_similarity_below_threshold"
