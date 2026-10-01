"""A query-time controller must not fire during SETUP (prerequisite) construction.

WHY THIS TEST EXISTS. Setup episodes write the state that scored episodes read. The generic runtime
hook lives in the episode loop, and the prerequisite builder CALLS that same loop, so an installed
controller fired during store construction -- scoring an arm's queries against a store the arm itself
changed. Control and arm then answer different questions, and every promote/reject decision carries
the confound.

Measured on the A9 reproduction round, from the `*_gate` prereq sidecars:

    control  27 prereq episodes,  0 firings
    arms     27 prereq episodes,  1 / 2 / 2 firings -- EVERY one with a merge effect

24 of 89 scored queries were transitively contaminated. The exclusion recovered the round; the guard
prevents the next one.

TWO LEVELS, because one is cheap and general and the other is the real thing:
  * the CONTRACT level (this file, pure Python): a controller offered a setup-phase state must not be
    consulted. No cluster, no GPU, runs in CI.
  * the EVALUATOR level: `patches/prereq_query_only_guard.diff` against the cluster evaluator, whose
    proof is a measured round reporting 0 prereq firings. Asserted structurally here.

I ALSO GOT THIS WRONG BY REASONING. I first concluded from the enclosing function that the prereq
builder could not reach the hook. It can. Only reading the sidecars settled it -- which is why the
structural assertions below check the actual patched source rather than describing intent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from anchoropt.runtime_hook import decide, install, installed, reset

REPO = Path(__file__).resolve().parent.parent
PATCH = REPO / "patches" / "prereq_query_only_guard.diff"


class Phi:
    """Fires on everything, so any consultation at all is visible."""

    name = "always"
    eta = {"primitive": "noop"}

    def __init__(self):
        self.seen: list = []

    def fires_on(self, state):
        self.seen.append(dict(state))
        return True


# ------------------------------------------------------------------------------------------------
# CONTRACT LEVEL: a host that guards correctly never consults the controller during setup
# ------------------------------------------------------------------------------------------------


def _host_loop(phi_locus: str, *, setup_tag: str, rows):
    """A minimal stand-in for a host episode loop that runs BOTH phases through one code path.

    That shared path is the whole point -- it is what made the real defect possible.
    """
    fired = []
    for row in rows:
        is_setup = row["phase"] == setup_tag
        # THE GUARD: skip the controller entirely during setup, and RECORD the skip.
        if is_setup:
            fired.append({"case": row["case"], "phase": row["phase"], "skipped_prereq_gate": True})
            continue
        d = decide(phi_locus, row["state"], host_default=False)
        fired.append({"case": row["case"], "phase": row["phase"],
                      "controller_fired_gate": bool(d.proceed and d.by_controller)})
    return fired


ROWS = [
    {"case": "setup_1", "phase": "snap", "state": {"x": 1}},
    {"case": "setup_2", "phase": "snap", "state": {"x": 2}},
    {"case": "query_1", "phase": "eval", "state": {"x": 3}},
    {"case": "query_2", "phase": "eval", "state": {"x": 4}},
]


@pytest.fixture(autouse=True)
def _clean_hook():
    reset()
    yield
    reset()


def test_a_controller_is_never_consulted_during_setup():
    phi = Phi()
    install("post_execution", phi)
    _host_loop("post_execution", setup_tag="snap", rows=ROWS)
    # The predicate records every state it was shown. It must have seen ONLY query states.
    assert len(phi.seen) == 2, f"controller was consulted {len(phi.seen)} times, expected 2 (queries)"
    assert [s["x"] for s in phi.seen] == [3, 4], "a setup state reached the controller"


def test_the_controller_still_fires_on_query_episodes():
    """The guard must not disarm the arm -- that would turn every arm into its own control."""
    phi = Phi()
    install("post_execution", phi)
    out = _host_loop("post_execution", setup_tag="snap", rows=ROWS)
    fired = [r for r in out if r.get("controller_fired_gate")]
    assert len(fired) == 2, "the controller stopped firing on queries too"
    assert all(r["phase"] == "eval" for r in fired)


def test_setup_episodes_record_the_skip_rather_than_going_silent():
    """A silent skip is indistinguishable from a controller that was never installed."""
    install("post_execution", Phi())
    out = _host_loop("post_execution", setup_tag="snap", rows=ROWS)
    skipped = [r for r in out if r.get("skipped_prereq_gate")]
    assert len(skipped) == 2
    for r in skipped:
        assert not r.get("controller_fired_gate")


def test_no_controller_installed_means_no_firing_anywhere():
    """Baseline sanity: the control arm must be inert in both phases."""
    assert not installed("post_execution")
    out = _host_loop("post_execution", setup_tag="snap", rows=ROWS)
    assert not [r for r in out if r.get("controller_fired_gate")]


def test_an_unguarded_loop_DOES_contaminate_setup():
    """The bug, reproduced -- so these tests cannot pass vacuously.

    If this ever stops failing to guard, the guard above is testing nothing.
    """
    phi = Phi()
    install("post_execution", phi)
    for row in ROWS:                       # no phase check at all: the pre-fix behaviour
        decide("post_execution", row["state"], host_default=False)
    assert len(phi.seen) == 4, "expected the unguarded loop to consult the controller on setup too"
    assert 1 in [s["x"] for s in phi.seen], "setup state should have reached the controller here"


# ------------------------------------------------------------------------------------------------
# EVALUATOR LEVEL: the shipped patch really gates the call, and names the skip so it survives
# ------------------------------------------------------------------------------------------------


def test_the_shipped_patch_exists_and_gates_the_hook_call():
    if not PATCH.exists():
        pytest.skip("patch not present in this checkout")
    src = PATCH.read_text()
    assert 'str(rollout_tag) == "snap"' in src, "the patch does not discriminate the setup phase"
    assert re.search(r"_cb = None if _cb_is_prereq else _hook_decide\(", src), \
        "the patch does not actually gate the hook decision"


def test_the_skip_telemetry_is_named_to_survive_the_sidecar():
    """The sidecar keeps `*_gate` keys and drops unknown ones -- a badly named skip is invisible."""
    if not PATCH.exists():
        pytest.skip("patch not present in this checkout")
    src = PATCH.read_text()
    m = re.search(r'step_record\["(controller_skipped_[a-z_]*)"\]', src)
    assert m, "the patch records no skip marker"
    assert m.group(1).endswith("_gate"), f"{m.group(1)!r} would be dropped by the sidecar"


def test_the_patch_does_not_also_disarm_query_episodes():
    """The guard must be a phase check only -- not a blanket disable."""
    if not PATCH.exists():
        pytest.skip("patch not present in this checkout")
    src = PATCH.read_text()
    # the gated expression must still call the hook on the else branch
    assert "else _hook_decide(" in src
    assert "_cb = None\n" not in src, "the patch disables the hook unconditionally"


# ---- phase scoping: narrowing the guard, not weakening it ---------------------------------------
#
# A storage-phase residual needs a controller that MAY act during prerequisite construction. The
# guard therefore became phase-SCOPED: a controller declares the phase it is valid in, and the default
# for an undeclared controller is still "not allowed during prereqs". Query-time controllers -- A9
# included, which declares nothing -- stay exactly as inert as before.

PATCH_UP = REPO / "patches" / "upstream_preexec_hook_and_phase_scope.diff"


class PhasePhi(Phi):
    """A controller that declares itself valid during storage construction."""

    def __init__(self, phase):
        super().__init__()
        self.phase = phase


def _phase_eligible(ctl, is_prereq):
    """The live evaluator's rule, mirrored. SYMMETRIC in both directions.

    An earlier version short-circuited on `not is_prereq`, which made every controller eligible during
    a query regardless of its declared phase -- so a prereq-only controller was confined to storage in
    one direction and unconfined in the other.
    """
    ph = str(getattr(ctl, "phase", "query")).lower()
    if ph == "any":
        return True
    return ph == ("prereq" if is_prereq else "query")


def _scoped_loop(*, setup_tag, rows, controllers):
    """Host loop whose guard consults each controller's DECLARED phase, in both directions."""
    out = []
    for row in rows:
        is_setup = row["phase"] == setup_tag
        allowed = True
        if is_setup:
            allowed = any(_phase_eligible(c, True) for c in controllers)
            out.append({"case": row["case"], "phase": row["phase"],
                        ("controller_prereq_allowed_gate" if allowed
                         else "controller_skipped_prereq_gate"): True})
            if not allowed:
                continue
        d = decide("post_execution", row["state"], host_default=False)
        rec = {"case": row["case"], "phase": row["phase"],
               "controller_fired_gate": bool(d.proceed and d.by_controller)}
        if is_setup:
            rec["controller_prereq_allowed_gate"] = True
        out.append(rec)
    return out


def test_a_query_only_controller_is_STILL_inert_during_setup():
    """A9 declares no phase. It must behave exactly as it did before phase scoping existed."""
    phi = Phi()                                  # no `phase` attribute at all
    install("post_execution", phi)
    out = _scoped_loop(setup_tag="snap", rows=ROWS, controllers=[phi])
    assert [s["x"] for s in phi.seen] == [3, 4], "a query-only controller saw a setup state"
    fired = [r for r in out if r.get("controller_fired_gate")]
    assert all(r["phase"] == "eval" for r in fired)
    assert len([r for r in out if r.get("controller_skipped_prereq_gate")]) == 2


@pytest.mark.parametrize("phase", ["prereq", "any", "PREREQ", "Any"])
def test_a_controller_declaring_the_storage_phase_is_permitted_there(phase):
    phi = PhasePhi(phase)
    install("post_execution", phi)
    _scoped_loop(setup_tag="snap", rows=ROWS, controllers=[phi])
    assert [s["x"] for s in phi.seen] == [1, 2, 3, 4], \
        f"phase={phase!r} was declared but the controller never saw the setup states"


@pytest.mark.parametrize("phase", ["query", "", "somethingelse"])
def test_any_other_declaration_stays_inert_during_setup(phase):
    """The default must be restrictive: an unrecognised phase is NOT a permission."""
    phi = PhasePhi(phase)
    install("post_execution", phi)
    _scoped_loop(setup_tag="snap", rows=ROWS, controllers=[phi])
    assert [s["x"] for s in phi.seen] == [3, 4], f"phase={phase!r} wrongly permitted setup access"


# ---- the shipped upstream (pre-execution) integration -------------------------------------------

def test_the_upstream_hook_reads_the_DISPATCH_list_not_the_decoded_list():
    """`_mg_exec_calls` is filtered at three points above the seam.

    A controller reading `decoded` would judge calls that may never dispatch -- and would then
    "prevent" something that was not going to happen, which is indistinguishable from working.
    """
    if not PATCH_UP.exists():
        pytest.skip("upstream patch not present in this checkout")
    src = PATCH_UP.read_text()
    assert "for _uc_i, _uc_call in enumerate(_mg_exec_calls or [])" in src
    assert "_mg_exec_calls = [c for _i, c in _up_keep]" in src, "withholding must rebuild the list"


def test_the_upstream_hook_enforces_a_retry_budget():
    if not PATCH_UP.exists():
        pytest.skip("upstream patch not present")
    src = PATCH_UP.read_text()
    assert "_spent < _budget" in src
    assert "upstream_budget_exhausted_gate" in src, "budget exhaustion must be visible"


def test_the_upstream_state_carries_no_post_execution_field():
    """Expose only what is genuinely observable at that boundary.

    Supplying a realized result upstream would make the WHERE comparison meaningless: the arm would
    be deciding on information the boundary does not have.
    """
    if not PATCH_UP.exists():
        pytest.skip("upstream patch not present")
    src = PATCH_UP.read_text()
    block = src.split("_up_ctls:")[-1].split("self._xl_site")[0]
    for post_only in ("best_similarity", "tool_results", "execution_results", "error_kind"):
        assert post_only not in block, f"{post_only!r} is post-execution and must not appear upstream"


def test_upstream_telemetry_survives_the_sidecar_and_records_retained_information():
    if not PATCH_UP.exists():
        pytest.skip("upstream patch not present")
    src = PATCH_UP.read_text()
    for key in ("upstream_fired_gate", "upstream_withheld_gate",
                "upstream_chars_withheld_gate", "upstream_hook_gate"):
        assert key in src, f"{key} missing"
        assert key.endswith("_gate"), f"{key} would be dropped by the sidecar"


def test_a_prereq_only_controller_cannot_act_during_QUERIES():
    """Direction 2, which the first implementation got wrong.

    The rule short-circuited on `not is_prereq`, so during a query EVERY controller was eligible
    regardless of declared phase. A controller declared for storage construction must not act on
    scored queries: its residual does not live there, and any effect would be attributed to the wrong
    mechanism.
    """
    assert _phase_eligible(PhasePhi("prereq"), True) is True
    assert _phase_eligible(PhasePhi("prereq"), False) is False


def test_phase_eligibility_is_symmetric():
    q, pr, both = Phi(), PhasePhi("prereq"), PhasePhi("any")
    assert (_phase_eligible(q, True), _phase_eligible(q, False)) == (False, True)
    assert (_phase_eligible(pr, True), _phase_eligible(pr, False)) == (True, False)
    assert (_phase_eligible(both, True), _phase_eligible(both, False)) == (True, True)


def test_an_undeclared_controller_still_behaves_as_query_only():
    """Adding phase support must not change how an existing controller behaves."""
    assert _phase_eligible(Phi(), False) is True
    assert _phase_eligible(Phi(), True) is False
