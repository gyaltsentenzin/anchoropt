"""A harness error is an ABSENCE, not a measured zero.

A gateway outage on 2026-09-22 06:31-07:54 made this concrete: 16 of 24 arms had every baseline episode
fail, and 27 of 64 arm evaluations produced no outcome -- yet all of them were recorded with gains and
losses, and every arm reported PHASE=DONE. Scored as unsolved against a clean incumbent, a void arm
reads as a LOSS on every case the incumbent solved: a fabricated measurement in the direction that looks
like evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "tau2"))

pytest.importorskip("tau2", reason="tau2_episodes imports tau2")

import tau2_episodes as E                                        # noqa: E402

assert E.VOID_TOLERANCE > 0, 'the tolerance knob must exist'


def _run(solved, termination, label="arm"):
    r = E.RunResult(label=label)
    r.solved = dict(solved)
    r.reward = {k: (1.0 if v else 0.0) for k, v in solved.items()}
    r.termination = dict(termination)
    return r


INCUMBENT = _run({f"t{i}": i < 4 for i in range(10)},
                 {f"t{i}": "TerminationReason.USER_STOP" for i in range(10)}, "P0")


def test_a_harness_error_is_counted_separately_from_a_scored_zero():
    arm = _run({f"t{i}": False for i in range(10)},
               {f"t{i}": ("harness_error: ServiceUnavailableError" if i < 6
                          else "TerminationReason.USER_STOP") for i in range(10)})
    assert arm.harness_errors == 6
    assert set(arm.scored_cases) == {"t6", "t7", "t8", "t9"}


def test_void_cases_are_excluded_from_the_paired_comparison():
    """Including them counts an infrastructure failure as the intervention's effect."""
    arm = _run({f"t{i}": False for i in range(10)},
               {f"t{i}": ("harness_error: x" if i < 4 else "TerminationReason.USER_STOP")
                for i in range(10)})
    pr = E.paired(arm, INCUMBENT)
    # t0..t3 are the cases the incumbent solved AND the arm could not run. Excluded, so no fake losses.
    assert pr["losses"] == (), pr["losses"]
    assert pr["harness_errors"] == 4
    assert pr["complete"] is False


def test_an_all_void_arm_yields_no_losses_at_all():
    """The worst case: every episode failed. Previously this scored -4 against this incumbent."""
    arm = _run({f"t{i}": False for i in range(10)},
               {f"t{i}": "harness_error: ServiceUnavailableError" for i in range(10)})
    pr = E.paired(arm, INCUMBENT)
    assert pr["gains"] == () and pr["losses"] == ()
    assert pr["n"] == 0
    assert pr["harness_errors"] == 10
    assert pr["complete"] is False


def test_a_clean_arm_is_marked_complete_and_scores_normally():
    arm = _run({f"t{i}": i < 6 for i in range(10)},
               {f"t{i}": "TerminationReason.USER_STOP" for i in range(10)})
    pr = E.paired(arm, INCUMBENT)
    assert pr["complete"] is True
    assert pr["harness_errors"] == 0
    assert set(pr["gains"]) == {"t4", "t5"} and pr["losses"] == ()
    assert pr["net"] == 2


def test_a_max_steps_termination_is_a_real_zero_and_is_NOT_treated_as_void():
    """A truncated episode did run and was scored. It is a measurement -- of the step budget as much as
    the model, which is why the budget must be set to the benchmark's own default."""
    arm = _run({f"t{i}": False for i in range(10)},
               {f"t{i}": "TerminationReason.MAX_STEPS" for i in range(10)})
    assert arm.harness_errors == 0
    pr = E.paired(arm, INCUMBENT)
    assert pr["complete"] is True
    assert len(pr["losses"]) == 4, "a scored zero must still count as a loss"


def test_the_run_record_carries_its_void_count_so_a_reader_cannot_miss_it():
    arm = _run({f"t{i}": False for i in range(10)},
               {f"t{i}": ("harness_error: x" if i < 3 else "TerminationReason.USER_STOP")
                for i in range(10)})
    assert arm.to_json()["harness_errors"] == 3


def test_the_runner_refuses_a_contaminated_incumbent_and_the_matrix_script_agrees():
    """Both halves of the fix must be present: the driver exits 3, and run_arm.sh turns that into a
    FAILED phase rather than the DONE_no_arms that hid the outage."""
    driver = (REPO / "benchmarks" / "tau2" / "run_anchoropt_round.py").read_text()
    assert "baseline_void.json" in driver
    assert "return 3" in driver

    arm_sh = (REPO / "benchmarks" / "tau2" / "scripts" / "run_arm.sh").read_text()
    assert "FAILED_baseline_void" in arm_sh
    assert 'rc" = "3"' in arm_sh

    check = (REPO / "benchmarks" / "tau2" / "scripts" / "check_matrix.sh").read_text()
    assert "VOID" in check and "--audit" in check


def test_the_step_budget_defaults_to_the_benchmarks_own_constant():
    """At 30, 25/25 telecom episodes hit the cap and scored 0 by construction."""
    arm_sh = (REPO / "benchmarks" / "tau2" / "scripts" / "run_arm.sh").read_text()
    assert 'MAX_STEPS:=200' in arm_sh, "the step budget must not silently truncate every episode"


def test_a_cached_baseline_cannot_be_reused_under_a_different_step_budget():
    """Skipping a cached baseline is what makes a re-run cheap, but it would silently ignore a newly
    requested MAX_STEPS -- measuring arms under the old budget while appearing to honour the new one."""
    arm_sh = (REPO / "benchmarks" / "tau2" / "scripts" / "run_arm.sh").read_text()
    assert "FAILED_step_budget_mismatch" in arm_sh
    assert "cached_steps" in arm_sh
    assert "stage_rerun.sh --apply --all" in arm_sh


def test_the_driver_preflights_both_endpoints_before_spending_anything():
    driver = (REPO / "benchmarks" / "tau2" / "run_anchoropt_round.py").read_text()
    assert "def _preflight(" in driver
    assert "preflight_failed.json" in driver
    assert "return 4" in driver
    arm_sh = (REPO / "benchmarks" / "tau2" / "scripts" / "run_arm.sh").read_text()
    assert "FAILED_preflight" in arm_sh


def test_evaluate_aborts_on_consecutive_fully_void_arms_instead_of_burning_the_budget():
    """Two arms in a row where nothing ran is an outage, not two bad arms."""
    driver = (REPO / "benchmarks" / "tau2" / "run_anchoropt_round.py").read_text()
    assert "consecutive_void" in driver
    assert "aborted_on_outage" in driver
    arm_sh = (REPO / "benchmarks" / "tau2" / "scripts" / "run_arm.sh").read_text()
    assert "OPEN_outage" in arm_sh


def test_the_submit_script_does_not_override_the_step_budget_with_a_truncating_default():
    """`submit_matrix.sh` passes MAX_STEPS explicitly via `-env`, so its own default WINS over the
    runner's. It said 30 while the runner said 200: a run asked for 200 got 30, and the cached-baseline
    guard could not catch it because a freshly reset arm has no baseline to compare against. 24 jobs
    were submitted and killed over this."""
    sub = (REPO / "benchmarks" / "tau2" / "scripts" / "submit_matrix.sh").read_text()
    assert 'MAX_STEPS:-200' in sub, "the submit default must not truncate every episode"
    assert "MAX_STEPS=$MAX_STEPS" in sub
    # and it must be visible in the plan, so a wrong budget is caught by reading the output
    assert "the step budget IS the experiment" in sub

    arm = (REPO / "benchmarks" / "tau2" / "scripts" / "run_arm.sh").read_text()
    assert "max_steps=$MAX_STEPS" in arm, "the arm must log the budget it actually received"


def test_the_two_scripts_agree_on_every_budget_default():
    """A default that differs between the submitter and the runner is invisible until it changes the
    result, because the submitter always wins."""
    sub = (REPO / "benchmarks" / "tau2" / "scripts" / "submit_matrix.sh").read_text()
    arm = (REPO / "benchmarks" / "tau2" / "scripts" / "run_arm.sh").read_text()
    import re
    KNOBS = ("TASKS", "MAX_ARMS", "MAX_STEPS", "CONCURRENCY", "SIM_TIMEOUT")

    def defaults(text):
        # Both house styles: `: "${NAME:=V}"` (run_arm) and `NAME="${NAME:-V}"` (submit_matrix).
        out = {}
        for knob in KNOBS:
            m = re.search(r'\$\{' + knob + r'\s*:[=-]\s*([0-9]+)\s*\}', text)
            if m:
                out[knob] = m.group(1)
        return out
    d_sub, d_arm = defaults(sub), defaults(arm)
    shared = set(d_sub) & set(d_arm)
    assert shared, "expected overlapping budget knobs"
    mismatched = {k: (d_sub[k], d_arm[k]) for k in shared if d_sub[k] != d_arm[k]}
    assert mismatched == {}, f"submitter vs runner default mismatch: {mismatched}"


# ============================================================ tolerance: outage vs a dropped connection
def test_a_few_void_cases_do_not_discard_the_whole_run():
    """A shared inference service drops the occasional connection -- litellm reports it as
    `AuthenticationError: ... All connection attempts failed`. An all-or-nothing rule discarded three
    arms that had each completed 21-24 of 25 episodes. What matters is whether enough ran to answer the
    question, not whether anything failed."""
    arm = _run({f"t{i}": i < 5 for i in range(25)},
               {f"t{i}": ("harness_error: AuthenticationError: All connection attempts failed"
                          if i >= 23 else "TerminationReason.USER_STOP") for i in range(25)})
    assert arm.harness_errors == 2
    assert len(arm.scored_cases) == 23
    assert arm.harness_errors / arm.n <= E.VOID_TOLERANCE


def test_the_tolerance_still_rejects_a_real_outage():
    arm = _run({f"t{i}": False for i in range(25)},
               {f"t{i}": "harness_error: ServiceUnavailableError" for i in range(25)})
    assert arm.harness_errors / arm.n > E.VOID_TOLERANCE


def test_the_round_runs_on_the_scored_case_set_so_every_arm_faces_the_same_cases():
    """The incumbent defines the case set. Arms must be loaded from it, not from the full task list, or
    an arm is measured on cases the incumbent was never scored on."""
    driver = (REPO / "benchmarks" / "tau2" / "run_anchoropt_round.py").read_text()
    assert '"scored_task_ids"' in driver
    assert 'base.get("scored_task_ids")' in driver, "evaluate must load the incumbent's scored set"
    # and the incumbent's solved map must be narrowed to it, or gains/losses count absent cases
    assert 'if k in set(case_ids)' in driver


def test_a_partial_arm_measurement_records_its_own_denominator():
    """A 19-of-21 comparison is a real measurement of 19 cases, as long as the row says so."""
    inc = _run({f"t{i}": i < 8 for i in range(21)},
               {f"t{i}": "TerminationReason.USER_STOP" for i in range(21)})
    arm = _run({f"t{i}": i < 10 for i in range(21)},
               {f"t{i}": ("harness_error: x" if i >= 19 else "TerminationReason.USER_STOP")
                for i in range(21)})
    pr = E.paired(arm, inc)
    assert pr["n"] == 19
    assert pr["harness_errors"] == 2
    assert pr["complete"] is False
    assert all(c not in pr["gains"] + pr["losses"] for c in ("t19", "t20"))


def test_the_submitter_never_queues_an_arm_that_already_has_a_live_job():
    """`--retry` skipped only DONE arms, so a retry meant for 4 failed arms resubmitted all 24 and
    every queued arm ended up with two jobs writing to the same directory. PHASE cannot catch this: a
    queued arm has no PHASE yet, so the queue itself must be consulted."""
    sub = (REPO / "benchmarks" / "tau2" / "scripts" / "submit_matrix.sh").read_text()
    assert "LIVE_ARMS" in sub
    assert "already queued or running" in sub
    # and a retry must chain behind what is queued rather than run beside it
    assert "chained after" in sub


def test_the_status_script_resolves_job_state_by_name_not_only_by_a_stored_jobid():
    """A duplicate submission overwrote 20 arms' JOBID files; after the duplicates were killed every
    one of those arms reported EXIT while its real job was still queued. The queue is the truth."""
    check = (REPO / "benchmarks" / "tau2" / "scripts" / "check_matrix.sh").read_text()
    assert "LIVE_BY_NAME" in check
    assert 'lsf_state "$name" "$jid"' in check


def test_within_tolerance_void_is_reported_as_a_denominator_not_an_alarm():
    """A baseline with one dropped connection in 25 proceeds correctly on 24 scored cases. An earlier
    version of the status script flagged it with `!`, counted it VOID, and told the user to re-run an
    arm that was working -- the same misreporting as before, in the pessimistic direction. `!`/VOID is
    reserved for a baseline that was REFUSED; `~` notes an excluded case."""
    check = (REPO / "benchmarks" / "tau2" / "scripts" / "check_matrix.sh").read_text()
    assert "scored_task_ids" in check, "the denominator must come from the scored set"
    assert "'~' if n < r['n']" in check
    # VOID must key on the refusal artifact only
    assert '[ -s "$d/baseline_void.json" ] && is_void=1' in check
    assert "case \"$base\" in *'!'*) is_void=1" not in check


# ============================================================ selection by split, not by prefix
def test_tasks_are_selected_by_the_benchmarks_own_split_not_a_string_prefix():
    """Sorting ids as STRINGS and taking the first N gave airline tasks 0-3 and 10-30 -- because
    "10" < "2" -- which mixed tau-bench's official train and test sets and burned 11 of the 20 airline
    held-out tasks. A lexicographic prefix is deterministic but it is not a sample of anything."""
    ep = (REPO / "benchmarks" / "tau2" / "tau2_episodes.py").read_text()
    assert "def split_ids(" in ep
    assert "get_task_splits_loader" in ep
    drv = (REPO / "benchmarks" / "tau2" / "run_anchoropt_round.py").read_text()
    assert '"--split"' in drv and '"train"' in drv
    sub = (REPO / "benchmarks" / "tau2" / "scripts" / "submit_matrix.sh").read_text()
    assert 'SPLIT="${SPLIT:-train}"' in sub
    assert 'TASKS="${TASKS:-0}"' in sub, "the whole split must be the default, not a subsample"


def test_a_domain_with_no_split_refuses_rather_than_inventing_a_subset():
    ep = (REPO / "benchmarks" / "tau2" / "tau2_episodes.py").read_text()
    assert "defines no task splits" in ep


def test_banking_is_excluded_from_the_matrix():
    """banking_knowledge arrived with tau-3 and is not part of the original tau-2."""
    sub = (REPO / "benchmarks" / "tau2" / "scripts" / "submit_matrix.sh").read_text()
    import re
    m = re.search(r'^DOMAINS=\(([^)]*)\)', sub, re.M)
    assert m, "DOMAINS not found"
    assert "banking" not in m.group(1), m.group(1)
    assert set(m.group(1).split()) == {"telecom", "retail", "airline"}


def test_a_priority_group_is_a_barrier_not_a_hint():
    """Wave packing spilled the next priority group into the current wave, so granite would have
    started before the qwen/minimax self-teach group finished."""
    sub = (REPO / "benchmarks" / "tau2" / "scripts" / "submit_matrix.sh").read_text()
    assert "PRIORITY IS A BARRIER" in sub
    assert 'next_group' in sub and '!= "$group"' in sub


def test_the_status_script_audits_engagement_and_flags_a_winner_that_never_fired():
    """A positive net from an arm whose controller executed nothing is variance, not a repair. The
    status script must surface that rather than let the outcome class speak for it -- qwen/telecom
    measured net +1 with interventions_executed=0 on a 1-case residual and core still reported
    TRAIN_IMPROVED_PENDING_VALIDATION."""
    check = (REPO / "benchmarks" / "tau2" / "scripts" / "check_matrix.sh").read_text()
    # engagement recomputed from the measurements, not trusted from the class
    assert "interventions_executed" in check
    assert "engd" in check, "the table must show how many measured arms actually fired"
    assert "NEVER FIRED is not a repair" in check
    # and it must not depend on the newer selection.json field being present, so old rounds audit too
    assert 'sel.get("winner_interventions_executed")' in check
    assert "best_measured_arm" in check
