"""The CCTU cycle-1 driver must not decide WHERE, WHAT or HOW either.

`tests/test_cycle2_thinness.py` holds the other benchmark's driver to this rule, for the reason stated
there: "A closed loop that picks its own answer is not evidence that the algorithm picks it." This is
the same guard for `benchmarks/cctu/cycle1_propose.py`, scoped to the PROPOSE phase — there is no
score phase yet, because no arm has run.

What is additionally pinned here, and each because skipping it would quietly change what the round
means:

  * the evidence is TRAJECTORY-derived, and `proposed_behavior_change` stays EMPTY. Filling it would
    be the driver proposing the intervention under the search's name.
  * the residual is ranked in DISTINCT FAILING EPISODES, not violation events.
  * the candidate set is narrowed by FIRING BEHAVIOUR, never by name — a driver that narrowed by
    hunch would be choosing the answer while appearing not to.
  * a null evaluator, so nothing is preferred offline.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "benchmarks" / "cctu" / "cycle1_propose.py"


def _code() -> str:
    """Source with the module docstring and comments stripped.

    The driver NAMES the forbidden imports in its docstring, to record that they are absent. A test
    that counted that as a violation would forbid documenting the property.
    """
    src = DRIVER.read_text().split('"""', 2)[-1]
    return "\n".join(line.split("#", 1)[0] for line in src.splitlines())


def test_the_driver_exists():
    assert DRIVER.exists(), "the cycle-1 propose driver is missing"


def test_the_driver_cannot_name_a_coordinate():
    """WHERE, WHAT and HOW belong to the core. A driver that can name one can pick one."""
    code = _code()
    for forbidden in ("IncisionPoint", "AnchorPolicyOpt", "SearchSpaceProposal",
                      "expand_and_resume", "group_blocked_by_clause", "build_arms",
                      "synthesize(", "feasible_actions("):
        assert forbidden not in code, (
            f"the driver references {forbidden!r} in code -- the search belongs to core, so the "
            f"driver must not be able to steer it")


def test_the_driver_delegates_to_optimize_residual():
    assert "optimize_residual(" in _code(), "the driver must call the core search"


def test_the_core_call_receives_evidence_only():
    """No locus, no signal, no action may cross the boundary into the search."""
    code = _code()
    call = code[code.index("optimize_residual("):]
    call = call[:call.index("\n\n")]
    for leak in ("signals=", "boundaries=", "locus", "action="):
        assert leak not in call, f"the core call passes {leak!r}, which would steer the search"


def test_the_evaluator_is_null_so_nothing_is_preferred_offline():
    """With a null evaluator every candidate scores identically, so the offline phase cannot rank."""
    code = _code()
    assert "evaluate=lambda _a: 0" in code
    assert "improves=lambda _o: False" in code


def test_the_behavioural_content_field_is_left_empty():
    """`proposed_behavior_change` is a provider's statement of what the agent should do instead.

    The driver filling it would be the driver proposing the intervention. Pinned as a literal because
    it is a one-word change to break and nothing else would notice.
    """
    code = _code()
    assert re.search(r'"proposed_behavior_change":\s*""', code), (
        "proposed_behavior_change must be written as an empty string, so the driver cannot smuggle a "
        "remedy into the evidence")


def test_the_consequential_decision_is_derived_from_the_trace():
    """Attribution walks the episode to its first refused proposal, rather than keying on a string."""
    code = _code()
    assert "def attribute(" in code
    assert "post_execution" in code, "attribution must read the boundary the refusal appears at"
    assert "violation_class" in code, "attribution must read the validator's own verdict"


def test_the_residual_is_ranked_in_episodes_and_the_event_ordering_is_shown():
    """Ranking by event volume buries the best target; both orderings must be reported."""
    code = _code()
    assert "by_linked_episodes" in code and "by_event_volume" in code
    assert "build_residual_problems(" in code, "grouping must use the core's own unit"


def test_the_candidate_set_is_narrowed_by_firing_behaviour():
    """The only legitimate reduction: two candidates that fire on the same states are one arm."""
    code = _code()
    assert "firing_vector(" in code, "narrowing must use the behavioural fingerprint"
    assert "def dedupe_behaviourally(" in code
    # and it must not narrow by anything that encodes a preference
    for smell in ("sort(key=", "sorted(candidates", "[:3]", "[:1]", "top_candidate"):
        assert smell not in code, (
            f"{smell!r} suggests the driver ranks or truncates candidates, which is measurement's job")


def test_the_oracle_guard_runs():
    code = _code()
    assert "assert_no_oracle_leak()" in code


def test_every_written_spec_is_validated_before_it_lands():
    """A spec that cannot be installed must fail here, not at the start of a paid run."""
    code = _code()
    assert "spec.validate()" in code


def test_no_benefit_is_claimed_offline():
    """The propose phase ends at structural rediscovery. Saying so is part of the output."""
    assert "NO BENEFIT IS CLAIMED" in DRIVER.read_text()
