"""The frozen common-prestate criterion, including the cases the first proxy got wrong."""

from anchoropt.learning.prestate_eligibility import (
    FIRE_KEY, eligible_cases, first_fire_index, prestate, qualifying_in_control,
)


def step(i, *, status="executed", decoded=(), fired=False, turn=0):
    s = {"turn": turn, "step": i, "status": status, "decoded": list(decoded)}
    if fired:
        s[FIRE_KEY] = True
    return s


def test_first_fire_index_finds_the_firing_step():
    steps = [step(0), step(1, fired=True), step(2)]
    assert first_fire_index(steps) == 1
    assert first_fire_index([step(0), step(1)]) is None


def test_prestate_is_strictly_before_the_firing_step():
    """The firing step is where the arm ACTS, so it is not part of the comparable prefix."""
    steps = [step(0, decoded=["a()"]), step(1, fired=True), step(2, decoded=["b()"])]
    assert len(prestate(steps, first_fire_index(steps))) == 1


def test_non_firing_episode_must_match_everywhere():
    """An arm that never fired had no treatment: its whole trajectory is prefix."""
    steps = [step(0, decoded=["a()"]), step(1, status="answer_end_turn")]
    assert len(prestate(steps, first_fire_index(steps))) == 2


def test_divergence_AFTER_firing_is_the_treatment_not_drift():
    ctl = {"c1": [step(0, decoded=[], status="answer_end_turn")]}
    arm = {"c1": [step(0, decoded=[], fired=True),
                  step(1, decoded=["archival_memory_retrieve(q)"]),
                  step(2, status="answer_end_turn")]}
    elig, drift = eligible_cases({"ctl": ctl, "arm": arm}, control="ctl")
    assert elig == {"c1"}, "post-firing divergence must NOT disqualify the case"
    assert drift["arm"] == set()


def test_divergence_BEFORE_firing_is_drift():
    ctl = {"c1": [step(0, decoded=["core_memory_retrieve(q)"]),
                  step(1, status="answer_end_turn")]}
    arm = {"c1": [step(0, decoded=[]),                    # differs before it fires
                  step(1, decoded=[], fired=True),
                  step(2, status="answer_end_turn")]}
    elig, drift = eligible_cases({"ctl": ctl, "arm": arm}, control="ctl")
    assert elig == set()
    assert drift["arm"] == {"c1"}


def test_SAME_COUNT_DIFFERENT_CALL_is_drift():
    """The step-0-count proxy called these equal. They are not the same experiment.

    This is the concrete reason the criterion compares call TEXT: one prefix read core, the other
    read archival, and a count-only comparison would have admitted the case as causally comparable.
    """
    ctl = {"c1": [step(0, decoded=["core_memory_retrieve(q)"]), step(1, fired=False)]}
    arm = {"c1": [step(0, decoded=["archival_memory_retrieve(q)"]), step(1, fired=True)]}
    elig, drift = eligible_cases({"ctl": ctl, "arm": arm}, control="ctl")
    assert elig == set(), "same decoded COUNT but a different call is a different prestate"
    assert drift["arm"] == {"c1"}


def test_divergence_at_a_LATER_step_before_a_LATER_firing_is_drift():
    """The step-0 proxy read only step 0, so it would have admitted this case."""
    ctl = {"c1": [step(0, decoded=["a()"]), step(1, decoded=["b()"]), step(2)]}
    arm = {"c1": [step(0, decoded=["a()"]), step(1, decoded=["ZZZ()"]), step(2, fired=True)]}
    elig, drift = eligible_cases({"ctl": ctl, "arm": arm}, control="ctl")
    assert elig == set()
    assert drift["arm"] == {"c1"}


def test_eligibility_requires_EVERY_arm_to_match():
    """One drifted arm disqualifies the case for the whole round -- so the two eta arms are always
    compared on the same population, which is the point of a 3-arm round."""
    ctl = {"c1": [step(0, decoded=["a()"]), step(1, decoded=[], status="answer_end_turn")]}
    good = {"c1": [step(0, decoded=["a()"]), step(1, decoded=[], fired=True),
                   step(2, decoded=["r()"])]}
    bad = {"c1": [step(0, decoded=["DRIFTED()"]), step(1, decoded=[], fired=True),
                  step(2, decoded=["r()"])]}
    elig, drift = eligible_cases({"ctl": ctl, "a1": good, "a2": bad}, control="ctl")
    assert elig == set()
    assert drift["a2"] == {"c1"} and drift["a1"] == set()


def test_a_step_ZERO_firing_leaves_NOTHING_to_compare():
    """A KNOWN LIMIT OF THIS CRITERION, recorded rather than hidden.

    When the earliest firing is at step 0 there is no prefix, so every case is trivially eligible and
    the criterion has no discriminating power. That is precisely R3's situation: the reprompt fires on
    the model's FIRST decision. Eligibility there reduces to "the arms agree on nothing observable
    before the intervention", which is vacuously true.

    Consequence for interpretation: on a step-0-firing signal, common-prestate eligibility CANNOT by
    itself certify comparability, and the pre-intervention drift it is meant to detect has to be read
    from the target population's own definition instead -- whether the CONTROL was a zero-call state
    (`qualifying_in_control`) and whether the arm fired there. Both are reported separately.
    """
    ctl = {"c1": [step(0, decoded=[], status="answer_end_turn")]}
    a = {"c1": [step(0, decoded=[], fired=True), step(1, decoded=["r()"])]}
    b = {"c1": [step(0, decoded=["anything()"], fired=True), step(1, decoded=["q()"])]}
    elig, drift = eligible_cases({"ctl": ctl, "a": a, "b": b}, control="ctl")
    assert elig == {"c1"} and drift["b"] == set()


def test_missing_case_is_not_eligible():
    ctl = {"c1": [step(0)], "c2": [step(0)]}
    arm = {"c1": [step(0)]}
    elig, _ = eligible_cases({"ctl": ctl, "arm": arm}, control="ctl")
    assert elig == {"c1"}, "a case absent from an arm is a denominator issue, not a comparison"


def test_target_population_read_from_control_only():
    assert qualifying_in_control([step(0, decoded=[], status="answer_end_turn")])
    assert not qualifying_in_control([step(0, decoded=["r()"], status="answer_end_turn")])
    assert not qualifying_in_control([])


def test_eligibility_never_consults_the_outcome():
    """Two cases with identical prestates but opposite outcomes must be treated identically."""
    ctl = {"win": [step(0, decoded=[], status="answer_end_turn")],
           "lose": [step(0, decoded=[], status="answer_end_turn")]}
    arm = {"win": [step(0, decoded=[], fired=True), step(1, decoded=["r()"])],
           "lose": [step(0, decoded=[], fired=True), step(1, decoded=["r()"])]}
    elig, _ = eligible_cases({"ctl": ctl, "arm": arm}, control="ctl")
    assert elig == {"win", "lose"}
