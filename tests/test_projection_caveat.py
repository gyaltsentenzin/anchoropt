"""A projected firing count must never be readable as measured support.

Pins the 201x overstatement: BFCL's `redundant_proposed_write` arm projected 201 of 590 firings while
the live comparator found exactly ONE genuinely redundant write. `candidate_selection` ranks by
support, so an overstated projection can send a deterministic policy to spend a GPU round on a
one-case mechanism.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchoropt.learning.external_evaluation import arm_manifest        # noqa: E402


class _Arm:
    def __init__(self):
        class V:
            value = "post_generation_pre_exec"
        class A:
            value = "suppress"
        self.label = "x/redundant_proposed_write/suppress:cancel_proposed"
        self.boundary, self.action = V(), A()
        self.signal = "redundant_proposed_write"
        self.eta, self.theta = {}, {}
        # `fires_on_states` is added by the DRIVER after arm_manifest returns, which is precisely why
        # the caveat has to live on the manifest core emits rather than beside the driver's number.
        self.fires_on_states, self.total_states = 201, 590
        class Op:
            value = "suppress"
        class Inst:
            operator, variant = Op(), "cancel_proposed"
        self.instantiated = Inst()


def test_the_manifest_carries_the_projection_caveat():
    m = arm_manifest([_Arm()], incumbent_id="inc", incumbent_token="tok")
    assert "projection_caveat" in m, "a projected count must ship with its caveat"
    cav = m["projection_caveat"]
    assert "UPPER BOUND" in cav
    assert "never measured support" in cav
    assert "201" in cav, "the measured overstatement must be named, not described vaguely"


def test_the_caveat_names_the_blindness_and_not_just_the_number():
    cav = arm_manifest([_Arm()], incumbent_id="i")["projection_caveat"]
    assert "blind to state it does not hold" in cav
