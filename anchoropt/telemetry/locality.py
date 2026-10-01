"""How LOCAL is a learned controller? -- the intervention-locality metrics.

A controller that fixes 4 cases by perturbing 40 is a different object from one that fixes 4 by
touching 24, even at identical accuracy. These metrics make that difference visible.

REUSES THE FROZEN ELIGIBILITY VOCABULARY rather than re-deriving it: `opportunity_created` and
`opportunity_missed` come from `anchoropt.learning.prestate_eligibility.engagement_split`, which was
frozen before the seed-2 run. Restating those definitions here would let the two drift apart, and the
whole point of freezing was that they cannot.

THE OFF-TARGET NUMBER IS THE LOAD-BEARING ONE. `changed_not_fired` counts episodes whose outcome
differs from the control although the controller never fired there. Those changes cannot be the
intervention -- the controller did nothing -- so they are substrate nondeterminism, and on this
substrate they are LARGE (11 of 89 episodes drifted before any injection at seed 42). An evaluation
that reports only accuracy attributes them to the arm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from anchoropt.learning.prestate_eligibility import (
    FIRE_KEY, engagement_split, first_fire_index, qualifying_in_control,
)


@dataclass(frozen=True)
class LocalityReport:
    """Locality of one arm against one control."""

    arm: str
    n_episodes: int
    n_target: int
    n_fired: int
    n_fired_on_target: int
    gains_fired: int
    losses_fired: int
    changed_not_fired: int
    gains_not_fired: int
    losses_not_fired: int
    opportunity_created: int
    opportunity_missed: int

    @property
    def episode_touch_rate(self) -> float | None:
        return self.n_fired / self.n_episodes if self.n_episodes else None

    @property
    def target_coverage(self) -> float | None:
        """Fraction of the control's target states the controller actually acted on."""
        return self.n_fired_on_target / self.n_target if self.n_target else None

    @property
    def on_target_precision(self) -> float | None:
        """Of the episodes it fired on-target, how many did it convert."""
        return self.gains_fired / self.n_fired_on_target if self.n_fired_on_target else None

    @property
    def off_target_firing_rate(self) -> float | None:
        """Fraction of firings that were NOT on a control target state."""
        return self.opportunity_created / self.n_fired if self.n_fired else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "counts": {
                "episodes": self.n_episodes, "target_states": self.n_target,
                "fired": self.n_fired, "fired_on_target": self.n_fired_on_target,
                "opportunity_created": self.opportunity_created,
                "opportunity_missed": self.opportunity_missed,
            },
            "fired_episodes": {"gains": self.gains_fired, "losses": self.losses_fired},
            "non_fired_episodes": {"changed": self.changed_not_fired,
                                   "gains": self.gains_not_fired,
                                   "losses": self.losses_not_fired,
                                   "note": "changes here cannot be the intervention -- the "
                                           "controller never fired; they are substrate drift"},
            "rates": {"episode_touch_rate": self.episode_touch_rate,
                      "target_coverage": self.target_coverage,
                      "on_target_precision": self.on_target_precision,
                      "off_target_firing_rate": self.off_target_firing_rate},
        }


def locality_report(arm: str,
                    control_steps: Mapping[str, Sequence[Mapping[str, Any]]],
                    arm_steps: Mapping[str, Sequence[Mapping[str, Any]]],
                    control_correct: Mapping[str, bool],
                    arm_correct: Mapping[str, bool],
                    *, fire_key: str = FIRE_KEY) -> LocalityReport:
    common = sorted(set(control_steps) & set(arm_steps) & set(control_correct) & set(arm_correct))
    split = engagement_split(control_steps, arm_steps, fire_key)
    target = {c for c in common if qualifying_in_control(control_steps[c])}
    fired = {c for c in common if first_fire_index(arm_steps[c], fire_key) is not None}

    g_f = sum(1 for c in fired if arm_correct[c] and not control_correct[c])
    l_f = sum(1 for c in fired if control_correct[c] and not arm_correct[c])
    nf = [c for c in common if c not in fired]
    g_nf = sum(1 for c in nf if arm_correct[c] and not control_correct[c])
    l_nf = sum(1 for c in nf if control_correct[c] and not arm_correct[c])
    return LocalityReport(
        arm=arm, n_episodes=len(common), n_target=len(target), n_fired=len(fired),
        n_fired_on_target=len(split["on_target_shared"] & set(common)),
        gains_fired=g_f, losses_fired=l_f,
        changed_not_fired=g_nf + l_nf, gains_not_fired=g_nf, losses_not_fired=l_nf,
        opportunity_created=len(split["opportunity_created"] & set(common)),
        opportunity_missed=len(split["opportunity_missed"] & set(common)))
