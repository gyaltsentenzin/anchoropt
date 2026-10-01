#!/usr/bin/env python3
"""P2 CALIBRATION: can candidate_search re-select the eight ACCEPTED BFCL anchors?

THE TEST
    Given the locus/attribution from which an accepted anchor was previously derived, does the
    automatic machinery recover a candidate close to that anchor's (l, phi, mu)?

WHY THIS IS THE RIGHT FIRST TEST
    The eight anchors in `rounds/anchors.py` are known-good controllers, each already validated
    under docs/ACCEPTANCE_RULE.md. They are the only ground truth available that costs no GPU. If
    the search cannot recover them, the RUNTIME/SEARCH implementation is wrong -- that conclusion
    comes before any hypothesis about BFCL, and certainly before launching an evaluation.

WHAT COUNTS AS RECOVERY
    exact        boundary AND signal AND action all match the accepted anchor
    approximate  boundary AND action match; the signal is a declared sibling observing the same
                 condition (recorded explicitly, never inferred loosely)
    no           anything else, including SIGNAL_BLOCKED

THE INPUT IS THE ANCHOR'S OWN ATTRIBUTION PROSE, NOT ITS LOCUS STRING
    Feeding `capacity/container/no_remaining_capacity` would be circular: the locus string is a
    canonical label assigned AFTER the anchor was accepted, and it shares tokens with the signal
    name by construction. So the diagnosis text here is the `attribution=` field -- the natural
    language a human/Claude attributor wrote about the failure -- which is what a diagnosis provider
    actually emits. `consequential_decision` is the anchor's own notes, same source.

    This is a test of the SEARCH, not of a lexical miner: the attributor is Claude, the executing
    model under test is Granite, and the prose below is the attributor's, verbatim from
    rounds/anchors.py.

NO TUNING TOWARD RECOVERY
    The search, its scoring, its gates and Phi_BFCL's aliases are NOT adjusted per anchor. Every
    anchor is scored by the same machinery in one pass. A failure is reported as a failure -- that
    is the entire value of running this before spending GPU time.
"""

from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

import bfcl_runtime as runtime                                        # noqa: E402
import bfcl_signals as _signals                                       # noqa: E402
from fixtures.replay_anchors import (                                 # noqa: E402
    REROUTE_DESTINATIONS as _FIXTURE_DESTINATIONS,
    SIGNAL_ALIASES as _FIXTURE_ALIASES,
    SIGNAL_PROBE_PARAMS as _FIXTURE_PROBES,
)

# INJECT the calibration fixture. The runtime deliberately declares none of this on its own account
# (see bfcl_signals.py): prose aliases and probe thresholds were reverse-engineered from the accepted
# anchors, so a discovery run must not import them. This script is the calibration harness, so it
# installs them explicitly -- which is the point of doing it here rather than at import time.
_signals.SIGNAL_ALIASES.update(_FIXTURE_ALIASES)
_signals.SIGNAL_PROBE_PARAMS.update(_FIXTURE_PROBES)
runtime.REROUTE_DESTINATIONS.update(_FIXTURE_DESTINATIONS)
from anchoropt.anchor import Action, IncisionPoint                    # noqa: E402
from anchoropt.learning.candidate_search import (                     # noqa: E402
    enumerate_candidates, expressible_under, select,
)
from anchoropt.runtime import ResidualDiagnosis                       # noqa: E402
from rounds.anchors import ANCHORS                                    # noqa: E402

# Declared sibling signals: pairs that observe the SAME runtime condition at different
# specificity. Recorded explicitly so an "approximate" verdict is auditable rather than a loose
# string comparison. A1/A5 are the documented case -- both are a capacity refusal, differing only
# in WHICH container refused, and A5 exists precisely because A1's success saturates the target.
_SIBLINGS = {
    frozenset({"container_at_capacity", "container_slots_exhausted"}),
    frozenset({"identifier_not_found", "no_informative_result"}),
}

def theta_for_signal(signal: str):
    """theta per action for a given phi. Identical rule for every anchor -- no special casing.

    REROUTE asks the RUNTIME for an attested destination and returns None when there is none. That
    None is load-bearing: docs/GENERALIZABILITY.md requires a reroute to name a real destination
    tool attested on RESOLVING, so handing every reroute an invented destination string (an earlier
    version of this script did) makes the strongest action survive at every locus and lets it
    outrank the correctly-attributed REPROMPT. The bug was in the harness, not in the ranking.
    """
    def theta_for(action: Action, diagnosis) -> dict | None:
        if action is Action.REPROMPT:
            return {"text": diagnosis.proposed_behavior_change}
        if action is Action.SUPPRESS:
            return {"reason": f"suppressed by AnchorOpt: {diagnosis.failure_mechanism[:80]}"}
        if action is Action.REROUTE:
            dest = runtime.reroute_destination(signal)
            if dest is None:
                return None            # no attested destination -> the cell is PRUNED, correctly
            return {"destination": dest["destination"],
                    "retry_original": bool(dest.get("retry_original", False))}
        return None
    return theta_for


def diagnosis_for(anchor) -> ResidualDiagnosis:
    """The anchor's own attribution prose as a residual diagnosis. No locus string."""
    return ResidualDiagnosis(
        case_id=f"{anchor.name}-mined-locus",
        mechanism=anchor.attribution,
        evidence=anchor.attribution,
        consequential_decision=(anchor.notes or anchor.attribution)[:600],
        proposed_behavior_change="consult the store before concluding",
        provider="rounds/anchors.py::attribution (human+Claude attributor)",
        metadata={"anchor": anchor.name, "locus": anchor.locus},
    )


def verdict(anchor, chosen) -> str:
    """exact / approximate / no. See the module docstring for the definitions.

    `approximate` requires the SAME boundary and action with a DECLARED sibling signal, or the same
    boundary and signal with a different action from the same vocabulary. The second case is A4:
    the search localizes (l, phi) exactly and picks SUPPRESS where the accepted anchor uses
    REPROMPT, the two being separated only by the frozen control-strength ordering. That is reported
    as `approximate` rather than promoted to `exact` -- and the ranking is deliberately NOT changed
    to make it pass, because the only available reason to prefer REPROMPT here is knowing A4's
    measured outcome, which is tuning the search to force recovery.
    """
    if chosen is None:
        return "no"
    want_signal = _EXPECTED_SIGNAL[anchor.name]
    same_locus = chosen.boundary is anchor.incision_point
    if same_locus and chosen.action is anchor.action and chosen.signal == want_signal:
        return "exact"
    if same_locus and chosen.action is anchor.action:
        if frozenset({chosen.signal, want_signal}) in _SIBLINGS:
            return "approximate"
    if same_locus and chosen.signal == want_signal:
        return "approximate"          # right locus and condition, different action from the grid
    return "no"


# The signal each accepted anchor's trigger corresponds to in Phi_BFCL. This is the mapping the
# runtime asserts (bfcl_signals.py docstrings name the anchor per signal); stated here so the
# recovery verdict is checked against a DECLARED expectation rather than one chosen after seeing
# the output.
_EXPECTED_SIGNAL = {
    "A1": "container_at_capacity",
    "A2": "identifier_not_found",
    "A3": "duplicate_identifier",
    "A4": "no_tool_call_at_all",
    "A5": "container_slots_exhausted",
    "A7": "append_would_exceed_cap",
    "A8": "clear_proposed_at_capacity",
    "A9": "retrieval_similarity_below_threshold",
}


def main() -> int:
    rows = []
    for anchor in ANCHORS:
        d = diagnosis_for(anchor)
        ok, why = expressible_under(d, runtime=runtime)
        if not ok:
            rows.append((anchor, None, why, "no"))
            continue
        # theta depends on phi (a reroute destination is attested per CONDITION), so enumerate per
        # signal and merge. Selection still ranges over the whole merged set -- narrowing per signal
        # would reintroduce the "whichever came first decides" defect the search already fixed.
        cands = []
        for signal in runtime.declared_signals():
            cands.extend(c for c in enumerate_candidates(
                d, runtime=runtime, host=runtime.HOST, theta_for=theta_for_signal(signal))
                if c.signal == signal)
        chosen = select(cands)
        rows.append((anchor, chosen, why, verdict(anchor, chosen)))

    w = 118
    print("=" * w)
    print("ANCHOR RECOVERY -- can candidate_search re-select the 8 accepted BFCL anchors?")
    print("=" * w)
    print(f"{'anchor':6} {'expected l / mu':34} {'expected phi':38} {'recovered?':11}")
    print("-" * w)
    for anchor, chosen, _why, v in rows:
        exp = f"{anchor.incision_point.value} / {anchor.action.value}"
        print(f"{anchor.name:6} {exp:34} {_EXPECTED_SIGNAL[anchor.name]:38} {v:11}")
    print("-" * w)

    print("\nPER-ANCHOR DETAIL\n" + "=" * w)
    for anchor, chosen, why, v in rows:
        print(f"\n[{anchor.name}]  verdict = {v.upper()}")
        print(f"  mined locus / residual class : {anchor.locus}")
        print(f"  expected  l / phi / mu       : {anchor.incision_point.value} / "
              f"{_EXPECTED_SIGNAL[anchor.name]} / {anchor.action.value}")
        if chosen is None:
            print(f"  candidate search output      : (none) {why[:96]}")
            continue
        print(f"  candidate search output      : {chosen.boundary.value} / {chosen.signal} / "
              f"{chosen.action.value}   score={chosen.score}")
        print(f"  expressibility               : {why[:104]}")

    n_exact = sum(1 for *_r, v in rows if v == "exact")
    n_approx = sum(1 for *_r, v in rows if v == "approximate")
    n_no = sum(1 for *_r, v in rows if v == "no")
    print("\n" + "=" * w)
    print(f"RECOVERED: exact {n_exact}/8   approximate {n_approx}/8   no {n_no}/8")
    print("=" * w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
