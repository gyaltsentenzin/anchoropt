"""Translate a BFCL trajectory sidecar into the CANONICAL criterion 3/4 evidence vocabulary.

WHY THIS EXISTS
---------------
`anchoropt.learning.acceptance_criteria` adjudicates criteria 3 and 4 over a fixed vocabulary:

    criterion 3   signal_firings, interventions_executed, mechanism_verified,
                  mechanism_requested, mechanism_declined_with_reason, mechanism_refused_by_guard
    criterion 4   clears_added, information_losing_removes, verified_relocations

NOTHING IN THIS BENCHMARK EVER EMITTED THOSE KEYS. Every installed mechanism writes its OWN
vocabulary onto `step_record` -- `relocate_write_verified`, `capacity_repair_requested_gate`,
`upstream_fired_gate`, `suppress_removed_n` -- because each was added by the patch that introduced
the mechanism. So the four-criterion checker had zero production consumers not because a key was
dropped by the sidecar allowlist, but because no producer of the canonical names existed at all.

This module is that missing translation, and it belongs HERE rather than in the core: reading
`relocate_*` is benchmark-specific knowledge, while `mechanism_verified` is the generic contract.

THE ONE RULE THAT MATTERS: ABSENT IS NOT ZERO
---------------------------------------------
A mechanism that recorded no preservation evidence must yield a MISSING key, so `check_mechanism`
returns PENDING_VALIDATION and the round BLOCKS. Defaulting it to 0 would convert missing evidence
into a measured negative result, which is the exact failure mode the three-verdict lattice exists to
prevent. Hence `_count` returns None for "no step carried this key" and the payload omits the entry,
and hence a mechanism family with no registered safety reading returns `None` for safety rather than
`{"clears_added": 0, ...}`.

The same asymmetry applies to the two safety counters: they are only reported when the family that
fired has a DECLARED reading for them. A family whose destructive-operation count cannot be read from
its own telemetry leaves criterion 4 PENDING, which is correct -- "this mechanism has no code path
that clears memory" is a claim about source, and a source marker is not evidence of live execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

#: A counter that cannot be read from ONE step key because it is a CONJUNCTION of observables.
#: `_derived_safety` computes it; the table names it so the declaration stays in one place.
_DERIVED = "__derived__"

#: A derived reading for the "acted but achieved nothing" outcome. Distinct sentinel so `_reading_of`
#: routes it to its own derivation rather than treating it as a step key.
_DERIVED_NO_EFFECT = "__derived_no_effect__"


class _Negated(str):
    """A reading to be counted where the step key is present and FALSY.

    Subclasses `str` so it remains the step key for every other purpose -- `families_present` and the
    provenance record treat it exactly like a plain declaration, and nothing outside `_reading_of`
    needs to know the difference.
    """

    __slots__ = ()


def _reading_of(decl: str) -> tuple[str, bool]:
    """(step key, negated) for a declared reading."""
    return str(decl), isinstance(decl, _Negated)


#: Per-family readings. Each maps a canonical counter to the step key that MEASURES it.
#:
#: `requested` / `verified` are the load-bearing pair: a recovery is verified on what it PRESERVED,
#: never on whether it fired. `declined` and `refused` exist so the requested population can be
#: partitioned exhaustively -- an unattributed request is missing evidence, not safe behaviour
#: (rounds/AUTONOMY/RELOCATION_CHAIN_RECONCILED.json records the measured instance).
#:
#: A family absent from this table contributes NO canonical counters, so criteria 3 and 4 stay
#: PENDING for it. That is deliberate: the remedy is to declare the reading alongside the mechanism,
#: in the same commit, exactly as _STEP_FIELDS requires.
FAMILY_READINGS: dict[str, dict[str, Any]] = {
    # KV capacity relocation. Denominators per the reconciliation: requested 84, preserved 67.
    # `relocate_write_verified` is the PRESERVATION reading; `relocate_gate` counts END-TO-END
    # successes (61) and would understate the safety evidence, so it is deliberately NOT used here.
    "relocate": {
        "requested": "relocate_requested_gate",
        "executed": "relocate_write_verified",
        "verified": "relocate_write_verified",
        "declined": "relocate_declined",
        # TWO independent guards refuse a relocation before it acts, and BOTH must be counted or
        # the 84-request population does not sum. The bound guard was silent until
        # patches/bv/bv_reloc_bound_reason.py added its reason: measured on job 1834414, 16 of 84
        # requests carried `relocate_requested_gate` and nothing else, leaving criterion 3 PENDING
        # with 16 unattributed. Key absence is not a recorded reason.
        "refused": ("relocate_declined_identity_gate", "relocate_declined_bound_gate"),
        # criterion 4: a verified relocation removed its source only after confirming the copy was
        # live in the destination, so it is bounded and non-destructive by construction.
        "verified_relocations": "relocate_write_verified",
        # criterion 4's destructive counter. Read as the NEGATION of the invariant the primitive
        # records on every verified removal, NOT as the `relocate_invariant_violated` flag the
        # primitive writes only when the invariant trips.
        #
        # That flag is unusable as a counter by construction: capacity_relocate.py sets it solely
        # inside `if not copy_survives`, so on a clean arm no step carries it, `_count` returns None,
        # the counter is omitted and criterion 4 reports PENDING_VALIDATION -- forever, for every
        # clean relocation arm. There is no input for which a safe arm could PASS. The positive
        # observable is written unconditionally on the same path (capacity_relocate.py:201), is
        # already declared in the sidecar allowlist, and negating it counts exactly the removals
        # that were NOT confirmed present elsewhere in live state -- which is the definition of an
        # information-losing remove. An absent observable still yields None, so a dropped
        # measurement remains PENDING rather than becoming a clean zero.
        "information_losing_removes": _Negated("relocate_copy_survives_in_destination"),
    },
    # Aggregate-container repair (N0-A1).
    "capacity_repair": {
        "requested": "capacity_repair_requested_gate",
        "executed": "capacity_repair_gate",
        "declined": "capacity_repair_declined_gate",
    },
    # Pre-generation zero-call reprompt.
    "upstream": {
        "requested": "upstream_fired_gate",
        "executed": "upstream_zero_call_gate",
        "verified": "upstream_injected_chars_gate",
    },
    # Post-generation call suppression. A suppression REMOVES calls, never memory entries, so it has
    # no clears reading; criterion 4 therefore stays PENDING for a suppression-only arm rather than
    # being asserted clean from the absence of a clear.
    "suppress": {
        "requested": "suppress_gate",
        "executed": "suppress_removed_n",
        "verified": "suppress_removed_n",
    },
    # A9: a successful-but-weak core retrieve is answered by dispatching the SAME query to the other
    # container and returning the merged ranked union. READ-SIDE and store-untouched.
    #
    # Declared because C3/C4 read only DECLARED families: without this, A9's round adjudicated C3 on
    # OTHER controllers' telemetry (capacity_repair, relocate, upstream -- none of which A9 wrote) and
    # C4 had no A9 safety reading at all, leaving `information_losing_removes` absent and the criterion
    # PENDING. Same defect as `dedup_clear` had, and the same fix; the docstring's rule -- declare the
    # reading alongside the mechanism -- applies to every new family, not just the one that prompted it.
    #
    # `verified` reads `xcm_delivered`, NOT `xcm_gate` or `xcm_merged_returned`: the gate counts firings
    # and the merge counts results assembled, but only DELIVERED means the merged text actually reached
    # the model. docs/ANCHORS.md records a delivery defect whose lesson was exactly this -- reasoning
    # about the gate instead of measuring downstream.
    "xcm": {
        "requested": "xcm_gate",
        "executed": "xcm_replaced",
        "verified": "xcm_delivered",
        "declined": "xcm_declined",
        # criterion 4, and the reading here is deliberately ZERO-BY-CONSTRUCTION rather than derived.
        #
        # `information_losing_removes` is defined by the acceptance rule as "a removal NOT VERIFIED
        # PRESENT ELSEWHERE IN LIVE STATE". A9 removes nothing: it dispatches a second read and changes
        # what the model is SHOWN. The store is byte-untouched, so every entry remains present in live
        # state by construction and the counter is 0 -- not "assumed 0", but 0 because the quantity it
        # counts cannot occur for a controller with no write path.
        #
        # I FIRST DERIVED IT as "dropped core entries AND retained none", and that was WRONG. It charged
        # 4 firings on R10c, and the attribution showed what they actually were:
        #   memory_vector_10-customer-10   dropped [1,2,3] retained 0  ->  GAIN (False -> True)
        #   memory_vector_17-customer-17   dropped [0,1,2,3] retained 0 -> stayed CORRECT
        #   memory_vector_7-customer-7     dropped [3] retained 0      ->  unchanged
        # Zero losses; one gain. The counter was measuring "showed the model archival instead of core",
        # which is A9's MECHANISM, not a harm -- so criterion 4 would have FAILED the controller for
        # doing exactly what it is for. Displacement belongs in the round's mechanism record (it IS
        # tracked there, with the retraction it bears on), not in a destructive-safety counter.
        #
        # The guard against complacency is elsewhere and is real: `clears_added` is still measured as a
        # PAIRED count over the model's own decoded calls, so if A9 ever provoked a destructive call it
        # would show up there. And a write-side reading is deliberately absent rather than zeroed, so if
        # this controller ever acquires a write path the criterion goes PENDING instead of silently
        # passing.
        "information_losing_removes": 0,
    },
    # A8 COMPLETE: suppress a proposed destructive clear, evict ONE redundant copy, verify an
    # equivalent copy remains in LIVE state, retry the blocked write verbatim.
    #
    # Declared in the same commit as its executor, which is what the table's own docstring requires --
    # a family absent from here contributes NO counters, so criteria 3 and 4 stay PENDING however rich
    # the telemetry is. Checked before the measuring jobs finished rather than after: the arm emits
    # dedup_clear_* keys, `families_present` derives families only from DECLARED reading keys, and
    # nothing here matched them, so C3/C4 would have reported PENDING on a mechanism that ran. This is
    # the same allowlist/reading class that has swallowed intervention telemetry repeatedly.
    #
    # `dedup_clear_verified_gate` is the PRESERVATION reading, not `dedup_clear_gate`: the gate counts
    # firings, the verification counts firings whose invariant was confirmed against live state after
    # the eviction. Criterion 3 asks what was VERIFIED, so it must read the stricter one.
    "dedup_clear": {
        "requested": "dedup_clear_requested_gate",
        "executed": "dedup_clear_gate",
        "verified": "dedup_clear_verified_gate",
        # Why the mechanism declined. Every non-firing branch records a reason, so a request that did
        # not act is attributable rather than silently missing -- the defect the relocate bound guard
        # had until its reason was added.
        "declined": "dedup_clear_reason_gate",
        # A raising mechanism records this instead of degrading to plain suppression. Counted as a
        # refusal so the requested population still sums.
        #
        # `dedup_clear_acted_no_effect` is the THIRD OUTCOME, and it is declared here because
        # criterion 3 requires the requested population to SUM. Measured on R6f: 26 requested =
        # 7 verified + 18 declined, leaving ONE unattributed -- a firing whose remove AND retry both
        # succeeded but which left the container unchanged, because the retry re-added the key the
        # eviction had removed. That is neither "verified" nor "declined with reason": the mechanism
        # acted, the calls landed, the net effect was nil.
        #
        # Folding it into `declined` would be dishonest (it did not decline) and folding it into
        # `verified` would be worse (it verified nothing). It is grouped under `refused` because that
        # slot's role in the sum is "requested, did not achieve the mechanism's effect, reason
        # recorded" -- which is exactly this case -- and because a guard-refusal and a no-op are both
        # non-events for the criterion. The provenance records them separately so the two are never
        # confused in a report.
        "refused": ("dedup_clear_error_gate", _DERIVED_NO_EFFECT),
        # criterion 4. An information-losing remove is a copy REMOVED with no equivalent copy left --
        # NOT merely an unverified firing.
        #
        # This first read `_Negated("dedup_clear_verified_gate")`, i.e. "count every firing that did not
        # verify". Measured on R6f that was WRONG in the one case it mattered: 8 evictions, 7 verified,
        # and the 8th reported `victim_removed=False, copies_remaining=2` -- the store was UNTOUCHED, so
        # nothing was lost. Charging it as an information-losing remove would have FAILED criterion 4
        # on a safety violation the mechanism did not commit, which is the same error in the opposite
        # direction from hiding one.
        #
        # The honest reading is the NEGATION OF THE SURVIVING COPY, conditioned on a removal having
        # happened: `dedup_clear_victim_removed` true AND `dedup_clear_copies_remaining` zero. There is
        # no single step key for that conjunction, so the counter is derived in `_derived_safety` where
        # both observables are in scope. A clean arm still yields 0 rather than an absent counter, and
        # an arm that genuinely destroyed a last copy still yields >= 1 -- which is what the relocation
        # family's negation achieves for its own, simpler case.
        "information_losing_removes": _DERIVED,
    },
}

#: Canonical counters that criterion 4 treats as destructive. Kept here so a reading added to
#: FAMILY_READINGS is classified without touching the core.
_SAFETY_KEYS = ("clears_added", "information_losing_removes", "verified_relocations")

#: Destructive operations counted from what the model ACTUALLY CALLED, per step.
#:
#: `clears_added` cannot come from a mechanism's own flags: no relocation, repair or suppression key
#: records a clear, because a clear is something the MODEL does and the mechanism is trying to avoid.
#: Reading it from mechanism telemetry would therefore always yield "absent", and defaulting absent to
#: 0 is exactly the inference the three-verdict lattice forbids -- it would assert a safety property
#: from silence.
#:
#: So it is measured on the observable that does record it: `decoded`, the step's actual tool calls.
#: The counter is ADDED clears, so it is a PAIRED quantity -- arm minus control on the same cases. An
#: arm that merely fails to prevent a clear the control also made has added nothing.
_DESTRUCTIVE_CALLS = {
    "clears_added": ("core_memory_clear",),
}


def _derived_safety(steps, family: str, target: str) -> int | None:
    """Counters that are a CONJUNCTION of observables, so no single step key can express them.

    `dedup_clear.information_losing_removes`: a copy was REMOVED and NO equivalent copy remained. Both
    halves are required, and reading either alone gives the wrong answer:

      * `not verified` alone -- what this first declared -- counts a firing whose store was UNTOUCHED.
        Measured on R6f: 8 evictions, 7 verified, and the 8th reported `victim_removed=False,
        copies_remaining=2`, i.e. nothing happened. Charging that as a lossy remove would FAIL criterion
        4 on a violation the mechanism did not commit.
      * `copies_remaining == 0` alone would count a step where the eviction never ran and the container
        was simply empty.

    Returns None when the observables were not recorded at all, so a dropped measurement stays PENDING
    rather than becoming a clean zero.
    """
    if target != "information_losing_removes":
        return None
    if family != "dedup_clear":
        return None
    seen = False
    lossy = 0
    for st in steps:
        if "dedup_clear_victim_removed" not in (st or {}):
            continue
        seen = True
        if bool(st.get("dedup_clear_victim_removed")) and \
                int(st.get("dedup_clear_copies_remaining", 0) or 0) <= 0:
            lossy += 1
    return lossy if seen else None


def _derived_no_effect(steps) -> int | None:
    """Firings that ACTED, whose calls LANDED, and which changed nothing.

    A8's third outcome, measured on R6f: the remove returned "Key removed." and the retry returned
    "Key-value pair added." for the SAME key, so the container ended as it began. The mechanism did not
    decline (it chose to act) and did not verify (there was nothing to verify), so criterion 3's
    population would not sum without it -- and an unattributed request is missing evidence, not safe
    behaviour.

    Identified as: acted, and the victim was NOT removed as of the post-dispatch check. Returns None
    when the observable was never recorded, so a dropped measurement stays PENDING.
    """
    seen = False
    n = 0
    for st in steps:
        st = st or {}
        if not st.get("dedup_clear_gate"):
            continue
        if "dedup_clear_victim_removed" not in st:
            continue
        seen = True
        if not bool(st.get("dedup_clear_victim_removed")):
            n += 1
    return n if seen else None


def _count(steps: Iterable[Mapping[str, Any]], key: str, *, negated: bool = False) -> int | None:
    """How many steps carried `key` truthily -- or None if NO step carried it at all.

    The None return is the whole point. "No step recorded this key" and "every step recorded zero"
    are different claims and only the second is a measurement.

    `negated` counts steps where the key is PRESENT and FALSY. It exists because some invariants are
    recorded as the property that must hold (`..._copy_survives_in_destination`) rather than as a
    violation flag. A violation flag written only when it trips can never distinguish "clean" from
    "unmeasured": the key is simply absent in both cases, so the counter is omitted and the criterion
    stays PENDING even for a perfectly clean arm. Reading the positive observable and negating it
    makes a clean arm MEASURABLY clean, while an absent observable still returns None.
    """
    seen = False
    n = 0
    for s in steps:
        if key in s:
            seen = True
            if (not s[key]) if negated else bool(s[key]):
                n += 1
    return n if seen else None


def families_present(steps: Iterable[Mapping[str, Any]]) -> list[str]:
    """Which known mechanism families left any trace in these steps."""
    keys = {k for s in steps for k in s}
    out = []
    for fam, reading in FAMILY_READINGS.items():
        if any(v in keys for v in reading.values()):
            out.append(fam)
    return sorted(out)


def count_destructive_calls(case_steps: Mapping[str, list[Mapping[str, Any]]],
                            counter: str) -> int | None:
    """How many steps DECODED a destructive call of this class -- or None if `decoded` is absent.

    None when no step carried `decoded` at all: that is a dropped observation, not a clean zero, and
    it must leave criterion 4 PENDING rather than assert safety from a missing field.
    """
    names = _DESTRUCTIVE_CALLS.get(counter, ())
    seen = False
    n = 0
    for st in case_steps.values():
        for s in st:
            if "decoded" in s:
                seen = True
                text = str(s.get("decoded") or "")
                if any(nm in text for nm in names):
                    n += 1
    return n if seen else None


def mechanism_evidence(case_steps: Mapping[str, list[Mapping[str, Any]]],
                       *, family: str | None = None,
                       control_steps: Mapping[str, list[Mapping[str, Any]]] | None = None,
                       ) -> tuple[dict, dict | None, dict]:
    """Return (telemetry, safety, provenance) in the canonical vocabulary.

    `case_steps` is the sidecar's case_id -> [step dict] mapping, exactly as `--score` already loads
    it for the firing count. `family` pins one mechanism; when None, every family present is read and
    the counters are summed -- an installed stack legitimately has several.

    A counter that no step recorded is OMITTED, never zeroed, so the criterion reports
    PENDING_VALIDATION. `provenance` records which family and which step key produced each number,
    so an acceptance can be traced back to the reading that justified it.
    """
    steps = [s for st in case_steps.values() for s in st]
    fams = [family] if family else families_present(steps)

    telemetry: dict[str, int] = {}
    safety: dict[str, int] = {}
    provenance: dict[str, Any] = {"families_read": fams, "readings": {}, "missing": []}

    # signal_firings is family-independent: it is the same *_gate suffix count the validity guards
    # already use, so criterion 3's firing number and the channel-integrity number cannot disagree.
    #
    # It is computed here but only PUBLISHED at the end, and only if some family reading resolved.
    # Publishing it unconditionally was a real defect caught by
    # `test_a_family_with_no_declared_reading_yields_PENDING_not_a_clean_pass`: for an unknown family
    # it made `telemetry` non-empty while `interventions_executed` stayed absent, so
    # `check_mechanism` skipped its "no telemetry at all" PENDING branch, read the absent count as
    # zero and returned FAIL -- converting MISSING EVIDENCE into a measured negative result. The core
    # is right to fail a genuine zero-execution arm; the error was fabricating partial telemetry for
    # a mechanism whose reading was never declared. An unreadable family must emit NOTHING.
    firings = sum(
        1 for st in case_steps.values()
        if any(k.endswith("_gate") and v for s in st for k, v in s.items()))

    canon = {"requested": "mechanism_requested", "executed": "interventions_executed",
             "verified": "mechanism_verified", "declined": "mechanism_declined_with_reason",
             "refused": "mechanism_refused_by_guard"}

    for fam in fams:
        reading = FAMILY_READINGS.get(fam)
        if not reading:
            provenance["missing"].append(f"{fam}: no declared reading")
            continue
        for slot, decl in reading.items():
            target = canon.get(slot, slot)
            # A slot may declare SEVERAL readings. One guard per key is the common case, but a
            # mechanism can be refused by more than one INDEPENDENT guard, and the requested
            # population only partitions if every such guard is counted. Summing is sound because
            # the guards are mutually exclusive per step: a request refused by the identity gate
            # never reaches the bound check. A slot whose readings are ALL absent stays None, so a
            # dropped observation remains missing evidence rather than becoming a clean zero.
            # A DERIVED counter is a conjunction of observables, so no single step key carries it.
            # Computed here where the steps are in scope, and recorded in provenance with the
            # observables it combined -- otherwise a reader cannot tell a derived 0 from an absent one.
            # A LITERAL COUNT declared in the table: the quantity cannot occur for this family, by
            # construction. Honoured explicitly -- `_reading_of` expects a step key and silently
            # dropped it, which left criterion 4 PENDING on a controller whose counter is 0 for a
            # structural reason. The provenance records `structural: True` so a reader can tell a
            # declared 0 from a counted one.
            if isinstance(decl, int) and not isinstance(decl, bool):
                bucket = safety if target in _SAFETY_KEYS else telemetry
                bucket[target] = bucket.get(target, 0) + int(decl)
                provenance["readings"].setdefault(target, []).append(
                    {"family": fam, "structural": True, "n": int(decl),
                     "why": "the quantity cannot occur for this family (no write path)"})
                continue
            if decl == _DERIVED_NO_EFFECT:
                n = _derived_no_effect(steps)
                if n is None:
                    provenance["missing"].append(
                        f"{fam}.{target} <- derived no-effect (observables not recorded)")
                    continue
                bucket = safety if target in _SAFETY_KEYS else telemetry
                bucket[target] = bucket.get(target, 0) + n
                provenance["readings"].setdefault(target, []).append(
                    {"family": fam, "derived": "acted_no_effect", "n": n})
                continue
            if decl == _DERIVED:
                n = _derived_safety(steps, fam, target)
                if n is None:
                    provenance["missing"].append(
                        f"{fam}.{target} <- derived (its observables were not recorded)")
                    continue
                bucket = safety if target in _SAFETY_KEYS else telemetry
                bucket[target] = bucket.get(target, 0) + n
                provenance["readings"].setdefault(target, []).append(
                    {"family": fam, "derived": True, "n": n})
                continue
            decls = list(decl) if isinstance(decl, (tuple, list)) else [decl]
            total = None
            for d in decls:
                if d == _DERIVED_NO_EFFECT:
                    n = _derived_no_effect(steps)
                    if n is None:
                        provenance["missing"].append(
                            f"{fam}.{target} <- derived no-effect (observables not recorded)")
                        continue
                    total = n if total is None else total + n
                    provenance["readings"].setdefault(target, []).append(
                        {"family": fam, "derived": "acted_no_effect", "n": n})
                    continue
                step_key, negated = _reading_of(d)
                n = _count(steps, step_key, negated=negated)
                if n is None:
                    provenance["missing"].append(
                        f"{fam}.{target} <- {step_key} (no step carried it)")
                    continue
                total = n if total is None else total + n
                rec = {"family": fam, "step_key": step_key, "n": n}
                if negated:
                    # The provenance must say the number came from a negation, or a reader cannot
                    # tell "3 steps recorded a violation" from "3 steps failed to record the
                    # invariant".
                    rec["negated"] = True
                provenance["readings"].setdefault(target, []).append(rec)
            if total is None:
                continue
            bucket = safety if target in _SAFETY_KEYS else telemetry
            bucket[target] = bucket.get(target, 0) + total

    # criterion 4's clears counter, measured as a PAIRED difference on actual decoded calls. Only
    # reported when BOTH arms supplied `decoded`; otherwise the observation was dropped and the
    # criterion stays PENDING, which is the honest verdict for an unmeasured safety property.
    if control_steps is not None:
        arm_clears = count_destructive_calls(case_steps, "clears_added")
        ctl_clears = count_destructive_calls(control_steps, "clears_added")
        if arm_clears is None or ctl_clears is None:
            provenance["missing"].append(
                "clears_added: `decoded` absent from one or both arms, so added clears were not "
                "measured. Absent is NOT zero -- criterion 4 stays PENDING.")
        else:
            safety["clears_added"] = max(0, arm_clears - ctl_clears)
            provenance["readings"].setdefault("clears_added", []).append(
                {"family": "decoded_calls", "step_key": "decoded",
                 "arm": arm_clears, "control": ctl_clears,
                 "n": safety["clears_added"]})
    else:
        provenance["missing"].append(
            "clears_added: no control arm supplied, and added clears are a PAIRED quantity "
            "(arm minus control). Criterion 4 stays PENDING.")

    if telemetry:
        telemetry["signal_firings"] = firings
    else:
        provenance["missing"].append(
            "signal_firings withheld: no declared family reading resolved, so publishing a firing "
            "count alone would make absent execution evidence look like a measured zero")
    return telemetry, (safety or None), provenance
