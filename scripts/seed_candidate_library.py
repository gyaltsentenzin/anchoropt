"""Recover what this project already discovered, before any round repeats it.

WHY THIS RUNS ONCE, AND WHAT IT IS NOT
--------------------------------------
The persistence audit found the accumulated discoveries were never lost -- they were never read back.
This script does the reading back: it seeds `rounds/GOLDEN/candidates.json` from records that already
exist on disk, so a round can ask "what is installed?" and "what is still pending?" instead of starting
from H0 by default.

It is NOT a proposer input. Nothing here is fed to the teacher or the search: the library exists for
persistence, deduplication, regression protection and retrospective coverage assessment. Handing a
proposer a list of known-good controllers and their gains would make any later "rediscovery"
unfalsifiable, which is precisely the claim this workstream is trying to establish honestly.

WHERE EACH RECORD COMES FROM
----------------------------
* ACCEPTED           <- rounds/GOLDEN/registry.json, the golden registry. Already carries spec,
                        provenance, firing telemetry and per-case outcomes, so these convert directly
                        and keep their measurements.
* PENDING_VALIDATION <- rounds/AUTORUN/r1/arm_manifest.json. Round 1 emitted six measurable arms. Two
                        were evaluated by a protocol that eliminated the prereq phase they act in
                        (NO_OPPORTUNITY -- not a refutation), and two were never evaluated at all. All
                        of them are AVAILABLE FOR VALIDATION and none is a measured loser.

A round-1 arm is deliberately recorded WITHOUT a measurement even where a number exists, because the
number came from an instrument that could not bear on it. Storing `net` beside `valid=False` is how a
broken measurement gets banked as evidence; the record carries the reason instead.

Usage:  python scripts/seed_candidate_library.py [--out rounds/GOLDEN/candidates.json] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from anchoropt.learning.candidate_library import (            # noqa: E402
    ACCEPTED, DISCOVERED, PENDING_VALIDATION, CandidateLibrary, CandidateRecord,
    EvaluationContext, Measurement)
from anchoropt.learning.golden_registry import ControllerIdentity, GoldenRegistry  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "rounds" / "GOLDEN" / "registry.json"
R1_MANIFEST = ROOT / "rounds" / "AUTORUN" / "r1" / "arm_manifest.json"

#: Why each round-1 arm is PENDING rather than measured or rejected. Quoted from the round's own
#: records, and kept as text in the library so a later round does not have to rediscover the reason.
R1_STATUS = {
    "post_execution/container_at_capacity/transform:relocate_entry_for_kv": (
        "EVALUATED BUT UNINTERPRETABLE. Job 1834414 fired 84 times in prereq and 0 times in query "
        "over 105 scored units; R1 held the store fixed and compared query accuracy, which "
        "eliminates the only phase this controller acts in. NO_OPPORTUNITY, not a refutation. "
        "R3 (rounds/AUTONOMY/R3_PREREQ_SCORED_STATE.json) read 258 facts rescued / 0 lost off live "
        "constructed state -- an effect on state, with downstream query accuracy still unmeasured."),
    "post_generation_pre_exec/proposal_already_refused/suppress:cancel_proposed": (
        "EVALUATED BUT UNINTERPRETABLE. Job 1834415 fired 62 times in prereq, 0 in query, under the "
        "same query-time protocol. NO_OPPORTUNITY."),
    "post_generation_pre_exec/redundant_proposed_write/suppress:cancel_proposed": (
        "NEVER EVALUATED -- omitted from --score, never reported as zero. Fires on 201/590 projected "
        "states. Being a SUPPRESSION arm, store_diag.json is a valid metric for it (blind only to "
        "post-execution reroute), so it may be scorable on constructed state with no new GPU work."),
    "post_generation_pre_exec/clear_proposed_at_capacity/suppress:cancel_proposed": (
        "NEVER EVALUATED -- omitted from --score, never reported as zero. Fires on 83/590 projected "
        "states. Suppression, so store_diag.json is a valid metric for it."),
    "post_execution/container_at_capacity/substitute:archival_memory_add+replace_original": (
        "EMITTED, NEVER EVALUATED. Carries the capability_id defect the pre-registered policy names: "
        "a `substitute` arm holding the RELOCATION executor's capability_id. Under the identity gate "
        "the runner refuses it, so an automatic install would have measured the control while "
        "reporting an arm. Recorded as DISCOVERED; needs capability_id derived per arm before it can "
        "be measured at all."),
    "post_execution/container_at_capacity/substitute:archival_memory_add+retry_after": (
        "EMITTED, NEVER EVALUATED. Same capability_id defect as its sibling variant."),
}

#: Arms whose spec cannot be executed as recorded. DISCOVERED, not PENDING: pending means "awaiting a
#: valid evaluation", and these are awaiting a valid SPEC, which is a different remedy.
R1_UNEXECUTABLE = {
    "post_execution/container_at_capacity/substitute:archival_memory_add+replace_original",
    "post_execution/container_at_capacity/substitute:archival_memory_add+retry_after",
}


def _cell_of(label: str, spec: dict) -> str:
    """The (domain x backend) cell an arm belongs to, from its own spec rather than guessed."""
    for token in ("kv", "vector", "rec_sum"):
        if token in str(spec.get("variant") or "") or token in label:
            return token
    return ""


def from_registry(reg: GoldenRegistry) -> list[CandidateRecord]:
    """Convert accepted controllers, preserving their measurements and provenance verbatim."""
    out: list[CandidateRecord] = []
    for e in reg.all():
        cell = (e.provenance.split or "").split()[0] if e.provenance.split else ""
        out.append(CandidateRecord(
            name=e.name,
            identity=e.identity,
            spec=dict(e.spec),
            state=ACCEPTED,
            context=EvaluationContext(
                # An entry accepted before the stack existed was measured against bare H0 plus
                # nothing else -- recorded as the empty stack, which fingerprints to "H0". Claiming
                # it was measured against the current stack would be a fabrication.
                incumbent_stack=(),
                incumbent_token=e.provenance.notes or "",
                cell=cell, split=e.provenance.split, model=e.provenance.model,
                acting_phase=e.identity.phase, scored_phase="query"),
            measurement=Measurement(
                arm_correct=e.outcome.arm_correct, control_correct=e.outcome.control_correct,
                n_scored=e.outcome.n_scored, gains=tuple(e.outcome.gains),
                losses=tuple(e.outcome.losses), valid=True, firings=e.telemetry.fired,
                detail=f"requested={e.telemetry.requested} fired={e.telemetry.fired} "
                       f"acted={e.telemetry.acted}"),
            provenance={"runtime": e.provenance.runtime, "patches": list(e.provenance.patches),
                        "config_fingerprint": e.provenance.config_fingerprint,
                        "origin": e.origin, "source": "rounds/GOLDEN/registry.json"},
            round_id="pre-lifecycle (golden registry)",
            notes=tuple(e.caveats),
            version=e.version))
    return out


def from_r1_manifest(path: pathlib.Path) -> list[CandidateRecord]:
    """Recover round 1's emitted arms as pending/discovered -- never as losers."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    out: list[CandidateRecord] = []
    for arm in raw.get("arms", ()):
        label = arm["arm_label"]
        spec = {k: arm.get(k) for k in ("signal", "boundary", "action", "operator", "variant", "eta")}
        spec["locus"] = arm.get("boundary")
        cell = _cell_of(label, arm)
        # capability_id is part of the IDENTITY. Round 1 did not emit one per arm -- that is the
        # recorded defect -- so it is derived from the variant, which is what the arm actually names.
        # Never left empty: an empty capability_id would let a two-executor cell run a different
        # mechanism under this candidate's name, and `validate()` refuses it for exactly that reason.
        cap = str(arm.get("variant") or arm.get("operator") or "unspecified")
        out.append(CandidateRecord(
            name="r1_" + label.replace("/", "__").replace(":", "_"),
            identity=ControllerIdentity(
                boundary=str(arm.get("boundary") or ""), signal=str(arm.get("signal") or ""),
                action=str(arm.get("action") or ""), operator=str(arm.get("operator") or ""),
                capability_id=cap,
                # Round 1's four specs all declared phase=prereq; the two substitute variants are
                # post_execution reroutes in the query phase. Taken from the round's record.
                phase="prereq"),
            spec=spec,
            state=DISCOVERED if label in R1_UNEXECUTABLE else PENDING_VALIDATION,
            context=EvaluationContext(
                incumbent_stack=(), incumbent_token=raw.get("incumbent_token", ""),
                cell=cell or "kv", split="kv train", model="granite-4.1-8b",
                acting_phase="prereq", scored_phase=""),
            # NO measurement, deliberately, even where a number exists: it came from an instrument
            # that could not bear on the candidate.
            measurement=None,
            provenance={"fires_on_states": arm.get("fires_on_states"),
                        "total_states": arm.get("total_states"),
                        "source": "rounds/AUTORUN/r1/arm_manifest.json"},
            round_id="AUTORUN/r1",
            notes=(R1_STATUS.get(label, "emitted by round 1; disposition not recorded"),)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "rounds" / "GOLDEN" / "candidates.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lib = CandidateLibrary.load(a.out)          # additive: never discards what is already recorded
    before = len(lib)
    if REGISTRY.exists():
        for r in from_registry(GoldenRegistry.load(REGISTRY)):
            if r.name not in lib:
                lib.upsert(r)
    for r in from_r1_manifest(R1_MANIFEST):
        if r.name not in lib:
            lib.upsert(r)

    print(f"library: {before} -> {len(lib)} records")
    for state in (ACCEPTED, PENDING_VALIDATION, DISCOVERED):
        rs = lib.in_state(state)
        print(f"\n{state}  ({len(rs)})")
        for r in rs:
            m = r.measurement
            net = f"net {m.net:+d} of {m.n_scored}" if m else "no valid measurement"
            print(f"  - {r.name}")
            print(f"      {r.identity.key()}")
            print(f"      cell={r.context.cell or '?'}  {net}")
    stack = lib.installed_stack()
    print(f"\nINSTALLED STACK ({len(stack)}) fingerprint={__import__('anchoropt.learning.candidate_library', fromlist=['x']).stack_fingerprint(stack)}")
    for k in stack:
        print(f"  {k}")
    if a.dry_run:
        print("\n(dry run -- nothing written)")
        return 0
    p = lib.save(a.out)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
