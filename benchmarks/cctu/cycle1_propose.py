#!/usr/bin/env python3
"""CYCLE 1, PROPOSE PHASE -- evidence in, candidate controllers out. The SEARCH is not here.

    python benchmarks/cctu/cycle1_propose.py --run results/granite/train_baseline \\
                                             --out rounds/CCTU_CYCLE1

No model, no GPU, no network. Reads a control run's artifacts, reconstructs the state trajectory of
every failing episode, hands the residual to the core, and writes whatever the core makes realizable
as `anchoropt.controller.v1` specs.

WHAT THIS SCRIPT MUST NOT DO, AND THE TEST THAT ENFORCES IT
-----------------------------------------------------------
It must not choose WHERE to intervene, WHAT condition decides, or HOW to act. All three belong to
`anchoropt.learning.structured_search.optimize_residual`. A loop that decides its own answer is not
evidence that the algorithm decides it, which is why `scripts/self_evolve_cycle2.py` is held to the
same rule by `tests/test_cycle2_thinness.py` and why this file is held to it by
`tests/test_cctu_cycle1_thinness.py`.

DELIBERATELY NOT IMPORTED: the incision-point enum, the action enum, AnchorPolicyOpt,
SearchSpaceProposal, build_arms, synthesize, expand_and_resume_predicates. This script cannot name a
coordinate, so it cannot pick one. The only coordinate-shaped strings it ever handles are read OFF a
controller the core already built, on the way to disk.

WHERE THE EVIDENCE COMES FROM, AND WHY IT IS NOT AUTHORED
--------------------------------------------------------
A `ResidualDiagnosis` carries four fields, and it matters which of them this script fills:

    case_id                 the failing episode                            -- observed
    mechanism               what the agent did                             -- observed
    evidence                the boundary evidence it is inferred from      -- observed
    consequential_decision  which decision must change for it not to recur -- DERIVED FROM THE TRACE
    proposed_behavior_change  the behavioural content                      -- LEFT EMPTY

The last two are the ones to be careful about. `consequential_decision` is derived by walking each
failing episode BACKWARD to the first turn whose proposal the validator refused -- the earliest
decision whose alternative would have made the episode reachable. That is trajectory-assigned, which
is the standard `docs/GENERALIZABILITY.md` sets: "A proxy may propose a locus; only the trajectory can
assign one." It is not a lexical guess about what went wrong.

`proposed_behavior_change` is left EMPTY on purpose. It is the field a diagnosis provider uses to say
what the agent should do instead, and filling it here would be this script proposing the intervention
under the search's name. `anchoropt/runtime.py` records the same concern one level up: a provider whose
own search space is prompt-heavy will propose prompt edits, and letting that reach the core biases
selection toward the provider's habits rather than the evidence.

RANKED BY LINKED LOSS, NOT BY EVENT VOLUME
------------------------------------------
The unit is DISTINCT FAILING EPISODES, not violation events, and the two disagree. On this corpus a
first pass by hand ranked the residual by event count and read `max_calls_per_tool` at 512 events --
but 512 events spread over far fewer episodes, and an anchor is judged per episode. `COLLABORATORS.md`
is blunt about the cost: "on our corpus the rarest locus (31 events) did the most damage per occurrence
... A frequency ranking buries your best target." Both orderings are printed so the difference is
visible rather than taken on trust.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cctu_adapter as runtime  # noqa: E402
import cctu_middleware as mwmod  # noqa: E402
import cctu_state as state_mod  # noqa: E402
from anchoropt.learning.candidate_search import expressible_under  # noqa: E402
from anchoropt.learning.learned_signal import firing_vector  # noqa: E402
from anchoropt.learning.residual_problem import build_residual_problems  # noqa: E402
from anchoropt.learning.structured_search import optimize_residual  # noqa: E402
from anchoropt.runtime import diagnoses_from  # noqa: E402

W = 96
PROVIDER = "cctu_trace_replay"

# The round tag that lands in every controller_id, and therefore in every run directory name. Set
# from `--tag` before any candidate is built; `c1` is the default so the rounds already on disk keep
# the names their sweep.json files record.
ROUND_TAG = "c1"
_TAG_OK = re.compile(r"^[a-z0-9]{1,8}$")


def assert_no_oracle_leak() -> None:
    """Nothing that carries a known answer may be in the room while candidates are proposed.

    The shipped probe tables are empty by design and must stay so during a discovery run: a probe
    value is a measured number from an accepted anchor, and exporting one into a proposal turns a
    discovery into a lookup. The other benchmark's smoke run caught exactly this, with the fixture
    reaching the runtime through an import nobody read as an import of the answer key.
    """
    import cctu_signals as sig
    if dict(sig.SIGNAL_PROBE_PARAMS):
        raise SystemExit(f"ORACLE LEAK: probe parameters are live: {dict(sig.SIGNAL_PROBE_PARAMS)}")
    if dict(sig.SIGNAL_ALIASES) and any(a.startswith("__") for v in sig.SIGNAL_ALIASES.values()
                                        for a in v):
        raise SystemExit("ORACLE LEAK: alias table carries injected calibration data")
    leaked = sorted(m for m in sys.modules if "replay_anchors" in m or "replay_exprs" in m)
    if leaked:
        raise SystemExit(f"ORACLE LEAK: anchor fixtures imported ({leaked})")


# ------------------------------------------------------------------------------------------------
# 1. evidence: replay each failing episode and reconstruct its state trajectory
# ------------------------------------------------------------------------------------------------
def replay_episode(sample: dict, messages: list) -> tuple[list[dict], list[dict]]:
    """A transcript -> (events, states), with the live constraint state at each boundary.

    The checker is REPLAYED, not simulated: a fresh `DialogueConstraintChecker` is advanced through
    the turns exactly as the live loop advances it, and snapshotted before each advance. So the
    budget-pressure fields carry what the agent's own run carried, which is the difference between
    searching over the real state and searching over a reconstruction of it.

    Two events per turn, because one derives a single boundary and a residual with one boundary can
    never be observed to move earlier.
    """
    from utils.constraint_checker import DialogueConstraintChecker

    checker = DialogueConstraintChecker(sample=sample, max_turns=20, validators_dir=str(DATA_DIR))
    events: list[dict] = []
    states: list[dict] = []
    seen_calls: list[tuple[str, str]] = []
    last_violation: str | None = None
    streak = 0
    turn = 0

    for row in runtime.raw_rows_from_messages(messages):
        event = runtime.normalize_event(row)
        snapshot = state_mod.snapshot_of(checker)
        carried = state_mod.carried_state(
            snapshot, event, round_index=turn, seen_calls=tuple(seen_calls),
            last_violation_class=last_violation, consecutive_violation_turns=streak)
        events.append(event)
        states.append(runtime.observable_state(event, carried))

        if runtime.boundary_key(event) == "post_execution":
            calls = event["tool_calls_raw"]
            # Advance the real checker the way the live loop does. Wrapped because a hallucinated tool
            # name makes upstream's own handler raise, and a replay must not die where the live run
            # merely dropped the episode.
            try:
                checker.get_feedback_if(is_final=not calls, content=event["content"],
                                        tool_calls=calls)
            except Exception as exc:                                        # noqa: BLE001
                REPLAY_ERRORS.append(f"{sample['id']} turn {turn}: {type(exc).__name__}: {exc}")
            seen_calls.extend(state_mod.call_keys(calls))
            if event["violation_class"] is not None:
                last_violation, streak = str(event["violation_class"]), streak + 1
            else:
                streak = 0
            turn += 1
    return events, states


REPLAY_ERRORS: list[str] = []
DATA_DIR = HERE / "data"


def attribute(events: list[dict]) -> tuple[str, str, str] | None:
    """Walk an episode's boundaries backward to the FIRST refused proposal.

    Returns `(consequential_decision, mechanism, evidence)`, or None when the episode carries no
    refusal at all -- in which case this script has nothing to attribute and says so rather than
    inventing a decision. Those episodes belong to a different failure class (the answer was never
    retrieved, and no constraint was violated on the way), and filing them under a refusal would
    corrupt the grouping every downstream rank depends on.

    "Backward to the earliest consequential decision" is the shape `docs/THE_LOOP.md` describes: for
    each failed episode, walk the trajectory in reverse to the earliest decision whose alternative
    would have made the answer reachable, and credit the loss THERE. Here that is the first turn the
    validator refused, because before it the episode was still on a reachable path.
    """
    refused = [(i, e) for i, e in enumerate(events)
               if runtime.boundary_key(e) == "post_execution" and e["violation_class"] is not None]
    if not refused:
        return None
    index, event = refused[0]
    cls = str(event["violation_class"])
    kind = "a tool call" if event["proposes_tool_call"] else "a final answer"
    # The DECISION, phrased from what the trace shows and nothing else: which kind of proposal, and
    # which constraint refused it. This string is a GROUPING KEY -- `build_residual_problems` groups on
    # it -- so it must name the decision and carry no remedy.
    decision = f"proposing {kind} that the {cls} constraint refuses"
    mechanism = (f"the episode proposed {kind} which was refused on {cls}, "
                 f"and went on to accumulate {len(refused)} refusal(s) in total")
    evidence = (f"first refusal at turn {index // 2}; classes seen: "
                + ", ".join(sorted({str(e['violation_class']) for _, e in refused})))
    return decision, mechanism, evidence


# ------------------------------------------------------------------------------------------------
# 2. the residual, ranked two ways
# ------------------------------------------------------------------------------------------------
def rank_two_ways(diagnoses, events_by_case) -> dict:
    """Linked-episode loss against event volume, so the difference is visible not asserted."""
    by_episode = collections.Counter(d.consequential_decision for d in diagnoses)
    by_event: collections.Counter = collections.Counter()
    for events in events_by_case.values():
        for event in events:
            if event.get("violation_class") is not None:
                by_event[str(event["violation_class"])] += len(event["violation_classes"])
    return {
        "by_linked_episodes": [{"key": k, "episodes": v} for k, v in by_episode.most_common()],
        "by_event_volume": [{"class": k, "events": v} for k, v in by_event.most_common()],
    }


# ------------------------------------------------------------------------------------------------
# 3. behavioural dedup
# ------------------------------------------------------------------------------------------------
def dedupe_behaviourally(candidates, states) -> tuple[list, int]:
    """Keep one candidate per distinct (boundary, action, variant, FIRING SET) on these states.

    The firing set is computed with `firing_vector`, the same behavioural fingerprint the other
    benchmark's loop is required to select on -- so the narrowing is a statement about the evidence
    rather than about anyone's preference. A candidate is dropped only when another candidate already
    in the set fires on exactly the same states at the same boundary with the same action.

    Order is preserved, so the representative kept for a behaviour is the first the search produced.
    That keeps the output deterministic without introducing a rank.
    """
    seen: set = set()
    kept: list = []
    for candidate in candidates:
        try:
            phi = _predicate_for(candidate.signal)
            vector = firing_vector(phi, states) if phi is not None else None
        except Exception:                                                   # noqa: BLE001
            vector = None
        instantiated = candidate.instantiated
        key = (str(getattr(candidate.boundary, "value", candidate.boundary)),
               str(getattr(instantiated.action, "value", instantiated.action)),
               str(instantiated.variant or ""), vector)
        if key in seen:
            continue
        seen.add(key)
        kept.append(candidate)
    return kept, len(candidates) - len(kept)


class _Phi:
    """Adapts a declared or expanded signal to the `.evaluate(state)` shape `firing_vector` wants."""

    def __init__(self, name: str) -> None:
        self.name = name

    def evaluate(self, state):
        return runtime.evaluate_signal(self.name, state, runtime.probe_params(self.name))


def _predicate_for(name: str):
    return _Phi(name) if name in runtime.declared_signals() else None


# ------------------------------------------------------------------------------------------------
# 4. candidates -> persisted specs
# ------------------------------------------------------------------------------------------------
class Unpersistable(RuntimeError):
    """A candidate whose signal cannot be written down, so it cannot be shipped as a spec.

    A REFUSAL OF ONE CANDIDATE, NOT OF THE ROUND. This used to be `SystemExit`, which does not derive
    from `Exception` -- so the per-candidate handler below (`except Exception`) could never catch it
    and a single unpersistable signal aborted the whole proposal, after the spec directory had already
    been cleared. `unexpressible_candidates` in `propose.json` is the field that was always meant to
    receive these, and an outcome recorded there is visible to the round; a traceback is not.

    Distinct from the ORACLE LEAK guards at the top of this module, which stay `SystemExit` on
    purpose: a leaked probe parameter invalidates every candidate, so there the round must not run.
    """


def spec_from(candidate, index: int, provenance: str):
    """One controller the core built -> one `anchoropt.controller.v1` spec.

    Every field is READ OFF the candidate. The theta payload is translated from the contract-side
    parameter names through `cctu_adapter.ETA_TO_THETA`, which exists because those two vocabularies
    are genuinely different and a proposer that guessed between them had every proposal declined for a
    reason that said nothing about its substance.

    `validate()` runs before the spec is written, so a candidate that cannot actually be installed
    fails here rather than at the start of a GPU run.
    """
    boundary = str(getattr(candidate.boundary, "value", candidate.boundary))
    instantiated = candidate.instantiated
    operator = str(getattr(instantiated.operator, "value", instantiated.operator))
    family = str(getattr(instantiated.action, "value", instantiated.action))

    mapping = runtime.ETA_TO_THETA.get(operator, {})
    theta: dict = {}
    for eta_key, theta_key in mapping.items():
        if eta_key not in instantiated.eta:
            continue
        value = instantiated.eta[eta_key]
        # A boolean theta key carries a REQUEST, not prose: the grounding describes what to delete in
        # words, and the executor derives the names per call. Passing the prose through would ask for
        # an argument literally named "delete exactly the argument names ...".
        theta[theta_key] = True if theta_key.startswith("drop_unknown") else value
    if operator == "suppress" and instantiated.variant == "withhold_execution":
        theta["keep_call"] = True

    # A SYNTHESIZED SIGNAL IS PERSISTED AS AN EXPRESSION, NOT AS A NAME.
    #
    # `spec.validate()` below checks the signal against the host's vocabulary, and in THIS process
    # expansion has registered every synthesized name -- so writing `signal=<synthesized name>`
    # validates here and is refused by every other process, including the runner. That is not a
    # hypothetical: it made 312 of 330 expanded candidates on granite and 300 of 318 on qwen
    # unloadable, and the screens reported them UNSCREENABLE rather than vetoed.
    #
    # `ControllerSpec` has carried the alternative from the start -- "`predicate` is accepted as an
    # alternative ... Exactly one of `signal` / `predicate` must be given" -- and this is the caller
    # that has to use it.
    shipped = candidate.signal in runtime.SIGNALS
    predicate: dict = {}
    signal = candidate.signal
    if not shipped:
        expr = runtime.expanded_expression(candidate.signal)
        if not expr:
            # REFUSE rather than write a spec that cannot be loaded. A signal with no serializable
            # form is a real outcome -- it fires this session and cannot be shipped -- and the round
            # must see it here, at propose time, not as an arm that will not start.
            raise Unpersistable(
                f"synthesized signal {candidate.signal!r} has no persistable expression: "
                f"`install_signal` was called without `expr` and none could be derived from the "
                f"predicate. Writing it as `signal` would produce a spec that validates here and is "
                f"refused by the runner.")
        predicate, signal = {"name": candidate.signal, **expr}, ""

    spec = mwmod.ControllerSpec(
        controller_id=f"cctu_{ROUND_TAG}_{index:02d}_{operator}_{instantiated.variant or 'default'}",
        boundary=boundary, action=family, signal=signal, predicate=predicate, theta=theta,
        provenance=provenance)
    spec.validate()
    return spec, {"controller_id": spec.controller_id, "boundary": boundary, "family": family,
                  "operator": operator, "variant": instantiated.variant,
                  "signal": candidate.signal, "eta": dict(instantiated.eta),
                  "theta": theta, "detail": instantiated.detail}


# ------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="CCTU cycle 1, propose phase. No model, no GPU.")
    ap.add_argument("--run", required=True, help="the CONTROL run directory to mine")
    ap.add_argument("--out", default=str(REPO / "rounds" / "CCTU_CYCLE1"))
    ap.add_argument("--split", default=None, help="corpus split; read from the manifest if omitted")
    ap.add_argument("--exhaust", action="store_true",
                    help="run the FULL schedule: exhaust every boundary and EXPAND Phi. Off by "
                         "default, because holding the vocabulary fixed while the policy is "
                         "optimised is the block-coordinate discipline -- expansion is what a STUCK "
                         "round escalates to, not where a first round starts")
    ap.add_argument("--tag", default="c1",
                    help="ROUND TAG in every controller_id, and therefore in every run directory "
                         "name: `train_cctu_<tag>_<index>_<operator>_<variant>`. Default `c1` for "
                         "backward compatibility with the rounds already on disk. PASS A FRESH ONE "
                         "PER ROUND. The index is positional, so two rounds' candidate #1 share a "
                         "run directory under one tag, and `cctu_run_arms.py` then SKIPS the new arm "
                         "as `already complete` and prints the OLD run's compare command -- measured "
                         "on cycle 4's first dry run, which silently adopted cycle 1's arms 01-03 and "
                         "collided with cycle 3's in-flight arm 04. Same hazard as the stale-spec one "
                         "fixed above, one level up: there across proposals into one directory, here "
                         "across rounds into one results tree")
    ap.add_argument("--metric", default="PSR", choices=("acc", "SR", "PSR"),
                    help="which failures form the residual (default PSR: the metric a constraint "
                         "intervention moves)")
    args = ap.parse_args()

    global ROUND_TAG
    if not _TAG_OK.match(str(args.tag)):
        raise SystemExit(f"--tag {args.tag!r} must match {_TAG_OK.pattern}: it becomes part of a "
                         f"controller_id, a spec filename and a run directory name")
    ROUND_TAG = str(args.tag)

    assert_no_oracle_leak()
    run = Path(args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((run / "run_manifest.json").read_text())
    split = args.split or manifest["split"]
    corpus = DATA_DIR / (f"input_data_{split}.jsonl" if split != "all" else "input_data.jsonl")
    samples = {json.loads(line)["id"]: json.loads(line)
               for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()}
    detail = {json.loads(x)["id"]: json.loads(x)
              for x in (run / "detail.jsonl").read_text().splitlines() if x.strip()}
    transcripts = {json.loads(x)["id"]: json.loads(x)["messages"]
                   for x in (run / "response.jsonl").read_text().splitlines() if x.strip()}

    print("=" * W)
    print(f"CCTU CYCLE 1 · PROPOSE · {run}")
    print("=" * W)
    print(f"  model={manifest['model']} split={split} config={manifest['config']} "
          f"residual metric={args.metric}")

    # ---- [1] evidence ---------------------------------------------------------------------------
    replicate = min({k.rsplit("_", 1)[1] for k in transcripts})
    failing = [k for k, v in detail.items()
               if k.rsplit("_", 1)[1] == replicate and not v[args.metric]]
    events_by_case: dict[str, list[dict]] = {}
    states_by_case: dict[str, list[dict]] = {}
    records: list[dict] = []
    unattributed: list[str] = []

    for case_id in sorted(failing):
        query = case_id.split("_")[0]
        if query not in samples or case_id not in transcripts:
            continue
        events, states = replay_episode({**samples[query], "id": case_id}, transcripts[case_id])
        events_by_case[case_id] = events
        states_by_case[case_id] = states
        got = attribute(events)
        if got is None:
            unattributed.append(case_id)
            continue
        decision, mechanism, evidence = got
        records.append({"case_id": case_id, "mechanism": mechanism, "evidence": evidence,
                        "consequential_decision": decision,
                        # EMPTY on purpose -- see the module docstring.
                        "proposed_behavior_change": "",
                        "metadata": {"termination": detail[case_id]["termination"]}})

    print(f"\n[1] EVIDENCE  {len(failing)} failing episodes on {args.metric} (replicate {replicate})")
    print(f"    attributed to a refused proposal: {len(records)}")
    print(f"    NOT attributed (no refusal in the episode): {len(unattributed)} -- a different "
          f"failure class, left out rather than filed under a refusal")
    if REPLAY_ERRORS:
        print(f"    replay errors (upstream handler raised): {len(REPLAY_ERRORS)}; "
              f"e.g. {REPLAY_ERRORS[0][:80]}")

    diagnoses = diagnoses_from(PROVIDER, records)

    # ---- [2] the residual, ranked -----------------------------------------------------------------
    ranks = rank_two_ways(diagnoses, events_by_case)
    print("\n[2] RANKED BY LINKED EPISODE LOSS (the unit an anchor is judged in)")
    for row in ranks["by_linked_episodes"][:6]:
        print(f"    {row['episodes']:4d} episodes   {row['key'][:70]}")
    print("    ... and by EVENT VOLUME, which is the ordering to distrust:")
    for row in ranks["by_event_volume"][:6]:
        print(f"    {row['events']:4d} events     {row['class']}")

    problems = build_residual_problems(
        diagnoses, expressible=lambda d: expressible_under(d, runtime=runtime)[0])
    print("\n[3] RESIDUAL PROBLEMS")
    for problem in problems[:6]:
        print(f"    R{problem.rank} support={problem.support:4d} "
              f"coverage={problem.coverage:.2f} expressible={problem.expressible}  "
              f"{problem.key[:58]}")
    if not problems:
        print("    none: nothing to search")
        return 1

    # ---- [4] THE SEARCH. Not this script's decision. ---------------------------------------------
    top = problems[0]
    events = [e for cid in sorted(top.case_ids) for e in events_by_case.get(cid, [])]
    states = [s for cid in sorted(top.case_ids) for s in states_by_case.get(cid, [])]
    print(f"\n[4] STRUCTURED SEARCH on R{top.rank}  ({len(events)} events, {len(states)} states)"
          f"   mode={'exhaust + expand Phi' if args.exhaust else 'declared Phi, structural stop'}")
    # TWO MODES, and the default is the disciplined one.
    #
    #   evaluate=None      the documented structural mode. The schedule stops at the first boundary
    #                      where a controller is REALIZABLE and does not expand Phi. Measured on this
    #                      residual: 6 candidates over 2 declared signals -- a round you can run.
    #   --exhaust          a null evaluator, so every candidate scores identically and nothing is
    #                      preferred, but the schedule exhausts each boundary and widens Phi.
    #                      Measured: 345 candidates over 78 signals, which is not a round.
    #
    # Default off, because `docs/THE_LOOP.md`'s block-coordinate rule is to hold the vocabulary fixed
    # and optimise the policy, and to refine the vocabulary only when no installable policy remains at
    # that resolution. The other benchmark followed exactly that order -- its first four anchors came
    # from the declared lexical vocabulary and expansion arrived at T5, when 3% residual coverage
    # showed the vocabulary was exhausted. Starting a first round with 78 synthesized signals inverts
    # the ladder and buys 340 arms nobody can run.
    if args.exhaust:
        outcome = optimize_residual(top, runtime=runtime, host=runtime.HOST, events=events,
                                    states=states, evaluate=lambda _a: 0, improves=lambda _o: False)
    else:
        outcome = optimize_residual(top, runtime=runtime, host=runtime.HOST, events=events,
                                    states=states, evaluate=None)
    print(f"    boundaries derived: {list(outcome.boundaries)}")
    print(f"    visited: {list(outcome.visited)}   moves_earlier={outcome.moves_earlier}")
    print(f"    state={outcome.state}  Phi expanded={outcome.expanded} "
          f"({len(outcome.signals_installed)})  candidates={len(outcome.candidates)}")
    for attempt in outcome.attempts:
        print(f"      {attempt.boundary:26s} {attempt.state:28s} built={attempt.candidates_built}")
    refusals = collections.Counter(c.state for c in outcome.certificates)
    if refusals:
        print(f"    refusals: {dict(refusals.most_common(6))}")

    # ---- [5] BEHAVIOURAL DEDUP, which is not a preference -----------------------------------------
    #
    # 345 candidates is not 345 arms. Phi-expansion emits many names over few distinct firing sets --
    # a threshold grid is the obvious case -- and two candidates that fire on the SAME states at the
    # same boundary with the same action are the same intervention under two labels. Measuring both
    # spends an arm that cannot differ in outcome.
    #
    # The comparison is `firing_vector` over the states this residual actually contains, which is why
    # the reduction is not a choice: nothing is ranked, nothing is preferred, and a candidate is only
    # dropped when another candidate ALREADY IN THE SET is indistinguishable from it on the evidence.
    # `tests/test_cctu_cycle1_thinness.py` pins that this is how the set is narrowed, because a driver
    # that narrowed by name or by hunch would be choosing the answer.
    kept, folded = dedupe_behaviourally(outcome.candidates, states)
    print(f"\n[5] BEHAVIOURAL DEDUP  {len(outcome.candidates)} candidates -> {len(kept)} distinct "
          f"behaviours ({folded} folded as indistinguishable on this residual's states)")

    # ---- [6] candidates -> specs ------------------------------------------------------------------
    specs_dir = out_dir / "candidates"
    specs_dir.mkdir(parents=True, exist_ok=True)
    # CLEARED BEFORE WRITING, and this is a defect fix rather than tidiness.
    #
    # Specs are named `cctu_c1_<index>_<operator>_<variant>.json`, so a re-proposal that assigns a
    # different (operator, variant) to a given index writes a NEW filename and leaves the old one in
    # place. The directory then accumulates candidates from every previous run, and nothing downstream
    # can tell them apart: `cctu_screens.py` screens whatever is in the directory and
    # `cctu_run_arms.py` queues from the screen report, so a stale candidate can be measured as though
    # this proposal had produced it.
    #
    # Measured before the fix: one qwen candidates/ directory held 578 files of which 237 were stale
    # from an earlier proposal -- 41% -- including six exact duplicates among the thirteen that passed
    # screening. `dedupe_behaviourally` was working correctly the whole time; the duplication was
    # across RUNS, where it cannot see.
    #
    # `evaluation.py` already applies exactly this discipline to `detail.jsonl` (`if args.detail and
    # os.path.exists(detail_path): os.remove(...)`) for the same reason: an artifact that appends
    # across runs is indistinguishable from one describing this run.
    stale = sorted(specs_dir.glob("*.json"))
    for path in stale:
        path.unlink()
    if stale:
        print(f"\n[5b] CLEARED {len(stale)} spec(s) from a previous proposal in {specs_dir}")
    written, failed = [], []
    provenance = f"cycle1_propose from {run} R{top.rank}"
    for i, candidate in enumerate(kept, start=1):
        try:
            spec, row = spec_from(candidate, i, provenance)
        except Exception as exc:                                            # noqa: BLE001
            failed.append(f"candidate {i}: {type(exc).__name__}: {exc}")
            continue
        spec.write(specs_dir / f"{spec.controller_id}.json")
        written.append(row)

    print(f"\n[6] CANDIDATE CONTROLLERS  {len(written)} written to {specs_dir}")
    for row in written:
        print(f"    {row['controller_id']:44s} {row['signal']}")
    if failed:
        print(f"    {len(failed)} candidate(s) could not be expressed as a spec:")
        for line in failed[:6]:
            print(f"      {line}")

    record = {
        "run": str(run), "model": manifest["model"], "split": split, "metric": args.metric, "exhaust": args.exhaust,
        "failing_episodes": len(failing), "attributed": len(records),
        "unattributed": unattributed,
        "ranking": ranks,
        "problems": [{"rank": p.rank, "support": p.support, "coverage": round(p.coverage, 4),
                      "expressible": p.expressible, "key": p.key} for p in problems],
        "searched": {"rank": top.rank, "key": top.key,
                     **{k: v for k, v in outcome.as_dict().items() if k != "promoted"}},
        "candidates_built": len(outcome.candidates), "candidates_distinct": len(kept),
        "candidates": written, "unexpressible_candidates": failed,
        "replay_errors": REPLAY_ERRORS,
    }
    (out_dir / "propose.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n[record] {out_dir / 'propose.json'}")

    print("=" * W)
    # A ROUND WITH NOTHING RUNNABLE IS NOT A SUCCESS, and it only became reachable when the
    # per-candidate refusal above stopped aborting the process. `propose.json` is already written, so
    # the evidence for WHY every candidate was refused survives -- but the exit status must not say
    # the round is ready when there is no arm to run.
    if kept and not written:
        print(f"NOTHING TO RUN: all {len(kept)} distinct candidate(s) were refused; see "
              f"`unexpressible_candidates` in propose.json.")
        print("=" * W)
        return 1
    print("NO BENEFIT IS CLAIMED. With a null evaluator the search reports structural rediscovery:")
    print("a controller was built and a primitive can run it. Which one helps is the arm's question,")
    print("and the acceptance criteria must be frozen BEFORE that arm runs.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
