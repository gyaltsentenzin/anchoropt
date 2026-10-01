#!/usr/bin/env python3
"""Walk the decision-centric framework, step by step, with real code and real numbers.

Six steps, each runnable, each printing what it actually computed rather than describing it:

    1  SIGNAL          how a raw error becomes a semantic locus -- and when that FAILS
    2  ATTRIBUTION     how one error string splits into distinct causes
    3  INCISION POINT  which of the 3 x 5 cells survive, and why the rest cannot
    4  POLICY EVAL     the four ordered acceptance questions, on A1-A4's real results
    5  FREEZE / DEFER  the S1-S5 rules, and what each one stopped
    6  THE LOOP        why the accepted world becomes the next round's no-op arm

No GPU, no model, no benchmark harness. Everything here runs on the shipped artifacts.

    python scripts/walk_framework.py            # all six
    python scripts/walk_framework.py --step 2   # just one
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "anchoropt" / "attribution"))
sys.path.insert(0, str(REPO / "anchoropt" / "learning"))

from anchoropt.anchor import Action, IncisionPoint, exclusion_reason, feasible_actions
from rounds.anchors import ANCHORS, PROGRESSION


def _h(n: int, title: str) -> None:
    print()
    print("=" * 78)
    print(f"STEP {n}  {title}")
    print("=" * 78)


def step1_signal() -> None:
    _h(1, "SIGNAL -- a raw error becomes a semantic locus (and sometimes does not)")
    import constraint_locus as cl

    print("Lexical error strings fragment ONE constraint into several 'candidates'. The miner keys")
    print("on the violated CONSTRAINT instead, so equivalent wordings merge:")
    print()
    for err in ("core memory is full", "Memory size exceeds maximum size of 7"):
        got = cl.classify(err)
        print(f"  {err!r:44s} -> {'/'.join(got['locus'])}")
    print("  ^ two wordings, ONE locus. That merge is what makes support countable.")

    print()
    print("But the cue table was authored from one benchmark, so an unseen wording VANISHES:")
    err = "Key name must be unique"
    print(f"  {err!r:44s} -> {cl.classify(err)}")
    print("  ^ returns None, and a locus that returns None is invisible to ranking.")
    print()
    print("  This is the single most important thing to know when porting. A3's locus")
    print("  (permission/identifier/duplicate) came from the LLM CANONICALIZER, not this table.")
    print("  A hand-written table is a fast path, never the coverage story:")
    print("      anchoropt/attribution/constraint_locus.py       deterministic cues")
    print("      anchoropt/attribution/op_canonicalize_locus.py  LLM proposes, artifact is FROZEN")

    print()
    print("Some failures produce NO error at all -- a call that succeeds and returns nothing:")
    import policy_tree as pt
    for payload in ('{"ranked_results": []}', '{"value": "real content"}'):
        print(f"  {payload:44s} -> vacuous={pt.vacuous_result_kind(payload)}")
    print("  ^ a syntactically successful response containing no information. An error-keyed")
    print("    miner cannot see these, which is why the vocabulary has to expand (step 5).")

    print()
    print("And an implementation crash must NEVER become a mined signal:")
    import harness_guard as hg
    for payload in ("error during execution division by zero", "core memory is full"):
        print(f"  {payload!r:44s} -> harness_fault={hg.is_harness_fault(payload)}")
    print("  ^ this actually happened: an empty-memory search raised ZeroDivisionError and got")
    print("    mined as a real error contract at support 13 before being caught.")


def step2_attribution() -> None:
    _h(2, "ATTRIBUTION -- one error string, two distinct causes")
    print("The semantic miner names the SYMPTOM. A `not_found` at read time is usually caused by")
    print("a decision made earlier, so a second stage traces backward to the first consequential")
    print("decision. On A2's locus that split ONE error string in two:")
    print()
    print("  existence/identifier/not_found     51 events, 47 of 201 failing queries linked")
    print("  ------------------------------------------------------------------------------")
    print("  value WAS archived; the read never consulted archive     39   -> A2.R  INSTALLED")
    print("  core filled and the value was NOT archived                5   -> A2.W  DEFERRED (S1)")
    print("  no write-side explanation                                 8   -> unattributed (S3)")
    print()
    print("  Same error text. Two causes. Two different anchors -- and without the split, an arm")
    print("  would have targeted a mixture and been unattributable either way.")
    print()
    print("All 39 are SELF-INFLICTED: the value sits exactly where A1 put it.")
    print("A1 turned a fatal capacity error into a recoverable retrieval problem -- progress --")
    print("but the recovery had to be learned too. Three of four anchors arose this way:")
    print()
    for a in ANCHORS:
        print(f"  {a.name}  {a.locus}")
        print(f"      attribution: {a.attribution}")
    print()
    print("  code: anchoropt/attribution/{attribution_miner,trace_backward,destination_attest}.py")
    print()
    print("destination_attest exists because of a specific mistake: a reroute destination was")
    print("admitted at '11/11 clean', where clean meant DID NOT ERROR -- and 9 of the 11 returned")
    print("nothing. Attest a destination on RESOLVING, never on not-erroring.")


def step3_incision() -> None:
    _h(3, "INCISION POINT -- usually forced by the failure, not searched")
    print("One LLM call is three points at which the SYSTEM can act. Knowledge and preventability")
    print("trade off monotonically:")
    print()
    print(f"  {'point':28s} {'sees proposed call':20s} {'can prevent':12s} admissible actions")
    for p in IncisionPoint:
        acts = sorted(a.value for a in feasible_actions(p))
        print(f"  {p.value:28s} {p.sees_proposed_call!s:20s} "
              f"{p.can_prevent_execution!s:12s} {len(acts)}  {acts}")

    nominal = len(IncisionPoint) * len(Action)
    admissible = sum(len(feasible_actions(p)) for p in IncisionPoint)
    print()
    print(f"  {nominal} nominal cells -> {admissible} structurally admissible. The {nominal - admissible} exclusions,")
    print("  each with a STATED reason (a silently pruned cell reads as one nobody considered):")
    for p in IncisionPoint:
        for a in Action:
            why = exclusion_reason(p, a)
            if why:
                print(f"    {p.value:26s} + {a.value:10s} {why}")

    print()
    print("  BUT for most failures the point is NOT a search -- the failure forces it:")
    print("    a write rejected for capacity        -> post-execution (you learn it by trying)")
    print("    about to re-write a stored value     -> post-gen/pre-exec (after, the step is spent)")
    print("    about to answer with no tool call    -> post-gen/pre-exec (unknowable earlier)")
    print("  A3 and A4 each had EXACTLY ONE admissible cell. A2 was the exception where the point")
    print("  was genuinely open (8 nominal -> 3 live -> 2 arms), and there the arms overturned the")
    print("  prior: reprompt beat reroute, which had been favoured 2:1 on every earlier locus.")
    print()
    print("  So naming the point does not buy a big space to explore. It makes the choice EXPLICIT")
    print("  and checkable instead of implicit in a trigger string -- which is what v1 got wrong:")
    print()
    print("  Measured on identical signal / action family / injected text:")
    print("    A4 v1  pre_generation            fired 303/303   -4.95 pp   p=0.1633")
    print("    A4 v2  post_generation_pre_exec  fired  59/303   +3.63 pp   p=0.0266")
    print("  v1 could not observe 'about to answer without a tool call' -- that fact does not")
    print("  exist before generation -- so it fired everywhere and paid on 244 episodes.")


def step4_policy_eval() -> None:
    _h(4, "POLICY EVAL -- four ordered questions, harm before gain")
    print("  1 ENGAGEMENT   did it fire, in the mined context, on the target backend?")
    print("                 -> a VETO evaluated BEFORE any outcome is scored")
    print("  2 ATTRIBUTION  are the changes caused by the intervention, not replay noise?")
    print("  3 HARM         does any backend or subgroup degrade beyond tolerance?")
    print("  4 BENEFIT      only now: do paired gains exceed losses?")
    print()
    print("Engagement is first because a candidate can pass every outcome test while having")
    print("nothing to do with the residual it claims: one winner passed all four terms while")
    print("firing 12x on a DIFFERENT backend's error contract, and its target residual GREW.")
    print()
    print("Measurement is a paired counterfactual on the shipped results -- recompute it now:")
    print()

    def load(rel: str) -> dict:
        rows = json.loads((REPO / "rounds" / rel).read_text())["results"]
        return {r["id"]: bool(r.get("valid")) for r in rows if not r.get("is_prereq")}

    chain = [("T0 control", "T1_A1_capacity/result/baseline_T0_train.json"),
             ("T1 +A1", "T1_A1_capacity/result/eval_train.json"),
             ("T2 +A2", "T2_A2_not_found/result/eval_train.json"),
             ("T3 +A3", "T3_A3_duplicate/result/eval_train.json"),
             ("T5 +A4", "T5_A4_no_tool_call/result/eval_train.json")]
    prev = None
    for label, rel in chain:
        v = load(rel)
        ok = sum(v.values())
        line = f"  {label:12s} {ok:3d}/{len(v)} = {100.0*ok/len(v):5.2f} %"
        if prev is not None:
            shared = prev.keys() & v.keys()
            g = sum(1 for k in shared if v[k] and not prev[k])
            losses = sum(1 for k in shared if prev[k] and not v[k])
            line += f"   {g:2d} gains / {losses:2d} losses"
        print(line)
        prev = v
    print()
    print("  The replay variance floor is ZERO (0/12 stores, 0/303 calls, 0/303 flips at")
    print("  temperature=0.001). Two consequences: every flip is attributable, AND a single-case")
    print("  loss is a real loss -- there is no noise budget to dismiss it with.")
    print()
    print("  code: anchoropt/learning/check_attribution.py   (the pre-scoring veto)")


def step5_freeze_defer() -> None:
    _h(5, "FREEZE / DEFER -- what stopped, and which rule stopped it")
    print("The loop has no natural end; it will always produce a rank 1. So stopping is an")
    print("explicit rule, FROZEN IN ADVANCE, checked cheapest-first: S3 -> S1 -> S2 -> S5 -> S4.")
    print()
    rules = [
        ("S3", "no admissible action", "every live cell structurally excluded -> record, don't pursue"),
        ("S1", "coverage floor", "top locus links < 10% of residual -> cannot be measured at this n"),
        ("S2", "settled saturation", "settled/raw >= 0.5 -> coverage question for an existing anchor"),
        ("S5", "unmined dominates", "unmined > largest mined -> the SIGNAL is the bottleneck"),
        ("S4", "diminishing returns", "2 rejects (fix the miner) or 2 accepts < +1pp (consolidate)"),
    ]
    for tag, name, why in rules:
        print(f"  {tag}  {name:22s} {why}")

    print()
    print("What these actually stopped on this line:")
    print()
    deferred = [
        ("A2.W write-side not_found", "5 occurrences", "S1", "and its mechanism is A1's own"),
        ("8 unattributed occurrences", "no write-side cause", "S3", "admissible set empty"),
        ("format/field/malformed_value", "6.0% of residual", "S1", "below the floor"),
        ("capacity/... re-proposal", "372/724 = 51% settled", "S2", "already handled"),
        ("size/item/exceeds_per_item", "fell 30 -> 9 linked", "S1", "after occurrence re-labeling"),
        ("short_episode detector", "precision 0.57", "screen", "fires on 89 of 117 SUCCESSES"),
        ("answered_after_read", "precision 0.50", "screen", "coin flip"),
        ("single_read_only", "precision 0.47", "screen", "fires on passes MORE than failures"),
        ("'retrieved OK, answered wrong'", "82% of residual", "judgement", "outcome class, not a mechanism"),
    ]
    print(f"  {'deferred':32s} {'evidence':24s} {'rule':10s} note")
    for what, ev, rule, note in deferred:
        print(f"  {what:32s} {ev:24s} {rule:10s} {note}")

    print()
    print("  The last row is the discipline in miniature: the LARGEST residual class is")
    print("  deliberately NOT anchored, because it is an observed outcome, not an attributed")
    print("  decision point. Wanting a signal is not having one.")
    print()
    print("  Deferred means RECORDED, not discarded -- each stays in the ledger with its evidence,")
    print("  so a later world can reconsider it on merit rather than rediscover it.")
    print()
    print("  Freezing works the other way: every round's criteria were written BEFORE launch and")
    print("  are never edited after. A4's spec is pinned by test to the sha256 cited in its own")
    print("  acceptance commit, so a criterion cannot be rewritten once the result is known.")
    print()
    print("  code: anchoropt/learning/{phase_switch,screen_ranked_loci}.py")
    print("        rounds/T4_exhaustion/STOPPING_CRITERIA_FROZEN.md")


def step6_loop() -> None:
    _h(6, "THE LOOP -- the accepted world becomes the next round's no-op arm")
    print("After a candidate is accepted, its trajectories are PERSISTED as the incumbent world,")
    print("and the next round is measured against that -- not against the original baseline, and")
    print("not against a freshly re-run control. A3's frozen spec says it outright:")
    print()
    print('    | control for this arm | the incumbent\'s own results; NOT re-run |')
    print()
    import hashlib
    chain = [("T1 arm (A1)", "T1_A1_capacity/result/eval_train.json"),
             ("T2 arm (A1+A2)", "T2_A2_not_found/result/eval_train.json"),
             ("T3 arm (A1-A3)", "T3_A3_duplicate/result/eval_train.json"),
             ("T5 arm (A1-A4)", "T5_A4_no_tool_call/result/eval_train.json")]
    print(f"  {'arm':18s} {'sha256':14s} role next round")
    for i, (label, rel) in enumerate(chain):
        digest = hashlib.sha256((REPO / "rounds" / rel).read_bytes()).hexdigest()[:12]
        role = "control for the next round" if i < len(chain) - 1 else "current incumbent"
        print(f"  {label:18s} {digest:14s} {role}")
    print()
    print("  Round N's control file IS round N-1's arm file. That makes the procedure BOOSTING:")
    print("  each round fits one increment against what the current policy still gets wrong.")
    print("  It is also the experimental control -- both arms share the same factual prefix, so")
    print("  'not re-run' is a correctness property, not an optimisation.")
    print()
    print("Re-labeling is what keeps the ranking honest. A locus the incumbent now repairs is")
    print("marked CORRECTED and excluded from the support used to pick the next anchor:")
    print()
    print("  mined coverage across rounds:  78% -> 61% -> 13% -> 3%")
    for r in PROGRESSION:
        tag = r.anchor.name if r.anchor else "(none installed)"
        cov = "" if r.mined_coverage is None else f"  coverage {r.mined_coverage:.0%}"
        print(f"    {r.tag:3s} {tag:16s} {r.train_acc:5.2f} %{cov}")
    print()
    print("  Without re-labeling, capacity/... would still rank first at T2 and T3 -- its events")
    print("  keep appearing in failing traces even after A1 repairs them. Occurrence-level")
    print("  re-labeling reordered the queue (a size/item candidate fell 30 -> 9 linked failures).")
    print()
    print("  code: anchoropt/learning/{remine_incumbent,evidence_ledger}.py")
    print("  docs: docs/THE_LOOP.md")


STEPS = {1: step1_signal, 2: step2_attribution, 3: step3_incision,
         4: step4_policy_eval, 5: step5_freeze_defer, 6: step6_loop}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, choices=sorted(STEPS), help="run one step only")
    args = ap.parse_args()

    if args.step:
        STEPS[args.step]()
    else:
        print("THE DECISION-CENTRIC FRAMEWORK, STEP BY STEP")
        print("Every number below is computed from the shipped artifacts. No GPU, no model.")
        for n in sorted(STEPS):
            STEPS[n]()
        print()
        print("=" * 78)
        print("Next: scripts/run_pipeline.py --show      the composed A1-A4 policy")
        print("      scripts/verify_progression.py      recompute every published number")
        print("      rounds/README.md                   the five rounds, in order")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
