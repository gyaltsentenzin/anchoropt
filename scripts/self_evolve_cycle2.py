#!/usr/bin/env python3
"""CYCLE #2 -- the closed loop, driven by the frozen core rather than by this script.

    python scripts/self_evolve_cycle2.py \
        --incumbent results/a9repro/<promoted arm> --out rounds/CYCLE2 --live

    moving incumbent's trajectories
      -> mine the CURRENT residual distribution (post-promotion, not the historical one)
      -> optimize_residual(): WHERE -> WHAT -> HOW, block-coordinate
      -> emit the next arm as a runnable cluster spec
      -> [GPU: paired evaluation]  --score-> promote or reject
      -> regenerate trajectories -> re-mine -> the residual distribution has MOVED

WHAT MAKES THIS CYCLE #2 AND NOT CYCLE #1 AGAIN. Cycle #1 mined the residual of the pre-A9 incumbent.
This one starts from the incumbent A9 was promoted INTO, so the residual distribution it mines is the
one A9's own success created. If the loop works, the next intervention it proposes is not A9 and not a
paraphrase of A9 -- it addresses what is left after A9 removed 17 failures. That shift is the thing to
report, and `[4] RESIDUAL SHIFT` measures it directly.

THE SEARCH IS NOT HERE. Everything from boundary derivation through signal expansion to eta grounding
is `anchoropt.learning.structured_search.optimize_residual`. This script supplies evidence (failed
trajectories -> attribution -> residual problems), turns the chosen controller into a runnable arm,
and scores the paired result when one comes back. It holds no IncisionPoint, no Action and no
AnchorPolicyOpt, so it cannot choose a locus, a signal or an action -- the same discipline
anchor_recovery_replay.py is now held to, and for the same reason: a loop that decides its own answer
is not evidence that the algorithm decides it.

TWO PHASES, because one of them needs a GPU.
    --propose   (default) mine -> search -> write the arm spec + runner. No GPU.
    --score     read the paired run back, decide promote/reject, then RE-MINE and report the shift.
"""

from __future__ import annotations

import argparse
import collections
from collections.abc import Mapping
import importlib.util
import json
import os
import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))
sys.path.insert(0, str(REPO / "integrations" / "claude_roles"))

import bfcl_runtime as runtime                                              # noqa: E402
from anchoropt.learning.candidate_search import expressible_under           # noqa: E402
from anchoropt.learning.learned_signal import firing_vector                 # noqa: E402
from anchoropt.learning.proposal_seams import (                             # noqa: E402
    AttributionSchemaError, ingest_attribution,
)
from anchoropt.learning.residual_problem import build_residual_problems     # noqa: E402
from anchoropt.learning.search_state import IMPROVED, NO_BENEFIT            # noqa: E402
from anchoropt.learning.external_evaluation import (                        # noqa: E402
    UNEVALUATED as EXT_UNEVALUATED,
    ExternalEvaluation, arm_manifest, select_on_measurement,
)
from anchoropt.learning.delayed_effect import evaluate_delayed_effect
from anchoropt.learning.golden_registry import ControllerIdentity       # noqa: E402
from anchoropt.learning.candidate_library import (                       # noqa: E402
    ACCEPTED, DISCOVERED, MEASURED_NEGATIVE, PENDING_VALIDATION, CandidateLibrary,
    CandidateRecord, EvaluationContext, Measurement, stack_fingerprint)
from anchoropt.learning.termination import (                                # noqa: E402
    Validity, channel_integrity, classify_residual, global_verdict,
)
from anchoropt.learning.structured_search import optimize_residual          # noqa: E402
from anchoropt.learning.acceptance_criteria import evaluate_criteria        # noqa: E402
# LOADED BY FILE PATH, deliberately, and neither as a bare top-level module nor through the package.
# A bare `import mechanism_evidence` resolved only when the CWD was that directory and broke
# `preflight_controller.py` (which imports this driver for its state loader) with ModuleNotFoundError.
# `from evaluator.mechanism_evidence import ...` is worse: `evaluator/__init__.py` eagerly imports the
# whole MemoryAnchorOptEvaluator, which drags in a heavyweight and currently broken gate import, so
# adjudicating acceptance would depend on the evaluator being importable in the scoring process. It
# must not. The translator is a pure function over sidecar dicts and is loaded as exactly that.
_ME_PATH = REPO / "benchmarks" / "bfcl_v4" / "evaluator" / "mechanism_evidence.py"
_ME_SPEC = importlib.util.spec_from_file_location("anchoropt_mechanism_evidence", _ME_PATH)
_ME = importlib.util.module_from_spec(_ME_SPEC)
_ME_SPEC.loader.exec_module(_ME)
mechanism_evidence = _ME.mechanism_evidence


class _Paired:
    """The paired train comparison in the shape `check_train` reads.

    `acceptance_criteria.check_train` wants `.net`, `.control`, `.gains`, `.losses` -- the
    `EvaluationResult` surface. The scoring phase already has exactly those numbers as plain dicts and
    lists, so this adapts them rather than constructing a synthetic EvaluationResult, which would
    invite a second source of truth for the paired result.

    `control` is the CELL, not the whole run: criterion 1's denominator must be the paired cases that
    were actually compared, which is what `n_paired` reports.
    """

    def __init__(self, bc, ac, cell, gains, losses):
        self.control = {c: bc[c] for c in cell}
        self.candidate = {c: ac[c] for c in cell}
        self.gains = tuple(gains)
        self.losses = tuple(losses)
        self.telemetry: dict = {}

    @property
    def net(self) -> int:
        return len(self.gains) - len(self.losses)

# NOT IMPORTED, deliberately: IncisionPoint, Action, AnchorPolicyOpt, SearchSpaceProposal,
# synthesize, expand_and_resume_predicates. This script cannot name a coordinate, so it cannot pick
# one. See tests/test_cycle2_thinness.py.

FIXTURES = ("replay_anchors", "replay_exprs", "fixtures.replay_anchors", "fixtures.replay_exprs")
W = 100


def assert_no_oracle_leak() -> None:
    leaked = sorted(m for m in sys.modules if any(f in m for f in FIXTURES))
    if leaked:
        raise SystemExit(f"ORACLE LEAK: anchor fixtures imported ({leaked})")
    import bfcl_signals as sig
    assert dict(sig.SIGNAL_ALIASES) == {}, "anchor-derived aliases are live"
    assert dict(sig.SIGNAL_PROBE_PARAMS) == {}, "a measured threshold is live"


def load_run(d: pathlib.Path):
    """(correct_by_case, steps_by_case) for one run directory. Accepts run/ or a flat dir."""
    for base in (d / "run", d):
        res = next(iter(sorted(base.glob("eval_*results*.json"))), None)
        if res is not None:
            break
    if res is None:
        raise SystemExit(f"no eval results under {d} (or {d}/run)")
    payload = json.load(open(res))
    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    correct = {str(r["id"]): bool(r.get("valid")) for r in rows if not r.get("is_prereq")}
    steps = {}
    for sub in ((d / "run" / "traj" / "query"), (d / "traj" / "query")):
        for fp in sorted(sub.glob("*.json")) if sub.exists() else ():
            try:
                ep = json.load(open(fp))
            except Exception:
                continue
            steps[str(ep.get("case_id"))] = ep.get("steps") or []
    return correct, steps


def keep_steps(steps):
    """Steps the runtime calls decisions, plus anything carrying a call.

    Mirrors the recovery harness: dropping the answer-commitment rows would delete the boundary the
    backward search needs, and the residual would derive exactly one boundary.
    """
    return [s for s in steps
            if (s.get("decoded") or []) or s.get("status") is not None or runtime.is_decision(s)]


def case_facts(cid: str, steps, backend: str, phase: str = "query") -> dict:
    """One failed case as RUNTIME FACTS: ordered trajectory, no label and no error class.

    Aggregate counts cannot express a conjunction ("the second store was never consulted"), which is
    the evidence gap that once blocked the whole expansion branch -- so the full ordered trace goes in.
    """
    rs = keep_steps(steps)
    trace, used_tools, used_stores = [], [], []
    for i, st in enumerate(rs):
        dec = [str(x) for x in (st.get("decoded") or [])]
        for c in dec:
            fn = c.split("(")[0].strip()
            if fn and fn not in used_tools:
                used_tools.append(fn)
            try:
                import adapter as _v
                store = _v.container_of(c)
            except Exception:
                store = None
            if store and store not in used_stores:
                used_stores.append(store)
        trace.append({
            "step": i, "status": st.get("status"), "tool_calls": dec or None,
            "tool_results": [str(x)[:300] for x in (st.get("tool_results") or [])] or None,
            "state": {k: st.get(k) for k in
                      ("turn", "failed_search_streak", "error_streak", "is_retrieval_success")
                      if st.get(k) is not None} or None,
            "stores_used_so_far": list(used_stores), "tools_used_so_far": list(used_tools),
        })
    calls = [c for s in trace for c in (s["tool_calls"] or [])]
    return {"case_id": cid, "backend": backend, "trajectory": trace,
            "episode_summary": {"steps": len(rs), "tool_calls": len(calls),
                                "distinct_tools_used": used_tools,
                                "distinct_stores_used": used_stores,
                                "made_no_tool_call": len(calls) == 0,
                                # THE OUTCOME MUST BE TRUE OF THIS EPISODE. A storage episode is not
                                # scored at all, so telling the attributor it "produced a final answer
                                # that was scored incorrect" is false evidence -- and false evidence
                                # about the outcome is exactly what biases a diagnosis toward the
                                # answer-side reasoning this phase does not contain.
                                "outcome": ("this episode was building the memory store and at least "
                                            "one of its operations returned an error"
                                            if phase == "prereq" else
                                            "the episode produced a final answer that was "
                                            "scored incorrect")}}


# The official substrings the capacity registry spec matches. Used to accumulate the same
# episode-scoped fact the LIVE hook does: a store that reported a limit at an earlier step is still
# at that limit when the next call is proposed. The offline projection must agree with the live hook
# field-for-field, or a predicate validates here and fires zero times at runtime.
_CAPACITY_MARKERS = ("is full", "exceeds maximum size", "at its entry limit", "no capacity")

# A result that REFUSED the write. Deliberately broad: the cost of missing a refusal is counting an
# uncommitted fact as stored, which overstates redundancy and risks suppressing a real write.
_ERRISH_RE = re.compile(r"error|cannot|unable|must be|full|exceed|invalid|unique", re.I)



def _is_read(call: str) -> bool:
    """Is this call a read? Asks the ADAPTER's vocabulary, never a local guess.

    Falls back to nothing: if the adapter cannot classify a call, the fact is absent rather than
    invented, which keeps "not a read" distinguishable from "unclassifiable".
    """
    try:
        import adapter as _v
        return bool(_v.is_read(call))
    except Exception:
        return False


def _is_write(call: str) -> bool:
    """Is this call a write? The ADAPTER's vocabulary, same contract as `_is_read`."""
    try:
        import adapter as _v
        return bool(_v.is_write(call))
    except Exception:
        return False


def _container_of(call: str):
    """Which store this call targets, or None when the adapter cannot say."""
    try:
        import adapter as _v
        return _v.container_of(call)
    except Exception:
        return None


def _backend_facts_for(case_id: str) -> dict:
    """Store identity + structure for one case, or {} when the adapter cannot say.

    Both halves go through the adapter -- `backend_of` to classify the case, `backend_facts` to derive
    the structural consequences from the declared constraints -- so this script holds no per-backend
    knowledge of its own. An unresolvable case yields {} and the fields are absent for that episode,
    which is the honest answer and is what keeps a predicate over them from firing on a guess.
    """
    try:
        import adapter as _v
        import bfcl_runtime as _rt
        return dict(_rt.backend_facts(_v.backend_of(str(case_id))) or {})
    except Exception:
        return {}


def observable_states(steps_by_case) -> list[dict]:
    """States as the runtime's own signals read them, with episode-level facts accumulated."""
    out = []
    for cid in sorted(steps_by_case):
        # THE STORE THIS EPISODE RUNS AGAINST, plus the structural consequences a policy depends on.
        #
        # Episode-static, so it is assigned once here rather than recomputed per step, and it leaks
        # nothing about any individual decision. Derived from the case id through the adapter
        # (`backend_of` -> `backend_facts`), which is the same path the live hook uses -- and verified
        # unambiguous: across the frozen-H0 corpus the trajectory's own `backend` field, the adapter's
        # `backend_of(case_id)` and the shard directory agree on 387/387 files with zero mismatches.
        #
        # `scenario_of` is deliberately NOT read. The case id carries the scenario too, so supplying
        # the id itself -- or the scenario -- would let a predicate memorise a favourable use case
        # instead of stating a structural premise.
        _bfacts = dict(_backend_facts_for(cid))
        seen = set()
        capacity_seen = False
        prior_error = None
        # (container, key) -> last normalised value written in THIS episode.
        store_seen: dict = {}
        # calls dispatched in EARLIER steps of this episode (see tool_calls_so_far below).
        calls_so_far = 0
        # normalised call text -> number of times the store REFUSED it, this episode.
        refused_sigs: dict = {}
        for st in steps_by_case[cid]:
            state = dict(st)
            # ALWAYS PRESENT, like `container_full` below: a key some states lack lets a predicate over
            # it silently answer False, which is the defect class this assembler exists to avoid. An
            # unknown backend yields {} from the adapter and the keys are simply absent for the whole
            # episode -- absent for every state of that episode, never for some of them.
            state.update(_bfacts)
            # `turn` is already in `st` for this corpus (2443/2443 real states carry it) -- but a
            # trajectory that omits it must not default to 0, which would assert "the first turn" and
            # feed a false value into the quantiles the grammar derives its thresholds from. Absent
            # stays absent, exactly as `proposed_payload_chars` is absent rather than 0 on a zero-call
            # row; the key is only normalised to int where it is genuinely present.
            if st.get("turn") is not None:
                try:
                    state["turn"] = int(st["turn"])
                except (TypeError, ValueError):
                    state.pop("turn", None)
            # EPISODE-SCOPED, like `searched_other_container` below, and STRICTLY BACKWARD-LOOKING.
            #
            # `container_full` is read at the COMMITMENT GATE, which is pre-dispatch: this step's own
            # tool results do not exist yet there. So the value carried on step N is what was known
            # after step N-1, and this step's results are folded in only AFTER the state is emitted.
            #
            # Getting this wrong leaks future information into a pre-execution predicate: the first
            # version of this loop read st["tool_results"] before emitting, so a step whose OWN write
            # returned "core memory is full" already carried container_full=True -- a condition the
            # live hook could not have known when it decided. That inflates firing and, worse, makes
            # an offline-validated predicate unreproducible at runtime.
            #
            # This mirrors the live hook exactly: there, `execution_results` at the gate holds the
            # PREVIOUS step's results (self._execute assigns the current step's results later, at
            # memory_evaluator.py:4653, well after the gate at ~4575).
            state["container_full"] = capacity_seen

            # ---- error_kind: THE FIELD 8 OF 9 DECLARED SIGNALS READ AND NOTHING SUPPLIED ----------
            #
            # `unsupplied_signal_fields` found it absent from EVERY projected state at EVERY boundary
            # in all three backends, so append_would_exceed_cap, container_at_capacity,
            # container_slots_exhausted, duplicate_identifier, identifier_not_found,
            # no_informative_result and clear_proposed_at_capacity could only ever answer False --
            # offline, where the search decides what to build. The adapter has classified it all along
            # (`adapter.error_kind`); the state assembler simply never wrote it down. It is abundant in
            # the raw traces: 218 no_capacity in kv, 121 entry_too_long in vector, 278
            # blob_would_overflow in rec_sum.
            #
            # TWO FIELDS, because the two boundaries know different things, and conflating them is how
            # a predicate validates offline and fires zero times at runtime:
            #
            #   error_kind        THIS step's own result. Valid at post_execution ONLY, where the
            #                     result exists. `states_at` returns post-execution states unchanged,
            #                     so this is the field those signals read there.
            #   prior_error_kind  the LAST error seen BEFORE this step, for the commitment gate, where
            #                     this step's result does not exist yet. Same backward-looking rule as
            #                     `container_full`, and assigned before the fold-in below.
            #
            # Absent-means-clean is deliberate: `error_kind` returns None for a clean result, and a
            # key whose value is None is still PRESENT, so the supply audit sees it and synthesis is
            # not offered a field that only some states carry.
            state["prior_error_kind"] = prior_error
            _this_error = None
            for r in (st.get("tool_results") or []):
                try:
                    import adapter as _v
                    k = _v.error_kind(str(r))
                except Exception:
                    k = None
                if k and _this_error is None:
                    _this_error = k
            state["error_kind"] = _this_error

            for r in (st.get("tool_results") or []):
                low = str(r).lower()
                if any(mk in low for mk in _CAPACITY_MARKERS):
                    capacity_seen = True
            if _this_error:
                prior_error = _this_error
            for r in (st.get("tool_results") or []):
                t = str(r)
                if "similarity_score" in t:
                    try:
                        vals = [float(i["similarity_score"])
                                for i in json.loads(t).get("result", [])]
                        if vals:
                            state["best_similarity"] = max(vals)
                    except Exception:
                        pass
            for c in (st.get("decoded") or []):
                try:
                    import adapter as _v
                    cont = _v.container_of(str(c))
                except Exception:
                    cont = None
                if cont:
                    seen.add(cont)
            state["searched_other_container"] = len(seen) > 1

            # ---- tool_calls_so_far: THE THIRD FIELD `no_tool_call_at_all` READS, SUPPLIED BY NOTHING -
            #
            # Same defect class as `error_kind` above, and as `proposes_read` below: declared, read by
            # a predicate, and written by no one. MEASURED on the frozen H0: the field is None in ALL
            # 620 query-phase step records across the three backends. `states_at` then coerces it with
            # `int(st.get("tool_calls_so_far") or 0)`, so the predicate's third clause `== 0` is
            # ALWAYS TRUE.
            #
            # The consequence is worse than inertness -- it is SATURATION. `no_tool_call_at_all` fired
            # on 105/105 kv, 89/89 vector and 108/109 rec_sum query episodes: coverage 1.0, with
            # precision exactly equal to the base failure rate (0.8286 vs 0.8286 on kv) and NO
            # non-firing stratum at all. A signal that fires everywhere cannot support a controller,
            # and "discriminates" becomes indistinguishable from "unevaluable".
            #
            # EPISODE-SCOPED and STRICTLY BACKWARD-LOOKING, like `container_full` above: the value
            # carried on step N counts the calls dispatched in steps 0..N-1, so a no-call step that
            # follows two retrievals is distinguishable from one that opens the episode. This step's
            # own calls are folded in only AFTER the state is emitted, which is what keeps the
            # commitment gate free of its own future.
            state["tool_calls_so_far"] = calls_so_far
            # DID THIS STEP PROPOSE A READ? Derived from the step's OWN decoded calls, which exist by
            # the time a post-execution state is built -- so this is backward-looking, not a
            # pre-dispatch fact leaking forward.
            #
            # `retrieval_similarity_below_threshold` requires it, and it was neither declared nor
            # supplied at post_execution: measured 0/53 on real query states while `best_similarity`
            # was present in 22. The signal was structurally unsatisfiable at the only boundary it is
            # declared on, so every read-side family -- the 9-to-11-support ones, with a WORKING
            # executor -- produced arms that could never fire.
            state["proposes_read"] = any(
                _is_read(str(c)) for c in (st.get("decoded") or []))

            # ---- proposal_is_redundant: RECONSTRUCTED per (destination store, key) ----------------
            #
            # `redundant_proposed_write` reads this. Offline there is no live store object, so the
            # store is reconstructed from the episode's OWN earlier writes -- the same information the
            # live comparator reads, arrived at differently, which is why the two must be checked
            # against each other rather than assumed equal.
            #
            # KEYED BY (container, key), because a key present in core says nothing about archival.
            # Taking one flat namespace was the defect the live path had: a core duplicate judged
            # against archival reads "absent -> not a duplicate".
            #
            # The comparison is the comparator's rule, not a looser one: normalised-identical value
            # under the same key. A same-key/DIFFERENT-value write is an UPDATE and answers False, so
            # the store is then updated to the new value -- otherwise a later exact repeat of the
            # UPDATED value would be missed.
            # PROJECTION BLINDNESS, measured and now named at the site that causes it.
            #
            # `store_seen` is built from THIS EPISODE'S own earlier writes. It cannot see the store the
            # PREREQUISITE PHASE already built, so a key genuinely absent from the live store can look
            # present here. Measured on r4_redun_arm (job 1840498): this reconstruction projected
            # 201/590 redundant writes while the LIVE comparator, reading the real store, found ONE --
            # a 201x overstatement. 312 of its 399 invocations answered "key not present in the
            # store".
            #
            # The comment below already records the same class of error one level shallower (PROPOSED
            # vs COMMITTED writes, ~2x). Both have the same shape: the projection is blind to state it
            # does not hold, and the live comparator is the authority.
            #
            # This number is therefore an UPPER BOUND on firings, never measured support. It is left
            # as-is rather than "corrected", because there is no offline way to see prerequisite
            # state -- the honest fix is to label it, which `arm_manifest` now does.
            _red = False
            store_before_step = dict(store_seen)
            refused_before_step = dict(refused_sigs)
            for c in (st.get("decoded") or []):
                cs = str(c)
                if not _is_write(cs):
                    continue
                _k = re.search(r"key\s*=\s*'([^']*)'", cs) or re.search(r'key\s*=\s*"([^"]*)"', cs)
                if not _k:
                    continue          # text-only backend: no identifier, nothing to compare
                _v = (re.search(r"value\s*=\s*'([^']*)'", cs)
                      or re.search(r'value\s*=\s*"([^"]*)"', cs))
                _cont = _container_of(cs) or "unnamed"
                _key = (_cont, _k.group(1))
                _val = " ".join((_v.group(1) if _v else "").split())
                if _key in store_seen and store_seen[_key] == _val and _val:
                    _red = True
                # COMMITTED, NOT MERELY PROPOSED -- and the distinction is the whole measure.
                #
                # This tracked every PROPOSED write, so a repeat of a write the store REFUSED counted
                # as redundant. Measured live on kv: of 396 exact repeats of an earlier proposal, only
                # 199 repeat a COMMITTED write; the other 197 follow a refusal, so the store never held
                # the fact and suppressing would LOSE it. My offline "408 exact repeats" therefore
                # overstated the redundancy by ~2x as a suppression target -- the live comparator,
                # which reads the real store, was right and this projection was wrong.
                #
                # A call whose own result is an error is NOT folded into the store.
                _res_i = (st.get("tool_results") or [])
                _idx = (st.get("decoded") or []).index(c) if c in (st.get("decoded") or []) else -1
                _r_txt = str(_res_i[_idx]) if 0 <= _idx < len(_res_i) else ""
                if not (_r_txt and _ERRISH_RE.search(_r_txt)):
                    store_seen[_key] = _val
            state["proposal_is_redundant"] = _red
            # ---- proposal_refused_count: the 197-case population A3's negative identified ---------
            #
            # The teacher asked repeatedly to "never emit a write whose arguments are byte-identical to
            # one that has already failed in this episode". That is a DIFFERENT population from the
            # committed-duplicate one: on kv prereq, 199 exact repeats follow a COMMITTED write and 197
            # follow a REFUSED write. Only the first had a signal, and suppressing it measured net -3.
            #
            # Strictly backward-looking: the count is of refusals recorded BEFORE this step, so it
            # carries nothing the live gate could not know. Keyed on the normalised CALL TEXT, because
            # "the same call" is what the repair is about -- not the same key, which would conflate an
            # update with a resubmission.
            state["refused_signatures"] = dict(refused_before_step)
            # The store as it stood BEFORE this step's calls, so the per-call projection can judge
            # each call individually rather than inheriting a step-level OR.
            state["store_before"] = dict(store_before_step)

            # ---- the other named supply gaps, all POST-EXECUTION and all adapter-classified -------
            #
            # `unsupplied_signal_fields` named these; each is read by a declared signal and was
            # supplied by nothing, so the signal could only answer False. Same fix as `error_kind`:
            # the adapter already knows how to classify, the assembler just never wrote it down.
            #
            # These are step-OWN facts and therefore valid only where the step's result exists.
            # `states_at` rebuilds the commitment gate from proposed calls and does not carry them
            # forward, so supplying them here cannot leak a post-execution fact into a pre-dispatch
            # information set -- the guards that pin that are still green.
            #
            # proposes_write: read by container_at_capacity and duplicate_identifier. Classified by
            # the adapter's own `is_write`, never a verb match -- ANCHORS.md records that keying a
            # remedy on the call kind proposed a suppression inspection showed benign in 15 of 17.
            state["proposes_write"] = any(
                _is_write(str(c)) for c in (st.get("decoded") or []))
            # result: read by no_informative_result, which delegates to the EXISTING
            # `vacuous_result_kind` detector. The raw payload text is what that classifier takes.
            _res = [str(r) for r in (st.get("tool_results") or [])]
            state["result"] = _res[0] if _res else None
            # container: read by container_slots_exhausted, to tell WHICH store refused. Absent when
            # the adapter cannot classify the call, so "unclassifiable" stays distinguishable from
            # "not archival" rather than being silently reported as the latter.
            _conts = [c for c in (_container_of(str(x)) for x in (st.get("decoded") or [])) if c]
            state["container"] = _conts[0] if _conts else None
            # Record this step's refusals for the NEXT step to read. After emission, so the state a
            # predicate sees never includes the outcome of the call it is judging.
            _rs = [str(r) for r in (st.get("tool_results") or [])]
            for _i, _c in enumerate([str(x) for x in (st.get("decoded") or [])]):
                _rt = _rs[_i] if _i < len(_rs) else ""
                if _rt and _ERRISH_RE.search(_rt):
                    _sig = " ".join(_c.split())
                    refused_sigs[_sig] = refused_sigs.get(_sig, 0) + 1
            out.append(state)
            # AFTER EMISSION, for the same reason `refused_sigs` is updated here: the state a
            # predicate sees must not include the outcome of the step it is judging. Counting this
            # step's own dispatched calls now makes the value on step N+1 mean 'calls made earlier'.
            calls_so_far += len(st.get("decoded") or [])
    return out


def write_teacher_usage(path: pathlib.Path, calls, *, stage: str, elapsed_s: float) -> dict:
    """Persist what the meta-model cost, from the CallRecords it already carries.

    `client._call` puts the provider's own usage on every record; nothing wrote it to disk, so a
    round could not say what its attribution cost. Appended (not overwritten) so a round with
    several teacher stages accumulates rather than the last one winning.

    A usage field the provider did not return stays None and is summed as absent: `*_tokens` is
    None when NO call reported it, rather than 0, because an unmeasured cost is not a free one.
    """
    rows = []
    for c in calls:
        u = dict(c.usage or {})
        rows.append({"stage": stage, "role": c.role, "model": c.model,
                     "meta_provider": c.meta_provider, "meta_base_url": c.meta_base_url or None,
                     "prompt_chars": c.prompt_chars,
                     "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
                     "error": c.error or None})
    def _sum(key):
        got = [r[key] for r in rows if r[key] is not None]
        return sum(got) if got else None
    rec = {"stage": stage, "calls": len(rows), "errors": sum(1 for r in rows if r["error"]),
           "input_tokens": _sum("input_tokens"), "output_tokens": _sum("output_tokens"),
           "elapsed_s": round(elapsed_s, 2),
           "model": rows[0]["model"] if rows else None,
           "meta_provider": rows[0]["meta_provider"] if rows else None,
           "calls_detail": rows}
    existing = []
    if path.exists():
        try:
            existing = json.load(open(path))
        except (ValueError, OSError):
            existing = []
    if not isinstance(existing, list):
        existing = [existing]
    existing.append(rec)
    json.dump(existing, open(path, "w"), indent=2, default=str)
    return rec


def attribute(cases, out_dir: pathlib.Path, live: bool):
    path = out_dir / "diagnoses.json"
    if live:
        import client as roles
        t0 = time.time()
        raw, calls = roles.attribute_batched(cases)
        elapsed = time.time() - t0
        errs = [c.error for c in calls if c.error]
        print(f"  {len(raw)} diagnoses / {len(cases)} cases, {len(calls)} calls, {len(errs)} errors")
        json.dump(list(raw), open(path, "w"), indent=2, default=str)
        cost = write_teacher_usage(out_dir / "teacher_usage.json", calls,
                                  stage="attribution", elapsed_s=elapsed)
        print(f"  teacher: {cost['model']} via {cost['meta_provider']}  "
              f"in={cost['input_tokens']} out={cost['output_tokens']} "
              f"elapsed={cost['elapsed_s']}s")
    else:
        raw = json.load(open(path)) if path.exists() else []
        print(f"  cached: {len(raw)}")
    diags = []
    for r in raw:
        try:
            diags.append(ingest_attribution(r, provider="claude-attributor-v1"))
        except AttributionSchemaError as exc:
            print(f"    REJECTED by schema: {exc}")
    return diags


def _has_tool_error(steps) -> bool:
    """Did this episode produce a tool error? Classified by the ADAPTER, never by prose.

    `error_kind` returns None for a clean result, so an unfamiliar error text simply does not match
    rather than being guessed at.
    """
    try:
        import adapter as _v
    except Exception:
        return False
    for st in steps or []:
        for r in (st.get("tool_results") or []):
            if _v.error_kind(str(r)):
                return True
    return False


def _load_prereq(run: pathlib.Path) -> dict:
    """Setup-episode trajectories for one run, if it kept any.

    Read so contamination is MEASURED per round rather than assumed absent -- the guard is new, and a
    guard believed rather than checked is how this defect survived a full round.
    """
    out = {}
    for sub in (run / "traj" / "prereq", run / "run" / "traj" / "prereq"):
        if not sub.exists():
            continue
        for fp in sorted(sub.glob("*.json")):
            try:
                ep = json.load(open(fp))
            except Exception:
                continue
            out[str(ep.get("case_id"))] = ep.get("steps") or []
    return out


def _resolved_store_identity(run: pathlib.Path, prov: dict) -> str | None:
    """The identity of the STATE this run's scored units were graded against, or None.

    Names the store, never the runner. A run that LOADED a store built nothing, and a run that
    BUILT one published a variant; if the two name one store instance, the scored units were graded
    against the same state however it got there. Encoding "who built it" instead -- the first
    version of this did -- makes a correctly shared store report as unmatched construction purely
    because one arm happened to run the builder, which is the artifact this whole protocol exists
    to remove.

    A MISS emits provenance BEFORE the build, because the host cannot know its variant_key until
    the storage triggers have fired (memory_evaluator.py: "variant_key is only knowable AFTER the
    build"). The resolved key is persisted afterwards INSIDE the published variant, so the store on
    disk is the authoritative record and the pre-build log line is not.
    """
    cache, base = prov.get("cache_dir"), prov.get("base_key")
    if not cache or not base:
        return None
    var = prov.get("source_variant") if str(prov.get("verdict")) == "HIT" else None
    if var is None:
        var = prov.get("variant_key")
    if var is None:
        # Read it back from the store the host published. Only one variant may match this run's
        # recorded prereq population, so an ambiguous directory resolves to nothing rather than to
        # a guess.
        base_dir = pathlib.Path(cache) / f"snap_{base}"
        hits = []
        for cp in sorted(base_dir.glob("v_*/cache_provenance.json")):
            try:
                d = json.load(open(cp))
            except Exception:
                continue
            if d.get("prereq_ids_sha256") == prov.get("prereq_ids_sha256"):
                hits.append(cp.parent.name)
        if len(hits) == 1:
            var = hits[0]
    if var is None:
        return None
    var = str(var)
    return f"store:{cache}/{base}/{var if var.startswith('v_') else 'v_' + var}"


def _arm_conditions(run: pathlib.Path, n_prereq_firings: int) -> dict:
    """The initial state one arm started from, and how it was built -- MEASURED, not asserted.

    `initial_state_fingerprint` is the store identity the run actually loaded, read back from the
    host's own provenance record. An earlier version of this call asserted "empty" for both arms,
    which makes `matched_initial_conditions` compare a constant against itself and report a match it
    never checked -- the same shape of error as reading absent telemetry as a zero.

    `construction_protocol` is the RESOLVED STORE IDENTITY, so two arms reading one store report
    matched construction and two arms that each built their own do not. When the store cannot be
    resolved it falls back to naming what the controller did during the build, which fails CLOSED:
    an unverifiable construction is left unproven rather than passing for a matched one.

    When provenance is unavailable the fingerprint is the run's own name, so two arms NEVER
    accidentally compare equal: an unknown initial state must not pass for a matched one.

    THE FINGERPRINT NAMES THE STATE, NOT WHO GOT THERE FIRST. It used to read
    `base_key/source_variant or 'built-here'`, so the arm that LOADED a shared store and the control
    that had just BUILT that same store reported different initial states -- `v_f74e49da8dfe` vs
    `built-here` -- for one identical state. That is the same defect already fixed for
    `construction_protocol`, left behind in the sibling field: it encodes the runner's history rather
    than the state, and it made a correctly shared store report as an unmatched start. Both fields
    now resolve through `_resolved_store_identity`, so "the state this run's scored units were graded
    against" has ONE answer per store however the run arrived at it.
    """
    fp = f"unknown:{run.name}"
    protocol = f"controller_fired_on_{n_prereq_firings}"
    prov = _store_provenance(run)
    if prov:
        ident = _resolved_store_identity(run, prov)
        if ident is not None:
            # One resolved store => one initial state AND one construction. Both arms naming this
            # store agree on both, which is exactly what a shared-construction protocol buys.
            fp = ident
            protocol = ident
        else:
            # Fail CLOSED on both, and keep them RUN-UNIQUE: an unresolvable store must never let
            # two arms compare equal on either question.
            fp = f"UNRESOLVED:{prov.get('base_key')}:{run.name}"
            protocol = (f"UNRESOLVED:{prov.get('cache_dir')}/{prov.get('base_key')}"
                        f":fired_on_{n_prereq_firings}")
    return {"initial_state_fingerprint": fp, "construction_protocol": protocol}


def _store_provenance(run: pathlib.Path) -> dict:
    """The host's cache-provenance record for one run: from a sidecar file, else from its log.

    The host PRINTS this record and does not persist it, so the log is the only place it exists. A
    fragile read, and preferable to the alternative: without it both arms fall back to a name-derived
    fingerprint, which fails CLOSED -- two arms never compare equal, so a matched initial state can
    never be asserted without evidence, only left unproven.
    """
    for cand in (run / "run" / "store_provenance.json", run / "store_provenance.json"):
        if cand.exists():
            try:
                return json.load(open(cand))
            except Exception:
                return {}
    logs = sorted((run.parent.parent / "logs").glob(f"{run.name}.*.log")) if run.name else []
    for lg in reversed(logs):  # newest first: a rerunnable job appends, and the last run is the one
        try:
            for line in reversed(lg.read_text(errors="replace").splitlines()):
                mark = "[snapshot_store][provenance] {"
                if mark in line:
                    return json.loads(line[line.index(mark) + len(mark) - 1:])
        except Exception:
            continue
    return {}


def _phases_executed(run: pathlib.Path, scored_units: int) -> frozenset[str]:
    """Which phases this run ACTUALLY executed, from the host's own record -- or nothing.

    Returning an empty set means "this host did not say", and the core then makes no opportunity
    claim in either direction. That default matters more than the positive cases: the absence of
    prereq TRAJECTORIES is NOT evidence the prereq phase was skipped, because a run may simply not
    retain them. Only the host's own `prereqs_skipped` count, checked against `n_prereqs`, says
    affirmatively that the phase did not run -- which is what a store HIT does, since reusing a
    constructed store is precisely the decision not to construct it again.

    So this reads the decision, not its shadow. A MISS built the store, so prereq ran.
    """
    prov = _store_provenance(run)
    if not prov:
        return frozenset()
    phases = {"query"} if scored_units else set()
    n_pre = prov.get("n_prereqs")
    skipped = prov.get("prereqs_skipped")
    if not isinstance(n_pre, int) or n_pre <= 0:
        # The host reported no prereq population, so it says nothing about that phase either way.
        return frozenset(phases) if phases else frozenset()
    if isinstance(skipped, int) and skipped >= n_pre:
        pass                      # every setup episode reused: the prereq phase did not execute
    else:
        phases.add("prereq")      # some or all were built here
    return frozenset(phases)

def _phase_telemetry(run: pathlib.Path, phase: str) -> dict:
    """Mechanism/safety counters aggregated over ONE phase's episodes of one run.

    Adapter-side on purpose: it knows this host's sidecar layout and key names, which the core must
    not. What it returns is the shape `check_mechanism`/`check_safety` already consume, so routing
    verification to the acting phase is a change of INPUT, not of criterion.

    Returns {} when the phase kept no episodes -- so a controller that acted where nothing was
    instrumented yields PENDING_VALIDATION rather than a fabricated zero. Do not "helpfully" default
    a missing counter to 0: that is how absent evidence became a pass before.
    """
    eps = _load_prereq(run) if phase == "prereq" else load_run(run)[1]
    if not eps:
        return {}
    keymap = {"controller_fired_gate": "signal_firings",
              "relocate_requested_gate": "mechanism_requested",
              "relocate_write_verified": "mechanism_verified",
              "relocate_declined": "mechanism_declined_with_reason",
              "reloc_bound_refused": "mechanism_refused_by_guard",
              "clears_added": "clears_added",
              "information_losing_removes": "information_losing_removes"}
    out: dict[str, int] = {}
    executed = 0
    for steps in eps.values():
        for st in steps:
            if not isinstance(st, dict):
                continue
            for k, v in st.items():
                if k in keymap and v:
                    out[keymap[k]] = out.get(keymap[k], 0) + 1
            if any(k.startswith("relocate_") and k.endswith("_gate") and v for k, v in st.items()):
                executed += 1
    if not out:
        return {}
    out["interventions_executed"] = executed
    out.setdefault("verified_relocations", out.get("mechanism_verified", 0))
    return out


_SPEC_CANDIDATES = ("controller_spec.json", "run/controller_spec.json", "spec.json")


def _installed_spec(run: pathlib.Path) -> dict:
    """The FULL spec the run recorded, or {}. Same lookup as `_installed_phase`, deliberately shared.

    Two readers of "what was installed" that disagree about which file is authoritative is a defect
    this project has already paid for, so the path list lives in one place.

    An empty dict means the run kept no spec. The caller must then NOT invent an identity: a record
    that names a mechanism it cannot re-execute is worse than no record, because later rounds would
    deduplicate against it.
    """
    for rel in _SPEC_CANDIDATES:
        cand = run / rel
        if cand.exists():
            try:
                d = json.load(open(cand))
                return dict(d) if isinstance(d, dict) else {}
            except Exception:
                return {}
    return {}


def _installed_phase(run: pathlib.Path) -> str | None:
    """The phase the installed controller DECLARED, read back from the run's own recorded spec.

    Read from the run rather than passed in, so the comparison is against what actually executed and
    not against what a caller believes was installed. None when the run kept no spec: an undeclared
    phase must not be invented, because "it ran where it said" and "it never said" are different
    claims and only the first is evidence.
    """
    for cand in (run / "controller_spec.json", run / "run" / "controller_spec.json",
                 run / "spec.json"):
        if cand.exists():
            try:
                return str(json.load(open(cand)).get("phase") or "") or None
            except Exception:
                return None
    return None


def states_at_boundary(steps_by_case, locus: str, *, runtime) -> list:
    """States as THAT boundary will see them -- reconstructed from pre-boundary information only.

    The original defect had two halves. The first (fields) is fixed in the adapter: `synthesis_fields`
    now offers only what the hook supplies. This is the second half: a predicate was also VALIDATED on
    post-execution-shaped states regardless of which boundary was being searched, so an upstream
    predicate could look discriminating on information that does not exist yet.

    For the commitment gate, one state per PROPOSED CALL, carrying only what is knowable before it
    runs: the call text, its index, how many calls were proposed together, the step, the phase. The
    step's own results are deliberately NOT included -- they are the consequence of the decision being
    judged, and including them is how a pre-execution predicate gets validated on post-execution facts.

    For post-execution, the existing per-step assembly is correct and is reused unchanged, so A9's
    validation is untouched.
    """
    want = str(getattr(locus, "value", locus))
    supplied = set(runtime.hook_state_fields(want)) if hasattr(runtime, "hook_state_fields") else set()
    if want != "post_generation_pre_exec":
        return observable_states(steps_by_case)

    out = []
    for cid in sorted(steps_by_case):
        for idx, st in enumerate(steps_by_case[cid] or []):
            calls = [str(x) for x in (st.get("decoded") or [])]
            for i, call in enumerate(calls):
                state = {"proposed_call": call, "call_index": i,
                         "n_proposed_calls": len(calls),
                         "step_index": st.get("step", idx),
                         "phase": "prereq" if "prereq" in cid else "query",
                         "boundary": want}
                # Typed facts DERIVED from the proposed call, via the adapter's own vocabulary. These
                # are pre-dispatch by construction -- they read the call, never its result.
                extra = getattr(runtime, "proposed_call_facts", None)
                if callable(extra):
                    try:
                        state.update(extra(call) or {})
                    except Exception:
                        pass
                # Keep only what the hook will actually supply, plus `boundary` which core reads.
                if supplied:
                    state = {k: v for k, v in state.items()
                             if k in supplied or k == "boundary"}
                out.append(state)
    return out


def boundary_state_shape(locus: str, *, runtime) -> set:
    """Which field names a controller installed at `locus` will actually BE GIVEN at runtime.

    NOT `synthesis_fields(locus)`. That is what the adapter DECLARES readable there; this is what the
    host's hook actually populates. The two differ, and the gap is not academic: WRITE1's search
    selected a controller at the commitment gate whose phi read `searched_other_container` -- declared
    at that boundary, supplied by the hook only post-execution -- so it would have fired 0 times and
    the round would have measured nothing. Declared-but-unpopulated is the same defect as a signal
    declared observable where its fact does not exist.

    Asked of the runtime so no benchmark shape is written here; an adapter that cannot answer yields
    an empty set and the caller skips the check rather than inventing a shape.
    """
    fn = getattr(runtime, "hook_state_fields", None)
    if callable(fn):
        try:
            return set(fn(locus) or ())
        except Exception:
            return set()
    return set()


def can_fire_at_boundary(arm, predicate, *, runtime, states) -> tuple[bool, str]:
    """Could this controller fire AT ITS OWN BOUNDARY, on that boundary's state shape?

    Two checks, and the first is the one WRITE1 needed:
      * every field phi reads must be one the boundary's hook actually supplies;
      * phi must fire on a non-trivial minority of the residual's states -- a predicate that fires
        never cannot be measured, and one that fires always is not a trigger.

    The states available for the second check are post-execution-shaped, so it is a necessary
    condition, not a sufficient one. That is why the field check comes first and is decisive.
    """
    locus = arm.boundary.value
    shape = boundary_state_shape(locus, runtime=runtime)
    reads = fields_read(predicate)
    if shape and reads and not reads <= shape:
        missing = sorted(reads - shape)
        return False, (f"phi reads {missing} which the {locus} hook does not supply "
                       f"(it supplies {sorted(shape)}) -- the controller would fire 0 times")
    fv = firing_vector(predicate, states)
    n = sum(fv)
    if n == 0:
        return False, "phi fires on none of the residual's observed states"
    if n == len(fv):
        # AN UNCONDITIONAL POLICY IS A LEGITIMATE POLICY, and it is a different thing from a predicate
        # that reads a field the boundary cannot supply. "Always act here" is a real intervention --
        # it is what a reprompt-on-every-proposed-write would be -- and forbidding it would rule out
        # the simplest controller in the space. What must never pass is a predicate that CANNOT be
        # evaluated, which the field check above already catches.
        #
        # It is still reported, because an unconditional controller and a trivially-true one look the
        # same in a candidate list and only the first is intended.
        if not reads:
            return True, (f"UNCONDITIONAL: no field read, fires on all {len(fv)} states -- a "
                          f"deliberate always-act policy, not a trigger")
        return True, (f"fires on every one of {len(fv)} observed states while reading "
                      f"{sorted(reads)} -- effectively unconditional on this residual; intended only "
                      f"if an always-act policy is what was wanted")
    return True, f"fires on {n}/{len(fv)} observed states; reads {sorted(reads) or 'n/a'}"


def fields_read(predicate) -> set:
    """Field names a synthesized predicate reads. Structural, from the grammar objects."""
    terms = getattr(predicate, "terms", None)
    if terms:
        return {t.field for t in terms if getattr(t, "field", None)}
    f = getattr(predicate, "field", None)
    return {f} if f else set()


def _firing_pool_key(problem, *, runtime, states_by_case) -> tuple:
    """R15: key a family on WHICH DECLARED SIGNALS ACTUALLY FIRE on its cases' own states.

    WHY THIS EXISTS, measured. The lexical key below uses `_signal_matches`, and R14 measured that
    matcher demanding a new signal on 1 of 4 historical anchors. R15 measured the SAME matcher
    inside POOLING and found it worse here than at the gate: `no_tool_call_at_all` has tokens
    ['tool', 'call'], BOTH in `_WEAK_TOKENS` and NO declared alias, so it can only match via the
    `len(matched) >= 2` escape hatch -- which any diagnosis of a tool-using agent satisfies. It
    therefore tagged 24 of 24 diagnoses in BOTH cells, carrying zero information, and the rank-1
    family became a MIXTURE of the zero-call population and the weak-retrieval population. The
    search acts on rank 1 only, so the zero-call condition was never built for the cases it explains.

    Keyed on firing instead, the same 24 vector diagnoses separate 12 / 10 / 2. This is the same
    substitution R14 specified for the gate -- compare declared OBSERVABLES, not names against prose
    -- applied at the place R15 measured it binding.

    OFF BY DEFAULT: V1 is the frozen comparator for V2, and `_signal_matches` is on every
    workstream's search path.
    """
    sigs = set()
    for cid in problem.case_ids:
        for sig in runtime.declared_signals():
            if sig in sigs:
                continue
            fn = runtime.SIGNALS.get(sig) or runtime.EXPANDED_SIGNALS.get(sig)
            if fn is None:
                continue
            for b in runtime.signal_boundaries(sig):
                for st in runtime.states_at(b, states_by_case.get(cid) or ()):
                    try:
                        if fn(st, dict(SIGNAL_PROBE_PARAMS_R15.get(sig, {}))):
                            sigs.add(sig)
                            break
                    except Exception:
                        pass
                if sig in sigs:
                    break
    return tuple(sorted(sigs))


#: Probe params for the R15 firing key ONLY. A parameterized signal cannot be evaluated without one,
#: and a signal left unparameterized here simply never joins a key -- it is not silently treated as
#: firing. Sourced from the grammar's own synthesized threshold on this corpus, not hand-tuned.
SIGNAL_PROBE_PARAMS_R15 = {"retrieval_similarity_below_threshold": {"below": 0.5}}


def pool_by_observable(problems, *, runtime, states_by_case=None) -> list:
    """Merge residual families that name the SAME OBSERVABLE CONDITION.

    WHY. Families are keyed on normalized attributor prose, which collapses paraphrases but not
    different sentences. Measured on the first cycle-2 propose run: 20 of 24 diagnoses split across
    FIVE families of support 4 --

        R1 accept the first retrieval's top-k hits as sufficient evidence...
        R2 commit to a final answer after the first retrieval pass...
        R3 commit on the basis of a first, weakly-grounded retrieval result...
        R4 stop retrieving and answer when the retrieved evidence is weak...
        R5 treat the first non-empty but weakly-matching retrieval as sufficient...

    -- which is one residual, "answers on a weak first retrieval", written five ways. Every one of
    them was below the declared MIN_SUPPORT floor, so the round could only have returned
    INSUFFICIENT_SUPPORT no matter what a GPU measured.

    THE KEY IS COMPUTED, NOT WRITTEN. It is the set of DECLARED SIGNALS whose coverage check matches
    the family's diagnoses -- the same `_signal_matches` the expressibility test already uses. So this
    pools on what the runtime can OBSERVE about the failure, never on how it was phrased, and it
    cannot invent a grouping the coverage check does not already support.

    Not `group_blocked_by_clause`: that pools by UNCOVERED clause and fires only on the SIGNAL_BLOCKED
    path. Here all 24 diagnoses are expressible and 0 carry an uncovered clause, so it yields nothing.
    """
    from anchoropt.learning.candidate_search import _signal_matches

    def key(problem) -> tuple:
        sigs = set()
        for d in problem.scoped_diagnoses():
            txt = " ".join((d.failure_mechanism or "", d.consequential_decision or "",
                            d.evidence or "")).lower()
            for s in runtime.declared_signals():
                if _signal_matches(s, txt, runtime):
                    sigs.add(s)
        return tuple(sorted(sigs))

    use_firing = bool(states_by_case) and os.environ.get("ANCHOROPT_POOL_KEY") == "firing"
    buckets: dict = {}
    for prob in problems:
        k = (_firing_pool_key(prob, runtime=runtime, states_by_case=states_by_case)
             if use_firing else key(prob))
        buckets.setdefault(k, []).append(prob)

    class Pooled:
        """One residual, presented to the search exactly as a ResidualProblem is."""

        def __init__(self, members, observables):
            self._m = list(members)
            self.observables = observables
            self.rank = min(p.rank for p in self._m)
            self.key = self._m[0].key
            self.expressible = any(p.expressible for p in self._m)
            self.merged_from = tuple(p.key for p in self._m)

        @property
        def support(self) -> int:
            return sum(p.support for p in self._m)

        @property
        def case_ids(self) -> tuple:
            return tuple(dict.fromkeys(c for p in self._m for c in p.case_ids))

        def scoped_diagnoses(self):
            return tuple(d for p in self._m for d in p.scoped_diagnoses())

    pooled = [Pooled(members, obs) for obs, members in buckets.items()]
    pooled.sort(key=lambda x: (-x.support, x.rank))
    for i, x in enumerate(pooled, 1):
        x.rank = i
    return pooled


def controller_spec(arm, predicate, *, phase: str = "") -> dict:
    """The chosen controller as DATA, translated from the predicate OBJECT, not from its name.

    The synthesized signal name is truncated to 60 chars by the expansion path, so parsing it would be
    lossy and would silently drop a conjunct on a long name. `Atom` carries (field, op, value)
    structurally and `Conjunction` carries its terms, so this translation is exact.

    Consumed by scripts/install_controller.py, whose grammar is an atom or a conjunction of atoms --
    which is precisely what the grammar synthesizes. Emitting data rather than a hand-written
    predicate is what keeps a human out of the loop at the point the loop is supposed to own.
    """
    def atom(a) -> dict:
        out = {"field": a.field, "op": a.op}
        if a.value is not None:
            out["value"] = a.value
        return out

    terms = getattr(predicate, "terms", None)
    pred = {"all": [atom(x) for x in terms]} if terms else atom(predicate)
    return {"name": arm.signal, "locus": arm.boundary.value,
            "action": arm.action.value, "operator": arm.instantiated.operator.value,
            "eta": {k: (v if isinstance(v, (int, float, bool, str)) else str(v))
                    for k, v in arm.eta.items()},
            "predicate": pred,
            # THE BINDING, on the SYNTHESIZED path too.
            #
            # `_spec_for`'s declared-signal branch emitted `capability_id` and this builder did not, so
            # every controller from an EXPANDED signal -- which is every arm in a Phi-expansion round --
            # installed without naming the executor core resolved for it. At a cell with one executor
            # that is merely missing provenance. At `post_generation_pre_exec/reprompt`, which has TWO
            # (a live trailing-message injector and a disabled sibling that only writes telemetry),
            # it is the difference between measuring a mechanism and measuring the control.
            "capability_id": getattr(arm, "capability_id", "") or "",
            # THE PHASE THE RESIDUAL WAS MINED FROM. `_phase_eligible` confines a controller to one
            # phase and an UNDECLARED controller defaults to "query" -- so a controller mined from
            # prereq (storage) episodes was installed, reached, and then SKIPPED in the only phase
            # its residual lives in. Measured: 408 upstream_skipped_prereq_gate and 0 firings in
            # prereq, against 85-108 hook reaches but 1-2 firings in query, because query episodes
            # are ~99% reads and a write-side predicate has almost nothing to act on there.
            #
            # The historical WRITE2 arm carried `"phase": "prereq"` HAND-ADDED to its spec, which is
            # exactly the human-in-the-loop step emitting data is supposed to remove. The driver
            # already knows which phase it mined, so it says so.
            "phase": phase or "query",
            "provenance": ("synthesized by anchoropt.learning.structured_search.optimize_residual; "
                           "translated from the predicate object, not parsed from its name; "
                           f"phase={phase or 'query'} is the phase the residual was mined from")}


def dependency_graph(cases_file: pathlib.Path | None) -> dict:
    """case_id -> the cases it depends on, as the CORPUS declares them. {} when unavailable.

    Read from the adapter's own shard file rather than reconstructed from id strings: a naming
    convention is not a dependency, and inferring one would invent provenance the corpus does not
    assert.
    """
    if not cases_file or not cases_file.exists():
        return {}
    try:
        rows = json.load(open(cases_file))
    except (ValueError, OSError):
        return {}
    return {str(c["id"]): [str(d) for d in (c.get("depends_on") or [])]
            for c in rows if isinstance(c, dict) and "id" in c}


def upstream_of(cid: str, graph: dict) -> list:
    """Transitive `depends_on` closure of one case, nearest first. Cycle-safe."""
    order, seen, frontier = [], {cid}, list(graph.get(cid) or [])
    while frontier:
        nxt = []
        for d in frontier:
            if d in seen:
                continue
            seen.add(d)
            order.append(d)
            nxt.extend(graph.get(d) or [])
        frontier = nxt
    return order


def with_upstream(facts: dict, cid: str, graph: dict, steps_by_case: dict, *,
                  max_episodes: int = 3, max_steps: int = 40) -> dict:
    """Attach the trajectories a failed case DEPENDS ON, as upstream evidence.

    WHY THIS EXISTS. A query episode reads a store that an earlier storage episode built. When the
    fact it needed was destroyed during construction, the query's own trajectory shows only a read
    that found nothing -- the consequential decision is not in this episode at all. Attribution over
    one episode can therefore see that a read failed but never that a write destroyed what it wanted,
    so a whole class of consequential decision is unreachable from downstream evidence.

    This is the general provenance mechanism, not a memory-benchmark special case: any host whose
    episodes share state through a declared dependency graph has the same structure, and any host
    that declares no graph simply gets no upstream block.

    LABELLED, NEVER MERGED. The upstream steps go in under their own key with their own case ids, so
    the attributor cannot mistake an upstream write for something this episode did. Bounded, because
    a full closure is 10 episodes of 18 steps here and the point is the consequential decision, not
    the whole history.
    """
    ups = upstream_of(cid, graph)
    if not ups:
        return facts
    blocks = []
    for up in ups[:max_episodes]:
        st = steps_by_case.get(up)
        if not st:
            continue
        blocks.append({
            "case_id": up,
            "relation": "this episode built the memory store the failing case reads",
            "trajectory": [
                {"step": i, "status": x.get("status"),
                 "tool_calls": [str(c) for c in (x.get("decoded") or [])] or None,
                 "tool_results": [str(r)[:300] for r in (x.get("tool_results") or [])] or None}
                for i, x in enumerate(keep_steps(st)[:max_steps])],
        })
    if not blocks:
        return facts
    out = dict(facts)
    out["upstream_episodes"] = blocks
    out["upstream_note"] = (
        "These episodes ran BEFORE the failing one and built the store it reads. A decision made "
        "here can be the cause of the failure below even though it appears nowhere in that "
        "episode's own trajectory. Cite the upstream case_id when you attribute to one.")
    return out




def _probe_params_for(signal: str, runtime) -> dict:
    """Parameters a declared signal needs to be EVALUABLE. Empty for a deterministic one.

    `probe_params` when the runtime offers one; otherwise the midpoint of each declared domain. This
    is not a tuning choice -- it is the value that makes the predicate answerable at all, and the
    optimizer searches the grid around it.
    """
    try:
        p = dict(runtime.probe_params(signal) or {})
        if p:
            return p
    except Exception:
        pass
    out = {}
    try:
        for d in (runtime.parameter_domains(signal) or ()):
            if getattr(d, "values", ()):
                out[d.name] = d.values[len(d.values) // 2]
            else:
                out[d.name] = (float(d.low) + float(d.high)) / 2.0
    except Exception:
        return {}
    return out



def _observed_values(states) -> dict:
    """field -> the numeric values it actually takes across the residual's states.

    What `candidate_thetas` needs to build a data-driven grid. Only numeric fields, because a
    threshold over a boolean is not a threshold, and bools are `int` in Python -- so they are
    excluded explicitly rather than by type alone.
    """
    out: dict = {}
    for st in (states or ()):
        if not isinstance(st, dict):
            continue
        for k, v in st.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            out.setdefault(k, []).append(float(v))
    return out


def _spec_for(arm, synth: dict, phase: str, runtime=None) -> dict:
    """The runnable controller spec for one arm, whatever kind of signal it carries.

    A SYNTHESIZED signal (from Phi expansion) has a grammar object, so the predicate is translated
    structurally. A DECLARED signal has none -- the host ships its implementation -- so the spec names
    it and the runner evaluates it by name. Both need a spec: it is what INSTALLS the controller.
    Emitting one only for the synthesized case left a focused round with no runnable artifact.
    """
    pred_obj = synth.get(arm.signal)
    if pred_obj is not None:
        return controller_spec(arm, pred_obj, phase=phase)
    return {
        "name": arm.signal,
        "locus": arm.boundary.value,
        "action": arm.action.value,
        "operator": arm.instantiated.operator.value,
        "variant": arm.instantiated.variant,
        "eta": {k: (v if isinstance(v, (int, float, bool, str)) else str(v))
                for k, v in arm.eta.items()},
        # A PARAMETERIZED SIGNAL CARRIES ITS PARAMETER. Without it the runner evaluates the signal
        # with empty params, the predicate raises, and "a raising phi is not a firing" makes the
        # controller silently never fire -- an installed arm that runs as the control.
        #
        # The value is the probe midpoint of the declared domain, recorded as `signal_params` so a
        # reader can see what the controller will actually test. The optimizer refines it by
        # measurement; this is the value the FIRST evaluation uses.
        "predicate": _predicate_for(arm, runtime),
        # THE BINDING: the executor core resolved for this arm. Installing a different sibling than
        # the one feasibility accepted would measure a mechanism core never validated.
        "capability_id": getattr(arm, "capability_id", "") or "",
        "phase": phase,
        "provenance": (
            "declared signal, seeded from the residual family's grounded observables by "
            "anchoropt.learning.structured_search.optimize_residual; the host ships the predicate, "
            f"so the runner evaluates it by name. phase={phase}"),
    }

def _applicability_of(arm) -> dict:
    """The APPLICABILITY the host declared for this arm, or {}. Read off the arm, never inferred.

    The marker rides on `instantiated.grounding`, which is where `with_applicability_variants` put it
    and which `action_contract.instantiate` copies verbatim onto the InstantiatedAction. So the
    conditioned arm and its spec cannot disagree: there is one source, the host's own grounding.
    """
    inst = getattr(arm, "instantiated", None)
    ap = dict(getattr(inst, "grounding", {}) or {}).get("applicability") or {}
    if not isinstance(ap, Mapping) or not ap.get("field") or ap.get("value") is None:
        return {}
    return {"field": str(ap["field"]), "value": ap["value"]}


def _predicate_for(arm, runtime=None) -> dict:
    """The installable predicate for a DECLARED-signal arm: the signal, optionally CONDITIONED.

    WHY THE CONDITION IS A SECOND CONJUNCT AND NOT A SCOPE. R19 measured two mechanisms whose sign
    REVERSES across stores (relocate kv +4 / vector -6; reprompt kv 0 / vector +4) and emitted 13
    specs, every one a bare `{"declared_signal": ..., "params": {}}`. Applicability was therefore never
    something measurement could decide -- it lived outside phi, in per-cell arm scoping chosen by the
    experimenter.

    And a variant NAME is not a gate: `relocate_entry_for_kv` carries no store condition, so its
    predicate read only `error_kind`/`proposes_write` and the executor dispatched on the RUNNING store.
    R19's telemetry: that arm executed 32 verified relocations ON VECTOR and lost 7 vector cases. The
    explicit conjunct is the only construct that actually confines it.

        {"all": [{"declared_signal": <the signal the search localized>, "params": {...}},
                 {"field": "backend", "op": "equals", "value": <the store the host declared>}]}

    The unconditional arm is built alongside this one by the same grounding, so a mechanism that
    generalizes keeps its general arm. Nothing here chooses between them -- measurement does.
    """
    base = {"declared_signal": arm.signal,
            "params": dict(_probe_params_for(arm.signal, runtime))}
    ap = _applicability_of(arm)
    if not ap:
        return base
    return {"all": [base, {"field": ap["field"], "op": "equals", "value": ap["value"]}]}


def residual_keys(diags) -> collections.Counter:
    return collections.Counter(
        " ".join(str(d.consequential_decision).lower().split()) for d in diags)


def _score_pair_and_persist(a, rec, stack, *, reason: str) -> None:
    """Score an already-measured pair and record it, on a path that never reached the search.

    WHY THIS EXISTS. `main()` has FIVE early exits before the scoring phase (empty attribution, no
    residual problem, no candidate, none fireable, all settled), and every one of them returned
    without honouring `--score`. So a paired run that had already cost GPU time was silently not
    scored whenever the PROPOSAL half came up empty -- and the two halves read different inputs, so
    one failing says nothing about the other. Measured instance: `rounds/AUTORUN/r1/cycle2.json` has
    `evaluation: null` and state UNEVALUATED, and the +4 that round is remembered for was computed by
    a separate script outside the driver.

    This does the PAIRED EVALUATION only -- gains, losses, net, denominator integrity, firings -- which
    is the part that needs nothing from the search. It does NOT adjudicate the four acceptance
    criteria: those consume the search outcome and the attribution report, which do not exist here.
    So the record it writes is always PENDING_VALIDATION: a believable net with no criteria applied is
    not an acceptance, and calling it one would be exactly the shortcut the acceptance rule forbids.
    """
    if a.score is None:
        return
    spec = _installed_spec(a.score)
    base = a.baseline or a.incumbent
    try:
        bc, _bsteps = load_run(base)
        ac, asteps = load_run(a.score)
    except SystemExit as exc:
        print(f"\n[6] SCORING SKIPPED: {exc}")
        rec["score_skipped"] = str(exc)
        return
    _bsteps = _bsteps or {}
    common = sorted(set(bc) & set(ac))
    cell = [c for c in common if a.cell in c]
    gains = [c for c in cell if not bc[c] and ac[c]]
    losses = [c for c in cell if bc[c] and not ac[c]]
    b_n, a_n = sum(bc[c] for c in cell), sum(ac[c] for c in cell)
    ok_denom = set(bc) == set(ac)
    fired = {c for c, st in asteps.items()
             if any(k.endswith("_gate") and v for s in st for k, v in s.items())}
    on_fired = [c for c in gains if c in fired]
    print(f"\n[6] PAIRED EVALUATION (scoring-only path: {reason})  n={len(cell)}")
    print(f"  control {b_n}/{len(cell)}   arm {a_n}/{len(cell)}   "
          f"+{len(gains)}/-{len(losses)}  net {len(gains) - len(losses):+d}")
    print(f"  DENOMINATOR INTEGRITY: {'OK' if ok_denom else 'MISMATCH'}")
    print(f"  firings (traj sidecar): {len(fired & set(cell))}; "
          f"gains on a fired case: {len(on_fired)}/{len(gains)}")

    # ---- THE FOUR CRITERIA, on this path too ----------------------------------------------------
    # An earlier version of this helper refused to adjudicate, on the grounds that the criteria
    # "consume the search outcome and the attribution report". That was WRONG, and checking rather
    # than assuming is what found it: `evaluate_criteria` reads only (a) the paired train comparison,
    # (b) a dev mapping, (c) safety counters, (d) mechanism telemetry. Every one of those comes from
    # the two RUN DIRECTORIES -- `_Paired` is a module-level adapter over the score dicts, and
    # `mechanism_evidence` reads the arm's own sidecar against the control's. None of it touches
    # `outcome`, `report` or `top`.
    #
    # So refusing here did not protect anything; it made acceptance unreachable on every path that
    # did not complete a search, which is the whole reason the lifecycle could not close. The
    # criteria themselves remain the only gate, and they still return PENDING (which BLOCKS) when
    # their evidence was never gathered -- that is the protection, and it lives in core where it
    # belongs, not in a caller's refusal to ask.
    dev = None
    if a.dev is not None:
        dev_base = a.dev_baseline or a.baseline or a.incumbent
        try:
            dbc, _ = load_run(dev_base)
            dac, _ = load_run(a.dev)
            dsel = a.dev_cell or a.cell
            dcell = [c for c in sorted(set(dbc) & set(dac)) if dsel in c]
            dev = {"arm_score": sum(dac[c] for c in dcell),
                   "control_score": sum(dbc[c] for c in dcell), "n": len(dcell)}
            print(f"  [6b] HELD-OUT: control {dev['control_score']}/{dev['n']}  "
                  f"arm {dev['arm_score']}/{dev['n']}  "
                  f"net {dev['arm_score'] - dev['control_score']:+d}")
        except SystemExit as exc:
            print(f"  [6b] dev split unreadable ({exc}) -- criterion 2 stays PENDING")
    # ---- READ THE MECHANISM WHERE IT ACTED, NOT WHERE IT WAS SCORED -----------------------------
    # `load_run` populates steps from `traj/query` ONLY, so feeding it alone to `mechanism_evidence`
    # asks a prereq-acting controller to prove itself in a phase it never runs in. Measured on the
    # real R1 pair: the relocation arm carries `controller_fired_gate` 84 times and `relocate_gate`
    # 61 times in `traj/prereq`, and ZERO gate keys in `traj/query` -- so criteria 3 and 4 both
    # reported PENDING for a controller with abundant execution evidence one directory over.
    #
    # This is the acting-phase / scoring-phase distinction, and the fix is to read the UNION. The
    # SCORE stays query-only (that is where the reward lives and must not move); only the EVIDENCE
    # widens, to the phases the controller actually executed in. Widening where evidence may be found
    # is not widening what counts as evidence: every counter still comes from the controller's own
    # declared readings, and a family with no declared reading still contributes nothing.
    _apre, _bpre = _load_prereq(a.score), _load_prereq(base)
    _arm_all = {**{f"prereq::{k}": v for k, v in _apre.items()}, **asteps}
    _ctl_all = {**{f"prereq::{k}": v for k, v in _bpre.items()}, **_bsteps}
    if _apre:
        _pf = sum(1 for st in _apre.values()
                  if any(k.endswith("_gate") and v for s2 in st for k, v in s2.items()))
        print(f"  mechanism evidence read over {len(_apre)} prereq + {len(asteps)} query episodes "
              f"({_pf} prereq episodes carry a firing)")
    mech_tel, safety, mech_prov = mechanism_evidence(_arm_all, control_steps=_ctl_all)
    report = evaluate_criteria(_Paired(bc, ac, cell, gains, losses), dev=dev,
                               safety=safety, telemetry=mech_tel)
    print(f"  [6c] ACCEPTANCE: {'ACCEPTED' if report.accepted else 'NOT ACCEPTED'}")
    for c in report.criteria:
        print(f"    C{c.number} {c.name:<32} {c.verdict}")
        if not c.allows_accept:
            print(f"         {c.detail}")
    # VALIDITY IS PRIOR TO ACCEPTANCE and is not one of the four. A denominator mismatch means no
    # paired comparison happened at all, so even four PASSes would be adjudicating a non-measurement.
    believable = ok_denom and len(cell) > 0
    accepted = bool(report.accepted) and believable
    if report.accepted and not believable:
        print("    *** NOT INSTALLED: the criteria passed but the measurement is not believable "
              f"(denominator_ok={ok_denom}, n={len(cell)}). Validity precedes acceptance.")
    rec["evaluation"] = {"n": len(cell), "control": b_n, "arm": a_n,
                         "gains": len(gains), "losses": len(losses),
                         "net": len(gains) - len(losses), "denominator_ok": ok_denom,
                         "firings": len(fired & set(cell)), "gains_on_fired": len(on_fired),
                         "criteria_adjudicated": True, "path": "scoring_only",
                         "believable": believable}
    rec["acceptance"] = report.to_dict()
    rec["acceptance"]["mechanism_provenance"] = mech_prov
    rec["acceptance"]["dev"] = dev
    rec["acceptance"]["installed"] = accepted
    if a.library is None:
        return
    if not spec:
        print("  NOT PERSISTED: the scored run recorded no controller spec (a control arm is "
              "expected to hit this).")
        rec["persist_skipped"] = "no controller spec in the scored run"
        return
    name = str(spec.get("name") or a.score.name)
    # WHICH STATE. Only a BELIEVABLE measurement earns a verdict:
    #   * four criteria pass + believable        -> ACCEPTED (installs into the incumbent)
    #   * believable, criteria FAIL on the merits -> MEASURED_NEGATIVE, true of THIS incumbent only
    #   * anything PENDING, or not believable     -> PENDING_VALIDATION, available forever
    # A criterion that is PENDING is unresolved evidence, never a refutation: banking it as a
    # negative would permanently exclude a mechanism whose evidence was simply never gathered.
    blocking_pending = [c.number for c in report.blocking if c.verdict == "PENDING_VALIDATION"]
    if accepted:
        state = ACCEPTED
    elif believable and not blocking_pending:
        state = MEASURED_NEGATIVE
    else:
        state = PENDING_VALIDATION
    keep_measurement = state in (ACCEPTED, MEASURED_NEGATIVE)
    try:
        lib = CandidateLibrary.load(a.library)
        lib.upsert(CandidateRecord(
            name=name,
            identity=ControllerIdentity(
                boundary=str(spec.get("locus") or spec.get("boundary") or ""),
                signal=str(spec.get("signal") or ""),
                action=str(spec.get("action") or ""),
                operator=str(spec.get("operator") or ""),
                capability_id=str(spec.get("capability_id") or spec.get("variant")
                                  or "unspecified"),
                phase=str(spec.get("phase") or a.phase)),
            spec=spec,
            state=state,
            context=EvaluationContext(
                incumbent_stack=tuple(stack), incumbent_token=str(base), cell=a.cell,
                split=str(a.cell), model=os.environ.get("ANCHOROPT_MODEL", ""),
                scored_phase=a.phase, acting_phase=str(spec.get("phase") or "")),
            measurement=(Measurement(
                arm_correct=a_n, control_correct=b_n, n_scored=len(cell),
                gains=tuple(gains), losses=tuple(losses), valid=True,
                firings=len(fired & set(cell)),
                detail=f"criteria {[c.verdict for c in report.criteria]}")
                if keep_measurement else None),
            provenance={"arm_run": str(a.score), "control_run": str(base), "out": str(a.out),
                        "paired_net": len(gains) - len(losses),
                        "denominator_ok": ok_denom,
                        "criteria": {f"C{c.number}": c.verdict for c in report.criteria},
                        "path": "scoring_only", "reason": reason},
            round_id=str(a.out.name),
            notes=((f"paired net {len(gains) - len(losses):+d} over n={len(cell)}; "
                    f"criteria {[c.verdict for c in report.criteria]}",)
                   if keep_measurement else
                   (f"PENDING: blocking criteria {blocking_pending or 'n/a'}; "
                    f"believable={believable} (denominator_ok={ok_denom}, n={len(cell)}). "
                    f"Evidence missing, NOT a refutation -- available for validation.",))))
        print(f"[9] PERSISTED {name} as {state} -> {lib.save(a.library)}")
        if state == PENDING_VALIDATION:
            print(f"    NOT a negative result -- stays available for validation")
    except Exception as exc:
        print(f"[9] PERSIST FAILED ({type(exc).__name__}: {exc}) -- round record still written")
        rec["persist_error"] = f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", type=pathlib.Path, required=True,
                    help="run dir of the MOVING INCUMBENT -- the trajectories to mine")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--library", type=pathlib.Path, default=None,
                    help="candidate library JSON (rounds/GOLDEN/candidates.json). Read to NAME the "
                         "incumbent stack this round measures against, and written after scoring so a "
                         "discovery survives the session. Omitted = no persistence, which is the old "
                         "behaviour and is why every round restarted from H0.")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--cell", default="vector")
    ap.add_argument("--cases", type=pathlib.Path, default=None,
                    help="the shard's case file. Supplies the `depends_on` graph, so a failed "
                         "QUERY can be attributed to the storage episode that built what it read. "
                         "Without it, attribution sees one episode at a time and a consequential "
                         "decision taken upstream is unreachable.")
    ap.add_argument("--phase", choices=("query", "prereq"), default="query",
                    help="which episodes to mine. `prereq` mines STORAGE-phase episodes, where the "
                         "writes happen -- query episodes in this corpus are ~99%% reads, so a "
                         "write-side residual is unminable from them. Storage episodes have no "
                         "pass/fail of their own, so failure is defined by the write ERRORS they "
                         "produce and the residual keeps its link to the dependent queries.")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--score", type=pathlib.Path, default=None,
                    help="run dir of the paired ARM; switches to the scoring phase")
    ap.add_argument("--min-on-fired", type=float, default=None,
                    help="fraction of gains that must sit on a FIRED case for the round to be "
                         "believed. Unset = report the fraction but enforce only the zero-firings "
                         "check; a default here would silently reclassify results already recorded.")
    ap.add_argument("--budget-exhausted", action="store_true",
                    help="declare the predeclared budget spent: stops with BUDGET_EXHAUSTED rather "
                         "than reporting a null.")
    ap.add_argument("--baseline", type=pathlib.Path, default=None,
                    help="run dir of the paired CONTROL (defaults to --incumbent)")
    ap.add_argument("--results", type=pathlib.Path, default=None,
                    help="JSON of EXTERNALLY measured paired results, keyed by `arm_label` as emitted "
                         "in arm_manifest.json. Routes selection through AnchorPolicyOpt.optimize on "
                         "real measurements. Without it the propose phase stops at UNEVALUATED and "
                         "names no winner -- it never guesses one from firing behaviour.")
    ap.add_argument("--dev", type=pathlib.Path, default=None,
                    help="run dir of the ARM on the independent held-out split. Criterion 2 is NO "
                         "AGGREGATE REGRESSION, not a required gain, and it is PENDING (which blocks) "
                         "when absent -- so a candidate measured on train alone cannot install.")
    ap.add_argument("--dev-baseline", type=pathlib.Path, default=None,
                    help="run dir of the CONTROL on the held-out split (pairs with --dev)")
    ap.add_argument("--dev-cell", default=None,
                    help="cell substring selecting the held-out cases (defaults to --cell). The split "
                         "is CELL-BASED because ordinary k-fold leaks across (domain x backend).")
    ap.add_argument("--eval-budget", type=int, default=None,
                    help="cap how many recorded results are consumed, to demonstrate or enforce "
                         "budget exhaustion. Arms left unmeasured keep the round OPEN "
                         "(BUDGET_EXHAUSTED), which is not a negative result.")
    a = ap.parse_args()
    # THE ACTIVE BACKEND, for adapter groundings that are backend-specific. A transform grounded for
    # another backend is not a distinct intervention -- `capacity_repair.repair()` dispatches on the
    # running backend, so it is an alias at best and infeasible at worst. Set before any adapter call
    # that might ground an action.
    os.environ.setdefault("ANCHOROPT_CELL", str(a.cell))
    a.out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"incumbent": str(a.incumbent), "cell": a.cell}

    # ---- THE MOVING INCUMBENT, NAMED ---------------------------------------------------------------
    # `rec["incumbent"]` is a PATH, which cannot say which controllers were installed while the run it
    # points at was produced. Measured consequence: rounds/AUTORUN/R1_RESULT.json recorded
    # controllers_installed=0 and incumbent/correct=18 on kv -- 18 being exactly the CONTROL score of
    # an already-accepted +4 controller. The round re-searched a mechanism the project had measured,
    # because nothing in the loop could ask "what is already installed?".
    #
    # This reads the library and RECORDS the answer. It deliberately does not change selection, mining
    # or acceptance: the library is persistence and deduplication, never a source of proposals. Feeding
    # known-good controllers to the proposer would make a later "rediscovery" unfalsifiable.
    _lib = CandidateLibrary.load(a.library) if a.library else CandidateLibrary()
    _stack = _lib.installed_stack(a.cell) if len(_lib) else ()
    rec["library"] = str(a.library) if a.library else ""
    rec["incumbent_stack"] = list(_stack)
    rec["stack_fingerprint"] = stack_fingerprint(_stack)
    rec["library_records"] = len(_lib)
    if len(_lib):
        print(f"\n[0] LIBRARY {a.library}: {len(_lib)} records · "
              f"{len(_lib.accepted())} accepted · {len(_lib.pending())} pending validation")
        print(f"    INCUMBENT STACK for cell {a.cell!r}: {len(_stack)} "
              f"controller(s)  [{rec['stack_fingerprint']}]")
        for _k in _stack:
            print(f"      {_k}")
        if not _stack:
            # Said out loud, because a silent empty stack is indistinguishable from "no library" and
            # that ambiguity is what let a bare-H0 baseline pass for a moving incumbent.
            print(f"      (none accepted for this cell yet -- measuring against bare H0)")
        _pend = _lib.pending()
        if _pend:
            print(f"    PENDING VALIDATION (available to measure, NOT losers):")
            for _p in _pend:
                print(f"      {_p.name}")

    print("=" * W)
    print(f"SELF-EVOLVE CYCLE #2  ·  incumbent = {a.incumbent.name}  ·  cell = {a.cell}")
    print("  the residual mined here is the one the PREVIOUS promotion created")
    print("=" * W)

    correct, steps = load_run(a.incumbent)
    if a.phase == "prereq":
        # STORAGE-PHASE MINING. A prereq episode is not scored, so "failed" cannot mean "wrong
        # answer": it means the episode produced at least one tool ERROR while building the store.
        # That is the residual a write-side controller addresses, and the link to the queries that
        # later read that store is kept so downstream value stays measurable.
        steps = _load_prereq(a.incumbent)
        failed = sorted(cid for cid, st in steps.items()
                        if a.cell in cid and _has_tool_error(st))
        print(f"  [phase=prereq] {len(steps)} storage episodes, {len(failed)} with a write error")
    else:
        failed = sorted(c for c, ok in correct.items() if not ok and a.cell in c)
    print(f"\n[1] INCUMBENT: {sum(correct.values())}/{len(correct)} correct overall; "
          f"{len(failed)} failed in {a.cell}")
    rec["incumbent_correct"] = sum(correct.values())
    rec["incumbent_scored"] = len(correct)
    rec["failed_in_cell"] = len(failed)

    # PROVENANCE. With the corpus's declared dependency graph, a failing case carries the
    # trajectories it DEPENDS ON as labelled upstream evidence. Without the graph nothing is
    # attached and behaviour is unchanged -- a host that declares no dependencies is not penalised.
    _graph = dependency_graph(a.cases)
    if _graph:
        _linked = sum(1 for c in failed[:a.limit] if upstream_of(c, _graph))
        print(f"  dependency graph: {len(_graph)} cases; {_linked}/{len(failed[:a.limit])} "
              f"failures have upstream episodes")
    # REPRESENTATIVE SAMPLING, not the first N by case id.
    #
    # `failed[:limit]` took the first 24 failures in LEXICOGRAPHIC ID ORDER, so which failures reached
    # the attributor was decided by string sorting. Measured on raw H0: the vector sample held 21%
    # zero-tool-call cases against a 34% population, and TWO OF FOUR SCENARIOS per backend were never
    # attributed at all. A mechanism concentrated in a late-sorting scenario is invisible to every round.
    #
    # The strata are read off the corpus's own case ids (scenario) and off an OBSERVABLE property of the
    # trajectory (did the episode call any tool). Neither is a mechanism name and neither is supplied by
    # me: the point is to make the sample look like the population, not to steer it toward a hypothesis.
    # Round-robin across strata, each stratum in id order, so the result is deterministic.
    def _stratum(cid: str) -> tuple:
        scenario = cid.split("-")[1] if "-" in cid else "?"
        made_call = any((st.get("decoded") or []) for st in steps.get(cid, []))
        return (scenario, made_call)

    _strata: dict[tuple, list[str]] = {}
    for c in failed:
        _strata.setdefault(_stratum(c), []).append(c)
    _sampled: list[str] = []
    _keys = sorted(_strata)
    while len(_sampled) < a.limit and any(_strata[k] for k in _keys):
        for k in _keys:
            if _strata[k] and len(_sampled) < a.limit:
                _sampled.append(_strata[k].pop(0))
    if len(_strata) > 1:
        print(f"  sampling: {len(_sampled)} of {len(failed)} failures across {len(_strata)} strata "
              f"(scenario x made-a-tool-call), round-robin")
        rec["sampling"] = {"n_failed": len(failed), "n_sampled": len(_sampled),
                          "strata": {"/".join(map(str, k)): len(v) + sum(
                              1 for c in _sampled if _stratum(c) == k)
                              for k, v in _strata.items()}}
    cases = [with_upstream(case_facts(c, steps.get(c, []), a.cell, a.phase), c, _graph, steps)
             for c in _sampled]
    print(f"\n[2] ATTRIBUTION over {len(cases)} current failures (autonomous)")
    diags = attribute(cases, a.out, a.live)
    print(f"  usable diagnoses: {len(diags)}")
    if not diags:
        # Not a negative result: attribution produced nothing usable, which says nothing about
        # whether this residual repays intervention.
        rec["state"] = "INSUFFICIENT_SUPPORT"
        rec["termination"] = {"outcome": "INSUFFICIENT_SUPPORT", "is_negative_result": False,
                              "detail": "attribution returned no usable diagnosis"}
        _score_pair_and_persist(a, rec, _stack, reason="attribution returned no usable diagnosis")
        json.dump(rec, open(a.out / "cycle2.json", "w"), indent=2, default=str)
        return 0
    rec["residual_keys_before"] = dict(residual_keys(diags).most_common())

    problems = build_residual_problems(
        diags, expressible=lambda d: expressible_under(d, runtime=runtime)[0])
    raw_problems = problems
    # R15: per-case projected states, so the OPT-IN firing key can be computed. Built only when the
    # alternative key is requested -- the default path is byte-identical to before.
    _pool_states = None
    if os.environ.get("ANCHOROPT_POOL_KEY") == "firing":
        _pool_states = {c: observable_states({c: steps.get(c, [])}) for c in _sampled}
    problems = pool_by_observable(problems, runtime=runtime, states_by_case=_pool_states)
    if len(problems) < len(raw_problems):
        print(f"\n[3a] POOLED {len(raw_problems)} prose-keyed families -> {len(problems)} by "
              f"OBSERVABLE CONDITION")
        for x in problems:
            if len(getattr(x, "merged_from", ())) > 1:
                print(f"   support {x.support:3d} on {list(x.observables)}")
                for k in x.merged_from:
                    print(f"      + {k[:84]}")
        rec["pooling"] = [{"rank": x.rank, "support": x.support,
                           "observables": list(x.observables),
                           "merged_from": list(x.merged_from)} for x in problems]

    # ---- [2b] THE PHASE DECISION -- made by ANCHOROPT's rule, not by a human reading output -------
    #
    # `phase_switch.decide` (S1 coverage >= 10%, S2 saturation < 50%, S5 unexplained-dominates) and its
    # benchmark-neutral twin `block_loop.residual_phase` both existed, were tested, and NOTHING CALLED
    # THEM -- so the continue/expand/stop decision was still a human reading the round's output. That
    # is the last hand-operated step in a loop whose whole claim is self-termination.
    #
    # `residual_phase` is used rather than `phase_switch.partition`: the latter reads BFCL trajectory
    # DIRECTORIES and re-mines to compute its own residual, while this driver already holds the
    # diagnoses. Same rule, same frozen thresholds (a test pins the two copies equal); only the input
    # shape differs.
    #
    # THE DISTINCTION THIS PRESERVES is the reason the module exists: "nothing survives the gates" is
    # NOT "the work is done". EXPAND_ATTRIBUTION means the limit is the REPRESENTATION and the
    # vocabulary should grow; STOP means the residual is genuinely addressed. Reporting the first as
    # the second would declare a family scientifically exhausted when in fact no signal can see it.
    from anchoropt.learning.block_loop import residual_phase as _residual_phase

    # GROUP LOCI THE WAY THE SEARCH DOES, or the rule and the search disagree about what one locus is.
    #
    # The default grouping is normalized PROSE, so twelve rewordings of one decision are twelve loci.
    # Measured on a real round: pooling merged them into FIVE problems with a top support of 14 of 24
    # (58% of the residual) while this rule saw twelve loci at 8.3% each -- every one failing S1 (>=10%)
    # -- and returned EXPAND_ATTRIBUTION. The representation was not the limit; the GROUPING was, and
    # expanding Phi on that verdict would have widened a vocabulary that already covered 58%.
    #
    # The key is the one `pool_by_observable` already computes: the set of declared signals whose
    # coverage check matches the diagnosis. Same function (`_signal_matches`), so there is no second
    # definition to drift -- and it groups on what the runtime can OBSERVE, never on phrasing.
    from anchoropt.learning.candidate_search import _signal_matches as _sm

    def _observable_key(d) -> str:
        txt = " ".join((d.failure_mechanism or "", d.consequential_decision or "",
                        d.evidence or "")).lower()
        sigs = tuple(sorted(s for s in runtime.declared_signals() if _sm(s, txt, runtime)))
        # A diagnosis no signal matches keeps its own prose key: it is SIGNAL_BLOCKED, and pooling all
        # such cases together would invent a locus out of "none of the above".
        return "|".join(sigs) if sigs else " ".join(str(d.consequential_decision).lower().split())

    _phase, _phase_reasons = _residual_phase(
        diags, expressible=lambda d: expressible_under(d, runtime=runtime)[0],
        group_key=_observable_key)
    print(f"\n[2b] PHASE DECISION (core's rule: S1 coverage, S2 saturation, S5 dominance)")
    for _line in _phase_reasons:
        print("   " + _line if _line else "")
    print(f"   PHASE = {_phase}")
    rec["phase_decision"] = {"phase": _phase, "reasons": list(_phase_reasons)}
    if _phase == "EXPAND_ATTRIBUTION":
        print("   -> the SIGNAL REPRESENTATION is the limit here, not the headroom. This is NOT "
              "exhaustion:")
        print("      the unexplained mass dominates, so no policy round can reach it. Do NOT force "
              "an anchor.")
    elif _phase == "STOP":
        print("   -> little actionable explained mass AND little unexplained mass: genuinely done "
              "at this residual.")

    print(f"\n[3] RESIDUAL PROBLEMS (rank-ordered by support)")
    for p in problems[:6]:
        print(f"  R{p.rank} support={p.support:3d} expressible={p.expressible}  {p.key[:66]}")
    rec["problems"] = [{"rank": p.rank, "support": p.support, "expressible": p.expressible,
                        "key": p.key} for p in problems]

    if not problems:
        # No residual problem survived expressibility and pooling. The proposal phase below indexes
        # `problems[0]`, so this must stop here rather than raise IndexError.
        #
        # `--score` IS still honoured here, by `_score_pair_and_persist`: the paired evaluation needs
        # nothing from the search, and a proposal phase that came up empty is not a reason to discard a
        # measurement that already exists on disk.
        rec["state"] = "INSUFFICIENT_SUPPORT"
        rec["termination"] = {"outcome": "INSUFFICIENT_SUPPORT", "is_negative_result": False,
                              "detail": "no residual problem survived expressibility/pooling"}
        rec["propose_phase"] = "STOPPED_NO_RESIDUAL_PROBLEM"
        _score_pair_and_persist(a, rec, _stack,
                                reason="no residual problem survived expressibility/pooling")
        json.dump(rec, open(a.out / "cycle2.json", "w"), indent=2, default=str)
        return 0

    top = problems[0]

    # ---- [3b] SEMANTIC LOCALIZATION: the attributor's behavioural reasoning reaches the search ----
    #
    # THE INTERFACE DEFECT THIS CLOSES. `proposed_behavior_change` was carried faithfully from the
    # attributor through ingestion and pooling to a consumer nobody called: the cycle-2 driver made
    # zero proposer calls, so Phi came from `residual.observables` -- a SET OF SIGNAL NAMES computed
    # by coverage match. For the R1 family the attributor named four repairs ("merge into an existing
    # entry, overwrite the least informative one, condense, remove a single superseded entry -- NEVER
    # clear the store"), exactly one of which had a declared signal. The search optimized that one and
    # produced a clear-suppression controller nobody proposed, which then measured net -3.
    #
    # THE DIVISION OF LABOUR IS UNCHANGED. The proposer returns (boundary, signal) only; core still
    # enumerates feasible actions, expands Phi, searches theta and selects on measurement. What the
    # proposer adds is WHICH condition at WHICH decision point the diagnosed repair implicates -- and
    # an explicit record of repairs no declared signal can express, which is strictly better than
    # silently substituting the nearest expressible one.
    seeded_from_proposer: tuple[str, ...] = ()
    if a.live:
        import client as roles
        _t0 = time.time()
        try:
            cands, prec = roles.propose_semantic(
                list(top.scoped_diagnoses()) if hasattr(top, "scoped_diagnoses") else list(diags),
                runtime=runtime, host=runtime.HOST)
        except Exception as exc:                      # a proposer that fails is RECORDED, not defaulted
            cands, prec = [], None
            print(f"\n[3b] SEMANTIC PROPOSER failed: {type(exc).__name__}: {exc}")
            rec["proposer_error"] = f"{type(exc).__name__}: {exc}"
        if prec is not None:
            write_teacher_usage(a.out / "teacher_usage.json", [prec],
                                stage="semantic_proposal", elapsed_s=time.time() - _t0)
        if prec is not None and prec.error:
            print(f"\n[3b] SEMANTIC PROPOSER error: {prec.error[:140]}")
            rec["proposer_error"] = prec.error
        declared = set(runtime.declared_signals()) if hasattr(runtime, "declared_signals") else set()
        proposals, unrealizable = [], []
        for c in (cands or []):
            if not isinstance(c, dict):
                continue
            if c.get("unrealizable"):
                unrealizable.append({"repair": str(c.get("unrealizable")),
                                     "blocked_by": str(c.get("blocked_by") or "not stated"),
                                     "rationale": str(c.get("rationale") or "")[:400]})
                continue
            sig = str(c.get("signal") or "")
            row = {"boundary": str(c.get("boundary") or ""), "signal": sig,
                   "policy_class": str(c.get("policy_class") or ""),
                   "rationale": str(c.get("rationale") or "")[:400],
                   "action_preference": c.get("action_preference"),
                   "declared": sig in declared}
            proposals.append(row)
        if proposals or unrealizable:
            print(f"\n[3b] SEMANTIC LOCALIZATION ({len(proposals)} proposal(s), "
                  f"{len(unrealizable)} unrealizable repair(s) recorded)")
            for r in proposals:
                mark = "" if r["declared"] else "   <- NOT a declared signal; core will refuse it"
                print(f"   {r['boundary']:26s} {r['signal']:36s} {r['policy_class']}{mark}")
                print(f"      why: {r['rationale'][:104]}")
            for u in unrealizable:
                print(f"   UNREALIZABLE: {u['repair'][:84]}")
                print(f"      blocked by: {u['blocked_by'][:92]}")
        rec["semantic_proposals"] = proposals
        rec["unrealizable_repairs"] = unrealizable
        # SEED Phi FROM THE PROPOSER, intersected with what the host declares. A proposal naming a
        # signal the runtime does not have is kept in the record and dropped from the space -- the
        # same fail-loud rule the family-seeded path uses.
        seeded_from_proposer = tuple(dict.fromkeys(
            r["signal"] for r in proposals if r["declared"]))
        # THE LOCALIZATION TRAVELS ON THE RESIDUAL, not as an argument to the search.
        #
        # `test_cycle2_thinness` forbids the driver passing `signals=` to `optimize_residual`, and it
        # is right: a driver naming the signals IS the driver steering the search. But a residual
        # family already carries `observables` -- the conditions attribution implicates -- and core
        # seeds Phi from exactly that. So the proposer's localization belongs there: it is a better
        # answer to the same question ("which conditions does the attribution implicate"), refined by
        # the behavioural repair rather than by coverage match alone.
        #
        # Core's contract is untouched: it still reads one field off the residual and still owns
        # actions, expansion, theta and selection.
        if seeded_from_proposer:
            try:
                top.observables = seeded_from_proposer
                print(f"   -> Phi seeded from the LOCALIZATION: {list(seeded_from_proposer)}")
            except Exception:
                print("   -> the residual does not accept a refined localization; "
                      "falling back to its own observables")

    print(f"\n[4] STRUCTURED SEARCH on R{top.rank} (core: WHERE -> WHAT -> HOW)")
    events = []
    for cid in sorted(top.case_ids):
        events.extend(keep_steps(steps.get(cid, [])))
    # The per-step states are the INPUT to per-boundary projection, not the thing predicates are
    # validated against: core calls `runtime.states_at(boundary, states)` and fails closed if the
    # boundary's own information set cannot be built. For the commitment gate that projection needs
    # the proposed calls, so `decoded` is carried through here rather than summarised away.
    states = observable_states({c: steps.get(c, []) for c in failed})
    _proj = sum(len(runtime.states_at("post_generation_pre_exec", states)) or 0 for _ in (1,))
    print(f"  states: {len(states)} per-step -> {_proj} projected at the commitment gate")
    # NO EVALUATOR IN THE PROPOSE PHASE, and that is the point.
    #
    # This used to pass `evaluate=lambda _a: 0, improves=lambda _o: False` to "let the schedule run to
    # completion". It did the opposite of letting the algorithm run: every candidate scored
    # identically, `improves` was never true, so the seeded Phi was declared NO_BENEFIT WITHOUT
    # MEASUREMENT, Phi expansion fired on that manufactured verdict, and the round re-enumerated the
    # grammar. Measured: 4 grounded observables in, 117 arms out, 113 of them from expansion.
    #
    # Core now halts at REALIZABLE_UNMEASURED when there is no evaluator -- keeping the boundary, the
    # seeded Phi and the candidate set -- so the propose phase emits the FOCUSED set and the measured
    # round decides what happens next. Expansion and backward movement follow real results.
    outcome = optimize_residual(top, runtime=runtime, host=runtime.HOST, events=events,
                               states=states)
    # RECOVER THE PREDICATE OBJECTS so the spec is translated structurally rather than parsed out of a
    # name the expansion path truncates to 60 chars. `install_signal` stores a callable, but core
    # builds it as `lambda st, _p=pred: bool(_p.evaluate(st))` -- the object survives in the closure's
    # default, so no adapter change is needed to read it back. A callable without that default simply
    # yields no spec, and the driver says so rather than emitting a guess.
    _SYNTH_PREDICATES = {}
    for _name, _fn in (getattr(runtime, "EXPANDED_SIGNALS", {}) or {}).items():
        _obj = (_fn.__defaults__ or (None,))[0] if hasattr(_fn, "__defaults__") else None
        if _obj is not None and hasattr(_obj, "evaluate"):
            _SYNTH_PREDICATES[_name] = _obj
    print(f"  boundaries: {list(outcome.boundaries)}   visited: {list(outcome.visited)}")
    print(f"  moves_earlier={outcome.moves_earlier}  state={outcome.state}  "
          f"Phi expanded={outcome.expanded} ({len(outcome.signals_installed)})")
    for att in outcome.attempts:
        print(f"    {att.boundary:26s} {att.state:26s} built={att.candidates_built:4d} "
              f"-> next {att.coordinate_changed}")
    rec["search"] = outcome.as_dict()

    if not outcome.candidates:
        print("\n  NO CANDIDATE -- the loop has no next intervention at this residual")
        rec["state"] = outcome.state
        _score_pair_and_persist(a, rec, _stack, reason="the search produced no candidate at this residual")
        json.dump(rec, open(a.out / "cycle2.json", "w"), indent=2, default=str)
        return 0

    # ---- pick the arm to measure: most-discriminating controller, by FIRING BEHAVIOUR -------------
    # Not by name and not by semantic preference. Among the controllers the search built, take the one
    # whose phi fires on a non-trivial minority of observed states: a predicate that fires everywhere
    # is not a trigger, and one that fires nowhere cannot be measured.
    scored: list = []
    unfireable: dict = {}
    # Grammar objects, so `fields_read` can inspect the predicate structurally rather than parse a
    # truncated name. Recovered from the closure default core stores at install time.
    _P_obj = {}
    for _n, _f in (getattr(runtime, "EXPANDED_SIGNALS", {}) or {}).items():
        _o = (_f.__defaults__ or (None,))[0] if hasattr(_f, "__defaults__") else None
        if _o is not None and hasattr(_o, "evaluate"):
            _P_obj[_n] = _o
    for arm in outcome.candidates:
        # A seeded Phi is made of DECLARED signals, which live in the runtime's own signal table --
        # `EXPANDED_SIGNALS` holds only what expansion installed. Reading just the latter dropped
        # every candidate the residual family seeded, so a correctly focused round emitted nothing.
        pred = runtime.EXPANDED_SIGNALS.get(arm.signal)
        if pred is None:
            _sig = arm.signal
            # A PARAMETERIZED SIGNAL NEEDS A PARAMETER TO BE EVALUATED AT ALL.
            #
            # Evaluating with `{}` raised KeyError on every state, and "a raising phi is not a firing"
            # turned that into an all-False vector -- so a signal that fires 15/132 at theta=0.2 and
            # 47/132 at 0.45 was reported "fires on none of the residual's observed states" and its
            # whole family was dropped. That rule is right at runtime and wrong in a fireability
            # check, exactly as the `.evaluate`/`.fires_on` round-trip defect was.
            #
            # The probe value is a MIDPOINT of the declared domain, used only to ask "can this
            # predicate discriminate here at all". It is NOT the theta that will run: the optimizer
            # searches the grid and measurement chooses.
            _probe = {}
            try:
                _probe = dict(runtime.probe_params(_sig) or {})
            except Exception:
                _probe = {}
            if not _probe:
                try:
                    for _d in (runtime.parameter_domains(_sig) or ()):
                        if getattr(_d, "values", ()):
                            _probe[_d.name] = _d.values[len(_d.values) // 2]
                        else:
                            _probe[_d.name] = (float(_d.low) + float(_d.high)) / 2.0
                except Exception:
                    _probe = {}

            def pred(st, _s=_sig, _p=_probe):           # noqa: E731 - declared-signal evaluator
                try:
                    return bool(runtime.evaluate_signal(_s, st, _p))
                except Exception:
                    # A signal that cannot be evaluated on this state has not fired. Same rule the
                    # runtime hook applies to a raising phi: not-evaluable is not a firing.
                    return False

        class _P:
            def evaluate(self, st, _p=pred):
                return bool(_p(st))

        # PROGRAMMATIC INVARIANT, not a prompt's promise: a controller must be able to fire at the
        # boundary the search placed it. WRITE1 selected one that could not, and no amount of prompt
        # revision would have caught it.
        # STATES MUST MATCH THE ARM'S BOUNDARY. Passing the per-step states here rejected all 37
        # commitment-gate candidates as "fires on none of the residual's observed states" -- true of
        # those states, and irrelevant: a pre-dispatch predicate cannot fire on a state that carries
        # no proposed call. The same defect the WHAT search had, reappearing in arm SELECTION.
        _arm_states = runtime.states_at(arm.boundary.value, states) \
            if hasattr(runtime, "states_at") else states
        ok, why = can_fire_at_boundary(arm, _P_obj.get(arm.signal, _P()), runtime=runtime,
                                       states=_arm_states)
        if not ok:
            unfireable.setdefault(f"{arm.boundary.value}/{arm.signal}", why)
            continue
        fv = firing_vector(_P_obj.get(arm.signal, _P()), _arm_states)
        n = sum(fv)
        # FIREABILITY IS A VALIDITY FILTER, NOT A RANKING. A predicate that fires on every observed
        # state is not a trigger, and one that fires on none cannot be measured at all -- both are
        # unmeasurable rather than unpromising, so they are excluded with a reason. Within the
        # measurable set, NOTHING here prefers one arm over another: that is what measurement is for.
        if 0 < n < len(fv):
            scored.append((n, arm))
        else:
            unfireable.setdefault(
                f"{arm.boundary.value}/{arm.signal}",
                f"fires on {n}/{len(fv)} observed states, so no paired contrast exists")
    # DETERMINISTIC ORDER ONLY -- by label, so the manifest is byte-stable across runs. This is
    # deliberately NOT a preference order: every arm below is emitted for evaluation.
    scored.sort(key=lambda t: t[1].instantiated.label)
    if unfireable:
        print(f"\n  REJECTED as unfireable at their own boundary: {len(unfireable)}")
        for k, why in list(unfireable.items())[:4]:
            print(f"     {k[:56]:56s} {why[:90]}")
        rec["unfireable"] = unfireable
    if not scored:
        # ARMS WERE BUILT; none has a paired contrast ON THIS RESIDUAL. That is UNEVALUATED, not a
        # negative result and not "insufficient support" -- support is about the residual's size,
        # and this is about the controllers' firing behaviour. A signal that fires on none of these
        # states may be exactly right for the residual a later incumbent leaves.
        #
        # The manifest is still written: the arms exist, they were grounded, and a round that
        # emitted nothing is indistinguishable from a round that searched nothing.
        print("\n  every built controller fires on all or none of the observed states -- "
              "not measurable on THIS residual")
        manifest = arm_manifest([arm for arm in outcome.candidates],
                                incumbent_id=str(a.incumbent.name),
                                incumbent_token=str(a.incumbent))
        for row in manifest["arms"]:
            row["fires_on_states"] = 0
            row["total_states"] = len(states)
            row["not_measurable"] = "fires on all or none of this residual's observed states"
        # THE SPEC IS WRITTEN HERE TOO. These arms are not measurable on THIS residual, but they are
        # built and grounded, and a later incumbent's residual may fire them. A manifest without an
        # installable spec is an arm nobody can run.
        _specs = [_spec_for(arm, _SYNTH_PREDICATES, a.phase, runtime) for arm in outcome.candidates]
        (a.out / "arm_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n")
        rec["arm_manifest"] = manifest
        if _specs:
            (a.out / "controllers.json").write_text(
                json.dumps(_specs, indent=2, default=str) + "\n")
            (a.out / "controller.json").write_text(
                json.dumps(_specs[0], indent=2, default=str) + "\n")
            rec["controller_specs"] = _specs
            rec["controller_spec"] = _specs[0]
            print(f"  {len(_specs)} controller spec(s) written (not measurable on this residual)")
        rec["state"] = EXT_UNEVALUATED
        rec["termination"] = {
            "outcome": EXT_UNEVALUATED, "is_negative_result": False,
            "detail": (f"{len(outcome.candidates)} arm(s) built and grounded; none fires on a "
                       f"non-trivial minority of this residual's states, so none has a paired "
                       f"contrast here. Nothing was measured.")}
        _score_pair_and_persist(a, rec, _stack,
                                reason="no built controller fires on a non-trivial minority of this residual's states")
        json.dump(rec, open(a.out / "cycle2.json", "w"), indent=2, default=str)
        return 0
    # ---- [5] EMIT EVERY MEASURABLE ARM. Selection requires measurement. --------------------------
    #
    # THE HEURISTIC THAT USED TO LIVE HERE. This block selected `scored[0]` after ranking by
    # `abs(firing_rate - 0.25)` -- the arm firing nearest 25% of observed states -- and the round then
    # reported NO_BENEFIT. Three separate problems, and removing it fixes all three:
    #
    #   1. it SELECTED WITHOUT MEASURING, so the "optimizer" never optimized: AnchorPolicyOpt.optimize
    #      exists, takes an `evaluate` callback, and was never called by this driver;
    #   2. reporting NO_BENEFIT asserted a measurement that never happened -- the same error class as
    #      believing a null from a channel that was never live;
    #   3. the 25% target has no theory behind it. A controller firing on 24% of states is not thereby
    #      better than one firing on 60%; the number was a guess about what a useful trigger rate is.
    #
    # It is NOT replaced by another heuristic. Evaluation on this host is not inline -- measuring one
    # arm means a GPU job -- so the propose phase emits EVERY measurable arm as a work order and stops
    # at UNEVALUATED. `--results` then feeds the recorded paired outcomes back through the SAME
    # AnchorPolicyOpt.optimize that the toy host exercises inline (see examples/toy_host/).
    arms_out = [arm for _n, arm in scored]
    # KEYED ON `arm.label`, which is what `arm_manifest` emits as `arm_label`:
    # "<boundary>/<signal>/<operator>:<variant>". Keying on `instantiated.label` -- just the last
    # segment -- matched nothing, so every arm in the manifest reported fires_on_states=0. That was
    # visibly false (the fireability filter had already EXCLUDED every all-zero arm), and it survived
    # because nothing read the field back. Caught on the first real BFCL run.
    fires_by_label = {arm.label: n for n, arm in scored}
    # ---- SETTLED INTERVENTIONS ARE EXCLUDED; LOCI AND SIGNALS ARE NOT ---------------------------
    #
    # A re-mine after a promotion re-emitted the ACCEPTED zero-call arms firing 86/174, because the
    # signal's STATE still appears in the trace: the controller fires at step 0, injects, and the
    # episode continues, so step 0 is still a no-call state -- while 0 of 66 remaining failures are
    # zero-call. Re-measuring a settled mechanism against an incumbent that already contains it costs
    # GPU and yields a meaningless near-zero contrast.
    #
    # THE EXCLUSION IS PER (boundary, signal, OPERATOR:VARIANT) -- the full intervention identity, at
    # the same granularity the ledger already records (`policies_tested` is keyed by policy AND
    # corpus). Blacklisting a locus or a signal would be wrong: the same decision point may still host
    # a DIFFERENT policy, and the same signal may still express a residual failure under another
    # action. Only the exact intervention already installed-and-accepted here is skipped.
    _settled: set = set()
    _installed_spec = os.environ.get("ANCHOROPT_INSTALLED_CONTROLLERS", "")
    for _sp in [x for x in _installed_spec.split(",") if x.strip()]:
        try:
            _sd = json.load(open(_sp.strip()))
        except Exception as _se:
            print(f"  ledger: could not read installed spec {_sp.strip()}: {_se}")
            continue
        _settled.add((str(_sd.get("locus")), str(_sd.get("name")),
                      "%s:%s" % (_sd.get("operator"), _sd.get("variant"))))
    if _settled:
        _before = len(scored)
        _kept = []
        for _n, _arm in scored:
            _ident = (_arm.boundary.value, _arm.signal,
                      "%s:%s" % (_arm.instantiated.operator.value, _arm.instantiated.variant))
            if _ident in _settled:
                # NOT silent: a skipped arm is recorded with its identity, because "the loop chose not
                # to re-measure this" and "the loop never found this" are different facts.
                rec.setdefault("settled_skipped", []).append(
                    {"boundary": _ident[0], "signal": _ident[1], "policy": _ident[2],
                     "reason": "already installed and accepted on this incumbent"})
                continue
            _kept.append((_n, _arm))
        if len(_kept) < _before:
            print(f"\n  SETTLED: skipped {_before - len(_kept)} arm(s) already installed and accepted")
            for _sk in rec.get("settled_skipped", []):
                print(f"     {_sk['boundary']}/{_sk['signal']}/{_sk['policy']}")
            # The same (boundary, signal) may survive under a DIFFERENT policy -- say so explicitly.
            _same = [f"{a.signal}/{a.instantiated.operator.value}:{a.instantiated.variant}"
                     for _x, a in _kept
                     if any(a.boundary.value == b and a.signal == g for b, g, _p in _settled)]
            if _same:
                print(f"     ... and KEPT {len(_same)} arm(s) at the same (boundary, signal) under a "
                      f"different policy: {_same[:4]}")
        scored = _kept
        arms_out = [a for _n, a in scored]
        fires_by_label = {a.label: n for n, a in scored}
    if not scored:
        print("\n  every measurable arm is an already-settled intervention on this incumbent")
        rec["state"] = EXT_UNEVALUATED
        rec["termination"] = {"outcome": EXT_UNEVALUATED, "is_negative_result": False,
                              "detail": ("all measurable arms are interventions already installed and "
                                         "accepted here; nothing new to measure at this residual")}
        _score_pair_and_persist(a, rec, _stack,
                                reason="every measurable arm is already installed and accepted here")
        json.dump(rec, open(a.out / "cycle2.json", "w"), indent=2, default=str)
        return 0

    print(f"\n[5] {len(arms_out)} MEASURABLE ARM(S) EMITTED -- none is preferred here")
    print("  selection requires measurement; this phase names no winner")
    for _n, arm in scored[:12]:
        print(f"     {arm.instantiated.label[:52]:52s} fires {_n:3d}/{len(states)}  "
              f"{arm.boundary.value}/{arm.signal}")

    manifest = arm_manifest(arms_out, incumbent_id=str(a.incumbent.name),
                            incumbent_token=str(a.incumbent))
    # Per-arm firing counts are recorded as OBSERVATION, never as a score. A reader may want to know
    # how selective an arm is; nothing in the pipeline may rank on it.
    # NO SEPARATE APPLICABILITY ROWS ARE NEEDED HERE. A conditioned arm is a REAL arm -- the host's
    # grounding emitted it, `build_arms` instantiated it, and `PolicyArm.label` already distinguishes
    # it (`...:search_other_container@backend=vector`). So `arm_manifest` rows it, `fires_by_label`
    # keys it, and `ExternalEvaluation` looks its result up, all unchanged.
    #
    # AN EARLIER VERSION SYNTHESIZED VARIANT ROWS IN THE DRIVER, and it was unmeasurable: such a row
    # has no arm, so `select_on_measurement` -- which rebuilds arms from the proposal and requests
    # results by `arm.label` -- could not address it. Its result read as `missing` and the round
    # reported UNEVALUATED. Verified by probe before this was rewritten.
    for row in manifest["arms"]:
        row["fires_on_states"] = fires_by_label.get(row["arm_label"], 0)
        row["total_states"] = len(states)
        _ap = next((_applicability_of(x) for x in arms_out if x.label == row["arm_label"]), {})
        if _ap:
            row["applicability"] = dict(_ap)
    specs = [_spec_for(arm, _SYNTH_PREDICATES, a.phase, runtime) for _n, arm in scored]
    (a.out / "arm_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"\n  manifest written: {a.out / 'arm_manifest.json'}")
    rec["arm_manifest"] = manifest
    rec["state"] = EXT_UNEVALUATED
    rec["termination"] = {
        "outcome": EXT_UNEVALUATED, "is_negative_result": False,
        "detail": (f"{len(arms_out)} arm(s) built and grounded; none measured. Evaluate them against "
                   f"the frozen incumbent and re-run with --results to select on measurement.")}

    # THE SPEC IS THE DELIVERABLE of the propose phase: a runner loads it, nobody retypes it. One spec
    # per arm, because no arm is privileged until something measures them.
    # ONE SPEC PER ARM, from a single builder used by every exit -- a round that emits a manifest
    # and no spec has produced an arm nobody can install, which is how a round silently stalls.
    # `specs` was built above, because the manifest needs it to row each conditioned variant.
    if specs:
        (a.out / "controllers.json").write_text(json.dumps(specs, indent=2, default=str) + "\n")
        print(f"  {len(specs)} controller spec(s) written: {a.out / 'controllers.json'}")
        rec["controller_specs"] = specs
        # Back-compat: the single-spec filename earlier runners read. It is the FIRST spec in
        # deterministic label order, and it carries no claim of being the best one.
        (a.out / "controller.json").write_text(json.dumps(specs[0], indent=2, default=str) + "\n")
        rec["controller_spec"] = specs[0]
    else:
        print("  NOTE no synthesized predicate objects -- these signals came from shipped Phi, so "
              "the host already knows how to evaluate them and no spec is needed.")

    # ---- [5b] SELECT ON MEASUREMENT, when measurements exist -------------------------------------
    #
    # THE WIRING THAT WAS MISSING. `AnchorPolicyOpt.optimize(proposal, incumbent=, evaluate=)` already
    # existed and this driver never called it -- that was the gap, not a missing optimizer.
    #
    # IT IS STILL NOT CALLED FROM HERE. The driver must not construct proposals or drive the optimizer
    # itself: the search belongs to core, and a loop that re-derives the search space can steer it.
    # So the driver hands core the arms it already has plus the measurements it loaded, and
    # `select_on_measurement` does the grouping, the optimization and the classification.
    if a.results is not None:
        sel = select_on_measurement(
            arms_out, runtime=runtime, host=runtime.HOST,
            incumbent_id=str(a.incumbent.name), incumbent_token=str(a.incumbent),
            evaluation=ExternalEvaluation.from_json(a.results, budget=a.eval_budget),
            case_ids=tuple(getattr(top, "case_ids", ()) or ()),
            # The residual's own observed values, so a PARAMETERIZED arm can build its theta grid.
            # Without them every such arm is rejected `no_parameter_grid` and a round holding real
            # paired measurements reports UNEVALUATED.
            observations=_observed_values(states))

        print("\n[5b] SELECTION ON MEASUREMENT (core: select_on_measurement)")
        print(f"  evaluations completed : {sel.evaluations_completed}"
              + (f" / budget {a.eval_budget}" if a.eval_budget is not None else ""))
        for row in sel.per_group:
            print(f"     {row['boundary']}/{row['signal']}: built {row['arms_built']}, "
                  f"measured {row['arms_measured']}, winner {row['winner'] or '(none)'}")
        if sel.unevaluated:
            print(f"  UNEVALUATED arms (no recorded result): {len(sel.unevaluated)}")

        rec["state"] = sel.outcome_class
        rec["termination"] = sel.as_dict()
        rec["policy_opt"] = [dict(r) for r in sel.per_group]
        if sel.improved:
            print(f"\n  WINNER ON MEASUREMENT : {sel.winner.label}  "
                  f"net {sel.winner_result.net:+d}")
            rec["selected_arm"] = {"label": sel.winner.label, "locus": sel.winner.boundary.value,
                                   "signal": sel.winner.signal, "action": sel.winner.action.value,
                                   "theta": dict(sel.winner_theta),
                                   "net": sel.winner_result.net}
        else:
            # WHICH non-improvement this is matters, and core keeps the three apart: UNEVALUATED when
            # nothing was measured, BUDGET_EXHAUSTED when arms were left open, and NO_BENEFIT only
            # when every arm was really measured and none helped.
            print(f"\n  NO WINNER -- {sel.outcome_class}: {sel.detail()}")

    # THE `next_arm` FIELD IS GONE, and its absence is the point. It recorded ONE arm as "the next
    # one" -- the last vestige of the firing-rate heuristic, and a NameError after the heuristic was
    # deleted (it still read the removed `fires` variable). Crashed on the first real BFCL run; the
    # skipped synthetic CLI tests could not reach this line.
    #
    # There is no singular next arm any more. `arm_manifest.json` records every measurable arm with its
    # own firing count, and `selected_arm` appears only when a MEASUREMENT chose one. Reintroducing a
    # single-arm field here would recreate the thing this release removed.
    rec["arms_emitted"] = len(arms_out)
    rec["expanded_signals_used"] = sorted(
        {a.signal for a in arms_out} & set(outcome.signals_installed))

    # ---- scoring phase ---------------------------------------------------------------------------
    if a.score is not None:
        base = a.baseline or a.incumbent
        bc, bsteps = load_run(base)
        ac, asteps = load_run(a.score)
        common = sorted(set(bc) & set(ac))
        cell = [c for c in common if a.cell in c]
        gains = [c for c in cell if not bc[c] and ac[c]]
        losses = [c for c in cell if bc[c] and not ac[c]]
        net = len(gains) - len(losses)
        b_n, a_n = sum(bc[c] for c in cell), sum(ac[c] for c in cell)
        delta = 100.0 * (a_n - b_n) / max(1, len(cell))
        print(f"\n[6] PAIRED EVALUATION  n={len(cell)}")
        print(f"  control {b_n}/{len(cell)}   arm {a_n}/{len(cell)}   "
              f"delta {delta:+.2f}pp   +{len(gains)}/-{len(losses)}")
        # DENOMINATOR INTEGRITY: the two runs must have scored the same cases.
        ok_denom = set(bc) == set(ac)
        print(f"  DENOMINATOR INTEGRITY: {'OK' if ok_denom else 'MISMATCH'}")
        # FIRINGS come from the trajectory sidecar, never from a registry-derived dict.
        fired = {c for c, st in asteps.items()
                 if any(k.endswith("_gate") and v for s in st for k, v in s.items())}
        on_fired = [c for c in gains if c in fired]
        print(f"  firings (traj sidecar): {len(fired & set(cell))}; "
              f"gains on a fired case: {len(on_fired)}/{len(gains)}")
        # VALIDITY BEFORE ACCEPTANCE. This line previously read
        #     state = IMPROVED if (net > 0 and ok_denom) else NO_BENEFIT
        # so a denominator MISMATCH -- two runs that scored different cases, i.e. no paired
        # comparison at all -- was recorded as NO_BENEFIT: a broken measurement banked as evidence
        # against the candidate. An unbelievable round has no result to classify.
        prereq_fired = {c for c, st in _load_prereq(a.score).items()
                        if any(k == "controller_fired_gate" and v for s in st for k, v in s.items())}
        ctl_prereq_fired = {c for c, st in _load_prereq(a.baseline).items()
                            if any(k == "controller_fired_gate" and v
                                   for s in st for k, v in s.items())}

        # ---- DELAYED EFFECTS: WHERE IT ACTED vs WHERE IT WAS SCORED ---------------------------
        # A controller mined from a state-building episode acts there and is scored later, so
        # demanding a firing inside the scored case's own trace makes that whole class of
        # intervention unmeasurable BY CONSTRUCTION. The previous two lines did exactly that, and
        # a measured round passed all four criteria on construction-phase telemetry while its
        # entire delta lived in the scored phase.
        #
        # `anchoropt.learning.delayed_effect` widens WHERE evidence may be found -- to the causal
        # cone the CORPUS declares -- and not what counts as evidence. A gain whose declared cone
        # contains no firing is still unattributed. What it also separates out is the other half of
        # the original defect: `contamination_free` used to mean "the arm never fired in prereq",
        # which condemns every state-building controller. It now means "the state the scored cases
        # are graded against was BUILT THE SAME WAY in both arms" -- which is the actual confound,
        # and is false exactly when the arm repaired the store during construction and the control
        # did not.
        graph = dependency_graph(a.cases)
        phase_of = ({c: "prereq" for c in _load_prereq(a.score)}
                    | {c: "query" for c in asteps})
        de = evaluate_delayed_effect(
            gained_units=gains,
            firings_by_unit={**{c: True for c in prereq_fired},
                             **{c: (c in fired) for c in cell}},
            phase_of=phase_of, graph=graph,
            telemetry_by_phase={"prereq": _phase_telemetry(a.score, "prereq"),
                                "query": _phase_telemetry(a.score, "query")},
            # Construction is matched only when BOTH arms built the state under the same protocol.
            # The arm firing in prereq while the control did not IS the unmatched protocol, so this
            # is measured from the two runs rather than asserted.
            arm_conditions=_arm_conditions(a.score, len(prereq_fired)),
            control_conditions=_arm_conditions(a.baseline, len(ctl_prereq_fired)),
            declared_phase=_installed_phase(a.score),
            # What the run ACTUALLY executed. Without this, a round that eliminated the controller's
            # acting phase while controlling for construction reports UNSUPPORTED -- a statement
            # about the host -- when the true finding is that the controller never had a chance to
            # act. Empty means the host did not say, and no opportunity claim is then made.
            phases_executed=_phases_executed(a.score, len(asteps)))

        chan_ok, chan_msg = channel_integrity(
            gains=len(gains), gains_on_fired=len(on_fired), firings=len(fired & set(cell)),
            min_fraction_on_fired=a.min_on_fired)
        # The acting phase decides which channel question is the right one to ask. For a controller
        # that acted in the scored phase the original same-phase check is exact and is kept; for one
        # that acted during construction it is the wrong question, and the cone-based answer
        # replaces it. Both are printed either way, so the substitution is never silent.
        acted_in_scored_phase = de.site.phase == "query" or not de.site.fired
        channel_ok_final = chan_ok if acted_in_scored_phase else de.channel_ok
        validity = Validity(denominator_ok=ok_denom,
                            contamination_free=de.contamination_free,
                            channel_ok=channel_ok_final,
                            detail=chan_msg if acted_in_scored_phase else de.attribution.detail)
        print(f"  channel integrity (same-phase): {chan_msg}")
        print(f"  ACTING SITE: phase={de.site.phase!r} units={len(de.site.units)} "
              f"declared={de.site.declared_phase!r} as-declared={de.site.phase_as_declared}")
        print(f"  PHASES EXECUTED: {sorted(de.phases_executed) or 'not reported by host'} "
              f"-- had_opportunity={de.had_opportunity}")
        if not de.had_opportunity:
            print("  *** NO OPPORTUNITY: the phase this controller declares it acts in did not "
                  "execute in this run, so no delta measured here is evidence about it. This is an "
                  "UNINTERPRETABLE evaluation, not a negative result.")
        print(f"  DELAYED ATTRIBUTION [{de.attribution.mode}]: {de.attribution.detail}")
        print(f"  MATCHED CONDITIONS: {'OK' if de.conditions.ok else 'NOT MATCHED'} -- "
              f"{de.conditions.detail}")
        print(f"  verification phase: {de.verification_note}")
        print(f"  prereq firings -- arm {len(prereq_fired)} / control {len(ctl_prereq_fired)}")
        print(f"  => channel_ok={channel_ok_final} (from "
              f"{'same-phase check' if acted_in_scored_phase else 'declared causal cone'}), "
              f"contamination_free={de.contamination_free}")
        # ---- THE FOUR CRITERIA ----------------------------------------------------------------
        # `net > 0` is criterion 1 ALONE, and it was the whole installation test until here. The
        # other three are adjudicated by `anchoropt.learning.acceptance_criteria`, which had zero
        # production consumers: every reference to it outside its own module was a test fixture.
        #
        # The validity guards above are NOT these criteria and are not replaced by them. Validity
        # asks "is this measurement believable at all" (same cases scored, no prereq contamination,
        # gains concentrated on fired cases); the criteria ask "does a believable measurement justify
        # installing". A round must clear both, and they stay separate so a broken measurement is
        # never banked as evidence against the candidate.
        #
        # MISSING EVIDENCE BLOCKS. Criteria 2/3/4 return PENDING_VALIDATION when their evidence was
        # never gathered, and PENDING is not a pass and not a measured negative result -- the round
        # reports BLOCKED and the question stands. No counter is defaulted to zero to fill a gap.
        dev = None
        if a.dev is not None:
            dev_base = a.dev_baseline or a.baseline or a.incumbent
            dbc, _ = load_run(dev_base)
            dac, _ = load_run(a.dev)
            dsel = a.dev_cell or a.cell
            dcell = [c for c in sorted(set(dbc) & set(dac)) if dsel in c]
            dev = {"arm_score": sum(dac[c] for c in dcell),
                   "control_score": sum(dbc[c] for c in dcell), "n": len(dcell)}
            print(f"\n[6b] HELD-OUT (criterion 2: NO AGGREGATE REGRESSION, a gain is not required)")
            print(f"  control {dev['control_score']}/{dev['n']}   arm {dev['arm_score']}/{dev['n']}"
                  f"   net {dev['arm_score'] - dev['control_score']:+d}")

        # Criterion 3/4 evidence is read from the ARM'S OWN trajectory sidecar -- the same `asteps`
        # the firing count uses -- and translated into the canonical counter vocabulary by the BFCL
        # adapter. A family with no declared reading contributes NOTHING, leaving the criterion
        # PENDING rather than passing on silence.
        # `bsteps` is the CONTROL's trajectory: criterion 4's clears counter is a PAIRED quantity
        # (arm minus control on the same cases), because an arm that merely fails to prevent a clear
        # the control also made has added nothing. Without it the counter stays unmeasured and
        # criterion 4 is PENDING -- which is the honest verdict, not a reason to default it to zero.
        mech_tel, safety, mech_prov = mechanism_evidence(asteps, control_steps=bsteps)
        report = evaluate_criteria(_Paired(bc, ac, cell, gains, losses), dev=dev,
                                  safety=safety, telemetry=mech_tel)
        print(f"\n[6c] ACCEPTANCE: {'ACCEPTED' if report.accepted else 'NOT ACCEPTED'}")
        for c in report.criteria:
            print(f"  C{c.number} {c.name:<32} {c.verdict}")
            if not c.allows_accept:
                print(f"       {c.detail}")
                if c.remedy:
                    print(f"       remedy: {c.remedy}")
        for miss in mech_prov["missing"]:
            print(f"       ! evidence not recorded: {miss}")
        rec["acceptance"] = report.to_dict()
        rec["acceptance"]["mechanism_provenance"] = mech_prov
        rec["acceptance"]["dev"] = dev

        # A criterion that is PENDING is unresolved evidence, not a refutation. Passing
        # `accepted=False` for it would let `classify_residual` bank a NEGATIVE RESULT against a
        # candidate whose evidence was never gathered, so a blocking PENDING is reported through
        # `validity` instead -- the same channel the driver already uses for an unbelievable round.
        pending = [c for c in report.blocking if c.verdict == "PENDING_VALIDATION"]
        if pending:
            validity = Validity(
                denominator_ok=ok_denom, contamination_free=not prereq_fired, channel_ok=chan_ok,
                evidence_complete=False,
                missing_evidence=tuple(f"C{c.number} {c.name}: {c.detail}" for c in pending),
                detail=chan_msg)
            print(f"       -> BLOCKED, not refuted: {len(pending)} criterion(s) unresolved")
        term = classify_residual(
            family=top.key, search_outcome=outcome, support=top.support,
            evaluation={"candidates_evaluated": len(outcome.candidates),
                        "accepted": bool(report.accepted)},
            validity=validity, budget_remaining=not a.budget_exhausted)
        state = IMPROVED if term.outcome == "PROMOTED" else NO_BENEFIT
        print(f"  -> {term.outcome}"
              + ("  [NEGATIVE RESULT]" if term.is_negative_result
                 else "  [not a negative result -- the question stands]"))
        if not validity.ok:
            for why in validity.reasons():
                print(f"       ! {why}")
        rec["evaluation"] = {"n": len(cell), "control": b_n, "arm": a_n, "delta_pp": delta,
                             "gains": len(gains), "losses": len(losses), "net": net,
                             "denominator_ok": ok_denom, "firings": len(fired & set(cell)),
                             "gains_on_fired": len(on_fired), "state": state}
        rec["termination"] = term.as_dict()
        rec["state"] = term.outcome
        gv = global_verdict([term], budget_remaining=not a.budget_exhausted,
                            families_eligible=len(problems))
        print(f"\n[8] GLOBAL: {gv.verdict} -- {gv.detail}")
        rec["global"] = gv.as_dict()

        if term.outcome == "PROMOTED":
            # ---- RE-MINE: the point of the cycle ----------------------------------------------
            print(f"\n[7] PROMOTED -> RE-MINE the residual the promotion just created")
            afailed = sorted(c for c in cell if not ac[c])
            acases = [case_facts(c, asteps.get(c, []), a.cell, a.phase) for c in afailed[:a.limit]]
            adiags = attribute(acases, a.out / "after", a.live) if acases else []
            (a.out / "after").mkdir(exist_ok=True)
            before, after = residual_keys(diags), residual_keys(adiags)
            gone = sorted(set(before) - set(after))
            new = sorted(set(after) - set(before))
            print(f"  residual keys: {len(before)} before -> {len(after)} after")
            print(f"    DISAPPEARED ({len(gone)}):")
            for k in gone[:4]:
                print(f"      -{before[k]:3d}  {k[:72]}")
            print(f"    NEW ({len(new)}):")
            for k in new[:4]:
                print(f"      +{after[k]:3d}  {k[:72]}")
            rec["residual_shift"] = {"before": dict(before.most_common()),
                                     "after": dict(after.most_common()),
                                     "disappeared": gone, "new": new,
                                     "moved": bool(gone or new)}
            print(f"\n  RESIDUAL DISTRIBUTION MOVED: {bool(gone or new)}")
    elif a.results is None:
        # ONLY when neither measurement path ran. This branch used to fire unconditionally and
        # overwrite the propose phase's UNEVALUATED -- so a round that had correctly reported "arms
        # built, nothing measured" was relabelled AWAITING_EVALUATION on its way out, and a round that
        # had SELECTED on measurement via --results lost its outcome class entirely.
        print(f"\n[6] awaiting paired evaluation -- rerun with --results <measured results.json>")
        print(f"     (or --score <arm run dir> to score one arm's run directly)")
        rec.setdefault("state", EXT_UNEVALUATED)

    # ---- PERSIST THE OUTCOME ---------------------------------------------------------------------
    # The round's verdict outlives the round only if it is written down. Without this the discovery
    # lived in a per-round directory nothing read back, and the next round started from H0.
    #
    # WHICH STATE, and why the mapping is conservative: a verdict is recorded ONLY when the
    # measurement was believable. `validity.ok` is the gate, and a round that failed it lands in
    # PENDING_VALIDATION with the reason attached -- never MEASURED_NEGATIVE. That is the distinction
    # this project kept losing: R1's four prereq candidates were evaluated by an instrument that
    # eliminated their acting phase, which is not a refutation, and banking it as one would have
    # permanently excluded a mechanism worth measuring.
    if a.library is not None and a.score is not None and not _installed_spec(a.score):
        # NO SPEC, NO RECORD. A control arm installs nothing, and a run that kept no spec cannot be
        # re-executed from what we would write. Inventing an identity from the directory name would
        # put a record in the library that later rounds deduplicate against -- worse than no record.
        print("\n[9] NOT PERSISTED: the scored run recorded no controller spec "
              f"(looked for {', '.join(_SPEC_CANDIDATES)}). A control arm is expected to hit this; "
              "an ARM hitting it means the spec was not written and the round cannot be banked.")
        rec["persist_skipped"] = "no controller spec in the scored run"
    elif a.library is not None and a.score is not None:
        _persist = CandidateLibrary.load(a.library)
        _ev = rec.get("evaluation") or {}
        _valid = bool(_ev.get("denominator_ok")) and bool(locals().get("validity") and validity.ok)
        _why = "" if _valid else "; ".join(validity.reasons()) if locals().get("validity") else \
               "no validity record produced"
        _spec_installed = _installed_spec(a.score)
        _name = str(_spec_installed.get("name") or a.score.name)
        if _valid and rec.get("state") == "PROMOTED":
            _state = ACCEPTED
        elif _valid and _ev:
            _state = MEASURED_NEGATIVE
        else:
            _state = PENDING_VALIDATION
        _m = Measurement(
            arm_correct=int(_ev.get("arm") or 0), control_correct=int(_ev.get("control") or 0),
            n_scored=int(_ev.get("n") or 0),
            gains=tuple(gains) if _ev else (), losses=tuple(losses) if _ev else (),
            valid=_valid, invalid_reason=_why, firings=int(_ev.get("firings") or 0),
            detail=f"delta {_ev.get('delta_pp')}pp; gains_on_fired {_ev.get('gains_on_fired')}") \
            if _ev else None
        # A MEASURED_NEGATIVE/ACCEPTED record must carry a measurement; a PENDING one must not carry
        # a verdict it cannot support. Dropping the counts here is what keeps an invalid round from
        # being quoted later as a number.
        if _state == PENDING_VALIDATION and _m is not None and not _valid:
            _m = None
        try:
            _persist.upsert(CandidateRecord(
                name=_name,
                identity=ControllerIdentity(
                    boundary=str(_spec_installed.get("locus") or _spec_installed.get("boundary") or ""),
                    signal=str(_spec_installed.get("signal") or ""),
                    action=str(_spec_installed.get("action") or ""),
                    operator=str(_spec_installed.get("operator") or ""),
                    capability_id=str(_spec_installed.get("capability_id")
                                      or _spec_installed.get("variant") or "unspecified"),
                    phase=str(_spec_installed.get("phase") or a.phase)),
                spec=_spec_installed,
                state=_state,
                context=EvaluationContext(
                    incumbent_stack=tuple(_stack), incumbent_token=str(a.baseline or a.incumbent),
                    cell=a.cell, split=str(a.cell), model=os.environ.get("ANCHOROPT_MODEL", ""),
                    scored_phase=a.phase,
                    acting_phase=str(_spec_installed.get("phase") or "")),
                measurement=_m,
                provenance={"arm_run": str(a.score), "control_run": str(a.baseline or a.incumbent),
                            "out": str(a.out)},
                round_id=str(a.out.name),
                notes=() if _valid else (f"EVALUATION NOT BELIEVABLE: {_why}",)))
            _p = _persist.save(a.library)
            print(f"\n[9] PERSISTED {_name} as {_state} -> {_p}")
            print(f"    measured against stack {rec['stack_fingerprint']} "
                  f"({len(_stack)} controller(s))")
            if _state == PENDING_VALIDATION:
                print(f"    NOT a negative result -- it stays available for validation")
        except Exception as exc:
            # A persistence failure must be LOUD and must not destroy the round's own record, which is
            # already complete at this point. Silence here would recreate the original defect.
            print(f"\n[9] PERSIST FAILED ({type(exc).__name__}: {exc}) -- "
                  f"round record still written; library NOT updated")
            rec["persist_error"] = f"{type(exc).__name__}: {exc}"

    assert_no_oracle_leak()
    json.dump(rec, open(a.out / "cycle2.json", "w"), indent=2, default=str)
    print(f"\n{'=' * W}\nwrote {a.out / 'cycle2.json'}\n{'=' * W}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
