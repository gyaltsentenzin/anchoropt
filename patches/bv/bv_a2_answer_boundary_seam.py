#!/usr/bin/env python3
"""A GENERIC controller seam at the ANSWER boundary, reading prior retrieval confidence from history.

WHY A NEW SEAM IS NEEDED, and why it is not a new framework
-----------------------------------------------------------
A2's condition is "the model is about to answer and its retrievals came back weak". Neither existing
`post_generation_pre_exec` path can express it:

  * the ZERO-CALL path (`_query_tool_calls == 0`) reaches only steps that proposed NO tool call. A2's
    population DID retrieve -- they retrieved and got weak results -- so this path never sees them.
  * the CALL-FILTERING path builds `_ust` per proposed call and its own comment states the rule:
    "Nothing post-execution is available here by construction -- that is the point of the boundary,
    and a field that does not exist yet must not be supplied." `best_similarity` is a post-execution
    fact, so supplying it there would violate the boundary discipline this codebase enforces.

THE PRECEDENT THIS COPIES. The A9-U gate already fires at the answer boundary and reads A9-R's own
telemetry from `trajectory_history` (`_a9u_strong_xcm`), on the stated ground that the terminal answer
must exist before the state is knowable. Reading COMPLETED steps is backward-looking, so it introduces
no future information: `trajectory_history` holds finished steps only, and the same argument is already
made in the call-filtering path's `container_full` supply.

So this seam is the A9-U pattern generalised from one hardcoded anchor to any declared controller:
same window, same history-only inputs, no new boundary.

WHAT IT SUPPLIES, and nothing more
----------------------------------
    best_similarity              top-1 score of the LAST retrieval in this turn
    retrieval_attempts           how many retrieves ran
    searched_other_container     did the turn touch both core and archival
    other_container_nonempty     does the container it did NOT read from hold entries
    proposes_read                True -- a retrieval did happen (that is why we are here)
    saw_all_of_other_container   did it call *_retrieve_all on the other container

`other_container_nonempty` and `saw_all_of_other_container` exist because the classification of R12's
regressions showed "never searched archival" does NOT discriminate (6 of 8 GAINS share that state), and
one loss had already called `retrieve_all()` on both containers -- no reprompt can help a model that has
already seen everything. A predicate needs to be able to exclude that case.

SCORE PARSING. `_xcm_entries` matches `[0-9.]+` and so cannot see a NEGATIVE similarity; the real corpus
contains -0.013. This seam parses signed scores, because a predicate testing "below threshold" that
silently cannot observe the lowest scores of all is the same declared-vs-supplied defect one layer down.

IDEMPOTENCE. Guarded on the MARKER this patch inserts, never on the anchor.
Run from a private directory, NOT /tmp: `/tmp/inspect.py` on this cluster shadows stdlib `inspect`.
"""

from __future__ import annotations

import ast
import pathlib
import sys

MARKER = "# [anchoropt-patch:a2_answer_boundary_seam]"
HELPER_MARKER = "# [anchoropt-patch:a2_history_helpers]"

HELPER_ANCHOR = '''def _a9u_strong_xcm(history):
'''

HELPER_BLOCK = HELPER_MARKER + '''
#: Signed similarity scores. `_XCM_SCORE_RE` and `_xcm_entries` both match `[0-9.]+`, so neither can
#: observe a NEGATIVE score -- and the corpus contains -0.013. A "below threshold" predicate that
#: cannot see the lowest scores is the declared-vs-supplied defect one layer down, so this is separate
#: rather than a change to the frozen A9 helpers.
_A2_SIGNED_SCORE_RE = re.compile(r'"similarity_score"\\s*:\\s*(-?[0-9.]+)')

#: The signals THIS seam supplies evidence for. A controller declaring anything else is skipped here
#: with a reason and left to the boundary that does supply it. Without this the seam dispatches to
#: whichever installed controller fires first, and a neighbour at the same locus answers for a
#: condition on a field this state does not carry.
_A2_SEAM_SIGNALS = frozenset({"retrieval_similarity_below_threshold"})


def _a2_retrieval_state(history):
    """Retrieval facts for THIS turn, from COMPLETED steps only.

    Backward-looking by construction: `trajectory_history` carries finished steps, so nothing here is
    future information. Same argument the A9-U gate and the call-filtering path's `container_full`
    supply already make.

    Returns (best_similarity, attempts, surfaces_read, retrieve_all_surfaces).
    """
    best = None
    attempts = 0
    surfaces = set()
    all_surfaces = set()
    for rec in history or ():
        dec = [str(c) for c in (rec.get("decoded") or [])]
        res = [str(r) for r in (rec.get("tool_results") or [])]
        for i, call in enumerate(dec):
            if "retrieve" not in call and "search" not in call:
                continue
            attempts += 1
            surf = "secondary" if "archival" in call else ("primary" if "core" in call else None)
            if surf:
                surfaces.add(surf)
                if "retrieve_all" in call:
                    all_surfaces.add(surf)
            body = res[i] if i < len(res) else ""
            scores = [float(x) for x in _A2_SIGNED_SCORE_RE.findall(body)]
            if scores:
                best = max(scores)          # LAST retrieval wins: the state at the answer boundary
    return best, attempts, surfaces, all_surfaces


def _a2_other_container_entries(involved_instances, surfaces_read):
    """How many entries the container it did NOT read from currently holds.

    Unwrapped through `._store` for object-backed stores, the same way
    `anchoropt.mechanisms.lossless_eviction._live_container` does. Returns None when unreadable --
    which a predicate must be able to tell apart from "empty".
    """
    insts = ((involved_instances or {}).values()
             if isinstance(involved_instances, dict) else (involved_instances or []))
    want = "archival_memory" if "secondary" not in surfaces_read else "core_memory"
    for inst in insts:
        holder = getattr(inst, want, None)
        if holder is None:
            continue
        inner = getattr(holder, "_store", None)
        store = inner if isinstance(inner, dict) else (holder if isinstance(holder, dict) else None)
        if store is not None:
            return len(store)
    return None


''' + HELPER_ANCHOR

# The seam goes in the SAME window as the A9-U gate, immediately before the terminal
# `step_record["status"] = "answer_end_turn"`.
SEAM_ANCHOR = '''                        step_record["status"] = "answer_end_turn"
'''

SEAM_BLOCK = '''                        ''' + MARKER + '''
                        # GENERIC CONTROLLER PATH at the ANSWER boundary.
                        #
                        # Same window as the A9-U and B1/F2 gates above, for the reason both record:
                        # the terminal answer must exist before "about to answer on a weak retrieval"
                        # is knowable. Inputs come from COMPLETED steps only, so no future information
                        # reaches the decision. No anchor flag is read and no trigger is encoded here:
                        # the predicate decides and the text is the controller's own eta.
                        if (not self.disable_gates and not _a2_reprompted
                                and step_count < self.max_steps_per_turn):
                            try:
                                from anchoropt.runtime_hook import installed as _ainst
                                _actls = [c for c in _ainst("post_generation_pre_exec")
                                          if _phase_eligible(c, str(rollout_tag) == "snap")]
                            except Exception as _ae0:
                                _actls = []
                                step_record["answer_phase_error_gate"] = "%s: %s" % (
                                    type(_ae0).__name__, _ae0)
                            if _actls:
                                _abest, _aatt, _asurf, _aall = _a2_retrieval_state(
                                    trajectory_history)
                                # Only a turn that actually RETRIEVED is in scope. A zero-call step is
                                # the other path's business, and claiming this one for it would make
                                # two controllers answer for one condition.
                                if _aatt > 0:
                                    _aother = _a2_other_container_entries(
                                        involved_instances, _asurf)
                                    # The surface it did NOT read from, named the same way
                                    # `_a2_retrieval_state` names them.
                                    _aother_surface = ("secondary" if "secondary" not in _asurf
                                                       else "primary")
                                    _ast = {
                                        "boundary": "post_generation_pre_exec",
                                        "phase": ("prereq" if str(rollout_tag) == "snap"
                                                  else "query"),
                                        "step_index": step_record.get("step"),
                                        "site": "answer_boundary",
                                        "best_similarity": _abest,
                                        "retrieval_attempts": int(_aatt),
                                        "searched_other_container": len(_asurf) > 1,
                                        "other_container_entries": _aother,
                                        "other_container_nonempty": bool(_aother),
                                        # Did it already dump the OTHER container wholesale? If so no
                                        # reprompt can help -- it has seen every entry. One loss in
                                        # R12 was exactly this (retrieve_all on both containers).
                                        "saw_all_of_other_container": bool(
                                            _aother_surface in _aall),
                                        "saw_all_of_any_container": bool(_aall),
                                        "proposes_read": True,
                                        "has_generation": True,
                                        "proposes_tool_call": False,
                                        "proposes_write": False,
                                        "proposes_clear": False,
                                        "proposes_remove": False,
                                        # REQUIRED, and its ABSENCE was a live defect. The zero-call
                                        # signal reads `int(state.get("tool_calls_so_far", 0)) == 0`,
                                        # so an omitted field DEFAULTS TO ZERO and that signal fires
                                        # here on every episode -- even after two retrievals. Measured:
                                        # 24 firings attributed to the zero-call controller at
                                        # best_similarity up to 0.548, which is not its condition at
                                        # all. Any signal declared at this boundary must find every
                                        # field it reads, or it answers on a default.
                                        "tool_calls_so_far": int(_query_tool_calls or 0),
                                    }
                                    step_record["a2_state_best_similarity"] = _abest
                                    step_record["a2_state_attempts"] = int(_aatt)
                                    step_record["a2_state_other_entries"] = _aother
                                    step_record["a2_state_searched_other"] = len(_asurf) > 1
                                    # DISPATCH BY DECLARED SIGNAL, never "first that fires".
                                    #
                                    # This seam exists to supply ONE signal's evidence
                                    # (retrieval confidence at the answer boundary). Other
                                    # controllers are installed at this same locus -- the zero-call
                                    # reprompt among them -- and iterating blindly let one of them
                                    # answer for a condition this seam was not built to give it. That
                                    # is the same defect class as `installed(locus)[0]`: a controller
                                    # selected by position or by iteration order rather than by name.
                                    #
                                    # A controller whose signal this seam does not supply is SKIPPED
                                    # with a reason, so its exclusion is attributable rather than
                                    # silent, and it is left to the boundary that does supply it.
                                    _ahit = None
                                    _askipped = []
                                    for _ac in _actls:
                                        # THE SIGNAL LIVES ON `.spec`, not on the object. The loaded
                                        # controller is a SpecPredicate exposing only
                                        # (name, phase, eta, spec, fires_on) -- `getattr(c, "signal")`
                                        # is ABSENT and returned "" for every controller, which made a
                                        # name-based allowlist skip ALL of them and fire nothing. Read
                                        # the spec, and fall back to the predicate's declared_signal.
                                        _aspec = getattr(_ac, "spec", None) or {}
                                        _asig = str(_aspec.get("signal")
                                                    or (_aspec.get("predicate") or {}).get(
                                                        "declared_signal")
                                                    or getattr(_ac, "signal", "") or "")
                                        if _asig not in _A2_SEAM_SIGNALS:
                                            _askipped.append("%s(%s)" % (
                                                getattr(_ac, "name", "?"), _asig or "no-signal"))
                                            continue
                                        try:
                                            if _ac.fires_on(_ast):
                                                _ahit = _ac
                                                break
                                        except Exception as _ae:
                                            step_record["a2_error_gate"] = "%s: %s" % (
                                                type(_ae).__name__, _ae)
                                    if _askipped:
                                        step_record["a2_skipped_other_signal"] = list(_askipped)
                                    if _ahit is not None:
                                        _amsg = str((getattr(_ahit, "eta", {}) or {}).get(
                                            "instruction") or "").strip()
                                        if _amsg:
                                            _a2_reprompted = True
                                            step_record["a2_gate"] = True
                                            step_record["a2_controller"] = getattr(
                                                _ahit, "name", "controller")
                                            step_record["a2_injected_chars"] = len(_amsg)
                                            step_record["a2_response_before"] = str(
                                                step_record.get("assistant_response") or "")[:2000]
                                            inference_data = (
                                                self.handler
                                                ._add_next_turn_user_message_prompting(
                                                    inference_data,
                                                    [{"role": "user", "content": _amsg}]))
                                            trajectory_history.append(step_record)
                                            step_count += 1
                                            continue
                                        step_record["a2_declined"] = "controller has no eta.instruction"
                                    else:
                                        # Declines are RECORDED: a silent refusing branch makes a
                                        # criterion unpassable (anchoropt-unmeasurable-guard-paths).
                                        step_record["a2_declined"] = (
                                            "no installed controller fired on best_similarity=%s "
                                            "attempts=%d other_entries=%s" % (
                                                _abest, _aatt, _aother))
                                else:
                                    step_record["a2_declined"] = "no retrieval in this turn"
''' + SEAM_ANCHOR

# The once-per-turn latch, declared beside the existing one.
# 12-space indent: the latch lives in the per-TURN scope beside `_xcm_fired` and `_a9u_fired`, not in
# the deeper step scope. A2 gets its OWN latch for the reason those two comments already give -- a
# shared flag lets one gate's firing suppress another's on the very turns where both belong.
LATCH_ANCHOR = '''            _a9u_fired = False
'''


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bv_a2_answer_boundary_seam.py <path to anchoropt/memory_evaluator.py>")
        return 2
    p = pathlib.Path(sys.argv[1])
    src = p.read_text()

    if MARKER in src:
        print(f"already applied ({MARKER} present) -- no change")
        return 0

    out = src
    # 1. helpers
    if HELPER_ANCHOR not in out:
        print("FAIL: _a9u_strong_xcm anchor not found -- wrong target?")
        return 1
    out = out.replace(HELPER_ANCHOR, HELPER_BLOCK, 1)

    # 2. the once-per-turn latch. `_a9u_fired` is initialised per TURN; put ours beside it.
    if LATCH_ANCHOR not in out:
        print("FAIL: _a9u_fired latch anchor not found")
        return 1
    n_latch = out.count(LATCH_ANCHOR)
    out = out.replace(LATCH_ANCHOR, LATCH_ANCHOR + '''            # A2: once per turn, its own latch for the same reason the two above are separate.
            _a2_reprompted = False
''', 1)
    print(f"  latch inserted beside _a9u_fired ({n_latch} occurrence(s) found, first patched)")

    # 3. the seam itself
    if SEAM_ANCHOR not in out:
        print("FAIL: answer_end_turn anchor not found")
        return 1
    out = out.replace(SEAM_ANCHOR, SEAM_BLOCK, 1)

    try:
        ast.parse(out)
    except SyntaxError as exc:
        print(f"FAIL: patched file does not parse ({exc}); NOT written")
        return 1

    bak = p.with_suffix(p.suffix + ".pre_a2_seam")
    if not bak.exists():
        bak.write_text(src)
        print(f"backup: {bak}")
    p.write_text(out)
    print(f"patched {p}")
    print(f"  markers: {MARKER} , {HELPER_MARKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
