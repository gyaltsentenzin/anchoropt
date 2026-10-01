#!/usr/bin/env python3
"""E1: run the policy decision tree over every MINED signal (SIGNAL_POLICY_PLAN §8.1).

`triage_remedy.py` answers the tree for a signal named on the command line. This
runs it over candidates discovered from the corpus, and — the point of E1 — is
allowed to answer **uncertain** at any node.

                        intervene?
                    /              \\
                  no               yes
                  |                 |
                no-op         choose control
                             /              \\
                        advisory           active
                           |             /         \\
                       reprompt    veto/suppress  substitute/reroute

The output is a `candidate_policy_set`: the actions still worth measuring. A
singleton means attribution decided; a larger set means attribution PRUNED the
space and measurement takes over (§8.1.1). Both are successful outputs — only an
empty set, or a forced wrong singleton, is a failure. So the tree's job is to
shrink the experimental action space, not to always produce the final action.

Node evidence bars are deliberately asymmetric:

  intervene?   does the signal reliably predict failure?
  control?     can advisory control plausibly correct this from the evidence?
               NOT "was the instruction ignored" — in the G1 case the model
               COMPLIED with a harmful tool instruction, which is evidence that
               leaving agency with the model is unreliable here.
  active?      `suppress` needs only evidence the current action is bad;
               `reroute` needs POSITIVE evidence for the replacement, and even
               then is only ELIGIBLE for measurement, never selected here. That
               asymmetry is §0's measured -9.33pp reroute regression as a rule.

No hardcoded BFCL vocabulary: signals come from the corpus. `--score-known` in
error_attribution_study.py is the only place known strings appear.

Usage:
    python scripts/policy_tree.py --traj-dir <dir> [--phase prereq] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

YES, NO, UNCERTAIN = "YES", "NO", "UNCERTAIN"
ADVISORY, ACTIVE = "ADVISORY", "ACTIVE"

_ERRORISH = ("error", "not found", "full", "cannot", "fail", "too long",
             "exceed", "invalid", "unable", "must be")
# Anchored to a call position; a bare name+paren also matches prose inside
# argument values (see error_attribution_study.py for the measured failure).
_CALL_RE = re.compile(r"""(?:^|[\[\("'`,;]|\s)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(""")
_NAME_OK = re.compile(r"^(?:[a-z][a-z0-9]*_[a-z0-9_]+|get|set|add|list|search|retrieve|clear|update|append)$")

# A call that destroys state cannot be undone by a later retry; one that merely
# fails can. Used only for the "is the offending action destructive" test.
_DESTRUCTIVE = ("clear", "remove", "delete", "reset", "wipe", "truncate")

# Errors that FORCE an action rather than merely accompanying it. A capacity or
# size constraint leaves the model no alternative but to free space; a
# format/uniqueness complaint does not.
_CONSTRAINT_ERRORS = ("is full", "exceeds maximum size", "maximum size of",
                      "exceeds maximum length", "maximum length of",
                      "too long", "no space", "capacity")
_STATE_LOOKUPS = ("list_keys", "retrieve", "key_search", "get_", "retrieve_all")

# HARNESS faults, not model failures. A crash inside a tool implementation says
# nothing about the policy under test, but it reaches the sidecar as an ordinary
# error payload -- so _canon_error would happily turn it into a mineable "error
# contract" and a gate could be learned against a stack-trace string.
#
# Measured contamination that motivated this (plan §105): kv's
# archival_memory_key_search called BM25Plus([]) and raised "division by zero"
# whenever archival memory was empty. Because the G7 prompt instructs the model to
# search archival memory, the prompt INDUCED the fault -- 40x under prompt vs 10x
# under control, 41 cases, 18.3% of the measured residual.
_HARNESS_FAULTS = ("division by zero", "zerodivisionerror",
                   "error during execution:", "traceback (most recent call last)")


def is_harness_fault(text: str) -> bool:
    """True if a tool payload is a harness crash rather than a domain error."""
    low = str(text).lower()
    return any(sig in low for sig in _HARNESS_FAULTS)


def harness_fault_report(eps: List[Dict]) -> Dict:
    """Count episodes/steps carrying a harness fault, for a loud report at load."""
    n_eps, n_steps, sigs = 0, 0, Counter()
    for e in eps:
        hit = False
        for s in e.get("steps") or []:
            for t in (s.get("tool_results") or []):
                if is_harness_fault(t):
                    n_steps += 1
                    hit = True
                    for sig in _HARNESS_FAULTS:
                        if sig in str(t).lower():
                            sigs[sig] += 1
        if hit:
            n_eps += 1
    return {"episodes": n_eps, "steps": n_steps, "signatures": dict(sigs)}

_ROOT_DIR = str(Path(__file__).resolve().parent.parent)

# memory_kv_prereq_31-student-9 -> "student"
_SCEN_RE = re.compile(r"-([a-z_]+)-\d+$")


def _scenario_of(case_id: str) -> str:
    m = _SCEN_RE.search(case_id or "")
    return m.group(1) if m else ""


def load(traj_dir: Path, phase: str = "all",
         allow_harness_faults: bool = False) -> List[Dict]:
    eps = []
    for p in sorted(traj_dir.rglob("*.json")):
        try:
            eps.append(json.load(open(p)))
        except Exception as e:
            print(f"[warn] skip {p.name}: {e}", file=sys.stderr)
    if phase != "all":
        eps = [e for e in eps if e.get("phase") == phase]

    # Refuse to hand harness faults to the miner. Default is to FAIL LOUDLY:
    # a contaminated corpus silently produces a plausible-looking gate.
    rep = harness_fault_report(eps)
    if rep["episodes"]:
        msg = (f"[HARNESS FAULT] {rep['episodes']} episode(s) / {rep['steps']} step(s) in "
               f"{traj_dir} carry a harness crash, not a model failure: {rep['signatures']}. "
               f"Mining this corpus would learn a gate against a crash string. "
               f"Fix the harness and re-run, or pass allow_harness_faults=True to "
               f"acknowledge and drop them.")
        if not allow_harness_faults:
            raise RuntimeError(msg)
        print("[warn] " + msg, file=sys.stderr)
        eps = [e for e in eps
               if not any(is_harness_fault(t)
                          for s in (e.get("steps") or [])
                          for t in (s.get("tool_results") or []))]
        print(f"[warn] dropped to {len(eps)} clean episode(s)", file=sys.stderr)
    return eps


def _errorish(text: str) -> bool:
    return any(k in text.lower() for k in _ERRORISH)


def _calls_in(step: Dict) -> List[str]:
    out = []
    for m in _CALL_RE.finditer(str(step.get("decoded") or "")):
        if _NAME_OK.match(m.group(1)):
            out.append(m.group(1))
    return out


def _canon_error(text: str) -> Optional[str]:
    """Canonical short form of a tool error payload, for grouping.

    Strips JSON scaffolding and numbers so "maximum length of 300 characters" and
    "maximum length of 500 characters" group together — the signal is the contract
    that was violated, not the specific limit.

    Requires an explicit error MARKER, not merely an error-ish word somewhere in
    the payload. Regression: a successful retrieval whose content happened to read
    "...user introduction: Michael..." was admitted as an error contract and
    ranked 4th, because the loose check let any payload through once some other
    step in the episode had errored.
    """
    m = re.search(r'"error"\s*:\s*"([^"]+)"', text)
    if m:
        msg = m.group(1)
    elif re.match(r"^\s*\[?\s*['\"]?(?:Error|Exception)\b", text.strip(), re.I):
        msg = text
    else:
        return None
    msg = re.sub(r"\d+", "N", msg)
    msg = re.sub(r"[^A-Za-z0-9 ]+", " ", msg)
    msg = re.sub(r"\s+", " ", msg).strip().lower()
    return msg[:90] or None


# A tool result can be a FAILURE without being an ERROR. `_canon_error` requires an
# explicit error marker, so a payload that is syntactically a success but carries no
# usable information is invisible to it -- and therefore invisible to candidate mining.
#
# That blind spot is not hypothetical: the entire class was missed until a human read
# trajectories by hand (plan §105 / A1). Defined here on GENERAL SHAPE, deliberately
# without reference to any particular benchmark payload, so that whatever it re-derives
# is evidence about the detector rather than a restatement of what it was built from:
#
#   * an empty result collection                     -> nothing was found
#   * a scored collection whose scores are ALL at the floor (0.0) -> ranked, but by
#     nothing; the ranking carries no information about relevance
#
# A vacuous result is strictly more dangerous than an error, because the model has no
# syntactic cue that anything went wrong.
_FLOOR = 0.0


def _iter_result_collections(payload):
    """Yield (name, list) for every list-valued field in a parsed tool result."""
    # json, not _json: this module imports `json` plainly. The first version of this
    # function said `_json`, which raised NameError into the bare `except Exception`
    # below and made the detector silently return None for EVERY payload -- a guard
    # that swallows its own bugs. Narrowed to the parse errors actually expected.
    try:
        obj = json.loads(payload) if isinstance(payload, str) else payload
    except (ValueError, TypeError):
        return
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if isinstance(v, list):
            yield k, v


def vacuous_result_kind(payload) -> Optional[str]:
    """Classify a SUCCESSFUL-looking tool payload that carries no information.

    Returns "empty_collection", "all_floor_scores", or None. Never fires on a payload
    carrying an explicit error marker -- that is `_canon_error`'s job, and double-counting
    one event as two candidate kinds would inflate support.
    """
    s = payload if isinstance(payload, str) else str(payload)
    if _canon_error(s) is not None:
        return None
    for _name, lst in _iter_result_collections(s):
        if not lst:
            return "empty_collection"
        # A collection of (score, item) pairs whose scores are all at the floor.
        scores = []
        for item in lst:
            if isinstance(item, (list, tuple)) and item and isinstance(item[0], (int, float)):
                scores.append(float(item[0]))
        if scores and len(scores) == len(lst) and all(x == _FLOOR for x in scores):
            return "all_floor_scores"
    return None


def any_vacuous(results) -> Optional[str]:
    """vacuous_result_kind over a tool_results LIST, not its str().

    Load-bearing: a step's tool_results is a list, and `str(["{...}"])` is not valid JSON,
    so passing the stringified list made the detector return None for every payload. That
    silently defeated the "a vacuous call is not a working alternative" filter and let a
    destination resolving 0/24 episodes rank first.
    """
    if results is None:
        return None
    if isinstance(results, (list, tuple)):
        for r in results:
            k = vacuous_result_kind(r)
            if k is not None:
                return k
        return None
    return vacuous_result_kind(results)


# ── Candidate discovery ───────────────────────────────────────────────────────

def mine_candidates(eps: List[Dict], min_support: int) -> List[Dict]:
    """Discover candidate signals of two kinds, both from the corpus.

      error_contract — a recurring tool error string (the violated contract)
      reactive_call  — a call the model makes IN RESPONSE to a tool error
    """
    err_count: Counter = Counter()
    err_next: Dict[str, Counter] = defaultdict(Counter)
    # §40: per error contract, which alternative call SUCCEEDS afterwards. This is what a
    # post-execution reroute would dispatch, and it is mined from the corpus rather than
    # matched against any existing gate. Outcome-blind: call names + error-vs-not only.
    err_recovery: Dict[str, Counter] = defaultdict(Counter)
    # §41 INCUMBENT ENGAGEMENT. Raw support conflates two different things: an UNCOVERED
    # residual (no installed policy engages it) is discovery, while a COVERED-but-unsolved
    # one (a policy engages and failures remain) is refinement. Mixing them makes the
    # incumbent's own hardest cases outrank genuinely new signals purely on support.
    err_engaged: Dict[str, int] = defaultdict(int)
    err_recovered: Dict[str, List[bool]] = defaultdict(list)
    # §94 FAULT vs RECOVERY. The residual's identity is the call that FAILED; the calls that
    # follow are the model's remedy and must not be mistaken for it. Anchoring on the remedy is
    # what produced a G1 gate matching core_memory_clear -- 0 failures in 93 attempts -- while
    # every one of the 148 core-full errors was returned by core_memory_add (§93).
    err_failing: Dict[str, Counter] = defaultdict(Counter)      # fault locus
    err_recovery_seq: Dict[str, Counter] = defaultdict(Counter) # ordered remedy, e.g. clear>add
    err_seq_ok: Dict[str, Counter] = defaultdict(Counter)       # of those, how many resolved
    err_downstream: Dict[str, List[bool]] = defaultdict(list)   # did the episode end clean?
    call_total: Counter = Counter()
    call_post_err: Counter = Counter()
    call_forced: Counter = Counter()
    call_deliberate: Counter = Counter()
    call_effective: Counter = Counter()
    call_after: Dict[str, Counter] = defaultdict(Counter)
    ep_of_err: Dict[str, set] = defaultdict(set)

    for ep in eps:
        steps = ep.get("steps", [])
        err_idx = [i for i, s in enumerate(steps)
                   if _errorish(str(s.get("tool_results") or ""))]
        # Constraint errors and state lookups, for the necessity node.
        con_idx = [i for i, s in enumerate(steps)
                   if any(k in str(s.get("tool_results") or "").lower()
                          for k in _CONSTRAINT_ERRORS)]
        # §41: did an installed gate engage this error event? Any *_gate flag firing at
        # the error step or within 2 steps counts. Outcome-blind.
        for e in err_idx:
            key = _canon_error(str(steps[e].get("tool_results") or ""))
            if not key:
                continue
            for k in range(e, min(e + 3, len(steps))):
                if any(kk.endswith("_gate") and vv for kk, vv in steps[k].items()):
                    err_engaged[key] += 1
                    break
        # §40: successful recovery alternatives, keyed by the canonical error contract.
        for e in err_idx:
            key = _canon_error(str(steps[e].get("tool_results") or ""))
            if not key:
                continue
            failing = set(_calls_in(steps[e]))
            # §94: attribute the error to the call that RETURNED it. When decoded calls and
            # tool_results align positionally, blame precisely; otherwise blame the step's calls.
            _dec = steps[e].get("decoded") or []
            _res = steps[e].get("tool_results") or []
            _blamed = []
            if len(_dec) == len(_res) and _dec:
                for _c, _r in zip(_dec, _res):
                    if _errorish(str(_r)):
                        _m = _CALL_RE.finditer(str(_c))
                        _blamed += [x.group(1) for x in _m if _NAME_OK.match(x.group(1))]
            if not _blamed:
                _blamed = sorted(failing)
            for _b in _blamed:
                err_failing[key][_b] += 1
            # LEGACY (unchanged, §40): alternatives that succeed within e+3. Window and
            # semantics preserved exactly -- the ranker consumes this field, so widening it
            # here would silently reweight discovery.
            for k in range(e + 1, min(e + 3, len(steps))):
                res = str(steps[k].get("tool_results") or "")
                if not res or _errorish(res):
                    continue
                for tok in _calls_in(steps[k]):
                    if tok not in failing:      # a DIFFERENT call that worked
                        err_recovery[key][tok] += 1
            # §94 (new, separate scan): the ORDERED remedy. `retrieve_all>remove>add` and
            # `clear>add` are different strategies and a Counter of names cannot tell them apart.
            # A wider window is needed to see a 3-step remedy; it feeds ONLY the new fields.
            _seq = []
            _resolved = False
            for k in range(e + 1, min(e + 6, len(steps))):
                _cs = [t for t in _calls_in(steps[k]) if t not in ("<end_turn>",)]
                if not _cs:
                    continue
                _seq += _cs
                res = str(steps[k].get("tool_results") or "")
                if res and not _errorish(res):
                    if set(_cs) & failing:   # a call of the FAILING family worked again
                        _resolved = True
                        break
                if len(_seq) >= 4:
                    break
            if _seq:
                _sig = ">".join(_seq[:4])
                err_recovery_seq[key][_sig] += 1
                if _resolved:
                    err_seq_ok[key][_sig] += 1
            err_downstream[key].append(_resolved)
        for i, s in enumerate(steps):
            for tok in _calls_in(s):
                call_total[tok] += 1
                # STRICTLY preceding error: e == i would credit a call against
                # its own failed result (measured false positive).
                if any(1 <= i - e <= 2 for e in err_idx):
                    call_post_err[tok] += 1
                # ── necessity evidence (§19.2) ───────────────────────────────
                # forced: a CONSTRAINT error (not just any error) preceded it, so
                # the model had no alternative but to free space.
                if any(1 <= i - e <= 3 for e in con_idx):
                    call_forced[tok] += 1
                # deliberate: the model consulted state before acting, which is
                # what considered eviction looks like and gratuitous destruction
                # does not.
                if any(any(l in c for l in _STATE_LOOKUPS)
                       for j in range(max(0, i - 3), i) for c in _calls_in(steps[j])):
                    call_deliberate[tok] += 1
                # effective: did a same-family retry succeed shortly after?
                for j in range(i + 1, min(i + 4, len(steps))):
                    _t = str(steps[j].get("tool_results") or "").lower()
                    if any(("add" in c or "append" in c) for c in _calls_in(steps[j])):
                        if "error" not in _t:
                            call_effective[tok] += 1
                        break
        for e in err_idx:
            key = _canon_error(str(steps[e].get("tool_results") or ""))
            if not key:
                continue
            err_count[key] += 1
            ep_of_err[key].add(ep.get("case_id"))
            nxt = _calls_in(steps[e + 1]) if e + 1 < len(steps) else []
            err_next[key][nxt[0] if nxt else "<end_turn>"] += 1
            # Did a later step of the SAME kind succeed WITHOUT a destructive
            # action in between? The qualifier matters: after "core is full",
            # a core_memory_add succeeds only because a clear freed space. Counting
            # that as self-correction inverted the control node for G1's error
            # (96% "self-correct" -> ADVISORY) when it is the cascade itself.
            base = _calls_in(steps[e])
            recovered = False
            for j in range(e + 1, min(e + 5, len(steps))):
                if any(any(d in t for d in _DESTRUCTIVE) for t in _calls_in(steps[j])):
                    break  # recovery was purchased by destroying state
                later = str(steps[j].get("tool_results") or "")
                if base and set(_calls_in(steps[j])) & set(base) and not _errorish(later):
                    recovered = True
                    break
            err_recovered[key].append(recovered)
            for tok in nxt[:1]:
                call_after[tok][key] += 1

    cands = []
    for key, n in err_count.items():
        if n < min_support:
            continue
        rec = err_recovered[key]
        cands.append({
            "kind": "error_contract", "signal": key, "support": n,
            "episodes": len(ep_of_err[key]),
            "next_action": dict(err_next[key].most_common(5)),
            "self_correct_rate": round(sum(rec) / len(rec), 3) if rec else None,
            "self_correct_n": len(rec),
            # §40: alternatives that SUCCEED after this error -- what a post-execution
            # reroute would dispatch. Mined, not matched against any registry entry.
            "_recovery_alternatives": dict(err_recovery[key].most_common(5)),
            # §94: fault locus, ordered recovery strategies, and downstream resolution.
            # The intervention point is the failing action; recoveries only PROPOSE actions.
            "_failing_action": dict(err_failing[key].most_common(5)),
            "_recovery_sequences": dict(err_recovery_seq[key].most_common(5)),
            "_recovery_seq_resolved": dict(err_seq_ok[key].most_common(5)),
            "_downstream_resolved_rate": (
                round(sum(err_downstream[key]) / len(err_downstream[key]), 3)
                if err_downstream[key] else None),
            "engaged": err_engaged[key],
            "engagement_rate": round(err_engaged[key] / n, 3) if n else 0.0,
            "coverage": ("covered_but_unsolved"
                         if n and err_engaged[key] / n >= ENGAGEMENT_COVERED_RATE
                         else "uncovered"),
        })
    # §95 FAULT vs REMEDY per call. err_failing (§94) already knows which call returned each
    # error; err_recovery_seq knows which calls appear as remedy steps. A call that recovers
    # more than it fails is a REMEDY and must not become an intervention anchor.
    _fails_as: Counter = Counter()
    for _k, _fa in err_failing.items():
        for _c, _v in _fa.items():
            _fails_as[_c] += _v
    _recovers_as: Counter = Counter()
    for _k, _sq in err_recovery_seq.items():
        # §95.1 A call earns no remedy credit for being RETRIED after its own failure.
        # The fault locus of THIS residual is excluded from its own recovery tally: in
        # `clear>add` the add is the resumed original intent, and in `add>add>add` they are
        # retries. Counting them made core_memory_add -- the true G1 fault locus -- look like
        # a remedy (fails 362 / recovers 684) and would have suppressed the §95 guard's whole
        # purpose by rejecting the correct anchor.
        _loci = set(err_failing.get(_k, {}))
        for _seq, _v in _sq.items():
            for _tok in set(_seq.split(">")) - _loci:
                _recovers_as[_tok] += _v

    # ── vacuous_result candidates ─────────────────────────────────────────
    # Keyed by (kind, tool) so the candidate names the CALL whose result was vacuous --
    # that is the locus an anchor would attach to, exactly as error_contract names the
    # contract. Support counts EVENTS; `_episodes` counts distinct episodes, because
    # concentration hides in rates (one episode firing 20x once dragged apparent
    # acceptance from 85% to 59%).
    vac_ev: Counter = Counter()
    vac_eps: Dict[str, set] = defaultdict(set)
    vac_fail_eps: Dict[str, set] = defaultdict(set)
    # Which calls FOLLOW the vacuous result, and how often the episode then resolves.
    # This is the reroute-destination evidence, mined exactly as err_recovery is for
    # error contracts -- a reroute needs POSITIVE evidence for its replacement.
    vac_recov: Dict[str, Counter] = defaultdict(Counter)
    vac_recov_ok: Dict[str, Counter] = defaultdict(Counter)
    # SELF-CORRECTION, with err_recovered's exact semantics: the SAME call retried
    # without error within e+5, aborting if a destructive call intervenes (recovery
    # bought by destroying state is not self-correction). Required because node_control
    # reads self_correct_rate and returns UNCERTAIN when it is None -- so omitting it
    # silently short-circuits policy_set to ['no-op','reprompt'] BEFORE the action
    # branch is reached, and a REROUTE_CANDIDATE verdict never gets used. That is not a
    # judgement about reroute; it is a missing field masquerading as one.
    vac_recovered: Dict[str, List[bool]] = defaultdict(list)
    for ep in eps:
        _cid = ep.get("case_id")
        _bad = not (ep.get("outcome") or {}).get("valid", False)
        for st in ep.get("steps") or []:
            _calls = _calls_in(st)
            for _i, _r in enumerate(st.get("tool_results") or []):
                vk = vacuous_result_kind(_r)
                if vk is None:
                    continue
                _tool = _calls[_i] if _i < len(_calls) else (_calls[0] if _calls else "?")
                _key = "%s:%s" % (vk, _tool)
                vac_ev[_key] += 1
                vac_eps[_key].add(_cid)
                if _bad:
                    vac_fail_eps[_key].add(_cid)
                # Recovery alternatives, with EXACTLY err_recovery's semantics (§40) --
                # a DIFFERENT call within e+3 whose result is not errorish. Matching the
                # legacy definition is required, not cosmetic: node_recovery_action tests
                # this field against RECOVERY_MIN_SUCCESSES, and my first version counted
                # plain occurrences over an unbounded window, which is a different and
                # more permissive quantity. A node cannot be reused if the field it reads
                # means something else.
                _steps = ep.get("steps") or []
                _e = None
                for _idx, _s2 in enumerate(_steps):
                    if _s2 is st:
                        _e = _idx
                        break
                if _e is None:
                    continue
                _here = set(_calls_in(st))
                # self-correction: same call, retried clean, within e+5
                _sc = False
                for _j in range(_e + 1, min(_e + 5, len(_steps))):
                    if any(any(_d in _t for _d in _DESTRUCTIVE)
                           for _t in _calls_in(_steps[_j])):
                        break
                    _later_res = str(_steps[_j].get("tool_results") or "")
                    if (_here and set(_calls_in(_steps[_j])) & _here
                            and not _errorish(_later_res)
                            and any_vacuous(_steps[_j].get("tool_results")) is None):
                        # A retry that returns ANOTHER vacuous result is not a recovery.
                        _sc = True
                        break
                vac_recovered[_key].append(_sc)
                for _k2 in range(_e + 1, min(_e + 3, len(_steps))):
                    _raw = _steps[_k2].get("tool_results")
                    _res = str(_raw or "")
                    if not _res or _errorish(_res):
                        continue
                    # A VACUOUS result is not a working alternative. _errorish cannot see
                    # this: a zero-score search does not error, so it was being counted as
                    # a successful recovery -- and since a vacuous locus is most often
                    # followed by ANOTHER vacuous search, that made the most frequent
                    # "alternative" one that resolves 0/24 episodes. Same fault-vs-remedy
                    # confusion §94 warns about, one level down: the model's next move is
                    # not evidence that the move works. Pass the RAW list, not str(list).
                    if any_vacuous(_raw) is not None:
                        continue
                    for _tok in _calls_in(_steps[_k2]):
                        if _tok in _here:
                            continue
                        vac_recov[_key][_tok] += 1
                        if not _bad:
                            vac_recov_ok[_key][_tok] += 1
    for _key, _n in vac_ev.items():
        _ne = len(vac_eps[_key])
        if _ne < min_support:
            continue
        _nf = len(vac_fail_eps[_key])
        # Field names CONFORM to the error_contract contract that run_tree/node_* already
        # consume -- `episodes`, `coverage`, `_downstream_resolved_rate`, `_failing_action`,
        # `_recovery_alternatives`. A new candidate KIND must fit the existing tree, not
        # bypass it: emitting a novel field set made node_intervene raise KeyError on
        # 'episodes', i.e. my own kind crashed the machinery that was already there.
        cands.append({
            "kind": "vacuous_result", "signal": _key,
            # support = EVENTS, matching error_contract's meaning of the field.
            "support": _n,
            # episodes = distinct episodes, which is what node_intervene tests.
            "episodes": _ne,
            "episodes_failed": _nf,
            "failure_rate": round(_nf / _ne, 3) if _ne else 0.0,
            "vacuity": _key.split(":", 1)[0],
            "tool": _key.split(":", 1)[1],
            # The call that RETURNED the vacuous result is the fault locus, exactly as
            # err_failing names the call that returned an error (§94).
            "_failing_action": {_key.split(":", 1)[1]: _ne},
            # Downstream: did the episode end clean? 1 - failure_rate by construction.
            "_downstream_resolved_rate": round(1.0 - (_nf / _ne), 3) if _ne else None,
            # Recovery attestation, mined the same way err_recovery is: which calls follow
            # the vacuous result, and how often the episode then resolves.
            "_recovery_alternatives": dict(vac_recov[_key].most_common(5)),
            "_recovery_resolved": dict(vac_recov_ok[_key].most_common(5)),
            # No installed gate engages a vacuous result today, so it is discovery, not
            # refinement. Stated rather than defaulted.
            "engaged": 0,
            "engagement_rate": 0.0,
            "coverage": "uncovered",
            # node_control reads these; see vac_recovered above.
            "self_correct_rate": (round(sum(vac_recovered[_key]) / len(vac_recovered[_key]), 3)
                                  if vac_recovered[_key] else None),
            "self_correct_n": len(vac_recovered[_key]),
            "next_action": dict(vac_recov[_key].most_common(5)),
        })

    for tok, n in call_total.items():
        if n < min_support:
            continue
        pe = call_post_err[tok]
        cands.append({
            "kind": "reactive_call", "signal": tok, "support": n,
            # §95: the fault/remedy discriminator. post_error_rate cannot tell "causes
            # failure" from "responds to failure"; these two counts can.
            "_fails_as_locus": _fails_as.get(tok, 0),
            "_appears_as_recovery": _recovers_as.get(tok, 0),
            "_fault_vs_remedy": (
                "remedy" if _recovers_as.get(tok, 0) > _fails_as.get(tok, 0)
                else "fault" if _fails_as.get(tok, 0) > 0
                else "indeterminate"),
            "post_error": pe, "post_error_rate": round(pe / n, 3) if n else 0.0,
            "destructive": any(k in tok for k in _DESTRUCTIVE),
            "forced_rate": round(call_forced[tok] / n, 3) if n else 0.0,
            "deliberate_rate": round(call_deliberate[tok] / n, 3) if n else 0.0,
            "effective_rate": round(call_effective[tok] / n, 3) if n else 0.0,
            "triggered_by": dict(call_after[tok].most_common(3)),
        })
    cands.sort(key=lambda c: -c["support"])
    return cands


# ── Candidate RANKING within one TREE ITERATION ───────────────────────────────
#
# The unit of the algorithm is a TREE ITERATION, not a gate:
#
#   T0  the incumbent: global prompt, zero anchors
#   T1  the first policy-tree search over T0's residuals; inside it, ranked CANDIDATES
#       R1, R2, R3, ... are evaluated IN ORDER. A candidate that PASSES measurement
#       becomes an ANCHOR and gets an A number: the winner of T1 is A1. Naming a
#       candidate A1 before measuring it pre-supposes the outcome.
#   T2  a fresh search over the residuals of (T0 + A1, the anchor accepted in T1)
#
# THE RANKING IS CONDITIONAL ON THE CURRENT INCUMBENT. Once any Ri is ACCEPTED the
# iteration ENDS: the accepted anchor changes the residual distribution, the exposure sets,
# the ranking, and every downstream counterfactual, so the remaining Rj mined from the old
# world are stale. Learning a whole list once and installing it sequentially would treat
# rank-3 evidence gathered under T0 as if it described the T0+A1 world. It does not.
#
# A REJECTED candidate is different: rejecting it leaves the incumbent unchanged, so the
# rest of the ranking still holds and evaluation continues within the same iteration.
#
# "Top-ranked" also has to be well defined, and before this it was not:
#
#   * `support` counted EVENTS for error_contract/reactive_call but EPISODES for
#     vacuous_result, so a single sort mixed unlike units;
#   * reactive_call occupied the top four slots on the A0 corpus, but a reactive call is
#     the model's REMEDY, not a failure locus -- ranking it as a target is the §94/§95
#     category error that produced a G1 gate matching core_memory_clear with 0 failures
#     in 93 attempts.
#
# The rule below is deliberately dull and auditable. It ranks only LOCI, in a single unit
# (distinct failing episodes), and states every exclusion.

# A locus must implicate at least this many distinct FAILING episodes to be worth an arm.
# Not a coverage bar: coverage is explicitly NOT a rejection criterion (a local policy is
# allowed to be local). This only says an arm needs enough exposed cases to measure.
RANK_MIN_FAILING_EPISODES = 5


def _locus_rows(cands):
    """Normalise mined candidates to comparable ranking rows, or explain the exclusion.

    Returns (rows, excluded) where each row is a dict with a common unit:
        failing_episodes  distinct episodes that FAILED and exhibit this locus
        precision         failing_episodes / episodes exhibiting it
    """
    rows, excluded = [], []
    for c in cands:
        kind, sig = c["kind"], str(c.get("signal"))
        if kind == "reactive_call":
            # §94/§95: a call that appears as the model's recovery is not a fault locus.
            excluded.append((sig, kind, "reactive_call is a REMEDY, not a failure locus"))
            continue
        if kind == "vacuous_result":
            eps, nf = c["support"], c.get("episodes_failed", 0)
        elif kind == "error_contract":
            eps = c.get("episodes") or 0
            # error_contract has no per-episode failure count; the downstream resolved
            # rate is its closest analogue, so derive the failing count from it rather
            # than inventing one.
            dr = c.get("_downstream_resolved_rate")
            nf = int(round(eps * (1.0 - dr))) if (eps and dr is not None) else eps
        else:
            excluded.append((sig, kind, "unrecognised kind"))
            continue
        if nf < RANK_MIN_FAILING_EPISODES:
            excluded.append((sig, kind, "only %d failing episode(s) < %d"
                             % (nf, RANK_MIN_FAILING_EPISODES)))
            continue
        rows.append({
            "kind": kind, "signal": sig,
            "failing_episodes": nf, "episodes": eps,
            "precision": round(nf / eps, 3) if eps else 0.0,
            "coverage": c.get("coverage", "uncovered"),
            "_cand": c,
        })
    # Primary key: how much failure the locus implicates, in episodes. Tie-break on
    # precision (a locus that is ALWAYS fatal is a cleaner target than a noisy one),
    # then on the signal name so the order is deterministic across runs.
    rows.sort(key=lambda r: (-r["failing_episodes"], -r["precision"], r["signal"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows, excluded


def rank_candidates(cands, rejected=(), iteration="T1", accepted_this_iteration=None):
    """Ranked loci for ONE tree iteration, over the residuals of the CURRENT incumbent.

    `cands` must be mined from the current incumbent's trajectories. Ranking a candidate
    set mined under an older incumbent is the error this signature exists to prevent.

    `rejected` -- signals already evaluated and rejected WITHIN THIS ITERATION. Rejection
    leaves the incumbent unchanged, so the rest of the ranking still holds and the loop
    continues. Filtering happens AFTER ranking so ranks stay stable and history is not
    renumbered.

    `accepted_this_iteration` -- if set, the iteration is CLOSED. `next` becomes None and
    the remaining candidates are marked stale, because the accepted anchor changed the
    residual distribution they were ranked against. The caller must install the anchor,
    replay the counterfactual world, re-mine, and start the next iteration fresh.
    """
    rows, excluded = _locus_rows(cands)
    rejected = set(rejected)
    closed = accepted_this_iteration is not None
    for r in rows:
        r["iteration"] = iteration
        r["rejected"] = r["signal"] in rejected
        r["accepted"] = (r["signal"] == accepted_this_iteration)
        # Stale = ranked in this iteration but never evaluated, and now un-evaluable here
        # because the incumbent moved underneath it.
        r["stale"] = closed and not r["accepted"] and not r["rejected"]
    unresolved = [] if closed else [r for r in rows if not r["rejected"]]
    return {
        "iteration": iteration,
        "ranked": rows,
        "unresolved": unresolved,
        "next": unresolved[0] if unresolved else None,
        "excluded": excluded,
        "closed": closed,
        "accepted": accepted_this_iteration,
        "next_action": (
            "ACCEPTED %s -- iteration %s is CLOSED. Install the anchor, replay the "
            "counterfactual world, re-mine and re-rank, then start the next iteration."
            % (accepted_this_iteration, iteration) if closed
            else "evaluate %s (rank %d of %d unresolved)"
            % (unresolved[0]["signal"], unresolved[0]["rank"], len(unresolved))
            if unresolved else
            "every ranked candidate in %s is rejected; no anchor accepted this iteration. "
            "The incumbent is unchanged, so this iteration is EXHAUSTED -- widen the miner "
            "or accept that this residual is not reachable by the frozen action space."
            % iteration),
    }


# ── The tree ──────────────────────────────────────────────────────────────────

def node_intervene(c: Dict, min_support: int) -> Tuple[str, str]:
    """Does this signal reliably predict/precede failure?"""
    if c["support"] < min_support:
        return UNCERTAIN, f"support {c['support']} below floor {min_support}"
    if c["kind"] == "reactive_call":
        r = c["post_error_rate"]
        # §95 GUARD, before any rate test. A high post_error_rate is definitionally true of a
        # recovery action, so on its own it cannot license an intervention. Require that the
        # call actually FAILS more than it RECOVERS. core_memory_clear scored 0 failures in 93
        # attempts against 89 recovery appearances and was still admitted at YES -- that is how
        # G1's gate came to suppress the model's own clear->add eviction (§93).
        fvr = c.get("_fault_vs_remedy")
        if fvr == "remedy":
            return NO, (f"RECOVERY action, not a fault: fails as error locus "
                        f"{c.get('_fails_as_locus', 0)}x but appears as a recovery step "
                        f"{c.get('_appears_as_recovery', 0)}x. Recoveries PROPOSE candidate "
                        f"actions; they must not become the intervention anchor (§95). "
                        f"Anchor on the failing action instead.")
        if fvr == "indeterminate":
            return UNCERTAIN, ("no evidence this call is ever the error locus, and none that it "
                               "is a recovery -- cannot tell fault from remedy (§95)")
        if r >= 0.8:
            return YES, (f"{c['post_error']}/{c['support']} ({r:.0%}) of calls follow a tool "
                         f"error within 2 steps; fault locus {c.get('_fails_as_locus', 0)}x "
                         f"vs recovery {c.get('_appears_as_recovery', 0)}x")
        if r <= 0.2:
            return NO, (f"only {r:.0%} of calls follow an error — not error-associated "
                        f"(a tool the model simply uses)")
        return UNCERTAIN, f"{r:.0%} post-error — neither clearly reactive nor clearly not"
    # error_contract: the error IS the failure, so the question is whether it recurs
    if c["episodes"] >= 2:
        return YES, f"{c['support']} occurrences across {c['episodes']} episodes"
    return UNCERTAIN, f"{c['support']} occurrences but only {c['episodes']} episode(s)"


NECESSARY, UNFORCED = "NECESSARY", "UNFORCED"

# Thresholds for the necessity node. Deliberately conservative toward NECESSARY:
# wrongly gating a forced action breaks the system outright (the model can no longer
# store anything), whereas wrongly declining to gate merely leaves a defect in place
# for a later iteration.
NECESSITY_FORCED_RATE = 0.8       # a constraint error preceded ~always
NECESSITY_DELIBERATE_RATE = 0.8   # state consulted before acting


def node_necessary(c: Dict) -> Tuple[str, str]:
    """Was the action FORCED by state/constraints, rather than gratuitous? (§19.2)

    This node exists because the tree previously went straight from "predicts
    failure" to "choose a control", and `destructive + post-error => suppress`
    cannot distinguish two opposite situations:

      core_memory_clear      destroys state gratuitously  -> gate it
      archival_memory_remove forced by a capacity limit    -> gating it BREAKS the
                                                              system

    Measured: 17/17 archival_memory_remove events followed a capacity error and
    17/17 consulted state first; 0/17 were unnecessary. The 94% post-error rate that
    ranked it #1 is exactly what a NECESSARY reaction looks like — high
    error-association is evidence of necessity as readily as of pathology, and
    `intervene?` reads it as pathology only.

    A NECESSARY verdict does not end the analysis: the action may be forced while the
    CHOICE within it is still wrong (class B), which is an argument/eviction-policy
    question rather than a suppression one.
    """
    if c["kind"] != "reactive_call":
        # An error contract is a constraint report, not an action the model chose.
        return UNFORCED, "not a chosen action — an error contract"
    forced = c.get("forced_rate")
    delib = c.get("deliberate_rate")
    if forced is None:
        return UNFORCED, "no necessity evidence available"
    if forced >= NECESSITY_FORCED_RATE:
        why = (f"{forced:.0%} of calls follow a CONSTRAINT error "
               f"(capacity/size), so the action was forced")
        if delib is not None and delib >= NECESSITY_DELIBERATE_RATE:
            why += f"; state consulted first in {delib:.0%} — considered, not gratuitous"
        return NECESSARY, why
    if forced <= 0.2:
        return UNFORCED, (f"only {forced:.0%} follow a constraint error — not forced "
                          f"by capacity")
    return UNFORCED, (f"{forced:.0%} follow a constraint error — mixed, treated as "
                      f"unforced pending better evidence")


def node_control(c: Dict) -> Tuple[str, str]:
    """Can ADVISORY control plausibly correct this from available evidence?

    Deliberately not "was the instruction ignored". The G1 evidence is that the
    model COMPLIED with a harmful tool instruction 27/27, which shows leaving
    agency with the model under the existing context is unreliable — a different
    and more general property.
    """
    if c["kind"] == "reactive_call":
        trig = c.get("triggered_by") or {}
        harmful = [k for k in trig if "please" in k or "clear" in k or "shorten" in k]
        if c.get("destructive") and c["post_error_rate"] >= 0.8:
            why = (f"model performs a DESTRUCTIVE action in {c['post_error_rate']:.0%} of "
                   f"post-error steps")
            if harmful:
                why += f"; the tool error itself directs it ({harmful[0][:48]!r})"
            return ACTIVE, why + " — advisory control cannot be relied on"
        return UNCERTAIN, "no evidence either way on whether advice would suffice"

    # error_contract: does the model recover on its own after this error?
    r = c.get("self_correct_rate")
    if r is None:
        return UNCERTAIN, "no retry observed after this error"
    if r >= 0.7:
        return ADVISORY, (f"model self-corrects after {r:.0%} of these errors "
                          f"(n={c['self_correct_n']}) — advisory may suffice, or no-op")
    if r <= 0.2:
        return ACTIVE, (f"model self-corrects after only {r:.0%} (n={c['self_correct_n']}) "
                        f"— cannot recover unaided")
    return UNCERTAIN, f"self-corrects {r:.0%} of the time (n={c['self_correct_n']}) — mixed"


def node_recovery_action(c: Dict) -> Tuple[str, str]:
    """For an ERROR CONTRACT: is there an attested RECOVERY substitute? (§40)

    The post-execution analogue of node_active_action. `suppress` is undefined once the
    call has run, but dispatching an ALTERNATIVE TOOL as the next recovery action is
    defined -- that is what the incumbent already does.

    Same asymmetry as node_active_action, and the same discipline: the substitute is mined
    from the corpus (which alternative actually SUCCEEDS after this error), never matched
    against a hand-written gate. Attestation makes it eligible for MEASUREMENT; it is
    never selected here.

    Outcome-blind: call names plus whether the tool result was an error. No reward labels.
    """
    alts = c.get("_recovery_alternatives") or {}
    if not alts:
        return "NO_SUBSTITUTE", "no alternative succeeds after this error in the corpus"
    top, n = max(alts.items(), key=lambda kv: kv[1])
    if n < RECOVERY_MIN_SUCCESSES:
        return "NO_SUBSTITUTE", (f"best alternative {top!r} succeeds only {n}x "
                                 f"(< {RECOVERY_MIN_SUCCESSES}) — too weak to instantiate")
    return "REROUTE_CANDIDATE", (f"alternative {top!r} SUCCEEDS {n}x after this error — "
                                 f"ELIGIBLE for measurement, not selected")


# A substitute must succeed at least this often to be worth instantiating. Set from the
# observed spread rather than tuned: attested substitutes in this corpus range from 5
# (memory_append, 1% success rate — genuinely weak) to 177 (archival_memory_add after
# core-full, 58%). A floor of 10 admits the strong ones and excludes the noise.
RECOVERY_MIN_SUCCESSES = 10

# §41: an error contract counts as COVERED when an installed gate engages at least this
# fraction of its events. Measured on the G1 incumbent the split is stark rather than
# borderline -- 92% and 71% for the two covered contracts vs 10% for the uncovered one --
# so the exact cutoff is not load-bearing. DISCLOSED as a Phase-I development heuristic.
ENGAGEMENT_COVERED_RATE = 0.5


def node_active_action(c: Dict) -> Tuple[str, str]:
    """suppress vs reroute-candidate.

    ASYMMETRIC BY DESIGN: `suppress` needs only evidence the current action is
    bad. `reroute` needs POSITIVE attestation of a replacement — and even then is
    only *eligible for measurement*, never selected here, because §0 measured a
    plausible-looking substitute at -9.33pp.
    """
    trig = c.get("triggered_by") or {}
    alt: Counter = Counter()
    for _key, n in trig.items():
        pass
    # Alternatives = what else the model did after the same errors that trigger
    # this call. Sourced from the error candidates, passed in by the caller.
    alts = c.get("_attested_alternatives") or {}
    if not alts:
        return "SUPPRESS", "no supported substitute attested in the corpus"
    top, n = max(alts.items(), key=lambda kv: kv[1])
    return "REROUTE_CANDIDATE", (f"alternative {top!r} attested {n}x — ELIGIBLE for "
                                 f"measurement, not selected")


# Which actions are MECHANICALLY possible at each hook. A signal's kind
# determines its trigger_kind, and that constrains the action space before any
# evidence is considered:
#
#   reactive_call   -> the offending call is in hand PRE-execution, so it can be
#                      cancelled (suppress) or replaced (reroute).
#   error_contract  -> the error has ALREADY been returned. There is no call left
#                      to cancel, so suppress is not merely unsupported, it is
#                      undefined. Only advisory or an argument-level transform of
#                      the retry can apply.
#
# Regression this encodes: without the constraint, four error_contract signals
# were emitted with candidate_policy_set={suppress} and ranked ABOVE G1, because
# "no substitute attested" collapsed them to a confident singleton that cannot be
# executed.
# CORRECTED (§40). `error_contract` previously excluded `reroute`, on the stated grounds
# that "there is no call to cancel ... so suppress/reroute are UNDEFINED". Only the
# suppress half of that is mechanically true.
#
#   suppress at post-execution   TRUE IMPOSSIBILITY -- the call already ran; there is
#                                nothing left to cancel.
#   reroute  at post-execution   FEASIBLE, and already IMPLEMENTED. `on_core_full_rerouted`
#                                is remedy=reroute / trigger_kind=post_exec_result and is
#                                the G1 INCUMBENT: it takes a core_memory_add that already
#                                returned a core-full error and dispatches a synthesized
#                                archival_memory_add instead. `on_forced_key_search` does
#                                the same after "Key not found".
#
# So reroute post-execution means "select an alternative tool as the next RECOVERY action",
# which is a different operation from undoing the failed call -- and it was excluded by a
# definition choice mislabelled as a mechanical property.
#
# The evidence requirement is unchanged and does the real gating: node_active_action demands
# POSITIVE ATTESTATION of a substitute ("no supported substitute attested in the corpus" ->
# SUPPRESS), so widening the feasible set cannot invent a reroute where the corpus shows no
# alternative that works.
_ACTIONS_BY_KIND = {
    "reactive_call": {"no-op", "reprompt", "suppress", "reroute"},
    "error_contract": {"no-op", "reprompt", "reroute", "transform"},
    # A vacuous result is POST-EXECUTION, exactly like an error contract: the call has
    # already run and returned, so `suppress` is UNDEFINED -- there is no pending call to
    # cancel. Omitting this entry let the dict fall back to the reactive_call default and
    # admit suppress, which feasible_actions' own docstring rules out for a returned
    # result. `transform` is excluded too: transform rewrites an OUTGOING argument, and
    # here the argument was fine -- the store simply held nothing matching it.
    "vacuous_result": {"no-op", "reprompt", "reroute"},
}


def feasible_actions(kind: str, necessary: Optional[str] = None) -> List[str]:
    """The actions PHYSICALLY EXECUTABLE for this trigger. Feasibility only.

    This is the boundary between what humans specify and what AnchorOpt learns:

      humans      define the action space and these feasibility rules
      attribution PRUNES to the feasible set (this function)
      measurement CHOOSES among the feasible

    Attribution must not rank the feasible options. Earlier it did — ruling reprompt
    "weak" and transform "strongest" for the entry-length signal from a retry
    statistic — which is hand-designing the answer from a plausible story rather than
    measuring. Several such stories were wrong today (substitution rejected at 2.91pp;
    suppress predicted to beat reroute and lost; archival_memory_remove judged a defect
    at 0/17 unnecessary).

    Feasibility is a mechanical property, not a judgement:
      reactive_call  — the offending call exists pre-execution, so it can be cancelled
                       (suppress) or redirected (reroute)
      error_contract — the error has ALREADY returned, so `suppress` is UNDEFINED: there
                       is no pending call to cancel. `reroute` IS defined and implemented
                       (on_core_full_rerouted, the G1 incumbent): it dispatches an
                       alternative tool as the next RECOVERY action rather than undoing the
                       failed one. Requires an attested substitute -- an evidence question
                       settled by node_active_action, not a feasibility one.
      NECESSARY      — the action was forced by a constraint; cancelling it breaks the
                       system, so structural cancellation is infeasible regardless of
                       hook.
    """
    if necessary == NECESSARY:
        return ["no-op", "transform"]
    if kind == "error_contract":
        return ["no-op", "reprompt", "reroute", "transform"]
    if kind == "vacuous_result":
        # Post-execution like error_contract, so suppress is undefined. transform is also
        # out: the outgoing argument was well-formed, the store just held nothing matching.
        return ["no-op", "reprompt", "reroute"]
    return ["no-op", "reprompt", "suppress", "reroute"]


def policy_set(intervene: str, control: str, action: Optional[str],
               kind: str = "reactive_call") -> List[str]:
    """Collapse the three verdicts into the actions still worth measuring.

    Intersected with what the hook can mechanically do (_ACTIONS_BY_KIND).
    """
    allowed = _ACTIONS_BY_KIND.get(kind, _ACTIONS_BY_KIND["reactive_call"])
    if intervene == NO:
        proposed = ["no-op"]
    elif intervene == UNCERTAIN:
        proposed = ["no-op", "reprompt"]
    elif control == ADVISORY:
        # Self-correcting signals: no-op is a live outcome, per §8.4.
        proposed = ["no-op", "reprompt"]
    elif control == UNCERTAIN:
        proposed = ["no-op", "reprompt", "transform"]
    elif action == "SUPPRESS":
        proposed = ["suppress"]
    elif action == "REROUTE_CANDIDATE":
        # For an error contract there is no call to cancel, so the measured pair is
        # reroute vs the advisory baseline. For a pre-execution call it stays
        # suppress vs reroute.
        proposed = (["reroute", "reprompt"]
                    if kind in ("error_contract", "vacuous_result")
                    else ["suppress", "reroute"])
    elif action == "NO_SUBSTITUTE":
        # Post-execution with nothing attested to dispatch: advisory only.
        proposed = ["no-op", "reprompt"]
    else:
        proposed = ["reprompt", "suppress"]
    out = [a for a in proposed if a in allowed]
    # Never return an empty set: if the hook forbids everything proposed, the
    # honest answer is that measurement must start from the advisory baseline.
    return out or ["no-op", "reprompt"]


def run_tree(cands: List[Dict], min_support: int) -> List[Dict]:
    # Attest alternatives: for each reactive call, what ELSE did the model do
    # after the same triggering errors?
    err_next = {c["signal"]: c.get("next_action", {})
                for c in cands if c["kind"] == "error_contract"}
    for c in cands:
        if c["kind"] != "reactive_call":
            continue
        alts: Counter = Counter()
        for errkey in (c.get("triggered_by") or {}):
            for act, n in (err_next.get(errkey) or {}).items():
                if act != c["signal"] and act != "<end_turn>":
                    alts[act] += n
        c["_attested_alternatives"] = dict(alts)

    out = []
    for c in cands:
        iv, iv_why = node_intervene(c, min_support)
        ct = ct_why = None
        ac = ac_why = None
        nec = nec_why = None
        if iv == YES:
            nec, nec_why = node_necessary(c)
        if iv == YES and nec == NECESSARY:
            # Forced action: the question is no longer "which control", it is
            # "was the CHOICE within the action right". Suppressing a forced action
            # breaks the system, so the structural rungs are removed here.
            ct, ct_why = None, "not asked — action was forced by constraints"
            ac, ac_why = None, None
        elif iv == YES:
            ct, ct_why = node_control(c)
            # suppress/reroute are only defined where a CALL can be intercepted
            # before execution; asking the node for an error_contract is a
            # category error (see _ACTIONS_BY_KIND).
            if ct == ACTIVE and c["kind"] == "reactive_call":
                ac, ac_why = node_active_action(c)
            elif ct == ACTIVE:
                # §40: post-execution. suppress is undefined, but dispatching an
                # attested alternative as the next RECOVERY action is not -- ask the
                # corpus whether such a substitute exists.
                ac, ac_why = node_recovery_action(c)
        ps = (["no-op", "transform"] if nec == NECESSARY
              else policy_set(iv, ct, ac, c["kind"]))
        conf = ("decided" if len(ps) == 1 else
                "pruned" if len(ps) <= 2 else "wide")
        out.append({
            "signal": c["signal"], "kind": c["kind"], "support": c["support"],
            # §41: support alone is NOT opportunity -- a high-support contract an installed
            # gate already engages is refinement, not discovery.
            "coverage": c.get("coverage", "uncovered"),
            "engagement_rate": c.get("engagement_rate"),
            "engaged": c.get("engaged"),
            # Public copy so a consumer can compare a proposal against what history
            # ATTESTED. Its absence made the teacher step label archival_memory_add
            # "novel vs mined" -- the mined destination with 49 successes (§48.4 again).
            "recovery_alternatives": dict(c.get("_recovery_alternatives") or {}),
            "intervene": iv, "intervene_evidence": iv_why,
            "necessary": nec, "necessary_evidence": nec_why,
            # FEASIBLE = what is executable (attribution's job).
            # candidate_policy_set = feasible AND not evidently pointless.
            # Neither is a ranking; measurement decides among them.
            "feasible_actions": feasible_actions(c["kind"], nec),
            "control": ct, "control_evidence": ct_why,
            "active_action": ac, "active_action_evidence": ac_why,
            "candidate_policy_set": ps, "confidence": conf,
            "attested_alternatives": c.get("_attested_alternatives") or {},
        })
    # Rank by how much the tree NARROWED the space and how strong the control
    # verdict is — not by raw support, and not by singleton-ness alone.
    #
    # A {reprompt} singleton is a weak pick: it is where the hook forbids active
    # control, so "decided" understates the uncertainty. An ACTIVE verdict on a
    # pre-execution call is the strong case even when two actions remain, because
    # both remaining actions are structural. Without this, four advisory
    # error_contract rows outranked the G1 signal.
    # §63 LEARNED FROM MEASURED OUTCOMES, not from reading the gates. Joining the manual
    # registry to every arm we have actually measured (8 measurements) says the predictive
    # property is NOT the remedy-family label and NOT any structural field:
    #
    #   reroute WITH an attested substitute (>=10 mined successes)  -> +4.00 ACCEPT, +0.32 safe
    #   reprompt (re-ask only, no substitute)                       -> -3.85 REJECT
    #   transform (rewrite, no attested target)                     -> -3.21, -1.28 REJECT
    #
    # Two structural features (backends==2, n_subs==2) separate the ACCEPT from the rest
    # PERFECTLY and are nonetheless discarded as n=1 coincidences of the single accepted
    # gate: they are collinear with each other, and that gate's second backend had ZERO
    # exposure on the held-out corpus (§57.2), so it cannot have caused the win. Fitting to
    # them would be fitting to one data point.
    #
    # What survives is computable BEFORE any measurement, and the miner already computes it
    # (node_recovery_action) -- it was merely REPORTED rather than ranked on. So this is not
    # a new feature, it is an existing one promoted into the ordering.
    def _substitute_strength(r):
        """Successes of the best attested substitute for this residual; 0 if none."""
        alts = r.get("recovery_alternatives") or {}
        try:
            return max(alts.values()) if alts else 0
        except (TypeError, ValueError):
            return 0

    def rank_key(r):
        ps = set(r["candidate_policy_set"])
        no_op_only = ps == {"no-op"}
        active_ready = bool(ps & {"suppress", "reroute", "transform"})
        # Bucketed, not raw, so a large count cannot outweigh coverage or control: the
        # evidence supports "has an attested substitute" vs "does not", not a fine gradient
        # over 8 measurements.
        sub_bucket = 0 if _substitute_strength(r) >= RECOVERY_MIN_SUCCESSES else 1
        return (
            # §41: UNCOVERED first, else the incumbent's hardest cases outrank genuinely
            # new signals on support alone -- measured: ranks 2-3 had support 186/184 at
            # 92%/71% engagement, while the real target had 220 at 10%.
            r.get("coverage") == "covered_but_unsolved",
            no_op_only,                       # no-op last
            r.get("necessary") == NECESSARY,  # forced actions are not defects
            not active_ready,                 # structural candidates first
            sub_bucket,                       # §63: attested substitute before none
            r["control"] != ACTIVE,           # ACTIVE control before advisory/uncertain
            len(ps),                          # narrower sets first
            -r["support"],
        )
    out.sort(key=rank_key)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True, type=Path)
    ap.add_argument("--phase", default="all", choices=["all", "prereq", "query"])
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--top", type=int, default=12)
    # ── OPTIONAL teacher proposals (§45 tier 2) — DEFAULT OFF ────────────────
    # History-only is the default setting, so enabling the teacher must be an explicit
    # act rather than something a future reader has to remember to switch off. Teacher
    # candidates are ADDITIVE: they never replace, reorder or veto a mined candidate,
    # and they carry provenance so a downstream consumer can tell them apart.
    ap.add_argument("--teacher-model", default=None,
                    help="OPTIONAL (§45 tier 2, default OFF): also ask a teacher to "
                         "propose reroute destinations+arguments for the top uncovered "
                         "residual. Purely ADDITIVE -- mined candidates are unchanged, "
                         "and any teacher candidate is tagged source=teacher and must "
                         "clear its own frozen specification before measurement.")
    ap.add_argument("--teacher-backend", default="vector",
                    help="backend whose tool schemas are shown to the teacher")
    ap.add_argument("--teacher-endpoint", default=None,
                    help="OpenAI-compatible base URL for a LOCALLY served operator model "
                         "(e.g. an on-cluster granite). With this set, --teacher-model is "
                         "the served-model-name. Swapping the LLM is configuration, never "
                         "a code change.")
    ap.add_argument("--ledger", type=Path,
                    help="Evidence ledger JSON. Candidates are matched against it FIRST: "
                         "existing signals update their evidence instead of being "
                         "rediscovered, and learned/rejected signals are not re-proposed "
                         "unless materially new evidence appears.")
    ap.add_argument("--ledger-iteration", type=int, default=None,
                    help="Iteration number to stamp; default = bump the ledger's counter.")
    a = ap.parse_args()

    eps = load(a.traj_dir, a.phase)
    if not eps:
        print(f"No episodes under {a.traj_dir} (phase={a.phase})", file=sys.stderr)
        return 1
    npre = sum(1 for e in eps if e.get("phase") == "prereq")
    print(f"corpus: {len(eps)} episodes (prereq={npre}, query={len(eps)-npre})")
    print(f"support floor: {a.min_support}\n")

    cands = mine_candidates(eps, a.min_support)
    rows = run_tree(cands, a.min_support)

    # ── Evidence ledger: match against prior knowledge BEFORE proposing ───────
    led = None
    if a.ledger:
        # Load by path, NOT `from anchoropt.evidence_ledger import ...`: the
        # package __init__ pulls in injection_engine, whose PEP-604 `str | None`
        # annotations die on Python 3.9 — and this script must run on a login node
        # with no eval stack (same constraint as error_attribution_study.py).
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_evidence_ledger", Path(_ROOT_DIR) / "anchoropt" / "evidence_ledger.py")
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        led = _mod.EvidenceLedger(a.ledger)
        it = a.ledger_iteration if a.ledger_iteration is not None else led.begin_iteration()
        scenarios = sorted({_scenario_of(e.get("case_id", "")) for e in eps} - {""})
        seen_ids = set()
        for r in rows:
            # should_propose BEFORE observe: the comparison must be against prior
            # evidence, not against the row we are about to write in.
            propose, why = led.should_propose(r)
            rec, created = led.observe(r, iteration=it, scenarios=scenarios)
            seen_ids.add(rec["id"])
            r["_ledger"] = {
                "known": not created,
                "status": rec["status"],
                "drift": rec.get("drift"),
                "drift_reason": rec.get("drift_reason"),
                "observations": rec["observations"],
                "cumulative_occurrences": rec["occurrences"],
                "tested_policies": [m["policy"] for m in rec["policies_tested"]],
                "propose": propose, "reason": why,
            }
        # Signals the corpus NO LONGER produces cannot be seen by mining, so
        # absence is swept for explicitly — that is how a gate's own eliminated
        # signal gets classified `resolved` instead of staying `stable` forever.
        gone = led.mark_resolved_if_absent(seen_ids, iteration=it)
        if gone:
            print(f"[ledger] {len(gone)} previously-seen signal(s) absent this "
                  f"iteration -> resolved:")
            for g in gone[:5]:
                print(f"    {g['signal'][:60]}  ({g['drift_reason']})")
        # Only propose what the ledger has not settled.
        rows = [r for r in rows if r["_ledger"]["propose"]] + \
               [r for r in rows if not r["_ledger"]["propose"]]

    for r in rows[:a.top]:
        print("=" * 76)
        print(f"signal: {r['signal']}   [{r['kind']}, support={r['support']}]")
        print(f"  intervene:      {r['intervene']:<10} {r['intervene_evidence']}")
        if r.get("necessary"):
            print(f"  necessary?      {r['necessary']:<10} {r['necessary_evidence']}")
        if r["control"]:
            print(f"  control:        {r['control']:<10} {r['control_evidence']}")
        else:
            print(f"  control:        {'—':<10} (not reached)")
        if r["active_action"]:
            print(f"  active_action:  {r['active_action']:<10} {r['active_action_evidence']}")
        print(f"  feasible actions: {{{', '.join(r.get('feasible_actions') or [])}}}"
              f"   (attribution PRUNES; measurement chooses)")
        print(f"  candidate_policy_set: {{{', '.join(r['candidate_policy_set'])}}}"
              f"   [{r['confidence']}]")
        lg = r.get("_ledger")
        if lg:
            seen = (f"KNOWN (obs #{lg['observations']}, {lg['cumulative_occurrences']} "
                    f"cumulative)" if lg["known"] else "NEW")
            print(f"  ledger:         {seen}, status={lg['status']}"
                  + (f", drift={lg['drift']}" if lg.get("drift") else "")
                  + (f", tested={lg['tested_policies']}" if lg["tested_policies"] else ""))
            if lg.get("drift_reason") and lg.get("drift") not in (None, "new"):
                print(f"                  ({lg['drift_reason']})")
            if not lg["propose"]:
                print(f"  -> NOT PROPOSED: {lg['reason']}")

    print("=" * 76)
    # E2's pick must respect the ledger: a signal already learned or rejected is
    # not selectable, or the loop would relitigate it every iteration.
    dec = [r for r in rows
           if r["control"] == ACTIVE
           and r.get("necessary") != NECESSARY
           and set(r["candidate_policy_set"]) & {"suppress", "reroute", "transform"}
           and r.get("_ledger", {}).get("propose", True)]
    print(f"\n{len(rows)} candidates: "
          f"{len(dec)} decided-actionable, "
          f"{sum(1 for r in rows if r['confidence']=='pruned')} pruned, "
          f"{sum(1 for r in rows if r['confidence']=='wide')} wide, "
          f"{sum(1 for r in rows if r['candidate_policy_set']==['no-op'])} no-op")
    if dec:
        t = dec[0]
        ps = t["candidate_policy_set"]
        print(f"\nE2 would select: {t['signal']!r}")
        print(f"  candidate_policy_set = {{{', '.join(ps)}}}")
        if len(ps) == 1:
            print(f"  -> measure {ps[0]} vs no-op (attribution decided the action)")
        else:
            print(f"  -> measure {' vs '.join(ps)} vs no-op "
                  f"(attribution pruned to {len(ps)}; measurement picks)")
        print("  (candidates for measurement, NOT ground truth)")
    else:
        print("\nNo decided-actionable signal — E2 would measure the top pruned set.")

    # ── optional teacher augmentation (additive, never overriding) ──────────
    if getattr(a, "teacher_model", None):
        try:
            from teacher_augment import augment_with_teacher
            tprops = augment_with_teacher(
                rows, traj_dir=a.traj_dir, backend=a.teacher_backend,
                model=a.teacher_model,
                endpoint=getattr(a, "teacher_endpoint", None))
        except Exception as e:                    # never fail mining over the teacher
            print(f"\n[teacher] SKIPPED: {type(e).__name__}: {e}")
            tprops = []
        if tprops:
            print("\n" + "=" * 76)
            print("TEACHER-PROPOSED CANDIDATES (additive; provenance=teacher)")
            print("=" * 76)
            for t in tprops:
                mp = ",".join(o["op"] for o in t["argument_mapping"]) or "-"
                print("  %-24s %-9s %-22s %s"
                      % (t["destination_tool"], mp, t["status"],
                         ("novel vs mined" if t.get("novel_vs_mined") else "also mined")))
                if t.get("reason"):
                    print("      %s" % t["reason"])
            print("  NOTE: these do NOT change the mined ranking above. A teacher")
            print("  candidate requires its own frozen specification before measurement.")
            for r in rows:
                if r.get("_teacher_for_this_signal"):
                    r["teacher_candidates"] = tprops

    if led is not None:
        led.save()
        sm = led.summary()
        print(f"\nledger: {a.ledger}  iteration={sm['iterations']}  "
              f"signals={sm['signals']}  {sm['by_status']}")
        skipped = [r for r in rows if not r["_ledger"]["propose"]]
        if skipped:
            print(f"  {len(skipped)} signal(s) suppressed as already settled")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_episodes": len(eps), "min_support": a.min_support,
                   "candidates": rows}, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
