"""Evidence ledger: persistent learning memory across gate-learning iterations.

SIGNAL_POLICY_PLAN §8 runs mine -> triage -> measure -> install -> **re-mine**. Without
memory, every iteration rediscovers and re-analyses the same signals from scratch,
re-proposes policies already measured and rejected, and cannot tell "this signal
is new" from "this signal is the one we settled two iterations ago".

This is that memory. Deliberately **structured and deterministic** — a JSON file
keyed by signal identity, not free-form LLM recall. Every field is something a
script computed, so a later iteration can compare numbers rather than re-read prose.

Per signal:

    identity      (kind, normalized pattern) -> stable id
    occurrences   total count, summed across iterations
    support       independent episodes AND scenarios (concentration matters:
                  §Stage 2's "100% precise on 2 episodes is not a signal")
    causal        analyses already performed, so they are not redone
    self_correct  rate + n, per the causal definition in §10
    verdict       latest policy-tree verdict + candidate_policy_set
    policies      every policy already measured, with its gain/regression
    status        unresolved | learned | rejected_no_op

Status governs re-proposal: `learned` and `rejected_no_op` signals are NOT
re-proposed unless **materially new evidence** appears (see
`is_materially_new`) — which is what stops the loop relitigating settled
questions while still allowing a genuine distribution shift to reopen one.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = 3

STATUS_UNRESOLVED = "unresolved"
STATUS_LEARNED = "learned"
STATUS_REJECTED = "rejected_no_op"
# A STRONG signal for which no currently-available policy cleared the acceptance
# bar. Distinct from `rejected_no_op`, which means the signal itself is not worth
# intervening on: this means the signal is real and we do not yet have an action
# that fixes it. Suppressed from immediate re-selection so a hard signal cannot
# block the loop, but eligible to reopen when the ACTION SPACE grows.
STATUS_DEFERRED = "unresolved_no_policy"
# A candidate whose LOCAL effect met the acceptance bar on the selection corpus,
# but for which no available confirmation corpus can resolve an effect that size.
# Distinct from every other status:
#   learned      — effect reproduced above a threshold the corpus COULD detect
#   rejected     — effect absent where the corpus COULD have detected it
#   deferred     — strong signal, no working policy in the current action space
#   UNDERPOWERED — the corpus cannot resolve the effect; the result carries no
#                  information either way
# Without this, an underpowered null is recorded as `rejected` and a real candidate
# is discarded on evidence that never existed. Measured instance: reroute's
# +11.54pp targeted gain against a test-split threshold of ±24.03pp (n=25).
STATUS_UNDERPOWERED = "underpowered_candidate"
_STATUSES = (STATUS_UNRESOLVED, STATUS_LEARNED, STATUS_REJECTED, STATUS_DEFERRED,
             STATUS_UNDERPOWERED)

# Statuses that suppress immediate re-selection. `unresolved` is absent by design:
# an unresolved signal is exactly what the loop should keep working on.
# UNDERPOWERED is included: re-selecting it would re-run a measurement already
# known to be inconclusive at this corpus size. It reopens on MORE EXPOSED CASES.
_SETTLED = (STATUS_LEARNED, STATUS_REJECTED, STATUS_DEFERRED, STATUS_UNDERPOWERED)

# ── Action-space versioning ──────────────────────────────────────────────────
# The set of remedy families the loop can actually install. Recorded on every
# measurement and on every deferral so the ledger can distinguish
#   "already failed under THIS action set"  from
#   "worth retrying because new actions are now available".
# Bump ACTION_SPACE_VERSION when a family is added (e.g. `transform`, or a
# semantic-suppress that also covers substitute calls like core_memory_remove).
ACTION_SPACE_VERSION = 1
ACTION_SPACE = ("no-op", "reprompt", "suppress", "reroute")


def action_space_id(families=None, version=None) -> str:
    """Stable identifier for an action space: version + sorted family list."""
    fams = sorted(families if families is not None else ACTION_SPACE)
    v = ACTION_SPACE_VERSION if version is None else version
    return "v%s:%s" % (v, ",".join(fams))

# ── Cross-iteration evidence comparison ──────────────────────────────────────
# Each signal keeps BOTH historical evidence (what we knew) and current-iteration
# evidence (what the freshly generated corpus shows). Comparing them classifies
# the signal, and the classification — not a raw occurrence count — decides
# whether to reopen it for triage.
DRIFT_STABLE = "stable"            # same evidence; do not re-analyze
DRIFT_RESOLVED = "resolved"        # support/association materially DROPPED
DRIFT_STRENGTHENED = "strengthened"  # broader support or stronger association
DRIFT_CHANGED = "changed"          # mechanism shifted; reopen for triage
DRIFT_NEW = "new"
_DRIFTS = (DRIFT_NEW, DRIFT_STABLE, DRIFT_RESOLVED, DRIFT_STRENGTHENED, DRIFT_CHANGED)

# Thresholds. Deliberately loose in the reopening direction: one extra triage pass
# is cheap, a wrong "converged" claim is not.
DRIFT_OCCURRENCE_FACTOR = 2.0     # occurrences doubled -> strengthened
DRIFT_RESOLVED_FACTOR = 0.34      # occurrences fell to <= 1/3 -> resolved
DRIFT_SELF_CORRECT_DELTA = 0.30   # self-correction moved >= 30pp -> changed
DRIFT_ASSOCIATION_DELTA = 0.25    # failure lift / post-error rate moved >= 25pp
DRIFT_COVERAGE_FACTOR = 2.0       # episode or scenario coverage doubled
DRIFT_COVERAGE_MIN = 2            # ...and reached at least this many

# Back-compat aliases (v2 callers).
REOPEN_OCCURRENCE_FACTOR = DRIFT_OCCURRENCE_FACTOR
REOPEN_SELF_CORRECT_DELTA = DRIFT_SELF_CORRECT_DELTA


def _assoc(cand: Dict) -> Optional[float]:
    """Failure association, whichever form the miner produced.

    `post_error_rate` for reactive calls, `lift` for labelled query-phase signals.
    Both are "how strongly does this signal track failure", on a comparable scale.
    """
    for k in ("post_error_rate", "lift", "failure_association"):
        v = cand.get(k)
        if v is not None:
            return float(v)
    return None


def classify_drift(hist: Optional[Dict], cur: Dict) -> Tuple[str, str]:
    """Compare historical vs current-iteration evidence for one signal.

    Returns (drift class, human-readable reason). This is the heart of
    cross-iteration updating: reopening must NOT hinge on cumulative occurrence
    count alone, because a signal can change in ways that leave its count flat —
    broader scenario coverage, a shifted failure association, or a changed
    self-correction mechanism.

    Precedence is deliberate:
      1. RESOLVED     — it went away. Checked first: a gate that eliminates its
                        own signal is the SUCCESS case (§8.3), and must never be
                        reported as "strengthened" or re-triaged.
      2. CHANGED      — the mechanism moved (self-correction / association).
                        Beats `strengthened` because a mechanism shift invalidates
                        the prior policy choice even if support merely grew.
      3. STRENGTHENED — same mechanism, more/broader evidence.
      4. STABLE       — nothing material moved; do not re-analyze.
    """
    if hist is None:
        return DRIFT_NEW, "first observation"

    h_sup = int(hist.get("support", 0) or 0)
    c_sup = int(cur.get("support", 0) or 0)
    h_eps = int(hist.get("episodes", 0) or 0)
    c_eps = int(cur.get("episodes", 0) or 0)
    h_scn = len(hist.get("scenarios") or [])
    c_scn = len(cur.get("scenarios") or [])
    h_sc, c_sc = hist.get("self_correct_rate"), cur.get("self_correct_rate")
    h_as, c_as = hist.get("failure_association"), cur.get("failure_association")

    # ── 1. resolved ──────────────────────────────────────────────────────────
    if h_sup > 0 and c_sup == 0:
        return DRIFT_RESOLVED, f"support {h_sup} -> 0 (signal no longer occurs)"
    if h_sup >= 3 and c_sup <= h_sup * DRIFT_RESOLVED_FACTOR:
        return DRIFT_RESOLVED, (f"support {h_sup} -> {c_sup} "
                                f"(<= {DRIFT_RESOLVED_FACTOR:.0%} of prior)")
    if h_as is not None and c_as is not None and h_as - c_as >= DRIFT_ASSOCIATION_DELTA:
        return DRIFT_RESOLVED, (f"failure association {h_as:.2f} -> {c_as:.2f} "
                                f"(dropped >= {DRIFT_ASSOCIATION_DELTA})")

    # ── 2. changed (mechanism) ───────────────────────────────────────────────
    if h_sc is not None and c_sc is not None and abs(c_sc - h_sc) >= DRIFT_SELF_CORRECT_DELTA:
        return DRIFT_CHANGED, (f"self-correction {h_sc:.0%} -> {c_sc:.0%} "
                               f"(>= {DRIFT_SELF_CORRECT_DELTA:.0%} shift)")
    if h_as is not None and c_as is not None and c_as - h_as >= DRIFT_ASSOCIATION_DELTA:
        return DRIFT_CHANGED, (f"failure association {h_as:.2f} -> {c_as:.2f} "
                               f"(rose >= {DRIFT_ASSOCIATION_DELTA})")
    # A mechanism can also change by the control verdict flipping.
    if hist.get("control") and cur.get("control") and hist["control"] != cur["control"]:
        return DRIFT_CHANGED, f"control verdict {hist['control']} -> {cur['control']}"

    # ── 3. strengthened (coverage or volume) ─────────────────────────────────
    if c_eps >= max(DRIFT_COVERAGE_MIN, h_eps * DRIFT_COVERAGE_FACTOR) and c_eps > h_eps:
        return DRIFT_STRENGTHENED, f"episode coverage {h_eps} -> {c_eps}"
    if c_scn >= max(DRIFT_COVERAGE_MIN, h_scn * DRIFT_COVERAGE_FACTOR) and c_scn > h_scn:
        return DRIFT_STRENGTHENED, f"scenario coverage {h_scn} -> {c_scn}"
    if h_sup > 0 and c_sup >= h_sup * DRIFT_OCCURRENCE_FACTOR:
        return DRIFT_STRENGTHENED, (f"occurrences {h_sup} -> {c_sup} "
                                    f"(>= {DRIFT_OCCURRENCE_FACTOR}x)")

    # ── 4. stable ────────────────────────────────────────────────────────────
    return DRIFT_STABLE, (f"support {h_sup} -> {c_sup}, episodes {h_eps} -> {c_eps}"
                          + (f", self-correct {h_sc:.0%} -> {c_sc:.0%}"
                             if h_sc is not None and c_sc is not None else ""))


# ── Trigger specifications ───────────────────────────────────────────────────
# A learned gate's trigger is ultimately a CONDITION OVER SIGNALS AND CONTEXT, not
# a single error string:
#
#   memory_full AND proposed_action=core_memory_clear AND destructive_risk_high
#       -> suppress
#
# The ledger therefore keys evidence on a `trigger_spec`, not a bare signal. A
# single-signal spec is the degenerate one-term case, and its id is byte-identical
# to the old `signal_id(kind, signal)` form so existing records need no migration.
#
# Deliberately NOT arbitrary rule learning (§8.10): conjunctions of already-mined
# signals and context predicates only. E2 stays on single signals.

def make_trigger(kind: str, signal: str, terms=None) -> Dict:
    """Build a trigger spec.

    terms: optional extra conjuncts, each {"feature": str, "op": str, "value": Any}.
    Order-insensitive — the id sorts them — so two refinements that add the same
    conjuncts in different orders are one trigger, not two.
    """
    return {"kind": kind, "signal": signal, "terms": list(terms or [])}


def _term_str(t: Dict) -> str:
    return "%s%s%s" % (t.get("feature", "?"), t.get("op", "="), t.get("value", ""))


def trigger_id(trigger: Dict) -> str:
    """Stable identity for a trigger spec.

    A single-term trigger hashes to exactly `kind:signal`, so records written
    before trigger_spec existed keep resolving.
    """
    base = signal_id(trigger.get("kind", ""), trigger.get("signal", ""))
    terms = trigger.get("terms") or []
    if not terms:
        return base
    return base + " AND " + " AND ".join(sorted(_term_str(t) for t in terms))


def trigger_str(trigger: Dict) -> str:
    """Human-readable form, e.g. 'memory_full AND action=core_memory_clear'."""
    parts = [str(trigger.get("signal", "?"))]
    parts += [_term_str(t) for t in sorted(trigger.get("terms") or [],
                                           key=_term_str)]
    return " AND ".join(parts)


def is_refinement_of(child: Dict, parent: Dict) -> bool:
    """True if `child` is `parent` plus one or more extra conjuncts.

    A refinement is strictly narrower, so its measurements must NOT be credited to
    the parent: the parent's policy search failing says nothing about a subset.
    """
    if signal_id(child.get("kind", ""), child.get("signal", "")) != \
       signal_id(parent.get("kind", ""), parent.get("signal", "")):
        return False
    ct = {_term_str(t) for t in (child.get("terms") or [])}
    pt = {_term_str(t) for t in (parent.get("terms") or [])}
    return pt < ct


def signal_id(kind: str, signal: str) -> str:
    """Stable identity for a mined signal.

    `signal` is already normalized upstream for error contracts (numeric limits
    collapsed to N by `policy_tree._canon_error`), so two runs that differ only in
    a limit value map to one ledger entry. Whitespace and case are folded here so
    trivially different renderings do not fork the record.
    """
    norm = re.sub(r"\s+", " ", (signal or "").strip().lower())
    return f"{kind}:{norm}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EvidenceLedger:
    """Load/update/save the ledger. All mutations are explicit method calls."""

    def __init__(self, path: Optional[Path] = None, data: Optional[Dict] = None):
        self.path = Path(path) if path else None
        if data is not None:
            self.data = data
        elif self.path and self.path.exists():
            self.data = json.load(open(self.path))
            self._migrate()
        else:
            self.data = {"schema": SCHEMA_VERSION, "iterations": 0, "signals": {}}

    # ── persistence ──────────────────────────────────────────────────────────
    def _migrate(self) -> None:
        self.data.setdefault("schema", SCHEMA_VERSION)
        self.data.setdefault("iterations", 0)
        self.data.setdefault("signals", {})

    def save(self, path: Optional[Path] = None) -> Path:
        p = Path(path) if path else self.path
        if p is None:
            raise ValueError("no path to save to")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
        return p

    # ── queries ──────────────────────────────────────────────────────────────
    def get(self, kind: str, signal: str) -> Optional[Dict]:
        return self.data["signals"].get(signal_id(kind, signal))

    def __len__(self) -> int:
        return len(self.data["signals"])

    def by_status(self, status: str) -> List[Dict]:
        return [r for r in self.data["signals"].values() if r["status"] == status]

    # ── the core operation ───────────────────────────────────────────────────
    def observe(self, cand: Dict, iteration: Optional[int] = None,
                scenarios: Optional[List[str]] = None) -> Tuple[Dict, bool]:
        """Record one mined candidate. Returns (record, created).

        UPDATES an existing record rather than creating a duplicate — that is the
        whole point of the ledger. `occurrences` accumulates across iterations;
        `support_episodes` / `support_scenarios` take the max seen (a later
        iteration with the gate installed may legitimately see fewer).
        """
        kind = cand.get("kind", "unknown")
        sig = cand.get("signal", "")
        # A candidate may carry an explicit trigger_spec; otherwise it is the
        # single-signal degenerate case, whose id equals the historical signal_id.
        trig = cand.get("trigger_spec") or make_trigger(kind, sig)
        sid = trigger_id(trig)
        it = self.data["iterations"] if iteration is None else iteration
        rec = self.data["signals"].get(sid)
        created = rec is None

        if created:
            rec = {
                "id": sid, "kind": kind, "signal": sig,
                "first_seen_iteration": it, "last_seen_iteration": it,
                "occurrences": 0, "observations": 0,
                "support_episodes": 0, "support_scenarios": [],
                "causal_analyses": [], "self_correct_rate": None,
                "self_correct_n": 0,
                "verdict": None, "candidate_policy_set": [],
                "policies_tested": [], "status": STATUS_UNRESOLVED,
                "history": [],
                # historical = evidence as of the END of the previous observation.
                # current   = evidence from the freshly generated corpus.
                # Kept apart so a post-install iteration can ask "did this change?"
                # instead of only "have I seen this string before?".
                "historical": None,
                "current": None,
                "drift": DRIFT_NEW,
                "drift_reason": "first observation",
                # Action space under which this signal's policies were last
                # searched. Lets a later iteration tell "already failed under this
                # action set" from "retry, new actions exist".
                "action_space_searched": None,
                # Trigger specification. A single-signal trigger is the degenerate
                # one-term case; refinements add conjuncts and get their OWN record,
                # linked by refines/refined_by, because a narrower trigger's
                # measurements must never be credited to the broader one.
                "trigger_spec": None,
                "refines": None,
                "refined_by": [],
            }
            rec["trigger_spec"] = trig
            self.data["signals"][sid] = rec

        # Roll current -> historical BEFORE overwriting, so the comparison is
        # against the last actually-observed evidence, not the running totals.
        if rec.get("current") is not None:
            rec["historical"] = rec["current"]
        cur = {
            "iteration": it,
            "support": int(cand.get("support", 0) or 0),
            "episodes": int(cand.get("episodes", 0) or 0),
            "scenarios": sorted(set(scenarios or [])),
            "self_correct_rate": cand.get("self_correct_rate"),
            "self_correct_n": int(cand.get("self_correct_n", 0) or 0),
            "failure_association": _assoc(cand),
            "candidate_policy_set": list(cand.get("candidate_policy_set") or []),
            "control": cand.get("control"),
            "active_action": cand.get("active_action"),
        }
        rec["current"] = cur
        if not created:
            rec["drift"], rec["drift_reason"] = classify_drift(rec["historical"], cur)

        rec["last_seen_iteration"] = it
        rec["observations"] += 1
        rec["occurrences"] += int(cand.get("support", 0) or 0)
        eps = int(cand.get("episodes", 0) or 0)
        rec["support_episodes"] = max(rec["support_episodes"], eps)
        if scenarios:
            rec["support_scenarios"] = sorted(set(rec["support_scenarios"]) | set(scenarios))
        if cand.get("self_correct_rate") is not None:
            rec["self_correct_rate"] = cand["self_correct_rate"]
            rec["self_correct_n"] = int(cand.get("self_correct_n", 0) or 0)
        if cand.get("intervene"):
            rec["verdict"] = {
                "intervene": cand.get("intervene"),
                "control": cand.get("control"),
                "active_action": cand.get("active_action"),
                "confidence": cand.get("confidence"),
                "iteration": it,
            }
        if cand.get("candidate_policy_set"):
            rec["candidate_policy_set"] = list(cand["candidate_policy_set"])

        rec["history"].append({
            "iteration": it, "at": _now(),
            "support": int(cand.get("support", 0) or 0),
            "episodes": eps,
            "self_correct_rate": cand.get("self_correct_rate"),
            "candidate_policy_set": list(cand.get("candidate_policy_set") or []),
        })
        return rec, created

    def add_causal_analysis(self, kind: str, signal: str, name: str,
                            finding: str, iteration: Optional[int] = None) -> Dict:
        """Record an analysis already performed, so it is not redone."""
        rec = self.get(kind, signal)
        if rec is None:
            raise KeyError(signal_id(kind, signal))
        it = self.data["iterations"] if iteration is None else iteration
        existing = {a["name"] for a in rec["causal_analyses"]}
        if name not in existing:
            rec["causal_analyses"].append(
                {"name": name, "finding": finding, "iteration": it, "at": _now()})
        return rec

    def record_measurement(self, kind: str, signal: str, policy: str,
                           gain_pp: Optional[float], corpus: str,
                           confirmed: Optional[bool] = None,
                           iteration: Optional[int] = None) -> Dict:
        """Record that a policy was MEASURED, with its gain or regression.

        This is what stops the loop re-testing a policy it already paid for.
        """
        rec = self.get(kind, signal)
        if rec is None:
            raise KeyError(signal_id(kind, signal))
        it = self.data["iterations"] if iteration is None else iteration
        for m in rec["policies_tested"]:
            if m["policy"] == policy and m["corpus"] == corpus:
                m.update({"gain_pp": gain_pp, "confirmed": confirmed,
                          "iteration": it, "at": _now(),
                          "action_space": action_space_id()})
                return rec
        rec["policies_tested"].append(
            {"policy": policy, "gain_pp": gain_pp, "corpus": corpus,
             "confirmed": confirmed, "iteration": it, "at": _now(),
             "action_space": action_space_id()})
        return rec

    def set_status(self, kind: str, signal: str, status: str,
                   reason: str = "", iteration: Optional[int] = None) -> Dict:
        if status not in _STATUSES:
            raise ValueError(f"status must be one of {_STATUSES}, got {status!r}")
        rec = self.get(kind, signal)
        if rec is None:
            raise KeyError(signal_id(kind, signal))
        it = self.data["iterations"] if iteration is None else iteration
        rec["status"] = status
        rec["status_reason"] = reason
        rec["status_iteration"] = it
        return rec

    def defer(self, kind: str, signal: str, reason: str = "",
              iteration: Optional[int] = None,
              action_space: Optional[str] = None) -> Dict:
        """Mark a STRONG signal that no available policy could fix.

        The defer-and-continue rule (§8.9): a hard signal must not block the loop.
        This records the action space it was searched under, so reopening can be
        driven by the action space GROWING rather than only by evidence drift.

        Distinct from set_status(STATUS_REJECTED): `rejected_no_op` asserts the
        signal is not worth intervening on, which is a claim about the SIGNAL.
        This is a claim about our current ACTIONS.
        """
        rec = self.set_status(kind, signal, STATUS_DEFERRED, reason, iteration)
        rec["action_space_searched"] = action_space or action_space_id()
        return rec

    # ── Trigger refinement (§8.10) ───────────────────────────────────────────
    def get_trigger(self, trigger: Dict) -> Optional[Dict]:
        return self.data["signals"].get(trigger_id(trigger))

    def refine(self, parent: Dict, terms: List[Dict], reason: str = "",
               iteration: Optional[int] = None) -> Dict:
        """Create a REFINED trigger record: parent's signal plus extra conjuncts.

        Why a separate record rather than mutating the parent: the parent's policy
        search failed *over all cases under that signal*. A refinement covers a
        SUBSET, so the parent's negative result carries no information about it —
        crediting the parent's measurements to the child would silently pre-reject
        the refinement, and vice versa would overstate the parent.

        Distinguishes the two reasons a strong signal fails policy search:
          - action-space insufficiency: trigger precise, no action works -> defer
          - trigger UNDERSPECIFICATION: different cases under one signal need
            different policies -> refine, then search again
        """
        it = self.data["iterations"] if iteration is None else iteration
        child = make_trigger(parent.get("kind", ""), parent.get("signal", ""),
                             list(parent.get("terms") or []) + list(terms))
        cid = trigger_id(child)
        if not is_refinement_of(child, parent):
            raise ValueError("refinement must add at least one new conjunct")
        prec = self.data["signals"].get(trigger_id(parent))
        rec = self.data["signals"].get(cid)
        if rec is None:
            rec = {
                "id": cid, "kind": child["kind"], "signal": child["signal"],
                "trigger_spec": child, "trigger_str": trigger_str(child),
                "refines": trigger_id(parent), "refined_by": [],
                "first_seen_iteration": it, "last_seen_iteration": it,
                "occurrences": 0, "observations": 0,
                "support_episodes": 0, "support_scenarios": [],
                "causal_analyses": [], "self_correct_rate": None,
                "self_correct_n": 0,
                "verdict": None, "candidate_policy_set": [],
                # Deliberately EMPTY: the parent's measurements do not transfer.
                "policies_tested": [],
                "status": STATUS_UNRESOLVED,
                "action_space_searched": None,
                "history": [],
                "historical": None, "current": None,
                "drift": DRIFT_NEW, "drift_reason": "refined from parent trigger",
                "refine_reason": reason,
            }
            self.data["signals"][cid] = rec
        if prec is not None and cid not in (prec.get("refined_by") or []):
            prec.setdefault("refined_by", []).append(cid)
        return rec

    def record_measurement_for(self, trigger: Dict, policy: str,
                               gain_pp: Optional[float], corpus: str,
                               confirmed: Optional[bool] = None,
                               iteration: Optional[int] = None) -> Dict:
        """Record a measurement against an EXACT trigger spec.

        Measurements must be tied to the trigger they were taken under: the same
        policy can win under a refined trigger and lose under the broad one, which
        is the whole reason refinement exists.
        """
        rec = self.get_trigger(trigger)
        if rec is None:
            raise KeyError(trigger_id(trigger))
        it = self.data["iterations"] if iteration is None else iteration
        for m in rec["policies_tested"]:
            if m["policy"] == policy and m["corpus"] == corpus:
                m.update({"gain_pp": gain_pp, "confirmed": confirmed,
                          "iteration": it, "at": _now(),
                          "action_space": action_space_id()})
                return rec
        rec["policies_tested"].append(
            {"policy": policy, "gain_pp": gain_pp, "corpus": corpus,
             "confirmed": confirmed, "iteration": it, "at": _now(),
             "action_space": action_space_id(),
             "trigger": trigger_id(trigger)})
        return rec

    def defer_for(self, trigger: Dict, reason: str = "",
                  iteration: Optional[int] = None,
                  action_space: Optional[str] = None) -> Dict:
        """defer(), keyed on an exact trigger spec.

        Should only be reached AFTER refinement has been attempted and failed
        (§8.10): deferring a coarse trigger without trying to refine it conflates
        "no action works" with "this trigger lumps unlike cases together".
        """
        rec = self.get_trigger(trigger)
        if rec is None:
            raise KeyError(trigger_id(trigger))
        it = self.data["iterations"] if iteration is None else iteration
        rec["status"] = STATUS_DEFERRED
        rec["status_reason"] = reason
        rec["status_iteration"] = it
        rec["action_space_searched"] = action_space or action_space_id()
        return rec

    def refinements_of(self, trigger: Dict) -> List[Dict]:
        pid = trigger_id(trigger)
        return [r for r in self.data["signals"].values() if r.get("refines") == pid]

    def mark_underpowered(self, kind: str, signal: str, policy: str,
                          observed_gain_pp: float, required_pp: float,
                          n_exposed: int, reason: str = "",
                          iteration: Optional[int] = None) -> Dict:
        """Record a candidate that no available corpus can confirm.

        Stores the POWER GAP explicitly — observed effect, the threshold the
        confirmation corpus could actually resolve, and how many exposed cases it
        had — so a later iteration can ask "do we have enough exposed cases NOW?"
        rather than re-deriving it. That count is the reopen key: this status is
        not about the signal, the action, or the trigger, but about corpus size.
        """
        rec = self.get(kind, signal)
        if rec is None:
            raise KeyError(signal_id(kind, signal))
        it = self.data["iterations"] if iteration is None else iteration
        # Cases needed for the observed effect to clear the required threshold,
        # holding the effect size fixed and scaling the sqrt-driven component.
        needed = None
        if observed_gain_pp > 0 and n_exposed > 0:
            try:
                needed = int(math.ceil(n_exposed * (required_pp / observed_gain_pp) ** 2))
            except Exception:
                needed = None
        rec["status"] = STATUS_UNDERPOWERED
        rec["status_reason"] = reason
        rec["status_iteration"] = it
        rec["power"] = {
            "candidate_policy": policy,
            "observed_gain_pp": round(observed_gain_pp, 2),
            "required_pp": round(required_pp, 2),
            "n_exposed": n_exposed,
            "n_exposed_needed_estimate": needed,
            "action_space": action_space_id(),
            "iteration": it,
        }
        return rec

    def power_gap(self, kind: str, signal: str) -> Optional[Dict]:
        rec = self.get(kind, signal)
        return (rec or {}).get("power")

    def tested_policies(self, kind: str, signal: str) -> List[str]:
        rec = self.get(kind, signal)
        return [m["policy"] for m in rec["policies_tested"]] if rec else []

    # ── re-proposal gating ───────────────────────────────────────────────────
    def is_materially_new(self, cand: Dict) -> Tuple[bool, str]:
        """Does this observation justify reopening a settled signal?

        Delegates to the DRIFT CLASS rather than testing a raw occurrence count.
        A signal can change materially while its count stays flat — broader
        scenario coverage, a moved failure association, a changed self-correction
        mechanism — and each of those must be able to reopen it.
        """
        rec = self.get(cand.get("kind", ""), cand.get("signal", ""))
        if rec is None:
            return True, "not in ledger"
        if rec["status"] == STATUS_UNRESOLVED:
            return True, "still unresolved"

        # ── Reopen condition 0: MORE EXPOSED CASES are now available ─────────
        # An underpowered candidate is not blocked by the signal, the action, or the
        # trigger — only by corpus size. It reopens when the exposed-case count
        # reaches the estimate recorded at deferral. Checked first and independently
        # of evidence drift: the evidence can be identical and the candidate still
        # newly confirmable.
        if rec["status"] == STATUS_UNDERPOWERED:
            pw = rec.get("power") or {}
            need = pw.get("n_exposed_needed_estimate")
            have = int(cand.get("episodes", 0) or 0) or int(cand.get("n_exposed", 0) or 0)
            if need and have >= need:
                return True, (f"exposed cases {have} >= {need} needed to resolve "
                              f"{pw.get('observed_gain_pp')}pp — now confirmable")
            return False, (f"underpowered: {have} exposed cases, need ~{need} to "
                           f"resolve {pw.get('observed_gain_pp')}pp "
                           f"(required ±{pw.get('required_pp')}pp)")

        # ── Reopen condition 1: the ACTION SPACE grew ────────────────────────
        # A signal deferred as unresolved_no_policy failed under a SPECIFIC action
        # set. If new families exist now (transform, semantic-suppress, ...), that
        # earlier failure says nothing about the new actions, so reopen. Checked
        # FIRST and independently of evidence drift: the evidence may be perfectly
        # stable and the signal still worth retrying. Read via the module global,
        # not a captured value, so a bumped ACTION_SPACE_VERSION takes effect.
        if rec["status"] == STATUS_DEFERRED:
            searched = rec.get("action_space_searched")
            now_space = action_space_id()
            if searched and searched != now_space:
                prev_fams = set(searched.split(":", 1)[-1].split(","))
                new_fams = set(now_space.split(":", 1)[-1].split(",")) - prev_fams
                if new_fams:
                    return True, (f"action space grew: {searched} -> {now_space} "
                                  f"(new: {sorted(new_fams)})")
                return True, f"action space changed: {searched} -> {now_space}"

        # Classify the incoming evidence against what we last observed. Note this
        # does NOT mutate the record — should_propose() may be called before or
        # after observe().
        #
        # If the last observation was an ABSENCE (support 0, written by
        # mark_resolved_if_absent), compare against the evidence BEFORE that
        # instead. Otherwise a faint reappearance reads as 0 -> 2 "stable", losing
        # the fact that the signal was resolved from a much higher level — and a
        # gate's own eliminated signal would drift back into the proposal queue.
        hist = rec.get("current") or rec.get("historical")
        if hist is not None and int(hist.get("support", 0) or 0) == 0:
            hist = rec.get("historical") or hist
        cur = {
            "support": int(cand.get("support", 0) or 0),
            "episodes": int(cand.get("episodes", 0) or 0),
            "scenarios": sorted(set(cand.get("scenarios") or [])),
            "self_correct_rate": cand.get("self_correct_rate"),
            "failure_association": _assoc(cand),
            "control": cand.get("control"),
        }
        drift, why = classify_drift(hist, cur)

        if drift in (DRIFT_CHANGED, DRIFT_STRENGTHENED):
            return True, f"{drift}: {why}"
        if drift == DRIFT_RESOLVED:
            # Do NOT reopen. A learned gate that eliminated its own signal is the
            # success case; re-triaging it would rediscover a problem we fixed.
            return False, f"{drift}: {why}"

        # An untested policy in the proposed set is itself grounds to reopen a
        # DEFERRED signal: the deferral asserts "nothing we TRIED worked", not
        # "nothing we could try would work". Not applied to learned/rejected --
        # those are decisions about the signal, already made.
        untested = set(cand.get("candidate_policy_set") or []) - set(
            self.tested_policies(cand.get("kind", ""), cand.get("signal", "")))
        untested.discard("no-op")
        if rec["status"] == STATUS_DEFERRED and untested:
            return True, f"untested policies remain: {sorted(untested)}"

        return False, f"{drift}: {why}"

    def should_propose(self, cand: Dict) -> Tuple[bool, str]:
        """Should this candidate go forward to policy measurement this iteration?"""
        rec = self.get(cand.get("kind", ""), cand.get("signal", ""))
        if rec is None:
            return True, "new signal"
        if rec["status"] == STATUS_UNRESOLVED:
            return True, "unresolved"
        ok, why = self.is_materially_new(cand)
        return (True, f"reopened: {why}") if ok else (False, why)

    def drift_of(self, kind: str, signal: str) -> Tuple[str, str]:
        """Latest recorded drift class for a signal."""
        rec = self.get(kind, signal)
        if rec is None:
            return DRIFT_NEW, "not in ledger"
        return rec.get("drift", DRIFT_NEW), rec.get("drift_reason", "")

    def mark_resolved_if_absent(self, seen_ids: set,
                                iteration: Optional[int] = None) -> List[Dict]:
        """Classify known signals that did NOT appear in this iteration.

        A signal absent from a freshly generated corpus is the strongest form of
        `resolved` — and it can only be detected here, because mining never emits
        a candidate for something that no longer occurs. Without this, a gate that
        eliminates its own signal leaves a permanently `stable` record.
        """
        it = self.data["iterations"] if iteration is None else iteration
        out = []
        for sid, rec in self.data["signals"].items():
            if sid in seen_ids:
                continue
            if rec.get("last_seen_iteration") == it:
                continue
            prev = rec.get("current") or rec.get("historical")
            if not prev or int(prev.get("support", 0) or 0) == 0:
                continue
            rec["historical"] = prev
            rec["current"] = {"iteration": it, "support": 0, "episodes": 0,
                              "scenarios": [], "self_correct_rate": None,
                              "failure_association": None,
                              "candidate_policy_set": [], "control": None,
                              "active_action": None}
            rec["drift"] = DRIFT_RESOLVED
            rec["drift_reason"] = (f"absent from iteration {it} corpus "
                                   f"(was {prev.get('support')} occurrences)")
            out.append(rec)
        return out

    # ── iteration bookkeeping ────────────────────────────────────────────────
    def begin_iteration(self) -> int:
        self.data["iterations"] += 1
        return self.data["iterations"]

    def summary(self) -> Dict:
        c = {s: 0 for s in _STATUSES}
        for r in self.data["signals"].values():
            c[r["status"]] = c.get(r["status"], 0) + 1
        return {"iterations": self.data["iterations"],
                "signals": len(self.data["signals"]), "by_status": c}
