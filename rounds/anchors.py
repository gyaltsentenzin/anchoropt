"""The anchor set and the measured progression, as data.

One source of truth. The README delta table, the per-round READMEs, and the stopping-rule
regression tests all read from here rather than restating numbers, because a number retyped in
four places is a number that will disagree with itself.

THE ACCEPTED STACK IS A1-A5 + A7. Six anchors, ONE progression, one acceptance rule applied to
every one of them -- see ACCEPTANCE_RULE below and docs/ACCEPTANCE_RULE.md.

    Accept an anchor iff ALL FOUR hold:
      1. TRAIN        positive NET effect vs the current incumbent, on paired cases
      2. DEV          no AGGREGATE regression vs the same incumbent
      3. ATTRIBUTION  the gain is concentrated where the anchor can engage, and the mechanism is
                      CAUSALLY VALIDATED from the anchor's own decision-point telemetry
      4. SAFETY       no catastrophic or prohibited failure mode introduced

    It is NOT "gains > 0 and zero losses". Net > 0 on train, net >= 0 on dev, with mechanism
    validation, gives monotonic improvement AT THE AGGREGATE LEVEL -- not on every case. An anchor
    that fixes nine and breaks two is an improvement; a zero-loss rule rejects it.

A6 IS NOT IN THE STACK. It was admitted earlier under a train-only rule and fails this one on two
criteria independently: dev net -1.19 pp, and its own target backend is NEGATIVE (-2.50 pp on
vector). Its SIGNAL is real and unaddressed, so it is DEFERRED rather than refuted -- see DEFERRED
and rounds/A6_deferred/. Dropping it costs 1.65 pp train and gains 1.19 pp dev.

TRAIN / DEV, AND WHY "DEV" IS THE HONEST WORD. The second split is used to ACCEPT AND REJECT
anchors, which makes it a VALIDATION set, not a test set. The reported dev figure is therefore
partly selected-on and is NOT a clean generalization estimate. An untouched final number would
need a third reserved split, which does not exist. Stated here so it is not quietly forgotten.

TWO ACCEPTANCE CLASSES, because an efficiency anchor cannot satisfy criterion 1 BY CONSTRUCTION --
its whole point is that behaviour is unchanged, so applying the accuracy rule to it is a CATEGORY
ERROR. The six anchors above are ACCURACY anchors and form PROGRESSION. E1 is an accepted
EFFICIENCY anchor, judged on: no task loss + provably useless work removed + no new harm, where a
0 pp delta is a PASS. It is kept out of PROGRESSION on purpose -- a 0 pp row in an accuracy table
reads as a failed accuracy anchor. See ACCEPTANCE_CLASSES and docs/ACCEPTANCE_RULE.md.

ACCEPTED IS NOT THE SAME AS ACTIVE. E1 is accepted (8/8 criteria) and currently NOT ACTIVE: its
gate key is absent from the frozen policy, so it fires 0 times behind every accuracy number here --
none of them are affected either way. That looks like an assembly oversight rather than a decision,
and it is recorded as an open item rather than asserted. See E1_EVIDENCE.

Provenance: prompt-free N0 line, Granite-4.1-8B, temperature=0.001, balanced cell-level fold,
leakage 0. Train n=303 scored queries, dev n=84. Numbers are the frozen results of each round's
arm; see each round's acceptance document for the criteria it was judged against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from anchoropt.anchor import Action, Anchor, IncisionPoint

# --------------------------------------------------------------------------------------------
# The four accepted anchors
# --------------------------------------------------------------------------------------------

A1 = Anchor(
    name="A1",
    locus="capacity/container/no_remaining_capacity",
    attribution="a write was blocked by a full container",
    incision_point=IncisionPoint.POST_EXECUTION,
    action=Action.REROUTE,
    notes=(
        "Semantic miner rank 1: 81 of 215 residual failures linked. Reroutes the blocked write "
        "to archival memory. Ranked by linked downstream loss, not event volume."
    ),
)

A2 = Anchor(
    name="A2",
    locus="existence/identifier/not_found",
    attribution="stored-but-unreachable: the value was archived, the read never consulted archive",
    incision_point=IncisionPoint.POST_EXECUTION,
    action=Action.REPROMPT,
    notes=(
        "Attribution split ONE error string into two causes: 39 read-side (this anchor) and 5 "
        "write-side (A2.W, deferred below the support floor). All 39 are self-inflicted -- the "
        "value sits where A1 put it. Destination attested on RESOLVING, not on not-erroring: "
        "archival_memory_retrieve demands an exact key match and A1 sanitises keys, so the "
        "fuzzy key_search is the reachable destination."
    ),
)

A3 = Anchor(
    name="A3",
    locus="permission/identifier/duplicate",
    attribution="the model re-writes a value A1 already archived, colliding with A1's own key",
    incision_point=IncisionPoint.POST_GENERATION_PRE_EXEC,
    action=Action.SUPPRESS,
    params={"predicate": "same key AND normalised-identical value"},
    notes=(
        "All 59 collisions are against keys A1 itself injected; 54/59 carry a token-identical "
        "value. Predicate is deliberately exact, never semantic: it fires on 55/59 and declines "
        "4 (three paraphrases, one genuinely new fact) rather than risk suppressing real "
        "information for 7% more coverage."
    ),
)

A4 = Anchor(
    name="A4",
    locus="no_tool_call_at_all",
    attribution="read-side decision failure: the fact was retrievable and the model did not look",
    incision_point=IncisionPoint.POST_GENERATION_PRE_EXEC,
    action=Action.REPROMPT,
    notes=(
        "The first anchor from the EXPANDED vocabulary, after the error-keyed miner was "
        "exhausted. Precision 0.80 against passing queries; the alternatives were rejected on "
        "the same test (short_episode 0.57, answered_after_read 0.50, single_read_only 0.47). "
        "Deliberately non-directive -- 23 of the 48 exposed episodes have nothing in the store, "
        "where forcing a read invites fabrication. See the v1-vs-v2 case study: identical "
        "signal, action and text at a different incision point scored -4.95 pp."
    ),
)

A5 = Anchor(
    name="A5",
    locus="capacity/container/no_remaining_slots",
    attribution=(
        "the container A1 reroutes INTO has reached its slot cap, so A1's own repair now has "
        "nowhere to put the payload"
    ),
    incision_point=IncisionPoint.POST_EXECUTION,
    action=Action.REROUTE,
    params={
        "victim": "last-written copy of a value with >= 2 copies",
        "invariant": "a copy must remain, VERIFIED against live state after the removal",
        "retry": "the ORIGINAL call, verbatim",
        "max_per_episode": 3,
    },
    notes=(
        "The residual A1's success creates, one level deeper than A2 and A3: not a bad read or a "
        "duplicate write, but the relocation target itself saturating. Eviction is admitted ONLY "
        "where it is provably lossless -- the container holds the same value twice, so removing one "
        "copy destroys nothing. No utility model, no model call, no fuzzy comparator: a fuzzy "
        "victim rule would break the very invariant that justifies the anchor. "
        "MECHANISM CAUSALLY VALIDATED on dev's vector shard from A5's own decision-point telemetry: "
        "30 evictions dispatched, the blocked write LANDED 30/30, the lossless invariant "
        "(copies_remaining >= 1) held 30/30 with copies_before 2-3, 0 eviction errors, and 24 "
        "correct declines where no victim existed. That is criterion 3 met on evidence. "
        "PROCESS NOTE, kept because it is the more instructive half: A5 was originally installed on "
        "an EXPLICIT OVERRIDE of a frozen zero-loss clause (3 train losses, net +4). A reader "
        "weighting pre-registration absolutely was right to call that unaccepted AS A PROCESS "
        "CRITICISM. Under the current rule it is settled on evidence -- +1.32 pp train, +8.33 pp "
        "dev, attribution in the one backend it engages. The override was procedurally wrong and "
        "substantively right, and the dev result does not erase the process failure. "
        "This anchor was RENAMED from E1 in the working repo; enable_archival_evict_duplicate IS "
        "A5. A separate REJECTED candidate (low-similarity reprompt, fired 22/22 and converted "
        "2/22) merely shared the A5 name -- see the naming trap in DEFERRED."
    ),
)

A7 = Anchor(
    name="A7",
    locus="size/blob/append_would_exceed_cap",
    attribution=(
        "the store is a single string at its cap and there is no second container, so the refused "
        "append is simply lost unless the blob itself is made shorter"
    ),
    incision_point=IncisionPoint.POST_EXECUTION,
    action=Action.REROUTE,
    params={
        "acceptance_rule": "v3: the rewrite must FIT and be SHORTER than the original",
        "retry": "the ORIGINAL append, verbatim",
        "scope": "one decision per overflow STATE, not per call",
        "max_per_episode": 3,
        "uses_llm_at_inference": True,
    },
    notes=(
        "The only accepted anchor that calls a model at inference time -- every other one computes "
        "its repair. Accepted because the alternative is losing the write outright, and because the "
        "SIMPLEST rule won: v3 (fit + shorter) +7.34 pp, v5 (+ numeric rescue) +4.59, v6 (fit only) "
        "+3.67, v4 (+ ratio floor) -7.34. Every added guard scored below plain v3. "
        "The 'must be shorter' clause is not a length rule and that is the transferable finding: "
        "constrained relative to its input the model REPHRASES and keeps content, constrained only "
        "to fit it RESTRUCTURES, which costs content at any length. One case pins it -- v3's rewrite "
        "is 1298 chars and v6's is 3094, and v3 is the one that keeps the fact, because v6 spent its "
        "length on a markdown outline. Ratio measures how much TEXT survived, not how much "
        "INFORMATION did. Gated to one backend, so quote +2.64 pp corpus-wide beside the +7.34 pp. "
        "CONFIRMED END TO END across all three backends (ten concurrent shards, fleet wall-clock "
        "2652 s): on the 258 unexposed cases the arm is a BIT-EXACT no-op -- 0 disagreements and "
        "byte-identical execution logs, so it is not merely accuracy-neutral off-target but the "
        "same computation. The confirmation also corrected the spliced dev baseline: the true "
        "incumbent dev figure is 33.33%, not the 34.52% the splice used, which UNDERSTATED A7's "
        "gain (+5.95 pp, not +4.76 pp). A7 is the one stochastic channel in the stack -- 1 of 26 "
        "compactions decoded differently -- so it is a RARE decode channel, not a pervasive one."
    ),
)

A8 = Anchor(
    name="A8",
    locus="capacity/container/clear_proposed_at_capacity",
    attribution=(
        "the container is full and the model proposes a DESTRUCTIVE CLEAR to make room, while the "
        "container still holds an exact duplicate that could be removed instead"
    ),
    incision_point=IncisionPoint.POST_GENERATION_PRE_EXEC,
    action=Action.SUPPRESS,
    params={
        "predicate": "proposed *_memory_clear + container at capacity + a known blocked write + an "
                     "exact duplicate in LIVE state (no fuzzy match)",
        "then": "suppress the clear, evict ONE redundant copy, verify a copy remains, retry verbatim",
        "otherwise": "FALL THROUGH UNCHANGED -- declining is the correct branch",
        "max_per_episode": 8,
    },
    notes=(
        "The kv instantiation of the capacity principle, and the first anchor to intercept a "
        "destructive action rather than repair a failed one. A5 evicts AFTER a write is refused; A8 "
        "acts BEFORE a clear executes, which is the only point where the destruction is still "
        "preventable. "
        "WHY IT IS NOT A5 TUNING, which was checked rather than assumed: A5's trigger regex matches "
        "the VECTOR error string while every observed clear here is kv, and A5 fires post-exec on a "
        "blocked add whereas A8 fires pre-exec on a proposed clear. A8 reuses A5's validated ACTION "
        "with a genuinely new TRIGGER. "
        "MECHANISM CAUSALLY VALIDATED at 15/15 firings: copies_before = 2 and copies_remaining = 1 on "
        "EVERY firing, live 50 -> 49, retry landed 15/15, zero invariant violations, each surviving "
        "copy verified against LIVE state. Exposure is exactly the 3 diagnosed episodes and is "
        "byte-reproducible across two independent runs. "
        "The victims are self-documenting: every one is a `*_unique`-suffixed retry the model created "
        "to work around 'Key name must be unique', byte-identical to its unsuffixed partner. True "
        "redundant copies, not near-duplicates. "
        "READ A8_EVIDENCE BEFORE QUOTING IT -- the whole effect is ONE (backend, scenario) cell."
    ),
)

A9 = Anchor(
    name="A9",
    locus="retrieval/similarity/core_max_below_threshold",
    attribution=(
        "a core retrieve returns nothing well-matched, and the model concludes the store does not "
        "hold the answer -- while the answer sits in the ARCHIVAL container, which it never searched"
    ),
    incision_point=IncisionPoint.POST_EXECUTION,
    action=Action.REROUTE,
    params={
        "trigger": "core_memory_retrieve whose best similarity_score < 0.30",
        "then": "dispatch the SAME query to archival, rank the UNION globally, replace the weak core "
                "result AT ITS OWN INDEX with the merged top-k",
        "tie_break": "toward CORE -- an equal-scoring archival entry can never evict a core entry",
        "otherwise": "return None and fall through unchanged",
        "threshold": 0.30,
        "scope": "vector only -- kv and rec_sum expose no per-entry similarity scores",
    },
    notes=(
        "NOT the first read-side anchor -- A2 is, and the two are worth comparing because they share "
        "a diagnosis and differ on everything that matters. A2 fires when the read ERRORS "
        "(`not_found`) and REPROMPTS the model to go search archival. A9 fires when the read "
        "SUCCEEDS and merely returns nothing well-matched -- no error is raised at all -- and it "
        "searches archival ITSELF, handing back merged results with no instruction to the model. "
        "So: A2 asks the model to retry against a reachable destination; A9 supplies the evidence. "
        "The trigger is the deeper difference. A2 needs an error string; A9 needs a SCORE, which is "
        "the structural-signal tier the lexical vocabulary cannot reach -- a read that fails "
        "silently looks like a successful read. "
        "Diagnosed from 13 of 13 verified cases where the needed information was present in archival "
        "and the model issued no archival read at all. The containers are lopsided -- vector-finance "
        "holds 5 core entries against 30 archival -- so a merged top-5 ranks over 35 candidates. "
        "ADDITIVE BY CONSTRUCTION, which is the safety argument: the union is ranked, so a core entry "
        "that still earns a top-k place keeps it. That is why a 66%-precision trigger is tolerable -- "
        "a false firing costs one extra read, never a displaced correct answer. "
        "The rejected alternative (winner-takes-container) was dropped on measurement, not taste: "
        "firing cases that already PASS have core_max median 0.159 vs 0.172 for failures, the same "
        "band, so a container-level switch cannot discriminate. "
        "MECHANISM CAUSALLY VALIDATED 24/24 at case level -- fired AND attested delivered AND archival "
        "promoted into the top-k AND the paired control fails. Best-similarity jumps like "
        "43-healthcare-13 0.034 -> 0.691 and 108-student-28 0.140 -> 0.617. "
        "READ A9_EVIDENCE BEFORE QUOTING IT: the deltas are VECTOR-SHARD figures, the whole-corpus "
        "figures are VECTOR-SHARD (the corpus deltas are +4.95 / +8.33 pp), its 'additive by "
        "construction' safety claim is RETRACTED, and this anchor's first two versions "
        "delivered nothing at all while reporting success."
    ),
)


# The accepted stack, in the order the anchors were learned. A6 is deliberately absent -- see
# DEFERRED. There is ONE progression and one acceptance rule; anchors are not grouped into an
# earlier pre-registered tier and a later qualified tier.
ANCHORS = (A1, A2, A3, A4, A5, A7, A8, A9)


# --------------------------------------------------------------------------------------------
# The governing acceptance rule -- ONE rule, applied to every anchor
# --------------------------------------------------------------------------------------------
#
# There is no earlier "pre-registered" tier and later "qualified" tier. Every anchor in ANCHORS was
# audited against the four criteria below, and A6 was dropped because it fails two of them. Keeping
# a single rule is what makes the progression a progression rather than a list of separately
# defended results.
# Two classes, different criteria. Applying the accuracy rule to an efficiency anchor is a category
# error: `Delta_train > 0` is unsatisfiable for an anchor whose defining property is that behaviour
# does not change. Omitting this distinction had a concrete cost -- it is how the efficiency anchor
# came to be dropped from the frozen policy without anyone deciding to drop it.
ACCEPTANCE_CLASSES = {
    "accuracy": (
        "the four criteria in ACCEPTANCE_RULE: net train > 0, net dev >= 0, attribution with a "
        "causally-validated mechanism, and safety"
    ),
    "efficiency": (
        "no task loss + provably useless work removed + no new harm. A 0 pp delta is a PASS, not a "
        "null result -- and a small POSITIVE delta is a warning sign, because it means the "
        "intervention leaked into behaviour."
    ),
}

# The three requirements that belong to the efficiency class alone, each from a measured failure.
EFFICIENCY_REQUIREMENTS = (
    ("saving is in EXECUTED calls, not proposals",
     ("the first implementation memoized AFTER _execute and saved nothing -- a behaviour change "
      "dressed as an efficiency one. Verify with a counting stub that _execute invocations for the "
      "targeted calls are 0.")),
    ("behaviour preserved BY CONSTRUCTION, not by hope",
     ("return the tool's VERBATIM result so the model's view is identical. One rejected predecessor "
      "injected a user message (-1.65 pp); another returned a NEW tool string (+0.33 pp but "
      "destructive calls 84 -> 90). Both said something new. The evidence is 0 pp with ZERO flips.")),
    ("the redundancy metric must count what the gate CHANGES",
     ("counting redundant calls in the decoded proposals reads 16 -> 16, because withheld calls "
      "still appear there by design. Counting EXECUTED redundant calls reads 16 -> 0 on the same "
      "data.")),
)

# A behaviour-preserving anchor does not trigger a re-mine: re-mining looks for loci whose
# trajectories shifted, and a 0 pp / zero-flip arm has none. Its locus leaves the EFFICIENCY
# residual, tracked separately from the accuracy residual.
EFFICIENCY_RESIDUAL_IS_SEPARATE = True

ACCEPTANCE_RULE = {
    "version": 2,
    "class": "accuracy",
    "criteria": (
        ("train", "positive NET effect vs the current incumbent, on paired cases"),
        ("dev", "no AGGREGATE regression vs the same incumbent"),
        ("attribution",
         ("the gain is concentrated where the anchor can engage, and the mechanism is CAUSALLY "
          "VALIDATED from the anchor's own decision-point telemetry")),
        ("safety", "no catastrophic or prohibited failure mode introduced"),
    ),
    "then": "freeze, re-evaluate, re-mine residuals from scratch",
    "is_not": (
        "gains > 0 AND zero losses. It is net > 0 on train and net >= 0 on dev with mechanism "
        "validation, which gives monotonic improvement AT THE AGGREGATE LEVEL, not on every "
        "individual case. An anchor that fixes nine and breaks two is an improvement; a zero-loss "
        "rule rejects it -- and would have rejected A5, now measured at +8.33 pp on dev with a "
        "30/30 causally-validated mechanism."
    ),
    "reporting": (
        "Report the paired (gains, losses) beside every net delta -- not as a gate, but because the "
        "SHAPE is diagnostic. 8 gains / 9 losses for net -1 is a different object from 0 gains / "
        "1 loss, and only the counts reveal it. This is also how 'one stochastic flip decides an "
        "anchor' is answered: by making the shape visible, not by absorbing it into a tolerance."
    ),
    # Why a threshold was NOT loosened to save A6, recorded because the reasoning is auditable.
    "epsilon_note": (
        "A tolerance of one case (1.19 pp) was considered and rejected on process grounds: 1.19 pp "
        "is EXACTLY A6's dev regression, so adopting it after seeing the table would be choosing "
        "the threshold that admits the one anchor it decides."
    ),
    "p_values": (
        "Diagnostic, never prescriptive, and especially weak here because the benchmark is "
        "DETERMINISTIC -- byte-identical execution logs across independent sharded runs spanning "
        "days, jobs and hosts. There is no sampling process over which a null distribution is "
        "defined, so a sign test on 84 fixed cases asks how surprising a split would be if the "
        "flips were coin tosses, and they are not. Report the paired counts and the mechanism."
    ),
}

# The audit that produced the current stack. Deltas are vs the incumbent at each point.
# `(train_pp, train_gl, dev_pp, dev_gl, attribution, verdict)`.
AUDIT = {
    "A1": (None, None, None, None, "stack floor, no isolated delta", "IN STACK"),
    "A2": (0.99, (5, 2), 0.00, (1, 1), "", "ACCEPT"),
    "A3": (3.96, (18, 6), 3.57, (3, 0), "", "ACCEPT"),
    "A4": (3.63, (16, 5), 3.57, (7, 4), "on-target +9.17, off-target 0.00", "ACCEPT"),
    "A5": (1.32, (7, 3), 8.33, (9, 2), "vector +17.50; kv 0.00, rec_sum 0.00", "ACCEPT"),
    "A6": (1.65, (14, 9), -1.19, (8, 9), "vector -2.50 -- ON-TARGET AND NEGATIVE",
           "REJECT (criteria 2 and 3)"),
    "A7": (2.64, (21, 13), 5.95, (5, 0), "rec_sum +25.00; kv 0.00, vector 0.00", "ACCEPT"),
    "A8": (1.65, (9, 4), 0.00, (0, 0), "kv +4.76; vector 0.00, rec_sum 0.00 (byte-identical)",
           "ACCEPT (criterion 2 met VACUOUSLY -- dev is a no-op)"),
}

# Safety, criterion 4, across the whole accepted stack.
SAFETY = {"clears_added": 0, "removes_added": 0}


# --------------------------------------------------------------------------------------------
# DEFERRED: A6's signal is real and unaddressed; the remedy was mis-specified
# --------------------------------------------------------------------------------------------
#
# A6 is the most instructive result in the project and it is NOT in the stack. It was admitted under
# an earlier train-only rule, with the dev regression reclassified as a "generalization diagnostic"
# -- which is exactly the move the current rule forbids. Under the current rule there was nothing to
# override: A6 should never have been accepted.
#
# It fails criterion 3 INDEPENDENTLY of criterion 2, and that is the cleaner statement: the anchor
# engages exactly where intended and makes things WORSE there (vector, its only target backend,
# -2.50 pp on dev). An anchor whose on-target effect is negative is not a tolerance question.
DEFERRED = {
    "A6": {
        "locus": "size/item/exceeds_per_item_limit",
        "status": "DEFERRED -- the signal is real and open; the remedy is not ready. Not refuted",
        "train_pp": 1.65,
        "dev_pp": -1.19,
        "dev_gl": (8, 9),
        "dev_discordant": 17,
        "on_target_dev_pp": -2.50,
        "cost_of_dropping": "1.65 pp train",
        "gain_of_dropping": "1.19 pp dev",
        "fails_criteria": (2, 3),
        # The mechanism, which is why this is a principled drop and not a threshold choice.
        "mechanism": (
            "On a saturated store A6 cannot ADD information -- it only reorders which phrasings "
            "occupy the fixed archival slots. Dev vector dispatch sites: A6's own site 0 -> 44 "
            "while a1_reroute COLLAPSES 22 -> 6, so A6 pre-empts A1 on the same payloads one error "
            "earlier. Final stores hold 57 vs 56 distinct facts with 54 destroyed and 53 added: a "
            "near-total content swap at constant capacity, same TOPICS in different PHRASINGS. "
            "Whether that helps is a coin flip on query-phrasing alignment, hence +1.98 train / "
            "-8.33 dev for the bare variant. A SELECTION intervention in a capacity costume."
        ),
        # The signal did not go away when the anchor did.
        "signal_still_present": {
            "entry_too_long_events": {"train": 55, "dev": 102},
            "vector_saturation": "62-85% at capacity",
            "byte_identical_duplicate_writes": 103,
            "paraphrase_tier": "15.8% of allowed writes are >= 0.90-similar; 73.7% < 0.50 distinct",
            "consequence": (
                "with A6 removed, over-long vector entries are rejected and lost again -- that is "
                "the 1.65 pp train cost, priced in knowingly"
            ),
        },
        # Worth keeping for whoever reopens this.
        "keep_for_reopening": (
            "RAW BYTE-EXACT dedup beat normalized dedup on every axis: dev -8.33 -> -1.19 pp, facts "
            "retained 56 vs 17 (net +39), 103 suppressions vs the normalized guard's 0. If "
            "duplicate suppression is revisited, start from raw byte equality. Suppression is NOT "
            "monotonically good -- informative variants matter."
        ),
        "reopen_conditions": (
            "a leave-one-chain-out protocol",
            "a selection-aware remedy that decides which fact to KEEP on explicit grounds",
            "a corpus where vector stores are not saturated for most of the trajectory",
            "a prose-capable fact detector (the lexical proxy is disqualified for such claims)",
        ),
        "standing_instruction": "do NOT simply re-enable the flags",
        # A7 is unaffected, which had to be checked rather than assumed.
        "a7_independence": (
            "A6 never fires on rec_sum (0 firings; the rec_sum shard is BIT-IDENTICAL with and "
            "without A6, 0/109 train and 0/20 dev). Structurally necessary: A6 targets the archival "
            "container and vector entries, rec_sum is a single string with neither. So A7 would "
            "have been mined identically had A6 never existed -- it was not a stepping stone."
        ),
    },
}


# --------------------------------------------------------------------------------------------
# The naming traps, recorded because three of them landed in one session
# --------------------------------------------------------------------------------------------
#
# Every one came from inferring a stack's contents from PROSE LABELS instead of from the policy
# file's flags plus firing telemetry. A label is a hypothesis; the flag list plus the firing counts
# are the evidence.
NAMING_TRAPS = (
    ("'E1' names TWO different anchors at two different times",
     ("The ACCURACY anchor now called A5 (archival-full duplicate eviction, "
      "`enable_archival_evict_duplicate`) was originally called E1 and RENAMED -- the working repo's "
      "log says 'A5 INSTALLED (formerly E1)'. The EFFICIENCY anchor called E1 in this repo is a "
      "different mechanism (deterministic-outcome memoization, gate key "
      "`on_memoized_deterministic_call`, its own standalone policy). Both are accepted; they are not "
      "the same anchor, and 'E1' alone is ambiguous across the working repo's history.")),
    ("a REJECTED candidate also shared the A5 name",
     ("`enable_low_similarity_reprompt` is the rejected A5-lowsim candidate -- it fired 22/22 and "
      "converted 2/22, about 5% of its predicted ceiling. It is not the accepted A5.")),
    ("A6's status flipped twice",
     ("frozen under a train-only rule, then dropped under the current one. The later record wins, "
      "and the policy artifact plus the next anchor's stated baseline are the tiebreakers.")),
    ("'A8' ALSO names two different anchors -- check before quoting either",
     ("In THIS repo A8 is dedup-first clear recovery (gate key `on_dedup_clear_recovery`, policy "
      "sha256 26792094baec5d50...). In the working repo, 'N0-A8' is a LATER and DIFFERENT candidate "
      "-- cross-container retrieval merge, gate key `on_low_similarity_cross_container` -- whose "
      "CONTROL is this repo's full accepted stack. Same numeral, different anchors, and the working "
      "repo's A8 is under acceptance rather than accepted. Read the gate key, not the numeral.")),
    ("the general lesson",
     ("every one of these came from inferring a stack's contents from PROSE LABELS instead of from "
      "the policy file's flags plus firing telemetry. A label is a hypothesis; the flag list plus "
      "the firing counts are the evidence.")),
)


# --------------------------------------------------------------------------------------------
# A6's measurements, kept in full because a REJECTED anchor's evidence is still evidence
# --------------------------------------------------------------------------------------------
#
# `displacement_check` in anchoropt/learning/exposure.py is the general diagnostic these
# observations motivated. Kept as data so the figures cannot drift from the claim they support.
#
# Note what the variant table shows: even the BEST variant is negative on dev. The conservative
# raw-exact variant was the one carried into the (now reversed) train-rule acceptance, and choosing
# on the train delta alone would have taken bare A6, which is 7x worse on dev. Under the current
# rule none of the three qualifies.
A6_EVIDENCE = {
    "in_stack": False,
    "train_delta_pp": 1.65,
    "dev_delta_pp": -1.19,
    "train_flips": "14 gains / 9 losses",
    "dev_flips": "8 gains / 9 losses",
    "dev_discordant": 17,
    "exposed_backend": "vector",
    "exposed_train_delta_pp": 5.62,
    "exposed_dev_delta_pp": -2.50,     # ON-TARGET AND NEGATIVE -- criterion 3 failure
    # The displacement that explains the sign flip. Dev, vector dispatch sites.
    "dev_sites": {"a6_reroute": (0, 44), "a1_reroute": (22, 6)},
    "final_store_distinct": (57, 56),
    "facts_swapped": (54, 53),
    "verdict": "SELECTION intervention at constant capacity, not a capacity intervention",
    "variants": {
        "bare_a6": {"train_pp": 1.98, "dev_pp": -8.33, "best_of_three": False},
        "normalized_idempotence": {"train_pp": 1.98, "dev_pp": -4.76, "best_of_three": False,
                                   "note": "fired 0x on train, so its train figure is bare A6's; "
                                           "all 34 dev firings came from a single episode"},
        "raw_exact_dedup": {"train_pp": 1.65, "dev_pp": -1.19, "best_of_three": True,
                            "note": "the conservative variant -- least harmful on dev and the one "
                                    "carried into the earlier train-rule acceptance, but still "
                                    "negative on dev, so it fails the current rule too"},
    },
    "best_variant": "raw_exact_dedup",
    "best_variant_is_train_maximum": False,
    "variants_tried": 3,
    "all_variants_negative_on_dev": True,
    "caveats": (
        ("Every variant tried is negative on dev. This is not a threshold-tuning problem, and A6 "
         "tuning is CLOSED -- the standing instruction is not to re-enable the flags."),
        ("A6 fails criterion 3 INDEPENDENTLY of criterion 2: its own target backend is negative "
         "(-2.50 pp on vector). An anchor that engages exactly where intended and makes things "
         "worse there is not a tolerance question."),
        ("The dev fold is ONE domain chain per backend, so it measures domain transfer, not "
         "within-distribution generalization. Any anchor whose effect depends on chain "
         "multiplicity or saturation depth will move on it -- which is measured here, not assumed."),
        ("Dropping A6 gains 1.19 pp on dev, which at n=84 is ~1 case. The DIRECTION is what "
         "carries, supported by the measured chain-multiplicity dependence and the mechanism."),
        ("The remaining pressure is the PARAPHRASE tier (15.8% of allowed writes are >= 0.90 "
         "similar). Exact matching cannot reach it, and merging paraphrases would change which "
         "phrasing survives -- the same coin flip. Do not loosen the predicate."),
    ),
}


# --------------------------------------------------------------------------------------------
# A7: exposure, because the within-backend figure is 2.8x the corpus figure
# --------------------------------------------------------------------------------------------
A7_EVIDENCE = {
    "exposed_backend": "rec_sum",
    "exposed_train_n": 109,
    "exposed_train_delta_pp": 7.34,
    "exposed_dev_n": 20,
    "exposed_dev_delta_pp": 20.00,
    "corpus_train_delta_pp": 2.64,
    "corpus_dev_delta_pp": 5.95,
    # MEASURED END TO END across all three backends -- ten concurrent shards, all rc=0, fleet
    # wall-clock 2652 s (the slowest single shard). The earlier spliced figures are superseded.
    "measured_end_to_end": True,
    "corpus_figures_are_spliced": False,
    # The off-target result is stronger than the criterion asked for: not merely accuracy-neutral
    # where A7 cannot act, but the SAME COMPUTATION.
    "off_target_cases": 258,
    "off_target_disagreements": 0,
    "off_target_exec_logs": "byte-identical",
    "train_flips": "21 gains / 13 losses",
    "dev_flips": "5 gains / 0 losses",
    "compactions_train": 21,
    "compactions_landed_train": 21,
    "uses_llm_at_inference": True,
    # A7 is the ONE stochastic channel in the stack, and it is rare rather than pervasive.
    "decode_channel": "1 of 26 compactions decoded differently (~4%)",
    # Every added guard scored BELOW the plain mechanical rule. The ordering IS the finding.
    "variants": {
        "v3_fit_and_shorter": 7.34,
        "v5_numeric_rescue": 4.59,
        "v6_fit_only": 3.67,
        "v4_ratio_floor": -7.34,
    },
    "caveats": (
        ("Gated to rec_sum at dispatch, which is 109/303 train and 20/84 dev. The other two "
         "backends are unchanged by construction, so +7.34 pp on rec_sum is +2.64 pp corpus-wide. "
         "Never quote the backend figure without the exposure."),
        ("19 of the 21 train gains are UNEXPLAINED by any measurement performed -- not explained "
         "by chance, unexplained. The available fact detector is capitalisation-based and cannot "
         "read lowercase prose answers."),
        ("2 of its 13 train losses are destructive rewrites. Three separate attempts to prevent "
         "them (v4, v5, v6) each cost more than the flaw does."),
        ("Dev never exercised the destructive mode: all 5 dev compactions retained 0.888-1.000, "
         "and at n=20 one case is 5 pp."),
        ("This anchor adds an LLM call, which reorders the trajectory -- coverage, state keys and "
         "downstream questions all move, so per-decision reasoning cannot bound run-level "
         "behaviour. It is also the only non-determinism in the stack."),
        ("The confirmation run CORRECTED an earlier spliced baseline: the incumbent's true dev "
         "figure is 33.33%, not the 34.52% the splice used, because A6 is harmful on dev vector. "
         "The splice had overstated the stack being built on, which UNDERSTATED A7's gain."),
    ),
}


# --------------------------------------------------------------------------------------------
# A8: the narrowest support in the stack, and the strongest mechanism validation
# --------------------------------------------------------------------------------------------
#
# A8 is the anchor to read if you want to see this project's evidence standards applied to a result
# that is genuinely thin. The MECHANISM is the best-validated of any anchor here -- 15/15 firings with
# the lossless invariant checked against live state on every one, byte-reproducible across two
# independent runs. The SUPPORT is the narrowest -- one (backend, scenario) cell.
#
# Both are true at once, and the acceptance rests on the first while the second bounds what may be
# claimed. Quoting +1.65 pp without "one cell" overstates it.
A8_EVIDENCE = {
    "train_delta_pp": 1.65,
    "train_flips": "9 gains / 4 losses",       # net +5, and the losses are real
    "dev_delta_pp": 0.00,
    "dev_flips": "0 gains / 0 losses",
    "exposed_backend": "kv",
    "exposed_train_delta_pp": 4.76,            # 35 vs 30 on n=105
    "unexposed_delta_pp": 0.00,                # vector AND rec_sum exactly 0.00
    "rec_sum_exec_logs": "byte-identical, so the 0.00 there is a CONSTRUCTION FACT",
    # Criterion 3, and this is the strongest mechanism record in the stack.
    "firings": 15,
    "copies_before": "2 on every firing",
    "copies_remaining": "1 on every firing -- LOSSLESS",
    "container_size": "50 -> 49 on every firing",
    "retry_landed": "15/15",
    "invariant_violations": 0,
    "byte_reproducible_across_runs": True,
    "firing_episodes": ("prereq_24-student-2 (4)", "prereq_25-student-3 (5)",
                        "prereq_29-student-7 (6)"),
    "exposure_matched_prediction": True,       # 3 predicted episodes, 4/5/6 removable copies
    # Criterion 4.
    "clears_executed": (5, 4),                 # control -> arm
    "clears_suppressed": 15,
    "force_quit": (0, 0),
    "median_prereq_steps": (19, 19),           # unchanged, so no runaway-loop pattern
    # Setup fidelity -- what makes the delta attributable at all.
    "control_reproduces_incumbent": "139/303 with 0 disagreements across all 303 cases",
    "cap_ever_bound": False,                   # max 6 per episode against a budget of 8
    "caveats": (
        ("THE ENTIRE EFFECT IS ONE CELL. All 3 exposed episodes and all 13 flips are in kv/student. "
         "The (backend, scenario) cell is the indivisible unit here, so this is n=1 CELL of support -- "
         "narrower than A5's single-backend exposure, which was already narrow."),
        ("DEV CANNOT CONFIRM IT. There is no kv/student cell in dev and the arm is a measured no-op "
         "there, with byte-identical exec logs. Criterion 2 is satisfied VACUOUSLY: it establishes "
         "'no harm' and supplies zero independent evidence for the gain. Pre-registered as a "
         "limitation before any number existed."),
        ("9 gains / 4 losses, not 9/0. The losses are real and in the same cell, so the arm REORDERS "
         "outcomes within kv/student rather than only adding to them."),
        ("DEDUP IS EXHAUSTIBLE. 4 archival clears still executed after the duplicates ran out, and "
         "clear PROPOSALS went 5 -> 19: suppressing a clear does not end the pressure -- the model "
         "re-proposes and each cycle spends one duplicate. This bounds what this layer can ever "
         "achieve and is the direct motivation for the next one."),
        ("One observability defect is still open: the gate's decline reason is never recorded, so the "
         "no-duplicate share cannot be read from the gate and must come from the destructive-call "
         "audit instead. No effect on any criterion."),
    ),
}

# The CAPACITY LADDER -- the ordering A8 confirmed by measurement, and where the next anchor goes.
#
# A9's record, and the caveats are the point. Two of them bound the claim and one is a warning about
# how the anchor's own telemetry lied for four experiments running.
#
# THE UNITS TRAP FIRST. All four accuracy figures below are VECTOR-SHARD, not whole-corpus. The
# harness prints "Overall accuracy" per shard and this shard is vector-only, so 60.7% read as a corpus
# number overstates the anchor by roughly 3x. The vector shard is 29.4% of train and 47.6% of dev.
A9_EVIDENCE = {
    "status": "ACCEPTED 2026-08-26 -- three-shard confirmed, whole corpus MEASURED",

    # AN OPEN CLASSIFICATION QUESTION, recorded rather than smoothed over. A9 is filed as REROUTE
    # because that is the family the working repo used and because the closed action set has no better
    # cell -- but the fit is poor, and pretending otherwise would let a taxonomy quietly stop
    # describing the code.
    #
    #   Action.REROUTE is defined as SUBSTITUTE the call: a different function, or the same function
    #   with different arguments. A1, A5 and A7 all match that literally -- each repairs state and
    #   then RETRIES THE ORIGINAL CALL, and the retried call is what executes.
    #
    #   A9 substitutes nothing. The core retrieve already RAN and SUCCEEDED. A9 issues an ADDITIONAL
    #   call and rewrites the RESULT of the first one before the model reads it. Nothing is retried and
    #   nothing is cancelled.
    #
    # So the operation is really AUGMENT-THE-OBSERVATION: leave the call and the store alone, enrich
    # what the decision-maker sees. That is also why A9 is NOT a registered `constraint_repair`
    # strategy the way A8 is -- the Repair enum (RELOCATE / REDUCE / EVICT) exists to admit a BLOCKED
    # WRITE, and A9 has no blocked write and no constraint violation to repair.
    #
    # Deliberately NOT fixed by adding a fifth action family now. The action set is closed and small on
    # purpose, an earlier `transform` family was merged away for failing to survive contact with the
    # code, and one anchor is a weak basis for a taxonomy change. The right trigger is a SECOND
    # observation-augmenting anchor, or the TauBench integration -- whichever comes first.
    "action_family_caveat": (
        "filed as REROUTE, but it substitutes no call: the retrieve already succeeded and A9 rewrites "
        "its RESULT after adding one read. The honest description is AUGMENT-THE-OBSERVATION. Not "
        "resolved by adding a family on the strength of a single anchor -- revisit on the second such "
        "anchor or at the TauBench integration. See docs/OBSERVABILITY_VS_DECISION_CONTEXT.md, where "
        "the same anchor breaks the reprompt-vs-execution split for the same underlying reason."
    ),

    "gate_key": "on_low_similarity_cross_container",
    "env_gate": "ANCHOROPT_XCM",
    "jobs": (1146571, 1146572, 1146573, 1146574),

    # MEASURED. Paired against the frozen incumbent control, same store, same cache.
    "vector_shard": {
        "train": {"arm": (54, 89), "control": (39, 89), "delta_pp": 16.9,
                  "gained": 15, "lost": 0, "mechanism": (15, 15)},
        "dev": {"arm": (21, 40), "control": (14, 40), "delta_pp": 17.5,
                "gained": 9, "lost": 2, "mechanism": (9, 9)},
    },

    # PROJECTED, and flagged as such everywhere it appears. Arithmetic on top of the measured vector
    # net (+15 train, +7 dev), assuming kv and rec_sum are unchanged. That assumption holds BY
    # CONSTRUCTION -- the mechanism returns None for non-vector backends -- but it has NOT been
    # confirmed by a three-shard run, and it should be before this figure goes in a paper.
    # MEASURED. Was a projection; the three-shard run confirmed it EXACTLY, so the figures did not
    # change -- only their standing did. Kept under the old key name so nothing silently re-reads a
    # projection, with the confirmation recorded alongside.
    "whole_corpus": {
        "train": {"from": (144, 303), "to": (159, 303), "delta_pp": 4.95, "gained": 15, "lost": 0},
        "dev": {"from": (34, 84), "to": (41, 84), "delta_pp": 8.33, "gained": 9, "lost": 2},
        "status": "MEASURED -- three-shard paired run, case by case",
        "vector_share": {"train": 0.294, "dev": 0.476},
        "was_projected": (
            "these were arithmetic on the vector shard until the three-shard run; it reproduced them "
            "exactly, which is a confirmation of the projection method rather than a correction to it"
        ),
    },
    # NON-INTERFERENCE IS NOW MEASURED, not inferred from the `!= vector` guard in code. That
    # distinction is the whole reason the projection needed confirming: "the code cannot touch those
    # backends" is an argument, and this line of work has been wrong before about arguments of exactly
    # that shape (see delivery_defect).
    "non_interference": {
        "kv": {"train": "35/35 byte-identical", "dev": "6/6 byte-identical"},
        "rec_sum": {"train": "70/70 byte-identical", "dev": "14/14 byte-identical"},
        "gate_firings_off_target": 0,
        "gate_declines_off_target": 0,
        "control_reproduces_incumbent": "exactly",
    },

    "acceptance": {
        1: "net train > 0 -- +15 cases",
        2: "net dev >= 0 -- +7 cases (+9 / -2)",
        3: "mechanism 24/24 validated at case level",
        4: (
            "safety -- 0 delivery violations, 0 append fallbacks, trigger unchanged. NET-POSITIVE, "
            "NOT non-destructive: see additivity_retracted"
        ),
    },

    # A RETRACTION INSIDE AN ACCEPTED ANCHOR. The verdict stands; the rationale it was argued on does
    # not, and the two must not be conflated.
    "additivity_retracted": {
        "claimed": (
            "'ADDITIVE by construction: the union is ranked, so a core entry that still ranks in the "
            "top k is retained. A false firing costs one extra read, NEVER a displaced correct "
            "answer.'"
        ),
        "falsified_by": (
            "both held-out losses had core_topk_before=5 and core_entries_retained=0 -- archival "
            "out-scored core ~2x, so global ranking evicted EVERY core entry. On 89-student-9 the "
            "control answered 'sailing' correctly from a core entry the merge dropped; the arm "
            "answered 'gaming' from archival. The evidence was absent, not overlooked"
        ),
        "reasoning_error": (
            "argued from a property of the MECHANISM (union ranking preserves relative order) to an "
            "OUTCOME (a used entry survives). Order preservation says nothing about which entries fall "
            "below the cut. Same shape as the delivery defect: reasoning about the gate instead of "
            "measuring the thing downstream"
        ),
        "and_displacement_is_the_expected_case": (
            "not an edge case -- the anchor only fires when the primary store scored WEAKLY, which is "
            "exactly when the sibling is likely to dominate it"
        ),
        "corrected": (
            "the merge returns the globally best top-k of the union; when the sibling out-ranks the "
            "primary, primary entries are dropped. Measured trade: 15/0 on train, 9/2 on dev. "
            "NET-POSITIVE, NOT NON-DESTRUCTIVE"
        ),
        "what_survives": (
            "ties break toward the primary store, so an EQUAL-scoring sibling entry cannot evict one "
            "the model already had. That is the only guarantee, and it is narrow"
        ),
        "why_acceptance_stands": (
            "the rule requires NET dev >= 0, not zero losses, and +7 clears it. A6 was deferred for a "
            "NEGATIVE net; this is a favourable measured trade. What changed is the rationale"
        ),
        "do_not_reuse": (
            "the phrase 'additive by construction' is retracted for merge-style anchors. The obvious "
            "repair (reserve k slots for the primary, or demand a margin) is a NEW MECHANISM needing "
            "its own frozen criteria: a 'retain one primary entry' guard would have fixed both losses "
            "AND changed the 24 wins, so its net effect is unknown, not positive"
        ),
        "scope": (
            "n=2 of 37 dev firings. Enough to falsify a universal claim -- one counterexample does "
            "that -- and enough to characterise the mode, since both show the identical signature. "
            "NOT enough to estimate a displacement rate across domains"
        ),
    },

    # Criterion 3 required all three per case, not inferred from the aggregate.
    "mechanism_conditions": (
        "A9 fired AND the merged result was ATTESTED DELIVERED into the real message list",
        "archival entries were PROMOTED into the returned top-k",
        "the case fails in the paired control",
    ),
    "similarity_jumps": {
        "43-healthcare-13": (0.034, 0.691),
        "95-student-15": (0.211, 0.713),
        "8-customer-8": (0.278, 0.709),
        "108-student-28": (0.140, 0.617),
    },
    # Recorded rather than absorbed into the net: both fired with promoted entries, so they are
    # genuine displacement by the merge, not gate failures.
    "dev_losses": ("123-student-43", "89-student-9"),

    # THE DEFECT. Read this before trusting any anchor's telemetry, this one included.
    "delivery_defect": {
        "what": "v1/v2 computed the merged payload on all 78 firings and delivered it on NONE",
        "why": (
            "the gate ran ~190 lines BELOW the point where the model prompt is assembled and wrote to "
            "step_record['tool_results'] -- the telemetry sidecar. Its write to the real "
            "execution_results was dead code: followed by `step_count += 1; continue`, so the loop "
            "restarted and rebuilt the list before anything consumed it"
        ),
        "how_it_looked": (
            "xcm_replaced=True was TRUE -- of the sidecar. Firing telemetry can be right about itself "
            "and wrong about the world"
        ),
        "the_tell_walked_past": (
            "the pre-fix arm scored EXACTLY the control's 35.0%, to two decimals, across two "
            "independent control runs -- the signature of a computationally inert anchor. And a probe "
            "found the model answered correctly on ALL TEN reconstruction cells while the real system "
            "always failed. 'My reconstruction always succeeds and the real system always fails' is a "
            "DELIVERY finding, not a framing finding"
        ),
        "voided": (
            "A9-R +0.00pp -> 'the model will not use the evidence'",
            "A9-R retrieval positive (gold-in-context 31/78 -> 50/78; measured on tool_results)",
            "the delivery audit that claimed 'verbatim in the actual model input'",
            "A9-U comprehension limit (a re-read of absent evidence)",
            "A9-R-top1 dilution ruling (top-1 of a discarded payload)",
            "the two-model probe (framing vs capability -- fed a passage the run never sent)",
            "sysprompt localisation (same void premise)",
        ),
        "fix": "v3 changes WHERE the result is applied and nothing else; _try_xcm_merge is byte-identical",
        "single_source_of_truth": (
            "execution_results is authoritative and tool_results is DERIVED from it. The prior form "
            "ran two independent index checks against two lists that need not be the same length, so "
            "the telemetry could record a positional replacement while the model input took an append "
            "fallback"
        ),
        "general_rule": "docs/CONSUMER_BOUNDARY_RULE.md -- validate at the CONSUMER boundary",
    },

    # Gate 0 ran BEFORE mechanism and accuracy: evidence validity first.
    "gate0": {
        "attested_deliveries": {"train": (55, 55), "dev": (23, 23)},
        "delivery_violations": 0,
        "append_fallbacks": 0,
        "trigger_unchanged": "all firings core_max < 0.30",
        "distinct_base_keys": "arm 9870c69d... vs control 526c02d2... -- no cache cross-contamination",
    },

    # BOTH ITEMS CLOSED, and the first one is instructive: it was never a code gap.
    "closed": {
        "prereq_delivery": (
            "CLOSED BY MEASUREMENT, and the original diagnosis was WRONG. The 11 'unattested' firings "
            "were STALE ARTIFACTS from the previous day's pre-capture run, left in the sidecar dir by "
            "a snapshot-cache HIT -- the v3 run skipped all 27 prereqs, so no prereq episode had "
            "executed at all. Episodes now carry run_id and Gate 0 counts only the current run: it "
            "reads 55/55 and 23/23 attested with ZERO unattested. A forced-rebuild run confirms the "
            "prereq path does capture (1 firing, 1 attested). So the fix was to the MEASUREMENT, not "
            "to the code -- and 'unverified' had been the right label for it either way"
        ),
        "three_shard_confirmation": (
            "CLOSED. Reproduced the projected corpus cells exactly, and measured non-interference "
            "rather than inferring it -- see non_interference"
        ),
    },

    # Worth carrying: a REPRODUCIBILITY floor, discovered while closing the prereq item.
    "noise_floor": (
        "the forced-rebuild run scored 61.8% on the SAME arm where the cache-HIT run scored 60.7% -- "
        "1.1 pp from store-build nondeterminism, not from A9. So single-shard differences under ~1 pp "
        "are within noise. This is why sharding is the default protocol, and why arm and control must "
        "SHARE a snapshot cache when the anchor is query-only (A9) and must NOT when it is not"
    ),

    # Kept for the record: what these said before they were closed.
    "was_open": (
        (
            "PREREQ-PHASE DELIVERY IS UNVERIFIED. 11 train / 3 dev firings occur while building the "
            "store, where prompts are not captured, so attestation ABSTAINS rather than reporting a "
            "pass it cannot support. Unattested means NOT MEASURED, which is not the same as not "
            "delivered -- conflating those two is the original A9 error in mirror image. Constrained "
            "to prereq-only so a query-phase gap cannot hide there. Fix: capture prompts on prereq."
        ),
        (
            "THREE-SHARD CONFIRMATION NOT RUN. kv and rec_sum are unchanged by construction, but the "
            "whole-corpus figure is arithmetic until that is measured. It is cheap and it is the "
            "right next measurement."
        ),
    ),

    "scope": (
        "VECTOR ONLY, enforced in code rather than by policy: the mechanism returns None unless the "
        "backend is vector, because the trigger needs per-entry similarity scores and the kv and "
        "rec_sum containers do not expose them."
    ),
    "reproducible": (
        "both splits ran under the sharding protocol (--store-workers 1, distinct ports, separate "
        "base_keys), so the figures are byte-reproducible -- dev replicated 52.5% exactly across two "
        "independent runs"
    ),
}


# This is the most useful thing A8 produced, and it was asked REGARDLESS of the verdict. Measured from
# the destructive-call audit, which snapshots the live container -- not from the gate's decline reasons
# (stranded by the observability defect) and not by replay (an exec-log replay misses model-written
# content and produced impossible size=0 readings at a clear).
#
# Of the archival clears that STILL execute on the accepted incumbent: 5 of 5 are at 50/50 capacity
# with NO exact duplicate. So 100% of the remaining clear-time residual is out of layer 1's reach by
# construction. Two of the five are the class-3 events the original diagnosis predicted would fall
# through; one is new -- it HAD duplicates, layer 1 spent all four, and the pressure returned.
CAPACITY_LADDER = {
    "principle": (
        "when memory is under capacity pressure, preserve information before resorting to destructive "
        "recovery"
    ),
    "layers": (
        (1, "dedup", "remove a provably redundant copy", "ACCEPTED as A8"),
        (2, "safe concat / consolidation", "combine entries losslessly", "NEXT -- 5 states identified"),
        (3, "destructive fallback", "clear, having exhausted the alternatives", "the status quo"),
    ),
    "confirmed_by": "live destructive-call audit on the accepted incumbent",
    "remaining_residual": "5 of 5 executing clears are at capacity with NO duplicate",
    "layer_1_is_exhausted_where": "the container is full and holds nothing redundant",
    "why_layer_1_alone": (
        "dedup-only by design, so its effect is attributable in isolation. Bundling consolidation into "
        "the same arm would have made the two indistinguishable."
    ),
}


# --------------------------------------------------------------------------------------------
# E1: the ACCEPTED anchor on the second objective
# --------------------------------------------------------------------------------------------
#
# E = EFFICIENCY. Its objective is accuracy per LLM call, not accuracy, so it is accepted on its
# own criterion -- no task loss + provably useless work removed + no new harm -- and a 0 pp accuracy
# delta is a PASS rather than a null result. See docs/EFFICIENCY_CLASS.md.
#
# ACCEPTED, and kept OUT of PROGRESSION deliberately: adding a 0 pp row to an accuracy table would
# read as a failed accuracy anchor. Different objective, different table. That separation is about
# the OBJECTIVE, not about strength of evidence.
#
# One thing to keep straight, because it caused three labelling errors in the working repo: E1 the
# EFFICIENCY anchor (deterministic-outcome memoization, gate key `on_memoized_deterministic_call`,
# its own standalone policy) is a different anchor from the accuracy anchor that was RENAMED from
# "E1" to "A5" (archival-full duplicate eviction, `enable_archival_evict_duplicate`). Two anchors,
# one name, at different times. See NAMING_TRAPS.
#
# It is a SUPPRESS: it cancels a tool call that cannot inform anything. The justification is
# determinism -- same state, same action, same outcome, always -- so executing the call a second
# time buys strictly nothing and costs a step.

E1 = Anchor(
    name="E1",
    locus="redundancy/deterministic_outcome/known_absent_target",
    attribution=(
        "the model re-issues a removal for a target it has already been told is absent; "
        "the tool is deterministic, so the second call cannot return anything new"
    ),
    incision_point=IncisionPoint.POST_GENERATION_PRE_EXEC,
    action=Action.SUPPRESS,
    params={
        "memo_key": "(container, target)",
        "predicate": "recorded not-found for this key AND target still absent from LIVE state",
        "replay": "the tool's VERBATIM result string",
        "fail_open": True,
    },
    notes=(
        "Suppresses a useless tool call. The general pattern is `known state + known action + "
        "deterministic tool -> reuse the recorded outcome instead of executing again`; it is "
        "instantiated for the one class whose determinism is verified against live state, a "
        "remove whose target is absent. The withheld call is NOT removed from the proposal "
        "record -- the tool's own result string is replayed verbatim in its place, so the "
        "model's view is byte-identical and behaviour is preserved BY CONSTRUCTION. That is "
        "what distinguishes it from A3, which removes the call outright and lets the model see "
        "nothing. ACCEPTED ON EFFICIENCY: +0.00 pp with ZERO FLIPS and 16 of 16 redundant "
        "executions eliminated. Fired in PREREQ episodes only -- see E1_EVIDENCE for the scope, "
        "which must travel with the numbers."
    ),
)

# Measured in the working repo, not re-measured here. Stated separately from the accuracy
# progression because the objective is different and mixing them would misrepresent both.
#
# READ THE SCOPE BEFORE QUOTING THE NUMBERS. +0.00 pp is over the full 303-case scored train
# split, but all 16 firings are in TWO vector PREREQ episodes -- so the anchor fired ZERO times
# in any SCORED episode. The 0 pp is therefore CONSISTENT WITH safety rather than evidence of
# safety at scale, and the honest phrasing is "16 of 16 redundant executions eliminated in the
# population where it fires; scored accuracy unchanged."
E1_EVIDENCE = {
    "objective": "accuracy per LLM call",
    "acceptance_class": "efficiency",
    "criteria_passed": "8/8",
    "accepted": True,
    # ACCEPTED IS NOT ACTIVE, and the distinction is load-bearing.
    "active_in_frozen_policy": False,
    "why_inactive": (
        "its gate key on_memoized_deterministic_call is absent from the frozen policy with "
        "gate_default=false, so it fires 0 times behind every accuracy number in this repo -- none "
        "of them are affected either way. This looks like an ASSEMBLY OVERSIGHT rather than a "
        "decision: one missing gate key in the same policy build that also carried an "
        "env-only-flag bug. It cannot be proven from artifacts, so it is an OPEN ITEM, not a claim."
    ),
    "to_reactivate": (
        "re-confirm the three EFFICIENCY_REQUIREMENTS against the CURRENT stack -- the last "
        "confirmation was against A1-A4, and A5/A7 have since changed the trajectories it operates "
        "on. The opportunity still exists: 15-23 repeated-identical remove calls per shard, plus 49 "
        "not-found results on kv train."
    ),
    "task_delta_pp": 0.00,
    "flips": 0,
    "calls_withheld": 16,
    "calls_sent_to_execute": 0,
    "executed_redundancy_before": 16,
    "executed_redundancy_after": 0,
    "splice_len_ok": "16/16",
    "train_n_scored": 303,
    "firing_episodes": ("vector_prereq_13-healthcare (10)", "vector_prereq_4-customer-4 (6)"),
    "firings_in_scored_episodes": 0,
    "backends_fired": ("vector",),
    "backends_claimed": ("kv", "vector"),
    "dev_run": None,
    "policy_diff": 'one key: "on_memoized_deterministic_call": true',
    "caveats": (
        ("No frozen pre-registration document exists for this anchor, unlike A1-A4; the "
         "acceptance conditions are transcribed from the verdict script that produced them."),
        ("No run artifacts or job id are recorded in the working repo, so the numbers are "
         "transcribed rather than recomputable here."),
        ("kv is claimed in the gate's backend metadata but has ZERO observed live firings; its "
         "determinism is unit-tested only."),
        ("The zero-flip acceptance bar depends on temperature=0.001 determinism (variance floor "
         "is literally zero on this line). Under sampling, a 0 pp / zero-flip arm is "
         "unobtainable and this criterion does not transfer as stated."),
        ("The step budget here is 20 (official). The published run used 15. A tighter budget "
         "returns a scarcer resource, so E1 may score BETTER at 15 -- untested, and the "
         "mechanism is indirect because the firings are in unscored prereq episodes."),
    ),
}


# --------------------------------------------------------------------------------------------
# Call cost -- the efficiency class's own criterion, measured on the ACCURACY stack
# --------------------------------------------------------------------------------------------
#
# THE STACK IS NOT UNIFORMLY CHEAPER, and reporting only the saving would be selective. A7 SAVES
# calls because landing a blocked append ends a retry loop; A5 SPENDS calls (evict, then retry) to
# buy accuracy. Both facts belong in the metric.
#
# This also explains why the efficiency anchor being inactive matters for a planned total-LLM-call
# comparison against baseline: that metric IS the efficiency class's acceptance criterion, so the
# efficiency anchors have to be live for it to mean anything.
CALL_COST = {
    "budget": 20,               # max_steps_per_turn, the official MAXIMUM_STEP_LIMIT
    "budget_ever_binding": False,   # force_quit is 0 everywhere, so this is free-running cost
    "measurements": {
        "A7_rec_sum_train": {"n": 109, "steps": 239, "control_steps": 261,
                             "delta": "-22 (-8.4%)", "accuracy_bought": "+7.34 pp",
                             "why": "landing a blocked append ends a retry loop"},
        "A7_kv_vector_train": {"steps": "identical to control", "delta": "0",
                               "why": "backend-gated, so it cannot act here"},
        "A5_vector_dev": {"n": 40, "steps": 105, "control_steps": 91,
                          "delta": "+14 (+15%)", "accuracy_bought": "+17.50 pp",
                          "why": "evict-then-retry spends a call to save a write"},
    },
    "caveats": (
        ("`steps` covers the QUERY PHASE ONLY. A5 and A7 mostly fire in the prereq phase, and E1's "
         "16 withheld calls lived there entirely, so the prereq phase needs separate counting "
         "before any total-call claim."),
        ("The 20-step ceiling is never reached on query cases (force_quit = 0 everywhere), so call "
         "count is a free-running cost rather than a truncation artifact."),
        ("No aggregate token or per-episode cost figure exists -- only call-level counts. Never "
         "state a percentage token saving."),
    ),
}


# --------------------------------------------------------------------------------------------
# The measured progression
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Round:
    """One iteration of the loop. `anchor is None` means nothing was installed.

    `dev_correct` is the same stack scored on the independent dev split, where present. It is not
    an afterthought: dev is the accept/reject criterion, so a round without a dev number has not
    been judged under the current rule.
    """

    tag: str
    anchor: Anchor | None
    train_correct: int
    train_n: int
    signal_summary: str
    note: str = ""
    mined_coverage: float | None = None  # share of residual the vocabulary explains
    dev_correct: int | None = None
    dev_n: int = 84
    # True when the WHOLE-CORPUS figure is arithmetic on a measured shard rather than a corpus run.
    # Defaults False, so a row is measured unless it says otherwise. Introduced for T9/A9, whose
    # vector shard is measured but whose corpus total awaits a three-shard confirmation -- putting a
    # projection in this table unflagged is exactly the kind of quiet upgrade this repo tests against.
    projected: bool = False

    @property
    def train_acc(self) -> float:
        return 100.0 * self.train_correct / self.train_n

    @property
    def dev_acc(self) -> float | None:
        if self.dev_correct is None:
            return None
        return 100.0 * self.dev_correct / self.dev_n


PROGRESSION: Sequence[Round] = (
    Round(
        tag="T0", anchor=None, train_correct=91, train_n=303, dev_correct=14,
        signal_summary="native agent -- no prompt, no anchors",
        note=(
            "Baseline, and now MEASURED rather than quoted -- see NATIVE_BASELINE. This line has NO "
            "global memory prompt of any kind. 91/303 replaces a long-quoted 88/303 for which no "
            "artifact on disk existed; dev reproduced exactly at 14/84."
        ),
        mined_coverage=0.78,
    ),
    Round(
        tag="T1", anchor=A1, train_correct=102, train_n=303, dev_correct=16,
        signal_summary="capacity/container/no_remaining_capacity",
        note=(
            "Miner rank 1 by linked downstream loss. Its own step is +3.63 pp against the MEASURED "
            "baseline, not the +4.62 pp computed from the old quoted one."
        ),
        mined_coverage=0.78,
    ),
    Round(
        tag="T2", anchor=A2, train_correct=105, train_n=303, dev_correct=16,
        signal_summary="existence/identifier/not_found",
        note="Residual CREATED BY A1: all 39 exposed cases are self-inflicted.",
        mined_coverage=0.61,
    ),
    Round(
        tag="T3", anchor=A3, train_correct=117, train_n=303, dev_correct=19,
        signal_summary="permission/identifier/duplicate",
        note="Residual CREATED BY A1 too: 59/59 collisions on keys A1 injected.",
        mined_coverage=0.13,
    ),
    Round(
        tag="T4", anchor=None, train_correct=117, train_n=303, dev_correct=19,
        signal_summary="no locus survives S1/S2 -> EXPAND_ATTRIBUTION",
        note=(
            "NOTHING INSTALLED, and that is a result. Every locus was both S1-STOP (below the "
            "10% coverage floor) and S2-STOP (majority already settled), so phase_switch "
            "returned EXPAND_ATTRIBUTION: the bottleneck had moved from policy to signal."
        ),
        mined_coverage=0.13,
    ),
    Round(
        tag="T5", anchor=A4, train_correct=128, train_n=303, dev_correct=22,
        signal_summary="no_tool_call_at_all (expanded vocabulary)",
        note="The expansion worked: a detector the error-keyed miner could not see.",
        mined_coverage=0.03,
    ),
    Round(
        tag="T6", anchor=A5, train_correct=132, train_n=303, dev_correct=29,
        signal_summary="capacity/container/no_remaining_slots",
        note=(
            "A1's relocation target saturates, so A1's own repair has nowhere to put the payload. "
            "Eviction is admitted ONLY where provably lossless. Mechanism causally validated on "
            "dev: 30 evictions, blocked write landed 30/30, lossless invariant held 30/30. The "
            "+8.33 pp dev step is the largest in the progression."
        ),
    ),
    Round(
        tag="T7", anchor=A7, train_correct=139, train_n=303, dev_correct=34,
        signal_summary="size/blob/append_would_exceed_cap",
        note=(
            "The re-mine after A5 relabels the largest remaining locus as blob overflow on the one "
            "backend with no second container to relocate into. Confirmed END TO END across all "
            "three backends: on the 258 cases it cannot reach the arm is a BIT-EXACT no-op. "
            "+7.34 pp on rec_sum, its only target, diluting to +2.31 pp corpus-wide from this "
            "incumbent."
        ),
    ),
    Round(
        tag="T8", anchor=A8, train_correct=144, train_n=303, dev_correct=34,
        signal_summary="capacity/container/clear_proposed_at_capacity",
        note=(
            "The kv instantiation of the capacity principle, and the first anchor to intercept a "
            "DESTRUCTIVE action before it executes rather than repair a failed one. Mechanism is the "
            "best-validated in the stack (15/15, lossless on every firing, byte-reproducible across "
            "two runs); support is the narrowest (one kv/student cell). Dev is a measured NO-OP, so "
            "the dev column does not move -- criterion 2 is satisfied vacuously. See A8_EVIDENCE."
        ),
    ),
    Round(
        tag="T9", anchor=A9, train_correct=159, train_n=303, dev_correct=41,
        signal_summary="retrieval/similarity/core_max_below_threshold",
        note=(
            "MEASURED on all three shards, case by case: 159/303 train and 41/84 dev, +15/-0 and "
            "+9/-2. These cells were a PROJECTION from the vector shard for one commit, and the "
            "confirmation run reproduced them EXACTLY -- a validation of the projection method rather "
            "than a correction to it. kv and rec_sum came back byte-identical between arms (35/35 and "
            "70/70 train; 6/6 and 14/14 dev) with ZERO off-target gate firings, so non-interference is "
            "MEASURED rather than inferred from the backend guard in code. Vector shard itself: train "
            "60.7% vs 43.8% (+16.9 pp), dev 52.5% vs 35.0% (+17.5 pp). "
            "ONE RETRACTION TRAVELS WITH THIS ROW: the 'additive by construction / never a displaced "
            "correct answer' rationale is FALSE -- both dev losses evicted the ENTIRE core set. The "
            "anchor is accepted on a favourable measured trade, not on non-destructiveness. See "
            "A9_EVIDENCE['additivity_retracted']. "
            "Read-side like A2, but triggered on a SCORE rather than an error string: the retrieve "
            "succeeded and simply returned nothing useful, so nothing raises. A2 reprompts the model "
            "to search archival; A9 searches it and returns merged evidence, telling the model "
            "nothing. It changes what the model is SHOWN, leaving the store "            "untouched. Also the anchor whose first two versions computed the right answer on 78 "
            "firings and delivered it on zero, voiding four downstream experiments. See A9_EVIDENCE "
            "and docs/CONSUMER_BOUNDARY_RULE.md before quoting any figure here."
        ),
    ),
)

# --------------------------------------------------------------------------------------------
# Which measurement generation a number comes from, and why it is the sharded one
# --------------------------------------------------------------------------------------------
#
# Two generations of paired train runs exist. Early ones ran the whole corpus in one job; later ones
# SHARD BY BACKEND -- one isolated job per backend, each serial internally, and all shards submitted
# CONCURRENTLY. Sharding was adopted for wall-clock and turned out to buy something more important:
# WHOLE-CORPUS RUNS ARE NOT BYTE-REPRODUCIBLE. Two runs of the SAME effective policy differed even
# where the treatment's guard fired zero times, because whole-corpus execution adds cross-chain
# sequencing that is not byte-stable even with store construction forced serial. Every sharded pair,
# by contrast, is bit-identical except in the single shard where the treatment actually fires.
#
# So a paired train number from a whole-corpus run should not be quoted. The one exception in this
# progression is A5's 132/303, which is its own acceptance run and is the figure the working repo's
# record settles on; the 131/303 that appears in some tables is the LATER post-repair incumbent, one
# case lower. Both are real and they are one case apart -- the progression uses 132 because that is
# the arm A5 was accepted on and the line A7 chains off.
MEASUREMENT_GENERATIONS = {
    "authoritative": "sharded, submitted concurrently",
    "a5_correct": 132,
    "a5_post_repair_incumbent_correct": 131,
    "why": (
        "Whole-corpus paired runs are not byte-reproducible: two runs of the same effective policy "
        "diverge even when the treatment fires 0 times. Sharded pairs are bit-identical except in "
        "the shard where the treatment fires, so sharding buys DETERMINISM and not merely speed."
    ),
    "concurrency": (
        "A 3-backend x 2-arm contrast is SIX SIMULTANEOUS JOBS, one GPU each, not six sequential "
        "ones. Every sub-job is independent -- own process, own store, own snapshot-cache key -- and "
        "chains are contiguous per backend so no chain can observe another backend's store. The A7 "
        "line ran its arms sequentially and paid roughly a 4x wall-clock penalty for it."
    ),
    "one_stochastic_channel": (
        "A7's compaction call is the only non-determinism anywhere in the stack: 1 of 26 compactions "
        "decoded differently (~4%). Rare, not pervasive -- an earlier claim that A7 arms are never "
        "byte-reproducible was generalised from the single small-n split where it happened to fire."
    ),
}


# --------------------------------------------------------------------------------------------
# The re-mine between rounds -- the step that makes this a LOOP rather than a ranked worklist
# --------------------------------------------------------------------------------------------
#
# After every accepted anchor the residual is re-mined and the ranking recomputed, because an
# accepted anchor CHANGES the failure distribution -- A1 created the residuals A2 and A3 then
# addressed. Each ledger is frozen with its own fingerprint, so "the miner chose this locus" is
# checkable rather than asserted.
#
# Both ledgers shipped here were produced on the A1-A5 line, and they are the evidence that the
# ranking drives the work rather than following it. `n0t6` ranks `size/item/exceeds_per_item_limit`
# first -- the locus A6 was built for. That A6 was subsequently REJECTED on its remedy does not
# retract the ranking: the signal is real and still unaddressed (see DEFERRED). `n0t7` shows the
# residual after that attempt, and relabels the largest remaining locus as the blob overflow that
# became A7.
#
# `settled` counts matter more than raw event counts: an event whose failure was subsequently
# repaired is not residual. See docs/THE_LOOP.md and anchoropt/learning/remine_incumbent.py.
REMINE_LEDGERS = {
    "n0t6": {
        "iteration": "N0-T6",
        "after_anchor": "A5",
        "fingerprint": "869316d8730866c4",
        "path": "rounds/A6_deferred/ledger/n0t6_locus.json",
        "rank1_locus": "size/item/exceeds_per_item_limit",
        "rank1_events": 71,
        "targeted_by": "A6",
        "note": (
            "rank 1 is the locus A6 targeted -- the miner selected it, and the frozen ledger proves "
            "the ranking was not written to fit a chosen mechanism. A6's remedy was later rejected; "
            "the SIGNAL stands."
        ),
    },
    "n0t7": {
        "iteration": "N0-T7",
        "after_anchor": "A6 (attempted)",
        "fingerprint": "edea7bb67e43e48b",
        "path": "rounds/T7_A7_blob_overflow/ledger/n0t7_locus.json",
        "rank1_locus": "size/item/exceeds_per_item_limit",
        "rank1_events": 55,
        "targeted_by": "A7",
        "note": (
            "the entry-length locus falls 71 -> 55 under the A6 attempt, and the largest remaining "
            "locus is relabelled blob overflow, which became A7. With A6 dropped, the entry-length "
            "events return -- that is the 1.65 pp train cost recorded in DEFERRED."
        ),
    },
}


# --------------------------------------------------------------------------------------------
# The two splits, and why the second one is called DEV
# --------------------------------------------------------------------------------------------
#
# Dev fold: customer/kv + finance/rec_sum + student/vector, n=84, leakage 0.
#
# IT IS CALLED DEV, NOT TEST, BECAUSE IT IS USED TO ACCEPT AND REJECT ANCHORS. That makes it a
# VALIDATION set in the standard-ML sense, so the reported dev figure is partly selected-on and is
# NOT a clean generalization estimate. Calling it "held-out" or "test" would claim a property the
# procedure does not have -- A6 was dropped ON dev evidence, which is exactly the selection.
#
# An untouched final number would require a THIRD reserved split. It does not exist. Stated here so
# the limitation travels with every dev number rather than being rediscovered later.
#
# WHAT THE FOLD MEASURES. One domain chain per backend, so it measures DOMAIN TRANSFER rather than
# within-distribution generalization. Any anchor whose effect depends on chain multiplicity or on
# how deeply the store saturates will move on it. A leave-one-chain-out protocol is the honest fix
# and is deferred, not done.
SPLITS = {
    "train": {"n": 303, "role": "discovery -- mine loci, fit and select the remedy"},
    "dev": {
        "n": 84,
        "role": "accept/reject criterion",
        "composition": "customer/kv + finance/rec_sum + student/vector, one chain per backend",
        "leakage": 0,
        "is_validation_not_test": True,
        "caveat": (
            "Used for accept/reject, so it is a VALIDATION set: the reported figure is partly "
            "selected-on and is not a clean generalization estimate. No third reserved split "
            "exists, so no untouched final number is available."
        ),
        "measures": "domain transfer, not within-distribution generalization",
        "one_case_pp": 1.19,
    },
}

# The cumulative result of the accepted stack, on both splits. ONE progression, ONE rule.
#
# No cumulative p-value is quoted, and the reason is not modesty: THE BENCHMARK IS DETERMINISTIC.
# Byte-identical execution logs were measured across independent sharded runs spanning days, jobs
# and hosts, so there is no sampling process over which a null distribution is defined. A sign test
# on 84 fixed cases asks how surprising a split would be if the flips were coin tosses, and they are
# not coin tosses -- they are the deterministic consequence of a policy change on a fixed corpus.
# Report the paired counts and the mechanism instead. (The p = 0.0266 that appears in earlier write-
# ups belongs to the A1-A4 sub-chain and is retained in those rounds' own specs as a diagnostic.)
# --------------------------------------------------------------------------------------------
# The native baseline, MEASURED -- it had been quoted for a long time and never run
# --------------------------------------------------------------------------------------------
#
# Six concurrent shards on the historical prompt-free control policy. The result moved the baseline,
# so it moves every cumulative figure computed from it -- recorded rather than quietly corrected.
#
#     train   quoted 29.04% (88/303)  ->  MEASURED 30.03% (91/303)   +3 cases
#     dev     quoted 16.67% (14/84)   ->  MEASURED 16.67% (14/84)    exact
#
# Dev reproducing exactly while train moves by 3 is consistent with the two known causes: the quoted
# figure predates a harness repair, and it predates the sharding protocol -- and whole-corpus runs are
# not byte-reproducible while sharded ones are. So the old value was most likely from a whole-corpus
# run and was never a reproducible number. The exact dev match suggests the divergence is confined to
# repaired cases rather than general drift.
#
# CONSEQUENCE: cumulative train is +17.49 pp to A8 (+15.84 pp to A7), not the +16.83 pp previously
# quoted, and A1's own step is +3.63 pp rather than +4.62 pp. Nothing above A1 shifts, because every
# later per-anchor delta was already measured against measured neighbours.
NATIVE_BASELINE = {
    "measured": True,
    "train_correct": 91,
    "train_n": 303,
    "train_acc": 30.03,
    "dev_correct": 14,
    "dev_n": 84,
    "dev_acc": 16.67,
    "previously_quoted_train": 88,
    "previously_quoted_train_acc": 29.04,
    "dev_reproduced_exactly": True,
    "shards": 6,
    "why_it_moved": (
        "the quoted figure predates a harness repair and predates the sharding protocol; whole-corpus "
        "runs are not byte-reproducible, so it was probably never a reproducible number"
    ),
    # "native" is not literally zero-intervention, and that must travel with the baseline.
    # MEASURED per backend, three concurrent shards, all rc=0. This localizes the +3 that the
    # record had noted but never pinned down -- and it turns out the aggregate was hiding structure.
    "per_backend": {"kv": (18, 105), "vector": (19, 89), "rec_sum": (54, 109)},
    "where_the_drift_lives": (
        "ENTIRELY in kv. vector (19/89) and rec_sum (54/109) reproduce the shipped artifact EXACTLY. "
        "And the net +3 conceals 5 case-level changes: 4 gained and 1 lost, all five in the SAME "
        "scenario cell (healthcare/kv). A net figure would have read as 'three cases drifted'; the "
        "per-case diff says one cell was re-scored. That is why the paired shape is reported beside "
        "every delta -- the same rule the acceptance criteria apply to arms, applied to a baseline."
    ),
    "drift_cases": {
        "gained": ("34-healthcare-4", "44-healthcare-14", "47-healthcare-17", "54-healthcare-24"),
        "lost": ("30-healthcare-0",),
    },
    "caveat": (
        "one gate (`on_core_full_rerouted`) resolves True even under an all-false gate map because it "
        "has a CODE-LEVEL default. It is identical in the baseline and in every arm, so it confounds "
        "no comparison -- but 'native' is therefore not literally zero-intervention."
    ),
}


CUMULATIVE = {
    "n_anchors": 8,
    "train_from": 30.03,
    "train_to": 52.48,
    "train_delta_pp": 22.44,
    "train_correct": (91, 159),
    "train_n": 303,
    "dev_from": 16.67,
    "dev_to": 48.81,
    "dev_delta_pp": 32.14,
    "dev_correct": (14, 41),
    "dev_n": 84,
    # FULLY MEASURED. This was True for exactly one commit, while A9's corpus contribution was
    # arithmetic on its vector shard. The three-shard confirmation reproduced those figures EXACTLY --
    # so the numbers never moved, only their standing did. The flag stays rather than being deleted
    # because "measured or projected?" must be answerable for any round, not just the current one.
    "endpoint_projected": False,
    "endpoint_confirmed": (
        "three-shard paired run, case by case. kv and rec_sum byte-identical between arms (35/35 and "
        "70/70 train; 6/6 and 14/14 dev) with ZERO off-target gate firings and zero declines, so "
        "non-interference is MEASURED rather than inferred from the backend guard in code -- a "
        "distinction this line of work has been burned by before (see A9_EVIDENCE.delivery_defect)"
    ),
    # T8 is still the last endpoint this repo can RECOMPUTE OFFLINE from shipped artifacts: A9's shard
    # results remain on the cluster. That is a different property from being measured, and it is the
    # one a reproduction cares about, so it keeps its own key.
    # See rounds/T9_A9_xcontainer_merge/result/NOT_SHIPPED.md.
    "measured_endpoint": {
        "through": "T8",
        "n_anchors": 7,
        "train": (144, 303),
        "train_acc": 47.52,
        "dev": (34, 84),
        "dev_acc": 40.48,
        "note": (
            "six-shard concurrent run, all rc=0, AND recomputable offline in this repo -- which is why "
            "tests reconcile shipped per-case artifacts against this rather than the T9 total"
        ),
        "recomputable": True,
    },
    # Every row of the progression is now backed by an artifact, and the chain is monotone
    # non-decreasing on BOTH splits at every step -- the property the acceptance rule is designed to
    # produce, achieved without any zero-loss requirement (A5 loses 3 on train and 2 on dev; A7 loses
    # 13; A8 loses 4).
    "monotone_on_both_splits": True,
    "p_value": None,
    "p_value_note": (
        "Not quoted: the benchmark is deterministic (byte-identical exec logs across independent "
        "runs), so there is no sampling distribution for a null. Paired counts and mechanism carry "
        "the claim. A7's compaction call is the single stochastic channel and licenses no inference."
    ),
    # T9's policy: T8's file plus exactly one gate key. The T8 sha is kept because it is the hash of
    # the fully-measured endpoint, which is the one a reproduction should start from.
    "policy_sha256": "c610e74fedf36899a9ca5b493c55df4e2ce4bb0649bc5c1327c5d24e1830c3c3",
    "policy_sha256_measured_endpoint_t8": (
        "26792094baec5d50ebf3783a415371d3058b59e306f20c30b587b61023812143"
    ),
}

# The per-backend split of the accepted stack, MEASURED rather than derived.
#
# Six shards, concurrent, all rc=0: 3 backends x {train, dev}, arm = the frozen accepted stack. Every
# cell matched the value previously derived from the corpus total, and both totals landed on the
# published figures -- so this replaces a derivation with a measurement rather than correcting one.
#
# It also settled a disagreement. `scripts/derive_per_backend.py` walks a SECOND route (last shipped
# artifact + recorded exposures) and put kv at 34.29% and rec_sum at 63.30%, one case off each. The
# run says the total-derived cells were right and that walk was wrong. Both routes agreed on the
# TOTAL and on vector, which is why a total-only check is insufficient: two compensating one-case
# errors sum correctly. Kept as a worked example of a cross-check earning its keep by disagreeing.
PER_BACKEND = {
    "measured": True,
    "shards": 6,
    "all_rc_zero": True,
    "train": {"kv": (35, 105), "vector": (39, 89), "rec_sum": (70, 109)},
    "dev": {"kv": (6, 24), "vector": (14, 40), "rec_sum": (14, 20)},
    "matched_the_derivation": True,
    "derivation_walk_was_wrong_by": {"kv": -1, "rec_sum": +1},
    "why_the_walk_drifts": (
        "the shipped T5 artifact predates a harness repair that the measured line includes -- the "
        "same one-case divergence already recorded for the T0 baseline. A property of the inputs, "
        "not of the arithmetic."
    ),
    "caveat": (
        "Per-backend splits stay DESCRIPTIVE. A cell this small cannot reach significance at any "
        "effect size, and the anchors reach the backends very unevenly -- A7 acts on rec_sum alone, "
        "A5 on vector, A8 on one kv cell. Measuring them does not make them independent evidence."
    ),
}


# Berkeley's public leaderboard, MEMORY task, as of 2026-04-12. External data, transcribed -- kept
# here so the README table has one source and a date stamp rather than a floating claim.
#
# The launch blog reports Claude 3 Sonnet at 53.55/63.87/67.74 per backend. That is a SNAPSHOT the
# live table has superseded in both directions, and the repo cited it as "the leader" for a while.
# Cite the date, always.
LEADERBOARD = {
    "as_of": "2026-04-12",
    "task": "BFCL v4 memory",
    "source": "https://gorilla.cs.berkeley.edu/leaderboard.html",
    # (model, memory overall, kv, vector, rec_sum)
    "top": (
        ("Claude-Opus-4-5 (FC)", 73.76, 70.97, 72.90, 77.42),
        ("Claude-Sonnet-4-5 (FC)", 64.95, 54.19, 57.42, 83.23),
        ("Gemini-3-Pro-Preview (Prompt)", 61.72, 59.35, 62.58, 63.23),
        ("Grok-4-0709 (FC)", 55.91, 57.42, 58.71, 51.61),
        ("GLM-4.6 (FC thinking)", 55.70, 43.87, 56.13, 67.10),
        ("Gemini-3-Pro-Preview (FC)", 54.84, 50.32, 63.23, 50.97),
        ("Claude-Haiku-4-5 (FC)", 54.41, 51.61, 55.48, 56.13),
        ("DeepSeek-V3.2-Exp (FC)", 54.19, 41.94, 61.29, 59.35),
    ),
    "superseded_blog_figures": {"kv": 53.55, "vector": 63.87, "rec_sum": 67.74},

    # HOW THE HEADLINE COLUMN IS BUILT. Verified against the table rather than assumed: the memory
    # score is the UNWEIGHTED MEAN of the three backend scores, exact on all 8 rows to within 0.0033
    # pp of rounding. So the three sub-tasks carry equal weight and equal case counts, and the
    # aggregate is a MACRO average -- not pooled over cases.
    #
    # This is worth pinning because our own totals are POOLED (144/303), and the two differ: our
    # macro mean is 47.12 while our pooled figure is 47.52. Quoting 47.52 against that column would
    # be comparing a micro average to a macro one -- a small error here, but the same class of error
    # as differencing accuracies across denominators, which is this project's oldest hazard.
    "aggregate_is": "unweighted mean of the three backend scores (MACRO, verified on all 8 rows)",
    "aggregate_max_residual_pp": 0.0033,

    "comparability": (
        "NOT like-for-like. Those runs are other models under Berkeley's own harness; ours come from "
        "the reimplemented loop on one 303-query fold. The only licensed comparison is our own "
        "control -> anchored delta, paired and within-job."
    ),
    # The one genuine cross-reference worth stating, and its limit in the same breath.
    "where_we_place": (
        "On rec_sum our 64.22% sits above 5 of the top 8. On kv (33.33) and vector (43.82) we beat "
        "none of them. That asymmetry tracks WHERE THE WORK WENT -- A7 acts on rec_sum alone, and kv "
        "has only A8 on a single cell -- not three independent results. On the AGGREGATE memory task, "
        "computed the leaderboard's own way (macro mean), we are at 47.12 and beat NONE of the top 8 "
        "-- 7.07 pp short of 8th. The per-backend win does not survive aggregation, because the "
        "aggregate weights the two backends we barely touched equally with the one we developed."
    ),
}


# The SUBSET this repo can recompute from shipped artifacts, as opposed to transcribe.
#
# T0-T5 ship their result JSONs, so `scripts/verify_progression.py` recomputes them case by case and
# fails loudly on any drift. T6 (A5) and T7 (A7) ship their policies and frozen ledgers but NOT their
# run artifacts, so their numbers are transcribed from the working repo -- as E1's are.
#
# Kept as a separate figure rather than folded into CUMULATIVE because "verified here" and "measured
# somewhere and copied" are different epistemic states, and a reader is entitled to know which is
# which for any number quoted.
RECOMPUTABLE = {
    "rounds": ("T0", "T1", "T2", "T3", "T4", "T5"),
    "anchors": ("A1", "A2", "A3", "A4"),
    "train_from": 29.04,        # the SHIPPED T0 artifact, which is the pre-correction 88/303
    "train_to": 42.24,
    "train_delta_pp": 13.20,
    "dev_from": 16.67,
    "dev_to": 26.19,
    "transcribed_rounds": ("T6", "T7"),   # T8's per-backend results now ship -- see PER_BACKEND
    # The shipped T0 result file is the OLD baseline arm (88/303). The corrected 91/303 comes from the
    # native measurement, whose artifacts are not here -- so verify_progression.py recomputes the old
    # chain from what it has, and the progression quotes the measured one. Both are stated; see
    # NATIVE_BASELINE. Do not "fix" the shipped artifact to match: it is what that arm produced.
    "shipped_t0_predates_the_native_measurement": True,
    "why": (
        "T0-T5 ship per-case result JSONs, so verify_progression.py recomputes every accuracy and "
        "the cumulative delta from them. T6/T7 ship policies and frozen ledgers but not run "
        "artifacts, so their numbers are transcribed. Different epistemic states, stated separately."
    ),
}


def delta_table() -> str:
    """Render the progression as the markdown table the README opens with.

    ONE table for the whole progression. Anchors are not split into an earlier tier and a later
    tier: every one was audited against the same four-criterion rule, and the one that failed it is
    not in the table at all -- it is in DEFERRED, which this function appends so a reader cannot see
    the gains without seeing what was rejected.
    """
    head = (
        "| round | anchor | signal | decision point | action "
        "| train | train Δ | dev | dev Δ |\n"
        "|---|---|---|---|---|---:|---:|---:|---:|"
    )
    rows = []
    train_base = PROGRESSION[0].train_acc
    prev_train = train_base
    prev_dev = PROGRESSION[0].dev_acc
    for r in PROGRESSION:
        a = r.anchor
        first = r is PROGRESSION[0]
        train_delta = "—" if first else f"**{r.train_acc - prev_train:+.2f}**"
        if r.dev_acc is None:
            dev_cell, dev_delta = "—", "—"
        else:
            dev_cell = f"{r.dev_acc:.2f} %"
            dev_delta = "—" if first or prev_dev is None else f"**{r.dev_acc - prev_dev:+.2f}**"
        rows.append(
            "| {tag} | {name} | `{sig}` | {pt} | {act} | {tr:.2f} % | {td} | {dv} | {dd} |".format(
                tag=r.tag,
                name=f"**{a.name}**" if a else "*none*",
                sig=r.signal_summary,
                pt=a.incision_point.value if a else "—",
                act=a.action.value if a else "—",
                tr=r.train_acc, td=train_delta, dv=dev_cell, dd=dev_delta,
            )
        )
        prev_train = r.train_acc
        if r.dev_acc is not None:
            prev_dev = r.dev_acc

    out = [head, *rows, ""]
    out.append(
        "{n} anchors, train {tf:.2f} % -> {tt:.2f} % (**{td:+.2f} pp**), "
        "dev {df:.2f} % -> {dt:.2f} % (**{dd:+.2f} pp**).".format(
            n=CUMULATIVE["n_anchors"],
            tf=CUMULATIVE["train_from"], tt=CUMULATIVE["train_to"],
            td=CUMULATIVE["train_delta_pp"],
            df=CUMULATIVE["dev_from"], dt=CUMULATIVE["dev_to"], dd=CUMULATIVE["dev_delta_pp"],
        )
    )
    out.append("")
    out.append("Rejected under the same rule, and not counted above:")
    for name, rec in DEFERRED.items():
        out.append(
            f"- **{name}** — {rec['status']}. train {rec['train_pp']:+.2f} pp, "
            f"dev {rec['dev_pp']:+.2f} pp, on-target dev {rec['on_target_dev_pp']:+.2f} pp. "
            f"Fails criteria {' and '.join(str(c) for c in rec['fails_criteria'])}."
        )
    return "\n".join(out)


def audit_table() -> str:
    """Render the per-anchor audit against the four acceptance criteria.

    Includes the REJECTED anchor. An audit that only lists what passed is not an audit, and the
    paired (gains, losses) are shown beside every delta because the SHAPE is what distinguishes
    an anchor that churns cases from one that is strictly additive.
    """
    head = (
        "| anchor | train net (g/l) | dev net (g/l) | attribution | verdict |\n"
        "|---|---:|---:|---|---|"
    )
    rows = []
    for name, (tr, trgl, dv, dvgl, attrib, verdict) in AUDIT.items():
        def fmt(pp, gl):
            if pp is None:
                return "—"
            cell = f"{pp:+.2f} pp"
            return f"{cell} ({gl[0]}/{gl[1]})" if gl else cell
        rows.append(
            f"| **{name}** | {fmt(tr, trgl)} | {fmt(dv, dvgl)} | {attrib or '—'} | {verdict} |"
        )
    return "\n".join([head, *rows])


if __name__ == "__main__":
    print(delta_table())
    print()
    print(audit_table())
    print()
    print("ACCEPTANCE RULE v%d -- accept iff ALL of:" % ACCEPTANCE_RULE["version"])
    for i, (name, text) in enumerate(ACCEPTANCE_RULE["criteria"], 1):
        print(f"  {i}. {name:12s} {text}")
    print()
    print("  It is NOT " + ACCEPTANCE_RULE["is_not"].split(".")[0] + ".")
    print()
    print("SPLITS: train n=%d (discovery), dev n=%d (accept/reject -> VALIDATION, not test)"
          % (SPLITS["train"]["n"], SPLITS["dev"]["n"]))
