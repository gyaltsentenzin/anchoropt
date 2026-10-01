# R3 replication — analysis FROZEN before the run

Written and committed **before** the seed-2 arms were submitted. Nothing below may be revised on the
basis of the seed-2 numbers; if the criterion turns out to be wrong, that is a finding to report,
not a reason to re-cut the analysis.

## What is being tested

Not "does se3a2 help". Whether **AnchorPolicyOpt's SELECTION among eta_mu candidates is stable**:
does the ordering `se3a2 > se3a1 > control` replicate on causally comparable states under a new
seed? That requires all three arms, since repeating only the winner tests the winner and not the
optimizer.

## Held identical (verified by checksum against the seed-42 runners)

attribution · (l, phi) · executor (`enable_zero_call_reprompt`) · both policy texts byte-for-byte ·
retry budgets (=1) · cases/split (vector/train, 89 scored) · model
(`granite-4.1-8b`, `MAX_MODEL_LEN=32768`) · acceptance rule · scorer.

**Changed: `--seed 42` -> `--seed 2`, and nothing else.** New ports and new per-arm snapshot caches,
because a cache keyed on the old seed's decode would serve stale prereq state.

## The eligibility criterion, and its measured limit

`anchoropt/learning/prestate_eligibility.py`, 12 tests.

The stated rule is: a case is comparable only if arm and control share the trajectory up to the
point the intervention could first fire. Implemented with a **shared** cut (the earliest firing index
across arms) — truncating each arm at its own index compares a non-firing control's whole trajectory
against a treated arm's prefix and turns the treatment itself into apparent drift. A test pins that.

**Measured limit, recorded before the run.** On the seed-42 data `common_fire_index` is **0 in all 35
cases where any arm fires** and None in the other 54. The compared prefix is therefore EMPTY for
every treated case: `eligible_cases` returns 85/89 having checked nothing where it matters. For a
signal evaluated on the model's first decision, common-prestate eligibility **passes vacuously and
cannot certify comparability**.

So the **primary criterion is the target-population split** (`engagement_split`), defined from the
CONTROL alone:

| cell | meaning | used for |
|---|---|---|
| `on_target_shared` | control was a true zero-call state AND the arm fired | **the causal comparison** |
| `opportunity_created` | arm fired, control did NOT qualify | sampling created the state — **excluded from attribution** |
| `opportunity_missed` | control qualified, arm did not fire | engagement failure — **kept in the denominator** |

An arm may not take credit for a state the control never presented, and may not be rewarded for
skipping cases.

On seed 42 this gives: both arms `on_target_shared = 25`; `opportunity_created` se3a1 **9** vs se3a2
**3**; `opportunity_missed` 1 each.

## Reported quantities (all eight, in this order)

1. denominator integrity (precondition — no deltas printed if it fails);
2. raw accuracy / net;
3. pre-intervention drift count (`opportunity_created` + prestate drift);
4. common-prestate / on-target sample size;
5. firing parity on that subset;
6. attributable gains/losses (`on_target_shared` only);
7. on-target conversion rate;
8. intervention and step overhead.

## Promotion rule (frozen)

Promote se3a2 **only if all four hold** on seed 2:

1. se3a2 positive on the on-target subset;
2. no material breakage of working cases;
3. mechanism unchanged: zero-call commitment → injected reprompt → **a real retrieval** → corrected
   answer;
4. se3a2 at least as good as se3a1 on this frozen criterion.

Then immediately: rerun the promoted incumbent → re-attribute → re-mine/re-rank → compare the new
residual distribution against the original 4/3/1.

**If the ranking flips or the effect vanishes: STOP and report instability.** No prompt tuning, no
other R1 locus.

## Interpretation to preserve

The validated mechanism is **narrow**: *intercept a zero-tool-call commitment and induce at least one
retrieval before answering.* "Search the other container" is **NOT** the validated mechanism — that
instruction's literal semantics are vacuous here, since the trigger is that nothing was searched.

The depth asymmetry (se3a1 1.96 reads/case → 8.3% conversion; se3a2 1.25 → 16.7%) is a **HYPOTHESIS**
for this replication to test — that se3a2 induces a shorter, less distrustful recovery trajectory —
not a conclusion.
