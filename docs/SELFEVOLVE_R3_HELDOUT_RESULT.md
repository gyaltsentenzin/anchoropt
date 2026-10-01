# R3 held-out validation — genuinely independent, and the arms TIE

Nine arms (3 test-only cells × 3 arms), all `rc=0`, `DENOMINATOR INTEGRITY: OK` on all nine, frozen
criterion and frozen η texts (policy md5s unchanged). No cell is shared with train.

## The headline

**The signal exists on held-out and the mechanism holds perfectly — but se3a2 and se3a1 are exactly
tied on the two uncontaminated cells.** The 2× conversion advantage that selected se3a2 on train does
not reproduce.

## Trigger availability: answered

Prior test-split "controls" were all the A1–A5+A7 stacked incumbent, in which A4 is already installed
(`enable_zero_call_reprompt=true`, 251-char instruction) — so their 0% zero-call rate was **the anchor
consuming its own trigger**, not the signal being absent. The first native held-out controls show:

| cell | n | control acc | target states | firings (control) |
|---|---|---|---|---|
| vector-student | 40 | 5/40 = 12.5% | **7** | 0 |
| kv-customer | 24 | 0/24 = 0.0% | **7** | 0 |
| rec_sum-finance | 20 | 9/20 = 45.0% | **5** | 0 |

19 target states across 84 queries (22.6%), close to train's 26/89 (29.2%).

## ⚠️ kv is CONFOUNDED and excluded

The gate fires **during prereq episodes** in kv — 22 times for se3a1, 12 for se3a2, **0 for the
control** — and the kv prereq call sequences differ across all three arms. Prereqs write the store the
queries read, so the kv arms were scored against **stores the arm itself had modified**. vector and
rec_sum prereq trajectories are byte-identical across arms.

This is what produced the one non-attributable gain: `memory_kv_23-customer-23` flipped to correct in
se3a2 on a case where **the arm never fired**, with all three arms issuing different step-0 calls.
That is the store differing, not the controller working.

A `[0b] PREREQ CONTAMINATION` check is now in the scorer, so this is detected rather than discovered.

## Results on the two clean cells

| | control | se3a1 | se3a2 |
|---|---|---|---|
| accuracy (pooled, n=60) | 14/60 = 23.33% | 16/60 = **26.67%** | 16/60 = **26.67%** |
| delta | — | **+3.33 pp** | **+3.33 pp** |
| gains / losses | — | +2 / **−0** | +2 / **−0** |
| firings | 0 | 12 | 12 |
| target coverage | — | **12/12 = 100%** | **12/12 = 100%** |
| off-target firings | — | **0** | **0** |
| on-target conversions | — | 2/12 = 16.7% | 2/12 = 16.7% |
| mechanism | — | **12/12** | **12/12** |

Per cell: vector +5.00 pp for both arms (2 conversions each of 7 firings, 28.6%); rec_sum **+0.00 pp**
for both (0 of 5 conversions, 0 breakage).

### Cost (real decisions only; gate bookkeeping rows excluded)

| cell | arm | steps | tool calls | retrievals | cost per fix |
|---|---|---|---|---|---|
| vector | se3a1 | 91 vs 77 (+14) | +14 | +14 | +7.00 steps |
| vector | se3a2 | 84 vs 77 (**+7**) | +7 | +7 | **+3.50 steps** |
| rec_sum | se3a1 | 40 vs 35 (+5) | +5 | +5 | undefined (0 fixes) |
| rec_sum | se3a2 | 40 vs 35 (+5) | +5 | +5 | undefined (0 fixes) |

**se3a2 remains cheaper where it converts** — half the extra steps per fix on vector — which is the
one place the train-time depth asymmetry (1.25 vs 1.96 reads/case) still shows.

## rec_sum: harmless but useless

5 firings, 100% on-target, every one producing a real `memory_retrieve()`, **0 conversions, 0
breakage**. 4 of the 5 fired on cases the control already answered correctly. The controller engages
the right states and changes nothing — pooling would hide this, which is why cells are reported first.

## Verdict on the frozen criterion

All four criteria PASS on the clean cells: se3a2 is positive (+3.33 pp), breaks nothing (0 losses,
0 on-target breakage), mechanism intact (12/12), and is ≥ se3a1 (tied on net and conversions, cheaper
per fix).

**So the rule says promote.** What the rule cannot say is that se3a2 is *better* than se3a1: on
independent data they are indistinguishable on accuracy, and se3a2's advantage is now cost, not
conversion. The train-time 4-vs-2 conversion gap was 2 cases and did not replicate.

## What this establishes

**Established.** The recovered controller generalizes across scenario: 100% target coverage, 0
off-target firings, mechanism 12/12, 0 regressions across 60 independent queries, +3.33 pp. The
narrow mechanism — *intercept a zero-tool-call commitment and induce at least one retrieval before
answering* — holds exactly as stated on data it never saw.

**Not established.** That eta_mu SELECTION is reliable. Both candidates perform identically on
held-out accuracy, so this round validates the controller, not the optimizer's choice between
realizations. Significance is also not established and was not attainable: 2 net gains on n=60, where
a sign test needs ≥5.
