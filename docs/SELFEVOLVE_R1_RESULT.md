# Self-Evolve R1: the first real paired Granite evaluation

Granite-4.1-8B · vector/train · **89 scored queries** · all `rc=0`

## Denominator integrity — checked before anything was interpreted

| arm | n | missing | extra | scorer disagreements | verdict |
|---|---|---|---|---|---|
| `se1ctl` | 89 | 0 | 0 | 0 | OK |
| `se1c2` | 89 | 0 | 0 | 0 | OK |
| `se1c3` | 89 | 0 | 0 | 0 | OK |

**DENOMINATOR INTEGRITY: OK** across two independent scorer paths. Only then were deltas computed.

(The shard file holds 116 rows; 89 are scored queries and the rest prereq setup. 89 is the
denominator, and both scorer paths agree on it.)

## Result: both candidates REJECTED, nothing promoted

| arm | install | accuracy | delta | gains/losses | net | firings | executed |
|---|---|---|---|---|---|---|---|
| `se1ctl` | control, gates off | 19/89 = 21.35% | — | — | — | — | — |
| `se1c2` | **faithful** (θ=0.75) | 16/89 = 17.98% | **−3.37 pp** | +1 / −4 | **−3** | 46 (52%) | 46 |
| `se1c3` | **approximate** | 19/89 = 21.35% | **+0.00 pp** | +0 / −0 | **0** | **0** | 0 |

AnchorOpt's acceptance rule rejected both: `net -3 <= 0` and `net 0 <= 0`. The proposer's own
ranking put candidate 1 first and candidate 2 second; measurement was allowed to disagree, and did.

## Candidate 2 — the faithful test, and why it lost

The intervention **engaged heavily and hurt slightly**, which is the informative case: this is not a
null result and not a plumbing failure.

| | n | control pass | candidate pass | net | gains/losses |
|---|---|---|---|---|---|
| gate FIRED | 46 | 11 | 9 | **−2** | +1 / −3 |
| gate did not fire | 43 | 8 | 7 | −1 | +0 / −1 |

* **Attribution holds.** 4 of the 5 changed cases fired the gate. The harm is concentrated where the
  controller engages, so the −3.37 pp is attributable to it rather than to noise.
* **The threshold was too loose.** The proposer chose `below: 0.75` from a single observed score of
  0.4302. At 0.75 the gate fires on **46 of 89 cases (52%)**, and **11 of those 46 were already
  passing** — it interrupts a working retrieval to demand re-verification. A5's frozen default is
  0.30; the autonomously chosen value was 2.5× that.
* **The one gain is real**: `memory_vector_1-customer-1`, the single case the attributor identified as
  *never consulting memory at all*, went 1 step → 4 steps and passed.
* **The `not fired` row is not zero (−1).** A reprompt changes the trajectory, and one non-firing case
  regressed anyway. Worth remembering before treating a non-firing subset as a clean control.

## Candidate 3 — the approximate test: zero firings

0 firings, byte-identical accuracy to control. The proposal's φ was **per-call vacuity**; the
installed trigger was a **failed-search streak ≥ N**, and on this shard that streak never occurs. So
this arm tested the approximation and the approximation is inert — it says nothing about the proposed
controller. Under `docs/ACCEPTANCE_RULE.md` a 0 pp delta with 0 firings is proof of **non-engagement**,
not of a subtle effect.

## Candidate 1 — never run, and that is a finding

Proposed as a pre-execution commitment-gate reprompt. Not installable: the evaluator's nearest gate
(`on_premature_idk`) is `TEACHER_FROZEN` *and* injects `trailing_user`, i.e. post-execution. Forcing
it would have moved ℓ and recreated the A4 v1/v2 confound (byte-identical text at the wrong boundary,
−4.95 pp). Recorded as **a controller outside the realizable host space U_H(ℓ)** — a fact about the
runtime, not a proposer failure. It was also the proposer's own first choice, covering all 8 cases.

## What this answers, and what it does not

**Answered.** The loop runs end to end on real infrastructure: real residual → autonomous attribution
→ autonomous structured proposal → legality pruning → **real paired Granite measurement** → acceptance
decision. No human edited a diagnosis, signal, threshold, prompt or action. No A1–A9 fixture was
imported. Denominator integrity passed. Measurement, not the proposer's ranking, decided.

**Not answered.** Whether the loop can produce a *measured improvement*. It did not, this round. One
faithful candidate engaged and cost 3 cases; one approximate candidate was inert; one was infeasible.
There is no promotion and therefore **no re-mine** — the incumbent is unchanged, so the
residual-guided property remains untested live.

## One telemetry defect found while scoring

The first scoring pass read `exec.jsonl` for `*_gate` keys and reported **0 firings** beside a
per-case count of 46. `exec.jsonl` is a tool-call log and carries no gate keys; the firings live in
the per-case `gates_fired` dict. A zero-firing arm and an unparsed telemetry channel look identical
in a summary — and had that stood, candidate 2's −3.37 pp would have been called non-attributable
when it is in fact well attributed. Fixed in `scripts/score_selfevolve_r1.py`.
