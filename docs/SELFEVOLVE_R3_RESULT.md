# Self-Evolve R3 — the first eta_mu-only counterfactual

`DENOMINATOR INTEGRITY: OK` on all three arms (89/89, no missing, no extra, zero
scorer-disagreements). All three `rc=0`. Granite-4.1-8B, vector/train,
one GPU per arm, own port and own snapshot cache (all three MISS — no stale snapshot).

## The design

Held FIXED: `l = post_generation_pre_exec`, `phi = no_tool_call_at_all`, `mu = REPROMPT`,
theta (the signal is deterministic — no grid). Varied ONLY `eta_mu`, the instruction string.
Both arms run the SAME executor (`enable_zero_call_reprompt`) at the same boundary on the same
signal, so a difference between them is attributable to the action parameter and nothing else.

`se3a3` (`state_absence_if_unfound`, retry_budget=0) was never run: the executor fixes
retry_budget=1 and coercing it would run different semantics under the proposal's name.

## Raw result

| arm | eta_mu | accuracy | delta | +/- | net | firings |
|---|---|---|---|---|---|---|
| se3ctl | — (native) | 19/89 = 21.35% | — | — | — | **0** |
| se3a1 | verify_before_answering | 25/89 = 28.09% | **+6.74 pp** | +6/-0 | **+6** | 34 |
| se3a2 | search_other_container | 24/89 = 26.97% | **+5.62 pp** | +6/-1 | +5 | 28 |

The control reproduced R1's control EXACTLY (19/89) with 26 qualifying states — the same 26 —
which rules out substrate, model and scoring drift.

## THE RAW RANKING IS AN ARTEFACT. Read the stable subset instead.

The firing-parity check built into the scorer fired: **34 vs 28 firings** at a signal that is
identical in both arms. Every episode is single-turn and every firing is on turn 0, so the
multi-turn explanation is ruled out. The real cause:

**step-0 generation is not reproducible across arms.** In 10 `se3a1` cases and 4 `se3a2` cases the
step-0 decode differs from the control's — *before any injection can act*. The arm cannot cause
that; the gate runs after step-0 generation. It is sampling nondeterminism, and it silently moves
cases into and out of the qualifying population.

Decomposing each arm's 6 raw gains:

| gain class | se3a1 | se3a2 |
|---|---|---|
| **attributable** — fired AND control was a true zero-call state | **2** | **4** |
| gained with NO firing at all (pure noise) | **2** | 0 |
| fired only because drift created the qualifying state | 2 | 2 |

So 2 of se3a1's 6 gains involve no intervention whatsoever.

## The drift-free comparison

On the **78 of 89** cases where all three arms agree on the step-0 state, both arms fire **24 times
on identical states** — a genuine eta_mu-only contrast:

| | se3a1 | se3a2 |
|---|---|---|
| net on stable subset | +3 (+3/-0) | **+4 (+4/-0)** |
| on-target delta (24 control-qualifying) | +8.3 pp | **+16.7 pp** |
| on-target conversions / precision | 2 / **8.3%** | 4 / **16.7%** |
| cases broken | 0 | 0 |
| overhead (extra steps on target) | +71 | **+57** |

**se3a2 wins on every drift-free measure, reversing the raw-net ranking**, and does so at lower
cost. Neither arm breaks a working case.

## What this establishes, and what it does not

**Established.** Both eta_mu realizations of the autonomously recovered controller are positive on
a native incumbent, with zero regressions on the drift-free subset, and measurement — not the
proposer's preference — chose between them. The action PARAMETER is worth optimizing: same
(l, phi, mu), same executor, same 24 firings, and 2x the conversion rate from the instruction
string alone.

**Not established.** Significance. n=24 on-target, and a 4-vs-2 conversion difference is two cases.
The single-cell caveat applies (vector/train only; held-out is a different scenario). And the
sampling nondeterminism means any future single-run paired comparison on this substrate needs the
stable-subset treatment or a seeded/repeated design — 11 of 89 cases drifted, which is larger than
the effect being measured.

**The raw-net winner and the attributable winner are different arms.** That is the round's main
methodological result: `net` alone would have promoted se3a1 on gains that include two cases where
the arm never fired.

## Mechanism validation (4/4 converted episodes traced)

The acceptance rule wants the mechanism checked, not assumed. Every one of se3a2's four drift-free
conversions has the same shape:

```
control : step0 answer_end_turn, ZERO calls                      -> wrong
se3a2   : step0 gate=True (no calls)                             <- injection
          step1 archival_memory_retrieve(query=...)              <- a real retrieval happens
          step2 answer_end_turn                                  -> correct
```

So the causal claim "the reprompt converts a no-retrieval answer into a retrieval that finds the
fact" is CONFIRMED on 4/4, and neither arm broke a working case.

### But the instruction's literal semantics are VACUOUS, and that matters

se3a2 says *"The container you searched may not hold this fact. Search the **other** memory container
before answering."* In the qualifying population the control searched **nothing** — zero calls is the
definition of the trigger — so there is no container for "the other" to be other than. The
instruction cannot be doing what it says.

What it measurably does is **steer container choice**:

| arm | first retrieval container (all fired cases) |
|---|---|
| se3a1 | **core 33 / archival 1** |
| se3a2 | **archival 20 / core 8** |

### RETRACTED: "archival is where the facts are"

That was my first hypothesis and the data refutes it. On the drift-free target population,
container choice does not predict conversion *within* se3a2 — archival 3/18 = 16.7% and core
1/6 = 16.7%, identical. The advantage is not the container.

### What separates the arms is SEARCH DEPTH, inversely

| on the 24 drift-free target cases | se3a1 | se3a2 |
|---|---|---|
| retrievals per case | **1.96** (dist 1:4, 2:17, 3:3) | **1.25** (dist 1:18, 2:6) |
| steps per case | 3.96 | 3.38 |
| conversion | 8.3% | **16.7%** |

**The arm that searches LESS converts MORE.** se3a1's instruction ("check whether the retrieved
content actually states the fact being asked for; if not, search again with different terms")
explicitly invites a second search, and it gets one in 20 of 24 cases — while converting at half
the rate. This is the same shape as the R1/R2 finding that instructing the model to distrust an
adequate retrieval is net-negative: the extra verification pass costs more than it recovers. Here it
does not go negative, but it does dilute.

So the mechanism is: **one decisive retrieval beats a retrieve-then-doubt loop**, and eta_mu is
where that difference lives. Same (l, phi, mu), same executor, same 24 firings, 2x the conversion.
