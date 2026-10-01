# T5 — A4: reprompt when the model is about to answer without calling a tool

**Result: 38.61 % → 42.24 %, +3.63 pp, p = 0.0266.** Installed. Cumulative **+13.20 pp**.

The most significant single anchor on this line, and the only one whose own delta reaches p < 0.05.

| | |
|---|---|
| incumbent | N0 + A1 + A2 + A3 — 117/303 = 38.61 % |
| signal | `no_tool_call_at_all` — **from the expanded vocabulary** |
| incision point | **post-generation, pre-execution** |
| action | **reprompt** — non-directive |
| criteria | [`FROZEN.md`](FROZEN.md) (sha256 `a204313cd34eb761`), [`T5_REMINE_FROZEN.md`](T5_REMINE_FROZEN.md) |

## Why this signal

The first anchor **not** keyed on an error string. The error-keyed miner was exhausted at
[T4](../T4_exhaustion/); this detector was proposed by the expansion step and scored on precision
against *passing* queries: **48 failing / 12 passing, precision 0.80**.

## The mechanism is only half the population — and that was stated before launch

Checked against real ground truth (155 answer entries), not assumed:

| group | n | answer was in the store | not in store |
|---|---|---|---|
| **no-call failures** | 48 | **25 (52 %)** | 23 |
| called-and-still-failed | 138 | 95 (69 %) | 43 |
| no-call successes | 12 | 9 | 3 |

In 25 cases the answer was **provably retrievable and the model did not look** — a read-side decision
failure. In 23 it was **absent**, so not calling was a reasonable response to an empty store and the
real fault is upstream. Note the no-call failures have the *lowest* availability rate of the three
groups, which is why "the model didn't look" cannot be asserted for all 48.

Attribution on the 25: read stage; **self-inflicted 0/25** — genuinely native, not an anchor artifact,
unlike A2 and A3. Answer location: archival-only 10, core-only 9, both 6 — so advice naming only core
would reach at most 15.

Budget: these episodes use a median of **1 step against a ceiling of 10**, so ~9 attempts are unused and
a reprompt costs essentially nothing to try.

## Why the policy text is deliberately non-directive

```
[Memory Gate] Before answering, consider whether this question depends on information
stored earlier. If it might, consult memory first -- core memory and archival memory are
both available. If it does not, answer directly.
```

It does not force retrieval and does not name a call. Three reasons, each measured:

- the gap is *one* call, and passing episodes find it themselves (they make exactly one:
  `memory_retrieve` 51, `core_memory_retrieve` 50);
- the answer is archival-only in 10 of 25, so naming a single container would miss them;
- **23 of the 48 have nothing in the store**, where pushing a read invites fabrication. The
  "if it does not, answer directly" clause exists for exactly those.

## The case study: v1 vs v2 — same everything, opposite sign

**This is the single clearest result in the project, and it is about the incision point.**

|  | v1 | v2 |
|---|---|---|
| incision point | **pre-generation** | **post-generation, pre-execution** |
| trigger | `step_count == 0` | query answered with **zero tool calls** |
| firings | **303/303** (6× over-fire) | **59/303** (60 no-call episodes exist) |
| Δ train | **−4.95 pp** | **+3.63 pp** |
| p | 0.1633 | 0.0266 |
| rec_sum net | −24 | +3 |

**Identical signal. Identical action family. Byte-identical injected text.** The only difference is
where it fires.

v1 could not observe "about to answer without calling a tool" — **that fact does not exist before
generation** — so it fired on every episode and paid the cost on the 244 that did not need it. v2 waits
until the answer exists and the call count is known.

No-call answers went **60 → 0 in both versions**, so the mechanism was never in doubt. Only the
targeting was.

Two notes on how this is recorded honestly:
- **v1 → v2 is one specification change**, not a tuning pass. Same signal, same action family, same
  text; a single field differs.
- The correction came from the **user's specification**, not from post-hoc analysis of v1's loss. The
  standing rule is *never fix a losing arm after seeing it lose* — a fixed version is a **new candidate
  measured fresh**, which is what v2 was.

## Two dead-code layers had to be repaired first

Recorded because both were **silent**, and either would have produced a null result indistinguishable
from "the remedy doesn't help":

1. `turn_needs_action_signal` existed in the ladder with a default of `False` and **no caller ever
   passed it** — so the hook could never fire.
2. The soft-ladder path reads **only** the policy's template text, never the registry fallback — so a
   fallback specification is decorative for a soft signal. The text has to live in the policy.

Verified before launch: signal ON at step 0 → 223 chars injected; ON at step 1 → silent (the ladder
restricts to step 0); OFF → silent, ships inert. **Behavioural verification of the live path, not source
inspection** — source scans have passed while the behaviour was broken.

## Held out

| | |
|---|---|
| held-out (n = 84) | A3 22.62 % → A4 **26.19 %**, +3.57 pp, 7 gains / 4 losses, p = 0.5488 |

The point estimate transfers almost exactly (+3.63 train → +3.57 held-out) and no backend degrades on
either split, but **significance is not established** at n = 84. Directional consistency was the
pre-stated bar.

Mechanism check on the gains is deliberately partial: the gate fired on only 10/16 train gains and 3/7
held-out gains, so **a large part of the effect is an indirect prereq-store change, not a query-time
intercept.** Directly attributable: 10g/3l train, 3g/2l test. Per-backend attribution does **not**
replicate across splits (train losses all rec_sum, held-out losses all kv), so it is reported as
aggregate only.

## What it changed for the next round

The T5 re-mine (`T5_REMINE_FROZEN.md`): residual 186 → 175, explained coverage **13 % → 3 %**, and
`no_tool_call_at_all` is **absent** — A4 consumed it. No locus survives S1/S2; S5 dominance fires again,
so the phase switch returns `EXPAND_ATTRIBUTION` a second time.

Signal expansion continues from here, and the second expansion is what produced the **write-side**
anchors: A5 ([T6](../T6_A5_archival_full/)), A7 ([T7](../T7_A7_blob_overflow/)) and A8
([T8](../T8_A8_dedup_clear/)). A6 was attempted at that locus and is **deferred**
([`../A6_deferred/`](../A6_deferred/)).

One deferral recorded rather than acted on: relaxing the support thresholds for high-precision,
low-support fixes is worth considering — rank 2 is exactly that shape — but **not mid-line**, or T1–T5
stop being comparable to each other.
