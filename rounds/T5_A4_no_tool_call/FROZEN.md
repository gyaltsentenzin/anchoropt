# N0-A4 acceptance criteria — FROZEN BEFORE LAUNCH

Prompt-free line. Written before job submission.

## Incumbent

| | |
|---|---|
| incumbent | **N0 + A1 + A2 + A3** — `results/n0a3_train/suppress/`, 117/303 = **38.61 %** |
| native baseline | 88/303 = 29.04 % (incumbent is **+9.57 pp**) |
| control | the incumbent's own results; **not re-run** |

A1, A2, A3 all unchanged and frozen.

## How this candidate was reached — signal discovery, then attribution, then policy

This is the **first anchor from the expanded signal vocabulary.** The error-keyed miner was
exhausted: at T4 every locus was S1-STOP **and** S2-STOP, and `phase_switch.py` returned
`EXPAND_ATTRIBUTION` (largest explained candidate 8 % of residual, below S1's 10 % floor;
unexplained 97 %).

**Signal.** `no_tool_call_at_all` — the model terminates at turn 0 having called no tool. Scored
against passing queries, not just counted: **48 failing / 12 passing, precision 0.80**. Three
plausible alternatives were rejected on the same test — `short_episode` (77 % coverage but fires
on 89 of 117 successes, precision 0.57), `answered_after_read` (0.50), `single_read_only` (0.47).
Coverage alone never proposes a detector.

**Mechanism, checked against real ground truth** (`possible_answer/BFCL_v4_memory.json`, 155
entries):

| group | n | answer was in the store | not in store |
|---|---|---|---|
| **no-call failures** | 48 | **25 (52 %)** | 23 |
| called-and-still-failed | 138 | 95 (69 %) | 43 |
| no-call successes | 12 | 9 | 3 |

**The mechanism is only half the population.** In 25 cases the answer was provably retrievable and
the model did not look — a read-side decision failure. In 23 it was absent, so not calling was a
reasonable response to an empty store and the real fault is upstream. Note the no-call failures have
the *lowest* availability rate of the three groups, which is why "the model didn't look" cannot be
asserted for all 48.

**Attribution on the 25 read-side cases:** read stage; **self-inflicted 0/25** (genuinely native,
not an anchor artifact); spread across vector 13, rec_sum 8, kv 4 and four domains. Answer location:
**archival-only 10, core-only 9, both 6** — so advice naming only core would reach at most 15.
Passing episodes make exactly **one** call (`memory_retrieve` 51, `core_memory_retrieve` 50).

**Budget.** These episodes use a median of **1 step against a ceiling of 10**, so ~9 attempts are
unused and a reprompt costs essentially nothing to try.

## The intervention

**`pre_generation` + reprompt — the only admissible cell.** No call exists yet, so `reroute` has
nothing to substitute for and `suppress` has nothing to cancel; `noop` is the control. This is the
**first anchor at the pre-generation incision point** on this line.

Policy text, in `templates.on_turn_start_action`:

    [Memory Gate] Before answering, consider whether this question depends on information
    stored earlier. If it might, consult memory first -- core memory and archival memory are
    both available. If it does not, answer directly.

**Deliberately non-directive.** It does not force retrieval and does not name a call. Three reasons,
each measured: the gap is *one* call and passing episodes find it themselves; the answer is
archival-only in 10 of 25, so naming a single container would miss them; and **23 of the 48 have
nothing in the store**, where pushing a read invites fabrication. The "if it does not, answer
directly" clause exists for exactly those.

**Two dead-code layers had to be repaired for this hook to work at all** — recorded because both
were silent:

1. `turn_needs_action_signal` existed in the ladder with a default of `False` and **no caller ever
   passed it**, so `on_turn_start_action` could never fire. The evaluator now supplies it, gated on
   `enable_turn_start_action`.
2. The soft-ladder path reads **only** the policy's template text (`template_engine.py:399`), never
   the registry fallback — so a `GateSpec` fallback is decorative for a soft signal. The text lives
   in the policy.

Verified before launch: signal ON at step 0 → **223 chars injected**; ON at step 1 → silent (ladder
restricts to step 0); OFF → silent (ships inert).

## Predictions, stated before the run

1. The gate fires on roughly the 48 no-call episodes. **Zero firings means a wiring fault, not a
   null result** — this hook was dead code twice over.
2. Direction positive but **small**, and the ceiling is ~25 cases (8 % of 303), because only half
   the exposed population has anything to retrieve.
3. **Harm is the live risk.** Firing on the 23 no-fact cases could push the model to answer from an
   empty store, and firing on genuinely memory-independent questions adds noise to 117 currently
   passing queries. The per-backend harm clause is binding, and a loss on the *passing* set is the
   specific failure to watch.

## Acceptance rule

**DO NO HARM FIRST, then gain.**

- **ACCEPT** if paired gains > losses on the 303 scored cases **and** no backend degrades by more
  than 2 cases (net) against the incumbent.
- **REJECT** if losses ≥ gains.
- **INCONCLUSIVE** only if the anchor is not demonstrably **live** — never fired, or fired outside
  the mined context. Coverage is a **diagnostic, never a gate**.

Significance reported, not required.

## On acceptance

Freeze as N0-A4, make its results the incumbent, re-mine under occurrence-level relabelling, and
re-run `phase_switch.py`. The expanded vocabulary now contains `no_tool_call_at_all`, so the next
phase check runs against it rather than the error-keyed set alone.
