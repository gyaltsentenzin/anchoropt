# N0-A3 acceptance criteria — FROZEN BEFORE LAUNCH

Prompt-free line, iteration N0-T3. Written before job submission.

## Incumbent

| | |
|---|---|
| incumbent | **N0 + A1 + A2** — `results/n0clean_train/a1a2/`, 109/303 = **35.97 %** |
| native baseline | 88/303 = 29.04 % (incumbent is +6.93 pp, p=0.0170) |
| control for this arm | the incumbent's own results; **not re-run** |

**A1 and A2 are unchanged.** A1 stays frozen; this is a downstream complementary policy.

## Selection, following the frozen sequence

clean incumbent → semantic re-mine → highest surviving locus → backward attribution within
that locus → policy search.

`permission/identifier/duplicate` is the highest-ranked locus surviving S1 and S2 on the clean
re-mine (`ledgers/n0t3clean_locus.json`): **35 linked failures of 194 = 18.0 %**, `settled = 0`.
Rank 1 (`capacity/…`) is **S2-STOPped** at 376/718 = 51 % settled; ranks 4–5 fail S1.

Attribution within the locus: **59 occurrences, 100 % write stage, `write_rejected`**, 56/59
self-inflicted.

## The causal chain, traced not assumed

1. Core memory fills → **A1** reroutes the write to archival under key *K*.
2. The model, unaware A1 acted, later issues its **own** `archival_memory_add(key=K, …)`.
3. The store answers **"Key name must be unique"**.

**All 59 collisions are against keys A1 itself injected** (verified against substitution
telemetry). And **54/59 = 92 %** carry a value token-identical to what A1 rescued.

> **A1 rescues the failed write. A3 suppresses the model's redundant re-write.**

## The intervention

**`post_generation_pre_execution` + suppress** — the only incision point where cancelling is
free (after execution the duplicate is already rejected and the step spent; before generation
there is no call to inspect), and **the first anchor this project has learned in that column**.

Predicate: same key, **normalised-identical value** (whitespace collapsed; nothing else).
Deliberately exact, never semantic.

| | |
|---|---|
| predicate fires | **55 / 59 = 93 %** |
| declines | 4 — three paraphrases plus one genuinely new fact (`lifestyle_changes`) |

Those 4 **stay residual by design** and will be re-mined. A fuzzy comparator would risk
suppressing real information for 7 % more coverage; that trade is refused.

Unlike every other suppress gate, the call name alone does **not** establish the call is wrong —
an `archival_memory_add` is normally correct. `match_substrings` only *selects* candidates; the
value-level predicate decides per call against **live store state**.

## Rejected alternatives, on evidence

| option | why not |
|---|---|
| `unique_kv_key` → `diet_focus_2` | readers request positional-suffixed keys **0 times** in the corpus — the write would succeed and remain unreachable by name. The A2.R failure mode. |
| `archival_memory_replace` | overwrites the value **A1 just rescued** — self-defeating against the incumbent anchor's purpose |
| semantic-suffix reroute | requires an inference-time LLM call; this line has none, and 93 % is reachable deterministically |

## Predictions, stated before the run

1. The gate fires on kv chains carrying the locus. **Zero firings would be a wiring fault, not
   a null result.**
2. Direction positive but **small**: suppressing a redundant write reclaims a step rather than
   adding information, so the gain must come from the reclaimed step being used productively.
   A plausible outcome is **+0.00 pp with no harm** — the write was redundant, so removing it
   changes nothing downstream.
3. **Harm is the real risk here, not absence of gain.** If the predicate mis-fires on a
   non-duplicate, information is lost silently. The per-backend harm clause is the binding one.

## Acceptance rule

**DO NO HARM FIRST, then gain.**

- **ACCEPT** if paired gains > losses on the 303 scored cases **and** no backend degrades by
  more than 2 cases (net) against the incumbent.
- **REJECT** if losses ≥ gains.
- **INCONCLUSIVE** only if the anchor is not demonstrably **live** — never fired, or fired
  outside the mined context. Coverage is reported as a **diagnostic, never a gate**.

Significance reported, not required.

## On acceptance

Freeze as N0-A3, make its results the incumbent and no-op counterfactual, re-mine (semantic then
attribution), and continue. **Then run the detector-expansion phase**: S5 has fired (mined
coverage 61 %, unmined 39 % > top locus 21.6 %), so use backward-trace classes
(`never_stored`, `stored_then_cleared`, `stored_but_not_retrieved`) to propose new *mineable
signals*, validate their support and consistency, and re-enter the normal sequence. Backward-trace
classes must **not** directly override the semantic ranking — they expand the miner when S5 fires.
