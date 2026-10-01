# T3 — A3: suppress a redundant re-write before it executes

**Result: 35.97 % → 38.61 %, +2.64 pp.** Installed. Cumulative **+9.57 pp** over the native baseline.

| | |
|---|---|
| incumbent | N0 + A1 + A2 — 109/303 = 35.97 % (+6.93 pp over native, p = 0.0170) |
| signal | `permission/identifier/duplicate` |
| incision point | **post-generation, pre-execution** — the first anchor in this column |
| action | **suppress** |
| criteria | [`FROZEN.md`](FROZEN.md), written before launch |

## Why this signal

Highest-ranked locus surviving both stopping screens on the clean re-mine: **35 linked failures of
194 = 18.0 %**, `settled = 0`. What the screens removed matters as much as what they kept:

- rank 1 (`capacity/…`) is **S2-STOP** — 376/718 = 51 % already settled by A1. Continuing there is a
  coverage question for an existing anchor, not a new anchor.
- ranks 4–5 fail **S1** — below the 10 % coverage floor, so no arm could measure them.

Attribution within the locus: **59 occurrences, 100 % write stage, `write_rejected`**, 56/59
self-inflicted.

## The lesson of this round: look for self-inflicted residuals

The causal chain was traced, not assumed:

1. Core memory fills → **A1** reroutes the write to archival under key *K*.
2. The model, **unaware A1 acted**, later issues its own `archival_memory_add(key=K, …)`.
3. The store answers *"Key name must be unique."*

**All 59 collisions are against keys A1 itself injected** (verified against substitution telemetry), and
**54/59 = 92 %** carry a value token-identical to what A1 rescued.

> A1 rescues the failed write. A3 suppresses the model's redundant re-write.

An anchor auditing its predecessor's side effects. This is the clearest single argument for re-mining
after every install: **this locus did not exist before A1**, and no ranking computed at T1 could have
contained it.

## Why post-generation / pre-execution + suppress

The only incision point where cancelling is **free**:

- **post-execution** — the duplicate is already rejected and the step already spent. Nothing to save.
- **pre-generation** — there is no call to inspect yet.
- **post-gen / pre-exec** — the call is visible, the world has not changed. Cancel it.

This is the first anchor this project learned in that column, and it is a **commitment gate** rather
than error recovery — the two kinds follow from the incision point, not from the signal.

### The predicate is deliberately exact, never semantic

Same key, **normalised-identical value** (whitespace collapsed; nothing else).

| | |
|---|---|
| predicate fires | **55 / 59 = 93 %** |
| declines | 4 — three paraphrases plus one genuinely new fact (`lifestyle_changes`) |

Those 4 **stay residual by design**. A fuzzy comparator would buy 7 % more coverage at the risk of
silently suppressing real information; that trade is refused. When the failure mode of being wrong is
*data loss*, precision beats coverage.

Unlike every other suppress gate, **the call name alone does not establish the call is wrong** — an
`archival_memory_add` is normally correct. So substring matching only *selects* candidates; a
value-level predicate decides per call against **live store state**.

## Alternatives rejected on evidence

| option | why not |
|---|---|
| `unique_kv_key` → `diet_focus_2` | readers request positional-suffixed keys **0 times** in the corpus — the write would succeed and remain unreachable by name. That is exactly the A2.R failure mode. |
| `archival_memory_replace` | overwrites the value **A1 just rescued** — self-defeating against the incumbent anchor's purpose |
| semantic-suffix reroute | requires an inference-time LLM call; this line has none, and 93 % is reachable deterministically |

## What was predicted

1. Fires on kv chains carrying the locus. **Zero firings would be a wiring fault, not a null result.**
2. Direction positive but **small** — suppressing a redundant write reclaims a *step*, it does not add
   information. The frozen doc explicitly allowed **+0.00 pp with no harm** as a plausible outcome: the
   write was redundant, so removing it might change nothing downstream.
3. **Harm is the real risk here, not absence of gain.** If the predicate mis-fires on a non-duplicate,
   information is lost silently. The per-backend harm clause is the binding one.

Outcome: **+2.64 pp** — the reclaimed step was used productively more often than not.

## What it changed for the next round

This was the last anchor the **error-keyed** vocabulary could produce. At the next re-mine every locus
was simultaneously S1-STOP and S2-STOP, and the loop had to decide whether that meant *done* or *blind*.

See **[T4 — exhaustion](../T4_exhaustion/)**.
