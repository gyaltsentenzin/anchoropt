# N0-A2.R acceptance criteria — FROZEN BEFORE LAUNCH

Prompt-free line, iteration N0-T2. Written before job submission; not editable after results land.

## Incumbent

| | |
|---|---|
| incumbent | **N0 + A1** — `results/n0a1v2_train/anchor/`, 102/303 = **33.66 %** |
| A1 held-out | 16/84 = 19.05 % (+2.38 pp, p=0.75) — directionally consistent, weak |
| control for this arm | the incumbent's own results; **not re-run** |

No global preamble anywhere: `on_memory_preamble` is empty in every template of this line.

## How this candidate was selected — two mining stages

**Semantic miner** (artifact `ledgers/n0t2_locus.json`, fp `1f3b951ec89436d5`, haiku-4-5, prompt v2)
ranked `existence/identifier/not_found` **rank 1** on the post-A1 residual: 51 events, **47 of 201
failing queries linked** — nearly 1 : 1 loss per occurrence, the tightest coupling of any locus.

**Attribution miner** then split that one error string by *first consequential decision*:

| first consequential decision | occ | anchor |
|---|---|---|
| value **was** archived; the read never consulted archive | **39** | **A2.R** — this arm |
| core filled and the value was **not** archived | 5 | A2.W — **deferred**, see below |
| no write-side explanation | 8 | `unattributed`, mostly `vector-customer` |

One error string, two causes, two anchors. The dominant one is read-side, and **all 39 are
self-inflicted**: the value sits where A1 put it.

**A2.W is retained in the ledger and deferred** — 5 occurrences is below any support floor, and its
mechanism is A1's own: A1 already rescues ~97 % of core-full events (finance 33/33, notetaker 28/28,
student 145/146). The residual leak concentrates in `vector-customer` (125/133) and
`vector-healthcare` (37/42), which is a coverage question for A1, not a new anchor.

## The intervention

Nominal grid **2 admissible points × 4 actions = 8 cells → 3 live**; the rest rule out structurally
(`suppress` has nothing to cancel pre-generation and cannot undo a completed read; `reroute` needs a
call to substitute for; `noop` is point-independent). Two arms are run:

| arm | cell | mechanism |
|---|---|---|
| **reroute** | post_execution + reroute | `on_key_not_found_rerouted` → `archival_memory_key_search(query=<key>)` |
| **reprompt** | post_execution + reprompt | `on_domain_error_key_not_found` — advise, do not synthesise |

`pre_generation + reprompt` is live but **not run**: it is a prose nudge on every turn, which this
line excludes by construction.

**The destination was changed on attestation, not assumption.** The registry declared
`archival_memory_retrieve`, which demands an **exact** key match — but A1's write-side synthesizer
sanitises and dedupes keys before archiving, so an exact retrieve misses on precisely the cases A1
created. Measured on the post-A1 residual: `archival_memory_retrieve` was tried **once**;
`archival_memory_key_search` (fuzzy BM25) resolved **11/11**. `synthesize_read_reroute_call` also had
to learn per-destination argument names — it emitted `key=` universally, which would have built an
invalid call for `key_search` and been misscored as "the substitute did not help".

**Exposure is fully buildable:** all 44 `Key not found` triggers carry `key=`, so the synthesizer
returns a call for every one (0 refusals).

## Predictions, stated before the run

1. The reroute arm fires on the kv chains carrying the locus (student, finance, healthcare,
   notetaker). Zero firings would mean a wiring fault, not a null result.
2. Direction is positive but the effect is **small**: 39 occurrences against a 201-case residual, and
   `archival_memory_key_search` returns *ranked keys*, not a value — so the model must still issue a
   follow-up retrieve. This arm removes a dead end; it does not hand over the answer.
3. reroute ≥ reprompt. The 2/3 prior for substitution over advice has held on every locus measured so
   far, and here the substitute's arguments are fully attested.

## Acceptance rule

**DO NO HARM FIRST, then gain.** Harm is checked before gain.

- **ACCEPT** the better arm if paired gains > losses on the 303 scored cases **and** no backend
  degrades beyond harm tolerance (net worse by more than 2 cases) against the incumbent.
- **REJECT** if losses ≥ gains for both arms.
- **INCONCLUSIVE** only if the anchor is not demonstrably **live** — it never fired, or fired outside
  the mined context. **Coverage is a diagnostic, never a gate**: the same anchor reads 100 % or 9 %
  depending on the denominator chosen, so any threshold would be a choice of denominator dressed as a
  criterion.

Significance is reported, not required — this corpus cannot reach p<0.05 on a ~5 pp effect.
Directional consistency is the bar.

## On acceptance

Freeze as **N0-A2**, make its results the incumbent and the no-op counterfactual, re-mine (semantic
then attribution), and continue down the ranked list. Ranks 2–5 of the current N0-T2 ordering become
stale on acceptance and must be recomputed.
