# N0-T5 — re-mine of the A1+A2+A3+A4 incumbent. FROZEN.

Artifact `ledgers/n0t5_locus.json`, fingerprint **`869316d8730866c4`**,
canonicalizer `anthropic/aws/claude-haiku-4-5`, `PROMPT_VERSION = v2`.

Frozen **before** any A5 candidate is specified, per the standing rule: *the LLM canonicalization is
frozen per iteration before ranking, and occurrence-level relabelling runs before selecting the next
anchor.*

## Incumbent

    results/n0a4v2_train/zerocall      A1 + A2 + A3 + A4      128/303 = 42.24 %

## Residual, after occurrence-level relabelling

| | T4 | **T5** |
|---|---|---|
| residual failing queries | 186 | **175** |
| corrections by the incumbent | 571 across 108 ep | **499 across 75 ep** |
| explained by the current vocabulary | 13 % | **3 %** |
| unexplained | 87 % | **97 %** |

Ranking is on **uncovered residual under the current incumbent** — `settled` occurrences (those an
incumbent anchor already acted on) are excluded from `support`, and `maxdep` is carried as severity
only, never added to support.

| rank | semantic locus | support | linked | settled | raw | maxdep | backends |
|---|---|---|---|---|---|---|---|
| 1 | `capacity/container/no_remaining_capacity` | 39 | 17 | 452 | 789 | 20 | kv 10, rec_sum 277, vector 50 |
| 2 | `size/item/exceeds_per_item_limit` | 4 | 10 | 47 | 71 | 21 | vector 24 |
| 3 | `existence/identifier/not_found` | 6 | 1 | 46 | 59 | 3 | kv 3, rec_sum 1, vector 9 |
| 4 | `format/field/duplicate` | 0 | 0 | 28 | 28 | 0 | — |
| 5 | `format/field/malformed_value` | 0 | 0 | 1 | 1 | 0 | — |

`no_tool_call_at_all` does **not** appear: A4 consumed it. The expanded vocabulary is in scope for
this check, so its absence is a result, not an omission.

## Phase verdict

    PHASE = EXPAND_ATTRIBUTION

| locus | S1 coverage (≥ 10 % of residual) | S2 saturation (settled/raw < 0.50) |
|---|---|---|
| `capacity/container/no_remaining_capacity` | 9.7 % — **STOP** | 57 % — **STOP** |
| `size/item/exceeds_per_item_limit` | 5.7 % — **STOP** | 66 % — **STOP** |

- no explained locus survives the policy gates
- largest explained candidate: **17 linked = 10 % of residual**
- unexplained residual: **169 = 97 %** → **S5 dominance fires**

**The signal representation is exhausted, not the headroom.** This is the same verdict that produced
A4, one level down: rank 1 is not unexploited headroom (the incumbent has already acted on 452 of its
occurrences) but the hard tail. Forcing a fifth anchor onto rank 1 is exactly what the stopping
criteria exist to prevent.

**Next admissible move:** attribution expansion over the 169 unexplained failures
(`scripts/expand_attribution.py`), not a new policy on an existing locus.

## Deferred, recorded — do NOT apply to this line

Support thresholds should relax as iterations proceed, so high-precision / low-support fixes become
admissible once high-support loci are exhausted. **Rank 2 is exactly that shape**: support 4, but 10
linked and `maxdep` 21 — a fixed floor rejects it, a precision-weighted floor would admit it.

Not changing the thresholds now. Doing so mid-line would make T1–T5 non-comparable, and the T5
verdict above is only interpretable against the same gates that produced T1–T4.

## Reproduce

    python scripts/remine_incumbent.py --incumbent results/n0a4v2_train/zerocall \
        --artifact ledgers/n0t5_locus.json --iteration N0-T5 --json /tmp/n0t5_rank.json
    python scripts/phase_switch.py --incumbent results/n0a4v2_train/zerocall \
        --artifact ledgers/n0t5_locus.json --json /tmp/phase5.json

`LocusArtifact.freeze()` refuses to overwrite, so re-running against the same path is safe: it
verifies rather than silently re-canonicalizes.
