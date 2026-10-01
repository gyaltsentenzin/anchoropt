# T9 / A9 — acceptance criteria, frozen before the arm ran

Frozen 2026-08-26, before implementation of v3. `ACCEPTANCE_RULE_FROZEN.md` v2, **accuracy** class.

This is a **void-and-retry of A9-R**, not a new mechanism. v1/v2 were computationally inert (see §Gate 0
and the round README), so their +0.00 pp was **void, not a rejection**, and the criteria below are the
ones the original arm was written against — unchanged.

---

## Gate 0 — evidence validity, evaluated BEFORE criteria 1–4

Added for this round and now general. If Gate 0 fails, the accuracy numbers are not evidence of anything
and the round does not proceed to the rule.

| # | condition | required |
|---|---|---|
| 0a | every **attested** firing delivered into the model-facing message list | 100% |
| 0b | delivery violations | 0 |
| 0c | positional replacement, no append fallback | 0 appended |
| 0d | trigger unchanged from the frozen spec (`core_max` < 0.30 on every firing) | all |
| 0e | arm and control on **distinct `base_key`s** (no cache cross-contamination) | required |
| 0f | unattested firings confined to the **prereq phase** | required |

**0f is the honest half.** Prompts are not captured on the prereq path, so delivery there cannot be
attested. Those firings must be reported as **not measured** — *not* as failures, and *not* as passes.
Conflating "not measured" with "not delivered" is the original A9 error in mirror image.

## The rule — all four must hold

| # | criterion | threshold |
|---|---|---|
| 1 | **train** | net > 0 vs the frozen incumbent, on paired cases |
| 2 | **dev** | net ≥ 0 vs the same incumbent |
| 3 | **attribution** | the gain is concentrated where the anchor engages, and the mechanism is causally validated from its own decision-point telemetry |
| 4 | **safety** | no catastrophic or prohibited failure mode introduced |

### Criterion 3 is defined per case, before any number exists

A gain counts toward the mechanism only if **all three** hold for that case:

1. A9 fired **and** the merged result was **attested delivered**
2. archival entries were **promoted** into the returned top-k
3. the case **fails** in the paired control

Aggregate agreement is not sufficient. A count of gains that merely correlates with firings does not
establish the mechanism.

### Non-destructiveness, measured not asserted

The merge is additive by construction — the union is ranked, so a core entry that still earns a top-k
place keeps it, and ties break toward core. This must be **recorded per firing**
(`core_entries_retained` against `core_topk_before`, with dropped ids listed), not argued from the design.

## Scope, declared in advance

**Vector only.** The trigger needs per-entry similarity scores; kv and rec_sum do not expose them, and
the mechanism returns `None` for those backends. So:

- the measured claim is a **vector-shard** delta;
- any whole-corpus figure is **arithmetic on top of it** and must be labelled a **projection** until a
  three-shard run confirms kv and rec_sum are unchanged.

Pre-registered because the units are a trap: read as a corpus number, a vector-shard accuracy overstates
this anchor by roughly 3×.

## Declared limitations, before the result

1. Whole-corpus figures will be projections until three-shard confirmation.
2. Prereq-phase delivery will be unverified.
3. Losses are permitted by the rule and must be **reported beside the net**, with firing/promotion
   checked so displacement is distinguished from gate failure.

## Protocol

Sharded by backend, `--store-workers 1`, distinct ports, arm and control on separate `base_key`s, so the
result is byte-reproducible. A6 (`ANCHOROPT_ELEN`) must be **off** in both arms — it is deferred, and
carrying a shard template's export would put a rejected anchor in both. A7 must be **on** in both: it is
part of the frozen incumbent.

Arm and control share the snapshot cache **deliberately**: A9 touches the query phase only, so the prereq
store is identical by construction and prereq trajectories are required to be byte-identical. (The
inverse of DCR, where the arm changed prereq writes and the `base_key` had to move.)

---

Result and the full delivery-defect record: [`README.md`](README.md). General rule adopted from this
round: [`../../docs/CONSUMER_BOUNDARY_RULE.md`](../../docs/CONSUMER_BOUNDARY_RULE.md).
