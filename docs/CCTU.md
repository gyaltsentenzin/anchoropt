# CCTU × AnchorOpt — start here

AnchorOpt applied to the **CCTU** benchmark on `Qwen/Qwen3.6-35B-A3B`. Five rounds, thirteen arms, **one
deployed anchor**. This page is the index; each document below is self-contained.

## In one paragraph

CCTU gives an agent a task with explicit constraints (response length, call budgets, formatting) and a
validator that reports violations turn by turn. AnchorOpt learns *anchors* — small controllers that watch the
agent's own state at a declared boundary and intervene when a condition holds. The deployed anchor watches for
a drafted final answer that exceeds the task's published length limit, at the moment the agent commits to it,
and issues one of two instructions depending on whether the agent has called a tool yet. It removes **11.6% of
all violations on train and 8.8% on the held-out split**, at a cost of **two correct answers on train and none
on test**. It does **not** move CCTU's headline `SR`/`PSR` metrics at all — nor did any of the thirteen arms.

## Reading order

| # | document | what it answers |
|---|---|---|
| 1 | `CCTU_RESULTS_QWEN.md` (not in this branch, see [`docs/RESULTS_POLICY.md`](../docs/RESULTS_POLICY.md)) | **What was measured.** Baseline, noise floor, the residual by class, every round's verdict, the deployed anchor against the one it replaced. The durable record — every number has a command. |
| 2 | [`CCTU_ANCHORS.md`](CCTU_ANCHORS.md) | **What the anchor is.** Both branches, why the split, the runtime configuration that is part of the policy, how to run it, what to check before porting it. |
| 3 | [`CCTU_REPRODUCE.md`](CCTU_REPRODUCE.md) | **How to check it.** Three levels; the cheapest re-derives every number from shipped transcripts with no GPU. Includes the traps that cost us time. |
| 4 | `CCTU_FINDINGS.md` (not in this branch, see [`docs/RESULTS_POLICY.md`](../docs/RESULTS_POLICY.md)) | **What was learned, including the mistakes.** Eight findings, the five-round history, the bars that were specified wrongly, and what a successor should do differently. |

If you only read one: **(1)** for the numbers, **(4)** for whether to trust them.

## The headline numbers

| | train (140 tasks × 2) | test, held out (60 × 2) |
|---|---:|---:|
| all violation classes | 3726 → **3294** = **−432 (−11.6%)** | 1866 → **1702** = **−164 (−8.8%)** |
| `max_length` (targeted) | 786 → **598** = **−188 (−23.9%)** | 538 → **488** = **−50 (−9.3%)** |
| `acc` | 23.57 → 22.86, **0 gains / 2 losses** | 20.00 → 20.00, **0g / 0l** |
| `SR` / `PSR` | unchanged | unchanged |

Costs are reported as gains and losses, never as a net: a destroyed answer and an unrelated new one are not
interchangeable.

## Three things to know before quoting any of this

**`SR` and `PSR` never moved.** Not once, in thirteen arms, on either split. `PSR` requires the violation
count at exactly zero across every class; the deployed anchor moves one class by a quarter and clears no
episode to zero. The honest summary of this work is *a 12% reduction in total violations with no movement in
the benchmark's own headline metrics.*

**64% of the residual was unreachable.** `identifiers`, `punctuation` and `format` — 2,394 of 3,726 violations
on train — are judged by per-episode validator files this integration withholds, so no declared field can
detect them and no anchor here could target them. They fall as a side effect; none was aimed at. That is the
ceiling this work ran into.

**The deployed anchor is better than the one it replaced everywhere except one cell.** On the targeted class on
the held-out split it is worse by 6 events (−50 vs −56), while being seven times better there across all
classes (−164 vs −22). It was accepted under a bar that compared against the incumbent on train but not on
test — a disclosed defect in the criteria, corrected for future rounds and deliberately not applied
retroactively. Both readings are laid out in (1) so you can disagree with the call.

## Where the rest lives

| | |
|---|---|
| adapter, screens, scoring | `benchmarks/cctu/` |
| standing rules, installed-anchor record | [`../benchmarks/cctu/ANCHOROPT.md`](../benchmarks/cctu/ANCHOROPT.md) |
| what is upstream vs ours, and the scorer fix | [`../benchmarks/cctu/VENDORED.md`](../benchmarks/cctu/VENDORED.md) |
| per-round pre-registrations and verdicts | `../rounds/CCTU_QWEN{1..5}/FROZEN.md`, `RESULT.md` |
| deployed anchor spec | [`../rounds/CCTU_QWEN5/controller.json`](../rounds/CCTU_QWEN5/controller.json) |
| raw artifacts | `benchmarks/cctu/results*/qwen/` |

**Granite is a separate integration** with a differently-shaped residual (round exhaustion rather than refusal
at the answer). Its numbers do not transfer and are not covered here.

## Method, briefly

Each round pre-registers its criteria in a `FROZEN.md`, hashes the file, and records the hash before the first
arm runs — so a reader can check that no threshold moved after a result was seen. Every acceptance walks a
fixed order of terms, each vetoing before the next is computed: execution, attribution, engagement,
withdrawal, harm, benefit, replication on the held-out split. Two rounds were voided and one rejected by their
own criteria. The rounds that failed, and the criteria that were themselves wrong, are documented in (4) at
the same length as the one that succeeded.
