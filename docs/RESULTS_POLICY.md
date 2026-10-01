# What this branch contains, and what it deliberately leaves out

This repo is assembled for external review. It ships the **method, the five benchmark adapters, the
measurement protocol and the reproducibility guide**. The paper's final results tables — across
backbones, benchmarks and baselines — are transcribed in [`PAPER_RESULTS.md`](PAPER_RESULTS.md) as
reported figures. What this repo does **not** ship, and the subject of this page, is the **per-case
data and campaign artifacts** behind most of those cells: this repo's own reproduction guarantee
(`scripts/verify_progression.py` recomputing a number from shipped per-case results, with no model and
no GPU) covers the granite-4.1-8B BFCL v4 progression and CCTU, and nothing wider.

That narrower scope is a deliberate choice, not an omission, and the reason is stated here rather than
left for a reader to infer from an absence.

---

## Why the results tables are not here

At the time this branch was cut, three of the four measurement campaigns were **open**: re-runs were
in flight, a pooling question was unresolved, or a harness-parity gap was unclosed. Publishing a
number whose own branch describes it as uninterpretable would be worse than publishing none.

| campaign | state when this branch was cut | consequence |
|---|---|---|
| BFCL v4 / qwen | source branch records the anchor results as **store-divergent and uninterpretable**; clean re-runs submitted on both splits | excluded — the numbers on that branch supersede themselves |
| BFCL v4 / minimax | factorial validation cells **pending**; the paper's Table 2 left unedited on that basis | excluded — incomplete, not negative |
| τ² (tau2) | AnchorOpt's runner and tau-bench's stock runner **disagree by ~1.5 tasks (z ≈ −1.0)** | excluded — a runner-parity gap is a confound, not a result |
| AppWorld | granite and minimax arms promoted and validated; the **qwen campaign closed at a null** (12 arms spanning −4 to +2, and a +2 promotion that reversed on held-out) | excluded here only because it is reported alongside the above |
| CCTU | **final** | the round chain ships in [`rounds/`](../rounds/) — see below |

The one exception is CCTU. Its results were final, so the full round chain
(`rounds/CCTU_CYCLE0`, `CCTU_QWEN1`…`CCTU_QWEN5`) ships, because
[`CCTU_REPRODUCE.md`](CCTU_REPRODUCE.md) Level 1 re-scores from those shipped transcripts with no GPU.
Removing them would have removed the only end-to-end reproduction in this branch that needs nothing
but a CPU.

**The numbers are not retracted.** They live on the working branches named below, with their own
caveats attached, and they are the right place to read them. What is asserted here is narrower: this
branch does not present them as settled.

---

## What *is* verifiable here, with no model and no GPU

The BFCL v4 A1–A8 progression **is** in this branch, in `README.md` and `rounds/anchors.py`, because
it is settled and because the test suite pins it: `scripts/verify_progression.py` recomputes every
published accuracy from the shipped per-case results and exits non-zero on drift, and
`tests/test_progression_integrity.py` fails if the README and `rounds/anchors.py` disagree by a single
digit. A figure in this repo is data in one file; prose never restates one.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the three tiers and what each costs.

---

## Provenance — the exact commits this branch was assembled from

Every source branch was live when this branch was cut; two received commits within minutes of it. So
the graft is pinned to **SHAs, not branch tips**, and they are recorded here so the assembly can be
audited or replayed.

| component | source branch | pinned commit |
|---|---|---|
| base: core, BFCL v4 adapter, TB2 adapter, `rounds/`, tests | trunk point shared by all three `exp/*` branches | `e9c200ce725d4b70770c9b3b2962d01ad216e28d` |
| `benchmarks/appworld/`, AppWorld docs | `adapter/appworld` (local tip, 4 commits ahead of `origin`) | `18c0a0c1a81f59a1c41f187f0036f731c206e787` |
| `benchmarks/tau2/`, `docs/TAU2_ADAPTER.md` | `bench/tau2-selfteach-round1` | `d2c68399769aac28bb320429fb8445059cb0a942` |
| `benchmarks/cctu/`, `rounds/CCTU_*`, CCTU docs | `cctu-results` | `f46bab2df82695db5661be1c66bf1a3531f8328d` |

### Why the base is the trunk point and not `adapter/appworld`

`adapter/appworld` carries the most developed adapter, so it is the natural place to look for a base.
It is the wrong one: it forked from trunk on 2026-09-20 and is **320 commits behind**, missing
**3,705 lines across 22 files of `anchoropt/learning/`** — including eleven modules that do not exist
on it at all (`acceptance_criteria`, `affordance_requirements`, `candidate_library`,
`candidate_selection`, `constraint_feasibility`, `delayed_effect`, `golden_registry`, `multi_round`,
`regression_protocol`, `round_ledger`, `state_cache_identity`) and roughly thirty test files.

Basing on trunk and grafting the three benchmark directories on top gets the newest core *and* all
five adapters. It is a clean graft rather than a merge because the three directories are disjoint —
`benchmarks/appworld/`, `benchmarks/tau2/` and `benchmarks/cctu/` each exist on exactly one source
branch.

### What was dropped in assembly

| dropped | size | reason |
|---|---|---|
| `docs/slides/*.pptx` | 7.3 MB | internal presentation decks; nothing in the repo links to them |
| `benchmarks/tau2/rounds/` | 4 MB | per-arm results from a round whose runner parity is unresolved; no test depends on them |
| six results documents | — | the campaigns in the table above |
| `docs/NEXT_SESSION_PROMPT.md` | — | internal session scaffolding, not documentation |

Markdown references to the dropped results documents were rewritten to plain text pointing here,
rather than left as links that resolve to nothing.

`rounds/` is otherwise intact: the test suite and `scripts/verify_progression.py` read nearly every
directory in it, so it is load-bearing evidence rather than bulk.
