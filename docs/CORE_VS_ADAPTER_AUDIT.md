# Core-vs-adapter change audit

`core-v2` (`22087e29`) → `exp/claude-granite-cycle1` (`ff71e2d2`).
**176 files, +40,954 / −238.** Most of that volume is experiment records, not code.

| group | files | +/− | what it is |
|---|---|---|---|
| **1. core** `anchoropt/` | 10 | +1,705 / −40 | 4 new modules, 6 edited |
| **2. adapter** `benchmarks/bfcl_v4/` | 7 | +1,034 / −10 | 2 new, 5 edited |
| **3. BV patches** `patches/` | 40 | +3,513 / −0 | all new, isolated-runtime only |
| **4. orchestration** `scripts/` | 6 | +1,609 / −18 | 4 new, 2 edited |
| **4b. backend mechanism** `scripts/capacity_relocate.py` | 1 | +226 | *counted in group 4; reported separately below* |
| **5. tests / docs / records** | 111 | +33,077 / −157 | tests +4,679 / −155 · docs +552 · records +27,744 |
| — `integrations/`, `examples/` | 3 | +116 / −13 | meta-model provider slot; toy-host demo |

**Code vs record.** Executable change is ~**7,900 added lines** across core, adapter, patches and
orchestration. The other ~33,000 is tests (4,679) plus `rounds/` and `results/` artifacts (27,744) —
provenance, not architecture.

---

## 1. Core (`anchoropt/`) — benchmark-independent

**New modules, all four reliability contracts rather than algorithm:**

| module | lines | kind |
|---|---|---|
| `regression_protocol.py` | +355 | reliability — L1/L2/L3 levels, certificates, composition |
| `golden_registry.py` | +312 | reliability — accepted controllers as permanent fixtures |
| `affordance_requirements.py` | +169 | reliability — what a repair would need in order to exist |
| `constraint_feasibility.py` | +157 | **algorithm** — operator must reduce the constrained *quantity* |

**Edited modules:**

| module | +/− | kind |
|---|---|---|
| `structured_search.py` | +411 / −12 | **algorithm** — `_fires_on_any` observability guard, `UNFIREABLE_HERE` advancing the boundary instead of expanding Φ, no-evaluator localization sweep |
| `executor_capability.py` | +130 / −0 | reliability — `capability_id`, `requires_eta`, `eta_delivered_via` (ghost-capability closure) |
| `external_evaluation.py` | +88 / −11 | reliability — `TRAIN_IMPROVED_PENDING_VALIDATION`, `INCOMPARABLE_INCUMBENT`, the incumbent-token precondition |
| `anchor_policy_opt.py` | +31 / −10 | algorithm — η grounding/θ search separation |
| `block_loop.py` | +26 / −5 | algorithm — grouping by observable condition, not prose |
| `search_state.py` | +26 / −2 | algorithm — `UNFIREABLE_HERE` state |

**Split: ~1,200 lines reliability contract, ~500 algorithm.** The only new *algorithmic* capability is
constraint feasibility — the KV lesson that a correct signal can carry an infeasible operator.

**Genericity: clean.** `tests/test_core_genericity.py` passes (85 checks), the frozen per-file debt
ledger **did not grow**, and all four new modules score **0** benchmark identifiers. `tests/
test_structured_search.py` still drives the whole WHERE→WHAT→HOW schedule against the toy host, which is
the positive proof a second benchmark can reuse core unchanged.

## 2. Adapter (`benchmarks/bfcl_v4/`) — benchmark-specific execution

| file | +/− | kind |
|---|---|---|
| `bfcl_runtime.py` | +540 / −10 | capability declarations, observation supply, `ground_capacity_relocations`, feasibility-gated `ground_transforms` |
| `bfcl_regression_host.py` | +319 | reliability — L1/L2 through the **real** installer, predicates, phase rule and dispatch guards |
| `bfcl_constraints.py` | +97 | benchmark semantics — the three capacity constraints and their units |
| `bfcl_signals.py` | +53 | 2 new predicates |
| `run_memory_eval.py`, `traj_sidecar.py`, `memory_evaluator.py` | +25 | hook wiring and telemetry allowlist |

This is where every memory-specific fact lives, correctly.

## 3. BV patches (`patches/`) — isolated runtime only

40 files, +3,513, **all new**. Each refuses the shared tree (exit 3) and is idempotent. The shared BV
evaluator carries **0** patch markers; 9 GEPA jobs were undisturbed throughout.

## 4. Orchestration (`scripts/`)

| file | +/− | kind |
|---|---|---|
| `self_evolve_cycle2.py` | +801 / −18 | the round driver: stratified sampling, phase decision, observation supply, settled-intervention ledger |
| `preflight_controller.py` | +237 | reliability — pre-GPU execution checks |
| `pair_runs_to_results.py` | +161 | reliability — paired scoring |
| `affordance_audit.py` | +160 | reliability — unrealizable-repair ledger |
| `install_controller.py` | +24 | the declared-signal predicate seam |

### 4b. Backend-specific mechanism, reported separately

`scripts/capacity_relocate.py` (+226) is a **BFCL memory mechanism**, not orchestration. It sits in
`scripts/` because `capacity_repair.py` does, and the BV runner imports both from there. Its backend
coupling is **declared as data** in `RELOCATION_PAIRS` (`core_memory` → `archival_memory`, caps 7→50),
not spread through control flow — which is why it *declines* cleanly on vector rather than misbehaving.
A future move to `benchmarks/bfcl_v4/` would be a rename, not a redesign.

## 5. Tests, docs, records

Tests **1,549 → 1,684** (+135 net; +4,679 / −155). Every new contract has a test that would have caught
the original defect. Docs +552 (`COLLABORATOR_STATUS.md`, the qwen workstream handover, the criterion-4
clarification). Records +27,744 across `rounds/` and `results/`.

---

## BFCL-specific assumptions remaining in generic code

Audited by identifier, then each hit read in context.

| location | finding | blocks unattended? | blocks Qwen? |
|---|---|---|---|
| `anchoropt/` (all 10 files) | **none.** Guard passes; debt ledger unchanged; 0 hits in new modules | no | no |
| `scripts/self_evolve_cycle2.py` | 4 hits (`rec_sum`, `no_capacity`, `entry_too_long`, `blob_would_overflow`) — **all in one comment block** recording raw-trace counts | no | no |
| `scripts/self_evolve_cycle2.py:49` | **`import bfcl_runtime as runtime` at module level** — the one real structural coupling | no | no |
| `scripts/affordance_audit.py:120` | `for cell in ("kv","vector","rec_sum")` — a hardcoded cell list in a diagnostic script | no | no |
| `integrations/claude_roles/` | 1 hit, a `granite-4.1-8b` **docstring example**. The meta-model slot is benchmark-free | no | no |

**On the one real coupling.** The driver uses the runtime through exactly **7 methods** —
`states_at`, `declared_signals`, `evaluate_signal`, `parameter_domains`, `probe_params`, `is_decision`,
`hook_state_fields` — all part of the documented adapter contract that `anchoropt/testing/
adapter_contract.py` checks. So a second benchmark needs the import swapped for a `--adapter` lookup,
**one line**, not a redesign of the driver. Diffuse assumptions would be the expensive case; this is not
that.

**Neither the unattended nor the Qwen experiment is blocked.** Qwen changes the *meta-model* slot
(`ANCHOROPT_META_*`), which is independent of the adapter and already supports `openai_compat`, and it
runs the same BFCL adapter. **Recommendation: do not refactor now.** Change the import to a lookup when a
second adapter actually exists; until then the change would be untested indirection.

## Flagged: the KV relocation bound mismatch

| | |
|---|---|
| documented | `MAX_RELOCATIONS_PER_EPISODE = 3` |
| **actual** | **3 per TURN** |
| cause | the patch declares `_relocations_done` beside `_capacity_repair_fired`, which the host resets **inside the turn loop** (`memory_evaluator.py:2198`, indent 12) |
| measured effect | 84 requested, **61 performed**, up to **8 in one episode**, 23 unserved; 8 of 24 episodes hit the bound |

**The measured implementation is preserved unchanged**, and the accepted result stands on its own
evidence: **67** relocations were write-verified and copy-survives-verified, **61** of which also landed
their retry, with **0** information-losing removes and **0** clears added. (The earlier "61/61" conflated
the preservation and end-to-end denominators — see
`rounds/AUTONOMY/RELOCATION_CHAIN_RECONCILED.json`. Corroborating the bound defect: 8 of the 23 unserved
requests occurred after *five* prior relocations in the same trajectory, impossible under a per-episode
bound.) What was wrong is the *documentation* of the
bound, now corrected in `rounds/GOLDEN/registry.json` rather than quietly restated.

**The open decision is whether the safety contract should say per-turn or per-episode.** Per-turn is a
weaker bound than intended; per-episode is what "unbounded eviction is a wholesale clear in slow motion"
argues for. Moving the counter to episode scope **changes execution semantics** and therefore requires a
paired re-measurement (L3) before the acceptance transfers. Not changed pending that decision.

## How close is a stable reusable core?

**Close, on the evidence rather than on intent:**

- core names no benchmark identifier, enforced by a two-tier guard whose frozen debt did not grow
- the full algorithm runs against a toy host with its own vocabulary, byte-identically
- the four new contracts are generic by construction and each emits actionable certificates
- three accepted controllers are protected by L1+L2 through the real host path, with L3 an honest SKIP
- the adapter seam is 7 methods, and the driver touches nothing else

**What remains, in priority order:** (1) the driver's one hard import, worth changing only when a second
adapter exists; (2) L3 needs a runtime to be more than a SKIP; (3) `capacity_relocate.py` belongs under
`benchmarks/bfcl_v4/` by a rename; (4) `affordance_audit.py`'s cell list should come from the adapter.
**None blocks the current experiments, and none is worth an engineering cycle now.**
