# Integration audit: reusing this repo for TerminalBench

Audit before implementation, as requested. Baseline: **562 tests passing** at
`tests/` before any change.

## Headline

**The core is more reusable than the task assumed, and two pieces of the requested
work are already done.** Specifically:

1. **The action-vocabulary reconciliation already exists and is enforced by a test.**
   `tests/test_paper_consistency.py` pins `PAPER_OPERATOR` (5 operators) against
   `Action` (4), asserts the divergence set is exactly `{A1, A5, A7, A9}`, and asserts
   every divergence is `REROUTE` splitting into `substitute`/`transform`. There is
   nothing to reconcile — only a decision about whether to *rename*.
2. **The adapter seam already exists, is duck-typed, and documents this exact task.**
   `anchoropt/attribution/__init__.py` says it deliberately stops short of a canonical
   event model "to let the second benchmark inform that shape." TB2 is that second
   benchmark.

`docs/GENERALIZABILITY.md` also already draws the seam and — independently of my TB2
work — reaches the same conclusion: the loop generalizes, signal extraction and action
execution do not.

---

## 1. Reuse / refactor map

| component | file(s) | class | note |
|---|---|---|---|
| typed anchor / controller | `anchoropt/anchor.py` (`Anchor`, `IncisionPoint`, `Action`) | **A reuse unchanged** | 3 incision points are already framework-neutral. `Anchor.__post_init__` validates structurally. |
| structural feasibility grid | `anchoropt/anchor.py` (`feasible_actions`, `_STRUCTURAL_EXCLUSIONS`) | **B small refactor** | Global today. Needs host parameterization → `U_H(ℓ)`. This is the *only* core change I propose. |
| attribution adapter / registry | `anchoropt/attribution/__init__.py` (`register_adapter`, `active_adapter`) | **A reuse unchanged** | One-way dependency already correct; extend the *duck-typed surface*, not the mechanism. |
| evidence ledger | `anchoropt/learning/evidence_ledger.py` | **A reuse unchanged** | Keyed on `(kind, normalized pattern)` — benchmark-agnostic. Already has `STATUS_DEFERRED` = "strong signal, no policy in the current action space… eligible to reopen when the ACTION SPACE grows." Exactly our C9/B-case bookkeeping. |
| phase switch / block coordinate | `anchoropt/learning/phase_switch.py` | **B small refactor** | CONTINUE / EXPAND / STOP logic is general; `partition()` reads BFCL artifact layout (`eval_train_results.json`, `traj/query/*.json`). Needs the residual source behind the adapter. |
| exposure & validation | `anchoropt/learning/exposure.py`, `check_attribution.py`, `target_spec.py` | **A / B** | `exposure_weighted_delta` + `displacement_check` are general disciplines. `TargetSpec` already declares `decision_point`; `check_attribution` calls `active_adapter()` — already seamed. |
| LLM / operator proposal | `anchoropt/attribution/llm_operator.py` | **B small refactor** | Proposal plumbing general; prompts carry memory-container vocabulary. |
| policy search | `anchoropt/learning/policy_tree.py` | **B small refactor** | The tree (intervene? → advisory/active → suppress/reroute) is general and maps 1:1 onto our TB2 mapping. Has BFCL references to seam. |
| attribution expansion / residual mining | `anchoropt/learning/expand_attribution.py`, `attribution/attribution_miner.py`, `learning/remine_incumbent.py` | **B / C** | coverage×precision scoring is **exactly** the generic EXPANDSIGNALS machinery asked for — reuse it. Miners themselves are BFCL-shaped (**C**). |
| locus canonicalization | `anchoropt/attribution/op_canonicalize_locus.py`, `constraint_locus.py` | **C stays in BFCL adapter** | Maps BFCL error strings to loci. `GENERALIZABILITY.md` already says this does not generalize. |
| mechanisms (9 files) | `anchoropt/mechanisms/*.py` | **C stays in BFCL adapter** | `core_memory_*`, containers, vec ids. These are BFCL action *implementations*. |
| **host/runtime action feasibility `U_H(ℓ)`** | — | **D missing** | Semantic vocabulary is general; executable subset is host-dependent. |
| **runtime/benchmark adapter for events + observables + apply_action** | — | **D missing** | Present seam covers *case-id/tool-name vocabulary*, not runtime event normalization. |
| **generic `ResidualDiagnosis` type** | — | **D missing** | So the first diagnosis provider is swappable. |
| **self-evolving outer orchestration** | — | **D missing** | The loop is *documented* (`docs/THE_LOOP.md`) and its pieces exist; the driver does not. |

**Nothing needs to be copied into a TB2 directory.** `repro/integration_v1/` stays
evidence/prototyping, as instructed.

---

## 2. Action-vocabulary reconciliation — recommendation

**Recommendation: adopt the 5-operator paper vocabulary as canonical *semantically*, and
do NOT rename the `Action` enum.**

Why not rename: `anchor.py` states the four-value enum "is written into frozen policy.json
artifacts," and `test_paper_consistency` pins the mapping precisely so it stays legible.
Renaming touches frozen historical records for zero measurement gain — and your own
instruction is not to change frozen records without a plan.

On your three questions, from the repo's own evidence:

- **Should REROUTE become SUBSTITUTE?** Semantically yes — the paper already calls it that
  for A1. Mechanically, keep `Action.REROUTE` as the stored value and treat `substitute` as
  its canonical *name*.
- **Is TRANSFORM genuinely separate?** **Yes, and the repo now has its own evidence for it.**
  `anchor.py`'s docstring argues against a split ("both are one operation"), but
  `A9_EVIDENCE["action_family_caveat"]` concedes A9 "substitutes no call" and "should really
  be called augment-the-observation," and `test_paper_consistency` enforces that caveat's
  survival. A9 changes *what the model observes*; A1/capacity_repair changes *which call
  runs*. Those have different preconditions (SUBSTITUTE needs positive evidence the
  replacement resolves; TRANSFORM does not). So TRANSFORM is a real primitive — the repo's
  anti-split argument predates A9.
  **Caveat, from my own TB2 work:** the *implementation* of TRANSFORM in the A2 prototype was
  boundary-broken (executable at only 1 of 3 boundaries). That is an argument for
  boundary-specific TRANSFORM semantics, not against the primitive.
- **Can BFCL anchors be represented canonically without changing history?** **Yes, already
  demonstrated** — `PAPER_OPERATOR` is a total map over all 8 accepted anchors, and the test
  suite proves it is a refinement, not a contradiction (`test_the_papers_operator_set_is_a_refinement_not_a_contradiction`).

**Migration plan (no frozen record changes):**
1. Add `CANONICAL_OPERATOR: Mapping[Action, str]` to `anchor.py` mapping the 4 stored values
   onto canonical names, with `REROUTE → {"substitute", "transform"}` resolved *per anchor*
   by the existing `PAPER_OPERATOR` table.
2. Keep `Action` unchanged; add `TRANSFORM` **only** if/when a host declares it executable
   and an anchor is filed under it natively. TB2 needs 0 TRANSFORM anchors (measured: 11
   residuals, TRANSFORM count 0), so this is not on the critical path.
3. `test_paper_consistency.py` stays the enforcement point.

---

## 3. Minimal TB2 adapter architecture

Fit to existing repo style: a **module** that registers itself (like `benchmarks/bfcl_v4/adapter.py`),
duck-typed, no ABC (the registry docstring explicitly declines to pin one).

```
benchmarks/tb2_deepagents/
    adapter.py        # normalize_event / observable_state / feasible_actions /
                      # evaluate_signal / apply_action  + register()
    signals.py        # Φ_TB2 — TB2's own signal evaluators
    README.md
```

Boundary mapping — **no fourth boundary**, per instruction:

```
POST_GENERATION_PRE_EXEC        (existing IncisionPoint, unchanged)
    ├── proposed tool action     ← langchain: tool_calls present
    └── proposed terminal response ← langchain: len(tool_calls) == 0
```

`terminal_response_proposed` is extracted **in the adapter** from the `AIMessage`; core never
sees `after_model`. `REPROMPT` is the semantic action; `jump_to="model"` is the adapter's
implementation of it.

## 4. Minimal `self_evolve` orchestration

`anchoropt/learning/self_evolve.py` — a driver that *calls existing components*:

```
execute incumbent (adapter)
  → residual diagnosis (provider: an earlier internal harness now, swappable)
  → evidence_ledger.update
  → phase_switch.decide  ──CONTINUE──> policy_tree → validate → evaluate → install
                         ──EXPAND────> expand_attribution → register signal → freeze
                         ──STOP─────>  done
```

## 5. Files: modify vs add

**Modify (3, all additive):**
- `anchoropt/anchor.py` — host-parameterized `feasible_actions(point, host=None)`, default
  preserves today's behaviour exactly; add `CANONICAL_OPERATOR`.
- `anchoropt/attribution/__init__.py` — nothing structural; docstring note that the runtime
  surface is now exercised by a second adapter.
- `docs/GENERALIZABILITY.md` — record the second adapter as evidence.

**Add:**
- `anchoropt/runtime.py` — `HostProfile` (`U_H(ℓ)`) + `ResidualDiagnosis`.
- `anchoropt/learning/self_evolve.py` — the outer loop driver.
- `benchmarks/tb2_deepagents/{adapter.py,signals.py,README.md}`.
- `tests/test_host_profile.py`, `tests/test_tb2_adapter.py`.

## 6. What makes part of this unnecessary

- **Do not build a new EXPANDSIGNALS.** `expand_attribution.py` already scores
  coverage×precision and already refuses signals that fire equally on successes — with a
  measured example (`single_read_then_answer`, precision 0.51). Register TB2 candidates
  through it.
- **Do not build a ledger.** `evidence_ledger.py` already has the exact status
  (`STATUS_DEFERRED`) for "valid signal, no action yet," which is our whole category-B set.
- **Do not build a policy search.** `policy_tree.py`'s tree already terminates in
  suppress/reroute vs reprompt — the same decision our TB2 mapping makes by hand.
- **Do not reconcile the vocabulary.** Already done and test-enforced.
- **`docs/CONSUMER_BOUNDARY_RULE.md` already exists** — the rule my TB2 audit rediscovered
  as "TRANSFORM reads a payload absent at its boundary." Cite it rather than restating it.
