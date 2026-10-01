# Porting Self-Evolve to a new benchmark — the adapter contract

Tag: `self-evolve-v0.1-adapter-mvp`

You should not need to modify the central self-evolve algorithm to get started. Everything
benchmark-specific belongs in **one adapter module**. If you find you cannot express something through
the interface below, that is useful evidence — record it (§6) rather than patching the optimizer.

```
Benchmark
   ↓
Runtime Adapter                     <-- you write this
   ↓
{state, loci, signals, feasible actions, executors, outcome}
   ↓
Self-Evolve                         <-- unchanged
```

## 1. What the adapter must provide

Duck-typed, no ABC to inherit. The reference implementation is
[`benchmarks/bfcl_v4/bfcl_runtime.py`](../benchmarks/bfcl_v4/bfcl_runtime.py); a second, deliberately
different one is [`benchmarks/tb2_deepagents/tb2_adapter.py`](../benchmarks/tb2_deepagents/tb2_adapter.py).

| Concern | Function | Contract |
|---|---|---|
| trajectory / state normalization | `normalize_event(event) -> dict` | one host event → a boundary-tagged, framework-neutral record. **Every** benchmark-specific string decode happens here. |
| | `observable_state(normalized, carried=None) -> dict` | what a signal may read at this boundary. Cross-boundary evidence must be passed **explicitly** via `carried`, not accumulated in hidden mutable state. |
| decision loci | `IncisionPoint` values you support | the boundaries at which a controller may act. |
| observable signals | `declared_signals() -> tuple[str,...]` | Φ's names. |
| | `evaluate_signal(name, state, params) -> bool` | dispatch with typed-param validation; an **unknown name must RAISE**, never return False. |
| HOST / feasible actions | `HOST = HostProfile(...)`, `feasible_actions(locus)` | U_H(locus): what the host *permits*. Declare a cell only where something demonstrably ran. |
| executor / materialization | `EXECUTORS[(locus, action)]`, `executor_supports(locus, action, signal, eta)` | what the host can **actually run**. See §4 — this is the distinction that costs the most. |
| candidate grounding | `ground_reprompt(...)`, `ground_suppress(...)`, … | η_μ realizations per action family, from declared capabilities. Several groundings = several arms. |
| outcome scoring | per-case `valid` + per-episode trajectory | the paired outcome. Two independent scorer paths are strongly recommended (§4). |
| prerequisite semantics | how setup episodes are separated from scored ones | **read §4 before you skip this.** |

The core never sees your domain vocabulary. `tests/test_tb2_adapter.py` greps the `anchoropt/`
package to keep it that way — mirror that test for your benchmark.

## 2. The loop you are plugging into

```
failed trajectories → attribution → residual ranking → localize (locus, signal)
  → runtime-constrained feasible actions → concrete candidate policies
  → AnchorPolicyOpt evaluates/selects → promote → rerun incumbent
  → re-attribute / re-mine residuals → next residual
```

Driver: [`scripts/self_evolve_cycle.py`](../scripts/self_evolve_cycle.py). One command; the only step
outside it is the GPU/agent rerun, passed in as `--after`.

Worked example from the BFCL v4 run, shipped in [`rounds/SE1_A4_promoted/`](../rounds/SE1_A4_promoted/):
policy, re-attributed diagnoses, next-round candidates, and the round log.

## 3. Minimal porting checklist

The first milestone is **not** an accuracy improvement. It is reaching candidate construction:

```
failed trajectories → attribution → residual → localization → feasible candidate construction
```

1. trajectories can be normalized into discrete **decision steps**;
2. failed cases can be handed to the Attributor as **runtime facts** (no labels, no error classes, no
   hints — it must explain the failure, not recognize a category you named);
3. signals are **observable at the claimed locus** (see §4);
4. `feasible_actions(locus)` returns something;
5. **materializable** actions are distinguishable from merely **admissible** ones (§4);
6. intervention firing is **recorded per step** and readable afterwards (§4);
7. task outcome can be scored per case;
8. setup/prerequisite state is **isolated from interventions** unless that is deliberately part of the
   treatment (§4);
9. one dry self-evolve round reaches candidate construction.

Then, if practical: evaluate → promote/reject → rerun → re-mine.

Run `python scripts/self_evolve_cycle.py --before <run> --after <run> --out <dir>` without `--live`
first: it exercises steps 1, 3–9 with no model calls.

## 4. Known issues — please do not rediscover these

Each cost us a round or a retracted result.

**Interventions must not silently modify prerequisite/setup state.** Prereq episodes write the store
that scored episodes read, and an executor armed for the whole run fires on them too. Measured: on one
held-out cell the arms fired 22 / 12 / 0 times during prereqs and their prereq trajectories diverged,
so that cell was **not a paired comparison** — the arms answered against different stores. It produced
a "gain" on a case where the arm never fired. Check per-arm prereq firings before quoting any delta.
([`docs/PREREQ_FIRING_CONTAMINATION.md`](PREREQ_FIRING_CONTAMINATION.md))

**Gate bookkeeping rows are not model decisions.** When an executor fires it may append its own
record (no status, no decoded calls). Counting that as a step compares the arm's *firing record*
against the control's *first real decision*. This produced a retracted "sampling drift" finding.
Filter such rows before any step/first-decision comparison.

**Gains on episodes where the controller did not fire are not attributable.** Audit that every
reported gain sits on a fired episode. Our pooled net was +4 while only 3 gains were on fired cases;
that arithmetic gap is what exposed the contamination above.

**Admissible ≠ materializable.** `U_H(locus)` declaring an action legal is necessary and *not*
sufficient. Require a real executor that exists for `(locus, action)`, covers the signal, and consumes
η **unchanged** — a parameter it would have to coerce must be **rejected**, because coercion runs
different semantics under the proposal's name. Conflating the two inflated a reported "113
counterfactuals" by 91 non-runnable arms; on the A4 locus, executor-backed feasibility pruned 21
admissible arms to 4.

**Signals must be observable at the claimed locus.** One anchor fired before generation on a
condition ("about to answer without looking") that does not exist there: it injected in 303/303
episodes instead of ~48 and cost −4.95 pp. That is a specification error, not a failed hypothesis.

**Route proposer output through the canonical ingest path.** `propose_semantic` emits
`action_preference` and **no** `action_set`. A hand-written parser looking for `action_set` finds
nothing and silently builds **zero** candidates from perfectly good proposals — our funnel read
`proposed=0`. Use the rule in `scripts/live_policy_recovery.to_search_space`: an absent or
single-action set **widens** U-hat to every host-executable action at that boundary. Likewise send
attributor records through `proposal_seams.ingest_attribution` — it owns the required-key list (the
field is `failure_mechanism`, not `mechanism`) *and* the forbidden-key guard that stops an attributor
from naming a locus, signal or action. A second parser bypasses that guard.

**Firing telemetry may not be where you expect.** On our host, remedy-flag executors do **not** appear
in the aggregated `gates_fired` dict (that is registry-derived), so the per-step trajectory sidecar is
the only true source. Reading the wrong channel reported 0 firings next to a real count of 46.
A zero-firing arm and an unparsed telemetry channel look identical in a summary — make them distinguishable.

**Residual-family canonicalization and experiment memory are v0.2 work.** Do **not** reimplement them
in an adapter. Known v0.1 limitation: residuals are grouped by the attributor's raw prose, so
paraphrases fragment the ranking — one of our re-mining rounds produced 65 diagnoses under 17 "distinct"
residuals, all the same failure mode. Centrally solved in v0.2; adapters should expect it to change.

## 5. Keep adapters thin

Avoid benchmark-specific changes to the central optimizer. Two guards worth copying:

* a test that greps `anchoropt/` for your domain vocabulary (container names, tool names, error
  wording, case-id shapes) and fails if any leaked in;
* if you inject fixtures for a recovery experiment, keep the runtime tables **empty** and inject
  explicitly in tests. Ours shipped an answer key transitively and a leak detector caught it.

## 6. When the interface is insufficient

That is a result, not a blocker. Open an issue with exactly:

```
benchmark
missing abstraction
example trajectory
why the current interface is insufficient
smallest proposed extension
```

We already know one real gap: `feasible_actions(locus)` takes only the locus, while materializability
depends on `(locus, action, signal, η)`. The eventual interface is sketched in
[`anchoropt/telemetry/roles.py`](../anchoropt/telemetry/roles.py) as typing-only Protocols — read them
for intent, do not implement against them yet.

## 7. Where to start reading

| Order | File | Why |
|---|---|---|
| 1 | `benchmarks/bfcl_v4/bfcl_runtime.py` | the reference adapter; its module docstring states the contract |
| 2 | `scripts/self_evolve_cycle.py` | the loop you plug into |
| 3 | `anchoropt/learning/anchor_policy_opt.py` | how candidates become arms; `build_arms` is the feasibility funnel |
| 4 | `anchoropt/learning/action_contract.py` | required η per action family |
| 5 | `anchoropt/learning/prestate_eligibility.py` | which cases may be compared causally, and why |
| 6 | `rounds/SE1_A4_promoted/` | a real completed cycle's artifacts |
| 7 | `docs/PREREQ_FIRING_CONTAMINATION.md` | the defect most likely to bite a new adapter |
| 8 | `benchmarks/tb2_deepagents/tb2_adapter.py` | a second adapter, so the shape is not BFCL-specific |
