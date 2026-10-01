# Changelog

## core-v2 (unreleased — tag deliberately not yet created)

The theme is **validity**: every change below removes a way the loop could report a result it had not
measured. Three defects were found in core itself, each of which had made the honest path unusable and
left a stub as the only thing that worked.

### Breaking

| change | why | what to do |
|---|---|---|
| `Action.SUPPRESS` no longer requires `preservation` in its family floor | remove-outright suppression preserves nothing and needs to; demanding the field made every such arm ground a safety string no executor read | nothing, if your grounding already emits `suppressed_operation`. A withhold-**and-replay** grounding must set `variant` to `withhold_and_replay` (or `replay_recorded_result`), where `preservation` is still required **and** must be executor-backed |
| `executor_supports` is superseded by `executor_capability` | the old hook was a declaration lookup: core could only pass your verdict through, never check it | keep `executor_supports` and nothing breaks; add `executor_capability` to get ghost and inert-eta checking. See [`docs/ADAPTER_GUIDE.md`](docs/ADAPTER_GUIDE.md) §3 |
| an evaluator may return `None` | an unmeasured arm has no score; returning `0` would rank it against real measurements | if your `evaluate` returned `0` for "not measured", return `None` instead |
| `PolicyOptResult` gained `unevaluated` | "we did not run it" was indistinguishable from "it did not work" | read it if you report outcomes |
| the cycle driver no longer prints a chosen `NEXT ARM` | selection by `abs(firing_rate − 0.25)` selected without measuring | the propose phase writes `arm_manifest.json` and stops at `UNEVALUATED`; pass `--results` to select on measurement |
| `controller.json` → `controllers.json` (one spec per arm) | no arm is privileged until something measures them | `controller.json` is still written for back-compat: the first spec in deterministic label order, carrying **no** claim of being best |

### Core defects fixed

1. **`optimize_residual` could not consume a real evaluator.** It compared results with `obj > best`,
   which works for a scalar and raises `TypeError` for the `ThetaResult` the optimizer produces. So the
   only evaluator the schedule could run was a stub returning a number — which is exactly what the BFCL
   driver passed (`evaluate=lambda _a: 0`). Ranking now goes through `train_objective`, the project's
   existing J_train; scalar evaluators behave as before.
2. **`AnchorPolicyOpt.optimize` crashed on an honest evaluator.** It passed every return to
   `self.objective`, so an evaluator declining to score an arm (`None`) raised `AttributeError`. `None`
   is now dropped, and such arms land in `unevaluated`.
3. **`destination_attest` could never import.** A `sys.path.insert` pointed at the wrong directory.
   Pre-existing since the original port; found only by importing every core module in a clean checkout.

### Driver defects, found by running the real BFCL path

Three defects that no synthetic fixture reached, all in the heuristic's removal:

4. **`NameError: 'fires' is not defined`.** A vestigial `next_arm` record block still read the variable
   the firing-rate heuristic had owned. The manifest wrote first, so a partial run looked like progress.
   The field is removed outright — it named one arm as "the next one", which is the heuristic's last
   vestige.
5. **Every manifest arm reported `fires_on_states: 0`.** `fires_by_label` keyed on
   `instantiated.label` (the last path segment) while the manifest keys on `arm.label` (the full path).
   Visibly false — the fireability filter had already excluded every all-zero arm — and it survived
   because nothing read the field back.
6. **The propose phase's `UNEVALUATED` was overwritten** by `AWAITING_EVALUATION` on the way out, and a
   `--results` round lost its outcome class entirely.

### Validity

* **Ghost executors rejected.** A capability must name a `binding` — the code that runs. The BFCL
  commitment-gate suppress cell named `enable_redundant_write_suppress`, a flag appearing **zero times**
  in the hook that withholds the call.
* **Inert eta rejected.** A capability declares `consumes` (eta keys the code reads) or
  `eta_is_computed=True` (it derives its own from live state). A contract-required field in neither is
  refused. An `enforced` claim such as `preservation` needs an executor regardless.
* **Inert actions disabled, not deleted.** BFCL's upstream `reprompt` writes its instruction to the step
  record and never injects it — two arms differing only in `instruction` were byte-identical across
  13/13 episodes. The cell is declared `disabled_reason=…`, so core refuses arms there **with the
  reason**; deleting it would make it look unconsidered.
* **Four disjoint outcome classes**, with a test against a refactor collapsing any pair:
  `IMPROVED` · `UNEVALUATED` (built, nothing ran — **not** a negative result) · `NO_BENEFIT` (all
  measured, none helped — a real null) · `BUDGET_EXHAUSTED` (arms left open — requeue).
* **A behavioural contract for bindings.** Core must not read your source, so a binding is established
  by running the mechanism with the controller absent and present and asserting the runs differ *in the
  way the action claims*. A telemetry-only mechanism fails it.

### Added

* `anchoropt/learning/executor_capability.py` — bound capabilities and the three rejection classes.
* `anchoropt/learning/external_evaluation.py` — `arm_manifest`, `ExternalEvaluation`,
  `select_on_measurement`, for hosts whose evaluation is a cluster job rather than a function call.
* `anchoropt/testing/` — `check_adapter_contract(MY_ADAPTER)`, tested by violation (21 tests, each
  negative case breaking exactly one thing). Mandatory vs optional is enforced, and each missing
  optional hook reports what it costs.
* `examples/toy_host/` — the **reference implementation**: a complete adapter plus a real executing
  host, deterministic, no model or network. `demo.py --no-late-repair` runs the whole algorithm.
* `docs/ADAPTER_GUIDE.md` — the porting guide, with its executable claims pinned against the code.

### Backward compatibility of archived results

**Every frozen result stays reproducible.** Five archived controller specs (`rounds/WRITE1`,
`WRITE2`, `WRITE2_SCORE`, `WRITE3`) carry `preservation` — the eta field the capability contract now
refuses to *build an arm* with. They are historical records of what actually ran, so they are **not**
rewritten.

The boundary that makes this safe, now pinned by a test: the eta contract governs **arm construction**
(what the search may propose and measure), not **controller installation** (what a runner executes from
a frozen spec). `preservation` is inert at install time — carried, never read — which is exactly the
diagnosis that moved it out of the family floor. Verified: each archived spec still installs at its
locus and the WRITE2 predicate still fires as recorded (`>272` → fires at 400, not at 50).

`ground_suppress` no longer emits the field, so **new** arms do not carry it. On the first real run the
inert-eta rule correctly refused all 39 suppress candidates at the commitment gate — including the
controller WRITE2 discovered — until the grounding was corrected.

### Preserved

The algorithm is unchanged: residual-driven backward localization, alternating signal/policy
optimization, sequential moving-incumbent acceptance, re-mining. Four action families, still closed. Both
accepted SUPPRESS variants remain valid and neither was weakened. No `WITHHOLD` action was added; no
second optimizer was written.

---

## Migration notes

### If you only consume core

Nothing to do unless your `evaluate` returned `0` for an unmeasured arm — return `None`.

### If you maintain an adapter

Optional but recommended, in order of value:

1. **Add `executor_capability`.** Ten minutes, and it is what buys ghost and inert-eta rejection. Keep
   `executor_supports` alongside it; core prefers the new hook and falls back.
2. **Add a behavioural probe per capability.** Copy `BoundaryProbe` from
   `tests/test_executor_behavioral_contract.py`. This is the only check that establishes the binding.
3. **Run the contract checker.** `check_adapter_contract(MY_ADAPTER, states=…, events=…)`. Pass `states`
   and `events`: without them the boundary-projection and localization checks are skipped.
4. **Tag your replay-variant groundings.** If you ground withhold-and-replay suppression, set its
   `variant` and keep `preservation`.

### If you run the cycle driver

The propose phase no longer names an arm. Two phases now:

```bash
# 1. emit every measurable arm; the round stops at UNEVALUATED
python scripts/self_evolve_cycle2.py --incumbent RUN --out OUT --cell vector
#    -> OUT/arm_manifest.json, OUT/controllers.json

# 2. evaluate the arms externally, then select on measurement
python scripts/self_evolve_cycle2.py --incumbent RUN --out OUT2 --cell vector --results results.json
```

`results.json` is keyed by `arm_label` exactly as the manifest emits it. **An arm you did not evaluate
must be omitted, not reported as zero** — a fabricated zero outranks a genuinely measured `−2` arm under
J_train, so an arm nobody ran would be selected over one measured and found harmful.

### Known limitations

* The driver's CLI path is now verified on **real BFCL artifacts** — see
  [`docs/BFCL_INTEGRATION_VERIFIED.md`](docs/BFCL_INTEGRATION_VERIFIED.md). The synthetic-corpus tests
  that used to skip have been **deleted**, not left in place: every one had a real-artifact replacement in
  `tests/test_bfcl_integration_e2e.py`, and a permanently skipped test verifies nothing. The
  artifact-dependent tests there skip only if `/tmp/c2` is absent (it is ephemeral); the four defect
  regressions are source-level and never skip.
* `tests/test_tb2_middleware.py` (9 tests) needs an optional pinned TB2 runtime and is not collected
  without it.
* Seven verbatim-ported core modules still manipulate `sys.path`; frozen as a ratchet that may shrink
  but never grow, and no release-critical module is among them.
