# Porting AnchorOpt to a new benchmark

You already have the benchmark running under your own harness: an episode loop, trajectories,
scoring, and model access. **You should not need to change anything in `anchoropt/`.** Implement
one adapter and pass your existing runner in as a callback.

```
cp adapters/adapter_template.py adapters/taubench.py     # fill in the TODOs
python scripts/check_adapter.py adapters.taubench        # 7 checks, no GPU
```

> **Start here instead if you are writing an adapter now:** [`ADAPTER_GUIDE.md`](ADAPTER_GUIDE.md) is
> the current mechanics — mandatory vs optional hooks, boundary-specific observability, executor
> capabilities and how to prove a binding by behaviour, intervention variants, trajectory format, and
> both evaluation paths (inline and external). The reference implementation is
> [`examples/toy_host/`](../examples/toy_host/), which passes the same contract checks yours must.
> This file remains the statement of **who owns what**, which has not changed.

---

## 1. Who owns what

| | owns | does NOT own |
|---|---|---|
| **AnchorOpt core** (`anchoropt/`) | **WHERE** to intervene (boundary localization + backward search), **WHAT** condition decides (Φ expressibility, signal synthesis/expansion), **WHAT/HOW** search (action + η), the block-coordinate schedule, promote/reject bookkeeping | your trace format, your tools, your scoring, your model |
| **Your adapter** (`adapters/yours.py`) | which events are decisions; what is observable at each; which actions your runtime can perform; how each action is parameterized; whether your executor can really run a given η | which boundary/signal/action to pick — it is never asked, and must never decide |
| **Your runner** (already exists) | executing an arm, producing trajectories, scoring | anything about the search |

The division has one test: **the adapter answers questions about your runtime; it never answers
questions about the failure.** If you find yourself writing "for this kind of failure, use action X",
that logic belongs to AnchorOpt and putting it in the adapter will invalidate the experiment.

---

## 2. The loop, and where you plug in

```
    your failed trajectories                          <- YOU (your harness)
      -> attribution                                  <- meta-model (configurable, §6)
      -> residual problems, rank-ordered              <- core
      -> optimize_residual():  WHERE -> WHAT -> HOW   <- core  (the algorithm)
      -> a candidate controller                       <- core
      -> paired evaluation                            <- YOU (your runner, as `evaluate=`)
      -> promote / reject                             <- core decides, you install
      -> regenerate trajectories                      <- YOU (rerun the new incumbent)
      -> re-mine the CHANGED residual distribution    <- core
      -> the next intervention
```

Two integration points only: **`evaluate=`** and **installing a promoted controller**.

### `evaluate=` — your runner, wrapped

```python
def evaluate(arm):
    control, treated = run_paired(arm)      # your existing runner, same case set
    if set(control) != set(treated):
        return None                         # denominator mismatch is not a result
    return sum(treated.values()) - sum(control.values())
```

Return any comparable objective (higher better), or `None` when unmeasurable. **`evaluate=None` is a
first-class mode**: the search still localizes, expands and grounds, and stops at
`REALIZABLE_UNMEASURED` — structural rediscovery with no claim of benefit. Port in that mode first.

### Installing a promoted controller

A `Controller` is `(locus, φ, action, η)` and is duck-typed: anything with `.fires_on(state)` and
`.eta` works. Two supported paths:

* **in-process** — `anchoropt.runtime_hook.install(locus, controller)`, then call
  `decide(locus, state, host_default=...)` at the corresponding point in your episode loop. It is a
  no-op until something is installed, so adopting it perturbs no existing measurement.
* **out-of-process** — emit the controller as JSON and have your runner load it. See
  `scripts/install_controller.py` for a worked example whose predicate grammar is an atom or a
  conjunction of atoms over declared fields.

**Your hook must be reached by everything that executes**, and it must be **trigger-free**: if your
executor carries its own arming condition it silently overrides φ, two differently-parameterized
controllers fire on identical states, and the abstraction becomes untestable. φ owns WHEN; the
executor owns only HOW.

---

## 3. The interface

`optimize_residual` duck-types the adapter, so a **module** works as well as an object.

### Required (8)

| hook | returns | purpose |
|---|---|---|
| `is_decision(event)` | bool | which events are consequential choices |
| `boundary_key(event)` | `IncisionPoint` value | which decision point this event belongs to |
| `boundary_from_key(key)` | `IncisionPoint` or None | the inverse; core owns no key→locus table |
| `synthesis_fields(boundary)` | `{name: Field}` | typed observables readable **at that boundary** |
| `declared_signals()` | tuple[str] | current Φ (may be empty) |
| `signal_boundaries(signal)` | frozenset[`IncisionPoint`] | where each signal is observable |
| `evaluate_signal(signal, state, params)` | bool | does it hold here |
| `HOST` | `HostProfile` | which actions are executable at each boundary |

### Optional, and what each unlocks

| hook | unlocks |
|---|---|
| `install_signal(name, pred, boundary=, provenance=)` | **Φ expansion** — without it a blocked residual stays blocked |
| `ground_reprompt` / `ground_suppress` / `ground_substitute_destinations` / `ground_transforms` | candidates for that action; absent ⇒ `*_INFEASIBLE` is reported, nothing crashes |
| `executor_supports(boundary, action, signal, eta)` | feasibility filtering — say **no** rather than coercing a parameter |
| `probe_params` / `parameter_domains` / `policy_class_for` | parameterized signals (θ search) |
| `normalize_event` / `observable_state` | your trace → core's dicts (you will want both) |
| `commits_to_answer` / `actions_in` | target-population accounting |
| `label_for` | figures only, never read by the search |
| `depends_on` | non-temporal boundary ordering; omit and realized order is used |

`IncisionPoint` and `Action` are **core** vocabulary — they describe any LLM call, not any benchmark.
Use them; `HostProfile` is typed on them. Your own boundary keys map back via `boundary_from_key`.

---

## 4. Files you write

| file | you | notes |
|---|---|---|
| `adapters/yours.py` | **write** | copy `adapter_template.py` |
| your `evaluate` callback | **write** | ~10 lines wrapping your runner |
| controller installation | **wire** | one call in your episode loop, or a JSON spec |
| everything in `anchoropt/` | **do not touch** | if you need to, that is a contract gap — tell us |
| `benchmarks/bfcl_v4/` | **do not read as a spec** | one instance, carrying a lot of history |

---

## 5. Smoke-test checklist

`python scripts/check_adapter.py adapters.yours [--events real_rows.json]`

1. **required hooks present** — a missing hook degrades to "nothing was possible here"
2. **a decision boundary exists** — none means the residual cannot be optimized at all
3. **backward search can move** — one boundary means WHERE is not a real coordinate for you
4. **a declared signal is observable** — *and evaluable where declared*, not merely declared
5. **an action can be grounded** — an ungroundable action is indistinguishable from an absent one
6. **a candidate is realizable** — `optimize_residual(evaluate=None)` reaches a controller
7. **core stays generic** — your identifiers did not leak into `anchoropt/`

Then, with a real evaluator: **one candidate completes a paired evaluation** with the denominator
intact.

Run it with `--events` on real trajectory rows before trusting it. Without them checks 2–3 use a
synthetic trajectory and exercise plumbing only.

---

## 6. The meta-model (attribution / proposal)

The model that *reads failures and proposes* is separate from the model *under test*. Select it by
environment, without touching prompts, schemas, search or acceptance logic:

```bash
export ANCHOROPT_META_PROVIDER=claude      # default
export ANCHOROPT_META_MODEL=claude-opus-5
# or
export ANCHOROPT_META_PROVIDER=openai_compat        # Granite, vLLM, any OpenAI-compatible endpoint
export ANCHOROPT_META_MODEL=granite-4.1-8b
export ANCHOROPT_META_BASE_URL=http://localhost:8000/v1
```

Every result record carries the meta-model, so a table row always says which model produced it. See
`docs/META_MODEL.md`.

---

## 7. Known BFCL assumptions that could bite you

These are honest limits of the current code, not of the interface:

1. **`prestate_eligibility.qualifying_in_control`** keeps a BFCL-shaped fallback for
   backward-compatible target-population accounting. Supply `commits_to_answer` and `actions_in` and
   your own semantics are used instead.
2. **24 files under `anchoropt/`** still name BFCL identifiers, catalogued in
   `docs/core_genericity_debt.json` and frozen by a ratchet test. None are on the WHERE→WHAT→HOW path,
   so they do not affect a port — but do not copy their style.
3. **Setup episodes.** If your benchmark builds state before scoring (BFCL calls these *prereqs*), a
   query-time controller must be **inert** during that build. We measured this: 1–2 firings per arm,
   each changing the store, contaminating 24 of 89 scored cases. Guard on whatever flag your runner
   already has to distinguish the phases, and record the skip in telemetry so it is visible.
4. **Two loci only.** BFCL exercises `post_generation_pre_exec` and `post_execution`;
   `pre_generation` allows only NOOP there. If your runtime can intervene before generation, declare
   it in `HOST` — core supports it, it is simply untested.
5. **Single-turn-ish residuals.** Residuals are grouped per episode. Benchmarks with long multi-turn
   dependency chains (AppWorld) may want `depends_on` so boundary order reflects dependence rather
   than wall-clock order.

---

## 8. One-page example

```python
from anchoropt.learning.structured_search import optimize_residual
from anchoropt.runtime_hook import install
from adapters.taubench import ADAPTER
from my_harness import run_episodes, score          # YOUR existing harness code

failed = [e for e in run_episodes(incumbent) if not score(e)]        # your loop, your scoring
events = [ADAPTER.normalize_event(r) for e in failed for r in e.rows]
states = [ADAPTER.observable_state(ev) for ev in events]

class Residual:                                    # whatever groups your failures
    case_ids = tuple(e.case_id for e in failed)

def evaluate(arm):
    install(arm.boundary.value, arm)               # or write a JSON spec your runner loads
    control, treated = run_paired(arm)             # YOUR runner, same case set both sides
    if set(control) != set(treated):
        return None                                # not a paired comparison
    return sum(treated.values()) - sum(control.values())

out = optimize_residual(Residual(), runtime=ADAPTER, host=ADAPTER.HOST,
                        events=events, states=states, evaluate=evaluate)

print(out.state, out.visited, out.moves_earlier, len(out.candidates))
if out.validated:                                  # measured improvement
    promote(out.promoted)                          # then regenerate + re-mine for cycle 2
```
