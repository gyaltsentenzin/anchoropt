# Building an AnchorOpt adapter — TauBench and AppWorld

A practical guide for implementing a benchmark-specific adapter. It assumes you already run your
benchmark (episode loop, trajectories, scoring, model access) and want AnchorOpt to learn
interventions over it.

**You should not need to change anything in `anchoropt/`.** If you do, that is a core defect — report
it rather than working around it in the adapter.

Read [`PORTING.md`](PORTING.md) first for who-owns-what. This document is the mechanics.

---

## 0. The 10-minute path

```bash
# 1. Run the reference implementation, so you know what "working" looks like.
python examples/toy_host/demo.py --no-late-repair

# 2. Copy the template and fill in the TODOs.
cp adapters/adapter_template.py adapters/taubench.py

# 3. Check your adapter against the contract, with no GPU and no model.
python - <<'PY'
from anchoropt.testing import check_adapter_contract
from adapters.taubench import ADAPTER
r = check_adapter_contract(ADAPTER, states=MY_STATES, events=MY_EVENTS)
print(r.summary()); assert r.ok
PY
```

`examples/toy_host/` is the **reference implementation**, not a sketch. It is a complete adapter plus a
real executing host in ~400 lines, it passes the same contract checks yours must, and
`tests/test_toy_host_e2e.py` runs the whole algorithm against it with real measurement. When this guide
and the toy host disagree, the toy host is right.

---

## 1. Mandatory vs optional, and what each optional hook buys

The checker enforces this distinction; it is not advisory. Missing **mandatory** hooks fail. Missing
**optional** hooks produce a NOTE saying what you lose — a first port is expected to pass with several.

### Mandatory — `optimize_residual` cannot run without these

| hook | what core does with it |
|---|---|
| `is_decision(event)` | filters your trajectory to consequential decisions; localization runs over these |
| `boundary_key(event)` | your own key for the decision point an event belongs to |
| `boundary_from_key(key)` | maps your key to a core `IncisionPoint`. **Core owns no key→locus table** — that would be a stage map imposed from outside |
| `synthesis_fields(boundary)` | the typed alphabet a new condition may be built from **at that boundary** |
| `declared_signals()` | current Φ: conditions your host can already evaluate |
| `signal_boundaries(signal)` | where each signal is observable |
| `evaluate_signal(signal, state, params)` | evaluate one condition on one state |
| `HOST` | a `HostProfile` naming executable actions per boundary — U_H(l) |

### Optional — capability, not correctness

| hook | what you lose without it |
|---|---|
| `states_at(boundary, states)` | **Strongly recommended.** Core fails closed: a predicate cannot be validated against its own boundary's states. This is the hook that stops you selecting a controller that can never fire |
| `install_signal(...)` | **Φ cannot be expanded.** A residual your shipped vocabulary cannot express reports `SIGNAL_BLOCKED` and stops |
| `ground_*` (any) | **No candidate can be built.** Every round ends `ACTION_UNAVAILABLE` |
| `executor_capability(...)` | materializability is unchecked: an admissible action with no executing code behind it is built and measured as if it ran |
| `parameter_domains(signal)` | only DETERMINISTIC policies; no θ_φ grid is searched |
| `normalize_event`, `observable_state`, `label_for`, `depends_on`, `probe_params`, `policy_class_for`, `reset_expanded_signals`, `expanded_signal_names` | convenience and reporting |

---

## 2. Boundary-specific observability — the most common porting bug

`synthesis_fields(boundary)` must be **boundary-truthful**. A field declared where the runtime does not
carry it makes every predicate over it answer `False` — for a structural reason that looks exactly like
the condition not holding.

The three loci, and what exists at each:

| locus | visible | can prevent execution |
|---|---|---|
| `PRE_GENERATION` | context/state. **No call exists yet** | everything |
| `POST_GENERATION_PRE_EXEC` | the proposed call and its arguments | the execution |
| `POST_EXECUTION` | the result or error; the world has moved | nothing — repair only |

```python
_FIELDS_AT = {
    IncisionPoint.POST_GENERATION_PRE_EXEC.value: {
        "payload_chars": _Field("payload_chars", int, (_PG,)),   # the proposed call IS visible
    },
    IncisionPoint.POST_EXECUTION.value: {
        "error_kind": _Field("error_kind", str, (_PE,)),          # a result CANNOT exist before dispatch
    },
}
```

A fact may legitimately appear at both (a step index, a proposed payload carried forward). A **result**
at a pre-dispatch boundary is always a defect. The checker reports the overlap as a NOTE because it
cannot tell which you meant.

`states_at` is the runtime half of the same rule: project each state onto one boundary's information
set. It may **filter, never invent** — the checker enforces that.

---

## 3. Executor capabilities — a declaration is not an executor

This is the part of the contract that changed most in core-v2, and it exists because of a real defect:
an adapter declared a commitment-gate suppression capability naming a remedy flag that appears **zero
times** in the code that withholds the call. The contract validated a name; the runtime ran something
else; both reported success.

Declare, per `(boundary, action)` cell:

```python
ExecutorCapability(
    boundary="post_generation_pre_exec", action="suppress",
    binding="my_host.runtime:withhold_proposed_call",   # the code that RUNS. No binding = GHOST
    consumes=("retry_budget",),                         # eta keys that code actually READS
    signals=(),                                         # () = any; or the conditions it covers
    signal_agnostic=True,                               # may an EXPANDED signal drive it?
    eta_is_computed=True,                               # derives its own params from live state
)
```

Three rules core enforces:

1. **`binding` must name something.** An unbound cell is refused as a ghost.
2. **`consumes` or `eta_is_computed` must be declared.** `consumes` lists eta keys the executing code
   reads. If it derives its parameters from runtime state instead, say `eta_is_computed=True` — that is
   a real and common case (a mechanism that synthesizes a replacement call reads none of the
   candidate's eta), and it must be declared rather than assumed.
3. **An `enforced` claim needs an executor.** `preservation` is the example: a safety clause no executor
   reads is not a clause. `eta_is_computed` does **not** exempt this.

### An action you cannot yet run: disable it, do not delete it

If a cell exists in code but its parameter does not reach the agent, declare it **disabled**:

```python
ExecutorCapability(
    boundary="post_generation_pre_exec", action="reprompt",
    binding="my_host.runtime:record_instruction",
    consumes=(),
    disabled_reason="the instruction is written to telemetry and never injected",
)
```

Core then refuses arms there *with that reason*. Deleting the cell instead makes it look unconsidered,
and the next porter re-adds it. This is exactly the state of BFCL's upstream reprompt cell: two arms
differing only in `instruction` produced byte-identical trajectories across 13/13 episodes.

### Proving your binding — the behavioural probe

Core cannot verify a binding without reading your source, which would couple it to your benchmark. So
prove it by **behaviour**. The reusable harness is
[`tests/test_executor_behavioral_contract.py`](../tests/test_executor_behavioral_contract.py):

```python
BoundaryProbe(
    name="taubench suppress", boundary="post_generation_pre_exec", action="suppress",
    run=lambda controller: run_my_episode(controller),   # controller=None is the control
    controller=my_controller,
    effect_claim="the proposed call is absent from what the host dispatched",
    asserts_effect=lambda control, treated: ...,
)
```

Run it with the controller absent, then installed, and assert the runs differ **in the way the action
claims**. A telemetry-only mechanism fails this — and must, since a firing flag beside an unchanged
trajectory is precisely the inert-action defect.

**"Telemetry says it fired" is not evidence.** Compare observable consequences: what was dispatched,
what the agent saw.

---

## 4. Implementing an intervention

The four action families are closed: `NOOP`, `REPROMPT`, `SUPPRESS`, `REROUTE`. Do not add a fifth —
`REROUTE` deliberately covers both `substitute` (replace the call) and `transform` (reshape what the
model observes), and an earlier split did not survive contact with the code.

`SUPPRESS` has two accepted variants with **different contracts**:

| variant | what the agent observes | `preservation` |
|---|---|---|
| remove outright | nothing in the call's place; it re-plans | not required — the operation never happened |
| withhold and replay | the tool's verbatim recorded result | **required**, and must be executor-backed |

Model this with the grounding's `variant`, not with a new action family. Core reads
`ActionContract.variant_required` to decide which fields a given variant owes.

**Keep the mechanisms distinct.** If a suppression supplies no instruction, your host must not
manufacture one — the agent should re-plan from the *absence* of its own call. The toy host had this bug
(`hint or "Archive it instead"`), which silently made suppression a reprompt and would have made any
comparison between the two families meaningless.

---

## 5. Trajectory format

There is no required schema. Core reads whatever your hooks return, so `normalize_event` is where you
adapt your rows. What core needs:

```python
events = [ADAPTER.normalize_event(r) for r in my_trajectory_rows]   # localization reads these
states = [ADAPTER.observable_state(e) for e in events]              # predicates are evaluated on these
```

Requirements, learned the hard way:

* **Keep the answer-commitment rows.** Dropping them deletes the boundary the backward search needs, and
  localization then derives exactly one boundary.
* **States must carry their boundary.** `states_at` needs to tell a gate state from a post-execution
  one. Tag them, or derive it from the event kind.
* **Firings must come from the run.** Count them where the mechanism executes, not from a registry-
  derived flag: a registry cannot see whether the code actually ran. `train_objective` requires
  `interventions_executed > 0` for an arm to count as engaged, so reporting zeros makes every arm look
  inert and the argmax falls through to a tiebreak.

---

## 6. Evaluation integration

### Inline (preferred, if you can afford it)

Pass a callback that runs a **paired** comparison against the frozen incumbent and returns a
`ThetaResult`:

```python
def evaluate(arm, theta=None):
    got = run_my_corpus(controller_for(arm))
    return ThetaResult(
        theta=dict(theta or {}),
        gains=tuple(c for c in got if got[c] and not baseline[c]),   # CASE IDS, not counts
        losses=tuple(c for c in got if not got[c] and baseline[c]),
        firings=got["interventions_executed"],
        cases_fired=got["cases_fired"], n=got["n"],
        interventions_executed=got["interventions_executed"])
```

`examples/toy_host/demo.py` is a working instance of exactly this.

### External (when evaluation means a cluster job)

If measuring one arm means submitting a job and scoring artifacts later, do **not** fake an inline
evaluator. Use the two-phase seam:

```python
from anchoropt.learning.external_evaluation import (
    ExternalEvaluation, arm_manifest, select_on_measurement)

# Phase 1 -- emit every built arm. Name no winner.
manifest = arm_manifest(arms, incumbent_id="P3", incumbent_token=str(run_dir))

# Phase 2 -- feed the recorded results back through the SAME optimizer.
sel = select_on_measurement(
    arms, runtime=ADAPTER, host=ADAPTER.HOST, incumbent_id="P3",
    evaluation=ExternalEvaluation.from_json("results.json"))
```

**The one rule: an arm you did not evaluate must be OMITTED, not reported as zero.** A zero is a
measurement; absence is not one. A fabricated zero outranks a genuinely measured `-2` arm under
J_train, so an arm nobody ran would be selected over one measured and found harmful.

### The four outcome classes — keep them apart

| class | meaning | negative result? |
|---|---|---|
| `IMPROVED` | a measured arm beat the incumbent | — |
| `UNEVALUATED` | arms built and grounded; **nothing ran** | **No** |
| `NO_BENEFIT` | every arm measured; none helped | **Yes** |
| `BUDGET_EXHAUSTED` | arms left unmeasured; round still open | **No** — requeue |

Collapsing any pair is how a null gets believed. Reporting `NO_BENEFIT` for an unevaluated arm asserts a
measurement that never happened.

---

## 7. What your adapter must never do

The adapter answers questions about **your runtime**. It never answers questions about **the failure**.

If you write "for this kind of failure, use action X", that logic belongs to core and putting it in the
adapter invalidates the experiment. Concretely, never:

* rank or select candidates — measurement does that;
* name a boundary, signal or action as preferred;
* filter candidates by anything but **materializability** (can my host run this?);
* let a benchmark identifier reach `anchoropt/` — `tests/test_core_genericity.py` enforces this.

Selecting by a firing-rate heuristic is the specific version of this that core-v2 removed.

---

## 8. Checklist before you trust a number

```bash
python -m pytest tests/test_adapter_contract_reusable.py   # the checker, tested by violation
python -m pytest tests/test_core_genericity.py             # no benchmark identifier in core
python examples/toy_host/demo.py --no-late-repair          # the algorithm, end to end
```

- [ ] `check_adapter_contract(MY_ADAPTER, states=…, events=…).ok`
- [ ] Every declared capability has a **behavioural probe** that passes
- [ ] An action whose parameter does not reach the agent is **disabled with a reason**
- [ ] Firing counts come from the run, not from a registry flag
- [ ] Arms you did not evaluate are **omitted**, never zero
- [ ] Your `evaluate` compares against **one frozen incumbent** for the whole round
- [ ] `states_at` filters and never invents

---

## 9. Where to look when something is wrong

| symptom | likely cause |
|---|---|
| every round `BOUNDARY_NOT_REPAIRABLE` | `is_decision` does not recognize your rows |
| exactly one boundary localized | you dropped the answer-commitment rows |
| `SIGNAL_BLOCKED` and no expansion | `install_signal` absent, or `synthesis_fields` empty at that boundary |
| controller built, never fires | a field declared at a boundary that does not carry it; or `states_at` missing |
| `ACTION_UNAVAILABLE` everywhere | no `ground_*` hook, or `HOST` declares nothing at that locus |
| `not_materializable_by_host_executor` | read the `detail` — it names the missing field or the conflict |
| arms all look inert in the argmax | you reported `interventions_executed=0`; count firings at the mechanism |
| a round reports `NO_BENEFIT` suspiciously fast | check whether anything was actually measured — it may be `UNEVALUATED` |
