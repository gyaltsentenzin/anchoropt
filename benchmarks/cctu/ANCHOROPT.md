# CCTU × AnchorOpt: the boundary map, `U_H(ℓ)`, and Φ_CCTU

The third runtime integration, after [`bfcl_v4`](../bfcl_v4/) and
[`tb2_deepagents`](../tb2_deepagents/). This document is **step 1**: it fixes the boundary mapping and
declares what each incision point may execute, *before* any adapter code exists — because
[`../../anchoropt/anchor.py`](../../anchoropt/anchor.py) exists to stop the incision point being left
implicit, and a mapping decided while writing the executor is a mapping nobody reviewed.

## Status — wired and measured; no anchor yet

| | state |
|---|---|
| `evaluation.py` — `acc`, `terminal_state`, `detail.jsonl`, fail-closed length check | **done**, and already anchor-aware |
| `cctu_capabilities.py` — the 33-field alphabet | **written** (step 2) |
| `cctu_signals.py` — Φ_CCTU and the three classifiers | **written** (step 2) |
| `cctu_state.py` — live constraint state, via upstream's own handlers | **written** (step 3) |
| `cctu_adapter.py` — the full runtime surface | **written** (step 3); `scripts/check_adapter.py` passes all 7 checks |
| `cctu_middleware.py` — the three hooks, `ControllerSpec`, the trace | **written** (step 4) |
| `cctu_apply.py` — directive → an edit of the turn, with the mechanisms | **written** (step 5) |
| `cctu_replay.py` — replay a frozen transcript, no model | **written** (step 5) |
| `response_generator.py` — the three hooks wired; `--controllers` / `--anchor-trace` / `--replay-from` live | **wired** (step 5) |
| `verify_plumbing.py` — 9 checks, no GPU | **written** (step 6); all 9 pass |
| `tests/test_cctu_adapter.py`, `tests/test_cctu_middleware.py` | **written** (step 7); 57 + 33 tests |
| baseline runs | **measured** — granite train+test and qwen train, under `results/<model>/<split>_baseline/`, each with a `run_manifest.json` |
| the cycle-0 record (residual, variance floor, headroom) | **analysed but NOT yet committed as a script** — the numbers below come from throwaway analysis and must be recomputable before they ground a round spec |
| an anchor | **none** |

`check_adapter.py` passing means the interface is wired and a controller is *realizable*, not that
anything works: it reports `REALIZABLE_UNMEASURED`, which is structural rediscovery with no claim
about benefit.

**As of step 5 the loop runs end to end with no model.** A replayed control and a replayed arm both
complete, are scored by upstream's own scorer, and produce paired `detail.jsonl` files over an
identical denominator:

```
control  exposures=120 firings=0 executed=0        transcripts complete
arm      exposures=124 firings=2 executed=2        one intervention per episode (one_shot)
```

What that does NOT establish is any accuracy claim. The replayed generations cannot respond to an
injected instruction — the model that produced them never saw it — so a replay is a plumbing test and
nothing more. The executor cells stay `driven: False` until a live run fires them.

**Live baselines exist, and no anchor has run.** So every accuracy figure in this document is a
*control* figure, and nothing here is an anchor result. The executor cells stay `driven: False` until a
live arm fires them.

A run's decode settings, corpus hash and policy hash are in its `run_manifest.json`; two manifests that
differ explain a delta the scores cannot.

---

## Why there is no loop to reimplement

The most misread thing about the BFCL integration is that its 3,600-line evaluator is a *BFCL
constraint*, not a method requirement. AnchorOpt needs three callbacks around an existing agent loop;
BFCL exposes none, so the loop had to be rebuilt around its pieces
([`../../COLLABORATORS.md`](../../COLLABORATORS.md)).

**CCTU exposes all three already.** `sample_process` (`response_generator.py:154`) is:

```
174   while not finish:
176       while times < args.max_retries:
183           responses = args.client.chat(messages=messages, tools=tools, **_gen)
188           tmp_message = responses["choices"][0]["message"]
190           finish, feedback, ... = get_feedback(tmp_message, sample, checker, args.use_vllm)
211           messages.append(tmp_message.copy())
213           messages.extend(deepcopy(feedback))
```

Three insertion points, no restructuring:

| ℓ | insertion site | what is observable |
|---|---|---|
| `PRE_GENERATION` | before line 183 | `messages`, `tools`, all live checker state |
| `POST_GENERATION_PRE_EXEC` | between lines 188 and 190 | the proposed `tool_calls` and their arguments; **nothing has run and nothing has been counted** |
| `POST_EXECUTION` | between lines 190 and 211 | tool results, constraint-violation messages, the updated checker state |

This is adapter work, not a reimplementation — which also means the fidelity liability BFCL carries
([`../../docs/FIDELITY_AUDIT.md`](../../docs/FIDELITY_AUDIT.md)) is much smaller here: upstream's
`DialogueConstraintChecker` and upstream's scoring both run unmodified. What we add is interception.

---

## The boundary mapping

### `POST_GENERATION_PRE_EXEC` has two sub-moments, and the difference is load-bearing

This is the one genuinely CCTU-specific structural fact, and it has no analogue in either existing
integration. **CCTU's environment runs its own validator inside the step**, and that validator *mutates
budget state*. Where an intervention stands relative to it decides what is still preventable.

`get_feedback` (`response_generator.py:114`) fuses three separate things:

```
131   if_fb     = checker.get_feedback_if(...)          # CONSTRAINT CHECK -- mutates checker state
140   if_args_fb = checker.get_feedback_tool_arguments(...)  # argument schema check
144   feedback   = get_feedback_tools(if_args_fb, ...)  # ACTUAL TOOL EXECUTION (utils.py:127)
```

The mutations at line 131 are not incidental. `get_feedback_if` calls `update_round()`
(`constraint_checker/core.py:125`), and the handlers it dispatches increment the budgets:

| site | mutation |
|---|---|
| `handlers/interact.py:59` | `checker.callTimes += 1`, **per proposed call** |
| `handlers/interact.py:86` | `checker.accum_max_parallelCallTypes = max(...)` |
| `handlers/tool.py:40` | `checker.callTimesPerTool[name] += 1` |
| `handlers/tool.py:116-118` | `first_tool_name`, `earliest_callTurnPerTool[name]` |

So:

```
POST_GENERATION_PRE_EXEC
    ├── ℓ2a  before the validator   nothing counted, nothing executed   → suppression is FREE
    └── ℓ2b  after the validator    violation verdict known, budget ALREADY SPENT
```

Both are the same incision point — **there is no fourth boundary**, exactly as `after_model` covering
both a proposed tool action and a proposed terminal response is one boundary in TB2
([`../tb2_deepagents/README.md`](../tb2_deepagents/README.md)). They are sub-cases of one point,
distinguished by an observable, not by a new locus.

**The consequence for any `SUPPRESS` implementation: the call must be removed from
`tmp_message["tool_calls"]` at ℓ2a, before line 190.** Suppressing at ℓ2b cancels the execution but
leaves `callTimes` incremented for a call that never ran — strictly worse than doing nothing. This is
CCTU's instance of the efficiency-anchor invariant AnchorOpt already paid for once: *look up before
executing, and execute the shortened batch; fewer proposals is not a saving, only fewer executions is*
([`../../docs/EFFICIENCY_CLASS.md`](../../docs/EFFICIENCY_CLASS.md)). Prove it the same way — with a
counting stub, `call_function` invocations for suppressed calls must be **0**.

### Two things the environment already does, which change what an anchor can contribute

Both were read off the code, not assumed, and both narrow the useful action space *before* any
measurement:

1. **A violating call is already not executed.** `args_checker.check` copies every tool-role
   constraint message through and skips that `call_id` from further checks
   (`constraint_checker/args_checker.py:34-53`); `get_feedback_tools` then sees it in `args_feedback`
   and `continue`s past `call_function` (`utils/utils.py:119-125`). So the harness performs a form of
   suppression already. An anchor's suppression differs in *which* calls it catches and in
   **preserving the budget**, not in withholding execution.

2. **The environment reprompts on every violation.** A non-final turn's violations return as
   `role: "tool"` messages, a final turn's as one merged `role: "user"` message
   (`constraint_checker/core.py:137-149`), and `finish` stays false while any feedback exists
   (`response_generator.py:150`) — so the loop already feeds violations back and lets the model retry.

Point 2 is a prediction, and it is worth writing down before measuring so it can be wrong on the
record: **`REPROMPT` should be weak on CCTU and `SUPPRESS`/`REROUTE` strong.** A reprompt anchor here
has to say something the environment's own feedback channel does not already say, and upstream's own
headline finding is that models "demonstrate limited capacity for self-refinement even after receiving
detailed feedback on constraint violations" ([`README.md`](README.md)). That is the same shape as
AnchorOpt's cross-benchmark observation that **constrained execution has beaten delegated recovery**
([`../../README.md`](../../README.md)) — so if CCTU contradicts it, that is a finding.

### Which constraints are checkable where

Not uniform, and it decides which failures are preventable at all. From the handlers' own
`is_final` guards:

| checked | constraints | earliest observable ℓ |
|---|---|---|
| per proposed call | `MAX CALL TIMES`, `MAX PARALLEL CALLS`, `MAX CALLS PER TOOL`, `TOOL ORDER`, `TOOL PARALLEL`, `TOOL ARGUMENTS` | ℓ2 — **preventable** |
| only at the turn that commits to an answer | `MIN ROUND`, `MIN CALL TIMES`, `MIN PARALLEL CALLS`, `MIN/MAX LENGTH`, `FORMAT`, `PUNCTUATION`, `IDENTIFIERS` | ℓ2, on the terminal-response sub-case |

The second group is the reason the terminal-response sub-case matters: a `MIN`-class or response-class
violation is observable exactly when the model proposes a final answer, and at that moment it is still
changeable — but the rounds it needed to satisfy `MIN ROUND` may already be gone. **Earlier points can
prevent more and know less; that trade is not an abstraction here, it is the corpus's dominant
structure.**

---

## `U_H(ℓ)` — draft, per-cell status

The rule this project is held to: **a cell is declared only where a real anchor with a real payload
has been driven through it.** A cell that passes the structural grid can still do nothing at runtime
while reporting success — that is [`../../docs/CONSUMER_BOUNDARY_RULE.md`](../../docs/CONSUMER_BOUNDARY_RULE.md),
and the TB2 prototype's version of it reported `executed=True` while appending its payload to the
empty string.

**The evidence level for every declared cell is DIRECTIVE-LEVEL, not live.** `apply_action` produces
a complete directive and `cctu_adapter.EXECUTORS` names what will run it; nothing has fired in an
episode, and every executor entry carries `driven: False`. Step 5 flips those.

As shipped in [`cctu_adapter.py`](cctu_adapter.py):

| ℓ | action | status | how it is executed here |
|---|---|---|---|
| `PRE_GENERATION` | `NOOP` | declared | — |
| `PRE_GENERATION` | `REPROMPT` | declared | append a message to `messages` before line 183. No extra turn and **no round charged** |
| `POST_GEN_PRE_EXEC` | `NOOP` | declared | — |
| `POST_GEN_PRE_EXEC` | `REPROMPT` | declared | discard the proposal, inject, re-decide. **No round charged** — the validator has not run |
| `POST_GEN_PRE_EXEC` | `SUPPRESS` | declared | two variants: `drop_call` (**at ℓ2a**, budget preserved) and `withhold_execution` (budget charged) |
| `POST_GEN_PRE_EXEC` | `REROUTE` | declared | one grounding: delete argument names the tool's schema rejects. Pure deletion, nothing synthesized |
| `POST_EXECUTION` | `NOOP` | declared | — |
| `POST_EXECUTION` | `REPROMPT` | declared | inject after the outcome, re-enter the loop. **This costs a round** |
| `POST_EXECUTION` | `REROUTE` (substitute) | **withheld** | needs semantic argument synthesis, and the one mechanical repair is strictly better at ℓ2a — see below |
| `POST_EXECUTION` | `REROUTE` (transform) | **withheld** | rewriting the observation before line 213 is implementable, unlike in TB2 — but not declared until driven |
| `POST_EXECUTION` | `SUPPRESS` | **structurally excluded** | nothing left to cancel; enforced independently by `anchor.exclusion_reason` |

Both withheld cells are *reported*, not silently absent: `ground_transforms` returns `[]`, so
`action_contract.instantiate` emits `TRANSFORM_INFEASIBLE` with the missing requirements named and the
exclusion appears in the search's certificates.

### Why `REROUTE` moved from planned to withheld at ℓ3 — a finding, not caution

Substituting a **different tool** needs that tool's arguments, and on CCTU they are semantic and
per-tool. [`../../docs/GENERALIZABILITY.md`](../../docs/GENERALIZABILITY.md) is explicit: "a reroute is
only deterministic if `COPY`/`RENAME` suffices ... emitting `key=` universally would have built an
invalid call and been misscored as *the substitute did not help*." There is no mechanical mapping from
one CCTU tool's arguments to another's, so that grounding is not available at any boundary.

What *is* mechanical is **argument repair**: `ToolArgsChecker` reports extra arguments by name
(`args_checker.py:90-96`), and deleting exactly those keys synthesizes nothing. `Action.REROUTE` covers
"the same function with different arguments" by its own definition, so this is squarely the family —
and it belongs at ℓ2a, before the budget is charged. Re-dispatching after the validator has already
counted the call is a worse version of the same intervention, which is why ℓ3 gets nothing rather than
a second-best copy.

On the destination not being a literal tool name: the tool is the one the model already chose, which
already resolves. Nothing new is attested, so the failure this project records — admitting a
destination at "11/11 clean" where clean meant *did not error*, 9 of the 11 returning nothing — cannot
arise. The attestation question only exists when the destination changes.

### One open decision, recorded rather than buried

**`PRE_GENERATION` / `REPROMPT`: declare it, or withhold it as BFCL does?**

BFCL declares `NOOP` only at that point, and the exclusion is *measured*: A4 v1 and v2 share a signal,
an action family and byte-identical injected text, differ only in the incision point, and score
**−4.95 pp** versus **+3.63 pp**. Declaring the cell there would let the search re-propose an arm that
project already measured as harmful.

On CCTU nothing has been measured, so withholding the cell would be **importing BFCL's result as an
assumption**. The recommendation is to declare it, carry the A4 note in `HOST.notes`, and let cycle 1
measure it — accepting that one arm may be spent confirming a known-shaped failure. Flipping this to
`withheld` is a legitimate alternative and costs only that arm.

---

## Φ_CCTU — what is pre-registered, and what is deliberately not

The discipline both existing integrations converged on, stated most sharply in
[`../bfcl_v4/bfcl_capabilities.py`](../bfcl_v4/bfcl_capabilities.py):

> The first version of this integration declared nine signals, one per accepted anchor. That makes the
> anchor library the design: a proposer restricted to those names can only ever re-select what is
> already known.

So the adapter declares an **alphabet**, and conditions are expressions over it
([`../../anchoropt/learning/signal_lang.py`](../../anchoropt/learning/signal_lang.py)). Only four
signals are pre-registered, each either structurally forced or handed to us by the environment's own
error contract:

| signal | ℓ | grounding |
|---|---|---|
| `terminal_response_proposed` | ℓ2 | CCTU's own loop-exit test is `is_final = (len(tool_calls) == 0)` (`response_generator.py:129`). Same name as TB2's — same observable, so the two adapters agree |
| `proposed_tool_action` | ℓ2 | the complement, so both sub-cases are first-class |
| `constraint_violation_reported` (param `violation_class`, optional) | ℓ3 | the lexical rung, and here it is *exact*: `INSTRUCTION FOLLOWING ERROR: <CLASS> NOT FOLLOWED!`, whose regex `evaluation.py:26` already carries |
| `tool_execution_error` | ℓ3 | `an error occured when call <tool>` (`utils/utils.py:132-137`) |

`tool_execution_error` is **not** a model failure signal. It covers a `FunctionTimedOut` from the 10-second
`@func_set_timeout` and any exception raised by the episode's own tool code, so it must stay
distinguishable from a constraint violation and from a real result.
[`../../anchoropt/attribution/harness_guard.py`](../../anchoropt/attribution/harness_guard.py) exists
because an implementation crash on the other benchmark got mined as a real error contract at support 13
before anyone noticed. **Use it on every trajectory read.**

The 14 violation classes, closed, from the handlers themselves:

```
interact.py   MIN ROUND · MIN CALL TIMES · MAX CALL TIMES · MIN PARALLEL CALLS · MAX PARALLEL CALLS
tool.py       MAX CALLS PER TOOL · TOOL ORDER · TOOL PARALLEL
response.py   MIN LENGTH · MAX LENGTH · FORMAT · PUNCTUATION · IDENTIFIERS
args_checker  TOOL ARGUMENTS
```

### The field alphabet is where the work goes

Everything interesting is a field for `synthesis_fields()` to expand over, not a pre-written
condition. CCTU suits this unusually well: the checker's state is fully readable and deterministic, so
structural predicates cost nothing to evaluate. **33 fields**, declared in
[`cctu_capabilities.py`](cctu_capabilities.py) — 7 readable before generation, 25 at the commitment
gate, all 33 after execution. The structure mirrors `HANDLER_REGISTRY` one handler at a time:

| group | n | what |
|---|---:|---|
| the decision just made | 7 | `proposes_tool_call`, `n_tool_calls`, `n_distinct_tool_names` (both parallel units), `tool_name`, `content_length`, `repeated_identical_call`, `has_generation` |
| constraint pressure, per turn | 6 | `call_times_over_cap`, `per_tool_over_cap`, `parallel_over_cap`, `order_prereq_unmet`, `parallel_group_incomplete`, `args_invalid` — **preventable at ℓ2** |
| MIN-class and response-class | 5 | `min_round_unmet`, `min_call_times_unmet`, `parallel_requirement_unmet`, `response_length_over_cap`, `response_length_under_min` — judged only where the turn is terminal |
| what came back | 8 | `violation_class` + four dimension booleans, `n_violations_this_turn`, `result_is_error`, `result_is_empty` |
| turn context and carried history | 7 | `round_index`, `rounds_remaining`, `call_times`, `call_times_remaining`, `calls_remaining_this_tool`, `last_violation_class`, `consecutive_violation_turns` |

**Carried fields are the reason no signal needs to straddle two boundaries.** A condition that spans
time reads a summary the wrapper maintains and is still evaluated at one point, so `U_H(ℓ)` keeps
meaning what it says.

### Four things step 2 settled, three of them by measurement

**1. The derived booleans are not optional, and not hand-engineering.** The grammar is
`phi ::= x | not x | x == c | x < θ | x > θ | phi₁ ∧ phi₂`, and every leaf compares **one field to a
constant**. So "this turn proposes more calls than the budget still allows" is *unreachable* from
`n_tool_calls` and `call_times_remaining` however the search composes them — no leaf compares two
fields. It has to be a field or it cannot be expressed. Both forms are declared anyway, raw and
derived, which makes the concern checkable: **if cycle 1 only ever selects the derived booleans and
never a threshold on a raw scalar, that is evidence the adapter did the work**, and it should be
reported that way rather than found later by a reader.

**2. The pressure fields need no mirror — they run upstream's own handlers on a copied state.** The
first draft of this called them a second implementation of upstream's rules and required a parity test
each. Step 3 found a better answer: every constraint handler reads and writes the checker through
**plain attributes** and calls no method on it, so a duck-typed stand-in built from a snapshot can be
handed to upstream's real `check()` with a throwaway `Feedback`. The handler reports exactly what it
would report, the stand-in absorbs every mutation, and the live checker is untouched — verified on a
real `DialogueConstraintChecker`, whose `callTimes` and `round` were still 0 afterwards.

> the predicate **is** upstream's logic, applied to a copy of the state

Five parity risks removed rather than managed, and the awkward parts come along for free — including
`ToolOrderHandler`'s corpus-specific override for `query_id == 96` (`tool.py:77-88`), where a
hand-written mirror would have disagreed on exactly one episode. `args_invalid` needs no stand-in at
all: `ToolArgsChecker.check` is already pure, so it is called directly and inherits
`schema_validate.validate_param_value`'s nested cases.

The three response validators stay excluded, by an explicit allowlist rather than by omission
(`cctu_state.PREDICTIVE_HANDLERS`), so a handler added upstream does not silently start executing
per-episode Python here.

**3. `inf` must never reach a declared numeric field — verified, not argued.** `to_int(None)` returns
`math.inf`, so an uncapped episode has `max_callTimes == inf`. Driving `signal_grammar.atoms_for` with
an `inf` in the observed values produces the atom pair `x < inf` (fires on everything finite) and
`x > inf` (never fires), and shifts every other quantile on that field. Replacing the `inf` with
`None` removes both atoms and changes nothing else. So `call_times_remaining` and
`calls_remaining_this_tool` are **None when unconstrained**, and `_numeric_values` skips them — the
field contributes no observation rather than a poisoned one.

**4. Only 4 of the 14 violation classes are reachable by synthesis.** `atoms_for` emits `equals` atoms
for the first `MAX_ATOMS_PER_FIELD` (= 4) enum values only. So `VIOLATION_CLASSES`' *ordering* decides
what the search can name, and `synthesis_reachable_classes()` exposes the consequence rather than
leaving a reader to conclude the search ignored ten conditions. The four chosen are the four
preventable at the commitment gate from checker state alone. **This ordering is a priori and should be
re-derived from cycle 0's measured distribution** — legitimate, being the same discipline as deriving
thresholds from observed values, and simply not possible before a residual exists. The four dimension
booleans exist so the search is not hostage to the choice in the meantime.

### Three things step 3 settled

**1. A trajectory must carry TWO events per turn, or `WHERE` is not a searchable coordinate.** One
iteration of `sample_process` produces one record holding the generation, the validator's verdict and
the tool results together. Fed to `normalize_event` as a single event it carries feedback, so
`boundary_key` places it at `POST_EXECUTION` — and an episode of such turns derives exactly **one**
boundary. `check_adapter.py` check 3 warns on precisely that, and the template calls it the most
valuable lesson from the first port: a residual with one boundary can never be observed to move
earlier. `raw_rows_from_messages` therefore expands a recorded episode into a commitment-gate row and
a post-execution row per turn. Measured: 1 boundary before, **2 boundaries and `moves_earlier=2`**
after. The `POST_EXECUTION` row is emitted even for a clean final turn — skipping it would make "the
validator ran and reported nothing" invisible, leaving the successes with no observation to contrast
the failures against.

**2. A `REPROMPT` costs a round at ℓ3 and nothing at ℓ2.** At the commitment gate the proposal can be
discarded and re-decided *before* `get_feedback_if` runs, so `checker.round` never advances. After
execution the validator has already advanced it, and the extra turn advances it again. For an episode
whose failure **is** round exhaustion — the `mid_tool_call` class — an ℓ3 reprompt makes the failure
worse. Both facts are in the executor's `fixed` params (`charges_round`) and in the directive
`apply_action` returns, so a candidate cannot be proposed for that residual without the cost being
visible to `exposure.displacement_check`.

**3. `SUPPRESS` has two variants that charge different things, and one has a precondition.**
`drop_call` removes the call before the validator counts it — the only form that helps a budget
failure. But if the turn proposed exactly one call, dropping it makes `len(tool_calls) == 0`, the
harness reads the turn as **final**, and the terminal handlers judge content written for a tool turn.
So `apply_action` **refuses** that combination rather than leaving the executor to remember it.
`withhold_execution` keeps the call, so the turn's kind and the budget are unchanged and only the
execution is saved — which is the efficiency shape from
[`../../docs/EFFICIENCY_CLASS.md`](../../docs/EFFICIENCY_CLASS.md), judged on work removed with 0 pp
accuracy as a **pass**. Both are grounded as separate arms because they optimize different objectives;
picking one here would be the adapter deciding an empirical question.

### Four things step 4 settled

**1. CCTU's runner is multi-threaded, and that has no precedent in the other two integrations.**
`response_generator.py` uses a `ThreadPoolExecutor` defaulting to `max_workers=4`, so four episodes
interleave in one process. Per-episode bookkeeping — the turn index, the calls already proposed, the
violation streak, the one-shot guard — therefore **cannot** live on the middleware: four episodes
would share one counter and every one of those fields would be silently wrong in a way no
single-episode test can reveal. So `AnchorOptMiddleware` holds only the specs and the aggregate
telemetry, and `episode(case_id)` hands out an `EpisodeHooks` that owns its own state. TB2 met the
weaker version of this — "nothing guarantees a fresh instance per episode" — and keyed its guard on
episode identity; here identity-keying alone is not enough, the state has to be separate objects.

Verified: **40 episodes × 6 turns on 4 threads**, every episode counting its own 6 turns and its own
12 calls, the one-shot guard holding once per episode (40 executed, 200 loop-prevented of 240
exposures), 240 trace rows with no torn lines and the counter agreeing exactly.

**2. The trace file needs a lock, and the reason is not tidiness.** Four threads appending to one
JSONL is how a trace loses lines, and a trace with missing lines *understates firings* — which is the
number that gates reading any accuracy at all. Appends and the aggregate counters go through one lock.

**3. A synthesized condition can now be persisted, which TB2's spec format cannot do.** TB2's
`ControllerSpec` names a declared signal, so a condition `optimize_residual` synthesizes this session
installs fine and then cannot be saved: a fresh process finds the name undeclared and refuses. CCTU's
spec accepts `predicate` as an alternative — an expression tree over the declared fields, compiled by
[`../../anchoropt/learning/signal_lang.py`](../../anchoropt/learning/signal_lang.py). Verified
round-trip: written to disk, reloaded in a fresh spec, installed, and fired.

Using the shared grammar rather than a local one is the point. `scripts/install_controller.py` carries
a smaller private grammar for the same job; `signal_lang` validates that every field is declared and
typed, forbids `eval` and code strings, and **derives** the boundary set as the intersection of the
referenced fields' boundaries.

**4. On this alphabet the boundary guard bites at ℓ1 and ℓ2, never through an empty intersection.**
`compile_signal` rejects an expression whose fields never coexist at any one decision point. Measured:
the boundaries common to *every* CCTU field are `{post_execution}` — every field is readable there — so
that rejection can never fire on this benchmark. What does bite is the narrower check: a predicate over
proposal-only fields is observable at `{post_generation_pre_exec, post_execution}` and is **refused**
when a spec tries to install it at `pre_generation`. Worth knowing, because a guard that cannot fire
looks the same as a guard that passed.

### Two smaller properties, both verified rather than assumed

**The control arm traces its declines.** With nothing installed, `runtime_hook.decide` returns the
host's own decision unchanged, every hook returns None — and the trace still records a row per
evaluation. That gives the exposure *denominator*, without which a firing count means nothing.

**A live precondition declines and records rather than crashing.** `SUPPRESS`/`drop_call` on a turn
proposing only one call raises inside `apply_action`; the middleware catches it, writes a trace row
with the reason, and returns None. The episode continues. A predictive or intervention path must never
be able to kill an episode — but it must never look clean because it crashed either, which is why the
reason lands in `telemetry["errors"]` rather than being swallowed.

### Three things step 5 settled, and two more upstream defects

**1. The two mechanisms with proof obligations both discharge them, verified with stubs.**

`withhold_execution` is judged against the four invariants in
[`../../docs/EFFICIENCY_CLASS.md`](../../docs/EFFICIENCY_CLASS.md), and each was checked rather than
argued. With a counting stub over `call_function`: a withheld call costs **0 executions** (the
shortened batch is what reaches the executor, not the full one); the replayed string is the tool's
**verbatim** recorded output; the merged feedback is rebuilt in the proposed calls' own order; and with
nothing recorded for an identical earlier call it **declines** rather than authoring a replacement.
That last one matters — the predecessor of this mechanism on the other benchmark substituted a *new*
message, gained +0.33 pp, and was rejected because destructive calls rose 84 → 90.

`drop_call` follows `constraint_repair.py`'s shape — detect, read live state, propose **or decline**,
verify the invariant. It removes trailing calls (principled, not arbitrary: `CallTimesHandler` and
`MaxCallsPerToolHandler` count in order, so the calls in violation are exactly the trailing ones), and
after each removal it **re-asks upstream's own handlers** whether the targeted class still fires.
Measured on a real episode with the per-tool cap set to 1: 4 calls → 1, `max_calls_per_tool` verified
clear. With the cap at 0 it is unrepairable and the mechanism **declines** rather than spending calls
on a guess.

**2. A `REPROMPT` at ℓ2 really does charge no round.** End to end, the arm shows 124 boundary
exposures against the control's 120 — the discarded generation and its re-decision add exposures — while
the round budget is untouched, because the discard happens before `validate_turn`. That is the
difference the two POST_GENERATION sub-moments exist to express, now observed rather than reasoned
about.

**3. `ReplayExhausted` is not retryable.** A transcript that does not cover the trajectory is
deterministic, so the original retry loop would have burned 15 attempts per episode learning nothing.
It is now recorded and the episode stops. This is `tb2_health.py`'s distinction applied to the replay
path: an infrastructure failure should be retried, a task failure never should, and this is neither —
it is a statement about the arm.

**Two more upstream defects, making four.** Both are in `response_generator.py`, which is already
locally modified, so fixing them is in scope; the validator and the scorer remain untouched.

| # | defect | consequence |
|---|---|---|
| 3 | `tool_calls_from_text` was set **unconditionally** on the `use_vllm` branch, whether or not extraction found any calls | under `--use-vllm` a plain final answer was flagged as text-format, so `sample_process` **discarded its feedback**. A terminal violation was computed, never shown to the model, and `finish` stayed False — the episode re-answered identically until the round budget ran out. Response-class constraints are 91–103 episodes each, so this reached most of the corpus, and the symptom is indistinguishable from a weak model |
| 4 | `times` was **never incremented** in the retry loop | a persistently failing episode spins **forever** rather than being dropped. A dead endpoint hangs the run instead of failing it |

Defect 3 is the more dangerous of the two, because it produces plausible-looking output. Both are
fixed: the flag now means what its name says, and the retry loop counts, reports the episode and the
last error, and stops.

### The closed violation set is verified against upstream's source

All 14 classes, extracted from the handlers' own message literals and canonicalized mechanically
(lowercase, spaces to underscores — no lookup table, so there is nothing to maintain and no unseen
wording to fall through):

```
resource    min_round · min_call_times · max_call_times · max_calls_per_tool
behavior    min_parallel_calls · max_parallel_calls · tool_order · tool_parallel
response    min_length · max_length · format · punctuation · identifiers
arguments   tool_arguments
```

**14 templates in upstream's source, 14 classified, 0 undeclared, 0 declared-but-never-emitted.** The
regex is byte-identical to `evaluation.py:26`'s, which is what decides SR and PSR — one owner plus a
parity test, following the precedent that moved five copies of one BFCL regex into one place. An
undeclared class is still returned *as a violation* and recorded in `UNDECLARED_CLASSES_SEEN` so a
test fails on upstream drift; that is deliberately better than the other benchmark's
None-and-vanish behaviour, which its own documentation flags as a weakness.

### Vacuity delegates to the shared detector, and extends it only where it structurally cannot see

`result_vacuity_kind` calls `policy_tree.vacuous_result_kind` — the existing definition, not a copy,
because two definitions of "vacuous" that can disagree is how this project has been bitten before. It
adds one kind, `empty_payload`, for the two shapes the shared detector cannot reach:
`_iter_result_collections` returns immediately unless the payload is a JSON **object** with
list-valued fields, so a top-level empty array, an empty string, or the literal `"None"` is invisible
to it. CCTU's tools produce all three, because `call_function` `json.dumps` a dict or list and
`str()`s everything else (`utils/utils.py:104-107`).

**Lexical vacuity is deliberately absent.** A payload reading "no results found" is very likely
vacuous and this module does not say so, because the wording would be invented rather than observed.
Those cues come from reading real payloads in cycle 0 — the same reason the other benchmark's whole
vacuity class "was invisible until a human read trajectories by hand".

### The harness-fault classifier, and upstream's spelling

`is_tool_fault` matches `an error occured when call <tool>: <exc>` — **one `r`, which is upstream's
spelling** (`utils/utils.py:134`). Matching the corrected spelling would make the classifier return
False for every fault, and a detector that never fires is indistinguishable from a benchmark with no
faults: the class would be invisible to mining rather than merely unhandled. Do not fix it here.

It covers both a `FunctionTimedOut` from the 10-second `func_set_timeout` and any exception from the
episode's own tool source under `exec`, and neither is a model failure. The signal
`tool_execution_error` exists **so the class can be excluded**, not optimized — and because leaving it
unnamed is exactly how it gets mined as a real error contract, which happened on the other benchmark
at support 13.

### The label rule

`unsolved_set` and `answer` are the **scoring labels**. They must not be reachable from
`observable_state`, and the adapter's test file must assert it. A softer version of this leak has
already cost this project a claim: importing the anchor fixtures put the answer key into every process
that merely touched the BFCL runtime, and the smoke run's leak detector caught it
([`../bfcl_v4/bfcl_signals.py`](../bfcl_v4/bfcl_signals.py)). A discovery made with the answer key in
the room is a lookup, however unused it looks.

Vacuity — a call that succeeded and returned nothing useful — must therefore be defined **without**
reference to the answer: an empty payload, an empty collection, a "no results" shape. Answer
containment is `solve_rate_is_one`'s job and stays there.

---

## The corpus, measured

Computed from the shipped `data/`, not quoted:

| | |
|---|---|
| episodes | **200**, balanced 50 per `data_source` (`Single-Hop`, `Multi-Hop`, `Parallel Single-Hop`, `Parallel Multi-Hop`) |
| executable constraint instances | **914** — 4.57 per episode |
| `(dimension, category)` pairs present | **10**, every one with a registered handler; every registered handler used |
| validator directories | **200**, one per `query_id`, under `data/check_code/` |
| train / test | **140 / 60**, balanced 35 / 15 per category, **0 overlap** |

Constraint instances by category, with the handler that enforces each:

| (dimension, category) | n | handler |
|---|---:|---|
| resource / interaction rounds | 200 | `RoundHandler` |
| resource / specific tool call count | 106 | `MaxCallsPerToolHandler` |
| response / format | 103 | `ResponseFormatHandler` |
| response / content_1 | 102 | `ResponseIdentifiersHandler` |
| response / content_2 | 95 | `ResponsePunctuationHandler` |
| response / length | 91 | `ResponseLengthHandler` |
| resource / tool call count | 87 | `CallTimesHandler` |
| behavior / sequential dependencies | 51 | `ToolOrderHandler` |
| behavior / parallel calls count | 41 | `ParallelCallsHandler` |
| behavior / parallel dependencies | 38 | `ToolParallelHandler` |

**`core.py:99-102` silently skips a constraint with no registered handler** — it prints
`cannot registry <p>+<s>` and continues, so an unhandled constraint is instructed in the prompt and
never enforced. Measured: **that branch never fires on this corpus.** Recorded as verified rather than
assumed, because the failure mode would be invisible — an anchor could be credited for a constraint
nothing was checking.

Two discrepancies worth resolving before any number is published, neither of them a bug claim:

- **4.57 executable constraints per episode against the paper's stated average of seven constraint
  types.** So either ~2.4 per episode are instructed in the prompt but not machine-checked, or the two
  figures count different things. Whichever it is, only the checked ones can carry a signal, and only
  the checked ones can lower PSR.
- **The `toolset` dimension is absent from `constraints_list`** although upstream's taxonomy names it
  as one of four. Presumably it is enforced by which tools the episode offers at all, rather than by a
  handler — but it means the toolset dimension is not directly observable to Φ.

### The split is positional, not randomized

`split_train_test.py` takes the **first 35** episodes of each category for train and the next 15 for
test. Deterministic, which is good; unrandomized, which means any ordering in the corpus — construction
batch, difficulty, authoring date — becomes a train/test difference rather than noise. Check for it
before reading a test-split delta as generalization.

And the power is small. Test is **60 episodes**, per-category cells are **15**. BFCL's dev is 84 and it
still states that a cell that size cannot reach significance and is descriptive only. The same applies
here, harder. Per-category CCTU cells are for locating where the work went, never for a claim.

---

## The objective: three failure classes, not one

CCTU has three distinguishable ways to fail, and an anchor that fixes one can break another. They need
separate residuals.

| class | detection | what could act on it |
|---|---|---|
| answer not obtained | `acc == 0` | `REROUTE` / `REPROMPT` on the tool path |
| answer obtained, constraint violated | `acc == 1`, `SR`/`PSR == 0` | `SUPPRESS` / `REROUTE` at ℓ2 |
| round budget exhausted | **two flavours**, see below | the resource-pressure class |

**The budget class has two terminal states, and keying on only one of them understates it.** An
earlier version of this table said `terminal_state == "mid_tool_call"`. That is the flavour where the
budget runs out on a **tool** turn: the last message is a tool result, and `compute_if_flags`
short-circuits to `(1, 1)` at `evaluation.py:71`, forcing both SR and PSR to 0 before any other check
runs. The other flavour is `ended_on_user` — the budget runs out on a **final-answer** turn, with the
validator's feedback as the last message. Measured: granite is dominated by the first (195/280) and
qwen by the second (147/280), so a decomposition keying on `mid_tool_call` alone filed most of qwen's
budget failures into classes 1 and 2.

`judge()` already records why the ranking metric cannot be PSR: **PSR = `acc` ∧ no-error-anywhere, so
ranking loci by PSR-failure is circular** — every locus scores precision 1.00 by construction. So rank
candidate loci by linked downstream loss against `acc` and against the violation class *separately*,
and report SR/PSR.

The third class is why `terminal_state` was added: it is not derivable from SR/PSR, and an aggregate of
those two hides it completely.

**Read [`../../docs/SATURATION_AND_SELECTION.md`](../../docs/SATURATION_AND_SELECTION.md) before
building anything for class 2 or 3.** An anchor that suppresses an over-budget call is a
resource-shaped intervention that may turn out to be a *selection* intervention — which is exactly what
deferred A6: it engaged precisely where designed and made things worse there. The two diagnostics that
catch it, `displacement_check` and `exposure_weighted_delta`, already ship in
[`../../anchoropt/learning/exposure.py`](../../anchoropt/learning/exposure.py).

---

## What does not port

**`anchoropt/mechanisms/` does not apply to CCTU.** `StoreAdapter`
([`../../anchoropt/mechanisms/constraint_repair.py`](../../anchoropt/mechanisms/constraint_repair.py))
is a *memory-store* seam — error phrasings, per-container caps, write verbs, an id keyword. CCTU has no
store. Nobody should try to carry A1, A5, A7, A8 or A9 across, and the most expensive class of bug on
the BFCL line came from exactly that kind of transfer: A5's helpers are one backend's shape, and a
retry that synthesized the wrong call failed on *every* firing of another *after* the eviction had
already succeeded.

What does carry is the **decision principle**:

> When under resource pressure, preserve information before resorting to destructive recovery.

CCTU's resource dimension is real — rounds, total call budget, per-tool budget — so the ladder has a
genuine instantiation:

| layer | BFCL | CCTU |
|---|---|---|
| 1 preserve | remove a provably redundant store entry | drop a provably redundant call |
| 2 consolidate | combine entries losslessly | batch calls into one parallel turn |
| 3 fall back | clear, having exhausted the alternatives | answer with what has been retrieved |

That is a hypothesis about this substrate, not a result. It is written here so that its failure is
visible rather than quietly dropped.

---

## The `acc` harm term: a standing rule, because two rounds got it wrong differently

A round targeting a violation COUNT needs an `acc` harm term, and the two qwen rounds specified it two
ways and neither is reusable. Written down here so a third round inherits a rule instead of inventing one.

**What was tried.**

| round | rule | outcome |
|---|---|---|
| `CCTU_QWEN1` | any **net** loss rejects | accepted arm 258 at 4 gains / 2 losses, net +2. Rejected 261 at 0g / 2l |
| `CCTU_QWEN2` | **zero** losses reject | would reject 258, the arm already installed |

The first is too loose in a way that matters: it lets an arm destroy two correct answers as long as it
happens to create three elsewhere, and a gain and a loss on different episodes are not interchangeable —
the loss is an answer the agent HAD and the anchor took. The second is too tight to ever accept anything,
which round 2 demonstrated by writing a bar its own incumbent fails.

**The rule for the next round, chosen before any arm of it runs and recorded here so it cannot be chosen
after:**

> An arm is rejected on `acc` if its losses exceed the INCUMBENT's losses on the same split, or if it has
> any loss at all when no anchor is installed for that residual. Report gains and losses separately, never
> only the net. An arm with equal losses and greater benefit is admissible and must say so explicitly,
> because it is trading the same accuracy cost for more repair — which is a defensible trade and not a
> free one.

Three properties this has and neither earlier rule had: it is satisfiable, it cannot be satisfied by
offsetting a destroyed answer with an unrelated new one, and it is stated relative to a measured baseline
rather than to zero. The incumbent's figure is part of the round's inputs, so it goes in FROZEN.md beside
the bar.

**`acc` gains and losses are per-episode flips against the paired control** (`analyze_residual --compare`
prints `Ng / Nl  net=`), and the qwen flip floor is 0 within-job and 0 between-job at `max_workers=1`, so a
single loss is attributable rather than noise. That floor is what makes a losses-based rule meaningful at
all; on a configuration with a nonzero flip floor the rule has to widen by the floor and say so.

## The benefit bar: state it relative to the incumbent ON EACH SPLIT

The same failure as the `acc` rule above, one term over, caught one round later. Rounds 2 through 5 set the
benefit bar relative to the installed anchor **on train** (`> 118`) and left `test` as "same direction and
sign". `rounds/CCTU_QWEN5` then passed that rule while being **marginally worse than the anchor it replaced
on the held-out split** — −188 against −118 on train, −50 against −56 on test. The rule as written could not
tell those apart, because on `test` it never compared the two at all.

**The rule for the next round, recorded before that round's arms exist:**

> Benefit is stated relative to the incumbent **on every split it is judged on**. An arm must beat the
> installed anchor's reduction on train AND not fall below it on the held-out split. "Same direction and
> sign" is a floor for a residual with no incumbent, not for one that has an anchor already deployed —
> against an incumbent, replication means beating what is already there where it was held out.

Two notes on applying it. The splits have different bases (786 train / 538 test on `max_length`), so compare
reductions as a **share of each base**, not as raw counts — 6 events on 60 test episodes is not the same
quantity as 6 on 140. And when an arm beats the incumbent on train and ties or loses on test, that is a
result to report as such rather than a pass or a fail: it is evidence the gain is split-specific, which is
exactly what a held-out split exists to reveal.

This was **not** applied retroactively to `rounds/CCTU_QWEN5`, whose acceptance stands under the bar it was
frozen with, with the limitation disclosed in its `RESULT.md`.

## Known limits and open items

Four things to fix or check as the integration is built. The first two are upstream defects that will
otherwise be blamed on the anchors.

1. **~~The retry loop swallows every exception.~~ FIXED in step 5.** It was
   `except Exception as e: err = str(e)` — no `break`, no `raise`, `err` never read, and `times`
   never incremented, so a persistently failing episode spun forever. It now counts attempts, names
   the episode and the last error, and stops. The two guards TB2 already had are in place: the
   middleware never kills an episode on a hook failure
   ([`../tb2_deepagents/tb2_middleware.py`](../tb2_deepagents/tb2_middleware.py)), and an
   infrastructure failure stays distinguishable from a task failure
   ([`../tb2_deepagents/tb2_health.py`](../tb2_deepagents/tb2_health.py)) — conflating the two is how
   two uninformative trials sat in a paired denominator as though the agent had tried and failed.

2. **~~The text-tool-call path never executes tools.~~ NOW FAILS LOUDLY (step 5).** This harness does
   not execute a text-extracted call, and `solve_rate_is_one` reads only `role: "tool"` messages, so
   **`acc` is structurally 0 for every such episode** — a whole run unscoreable in a way that looks
   exactly like a weak model. `sample_process` now raises with the fix named (vLLM needs
   `--tool-call-parser` / `--enable-auto-tool-choice`), and `--allow-text-tool-calls` is the explicit
   opt-out for recording a run as unscoreable on purpose. **Still verify the endpoint returns
   structured `tool_calls` before cycle 0** — the guard turns a silent failure into a loud one, it does
   not make the endpoint correct.

3. **A tool name the episode does not offer CRASHES the validator, and the episode is dropped rather
   than scored.** `handlers/tool.py:40` does `checker.callTimesPerTool[name] += 1` on a plain dict
   built from the episode's tool list, so a hallucinated name raises `KeyError` inside
   `get_feedback_if`. That propagates out of `get_feedback`, `sample_process` retries it, and the
   episode leaves the **denominator** — which is worse than a wrong score, because a missing episode
   is invisible. `evaluation.py` fails closed on the resulting size mismatch, so a run aborts at
   scoring rather than mis-scoring, and `[drop]` plus `anchoropt_errors` name the episode.

   **Measured: latent, not active.** Across 140 granite train episodes there were **0** hallucinated
   tool calls and 0 recorded errors — the tool list travels with the request, so the model stays
   inside it. Left in the validator rather than patched, per the don't-edit-upstream rule, and pinned
   by `tests/test_cctu_adapter.py`, which also asserts the *predictive* path does not inherit the
   crash: `would_violate` records the failure and answers "nothing predicted" instead of raising. A
   guard that dies on a hallucinated tool name would take the episode with it.

4. **The variance floor is not zero by default, and `@func_set_timeout(10)` is the likely reason.**
   Tool execution is wall-clock bounded (`utils/utils.py:95`) and the runner is a
   `ThreadPoolExecutor(max_workers=4)`, so a loaded host can time out a call that succeeded on the
   previous run. AnchorOpt's harm tolerances assume a floor near zero — BFCL measures 0 flips at
   `temperature=0.001`, which is what licenses treating a one-case loss as real. **Measure the floor
   first**: run the identical policy twice and count outcome flips. If it is not near zero, every
   tolerance widens and small deltas are not interpretable at all.

5. **Sharding is sound here, and should be asserted rather than assumed.** Each `sample_process` builds
   its own `DialogueConstraintChecker` from its own sample and reads nothing else, so no case dependency
   crosses an episode and CCTU has no prerequisite phase at all — the hardest constraint on BFCL's
   sharding simply does not exist. Shard by `data_source`, four ways, submitted concurrently; that also
   yields the per-category breakdown for free. Assert the independence at shard time, because if it
   ever changes a shard would build from partial state and be **silently wrong rather than failing**.

6. **`min_length`, `min_call_times` and `repeated_identical_calls` moved materially in an accepted
   arm (`rounds/CCTU_QWEN1`) and none was a watched harm term.** The accepted arm overcorrected into
   `min_length` violations from a base of zero, added to `min_call_times` against a small base, and
   roughly tripled `repeated_identical_calls` — an anchor that removes one violation by creating
   another kind is not obviously a repair. `min_length` and `min_call_times` need no code change:
   they already flow through `evaluation.judge()`'s `violations` dict into `analyze_residual.compare()`'s
   `counts` block, keyed by whatever classes are present. `repeated_identical_calls` now gets the
   same paired-over-`shared` treatment (`compare()`'s own block, next to `counts`/`work`), rather than
   the single-run aggregate `_mine()` always reported. **Every future round's `FROZEN.md` should name
   a tolerance for all three before its first arm runs** — a round that watches them is what would have
   caught this one.

   **RESOLVED, and it earned its place immediately.** Rounds 2 onward name tolerances for all three.
   `repeated_identical_calls > 200` is the only harm term that has ever rejected an arm on its own:
   `rounds/CCTU_QWEN4` produced the largest `max_length` reduction ever measured here (−326) and was
   rejected at **+409**. The anchor now installed keeps `min_length` at +0 on both splits and
   `repeated_identical_calls` at +23 / +8, against the arm it replaced at +6 and +214.

## The installed anchor

| | |
|---|---|
| spec | [`../../rounds/CCTU_QWEN5/controller.json`](../../rounds/CCTU_QWEN5/controller.json) |
| φ | `response_length_over_cap` partitioned on `call_times`, two controllers at `post_generation_pre_exec` |
| θ | `call_tool_first` when `call_times < 1`; round 1's `state_remaining_budget` when `call_times >= 1` |
| **required runtime** | **`one_shot=True`**, `max_workers=1`, `redecide_budget=1` |
| accepted by | [`../../rounds/CCTU_QWEN5/RESULT.md`](../../rounds/CCTU_QWEN5/RESULT.md), FROZEN `fba003d9…` |
| train | `max_length` **−188**, `acc` 0 gains / **2 losses** |
| test | `max_length` **−50**, `acc` 0 gains / 0 losses |
| replaced | `cctu_c1_258_reprompt_state_remaining_budget` (−118 train / −56 test) |

**`one_shot=True` is part of the policy, not a default.** The same specs at `one_shot=False` are
`rounds/CCTU_QWEN4`, rejected on `repeated_identical_calls` at +409. `controller.json` records this under a
`requires` key, which `ControllerSpec.load` ignores because it keys off `"controllers"`.

**Known limitation, carried from the acceptance:** this anchor is better than the one it replaced on train
(−188 vs −118) and **marginally worse on the held-out split** (−50 vs −56). It passed the bar it was frozen
with, which compared against the incumbent on train only; see the benefit-bar rule above, which is the fix
and was not applied retroactively.

## How this is run

```bash
python verify_plumbing.py                       # ~15s, no model: proves the control is inert

# the control. Omitting --controllers IS the control, by construction.
python run_hosted.py --model <m> --evaluate --split train --anchor-trace

# the variance floor: two replicates in ONE run, which evaluation.py aggregates
python run_hosted.py --model <m> --evaluate --split train --anchor-trace --repeat 2 --max-workers 1

# one arm: a persisted anchoropt.controller.v1 spec
python run_hosted.py --model <m> --evaluate --split train --anchor-trace \
    --controllers rounds/<round>/controller.json
```

### Where the artifacts go, and what identifies a run

One owner, [`cctu_paths.py`](cctu_paths.py), consulted by `response_generator.py`, `run_hosted.py`,
`run_vllm.py` and `analyze_response.py` alike:

```
results/<model>/<split>_<config>/
    response.jsonl  analysis.jsonl  scores.json  detail.jsonl
    anchor_trace.jsonl  anchor_telemetry.json  run_manifest.json
```

`<model>` collapses every spelling of one model into one directory — `granite-4-1-8b`,
`granite-4.1-8b` and `ibm-granite/granite-4.1-8b` all resolve to `granite` — because
`analyze_response.py` already recorded the reason: "runs are kept under a per-MODEL root so two
models' outputs cannot be differenced by accident; anchors are mined from one model's residual and do
not transfer as artifacts." An unknown model falls back to its slugified id rather than raising, which
is safe here in a way it is not for a case id: the slug is still injective, so isolation holds, and a
wrong guess costs an ugly directory name rather than a corrupted aggregate.

`<config>` is `baseline` or the controller spec's stem, so the directory names the **policy** and not
anyone's intent.

**The run subdirectory is not `trial_N_<split>` under a new name.** `detail.jsonl`,
`anchor_trace.jsonl` and `anchor_telemetry.json` all have fixed names, so two arms sharing one
directory would overwrite exactly the two files a paired comparison is computed from. Keyed by split
and policy, they cannot collide — verified by running both arms into one model root and finding all
fourteen artifacts intact.

### `--trial` is gone, and `--repeat` is the replicate axis

`--trial` was declared in `response_generator.py` and **never read**; `run_hosted.py` used it only to
build a directory name. The replicate axis the code actually models is `--repeat`, which gives each
replicate its own episode id (`<query>_<i>`) and which `evaluation.py` aggregates — including the
honest note that with `--repeat 1` the reported spread "is identically 0.00 and means nothing at all".

So a variance floor is `--repeat 2` in **one** run, under one set of conditions, rather than two
separately launched directories. That also happens to be what
[`../../COLLABORATORS.md`](../../COLLABORATORS.md) demands: never difference accuracies across jobs.

### Decode is pinned, and recorded rather than assumed

`--temperature 0.0`, `--seed 42` and `--top-p 1.0` are the defaults and all three are **sent
explicitly**, not left to a client or server default. `run_manifest.json` is written *before* the run
— so a crashed run still says what it attempted — and carries the decode settings, the exact kwargs
sent, a `deterministic` boolean, the provider, `max_workers`, and sha256 of both the corpus file and
the controller spec. Two manifests that differ explain a delta the scores cannot.

`max_workers` is recorded because it changes no episode's content — episodes share no state — but a
loaded host fires the 10-second `func_set_timeout` on tool execution more often, which is the likeliest
source of a nonzero variance floor. Use `--max-workers 1` for floor runs.

**Check the firing counts in `anchor_trace.jsonl` before reading any accuracy.** Nothing in CCTU can
validate an anchor — the controllers live outside it, so no part of the harness would flag a broken
one. On BFCL an anchor shipped with a broken predicate import and suppressed every archival write
instead of only duplicates; it was invisible to every layer except trajectory telemetry, which that run
had not captured.

## See also

| | |
|---|---|
| [`VENDORED.md`](VENDORED.md) | what here is upstream and what is not |
| [`../../COLLABORATORS.md`](../../COLLABORATORS.md) | the six-step flow, and the decision at each step |
| [`../../docs/GENERALIZABILITY.md`](../../docs/GENERALIZABILITY.md) | what ports, what is glue, and why no adapter base class ships |
| [`../../adapters/adapter_template.py`](../../adapters/adapter_template.py) | the hook contract — `REQUIRED_HOOKS` / `OPTIONAL_HOOKS` |
| [`../../scripts/check_adapter.py`](../../scripts/check_adapter.py) | run this before spending any GPU on the port |
| [`../tb2_deepagents/README.md`](../tb2_deepagents/README.md) | the second integration — this one's model for file layout and size |
