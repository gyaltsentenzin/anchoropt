# For collaborators: applying this to your benchmark


The transferable thing here is the **flow**: a procedure for deciding what to change in an agentic workflow, and, far more
often, what to leave alone. This page is the flow, with the decision you have to make at each step.

Start by running it on our data (~10 s, no GPU, no model):

```bash
git clone <this repository> && cd anchoropt
uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e ".[dev]"

python scripts/walk_framework.py        # the six steps below, computing real numbers
python scripts/verify_progression.py    # recomputes every published figure; exits non-zero on drift
```

---

## The flow

```
   reward
     │  1  MINE          which tasks still fail?
     v
   residual
     │  2  ATTRIBUTE     which DECISION made each failure avoidable?  (not: which error printed)
     v
   decision + cause
     │  3  LOCATE        which of the 3 points in the LLM call can act on it?
     v
   incision point
     │  4  PRUNE         which actions are even possible there?  state every exclusion
     v
   1-3 candidates
     │  5  MEASURE       one paired counterfactual arm per candidate
     v
   verdict
     │  6  ACCEPT/DEFER  engagement -> attribution -> harm -> benefit, in that order
     v
   new incumbent  ──────► its trajectories become the next round's no-op control, then re-mine
```

### 1. Mine — start from the reward, not necessarily the error log (which may be vague and non-actionable)

Take the tasks your agent still gets wrong. Rank candidate failure loci by **linked downstream loss**
(how many task failures trace back to them), never by how often the error appears.

*Why it matters:* on our corpus the rarest locus (31 events) did the most damage per occurrence — 1.29
linked failures each — while the most common (543 events) did the least, 0.149. A frequency ranking
buries your best target. Rank 3 outranked ranks 4–5 on *fewer* events.

**Your decision:** what counts as a "locus" in your benchmark, and can you link one to task failures?
If you cannot draw that link, ranking is meaningless and nothing downstream works.

> **Code:** [`anchoropt/learning/policy_tree.py`](anchoropt/learning/policy_tree.py) — `mine_candidates()`
> discovers candidate signals; `rank_candidates()` orders them in one unit (distinct failing episodes)
> and excludes recovery actions with a stated reason. `vacuous_result_kind()` catches the calls that
> *succeed* and return nothing — the ones no error-keyed miner sees.
> [`anchoropt/learning/remine_incumbent.py`](anchoropt/learning/remine_incumbent.py) — re-mines with
> occurrence-level re-labeling after each install; **this is the one you cannot skip** (a step-level
> version left 47 % of already-repaired occurrences counted as live support, which flipped rank 1).

### 2. Attribute — the error is a symptom, not the cause

Trace backward from the failure to the first consequential decision.

*Why it matters:* one error string in our corpus (`not_found`) split into **two different causes** —
39 cases where the value was stored but the read looked in the wrong place, 5 where it was never
stored. Two causes, two different anchors, and without the split an arm would have targeted a mixture
and been unattributable either way.

**Watch for self-inflicted residuals.** Most of our anchors exist because their *predecessor* created
the failure they fix. A1 reroutes writes to archival — which produces the `not_found` reads A2 fixes
(39/39 self-inflicted), the key collisions A3 fixes (59/59 against A1's own keys), and eventually the
saturated container A5 has to manage. This is why step 6 loops.

> **Code:** [`anchoropt/attribution/constraint_locus.py`](anchoropt/attribution/constraint_locus.py) —
> `classify()` maps an error string to a semantic locus (returns `None` on an unseen wording; see the
> ladder below).
> [`anchoropt/attribution/attribution_miner.py`](anchoropt/attribution/attribution_miner.py) — splits
> one locus by *first consequential decision*, which is what separated A2's 39 read-side cases from its
> 5 write-side ones.
> [`anchoropt/attribution/trace_backward.py`](anchoropt/attribution/trace_backward.py) — walks a chain
> backward to the divergence step (never stored / stored-then-cleared / rejected / mangled key).
> [`anchoropt/attribution/harness_guard.py`](anchoropt/attribution/harness_guard.py) — **use this on
> every trajectory read.** An implementation crash must never become a mined signal; ours got mined at
> support 13 before we caught it.

### 3. Locate a specific local intervention point

| point | knows | can do |
|---|---|---|
| **pre-generation** | context and state — *not* what the model will do | inject instruction, modify context, prune tools |
| **post-gen / pre-exec** | the proposed call **and its arguments**; nothing has run | suppress, reroute (new target *or* new arguments), validate |
| **post-execution** | the result/error; the world already changed | reprompt, retry, reroute, recover |

**Usually the failure forces the point** — a capacity error is only visible post-execution; "about to
answer without calling a tool" does not exist before generation. Two of our four had exactly one
admissible cell.

*But get it wrong and it dominates everything.* Our A4 exists in two versions with **identical signal,
identical action, byte-identical injected text**, differing only in the point:

| | pre-generation | post-gen / pre-exec |
|---|---|---|
| fired | 303/303 (6× over-fire) | 59/303 |
| Δ accuracy | **−4.95 pp** | **+3.63 pp** |

Same idea, opposite sign. The earlier version could not *observe* its own trigger condition, so it
fired everywhere and paid the cost on 244 episodes that needed nothing.

**Your decision:** does your harness expose all three points? If it only lets you touch the prompt,
you have one row of the table and eight of the nine cells are unreachable.

> **Code:** [`anchoropt/anchor.py`](anchoropt/anchor.py) — `IncisionPoint` with the two properties that
> decide everything: `sees_proposed_call` and `can_prevent_execution`. The `Anchor` dataclass *requires*
> the incision point as a typed field, which is the fix for the −4.95 pp version: it cannot be left
> implicit in a trigger string.

### 4. Prune — state every exclusion

Actions are `{noop, reprompt, suppress, reroute}`. `reroute` covers both a different tool *and* the
same tool with different arguments — we merged those, because they are one operation ("substitute the
call") with one evidential requirement.

Most cells prune **structurally**: you cannot suppress a call that has not been proposed, and you
cannot undo one that already executed. Write the reason down each time — a silently dropped option is
indistinguishable from one nobody considered.

**Attest destinations on *resolving*, not on *not erroring*.** We once admitted a reroute target at
"11/11 clean", where clean meant *did not error* — and 9 of the 11 returned nothing at all.

> **Code:** [`anchoropt/anchor.py`](anchoropt/anchor.py) — `feasible_actions(point)` returns the
> admissible set; `exclusion_reason(point, action)` returns *why* a cell is dead, so a report can state
> every exclusion instead of showing a shrunken grid.
> [`anchoropt/attribution/destination_attest.py`](anchoropt/attribution/destination_attest.py) —
> `classify_outcome()` / `attest()`, which is the "resolving, not not-erroring" check.
> [`anchoropt/learning/screen_ranked_loci.py`](anchoropt/learning/screen_ranked_loci.py) — the cheap
> offline queue screen; each of its four tests has already changed a real decision.

### 5. Measure — paired counterfactual, same prefix

Same factual history, one intervention, replay the causal suffix. Both arms in the **same job**.

**Measure your variance floor first, before any candidate.** Run the identical policy twice and count
outcome flips. Ours is *zero* (0/12 stores, 0/303 calls, 0/303 flips at `temperature=0.001`), which is
why we treat a single-case loss as real. If yours is not near zero, every harm tolerance here needs
widening and small deltas are not interpretable at all.

**Never difference accuracies across jobs.** Two of our controls read 29.04 % and 24.36 % purely from
corpus composition, while agreeing to 1 flip in 228 on shared cases.

**SHARD BY BACKEND, AND SUBMIT THE SHARDS CONCURRENTLY.** This is the single highest-leverage thing to
copy from this project's run protocol, and it is not an optimisation — it is what makes a paired number
reproducible at all.

```bash
python benchmarks/bfcl_v4/run.py --compare --shard all --dry-run   # 6 commands, ready to submit
```

A 3-backend × 2-arm contrast is **six independent jobs**, one GPU each. Not six sequential ones.

**Why it matters more than wall-clock: whole-corpus runs are not byte-reproducible.** Two runs of the
*same effective policy* diverge even when the treatment fires **zero times**, because whole-corpus
execution adds cross-chain sequencing that is not byte-stable. Sharded pairs are **bit-identical except
in the shard where the treatment actually fires**. So a paired number from a whole-corpus run cannot be
quoted, and several of our own early numbers were retroactively demoted for exactly that reason.

It also gives you the **per-backend split for free**, which is what turns "the arm gained 1.65 pp" into
"the arm gained 4.76 pp on the one backend it can reach and exactly 0.00 pp on the other two." That
decomposition is how non-attribution gets caught — see the exposure discussion below.

Three things each shard needs. Each has broken a real fleet of ours when omitted:

| | why |
|---|---|
| a port per **(arm × backend)** | not per backend. Two arms sharing a port were safe once only because the scheduler happened to place them on different hosts. A table plus a fallback is how our first version put `control/kv` and `a8/vector` on the same port — derive it from both coordinates |
| its own snapshot cache | the cache key hashes the prerequisite id list, so shards necessarily differ. That is correct, and it means **equivalence must be checked on store CONTENT, never on cache keys** |
| the **same** store fingerprint within an arm | that is what makes the shards' results comparable once concatenated |

**Sharding is only sound if no case dependency crosses a shard boundary.** On this corpus none does
(0 of 387, asserted at shard time rather than assumed) — if that ever changes, a shard would build its
stores from partial chains and the run would be silently wrong rather than failing. `--shard` refuses in
that case. **Check the same property on your corpus before sharding it.**

**And keep `--workers 1` inside each shard.** Parallelism *across* isolated runs is measured at 0 flips;
parallelism *within* one store build is forbidden (parallel replicates diverge reproducibly). Different
axes — sharding is not permission to raise `--workers`.

> **Code:** [`scripts/verify_progression.py`](scripts/verify_progression.py) — the paired sign test and
> the report artifact; read this for the comparison shape rather than reimplementing it.
> [`benchmarks/bfcl_v4/run.py`](benchmarks/bfcl_v4/run.py) — `--compare` runs both conditions in ONE
> job, which is the part that makes the comparison licensed, and `--shard` produces the six concurrent
> jobs plus the environment each needs. Copy its shape for your benchmark:
> `benchmarks/<yours>/run.py`.

### 6. Accept or defer — in this order

```
1  ENGAGEMENT   did it fire, in the mined context, on the target?   <- veto BEFORE scoring
2  ATTRIBUTION  are the changes caused by it, not replay noise?
3  HARM         does any subgroup degrade beyond tolerance?
4  BENEFIT      only now: do paired gains exceed losses?
```

Engagement comes first because a candidate can pass every outcome test while having nothing to do with
the residual it claims. That happened: a winner passed all four terms while firing 12× on a *different*
backend's error contract, and its own target residual **grew**.

**Then persist the accepted world as the next round's no-op control** and re-mine. Round N's control
file *is* round N−1's arm file — that is what makes this boosting rather than a static ranking, and it
is a correctness property, not an optimisation.

**Freeze your criteria before launch, with a hash.** Every round here ships a pre-registered spec; a
stopping rule chosen after seeing the result is a rationalisation, not a rule.

> **Code:** [`anchoropt/learning/check_attribution.py`](anchoropt/learning/check_attribution.py) —
> `scan_arm()`, the engagement veto that runs *before* any outcome is scored. This is the one most
> likely to be skipped and the one that caught a candidate passing every outcome term while firing on
> the wrong contract.
> [`anchoropt/learning/evidence_ledger.py`](anchoropt/learning/evidence_ledger.py) — cross-round memory:
> `should_propose()` gates re-proposing a settled signal, `record_measurement()` keeps per-policy
> history, `mark_underpowered()` records that an arm *could not* have detected its effect.
> [`rounds/*/FROZEN.md`](rounds/) — what a pre-registered spec looks like; A4's is pinned by test to the
> sha256 cited in its own acceptance commit.

---

## Where the anchors actually live: our flow vs standard BFCL v4

The single most useful thing to understand before porting. **Our code is not part of the BFCL harness** —
it wraps it. `bfcl generate` is never invoked.

### First, what AnchorOpt actually requires

Read this before the diagrams. The diagrams look alarming out of context, and the requirement is far
smaller than they suggest.

**AnchorOpt needs three hooks:**

```
   before you generate            ...check this, maybe add context
   before you EXECUTE a call      ...check this, maybe block or substitute it
   after a tool result or error   ...check this, maybe reprompt or repair
```

That is the whole interface: three callbacks around an existing agent loop. An anchor is a predicate
plus an action at one of those moments.

**So why is there a 3,600-line evaluator here?** BFCL v4 exposes none of them. Its loop calls the model
and executes whatever comes back, with nothing in between:

```python
# bfcl_eval/model_handler/base_handler.py -- the shape, paraphrased
while not done and count < MAXIMUM_STEP_LIMIT:
    response = self.model_call(...)                    # generate
    decoded  = self.decode(response)                   #   <-- nowhere to stand
    results  = execute_multi_turn_func_call(decoded)   # execute
    #                                                  #   <-- nowhere to stand
    count += 1
```

No `before_execute`, no `after_result` to register against. Getting a decision point at all meant
reimplementing the loop around the harness's pieces. **That is a BFCL implementation constraint, and it
is the most misread thing about this work.**

| | |
|---|---|
| the method requires | three hooks around an agent loop |
| BFCL provides | none of them |
| so this repo contains | a reimplemented loop borrowing BFCL's executor, backends, and official scorer |

**Where the hooks already exist, AnchorOpt is an extension.** Any harness with a pre-execution
tool-call hook that can *block* a call gives you incision point 2 — the hardest one, and the only point
that sees the concrete call while it can still be stopped. Point 3 is a post-result callback; point 1 is
context assembly, which nearly everything exposes.

Pi, for example: its `tool_call` hook runs before execution and can block the call, so the anchors
register as three callbacks with no loop replacement.

**Measure your harness before budgeting the work.** Pre-execution and post-result hooks make this an
afternoon of adapter code. A loop shaped like the one above means you are reimplementing it — read
[`docs/FIDELITY_AUDIT.md`](docs/FIDELITY_AUDIT.md) first for what that costs in credibility.

### The diagrams, with that in mind

### Standard BFCL v4 (no AnchorOpt)

```
$ bfcl generate --model X --test-category memory
$ bfcl evaluate --model X

  bfcl_eval/__main__.py                    generate()                    ← official CLI
    └─ _llm_response_generation.py         main() → generate_results()
         └─ model_handler/base_handler.py  inference()          line 103 ← BFCL'S OWN LOOP
              ├─ model call
              ├─ execute_multi_turn_func_call(...)                       ← tool execution
              └─ repeat until done / step limit
    └─ eval_checker/... agentic_checker                                  ← scoring
```

One loop, one process, **no interception points**. The model proposes a call, the call executes, repeat.
A duplicate write happens and the store returns an error; nothing sits between proposal and execution.

### With AnchorOpt

```
$ python benchmarks/bfcl_v4/run.py --compare                            ← OURS

  benchmarks/bfcl_v4/run.py                                             ← ours: composes the policy
    └─ benchmarks/bfcl_v4/run_memory_eval.py            line 300        ← ours: builds the evaluator
         └─ evaluator/memory_evaluator.py                               ← OURS: REPLACES inference()
              │      MemoryAnchorOptEvaluator, 3,621 lines, no upstream counterpart
              │
              ├─ handler.inference_step(...)                → BFCL's handler
              ├─ ◀ A4  zero-call reprompt      line 2754    ← pre-generation-ish (before answer commits)
              ├─ ◀ A3  suppress duplicate      line 2579    ← post-gen / PRE-EXEC
              ├─ execute_multi_turn_func_call(...)          → BFCL's tool executor
              ├─ ◀ A1  capacity repair         line 3198    ← POST-EXEC
              ├─ ◀ A2  key-not-found reprompt  line 3220    ← POST-EXEC
              └─ agentic_checker(...)                       → BFCL's OFFICIAL scorer
```

### What is borrowed vs replaced

| | whose | used? |
|---|---|---|
| `bfcl` CLI, `_llm_response_generation.py`, `base_handler.inference()` | Berkeley's | **no** — replaced |
| memory backends (`memory_kv/vector/rec_sum`) | Berkeley's | **yes** |
| `execute_multi_turn_func_call` | Berkeley's | **yes** |
| `agentic_checker` + official ground truth | Berkeley's | **yes** |
| handler methods for context assembly | Berkeley's | **yes** |
| `run.py`, `run_memory_eval.py`, `memory_evaluator.py`, every anchor | ours | — |

So: **the episode loop is replaced; tool execution and scoring are borrowed.** A scored case means what
BFCL says it means. The loop around it — step limits, turn construction, interception — is ours.

### Where each anchor is declared, and where it fires

Two files. The **declaration** says *what* and *where*; the **firing site** does it.

| | file | what |
|---|---|---|
| declaration | `harness/bfcl_eval/model_handler/memory_gates.py` | `class GateSpec` (line 68). **`trigger_kind` (line 78) IS the incision point.** `MEMORY_GATE_REGISTRY` holds the four entries |
| firing | `evaluator/memory_evaluator.py` | lines 2624 (A3), 2804 (A4), 3248 (A1), 3270 (A2) — verified by test, since line numbers drift |

```
key                              trigger_kind        = incision point      remedy     telemetry flag
on_redundant_write_suppressed    pre_exec_decoded    post-gen/pre-exec      suppress   redundant_write_gate
on_domain_error_core_full        post_exec_result    post-execution         reprompt   a1_core_full_gate
on_core_full_rerouted            post_exec_result    post-execution         reroute    reroute_gate
on_domain_error_key_not_found    post_exec_result    post-execution         reprompt   a2_key_not_found_gate
```

**For your benchmark, this is the question to answer first: does your harness expose those three
moments?** If your runner is a single `while` loop calling the model and executing whatever it returns,
you have BFCL's shape — one interception point (the prompt) and eight of the nine grid cells unreachable.
Getting the other two requires restructuring the loop, which is what `memory_evaluator.py` is.

### Two consequences worth internalising

**Nothing in BFCL can validate your anchors.** Because the gates live outside it, no part of the harness
would flag a broken one. A3 shipped here with a broken predicate import and **suppressed every archival
write instead of only duplicates** — invisible to every layer except trajectory telemetry, which that run
had not captured. Set `ANCHOROPT_TRAJ_DIR` and check firing counts *before* reading any accuracy.

**A reimplemented loop is a fidelity liability, and it must be audited.**
[`docs/FIDELITY_AUDIT.md`](docs/FIDELITY_AUDIT.md) enumerates six divergences from official BFCL,
file:line cited, five still present. Budget for writing that document for your own port — it is what
tells a reader which of your numbers are comparable to the official leaderboard (none of ours are,
directly) and which are internally valid (the paired deltas are).

## A second objective: efficiency anchors

Everything above optimises **accuracy**. There is a second axis, and it needs a different bar.

The objective is **accuracy per LLM call**. It improves two ways — raise the numerator (the accuracy
anchors) or lower the denominator (**E1**, where *E* stands for efficiency). A pipeline reported only on
accuracy makes the second kind of progress invisible, and worse, *scores an efficiency anchor as a
failure for passing*: judged as an accuracy anchor, E1 is +0.00 pp and looks like a null result.

Applying the accuracy rule to an efficiency anchor is a **category error** — `Δ_train > 0` is
unsatisfiable for an anchor whose defining property is that behaviour does not change. Note also that
**accepted is not the same as active**: E1 is accepted (8/8) and is currently switched off in the frozen
policy, which appears to be an assembly oversight rather than a decision. See
[`docs/ACCEPTANCE_RULE.md`](docs/ACCEPTANCE_RULE.md).

| class | accepted when |
|---|---|
| **accuracy anchor** | positive paired end-task delta surviving the harm clause |
| **efficiency anchor** | **no task loss** + **provably useless work removed** + **no new harm**. 0 pp is a **PASS** |

**Why 0 pp is the strongest result for this class, not the weakest.** For a behaviour-preserving
anchor, a small *positive* delta is a warning sign — the change was supposed to be observationally
inert, so a gain means it leaked into behaviour and got lucky. Zero flips means the trajectories are
identical outside the withheld calls, which is precisely the claim.

> **Code:** `anchoropt/mechanisms/memoize_guard.py` (the predicate),
> [`rounds/E1_efficiency/`](rounds/E1_efficiency/) (the record),
> [`docs/EFFICIENCY_CLASS.md`](docs/EFFICIENCY_CLASS.md) (the class and its caveats)

### The pattern to look for in your benchmark

> **known state + known action + deterministic tool → reuse the recorded outcome.**

Not a heuristic that a call is probably unhelpful — a *proof* that it is uninformative. If the tool is
deterministic and the state has not changed, a repeated call cannot return anything new. It can only
cost a step. E1 instantiates this for a removal whose target is already known absent.

**Four invariants. Port them or it is not this anchor:**

1. **Look up BEFORE executing**, and execute the *shortened* batch. The first implementation memoised
   *after* the executor ran and saved nothing — a behaviour change dressed as an efficiency one, and it
   passed every predicate test. Prove it with a counting stub: executor invocations for the targeted
   calls must be **0**. *Fewer proposals is not a saving; only fewer executions is.*
2. **Splice the cached result back at its original index.** Result delivery zips positionally, so one
   missing slot shifts every later result onto the wrong call.
3. **Replay the tool's VERBATIM string.** Never author a new one. The predecessor that substituted a
   *new* message gained +0.33 pp and was **rejected** — destructive calls rose 84 → 90.
4. **Record only from calls that actually executed.**

And when you measure it: **count executed redundancy, not proposed.** In the working repo the first
verdict counted redundant *proposals*, where withheld calls still appear by design — it read 16 → 16
and failed the anchor. Executed redundancy read **16 → 0** on the same data.

### Two consequences for the loop

- **A behaviour-preserving anchor does not trigger a re-mine.** Re-mining looks for loci whose
  trajectories shifted; a 0 pp / zero-flip arm has none.
- **Track an efficiency residual separately** from the accuracy residual. (In the working repo this is
  so far a design commitment, not built infrastructure — the accuracy residual is a real ledger, the
  efficiency one is not yet.)

### Read the scope before you quote the numbers

E1 fired **16 times, all in two `vector` prereq episodes — zero times in any scored episode.** So:

- ✅ *"16 of 16 redundant executions eliminated in the population where it fires; scored accuracy
  unchanged, zero flips."*
- ❌ *"saved 16 calls across 303 cases"* — implies a rate that does not exist.

It also has **no frozen pre-registration** (unlike A1–A4), no dev run, and its `kv` support is
unit-tested but never observed firing live. The biggest portability caveat: the zero-flip bar depends
on **deterministic decoding** — under sampling, a 0 pp / zero-flip arm is unobtainable and this
criterion does not transfer as written.

## When you run out of signals: the escalation ladder

This is the part most likely to be useful and least likely to be obvious. Two orthogonal moves.

**Richer evidence, cheapest rung first:**

| rung | key on | exhausted when |
|---|---|---|
| **lexical** | the error string the environment gave you | failures stop producing distinct strings |
| **semantic** | canonicalize wordings onto one constraint | failures produce **no error at all** |
| **structural** | trajectory *shape* — what was called, what came back, what happened next | — |

A caution on rung 2: a hand-written cue table is a fast path, **not** the coverage story. An unseen
wording returns `None` and *vanishes from the ranking*. Our A3's locus came from an LLM canonicalizer,
not the table. `python scripts/walk_framework.py --step 1` shows that failure live.

**And earlier in the workflow — query → write.** Failures are *observed* where scoring happens but
often *caused* several turns earlier:

- One iteration's rank-1 candidate was rejected on exactly this: an empty archival search resolved in
  **0 of 50** episodes. No read-side action conjures a fact that was never written — the fault was
  upstream.
- **220/220** vector-length and **384/386** capacity events occur in *prereq* episodes, ~0 in query. A
  query-only evaluation cannot exhibit them at all.

**Know when to stop.** Frozen rules, checked cheapest-first: no admissible action (S3); top locus
covers < 10 % of residual (S1 — *derive that floor from your own corpus power*, don't copy ours);
locus already mostly handled (S2); unmined mass exceeds the largest mined candidate (S5 — the
*signal* is your bottleneck, not the policy); diminishing returns (S4).

Stopping does **not** mean the method failed. Two of our six rounds installed nothing.

---

## What generalizes, and what you will have to write

**Transfers:** the flow above, the incision-point decomposition, acceptance ordering, the stopping
rules' *derivations*, the ladder.

**Does not — this is real work, scope it before starting:**

1. **Signal extraction.** Classifying tool results into errored / vacuous / resolving / *harness fault*.
   Budget for the last one: an implementation crash must never become a mined signal. On our benchmark
   an empty-memory search raised `ZeroDivisionError` and got mined as a real error contract at
   support 13 before we caught it. Assume yours has one too.
2. **Policy-space trimming below the grid** — whether a substitute destination exists and *resolves*,
   whether arguments map mechanically.
3. **Action execution with telemetry that survives it.** A suppressed call that simply vanishes from
   the trajectory erases the evidence needed to evaluate the suppression. That bug recurred **four
   times** here.

**There is one adapter seam, and it is deliberately narrow.** Everything benchmark-specific about a
memory backend lives in a `StoreAdapter` — the error phrasings, the per-container caps, the write verbs,
the id keyword, and how to read live state. Register one and the core algorithm works:

```python
from anchoropt.mechanisms.constraint_repair import Constraint, StoreAdapter, register_adapter, plan_repair

register_adapter(StoreAdapter(
    name="my_store",
    error_patterns={Constraint.NO_SLOTS: r"capacity exceeded",
                    Constraint.ENTRY_TOO_LONG: r"item too large"},
    entry_caps={"main": 512, "overflow": 4096},
    relocation_target="overflow",
    write_verbs=("store_add",),
    id_kwarg="item_id",
))

plan = plan_repair(call, result, state, backend="my_store")   # eviction / relocation / reduction
```

`tests/test_constraint_repair.py` proves a fresh benchmark reaches all three repair strategies that way,
with no core edits. The four fields an adapter supplies are exactly the four that are *not* portable, and
each is a place a port has silently broken here before — two backends of the *same* benchmark phrase
"container full" differently, and an anchor keyed on one string never fired on the other at all.

**What is still deliberately absent is a base class to inherit.** The shipped BFCL adapters are a worked
example, not a framework: this loop has run end-to-end on one benchmark, so an inheritance hierarchy
generalized from it would encode our accidents. `docs/GENERALIZABILITY.md` names the remaining seams.
Port to a second benchmark and we abstract from two examples — still the most useful thing a
collaborator could do next.

---

## Status, honestly

| | |
|---|---|
| A1–A4 | frozen; every published number reproducible offline in ~10 s |
| A5, A7 | accepted under the same rule; policies and frozen ledgers ship, **numbers transcribed** (no run artifacts here) |
| A6 | **deferred** — the signal is real, the remedy regresses the validation split. [`rounds/A6_deferred/`](rounds/A6_deferred/) |
| E1 | accepted on the efficiency objective, and currently **not active** — see above |
| the flow above | documented, runnable, and covered by the test suite |
| running BFCL end-to-end here | the full 303-case run completes; numbers agree with the frozen artifacts, but A3 was mis-scoped in that run and **re-measurement is pending**. Read [`docs/FIDELITY_AUDIT.md`](docs/FIDELITY_AUDIT.md) — this is not the official BFCL pipeline |
| A9 | accepted and **three-shard confirmed**: +4.95 pp train / +8.33 pp dev corpus-wide, with kv and rec_sum byte-identical between arms. Its **shard artifacts are still on the cluster**, so T8 stays the last endpoint you can recompute offline here — [`NOT_SHIPPED.md`](rounds/T9_A9_xcontainer_merge/result/NOT_SHIPPED.md) |
| A9's retracted rationale | it was argued "additive by construction — never a displaced correct answer". **False**: both dev losses evicted the entire core set. Accepted on a favourable measured trade instead. If you build a merge-style anchor, do not reuse that phrase |
| naming, resolved | the working repo called this candidate "A8" while this repo's A8 is dedup-clear recovery. It is **A9** here, gate key `on_low_similarity_cross_container`. **Read the gate key, not the numeral** — the working repo also has an "N0-A9-U" and an "A9-R-top1" that are both REJECTED/VOID |
| beyond A9 | in flight in the working repo |

On that third row: five GPU submissions found four real bugs that 177 passing tests and a dry-run
could not — an argparse footgun, a package-name collision, another user's hardcoded path, and a
bare-filename assumption. If you port this, **run it early on real hardware.** Static checks will tell
you the command looks right.

## Reading order

1. `README.md` — the result, then the framework
2. `rounds/` — five rounds in order, **including the two that installed nothing**
3. `docs/THE_LOOP.md` — counterfactual worlds, deferral, the expansion ladder
4. `docs/GENERALIZABILITY.md` — before you plan your own port
5. `docs/KEEP_OR_DEFER.md` — the criteria, and what they stopped

## Prior work

The gate-dispatch machinery A1–A4 run on — `GateSpec`, `suppress_specs`, the signal ladder, the generic
executor along with ~30 hand-authored gates that established the premise
this all rests on: that a local intervention at a specific decision point can move this benchmark. Those
gates are retired here (a control carrying a hand-tuned gate is not a control for a learned one), not
repudiated. See `benchmarks/bfcl_v4/VENDORED.md`.
