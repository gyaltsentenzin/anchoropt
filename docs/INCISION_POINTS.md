# Incision points: one LLM call is three decisions, not one

The conceptual move the rest of the method depends on.

## The problem with treating a call as atomic

A tool-using agent step is usually modelled as one atomic event: context goes in, a tool call comes out,
the world changes. Under that model there is exactly **one** place to intervene — the context — and
therefore exactly one technique: prompt engineering.

That framing makes whole classes of failure unaddressable. If a model proposes a destructive call
against an already-empty container, the useful intervention is *not* to rewrite the system prompt; it is
to look at the proposed call and decline it. A prompt cannot do that, because **a prompt cannot inspect
a call that does not exist yet.**

## The decomposition

Every LLM call is split into three points at which the *system* — not the model — may act:

| # | point | the system knows | can do | cannot do |
|---|---|---|---|---|
| 1 | `pre_generation` | context, history, world state. **Not** what the model is about to do | modify context, inject instruction, prune tools, retrieve | nothing call-specific |
| 2 | `post_generation_pre_exec` | the **proposed call and its arguments**; world unchanged | suppress, reroute (new target *or* new arguments), validate | cannot undo — nothing has run |
| 3 | `post_execution` | the tool result or error; world **already changed** | reprompt, retry, reroute, recover | cannot prevent — the damage is done |

Two properties trade off monotonically, and this is the whole design tension:

```
      information available          ────────────────────────────>   increases
      damage still preventable       <────────────────────────────   decreases

      pre_generation        post_generation_pre_exec        post_execution
      knows least                                          knows most
      prevents most                                        prevents nothing
```

Point 2 is the only one holding both: it sees the concrete call *and* can still stop it. That is why
suppression and argument repair live there and nowhere else.

In code, these are properties rather than prose — see `anchoropt/anchor.py`:

```python
IncisionPoint.PRE_GENERATION.sees_proposed_call      # False  <- the fact A4 v1 needed
IncisionPoint.POST_EXECUTION.can_prevent_execution   # False  <- repair only
```

## Usually the point is obvious — and that is the useful part

For most failures the decision point is not a search problem. **The failure itself tells you where to
act**, because it tells you what is knowable and what is still preventable:

| the failure | where you must act | why there is no choice |
|---|---|---|
| a write was rejected for capacity | **post-execution** | you only learn the container is full by trying |
| the model is about to re-write a value that is already stored | **post-gen / pre-exec** | after execution the duplicate is rejected and the step is spent |
| the model is about to answer having called no tool | **post-gen / pre-exec** | "answered without a tool call" does not exist before generation |
| a read returned nothing | **post-execution** | there is no result to react to until there is a result |

Every accepted anchor landed this way. **A3 and A4 each had exactly one admissible cell** — the
design was forced, and the only open question was whether to act at all, which is a measurement.

So the value of naming the point is *not* that it opens a large space to explore. It is that the choice
becomes **explicit, stated, and checkable** instead of implicit in a trigger string. That is what
A4 v1 got wrong (below): the point was never chosen, it was assumed — and the assumption cost 4.95 pp.

## The grid, for completeness

Three points × four action families, mostly to record which combinations are *impossible*:

| | noop | reprompt | suppress | reroute |
|---|---|---|---|---|
| **pre_generation** | ✓ | ✓ | ✗ | ✗ |
| **post_gen_pre_exec** | ✓ | ✓ | ✓ | ✓ |
| **post_execution** | ✓ | ✓ | ✗ | ✓ |

Nine of twelve are admissible. The three exclusions are **structural** — they follow from what exists at
each point, not from any measurement, so they hold for every benchmark:

| cell | why inadmissible |
|---|---|
| pre-gen + suppress | nothing to cancel: no call has been proposed yet |
| pre-gen + reroute | no call to substitute for: neither its target nor its arguments exist yet |
| post-exec + suppress | cannot undo a completed execution; the step is already spent |

Each carries a machine-readable reason (`anchor.exclusion_reason`), because **a silently pruned cell is
indistinguishable from one nobody thought of** — and when *every* cell is excluded, that is rule **S3**
in [`KEEP_OR_DEFER.md`](KEEP_OR_DEFER.md): the locus is identified but not actionable, recorded rather
than pursued.

Two things the grid makes visible that the per-failure table does not:

**Prompting is one row, not the whole method.** At `pre_generation` only `{noop, reprompt}` survive — so
an unconditional global prompt is *the only lever available there*. Not a criticism of prompting, a
statement of its reach: the other seven admissible cells are unreachable by any prompt, because a prompt
cannot inspect a proposed call.

**`noop` is available at every point.** It means *let the original pipeline run unmodified here* —
the agent still acts, it just acts as it would have without us. That is what makes it the control
arm, and it has to be point-independent or comparisons lose their baseline. It is also a
legitimate verdict, not a failure: two of six rounds selected it.

### One locus where the point *was* genuinely open

A2 is the exception that shows the rule. Its failure — a read that found nothing, where the value had
been archived — was addressable either at the read (react to the miss) or earlier at the write. Two
points × four actions gave **8 nominal cells → 3 live → 2 arms actually run**, and the arms disagreed
with the prior: `reprompt` beat `reroute`, which had been favoured 2:1 on every previous locus.

That is when measurement is needed. When one cell survives, don't dress the decision up as a search.

## Why the point is a first-class design choice: A4 v1 vs v2

The strongest evidence in this project, because it is a controlled comparison with a single variable.

|  | v1 | v2 |
|---|---|---|
| signal | `no_tool_call_at_all` | `no_tool_call_at_all` |
| action family | reprompt | reprompt |
| injected text | *(identical, byte for byte)* | *(identical, byte for byte)* |
| **incision point** | **`pre_generation`** | **`post_generation_pre_exec`** |
| trigger | `step_count == 0` | answer proposed with zero tool calls |
| firings | **303/303** — 6× over-fire | **59/303** |
| **Δ train** | **−4.95 pp** | **+3.63 pp** |
| p | 0.1633 | 0.0266 |
| rec_sum net | −24 | +3 |

No-call answers went **60 → 0 in both versions**. The mechanism worked either way; only the targeting
differed.

The reason is exactly the `sees_proposed_call` property. The condition "the model is about to answer
without calling a tool" **is not observable before generation** — the answer does not exist yet. So v1
had to approximate it with `step_count == 0`, which is true of *every* episode's first turn. It fired
everywhere and paid the cost on 244 episodes that needed nothing.

v2 waits until the answer exists and the call count is known, then intervenes before the answer commits.

**Same idea. Opposite sign. One field different.**

## A second, independent instance

The accepted set makes the same point without the failure: **A2 and A4 use the same action family at
different incision points, and each had to earn acceptance separately.**

| anchor | point | action | kind | objective |
|---|---|---|---|---|
| A1 | post-execution | reroute | error recovery | accuracy |
| A2 | post-execution | **reprompt** | error recovery | accuracy |
| A3 | post-gen / pre-exec | suppress | commitment gate | accuracy |
| A4 | post-gen / pre-exec | **reprompt** | commitment gate | accuracy |
| E1 | post-gen / pre-exec | **suppress** | commitment gate | **efficiency** |

Five anchors, four distinct cells. **An action alone does not identify an intervention** — which is why
`Anchor` requires the point as a separate, typed field rather than encoding it in a trigger string.

**A3 and E1 share a cell, and that is the interesting case.** Same point, same action family — yet they
differ on something the cell does not encode: what the model is left holding.

```
                does the call EXECUTE?     what does the model OBSERVE?
   A3           no                         nothing -- the call is gone from the record
   E1           no                         the tool's VERBATIM result, replayed
```

A3 removes a redundant write, and removing the observation is the point: the model should not see a
success it did not earn. E1 withholds a provably uninformative call but replays exactly what the tool
would have returned, so the turn is *observationally identical* and only the execution is saved. Two
rejected predecessors establish that the distinction is load-bearing rather than pedantic — one
injected a user message (−1.65 pp) and one substituted a *new* tool string (+0.33 pp, but destructive
calls rose 84 → 90 and it was rejected anyway).

So a full specification of an intervention is **(point, action, what the model observes)**. The grid
gives you the first two. See [`EFFICIENCY_CLASS.md`](EFFICIENCY_CLASS.md).

Note also that the two intervention **kinds** fall out of the point, not the signal:

- **post-execution → error recovery.** Something already failed; repair it.
- **post-gen/pre-exec → commitment gate.** A call is about to happen; intercept it.

## Practical guidance

**Choosing a point for a new anchor:**

1. Ask what the condition needs to *observe*. If it references a proposed call or its arguments, point 1
   is structurally out. If it references a tool result, only point 3 has it.
2. Ask whether the damage is still preventable. If yes, prefer the earliest point that can observe the
   condition — later points spend a step to achieve the same thing.
3. If several cells remain admissible, that is a **measurement** question, not an argument. Run the arms.

**The trap to avoid:** approximating a point-2 or point-3 condition with a point-1 proxy. That is
precisely what A4 v1 did, and the failure mode is characteristic — the mechanism looks like it works
(no-call answers → 0) while the aggregate goes backwards, because the over-firing cost is paid on
episodes that were already fine.

**A firing count is the first diagnostic to read.** 303/303 on a signal with 60 known exposed episodes
is a 6× over-fire and should be caught before any accuracy is examined. Verify liveness *and* selectivity
behaviourally on the live path — source inspection has passed while the behaviour was broken.

## See also

- [`../rounds/T5_A4_no_tool_call/`](../rounds/T5_A4_no_tool_call/) — the full v1/v2 round record
- [`../rounds/T3_A3_duplicate/`](../rounds/T3_A3_duplicate/) — the first anchor learned at point 2
- [`KEEP_OR_DEFER.md`](KEEP_OR_DEFER.md) — what happens when no admissible cell has evidence (rule S3)
- `anchoropt/anchor.py` — the grid, the exclusions, and their stated reasons
