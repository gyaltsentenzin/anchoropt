# Observability is not decision context

A distinction the rest of this repo mostly collapses, written down because collapsing it cost real
time on this line — we kept trying to improve the **policy** while the **observability** layer was
still incomplete, and a policy fitted against an incomplete view is fitted against the wrong world.

## The two questions

They sound similar and they are answered by different machinery.

| | question | for whom | when |
|---|---|---|---|
| **observability** | what can we see and reconstruct about what happened? | **us** — diagnosis, attribution, debugging, audit | *afterward* |
| **decision context** | what information is available to the decision-maker at the moment it must choose? | **the agent** | *before it acts* |

Compactly:

```
   observability     = information for UNDERSTANDING the decision
   decision context  = information USED TO MAKE the decision
   decision policy   = how that context is MAPPED to an action
```

Observability is instrumentation: tool calls, controller calls, before/after state, error strings,
state hashes, who issued the action. Decision context is **part of the control policy itself** — a
choice about what the model is shown, not a choice about what we log.

## The worked example: the archival-full recovery point

The case that forced the distinction.

**What observability tells us, afterward:**

> archival was full → the model cleared **core** → core was empty → the archival write still failed.

That is a complete and correct reconstruction. It is enough to attribute the failure, and it is
*not* enough to fix it, because the model never had it.

**What decision context would tell the model, before it acts:**

> the failed write targets **archival**; archival is full; **core is irrelevant to freeing archival
> capacity**; here is the relevant archival state.

Same episode, different layer. The first is a diagnosis we hold. The second is a policy input the
agent holds.

**And the trap:** fixing observability alone does not fix the agent. It lets us *correctly see* the
failure. Those are different achievements, and it is easy to feel progress from the first while the
agent behaves exactly as badly as before.

## Five layers, not three

The loop described in [`THE_LOOP.md`](THE_LOOP.md) is really:

```
   observability  ->  error attribution  ->  decision context  ->  local policy  ->  action
   ─────────────      ─────────────────      ────────────────      ───────────      ──────
   what we can        which decision         what the model        how that maps    what
   reconstruct        caused the loss        sees when it          to an action     happens
                                            must choose
```

Two consequences worth stating:

**Error attribution carries far more weight than its billing suggests.** It sits between what we can
see and what we can act on, and it is the layer that turns a *symptom* into a *decision point*. Most
of the effort on this line went here, not into choosing actions — the action space is small and
mostly forced (see [`INCISION_POINTS.md`](INCISION_POINTS.md)), while attribution is where the real
search happens. A ranked list of frequent errors is not attribution; attribution is what says *this
loss was caused by that decision*.

**A failure at any layer looks like a policy problem from the outside.** The agent does the wrong
thing, so the instinct is to change the rule. But the same visible symptom is produced by an
incomplete observability layer (we cannot see the cause), a wrong attribution (we blame the wrong
decision), a starved decision context (the model cannot see what it needs), or a genuinely bad policy.
Only the last is fixed by changing the rule.

## Where this repo actually sits

Honest accounting, layer by layer.

| layer | state on this line |
|---|---|
| observability | **built, and it was the bottleneck twice.** Trajectory telemetry per step, live-state reads at decision time, the same-commit rule for new flags. Reconstruction produced a wrong answer three separate times before the guards read live state instead. |
| error attribution | **where most of the work went.** Locus canonicalization, backward tracing, occurrence-level re-labeling, destination attestation, the harness-fault guard. |
| decision context | **barely touched until A9, which is the first anchor to enrich it.** See below. |
| local policy | **eight accuracy anchors + one efficiency anchor**, each accepted on a paired counterfactual. |

**Decision context is the underdeveloped layer, and the anchors show it.** The dividing rule is that
**only the reprompt family can supply decision context** — every other action changes what executes
without changing what the model knows. Applying it to the accepted set:

- **A1, A3, A5, A7, A8, E1** change *what executes* — they never add information to the model's view.
  A1 reroutes, A3 removes a call, A5 evicts a provable duplicate and replays the original write, A7
  rewrites the store and retries verbatim, A8 cancels a destructive clear and frees the slot another
  way, E1 replays a recorded result. E1 is explicit that the model's view must be **byte-identical**.

  **A8 sharpens the point.** It prevents the model from destroying 50 entries, and it never tells the
  model that — no explanation, no warning, no revised view of the store. The agent proposes a clear and
  simply observes that the write it wanted now succeeds. That is constrained execution in its purest
  form on this line.
- **A2, A4** are `reprompt`, the family that *could* supply decision context. A4 is deliberately
  **non-directive** — 23 of its 48 exposed episodes have nothing in the store, so telling the model
  what to look for invites fabrication. A2 does point the read at a reachable destination, which is
  the closest thing here to a decision-context intervention.

### A9 is a third category, and it answers this document's own closing question

The binary above — reprompt informs, everything else constrains execution — **does not cleanly hold for
A9**, and the exception is more interesting than the rule.

A9 issues no instruction. The model is never told to search elsewhere, never warned, never corrected. So
by the letter of the rule it is not a reprompt and should fall under "changes what executes". But what it
changes is **the content of the tool result the model reads before it answers** — the merged top-k
replaces a weak one at its own index. The model's view of the world is *materially different*, and
strictly better informed, without a single word of instruction.

> **So decision context can be enriched through the DATA CHANNEL rather than the INSTRUCTION CHANNEL.**
> A9 supplies decision context by improving what a tool returns, not by telling the model anything.

That distinction was not visible while every anchor acted on writes. It also means the closing question
below — *what is the minimal decision context the model needs so it stops acting on the wrong
container?* — has a first measured answer on the vector backend: **give it the other container's entries,
ranked against the ones it already had.** +16.9 / +17.5 pp on that shard, and the model needed no
instruction to use them.

The caution that comes with it: A9's own first two versions changed the telemetry rather than the data
channel, and the model's behaviour was *correct on the input it actually received* for 78 firings. An
intervention on the data channel is only as real as its delivery — see
[`CONSUMER_BOUNDARY_RULE.md`](CONSUMER_BOUNDARY_RULE.md).

So **all but three anchors** improve the agent *without changing what it knows at all*,
and the ratio got more lopsided as the work continued: A5 and A7 are reroutes, A8 a suppression. That is a real
finding — constrained execution beat delegated recovery, and the arm that tried the opposite (B1: a
generic "reconsider and try again") scored **−6.60 pp, p = 0.0352**. It is also a statement of what has
*not* been tried: none of these anchors asked whether the model would have chosen correctly given
better information.

**A7 is the partial exception worth naming.** It calls a model at inference time — the only accepted
anchor that does — but it uses that call to *rewrite the store*, not to inform the agent's decision.
The agent is never told the compaction happened. So even the anchor with an LLM in the loop operates
on execution rather than on decision context.

## The next scientific question

For the archival-full case the question is no longer *"what should we log?"* — observability there is
adequate. It is:

> **What is the minimal decision context the model needs at this recovery point so that it stops
> acting on the wrong container?**

Three properties make that a well-posed question rather than a prompt-engineering hunch:

1. **Minimal.** The interesting quantity is the *smallest* sufficient context, because everything
   added is tokens spent and attention diluted. "Show it more" is not an answer.
2. **Local to a decision point.** Not a global preamble. This line is prompt-free by construction, and
   a preamble would invalidate every comparison in it (`run.py` refuses a policy carrying one).
3. **Measured the same way.** A decision-context intervention is an arm like any other: paired,
   engagement-vetoed, harm-before-benefit, criteria frozen first. B1 is the warning — delegating the
   remedy to the model was tried and it lost significantly. The open question is whether *informing*
   the model, without delegating the choice, does better.

**Why B1's failure does not settle it.** B1 said "reconsider and try again" — that delegates the
decision without supplying anything. Supplying the relevant archival state while keeping the action
constrained is a different intervention, and it has not been run. Distinguishing "the model chose
badly" from "the model could not see what it needed" is the whole point of separating these layers.

## For your own benchmark

Ask the four questions in order. Answering them out of order is how a policy gets fitted to an
incomplete view.

1. **Can I reconstruct what happened?** If not, instrument first — and read **live state at the
   decision point** rather than reconstructing it later. That mistake recurred three times here.
2. **Can I say which decision caused the loss?** Not which error appeared most often. Frequency is
   not attribution.
3. **What did the model actually see when it chose?** Often less than you assume, and often less than
   your logs contain. Your reconstruction is not the agent's view.
4. **Only now: what rule should apply?** And decide whether the fix is a *different action* or the
   *same action with better information* — they are different interventions with different failure
   modes, and this line has only seriously tested the first.

## See also

- [`THE_LOOP.md`](THE_LOOP.md) — the policy/signal coordinate descent these layers sit inside
- [`INCISION_POINTS.md`](INCISION_POINTS.md) — why the action space is small, which is why attribution carries the weight
- [`KEEP_OR_DEFER.md`](KEEP_OR_DEFER.md) — the acceptance rules any decision-context arm would also face
- [`EFFICIENCY_CLASS.md`](EFFICIENCY_CLASS.md) — E1, whose acceptance *requires* the model's view be unchanged
