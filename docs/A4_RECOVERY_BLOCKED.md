# Targeted A4 recovery: blocked at materialization, before GPU

The autonomous half **succeeded**. The experiment could not run because the evaluator has no gate
that realizes the recovered controller, and forcing one would have measured a different controller
under its name. No GPU jobs were launched.

## What the loop achieved autonomously

With every accepted-anchor identifier scrubbed from the learner-visible runtime and no fixture
imported, the live proposer, given only R₁'s four diagnoses, localized to:

```
l = post_generation_pre_exec    phi = no_tool_call_at_all    U-hat = {reprompt, suppress}
rationale: "every residual is a commitment made at a turn that contained no further tool call"
```

That is the historical A4 locus and boundary, reached from attribution alone. `AnchorPolicyOpt`
then instantiated **3 grounded REPROMPT arms** (distinct instructions × retry budgets) and correctly
recorded `SUPPRESS_INFEASIBLE` with its missing requirements.

## Why the arms could not be materialized

The controller needs an intervention that fires when the model **is about to answer without having
called a tool**. Surveying every gate in the evaluator's registry:

| candidate gate | condition | fires on the 26 target cases |
|---|---|---|
| `on_premature_idk` | empty decode **AND** an IDK phrase (`"i don't know"`, `"cannot find"`, …) | **0 / 26** |
| `on_retrieval_success_pre_answer` | `step_n >= 1` **AND** a *successful retrieval* | **0 / 26** |
| 6 × `turn_start_user` gates | `step_n == 0`, before the decision exists | fires on all 89 — cannot observe the condition |

Measured on the native incumbent (`se1ctl`, vector/train, n=89):

* `no_tool_call_at_all` fires on **26** episodes.
* Of those 26, **0** contain an IDK phrase — every one is a *confident-but-wrong* answer. So
  `on_premature_idk`, the only pre-execution gate with `remedy_type=reprompt`, is inert on exactly
  the population the controller targets.
* Overlap between "no tool call" and "a successful retrieval" is **0 by construction** — an episode
  with no tool call cannot have completed a read. The two conditions are mutually exclusive.

## The replication gap, named

Per the pre-registered diagnosis list, the blocker is the fourth one:

> **the historical A4 controller depended on behavior not represented by the current action
> contract.**

More precisely, it depends on a *firing condition* the evaluator does not expose. Historical A4 fired
on "zero tool calls at post-generation"; the shipped registry offers only *IDK-phrase-gated* or
*post-successful-retrieval* reprompts. This is a **host-capability gap in `U_H(ℓ)`**, not a defect in
attribution, localization, policy search, or the action contracts — each of which did its job.

It is the same class of finding as candidate 1 in Self-Evolve R1, and now measured rather than
inferred: the recovered controller lies outside the realizable host space.

## What was NOT done, and why

* **No arm was forced into `on_premature_idk`.** It would fire 0/26, producing a guaranteed 0 pp
  arm with 0 firings — proof of non-engagement, dressed as a null result for the controller.
* **No arm was forced into `on_retrieval_success_pre_answer`.** Zero overlap with the target
  population; it would measure a different controller under A4's name, which is the confound
  `docs/ANCHORS.md` records at −4.95 pp for firing the right text at the wrong boundary.
* **R₁ is NOT marked exhausted.** Its other two loci remain untried in the ledger:
  `post_execution / no_informative_result` and
  `post_execution / retrieval_similarity_below_threshold`.

## Honest scope

This is a *targeted leave-one-anchor-out calibration*, not evidence that the autonomous loop selects
A4 over all other candidate loci — the proposer offered three and the other two were not evaluated.

The historical A4 effect is **not** quoted here for comparison, because there is no frozen autonomous
result to compare it against.

## The one change that would unblock it

A gate firing on empty-decode alone, without the IDK-phrase conjunction — i.e. `on_premature_idk`'s
condition minus `gate_match_any`. That is a **one-predicate change to the evaluator's host
capability**, not a change to AnchorOpt, and it should be a deliberate decision rather than something
smuggled in as part of a recovery run.
