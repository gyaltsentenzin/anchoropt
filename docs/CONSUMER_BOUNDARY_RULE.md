# Validate an anchor at its consumer boundary, not at its gate boundary

Adopted 2026-08-26, after the A9 delivery defect. Applies to every anchor in the line, present and
future.

> **An anchor's effect is established only by reading it back from the object its consumer actually
> reads. Gate-side telemetry establishes *intent*. It never establishes *effect*.**

## Why this needed to become a rule

AnchorOpt already had an arm-identity rule: *only firing telemetry identifies an arm* — because flags,
filenames and fingerprints once all agreed while the behaviour differed. A9 showed that rule has a second
half.

**Firing telemetry can be right about itself and wrong about the world.**

A9 computed a merged retrieval payload and positionally replaced an entry. It recorded
`xcm_replaced=True`. That was **true** — of `step_record["tool_results"]`, which is a telemetry dict. The
model reads `inference_data["message"]`, assembled from `execution_results`. Same intent, different
object, **no delivery**.

The gate ran ~190 lines below the point where the prompt is assembled. Its write to the real list was
dead code: immediately followed by `continue`, so the loop restarted and rebuilt the list before anything
consumed it.

**78 firings computed. 0 delivered.**

## What it cost

| artefact | what it concluded | status |
|---|---|---|
| A9-R accuracy | +0.00 pp → "the model won't use the evidence" | **VOID** — evidence never delivered |
| A9-R retrieval positive | gold-in-context 31/78 → 50/78 | **UNVERIFIED** — measured on `tool_results` |
| delivery audit | "verbatim in the actual model input" | **RETRACTED** — audited the wrong object |
| A9-U | comprehension / grounding limit | **VOID** — a re-read of absent evidence |
| A9-R-top1 | dilution ruled out | **VOID** — top-1 of a discarded payload |
| two-model probe | framing, not capability | **VOID PREMISE** — fed a passage the run never sent |
| sysprompt localisation | system prompt innocent | **VOID PREMISE** — same |

Four experiments and roughly a week of GPU time, all downstream of **one unverified substitution**: the
telemetry dict was read and called the model input. Every subsequent result was internally consistent and
collectively meaningless.

## The tells, and why they were missed

Two signals were available before any of the void work was done.

**1. The arm scored *exactly* the control.** 35.0% against 35.0%, to two decimals, across two independent
control runs. An anchor that fires 78 times and moves nothing at all is far more likely to be inert than
to be perfectly neutral. **Exact equality is a defect signature, not a null result** — a genuine null
looks like ±1–2 cases of noise, not bit equality.

**2. Every reconstruction succeeded while the real system always failed.** A probe fed the retrieved
passage to the model ten different ways — plain, tool-shaped, and under the full 11,208-char harness
system prompt — and it answered correctly **ten for ten**, while the real run always failed.

*"My reconstruction always succeeds and the real system always fails"* is not a finding about framing or
comprehension. It is a finding that **the real system is not receiving my input.** The response was to
generate two mechanism hypotheses instead of questioning the premise.

## Why gate-side checks cannot catch this

A gate-side check asks *did I write the thing?* — and the answer was yes. What it cannot ask is *did the
write reach the object that gets consumed?* Those differ whenever:

- there is more than one representation of the same data (a live list and a telemetry copy)
- the write happens at a different point in the loop than the read
- an early `continue` / `return` / retry rebuilds state after the write
- two index checks run against two lists that need not be the same length

That last one is the quiet killer. It lets telemetry record a positional replacement while the real path
takes an append fallback, and **both records are locally honest**.

## The practice

1. **Name the consumer.** For each anchor, write down the exact object its effect must reach: the model's
   message list, the store's live state, the scored answer. Not "the harness".
2. **Read it back from there.** Capture the consumer-side object and search it for the payload. The check
   should be end-of-pipeline and slightly dumb — a substring probe on the delivered text beats a
   structural assertion about indices, because an assertion about indices is exactly what failed.
3. **One authoritative object; derive the rest.** A9's fix made `execution_results` authoritative and
   `tool_results` a derived copy. Two independently maintained representations of the same fact will
   eventually disagree, and the disagreement will favour whichever one you are looking at.
4. **Distinguish *not measured* from *not delivered*.** If the consumer-side object was not captured for
   some firings, **abstain** on those and report them separately. A9's first Gate 0 pass scored
   unattested firings as failures and produced "55/66 delivered, 0 violations" — self-inconsistent, and
   the original error in mirror image. Then constrain where abstentions may occur (A9's are prereq-phase
   only) so a real gap cannot hide among them.
5. **Treat exact equality with the control as a defect alarm.** Before writing up a 0.00 pp result,
   confirm the anchor was computationally live: firings > 0 *and* a consumer-side attestation.

## Audit status

One demonstrated gap, now fixed: **A9** (`docs/../rounds/T9_A9_xcontainer_merge/README.md`).

One exposure left unpatched: **G3**, which had 0 firings in the captured run, so there is nothing to
attest and nothing to correct — recorded rather than closed.

The other anchors in the stack are structurally less exposed, because their consumer is the **store**
rather than the prompt, and their effect is read back from live state by design: A5 and A8 verify against
the live container after acting, and A1's reroute is confirmed by the write landing. That is a reason for
lower risk, not a proof of absence — the rule applies to them too.

---

Related: [`ACCEPTANCE_RULE.md`](ACCEPTANCE_RULE.md) (Gate 0 runs before criteria 1–4, because evidence
validity precedes evidence), and `A9_EVIDENCE` in [`../rounds/anchors.py`](../rounds/anchors.py).
