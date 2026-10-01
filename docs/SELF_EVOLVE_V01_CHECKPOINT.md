# Checkpoint: AnchorOpt Self-Evolve BFCL v0.1 architecture

Tag: `selfevolve-bfcl-v0.1`  ·  commit `61d1838`  ·  2026-09-17

Verified at this commit:

| gate | result |
|---|---|
| working tree | clean (no modified, no untracked) |
| test suite | **721 passed** (was 652 before this work) |
| anchor recovery | **7/8 exact, 1/8 approximate, 0 no** |
| replay expressibility | **8/8** historical triggers compile from the declared alphabet |
| existing BFCL adapter | **unmodified** — asserted by test, not by inspection |

---

## Provenance warning: read this before `git log`

Three commits carry this architecture, and **two of them are titled as TB2 work**. A parallel
session was committing a TB2 rerun on this branch during the same hours, and its `git add` swept up
whichever architecture files happened to be on disk at the time. Nothing was lost and nothing was
overwritten, but the commit messages do not describe the files they contain, and `git log` for this
architecture is therefore misleading on its own.

| file | committed under | that commit's subject is about |
|---|---|---|
| `bfcl_runtime.py`, `bfcl_capabilities.py`, `bfcl_signals.py` | `ecb6356` | TB2 round-2 re-read |
| `fixtures/replay_anchors.py`, `fixtures/replay_exprs.py` | `ecb6356` | TB2 round-2 re-read |
| `learning/signal_lang.py`, `tests/test_signal_lang.py` | `ecb6356` | TB2 round-2 re-read |
| `learning/candidate_search.py` (3 defect fixes) | `ecb6356` | TB2 round-2 re-read |
| `learning/round_runner.py`, `scripts/anchor_recovery.py` | `d86513c` | TB2 timeout diagnosis |
| `tests/test_bfcl_runtime.py` | `d86513c` | TB2 timeout diagnosis |
| `learning/block_loop.py`, `learning/proposal_seams.py` | `61d1838` | **this architecture** |
| `tests/test_block_loop.py`, `tests/test_proposal_seams.py` | `61d1838` | **this architecture** |

**This file plus the tag is the provenance record.** To see the architecture as a unit, use the tag,
not the log.

Also note the branch moved: this work began on `decision-architecture` and the tree is now on
`main`, again from the parallel session. The tag pins the content regardless.

---

## What v0.1 is, in one paragraph

A residual-guided block-coordinate loop. The current incumbent is executed, its failures are
attributed to a causal region, structured controllers are proposed over a closed typed space,
AnchorOpt decides which proposals are *legal*, the survivors are paired-evaluated under a
denominator-integrity precondition, and the best measured one is promoted — after which the residual
is **re-mined against the new incumbent**. Two blocks alternate: POLICY holds Φ frozen and searches
(ℓ, μ, θ); SIGNAL proposes a new declarative φ, validates it, freezes Φ again, and returns.

## The layering, and what each layer may not do

```
capabilities   bfcl_capabilities.py   observable fields, carried summaries, tool schema
runtime        bfcl_runtime.py        U_H(l), evaluate_signal, apply_action, normalize
language       signal_lang.py         phi as DATA: expression tree, compiled by us
seams          proposal_seams.py      attributor | proposer | AnchorOpt legality
loop           block_loop.py          POLICY / SIGNAL, phase switch, top-K, promote
evaluation     round_runner.py        paired arms, denominator gate, moving incumbent
fixtures       fixtures/              A1-A9 answer key -- CALIBRATION ONLY
```

* the **attributor** may not name ℓ, φ, μ, θ, support, or linked loss — enforced by schema
  (`_FORBIDDEN_ATTRIBUTOR_KEYS`), because a prompt asking it not to is not a constraint
* the **proposer** may not author a new φ during a POLICY round, and may not name a field, operator,
  tool, or action outside the declared vocabulary
* **AnchorOpt** decides legality, the phase switch, and (via measurement) which controller wins
* the **fixtures** must never be imported by the autonomous loop

## Defects fixed in the core, found by the recovery calibration

Recovery began at 2/8. Three were real bugs in `candidate_search`, not in BFCL:

1. **ties fell through to the alphabet** — `select` ordered survivors lexicographically after
   control strength, so the alphabetically-first boundary and signal won regardless of evidence. It
   selected `container_at_capacity` for a retrieval-similarity failure: an unattributable pick of
   exactly the kind the attribution gate exists to prevent, arrived at one step later.
2. **a parameterized signal could never be selected** — enumeration probed with `{}`, a required
   param raised `ValueError`, and the cell was recorded `signal_not_observable_at_boundary`. The
   reason was also wrong: the signal *is* observable there, it merely needed configuration. Cost A9
   its recovery entirely. Fixed with `probe_params` + a distinct `signal_params_unconfigured` reason.
3. **evidence must outrank control strength** — a candidate matching one incidental phrase beat the
   correctly-attributed controller matching three, because the fixed vocabulary weight was checked
   first. Control strength describes the *vocabulary*; evidence describes *this failure*.

A fourth was mine, in the recovery harness: it handed every REROUTE an invented destination, so the
strongest action survived at every locus. Fixed by asking the runtime, which returns `None` where
nothing is attested — and `None` prunes the cell.

## S5 gained a third gate

A locus no declared signal can observe was clearing S1 on coverage alone and sending the loop into a
POLICY round with nothing legal to propose. That is reported as `no_candidate`, which reads as policy
exhaustion and hides that the real limit is the **representation** — the one distinction S5 exists to
preserve. The gate now asks expressibility per locus.

## Deliberately not fixed

**A4 recovers as `approximate`, not `exact`.** The search localizes (ℓ, φ) exactly and picks SUPPRESS
where the accepted anchor uses REPROMPT; the two are separated only by the frozen control-strength
ordering. The only available reason to prefer REPROMPT is knowing A4's measured outcome, which is
tuning the search to force recovery. Left as a reported near-miss.

## What v0.1 is not

* **not an efficacy claim.** No live LLM attributor or proposer is wired at this commit; the
  end-to-end verification used mocks. Nothing here has optimized anything on a real benchmark.
* **not a discovery result.** Recovering known-good controllers shows the representation can *hold*
  them. It is silent on whether the loop can find new ones.
* **linked downstream loss is not implemented.** `compute_support` groups diagnoses by consequential
  decision — a real count, but the weaker measure. A1's ranking used linked loss, and the gap is
  stated rather than papered over with a plausible integer.
* **one split.** The dev/held-out no-regression term is still absent from `_default_accept`.
