# Self-Evolve AnchorOpt v0.1 — one page

**What this is.** One autonomous round of decision-centric harness optimization: run an agent
harness on a benchmark, diagnose the failures, let AnchorOpt *choose a structured runtime
controller*, install it, evaluate it against the unmodified harness, and promote it if it wins.

**What makes it different from prompt search.** AnchorOpt never writes a patch. It selects a point
in a typed space and the runtime executes it:

$$(\ell, \phi, \mu, \theta)$$

- **ℓ** — the *decision boundary* it attaches to: `PRE_GENERATION`, `POST_GENERATION_PRE_EXEC`,
  or `POST_EXECUTION`
- **φ** — an *observable condition* from a declared vocabulary Φ, evaluated at ℓ
- **µ** — a *semantic action*: `NOOP`, `REPROMPT`, `SUPPRESS`, `REROUTE`
- **θ** — typed parameters for that action

Two independent gates bound the space before anything is measured. `feasible_actions(ℓ)` is
*structural* — `SUPPRESS` at `PRE_GENERATION` has nothing to cancel, in any runtime. `U_H(ℓ)` is
*operational* — what this host can actually execute there, declared by its adapter and always a
subset. A candidate failing either is pruned with a named reason, never silently dropped.

## The v0.1 loop

```
        ┌──────────────── BENCHMARK / RUNTIME (your harness) ─────────────────────────┐
        │  execute incumbent P_t   →  trials, verifier rewards, agent transcripts     │
        └───────────────────────────────────┬────────────────────────────────────────┘
                                            ▼
                        ResidualDiagnosis(case_id, evidence,
                                          failure_mechanism, consequential_decision)
                                            ▼
        ┌──────────────────────────── ANCHOROPT CORE ────────────────────────────────┐
        │  candidate loci around the causal step                                     │
        │  Φ-expressibility  ──not expressible──►  mark SIGNAL_BLOCKED, move on      │
        │        │ expressible                                                        │
        │        ▼                                                                    │
        │  enumerate (ℓ, φ, µ, θ)  →  filter feasible_actions(ℓ) ∩ U_H(ℓ)            │
        │  rank  →  select one candidate                                              │
        └───────────────────────────────────┬────────────────────────────────────────┘
                                            ▼
        materialize: controller → ControllerSpec + harness surface (one appended block)
                                            ▼
        evaluate: candidate arm vs control arm, identical everything else
                                            ▼
        AnchorOpt acceptance: net > 0, engagement non-zero, control clean
                                            ▼
        promote:  P_{t+1} = P_t ⊕ A_t   →  rerun  →  residuals R_{t+1}
```

**Deliberately out of scope for v0.1.** No automatic Φ-expansion — a diagnosis the current
vocabulary cannot express is recorded as `SIGNAL_BLOCKED` and skipped. No clustering, no borrowed
proposer, hook menu, or acceptance criterion from another harness. Those are the later
block-coordinate extension, where `SIGNAL_BLOCKED` cases become the trigger for Φ_t → Φ_{t+1}.

## Ownership

| layer | owner | why |
|---|---|---|
| execute, transcripts, verifier reward | your harness | working infrastructure, nothing novel to rewrite |
| trace normalization, causal diagnosis | your harness's diagnosis provider | `normalize_trace_steps` is already the step model AnchorOpt assumes |
| **controller space, search, acceptance, ledger** | **AnchorOpt** | the contribution |
| promote / branch bookkeeping | your harness's workflow layer | branch-advance semantics (supersede parent, depth+1) already exist in most harnesses |

AnchorOpt is the **driver**: it calls the benchmark round, never the reverse. The seam carries only
`ResidualDiagnosis` upward and a controller-as-surface downward. Hook names and step indices do not
cross it — a step index localizes a *region*, and choosing ℓ within that region is AnchorOpt's job.

## API

```python
from anchoropt.learning.self_evolve import step

result = step(incumbent, runtime=tb2, diagnose=provider.diagnose)

result.diagnoses            # tuple[ResidualDiagnosis]
result.candidates_considered# tuple[ScoredCandidate]  -- with prune reasons
result.selected_candidate   # Candidate | None
result.blocked_diagnoses    # tuple[(ResidualDiagnosis, "SIGNAL_BLOCKED: ...")]
result.evaluation           # EvaluationResult | None
result.accepted             # bool
result.new_incumbent        # Incumbent | None
```

`step()` is one round. Multiple rounds are a loop over `step()`; v0.1 does not claim multi-round
convergence.

## CLI

```bash
python examples/self_evolve_tb2.py --cases cases.txt --rounds 1
```

Needs: Docker, a TB2 task clone, an OpenAI-compatible endpoint. See `examples/README.md`.

## Reading the output

A round can end four ways, and they are **not** the same:

| outcome | meaning |
|---|---|
| accepted | controller won on the paired evaluation and was promoted |
| rejected | controller was measured and did not clear the acceptance bar |
| no candidate | every diagnosis was `SIGNAL_BLOCKED`, or every candidate was pruned by `U_H(ℓ)` |
| blocked | the round could not be evaluated (infrastructure) |

"No candidate" is the interesting one for research: it means the *representation*, not the policy,
is the limit — precisely the signal that Φ needs to grow.

---

## Status: what v0.1 ships, and what is still missing

### Implemented for v0.1

| piece | file | note |
|---|---|---|
| structured candidate search | `anchoropt/learning/candidate_search.py` | **new** — the piece that makes the round autonomous |
| `step()` / `StepResult` | `anchoropt/learning/self_evolve.py` | **new**, appended; `run()` untouched |
| controller materializer | `benchmarks/tb2_deepagents/tb2_materialize.py` | **new** — was an ad-hoc script; append-only, verified 0 removed lines |
| signal aliases | `benchmarks/tb2_deepagents/tb2_signals.py` | **new** `SIGNAL_ALIASES` + `signal_aliases()` on the adapter |
| `failure_mechanism` alias | `anchoropt/runtime.py` | property alias so the documented schema name works without renaming a field the provider and frozen artifacts use |
| collaborator example | `examples/self_evolve_tb2.py` | `--dry-run` needs no Docker, endpoint, or TB2 clone |
| tests | `tests/test_candidate_search.py` | 13 tests, each pinning a defect found during development |

Reused unchanged: `anchor.py`, `runtime.py`'s `HostProfile`/`U_H(ℓ)`, `validate_candidates`,
`ControllerSpec`, `build_middleware`, the TB2 adapter, the diagnosis provider, `evidence_ledger`,
`phase_switch`, `exposure`, `ACCEPTANCE_RULE`. **No AnchorOpt core component was replaced.**

### Missing before v0.1 is collaborator-runnable end-to-end

1. **`evaluate` and `promote` wiring.** `step()` accepts both as injected seams and works without
   them (`selected_only`). A hardened runner and a branch-advance implementation like
   `create_branch_from_surfaces` exist in principle but are not yet joined to `step()`. **This is
   the one genuinely blocking gap.**
2. **Live diagnosis.** `--dry-run` reads the frozen H1 brief. Live mining needs a per-round
   diagnosis driver; `run_live()` deliberately raises rather than duplicating one ad hoc.
3. **A committed incumbent workspace.** The H0 surface is assembled ad hoc in `/tmp`; a collaborator
   needs a documented `make workspace` step (surface + wrapper + telemetry + the endpoint wrapper).
4. **`examples/README.md`** with the endpoint/Docker prerequisites.
5. **Dev/held-out split** in `_default_accept`. v0.1 has one split, so the no-regression term is
   absent — the reason v0.1 is a plumbing milestone, not an efficacy claim.

### Honest limits of the v0.1 search

The Φ-expressibility test is **keyword matching over declared signal names plus adapter-declared
aliases.** It is not semantic, and calibrating it exposed three real defects, all now pinned by
tests: a single generic token (`response`) matched an unrelated failure; single verbs
(`concluded`) matched incidental prose in a DAG-reasoning error; and scoring attribution instead of
gating it let an unimplicated candidate outrank well-attributed ones. It currently marks 5 of the
11 frozen H1 residuals `SIGNAL_BLOCKED`, which matches the earlier hand analysis (domain-reasoning
and environment-infeasibility cases).

Treat `SIGNAL_BLOCKED` counts as **approximate**. The principled replacement is the
`consequential_decision` → observable mapping that Φ-expansion builds; until then this heuristic is
a stand-in that is deliberately visible rather than hidden behind a confident-looking score.
