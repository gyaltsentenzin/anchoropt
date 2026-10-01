# Vendored code in this directory

Three things here are **copies**, not original work. They are vendored so that
`python benchmarks/bfcl_v4/run.py --compare` can actually run instead of pointing at a repo you may
not have.

| path | origin | licence |
|---|---|---|
| `harness/` | [Gorilla / Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla), **modified fork** | Apache-2.0 (`harness/LICENSE`) |
| `evaluator/`, `run_memory_eval.py`, `run_memory_train.py` | the AnchorOpt working repo (`decision_ops_agentic/anchoropt`) | same as this repo |
| `data/` | BFCL v4 Agent Memory corpus | as upstream |

## `harness/` is a modified fork — that matters

It is **not** stock BFCL v4. The published A1–A4 numbers depend on the modifications, so replacing it
with an upstream checkout will not reproduce them. The differences that are load-bearing:

- **The empty-memory `ZeroDivisionError` fix.** `memory_kv._similarity_search` called `BM25Plus([])`
  on a search over empty memory, which divides by average document length. It reached both
  `core_memory_key_search` and `archival_memory_key_search`. This is a genuine upstream bug, and it
  was **asymmetric between arms** — a policy that instructs "search archival before concluding
  unavailable" induces its own penalty, 4× more firings under the treatment than the control. Any
  result measured before this fix understates the treatment.
- **A diagnostic-only correction to the kv key-format error message** (0 semantic change). The
  validator rejects `a1c_level` because its first segment contains a digit, while the message says
  "cannot contain spaces" — so the model retried the same key 19 times and never learned. The
  constraint is unchanged; only the message now describes it.
- **Memory gates** in `bfcl_eval/model_handler/memory_gates.py`, the interception points the anchors
  act at.

## Two local deviations from upstream, both recorded

Beyond the fork's own changes above, this copy differs from what was vendored in exactly two ways —
both to satisfy IBM secret scanning, neither touching the memory path:

1. **Non-memory datasets removed.** `bfcl_eval/data/` went from 12 MB to 468 KB. The multi-turn and
   live categories contain ~40 synthetic passwords in their fixtures; the memory task never reads them.
   What remains is exactly what it does read: `BFCL_v4_memory.json` (155 entries),
   `possible_answer/BFCL_v4_memory.json` (155, matching), and `memory_prereq_conversation/`.
   Re-vendor from upstream if you need another category.
2. **`posting_api.py`** hardcoded a dummy credential literal for a mock social-media API. It is now
   derived from the mock username and overridable via `BFCL_MOCK_POSTING_PW`. **Behaviour is
   identical** — the computed default equals upstream's literal, verified by a runtime assertion
   rather than by trusting the edit.

Requesting a scanning exemption to publish unused third-party fixtures would have been the wrong
trade; carrying less is better than carrying a bypass.

## The manually tuned gates are RETIRED — with credit to the prior work

### What they were

The harness arrived carrying ~30 hand-authored memory gates (G1/G3/G4/G5, `on_turn_start_action`,
`on_blob_pressure`, `on_premature_idk`, …) and the whole gate-dispatch machinery — the `GateSpec`
abstraction, `suppress_specs`, `gate_match_any`, the signal ladder, and a generic executor that makes
a gate installable as pure registry data with no per-signal wiring. That is earlier hand-authored
work this project builds on, and **A1–A4 fire through it.** It is not scaffolding that was replaced;
it is the substrate the learned anchors run on.

That groundwork established the premise everything here rests on: **a local intervention at a specific
decision point can move this benchmark** — hand-tuned anchors reached 58.67 % where a global prompt
reached 41.33 %, which is what made "locality is where the value is" worth pursuing. It also identified
the right decision points and action vocabulary; this repo's incision-point taxonomy formalises
distinctions that machinery already drew.

> **Those numbers are on a different split and are NOT directly comparable to A1–A4's.** 58.67 % /
> 41.33 % come from the earlier single-domain healthcare holdout (n = 75). A1–A4's 29.04 % → 42.24 %
> is the 303-query train side of the balanced cell fold. Different corpora, different denominators —
> exactly the cross-job comparison this project forbids elsewhere. Read 58.67 % as "locality can be
> worth a lot on the corpus it was measured on", never as a bar A1–A4 fell short of. (The A1–A4 figures
> are kept here because they are what this section compares; the accepted stack has since grown to
> A1–A5 + A7 + A8 at 47.52 % on the same fold — see the top-level README.)

What hand-authoring could not settle is **how to find such interventions systematically** — each gate
came from a person reading trajectories and forming a hypothesis, so the method did not generalise past
the author's attention. A1–A4 makes that a procedure: mine the residual, attribute to a decision,
enumerate admissible actions, install only what survives a paired counterfactual against pre-registered
criteria.

### Why they are not enabled here

Keeping them on would confound the two questions. **A "control" carrying a hand-tuned gate is not a
control for a learned one**, and the delta would no longer isolate what the learning procedure
contributed. They are retired, not repudiated.

### How the retirement works, and what it is verified to do

`MEMORY_GATE_REGISTRY` now contains **exactly four** entries — the dispatch A1–A4 needs:

| key | anchor |
|---|---|
| `on_domain_error_core_full` | A1 — opens the reroute block on a capacity error |
| `on_core_full_rerouted` | A1 — the reroute itself |
| `on_domain_error_key_not_found` | A2 — advise searching archival after a failed read |
| `on_redundant_write_suppressed` | A3 — cancel a normalised-identical re-write |

A1's repair and A4's trigger are code paths (`enable_capacity_repair` / `enable_reroute` /
`enable_zero_call_reprompt`), not registry entries, so they are unaffected.

`base_handler.py` still holds ~26 `REGISTRY_BY_KEY["on_premature_idk"]`-style **bracket** lookups at the
retired call sites, and a bare `KeyError` there would crash mid-episode. Deleting 26 references from an
1854-line upstream file we do not own is the riskier option — it edits the exact path every frozen result
depends on. So a retired key resolves to an **inert spec** (`_RetiredGates.__missing__`): empty trigger
substrings, `backends=("__retired__",)`, `default=False`. Verified:

| | before retirement | now |
|---|---|---|
| gates in registry | 34 | **4** |
| bare policy suppresses | `on_core_clear_blocked` (default-ON) | **nothing** |
| retired key lookup | returns a live spec | inert — `gate_applies` and `gate_match_any` both `False` |
| A1–A4 policy suppresses | `on_redundant_write_suppressed` | unchanged |

`run.py` additionally refuses any condition whose policy omits `gate_default: false`, and tests assert
the registry holds exactly those four keys and that no retired key can fire.

A stale comment to ignore: `base_handler.py` still says G1 "hard-blocks" the clear. That code was
generalised to iterate `suppress_specs(load_anchoropt_policy(), …)` long ago; the comment describes
what it used to do.

## The vendored evaluator is the PRE-A5 state — deliberately

Two files here differ from the working repo's current copies, and the direction matters:

| file | lines only upstream | lines only here | what the additions are |
|---|---|---|---|
| `evaluator/memory_evaluator.py` | 263 | 3 | an `ANCHOROPT_DESTRUCTIVE_AUDIT`-gated live-state audit, plus `on_destructive_authorization` and `on_kv_redundant_write_suppressed` handling |
| `evaluator/traj_sidecar.py` | several | **0** | telemetry fields marked `A5-v2` in the source |

Both are **strict predecessors**: the working repo has moved on to A5/C2/D-era work, and none of it
exists in A1–A4. The three lines that appear only here are the *older form* of blocks that upstream
later extended for those post-A4 gates. Checked directly: **no upstream change touches the A1–A4
execution path** (`capacity_repair`, `enable_reroute`, `on_redundant_write_suppressed`,
`enable_zero_call_reprompt`).

So this is the right code for reproducing A1–A4, and re-syncing from the working repo would be a
regression for that purpose — it would score the frozen anchors under machinery built for later rounds.

### Consequence: the store fingerprint will not match the published pin

The published runs pinned `ANCHOROPT_STORE_CODE_FINGERPRINT=8f1c04ae2b73d915`. Because the
store-building code here differs (by exactly the post-A4 additions above), this repo computes
`fc391e2af7751293` instead, so the shared snapshot cache **misses** and a fresh store is built.

A fresh store is a *different world* — the model reasons over memory contents that were generated
independently — so per-case results can differ from the published run even with an identical policy.
Observed on the first full reproduction: the control arm scored **90/303 (29.70 %)** against the
published **88/303 (29.04 %)**, a 2-case difference.

**What is and is not affected — and I had this wrong at first.** I initially wrote that both conditions
in a `--compare` run "share the same freshly built store". **They do not, and they must not.** All four
anchors are declared `pre_snapshot` (see `GateSpec.snapshot_anchor`), meaning each can alter what the
storage phase *produces* — A1 reroutes writes, A3 suppresses them. So the store cache key includes
`gate_config_fingerprint`, and control and anchors legitimately build **different worlds**. Verified:
both full runs show two distinct `base_key` values.

That is correct behaviour, not a flaw: scoring an anchors arm against a store built without the anchors
would measure the wrong counterfactual. But it means the paired comparison here is **not** "same prefix,
one intervention" in the strict sense the README's §2.4 describes. The intervention changes the prefix,
by design, and the pairing is at the level of *case identity* rather than shared world state.

Practical consequence: read the delta as "policy A vs policy B, each in the world it creates" — which is
the right question for a storage-affecting anchor, and a weaker claim than a shared-prefix A/B.

**The fingerprint is a feature, and it fired usefully.** Fixing A3's import edited
`memory_evaluator.py`, which moved the store fingerprint `fc391e2af7751293` → `d4f3ede6597f6920` and
correctly invalidated the cache. A store built while A3 was over-suppressing **must not** be reused
under a corrected A3 — its contents were shaped by the bug. So a code change that can alter what a
prereq build *produces* forces a rebuild, which is the whole point of hashing the store code. The cost
is ~40 min; the alternative is silently scoring a fixed policy against a world the broken one built.

## Not linted, not reformatted

`harness/` and `evaluator/` are excluded from `ruff` (see `pyproject.toml`) and from the docs-link
tests. Linting them would mean rewriting code we do not own, or code the frozen results depend on, and
would destroy the diff against the source. **Fix bugs upstream and re-vendor** rather than editing here.

## Dependencies

The harness's own `pyproject.toml` declares 32 dependencies, including SDKs for OpenAI, Anthropic,
Mistral, Cohere, Google and Writer. **The memory task touches none of them.** Measured by importing
each backend into an empty environment:

| backend | needs |
|---|---|
| `memory_kv` | `overrides`, `filelock`, `rank_bm25` |
| `memory_rec_sum` | nothing beyond the above |
| `memory_vector` | **+ `sentence-transformers`** (the only heavy one) |

So `pip install -e ".[bfcl]"` gets you kv and rec_sum anywhere; add `".[bfcl-vector]"` for the
embedding stack. Generating trajectories additionally needs a served model — see README §3.4.
