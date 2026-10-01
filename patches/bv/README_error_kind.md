# `bv_supply_error_kind.py`

Closes the `error_kind` supply gap in the **live** cluster post-execution hook.

## Why

Six of the nine signals this host declares read `error_kind`. `_hook_state` builds the
post-execution information set as `dict(step_record)` plus `best_similarity`, `scored_entries`,
`searched_other_container`, `query` — and **never supplies `error_kind`**; `step_record` never
carries it either. Verified on the cluster by grep and by dumping a real prereq step's keys.

So five of the six can only answer False on this host, and `clear_proposed_at_capacity` survives
only on its `container_full` disjunct. Every round that reported "no candidate" for a capacity or
duplicate family was reporting an unsupplied field, not an absent mechanism.

This is the same declared-vs-supplied gap the evaluator's own comments already describe twice
(`container_full` at the gate, then the typed call facts) — one boundary further out.

## Apply

```
mkdir -p ~/pf_$$ && cd ~/pf_$$          # NOT /tmp: /tmp/inspect.py shadows stdlib inspect
python3 <repo>/patches/bv/bv_supply_error_kind.py \
        <remote-checkout>/anchoropt/memory_evaluator.py
```

Writes a one-time `.pre_error_kind` backup. **Idempotent, guarded on the MARKER it inserts** — not
on the anchor it inserts after. Guarding on the anchor is why `_CAPACITY_MARKERS` and
`_capacity_seen` each appear **three times** in the live evaluator (byte-identical, so the last wins
and behaviour is unaffected — but the pattern is the bug).

## Verified before shipping

- **Idempotence**: three consecutive runs → 1 marker, 1 anchor, valid syntax.
- **Classification**: 7/7 label cases correct; `error_kind` key **always present** (None for a clean
  result, because a predicate over an *absent* field silently answers False).
- **Fidelity**: **100% agreement with `adapter.error_kind` across 1771 real result strings** from the
  frozen H0 prereq trajectories of all three backends.
- **Conjuncts too**: `proposes_write` / `proposes_read` / `container` / `result` are supplied from the
  step's own decoded calls, or an `error_kind` match is necessary but not sufficient and the signal
  still cannot fire.

## Boundary discipline

Post-execution **only**. Nothing is carried forward; the commitment gate builds its own state
elsewhere and is untouched, so no post-execution fact reaches a pre-dispatch predicate. That
separation is deliberate: at the gate the equivalent value must be named `prior_error_kind`, because
`error_kind` means "the result of THIS call" and that call has not run yet.

## Expected effect (offline projection over frozen H0, same trajectories the cluster holds)

| backend | signal | before | after |
|---|---|---:|---:|
| rec_sum | `append_would_exceed_cap` | 0 | **278** |
| kv | `container_at_capacity` | 0 | **84** |
| vector | `container_at_capacity` | 0 | **39** |
| kv | `duplicate_identifier` | 0 | 4 |
| vector | `no_informative_result` | 0 | 2 |

Re-measure live after applying — these are projections, and the live hook is a different code path.
