# Core interface change: `runtime.states_at(boundary, states)`

**This is a correctness/interface change to the frozen core, recorded as such.** It adds one required
adapter method and no algorithmic behaviour. Committed separately from any experiment.

## What changed

`anchoropt/learning/structured_search.py::_expand_phi_at` now projects states onto the boundary it is
searching, and **fails closed**:

```python
project = getattr(runtime, "states_at", None)
if fields and not callable(project):
    raise RuntimeError(...)          # contract gap, not a licence to guess
if fields:
    projected = project(boundary, states)   # exceptions propagate
    if not projected:
        raise RuntimeError(...)       # empty information set != exhausted search
    states = projected
```

## Why, and why fail closed

WHAT search validated predicates on whatever states the caller passed — per-step, post-execution-shaped
— regardless of which boundary was being searched. A predicate checked against a result that does not
exist yet can look discriminating and then fire **zero times** at runtime. That is not hypothetical: it
is how WRITE1's search selected a commitment-gate controller reading `searched_other_container`, a
post-execution field.

A silent fallback would reintroduce exactly that defect while looking like success. Two distinct
failures are therefore raised rather than absorbed:

| situation | old behaviour | new behaviour |
|---|---|---|
| runtime declares fields but no `states_at` | validate on the caller's states | **raise** |
| `states_at` returns nothing | validate on the caller's states | **raise** |
| `states_at` raises | swallowed | **propagates** |

The second matters most: returning `[]` would surface as `SIGNAL_EXPANSION_EXHAUSTED` for a boundary
that was never searched — an unimplemented boundary reported as an exhausted one.

## What an adapter must now implement

```python
def states_at(boundary, states) -> list[dict]:
    """Project observed states onto ONE boundary's information set."""
```

* If your states are already per-boundary, return them unchanged.
* If a boundary sees something **narrower** — a commitment gate sees the proposed call but not its
  result — rebuild from only what is knowable there.
* Return `[]` rather than something approximate. Core fails closed, which is the safe default.

Required only for boundaries where `synthesis_fields` returns anything; a boundary declaring no fields
never reaches this.

## Verified

`tests/test_boundary_states_integration.py` (8 tests) intercepts what `_expand_phi_at` actually passes
to synthesis:

* the commitment gate receives one state per **proposed call**, carrying typed pre-dispatch facts;
* **no** post-execution fact reaches it (`best_similarity`, `tool_results`, `scored_entries`,
  `error_kind` all absent), tested with those facts deliberately present in the input;
* every field offered to synthesis is present in the states it is validated on;
* `post_execution` states pass through **unchanged**, so A9's validation is untouched;
* both fail-closed paths raise;
* real `c2inc` storage trajectories project 791 per-step → 543 upstream states.

## One equivalence, established by measurement

The projection reads `decoded`; the live hook reads the filtered `_mg_exec_calls`. They are equivalent
for this corpus: **no filter fired in any of the 791 storage-phase steps**. A test asserts it, so if a
filter ever does fire the projection must switch to capturing the real pre-dispatch list rather than
inferring it.
