# Known issues

Recorded rather than fixed, because fixing any of them would mean changing what a published number or
an acceptance verdict means, and that is a decision for the authors rather than a tidy-up.

---

## 1. Nine tests encode a pre-held-out-validation acceptance semantics

**Status:** failing on this branch, and on every branch in the lineage. **Not a regression introduced
by assembling this branch.**

```
tests/test_real_acceptance.py     (5 tests)
tests/test_two_round_lifecycle.py (4 tests)
```

### What fails

Each asserts that a controller reaches `ACCEPTED`, or that criterion 3 returns `PASS`. Criterion 3
returns `PENDING_VALIDATION` instead, so the round rejects and the assertions fail:

```
assert by_n[3] == "PASS"
E   AssertionError: assert 'PENDING_VALIDATION' == 'PASS'
```

### Why the production code is right and the tests are stale

Criterion 3 is *attribution + a causally validated mechanism, from the controller's own telemetry*.
[`acceptance_criteria.py`](../anchoropt/learning/acceptance_criteria.py) documents the intended
behaviour explicitly, at three separate points:

- line 20 — "A criterion can PASS, FAIL, or be unresolved. Unresolved is `PENDING_VALIDATION` and it is
  **NOT a pass**."
- line 32 — "So absent telemetry yields PENDING_VALIDATION, which **BLOCKS acceptance**."
- line 85 — "`PENDING_VALIDATION` is absent because **absent evidence is not** [a pass]."

The fixtures in these two files supply a controller without the telemetry criterion 3 now requires. The
gate therefore refuses to install it — which is the documented, intended behaviour.

Three independent confirmations that this is the current intent and not a defect:

1. **`tests/test_round_ledger.py:103` asserts the opposite and passes**:
   `assert rep.by_number(3).verdict == "PENDING_VALIDATION"`. Two test files encode contradictory
   expectations for the same criterion; the newer one agrees with the code.
2. **`tests/test_external_evaluation.py:502`** asserts `"ACCEPT" not in
   TRAIN_IMPROVED_PENDING_VALIDATION...` — the pending state is deliberately *not* an acceptance.
3. **`external_evaluation.py:213`** explains the naming: a train win "is named
   `TRAIN_IMPROVED_PENDING_VALIDATION` because train net > 0 is criterion 1 of four — the round has a
   provisional winner, not an accepted controller."

### Why they are left failing

**The failures are the gate being stricter than the old tests expect, not looser.** Nothing is being
silently accepted; the tests assert that an under-evidenced controller *should* be installed, and the
current gate refuses. That is the safe direction, and it is the direction the held-out validation
protocol was introduced to enforce.

Editing nine assertions to match would redefine, in the test suite, what `ACCEPTED` means — and the
acceptance rule is this project's central claim, not an implementation detail. `ACCEPTANCE_RULE.md`
warns specifically against substituting a rule chosen after seeing the data for the preregistered one,
in either direction.

**Two legitimate resolutions, for the authors to choose between:**

- **update the fixtures** so they carry the controller telemetry criterion 3 requires — this preserves
  the tests' original intent (an end-to-end four-criterion acceptance) under the current protocol, and
  is probably what was meant; or
- **update the assertions** to `PENDING_VALIDATION`, and add a separate test that supplies telemetry
  and reaches `ACCEPTED`, so the milestone stays covered.

Either way the reason should be recorded, because "a test was changed to green" and "a protocol got
stricter and its tests were brought along" look identical in a diff.

### Reproducing the diagnosis

```bash
pytest tests/test_real_acceptance.py tests/test_two_round_lifecycle.py -q   # 9 failed, 5 passed
pytest tests/test_round_ledger.py -q                                        # passes, asserting the opposite
```

Confirmed identical on the three `exp/*` branch tips. `adapter/appworld` appears green only because
these two test files do not exist on it.

---

## 2. `test_docs_links.py` walked in-tree virtual environments — **fixed here**

`_SKIP_DIRS` matched the directory name `.venv` exactly, so a venv created in-tree under any other name
(`.venv-harness`, `.venv-appworld-anchoropt`) was still walked. The suite then failed on a
*dependency's* own markdown — `dashscope` ships a skill example containing a markdown link whose target
is the literal placeholder word `url` — making the result depend on which packages happened to be
installed inside the checkout.

Fixed on this branch by skipping any path component beginning with `.venv`. Which third-party markdown
a checkout contains is not this repo's correctness.

---

## 3. Clone size is above the project's own target

`.gitignore` states the repo "stays clonable (<25 MB)". This branch is **36.6 MB** of tracked content
across 1,420 files, after dropping 7.3 MB of slide decks and 4 MB of τ² round artifacts.

Two directories are 29.3 MB of that, and both are load-bearing:

| area | tracked | why it stays |
|---|---|---|
| `benchmarks/bfcl_v4/` | 14.9 MB | split definitions under `data/` — the corpus the A1–A8 progression is measured over |
| `rounds/` | 14.4 MB | `scripts/verify_progression.py` and most of the integrity suite read nearly every directory in it |

Getting under 25 MB means moving the BFCL split data to a release asset or an LFS pointer, which changes
how Tier 1 is set up — a packaging decision, not a cleanup.

---

## 4. τ² runner parity is unresolved

AnchorOpt's τ² runner and stock tau-bench's runner disagree by roughly **1.5 tasks (z ≈ −1.0)** across
four runs. Until that closes, a τ² number from this adapter is not comparable to a published tau-bench
number. `TAU2_ADAPTER.md` §3 states the variance floor; the τ² results themselves are not in this
branch (see [`RESULTS_POLICY.md`](RESULTS_POLICY.md)).
