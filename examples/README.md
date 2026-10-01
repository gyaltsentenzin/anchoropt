# Examples

## `self_evolve_tb2.py` — one AnchorOpt self-evolve round

### Start here: no setup required

```bash
python examples/self_evolve_tb2.py --dry-run
```

Runs the real structured search over saved diagnoses and prints the controller AnchorOpt would
install, plus every candidate it pruned and why. No Docker, no model endpoint, no TB2 clone.

Expected shape of the output: 11 diagnoses → ~108 candidates enumerated → ~15 surviving
`U_H(ℓ)` + Φ + attribution → 5 `SIGNAL_BLOCKED` → one selected `(ℓ, φ, µ, θ)`.

### Live mode prerequisites

```bash
uv venv .venv-dg --python 3.12
uv pip install --python .venv-dg/bin/python \
    "harbor==0.22.0" "deepagents==0.6.12" langchain-openai pytest
git clone --depth 1 https://github.com/laude-institute/terminal-bench-2.git /tmp/tb2_tasks
export OPENAI_BASE_URL=... OPENAI_API_KEY=...
```

Docker must be running (one container per task). An incumbent workspace must contain
`repo_baseline.py`, the `self_harness_harbor` wrapper, and the telemetry module — see
`integrations/self_harness/round2/SUBSTRATE_RECOVERY.md`, which records the two traps that cost
the most time: use `harbor_wrapper_rits.py` (not the pristine wrapper) or langchain routes to
`/v1/responses`, and warm the endpoint first because a cold model load can exceed 180 s.

```bash
python examples/self_evolve_tb2.py --cases cases.txt --workspace path/to/h0_compat --rounds 1
```

`--rounds N` loops `step()`. v0.1 does not claim multi-round convergence.

### Reading the outcome

| outcome | meaning |
|---|---|
| `accepted` | controller won the paired evaluation and was promoted |
| `rejected` | measured, did not clear the acceptance bar |
| `no_candidate` | every diagnosis `SIGNAL_BLOCKED`, or every cell pruned — a **representation** limit |
| `blocked` | not evaluable (infrastructure); denominator integrity failed |
| `selected_only` | search ran, no evaluator wired (this is `--dry-run`) |
