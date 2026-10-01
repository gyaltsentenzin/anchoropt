# Cluster patches — reproducible, not ad hoc

The GPU cluster checkout this project ran on was **not** this repository. It had its own divergent
copies of the evaluator and the sidecar (more evolved in places — its `_STEP_FIELDS` carried gate
telemetry this tree does not), it had **no** `benchmarks/` tree, and it had **no git remote**. So a
change needed on the cluster could not be delivered by checking out a branch there.

Every cluster-side change this experiment depends on is therefore an **idempotent script in this
directory**, committed on the experiment branch. Each one asserts what it expects before editing and
is safe to re-run: it reports "already patched" rather than applying twice.

| file | what it does | why the cluster needs it |
|---|---|---|
| `bfcl_declared_signals.py` | the host's 9 declared signals, **function bodies copied verbatim** from `benchmarks/bfcl_v4/bfcl_signals.py` | the cluster has no `bfcl_signals`/`bfcl_runtime`, so a controller naming a declared signal had nothing to evaluate |
| `bv_point_installer.py` | points the installer's `declared_signal` branch at that module | one definition, evaluated by both the optimizer and the runner |
| `bv_declared_signal.py` | teaches the installer to accept a `{"declared_signal": ...}` predicate | a Phi seeded from the residual family emits declared signals, not grammar atoms |
| `bv_patch_installer.py` | aliases `equals` to `eq` in `_OPS` | `signal_grammar` spells equality `equals`; 6 of 219 arms raised `unknown op` at install |
| `bv_capacity_fact.py` | supplies episode-scoped `container_full` at the commitment gate | `clear_proposed_at_capacity` reads it; without it the signal was unsatisfiable live |
| `bv_fix_capacity_unbound.py` | **apply after `bv_capacity_fact.py`** — reads `trajectory_history` instead of `execution_results` | `execution_results` is not a local at the gate; every prereq episode raised `UnboundLocalError`, the snapshot store never published, and both ARM jobs died `rc=1` while the controls ran clean |

### Verification probes (run these before trusting a run)

| file | proves |
|---|---|
| `bv_sig_probe.py` | the 9 declared signals load and `clear_proposed_at_capacity` fires on `container_full` AND on `error_kind == "no_capacity"` |
| `bv_install_probe.py` | the emitted spec installs, discriminates on four cases, and registers into the live dispatch |
| `bv_suppress_smoke.py` | the hook actually WITHHOLDS the clear (not merely registers): clear suppressed, write/read/clear-without-limit kept, mixed batch partially withheld, and no firing without a prior result |

## Applying them

Each script is idempotent and asserts what it expects before editing, so this is a sketch of the
shape rather than a runnable recipe against any specific host — adapt the paths to your own remote
checkout:

```bash
scp patches/bv/bfcl_declared_signals.py <remote>:<remote-checkout>/anchoropt/
scp patches/bv/bv_*.py <remote>:/tmp/
ssh <remote> 'cd <remote-checkout> && for p in bv_patch_installer bv_declared_signal bv_capacity_fact bv_fix_capacity_unbound bv_point_installer; do python3 /tmp/$p.py; done'

# then verify, from a directory that is NOT /tmp
ssh <remote> 'mkdir -p ~/sigprobe && cp /tmp/bv_*probe*.py /tmp/bv_suppress_smoke.py ~/sigprobe/ && cd ~/sigprobe && for p in bv_sig_probe bv_install_probe bv_suppress_smoke; do python $p.py; done'
```

Run them from a directory that is **not** `/tmp`: on at least one target cluster a `/tmp/inspect.py`
shadowed the stdlib `inspect`, and `dataclasses` failed with `module 'inspect' has no attribute
'signature'`.

## Why `bfcl_declared_signals.py` is a copy and not an import

The alternative was hand-translating each emitted controller into a field/op atom the installer
already understood. That was tried and rejected: the first attempt wrote
`proposes_clear AND container_full` and silently dropped the `error_kind == "no_capacity"` alternative
the real signal carries. A translation re-derived per controller is a place for exactly that drift.

`tests/test_offline_live_signal_equivalence.py` pins that the copy and the adapter's original agree,
including that case.

## Known limitation, recorded deliberately

`capability_id` is emitted on every spec and is **not read by the live dispatch** (0 references in
`memory_evaluator.py` / `runtime_hook.py`). For a cell with one executor there is no sibling to
mis-dispatch to, so the binding holds by uniqueness — which is why the H0 suppress runs are safe. It
does **not** hold at `post_generation_pre_exec/reprompt`, which has two executors (the injector and the
inert one). **Enforce the id in dispatch before evaluating any arm at a multi-executor cell.**
