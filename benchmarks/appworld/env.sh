# Site configuration for the AppWorld adapter. Source it; do not execute it.
#
#   source benchmarks/appworld/env.sh
#
# WHY THIS FILE EXISTS
# --------------------------------------------------------------------------------------------------
# Every path the adapter needs was previously hardcoded to one machine's home directory across many
# tracked files. That is invisible to the original developer and fatal to anyone else: the scripts
# do not fail with "configure this", they fail with a missing directory, or worse, silently mine
# the wrong tree.
#
# Each variable below is overridable. The two that have no sensible shared default (APPWORLD_PY,
# APPWORLD_ROUNDS -- see point 3 below) fail loudly if unset rather than silently falling back to
# whoever last developed this; the rest fall back to a reasonable per-user default.
#
# WHAT IS GENUINELY SITE-SPECIFIC, STATED PLAINLY
# --------------------------------------------------------------------------------------------------
# Three things here are NOT paths and cannot be fixed by setting a variable. They are documented in
# docs/REPRODUCING_EXTERNALLY.md and summarised here so nobody discovers them the hard way:
#
#   1. LSF + GPFS. The measurement code does not need a scheduler; only cluster job-submission
#      wrappers do, and those are not shipped. Anything that can run a command with GPUs and an
#      OpenAI-compatible endpoint works.
#   2. An inference gateway behind an auth proxy. Our setup needs a per-model proxy on a fixed
#      port, because that gateway wants a non-standard auth header. Any OpenAI-compatible
#      base_url substitutes -- set it in the model config, not here.
#   3. The frozen incumbents. These are DATA, not code: the `train`/`sh_heldout` baselines live in
#      round cells from an earlier internal harness, with per-task logs that `mine_residual` reads.
#      They cannot be shipped (a single run's logs are 80-538 MB) and they are not regenerable from
#      this repo alone. An external reproducer must run the stock agent themselves to create an
#      incumbent. That is the real barrier to reproduction, and no amount of path templating removes it.

# The repo itself. Derived from this file's location, so it is correct wherever the repo is cloned.
ANCHOROPT_REPO="${ANCHOROPT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ANCHOROPT_REPO
export APPWORLD_ADAPTER="$ANCHOROPT_REPO/benchmarks/appworld"

# The AppWorld checkout: needs `data/` and `experiments/{configs,prompts,code,outputs}`.
# `appworld.common.path_store` defaults its root to os.getcwd(), so leaving this unset does not merely
# fail -- it mines whatever directory you happen to be in and writes a run tree there.
export APPWORLD_ROOT="${APPWORLD_ROOT:-$HOME/appworld}"

# Python with `appworld` and this repo importable. Ours lived in a sibling project's venv, which was
# an accident of how the two projects grew, not a requirement: any interpreter that can import both
# works. No fallback is provided -- set this to your own interpreter.
export APPWORLD_PY="${APPWORLD_PY:?set APPWORLD_PY to a python with appworld and anchoropt importable}"

# Where the frozen incumbents (baseline round cells) live. See point 3 above: this is data, and it
# has no shippable fallback -- set this to wherever you generated your own baseline round cells.
export APPWORLD_ROUNDS="${APPWORLD_ROUNDS:?set APPWORLD_ROUNDS to your baseline round-cell tree}"

export PYTHONPATH="$ANCHOROPT_REPO:$APPWORLD_ADAPTER${PYTHONPATH:+:$PYTHONPATH}"

# Fail loudly and early rather than halfway through a scoring run.
appworld_env_check() {
  local bad=0
  [ -x "$APPWORLD_PY" ] || { echo "env.sh: APPWORLD_PY not executable: $APPWORLD_PY" >&2; bad=1; }
  [ -d "$APPWORLD_ROOT/data" ] || { echo "env.sh: no data/ under APPWORLD_ROOT=$APPWORLD_ROOT" >&2; bad=1; }
  [ -d "$APPWORLD_ROOT/experiments/configs" ] || {
    echo "env.sh: no experiments/configs under APPWORLD_ROOT=$APPWORLD_ROOT" >&2; bad=1; }
  "$APPWORLD_PY" -c 'import appworld, anchoropt' 2>/dev/null || {
    echo "env.sh: $APPWORLD_PY cannot import both appworld and anchoropt (PYTHONPATH=$PYTHONPATH)" >&2
    bad=1; }
  [ -d "$APPWORLD_ROUNDS" ] || {
    echo "env.sh: APPWORLD_ROUNDS=$APPWORLD_ROUNDS not found -- the frozen incumbents live there." >&2
    echo "env.sh:   Without them you can run the adapter's offline tests but cannot mine a residual" >&2
    echo "env.sh:   or score an arm. See docs/REPRODUCING_EXTERNALLY.md." >&2; bad=1; }
  [ "$bad" = 0 ] && echo "env.sh: ok (APPWORLD_ROOT=$APPWORLD_ROOT, APPWORLD_PY=$APPWORLD_PY)"
  return "$bad"
}
