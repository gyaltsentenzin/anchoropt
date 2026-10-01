#!/bin/bash
# Stock-pipeline check, qwen3.6 / retail / test. SIDE=ours|tau2runner; ours: validate --p0-only --k 2 in $DIR;
# tau2runner: GEPA's eval_prompt.py --stock, one run per seed in $SEEDS. Snapshot copy; do not edit.
set -uo pipefail
: "${ANCHOROPT_TAU2_REPO:?set ANCHOROPT_TAU2_REPO to your tau2-bench checkout}"
: "${ANCHOROPT_REPO:?set ANCHOROPT_REPO to your anchoropt checkout}"
set -a; . "$ANCHOROPT_TAU2_REPO/.env"; set +a
PY="$ANCHOROPT_TAU2_REPO/.venv/bin/python"
echo "side=$SIDE host=$(hostname) start=$(date -Is)"
if [ "$SIDE" = ours ]; then
  export TAU2_LLM_NL_ASSERTIONS=openai/claude-sonnet-5 PYTHONPATH="$ANCHOROPT_REPO" ANCHOROPT_TAU2_PORTAL=hosted
  "$PY" "$ANCHOROPT_REPO/benchmarks/tau2/run_anchoropt_round.py" validate --out "$DIR" --split test \
        --k 2 --p0-only --concurrency 3 --portal hosted --call-timeout 180 --retries 1 || exit 20
else
  : "${GEPA_SRC:?set GEPA_SRC to your gepa checkout's src/ directory}"
  export PYTHONPATH="$GEPA_SRC"
  for s in $SEEDS; do
    out="$DIR/stock_seed$s.json"; [ -s "$out" ] && { echo "skip seed $s"; continue; }
    echo "-- seed $s $(date -Is)"
    "$PY" "$(dirname "$GEPA_SRC")/experiments/tau2/eval_prompt.py" --model-name qwen3.6-35b-a3b --domain retail \
          --split test --stock --arm "stock_s$s" --seed "$s" --max-concurrency 3 \
          --artifacts-dir "$DIR/cells" --out "$out" 2>&1 | grep --line-buffered -vE "isn't mapped yet|^$" || exit 20
  done
fi
echo "done=$(date -Is)"
