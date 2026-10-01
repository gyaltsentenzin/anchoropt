#!/bin/bash
# ONE benchmark arm: (agent model, teacher config, domain) -> baseline, propose, evaluate, select.
#
# This is what LSF runs. It is deliberately a shell script rather than a Python entry point so the
# four phases stay separately restartable: if `evaluate` dies halfway, re-running the arm resumes from
# the results file instead of re-spending the baseline.
#
# Required: ARM_DIR, AGENT_MODEL, TEACHER, DOMAIN, ANCHOROPT_TAU2_REPO
set -uo pipefail

: "${ARM_DIR:?ARM_DIR is required}"
: "${AGENT_MODEL:?AGENT_MODEL is required}"
: "${DOMAIN:?DOMAIN is required}"
: "${TEACHER:=}"                       # empty = history-only
: "${ANCHOROPT_TAU2_REPO:?ANCHOROPT_TAU2_REPO is required}"
: "${ANCHOROPT_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${USER_MODEL:=claude-sonnet-5}"
: "${TASKS:=0}"          # 0 = the whole split, no subsampling
: "${SPLIT:=train}"       # tau-bench's OWN split; optimizing on train keeps test clean
# WHERE THE AGENT MODEL IS SERVED. `hosted` is the shared gateway; `vllm` is a model we serve
# ourselves. The point is not latency -- at the episode level that is only ~1.4x, because the user
# simulator is 58% of the time and stays on the gateway either way. The point is CAPACITY: an arm on
# `vllm` makes ZERO shared-gateway requests, so it frees gateway headroom for the other arms and can
# run outside the wave barrier without adding to the load that made connections drop.
: "${PORTAL:=hosted}"
export ANCHOROPT_TAU2_PORTAL="$PORTAL"
: "${MAX_ARMS:=8}"
: "${CONCURRENCY:=4}"
: "${MAX_STEPS:=200}"   # tau-bench's own DEFAULT_MAX_STEPS. At 30, 25/25 telecom episodes hit the
                       # cap and scored 0 by construction -- a measurement of the budget, not the model.
: "${TIMEOUT:=600}"        # per LLM call
: "${SIM_TIMEOUT:=3600}"   # per simulation wall clock; the queue has no RUNLIMIT to fall back on

# Credentials live in the tau-bench checkout's .env: HOSTED_API_KEY for the agent models, and
# OPENAI_BASE_URL/KEY for the gateway (user simulator, NL-assertion judge, external teacher).
set -a; . "$ANCHOROPT_TAU2_REPO/.env"; set +a
# The NL-assertion judge defaults to a gpt-4.1 this gateway forbids; retail tasks carry NL assertions,
# and an unscored simulation still counts in the denominator, so leaving it wrong reports a plausible
# fraction over a biased subset rather than failing.
export TAU2_LLM_NL_ASSERTIONS="${TAU2_LLM_NL_ASSERTIONS:-openai/claude-sonnet-5}"

PY="$ANCHOROPT_TAU2_REPO/.venv/bin/python"
export PYTHONPATH="$ANCHOROPT_REPO${PYTHONPATH:+:$PYTHONPATH}"
DRIVER="$ANCHOROPT_REPO/benchmarks/tau2/run_anchoropt_round.py"

mkdir -p "$ARM_DIR"
LOG="$ARM_DIR/arm.log"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
phase() { echo "$1" > "$ARM_DIR/PHASE"; say "PHASE $1"; }

say "arm: agent=$AGENT_MODEL teacher=${TEACHER:-history_only} domain=$DOMAIN"
say "     split=$SPLIT tasks=$TASKS max_arms=$MAX_ARMS concurrency=$CONCURRENCY user=$USER_MODEL"
say "     max_steps=$MAX_STEPS sim_timeout=${SIM_TIMEOUT}s call_timeout=${TIMEOUT}s portal=$PORTAL"
say "     host=$(hostname) dir=$ARM_DIR"
echo "{\"agent\":\"$AGENT_MODEL\",\"teacher\":\"${TEACHER:-history_only}\",\"domain\":\"$DOMAIN\",\"tasks\":$TASKS,\"max_arms\":$MAX_ARMS,\"started\":\"$(date -Is)\"}" > "$ARM_DIR/arm.json"

# ---- 1. baseline (frozen incumbent). Skipped if already recorded, so a restart is cheap. ----
if [ -s "$ARM_DIR/baseline.json" ]; then
  # A CACHED BASELINE PINS THE STEP BUDGET. Skipping the baseline is what makes a re-run cheap, but it
  # also means a newly-requested MAX_STEPS would be silently ignored: the arms would be measured under
  # the OLD budget, against an incumbent recorded under it, and the run would look like it honoured the
  # new one. Refuse instead and say what to do -- the step budget is the difference between measuring a
  # model and measuring a truncation.
  cached_steps=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('max_steps',''))"                  "$ARM_DIR/baseline.json" 2>/dev/null || echo "")
  if [ -n "$cached_steps" ] && [ "$cached_steps" != "$MAX_STEPS" ]; then
    phase FAILED_step_budget_mismatch
    say "cached baseline was recorded with max_steps=$cached_steps but this run asks for $MAX_STEPS."
    say "Re-running arms under a different budget than their incumbent is not a comparison."
    say "Reset this arm first:  scripts/stage_rerun.sh --apply --all"
    exit 14
  fi
  phase baseline_cached
else
  phase baseline
  "$PY" "$DRIVER" baseline --domain "$DOMAIN" --limit "$TASKS" --split "$SPLIT" --max-steps "$MAX_STEPS" \
        --concurrency "$CONCURRENCY" --timeout "$TIMEOUT" --sim-timeout "$SIM_TIMEOUT" \
        --agent-llm "$AGENT_MODEL" --user-llm "$USER_MODEL" --out "$ARM_DIR" >>"$LOG" 2>&1
  rc=$?
  if [ "$rc" = "4" ]; then
    phase FAILED_preflight
    say "preflight failed: an endpoint is unreachable. Nothing was spent; re-run this arm."
    exit 12
  elif [ "$rc" = "3" ]; then
    phase FAILED_baseline_void
    say "baseline episodes produced no outcome (endpoint outage?) -- NOT a result; re-run this arm"
    exit 11
  elif [ "$rc" != "0" ]; then
    phase FAILED_baseline; say "baseline FAILED (rc=$rc)"; exit 10
  fi
fi

# ---- 2. propose. No episodes run here; this is where the teacher acts. ----
phase propose
"$PY" "$DRIVER" propose --out "$ARM_DIR" ${TEACHER:+--teacher "$TEACHER"} >>"$LOG" 2>&1 \
  || { phase FAILED_propose; say "propose FAILED"; exit 20; }

# A round with no residual, or no arm built, is a legitimate end state and not a failure -- BUT a
# baseline whose episodes never ran is neither. `baseline` writes baseline_void.json and exits 3 in that
# case; treating it as "no arms to measure" is what let 16 of 24 arms report DONE through a gateway outage.
if [ -s "$ARM_DIR/baseline_void.json" ]; then
  phase FAILED_baseline_void
  say "baseline episodes produced no outcome (endpoint outage?) -- NOT a result; re-run this arm"
  exit 11
fi
if ! grep -q '"n_arms": [1-9]' "$ARM_DIR/arm_manifest.json" 2>/dev/null; then
  phase DONE_no_arms; say "no arms built -- nothing to measure; see arm_manifest.json"; exit 0
fi

# ---- 3. evaluate, paired against the one frozen incumbent. --resume so a restart is cheap. ----
phase evaluate
"$PY" "$DRIVER" evaluate --out "$ARM_DIR" --max-arms "$MAX_ARMS" \
      --concurrency "$CONCURRENCY" --resume >>"$LOG" 2>&1 \
  || { phase FAILED_evaluate; say "evaluate FAILED"; exit 30; }

if grep -q '"aborted_on_outage": true' "$ARM_DIR/results.json" 2>/dev/null; then
  phase OPEN_outage
  say "evaluate aborted on an endpoint outage -- genuine measurements kept, round left OPEN."
  say "re-run this arm to finish it: evaluate --resume picks up where it stopped."
  exit 13
fi

# ---- 4. select. Core's argmax over J_train on what was actually measured. ----
phase select
"$PY" "$DRIVER" select --out "$ARM_DIR" >>"$LOG" 2>&1 \
  || { phase FAILED_select; say "select FAILED"; exit 40; }

phase DONE
say "arm complete: $(grep -o '"outcome_class": "[A-Z_]*"' "$ARM_DIR/selection.json" 2>/dev/null | head -1)"
