#!/usr/bin/env bash
# Live progress watcher for the task-sensitivity evaluation.
# Runs alongside run_task_sensitivity_eval.sh and appends a one-line
# progress snapshot every 15 seconds to a dedicated progress log.
#
# Usage:
#   bash LeakageEvaluationFramework/scripts/watch_task_sensitivity_eval.sh [interval_seconds]
#
# Tail it with: tail -f LeakageEvaluationFramework/logs/task_sensitivity_eval/progress.log

set -uo pipefail
cd "$(dirname "$0")/../.."

INTERVAL="${1:-15}"
OUT_BASE="runtime/task_sensitivity_eval/full"
LOG_DIR="LeakageEvaluationFramework/logs/task_sensitivity_eval"
PROGRESS="$LOG_DIR/progress.log"
TOTAL=337
mkdir -p "$LOG_DIR"

echo "=== progress watcher started $(date -Is)  interval=${INTERVAL}s  total_per_method=${TOTAL} ===" | tee -a "$PROGRESS"

while true; do
  parts=()
  total_done=0
  total_err=0
  for m in b1 b2 b3 b4; do
    d="$OUT_BASE/$m"
    n=0; e=0
    [[ -f "$d/instance_results.jsonl" ]] && n=$(wc -l < "$d/instance_results.jsonl")
    [[ -f "$d/errors.jsonl" ]] && e=$(wc -l < "$d/errors.jsonl")
    total_done=$((total_done + n))
    total_err=$((total_err + e))
    pct=$(( n * 100 / TOTAL ))
    parts+=("${m}=${n}/${TOTAL}(${pct}%)${e:+ err=$e}")
  done
  grand_total=$((4 * TOTAL))
  reached=$((total_done + total_err))
  grand_pct=$(( reached * 100 / grand_total ))
  ts=$(date +%H:%M:%S)
  echo "[$ts] ${parts[*]}  | reached=${reached}/${grand_total}(${grand_pct}%)  done=${total_done}  errors=${total_err}" | tee -a "$PROGRESS"

  # Exit when every instance has either succeeded or been recorded as an error.
  if [[ "$reached" -ge "$grand_total" ]]; then
    echo "=== all 4 methods reached ${TOTAL} instances (done=${total_done}, errors=${total_err}) at $(date -Is) ===" | tee -a "$PROGRESS"
    break
  fi
  sleep "$INTERVAL"
done
