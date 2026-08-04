#!/usr/bin/env bash
# Retry the 8 errored instances from the task-sensitivity eval.
# - Resume logic in the pipeline skips anything already in instance_results.jsonl.
# - We pass the specific failed instance_ids per method so the rerun is targeted.
# - errors.jsonl is rewritten in-place (cleared) so we can diff what's still failing.

set -uo pipefail
cd "$(dirname "$0")/../.."

CONFIG="LeakageEvaluationFramework/configs/leakage_eval_task_sensitivity.yaml"
DATASET="dataset/evaluation/task_sensitivity_full_metadata.json"
INPUT_BASE="runtime/task_sensitivity_full/qwen35_35b"
OUTPUT_BASE="runtime/task_sensitivity_eval/full"
MODEL="qwen/qwen3.5-35b-a3b"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="LeakageEvaluationFramework/logs/task_sensitivity_eval"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/retry_${TS}.log"

declare -A IDS=(
  [b1]="salary_wayne_ellington_2019,salary_rudy_gay_2021,salary_robin_lopez_2021"
  [b2]="salary_jae_crowder_2020,salary_demar_derozan_2021"
  [b3]="stock_45"
  [b4]="salary_lauri_markkanen_2021,salary_dario_saric_2024"
)

echo "=== Retry failed eval | started $(date -Is) ===" | tee -a "$LOG"

for method in b1 b2 b3 b4; do
  ids="${IDS[$method]}"
  out_dir="$OUTPUT_BASE/$method"
  in_file="$INPUT_BASE/$method/completions.json"

  # Clear old errors for these specific IDs (keep others, though here all are being retried)
  if [[ -f "$out_dir/errors.jsonl" ]]; then
    cp "$out_dir/errors.jsonl" "$out_dir/errors.jsonl.pre_retry_${TS}.bak"
    : > "$out_dir/errors.jsonl"
  fi

  echo "--- [$method] retry $ids ---" | tee -a "$LOG"
  uv run python -u -m LeakageEvaluationFramework.leakage_eval.cli run \
      --config "$CONFIG" \
      --input  "$in_file" \
      --dataset "$DATASET" \
      --output "$out_dir" \
      --model  "$MODEL" \
      --method "$method" \
      --instance-ids "$ids" \
      --taxonomy-backend llm \
      --concurrency 4 \
      --checkpoint-every 5 \
      2>&1 | tee -a "$LOG"

  done_n=$(wc -l < "$out_dir/instance_results.jsonl")
  err_n=$([[ -f "$out_dir/errors.jsonl" ]] && wc -l < "$out_dir/errors.jsonl" || echo 0)
  echo "--- [$method] retry done: total instances=$done_n, residual errors=$err_n ---" | tee -a "$LOG"
done

echo "=== Retry finished $(date -Is) ===" | tee -a "$LOG"
