#!/usr/bin/env bash
# Sequential evaluation of B1-B4 on the task_sensitivity dataset
# with crash-safe incremental writes and a single combined progress log.
#
# Resume semantics: each method has its own output dir. On rerun, the
# pipeline skips instances already present in instance_results.jsonl.
#
# Progress: every per-method run prints "Processing chunk a-b / N" lines
# via the pipeline logger, so `tail -f` shows live progress.

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
COMBINED="$LOG_DIR/full_${TS}.log"

echo "=== Task-sensitivity evaluation (B1..B4) | started $(date -Is) ===" | tee -a "$COMBINED"
echo "Config:   $CONFIG"   | tee -a "$COMBINED"
echo "Dataset:  $DATASET"  | tee -a "$COMBINED"
echo "Outputs:  $OUTPUT_BASE/{b1,b2,b3,b4}" | tee -a "$COMBINED"
echo "Log:      $COMBINED" | tee -a "$COMBINED"
echo "" | tee -a "$COMBINED"

for method in b1 b2 b3 b4; do
  in_file="$INPUT_BASE/$method/completions.json"
  out_dir="$OUTPUT_BASE/$method"
  mkdir -p "$out_dir"

  echo "--- [$method] start $(date -Is) ---" | tee -a "$COMBINED"

  uv run python -u -m LeakageEvaluationFramework.leakage_eval.cli run \
      --config "$CONFIG" \
      --input  "$in_file" \
      --dataset "$DATASET" \
      --output "$out_dir" \
      --model  "$MODEL" \
      --method "$method" \
      --taxonomy-backend llm \
      --concurrency 4 \
      --checkpoint-every 10 \
      2>&1 | tee -a "$COMBINED"
  rc=${PIPESTATUS[0]}

  done_count=0
  if [[ -f "$out_dir/instance_results.jsonl" ]]; then
    done_count=$(wc -l < "$out_dir/instance_results.jsonl")
  fi
  err_count=0
  if [[ -f "$out_dir/errors.jsonl" ]]; then
    err_count=$(wc -l < "$out_dir/errors.jsonl")
  fi

  echo "--- [$method] end   $(date -Is)  rc=$rc  done=$done_count  errors=$err_count ---" | tee -a "$COMBINED"
  echo "" | tee -a "$COMBINED"
done

echo "=== All methods finished $(date -Is) ===" | tee -a "$COMBINED"
