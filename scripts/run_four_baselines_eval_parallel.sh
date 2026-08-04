#!/usr/bin/env bash
# Run leakage_eval on Qwen 3.5 35B completions for b1–b4 in parallel (four log files).
# Requires OPENROUTER_API_KEY. Tune --concurrency if you hit rate limits.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/eval_four_baselines_${TS}"
mkdir -p "$LOG_DIR"

CONFIG="${ROOT}/LeakageEvaluationFramework/configs/leakage_eval_default.yaml"
DATASET="${ROOT}/dataset/benchmark/final/binary.jsonl"
MODEL="qwen/qwen3.5-35b-a3b"
CONCURRENCY="${CONCURRENCY:-8}"

for bm in b1 b2 b3 b4; do
  OUT="${ROOT}/runtime/benchmark/qwen35_35b/eval_results/${bm}"
  IN="${ROOT}/runtime/benchmark/qwen35_35b/${bm}/completions.json"
  LOG="${LOG_DIR}/eval_${bm}.log"
  echo "Starting ${bm} → ${LOG}"
  (
    uv run python -m LeakageEvaluationFramework.leakage_eval.cli run \
      --config "$CONFIG" \
      --input "$IN" \
      --dataset "$DATASET" \
      --output "$OUT" \
      --model "$MODEL" \
      --method "$bm" \
      --taxonomy-backend llm \
      --concurrency "$CONCURRENCY" \
      2>&1 | tee "$LOG"
  ) &
done

wait
echo "Done. Logs in ${LOG_DIR}"
