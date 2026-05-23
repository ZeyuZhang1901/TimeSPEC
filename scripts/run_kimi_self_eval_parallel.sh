#!/usr/bin/env bash
# Run leakage_eval (self-consistent Kimi judge + Kimi Shapley via completions.model)
# on Kimi K2.5 completions for b1–b4 in parallel.
#
# Requires OPENROUTER_API_KEY. Override concurrency:
#   CONCURRENCY=6 bash scripts/run_kimi_self_eval_parallel.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/kimi_k25_self_eval_${TS}"
mkdir -p "$LOG_DIR"

CONFIG="${ROOT}/LeakageEvaluationFramework/configs/leakage_eval_kimi_self.yaml"
DATASET="${ROOT}/dataset/benchmark/final/binary.jsonl"
MODEL="moonshotai/kimi-k2.5"
KIMI_ROOT="${ROOT}/runtime/benchmark/kimi_k25"
CONCURRENCY="${CONCURRENCY:-8}"

echo "Kimi self-eval parallel run — logs: ${LOG_DIR}"
echo "Results: ${KIMI_ROOT}/eval_results_self/{b1,b2,b3,b4}"

for bm in b1 b2 b3 b4; do
  OUT="${KIMI_ROOT}/eval_results_self/${bm}"
  IN="${KIMI_ROOT}/${bm}/completions.json"
  LOG="${LOG_DIR}/eval_${bm}.log"
  mkdir -p "$OUT"
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
