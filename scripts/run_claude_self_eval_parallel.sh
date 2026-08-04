#!/usr/bin/env bash
# Run leakage_eval (self-consistent Claude judge + Claude Shapley via completions.model)
# on Claude Sonnet 4 completions for b1–b4 in parallel.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/claude_sonnet4_self_eval_${TS}"
mkdir -p "$LOG_DIR"

CONFIG="${ROOT}/LeakageEvaluationFramework/configs/leakage_eval_claude_self.yaml"
DATASET="${ROOT}/dataset/benchmark/final/binary.jsonl"
MODEL="anthropic/claude-sonnet-4"
CLAUDE_ROOT="${ROOT}/runtime/benchmark/claude_sonnet4"
CONCURRENCY="${CONCURRENCY:-8}"

echo "Claude self-eval parallel run — logs: ${LOG_DIR}"
echo "Results: ${CLAUDE_ROOT}/eval_results_self/{b1,b2,b3,b4}"

for bm in b1 b2 b3 b4; do
  OUT="${CLAUDE_ROOT}/eval_results_self/${bm}"
  IN="${CLAUDE_ROOT}/${bm}/completions.json"
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
