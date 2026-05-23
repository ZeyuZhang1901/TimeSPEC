#!/usr/bin/env bash
# Submit Claude Sonnet 4 self-evaluation via Anthropic Batch API (50% discount)
# for all 4 baselines in parallel. Each baseline is its own process that
# submits → polls → fetches → writes results. Anthropic processes batches
# asynchronously server-side, so the four pollers don't compete.
#
# Output: LeakageEvaluationFramework/runtime/claude_sonnet4/eval_results_self_batch/{b1,b2,b3,b4}/
# Logs:   LeakageEvaluationFramework/logs/claude_sonnet4_self_eval_batch_<TS>/eval_<b>.log
#
# Restart-safe: each output dir has state/ with batch IDs. Re-running the
# same script resumes from disk (won't re-submit completed phases).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/LeakageEvaluationFramework/logs/claude_sonnet4_self_eval_batch_${TS}"
mkdir -p "$LOG_DIR"

CONFIG="${ROOT}/LeakageEvaluationFramework/configs/leakage_eval_claude_self_batch.yaml"
DATASET="${ROOT}/dataset/benchmark/final/binary.jsonl"
MODEL="anthropic/claude-sonnet-4"
CLAUDE_ROOT="${ROOT}/LeakageEvaluationFramework/runtime/claude_sonnet4"
POLL_SEC="${POLL_SEC:-60}"

echo "Claude self-eval (Anthropic Batch API) — logs: ${LOG_DIR}"
echo "Results:  ${CLAUDE_ROOT}/eval_results_self_batch/{b1,b2,b3,b4}"

for bm in b1 b2 b3 b4; do
  OUT="${CLAUDE_ROOT}/eval_results_self_batch/${bm}"
  IN="${CLAUDE_ROOT}/${bm}/completions.json"
  LOG="${LOG_DIR}/eval_${bm}.log"
  mkdir -p "$OUT"
  echo "Starting ${bm} -> ${LOG}"
  (
    uv run python -m LeakageEvaluationFramework.batch_eval.cli run \
      --config "$CONFIG" \
      --input "$IN" \
      --dataset "$DATASET" \
      --output "$OUT" \
      --model "$MODEL" \
      --method "$bm" \
      --poll-interval-sec "$POLL_SEC" \
      2>&1 | tee "$LOG"
  ) &
done

wait
echo "All 4 baselines finished. Logs in ${LOG_DIR}"
