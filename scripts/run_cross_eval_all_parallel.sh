#!/usr/bin/env bash
# Cross-evaluation: run leakage_eval with fixed Gemini 3.1 Flash Lite judge + Shapley
# on all three generators (Qwen, Kimi, Claude) × b1–b4.
#
# Runs ONE combination at a time (sequential) to avoid API rate limits.
# Supports resume: re-running skips already-completed instances automatically.
#
# Requires OPENROUTER_API_KEY. Override concurrency:
#   CONCURRENCY=6 bash scripts/run_cross_eval_all_parallel.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/cross_eval_gemini_${TS}"
mkdir -p "$LOG_DIR"

CONFIG="${ROOT}/configs/leakage_eval_cross_gemini.yaml"
DATASET="${ROOT}/dataset/final/binary.jsonl"
CONCURRENCY="${CONCURRENCY:-8}"
CHECKPOINT="${CHECKPOINT:-200}"

declare -A MODEL_IDS=(
  [qwen35_35b]="qwen/qwen3.5-35b-a3b"
  [kimi_k25]="moonshotai/kimi-k2.5"
  [claude_sonnet4]="anthropic/claude-sonnet-4"
)

echo "Cross-eval (Gemini 3.1 Flash Lite) — logs: ${LOG_DIR}"
echo "Mode: ONE combination at a time (sequential)"
echo "Checkpoint every: ${CHECKPOINT} instances"
echo "Concurrency per job: ${CONCURRENCY}"
echo ""

for gen_dir in qwen35_35b kimi_k25 claude_sonnet4; do
  MODEL="${MODEL_IDS[$gen_dir]}"
  GEN_ROOT="${ROOT}/runtime/${gen_dir}"

  for bm in b1 b2 b3 b4; do
    IN="${GEN_ROOT}/${bm}/completions.json"
    OUT="${GEN_ROOT}/eval_results_cross/${bm}"
    LOG="${LOG_DIR}/eval_${gen_dir}_${bm}.log"
    mkdir -p "$OUT"

    # Skip if already completed
    if [ -f "$OUT/summary.json" ]; then
      echo "[SKIP] ${gen_dir}/${bm} — already has summary.json"
      continue
    fi

    echo "[START] ${gen_dir}/${bm} → ${LOG}"
    uv run python -m LeakageEvaluationFramework.leakage_eval.cli run \
      --config "$CONFIG" \
      --input "$IN" \
      --dataset "$DATASET" \
      --output "$OUT" \
      --model "$MODEL" \
      --method "$bm" \
      --taxonomy-backend llm \
      --concurrency "$CONCURRENCY" \
      --checkpoint-every "$CHECKPOINT" \
      2>&1 | tee "$LOG"
    echo "[DONE] ${gen_dir}/${bm}"
    echo ""
  done
done

echo "All done. Logs in ${LOG_DIR}"
