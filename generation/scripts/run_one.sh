#!/usr/bin/env bash
# Run B1-B4 generation methods for one (model, output-dir) combination.
#
# Usage:
#   bash LeakageEvaluationFramework/generation/scripts/run_one.sh \
#       MODEL_ID OUTPUT_DIR [METHODS]
#
# Examples:
#   bash LeakageEvaluationFramework/generation/scripts/run_one.sh \
#       anthropic/claude-sonnet-4 runtime/benchmark/claude_sonnet4
#
#   METHODS=b1,b3 CONCURRENCY=8 bash LeakageEvaluationFramework/generation/scripts/run_one.sh \
#       qwen/qwen3.5-35b-a3b runtime/benchmark/qwen35_35b

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
cd "$PROJECT_ROOT"

MODEL="${1:?Usage: run_one.sh MODEL_ID OUTPUT_DIR [METHODS]}"
OUTPUT="${2:?Usage: run_one.sh MODEL_ID OUTPUT_DIR [METHODS]}"
METHODS="${3:-b1,b2,b3,b4}"
DATASET="${DATASET:-dataset/benchmark/final/binary.jsonl}"
SPLIT="${SPLIT:-all}"
MAX_INSTANCES="${MAX_INSTANCES:-}"
CONCURRENCY="${CONCURRENCY:-8}"

echo "=== Generation Run ==="
echo "  Model:       ${MODEL}"
echo "  Output:      ${OUTPUT}"
echo "  Methods:     ${METHODS}"
echo "  Dataset:     ${DATASET}"
echo "  Split:       ${SPLIT}"
echo "  Concurrency: ${CONCURRENCY}"
echo ""

EXTRA_ARGS=""
if [[ -n "${MAX_INSTANCES}" ]]; then
    EXTRA_ARGS="--max-instances ${MAX_INSTANCES}"
fi

uv run python -m LeakageEvaluationFramework.generation.cli run \
    --dataset     "${DATASET}" \
    --output      "${OUTPUT}" \
    --model       "${MODEL}" \
    --methods     "${METHODS}" \
    --split       "${SPLIT}" \
    --concurrency "${CONCURRENCY}" \
    ${EXTRA_ARGS}
