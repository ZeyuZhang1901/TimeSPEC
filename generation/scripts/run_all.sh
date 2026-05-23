#!/usr/bin/env bash
# Run B1-B4 generation for all three models on the benchmark binary benchmark.
#
# Override env variables for partial runs without editing this file:
#   METHODS=b1,b3    — run only specified methods
#   MODEL_FILTER=qwen — run only models whose output dir contains this string
#   SPLIT=treatment  — run only the treatment split
#   MAX_INSTANCES=50 — cap for fast debugging
#   CONCURRENCY=8    — parallel threads for RAG retrieval and B2/B4 supervisor
#
# Examples:
#   # Full run, all models
#   bash LeakageEvaluationFramework/generation/scripts/run_all.sh
#
#   # Only Qwen model, 8 parallel workers
#   CONCURRENCY=8 MODEL_FILTER=qwen bash LeakageEvaluationFramework/generation/scripts/run_all.sh
#
#   # Quick debug on 50 treatment instances
#   METHODS=b1,b3 SPLIT=treatment MAX_INSTANCES=50 \
#     bash LeakageEvaluationFramework/generation/scripts/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
cd "$PROJECT_ROOT"

METHODS="${METHODS:-b1,b2,b3,b4}"
MODEL_FILTER="${MODEL_FILTER:-}"
SPLIT="${SPLIT:-all}"
MAX_INSTANCES="${MAX_INSTANCES:-}"
DATASET="${DATASET:-dataset/benchmark/final/binary.jsonl}"
CONCURRENCY="${CONCURRENCY:-8}"

# MODEL_ID | OUTPUT_DIR
declare -a COMBOS=(
    "anthropic/claude-sonnet-4   runtime/benchmark/claude_sonnet4"
    "qwen/qwen3.5-35b-a3b       runtime/benchmark/qwen35_35b"
    "moonshotai/kimi-k2.5        runtime/benchmark/kimi_k25"
)

echo "========================================"
echo "  benchmark Generation — All Models"
echo "  Methods:     ${METHODS}"
echo "  Split:       ${SPLIT}"
echo "  Concurrency: ${CONCURRENCY}"
echo "  Filter:      ${MODEL_FILTER:-<none>}"
echo "========================================"
echo ""

for combo in "${COMBOS[@]}"; do
    read -r model outdir <<< "$combo"

    if [[ -n "${MODEL_FILTER}" && "${outdir}" != *"${MODEL_FILTER}"* ]]; then
        echo "Skipping ${model} (filtered out by MODEL_FILTER=${MODEL_FILTER})"
        continue
    fi

    echo "--- ${model} | methods=${METHODS} ---"

    EXTRA_ARGS=""
    if [[ -n "${MAX_INSTANCES}" ]]; then
        EXTRA_ARGS="--max-instances ${MAX_INSTANCES}"
    fi

    uv run python -m LeakageEvaluationFramework.generation.cli run \
        --dataset     "${DATASET}" \
        --output      "${outdir}" \
        --model       "${model}" \
        --methods     "${METHODS}" \
        --split       "${SPLIT}" \
        --concurrency "${CONCURRENCY}" \
        ${EXTRA_ARGS}

    echo "Done: ${model}"
    echo ""
done

echo "========================================"
echo "  All runs complete."
echo "  Results: runtime/benchmark/<model>/{b1,b2,b3,b4}/completions.json"
echo "========================================"
