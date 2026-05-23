#!/usr/bin/env bash
# Launch the full benchmark generation run in the background using nohup.
#
# Each model is launched as a separate nohup process so you can close your
# terminal without interrupting the run.
#
# Usage:
#   bash LeakageEvaluationFramework/generation/scripts/run_background.sh
#
# Optional env overrides (same as run_all.sh):
#   METHODS=b1,b2,b3,b4   CONCURRENCY=8   MODEL_FILTER=qwen
#   SPLIT=all              MAX_INSTANCES=50
#
# Monitor progress:
#   tail -f logs/gen_claude_sonnet4_<timestamp>.log
#   tail -f logs/gen_qwen35_35b_<timestamp>.log
#
# Check if runs are still alive:
#   cat logs/pids_<timestamp>.txt

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
SHARED_RAG_CACHE="${SHARED_RAG_CACHE:-runtime/benchmark/shared_rag_cache}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# MODEL_ID | OUTPUT_DIR
declare -a COMBOS=(
    "anthropic/claude-sonnet-4   runtime/benchmark/claude_sonnet4"
    "qwen/qwen3.5-35b-a3b       runtime/benchmark/qwen35_35b"
    "moonshotai/kimi-k2.5        runtime/benchmark/kimi_k25"
)

PID_FILE="${LOG_DIR}/pids_${TS}.txt"
echo "# Generation run started at $(date)" > "${PID_FILE}"
echo "# Methods: ${METHODS}  Split: ${SPLIT}  Concurrency: ${CONCURRENCY}" >> "${PID_FILE}"
echo "" >> "${PID_FILE}"

echo "========================================"
echo "  benchmark — Background Launch"
echo "  Methods:     ${METHODS}"
echo "  Split:       ${SPLIT}"
echo "  Concurrency: ${CONCURRENCY}"
echo "  Filter:      ${MODEL_FILTER:-<none>}"
echo "  RAG cache:   ${SHARED_RAG_CACHE}"
echo "  Logs:        ${LOG_DIR}/"
echo "========================================"
echo ""

for combo in "${COMBOS[@]}"; do
    read -r model outdir <<< "$combo"

    if [[ -n "${MODEL_FILTER}" && "${outdir}" != *"${MODEL_FILTER}"* ]]; then
        echo "Skipping ${model} (MODEL_FILTER=${MODEL_FILTER})"
        continue
    fi

    tag="$(basename "${outdir}")"
    logfile="${LOG_DIR}/gen_${tag}_${TS}.log"

    EXTRA_ARGS=""
    if [[ -n "${MAX_INSTANCES}" ]]; then
        EXTRA_ARGS="--max-instances ${MAX_INSTANCES}"
    fi

    echo "Launching ${model} → ${logfile}"

    nohup uv run python -m LeakageEvaluationFramework.generation.cli run \
        --dataset          "${DATASET}" \
        --output           "${outdir}" \
        --model            "${model}" \
        --methods          "${METHODS}" \
        --split            "${SPLIT}" \
        --concurrency      "${CONCURRENCY}" \
        --shared-rag-cache "${SHARED_RAG_CACHE}" \
        ${EXTRA_ARGS} \
        > "${logfile}" 2>&1 &

    pid=$!
    echo "  PID: ${pid}  Log: ${logfile}"
    echo "${model}  PID=${pid}  log=${logfile}" >> "${PID_FILE}"
done

echo ""
echo "All jobs launched. PIDs recorded in ${PID_FILE}"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOG_DIR}/gen_*_${TS}.log"
echo ""
echo "Check if still running:"
echo "  while read -r line; do"
echo "    pid=\$(echo \"\$line\" | grep -o 'PID=[0-9]*' | cut -d= -f2)"
echo "    [[ -n \"\$pid\" ]] && (kill -0 \"\$pid\" 2>/dev/null && echo \"ALIVE: \$line\" || echo \"DONE:  \$line\")"
echo "  done < ${PID_FILE}"
