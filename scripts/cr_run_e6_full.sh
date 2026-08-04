#!/usr/bin/env bash
# Camera-ready E6: instrumented 100-question x 4-method generation + evaluation.
# Designed to run detached (nohup). All LLM calls are logged to CR_USAGE_LOG.
set -uo pipefail

LEF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LEF"
unset OPENROUTER_API_KEY PERPLEXITY_API_KEY || true

export PYTHONPATH="$(dirname "$LEF")"
export CR_USAGE_LOG="$LEF/camera_ready/e6_cost/full_usage_log.jsonl"
PY="${PYTHON:-python3}"
MODEL="qwen/qwen3.5-35b-a3b"
JUDGE="google/gemini-3.1-flash-lite"
DATASET="$LEF/camera_ready/e6_cost/sample_100.jsonl"
OUT="$LEF/camera_ready/e6_cost/full"

snapshot_usage () {
  KEY=$(grep '^OPENROUTER_API_KEY' "$LEF/.env" | cut -d= -f2 | tr -d ' "')
  curl -s https://openrouter.ai/api/v1/auth/key -H "Authorization: Bearer $KEY" | \
    "$PY" -c "
import json,sys,time
d=json.load(sys.stdin)['data']
print(json.dumps({'ts':time.strftime('%Y-%m-%dT%H:%M:%S'),'stage':'$1','usage':d['usage'],'remaining':d['limit_remaining']}))
" >> "$LEF/camera_ready/usage_snapshots.jsonl"
}

echo "[e6] generation start $(date)"
snapshot_usage e6_generation_start
"$PY" -m LeakageEvaluationFramework.generation.cli run \
  --dataset "$DATASET" \
  --output "$OUT" \
  --model "$MODEL" \
  --methods b1,b2,b3,b4 \
  --judge-model "$JUDGE" \
  --concurrency 8 \
  --shared-rag-cache "$LEF/camera_ready/e6_cost/rag_cache"
echo "[e6] generation done $(date)"
snapshot_usage e6_generation_done

for METHOD in b1 b2 b3 b4; do
  echo "[e6] eval $METHOD start $(date)"
  "$PY" -m LeakageEvaluationFramework.leakage_eval.cli run \
    --config "$LEF/configs/cr_snapshot_eval.yaml" \
    --input "$OUT/$METHOD/completions.json" \
    --dataset "$DATASET" \
    --output "$LEF/camera_ready/e6_cost/eval/$METHOD" \
    --model "$MODEL" \
    --method "$METHOD" \
    --checkpoint-every 20
  echo "[e6] eval $METHOD done $(date)"
  snapshot_usage "e6_eval_${METHOD}_done"
done

echo "[e6] ALL DONE $(date)"
