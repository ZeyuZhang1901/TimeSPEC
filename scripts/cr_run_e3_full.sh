#!/usr/bin/env bash
# Camera-ready E3 full chain: waits for the Wayback crawl, then runs
# snapshot-RAG generation (200-question seeded subset), fresh b3/b4 generation
# on the same questions, Shapley-DCLR evaluation of all three arms, and the
# final bootstrap comparison + E5 replication analysis. Designed for nohup.
set -uo pipefail

LEF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LEF"
unset OPENROUTER_API_KEY PERPLEXITY_API_KEY || true

export PYTHONPATH="$(dirname "$LEF")"
export CR_USAGE_LOG="$LEF/camera_ready/e3_snapshot/full_usage_log.jsonl"
PY="${PYTHON:-python3}"
MODEL="qwen/qwen3.5-35b-a3b"
JUDGE="google/gemini-3.1-flash-lite"
E3="$LEF/camera_ready/e3_snapshot"

snapshot_usage () {
  KEY=$(grep '^OPENROUTER_API_KEY' "$LEF/.env" | cut -d= -f2 | tr -d ' "')
  curl -s https://openrouter.ai/api/v1/auth/key -H "Authorization: Bearer $KEY" | \
    "$PY" -c "
import json,sys,time
d=json.load(sys.stdin)['data']
print(json.dumps({'ts':time.strftime('%Y-%m-%dT%H:%M:%S'),'stage':'$1','usage':d['usage'],'remaining':d['limit_remaining']}))
" >> "$LEF/camera_ready/usage_snapshots.jsonl"
}

# ---- Stage 0: wait for the crawl to finish, then finalize contexts ----
CRAWL_PID=$(cat "$LEF/camera_ready/logs/e3_crawl.pid" 2>/dev/null || echo "")
if [ -n "$CRAWL_PID" ]; then
  while kill -0 "$CRAWL_PID" 2>/dev/null; do
    echo "[e3] waiting for crawl (pid $CRAWL_PID) $(date)"
    sleep 120
  done
fi
echo "[e3] crawl finished, finalizing contexts $(date)"
"$PY" scripts/cr_snapshot_crawl.py \
  --dataset "$E3/sample_350.jsonl" --output-dir "$E3" --finalize-only

# ---- Stage 1: Snapshot-RAG generation on seeded 200-question subset ----
echo "[e3] snapshot-rag generation start $(date)"
snapshot_usage e3_snapshot_gen_start
"$PY" scripts/cr_run_snapshot_rag.py \
  --dataset "$E3/sample_350.jsonl" \
  --contexts "$E3/snapshot_contexts.json" \
  --output-dir "$E3/gen/snapshot_rag" \
  --model "$MODEL" --judge-model "$JUDGE" \
  --subset-size 200 --seed 42
echo "[e3] snapshot-rag generation done $(date)"
snapshot_usage e3_snapshot_gen_done

# ---- Stage 2: fresh b3 + b4 generation on the SAME matched questions ----
echo "[e3] b3/b4 generation start $(date)"
"$PY" -m LeakageEvaluationFramework.generation.cli run \
  --dataset "$E3/gen/snapshot_rag/dataset.jsonl" \
  --output "$E3/gen/live" \
  --model "$MODEL" \
  --methods b3,b4 \
  --judge-model "$JUDGE" \
  --concurrency 8 \
  --shared-rag-cache "$E3/gen/rag_cache"
echo "[e3] b3/b4 generation done $(date)"
snapshot_usage e3_live_gen_done

# ---- Stage 3: Shapley-DCLR evaluation of all three arms ----
declare -A INPUTS=(
  [snapshot_rag]="$E3/gen/snapshot_rag/completions.json"
  [b3]="$E3/gen/live/b3/completions.json"
  [b4]="$E3/gen/live/b4/completions.json"
)
for ARM in snapshot_rag b3 b4; do
  echo "[e3] eval $ARM start $(date)"
  "$PY" -m LeakageEvaluationFramework.leakage_eval.cli run \
    --config "$LEF/configs/cr_snapshot_eval.yaml" \
    --input "${INPUTS[$ARM]}" \
    --dataset "$E3/gen/snapshot_rag/dataset.jsonl" \
    --output "$E3/eval/$ARM" \
    --model "$MODEL" \
    --method "$ARM" \
    --checkpoint-every 20
  echo "[e3] eval $ARM done $(date)"
  snapshot_usage "e3_eval_${ARM}_done"
done

# ---- Stage 4: analyses (offline) ----
echo "[e3] analysis start $(date)"
"$PY" scripts/cr_analyze_snapshot.py \
  --dataset "$E3/gen/snapshot_rag/dataset.jsonl" \
  --arm "snapshot_rag=$E3/eval/snapshot_rag/instance_results.jsonl" \
  --arm "b3=$E3/eval/b3/instance_results.jsonl" \
  --arm "b4=$E3/eval/b4/instance_results.jsonl" \
  --output "$E3/comparison.json"

# E5 replication on fresh binary arms (OLR/Top-K/LOO vs DCLR from new caches)
"$PY" scripts/cr_e5_metric_comparison.py \
  --results-root "$E3/eval" \
  --methods snapshot_rag,b3,b4 \
  --output "$LEF/camera_ready/e5_shapley/e3_binary_metric_comparison.json"

echo "[e3] ALL DONE $(date)"
snapshot_usage e3_all_done
