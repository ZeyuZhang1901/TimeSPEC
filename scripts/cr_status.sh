#!/usr/bin/env bash
# One-command status check for all camera-ready jobs.
LEF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LEF"

echo "==================== CAMERA-READY STATUS $(date) ===================="
for JOB in e3_crawl e3_full e6_full; do
  PIDFILE="camera_ready/logs/$JOB.pid"
  LOGFILE="camera_ready/logs/$JOB.log"
  [ -f "$PIDFILE" ] || continue
  PID=$(cat "$PIDFILE")
  if kill -0 "$PID" 2>/dev/null; then STATE="RUNNING (pid $PID)"; else STATE="finished/stopped"; fi
  echo ""
  echo "--- $JOB: $STATE"
  [ -f "$LOGFILE" ] && tail -3 "$LOGFILE" | sed 's/^/    /'
done

echo ""
echo "--- result file counts"
C="camera_ready/e3_snapshot/crawl_progress.jsonl"
[ -f "$C" ] && echo "    crawl progress: $(wc -l < "$C") / 350"
for F in \
  camera_ready/e3_snapshot/gen/snapshot_rag/completions.json \
  camera_ready/e3_snapshot/gen/live/b3/completions.json \
  camera_ready/e3_snapshot/gen/live/b4/completions.json ; do
  [ -f "$F" ] && echo "    $F: $(python3 -c "import json;print(len(json.load(open('$F'))))" 2>/dev/null) completions"
done
for D in camera_ready/e3_snapshot/eval/snapshot_rag camera_ready/e3_snapshot/eval/b3 camera_ready/e3_snapshot/eval/b4 \
         camera_ready/e6_cost/eval/b1 camera_ready/e6_cost/eval/b2 camera_ready/e6_cost/eval/b3 camera_ready/e6_cost/eval/b4; do
  F="$D/instance_results.jsonl"
  [ -f "$F" ] && echo "    $F: $(wc -l < "$F") instances"
done
for F in camera_ready/e3_snapshot/comparison.json \
         camera_ready/e5_shapley/probe_metric_comparison.json \
         camera_ready/e5_shapley/e3_binary_metric_comparison.json ; do
  [ -f "$F" ] && echo "    DONE: $F"
done

echo ""
echo "--- OpenRouter usage"
tail -2 camera_ready/usage_snapshots.jsonl 2>/dev/null | sed 's/^/    /'
echo "======================================================================"
