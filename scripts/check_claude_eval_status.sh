#!/usr/bin/env bash
# One-shot status snapshot for the Claude self-eval batch run.
# Usage: bash LeakageEvaluationFramework/scripts/check_claude_eval_status.sh
#
# Shows, per baseline:
#   - whether the worker process is alive
#   - taxonomy phase status (done / pending)
#   - Shapley chunks submitted on Anthropic (= what's billed and in-flight)
#   - per-chunk processing_status (succeeded / processing / errored)
#   - final summary.json snapshot when the baseline finishes

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUNTIME="LeakageEvaluationFramework/runtime/claude_sonnet4/eval_results_self_batch"

echo "============================================================"
echo " Claude self-eval (batch) status — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

for bm in b1 b2 b3 b4; do
  echo
  echo "-------- $bm --------"

  # Worker alive?
  pid=$(pgrep -f "batch_eval\.cli run.*--method $bm" | head -1 || true)
  if [ -n "$pid" ]; then echo "  worker:        ALIVE (pid $pid)"; else echo "  worker:        not running"; fi

  # Taxonomy state
  if [ -f "$RUNTIME/$bm/state/taxonomy_done.flag" ]; then
    echo "  taxonomy:      done"
  elif [ -f "$RUNTIME/$bm/state/taxonomy_batch.json" ]; then
    echo "  taxonomy:      submitted (polling)"
  else
    echo "  taxonomy:      not yet submitted"
  fi

  # Shapley chunks submitted
  sf="$RUNTIME/$bm/state/shapley_batches.json"
  if [ -f "$sf" ]; then
    uv run --quiet python - "$sf" "$bm" <<'PYEOF'
import json, os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))
from anthropic import Anthropic
path, bm = sys.argv[1], sys.argv[2]
data = json.load(open(path))
print(f"  shapley:       {len(data)} chunk(s) submitted")
try:
    c = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    for i, h in enumerate(data, 1):
        b = c.messages.batches.retrieve(h['batch_id'])
        rc = b.request_counts
        total = rc.processing + rc.succeeded + rc.errored + rc.canceled + rc.expired
        print(f"    chunk {i}/{len(data)} ({h['request_count']:>6} req) status={b.processing_status:<11} "
              f"succeeded={rc.succeeded:>6}/{total} errored={rc.errored} expired={rc.expired}")
except Exception as e:
    print(f"    (could not query Anthropic for live status: {e})")
PYEOF
  elif [ -f "$RUNTIME/$bm/state/taxonomy_done.flag" ]; then
    echo "  shapley:       not yet submitted (worker may be retrying on 429)"
  else
    echo "  shapley:       waiting for taxonomy"
  fi

  # Final summary?
  if [ -f "$RUNTIME/$bm/summary.json" ]; then
    echo "  ** FINISHED — summary.json present **"
    python3 -c "
import json
s = json.load(open('$RUNTIME/$bm/summary.json'))
print(f\"     n_instances={s['n_instances']}  olr={s['olr']['mean']:.3f}  shapley_dclr={s['shapley_dclr']['mean']:.3f}  performance={s['performance']['mean']:.3f}\")
"
  fi
done

echo
echo "Tip: tail -f <log> on any of these for live progress:"
for bm in b1 b3; do
  log=$(ls -t LeakageEvaluationFramework/logs/claude_sonnet4_self_eval_batch_*/eval_$bm.log 2>/dev/null | head -1)
  [ -n "$log" ] && echo "  $bm: $log"
done
for bm in b2 b4; do
  log=$(ls -t LeakageEvaluationFramework/logs/claude_self_eval_batch_restart2_*/eval_$bm.log 2>/dev/null | head -1)
  [ -n "$log" ] && echo "  $bm: $log"
done
