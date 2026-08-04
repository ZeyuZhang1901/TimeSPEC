#!/usr/bin/env bash
# Re-evaluate the rescued B4 stock instances and refresh the per-task aggregates.
#
# Pre-requisite: rescue_b4_stock_leakage.py has finished, so
#   runtime/task_sensitivity_full/qwen35_35b/b4/completions.json
# contains the patched (scrubbed) evidence for the affected stock_* IDs.
#
# Steps:
#   1) Snapshot the existing B4 eval artefacts.
#   2) Remove the affected stock_* rows from instance_results.jsonl,
#      claim_results.jsonl, shapley_results.jsonl, and errors.jsonl so the
#      pipeline's resume logic re-attempts them.
#   3) Invoke the eval CLI with --method b4 --instance-ids "<affected>".
#   4) Print the refreshed per-task aggregates (legal / salary / stock).
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
COMPLETIONS="$ROOT/runtime/task_sensitivity_full/qwen35_35b/b4/completions.json"
META="$ROOT/dataset/evaluation/task_sensitivity_full_metadata.json"
EVAL_OUT="$ROOT/runtime/task_sensitivity_eval/full/b4"
CFG="$ROOT/LeakageEvaluationFramework/configs/leakage_eval_task_sensitivity.yaml"
LOG_DIR="$ROOT/LeakageEvaluationFramework/logs/task_sensitivity_eval"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/b4_reeval_${TS}.log"
mkdir -p "$LOG_DIR"

# --- step 1: identify affected IDs from the rescue state file (these are the
#             exact instances the patched B4 just re-generated; we must re-eval
#             every one of them since their evidence list has been overwritten).
RESCUE_STATE="$ROOT/runtime/task_sensitivity_full/qwen35_35b/_rescue_state/rescued_b4_stock.jsonl"
AFFECTED=$(python - "$RESCUE_STATE" <<'PY'
import json, sys
ids = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    ids.append(json.loads(line)["instance_id"])
print(",".join(ids))
PY
)
IFS=',' read -r -a ARR <<< "$AFFECTED"
N=${#ARR[@]}
echo "[$(date -Is)] Re-evaluating $N affected B4 stock instances" | tee -a "$LOG"
if [[ $N -eq 0 ]]; then
  echo "Nothing to do." | tee -a "$LOG"
  exit 0
fi

# --- step 2: snapshot + prune existing eval rows for these IDs.
BACKUP="$EVAL_OUT/pre_b4_rescue_${TS}"
mkdir -p "$BACKUP"
for f in instance_results.jsonl claim_results.jsonl shapley_results.jsonl errors.jsonl summary.json; do
  if [[ -f "$EVAL_OUT/$f" ]]; then
    cp "$EVAL_OUT/$f" "$BACKUP/$f"
  fi
done
echo "[$(date -Is)] Snapshotted to $BACKUP" | tee -a "$LOG"

python - "$EVAL_OUT" "$AFFECTED" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
eval_out = Path(sys.argv[1])
affected = set(s for s in sys.argv[2].split(",") if s)
for fname in ("instance_results.jsonl", "claim_results.jsonl",
              "shapley_results.jsonl", "errors.jsonl"):
    fp = eval_out / fname
    if not fp.exists():
        continue
    kept, dropped = [], 0
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        iid = row.get("instance_id")
        if iid in affected:
            dropped += 1
            continue
        kept.append(line)
    fp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"  {fname}: dropped {dropped} rows, kept {len(kept)}")
PY

# --- step 3: re-run the eval pipeline just for these IDs.
echo "[$(date -Is)] Launching eval --method b4 --instance-ids ($N ids)" | tee -a "$LOG"
cd "$ROOT"
uv run python -m LeakageEvaluationFramework.leakage_eval.cli run \
    --config      "$CFG" \
    --input       "$COMPLETIONS" \
    --dataset     "$META" \
    --output      "$EVAL_OUT" \
    --model       qwen/qwen3.5-35b-a3b \
    --method      b4 \
    --instance-ids "$AFFECTED" \
    --concurrency 4 \
    --checkpoint-every 5 \
    2>&1 | tee -a "$LOG"

echo "[$(date -Is)] Eval finished." | tee -a "$LOG"

# --- step 4: print refreshed per-task aggregates.
python - "$EVAL_OUT/instance_results.jsonl" <<'PY' | tee -a "$LOG"
import json, sys, statistics
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by_task = defaultdict(list)
for r in rows:
    by_task[r["task"]].append(r)
print()
print(f"{'task':<10}{'n':<6}{'OLR':<10}{'DCLR':<10}{'Perf':<10}")
print("-" * 50)
for task, rs in sorted(by_task.items()):
    olr  = statistics.mean([r["olr"]          for r in rs if r.get("olr")          is not None])
    dclr = statistics.mean([r["shapley_dclr"] for r in rs if r.get("shapley_dclr") is not None])
    perf = statistics.mean([r["performance"]  for r in rs if r.get("performance")  is not None])
    print(f"{task:<10}{len(rs):<6}{olr:<10.4f}{dclr:<10.4f}{perf:<10.4f}")
PY
