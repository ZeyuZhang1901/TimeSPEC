"""Generate Snapshot-RAG forecasts from usable pre-cutoff Wayback contexts."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from LeakageEvaluationFramework.generation.methods import b3_rag
from LeakageEvaluationFramework.generation.runner import build_config
from LeakageEvaluationFramework.generation.schemas import instance_from_row


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "rebuttal_validation" / "data" / "autocast_rebuttal_20.jsonl"
CONTEXTS = ROOT / "rebuttal_validation" / "results" / "snapshot_contexts.json"
OUTPUT = ROOT / "rebuttal_validation" / "results" / "snapshot_rag"
MODEL = "qwen/qwen3.5-35b-a3b"


def main() -> None:
    rows = [json.loads(line) for line in DATASET.open(encoding="utf-8") if line.strip()][:12]
    context_report = json.loads(CONTEXTS.read_text(encoding="utf-8"))
    usable = {
        instance_id: row
        for instance_id, row in context_report["contexts"].items()
        if row["usable"]
    }
    selected = [row for row in rows if row["id"] in usable]
    if len(selected) < 8:
        raise RuntimeError(f"Snapshot gate failed: only {len(selected)} usable questions")
    instances = [instance_from_row(row) for row in selected]
    context_map = {}
    for instance in instances:
        snapshot = usable[instance.instance_id]
        context_map[instance.instance_id] = (
            f"FROZEN WEB SNAPSHOT: {snapshot['snapshot_timestamp']}\n"
            f"ORIGINAL SOURCE: {snapshot['source_url']}\n"
            f"ARCHIVED TEXT:\n{snapshot['text'][:3000]}"
        )
    cfg = build_config(
        model_name=MODEL,
        temperature=0.0,
        max_tokens=4096,
        judge_model="google/gemini-3.1-flash-lite",
        batch_size=4,
        max_regen_cycles=1,
    )
    started = time.perf_counter()
    completions = b3_rag.generate(instances, cfg, context_map=context_map)
    elapsed = time.perf_counter() - started
    output_rows = []
    for completion in completions:
        row = asdict(completion)
        row["method"] = "snapshot_rag"
        row["model"] = MODEL
        row["task"] = row["answer_type"]
        output_rows.append(row)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "completions.json").write_text(json.dumps(output_rows, indent=2), encoding="utf-8")
    with (OUTPUT / "dataset.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUTPUT / "run_summary.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "n_questions": len(selected),
                "parse_success": sum(bool(row["parse_success"]) for row in output_rows),
                "generation_wall_seconds": elapsed,
                "snapshot_coverage_of_original_12": len(selected) / 12,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print((OUTPUT / "run_summary.json").read_text())


if __name__ == "__main__":
    main()
