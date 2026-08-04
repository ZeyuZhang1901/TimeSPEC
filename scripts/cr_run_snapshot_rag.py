"""Generate Snapshot-RAG forecasts from pre-cutoff Wayback contexts (camera-ready E3).

Parametrized version of rebuttal_run_snapshot_rag.py: takes any dataset and
snapshot-context report, selects questions with usable contexts (optionally a
seeded subset of fixed size), and writes completions in the standard format
consumed by the leakage evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from LeakageEvaluationFramework.generation.methods import b3_rag
from LeakageEvaluationFramework.generation.runner import build_config
from LeakageEvaluationFramework.generation.schemas import instance_from_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--contexts", required=True, help="snapshot_contexts.json from cr_snapshot_crawl.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--judge-model", default="google/gemini-3.1-flash-lite")
    parser.add_argument("--subset-size", type=int, default=None, help="Seeded cap on usable questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-instances", type=int, default=None, help="Hard cap (smoke tests)")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.dataset).open(encoding="utf-8") if line.strip()]
    report = json.loads(Path(args.contexts).read_text(encoding="utf-8"))
    usable = {iid: ctx for iid, ctx in report["contexts"].items() if ctx["usable"]}

    selected = [row for row in rows if row["id"] in usable]
    if args.subset_size is not None and len(selected) > args.subset_size:
        rng = random.Random(args.seed)
        selected = rng.sample(selected, args.subset_size)
        selected.sort(key=lambda row: row["id"])
    if args.max_instances is not None:
        selected = selected[: args.max_instances]
    print(f"[snapshot-rag] {len(selected)} questions selected ({len(usable)} usable contexts)")

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
        model_name=args.model,
        temperature=0.0,
        max_tokens=4096,
        judge_model=args.judge_model,
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
        row["model"] = args.model
        row["task"] = row["answer_type"]
        output_rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "completions.json").write_text(
        json.dumps(output_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "dataset.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    parse_success = sum(1 for row in output_rows if row.get("parse_success"))
    (output_dir / "run_summary.json").write_text(json.dumps({
        "n": len(output_rows),
        "parse_success": parse_success,
        "elapsed_s": round(elapsed, 1),
        "model": args.model,
        "usable_contexts_total": len(usable),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(output_rows), "parse_success": parse_success, "elapsed_s": round(elapsed, 1)}, indent=2))


if __name__ == "__main__":
    main()
