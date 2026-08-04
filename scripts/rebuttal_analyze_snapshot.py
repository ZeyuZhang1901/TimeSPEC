"""Compare Snapshot-RAG, live Temporal RAG, and TimeSPEC on matched questions."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "rebuttal_validation" / "results"
OUTPUT = RESULTS / "snapshot_rag" / "comparison.json"


def rows(path: Path, ids: set[str]) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if (row := json.loads(line))["instance_id"] in ids]


def summarize(values: list[dict]) -> dict:
    return {
        "n": len(values),
        "performance": statistics.mean(row["performance"] for row in values),
        "olr": statistics.mean(row["olr"] for row in values),
        "dclr": statistics.mean(row["shapley_dclr"] for row in values),
        "reconstruction_gap": statistics.mean(row["full_reconstruction_gap"] for row in values),
        "high_medium_faithfulness_rate": statistics.mean(
            row["faithfulness_status"] != "low_faithfulness" for row in values
        ),
    }


def main() -> None:
    dataset = RESULTS / "snapshot_rag" / "dataset.jsonl"
    ids = {json.loads(line)["id"] for line in dataset.open(encoding="utf-8")}
    paths = {
        "snapshot_rag": RESULTS / "snapshot_rag" / "eval" / "instance_results.jsonl",
        "live_temporal_rag": RESULTS / "mini_qwen_12" / "eval" / "b3" / "instance_results.jsonl",
        "timespec": RESULTS / "mini_qwen_12" / "eval" / "b4" / "instance_results.jsonl",
    }
    summaries = {name: summarize(rows(path, ids)) for name, path in paths.items()}
    live = summaries["live_temporal_rag"]
    snapshot = summaries["snapshot_rag"]
    timespec = summaries["timespec"]
    report = {
        "snapshot_coverage": len(ids) / 12,
        "matched_question_ids": sorted(ids),
        "methods": summaries,
        "snapshot_dclr_reduction_vs_live": 1 - snapshot["dclr"] / live["dclr"],
        "timespec_dclr_reduction_vs_live": 1.0 if live["dclr"] and timespec["dclr"] == 0 else 1 - timespec["dclr"] / live["dclr"],
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
