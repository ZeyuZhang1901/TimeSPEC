"""Sample treatment questions with crawlable source URLs for the camera-ready Snapshot-RAG run."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to binary.jsonl")
    parser.add_argument("--output", required=True, help="Output sampled jsonl")
    parser.add_argument("--n", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.dataset).open(encoding="utf-8") if line.strip()]
    eligible = [
        row for row in rows
        if row.get("group") == "treatment" and row.get("background_source_urls")
    ]
    rng = random.Random(args.seed)
    sample = rng.sample(eligible, min(args.n, len(eligible)))
    sample.sort(key=lambda row: row["id"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "total_rows": len(rows),
        "eligible_treatment": len(eligible),
        "sampled": len(sample),
        "seed": args.seed,
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
