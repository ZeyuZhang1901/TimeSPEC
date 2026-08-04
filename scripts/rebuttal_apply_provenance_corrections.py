"""Apply student provenance corrections to the human-verification sheet."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"
CSV_PATH = AUDIT / "human_verification.csv"
CORRECTIONS = AUDIT / "student_round2_corrections.jsonl"


def main() -> None:
    corrections = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in CORRECTIONS.open(encoding="utf-8")
            if line.strip()
        )
    }
    with CSV_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    for row in rows:
        correction = corrections.get(row["sample_id"])
        if not correction:
            continue
        row["human_confirm"] = "corrected"
        row["human_corrected_label"] = correction["label"]
        row["human_note"] = (
            f"[{correction['confidence']}] {correction['note']}"
        )
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    labeled = [row for row in rows if row["human_corrected_label"]]
    counts = Counter(row["human_corrected_label"] for row in labeled)
    summary = {
        "human_labeled_items": len(labeled),
        "label_counts": dict(counts),
        "remaining_unlabeled": len(rows) - len(labeled),
        "round2_corrections_applied": len(corrections),
    }
    (AUDIT / "human_progress_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
