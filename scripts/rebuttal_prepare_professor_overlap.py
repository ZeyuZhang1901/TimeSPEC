"""Prepare a blinded 30-item overlap pack for professor annotation."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"
HUMAN_CSV = AUDIT / "human_verification.csv"
AUTO_JSONL = AUDIT / "conservative_auto_results.jsonl"


def main() -> None:
    with HUMAN_CSV.open(encoding="utf-8") as handle:
        human_rows = {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row["human_corrected_label"]
        }
    auto_rows = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in AUTO_JSONL.open(encoding="utf-8")
            if line.strip()
        )
    }
    strata = defaultdict(list)
    for sample_id, row in human_rows.items():
        strata[row["human_corrected_label"]].append(auto_rows[sample_id])
    rng = random.Random(20260712)
    for rows in strata.values():
        rng.shuffle(rows)
    selected = strata["A"][:14] + strata["B"][:14] + strata["C"][:2]
    rng.shuffle(selected)

    fields = [
        "sample_id", "cutoff_date", "retrieved_text", "source_url",
        "pre_snapshot_url", "post_snapshot_url",
        "professor_label", "professor_confidence", "professor_note",
    ]
    csv_path = AUDIT / "professor_overlap_30.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# Professor Provenance Audit — 30 Blinded Overlap Items",
        "",
        "For each item, assign:",
        "",
        "- `A`: exact information was publicly available before cutoff;",
        "- `B`: exact information became available only after cutoff;",
        "- `C`: historical evidence is insufficient to determine either.",
        "",
        "Use `H`, `M`, or `L` confidence. Work independently without viewing the "
        "student or automatic labels. Enter results in `professor_overlap_30.csv`.",
        "",
    ]
    for index, row in enumerate(selected, 1):
        lines.extend(
            [
                f"## Item {index}: {row['sample_id']}",
                "",
                f"- Cutoff: {row['cutoff_date']}",
                f"- Claim: {row['retrieved_text'][:700].replace(chr(10), ' ')}",
                f"- Source: {row['source_url']}",
                f"- Pre-cutoff snapshot: {row.get('pre_snapshot_url') or 'none'}",
                f"- Post-cutoff snapshot: {row.get('post_snapshot_url') or 'none'}",
                "",
                "Professor label:  ",
                "Confidence:  ",
                "Note:  ",
                "",
            ]
        )
    md_path = AUDIT / "professor_overlap_30.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"n": len(selected), "markdown": str(md_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
