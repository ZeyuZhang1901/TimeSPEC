"""Prepare a blinded human C0-C8 annotation sheet for the 120-claim challenge."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "rebuttal_validation" / "data" / "taxonomy_date_challenge.jsonl"
OUTPUT = ROOT / "rebuttal_validation" / "human_taxonomy_120"


def main() -> None:
    rows = [
        json.loads(line)
        for line in INPUT.open(encoding="utf-8")
        if line.strip()
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mapping = []
    fields = [
        "human_id", "claim_text", "cutoff_date", "target_description",
        "source_url", "human_category", "confidence", "note",
    ]
    with (OUTPUT / "human_labels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            human_id = f"H{index:03d}"
            writer.writerow(
                {
                    "human_id": human_id,
                    "claim_text": row["fact"],
                    "cutoff_date": row["cutoff_date"],
                    "target_description": row["target_description"],
                    "source_url": row["source_url"],
                }
            )
            mapping.append(
                {
                    "human_id": human_id,
                    "original_id": row["id"],
                    "subset": row["subset"],
                }
            )
    (OUTPUT / "private_id_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )

    lines = [
        "# Human C0-C8 Annotation — 120 Claims",
        "",
        "Label each claim using Appendix B. Do not open `private_id_mapping.json` "
        "or any judge-result files before completing the annotations.",
        "",
        "## Category summary",
        "",
        "- C0: timeless definition/logical truth",
        "- C1: stable entity attribute",
        "- C2: residual time-indexed state/status/trend",
        "- C3: quantitative measurement",
        "- C4: comparative/relational claim",
        "- C5: discrete event",
        "- C6: explicit publication/source claim",
        "- C7: target outcome being predicted",
        "- C8: post-outcome consequence",
        "",
        "Apply precedence: C7 > C8 > C6 > C3 > C4 > C5 > C2 > C1 > C0.",
        "",
        "Enter labels in `human_labels.csv`, or respond as:",
        "",
        "```text",
        "H001 | C0",
        "H002 | C1",
        "...",
        "```",
        "",
        "Confidence and notes are optional except for ambiguous cases.",
        "",
        "## Claims",
        "",
    ]
    for index, row in enumerate(rows, 1):
        human_id = f"H{index:03d}"
        lines.extend(
            [
                f"### {human_id}",
                "",
                f"- Claim: {row['fact']}",
                f"- Cutoff: {row['cutoff_date']}",
                f"- Target: {row['target_description'] or 'not applicable'}",
                f"- Source: {row['source_url']}",
                "",
            ]
        )
    (OUTPUT / "human_annotation_120.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n": len(rows),
                "markdown": str(OUTPUT / "human_annotation_120.md"),
                "csv": str(OUTPUT / "human_labels.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
