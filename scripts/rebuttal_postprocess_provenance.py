"""Apply conservative evidence requirements to saved provenance audit results."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"
INPUT = AUDIT / "auto_audit_results.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in INPUT.open(encoding="utf-8") if line.strip()]
    for index, row in enumerate(rows, 1):
        row["sample_id"] = f"R{index:03d}"
        if row["auto_label"] == "A" and not row.get("pre_snapshot_text"):
            row["auto_label"] = "C"
            row["auto_confidence"] = "M"
            row["auto_reason"] = (
                "No usable pre-cutoff archive was available; content-level "
                "availability remains unverified."
            )
    with (AUDIT / "conservative_auto_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "sample_id", "cutoff_date", "retrieved_text", "source_url",
        "pre_snapshot_url", "post_snapshot_url",
        "auto_label", "auto_confidence", "auto_reason",
        "human_confirm", "human_corrected_label", "human_note",
    ]
    with (AUDIT / "human_verification.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    counts = Counter(row["auto_label"] for row in rows)
    high = Counter(row["auto_label"] for row in rows if row["auto_confidence"] == "H")
    summary = {
        "n": len(rows),
        "conservative_auto_counts": dict(counts),
        "high_confidence_counts": dict(high),
        "pre_snapshot_text_usable": sum(bool(row.get("pre_snapshot_text")) for row in rows),
        "verification_file": str(AUDIT / "human_verification.csv"),
    }
    (AUDIT / "conservative_auto_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    shortlist = [
        row
        for row in rows
        if row["auto_label"] in {"A", "B"} and row["auto_confidence"] == "H"
    ]
    lines = [
        "# Provenance Audit: Human Verification Shortlist",
        "",
        "Verify each automatic A/B label. Respond `CONFIRM ALL` or list corrections as "
        "`R### | A/B/C | H/M/L | note`.",
        "",
    ]
    for row in shortlist:
        lines.extend(
            [
                f"## {row['sample_id']} — Auto {row['auto_label']} ({row['auto_confidence']})",
                "",
                f"- Cutoff: {row['cutoff_date']}",
                f"- Claim: {row['retrieved_text'][:500].replace(chr(10), ' ')}",
                f"- Source: {row['source_url']}",
                f"- Pre-cutoff snapshot: {row.get('pre_snapshot_url') or 'none'}",
                f"- Post-cutoff snapshot: {row.get('post_snapshot_url') or 'none'}",
                f"- Auto reason: {row['auto_reason']}",
                "",
            ]
        )
    (AUDIT / "verification_shortlist.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
