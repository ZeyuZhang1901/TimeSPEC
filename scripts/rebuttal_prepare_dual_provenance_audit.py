"""Prepare a dual source-level / claim-level adjudication sheet."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"


def main() -> None:
    with (AUDIT / "human_verification.csv").open(encoding="utf-8") as handle:
        human = {row["sample_id"]: row for row in csv.DictReader(handle)}
    auto = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (AUDIT / "conservative_auto_results.jsonl").open(encoding="utf-8")
            if line.strip()
        )
    }
    professor = [
        json.loads(line)
        for line in (AUDIT / "professor_labels.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    fields = [
        "sample_id", "cutoff_date", "retrieved_text", "source_url",
        "pre_snapshot_url", "post_snapshot_url",
        "professor_claim_label", "professor_source_label",
        "student_claim_label", "student_source_label",
        "adjudicated_claim_label", "adjudicated_source_label", "adjudication_note",
    ]
    output = AUDIT / "dual_dimension_adjudication_30.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prof in professor:
            sample_id = prof["sample_id"]
            source = auto[sample_id]
            student = human[sample_id]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "cutoff_date": source["cutoff_date"],
                    "retrieved_text": source["retrieved_text"],
                    "source_url": source["source_url"],
                    "pre_snapshot_url": source.get("pre_snapshot_url") or "",
                    "post_snapshot_url": source.get("post_snapshot_url") or "",
                    # Professor's prior response addressed public knowability.
                    "professor_claim_label": prof["label"],
                    "professor_source_label": "",
                    # Prior student labels mixed the two constructs and must be
                    # confirmed/revised in the dual-dimension sheet.
                    "student_claim_label": student["human_corrected_label"],
                    "student_source_label": source["auto_label"],
                    "adjudicated_claim_label": "",
                    "adjudicated_source_label": "",
                    "adjudication_note": prof["note"],
                }
            )
    (AUDIT / "DUAL_DIMENSION_INSTRUCTIONS.md").write_text(
        "# Dual-Dimension Provenance Audit\n\n"
        "Do not edit the retrieved claim text, including factual errors. Label those errors `F`.\n\n"
        "## CLAIM label: public knowability anywhere\n"
        "- A: exact claim was publicly knowable before cutoff.\n"
        "- B: exact claim became knowable only after cutoff.\n"
        "- C: insufficient evidence.\n"
        "- F: claim is factually false/unsupported as stated.\n\n"
        "## SOURCE label: cited-page version compliance\n"
        "- A: exact information appears in the cited source's pre-cutoff version.\n"
        "- B: exact information appears only in a post-cutoff version.\n"
        "- C: source history is insufficient.\n\n"
        "The professor's prior labels are prefilled under `professor_claim_label`. "
        "The professor should fill only `professor_source_label`. The student should "
        "review both of their prefilled columns. Then jointly fill the adjudicated columns.\n",
        encoding="utf-8",
    )
    print(json.dumps({"n": len(professor), "csv": str(output)}, indent=2))


if __name__ == "__main__":
    main()
