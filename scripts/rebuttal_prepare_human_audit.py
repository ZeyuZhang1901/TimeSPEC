"""Prepare a blinded 60-claim C0-C8 annotation pack and private answer key."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHALLENGE = ROOT / "rebuttal_validation" / "data" / "taxonomy_date_challenge.jsonl"
AUTO = ROOT / "rebuttal_validation" / "results" / "judge_robustness_final_v2" / "predictions.jsonl"
OUTPUT = ROOT / "rebuttal_validation" / "human_audit"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    challenge = read_jsonl(CHALLENGE)
    by_category = {
        category: [row for row in challenge if row["subset"] == "taxonomy" and row["gold_category"] == category]
        for category in [f"C{i}" for i in range(9)]
    }
    selected = [row for category_rows in by_category.values() for row in category_rows[:5]]
    selected_ids = {row["id"] for row in selected}
    priority = [
        "c1_08", "c1_09", "c2_05", "c4_05", "c4_06", "c4_07", "c4_08", "c4_09",
        "c7_01", "c7_03", "c8_01", "c8_03",
        "boundary_01_before", "boundary_01_after", "boundary_05_before",
    ]
    by_id = {row["id"]: row for row in challenge}
    for item_id in priority:
        if item_id not in selected_ids:
            selected.append(by_id[item_id])
            selected_ids.add(item_id)
    for row in challenge:
        if len(selected) >= 60:
            break
        if row["id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["id"])
    selected = selected[:60]

    auto_rows = [
        row
        for row in read_jsonl(AUTO)
        if row["judge_model"] == "google/gemini-3.1-flash-lite" and row["repeat"] == 1
    ]
    auto_by_id = {row["id"]: row for row in auto_rows}
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "annotation_id", "claim_text", "cutoff_date", "target_description", "source_url",
        "atomicity", "human_category", "human_tier", "human_effective_date",
        "human_is_leaked", "confidence", "notes",
    ]
    for annotator in ("annotator_a", "annotator_b"):
        with (OUTPUT / f"{annotator}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in selected:
                writer.writerow(
                    {
                        "annotation_id": row["id"],
                        "claim_text": row["fact"],
                        "cutoff_date": row["cutoff_date"],
                        "target_description": row["target_description"],
                        "source_url": row["source_url"],
                    }
                )

    with (OUTPUT / "private_answer_key.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            auto = auto_by_id.get(row["id"], {})
            handle.write(
                json.dumps(
                    {
                        "annotation_id": row["id"],
                        "gold_category": row["gold_category"],
                        "gold_tier": row["gold_tier"],
                        "gold_effective_date": row["gold_effective_source_date"],
                        "gold_is_leaked": row["gold_is_leaked"],
                        "gemini_category": (auto.get("prediction") or {}).get("category"),
                        "gemini_effective_date": (auto.get("prediction") or {}).get("effective_source_date"),
                        "gemini_is_leaked": (auto.get("prediction") or {}).get("is_leaked"),
                    }
                )
                + "\n"
            )
    (OUTPUT / "README.md").write_text(
        "# Human Taxonomy Audit\n\n"
        "Annotators should work independently and must not open `private_answer_key.jsonl`.\n"
        "Use Appendix B as the rubric. Fill atomicity, C0-C8 category, operational tier,\n"
        "effective date or `unresolved`, leakage label, confidence, and notes.\n"
        "After both files are complete, adjudicate disagreements before scoring.\n",
        encoding="utf-8",
    )
    print(json.dumps({"n_claims": len(selected), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
