"""Prepare a blinded, re-shuffled annotation sheet for the second independent annotator.

Same 120 claims as the original human taxonomy audit, but in a different random
order with fresh blind IDs (A001..A120), so the second annotation is independent
of the first. Adds effective-date and leakage columns so inter-annotator agreement
can be scored at category, tier, and leakage granularity.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "rebuttal_validation" / "data" / "taxonomy_date_challenge.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "camera_ready" / "e1_human"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = [json.loads(line) for line in INPUT.open(encoding="utf-8") if line.strip()]
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fields = [
        "annotator2_id", "claim_text", "cutoff_date", "target_description", "source_url",
        "human_category", "human_effective_date", "human_is_leaked", "confidence", "note",
    ]
    mapping = []
    with (output / "annotator2_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, original_index in enumerate(order, 1):
            row = rows[original_index]
            blind_id = f"A{rank:03d}"
            writer.writerow({
                "annotator2_id": blind_id,
                "claim_text": row["fact"],
                "cutoff_date": row["cutoff_date"],
                "target_description": row["target_description"],
                "source_url": row["source_url"],
            })
            mapping.append({
                "annotator2_id": blind_id,
                "original_id": row["id"],
                "subset": row["subset"],
            })
    (output / "annotator2_private_id_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )

    instructions = [
        "# Second-Annotator C0-C8 Audit (120 Claims)",
        "",
        "You are the second independent annotator. Work alone, using only the",
        "Appendix B taxonomy rubric. Do NOT consult the first annotator, any",
        "judge-result files, or `annotator2_private_id_mapping.json`.",
        "",
        "For each claim in `annotator2_labels.csv`, fill in:",
        "",
        "- `human_category`: one of C0-C8 (apply precedence C7 > C8 > C6 > C3 > C4 > C5 > C2 > C1 > C0)",
        "- `human_effective_date`: earliest date (YYYY-MM-DD) the claim was publicly",
        "  knowable, or `timeless` for C0/C1, or `unresolved` if you cannot decide",
        "- `human_is_leaked`: `yes` if effective date is after the cutoff (C7/C8 are",
        "  always `yes`), otherwise `no`",
        "- `confidence`: high / medium / low",
        "- `note`: required only for low-confidence or ambiguous cases",
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
        "Estimated effort: 3-5 hours. You may look up public sources to establish",
        "dates, but do not use LLM assistants for the category decision.",
    ]
    (output / "ANNOTATOR2_INSTRUCTIONS.md").write_text("\n".join(instructions), encoding="utf-8")

    print(json.dumps({
        "n": len(rows),
        "csv": str(output / "annotator2_labels.csv"),
        "instructions": str(output / "ANNOTATOR2_INSTRUCTIONS.md"),
        "seed": args.seed,
    }, indent=2))


if __name__ == "__main__":
    main()
