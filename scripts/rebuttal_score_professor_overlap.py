"""Score student/professor provenance agreement and flag adjudication cases."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"


def kappa(pairs: list[tuple[str, str]]) -> float:
    labels = ("A", "B", "C")
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum((left[label] / n) * (right[label] / n) for label in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main() -> None:
    with (AUDIT / "human_verification.csv").open(encoding="utf-8") as handle:
        student = {row["sample_id"]: row for row in csv.DictReader(handle)}
    professor = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (AUDIT / "professor_labels.jsonl").open(encoding="utf-8")
            if line.strip()
        )
    }
    temporal_pairs = []
    disagreements = []
    factual_invalid = []
    for sample_id, prof in professor.items():
        student_label = student[sample_id]["human_corrected_label"]
        if prof["label"] == "F":
            factual_invalid.append(
                {
                    "sample_id": sample_id,
                    "student_label": student_label,
                    "professor_label": "F",
                    "note": prof["note"],
                }
            )
            continue
        temporal_pairs.append((student_label, prof["label"]))
        if student_label != prof["label"]:
            disagreements.append(
                {
                    "sample_id": sample_id,
                    "student_label": student_label,
                    "professor_label": prof["label"],
                    "note": prof["note"],
                }
            )
    report = {
        "n_overlap": len(professor),
        "n_temporally_scorable": len(temporal_pairs),
        "n_factual_invalid": len(factual_invalid),
        "exact_temporal_agreement": sum(a == b for a, b in temporal_pairs)
        / len(temporal_pairs),
        "cohen_kappa": kappa(temporal_pairs),
        "disagreements": disagreements,
        "factual_invalid": factual_invalid,
        "adjudication_note": (
            "F denotes a factually false/unsupported claim and is excluded from "
            "A/B/C temporal agreement. R040 should be adjudicated because the "
            "professor note identifies a post-cutoff formal adoption date."
        ),
    }
    (AUDIT / "professor_overlap_score.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
