"""Score human labels on atomic claims flagged as post-cutoff."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"


def main() -> None:
    labels = [
        json.loads(line)
        for line in (AUDIT / "atomic_flagged_human_labels.jsonl").open(
            encoding="utf-8"
        )
        if line.strip()
    ]
    counts = Counter(row["label"] for row in labels)
    temporal = counts["A"] + counts["B"]
    parent_labels = defaultdict(list)
    for row in labels:
        parent = row["claim_id"].split(".", 1)[0]
        parent_labels[parent].append(row["label"])
    parent_with_confirmed_leak = {
        parent: "B" in values for parent, values in parent_labels.items()
    }
    professor = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (AUDIT / "professor_labels.jsonl").open(encoding="utf-8")
            if line.strip()
        )
        if row["label"] in {"A", "B"} and row["sample_id"] != "R040"
    }
    predicted = {
        row["sample_id"]: bool(row["predicted_is_leaked"])
        for row in (
            json.loads(line)
            for line in (
                AUDIT / "verifier_vs_professor_predictions.jsonl"
            ).open(encoding="utf-8")
            if line.strip()
        )
    }
    adjudicated_gold = {
        sample_id: (row["label"] == "B") for sample_id, row in professor.items()
    }
    for parent, confirmed in parent_with_confirmed_leak.items():
        adjudicated_gold[parent] = confirmed
    tp = fp = tn = fn = 0
    for sample_id, expected in adjudicated_gold.items():
        value = predicted[sample_id]
        if expected and value:
            tp += 1
        elif expected and not value:
            fn += 1
        elif not expected and value:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    report = {
        "n_flagged_atomic_claims": len(labels),
        "human_label_counts": dict(counts),
        "temporally_scorable_flagged_claims": temporal,
        "flagged_claim_precision_for_post_cutoff": counts["B"] / temporal,
        "factual_error_rate_among_flagged": counts["F"] / len(labels),
        "unresolved_rate_among_flagged": counts["C"] / len(labels),
        "n_parent_paragraphs": len(parent_labels),
        "parents_with_at_least_one_confirmed_post_cutoff_claim": sum(
            parent_with_confirmed_leak.values()
        ),
        "parent_level_contamination_precision": sum(
            parent_with_confirmed_leak.values()
        )
        / len(parent_labels),
        "parents_without_confirmed_post_cutoff_claim": [
            parent
            for parent, confirmed in parent_with_confirmed_leak.items()
            if not confirmed
        ],
        "adjudicated_parent_status": {
            "n": len(adjudicated_gold),
            "accuracy": (tp + tn) / len(adjudicated_gold),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "interpretation": (
            "Precision is conditional on claims the verifier flagged; safe-claim "
            "recall/overall accuracy requires human labels for an additional sample "
            "of automatically safe atomic claims."
        ),
    }
    (AUDIT / "atomic_adjudication_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
