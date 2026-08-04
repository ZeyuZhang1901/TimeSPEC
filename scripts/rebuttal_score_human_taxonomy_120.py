"""Score Gemini/Claude/Kimi taxonomy labels against human C0-C8 labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HUMAN_ROOT = ROOT / "rebuttal_validation" / "human_taxonomy_120"
PREDICTIONS = (
    ROOT
    / "rebuttal_validation"
    / "results"
    / "judge_robustness_final_v2"
    / "predictions.jsonl"
)
CATEGORIES = [f"C{i}" for i in range(9)]


def tier(category: str) -> str:
    if category in {"C0", "C1"}:
        return "safe"
    if category in {"C7", "C8"}:
        return "leaked"
    return "verify"


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def cohen_kappa(gold: list[str], predicted: list[str], labels: list[str]) -> float:
    n = len(gold)
    observed = sum(left == right for left, right in zip(gold, predicted)) / n
    gold_counts = Counter(gold)
    pred_counts = Counter(predicted)
    expected = sum(
        (gold_counts[label] / n) * (pred_counts[label] / n)
        for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def classification(gold: list[str], predicted: list[str]) -> dict:
    per_class = {}
    for label in CATEGORIES:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
            "support": sum(g == label for g in gold),
        }
    gold_tier = [tier(value) for value in gold]
    pred_tier = [tier(value) for value in predicted]
    return {
        "n": len(gold),
        "accuracy": sum(g == p for g, p in zip(gold, predicted)) / len(gold),
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(CATEGORIES),
        "cohen_kappa": cohen_kappa(gold, predicted, CATEGORIES),
        "tier_accuracy": sum(g == p for g, p in zip(gold_tier, pred_tier)) / len(gold),
        "tier_kappa": cohen_kappa(gold_tier, pred_tier, ["safe", "verify", "leaked"]),
        "per_class": per_class,
        "errors": [
            {"human_id": index, "gold": g, "predicted": p}
            for index, (g, p) in enumerate(zip(gold, predicted), 1)
            if g != p
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument(
        "--output-prefix",
        default="human_vs_judges",
        help="Output filename prefix under human_taxonomy_120.",
    )
    args = parser.parse_args()
    with (HUMAN_ROOT / "human_labels.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = [row["human_id"] for row in rows if row["human_category"] not in CATEGORIES]
    if missing:
        raise ValueError(f"Missing/invalid human labels: {missing}")
    human = {row["human_id"]: row["human_category"] for row in rows}
    mapping = json.loads((HUMAN_ROOT / "private_id_mapping.json").read_text(encoding="utf-8"))
    original_to_human = {row["original_id"]: row["human_id"] for row in mapping}
    subset_by_human = {row["human_id"]: row["subset"] for row in mapping}

    predictions = [
        json.loads(line)
        for line in args.predictions.open(encoding="utf-8")
        if line.strip()
    ]
    grouped = {}
    for row in predictions:
        key = f"{row['judge_model']}#run{row['repeat']}"
        grouped.setdefault(key, {})[original_to_human[row["id"]]] = row["prediction"]["category"]

    report = {}
    for key, predicted_by_human in sorted(grouped.items()):
        ids_all = sorted(human)
        ids_balanced = [item_id for item_id in ids_all if subset_by_human[item_id] == "taxonomy"]
        report[key] = {
            "all_120": classification(
                [human[item_id] for item_id in ids_all],
                [predicted_by_human[item_id] for item_id in ids_all],
            ),
            "balanced_90": classification(
                [human[item_id] for item_id in ids_balanced],
                [predicted_by_human[item_id] for item_id in ids_balanced],
            ),
        }

    output = HUMAN_ROOT / f"{args.output_prefix}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        key: {
            "all_120_accuracy": value["all_120"]["accuracy"],
            "balanced_90_accuracy": value["balanced_90"]["accuracy"],
            "balanced_90_macro_f1": value["balanced_90"]["macro_f1"],
            "balanced_90_kappa": value["balanced_90"]["cohen_kappa"],
            "balanced_90_tier_accuracy": value["balanced_90"]["tier_accuracy"],
            "balanced_90_tier_kappa": value["balanced_90"]["tier_kappa"],
        }
        for key, value in report.items()
    }
    (HUMAN_ROOT / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
