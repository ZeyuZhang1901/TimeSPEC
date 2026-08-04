"""Evaluate two semantically equivalent complete C0-C8 prompt variants."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from LeakageEvaluationFramework.leakage_eval.taxonomy import prompts
from rebuttal_run_judges import read_jsonl, run_all, tier


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "rebuttal_validation" / "data" / "taxonomy_date_challenge.jsonl"
BASELINE = ROOT / "rebuttal_validation" / "results" / "judge_robustness_final_v2" / "predictions.jsonl"
OUTPUT = ROOT / "rebuttal_validation" / "results" / "prompt_robustness"
MODEL = "google/gemini-3.1-flash-lite"


VARIANTS = {
    "rules_first": """
Assign exactly one category using this precedence: C7 > C8 > C6 > C3 > C4 > C5 > C2 > C1 > C0.
C7: the prediction target/outcome itself; always leaked.
C8: a consequence that can exist only after the target outcome; always leaked.
C6: information explicitly attributed to a dated publication, report, filing, opinion, or release; use its release date.
C3: a numeric measurement such as a rate, count, price, revenue, score, or salary; use its public-release date.
C4: a comparison or relation among entities, values, or periods; use the latest date required by the comparison.
C5: a discrete occurrence such as an announcement, decision, signing, filing, launch, enactment, or appointment.
C2: any remaining time-indexed condition, role, status, or trend.
C1: a stable entity attribute publicly knowable through the cutoff; changing/time-indexed attributes are not C1.
C0: a timeless definition, mathematical identity, or logical truth.
When multiple categories apply, choose the highest-precedence category.
""",
    "tier_first": """
Classify each claim into one of nine mutually exclusive categories.
LEAKED TIER: C7 is the exact target result being forecast. C8 is a downstream consequence of that result.
SOURCE-DATE TIER: C6 explicitly cites a dated publication/source. C3 states a quantitative measurement.
C4 compares multiple facts or periods and uses the latest required date. C5 describes a discrete event.
C2 is the residual category for any other temporally indexed state, role, condition, or trend.
SAFE TIER: C1 is a genuinely stable entity attribute already public through the cutoff. C0 is definitional,
mathematical, logical, or timeless.
Resolve overlap with the strict order C7 > C8 > C6 > C3 > C4 > C5 > C2 > C1 > C0.
C7/C8 are always leaked. C0/C1 are safe. C2-C6 require the effective public-availability date.
""",
}


def keyed(rows: list[dict]) -> dict[str, dict]:
    return {
        row["id"]: row
        for row in rows
        if row["judge_model"] == MODEL and row["repeat"] == 1
    }


def compare(reference: dict[str, dict], candidate: dict[str, dict]) -> dict:
    ids = sorted(set(reference) & set(candidate))
    return {
        "n": len(ids),
        "category_agreement": sum(
            reference[item_id]["prediction"]["category"] == candidate[item_id]["prediction"]["category"]
            for item_id in ids
        )
        / len(ids),
        "tier_agreement": sum(
            tier(reference[item_id]["prediction"]["category"]) == tier(candidate[item_id]["prediction"]["category"])
            for item_id in ids
        )
        / len(ids),
        "leakage_agreement": sum(
            reference[item_id]["prediction"]["is_leaked"] == candidate[item_id]["prediction"]["is_leaked"]
            for item_id in ids
        )
        / len(ids),
        "date_agreement": sum(
            reference[item_id]["prediction"]["effective_source_date"]
            == candidate[item_id]["prediction"]["effective_source_date"]
            for item_id in ids
        )
        / len(ids),
        "gold_category_accuracy": sum(
            candidate[item_id]["gold_category"] == candidate[item_id]["prediction"]["category"]
            for item_id in ids
        )
        / len(ids),
        "gold_leakage_accuracy": sum(
            candidate[item_id]["gold_is_leaked"] == candidate[item_id]["prediction"]["is_leaked"]
            for item_id in ids
        )
        / len(ids),
    }


def main() -> None:
    challenge = read_jsonl(INPUT)
    baseline = keyed(read_jsonl(BASELINE))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    original_tree = prompts._TAXONOMY_DECISION_TREE
    try:
        for name, taxonomy_text in VARIANTS.items():
            prompts._TAXONOMY_DECISION_TREE = taxonomy_text
            predictions, usage = asyncio.run(run_all(challenge, {MODEL: 1}, concurrency=4))
            with (OUTPUT / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as handle:
                for row in predictions:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            candidate = keyed(predictions)
            results[name] = {
                **compare(baseline, candidate),
                "requests": len(usage),
                "tokens": sum((row["total_tokens"] or 0) for row in usage),
            }
    finally:
        prompts._TAXONOMY_DECISION_TREE = original_tree
    (OUTPUT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
