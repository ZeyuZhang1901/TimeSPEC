"""Compare the actual TimeSPEC verifier with professor claim-level labels."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import (
    OpenRouterConfig,
    OpenRouterTextClient,
)
from LeakageEvaluationFramework.leakage_eval.schemas import ClaimRecord
from LeakageEvaluationFramework.leakage_eval.taxonomy.date_verifier import LlmDateVerifier


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"
MODEL = "google/gemini-3.1-flash-lite"


async def verify_one(
    client: OpenRouterTextClient,
    source: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    record = ClaimRecord(
        instance_id=source["sample_id"],
        claim_id=1,
        claim_text=source["retrieved_text"],
        declared_source_date=source.get("auto_evidence_date"),
        cutoff_date=source["cutoff_date"],
        task="binary",
        target_description=source["question"],
        model="retrieval_audit",
        method="timespec_verifier",
    )
    async with semaphore:
        decision = (await LlmDateVerifier(client, batch_size=1).verify([record]))[0]
    return {
        "sample_id": source["sample_id"],
        "predicted_category": decision.category,
        "predicted_effective_date": decision.effective_source_date,
        "predicted_is_leaked": decision.is_leaked,
        "confidence": decision.confidence,
        "reason": decision.reason,
    }


async def run(rows: list[dict]) -> list[dict]:
    client = OpenRouterTextClient(
        OpenRouterConfig(model=MODEL, temperature=0.0, max_tokens=1024)
    )
    semaphore = asyncio.Semaphore(5)
    return await asyncio.gather(*(verify_one(client, row, semaphore) for row in rows))


def metrics(gold: dict[str, bool], predictions: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    disagreements = []
    for row in predictions:
        expected = gold[row["sample_id"]]
        predicted = bool(row["predicted_is_leaked"])
        if expected and predicted:
            tp += 1
        elif expected and not predicted:
            fn += 1
        elif not expected and predicted:
            fp += 1
        else:
            tn += 1
        if expected != predicted:
            disagreements.append(
                {
                    "sample_id": row["sample_id"],
                    "gold_is_leaked": expected,
                    **row,
                }
            )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(predictions),
        "accuracy": (tp + tn) / len(predictions),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "disagreements": disagreements,
    }


def main() -> None:
    professor = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (AUDIT / "professor_labels.jsonl").open(encoding="utf-8")
            if line.strip()
        )
        if row["label"] in {"A", "B"} and row["sample_id"] != "R040"
    }
    sources = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (AUDIT / "conservative_auto_results.jsonl").open(encoding="utf-8")
            if line.strip()
        )
    }
    selected = [sources[sample_id] for sample_id in professor]
    gold = {sample_id: row["label"] == "B" for sample_id, row in professor.items()}
    predictions = asyncio.run(run(selected))
    report = {
        "judge_model": MODEL,
        "gold_definition": "Professor claim-level public knowability: A=safe, B=leaked",
        "excluded": {
            "R006": "factually false",
            "R022": "factually false",
            "R040": "temporally disputed/borderline",
        },
        **metrics(gold, predictions),
    }
    (AUDIT / "verifier_vs_professor.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (AUDIT / "verifier_vs_professor_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
