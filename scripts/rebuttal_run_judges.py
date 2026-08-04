"""Run and score the rebuttal taxonomy/date challenge across judge models."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path

from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import (
    OpenRouterConfig,
    OpenRouterTextClient,
)
from LeakageEvaluationFramework.leakage_eval.schemas import ClaimRecord
from LeakageEvaluationFramework.leakage_eval.taxonomy.date_verifier import LlmDateVerifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "rebuttal_validation" / "data" / "taxonomy_date_challenge.jsonl"
DEFAULT_OUTPUT = ROOT / "rebuttal_validation" / "results" / "judge_robustness"
DEFAULT_RUNS = {
    "google/gemini-3.1-flash-lite": 3,
    "anthropic/claude-sonnet-4": 1,
    "moonshotai/kimi-k2.5": 1,
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tier(category: str) -> str:
    if category in {"C0", "C1"}:
        return "safe"
    if category in {"C7", "C8"}:
        return "leaked"
    return "verify"


def group_rows(rows: list[dict], max_group_size: int = 20) -> list[list[dict]]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["cutoff_date"], row["task"], row["target_description"])
        groups[key].append(row)
    chunks: list[list[dict]] = []
    for group in groups.values():
        for start in range(0, len(group), max_group_size):
            chunks.append(group[start : start + max_group_size])
    return chunks


async def run_group(model: str, repeat: int, rows: list[dict], semaphore: asyncio.Semaphore) -> tuple[list[dict], dict]:
    async with semaphore:
        first = rows[0]
        client = OpenRouterTextClient(OpenRouterConfig(model=model, temperature=0.0, max_tokens=8192))
        verifier = LlmDateVerifier(client, batch_size=20)
        records = [
            ClaimRecord(
                instance_id=row["id"],
                claim_id=index,
                claim_text=row["fact"],
                declared_source_date=row["declared_source_date"],
                cutoff_date=row["cutoff_date"],
                task=row["task"],
                target_description=row["target_description"],
                model="challenge",
                method="judge_validation",
            )
            for index, row in enumerate(rows, 1)
        ]
        decisions = await verifier.verify(records)
        enriched: list[dict] = []
        by_claim = {decision.claim_id: decision for decision in decisions}
        for index, row in enumerate(rows, 1):
            decision = by_claim[index]
            enriched.append(
                {
                    **row,
                    "judge_model": model,
                    "repeat": repeat,
                    "prediction": asdict(decision),
                    "predicted_tier": tier(decision.category),
                }
            )
        usage = client.last_usage
        usage_row = {
            "judge_model": model,
            "repeat": repeat,
            "group_size": len(rows),
            "cutoff_date": first["cutoff_date"],
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        }
        return enriched, usage_row


async def run_all(rows: list[dict], run_map: dict[str, int], concurrency: int) -> tuple[list[dict], list[dict]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    tasks = []
    groups = group_rows(rows)
    for model, repeats in run_map.items():
        for repeat in range(1, repeats + 1):
            for group in groups:
                tasks.append(run_group(model, repeat, group, semaphore))
    results = await asyncio.gather(*tasks)
    predictions: list[dict] = []
    usage: list[dict] = []
    for rows_out, usage_out in results:
        predictions.extend(rows_out)
        usage.append(usage_out)
    return predictions, usage


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def classification_metrics(rows: list[dict]) -> dict:
    labels = [f"C{i}" for i in range(9)]
    confusion = {gold: {pred: 0 for pred in labels} for gold in labels}
    for row in rows:
        confusion[row["gold_category"]][row["prediction"]["category"]] += 1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in labels if gold != label)
        fn = sum(confusion[label][pred] for pred in labels if pred != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
            "support": sum(confusion[label].values()),
        }
    return {
        "accuracy": safe_div(sum(confusion[label][label] for label in labels), len(rows)),
        "macro_f1": sum(value["f1"] for value in per_class.values()) / len(labels),
        "per_class": per_class,
        "confusion": confusion,
    }


def binary_metrics(rows: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        gold = bool(row["gold_is_leaked"])
        pred = bool(row["prediction"]["is_leaked"])
        if gold and pred:
            tp += 1
        elif gold and not pred:
            fn += 1
        elif not gold and pred:
            fp += 1
        else:
            tn += 1
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "accuracy": safe_div(tp + tn, len(rows)),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def date_errors(rows: list[dict]) -> dict:
    errors: list[int] = []
    for row in rows:
        if row["subset"] != "date_boundary":
            continue
        predicted = row["prediction"]["effective_source_date"]
        gold = row["gold_effective_source_date"]
        if not predicted or not gold:
            continue
        try:
            errors.append(abs((date.fromisoformat(predicted) - date.fromisoformat(gold)).days))
        except ValueError:
            continue
    return {
        "n": len(errors),
        "mean_absolute_days": sum(errors) / len(errors) if errors else None,
        "median_absolute_days": sorted(errors)[len(errors) // 2] if errors else None,
        "within_7_days": safe_div(sum(value <= 7 for value in errors), len(errors)),
        "within_30_days": safe_div(sum(value <= 30 for value in errors), len(errors)),
    }


def agreement_summary(predictions: list[dict], primary_model: str) -> dict:
    by_model_repeat: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in predictions:
        by_model_repeat[(row["judge_model"], row["repeat"])][row["id"]] = row

    primary_keys = sorted(key for key in by_model_repeat if key[0] == primary_model)
    common_ids = sorted(set.intersection(*(set(by_model_repeat[key]) for key in primary_keys)))
    repeat_fields = {}
    for field in ("category", "is_leaked", "effective_source_date"):
        matches = 0
        for item_id in common_ids:
            values = [by_model_repeat[key][item_id]["prediction"][field] for key in primary_keys]
            matches += len(set(values)) == 1
        repeat_fields[field] = safe_div(matches, len(common_ids))

    base_key = (primary_model, 1)
    cross = {}
    for key, rows in by_model_repeat.items():
        if key[1] != 1 or key == base_key:
            continue
        ids = sorted(set(by_model_repeat[base_key]) & set(rows))
        cross[key[0]] = {
            "n": len(ids),
            "category_agreement": safe_div(
                sum(by_model_repeat[base_key][item_id]["prediction"]["category"] == rows[item_id]["prediction"]["category"] for item_id in ids),
                len(ids),
            ),
            "tier_agreement": safe_div(
                sum(by_model_repeat[base_key][item_id]["predicted_tier"] == rows[item_id]["predicted_tier"] for item_id in ids),
                len(ids),
            ),
            "leakage_agreement": safe_div(
                sum(by_model_repeat[base_key][item_id]["prediction"]["is_leaked"] == rows[item_id]["prediction"]["is_leaked"] for item_id in ids),
                len(ids),
            ),
        }
    return {"repeatability": repeat_fields, "cross_judge_vs_primary": cross}


def summarize(predictions: list[dict], usage: list[dict], primary_model: str) -> dict:
    by_run: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in predictions:
        by_run[(row["judge_model"], row["repeat"])].append(row)
    runs = {}
    for (model, repeat), rows in sorted(by_run.items()):
        taxonomy_rows = [row for row in rows if row["subset"] == "taxonomy"]
        runs[f"{model}#run{repeat}"] = {
            "n": len(rows),
            "taxonomy": classification_metrics(taxonomy_rows),
            "tier_accuracy": safe_div(sum(row["gold_tier"] == row["predicted_tier"] for row in taxonomy_rows), len(taxonomy_rows)),
            "leakage": binary_metrics(rows),
            "date_boundary": date_errors(rows),
        }
    return {
        "primary_model": primary_model,
        "runs": runs,
        "agreement": agreement_summary(predictions, primary_model),
        "usage": {
            "requests": len(usage),
            "prompt_tokens": sum(row["prompt_tokens"] or 0 for row in usage),
            "completion_tokens": sum(row["completion_tokens"] or 0 for row in usage),
            "total_tokens": sum(row["total_tokens"] or 0 for row in usage),
        },
    }


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Judge Robustness Results",
        "",
        f"Primary judge: `{summary['primary_model']}`",
        "",
        "## Run-level results",
        "",
    ]
    for name, result in summary["runs"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- C0-C8 accuracy: {result['taxonomy']['accuracy']:.3f}",
                f"- C0-C8 macro-F1: {result['taxonomy']['macro_f1']:.3f}",
                f"- Operational-tier accuracy: {result['tier_accuracy']:.3f}",
                f"- Leakage F1: {result['leakage']['f1']:.3f}",
                f"- Leakage FP/FN: {result['leakage']['fp']}/{result['leakage']['fn']}",
                f"- Date MAE (days): {result['date_boundary']['mean_absolute_days']}",
                "",
            ]
        )
    lines.extend(["## Agreement", "", f"```json\n{json.dumps(summary['agreement'], indent=2)}\n```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_runs(values: list[str] | None) -> dict[str, int]:
    if not values:
        return DEFAULT_RUNS
    out = {}
    for value in values:
        model, repeats = value.rsplit("=", 1)
        out[model] = int(repeats)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run", action="append", help="MODEL=REPEATS; may be repeated")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    run_map = parse_runs(args.run)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions, usage = asyncio.run(run_all(rows, run_map, args.concurrency))

    prediction_path = args.output / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output / "usage.jsonl").open("w", encoding="utf-8") as handle:
        for row in usage:
            handle.write(json.dumps(row) + "\n")

    primary = "google/gemini-3.1-flash-lite"
    summary = summarize(predictions, usage, primary)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, args.output / "summary.md")
    print(json.dumps({"output": str(args.output), "n_predictions": len(predictions), "usage": summary["usage"]}, indent=2))


if __name__ == "__main__":
    main()
