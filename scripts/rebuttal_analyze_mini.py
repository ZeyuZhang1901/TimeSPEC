"""Analyze the compact rebuttal mini-pipeline and write reviewer-ready results."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from LeakageEvaluationFramework.leakage_eval.io.loaders import (
    build_eval_inputs,
    load_completion_rows,
    load_dataset_metadata,
)
from LeakageEvaluationFramework.leakage_eval.metrics.dclr import shapley_dclr
from LeakageEvaluationFramework.leakage_eval.shapley.prompts import build_coalition_messages


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "rebuttal_validation" / "results" / "mini_qwen_12"
METHODS = ("b1", "b2", "b3", "b4")
METHOD_NAMES = {
    "b1": "Temporal Hint",
    "b2": "Self-Correction",
    "b3": "Temporal RAG",
    "b4": "TimeSPEC",
}
# Observed foreground runtimes from this exact pilot run.
EVAL_SECONDS = {"b1": 64.693, "b2": 52.166, "b3": 129.091, "b4": 116.863}
GENERATION_SECONDS_TOTAL = 184.880
QWEN_INPUT_USD_PER_MILLION = 0.1625
QWEN_OUTPUT_USD_PER_MILLION = 1.30


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_instance(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[row["instance_id"]].append(row)
    return out


def shapley_map(path: Path) -> dict[str, dict]:
    return {row["instance_id"]: row for row in read_jsonl(path)}


def leakage_map(path: Path) -> dict[str, list[bool]]:
    grouped = by_instance(read_jsonl(path))
    return {
        instance_id: [bool(row["is_leaked"]) for row in sorted(rows, key=lambda value: value["claim_id"])]
        for instance_id, rows in grouped.items()
    }


def loo_dclr(eval_dir: Path) -> dict[str, float]:
    coalition_rows = by_instance(read_jsonl(eval_dir / "audit" / "coalitions.jsonl"))
    labels = leakage_map(eval_dir / "claim_results.jsonl")
    results: dict[str, float] = {}
    for instance_id, rows in coalition_rows.items():
        valid = [row for row in rows if row.get("value") is not None]
        if not valid:
            continue
        n = max(len(row["mask"]) for row in valid)
        by_mask = {tuple(row["mask"]): float(row["value"]) for row in valid}
        full = tuple([1] * n)
        if full not in by_mask:
            continue
        weights = []
        for index in range(n):
            mask = list(full)
            mask[index] = 0
            key = tuple(mask)
            if key not in by_mask:
                break
            weights.append(abs(by_mask[full] - by_mask[key]))
        if len(weights) != n or len(labels.get(instance_id, [])) != n:
            continue
        results[instance_id] = shapley_dclr(weights, labels[instance_id])["shapley_dclr"]
    return results


def estimate_coalition_tokens(root: Path, method: str) -> dict[str, float]:
    completions_path = root / method / "completions.json"
    dataset_path = ROOT / "rebuttal_validation" / "data" / "autocast_rebuttal_20.jsonl"
    rows = load_completion_rows(completions_path)
    metadata = load_dataset_metadata(dataset_path)
    instances = {
        instance.instance_id: instance
        for instance in build_eval_inputs(
            rows,
            metadata,
            model="qwen/qwen3.5-35b-a3b",
            method=method,
        )
    }
    coalition_rows = read_jsonl(root / "eval" / method / "audit" / "coalitions.jsonl")
    input_tokens = 0
    output_tokens = 0
    calls = 0
    for row in coalition_rows:
        if row.get("is_empty") or row.get("prediction") is None:
            continue
        instance = instances[row["instance_id"]]
        subset = [evidence for evidence, bit in zip(instance.evidence, row["mask"]) if bit]
        system, user = build_coalition_messages(instance, subset)
        input_tokens += math.ceil((len(system) + len(user)) / 4)
        output_tokens += math.ceil(len(row.get("raw_response") or json.dumps(row["prediction"])) / 4)
        calls += 1
    cost = (
        input_tokens * QWEN_INPUT_USD_PER_MILLION
        + output_tokens * QWEN_OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return {
        "calls": calls,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_qwen_cost_usd": cost,
    }


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def summarize_method(root: Path, method: str) -> tuple[dict, list[dict]]:
    eval_dir = root / "eval" / method
    instances = read_jsonl(eval_dir / "instance_results.jsonl")
    summary = read_json(eval_dir / "summary.json")
    assert isinstance(summary, dict)
    loo = loo_dclr(eval_dir)
    olr = [float(row["olr"]) for row in instances]
    dclr = [float(row["shapley_dclr"]) for row in instances]
    gaps = [float(row["full_reconstruction_gap"]) for row in instances if row["full_reconstruction_gap"] is not None]
    high_medium = [row for row in instances if row["faithfulness_status"] in {"high_faithfulness", "medium_faithfulness"}]
    corr = spearmanr(olr, dclr).statistic if len(set(olr)) > 1 and len(set(dclr)) > 1 else 1.0
    completions = read_json(root / method / "completions.json")
    assert isinstance(completions, list)
    regen_cycles = [int(row.get("regen_cycles", 0)) for row in completions]

    taxonomy_usage = read_jsonl(eval_dir / "audit" / "taxonomy_batches.jsonl")
    taxonomy_tokens = sum((row.get("usage") or {}).get("total_tokens", 0) for row in taxonomy_usage)
    coalition_estimate = estimate_coalition_tokens(root, method)
    leaked_ids = [row.get("leaked_claim_ids") or [] for row in completions]
    generation_model_calls = []
    judge_calls = []
    for cycles, leaked in zip(regen_cycles, leaked_ids):
        generation_model_calls.append(1 + cycles + (1 if leaked else 0))
        judge_calls.append((1 + cycles) if method in {"b2", "b4"} else 0)

    result = {
        "method": METHOD_NAMES[method],
        "n": len(instances),
        "performance": mean([float(row["performance"]) for row in instances]),
        "olr": mean(olr),
        "dclr": mean(dclr),
        "olr_dclr_spearman": float(corr),
        "mean_abs_olr_dclr_gap": mean([abs(x - y) for x, y in zip(olr, dclr)]),
        "reconstruction_gap_mean": mean(gaps),
        "reconstruction_gap_median": median(gaps),
        "faithfulness_high_medium_rate": len(high_medium) / len(instances),
        "dclr_high_medium": mean([float(row["shapley_dclr"]) for row in high_medium]),
        "loo_dclr": mean(list(loo.values())),
        "loo_instances": len(loo),
        "claims": int(summary["total_claims"]),
        "leaked_claims": int(summary["total_leaked_claims"]),
        "coalitions": int(summary["total_coalitions_evaluated"]),
        "coalition_model_calls": int(summary["total_model_calls"]),
        "coalition_estimated_input_tokens": coalition_estimate["estimated_input_tokens"],
        "coalition_estimated_output_tokens": coalition_estimate["estimated_output_tokens"],
        "coalition_estimated_qwen_cost_usd": coalition_estimate["estimated_qwen_cost_usd"],
        "taxonomy_tokens": taxonomy_tokens,
        "regen_rate": mean([float(value > 0) for value in regen_cycles]),
        "mean_regen_cycles": mean([float(value) for value in regen_cycles]),
        "mean_generation_model_calls": mean([float(value) for value in generation_model_calls]),
        "mean_generation_judge_calls": mean([float(value) for value in judge_calls]),
        "eval_wall_seconds": EVAL_SECONDS[method],
    }
    return result, instances


def sensitivity(root: Path, repeats: int = 200) -> dict:
    rng = random.Random(42)
    baseline_order: list[str] = []
    baseline_means = {}
    method_data = {}
    for method in METHODS:
        eval_dir = root / "eval" / method
        labels = leakage_map(eval_dir / "claim_results.jsonl")
        shapley = shapley_map(eval_dir / "shapley_results.jsonl")
        method_data[method] = (labels, shapley)
        values = []
        for instance_id, item_labels in labels.items():
            phi = [
                row["shapley_value"]
                for row in sorted(shapley[instance_id]["claim_shapley"], key=lambda value: value["claim_id"])
            ]
            values.append(shapley_dclr(phi, item_labels)["shapley_dclr"])
        baseline_means[method] = mean(values)
    baseline_order = sorted(METHODS, key=lambda method: baseline_means[method])

    out = {}
    for rate in (0.05, 0.10, 0.20):
        changes = []
        flips = 0
        supervision_with_search_preserved = 0
        supervision_without_search_preserved = 0
        for _ in range(repeats):
            perturbed_means = {}
            for method in METHODS:
                labels, shapley = method_data[method]
                modified_by_instance = {instance_id: list(item_labels) for instance_id, item_labels in labels.items()}
                positions = [
                    (instance_id, index)
                    for instance_id, item_labels in modified_by_instance.items()
                    for index in range(len(item_labels))
                ]
                n_flip = round(len(positions) * rate)
                for instance_id, index in rng.sample(positions, min(n_flip, len(positions))):
                    modified_by_instance[instance_id][index] = not modified_by_instance[instance_id][index]
                values = []
                for instance_id, modified in modified_by_instance.items():
                    phi = [
                        row["shapley_value"]
                        for row in sorted(shapley[instance_id]["claim_shapley"], key=lambda value: value["claim_id"])
                    ]
                    values.append(shapley_dclr(phi, modified)["shapley_dclr"])
                perturbed_means[method] = mean(values)
                changes.append(abs(perturbed_means[method] - baseline_means[method]))
            order = sorted(METHODS, key=lambda method: perturbed_means[method])
            flips += order != baseline_order
            supervision_with_search_preserved += perturbed_means["b4"] < perturbed_means["b3"]
            supervision_without_search_preserved += perturbed_means["b2"] < perturbed_means["b1"]
        out[str(rate)] = {
            "mean_absolute_dclr_change": mean(changes),
            "method_order_flip_rate": flips / repeats,
            "timespec_lower_than_temporal_rag_rate": supervision_with_search_preserved / repeats,
            "self_correction_lower_than_temporal_hint_rate": supervision_without_search_preserved / repeats,
            "repeats": repeats,
        }
    return {"baseline_means": baseline_means, "baseline_order": baseline_order, "cross_tier_label_flip": out}


def unresolved_bounds(root: Path) -> dict:
    output = {}
    for method in METHODS:
        eval_dir = root / "eval" / method
        grouped = by_instance(read_jsonl(eval_dir / "claim_results.jsonl"))
        shapley = shapley_map(eval_dir / "shapley_results.jsonl")
        lower = []
        upper = []
        unresolved_count = 0
        for instance_id, rows in grouped.items():
            ordered = sorted(rows, key=lambda row: row["claim_id"])
            phi = [
                row["shapley_value"]
                for row in sorted(shapley[instance_id]["claim_shapley"], key=lambda value: value["claim_id"])
            ]
            unresolved = [bool(row.get("needs_manual_review")) or row.get("confidence") == "low" for row in ordered]
            unresolved_count += sum(unresolved)
            lower_labels = [bool(row["is_leaked"]) and not flag for row, flag in zip(ordered, unresolved)]
            upper_labels = [bool(row["is_leaked"]) or flag for row, flag in zip(ordered, unresolved)]
            lower.append(shapley_dclr(phi, lower_labels)["shapley_dclr"])
            upper.append(shapley_dclr(phi, upper_labels)["shapley_dclr"])
        output[method] = {
            "unresolved_claims": unresolved_count,
            "lower_dclr": mean(lower),
            "upper_dclr": mean(upper),
        }
    return output


def markdown_report(methods: list[dict], robustness: dict, bounds: dict) -> str:
    lines = [
        "# Compact Qwen Rebuttal Pilot",
        "",
        "Targeted validation pilot on 12 public Autocast Treatment questions across three domains.",
        "",
        "## Main results",
        "",
        "| Method | Perf. | OLR | DCLR | Recon. gap | High/Med faithful | LOO-DCLR | Claims | Coalitions | Eval sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in methods:
        lines.append(
            f"| {row['method']} | {row['performance']:.3f} | {row['olr']:.3f} | {row['dclr']:.3f} | "
            f"{row['reconstruction_gap_mean']:.3f} | {row['faithfulness_high_medium_rate']:.1%} | "
            f"{row['loo_dclr']:.3f} | {row['claims']} | {row['coalitions']} | {row['eval_wall_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"Generation wall time for all four methods (shared phases/retrieval): {GENERATION_SECONDS_TOTAL:.1f} seconds.",
            "",
            "## Taxonomy-label sensitivity",
            "",
            f"```json\n{json.dumps(robustness, indent=2)}\n```",
            "",
            "## DCLR bounds for flagged/low-confidence claims",
            "",
            f"```json\n{json.dumps(bounds, indent=2)}\n```",
            "",
            "## Scope",
            "",
            "This is a small validation pilot, not a replacement for the submitted 2,769-instance experiment.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    method_results = []
    instance_rows = {}
    for method in METHODS:
        result, rows = summarize_method(args.root, method)
        method_results.append(result)
        instance_rows[method] = rows
    robustness = sensitivity(args.root)
    bounds = unresolved_bounds(args.root)
    report = {
        "pilot_n_questions": 12,
        "generation_wall_seconds_total": GENERATION_SECONDS_TOTAL,
        "methods": method_results,
        "sensitivity": robustness,
        "unresolved_bounds": bounds,
    }
    out_dir = args.root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(markdown_report(method_results, robustness, bounds), encoding="utf-8")
    print(json.dumps({"output": str(out_dir), "methods": method_results}, indent=2))


if __name__ == "__main__":
    main()
