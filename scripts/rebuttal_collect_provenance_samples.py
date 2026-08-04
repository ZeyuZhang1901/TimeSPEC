"""Collect 100 actual Perplexity retrieval outputs for provenance auditing."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from LeakageEvaluationFramework.generation.perplexity import PerplexitySearchClient
from LeakageEvaluationFramework.generation.prompts import build_query_generation_prompt
from LeakageEvaluationFramework.generation.runner import _parse_query_json, build_config
from LeakageEvaluationFramework.generation.schemas import instance_from_row
from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import (
    OpenRouterConfig,
    OpenRouterTextClient,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "rebuttal_validation" / "data" / "autocast_provenance_60.jsonl"
OUTPUT = ROOT / "rebuttal_validation" / "provenance_audit" / "retrieval_samples.jsonl"


async def collect_instance(
    instance,
    query_client: OpenRouterTextClient,
    search_client: PerplexitySearchClient,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    async with semaphore:
        query_text = await query_client.call_text(build_query_generation_prompt(instance))
        queries = _parse_query_json(query_text)[:3]
    rows = []
    for query in queries:
        result = await asyncio.to_thread(
            search_client.search,
            query,
            cutoff_date=instance.temporal_cutoff,
            max_results=1,
        )
        if not result:
            continue
        hit = result[0]
        rows.append(
            {
                "instance_id": instance.instance_id,
                "domain": instance.domain,
                "question": instance.question,
                "cutoff_date": instance.temporal_cutoff,
                "query": query,
                "retrieved_text": hit.get("snippet", ""),
                "source_url": hit.get("url", ""),
                "citations": hit.get("citations", []),
                "api_filter_claim": f"before {instance.temporal_cutoff}",
            }
        )
    return rows


async def run(instances, limit: int) -> list[dict]:
    cfg = build_config(
        model_name="qwen/qwen3.5-35b-a3b",
        temperature=0.0,
        max_tokens=4096,
        judge_model="google/gemini-3.1-flash-lite",
        batch_size=4,
        max_regen_cycles=1,
    )
    query_client = OpenRouterTextClient(
        OpenRouterConfig(model=cfg.judge_model, temperature=0.0, max_tokens=1024),
        api_key=cfg.openrouter_api_key,
    )
    search_client = PerplexitySearchClient(cfg.perplexity_api_key)
    semaphore = asyncio.Semaphore(5)
    tasks = [
        collect_instance(instance, query_client, search_client, semaphore)
        for instance in instances
    ]
    results = await asyncio.gather(*tasks)
    flattened = [row for group in results for row in group]
    # Prioritize rows with a source URL, but retain source-less outputs as
    # meaningful unverifiable cases.
    flattened.sort(key=lambda row: (not bool(row["source_url"]), row["instance_id"], row["query"]))
    selected = flattened[:limit]
    for index, row in enumerate(selected, 1):
        row["sample_id"] = f"P{index:03d}"
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    rows = [json.loads(line) for line in DATASET.open(encoding="utf-8") if line.strip()]
    instances = [instance_from_row(row) for row in rows]
    samples = asyncio.run(run(instances, args.limit))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "n": len(samples),
                "with_source": sum(bool(row["source_url"]) for row in samples),
                "without_source": sum(not bool(row["source_url"]) for row in samples),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
