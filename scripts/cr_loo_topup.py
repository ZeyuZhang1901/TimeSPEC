"""E5: top up missing leave-one-out coalitions v(N \\ {i}) using cached results.

For each Shapley-computed instance, find claims whose size-(n-1) coalition value
is not in the cached audit log, evaluate only those via the same closed-world
coalition predictor, and write supplemental coalition rows in the cache format.
cr_e5_metric_comparison.py consumes these via --supplement.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import (
    OpenRouterCoalitionPredictor,
)
from LeakageEvaluationFramework.leakage_eval.io.loaders import (
    build_eval_inputs,
    load_completion_rows,
    load_dataset_metadata,
)
from LeakageEvaluationFramework.leakage_eval.shapley.value_function import evaluate_coalition


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


async def topup_method(
    *,
    completions_path: Path,
    dataset_path: Path,
    eval_dir: Path,
    output_path: Path,
    model: str,
    method: str,
    max_instances: int | None,
    concurrency: int,
) -> dict:
    inputs = build_eval_inputs(
        load_completion_rows(completions_path),
        load_dataset_metadata(dataset_path),
        model=model,
        method=method,
    )
    inputs_by_id = {i.instance_id: i for i in inputs}
    shapley = {r["instance_id"]: r for r in load_jsonl(eval_dir / "shapley_results.jsonl")}
    coalitions = defaultdict(list)
    for row in load_jsonl(eval_dir / "audit" / "coalitions.jsonl"):
        coalitions[row["instance_id"]].append(row)

    already = set()
    if output_path.exists():
        for row in load_jsonl(output_path):
            already.add((row["instance_id"], tuple(row["mask"])))

    # Find missing size-(n-1) masks per Shapley-computed instance.
    jobs: list[tuple[str, tuple[int, ...]]] = []
    for iid, srow in shapley.items():
        if not srow.get("claim_shapley") or iid not in inputs_by_id:
            continue
        n = len(inputs_by_id[iid].evidence)
        if n < 2:
            continue
        cached_masks = {
            tuple(c["mask"]) for c in coalitions[iid]
            if c.get("mask") and c.get("value") is not None
        }
        for pos in range(n):
            mask = tuple(1 if j != pos else 0 for j in range(n))
            if mask not in cached_masks and (iid, mask) not in already:
                jobs.append((iid, mask))

    seen_instances = sorted({iid for iid, _ in jobs})
    if max_instances is not None:
        keep = set(seen_instances[:max_instances])
        jobs = [(iid, mask) for iid, mask in jobs if iid in keep]

    print(f"[{method}] {len(jobs)} missing LOO coalitions across {len({i for i, _ in jobs})} instances")
    if not jobs:
        return {"method": method, "topped_up": 0}

    predictor = OpenRouterCoalitionPredictor(model=model, temperature=0.0)
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done = 0

    async def worker(iid: str, mask: tuple[int, ...]) -> None:
        nonlocal done
        instance = inputs_by_id[iid]
        try:
            async with semaphore:
                value, compact = await evaluate_coalition(instance, mask, predictor)
            row = {
                "instance_id": iid,
                "mask": list(mask),
                "coalition_size": sum(mask),
                "value": value,
                "prediction": compact,
                "source": "cr_loo_topup",
                "task": instance.task,
            }
        except Exception as exc:
            row = {
                "instance_id": iid,
                "mask": list(mask),
                "coalition_size": sum(mask),
                "value": None,
                "error": str(exc)[:200],
                "source": "cr_loo_topup",
                "task": instance.task,
            }
        async with lock:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            done += 1
            if done % 20 == 0:
                print(f"[{method}] {done}/{len(jobs)}", flush=True)

    await asyncio.gather(*(worker(iid, mask) for iid, mask in jobs))
    failed = sum(1 for r in load_jsonl(output_path) if r.get("value") is None)
    return {"method": method, "topped_up": done, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--method", required=True)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = asyncio.run(topup_method(
        completions_path=Path(args.completions),
        dataset_path=Path(args.dataset),
        eval_dir=Path(args.eval_dir),
        output_path=output_path,
        model=args.model,
        method=args.method,
        max_instances=args.max_instances,
        concurrency=args.concurrency,
    ))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
