"""Re-generate B4 stock completions that emitted post-cutoff or self-flagged evidence.

Why this exists
---------------
The original ``b4_timespec.py`` only invoked the closed-world aggregator when at
least one supervisor-validated claim survived ``max_regen_cycles``. When every
cycle was fully leaked (common for stock tasks where the model has memorised
2020 closing prices), the aggregator was silently skipped and the **last
leaked regeneration was emitted verbatim**.

``b4_timespec.py`` has since been patched with two defences:
  1. Always invoke the aggregator when leakage is detected (even with empty
     validated evidence; the aggregator's "use ONLY this evidence, do not
     introduce new facts" instruction degrades gracefully).
  2. Final ``_scrub_evidence`` pass that drops any claim whose ``id`` is in the
     last supervisor pass's flagged set OR whose ``source_date`` is strictly
     later than the temporal cutoff.

This script re-runs B4 with the patched code for the specific instances that
were tainted by the original bug, splicing the rescued completions back into
``b4/completions.json`` (with a ``.bak3`` backup of the original).

Usage:
    uv run python -m LeakageEvaluationFramework.generation.scripts.rescue_b4_stock_leakage \
        --root      runtime/task_sensitivity_full/qwen35_35b \
        --dataset   dataset/evaluation/task_sensitivity_full.jsonl \
        --rag-cache runtime/task_sensitivity_full/shared_rag_cache/search_cache.json \
        --model     qwen/qwen3.5-35b-a3b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from LeakageEvaluationFramework.generation._openrouter_gen import OpenRouterGenerator
from LeakageEvaluationFramework.generation.methods import b4_timespec
from LeakageEvaluationFramework.generation.prompts import build_generation_prompt
from LeakageEvaluationFramework.generation.runner import build_config
from LeakageEvaluationFramework.generation.schemas import (
    Completion,
    ForecastingTaskInstance,
    instance_from_row,
)

logger = logging.getLogger("rescue_b4_stock_leakage")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory containing <method>/completions.json subdirs.")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="task_sensitivity_full.jsonl source dataset.")
    parser.add_argument("--rag-cache", type=Path, required=True,
                        help="Shared RAG search cache JSON file.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-prefix", default="stock_",
                        help="Only rescue instances whose ID starts with this prefix.")
    parser.add_argument("--cutoff-override", default=None,
                        help="If set, use this cutoff for leakage detection instead of "
                             "each instance's temporal_cutoff.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-regen-cycles", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    load_dotenv(override=False)

    instances_by_id = _load_instances(args.dataset)
    rag_cache = _load_rag_cache(args.rag_cache)

    b4_path = args.root / "b4" / "completions.json"
    if not b4_path.exists():
        raise FileNotFoundError(f"B4 completions missing: {b4_path}")
    b4_rows = json.loads(b4_path.read_text(encoding="utf-8"))

    affected = _identify_affected(
        b4_rows, instances_by_id, args.task_prefix, args.cutoff_override,
    )
    logger.info("Identified %d affected B4 %s* instances", len(affected), args.task_prefix)
    if args.dry_run:
        for iid in affected:
            print(iid)
        return
    if not affected:
        return

    cfg = build_config(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=1,
        max_regen_cycles=args.max_regen_cycles,
    )

    state_dir = args.root / "_rescue_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    rescued_path = state_dir / "rescued_b4_stock.jsonl"
    already = _load_rescued_state(rescued_path)
    if already:
        before = len(affected)
        affected = [iid for iid in affected if iid not in already]
        logger.info("Resume: %d previously rescued; %d remain", before - len(affected), len(affected))

    from QualityGatedExperiment.src.quality_gated_experiment.rag_evaluate import PerplexitySearchClient
    search_client = PerplexitySearchClient(cfg.perplexity_api_key)

    instances = [instances_by_id[iid] for iid in affected]
    context_map = {
        iid: rag_cache.get(f"{iid}:{instances_by_id[iid].temporal_cutoff}", "")
        for iid in affected
    }

    # ``b4_timespec.generate`` internally builds its own OpenRouterGenerator
    # with the default ``concurrency=1`` and would therefore execute Phase 1B
    # sequentially. Pre-generate Phase 1B in parallel here and feed the texts
    # in via ``phase1_texts`` so the only sequential bottleneck disappears.
    logger.info("Phase 1B: generating prompts for %d instances (concurrency=%d)",
                len(instances), args.concurrency)
    phase1_generator = OpenRouterGenerator(
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_key=cfg.openrouter_api_key,
        concurrency=args.concurrency,
    )
    phase1_prompts = [
        build_generation_prompt(inst, context_block=context_map.get(inst.instance_id, ""))
        for inst in instances
    ]
    phase1_texts = phase1_generator.generate_batch(phase1_prompts, batch_size=args.concurrency)
    logger.info("Phase 1B: done. Running supervised regen+aggregate for %d instances "
                "(concurrency=%d)", len(instances), args.concurrency)

    completions = b4_timespec.generate(
        instances, cfg,
        context_map=context_map,
        search_client=search_client,
        phase1_texts=phase1_texts,
        concurrency=args.concurrency,
    )

    rescued: dict[str, dict] = {}
    for comp in completions:
        comp.model = cfg.model_name
        rescued[comp.instance_id] = asdict(comp)
        _append_state(rescued_path, comp.instance_id, asdict(comp))
        logger.info("Rescued %s  evidence=%d  leaked_ids=%d",
                    comp.instance_id, len(comp.evidence), len(comp.leaked_claim_ids))

    rescued.update(already)
    _splice_results(b4_path, rescued)
    _report(b4_rows, rescued, instances_by_id, args.cutoff_override)


# ---------------------------------------------------------------------------
# Identification of affected rows
# ---------------------------------------------------------------------------

def _identify_affected(
    rows: list[dict],
    instances_by_id: dict[str, ForecastingTaskInstance],
    task_prefix: str,
    cutoff_override: str | None,
) -> list[str]:
    affected: list[str] = []
    for r in rows:
        iid = r["instance_id"]
        if not iid.startswith(task_prefix):
            continue
        if iid not in instances_by_id:
            continue
        cutoff = (cutoff_override or instances_by_id[iid].temporal_cutoff or "").strip()
        ev = r.get("evidence") or []
        flagged = set(r.get("leaked_claim_ids") or [])
        has_post = cutoff and any((e.get("source_date") or "") > cutoff for e in ev)
        has_flagged = any(e.get("id") in flagged for e in ev)
        if has_post or has_flagged:
            affected.append(iid)
    return affected


# ---------------------------------------------------------------------------
# State + I/O (mirrors retry_failed.py)
# ---------------------------------------------------------------------------

def _load_instances(path: Path) -> dict[str, ForecastingTaskInstance]:
    out: dict[str, ForecastingTaskInstance] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["id"]] = instance_from_row(row)
    return out


def _load_rag_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rescued_state(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["instance_id"]] = row["completion"]
    return out


def _append_state(path: Path, iid: str, completion: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"instance_id": iid, "completion": completion},
                            ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _splice_results(path: Path, rescued: dict[str, dict]) -> None:
    if not rescued:
        return
    rows = json.loads(path.read_text(encoding="utf-8"))
    backup = path.with_suffix(".json.bak3")
    if not backup.exists():
        shutil.copy(path, backup)
    spliced = 0
    for i, row in enumerate(rows):
        iid = row["instance_id"]
        if iid in rescued:
            rows[i] = rescued[iid]
            spliced += 1
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Spliced %d rescued rows into %s", spliced, path)


def _report(
    original_rows: list[dict],
    rescued: dict[str, dict],
    instances_by_id: dict[str, ForecastingTaskInstance],
    cutoff_override: str | None,
) -> None:
    if not rescued:
        return
    print()
    print(f"{'instance_id':<24}{'orig_post':<11}{'orig_flag':<11}{'new_post':<10}{'new_flag':<10}")
    print("-" * 70)
    by_id = {r["instance_id"]: r for r in original_rows}
    for iid, new_row in sorted(rescued.items()):
        orig = by_id.get(iid, {})
        cutoff = (cutoff_override or instances_by_id[iid].temporal_cutoff or "").strip()
        op = sum(1 for e in (orig.get("evidence") or []) if (e.get("source_date") or "") > cutoff)
        of = sum(1 for e in (orig.get("evidence") or [])
                 if e.get("id") in set(orig.get("leaked_claim_ids") or []))
        np = sum(1 for e in (new_row.get("evidence") or []) if (e.get("source_date") or "") > cutoff)
        nf = sum(1 for e in (new_row.get("evidence") or [])
                 if e.get("id") in set(new_row.get("leaked_claim_ids") or []))
        print(f"{iid:<24}{op:<11d}{of:<11d}{np:<10d}{nf:<10d}")


if __name__ == "__main__":
    main()
