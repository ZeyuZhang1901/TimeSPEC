"""Re-generate the residual parse_success=False completions with anti-loop sampling.

Steps:
  1. Scan each method's completions.json for parse_success=False rows.
  2. Group failed instance_ids by phase (B1/B2 share phase-1A; B3/B4 share phase-1B).
  3. Re-generate phase-1A for B1/B2 failures and phase-1B for B3/B4 failures with
     ``frequency_penalty`` + ``presence_penalty`` enabled to break repetition loops.
  4. Re-invoke each failing (method, instance_id) through its method's generate()
     function (B2/B4 re-run the supervisor + regen cycle from scratch).
  5. Splice the recovered completions in place; back up the original completions.json
     as completions.json.bak2 first (preserves the earlier .bak from step-1 reparse).

Crash-safe: rescued completions are appended to a JSONL log as they arrive, so an
abort partway through can be resumed without re-spending on already-done instances.

Usage:
    uv run python -m LeakageEvaluationFramework.generation.scripts.retry_failed \
        --root        runtime/task_sensitivity_full/qwen35_35b \
        --dataset     dataset/evaluation/task_sensitivity_full.jsonl \
        --rag-cache   runtime/task_sensitivity_full/shared_rag_cache/search_cache.json \
        --model       qwen/qwen3.5-35b-a3b \
        --freq-penalty 0.3 \
        --pres-penalty 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from LeakageEvaluationFramework.generation._openrouter_gen import OpenRouterGenerator
from LeakageEvaluationFramework.generation.methods import (
    b1_temporal_hint,
    b2_no_search_supervisor,
    b3_rag,
    b4_timespec,
)
from LeakageEvaluationFramework.generation.parsers import parse_completion_lenient
from LeakageEvaluationFramework.generation.prompts import build_generation_prompt
from LeakageEvaluationFramework.generation.runner import build_config
from LeakageEvaluationFramework.generation.schemas import (
    Completion,
    ForecastingTaskInstance,
    GenerationConfig,
    instance_from_row,
)


logger = logging.getLogger("retry_failed")

_METHODS = ("b1", "b2", "b3", "b4")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rag-cache", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--freq-penalty", type=float, default=0.3)
    parser.add_argument("--pres-penalty", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-regen-cycles", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke-test: only retry the first N failures.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Identify failures without re-generating.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    load_dotenv(override=False)

    instances_by_id = _load_instances(args.dataset)
    rag_cache = _load_rag_cache(args.rag_cache)
    failures = _collect_failures(args.root)
    if args.limit is not None:
        failures = failures[: args.limit]
    logger.info("Identified %d failed (method, instance_id) pairs", len(failures))
    if args.dry_run:
        for method, iid in failures:
            print(f"  {method:<4}  {iid}")
        return

    # Crash-safe resume: skip pairs already rescued in a previous incomplete run.
    state_dir = args.root / "_retry_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    rescued_path = state_dir / "rescued.jsonl"
    already_rescued = _load_rescued_state(rescued_path)
    if already_rescued:
        before = len(failures)
        failures = [(m, i) for (m, i) in failures if (m, i) not in already_rescued]
        logger.info("Resume: skipping %d already-rescued pairs; %d remain",
                    before - len(failures), len(failures))

    cfg = build_config(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=1,
        max_regen_cycles=args.max_regen_cycles,
    )
    cfg = replace(
        cfg,
        frequency_penalty=args.freq_penalty,
        presence_penalty=args.pres_penalty,
    )

    p1a_iids = sorted({i for m, i in failures if m in {"b1", "b2"}})
    p1b_iids = sorted({i for m, i in failures if m in {"b3", "b4"}})

    new_p1a = _regen_phase1(p1a_iids, instances_by_id, cfg, with_context=False)
    new_p1b = _regen_phase1(p1b_iids, instances_by_id, cfg, with_context=True, rag_cache=rag_cache)

    rescued: dict[tuple[str, str], dict] = {}
    for method, iid in failures:
        inst = instances_by_id[iid]
        try:
            if method == "b1":
                comp = _retry_b1(inst, new_p1a[iid], cfg)
            elif method == "b2":
                comp = _retry_b2(inst, new_p1a[iid], cfg)
            elif method == "b3":
                ctx = rag_cache.get(f"{iid}:{inst.temporal_cutoff}", "")
                comp = _retry_b3(inst, new_p1b[iid], ctx, cfg)
            elif method == "b4":
                ctx = rag_cache.get(f"{iid}:{inst.temporal_cutoff}", "")
                comp = _retry_b4(inst, new_p1b[iid], ctx, cfg)
            else:
                raise ValueError(method)
        except Exception:
            logger.exception("Retry %s/%s raised; marking as still-failed", method, iid)
            continue
        comp.model = cfg.model_name
        rescued[(method, iid)] = asdict(comp)
        _append_state(rescued_path, method, iid, asdict(comp))
        logger.info("Rescued %s/%s parse_success=%s", method, iid, comp.parse_success)

    rescued.update(already_rescued)
    _splice_results(args.root, rescued)
    _report(args.root)


# ---------------------------------------------------------------------------
# Phase-1 regeneration (parallel, single-shot per instance)
# ---------------------------------------------------------------------------

def _regen_phase1(
    iids: list[str],
    instances_by_id: dict[str, ForecastingTaskInstance],
    cfg: GenerationConfig,
    with_context: bool,
    rag_cache: dict[str, str] | None = None,
) -> dict[str, str]:
    if not iids:
        return {}
    label = "Phase 1B (RAG)" if with_context else "Phase 1A (no-RAG)"
    logger.info("Re-generating %s for %d failing instances (freq=%.2f, pres=%.2f)",
                label, len(iids), cfg.frequency_penalty, cfg.presence_penalty)
    generator = OpenRouterGenerator(
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_key=cfg.openrouter_api_key,
        concurrency=min(8, len(iids)),
        frequency_penalty=cfg.frequency_penalty,
        presence_penalty=cfg.presence_penalty,
    )
    prompts = []
    for iid in iids:
        inst = instances_by_id[iid]
        ctx = ""
        if with_context and rag_cache is not None:
            ctx = rag_cache.get(f"{iid}:{inst.temporal_cutoff}", "")
        prompts.append(build_generation_prompt(inst, context_block=ctx))
    texts = generator.generate_batch(prompts, batch_size=len(iids))
    return dict(zip(iids, texts))


# ---------------------------------------------------------------------------
# Method-specific retry (one instance each)
# ---------------------------------------------------------------------------

def _retry_b1(inst, phase1_text, cfg) -> Completion:
    completions = b1_temporal_hint.generate([inst], cfg, phase1_texts=[phase1_text])
    return completions[0]


def _retry_b2(inst, phase1_text, cfg) -> Completion:
    completions = b2_no_search_supervisor.generate(
        [inst], cfg, phase1_texts=[phase1_text], concurrency=1,
    )
    return completions[0]


def _retry_b3(inst, phase1_text, context_block, cfg) -> Completion:
    completions = b3_rag.generate(
        [inst], cfg, context_map={inst.instance_id: context_block},
        phase1_texts=[phase1_text],
    )
    return completions[0]


def _retry_b4(inst, phase1_text, context_block, cfg) -> Completion:
    from QualityGatedExperiment.src.quality_gated_experiment.rag_evaluate import PerplexitySearchClient
    search_client = PerplexitySearchClient(cfg.perplexity_api_key)
    completions = b4_timespec.generate(
        [inst], cfg, context_map={inst.instance_id: context_block},
        search_client=search_client, phase1_texts=[phase1_text], concurrency=1,
    )
    return completions[0]


# ---------------------------------------------------------------------------
# State + I/O
# ---------------------------------------------------------------------------

def _collect_failures(root: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for method in _METHODS:
        path = root / method / "completions.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            if not row.get("parse_success"):
                out.append((method, row["instance_id"]))
    return out


def _load_instances(path: Path) -> dict[str, ForecastingTaskInstance]:
    by_id: dict[str, ForecastingTaskInstance] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_id[r["id"]] = instance_from_row(r)
    return by_id


def _load_rag_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rescued_state(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[(row["method"], row["instance_id"])] = row["completion"]
    return out


def _append_state(path: Path, method: str, iid: str, completion: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"method": method, "instance_id": iid, "completion": completion},
                            ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _splice_results(root: Path, rescued: dict[tuple[str, str], dict]) -> None:
    by_method: dict[str, dict[str, dict]] = {m: {} for m in _METHODS}
    for (method, iid), comp in rescued.items():
        by_method[method][iid] = comp
    for method in _METHODS:
        if not by_method[method]:
            continue
        path = root / method / "completions.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        backup = path.with_suffix(".json.bak2")
        if not backup.exists():
            shutil.copy(path, backup)
        for i, row in enumerate(rows):
            iid = row["instance_id"]
            if iid in by_method[method]:
                rows[i] = by_method[method][iid]
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Spliced %d rescued rows into %s", len(by_method[method]), path)


def _report(root: Path) -> None:
    print()
    print(f"{'method':<8}{'total':<8}{'parsed':<10}{'failed':<10}{'rate':<8}")
    print("-" * 50)
    for method in _METHODS:
        path = root / method / "completions.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        n = len(rows)
        n_pass = sum(1 for r in rows if r.get("parse_success"))
        print(f"{method:<8}{n:<8}{n_pass:<10}{n-n_pass:<10}{n_pass/n*100:5.1f}%")


if __name__ == "__main__":
    main()
