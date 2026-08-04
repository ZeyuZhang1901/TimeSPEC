"""Orchestrator for the B1–B4 generation pipeline.

Responsibilities:
- Load benchmark rows and adapt to ForecastingTaskInstances.
- Run shared Phase 1A (B1/B2 source) and Phase 1B (B3/B4 source).
- Run Perplexity retrieval in parallel (ThreadPoolExecutor) and build per-instance context blocks.
- Dispatch to each requested method (with concurrency for supervisor-heavy B2/B4).
- Persist completions and caches to the output directory.
- Quality gates after each phase: abort early if output quality is unacceptable.
- Incremental saves for B-methods: partial results survive interruption.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from LeakageEvaluationFramework.generation._openrouter_gen import OpenRouterGenerator
from LeakageEvaluationFramework.generation.prompts import build_generation_prompt, build_query_generation_prompt
from LeakageEvaluationFramework.generation.schemas import (
    Completion,
    ForecastingTaskInstance,
    GenerationConfig,
    instance_from_row,
)


class _PerInstanceWriter:
    """Thread-safe append-only JSONL writer.

    Each call writes a single completion as one JSON line and fsyncs, so an
    abrupt crash (OOM, SIGKILL) loses at most the in-flight instance — never
    a batch of already-completed ones.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, completion: Completion) -> None:
        line = json.dumps(asdict(completion), ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

logger = logging.getLogger(__name__)

_DEFAULT_JUDGE_MODEL = "google/gemini-2.0-flash-lite-001"
_DEFAULT_JUDGE_MAX_TOKENS = 4096
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_BATCH_SIZE = 8
_DEFAULT_MAX_REGEN_CYCLES = 1
_DEFAULT_RAG_CONTEXT_BUDGET = 3000
_DEFAULT_RAG_MAX_RESULTS = 5
_DEFAULT_CONCURRENCY = 4
_QUALITY_GATE_THRESHOLD = 0.95  # abort if good-output rate drops below this


class QualityGateError(RuntimeError):
    """Raised when a phase fails its quality check."""


# ======================================================================
# Public entry point
# ======================================================================

def run_all_methods(
    dataset_path: Path,
    output_dir: Path,
    model_name: str,
    cfg: GenerationConfig,
    methods: list[str],
    split: str = "all",
    max_instances: int | None = None,
    concurrency: int = _DEFAULT_CONCURRENCY,
    shared_rag_cache_dir: Path | None = None,
) -> None:
    """Run the requested generation methods for one (model, dataset) combination.

    Args:
        dataset_path: Path to the benchmark JSONL file.
        output_dir: Root output directory for this model run.
        model_name: OpenRouter model ID (e.g. "anthropic/claude-sonnet-4").
        cfg: Resolved GenerationConfig.
        methods: Subset of ["b1", "b2", "b3", "b4"] to run.
        split: "treatment", "control", or "all".
        max_instances: Cap on number of instances (None = all).
        concurrency: Number of parallel worker threads for RAG retrieval and
            per-instance supervisor calls in B2/B4.
        shared_rag_cache_dir: If set, RAG search results are persisted here and
            shared across all model runs for the same dataset.  RAG queries depend
            only on the question content, not on the generator model, so sharing
            is correct and eliminates redundant Perplexity calls.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = output_dir / "generation"
    gen_dir.mkdir(exist_ok=True)

    instances = _load_instances(dataset_path, split, max_instances)
    logger.info("Loaded %d instances (split=%s)", len(instances), split)

    _write_json(output_dir / "run_config.json", {
        "model_name": model_name,
        "methods": methods,
        "split": split,
        "n_instances": len(instances),
        "concurrency": concurrency,
    })

    # RAG search cache: prefer shared across models to avoid duplicate Perplexity calls.
    # Shared cache is written atomically after each retrieval so concurrent model runs
    # (launched in separate processes) also benefit from partial overlap.
    if shared_rag_cache_dir is not None:
        shared_rag_cache_dir.mkdir(parents=True, exist_ok=True)
        rag_cache_path = shared_rag_cache_dir / "search_cache.json"
    else:
        rag_cache_path = gen_dir / "search_cache.json"

    search_cache = _load_json_cache(rag_cache_path)
    claim_cache = _load_json_cache(gen_dir / "claim_cache.json")

    needs_rag = bool(set(methods) & {"b3", "b4"})

    generator = OpenRouterGenerator(
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_key=cfg.openrouter_api_key,
        concurrency=concurrency,
    )

    # Phase 1A: generate without RAG (B1/B2 source)
    phase1a_texts: list[str] | None = None
    if bool(set(methods) & {"b1", "b2"}):
        phase1a_path = gen_dir / "phase1a_raw.jsonl"
        if phase1a_path.exists():
            phase1a_texts = _load_phase1_texts(phase1a_path, instances)
            logger.info("Loaded %d Phase 1A texts from cache", len(phase1a_texts))
        else:
            logger.info("Running Phase 1A (no-RAG) for %d instances …", len(instances))
            prompt_inputs = [build_generation_prompt(inst) for inst in instances]
            phase1a_texts = generator.generate_batch(prompt_inputs, batch_size=cfg.batch_size, stage="phase1a_generation")
            _write_phase1_jsonl(phase1a_path, instances, phase1a_texts)
        _validate_phase1(phase1a_texts, "Phase 1A", instances)

    # RAG retrieval (B3 + B4 source) — parallelised
    context_map: dict[str, str] = {}
    if needs_rag:
        context_map = _build_context_map(instances, cfg, search_cache, concurrency=concurrency)
        _write_json_cache(rag_cache_path, search_cache)

    # Phase 1B: generate with RAG (B3 + B4 source)
    phase1b_texts: list[str] | None = None
    if needs_rag:
        phase1b_path = gen_dir / "phase1b_raw.jsonl"
        if phase1b_path.exists():
            phase1b_texts = _load_phase1_texts(phase1b_path, instances)
            logger.info("Loaded %d Phase 1B texts from cache", len(phase1b_texts))
        else:
            logger.info("Running Phase 1B (RAG) for %d instances …", len(instances))
            prompt_inputs = [
                build_generation_prompt(inst, context_block=context_map.get(inst.instance_id, ""))
                for inst in instances
            ]
            phase1b_texts = generator.generate_batch(prompt_inputs, batch_size=cfg.batch_size, stage="phase1b_generation")
            _write_phase1_jsonl(phase1b_path, instances, phase1b_texts)
        _validate_phase1(phase1b_texts, "Phase 1B", instances)

    # Dispatch methods — with incremental saves for crash resilience
    for method in methods:
        method_dir = output_dir / method
        completions_path = method_dir / "completions.json"
        if completions_path.exists():
            logger.info("Skipping %s — completions already exist at %s", method, completions_path)
            continue

        method_dir.mkdir(exist_ok=True)
        partial_path = method_dir / "completions_partial.jsonl"

        # Load any partial results from a previous interrupted run
        done_ids: set[str] = set()
        if partial_path.exists():
            with partial_path.open(encoding="utf-8") as pf:
                for pline in pf:
                    pline = pline.strip()
                    if pline:
                        done_ids.add(json.loads(pline)["instance_id"])
            logger.info("Resuming %s — %d/%d already completed", method, len(done_ids), len(instances))

        remaining_idx = [i for i, inst in enumerate(instances) if inst.instance_id not in done_ids]

        if remaining_idx:
            remaining_instances = [instances[i] for i in remaining_idx]
            remaining_p1a = [phase1a_texts[i] for i in remaining_idx] if phase1a_texts else None
            remaining_p1b = [phase1b_texts[i] for i in remaining_idx] if phase1b_texts else None

            logger.info("Running method %s for %d instances …", method, len(remaining_instances))

            # Per-instance writer: each completion is fsync'd to disk as soon as it
            # is produced. If the process crashes mid-method, only the in-flight
            # instances are lost; everything else persists for resume.
            writer = _PerInstanceWriter(partial_path)

            def _stamp_and_persist(c: Completion) -> None:
                c.model = cfg.model_name
                writer.write(c)

            completions = _run_method(
                method, remaining_instances, cfg, remaining_p1a, remaining_p1b, context_map,
                concurrency=concurrency, on_complete=_stamp_and_persist,
            )

        # Consolidate partial → final completions.json
        all_completions: list[dict[str, Any]] = []
        with partial_path.open(encoding="utf-8") as pf:
            for pline in pf:
                pline = pline.strip()
                if pline:
                    all_completions.append(json.loads(pline))

        with completions_path.open("w", encoding="utf-8") as f:
            json.dump(all_completions, f, ensure_ascii=False, indent=2)
        partial_path.unlink(missing_ok=True)

        n_success = sum(1 for c in all_completions if c.get("parse_success"))
        _write_json(method_dir / "run_config.json", {
            "method": method,
            "model": cfg.model_name,
            "n_completions": len(all_completions),
            "parse_success_rate": n_success / max(len(all_completions), 1),
        })
        logger.info("Wrote %d completions for method %s (%.1f%% parse success)",
                     len(all_completions), method, n_success / max(len(all_completions), 1) * 100)

    _write_json_cache(gen_dir / "claim_cache.json", claim_cache)


# ======================================================================
# Quality gate
# ======================================================================

def _phase1_output_complete(text: str) -> bool:
    """Heuristic: response looks structurally complete (not mid-generation cutoff).

    Some models wrap JSON in markdown fences (``` ... ```); bare JSON ends with ``}``.
    """
    s = text.strip()
    if not s:
        return False
    if s.endswith("}"):
        return True
    # Markdown-closed JSON (e.g. Kimi-K2.5 / Claude-style ```json blocks)
    if s.endswith("```"):
        return True
    return False


def _validate_phase1(
    texts: list[str],
    phase_name: str,
    instances: list[ForecastingTaskInstance],
) -> None:
    """Check Phase 1A/1B output quality and abort if below threshold."""
    total = len(texts)
    empty = sum(1 for t in texts if not t.strip())
    truncated = sum(1 for t in texts if t.strip() and not _phase1_output_complete(t))
    good = total - empty - truncated
    good_rate = good / max(total, 1)

    logger.info(
        "%s quality: %d/%d good (%.1f%%), %d empty, %d truncated",
        phase_name, good, total, good_rate * 100, empty, truncated,
    )

    if good_rate < _QUALITY_GATE_THRESHOLD:
        msg = (
            f"{phase_name} QUALITY GATE FAILED: {good_rate:.1%} good "
            f"({empty} empty, {truncated} truncated out of {total}). "
            f"Threshold is {_QUALITY_GATE_THRESHOLD:.0%}. "
            f"Aborting to avoid wasting downstream API calls."
        )
        logger.error(msg)
        raise QualityGateError(msg)


# ======================================================================
# Method dispatch
# ======================================================================

def _run_method(
    method: str,
    instances: list[ForecastingTaskInstance],
    cfg: GenerationConfig,
    phase1a_texts: list[str] | None,
    phase1b_texts: list[str] | None,
    context_map: dict[str, str],
    concurrency: int = 1,
    on_complete: "Callable[[Completion], None] | None" = None,
) -> list[Completion]:
    if method == "b1":
        from LeakageEvaluationFramework.generation.methods import b1_temporal_hint
        return b1_temporal_hint.generate(
            instances, cfg, phase1_texts=phase1a_texts, on_complete=on_complete,
        )

    if method == "b2":
        from LeakageEvaluationFramework.generation.methods import b2_no_search_supervisor
        return b2_no_search_supervisor.generate(
            instances, cfg, phase1_texts=phase1a_texts, concurrency=concurrency,
            on_complete=on_complete,
        )

    if method == "b3":
        from LeakageEvaluationFramework.generation.methods import b3_rag
        return b3_rag.generate(
            instances, cfg, context_map=context_map, phase1_texts=phase1b_texts,
            on_complete=on_complete,
        )

    if method == "b4":
        from LeakageEvaluationFramework.generation.methods import b4_timespec
        from LeakageEvaluationFramework.generation.perplexity import PerplexitySearchClient
        search_client = PerplexitySearchClient(cfg.perplexity_api_key)
        return b4_timespec.generate(
            instances, cfg, context_map=context_map,
            search_client=search_client, phase1_texts=phase1b_texts,
            concurrency=concurrency, on_complete=on_complete,
        )

    raise ValueError(f"Unknown method: {method!r}")


# ======================================================================
# RAG retrieval (parallel)
# ======================================================================

def _build_context_map(
    instances: list[ForecastingTaskInstance],
    cfg: GenerationConfig,
    search_cache: dict[str, Any],
    concurrency: int = 4,
) -> dict[str, str]:
    """Build per-instance RAG context blocks via Perplexity, using cache.

    Retrieval calls are parallelised with ThreadPoolExecutor to reduce wall-clock
    time from O(N * latency) to O(ceil(N/concurrency) * latency).
    The search_cache is updated under a lock for thread safety.
    """

    from LeakageEvaluationFramework.generation.perplexity import (
        PerplexitySearchClient,
        retrieve_and_build_context,
    )
    from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import (
        OpenRouterConfig,
        OpenRouterTextClient,
    )

    search_client = PerplexitySearchClient(cfg.perplexity_api_key)
    or_client = OpenRouterTextClient(
        OpenRouterConfig(model=cfg.judge_model, max_tokens=cfg.judge_max_tokens),
        api_key=cfg.openrouter_api_key,
    )

    cache_lock = threading.Lock()
    context_map: dict[str, str] = {}

    # Pre-fill from cache
    uncached: list[ForecastingTaskInstance] = []
    for inst in instances:
        cache_key = f"{inst.instance_id}:{inst.temporal_cutoff}"
        if cache_key in search_cache:
            context_map[inst.instance_id] = search_cache[cache_key]
        else:
            uncached.append(inst)

    logger.info(
        "RAG retrieval: %d cached, %d to fetch (concurrency=%d)",
        len(instances) - len(uncached), len(uncached), concurrency,
    )

    if not uncached:
        return context_map

    def _fetch_one(inst: ForecastingTaskInstance) -> tuple[str, str]:
        """Returns (instance_id, context_block)."""
        try:
            queries = _generate_queries(inst, or_client)
            context_block, _, _, _ = retrieve_and_build_context(
                search_client,
                queries,
                cutoff_date=inst.temporal_cutoff,
                budget=cfg.rag_context_budget,
                max_results_per_query=cfg.rag_max_results,
            )
        except Exception as exc:
            logger.warning("RAG retrieval failed for %s: %s", inst.instance_id, exc)
            context_block = ""
        return inst.instance_id, context_block

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_fetch_one, inst): inst for inst in uncached}
        done_count = 0
        for future in as_completed(futures):
            inst = futures[future]
            try:
                iid, context_block = future.result()
            except Exception as exc:
                logger.error("Unexpected RAG error for %s: %s", inst.instance_id, exc)
                iid, context_block = inst.instance_id, ""

            cache_key = f"{iid}:{inst.temporal_cutoff}"
            context_map[iid] = context_block
            with cache_lock:
                search_cache[cache_key] = context_block

            done_count += 1
            if done_count % 100 == 0 or done_count == len(uncached):
                logger.info("RAG retrieval progress: %d / %d fetched", done_count, len(uncached))

    return context_map


def _generate_queries(
    inst: ForecastingTaskInstance,
    or_client: "OpenRouterTextClient",
) -> list[str]:
    import asyncio

    prompt = build_query_generation_prompt(inst)
    text = asyncio.run(or_client.call_text(prompt, stage="query_generation"))
    return _parse_query_json(text)


def _parse_query_json(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        arr = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    queries: list[str] = []
    for item in arr:
        if isinstance(item, dict):
            q = item.get("query", "")
        elif isinstance(item, str):
            q = item
        else:
            continue
        if q.strip():
            queries.append(q.strip())
    return queries


# ======================================================================
# I/O helpers
# ======================================================================

def _load_instances(
    dataset_path: Path,
    split: str,
    max_instances: int | None,
) -> list[ForecastingTaskInstance]:
    rows: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if split != "all":
        rows = [r for r in rows if r.get("group") == split]

    if max_instances is not None:
        rows = rows[:max_instances]

    return [instance_from_row(r) for r in rows]


def _write_completions(path: Path, completions: list[Completion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(c) for c in completions]
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _write_phase1_jsonl(
    path: Path,
    instances: list[ForecastingTaskInstance],
    texts: list[str],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for inst, text in zip(instances, texts):
            f.write(json.dumps({"instance_id": inst.instance_id, "text": text}, ensure_ascii=False) + "\n")


def _load_phase1_texts(
    path: Path,
    instances: list[ForecastingTaskInstance],
) -> list[str]:
    by_id: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_id[row["instance_id"]] = row["text"]
    return [by_id.get(inst.instance_id, "") for inst in instances]


def _load_json_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def _write_json_cache(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ======================================================================
# Config builder
# ======================================================================

def build_config(
    model_name: str,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
    judge_max_tokens: int = _DEFAULT_JUDGE_MAX_TOKENS,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    max_regen_cycles: int = _DEFAULT_MAX_REGEN_CYCLES,
    rag_context_budget: int = _DEFAULT_RAG_CONTEXT_BUDGET,
    rag_max_results: int = _DEFAULT_RAG_MAX_RESULTS,
) -> GenerationConfig:
    """Build a GenerationConfig from arguments, loading API keys from environment."""

    load_dotenv(override=False)
    pplx_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not or_key:
        raise ValueError("Missing OPENROUTER_API_KEY in environment")

    return GenerationConfig(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        judge_model=judge_model,
        judge_max_tokens=judge_max_tokens,
        batch_size=batch_size,
        max_regen_cycles=max_regen_cycles,
        perplexity_api_key=pplx_key,
        openrouter_api_key=or_key,
        rag_context_budget=rag_context_budget,
        rag_max_results=rag_max_results,
    )
