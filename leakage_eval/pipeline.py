"""Unified leakage evaluation for every baseline (b1–b4, …).

**Inputs.** Each method contributes the same structured instance payload (parsed into
:class:`EvalInput`): evidence list (claims), prediction, and rationale text among raw
fields. Generation-side metadata such as ``leaked_claim_ids`` is *not* used here—only
this evaluator decides leakage.

**Per-instance steps:**

1. **Claim leakage taxonomy.** Run the date / C0–C8 verifier (:class:`LlmDateVerifier`
   or ``local``) on every evidence claim → :class:`ClaimDecision` rows and binary
   leakage labels :func:`~LeakageEvaluationFramework.leakage_eval.metrics.leakage.decision_labels`.

2. **Conditional Shapley.** If ``skip_if_no_leakage`` and no claim leaked, skip coalition
   evaluation (Shapley-DCLR is exactly zero). Otherwise treat **all** evidence claims as
   Shapley players and run :class:`~LeakageEvaluationFramework.leakage_eval.shapley.runner.ShapleyRunner`
   (exact when ``n`` ≤ ``exact_threshold_n``, else leverage).

3. **Metrics.** ``olr`` = :func:`~LeakageEvaluationFramework.leakage_eval.metrics.dclr.leakage_rate`
   (mean leakage fraction); ``shapley_dclr`` =
   :func:`~LeakageEvaluationFramework.leakage_eval.metrics.dclr.shapley_dclr`; ``performance``
   = :func:`~LeakageEvaluationFramework.leakage_eval.metrics.performance.performance_score`
   against dataset ground truth.

Artifacts are written by :func:`run_evaluation_job`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Literal

from LeakageEvaluationFramework.leakage_eval.audit.logger import AuditLogger
from LeakageEvaluationFramework.leakage_eval.config import EvalJobConfig
from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import (
    OpenRouterCoalitionPredictor,
    OpenRouterConfig,
    OpenRouterTextClient,
)
from LeakageEvaluationFramework.leakage_eval.shapley.value_function import CoalitionPredictor
from LeakageEvaluationFramework.leakage_eval.io.loaders import (
    build_eval_inputs,
    load_completion_rows,
    load_dataset_metadata,
)
from LeakageEvaluationFramework.leakage_eval.io.writers import ResultWriter
from LeakageEvaluationFramework.leakage_eval.metrics.aggregation import aggregate_instance_rows
from LeakageEvaluationFramework.leakage_eval.metrics.dclr import leakage_rate, shapley_dclr
from LeakageEvaluationFramework.leakage_eval.metrics.leakage import decision_labels
from LeakageEvaluationFramework.leakage_eval.metrics.performance import performance_score
from LeakageEvaluationFramework.leakage_eval.schemas import (
    ClaimDecision,
    ClaimRecord,
    EvalInput,
    EvalOutput,
    ShapleyResult,
)
from LeakageEvaluationFramework.leakage_eval.shapley.runner import ShapleyRunner
from LeakageEvaluationFramework.leakage_eval.taxonomy.date_verifier import LlmDateVerifier, local_claim_decision

TaxonomyBackend = Literal["llm", "local"]


async def run_evaluation_job(
    config: EvalJobConfig,
    *,
    model: str,
    method: str,
    max_instances: int | None = None,
    instance_ids: tuple[str, ...] | None = None,
    taxonomy_backend: TaxonomyBackend = "llm",
    concurrency: int = 1,
    checkpoint_every: int = 200,
) -> dict:
    """Run the unified evaluator on completions → leakage labels → optional Shapley → metrics.

    See module docstring for the baseline-independent evaluation contract.
    """
    log = logging.getLogger(__name__)

    rows = load_completion_rows(config.input_path)
    if instance_ids is not None:
        by_id = {str(r["instance_id"]): r for r in rows}
        missing = [i for i in instance_ids if i not in by_id]
        if missing:
            raise KeyError(f"instance_ids not found in input: {missing}")
        rows = [by_id[i] for i in instance_ids]
    if max_instances is not None:
        rows = rows[:max_instances]
    metadata = load_dataset_metadata(config.dataset_path)
    instances = build_eval_inputs(rows, metadata, model=model, method=method)

    writer = ResultWriter(config.output_dir)
    audit_logger = AuditLogger(writer.audit_dir)
    writer.write_config(config)

    # Resume: skip instances already in instance_results.jsonl or errors.jsonl
    done_ids = _load_completed_ids(config.output_dir)
    if done_ids:
        before = len(instances)
        instances = [inst for inst in instances if inst.instance_id not in done_ids]
        log.info("Resuming: %d already done, %d remaining (was %d)", len(done_ids), len(instances), before)

    writer.write_manifest(
        {
            "model": model,
            "method": method,
            "input_path": str(config.input_path),
            "dataset_path": str(config.dataset_path),
            "taxonomy_backend": taxonomy_backend,
            "shapley_backend": config.shapley.evaluator_backend,
            "shapley_estimator_config": {
                "coalition_budget_c": config.shapley.coalition_budget_c,
                "coalition_budget_cap": config.shapley.coalition_budget_cap,
                "exact_threshold_n": config.shapley.exact_threshold_n,
                "random_seed": config.shapley.random_seed,
                "skip_if_no_leakage": config.shapley.skip_if_no_leakage,
            },
            "n_instances": len(instances),
            "n_resumed_from": len(done_ids),
            "instance_ids_filter": list(instance_ids) if instance_ids else None,
            "concurrency": concurrency,
            "checkpoint_every": checkpoint_every,
        }
    )

    verifier = _build_verifier(config, taxonomy_backend, audit_logger)
    instance_rows: list[dict] = []

    # Process in chunks for incremental checkpointing
    for chunk_start in range(0, len(instances), checkpoint_every):
        chunk = instances[chunk_start : chunk_start + checkpoint_every]
        log.info(
            "Processing chunk %d–%d / %d",
            chunk_start + 1,
            min(chunk_start + len(chunk), len(instances)),
            len(instances),
        )

        results = await _run_instances(
            chunk,
            config,
            verifier,
            taxonomy_backend,
            audit_logger,
            concurrency=max(1, concurrency),
        )

        for instance, output, error in results:
            if error is not None:
                writer.append_error({"instance_id": instance.instance_id, "error": error})
                continue
            assert output is not None

            instance_row = _instance_row(output)
            instance_rows.append(instance_row)
            writer.append_instance(instance_row)
            writer.append_shapley(asdict(output.shapley_result))
            if instance_row["faithfulness_status"] == "low_faithfulness":
                writer.append_low_faithfulness(instance_row)
            for decision in output.claim_decisions:
                writer.append_claim(asdict(decision))

        log.info(
            "Checkpoint: %d instances done so far (%d total with resumed)",
            len(instance_rows),
            len(instance_rows) + len(done_ids),
        )

    # Re-read all instance rows (resumed + new) for final summary
    all_rows = _load_all_instance_rows(config.output_dir)
    summary = aggregate_instance_rows(all_rows)
    writer.write_summary(summary)
    return summary


def _load_completed_ids(output_dir: Path) -> set[str]:
    """Read instance IDs that already have successful results (not errors)."""
    ids: set[str] = set()
    path = output_dir / "instance_results.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["instance_id"])
    return ids


def _load_all_instance_rows(output_dir: Path) -> list[dict]:
    """Read all instance result rows for final aggregation."""
    path = output_dir / "instance_results.jsonl"
    rows: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


async def _run_instances(
    instances: list[EvalInput],
    config: EvalJobConfig,
    verifier: LlmDateVerifier | None,
    taxonomy_backend: TaxonomyBackend,
    audit_logger: AuditLogger,
    *,
    concurrency: int,
) -> list[tuple[EvalInput, EvalOutput | None, str | None]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one(instance: EvalInput) -> tuple[EvalInput, EvalOutput | None, str | None]:
        async with semaphore:
            try:
                # Use per-instance predictors to avoid raw-response races during concurrent pilot runs.
                output = await _evaluate_instance(instance, config, verifier, {}, taxonomy_backend, audit_logger)
                return instance, output, None
            except Exception as exc:
                return instance, None, str(exc)

    return await asyncio.gather(*[_run_one(instance) for instance in instances])


async def _evaluate_instance(
    instance: EvalInput,
    config: EvalJobConfig,
    verifier: LlmDateVerifier | None,
    predictors: dict[str, CoalitionPredictor],
    taxonomy_backend: TaxonomyBackend,
    audit_logger: AuditLogger,
) -> EvalOutput:
    """One structured prediction → per-claim leakage → Shapley (all claims) if warranted."""
    decisions = await _claim_decisions(instance, verifier, taxonomy_backend)
    labels = decision_labels(decisions)
    leaked_claims = sum(labels)

    if config.shapley.skip_if_no_leakage and leaked_claims == 0:
        shapley_result = _skipped_shapley(instance, "No leaked claims; Shapley-DCLR is exactly zero.")
        shapley_value = 0.0
    else:
        predictor = _predictor_for_instance(instance, config, predictors)
        shapley_result = await ShapleyRunner(config.shapley, predictor, audit_logger=audit_logger).run(instance)
        values = [item.shapley_value for item in sorted(shapley_result.claim_shapley, key=lambda x: x.claim_id)]
        shapley_value = shapley_dclr(values, labels)["shapley_dclr"]

    return EvalOutput(
        instance_id=instance.instance_id,
        model=instance.model,
        method=instance.method,
        task=instance.task,
        n_claims=len(instance.evidence),
        leaked_claims=leaked_claims,
        olr=leakage_rate(labels),
        shapley_dclr=shapley_value,
        performance=performance_score(instance.task, instance.prediction, instance.ground_truth),
        claim_decisions=decisions,
        shapley_result=shapley_result,
    )


async def _claim_decisions(
    instance: EvalInput,
    verifier: LlmDateVerifier | None,
    taxonomy_backend: TaxonomyBackend,
) -> list[ClaimDecision]:
    records = [
        ClaimRecord(
            instance_id=instance.instance_id,
            claim_id=item.id,
            claim_text=item.fact,
            declared_source_date=item.source_date,
            cutoff_date=instance.cutoff_date,
            task=instance.task,
            target_description=instance.target_description,
            model=instance.model,
            method=instance.method,
        )
        for item in instance.evidence
    ]
    if taxonomy_backend == "local":
        return [local_claim_decision(record) for record in records]
    if verifier is None:
        raise ValueError("LLM taxonomy backend requested without verifier")
    return await verifier.verify(records)


def _build_verifier(config: EvalJobConfig, backend: TaxonomyBackend, audit_logger: AuditLogger) -> LlmDateVerifier | None:
    if backend == "local":
        return None
    client = OpenRouterTextClient(
        OpenRouterConfig(
            model=config.leakage.judge_model,
            temperature=0.0,
            max_tokens=8192,
        )
    )
    return LlmDateVerifier(client, batch_size=config.leakage.taxonomy_batch_size, audit_logger=audit_logger)


def _predictor_for_instance(
    instance: EvalInput,
    config: EvalJobConfig,
    predictors: dict[str, CoalitionPredictor],
) -> CoalitionPredictor:
    backend = config.shapley.evaluator_backend
    if backend != "openrouter":
        raise ValueError(f"Unsupported Shapley evaluator backend: {backend}")
    model = config.shapley.evaluator_model_override or instance.model
    key = model
    if key not in predictors:
        predictors[key] = OpenRouterCoalitionPredictor(
            model=model,
            temperature=config.shapley.temperature,
            max_tokens=_coalition_max_tokens(instance),
        )
    return predictors[key]


def _coalition_max_tokens(instance: EvalInput) -> int:
    """Token budget for coalition evaluation.

    Thinking is disabled via reasoning.effort="none" in the API call,
    so all models output only the bare JSON (e.g. {"probability_yes": 0.85}).
    128 tokens is sufficient for any task type.
    """
    return 128


def _skipped_shapley(instance: EvalInput, reason: str) -> ShapleyResult:
    return ShapleyResult(
        instance_id=instance.instance_id,
        shapley_status="skipped_no_leakage",
        estimator=None,
        n_claims=len(instance.evidence),
        m_total=0,
        v_empty=0.0,
        v_full=None,
        original_prediction=instance.prediction,
        closed_world_full_prediction=None,
        full_reconstruction_gap=None,
        num_model_calls=0,
        claim_shapley=[],
        reason=reason,
    )


def _instance_row(output: EvalOutput) -> dict:
    gap = output.shapley_result.full_reconstruction_gap
    return {
        "instance_id": output.instance_id,
        "model": output.model,
        "method": output.method,
        "task": output.task,
        "n_claims": output.n_claims,
        "leaked_claims": output.leaked_claims,
        "olr": output.olr,
        "shapley_dclr": output.shapley_dclr,
        "performance": output.performance,
        "shapley_status": output.shapley_result.shapley_status,
        "full_reconstruction_gap": gap,
        "faithfulness_status": faithfulness_status(gap),
        "closed_world_full_prediction": output.shapley_result.closed_world_full_prediction,
        "original_prediction": output.shapley_result.original_prediction,
        "shapley_estimator": output.shapley_result.estimator,
        "m_total": output.shapley_result.m_total,
        "num_model_calls": output.shapley_result.num_model_calls,
    }


def faithfulness_status(gap: float | None) -> str:
    """Classify reconstruction faithfulness for downstream analysis."""

    if gap is None:
        return "not_applicable"
    if gap <= 0.05:
        return "high_faithfulness"
    if gap <= 0.15:
        return "medium_faithfulness"
    return "low_faithfulness"
