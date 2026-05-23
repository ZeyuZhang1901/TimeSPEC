"""Batch leakage evaluation pipeline (Anthropic Message Batches API).

Mirrors `leakage_eval/pipeline.py` semantics but uses two async batches per job:

Phase 1 — taxonomy:
  Build one BatchRequest per chunk of N claims (default 100), submit, poll, fetch,
  parse C0-C8 labels, write `claim_results.jsonl`. Produces per-instance leakage
  labels for Phase 2.

Phase 2 — Shapley:
  For each instance with ≥1 leaked claim (when skip_if_no_leakage=True), compute
  all coalition masks upfront, bundle as one BatchRequest per coalition, submit
  in chunks of MAX_BATCH_REQUESTS, poll, fetch, solve regression locally,
  write `instance_results.jsonl` and `shapley_results.jsonl`.

Output schema is identical to `leakage_eval/pipeline.py` so downstream summary
aggregation works unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from LeakageEvaluationFramework.batch_eval.anthropic_batch import (
    AnthropicBatchClient,
    BatchRequest,
    BatchResult,
    MAX_BATCH_REQUESTS,
)
from LeakageEvaluationFramework.batch_eval.state import BatchHandle, JobState
from LeakageEvaluationFramework.leakage_eval.config import EvalJobConfig
from LeakageEvaluationFramework.leakage_eval.evaluators.parsers import parse_prediction
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
from LeakageEvaluationFramework.leakage_eval.pipeline import (
    _coalition_max_tokens,
    _instance_row,
    _skipped_shapley,
    faithfulness_status,
)
from LeakageEvaluationFramework.leakage_eval.schemas import (
    ClaimDecision,
    ClaimRecord,
    EvalInput,
    EvalOutput,
    EvidenceItem,
    ShapleyResult,
    ClaimShapleyValue,
)
from LeakageEvaluationFramework.leakage_eval.shapley.coalition_sampler import (
    compute_budget,
    exact_coalitions,
    leverage_shap_coalitions,
)
from LeakageEvaluationFramework.leakage_eval.shapley.exact import exact_shapley
from LeakageEvaluationFramework.leakage_eval.shapley.leverage_regression import (
    solve_projected_regression,
)
from LeakageEvaluationFramework.leakage_eval.shapley.paper_leverage_shap import (
    paper_leverage_masks,
    solve_paper_leverage_regression,
)
from LeakageEvaluationFramework.leakage_eval.shapley.prompts import build_coalition_messages
from LeakageEvaluationFramework.leakage_eval.shapley.value_function import (
    compact_coalition_prediction,
    reconstruction_value,
)
from LeakageEvaluationFramework.leakage_eval.taxonomy.date_verifier import (
    _decision_from_llm,
    _parse_llm_decisions,
    local_claim_decision,
)
from LeakageEvaluationFramework.leakage_eval.taxonomy.prompts import (
    TAXONOMY_DATE_SYSTEM_PROMPT,
    build_taxonomy_date_prompt,
)

log = logging.getLogger(__name__)


# ============================================================
# Public entry point
# ============================================================


def run_batch_job(
    config: EvalJobConfig,
    *,
    model: str,
    method: str,
    max_instances: int | None = None,
    poll_interval_sec: float = 30.0,
) -> dict:
    """Run a batch-mode leakage evaluation job (taxonomy → Shapley → metrics).

    Same output schema as `leakage_eval.pipeline.run_evaluation_job`:
        - manifest.json, config.json, instance_results.jsonl,
          claim_results.jsonl, shapley_results.jsonl, summary.json, errors.jsonl

    Plus state/ subdir for batch IDs (to support resume).
    """
    # 1. Load instances
    rows = load_completion_rows(config.input_path)
    if max_instances is not None:
        rows = rows[:max_instances]
    metadata = load_dataset_metadata(config.dataset_path)
    instances = build_eval_inputs(rows, metadata, model=model, method=method)
    log.info("Loaded %d instances for batch evaluation", len(instances))

    # 2. Set up output + state
    writer = ResultWriter(config.output_dir)
    state = JobState(config.output_dir)
    writer.write_config(config)
    writer.write_manifest(
        {
            "model": model,
            "method": method,
            "input_path": str(config.input_path),
            "dataset_path": str(config.dataset_path),
            "judge_model": config.leakage.judge_model,
            "shapley_evaluator_model": (
                config.shapley.evaluator_model_override or model
            ),
            "shapley_estimator_config": {
                "coalition_budget_c": config.shapley.coalition_budget_c,
                "coalition_budget_cap": config.shapley.coalition_budget_cap,
                "exact_threshold_n": config.shapley.exact_threshold_n,
                "random_seed": config.shapley.random_seed,
                "skip_if_no_leakage": config.shapley.skip_if_no_leakage,
            },
            "n_instances": len(instances),
            "mode": "anthropic_batch",
        }
    )

    client = AnthropicBatchClient()

    # 3. Phase 1: Taxonomy
    decisions_by_instance = _run_taxonomy_phase(
        instances, config, client, state, writer, poll_interval_sec=poll_interval_sec
    )

    # 4. Phase 2: Shapley
    _run_shapley_phase(
        instances, decisions_by_instance, config, client, state, writer,
        model=model, poll_interval_sec=poll_interval_sec,
    )

    # 5. Aggregate summary
    all_rows = _load_all_instance_rows(config.output_dir)
    summary = aggregate_instance_rows(all_rows)
    writer.write_summary(summary)
    return summary


# ============================================================
# Phase 1: Taxonomy
# ============================================================


def _run_taxonomy_phase(
    instances: list[EvalInput],
    config: EvalJobConfig,
    client: AnthropicBatchClient,
    state: JobState,
    writer: ResultWriter,
    *,
    poll_interval_sec: float,
) -> dict[str, list[ClaimDecision]]:
    """Submit taxonomy batch (if not already), poll, parse, write claim_results.

    Returns: instance_id → list[ClaimDecision]
    """
    if state.is_taxonomy_done():
        log.info("Taxonomy phase already done; loading cached claim_results.jsonl")
        return _load_claim_decisions_from_disk(config.output_dir)

    batch_size = config.leakage.taxonomy_batch_size
    judge_model = config.leakage.judge_model

    # Build taxonomy requests: each "request" = one batch of up to batch_size claims
    requests, request_records = _build_taxonomy_requests(instances, batch_size, judge_model)

    if not requests:
        log.info("No claims to verify; marking taxonomy done.")
        state.mark_taxonomy_done()
        return {}

    handle = state.load_taxonomy_batch()
    if handle is None:
        log.info("Submitting taxonomy batch (%d requests)", len(requests))
        if len(requests) > MAX_BATCH_REQUESTS:
            raise NotImplementedError(
                f"Taxonomy needs {len(requests)} requests, exceeds {MAX_BATCH_REQUESTS}. "
                "Multi-batch taxonomy split not yet implemented."
            )
        batch_id = client.submit(requests)
        state.save_taxonomy_batch(batch_id, len(requests))
        state.save_taxonomy_requests(request_records)
        handle = state.load_taxonomy_batch()
        assert handle is not None

    log.info("Polling taxonomy batch %s ...", handle.batch_id)
    client.poll(handle.batch_id, interval_sec=poll_interval_sec)

    log.info("Fetching taxonomy batch results ...")
    results_by_id = {r.custom_id: r for r in client.fetch_results(handle.batch_id)}

    # Re-load request records (so this is restart-safe even when called fresh)
    request_records = state.load_taxonomy_requests()

    # Reconstruct ClaimRecord objects for parsing
    records_by_instance: dict[str, list[ClaimRecord]] = {
        inst.instance_id: _claim_records_for_instance(inst) for inst in instances
    }
    instance_by_id = {inst.instance_id: inst for inst in instances}

    decisions_by_instance: dict[str, list[ClaimDecision]] = {
        inst.instance_id: [] for inst in instances
    }

    for record in request_records:
        custom_id = record["custom_id"]
        instance_id = record["instance_id"]
        claim_ids = record["claim_ids"]  # list[int]
        result = results_by_id.get(custom_id)
        records_in_chunk = [
            r for r in records_by_instance[instance_id] if r.claim_id in set(claim_ids)
        ]
        if result is None or result.error or not result.text:
            # Fallback: use local conservative decision for each claim in this chunk
            err = (result.error if result else "missing result") or "no text"
            log.warning("Taxonomy chunk %s failed (%s); using local fallback", custom_id, err)
            for rec in records_in_chunk:
                decision = local_claim_decision(rec)
                decisions_by_instance[instance_id].append(decision)
                writer.append_claim(asdict(decision))
            continue

        try:
            by_id = _parse_llm_decisions(result.text)
        except ValueError as exc:
            log.warning("Taxonomy chunk %s parse failed (%s); using local fallback", custom_id, exc)
            for rec in records_in_chunk:
                decision = local_claim_decision(rec)
                decisions_by_instance[instance_id].append(decision)
                writer.append_claim(asdict(decision))
            continue

        for rec in records_in_chunk:
            decision = _decision_from_llm(rec, by_id.get(rec.claim_id))
            decisions_by_instance[instance_id].append(decision)
            writer.append_claim(asdict(decision))

    state.mark_taxonomy_done()
    log.info("Taxonomy phase complete for %d instances.", len(instances))
    return decisions_by_instance


def _build_taxonomy_requests(
    instances: list[EvalInput],
    batch_size: int,
    judge_model: str,
) -> tuple[list[BatchRequest], list[dict]]:
    """Build BatchRequest objects and persistent request manifests for taxonomy.

    One BatchRequest per chunk of up to `batch_size` claims from one instance.
    custom_id format: 'tax:<instance_id>:<chunk_idx>'
    """
    requests: list[BatchRequest] = []
    records: list[dict] = []
    for inst in instances:
        claim_records = _claim_records_for_instance(inst)
        for chunk_idx, start in enumerate(range(0, len(claim_records), batch_size)):
            chunk = claim_records[start : start + batch_size]
            if not chunk:
                continue
            items = [
                {
                    "id": r.claim_id,
                    "claim": r.claim_text,
                    "declared_source_date": r.declared_source_date,
                }
                for r in chunk
            ]
            user_prompt = build_taxonomy_date_prompt(
                json.dumps(items, ensure_ascii=False, separators=(",", ":")),
                cutoff_date=chunk[0].cutoff_date,
                task=chunk[0].task,
                target_description=chunk[0].target_description,
            )
            custom_id = _sanitize_id(f"tax-{inst.instance_id}-{chunk_idx}")
            requests.append(
                BatchRequest(
                    custom_id=custom_id,
                    system=TAXONOMY_DATE_SYSTEM_PROMPT,
                    user=user_prompt,
                    model=judge_model,
                    max_tokens=8192,
                    temperature=0.0,
                )
            )
            records.append(
                {
                    "custom_id": custom_id,
                    "instance_id": inst.instance_id,
                    "claim_ids": [r.claim_id for r in chunk],
                }
            )
    return requests, records


def _claim_records_for_instance(instance: EvalInput) -> list[ClaimRecord]:
    return [
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


def _load_claim_decisions_from_disk(output_dir: Path) -> dict[str, list[ClaimDecision]]:
    out: dict[str, list[ClaimDecision]] = {}
    path = output_dir / "claim_results.jsonl"
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            decision = ClaimDecision(**row)
            out.setdefault(decision.instance_id, []).append(decision)
    return out


# ============================================================
# Phase 2: Shapley
# ============================================================


def _run_shapley_phase(
    instances: list[EvalInput],
    decisions_by_instance: dict[str, list[ClaimDecision]],
    config: EvalJobConfig,
    client: AnthropicBatchClient,
    state: JobState,
    writer: ResultWriter,
    *,
    model: str,
    poll_interval_sec: float,
) -> None:
    """Submit all coalition prompts as one or more batches, then solve Shapley locally."""
    if state.is_shapley_done():
        log.info("Shapley phase already done; skipping.")
        return

    shapley_model = config.shapley.evaluator_model_override or model

    # 1. For each instance, decide if we need Shapley + pre-compute its masks
    plans: list[dict] = []
    for inst in instances:
        decisions = decisions_by_instance.get(inst.instance_id, [])
        if not decisions:
            # No claims at all → write a skipped result and move on
            _write_skipped_instance(inst, decisions, config, writer, reason="no_claims")
            continue
        labels = decision_labels(decisions)
        leaked_claims = sum(labels)
        if config.shapley.skip_if_no_leakage and leaked_claims == 0:
            _write_skipped_instance(inst, decisions, config, writer, reason="no_leakage")
            continue
        plan = _plan_instance(inst, config, decisions, labels)
        plans.append(plan)

    if not plans:
        log.info("No instances need Shapley; phase 2 done.")
        state.mark_shapley_done()
        return

    log.info("Shapley phase: %d instances will be evaluated.", len(plans))

    # 2. Build the full list of coalition requests
    requests, request_records = _build_shapley_requests(plans, shapley_model)
    log.info("Built %d coalition requests across %d instances.", len(requests), len(plans))

    # 3. Submit (in chunks if needed)
    # Chunks are deterministic across restarts (same masks from same seed,
    # same chunk size). We persist the requests manifest and per-chunk handles
    # after every successful submit so a crash can never cause a duplicate
    # submission of an already-submitted chunk.
    handles = state.load_shapley_batches()
    n_total_chunks = (len(requests) + MAX_BATCH_REQUESTS - 1) // MAX_BATCH_REQUESTS
    if len(handles) < n_total_chunks:
        if not handles:
            # First time submitting for this job: persist the requests manifest
            # upfront so chunking is reproducible across restarts.
            state.save_shapley_requests(request_records)
        else:
            log.info(
                "Resuming Shapley submission: %d/%d chunks already submitted",
                len(handles), n_total_chunks,
            )
        for chunk_idx in range(len(handles), n_total_chunks):
            chunk_start = chunk_idx * MAX_BATCH_REQUESTS
            chunk = requests[chunk_start : chunk_start + MAX_BATCH_REQUESTS]
            log.info(
                "Submitting Shapley chunk %d/%d (%d requests) ...",
                chunk_idx + 1, n_total_chunks, len(chunk),
            )
            batch_id = client.submit(chunk)
            handles.append(BatchHandle(
                batch_id=batch_id,
                submitted_at=datetime.utcnow().isoformat() + "Z",
                request_count=len(chunk),
            ))
            # Persist immediately — atomic write means no partial state on crash.
            state.save_shapley_batches(handles)

    # 4. Poll each batch
    for h in handles:
        log.info("Polling Shapley batch %s ...", h.batch_id)
        client.poll(h.batch_id, interval_sec=poll_interval_sec)

    # 5. Fetch all results
    log.info("Fetching Shapley batch results ...")
    results_by_id: dict[str, BatchResult] = {}
    for h in handles:
        for r in client.fetch_results(h.batch_id):
            results_by_id[r.custom_id] = r

    # 6. Re-load request records to keep restart-safe
    request_records = state.load_shapley_requests()

    # 7. Group results by instance_id
    results_by_instance: dict[str, dict[tuple[int, ...], BatchResult]] = {}
    for rec in request_records:
        custom_id = rec["custom_id"]
        instance_id = rec["instance_id"]
        mask = tuple(rec["mask"])
        result = results_by_id.get(custom_id)
        if result is not None:
            results_by_instance.setdefault(instance_id, {})[mask] = result

    # 8. For each planned instance, solve Shapley and write results
    plan_by_id = {p["instance_id"]: p for p in plans}
    for plan in plans:
        try:
            output = _finalize_instance(plan, results_by_instance.get(plan["instance_id"], {}), config)
            instance_row = _instance_row(output)
            writer.append_instance(instance_row)
            writer.append_shapley(asdict(output.shapley_result))
            if instance_row["faithfulness_status"] == "low_faithfulness":
                writer.append_low_faithfulness(instance_row)
        except Exception as exc:
            log.warning("Instance %s failed during finalization: %s", plan["instance_id"], exc)
            writer.append_error({"instance_id": plan["instance_id"], "error": str(exc)})

    state.mark_shapley_done()
    log.info("Shapley phase complete.")


def _plan_instance(
    instance: EvalInput,
    config: EvalJobConfig,
    decisions: list[ClaimDecision],
    labels: list[int],
) -> dict:
    """Pre-compute the coalition masks for one instance."""
    n_claims = len(instance.evidence)
    budget = compute_budget(
        n_claims,
        c=config.shapley.coalition_budget_c,
        cap=config.shapley.coalition_budget_cap,
        exact_threshold_n=config.shapley.exact_threshold_n,
    )
    is_exact = budget >= 2 ** n_claims
    sampling_probs: np.ndarray | None = None
    if is_exact:
        masks = exact_coalitions(n_claims)
    elif config.shapley.use_paper_leverage_shap:
        masks, sampling_probs = paper_leverage_masks(
            n_claims, budget, seed=config.shapley.random_seed, pairing_trick=True
        )
    else:
        masks = leverage_shap_coalitions(n_claims, budget, seed=config.shapley.random_seed)

    return {
        "instance_id": instance.instance_id,
        "instance": instance,
        "decisions": decisions,
        "labels": labels,
        "masks": masks,
        "is_exact": is_exact,
        "sampling_probs": sampling_probs,
    }


def _build_shapley_requests(plans: list[dict], shapley_model: str) -> tuple[list[BatchRequest], list[dict]]:
    """Build coalition BatchRequests for every (instance, mask) pair where mask is nonzero."""
    requests: list[BatchRequest] = []
    records: list[dict] = []
    for plan in plans:
        instance: EvalInput = plan["instance"]
        max_tokens = _coalition_max_tokens(instance)
        for mask_idx, mask in enumerate(plan["masks"]):
            if sum(mask) == 0:
                continue  # empty coalition has value 0, no API call needed
            evidence_subset = [
                item for item, bit in zip(instance.evidence, mask) if bit
            ]
            system_msg, user_msg = build_coalition_messages(instance, evidence_subset)
            custom_id = _sanitize_id(f"shap-{instance.instance_id}-{mask_idx}")
            requests.append(
                BatchRequest(
                    custom_id=custom_id,
                    system=system_msg,
                    user=user_msg,
                    model=shapley_model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
            )
            records.append(
                {
                    "custom_id": custom_id,
                    "instance_id": instance.instance_id,
                    "mask": list(mask),
                }
            )
    return requests, records


def _finalize_instance(
    plan: dict,
    results_by_mask: dict[tuple[int, ...], BatchResult],
    config: EvalJobConfig,
) -> EvalOutput:
    """Solve Shapley regression locally from batch results."""
    instance: EvalInput = plan["instance"]
    decisions: list[ClaimDecision] = plan["decisions"]
    labels: list[int] = plan["labels"]
    masks: list[tuple[int, ...]] = plan["masks"]
    is_exact: bool = plan["is_exact"]
    sampling_probs = plan["sampling_probs"]

    full_mask = tuple([1] * len(instance.evidence))
    values: list[float] = []
    full_prediction: dict | None = None
    num_model_calls = 0

    for mask in masks:
        if sum(mask) == 0:
            values.append(0.0)
            continue
        result = results_by_mask.get(mask)
        if result is None or result.error or not result.text:
            # Use a neutral fallback value (0.5 reconstruction) so regression doesn't break.
            # Audit logging not implemented in batch mode for now.
            log.warning(
                "Coalition result missing for %s mask=%s (err=%s); using fallback 0.0",
                instance.instance_id, mask, (result.error if result else "missing"),
            )
            values.append(0.0)
            continue
        try:
            prediction = parse_prediction(instance.task, result.text)
            compact = compact_coalition_prediction(instance.task, prediction)
            value = reconstruction_value(instance.task, compact, instance.prediction)
        except Exception as exc:
            log.warning(
                "Failed to parse coalition for %s mask=%s: %s",
                instance.instance_id, mask, exc,
            )
            values.append(0.0)
            continue
        values.append(value)
        num_model_calls += 1
        if mask == full_mask:
            full_prediction = compact

    v_full_idx = masks.index(full_mask)
    v_full = values[v_full_idx]

    if is_exact:
        phi = exact_shapley(masks, values)
    elif config.shapley.use_paper_leverage_shap:
        assert sampling_probs is not None
        phi = solve_paper_leverage_regression(
            masks, values, sampling_probs, v0=0.0, v1=float(v_full)
        )
    else:
        phi = solve_projected_regression(masks, values, v_full)

    claim_values = _rank_claims(phi.tolist())
    estimator_name = (
        "exact"
        if is_exact
        else (
            "paper_leverage_shap"
            if config.shapley.use_paper_leverage_shap
            else "leverage_shap_projected_regression"
        )
    )

    shapley_result = ShapleyResult(
        instance_id=instance.instance_id,
        shapley_status="computed",
        estimator=estimator_name,
        n_claims=len(instance.evidence),
        m_total=len(masks),
        v_empty=0.0,
        v_full=float(v_full),
        original_prediction=instance.prediction,
        closed_world_full_prediction=full_prediction,
        full_reconstruction_gap=_prediction_gap(instance.task, instance.prediction, full_prediction),
        num_model_calls=num_model_calls,
        claim_shapley=claim_values,
    )

    shapley_value = shapley_dclr(
        [item.shapley_value for item in sorted(shapley_result.claim_shapley, key=lambda x: x.claim_id)],
        labels,
    )["shapley_dclr"]
    leaked_claims = sum(labels)

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


def _write_skipped_instance(
    instance: EvalInput,
    decisions: list[ClaimDecision],
    config: EvalJobConfig,
    writer: ResultWriter,
    *,
    reason: str,
) -> None:
    labels = decision_labels(decisions) if decisions else []
    leaked_claims = sum(labels) if labels else 0
    shapley_result = _skipped_shapley(
        instance,
        "No leaked claims; Shapley-DCLR is exactly zero." if reason == "no_leakage" else f"Skipped: {reason}",
    )
    output = EvalOutput(
        instance_id=instance.instance_id,
        model=instance.model,
        method=instance.method,
        task=instance.task,
        n_claims=len(instance.evidence),
        leaked_claims=leaked_claims,
        olr=leakage_rate(labels) if labels else 0.0,
        shapley_dclr=0.0,
        performance=performance_score(instance.task, instance.prediction, instance.ground_truth),
        claim_decisions=decisions,
        shapley_result=shapley_result,
    )
    row = _instance_row(output)
    writer.append_instance(row)
    writer.append_shapley(asdict(shapley_result))


def _rank_claims(values: list[float]) -> list[ClaimShapleyValue]:
    ranked_ids = sorted(range(len(values)), key=lambda idx: abs(values[idx]), reverse=True)
    ranks = {idx: rank + 1 for rank, idx in enumerate(ranked_ids)}
    return [
        ClaimShapleyValue(
            claim_id=idx + 1,
            shapley_value=float(v),
            abs_shapley_value=float(abs(v)),
            shapley_rank=ranks[idx],
        )
        for idx, v in enumerate(values)
    ]


def _prediction_gap(task: str, original: dict, reconstructed: dict | None) -> float | None:
    if reconstructed is None:
        return None
    return 1.0 - reconstruction_value(task, reconstructed, original)


_CUSTOM_ID_BAD = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_id(raw: str) -> str:
    """Coerce an arbitrary string into the Anthropic custom_id regex ``^[a-zA-Z0-9_-]{1,64}$``.

    Replaces unsupported chars with ``_``. If the result exceeds 64 chars,
    truncates and appends an 8-char hash of the original for uniqueness.
    """
    cleaned = _CUSTOM_ID_BAD.sub("_", raw)
    if len(cleaned) <= 64:
        return cleaned
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return cleaned[: 64 - 9] + "-" + digest


def _load_all_instance_rows(output_dir: Path) -> list[dict]:
    path = output_dir / "instance_results.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
