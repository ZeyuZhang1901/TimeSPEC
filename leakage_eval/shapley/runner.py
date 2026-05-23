"""Shapley runner for one evaluation instance."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from LeakageEvaluationFramework.leakage_eval.audit.logger import AuditLogger, stable_text_hash
from LeakageEvaluationFramework.leakage_eval.config import ShapleyConfig
from LeakageEvaluationFramework.leakage_eval.schemas import ClaimShapleyValue, EvalInput, ShapleyResult
from LeakageEvaluationFramework.leakage_eval.shapley.prompts import build_coalition_messages

from .coalition_sampler import compute_budget, exact_coalitions, leverage_shap_coalitions
from .exact import exact_shapley
from .leverage_regression import solve_projected_regression
from .paper_leverage_shap import paper_leverage_masks, solve_paper_leverage_regression
from .value_function import CoalitionPredictor, evaluate_coalition, reconstruction_value

logger = logging.getLogger(__name__)


class ShapleyRunner:
    """Compute exact or Leverage SHAP values for one instance."""

    def __init__(
        self,
        config: ShapleyConfig,
        predictor: CoalitionPredictor,
        baseline_stats: dict | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.config = config
        self.predictor = predictor
        self.baseline_stats = baseline_stats or {}
        self.audit_logger = audit_logger

    async def run(self, instance: EvalInput) -> ShapleyResult:
        """Run Shapley computation for one instance.

        Coalition evaluations are parallelised via asyncio.gather with a
        concurrency semaphore (config.coalition_concurrency) to avoid
        overwhelming the API while keeping wall-clock time manageable.
        """

        n_claims = len(instance.evidence)
        if n_claims == 0:
            return _empty_result(instance.instance_id, instance.prediction)

        budget = compute_budget(
            n_claims,
            c=self.config.coalition_budget_c,
            cap=self.config.coalition_budget_cap,
            exact_threshold_n=self.config.exact_threshold_n,
        )
        is_exact = budget >= 2 ** n_claims
        sampling_probs: np.ndarray | None = None
        if is_exact:
            masks = exact_coalitions(n_claims)
        elif self.config.use_paper_leverage_shap:
            masks, sampling_probs = paper_leverage_masks(
                n_claims, budget, seed=self.config.random_seed, pairing_trick=True
            )
        else:
            masks = leverage_shap_coalitions(n_claims, budget, seed=self.config.random_seed)

        full_mask = tuple([1] * n_claims)
        concurrency = max(1, self.config.coalition_concurrency)
        semaphore = asyncio.Semaphore(concurrency)

        async def _eval_one(mask: tuple[int, ...]) -> tuple[tuple[int, ...], float, dict | None]:
            async with semaphore:
                try:
                    value, prediction = await evaluate_coalition(
                        instance, mask, self.predictor, self.baseline_stats
                    )
                except Exception as exc:
                    self._log_coalition_failure(instance, mask, exc)
                    raise
                self._log_coalition(instance, mask, value, prediction)
                return mask, value, prediction

        results = await asyncio.gather(*[_eval_one(m) for m in masks])

        values: list[float] = []
        full_prediction: dict | None = None
        num_model_calls = 0
        for mask, value, prediction in results:
            values.append(value)
            if sum(mask) > 0:
                num_model_calls += 1
            if mask == full_mask:
                full_prediction = prediction

        v_full = values[masks.index(full_mask)]
        if is_exact:
            phi = exact_shapley(masks, values)
        elif self.config.use_paper_leverage_shap:
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
                if self.config.use_paper_leverage_shap
                else "leverage_shap_projected_regression"
            )
        )

        logger.info(
            "Shapley done for %s: %d coalitions, %d model calls, v_full=%.4f",
            instance.instance_id, len(masks), num_model_calls, v_full,
        )

        return ShapleyResult(
            instance_id=instance.instance_id,
            shapley_status="computed",
            estimator=estimator_name,
            n_claims=n_claims,
            m_total=len(masks),
            v_empty=0.0,
            v_full=float(v_full),
            original_prediction=instance.prediction,
            closed_world_full_prediction=full_prediction,
            full_reconstruction_gap=_prediction_gap(instance.task, instance.prediction, full_prediction),
            num_model_calls=num_model_calls,
            claim_shapley=claim_values,
        )

    def _log_coalition_failure(self, instance: EvalInput, mask: tuple[int, ...], exc: Exception) -> None:
        if self.audit_logger is None:
            return
        evidence_subset = [item for item, bit in zip(instance.evidence, mask) if bit]
        prompt_text = ""
        if evidence_subset:
            sys, usr = build_coalition_messages(instance, evidence_subset)
            prompt_text = f"{sys}\n\n{usr}"
        self.audit_logger.log_coalition(
            {
                "instance_id": instance.instance_id,
                "task": instance.task,
                "mask": list(mask),
                "evidence_ids": [item.id for item in evidence_subset],
                "coalition_size": len(evidence_subset),
                "is_empty": len(evidence_subset) == 0,
                "is_full": len(evidence_subset) == len(instance.evidence),
                "prompt_hash": stable_text_hash(prompt_text) if prompt_text else None,
                "prediction": None,
                "value": None,
                "raw_response": getattr(self.predictor, "last_raw_text", None) if evidence_subset else None,
                "error": str(exc),
                "estimator_budget_c": self.config.coalition_budget_c,
                "estimator_budget_cap": self.config.coalition_budget_cap,
                "random_seed": self.config.random_seed,
            }
        )

    def _log_coalition(
        self,
        instance: EvalInput,
        mask: tuple[int, ...],
        value: float,
        prediction: dict | None,
    ) -> None:
        if self.audit_logger is None:
            return
        evidence_subset = [item for item, bit in zip(instance.evidence, mask) if bit]
        prompt_text = ""
        if evidence_subset:
            sys, usr = build_coalition_messages(instance, evidence_subset)
            prompt_text = f"{sys}\n\n{usr}"
        self.audit_logger.log_coalition(
            {
                "instance_id": instance.instance_id,
                "task": instance.task,
                "mask": list(mask),
                "evidence_ids": [item.id for item in evidence_subset],
                "coalition_size": len(evidence_subset),
                "is_empty": len(evidence_subset) == 0,
                "is_full": len(evidence_subset) == len(instance.evidence),
                "prompt_hash": stable_text_hash(prompt_text) if prompt_text else None,
                "prediction": prediction,
                "value": value,
                "raw_response": getattr(self.predictor, "last_raw_text", None) if evidence_subset else None,
                "estimator_budget_c": self.config.coalition_budget_c,
                "estimator_budget_cap": self.config.coalition_budget_cap,
                "random_seed": self.config.random_seed,
            }
        )


def _rank_claims(values: list[float]) -> list[ClaimShapleyValue]:
    ranked_ids = sorted(range(len(values)), key=lambda idx: abs(values[idx]), reverse=True)
    ranks = {idx: rank + 1 for rank, idx in enumerate(ranked_ids)}
    return [
        ClaimShapleyValue(
            claim_id=idx + 1,
            shapley_value=float(value),
            abs_shapley_value=float(abs(value)),
            shapley_rank=ranks[idx],
        )
        for idx, value in enumerate(values)
    ]


def _empty_result(instance_id: str, prediction: dict) -> ShapleyResult:
    return ShapleyResult(
        instance_id=instance_id,
        shapley_status="computed",
        estimator="exact",
        n_claims=0,
        m_total=1,
        v_empty=0.0,
        v_full=0.0,
        original_prediction=prediction,
        closed_world_full_prediction=None,
        full_reconstruction_gap=None,
        num_model_calls=0,
        claim_shapley=[],
    )


def _prediction_gap(task: str, original: dict, reconstructed: dict | None) -> float | None:
    if reconstructed is None:
        return None
    return 1.0 - reconstruction_value(task, reconstructed, original)

