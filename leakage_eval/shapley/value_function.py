"""Closed-world value-function helpers."""

from __future__ import annotations

from typing import Any, Protocol

from LeakageEvaluationFramework.leakage_eval.metrics.performance import _stock_rho
from LeakageEvaluationFramework.leakage_eval.schemas import EvalInput, EvidenceItem


def compact_coalition_prediction(task: str, prediction: dict[str, Any]) -> dict[str, Any]:
    """Reduce coalition model output to fields required by :func:`reconstruction_value` only.

    Used for Shapley logging and ``closed_world_full_prediction`` so audit artifacts stay small.
    """

    if task == "binary":
        p = float(prediction["probability_yes"])
        return {"probability_yes": p}

    if task == "multiple_choice":
        probs = prediction.get("probabilities", {})
        if isinstance(probs, dict) and probs:
            out_probs = {str(k): float(v) for k, v in probs.items()}
            return {"probabilities": out_probs}
        pred = prediction.get("prediction")
        if pred is not None:
            return {"prediction": str(pred)}
        raise ValueError("multiple_choice coalition prediction missing probabilities or prediction")

    if task == "numeric":
        return {"prediction": float(prediction["prediction"])}

    if task == "legal":
        return {"probability_petitioner": float(prediction["probability_petitioner"])}

    if task == "salary":
        return {"predicted_salary": float(prediction["predicted_salary"])}

    if task == "stock":
        ranking = prediction["ranking"]
        if not isinstance(ranking, list):
            raise ValueError("stock coalition prediction ranking must be a list")
        return {"ranking": [str(x).strip() for x in ranking if str(x).strip()]}

    raise ValueError(f"Unsupported task: {task}")


class CoalitionPredictor(Protocol):
    """Protocol for model-backed coalition prediction."""

    async def predict(self, instance: EvalInput, evidence_subset: list[EvidenceItem]) -> dict:
        """Predict from a closed-world evidence subset."""


async def evaluate_coalition(
    instance: EvalInput,
    mask: tuple[int, ...],
    predictor: CoalitionPredictor,
    baseline_stats: dict | None = None,
) -> tuple[float, dict | None]:
    """Evaluate normalized value for one coalition mask.

    Returns ``(value, compact_prediction)`` where ``compact_prediction`` contains
    only keys required for :func:`reconstruction_value` (suitable for audits).
    """

    if len(mask) != len(instance.evidence):
        raise ValueError("Coalition mask length must match evidence length")
    if sum(mask) == 0:
        # v(emptyset) = 0: with no evidence the model has zero reconstruction signal.
        return 0.0, None

    evidence_subset = [item for item, bit in zip(instance.evidence, mask) if bit]
    prediction = await predictor.predict(instance, evidence_subset)
    compact = compact_coalition_prediction(instance.task, prediction)
    value = reconstruction_value(instance.task, compact, instance.prediction)
    return value, compact


def reconstruction_value(task: str, prediction: dict, original_prediction: dict) -> float:
    """Score how well a coalition reconstructs the original prediction.

    Returns a value in [0, 1]; higher = more faithful reconstruction.
    """

    if task == "binary":
        p_hat  = float(prediction.get("probability_yes", 0.5))
        p_orig = float(original_prediction.get("probability_yes", 0.5))
        return float(max(0.0, 1.0 - abs(p_hat - p_orig)))

    if task == "multiple_choice":
        probs_hat  = prediction.get("probabilities", {})
        probs_orig = original_prediction.get("probabilities", {})
        keys = set(probs_orig) | set(probs_hat)
        if not keys:
            return 1.0 if prediction.get("prediction") == original_prediction.get("prediction") else 0.0
        total_diff = sum(abs(probs_hat.get(k, 0.0) - probs_orig.get(k, 0.0)) for k in keys)
        return float(max(0.0, 1.0 - total_diff / len(keys)))

    if task == "numeric":
        pred = float(prediction.get("prediction", 0.0))
        orig = float(original_prediction.get("prediction", 0.0))
        if orig == 0.0:
            return 1.0 if pred == 0.0 else 0.0
        return float(max(0.0, 1.0 - abs(pred - orig) / abs(orig)))

    if task == "legal":
        p_hat  = float(prediction["probability_petitioner"])
        p_orig = float(original_prediction["probability_petitioner"])
        return float(max(0.0, 1.0 - abs(p_hat - p_orig)))

    if task == "salary":
        pred = float(prediction["predicted_salary"])
        orig = float(original_prediction["predicted_salary"])
        if orig <= 0:
            # Model refused/failed to predict a positive value. Treat the
            # closed-world reconstruction as faithful iff the coalition also
            # produced a non-positive value (mirrors the numeric==0 branch).
            return 1.0 if pred <= 0 else 0.0
        return float(max(0.0, 1.0 - abs(pred - orig) / orig))

    if task == "stock":
        return float((_stock_rho(prediction, {"ranking": original_prediction["ranking"]}) + 1.0) / 2.0)

    raise ValueError(f"Unsupported task: {task}")

