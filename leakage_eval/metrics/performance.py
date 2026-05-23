"""Task performance and normalized Shapley value functions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr


def performance_score(task: str, prediction: dict[str, Any], ground_truth: dict[str, Any]) -> float:
    """Compute task performance in [0, 1]."""

    if task == "binary":
        return _binary_score(prediction, ground_truth)
    if task == "multiple_choice":
        return _mc_score(prediction, ground_truth)
    if task == "numeric":
        return _numeric_score(prediction, ground_truth)
    if task == "legal":
        return _legal_score(prediction, ground_truth)
    if task == "salary":
        return _salary_score(prediction, ground_truth)
    if task == "stock":
        return (_stock_rho(prediction, ground_truth) + 1.0) / 2.0
    raise ValueError(f"Unsupported task: {task}")


def normalized_value(task: str, prediction: dict[str, Any], ground_truth: dict[str, Any], baseline_stats: dict[str, Any] | None = None) -> float:
    """Compute normalized above-baseline Shapley value v(S)."""

    baseline_stats = baseline_stats or {}
    if task == "binary":
        return _binary_score(prediction, ground_truth)
    if task == "multiple_choice":
        return _mc_score(prediction, ground_truth)
    if task == "numeric":
        return _numeric_score(prediction, ground_truth)
    if task == "legal":
        return (_legal_score(prediction, ground_truth) - 0.75) / 0.25
    if task == "stock":
        return _stock_rho(prediction, ground_truth)
    if task == "salary":
        score = _salary_score(prediction, ground_truth)
        actual = _actual_salary(ground_truth)
        if "salary_baseline" not in baseline_stats:
            return score
        baseline_salary = float(baseline_stats["salary_baseline"])
        baseline_score = max(0.0, 1.0 - abs(baseline_salary - actual) / actual)
        return (score - baseline_score) / max(1.0 - baseline_score, 1e-8)
    raise ValueError(f"Unsupported task: {task}")


def _binary_score(prediction: dict[str, Any], ground_truth: dict[str, Any]) -> float:
    """1 − Brier score for a binary probability forecast."""
    p_yes = float(prediction.get("probability_yes", prediction.get("p", 0.5)))
    answer = str(ground_truth.get("answer", "")).strip().lower()
    y = 1.0 if answer in {"yes", "true", "1"} else 0.0
    return float(1.0 - (p_yes - y) ** 2)


def _mc_score(prediction: dict[str, Any], ground_truth: dict[str, Any]) -> float:
    """1 − normalised multiclass Brier score."""
    probs = prediction.get("probabilities", {})
    correct = str(ground_truth.get("answer", "")).strip()
    if not probs or correct not in probs:
        # Fall back: 1 if prediction matches, 0 otherwise
        pred_label = str(prediction.get("prediction", "")).strip()
        return 1.0 if pred_label == correct else 0.0
    n = len(probs)
    brier = sum((v - (1.0 if k == correct else 0.0)) ** 2 for k, v in probs.items())
    return float(max(0.0, 1.0 - brier / n))


def _numeric_score(prediction: dict[str, Any], ground_truth: dict[str, Any]) -> float:
    """Relative accuracy: max(0, 1 − |pred − truth| / |truth|)."""
    pred = float(prediction.get("prediction", 0.0))
    truth_raw = ground_truth.get("answer", 0.0)
    try:
        truth = float(truth_raw)
    except (TypeError, ValueError):
        return 0.0
    if truth == 0.0:
        return 1.0 if pred == 0.0 else 0.0
    return float(max(0.0, 1.0 - abs(pred - truth) / abs(truth)))


def _legal_score(prediction: dict[str, Any], ground_truth: dict[str, Any]) -> float:
    p_hat = float(prediction["probability_petitioner"])
    y = _legal_label(ground_truth)
    return float(1.0 - (p_hat - y) ** 2)


def _legal_label(ground_truth: dict[str, Any]) -> float:
    winner = str(ground_truth.get("winner") or ground_truth.get("outcome") or "").upper()
    if winner in {"PETITIONER", "PETITIONER_WINS", "1", "TRUE"}:
        return 1.0
    if winner in {"RESPONDENT", "RESPONDENT_WINS", "0", "FALSE"}:
        return 0.0
    if "petitioner_wins" in ground_truth:
        return 1.0 if bool(ground_truth["petitioner_wins"]) else 0.0
    raise ValueError(f"Cannot infer legal ground-truth label from {ground_truth}")


def _salary_score(prediction: dict[str, Any], ground_truth: dict[str, Any]) -> float:
    pred = float(prediction["predicted_salary"])
    actual = _actual_salary(ground_truth)
    return float(max(0.0, 1.0 - abs(pred - actual) / actual))


def _actual_salary(ground_truth: dict[str, Any]) -> float:
    actual = float(ground_truth.get("aav") or ground_truth.get("actual_salary") or 0.0)
    if actual <= 0:
        raise ValueError(f"Salary ground truth must contain positive aav: {ground_truth}")
    return actual


def _stock_rho(prediction: dict[str, Any], ground_truth: dict[str, Any]) -> float:
    pred = [str(x) for x in prediction["ranking"]]
    truth = [str(x) for x in ground_truth["ranking"]]
    common = [ticker for ticker in truth if ticker in pred]
    if len(common) < 2:
        return 0.0
    pred_rank = {ticker: rank for rank, ticker in enumerate(pred)}
    truth_rank = {ticker: rank for rank, ticker in enumerate(truth)}
    pred_values = np.array([pred_rank[ticker] for ticker in common], dtype=float)
    truth_values = np.array([truth_rank[ticker] for ticker in common], dtype=float)
    rho = spearmanr(pred_values, truth_values).correlation
    if rho is None or np.isnan(rho):
        return 0.0
    return float(rho)

