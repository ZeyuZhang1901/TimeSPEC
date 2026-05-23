"""Evaluator sensitivity audit data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluatorSensitivityRow:
    """Comparison between matching and candidate Shapley evaluators."""

    instance_id: str
    default_evaluator: str
    candidate_evaluator: str
    dclr_difference: float
    abs_phi_rank_correlation: float
    full_prediction_gap: float | None

