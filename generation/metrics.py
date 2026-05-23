"""Performance scoring for generation completions by answer type."""

from __future__ import annotations


def performance_score(answer_type: str, completion_payload: dict, ground_truth: str) -> float:
    """Compute a normalized performance score in [0, 1].

    Args:
        answer_type: "binary", "multiple_choice", or "numeric".
        completion_payload: Parsed completion dict from parsers.parse_completion.
        ground_truth: Ground-truth answer string from the dataset.

    Returns:
        Score in [0.0, 1.0]. Returns 0.0 on missing or malformed fields.
    """

    if answer_type == "binary":
        return _binary_score(completion_payload, ground_truth)
    if answer_type == "multiple_choice":
        return _mc_score(completion_payload, ground_truth)
    if answer_type == "numeric":
        return _numeric_score(completion_payload, ground_truth)
    raise ValueError(f"Unknown answer_type: {answer_type!r}")


def _binary_score(payload: dict, ground_truth: str) -> float:
    """Brier-based score: 1 - (prob_yes - y)^2."""

    try:
        prob_yes = float(payload["probability_yes"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    y = 1.0 if ground_truth.strip().lower() == "yes" else 0.0
    return float(max(0.0, 1.0 - (prob_yes - y) ** 2))


def _mc_score(payload: dict, ground_truth: str) -> float:
    """Normalized multiclass Brier score: 1 - 0.5 * sum_c (p_c - 1[c==answer])^2."""

    probs = payload.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return 0.0
    target = ground_truth.strip()
    total = sum(probs.values())
    if total <= 0:
        return 0.0
    brier_sum = 0.0
    for choice, prob in probs.items():
        y = 1.0 if choice == target else 0.0
        brier_sum += (float(prob) / total - y) ** 2
    return float(max(0.0, 1.0 - 0.5 * brier_sum))


def _numeric_score(payload: dict, ground_truth: str) -> float:
    """Relative error score: max(0, 1 - |pred - gt| / max(|gt|, 1.0))."""

    try:
        prediction = float(payload["prediction"])
        gt = float(ground_truth.strip())
    except (KeyError, TypeError, ValueError):
        return 0.0
    denom = max(abs(gt), 1.0)
    return float(max(0.0, 1.0 - abs(prediction - gt) / denom))
