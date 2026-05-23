"""Projected regression solver for Leverage SHAP."""

from __future__ import annotations

import math

import numpy as np


def solve_projected_regression(masks: list[tuple[int, ...]], values: list[float], v_full: float) -> np.ndarray:
    """Solve projected Leverage SHAP regression.

    Args:
        masks: Coalition masks, including or excluding boundary masks.
        values: Normalized values aligned with masks.
        v_full: Normalized full-coalition value.

    Returns:
        Shapley value vector satisfying efficiency up to numerical precision.
    """

    if len(masks) != len(values):
        raise ValueError("masks and values must have the same length")
    if not masks:
        return np.array([], dtype=float)
    n_claims = len(masks[0])
    if n_claims == 0:
        return np.array([], dtype=float)

    rows: list[tuple[int, ...]] = []
    targets: list[float] = []
    weights: list[float] = []
    for mask, value in zip(masks, values):
        _validate_mask(mask, n_claims)
        size = sum(mask)
        if size == 0 or size == n_claims:
            continue
        rows.append(mask)
        targets.append(float(value))
        weights.append(_row_weight(n_claims, size))

    if not rows:
        return np.full(n_claims, v_full / n_claims, dtype=float)

    z = np.array(rows, dtype=float)
    y = np.array(targets, dtype=float)
    w = np.array(weights, dtype=float)

    one = np.ones(n_claims, dtype=float)
    projection = np.eye(n_claims) - np.outer(one, one) / n_claims
    base = (float(v_full) / n_claims) * one
    centered_target = y - z @ base

    sqrt_w = np.sqrt(w)
    lhs = (z @ projection) * sqrt_w[:, None]
    rhs = centered_target * sqrt_w

    theta, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    phi = projection @ theta + base
    return phi


def _row_weight(n_claims: int, size: int) -> float:
    shap_weight = 1.0 / (math.comb(n_claims, size) * size * (n_claims - size))
    sampling_prob = 1.0 / ((n_claims - 1) * math.comb(n_claims, size))
    return shap_weight / sampling_prob


def _validate_mask(mask: tuple[int, ...], n_claims: int) -> None:
    if len(mask) != n_claims:
        raise ValueError("All coalition masks must have the same length")
    if any(bit not in (0, 1) for bit in mask):
        raise ValueError(f"Coalition mask must be binary: {mask}")

