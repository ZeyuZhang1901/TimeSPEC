"""Exact Shapley computation from complete coalition values."""

from __future__ import annotations

import math

import numpy as np


def exact_shapley(masks: list[tuple[int, ...]], values: list[float]) -> np.ndarray:
    """Compute exact Shapley values from all coalition values."""

    if len(masks) != len(values):
        raise ValueError("masks and values must have the same length")
    if not masks:
        return np.array([], dtype=float)
    n_claims = len(masks[0])
    value_by_mask = {mask: float(value) for mask, value in zip(masks, values)}
    expected = 2 ** n_claims
    if len(value_by_mask) != expected:
        raise ValueError(f"Exact Shapley requires all {expected} coalitions")

    phi = np.zeros(n_claims, dtype=float)
    n_factorial = math.factorial(n_claims)
    for i in range(n_claims):
        for mask in value_by_mask:
            if mask[i] == 1:
                continue
            size = sum(mask)
            with_i = list(mask)
            with_i[i] = 1
            weight = math.factorial(size) * math.factorial(n_claims - size - 1) / n_factorial
            phi[i] += weight * (value_by_mask[tuple(with_i)] - value_by_mask[mask])
    return phi

