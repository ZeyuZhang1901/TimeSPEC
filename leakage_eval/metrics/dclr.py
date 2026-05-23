"""OLR and Shapley-DCLR metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def leakage_rate(leakage_labels: list[bool]) -> float:
    """Compute unweighted overall leakage rate."""

    if not leakage_labels:
        return 0.0
    return float(np.mean(np.array(leakage_labels, dtype=bool)))


def shapley_dclr(shapley_values: list[float], leakage_labels: list[bool]) -> dict[str, Any]:
    """Compute Shapley-weighted decision-critical leakage rate."""

    if len(shapley_values) != len(leakage_labels):
        raise ValueError(
            f"Length mismatch: {len(shapley_values)} Shapley values for "
            f"{len(leakage_labels)} leakage labels"
        )
    if not shapley_values:
        return {
            "shapley_dclr": 0.0,
            "total_shapley_weight": 0.0,
            "leaked_shapley_weight": 0.0,
        }

    phi = np.array(shapley_values, dtype=float)
    leaked = np.array(leakage_labels, dtype=bool)
    abs_phi = np.abs(phi)
    total_weight = float(abs_phi.sum())
    leaked_weight = float(abs_phi[leaked].sum())
    dclr = leaked_weight / total_weight if total_weight > 0 else 0.0
    return {
        "shapley_dclr": float(dclr),
        "total_shapley_weight": total_weight,
        "leaked_shapley_weight": leaked_weight,
    }

