"""Aggregate per-instance evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def aggregate_instance_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate instance result rows."""

    if not rows:
        return {"n_instances": 0}
    reconstruction_gaps = [
        float(row["full_reconstruction_gap"])
        for row in rows
        if row.get("full_reconstruction_gap") is not None
    ]
    faithfulness_counts = {
        status: sum(1 for row in rows if row.get("faithfulness_status") == status)
        for status in ("high_faithfulness", "medium_faithfulness", "low_faithfulness", "not_applicable")
    }
    return {
        "n_instances": len(rows),
        "olr": _mean_std([float(row["olr"]) for row in rows]),
        "shapley_dclr": _mean_std([float(row["shapley_dclr"]) for row in rows]),
        "performance": _mean_std([float(row["performance"]) for row in rows]),
        "full_reconstruction_gap": _mean_std(reconstruction_gaps),
        "faithfulness_counts": faithfulness_counts,
        "faithfulness_rate_high_or_medium": (
            (faithfulness_counts["high_faithfulness"] + faithfulness_counts["medium_faithfulness"]) / len(rows)
        ),
        "shapley_computed": sum(1 for row in rows if row.get("shapley_status") == "computed"),
        "shapley_skipped_no_leakage": sum(
            1 for row in rows if row.get("shapley_status") == "skipped_no_leakage"
        ),
        "total_coalitions_evaluated": sum(int(row.get("m_total", 0)) for row in rows),
        "total_model_calls": sum(int(row.get("num_model_calls", 0)) for row in rows),
        "total_claims": sum(int(row.get("n_claims", 0)) for row in rows),
        "total_leaked_claims": sum(int(row.get("leaked_claims", 0)) for row in rows),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }

