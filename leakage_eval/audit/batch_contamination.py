"""Batch-contamination audit data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchContaminationRow:
    """Comparison between single-coalition and batched coalition prediction."""

    instance_id: str
    coalition_key: str
    value_single: float
    value_batch: float
    value_difference: float

