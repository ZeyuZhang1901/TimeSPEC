"""Convergence audit data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvergenceRow:
    """One Shapley convergence audit row."""

    instance_id: str
    n_claims: int
    budget: int
    normalized_budget: float
    dclr_error: float
    mae: float
    rank_correlation: float

