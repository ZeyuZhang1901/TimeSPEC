"""Helpers for claim-level leakage labels."""

from __future__ import annotations

from datetime import date

from LeakageEvaluationFramework.leakage_eval.schemas import ClaimDecision


def decision_labels(decisions: list[ClaimDecision]) -> list[bool]:
    """Return leakage labels aligned with claim decisions."""

    return [decision.is_leaked for decision in sorted(decisions, key=lambda d: d.claim_id)]


def parse_date(value: str | None) -> date | None:
    """Parse an ISO date string."""

    if not value:
        return None
    return date.fromisoformat(value)

