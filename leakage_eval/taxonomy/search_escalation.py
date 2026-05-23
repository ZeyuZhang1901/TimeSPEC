"""Search escalation selection for leakage detection."""

from __future__ import annotations

from datetime import date, timedelta

from LeakageEvaluationFramework.leakage_eval.schemas import ClaimDecision


HIGH_RISK_CATEGORIES = {"C3", "C4", "C6", "C7", "C8"}


def should_escalate(decision: ClaimDecision, cutoff_date: str, *, near_cutoff_days: int = 14) -> bool:
    """Return whether a claim decision should be escalated to search."""

    if decision.verification_method in {"missing_date_conservative"}:
        return True
    if decision.confidence == "low":
        return True
    if decision.category in HIGH_RISK_CATEGORIES and _is_near_cutoff(decision.effective_source_date, cutoff_date, near_cutoff_days):
        return True
    return False


def _is_near_cutoff(effective_date: str | None, cutoff_date: str, days: int) -> bool:
    if effective_date is None:
        return False
    try:
        eff = date.fromisoformat(effective_date)
        cutoff = date.fromisoformat(cutoff_date)
    except ValueError:
        return False
    return abs(eff - cutoff) <= timedelta(days=days)

