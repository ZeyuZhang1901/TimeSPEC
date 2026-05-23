"""Shared schemas for leakage evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


TaskName = Literal["legal", "salary", "stock", "binary", "multiple_choice", "numeric"]
ClaimCategory = Literal["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
Confidence = Literal["high", "medium", "low"]
ShapleyStatus = Literal["computed", "skipped_no_leakage", "failed"]


@dataclass(frozen=True)
class EvidenceItem:
    """One generated evidence item."""

    id: int
    fact: str
    source_date: str | None = None


@dataclass(frozen=True)
class EvalInput:
    """One parsed method output aligned with dataset metadata."""

    instance_id: str
    task: TaskName
    question: str
    cutoff_date: str
    ground_truth: dict[str, Any]
    evidence: list[EvidenceItem]
    prediction: dict[str, Any]
    reasoning: str
    model: str
    method: str
    target_description: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClaimRecord:
    """Evidence claim record used by taxonomy and leakage detection."""

    instance_id: str
    claim_id: int
    claim_text: str
    declared_source_date: str | None
    cutoff_date: str
    task: TaskName
    target_description: str
    model: str
    method: str


@dataclass(frozen=True)
class ClaimDecision:
    """Taxonomy and leakage decision for one evidence claim."""

    instance_id: str
    claim_id: int
    claim_text: str
    category: ClaimCategory
    declared_source_date: str | None
    effective_source_date: str | None
    is_leaked: bool
    verification_method: str
    date_precision: str
    confidence: Confidence
    reason: str
    declared_date_plausible: bool | None = None
    date_corrected: bool = False
    postcheck_flags: list[str] = field(default_factory=list)
    needs_manual_review: bool = False


@dataclass(frozen=True)
class ClaimShapleyValue:
    """Shapley attribution for one evidence claim."""

    claim_id: int
    shapley_value: float
    abs_shapley_value: float
    shapley_rank: int
    bootstrap_se: float | None = None


@dataclass(frozen=True)
class ShapleyResult:
    """Shapley computation result for one instance."""

    instance_id: str
    shapley_status: ShapleyStatus
    estimator: str | None
    n_claims: int
    m_total: int
    v_empty: float
    v_full: float | None
    original_prediction: dict[str, Any]
    closed_world_full_prediction: dict[str, Any] | None
    full_reconstruction_gap: float | None
    num_model_calls: int
    claim_shapley: list[ClaimShapleyValue]
    reason: str | None = None


@dataclass(frozen=True)
class EvalOutput:
    """Final per-instance leakage evaluation output."""

    instance_id: str
    model: str
    method: str
    task: TaskName
    n_claims: int
    leaked_claims: int
    olr: float
    shapley_dclr: float
    performance: float
    claim_decisions: list[ClaimDecision]
    shapley_result: ShapleyResult

