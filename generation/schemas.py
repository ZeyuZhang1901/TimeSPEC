"""Shared dataclasses for the generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


_AnswerType = Literal[
    "binary", "multiple_choice", "numeric",
    "legal", "salary", "stock",
]


@dataclass(frozen=True)
class ForecastingTaskInstance:
    """One benchmark row adapted for generation.

    The ``answer`` field holds ground truth for post-hoc scoring only and
    must never be included in any prompt passed to the model.
    """

    instance_id: str
    question: str
    background: str
    temporal_cutoff: str          # YYYY-MM-DD
    resolution_date: str          # YYYY-MM-DD
    group: Literal["treatment", "control"]
    domain: str
    answer_type: _AnswerType
    choices: list[str] | None     # non-None for multiple_choice and stock (ticker list)
    answer: str                   # ground truth — NEVER passed to the model
    unit: str | None              # non-None when answer_type == "numeric"
    source_url: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Completion:
    """Structured output produced by one generation method for one instance."""

    instance_id: str
    method: Literal["b1", "b2", "b3", "b4"]
    model: str
    answer_type: _AnswerType
    evidence: list[dict[str, Any]]   # [{id, fact, source_date}]
    prediction: str | float          # "yes"/"no"; choice string; float for numeric; "" for legacy tasks
    reasoning: str
    probability: float | None        # probability_yes for binary
    probabilities: dict[str, float] | None  # per-choice probabilities for MC
    raw_text: str
    parse_success: bool
    regen_cycles: int                # 0 for B1/B3; cycles used by B2/B4 supervisor
    leaked_claim_ids: list[int]      # claim IDs flagged before final regen; [] for B1/B3
    timestamp: str
    # Legacy-task-specific predictions. Exactly one is non-None for legal/salary/stock;
    # all three are None for binary/multiple_choice/numeric.
    ranking: list[str] | None = None                # answer_type == "stock"
    probability_petitioner: float | None = None     # answer_type == "legal"
    predicted_salary: float | None = None           # answer_type == "salary"


@dataclass(frozen=True)
class GenerationConfig:
    """Runtime configuration for one (model, dataset) generation run."""

    model_name: str                 # OpenRouter model ID (e.g. "anthropic/claude-sonnet-4")
    temperature: float
    max_tokens: int
    judge_model: str                # OpenRouter model for C0-C8 verifier
    judge_max_tokens: int
    batch_size: int
    max_regen_cycles: int           # max supervisor–regen iterations for B2/B4
    perplexity_api_key: str
    openrouter_api_key: str
    rag_context_budget: int         # max chars of RAG context to inject
    rag_max_results: int
    # Anti-repetition sampling; zero by default (greedy). Used during retry of
    # instances where the model entered degenerate token loops.
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0            # max Perplexity results per query


def instance_from_row(row: dict[str, Any]) -> ForecastingTaskInstance:
    """Build a ForecastingTaskInstance from a validated BenchmarkRow dict."""

    return ForecastingTaskInstance(
        instance_id=row["id"],
        question=row["question"],
        background=row["background"],
        temporal_cutoff=row["temporal_cutoff"],
        resolution_date=row["resolution_date"],
        group=row["group"],
        domain=row["domain"],
        answer_type=row["answer_type"],
        choices=row.get("choices"),
        answer=row.get("answer", ""),
        unit=row.get("unit"),
        source_url=row.get("source_url", ""),
        metadata=row,
    )
